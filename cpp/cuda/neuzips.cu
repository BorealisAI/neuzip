// #include <DietGpu.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <nvcomp.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/zstd.hpp>
#include <vector>

#define THREADS 512

#define CUDA_CHECK(cond)                                             \
  do {                                                               \
    cudaError_t err = cond;                                          \
    if (err != cudaSuccess) {                                        \
      std::cerr << "Failure\n";                                      \
      std::cerr << cudaGetErrorString(err) << " " << __FILE__ << ":" \
                << __LINE__ << std::endl;                            \
      exit(1);                                                       \
    }                                                                \
  } while (false)

template <typename scalar_t,
          typename frac_t,
          typename value_t,
          int frac_len,
          int exp_len,
          int precision>
__global__ void split_kernel(scalar_t* data,
                             uint8_t* exponents,
                             uint32_t* fractions,
                             size_t size) {
  __shared__ uint32_t fshared[THREADS / 32 * (precision + 1)];
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  uint32_t min_idx = blockIdx.x * blockDim.x;
  uint32_t max_idx = min_idx + blockDim.x;
  uint32_t min_block_idx = min_idx * (precision + 1) >> 5;
  uint32_t max_block_idx = max_idx * (precision + 1) >> 5;
  uint32_t block_idx = idx * (precision + 1) >> 5;
  if (threadIdx.x + min_block_idx < max_block_idx) {
    fshared[threadIdx.x] = 0;
  }

  value_t value = *(value_t*)(data + idx);
  uint32_t sign = (value >> (frac_len + exp_len)) & 0x1;
  uint32_t repr = value & ((1 << frac_len) - 1);
  uint32_t shift = 31 - precision - ((idx * (precision + 1)) & 31);
  uint8_t carry =
      (frac_len > precision) & ((repr >> (frac_len - precision - 1)) & 1);

  // repr -> compact fraction
  repr = repr >> (frac_len - precision);

  uint8_t overflow = (__popc(repr) == precision) & carry;

  // repr -> (sign, compact fraction)
  repr = (sign << precision) | (((1 << precision) - 1) & (repr + carry));
  // starting to store the fraction
  __syncthreads();  // wait for fractions to be initialized
  atomicOr(fshared + block_idx - min_block_idx, repr << shift);

  uint32_t exponent = (value >> frac_len) & ((1 << exp_len) - 1);
  // possibly resulting in infinity
  exponents[idx] = exponent + overflow;

  __syncthreads();  // wait for fractions to be stored
  if (threadIdx.x + min_block_idx < max_block_idx) {
    fractions[min_block_idx + threadIdx.x] = fshared[threadIdx.x];
  }

  /*
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  const int bias = (1 << (exp_len - 1)) - 1;

  value_t value = *(value_t*)(data + idx);
  value_t sign = (value >> (frac_len + exp_len)) & 0x1;
  value_t exponent = (value >> frac_len) & ((1 << exp_len) - 1);
  value_t fraction = value & ((1 << frac_len) - 1);

  fractions[idx] = fraction | (sign << frac_len);
  exponents[idx] = exponent;
  */
}

template <typename scalar_t,
          typename frac_t,
          typename value_t,
          int frac_len,
          int exp_len,
          int precision>
__global__ void merge_kernel(scalar_t* __restrict__ data,
                             uint8_t* __restrict__ exponents,
                             uint32_t* __restrict__ fractions,
                             size_t size) {
  __shared__ uint32_t fshared[THREADS / 32 * (precision + 1)];

  uint32_t min_idx = blockIdx.x * blockDim.x;
  uint32_t max_idx = min_idx + blockDim.x;

  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  uint32_t min_block_idx = min_idx * (precision + 1) >> 5;
  uint32_t max_block_idx = max_idx * (precision + 1) >> 5;
  uint32_t block_idx = idx * (precision + 1) >> 5;

  if (threadIdx.x + min_block_idx < max_block_idx) {
    fshared[threadIdx.x] = fractions[threadIdx.x + min_block_idx];
  }

  uint32_t shift_right = 31 - precision - ((idx * (precision + 1)) & 31);

  value_t exponent = exponents[idx] << frac_len;

  __syncthreads();

  value_t repr = (fshared[block_idx - min_block_idx] >> shift_right);

  value_t fraction = (repr & ((1 << precision) - 1)) << (frac_len - precision);
  value_t sign = (repr >> precision) & 0x1;

  value_t value = exponent | fraction | (sign << (frac_len + exp_len));

  data[idx] = *(scalar_t*)&value;
}

enum class Algorithm { ans, bitcomp, lz4, zstd, gdeflate };

// ********** Manager class *************

template <int precision>
struct Manager {
  const int chunk_size = 1 << 16;
  cudaStream_t estream;

  nvcomp::nvcompManagerBase* emanager;

  uint8_t *gl_exponents, *gl_comp_buffer;

  std::unordered_map<
      std::string,
      std::tuple<nvcomp::CompressionConfig, torch::Tensor, torch::Tensor>>
      compress_cache;

  std::unordered_map<std::string, std::tuple<at::ScalarType, int64_t>>
      meta_cache;

  Manager(const Algorithm& algorithm) {
    CUDA_CHECK(cudaStreamCreate(&estream));

    if (algorithm == Algorithm::ans) {
      emanager = new nvcomp::ANSManager(chunk_size, nvcompBatchedANSDefaultOpts,
                                        estream);
    } else if (algorithm == Algorithm::bitcomp) {
      nvcompBatchedBitcompFormatOpts format_opts{0, NVCOMP_TYPE_UCHAR};

      emanager = new nvcomp::BitcompManager(chunk_size, format_opts, estream);
    } else if (algorithm == Algorithm::lz4) {
      nvcompBatchedLZ4Opts_t format_opts{NVCOMP_TYPE_CHAR};
      emanager = new nvcomp::LZ4Manager(chunk_size, format_opts, estream);
    } else if (algorithm == Algorithm::zstd) {
      emanager = new nvcomp::ZstdManager(chunk_size,
                                         nvcompBatchedZstdDefaultOpts, estream);
    } else if (algorithm == Algorithm::gdeflate) {
      // 0: high-thruput, 1: high-comp, 2: entropy-only
      nvcompBatchedGdeflateOpts_t format_opts{2};
      emanager = new nvcomp::GdeflateManager(chunk_size, format_opts, estream);
    } else {
      throw std::runtime_error("Unsupported algorithm");
    }
  }

  template <typename scalar_t,
            typename frac_t,
            typename value_t,
            int frac_len,
            int exp_len>
  void _write_to_cache(const std::string& name, const torch::Tensor& input) {
    const int threads = THREADS;
    long size = input.numel();
    int blocks = (size + threads - 1) / threads;

    // CUDA_CHECK(cudaMallocAsync(&gl_exponents, size, estream));

    long frac_size = (size * (precision + 1) + 31) >> 5;

    torch::Tensor fractions_comp = torch::empty(
        {frac_size},
        torch::TensorOptions().dtype(torch::kUInt32).device(torch::kCUDA));

    torch::Tensor exponents_input_buffer = torch::empty(
        {size},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    split_kernel<scalar_t, frac_t, value_t, frac_len, exp_len, precision>
        <<<blocks, threads, 0, estream>>>(
            input.data_ptr<scalar_t>(),
            exponents_input_buffer.data_ptr<uint8_t>(),
            fractions_comp.data_ptr<uint32_t>(), input.numel());

    nvcomp::CompressionConfig comp_config =
        emanager->configure_compression(size);

    // CUDA_CHECK(cudaMallocAsync(
    //     &gl_comp_buffer, comp_config.max_compressed_buffer_size, estream));
    // std::cout << "Max compressed buffer size: "
    //           << static_cast<long>(comp_config.max_compressed_buffer_size)
    //           << std::endl;
    torch::Tensor exponents_output_buffer = torch::empty(
        {static_cast<long>(comp_config.max_compressed_buffer_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    emanager->compress(exponents_input_buffer.data_ptr<uint8_t>(),
                       exponents_output_buffer.data_ptr<uint8_t>(),
                       comp_config);

    // CUDA_CHECK(cudaFreeAsync(gl_exponents, estream));

    long compressed_size = emanager->get_compressed_output_size(
        exponents_output_buffer.data_ptr<uint8_t>());

    // std::cout << "Compressed size: " << compressed_size << std::endl;
    // option 1: create and copy
    // torch::Tensor exponents_comp = torch::empty(
    //     {compressed_size},
    //     torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    // CUDA_CHECK(cudaMemcpyAsync(exponents_comp.data_ptr<uint8_t>(),
    //                            exponents_output_buffer.data_ptr<uint8_t>(),
    //                            compressed_size, cudaMemcpyDeviceToDevice,
    //                            estream));

    // option 2: slice
    exponents_output_buffer = exponents_output_buffer.index(
        {torch::indexing::Slice(0, compressed_size)});

    compress_cache.insert({name,
                           {comp_config, std::move(exponents_output_buffer),
                            std::move(fractions_comp)}});
  }

  void write(const std::string& name, torch::Tensor tensor) {
    if (!tensor.is_cuda()) {
      tensor = tensor.to(torch::kCUDA);
    }

    if (meta_cache.find(name) != meta_cache.end()) {
      meta_cache.erase(name);
      compress_cache.erase(name);
    }

    // std::cout << "Writing " << name << " to cache" << std::endl;

    // std::cout << "Data type: " << tensor.dtype().toScalarType() <<
    // std::endl;

    // std::cout << "Shape: " << tensor.sizes() << std::endl;

    meta_cache.insert({name, {tensor.dtype().toScalarType(), tensor.numel()}});

    if (tensor.dtype().toScalarType() == at::ScalarType::Float) {
      const size_t frac_len = 23;
      const size_t exp_len = 8;
      return _write_to_cache<float, uint32_t, uint32_t, frac_len, exp_len>(
          name, tensor);
    } else if (tensor.dtype().toScalarType() == at::ScalarType::BFloat16) {
      const size_t frac_len = 7;
      const size_t exp_len = 8;
      return _write_to_cache<at::BFloat16, uint8_t, uint16_t, frac_len,
                             exp_len>(name, tensor);
    } else if (tensor.dtype().toScalarType() == at::ScalarType::Half) {
      const size_t frac_len = 10;
      const size_t exp_len = 5;
      return _write_to_cache<at::Half, uint16_t, uint16_t, frac_len, exp_len>(
          name, tensor);
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }

  template <typename scalar_t,
            typename frac_t,
            typename value_t,
            size_t frac_len,
            size_t exp_len>
  torch::Tensor _decompress_and_merge(const std::string& name, long size) {
    const int threads = THREADS;
    const at::ScalarType dtype = torch::CppTypeToScalarType<scalar_t>();

    torch::Tensor result = torch::empty(
        {size}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

    int blocks = (size + threads - 1) / threads;

    auto [exponents_config, exponents_comp, fractions_comp] =
        compress_cache.at(name);

    nvcomp::DecompressionConfig exp_decomp_config =
        emanager->configure_decompression(exponents_config);

    // CUDA_CHECK(cudaMallocAsync(&gl_exponents,
    //  exp_decomp_config.decomp_data_size, estream));
    torch::Tensor exponents_output_buffer = torch::empty(
        {static_cast<long>(exp_decomp_config.decomp_data_size)},
        torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA));

    emanager->decompress(exponents_output_buffer.data_ptr<uint8_t>(),
                         exponents_comp.data_ptr<uint8_t>(), exp_decomp_config);

    merge_kernel<scalar_t, frac_t, value_t, frac_len, exp_len, precision>
        <<<blocks, threads, 0, estream>>>(
            result.data_ptr<scalar_t>(),
            exponents_output_buffer.data_ptr<uint8_t>(),
            fractions_comp.data_ptr<uint32_t>(), size);

    CUDA_CHECK(cudaStreamSynchronize(estream));

    // CUDA_CHECK(cudaFreeAsync(gl_exponents, estream));

    return result;
  }

  uint64_t size(const std::string& name) {
    if (compress_cache.find(name) == compress_cache.end()) {
      return 0;
    }
    auto [_, exponents_comp, fractions_comp] = compress_cache.at(name);
    return exponents_comp.numel() * exponents_comp.element_size() +
           fractions_comp.numel() * fractions_comp.element_size();
  }

  torch::Tensor read(const std::string& name) {
    if (meta_cache.find(name) == meta_cache.end()) {
      throw std::runtime_error("Data not found");
    }

    auto [dtype, size] = meta_cache.at(name);

    if (dtype == at::ScalarType::Float) {
      const int frac_len = 23;
      const int exp_len = 8;
      return _decompress_and_merge<float, uint32_t, uint32_t, frac_len,
                                   exp_len>(name, size);
    } else if (dtype == at::ScalarType::Half) {
      const int frac_len = 10;
      const int exp_len = 5;
      return _decompress_and_merge<at::Half, uint16_t, uint16_t, frac_len,
                                   exp_len>(name, size);
    } else if (dtype == at::ScalarType::BFloat16) {
      const int frac_len = 7;
      const int exp_len = 8;
      return _decompress_and_merge<at::BFloat16, uint8_t, uint16_t, frac_len,
                                   exp_len>(name, size);
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }
};

// ********** Pybind11 *************

namespace py = pybind11;

template <int precision>
void create_manager_with_precision(py::module& m) {
  if ((precision + 1) & precision) {
    throw std::runtime_error("Precision must be (2^n - 1) or (-1)");
  }
  const std::string name =
      precision < 0 ? "Manager" : "ManagerM" + std::to_string(precision);
  using Class = Manager<precision>;
  py::class_<Class>(m, name.c_str())
      .def(py::init<const Algorithm&>(), py::arg("algorithm") = Algorithm::ans)
      .def("read", &Class::read)
      .def("write", &Class::write)
      .def("size", &Class::size);
}

PYBIND11_MODULE(neuzips_cuda, m) {
  py::enum_<Algorithm>(m, "Algorithm")
      .value("ans", Algorithm::ans)
      .value("bitcomp", Algorithm::bitcomp)
      .value("zstd", Algorithm::zstd)
      .value("lz4", Algorithm::lz4)
      .value("gdeflate", Algorithm::gdeflate);
  create_manager_with_precision<7>(m);
  create_manager_with_precision<3>(m);
  create_manager_with_precision<1>(m);
  create_manager_with_precision<0>(m);
}
