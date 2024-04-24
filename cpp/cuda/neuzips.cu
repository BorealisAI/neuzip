// #include <DietGpu.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <nvcomp.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/zstd.hpp>
#include <vector>

#define THREADS 1024
#define EXP_THREASHOLD 120

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

__device__ __forceinline__ float _fraction_to_base_float(uint32_t fraction) {
  constexpr uint32_t bias = 0x7f << 23;
  return __uint_as_float(fraction | bias);
}

__device__ __forceinline__ uint32_t _float_to_fraction(float number) {
  return __float_as_uint(number) & ((1 << 23) - 1);
}

template <typename scalar_t, /* half, bfloat16 */
          typename frac_t,   /* uint8_t, uint16_t */
          typename value_t,  /* uint16_t */
          int f_bits,        /* 0, 1, 3, 7 */
          int e_bits,
          int f_bits_save,
          int threads_per_block>
__global__ void kernel_get_full_fraction_if_exponents(
    uint8_t* __restrict__ exponents,       // in
    uint8_t* __restrict__ full_fractions,  // out
    size_t size) {
  constexpr uint32_t exp_threshold = EXP_THREASHOLD;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  const uint8_t maybe_full_fraction = (exponents[idx] > exp_threshold) ? 1 : 0;

  full_fractions[idx] = maybe_full_fraction;
}

template <typename scalar_t, /* half, bfloat16 */
          typename frac_t,   /* uint8_t, uint16_t */
          typename value_t,  /* uint16_t */
          int f_bits,        /* 0, 1, 3, 7 */
          int e_bits,
          int f_bits_save,
          int threads_per_block>
__global__ void kernel_aligned_split(scalar_t* __restrict__ data,
                                     uint8_t* __restrict__ exponents,
                                     uint8_t* __restrict__ fractions,
                                     uint8_t* __restrict__ full_fractions,
                                     size_t size) {
  // compile-time constants
  constexpr uint32_t threads_per_warp = 32;
  constexpr uint32_t warps_per_block = threads_per_block / threads_per_warp;
  constexpr uint32_t bytes_per_warp = (f_bits_save + 1) * 4;
  constexpr uint32_t logical_threads_per_warp = 8 / (f_bits_save + 1);
  constexpr uint32_t bytes_per_block = warps_per_block * bytes_per_warp;
  constexpr uint32_t exp_threshold = EXP_THREASHOLD;

  using WarpReduce = cub::WarpReduce<uint8_t, logical_threads_per_warp>;

  __shared__ typename WarpReduce::TempStorage warp_storage[bytes_per_block];

  // dynamic for each thread
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t byte_idx = idx * (f_bits_save + 1) / 8;
  const uint32_t bit_idx_in_byte = (idx * (f_bits_save + 1)) & 7;
  const uint32_t byte_idx_in_block = threadIdx.x * (f_bits_save + 1) / 8;
  const uint32_t shift = 7 - f_bits_save - bit_idx_in_byte;

  scalar_t scalar = (idx < size) ? data[idx] : static_cast<scalar_t>(0);

  value_t value = *(value_t*)(&scalar);
  const uint8_t sign = (value >> (f_bits + e_bits)) & 0x1;
  const uint8_t fraction = static_cast<uint8_t>(value & ((1 << f_bits) - 1));
  const uint8_t carry =
      (f_bits > f_bits_save) & ((fraction >> (f_bits - f_bits_save - 1)) & 1);
  const uint8_t exponent = (value >> f_bits) & ((1 << e_bits) - 1);

  // repr -> compact fraction
  uint8_t repr = fraction >> (f_bits - f_bits_save);

  uint8_t overflow = (__popc(repr) == f_bits_save) & carry;

  // repr -> (sign, compact fraction)
  repr = (sign << f_bits_save) | (((1 << f_bits_save) - 1) & (repr + carry));
  // starting to store the fraction

  // printf(
  //     "idx: %d, byte_idx: %d, bit_idx_in_byte: %d, byte_idx_in_block: %d, "
  //     "shift: %d, repr: %d, overflow: %d\n",
  //     idx, byte_idx, bit_idx_in_byte, byte_idx_in_block, shift, repr,
  //     overflow);
  const uint8_t byte_repr = (f_bits_save == 7)
                                ? repr
                                : WarpReduce(warp_storage[byte_idx_in_block])
                                      .Reduce(repr << shift, cub::Sum());

  // store the fraction
  if (bit_idx_in_byte == 0) {
    // only some threads write to the global memory
    fractions[byte_idx] = byte_repr;
  }

  // store the exponent
  // possibly resulting in infinity
  if (idx < size) {
    exponents[idx] = exponent + overflow;
    if (exponent + overflow > exp_threshold) {
      full_fractions[idx] = overflow ? 1 : fraction + 1;
    } else {
      full_fractions[idx] = 0;
    }
  }
}

template <typename scalar_t,
          typename frac_t,
          typename value_t,
          int f_bits,
          int e_bits,
          int f_bits_save,
          int threads_per_block>
__global__ void kernel_aligned_merge(scalar_t* __restrict__ data,
                                     uint8_t* __restrict__ exponents,
                                     uint8_t* __restrict__ fractions,
                                     uint8_t* __restrict__ full_fractions,
                                     size_t size) {
  constexpr uint32_t threads_per_warp = 32;
  constexpr uint32_t warps_per_block = threads_per_block / threads_per_warp;
  constexpr uint32_t bytes_per_warp = (f_bits_save + 1) * 4;
  constexpr uint32_t bytes_per_block = warps_per_block * bytes_per_warp;
  constexpr uint32_t exp_threshold = EXP_THREASHOLD;

  __shared__ uint8_t fshared[bytes_per_block];

  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t byte_idx = idx * (f_bits_save + 1) / 8;
  const uint32_t bit_idx_in_byte = (idx * (f_bits_save + 1)) & 7;
  const uint32_t byte_idx_in_block = threadIdx.x * (f_bits_save + 1) / 8;
  const uint32_t shift = 7 - f_bits_save - bit_idx_in_byte;

  if (bit_idx_in_byte == 0) {
    // load in shared memory to avoid reading from global memory multiple
    // times
    fshared[byte_idx_in_block] = fractions[byte_idx];
  }

  // assert that the exponents are in the correct range

  value_t exponent = exponents[idx];
  uint8_t maybe_full_fraction =
      (exponent > exp_threshold) ? full_fractions[idx] - 1 : 0;

  __syncthreads();

  const value_t repr = (fshared[byte_idx_in_block] >> shift);

  const value_t fraction =
      maybe_full_fraction
          ? maybe_full_fraction
          : ((repr & ((1 << f_bits_save) - 1)) << (f_bits - f_bits_save));
  const value_t sign = (repr >> f_bits_save) & 0x1;

  const value_t value =
      (sign << (f_bits + e_bits)) | (exponent << f_bits) | fraction;

  if (idx < size) {
    data[idx] = *(scalar_t*)&value;
  }
}

enum class Algorithm { ans, bitcomp, lz4, zstd, gdeflate };

// ********** Manager class *************

template <int f_bits_save>
struct Manager {
  const int chunk_size = 1 << 16;
  cudaStream_t estream;

  nvcomp::nvcompManagerBase* emanager;

  uint8_t *gl_exponents, *gl_comp_buffer;

  std::unordered_map<std::string,
                     std::tuple<nvcomp::CompressionConfig,
                                torch::Tensor,
                                torch::Tensor,
                                torch::Tensor>>
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
            int f_bits,
            int e_bits>
  void _write_to_cache(const std::string& name, const torch::Tensor& input) {
    constexpr int threads = THREADS;
    long size = input.numel();
    int blocks = (size + threads - 1) / threads;

    // CUDA_CHECK(cudaMallocAsync(&gl_exponents, size, estream));

    torch::Tensor fractions_comp = torch::empty(
        {(size * (f_bits_save + 1) + 7) / 8},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    torch::Tensor exponents_input_buffer = torch::empty(
        {size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    torch::Tensor full_fractions = torch::empty(
        {size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    kernel_aligned_split<scalar_t, frac_t, value_t, f_bits, e_bits, f_bits_save,
                         threads><<<blocks, threads, 0, estream>>>(
        input.data_ptr<scalar_t>(), exponents_input_buffer.data_ptr<uint8_t>(),
        fractions_comp.data_ptr<uint8_t>(), full_fractions.data_ptr<uint8_t>(),
        input.numel());

    full_fractions =
        full_fractions.index_select(0, full_fractions.nonzero().squeeze());

    nvcomp::CompressionConfig comp_config =
        emanager->configure_compression(size);

    // CUDA_CHECK(cudaMallocAsync(
    //     &gl_comp_buffer, comp_config.max_compressed_buffer_size, estream));
    // std::cout << "Max compressed buffer size: "
    //           << static_cast<long>(comp_config.max_compressed_buffer_size)
    //           << std::endl;
    torch::Tensor exponents_output_buffer = torch::empty(
        {static_cast<long>(comp_config.max_compressed_buffer_size)},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    emanager->compress(exponents_input_buffer.data_ptr<uint8_t>(),
                       exponents_output_buffer.data_ptr<uint8_t>(),
                       comp_config);

    // CUDA_CHECK(cudaFreeAsync(gl_exponents, estream));

    long compressed_size = emanager->get_compressed_output_size(
        exponents_output_buffer.data_ptr<uint8_t>());

    // std::cout << "Compressed size: " << compressed_size << std::endl;
    // option 1: create and copy
    torch::Tensor exponents_comp = torch::empty(
        {compressed_size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    CUDA_CHECK(cudaMemcpyAsync(exponents_comp.data_ptr<uint8_t>(),
                               exponents_output_buffer.data_ptr<uint8_t>(),
                               compressed_size, cudaMemcpyDeviceToDevice,
                               estream));

    // option 2: slice
    // exponents_output_buffer = exponents_output_buffer.index(
    //     {torch::indexing::Slice(0, compressed_size)});

    compress_cache.insert(
        {name,
         {comp_config, std::move(exponents_comp), std::move(fractions_comp),
          std::move(full_fractions)}});
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
      const size_t f_bits = 23;
      const size_t e_bits = 8;
      return _write_to_cache<float, uint32_t, uint32_t, f_bits, e_bits>(name,
                                                                        tensor);
    } else if (tensor.dtype().toScalarType() == at::ScalarType::BFloat16) {
      const size_t f_bits = 7;
      const size_t e_bits = 8;
      return _write_to_cache<at::BFloat16, uint8_t, uint16_t, f_bits, e_bits>(
          name, tensor);
    } else if (tensor.dtype().toScalarType() == at::ScalarType::Half) {
      const size_t f_bits = 10;
      const size_t e_bits = 5;
      return _write_to_cache<at::Half, uint16_t, uint16_t, f_bits, e_bits>(
          name, tensor);
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }

  template <typename scalar_t,
            typename frac_t,
            typename value_t,
            size_t f_bits,
            size_t e_bits>
  torch::Tensor _decompress_and_merge(const std::string& name, long size) {
    constexpr int threads = THREADS;
    const at::ScalarType dtype = torch::CppTypeToScalarType<scalar_t>();

    int blocks = (size + threads - 1) / threads;

    auto [exponents_config, exponents_comp, fractions_comp, c_full_fractions] =
        compress_cache.at(name);

    nvcomp::DecompressionConfig exp_decomp_config =
        emanager->configure_decompression(exponents_config);

    // CUDA_CHECK(cudaMallocAsync(&gl_exponents,
    //  exp_decomp_config.decomp_data_size, estream));
    torch::Tensor exponents_output_buffer = torch::empty(
        {static_cast<long>(exp_decomp_config.decomp_data_size)},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    emanager->decompress(exponents_output_buffer.data_ptr<uint8_t>(),
                         exponents_comp.data_ptr<uint8_t>(), exp_decomp_config);

    // torch::Tensor full_fractions = torch::empty(
    //     {size},
    //     torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    // kernel_get_full_fraction_if_exponents<scalar_t, frac_t, value_t, f_bits,
    //                                       e_bits, f_bits_save, threads>
    //     <<<blocks, threads, 0, estream>>>(
    //         exponents_output_buffer.data_ptr<uint8_t>(),
    //         full_fractions.data_ptr<uint8_t>(), size);

    // CUDA_CHECK(cudaStreamSynchronize(estream));
    torch::Tensor full_fractions =
        torch::empty_like(exponents_output_buffer)
            .index_put_({(exponents_output_buffer > EXP_THREASHOLD)
                             .nonzero()
                             .squeeze()},
                        c_full_fractions);

    torch::Tensor result = torch::empty(
        {size}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

    kernel_aligned_merge<scalar_t, frac_t, value_t, f_bits, e_bits, f_bits_save,
                         threads><<<blocks, threads, 0, estream>>>(
        result.data_ptr<scalar_t>(),
        exponents_output_buffer.data_ptr<uint8_t>(),
        fractions_comp.data_ptr<uint8_t>(), full_fractions.data_ptr<uint8_t>(),
        size);

    CUDA_CHECK(cudaStreamSynchronize(estream));

    // CUDA_CHECK(cudaFreeAsync(gl_exponents, estream));

    return result;
  }

  uint64_t size(const std::string& name) {
    if (compress_cache.find(name) == compress_cache.end()) {
      return 0;
    }
    auto [_, exponents_comp, fractions_comp, c_full_fractions] =
        compress_cache.at(name);
    return exponents_comp.numel() * exponents_comp.element_size() +
           fractions_comp.numel() * fractions_comp.element_size() +
           c_full_fractions.numel() * c_full_fractions.element_size();
  }

  torch::Tensor read(const std::string& name) {
    if (meta_cache.find(name) == meta_cache.end()) {
      throw std::runtime_error("Data not found");
    }

    auto [dtype, size] = meta_cache.at(name);

    if (dtype == at::ScalarType::Float) {
      const int f_bits = 23;
      const int e_bits = 8;
      return _decompress_and_merge<float, uint32_t, uint32_t, f_bits, e_bits>(
          name, size);
    } else if (dtype == at::ScalarType::Half) {
      const int f_bits = 10;
      const int e_bits = 5;
      return _decompress_and_merge<at::Half, uint16_t, uint16_t, f_bits,
                                   e_bits>(name, size);
    } else if (dtype == at::ScalarType::BFloat16) {
      const int f_bits = 7;
      const int e_bits = 8;
      return _decompress_and_merge<at::BFloat16, uint8_t, uint16_t, f_bits,
                                   e_bits>(name, size);
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }

  torch::Tensor linear(const std::string& name,
                       const torch::Tensor& input,
                       const at::IntArrayRef& shape,
                       const bool& transpose = false) {
    if (transpose) {
      return torch::matmul(input, this->read(name).view(shape).t());
    } else {
      return torch::matmul(input, this->read(name).view(shape));
    }
  }
};

// std::cout << "Compressed size: " << compressed_size << std::endl;
// option 1: create and copy
// torch::Tensor exponents_comp = torch::empty(
//     {compressed_size},
//     torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

// CUDA_CHECK(cudaMemcpyAsync(exponents_comp.data_ptr<uint8_t>(),
//

// ********** Pybind11 *************

namespace py = pybind11;

template <int f_bits_save>
void create_manager_with_f_bits_save(py::module& m) {
  if ((f_bits_save + 1) & f_bits_save) {
    throw std::runtime_error("f_bits_save must be (2^n - 1) or (-1)");
  }
  const std::string name =
      f_bits_save < 0 ? "Manager" : "ManagerM" + std::to_string(f_bits_save);
  using Class = Manager<f_bits_save>;
  py::class_<Class>(m, name.c_str())
      .def(py::init<const Algorithm&>(), py::arg("algorithm") = Algorithm::ans)
      .def("read", &Class::read)
      .def("write", &Class::write)
      .def("size", &Class::size)
      .def("linear", &Class::linear);
}

PYBIND11_MODULE(neuzips_cuda, m) {
  py::enum_<Algorithm>(m, "Algorithm")
      .value("ans", Algorithm::ans)
      .value("bitcomp", Algorithm::bitcomp)
      .value("zstd", Algorithm::zstd)
      .value("lz4", Algorithm::lz4)
      .value("gdeflate", Algorithm::gdeflate);
  create_manager_with_f_bits_save<7>(m);
  create_manager_with_f_bits_save<3>(m);
  create_manager_with_f_bits_save<1>(m);
  create_manager_with_f_bits_save<0>(m);
}
