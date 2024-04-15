// #include <DietGpu.h>
#include <torch/extension.h>
#include <nvcomp.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/lz4.hpp>
#include <nvcomp/zstd.hpp>
#include <nvcomp/gdeflate.hpp>
#include <vector>

#define DEFAULT_PRECISION 20

#define CUDA_CHECK(cond)                   \
  do {                                     \
    cudaError_t err = cond;                \
    if (err != cudaSuccess) {              \
      std::cerr << "Failure" << std::endl; \
      exit(1);                             \
    }                                      \
  } while (false)

enum class Algorithm { ans, bitcomp, lz4, zstd, gdeflate};

template <typename frac_t, int frac_len>
__device__ __forceinline__ frac_t clip_fraction(int real_exponent,
                                                frac_t fraction,
                                                int precision) {
  int last_n = min(frac_len, max(0, -real_exponent + frac_len - precision));
  return (fraction >> last_n) << last_n;
}

template <typename scalar_t, typename frac_t, typename value_t, int frac_len,
          int exp_len>
__global__ void split_kernel(scalar_t* data, uint8_t* exponents,
                             frac_t* fractions, size_t size, int precision) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }
  const int bias = (1 << (exp_len - 1)) - 1;

  value_t value = *(value_t*)(data + idx);
  value_t sign = (value >> (frac_len + exp_len)) & 0x1;
  value_t exponent = (value >> frac_len) & ((1 << exp_len) - 1);

  frac_t fraction = clip_fraction<frac_t, frac_len>(
      exponent - bias, (value & ((1 << frac_len) - 1)), precision);

  fractions[idx] = fraction | (sign << frac_len);
  exponents[idx] = exponent;
}

template <typename scalar_t, typename frac_t, typename value_t, int frac_len,
          int exp_len>
__global__ void merge_kernel(scalar_t* data, uint8_t* exponents,
                             frac_t* fractions, size_t size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= size) {
    return;
  }

  value_t exponent = exponents[idx];
  value_t fraction = fractions[idx] & ((1 << frac_len) - 1);
  value_t sign = (fractions[idx] >> frac_len) & 0x1;

  value_t value =
      (exponent << frac_len) | fraction | (sign << (frac_len + exp_len));

  data[idx] = *(scalar_t*)&value;
}

// ********** Manager class *************
struct Manager {
  const int chunk_size = 1 << 16;
  cudaStream_t stream;
  nvcomp::nvcompManagerBase* manager;
  const int _precision;

  Manager(const Algorithm& algorithm, const int& precision = DEFAULT_PRECISION)
      : _precision(precision) {
    CUDA_CHECK(cudaStreamCreate(&stream));

    if (algorithm == Algorithm::ans) {
      manager = new nvcomp::ANSManager(chunk_size, nvcompBatchedANSDefaultOpts,
                                       stream);
    } else if (algorithm == Algorithm::bitcomp) {
      nvcompBatchedBitcompFormatOpts format_opts{0, NVCOMP_TYPE_UCHAR};
      manager = new nvcomp::BitcompManager(chunk_size, format_opts, stream);
    } else if (algorithm == Algorithm::lz4) {
      nvcompBatchedLZ4Opts_t format_opts{NVCOMP_TYPE_CHAR};
      manager = new nvcomp::LZ4Manager(chunk_size, format_opts, stream);
    } else if (algorithm == Algorithm::zstd) {
      manager = new nvcomp::ZstdManager(chunk_size,
                                        nvcompBatchedZstdDefaultOpts, stream);
    } else if (algorithm == Algorithm::gdeflate) {
      // 0: high-thruput, 1: high-comp, 2: entropy-only
      nvcompBatchedGdeflateOpts_t format_opts{2};
      manager = new nvcomp::GdeflateManager(chunk_size, format_opts, stream);
    } else {
      throw std::runtime_error("Unsupported algorithm");
    }
  }

  ~Manager() {
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete manager;
  }

  template <typename T>
  torch::Tensor _array_compress(size_t size, T* data) {
    nvcomp::CompressionConfig comp_config =
        manager->configure_compression(size * sizeof(T));
    uint8_t* comp_buffer;
    CUDA_CHECK(
        cudaMalloc(&comp_buffer, comp_config.max_compressed_buffer_size));
    manager->compress((uint8_t*)data, comp_buffer, comp_config);
    int compressed_size = manager->get_compressed_output_size(comp_buffer);
    torch::Tensor result = torch::empty(
        {compressed_size},
        at::TensorOptions().device(at::kCUDA).dtype(at::ScalarType::Byte));
    cudaMemcpy(result.data_ptr(), comp_buffer, compressed_size,
               cudaMemcpyDeviceToDevice);

    CUDA_CHECK(cudaFree(comp_buffer));

    return result;
  }

  template <typename scalar_t, typename frac_t, typename value_t, int frac_len,
            int exp_len, bool compress>
  std::vector<torch::Tensor> _split_and_compress(torch::Tensor& input,
                                                 int precision) {
    const int threads = 1024;
    size_t size = input.numel();
    int blocks = (size + threads - 1) / threads;

    uint8_t* exponents;
    frac_t* fractions;

    CUDA_CHECK(cudaMalloc(&exponents, size));
    CUDA_CHECK(cudaMalloc(&fractions, size * sizeof(frac_t)));

    split_kernel<scalar_t, frac_t, value_t, frac_len, exp_len>
        <<<blocks, threads>>>(input.data_ptr<scalar_t>(), exponents, fractions,
                              input.numel(), precision);

    std::vector<torch::Tensor> results;

    if (compress) {
      results.emplace_back(_array_compress<uint8_t>(size, exponents));
      results.emplace_back(_array_compress<frac_t>(size, fractions));
    } else {
      torch::Tensor exponents_tensor =
          torch::empty({input.numel()},
                       at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
      torch::Tensor fractions_tensor = torch::empty(
          {input.numel()}, at::TensorOptions().device(at::kCUDA).dtype(
                               torch::CppTypeToScalarType<frac_t>()));
      // there is also ScalarTypeToCPPType
      cudaMemcpyAsync(exponents_tensor.data_ptr(), exponents, size,
                      cudaMemcpyDeviceToDevice, stream);
      cudaMemcpyAsync(fractions_tensor.data_ptr(), fractions,
                      size * sizeof(frac_t), cudaMemcpyDeviceToDevice, stream);
      results.push_back(exponents_tensor);
      results.push_back(fractions_tensor);
    }

    cudaStreamSynchronize(stream);

    cudaFree(exponents);
    cudaFree(fractions);

    return results;
  }

  std::vector<torch::Tensor> split_and_compress(torch::Tensor& input) {
    if (input.device() != torch::kCUDA) {
      input = input.to(at::kCUDA);
    }
    input = input.reshape({-1});

    std::vector<torch::Tensor> results;
    if (input.dtype().toScalarType() == at::ScalarType::Float) {
      const size_t frac_len = 23;
      const size_t exp_len = 8;
      return _split_and_compress<float, uint32_t, uint32_t, frac_len, exp_len,
                                 true>(input, _precision);
    } else if (input.dtype().toScalarType() == at::ScalarType::BFloat16) {
      const size_t frac_len = 7;
      const size_t exp_len = 8;
      return _split_and_compress<at::BFloat16, uint8_t, uint16_t, frac_len,
                                 exp_len, true>(input, _precision);
    } else if (input.dtype().toScalarType() == at::ScalarType::Half) {
      const size_t frac_len = 10;
      const size_t exp_len = 5;
      return _split_and_compress<at::Half, uint16_t, uint16_t, frac_len,
                                 exp_len, true>(input, _precision);
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }

  std::vector<torch::Tensor> split_only(torch::Tensor& input) {
    if (input.device() != torch::kCUDA) {
      input = input.to(at::kCUDA);
    }
    input = input.reshape({-1});

    std::vector<torch::Tensor> results;
    if (input.dtype().toScalarType() == at::ScalarType::Float) {
      const size_t frac_len = 23;
      const size_t exp_len = 8;
      return _split_and_compress<float, uint32_t, uint32_t, frac_len, exp_len,
                                 false>(input, _precision);
    } else if (input.dtype().toScalarType() == at::ScalarType::BFloat16) {
      const size_t frac_len = 7;
      const size_t exp_len = 8;
      return _split_and_compress<at::BFloat16, uint8_t, uint16_t, frac_len,
                                 exp_len, false>(input, _precision);
    } else if (input.dtype().toScalarType() == at::ScalarType::Half) {
      const size_t frac_len = 10;
      const size_t exp_len = 5;
      return _split_and_compress<at::Half, uint16_t, uint16_t, frac_len,
                                 exp_len, false>(input, _precision);
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }

  template <typename scalar_t>
  torch::Tensor _array_decompress(torch::Tensor& data) {
    uint8_t* comp_buffer = data.data_ptr<uint8_t>();
    nvcomp::DecompressionConfig decomp_config =
        manager->configure_decompression(comp_buffer);

    uint8_t* decomp_buffer;
    CUDA_CHECK(cudaMalloc(&decomp_buffer, decomp_config.decomp_data_size));

    int size = static_cast<int>(decomp_config.decomp_data_size);

    // std::cout << "Decompressing size: " << size << std::endl;

    manager->decompress(decomp_buffer, comp_buffer, decomp_config);

    // std::cout << "Decompressed" << std::endl;

    CUDA_CHECK(cudaStreamSynchronize(stream));

    // std::cout << "Creating tensor" << std::endl;

    torch::Tensor result =
        torch::empty({size}, at::TensorOptions().device(at::kCUDA).dtype(
                                 torch::CppTypeToScalarType<scalar_t>()));

    // std::cout << "Copying to tensor" << std::endl;

    cudaMemcpy((uint8_t*)result.data_ptr(), decomp_buffer, size,
               cudaMemcpyDeviceToDevice);

    // std::cout << "Freeing memory" << std::endl;

    // CUDA_CHECK(cudaFree(comp_buffer));

    // std::cout << "Freeing memory" << std::endl;

    CUDA_CHECK(cudaFree(decomp_buffer));

    return result;
  }

  template <typename scalar_t, typename frac_t, typename value_t,
            size_t frac_len, size_t exp_len>
  torch::Tensor _decompress_and_merge(torch::Tensor& exponents,
                                      torch::Tensor& fractions,
                                      torch::Tensor& result) {
    const int threads = 1024;
    size_t size = result.numel();
    int blocks = (size + threads - 1) / threads;

    // std::cout << "Decompressing exponents" << std::endl;

    exponents = _array_decompress<uint8_t>(exponents);
    fractions = _array_decompress<frac_t>(fractions);

    at::ScalarType dtype = torch::CppTypeToScalarType<scalar_t>();

    merge_kernel<scalar_t, frac_t, value_t, frac_len, exp_len>
        <<<blocks, threads>>>(result.data_ptr<scalar_t>(),
                              exponents.data_ptr<uint8_t>(),
                              fractions.data_ptr<frac_t>(), size);

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return result;
  }

  torch::Tensor decompress_and_merge(torch::Tensor& exponents,
                                     torch::Tensor& fractions,
                                     torch::Tensor& result) {
    if (result.dtype().toScalarType() == at::ScalarType::Float) {
      const int frac_len = 23;
      const int exp_len = 8;
      return _decompress_and_merge<float, uint32_t, uint32_t, frac_len,
                                   exp_len>(exponents, fractions, result);
    } else if (result.dtype().toScalarType() == at::ScalarType::Half) {
      const int frac_len = 10;
      const int exp_len = 5;
      return _decompress_and_merge<at::Half, uint16_t, uint16_t, frac_len,
                                   exp_len>(exponents, fractions, result);
    } else if (result.dtype().toScalarType() == at::ScalarType::BFloat16) {
      const int frac_len = 7;
      const int exp_len = 8;
      return _decompress_and_merge<at::BFloat16, uint8_t, uint16_t, frac_len,
                                   exp_len>(exponents, fractions, result);
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }
};
namespace py = pybind11;

PYBIND11_MODULE(neuzips, m) {
  // m.def("decompress", &decompress, "Decompress data");
  // m.def("compress", &compress, "Compress data");
  // m.def("forward", &forward, "Fetch data from weight");
  // m.def("backward", &backward, "Compress data to weight");
  py::enum_<Algorithm>(m, "Algorithm")
      .value("ans", Algorithm::ans)
      .value("bitcomp", Algorithm::bitcomp)
      .value("zstd", Algorithm::zstd)
      .value("lz4", Algorithm::lz4)
      .value("gdeflate", Algorithm::gdeflate);
  py::class_<Manager>(m, "Manager")
      .def(py::init<const Algorithm&, const int&>(),
           py::arg("algorithm") = Algorithm::ans,
           py::arg("precision") = DEFAULT_PRECISION)
      .def("split_and_compress", &Manager::split_and_compress)
      .def("split_only", &Manager::split_only)
      .def("decompress_and_merge", &Manager::decompress_and_merge);
}
