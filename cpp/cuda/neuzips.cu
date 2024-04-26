// #include <DietGpu.h>
#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <curand_kernel.h>
#include <cub/cub.cuh>
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

__device__ __forceinline__ float _fraction_to_base_float(uint32_t fraction) {
  constexpr uint32_t bias = 0x7f << 23;
  return __uint_as_float(bias | fraction);
}

__device__ __forceinline__ uint32_t _float_to_fraction(float number) {
  return __float_as_uint(number) & ((1 << 23) - 1);
}

template <typename T>
__device__ __forceinline__ T _has_carry_after_shift_and_around(T bits,
                                                               uint32_t shift) {
  return (bits >> (shift - 1)) & 1;
}

template <typename T>
__device__ __forceinline__ T _shift_and_around(T bits, uint32_t shift) {
  uint32_t full_bits = sizeof(T) * 8;
  uint32_t overflow = _has_carry_after_shift_and_around(bits, shift);
  return (bits >> shift) + overflow;
}

__global__ void kernel_setup_random_state(curandState_t* states,
                                          uint64_t seed) {
  curand_init(seed, threadIdx.x, 0, &states[threadIdx.x]);
}

__global__ void kernel_advance_random_state(curandState_t* states,
                                            unsigned long long step) {
  skipahead(step, &states[threadIdx.x]);
}

/**
 * Split the input tensor into exponents and fractions. It does two things:
 * 1. Find a normalizer for each block of @p THREADS elements. The normalizer
 * has a sign=0, exponent=0. The fraction is the fraction of the
 * max(abs(element)) in the block. The normalizer is later used to recover the
 * original value.
 * 2. After normalizing, the fraction is stored in the output tensor. The
 * exponent is stored in the output tensor.
 *
 * @
 */
template <typename scalar_t, /* half, bfloat16 */
          typename frac_t,   /* uint8_t, uint16_t */
          typename value_t,  /* uint16_t */
          int f_bits,        /* 0, 1, 3, 7 */
          int e_bits,
          int f_bits_save,
          int threads_per_block,
          int threads_per_normalizer,
          bool squared,
          bool normalized,
          bool stochastic>
__global__ void kernel_aligned_split(scalar_t* __restrict__ data,
                                     uint8_t* __restrict__ exponents,
                                     uint8_t* __restrict__ fractions,
                                     uint8_t* __restrict__ normalizers,
                                     curandState_t* states,
                                     size_t size) {
  // compile-time constants
  constexpr uint32_t threads_per_warp = 32;
  constexpr uint32_t warps_per_block = threads_per_block / threads_per_warp;
  constexpr uint32_t bytes_per_warp = (f_bits_save + 1) * 4;
  constexpr uint32_t logical_threads_per_warp = 8 / (f_bits_save + 1);
  constexpr uint32_t bytes_per_block = warps_per_block * bytes_per_warp;
  constexpr uint32_t normalizers_per_block =
      threads_per_block / threads_per_normalizer;

  using BlockReduce = cub::BlockReduce<float, threads_per_normalizer>;
  using WarpReduce = cub::WarpReduce<uint8_t, logical_threads_per_warp>;

  __shared__ typename WarpReduce::TempStorage warp_storage[bytes_per_block];
  __shared__
      typename BlockReduce::TempStorage block_storage[normalizers_per_block];
  __shared__ uint32_t block_normalizer[normalizers_per_block];

  // dynamic for each thread
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t byte_idx = idx * (f_bits_save + 1) / 8;
  const uint32_t bit_idx_in_byte = (idx * (f_bits_save + 1)) & 7;
  const uint32_t byte_idx_in_block = threadIdx.x * (f_bits_save + 1) / 8;
  const uint32_t shift = 7 - f_bits_save - bit_idx_in_byte;
  const uint32_t normalizer_idx = threadIdx.x / threads_per_normalizer;
  const uint32_t normalizer_mod = threadIdx.x % threads_per_normalizer;
  scalar_t scalar = (idx < size) ? data[idx] : static_cast<scalar_t>(0);

  curandState_t state = states[threadIdx.x];
  skipahead(blockIdx.x, &state);

  if (normalized) {
    // find the normalizer

    // only 1 thread per block has this value
    // printf("%d -> %d, scalar: %f\n", idx, normalizer_idx,
    //  static_cast<float>(scalar));
    float float_block_absmax = BlockReduce(block_storage[normalizer_idx])
                                   .Reduce(abs(static_cast<float>(scalar)),
                                           cub::Max(), threads_per_normalizer);

    // broadcast the normalizer to all threads
    if (normalizer_mod == 0) {
      block_normalizer[normalizer_idx] = _float_to_fraction(float_block_absmax);
    }
    __syncthreads();

    float_block_absmax =
        _fraction_to_base_float(block_normalizer[normalizer_idx]);

    // if (_float_to_fraction(static_cast<float>(scalar)) ==
    //     block_normalizer[normalizer_idx]) {
    //   float __s = static_cast<float>(scalar);
    //   if (_float_to_fraction(__s / float_block_absmax) != 0) {
    //     printf("scalar: %.10e, absmax: %.10e, normalizer: %d, div %.10e\n",
    //     __s,
    //            float_block_absmax, block_normalizer[normalizer_idx],
    //            __s / float_block_absmax);
    //   }
    // }

    scalar =
        static_cast<scalar_t>(static_cast<float>(scalar) / float_block_absmax);

    // all threads has the normalizer

    if (normalizer_mod == 0) {
      normalizers[normalizer_idx + blockIdx.x * normalizers_per_block] =
          static_cast<uint8_t>(block_normalizer[normalizer_idx] >> (23 - 8));
    }
  }

  if (squared) {
    scalar = static_cast<float>(scalar) * 134217728.0f * 134217728.0f *
             abs(static_cast<float>(scalar)) * 134217728.0f * 134217728.0f;
  }

  value_t value = *(value_t*)(&scalar);

  const uint8_t sign = (value >> (f_bits + e_bits)) & 0x1;
  uint8_t repr = static_cast<uint8_t>(value & ((1 << f_bits) - 1));
  uint8_t carry;
  if (!stochastic) {
    carry =
        (f_bits > f_bits_save) ? ((repr >> (f_bits - f_bits_save - 1)) & 1) : 0;
  } else {
    const uint8_t rd = curand(&state) & ((1 << f_bits) - 1);
    carry = (repr > rd) ? 1 : 0;
  }
  const uint8_t exponent = (value >> f_bits) & ((1 << e_bits) - 1);

  // repr -> compact fraction
  repr = repr >> (f_bits - f_bits_save);

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
  }
}

template <typename scalar_t,
          typename frac_t,
          typename value_t,
          int f_bits,
          int e_bits,
          int f_bits_save,
          int threads_per_block,
          int threads_per_normalizer,
          bool squared>
__global__ void kernel_aligned_merge(scalar_t* __restrict__ data,
                                     uint8_t* __restrict__ exponents,
                                     uint8_t* __restrict__ fractions,
                                     uint8_t* __restrict__ normalizers,
                                     size_t size) {
  constexpr uint32_t threads_per_warp = 32;
  constexpr uint32_t warps_per_block = threads_per_block / threads_per_warp;
  constexpr uint32_t bytes_per_warp = (f_bits_save + 1) * 4;
  constexpr uint32_t bytes_per_block = warps_per_block * bytes_per_warp;
  constexpr uint32_t normalizers_per_block =
      threads_per_block / threads_per_normalizer;

  __shared__ uint8_t fshared[bytes_per_block], nshared[normalizers_per_block];

  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t byte_idx = idx * (f_bits_save + 1) / 8;
  const uint32_t bit_idx_in_byte = (idx * (f_bits_save + 1)) & 7;
  const uint32_t byte_idx_in_block = threadIdx.x * (f_bits_save + 1) / 8;
  const uint32_t shift = 7 - f_bits_save - bit_idx_in_byte;
  const uint32_t normalizer_idx = threadIdx.x / threads_per_normalizer;

  if (threadIdx.x % threads_per_normalizer == 0) {
    // nshared[0] = normalizers[blockIdx.x];
    nshared[normalizer_idx] =
        normalizers[normalizer_idx + blockIdx.x * normalizers_per_block];
  }
  if (bit_idx_in_byte == 0) {
    // load in shared memory to avoid reading from global memory multiple times
    fshared[byte_idx_in_block] = fractions[byte_idx];
  }

  const value_t exponent = exponents[idx] << f_bits;
  __syncthreads();

  const value_t repr = (fshared[byte_idx_in_block] >> shift);

  const value_t fraction = (repr & ((1 << f_bits_save) - 1))
                           << (f_bits - f_bits_save);
  const value_t sign = (repr >> f_bits_save) & 0x1;

  const value_t value = exponent | fraction;

  if (idx < size) {
    if (squared) {
      data[idx] =
          _fraction_to_base_float(static_cast<uint32_t>(nshared[normalizer_idx])
                                  << (23 - 8)) *
          sqrt(static_cast<float>(*(scalar_t*)&value)) / 134217728.0f /
          134217728.0f * (sign ? -1.0f : 1.0f);
    } else {
      data[idx] =
          _fraction_to_base_float(static_cast<uint32_t>(nshared[normalizer_idx])
                                  << (23 - 8)) *
          static_cast<float>(*(scalar_t*)&value) * (sign ? -1.0f : 1.0f);
    }
  }
}

enum class Algorithm { ans, bitcomp, lz4, zstd, gdeflate };

// ********** Manager class *************

template <int f_bits_save,
          int threads_per_normalizer,
          bool squared,
          bool stochastic>
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

  curandState_t* states;

  Manager(const Algorithm& algorithm, uint64_t seed = 0) {
    CUDA_CHECK(cudaStreamCreate(&estream));
    for (int i = 0; i < THREADS; i++) {
      CUDA_CHECK(cudaMalloc(&states, THREADS * sizeof(curandState_t)));
      kernel_setup_random_state<<<1, THREADS>>>(states, seed);
    }

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

    torch::Tensor normalizers = torch::empty(
        {(size + threads_per_normalizer - 1) / threads_per_normalizer},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    kernel_aligned_split<scalar_t, frac_t, value_t, f_bits, e_bits, f_bits_save,
                         threads, threads_per_normalizer, squared,
                         (threads_per_normalizer != 0), stochastic>
        <<<blocks, threads, 0, estream>>>(
            input.data_ptr<scalar_t>(),
            exponents_input_buffer.data_ptr<uint8_t>(),
            fractions_comp.data_ptr<uint8_t>(), normalizers.data_ptr<uint8_t>(),
            states, input.numel());

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

    kernel_advance_random_state<<<1, THREADS, 0, estream>>>(states, blocks);

    compress_cache.insert(
        {name,
         {comp_config, std::move(exponents_comp), std::move(fractions_comp),
          std::move(normalizers)}});
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
    tensor = tensor.detach();

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

    torch::Tensor result = torch::empty(
        {size}, torch::TensorOptions().dtype(dtype).device(torch::kCUDA));

    int blocks = (size + threads - 1) / threads;

    const auto& content = compress_cache.at(name);
    const auto& exponents_config = std::get<0>(content);
    const auto& exponents_comp = std::get<1>(content);
    const auto& fractions_comp = std::get<2>(content);
    const auto& normalizers_comp = std::get<3>(content);

    nvcomp::DecompressionConfig exp_decomp_config =
        emanager->configure_decompression(exponents_config);

    // CUDA_CHECK(cudaMallocAsync(&gl_exponents,
    //  exp_decomp_config.decomp_data_size, estream));
    torch::Tensor exponents_output_buffer = torch::empty(
        {static_cast<long>(exp_decomp_config.decomp_data_size)},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    emanager->decompress(exponents_output_buffer.data_ptr<uint8_t>(),
                         exponents_comp.data_ptr<uint8_t>(), exp_decomp_config);

    kernel_aligned_merge<scalar_t, frac_t, value_t, f_bits, e_bits, f_bits_save,
                         threads, threads_per_normalizer, squared>
        <<<blocks, threads, 0, estream>>>(
            result.data_ptr<scalar_t>(),
            exponents_output_buffer.data_ptr<uint8_t>(),
            fractions_comp.data_ptr<uint8_t>(),
            normalizers_comp.data_ptr<uint8_t>(), size);

    CUDA_CHECK(cudaStreamSynchronize(estream));

    // CUDA_CHECK(cudaFreeAsync(gl_exponents, estream));

    return result;
  }

  uint64_t size(const std::string& name) {
    if (compress_cache.find(name) == compress_cache.end()) {
      return 0;
    }
    const auto& content = compress_cache.at(name);
    const auto& config = std::get<0>(content);
    const auto& exponents_comp = std::get<1>(content);
    const auto& fractions_comp = std::get<2>(content);
    const auto& normalizers_comp = std::get<3>(content);

    return exponents_comp.numel() * exponents_comp.element_size() +
           fractions_comp.numel() * fractions_comp.element_size() +
           normalizers_comp.numel() * normalizers_comp.element_size();
  }

  torch::Tensor read(const std::string& name) {
    if (meta_cache.find(name) == meta_cache.end()) {
      throw std::runtime_error("Data not found");
    }

    const auto& content = meta_cache.at(name);
    const auto& dtype = std::get<0>(content);
    const auto& size = std::get<1>(content);

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

  template <typename scalar_t,
            typename frac_t,
            typename value_t,
            int f_bits,
            int e_bits>
  std::vector<torch::Tensor> _split(const torch::Tensor& input) {
    constexpr int threads = THREADS;
    long size = input.numel();
    int blocks = (size + threads - 1) / threads;

    // CUDA_CHECK(cudaMallocAsync(&gl_exponents, size, estream));

    torch::Tensor fractions = torch::empty(
        {size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    torch::Tensor exponents = torch::empty(
        {size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    torch::Tensor normalizers = torch::zeros(
        {(size + threads_per_normalizer - 1) / threads_per_normalizer},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    kernel_aligned_split<scalar_t, frac_t, value_t, f_bits, e_bits, 7, threads,
                         threads_per_normalizer, false,
                         (threads_per_normalizer != 0), stochastic>
        <<<blocks, threads, 0, estream>>>(
            input.data_ptr<scalar_t>(), exponents.data_ptr<uint8_t>(),
            fractions.data_ptr<uint8_t>(), normalizers.data_ptr<uint8_t>(),
            states, input.numel());

    return {exponents, fractions};
  }

  std::vector<torch::Tensor> split(torch::Tensor input) {
    if (!input.is_cuda()) {
      input = input.to(torch::kCUDA);
    }

    if (input.dtype().toScalarType() == at::ScalarType::Float) {
      const size_t f_bits = 23;
      const size_t e_bits = 8;
      return _split<float, uint32_t, uint32_t, f_bits, e_bits>(input);
    } else if (input.dtype().toScalarType() == at::ScalarType::BFloat16) {
      const size_t f_bits = 7;
      const size_t e_bits = 8;
      return _split<at::BFloat16, uint8_t, uint16_t, f_bits, e_bits>(input);
    } else if (input.dtype().toScalarType() == at::ScalarType::Half) {
      const size_t f_bits = 10;
      const size_t e_bits = 5;
      return _split<at::Half, uint16_t, uint16_t, f_bits, e_bits>(input);
    } else {
      throw std::runtime_error("Unsupported data type");
    }
  }
};

// ********** Pybind11 *************

namespace py = pybind11;

template <int f_bits_save, int threads_per_normalizer>
constexpr void create_manager(py::module& m) {
  if ((f_bits_save + 1) & f_bits_save) {
    throw std::runtime_error("f_bits_save must be (2^n - 1) or (-1)");
  }

  std::string name = "Manager_f" + std::to_string(f_bits_save) + "_n" +
                     std::to_string(threads_per_normalizer) +
                     (false ? "_sqt" : "_sqf") + (false ? "_srt" : "_srf");
  using Class0 = Manager<f_bits_save, threads_per_normalizer, false, false>;
  py::class_<Class0>(m, name.c_str())
      .def(py::init<const Algorithm&, uint64_t>(),
           py::arg("algorithm") = Algorithm::ans, py::arg("seed") = 0)
      .def("read", &Class0::read)
      .def("write", &Class0::write)
      .def("size", &Class0::size)
      .def("linear", &Class0::linear)
      .def("split", &Class0::split);

  name = "Manager_f" + std::to_string(f_bits_save) + "_n" +
         std::to_string(threads_per_normalizer) + (false ? "_sqt" : "_sqf") +
         (true ? "_srt" : "_srf");
  using Class1 = Manager<f_bits_save, threads_per_normalizer, false, true>;
  py::class_<Class1>(m, name.c_str())
      .def(py::init<const Algorithm&, uint64_t>(),
           py::arg("algorithm") = Algorithm::ans, py::arg("seed") = 0)
      .def("read", &Class1::read)
      .def("write", &Class1::write)
      .def("size", &Class1::size)
      .def("linear", &Class1::linear)
      .def("split", &Class1::split);

  name = "Manager_f" + std::to_string(f_bits_save) + "_n" +
         std::to_string(threads_per_normalizer) + (true ? "_sqt" : "_sqf") +
         (false ? "_srt" : "_srf");
  using Class2 = Manager<f_bits_save, threads_per_normalizer, true, false>;
  py::class_<Class2>(m, name.c_str())
      .def(py::init<const Algorithm&, uint64_t>(),
           py::arg("algorithm") = Algorithm::ans, py::arg("seed") = 0)
      .def("read", &Class2::read)
      .def("write", &Class2::write)
      .def("size", &Class2::size)
      .def("linear", &Class2::linear)
      .def("split", &Class2::split);

  name = "Manager_f" + std::to_string(f_bits_save) + "_n" +
         std::to_string(threads_per_normalizer) + (true ? "_sqt" : "_sqf") +
         (true ? "_srt" : "_srf");
  using Class3 = Manager<f_bits_save, threads_per_normalizer, true, true>;
  py::class_<Class3>(m, name.c_str())
      .def(py::init<const Algorithm&, uint64_t>(),
           py::arg("algorithm") = Algorithm::ans, py::arg("seed") = 0)
      .def("read", &Class3::read)
      .def("write", &Class3::write)
      .def("size", &Class3::size)
      .def("linear", &Class3::linear)
      .def("split", &Class3::split);
}

// const std::string name = "Manager_f" + std::to_string(f_bits_save) + "_n" +
//                          std::to_string(threads_per_normalizer) +
//                          (squared ? "_sqt" : "_sqf");
// using Class = Manager<f_bits_save, threads_per_normalizer, squared>;
// py::class_<Class>(m, name.c_str())
//     .def(py::init<const Algorithm&>(), py::arg("algorithm") = Algorithm::ans)
//     .def("read", &Class::read)
//     .def("write", &Class::write)
//     .def("size", &Class::size)
//     .def("linear", &Class::linear)
//     .def("split", &Class::split);
// }
// const std::string name = "ManagerM" + std::to_string(f_bits_save) + "N" +
//                          std::to_string(threads_per_normalizer);
// using Class = Manager<f_bits_save, threads_per_normalizer, false>;
// py::class_<Class>(m, name.c_str())
//     .def(py::init<const Algorithm&>(), py::arg("algorithm") = Algorithm::ans)
//     .def("read", &Class::read)
//     .def("write", &Class::write)
//     .def("size", &Class::size)
//     .def("linear", &Class::linear)
//     .def("split", &Class::split);

// const std::string name2 = "ManagerM" + std::to_string(f_bits_save) + "N" +
//                           std::to_string(threads_per_normalizer) + "S";
// using Class1 = Manager<f_bits_save, threads_per_normalizer, true>;
// py::class_<Class1>(m, name2.c_str())
//     .def(py::init<const Algorithm&>(), py::arg("algorithm") = Algorithm::ans)
//     .def("read", &Class1::read)
//     .def("write", &Class1::write)
//     .def("size", &Class1::size)
//     .def("linear", &Class1::linear)
//     .def("split", &Class1::split);
// }

PYBIND11_MODULE(neuzips_cuda, m) {
  py::enum_<Algorithm>(m, "Algorithm")
      .value("ans", Algorithm::ans)
      .value("bitcomp", Algorithm::bitcomp)
      .value("zstd", Algorithm::zstd)
      .value("lz4", Algorithm::lz4)
      .value("gdeflate", Algorithm::gdeflate);
  create_manager<0, 1>(m);
  create_manager<0, 2>(m);
  create_manager<0, 4>(m);
  create_manager<0, 8>(m);
  create_manager<0, 16>(m);
  create_manager<0, 32>(m);
  create_manager<1, 1>(m);
  create_manager<1, 2>(m);
  create_manager<1, 4>(m);
  create_manager<1, 8>(m);
  create_manager<1, 16>(m);
  create_manager<1, 32>(m);
  create_manager<3, 1>(m);
  create_manager<3, 2>(m);
  create_manager<3, 4>(m);
  create_manager<3, 8>(m);
  create_manager<3, 16>(m);
  create_manager<3, 32>(m);
  create_manager<7, 1>(m);
  create_manager<7, 2>(m);
  create_manager<7, 4>(m);
  create_manager<7, 8>(m);
  create_manager<7, 16>(m);
  create_manager<7, 32>(m);

#if THREADS >= 64
  create_manager<0, 64>(m);
  create_manager<1, 64>(m);
  create_manager<3, 64>(m);
  create_manager<7, 64>(m);
#endif
#if THREADS >= 128
  create_manager<0, 128>(m);
  create_manager<1, 128>(m);
  create_manager<3, 128>(m);
  create_manager<7, 128>(m);
#endif
#if THREADS >= 256
  create_manager<0, 256>(m);
  create_manager<1, 256>(m);
  create_manager<3, 256>(m);
  create_manager<7, 256>(m);
#endif
#if THREADS >= 512
  create_manager<0, 512>(m);
  create_manager<1, 512>(m);
  create_manager<3, 512>(m);
  create_manager<7, 512>(m);
#endif
#if THREADS >= 1024
  create_manager<0, 1024>(m);
  create_manager<1, 1024>(m);
  create_manager<3, 1024>(m);
  create_manager<7, 1024>(m);
#endif
}
