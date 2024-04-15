// Copyright (c) 2024-present, Royal Bank of Canada.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>
#include <nvcomp.hpp>
#include <nvcomp/ans.hpp>
#include <nvcomp/bitcomp.hpp>
#include <nvcomp/gdeflate.hpp>
#include <nvcomp/lz4.hpp>
#include <vector>

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

template <typename scalar_t, /* half, bfloat16 */
          typename frac_t,   /* uint8_t, uint16_t */
          typename value_t,  /* uint16_t */
          int f_bits,        /* 0, 1, 3, 7 */
          int e_bits,
          int f_bits_save,
          int threads_per_block,
          int threads_per_normalizer,
          bool normalized>
__global__ void kernel_aligned_split(scalar_t* __restrict__ data,
                                     uint8_t* __restrict__ exponents,
                                     uint8_t* __restrict__ fractions,
                                     uint8_t* __restrict__ normalizers,
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

  if (normalized) {
    // find the normalizer
    // only 1 thread per block has this value
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

    scalar =
        static_cast<scalar_t>(static_cast<float>(scalar) / float_block_absmax);

    // all threads has the normalizer
    if (normalizer_mod == 0) {
      normalizers[normalizer_idx + blockIdx.x * normalizers_per_block] =
          static_cast<uint8_t>(block_normalizer[normalizer_idx] >> (23 - 8));
    }
  }

  value_t value = *(value_t*)(&scalar);

  const uint8_t sign = (value >> (f_bits + e_bits)) & 0x1;
  uint8_t repr = static_cast<uint8_t>(value & ((1 << f_bits) - 1));
  uint8_t carry;

  carry =
      (f_bits > f_bits_save) ? ((repr >> (f_bits - f_bits_save - 1)) & 1) : 0;

  const uint8_t exponent = (value >> f_bits) & ((1 << e_bits) - 1);

  // repr -> compact fraction
  repr = repr >> (f_bits - f_bits_save);

  uint8_t overflow = (__popc(repr) == f_bits_save) & carry;

  // repr -> (sign, compact fraction)
  repr = (sign << f_bits_save) | (((1 << f_bits_save) - 1) & (repr + carry));

  // starting to store the fraction
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
          bool normalized>
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

  const float normalized_value =
      normalized
          ? _fraction_to_base_float(
                static_cast<uint32_t>(nshared[normalizer_idx]) << (23 - 8))
          : 1.0f;
  if (idx < size) {
    data[idx] = normalized_value * static_cast<float>(*(scalar_t*)&value) *
                (sign ? -1.0f : 1.0f);
  }
}
