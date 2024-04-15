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
#include <nvcomp/zstd.hpp>
#include <vector>

#include "cuda/algorithms.cuh"
#include "cuda/kernels.cuh"

// ********** Manager class *************

template <int f_bits_save, int threads_per_normalizer>
struct Manager {
  const int chunk_size;
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

  Manager(const Algorithm& algorithm, const int chunk_size)
      : chunk_size(chunk_size) {
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

    torch::Tensor fractions_comp = torch::empty(
        {(size * (f_bits_save + 1) + 7) / 8},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    torch::Tensor exponents_input_buffer = torch::empty(
        {size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    torch::Tensor normalizers = torch::empty(
        {threads_per_normalizer > 0
             ? (size + threads_per_normalizer - 1) / threads_per_normalizer
             : (size + threads - 1) / threads},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    kernel_aligned_split<
        scalar_t, frac_t, value_t, f_bits, e_bits, f_bits_save, threads,
        (threads_per_normalizer > 0 ? threads_per_normalizer : threads),
        (threads_per_normalizer > 0)><<<blocks, threads, 0, estream>>>(
        input.data_ptr<scalar_t>(), exponents_input_buffer.data_ptr<uint8_t>(),
        fractions_comp.data_ptr<uint8_t>(), normalizers.data_ptr<uint8_t>(),
        input.numel());

    nvcomp::CompressionConfig comp_config =
        emanager->configure_compression(size);

    torch::Tensor exponents_output_buffer = torch::empty(
        {static_cast<long>(comp_config.max_compressed_buffer_size)},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    emanager->compress(exponents_input_buffer.data_ptr<uint8_t>(),
                       exponents_output_buffer.data_ptr<uint8_t>(),
                       comp_config);

    long compressed_size = emanager->get_compressed_output_size(
        exponents_output_buffer.data_ptr<uint8_t>());

    torch::Tensor exponents_comp = torch::empty(
        {compressed_size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    CUDA_CHECK(cudaMemcpyAsync(exponents_comp.data_ptr<uint8_t>(),
                               exponents_output_buffer.data_ptr<uint8_t>(),
                               compressed_size, cudaMemcpyDeviceToDevice,
                               estream));

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

    torch::Tensor exponents_output_buffer = torch::empty(
        {static_cast<long>(exp_decomp_config.decomp_data_size)},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    emanager->decompress(exponents_output_buffer.data_ptr<uint8_t>(),
                         exponents_comp.data_ptr<uint8_t>(), exp_decomp_config);

    kernel_aligned_merge<
        scalar_t, frac_t, value_t, f_bits, e_bits, f_bits_save, threads,
        (threads_per_normalizer > 0 ? threads_per_normalizer : threads),
        (threads_per_normalizer > 0)><<<blocks, threads, 0, estream>>>(
        result.data_ptr<scalar_t>(),
        exponents_output_buffer.data_ptr<uint8_t>(),
        fractions_comp.data_ptr<uint8_t>(),
        normalizers_comp.data_ptr<uint8_t>(), size);

    CUDA_CHECK(cudaStreamSynchronize(estream));

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
                       const torch::Tensor& bias) {
    return torch::addmm(bias, input, this->read(name).view(shape).t());
  }

  torch::Tensor linear_without_bias(const std::string& name,
                                    const torch::Tensor& input,
                                    const at::IntArrayRef& shape) {
    return torch::matmul(input, this->read(name).view(shape).t());
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

    torch::Tensor fractions = torch::empty(
        {size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    torch::Tensor exponents = torch::empty(
        {size},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    torch::Tensor normalizers = torch::zeros(
        {threads_per_normalizer > 0
             ? (size + threads_per_normalizer - 1) / threads_per_normalizer
             : (size + threads - 1) / threads},
        torch::TensorOptions().dtype(torch::kByte).device(torch::kCUDA));

    kernel_aligned_split<scalar_t, frac_t, value_t, f_bits, e_bits, 7, threads,
                         (threads_per_normalizer > 0 ? threads_per_normalizer
                                                     : threads),
                         (threads_per_normalizer > 0)>
        <<<blocks, threads, 0, estream>>>(
            input.data_ptr<scalar_t>(), exponents.data_ptr<uint8_t>(),
            fractions.data_ptr<uint8_t>(), normalizers.data_ptr<uint8_t>(),
            input.numel());

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