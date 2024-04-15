// Copyright (c) 2024-present, Royal Bank of Canada.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#define THREADS 1024

#include <torch/extension.h>

#include <nvcomp.h>
#include <nvcomp/ans.h>
#include <nvcomp/ans.hpp>
#include "cuda/algorithms.cuh"
#include "cuda/high_level.cu"

namespace py = pybind11;

template <int f_bits_save, int threads_per_normalizer>
constexpr void create_manager(py::module& m) {
  if ((f_bits_save + 1) & f_bits_save) {
    throw std::runtime_error("f_bits_save must be (2^n - 1) or (-1)");
  }

  std::string name = "Manager_f" + std::to_string(f_bits_save) + "_n" +
                     std::to_string(threads_per_normalizer);
  using Class0 = Manager<f_bits_save, threads_per_normalizer>;
  py::class_<Class0>(m, name.c_str())
      .def(py::init<const Algorithm&, const int>(),
           py::arg("algorithm") = Algorithm::ans,
           py::arg("chunk_size") = 1 << 16)
      .def("read", &Class0::read)
      .def("write", &Class0::write)
      .def("size", &Class0::size)
      .def("linear", &Class0::linear)
      .def("linear_without_bias", &Class0::linear_without_bias)
      .def("split", &Class0::split);
}

PYBIND11_MODULE(_cuda, m) {
  py::enum_<Algorithm>(m, "Algorithm")
      .value("ans", Algorithm::ans)
      .value("bitcomp", Algorithm::bitcomp)
      .value("zstd", Algorithm::zstd)
      .value("lz4", Algorithm::lz4)
      .value("gdeflate", Algorithm::gdeflate);
  create_manager<0, 0>(m);
  create_manager<0, 1>(m);
  create_manager<0, 2>(m);
  create_manager<0, 4>(m);
  create_manager<0, 8>(m);
  create_manager<0, 16>(m);
  create_manager<0, 32>(m);
  create_manager<1, 0>(m);
  create_manager<1, 1>(m);
  create_manager<1, 2>(m);
  create_manager<1, 4>(m);
  create_manager<1, 8>(m);
  create_manager<1, 16>(m);
  create_manager<1, 32>(m);
  create_manager<3, 0>(m);
  create_manager<3, 1>(m);
  create_manager<3, 2>(m);
  create_manager<3, 4>(m);
  create_manager<3, 8>(m);
  create_manager<3, 16>(m);
  create_manager<3, 32>(m);
  create_manager<7, 0>(m);
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
