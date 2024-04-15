# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NEUZIP_VERSION = "0.0.1"

setup(
    name="neuzip",
    ext_modules=[
        CUDAExtension(
            name="neuzip._cuda",
            sources=[
                "cuda/neuzip.cu",
            ],
            include_dirs=[
                os.path.dirname(os.path.abspath(__file__)),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda"),
            ],
            libraries=["nvcomp"],
            extra_compile_args=[
                "-w",
                "-O3",
                f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
            ],
        ),
    ],
    version=NEUZIP_VERSION,
    cmdclass={"build_ext": BuildExtension},
)
