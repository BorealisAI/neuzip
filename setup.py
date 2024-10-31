# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

import torch
from setuptools import find_packages, setup
from torch.torch_version import Version
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NEUZIP_VERSION = "0.1.1"

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
    install_requires=[
        "torch",
        "transformers",
        "numpy",
        f"nvidia-nvcomp-cu{Version(torch.version.cuda).major}",
    ],
    packages=find_packages(),
    package_data={
        "neuzip._cuda": ["*.cuh", "*.cu"],
    },
    include_package_data=True,
    url="https://github.com/BorealisAI/neuzip",
    license="CC BY-NC-SA 4.0",
    version=NEUZIP_VERSION,
    cmdclass={"build_ext": BuildExtension},
)
