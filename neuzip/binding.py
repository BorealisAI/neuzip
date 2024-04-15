# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import enum
from typing import Optional

import torch

from neuzip.utils import _convert

try:
    import neuzip._cuda
except ImportError:
    raise ImportError(
        "The CUDA extension is not built. Please run `python setup.py install`"
    )


class Algorithm(enum.Enum):
    ans = neuzip._cuda.Algorithm.ans
    gdeflate = neuzip._cuda.Algorithm.gdeflate
    zstd = neuzip._cuda.Algorithm.zstd
    dummy = "dummy"


@torch.no_grad()
def _dummy_linear(weights, inputs: torch.Tensor, bias=None) -> torch.Tensor:
    return torch.nn.functional.linear(inputs, weights, bias)


class DummyBackend:
    def write(self, handle: str, tensor: torch.Tensor) -> None:
        cuda_tensor = tensor.cuda()
        del tensor
        setattr(self, handle, cuda_tensor)

    def read(self, handle: str) -> torch.Tensor:
        return getattr(self, handle)

    def size(self, handle: str) -> int:
        result = getattr(self, handle)
        return result.numel() * result.element_size()

    def linear(
        self, handle: str, inputs: torch.Tensor, shape: torch.Size, bias=None
    ) -> torch.Tensor:
        weights = self.read(handle).reshape(shape)
        return _dummy_linear(weights, inputs, bias)


class Manager:
    def __init__(
        self,
        algorithm: Algorithm = Algorithm.ans,
        precision: int = 7,
        normalizer_size=None,
        chunk_size=2**16,
    ) -> None:
        if normalizer_size is None:
            normalizer_size = 0

        if algorithm == Algorithm.dummy:
            self._be = DummyBackend()
        else:
            module_name = f"Manager_f{precision}_n{normalizer_size}"
            self._be = getattr(neuzip._cuda, module_name)(algorithm.value, chunk_size)

        self.meta_dict = {}

    def write(self, tensor: torch.Tensor, handle: Optional[str] = None) -> str:
        if handle is None:
            handle = str(len(self.meta_dict))

        oshape = tensor.shape

        self.meta_dict[handle] = oshape

        self._be.write(handle, tensor)

        del tensor

        return handle

    def read(self, handle: str) -> torch.Tensor:
        shape = self.meta_dict[handle]
        tensor = self._be.read(handle).requires_grad_(False)
        tensor = tensor.reshape(shape)
        return tensor

    def size(self, handle: str) -> int:
        return self._be.size(handle)

    def linear(self, handle: str, inputs: torch.Tensor, bias=None) -> torch.Tensor:
        shape = self.meta_dict[handle]
        if bias is not None:
            return self._be.linear(handle, inputs, shape, bias)
        else:
            return self._be.linear_without_bias(handle, inputs, shape)

    # additional methods
    def convert(self, module: torch.nn.Module) -> torch.nn.Module:
        return _convert(self, module)

    def split(self, tensor: torch.Tensor) -> str:
        shape = tensor.shape
        exp, frac = self._be.split(tensor)
        return exp.reshape(shape), frac.reshape(shape)

    def __del__(self):
        del self._be
        del self.meta_dict
