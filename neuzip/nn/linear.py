# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch

from neuzip.nn.module import Parameter
from neuzip.ops import CompressedLinear


class Linear(torch.nn.Module):
    def __init__(self, manager, linear: torch.nn.Linear) -> None:
        super().__init__()
        self.manager = manager
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self._p = Parameter(self.manager, linear.weight.data)

        if linear.bias is not None:
            self.bias = torch.nn.Parameter(linear.bias.data)
        else:
            self.register_parameter("bias", None)

    def extra_repr(self) -> str:
        return f"! in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"

    def forward(self, x):
        return CompressedLinear.apply(x, self._p, self.bias)

    @property
    def weight(self):
        return self._p
