# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch


class Parameter(torch.nn.Parameter):
    def __new__(cls, manager, data, requires_grad=True) -> "Parameter":
        p = super().__new__(cls, torch.zeros(0).to(data), requires_grad)

        p._handle = manager.write(data)
        p._gradient = None
        p._manager = manager
        p._shape = data.shape

        del data
        return p

    @torch.no_grad()
    def __call__(self) -> torch.Tensor:
        return self._manager.read(self._handle).reshape(self._shape).to(self.device).requires_grad_(False)

    def linear(self, inputs, bias=None):
        return self._manager.linear(self._handle, inputs, bias).to(self.device)
