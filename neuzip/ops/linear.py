# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch


class CompressedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, _p, bias):
        # manual cast
        ctx._inputs_dtype = inputs.dtype
        if torch.is_autocast_enabled():
            inputs = inputs.to(torch.get_autocast_gpu_dtype())
        output = _p.linear(inputs, bias)
        ctx.save_for_backward(inputs, bias)
        ctx.carry = (_p,)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (inputs, bias) = ctx.saved_tensors
        (_p,) = ctx.carry

        inputs_shape = inputs.shape

        inputs = inputs.reshape(-1, inputs.shape[-1])
        grad_output = grad_output.reshape(-1, grad_output.shape[-1])
        _weight = _p()

        grad_inputs = torch.mm(grad_output, _weight).reshape(inputs_shape).to(ctx._inputs_dtype)
        _weight = torch.mm(grad_output.t(), inputs)

        if _p._gradient is not None:
            _p._gradient.add_(_weight)
        else:
            _p._gradient = _weight

        return (
            grad_inputs,
            torch.empty_like(_p.data),
            grad_output.sum(0) if bias is not None else None,
        )
