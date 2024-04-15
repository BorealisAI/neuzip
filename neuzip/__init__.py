# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch

from neuzip import nn, ops
from neuzip.binding import Algorithm, Manager

__all__ = ["ops", "nn", "Algorithm", "Manager"]


def lomo_hook(scheduler, param):
    if param.grad is None:
        return
    learning_rate = scheduler.get_last_lr()[0]
    with torch.no_grad():
        param.data.add_(param.grad, alpha=-learning_rate)
    del param.grad


def compressed_lomo_hook(scheduler, param):
    if param._gradient is None:
        return
    learning_rate = scheduler.get_last_lr()[0]
    with torch.no_grad():
        weight = param()
        weight.add_(param._gradient, alpha=-learning_rate)
        param._manager.write(weight, param._handle)
    del param._gradient
    del param.grad
    param._gradient = None
