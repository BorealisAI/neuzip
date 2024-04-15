# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import logging

import torch
import transformers

from neuzip.nn import Embedding, Linear

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _set_module_by_name(parent_module, name, child_module):
    module_names = name.split(".")
    if len(module_names) == 1:
        setattr(parent_module, name, child_module)
    else:
        next_module = parent_module
        for idx in range(len(module_names) - 1):
            next_module_name = module_names[idx]
            if next_module_name.isnumeric():
                next_module = next_module[int(next_module_name)]
            else:
                next_module = getattr(next_module, next_module_name)
        setattr(next_module, module_names[-1], child_module)


def _delete_module(module):
    for _name, _param in module.named_parameters():
        # Save device memory by clearing parameters
        setattr(module, _name, None)
        del _param


def _unroll_convert(manager, module):
    # Quantization happens in-place
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Linear):
            full_name = f"{name}"
            logger.debug(f"Quantizing {full_name}: {m}")
            _set_module_by_name(module, name, Linear(manager, m))
            _delete_module(m)
        elif isinstance(m, torch.nn.Embedding):
            full_name = f"{name}"
            logger.debug(f"Quantizing {full_name}: {m}")
            _set_module_by_name(module, name, Embedding(manager, m))
            _delete_module(m)
    return module


def _convert(manager, module, prefix=""):
    for name, child in module.named_children():
        if "lm_head" in name:
            pass
        elif isinstance(child, (torch.nn.Linear, transformers.pytorch_utils.Conv1D)):
            full_name = f"{prefix}.{name}" if prefix else name
            logger.debug(f"Converting {full_name}: {child}")
            setattr(module, name, Linear(manager, child))
            _delete_module(child)
        else:
            _convert(manager, child, f"{prefix}.{name}" if prefix else name)
    return module
