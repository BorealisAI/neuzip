# Copyright (c) 2024-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch

from neuzip.nn.module import Parameter


class Embedding(torch.nn.Module):
    def __init__(self, manager, embedding: torch.nn.Embedding) -> None:
        super().__init__()
        self.manager = manager
        self.num_embeddings = embedding.num_embeddings
        self.embedding_dim = embedding.embedding_dim
        self.padding_idx = embedding.padding_idx
        self.max_norm = embedding.max_norm
        self.norm_type = embedding.norm_type
        self.scale_grad_by_freq = embedding.scale_grad_by_freq
        self.sparse = embedding.sparse

        self._p = Parameter(self.manager, embedding.weight)

    def extra_repr(self) -> str:
        return f"! num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}, max_norm={self.max_norm}, norm_type={self.norm_type}, scale_grad_by_freq={self.scale_grad_by_freq}, sparse={self.sparse}"  # noqa

    def forward(self, x):
        return torch.nn.functional.embedding(
            x,
            self._p(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
