# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Common TT-NN helpers for the Aurora port.

These utilities convert PyTorch weights from a real Aurora checkpoint into
TT-NN tensors and provide thin wrappers around ``ttnn`` primitives that match
the numerics of the reference ``torch`` implementation.
"""

from __future__ import annotations

import numpy as np
import torch

import ttnn


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation between two tensors, flattened. Used in tests/demo."""
    a = a.detach().flatten().float().numpy()
    b = b.detach().flatten().float().numpy()
    if np.allclose(a, b):
        return 1.0
    return float(np.corrcoef(a, b)[0, 1])


def to_tt(x: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    """Move a torch tensor onto the mesh device in tile layout."""
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=device)


def from_tt(x) -> torch.Tensor:
    """Bring a TT-NN tensor back to host as a torch tensor."""
    return ttnn.to_torch(x)


class TtLinear:
    """A ``torch.nn.Linear``-equivalent backed by ``ttnn.linear``.

    ``torch`` stores the weight as ``(out, in)`` and computes ``x @ W^T + b``.
    ``ttnn.linear`` computes ``x @ W + b`` so we pre-transpose the weight once,
    at construction time, when uploading it to the device.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None, device,
                 weight_dtype=ttnn.bfloat16, bias_dtype=ttnn.bfloat16, activation=None):
        self.out_features, self.in_features = weight.shape
        # ``bfloat8_b`` (block float8) halves weight bandwidth on the big matmuls.
        self.weight = to_tt(weight.t().contiguous(), device, dtype=weight_dtype)
        if bias is not None:
            self.bias = to_tt(bias.reshape(1, -1), device, dtype=bias_dtype)
        else:
            self.bias = None
        self.activation = activation

    def __call__(self, x, compute_kernel_config=None, core_grid=None):
        return ttnn.linear(
            x,
            self.weight,
            bias=self.bias,
            activation=self.activation,
            compute_kernel_config=compute_kernel_config,
            core_grid=core_grid,
        )


def layer_norm(x, weight=None, bias=None, eps: float = 1e-5):
    """LayerNorm over the last dim using ttnn, matching ``torch.nn.LayerNorm``."""
    return ttnn.layer_norm(x, weight=weight, bias=bias, epsilon=eps)
