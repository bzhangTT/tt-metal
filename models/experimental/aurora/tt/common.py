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


def num_devices(device) -> int:
    """Number of physical devices behind a (possibly mesh) device handle."""
    return getattr(device, "get_num_devices", lambda: 1)()


def to_tt(x: torch.Tensor, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    """Move a torch tensor onto the (mesh) device in tile layout.

    On a multi-chip mesh, tensors are replicated across all chips by default
    (``mesh_mapper=None``); pass an explicit ``ttnn.ShardTensorToMesh`` to shard
    a weight for tensor parallelism instead.
    """
    if mesh_mapper is None and num_devices(device) > 1:
        mesh_mapper = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(x, dtype=dtype, layout=layout, device=device, mesh_mapper=mesh_mapper)


def from_tt(x) -> torch.Tensor:
    """Bring a TT-NN tensor back to host. For a replicated mesh tensor, return a
    single replica (all are identical); for a single device, the tensor itself."""
    nd = num_devices(x.device()) if hasattr(x, "device") else 1
    if nd > 1:
        full = ttnn.to_torch(x, mesh_composer=ttnn.ConcatMeshToTensor(x.device(), dim=0))
        return full[: full.shape[0] // nd]
    return ttnn.to_torch(x)


class TtLinear:
    """A ``torch.nn.Linear``-equivalent backed by ``ttnn.linear``.

    ``torch`` stores the weight as ``(out, in)`` and computes ``x @ W^T + b``.
    ``ttnn.linear`` computes ``x @ W + b`` so we pre-transpose the weight once,
    at construction time, when uploading it to the device.
    """

    def __init__(
        self,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        device,
        weight_dtype=ttnn.bfloat16,
        bias_dtype=ttnn.bfloat16,
        activation=None,
        tp=None,
    ):
        """tp: tensor-parallel mode across the mesh.
        None  -> weight replicated on every chip (default).
        "col" -> column-parallel: shard the output dim; output is mesh-sharded
                 on its last dim (feeds a "row" linear with no gather).
        "row" -> row-parallel: shard the input dim; each chip computes a partial
                 sum which is combined with ttnn.all_reduce, then bias is added.
        """
        self.out_features, self.in_features = weight.shape
        self.device = device
        self.nd = num_devices(device)
        self.tp = tp if self.nd > 1 else None
        wt = weight.t().contiguous()  # (in, out) for ttnn.linear
        # ``bfloat8_b`` (block float8) halves weight bandwidth on the big matmuls.
        if self.tp == "col":
            self.weight = to_tt(wt, device, dtype=weight_dtype, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1))
            self.bias = (
                to_tt(bias.reshape(1, -1), device, dtype=bias_dtype, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=1))
                if bias is not None
                else None
            )
        elif self.tp == "row":
            self.weight = to_tt(wt, device, dtype=weight_dtype, mesh_mapper=ttnn.ShardTensorToMesh(device, dim=0))
            # bias is replicated and added once, after the all_reduce.
            self.bias = to_tt(bias.reshape(1, -1), device, dtype=bias_dtype) if bias is not None else None
        else:
            self.weight = to_tt(wt, device, dtype=weight_dtype)
            self.bias = to_tt(bias.reshape(1, -1), device, dtype=bias_dtype) if bias is not None else None
        self.activation = activation

    def __call__(self, x, compute_kernel_config=None, core_grid=None):
        if self.tp == "row":
            y = ttnn.linear(
                x,
                self.weight,
                bias=None,
                activation=self.activation,
                compute_kernel_config=compute_kernel_config,
                core_grid=core_grid,
            )
            y = ttnn.all_reduce(y)  # sum partial products across the mesh
            if self.bias is not None:
                y = ttnn.add(y, self.bias)
            return y
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
