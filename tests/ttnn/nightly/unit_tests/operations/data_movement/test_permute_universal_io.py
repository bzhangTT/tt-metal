# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Tests for permute universal input/output support.

Validates that ttnn::permute works correctly with all memory configurations:
- L1 interleaved, DRAM interleaved
- Height-sharded, Width-sharded, Block-sharded

Also validates that transpose NH/NW/CW (which delegate to permute) work with
sharded inputs, since fixing permute fixes those transpose paths.
"""

import pytest
import torch

import ttnn

from loguru import logger
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_equal


def run_permute_test(
    shape,
    dims,
    device,
    input_layout=ttnn.TILE_LAYOUT,
    input_mem_config=None,
    output_mem_config=None,
    dtype=ttnn.bfloat16,
):
    """Helper to run a single permute test with the given configs."""
    torch.manual_seed(12345)
    x = torch.rand(shape).bfloat16().float()

    if input_mem_config is None:
        input_mem_config = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)

    ttnn_input = ttnn.from_torch(x, layout=input_layout, dtype=dtype, device=device, memory_config=input_mem_config)
    result = ttnn.permute(ttnn_input, dims, memory_config=output_mem_config)

    ref = x.permute(dims)
    got = ttnn.to_torch(result.cpu().to(ttnn.ROW_MAJOR_LAYOUT))

    passing, output = comp_equal(ref, got)
    logger.info(output)
    assert passing, f"Permute mismatch for shape={shape}, dims={dims}"


def _height_shard_config(shape, device, num_cores=4):
    """Create a height-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_height = 1
    for d in shape[:-1]:
        total_height *= d
    shard_shape = (total_height // num_cores, shape[-1])
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec)


def _width_shard_config(shape, device, num_cores=4):
    """Create a width-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    num_cores = min(num_cores, compute_grid.x * compute_grid.y)
    shard_grid = ttnn.num_cores_to_corerangeset(num_cores, compute_grid, True)
    total_height = 1
    for d in shape[:-1]:
        total_height *= d
    shard_shape = (total_height, shape[-1] // num_cores)
    shard_spec = ttnn.ShardSpec(shard_grid, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, shard_spec)


def _block_shard_config(shape, device):
    """Create a 2x2 block-sharded MemoryConfig for the given 4D shape."""
    compute_grid = device.compute_with_storage_grid_size()
    grid_x = min(2, compute_grid.x)
    grid_y = min(2, compute_grid.y)
    total_height = 1
    for d in shape[:-1]:
        total_height *= d
    shard_shape = (total_height // grid_y, shape[-1] // grid_x)
    shard_spec = ttnn.ShardSpec(
        ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, grid_y - 1))}),
        shard_shape,
        ttnn.ShardOrientation.ROW_MAJOR,
    )
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1, shard_spec)


# ──────────────────────────────────────────────────────────────
# 1. Interleaved baseline (regression) — existing paths preserved
# ──────────────────────────────────────────────────────────────


class TestPermuteInterleaved:
    """Interleaved input — must not regress."""

    def test_identity(self, device):
        run_permute_test((2, 4, 32, 64), (0, 1, 2, 3), device)

    def test_wh(self, device):
        run_permute_test((2, 4, 32, 64), (0, 1, 3, 2), device)

    def test_hc(self, device):
        run_permute_test((2, 4, 32, 64), (0, 2, 1, 3), device)

    def test_cn(self, device):
        run_permute_test((2, 4, 32, 64), (1, 0, 2, 3), device)

    def test_non_decomposable_nhcw(self, device):
        """(2,0,1,3) — not decomposable, goes through prim::permute."""
        run_permute_test((2, 4, 32, 64), (2, 0, 1, 3), device)

    def test_non_decomposable_whcn(self, device):
        """(3,2,1,0) — full reverse, goes through prim::permute."""
        run_permute_test((2, 4, 32, 64), (3, 2, 1, 0), device)

    def test_rm_identity(self, device):
        run_permute_test((2, 4, 32, 64), (0, 1, 2, 3), device, input_layout=ttnn.ROW_MAJOR_LAYOUT)

    def test_rm_wh(self, device):
        run_permute_test((2, 4, 32, 64), (0, 1, 3, 2), device, input_layout=ttnn.ROW_MAJOR_LAYOUT)

    def test_rm_non_decomposable(self, device):
        run_permute_test((2, 4, 32, 64), (2, 0, 1, 3), device, input_layout=ttnn.ROW_MAJOR_LAYOUT)


# ──────────────────────────────────────────────────────────────
# 2. Sharded input — transpose-decomposable (N=0)
# ──────────────────────────────────────────────────────────────


class TestPermuteShardedDecomposable:
    """Sharded inputs where N=0, decomposed into transpose chains."""

    def test_wh_height_sharded(self, device):
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (0, 1, 3, 2), device, input_mem_config=mem)

    def test_hc_height_sharded(self, device):
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (0, 2, 1, 3), device, input_mem_config=mem)

    def test_0231_height_sharded(self, device):
        """(0,2,3,1) = transpose_wh(transpose_hc(...))"""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (0, 2, 3, 1), device, input_mem_config=mem)

    def test_0312_height_sharded(self, device):
        """(0,3,1,2) = transpose_hc(transpose_wh(...))"""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (0, 3, 1, 2), device, input_mem_config=mem)

    def test_0321_height_sharded(self, device):
        """(0,3,2,1) = transpose_wh(transpose_hc(transpose_wh(...)))"""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (0, 3, 2, 1), device, input_mem_config=mem)

    def test_cn_height_sharded(self, device):
        """(1,0,2,3) CN — NEW in sharded branch."""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (1, 0, 2, 3), device, input_mem_config=mem)


# ──────────────────────────────────────────────────────────────
# 3. Sharded input — non-decomposable (direct sharded read)
# ──────────────────────────────────────────────────────────────


class TestPermuteShardedFallback:
    """Sharded inputs where permutation cannot be decomposed into transpose chains.
    These go through prim::permute with L1 interleaved output."""

    def test_2013_height_sharded(self, device):
        """(2,0,1,3) — N moves, no decomp."""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (2, 0, 1, 3), device, input_mem_config=mem)

    def test_3210_height_sharded(self, device):
        """(3,2,1,0) — full reverse."""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (3, 2, 1, 0), device, input_mem_config=mem)

    def test_1203_height_sharded(self, device):
        """(1,2,0,3) — N and C swap with H."""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (1, 2, 0, 3), device, input_mem_config=mem)


# ──────────────────────────────────────────────────────────────
# 4. Transpose-delegated perms (fixes transpose NH/NW/CW)
# ──────────────────────────────────────────────────────────────


class TestPermuteTransposeDelegated:
    """These permutations are what ttnn::transpose delegates to permute for
    NH, NW, CW dims. Fixing permute fixes these transpose paths."""

    def test_nh_height_sharded(self, device):
        """NH: transpose(a, 0, 2) → permute(a, {2,1,0,3})"""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (2, 1, 0, 3), device, input_mem_config=mem)

    def test_nw_height_sharded(self, device):
        """NW: transpose(a, 0, 3) → permute(a, {3,1,2,0})"""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (3, 1, 2, 0), device, input_mem_config=mem)

    def test_cw_height_sharded(self, device):
        """CW: transpose(a, 1, 3) → permute(a, {0,3,2,1})"""
        shape = (2, 4, 32, 64)
        mem = _height_shard_config(shape, device)
        run_permute_test(shape, (0, 3, 2, 1), device, input_mem_config=mem)


# ──────────────────────────────────────────────────────────────
# 5. Block-sharded input
# ──────────────────────────────────────────────────────────────


class TestPermuteBlockSharded:
    """Block-sharded inputs — decomposable perms go through transpose chains
    (which fall back to L1 interleaved), non-decomposable go through factory."""

    def test_wh_block_sharded(self, device):
        shape = (2, 2, 64, 64)
        mem = _block_shard_config(shape, device)
        run_permute_test(shape, (0, 1, 3, 2), device, input_mem_config=mem)

    def test_non_decomposable_block_sharded(self, device):
        shape = (2, 2, 64, 64)
        mem = _block_shard_config(shape, device)
        run_permute_test(shape, (2, 0, 1, 3), device, input_mem_config=mem)


# ──────────────────────────────────────────────────────────────
# 6. Width-sharded input
# ──────────────────────────────────────────────────────────────


class TestPermuteWidthSharded:
    """Width-sharded inputs."""

    def test_wh_width_sharded(self, device):
        shape = (2, 4, 32, 128)
        mem = _width_shard_config(shape, device)
        run_permute_test(shape, (0, 1, 3, 2), device, input_mem_config=mem)

    def test_non_decomposable_width_sharded(self, device):
        shape = (2, 4, 32, 128)
        mem = _width_shard_config(shape, device)
        run_permute_test(shape, (2, 0, 1, 3), device, input_mem_config=mem)


# ──────────────────────────────────────────────────────────────
# 7. Cross-memory-config
# ──────────────────────────────────────────────────────────────


class TestPermuteCrossConfig:
    """Input and output in different memory configurations."""

    def test_height_sharded_to_l1_interleaved(self, device):
        shape = (2, 4, 32, 64)
        in_mem = _height_shard_config(shape, device)
        out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        run_permute_test(shape, (2, 0, 1, 3), device, input_mem_config=in_mem, output_mem_config=out_mem)

    def test_l1_interleaved_to_dram(self, device):
        shape = (2, 4, 32, 64)
        in_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        out_mem = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        run_permute_test(shape, (0, 1, 3, 2), device, input_mem_config=in_mem, output_mem_config=out_mem)


# ──────────────────────────────────────────────────────────────
# 8. Rank > 4
# ──────────────────────────────────────────────────────────────


class TestPermuteRankGt4:
    """Rank > 4 — validates transpose_nd path as well."""

    def test_5d_interleaved(self, device):
        run_permute_test((2, 2, 2, 32, 64), (0, 2, 1, 4, 3), device)

    def test_5d_interleaved_rm(self, device):
        run_permute_test((2, 2, 2, 32, 64), (0, 2, 1, 3, 4), device, input_layout=ttnn.ROW_MAJOR_LAYOUT)

    def test_5d_height_sharded(self, device):
        shape = (2, 2, 2, 32, 64)
        in_mem = _height_shard_config(shape, device, num_cores=8)
        run_permute_test(shape, (0, 2, 1, 4, 3), device, input_mem_config=in_mem)

    def test_5d_height_sharded_rm(self, device):
        shape = (2, 2, 2, 32, 64)
        in_mem = _height_shard_config(shape, device, num_cores=8)
        run_permute_test(shape, (0, 2, 1, 3, 4), device, input_layout=ttnn.ROW_MAJOR_LAYOUT, input_mem_config=in_mem)
