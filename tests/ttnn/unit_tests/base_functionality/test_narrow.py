# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_equal, assert_with_pcc, tt_dtype_to_torch_dtype


@pytest.mark.parametrize(
    "dtype",
    [ttnn.uint8, ttnn.uint16, ttnn.int32, ttnn.uint32, ttnn.float32, ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
)
@pytest.mark.parametrize(
    "input_shape, dim, start, length, memory_config, layout",
    [
        ((256, 32, 17, 32), 0, 168, 16, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM), ttnn.TILE_LAYOUT),
        ((1, 32, 168, 16), 1, 5, 8, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM), ttnn.ROW_MAJOR_LAYOUT),
        (
            (1, 8, 64, 128),
            1,
            4,
            4,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                    (32, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
        (
            (1, 8, 64, 128),
            3,
            32,
            32,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                    (32, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
        (
            (1, 8, 128, 128),
            2,
            96,
            32,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                    (64, 128),
                    ttnn.ShardOrientation.COL_MAJOR,
                ),
            ),
            ttnn.ROW_MAJOR_LAYOUT,
        ),
        (
            (1, 1, 32, 576),
            -1,
            512,
            64,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))}),
                    (32, 64),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
        (
            (1, 1, 8, 576),
            3,
            512,
            64,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 3))}),
                    (8, 64),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.ROW_MAJOR_LAYOUT,
        ),
        (
            (1, 32, 16, 192),
            3,
            128,
            64,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.BLOCK_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(6, 7))}),
                    (128, 32),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
    ],
    ids=[
        "dram_dim0_tile",
        "dram_dim1_rm",
        "l1_height_sharded_dim1_tile",
        "l1_height_sharded_dim3_tile",
        "l1_height_sharded_dim2_rm",
        "l1_width_sharded_dim3_tile",
        "l1_width_sharded_dim3_rm",
        "l1_block_sharded_dim3_tile",
    ],
)
def test_narrow(input_shape, dim, start, length, memory_config, layout, dtype, device):
    if (dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b) and layout == ttnn.ROW_MAJOR_LAYOUT:
        pytest.skip("Skipping test for bfloat8_b or bfloat4_b with ROW_MAJOR_LAYOUT")

    if dtype in [ttnn.uint8, ttnn.uint16, ttnn.int32, ttnn.uint32]:
        torch_input_tensor = torch.randint(0, 128, input_shape, dtype=tt_dtype_to_torch_dtype[dtype])
    else:
        torch_input_tensor = torch.randn(input_shape, dtype=tt_dtype_to_torch_dtype[dtype])
    torch_result = torch.narrow(torch_input_tensor, dim, start, length)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, dtype=dtype, device=device, memory_config=memory_config
    )
    ttnn_output = ttnn.narrow(input_tensor, dim, start, length)

    assert layout == ttnn_output.layout
    assert memory_config.buffer_type == ttnn_output.memory_config().buffer_type
    assert memory_config.memory_layout == ttnn_output.memory_config().memory_layout
    output = ttnn.to_torch(ttnn_output)
    if dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b:
        target_pcc = 0.95 if dtype == ttnn.bfloat4_b else 0.99
        assert_with_pcc(torch_result, output, target_pcc)
    else:
        assert_equal(torch_result, output)


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b, ttnn.bfloat4_b, ttnn.bfloat16],
)
@pytest.mark.parametrize(
    "input_shape, dim, start, length",
    [
        # input_tensor_shape[dim] is not an integer multiple of length, and
        # the truncated reduction_factor doesn't divide the source tile count
        # cleanly — exactly the case where the old size = src_size / RF was
        # not a multiple of the tile page size, tripping Buffer size validation.
        # Small triggering case: 160/64 = 2.5, src_tiles=5, 5 % 2 = 1.
        ((1, 1, 160, 32), 2, 0, 64),
        # Mirrors the MoE routed-expert chunking ratio: 6400/2048 = 3.125,
        # 50 tiles % 6 = 2 (analogous to 6400/1024 in test_ttnn_moe).
        ((1, 1, 6400, 32), 2, 0, 1024),
        # Same MoE-style ratio with a non-zero, bank-aligned start
        # (start_page_id = 768*32/1024 = 24, multiple of 8 and 12).
        ((1, 1, 1600, 32), 2, 768, 256),
    ],
    ids=[
        "dram_dim2_tile_nondivisible_small",
        "dram_dim2_tile_moe_ratio_start0",
        "dram_dim2_tile_moe_ratio_mid",
    ],
)
def test_narrow_dram_tile_nondivisible(input_shape, dim, start, length, dtype, device):
    """Regression test: narrow on TILE/DRAM-interleaved must succeed when the
    source dim along which we narrow isn't a multiple of length. Previously
    the buffer size was computed as src_size / (src_dim / length), where the
    inner integer division silently truncated, leaving the new size off by a
    non-tile-aligned remainder."""
    memory_config = ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM)
    layout = ttnn.TILE_LAYOUT

    torch_input_tensor = torch.randn(input_shape, dtype=tt_dtype_to_torch_dtype[dtype])
    torch_result = torch.narrow(torch_input_tensor, dim, start, length)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, dtype=dtype, device=device, memory_config=memory_config
    )
    ttnn_output = ttnn.narrow(input_tensor, dim, start, length)

    assert layout == ttnn_output.layout
    assert memory_config.buffer_type == ttnn_output.memory_config().buffer_type
    assert memory_config.memory_layout == ttnn_output.memory_config().memory_layout
    output = ttnn.to_torch(ttnn_output)
    if dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b:
        target_pcc = 0.95 if dtype == ttnn.bfloat4_b else 0.99
        assert_with_pcc(torch_result, output, target_pcc)
    else:
        assert_equal(torch_result, output)


@pytest.mark.parametrize(
    "input_shape, dim, start, length, memory_config, layout",
    [
        (
            (8, 4, 128, 128),
            3,
            32,
            32,
            ttnn.MemoryConfig(
                buffer_type=ttnn.BufferType.L1,
                memory_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                shard_spec=ttnn.ShardSpec(
                    ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 3))}),
                    (32, 128),
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            ),
            ttnn.TILE_LAYOUT,
        ),
    ],
    ids=["l1_height_sharded"],
)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_narrow_mesh(input_shape, dim, start, length, memory_config, layout, mesh_device):
    torch_input_tensor = torch.randn(input_shape, dtype=torch.bfloat16)
    torch_result = torch.narrow(torch_input_tensor, dim, start, length)
    mesh_config = ttnn.MeshMapperConfig([ttnn.PlacementShard(0), ttnn.PlacementShard(1)], mesh_device.shape)

    input_tensor_mesh = ttnn.from_torch(
        torch_input_tensor,
        device=mesh_device,
        layout=layout,
        dtype=ttnn.bfloat16,
        memory_config=memory_config,
        mesh_mapper=ttnn.create_mesh_mapper(mesh_device, mesh_config),
    )
    ttnn_output = ttnn.narrow(input_tensor_mesh, dim, start, length)

    output = ttnn.to_torch(
        ttnn_output, mesh_composer=ttnn.create_mesh_composer(mesh_device, ttnn.MeshComposerConfig(dims=[0, 1]))
    )
    assert_equal(torch_result, output)


@pytest.mark.parametrize(
    "dtype",
    [ttnn.bfloat8_b],
)
@pytest.mark.parametrize(
    "input_shape, dim, start, length, memory_config, layout",
    [((14336, 7168), 0, 0, 96, ttnn.MemoryConfig(buffer_type=ttnn.BufferType.DRAM), ttnn.TILE_LAYOUT)],
)
def test_narrow_regression(input_shape, dim, start, length, memory_config, layout, dtype, device):
    torch_input_tensor = torch.randn(input_shape, dtype=tt_dtype_to_torch_dtype[dtype])
    torch_result = torch.narrow(torch_input_tensor, dim, start, length)

    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=layout, dtype=dtype, device=device, memory_config=memory_config
    )
    ttnn_output = ttnn.narrow(input_tensor, dim, start, length)

    assert layout == ttnn_output.layout
    assert memory_config.buffer_type == ttnn_output.memory_config().buffer_type
    assert memory_config.memory_layout == ttnn_output.memory_config().memory_layout
    output = ttnn.to_torch(ttnn_output)
    assert_with_pcc(torch_result, output, 0.99)
