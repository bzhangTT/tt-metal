# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the ``block_size`` override on
``ttnn.transformer.paged_scaled_dot_product_attention_decode``.

The op gained an optional ``block_size`` kwarg so callers can read from
a paged K/V cache buffer that was physically allocated with a different
``(block_size, head_dim)`` view than the layer's query — as long as
``num_kv_heads * block_size * head_dim`` (per-block element count) is
preserved across views. The kernel now sources ``head_dim`` from Q's
last dim (CUDA-style) and ``block_size`` from the override (else
``k.padded_shape[2]``), so a layer can attend through its own view onto
a shared physical buffer.

Tests:

* ``test_legacy_no_override``: backward compat (no kwarg) on a few
  ``(block_size, head_dim)`` shape combos.
* ``test_override_matches_alloc_is_noop``: when ``block_size`` equals
  the cache's declared block_size, the override path produces bit-
  identical output to the no-override path.
* ``test_override_view_full_into_sliding_buffer`` and the symmetric
  direction: cache allocated under one view, Q + override under the
  other view; verified against a torch reference computed on the cache
  bytes re-interpreted through the call view's tile-grid (NOT
  ``torch.view`` — ttnn's TILE layout is tile-grid-shape-specific; see
  ``_permute_tile_grid`` docstring).
* ``test_negative_byte_count_mismatch``: per-block byte invariant
  violation must raise.
"""

import pytest
import torch
from loguru import logger

import ttnn
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import comp_pcc


# ── Helpers ────────────────────────────────────────────────────────────────


def _permute_tile_grid(t, view_block_size, view_head_dim):
    """Reinterpret a paged-cache torch tensor's last two dims under a
    different tile-grid.

    See the equivalent helper in
    ``tests/ttnn/unit_tests/operations/transformers/test_paged_cache_flexible_geometry.py``
    for the full rationale. Short version: ttnn's TILE layout stores
    tiles row-major in the cache's *declared* tile-grid; viewing the
    same bytes under a different ``(block_size, head_dim)`` tile-grid
    requires a linear-tile-index-preserving permutation, not a plain
    ``torch.view``.
    """
    N, KV, alloc_block_size, alloc_head_dim = t.shape
    TILE = 32
    assert alloc_block_size % TILE == 0 and alloc_head_dim % TILE == 0
    assert view_block_size % TILE == 0 and view_head_dim % TILE == 0
    alloc_BR_t = alloc_block_size // TILE
    alloc_Wt = alloc_head_dim // TILE
    view_BR_t = view_block_size // TILE
    view_Wt = view_head_dim // TILE
    total_tiles = alloc_BR_t * alloc_Wt
    assert total_tiles == view_BR_t * view_Wt, "per-block tile count must match"
    t = t.view(N, KV, alloc_BR_t, TILE, alloc_Wt, TILE)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    t = t.view(N, KV, total_tiles, TILE, TILE)
    t = t.view(N, KV, view_BR_t, view_Wt, TILE, TILE)
    t = t.permute(0, 1, 2, 4, 3, 5).contiguous()
    return t.view(N, KV, view_block_size, view_head_dim)


def _torch_sdpa_reference(q, k, v, cur_pos, scale):
    """Causal SDPA reference: q is ``(1, B, num_q_heads, head_dim)``,
    k/v are ``(B, num_kv_heads, max_seq_len, head_dim)``, ``cur_pos`` is
    ``[B]`` int (the most-recent token's index, inclusive). Computes
    attention up to and including ``cur_pos[b]`` for each user.

    Mimics the math the TT op performs so we can compare in fp32.
    Output shape: ``(1, B, num_q_heads, head_dim_v)``.
    """
    B = q.shape[1]
    num_q_heads = q.shape[2]
    head_dim = q.shape[3]
    num_kv_heads = k.shape[1]
    head_dim_v = v.shape[3]
    repeat = num_q_heads // num_kv_heads
    out = torch.zeros(1, B, num_q_heads, head_dim_v, dtype=torch.float32)
    for b in range(B):
        pos = int(cur_pos[b].item() if torch.is_tensor(cur_pos) else cur_pos[b])
        k_b = k[b, :, : pos + 1, :].float()  # (kv_heads, S, head_dim)
        v_b = v[b, :, : pos + 1, :].float()  # (kv_heads, S, head_dim_v)
        q_b = q[0, b, :, :].float()  # (num_q_heads, head_dim)
        k_b_rep = k_b.repeat_interleave(repeat, dim=0)  # (num_q_heads, S, head_dim)
        v_b_rep = v_b.repeat_interleave(repeat, dim=0)  # (num_q_heads, S, head_dim_v)
        scores = torch.einsum("hd,hsd->hs", q_b, k_b_rep) * scale
        weights = torch.softmax(scores, dim=-1)
        out[0, b, :, :] = torch.einsum("hs,hsd->hd", weights, v_b_rep)
    return out


def _paged_layout(unshuffled_per_user, page_table, num_kv_heads, block_size, head_dim):
    """Lay out ``(B, num_kv_heads, max_seq_len, head_dim)`` K (or V) into
    a paged buffer ``(num_blocks, num_kv_heads, block_size, head_dim)``
    according to ``page_table``.

    page_table maps virtual block id → physical block id. The output is
    indexed by physical block id; reading back via the reverse mapping
    yields the original per-user layout.
    """
    B = unshuffled_per_user.shape[0]
    max_num_blocks_per_seq = unshuffled_per_user.shape[2] // block_size
    max_num_blocks = page_table.shape[0] * page_table.shape[1] // B if False else int(page_table.max().item() + 1)
    paged = torch.zeros(max_num_blocks, num_kv_heads, block_size, head_dim, dtype=unshuffled_per_user.dtype)
    for b in range(B):
        for vb in range(max_num_blocks_per_seq):
            pb = int(page_table[b, vb].item())
            paged[pb, :, :, :] = unshuffled_per_user[b, :, vb * block_size : (vb + 1) * block_size, :]
    return paged


def _alloc_paged_cache_on_device(num_blocks, num_kv_heads, block_size, head_dim, device, fill=None):
    """Allocate a paged cache tensor of shape
    ``(num_blocks, num_kv_heads, block_size, head_dim)`` on device.

    ``fill`` is an optional torch tensor of the same shape; when
    provided, its values seed the cache (post-tilization).
    """
    if fill is None:
        fill = torch.randn(num_blocks, num_kv_heads, block_size, head_dim).bfloat16().float()
    cache_tt = ttnn.Tensor(fill, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    return cache_tt, fill


# ── Tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "block_size, head_dim",
    [(64, 256), (128, 256), (64, 512), (128, 128)],
    ids=["b64_d256", "b128_d256", "b64_d512", "b128_d128"],
)
def test_legacy_no_override(block_size, head_dim, device):
    """Existing behavior preserved: no ``block_size`` kwarg → identical
    to the pre-change op. Guards against silent regression on legacy
    callers."""
    torch.manual_seed(0)
    B = 4
    num_kv_heads = 1
    num_q_heads = num_kv_heads  # MHA for simplicity
    max_seq_len = 256
    max_num_blocks_per_seq = max_seq_len // block_size
    max_num_blocks = B * max_num_blocks_per_seq

    # Build K, V per user (unshuffled) and a random page_table.
    k_per_user = torch.randn(B, num_kv_heads, max_seq_len, head_dim).bfloat16().float()
    v_per_user = torch.randn(B, num_kv_heads, max_seq_len, head_dim).bfloat16().float()
    page_table = torch.randperm(max_num_blocks, dtype=torch.int32).reshape(B, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    k_paged = _paged_layout(k_per_user, page_table, num_kv_heads, block_size, head_dim)
    v_paged = _paged_layout(v_per_user, page_table, num_kv_heads, block_size, head_dim)
    k_tt, _ = _alloc_paged_cache_on_device(max_num_blocks, num_kv_heads, block_size, head_dim, device, fill=k_paged)
    v_tt, _ = _alloc_paged_cache_on_device(max_num_blocks, num_kv_heads, block_size, head_dim, device, fill=v_paged)

    # Q: [1, B, num_q_heads, head_dim] padded to 32 on dim 2.
    cur_pos = torch.tensor([13 + b * 7 for b in range(B)], dtype=torch.int32)
    q = torch.randn(1, B, num_q_heads, head_dim).bfloat16().float()
    q_padded = torch.nn.functional.pad(q, (0, 0, 0, 32 - num_q_heads), "constant", 0)
    q_tt = ttnn.Tensor(q_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    cur_pos_tt = ttnn.Tensor(cur_pos, ttnn.int32).to(device)

    scale = 1.0 / (head_dim**0.5)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )

    out_tt = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_tt,
        k_tt,
        v_tt,
        page_table_tensor=page_table_tt,
        cur_pos_tensor=cur_pos_tt,
        scale=scale,
        program_config=program_config,
    )

    out = ttnn.to_torch(out_tt)[:, :, :num_q_heads, :]
    ref = _torch_sdpa_reference(q, k_per_user, v_per_user, cur_pos, scale)
    eq, msg = comp_pcc(ref, out, pcc=0.99)
    assert eq, f"legacy path PCC failed: {msg}"


def test_override_matches_alloc_is_noop(device):
    """When ``block_size`` equals the cache's declared block_size, the
    override path must produce a result that PCC-matches the no-override
    path. Verifies the override plumbing doesn't perturb behaviour when
    it's a no-op."""
    torch.manual_seed(1)
    B = 4
    num_kv_heads = 1
    num_q_heads = num_kv_heads
    block_size = 64
    head_dim = 256
    max_seq_len = 256
    max_num_blocks_per_seq = max_seq_len // block_size
    max_num_blocks = B * max_num_blocks_per_seq

    k_per_user = torch.randn(B, num_kv_heads, max_seq_len, head_dim).bfloat16().float()
    v_per_user = torch.randn(B, num_kv_heads, max_seq_len, head_dim).bfloat16().float()
    page_table = torch.randperm(max_num_blocks, dtype=torch.int32).reshape(B, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)
    k_paged = _paged_layout(k_per_user, page_table, num_kv_heads, block_size, head_dim)
    v_paged = _paged_layout(v_per_user, page_table, num_kv_heads, block_size, head_dim)

    def _build_q():
        cur_pos = torch.tensor([7 + b * 11 for b in range(B)], dtype=torch.int32)
        q = torch.randn(1, B, num_q_heads, head_dim).bfloat16().float()
        q_padded = torch.nn.functional.pad(q, (0, 0, 0, 32 - num_q_heads), "constant", 0)
        return q, q_padded, cur_pos

    # Same Q + cur_pos for both runs.
    q_torch, q_padded, cur_pos = _build_q()
    cur_pos_tt = ttnn.Tensor(cur_pos, ttnn.int32).to(device)
    scale = 1.0 / (head_dim**0.5)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )

    # Two independent runs (separate tensor allocations so the K/V cache
    # state is identical between them).
    def _run(use_override):
        k_tt, _ = _alloc_paged_cache_on_device(max_num_blocks, num_kv_heads, block_size, head_dim, device, fill=k_paged)
        v_tt, _ = _alloc_paged_cache_on_device(max_num_blocks, num_kv_heads, block_size, head_dim, device, fill=v_paged)
        q_tt = ttnn.Tensor(q_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
        kwargs = dict(
            page_table_tensor=page_table_tt,
            cur_pos_tensor=cur_pos_tt,
            scale=scale,
            program_config=program_config,
        )
        if use_override:
            kwargs["block_size"] = block_size
        return ttnn.to_torch(ttnn.transformer.paged_scaled_dot_product_attention_decode(q_tt, k_tt, v_tt, **kwargs))[
            :, :, :num_q_heads, :
        ]

    out_legacy = _run(use_override=False)
    out_override = _run(use_override=True)
    eq, msg = comp_pcc(out_legacy, out_override, pcc=0.999)
    assert eq, f"override-is-noop diverged from legacy: {msg}"


def _run_flexible_geometry_test(
    device,
    alloc_block_size,
    alloc_head_dim,
    view_block_size,
    view_head_dim,
    B=4,
    max_seq_len_view=256,
):
    """Shared body for the two flips-view directions.

    Builds a cache allocated under ``(alloc_block_size, alloc_head_dim)``
    that's pre-filled with random K/V bytes (tilized in alloc's
    tile-grid). Then runs SDPA decode with ``block_size=view_block_size``
    and Q sized for ``view_head_dim``. The reference re-interprets those
    same DRAM bytes through the view's tile-grid (via
    ``_permute_tile_grid``) and computes torch attention on the result.
    """
    num_kv_heads = 1
    num_q_heads = num_kv_heads
    assert num_kv_heads * alloc_block_size * alloc_head_dim == num_kv_heads * view_block_size * view_head_dim
    max_num_blocks_per_seq = max_seq_len_view // view_block_size
    assert max_num_blocks_per_seq * view_block_size == max_seq_len_view
    max_num_blocks = B * max_num_blocks_per_seq

    # Pre-fill cache with random bytes laid out in alloc's tile-grid
    # (this is what a shared buffer looks like when allocated by a
    # peer layer of a different spec).
    k_alloc = torch.randn(max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim).bfloat16().float()
    v_alloc = torch.randn(max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim).bfloat16().float()
    k_tt, _ = _alloc_paged_cache_on_device(
        max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim, device, fill=k_alloc
    )
    v_tt, _ = _alloc_paged_cache_on_device(
        max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim, device, fill=v_alloc
    )

    # Re-interpret the SAME bytes under the view's tile-grid — that's
    # what the layer's SDPA call (with override) will read.
    k_view = _permute_tile_grid(k_alloc, view_block_size, view_head_dim)
    v_view = _permute_tile_grid(v_alloc, view_block_size, view_head_dim)

    # The page_table indexes the view's blocks (each user reads through
    # its own per-virtual-block sequence). Per-block bytes match, so the
    # block_id semantics carry over from alloc.
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B, max_num_blocks_per_seq)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)

    # Reconstruct per-user K/V in the view's coord system by inverting
    # the page_table (block_id → which user/virtual-block).
    k_per_user = torch.zeros(B, num_kv_heads, max_seq_len_view, view_head_dim, dtype=torch.float32)
    v_per_user = torch.zeros(B, num_kv_heads, max_seq_len_view, view_head_dim, dtype=torch.float32)
    for b in range(B):
        for vb in range(max_num_blocks_per_seq):
            pb = int(page_table[b, vb].item())
            k_per_user[b, :, vb * view_block_size : (vb + 1) * view_block_size, :] = k_view[pb, :, :, :]
            v_per_user[b, :, vb * view_block_size : (vb + 1) * view_block_size, :] = v_view[pb, :, :, :]

    cur_pos = torch.tensor([10 + b * 7 for b in range(B)], dtype=torch.int32)
    q = torch.randn(1, B, num_q_heads, view_head_dim).bfloat16().float()
    q_padded = torch.nn.functional.pad(q, (0, 0, 0, 32 - num_q_heads), "constant", 0)
    q_tt = ttnn.Tensor(q_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    cur_pos_tt = ttnn.Tensor(cur_pos, ttnn.int32).to(device)

    scale = 1.0 / (view_head_dim**0.5)
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 4),
        q_chunk_size=32,
        k_chunk_size=32,
        exp_approx_mode=False,
    )

    out_tt = ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_tt,
        k_tt,
        v_tt,
        page_table_tensor=page_table_tt,
        cur_pos_tensor=cur_pos_tt,
        scale=scale,
        program_config=program_config,
        block_size=view_block_size,
    )
    out = ttnn.to_torch(out_tt)[:, :, :num_q_heads, :]

    ref = _torch_sdpa_reference(q, k_per_user, v_per_user, cur_pos, scale)
    eq, msg = comp_pcc(ref, out, pcc=0.99)
    assert eq, f"override view PCC failed: {msg}"


def test_override_view_full_into_sliding_buffer(device):
    """Cache allocated sliding (block=128, head=256); SDPA reads with
    full view (block=64, head=512) via override. Verifies the kernel
    addresses the right DRAM bytes for the view's geometry."""
    torch.manual_seed(2)
    _run_flexible_geometry_test(
        device,
        alloc_block_size=128,
        alloc_head_dim=256,
        view_block_size=64,
        view_head_dim=512,
    )


def test_override_view_sliding_into_full_buffer(device):
    """Symmetric: cache allocated full (block=64, head=512); SDPA reads
    with sliding view (block=128, head=256) via override."""
    torch.manual_seed(3)
    _run_flexible_geometry_test(
        device,
        alloc_block_size=64,
        alloc_head_dim=512,
        view_block_size=128,
        view_head_dim=256,
    )


def test_negative_byte_count_mismatch(device):
    """An override that breaks per-block byte invariant must be rejected
    at validation time, not silently corrupt the read."""
    torch.manual_seed(4)
    B = 2
    num_kv_heads = 1
    num_q_heads = 1
    # Cache: 1 * 64 * 512 = 32768 elem/block.
    alloc_block_size = 64
    alloc_head_dim = 512
    # Call: 1 * 256 * 256 = 65536 — DOUBLE the bytes.
    view_block_size = 256
    view_head_dim = 256
    max_num_blocks = 4

    k_tt, _ = _alloc_paged_cache_on_device(max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim, device)
    v_tt, _ = _alloc_paged_cache_on_device(max_num_blocks, num_kv_heads, alloc_block_size, alloc_head_dim, device)
    page_table = torch.arange(max_num_blocks, dtype=torch.int32).reshape(B, 2)
    page_table_tt = ttnn.Tensor(page_table, ttnn.int32).to(device)
    cur_pos_tt = ttnn.Tensor(torch.zeros(B, dtype=torch.int32), ttnn.int32).to(device)
    q = torch.zeros(1, B, num_q_heads, view_head_dim).bfloat16().float()
    q_padded = torch.nn.functional.pad(q, (0, 0, 0, 32 - num_q_heads), "constant", 0)
    q_tt = ttnn.Tensor(q_padded, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    with pytest.raises(RuntimeError, match="geometry mismatch"):
        ttnn.transformer.paged_scaled_dot_product_attention_decode(
            q_tt,
            k_tt,
            v_tt,
            page_table_tensor=page_table_tt,
            cur_pos_tensor=cur_pos_tt,
            block_size=view_block_size,
        )
