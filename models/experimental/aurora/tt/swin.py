# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT-NN implementation of Aurora's 3D Swin transformer backbone.

The backbone dominates Aurora's FLOPs and parameters (the 1.3B configuration is
almost entirely backbone), so it is the natural target for hardware
acceleration.  The dense linear algebra of every block -- QKV / attention /
projection / adaptive layer-norm / MLP -- runs on the Tenstorrent device through
``ttnn``.  The exotic, memory-layout-only glue (3D window partition / reverse,
cyclic shift, padding, patch merge / split) is reused verbatim from the
reference implementation on the host, which guarantees bit-for-bit-identical
indexing and lets us focus correctness effort on the math.

All modules are configuration-driven so the same code runs the small pretrained
checkpoint (embed_dim=256, no LoRA) and the full 1.3B checkpoint
(embed_dim=512, LoRA).
"""

from __future__ import annotations

from datetime import timedelta

import torch

import ttnn

from aurora.model.fourier import lead_time_expansion
from aurora.model.swin3d import (
    compute_3d_shifted_window_mask,
    get_three_sidded_padding,
)
from aurora.model.util import maybe_adjust_windows

from models.experimental.aurora.tt.common import TtLinear, from_tt, to_tt


# Matmul fidelity is tunable: HiFi4 (4 phases) is most accurate but ~2x slower
# than HiFi2 and ~4x slower than LoFi. Weather residuals are small so we default
# to the safe HiFi4 + fp32 accumulation, but the dynamic range of the real
# checkpoint may tolerate HiFi2; gate any change on the PCC tests. Set via
# ``set_compute_fidelity`` before building the backbone.
# HiFi2 + fp32 accumulation measured to match HiFi4 worst-variable PCC (0.99992
# vs 0.99991 at 0.5deg on the real 1.3B checkpoint) while running the matmuls
# ~1.25x faster, so it is the default. bfp8/LoFi gave no speedup (the backbone is
# data-movement bound, not compute bound) and LoFi cost accuracy, so they are not
# defaults. See aurora_p1b.log.
_MATH_FIDELITY = ttnn.MathFidelity.HiFi2
_FP32_DEST_ACC = True

# Flash attention (ttnn.transformer.scaled_dot_product_attention) in the window
# attention, in place of the explicit QK^T/softmax/(.)V path. SDPA fuses
# scale+mask+softmax and never materializes the (N, N) scores in DRAM.
#
# OFF by default for the validated 0.25deg small/1.3B configs: even with a
# single-chunk program config + exact exp + HiFi2/fp32, the flash online-softmax
# in bf16 is ~0.9997/block, which compounds over ~48 blocks to backbone PCC
# ~0.989 (vs 0.998 exact) and pushes worst-variable full-model PCC to ~0.89 --
# below the 0.97 gate. At Aurora's window N=144 the (N,N) scores are tiny and the
# backbone is data-movement bound, not attention-FLOP bound, so there is no speed
# win to pay for that accuracy. SDPA is the right call only for the high-res
# variants (patch_size=10) where N and the window count explode; enable it there
# and re-check PCC. The exact manual path below is the default.
_USE_SDPA = False

# Fold single-mode LoRA into the base qkv/proj weights for inference:
# W' = W + (alpha/r)*(B@A). Exact (it is the same linear map), and it removes the
# two extra LoRA matmuls per projection per block (~4 matmuls/block over ~48
# blocks). Only valid for lora_mode="single" (one LoRA shared across rollout
# steps), which is what the port implements; the runtime two-matmul path is kept
# for debugging / a future per-step "all" mode. (OPTIMIZATION.md item 7.)
_FOLD_LORA = True


def set_fold_lora(enabled: bool):
    """Fold single-mode LoRA into the base weights at construction (default on)."""
    global _FOLD_LORA
    _FOLD_LORA = enabled


# Explicit core grid for the big QKV / proj / MLP matmuls. None -> ttnn
# auto-selects the grid (the default, and what the validated configs use). At
# Aurora's 0.25deg small/1.3B widths the backbone is data-movement / dispatch
# bound -- HiFi2 helps but bfp8/LoFi did not, dispatch is killed by the trace
# runner, and host round-trips by device-resident windowing -- so pinning the
# matmul grid is not a measurable win there (and a shared single chip makes clean
# matmul microbenchmarks impossible). It is the right lever for the compute-bound
# high-res variants (patch_size=10, the 2048-wide deepest stages); set the full
# device grid there and re-check PCC (the grid does not change the math).
_MATMUL_CORE_GRID = None


def set_matmul_core_grid(core_grid):
    """Pin the QKV/proj/MLP matmuls to an explicit ttnn.CoreGrid (default: auto)."""
    global _MATMUL_CORE_GRID
    _MATMUL_CORE_GRID = core_grid


# Tensor parallelism (Megatron-style col/row sharding of the MLP, with an
# all_reduce per block) requires a healthy inter-chip Ethernet fabric. It is
# OFF by default; enable only on a mesh whose fabric passes the collective
# handshake (see set_tensor_parallel). Validated correct on 1 chip (no-op);
# blocked on the bh_rev_c_ bring-up partition where the fabric handshake times out.
_TENSOR_PARALLEL = False


def set_tensor_parallel(enabled: bool):
    """Enable Megatron-style tensor parallelism for the MLP (needs working CCL)."""
    global _TENSOR_PARALLEL
    _TENSOR_PARALLEL = enabled


def set_compute_fidelity(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc=True):
    """Override the global matmul fidelity/accumulation used by the backbone."""
    global _MATH_FIDELITY, _FP32_DEST_ACC
    _MATH_FIDELITY = math_fidelity
    _FP32_DEST_ACC = fp32_dest_acc


def hifi_kernel_config():
    """Matmul compute-kernel config (fidelity + fp32 accumulation in DST).

    Aurora is numerically sensitive (weather residuals are small), so the default
    trades throughput for accuracy on the reduction-heavy matmuls; tune with
    ``set_compute_fidelity`` and re-check PCC.
    """
    return ttnn.WormholeComputeKernelConfig(  # also valid on Blackhole
        math_fidelity=_MATH_FIDELITY,
        math_approx_mode=False,
        fp32_dest_acc_en=_FP32_DEST_ACC,
        packer_l1_acc=True,
    )


# --- On-device window glue ---------------------------------------------------
# These replace the host torch.roll / window_partition_3d / window_reverse_3d /
# pad / crop with ttnn equivalents so the whole backbone stays resident on the
# device (one upload at the start of the backbone, one download at the end)
# instead of round-tripping the activation to host ~5x per block. The reshape +
# rank-8 permute pattern is identical indexing to the reference (validated to
# PCC 1.0). All shape gymnastics run in ROW_MAJOR; matmul/norm ops run in TILE.


def tt_window_partition_3d(xt, ws):
    """(B, C, H, W, D) -> (nW*B, Wc*Wh*Ww, D), matching window_partition_3d."""
    B, C, H, W, D = xt.shape
    wc, wh, ww = ws
    C1, H1, W1 = C // wc, H // wh, W // ww
    xt = ttnn.reshape(xt, (B, C1, wc, H1, wh, W1, ww, D))
    xt = ttnn.permute(xt, (0, 1, 3, 5, 2, 4, 6, 7))  # B C1 H1 W1 Wc Wh Ww D
    return ttnn.reshape(xt, (B * C1 * H1 * W1, wc * wh * ww, D))


def tt_window_reverse_3d(xt, ws, C, H, W, B=1):
    """(nW*B, Wc*Wh*Ww, D) -> (B, C, H, W, D), matching window_reverse_3d."""
    wc, wh, ww = ws
    C1, H1, W1 = C // wc, H // wh, W // ww
    D = xt.shape[-1]
    xt = ttnn.reshape(xt, (B, C1, H1, W1, wc, wh, ww, D))
    xt = ttnn.permute(xt, (0, 1, 4, 2, 5, 3, 6, 7))  # B C1 Wc H1 Wh W1 Ww D
    return ttnn.reshape(xt, (B, C, H, W, D))


def tt_pad_3d(xt, pad_size):
    """Two-sided pad on (B, C, H, W, D); C padding is always 0 for Aurora so we
    fold (B, C) into one dim and pad as rank-4 (rank-5 ttnn.pad is buggy)."""
    Cp, Hp, Wp = pad_size
    if Cp == 0 and Hp == 0 and Wp == 0:
        return xt
    assert Cp == 0, "device path assumes no level (C) padding"
    B, C, H, W, D = xt.shape
    pl, pr, pt, pb, _, _ = get_three_sidded_padding(Cp, Hp, Wp)
    r4 = ttnn.reshape(xt, (B * C, H, W, D))
    r4 = ttnn.pad(r4, padding=[(0, 0), (pt, pb), (pl, pr), (0, 0)], value=0.0)
    return ttnn.reshape(r4, (B, C, H + pt + pb, W + pl + pr, D))


def tt_crop_3d(xt, pad_size):
    """Undo tt_pad_3d by slicing (inverse of pad)."""
    Cp, Hp, Wp = pad_size
    if Cp == 0 and Hp == 0 and Wp == 0:
        return xt
    B, C, H, W, D = xt.shape
    pl, pr, pt, pb, pf, pbk = get_three_sidded_padding(Cp, Hp, Wp)
    return ttnn.slice(xt, [0, pf, pt, pl, 0], [B, C - pbk, H - pb, W - pr, D])


class TtAdaptiveLayerNorm:
    """FiLM-style adaptive layer-norm conditioned on the lead-time embedding.

    Reference: ``ln(x) * (scale_bias + scale) + shift`` where ``(shift, scale)``
    come from ``Linear(SiLU(c))``.  The affine LayerNorm here has no learnable
    weight/bias (``elementwise_affine=False``).
    """

    def __init__(self, sd, prefix, dim, device, scale_bias=0.0, eps=1e-5):
        self.dim = dim
        self.eps = eps
        self.scale_bias = scale_bias
        self.device = device
        # ln_modulation = Sequential(SiLU(), Linear(context_dim, dim*2))
        self.mod = TtLinear(
            sd[f"{prefix}.ln_modulation.1.weight"],
            sd[f"{prefix}.ln_modulation.1.bias"],
            device,
        )

    def __call__(self, x, c):
        # c: (B, context_dim) lead-time embedding. ln_modulation = SiLU -> Linear.
        mod = self.mod(ttnn.silu(c))  # (B, 2*dim)
        mod = ttnn.reshape(mod, (mod.shape[0], 1, 2 * self.dim))
        shift = mod[:, :, : self.dim]
        scale = mod[:, :, self.dim :]
        normed = ttnn.layer_norm(x, epsilon=self.eps)
        scaled = ttnn.multiply(normed, ttnn.add(scale, self.scale_bias))
        return ttnn.add(scaled, shift)


class TtMLP:
    """fc1 -> GELU -> fc2, with the GELU fused into the fc1 matmul epilogue."""

    def __init__(self, sd, prefix, device, weight_dtype=ttnn.bfloat16):
        # Tensor-parallel: fc1 column-parallel (shard the 4D hidden), fc2
        # row-parallel (all_reduce). The hidden activation never leaves the mesh
        # sharded, so only one all_reduce (in fc2) is paid per MLP. No-op on 1 chip.
        # Gated by _TENSOR_PARALLEL because it needs the inter-chip Ethernet fabric
        # (ttnn.all_reduce); leave off for single-chip or data-parallel meshes.
        tp1, tp2 = ("col", "row") if _TENSOR_PARALLEL else (None, None)
        self.fc1 = TtLinear(
            sd[f"{prefix}.fc1.weight"],
            sd[f"{prefix}.fc1.bias"],
            device,
            weight_dtype=weight_dtype,
            activation="gelu",
            tp=tp1,
        )
        self.fc2 = TtLinear(
            sd[f"{prefix}.fc2.weight"], sd[f"{prefix}.fc2.bias"], device, weight_dtype=weight_dtype, tp=tp2
        )
        self.kc = hifi_kernel_config()

    def __call__(self, x):
        h = self.fc1(x, compute_kernel_config=self.kc, core_grid=_MATMUL_CORE_GRID)  # GELU fused
        return self.fc2(h, compute_kernel_config=self.kc, core_grid=_MATMUL_CORE_GRID)


class TtWindowAttention:
    """Window-based multi-head self attention with optional additive mask and LoRA.

    Operates on windows of shape ``(nW*B, N, D)``.  Attention is computed with
    ``ttnn.transformer.scaled_dot_product_attention`` (flash attention): scale,
    additive shifted-window mask and softmax are fused and the ``(N, N)`` score
    matrix is never materialized in DRAM. LoRA corrections
    (``x @ A^T @ B^T * scaling``) compose with the base projections as in the
    reference. Set ``_USE_SDPA = False`` to fall back to the explicit
    QK^T / softmax / (.)V path (kept for debugging / numerical comparison).
    """

    def __init__(self, sd, prefix, dim, num_heads, device, use_lora=False, weight_dtype=ttnn.bfloat16):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.device = device
        self.use_lora = use_lora
        self.kc = hifi_kernel_config()

        qkv_w = sd[f"{prefix}.qkv.weight"]
        proj_w = sd[f"{prefix}.proj.weight"]
        # Runtime LoRA (two extra matmuls/projection) only when LoRA is on AND we
        # are not folding it into the base weight.
        self.lora_runtime = use_lora and not _FOLD_LORA
        if use_lora and _FOLD_LORA:
            qkv_w = self._fold(qkv_w, sd[f"{prefix}.lora_qkv.loras.0.lora_A"], sd[f"{prefix}.lora_qkv.loras.0.lora_B"])
            proj_w = self._fold(
                proj_w, sd[f"{prefix}.lora_proj.loras.0.lora_A"], sd[f"{prefix}.lora_proj.loras.0.lora_B"]
            )

        self.qkv = TtLinear(qkv_w, sd[f"{prefix}.qkv.bias"], device, weight_dtype=weight_dtype)
        self.proj = TtLinear(proj_w, sd[f"{prefix}.proj.bias"], device, weight_dtype=weight_dtype)

        if self.lora_runtime:
            # single-mode LoRA: loras[0]. A: (r, in), B: (out, r). correction =
            # (x @ A^T) @ B^T * (alpha/r). We fold scaling into B.
            self.qkv_A = to_tt(sd[f"{prefix}.lora_qkv.loras.0.lora_A"].t().contiguous(), device)
            qkv_B = sd[f"{prefix}.lora_qkv.loras.0.lora_B"]
            self.qkv_B = to_tt((qkv_B * (8.0 / qkv_B.shape[1])).t().contiguous(), device)
            self.proj_A = to_tt(sd[f"{prefix}.lora_proj.loras.0.lora_A"].t().contiguous(), device)
            proj_B = sd[f"{prefix}.lora_proj.loras.0.lora_B"]
            self.proj_B = to_tt((proj_B * (8.0 / proj_B.shape[1])).t().contiguous(), device)

    @staticmethod
    def _fold(W, A, B):
        """Fold single-mode LoRA into a base weight: W' = W + (alpha/r)*(B@A).

        W: (out, in), A: (r, in), B: (out, r); alpha=8 matches the runtime path's
        scaling (8.0 / r). Done in fp32 on host before upload, so it is exact.
        """
        r = A.shape[0]
        return (W + (8.0 / r) * (B @ A)).contiguous()

    def _lora(self, x, A, B):
        return ttnn.matmul(ttnn.matmul(x, A, compute_kernel_config=self.kc), B, compute_kernel_config=self.kc)

    def __call__(self, x, mask_tt=None):
        # x: (nWB, N, D)
        nWB, N, D = x.shape
        qkv = self.qkv(x, compute_kernel_config=self.kc, core_grid=_MATMUL_CORE_GRID)
        if self.lora_runtime:
            qkv = ttnn.add(qkv, self._lora(x, self.qkv_A, self.qkv_B))
        # (nWB, N, 3D) -> (nWB, N, 3, nH, hd) -> split
        qkv = ttnn.reshape(qkv, (nWB, N, 3, self.num_heads, self.head_dim))
        qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4))  # (3, nWB, nH, N, hd)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        if _USE_SDPA:
            # Flash attention: fuses scale + additive mask + softmax + (.)V and
            # never materializes the (N, N) scores in DRAM. The default scale is
            # 1/sqrt(head_dim) == self.scale; the additive mask broadcasts over heads.
            # The chunk size must cover the (tile-padded) window length N in a
            # single chunk and exp must be exact -- the default chunking + approx
            # exp drops per-block PCC ~0.996 -> compounds badly over ~48 blocks.
            chunk = ((N + 31) // 32) * 32
            pc = ttnn.SDPAProgramConfig(
                compute_with_storage_grid_size=self.device.compute_with_storage_grid_size(),
                q_chunk_size=chunk,
                k_chunk_size=chunk,
                exp_approx_mode=False,
            )
            out = ttnn.transformer.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask_tt,
                is_causal=False,
                scale=self.scale,
                compute_kernel_config=self.kc,
                program_config=pc,
            )  # (nWB, nH, N, hd)
        else:
            kt = ttnn.permute(k, (0, 1, 3, 2))  # (nWB, nH, hd, N)
            scores = ttnn.matmul(q, kt, compute_kernel_config=self.kc)  # (nWB, nH, N, N)
            scores = ttnn.multiply(scores, self.scale)
            if mask_tt is not None:
                scores = ttnn.add(scores, mask_tt)  # broadcast over heads
            attn = ttnn.softmax(scores, dim=-1)
            out = ttnn.matmul(attn, v, compute_kernel_config=self.kc)  # (nWB, nH, N, hd)
        out = ttnn.permute(out, (0, 2, 1, 3))  # (nWB, N, nH, hd)
        out = ttnn.reshape(out, (nWB, N, D))
        proj = self.proj(out, compute_kernel_config=self.kc, core_grid=_MATMUL_CORE_GRID)
        if self.lora_runtime:
            proj = ttnn.add(proj, self._lora(out, self.proj_A, self.proj_B))
        return proj


class TtSwin3DTransformerBlock:
    """One 3D Swin block. Window glue runs on host; dense math runs on device."""

    def __init__(
        self, sd, prefix, dim, num_heads, window_size, shift_size, device, use_lora=False, weight_dtype=ttnn.bfloat16
    ):
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.device = device
        # FiLM modulation is tiny + sensitive (produces scale/shift) -> keep bf16.
        self.norm1 = TtAdaptiveLayerNorm(sd, f"{prefix}.norm1", dim, device)
        self.norm2 = TtAdaptiveLayerNorm(sd, f"{prefix}.norm2", dim, device)
        self.attn = TtWindowAttention(
            sd, f"{prefix}.attn", dim, num_heads, device, use_lora=use_lora, weight_dtype=weight_dtype
        )
        self.mlp = TtMLP(sd, f"{prefix}.mlp", device, weight_dtype=weight_dtype)
        # Device-resident additive-mask cache. The host mask builder
        # (compute_3d_shifted_window_mask) is lru_cached, so for a fixed
        # (C, H, W, ws, ss) it returns the same tensor object every call; we key
        # the uploaded ttnn mask on its identity. Across an autoregressive
        # rollout the whole backbone (and thus this block) re-runs each 6 h step
        # with identical masks, so this turns ~one mask upload/block/step into a
        # one-time upload. (OPTIMIZATION.md item 7.)
        self._mask_cache: dict = {}

    def __call__(self, x_tt, c_tt, res, warped=True):
        """x_tt: (B, L, D) device tensor (TILE). c_tt: (B, D) conditioning on device.

        Fully device-resident: the window glue (shift/pad/partition/reverse/crop)
        runs in ROW_MAJOR via ttnn, the attention/norm/MLP in TILE, with cheap
        on-device layout conversions at the boundaries. No host round-trip.
        """
        C, H, W = res
        B, L, D = x_tt.shape
        ws, ss = maybe_adjust_windows(self.window_size, self.shift_size, res)
        shifted_block = not all(s == 0 for s in ss)

        shortcut = x_tt  # (B, L, D) TILE

        # --- windowed attention path, in ROW_MAJOR ---
        x = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, C, H, W, D))
        if shifted_block:
            x = ttnn.roll(x, [-ss[0], -ss[1], -ss[2]], [1, 2, 3])
            attn_mask, _ = compute_3d_shifted_window_mask(
                C, H, W, ws, ss, torch.device("cpu"), torch.float32, warped=warped
            )
        else:
            attn_mask = None

        pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
        x = tt_pad_3d(x, pad_size)
        _, pC, pH, pW, _ = x.shape
        x_windows = tt_window_partition_3d(x, ws)  # (nWB, N, D) ROW_MAJOR
        N = ws[0] * ws[1] * ws[2]

        # Broadcastable additive mask (nWB, 1, N, N), uploaded once and reused
        # across blocks/rollout steps with the same shift pattern (see _mask_cache).
        mask_tt = None
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            Bn = x_windows.shape[0] // nW
            cache_key = (id(attn_mask), Bn, N)
            mask_tt = self._mask_cache.get(cache_key)
            if mask_tt is None:
                m = attn_mask.unsqueeze(0).repeat(Bn, 1, 1, 1).reshape(-1, 1, N, N)
                mask_tt = to_tt(m.float(), self.device)
                self._mask_cache[cache_key] = mask_tt

        xw_tt = ttnn.to_layout(x_windows, ttnn.TILE_LAYOUT)
        attn_tt = self.attn(xw_tt, mask_tt=mask_tt)  # (nWB, N, D) TILE
        attn_windows = ttnn.to_layout(attn_tt, ttnn.ROW_MAJOR_LAYOUT)

        x = tt_window_reverse_3d(attn_windows, ws, pC, pH, pW, B)
        x = tt_crop_3d(x, pad_size)
        if shifted_block:
            x = ttnn.roll(x, [ss[0], ss[1], ss[2]], [1, 2, 3])
        x = ttnn.reshape(x, (B, C * H * W, D))

        # --- post-norm residuals, in TILE ---
        x_tt = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x_tt = ttnn.add(shortcut, self.norm1(x_tt, c_tt))
        x_tt = ttnn.add(x_tt, self.norm2(self.mlp(x_tt), c_tt))
        return x_tt


class TtPatchMerging3D:
    """Downsample: 2x2 spatial merge -> LayerNorm(4D) -> Linear(4D, 2D)."""

    def __init__(self, sd, prefix, dim, device):
        self.dim = dim
        self.device = device
        self.norm_w = to_tt(sd[f"{prefix}.norm.weight"], device)
        self.norm_b = to_tt(sd[f"{prefix}.norm.bias"], device)
        self.reduction = TtLinear(sd[f"{prefix}.reduction.weight"], None, device)
        self.kc = hifi_kernel_config()

    def __call__(self, x_tt, res):
        """x_tt: (B, L, D) device tensor (TILE). Returns (B, L/4, 2D) device TILE."""
        C, H, W = res
        B, L, D = x_tt.shape
        x = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, C, H, W, D))
        x = tt_pad_3d(x, (0, H % 2, W % 2))
        nH, nW = x.shape[2], x.shape[3]
        x = ttnn.reshape(x, (B, C, nH // 2, 2, nW // 2, 2, D))
        # "B C H h W w D -> B (C H W) (h w D)"
        x = ttnn.permute(x, (0, 1, 2, 4, 3, 5, 6))
        x = ttnn.reshape(x, (B, C * (nH // 2) * (nW // 2), 4 * D))
        x_tt = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x_tt = ttnn.layer_norm(x_tt, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5)
        return self.reduction(x_tt, compute_kernel_config=self.kc)


class TtPatchSplitting3D:
    """Upsample: Linear(D, 2D) -> 2x2 split -> LayerNorm(D/2) -> Linear(D/2, D/2)."""

    def __init__(self, sd, prefix, dim, device):
        self.dim = dim
        self.device = device
        self.lin1 = TtLinear(sd[f"{prefix}.lin1.weight"], None, device)
        self.lin2 = TtLinear(sd[f"{prefix}.lin2.weight"], None, device)
        self.norm_w = to_tt(sd[f"{prefix}.norm.weight"], device)
        self.norm_b = to_tt(sd[f"{prefix}.norm.bias"], device)
        self.kc = hifi_kernel_config()

    def __call__(self, x_tt, res, crop=(0, 0, 0)):
        """x_tt: (B, L, D) device tensor (TILE). Returns (B, ~4L, D/2) device TILE."""
        C, H, W = res
        B, L, D = x_tt.shape
        x_tt = self.lin1(x_tt, compute_kernel_config=self.kc)  # (B, L, 2D) TILE
        Dx = x_tt.shape[-1]
        x = ttnn.to_layout(x_tt, ttnn.ROW_MAJOR_LAYOUT)
        x = ttnn.reshape(x, (B, C, H, W, 2, 2, Dx // 4))
        # "B C H W h w D -> B C (H h) (W w) D"
        x = ttnn.permute(x, (0, 1, 2, 4, 3, 5, 6))
        x = ttnn.reshape(x, (B, C, 2 * H, 2 * W, Dx // 4))
        x = tt_crop_3d(x, crop)
        _, Cc, Hc, Wc, Dc = x.shape
        x = ttnn.reshape(x, (B, Cc * Hc * Wc, Dc))
        x_tt = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        x_tt = ttnn.layer_norm(x_tt, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5)
        return self.lin2(x_tt, compute_kernel_config=self.kc)


class TtBasicLayer3D:
    """A stage: a stack of Swin blocks followed by an optional down/upsample."""

    def __init__(
        self,
        sd,
        prefix,
        dim,
        depth,
        num_heads,
        window_size,
        device,
        downsample=False,
        upsample=False,
        use_lora=False,
        weight_dtype=ttnn.bfloat16,
    ):
        self.depth = depth
        self.blocks = []
        for i in range(depth):
            shift = (0, 0, 0) if i % 2 == 0 else (window_size[0] // 2, window_size[1] // 2, window_size[2] // 2)
            self.blocks.append(
                TtSwin3DTransformerBlock(
                    sd,
                    f"{prefix}.blocks.{i}",
                    dim,
                    num_heads,
                    window_size,
                    shift,
                    device,
                    use_lora=use_lora,
                    weight_dtype=weight_dtype,
                )
            )
        self.downsample = TtPatchMerging3D(sd, f"{prefix}.downsample", dim, device) if downsample else None
        self.upsample = TtPatchSplitting3D(sd, f"{prefix}.upsample", dim, device) if upsample else None

    def __call__(self, x_tt, c_tt, res, crop=(0, 0, 0)):
        for blk in self.blocks:
            x_tt = blk(x_tt, c_tt, res)
        if self.downsample is not None:
            return self.downsample(x_tt, res), x_tt
        if self.upsample is not None:
            return self.upsample(x_tt, res, crop), x_tt
        return x_tt, None


class TtSwin3DTransformerBackbone:
    """TT-NN port of Aurora's U-Net of 3D Swin transformer stages."""

    def __init__(
        self,
        sd,
        prefix,
        *,
        embed_dim,
        encoder_depths,
        encoder_num_heads,
        decoder_depths,
        decoder_num_heads,
        window_size,
        device,
        use_lora=False,
        weight_dtype=ttnn.bfloat16,
    ):
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.device = device
        self.weight_dtype = weight_dtype
        self.num_encoder_layers = len(encoder_depths)
        self.num_decoder_layers = len(decoder_depths)

        # time_mlp = Linear -> SiLU -> Linear
        self.time_lin0 = TtLinear(sd[f"{prefix}.time_mlp.0.weight"], sd[f"{prefix}.time_mlp.0.bias"], device)
        self.time_lin2 = TtLinear(sd[f"{prefix}.time_mlp.2.weight"], sd[f"{prefix}.time_mlp.2.bias"], device)

        self.encoder_layers = []
        for i in range(self.num_encoder_layers):
            self.encoder_layers.append(
                TtBasicLayer3D(
                    sd,
                    f"{prefix}.encoder_layers.{i}",
                    int(embed_dim * 2**i),
                    encoder_depths[i],
                    encoder_num_heads[i],
                    window_size,
                    device,
                    downsample=(i < self.num_encoder_layers - 1),
                    use_lora=use_lora,
                    weight_dtype=weight_dtype,
                )
            )
        self.decoder_layers = []
        for i in range(self.num_decoder_layers):
            exponent = self.num_decoder_layers - i - 1
            self.decoder_layers.append(
                TtBasicLayer3D(
                    sd,
                    f"{prefix}.decoder_layers.{i}",
                    int(embed_dim * 2**exponent),
                    decoder_depths[i],
                    decoder_num_heads[i],
                    window_size,
                    device,
                    upsample=(i < self.num_decoder_layers - 1),
                    use_lora=use_lora,
                    weight_dtype=weight_dtype,
                )
            )

    def get_encoder_specs(self, patch_res):
        all_res = [patch_res]
        padded_outs = []
        for _ in range(1, self.num_encoder_layers):
            C, H, W = all_res[-1]
            pad_H, pad_W = H % 2, W % 2
            padded_outs.append((0, pad_H, pad_W))
            all_res.append((C, (H + pad_H) // 2, (W + pad_W) // 2))
        padded_outs.append((0, 0, 0))
        return all_res, padded_outs

    def compute_conditioning(self, lead_time: timedelta, B, dtype=torch.float32):
        """Lead-time -> SiLU(time_mlp) conditioning ``c_tt`` (B, embed_dim) on
        device. Constant across an autoregressive rollout (same lead time), so the
        rollout runner computes it once and treats it as a trace constant."""
        lead_hours = lead_time / timedelta(hours=1)
        lead_times = lead_hours * torch.ones(B, dtype=torch.float32)
        c_torch = lead_time_expansion(lead_times, self.embed_dim).to(dtype=dtype)
        c_tt = self.time_lin0(to_tt(c_torch, self.device))
        c_tt = ttnn.silu(c_tt)
        return self.time_lin2(c_tt)  # (B, embed_dim)

    def forward_device(self, x, c_tt, patch_res):
        """Device-in / device-out U-Net core. ``x``: (B, L, D) device tensor (TILE);
        ``c_tt``: conditioning on device. All skips, residual adds and the final
        concat stay on device. Pure device ops (host work is only Python shape math
        and cache-hit mask lookups), so this is the region captured by the rollout
        trace runner."""
        all_enc_res, padded_outs = self.get_encoder_specs(patch_res)
        skips = []
        for i, layer in enumerate(self.encoder_layers):
            x, x_unscaled = layer(x, c_tt, all_enc_res[i])
            skips.append(x_unscaled)
        for i, layer in enumerate(self.decoder_layers):
            index = self.num_decoder_layers - i - 1
            x, _ = layer(x, c_tt, all_enc_res[index], padded_outs[index - 1])
            if 0 < i < self.num_decoder_layers - 1:
                x = ttnn.add(x, skips[index - 1])
            elif i == self.num_decoder_layers - 1:
                x = ttnn.concat([x, skips[0]], dim=-1)
        return x

    def __call__(self, x_torch, lead_time: timedelta, patch_res):
        """Device-resident: upload the latent once, run the whole U-Net on device,
        download once at the end. The encoder/decoder host glue never sees an
        intermediate. For multi-step rollout/serving use ``TtBackboneRunner``
        (tt/runner.py), which traces this graph and replays it per step."""
        B = x_torch.shape[0]
        c_tt = self.compute_conditioning(lead_time, B, dtype=x_torch.dtype)
        x = to_tt(x_torch, self.device)  # single host->device upload
        out = self.forward_device(x, c_tt, patch_res)
        return from_tt(out)  # single device->host download
