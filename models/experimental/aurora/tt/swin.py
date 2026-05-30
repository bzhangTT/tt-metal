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
    crop_3d,
    pad_3d,
    window_partition_3d,
    window_reverse_3d,
)
from aurora.model.util import maybe_adjust_windows

from models.experimental.aurora.tt.common import TtLinear, from_tt, to_tt


def hifi_kernel_config():
    """High-fidelity matmul config: HiFi4 + fp32 accumulation in DST.

    Aurora is numerically sensitive (weather residuals are small), so we trade a
    little throughput for accuracy on the reduction-heavy matmuls.
    """
    return ttnn.WormholeComputeKernelConfig(  # also valid on Blackhole
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=True,
    )


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
        self.fc1 = TtLinear(
            sd[f"{prefix}.fc1.weight"], sd[f"{prefix}.fc1.bias"], device, weight_dtype=weight_dtype, activation="gelu"
        )
        self.fc2 = TtLinear(sd[f"{prefix}.fc2.weight"], sd[f"{prefix}.fc2.bias"], device, weight_dtype=weight_dtype)
        self.kc = hifi_kernel_config()

    def __call__(self, x):
        h = self.fc1(x, compute_kernel_config=self.kc)  # GELU fused
        return self.fc2(h, compute_kernel_config=self.kc)


class TtWindowAttention:
    """Window-based multi-head self attention with optional additive mask and LoRA.

    Operates on windows of shape ``(nW*B, N, D)``.  Implemented as explicit
    QK^T / softmax / (.)V so the additive shifted-window mask can be applied
    exactly, and so LoRA corrections (``x @ A^T @ B^T * scaling``) compose with
    the base projections as in the reference.
    """

    def __init__(self, sd, prefix, dim, num_heads, device, use_lora=False, weight_dtype=ttnn.bfloat16):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.device = device
        self.use_lora = use_lora
        self.kc = hifi_kernel_config()

        self.qkv = TtLinear(sd[f"{prefix}.qkv.weight"], sd[f"{prefix}.qkv.bias"], device, weight_dtype=weight_dtype)
        self.proj = TtLinear(sd[f"{prefix}.proj.weight"], sd[f"{prefix}.proj.bias"], device, weight_dtype=weight_dtype)

        if use_lora:
            # single-mode LoRA: loras[0]. A: (r, in), B: (out, r). correction =
            # (x @ A^T) @ B^T * (alpha/r). We fold scaling into B.
            self.qkv_A = to_tt(sd[f"{prefix}.lora_qkv.loras.0.lora_A"].t().contiguous(), device)
            qkv_B = sd[f"{prefix}.lora_qkv.loras.0.lora_B"]
            self.qkv_B = to_tt((qkv_B * (8.0 / qkv_B.shape[1])).t().contiguous(), device)
            self.proj_A = to_tt(sd[f"{prefix}.lora_proj.loras.0.lora_A"].t().contiguous(), device)
            proj_B = sd[f"{prefix}.lora_proj.loras.0.lora_B"]
            self.proj_B = to_tt((proj_B * (8.0 / proj_B.shape[1])).t().contiguous(), device)

    def _lora(self, x, A, B):
        return ttnn.matmul(ttnn.matmul(x, A, compute_kernel_config=self.kc), B, compute_kernel_config=self.kc)

    def __call__(self, x, mask_tt=None):
        # x: (nWB, N, D)
        nWB, N, D = x.shape
        qkv = self.qkv(x, compute_kernel_config=self.kc)
        if self.use_lora:
            qkv = ttnn.add(qkv, self._lora(x, self.qkv_A, self.qkv_B))
        # (nWB, N, 3D) -> (nWB, N, 3, nH, hd) -> split
        qkv = ttnn.reshape(qkv, (nWB, N, 3, self.num_heads, self.head_dim))
        qkv = ttnn.permute(qkv, (2, 0, 3, 1, 4))  # (3, nWB, nH, N, hd)
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        kt = ttnn.permute(k, (0, 1, 3, 2))  # (nWB, nH, hd, N)
        scores = ttnn.matmul(q, kt, compute_kernel_config=self.kc)  # (nWB, nH, N, N)
        scores = ttnn.multiply(scores, self.scale)
        if mask_tt is not None:
            scores = ttnn.add(scores, mask_tt)  # broadcast over heads
        attn = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn, v, compute_kernel_config=self.kc)  # (nWB, nH, N, hd)
        out = ttnn.permute(out, (0, 2, 1, 3))  # (nWB, N, nH, hd)
        out = ttnn.reshape(out, (nWB, N, D))
        proj = self.proj(out, compute_kernel_config=self.kc)
        if self.use_lora:
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

    def __call__(self, x_torch, c_tt, res, warped=True):
        """x_torch: (B, L, D) host tensor. c_tt: (B, D) SiLU-conditioning on device."""
        C, H, W = res
        B, L, D = x_torch.shape
        ws, ss = maybe_adjust_windows(self.window_size, self.shift_size, res)

        shortcut = x_torch
        x = x_torch.view(B, C, H, W, D)

        if not all(s == 0 for s in ss):
            shifted = torch.roll(x, shifts=(-ss[0], -ss[1], -ss[2]), dims=(1, 2, 3))
            attn_mask, _ = compute_3d_shifted_window_mask(C, H, W, ws, ss, x.device, x.dtype, warped=warped)
        else:
            shifted = x
            attn_mask = None

        pad_size = ((-C) % ws[0], (-H) % ws[1], (-W) % ws[2])
        shifted = pad_3d(shifted, pad_size)
        x_windows = window_partition_3d(shifted, ws)
        N = ws[0] * ws[1] * ws[2]
        x_windows = x_windows.reshape(-1, N, D)  # (nWB, N, D)

        # Build broadcastable additive mask (nWB, 1, N, N) and upload once,
        # reusing the device tensor across blocks/rollout steps with the same
        # shift pattern (see _mask_cache).
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

        xw_tt = to_tt(x_windows, self.device)
        attn_tt = self.attn(xw_tt, mask_tt=mask_tt)
        attn_windows = from_tt(attn_tt).reshape(-1, ws[0], ws[1], ws[2], D)

        _, pC, pH, pW, _ = shifted.shape
        shifted = window_reverse_3d(attn_windows, ws, pC, pH, pW)
        shifted = crop_3d(shifted, pad_size)

        if not all(s == 0 for s in ss):
            x = torch.roll(shifted, shifts=(ss[0], ss[1], ss[2]), dims=(1, 2, 3))
        else:
            x = shifted
        x = x.reshape(B, C * H * W, D)

        # Post-norm residuals, on device.
        x_tt = to_tt(x, self.device)
        sc_tt = to_tt(shortcut, self.device)
        x_tt = ttnn.add(sc_tt, self.norm1(x_tt, c_tt))
        x_tt = ttnn.add(x_tt, self.norm2(self.mlp(x_tt), c_tt))
        return from_tt(x_tt)


class TtPatchMerging3D:
    """Downsample: 2x2 spatial merge -> LayerNorm(4D) -> Linear(4D, 2D)."""

    def __init__(self, sd, prefix, dim, device):
        self.dim = dim
        self.device = device
        self.norm_w = to_tt(sd[f"{prefix}.norm.weight"], device)
        self.norm_b = to_tt(sd[f"{prefix}.norm.bias"], device)
        self.reduction = TtLinear(sd[f"{prefix}.reduction.weight"], None, device)
        self.kc = hifi_kernel_config()

    def __call__(self, x_torch, res):
        C, H, W = res
        B, L, D = x_torch.shape
        x = x_torch.view(B, C, H, W, D)
        x = pad_3d(x, (0, H % 2, W % 2))
        nH, nW = x.shape[2], x.shape[3]
        x = x.reshape(B, C, nH // 2, 2, nW // 2, 2, D)
        # "B C H h W w D -> B (C H W) (h w D)"
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, C * (nH // 2) * (nW // 2), 4 * D)
        x_tt = to_tt(x, self.device)
        x_tt = ttnn.layer_norm(x_tt, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5)
        x_tt = self.reduction(x_tt, compute_kernel_config=self.kc)
        return from_tt(x_tt)


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

    def __call__(self, x_torch, res, crop=(0, 0, 0)):
        C, H, W = res
        B, L, D = x_torch.shape
        x_tt = self.lin1(to_tt(x_torch, self.device), compute_kernel_config=self.kc)  # (B, L, 2D)
        x = from_tt(x_tt)
        Dx = x.shape[-1]
        x = x.view(B, C, H, W, 2, 2, Dx // 4)
        # "B C H W h w D -> B C (H h) (W w) D"
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(B, C, 2 * H, 2 * W, Dx // 4)
        x = crop_3d(x, crop)
        x = x.reshape(B, -1, Dx // 4)
        x_tt = to_tt(x, self.device)
        x_tt = ttnn.layer_norm(x_tt, weight=self.norm_w, bias=self.norm_b, epsilon=1e-5)
        x_tt = self.lin2(x_tt, compute_kernel_config=self.kc)
        return from_tt(x_tt)


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

    def __call__(self, x_torch, c_tt, res, crop=(0, 0, 0)):
        for blk in self.blocks:
            x_torch = blk(x_torch, c_tt, res)
        if self.downsample is not None:
            return self.downsample(x_torch, res), x_torch
        if self.upsample is not None:
            return self.upsample(x_torch, res, crop), x_torch
        return x_torch, None


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

    def __call__(self, x_torch, lead_time: timedelta, patch_res):
        B = x_torch.shape[0]
        all_enc_res, padded_outs = self.get_encoder_specs(patch_res)

        lead_hours = lead_time / timedelta(hours=1)
        lead_times = lead_hours * torch.ones(B, dtype=torch.float32)
        c_torch = lead_time_expansion(lead_times, self.embed_dim).to(dtype=x_torch.dtype)
        c_tt = self.time_lin0(to_tt(c_torch, self.device))
        c_tt = ttnn.silu(c_tt)
        c_tt = self.time_lin2(c_tt)  # (B, embed_dim)

        skips = []
        for i, layer in enumerate(self.encoder_layers):
            x_torch, x_unscaled = layer(x_torch, c_tt, all_enc_res[i])
            skips.append(x_unscaled)
        for i, layer in enumerate(self.decoder_layers):
            index = self.num_decoder_layers - i - 1
            x_torch, _ = layer(x_torch, c_tt, all_enc_res[index], padded_outs[index - 1])
            if 0 < i < self.num_decoder_layers - 1:
                x_torch = x_torch + skips[index - 1]
            elif i == self.num_decoder_layers - 1:
                x_torch = torch.cat([x_torch, skips[0]], dim=-1)
        return x_torch
