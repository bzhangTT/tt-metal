# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of Aurora's Perceiver resampler (encoder/decoder level aggregation).

Aurora's encoder and decoder are mostly cheap, irregular host preprocessing
(Fourier expansions, the 3D-conv patch embed, per-variable normalisation,
un-patchify) -- but the level aggregation / de-aggregation is a
``PerceiverResampler`` (Flamingo-style cross-attention), which is pure
``linear`` / attention / ``layer_norm`` / GELU-MLP, the same primitives already
on device in the backbone. At Aurora's native 0.25 deg resolution the
encoder/decoder are run over a ~720x1440 grid, so this cross-attention is the
device-portable part of the host tail; the Fourier/conv/normalisation around it
stay on host.

This module is a faithful, config-driven port of ``aurora.model.perceiver``
(``PerceiverAttention`` + ``MLP`` + ``PerceiverResampler``) validated standalone
against the reference (``test_perceiver_resampler``). The cross-attention context
is small (levels x sources, not the spatial grid), so attention is computed
exactly (QK^T / softmax / (.)V) rather than via flash attention.
"""

from __future__ import annotations

import ttnn

from models.experimental.aurora.tt.common import TtLinear, to_tt
from models.experimental.aurora.tt.swin import hifi_kernel_config


class TtPerceiverAttention:
    """Perceiver cross attention: latents attend to context. Biasless q/kv/out
    projections, optional pre-split LayerNorm on k and q (``ln_k_q``)."""

    def __init__(self, sd, prefix, *, latent_dim, context_dim, head_dim, num_heads, ln_k_q, device):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = head_dim * num_heads
        self.scale = head_dim**-0.5
        self.device = device
        self.kc = hifi_kernel_config()

        self.to_q = TtLinear(sd[f"{prefix}.to_q.weight"], None, device)
        self.to_kv = TtLinear(sd[f"{prefix}.to_kv.weight"], None, device)
        self.to_out = TtLinear(sd[f"{prefix}.to_out.weight"], None, device)

        self.ln_k_q = ln_k_q
        if ln_k_q:
            self.ln_k_w = to_tt(sd[f"{prefix}.ln_k.weight"], device)
            self.ln_k_b = to_tt(sd[f"{prefix}.ln_k.bias"], device)
            self.ln_q_w = to_tt(sd[f"{prefix}.ln_q.weight"], device)
            self.ln_q_b = to_tt(sd[f"{prefix}.ln_q.bias"], device)

    def _heads(self, t, L):
        # (B, L, inner) -> (B, num_heads, L, head_dim)
        B = t.shape[0]
        t = ttnn.reshape(t, (B, L, self.num_heads, self.head_dim))
        return ttnn.permute(t, (0, 2, 1, 3))

    def __call__(self, latents, x):
        # latents: (B, L1, latent_dim); x: (B, L2, context_dim)
        B, L1, _ = latents.shape
        L2 = x.shape[1]
        q = self.to_q(latents, compute_kernel_config=self.kc)
        kv = self.to_kv(x, compute_kernel_config=self.kc)
        k = kv[:, :, : self.inner_dim]
        v = kv[:, :, self.inner_dim :]
        # LayerNorm is applied before (!) splitting the heads.
        if self.ln_k_q:
            k = ttnn.layer_norm(k, weight=self.ln_k_w, bias=self.ln_k_b, epsilon=1e-5)
            q = ttnn.layer_norm(q, weight=self.ln_q_w, bias=self.ln_q_b, epsilon=1e-5)

        q = self._heads(q, L1)  # (B, H, L1, hd)
        k = self._heads(k, L2)  # (B, H, L2, hd)
        v = self._heads(v, L2)
        kt = ttnn.permute(k, (0, 1, 3, 2))  # (B, H, hd, L2)
        scores = ttnn.matmul(q, kt, compute_kernel_config=self.kc)  # (B, H, L1, L2)
        scores = ttnn.multiply(scores, self.scale)
        attn = ttnn.softmax(scores, dim=-1)
        out = ttnn.matmul(attn, v, compute_kernel_config=self.kc)  # (B, H, L1, hd)
        out = ttnn.permute(out, (0, 2, 1, 3))  # (B, L1, H, hd)
        out = ttnn.reshape(out, (B, L1, self.inner_dim))
        return self.to_out(out, compute_kernel_config=self.kc)


class TtPerceiverMLP:
    """Linear -> GELU -> Linear (GELU fused into the first matmul epilogue)."""

    def __init__(self, sd, prefix, device):
        self.fc1 = TtLinear(sd[f"{prefix}.net.0.weight"], sd[f"{prefix}.net.0.bias"], device, activation="gelu")
        self.fc2 = TtLinear(sd[f"{prefix}.net.2.weight"], sd[f"{prefix}.net.2.bias"], device)
        self.kc = hifi_kernel_config()

    def __call__(self, x):
        return self.fc2(self.fc1(x, compute_kernel_config=self.kc), compute_kernel_config=self.kc)


class TtPerceiverResampler:
    """Stack of (cross-attention, MLP) blocks with post-residual LayerNorms.

    Matches ``aurora.model.perceiver.PerceiverResampler`` exactly:
        attn_out = ln1(attn(latents, x)); latents = attn_out [+ latents];
        latents  = ln2(ff(latents)) + latents
    """

    def __init__(
        self,
        sd,
        prefix,
        *,
        latent_dim,
        context_dim,
        depth,
        head_dim,
        num_heads,
        residual_latent=True,
        ln_eps=1e-5,
        ln_k_q=False,
        device,
    ):
        self.residual_latent = residual_latent
        self.ln_eps = ln_eps
        self.layers = []
        for i in range(depth):
            p = f"{prefix}.layers.{i}"
            attn = TtPerceiverAttention(
                sd,
                f"{p}.0",
                latent_dim=latent_dim,
                context_dim=context_dim,
                head_dim=head_dim,
                num_heads=num_heads,
                ln_k_q=ln_k_q if i == 0 else False,
                device=device,
            )
            ff = TtPerceiverMLP(sd, f"{p}.1", device)
            ln1_w, ln1_b = to_tt(sd[f"{p}.2.weight"], device), to_tt(sd[f"{p}.2.bias"], device)
            ln2_w, ln2_b = to_tt(sd[f"{p}.3.weight"], device), to_tt(sd[f"{p}.3.bias"], device)
            self.layers.append((attn, ff, ln1_w, ln1_b, ln2_w, ln2_b))

    def __call__(self, latents, x):
        for attn, ff, ln1_w, ln1_b, ln2_w, ln2_b in self.layers:
            attn_out = ttnn.layer_norm(attn(latents, x), weight=ln1_w, bias=ln1_b, epsilon=self.ln_eps)
            latents = ttnn.add(attn_out, latents) if self.residual_latent else attn_out
            ff_out = ttnn.layer_norm(ff(latents), weight=ln2_w, bias=ln2_b, epsilon=self.ln_eps)
            latents = ttnn.add(ff_out, latents)
        return latents
