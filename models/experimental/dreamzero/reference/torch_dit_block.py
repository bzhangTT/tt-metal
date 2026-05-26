# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DiT Block for DreamZero - PyTorch Reference Implementation.

Implements the Diffusion Transformer (DiT) block used in the Wan2.1 backbone.
Includes self-attention with 3D RoPE, cross-attention for text/image conditioning,
and adaptive layer norm modulation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    """Generate 1D sinusoidal positional embeddings."""
    sinusoid = torch.outer(
        position.float(),
        torch.pow(
            10000,
            -torch.arange(dim // 2, dtype=torch.float32, device=position.device).div(dim // 2),
        ),
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(dtype) * self.weight


class RotaryPositionEmbedding3D(nn.Module):
    """3D Rotary Position Embedding for video tokens (frame, height, width)."""

    def __init__(self, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.freqs = self._precompute_freqs_3d(head_dim)

    def _precompute_freqs_1d(self, dim: int, end: int = 1024, theta: float = 10000.0):
        """Precompute 1D frequency components."""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].double() / dim))
        freqs = torch.outer(torch.arange(end), freqs)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        return freqs_cis

    def _precompute_freqs_3d(self, dim: int, end: int = 1024, theta: float = 10000.0):
        """Precompute 3D frequency components for frame/height/width."""
        f_freqs = self._precompute_freqs_1d(dim - 2 * (dim // 3), end, theta)
        h_freqs = self._precompute_freqs_1d(dim // 3, end, theta)
        w_freqs = self._precompute_freqs_1d(dim // 3, end, theta)
        return {"f": f_freqs, "h": h_freqs, "w": w_freqs}

    def forward(self, f: int, h: int, w: int) -> torch.Tensor:
        """
        Compute 3D RoPE frequencies for given grid dimensions.

        Args:
            f: Number of frames (temporal)
            h: Height (spatial)
            w: Width (spatial)

        Returns:
            Complex frequency tensor of shape (f*h*w, 1, head_dim//2)
        """
        freqs = torch.cat(
            [
                self.freqs["f"][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs["h"][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs["w"][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(f * h * w, 1, -1)
        return freqs


def rope_apply(x: torch.Tensor, freqs: torch.Tensor, num_heads: int) -> torch.Tensor:
    """Apply rotary position embeddings to query/key tensors."""
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_complex = torch.view_as_complex(
        x.to(torch.float64).reshape(x.shape[0], x.shape[1], x.shape[2], -1, 2)
    )
    freqs = freqs.to(x_complex.device)
    x_out = torch.view_as_real(x_complex * freqs).flatten(2)
    return x_out.to(x.dtype)


class SelfAttention(nn.Module):
    """Multi-head self-attention with RoPE."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)

        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)

        # Reshape for attention
        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)")
        return self.o(x)


class CrossAttention(nn.Module):
    """Multi-head cross-attention for text/image conditioning."""

    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.has_image_input = has_image_input

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        if self.has_image_input:
            img = context[:, :257]
            ctx = context[:, 257:]
        else:
            ctx = context

        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)

        q = rearrange(q, "b s (n d) -> b n s d", n=self.num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=self.num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=self.num_heads)

        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)")

        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            q_for_img = rearrange(q, "b n s d -> b s (n d)")
            q_for_img = rearrange(q_for_img, "b s (n d) -> b n s d", n=self.num_heads)
            k_img = rearrange(k_img, "b s (n d) -> b n s d", n=self.num_heads)
            v_img = rearrange(v_img, "b s (n d) -> b n s d", n=self.num_heads)
            img_attn = F.scaled_dot_product_attention(q_for_img, k_img, v_img)
            img_attn = rearrange(img_attn, "b n s d -> b s (n d)")
            x = x + img_attn

        return self.o(x)


class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block.

    Combines self-attention, cross-attention, and FFN with adaptive
    layer norm modulation conditioned on the diffusion timestep.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        eps: float = 1e-6,
        has_image_input: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )
        # 6 modulation params: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        t_mod: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through DiT block.

        Args:
            x: Input hidden states (B, S, D)
            context: Cross-attention context (text + image embeddings)
            t_mod: Timestep modulation signal (B, 6, D)
            freqs: RoPE frequencies

        Returns:
            Output hidden states (B, S, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod
        ).chunk(6, dim=1)

        # Self-attention with modulation
        input_x = self.norm1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.self_attn(input_x.to(t_mod.dtype), freqs)

        # Cross-attention
        x = x + self.cross_attn(self.norm3(x).to(t_mod.dtype), context.to(t_mod.dtype))

        # FFN with modulation
        input_x = self.norm2(x) * (1 + scale_mlp) + shift_mlp
        x = x + gate_mlp * self.ffn(input_x)

        return x


class DiTHead(nn.Module):
    """Final projection head for DiT output."""

    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, t_mod: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        shift, scale = (
            self.modulation.repeat(batch_size, 1, 1).to(dtype=t_mod.dtype, device=t_mod.device)
            + t_mod.unsqueeze(1)
        ).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + scale) + shift)
        return x
