# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Action Head for DreamZero - PyTorch Reference Implementation.

Implements the flow-matching action prediction head that operates on top of
the DiT backbone features. Uses the DiT backbone to jointly denoise video
latents and action tokens, then projects action tokens to robot actions.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models.experimental.dreamzero.common.configs import ActionHeadConfig, WanDiTConfig
from models.experimental.dreamzero.reference.torch_dit_block import (
    DiTBlock,
    DiTHead,
    RMSNorm,
    RotaryPositionEmbedding3D,
    sinusoidal_embedding_1d,
)
from models.experimental.dreamzero.reference.torch_flow_matching import FlowMatchScheduler


class ActionMLP(nn.Module):
    """MLP that projects action tokens from DiT hidden dim to action space."""

    def __init__(self, hidden_dim: int, action_dim: int, action_horizon: int):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project DiT features to action predictions.

        Args:
            x: Action token features (B, action_horizon, hidden_dim)

        Returns:
            Action predictions (B, action_horizon, action_dim)
        """
        return self.proj(x)


class WanDiTModel(nn.Module):
    """
    Wan2.1 Video Diffusion Transformer with action token support.

    This is the core backbone of DreamZero that processes both video latents
    and action tokens through shared transformer layers with 3D RoPE and
    timestep-conditioned modulation.
    """

    def __init__(self, config: WanDiTConfig):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.freq_dim = config.freq_dim
        self.patch_size = config.patch_size

        # Patch embedding for video latents
        # When has_image_input, input is concat of x (in_dim) and ref (in_dim) = 2*in_dim
        patch_in_dim = config.in_dim * 2 if config.has_image_input else config.in_dim
        self.patch_embedding = nn.Conv3d(
            patch_in_dim, config.dim, kernel_size=config.patch_size, stride=config.patch_size
        )

        # Text conditioning
        self.text_embedding = nn.Sequential(
            nn.Linear(config.text_dim, config.dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.dim, config.dim),
        )

        # Timestep conditioning
        self.time_embedding = nn.Sequential(
            nn.Linear(config.freq_dim, config.dim),
            nn.SiLU(),
            nn.Linear(config.dim, config.dim),
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.dim, config.dim * 6),
        )

        # DiT blocks
        self.blocks = nn.ModuleList([
            DiTBlock(config.dim, config.num_heads, config.ffn_dim, config.eps, config.has_image_input)
            for _ in range(config.num_layers)
        ])

        # Output head
        self.head = DiTHead(config.dim, config.out_dim, config.patch_size, config.eps)

        # RoPE
        head_dim = config.dim // config.num_heads
        self.rope = RotaryPositionEmbedding3D(num_heads=config.num_heads, head_dim=head_dim)

        # Image embedding (CLIP features)
        if config.has_image_input:
            self.img_emb = nn.Sequential(
                nn.LayerNorm(config.clip_feature_dim),
                nn.Linear(config.clip_feature_dim, config.clip_feature_dim),
                nn.GELU(),
                nn.Linear(config.clip_feature_dim, config.dim),
                nn.LayerNorm(config.dim),
            )

    def patchify(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """Convert video tensor to sequence of patch tokens."""
        x = self.patch_embedding(x)
        grid_size = x.shape[2:]  # (f, h, w)
        x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()
        return x, grid_size

    def unpatchify(self, x: torch.Tensor, grid_size: Tuple[int, int, int]) -> torch.Tensor:
        """Convert patch token sequence back to video tensor."""
        f, h, w = grid_size
        px, py, pz = self.patch_size
        return rearrange(
            x,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=f, h=h, w=w, x=px, y=py, z=pz,
        )

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        y: Optional[torch.Tensor] = None,
        action_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the DiT model.

        Args:
            x: Video latent tensor (B, C, F, H, W)
            timestep: Diffusion timestep (B,)
            context: Text embeddings (B, S_text, text_dim)
            clip_feature: CLIP image features (B, 257, 1280) [optional]
            y: Reference image latent for I2V (B, C, 1, H, W) [optional]
            action_tokens: Action token embeddings (B, action_horizon, dim) [optional]

        Returns:
            Tuple of (video_output, action_output)
        """
        # Time conditioning
        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))

        # Text embedding
        context = self.text_embedding(context)

        # Image embedding and concatenation
        if self.config.has_image_input and clip_feature is not None:
            if y is not None:
                x = torch.cat([x, y], dim=1)  # Concat ref image latent
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        # Patchify video
        x, grid_size = self.patchify(x)
        f, h, w = grid_size

        # Concatenate action tokens if provided
        num_video_tokens = x.shape[1]
        if action_tokens is not None:
            x = torch.cat([x, action_tokens], dim=1)

        # Compute RoPE frequencies (for video tokens + action token positions)
        freqs = self.rope(f=f, h=h, w=w)

        # Extend RoPE for action tokens
        if action_tokens is not None:
            num_action_tokens = action_tokens.shape[1]
            # Action tokens use identity rotation (ones in complex space = no positional encoding)
            # This means action tokens have no spatial position bias
            action_freqs = torch.ones(
                num_action_tokens, 1, freqs.shape[-1],
                dtype=freqs.dtype, device=freqs.device
            )
            freqs = torch.cat([freqs, action_freqs], dim=0)

        # Run through DiT blocks
        for block in self.blocks:
            x = block(x, context, t_mod, freqs)

        # Split video and action outputs
        if action_tokens is not None:
            video_hidden = x[:, :num_video_tokens]
            action_hidden = x[:, num_video_tokens:]
        else:
            video_hidden = x
            action_hidden = None

        # Project video output
        video_output = self.head(video_hidden, t)

        return video_output, action_hidden


class DreamZeroActionHead(nn.Module):
    """
    DreamZero Action Prediction Head.

    Uses flow matching to denoise action tokens through the shared DiT backbone,
    then projects them to robot action space.
    """

    def __init__(self, config: ActionHeadConfig, dit_config: WanDiTConfig):
        super().__init__()
        self.config = config
        self.dit_config = dit_config

        # Action token embedding (projects noise/actions into DiT hidden space)
        self.action_in_proj = nn.Linear(config.action_dim, dit_config.dim)

        # Action output projection
        self.action_mlp = ActionMLP(
            hidden_dim=dit_config.dim,
            action_dim=config.action_dim,
            action_horizon=config.action_horizon,
        )

        # Flow matching scheduler
        self.scheduler = FlowMatchScheduler(
            num_inference_steps=config.num_inference_steps,
            shift=config.shift,
            sigma_max=config.sigma_max,
            sigma_min=config.sigma_min,
        )

    def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Embed noisy actions as tokens for the DiT backbone.

        Args:
            actions: Noisy action tensor (B, action_horizon, action_dim)

        Returns:
            Action token embeddings (B, action_horizon, dim)
        """
        return self.action_in_proj(actions)

    def predict_velocity(
        self,
        action_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict velocity (denoising direction) from action hidden states.

        Args:
            action_hidden: Hidden states for action tokens from DiT (B, action_horizon, dim)

        Returns:
            Predicted velocity (B, action_horizon, action_dim)
        """
        return self.action_mlp(action_hidden)

    @torch.no_grad()
    def sample_actions(
        self,
        dit_model: "WanDiTModel",
        video_latent: torch.Tensor,
        timestep_video: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        ref_latent: Optional[torch.Tensor] = None,
        batch_size: int = 1,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        Generate actions through iterative denoising.

        Args:
            dit_model: The DiT backbone model
            video_latent: Video latent for joint processing
            timestep_video: Video diffusion timestep
            context: Text conditioning
            clip_feature: CLIP image features
            ref_latent: Reference image latent
            batch_size: Batch size
            device: Device for tensor creation

        Returns:
            Predicted actions (B, action_horizon, action_dim)
        """
        if device is None:
            device = video_latent.device

        # Start from noise
        noisy_actions = torch.randn(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            device=device,
            dtype=video_latent.dtype,
        )

        # Iterative denoising
        for i, t in enumerate(self.scheduler.timesteps):
            # Embed current noisy actions
            action_tokens = self.embed_actions(noisy_actions)

            # Forward through DiT (joint video + action processing)
            _, action_hidden = dit_model(
                x=video_latent,
                timestep=timestep_video,
                context=context,
                clip_feature=clip_feature,
                y=ref_latent,
                action_tokens=action_tokens,
            )

            # Predict velocity
            velocity = self.predict_velocity(action_hidden)

            # Denoise one step
            noisy_actions = self.scheduler.step(
                model_output=velocity,
                timestep=t,
                sample=noisy_actions,
                to_final=(i == len(self.scheduler.timesteps) - 1),
            )

        return noisy_actions
