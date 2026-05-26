# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DreamZero World Action Model - PyTorch Reference Implementation.

This module provides the complete DreamZero model that orchestrates all components:
    - WanDiTModel: Video diffusion transformer backbone (Wan2.1-I2V)
    - DreamZeroActionHead: Flow matching action prediction head
    - FlowMatchScheduler: Denoising schedule for action generation

DreamZero jointly predicts video frames and robot actions using a shared DiT
backbone, enabling zero-shot transfer to new tasks and embodiments.

Reference: "World Action Models are Zero-shot Policies" (arXiv:2602.15922)
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from models.experimental.dreamzero.common.configs import (
    ActionHeadConfig,
    DreamZeroConfig,
    WanDiTConfig,
)
from models.experimental.dreamzero.common.weight_loader import DreamZeroWeightLoader
from models.experimental.dreamzero.reference.torch_action_head import (
    DreamZeroActionHead,
    WanDiTModel,
)


class DreamZeroModel(nn.Module):
    """
    Complete DreamZero World Action Model (PyTorch reference).

    This class orchestrates all components for inference:
    1. Encodes video observations into latents (VAE encoder - external)
    2. Encodes text instructions (text encoder - external)
    3. Encodes images for conditioning (CLIP - external)
    4. Runs joint video + action denoising through DiT backbone
    5. Outputs predicted actions for the robot

    The model takes pre-encoded inputs (latents, text embeddings, CLIP features)
    and focuses on the core DiT + action head computation.
    """

    def __init__(self, config: DreamZeroConfig, weight_loader: Optional[DreamZeroWeightLoader] = None):
        """
        Initialize DreamZero model.

        Args:
            config: Model configuration
            weight_loader: Optional weight loader for pretrained weights
        """
        super().__init__()
        self.config = config

        # Initialize DiT backbone
        self.dit = WanDiTModel(config.dit_config)

        # Initialize action head
        self.action_head = DreamZeroActionHead(config.action_head_config, config.dit_config)

        # Load pretrained weights if available
        if weight_loader is not None:
            self._load_weights(weight_loader)

    def _load_weights(self, weight_loader: DreamZeroWeightLoader):
        """Load pretrained weights into model components."""
        state_dict = weight_loader.state_dict

        # Load DiT weights
        dit_weights = weight_loader.get_dit_weights()
        if dit_weights:
            missing, unexpected = self.dit.load_state_dict(dit_weights, strict=False)
            if missing:
                print(f"DiT missing keys: {len(missing)}")
            if unexpected:
                print(f"DiT unexpected keys: {len(unexpected)}")

        # Load action MLP weights
        action_mlp_weights = weight_loader.get_action_mlp_weights()
        if action_mlp_weights:
            missing, unexpected = self.action_head.action_mlp.load_state_dict(
                action_mlp_weights, strict=False
            )

        # Load action input projection if present
        action_in_proj_key = "action_head.action_in_proj.weight"
        if action_in_proj_key in state_dict:
            self.action_head.action_in_proj.weight.data = state_dict[action_in_proj_key]
        action_in_proj_bias_key = "action_head.action_in_proj.bias"
        if action_in_proj_bias_key in state_dict:
            self.action_head.action_in_proj.bias.data = state_dict[action_in_proj_bias_key]

    @torch.no_grad()
    def get_actions(
        self,
        video_latent: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        ref_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate robot actions from encoded observations.

        This is the main inference entry point. It runs the flow matching
        denoising loop to produce action predictions.

        Args:
            video_latent: Encoded video observations (B, C, F, H, W)
                          where C=16 (VAE latent dim), F=frames/4, H=height/8, W=width/8
            context: Text instruction embeddings (B, S, 4096) from UMT5-XXL
            clip_feature: CLIP image embeddings (B, 257, 1280) [optional]
            ref_latent: Reference frame latent for I2V conditioning (B, C, 1, H, W) [optional]

        Returns:
            Predicted actions (B, action_horizon, action_dim)
        """
        batch_size = video_latent.shape[0]
        device = video_latent.device

        # Use zero timestep for video (we're not denoising video, just using backbone)
        timestep_video = torch.zeros(batch_size, device=device)

        # Generate actions through flow matching
        actions = self.action_head.sample_actions(
            dit_model=self.dit,
            video_latent=video_latent,
            timestep_video=timestep_video,
            context=context,
            clip_feature=clip_feature,
            ref_latent=ref_latent,
            batch_size=batch_size,
            device=device,
        )

        return actions

    @torch.no_grad()
    def joint_video_action(
        self,
        video_latent: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        ref_latent: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joint video and action prediction (single denoising step).

        Used for autoregressive generation where both video frames and
        actions are predicted simultaneously.

        Args:
            video_latent: Noisy video latent (B, C, F, H, W)
            timestep: Current diffusion timestep (B,)
            context: Text embeddings (B, S, 4096)
            clip_feature: CLIP features (B, 257, 1280) [optional]
            ref_latent: Reference latent (B, C, 1, H, W) [optional]

        Returns:
            Tuple of (video_prediction, action_prediction)
        """
        batch_size = video_latent.shape[0]
        device = video_latent.device

        # Start action tokens from noise for this step
        noisy_actions = torch.randn(
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
            device=device,
            dtype=video_latent.dtype,
        )

        action_tokens = self.action_head.embed_actions(noisy_actions)

        # Joint forward pass
        video_output, action_hidden = self.dit(
            x=video_latent,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=ref_latent,
            action_tokens=action_tokens,
        )

        # Project action hidden to action space
        action_pred = self.action_head.predict_velocity(action_hidden)

        return video_output, action_pred

    def forward(
        self,
        video_latent: torch.Tensor,
        context: torch.Tensor,
        clip_feature: Optional[torch.Tensor] = None,
        ref_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass - alias for get_actions."""
        return self.get_actions(
            video_latent=video_latent,
            context=context,
            clip_feature=clip_feature,
            ref_latent=ref_latent,
        )
