# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Common configurations for DreamZero World Action Model components.

DreamZero is a World Action Model (WAM) built on Wan2.1-I2V-14B video diffusion
backbone that jointly predicts actions and video for robot control.
Reference: "World Action Models are Zero-shot Policies" (arXiv:2602.15922)
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class WanDiTConfig:
    """Configuration for the Wan2.1 Video DiT backbone."""

    dim: int = 5120
    in_dim: int = 16  # VAE latent channels
    ffn_dim: int = 13824
    out_dim: int = 16
    freq_dim: int = 256
    eps: float = 1e-6
    num_heads: int = 40
    num_layers: int = 40
    text_dim: int = 4096  # UMT5-XXL text embedding dim
    patch_size: Tuple[int, int, int] = (1, 2, 2)
    has_image_input: bool = True
    clip_feature_dim: int = 1280

    @classmethod
    def wan_14b(cls) -> "WanDiTConfig":
        """Wan2.1-I2V-14B configuration (default DreamZero backbone)."""
        return cls(
            dim=5120,
            in_dim=16,
            ffn_dim=13824,
            out_dim=16,
            freq_dim=256,
            num_heads=40,
            num_layers=40,
            text_dim=4096,
        )

    @classmethod
    def wan_5b(cls) -> "WanDiTConfig":
        """Wan2.2-TI2V-5B configuration (smaller backbone)."""
        return cls(
            dim=4096,
            in_dim=16,
            ffn_dim=11008,
            out_dim=16,
            freq_dim=256,
            num_heads=32,
            num_layers=32,
            text_dim=4096,
        )


@dataclass
class ActionHeadConfig:
    """Configuration for the flow-matching action prediction head."""

    action_dim: int = 7  # Robot action dimension (e.g., 6DOF + gripper)
    action_horizon: int = 24  # Number of future action steps to predict
    hidden_dim: int = 5120  # Must match DiT backbone dim
    num_inference_steps: int = 10  # Flow matching denoising steps
    shift: float = 3.0  # Flow matching time shift
    sigma_max: float = 1.0
    sigma_min: float = 0.003 / 1.002


@dataclass
class VAEConfig:
    """Configuration for the Wan video VAE."""

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 16
    spatial_downsample: int = 8
    temporal_downsample: int = 4


@dataclass
class DreamZeroConfig:
    """Complete configuration for DreamZero World Action Model."""

    # Core architecture
    dit_config: WanDiTConfig = field(default_factory=WanDiTConfig.wan_14b)
    action_head_config: ActionHeadConfig = field(default_factory=ActionHeadConfig)
    vae_config: VAEConfig = field(default_factory=VAEConfig)

    # Input specifications
    num_frames: int = 33  # Number of video frames
    image_height: int = 176  # Input image height
    image_width: int = 320  # Input image width
    num_camera_views: int = 3  # Number of camera views

    # Action specifications
    action_dim: int = 7  # Robot action dimension
    action_horizon: int = 24  # Action prediction horizon

    # Inference settings
    num_inference_steps: int = 10
    precision: str = "bfloat16"

    # Model variant
    variant: str = "14b"  # "14b" or "5b"

    def __post_init__(self):
        if self.variant == "14b":
            self.dit_config = WanDiTConfig.wan_14b()
        elif self.variant == "5b":
            self.dit_config = WanDiTConfig.wan_5b()
        self.action_head_config.action_dim = self.action_dim
        self.action_head_config.action_horizon = self.action_horizon
        self.action_head_config.hidden_dim = self.dit_config.dim

    @classmethod
    def droid(cls) -> "DreamZeroConfig":
        """Configuration for DreamZero-DROID (14B, 7-DOF actions)."""
        return cls(
            action_dim=7,
            action_horizon=24,
            num_frames=33,
            image_height=176,
            image_width=320,
            num_camera_views=3,
            variant="14b",
        )

    @classmethod
    def droid_5b(cls) -> "DreamZeroConfig":
        """Configuration for DreamZero-DROID with 5B backbone."""
        return cls(
            action_dim=7,
            action_horizon=24,
            num_frames=33,
            image_height=176,
            image_width=320,
            num_camera_views=3,
            variant="5b",
        )
