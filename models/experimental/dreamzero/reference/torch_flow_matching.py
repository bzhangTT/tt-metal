# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Flow Matching Scheduler for DreamZero.

Implements the flow matching denoising schedule used for action prediction.
Based on the shifted sigmoid schedule from Wan2.1 video diffusion.
"""

import torch


class FlowMatchScheduler:
    """
    Flow matching scheduler for DreamZero action denoising.

    Uses a shifted linear schedule to transform noise into clean action predictions
    over multiple inference steps.

    Args:
        num_inference_steps: Number of denoising steps at inference time
        num_train_timesteps: Total number of training timesteps
        shift: Time shift parameter (controls noise schedule skew)
        sigma_max: Maximum noise level
        sigma_min: Minimum noise level
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        num_train_timesteps: int = 1000,
        shift: float = 3.0,
        sigma_max: float = 1.0,
        sigma_min: float = 0.003 / 1.002,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.set_timesteps(num_inference_steps)

    def set_timesteps(self, num_inference_steps: int, denoising_strength: float = 1.0):
        """Compute the sigma and timestep schedules."""
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        # Apply shift
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        self.timesteps = self.sigmas * self.num_train_timesteps

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        to_final: bool = False,
    ) -> torch.Tensor:
        """
        Perform one denoising step.

        Args:
            model_output: Predicted velocity from the model
            timestep: Current timestep
            sample: Current noisy sample
            to_final: If True, step directly to clean sample

        Returns:
            Denoised sample after one step
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin((self.timesteps - timestep).abs())
        sigma = self.sigmas[timestep_id]

        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_next = 0.0
        else:
            sigma_next = self.sigmas[timestep_id + 1]

        prev_sample = sample + model_output * (sigma_next - sigma)
        return prev_sample

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples for training.

        Args:
            original_samples: Clean action samples
            noise: Gaussian noise tensor
            timestep: Timestep for noise level

        Returns:
            Noisy samples
        """
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.cpu()
        timestep_id = torch.argmin(
            (self.timesteps.unsqueeze(1) - timestep.unsqueeze(0)).abs(), dim=0
        )
        sigma = self.sigmas[timestep_id].to(
            device=original_samples.device, dtype=original_samples.dtype
        )
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample

    def training_target(
        self,
        sample: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training target (velocity) for flow matching."""
        return noise - sample
