# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Weight loader for DreamZero World Action Model.

Handles loading pretrained weights from HuggingFace (GEAR-Dreams/DreamZero-DROID)
or local safetensors files. Supports sharded checkpoint loading.

Weight Structure:
    - backbone.*: Identity backbone (passthrough)
    - action_head.model.*: Wan DiT model with action tokens
    - action_head.action_mlp.*: Action projection MLP
    - action_head.vae.*: Video VAE encoder/decoder
    - action_head.clip_model.*: CLIP image encoder
    - action_head.text_encoder.*: UMT5-XXL text encoder
"""

import gc
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

try:
    from safetensors.torch import load_file as safetensors_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


@dataclass
class DreamZeroWeightInfo:
    """Metadata about loaded weights."""

    num_parameters: int = 0
    shard_files: List[str] = None
    has_lora: bool = False
    dit_layers: int = 0


class DreamZeroWeightLoader:
    """
    Loads and manages DreamZero model weights.

    Supports:
        - Single safetensors file
        - Sharded safetensors (model.safetensors.index.json)
        - LoRA weight merging
        - Selective component loading (DiT only, action head only, etc.)
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "cpu",
        load_vae: bool = True,
        load_text_encoder: bool = True,
        load_clip: bool = True,
    ):
        """
        Initialize weight loader.

        Args:
            model_path: Path to model checkpoint directory
            device: Device to load weights to
            load_vae: Whether to load VAE weights
            load_text_encoder: Whether to load text encoder weights
            load_clip: Whether to load CLIP weights
        """
        self.model_path = Path(model_path)
        self.device = device
        self.load_vae = load_vae
        self.load_text_encoder = load_text_encoder
        self.load_clip = load_clip

        self._state_dict: Optional[Dict[str, torch.Tensor]] = None
        self._weight_info = DreamZeroWeightInfo()

        if self.model_path.exists():
            self._load_weights()

    def _load_weights(self):
        """Load weights from checkpoint directory."""
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("safetensors is required for weight loading. pip install safetensors")

        safetensors_path = self.model_path / "model.safetensors"
        index_path = self.model_path / "model.safetensors.index.json"

        if index_path.exists():
            self._load_sharded(index_path)
        elif safetensors_path.exists():
            self._load_single(safetensors_path)
        else:
            raise FileNotFoundError(
                f"No weights found at '{self.model_path}'. "
                "Expected 'model.safetensors' or 'model.safetensors.index.json'."
            )

    def _load_sharded(self, index_path: Path):
        """Load sharded safetensors checkpoint."""
        with open(index_path, "r") as f:
            index = json.load(f)

        shard_files = sorted(set(index["weight_map"].values()))
        self._weight_info.shard_files = shard_files

        self._state_dict = {}
        for shard_file in shard_files:
            shard_path = self.model_path / shard_file
            shard_dict = safetensors_load_file(str(shard_path), device=self.device)

            # Filter based on component loading settings
            for key, value in shard_dict.items():
                if self._should_load_key(key):
                    self._state_dict[key] = value

            del shard_dict
            gc.collect()

        self._weight_info.num_parameters = sum(
            p.numel() for p in self._state_dict.values()
        )

    def _load_single(self, safetensors_path: Path):
        """Load single safetensors file."""
        full_dict = safetensors_load_file(str(safetensors_path), device=self.device)

        self._state_dict = {
            k: v for k, v in full_dict.items() if self._should_load_key(k)
        }
        del full_dict
        gc.collect()

        self._weight_info.num_parameters = sum(
            p.numel() for p in self._state_dict.values()
        )

    def _should_load_key(self, key: str) -> bool:
        """Determine if a weight key should be loaded based on settings."""
        if not self.load_vae and "vae" in key:
            return False
        if not self.load_text_encoder and "text_encoder" in key:
            return False
        if not self.load_clip and "clip_model" in key:
            return False
        return True

    @property
    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Get full state dict."""
        if self._state_dict is None:
            raise RuntimeError("Weights not loaded. Check model_path exists.")
        return self._state_dict

    @property
    def weight_info(self) -> DreamZeroWeightInfo:
        """Get weight metadata."""
        return self._weight_info

    def get_dit_weights(self) -> Dict[str, torch.Tensor]:
        """Get only DiT backbone weights (action_head.model.*)."""
        prefix = "action_head.model."
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_action_mlp_weights(self) -> Dict[str, torch.Tensor]:
        """Get action MLP projection weights."""
        prefix = "action_head.action_mlp."
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_vae_weights(self) -> Dict[str, torch.Tensor]:
        """Get VAE weights."""
        prefix = "action_head.vae."
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_clip_weights(self) -> Dict[str, torch.Tensor]:
        """Get CLIP image encoder weights."""
        prefix = "action_head.clip_model."
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def get_text_encoder_weights(self) -> Dict[str, torch.Tensor]:
        """Get text encoder weights."""
        prefix = "action_head.text_encoder."
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict.items()
            if k.startswith(prefix)
        }

    def load_lora_weights(self, lora_path: Union[str, Path]) -> Dict[str, torch.Tensor]:
        """
        Load LoRA weights from a separate checkpoint.

        Args:
            lora_path: Path to LoRA weights directory

        Returns:
            Dictionary of LoRA weight tensors
        """
        lora_path = Path(lora_path)
        safetensors_path = lora_path / "model.safetensors"
        index_path = lora_path / "model.safetensors.index.json"

        lora_dict = {}
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
            for shard_file in sorted(set(index["weight_map"].values())):
                shard_path = lora_path / shard_file
                shard_dict = safetensors_load_file(str(shard_path), device=self.device)
                # Only keep LoRA keys
                for k, v in shard_dict.items():
                    if "lora" in k.lower():
                        lora_dict[k] = v
                del shard_dict
                gc.collect()
        elif safetensors_path.exists():
            full_dict = safetensors_load_file(str(safetensors_path), device=self.device)
            lora_dict = {k: v for k, v in full_dict.items() if "lora" in k.lower()}
            del full_dict
            gc.collect()
        else:
            raise FileNotFoundError(f"No LoRA weights found at '{lora_path}'")

        self._weight_info.has_lora = True
        return lora_dict
