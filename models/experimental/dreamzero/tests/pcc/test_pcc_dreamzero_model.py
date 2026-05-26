# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
DreamZero Reference Model Test - PyTorch forward pass validation.

Tests that the DreamZero reference model can:
    1. Load pretrained weights from checkpoint
    2. Process encoded inputs (video latents, text embeddings, CLIP features)
    3. Generate action predictions with correct shapes
    4. Produce deterministic outputs with fixed seeds

Config:
    - Checkpoint: $TT_METAL_HOME/models/experimental/dreamzero/weights/dreamzero_droid
    - Denoising steps: 10
    - Batch size: 1
    - Action dim: 7 (DROID 6-DOF + gripper)
    - Action horizon: 24

Usage:
    pytest test_pcc_dreamzero_model.py -v
    python test_pcc_dreamzero_model.py  # standalone
"""

import os
import sys
import time
from pathlib import Path

import pytest
import torch

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from models.experimental.dreamzero.common.configs import DreamZeroConfig
from models.experimental.dreamzero.common.weight_loader import DreamZeroWeightLoader
from models.experimental.dreamzero.reference.torch_dreamzero_model import DreamZeroModel


# =============================================================================
# CONFIGURATION
# =============================================================================
TT_METAL_HOME = os.environ.get("TT_METAL_HOME", "")
CHECKPOINT_PATH = os.path.join(
    TT_METAL_HOME, "models/experimental/dreamzero/weights/dreamzero_droid"
)
BATCH_SIZE = 1
SEED = 42


def create_tiny_config() -> DreamZeroConfig:
    """Create a tiny DreamZero config for testing without OOM."""
    from models.experimental.dreamzero.common.configs import WanDiTConfig, ActionHeadConfig

    config = DreamZeroConfig(
        action_dim=7,
        action_horizon=24,
        num_frames=5,
        image_height=64,
        image_width=64,
        num_camera_views=1,
        variant="14b",  # Will be overridden
    )
    # Override with tiny DiT config
    config.dit_config = WanDiTConfig(
        dim=1024,
        in_dim=16,
        ffn_dim=2048,
        out_dim=16,
        freq_dim=256,
        eps=1e-6,
        num_heads=8,
        num_layers=2,  # Only 2 layers for testing
        text_dim=4096,
        patch_size=(1, 2, 2),
        has_image_input=True,
        clip_feature_dim=1280,
    )
    config.action_head_config = ActionHeadConfig(
        action_dim=7,
        action_horizon=24,
        hidden_dim=1024,
        num_inference_steps=3,  # Fewer steps for speed
    )
    return config


def create_config() -> DreamZeroConfig:
    """Create DreamZero-DROID configuration."""
    return DreamZeroConfig.droid()


def create_test_inputs(config: DreamZeroConfig, batch_size: int = 1, device: str = "cpu"):
    """
    Create synthetic test inputs matching DreamZero's expected format.

    These simulate the outputs of:
    - VAE encoder (video_latent)
    - UMT5-XXL text encoder (context)
    - CLIP image encoder (clip_feature)
    - VAE encoder on reference frame (ref_latent)
    """
    dtype = torch.bfloat16 if config.precision == "bfloat16" else torch.float32

    # Video latent: (B, C=16, F=frames/temporal_downsample, H=height/spatial_ds, W=width/spatial_ds)
    # For 33 frames at 4x temporal downsample: F = ceil(33/4) = 9
    # For 176x320 at 8x spatial downsample: H=22, W=40
    latent_frames = (config.num_frames + config.vae_config.temporal_downsample - 1) // config.vae_config.temporal_downsample
    latent_h = config.image_height // config.vae_config.spatial_downsample
    latent_w = config.image_width // config.vae_config.spatial_downsample
    latent_c = config.vae_config.latent_channels

    video_latent = torch.randn(
        batch_size, latent_c, latent_frames, latent_h, latent_w,
        dtype=dtype, device=device,
    )

    # Text context: (B, seq_len, text_dim=4096) from UMT5-XXL
    context = torch.randn(
        batch_size, 512, config.dit_config.text_dim,
        dtype=dtype, device=device,
    )

    # CLIP features: (B, 257, 1280) - 256 patch tokens + 1 CLS token
    clip_feature = torch.randn(
        batch_size, 257, config.dit_config.clip_feature_dim,
        dtype=dtype, device=device,
    )

    # Reference latent (for I2V): (B, C, F, H, W) - same as video but different channels
    # In Wan I2V, reference frame is repeated/padded to match video temporal dim
    ref_latent = torch.zeros(
        batch_size, latent_c, latent_frames, latent_h, latent_w,
        dtype=dtype, device=device,
    )
    # First frame is the actual reference, rest are zeros
    ref_latent[:, :, 0] = torch.randn(batch_size, latent_c, latent_h, latent_w, dtype=dtype, device=device)

    return {
        "video_latent": video_latent,
        "context": context,
        "clip_feature": clip_feature,
        "ref_latent": ref_latent,
    }


# =============================================================================
# TEST: Model initialization without weights (shape validation)
# =============================================================================


def test_dreamzero_model_shapes():
    """Test that DreamZero model produces correct output shapes without pretrained weights."""
    torch.manual_seed(SEED)

    config = create_tiny_config()
    config.precision = "float32"  # Use float32 for CPU testing
    model = DreamZeroModel(config)
    model.eval()

    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)

    with torch.no_grad():
        actions = model.get_actions(
            video_latent=inputs["video_latent"],
            context=inputs["context"],
            clip_feature=inputs["clip_feature"],
            ref_latent=inputs["ref_latent"],
        )

    # Validate output shape
    expected_shape = (BATCH_SIZE, config.action_horizon, config.action_dim)
    assert actions.shape == expected_shape, (
        f"Expected action shape {expected_shape}, got {actions.shape}"
    )
    print(f"✅ Action output shape: {actions.shape}")


def test_dreamzero_model_deterministic():
    """Test that model produces deterministic outputs with fixed seed."""
    config = create_tiny_config()
    config.precision = "float32"
    model = DreamZeroModel(config)
    model.eval()

    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)

    # Run twice with same seed
    torch.manual_seed(SEED)
    with torch.no_grad():
        actions1 = model.get_actions(**inputs)

    torch.manual_seed(SEED)
    with torch.no_grad():
        actions2 = model.get_actions(**inputs)

    assert torch.allclose(actions1, actions2, atol=1e-5), "Model is not deterministic"
    print("✅ Model produces deterministic outputs")


def test_dreamzero_joint_video_action():
    """Test joint video + action prediction mode."""
    torch.manual_seed(SEED)

    config = create_tiny_config()
    config.precision = "float32"
    model = DreamZeroModel(config)
    model.eval()

    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)
    timestep = torch.tensor([500.0])

    with torch.no_grad():
        video_out, action_out = model.joint_video_action(
            video_latent=inputs["video_latent"],
            timestep=timestep,
            context=inputs["context"],
            clip_feature=inputs["clip_feature"],
            ref_latent=inputs["ref_latent"],
        )

    assert action_out.shape == (BATCH_SIZE, config.action_horizon, config.action_dim)
    print(f"✅ Joint prediction - video: {video_out.shape}, action: {action_out.shape}")


# =============================================================================
# TEST: Weight loading from pretrained checkpoint
# =============================================================================


def test_dreamzero_weight_loading():
    """Test loading pretrained weights into DreamZero model."""
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}. Run download_pretrained_weights.py first.")

    config = create_config()
    weight_loader = DreamZeroWeightLoader(str(checkpoint_path))

    # Verify weights loaded
    assert weight_loader.weight_info.num_parameters > 0, "No parameters loaded"
    print(f"✅ Loaded {weight_loader.weight_info.num_parameters:,} parameters")

    # Verify weight components are accessible
    dit_weights = weight_loader.get_dit_weights()
    assert len(dit_weights) > 0, "No DiT weights found"
    print(f"   DiT weights: {len(dit_weights)} tensors")

    # Initialize model with weights
    model = DreamZeroModel(config, weight_loader)
    model.eval()

    # Verify model runs
    inputs = create_test_inputs(config, batch_size=BATCH_SIZE)
    torch.manual_seed(SEED)
    with torch.no_grad():
        actions = model.get_actions(**inputs)

    assert actions.shape == (BATCH_SIZE, config.action_horizon, config.action_dim)
    print(f"✅ Model with pretrained weights produces actions: {actions.shape}")


def test_dreamzero_weight_loader_selective():
    """Test selective weight loading (e.g., DiT only, no VAE)."""
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")

    # Load without VAE and text encoder
    loader = DreamZeroWeightLoader(
        str(checkpoint_path),
        load_vae=False,
        load_text_encoder=False,
        load_clip=False,
    )

    # Should have no VAE weights
    vae_weights = loader.get_vae_weights()
    assert len(vae_weights) == 0, "VAE weights should not be loaded"

    # Should still have DiT weights
    dit_weights = loader.get_dit_weights()
    assert len(dit_weights) > 0, "DiT weights should be loaded"
    print(f"✅ Selective loading: {len(dit_weights)} DiT tensors, 0 VAE tensors")


def test_dreamzero_lora_loading():
    """Test LoRA weight loading."""
    checkpoint_path = Path(CHECKPOINT_PATH)
    lora_path = checkpoint_path / "lora"  # Expected LoRA directory

    if not checkpoint_path.exists():
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    if not lora_path.exists():
        pytest.skip(f"LoRA weights not found: {lora_path}")

    loader = DreamZeroWeightLoader(str(checkpoint_path))
    lora_weights = loader.load_lora_weights(str(lora_path))

    assert len(lora_weights) > 0, "No LoRA weights loaded"
    print(f"✅ Loaded {len(lora_weights)} LoRA weight tensors")


# =============================================================================
# TEST: Flow matching scheduler
# =============================================================================


def test_flow_match_scheduler():
    """Test flow matching scheduler correctness."""
    from models.experimental.dreamzero.reference.torch_flow_matching import FlowMatchScheduler

    scheduler = FlowMatchScheduler(num_inference_steps=10, shift=3.0)

    # Verify timestep schedule is monotonically decreasing
    assert len(scheduler.timesteps) == 10
    for i in range(len(scheduler.timesteps) - 1):
        assert scheduler.timesteps[i] > scheduler.timesteps[i + 1], (
            f"Timesteps not monotonically decreasing at index {i}"
        )

    # Verify denoising step reduces noise
    sample = torch.randn(1, 24, 7)
    velocity = torch.randn(1, 24, 7)

    step_result = scheduler.step(velocity, scheduler.timesteps[0], sample)
    assert step_result.shape == sample.shape

    # Verify noise addition
    clean = torch.ones(1, 24, 7)
    noise = torch.randn(1, 24, 7)
    noisy = scheduler.add_noise(clean, noise, scheduler.timesteps[0:1])
    assert noisy.shape == clean.shape
    # Should not be identical to the clean signal (noise was added)
    assert not torch.allclose(noisy, clean)

    print("✅ Flow matching scheduler works correctly")


# =============================================================================
# TEST: DiT block
# =============================================================================


def test_dit_block():
    """Test DiT block forward pass."""
    from models.experimental.dreamzero.reference.torch_dit_block import DiTBlock, RotaryPositionEmbedding3D

    dim = 1024  # Use dim that gives head_dim=128 (cleanly divisible for 3D RoPE)
    num_heads = 8
    ffn_dim = 2048
    batch_size = 1
    seq_len = 16  # 2*2*4 = 16

    block = DiTBlock(dim=dim, num_heads=num_heads, ffn_dim=ffn_dim, has_image_input=False)

    x = torch.randn(batch_size, seq_len, dim)
    context = torch.randn(batch_size, 32, dim)
    t_mod = torch.randn(batch_size, 6, dim)

    # Create RoPE frequencies (head_dim=128 splits cleanly: 44+42+42=128)
    rope = RotaryPositionEmbedding3D(num_heads=num_heads, head_dim=dim // num_heads)
    freqs = rope(f=2, h=2, w=4)  # 2*2*4 = 16 = seq_len

    output = block(x, context, t_mod, freqs)
    assert output.shape == x.shape, f"Expected {x.shape}, got {output.shape}"
    print(f"✅ DiT block output shape: {output.shape}")


# =============================================================================
# STANDALONE RUNNER
# =============================================================================


def main():
    """Run all tests standalone."""
    print("=" * 80)
    print("  DreamZero World Action Model - Reference Tests")
    print("=" * 80)

    tests = [
        ("Flow Match Scheduler", test_flow_match_scheduler),
        ("DiT Block", test_dit_block),
        ("Model Shapes", test_dreamzero_model_shapes),
        ("Model Deterministic", test_dreamzero_model_deterministic),
        ("Joint Video+Action", test_dreamzero_joint_video_action),
    ]

    # Add weight-dependent tests if checkpoint exists
    checkpoint_path = Path(CHECKPOINT_PATH)
    if checkpoint_path.exists():
        tests.extend([
            ("Weight Loading", test_dreamzero_weight_loading),
            ("Selective Loading", test_dreamzero_weight_loader_selective),
            ("LoRA Loading", test_dreamzero_lora_loading),
        ])
    else:
        print(f"\n⚠️  Checkpoint not found at: {checkpoint_path}")
        print("   Skipping weight-loading tests.")
        print("   Run download_pretrained_weights.py to enable full testing.\n")

    passed = 0
    failed = 0
    skipped = 0

    for name, test_fn in tests:
        print(f"\n--- {name} ---")
        try:
            test_fn()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"⏭️  Skipped: {e}")
            skipped += 1
        except Exception as e:
            print(f"❌ Failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 80)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 80)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
