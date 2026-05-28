#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
PI0 Closed-Loop Simulation with ALOHA (MuJoCo)

Runs the PI0 TTNN model in a closed-loop evaluation with the gym-aloha
MuJoCo environment. The model observes the simulated robot's cameras,
predicts action chunks, and the robot executes them step by step.

Prerequisites:
    pip install gymnasium gym-aloha mujoco imageio[ffmpeg]

Usage:
    # Run with TT hardware (requires pretrained weights)
    python run_aloha_sim_closed_loop.py

    # Run with PyTorch reference only (no TT hardware needed)
    python run_aloha_sim_closed_loop.py --backend torch

    # Record video
    python run_aloha_sim_closed_loop.py --record-video

    # Multiple episodes
    python run_aloha_sim_closed_loop.py --num-episodes 5

    # Custom environment
    python run_aloha_sim_closed_loop.py --env-id gym_aloha/AlohaTransferCube-v0
"""

import argparse
import sys
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Demo folder location
DEMO_DIR = Path(__file__).parent

# Add parent paths for imports
sys.path.insert(0, str(DEMO_DIR.parent.parent.parent.parent))


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_ENV_ID = "gym_aloha/AlohaTransferCube-v0"
DEFAULT_TASK_PROMPT = "Transfer cube"
IMAGE_SIZE = 224  # SigLIP input size
MAX_EPISODE_STEPS = 400  # ALOHA sim default
ACTION_HORIZON = 50  # PI0 predicts 50-step action chunks
# ALOHA sim uses 14-DOF actions (7 per arm), PI0 predicts 32-dim
# We map the first 14 dims to the ALOHA action space
ALOHA_ACTION_DIM = 14
SEED = 42


# =============================================================================
# IMAGE PREPROCESSING
# =============================================================================


def preprocess_obs_image(obs_pixels: np.ndarray, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    Preprocess a raw observation image from the gym environment for PI0.

    Args:
        obs_pixels: Raw pixel observation from env, shape (H, W, 3), uint8
        image_size: Target image size for SigLIP

    Returns:
        Preprocessed image tensor, shape (1, 3, image_size, image_size), float32
    """
    from PIL import Image

    # Convert numpy array to PIL Image for consistent resizing
    img = Image.fromarray(obs_pixels)
    img = img.resize((image_size, image_size), Image.BILINEAR)

    # Convert to tensor and normalize to [-1, 1] (SigLIP preprocessing)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
    img_tensor = (img_tensor - 0.5) / 0.5  # Normalize to [-1, 1]

    return img_tensor.unsqueeze(0)  # Add batch dim


def tokenize_prompt(prompt: str, max_length: int = 32):
    """Simple tokenization for demo (same as run_aloha_sim_demo.py)."""
    tokens = [ord(char) % 256000 for char in prompt[:max_length]]
    while len(tokens) < max_length:
        tokens.append(0)
    tokens = torch.tensor([tokens[:max_length]], dtype=torch.long)
    mask = torch.ones(1, max_length, dtype=torch.bool)
    mask[0, len(prompt) :] = False
    return tokens, mask


# =============================================================================
# ACTION MAPPING
# =============================================================================


def map_pi0_actions_to_aloha(pi0_actions: np.ndarray) -> np.ndarray:
    """
    Map PI0's 32-dim action output to ALOHA's 14-DOF action space.

    PI0 action space (32-dim): model-internal representation that includes
    joint positions for both arms + gripper states + padding.

    ALOHA action space (14-dim): 7-DOF per arm
    [left_arm(6 joints + 1 gripper), right_arm(6 joints + 1 gripper)]

    We take the first 14 dimensions and clip to the valid range [-1, 1].
    """
    actions = pi0_actions[:ALOHA_ACTION_DIM]
    return np.clip(actions, -1.0, 1.0)


# =============================================================================
# MODEL BACKENDS
# =============================================================================


class PI0Backend:
    """Base class for PI0 inference backends."""

    def predict_actions(
        self,
        images: list,
        task_prompt: str,
        state: np.ndarray,
    ) -> np.ndarray:
        """
        Predict an action chunk given observations.

        Args:
            images: List of camera images (preprocessed tensors)
            task_prompt: Language instruction
            state: Current robot state (joint positions)

        Returns:
            Action chunk of shape (action_horizon, action_dim)
        """
        raise NotImplementedError


class PI0TorchBackend(PI0Backend):
    """PyTorch reference backend (runs on CPU, no TT hardware needed)."""

    def __init__(self, checkpoint_path: str, config=None):
        from models.experimental.pi0.reference.torch_pi0_model import PI0Model as PI0ModelTorch
        from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
        from models.experimental.pi0.common.weight_loader import PI0WeightLoader

        if config is None:
            config = PI0ModelConfig(
                action_dim=32,
                action_horizon=ACTION_HORIZON,
                state_dim=32,
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
                pi05=False,
            )
            config.siglip_config = SigLIPConfig(
                hidden_size=1152,
                intermediate_size=4304,
                num_hidden_layers=27,
                num_attention_heads=16,
                image_size=IMAGE_SIZE,
                patch_size=14,
            )

        weight_loader = PI0WeightLoader(checkpoint_path)
        self.model = PI0ModelTorch(config, weight_loader)
        self.config = config
        print(f"  ✅ PyTorch reference model loaded")

    def predict_actions(self, images, task_prompt, state):
        lang_tokens, lang_masks = tokenize_prompt(task_prompt)
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in images]
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # Pad state to model's state_dim
        if state_tensor.shape[-1] < self.config.state_dim:
            padding = torch.zeros(1, self.config.state_dim - state_tensor.shape[-1])
            state_tensor = torch.cat([state_tensor, padding], dim=-1)

        with torch.no_grad():
            actions = self.model.sample_actions(
                images=images,
                img_masks=img_masks,
                lang_tokens=lang_tokens,
                lang_masks=lang_masks,
                state=state_tensor,
            )

        return actions.squeeze(0).numpy()  # (action_horizon, action_dim)


class PI0TTNNBackend(PI0Backend):
    """TTNN backend (runs on Tenstorrent hardware)."""

    def __init__(self, checkpoint_path: str, device_id: int = 0, config=None):
        import ttnn

        from models.experimental.pi0.tt.ttnn_pi0_model import PI0ModelTTNN
        from models.experimental.pi0.common.configs import PI0ModelConfig, SigLIPConfig
        from models.experimental.pi0.common.weight_loader import PI0WeightLoader

        if config is None:
            config = PI0ModelConfig(
                action_dim=32,
                action_horizon=ACTION_HORIZON,
                state_dim=32,
                paligemma_variant="gemma_2b",
                action_expert_variant="gemma_300m",
                pi05=False,
            )
            config.siglip_config = SigLIPConfig(
                hidden_size=1152,
                intermediate_size=4304,
                num_hidden_layers=27,
                num_attention_heads=16,
                image_size=IMAGE_SIZE,
                patch_size=14,
            )

        self.device = ttnn.open_device(device_id=device_id, l1_small_size=24576)
        weight_loader = PI0WeightLoader(checkpoint_path)

        torch.manual_seed(SEED)
        self.model = PI0ModelTTNN(config, weight_loader, self.device)
        self.config = config
        self.ttnn = ttnn
        print(f"  ✅ TTNN model loaded on device {device_id}")

    def predict_actions(self, images, task_prompt, state):
        ttnn = self.ttnn
        lang_tokens, lang_masks = tokenize_prompt(task_prompt)
        img_masks = [torch.ones(1, dtype=torch.bool) for _ in images]
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)

        # Pad state to model's state_dim
        if state_tensor.shape[-1] < self.config.state_dim:
            padding = torch.zeros(1, self.config.state_dim - state_tensor.shape[-1])
            state_tensor = torch.cat([state_tensor, padding], dim=-1)

        # Convert to TTNN tensors
        images_ttnn = [
            ttnn.from_torch(
                img,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=self.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for img in images
        ]
        lang_tokens_ttnn = ttnn.from_torch(
            lang_tokens,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
        )
        lang_masks_ttnn = ttnn.from_torch(
            lang_masks.float(),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )
        state_ttnn = ttnn.from_torch(
            state_tensor,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
        )

        with torch.no_grad():
            actions_ttnn = self.model.sample_actions(
                images=images_ttnn,
                img_masks=img_masks,
                lang_tokens=lang_tokens_ttnn,
                lang_masks=lang_masks_ttnn,
                state=state_ttnn,
            )

        # Convert back to numpy
        if isinstance(actions_ttnn, ttnn.Tensor):
            actions_torch = ttnn.to_torch(actions_ttnn)
        else:
            actions_torch = actions_ttnn

        return actions_torch.squeeze(0).float().numpy()  # (action_horizon, action_dim)

    def close(self):
        self.ttnn.close_device(self.device)


# =============================================================================
# SIMULATION LOOP
# =============================================================================


def run_episode(
    env,
    backend: PI0Backend,
    task_prompt: str,
    max_steps: int = MAX_EPISODE_STEPS,
    action_repeat: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Run a single closed-loop episode.

    The model predicts action chunks of size ACTION_HORIZON. We execute
    each action in the chunk sequentially before requesting a new prediction.

    Args:
        env: Gymnasium environment
        backend: PI0 inference backend
        task_prompt: Language instruction for the task
        max_steps: Maximum environment steps
        action_repeat: How many times to repeat each action
        verbose: Print step-by-step info

    Returns:
        Episode metrics dict
    """
    obs, info = env.reset()
    total_reward = 0.0
    steps = 0
    inference_times = []
    done = False

    if verbose:
        print(f"\n  🎬 Episode started (max {max_steps} steps)")
        print(f'  📝 Task: "{task_prompt}"')

    while not done and steps < max_steps:
        # Extract camera images from observation
        # gym-aloha provides obs as dict with 'pixels' key containing camera images
        # or as a dict with individual camera keys
        images = _extract_images_from_obs(obs)

        # Extract robot state (joint positions) if available
        state = _extract_state_from_obs(obs)

        # Run PI0 inference to get action chunk
        t0 = time.time()
        action_chunk = backend.predict_actions(images, task_prompt, state)
        inference_time = time.time() - t0
        inference_times.append(inference_time)

        if verbose and steps == 0:
            print(f"  ⚡ First inference: {inference_time*1000:.1f}ms")
            print(f"  📊 Action chunk shape: {action_chunk.shape}")

        # Execute action chunk step-by-step
        chunk_steps = min(ACTION_HORIZON, max_steps - steps)
        for j in range(chunk_steps):
            # Map PI0's 32-dim output to ALOHA's 14-DOF action space
            action = map_pi0_actions_to_aloha(action_chunk[j])

            # Repeat action for temporal consistency
            for _ in range(action_repeat):
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

                if done:
                    break
            if done:
                break

        if verbose and steps % 50 == 0:
            avg_hz = 1.0 / np.mean(inference_times[-5:]) if inference_times else 0
            print(f"  📍 Step {steps}/{max_steps} | Reward: {total_reward:.3f} | Avg Hz: {avg_hz:.1f}")

    # Compute metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    metrics = {
        "total_reward": total_reward,
        "steps": steps,
        "success": info.get("is_success", False) if info else False,
        "avg_inference_ms": avg_inference_time * 1000,
        "avg_hz": 1.0 / avg_inference_time if avg_inference_time > 0 else 0,
        "num_inferences": len(inference_times),
    }

    if verbose:
        status = "✅ SUCCESS" if metrics["success"] else "❌ FAILED"
        print(f"\n  {status} | Steps: {steps} | Reward: {total_reward:.3f}")
        print(f"  ⚡ Avg inference: {metrics['avg_inference_ms']:.1f}ms ({metrics['avg_hz']:.1f} Hz)")

    return metrics


def _extract_images_from_obs(obs) -> list:
    """Extract and preprocess camera images from gym observation."""
    images = []

    if isinstance(obs, dict):
        # gym-aloha typically provides observations as a dict
        # Look for pixel/image keys
        pixel_keys = [k for k in obs.keys() if "image" in k.lower() or "pixel" in k.lower() or "rgb" in k.lower()]

        if not pixel_keys:
            # Some envs put pixels under 'observation' -> 'pixels'
            if "pixels" in obs:
                pixel_data = obs["pixels"]
                if isinstance(pixel_data, dict):
                    pixel_keys = list(pixel_data.keys())
                    for key in sorted(pixel_keys)[:2]:
                        img_tensor = preprocess_obs_image(pixel_data[key])
                        images.append(img_tensor)
                elif isinstance(pixel_data, np.ndarray):
                    img_tensor = preprocess_obs_image(pixel_data)
                    images.append(img_tensor)
            elif "observation" in obs and isinstance(obs["observation"], dict):
                inner = obs["observation"]
                pixel_keys = [
                    k for k in inner.keys() if "image" in k.lower() or "pixel" in k.lower() or "rgb" in k.lower()
                ]
                for key in sorted(pixel_keys)[:2]:
                    img_tensor = preprocess_obs_image(inner[key])
                    images.append(img_tensor)
        else:
            for key in sorted(pixel_keys)[:2]:
                img_tensor = preprocess_obs_image(obs[key])
                images.append(img_tensor)

    elif isinstance(obs, np.ndarray):
        # Raw pixel observation
        if obs.ndim == 3:  # Single image (H, W, C)
            img_tensor = preprocess_obs_image(obs)
            images.append(img_tensor)
        elif obs.ndim == 4:  # Multiple images (N, H, W, C)
            for i in range(min(2, obs.shape[0])):
                img_tensor = preprocess_obs_image(obs[i])
                images.append(img_tensor)

    # PI0 expects exactly 2 images; duplicate if we only have 1
    if len(images) == 0:
        # Fallback: create a dummy black image
        dummy = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        images.append(preprocess_obs_image(dummy))

    while len(images) < 2:
        images.append(images[0].clone())

    return images[:2]


def _extract_state_from_obs(obs) -> np.ndarray:
    """Extract robot state (joint positions) from gym observation."""
    if isinstance(obs, dict):
        # Look for state/qpos keys
        for key in ["agent_pos", "qpos", "state", "joint_positions", "robot_state"]:
            if key in obs:
                state = np.asarray(obs[key], dtype=np.float32).flatten()
                return state
        # Check nested observation
        if "observation" in obs and isinstance(obs["observation"], dict):
            inner = obs["observation"]
            for key in ["agent_pos", "qpos", "state", "joint_positions"]:
                if key in inner:
                    state = np.asarray(inner[key], dtype=np.float32).flatten()
                    return state

    # Fallback: zero state
    return np.zeros(ALOHA_ACTION_DIM, dtype=np.float32)


# =============================================================================
# VIDEO RECORDING
# =============================================================================


def make_video_env(env_id: str, video_dir: str, episode_trigger=None):
    """Wrap environment with video recording."""
    import gymnasium as gym

    env = gym.make(env_id, render_mode="rgb_array")

    env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=episode_trigger or (lambda ep: True),  # Record all episodes
        name_prefix="pi0_aloha",
    )
    return env


# =============================================================================
# MAIN
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="PI0 Closed-Loop Robotics Simulation (ALOHA + MuJoCo)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with TTNN backend (requires TT hardware + pretrained weights)
  python run_aloha_sim_closed_loop.py

  # Run with PyTorch reference (no hardware needed)
  python run_aloha_sim_closed_loop.py --backend torch

  # Record video of episodes
  python run_aloha_sim_closed_loop.py --record-video --num-episodes 3

  # Use a different task environment
  python run_aloha_sim_closed_loop.py --env-id gym_aloha/AlohaInsertionPeg-v0 --task "Insert peg"
""",
    )
    parser.add_argument(
        "--backend",
        choices=["ttnn", "torch"],
        default="ttnn",
        help="Inference backend: 'ttnn' (TT hardware) or 'torch' (CPU reference)",
    )
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID, help=f"Gym environment ID (default: {DEFAULT_ENV_ID})")
    parser.add_argument("--task", default=DEFAULT_TASK_PROMPT, help=f'Task prompt (default: "{DEFAULT_TASK_PROMPT}")')
    parser.add_argument("--num-episodes", type=int, default=1, help="Number of episodes to run (default: 1)")
    parser.add_argument("--max-steps", type=int, default=MAX_EPISODE_STEPS, help="Max steps per episode")
    parser.add_argument("--record-video", action="store_true", help="Record video of episodes")
    parser.add_argument("--video-dir", default=None, help="Video output directory")
    parser.add_argument("--device-id", type=int, default=0, help="TT device ID (default: 0)")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--weights", default=None, help="Path to PI0 weights (auto-detected if not set)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  🤖 PI0 CLOSED-LOOP ROBOTICS SIMULATION")
    print("     Model: PI0 (Vision-Language-Action)")
    print("     Environment: ALOHA Sim (MuJoCo)")
    print("=" * 70)

    # ─── Check dependencies ───────────────────────────────────────────────
    try:
        import gymnasium as gym  # noqa: F401
    except ImportError:
        print("\n❌ gymnasium not installed. Install with:")
        print("   pip install gymnasium")
        sys.exit(1)

    try:
        import gym_aloha  # noqa: F401
    except ImportError:
        print("\n❌ gym-aloha not installed. Install with:")
        print("   pip install gym-aloha")
        print("   (This provides the ALOHA MuJoCo simulation environments)")
        sys.exit(1)

    # ─── Resolve weights path ─────────────────────────────────────────────
    if args.weights:
        checkpoint_path = args.weights
    else:
        tt_metal_home = os.environ.get("TT_METAL_HOME", "")
        checkpoint_path = os.path.join(tt_metal_home, "models/experimental/pi0/weights/pi0_base")

    if not Path(checkpoint_path).exists():
        print(f"\n❌ Weights not found: {checkpoint_path}")
        print("   Download with: python models/experimental/pi0/tests/download_pretrained_weights.py")
        sys.exit(1)

    # ─── Initialize backend ───────────────────────────────────────────────
    print(f"\n📦 Loading PI0 model (backend: {args.backend})...")

    if args.backend == "ttnn":
        backend = PI0TTNNBackend(checkpoint_path, device_id=args.device_id)
    else:
        backend = PI0TorchBackend(checkpoint_path)

    # ─── Create environment ───────────────────────────────────────────────
    print(f"\n🌍 Creating environment: {args.env_id}")
    import gymnasium as gym

    if args.record_video:
        video_dir = args.video_dir or str(DEMO_DIR / "videos")
        print(f"  📹 Recording video to: {video_dir}")
        env = make_video_env(args.env_id, video_dir)
    else:
        env = gym.make(args.env_id)

    # Seed for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ─── Run episodes ─────────────────────────────────────────────────────
    print(f"\n🎮 Running {args.num_episodes} episode(s)...")
    print("-" * 70)

    all_metrics = []
    for ep in range(args.num_episodes):
        print(f"\n{'─'*40}")
        print(f"  Episode {ep + 1}/{args.num_episodes}")
        print(f"{'─'*40}")

        metrics = run_episode(
            env=env,
            backend=backend,
            task_prompt=args.task,
            max_steps=args.max_steps,
            verbose=not args.quiet,
        )
        all_metrics.append(metrics)

    # ─── Summary ──────────────────────────────────────────────────────────
    env.close()

    print("\n" + "=" * 70)
    print("  📊 EVALUATION SUMMARY")
    print("=" * 70)

    num_success = sum(1 for m in all_metrics if m["success"])
    avg_reward = np.mean([m["total_reward"] for m in all_metrics])
    avg_steps = np.mean([m["steps"] for m in all_metrics])
    avg_hz = np.mean([m["avg_hz"] for m in all_metrics])
    avg_inference_ms = np.mean([m["avg_inference_ms"] for m in all_metrics])

    print(f"\n  Environment: {args.env_id}")
    print(f'  Task:        "{args.task}"')
    print(f"  Backend:     {args.backend}")
    print(f"  Episodes:    {args.num_episodes}")
    print(f"\n  Results:")
    print(f"    Success Rate:      {num_success}/{args.num_episodes} ({100*num_success/args.num_episodes:.0f}%)")
    print(f"    Avg Reward:        {avg_reward:.3f}")
    print(f"    Avg Steps:         {avg_steps:.0f}")
    print(f"    Avg Inference:     {avg_inference_ms:.1f} ms")
    print(f"    Avg Control Freq:  {avg_hz:.1f} Hz")

    if args.record_video:
        video_dir = args.video_dir or str(DEMO_DIR / "videos")
        print(f"\n  📹 Videos saved to: {video_dir}")

    print("\n" + "=" * 70)

    # Cleanup TTNN device
    if args.backend == "ttnn" and hasattr(backend, "close"):
        backend.close()

    # Return success if any episode succeeded, or 0 for "ran without error"
    return 0


if __name__ == "__main__":
    sys.exit(main())
