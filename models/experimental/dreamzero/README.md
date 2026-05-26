# DreamZero: World Action Models are Zero-shot Policies

PyTorch reference implementation of the DreamZero World Action Model for the Tenstorrent platform.

**Paper:** [World Action Models are Zero-shot Policies](https://arxiv.org/abs/2602.15922) (arXiv:2602.15922)  
**Original Code:** [github.com/dreamzero0/dreamzero](https://github.com/dreamzero0/dreamzero)  
**Checkpoint:** [GEAR-Dreams/DreamZero-DROID](https://huggingface.co/GEAR-Dreams/DreamZero-DROID)

## Overview

DreamZero is a World Action Model (WAM) from NVIDIA GEAR Lab that **jointly predicts actions and videos** for robot control. Built on the Wan2.1-I2V-14B video diffusion backbone, it uses flow matching to denoise action tokens through shared transformer layers, achieving strong zero-shot performance on unseen robot tasks.

### Key Architecture

```
[Video Observations] → [VAE Encoder] → Video Latents ─┐
[Language Instruction] → [UMT5-XXL] → Text Embeddings ─┼─→ [DiT Backbone] → Action Tokens → [Action MLP] → Robot Actions
[Camera Images] → [CLIP] → Image Features ────────────┘         ↑
                                                          [Flow Matching Denoising]
```

- **Backbone:** Wan2.1-I2V-14B Video Diffusion Transformer (40 layers, 5120 dim, 40 heads)
- **Action Head:** Flow matching with 10 denoising steps
- **Input:** 3 camera views, 33 frames at 176×320 resolution
- **Output:** 24-step action horizon, 7-DOF (6-DOF + gripper)

## Directory Structure

```
models/experimental/dreamzero/
├── common/
│   ├── configs.py          # Model configurations (DreamZeroConfig, WanDiTConfig, etc.)
│   └── weight_loader.py    # Pretrained weight loading (safetensors, sharded, LoRA)
├── reference/
│   ├── torch_dreamzero_model.py  # Complete model orchestrator
│   ├── torch_dit_block.py        # DiT block with 3D RoPE, cross-attention
│   ├── torch_action_head.py      # Action head + WanDiT backbone
│   └── torch_flow_matching.py    # Flow matching scheduler
├── tests/
│   ├── download_pretrained_weights.py  # HuggingFace weight downloader
│   └── pcc/
│       └── test_pcc_dreamzero_model.py # Shape, determinism, and weight loading tests
└── README.md
```

## Quick Start

### 1. Download Pretrained Weights

```bash
# Option A: Using the download script
python models/experimental/dreamzero/tests/download_pretrained_weights.py

# Option B: Using huggingface-cli directly
huggingface-cli download GEAR-Dreams/DreamZero-DROID \
    --local-dir $TT_METAL_HOME/models/experimental/dreamzero/weights/dreamzero_droid
```

### 2. Run Tests

```bash
# Shape validation tests (no weights needed)
pytest models/experimental/dreamzero/tests/pcc/test_pcc_dreamzero_model.py -v -k "shapes or deterministic or scheduler or dit_block"

# Full tests with pretrained weights
pytest models/experimental/dreamzero/tests/pcc/test_pcc_dreamzero_model.py -v

# Standalone test runner
python models/experimental/dreamzero/tests/pcc/test_pcc_dreamzero_model.py
```

### 3. Use the Model

```python
import torch
from models.experimental.dreamzero.common.configs import DreamZeroConfig
from models.experimental.dreamzero.common.weight_loader import DreamZeroWeightLoader
from models.experimental.dreamzero.reference.torch_dreamzero_model import DreamZeroModel

# Initialize
config = DreamZeroConfig.droid()
weight_loader = DreamZeroWeightLoader("path/to/checkpoint")
model = DreamZeroModel(config, weight_loader)
model.eval()

# Create inputs (normally from VAE/CLIP/text encoder)
video_latent = torch.randn(1, 16, 9, 22, 40, dtype=torch.bfloat16)
context = torch.randn(1, 512, 4096, dtype=torch.bfloat16)
clip_feature = torch.randn(1, 257, 1280, dtype=torch.bfloat16)

# Generate actions
with torch.no_grad():
    actions = model.get_actions(
        video_latent=video_latent,
        context=context,
        clip_feature=clip_feature,
    )
print(f"Actions: {actions.shape}")  # (1, 24, 7)
```

## Model Configurations

| Config | Backbone | Parameters | Action Dim | Horizon |
|--------|----------|-----------|------------|---------|
| `DreamZeroConfig.droid()` | Wan2.1-14B | ~14B | 7 | 24 |
| `DreamZeroConfig.droid_5b()` | Wan2.2-5B | ~5B | 7 | 24 |

## Weight Loading

The weight loader supports multiple formats:

- **Single file:** `model.safetensors`
- **Sharded:** `model.safetensors.index.json` + shard files
- **LoRA:** Separate LoRA weight directory
- **Selective loading:** Load only DiT, skip VAE/CLIP/text encoder

## Citation

```bibtex
@misc{ye2026worldactionmodelszeroshot,
    title={World Action Models are Zero-shot Policies},
    author={Seonghyeon Ye and Yunhao Ge and Kaiyuan Zheng and others},
    year={2026},
    eprint={2602.15922},
    archivePrefix={arXiv},
    primaryClass={cs.RO},
}
```
