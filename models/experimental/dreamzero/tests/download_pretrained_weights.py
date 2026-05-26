# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Download pretrained DreamZero weights from HuggingFace.

Usage:
    python download_pretrained_weights.py [--model-id GEAR-Dreams/DreamZero-DROID]
                                          [--output-dir ./weights/dreamzero_droid]
"""

import argparse
import os
import sys
from pathlib import Path


def download_weights(model_id: str, output_dir: str):
    """Download DreamZero pretrained weights from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id} to {output_path}...")
    print("This may take a while for the full 14B model (~28GB).")
    print()

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(output_path),
            repo_type="model",
        )
        print(f"\n✅ Download complete: {output_path}")
        print(f"   Use this path as checkpoint_path in tests.")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nTroubleshooting:")
        print("  1. Set HF_TOKEN environment variable if repo requires authentication")
        print("  2. Check your internet connection")
        print("  3. Verify the model ID exists on HuggingFace")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download DreamZero pretrained weights")
    parser.add_argument(
        "--model-id",
        type=str,
        default="GEAR-Dreams/DreamZero-DROID",
        help="HuggingFace model ID (default: GEAR-Dreams/DreamZero-DROID)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: $TT_METAL_HOME/models/experimental/dreamzero/weights/dreamzero_droid)",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        tt_metal_home = os.environ.get("TT_METAL_HOME", ".")
        model_name = args.model_id.split("/")[-1].lower().replace("-", "_")
        args.output_dir = os.path.join(
            tt_metal_home, "models", "experimental", "dreamzero", "weights", model_name
        )

    download_weights(args.model_id, args.output_dir)


if __name__ == "__main__":
    main()
