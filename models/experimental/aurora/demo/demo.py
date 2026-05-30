# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""End-to-end Aurora inference demo on Tenstorrent Blackhole.

Loads the real ``aurora-0.25-small-pretrained.ckpt`` weights, runs one 6-hour
forecast step with the TT-NN-accelerated Swin backbone, and reports the
agreement with the pure-PyTorch reference.

Usage (from repo root, inside the tt-metal venv):
    python models/experimental/aurora/demo/demo.py
"""

from datetime import datetime, timedelta

import torch

import ttnn

from aurora import AuroraSmallPretrained, Batch, Metadata

from models.experimental.aurora.tt.common import pcc
from models.experimental.aurora.tt.model import attach_tt_backbone


def make_batch(h=32, w=64, levels=(100, 250, 500, 850), t=2):
    surf = ("2t", "10u", "10v", "msl")
    atmos = ("z", "u", "v", "t", "q")
    static = ("lsm", "z", "slt")
    torch.manual_seed(0)
    return Batch(
        surf_vars={k: torch.randn(1, t, h, w) for k in surf},
        static_vars={k: torch.randn(h, w) for k in static},
        atmos_vars={k: torch.randn(1, t, len(levels), h, w) for k in atmos},
        metadata=Metadata(
            lat=torch.linspace(90, -90, h),
            lon=torch.linspace(0, 360, w + 1)[:-1],
            time=(datetime(2020, 6, 1, 12, 0),),
            atmos_levels=levels,
        ),
    )


def main():
    print("Loading real Aurora small pretrained checkpoint ...")
    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval()

    batch = make_batch()

    print("Running reference (PyTorch, CPU) forward ...")
    with torch.no_grad():
        ref = model.forward(batch)

    print("Opening Blackhole mesh device and attaching TT-NN backbone ...")
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        attach_tt_backbone(model, device, use_lora=False)
        print("Running TT-NN-accelerated forward on Blackhole ...")
        with torch.no_grad():
            out = model.forward(batch)
    finally:
        ttnn.close_mesh_device(device)

    print("\n=== 6h forecast agreement (TT-NN backbone vs PyTorch reference) ===")
    for k in ref.surf_vars:
        print(f"  surf  {k:>4}: PCC={pcc(ref.surf_vars[k], out.surf_vars[k]):.5f}")
    for k in ref.atmos_vars:
        print(f"  atmos {k:>4}: PCC={pcc(ref.atmos_vars[k], out.atmos_vars[k]):.5f}")
    print("\nExample predicted 2-metre temperature field (first 4x4 patch):")
    print(out.surf_vars["2t"][0, :4, :4])


if __name__ == "__main__":
    main()
