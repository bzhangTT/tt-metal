# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Latency benchmark for the Aurora TT-NN port on Blackhole.

Measures, on the real ``aurora-0.25-small-pretrained.ckpt``:
  - full hybrid model forward (host encoder + TT backbone + host decoder),
  - the TT backbone alone, eager vs. the trace-captured rollout runner.

Run (from repo root):
    python models/experimental/aurora/demo/benchmark.py [--grid H W] [--iters N]

NOTE on interpreting the numbers: this is a single Blackhole chip that may be
*shared* with other jobs on the host, and the default grid is the small test
resolution (the backbone is then dispatch-bound, which is exactly what the trace
runner removes). The end-to-end picture at Aurora's native 0.25 deg resolution is
different (the host encoder/decoder grow with the ~720x1440 grid); see
OPTIMIZATION.md. Treat these as relative, same-machine comparisons.
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import torch

import ttnn

from aurora import AuroraSmallPretrained, Batch, Metadata

from models.experimental.aurora.tt.model import attach_tt_backbone, attach_tt_perceiver
from models.experimental.aurora.tt.runner import TtBackboneRunner


def make_batch(h, w, levels=(100, 250, 500, 850), t=2):
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


def bench(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t) / iters * 1e3  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs=2, default=(32, 64), metavar=("H", "W"))
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()
    h, w = args.grid

    model = AuroraSmallPretrained()
    model.load_checkpoint()
    model.eval()
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=120_000_000)

    batch = make_batch(h, w)
    patch_res = (model.encoder.latent_levels, h // model.encoder.patch_size, w // model.encoder.patch_size)
    lead = timedelta(hours=6)

    # Full hybrid model (host enc + TT backbone + host dec).
    attach_tt_backbone(model, dev, use_lora=False)
    with torch.no_grad():
        full_ms = bench(lambda: model.forward(batch), args.iters)

    # + on-device Perceiver level (de)aggregation.
    attach_tt_perceiver(model, dev)
    with torch.no_grad():
        full_perc_ms = bench(lambda: model.forward(batch), args.iters)

    # Backbone alone: eager vs trace runner.
    bb = model.backbone.tt_backbone
    L = patch_res[0] * patch_res[1] * patch_res[2]
    x = torch.randn(1, L, bb.embed_dim)
    eager_ms = bench(lambda: bb(x, lead, patch_res), args.iters)
    runner = TtBackboneRunner(bb).capture(x, lead, patch_res)
    trace_ms = bench(lambda: runner.run(x), args.iters)
    runner.release()

    print("\n==== Aurora TT-NN latency (single Blackhole, possibly shared) ====")
    print(f"grid={h}x{w}  patch_res={patch_res}  embed_dim={bb.embed_dim}  iters={args.iters}")
    print(f"  full hybrid forward (TT backbone)            : {full_ms:8.1f} ms")
    print(f"  full hybrid forward (TT backbone + Perceiver): {full_perc_ms:8.1f} ms")
    print(f"  backbone eager  (per step)                   : {eager_ms:8.1f} ms")
    print(f"  backbone traced (per step, rollout runner)   : {trace_ms:8.1f} ms   ({eager_ms/trace_ms:.1f}x)")

    ttnn.close_mesh_device(dev)


if __name__ == "__main__":
    main()
