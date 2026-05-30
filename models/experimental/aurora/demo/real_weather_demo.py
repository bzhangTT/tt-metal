# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Real-weather inference: actual ERA5/HRES fields in -> real forecast out.

Downloads a REAL weather batch (WeatherBench2 HRES T0 on Google Cloud, no auth)
for a chosen day, spatially coarsens it (so the full 1.3B model is tractable on
CPU + a single Blackhole chip while keeping real physical values), runs the real
pretrained Aurora with the TT-NN backbone on Blackhole, and checks:

  1. the forecast is physically plausible (2t in Kelvin, sane global mean),
  2. the 6h forecast is highly correlated with the input (weather persists),
  3. the TT-NN backbone matches the PyTorch reference (PCC).

This is the entry point for running on *your own* real data: pick any day in the
2016-2022 HRES-T0 range and any coarsening stride.

Usage (repo root, in venv):
    python models/experimental/aurora/demo/real_weather_demo.py
    python models/experimental/aurora/demo/real_weather_demo.py --day 2021-01-15
    python models/experimental/aurora/demo/real_weather_demo.py --day 2022-05-11 --steps 4
"""

import argparse
import pickle
from pathlib import Path

import fsspec
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

import ttnn

from aurora import AuroraPretrained, Batch, Metadata, rollout

from models.experimental.aurora.tt.common import pcc
from models.experimental.aurora.tt.model import attach_tt_backbone

# coarsen 0.25deg -> ~2deg: stride 9 gives 721->81 lat (Aurora crops to 80) and
# 1440->160 lon, which keeps the full 1.3B model tractable on CPU.
DEFAULT_STRIDE = 9
DEFAULT_DAY = "2022-05-11"
HRES_T0_URL = "gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr"

SURF = {
    "2t": "2m_temperature",
    "10u": "10m_u_component_of_wind",
    "10v": "10m_v_component_of_wind",
    "msl": "mean_sea_level_pressure",
}
ATMOS = {
    "t": "temperature",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "q": "specific_humidity",
    "z": "geopotential",
}


def load_real_batch(day=DEFAULT_DAY, stride=DEFAULT_STRIDE, cache=Path("~/downloads/hres_t0_demo").expanduser()):
    """Build an Aurora ``Batch`` from real WeatherBench2 HRES-T0 fields for ``day``."""
    cache.mkdir(parents=True, exist_ok=True)
    print(f"Opening real WeatherBench2 HRES T0 zarr for {day} (coarsen stride={stride}) ...")
    # Public bucket -> anonymous access (no Google credentials needed).
    mapper = fsspec.get_mapper(HRES_T0_URL, token="anon")
    ds = xr.open_zarr(mapper, chunks=None)
    ds = ds.sel(time=day).isel(latitude=slice(None, None, stride), longitude=slice(None, None, stride))

    print("Downloading real surface + atmospheric fields ...")
    surf_ds = ds[list(SURF.values())].compute()
    atmos_ds = ds[list(ATMOS.values())].compute()

    # Real static variables (land-sea mask, orography, soil type) from HuggingFace.
    static_path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-static.pickle")
    with open(static_path, "rb") as f:
        static = pickle.load(f)
    static = {k: v[::stride, ::stride] for k, v in static.items()}

    i = 2  # use times (i-1, i) as the 2-step history Aurora expects

    def prep(x):
        # Aurora wants latitudes north->south, so flip the lat axis (..., ::-1, :).
        return torch.from_numpy(x[[i - 1, i]][None][..., ::-1, :].copy())

    return Batch(
        surf_vars={k: prep(surf_ds[v].values) for k, v in SURF.items()},
        static_vars={k: torch.from_numpy(static[k][::-1].copy()) for k in ("lsm", "z", "slt")},
        atmos_vars={k: prep(atmos_ds[v].values) for k, v in ATMOS.items()},
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values[::-1].copy()),
            lon=torch.from_numpy(surf_ds.longitude.values),
            time=(surf_ds.time.values.astype("datetime64[s]").tolist()[i],),
            atmos_levels=tuple(int(level) for level in atmos_ds.level.values),
        ),
    )


def describe(name, field):
    k = field.float()
    print(
        f"  {name}: shape={tuple(k.shape)} min={k.min():.1f}K mean={k.mean():.1f}K "
        f"max={k.max():.1f}K  (={k.mean() - 273.15:.1f}C mean)"
    )


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--day", default=DEFAULT_DAY, help="UTC day to forecast, YYYY-MM-DD (2016-2022).")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Coarsening stride; 9 ~= 2 degrees.")
    p.add_argument("--steps", type=int, default=1, help="Number of 6h autoregressive rollout steps.")
    return p.parse_args()


def main():
    args = parse_args()
    batch = load_real_batch(args.day, args.stride)
    print(
        f"\nReal input grid: lat={len(batch.metadata.lat)} lon={len(batch.metadata.lon)} "
        f"levels={batch.metadata.atmos_levels} time={batch.metadata.time}"
    )
    describe("input  2t", batch.surf_vars["2t"][0, -1])

    print("\nLoading real Aurora 0.25 pretrained (1.3B) weights ...")
    model = AuroraPretrained()
    model.load_checkpoint()
    model.eval()

    # Reference single-step forecast (PyTorch, CPU) for the correctness check.
    print("Running PyTorch reference forecast (CPU) ...")
    with torch.no_grad():
        ref = model.forward(batch)
    describe("ref 6h 2t", ref.surf_vars["2t"][0])

    print(f"\nAttaching TT-NN backbone and running {args.steps}x6h on Blackhole ...")
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        attach_tt_backbone(model, device, use_lora=False)
        with torch.no_grad():
            # rollout() replays the whole backbone each 6h step; the device mask
            # cache makes every step after the first reuse its uploaded masks.
            preds = [p.to("cpu") for p in rollout(model, batch, steps=args.steps)]
    finally:
        ttnn.close_mesh_device(device)
    for step, p in enumerate(preds, start=1):
        describe(f"TT {6 * step:>2}h 2t", p.surf_vars["2t"][0])
    tt_6h = preds[0]

    # Physical-plausibility & persistence checks on the reference forecast.
    H, W = ref.surf_vars["2t"].shape[-2:]
    in2t = batch.surf_vars["2t"][0, -1].float()[:H, :W]  # crop input to forecast grid
    ref2t = ref.surf_vars["2t"][0].float()
    persist_corr = pcc(in2t, ref2t)
    print("\n=== checks ===")
    print(
        f"  forecast 2t global mean = {ref2t.mean():.1f} K "
        f"({'PLAUSIBLE' if 230 < ref2t.mean() < 310 else 'IMPLAUSIBLE'})"
    )
    print(
        f"  6h persistence corr (input 2t vs forecast 2t) = {persist_corr:.4f} "
        f"({'OK' if persist_corr > 0.9 else 'LOW'})"
    )
    ps = [pcc(ref.surf_vars[k], tt_6h.surf_vars[k]) for k in ref.surf_vars]
    ps += [pcc(ref.atmos_vars[k], tt_6h.atmos_vars[k]) for k in ref.atmos_vars]
    print(f"  TT-vs-reference worst-variable PCC (6h) = {min(ps):.4f}")


if __name__ == "__main__":
    main()
