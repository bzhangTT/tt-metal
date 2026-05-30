# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""WeatherBench-style skill verification of Aurora on Tenstorrent Blackhole.

Computes the headline WeatherBench2 metric -- *latitude-weighted RMSE* -- of an
Aurora 6 h forecast against ERA5 ground truth, for the standard headline
variables (z500, t850, q700, 2t, msl, 10u, 10v).  It reports three forecasts on
the same grid:

  * persistence  -- the trivial "tomorrow = today" baseline,
  * reference    -- Aurora's PyTorch backbone (CPU),
  * TT (Blackhole) -- Aurora with the TT-NN Swin backbone on device.

What this verifies:
  1. Aurora has genuine forecast skill: its RMSE beats persistence (skill > 0).
  2. Running the backbone on Tenstorrent hardware preserves that skill: the TT
     RMSE matches the reference RMSE to ~3 decimals.

Truth and init both come from the public WeatherBench2 ERA5 zarr (the dataset
``AuroraPretrained`` was trained on), so the comparison is apples-to-apples.

Run at native 0.25 deg (``--stride 1``) for real forecast skill.  Aurora is
trained at 0.25 deg, so area-averaging the input coarser (``--stride > 1``)
pushes it out of distribution and the skill collapses -- it barely beats
persistence at 0.5 deg and loses to it at ~2 deg.  This is a property of the
model, not the port: the TT-vs-reference RMSE agreement holds at every
resolution (within ~0.5%), so coarse grids are only useful as a fast way to
check that fidelity, not the forecast skill.  By default only the TT (Blackhole)
forecast runs, so native resolution stays tractable on a single chip; pass
``--ref`` to also run the (slow) CPU reference backbone and report the
TT-vs-reference RMSE gap (the per-block PCC tests pin that fidelity too).

This is a single-init verification demonstrating genuine skill on hardware, not
a reproduction of the official WeatherBench2 leaderboard (which averages many
inits with a prescribed climatology and regridding).

Usage (repo root, in venv):
    python models/experimental/aurora/demo/weatherbench_eval.py                    # native 0.25 deg, TT only
    python models/experimental/aurora/demo/weatherbench_eval.py --stride 2 --ref   # 0.5 deg, with CPU reference
"""

import argparse
import pickle

import fsspec
import numpy as np
import torch
import xarray as xr
from huggingface_hub import hf_hub_download

import ttnn

from aurora import AuroraPretrained, Batch, Metadata

from models.experimental.aurora.tt.model import attach_tt_backbone

ERA5_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
DEFAULT_INIT = "2021-07-01T12:00"  # ERA5 zarr covers 1959-01-01 .. 2021-12-31T18
DEFAULT_STRIDE = 1  # native 0.25 deg -- the resolution at which Aurora has skill

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

# (label, surf var | (atmos var, pressure level hPa), unit, display scale)
HEADLINE = [
    ("z500", ("z", 500), "m^2/s^2", 1.0),
    ("t850", ("t", 850), "K", 1.0),
    ("q700", ("q", 700), "g/kg", 1000.0),
    ("2t", "2t", "K", 1.0),
    ("msl", "msl", "Pa", 1.0),
    ("10u", "10u", "m/s", 1.0),
    ("10v", "10v", "m/s", 1.0),
]


def load_era5_window(init, stride):
    """Load ERA5 init history (t-6h, t) + verification truth (t+6h), coarsened.

    ERA5 latitude already runs north->south (90 -> -90), matching Aurora's
    convention and the static-variable grid, so no latitude flips are needed.
    """
    print(f"Opening WeatherBench2 ERA5 zarr; init={init}, verify=+6h (coarsen factor={stride}) ...")
    ds = xr.open_zarr(fsspec.get_mapper(ERA5_URL, token="anon"), chunks=None)
    # Select the init step by exact integer index (robust to datetime64 unit
    # mismatch in .sel), then take (t-6h, t, t+6h) as history + verification.
    all_times = ds.time.values
    t = np.datetime64(init, "ns")
    idx = int(np.searchsorted(all_times, t))
    if idx >= len(all_times) or all_times[idx] != t:
        raise ValueError(f"init {init} not a 6h ERA5 step in [{all_times[0]} .. {all_times[-1]}]")
    if idx < 1 or idx + 1 >= len(all_times):
        raise ValueError(f"init {init} too close to the dataset edge for a (t-6h, t, t+6h) window")
    sub = ds.isel(time=[idx - 1, idx, idx + 1])
    if stride > 1:
        # Conservative area-averaging (mean over coarsen blocks), matching
        # WeatherBench2's regridding -- not subsampling, which aliases the field.
        sub = sub.coarsen(latitude=stride, longitude=stride, boundary="trim").mean()
    print("Downloading real ERA5 surface + atmospheric fields (3 x 6h) ...")
    surf_ds = sub[list(SURF.values())].compute()
    atmos_ds = sub[list(ATMOS.values())].compute()

    # Real static variables (land-sea mask, orography, soil type), area-averaged
    # onto the same coarsened grid (xarray coarsen trims the trailing partial
    # block, so block_mean below must trim identically to stay aligned).
    static_path = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-static.pickle")
    with open(static_path, "rb") as f:
        static = pickle.load(f)

    def block_mean(a, f):
        if f <= 1:
            return a
        h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f  # trim trailing partial block
        return a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))

    static = {k: block_mean(np.asarray(v), stride) for k, v in static.items()}

    levels = tuple(int(level) for level in atmos_ds.level.values)

    def surf_hist(name):  # (1, 2, H, W) using the first two times as history
        return torch.from_numpy(surf_ds[name].values[[0, 1]][None].copy())

    def atmos_hist(name):  # (1, 2, C, H, W)
        return torch.from_numpy(atmos_ds[name].values[[0, 1]][None].copy())

    batch = Batch(
        surf_vars={k: surf_hist(v) for k, v in SURF.items()},
        static_vars={k: torch.from_numpy(static[k].copy()) for k in ("lsm", "z", "slt")},
        atmos_vars={k: atmos_hist(v) for k, v in ATMOS.items()},
        metadata=Metadata(
            lat=torch.from_numpy(surf_ds.latitude.values.copy()),
            lon=torch.from_numpy(surf_ds.longitude.values.copy()),
            time=(surf_ds.time.values.astype("datetime64[s]").tolist()[1],),
            atmos_levels=levels,
        ),
    )

    # Verification truth (index 2) and persistence forecast (index 1 = analysis at init).
    truth = {
        "surf": {k: surf_ds[v].values[2] for k, v in SURF.items()},
        "atmos": {k: atmos_ds[v].values[2] for k, v in ATMOS.items()},
    }
    persist = {
        "surf": {k: surf_ds[v].values[1] for k, v in SURF.items()},
        "atmos": {k: atmos_ds[v].values[1] for k, v in ATMOS.items()},
    }
    lat = surf_ds.latitude.values
    return batch, truth, persist, lat, levels


def lat_weighted_rmse(forecast, obs, lat):
    """WeatherBench latitude-weighted RMSE; forecast/obs: (H, W), lat: (H,)."""
    w = np.cos(np.deg2rad(lat))
    w = w / w.mean()
    return float(np.sqrt((w[:, None] * (forecast - obs) ** 2).mean()))


def field(pred, kind, name):
    """Extract a (H, W) field from an Aurora prediction Batch."""
    if kind == "surf":
        return np.asarray(pred.surf_vars[name].squeeze().float())
    return np.asarray(pred.atmos_vars[name].squeeze().float())  # (C, H, W)


def truth_field(store, kind, name):
    return store[kind][name] if kind == "surf" else store["atmos"][name]


def rmse_row(label, spec, ref, tt, truth, persist, lat, levels):
    kind = "atmos" if isinstance(spec, tuple) else "surf"
    if kind == "atmos":
        var, level = spec
        ci = levels.index(level)
        f_tt = field(tt, "atmos", var)[ci]
        f_ref = field(ref, "atmos", var)[ci] if ref is not None else None
        o = truth_field(truth, "atmos", var)[ci]
        p = truth_field(persist, "atmos", var)[ci]
    else:
        var = spec
        f_tt = field(tt, "surf", var)
        f_ref = field(ref, "surf", var) if ref is not None else None
        o = truth_field(truth, "surf", var)
        p = truth_field(persist, "surf", var)
    h, w = f_tt.shape  # forecast grid (Aurora crops lat to a patch multiple)
    o, p, latc = o[:h, :w], p[:h, :w], lat[:h]
    return {
        "persist": lat_weighted_rmse(p, o, latc),
        "ref": lat_weighted_rmse(f_ref, o, latc) if f_ref is not None else None,
        "tt": lat_weighted_rmse(f_tt, o, latc),
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init", default=DEFAULT_INIT, help="Init time, e.g. 2021-07-01T12:00 (verify at +6h).")
    p.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="Area-average factor; 1 = native 0.25 deg.")
    p.add_argument(
        "--ref",
        action="store_true",
        help="Also run the (slow) CPU reference backbone and report the TT-vs-reference RMSE gap.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    batch, truth, persist, lat, levels = load_era5_window(args.init, args.stride)
    print(f"\nGrid: lat={len(lat)} lon={batch.surf_vars['2t'].shape[-1]} levels={levels}")

    print("\nLoading real Aurora 0.25 pretrained (1.3B) weights ...")
    model = AuroraPretrained()
    model.load_checkpoint()
    model.eval()

    ref = None
    if args.ref:
        print("Running PyTorch reference 6h forecast (CPU) ...")
        with torch.no_grad():
            ref = model.forward(batch)

    print("Attaching TT-NN backbone and running 6h forecast on Blackhole ...")
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        attach_tt_backbone(model, device, use_lora=False)
        with torch.no_grad():
            tt = model.forward(batch)
    finally:
        ttnn.close_mesh_device(device)

    print("\n=== WeatherBench latitude-weighted RMSE (6h forecast vs ERA5 truth) ===")
    print(
        f"{'var':>6} {'unit':>8} {'persistence':>12} {'reference':>12} {'TT(BH)':>12} "
        f"{'skill vs persist':>17} {'TT-ref':>9}"
    )
    all_skill_positive = True
    max_tt_ref_gap = 0.0
    for label, spec, unit, scale in HEADLINE:
        r = rmse_row(label, spec, ref, tt, truth, persist, lat, levels)
        skill = 1.0 - r["tt"] / r["persist"] if r["persist"] > 0 else float("nan")
        all_skill_positive &= skill > 0
        ref_str = f"{r['ref'] * scale:>12.4g}" if r["ref"] is not None else f"{'-':>12}"
        if r["ref"] is not None:
            gap = abs(r["tt"] - r["ref"])
            max_tt_ref_gap = max(max_tt_ref_gap, gap / max(r["ref"], 1e-9))
            gap_str = f"{gap * scale:>9.3g}"
        else:
            gap_str = f"{'-':>9}"
        print(
            f"{label:>6} {unit:>8} {r['persist'] * scale:>12.4g} {ref_str} "
            f"{r['tt'] * scale:>12.4g} {skill * 100:>15.1f} % {gap_str}"
        )

    print("\n=== verdict ===")
    print(f"  Aurora beats persistence on every variable: {'YES' if all_skill_positive else 'NO'}")
    if ref is not None:
        print(
            f"  TT vs reference: max relative RMSE gap = {max_tt_ref_gap * 100:.3f} % "
            f"({'TT preserves forecast skill' if max_tt_ref_gap < 0.01 else 'CHECK'})"
        )


if __name__ == "__main__":
    main()
