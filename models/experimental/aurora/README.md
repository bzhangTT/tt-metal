# Aurora (Microsoft) — TT-NN port

**Platforms:** Blackhole. The backbone runs on a single Blackhole chip
(`ttnn.MeshShape(1, 1)`); validated on one chip of a 32-chip Blackhole Galaxy
host. Multi-chip mesh sharding is a documented optimization, not yet wired —
see [OPTIMIZATION.md](OPTIMIZATION.md).

[Aurora](https://github.com/microsoft/aurora) is Microsoft's 1.3B-parameter
foundation model for the Earth's atmosphere. It predicts surface and
atmospheric variables (temperature, winds, geopotential, humidity, …) and is
fine-tuned for medium-resolution weather, high-resolution weather, air
pollution, and ocean waves.

This directory ports Aurora to Tenstorrent's TT-NN / TT-Metalium stack and runs
real-weight inference on Blackhole.

## Architecture and what runs where

Aurora is `Perceiver3DEncoder → Swin3DTransformerBackbone → Perceiver3DDecoder`.
The **backbone** is a 3D Swin-V2 U-Net (windowed shifted attention, FiLM
adaptive layer-norm conditioned on lead time, optional LoRA, patch
merge/split, skip connections) and accounts for essentially all of the FLOPs
and parameters.

| Component | Device | Why |
|-----------|--------|-----|
| Encoder (Fourier expansions, 3D-conv patch embed, Perceiver level aggregation) | host (torch) | cheap, irregular, input-shaped preprocessing |
| **Swin3D backbone** (QKV / attention / projection / adaptive-LN / MLP / patch merge-split / time-MLP) | **Blackhole (TT-NN)** | dominant compute |
| Decoder (Perceiver de-aggregation, linear heads, un-patchify) | host (torch) | cheap post-processing |

The exotic, memory-layout-only glue inside the backbone (3D window
partition/reverse, cyclic shift, padding, 2×2 patch merge/split reshapes) is
reused verbatim from the reference implementation so indexing is
bit-for-bit identical; the dense linear algebra runs on the Tensix cores.

## Layout

```
models/experimental/aurora/
├── tt/
│   ├── common.py   # TT-NN helpers: TtLinear, layer_norm, dtype/upload, pcc
│   ├── swin.py     # TtWindowAttention, TtAdaptiveLayerNorm, TtMLP,
│   │               # TtSwin3DTransformerBlock, TtPatchMerging/Splitting3D,
│   │               # TtBasicLayer3D, TtSwin3DTransformerBackbone
│   └── model.py    # attach_tt_backbone(): swaps the reference backbone for TT-NN
├── tests/test_aurora.py   # PCC vs reference, REAL weights
├── demo/demo.py           # end-to-end real-weight forecast on Blackhole
├── OPTIMIZATION.md         # deep dive on low-level TT optimizations
└── README.md
```

## Prerequisites

Build tt-metal from source and create the venv (see repo `INSTALLING.md`):

```bash
./build_metal.sh
./create_venv.sh
source python_env/bin/activate
python -m pip install microsoft-aurora   # reference model + real checkpoints
```

## Run

```bash
# Correctness (downloads aurora-0.25-small-pretrained.ckpt from HuggingFace):
python -m pytest models/experimental/aurora/tests/test_aurora.py -s

# End-to-end forecast demo on Blackhole:
python models/experimental/aurora/demo/demo.py
```

## Results (real `aurora-0.25-small-pretrained.ckpt`, single Blackhole chip)

Reproduced on this hardware with a freshly built `tt-metal` (`build_metal.sh` +
`create_venv.sh`); every row is a committed test in `tests/test_aurora.py`:

| Test (`test_aurora.py`) | Metric |
|------|--------|
| Swin3D backbone vs reference, random weights (`test_backbone_pcc_random`) | PCC = 1.00000 |
| Swin3D backbone vs reference, real weights (`test_backbone_pcc`) | PCC = 0.99846 |
| Full model, worst-of-9-variables forecast, bf16 (`test_full_model_pcc`) | PCC = 0.97892 |
| Full model, worst-of-9-variables forecast, **bfp8_b** (`test_full_model_pcc_bfp8`) | PCC = 0.97945 |

Per-variable (bf16): `z` 0.99999, `t` 0.99976, `10v` 0.99831, `10u` 0.99722,
`q` 0.99477, `u` 0.98673, `msl` 0.98682, `2t` 0.98392, `v` 0.97892.

(The full-model numbers are with the default HiFi2 + fp32-accumulation matmul
fidelity and device-resident windowing; HiFi4 gives ~0.984 worst-variable at
~1.25× the matmul cost — flip back with `set_compute_fidelity` if you need it.)

The same config-driven code runs the full 1.3B checkpoint (`embed_dim=512`,
LoRA enabled) — set `use_lora=True` in `attach_tt_backbone`. Pass
`weight_dtype=ttnn.bfloat8_b` to `attach_tt_backbone` for the block-float8
optimization path.

### Real-weather validation (`demo/real_weather_demo.py`)

Real ERA5/HRES fields → real forecast, with the **full 1.3B pretrained**
checkpoint. Downloads actual WeatherBench2 HRES T0 data (2022-05-11, no auth),
coarsened to ~2° so the 1.3B model is tractable on CPU + a single Blackhole
chip (values are real physical fields, lower spatial resolution than native
0.25°):

| Field | Value |
|-------|-------|
| Real input 2t | 202–319 K, mean 278.9 K |
| Reference 6h forecast 2t | 210.7–302.8 K, mean 277.3 K |
| TT-NN 6h forecast 2t (Blackhole) | 210.8–303.0 K, mean 277.3 K |
| 6h persistence correlation (input vs forecast) | **0.9816** |
| TT-vs-reference worst-variable PCC | **0.9997** |

The forecast is physically plausible (sane global mean, extremes smoothed) and
tightly tracks the input as a correct short-range forecast must — confirming
real weather in → real weather out, not noise matching.

Run on any day in the HRES-T0 range (2016–2022) and any number of 6 h
autoregressive steps:

```bash
python models/experimental/aurora/demo/real_weather_demo.py            # 2022-05-11, 1 step
python models/experimental/aurora/demo/real_weather_demo.py --day 2021-01-15 --steps 4
```

`--steps > 1` drives a real autoregressive rollout (`aurora.rollout`); the whole
backbone replays each step and the device mask cache makes every step after the
first reuse its uploaded masks. Verified on 2021-01-15 (a winter day): colder
input (mean 276.8 K) than the May day, plausible 6 h/12 h forecasts, and the
same 0.9997 TT-vs-reference worst-variable PCC.

### WeatherBench skill (`demo/weatherbench_eval.py`)

The checks above prove the port is *faithful* (TT matches the reference) and the
output *looks like weather*, but not that the forecast has *skill*. This computes
the headline WeatherBench2 metric — **latitude-weighted RMSE** of the 6 h
forecast against **ERA5 ground truth** — for the standard headline variables, and
compares Aurora to a persistence ("tomorrow = today") baseline. ERA5 is both the
init and the truth (the dataset `AuroraPretrained` was trained on), so it is an
apples-to-apples verification.

At Aurora's native **0.25°** on a single Blackhole chip (init 2021-07-01 12:00 UTC,
verify +6 h), the TT-NN forecast beats persistence on every variable:

| Variable | Persistence RMSE | **Aurora TT (Blackhole)** | Skill vs persistence |
|----------|------------------|---------------------------|----------------------|
| z500 (m²/s²) | 231.2 | **20.1** | **91.3 %** |
| t850 (K)     | 1.548 | **0.292** | **81.2 %** |
| q700 (g/kg)  | 1.027 | **0.255** | **75.1 %** |
| 2t (K)       | 3.399 | **0.506** | **85.1 %** |
| msl (Pa)     | 228.5 | **23.2**  | **89.9 %** |
| 10u (m/s)    | 2.152 | **0.356** | **83.5 %** |
| 10v (m/s)    | 2.48  | **0.360** | **85.5 %** |

```bash
python models/experimental/aurora/demo/weatherbench_eval.py                    # native 0.25°, the table above
python models/experimental/aurora/demo/weatherbench_eval.py --stride 2 --ref   # 0.5°, also runs the CPU reference
```

**Skill is resolution-dependent — run at native 0.25°.** Aurora is trained at
0.25°; area-averaging the input coarser pushes it out of distribution and the
skill collapses (≈0 at 0.5°, negative at 2°), monotonically with resolution. This
is a property of the model, not the port — the TT and reference RMSEs agree to
within **0.53 %** (worst variable, 0.5°) regardless, so the TT backbone preserves
whatever skill the resolution affords. The eval runs TT-only by default so native
resolution stays tractable; `--ref` adds the (slow) CPU reference backbone, and
the faithfulness is pinned separately by the per-block PCC tests (0.998–1.0) and
the 0.5°/2° RMSE gaps.

> These are a single-init verification demonstrating genuine skill on hardware,
> not a reproduction of the official WeatherBench2 leaderboard (which averages
> many inits with a prescribed climatology and regridding).

See [OPTIMIZATION.md](OPTIMIZATION.md) for the low-level Tenstorrent
optimizations (dtype packing, device-resident windowing, mesh sharding across
the 32-chip Galaxy, fused SDPA, trace + multi-CQ serving).
