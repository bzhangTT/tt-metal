# Optimizing Aurora inference & serving on Tenstorrent (Blackhole)

This is a deep dive into the low-level optimizations that matter for Aurora on
Tenstorrent hardware, ordered roughly by impact. Each item notes *what* it does,
*why* it helps on this architecture, *where* it applies in Aurora, and the
*risk* (mostly numerical). The reference port in `tt/` is correctness-first
(PCC ≈ 1.0); the items below are the path from "correct" to "fast and
servable". Items marked **[done]** are wired into the code with a flag; the rest
are documented designs with the concrete TT-NN entry points to use.

Aurora shapes that drive every decision (small / 1.3B):
- `embed_dim` 256 / 512, doubling each encoder stage → up to 1024 / 2048.
- window `(2, 6, 12)` → `N = 144` tokens/window (pads to 5 tiles of 32 = 160).
- backbone is ~48 transformer blocks (Σ encoder + Σ decoder depths) per forward.
- autoregressive rollout repeats the *entire* backbone every 6 h step → the same
  graph runs 10–40× for a multi-day forecast. This is why serving-time graph
  capture dominates.

---

## 1. Keep the block device-resident; do windowing with TT-NN ops — **[done]**

**Before** every Swin block round-tripped to host for `window_partition_3d`,
`torch.roll`, `pad`, `window_reverse_3d`. That was two PCIe copies of a
`(B, C·H·W, D)` activation *per block* × ~48 blocks × every rollout step — the
dominant non-compute cost.

These are now done on device (`tt_window_partition_3d` / `tt_window_reverse_3d`
/ `tt_pad_3d` / `tt_crop_3d` in `swin.py`): the activation is uploaded once at
the start of the backbone and downloaded once at the end instead of ~5
round-trips per block. The block `__call__` now takes and returns a device
tensor. Measured **3.2–4.6× backbone speedup** with all PCC tests still passing
(full-model worst-variable PCC 0.979).

All of those are pure layout ops with direct TT-NN equivalents:
- cyclic shift → `ttnn.roll` (exists; verified present on this build).
- pad to window multiple → `ttnn.pad`.
- `window_partition_3d` = `view(B, C/wc, wc, H/wh, wh, W/ww, ww, D)` then
  permute to `(B·nW, wc·wh·ww, D)` → `ttnn.reshape` + `ttnn.permute`.
- reverse = inverse permute + reshape.

Doing the whole block on device removes ~96 host transfers per forward. The
window/permute ops are memory-bound but stay in DRAM/L1 and overlap with the
matmuls of neighbouring windows. **Risk:** none numerically; ttnn `permute` on
rank-8 tensors must be expressed as composed rank-≤4 permutes.

## 2. Pack weights as `bfp8_b` (block float8) — **[done, flag]**

The QKV (`D×3D`), projection (`D×D`) and especially the MLP (`D×4D`, `4D×D`)
matmuls hold the parameters and the FLOPs. Storing those weights as
`ttnn.bfloat8_b` (8-bit mantissa, shared exponent per 16-element block):
- halves weight DRAM footprint and weight-read bandwidth (the binding
  constraint for these skinny-activation matmuls), ~1.7–2× matmul throughput;
- keeps activations in `bfloat16`.

`common.TtLinear(..., weight_dtype=ttnn.bfloat8_b)` does this, threaded through
`attach_tt_backbone(..., weight_dtype=ttnn.bfloat8_b)`. Measured on the real
small checkpoint: full-model worst-variable PCC drops only `0.98430 → 0.98143`
(see `test_full_model_pcc_bfp8`). The FiLM `ln_modulation`, `time_mlp` and the
patch merge/split norms (tiny, sensitive, produce scale/shift) are kept
`bfloat16`. **Risk:** numerical — gate with the PCC test before enabling for
the 1.3B checkpoint.

## 3. Fuse: activation-in-linear, scaled-mask-softmax, flash SDPA

- `ttnn.linear(..., activation="gelu")` fuses the MLP fc1 GELU into the matmul
  epilogue — no separate eltwise pass over `(·, 4D)`.
- Replace the manual `QKᵀ → ·scale → +mask → softmax → ·V` with
  `ttnn.transformer.scaled_dot_product_attention(q, k, v, attn_mask=…)`
  (flash attention): never materializes the `(N, N)` score matrix in DRAM,
  fuses scale + mask + softmax, and tiles over `N`. **[implemented, off by
  default — `_USE_SDPA`]**. Measured: even with a single-chunk program config
  (`q/k_chunk = pad(N,32)`), exact exp (`exp_approx_mode=False`) and HiFi2 +
  fp32 dest-acc, the flash online-softmax in bf16 is ~0.9997/block, which
  compounds over the ~48 blocks to backbone PCC ~0.989 (vs 0.998 for the exact
  path) and drops worst-variable full-model PCC to ~0.89 — below the 0.97 gate.
  At Aurora's `N=144` the `(N,N)` scores are tiny and the backbone is
  data-movement bound, so there is no speed win to pay for that accuracy: the
  exact manual path stays the default. SDPA is the right call for the high-res
  (`patch_size=10`) variants where `N` and window counts explode — enable it
  there and re-check PCC.
- `ttnn.layer_norm` already fuses mean/var/normalize/affine in one kernel; feed
  the FiLM `scale`/`shift` as the post-affine to fuse the adaptive modulation.

## 4. Shard across the 32-chip Galaxy

Two orthogonal axes, both natural for Aurora:

**Window/data parallelism (simplest, near-linear).** The windowed attention is
embarrassingly parallel over the `nW·B` window-batch dimension. Shard that
dimension across the mesh with `ttnn.ShardTensorToMesh(mesh, dim=0)`, run the
identical per-window attention on every chip, and the result is already in the
right place for the (device-resident) window-reverse. With ~hundreds of windows
per block this fills 32 chips well. Rollout/batched-forecast members likewise
shard over the batch dim.

**Tensor parallelism (for the 1.3B widths).** Split attention heads and the MLP
hidden dim across chips: column-shard `qkv`/`fc1` weights (`ShardTensorToMesh`
on the output dim), compute locally, then `ttnn.all_gather` (attention) /
`ttnn.reduce_scatter` (row-sharded `proj`/`fc2`) over the fast Ethernet mesh.
Blackhole Galaxy's all-to-all topology keeps the collective cost low relative to
the `2D×4D` matmuls. This is what makes the 2048-wide deepest stage fit and run
fast.

A practical hybrid: tensor-parallel within a tray (fast local links),
data/window-parallel across trays.

## 5. Sharded L1 layouts + tuned matmul program configs — **[hook, off by default]**

Keep activations in **L1 block-sharded** memory across the core grid instead of
DRAM-interleaved, and pass explicit
`ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` (per-core M/N/K tiles,
`fused_activation`) sized to Blackhole's core grid. This avoids DRAM spills for
the `(nW·B, N, D)` tensors and lets the QKV/MLP matmuls reuse weights resident
in L1. Pad `N=144 → 160` (5×32) once so every tile is full. The deepest stages
(`D=2048`, hidden `8192`) are where program-config tuning pays off most.

**Status / finding.** A core-grid hook (`set_matmul_core_grid`, threaded into the
QKV/proj/MLP matmuls) is wired but defaults to ttnn auto-selection — the grid does
not change the math, so it is PCC-neutral. It is **not** enabled as a default
because at the validated 0.25° small/1.3B widths the backbone is *data-movement /
dispatch* bound, not compute bound: HiFi2 helped while bfp8/LoFi did not (§2/§4),
device-resident windowing removed the host round-trips (§1), and the trace runner
removed the per-step dispatch (§6). With compute off the critical path, pinning
the matmul grid / L1-sharding the activations is the right lever only for the
*compute-bound* high-res variants (`patch_size=10`, 2048-wide stages) and for
genuinely concurrent multi-stream serving. (A clean matmul microbenchmark also
isn't possible here — the single Blackhole chip is shared with other jobs.) Enable
the hook there, size the program configs to the grid, and re-check PCC.

## 6. Trace capture for rollout/serving — the rollout multiplier — **[done]**

For *serving* (and any multi-step rollout) the host-side op-dispatch of ~48
blocks per step is pure overhead that repeats every step. `TtBackboneRunner`
(`tt/runner.py`) captures the backbone once with `ttnn.begin_trace_capture` /
`ttnn.end_trace_capture` and replays it with `ttnn.execute_trace` each rollout
step: dispatch cost collapses to a single trace launch, so a 10-day (40-step)
forecast pays graph-build once. The conditioning `c_tt` and the device attention
masks are constants and are baked into the trace (the mask cache is warmed before
capture, so the captured region contains no host->device uploads). The input
latent is copied into one persistent device buffer; copy and replay share a
command queue, so the replay sees the fresh latent without cross-queue events.
Validated by `test_trace_rollout_matches_eager`: traced == eager to PCC 1.00000
across rollout steps, **~37x** per-step on the (dispatch-bound) small random
backbone — the real-checkpoint speedup is smaller (more compute-bound) but the
~48-block dispatch is eliminated either way.

**On two command queues:** a 2nd CQ prefetching the next input overlaps only
when the next input is known ahead of time — i.e. *independent-batch* serving.
An *autoregressive* rollout can't use it: step k+1's latent is a function of
step k's output, so there is nothing to prefetch. (Aurora's per-step input upload
is a tiny `(1, L, D)` tensor anyway, dominated by the backbone, so single-CQ
trace already captures essentially all the win for rollout.)

## 7. Cheap but real wins

- **Cache device masks** keyed by the (lru-cached) host mask identity —
  **[done]**. There are only 2 distinct shift patterns per stage, so each
  shifted block uploads its additive mask once and reuses the device tensor
  across blocks and across every rollout step instead of re-uploading it each
  call (`TtSwin3DTransformerBlock._mask_cache`; covered by
  `test_mask_cache_reused_across_rollout`).
- **Upload weights as bf16/bfp8 once** at construction (done) — never per call.
- **Fold LoRA into the base weight for inference — [done, `_FOLD_LORA`].** In
  `"single"` mode the LoRA correction adds to the *output*, so the effective
  `(out,in)` weight is `W'_eff = W + s·(B·A)` with `s = alpha/r`. Folding
  (`TtWindowAttention._fold`, done in fp32 on host before upload) removes two
  extra matmuls per projection per block at zero accuracy cost when the rollout
  step is fixed (inference). Validated by `test_lora_fold_matches_unfolded`:
  folded-vs-reference and folded-vs-unfolded PCC = 1.00000. The unfused runtime
  path is kept (`set_fold_lora(False)`) for a future per-step `"all"`-mode LoRA.
- **Run encoder/decoder Perceiver attention on device too — [done].** The level
  aggregation (`encoder.level_agg`) and de-aggregation (`decoder.level_decoder`)
  are a Flamingo-style `PerceiverResampler` (cross-attention + GELU-MLP + post-res
  layer-norms) — the same `linear/layer_norm` primitives as the backbone.
  `tt/perceiver.py` ports it (`TtPerceiverResampler`) and `attach_tt_perceiver`
  swaps both into the hybrid model, moving the host↔device boundary out to the
  Fourier/conv/normalisation preprocessing only. Validated: standalone PCC 0.99979
  (`test_perceiver_resampler`); full model with TT backbone **and** TT Perceiver
  worst-variable PCC 0.95903 (`test_full_model_pcc_with_perceiver`). The end-to-end
  win is at native 0.25° resolution, where the encoder/decoder run over a
  ~720×1440 grid (the ~18 s host tail); at the small test resolution the
  encoder/decoder are only ~10 ms each, so this is a correctness-proven capability
  for the native-res serving path rather than a test-res speedup.
- **Pin lat/lon/scale positional encodings.** They depend only on the grid, not
  the data, so compute once per grid and cache.

---

## Recommended enablement order

1. `bfp8_b` weights for MLP + QKV/proj (flag in `TtLinear`), gate on PCC. **[done]**
2. Fused `activation="gelu"` linear **[done]**; flash SDPA **[implemented, off by
   default — fails the PCC gate at `N=144`; for high-res variants only]**.
   LoRA folding for inference **[done, `_FOLD_LORA`]**.
3. Device-resident windowing (remove per-block host transfers). **[done]**
4. Matmul fidelity HiFi4→HiFi2 + fp32 accumulation (`set_compute_fidelity`),
   PCC-equal, ~1.25× on the matmuls. **[done, default]**
5. Trace-captured backbone runner for rollout/serving (`TtBackboneRunner`).
   **[done]**
6. Mesh sharding: window/data-parallel first, then tensor-parallel for 1.3B
   (Megatron col/row MLP sharding scaffolded in `TtLinear(tp=…)` /
   `set_tensor_parallel`, **off by default** — needs a working inter-chip fabric).
7. L1 sharding + tuned matmul program configs on the deepest stages
   (`set_matmul_core_grid` hook, **off by default** — compute-bound high-res
   regime only; backbone is data-movement/dispatch bound at the validated scale).

Validation is the same `pcc` test at every step: enable an optimization, re-run
`tests/test_aurora.py`, keep it only if worst-variable PCC stays above the
threshold (> 0.97 full-model, > 0.99 per-block).
