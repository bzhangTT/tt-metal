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

## 1. Keep the block device-resident; do windowing with TT-NN ops

**Today** every Swin block round-trips to host for `window_partition_3d`,
`torch.roll`, `pad`, `window_reverse_3d`. That is two PCIe copies of a
`(B, C·H·W, D)` activation *per block* × ~48 blocks × every rollout step — the
dominant non-compute cost.

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
  fuses scale + mask + softmax, and tiles over `N`. For Aurora's small `N=144`
  the win is modest, but it removes a DRAM round-trip of the scores per window
  and is the right call for the high-res (`patch_size=10`) variants where `N`
  and window counts explode.
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

## 5. Sharded L1 layouts + tuned matmul program configs

Keep activations in **L1 block-sharded** memory across the core grid instead of
DRAM-interleaved, and pass explicit
`ttnn.MatmulMultiCoreReuseMultiCastProgramConfig` (per-core M/N/K tiles,
`fused_activation`) sized to Blackhole's core grid. This avoids DRAM spills for
the `(nW·B, N, D)` tensors and lets the QKV/MLP matmuls reuse weights resident
in L1. Pad `N=144 → 160` (5×32) once so every tile is full. The deepest stages
(`D=2048`, hidden `8192`) are where program-config tuning pays off most.

## 6. Trace capture + multi-CQ for serving — the rollout multiplier

For *serving* (and any multi-step rollout) the host-side op-dispatch of ~48
blocks per step is pure overhead that repeats every step. Capture the backbone
once with `ttnn.begin_trace_capture` / `ttnn.end_trace_capture` and replay with
`ttnn.execute_trace` each rollout step: dispatch cost collapses to a single
trace launch, so a 10-day (40-step) forecast pays graph-build once.

Use **two command queues**: CQ0 runs the traced backbone while CQ1 prefetches
the next step's encoder output (and uploads the constant masks/weights, already
resident). Double-buffer the encoder→backbone handoff so the device never
stalls on host. Weights and the per-`(C,H,W,shift)` attention masks are
constants across steps — upload once and cache the *device* tensors (the host
mask builder is already `lru_cache`d; extend the cache to hold the uploaded
`ttnn.Tensor`).

## 7. Cheap but real wins

- **Cache device masks** keyed by the (lru-cached) host mask identity —
  **[done]**. There are only 2 distinct shift patterns per stage, so each
  shifted block uploads its additive mask once and reuses the device tensor
  across blocks and across every rollout step instead of re-uploading it each
  call (`TtSwin3DTransformerBlock._mask_cache`; covered by
  `test_mask_cache_reused_across_rollout`).
- **Upload weights as bf16/bfp8 once** at construction (done) — never per call.
- **Fold LoRA into the base weight for inference.** In `"single"` mode the LoRA
  correction is `x·(Aᵀ·Bᵀ)·s`; precompute `W' = W + s·BᵀAᵀ`... wait, it adds to
  the *output* so `W'_eff = W + s·(B·A)` on the `(out,in)` weight. Folding
  removes two extra matmuls per attention per block at zero accuracy cost when
  the rollout step is fixed (inference). Keep the unfused path only if you need
  per-step `"all"`-mode LoRA.
- **Run encoder/decoder Perceiver attention on device too** once the backbone is
  device-resident — they are the same `linear/SDPA/layer_norm` primitives, so
  the host↔device boundary can move out to the Fourier/conv preprocessing only.
- **Pin lat/lon/scale positional encodings.** They depend only on the grid, not
  the data, so compute once per grid and cache.

---

## Recommended enablement order

1. `bfp8_b` weights for MLP + QKV/proj (flag in `TtLinear`), gate on PCC. **[done]**
2. Fused `activation="gelu"` linear + flash SDPA. **[done for GELU]**
3. Device-resident windowing (remove per-block host transfers).
4. Trace + 2-CQ backbone runner for rollout/serving.
5. Mesh sharding: window/data-parallel first, then tensor-parallel for 1.3B.
6. L1 sharding + tuned matmul program configs on the deepest stages.

Validation is the same `pcc` test at every step: enable an optimization, re-run
`tests/test_aurora.py`, keep it only if worst-variable PCC stays above the
threshold (> 0.97 full-model, > 0.99 per-block).
