# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Trace-captured rollout/serving runner for the Aurora TT-NN backbone.

An autoregressive Aurora forecast replays the *entire* Swin backbone every 6 h
step -- 10-40x for a multi-day forecast -- with an identical device graph: the
lead time, the shifted-window attention masks and all shapes are constant across
steps, and only the input latent changes. The per-step host op-dispatch of the
~48 blocks is then pure repeated overhead.

``TtBackboneRunner`` captures that graph once with ``ttnn`` trace
(``begin_trace_capture`` / ``end_trace_capture``) and replays it with
``execute_trace`` each step, collapsing the host dispatch to a single trace
launch. The conditioning ``c_tt`` and the device attention masks are constants
and are baked into the trace (the mask cache is warmed before capture so no
host->device upload is recorded inside the traced region).

The input latent is copied into a single persistent device buffer and the trace
is replayed on the same command queue, so the copy and the replay are naturally
ordered (correct without cross-queue events). This is the right model for an
*autoregressive* rollout: step k+1's latent depends on step k's output, so there
is nothing to prefetch on a second queue. A second command queue (CQ1 input
upload overlapping CQ0 execute) only helps *independent-batch* serving, where the
next input is known ahead of time -- that is a separate runner, not this one.

Open the device with a trace region, e.g.::

    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=ts)

Usage::

    runner = TtBackboneRunner(tt_backbone).capture(x0, lead_time, patch_res)
    for step in range(n_steps):
        latent_out = runner.run(latent_in)   # torch in, torch out
    runner.release()
"""

from __future__ import annotations

from datetime import timedelta

import ttnn

from models.experimental.aurora.tt.common import from_tt, to_tt


class TtBackboneRunner:
    def __init__(self, backbone, *, cq_id=0):
        """backbone: a built ``TtSwin3DTransformerBackbone``. cq_id: command queue
        used for both the input upload and the trace replay (kept on one queue so
        they stay ordered)."""
        self.bb = backbone
        self.device = backbone.device
        self.cq_id = cq_id
        self.tid = None
        self.x_dev = None
        self.out_dev = None

    def capture(self, x_torch, lead_time: timedelta, patch_res):
        """Warm up (compile kernels + upload the constant masks) then capture the
        device backbone graph as a replayable trace."""
        bb = self.bb
        B = x_torch.shape[0]
        self.patch_res = patch_res
        self._dtype = ttnn.bfloat16

        # Conditioning is constant across the rollout (same lead time): compute
        # once, outside the trace, and reuse as a captured constant.
        self.c_tt = bb.compute_conditioning(lead_time, B, dtype=x_torch.dtype)

        # Persistent input buffer the trace reads from; we copy each step's latent
        # into it in place before replaying.
        self.x_dev = to_tt(x_torch, self.device, dtype=self._dtype)

        # Warm-up run: JIT-build every kernel and populate the per-block device
        # mask cache (the only host->device uploads), so the captured region has
        # no uploads -- only pure device ops.
        warm = bb.forward_device(self.x_dev, self.c_tt, patch_res)
        warm.deallocate()

        self.tid = ttnn.begin_trace_capture(self.device, cq_id=self.cq_id)
        self.out_dev = bb.forward_device(self.x_dev, self.c_tt, patch_res)
        ttnn.end_trace_capture(self.device, self.tid, cq_id=self.cq_id)
        return self

    def run(self, x_torch):
        """Run one rollout step: upload the latent into the persistent buffer,
        replay the traced backbone, and return the output latent as torch. The
        copy and the trace replay share one queue, so the replay sees the freshly
        uploaded latent."""
        assert self.tid is not None, "call capture() before run()"
        host = ttnn.from_torch(x_torch, dtype=self._dtype, layout=ttnn.TILE_LAYOUT)
        ttnn.copy_host_to_device_tensor(host, self.x_dev, cq_id=self.cq_id)
        ttnn.execute_trace(self.device, self.tid, cq_id=self.cq_id, blocking=True)
        return from_tt(self.out_dev)

    def release(self):
        if self.tid is not None:
            ttnn.release_trace(self.device, self.tid)
            self.tid = None
