# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Hybrid Aurora model: reference encoder/decoder + TT-NN Swin backbone.

The encoder and decoder are cheap, irregular, input-dependent preprocessing
(Fourier expansions, a 3D-conv patch embed, Perceiver level (de)aggregation,
un-patchify).  The backbone is ~all of the FLOPs and parameters and runs on the
Tenstorrent device.  This mirrors the standard tt-metal "port the dominant
compute first" pattern and keeps numerics anchored to the reference.
"""

from __future__ import annotations

from datetime import timedelta

import torch

import ttnn

from aurora.model.swin3d import Swin3DTransformerBackbone

from models.experimental.aurora.tt.swin import TtSwin3DTransformerBackbone


class _TtBackboneAdapter(torch.nn.Module):
    """Exposes the reference ``Swin3DTransformerBackbone`` call signature but
    dispatches the heavy compute to the TT-NN implementation on device."""

    def __init__(self, tt_backbone: TtSwin3DTransformerBackbone, out_dtype=torch.float32):
        super().__init__()
        self.tt_backbone = tt_backbone
        self.out_dtype = out_dtype

    def forward(self, x, lead_time: timedelta, rollout_step: int, patch_res):
        x_cpu = x.detach().to("cpu", torch.float32)
        out = self.tt_backbone(x_cpu, lead_time, patch_res)
        return out.to(device=x.device, dtype=self.out_dtype)


def attach_tt_backbone(model, device, *, use_lora=None, weight_dtype=ttnn.bfloat16):
    """Replace ``model.backbone`` with a TT-NN-accelerated adapter, built from
    the real weights already loaded into ``model``.

    Args:
        model: a reference :class:`aurora.Aurora` (or subclass) with weights loaded.
        device: an open ttnn mesh device.
        use_lora: override LoRA usage. Defaults to the model's setting.
    """
    # Preserve the original reference backbone so re-attaching (e.g. with a
    # different weight dtype) rebuilds from the real weights, not a prior adapter.
    if isinstance(model.backbone, _TtBackboneAdapter):
        model.backbone = model._orig_backbone
    else:
        model._orig_backbone = model.backbone

    ref: Swin3DTransformerBackbone = model.backbone
    # Re-key under a root prefix so f"{prefix}.<name>" never yields a leading dot.
    sd = {f"bb.{k}": v.detach().cpu().float() for k, v in ref.state_dict().items()}

    # Infer config from the reference module.
    embed_dim = ref.embed_dim
    enc_heads = [layer.blocks[0].num_heads for layer in ref.encoder_layers]
    dec_heads = [layer.blocks[0].num_heads for layer in ref.decoder_layers]
    enc_depths = [len(layer.blocks) for layer in ref.encoder_layers]
    dec_depths = [len(layer.blocks) for layer in ref.decoder_layers]
    window_size = ref.window_size
    if use_lora is None:
        use_lora = getattr(model, "use_lora", False)

    tt_backbone = TtSwin3DTransformerBackbone(
        sd,
        prefix="bb",
        embed_dim=embed_dim,
        encoder_depths=tuple(enc_depths),
        encoder_num_heads=tuple(enc_heads),
        decoder_depths=tuple(dec_depths),
        decoder_num_heads=tuple(dec_heads),
        window_size=window_size,
        device=device,
        use_lora=use_lora,
        weight_dtype=weight_dtype,
    )
    model.backbone = _TtBackboneAdapter(tt_backbone)
    return model
