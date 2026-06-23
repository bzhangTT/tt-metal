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

from models.experimental.aurora.tt.common import from_tt, to_tt
from models.experimental.aurora.tt.perceiver import TtPerceiverResampler
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


class _TtPerceiverAdapter(torch.nn.Module):
    """Exposes a ``PerceiverResampler`` call signature ``(latents, context)`` but
    runs the cross-attention / MLP / layer-norms on device. Used for the encoder
    level aggregation and decoder level de-aggregation."""

    def __init__(self, tt_resampler: TtPerceiverResampler, device, out_dtype=torch.float32):
        super().__init__()
        self.tt = tt_resampler
        self.device = device
        self.out_dtype = out_dtype

    def forward(self, latents, x):
        lat = to_tt(latents.detach().to("cpu", torch.float32), self.device)
        ctx = to_tt(x.detach().to("cpu", torch.float32), self.device)
        out = from_tt(self.tt(lat, ctx))
        return out.to(device=latents.device, dtype=self.out_dtype)


def _build_tt_resampler(ref_resampler, device):
    """Build a ``TtPerceiverResampler`` from a reference ``PerceiverResampler``,
    inferring its config (dims, depth, heads, ln_k_q, residual, eps) from the module."""
    attn0 = ref_resampler.layers[0][0]
    ln1 = ref_resampler.layers[0][2]
    ln_k_q = isinstance(attn0.ln_k, torch.nn.LayerNorm)
    sd = {f"pr.{k}": v.detach().cpu().float() for k, v in ref_resampler.state_dict().items()}
    return TtPerceiverResampler(
        sd,
        prefix="pr",
        latent_dim=attn0.to_q.in_features,
        context_dim=attn0.to_kv.in_features,
        depth=len(ref_resampler.layers),
        head_dim=attn0.head_dim,
        num_heads=attn0.num_heads,
        residual_latent=ref_resampler.residual_latent,
        ln_eps=ln1.eps,
        ln_k_q=ln_k_q,
        device=device,
    )


def attach_tt_perceiver(model, device):
    """Run the encoder level aggregation and decoder level de-aggregation
    (Perceiver cross-attention) on device. The surrounding Fourier expansions,
    3D-conv patch embed, normalisation and un-patchify stay on host. Idempotent:
    re-attaching rebuilds from the saved reference modules."""
    enc = model.encoder
    if isinstance(enc.level_agg, _TtPerceiverAdapter):
        enc.level_agg = enc._orig_level_agg
    else:
        enc._orig_level_agg = enc.level_agg
    enc.level_agg = _TtPerceiverAdapter(_build_tt_resampler(enc._orig_level_agg, device), device)

    dec = model.decoder
    if isinstance(dec.level_decoder, _TtPerceiverAdapter):
        dec.level_decoder = dec._orig_level_decoder
    else:
        dec._orig_level_decoder = dec.level_decoder
    dec.level_decoder = _TtPerceiverAdapter(_build_tt_resampler(dec._orig_level_decoder, device), device)

    if getattr(dec, "separate_perceiver", ()):
        if isinstance(dec.level_decoder_alternate, _TtPerceiverAdapter):
            dec.level_decoder_alternate = dec._orig_level_decoder_alternate
        else:
            dec._orig_level_decoder_alternate = dec.level_decoder_alternate
        dec.level_decoder_alternate = _TtPerceiverAdapter(
            _build_tt_resampler(dec._orig_level_decoder_alternate, device), device
        )
    return model
