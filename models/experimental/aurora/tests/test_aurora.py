# Copyright (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Correctness tests for the Aurora TT-NN port, validated with REAL weights.

Run (from repo root):
    python -m pytest models/experimental/aurora/tests/test_aurora.py -s

The tests download the real ``aurora-0.25-small-pretrained.ckpt`` from
HuggingFace (microsoft/aurora) and compare the TT-NN backbone / full model
against the reference PyTorch implementation.
"""

from datetime import datetime, timedelta

import pytest
import torch

import ttnn

from aurora import AuroraSmallPretrained, Batch, Metadata
from aurora.model.swin3d import Swin3DTransformerBackbone

from models.experimental.aurora.tt.common import pcc
from models.experimental.aurora.tt.model import _TtBackboneAdapter, attach_tt_backbone
from models.experimental.aurora.tt.swin import TtSwin3DTransformerBackbone


def _make_batch(h=32, w=64, levels=(100, 250, 500, 850), t=2):
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


@pytest.fixture(scope="module")
def device():
    dev = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    yield dev
    ttnn.close_mesh_device(dev)


@pytest.fixture(scope="module")
def ref_model():
    model = AuroraSmallPretrained()
    model.load_checkpoint()  # real weights
    model.eval()
    return model


@pytest.fixture(scope="module")
def ref_forecast(ref_model):
    """The true PyTorch reference forecast, computed once on the unmodified model.

    The full-model tests attach a TT backbone to the (module-scoped) ``ref_model``
    in place, so each test must measure against this clean reference rather than
    re-running ``forward`` after a previous test left a TT adapter attached.
    """
    if isinstance(ref_model.backbone, _TtBackboneAdapter):
        ref_model.backbone = ref_model._orig_backbone
    batch = _make_batch()
    with torch.no_grad():
        return batch, ref_model.forward(batch)


def test_backbone_pcc(device, ref_model):
    """TT backbone vs reference backbone on a representative latent input."""
    ref_bb: Swin3DTransformerBackbone = ref_model.backbone
    sd = {f"bb.{k}": v.detach().cpu().float() for k, v in ref_bb.state_dict().items()}

    patch_res = (4, 8, 16)  # (latent_levels, H/patch, W/patch)
    B = 1
    L = patch_res[0] * patch_res[1] * patch_res[2]
    D = ref_bb.embed_dim
    torch.manual_seed(1)
    x = torch.randn(B, L, D)
    lead = timedelta(hours=6)

    with torch.no_grad():
        ref_out = ref_bb(x, lead_time=lead, rollout_step=0, patch_res=patch_res)

    tt_bb = TtSwin3DTransformerBackbone(
        sd,
        prefix="bb",
        embed_dim=D,
        encoder_depths=tuple(len(l.blocks) for l in ref_bb.encoder_layers),
        encoder_num_heads=tuple(l.blocks[0].num_heads for l in ref_bb.encoder_layers),
        decoder_depths=tuple(len(l.blocks) for l in ref_bb.decoder_layers),
        decoder_num_heads=tuple(l.blocks[0].num_heads for l in ref_bb.decoder_layers),
        window_size=ref_bb.window_size,
        device=device,
        use_lora=False,
    )
    tt_out = tt_bb(x, lead, patch_res)
    p = pcc(ref_out, tt_out)
    print(f"\n[backbone] shape={tuple(tt_out.shape)} PCC={p:.5f}")
    assert p > 0.98, f"backbone PCC too low: {p}"


def test_backbone_pcc_random(device):
    """TT backbone vs reference on a randomly-initialised backbone (no checkpoint).

    Isolates the TT port's numerics from real-weight dynamic range: with random
    weights the only error source is the bf16 device math, so PCC is ~1.0.
    """
    torch.manual_seed(0)
    ref_bb = Swin3DTransformerBackbone(embed_dim=96, window_size=(2, 6, 12)).eval()
    sd = {f"bb.{k}": v.detach().cpu().float() for k, v in ref_bb.state_dict().items()}

    patch_res = (4, 8, 16)
    D = ref_bb.embed_dim
    x = torch.randn(1, patch_res[0] * patch_res[1] * patch_res[2], D)
    lead = timedelta(hours=6)
    with torch.no_grad():
        ref_out = ref_bb(x, lead_time=lead, rollout_step=0, patch_res=patch_res)

    tt_bb = TtSwin3DTransformerBackbone(
        sd,
        prefix="bb",
        embed_dim=D,
        encoder_depths=tuple(len(l.blocks) for l in ref_bb.encoder_layers),
        encoder_num_heads=tuple(l.blocks[0].num_heads for l in ref_bb.encoder_layers),
        decoder_depths=tuple(len(l.blocks) for l in ref_bb.decoder_layers),
        decoder_num_heads=tuple(l.blocks[0].num_heads for l in ref_bb.decoder_layers),
        window_size=ref_bb.window_size,
        device=device,
        use_lora=False,
    )
    p = pcc(ref_out, tt_bb(x, lead, patch_res))
    print(f"\n[backbone, random weights] PCC={p:.5f}")
    assert p > 0.999, f"random-weight backbone PCC too low: {p}"


def test_lora_fold_matches_unfolded(device):
    """LoRA folding (W' = W + (alpha/r)*B@A) matches both the reference and the
    unfolded runtime LoRA path on a random LoRA-enabled backbone.

    The real small/1.3B checkpoints used elsewhere are exercised without LoRA or
    with it always folded; this isolates the fold itself. ``lora_B`` is zero-init
    in the reference (LoRA starts as identity), so we randomise it to make the
    correction non-trivial, then check (a) folded TT vs reference torch backbone
    and (b) folded TT vs the explicit two-matmul TT path agree to ~1.0.
    """
    from models.experimental.aurora.tt.swin import set_fold_lora

    torch.manual_seed(0)
    ref_bb = Swin3DTransformerBackbone(embed_dim=96, window_size=(2, 6, 12), use_lora=True).eval()
    with torch.no_grad():
        for name, prm in ref_bb.named_parameters():
            if "lora_B" in name:  # zero-init by default -> randomise for a real test
                prm.normal_(0.0, 0.02)
    sd = {f"bb.{k}": v.detach().cpu().float() for k, v in ref_bb.state_dict().items()}

    patch_res = (4, 8, 16)
    D = ref_bb.embed_dim
    x = torch.randn(1, patch_res[0] * patch_res[1] * patch_res[2], D)
    lead = timedelta(hours=6)
    with torch.no_grad():
        ref_out = ref_bb(x, lead_time=lead, rollout_step=0, patch_res=patch_res)

    def build():
        return TtSwin3DTransformerBackbone(
            sd,
            prefix="bb",
            embed_dim=D,
            encoder_depths=tuple(len(l.blocks) for l in ref_bb.encoder_layers),
            encoder_num_heads=tuple(l.blocks[0].num_heads for l in ref_bb.encoder_layers),
            decoder_depths=tuple(len(l.blocks) for l in ref_bb.decoder_layers),
            decoder_num_heads=tuple(l.blocks[0].num_heads for l in ref_bb.decoder_layers),
            window_size=ref_bb.window_size,
            device=device,
            use_lora=True,
        )

    try:
        set_fold_lora(True)
        folded = build()(x, lead, patch_res)
        set_fold_lora(False)
        unfolded = build()(x, lead, patch_res)
    finally:
        set_fold_lora(True)  # restore default

    p_fold = pcc(ref_out, folded)
    p_unfold = pcc(ref_out, unfolded)
    p_equiv = pcc(folded, unfolded)
    print(f"\n[lora fold] folded-vs-ref={p_fold:.5f} unfolded-vs-ref={p_unfold:.5f} folded-vs-unfolded={p_equiv:.5f}")
    assert p_fold > 0.999, f"folded LoRA backbone PCC too low: {p_fold}"
    assert p_equiv > 0.999, f"folded vs unfolded LoRA mismatch: {p_equiv}"


def test_mask_cache_reused_across_rollout(device, ref_model):
    """Device shifted-window masks are uploaded once and reused on later steps.

    An autoregressive Aurora rollout replays the whole backbone every 6 h step
    with identical attention masks, so the per-block device-mask cache must not
    grow after the first forward (otherwise we'd re-upload masks every step).
    """
    ref_bb: Swin3DTransformerBackbone = ref_model.backbone
    sd = {f"bb.{k}": v.detach().cpu().float() for k, v in ref_bb.state_dict().items()}
    patch_res = (4, 8, 16)
    D = ref_bb.embed_dim
    torch.manual_seed(2)
    x = torch.randn(1, patch_res[0] * patch_res[1] * patch_res[2], D)
    lead = timedelta(hours=6)

    tt_bb = TtSwin3DTransformerBackbone(
        sd,
        prefix="bb",
        embed_dim=D,
        encoder_depths=tuple(len(l.blocks) for l in ref_bb.encoder_layers),
        encoder_num_heads=tuple(l.blocks[0].num_heads for l in ref_bb.encoder_layers),
        decoder_depths=tuple(len(l.blocks) for l in ref_bb.decoder_layers),
        decoder_num_heads=tuple(l.blocks[0].num_heads for l in ref_bb.decoder_layers),
        window_size=ref_bb.window_size,
        device=device,
        use_lora=False,
    )
    blocks = [b for layer in (*tt_bb.encoder_layers, *tt_bb.decoder_layers) for b in layer.blocks]

    tt_bb(x, lead, patch_res)  # step 0: populates caches
    sizes_after_step0 = [len(b._mask_cache) for b in blocks]
    assert any(s > 0 for s in sizes_after_step0), "shifted blocks should cache a device mask"

    tt_bb(x, lead, patch_res)  # step 1: must reuse, not re-upload
    sizes_after_step1 = [len(b._mask_cache) for b in blocks]
    assert sizes_after_step1 == sizes_after_step0, "mask cache must not grow across rollout steps"
    print(f"\n[mask cache] cached masks/block={sizes_after_step0} (stable across steps)")


def test_full_model_pcc(device, ref_model, ref_forecast):
    """Full hybrid Aurora (TT backbone) vs full reference, real weights."""
    batch, ref_pred = ref_forecast

    attach_tt_backbone(ref_model, device, use_lora=False)
    with torch.no_grad():
        tt_pred = ref_model.forward(batch)

    ps = []
    for k in ref_pred.surf_vars:
        p = pcc(ref_pred.surf_vars[k], tt_pred.surf_vars[k])
        ps.append(p)
        print(f"[surf {k}] PCC={p:.5f}")
    for k in ref_pred.atmos_vars:
        p = pcc(ref_pred.atmos_vars[k], tt_pred.atmos_vars[k])
        ps.append(p)
        print(f"[atmos {k}] PCC={p:.5f}")
    worst = min(ps)
    print(f"\n[full model] worst-variable PCC={worst:.5f}")
    assert worst > 0.97, f"full-model PCC too low: {worst}"


def test_full_model_pcc_bfp8(device, ref_model, ref_forecast):
    """Optimization path: bfloat8_b-packed backbone weights (2x weight bandwidth).

    Validates that block-float8 weight packing on QKV/proj/MLP keeps the
    full-model forecast within tolerance of the reference.
    """
    batch, ref_pred = ref_forecast

    attach_tt_backbone(ref_model, device, use_lora=False, weight_dtype=ttnn.bfloat8_b)
    with torch.no_grad():
        tt_pred = ref_model.forward(batch)

    ps = []
    for k in ref_pred.surf_vars:
        ps.append(pcc(ref_pred.surf_vars[k], tt_pred.surf_vars[k]))
    for k in ref_pred.atmos_vars:
        ps.append(pcc(ref_pred.atmos_vars[k], tt_pred.atmos_vars[k]))
    worst = min(ps)
    print(f"\n[full model, bfp8_b weights] worst-variable PCC={worst:.5f}")
    assert worst > 0.95, f"bfp8_b full-model PCC too low: {worst}"
