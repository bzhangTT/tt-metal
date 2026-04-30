# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc


def torch_snake_beta(x, alpha, beta):
    """Reference: y = x + sin(alpha*x)^2 / beta (canonical SnakeBeta)."""
    return x + torch.pow(torch.sin(alpha * x), 2) / beta


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_issue_shape(device):
    """Issue #43337 exact shapes: x=[1,1,640,48], alpha=beta=[48], bf16."""
    torch.manual_seed(0)
    x_torch = torch.randn(1, 1, 640, 48, dtype=torch.bfloat16)
    alpha_torch = torch.randn(48, dtype=torch.bfloat16) * 0.5 + 1.0  # ~ 1.0 +/- 0.5
    beta_torch = torch.randn(48, dtype=torch.bfloat16) * 0.5 + 1.0  # ensure non-zero

    expected = torch_snake_beta(x_torch, alpha_torch, beta_torch)

    x_tt = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, device=device)
    alpha_tt = ttnn.from_torch(alpha_torch, layout=ttnn.TILE_LAYOUT, device=device)
    beta_tt = ttnn.from_torch(beta_torch, layout=ttnn.TILE_LAYOUT, device=device)

    result_tt = ttnn.snake_beta(x_tt, alpha_tt, beta_tt)
    result = ttnn.to_torch(result_tt)

    assert_with_pcc(expected, result, pcc=0.999)


@pytest.mark.parametrize(
    "dtype,torch_dtype,pcc",
    [
        (ttnn.bfloat16, torch.bfloat16, 0.999),
        (ttnn.float32, torch.float32, 0.9999),
    ],
)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_dtypes(device, dtype, torch_dtype, pcc):
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 32, dtype=torch_dtype)
    a = torch.randn(32, dtype=torch_dtype) * 0.5 + 1.0
    b = torch.randn(32, dtype=torch_dtype) * 0.5 + 1.0
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=dtype, device=device)

    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_pcc(expected, result, pcc=pcc)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_no_broadcast(device):
    """alpha/beta same full shape as x — exercises NONE broadcast path."""
    torch.manual_seed(1)
    x = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16)
    a = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16) * 0.5 + 1.0
    b = torch.randn(1, 1, 32, 32, dtype=torch.bfloat16) * 0.5 + 1.0
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_pcc(expected, result, pcc=0.999)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_outer_bcast(device):
    """Larger outer dims; alpha/beta still [W] — exercises OUTER_BCAST path."""
    torch.manual_seed(2)
    x = torch.randn(2, 4, 64, 32, dtype=torch.bfloat16)
    a = torch.randn(32, dtype=torch.bfloat16) * 0.5 + 1.0
    b = torch.randn(32, dtype=torch.bfloat16) * 0.5 + 1.0
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_pcc(expected, result, pcc=0.999)


@pytest.mark.parametrize("alpha_val", [0.0, 1.0, 3.0])
@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_alpha_variations(device, alpha_val):
    """alpha=0 reduces to identity; alpha=1 standard; alpha=3 stresses range reduction."""
    torch.manual_seed(3)
    x = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16)
    a = torch.full((32,), alpha_val, dtype=torch.bfloat16)
    b = torch.ones(32, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    # PCC tolerance loosens slightly for alpha=3 due to bf16 precision in alpha*x
    pcc = 0.998 if alpha_val == 3.0 else 0.999
    assert_with_pcc(expected, result, pcc=pcc)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_beta_small(device):
    """beta near 0.1 — tests reciprocal accuracy."""
    torch.manual_seed(4)
    x = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16) * 0.1
    a = torch.ones(32, dtype=torch.bfloat16)
    b = torch.full((32,), 0.1, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_pcc(expected, result, pcc=0.99)  # looser due to small-beta amplification


# --- Validation failure tests ---


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_shape_mismatch_alpha_beta(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"alpha.shape == beta.shape|Broadcasting rule violation"):
        ttnn.snake_beta(x, a, b)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_dtype_mismatch(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.float32), layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"dtype"):
        ttnn.snake_beta(x, a, b)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_w_mismatch(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"input.W == alpha.W|Broadcasting rule violation"):
        ttnn.snake_beta(x, a, b)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_unsupported_broadcast(device):
    """alpha non-W dim is non-1 (H=64) — invalid for v1."""
    x = ttnn.from_torch(torch.zeros(1, 1, 64, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(1, 1, 64, 1, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(1, 1, 64, 1, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"non-1 size only on the last dim|input.W == alpha.W|broadcast"):
        ttnn.snake_beta(x, a, b)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 0}], indirect=True)
def test_snake_beta_row_major_layout(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"[Tt]ile [Ll]ayout|tile layout"):
        ttnn.snake_beta(x, a, b)
