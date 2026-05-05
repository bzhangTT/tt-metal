# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from tests.ttnn.utils_for_testing import assert_with_pcc, assert_with_ulp


def torch_snake_beta(x, alpha, beta):
    """Reference: y = x + sin(alpha*x)^2 / beta (canonical SnakeBeta)."""
    return x + torch.pow(torch.sin(alpha * x), 2) / beta


def test_snake_beta_issue_shape(device):
    """Issue #43337 exact shapes: x=[1,1,640,48], alpha=beta=[48], bf16."""
    torch.manual_seed(0)
    x_torch = torch.randn(1, 1, 640, 48, dtype=torch.bfloat16)
    alpha_torch = torch.randn(48, dtype=torch.bfloat16) * 0.5 + 1.0
    beta_torch = torch.randn(48, dtype=torch.bfloat16) * 0.5 + 1.0

    expected = torch_snake_beta(x_torch, alpha_torch, beta_torch)

    x_tt = ttnn.from_torch(x_torch, layout=ttnn.TILE_LAYOUT, device=device)
    alpha_tt = ttnn.from_torch(alpha_torch, layout=ttnn.TILE_LAYOUT, device=device)
    beta_tt = ttnn.from_torch(beta_torch, layout=ttnn.TILE_LAYOUT, device=device)

    result_tt = ttnn.snake_beta(x_tt, alpha_tt, beta_tt)
    result = ttnn.to_torch(result_tt)

    assert_with_pcc(expected, result, pcc=0.999)


def test_snake_beta_dtype_fp32(device):
    """fp32 dtype path."""
    torch.manual_seed(0)
    x = torch.randn(1, 1, 64, 32, dtype=torch.float32)
    a = torch.randn(32, dtype=torch.float32) * 0.5 + 1.0
    b = torch.randn(32, dtype=torch.float32) * 0.5 + 1.0
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)

    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert torch.allclose(expected, result, rtol=1e-5, atol=1e-5)


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
def test_snake_beta_alpha_variations(device, alpha_val):
    """alpha = 0 / 1 / 3."""
    torch.manual_seed(3)
    x = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16)
    a = torch.full((32,), alpha_val, dtype=torch.bfloat16)
    b = torch.ones(32, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    if alpha_val == 0.0:
        assert_with_ulp(expected, result, ulp_threshold=1)
    elif alpha_val == 1.0:
        assert_with_ulp(expected, result, ulp_threshold=3)
    else:
        assert_with_pcc(expected, result, pcc=0.998)


def test_snake_beta_beta_small(device):
    """beta = 0.1 — small-divisor reciprocal."""
    torch.manual_seed(4)
    x = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16) * 0.1
    a = torch.ones(32, dtype=torch.bfloat16)
    b = torch.full((32,), 0.1, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_pcc(expected, result, pcc=0.99)


# --- Edge-case / close-to-boundary tests ---


def test_snake_beta_zero_x(device):
    """x = 0 → y = 0."""
    x = torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16)
    a = torch.ones(32, dtype=torch.bfloat16)
    b = torch.ones(32, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_ulp(expected, result, ulp_threshold=1)


def test_snake_beta_pi_boundary(device):
    """alpha*x at multiples of pi/2 — cosine quadrant boundaries."""
    import math

    x_vals = torch.tensor(
        [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi, -math.pi / 2, -math.pi],
        dtype=torch.bfloat16,
    )
    x_tile = x_vals.repeat(math.ceil(1024 / len(x_vals)))[:1024].reshape(1, 1, 32, 32)
    a = torch.full((32,), 1.0, dtype=torch.bfloat16)
    b = torch.ones(32, dtype=torch.bfloat16)
    expected = torch_snake_beta(x_tile, a, b)

    x_tt = ttnn.from_torch(x_tile, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_ulp(expected, result, ulp_threshold=1)


def test_snake_beta_large_alpha(device):
    """Large alpha (10, 50) — stresses range reduction."""
    torch.manual_seed(5)
    for alpha_val in [10.0, 50.0]:
        x = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16)
        a = torch.full((32,), alpha_val, dtype=torch.bfloat16)
        b = torch.ones(32, dtype=torch.bfloat16)
        expected = torch_snake_beta(x, a, b)

        x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
        a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
        b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
        result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
        assert_with_pcc(expected, result, pcc=0.97)


def test_snake_beta_near_origin(device):
    """Small x: near-origin polynomial regime."""
    torch.manual_seed(6)
    x = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16) * 0.01
    a = torch.ones(32, dtype=torch.bfloat16)
    b = torch.ones(32, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_ulp(expected, result, ulp_threshold=2)


def test_snake_beta_negative_alpha(device):
    """Negative alpha — sin² is even."""
    torch.manual_seed(7)
    x = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16)
    a_pos = torch.ones(32, dtype=torch.bfloat16)
    a_neg = -torch.ones(32, dtype=torch.bfloat16)
    b = torch.ones(32, dtype=torch.bfloat16)

    expected_pos = torch_snake_beta(x, a_pos, b)
    expected_neg = torch_snake_beta(x, a_neg, b)
    assert torch.allclose(expected_pos, expected_neg, atol=1e-3)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a_neg, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_ulp(expected_neg, result, ulp_threshold=3)


def test_snake_beta_large_x(device):
    """Large x values (±10)."""
    torch.manual_seed(8)
    x = torch.randn(1, 1, 64, 32, dtype=torch.bfloat16) * 10.0
    a = torch.ones(32, dtype=torch.bfloat16)
    b = torch.ones(32, dtype=torch.bfloat16)
    expected = torch_snake_beta(x, a, b)

    x_tt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=device)
    a_tt = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=device)
    b_tt = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=device)
    result = ttnn.to_torch(ttnn.snake_beta(x_tt, a_tt, b_tt))
    assert_with_ulp(expected, result, ulp_threshold=3)


# --- Validation failure tests ---


def test_snake_beta_shape_mismatch_alpha_beta(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"alpha.shape == beta.shape|Broadcasting rule violation"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_dtype_mismatch(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.float32), layout=ttnn.TILE_LAYOUT, dtype=ttnn.float32, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"dtype"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_w_mismatch(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 64, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"input.W == alpha.W|Broadcasting rule violation"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_unsupported_broadcast(device):
    """alpha non-W dim is non-1 (H=64) — invalid for v1."""
    x = ttnn.from_torch(torch.zeros(1, 1, 64, 32, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(1, 1, 64, 1, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(1, 1, 64, 1, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"non-1 size only on the last dim|input.W == alpha.W|broadcast"):
        ttnn.snake_beta(x, a, b)


def test_snake_beta_row_major_layout(device):
    x = ttnn.from_torch(torch.zeros(1, 1, 32, 32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    a = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    b = ttnn.from_torch(torch.ones(32, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    with pytest.raises(RuntimeError, match=r"[Tt]ile [Ll]ayout|tile layout"):
        ttnn.snake_beta(x, a, b)
