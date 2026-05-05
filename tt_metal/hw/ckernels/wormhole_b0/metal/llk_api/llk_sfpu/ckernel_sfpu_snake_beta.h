// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_recip.h"

using namespace sfpi;

namespace ckernel::sfpu {

// SnakeBeta activation: y = x + sin²(alpha*x) / beta
//
// Algorithm:
//   1. ax      = alpha * x
//   2. k       = round(ax / π)  via float_to_int16
//   3. a       = (ax/π - k) * π   a ∈ (-π/2, π/2]
//      This single-pass formula keeps full float32 precision in the fractional
//      part of ax/π; a 2-stage CW split of π would accumulate k×residual error
//      which for large k exceeds the single-pass approach.
//   4. Evaluate sin(a) using the proven minimax polynomial from calculate_sine():
//      - fp32 (degree-3): a + a·s·(C0 + s·(C1 + s·(C2 + s·C3)))   s = a²
//      - bf16 (degree-2): a + a·s·(C0 + s·(C1 + s·C2))
//      No quadrant sign-flip needed: sin² is even (sin²(a) = sin²(-a)).
//   5. sin²(ax) = sin(a)²     (0.41 ULP max vs 2.26 ULP from the old degree-5 sin² poly)
//   6. y = x + sin²(ax) / beta
//      _sfpu_reciprocal_<2> for fp32 (≤1 ULP), <1> for bf16 (≤0.5 ULP, 1 NR sufficient).
//
// vConstFloatPrgm0/1/2 are dedicated to the reciprocal quadratic-estimate init
// throughout the kernel.  The range reduction uses only local constants.
//
// dst layout (matches ternary_sfpu_*_ttt compute kernel):
//   dst_index_x     : input x  (also written as output when dst_index_out == dst_index_x)
//   dst_index_alpha : input alpha (broadcast in reader)
//   dst_index_beta  : input beta  (broadcast in reader)
//   dst_index_out   : output y
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS = 8>
inline void calculate_snake_beta(uint dst_index_x, uint dst_index_alpha, uint dst_index_beta, uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "snake_beta supports only Float32 and Float16_b");

    constexpr uint dst_tile_size_sfpi = 32;
    constexpr float one_over_pi = 0.318309886183791f;
    constexpr float pi_f = 3.141592653589793f;

    // sin(a) minimax polynomial on a ∈ (-π/2, π/2]: sin(a) = a + a·s·poly(s), s = a².
    // Coefficients identical to calculate_sine() in ckernel_sfpu_trigonometry.h.
    // fp32 degree-3: max 0.20 ULP; bf16 degree-2: max 0.58 bf16-ULP (< 1 bf16-ULP).
    constexpr float fp32_C3 = 0x1.5dc908p-19f;
    constexpr float fp32_C2 = -0x1.9f70fp-13f;
    constexpr float fp32_C1 = 0x1.110edap-7f;
    constexpr float fp32_C0 = -0x1.55554cp-3f;
    constexpr float bf16_C2 = -0x1.8b10a4p-13f;
    constexpr float bf16_C1 = 0x1.10c2a2p-7f;
    constexpr float bf16_C0 = -0x1.5554a4p-3f;

#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[dst_index_x * dst_tile_size_sfpi];
        vFloat alpha = dst_reg[dst_index_alpha * dst_tile_size_sfpi];
        vFloat beta = dst_reg[dst_index_beta * dst_tile_size_sfpi];

        vFloat ax = alpha * x;

        // Range reduction: a = (ax/π - k) * π  where k = round(ax/π), a ∈ (-π/2, π/2].
        // float_to_int16 avoids using vConstFloatPrgm2 (reserved for the reciprocal init).
        vFloat ax_over_pi = ax * one_over_pi;
        vInt k = float_to_int16(ax_over_pi, 0);
        vFloat k_f = int32_to_float(k, 0);
        vFloat a = (ax_over_pi - k_f) * pi_f;

        // Evaluate sin(a) on a ∈ (-π/2, π/2]; no quadrant sign flip needed (sin² is even).
        // Horner: sin(a) = a + a*s*(C0 + s*(C1 + s*(C2 [+ s*C3]))), s = a².
        vFloat s = a * a;
        vFloat r, c;
        if constexpr (is_fp32_dest_acc_en) {
            r = fp32_C3 * s + fp32_C2;
            r = r * s + fp32_C1;
            c = a * s;  // a³
            r = r * s + fp32_C0;
            r = r * c + a;
        } else {
            r = bf16_C2 * s + bf16_C1;
            c = a * s;  // a³
            r = r * s + bf16_C0;
            r = r * c + a;
        }

        // sin²(ax) = sin(a)²: 0.41 ULP max vs 2.26 ULP for the old degree-5 sin² poly.
        vFloat sin2_ax = r * r;

        // 1 NR iter for bf16 (≤0.5 ULP); 2 iters for fp32 (≤1 ULP).
        // Uses vConstFloatPrgm0/1/2 loaded by snake_beta_init().
        vFloat inv_beta;
        if constexpr (is_fp32_dest_acc_en) {
            inv_beta = _sfpu_reciprocal_<2>(beta);
        } else {
            inv_beta = _sfpu_reciprocal_<1>(beta);
        }
        vFloat result = x + sin2_ax * inv_beta;

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result);
        }

        dst_reg[dst_index_out * dst_tile_size_sfpi] = result;
        dst_reg++;
    }
}

template <bool APPROXIMATE>
inline void snake_beta_init() {
    _init_sfpu_reciprocal_<APPROXIMATE>();
}

}  // namespace ckernel::sfpu
