// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "ckernel_defs.h"
#include "noc_nonblocking_api.h"
#include "sfpi.h"
#include "sfpu/ckernel_sfpu_recip.h"
#include "sfpu/ckernel_sfpu_trigonometry.h"  // legacy _sfpu_sine_maclaurin_series_

using namespace sfpi;

namespace ckernel::sfpu {

// SnakeBeta activation: y = x + sin(alpha*x)^2 / beta
//
// dst layout (matches ternary_sfpu_*_ttt compute kernel):
//   dst_index_x     : input x  (also overwritten by output if dst_index_out == dst_index_x)
//   dst_index_alpha : input alpha (broadcast in reader)
//   dst_index_beta  : input beta  (broadcast in reader)
//   dst_index_out   : output y
//
// Each tile is 32 sfpi rows; outer loop processes 1 tile across ITERATIONS=8 chunks
// (8 chunks * 4 sfpi rows = 32 rows; matches the cadence used by lerp/addcdiv).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS = 8>
inline void calculate_snake_beta(uint dst_index_x, uint dst_index_alpha, uint dst_index_beta, uint dst_index_out) {
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b,
        "snake_beta supports only Float32 and Float16_b");

    constexpr uint dst_tile_size_sfpi = 32;
    constexpr float pi_f = 3.141592653589793f;
    constexpr float one_over_pi = 0.318309886183791f;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat x = dst_reg[dst_index_x * dst_tile_size_sfpi];
        vFloat alpha = dst_reg[dst_index_alpha * dst_tile_size_sfpi];
        vFloat beta = dst_reg[dst_index_beta * dst_tile_size_sfpi];

        // Range reduction: bring (alpha*x) into [-pi, pi], track sign for odd half-periods.
        // Pattern mirrors the prologue of _calculate_sine_ in legacy ckernel_sfpu_trigonometry.h.
        vFloat ax = alpha * x;
        vFloat ax_over_pi = ax * one_over_pi;
        vInt whole = float_to_int16(ax_over_pi, 0);
        vFloat whole_f = int32_to_float(whole, 0);
        vFloat ax_reduced = (ax_over_pi - whole_f) * pi_f;  // in [-pi, pi]

        vFloat s = _sfpu_sine_maclaurin_series_<APPROXIMATION_MODE>(ax_reduced);

        // Sign flip when whole-part is odd (tracks half-period parity).
        vInt parity = whole & 0x1;
        v_if(parity != 0) { s = -s; }
        v_endif;

        vFloat s2 = s * s;
        vFloat inv_beta = _sfpu_reciprocal_<2>(beta);  // 2 NR iterations, ~22-bit precision
        vFloat result = x + s2 * inv_beta;

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
    // Trig helpers in legacy header don't require an init; range reduction is open-coded above.
}

}  // namespace ckernel::sfpu
