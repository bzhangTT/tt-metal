// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "lltt.h"

namespace ckernel
{
namespace sfpu
{
// Calculates SQUARE for number of rows of output SFPU ops (Quasar = 2 rows)
template <bool APPROXIMATION_MODE>
inline void _calculate_square_sfp_rows_()
{
    TTI_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0); // load from dest into lreg[0]
    // Multiply LREG0 * LREG0, store result in LREG0
    TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0);
    TTI_SFPNOP(0, 0, 0);
    // Store result back to destination
    TTI_SFPSTORE(p_sfpu::LREG0, p_sfpu::sfpmem::DEFAULT, ADDR_MOD_7, 0, 0);
}

inline void _calculate_square_(const int iterations)
{
    int d = 0;
    for (; d < (iterations & ~7); d += 8)
    {
        lltt::record<lltt::Exec>(0, 5);
        _calculate_square_sfp_rows_<false>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
        lltt::replay(0, 5);
        lltt::replay(0, 5);
        lltt::replay(0, 5);
        lltt::replay(0, 5);
        lltt::replay(0, 5);
        lltt::replay(0, 5);
        lltt::replay(0, 5);
    }
    for (; d < iterations; d++)
    {
        _calculate_square_sfp_rows_<false>();
        ckernel::math::_incr_counters_<0x0, 0x0, ckernel::math::SFP_ROWS, 0x0>();
    }
}

} // namespace sfpu
} // namespace ckernel
