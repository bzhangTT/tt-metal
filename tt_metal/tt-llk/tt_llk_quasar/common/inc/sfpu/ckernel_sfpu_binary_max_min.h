// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_trisc_common.h"
#include "cmath_common.h"
#include "sfpi.h"

namespace ckernel
{
namespace sfpu
{

// imm12 bit 11 = 1: SFPSETCC interprets src_c as two's-complement INT32, not FP32/SMAG32
constexpr std::uint32_t SFPSETCC_INT32_SIGNBIT = 0x800;

// mod1=9 — RESERVED in assembly.yaml but maps to the `default:` case in the
// Confluence SFPSWAP functional model, which sets VDGetsMin=0 (DST=max, VC=min).
// Used for the unsigned INT32 initial swap (IS_UNSIGNED=true path).
constexpr std::uint32_t SFPSWAP_DEFAULT_VD_MAX = 9;

// Float-variant init: empty no-op on Quasar.
// The reference's init body programs SFPLOADMACRO tables via TTI_SFPCONFIG;
// those tables are unconditionally disabled on Quasar, so there is nothing to do.
template <bool IS_MAX_OP = true>
inline void binary_max_min_init()
{
}

// Int32-variant init: same rationale.
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init()
{
}

// Float variant — inner row body.
// Loads two FP rows from separate Dest tile regions, computes element-wise
// min+max via SFPSWAP(mod1=1), then stores the result for IS_MAX_OP.
// IS_FP16A: set to true when the Dest format is IEEE Float16 (FP16A, 5-bit exponent).
// In 16-bit SFPU mode with DEFAULT load, Float16A values are not expanded to proper FP32
// in LREG, causing SFPSWAP sign-magnitude comparison to fail for negative pairs.
// Explicitly using FP16A mode fixes this by ensuring correct FP16A→FP32 expansion.
//
// @param offset0    Base Dest address (in 16b units) of the in0 tile region.
// @param offset1    Base Dest address (in 16b units) of the in1 tile region.
// @param offset2    Base Dest address (in 16b units) of the output tile region.
// @param row_index  Row index within the tile [0, ITERATIONS); selects which of the 32-lane SFPU rows to process.
template <bool IS_MAX_OP = true, bool IS_FP16A = false>
inline void _calculate_binary_max_min_sfp_rows_(const std::uint32_t offset0, const std::uint32_t offset1, const std::uint32_t offset2, const int row_index)
{
    constexpr std::uint32_t load_mode = IS_FP16A ? p_sfpu::sfpmem::FP16A : p_sfpu::sfpmem::DEFAULT;
    TT_SFPLOAD(p_sfpu::LREG0, load_mode, ADDR_MOD_7, 0 /* done */, offset0 + (row_index << 1)); // load FP row from in0
    TT_SFPLOAD(p_sfpu::LREG1, load_mode, ADDR_MOD_7, 0 /* done */, offset1 + (row_index << 1)); // load FP row from in1
    // VEC_MIN_MAX: VD=LREG0 → min, VC=LREG1 → max (sign-magnitude order = FP32 total order)
    TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);                        // 2-cycle; LREG0=min, LREG1=max
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */);                                       // post-SFPSWAP stall avoidance
    TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, load_mode, ADDR_MOD_7, 0 /* done */, offset2 + (row_index << 1)); // store max (LREG1) or min (LREG0)
}

// Float variant — outer loop.
// Explicit offset arithmetic for all three tile regions; no _incr_counters_
// needed because loads/stores use absolute addresses, not the auto-increment pointer.
//
// @param dst_index_in0  Dest tile index of input 0 (in tile units).
// @param dst_index_in1  Dest tile index of input 1 (in tile units).
// @param dst_index_out  Dest tile index where the result tile is written (in tile units).
template <bool IS_MAX_OP = true, bool IS_FP16A = false, int ITERATIONS = 8>
inline void calculate_binary_max_min(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    const std::uint32_t offset0 = (dst_index_in0 * 32) << 1;
    const std::uint32_t offset1 = (dst_index_in1 * 32) << 1;
    const std::uint32_t offset2 = (dst_index_out * 32) << 1;
#pragma GCC unroll 8
    for (int row_index = 0; row_index < ITERATIONS; row_index++)
    {
        _calculate_binary_max_min_sfp_rows_<IS_MAX_OP, IS_FP16A>(offset0, offset1, offset2, row_index);
    }
}

// Int32 variant — inner row body.
// SFPSWAP uses sign-magnitude comparison, which differs from two's-complement
// for pairs of negative values. A CC-guarded correction swap fixes those cases.
//
// @param offset0    Base Dest address (in 16b units) of the in0 tile region.
// @param offset1    Base Dest address (in 16b units) of the in1 tile region.
// @param offset2    Base Dest address (in 16b units) of the output tile region.
// @param row_index  Row index within the tile; selects which of the 32-lane SFPU rows to process.
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void _calculate_binary_max_min_int32_sfp_rows_(
    const std::uint32_t offset0, const std::uint32_t offset1, const std::uint32_t offset2, const int row_index)
{
    TT_SFPLOAD(p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, offset0 + (row_index << 1)); // load INT32 row from in0
    TT_SFPLOAD(p_sfpu::LREG1, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, offset1 + (row_index << 1)); // load INT32 row from in1

    // Step 1: sign-magnitude min/max.
    // Signed (IS_UNSIGNED=false): mod1=1 → VD=LREG0 gets SM-min, VC=LREG1 gets SM-max.
    // Unsigned (IS_UNSIGNED=true): mod1=9 (default case per Confluence) → VD=LREG0 gets SM-max, VC=LREG1 gets SM-min.
    if constexpr (IS_UNSIGNED)
    {
        TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, SFPSWAP_DEFAULT_VD_MAX); // 2-cycle; VD=max, VC=min in SM order
    }
    else
    {
        TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX); // 2-cycle; VD=min, VC=max in SM order
    }
    TTI_SFPNOP(0 /* srcs_wr_done */, 0 /* srcs_rd_done */, 0 /* dest_done */); // post-SFPSWAP stall avoidance

    // Step 2: CC-guarded correction swap.
    // For sign-magnitude, negative two's-complement values (bit 31=1) look like
    // large positive values, so SFPSWAP inverts the ordering for (neg, neg) pairs.
    // Detect the condition on both operands, then re-swap only those rows.
    // imm12=SFPSETCC_INT32_SIGNBIT tells SFPSETCC to interpret src_c as INT32.
    TTI_SFPSETCC(
        SFPSETCC_INT32_SIGNBIT,
        p_sfpu::LREG0,
        IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0); // set CC where LREG0 meets sign condition
    TTI_SFPSETCC(
        SFPSETCC_INT32_SIGNBIT,
        p_sfpu::LREG1,
        IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);   // extend CC where LREG1 also meets condition
    TTI_SFPSWAP(0 /* imm12 */, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP); // re-swap rows where both operands met the condition
    TTI_SFPENCC(0 /* imm12 */, 0 /* mod1: clear CC */);                                // clear CC result

    TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, p_sfpu::sfpmem::INT32, ADDR_MOD_7, 0 /* done */, offset2 + (row_index << 1)); // store INT32 result
}

// Int32 variant — outer loop.
//
// @param dst_index_in0  Dest tile index of input 0 (in tile units).
// @param dst_index_in1  Dest tile index of input 1 (in tile units).
// @param dst_index_out  Dest tile index where the result tile is written (in tile units).
template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{
    const std::uint32_t offset0 = (dst_index_in0 * 32) << 1;
    const std::uint32_t offset1 = (dst_index_in1 * 32) << 1;
    const std::uint32_t offset2 = (dst_index_out * 32) << 1;
#pragma GCC unroll 8
    for (int row_index = 0; row_index < ITERATIONS; row_index++)
    {
        _calculate_binary_max_min_int32_sfp_rows_<IS_MAX_OP, IS_UNSIGNED>(offset0, offset1, offset2, row_index);
    }
}

} // namespace sfpu
} // namespace ckernel
