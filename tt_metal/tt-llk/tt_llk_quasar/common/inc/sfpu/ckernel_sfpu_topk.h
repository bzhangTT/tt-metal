// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

#include "ckernel_ops.h"
#include "ckernel_trisc_common.h"
#include "cmath_common.h"

// Quasar's replay buffer does NOT preserve the SFPSWAP auto-stall behavior, so
// SFPSWAPs executed via TTI_REPLAY produce incorrect results (observed as
// all-zeros output during initial bring-up). Force the inline (no-replay) code
// paths until Quasar replay-buffer behavior is fully characterized. Callers may
// still override by defining TOPK_DISABLE_REPLAY=0 at the build level.
#ifndef TOPK_DISABLE_REPLAY
#define TOPK_DISABLE_REPLAY
#endif

namespace ckernel
{
namespace sfpu
{

// SFPSWAP modes (mode-to-int mapping is the same as the Blackhole reference).
// Defined locally because Quasar's ckernel_instr_params.h does not provide p_sfpswap.
struct p_sfpswap
{
    constexpr static std::uint32_t UNCONDITIONALLY = 0;
    constexpr static std::uint32_t ALL_ROWS_MAX    = 1;
    constexpr static std::uint32_t ROWS_01_MAX     = 2;
    constexpr static std::uint32_t ROWS_02_MAX     = 3;
    constexpr static std::uint32_t ROWS_03_MAX     = 4;
    constexpr static std::uint32_t ROW_0_MAX       = 5;
    constexpr static std::uint32_t ROW_1_MAX       = 6;
    constexpr static std::uint32_t ROW_2_MAX       = 7;
    constexpr static std::uint32_t ROW_3_MAX       = 8;
};

// Sort direction for topk. Defined locally — Quasar's llk_defs.h does not define SortDir.
enum SortDir : bool
{
    ArgMax = false,
    ArgMin = true,
};

// Tracks which replay-buffer contents are currently loaded across topk pipeline calls.
// Reset by _init_topk; written by phase 5/6 (local sort / merge / rebuild) helpers.
static std::int32_t topk_replay_init = 0;

// Set the per-TRISC dest section base register for the math TRISC.
// Quasar has separate SEC0..SEC3 registers (one per TRISC); SFPLOAD/SFPSTORE on the
// math TRISC compute their effective dest address as SEC1 + dest_counter + dest_reg_addr.
inline void set_dst_write_addr(std::uint32_t addr)
{
    std::uint32_t dst_index = addr + ckernel::trisc::_get_dest_buffer_base_();
    ckernel::trisc::_set_dest_section_base_<ckernel::math::TRISC_ID>(dst_index);
}

// Advance the dest RWC counter by `inc` rows in groups of 8.
// `cr=true` issues an additional carriage-return bit to clear the column counter.
// Quasar TTI_INCRWC arg order is (rwc_cr, rwc_a, rwc_b, rwc_d) — dest is the 4th argument.
inline void bitonic_topk_inc_x8_dest(std::uint32_t inc, bool cr)
{
    std::uint32_t inc_grp8 = inc >> 3;
    if (cr)
    {
        for (std::uint32_t i = 0; i < inc_grp8; i++)
        {
            TTI_INCRWC(0b100, 0, 0, 8);
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < inc_grp8; i++)
        {
            TTI_INCRWC(0, 0, 0, 8);
        }
    }
}

// Same as bitonic_topk_inc_x8_dest but increments by 4 rows per call.
// Kept for reference parity; not used by the current topk body.
inline void bitonic_topk_inc_x4_dest(std::uint32_t inc, bool cr)
{
    std::uint32_t inc_grp4 = inc >> 2;
    if (cr)
    {
        for (std::uint32_t i = 0; i < inc_grp4; i++)
        {
            TTI_INCRWC(0b100, 0, 0, 4);
        }
    }
    else
    {
        for (std::uint32_t i = 0; i < inc_grp4; i++)
        {
            TTI_INCRWC(0, 0, 0, 4);
        }
    }
}

inline void _init_topk()
{
    // Reset file-scope replay-buffer tracker so phase 5/6 know nothing is loaded yet.
    topk_replay_init = 0;

    // Write 0x4 to LaneConfig (config_dest=0xF) to set bit [2] = ENABLE_DEST_INDEX.
    // With this bit set, SFPSWAP performs argmin/argmax: when it
    // conditionally swaps LREG[VC] <-> LREG[VD], it also swaps
    // LREG[4 + (VC&3)] <-> LREG[4 + (VD&3)] in lockstep — letting topk track input
    // indices alongside the values being sorted.
    //
    // [Hypothesis B/F] Use the BH-reference SFPCONFIG path: load value into LREG0
    // via SFPLOADI, then SFPCONFIG with InstrMod=0 (LREG0-source). Empirically,
    // this gives better results than the Imm16-source path on Quasar — most likely
    // due to a simulator-side timing difference between the two code paths.
    ckernel::math::_sfpu_load_config32_(0xF, 0x0, 0x4);
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPNOP(0, 0, 0);
}

// Load 8 lanes (one value LREG pair + one index LREG pair) from Dest at runtime offsets.
// Values land in LREG0,1; indices (offset by dst_indices_offset = 128) land in LREG4,5.
// Index format mode is INT32 when dest_acc=fp32, FP16B otherwise (workaround for
// UINT16/LO16 dest-cell layout mismatch on Quasar — indices 0..127 are exactly
// representable in Float16_b so we use the upper-half FP16B path).
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    // Workaround: Use FP16B (bfloat16) mode for indices instead of UINT16 (LO16, mode 0b0110).
    // Indices 0..127 are exactly representable in Float16_b. The UINT16/LO16 path has a
    // dest-cell layout mismatch on Quasar (unpacker writes upper half via Float16_b A2D
    // datacopy, but SFPU LO16 reads the lower half). FP16B reads/writes the upper half,
    // which matches what the unpacker deposits.
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::FP16B;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Values
    TT_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, ld_offset);
    TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, ld_offset + dist);

    // Indices (paired with LREG0,1; shifted by dst_indices_offset).
    TT_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset);
    TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset + dist);
}

// Store 8 lanes (the same LREGs that bitonic_topk_load8 fills) back into Dest. Mirrors load8.
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_store8(std::uint32_t offset, std::uint32_t dist)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    // FP16B index mode (see bitonic_topk_load8 for rationale).
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::FP16B;

    std::uint32_t face_offset = offset >> 4;
    std::uint32_t ld_offset   = (offset & 0xF) + face_offset * 32;

    // Values
    TT_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, ld_offset);
    TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, ld_offset + dist);

    // Indices
    TT_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset);
    TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + ld_offset + dist);
}

// Load 16 lanes (LREG0..3 values + LREG4..7 indices) from Dest at strided offsets
// (0, dist0, dist1, dist1+dist0). The (dist0,dist1)==(4,8) call site is the hot path
// used by phases 5/6 — its addresses are constexpr so we hand them to TTI_SFPLOAD.
template <bool is_fp32_dest_acc_en>
inline void bitonic_topk_load16(std::uint32_t dist0, std::uint32_t dist1)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    // FP16B index mode (see bitonic_topk_load8 for rationale).
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::FP16B;

    // Values
    TTI_SFPLOAD(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, 4);
        TTI_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 0, 8);
        TTI_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 0, 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, dist0);
        TT_SFPLOAD(p_sfpu::LREG2, 0, ADDR_MOD_7, 0, dist1);
        TT_SFPLOAD(p_sfpu::LREG3, 0, ADDR_MOD_7, 0, dist1 + dist0);
    }

    // Indices (paired with LREG0..3; shifted by dst_indices_offset).
    TTI_SFPLOAD(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 4);
        TTI_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 8);
        TTI_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 12);
    }
    else
    {
        TT_SFPLOAD(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist0);
        TT_SFPLOAD(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1);
        TT_SFPLOAD(p_sfpu::LREG7, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1 + dist0);
    }
}

// Store 16 lanes (the same LREGs that bitonic_topk_load16 fills) back into Dest.
// When alt_addr_mod=true, the FINAL index store (LREG7) uses ADDR_MOD_6 instead of
// ADDR_MOD_7 — phase 6 configures ADDR_MOD_6 with dest.incr=32 so this auto-advances
// Dest by 32 rows after the last store of a 16-element block.
template <bool is_fp32_dest_acc_en, bool alt_addr_mod = false>
inline void bitonic_topk_store16(std::uint32_t dist0, std::uint32_t dist1)
{
    constexpr std::uint32_t dst_indices_offset = 128;
    // FP16B index mode (see bitonic_topk_load8 for rationale).
    constexpr std::uint32_t instr_mod_index = is_fp32_dest_acc_en ? p_sfpu::sfpmem::INT32 : p_sfpu::sfpmem::FP16B;

    // Values
    TTI_SFPSTORE(p_sfpu::LREG0, 0, ADDR_MOD_7, 0, 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, 4);
        TTI_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 0, 8);
        TTI_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 0, 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG1, 0, ADDR_MOD_7, 0, dist0);
        TT_SFPSTORE(p_sfpu::LREG2, 0, ADDR_MOD_7, 0, dist1);
        TT_SFPSTORE(p_sfpu::LREG3, 0, ADDR_MOD_7, 0, dist1 + dist0);
    }

    // Indices — last store optionally swaps to ADDR_MOD_6 for the auto-advance.
    TTI_SFPSTORE(p_sfpu::LREG4, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 0);
    if ((dist0 == 4) && (dist1 == 8))
    {
        TTI_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 4);
        TTI_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + 8);
        TTI_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, 0, dst_indices_offset + 12);
    }
    else
    {
        TT_SFPSTORE(p_sfpu::LREG5, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist0);
        TT_SFPSTORE(p_sfpu::LREG6, instr_mod_index, ADDR_MOD_7, 0, dst_indices_offset + dist1);
        TT_SFPSTORE(p_sfpu::LREG7, instr_mod_index, alt_addr_mod ? ADDR_MOD_6 : ADDR_MOD_7, 0, dst_indices_offset + dist1 + dist0);
    }
}

// Phase 0, step 1 sort building block. Wrapped between two SFPTRANSPs so the swap layer
// operates across the post-transpose lane layout. STABLE_SORT=true variant duplicates each
// SFPSWAP pair (4 swaps interleaved on disjoint LREGs) for stable index tie-breaking.
template <bool STABLE_SORT>
inline void bitonic_topk_ph0_st1_to_1();

template <>
inline void bitonic_topk_ph0_st1_to_1<true>()
{
    TTI_SFPTRANSP;

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP;
}

template <>
inline void bitonic_topk_ph0_st1_to_1<false>()
{
    TTI_SFPTRANSP;

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);

    TTI_SFPTRANSP;
}

// Phase 1, steps 2 then 1. Wrapped between two SFPTRANSPs.
// STABLE_SORT=true: 4 swaps per step (interleaved on disjoint LREGs); 1-cycle stall
// between step 2 and step 1 because they share LREG1.
template <bool STABLE_SORT>
inline void bitonic_topk_ph1_st2_to_1();

template <>
inline void bitonic_topk_ph1_st2_to_1<true>()
{
    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/2 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX); // Hides LREG1/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/2 NOP

    // Step 1 (1-cycle stall: shares LREG1 with Step 2 above)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP;
}

template <>
inline void bitonic_topk_ph1_st2_to_1<false>()
{
    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_02_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_02_MAX);

    TTI_SFPTRANSP;
}

// Phase 2, steps 3, 2, then 1. Step 3 runs on the natural lane layout (no leading TRANSP),
// then a single TRANSP separates it from steps 2/1. STABLE_SORT=false adds an unconditional
// SFPSWAP(LREG2, LREG3) after step 3 — a deliberate post-step-3 reorder copied from the
// Blackhole reference (matches the BH algorithm exactly; do not remove).
template <bool STABLE_SORT>
inline void bitonic_topk_ph2_st3_to_1();

template <>
inline void bitonic_topk_ph2_st3_to_1<true>()
{
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/2 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX); // Hides LREG1/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/2 NOP

    // Step 1 (1-cycle stall: shares LREG1 with Step 2 above)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/1 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX); // Hides LREG2/3 NOP
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX); // Hides LREG0/1 NOP

    TTI_SFPTRANSP;
}

template <>
inline void bitonic_topk_ph2_st3_to_1<false>()
{
    // Step 3
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::UNCONDITIONALLY);

    TTI_SFPTRANSP;

    // Step 2
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    // Step 1
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ROWS_01_MAX);
    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ROWS_01_MAX);

    TTI_SFPTRANSP;
}

// Step-N inner-most layer used by the outer-loop in _bitonic_topk_phases_steps for the
// largest-stride compares. No SFPTRANSP — operates directly on LREG0..LREG3.
// dir==ArgMax: SFPSWAP(LREG0,LREG2) + SFPSWAP(LREG1,LREG3).
// dir==ArgMin: SFPSWAP arg order flipped (LREG2,LREG0) + (LREG3,LREG1) — comparison-direction
// inversion that's distinct from the SFPCONFIG bit-8 EXCHANGE_SRCB_SRCC mechanism phase 4 uses.
// CALLER RESPONSIBILITY: there is no trailing SFPTRANSP, so the caller must follow this with
// either a different-LREG SFPSWAP, an SFPTRANSP, or an explicit TTI_SFPNOP(0,0,0) before any
// SFPSTORE that consumes LREG0..LREG3 (avoids the SFPSWAP→SFPSTORE auto-stall hardware bug).
template <bool STABLE_SORT>
inline void bitonic_topk_step_N(bool dir);

template <>
inline void bitonic_topk_step_N<true>(bool dir)
{
    // Step N
    if (dir == static_cast<bool>(SortDir::ArgMax))
    {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX); // Hides LREG1/3 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX); // Hides LREG0/2 NOP
    }
    else
    {
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/0 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX); // Hides LREG3/1 NOP
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX); // Hides LREG2/0 NOP
    }
}

template <>
inline void bitonic_topk_step_N<false>(bool dir)
{
    // Step N
    if (dir == static_cast<bool>(SortDir::ArgMax))
    {
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
    }
    else
    {
        // Min — operand order swapped relative to ArgMax.
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG0, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG3, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
    }
}

// Phase 3 step-4-to-1: the building block for the largest-stride bitonic compare layer.
// Performs two passes of "step 4 then step 3" SFPSWAPs followed by an SFPTRANSP. The
// double execution implements the full step-4-to-1 sweep.
//
// dir==ArgMin temporarily flips LaneConfig bit [8] (EXCHANGE_SRCB_SRCC) via SFPCONFIG so
// SFPSWAP compares with reversed min/max polarity; bit [8] is cleared again before return.
// Both 0x104 (set) and 0x004 (clear) keep bit [2] (ENABLE_DEST_INDEX) so the index pairing
// invariant established by _init_topk() is preserved across calls.
//
// `replay_start` is a template int because TTI_REPLAY's start operand has a "i" inline-asm
// constraint (compile-time immediate). Phase 5 callers pass 16; phase 6 callers pass 8.
// `replay_count` = STABLE_SORT ? 9 : 5.
//
// Default build uses the replay buffer (matches the Blackhole reference). Building with
// -DTOPK_DISABLE_REPLAY emits the body inline twice instead — a diagnostic fallback for
// isolating replay-buffer ↔ SFPSWAP timing issues without changing the caller-visible
// signature or semantics.
template <bool STABLE_SORT, int replay_start>
inline void bitonic_topk_ph3_st4_to_1(bool dir, bool& init_replay)
{
#ifndef TOPK_DISABLE_REPLAY
    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(0x104, 0xF, 1); // Reverse the max/min behaviour of SWAP
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPNOP(0, 0, 0);
    }

    constexpr int replay_count = STABLE_SORT ? 9 : 5;

    if (init_replay)
    {
        if constexpr (STABLE_SORT)
        {
            load_replay_buf<replay_start, replay_count, true>(
                []
                {
                    // Step 4 — 4 interleaved SFPSWAPs on disjoint pairs (0,2)/(1,3).
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                    // Step 3 — 4 interleaved SFPSWAPs on disjoint pairs (2,3)/(0,1).
                    // 1-cycle stall vs Step 4's tail because they share LREG3.
                    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);

                    TTI_SFPTRANSP;
                });
        }
        else
        {
            load_replay_buf<replay_start, replay_count, true>(
                []
                {
                    // Step 4
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                    // Step 3
                    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);

                    TTI_SFPTRANSP;
                });
        }
        init_replay = false;
    }
    else
    {
        TTI_REPLAY(replay_start, replay_count, 0, 0, 0, 0);
    }

    // Second execution of the same buffered body. The first execution ran during the
    // load above (exec_while_loading=true) on the init_replay==true path, or via the
    // explicit TTI_REPLAY in the else branch — so by here the buffer is loaded and we
    // can replay it unconditionally for the second pass.
    TTI_REPLAY(replay_start, replay_count, 0, 0, 0, 0);

    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(0x004, 0xF, 1); // Restore the max/min behaviour of SWAP
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPNOP(0, 0, 0);
    }
#else
    // Diagnostic build — replay buffer disabled. The body is unrolled twice in source so
    // the LREG state after this call matches the replay-variant call (which executes the
    // body twice via load+replay or replay+replay). `init_replay` is unused here but kept
    // in the signature so callers don't need a separate ifdef around their state machine.
    (void)init_replay;

    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(0x104, 0xF, 1);
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPNOP(0, 0, 0);
    }

    if constexpr (STABLE_SORT)
    {
        // First execution
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPTRANSP;

        // Second execution (identical body)
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPTRANSP;
    }
    else
    {
        // First execution
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPTRANSP;

        // Second execution (identical body)
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG2, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPSWAP(0, p_sfpu::LREG2, p_sfpu::LREG3, p_sfpswap::ALL_ROWS_MAX);
        TTI_SFPTRANSP;
    }

    if (dir == static_cast<bool>(SortDir::ArgMin))
    {
        TTI_SFPCONFIG(0x004, 0xF, 1);
        TTI_SFPNOP(0, 0, 0);
        TTI_SFPNOP(0, 0, 0);
    }
#endif // TOPK_DISABLE_REPLAY
}

// Top-level local-sort orchestrator. For each (face, col) sub-region of the dest tile,
// walks phases [i_start_phase, i_end_phase]; phases 0..3 each emit 4 groups of
// [load16, swap-body, store16] sequences; phase 4+ falls back to inline step-N..5
// loops followed by phase-3 replay for steps 4..1.
//
// Replay-buffer slot allocation:
//   [0,  8)  = bitonic_topk_load16<is_fp32>(4, 8) body  (4 value loads + 4 index loads)
//   [8, 16)  = bitonic_topk_store16<is_fp32, alt_addr_mod=true>(4, 8) body
//   [16, 16+replay_count) = per-phase swap body (re-recorded at start of each new phase)
// Maximum slot used is 29 (stable phase 2). Quasar's replay buffer is 32 deep.
//
// `topk_replay_init` (file-scope) is read on entry: >=0 means slots [0,16) are still
// valid from a previous call (set by _init_topk to 0 or by phase 6 rebuild) and we
// can skip re-recording load16/store16. On exit it is written -1 to signal that no
// slots are reliably loaded for the next caller.
//
// -DTOPK_DISABLE_REPLAY emits a parallel implementation with NO replay buffer
// instructions — every load16/store16/swap-body is called inline. Used as a
// diagnostic to isolate replay-buffer ↔ SFPSWAP timing issues.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void _bitonic_topk_phases_steps(const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
#ifndef TOPK_DISABLE_REPLAY
    bool init_load  = (topk_replay_init >= 0) ? true : false;
    bool init_store = (topk_replay_init >= 0) ? true : false;
    bool init_phase;

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            bool dir = idir;
            for (int ph = i_start_phase; ph < (i_end_phase + 1); ph++)
            {
                init_phase = true; // each new phase re-records its body slot

                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                switch (ph)
                {
                    case 0:
                        for (int d = 0; d < 4; d++)
                        {
                            // Group of 16 datums: load16, swap, store16.
                            if (init_load)
                            {
                                load_replay_buf<0, 8, true>([] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
                                init_load = false;
                            }
                            else
                            {
                                TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            }

                            constexpr int replay_count = STABLE_SORT ? 6 : 4;
                            if (init_phase)
                            {
                                load_replay_buf<16, replay_count, true>([] { bitonic_topk_ph0_st1_to_1<STABLE_SORT>(); });
                                init_phase = false;
                            }
                            else
                            {
                                TTI_REPLAY(16, replay_count, 0, 0, 0, 0);
                            }

                            if (init_store)
                            {
                                load_replay_buf<8, 8, true>([] { bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8); });
                                init_store = false;
                            }
                            else
                            {
                                TTI_REPLAY(8, 8, 0, 0, 0, 0);
                            }
                        }
                        break;

                    case 1:
                        for (int d = 0; d < 4; d++)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);

                            constexpr int replay_count = STABLE_SORT ? 10 : 6;
                            if (init_phase)
                            {
                                load_replay_buf<16, replay_count, true>([] { bitonic_topk_ph1_st2_to_1<STABLE_SORT>(); });
                                init_phase = false;
                            }
                            else
                            {
                                TTI_REPLAY(16, replay_count, 0, 0, 0, 0);
                            }

                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                        }
                        break;

                    case 2:
                        for (int d = 0; d < 4; d++)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);

                            constexpr int replay_count = STABLE_SORT ? 14 : 9;
                            if (init_phase)
                            {
                                load_replay_buf<16, replay_count, true>([] { bitonic_topk_ph2_st3_to_1<STABLE_SORT>(); });
                                init_phase = false;
                            }
                            else
                            {
                                TTI_REPLAY(16, replay_count, 0, 0, 0, 0);
                            }

                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                        }
                        break;

                    case 3:
                        for (int d = 0; d < 4; d++)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            // Phase-4 helper records its own body in slots [16, 16+replay_count)
                            // when init_phase==true, then runs body twice.
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 16>(dir, init_phase);
                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                            dir = !dir;
                        }
                        break;

                    default:
                    {
                        // Phases 4..N: steps `num_steps`..5 are emitted inline (no replay);
                        // steps 4..1 fall back to the same phase-4 helper as case 3.
                        std::uint32_t num_steps               = ph + 1;
                        std::uint32_t start_step              = (i_start_phase == i_end_phase) ? i_start_step : num_steps;
                        std::uint32_t end_step                = (i_start_phase == i_end_phase) ? i_end_step : 4;
                        std::uint32_t sorted_seq_length       = 1 << num_steps;
                        std::uint32_t datums_compared         = 0;
                        std::uint32_t total_datums_to_compare = 64;

                        for (std::uint32_t ss = start_step; ss > end_step; ss--)
                        {
                            // Steps N..5 (inline)
                            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                            dir                      = idir;
                            std::uint32_t dist       = (ss == 5) ? 16 : 32;
                            std::uint32_t inner_d    = dist >> 3;
                            datums_compared          = 0;
                            std::uint32_t dst_offset = 0;

                            while (datums_compared < total_datums_to_compare)
                            {
                                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                                {
                                    bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist);
                                    bitonic_topk_step_N<STABLE_SORT>(dir);
                                    bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist);

                                    std::uint32_t dst_inc = 8;
                                    dst_offset += dst_inc;
                                    bool dst_cr = false;
                                    if (ii == (inner_d - 1))
                                    {
                                        dst_cr     = true;
                                        dst_inc    = 4 * dist;
                                        dst_offset = 2 * dist;
                                    }
                                    else if (dst_offset == 16)
                                    {
                                        dst_cr  = true;
                                        dst_inc = 32;
                                    }
                                    bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                    datums_compared += 16;
                                }
                                dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                            }
                        }

                        // Steps 4..1 (replay - same as case 3)
                        dir = idir;
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                        datums_compared = 0;

                        while (datums_compared < total_datums_to_compare)
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 16>(dir, init_phase);
                            TTI_REPLAY(8, 8, 0, 0, 0, 0);
                            datums_compared += 16;
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                        break;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }

    // Mark replay slots invalid for next caller.
    topk_replay_init = -1;
#else
    // Diagnostic build — replay buffer disabled. Each load16/store16/swap-body is
    // emitted inline every time. init_phase is still passed by reference to the
    // phase-4 helper (whose non-replay branch ignores it) so we keep it declared.
    bool init_phase;

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            bool dir = idir;
            for (int ph = i_start_phase; ph < (i_end_phase + 1); ph++)
            {
                init_phase = true;

                TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                switch (ph)
                {
                    case 0:
                        for (int d = 0; d < 4; d++)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph0_st1_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        }
                        break;

                    case 1:
                        for (int d = 0; d < 4; d++)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        }
                        break;

                    case 2:
                        for (int d = 0; d < 4; d++)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        }
                        break;

                    case 3:
                        for (int d = 0; d < 4; d++)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            // Phase-4 helper's TOPK_DISABLE_REPLAY branch handles the
                            // inlined SFPSWAP body. init_phase is unused but still passed.
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 16>(dir, init_phase);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                            dir = !dir;
                        }
                        break;

                    default:
                    {
                        std::uint32_t num_steps               = ph + 1;
                        std::uint32_t start_step              = (i_start_phase == i_end_phase) ? i_start_step : num_steps;
                        std::uint32_t end_step                = (i_start_phase == i_end_phase) ? i_end_step : 4;
                        std::uint32_t sorted_seq_length       = 1 << num_steps;
                        std::uint32_t datums_compared         = 0;
                        std::uint32_t total_datums_to_compare = 64;

                        for (std::uint32_t ss = start_step; ss > end_step; ss--)
                        {
                            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                            dir                      = idir;
                            std::uint32_t dist       = (ss == 5) ? 16 : 32;
                            std::uint32_t inner_d    = dist >> 3;
                            datums_compared          = 0;
                            std::uint32_t dst_offset = 0;

                            while (datums_compared < total_datums_to_compare)
                            {
                                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                                {
                                    bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist);
                                    bitonic_topk_step_N<STABLE_SORT>(dir);
                                    bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist);

                                    std::uint32_t dst_inc = 8;
                                    dst_offset += dst_inc;
                                    bool dst_cr = false;
                                    if (ii == (inner_d - 1))
                                    {
                                        dst_cr     = true;
                                        dst_inc    = 4 * dist;
                                        dst_offset = 2 * dist;
                                    }
                                    else if (dst_offset == 16)
                                    {
                                        dst_cr  = true;
                                        dst_inc = 32;
                                    }
                                    bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                    datums_compared += 16;
                                }
                                dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                            }
                        }

                        dir = idir;
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                        datums_compared = 0;
                        while (datums_compared < total_datums_to_compare)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 16>(dir, init_phase);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                            datums_compared += 16;
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                        break;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
    topk_replay_init = -1;
#endif // TOPK_DISABLE_REPLAY
}

// Thin forwarding wrapper: the test passes this as the `fn` argument to
// _llk_math_eltwise_unary_sfpu_params_, which calls it with the runtime int args.
//
// [Hypothesis A] Re-assert ENABLE_DEST_INDEX (LaneConfig bit [2]) at entry.
// The init in _init_topk() runs once before the main pipeline loop. We re-issue
// SFPCONFIG here as a defensive measure to ensure bit [2] is set immediately
// before SFPSWAP execution. The diagnostic confirmed bit [2] IS honored on
// Quasar — but the kernel's complex sort sequence may interact with bank flips,
// datacopy MOPs, or other state changes between iterations.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_phases_steps(const int idir, const int i_end_phase, const int i_start_phase, const int i_end_step, const int i_start_step)
{
    // [Hypothesis B/F] Re-assert ENABLE_DEST_INDEX via BH-reference SFPCONFIG path.
    ckernel::math::_sfpu_load_config32_(0xF, 0x0, 0x4);
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPNOP(0, 0, 0);
    _bitonic_topk_phases_steps<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT>(idir, i_end_phase, i_start_phase, i_end_step, i_start_step);
}

// Merge: one bitonic merge pass over already-sorted runs in Dest. For each (face, col)
// sub-region, repeatedly load 8 lanes (LREG0,1 values + LREG4,5 indices) via
// bitonic_topk_load8, run an SFPSWAP ALL_ROWS_MAX whose operand order is selected at
// compile time by `top_min` (false → ArgMax: LREG0, LREG1; true → ArgMin: LREG1, LREG0),
// then store back via bitonic_topk_store8. STABLE_SORT issues a duplicate SFPSWAP on the
// same operands — the value-swap is a no-op but with ENABLE_DEST_INDEX it re-evaluates
// the index conditional swap, breaking ties stably (1-cycle stall on the duplicate is
// intentional and matches the Blackhole reference).
//
// Merge does NOT touch the replay buffer; the inner body is short enough that recording
// would not pay off, and avoiding replay here means rebuild is free to own the slot range.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool top_min, bool STABLE_SORT = false>
inline void _bitonic_topk_merge(const int m_iter, const int k)
{
    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);

            int k_max                             = k > 32 ? 32 : k;
            std::uint32_t inner_d                 = k_max >> 2; // inner-loop comparisons to sort a length-K sequence
            std::uint32_t total_datums_to_compare = ((64 >> m_iter) < 2 * k_max) ? 2 * k_max : (64 >> m_iter);
            std::uint32_t dist                    = (k_max << m_iter) > 32 ? 32 : (k_max << m_iter);
            std::uint32_t ld_dist                 = (dist < 16) ? dist : 2 * dist; // accounts for face offsets within a tile
            std::uint32_t datums_compared         = 0;
            std::uint32_t dst_offset              = 0;
            std::uint32_t dst_cr                  = 0;

            while (datums_compared < total_datums_to_compare)
            {
                for (std::uint32_t ii = 0; ii < inner_d; ii++)
                {
                    bitonic_topk_load8<is_fp32_dest_acc_en>(dst_offset, ld_dist);
                    TTI_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    if constexpr (STABLE_SORT)
                    {
                        // Duplicate swap on identical operands: with ENABLE_DEST_INDEX
                        // the value-swap is a no-op but the index conditional swap
                        // re-evaluates for stable tie-break (1-cycle stall is intentional).
                        TTI_SFPSWAP(0, top_min ? p_sfpu::LREG1 : p_sfpu::LREG0, top_min ? p_sfpu::LREG0 : p_sfpu::LREG1, p_sfpswap::ALL_ROWS_MAX);
                    }
                    bitonic_topk_store8<is_fp32_dest_acc_en>(dst_offset, ld_dist);

                    datums_compared += 8;
                    if (ii == (inner_d - 1))
                    {
                        dst_cr += 2 * dist;
                        dst_offset = dst_cr;
                    }
                    else
                    {
                        dst_offset += 4;
                    }
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
}

// Thin forwarding wrapper: matches the call signature of
// `_llk_math_eltwise_unary_sfpu_params_`. `idir` is named to mirror the BH metal-side
// wrapper's third template parameter (the sort-direction bit forwarded as `top_min`).
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool idir = false, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_merge(const int m_iter, const int k)
{
    // [Hypothesis B/F] Re-assert ENABLE_DEST_INDEX via BH-reference SFPCONFIG path.
    ckernel::math::_sfpu_load_config32_(0xF, 0x0, 0x4);
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPNOP(0, 0, 0);
    _bitonic_topk_merge<APPROXIMATION_MODE, is_fp32_dest_acc_en, idir, STABLE_SORT>(m_iter, k);
}

// Rebuild: re-runs phase (logK-1) on a merged tile pair to reextract sorted runs.
// A switch on `(logk - 1)` selects one of cases 0/1/2/3/default; each emits a different
// load/swap-sequence/store pattern. For non-stable sort, most cases are wrapped in a
// replay buffer with INCRWC counter advances embedded in the lambda body.
//
// Replay-slot allocation (when -DTOPK_DISABLE_REPLAY is not set):
//   case 1 m_iter>=2 (non-stable) : slots [0, 22) — load8 + ph1_st2_to_1 + store8 + 8 INCRWC
//   case 1 m_iter<2  (non-stable) : slots [0, 26) — load16 + ph1_st2_to_1 + store16 + 4 INCRWC
//   case 2          (non-stable)  : slots [0, 29) — load16 + ph2_st3_to_1 + store16 + 4 INCRWC
//   case 3          (non-stable)  : slots [0,  8) load16, [8, 13) ph3 helper, [13, 25) store16 + 4 INCRWC
//   default Part 2  (non-stable)  : slots [0,  8) load16, [8, 13) ph3 helper, [17, 25) store16
//
// `topk_replay_init` is read on entry: equal to `m_iter + 1` means a previous rebuild call
// for this same `m_iter` has already loaded the slots, so the replay-buffer-load path is
// skipped (init_rebuild=false). On exit it is written `m_iter + 1` to mark the slots
// valid for the next caller. Note this collides with phase-5's slot range [0, 16) when
// phases_steps runs after rebuild — phase 5's `init_load = (topk_replay_init >= 0)` will
// re-record on the next local_sort, so the contract is consistent.
//
// -DTOPK_DISABLE_REPLAY emits the body inline every time (no replay buffer); used as a
// diagnostic to isolate replay-buffer ↔ SFPSWAP timing issues.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void _bitonic_topk_rebuild(const bool idir, const int m_iter, const int k, const int logk, const int skip_second)
{
#ifndef TOPK_DISABLE_REPLAY
    bool init_rebuild = (topk_replay_init != m_iter + 1) ? true : false;

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            std::uint32_t total_datums_shift = (skip_second & 0x1);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
            std::uint32_t rebuild_m               = m_iter + 1;
            std::uint32_t total_datums_to_compare = ((64 >> rebuild_m) < 2 * k) ? 2 * k : (64 >> rebuild_m);
            total_datums_to_compare               = total_datums_to_compare >> total_datums_shift;
            std::uint32_t dist                    = (k << rebuild_m) > 32 ? 32 : (k << rebuild_m);
            std::uint32_t ld_offset               = (dist >> 4) * 32 + (dist & 0xF);
            std::uint32_t ld_dist;
            int ph                        = logk - 1;
            bool dir                      = idir;
            std::uint32_t datums_compared = 0;

            switch (ph)
            {
                case 0:
                    break;

                case 1:
                    if (m_iter >= 2)
                    {
                        while (datums_compared < total_datums_to_compare)
                        {
                            if constexpr (STABLE_SORT)
                            {
                                bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                                bitonic_topk_inc_x8_dest(64, false);
                            }
                            else
                            {
                                if (init_rebuild)
                                {
                                    load_replay_buf<0, 22, true>(
                                        [ld_offset]
                                        {
                                            bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                            bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                                            bitonic_topk_inc_x8_dest(64, false);
                                        });
                                    init_rebuild = false;
                                }
                                else
                                {
                                    TTI_REPLAY(0, 22, 0, 0, 0, 0);
                                }
                            }
                            datums_compared += 16;
                        }
                        break;
                    }
                    else
                    {
                        ld_dist = (ld_offset < 16) ? 4 * ld_offset : 2 * ld_offset;
                        while (datums_compared < total_datums_to_compare)
                        {
                            if constexpr (STABLE_SORT)
                            {
                                bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                                bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
                                TTI_INCRWC(0, 0, 0, 8);
                                TTI_INCRWC(0, 0, 0, 8);
                                TTI_INCRWC(0, 0, 0, 8);
                                TTI_INCRWC(0, 0, 0, 8);
                            }
                            else
                            {
                                if (init_rebuild)
                                {
                                    load_replay_buf<0, 26, true>(
                                        [ld_offset, ld_dist]
                                        {
                                            bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
                                            TTI_INCRWC(0, 0, 0, 8);
                                            TTI_INCRWC(0, 0, 0, 8);
                                            TTI_INCRWC(0, 0, 0, 8);
                                            TTI_INCRWC(0, 0, 0, 8);
                                        });
                                    init_rebuild = false;
                                }
                                else
                                {
                                    TTI_REPLAY(0, 26, 0, 0, 0, 0);
                                }
                            }
                            datums_compared += 16;
                        }
                        break;
                    }

                case 2:
                    while (datums_compared < total_datums_to_compare)
                    {
                        if constexpr (STABLE_SORT)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                            bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                        }
                        else
                        {
                            if (init_rebuild)
                            {
                                load_replay_buf<0, 29, true>(
                                    [ld_offset]
                                    {
                                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                                        bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                                        TTI_INCRWC(0, 0, 0, 8);
                                        TTI_INCRWC(0, 0, 0, 8);
                                        TTI_INCRWC(0, 0, 0, 8);
                                        TTI_INCRWC(0, 0, 0, 8);
                                    });
                                init_rebuild = false;
                            }
                            else
                            {
                                TTI_REPLAY(0, 29, 0, 0, 0, 0);
                            }
                        }
                        datums_compared += 16;
                    }
                    break;

                case 3:
                    while (datums_compared < total_datums_to_compare)
                    {
                        if constexpr (STABLE_SORT)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir, init_rebuild);
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                        }
                        else
                        {
                            if (init_rebuild)
                            {
                                load_replay_buf<0, 8, true>([] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
                                bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir, init_rebuild);
                                load_replay_buf<13, 12, true>(
                                    []
                                    {
                                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                                        TTI_INCRWC(0, 0, 0, 8);
                                        TTI_INCRWC(0, 0, 0, 8);
                                        TTI_INCRWC(0, 0, 0, 8);
                                        TTI_INCRWC(0, 0, 0, 8);
                                    });
                            }
                            else
                            {
                                TTI_REPLAY(0, 8, 0, 0, 0, 0);
                                bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir, init_rebuild);
                                TTI_REPLAY(13, 12, 0, 0, 0, 0);
                            }
                        }
                        datums_compared += 16;
                        dir = !dir;
                    }
                    break;

                default:
                {
                    // ph >= 4: two-part sort. Part 1 emits steps N..5 inline (no replay);
                    // Part 2 emits steps 4..1 with replay, reusing slots [0, 8) for load16,
                    // [8, 13) for the ph3 helper body, and [17, 25) for store16.
                    std::uint32_t num_steps               = ph + 1;
                    std::uint32_t start_step              = num_steps;
                    std::uint32_t end_step                = 4;
                    std::uint32_t sorted_seq_length       = 1 << num_steps;
                    std::uint32_t total_datums_to_compare = 64; // shadows outer; intentional, matches BH

                    for (std::uint32_t ss = start_step; ss > end_step; ss--)
                    {
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                        dir                      = idir;
                        datums_compared          = 0;
                        std::uint32_t dist       = (ss == 5) ? 16 : 32;
                        std::uint32_t inner_d    = dist >> 3;
                        std::uint32_t dst_offset = 0;

                        while (datums_compared < total_datums_to_compare)
                        {
                            for (std::uint32_t ii = 0; ii < inner_d; ii++)
                            {
                                bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist);
                                bitonic_topk_step_N<STABLE_SORT>(dir);
                                bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist);

                                std::uint32_t dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (ii == (inner_d - 1))
                                {
                                    dst_cr     = true;
                                    dst_inc    = 4 * dist;
                                    dst_offset = 2 * dist;
                                }
                                else if (dst_offset == 16)
                                {
                                    dst_cr  = true;
                                    dst_inc = 32;
                                }
                                bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                    }

                    // Steps 4..1
                    dir             = idir;
                    datums_compared = 0;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);

                    while (datums_compared < total_datums_to_compare)
                    {
                        if (init_rebuild)
                        {
                            load_replay_buf<0, 8, true>([] { bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8); });
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir, init_rebuild);
                            load_replay_buf<17, 8, true>([] { bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8); });
                        }
                        else
                        {
                            TTI_REPLAY(0, 8, 0, 0, 0, 0);
                            bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir, init_rebuild);
                            TTI_REPLAY(17, 8, 0, 0, 0, 0);
                        }
                        datums_compared += 16;
                        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                    }
                    break;
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
    topk_replay_init = m_iter + 1;
#else
    // Diagnostic build — replay buffer disabled. Each load/swap-body/store/INCRWC is
    // emitted inline every time. `init_rebuild` is unused but kept declared so the
    // phase-3 helper's reference parameter still has a binding target.
    bool init_rebuild = false;

    std::uint32_t dst_addr_offset = 0;
    for (int face = 0; face < 2; face++)
    {
        for (int col = 0; col < 2; col++)
        {
            std::uint32_t total_datums_shift = (skip_second & 0x1);
            TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
            std::uint32_t rebuild_m               = m_iter + 1;
            std::uint32_t total_datums_to_compare = ((64 >> rebuild_m) < 2 * k) ? 2 * k : (64 >> rebuild_m);
            total_datums_to_compare               = total_datums_to_compare >> total_datums_shift;
            std::uint32_t dist                    = (k << rebuild_m) > 32 ? 32 : (k << rebuild_m);
            std::uint32_t ld_offset               = (dist >> 4) * 32 + (dist & 0xF);
            std::uint32_t ld_dist;
            int ph                        = logk - 1;
            bool dir                      = idir;
            std::uint32_t datums_compared = 0;

            switch (ph)
            {
                case 0:
                    break;

                case 1:
                    if (m_iter >= 2)
                    {
                        while (datums_compared < total_datums_to_compare)
                        {
                            bitonic_topk_load8<is_fp32_dest_acc_en>(0, ld_offset);
                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                            bitonic_topk_store8<is_fp32_dest_acc_en>(0, ld_offset);
                            bitonic_topk_inc_x8_dest(64, false);
                            datums_compared += 16;
                        }
                        break;
                    }
                    else
                    {
                        ld_dist = (ld_offset < 16) ? 4 * ld_offset : 2 * ld_offset;
                        while (datums_compared < total_datums_to_compare)
                        {
                            bitonic_topk_load16<is_fp32_dest_acc_en>(ld_offset, ld_dist);
                            bitonic_topk_ph1_st2_to_1<STABLE_SORT>();
                            bitonic_topk_store16<is_fp32_dest_acc_en, true>(ld_offset, ld_dist);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            TTI_INCRWC(0, 0, 0, 8);
                            datums_compared += 16;
                        }
                        break;
                    }

                case 2:
                    while (datums_compared < total_datums_to_compare)
                    {
                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, ld_offset);
                        bitonic_topk_ph2_st3_to_1<STABLE_SORT>();
                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, ld_offset);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        datums_compared += 16;
                    }
                    break;

                case 3:
                    while (datums_compared < total_datums_to_compare)
                    {
                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                        // Phase-3 helper's TOPK_DISABLE_REPLAY branch handles the inlined body.
                        bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir, init_rebuild);
                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        TTI_INCRWC(0, 0, 0, 8);
                        datums_compared += 16;
                        dir = !dir;
                    }
                    break;

                default:
                {
                    std::uint32_t num_steps               = ph + 1;
                    std::uint32_t start_step              = num_steps;
                    std::uint32_t end_step                = 4;
                    std::uint32_t sorted_seq_length       = 1 << num_steps;
                    std::uint32_t total_datums_to_compare = 64;

                    for (std::uint32_t ss = start_step; ss > end_step; ss--)
                    {
                        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);
                        dir                      = idir;
                        datums_compared          = 0;
                        std::uint32_t dist       = (ss == 5) ? 16 : 32;
                        std::uint32_t inner_d    = dist >> 3;
                        std::uint32_t dst_offset = 0;

                        while (datums_compared < total_datums_to_compare)
                        {
                            for (std::uint32_t ii = 0; ii < inner_d; ii++)
                            {
                                bitonic_topk_load16<is_fp32_dest_acc_en>(4, 2 * dist);
                                bitonic_topk_step_N<STABLE_SORT>(dir);
                                bitonic_topk_store16<is_fp32_dest_acc_en, false>(4, 2 * dist);

                                std::uint32_t dst_inc = 8;
                                dst_offset += dst_inc;
                                bool dst_cr = false;
                                if (ii == (inner_d - 1))
                                {
                                    dst_cr     = true;
                                    dst_inc    = 4 * dist;
                                    dst_offset = 2 * dist;
                                }
                                else if (dst_offset == 16)
                                {
                                    dst_cr  = true;
                                    dst_inc = 32;
                                }
                                bitonic_topk_inc_x8_dest(dst_inc, dst_cr);
                                datums_compared += 16;
                            }
                            dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                        }
                    }

                    dir             = idir;
                    datums_compared = 0;
                    TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, p_setrwc::SET_D);

                    while (datums_compared < total_datums_to_compare)
                    {
                        bitonic_topk_load16<is_fp32_dest_acc_en>(4, 8);
                        bitonic_topk_ph3_st4_to_1<STABLE_SORT, 8>(dir, init_rebuild);
                        bitonic_topk_store16<is_fp32_dest_acc_en, true>(4, 8);
                        datums_compared += 16;
                        dir = (datums_compared == sorted_seq_length) ? !dir : dir;
                    }
                    break;
                }
            }
            dst_addr_offset += 2;
            set_dst_write_addr(dst_addr_offset);
        }
        dst_addr_offset = 16;
        set_dst_write_addr(dst_addr_offset);
    }
    topk_replay_init = m_iter + 1;
#endif // TOPK_DISABLE_REPLAY
}

// Thin forwarding wrapper. `idir` is `int` because the variadic
// `_llk_math_eltwise_unary_sfpu_params_` forwards arguments unchanged from the test, where
// it is an `int` (TOPK_SORT_DIRECTION value). The implicit int->bool conversion happens at
// the `_bitonic_topk_rebuild(idir, …)` call site.
template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void calculate_bitonic_topk_rebuild(const int idir, const int m_iter, const int k, const int logk, const int skip_second)
{
    // [Hypothesis B/F] Re-assert ENABLE_DEST_INDEX via BH-reference SFPCONFIG path.
    ckernel::math::_sfpu_load_config32_(0xF, 0x0, 0x4);
    TTI_SFPNOP(0, 0, 0);
    TTI_SFPNOP(0, 0, 0);
    _bitonic_topk_rebuild<APPROXIMATION_MODE, is_fp32_dest_acc_en, STABLE_SORT>(idir, m_iter, k, logk, skip_second);
}

} // namespace sfpu
} // namespace ckernel
