// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstdint>

#include "ckernel_addrmod.h"
#include "ckernel_sfpu_topk.h"
#include "cmath_common.h"
#include "llk_defs.h"
#include "llk_math_eltwise_unary_sfpu_common.h"

using namespace ckernel::math;

// VectorMode enum — Quasar's llk_defs.h does not define VectorMode (Blackhole's does).
// We define it locally here so the test's `(int)VectorMode::RC_custom` resolves identically
// (same numeric encoding as Blackhole's llk_defs.h:14–22).
enum VectorMode
{
    None      = 0,
    R         = 1,
    C         = 2,
    RC        = 4,
    RC_custom = 6,
    Invalid   = 0xFF,
};

/**
 * @brief SfpuType-templated init overload for topk.
 *
 * The shared topk test calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>()`
 * directly, then `ckernel::sfpu::_init_topk()`. The base un-templated init in
 * llk_math_eltwise_unary_sfpu_common.h takes no SfpuType parameter, so this is an overload.
 *
 * On top of the standard SFPU init (config reg, addrmod 7, RWC reset), this configures
 * ADDR_MOD_6 with dest.incr=32 — required because `bitonic_topk_store16<is_fp32, alt_addr_mod=true>`
 * uses ADDR_MOD_6 on the FINAL index store to auto-advance Dest by 32 rows. ADDR_MOD_6 is
 * otherwise undefined in default SFPU init.
 *
 * The caller must follow up with `ckernel::sfpu::_init_topk()` to set LaneConfig bit [2]
 * (ENABLE_DEST_INDEX) — that is a separate step left to the caller so a single
 * init_<topk_local_sort>() does not silently configure features the caller may not want.
 */
template <SfpuType sfpu_op>
inline void _llk_math_eltwise_unary_sfpu_init_()
{
    _init_sfpu_config_reg_();
    _eltwise_unary_sfpu_configure_addrmod_();
    _reset_counters_<p_setrwc::SET_ABD_F>();

    if constexpr (sfpu_op == SfpuType::topk_local_sort || sfpu_op == SfpuType::topk_merge || sfpu_op == SfpuType::topk_rebuild)
    {
        // ADDR_MOD_6: dest.incr=32 — used by alt_addr_mod=true store16 path.
        addr_mod_t {
            .srca = {.incr = 0},
            .srcb = {.incr = 0},
            .dest = {.incr = 32},
        }
            .set(ADDR_MOD_6, csr_read<CSR::TRISC_ID>());
    }
}

/**
 * @brief Overload of `_llk_math_eltwise_unary_sfpu_params_` accepting `int vector_mode`.
 *
 * For VectorMode::RC (4), iterates over NUM_FACES, calling sfpu_func once per face with a
 * dest-face-addr increment between iterations (matches the base un-templated version's
 * behavior).
 *
 * For VectorMode::RC_custom (6) and any other value, calls sfpu_func once. Topk manages
 * its own (face, col) iteration internally via `set_dst_write_addr`, so the LLK API must
 * NOT iterate over faces externally.
 *
 * This overload is more specialized than the variadic version in
 * `llk_math_eltwise_unary_sfpu_common.h` (it has an explicit `int` parameter where the
 * other has `ARGS&&...`), so calls of the form
 * `_llk_math_eltwise_unary_sfpu_params_(fn, dst_index, vector_mode, args...)` resolve here.
 */
template <class F, class... ARGS>
inline void _llk_math_eltwise_unary_sfpu_params_(F&& sfpu_func, std::uint32_t dst_tile_index, int vector_mode, ARGS&&... args)
{
    _llk_math_eltwise_unary_sfpu_start_(dst_tile_index);

    if (vector_mode == static_cast<int>(VectorMode::RC))
    {
        for (std::uint32_t face = 0; face < NUM_FACES; face++)
        {
            sfpu_func(static_cast<ARGS&&>(args)...);
            _llk_math_eltwise_unary_sfpu_inc_dst_face_addr_();
        }
    }
    else
    {
        // RC_custom or any other value: call once. Topk's `_bitonic_topk_*` walk all
        // (face, col) sub-regions internally via `set_dst_write_addr`.
        sfpu_func(static_cast<ARGS&&>(args)...);
    }

    _llk_math_eltwise_unary_sfpu_done_();
}

namespace ckernel
{

template <bool APPROXIMATE>
inline void llk_math_eltwise_unary_sfpu_topk_init()
{
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    ckernel::sfpu::_init_topk();
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_local_sort(
    std::uint32_t dst_index,
    int idir,
    int i_end_phase,
    int i_start_phase,
    int i_end_step,
    int i_start_step,
    int vector_mode = static_cast<int>(VectorMode::RC_custom))
{
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROXIMATE, is_fp32_dest_acc_en, STABLE_SORT>,
        dst_index,
        vector_mode,
        idir,
        i_end_phase,
        i_start_phase,
        i_end_step,
        i_start_step);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool top_min = false, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_merge(std::uint32_t dst_index, int m_iter, int k, int vector_mode = static_cast<int>(VectorMode::RC_custom))
{
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_bitonic_topk_merge<APPROXIMATE, is_fp32_dest_acc_en, top_min, STABLE_SORT>, dst_index, vector_mode, m_iter, k);
}

template <bool APPROXIMATE, bool is_fp32_dest_acc_en, bool STABLE_SORT = false>
inline void llk_math_eltwise_unary_sfpu_topk_rebuild(
    std::uint32_t dst_index, int idir, int m_iter, int k, int logk, int skip_second, int vector_mode = static_cast<int>(VectorMode::RC_custom))
{
    _llk_math_eltwise_unary_sfpu_params_(
        ckernel::sfpu::calculate_bitonic_topk_rebuild<APPROXIMATE, is_fp32_dest_acc_en, STABLE_SORT>,
        dst_index,
        vector_mode,
        idir,
        m_iter,
        k,
        logk,
        skip_second);
}

} // namespace ckernel
