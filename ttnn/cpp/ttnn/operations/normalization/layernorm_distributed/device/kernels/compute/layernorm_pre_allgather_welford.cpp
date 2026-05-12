// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

/*
 * This kernel computes layernorm statistics.
 * For layernorm it computes E(x**2) and E(x) and returns them as a two tile wide output tensor containing E(x**2) and
 * E(x) in the left most columns per tile. For rmsnorm it computes E(x**2) and returns it as a one tile wide output
 * tensor containing E(x**2) in the left most column per tile.
 */

#include <cstdint>
#include <cstring>

#define REDUCE_OP PoolType::SUM
#define REDUCE_DIM ReduceDim::REDUCE_ROW

#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/layernorm.h"
#include "api/compute/transpose_wh.h"
#include "api/compute/welford.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "ttnn/operations/normalization/kernel_util/compute/memory.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose_wh_dest.h"

template <typename To, typename From>
inline To _bit_cast_(const From& from) noexcept {
    static_assert(sizeof(To) == sizeof(From), "Types must have same size");
    static_assert(std::is_trivially_copyable_v<From>, "From must be trivially copyable");
    static_assert(std::is_trivially_copyable_v<To>, "To must be trivially copyable");
    To to;
    std::memcpy(&to, &from, sizeof(To));
    return to;
}
void kernel_main() {
    uint32_t NCHt = get_arg_val<uint32_t>(0);
    namespace kutil = norm::kernel_util;
    constexpr uint32_t Wt = get_compile_time_arg_val(0);
    constexpr uint32_t W = get_compile_time_arg_val(1);

    constexpr uint32_t cb_inp = tt::CBIndex::c_0;
    constexpr uint32_t cb_out = tt::CBIndex::c_14;
    constexpr uint32_t cb_x2 = tt::CBIndex::c_1;           // x**2
    constexpr uint32_t cb_reciprocals = tt::CBIndex::c_2;  // recip table

    compute_kernel_hw_startup(cb_inp, cb_inp, cb_x2);
    // Get pointer to the reciprocal LUT
    using recip_lut_t = std::array<uint32_t, W>;
    auto p_reciprocals = kutil::compute::memory::get_pointer_to_cb_data<recip_lut_t>(cb_reciprocals, 0);
    // The number of valid columns in the last tile in width dimension.
    // Because the Welford's llk is given transposed data, skip some rows when
    // we want to skip some columns from getting processed by layer_norm.
    constexpr uint32_t last_tile_rows = (W % 32) == 0 ? 32 : W % 32;

    for (uint32_t ncht = 0; ncht < NCHt; ncht++) {
        constexpr uint32_t dst0 = 0;
        constexpr uint32_t dst1 = 1;
        constexpr uint32_t dst2 = 2;

        reconfig_data_format(cb_inp, cb_inp);
        pack_reconfig_data_format(cb_x2);

        tile_regs_acquire();
        uint32_t start_N = 0;
        transpose_wh_init(cb_inp, cb_x2);
        welford_init();

        // When the input CB carries Float32 with fp32_dest_acc_en=true, the program factory
        // sets unpack_to_dest_mode=UnpackToDestFp32 for cb_inp so transpose_wh_tile takes the
        // UnpackToDest fp32 path (preserves full 23-mantissa-bit fp32 into DEST rather than
        // downcasting to TF32 in SrcA). That path calls llk_math_transpose_dest, which writes
        // to SFPU replay buffer slot 0; the same slot welford_init programmed with the
        // welford recurrence. Re-establish welford state after each transpose_wh_tile so
        // welford_update replays welford ops instead of stale transpose-dest ops. LREG4/5 (the
        // running mean / M2 accumulator) survive transpose_dest because it only uses FPU MOVs.
        for (uint32_t wt = 0; wt < (Wt - 1); wt++) {
            cb_wait_front(cb_inp, 1);  // cumulative wait
            transpose_wh_init_short(cb_inp);
            transpose_wh_tile(cb_inp, 0, dst0);
            welford_reinit(cb_inp);
            MATH((llk_math_welfords_sfpu_init()));
            // welford_tile<dst0, dst1, dst2, true, 0>((wt) * 32, W, 0, {});
            welford_update<W>(dst0, start_N, *p_reciprocals);
            start_N += 32;
            cb_pop_front(cb_inp, 1);
        }
        cb_wait_front(cb_inp, 1);  // cumulative wait
        transpose_wh_init_short(cb_inp);
        transpose_wh_tile(cb_inp, 0, dst0);
        welford_reinit(cb_inp);
        MATH((llk_math_welfords_sfpu_init()));
        welford_update_rows<W>(dst0, start_N, 0, last_tile_rows, *p_reciprocals);
        cb_pop_front(cb_inp, 1);
        welford_finalize_to_row<W>(dst1, W - 1, *p_reciprocals);
        // tt-llk/issues/549
        // BUG: using transpose_dest here causes a bug. where the kernel hangs
        //  transpose_wh_dest_init_short();
        //  transpose_wh_dest(dst1);
        //  transpose_wh_dest(dst2);
        cb_reserve_back(cb_x2, 2);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst1, cb_x2);
        pack_tile(dst2, cb_x2);
        cb_push_back(cb_x2, 2);
        tile_regs_release();
        reconfig_data_format(cb_x2, cb_x2);
        pack_reconfig_data_format(cb_out);
        transpose_wh_init_short(cb_x2);
        tile_regs_acquire();
        cb_wait_front(cb_x2, 2);  // cumulative wait
        transpose_wh_tile(cb_x2, 0, dst0);
        transpose_wh_tile(cb_x2, 1, dst1);
        cb_pop_front(cb_x2, 2);

        tile_regs_commit();
        tile_regs_wait();
        pack_tile(dst0, cb_out);
        pack_tile(dst1, cb_out);
        cb_push_back(cb_out, 2);
        tile_regs_release();
    }
}
