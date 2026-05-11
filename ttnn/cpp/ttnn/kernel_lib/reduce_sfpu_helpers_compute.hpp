// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

#include "api/compute/reduce.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp"

/**
 * @file reduce_sfpu_helpers_compute.hpp
 * @brief SFPU reduce helper for tile MAX/MIN when the FPU GMPOOL path cannot be used.
 *
 * `reduce_sfpu` is the SFPU sibling of `compute_kernel_lib::reduce`:
 * - `reduce()` uses FPU GMPOOL (`reduce_tile`), which does not give correct Int32 MAX/MIN on device.
 * - `reduce_sfpu()` uses SFPU `sfpu_reduce` for within-tile reduction and
 *   `binary_{max,min}_int32_tile` for cross-tile folds along the reduce axis.
 *
 * Supports REDUCE_ROW (reduces W -> Ht outputs) and REDUCE_COL (reduces H -> Wt outputs).
 * MIN + REDUCE_ROW is rejected at compile time; the host launches reduce_sfpu_w_neg.cpp directly
 * for that case, mirroring how the FPU path uses reduce_w_neg.cpp.
 *
 * Library responsibilities: DST acquire/commit/wait/release, CB wait/pop/reserve/push,
 * pack_tile, and the packer reduce mask (sfpu_reduce does not configure it).
 *
 * IMPORTANT: Do not call `compute_kernel_hw_startup()`; this helper calls `init_sfpu`
 * and `copy_tile_to_dst_init_short` itself.
 *
 * IMPORTANT: The scaler CB must contain one tile before entry. `sfpu_reduce` does not consume it;
 * the helper waits and pops it so dataflow matches the FPU kernel.
 *
 * Basic usage:
 *   compute_kernel_lib::reduce_sfpu<ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW, DataFormat::Int32>(
 *       cb_in, cb_scaler, cb_out,
 *       compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC),
 *       post_mul_scaler_bits);
 */

namespace compute_kernel_lib {

/**
 * @brief SFPU reduce for Int32 MAX/MIN along one tile axis.
 *
 * @tparam pool_type   PoolType::MAX or PoolType::MIN (MIN + REDUCE_ROW is rejected at compile time;
 *                     host launches reduce_sfpu_w_neg.cpp directly for that case).
 * @tparam reduce_dim  ReduceDim::REDUCE_ROW (W) or ReduceDim::REDUCE_COL (H).
 * @tparam format      DataFormat::Int32 (only supported format).
 *
 * @param input_cb_id        Tiles to reduce (streaming order matches `reduce()` for same dim).
 * @param scaler_cb_id       Scaler tile CB (waited/popped; not passed into sfpu_reduce).
 * @param output_cb_id       Reduced output tiles.
 * @param input_block_shape  Tile grid (Ht, Wt, NC).
 * @param post_mul_scaler_bits  Packed fp32 user scalar; used only when `REDUCE_POST_MUL` is set.
 */
template <ckernel::PoolType pool_type, ckernel::ReduceDim reduce_dim, DataFormat format>
ALWI void reduce_sfpu(
    uint32_t input_cb_id,
    uint32_t scaler_cb_id,
    uint32_t output_cb_id,
    ReduceInputBlockShape input_block_shape,
    uint32_t post_mul_scaler_bits);

}  // namespace compute_kernel_lib

#include "ttnn/cpp/ttnn/kernel_lib/reduce_sfpu_helpers_compute.inl"
