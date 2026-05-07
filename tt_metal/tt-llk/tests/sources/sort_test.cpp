// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// LLK regression for ttnn.sort's bitonic-merge algorithm. Mirrors
//   ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_single_row_single_core.cpp
//   ttnn/cpp/ttnn/operations/data_movement/sort/device/kernels/compute/sort_common.hpp
// at the LLK level — local_sort + merge ONLY (no rebuild), which is the path
// that fails in tt-metal#37571 (topk_test.cpp covers the rebuild path).
//
// buffer_A per tile-row: [0..Wt-1] = value tiles, [Wt..2*Wt-1] = uint16 index tiles.

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include "ckernel.h"
#include "llk_defs.h"
#include "params.h"

// Globals required by the test framework.
std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

// Stages.
enum class Stage : int
{
    Values  = 0,
    Indices = 1
};

constexpr int NUM_STAGES = 2;

// DEST register layout (matches metal sort kernel):
//   DEST[0] = value tile low,  DEST[1] = value tile high
//   DEST[2] = index tile low,  DEST[3] = index tile high
constexpr int input_dest_start = 0;
constexpr int input_dest_end   = 1;
constexpr int index_dest_start = 2;
constexpr int index_dest_end   = 3;

// Same hard-coded constants as the metal sort kernel.
constexpr int SORT_K         = 64;
constexpr int SORT_END_PHASE = 5; // log2(64)-1

// Sort direction (this LLK reproducer always sorts ascending).
constexpr bool ASCENDING = true;

// Helper: count log2 of a power-of-2 integer.
static constexpr int ilog2_ce(int n)
{
    int r = 0;
    while ((1 << r) < n)
    {
        ++r;
    }
    return r;
}

// ============================================================================
// UNPACK TRISC
// ============================================================================

#ifdef LLK_TRISC_UNPACK
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const int FULL_CT_DIM = params.FULL_CT_DIM;
    const int Wt          = FULL_CT_DIM / NUM_STAGES;
    const int NUM_ROWS    = params.FULL_RT_DIM;
    const int STAGES      = ilog2_ce(Wt);

    const std::uint32_t unpack_src_data_types[NUM_STAGES] = {formats.unpack_A_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t unpack_dst_data_types[NUM_STAGES] = {formats.unpack_A_dst, ckernel::to_underlying(DataFormat::UInt16)};

    bool first_hw_config = true;

    auto setup_for_stage = [&](Stage stage, bool transpose_faces)
    {
        const int stage_index                 = static_cast<int>(stage);
        const std::uint32_t unpack_src_format = unpack_src_data_types[stage_index];
        const std::uint32_t unpack_dst_format = unpack_dst_data_types[stage_index];

        if (first_hw_config)
        {
            _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
                unpack_src_format, unpack_src_format, unpack_dst_format, unpack_dst_format, FACE_R_DIM, FACE_R_DIM, 4, 4);
            first_hw_config = false;
        }
        else
        {
            _llk_unpack_reconfig_data_format_srca_impl_<is_fp32_dest_acc_en, p_dim_stride_target::IGNORE, false>(
                unpack_src_format, unpack_dst_format, 16 * 16 * 4);
        }

        _llk_unpack_A_init_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            transpose_faces ? 1 : 0, transpose_faces ? 1 : 0, FACE_R_DIM, 4, unpack_src_format, unpack_dst_format);
    };

    auto unpack_tile = [&](Stage stage, int l1_tile_index)
    {
        const int stage_index                 = static_cast<int>(stage);
        const std::uint32_t unpack_src_format = unpack_src_data_types[stage_index];
        const std::uint32_t unpack_dst_format = unpack_dst_data_types[stage_index];

        _llk_unpack_A_<BroadcastType::NONE, false, EltwiseBinaryReuseDestType::NONE, unpack_to_dest>(
            L1_ADDRESS(params.buffer_A[l1_tile_index]), unpack_src_format, unpack_dst_format);
    };

    for (int row = 0; row < NUM_ROWS; ++row)
    {
        const int row_offset = row * FULL_CT_DIM;

        // ----- 1. Bitonic-sequence formation -----
        for (int wt = 0; wt < Wt; wt += 2)
        {
            // Values pair (wt, wt+1) - face-level transpose.
            setup_for_stage(Stage::Values, /*transpose_faces=*/true);
            unpack_tile(Stage::Values, row_offset + wt);
            unpack_tile(Stage::Values, row_offset + wt + 1);

            // Indices pair (Wt+wt, Wt+wt+1) - face-level transpose.
            setup_for_stage(Stage::Indices, /*transpose_faces=*/true);
            unpack_tile(Stage::Indices, row_offset + Wt + wt);
            unpack_tile(Stage::Indices, row_offset + Wt + wt + 1);
        }

        // ----- 2. Bitonic merge stages -----
        for (int stage = 2; stage <= STAGES; ++stage)
        {
            for (int sub = stage; sub > 0; --sub)
            {
                const int sub_dist = 1 << (sub - 1);
                for (int i = 0; i < Wt; ++i)
                {
                    const int j = i ^ sub_dist;
                    if (j > i)
                    {
                        // Values - no face transpose (data already column-wise after local_sort).
                        setup_for_stage(Stage::Values, /*transpose_faces=*/false);
                        unpack_tile(Stage::Values, row_offset + i);
                        unpack_tile(Stage::Values, row_offset + j);

                        setup_for_stage(Stage::Indices, /*transpose_faces=*/false);
                        unpack_tile(Stage::Indices, row_offset + Wt + i);
                        unpack_tile(Stage::Indices, row_offset + Wt + j);
                    }
                }
            }
        }

        // ----- 3. Final result-copy pass: unpack each tile (no transpose) -----
        for (int t = 0; t < FULL_CT_DIM; ++t)
        {
            // First Wt tiles are values, second Wt tiles are indices.
            const Stage st = (t < Wt) ? Stage::Values : Stage::Indices;
            setup_for_stage(st, /*transpose_faces=*/false);
            unpack_tile(st, row_offset + t);
        }
    }
}
#endif // LLK_TRISC_UNPACK

// ============================================================================
// MATH TRISC
// ============================================================================

#ifdef LLK_TRISC_MATH
#include "ckernel_sfpu.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"

using namespace ckernel;

#define DST_SYNC_MODE  dest_sync
#define DST_ACCUM_MODE is_fp32_dest_acc_en
#include "llk_sfpu/llk_math_eltwise_unary_sfpu_topk.h"
#undef DST_SYNC_MODE
#undef DST_ACCUM_MODE

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const int FULL_CT_DIM = params.FULL_CT_DIM;
    const int Wt          = FULL_CT_DIM / NUM_STAGES;
    const int NUM_ROWS    = params.FULL_RT_DIM;
    const int STAGES      = ilog2_ce(Wt);

    constexpr bool APPROX             = false;
    constexpr std::uint32_t dst_index = 0;
    constexpr int vector_mode         = (int)VectorMode::RC_custom;
    constexpr int start_phase         = 0;
    constexpr int end_step            = 0;
    constexpr int start_step          = 0;

    const std::uint32_t math_data_types[NUM_STAGES] = {formats.math, ckernel::to_underlying(DataFormat::UInt16)};

    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_eltwise_unary_sfpu_init_<SfpuType::topk_local_sort>();
    ckernel::sfpu::_init_topk();

    bool first_hw_config = true;

    auto reconfig_math = [&](Stage stage)
    {
        const int stage_index           = static_cast<int>(stage);
        const std::uint32_t math_format = math_data_types[stage_index];

        if (first_hw_config)
        {
            _llk_math_hw_configure_<is_fp32_dest_acc_en>(math_format, math_format);
            first_hw_config = false;
        }
        else
        {
            _llk_math_reconfig_data_format_srca_<is_fp32_dest_acc_en, false>(math_format);
        }

#ifdef ARCH_BLACKHOLE
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false, false>(
            /*num_rows_per_matrix=*/4, /*math_format=*/math_format);
#else
        _llk_math_eltwise_unary_datacopy_init_<DataCopyType::A2D, is_fp32_dest_acc_en, BroadcastType::NONE, false>(
            /*num_rows_per_matrix=*/4, /*math_format=*/math_format);
#endif
    };

    auto datacopy_one = [&](Stage stage, int dst_tile)
    {
        const std::uint32_t math_format = math_data_types[static_cast<int>(stage)];
        _llk_math_eltwise_unary_datacopy_<DataCopyType::A2D, DstSync::SyncHalf, is_fp32_dest_acc_en, BroadcastType::NONE, unpack_to_dest>(
            dst_tile, math_format, math_format);
    };

    for (int row = 0; row < NUM_ROWS; ++row)
    {
        // ----- 1. Bitonic-sequence formation -----
        bool ascending_local = ASCENDING;
        for (int wt = 0; wt < Wt; wt += 2)
        {
            _llk_math_wait_for_dest_available_<dest_sync>();

            reconfig_math(Stage::Values);
            datacopy_one(Stage::Values, input_dest_start);
            datacopy_one(Stage::Values, input_dest_end);

            reconfig_math(Stage::Indices);
            datacopy_one(Stage::Indices, index_dest_start);
            datacopy_one(Stage::Indices, index_dest_end);

            _llk_math_eltwise_unary_sfpu_params_(
                ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROX, is_fp32_dest_acc_en, /*STABLE=*/false>,
                dst_index,
                vector_mode,
                /*idir=*/(int)ascending_local,
                SORT_END_PHASE,
                start_phase,
                end_step,
                start_step);

            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            ascending_local = !ascending_local;
        }

        // ----- 2. Bitonic merge stages -----
        for (int stage = 2; stage <= STAGES; ++stage)
        {
            const int m_iter = stage - 1;
            for (int sub = stage; sub > 0; --sub)
            {
                const int sub_dist = 1 << (sub - 1);
                for (int i = 0; i < Wt; ++i)
                {
                    const int j = i ^ sub_dist;
                    if (j > i)
                    {
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir             = (ascending_block == ASCENDING);

                        _llk_math_wait_for_dest_available_<dest_sync>();

                        reconfig_math(Stage::Values);
                        datacopy_one(Stage::Values, input_dest_start);
                        datacopy_one(Stage::Values, input_dest_end);

                        reconfig_math(Stage::Indices);
                        datacopy_one(Stage::Indices, index_dest_start);
                        datacopy_one(Stage::Indices, index_dest_end);

                        if (sub == 1)
                        {
                            _llk_math_eltwise_unary_sfpu_params_(
                                ckernel::sfpu::calculate_bitonic_topk_phases_steps<APPROX, is_fp32_dest_acc_en, /*STABLE=*/false>,
                                dst_index,
                                vector_mode,
                                /*idir=*/(int)dir,
                                SORT_END_PHASE,
                                start_phase,
                                end_step,
                                start_step);
                        }
                        else
                        {
                            // top_min = false (compile-time), matching metal kernel (default idir=false).
                            _llk_math_eltwise_unary_sfpu_params_(
                                ckernel::sfpu::calculate_bitonic_topk_merge<APPROX, is_fp32_dest_acc_en, /*top_min=*/false, /*STABLE=*/false>,
                                dst_index,
                                vector_mode,
                                m_iter,
                                SORT_K);
                        }

                        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }

        // ----- 3. Final result-copy pass: datacopy each tile to DEST[0] -----
        for (int t = 0; t < FULL_CT_DIM; ++t)
        {
            const Stage st = (t < Wt) ? Stage::Values : Stage::Indices;
            _llk_math_wait_for_dest_available_<dest_sync>();
            reconfig_math(st);
            datacopy_one(st, /*dst_tile=*/0);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}
#endif // LLK_TRISC_MATH

// ============================================================================
// PACK TRISC
// ============================================================================

#ifdef LLK_TRISC_PACK
#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const int FULL_CT_DIM = params.FULL_CT_DIM;
    const int Wt          = FULL_CT_DIM / NUM_STAGES;
    const int NUM_ROWS    = params.FULL_RT_DIM;
    const int STAGES      = ilog2_ce(Wt);

#ifdef ARCH_BLACKHOLE
    _llk_pack_dest_init_<dest_sync, is_fp32_dest_acc_en>();
#else
    _llk_pack_dest_init_<dest_sync, false, false>();
#endif

    const std::uint32_t pack_src_data_types[NUM_STAGES] = {formats.pack_src, ckernel::to_underlying(DataFormat::UInt16)};
    const std::uint32_t pack_dst_data_types[NUM_STAGES] = {formats.pack_dst, ckernel::to_underlying(DataFormat::UInt16)};

    bool first_hw_config = true;

    auto reconfig_pack = [&](Stage stage)
    {
        const int stage_index               = static_cast<int>(stage);
        const std::uint32_t pack_src_format = pack_src_data_types[stage_index];
        const std::uint32_t pack_dst_format = pack_dst_data_types[stage_index];

        if (first_hw_config)
        {
#ifdef ARCH_BLACKHOLE
            _llk_pack_hw_configure_<is_fp32_dest_acc_en, false, false>(pack_src_format, pack_dst_format, 16 * 16 * 4);
#else
            _llk_pack_hw_configure_<is_fp32_dest_acc_en, false>(pack_src_format, pack_dst_format, 16 * 16 * 4);
#endif
            first_hw_config = false;
        }
        else
        {
#ifdef ARCH_BLACKHOLE
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, false>(pack_src_format, pack_dst_format, 16 * 16 * 4, FACE_R_DIM, TILE_C_DIM, 4, false, 1);
#else
            _llk_pack_reconfig_data_format_<is_fp32_dest_acc_en, false>(pack_src_format, pack_dst_format, 16 * 16 * 4, FACE_R_DIM, 4, false, false);
#endif
        }

        _llk_pack_init_wrapper_<false, false>(pack_dst_format);
    };

    for (int row = 0; row < NUM_ROWS; ++row)
    {
        const int row_offset = row * FULL_CT_DIM;

        // ----- 1. Bitonic-sequence formation -----
        for (int wt = 0; wt < Wt; wt += 2)
        {
            _llk_packer_wait_for_math_done_();

            reconfig_pack(Stage::Values);
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(input_dest_start, L1_ADDRESS(params.buffer_A[row_offset + wt]));
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(input_dest_end, L1_ADDRESS(params.buffer_A[row_offset + wt + 1]));

            reconfig_pack(Stage::Indices);
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(index_dest_start, L1_ADDRESS(params.buffer_A[row_offset + Wt + wt]));
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(index_dest_end, L1_ADDRESS(params.buffer_A[row_offset + Wt + wt + 1]));

            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }

        // ----- 2. Bitonic merge stages -----
        for (int stage = 2; stage <= STAGES; ++stage)
        {
            for (int sub = stage; sub > 0; --sub)
            {
                const int sub_dist = 1 << (sub - 1);
                for (int i = 0; i < Wt; ++i)
                {
                    const int j = i ^ sub_dist;
                    if (j > i)
                    {
                        const bool ascending_block = ((i >> stage) & 1) == 0;
                        const bool dir             = (ascending_block == ASCENDING);

                        // Direction handling: for sub != 1 (merge with top_min=false),
                        // DEST[0] holds the SMALLER value and DEST[1] holds the LARGER.
                        // - dir == true (ascending): low tile id (i) gets smaller -> DEST[0],
                        //   high tile id (j) gets larger -> DEST[1].  But the metal code shows
                        //   the OPPOSITE swap: when dir, swap so tile_input_low = input_dest_end (1).
                        //   Reading the metal kernel carefully:
                        //     "topk_merge puts smallest values in DEST[0] and largest in DEST[1]"
                        //     "We swap their indices when using descending order"
                        //   But the code is:  if (dir) { swap; }   and `dir` is computed as
                        //   `dir = ascending_block == ascending` where ascending=true means
                        //   the global request is ascending.  So `dir==true` means ASCENDING.
                        //   The comment says "swap when descending" but the code swaps when `dir==true`.
                        //   This is consistent if we read it as: top_min=false means DEST[0]=largest
                        //   and DEST[1]=smallest (top-K of largest values goes to DEST[0]).
                        //   In ascending order, smaller goes left -> we want DEST[1] in left, hence swap.
                        //   We faithfully mirror the metal code's behaviour.
                        int v_low_dst, v_high_dst, x_low_dst, x_high_dst;
                        if (sub != 1 && dir)
                        {
                            v_low_dst  = input_dest_end;
                            v_high_dst = input_dest_start;
                            x_low_dst  = index_dest_end;
                            x_high_dst = index_dest_start;
                        }
                        else
                        {
                            v_low_dst  = input_dest_start;
                            v_high_dst = input_dest_end;
                            x_low_dst  = index_dest_start;
                            x_high_dst = index_dest_end;
                        }

                        _llk_packer_wait_for_math_done_();

                        reconfig_pack(Stage::Values);
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(v_low_dst, L1_ADDRESS(params.buffer_A[row_offset + i]));
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(v_high_dst, L1_ADDRESS(params.buffer_A[row_offset + j]));

                        reconfig_pack(Stage::Indices);
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(x_low_dst, L1_ADDRESS(params.buffer_A[row_offset + Wt + i]));
                        _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(x_high_dst, L1_ADDRESS(params.buffer_A[row_offset + Wt + j]));

                        _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                    }
                }
            }
        }

        // ----- 3. Final result-copy pass: pack each tile from DEST[0] to buffer_Res -----
        for (int t = 0; t < FULL_CT_DIM; ++t)
        {
            const Stage st = (t < Wt) ? Stage::Values : Stage::Indices;
            _llk_packer_wait_for_math_done_();
            reconfig_pack(st);
            _llk_pack_<dest_sync, is_fp32_dest_acc_en, false>(0, L1_ADDRESS(params.buffer_Res[row_offset + t]));
            _llk_pack_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
    }
}
#endif // LLK_TRISC_PACK
