// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// AI-generated — run_id: 2026-04-29_binary_max_min_quasar_dualpath
//
// Three-operand SFPU test for binary max/min on Quasar.
//
// Two execution paths, selected at runtime by `unpack_to_dest`:
//   * unpack_to_dest=true   — UNPACK → Dest directly. Used for 32-bit Dest formats
//                             (Float32, Int32 with dest_acc=Yes).
//   * unpack_to_dest=false  — UNPACK → SrcA → FPU datacopy → Dest. Required for
//                             non-32-bit and MX block formats (Float16_b, MxFp8R,
//                             MxFp8P), which the unpacker cannot route directly to
//                             Dest because the block-exponent conversion happens
//                             on the FPU datacopy side.
//
// Buffer layout (params.TILE_CNT == 2):
//   buffer_A[0] = in0 tile   → Dest[DST_INDEX + 0]
//   buffer_A[1] = in1 tile   → Dest[DST_INDEX + 1]
//   SFPU writes              → Dest[DST_INDEX + 2]
//   buffer_Res[0] = result   ← Dest[DST_INDEX + 2] (1 tile packed)

#include <cstdint>

#include "ckernel.h"
#include "llk_defs.h"
#include "llk_memory_checks.h"
#include "sfpu_stub.h"

#ifdef LLK_TRISC_UNPACK

#include "llk_math_common.h"
#include "llk_unpack_common.h"
#include "llk_unpack_unary_operand.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    const std::uint32_t buf_desc_id = 0;
    const std::uint32_t num_tiles   = params.TILE_CNT;

    if (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        _llk_math_upk_to_dest_hw_configure_<IMPLIED_MATH_FORMAT, is_fp32_dest_acc_en, false /*is_int_fpu_en*/>();
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::UNPACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = L1_ADDRESS(params.buffer_A[0]);
    bd_val.f.format            = static_cast<std::uint8_t>(formats.unpack_A_src);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t td_val;
    td_val.buf_desc        = bd_val;
    td_val.buf_desc_id     = buf_desc_id;
    td_val.reg_data_format = static_cast<std::uint8_t>(formats.unpack_A_dst);
    _configure_buf_desc_table_(td_val.buf_desc_id, td_val.buf_desc);

    // For 32-bit Dest with the FPU-datacopy path, Mov2D is implemented via ELWADD,
    // so both UNP_A and UNP_B must be configured with the same descriptor.
    if (is_fp32_dest_acc_en && !unpack_to_dest)
    {
        _llk_unpack_configure_binary_<p_unpacr::UNP_A, p_unpacr::UNP_B>(td_val, td_val);
    }
    else
    {
        _llk_unpack_configure_unary_<UNPACKER_ENGINE_SEL>(td_val);
    }

    _llk_unpack_unary_operand_init_<UNPACKER_ENGINE_SEL, false /*transpose*/, is_fp32_dest_acc_en>(buf_desc_id, num_tiles);
    _llk_unpack_unary_operand_<UNPACKER_ENGINE_SEL>(0);

    if (unpack_to_dest)
    {
        _llk_unpack_dest_dvalid_section_done_<dest_sync>();
    }
}

#endif // LLK_TRISC_UNPACK

#ifdef LLK_TRISC_MATH

const bool is_int_fpu_en = false;

#include "cfg_defines.h"
#include "cmath_common.h"
#include "experimental/ckernel_sfpu_binary_max_min.h"
#include "llk_math_common.h"
#include "llk_math_eltwise_unary_datacopy.h"
#include "llk_math_eltwise_unary_sfpu_common.h"
#include "params.h"

using namespace ckernel;
using namespace ckernel::math;
using namespace ckernel::sfpu;

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif

    if (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::FPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
        set_up_dest_dvalid_per_thread<dest_dvalid_client::SFPU>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    DataFormat src_format = static_cast<DataFormat>(formats.math);
    _llk_math_srcAB_hw_configure_<false, is_fp32_dest_acc_en, is_int_fpu_en>(src_format, src_format);

    // FPU-datacopy path: move all TILE_CNT input tiles from SrcA into Dest at
    // DST_INDEX + i. Required for non-32-bit and MX formats; skipped when the
    // unpacker has already placed data directly in Dest.
    if (!unpack_to_dest)
    {
        const std::uint32_t num_rows = params.num_faces * params.TEST_FACE_R_DIM;
        _llk_math_eltwise_unary_datacopy_init_<DATA_COPY_TYPE, is_fp32_dest_acc_en>(num_rows, 1);
        for (std::uint32_t i = 0; i < params.TILE_CNT; ++i)
        {
            _llk_math_eltwise_unary_datacopy_(num_rows, params.DST_INDEX + i);
        }
        _llk_math_set_dvalid_<p_cleardvalid::FPU, dest_sync>();
    }

    _llk_math_eltwise_unary_sfpu_init_();

    // SFPU computes max/min(Dest[DST_INDEX+0], Dest[DST_INDEX+1]) → Dest[DST_INDEX+2].
    // The 3-operand offsets are relative to dst_tile_index (set as the Dest base
    // by _llk_math_eltwise_unary_sfpu_start_).
    //
    // The int32-variant is picked at runtime from the data format. The float
    // path uses sfpmem::DEFAULT for SFPLOAD/SFPSTORE — the unpacker /
    // _llk_math_srcAB_hw_configure_ already programs SrcB format and SFPU_Fp32,
    // which DEFAULT resolves through — so no compile-time format dispatch needed.
    const bool is_int32 = (static_cast<DataFormat>(formats.math) == DataFormat::Int32);

    if (is_int32)
    {
        binary_max_min_int32_init<IS_MAX_OP, false /*IS_UNSIGNED*/>();
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_binary_max_min_int32<IS_MAX_OP, false /*IS_UNSIGNED*/, 8 /*ITERATIONS*/>,
            params.DST_INDEX,
            /* dst_index_in0 */ 0U,
            /* dst_index_in1 */ 1U,
            /* dst_index_out */ 2U);
    }
    else
    {
        binary_max_min_init<IS_MAX_OP>();
        _llk_math_eltwise_unary_sfpu_params_(
            ckernel::sfpu::calculate_binary_max_min<IS_MAX_OP, 8 /*ITERATIONS*/>,
            params.DST_INDEX,
            /* dst_index_in0 */ 0U,
            /* dst_index_in1 */ 1U,
            /* dst_index_out */ 2U);
    }

    _llk_math_set_dvalid_<p_cleardvalid::SFPU, dest_sync>();

    wait_sfpu_idle();
    wait_fpu_idle();
    wait_mop_idle();
}

#endif // LLK_TRISC_MATH

#ifdef LLK_TRISC_PACK

#include "cfg_defines.h"
#include "llk_pack.h"
#include "llk_pack_common.h"
#include "params.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
#if defined(RUNTIME_FORMATS) && !defined(SPEED_OF_LIGHT)
    const FormatConfig& formats = params.formats;
#endif
    std::uint32_t const buf_desc_id = 8;
    // Only one result tile is packed (the SFPU output); the 2 input tiles in
    // Dest are not part of the result.
    const std::uint32_t num_tiles_per_pack = 1;

    if (unpack_to_dest)
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::UNPACK, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }
    else
    {
        set_up_dest_dvalid_per_thread<dest_dvalid_client::PACK>({dest_dvalid_client::FPU, dest_dvalid_client::SFPU, dest_dvalid_client::PACK});
    }

    buffer_descriptor_u bd_val = {0};
    bd_val.f.l1_addr_16B       = params.buffer_Res[0] / 16;
    bd_val.f.format            = static_cast<std::uint8_t>(formats.pack_dst);
    bd_val.f.x_dim             = params.TEST_FACE_C_DIM;
    bd_val.f.y_dim             = params.TEST_FACE_R_DIM;
    bd_val.f.z_dim             = params.num_faces;

    tdma_descriptor_t tdma_desc;
    tdma_desc.buf_desc        = bd_val;
    tdma_desc.buf_desc_id     = buf_desc_id;
    tdma_desc.reg_data_format = static_cast<std::uint8_t>(formats.pack_src);
    _configure_buf_desc_table_(tdma_desc.buf_desc_id, tdma_desc.buf_desc);

    _llk_pack_hw_configure_<p_pacr::PACK0>(tdma_desc);
    _llk_pack_init_(buf_desc_id, num_tiles_per_pack);
    _llk_pack_(params.DST_INDEX + 2, 0);
    _llk_pack_dest_dvalid_section_done_<dest_sync, is_fp32_dest_acc_en>();
}
#endif // LLK_TRISC_PACK
