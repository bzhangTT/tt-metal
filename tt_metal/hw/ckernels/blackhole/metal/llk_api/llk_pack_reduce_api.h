// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_pack_common_api.h"
#include "llk_param_structs.h"

/*************************************************************************
 * LLK PACK REDUCE
 *************************************************************************/

template <bool untilize = false, ReduceDim dim>
inline void llk_pack_reduce_mask_config() {
    _llk_pack_reduce_mask_config_<untilize, dim>();
}

inline void llk_pack_reduce_mask_clear() { _llk_pack_reduce_mask_clear_(); }

// FIXME-WH-UPLIFT
template <ReduceDim dim, bool is_fp32_dest_acc_en, bool at_kernel_start = false, bool revert = false>
inline void llk_pack_reduce_config_v2(uint32_t icb_out) {
    const bool untilize = false;
    if constexpr (at_kernel_start) {
        const std::uint32_t output_id = get_output_id(icb_out);
        const std::uint32_t face_r_dim = get_output_face_r_dim(output_id);
        const std::uint32_t tile_c_dim = get_output_tile_c_dim(output_id);
        const std::uint32_t num_faces = get_output_num_faces(output_id);
        const bool partial_face = get_output_partial_face(output_id);
        const std::uint32_t tile_size = get_local_cb_interface(output_id).fifo_page_size;
        const llk_relu_config_u relu_config = {
            .f = {
                .ApplyRelu = (std::uint32_t)ReluType::NO_RELU,
                .Threshold = 0,
            }};

        _llk_pack_hw_configure_<is_fp32_dest_acc_en, untilize>(
            pack_src_format[output_id],
            pack_dst_format[output_id],
            tile_size,
            face_r_dim,
            tile_c_dim,
            num_faces,
            partial_face,
            relu_config.val);
    }

    if constexpr (revert) {
        _llk_pack_reduce_mask_clear_();
    } else {
        _llk_pack_reduce_mask_config_<untilize, dim>();
    }
}
