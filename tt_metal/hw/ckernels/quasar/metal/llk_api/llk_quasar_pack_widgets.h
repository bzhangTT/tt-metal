// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_defs.h"
#include "llk_pack_common.h"

/**
 * @brief Clears the data valid for destination register after Packer 0 is done packing
 * and zeroes out the dest bank(s) used by packer 0
 *
 * @tparam DST: Destination register buffering mode, values = [DstSync::SyncHalf, DstSync::SyncFull]
 * @tparam EN_32BIT_DEST: flag to show if math destination register is set to 32-bit mode
 **/
template <DstSync DST, bool EN_32BIT_DEST>
inline void llk_pack_dest_dvalid_section_done() {
    _llk_pack_dest_dvalid_section_done_<DST, EN_32BIT_DEST>();
}
