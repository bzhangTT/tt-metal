// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "llk_pack.h"
#include "llk_pack_common.h"

inline void llk_pack_set_fp32_dest_acc(bool enable) { _llk_pack_set_fp32_dest_acc_(enable); }

inline void llk_packer_wait_math_semaphore() { _llk_packer_wait_for_math_done_(); }

template <uint WaitRes = p_stall::NONE>
inline void llk_packer_set_math_semaphore() {
    _llk_packer_set_math_semaphore_<WaitRes>();
}

inline void llk_pack_debug_dump(std::uint8_t* data, std::uint32_t byte_size) { _llk_pack_debug_dump_(data, byte_size); }

inline void llk_pack_debug_dump_seek(std::uint8_t offset) { _llk_pack_debug_dump_seek_(offset); }
