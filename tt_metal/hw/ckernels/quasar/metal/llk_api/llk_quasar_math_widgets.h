// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "llk_math_common.h"

/**
 * @brief Sets the dest dvalid for FPU/SFPU
 *
 * @tparam SET_DEST_DVALID: which client to set data valid for, values = p_cleardvalid::FPU/SFPU
 **/
template <std::uint8_t SET_DEST_DVALID>
inline void llk_math_set_dvalid() {
    _llk_math_set_dvalid_<SET_DEST_DVALID>();
}
