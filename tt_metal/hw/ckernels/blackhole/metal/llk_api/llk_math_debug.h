// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ckernel.h"
#include "llk_math_common.h"

inline std::uint32_t llk_math_get_compute_special_value_flags() { return _llk_math_get_compute_special_value_flags_(); }

inline std::uint32_t llk_math_get_compute_special_value_flags_fpu(std::uint32_t special_value_flags_reg) {
    constexpr std::uint32_t special_value_flags_fpu_mask = 0xf;
    constexpr std::uint32_t special_value_flags_fpu_shift = 4;
    return (special_value_flags_reg & special_value_flags_fpu_mask) >> special_value_flags_fpu_shift;
}

inline std::uint32_t llk_math_get_compute_special_value_flags_sfpu(std::uint32_t special_value_flags_reg) {
    constexpr std::uint32_t special_value_flags_sfpu_mask = 0xf;
    constexpr std::uint32_t special_value_flags_sfpu_shift = 0;
    return (special_value_flags_reg & special_value_flags_sfpu_mask) >> special_value_flags_sfpu_shift;
}

inline void llk_math_clear_compute_special_value_flags() { _llk_math_clear_compute_special_value_flags_(); }

inline void llk_math_store_compute_special_value_flags_to_l1(std::uint32_t l1_addr) {
    volatile tt_l1_ptr std::uint32_t* l1_addr_ptr = reinterpret_cast<volatile tt_l1_ptr std::uint32_t*>(l1_addr);
    l1_addr_ptr[0] = _llk_math_get_compute_special_value_flags_();
}
