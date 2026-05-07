// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/untilize_with_unpadding/device/untilize_with_unpadding_device_operation_types.hpp"

namespace ttnn::prim {

struct UntilizeWithUnpaddingSingleCoreProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output);
};

}  // namespace ttnn::prim
