// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "scatter_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct ScatterReduceBfloat16ProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ScatterParams& operation_attributes, const ScatterInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
