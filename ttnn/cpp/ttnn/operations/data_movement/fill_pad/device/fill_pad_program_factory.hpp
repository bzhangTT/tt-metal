// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "fill_pad_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim::detail {

const std::map<ttnn::DataType, uint32_t> data_type_to_size = {
    {ttnn::DataType::BFLOAT16, 2},
    {ttnn::DataType::FLOAT32, 4},
    {ttnn::DataType::UINT16, 2},
    {ttnn::DataType::UINT32, 4},
    {ttnn::DataType::INT32, 4},
    {ttnn::DataType::UINT8, 1},
};

}  // namespace ttnn::prim::detail

namespace ttnn::prim {

struct FillPadProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
