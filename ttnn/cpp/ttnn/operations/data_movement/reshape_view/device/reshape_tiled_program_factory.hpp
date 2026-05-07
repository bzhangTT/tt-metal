// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/device/reshape_device_operation_types.hpp"

namespace ttnn::prim {

struct ReshapeViewTiledProgramFactory {
    // Op-owned tensor that must survive across cache hits.  The framework reflects
    // through Resources so its buffer participates in BufferBinding patching and
    // tensor lifetime alongside tensor_args / tensor_return_value.
    //
    // Wrapped in std::optional because Tensor has no default constructor and the
    // framework default-initialises resource_t before prepare_resources populates it.
    struct Resources {
        std::optional<Tensor> mapping_tensor;
    };

    static Resources prepare_resources(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value);

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ReshapeViewParams& operation_attributes,
        const ReshapeViewInputs& tensor_args,
        Tensor& tensor_return_value,
        Resources& resources);
};

}  // namespace ttnn::prim
