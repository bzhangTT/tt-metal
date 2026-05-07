// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;
using namespace ttnn::operations::data_movement;

ProgramDescriptor FillPadProgramFactory::create_descriptor(
    const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const Tensor& input_tensor = tensor_args.input;
    const float fill_value = operation_attributes.fill_value;
    IDevice* device = input_tensor.device();

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    Buffer* tens_buffer = input_tensor.buffer();
    TT_ASSERT(tens_buffer != nullptr, "Input buffer should be allocated on device!");

    const uint32_t input_element_size_bytes = detail::data_type_to_size.at(input_tensor.dtype());
    const uint32_t cb_page_size = (input_element_size_bytes * tt::constants::FACE_HEIGHT) + sizeof(uint16_t);
    const uint32_t height = input_tensor.logical_shape()[-2];
    const uint32_t width = input_tensor.logical_shape()[-1];

    const uint32_t problem_size = input_tensor.logical_shape()[-3];

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, problem_size);
    const uint32_t g1_numcores = core_group_1.num_cores();

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;

    const bool src_is_dram = tens_buffer->buffer_type() == BufferType::DRAM;

    // pack bf16 vals
    uint32_t packed_fill_value = 0;
    if (input_tensor.dtype() == DataType::BFLOAT16) {
        packed_fill_value = pack_two_bfloat16_into_uint32({bfloat16(fill_value), bfloat16(fill_value)});
    } else if (input_tensor.dtype() == DataType::UINT16) {
        packed_fill_value = pack_two_uint16_into_uint32({fill_value, fill_value});
    } else if (input_tensor.dtype() == DataType::FLOAT32) {
        packed_fill_value = std::bit_cast<uint32_t>(fill_value);
    } else {
        packed_fill_value = static_cast<std::uint32_t>(fill_value);
    }

    const uint32_t padded_height = tt::div_up(height, tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    const uint32_t padded_width = tt::div_up(width, tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    const uint32_t tiles_per_2d_tensor =
        padded_height / tt::constants::TILE_HEIGHT * padded_width / tt::constants::TILE_HEIGHT;
    const uint32_t tiles_per_tile_row = padded_width / tt::constants::TILE_HEIGHT;

    const bool sharded = input_tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_page_size * 2,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = cb_page_size,
        }}},
    });

    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<std::uint32_t>(src0_cb_index),
        static_cast<std::uint32_t>(src_is_dram),
        static_cast<std::uint32_t>(packed_fill_value),
        static_cast<std::uint32_t>(input_element_size_bytes),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(padded_height),
        static_cast<std::uint32_t>(padded_width),
        static_cast<std::uint32_t>(tiles_per_2d_tensor),
        static_cast<std::uint32_t>(tiles_per_tile_row),
        static_cast<std::uint32_t>(tt::constants::TILE_HEIGHT),
        static_cast<std::uint32_t>(tt::constants::FACE_HEIGHT)};

    KernelDescriptor::Defines writer_defines;
    if (sharded) {
        shard_builder::extend_sharding_compile_time_args(input_tensor, writer_compile_time_args);
        writer_defines.emplace_back("SHARDED", "1");
    } else {
        TensorAccessorArgs(*tens_buffer).append_to(writer_compile_time_args);
    }

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = std::move(writer_defines);
    // writer only for in-place operation
    writer_desc.config = WriterConfigDescriptor{};

    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t local_num_2d_tensors =
            i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Buffer* at slot 0 registers a binding so the framework patches the in-place
        // tensor's address on cache hits without re-running create_descriptor().
        KernelDescriptor::RTArgList args;
        args.push_back(tens_buffer);
        args.push_back(cb_page_size);
        args.push_back(tile_offset);
        args.push_back(local_num_2d_tensors);
        if (sharded) {
            std::vector<uint32_t> shard_args;
            shard_builder::extend_sharding_run_time_args(input_tensor, shard_args);
            args.append(shard_args);
        }
        writer_desc.emplace_runtime_args(core, args);

        tile_offset += local_num_2d_tensors * tiles_per_2d_tensor;
    }

    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
