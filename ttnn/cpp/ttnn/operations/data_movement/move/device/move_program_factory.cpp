// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_program_factory.hpp"

#include <cmath>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor MoveProgramFactory::create_descriptor(
    const MoveOperationAttributes& operation_attributes,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::constants;

    const Tensor& input = tensor_args.input_tensor;
    Tensor& output = tensor_return_value;
    const bool backwards = operation_attributes.backwards;

    const bool tilized = output.layout() == Layout::TILE;
    const bool sharded = input.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(input.dtype());
    uint32_t input_unit_size =
        tilized ? tt::tile_size(input_cb_data_format) : input.padded_shape()[-1] * input.element_size();
    const uint32_t full_input_row = input_unit_size;
    if (sharded && !tilized) {
        input_unit_size = input.memory_config().shard_spec()->shape[1] * input.element_size();
    }
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_unit_size =
        tilized ? tt::tile_size(output_cb_data_format) : output.padded_shape()[-1] * output.element_size();
    const uint32_t full_output_row = output_unit_size;
    if (sharded && !tilized) {
        output_unit_size = output.memory_config().shard_spec()->shape[1] * output.element_size();
    }

    const bool convert_dtype = input_cb_data_format != output_cb_data_format;

    const uint32_t num_units =
        tilized ? output.physical_volume() / TILE_HW : output.physical_volume() / output.padded_shape()[-1];

    IDevice* device = output.device();

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_units_per_core_group_1, num_units_per_core_group_2] =
            split_work_to_cores(compute_with_storage_grid_size, num_units);

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    ProgramDescriptor desc;

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t num_input_units = 2;
    const uint32_t input_alignment = src_buffer->alignment();
    const uint32_t aligned_input_unit_size = tt::align(input_unit_size, input_alignment);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_units * aligned_input_unit_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = aligned_input_unit_size,
        }}},
    });

    uint32_t output_cb_index = src0_cb_index;  // same as input cb
    if (convert_dtype) {
        output_cb_index = tt::CBIndex::c_16;
        const uint32_t num_output_units = 2;
        const uint32_t output_alignment = dst_buffer->alignment();
        const uint32_t aligned_output_unit_size = tt::align(output_unit_size, output_alignment);
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_output_units * aligned_output_unit_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(output_cb_index),
                .data_format = output_cb_data_format,
                .page_size = aligned_output_unit_size,
            }}},
        });
    }

    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> writer_compile_time_args;
    if (tilized) {
        writer_compile_time_args = {output_cb_index};
    } else {
        reader_compile_time_args = {src0_cb_index, input_unit_size};
        writer_compile_time_args = {output_cb_index, output_unit_size};
    }
    std::vector<std::pair<std::string, std::string>> kernel_defines;
    if (sharded) {
        kernel_defines.emplace_back("SHARDED", "1");
        shard_builder::extend_sharding_compile_time_args(input, writer_compile_time_args);
        shard_builder::extend_sharding_compile_time_args(input, reader_compile_time_args);
    } else {
        TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);
    }
    if (backwards) {
        kernel_defines.emplace_back("BACKWARDS", "1");
    }

    const std::string reader_rm_path =
        sharded ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_stick_start_id.cpp"
                : "ttnn/cpp/ttnn/kernel/dataflow/reader_unary_stick_layout_interleaved_start_id.cpp";
    const std::string reader_kernel_path =
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/reader_unary_start_id.cpp"
                : reader_rm_path;

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = kernel_defines;
    reader_desc.config = ReaderConfigDescriptor{};

    const std::string writer_rm_path =
        sharded ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_stick_start_id.cpp"
                : "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp";
    const std::string writer_kernel_path =
        tilized ? "ttnn/cpp/ttnn/operations/data_movement/copy/device/kernels/writer_unary_start_id.cpp"
                : writer_rm_path;

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_path;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = kernel_defines;
    writer_desc.config = WriterConfigDescriptor{};

    // Optional dtype-conversion compute kernels (one per core group).  Pushed last so
    // reader/writer keep kernel indices 0/1 — matches the original CreateKernel order.
    KernelDescriptor compute_g1_desc;
    KernelDescriptor compute_g2_desc;
    bool has_compute_g2 = false;
    if (convert_dtype) {
        compute_g1_desc.kernel_source = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp";
        compute_g1_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_g1_desc.core_ranges = core_group_1;
        compute_g1_desc.compile_time_args = {num_units_per_core_group_1};
        compute_g1_desc.config = ComputeConfigDescriptor{};

        if (!core_group_2.ranges().empty()) {
            compute_g2_desc.kernel_source = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp";
            compute_g2_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_g2_desc.core_ranges = core_group_2;
            compute_g2_desc.compile_time_args = {num_units_per_core_group_2};
            compute_g2_desc.config = ComputeConfigDescriptor{};
            has_compute_g2 = true;
        }
    }

    uint32_t start_id = 0;
    if (backwards) {
        start_id = num_units - 1;
    }

    const uint32_t g1_numcores = core_group_1.num_cores();
    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores.at(i);
        const uint32_t num_units_per_core = i < g1_numcores ? num_units_per_core_group_1 : num_units_per_core_group_2;

        // Buffer* entries register BufferBindings at arg slot 0; the framework patches
        // them on cache hits without rebuilding the descriptor.
        if (tilized) {
            KernelDescriptor::RTArgList reader_runtime_args;
            reader_runtime_args.push_back(src_buffer);
            reader_runtime_args.push_back(num_units_per_core);
            reader_runtime_args.push_back(start_id);

            KernelDescriptor::RTArgList writer_runtime_args;
            writer_runtime_args.push_back(dst_buffer);
            writer_runtime_args.push_back(num_units_per_core);
            writer_runtime_args.push_back(start_id);

            if (sharded) {
                std::vector<uint32_t> sharding_args;
                shard_builder::extend_sharding_run_time_args(input, sharding_args);
                reader_runtime_args.append(sharding_args);
                writer_runtime_args.append(sharding_args);
            }
            reader_desc.emplace_runtime_args(core, reader_runtime_args);
            writer_desc.emplace_runtime_args(core, writer_runtime_args);
        } else {
            KernelDescriptor::RTArgList reader_runtime_args;
            reader_runtime_args.push_back(src_buffer);
            reader_runtime_args.push_back(input_unit_size);
            reader_runtime_args.push_back(num_units_per_core);
            reader_runtime_args.push_back(start_id);
            reader_runtime_args.push_back(full_input_row / input_unit_size);

            KernelDescriptor::RTArgList writer_runtime_args;
            writer_runtime_args.push_back(dst_buffer);
            writer_runtime_args.push_back(output_unit_size);
            writer_runtime_args.push_back(num_units_per_core);
            writer_runtime_args.push_back(start_id);
            writer_runtime_args.push_back(full_output_row / output_unit_size);

            if (sharded) {
                std::vector<uint32_t> sharding_args;
                shard_builder::extend_sharding_run_time_args(input, sharding_args);
                reader_runtime_args.append(sharding_args);
                writer_runtime_args.append(sharding_args);
            }
            reader_desc.emplace_runtime_args(core, reader_runtime_args);
            writer_desc.emplace_runtime_args(core, writer_runtime_args);
        }
        if (backwards) {
            start_id -= num_units_per_core;
        } else {
            start_id += num_units_per_core;
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (convert_dtype) {
        desc.kernels.push_back(std::move(compute_g1_desc));
        if (has_compute_g2) {
            desc.kernels.push_back(std::move(compute_g2_desc));
        }
    }

    return desc;
}

}  // namespace ttnn::prim
