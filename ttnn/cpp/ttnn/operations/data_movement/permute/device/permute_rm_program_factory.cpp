// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/device/permute_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::data_movement {

namespace detail {
uint32_t num_pages(const ttnn::Tensor& input_tensor) {
    const auto& shape = input_tensor.logical_shape();
    return static_cast<uint32_t>(shape.volume() / shape[-1]);
}

uint32_t page_size(const ttnn::Tensor& input_tensor) {
    auto BUFFER_ALIGNMENT = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                ? tt::tt_metal::hal::get_dram_alignment()
                                : tt::tt_metal::hal::get_l1_alignment();
    const auto& shape = input_tensor.logical_shape();
    return tt::round_up(shape[-1] * input_tensor.element_size(), BUFFER_ALIGNMENT);
}

std::vector<uint32_t> get_row_strides(const ttnn::Shape& shape) {
    std::vector<uint32_t> strides(shape.rank());
    strides[shape.rank() - 1] = 1;
    strides[shape.rank() - 2] = 1;
    for (int i = shape.rank() - 3; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

}  // namespace detail

tt::tt_metal::ProgramDescriptor PermuteDeviceOperation::MultiCoreRowInvariant::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t input_rm_page_size = detail::page_size(input_tensor);
    uint32_t output_rm_page_size = detail::page_size(tensor_return_value);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_pages_to_read = 2;

    uint32_t num_rows = detail::num_pages(input_tensor);

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows);

    uint32_t N = operation_attributes.dims.size();

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages_to_read * input_rm_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = input_rm_page_size,
        }}},
    });

    // Reader kernel
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_rm_row_invariant.cpp";
    reader_desc.core_ranges = all_cores;
    reader_desc.named_compile_time_args = {{"N", N}, {"page_size", input_rm_page_size}, {"num_rows", num_rows}};
    reader_desc.config = ReaderConfigDescriptor{};
    TensorAccessorArgs(*src_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_desc.compile_time_args, reader_desc.common_runtime_args);

    // Writer kernel
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_rm_row_invariant.cpp";
    writer_desc.core_ranges = all_cores;
    writer_desc.named_compile_time_args = {{"N", N}, {"page_size", output_rm_page_size}, {"num_rows", num_rows}};
    writer_desc.config = WriterConfigDescriptor{};
    TensorAccessorArgs(*dst_buffer).append_to(writer_desc.compile_time_args);

    auto input_shape_view = input_tensor.logical_shape().view();
    auto output_strides = detail::get_row_strides(output_tensor.logical_shape());

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_row = 0;
    for (const auto& core : cores) {
        uint32_t num_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_tiles_per_core_group_2;
        }
        uint32_t end_row = start_row + num_rows_per_core;

        KernelDescriptor::CoreRuntimeArgs reader_args = {src_buffer->address(), start_row, end_row};
        reader_desc.runtime_args.emplace_back(core, std::move(reader_args));

        KernelDescriptor::CoreRuntimeArgs writer_args = {dst_buffer->address(), start_row, end_row};
        writer_args.insert(writer_args.end(), input_shape_view.begin(), input_shape_view.end());
        writer_args.insert(writer_args.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
        writer_args.insert(writer_args.end(), output_strides.begin(), output_strides.end());
        writer_desc.runtime_args.emplace_back(core, std::move(writer_args));

        start_row = end_row;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

tt::tt_metal::ProgramDescriptor PermuteDeviceOperation::MultiCoreBlockedGeneric::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto* src_buffer = input_tensor.buffer();
    auto* dst_buffer = output_tensor.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t w_block_size = constants::TILE_WIDTH;
    uint32_t input_cb_page_size = w_block_size * input_tensor.element_size();

    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t x_block_size = constants::TILE_HEIGHT;
    uint32_t output_cb_page_size = x_block_size * input_tensor.element_size();

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t src1_cb_index = tt::CBIndex::c_2;
    uint32_t src2_cb_index = tt::CBIndex::c_1;
    uint32_t num_input_pages_to_read = 2;

    uint32_t x_dim = operation_attributes.dims.back();
    uint32_t X = input_tensor.logical_shape()[x_dim];
    auto input_strides = detail::get_row_strides(input_tensor.logical_shape());
    uint32_t X_stride = input_strides[x_dim];

    auto output_strides = detail::get_row_strides(output_tensor.logical_shape());
    uint32_t W = input_tensor.logical_shape()[-1];
    uint32_t W_stride = output_strides[x_dim];

    uint32_t N = operation_attributes.dims.size();
    uint32_t num_rows = detail::num_pages(input_tensor);

    uint32_t x_blocks = tt::div_up(X, x_block_size);
    uint32_t w_blocks = tt::div_up(W, w_block_size);
    uint32_t num_blocks_total = (num_rows / X) * x_blocks * w_blocks;

    auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks_total);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages_to_read * input_cb_page_size * x_block_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = input_cb_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages_to_read * output_cb_page_size * w_block_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src1_cb_index),
            .data_format = cb_data_format,
            .page_size = output_cb_page_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_pages_to_read * x_block_size * w_block_size * input_tensor.element_size(),
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src2_cb_index),
            .data_format = cb_data_format,
            .page_size = x_block_size * w_block_size * input_tensor.element_size(),
        }}},
    });

    // Reader kernel
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "reader_permute_interleaved_rm_blocked_generic.cpp";
    reader_desc.core_ranges = all_cores;
    reader_desc.named_compile_time_args = {
        {"N", N},
        {"page_size", input_cb_page_size},
        {"num_rows", num_rows},
        {"x_dim", x_dim},
        {"num_blocks_total", num_blocks_total},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", x_block_size},
        {"w_block_size", w_block_size},
        {"element_size", input_tensor.element_size()},
        {"input_tensor_page_size", static_cast<uint32_t>(src_buffer->aligned_page_size())}};
    reader_desc.config = ReaderConfigDescriptor{};
    TensorAccessorArgs(*src_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_desc.compile_time_args, reader_desc.common_runtime_args);

    // Writer kernel
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/dataflow/"
        "writer_permute_interleaved_rm_blocked_generic.cpp";
    writer_desc.core_ranges = all_cores;
    writer_desc.named_compile_time_args = {
        {"N", N},
        {"output_page_size", output_cb_page_size},
        {"num_rows", num_rows},
        {"X", X},
        {"X_stride", X_stride},
        {"x_dim", x_dim},
        {"W_stride", W_stride},
        {"input_page_size", input_cb_page_size},
        {"element_size", input_tensor.element_size()},
        {"num_blocks_total", num_blocks_total},
        {"x_blocks", x_blocks},
        {"w_blocks", w_blocks},
        {"x_block_size", x_block_size},
        {"w_block_size", w_block_size},
        {"W", W},
        {"output_tensor_page_size", static_cast<uint32_t>(dst_buffer->aligned_page_size())}};
    writer_desc.config = WriterConfigDescriptor{};
    TensorAccessorArgs(*dst_buffer).append_to(writer_desc.compile_time_args);

    // Compute kernel
    bool fp32_dest_acc_en = cb_data_format_output == tt::DataFormat::Float32 ||
                            cb_data_format_output == tt::DataFormat::Int32 ||
                            cb_data_format_output == tt::DataFormat::UInt32;

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/permute/device/kernels/compute/transpose_xw_rm_single_tile_size.cpp";
    compute_desc.core_ranges = all_cores;
    compute_desc.named_compile_time_args = {{"x_block_size", x_block_size}, {"w_block_size", w_block_size}};
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    auto input_shape_view = input_tensor.logical_shape().view();

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_block = 0;
    for (const auto& core : cores) {
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_tiles_per_core_group_2;
        }
        uint32_t end_block = start_block + num_blocks_per_core;

        KernelDescriptor::CoreRuntimeArgs reader_args = {src_buffer->address(), start_block, end_block};
        reader_args.insert(reader_args.end(), input_shape_view.begin(), input_shape_view.end());
        reader_args.insert(reader_args.end(), input_strides.begin(), input_strides.end());
        reader_desc.runtime_args.emplace_back(core, std::move(reader_args));

        KernelDescriptor::CoreRuntimeArgs writer_args = {dst_buffer->address(), start_block, end_block};
        writer_args.insert(writer_args.end(), input_shape_view.begin(), input_shape_view.end());
        writer_args.insert(writer_args.end(), operation_attributes.dims.begin(), operation_attributes.dims.end());
        writer_args.insert(writer_args.end(), output_strides.begin(), output_strides.end());
        writer_desc.runtime_args.emplace_back(core, std::move(writer_args));

        compute_desc.runtime_args.emplace_back(core, KernelDescriptor::CoreRuntimeArgs{num_blocks_per_core});

        start_block = end_block;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::data_movement
