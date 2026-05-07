// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor SliceTileProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    IDevice* device = input.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    Buffer* src0_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    // --- CB Descriptor ---
    constexpr uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;

    ProgramDescriptor program_descriptor;

    CBDescriptor cb_desc;
    cb_desc.total_size = num_input_tiles * single_tile_size;
    cb_desc.core_ranges = all_cores;
    cb_desc.format_descriptors.push_back(CBFormatDescriptor{
        .buffer_index = static_cast<uint8_t>(src0_cb_index),
        .data_format = cb_data_format,
        .page_size = single_tile_size});
    program_descriptor.cbs.push_back(std::move(cb_desc));

    // --- Reader Kernel Descriptor ---
    // CB index via named compile-time arg (essential for fusion CB remapping).
    std::vector<uint32_t> reader_compile_time_args = {num_dims};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    // Reader common runtime args: [src_addr, num_unpadded_per_dim..., num_padded_per_dim...]
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    std::vector<uint32_t> reader_common_args(1 + (num_dims * 2));
    reader_common_args[0] = src0_buffer->address();
    uint32_t* num_unpadded_tiles_per_dim = reader_common_args.data() + 1;
    uint32_t* num_padded_tiles_per_dim = num_unpadded_tiles_per_dim + num_dims;
    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    // Reader common arg slot 0 is src_addr (raw buffer.address(), no offset). Register it as a CommonBufferBinding
    // for the fast cache-hit patch path; the start_offset/start_id math is encoded in the per-core args below.
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "reader_unary_unpad_dims_interleaved_start_id.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = reader_compile_time_args;
    reader_kernel_desc.named_compile_time_args = {{"cb_in", src0_cb_index}};
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    // emplace_common_runtime_args pushes Buffer* at slot 0 as a CommonBufferBinding so the framework
    // patches the new buffer.address() on cache hits without rebuilding the descriptor.
    KernelDescriptor::RTArgList reader_common_rtargs;
    reader_common_rtargs.reserve(reader_common_args.size());
    reader_common_rtargs.push_back(src0_buffer);
    for (size_t i = 1; i < reader_common_args.size(); ++i) {
        reader_common_rtargs.push_back(reader_common_args[i]);
    }
    reader_kernel_desc.emplace_common_runtime_args(reader_common_rtargs);

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input, args.slice_start);

    // Reader per-core runtime args: [start_id, num_tiles, id_per_dim...]
    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            std::vector<uint32_t> reader_args(2 + num_dims, 0);
            reader_kernel_desc.runtime_args.emplace_back(core, std::move(reader_args));
            continue;
        }

        std::vector<uint32_t> reader_args(2 + num_dims);
        // Compute per-dim indices for this core's starting position
        reader_args[2] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = reader_args[2] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            reader_args[2 + j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += reader_args[2 + j] * accumulated_total_per_dim[j - 1];
        }
        reader_args[0] = start_id;
        reader_args[1] = num_tiles_per_core;

        reader_kernel_desc.runtime_args.emplace_back(core, std::move(reader_args));
        num_tiles_written += num_tiles_per_core;
    }

    program_descriptor.kernels.push_back(std::move(reader_kernel_desc));

    // --- Writer Kernel Descriptor ---
    // CB index via named compile-time arg (essential for fusion CB remapping).
    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "writer_unary_interleaved_start_id.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = writer_compile_time_args;
    writer_kernel_desc.named_compile_time_args = {{"cb_out", src0_cb_index}};
    writer_kernel_desc.config = WriterConfigDescriptor{};

    // Writer per-core runtime args: [dst_addr, num_tiles, start_id]
    // Slot 0 holds raw dst_buffer.address() (no offset). Active cores register Buffer* for fast cache-hit
    // patching; idle (no-op) cores pass 0u to skip BufferBinding registration since the kernel
    // short-circuits when num_tiles is 0 and never dereferences the address.
    num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            writer_kernel_desc.emplace_runtime_args(core, {0u, 0u, 0u});
            continue;
        }

        writer_kernel_desc.emplace_runtime_args(core, {dst_buffer, num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    program_descriptor.kernels.push_back(std::move(writer_kernel_desc));

    return program_descriptor;
}

}  // namespace ttnn::prim
