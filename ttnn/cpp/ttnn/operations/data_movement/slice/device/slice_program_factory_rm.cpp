// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

constexpr uint32_t MAX_READ_SIZE = 4096;

std::tuple<uint32_t, uint32_t, uint32_t> compute_cb_size(
    const Tensor& input,
    const Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const uint32_t num_sticks_per_core_group_1,
    const uint32_t num_sticks_per_core_group_2) {
    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    const auto single_alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    auto alignment = single_alignment;

    // if begins is not aligned then we need to pad the cb size, so that we can read from the nearest aligned address
    uint32_t begins_bytes = output_tensor_start[-1] * input.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    const ttnn::Shape& output_shape = output.padded_shape();
    const uint32_t unpadded_row_size_bytes = output_shape[-1] * input.element_size();
    const uint32_t cb_page_size = tt::round_up(unpadded_row_size_bytes, alignment);
    // Kernel runtime args use the single-aligned stick stride; CB sizing must compute num_read_per_barrier
    // against the same stride, otherwise the CB pages vs the kernel's reserve_back(N) can diverge and the
    // reader deadlocks on cb_reserve_back.
    const uint32_t stick_stride_for_merge = tt::round_up(unpadded_row_size_bytes, single_alignment);
    const uint32_t num_input_pages = num_sticks_per_core_group_1 > num_sticks_per_core_group_2
                                         ? num_sticks_per_core_group_1
                                         : num_sticks_per_core_group_2;
    uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
    if (num_input_pages != 0) {
        auto num_sticks_per_core_pad32 = num_input_pages + ((32 - num_input_pages % 32) % 32);
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, stick_stride_for_merge, MAX_READ_SIZE);
        num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;
    }

    return std::make_tuple(cb_page_size, num_read_per_barrier, misalignment);
}

}  // namespace

ProgramDescriptor SliceRmProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    IDevice* device = input.device();

    uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    Buffer* src0_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());

    constexpr uint32_t src0_cb_index = 0;

    const auto [cb_page_size, num_read_per_barrier, misalignment] =
        compute_cb_size(input, output, args.slice_start, num_sticks_per_core_group_1, num_sticks_per_core_group_2);

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_read_per_barrier * 2 * cb_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format,
            .page_size = cb_page_size,
        }}},
    });

    // Compute reader common runtime args + per-core args.
    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();

    uint32_t padded_row_size_bytes = input_shape[-1] * input.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input.element_size();

    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    uint32_t begins_bytes = args.slice_start[-1] * input.element_size();
    uint32_t local_misalignment = begins_bytes % src_buffer_alignment;
    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, alignment);
    uint32_t start_addr = src0_buffer->address();

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> accumulated_total_per_dim(num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    // This currently just matches tile version where we iterate over the row as well
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (int32_t i = 1; i < static_cast<int32_t>(num_dims); i++) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    // --- Reader Kernel ---
    std::vector<uint32_t> reader_compile_time_args_vec;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args_vec);

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args_vec);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    auto all_cores_vec = corerange_to_cores(all_cores);
    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input, args.slice_start);
    uint32_t num_sticks_written = 0;
    std::vector<uint32_t> id_per_dim(num_dims);

    // --- Writer Kernel descriptor (built in same loop with reader args) ---
    std::vector<uint32_t> writer_compile_time_args_vec = {static_cast<uint32_t>(src0_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args_vec);

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_writer_unary_stick_layout_interleaved_start_id.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args_vec);
    writer_kernel_desc.config = WriterConfigDescriptor{};

    for (const auto& core : all_cores_vec) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            // no-op
            num_sticks_per_core = 0;
        }

        // issue more reads before calling barrier
        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier_local = 0;
        if (num_sticks_per_core != 0) {
            auto num_sticks_per_core_pad32 = num_sticks_per_core + ((32 - num_sticks_per_core % 32) % 32);
            num_sticks_per_core_read = tt::tt_metal::merge_num_sticks_to_read(
                num_sticks_per_core_pad32, unpadded_row_size_bytes_offset, MAX_READ_SIZE);
            num_read_per_barrier_local = num_sticks_per_core_pad32 / num_sticks_per_core_read;
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;

        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        // Reader runtime arg layout:
        //   [0] src buffer base + begins_bytes - misalignment   (offset-bearing — plain uint32_t)
        //   [1..9] padded/unpadded row size, num_dims, misalignment, start_id, etc.
        //   [10..]  per-dim stride tables
        // Slot 0 carries a non-zero offset so the kernel reads from the nearest aligned address; we cannot use
        // BufferBinding here because the framework would patch raw buffer.address() and lose the offset/misalignment
        // adjustment. The slow cache-hit path re-runs create_descriptor() each dispatch, which recomputes this slot
        // exactly the way legacy override_runtime_arguments() did.
        std::vector<std::variant<uint32_t, Buffer*>> reader_args;
        reader_args.reserve(10 + 3 * num_dims);
        reader_args.emplace_back(start_addr + begins_bytes - local_misalignment);
        reader_args.emplace_back(padded_row_size_bytes);
        reader_args.emplace_back(unpadded_row_size_bytes);
        reader_args.emplace_back(unpadded_row_size_bytes_offset);
        reader_args.emplace_back(num_dims);
        reader_args.emplace_back(local_misalignment);
        reader_args.emplace_back(start_id);
        reader_args.emplace_back(num_sticks_per_core);
        reader_args.emplace_back(num_sticks_per_core_read);
        reader_args.emplace_back(num_read_per_barrier_local);
        for (uint32_t v : num_unpadded_sticks_per_dim) {
            reader_args.emplace_back(v);
        }
        for (uint32_t v : num_padded_sticks_per_dim) {
            reader_args.emplace_back(v);
        }
        for (uint32_t v : id_per_dim) {
            reader_args.emplace_back(v);
        }
        reader_kernel_desc.emplace_runtime_args(core, reader_args);

        // Writer slot 0 is the raw output buffer base address (no offset). Use Buffer* for fast-path patching
        // on active cores; idle cores pass 0u to skip BufferBinding registration since the kernel
        // short-circuits when num_sticks_per_core is 0 and never dereferences the address.
        if (num_sticks_per_core != 0) {
            writer_kernel_desc.emplace_runtime_args(
                core,
                {dst_buffer,
                 unpadded_row_size_bytes,
                 unpadded_row_size_bytes_offset,
                 num_sticks_per_core,
                 num_sticks_per_core_read,
                 num_read_per_barrier_local,
                 num_sticks_written,
                 0u});
        } else {
            writer_kernel_desc.emplace_runtime_args(
                core,
                {0u, unpadded_row_size_bytes, unpadded_row_size_bytes_offset, 0u, 0u, 0u, num_sticks_written, 0u});
        }

        num_sticks_written += num_sticks_per_core;
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));

    return desc;
}

}  // namespace ttnn::prim
