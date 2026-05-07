// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/common/constants.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create_descriptor(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t num_blocks = input_shape[-1] == 0 ? 0 : a.physical_volume() / input_shape[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t padded_row_size_bytes;
    uint32_t unpadded_row_size_bytes;

    if (a.dtype() == DataType::BFLOAT8_B) {
        padded_row_size_bytes = input_shape[-1] * output.element_size();
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
    } else {
        padded_row_size_bytes = input_shape[-1] * a.element_size();
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
    }

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_index = tt::CBIndex::c_16;

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    ProgramDescriptor desc;

    // Input CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_per_row * input_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    // Output CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles_per_row * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    /** reader
     */
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {
        static_cast<uint32_t>(
            input_cb_data_format == tt::DataFormat::Float32 || input_cb_data_format == tt::DataFormat::UInt32 ||
            input_cb_data_format == tt::DataFormat::Int32),
        unpadded_row_size_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multicore.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};

    /** compute
     */
    std::vector<std::pair<std::string, std::string>> compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }
    const std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");

    std::optional<KernelDescriptor> compute_desc;
    if (!core_range.empty()) {
        KernelDescriptor cd;
        cd.kernel_source = compute_kernel;
        cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd.core_ranges = core_range;
        cd.compile_time_args = {nblocks_per_core, num_tiles_per_row, src0_cb_index, output_cb_index};
        cd.defines = compute_kernel_defines;
        cd.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
        };
        compute_desc = std::move(cd);
    }

    std::optional<KernelDescriptor> compute_cliff_desc;
    if (has_cliff) {
        KernelDescriptor cd;
        cd.kernel_source = compute_kernel;
        cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
        cd.core_ranges = core_range_cliff;
        cd.compile_time_args = {nblocks_per_core_cliff, num_tiles_per_row, src0_cb_index, output_cb_index};
        cd.defines = std::move(compute_kernel_defines);
        cd.config = ComputeConfigDescriptor{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        };
        compute_cliff_desc = std::move(cd);
    }

    uint32_t tile_height = output.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;

    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // writer runtime args — Buffer* slot auto-registers as a BufferBinding so the
        // framework patches addresses on cache hits.
        KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.reserve(4 + assignment.size() * 5);
        writer_rt_args.push_back(dst_buffer);
        writer_rt_args.push_back(padded_row_size_bytes);
        writer_rt_args.push_back(row_start_id);
        writer_rt_args.push_back(static_cast<uint32_t>(assignment.size()));

        uint32_t nblocks_per_core_local = 0;

        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
        for (const auto& el : assignment) {
            nblocks_per_core_local += el.block_count();
            row_start_id += el.data_row_count();
            if (compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                // push back information for previous elements
                writer_rt_args.push_back(ref_el.n_data);
                writer_rt_args.push_back(ref_el.n_mixed);
                writer_rt_args.push_back(ref_el.n_pads);
                writer_rt_args.push_back(ref_el.times);
                writer_rt_args.push_back(count_repeated);
                // Set up assignment for this element
                ref_el = el;
                count_repeated = 1;
            }
        }
        writer_rt_args.push_back(ref_el.n_data);
        writer_rt_args.push_back(ref_el.n_mixed);
        writer_rt_args.push_back(ref_el.n_pads);
        writer_rt_args.push_back(ref_el.times);
        writer_rt_args.push_back(count_repeated);

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core_local;

        // reader runtime args — Buffer* slot auto-registers as a BufferBinding.
        reader_desc.emplace_runtime_args(core, {src0_buffer, num_tiles_per_core, tile_start_id});
        writer_desc.emplace_runtime_args(core, writer_rt_args);

        tile_start_id += num_tiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (compute_desc.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc));
    }
    if (compute_cliff_desc.has_value()) {
        desc.kernels.push_back(std::move(*compute_cliff_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
