// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_sharded_program_factory.hpp"

#include <cmath>

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

ProgramDescriptor UntilizeWithUnpaddingMultiCoreShardedProgramFactory::create_descriptor(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    bool src_sharded = a.memory_config().is_sharded();
    bool out_sharded = output.memory_config().is_sharded();
    // Special handling for tensors of W=16 and H%32==0
    // In this case skip untilizing on compute and in writer kernel just copy face0 and face2,
    // and skip face1 and face3.
    bool unpad_tensor_w_16 = output.padded_shape()[-1] == 16 && output.padded_shape()[-2] % TILE_HEIGHT == 0;
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t num_rows_block = 0, block_row_size = 0, output_row_size = 0, last_block_row_size_unpadded = 0,
             num_output_rows_unpadded = 0;
    CoreCoord end_core;
    uint32_t last_idx = 0;
    auto shard_spec = a.shard_spec().value();

    // I am not sure it is correct to ever use the shard_spec here.
    auto out_shard_spec = output.shard_spec().has_value() ? output.shard_spec().value() : shard_spec;

    bool row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    auto all_cores = shard_spec.grid;
    uint32_t ntiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
    uint32_t nblocks_per_core = shard_spec.shape[0] / TILE_HEIGHT;
    uint32_t global_batch = a.physical_volume() / (a.padded_shape()[-2] * a.padded_shape()[-1]);
    uint32_t batch =
        a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED
            ? std::max(1u, (shard_spec.shape[0] * shard_spec.shape[1]) / (a.padded_shape()[-2] * a.padded_shape()[-1]))
            : global_batch;
    uint32_t ntiles_per_batch = ntiles_per_block * nblocks_per_core / batch;

    num_rows_block = out_shard_spec.shape[0];
    block_row_size = out_shard_spec.shape[1] * output.element_size();     // in0_block_w * TILE_WIDTH * dtype_nbytes
    output_row_size = output.padded_shape()[-1] * output.element_size();  // output row size bytes
    last_block_row_size_unpadded = block_row_size - (tt::round_up(output.padded_shape()[-1], out_shard_spec.shape[1]) -
                                                     output.padded_shape()[-1]) *
                                                        output.element_size();
    uint32_t num_output_rows = output.physical_volume() / output.padded_shape()[-1];
    num_output_rows_unpadded =
        num_rows_block - (tt::round_up(num_output_rows, out_shard_spec.shape[0]) - num_output_rows);
    if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
        last_idx = tt::div_up(output.padded_shape()[-1], out_shard_spec.shape[1]) - 1;
    } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
        last_idx = tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1;
    } else {
        end_core = {
            tt::div_up(output.padded_shape()[-1], out_shard_spec.shape[1]) - 1,
            tt::div_up(num_output_rows, out_shard_spec.shape[0]) - 1};
    }
    if (!row_major) {
        std::swap(end_core.x, end_core.y);
    }

    uint32_t num_input_tiles = ntiles_per_block * nblocks_per_core;
    uint32_t num_output_tiles = out_sharded ? (unpad_tensor_w_16 ? 16 : ntiles_per_batch * 2) : ntiles_per_block * 2;
    uint32_t aligned_page_size = static_cast<uint32_t>(output.buffer()->aligned_page_size());

    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const uint32_t src0_cb_index = tt::CBIndex::c_0;
    const uint32_t output_cb_index = tt::CBIndex::c_16;
    const uint32_t sharded_output_cb_index = tt::CBIndex::c_17;

    ProgramDescriptor desc;

    // Sharded input CB — globally allocated to the input buffer; framework patches
    // the CB address on cache hits via cb.buffer.
    {
        CBDescriptor cb_src0;
        cb_src0.total_size = num_input_tiles * input_single_tile_size;
        cb_src0.core_ranges = all_cores;
        cb_src0.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        });
        if (src_sharded) {
            cb_src0.buffer = a.buffer();
        }
        desc.cbs.push_back(std::move(cb_src0));
    }

    // Intermediate (untilized) output CB
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });

    if (out_sharded) {
        // The kernel advances the write pointer by aligned_page_size (which may be
        // larger than block_row_size due to buffer alignment padding), so the CB
        // page size must match to avoid overflow.
        // Sharded output CB — globally allocated to the output buffer.
        CBDescriptor cb_sharded_output;
        cb_sharded_output.total_size = num_output_rows_unpadded * aligned_page_size;
        cb_sharded_output.core_ranges = all_cores;
        cb_sharded_output.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sharded_output_cb_index),
            .data_format = output_cb_data_format,
            .page_size = aligned_page_size,
        });
        cb_sharded_output.buffer = dst_buffer;
        desc.cbs.push_back(std::move(cb_sharded_output));
    }

    /** reader
     */
    std::vector<uint32_t> reader_ct_args = {src0_cb_index};

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};

    /** writer
     */
    KernelDescriptor writer_desc;
    if (out_sharded) {
        std::vector<uint32_t> writer_ct_args{output_cb_index, sharded_output_cb_index, aligned_page_size};
        writer_desc.kernel_source =
            unpad_tensor_w_16
                ? "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                  "writer_unary_unpad_width_16_sharded.cpp"
                : "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
                  "writer_unary_unpad_batch_rows_sharded.cpp";
        writer_desc.compile_time_args = std::move(writer_ct_args);
    } else {
        std::vector<uint32_t> writer_ct_args = {
            static_cast<uint32_t>(
                input_cb_data_format == tt::DataFormat::Float32 || input_cb_data_format == tt::DataFormat::UInt32 ||
                input_cb_data_format == tt::DataFormat::Int32),
            output_row_size};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
        writer_desc.kernel_source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_blocks.cpp";
        writer_desc.compile_time_args = std::move(writer_ct_args);
    }
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.config = WriterConfigDescriptor{};

    /** compute
     */
    std::vector<uint32_t> compute_args = {
        nblocks_per_core,  // per_core_block_cnt
        ntiles_per_block,  // per_block_ntiles
        src0_cb_index,
        output_cb_index,
    };

    std::vector<std::pair<std::string, std::string>> compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }
    std::string compute_kernel("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
    if (unpad_tensor_w_16) {
        // Use copy compute kernel just for a potential data type conversion.
        compute_kernel = "ttnn/cpp/ttnn/kernel/compute/eltwise_copy.cpp";
        compute_args[0] = num_input_tiles;  // per_core_tile_cnt
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.defines = std::move(compute_kernel_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    // reader runtime args — sharded reader CB itself carries the buffer binding,
    // so only the per-launch tile count is needed.
    KernelDescriptor::RTArgList reader_rt_args;
    reader_rt_args.append(std::vector<uint32_t>{ntiles_per_block * nblocks_per_core});
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt, row_major)) {
        reader_desc.emplace_runtime_args(core, reader_rt_args);
    }

    if (out_sharded) {
        // Sharded writer: CB carries the buffer binding; only per-shard counters are needed.
        std::vector<uint32_t> writer_rt_args_vec;
        if (unpad_tensor_w_16) {
            writer_rt_args_vec = {num_output_rows_unpadded, num_input_tiles};
        } else {
            writer_rt_args_vec = {
                num_output_rows_unpadded,
                ntiles_per_batch,
                static_cast<uint32_t>(out_shard_spec.shape[0] / batch),
                static_cast<uint32_t>(shard_spec.shape[1] * output.element_size()),
                block_row_size,
                batch};
        }
        KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.append(writer_rt_args_vec);
        for (const auto& core : corerange_to_cores(all_cores, std::nullopt, row_major)) {
            writer_desc.emplace_runtime_args(core, writer_rt_args);
        }
    } else {
        const auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);
        for (uint32_t i = 0; i < cores.size(); ++i) {
            const CoreCoord& core = cores[i];

            // writer runtime args
            uint32_t block_start_row_offset;
            uint32_t block_start_row_id_offset;
            uint32_t row_size_unpadded = block_row_size;
            uint32_t num_rows_unpadded = num_rows_block;
            if (a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
                block_start_row_offset = i * block_row_size;
                block_start_row_id_offset = 0;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    num_rows_unpadded = num_output_rows_unpadded;
                    if (i == last_idx) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                }
            } else if (a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                block_start_row_offset = 0;
                block_start_row_id_offset = i * num_rows_block;
                if (i > last_idx) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                } else {
                    if (i == last_idx) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                    row_size_unpadded = last_block_row_size_unpadded;
                }
            } else {
                if (row_major) {
                    block_start_row_offset = core.x * block_row_size;
                    block_start_row_id_offset = core.y * num_rows_block;
                    if (core.x == end_core.x) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.y == end_core.y) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                } else {
                    block_start_row_offset = core.y * block_row_size;
                    block_start_row_id_offset = core.x * num_rows_block;
                    if (core.y == end_core.y) {
                        row_size_unpadded = last_block_row_size_unpadded;
                    }
                    if (core.x == end_core.x) {
                        num_rows_unpadded = num_output_rows_unpadded;
                    }
                }
                if (core.x > end_core.x || core.y > end_core.y) {
                    row_size_unpadded = 0;
                    num_rows_unpadded = 0;
                }
            }

            // writer runtime args — Buffer* slot auto-registers as a BufferBinding so the
            // framework patches addresses on cache hits.
            writer_desc.emplace_runtime_args(
                core,
                {dst_buffer,  // dst_addr
                 num_rows_block,
                 block_row_size,
                 1u,
                 1u,
                 1u,
                 row_size_unpadded,
                 num_rows_unpadded,
                 block_start_row_id_offset,
                 block_start_row_offset});
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
