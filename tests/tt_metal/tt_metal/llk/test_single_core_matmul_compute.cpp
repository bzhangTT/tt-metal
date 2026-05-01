// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <fmt/base.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <unistd.h>
#include <functional>
#include <map>
#include <memory>
#include <ostream>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/kernel_types.hpp>
#include "device_fixture.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/tt_backend_api_types.hpp>
#include "tt_metal/test_utils/comparison.hpp"
#include "tt_metal/test_utils/df/float32.hpp"
#include "tt_metal/test_utils/float8_utils.hpp"
#include "tt_metal/test_utils/packing.hpp"
#include "tt_metal/test_utils/print_helpers.hpp"
#include "tt_metal/test_utils/stimulus.hpp"

namespace tt::tt_metal {
class IDevice;
}  // namespace tt::tt_metal

namespace tt::tt_metal {

using namespace tt;
using namespace tt::test_utils;
using namespace tt::test_utils::df;

namespace unit_tests::compute::matmul {

void create_CBs_for_fused_matmul(
    distributed::MeshWorkload& workload,
    const std::shared_ptr<distributed::MeshDevice>& /*mesh_device*/,
    CoreCoord core,
    bool activations_rm,
    bool output_rm,
    uint32_t M,
    uint32_t N,
    uint32_t in0_block_w,
    uint32_t /*out_subblock_h*/) {
    uint32_t num_bytes_for_df = 2;
    uint32_t in0_cb = 0;
    uint32_t in1_cb = 1;
    uint32_t tilize_mode_tilized_in0_cb = 24;
    uint32_t matmul_partials_cb = 25;
    uint32_t untilize_mode_final_matmul_partials_cb = 26;
    uint32_t untilize_mode_reblock_cb = 27;
    uint32_t out0_cb = 16;

    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    auto& program = workload.get_programs().at(device_range);

    uint32_t single_tile_size = num_bytes_for_df * 1024;

    uint32_t num_output_tiles = M * N;

    // Invariants
    uint32_t cb0_tiles = M * in0_block_w * 2;
    tt_metal::CircularBufferConfig l1_input0_cb_config =
        tt_metal::CircularBufferConfig(cb0_tiles * single_tile_size, {{in0_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in0_cb, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, l1_input0_cb_config);

    uint32_t cb1_tiles = N * in0_block_w * 2;
    tt_metal::CircularBufferConfig cb_in1_config =
        tt_metal::CircularBufferConfig(cb1_tiles * single_tile_size, {{in1_cb, tt::DataFormat::Float16_b}})
            .set_page_size(in1_cb, single_tile_size);
    tt_metal::CreateCircularBuffer(program, core, cb_in1_config);

    if (not activations_rm and not output_rm) {  // no tilize, no untilize
        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Partials share same L1 address space as output
        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (not activations_rm and output_rm) {  // no tilize, just untilize

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Need a new CB to push output block to since other
        // intermediate read pointer changes in enable reload
        // block
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N;  // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config =
            tt_metal::CircularBufferConfig(
                reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else if (activations_rm and not output_rm) {  // just tilize, no untilize

        tt_metal::CircularBufferConfig cb_src0_tilized_config =
            tt_metal::CircularBufferConfig(
                cb0_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);

    } else {  // tilize activations and untilize output

        // Used for placing tilized activations
        tt_metal::CircularBufferConfig cb_src0_tilized_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{tilize_mode_tilized_in0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(tilize_mode_tilized_in0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_src0_tilized_config);

        tt_metal::CircularBufferConfig cb_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size, {{matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_matmul_partials_config);

        // Shares same address space as matmul partials
        tt_metal::CircularBufferConfig cb_final_matmul_partials_config =
            tt_metal::CircularBufferConfig(
                num_output_tiles * single_tile_size,
                {{untilize_mode_final_matmul_partials_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_final_matmul_partials_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_final_matmul_partials_config);

        // Supposed to be a small CB only responsible for reorganizing
        // the output blocks to fill the whole "per core output block width"
        uint32_t reblock_cb_tiles = N;  // Only space for one row
        tt_metal::CircularBufferConfig cb_reblock_config =
            tt_metal::CircularBufferConfig(
                reblock_cb_tiles * single_tile_size, {{untilize_mode_reblock_cb, tt::DataFormat::Float16_b}})
                .set_page_size(untilize_mode_reblock_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_reblock_config);

        tt_metal::CircularBufferConfig cb_output_config =
            tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{out0_cb, tt::DataFormat::Float16_b}})
                .set_page_size(out0_cb, single_tile_size);
        tt_metal::CreateCircularBuffer(program, core, cb_output_config);
    }
}

// Single-tile matmul. Inputs and output formats are programmable; default
// (Float16_b in, Float16_b out, fp32_dest_acc_en=false) preserves the legacy
// BF16 test semantics. Pass Fp8_e4m3 for the FP8 enablement variants on
// Blackhole (see PR #40287/#41142 for the LLK family fix-up).
bool single_tile_matmul(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b,
    tt::DataFormat out_fmt = tt::DataFormat::Float16_b,
    bool fp32_dest_acc_en = false) {
    bool pass = true;
    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const uint32_t in_tile_size = tt::tile_size(in_fmt);
    const uint32_t out_tile_size = tt::tile_size(out_fmt);

    auto* device = mesh_device->get_devices()[0];
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::InterleavedBufferConfig dram_in_config{
        .device = device,
        .size = in_tile_size,
        .page_size = in_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_out_config{
        .device = device,
        .size = out_tile_size,
        .page_size = out_tile_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_in_config);
    auto input1_dram_buffer = CreateBuffer(dram_in_config);
    auto output_dram_buffer = CreateBuffer(dram_out_config);

    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(in_tile_size, {{in0_cb_index, in_fmt}})
            .set_page_size(in0_cb_index, in_tile_size));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(in_tile_size, {{in1_cb_index, in_fmt}})
            .set_page_size(in1_cb_index, in_tile_size));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(out_tile_size, {{out_cb_index, out_fmt}})
            .set_page_size(out_cb_index, out_tile_size));

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/single_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = {in0_cb_index, in1_cb_index, out_cb_index}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus & Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    // Random U(-1, +1) stimulus for both formats; golden is the actual
    // matmul evaluated host-side over the post-quantization input floats.
    // Trivial stimulus is intentionally avoided because it masks structural
    // bugs (e.g. a wrong K-stride is invisible if every K-slice tile is
    // bit-identical) -- exactly the pattern that hid the multi_tile_compute
    // in1_index_c_offset bug for the BF16 path.
    std::vector<uint32_t> packed_input0;
    std::vector<uint32_t> packed_input1;
    std::vector<float> in0_floats;
    std::vector<float> in1_floats;
    std::vector<float> golden_floats;
    constexpr float rng = 1.0f;

    if (in_fmt == tt::DataFormat::Fp8_e4m3) {
        packed_input0 = generate_packed_uniform_random_vector<uint32_t, float8_e4m3>(
            float8_e4m3(-rng), float8_e4m3(+rng), tt::constants::TILE_HW, /*seed=*/0);
        packed_input1 = generate_packed_uniform_random_vector<uint32_t, float8_e4m3>(
            float8_e4m3(-rng), float8_e4m3(+rng), tt::constants::TILE_HW, /*seed=*/1);
        in0_floats = fp8_to_floats(packed_input0);
        in1_floats = fp8_to_floats(packed_input1);
    } else {  // bfloat16
        packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
            bfloat16(-rng), bfloat16(+rng), tt::constants::TILE_HW, /*seed=*/0);
        packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
            bfloat16(-rng), bfloat16(+rng), tt::constants::TILE_HW, /*seed=*/1);
        in0_floats = bf16_to_floats(packed_input0);
        in1_floats = bf16_to_floats(packed_input1);
    }
    golden_floats.assign(tt::constants::TILE_HW, 0.0f);
    for (uint32_t y = 0; y < tt::constants::TILE_HEIGHT; y++) {
        for (uint32_t x = 0; x < tt::constants::TILE_WIDTH; x++) {
            float acc = 0.0f;
            for (uint32_t z = 0; z < tt::constants::TILE_WIDTH; z++) {
                acc += in0_floats[byte_tile_face_major_index(z, y)] * in1_floats[byte_tile_face_major_index(x, z)];
            }
            golden_floats[byte_tile_face_major_index(x, y)] = acc;
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {(uint32_t)input0_dram_buffer->address(),
         (uint32_t)0,
         (uint32_t)input1_dram_buffer->address(),
         (uint32_t)0,
         (uint32_t)1});
    tt_metal::SetRuntimeArgs(
        program_, writer_kernel, core, {(uint32_t)output_dram_buffer->address(), (uint32_t)0, (uint32_t)1});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    const bool fp8_out = (out_fmt == tt::DataFormat::Fp8_e4m3);
    auto dest_floats = fp8_out ? fp8_to_floats(dest_buffer_data) : bf16_to_floats(dest_buffer_data);

    // Tolerances under random U(-1, +1) stimulus: FP8 output tracks ~1/8
    // quantization error; BF16 output (with default fp32_dest_acc=false uses
    // BF16 dest accumulation, so per-element error grows with the K=32 inner
    // dim) needs a few-percent slack.
    const float rtol = fp8_out ? 0.125f : 0.05f;
    const float atol = fp8_out ? 0.125f : 0.2f;
    pass &= tt::test_utils::is_close_vectors<float>(dest_floats, golden_floats, rtol, atol);
    return pass;
}
// Single-block matmul: blocking that still fits within Dst (no spill/reload).
// Inputs and output formats are programmable; defaults preserve the legacy
// BF16 test semantics.
bool single_block_matmul(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    tt::DataFormat in_fmt = tt::DataFormat::Float16_b,
    tt::DataFormat out_fmt = tt::DataFormat::Float16_b,
    bool fp32_dest_acc_en = false) {
    bool pass = true;
    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const size_t in_tile_size = tt::tile_size(in_fmt);
    const size_t out_tile_size = tt::tile_size(out_fmt);
    const size_t in0_byte_size = M * K * in_tile_size;
    const size_t in1_byte_size = K * N * in_tile_size;
    const size_t out_byte_size = M * N * out_tile_size;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config_0{
        .device = device,
        .size = in0_byte_size,
        .page_size = in0_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_config_1{
        .device = device,
        .size = in1_byte_size,
        .page_size = in1_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};
    tt::tt_metal::InterleavedBufferConfig dram_config_out{
        .device = device,
        .size = out_byte_size,
        .page_size = out_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_config_0);
    auto input1_dram_buffer = CreateBuffer(dram_config_1);
    auto output_dram_buffer = CreateBuffer(dram_config_out);

    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(in0_byte_size, {{in0_cb_index, in_fmt}})
            .set_page_size(in0_cb_index, in_tile_size));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(in1_byte_size, {{in1_cb_index, in_fmt}})
            .set_page_size(in1_cb_index, in_tile_size));
    tt_metal::CreateCircularBuffer(
        program_,
        core,
        tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, out_fmt}})
            .set_page_size(out_cb_index, out_tile_size));

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_tile_compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = {in0_cb_index, in1_cb_index, out_cb_index, M * K, K * N, M * N, M, N, K}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus & Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    // Random U(-1, +1) stimulus for both formats; golden is the actual
    // matmul evaluated host-side over the post-quantization input floats.
    // Trivial stimulus is intentionally avoided because it masks structural
    // bugs (e.g. a wrong K-stride is invisible if every K-slice tile is
    // bit-identical) -- exactly the pattern that hid the multi_tile_compute
    // in1_index_c_offset bug for the BF16 path.
    std::vector<uint32_t> packed_input0;
    std::vector<uint32_t> packed_input1;
    std::vector<float> in0_floats;
    std::vector<float> in1_floats;
    std::vector<float> golden_floats;
    constexpr float rng = 1.0f;

    if (in_fmt == tt::DataFormat::Fp8_e4m3) {
        packed_input0 = generate_packed_uniform_random_vector<uint32_t, float8_e4m3>(
            float8_e4m3(-rng), float8_e4m3(+rng), tt::constants::TILE_HW * M * K, /*seed=*/0);
        packed_input1 = generate_packed_uniform_random_vector<uint32_t, float8_e4m3>(
            float8_e4m3(-rng), float8_e4m3(+rng), tt::constants::TILE_HW * K * N, /*seed=*/1);
        in0_floats = fp8_to_floats(packed_input0);
        in1_floats = fp8_to_floats(packed_input1);
    } else {  // bfloat16
        packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
            bfloat16(-rng), bfloat16(+rng), tt::constants::TILE_HW * M * K, /*seed=*/0);
        packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
            bfloat16(-rng), bfloat16(+rng), tt::constants::TILE_HW * K * N, /*seed=*/1);
        in0_floats = bf16_to_floats(packed_input0);
        in1_floats = bf16_to_floats(packed_input1);
    }
    golden_floats.assign(M * N * tt::constants::TILE_HW, 0.0f);
    for (uint32_t mt = 0; mt < M; mt++) {
        for (uint32_t nt = 0; nt < N; nt++) {
            const size_t out_tile_off = (mt * N + nt) * tt::constants::TILE_HW;
            for (uint32_t y = 0; y < tt::constants::TILE_HEIGHT; y++) {
                for (uint32_t x = 0; x < tt::constants::TILE_WIDTH; x++) {
                    float acc = 0.0f;
                    for (uint32_t kt = 0; kt < K; kt++) {
                        const size_t in0_tile_off = (mt * K + kt) * tt::constants::TILE_HW;
                        const size_t in1_tile_off = (kt * N + nt) * tt::constants::TILE_HW;
                        for (uint32_t z = 0; z < tt::constants::TILE_WIDTH; z++) {
                            acc += in0_floats[in0_tile_off + byte_tile_face_major_index(z, y)] *
                                   in1_floats[in1_tile_off + byte_tile_face_major_index(x, z)];
                        }
                    }
                    golden_floats[out_tile_off + byte_tile_face_major_index(x, y)] = acc;
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////
    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {(uint32_t)input0_dram_buffer->address(),
         (uint32_t)0,
         (uint32_t)input1_dram_buffer->address(),
         (uint32_t)0,
         (uint32_t)1,              // num_blocks
         (uint32_t)M * K,          // in0_block_tile_cnt
         (uint32_t)K * N,          // in1_block_tile_cnt
         (uint32_t)in0_byte_size,  // in0_block_size_bytes
         (uint32_t)in1_byte_size});
    tt_metal::SetRuntimeArgs(
        program_, writer_kernel, core, {(uint32_t)output_dram_buffer->address(), (uint32_t)0, (uint32_t)M * N});

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    const bool fp8_out = (out_fmt == tt::DataFormat::Fp8_e4m3);
    auto dest_floats = fp8_out ? fp8_to_floats(dest_buffer_data) : bf16_to_floats(dest_buffer_data);

    // Tolerances under random U(-1, +1) stimulus. FP8 output: ~1/8
    // quantization plus deeper accumulation rounding for K>1. BF16 output
    // (default fp32_dest_acc=false uses BF16 dest accumulation, so per-element
    // error grows with K * TILE_WIDTH inner-dim length) needs a few-percent
    // slack. PCC backstop catches structural mis-permutations that pointwise
    // tolerances would let slip.
    const float rtol = fp8_out ? 0.125f : 0.05f;
    const float atol = fp8_out ? ((K > 1) ? 0.25f : 0.125f) : ((K > 1) ? 0.4f : 0.2f);
    pass &= tt::test_utils::is_close_vectors<float>(dest_floats, golden_floats, rtol, atol);
    const double min_pcc = fp8_out ? ((K > 1) ? 0.98 : 0.99) : 0.99;
    pass &= check_pcc(dest_floats, golden_floats, min_pcc);
    return pass;
}
// blocked matmul has blocking on output, spill/reloads using intermediate
bool blocked_matmul(const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t M, uint32_t K, uint32_t N) {
    bool pass = true;
    // FIXME: Convert to config
    CoreCoord core(0, 0);
    const uint32_t in0_cb_index = 0;
    const uint32_t in1_cb_index = 1;
    const uint32_t out_cb_index = 16;
    const uint32_t partials_cb_index = 24;
    const size_t cb_page_size = 2 * tt::constants::TILE_HW;  // 2 bytes per bfloat16 element
    const size_t in0_byte_size = M * K * cb_page_size;
    const size_t in1_byte_size = K * N * cb_page_size;
    const size_t out_byte_size = M * N * cb_page_size;
    const size_t num_blocks = 1;
    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    auto& cq = mesh_device->mesh_command_queue();
    auto zero_coord = distributed::MeshCoordinate(0, 0);
    auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
    distributed::MeshWorkload workload;
    auto* device = mesh_device->get_devices()[0];

    tt::tt_metal::InterleavedBufferConfig dram_config_0{
        .device = device,
        .size = in0_byte_size,
        .page_size = in0_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_1{
        .device = device,
        .size = in1_byte_size,
        .page_size = in1_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt::tt_metal::InterleavedBufferConfig dram_config_out{
        .device = device,
        .size = out_byte_size,
        .page_size = out_byte_size,
        .buffer_type = tt::tt_metal::BufferType::DRAM};

    tt_metal::Program program = tt_metal::CreateProgram();
    workload.add_program(device_range, std::move(program));
    auto& program_ = workload.get_programs().at(device_range);

    auto input0_dram_buffer = CreateBuffer(dram_config_0);
    const uint32_t in0_dram_addr = input0_dram_buffer->address();
    auto input1_dram_buffer = CreateBuffer(dram_config_1);
    const uint32_t in1_dram_addr = input1_dram_buffer->address();
    auto output_dram_buffer = CreateBuffer(dram_config_out);
    const uint32_t out_dram_addr = output_dram_buffer->address();

    tt_metal::CircularBufferConfig l1_input0_cb_config = tt_metal::CircularBufferConfig(in0_byte_size, {{in0_cb_index, tt::DataFormat::Float16_b}})
        .set_page_size(in0_cb_index, cb_page_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_input0_cb_config);

    tt_metal::CircularBufferConfig l1_input1_cb_config =
        tt_metal::CircularBufferConfig(in1_byte_size, {{in1_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(in1_cb_index, cb_page_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_input1_cb_config);

    tt_metal::CircularBufferConfig l1_output_cb_config =
        tt_metal::CircularBufferConfig(out_byte_size, {{out_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(out_cb_index, cb_page_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_output_cb_config);

    tt_metal::CircularBufferConfig l1_partials_cb_config =
        tt_metal::CircularBufferConfig(out_byte_size, {{partials_cb_index, tt::DataFormat::Float16_b}})
            .set_page_size(partials_cb_index, cb_page_size);
    tt_metal::CreateCircularBuffer(program_, core, l1_partials_cb_config);

    auto reader_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/reader_binary_blocked.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt_metal::NOC::RISCV_1_default,
            .compile_args = {in0_cb_index, in1_cb_index}});

    auto writer_kernel = tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/writer_unary.cpp",
        core,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt_metal::NOC::RISCV_0_default,
            .compile_args = {out_cb_index}});

    tt_metal::CreateKernel(
        program_,
        "tests/tt_metal/tt_metal/test_kernels/compute/unit_tests/matmul/multi_block_compute.cpp",
        core,
        tt_metal::ComputeConfig{
            .compile_args = {
                in0_cb_index,
                in1_cb_index,
                out_cb_index,
                partials_cb_index,
                M * K,
                K * N,
                M * N,
                M,
                N,
                K,
                num_blocks}});

    ////////////////////////////////////////////////////////////////////////////
    //                      Stimulus Generation
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> packed_input0 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f, 1.0f, in0_byte_size / sizeof(bfloat16), std::chrono::system_clock::now().time_since_epoch().count());
    std::vector<uint32_t> packed_input1 = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        0.03125f,
        0.03125f,
        in1_byte_size / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());
    ////////////////////////////////////////////////////////////////////////////
    //                      Golden Generation
    ////////////////////////////////////////////////////////////////////////////
    auto packed_golden = generate_packed_uniform_random_vector<uint32_t, bfloat16>(
        1.0f * K,
        1.0f * K,
        (out_byte_size) / sizeof(bfloat16),
        std::chrono::system_clock::now().time_since_epoch().count());

    ////////////////////////////////////////////////////////////////////////////
    //                      Compile and Execute Application
    ////////////////////////////////////////////////////////////////////////////

    tt_metal::detail::WriteToBuffer(input0_dram_buffer, packed_input0);
    tt_metal::detail::WriteToBuffer(input1_dram_buffer, packed_input1);

    tt_metal::SetRuntimeArgs(
        program_,
        reader_kernel,
        core,
        {
            (uint32_t)in0_dram_addr,
            (uint32_t)0,
            (uint32_t)in1_dram_addr,
            (uint32_t)0,
            (uint32_t)1,              // num_blocks
            (uint32_t)M * K,          // in0_block_tile_cnt
            (uint32_t)K * N,          // in1_block_tile_cnt
            (uint32_t)in0_byte_size,  // in0_block_size_bytes
            (uint32_t)in1_byte_size,  // in1_block_size_bytes
        });
    tt_metal::SetRuntimeArgs(
        program_,
        writer_kernel,
        core,
        {
            (uint32_t)out_dram_addr,
            (uint32_t)0,
            (uint32_t)M * N,
        });

    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);
    ////////////////////////////////////////////////////////////////////////////
    //                      Comparison Checking
    ////////////////////////////////////////////////////////////////////////////
    std::vector<uint32_t> dest_buffer_data;
    tt_metal::detail::ReadFromBuffer(output_dram_buffer, dest_buffer_data);
    int failed_index;
    pass &= is_close_packed_vectors<bfloat16, uint32_t>(
        dest_buffer_data,
        packed_golden,
        [&](const bfloat16& a, const bfloat16& b) { return is_close(a, b, 0.015f); },
        &failed_index);
    if (not pass) {
        log_info(tt::LogTest, "Failed Index={}", failed_index);
        print_vector_fixed_numel_per_row(
            unpack_vector<bfloat16, uint32_t>(dest_buffer_data), tt::constants::TILE_WIDTH);
    }
    return pass;
}

}  // namespace unit_tests::compute::matmul

TEST_F(MeshDeviceFixture, TensixTestSingleCoreSingleTileComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(this->devices_.at(id)));
    }
}
TEST_F(MeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 1, 1, 1));
    }
}
TEST_F(MeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileAccumulationComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 1, 2, 1));
    }
}
TEST_F(MeshDeviceFixture, TensixTestSingleCoreSingleBlockSingleTileNoAccumulationComputeMatmul) {
    for (unsigned int id = 0; id < num_devices_; id++) {
        ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(this->devices_.at(id), 2, 1, 2));
    }
}

TEST_F(BlackholeSingleCardFixture, TensixTestSingleCoreSingleTileComputeMatmulFp8e4m3) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(
        this->devices_.at(0),
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/true));
}

TEST_F(BlackholeSingleCardFixture, TensixTestSingleCoreSingleTileComputeMatmulFp8e4m3OutBf16) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_tile_matmul(
        this->devices_.at(0),
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Float16_b,
        /*fp32_dest_acc_en=*/true));
}

TEST_F(BlackholeSingleCardFixture, TensixTestSingleCoreSingleBlockSingleTileComputeMatmulFp8e4m3) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(
        this->devices_.at(0),
        1,
        1,
        1,
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/true));
}

TEST_F(BlackholeSingleCardFixture, TensixTestSingleCoreSingleBlockSingleTileAccumulationComputeMatmulFp8e4m3) {
    ASSERT_TRUE(unit_tests::compute::matmul::single_block_matmul(
        this->devices_.at(0),
        1,
        2,
        1,
        tt::DataFormat::Fp8_e4m3,
        tt::DataFormat::Fp8_e4m3,
        /*fp32_dest_acc_en=*/true));
}

}  // namespace tt::tt_metal
