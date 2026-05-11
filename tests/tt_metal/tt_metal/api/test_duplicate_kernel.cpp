// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include "llrt/core_descriptor.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "compile_program_with_kernel_path_env_var_fixture.hpp"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/dispatch_core_common.hpp>
#include "mesh_dispatch_fixture.hpp"
#include <tracy/Tracy.hpp>
#include <tt-metalium/distributed.hpp>
#include "gtest/gtest.h"
#include <tt-metalium/program.hpp>
#include <umd/device/types/core_coordinates.hpp>
#include <umd/device/types/xy_pair.hpp>

namespace tt::tt_metal {

using namespace tt;

// Ensures we cannot create duplicate kernels
TEST_F(MeshDispatchFixture, TensixFailOnDuplicateKernelCreationDataflow) {
    ZoneScopedN("TensixFailOnDuplicateKernelCreationDataflow");
    for (const auto& device : this->devices_) {
        ZoneScopedN("TensixFailOnDuplicateKernelCreationDataflow device");
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = [&] {
            ZoneScopedN("Dataflow create program");
            return CreateProgram();
        }();
        {
            ZoneScopedN("Dataflow add program to workload");
            workload.add_program(device_range, std::move(program));
        }
        auto& program_ = [&]() -> tt_metal::Program& {
            ZoneScopedN("Dataflow get program from workload");
            return workload.get_programs().at(device_range);
        }();

        CoreCoord compute_grid = [&] {
            ZoneScopedN("Dataflow compute grid size");
            return device->compute_with_storage_grid_size();
        }();
        auto create_duplicate_dataflow_kernel = [&] {
            ZoneScopedN("Dataflow duplicate creation expect throw body");
            {
                ZoneScopedN("Dataflow create first dram_copy kernel");
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
            }
            {
                ZoneScopedN("Dataflow create duplicate dram_copy kernel");
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
            }
        };
        EXPECT_THROW(create_duplicate_dataflow_kernel(), std::exception);
    }
}

TEST_F(MeshDispatchFixture, TensixFailOnDuplicateKernelCreationCompute) {
    ZoneScopedN("TensixFailOnDuplicateKernelCreationCompute");
    for (const auto& device : this->devices_) {
        ZoneScopedN("TensixFailOnDuplicateKernelCreationCompute device");
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = [&] {
            ZoneScopedN("Compute create program");
            return CreateProgram();
        }();
        {
            ZoneScopedN("Compute add program to workload");
            workload.add_program(device_range, std::move(program));
        }
        auto& program_ = [&]() -> tt_metal::Program& {
            ZoneScopedN("Compute get program from workload");
            return workload.get_programs().at(device_range);
        }();

        CoreCoord compute_grid = [&] {
            ZoneScopedN("Compute compute grid size");
            return device->compute_with_storage_grid_size();
        }();
        std::vector<uint32_t> compute_kernel_args = {};
        auto create_duplicate_compute_kernel = [&] {
            ZoneScopedN("Compute duplicate creation expect throw body");
            {
                ZoneScopedN("Compute create broadcast kernel");
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/compute/broadcast.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    ComputeConfig{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false,
                        .compile_args = compute_kernel_args,
                        .opt_level = KernelBuildOptLevel::O3});
            }
            {
                ZoneScopedN("Compute create duplicate matmul kernel");
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    ComputeConfig{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false,
                        .compile_args = compute_kernel_args,
                        .opt_level = KernelBuildOptLevel::O3});
            }
        };
        EXPECT_THROW(create_duplicate_compute_kernel(), std::exception);
    }
}

TEST_F(MeshDispatchFixture, TensixPassOnNormalKernelCreation) {
    ZoneScopedN("TensixPassOnNormalKernelCreation");
    for ([[maybe_unused]] const auto& mesh_device : this->devices_) {
        ZoneScopedN("TensixPassOnNormalKernelCreation device");
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = [&] {
            ZoneScopedN("Normal create program");
            return CreateProgram();
        }();
        {
            ZoneScopedN("Normal add program to workload");
            workload.add_program(device_range, std::move(program));
        }
        auto& program_ = [&]() -> tt_metal::Program& {
            ZoneScopedN("Normal get program from workload");
            return workload.get_programs().at(device_range);
        }();
        std::vector<uint32_t> compute_kernel_args = {};
        auto create_normal_kernels = [&] {
            ZoneScopedN("Normal creation expect no throw body");
            {
                ZoneScopedN("Normal create broadcast kernel");
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/compute/broadcast.cpp",
                    CoreCoord(1, 0),
                    ComputeConfig{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false,
                        .compile_args = compute_kernel_args,
                        .opt_level = KernelBuildOptLevel::O3});
            }
            {
                ZoneScopedN("Normal create matmul kernel");
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
                    CoreCoord(0, 0),
                    ComputeConfig{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false,
                        .compile_args = compute_kernel_args,
                        .opt_level = KernelBuildOptLevel::O3});
            }
        };
        EXPECT_NO_THROW(create_normal_kernels());
    }
}

TEST_F(MeshDispatchFixture, TensixPassOnMixedOverlapKernelCreation) {
    ZoneScopedN("TensixPassOnMixedOverlapKernelCreation");
    for (const auto& device : this->devices_) {
        ZoneScopedN("TensixPassOnMixedOverlapKernelCreation device");
        distributed::MeshWorkload workload;
        auto zero_coord = distributed::MeshCoordinate(0, 0);
        auto device_range = distributed::MeshCoordinateRange(zero_coord, zero_coord);
        tt_metal::Program program = [&] {
            ZoneScopedN("Mixed overlap create program");
            return CreateProgram();
        }();
        {
            ZoneScopedN("Mixed overlap add program to workload");
            workload.add_program(device_range, std::move(program));
        }
        auto& program_ = [&]() -> tt_metal::Program& {
            ZoneScopedN("Mixed overlap get program from workload");
            return workload.get_programs().at(device_range);
        }();
        CoreCoord compute_grid = [&] {
            ZoneScopedN("Mixed overlap compute grid size");
            return device->compute_with_storage_grid_size();
        }();
        std::vector<uint32_t> compute_kernel_args = {};
        auto create_mixed_overlap_kernels = [&] {
            ZoneScopedN("Mixed overlap creation expect no throw body");
            {
                ZoneScopedN("Mixed overlap create dram_copy kernel");
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/dataflow/dram_copy.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    DataMovementConfig{
                        .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});
            }
            {
                ZoneScopedN("Mixed overlap create matmul kernel");
                tt_metal::CreateKernel(
                    program_,
                    "tests/tt_metal/tt_metal/test_kernels/compute/matmul.cpp",
                    CoreRange(CoreCoord(0, 0), CoreCoord(compute_grid.x, compute_grid.y)),
                    ComputeConfig{
                        .math_fidelity = MathFidelity::HiFi4,
                        .fp32_dest_acc_en = false,
                        .math_approx_mode = false,
                        .compile_args = compute_kernel_args,
                        .opt_level = KernelBuildOptLevel::O3});
            }
        };
        EXPECT_NO_THROW(create_mixed_overlap_kernels());
    }
}

}  // namespace tt::tt_metal
