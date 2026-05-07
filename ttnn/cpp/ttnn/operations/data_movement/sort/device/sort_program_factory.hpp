// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "sort_device_operation_types.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/device_operation.hpp"

#include <cstdint>

namespace ttnn::prim {
using namespace tt::tt_metal;

// Single row - single core
struct SortProgramFactorySingleRowSingleCore {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors);
};

// SortProgramFactoryCrossCoreDataExchange - single row, multi core with processing multiple tiles on one core with
// cross core data exchange
struct SortProgramFactoryCrossCoreDataExchange {
    // Per-program op-owned tensors that must survive across cache hits.  The framework
    // visits Tensor fields here via reflection so their buffers participate in
    // BufferBinding patching alongside tensor_args / tensor_return_value.
    //
    // Wrapped in std::optional because Tensor has no default constructor and the
    // framework default-initialises resource_t before prepare_resources populates it.
    struct Resources {
        std::optional<Tensor> physical_core_lookup_table_tensor;
    };

    static Resources prepare_resources(
        const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors);

    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SortParams& attributes,
        const SortInputs& tensor_args,
        std::vector<Tensor>& output_tensors,
        Resources& resources);

    /**
     * @brief Strategies for slicing work across cores in cross-core data exchange sort.
     */
    enum class CrossCoreDataExchangeSortSlicingStrategy : uint8_t {
        USE_AS_MANY_CORES,  ///< Use all available cores to process the same line, optimizing for latency.
        FILL_CORES_FIRST,   ///< Fill cores sequentially before assigning additional work.
    };

    static uint32_t get_number_of_tiles_per_core(
        uint32_t total_number_of_cores,
        uint32_t Wt,
        const DataType& input_dtype,
        const DataType& index_dtype,
        CrossCoreDataExchangeSortSlicingStrategy slicing_strategy =
            CrossCoreDataExchangeSortSlicingStrategy::USE_AS_MANY_CORES);

    static uint32_t rounddown_pow2(uint32_t n);
};

// Single row - multi core
struct SortProgramFactorySingleRowMultiCore {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const SortParams& attributes, const SortInputs& tensor_args, std::vector<Tensor>& output_tensors);
};

}  // namespace ttnn::prim
