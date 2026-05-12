// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::prim {

struct SdpaDecodeParams {
    bool is_causal = false;
    bool paged_attention = false;
    std::vector<uint32_t> cur_pos;
    std::optional<float> scale = std::nullopt;
    std::optional<uint32_t> sliding_window_size = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<ttnn::operations::transformer::SDPAProgramConfig> program_config = std::nullopt;
    DeviceComputeKernelConfig compute_kernel_config;
    uint32_t k_chunk_size = 0;
    // Share cache is only meaningful for some unpaged configurations; default is false.
    std::optional<bool> share_cache = std::nullopt;
    // When true, enables multi-latent attention (MLA) path where V is derived from K.
    std::optional<bool> use_mla = std::nullopt;
    std::optional<uint32_t> head_dim_v = std::nullopt;
    // Optional override of the per-block token capacity for paged attention.
    // When unset, the kernel derives ``block_size`` from
    // ``k.padded_shape[2]`` (the legacy path) and ``head_dim`` from
    // ``k.padded_shape[-1]``. When set, the caller is reinterpreting the
    // same physical K/V buffer with a different ``(block_size, head_dim)``
    // tile arrangement than the cache's declared shape — Q's last dim
    // then becomes the source of truth for ``head_dim``. Use case: vLLM's
    // hybrid kv-cache-groups manager equalises per-block bytes across
    // groups by adjusting ``block_size`` per group, leaving one physical
    // buffer shared across layers with different ``(block_size, head_dim)``
    // views. See the validation in ``validate_on_program_cache_miss`` for
    // the byte-count consistency invariant.
    std::optional<uint32_t> block_size_override = std::nullopt;
};

struct SdpaDecodeInputs {
    // Mandatory tensors
    Tensor q;
    Tensor k;

    // Optional V tensor; when MLA is enabled, V is derived from K and this may be nullopt.
    std::optional<Tensor> v;

    // Optional auxiliary tensors
    std::optional<Tensor> cur_pos_tensor;
    std::optional<Tensor> page_table_tensor;
    std::optional<Tensor> attn_mask;
    std::optional<Tensor> attention_sink;
};

}  // namespace ttnn::prim
