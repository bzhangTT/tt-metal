// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace tt::tt_metal {

/**
 * @brief Expected signature for validating that a telemetry buffer contains dispatch telemetry data.
 */
constexpr uint32_t DISPATCH_TELEMETRY_SIGNATURE = 0x12345678;
constexpr uint32_t DISPATCH_TELEMETRY_VERSION = 1;

/**
 * @brief Telemetry for prefetch.
 */
struct PrefetchTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = DISPATCH_TELEMETRY_SIGNATURE;
    uint64_t blocked_count = 0;
    uint64_t unblocked_count = 0;
    uint64_t command_count = 0;
};

/**
 * @brief Telemetry for dispatch.
 */
struct DispatchTelemetry {
    uint32_t version = DISPATCH_TELEMETRY_VERSION;
    uint32_t signature = DISPATCH_TELEMETRY_SIGNATURE;
    uint64_t blocked_count = 0;
    uint64_t unblocked_count = 0;
};

}  // namespace tt::tt_metal
