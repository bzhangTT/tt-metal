// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/experimental/mock_allocator.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt_stl/assert.hpp>

#include "impl/allocator/l1_banking_allocator.hpp"
#include "impl/context/metal_context.hpp"
#include "tt_metal/distributed/mesh_device_impl.hpp"

namespace tt::tt_metal::experimental {

class MockAllocator : public L1BankingAllocator {
public:
    using L1BankingAllocator::L1BankingAllocator;
};

std::unique_ptr<AllocatorImpl> make_mock_allocator(const AllocatorConfig& config) {
    return std::make_unique<MockAllocator>(config);
}

// AllocatorImpl has no virtual destructor, so a static pointer-set registry would silently
// leak entries when MockAllocators are destroyed (slicing skips ~MockAllocator()), and the
// next L1BankingAllocator at a reused heap address would false-positive as mock. Instead,
// derive mockness from the device's MetalContext cluster type — that's address-stable and
// matches how Device::initialize_allocator() picked MockAllocator in the first place.
MockAllocator* get_mock_allocator(distributed::MeshDevice* device) {
    if (device == nullptr) {
        return nullptr;
    }
    auto context_id = device->impl().get_context_id();
    if (!MetalContext::instance_exists(context_id)) {
        return nullptr;
    }
    if (MetalContext::instance(context_id).get_cluster().get_target_device_type() != tt::TargetDevice::Mock) {
        return nullptr;
    }
    return static_cast<MockAllocator*>(device->allocator_impl().get());
}

AllocatorState extract_mock_allocator_state(distributed::MeshDevice* device) {
    auto* mock = get_mock_allocator(device);
    TT_FATAL(mock != nullptr, "extract_mock_allocator_state requires a mock device");
    return mock->extract_state();
}

void override_mock_allocator_state(distributed::MeshDevice* device, const AllocatorState& state) {
    auto* mock = get_mock_allocator(device);
    TT_FATAL(mock != nullptr, "override_mock_allocator_state requires a mock device");
    mock->override_state(state);
}

}  // namespace tt::tt_metal::experimental
