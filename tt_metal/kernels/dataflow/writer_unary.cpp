// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "experimental/endpoints.h"

#ifdef ARCH_QUASAR
#include "experimental/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#else
#include "experimental/circular_buffer.h"
#endif

void kernel_main() {
#ifdef ARCH_QUASAR
    uint32_t dst_addr = get_arg(args::dst_addr);
    uint32_t bank_id = get_arg(args::bank_id);
    uint32_t num_tiles = get_arg(args::num_tiles);
#else
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t bank_id = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);
#endif

    experimental::Noc noc;
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t ublock_size_bytes;

    // single-tile ublocks
#ifdef ARCH_QUASAR
    experimental::DataflowBuffer buff_out(dfb::in);
    ublock_size_bytes = buff_out.get_entry_size() * ublock_size_tiles;
#else
    constexpr uint32_t cb_out_id = tt::CBIndex::c_16;
    experimental::CircularBuffer buff_out(cb_out_id);
    ublock_size_bytes = get_tile_size(cb_out_id) * ublock_size_tiles;
#endif

    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        buff_out.wait_front(ublock_size_tiles);
        noc.async_write(
            buff_out,
            experimental::AllocatorBank<experimental::AllocatorBankType::DRAM>{},
            ublock_size_bytes,
            {},
            {.bank_id = bank_id, .addr = dst_addr});
        noc.async_write_barrier();
        buff_out.pop_front(ublock_size_tiles);
        dst_addr += ublock_size_bytes;
    }
}
