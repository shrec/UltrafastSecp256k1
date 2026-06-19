/**
 * lbtc_batch_demo.cpp — minimal skeleton showing how a node (libbitcoin) drives
 * the bridge: pack script-signature triples into one table, verify in one call,
 * read back per-row results, and map failures to the node's own ids.
 *
 * This is the integration shape the libbitcoin team builds their marshalling
 * around; we then link the GPU side and tune against the real IBD pipeline.
 */
#include "ufsecp_libbitcoin.h"
#include "ufsecp.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

int main() {
    /* Default C++ controller == UFSECP_LBTC_AUTO: GPU if usable, otherwise the
     * CPU consensus reference path. Identical results either way. */
    ufsecp::lbtc::Controller context;
    if (!context.ok()) {
        std::fprintf(stderr, "controller create failed\n");
        return 1;
    }
    const char* names[] = {"CPU", "CUDA", "OpenCL", "Metal"};
    std::printf("backend: %s (%s)\n", names[context.backend()],
                ufsecp_lbtc_ctrl_device_name(context.get()));

    /* --- build an ECDSA table: each row = 32 msg | 33 pubkey | 64 sig + 3-byte
     *     opaque block-id tag (the node's correlation key). --- */
    ufsecp_ctx* sctx = nullptr;
    ufsecp_ctx_create(&sctx);

    const size_t N = 8, KEY = 3;                  /* 3-byte block id per row    */
    const size_t stride = UFSECP_LBTC_ECDSA_RECORD + KEY;
    std::vector<uint8_t> rows(N * stride, 0);

    for (size_t i = 0; i < N; ++i) {
        uint8_t* r = rows.data() + i * stride;
        uint8_t sk[32] = {0}, msg[32] = {0}, pub[33];
        sk[31] = (uint8_t)(i + 1);
        msg[0] = (uint8_t)(0x10 + i);
        ufsecp_pubkey_create(sctx, sk, pub);
        std::memcpy(r, msg, 32);
        std::memcpy(r + 32, pub, 33);
        ufsecp_ecdsa_sign(sctx, msg, sk, r + 65);
        /* opaque block-id tag (little-endian 3 bytes) */
        const uint32_t block_id = 800000u + (uint32_t)i;
        r[UFSECP_LBTC_ECDSA_RECORD + 0] = (uint8_t)(block_id);
        r[UFSECP_LBTC_ECDSA_RECORD + 1] = (uint8_t)(block_id >> 8);
        r[UFSECP_LBTC_ECDSA_RECORD + 2] = (uint8_t)(block_id >> 16);
    }

    /* simulate a bad signature in row 5 */
    rows[5 * stride + 70] ^= 0x01;

    /* --- one call --- results[i] == 1 valid / 0 invalid is the only output.
     * Map failures back to block ids by scanning it (exactly how libbitcoin's
     * verify_signatures collects failed-token identifiers). --- */
    std::vector<uint8_t> results(N);
    context.verify_ecdsa(rows.data(), N, KEY, results.data());

    size_t ninvalid = 0;
    for (size_t i = 0; i < N; ++i) if (!results[i]) ++ninvalid;
    std::printf("invalid rows: %zu\n", ninvalid);
    for (size_t i = 0; i < N; ++i) {
        if (results[i]) continue;
        const uint8_t* tag = rows.data() + i * stride + UFSECP_LBTC_ECDSA_RECORD;
        const uint32_t block_id =
            (uint32_t)tag[0] | ((uint32_t)tag[1] << 8) | ((uint32_t)tag[2] << 16);
        std::printf("  row %zu invalid -> block %u\n", i, block_id);
    }

    ufsecp_ctx_destroy(sctx);
    return 0;
}
