/* ============================================================================
 * UltrafastSecp256k1 -- GPU Backend Matrix Test
 * ============================================================================
 * Enumerates all compiled backends and reports per-backend:
 *   1. Backend availability and device discovery
 *   2. Per-backend available ops (probe each with small input)
 *   3. Expected UNSUPPORTED returns
 *   4. Device info sanity checks
 *
 * Always passes (report-style). Outputs structured summary.
 * ============================================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "ufsecp/ufsecp_gpu.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (cond) { ++g_pass; }                                     \
        else { ++g_fail; std::printf("  FAIL: %s\n", msg); }       \
    } while (0)

/* Probe whether an op is supported on a given context */
static const char* probe_status(ufsecp_error_t err) {
    if (err == UFSECP_OK) return "implemented";
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) return "unsupported";
    return "error";
}

static void probe_ops(ufsecp_gpu_ctx* ctx, const char* backend_name) {
    /* Use small dummy buffers (content doesn't matter for probing) */
    uint8_t scalar[32] = {};
    scalar[31] = 1; /* scalar = 1 */
    uint8_t pub33[33] = {0x02};
    uint8_t sig64[64] = {};
    uint8_t hash32[32] = {};
    uint8_t out[128] = {};
    int recid = 0;

    auto e1 = ufsecp_gpu_generator_mul_batch(ctx, scalar, 1, out);
    auto e2 = ufsecp_gpu_ecdsa_verify_batch(ctx, hash32, pub33, sig64, 1, out);
    auto e3 = ufsecp_gpu_schnorr_verify_batch(ctx, hash32, hash32, sig64, 1, out);
    auto e4 = ufsecp_gpu_ecdh_batch(ctx, scalar, pub33, 1, out);
    auto e5 = ufsecp_gpu_hash160_pubkey_batch(ctx, pub33, 1, out);
    auto e6 = ufsecp_gpu_msm(ctx, scalar, pub33, 1, out);
    auto e7 = ufsecp_gpu_ecrecover_batch(ctx, hash32, sig64, &recid, 1, out, out + 64);
    // Issue #335: multi-spend-key BIP-352 scan. scalar (=1) doubles as a
    // dummy scan private key; pub33 doubles as both the single spend and
    // single tweak candidate -- content doesn't matter for probing.
    auto e8 = ufsecp_gpu_bip352_scan_batch_multispend(
        ctx, scalar, pub33, 1, pub33, 1, reinterpret_cast<uint64_t*>(out));

    std::printf("  +----------------------------------------+\n");
    std::printf("  | %-12s | %-24s |\n", "Operation", "Status");
    std::printf("  +----------------------------------------+\n");
    std::printf("  | %-12s | %-24s |\n", "gen_mul",     probe_status(e1));
    std::printf("  | %-12s | %-24s |\n", "ecdsa_vrfy",  probe_status(e2));
    std::printf("  | %-12s | %-24s |\n", "schnorr_vrfy",probe_status(e3));
    std::printf("  | %-12s | %-24s |\n", "ecdh",        probe_status(e4));
    std::printf("  | %-12s | %-24s |\n", "hash160",     probe_status(e5));
    std::printf("  | %-12s | %-24s |\n", "msm",         probe_status(e6));
    std::printf("  | %-12s | %-24s |\n", "ecrecover",   probe_status(e7));
    std::printf("  | %-12s | %-24s |\n", "bip352_multi",probe_status(e8));
    std::printf("  +----------------------------------------+\n");
}

static void test_backend_enumeration() {
    std::printf("[gpu_matrix] Backend enumeration\n");

    uint32_t ids[4] = {};
    uint32_t count = ufsecp_gpu_backend_count(ids, 4);
    CHECK(count <= 3, "backend_count <= 3");

    std::printf("  Compiled backends: %u\n", count);
    for (uint32_t i = 0; i < count; ++i) {
        std::printf("    [%u] %s (id=%u, available=%d, devices=%u)\n",
                    i,
                    ufsecp_gpu_backend_name(ids[i]),
                    ids[i],
                    ufsecp_gpu_is_available(ids[i]),
                    ufsecp_gpu_device_count(ids[i]));
    }
}

static void test_device_info_sanity() {
    std::printf("[gpu_matrix] Device info sanity\n");

    uint32_t ids[4] = {};
    uint32_t n = ufsecp_gpu_backend_count(ids, 4);

    for (uint32_t i = 0; i < n; ++i) {
        uint32_t dcount = ufsecp_gpu_device_count(ids[i]);
        for (uint32_t d = 0; d < dcount; ++d) {
            ufsecp_gpu_device_info_t info{};
            auto err = ufsecp_gpu_device_info(ids[i], d, &info);
            if (err == UFSECP_OK) {
                char msg[256];
                std::snprintf(msg, sizeof(msg),
                    "%s device %u: name non-empty", ufsecp_gpu_backend_name(ids[i]), d);
                CHECK(info.name[0] != '\0', msg);

                std::snprintf(msg, sizeof(msg),
                    "%s device %u: backend_id match", ufsecp_gpu_backend_name(ids[i]), d);
                CHECK(info.backend_id == ids[i], msg);

                std::snprintf(msg, sizeof(msg),
                    "%s device %u: device_index match", ufsecp_gpu_backend_name(ids[i]), d);
                CHECK(info.device_index == d, msg);

                std::printf("    %s [%u]: %s (%lu MB, %u CUs, %u MHz)\n",
                    ufsecp_gpu_backend_name(ids[i]), d, info.name,
                    (unsigned long)(info.global_mem_bytes / (1024*1024)),
                    info.compute_units, info.max_clock_mhz);
            }
        }
    }
}

static void test_per_backend_ops() {
    std::printf("[gpu_matrix] Per-backend op support\n");

    uint32_t ids[4] = {};
    uint32_t n = ufsecp_gpu_backend_count(ids, 4);

    for (uint32_t i = 0; i < n; ++i) {
        if (!ufsecp_gpu_is_available(ids[i])) {
            std::printf("  %s: unavailable (no devices)\n",
                        ufsecp_gpu_backend_name(ids[i]));
            continue;
        }

        std::printf("  %s backend:\n", ufsecp_gpu_backend_name(ids[i]));

        ufsecp_gpu_ctx* ctx = nullptr;
        auto err = ufsecp_gpu_ctx_create(&ctx, ids[i], 0);
        if (err != UFSECP_OK || !ctx) {
            std::printf("    (could not create context: %s)\n",
                        ufsecp_gpu_error_str(err));
            continue;
        }

        probe_ops(ctx, ufsecp_gpu_backend_name(ids[i]));
        ufsecp_gpu_ctx_destroy(ctx);
    }
}

int main() {
    std::printf("=== GPU Backend Matrix Test ===\n\n");

    test_backend_enumeration();
    test_device_info_sanity();
    test_per_backend_ops();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
