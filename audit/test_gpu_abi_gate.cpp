/* ============================================================================
 * UltrafastSecp256k1 -- GPU ABI Gate Test
 * ============================================================================
 * Validates the GPU C ABI surface:
 *   1. Backend discovery (ufsecp_gpu_backend_count, device_count, etc.)
 *   2. Context lifecycle (create, destroy, error tracking)
 *   3. Negative cases (NULL args, invalid backend, bad device index)
 *   4. Unsupported op returns correct error code
 *   5. If a real GPU is available: generator_mul_batch equivalence vs CPU
 *
 * This test DOES NOT require a GPU. All negative / discovery paths work
 * without hardware. GPU-specific ops are tested only when available.
 * ============================================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#include "ufsecp/ufsecp_gpu.h"
#include "ufsecp/ufsecp.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (cond) { ++g_pass; }                                     \
        else { ++g_fail; std::printf("  FAIL: %s\n", msg); }       \
    } while (0)

/* ============================================================================ */

static void test_backend_discovery() {
    std::printf("[gpu_abi_gate] Backend discovery\n");

    const uint32_t count = ufsecp_gpu_backend_count(nullptr, 0);
    CHECK(count <= 3, "backend_count <= 3 (max: CUDA + OpenCL + Metal)");

    /* Backend name for valid IDs */
    CHECK(std::strcmp(ufsecp_gpu_backend_name(0), "none") == 0,
          "backend_name(0) == 'none'");
    CHECK(std::strcmp(ufsecp_gpu_backend_name(1), "CUDA") == 0,
          "backend_name(1) == 'CUDA'");
    CHECK(std::strcmp(ufsecp_gpu_backend_name(2), "OpenCL") == 0,
          "backend_name(2) == 'OpenCL'");
    CHECK(std::strcmp(ufsecp_gpu_backend_name(3), "Metal") == 0,
          "backend_name(3) == 'Metal'");
    CHECK(std::strcmp(ufsecp_gpu_backend_name(99), "none") == 0,
          "backend_name(99) == 'none'");

    /* List backend IDs */
    if (count > 0) {
        uint32_t ids[4] = {};
        const uint32_t n = ufsecp_gpu_backend_count(ids, 4);
        CHECK(n == count, "backend_count with ids returns same count");
        for (uint32_t i = 0; i < n; ++i) {
            CHECK(ids[i] >= 1 && ids[i] <= 3,
                  "backend id in range [1,3]");
        }
    }

    /* is_available for non-existent backend */
    CHECK(ufsecp_gpu_is_available(0) == 0, "is_available(NONE) == 0");
    CHECK(ufsecp_gpu_is_available(99) == 0, "is_available(99) == 0");

    /* device_count for non-existent backend */
    CHECK(ufsecp_gpu_device_count(99) == 0, "device_count(99) == 0");
}

static void test_device_info() {
    std::printf("[gpu_abi_gate] Device info\n");

    /* Invalid backend */
    ufsecp_gpu_device_info_t info{};
    CHECK(ufsecp_gpu_device_info(99, 0, &info) != UFSECP_OK,
          "device_info(99, 0) fails");

    /* NULL info_out */
    CHECK(ufsecp_gpu_device_info(1, 0, nullptr) == UFSECP_ERR_NULL_ARG,
          "device_info NULL info_out returns ERR_NULL_ARG");

    /* If we have any backend, try querying device 0 */
    uint32_t ids[4] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 4);
    for (uint32_t i = 0; i < n; ++i) {
        const uint32_t dcount = ufsecp_gpu_device_count(ids[i]);
        if (dcount > 0) {
            ufsecp_gpu_device_info_t di{};
            auto err = ufsecp_gpu_device_info(ids[i], 0, &di);
            CHECK(err == UFSECP_OK, "device_info succeeds for available device");
            CHECK(di.name[0] != '\0', "device name is non-empty");
            CHECK(di.backend_id == ids[i], "device backend_id matches");
            CHECK(di.device_index == 0, "device_index == 0");
            std::printf("    Device: %s (mem=%lu MB, CUs=%u, %u MHz)\n",
                        di.name,
                        (unsigned long)(di.global_mem_bytes / (1024ULL * 1024ULL)),
                        di.compute_units, di.max_clock_mhz);
        }
    }
}

static void test_context_lifecycle() {
    std::printf("[gpu_abi_gate] Context lifecycle\n");

    /* NULL ctx_out */
    CHECK(ufsecp_gpu_ctx_create(nullptr, 1, 0) == UFSECP_ERR_NULL_ARG,
          "ctx_create(NULL) returns ERR_NULL_ARG");

    /* Invalid backend */
    ufsecp_gpu_ctx* ctx = nullptr;
    CHECK(ufsecp_gpu_ctx_create(&ctx, 99, 0) == UFSECP_ERR_GPU_UNAVAILABLE,
          "ctx_create(99) returns ERR_GPU_UNAVAILABLE");
    CHECK(ctx == nullptr, "ctx stays NULL on failure");

    /* Invalid backend_id=0 */
    CHECK(ufsecp_gpu_ctx_create(&ctx, 0, 0) == UFSECP_ERR_GPU_UNAVAILABLE,
          "ctx_create(NONE) returns ERR_GPU_UNAVAILABLE");

    /* Destroy NULL is safe */
    ufsecp_gpu_ctx_destroy(nullptr); /* should not crash */
    CHECK(1, "ctx_destroy(NULL) does not crash");

    /* is_ready: NULL guard */
    CHECK(ufsecp_gpu_is_ready(nullptr) == 0, "is_ready(NULL) == 0");
    /* ctx is still nullptr here after all failed creates above */
    CHECK(ufsecp_gpu_is_ready(ctx) == 0, "is_ready(NULL ctx after failed create) == 0");

    /* Error queries on NULL */
    CHECK(ufsecp_gpu_last_error(nullptr) == UFSECP_ERR_NULL_ARG,
          "last_error(NULL) returns ERR_NULL_ARG");
    CHECK(std::strcmp(ufsecp_gpu_last_error_msg(nullptr), "NULL GPU context") == 0,
          "last_error_msg(NULL) returns expected string");
}

static void test_null_buffer_ops() {
    std::printf("[gpu_abi_gate] NULL buffer operations\n");

    /* All batch ops with NULL ctx should return ERR_NULL_ARG */
    uint8_t dummy[64] = {};
    CHECK(ufsecp_gpu_generator_mul_batch(nullptr, dummy, 1, dummy) == UFSECP_ERR_NULL_ARG,
          "generator_mul_batch(NULL ctx)");
    CHECK(ufsecp_gpu_ecdsa_verify_batch(nullptr, dummy, dummy, dummy, 1, dummy) == UFSECP_ERR_NULL_ARG,
          "ecdsa_verify_batch(NULL ctx)");
    CHECK(ufsecp_gpu_schnorr_verify_batch(nullptr, dummy, dummy, dummy, 1, dummy) == UFSECP_ERR_NULL_ARG,
          "schnorr_verify_batch(NULL ctx)");
    CHECK(ufsecp_gpu_ecdh_batch(nullptr, dummy, dummy, 1, dummy) == UFSECP_ERR_NULL_ARG,
          "ecdh_batch(NULL ctx)");
    CHECK(ufsecp_gpu_hash160_pubkey_batch(nullptr, dummy, 1, dummy) == UFSECP_ERR_NULL_ARG,
          "hash160_pubkey_batch(NULL ctx)");
    CHECK(ufsecp_gpu_msm(nullptr, dummy, dummy, 1, dummy) == UFSECP_ERR_NULL_ARG,
          "msm(NULL ctx)");
}

static void test_error_strings() {
    std::printf("[gpu_abi_gate] Error strings\n");

      const char* unknown_fallback = "unknown error";

    CHECK(std::strcmp(ufsecp_gpu_error_str(UFSECP_OK), "OK") == 0,
          "error_str(OK)");
    CHECK(std::strcmp(ufsecp_gpu_error_str(UFSECP_ERR_GPU_UNAVAILABLE),
                      "GPU backend unavailable") == 0,
          "error_str(GPU_UNAVAILABLE)");
    CHECK(std::strcmp(ufsecp_gpu_error_str(UFSECP_ERR_GPU_UNSUPPORTED),
                      "operation not supported on this GPU backend") == 0,
          "error_str(GPU_UNSUPPORTED)");
      CHECK(std::strcmp(ufsecp_gpu_error_str(999), unknown_fallback) == 0,
          "code 999 uses the fallback string");
}

static void test_gpu_ops_if_available() {
    std::printf("[gpu_abi_gate] GPU ops (if available)\n");

    /* Find first available backend */
    uint32_t ids[4] = {};
    const uint32_t n = ufsecp_gpu_backend_count(ids, 4);
    uint32_t avail_id = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) {
            avail_id = ids[i];
            break;
        }
    }

    if (avail_id == 0) {
        std::printf("  (no GPU available -- skipping ops tests)\n");
        return;
    }

    std::printf("  Using backend: %s\n", ufsecp_gpu_backend_name(avail_id));

    ufsecp_gpu_ctx* ctx = nullptr;
    auto err = ufsecp_gpu_ctx_create(&ctx, avail_id, 0);
    CHECK(err == UFSECP_OK, "ctx_create succeeds");
    if (err != UFSECP_OK || !ctx) return;

    /* is_ready: valid ctx → 1 (smoke) */
    CHECK(ufsecp_gpu_is_ready(ctx) == 1, "is_ready(valid ctx) == 1 (smoke)");

    /* Test generator_mul_batch with known test vector:
       scalar = 1 → result = generator G
       G compressed = 02 79BE667E F9DCBBAC 55A06295 CE870B07 029BFCDB 2DCE28D9 59F2815B 16F81798 */
    {
        uint8_t scalar_one[32] = {};
        scalar_one[31] = 1;
        uint8_t pubkey[33] = {};
        err = ufsecp_gpu_generator_mul_batch(ctx, scalar_one, 1, pubkey);

        if (err == UFSECP_OK) {
            /* Verify against known generator point */
            static const uint8_t gen_compressed[33] = {
                0x02,
                0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
                0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
                0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
                0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
            };
            CHECK(std::memcmp(pubkey, gen_compressed, 33) == 0,
                  "1*G == generator (compressed)");
        } else if (err == UFSECP_ERR_GPU_UNSUPPORTED) {
            std::printf("  (generator_mul_batch not supported on this backend)\n");
        } else {
            CHECK(0, "generator_mul_batch unexpected error");
            std::printf("    error: %d (%s)\n", err, ufsecp_gpu_error_str(err));
        }
    }

    /* Test count=0 is a no-op */
    {
        err = ufsecp_gpu_generator_mul_batch(ctx, nullptr, 0, nullptr);
        CHECK(err == UFSECP_OK || err == UFSECP_ERR_GPU_UNSUPPORTED,
              "generator_mul_batch(count=0) is OK or UNSUPPORTED");
    }

    /* Test NULL buffer with non-zero count */
    {
        uint8_t out[33] = {};
        err = ufsecp_gpu_generator_mul_batch(ctx, nullptr, 1, out);
        CHECK(err != UFSECP_OK, "generator_mul_batch(NULL scalars) fails");
    }

    ufsecp_gpu_ctx_destroy(ctx);
    CHECK(1, "ctx_destroy succeeds");
}

int test_gpu_abi_gate_run() {
    g_pass = 0; g_fail = 0;
    std::printf("=== GPU ABI Gate Test ===\n\n");

    test_backend_discovery();
    test_device_info();
    test_context_lifecycle();
    test_null_buffer_ops();
    test_error_strings();
    test_gpu_ops_if_available();

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_gpu_abi_gate_run(); }
#endif
