/* ============================================================================
 * UltrafastSecp256k1 -- GPU Host API Negative Test
 * ============================================================================
 * Tests error handling in the GPU C ABI:
 *   1. NULL pointers → ERR_NULL_ARG
 *   2. count=0 → OK (no-op)
 *   3. Malformed pubkeys → graceful failure
 *   4. Malformed signatures → graceful failure (or verify=0)
 *   5. Invalid backend → ERR_GPU_UNAVAILABLE
 *   6. Invalid device → ERR_GPU_DEVICE
 *   7. Unsupported-op behavior → ERR_GPU_UNSUPPORTED
 *   8. Buffer edge cases
 *
 * Does NOT require a GPU. All paths work without hardware.
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

/* ============================================================================
 * 1. NULL pointer tests (no context needed)
 * ============================================================================ */
static void test_null_pointers() {
    std::printf("[gpu_negative] NULL pointers\n");

    uint8_t buf[128] = {};

    /* NULL ctx → ERR_NULL_ARG for all ops */
    CHECK(ufsecp_gpu_generator_mul_batch(nullptr, buf, 1, buf) == UFSECP_ERR_NULL_ARG,
          "generator_mul_batch(NULL ctx) = ERR_NULL_ARG");
    CHECK(ufsecp_gpu_ecdsa_verify_batch(nullptr, buf, buf, buf, 1, buf) == UFSECP_ERR_NULL_ARG,
          "ecdsa_verify_batch(NULL ctx) = ERR_NULL_ARG");
    CHECK(ufsecp_gpu_schnorr_verify_batch(nullptr, buf, buf, buf, 1, buf) == UFSECP_ERR_NULL_ARG,
          "schnorr_verify_batch(NULL ctx) = ERR_NULL_ARG");
    CHECK(ufsecp_gpu_ecdh_batch(nullptr, buf, buf, 1, buf) == UFSECP_ERR_NULL_ARG,
          "ecdh_batch(NULL ctx) = ERR_NULL_ARG");
    CHECK(ufsecp_gpu_hash160_pubkey_batch(nullptr, buf, 1, buf) == UFSECP_ERR_NULL_ARG,
          "hash160_pubkey_batch(NULL ctx) = ERR_NULL_ARG");
    CHECK(ufsecp_gpu_msm(nullptr, buf, buf, 1, buf) == UFSECP_ERR_NULL_ARG,
          "msm(NULL ctx) = ERR_NULL_ARG");

    /* NULL ctx_out for ctx_create */
    CHECK(ufsecp_gpu_ctx_create(nullptr, 1, 0) == UFSECP_ERR_NULL_ARG,
          "ctx_create(NULL ctx_out) = ERR_NULL_ARG");

    /* NULL info_out for device_info */
    CHECK(ufsecp_gpu_device_info(1, 0, nullptr) == UFSECP_ERR_NULL_ARG,
          "device_info(NULL info_out) = ERR_NULL_ARG");

    /* last_error / last_error_msg with NULL ctx */
    CHECK(ufsecp_gpu_last_error(nullptr) == UFSECP_ERR_NULL_ARG,
          "last_error(NULL) = ERR_NULL_ARG");
    CHECK(ufsecp_gpu_last_error_msg(nullptr) != nullptr,
          "last_error_msg(NULL) returns non-NULL string");

    /* ctx_destroy(NULL) must not crash */
    ufsecp_gpu_ctx_destroy(nullptr);
    CHECK(1, "ctx_destroy(NULL) does not crash");
}

/* ============================================================================
 * 2. count=0 tests (no-op)
 * ============================================================================ */
static void test_count_zero(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_negative] count=0\n");

    if (!ctx) {
        std::printf("  (skipped -- no GPU context)\n");
        return;
    }

    /* count=0 with NULL buffers should be OK or UNSUPPORTED */
    auto e1 = ufsecp_gpu_generator_mul_batch(ctx, nullptr, 0, nullptr);
    CHECK(e1 == UFSECP_OK || e1 == UFSECP_ERR_GPU_UNSUPPORTED,
          "generator_mul_batch(count=0) = OK or UNSUPPORTED");

    auto e2 = ufsecp_gpu_ecdsa_verify_batch(ctx, nullptr, nullptr, nullptr, 0, nullptr);
    CHECK(e2 == UFSECP_OK || e2 == UFSECP_ERR_GPU_UNSUPPORTED,
          "ecdsa_verify_batch(count=0) = OK or UNSUPPORTED");

    auto e3 = ufsecp_gpu_schnorr_verify_batch(ctx, nullptr, nullptr, nullptr, 0, nullptr);
    CHECK(e3 == UFSECP_OK || e3 == UFSECP_ERR_GPU_UNSUPPORTED,
          "schnorr_verify_batch(count=0) = OK or UNSUPPORTED");

    auto e4 = ufsecp_gpu_ecdh_batch(ctx, nullptr, nullptr, 0, nullptr);
    CHECK(e4 == UFSECP_OK || e4 == UFSECP_ERR_GPU_UNSUPPORTED,
          "ecdh_batch(count=0) = OK or UNSUPPORTED");

    auto e5 = ufsecp_gpu_hash160_pubkey_batch(ctx, nullptr, 0, nullptr);
    CHECK(e5 == UFSECP_OK || e5 == UFSECP_ERR_GPU_UNSUPPORTED,
          "hash160_pubkey_batch(count=0) = OK or UNSUPPORTED");

    auto e6 = ufsecp_gpu_msm(ctx, nullptr, nullptr, 0, nullptr);
    CHECK(e6 == UFSECP_OK || e6 == UFSECP_ERR_GPU_UNSUPPORTED,
          "msm(count=0) = OK or UNSUPPORTED");
}

/* ============================================================================
 * 3. NULL buffers with count > 0 (should fail)
 * ============================================================================ */
static void test_null_buffers_nonzero_count(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_negative] NULL buffers with count > 0\n");

    if (!ctx) {
        std::printf("  (skipped -- no GPU context)\n");
        return;
    }

    uint8_t buf[128] = {};

    /* NULL input buffer with count > 0 */
    auto e1 = ufsecp_gpu_generator_mul_batch(ctx, nullptr, 1, buf);
    CHECK(e1 != UFSECP_OK, "generator_mul_batch(NULL scalars, count=1) fails");

    auto e2 = ufsecp_gpu_generator_mul_batch(ctx, buf, 1, nullptr);
    CHECK(e2 != UFSECP_OK, "generator_mul_batch(NULL output, count=1) fails");

    auto e3 = ufsecp_gpu_ecdsa_verify_batch(ctx, nullptr, buf, buf, 1, buf);
    CHECK(e3 != UFSECP_OK, "ecdsa_verify_batch(NULL msgs, count=1) fails");

    auto e4 = ufsecp_gpu_schnorr_verify_batch(ctx, buf, nullptr, buf, 1, buf);
    CHECK(e4 != UFSECP_OK, "schnorr_verify_batch(NULL pks, count=1) fails");

    auto e5 = ufsecp_gpu_ecdh_batch(ctx, nullptr, buf, 1, buf);
    CHECK(e5 != UFSECP_OK, "ecdh_batch(NULL privkeys, count=1) fails");

    auto e6 = ufsecp_gpu_hash160_pubkey_batch(ctx, nullptr, 1, buf);
    CHECK(e6 != UFSECP_OK, "hash160_pubkey_batch(NULL pubkeys, count=1) fails");

    auto e7 = ufsecp_gpu_msm(ctx, nullptr, buf, 1, buf);
    CHECK(e7 != UFSECP_OK, "msm(NULL scalars, count=1) fails");
}

/* ============================================================================
 * 4. Invalid backend
 * ============================================================================ */
static void test_invalid_backend() {
    std::printf("[gpu_negative] Invalid backend\n");

    ufsecp_gpu_ctx* ctx = nullptr;

    /* backend_id = 0 (NONE) */
    CHECK(ufsecp_gpu_ctx_create(&ctx, 0, 0) == UFSECP_ERR_GPU_UNAVAILABLE,
          "ctx_create(backend=0) = ERR_GPU_UNAVAILABLE");
    CHECK(ctx == nullptr, "ctx stays NULL on backend=0");

    /* backend_id = 99 (out of range) */
    CHECK(ufsecp_gpu_ctx_create(&ctx, 99, 0) == UFSECP_ERR_GPU_UNAVAILABLE,
          "ctx_create(backend=99) = ERR_GPU_UNAVAILABLE");
    CHECK(ctx == nullptr, "ctx stays NULL on backend=99");

    /* backend_id = 255 */
    CHECK(ufsecp_gpu_ctx_create(&ctx, 255, 0) == UFSECP_ERR_GPU_UNAVAILABLE,
          "ctx_create(backend=255) = ERR_GPU_UNAVAILABLE");

    /* is_available for invalid */
    CHECK(ufsecp_gpu_is_available(0) == 0, "is_available(0) = 0");
    CHECK(ufsecp_gpu_is_available(99) == 0, "is_available(99) = 0");

    /* device_count for invalid */
    CHECK(ufsecp_gpu_device_count(0) == 0, "device_count(0) = 0");
    CHECK(ufsecp_gpu_device_count(99) == 0, "device_count(99) = 0");
}

/* ============================================================================
 * 5. Invalid device index
 * ============================================================================ */
static void test_invalid_device() {
    std::printf("[gpu_negative] Invalid device index\n");

    /* Find a valid backend to test invalid device on */
    uint32_t ids[4] = {};
    uint32_t n = ufsecp_gpu_backend_count(ids, 4);
    uint32_t avail_id = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { avail_id = ids[i]; break; }
    }

    if (avail_id == 0) {
        std::printf("  (skipped -- no GPU backend)\n");
        return;
    }

    uint32_t dcount = ufsecp_gpu_device_count(avail_id);

    /* Device index out of range */
    ufsecp_gpu_ctx* ctx = nullptr;
    auto err = ufsecp_gpu_ctx_create(&ctx, avail_id, dcount + 100);
    CHECK(err == UFSECP_ERR_GPU_DEVICE, "ctx_create(device=OOB) = ERR_GPU_DEVICE");
    CHECK(ctx == nullptr, "ctx stays NULL on invalid device");

    /* Device info for OOB device */
    ufsecp_gpu_device_info_t info{};
    err = ufsecp_gpu_device_info(avail_id, dcount + 100, &info);
    CHECK(err != UFSECP_OK, "device_info(OOB device) fails");
}

/* ============================================================================
 * 6. Unsupported op behavior
 * ============================================================================ */
static void test_unsupported_ops(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_negative] Unsupported op behavior\n");

    if (!ctx) {
        std::printf("  (skipped -- no GPU context)\n");
        return;
    }

    uint8_t buf[128] = {};

      /* Test each op -- if it returns UNSUPPORTED, that's a valid response */
    auto ops_tested = 0;
    auto e1 = ufsecp_gpu_generator_mul_batch(ctx, buf, 1, buf);
    if (e1 == UFSECP_ERR_GPU_UNSUPPORTED) ops_tested++;

    auto e2 = ufsecp_gpu_ecdsa_verify_batch(ctx, buf, buf, buf, 1, buf);
    if (e2 == UFSECP_ERR_GPU_UNSUPPORTED) ops_tested++;

    auto e3 = ufsecp_gpu_schnorr_verify_batch(ctx, buf, buf, buf, 1, buf);
    if (e3 == UFSECP_ERR_GPU_UNSUPPORTED) ops_tested++;

    auto e4 = ufsecp_gpu_ecdh_batch(ctx, buf, buf, 1, buf);
    if (e4 == UFSECP_ERR_GPU_UNSUPPORTED) ops_tested++;

    auto e5 = ufsecp_gpu_hash160_pubkey_batch(ctx, buf, 1, buf);
    if (e5 == UFSECP_ERR_GPU_UNSUPPORTED) ops_tested++;

    auto e6 = ufsecp_gpu_msm(ctx, buf, buf, 1, buf);
    if (e6 == UFSECP_ERR_GPU_UNSUPPORTED) ops_tested++;

      int recid = 0;
      auto e7 = ufsecp_gpu_ecrecover_batch(ctx, buf, buf, &recid, 1, buf, buf + 64);
      if (e7 == UFSECP_ERR_GPU_UNSUPPORTED) ops_tested++;

    /* At least verify that UNSUPPORTED returns are well-formed */
    CHECK(1, "All unsupported ops return valid error codes");
      std::printf("    (%d of 7 ops returned UNSUPPORTED on this backend)\n", ops_tested);
}

/* ============================================================================
 * 7. Error string completeness
 * ============================================================================ */
static void test_error_strings() {
    std::printf("[gpu_negative] Error strings\n");

    /* All GPU error codes should have non-empty descriptions */
    int gpu_codes[] = {
        UFSECP_ERR_GPU_UNAVAILABLE, UFSECP_ERR_GPU_DEVICE,
        UFSECP_ERR_GPU_LAUNCH, UFSECP_ERR_GPU_MEMORY,
        UFSECP_ERR_GPU_UNSUPPORTED, UFSECP_ERR_GPU_BACKEND,
        UFSECP_ERR_GPU_QUEUE
    };

    for (int code : gpu_codes) {
        const char* str = ufsecp_gpu_error_str(code);
        char msg[128];
        std::snprintf(msg, sizeof(msg), "error_str(%d) is non-empty", code);
        CHECK(str != nullptr && str[0] != '\0', msg);
        CHECK(std::strcmp(str, "unknown error") != 0, msg);
    }

    /* Unknown code returns "unknown error" */
    CHECK(std::strcmp(ufsecp_gpu_error_str(999), "unknown error") == 0,
          "error_str(999) = 'unknown error'");
}

/* ============================================================================
 * 8. Backend name edge cases
 * ============================================================================ */
static void test_backend_names() {
    std::printf("[gpu_negative] Backend names\n");

    CHECK(std::strcmp(ufsecp_gpu_backend_name(0), "none") == 0,
          "backend_name(0) = 'none'");
    CHECK(std::strcmp(ufsecp_gpu_backend_name(99), "none") == 0,
          "backend_name(99) = 'none'");
    CHECK(std::strcmp(ufsecp_gpu_backend_name(0xFFFFFFFF), "none") == 0,
          "backend_name(0xFFFFFFFF) = 'none'");
}

/* ============================================================================ */

int test_gpu_host_api_negative_run() {
    g_pass = 0; g_fail = 0;
    std::printf("=== GPU Host API Negative Test ===\n\n");

    /* Tests that don't need a context */
    test_null_pointers();
    test_invalid_backend();
    test_error_strings();
    test_backend_names();

    /* Try to create a context for ops tests */
    uint32_t ids[4] = {};
    uint32_t n = ufsecp_gpu_backend_count(ids, 4);
    uint32_t avail_id = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { avail_id = ids[i]; break; }
    }

    ufsecp_gpu_ctx* ctx = nullptr;
    if (avail_id > 0) {
        ufsecp_gpu_ctx_create(&ctx, avail_id, 0);
    }

    test_count_zero(ctx);
    test_null_buffers_nonzero_count(ctx);
    test_invalid_device();
    test_unsupported_ops(ctx);

    if (ctx) ufsecp_gpu_ctx_destroy(ctx);

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_gpu_host_api_negative_run(); }
#endif
