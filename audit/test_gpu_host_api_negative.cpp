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
#include "ufsecp/ufsecp.h"

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (cond) { ++g_pass; }                                     \
        else { ++g_fail; std::printf("  FAIL: %s\n", msg); }       \
    } while (0)

/* "GPU runtime not actually usable" — used to broaden the accepted error
 * set for invalid-input batch ops. On real hardware the backend returns
 * UFSECP_OK with out_result[0]==0, UFSECP_ERR_GPU_UNSUPPORTED, or
 * UFSECP_ERR_BAD_INPUT. On GitHub-hosted macos-latest with software Metal
 * (no GPU acceleration), kernel dispatch / memory alloc / queue setup
 * intermittently fails with LAUNCH/MEMORY/BACKEND/QUEUE/DEVICE — the
 * security property "invalid input does not produce a verify-OK result"
 * still holds in those cases (no signature is forged). Treat them as
 * equivalent to UNSUPPORTED for the negative-input assertions. */
static inline bool gpu_runtime_unusable(int err) {
    return err == UFSECP_ERR_GPU_UNSUPPORTED ||
           err == UFSECP_ERR_GPU_LAUNCH      ||
           err == UFSECP_ERR_GPU_MEMORY      ||
           err == UFSECP_ERR_GPU_BACKEND     ||
           err == UFSECP_ERR_GPU_QUEUE       ||
           err == UFSECP_ERR_GPU_DEVICE;
}

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
    /* Post-condition: null-guard queries still work after destroy(NULL). */
    CHECK(ufsecp_gpu_last_error(nullptr) == UFSECP_ERR_NULL_ARG,
          "ctx_destroy(NULL) did not crash; null-guard queries still functional");
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
 * 4. Invalid content on core ops
 * ============================================================================ */
static void test_invalid_content_core_ops(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_negative] Invalid content on core ops\n");

    if (!ctx) {
        std::printf("  (skipped -- no GPU context)\n");
        return;
    }

    ufsecp_ctx* cpu_ctx = nullptr;
      CHECK(ufsecp_ctx_create(&cpu_ctx) == UFSECP_OK, "cpu ctx for core-op fixture setup");
    if (!cpu_ctx) {
        return;
    }

    uint8_t msg32[32] = {};
    uint8_t seckey32[32] = {};
    uint8_t aux32[32] = {};
    uint8_t valid_pub33[33] = {};
    uint8_t invalid_pub33[33] = {};
    uint8_t xonly_pub32[32] = {};
    uint8_t ecdsa_sig64[64] = {};
    uint8_t schnorr_sig64[64] = {};
    uint8_t out_result[1] = {1};
    uint8_t out_pub33[33] = {};
    uint8_t out_hash20[20] = {};
    uint8_t out_secret32[32] = {};

    msg32[0] = 0x42;
    seckey32[31] = 1;
    aux32[0] = 7;

    CHECK(ufsecp_pubkey_create(cpu_ctx, seckey32, valid_pub33) == UFSECP_OK,
          "cpu pubkey_create for core-op fixture setup");
    CHECK(ufsecp_pubkey_xonly(cpu_ctx, seckey32, xonly_pub32) == UFSECP_OK,
          "cpu pubkey_xonly for core-op fixture setup");
    CHECK(ufsecp_ecdsa_sign(cpu_ctx, msg32, seckey32, ecdsa_sig64) == UFSECP_OK,
          "cpu ecdsa_sign for core-op fixture setup");
    CHECK(ufsecp_schnorr_sign(cpu_ctx, msg32, seckey32, aux32, schnorr_sig64) == UFSECP_OK,
          "cpu schnorr_sign for core-op fixture setup");

    std::memcpy(invalid_pub33, valid_pub33, sizeof(valid_pub33));
    invalid_pub33[0] = 0x05;

      uint8_t sentinel_pub33[33];
      std::memset(sentinel_pub33, 0xA5, sizeof(sentinel_pub33));
      std::memcpy(out_pub33, sentinel_pub33, sizeof(out_pub33));
    auto e1 = ufsecp_gpu_generator_mul_batch(ctx, msg32, 1, out_pub33);
      bool const out_changed =
            std::memcmp(out_pub33, sentinel_pub33, sizeof(out_pub33)) != 0;
      CHECK(e1 != UFSECP_ERR_NULL_ARG,
              "generator_mul_batch zero scalar is handled as a real operation");
    CHECK(e1 != UFSECP_OK || out_changed,
          "generator_mul_batch success path writes an output buffer");

    out_result[0] = 1;
    auto e2 = ufsecp_gpu_ecdsa_verify_batch(ctx, msg32, invalid_pub33, ecdsa_sig64, 1, out_result);
    CHECK(gpu_runtime_unusable(e2) || e2 == UFSECP_ERR_BAD_INPUT
              || (e2 == UFSECP_OK && out_result[0] == 0),
          "ecdsa_verify_batch invalid pubkey: UNSUPPORTED, ERR_BAD_INPUT, or marks result 0");

    schnorr_sig64[0] ^= 0x80;
    out_result[0] = 1;
    auto e3 = ufsecp_gpu_schnorr_verify_batch(ctx, msg32, xonly_pub32, schnorr_sig64, 1, out_result);
    CHECK(gpu_runtime_unusable(e3) || e3 == UFSECP_ERR_BAD_INPUT
              || (e3 == UFSECP_OK && out_result[0] == 0),
          "schnorr_verify_batch invalid signature: UNSUPPORTED, ERR_BAD_INPUT, or marks result 0");
    schnorr_sig64[0] ^= 0x80;

    auto e4 = ufsecp_gpu_ecdh_batch(ctx, seckey32, invalid_pub33, 1, out_secret32);
    CHECK(gpu_runtime_unusable(e4) || e4 != UFSECP_OK,
          "ecdh_batch invalid peer pubkey rejects malformed input");

    auto e5 = ufsecp_gpu_hash160_pubkey_batch(ctx, invalid_pub33, 1, out_hash20);
    CHECK(gpu_runtime_unusable(e5) || e5 != UFSECP_OK,
          "hash160_pubkey_batch invalid compressed pubkey rejects malformed input");

    ufsecp_ctx_destroy(cpu_ctx);
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
    const uint32_t n = ufsecp_gpu_backend_count(ids, 4);
    uint32_t avail_id = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) { avail_id = ids[i]; break; }
    }

    if (avail_id == 0) {
        std::printf("  (skipped -- no GPU backend)\n");
        return;
    }

    const uint32_t dcount = ufsecp_gpu_device_count(avail_id);

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
 * 6. ecrecover zero-edge and invalid content
 * ============================================================================ */
static void test_ecrecover_zero_and_invalid(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_negative] ecrecover count=0 and invalid recid\n");

    if (!ctx) {
        std::printf("  (skipped -- no GPU context)\n");
        return;
    }

    CHECK(ufsecp_gpu_ecrecover_batch(ctx, nullptr, nullptr, nullptr, 0, nullptr, nullptr) == UFSECP_OK,
          "ecrecover_batch(count=0) = OK");

    ufsecp_ctx* cpu_ctx = nullptr;
      CHECK(ufsecp_ctx_create(&cpu_ctx) == UFSECP_OK, "cpu ctx for ecrecover fixture setup");
    if (!cpu_ctx) {
        return;
    }

    uint8_t msg32[32] = {};
    uint8_t seckey32[32] = {};
    uint8_t sig64[64] = {};
    uint8_t out_pub33[33] = {};
    uint8_t out_valid[1] = {1};
    int recid = 0;
    int invalid_recid = 9;

    msg32[0] = 0x33;
    seckey32[31] = 1;

    CHECK(ufsecp_ecdsa_sign_recoverable(cpu_ctx, msg32, seckey32, sig64, &recid) == UFSECP_OK,
          "cpu recoverable sign for ecrecover fixture setup");

    auto err = ufsecp_gpu_ecrecover_batch(ctx, msg32, sig64, &invalid_recid, 1, out_pub33, out_valid);
    CHECK(err == UFSECP_ERR_GPU_UNSUPPORTED || err == UFSECP_ERR_BAD_INPUT
              || (err == UFSECP_OK && out_valid[0] == 0),
          "ecrecover_batch invalid recid: UNSUPPORTED, ERR_BAD_INPUT, or marks valid[0]=0");

    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 7. Extended ops zero-edge and invalid content
 * ============================================================================ */
static void test_extended_ops_zero_and_invalid(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_negative] Extended ops count=0 and invalid content\n");

    if (!ctx) {
        std::printf("  (skipped -- no GPU context)\n");
        return;
    }

    uint8_t scalar32[32] = {};
    uint8_t compressed33[33] = {0x02};
    uint8_t invalid_compressed33[33] = {0x05};
    uint8_t proof64[64] = {};
    uint8_t point65[65] = {0x04};
    uint8_t invalid_point65[65] = {0x05};
    uint8_t proof324[324] = {};
    uint8_t out_result[1] = {1};
    uint8_t out_valid[1] = {1};
    uint8_t key32[32] = {};
    uint8_t nonce12[12] = {};
    uint8_t plaintext32[32] = {};
    uint8_t wire51[51] = {};
    uint8_t plain_out32[32] = {};
    uint32_t size_ok[1] = {4};
    uint32_t size_too_big[1] = {33};

    proof324[0] = 0x05;
    proof324[65] = 0x04;
    proof324[130] = 0x04;
    proof324[195] = 0x04;

      CHECK(ufsecp_gpu_frost_verify_partial_batch(ctx, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                                        nullptr, nullptr, nullptr, 0, nullptr) == UFSECP_OK,
          "frost_verify_partial_batch(count=0) = OK");
    CHECK(ufsecp_gpu_zk_knowledge_verify_batch(ctx, nullptr, nullptr, nullptr, 0, nullptr) == UFSECP_OK,
          "zk_knowledge_verify_batch(count=0) = OK");
    CHECK(ufsecp_gpu_zk_dleq_verify_batch(ctx, nullptr, nullptr, nullptr, nullptr, nullptr, 0, nullptr) == UFSECP_OK,
          "zk_dleq_verify_batch(count=0) = OK");
    CHECK(ufsecp_gpu_bulletproof_verify_batch(ctx, nullptr, nullptr, nullptr, 0, nullptr) == UFSECP_OK,
          "bulletproof_verify_batch(count=0) = OK");
    CHECK(ufsecp_gpu_bip324_aead_encrypt_batch(ctx, nullptr, nullptr, nullptr, nullptr, 32, 0, nullptr) == UFSECP_OK,
          "bip324_aead_encrypt_batch(count=0) = OK");
    CHECK(ufsecp_gpu_bip324_aead_decrypt_batch(ctx, nullptr, nullptr, nullptr, nullptr, 32, 0, nullptr, nullptr) == UFSECP_OK,
          "bip324_aead_decrypt_batch(count=0) = OK");

    auto e1 = ufsecp_gpu_frost_verify_partial_batch(ctx, scalar32, invalid_compressed33, compressed33,
                                                    compressed33, scalar32, scalar32, out_result,
                                                    out_result, 1, out_result);
    CHECK(gpu_runtime_unusable(e1) || e1 == UFSECP_ERR_BAD_INPUT
              || (e1 == UFSECP_OK && out_result[0] == 0),
          "frost_verify_partial_batch invalid point: UNSUPPORTED, ERR_BAD_INPUT, or marks 0");

    out_result[0] = 1;
    auto e2 = ufsecp_gpu_zk_knowledge_verify_batch(ctx, proof64, invalid_point65, scalar32, 1, out_result);
    CHECK(gpu_runtime_unusable(e2) || e2 == UFSECP_ERR_BAD_INPUT
              || (e2 == UFSECP_OK && out_result[0] == 0),
          "zk_knowledge_verify_batch invalid pubkey: UNSUPPORTED, ERR_BAD_INPUT, or marks 0");

    out_result[0] = 1;
    auto e3 = ufsecp_gpu_zk_dleq_verify_batch(ctx, proof64, invalid_point65, point65,
                                              point65, point65, 1, out_result);
    CHECK(gpu_runtime_unusable(e3) || e3 == UFSECP_ERR_BAD_INPUT
              || (e3 == UFSECP_OK && out_result[0] == 0),
          "zk_dleq_verify_batch invalid point: UNSUPPORTED, ERR_BAD_INPUT, or marks 0");

    out_result[0] = 1;
    auto e4 = ufsecp_gpu_bulletproof_verify_batch(ctx, proof324, point65, point65, 1, out_result);
    CHECK(gpu_runtime_unusable(e4) || e4 == UFSECP_ERR_BAD_INPUT
              || (e4 == UFSECP_OK && out_result[0] == 0),
          "bulletproof_verify_batch invalid point: UNSUPPORTED, ERR_BAD_INPUT, or marks 0");

    auto e5 = ufsecp_gpu_bip324_aead_encrypt_batch(ctx, key32, nonce12, plaintext32,
                                                   size_too_big, 32, 1, wire51);
    CHECK(gpu_runtime_unusable(e5) || e5 != UFSECP_OK,
          "bip324_aead_encrypt_batch invalid oversized packet rejects malformed input");

    out_valid[0] = 1;
    auto e6 = ufsecp_gpu_bip324_aead_decrypt_batch(ctx, key32, nonce12, wire51,
                                                   size_too_big, 32, 1, plain_out32, out_valid);
    CHECK(gpu_runtime_unusable(e6) || e6 == UFSECP_ERR_BAD_INPUT
              || (e6 == UFSECP_OK && out_valid[0] == 0),
          "bip324_aead_decrypt_batch oversized: UNSUPPORTED, ERR_BAD_INPUT, or marks invalid");

    out_valid[0] = 1;
    auto e7 = ufsecp_gpu_bip324_aead_decrypt_batch(ctx, key32, nonce12, wire51,
                                                   size_ok, 32, 1, plain_out32, out_valid);
    CHECK(gpu_runtime_unusable(e7) || e7 == UFSECP_ERR_BAD_INPUT
              || (e7 == UFSECP_OK && out_valid[0] == 0),
          "bip324_aead_decrypt_batch bad tag: UNSUPPORTED, ERR_BAD_INPUT, or marks invalid");

    /* ── schnorr_snark_witness_batch: zero_edge + invalid_content ──── */

    /* count=0 → OK (vacuous batch) */
    CHECK(ufsecp_gpu_zk_schnorr_snark_witness_batch(ctx, nullptr, nullptr,
                                                     nullptr, 0, nullptr) == UFSECP_OK,
          "schnorr_snark_witness_batch(count=0) = OK");

    /* Invalid content: all-zero inputs (bad R.x, bad P.x, zero s) */
    {
        uint8_t bad_msg[32] = {};
        uint8_t bad_pk[32] = {};
        uint8_t bad_sig[64] = {};
        uint8_t witness_out[512] = {};  /* oversized to be safe */
        auto ew = ufsecp_gpu_zk_schnorr_snark_witness_batch(
            ctx, bad_msg, bad_pk, bad_sig, 1, witness_out);
        /* Either UNSUPPORTED (no GPU kernel yet) or ERR_BAD_INPUT — never OK for
         * all-zero input. Also accept LAUNCH/MEMORY/etc. for software Metal. */
        CHECK(gpu_runtime_unusable(ew) || ew == UFSECP_ERR_BAD_INPUT,
              "schnorr_snark_witness_batch invalid content returns expected error code");
    }
}

/* ============================================================================
 * 8. Unsupported op behavior
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

      const int recid = 0;
      auto e7 = ufsecp_gpu_ecrecover_batch(ctx, buf, buf, &recid, 1, buf, buf + 64);
      if (e7 == UFSECP_ERR_GPU_UNSUPPORTED) ops_tested++;

    /* Verify that unsupported ops return a non-OK error code (not silent success). */
    CHECK(e1 != UFSECP_OK && e2 != UFSECP_OK && e3 != UFSECP_OK &&
          e4 != UFSECP_OK && e5 != UFSECP_OK && e6 != UFSECP_OK && e7 != UFSECP_OK,
          "Unsupported GPU ops all return non-OK error codes (not silent success)");
      std::printf("    (%d of 7 ops returned UNSUPPORTED on this backend)\n", ops_tested);
}

/* ============================================================================
 * 9. Error string completeness
 * ============================================================================ */
static void test_error_strings() {
    std::printf("[gpu_negative] Error strings\n");

      const char* unknown_fallback = "unknown error";

    /* All GPU error codes should have non-empty descriptions */
    const int gpu_codes[] = {
        UFSECP_ERR_GPU_UNAVAILABLE, UFSECP_ERR_GPU_DEVICE,
        UFSECP_ERR_GPU_LAUNCH, UFSECP_ERR_GPU_MEMORY,
        UFSECP_ERR_GPU_UNSUPPORTED, UFSECP_ERR_GPU_BACKEND,
        UFSECP_ERR_GPU_QUEUE
    };

    for (const int code : gpu_codes) {
        const char* str = ufsecp_gpu_error_str(code);
        char msg[128];
        (void)std::snprintf(msg, sizeof(msg), "error_str(%d) is non-empty", code);
        CHECK(str != nullptr && str[0] != '\0', msg);
        if (str != nullptr) {
            CHECK(std::strcmp(str, "unknown error") != 0, msg);
        }
    }

    /* Unknown code returns "unknown error" */
      CHECK(std::strcmp(ufsecp_gpu_error_str(999), unknown_fallback) == 0,
          "code 999 uses the fallback string");
}

/* ============================================================================
 * 10. Backend name edge cases
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
    const uint32_t n = ufsecp_gpu_backend_count(ids, 4);
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
      test_invalid_content_core_ops(ctx);
    test_invalid_device();
      test_ecrecover_zero_and_invalid(ctx);
      test_extended_ops_zero_and_invalid(ctx);
    test_unsupported_ops(ctx);

    if (ctx) ufsecp_gpu_ctx_destroy(ctx);

    std::printf("\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_gpu_host_api_negative_run(); }
#endif
