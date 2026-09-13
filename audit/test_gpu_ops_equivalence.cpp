/* ============================================================================
 * UltrafastSecp256k1 -- GPU Ops Equivalence Test
 * ============================================================================
 * For each first-wave GPU operation, compares GPU result against CPU reference.
 * Only tests ops that the current backend actually supports (not UNSUPPORTED).
 *
 * Ops tested (where supported):
 *   1. generator_mul_batch   -- k*G
 *   2. ecdsa_verify_batch    -- ECDSA verification
 *   3. schnorr_verify_batch  -- BIP-340 Schnorr verification
 *   4. ecdh_batch            -- ECDH shared secret
 *   5. hash160_pubkey_batch  -- RIPEMD160(SHA256(pubkey))
 *   6. msm                   -- multi-scalar multiplication
 *
 * Requires at least one GPU backend with devices. Skips gracefully otherwise.
 * ============================================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <array>
#include <vector>

#include "ufsecp/ufsecp_gpu.h"
#include "ufsecp/ufsecp.h"

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (cond) { ++g_pass; }                                     \
        else { ++g_fail; std::printf("  FAIL: %s\n", msg); }       \
    } while (0)

#define CHECK_OK(expr, msg) CHECK((expr) == UFSECP_OK, msg)

#define SKIP(msg)                                                   \
    do { ++g_skip; std::printf("  SKIP: %s\n", msg); } while (0)

/* Deterministic pseudo-random bytes for test vectors */
static void fill_deterministic(uint8_t* buf, size_t len, uint8_t seed) {
    uint32_t state = seed;
    for (size_t i = 0; i < len; ++i) {
        state = state * 1103515245u + 12345u;
        buf[i] = static_cast<uint8_t>((state >> 16) & 0xFF);
    }
}

/* ============================================================================
 * 1. generator_mul_batch equivalence
 * ============================================================================ */
static void test_generator_mul_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] generator_mul_batch\n");

    constexpr size_t N = 8;
    uint8_t scalars[N * 32];
    fill_deterministic(scalars, sizeof(scalars), 0x42);
    /* Ensure scalars are valid (set high byte < 0xFF to stay in range) */
    for (size_t i = 0; i < N; ++i) scalars[i * 32] &= 0x7F;

    uint8_t gpu_pubs[N * 33];
    auto err = ufsecp_gpu_generator_mul_batch(ctx, scalars, N, gpu_pubs);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("generator_mul_batch unsupported"); return; }
    CHECK(err == UFSECP_OK, "GPU generator_mul_batch succeeds");
    if (err != UFSECP_OK) return;

    /* CPU reference */
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    for (size_t i = 0; i < N; ++i) {
        uint8_t cpu_pub[33];
        auto cerr = ufsecp_pubkey_create(cpu_ctx, scalars + i * 32, cpu_pub);
        CHECK(cerr == UFSECP_OK, "CPU pubkey_create succeeds");
        if (cerr == UFSECP_OK) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "generator_mul[%zu] GPU == CPU", i);
            CHECK(std::memcmp(gpu_pubs + i * 33, cpu_pub, 33) == 0, msg);
        }
    }
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 2. ecdsa_verify_batch equivalence
 * ============================================================================ */
static void test_ecdsa_verify_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] ecdsa_verify_batch\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t msg_hashes[N * 32];
    uint8_t privkeys[N * 32];
    uint8_t pubkeys[N * 33];
    uint8_t sigs[N * 64];

    fill_deterministic(msg_hashes, sizeof(msg_hashes), 0xAA);
    fill_deterministic(privkeys, sizeof(privkeys), 0xBB);

    /* Generate keys and sign with CPU */
    for (size_t i = 0; i < N; ++i) {
        privkeys[i * 32] &= 0x7F;
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, privkeys + i * 32, pubkeys + i * 33),
                 "CPU pubkey_create succeeds");

        CHECK_OK(ufsecp_ecdsa_sign(cpu_ctx, msg_hashes + i * 32, privkeys + i * 32,
                 sigs + i * 64),
                 "CPU ecdsa_sign succeeds");
    }

    /* GPU verify */
    uint8_t gpu_results[N];
    auto err = ufsecp_gpu_ecdsa_verify_batch(ctx, msg_hashes, pubkeys, sigs, N, gpu_results);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("ecdsa_verify_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU ecdsa_verify_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    /* CPU verify for comparison */
    for (size_t i = 0; i < N; ++i) {
        auto cerr = ufsecp_ecdsa_verify(cpu_ctx, msg_hashes + i * 32,
                                         sigs + i * 64, pubkeys + i * 33);
        uint8_t cpu_ok = (cerr == UFSECP_OK) ? 1 : 0;
        char msg[128];
        std::snprintf(msg, sizeof(msg), "ecdsa_verify[%zu] GPU(%d) == CPU(%d)", i,
                      gpu_results[i], cpu_ok);
        CHECK(gpu_results[i] == cpu_ok, msg);
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 3. schnorr_verify_batch equivalence
 * ============================================================================ */
static void test_schnorr_verify_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] schnorr_verify_batch\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t msg_hashes[N * 32];
    uint8_t privkeys[N * 32];
    uint8_t pubkeys_x[N * 32];
    uint8_t sigs[N * 64];

    fill_deterministic(msg_hashes, sizeof(msg_hashes), 0xCC);
    fill_deterministic(privkeys, sizeof(privkeys), 0xDD);

    /* Generate keypairs and sign with CPU */
    for (size_t i = 0; i < N; ++i) {
        privkeys[i * 32] &= 0x7F;
        uint8_t xonly[32];
        CHECK_OK(ufsecp_pubkey_xonly(cpu_ctx, privkeys + i * 32, xonly),
                 "CPU pubkey_xonly succeeds");
        std::memcpy(pubkeys_x + i * 32, xonly, 32);

        uint8_t aux[32] = {};
        CHECK_OK(ufsecp_schnorr_sign(cpu_ctx, msg_hashes + i * 32, privkeys + i * 32,
                 aux, sigs + i * 64),
                 "CPU schnorr_sign succeeds");
    }

    /* GPU verify */
    uint8_t gpu_results[N];
    auto err = ufsecp_gpu_schnorr_verify_batch(ctx, msg_hashes, pubkeys_x, sigs, N, gpu_results);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("schnorr_verify_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU schnorr_verify_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    for (size_t i = 0; i < N; ++i) {
        auto cerr = ufsecp_schnorr_verify(cpu_ctx, msg_hashes + i * 32,
                                           sigs + i * 64, pubkeys_x + i * 32);
        uint8_t cpu_ok = (cerr == UFSECP_OK) ? 1 : 0;
        char msg[128];
        std::snprintf(msg, sizeof(msg), "schnorr_verify[%zu] GPU(%d) == CPU(%d)", i,
                      gpu_results[i], cpu_ok);
        CHECK(gpu_results[i] == cpu_ok, msg);
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 4. ecdh_batch equivalence
 * ============================================================================ */
static void test_ecdh_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] ecdh_batch\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t alice_keys[N * 32];
    uint8_t bob_keys[N * 32];
    uint8_t bob_pubs[N * 33];

    fill_deterministic(alice_keys, sizeof(alice_keys), 0xA1);
    fill_deterministic(bob_keys, sizeof(bob_keys), 0xB2);

    for (size_t i = 0; i < N; ++i) {
        alice_keys[i * 32] &= 0x7F;
        bob_keys[i * 32] &= 0x7F;
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, bob_keys + i * 32, bob_pubs + i * 33),
                 "CPU pubkey_create succeeds");
    }

    /* GPU ECDH */
    uint8_t gpu_secrets[N * 32];
    auto err = ufsecp_gpu_ecdh_batch(ctx, alice_keys, bob_pubs, N, gpu_secrets);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("ecdh_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU ecdh_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    /* CPU ECDH for comparison */
    for (size_t i = 0; i < N; ++i) {
        uint8_t cpu_secret[32];
        auto cerr = ufsecp_ecdh(cpu_ctx, alice_keys + i * 32,
                                bob_pubs + i * 33, cpu_secret);
        CHECK(cerr == UFSECP_OK, "CPU ecdh succeeds");
        if (cerr == UFSECP_OK) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "ecdh[%zu] GPU == CPU", i);
            CHECK(std::memcmp(gpu_secrets + i * 32, cpu_secret, 32) == 0, msg);
        }
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 5. hash160_pubkey_batch equivalence
 * ============================================================================ */
static void test_hash160_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] hash160_pubkey_batch\n");

    constexpr size_t N = 8;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t privkeys[N * 32];
    uint8_t pubkeys[N * 33];

    fill_deterministic(privkeys, sizeof(privkeys), 0xE1);
    for (size_t i = 0; i < N; ++i) {
        privkeys[i * 32] &= 0x7F;
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, privkeys + i * 32, pubkeys + i * 33),
                 "CPU pubkey_create succeeds");
    }

    /* GPU hash160 */
    uint8_t gpu_hashes[N * 20];
    auto err = ufsecp_gpu_hash160_pubkey_batch(ctx, pubkeys, N, gpu_hashes);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("hash160_pubkey_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU hash160_pubkey_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    /* CPU hash160 for comparison */
    for (size_t i = 0; i < N; ++i) {
        uint8_t cpu_hash[20];
        auto cerr = ufsecp_hash160(pubkeys + i * 33, 33, cpu_hash);
        CHECK(cerr == UFSECP_OK, "CPU hash160 succeeds");
        if (cerr == UFSECP_OK) {
            char msg[128];
            std::snprintf(msg, sizeof(msg), "hash160[%zu] GPU == CPU", i);
            CHECK(std::memcmp(gpu_hashes + i * 20, cpu_hash, 20) == 0, msg);
        }
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 6. ecrecover_batch equivalence
 * ============================================================================ */
static void test_ecrecover_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] ecrecover_batch\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    uint8_t msg_hashes[N * 32];
    uint8_t privkeys[N * 32];
    uint8_t sigs[N * 64];
    int recids[N] = {};
    uint8_t expected_pubs[N * 33];
    ufsecp_error_t err = UFSECP_OK;

    fill_deterministic(msg_hashes, sizeof(msg_hashes), 0x91);
    fill_deterministic(privkeys, sizeof(privkeys), 0xA7);

    for (size_t i = 0; i < N; ++i) {
        privkeys[i * 32] &= 0x7F;
        auto perr = ufsecp_pubkey_create(cpu_ctx, privkeys + i * 32, expected_pubs + i * 33);
        CHECK(perr == UFSECP_OK, "CPU pubkey_create succeeds");
        if (perr != UFSECP_OK) goto cleanup;

        auto serr = ufsecp_ecdsa_sign_recoverable(cpu_ctx,
                                                  msg_hashes + i * 32,
                                                  privkeys + i * 32,
                                                  sigs + i * 64,
                                                  &recids[i]);
        CHECK(serr == UFSECP_OK, "CPU ecdsa_sign_recoverable succeeds");
        if (serr != UFSECP_OK) goto cleanup;
    }

    uint8_t gpu_pubs[N * 33];
    uint8_t gpu_valid[N];
    err = ufsecp_gpu_ecrecover_batch(ctx, msg_hashes, sigs, recids, N, gpu_pubs, gpu_valid);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("ecrecover_batch unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU ecrecover_batch succeeds");
    if (err != UFSECP_OK) goto cleanup;

    for (size_t i = 0; i < N; ++i) {
        uint8_t cpu_pub[33];
        auto rerr = ufsecp_ecdsa_recover(cpu_ctx,
                                         msg_hashes + i * 32,
                                         sigs + i * 64,
                                         recids[i],
                                         cpu_pub);
        CHECK(rerr == UFSECP_OK, "CPU ecdsa_recover succeeds");
        if (rerr != UFSECP_OK) continue;

        char msg_valid[128];
        std::snprintf(msg_valid, sizeof(msg_valid), "ecrecover[%zu] valid flag set", i);
        CHECK(gpu_valid[i] == 1, msg_valid);

        char msg_gpu_cpu[128];
        std::snprintf(msg_gpu_cpu, sizeof(msg_gpu_cpu), "ecrecover[%zu] GPU == CPU recover", i);
        CHECK(std::memcmp(gpu_pubs + i * 33, cpu_pub, 33) == 0, msg_gpu_cpu);

        char msg_expected[128];
        std::snprintf(msg_expected, sizeof(msg_expected), "ecrecover[%zu] recovered pubkey matches signer", i);
        CHECK(std::memcmp(gpu_pubs + i * 33, expected_pubs + i * 33, 33) == 0, msg_expected);
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 7. msm equivalence
 * ============================================================================ */
static void test_msm_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] msm\n");

    constexpr size_t N = 4;
    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");

    /* Use known points: k[i]*G for small k values */
    uint8_t base_scalars[N * 32] = {};
    uint8_t points[N * 33];
    uint8_t msm_scalars[N * 32];

    /* base_scalars = {1, 2, 3, 4} → points = {G, 2G, 3G, 4G} */
    for (size_t i = 0; i < N; ++i) {
        base_scalars[i * 32 + 31] = static_cast<uint8_t>(i + 1);
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, base_scalars + i * 32, points + i * 33),
                 "CPU pubkey_create succeeds");
    }

    /* msm_scalars = {5, 6, 7, 8} */
    for (size_t i = 0; i < N; ++i) {
        std::memset(msm_scalars + i * 32, 0, 32);
        msm_scalars[i * 32 + 31] = static_cast<uint8_t>(i + 5);
    }

    /* GPU MSM: sum = 5*G + 6*2G + 7*3G + 8*4G = (5+12+21+32)*G = 70*G */
    uint8_t gpu_result[33];
    auto err = ufsecp_gpu_msm(ctx, msm_scalars, points, N, gpu_result);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) { SKIP("msm unsupported"); goto cleanup; }
    CHECK(err == UFSECP_OK, "GPU msm succeeds");
    if (err != UFSECP_OK) goto cleanup;

    /* CPU reference: 70*G */
    {
        uint8_t scalar_70[32] = {};
        scalar_70[31] = 70;
        uint8_t cpu_pub[33];
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, scalar_70, cpu_pub),
                 "CPU pubkey_create for MSM reference succeeds");
        CHECK(std::memcmp(gpu_result, cpu_pub, 33) == 0, "msm result == 70*G (CPU reference)");
    }

cleanup:
    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================
 * 8. msm equivalence at a batch large enough to reach the multi-level tail
 * ============================================================================ */
static void test_msm_large_n_equiv(ufsecp_gpu_ctx* ctx) {
    std::printf("[gpu_equiv] msm (large N, multi-level tail reduce)\n");

    /* The CUDA backend block-reduces the scatter output and then finishes on a
     * single thread, so every point left to that finisher costs a serial point
     * addition. msm() therefore keeps block-reducing while the tail is longer
     * than 32, and N here is picked to make that loop actually run: 70000 points
     * give 274 blocks, one pass takes 274 down to 2, and the finisher sees 2 --
     * covering the loop body, the ping-pong buffer swap, and a finisher count
     * that is neither the old full block count nor 1.
     *
     * A SECOND loop pass would need N > 2M, far too slow for a CPU-reference MSM
     * in an audit test, and it re-runs the same code the first pass does.
     * test_msm_equiv (N=4) covers the other side: a tail short enough that the
     * loop must not run at all. */
    constexpr size_t N = 70000;
    constexpr size_t kDistinctPoints = 256;

    ufsecp_ctx* cpu_ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&cpu_ctx), "CPU ctx_create succeeds");
    if (!cpu_ctx) return;

    /* Points repeat on a 256-entry cycle: the reduction tree is indifferent to
     * which point sits where, and generating 70000 distinct pubkeys on the CPU
     * would dominate the test's runtime for no added coverage. */
    std::vector<uint8_t> base(kDistinctPoints * 33);
    for (size_t i = 0; i < kDistinctPoints; ++i) {
        uint8_t sk[32] = {};
        fill_deterministic(sk, 32, static_cast<uint8_t>(i + 1));
        sk[0] = 0;                        /* stay well below the group order */
        sk[31] = static_cast<uint8_t>(sk[31] | 1u);   /* and non-zero */
        CHECK_OK(ufsecp_pubkey_create(cpu_ctx, sk, base.data() + i * 33),
                 "CPU pubkey_create succeeds");
    }

    std::vector<uint8_t> points(N * 33);
    std::vector<uint8_t> scalars(N * 32);
    for (size_t i = 0; i < N; ++i) {
        std::memcpy(points.data() + i * 33,
                    base.data() + (i % kDistinctPoints) * 33, 33);
        uint8_t* s = scalars.data() + i * 32;
        fill_deterministic(s, 32, static_cast<uint8_t>(i & 0xFF));
        s[0] = 0;                                   /* < group order */
        s[1] = static_cast<uint8_t>(i >> 8);        /* all N scalars distinct */
        s[31] = static_cast<uint8_t>(s[31] | 1u);   /* non-zero */
    }

    uint8_t gpu_result[33] = {};
    auto err = ufsecp_gpu_msm(ctx, scalars.data(), points.data(), N, gpu_result);
    if (err == UFSECP_ERR_GPU_UNSUPPORTED) {
        SKIP("msm unsupported");
        ufsecp_ctx_destroy(cpu_ctx);
        return;
    }
    CHECK(err == UFSECP_OK, "GPU msm (N=70000) succeeds");

    uint8_t cpu_result[33] = {};
    auto cpu_err = ufsecp_multi_scalar_mul(cpu_ctx, scalars.data(), points.data(),
                                           N, cpu_result);
    CHECK(cpu_err == UFSECP_OK, "CPU multi_scalar_mul (N=70000) succeeds");

    if (err == UFSECP_OK && cpu_err == UFSECP_OK) {
        CHECK(std::memcmp(gpu_result, cpu_result, 33) == 0,
              "GPU msm == CPU multi_scalar_mul at N=70000");
    }

    ufsecp_ctx_destroy(cpu_ctx);
}

/* ============================================================================ */

int main() {
    std::printf("=== GPU Ops Equivalence Test ===\n\n");

    /* Find first available backend */
    uint32_t ids[4] = {};
    uint32_t n = ufsecp_gpu_backend_count(ids, 4);
    uint32_t avail_id = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (ufsecp_gpu_is_available(ids[i])) {
            avail_id = ids[i];
            break;
        }
    }

    if (avail_id == 0) {
        std::printf("  No GPU backend available -- skipping all equivalence tests.\n");
        std::printf("\n=== Results: 0 passed, 0 failed, 0 skipped (no GPU) ===\n");
        return 0;
    }

    std::printf("  Using backend: %s\n\n", ufsecp_gpu_backend_name(avail_id));

    ufsecp_gpu_ctx* ctx = nullptr;
    auto err = ufsecp_gpu_ctx_create(&ctx, avail_id, 0);
    if (err != UFSECP_OK || !ctx) {
        std::printf("  FAIL: could not create GPU context (err=%d)\n", err);
        return 1;
    }

    test_generator_mul_equiv(ctx);
    test_ecdsa_verify_equiv(ctx);
    test_schnorr_verify_equiv(ctx);
    test_ecdh_equiv(ctx);
    test_hash160_equiv(ctx);
    test_ecrecover_equiv(ctx);
    test_msm_equiv(ctx);
    test_msm_large_n_equiv(ctx);

    ufsecp_gpu_ctx_destroy(ctx);

    std::printf("\n=== Results: %d passed, %d failed, %d skipped ===\n",
                g_pass, g_fail, g_skip);
    return g_fail > 0 ? 1 : 0;
}
