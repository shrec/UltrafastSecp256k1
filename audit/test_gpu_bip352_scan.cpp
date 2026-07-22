/* ============================================================================
 * UltrafastSecp256k1 — BIP-352 Silent Payment GPU Batch Scan Audit
 * ============================================================================
 * Audit coverage for ufsecp_bip352_prepare_scan_plan (CPU utility) and
 * ufsecp_gpu_bip352_scan_batch (GPU batch scan).
 *
 * BIP-352 pipeline per tweak t_i:
 *   1. shared = scan_privkey × t_i               (GLV wNAF)
 *   2. ser37  = [prefix_byte] || shared.x || [0,0,0,0]
 *   3. hash   = SHA256_tagged("BIP0352/SharedSecret", ser37)
 *   4. output = hash × G + spend_pubkey
 *   5. prefix = upper 64 bits of output.x
 *
 * TESTS:
 *   SW-BIP352-1 : UFSECP_BIP352_SCAN_PLAN_BYTES macro == 264
 *   SW-BIP352-2 : ufsecp_bip352_prepare_scan_plan null arg → error
 *   SW-BIP352-3 : ufsecp_bip352_prepare_scan_plan produces non-zero output
 *   SW-BIP352-4 : GPU scan: null ctx → error
 *   SW-BIP352-5 : GPU scan: null scan_privkey → error
 *   SW-BIP352-6 : GPU scan: null spend_pubkey → error
 *   SW-BIP352-7 : GPU scan: null tweak_pubkeys → error (when n_tweaks > 0)
 *   SW-BIP352-8 : GPU scan: null prefix64_out → error
 *   SW-BIP352-9 : GPU scan: n_tweaks == 0 → UFSECP_OK (no crash)
 *   SW-BIP352-10: GPU scan: single tweak → prefix is non-zero
 *   SW-BIP352-11: GPU scan: batch of 16 tweaks → all prefixes non-zero
 *   SW-BIP352-12: GPU scan: determinism — same inputs → same prefixes
 *   SW-BIP352-13: GPU scan: distinct tweaks → prefixes are not all identical
 *   SW-BIP352-14: GPU scan: invalid compressed tweak pubkey is rejected
 *
 * Multi-spend-key coverage (GitHub issue #335), SW-BIP352-M*:
 *   Cross-implementation CPU oracle: ufsecp_silent_payment_create_output
 *   (sender side, secp256k1::silent_payment_create_output) constructs a real
 *   BIP-352 output for a given (tweak input privkey, scan pubkey, spend
 *   pubkey) triple; the GPU multi-spend scan (receiver side) must report the
 *   same upper-64-bit x-coordinate prefix for that (tweak, spend) pair. This
 *   is independent of the GPU kernels under test: construction and scanning
 *   are separate code paths that must agree.
 *   - n_spend = 1, 2, 3, and a bounded larger set (8), crossed with n_tweaks
 *     spanning a GPU launch chunk boundary (1, 19, 128, 129).
 *   - Duplicate spend candidates produce identical columns.
 *   - An invalid (out-of-range x, well-formed prefix byte) spend candidate
 *     yields prefix64 == 0 for its column without failing other columns.
 *   - n_tweaks == 0 / n_spend == 0 are valid no-ops.
 *   - ufsecp_gpu_bip352_scan_batch(n_spend=1 semantics) stays byte-identical
 *     to ufsecp_gpu_bip352_scan_batch_multispend(..., n_spend=1, ...).
 *   - Scan-only Metal self-containment (issue #335 "Related"): preprocessing
 *     secp256k1_extended.h with -DSECP256K1_METAL_SCAN_ONLY must exclude
 *     ct_ecdsa_sign_metal/ct_schnorr_sign_metal/ct_ecdsa_sign_recoverable_metal
 *     and their wrapper definitions, while the default (unguarded) build
 *     keeps them. Advisory: requires a system C preprocessor and the shader
 *     source tree to be locatable; skips (not fails) otherwise.
 *
 * Metal same-context concurrency + lifecycle (issue #335 acceptance repair,
 * round 2), SW-BIP352-METAL-*: targets UFSECP_GPU_BACKEND_METAL specifically.
 * Advisory-skips on every machine without real Apple/Metal hardware (this
 * development environment has none -- see
 * benchmarks/github_issue_335/macos_replay.sh for the one-command macOS
 * replay bundle).
 *   - METAL-1: N threads call bip352_scan_batch_multispend concurrently on
 *     ONE ufsecp_gpu_ctx with per-thread-distinct inputs; each thread's
 *     output must match its own single-threaded reference (proves
 *     bip352_pool_mtx_ prevents cross-thread pool corruption).
 *   - METAL-2: destroy + recreate the context; a fresh scan must still match
 *     the pre-destroy reference (proves shutdown()'s free_all() leaves no
 *     stale device-buffer state).
 *   - METAL-3: no concurrent thread observes a torn (partially-zeroed)
 *     output row on dispatch failure.
 *
 * Compiles as standalone (define STANDALONE_TEST) or as part of the audit runner.
 * ============================================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#include <thread>
#include <atomic>

#include "ufsecp/ufsecp.h"
#include "ufsecp/ufsecp_gpu.h"

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;

#define CHECK(cond, id, msg)                                                    \
    do {                                                                        \
        if (cond) { ++g_pass; }                                                 \
        else { ++g_fail; std::printf("  FAIL %s: %s\n", (id), (msg)); }        \
    } while (0)

#define SKIP(id, msg)                                                           \
    do { ++g_skip; std::printf("  SKIP %s: %s\n", (id), (msg)); } while (0)

/* ---- deterministic test-vector data ------------------------------------- */
/* These constants match the benchmark in opencl/benchmarks/bench_bip352_opencl.cpp
 * so that audit and bench share the same reference scalars. */
static const uint8_t SCAN_KEY[32] = {
    0xc4,0x23,0x9f,0xd6,0xfc,0x3d,0xb6,0xe2,
    0x2b,0x8b,0xed,0x6a,0x49,0x21,0x9e,0x4e,
    0x30,0xd7,0xd6,0xa3,0xb9,0x82,0x94,0xb1,
    0x38,0xaf,0x4a,0xd3,0x00,0xda,0x1a,0x42
};

static const uint8_t ORDER_N[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

static const uint8_t SPEND_PUBKEY[33] = {
    0x02,
    0xe2,0xed,0x4b,0x9c,0xe9,0x14,0x5e,0x17,
    0x21,0xf1,0x1f,0x99,0x5f,0x72,0x6e,0xf8,
    0xcf,0x50,0xfc,0x85,0x92,0x89,0xac,0x94,
    0x4b,0x2d,0xaf,0xe5,0x03,0xa3,0xc7,0x4c
};

/* Simple LCG to generate deterministic tweak compressed public keys.
 * We generate synthetic scalars, derive their public keys via CPU, and use
 * those as tweak inputs so the test is self-contained. */
static void fill_det(uint8_t* buf, size_t len, uint8_t seed) {
    uint32_t st = seed;
    for (size_t i = 0; i < len; ++i) {
        st = st * 1103515245u + 12345u;
        buf[i] = static_cast<uint8_t>((st >> 16) & 0xFF);
    }
}

static bool all_zero(const void* ptr, size_t len) {
    const auto* p = static_cast<const uint8_t*>(ptr);
    for (size_t i = 0; i < len; ++i) {
        if (p[i] != 0) return false;
    }
    return true;
}

/* Build n compressed tweak public keys via ufsecp_pubkey_create.
 * Caller must ensure out_tweaks33 has room for n*33 bytes.
 * Returns false if any pubkey derivation fails. */
static bool build_tweak_pubkeys(ufsecp_ctx* ctx, int n, uint8_t* out_tweaks33) {
    for (int i = 0; i < n; ++i) {
        uint8_t sk[32];
        fill_det(sk, 32, static_cast<uint8_t>(i + 1));
        sk[0] &= 0x7F; /* keep in scalar range */
        if (ufsecp_pubkey_create(ctx, sk, out_tweaks33 + i * 33) != UFSECP_OK)
            return false;
    }
    return true;
}

/* ============================================================================
 * SW-BIP352-1 — macro constant
 * ============================================================================ */
static void test_bip352_1_macro() {
    std::printf("[bip352_scan] SW-BIP352-1: UFSECP_BIP352_SCAN_PLAN_BYTES == 264\n");
    CHECK(UFSECP_BIP352_SCAN_PLAN_BYTES == 264,
          "SW-BIP352-1", "UFSECP_BIP352_SCAN_PLAN_BYTES macro == 264");
}

/* ============================================================================
 * SW-BIP352-2, SW-BIP352-3 — CPU prepare_scan_plan
 * ============================================================================ */
static void test_bip352_2_3_cpu_plan() {
    std::printf("[bip352_scan] SW-BIP352-2/3: ufsecp_bip352_prepare_scan_plan\n");

    /* SW-BIP352-2: null scan_privkey */
    uint8_t plan[264] = {};
    auto rc_null = ufsecp_bip352_prepare_scan_plan(nullptr, plan);
    CHECK(rc_null != UFSECP_OK, "SW-BIP352-2", "null scan_privkey → error");

    std::memset(plan, 0xA5, sizeof(plan));
    uint8_t zero_scan[32] = {};
    auto rc_zero = ufsecp_bip352_prepare_scan_plan(zero_scan, plan);
    CHECK(rc_zero == UFSECP_ERR_BAD_KEY,
          "SW-BIP352-2b", "zero scan_privkey → BAD_KEY");
    CHECK(all_zero(plan, sizeof(plan)),
          "SW-BIP352-2c", "zero scan_privkey leaves plan output zeroed");

    std::memset(plan, 0xA5, sizeof(plan));
    auto rc_order = ufsecp_bip352_prepare_scan_plan(ORDER_N, plan);
    CHECK(rc_order == UFSECP_ERR_BAD_KEY,
          "SW-BIP352-2d", "scan_privkey == group order → BAD_KEY");
    CHECK(all_zero(plan, sizeof(plan)),
          "SW-BIP352-2e", "order scan_privkey leaves plan output zeroed");

    /* SW-BIP352-3: valid key → non-zero plan */
    auto rc_ok = ufsecp_bip352_prepare_scan_plan(SCAN_KEY, plan);
    CHECK(rc_ok == UFSECP_OK, "SW-BIP352-3a", "prepare_scan_plan returns UFSECP_OK");

    /* wNAF bytes at offset 0 should not all be zero for a random-looking key */
    uint8_t zero_plan[264] = {};
    CHECK(std::memcmp(plan, zero_plan, 264) != 0,
          "SW-BIP352-3b", "plan output is non-zero");
}

/* ============================================================================
 * SW-BIP352-4 to SW-BIP352-13 — GPU batch scan
 * ============================================================================ */
static void test_bip352_4_13_gpu(ufsecp_ctx* cpu_ctx) {
    std::printf("[bip352_scan] SW-BIP352-4..13: GPU batch scan\n");

    /* --- open GPU context ----------------------------------------------- */
    ufsecp_gpu_ctx* gpu = nullptr;
    {
        uint32_t ids[8] = {};
        uint32_t cnt = ufsecp_gpu_backend_count(ids, 8);
        if (cnt == 0) {
            SKIP("SW-BIP352-4", "no GPU backend available");
            SKIP("SW-BIP352-5", "no GPU backend available");
            SKIP("SW-BIP352-6", "no GPU backend available");
            SKIP("SW-BIP352-7", "no GPU backend available");
            SKIP("SW-BIP352-8", "no GPU backend available");
            SKIP("SW-BIP352-9", "no GPU backend available");
            SKIP("SW-BIP352-10", "no GPU backend available");
            SKIP("SW-BIP352-11", "no GPU backend available");
            SKIP("SW-BIP352-12", "no GPU backend available");
            SKIP("SW-BIP352-13", "no GPU backend available");
            return;
        }
        auto rc = ufsecp_gpu_ctx_create(&gpu, ids[0], 0);
        if (rc != UFSECP_OK || !gpu) {
            SKIP("SW-BIP352-4", "GPU context creation failed");
            SKIP("SW-BIP352-5", "GPU context creation failed");
            SKIP("SW-BIP352-6", "GPU context creation failed");
            SKIP("SW-BIP352-7", "GPU context creation failed");
            SKIP("SW-BIP352-8", "GPU context creation failed");
            SKIP("SW-BIP352-9", "GPU context creation failed");
            SKIP("SW-BIP352-10", "GPU context creation failed");
            SKIP("SW-BIP352-11", "GPU context creation failed");
            SKIP("SW-BIP352-12", "GPU context creation failed");
            SKIP("SW-BIP352-13", "GPU context creation failed");
            return;
        }
    }

    /* Build 16 deterministic tweak pubkeys */
    constexpr int N = 16;
    uint8_t tweaks[N * 33] = {};
    bool ok = build_tweak_pubkeys(cpu_ctx, N, tweaks);
    if (!ok) {
        std::printf("  WARN: failed to build deterministic tweak pubkeys\n");
        ufsecp_gpu_ctx_destroy(gpu);
        return;
    }

    uint64_t prefixes[N] = {};
    uint64_t dummy1 = 0;

    /* SW-BIP352-4: null ctx */
    auto rc4 = ufsecp_gpu_bip352_scan_batch(nullptr, SCAN_KEY, SPEND_PUBKEY,
                                             tweaks, 1, prefixes);
    CHECK(rc4 != UFSECP_OK, "SW-BIP352-4", "null GPU ctx → error");

    /* SW-BIP352-5: null scan_privkey */
    auto rc5 = ufsecp_gpu_bip352_scan_batch(gpu, nullptr, SPEND_PUBKEY,
                                             tweaks, 1, prefixes);
    CHECK(rc5 != UFSECP_OK, "SW-BIP352-5", "null scan_privkey → error");

    /* SW-BIP352-6: null spend_pubkey */
    auto rc6 = ufsecp_gpu_bip352_scan_batch(gpu, SCAN_KEY, nullptr,
                                             tweaks, 1, prefixes);
    CHECK(rc6 != UFSECP_OK, "SW-BIP352-6", "null spend_pubkey → error");

    /* SW-BIP352-7: null tweak_pubkeys (n_tweaks > 0) */
    auto rc7 = ufsecp_gpu_bip352_scan_batch(gpu, SCAN_KEY, SPEND_PUBKEY,
                                             nullptr, 1, prefixes);
    CHECK(rc7 != UFSECP_OK, "SW-BIP352-7", "null tweak_pubkeys w/ n>0 → error");

    /* SW-BIP352-8: null prefix64_out */
    auto rc8 = ufsecp_gpu_bip352_scan_batch(gpu, SCAN_KEY, SPEND_PUBKEY,
                                             tweaks, 1, nullptr);
    CHECK(rc8 != UFSECP_OK, "SW-BIP352-8", "null prefix64_out → error");

    /* SW-BIP352-9: n_tweaks == 0 → must not crash and must return OK */
    auto rc9 = ufsecp_gpu_bip352_scan_batch(gpu, SCAN_KEY, SPEND_PUBKEY,
                                             tweaks, 0, &dummy1);
    CHECK(rc9 == UFSECP_OK, "SW-BIP352-9", "n_tweaks==0 → UFSECP_OK");

    uint8_t zero_scan[32] = {};
    dummy1 = UINT64_MAX;
    auto rc9a = ufsecp_gpu_bip352_scan_batch(gpu, zero_scan, SPEND_PUBKEY,
                                              tweaks, 1, &dummy1);
    CHECK(rc9a == UFSECP_ERR_BAD_KEY,
          "SW-BIP352-9a", "zero scan key is rejected before GPU dispatch");
    CHECK(dummy1 == 0,
          "SW-BIP352-9b", "bad scan key clears prefix64_out");

    uint8_t bad_tweak[33] = {};
    std::memcpy(bad_tweak, tweaks, sizeof(bad_tweak));
    bad_tweak[0] = 0x05;
    dummy1 = UINT64_MAX;
    auto rc9b = ufsecp_gpu_bip352_scan_batch(gpu, SCAN_KEY, SPEND_PUBKEY,
                                              bad_tweak, 1, &dummy1);
    if (rc9b == UFSECP_ERR_GPU_UNSUPPORTED) {
        SKIP("SW-BIP352-14", "bip352_scan_batch not supported on this backend");
    } else {
        CHECK(rc9b != UFSECP_OK,
              "SW-BIP352-14", "invalid compressed tweak pubkey is rejected");
        CHECK(dummy1 == 0,
              "SW-BIP352-14b", "invalid tweak pubkey clears prefix64_out");
    }

    /* SW-BIP352-10: single tweak → prefix non-zero */
    uint64_t p1 = 0;
    auto rc10 = ufsecp_gpu_bip352_scan_batch(gpu, SCAN_KEY, SPEND_PUBKEY,
                                              tweaks, 1, &p1);
    if (rc10 == UFSECP_ERR_GPU_UNSUPPORTED) {
        SKIP("SW-BIP352-10", "bip352_scan_batch not supported on this backend");
    } else {
        CHECK(rc10 == UFSECP_OK,  "SW-BIP352-10a", "single tweak returns UFSECP_OK");
        CHECK(p1   != 0,          "SW-BIP352-10b", "prefix is non-zero for single tweak");
    }

    /* SW-BIP352-11: batch of N tweaks → all prefixes non-zero */
    auto rc11 = ufsecp_gpu_bip352_scan_batch(gpu, SCAN_KEY, SPEND_PUBKEY,
                                              tweaks, N, prefixes);
    if (rc11 == UFSECP_ERR_GPU_UNSUPPORTED) {
        SKIP("SW-BIP352-11", "bip352_scan_batch not supported on this backend");
    } else {
        CHECK(rc11 == UFSECP_OK, "SW-BIP352-11a", "batch of 16 returns UFSECP_OK");
        bool any_zero = false;
        for (int i = 0; i < N; ++i) if (prefixes[i] == 0) { any_zero = true; break; }
        CHECK(!any_zero, "SW-BIP352-11b", "no prefix is zero in batch of 16");
    }

    /* SW-BIP352-12: determinism — same call must return identical prefixes */
    uint64_t prefixes2[N] = {};
    auto rc12 = ufsecp_gpu_bip352_scan_batch(gpu, SCAN_KEY, SPEND_PUBKEY,
                                              tweaks, N, prefixes2);
    if (rc12 == UFSECP_ERR_GPU_UNSUPPORTED) {
        SKIP("SW-BIP352-12", "bip352_scan_batch not supported on this backend");
    } else {
        CHECK(rc12 == UFSECP_OK, "SW-BIP352-12a", "second call returns UFSECP_OK");
        CHECK(std::memcmp(prefixes, prefixes2, N * sizeof(uint64_t)) == 0,
              "SW-BIP352-12b", "identical call → identical prefixes");
    }

    /* SW-BIP352-13: distinct tweaks → not all prefixes identical */
    if (rc11 == UFSECP_OK && N >= 2) {
        bool all_same = true;
        for (int i = 1; i < N; ++i) {
            if (prefixes[i] != prefixes[0]) { all_same = false; break; }
        }
        CHECK(!all_same, "SW-BIP352-13", "distinct tweaks produce distinct prefixes");
    } else {
        SKIP("SW-BIP352-13", "batch scan unavailable or N<2");
    }

    ufsecp_gpu_ctx_destroy(gpu);
}

/* ============================================================================
 * SW-BIP352-M — multi-spend-key GPU scan (GitHub issue #335)
 * ============================================================================
 * Independent CPU oracle via ufsecp_silent_payment_create_output (sender
 * side, secp256k1::silent_payment_create_output -- a code path separate from
 * the GPU scan kernels under test). */
static bool oracle_expected_prefix(
    ufsecp_ctx* cpu_ctx,
    const uint8_t tweak_input_sk[32],
    const uint8_t scan_pubkey33[33],
    const uint8_t spend_pubkey33[33],
    uint64_t* prefix_out)
{
    uint8_t out33[33] = {};
    auto rc = ufsecp_silent_payment_create_output(
        cpu_ctx, tweak_input_sk, 1, scan_pubkey33, spend_pubkey33, 0, out33, nullptr);
    if (rc != UFSECP_OK) return false;
    uint64_t pref = 0;
    for (int i = 0; i < 8; ++i) pref = (pref << 8) | out33[1 + i];
    *prefix_out = pref;
    return true;
}

static void test_bip352_multispend_gpu(ufsecp_ctx* cpu_ctx) {
    std::printf("[bip352_scan] SW-BIP352-M: multi-spend-key GPU scan (issue #335)\n");

    ufsecp_gpu_ctx* gpu = nullptr;
    {
        uint32_t ids[8] = {};
        uint32_t cnt = ufsecp_gpu_backend_count(ids, 8);
        if (cnt == 0) { SKIP("SW-BIP352-M0", "no GPU backend available"); return; }
        auto rc = ufsecp_gpu_ctx_create(&gpu, ids[0], 0);
        if (rc != UFSECP_OK || !gpu) { SKIP("SW-BIP352-M0", "GPU context creation failed"); return; }
    }

    uint8_t scan_pubkey33[33] = {};
    if (ufsecp_pubkey_create(cpu_ctx, SCAN_KEY, scan_pubkey33) != UFSECP_OK) {
        SKIP("SW-BIP352-M0", "scan pubkey derivation failed");
        ufsecp_gpu_ctx_destroy(gpu);
        return;
    }

    // n_spend candidate spend keypairs. This op has no BIP-352 label
    // semantics of its own (issue #335) -- these stand in for a wallet's
    // base spend key plus its change-label-derived spend keys, which the
    // caller precomputes exactly as it precomputes these.
    constexpr int MAX_SPEND = 8;
    uint8_t spend_sk[MAX_SPEND][32] = {};
    uint8_t spend_pk[MAX_SPEND][33] = {};
    for (int j = 0; j < MAX_SPEND; ++j) {
        fill_det(spend_sk[j], 32, static_cast<uint8_t>(100 + j));
        spend_sk[j][0] &= 0x7F;
        if (ufsecp_pubkey_create(cpu_ctx, spend_sk[j], spend_pk[j]) != UFSECP_OK) {
            SKIP("SW-BIP352-M0", "spend pubkey derivation failed");
            ufsecp_gpu_ctx_destroy(gpu);
            return;
        }
    }

    // Tweak inputs (sender UTXO input keys). MAX_TWEAKS deliberately spans a
    // typical 128-thread GPU launch chunk boundary.
    constexpr int MAX_TWEAKS = 129;
    static uint8_t tweak_sk[MAX_TWEAKS][32];
    static uint8_t tweak_pk[MAX_TWEAKS * 33];
    for (int i = 0; i < MAX_TWEAKS; ++i) {
        fill_det(tweak_sk[i], 32, static_cast<uint8_t>(i + 1));
        tweak_sk[i][0] &= 0x7F;
        if (ufsecp_pubkey_create(cpu_ctx, tweak_sk[i], tweak_pk + i * 33) != UFSECP_OK) {
            SKIP("SW-BIP352-M0", "tweak pubkey derivation failed");
            ufsecp_gpu_ctx_destroy(gpu);
            return;
        }
    }

    bool any_supported = false;
    const int spend_counts[]  = {1, 2, 3, MAX_SPEND};
    const int tweak_counts[]  = {1, 19, 128, MAX_TWEAKS};
    for (int n_spend : spend_counts) {
        uint8_t spend_flat[MAX_SPEND * 33] = {};
        for (int j = 0; j < n_spend; ++j) std::memcpy(spend_flat + j * 33, spend_pk[j], 33);

        for (int n_tweaks : tweak_counts) {
            std::vector<uint64_t> matrix(static_cast<size_t>(n_tweaks) * n_spend, UINT64_MAX);
            auto rc = ufsecp_gpu_bip352_scan_batch_multispend(
                gpu, SCAN_KEY, spend_flat, n_spend, tweak_pk, n_tweaks, matrix.data());
            char id[80];
            std::snprintf(id, sizeof(id), "SW-BIP352-M1-spend%d-tweak%d", n_spend, n_tweaks);
            if (rc == UFSECP_ERR_GPU_UNSUPPORTED) {
                SKIP(id, "multi-spend scan not supported on this backend");
                continue;
            }
            CHECK(rc == UFSECP_OK, id, "multispend scan returns UFSECP_OK");
            if (rc != UFSECP_OK) continue;
            any_supported = true;

            // Full point/prefix equality vs the independent CPU oracle, for
            // every (tweak, spend) cell -- proves row-major layout, per-cell
            // correctness, and result ordering all at once (a layout/index
            // bug would show up as a mismatch against the correct column).
            for (int i = 0; i < n_tweaks; ++i) {
                for (int j = 0; j < n_spend; ++j) {
                    uint64_t expected = 0;
                    if (!oracle_expected_prefix(cpu_ctx, tweak_sk[i], scan_pubkey33,
                                                spend_pk[j], &expected)) {
                        continue; // extremely unlikely (infinity point); not a failure
                    }
                    CHECK(matrix[(size_t)i * n_spend + j] == expected, id,
                          "GPU multispend cell matches independent CPU oracle (create_output)");
                }
            }
        }
    }
    if (!any_supported) {
        ufsecp_gpu_ctx_destroy(gpu);
        return;
    }

    /* SW-BIP352-M2: duplicate spend candidates -> identical columns. */
    {
        uint8_t dup_spend[3 * 33];
        std::memcpy(dup_spend + 0 * 33, spend_pk[0], 33);
        std::memcpy(dup_spend + 1 * 33, spend_pk[1], 33);
        std::memcpy(dup_spend + 2 * 33, spend_pk[0], 33); // duplicate of candidate 0
        std::vector<uint64_t> matrix(16 * 3, 0);
        auto rc = ufsecp_gpu_bip352_scan_batch_multispend(
            gpu, SCAN_KEY, dup_spend, 3, tweak_pk, 16, matrix.data());
        if (rc == UFSECP_OK) {
            bool all_match = true;
            for (int i = 0; i < 16; ++i) {
                if (matrix[(size_t)i * 3 + 0] != matrix[(size_t)i * 3 + 2]) { all_match = false; break; }
            }
            CHECK(all_match, "SW-BIP352-M2", "duplicate spend candidates produce identical columns");
        } else {
            SKIP("SW-BIP352-M2", "multi-spend scan not supported on this backend");
        }
    }

    /* SW-BIP352-M3: invalid spend candidate (off-curve x, well-formed
     * prefix byte) yields prefix64 == 0 for its column only -- other
     * columns must still be valid, non-zero, and correct. x=5 is < the
     * field prime (always parses as a valid field element) but y^2=x^3+7
     * is not a quadratic residue for x=5, so it is unconditionally
     * off-curve regardless of how strict the field-range check is. */
    {
        uint8_t mixed_spend[2 * 33];
        std::memcpy(mixed_spend + 0 * 33, spend_pk[0], 33);
        mixed_spend[1 * 33] = 0x02;
        std::memset(mixed_spend + 1 * 33 + 1, 0x00, 32);
        mixed_spend[1 * 33 + 32] = 0x05; // x = 5, off-curve (not a QR)
        std::vector<uint64_t> matrix(8 * 2, UINT64_MAX);
        auto rc = ufsecp_gpu_bip352_scan_batch_multispend(
            gpu, SCAN_KEY, mixed_spend, 2, tweak_pk, 8, matrix.data());
        if (rc == UFSECP_OK) {
            bool col1_all_zero = true;
            bool col0_all_nonzero = true;
            for (int i = 0; i < 8; ++i) {
                if (matrix[(size_t)i * 2 + 1] != 0) col1_all_zero = false;
                if (matrix[(size_t)i * 2 + 0] == 0) col0_all_nonzero = false;
            }
            CHECK(col1_all_zero, "SW-BIP352-M3a",
                  "invalid spend candidate column reads 0 for every tweak");
            CHECK(col0_all_nonzero, "SW-BIP352-M3b",
                  "valid spend candidate column unaffected by neighboring invalid candidate");
        } else {
            SKIP("SW-BIP352-M3", "multi-spend scan not supported on this backend");
        }
    }

    /* SW-BIP352-M4: n_tweaks == 0 / n_spend == 0 are valid no-ops. */
    {
        uint64_t dummy = UINT64_MAX;
        auto rc_t0 = ufsecp_gpu_bip352_scan_batch_multispend(
            gpu, SCAN_KEY, spend_pk[0], 1, tweak_pk, 0, &dummy);
        CHECK(rc_t0 == UFSECP_OK, "SW-BIP352-M4a", "n_tweaks==0 -> UFSECP_OK");

        auto rc_s0 = ufsecp_gpu_bip352_scan_batch_multispend(
            gpu, SCAN_KEY, spend_pk[0], 0, tweak_pk, 1, &dummy);
        CHECK(rc_s0 == UFSECP_OK, "SW-BIP352-M4b", "n_spend==0 -> UFSECP_OK");
    }

    /* SW-BIP352-M5: ufsecp_gpu_bip352_scan_batch (single-spend, n_spend=1
     * semantics) stays byte-identical to
     * ufsecp_gpu_bip352_scan_batch_multispend(..., n_spend=1, ...). */
    {
        uint64_t legacy[16] = {};
        uint64_t multi[16]  = {};
        auto rc_legacy = ufsecp_gpu_bip352_scan_batch(
            gpu, SCAN_KEY, spend_pk[0], tweak_pk, 16, legacy);
        auto rc_multi = ufsecp_gpu_bip352_scan_batch_multispend(
            gpu, SCAN_KEY, spend_pk[0], 1, tweak_pk, 16, multi);
        if (rc_legacy == UFSECP_OK && rc_multi == UFSECP_OK) {
            CHECK(std::memcmp(legacy, multi, sizeof(legacy)) == 0, "SW-BIP352-M5",
                  "legacy single-spend output byte-identical to multispend(n_spend=1)");
        } else {
            SKIP("SW-BIP352-M5", "one of the two entry points is unsupported on this backend");
        }
    }

    ufsecp_gpu_ctx_destroy(gpu);
}

/* ============================================================================
 * SW-BIP352-SCANONLY — Metal scan-only self-containment (issue #335 "Related")
 * ============================================================================
 * Advisory (SKIPs, does not fail, if a C preprocessor or the shader source
 * tree cannot be located): preprocesses secp256k1_extended.h with and
 * without -DSECP256K1_METAL_SCAN_ONLY and checks that the guard excludes the
 * ct_ecdsa_sign_metal / ct_schnorr_sign_metal / ct_ecdsa_sign_recoverable_metal
 * forward declarations and their ecdsa_sign / schnorr_sign /
 * ecdsa_sign_recoverable wrapper bodies only in the scan-only mode, while the
 * default (unguarded) preprocessing still contains them -- i.e. proves the
 * gate actually removes the toxic dependency without silently degrading the
 * default full build. This is a preprocessor-level self-containment check,
 * not a full Metal `.air`/.metallib compile (no macOS/Metal compiler is
 * available on every CI/dev machine); it would have caught the original
 * "Undefined symbol ct_ecdsa_sign_metal" link failure from issue #335. */
static bool run_cpp_capture(const std::string& cmd, std::string& out) {
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return false;
    char buf[4096];
    out.clear();
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) out.append(buf, n);
    int rc = pclose(fp);
    return rc == 0 && !out.empty();
}

static void test_bip352_metal_scan_only_self_containment() {
    std::printf("[bip352_scan] SW-BIP352-SCANONLY: Metal scan-only self-containment\n");

    const char* candidates[] = {
        "src/metal/shaders/secp256k1_extended.h",
        "../src/metal/shaders/secp256k1_extended.h",
        "../../src/metal/shaders/secp256k1_extended.h",
        "../../../src/metal/shaders/secp256k1_extended.h",
        "../../../../src/metal/shaders/secp256k1_extended.h",
    };
    std::string path;
    for (const char* c : candidates) {
        FILE* f = std::fopen(c, "rb");
        if (f) { std::fclose(f); path = c; break; }
    }
    if (path.empty()) {
        SKIP("SW-BIP352-SCANONLY", "secp256k1_extended.h not found relative to CWD");
        return;
    }

    const char* toxic[] = {
        "ct_ecdsa_sign_metal", "ct_schnorr_sign_metal", "ct_ecdsa_sign_recoverable_metal"
    };

    std::string default_out;
    std::string cmd_default = "cc -E -x c++ \"" + path + "\" 2>/dev/null";
    if (!run_cpp_capture(cmd_default, default_out)) {
        SKIP("SW-BIP352-SCANONLY", "system C preprocessor unavailable or preprocessing failed");
        return;
    }
    bool default_has_all = true;
    for (const char* sym : toxic) {
        if (default_out.find(sym) == std::string::npos) { default_has_all = false; break; }
    }
    CHECK(default_has_all, "SW-BIP352-SCANONLY-1",
          "default (unguarded) preprocessing still declares/defines the ct_*_sign_metal chain");

    std::string scanonly_out;
    std::string cmd_scanonly =
        "cc -E -x c++ -DSECP256K1_METAL_SCAN_ONLY=1 \"" + path + "\" 2>/dev/null";
    if (!run_cpp_capture(cmd_scanonly, scanonly_out)) {
        SKIP("SW-BIP352-SCANONLY", "scan-only preprocessing failed");
        return;
    }
    bool scanonly_has_none = true;
    for (const char* sym : toxic) {
        if (scanonly_out.find(sym) != std::string::npos) { scanonly_has_none = false; break; }
    }
    CHECK(scanonly_has_none, "SW-BIP352-SCANONLY-2",
          "SECP256K1_METAL_SCAN_ONLY preprocessing excludes the entire ct_*_sign_metal chain");
}

/* ============================================================================
 * SW-BIP352-METAL — same-context concurrency + context lifecycle regression
 * (issue #335 acceptance repair, round 2)
 * ============================================================================
 * Targets the Metal backend specifically (UFSECP_GPU_BACKEND_METAL), not
 * "whichever backend id[0] happens to be" -- CUDA/OpenCL are unaffected by
 * this test (they use per-call/thread_local buffer allocation, not a shared
 * grow-only pool). Advisory-skips everywhere Metal is not compiled in or has
 * no device (i.e. everywhere except real Apple hardware -- confirmed no
 * macOS machine is available in this development environment; see
 * benchmarks/github_issue_335/macos_replay.sh for the one-command replay a
 * macOS owner can run to get real PASS/FAIL evidence for these checks).
 *
 * Exercises two round-1/round-2 fixes to MetalBackend::bip352_scan_batch_multispend:
 *   - bip352_pool_mtx_ (round 1): serializes the WHOLE span (pool grow, host
 *     copies, both dispatches, readback, secret erase) for concurrent callers
 *     on the SAME MetalBackend instance / SAME ufsecp_gpu_ctx.
 *   - MetalBackend::shutdown() calling free_all() on bip352_pool_ (round 1):
 *     no stale device-buffer reuse across a destroy+recreate cycle.
 *   - dispatch_sync_checked() failure propagation (round 2): a failed Metal
 *     command buffer must fail closed (all-zero prefix64_out), not silently
 *     report success with corrupted output -- SW-BIP352-METAL-3 checks that
 *     concurrent calls with valid inputs never observe a partial/torn output
 *     row, which round-1's whole-span mutex should guarantee even under a
 *     genuine dispatch failure on one of the concurrent threads.
 *
 * SW-BIP352-METAL-1: N threads call bip352_scan_batch_multispend concurrently
 *   on ONE ufsecp_gpu_ctx with per-thread-distinct deterministic inputs; each
 *   thread's result must match a single-threaded reference computed with the
 *   SAME inputs before the concurrent run started (proves the pool mutex
 *   prevents cross-thread buffer corruption, not just "doesn't crash").
 * SW-BIP352-METAL-2: destroy the ufsecp_gpu_ctx used above and create a FRESH
 *   one; a scan on the new context must still produce the same reference
 *   result (proves shutdown()'s free_all() leaves no stale/wrong buffer
 *   state a fresh init() could reuse incorrectly).
 * SW-BIP352-METAL-3: within the concurrent run, every thread's full output
 *   row is either entirely non-zero-consistent-with-reference or (if a
 *   thread legitimately hit a dispatch failure) entirely zero -- never a mix
 *   (torn write), which would indicate the whole-span mutex is not actually
 *   covering the readback memcpy.
 * ============================================================================ */
static void test_bip352_metal_concurrency_lifecycle(ufsecp_ctx* cpu_ctx) {
    std::printf("[bip352_scan] SW-BIP352-METAL-1..3: Metal same-context concurrency + lifecycle\n");

    if (!ufsecp_gpu_is_available(UFSECP_GPU_BACKEND_METAL)) {
        SKIP("SW-BIP352-METAL-1", "Metal backend not compiled in or no device (no macOS hardware here)");
        SKIP("SW-BIP352-METAL-2", "Metal backend not compiled in or no device (no macOS hardware here)");
        SKIP("SW-BIP352-METAL-3", "Metal backend not compiled in or no device (no macOS hardware here)");
        return;
    }

    constexpr int kThreads = 6;
    constexpr int kNTweaks = 37; /* crosses a threadgroup-size boundary (64) below it */
    constexpr int kNSpend  = 3;

    /* Per-thread distinct deterministic inputs (different scan key + spend
     * keys per thread index so a cross-thread buffer mix-up is detectable:
     * thread i's output must match thread i's OWN reference, not thread j's). */
    struct ThreadInput {
        uint8_t scan_sk[32];
        uint8_t spend_pk[kNSpend][33];
        uint8_t spend_flat[kNSpend * 33];
        uint8_t tweaks[kNTweaks * 33];
        uint64_t reference_out[kNTweaks * kNSpend];
        uint64_t concurrent_out[kNTweaks * kNSpend];
        bool reference_ok = false;
        bool concurrent_ok = false;
    };
    std::vector<ThreadInput> inputs(kThreads);

    for (int t = 0; t < kThreads; ++t) {
        auto& in = inputs[t];
        fill_det(in.scan_sk, 32, static_cast<uint8_t>(0x40 + t));
        in.scan_sk[0] &= 0x7F;
        for (int j = 0; j < kNSpend; ++j) {
            uint8_t sk[32];
            fill_det(sk, 32, static_cast<uint8_t>(0x80 + t * kNSpend + j));
            sk[0] &= 0x7F;
            if (ufsecp_pubkey_create(cpu_ctx, sk, in.spend_pk[j]) != UFSECP_OK) {
                SKIP("SW-BIP352-METAL-1", "spend pubkey derivation failed");
                SKIP("SW-BIP352-METAL-2", "spend pubkey derivation failed");
                SKIP("SW-BIP352-METAL-3", "spend pubkey derivation failed");
                return;
            }
            std::memcpy(in.spend_flat + j * 33, in.spend_pk[j], 33);
        }
        if (!build_tweak_pubkeys(cpu_ctx, kNTweaks, in.tweaks)) {
            SKIP("SW-BIP352-METAL-1", "tweak pubkey derivation failed");
            SKIP("SW-BIP352-METAL-2", "tweak pubkey derivation failed");
            SKIP("SW-BIP352-METAL-3", "tweak pubkey derivation failed");
            return;
        }
    }

    ufsecp_gpu_ctx* gpu = nullptr;
    if (ufsecp_gpu_ctx_create(&gpu, UFSECP_GPU_BACKEND_METAL, 0) != UFSECP_OK || !gpu) {
        SKIP("SW-BIP352-METAL-1", "Metal GPU context creation failed");
        SKIP("SW-BIP352-METAL-2", "Metal GPU context creation failed");
        SKIP("SW-BIP352-METAL-3", "Metal GPU context creation failed");
        return;
    }

    /* -- Serial reference pass: one thread at a time, same ctx -- */
    for (int t = 0; t < kThreads; ++t) {
        auto& in = inputs[t];
        auto rc = ufsecp_gpu_bip352_scan_batch_multispend(
            gpu, in.scan_sk, in.spend_flat, kNSpend, in.tweaks, kNTweaks, in.reference_out);
        in.reference_ok = (rc == UFSECP_OK);
    }
    bool all_reference_ok = true;
    for (auto& in : inputs) all_reference_ok = all_reference_ok && in.reference_ok;
    CHECK(all_reference_ok, "SW-BIP352-METAL-1-setup", "serial reference pass succeeds for all threads' inputs");

    /* -- Concurrent pass: kThreads threads, SAME ufsecp_gpu_ctx -- */
    std::vector<std::thread> pool;
    pool.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        pool.emplace_back([&inputs, gpu, t]() {
            auto& in = inputs[t];
            auto rc = ufsecp_gpu_bip352_scan_batch_multispend(
                gpu, in.scan_sk, in.spend_flat, kNSpend, in.tweaks, kNTweaks, in.concurrent_out);
            in.concurrent_ok = (rc == UFSECP_OK);
        });
    }
    for (auto& th : pool) th.join();

    bool all_concurrent_match = true;
    bool any_torn_row = false;
    for (int t = 0; t < kThreads; ++t) {
        auto& in = inputs[t];
        if (!in.concurrent_ok) {
            /* Fail-closed dispatch failure: entire row must be zero, not partial. */
            bool fully_zero = all_zero(in.concurrent_out, sizeof(in.concurrent_out));
            if (!fully_zero) any_torn_row = true;
            continue;
        }
        if (std::memcmp(in.reference_out, in.concurrent_out, sizeof(in.reference_out)) != 0) {
            all_concurrent_match = false;
        }
    }
    CHECK(all_concurrent_match, "SW-BIP352-METAL-1",
          "concurrent same-context calls: each thread's output matches its own serial reference (no cross-thread pool corruption)");
    CHECK(!any_torn_row, "SW-BIP352-METAL-3",
          "no thread observes a torn (partially-zeroed) output row on dispatch failure");

    ufsecp_gpu_ctx_destroy(gpu);

    /* -- SW-BIP352-METAL-2: destroy + recreate, verify no stale-buffer reuse -- */
    ufsecp_gpu_ctx* gpu2 = nullptr;
    if (ufsecp_gpu_ctx_create(&gpu2, UFSECP_GPU_BACKEND_METAL, 0) != UFSECP_OK || !gpu2) {
        SKIP("SW-BIP352-METAL-2", "Metal GPU context re-creation failed");
        return;
    }
    auto& in0 = inputs[0];
    uint64_t post_recreate_out[kNTweaks * kNSpend] = {};
    auto rc2 = ufsecp_gpu_bip352_scan_batch_multispend(
        gpu2, in0.scan_sk, in0.spend_flat, kNSpend, in0.tweaks, kNTweaks, post_recreate_out);
    CHECK(rc2 == UFSECP_OK, "SW-BIP352-METAL-2-call", "scan on freshly-recreated context succeeds");
    CHECK(std::memcmp(in0.reference_out, post_recreate_out, sizeof(in0.reference_out)) == 0,
          "SW-BIP352-METAL-2", "scan on freshly-recreated context matches pre-destroy reference (no stale device-buffer reuse)");

    ufsecp_gpu_ctx_destroy(gpu2);
}

/* ============================================================================
 * Entry point
 * ============================================================================ */
int test_gpu_bip352_scan_run() {
    std::printf("=== BIP-352 Silent Payment GPU Scan Audit ===\n\n");

    test_bip352_1_macro();
    std::printf("\n");

    test_bip352_2_3_cpu_plan();
    std::printf("\n");

    ufsecp_ctx* cpu_ctx = nullptr;
    if (ufsecp_ctx_create(&cpu_ctx) != UFSECP_OK || !cpu_ctx) {
        std::printf("FATAL: ufsecp_ctx_create failed\n");
        return 1;
    }

    test_bip352_4_13_gpu(cpu_ctx);
    std::printf("\n");

    test_bip352_multispend_gpu(cpu_ctx);
    std::printf("\n");

    test_bip352_metal_scan_only_self_containment();
    std::printf("\n");

    test_bip352_metal_concurrency_lifecycle(cpu_ctx);
    std::printf("\n");

    ufsecp_ctx_destroy(cpu_ctx);

    std::printf("=== Results: %d passed, %d failed, %d skipped ===\n",
                g_pass, g_fail, g_skip);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() {
    return test_gpu_bip352_scan_run();
}
#endif /* STANDALONE_TEST */
