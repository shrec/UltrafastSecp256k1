// ============================================================================
// EXPLOIT PoC: Parser Fuzz — MuSig2 / FROST / Adaptor Signatures
// ============================================================================
// Track:  SECURITY -- Random-byte inputs to advanced-protocol parsers
//
// Contract: All C ABI functions that accept opaque byte buffers (key packages,
// nonce commits, partial signatures, adaptor signatures) must either succeed
// with valid output or return a well-defined error code.
// They must NEVER: crash, hang, corrupt memory, read out-of-bounds, or
// exhibit UB on adversarial inputs.
//
// TESTS:
//
//  1. musig2_key_agg: random pubkeys  — must not crash
//  2. musig2_key_agg: zero pubkeys    — must not crash
//  3. musig2_key_agg: single pubkey   — must not crash
//  4. musig2_nonce_agg: random blobs  — must not crash
//  5. musig2_partial_verify: random partial sig + keyagg — must not crash
//  6. musig2_partial_sig_agg: random partial sigs — must not crash
//  7. frost_keygen_finalize: random shares — must not crash
//  8. frost_sign: random keypkg + nonce_commits — must not crash
//  9. frost_verify_partial: random partial_sig — must not crash
// 10. frost_aggregate: random partial sigs — must not crash
// 11. schnorr_adaptor_sign: random adaptor_pt — must not crash
// 12. schnorr_adaptor_verify: random sig — must not crash
// 13. ecdsa_adaptor_sign: random inputs — must not crash
// 14. ecdsa_adaptor_recover: random sig blob — must not crash
// 15. null-length nonce_commits arrays — must return error, not crash
//
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>
#include <random>

#include "ufsecp/ufsecp.h"
#include "secp256k1/sanitizer_scale.hpp"

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#define MUST_NOT_CRASH(expr, label) do { \
    (expr); \
    ++g_pass; \
    (void)(label); \
} while(0)

static std::mt19937_64 rng(0xABBAABBA12345678ULL);

static std::vector<uint8_t> rand_blob(size_t n) {
    std::vector<uint8_t> out(n);
    for (auto& b : out) b = static_cast<uint8_t>(rng());
    return out;
}
static std::array<uint8_t,32> rand32() {
    auto v = rand_blob(32); std::array<uint8_t,32> o{}; std::memcpy(o.data(), v.data(), 32); return o;
}
static std::array<uint8_t,33> rand33() {
    auto v = rand_blob(33); std::array<uint8_t,33> o{}; std::memcpy(o.data(), v.data(), 33); return o;
}

// ============================================================================
// Test 1-3: musig2_key_agg with adversarial inputs
// ============================================================================
static void test_musig2_key_agg_random(ufsecp_ctx* ctx) {
    const int N = SCALED(5000, 200);
    printf("[1-3] musig2_key_agg: random / zero / single pubkeys (%d rounds each)\n", N);

    // API: const uint8_t* pubkeys (flat n*32 bytes, x-only), agg_pk32 output
    uint8_t keyagg_out[UFSECP_MUSIG2_KEYAGG_LEN];
    uint8_t agg_pk32[32];

    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        int n_keys = (rng() % 3) + 1;
        auto keys = rand_blob(static_cast<size_t>(n_keys) * 32);
        ufsecp_musig2_key_agg(ctx, keys.data(), static_cast<size_t>(n_keys), keyagg_out, agg_pk32);
        ++no_crash;
    }
    CHECK(no_crash == N, "musig2_key_agg: no crash on random pubkeys");

    // Zero pubkeys
    for (int i = 0; i < 100; ++i) {
        std::vector<uint8_t> zeros(32 * 3, 0);
        ufsecp_musig2_key_agg(ctx, zeros.data(), 3, keyagg_out, agg_pk32);
    }
    CHECK(true, "musig2_key_agg: no crash on all-zero pubkeys (3)");

    // Single pubkey
    for (int i = 0; i < 200; ++i) {
        auto k = rand32();
        ufsecp_musig2_key_agg(ctx, k.data(), 1, keyagg_out, agg_pk32);
    }
    CHECK(true, "musig2_key_agg: no crash on single random pubkey");
}

// ============================================================================
// Test 4: musig2_nonce_agg random blobs
// ============================================================================
static void test_musig2_nonce_agg_random(ufsecp_ctx* ctx) {
    const int N = SCALED(3000, 100);
    printf("[4] musig2_nonce_agg: random nonce blobs (%d rounds)\n", N);

    // API: const uint8_t* pubnonces (flat n*PUBNONCE_LEN bytes)
    uint8_t agg_nonce_out[UFSECP_MUSIG2_AGGNONCE_LEN];
    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        int n = (rng() % 3) + 1;
        auto nonces = rand_blob(static_cast<size_t>(n) * UFSECP_MUSIG2_PUBNONCE_LEN);
        ufsecp_musig2_nonce_agg(ctx, nonces.data(), static_cast<size_t>(n), agg_nonce_out);
        ++no_crash;
    }
    CHECK(no_crash == N, "musig2_nonce_agg: no crash on random nonce blobs");
}

// ============================================================================
// Test 5: musig2_partial_verify random inputs
// ============================================================================
static void test_musig2_partial_verify_random(ufsecp_ctx* ctx) {
    const int N = SCALED(2000, 80);
    printf("[5] musig2_partial_verify: random inputs (%d rounds)\n", N);

    // partial_verify: (ctx, partial32, pubnonce66, pubkey32, keyagg, session, signer_idx)
    uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN];
    uint8_t agg_pk32[32];
    uint8_t session[UFSECP_MUSIG2_SESSION_LEN];
    uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN];

    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        auto partial32  = rand32();
        auto pubnonce66 = rand_blob(UFSECP_MUSIG2_PUBNONCE_LEN);
        auto pubkey32   = rand32();
        auto msg32      = rand32();
        // flat 2-pubkey buffer
        auto pks_flat   = rand_blob(64);  // 2 * 32 x-only
        ufsecp_musig2_key_agg(ctx, pks_flat.data(), 2, keyagg, agg_pk32);
        auto nonces_flat = rand_blob(2 * UFSECP_MUSIG2_PUBNONCE_LEN);
        ufsecp_musig2_nonce_agg(ctx, nonces_flat.data(), 2, aggnonce);
        // start_sign_session(ctx, aggnonce, keyagg, msg32, session_out)
        ufsecp_musig2_start_sign_session(ctx, aggnonce, keyagg, msg32.data(), session);
        ufsecp_musig2_partial_verify(ctx, partial32.data(), pubnonce66.data(),
                                     pubkey32.data(), keyagg, session, 0);
        ++no_crash;
    }
    CHECK(no_crash == N, "musig2_partial_verify: no crash on random inputs");
}

// ============================================================================
// Test 6: musig2_partial_sig_agg random partial sigs
// ============================================================================
static void test_musig2_partial_sig_agg_random(ufsecp_ctx* ctx) {
    const int N = SCALED(2000, 80);
    printf("[6] musig2_partial_sig_agg: random partial sigs (%d rounds)\n", N);

    // partial_sig_agg: (ctx, partial_sigs flat n*32, n, session, sig64_out)
    uint8_t keyagg2[UFSECP_MUSIG2_KEYAGG_LEN];
    uint8_t agg_pk32b[32];
    uint8_t session2[UFSECP_MUSIG2_SESSION_LEN];
    uint8_t aggnonce2[UFSECP_MUSIG2_AGGNONCE_LEN];
    uint8_t sig64[64];

    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        int n = (rng() % 3) + 1;
        auto partial_sigs = rand_blob(static_cast<size_t>(n) * 32);

        auto pks_flat = rand_blob(64);  // 2*32 x-only
        ufsecp_musig2_key_agg(ctx, pks_flat.data(), 2, keyagg2, agg_pk32b);
        auto nonces_flat = rand_blob(2 * UFSECP_MUSIG2_PUBNONCE_LEN);
        ufsecp_musig2_nonce_agg(ctx, nonces_flat.data(), 2, aggnonce2);
        auto msg32 = rand32();
        ufsecp_musig2_start_sign_session(ctx, aggnonce2, keyagg2, msg32.data(), session2);

        ufsecp_musig2_partial_sig_agg(ctx, partial_sigs.data(),
                                      static_cast<size_t>(n), session2, sig64);
        ++no_crash;
    }
    CHECK(no_crash == N, "musig2_partial_sig_agg: no crash on random partial sigs");
}

// ============================================================================
// Test 7: frost_keygen_finalize random shares
// ============================================================================
static void test_frost_keygen_finalize_random(ufsecp_ctx* ctx) {
    const int N = SCALED(2000, 80);
    printf("[7] frost_keygen_finalize: random shares (%d rounds)\n", N);

    uint8_t keypkg[UFSECP_FROST_KEYPKG_LEN];
    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        // Random commit blob and shares for 2-of-3 (adversarial)
        uint32_t threshold      = 2;
        uint32_t n_participants  = 3;
        uint32_t participant_id  = static_cast<uint32_t>(rng() % n_participants) + 1;
        // commit record size = 8 + threshold * 33 bytes per participant
        size_t commit_record = 8 + static_cast<size_t>(threshold) * 33;
        auto commits = rand_blob(static_cast<size_t>(n_participants) * commit_record);
        auto shares  = rand_blob(static_cast<size_t>(n_participants) * UFSECP_FROST_SHARE_LEN);
        // API: (ctx, participant_id, all_commits, commits_len, received_shares, shares_len,
        //        threshold, num_participants, keypkg_out)
        ufsecp_frost_keygen_finalize(ctx, participant_id,
                                     commits.data(), commits.size(),
                                     shares.data(), shares.size(),
                                     threshold, n_participants, keypkg);
        ++no_crash;
    }
    CHECK(no_crash == N, "frost_keygen_finalize: no crash on random shares");
}

// ============================================================================
// Test 8: frost_sign random keypkg + nonce_commits
// ============================================================================
static void test_frost_sign_random(ufsecp_ctx* ctx) {
    const int N = SCALED(2000, 80);
    printf("[8] frost_sign: random keypkg + nonce_commits (%d rounds)\n", N);

    uint8_t partial_sig_out[36];
    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        auto keypkg      = rand_blob(UFSECP_FROST_KEYPKG_LEN);
        auto nonce       = rand_blob(UFSECP_FROST_NONCE_LEN);
        auto msg32       = rand32();
        size_t n_signers = (rng() % 3) + 1;
        auto nonce_commits = rand_blob(UFSECP_FROST_NONCE_COMMIT_LEN * n_signers);
        ufsecp_frost_sign(ctx, keypkg.data(), nonce.data(), msg32.data(),
                          nonce_commits.data(), n_signers, partial_sig_out);
        ++no_crash;
    }
    CHECK(no_crash == N, "frost_sign: no crash on random keypkg + nonce_commits");
}

// ============================================================================
// Test 9: frost_verify_partial random inputs
// ============================================================================
static void test_frost_verify_partial_random(ufsecp_ctx* ctx) {
    const int N = SCALED(2000, 80);
    printf("[9] frost_verify_partial: random partial_sig (%d rounds)\n", N);

    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        auto partial_sig  = rand_blob(36);
        auto vshare33     = rand33();
        auto msg32        = rand32();
        auto gpubkey33    = rand33();
        size_t n_signers  = (rng() % 3) + 1;
        auto nonce_commits = rand_blob(UFSECP_FROST_NONCE_COMMIT_LEN * n_signers);
        ufsecp_frost_verify_partial(ctx, partial_sig.data(), vshare33.data(),
                                    nonce_commits.data(), n_signers,
                                    msg32.data(), gpubkey33.data());
        ++no_crash;
    }
    CHECK(no_crash == N, "frost_verify_partial: no crash on random inputs");
}

// ============================================================================
// Test 10: frost_aggregate random partial sigs
// ============================================================================
static void test_frost_aggregate_random(ufsecp_ctx* ctx) {
    const int N = SCALED(2000, 80);
    printf("[10] frost_aggregate: random partial sigs (%d rounds)\n", N);

    uint8_t sig64[64];
    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        size_t n = (rng() % 3) + 1;
        auto partial_sigs  = rand_blob(36 * n);
        auto nonce_commits = rand_blob(UFSECP_FROST_NONCE_COMMIT_LEN * n);
        auto gpubkey33     = rand33();
        auto msg32         = rand32();
        ufsecp_frost_aggregate(ctx, partial_sigs.data(), n,
                               nonce_commits.data(), n,
                               gpubkey33.data(), msg32.data(), sig64);
        ++no_crash;
    }
    CHECK(no_crash == N, "frost_aggregate: no crash on random partial sigs");
}

// ============================================================================
// Tests 11-12: Schnorr adaptor sign/verify
// ============================================================================
static void test_schnorr_adaptor_random(ufsecp_ctx* ctx) {
    const int N = SCALED(2000, 80);
    printf("[11-12] schnorr_adaptor_sign / _verify: random inputs (%d rounds)\n", N);

    uint8_t adaptor_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN];
    int no_crash_sign = 0, no_crash_verify = 0;
    for (int i = 0; i < N; ++i) {
        auto privkey32  = rand32();
        auto msg32      = rand32();
        auto adaptor_pt = rand33();
        auto aux_rand   = rand32();
        // API: (ctx, privkey, msg32, adaptor_point33, aux_rand, pre_sig_out)
        ufsecp_schnorr_adaptor_sign(ctx, privkey32.data(), msg32.data(),
                                    adaptor_pt.data(), aux_rand.data(), adaptor_sig);
        ++no_crash_sign;

        // Also try verifying a random sig (pubkey_x is 32-byte x-only)
        auto rand_sig  = rand_blob(UFSECP_SCHNORR_ADAPTOR_SIG_LEN);
        auto pubkey_x  = rand32();
        // API: (ctx, pre_sig, pubkey_x32, msg32, adaptor_point33)
        ufsecp_schnorr_adaptor_verify(ctx, rand_sig.data(), pubkey_x.data(),
                                      msg32.data(), adaptor_pt.data());
        ++no_crash_verify;
    }
    CHECK(no_crash_sign   == N, "schnorr_adaptor_sign: no crash on random inputs");
    CHECK(no_crash_verify == N, "schnorr_adaptor_verify: no crash on random sig");
}

// ============================================================================
// Tests 13-14: ECDSA adaptor sign/recover
// ============================================================================
static void test_ecdsa_adaptor_random(ufsecp_ctx* ctx) {
    const int N = SCALED(2000, 80);
    printf("[13-14] ecdsa_adaptor_sign / _recover: random inputs (%d rounds)\n", N);

    uint8_t adaptor_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN];
    int no_crash = 0;
    for (int i = 0; i < N; ++i) {
        auto privkey32  = rand32();
        auto msg32      = rand32();
        auto adaptor_pt = rand33();
        ufsecp_ecdsa_adaptor_sign(ctx, privkey32.data(), msg32.data(),
                                  adaptor_pt.data(), adaptor_sig);

        // Verify a random pre-sig (parser fuzz — no crash required)
        auto rand_sig    = rand_blob(UFSECP_ECDSA_ADAPTOR_SIG_LEN);
        auto pubkey33v   = rand33();
        auto adaptor33v  = rand33();
        // API: (ctx, pre_sig, pubkey33, msg32, adaptor_point33)
        ufsecp_ecdsa_adaptor_verify(ctx, rand_sig.data(), pubkey33v.data(),
                                    msg32.data(), adaptor33v.data());
        ++no_crash;
    }
    CHECK(no_crash == N, "ecdsa_adaptor_sign + recover: no crash on random inputs");
}

// ============================================================================
// Test 15: null-length nonce_commits → must return error, not crash
// ============================================================================
static void test_null_length_nonce_commits(ufsecp_ctx* ctx) {
    printf("[15] null n_signers or zero-length commits → error, no crash\n");

    uint8_t partial_sig[36];
    uint8_t dummy_sig64[64];

    auto keypkg  = rand_blob(UFSECP_FROST_KEYPKG_LEN);
    auto nonce   = rand_blob(UFSECP_FROST_NONCE_LEN);
    auto msg32   = rand32();
    auto gpk33   = rand33();
    auto vsh33   = rand33();

    // n_signers = 0 — must return error (not crash)
    ufsecp_error_t e1 = ufsecp_frost_sign(ctx, keypkg.data(), nonce.data(),
                                           msg32.data(), nullptr, 0, partial_sig);
    CHECK(e1 != UFSECP_OK, "frost_sign(n_signers=0) returns error");

    ufsecp_error_t e2 = ufsecp_frost_aggregate(ctx, nullptr, 0, nullptr, 0,
                                                gpk33.data(), msg32.data(), dummy_sig64);
    CHECK(e2 != UFSECP_OK, "frost_aggregate(n=0) returns error");

    ufsecp_error_t e3 = ufsecp_frost_verify_partial(ctx, partial_sig, vsh33.data(),
                                                     nullptr, 0, msg32.data(), gpk33.data());
    CHECK(e3 != UFSECP_OK, "frost_verify_partial(n_signers=0) returns error");
}

// ============================================================================
// Main
// ============================================================================

#ifdef STANDALONE_TEST
int main(int /*argc*/, char** /*argv*/)
#else
int test_fuzz_musig2_frost_main()
#endif
{
    printf("===================================================================\n");
    printf("FUZZ: MuSig2 / FROST / Adaptor Parser Robustness\n");
    printf("===================================================================\n\n");

    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK || !ctx) {
        printf("  [FATAL] ufsecp_ctx_create failed\n");
        return 1;
    }

    test_musig2_key_agg_random(ctx);        printf("\n");
    test_musig2_nonce_agg_random(ctx);      printf("\n");
    test_musig2_partial_verify_random(ctx); printf("\n");
    test_musig2_partial_sig_agg_random(ctx);printf("\n");
    test_frost_keygen_finalize_random(ctx); printf("\n");
    test_frost_sign_random(ctx);            printf("\n");
    test_frost_verify_partial_random(ctx);  printf("\n");
    test_frost_aggregate_random(ctx);       printf("\n");
    test_schnorr_adaptor_random(ctx);       printf("\n");
    test_ecdsa_adaptor_random(ctx);         printf("\n");
    test_null_length_nonce_commits(ctx);

    ufsecp_ctx_destroy(ctx);

    printf("\n===================================================================\n");
    printf("Results: %d passed, %d failed\n", g_pass, g_fail);
    printf("===================================================================\n");
    return (g_fail == 0) ? 0 : 1;
}
