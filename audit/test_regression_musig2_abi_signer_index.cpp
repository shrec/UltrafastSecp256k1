// ============================================================================
// REGRESSION TEST: MuSig2 ABI signer-index cross-validation (SEC-001)
// ============================================================================
// Track: SECURITY — MuSig2 partial_sign signer_index ↔ privkey mismatch
//
// BACKGROUND:
//   ufsecp_musig2_partial_sign() deserialises the keyagg blob (which stores
//   only key coefficients, NOT individual public keys) and uses signer_index
//   to select the coefficient a_i for the partial signature formula:
//
//     s_i = k_eff + e * a_i * d_i
//
//   Because the individual pubkeys P_i are absent from the blob, the ABI layer
//   cannot check that d_i (privkey) really belongs to signer i.  A caller that
//   passes (wrong_signer_index, signer0_privkey) receives UFSECP_OK with a
//   partial signature computed under the WRONG coefficient — the signature
//   contribution is silently misrouted and only fails at aggregation.
//
// FIX (SEC-001):
//   ufsecp_musig2_partial_sign_v2() accepts the original pubkeys array so
//   that it can derive pubkey = ct::generator_mul(privkey) and compare
//   against pubkeys[signer_index] before any secret material is consumed.
//   A mismatch returns UFSECP_ERR_BAD_KEY immediately.
//
// TESTS:
//   1. Wrong signer_index with v2 → UFSECP_ERR_BAD_KEY (not UFSECP_OK).
//   2. Correct signer_index with v2 → UFSECP_OK.
//   3. NULL pubkeys with v2 → UFSECP_ERR_NULL_ARG.
//   4. signer_index out-of-range with v2 → UFSECP_ERR_BAD_INPUT.
//   5. 3-of-3: every signer succeeds with the correct index.
//   6. 3-of-3: every signer fails when given a neighbour's index.
//   7. Full 2-of-2 round-trip via v2 produces a valid Schnorr signature.
//
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

// C ABI under test
#include "ufsecp.h"

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; printf("    [OK] %s\n", msg); }
    else       { ++g_fail; printf("  [FAIL] %s\n", msg); }
    fflush(stdout);
}

// Fixed test keys — chosen so they are valid secp256k1 scalars.
static const uint8_t SK0[32] = {
    0x3B,0x91,0x8F,0xC2,0x3A,0xA4,0x77,0xD8,
    0xDE,0x9B,0x28,0x12,0xF3,0xBE,0x60,0xCE,
    0x8B,0x7E,0x45,0x26,0xA3,0x81,0x25,0x60,
    0xB9,0x92,0x21,0x3F,0x19,0x33,0xAE,0x71
};
static const uint8_t SK1[32] = {
    0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
    0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
    0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18,
    0x21,0x22,0x23,0x24,0x25,0x26,0x27,0x28
};
static const uint8_t SK2[32] = {
    0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x00,0x11,
    0x22,0x33,0x44,0x55,0x66,0x77,0x88,0x99,
    0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
    0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10
};

static const uint8_t MSG32[32] = {
    0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
    0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
    0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
    0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE
};

// Helper: derive compressed pubkey from private key via the C ABI.
static bool derive_pubkey(ufsecp_ctx* ctx, const uint8_t sk[32], uint8_t pk33_out[33]) {
    return ufsecp_pubkey_create(ctx, sk, pk33_out) == UFSECP_OK;
}

// ─── test_wrong_signer_index_rejected ────────────────────────────────────────
// Core SEC-001 regression: v2 must reject wrong signer_index.
static void test_wrong_signer_index_rejected() {
    printf("  [signer_index_cross_validation]\n");

    ufsecp_ctx* ctx = nullptr;
    check(ufsecp_ctx_create(&ctx) == UFSECP_OK, "ctx create");

    // Derive pubkeys
    uint8_t pk0[33], pk1[33];
    check(derive_pubkey(ctx, SK0, pk0), "pubkey0 derived");
    check(derive_pubkey(ctx, SK1, pk1), "pubkey1 derived");

    // Concatenate pubkeys for key_agg (2-of-2)
    uint8_t pubkeys_buf[66];
    std::memcpy(pubkeys_buf,      pk0, 33);
    std::memcpy(pubkeys_buf + 33, pk1, 33);

    // Key aggregation
    uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN];
    uint8_t agg_pk32[32];
    check(ufsecp_musig2_key_agg(ctx, pubkeys_buf, 2, keyagg, agg_pk32) == UFSECP_OK,
          "key_agg ok");

    // Nonce generation (signer 0)
    uint8_t secnonce0[UFSECP_MUSIG2_SECNONCE_LEN];
    uint8_t pubnonce0[UFSECP_MUSIG2_PUBNONCE_LEN];
    static const uint8_t EXTRA[32] = {0xAA};
    check(ufsecp_musig2_nonce_gen(ctx, SK0, pk0, agg_pk32, MSG32, EXTRA,
                                  secnonce0, pubnonce0) == UFSECP_OK,
          "nonce_gen signer0 ok");

    // Nonce generation (signer 1)
    uint8_t secnonce1[UFSECP_MUSIG2_SECNONCE_LEN];
    uint8_t pubnonce1[UFSECP_MUSIG2_PUBNONCE_LEN];
    static const uint8_t EXTRA1[32] = {0xBB};
    check(ufsecp_musig2_nonce_gen(ctx, SK1, pk1, agg_pk32, MSG32, EXTRA1,
                                  secnonce1, pubnonce1) == UFSECP_OK,
          "nonce_gen signer1 ok");

    // Nonce aggregation
    uint8_t pubnonces_buf[132];
    std::memcpy(pubnonces_buf,      pubnonce0, 66);
    std::memcpy(pubnonces_buf + 66, pubnonce1, 66);
    uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN];
    check(ufsecp_musig2_nonce_agg(ctx, pubnonces_buf, 2, aggnonce) == UFSECP_OK,
          "nonce_agg ok");

    // Start sign session
    uint8_t session[UFSECP_MUSIG2_SESSION_LEN];
    check(ufsecp_musig2_start_sign_session(ctx, aggnonce, keyagg, MSG32, session) == UFSECP_OK,
          "start_sign_session ok");

    // --- TEST 1: wrong signer_index (use SK0 but claim index 1) must be rejected ---
    // Need a fresh secnonce0 copy because a failed v2 still erases it.
    uint8_t secnonce0_copy[UFSECP_MUSIG2_SECNONCE_LEN];
    std::memcpy(secnonce0_copy, secnonce0, UFSECP_MUSIG2_SECNONCE_LEN);

    uint8_t psig_bad[32];
    ufsecp_error_t rc_bad = ufsecp_musig2_partial_sign_v2(
        ctx,
        secnonce0_copy,   // signer 0's nonce
        SK0,              // signer 0's privkey
        pubkeys_buf,      // original pubkeys array
        keyagg, session,
        1,                // WRONG: claim to be signer 1
        psig_bad);
    check(rc_bad != UFSECP_OK,
          "v2: wrong signer_index must return error (not UFSECP_OK)");
    check(rc_bad == UFSECP_ERR_BAD_KEY,
          "v2: wrong signer_index returns UFSECP_ERR_BAD_KEY");

    // Regenerate a fresh secnonce0 (previous copy was erased by the failed call).
    // We need another nonce pair for the correct-index test.
    uint8_t secnonce0b[UFSECP_MUSIG2_SECNONCE_LEN];
    uint8_t pubnonce0b[UFSECP_MUSIG2_PUBNONCE_LEN];
    static const uint8_t EXTRA2[32] = {0xCC};
    check(ufsecp_musig2_nonce_gen(ctx, SK0, pk0, agg_pk32, MSG32, EXTRA2,
                                  secnonce0b, pubnonce0b) == UFSECP_OK,
          "re-nonce_gen signer0 ok");

    // --- TEST 2: correct signer_index (SK0, index 0) must succeed ---
    uint8_t psig_ok[32];
    ufsecp_error_t rc_ok = ufsecp_musig2_partial_sign_v2(
        ctx,
        secnonce0b,
        SK0,
        pubkeys_buf,
        keyagg, session,
        0,               // CORRECT: signer 0
        psig_ok);
    check(rc_ok == UFSECP_OK, "v2: correct signer_index returns UFSECP_OK");

    // --- TEST 3: NULL pubkeys → UFSECP_ERR_NULL_ARG ---
    uint8_t secnonce1_copy[UFSECP_MUSIG2_SECNONCE_LEN];
    std::memcpy(secnonce1_copy, secnonce1, UFSECP_MUSIG2_SECNONCE_LEN);
    uint8_t psig_null[32];
    ufsecp_error_t rc_null = ufsecp_musig2_partial_sign_v2(
        ctx, secnonce1_copy, SK1, nullptr, keyagg, session, 1, psig_null);
    check(rc_null == UFSECP_ERR_NULL_ARG,
          "v2: NULL pubkeys returns UFSECP_ERR_NULL_ARG");

    // --- TEST 4: signer_index out of range → UFSECP_ERR_BAD_INPUT ---
    // Regenerate nonce1 (previous was consumed or possibly erased by null test).
    uint8_t secnonce1b[UFSECP_MUSIG2_SECNONCE_LEN];
    uint8_t pubnonce1b[UFSECP_MUSIG2_PUBNONCE_LEN];
    static const uint8_t EXTRA3[32] = {0xDD};
    check(ufsecp_musig2_nonce_gen(ctx, SK1, pk1, agg_pk32, MSG32, EXTRA3,
                                  secnonce1b, pubnonce1b) == UFSECP_OK,
          "re-nonce_gen signer1 ok");
    uint8_t psig_oob[32];
    ufsecp_error_t rc_oob = ufsecp_musig2_partial_sign_v2(
        ctx, secnonce1b, SK1, pubkeys_buf, keyagg, session,
        99,  // out of range for a 2-signer keyagg
        psig_oob);
    check(rc_oob == UFSECP_ERR_BAD_INPUT,
          "v2: signer_index out-of-range returns UFSECP_ERR_BAD_INPUT");

    ufsecp_ctx_destroy(ctx);
    printf("\n");
}

// ─── test_three_signers_all_indices ──────────────────────────────────────────
// 3-of-3: each signer succeeds with own index, fails with a neighbour's index.
static void test_three_signers_all_indices() {
    printf("  [three_signers_index_validation]\n");

    ufsecp_ctx* ctx = nullptr;
    check(ufsecp_ctx_create(&ctx) == UFSECP_OK, "ctx create (3-of-3)");

    const uint8_t* sks[3] = { SK0, SK1, SK2 };
    uint8_t pks[3][33];
    uint8_t pubkeys_buf[99];
    for (int i = 0; i < 3; ++i) {
        check(derive_pubkey(ctx, sks[i], pks[i]), "pubkey derived");
        std::memcpy(pubkeys_buf + i * 33, pks[i], 33);
    }

    uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN];
    uint8_t agg_pk32[32];
    check(ufsecp_musig2_key_agg(ctx, pubkeys_buf, 3, keyagg, agg_pk32) == UFSECP_OK,
          "key_agg 3-of-3 ok");

    // Generate nonces for all three signers
    uint8_t secnonces[3][UFSECP_MUSIG2_SECNONCE_LEN];
    uint8_t pubnonces_flat[3 * UFSECP_MUSIG2_PUBNONCE_LEN];
    static const uint8_t EXTRAS[3][32] = {{0x11},{0x22},{0x33}};
    for (int i = 0; i < 3; ++i) {
        uint8_t pn[UFSECP_MUSIG2_PUBNONCE_LEN];
        check(ufsecp_musig2_nonce_gen(ctx, sks[i], pks[i], agg_pk32, MSG32,
                                      EXTRAS[i], secnonces[i], pn) == UFSECP_OK,
              "nonce_gen ok");
        std::memcpy(pubnonces_flat + i * UFSECP_MUSIG2_PUBNONCE_LEN, pn, 66);
    }

    uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN];
    check(ufsecp_musig2_nonce_agg(ctx, pubnonces_flat, 3, aggnonce) == UFSECP_OK,
          "nonce_agg 3-of-3 ok");

    uint8_t session[UFSECP_MUSIG2_SESSION_LEN];
    check(ufsecp_musig2_start_sign_session(ctx, aggnonce, keyagg, MSG32, session) == UFSECP_OK,
          "start_sign_session 3-of-3 ok");

    // TEST 5: Each signer succeeds with its own correct index.
    // Need fresh nonces since v2 erases them on success.
    uint8_t secnonces2[3][UFSECP_MUSIG2_SECNONCE_LEN];
    uint8_t pubnonces_flat2[3 * UFSECP_MUSIG2_PUBNONCE_LEN];
    static const uint8_t EXTRAS2[3][32] = {{0x44},{0x55},{0x66}};
    for (int i = 0; i < 3; ++i) {
        uint8_t pn[UFSECP_MUSIG2_PUBNONCE_LEN];
        check(ufsecp_musig2_nonce_gen(ctx, sks[i], pks[i], agg_pk32, MSG32,
                                      EXTRAS2[i], secnonces2[i], pn) == UFSECP_OK,
              "nonce_gen2 ok");
        std::memcpy(pubnonces_flat2 + i * UFSECP_MUSIG2_PUBNONCE_LEN, pn, 66);
    }
    uint8_t aggnonce2[UFSECP_MUSIG2_AGGNONCE_LEN];
    check(ufsecp_musig2_nonce_agg(ctx, pubnonces_flat2, 3, aggnonce2) == UFSECP_OK,
          "nonce_agg2 ok");
    uint8_t session2[UFSECP_MUSIG2_SESSION_LEN];
    check(ufsecp_musig2_start_sign_session(ctx, aggnonce2, keyagg, MSG32, session2) == UFSECP_OK,
          "start_sign_session2 ok");

    for (int i = 0; i < 3; ++i) {
        uint8_t psig[32];
        ufsecp_error_t rc = ufsecp_musig2_partial_sign_v2(
            ctx, secnonces2[i], sks[i], pubkeys_buf, keyagg, session2,
            static_cast<size_t>(i), psig);
        check(rc == UFSECP_OK, "v2: correct own index succeeds");
    }

    // TEST 6: Each signer fails when given a neighbour's index.
    // Generate fresh nonces for the wrong-index tests.
    static const uint8_t EXTRAS3[3][32] = {{0x77},{0x88},{0x99}};
    uint8_t secnonces3[3][UFSECP_MUSIG2_SECNONCE_LEN];
    uint8_t pubnonces_flat3[3 * UFSECP_MUSIG2_PUBNONCE_LEN];
    for (int i = 0; i < 3; ++i) {
        uint8_t pn[UFSECP_MUSIG2_PUBNONCE_LEN];
        check(ufsecp_musig2_nonce_gen(ctx, sks[i], pks[i], agg_pk32, MSG32,
                                      EXTRAS3[i], secnonces3[i], pn) == UFSECP_OK,
              "nonce_gen3 ok");
        std::memcpy(pubnonces_flat3 + i * UFSECP_MUSIG2_PUBNONCE_LEN, pn, 66);
    }
    uint8_t aggnonce3[UFSECP_MUSIG2_AGGNONCE_LEN];
    check(ufsecp_musig2_nonce_agg(ctx, pubnonces_flat3, 3, aggnonce3) == UFSECP_OK,
          "nonce_agg3 ok");
    uint8_t session3[UFSECP_MUSIG2_SESSION_LEN];
    check(ufsecp_musig2_start_sign_session(ctx, aggnonce3, keyagg, MSG32, session3) == UFSECP_OK,
          "start_sign_session3 ok");

    for (int i = 0; i < 3; ++i) {
        // Claim neighbour's index: (i+1) mod 3
        size_t wrong_idx = static_cast<size_t>((i + 1) % 3);
        uint8_t psig_wrong[32];
        ufsecp_error_t rc = ufsecp_musig2_partial_sign_v2(
            ctx, secnonces3[i], sks[i], pubkeys_buf, keyagg, session3,
            wrong_idx, psig_wrong);
        check(rc == UFSECP_ERR_BAD_KEY,
              "v2: wrong neighbour index rejected with UFSECP_ERR_BAD_KEY");
    }

    ufsecp_ctx_destroy(ctx);
    printf("\n");
}

// ─── test_full_roundtrip_via_v2 ──────────────────────────────────────────────
// TEST 7: Full 2-of-2 round-trip using v2 produces a valid Schnorr signature.
static void test_full_roundtrip_via_v2() {
    printf("  [full_roundtrip_via_v2]\n");

    ufsecp_ctx* ctx = nullptr;
    check(ufsecp_ctx_create(&ctx) == UFSECP_OK, "ctx create (roundtrip)");

    uint8_t pk0[33], pk1[33];
    check(derive_pubkey(ctx, SK0, pk0), "pk0");
    check(derive_pubkey(ctx, SK1, pk1), "pk1");

    uint8_t pubkeys_buf[66];
    std::memcpy(pubkeys_buf,      pk0, 33);
    std::memcpy(pubkeys_buf + 33, pk1, 33);

    uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN];
    uint8_t agg_pk32[32];
    check(ufsecp_musig2_key_agg(ctx, pubkeys_buf, 2, keyagg, agg_pk32) == UFSECP_OK,
          "key_agg ok");

    static const uint8_t E0[32] = {0xE0};
    static const uint8_t E1[32] = {0xE1};
    uint8_t sn0[UFSECP_MUSIG2_SECNONCE_LEN], pn0[UFSECP_MUSIG2_PUBNONCE_LEN];
    uint8_t sn1[UFSECP_MUSIG2_SECNONCE_LEN], pn1[UFSECP_MUSIG2_PUBNONCE_LEN];
    check(ufsecp_musig2_nonce_gen(ctx, SK0, pk0, agg_pk32, MSG32, E0, sn0, pn0) == UFSECP_OK,
          "nonce_gen0");
    check(ufsecp_musig2_nonce_gen(ctx, SK1, pk1, agg_pk32, MSG32, E1, sn1, pn1) == UFSECP_OK,
          "nonce_gen1");

    uint8_t pubnonces_buf[132];
    std::memcpy(pubnonces_buf,      pn0, 66);
    std::memcpy(pubnonces_buf + 66, pn1, 66);
    uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN];
    check(ufsecp_musig2_nonce_agg(ctx, pubnonces_buf, 2, aggnonce) == UFSECP_OK,
          "nonce_agg");

    uint8_t session[UFSECP_MUSIG2_SESSION_LEN];
    check(ufsecp_musig2_start_sign_session(ctx, aggnonce, keyagg, MSG32, session) == UFSECP_OK,
          "start_sign_session");

    uint8_t psig0[32], psig1[32];
    check(ufsecp_musig2_partial_sign_v2(ctx, sn0, SK0, pubkeys_buf, keyagg, session, 0, psig0)
          == UFSECP_OK, "partial_sign_v2 signer0");
    check(ufsecp_musig2_partial_sign_v2(ctx, sn1, SK1, pubkeys_buf, keyagg, session, 1, psig1)
          == UFSECP_OK, "partial_sign_v2 signer1");

    uint8_t partial_sigs[64];
    std::memcpy(partial_sigs,      psig0, 32);
    std::memcpy(partial_sigs + 32, psig1, 32);
    uint8_t sig64[64];
    check(ufsecp_musig2_partial_sig_agg(ctx, partial_sigs, 2, session, sig64) == UFSECP_OK,
          "partial_sig_agg ok");

    // Verify the final Schnorr signature against the aggregated pubkey
    check(ufsecp_schnorr_verify(ctx, MSG32, agg_pk32, sig64) == UFSECP_OK,
          "final Schnorr sig verifies against aggregated key");

    ufsecp_ctx_destroy(ctx);
    printf("\n");
}

// ============================================================================
// Entry point
// ============================================================================
int test_regression_musig2_abi_signer_index_run() {
    printf("====================================================================\n");
    printf("REGRESSION: MuSig2 ABI signer-index cross-validation (SEC-001)\n");
    printf("====================================================================\n\n");
    printf("Verifies that ufsecp_musig2_partial_sign_v2() enforces\n");
    printf("privkey ↔ signer_index consistency at the ABI boundary.\n\n");

    test_wrong_signer_index_rejected();
    test_three_signers_all_indices();
    test_full_roundtrip_via_v2();

    printf("====================================================================\n");
    printf("Result: %d passed, %d failed\n", g_pass, g_fail);
    printf("====================================================================\n");

    return (g_fail > 0) ? 1 : 0;
}

#if defined(STANDALONE_TEST)
int main() {
    return test_regression_musig2_abi_signer_index_run();
}
#endif
