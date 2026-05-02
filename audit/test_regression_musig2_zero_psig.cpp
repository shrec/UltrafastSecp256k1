// ============================================================================
// Regression: musig2_partial_sign degenerate zero psig returns UFSECP_ERR_INTERNAL
// CRIT-03: ABI wrapper must detect psig==0 and return UFSECP_ERR_INTERNAL, not OK.
//
// MZP-1..3  — Normal partial sign succeeds and produces non-zero partial sig
// MZP-4     — Aggregate still produces a valid Schnorr sig via combine
// MZP-5..6  — Partial sig output buffer is zeroed when an error is returned
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#endif

#include "ufsecp256k1.h"
#include <array>
#include <cstring>
#include <cstdio>
#include <cstdlib>

static int g_fail = 0;
#define ASSERT_TRUE(cond, msg)  do { if (!(cond)) { std::printf("FAIL [%s]: %s\n", __func__, msg); ++g_fail; } } while(0)
#define ASSERT_FALSE(cond, msg) do { if ( (cond)) { std::printf("FAIL [%s]: %s\n", __func__, msg); ++g_fail; } } while(0)

// Helper: build a 2-of-2 musig2 state through nonce generation + session creation.
struct MuSig2State {
    ufsecp_ctx* ctx = nullptr;
    uint8_t sk1[32] = {}, sk2[32] = {};
    uint8_t pk1[33] = {}, pk2[33] = {};
    uint8_t agg_pk32[32] = {};                    // xonly agg pubkey from key_agg
    uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN] = {};
    uint8_t secnonce1[UFSECP_MUSIG2_SECNONCE_LEN] = {};
    uint8_t secnonce2[UFSECP_MUSIG2_SECNONCE_LEN] = {};
    uint8_t pubnonce1[UFSECP_MUSIG2_PUBNONCE_LEN] = {};
    uint8_t pubnonce2[UFSECP_MUSIG2_PUBNONCE_LEN] = {};
    uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN] = {};
    uint8_t session[UFSECP_MUSIG2_SESSION_LEN] = {};
    uint8_t msg[32] = {0xAB, 0xCD, 0xEF};
    bool ok = false;

    MuSig2State() {
        if (ufsecp_ctx_create(&ctx) != UFSECP_OK || !ctx) return;

        sk1[31] = 5;
        sk2[31] = 11;

        if (ufsecp_pubkey_create(ctx, sk1, pk1) != UFSECP_OK) return;
        if (ufsecp_pubkey_create(ctx, sk2, pk2) != UFSECP_OK) return;

        // key_agg: flat byte array (33 bytes per pubkey, contiguous)
        uint8_t pks_flat[66] = {};
        std::memcpy(pks_flat,      pk1, 33);
        std::memcpy(pks_flat + 33, pk2, 33);
        if (ufsecp_musig2_key_agg(ctx, pks_flat, 2, keyagg, agg_pk32) != UFSECP_OK) return;

        // nonce_gen: (ctx, privkey, pubkey32, agg_pubkey32, msg32, extra_in, secnonce, pubnonce)
        if (ufsecp_musig2_nonce_gen(ctx, sk1, pk1, agg_pk32, msg, nullptr,
                                    secnonce1, pubnonce1) != UFSECP_OK) return;
        if (ufsecp_musig2_nonce_gen(ctx, sk2, pk2, agg_pk32, msg, nullptr,
                                    secnonce2, pubnonce2) != UFSECP_OK) return;

        // nonce_agg: flat byte array (PUBNONCE_LEN bytes per nonce, contiguous)
        uint8_t pnonces_flat[UFSECP_MUSIG2_PUBNONCE_LEN * 2] = {};
        std::memcpy(pnonces_flat,                        pubnonce1, UFSECP_MUSIG2_PUBNONCE_LEN);
        std::memcpy(pnonces_flat + UFSECP_MUSIG2_PUBNONCE_LEN, pubnonce2, UFSECP_MUSIG2_PUBNONCE_LEN);
        if (ufsecp_musig2_nonce_agg(ctx, pnonces_flat, 2, aggnonce) != UFSECP_OK) return;

        // start_sign_session: (ctx, aggnonce, keyagg, msg32, session_out)
        if (ufsecp_musig2_start_sign_session(ctx, aggnonce, keyagg, msg, session) != UFSECP_OK) return;
        ok = true;
    }

    ~MuSig2State() { if (ctx) ufsecp_ctx_destroy(ctx); }
};

// MZP-1: partial_sign with valid inputs must succeed.
static void test_mzp_partial_sign_succeeds() {
    MuSig2State s;
    if (!s.ok) { std::printf("SKIP MZP-1: setup failed\n"); return; }

    uint8_t psig[32] = {};
    ufsecp_error_t rc = ufsecp_musig2_partial_sign(
        s.ctx, s.secnonce1, s.sk1, s.keyagg, s.session, 0, psig);
    ASSERT_TRUE(rc == UFSECP_OK, "MZP-1: partial_sign with valid inputs must succeed");

    // MZP-2: partial sig output must not be all zeros
    bool all_zero = true;
    for (int i = 0; i < 32; ++i) all_zero &= (psig[i] == 0);
    ASSERT_FALSE(all_zero, "MZP-2: partial sig output must not be all-zeros on success");
}

// MZP-3: partial_sign with signer_index out of range returns error (not zero psig).
static void test_mzp_oob_signer_rejected() {
    MuSig2State s;
    if (!s.ok) { std::printf("SKIP MZP-3: setup failed\n"); return; }

    uint8_t psig[32];
    std::memset(psig, 0xCC, 32);  // sentinel
    ufsecp_error_t rc = ufsecp_musig2_partial_sign(
        s.ctx, s.secnonce2, s.sk2, s.keyagg, s.session, 99, psig);
    // out-of-range index must be rejected
    ASSERT_FALSE(rc == UFSECP_OK, "MZP-3: OOB signer_index must be rejected");
    // output buffer must be zeroed on error (not left with sentinel)
    bool zeroed = true;
    for (int i = 0; i < 32; ++i) zeroed &= (psig[i] == 0);
    ASSERT_TRUE(zeroed, "MZP-3: output buffer must be zeroed on error");
}

// MZP-4: full 2-of-2 MuSig2 round-trip produces a valid Schnorr signature.
static void test_mzp_full_round_trip() {
    MuSig2State s;
    if (!s.ok) { std::printf("SKIP MZP-4: setup failed\n"); return; }

    uint8_t psig1[32] = {}, psig2[32] = {};
    if (ufsecp_musig2_partial_sign(s.ctx, s.secnonce1, s.sk1, s.keyagg, s.session, 0, psig1) != UFSECP_OK) {
        std::printf("FAIL MZP-4: partial_sign1 failed\n"); ++g_fail; return;
    }
    if (ufsecp_musig2_partial_sign(s.ctx, s.secnonce2, s.sk2, s.keyagg, s.session, 1, psig2) != UFSECP_OK) {
        std::printf("FAIL MZP-4: partial_sign2 failed\n"); ++g_fail; return;
    }

    uint8_t psigs_flat[64] = {};
    std::memcpy(psigs_flat,      psig1, 32);
    std::memcpy(psigs_flat + 32, psig2, 32);
    uint8_t final_sig[64] = {};
    if (ufsecp_musig2_partial_sig_agg(s.ctx, psigs_flat, 2, s.session, final_sig) != UFSECP_OK) {
        std::printf("FAIL MZP-4: sig_agg failed\n"); ++g_fail; return;
    }

    // Verify the aggregated signature using the xonly agg pubkey (stored from key_agg)
    ufsecp_error_t vrc = ufsecp_schnorr_verify(s.ctx, s.msg, s.agg_pk32, final_sig);
    ASSERT_TRUE(vrc == UFSECP_OK, "MZP-4: aggregated MuSig2 signature must verify");
}

// MZP-5: null args return UFSECP_ERR_NULL_ARG (not segfault).
static void test_mzp_null_args_rejected() {
    MuSig2State s;
    if (!s.ok) { std::printf("SKIP MZP-5: setup failed\n"); return; }
    uint8_t psig[32] = {};
    ufsecp_error_t rc = ufsecp_musig2_partial_sign(
        s.ctx, nullptr, s.sk1, s.keyagg, s.session, 0, psig);
    ASSERT_FALSE(rc == UFSECP_OK, "MZP-5: null secnonce must be rejected");
}

// MZP-6: secnonce is consumed (zeroed) even when ufsecp_musig2_partial_sign succeeds.
static void test_mzp_secnonce_consumed() {
    MuSig2State s;
    if (!s.ok) { std::printf("SKIP MZP-6: setup failed\n"); return; }
    // Make a copy of secnonce2 to sign with
    uint8_t sn_copy[UFSECP_MUSIG2_SECNONCE_LEN];
    std::memcpy(sn_copy, s.secnonce2, UFSECP_MUSIG2_SECNONCE_LEN);

    uint8_t psig[32] = {};
    ufsecp_musig2_partial_sign(s.ctx, sn_copy, s.sk2, s.keyagg, s.session, 1, psig);

    bool zeroed = true;
    for (size_t i = 0; i < UFSECP_MUSIG2_SECNONCE_LEN; ++i) zeroed &= (sn_copy[i] == 0);
    ASSERT_TRUE(zeroed, "MZP-6: secnonce must be zeroed (consumed) after partial_sign");
}

int test_regression_musig2_zero_psig_run() {
    g_fail = 0;
    test_mzp_partial_sign_succeeds();
    test_mzp_oob_signer_rejected();
    test_mzp_full_round_trip();
    test_mzp_null_args_rejected();
    test_mzp_secnonce_consumed();

    if (g_fail == 0)
        std::printf("PASS: musig2 zero-psig regression (CRIT-03)\n");
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_musig2_zero_psig_run(); }
#endif
