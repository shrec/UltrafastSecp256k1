// ============================================================================
// test_regression_shim_security_v7.cpp — Shim security regression guards
// ============================================================================
// Covers 2026-05-13 v7 audit findings for the libsecp256k1 compatibility shim:
//   T-01: MuSig2 secp256k1_musig_partial_sign now applies ContextBlindingScope
//   T-07: ecdsa_sig_from_data / rsig_from_data use parse_bytes_strict (not from_bytes)
//   T-08: ShimSchnorrCache get() verifies full 32-byte pubkey (not fingerprint only)
//   T-10: secp256k1_context_randomize(NULL) triggers illegal callback, returns 0
//
// advisory=true: compiled into unified_audit_runner only when secp256k1_shim is linked.
// Otherwise shim_run_stubs_unified.cpp provides a stub that returns ADVISORY_SKIP_CODE.
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
#endif

#include "secp256k1.h"
#include "secp256k1_musig.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"

#include <cstdio>
#include <cstring>

namespace {

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("  [FAIL] %s\n", (msg)); } \
} while(0)

// ── T-10: context_randomize(NULL) must return 0 without UB ──────────────────

static void test_context_randomize_null_ctx() {
    std::printf("  [context_randomize_null_ctx]\n");
    unsigned char seed[32] = {};
    // T-10 fix: must return 0 (and call illegal callback, not crash or UB).
    int rc = secp256k1_context_randomize(nullptr, seed);
    CHECK(rc == 0, "context_randomize(NULL) returns 0");
}

// ── T-07: verify rejects opaque sig with r=n (should zero after strict parse) ─

static void test_sig_parse_strict() {
    std::printf("  [sig_parse_strict]\n");
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY |
                                                       SECP256K1_CONTEXT_SIGN);
    // secp256k1 order n (big-endian):
    static const unsigned char n_bytes[32] = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };
    // opaque sig: data[0..31]=r, data[32..63]=s
    secp256k1_ecdsa_signature sig;
    std::memset(&sig, 0, sizeof(sig));
    std::memcpy(sig.data, n_bytes, 32);  // r = n (>= n → strict parse zeros it)
    sig.data[63] = 1;                    // s = 1

    unsigned char seckey[32] = {1};
    secp256k1_pubkey pubkey;
    secp256k1_ec_pubkey_create(ctx, &pubkey, seckey);

    unsigned char msg[32] = {0xAA};
    int rc = secp256k1_ecdsa_verify(ctx, &sig, msg, &pubkey);
    CHECK(rc == 0, "sig with r=n fails verify (T-07: strict parse zeroes r)");

    secp256k1_context_destroy(ctx);
}

// ── T-08: ShimSchnorrCache cross-key correctness ────────────────────────────

static void test_schnorr_cache_correctness() {
    std::printf("  [schnorr_cache_correctness]\n");
    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    unsigned char sk1[32] = {1}, sk2[32] = {2};
    secp256k1_keypair kp1, kp2;
    secp256k1_keypair_create(ctx, &kp1, sk1);
    secp256k1_keypair_create(ctx, &kp2, sk2);

    secp256k1_xonly_pubkey xonly1, xonly2;
    secp256k1_keypair_xonly_pub(ctx, &xonly1, nullptr, &kp1);
    secp256k1_keypair_xonly_pub(ctx, &xonly2, nullptr, &kp2);

    unsigned char msg[32] = {0x55};
    unsigned char sig1[64], sig2[64];
    secp256k1_schnorrsig_sign32(ctx, sig1, msg, &kp1, nullptr);
    secp256k1_schnorrsig_sign32(ctx, sig2, msg, &kp2, nullptr);

    // Warm cache for both keys (first + second encounter triggers GLV build).
    for (int i = 0; i < 3; ++i) {
        secp256k1_schnorrsig_verify(ctx, sig1, msg, 32, &xonly1);
        secp256k1_schnorrsig_verify(ctx, sig2, msg, 32, &xonly2);
    }

    CHECK(secp256k1_schnorrsig_verify(ctx, sig1, msg, 32, &xonly1) == 1,
          "sig1 verifies against xonly1 after cache warm");
    CHECK(secp256k1_schnorrsig_verify(ctx, sig2, msg, 32, &xonly2) == 1,
          "sig2 verifies against xonly2 after cache warm");
    CHECK(secp256k1_schnorrsig_verify(ctx, sig1, msg, 32, &xonly2) == 0,
          "sig1 must NOT verify against xonly2 (T-08: full memcmp, not fingerprint)");
    CHECK(secp256k1_schnorrsig_verify(ctx, sig2, msg, 32, &xonly1) == 0,
          "sig2 must NOT verify against xonly1 (T-08: full memcmp, not fingerprint)");

    secp256k1_context_destroy(ctx);
}

// ── T-01: MuSig2 partial_sign with blinding produces valid final sig ─────────

static void test_musig2_partial_sign_with_blinding() {
    std::printf("  [musig2_partial_sign_with_blinding]\n");
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    unsigned char seed[32] = {0x42};
    secp256k1_context_randomize(ctx, seed);  // activate blinding

    unsigned char sk1[32] = {1}, sk2[32] = {2};
    secp256k1_keypair kp1, kp2;
    secp256k1_keypair_create(ctx, &kp1, sk1);
    secp256k1_keypair_create(ctx, &kp2, sk2);

    secp256k1_pubkey pub1, pub2;
    secp256k1_ec_pubkey_create(ctx, &pub1, sk1);
    secp256k1_ec_pubkey_create(ctx, &pub2, sk2);

    const secp256k1_pubkey* pubs[2] = {&pub1, &pub2};
    secp256k1_musig_keyagg_cache kagg;
    secp256k1_xonly_pubkey aggpub;
    CHECK(secp256k1_musig_pubkey_agg(ctx, nullptr, &aggpub, &kagg, pubs, 2) == 1,
          "pubkey_agg");

    secp256k1_musig_secnonce sn1, sn2;
    secp256k1_musig_pubnonce pn1, pn2;
    unsigned char sid1[32] = {1}, sid2[32] = {2};
    secp256k1_musig_nonce_gen(ctx, &sn1, &pn1, sid1, &kp1, nullptr, nullptr, nullptr);
    secp256k1_musig_nonce_gen(ctx, &sn2, &pn2, sid2, &kp2, nullptr, nullptr, nullptr);

    const secp256k1_musig_pubnonce* pnonces[2] = {&pn1, &pn2};
    secp256k1_musig_aggnonce aggnonce;
    secp256k1_musig_nonce_agg(ctx, &aggnonce, pnonces, 2);

    unsigned char msg[32] = {0xBB};
    secp256k1_musig_session sess;
    secp256k1_musig_nonce_process(ctx, &sess, &aggnonce, msg, &kagg);

    secp256k1_musig_partial_sig psig1, psig2;
    CHECK(secp256k1_musig_partial_sign(ctx, &psig1, &sn1, &kp1, &kagg, &sess) == 1,
          "partial_sign signer-1 succeeds with blinding (T-01)");
    CHECK(secp256k1_musig_partial_sign(ctx, &psig2, &sn2, &kp2, &kagg, &sess) == 1,
          "partial_sign signer-2 succeeds with blinding (T-01)");

    const secp256k1_musig_partial_sig* psigs[2] = {&psig1, &psig2};
    unsigned char fsig[64];
    secp256k1_musig_partial_sig_agg(ctx, fsig, &sess, psigs, 2);

    secp256k1_context* vctx = secp256k1_context_create(SECP256K1_CONTEXT_VERIFY);
    CHECK(secp256k1_schnorrsig_verify(vctx, fsig, msg, 32, &aggpub) == 1,
          "musig2 final signature verifies (T-01 blinding path)");
    secp256k1_context_destroy(vctx);
    secp256k1_context_destroy(ctx);
}

} // namespace

int test_regression_shim_security_v7_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_shim_security_v7] T-01/T-07/T-08/T-10 shim security regression\n");

    test_context_randomize_null_ctx();
    test_sig_parse_strict();
    test_schnorr_cache_correctness();
    test_musig2_partial_sign_with_blinding();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_shim_security_v7_run(); }
#endif
