// ============================================================================
// test_regression_musig2_session_token.cpp
// Regression: MuSig2 global keyagg map must not be keyed on struct address.
//
// Bug fixed: g_ka map was keyed on const void* (struct address). A
// stack-allocated secp256k1_musig_keyagg_cache freed and reallocated at the
// same address would retrieve the old session's key_coefficients, causing
// partial signing with wrong data.
// Fix: monotonic token written into keyagg_cache->data[0..7] at pubkey_agg;
// map keyed on token — address reuse no longer produces stale lookup.
//
// Tests:
//   MST-1: pubkey_agg assigns a non-zero token into data[0..7]
//   MST-2: two consecutive pubkey_agg calls get distinct tokens
//   MST-3: a new pubkey_agg call overwrites any old token in the struct
//   MST-4: full 2-of-2 MuSig2 sign/verify round-trip succeeds
//   negated-key-term: partial_sig_verify (shim) and the core 33-byte
//     musig2_partial_verify reject s' = k - e*a_i*g*gacc*d_i (BIP-327 checks
//     only the key's real parity); the honest partial sigs still verify.
// ============================================================================

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;
#include <array>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include "secp256k1.h"
#include "secp256k1_musig.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"  // secp256k1_schnorrsig_verify (MST-4 final-sig check)
#include "secp256k1/musig2.hpp"    // core 33-byte / x-only musig2_partial_verify

static const unsigned char kSk1[32] = {
    0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
    0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
    0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01,
    0x01,0x01,0x01,0x01,0x01,0x01,0x01,0x01
};
static const unsigned char kSk2[32] = {
    0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,
    0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,
    0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02,
    0x02,0x02,0x02,0x02,0x02,0x02,0x02,0x02
};

static std::uint64_t read_token(const secp256k1_musig_keyagg_cache* p) {
    std::uint64_t tok = 0;
    std::memcpy(&tok, p->data, sizeof(tok));
    return tok;
}

// ── MST-1: pubkey_agg writes non-zero token ───────────────────────────────
static void test_token_written() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_pubkey pk1, pk2;
    secp256k1_ec_pubkey_create(ctx, &pk1, kSk1);
    secp256k1_ec_pubkey_create(ctx, &pk2, kSk2);
    const secp256k1_pubkey* pks[2] = {&pk1, &pk2};

    secp256k1_musig_keyagg_cache ka;
    std::memset(&ka, 0, sizeof(ka));
    secp256k1_xonly_pubkey agg_pk;
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka, pks, 2) == 1, "[MST-1a] pubkey_agg");
    std::uint64_t tok = read_token(&ka);
    check(tok != 0, "[MST-1b] token is non-zero after pubkey_agg");
    secp256k1_context_destroy(ctx);
}

// ── MST-2: two agg calls get distinct tokens ──────────────────────────────
static void test_distinct_tokens() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_pubkey pk1, pk2;
    secp256k1_ec_pubkey_create(ctx, &pk1, kSk1);
    secp256k1_ec_pubkey_create(ctx, &pk2, kSk2);
    const secp256k1_pubkey* pks[2] = {&pk1, &pk2};

    secp256k1_musig_keyagg_cache ka1, ka2;
    secp256k1_xonly_pubkey agg_pk;
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka1, pks, 2) == 1, "[MST-2a] agg1");
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka2, pks, 2) == 1, "[MST-2b] agg2");
    std::uint64_t t1 = read_token(&ka1), t2 = read_token(&ka2);
    check(t1 != 0 && t2 != 0, "[MST-2c] both tokens non-zero");
    check(t1 != t2, "[MST-2d] distinct structs get distinct tokens");
    secp256k1_context_destroy(ctx);
}

// ── MST-3: re-using same struct gets fresh token ──────────────────────────
static void test_struct_reuse_fresh_token() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_pubkey pk1, pk2;
    secp256k1_ec_pubkey_create(ctx, &pk1, kSk1);
    secp256k1_ec_pubkey_create(ctx, &pk2, kSk2);
    const secp256k1_pubkey* pks[2] = {&pk1, &pk2};

    secp256k1_musig_keyagg_cache ka;
    secp256k1_xonly_pubkey agg_pk;
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka, pks, 2) == 1, "[MST-3a] first agg");
    std::uint64_t tok_first = read_token(&ka);

    // Re-use the same struct (simulating stack reuse)
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka, pks, 2) == 1, "[MST-3b] second agg on same struct");
    std::uint64_t tok_second = read_token(&ka);

    check(tok_first != 0 && tok_second != 0, "[MST-3c] both tokens valid");
    check(tok_first != tok_second, "[MST-3d] reused struct gets fresh token — old session evicted");
    secp256k1_context_destroy(ctx);
}

// ── MST-4: 2-of-2 MuSig2 sign/verify round-trip ─────────────────────────
static void test_2of2_roundtrip() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_keypair kp1, kp2;
    check(secp256k1_keypair_create(ctx, &kp1, kSk1) == 1, "[MST-4a] kp1");
    check(secp256k1_keypair_create(ctx, &kp2, kSk2) == 1, "[MST-4b] kp2");

    secp256k1_pubkey pub1, pub2;
    secp256k1_keypair_pub(ctx, &pub1, &kp1);
    secp256k1_keypair_pub(ctx, &pub2, &kp2);
    const secp256k1_pubkey* pks[2] = {&pub1, &pub2};

    secp256k1_musig_keyagg_cache ka;
    secp256k1_xonly_pubkey agg_pk;
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka, pks, 2) == 1, "[MST-4c] pubkey_agg");

    unsigned char pk1_comp[33], pk2_comp[33]; size_t plen = 33;
    secp256k1_ec_pubkey_serialize(ctx, pk1_comp, &plen, &pub1, SECP256K1_EC_COMPRESSED);
    secp256k1_ec_pubkey_serialize(ctx, pk2_comp, &plen, &pub2, SECP256K1_EC_COMPRESSED);

    const unsigned char msg[32] = {0xDE,0xAD,0xBE,0xEF};
    unsigned char extra1[32] = {0x01}, extra2[32] = {0x02};

    secp256k1_musig_secnonce sn1, sn2;
    secp256k1_musig_pubnonce pn1, pn2;
    check(secp256k1_musig_nonce_gen(ctx, &sn1, &pn1, pk1_comp+1, kSk1, &pub1, msg, &ka, extra1) == 1, "[MST-4d] nonce1");
    check(secp256k1_musig_nonce_gen(ctx, &sn2, &pn2, pk2_comp+1, kSk2, &pub2, msg, &ka, extra2) == 1, "[MST-4e] nonce2");

    const secp256k1_musig_pubnonce* pns[2] = {&pn1, &pn2};
    secp256k1_musig_aggnonce agg_nonce;
    check(secp256k1_musig_nonce_agg(ctx, &agg_nonce, pns, 2) == 1, "[MST-4f] nonce_agg");

    secp256k1_musig_session sess;
    check(secp256k1_musig_nonce_process(ctx, &sess, &agg_nonce, msg, &ka) == 1, "[MST-4g] nonce_process");

    secp256k1_musig_partial_sig ps1, ps2;
    check(secp256k1_musig_partial_sign(ctx, &ps1, &sn1, &kp1, &ka, &sess) == 1, "[MST-4h] partial_sign1");
    check(secp256k1_musig_partial_sign(ctx, &ps2, &sn2, &kp2, &ka, &sess) == 1, "[MST-4i] partial_sign2");

    check(secp256k1_musig_partial_sig_verify(ctx, &ps1, &pn1, &pub1, &ka, &sess) == 1, "[MST-4j] verify_ps1");
    check(secp256k1_musig_partial_sig_verify(ctx, &ps2, &pn2, &pub2, &ka, &sess) == 1, "[MST-4k] verify_ps2");

    const secp256k1_musig_partial_sig* pss[2] = {&ps1, &ps2};
    unsigned char final_sig[64];
    check(secp256k1_musig_partial_sig_agg(ctx, final_sig, &sess, pss, 2) == 1, "[MST-4l] sig_agg");
    check(secp256k1_schnorrsig_verify(ctx, final_sig, msg, 32, &agg_pk) == 1, "[MST-4m] verify final sig");

    secp256k1_context_destroy(ctx);
}

// ── partial_sig_verify must check only the signer key's real parity ───────
// BIP-327 PartialSigVerify checks s*G == R_eff + e*a_i*g*gacc*cpoint(pk_i) with
// the key's REAL parity. A signer that returns s' = k1 + b*k2 - e*a_i*g*gacc*d_i
// (the key term negated) must be rejected: s' passes a parity-lenient check
// against -P_i, yet the aggregate signature is invalid, so blame attribution
// would fail. Regression for NF-2026-00028 (shim passed an x-only key to core).
static void test_negated_key_term_rejected() {
    secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    secp256k1_keypair kp1, kp2;
    check(secp256k1_keypair_create(ctx, &kp1, kSk1) == 1, "[negated-key-term] kp1");
    check(secp256k1_keypair_create(ctx, &kp2, kSk2) == 1, "[negated-key-term] kp2");

    secp256k1_pubkey pub1, pub2;
    secp256k1_keypair_pub(ctx, &pub1, &kp1);
    secp256k1_keypair_pub(ctx, &pub2, &kp2);
    const secp256k1_pubkey* pks[2] = {&pub1, &pub2};

    secp256k1_musig_keyagg_cache ka;
    secp256k1_xonly_pubkey agg_pk;
    check(secp256k1_musig_pubkey_agg(ctx, &agg_pk, &ka, pks, 2) == 1, "[negated-key-term] pubkey_agg");

    unsigned char pk1_comp[33], pk2_comp[33];
    size_t plen = 33;
    secp256k1_ec_pubkey_serialize(ctx, pk1_comp, &plen, &pub1, SECP256K1_EC_COMPRESSED);
    plen = 33;
    secp256k1_ec_pubkey_serialize(ctx, pk2_comp, &plen, &pub2, SECP256K1_EC_COMPRESSED);

    const unsigned char msg[32] = {0x4E,0x45,0x47,0x4B,0x45,0x59};
    unsigned char sid1[32] = {0x11}, sid2[32] = {0x22};

    secp256k1_musig_secnonce sn1, sn2;
    secp256k1_musig_pubnonce pn1, pn2;
    check(secp256k1_musig_nonce_gen(ctx, &sn1, &pn1, sid1, kSk1, &pub1, msg, &ka, nullptr) == 1,
          "[negated-key-term] nonce1");
    check(secp256k1_musig_nonce_gen(ctx, &sn2, &pn2, sid2, kSk2, &pub2, msg, &ka, nullptr) == 1,
          "[negated-key-term] nonce2");

    const secp256k1_musig_pubnonce* pns[2] = {&pn1, &pn2};
    secp256k1_musig_aggnonce agg_nonce;
    check(secp256k1_musig_nonce_agg(ctx, &agg_nonce, pns, 2) == 1, "[negated-key-term] nonce_agg");

    secp256k1_musig_session sess;
    check(secp256k1_musig_nonce_process(ctx, &sess, &agg_nonce, msg, &ka) == 1,
          "[negated-key-term] nonce_process");

    secp256k1_musig_partial_sig ps1, ps2;
    check(secp256k1_musig_partial_sign(ctx, &ps1, &sn1, &kp1, &ka, &sess) == 1, "[negated-key-term] partial_sign1");
    check(secp256k1_musig_partial_sign(ctx, &ps2, &sn2, &kp2, &ka, &sess) == 1, "[negated-key-term] partial_sign2");

    // Recompute the same session in core from the public transcript to get
    // e, a_1, g (Q_negated) and gacc, then build the forged partial signature.
    std::array<std::uint8_t, 33> c1{}, c2{};
    std::memcpy(c1.data(), pk1_comp, 33);
    std::memcpy(c2.data(), pk2_comp, 33);
    const auto kctx = secp256k1::musig2_key_agg({c1, c2});

    unsigned char pn1_66[66], pn2_66[66];
    check(secp256k1_musig_pubnonce_serialize(ctx, pn1_66, &pn1) == 1, "[negated-key-term] pubnonce1 serialize");
    check(secp256k1_musig_pubnonce_serialize(ctx, pn2_66, &pn2) == 1, "[negated-key-term] pubnonce2 serialize");
    std::array<std::uint8_t, 66> pn1_arr{}, pn2_arr{};
    std::memcpy(pn1_arr.data(), pn1_66, 66);
    std::memcpy(pn2_arr.data(), pn2_66, 66);
    const auto cpn1 = secp256k1::MuSig2PubNonce::deserialize(pn1_arr);
    const auto cpn2 = secp256k1::MuSig2PubNonce::deserialize(pn2_arr);
    const auto cagg = secp256k1::musig2_nonce_agg({cpn1, cpn2});
    std::array<std::uint8_t, 32> m{};
    std::memcpy(m.data(), msg, 32);
    const auto csess = secp256k1::musig2_start_sign_session(cagg, kctx, m);

    using secp256k1::fast::Scalar;
    const Scalar s1 = Scalar::from_bytes(ps1.data);
    Scalar ea = csess.e * kctx.key_coefficients[0];
    if (kctx.Q_negated) ea = ea.negate();
    ea = ea * kctx.gacc;
    const Scalar key_term = ea * Scalar::from_bytes(kSk1);
    // honest s1 = k + key_term  ->  forged = k - key_term
    const Scalar forged = s1 - key_term - key_term;

    std::array<std::uint8_t, 32> pk1_x{};
    std::memcpy(pk1_x.data(), pk1_comp + 1, 32);

    // Core: the 33-byte overload is exact (BIP-327); the x-only overload is
    // parity-lenient by contract and still accepts the forgery.
    check(secp256k1::musig2_partial_verify(s1, cpn1, c1, kctx, csess, 0),
          "[negated-key-term] core 33-byte verify accepts the honest partial sig");
    check(!secp256k1::musig2_partial_verify(forged, cpn1, c1, kctx, csess, 0),
          "[negated-key-term] core 33-byte verify rejects the negated-key-term partial sig");
    check(secp256k1::musig2_partial_verify(forged, cpn1, pk1_x, kctx, csess, 0),
          "[negated-key-term] core x-only verify is parity-lenient (documented)");

    // Shim: must match upstream libsecp256k1 and return 0 for the forgery.
    secp256k1_musig_partial_sig forged_ps = ps1;
    const auto forged_bytes = forged.to_bytes();
    std::memcpy(forged_ps.data, forged_bytes.data(), 32);
    check(secp256k1_musig_partial_sig_verify(ctx, &ps1, &pn1, &pub1, &ka, &sess) == 1,
          "[negated-key-term] shim verify accepts the honest partial sig");
    check(secp256k1_musig_partial_sig_verify(ctx, &ps2, &pn2, &pub2, &ka, &sess) == 1,
          "[negated-key-term] shim verify accepts the second honest partial sig");
    check(secp256k1_musig_partial_sig_verify(ctx, &forged_ps, &pn1, &pub1, &ka, &sess) == 0,
          "[negated-key-term] shim verify rejects the negated-key-term partial sig");

    // The forgery really breaks the aggregate: this is why it must be blamed.
    const secp256k1_musig_partial_sig* bad[2] = {&forged_ps, &ps2};
    unsigned char bad_sig[64];
    if (secp256k1_musig_partial_sig_agg(ctx, bad_sig, &sess, bad, 2) == 1) {
        check(secp256k1_schnorrsig_verify(ctx, bad_sig, msg, 32, &agg_pk) == 0,
              "[negated-key-term] aggregate with the forged partial sig is invalid");
    }

    secp256k1_context_destroy(ctx);
}

int test_regression_musig2_session_token_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_musig2_session_token] token-keyed map — no address-reuse hazard\n");
    test_token_written();
    test_distinct_tokens();
    test_struct_reuse_fresh_token();
    test_2of2_roundtrip();
    test_negated_key_term_rejected();
    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_musig2_session_token_run(); }
#endif
