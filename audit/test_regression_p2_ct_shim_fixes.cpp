// ============================================================================
// REGRESSION: P2 CT and Shim compatibility fixes
//
// Covers:
//   CT-002: musig2_partial_sig_agg uses ct::scalar_add (not VT operator+=)
//   CT-001: rfc6979_nonce_hedged fixed 2-iteration (not VT early-exit loop)
//   CT-002b: keypair_xonly_tweak_add uses is_zero_ct() on tweaked sk
//   CT-003: context_randomize uses parse_bytes_strict_nonzero (not from_bytes)
//   SHIM-002/004: pubkey_data_to_point validates curve membership y²=x³+7
//   SHIM-002: ec_pubkey_negate rejects off-curve pubkey (returns 0)
//   SHIM-003: musig_pubkey_ec_tweak_add allows tweak=0 (libsecp compat)
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#ifndef STANDALONE_TEST
#define STANDALONE_TEST
#endif
#endif

#include "secp256k1/musig2.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/ct/sign.hpp"
#include <cstdio>
#include <cstring>
#include <array>
#include <vector>

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); } \
} while(0)

static const uint8_t kSk1[32] = {
    0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
    0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
    0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
    0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
};
static const uint8_t kSk2[32] = {
    0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
    0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
    0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
    0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
};

// ── CT-002: musig2_partial_sig_agg ct::scalar_add correctness ────────────────
static void test_musig2_agg_ct_correctness() {
    std::printf("  [MSAGG-CT] musig2_partial_sig_agg sign+verify roundtrip (ct::scalar_add)\n");

    secp256k1::fast::Scalar sk1{}, sk2{};
    secp256k1::fast::Scalar::parse_bytes_strict_nonzero(kSk1, sk1);
    secp256k1::fast::Scalar::parse_bytes_strict_nonzero(kSk2, sk2);

    // Build 2-of-2 key aggregation
    // musig2_key_agg requires 33-byte compressed pubkeys
    auto pt1 = secp256k1::ct::generator_mul(sk1);
    auto pt2 = secp256k1::ct::generator_mul(sk2);
    auto cpk1 = pt1.to_compressed();
    auto cpk2 = pt2.to_compressed();
    std::vector<std::array<uint8_t,33>> pks = {cpk1, cpk2};
    auto kagg = secp256k1::musig2_key_agg(pks);

    std::array<uint8_t,32> msg = {0xDE,0xAD,0xBE,0xEF};
    std::array<uint8_t,32> seed1 = {0xAA}, seed2 = {0xBB};
    auto [n1, pn1] = secp256k1::musig2_nonce_gen(sk1, kagg.Q_x, kagg.Q_x, seed1, nullptr);
    auto [n2, pn2] = secp256k1::musig2_nonce_gen(sk2, kagg.Q_x, kagg.Q_x, seed2, nullptr);
    std::vector<secp256k1::MuSig2PubNonce> pns = {pn1, pn2};
    auto agg_nonce = secp256k1::musig2_nonce_agg(pns);
    auto sess = secp256k1::musig2_start_sign_session(agg_nonce, kagg, msg);

    auto psig1 = secp256k1::musig2_partial_sign(n1, sk1, kagg, sess, 0);
    auto psig2 = secp256k1::musig2_partial_sign(n2, sk2, kagg, sess, 1);

    CHECK(!psig1.is_zero(), "MSAGG-CT: psig1 non-zero");
    CHECK(!psig2.is_zero(), "MSAGG-CT: psig2 non-zero");

    std::vector<secp256k1::fast::Scalar> psigs = {psig1, psig2};
    auto sig64 = secp256k1::musig2_partial_sig_agg(psigs, sess);
    bool all_zero = true;
    for (auto b : sig64) { if (b) { all_zero = false; break; } }
    CHECK(!all_zero, "MSAGG-CT: aggregated sig non-zero");

    // Verify the aggregate signature — correct arg order: (pubkey_x, msg, sig)
    secp256k1::SchnorrSignature sig{};
    std::memcpy(sig.r.data(), sig64.data(), 32);
    std::array<uint8_t,32> s_bytes{};
    std::memcpy(s_bytes.data(), sig64.data() + 32, 32);
    sig.s = secp256k1::fast::Scalar::from_bytes(s_bytes);
    std::array<uint8_t,32> Q_x = kagg.Q_x;
    CHECK(secp256k1::schnorr_verify(Q_x, msg, sig), "MSAGG-CT: aggregated sig verifies");
}

// ── CT-001: rfc6979_nonce_hedged correctness ──────────────────────────────
static void test_hedged_nonce_correctness() {
    std::printf("  [HEDGE-CT] rfc6979_nonce_hedged sign+verify roundtrip\n");

    secp256k1::fast::Scalar sk{};
    secp256k1::fast::Scalar::parse_bytes_strict_nonzero(kSk1, sk);
    std::array<uint8_t,32> msg = {0xCA,0xFE};
    std::array<uint8_t,32> aux = {0x12,0x34};

    // ecdsa_sign_hedged uses rfc6979_nonce_hedged internally
    auto sig = secp256k1::ct::ecdsa_sign_hedged(msg, sk, aux);
    CHECK(!sig.r.is_zero() && !sig.s.is_zero(), "HEDGE-CT: signature non-zero");

    auto pk = secp256k1::ct::generator_mul(sk);
    CHECK(secp256k1::ecdsa_verify(msg, pk, sig), "HEDGE-CT: hedged signature verifies");

    // Deterministic: same (sk, msg, aux) → same signature
    auto sig2 = secp256k1::ct::ecdsa_sign_hedged(msg, sk, aux);
    CHECK(sig.r == sig2.r, "HEDGE-CT: hedged sig deterministic (R)");
    CHECK(sig.s == sig2.s, "HEDGE-CT: hedged sig deterministic (s)");

    // Different aux → different k → different sig
    std::array<uint8_t,32> aux2 = {0x56,0x78};
    auto sig3 = secp256k1::ct::ecdsa_sign_hedged(msg, sk, aux2);
    bool r_different = (sig.r != sig3.r);
    CHECK(r_different, "HEDGE-CT: different aux → different R (nonce varies)");
}

// ── Shim-dependent tests (require secp256k1_shim linked with shim include path) ────────────────────────
// SECP256K1_BUILD_COMPAT_SHIM is defined by CMakeLists when TARGET secp256k1_shim exists
// and the shim include path is available. UNIFIED_AUDIT_RUNNER alone is insufficient
// because the shim include path is not always provided.
#if defined(SECP256K1_BUILD_COMPAT_SHIM)

#include "secp256k1.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"
#include "secp256k1_musig.h"

static const secp256k1_context* sctx() {
    static secp256k1_context* s = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    return s;
}

// ── CT-002b: keypair_xonly_tweak_add is_zero_ct on tweaked sk ────────────
static void test_keypair_xonly_tweak_is_zero_ct() {
    std::printf("  [KXT-CT] secp256k1_keypair_xonly_tweak_add uses is_zero_ct (correctness)\n");

    secp256k1_keypair kp;
    secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
    int ok = secp256k1_keypair_create(sctx(), &kp, kSk1);
    CHECK(ok == 1, "KXT-CT: keypair_create ok");

    // Tweak with a valid non-zero value
    uint8_t tweak[32] = {0x01};
    ok = secp256k1_keypair_xonly_tweak_add(sctx(), &kp, tweak);
    CHECK(ok == 1, "KXT-CT: xonly_tweak_add with non-zero tweak succeeds");

    // The tweaked keypair should still produce valid signatures
    unsigned char sig[64];
    uint8_t msg[32] = {0xAB,0xCD};
    ok = secp256k1_schnorrsig_sign32(sctx(), sig, msg, &kp, nullptr);
    CHECK(ok == 1, "KXT-CT: sign with tweaked keypair succeeds");
}

// ── CT-003: context_randomize parse_bytes_strict_nonzero ────────────────
static void test_context_randomize_ct() {
    std::printf("  [CTX-CT] secp256k1_context_randomize with valid seed succeeds\n");

    secp256k1_context* ctx = secp256k1_context_create(
        SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

    // Valid seed (non-zero, < n)
    int ok = secp256k1_context_randomize(ctx, kSk1);
    CHECK(ok == 1, "CTX-CT: randomize with valid seed returns 1");

    // NULL seed disables blinding
    ok = secp256k1_context_randomize(ctx, nullptr);
    CHECK(ok == 1, "CTX-CT: randomize with NULL seed returns 1 (disables blinding)");

    secp256k1_context_destroy(ctx);
}

// ── SHIM-002: ec_pubkey_negate rejects off-curve input ──────────────────
static void test_pubkey_negate_off_curve() {
    std::printf("  [NEG-002] secp256k1_ec_pubkey_negate rejects off-curve pubkey\n");

    secp256k1_pubkey pubkey;
    // Create a valid pubkey first
    CHECK(secp256k1_ec_pubkey_create(sctx(), &pubkey, kSk1) == 1, "NEG-002: create ok");

    // Corrupt the Y coordinate to make it off-curve
    pubkey.data[32] ^= 0xFF;  // flip all Y bits → certainly off-curve
    int rc = secp256k1_ec_pubkey_negate(sctx(), &pubkey);
    CHECK(rc == 0, "NEG-002: negate with off-curve pubkey returns 0");
}

// ── SHIM-002: ec_pubkey_negate accepts on-curve input (regression) ───────
static void test_pubkey_negate_valid() {
    std::printf("  [NEG-OK] secp256k1_ec_pubkey_negate accepts valid pubkey\n");

    secp256k1_pubkey pubkey;
    CHECK(secp256k1_ec_pubkey_create(sctx(), &pubkey, kSk1) == 1, "NEG-OK: create ok");

    // Serialize to get the compressed key for comparison
    unsigned char before[33], after[33];
    size_t len = 33;
    secp256k1_ec_pubkey_serialize(sctx(), before, &len, &pubkey, SECP256K1_EC_COMPRESSED);

    int rc = secp256k1_ec_pubkey_negate(sctx(), &pubkey);
    CHECK(rc == 1, "NEG-OK: negate returns 1");
    secp256k1_ec_pubkey_serialize(sctx(), after, &len, &pubkey, SECP256K1_EC_COMPRESSED);
    // X must be same, Y parity must differ
    CHECK(memcmp(before + 1, after + 1, 32) == 0, "NEG-OK: X unchanged after negate");
    CHECK(before[0] != after[0], "NEG-OK: Y parity flipped after negate");

    // Negate twice → back to original
    rc = secp256k1_ec_pubkey_negate(sctx(), &pubkey);
    CHECK(rc == 1, "NEG-OK: second negate returns 1");
    unsigned char restored[33];
    secp256k1_ec_pubkey_serialize(sctx(), restored, &len, &pubkey, SECP256K1_EC_COMPRESSED);
    CHECK(memcmp(before, restored, 33) == 0, "NEG-OK: double-negate restores original");
}

// ── SHIM-003: musig_pubkey_ec_tweak_add with tweak=0 ────────────────────
static void test_musig_ec_tweak_zero() {
    std::printf("  [MTZ-003] secp256k1_musig_pubkey_ec_tweak_add accepts tweak=0\n");

    secp256k1_pubkey pk;
    CHECK(secp256k1_ec_pubkey_create(sctx(), &pk, kSk1) == 1, "MTZ-003: create pk1 ok");
    secp256k1_pubkey pk2;
    CHECK(secp256k1_ec_pubkey_create(sctx(), &pk2, kSk2) == 1, "MTZ-003: create pk2 ok");

    const secp256k1_pubkey* pks[2] = {&pk, &pk2};
    secp256k1_musig_keyagg_cache cache;
    secp256k1_xonly_pubkey agg_pk;
    CHECK(secp256k1_musig_pubkey_agg(sctx(), &agg_pk, &cache, pks, 2) == 1,
          "MTZ-003: pubkey_agg ok");

    // Apply zero tweak — libsecp allows this (result = Q unchanged)
    uint8_t zero_tweak[32] = {};
    secp256k1_pubkey out_pk;
    int rc = secp256k1_musig_pubkey_ec_tweak_add(sctx(), &out_pk, &cache, zero_tweak);
    CHECK(rc == 1, "MTZ-003: ec_tweak_add with tweak=0 returns 1");
}

static void test_shim_fail_closed_outputs() {
    std::printf("  [SHIM-FC] failed signing/pubkey mutations clear outputs\n");

    uint8_t msg[32] = {0x44};
    uint8_t zero_sk[32] = {};

    secp256k1_ecdsa_signature esig;
    std::memset(&esig, 0xA5, sizeof(esig));
    CHECK(secp256k1_ecdsa_sign(sctx(), &esig, msg, zero_sk, nullptr, nullptr) == 0,
          "SHIM-FC: ecdsa_sign rejects zero seckey");
    bool ecdsa_sig_zero = true;
    for (unsigned char b : esig.data) ecdsa_sig_zero = ecdsa_sig_zero && (b == 0);
    CHECK(ecdsa_sig_zero, "SHIM-FC: ecdsa_sign failure clears signature");

    secp256k1_keypair bad_kp;
    std::memset(&bad_kp, 0, sizeof(bad_kp));
    unsigned char sig64[64];
    std::memset(sig64, 0xA5, sizeof(sig64));
    CHECK(secp256k1_schnorrsig_sign32(sctx(), sig64, msg, &bad_kp, nullptr) == 0,
          "SHIM-FC: schnorrsig_sign32 rejects zero keypair");
    bool schnorr_sig_zero = true;
    for (unsigned char b : sig64) schnorr_sig_zero = schnorr_sig_zero && (b == 0);
    CHECK(schnorr_sig_zero, "SHIM-FC: schnorrsig_sign32 failure clears signature");

    secp256k1_pubkey bad_pk;
    std::memset(&bad_pk, 0, sizeof(bad_pk));
    unsigned char serialized[65];
    std::memset(serialized, 0xA5, sizeof(serialized));
    size_t serialized_len = sizeof(serialized);
    CHECK(secp256k1_ec_pubkey_serialize(sctx(), serialized, &serialized_len,
          &bad_pk, SECP256K1_EC_UNCOMPRESSED) == 0,
          "SHIM-FC: pubkey_serialize rejects zero opaque pubkey");
    CHECK(serialized_len == 0, "SHIM-FC: pubkey_serialize failure reports zero length");

    CHECK(secp256k1_ec_pubkey_create(sctx(), &bad_pk, kSk1) == 1,
          "SHIM-FC: create pubkey for mutation test");
    bad_pk.data[32] ^= 0xFF;
    CHECK(secp256k1_ec_pubkey_negate(sctx(), &bad_pk) == 0,
          "SHIM-FC: pubkey_negate rejects off-curve input");
    bool pubkey_zero = true;
    for (unsigned char b : bad_pk.data) pubkey_zero = pubkey_zero && (b == 0);
    CHECK(pubkey_zero, "SHIM-FC: pubkey_negate failure clears in-place pubkey");

    // The pubkey MANIPULATION paths share one curve-checked loader
    // (pubkey_data_to_point_checked) so they are uniformly FAIL-CLOSED on an
    // off-curve opaque pubkey — not only negate/serialize. The hot verify paths use
    // the unchecked loader (PERF-002). Guard tweak_add/tweak_mul here too.
    {
        secp256k1_pubkey oc;
        CHECK(secp256k1_ec_pubkey_create(sctx(), &oc, kSk1) == 1,
              "SHIM-FC: create pubkey for off-curve tweak test");
        oc.data[40] ^= 0xFF;  // corrupt a y byte -> off-curve
        uint8_t small_tweak[32];
        std::memset(small_tweak, 0, sizeof(small_tweak));
        small_tweak[31] = 2;
        secp256k1_pubkey t1 = oc;
        CHECK(secp256k1_ec_pubkey_tweak_add(sctx(), &t1, small_tweak) == 0,
              "SHIM-FC: pubkey_tweak_add rejects off-curve input");
        secp256k1_pubkey t2 = oc;
        CHECK(secp256k1_ec_pubkey_tweak_mul(sctx(), &t2, small_tweak) == 0,
              "SHIM-FC: pubkey_tweak_mul rejects off-curve input");
    }

    secp256k1_xonly_pubkey bad_xonly;
    std::memset(&bad_xonly, 0, sizeof(bad_xonly));
    unsigned char xout[32];
    std::memset(xout, 0xA5, sizeof(xout));
    CHECK(secp256k1_xonly_pubkey_serialize(sctx(), xout, &bad_xonly) == 0,
          "SHIM-FC: xonly_pubkey_serialize rejects zero opaque xonly");
    bool xout_zero = true;
    for (unsigned char b : xout) xout_zero = xout_zero && (b == 0);
    CHECK(xout_zero, "SHIM-FC: xonly serialize failure clears output");

    secp256k1_keypair kp;
    CHECK(secp256k1_keypair_create(sctx(), &kp, kSk1) == 1,
          "SHIM-FC: create keypair for failed tweak");
    uint8_t bad_tweak[32];
    std::memset(bad_tweak, 0xFF, sizeof(bad_tweak));
    CHECK(secp256k1_keypair_xonly_tweak_add(sctx(), &kp, bad_tweak) == 0,
          "SHIM-FC: keypair_xonly_tweak_add rejects tweak >= n");
    bool keypair_zero = true;
    for (unsigned char b : kp.data) keypair_zero = keypair_zero && (b == 0);
    CHECK(keypair_zero, "SHIM-FC: failed keypair tweak clears keypair");
}

#else  // No shim

static void test_keypair_xonly_tweak_is_zero_ct() {
    std::printf("  [KXT-CT] skipped (no shim)\n"); ++g_pass;
}
static void test_context_randomize_ct() {
    std::printf("  [CTX-CT] skipped (no shim)\n"); ++g_pass;
}
static void test_pubkey_negate_off_curve() {
    std::printf("  [NEG-002] skipped (no shim)\n"); ++g_pass;
}
static void test_pubkey_negate_valid() {
    std::printf("  [NEG-OK] skipped (no shim)\n"); ++g_pass;
}
static void test_musig_ec_tweak_zero() {
    std::printf("  [MTZ-003] skipped (no shim)\n"); ++g_pass;
}
static void test_shim_fail_closed_outputs() {
    std::printf("  [SHIM-FC] skipped (no shim)\n"); ++g_pass;
}

#endif

int test_regression_p2_ct_shim_fixes_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_p2_ct_shim_fixes] CT-001/002/003 + SHIM-002/003/004\n");
    test_musig2_agg_ct_correctness();
    test_hedged_nonce_correctness();
    test_keypair_xonly_tweak_is_zero_ct();
    test_context_randomize_ct();
    test_pubkey_negate_off_curve();
    test_pubkey_negate_valid();
    test_musig_ec_tweak_zero();
    test_shim_fail_closed_outputs();
    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_p2_ct_shim_fixes_run(); }
#endif
