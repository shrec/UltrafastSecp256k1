// =============================================================================
// REGRESSION TEST: musig2_partial_verify bugs (MVV-1..MVV-6)
// =============================================================================
//
// BACKGROUND:
//   Two bugs were found and fixed in musig2_partial_verify() in
//   cpu/src/musig2.cpp (commit after 2026-04-23):
//
//   BUG-1 — Missing bounds check on signer_index:
//     musig2_partial_sign() rejected out-of-range signer_index with a guard:
//       if (signer_index >= key_agg_ctx.key_coefficients.size()) return Scalar::zero();
//     musig2_partial_verify() had NO such guard, so calling it with
//     signer_index >= n_signers caused undefined behavior (out-of-bounds
//     std::vector access).  The C ABI wrapper (ufsecp_musig2_partial_verify)
//     did have the check, but the raw C++ API did not.
//
//   BUG-2 — Missing infinity-point check on public nonce:
//     After decompress_point(pub_nonce.R1) and decompress_point(pub_nonce.R2),
//     there was no check for is_infinity().  BIP-327 §4 PartialSigVerify
//     requires "fail if [nonce deserialization] fails", which includes the
//     infinity point.  A caller passing an all-zero nonce (or any nonce
//     whose decompression fails) would cause the verify to run with the
//     identity point, potentially accepting invalid partial signatures.
//
//   FIXES applied:
//     - Added bounds check identical to musig2_partial_sign: return false on OOB.
//     - Added R1_i.is_infinity() || R2_i.is_infinity() guard after decompression.
//
// HOW THIS TEST CATCHES THE BUGS:
//   MVV-1  signer_index == n_signers (exactly OOB) must return false.
//   MVV-2  signer_index == SIZE_MAX must return false.
//   MVV-3  pub_nonce with prefix byte 0x00 (infinity encoding) must return false.
//   MVV-4  pub_nonce with r-bytes 0x00*32 (point not on curve) must return false.
//   MVV-5  Legitimate 2-of-2 round-trip still produces verified partial sigs.
//   MVV-6  Legitimate 3-of-3 round-trip still produces verified partial sigs.
//
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <vector>

#include "secp256k1/musig2.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "audit_check.hpp"

#ifndef STANDALONE_TEST
// Entry point for unified audit runner
int test_regression_musig2_verify_run();
#endif

static int g_pass = 0;
static int g_fail = 0;

static void record(bool cond, const char* desc) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  FAIL: %s\n", desc);
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using Scalar = secp256k1::fast::Scalar;

// Fixed private keys for deterministic tests
static const std::array<uint8_t, 32> kPriv1 = {
    0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef,
    0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef,
    0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0xef,
    0x01,0x23,0x45,0x67, 0x89,0xab,0xcd,0x01
};
static const std::array<uint8_t, 32> kPriv2 = {
    0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10,
    0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10,
    0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x10,
    0xfe,0xdc,0xba,0x98, 0x76,0x54,0x32,0x02
};
static const std::array<uint8_t, 32> kPriv3 = {
    0xaa,0xbb,0xcc,0xdd, 0xee,0xff,0x00,0x11,
    0x22,0x33,0x44,0x55, 0x66,0x77,0x88,0x99,
    0xaa,0xbb,0xcc,0xdd, 0xee,0xff,0x00,0x11,
    0x22,0x33,0x44,0x55, 0x66,0x77,0x88,0x03
};
static const std::array<uint8_t, 32> kMsg = {
    0xde,0xad,0xbe,0xef, 0xca,0xfe,0xba,0xbe,
    0x01,0x02,0x03,0x04, 0x05,0x06,0x07,0x08,
    0x11,0x22,0x33,0x44, 0x55,0x66,0x77,0x88,
    0x99,0xaa,0xbb,0xcc, 0xdd,0xee,0xff,0x00
};

static std::array<uint8_t, 32> xonly_from_priv(const std::array<uint8_t, 32>& priv_bytes) {
    auto sk = Scalar::from_bytes(priv_bytes);
    auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    return kp.px;
}

// Run a full n-of-n MuSig2 round-trip; returns true if final Schnorr verify passes.
static bool musig2_round_trip(
    const std::vector<std::array<uint8_t, 32>>& privs,
    const std::array<uint8_t, 32>& msg,
    std::vector<bool>* partial_ok_out = nullptr)
{
    std::size_t const n = privs.size();

    // Build xonly pubkeys
    std::vector<std::array<uint8_t, 32>> pks;
    for (auto const& p : privs) pks.push_back(xonly_from_priv(p));

    // Key aggregation
    auto key_agg = secp256k1::musig2_key_agg(pks);
    if (key_agg.Q.is_infinity()) return false;

    // Nonce generation
    std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
    std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
    for (std::size_t i = 0; i < n; ++i) {
        auto sk = Scalar::from_bytes(privs[i]);
        auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pks[i], key_agg.Q_x, msg, nullptr);
        sec_nonces.push_back(sec);
        pub_nonces.push_back(pub);
    }

    // Nonce aggregation
    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);

    // Start signing session
    auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, msg);

    // Partial signing + verification
    std::vector<Scalar> partial_sigs;
    for (std::size_t i = 0; i < n; ++i) {
        auto sk = Scalar::from_bytes(privs[i]);
        auto si = secp256k1::musig2_partial_sign(sec_nonces[i], sk, key_agg, session, i);
        partial_sigs.push_back(si);

        bool const pv = secp256k1::musig2_partial_verify(si, pub_nonces[i], pks[i], key_agg, session, i);
        if (partial_ok_out) partial_ok_out->push_back(pv);
    }

    // Aggregate and final verify
    auto sig64 = secp256k1::musig2_partial_sig_agg(partial_sigs, session);
    auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig64);
    return secp256k1::schnorr_verify(key_agg.Q_x, msg, schnorr_sig);
}

// ---------------------------------------------------------------------------
// MVV-1: OOB signer_index (== n_signers) must return false
// ---------------------------------------------------------------------------
static void test_mvv1_oob_signer_index_exact() {
    std::printf("  [MVV-1] musig2_partial_verify: signer_index == n_signers -> false\n");

    std::vector<std::array<uint8_t, 32>> privs = {kPriv1, kPriv2};
    std::vector<std::array<uint8_t, 32>> pks;
    for (auto const& p : privs) pks.push_back(xonly_from_priv(p));

    auto key_agg = secp256k1::musig2_key_agg(pks);

    std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
    std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
    for (std::size_t i = 0; i < 2; ++i) {
        auto sk = Scalar::from_bytes(privs[i]);
        auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pks[i], key_agg.Q_x, kMsg, nullptr);
        sec_nonces.push_back(sec);
        pub_nonces.push_back(pub);
    }

    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
    auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, kMsg);

    auto sk0 = Scalar::from_bytes(privs[0]);
    auto sig0 = secp256k1::musig2_partial_sign(sec_nonces[0], sk0, key_agg, session, 0);

    // signer_index == 2 (n_signers) is out of bounds
    bool const bad = secp256k1::musig2_partial_verify(sig0, pub_nonces[0], pks[0], key_agg, session, 2);
    record(!bad, "OOB signer_index == n_signers returns false");
}

// ---------------------------------------------------------------------------
// MVV-2: OOB signer_index (== SIZE_MAX) must return false
// ---------------------------------------------------------------------------
static void test_mvv2_oob_signer_index_max() {
    std::printf("  [MVV-2] musig2_partial_verify: signer_index == SIZE_MAX -> false\n");

    std::vector<std::array<uint8_t, 32>> privs = {kPriv1, kPriv2};
    std::vector<std::array<uint8_t, 32>> pks;
    for (auto const& p : privs) pks.push_back(xonly_from_priv(p));

    auto key_agg = secp256k1::musig2_key_agg(pks);

    std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
    std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
    for (std::size_t i = 0; i < 2; ++i) {
        auto sk = Scalar::from_bytes(privs[i]);
        auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pks[i], key_agg.Q_x, kMsg, nullptr);
        sec_nonces.push_back(sec);
        pub_nonces.push_back(pub);
    }

    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
    auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, kMsg);

    auto sk0 = Scalar::from_bytes(privs[0]);
    auto sig0 = secp256k1::musig2_partial_sign(sec_nonces[0], sk0, key_agg, session, 0);

    constexpr std::size_t kMax = static_cast<std::size_t>(-1);
    bool const bad = secp256k1::musig2_partial_verify(sig0, pub_nonces[0], pks[0], key_agg, session, kMax);
    record(!bad, "OOB signer_index == SIZE_MAX returns false");
}

// ---------------------------------------------------------------------------
// MVV-3: pub_nonce with prefix 0x00 (infinity / invalid encoding) -> false
// ---------------------------------------------------------------------------
static void test_mvv3_infinity_nonce_prefix() {
    std::printf("  [MVV-3] musig2_partial_verify: infinity nonce (prefix 0x00) -> false\n");

    std::vector<std::array<uint8_t, 32>> privs = {kPriv1, kPriv2};
    std::vector<std::array<uint8_t, 32>> pks;
    for (auto const& p : privs) pks.push_back(xonly_from_priv(p));

    auto key_agg = secp256k1::musig2_key_agg(pks);

    std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
    std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
    for (std::size_t i = 0; i < 2; ++i) {
        auto sk = Scalar::from_bytes(privs[i]);
        auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pks[i], key_agg.Q_x, kMsg, nullptr);
        sec_nonces.push_back(sec);
        pub_nonces.push_back(pub);
    }

    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
    auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, kMsg);

    auto sk0 = Scalar::from_bytes(privs[0]);
    auto sig0 = secp256k1::musig2_partial_sign(sec_nonces[0], sk0, key_agg, session, 0);

    // Craft a pub_nonce with invalid prefix (0x00 = not a valid compressed point)
    secp256k1::MuSig2PubNonce bad_nonce = pub_nonces[0];
    bad_nonce.R1[0] = 0x00; // breaks decompression -> infinity

    bool const bad = secp256k1::musig2_partial_verify(sig0, bad_nonce, pks[0], key_agg, session, 0);
    record(!bad, "infinity nonce R1 (prefix 0x00) rejected");

    // Also test bad R2
    secp256k1::MuSig2PubNonce bad_nonce2 = pub_nonces[0];
    bad_nonce2.R2[0] = 0x00;
    bool const bad2 = secp256k1::musig2_partial_verify(sig0, bad_nonce2, pks[0], key_agg, session, 0);
    record(!bad2, "infinity nonce R2 (prefix 0x00) rejected");
}

// ---------------------------------------------------------------------------
// MVV-4: pub_nonce with all-zero x-bytes (not on curve) -> false
// ---------------------------------------------------------------------------
static void test_mvv4_zero_x_nonce() {
    std::printf("  [MVV-4] musig2_partial_verify: nonce with x=0 (not on curve) -> false\n");

    std::vector<std::array<uint8_t, 32>> privs = {kPriv1, kPriv2};
    std::vector<std::array<uint8_t, 32>> pks;
    for (auto const& p : privs) pks.push_back(xonly_from_priv(p));

    auto key_agg = secp256k1::musig2_key_agg(pks);

    std::vector<secp256k1::MuSig2SecNonce> sec_nonces;
    std::vector<secp256k1::MuSig2PubNonce> pub_nonces;
    for (std::size_t i = 0; i < 2; ++i) {
        auto sk = Scalar::from_bytes(privs[i]);
        auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pks[i], key_agg.Q_x, kMsg, nullptr);
        sec_nonces.push_back(sec);
        pub_nonces.push_back(pub);
    }

    auto agg_nonce = secp256k1::musig2_nonce_agg(pub_nonces);
    auto session = secp256k1::musig2_start_sign_session(agg_nonce, key_agg, kMsg);

    auto sk0 = Scalar::from_bytes(privs[0]);
    auto sig0 = secp256k1::musig2_partial_sign(sec_nonces[0], sk0, key_agg, session, 0);

    // x=0 is not on the curve (y^2 = 7, not a quadratic residue mod p)
    secp256k1::MuSig2PubNonce bad_nonce = pub_nonces[0];
    bad_nonce.R1[0] = 0x02;
    std::memset(bad_nonce.R1.data() + 1, 0x00, 32); // x = 0

    bool const bad = secp256k1::musig2_partial_verify(sig0, bad_nonce, pks[0], key_agg, session, 0);
    record(!bad, "nonce with x=0 (not on curve) rejected");
}

// ---------------------------------------------------------------------------
// MVV-5: Legitimate 2-of-2 round-trip must still work after the fixes
// ---------------------------------------------------------------------------
static void test_mvv5_roundtrip_2of2() {
    std::printf("  [MVV-5] musig2_partial_verify: 2-of-2 round-trip correct after fix\n");

    std::vector<std::array<uint8_t, 32>> privs = {kPriv1, kPriv2};
    std::vector<bool> partial_ok;
    bool const final_ok = musig2_round_trip(privs, kMsg, &partial_ok);

    record(partial_ok.size() == 2, "got 2 partial_verify results");
    for (std::size_t i = 0; i < partial_ok.size(); ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "signer %zu partial sig verifies (2-of-2)", i);
        record(partial_ok[i], buf);
    }
    record(final_ok, "final 2-of-2 Schnorr sig verifies");
}

// ---------------------------------------------------------------------------
// MVV-6: Legitimate 3-of-3 round-trip must still work after the fixes
// ---------------------------------------------------------------------------
static void test_mvv6_roundtrip_3of3() {
    std::printf("  [MVV-6] musig2_partial_verify: 3-of-3 round-trip correct after fix\n");

    std::vector<std::array<uint8_t, 32>> privs = {kPriv1, kPriv2, kPriv3};
    std::vector<bool> partial_ok;
    bool const final_ok = musig2_round_trip(privs, kMsg, &partial_ok);

    record(partial_ok.size() == 3, "got 3 partial_verify results");
    for (std::size_t i = 0; i < partial_ok.size(); ++i) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "signer %zu partial sig verifies (3-of-3)", i);
        record(partial_ok[i], buf);
    }
    record(final_ok, "final 3-of-3 Schnorr sig verifies");
}

// ---------------------------------------------------------------------------
// Runner
// ---------------------------------------------------------------------------

int test_regression_musig2_verify_run() {
    g_pass = 0;
    g_fail = 0;

    std::printf("[regression_musig2_verify] musig2_partial_verify bug regression\n");
    std::printf("  BUG-1: missing signer_index bounds check in musig2_partial_verify\n");
    std::printf("  BUG-2: missing infinity check for R1_i / R2_i after decompress_point\n\n");

    test_mvv1_oob_signer_index_exact();
    test_mvv2_oob_signer_index_max();
    test_mvv3_infinity_nonce_prefix();
    test_mvv4_zero_x_nonce();
    test_mvv5_roundtrip_2of2();
    test_mvv6_roundtrip_3of3();

    std::printf("\n  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_musig2_verify_run();
}
#endif
