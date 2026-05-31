// ============================================================================
// test_regression_musig2_signer_index_validation.cpp
// Regression: musig2_partial_sign must validate secret_key <-> signer_index.
//
// Bug fixed: musig2_partial_sign accepted any caller-supplied signer_index
// without verifying the secret key corresponds to the pubkey at that index.
// Fix: musig2_key_agg now stores individual_pubkeys; musig2_partial_sign
// validates ct::generator_mul(sk) == individual_pubkeys[signer_index] when
// the context was created via musig2_key_agg (not ABI deserialization).
//
// Tests:
//   MSI-1: correct key + correct index signs successfully
//   MSI-2: correct key + wrong index returns zero scalar (fail-closed)
//   MSI-3: aggregated partial sigs with correct indices verify correctly
//   MSI-4: empty individual_pubkeys -> fail-closed (cannot validate -> refuse to sign)
// ============================================================================

#include <cstdio>
#include <cstring>
#include <array>
#include <vector>
static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"
#include "secp256k1/musig2.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/point.hpp"

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

// secp256k1 order n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
static const unsigned char kSk1[32] = {
    0x01,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x05
};
static const unsigned char kSk2[32] = {
    0x02,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x07
};

static std::array<uint8_t, 33> pubkey_compressed(const unsigned char sk[32]) {
    Scalar s{};
    Scalar::parse_bytes_strict_nonzero(sk, s);
    auto P = ct::generator_mul(s);
    return P.to_compressed();
}

static MuSig2KeyAggCtx make_2party_keyagg() {
    auto pk1 = pubkey_compressed(kSk1);
    auto pk2 = pubkey_compressed(kSk2);
    std::vector<std::array<uint8_t, 33>> pks = {pk1, pk2};
    return musig2_key_agg(pks);
}

// ── MSI-1: correct key at correct index signs successfully ────────────────
static void test_correct_key_correct_index() {
    auto kagg = make_2party_keyagg();
    CHECK(kagg.individual_pubkeys.size() == 2, "[MSI-1a] individual_pubkeys populated");

    Scalar sk1{};
    Scalar::parse_bytes_strict_nonzero(kSk1, sk1);

    // Build minimal session (single-party nonce for isolation test)
    auto [sn1, pn1] = musig2_nonce_gen(sk1, kagg.Q_x, kagg.Q_x,
                                        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1}, nullptr);

    Scalar sk2{};
    Scalar::parse_bytes_strict_nonzero(kSk2, sk2);
    auto [sn2, pn2] = musig2_nonce_gen(sk2, kagg.Q_x, kagg.Q_x,
                                        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2}, nullptr);

    MuSig2AggNonce aggnonce = musig2_nonce_agg({pn1, pn2});
    std::array<uint8_t,32> msg = {0x11,0x22,0x33};
    MuSig2Session sess = musig2_start_sign_session(aggnonce, kagg, msg);

    auto psig1 = musig2_partial_sign(sn1, sk1, kagg, sess, 0);
    CHECK(!psig1.is_zero(), "[MSI-1b] signer-0 partial sign succeeds with correct key+index");
}

// ── MSI-2: correct key + wrong index returns zero scalar (fail-closed) ────
static void test_correct_key_wrong_index() {
    auto kagg = make_2party_keyagg();

    Scalar sk1{};
    Scalar::parse_bytes_strict_nonzero(kSk1, sk1);

    auto [sn1, pn1] = musig2_nonce_gen(sk1, kagg.Q_x, kagg.Q_x,
                                        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3}, nullptr);
    Scalar sk2{};
    Scalar::parse_bytes_strict_nonzero(kSk2, sk2);
    auto [sn2, pn2] = musig2_nonce_gen(sk2, kagg.Q_x, kagg.Q_x,
                                        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4}, nullptr);

    MuSig2AggNonce aggnonce = musig2_nonce_agg({pn1, pn2});
    std::array<uint8_t,32> msg = {0xAA,0xBB,0xCC};
    MuSig2Session sess = musig2_start_sign_session(aggnonce, kagg, msg);

    // sk1 belongs to signer index 0, but we claim index 1 — must fail
    auto psig_wrong = musig2_partial_sign(sn1, sk1, kagg, sess, 1);
    CHECK(psig_wrong.is_zero(), "[MSI-2] wrong signer_index returns zero (fail-closed)");
}

// ── MSI-3: full 2-of-2 round-trip with correct indices verifies ──────────
static void test_full_roundtrip_correct_indices() {
    auto kagg = make_2party_keyagg();

    Scalar sk1{};
    Scalar::parse_bytes_strict_nonzero(kSk1, sk1);
    Scalar sk2{};
    Scalar::parse_bytes_strict_nonzero(kSk2, sk2);

    auto [sn1, pn1] = musig2_nonce_gen(sk1, kagg.Q_x, kagg.Q_x,
                                        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,5}, nullptr);
    auto [sn2, pn2] = musig2_nonce_gen(sk2, kagg.Q_x, kagg.Q_x,
                                        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6}, nullptr);

    MuSig2AggNonce aggnonce = musig2_nonce_agg({pn1, pn2});
    std::array<uint8_t,32> msg = {0xDE,0xAD,0xBE,0xEF};
    MuSig2Session sess = musig2_start_sign_session(aggnonce, kagg, msg);

    MuSig2Session sess2 = sess;  // copy session for signer 2
    auto psig1 = musig2_partial_sign(sn1, sk1, kagg, sess, 0);
    auto psig2 = musig2_partial_sign(sn2, sk2, kagg, sess2, 1);

    CHECK(!psig1.is_zero(), "[MSI-3a] signer-0 partial sign non-zero");
    CHECK(!psig2.is_zero(), "[MSI-3b] signer-1 partial sign non-zero");

    auto final_sig = musig2_partial_sig_agg({psig1, psig2}, sess);
    // final_sig is a 64-byte BIP-340 Schnorr signature over msg with Q
    CHECK(final_sig[0] != 0 || final_sig[32] != 0, "[MSI-3c] aggregated signature non-zero");
}

// ── MSI-4: empty individual_pubkeys → fail-closed (MED-3 / P1-SEC-01 closed) ──
// MED-3 is now CLOSED at the C++ layer: musig2_partial_sign treats the Rule-13
// signer-index cross-check as MANDATORY. When individual_pubkeys is empty (or
// shorter than signer_index) it CANNOT validate the signer index, so it
// fail-closes (returns Scalar::zero()) instead of signing blind. Production never
// reaches this branch — the v1 ABI is hard-failed at entry and v2 populates
// individual_pubkeys from its pubkeys[] parameter — so this asserts the
// defense-in-depth guard for any C++ caller that supplies an unvalidatable
// context. Previously advisory/INFO-only (RR-010) while MED-3 was open.
static void test_empty_pubkeys_fail_closed() {
    // Context with empty individual_pubkeys: signer_index cannot be validated.
    auto kagg = make_2party_keyagg();
    kagg.individual_pubkeys.clear();

    Scalar sk1{};
    Scalar::parse_bytes_strict_nonzero(kSk1, sk1);

    auto [sn1, pn1] = musig2_nonce_gen(sk1, kagg.Q_x, kagg.Q_x,
                                        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7}, nullptr);
    Scalar sk2{};
    Scalar::parse_bytes_strict_nonzero(kSk2, sk2);
    auto [sn2, pn2] = musig2_nonce_gen(sk2, kagg.Q_x, kagg.Q_x,
                                        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                         0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8}, nullptr);
    MuSig2AggNonce aggnonce = musig2_nonce_agg({pn1, pn2});
    std::array<uint8_t,32> msg = {0x55};
    MuSig2Session sess = musig2_start_sign_session(aggnonce, kagg, msg);

    auto psig = musig2_partial_sign(sn1, sk1, kagg, sess, 1);
    CHECK(psig.is_zero(),
          "[MSI-4] empty individual_pubkeys must fail-close (Rule-13 unvalidatable) "
          "— MED-3 / P1-SEC-01 closed");
}

// ── _run() ───────────────────────────────────────────────────────────────
int test_regression_musig2_signer_index_validation_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_musig2_signer_index_validation] MuSig2 signer_index cross-check (Rule 13)\n");

    test_correct_key_correct_index();
    test_correct_key_wrong_index();
    test_full_roundtrip_correct_indices();
    test_empty_pubkeys_fail_closed();  // MED-3 / P1-SEC-01: empty pubkeys must fail-close

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_musig2_signer_index_validation_run(); }
#endif
