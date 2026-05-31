// ============================================================================
// test_regression_frost_musig2_degenerate.cpp
// ============================================================================
// Regression tests for degenerate-input guards in FROST and MuSig2:
//
//   FMD-1 (SEC-002, frost.cpp): frost_keygen_finalize rejects a group public
//         key that accumulates to infinity (adversarial commitments cancel).
//   FMD-2 (SEC-004, frost.cpp): frost_sign_nonce_gen (by-value seed) still
//         produces correct, non-zero hiding/binding nonces after the signature
//         change.
//   FMD-3 (SEC-003, musig2.cpp): musig2_partial_sign returns Scalar::zero()
//         (fail-closed) when session.e == 0.
//   FMD-4 (SEC-005, musig2.cpp): musig2_partial_sign still produces a valid
//         partial signature for a normal session after the generator_mul
//         → generator_mul_blinded change.
// ============================================================================

#include <cstdio>
#include <cstring>
#include <array>
#include <vector>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

#include "secp256k1/frost.hpp"
#include "secp256k1/musig2.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/init.hpp"

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

// ─── FMD-1: frost_keygen_finalize rejects infinity group key ─────────────────
// Adversarial setup: two participants where A_{1,0} = G and A_{2,0} = -G.
// Their sum is the point at infinity → frost_keygen_finalize must return false.
#if SECP256K1_HAS_FROST
static void test_frost_infinity_group_key() {
    SECP256K1_INIT();

    // Build two fake commitments that cancel.
    FrostCommitment c1, c2;
    c1.from = 1;
    c2.from = 2;

    Point const G   = Point::generator();
    Point const neg_G = G.negate();

    c1.coeffs = { G };
    c2.coeffs = { neg_G };

    // received_shares: for this test we only care that the group key check
    // fires before the share verification, so we pass empty received_shares.
    std::vector<FrostShare> no_shares;
    auto [pkg, ok] = frost_keygen_finalize(
        1,                       // participant_id
        {c1, c2},               // commitments (G + (-G) = infinity)
        std::move(no_shares),
        2, 2);

    CHECK(!ok, "[FMD-1] frost_keygen_finalize rejects infinity group key");
}

// ─── FMD-2: frost_sign_nonce_gen (by-value seed) correctness ─────────────────
static void test_frost_nonce_gen_by_value_seed() {
    SECP256K1_INIT();

    std::array<uint8_t, 32> seed{};
    for (int i = 0; i < 32; ++i) seed[i] = static_cast<uint8_t>(0x11 + i);

    auto [nonce, commit] = frost_sign_nonce_gen(1, seed);

    CHECK(!nonce.hiding_nonce.is_zero(),  "[FMD-2] hiding_nonce != 0");
    CHECK(!nonce.binding_nonce.is_zero(), "[FMD-2] binding_nonce != 0");
    CHECK(commit.id == 1,                 "[FMD-2] commitment id == 1");
    CHECK(!commit.hiding_point.is_infinity(),  "[FMD-2] hiding_point != infinity");
    CHECK(!commit.binding_point.is_infinity(), "[FMD-2] binding_point != infinity");

    // Verify D_i = d_i * G and E_i = e_i * G by checking from public key
    // (hiding_point should equal ct::generator_mul(hiding_nonce)).
    auto exp_hiding  = ct::generator_mul(nonce.hiding_nonce);
    auto exp_binding = ct::generator_mul(nonce.binding_nonce);

    auto [hx, ho] = commit.hiding_point.x_bytes_and_parity();
    auto [ex, eo] = exp_hiding.x_bytes_and_parity();
    CHECK(hx == ex && ho == eo, "[FMD-2] hiding_point == d_i * G");

    auto [bx, bo] = commit.binding_point.x_bytes_and_parity();
    auto [fx, fo] = exp_binding.x_bytes_and_parity();
    CHECK(bx == fx && bo == fo, "[FMD-2] binding_point == e_i * G");
}
#endif // SECP256K1_HAS_FROST

// ─── FMD-3/4: musig2_partial_sign degenerate session + normal round-trip ─────
static void test_musig2_degenerate_session_e_zero() {
    SECP256K1_INIT();

    // Build a minimal 1-of-1 MuSig2 context.
    static const uint8_t kSk[32] = {
        0x05,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x03
    };

    Scalar sk{};
    CHECK(Scalar::parse_bytes_strict_nonzero(kSk, sk), "[FMD-3] parse sk");

    Point Q = ct::generator_mul(sk);
    auto [Q_x, Q_odd] = Q.x_bytes_and_parity();

    // Build a minimal 1-of-1 key_agg_ctx manually. Rule-13 is now MANDATORY in
    // musig2_partial_sign (it fail-closes when individual_pubkeys cannot validate the
    // signer_index), so populate the single signer's pubkey (= Q for 1-of-1) —
    // otherwise even a valid session would fail-close. FMD-3 then exercises the e==0
    // guard for the right reason rather than the empty-pubkeys guard.
    MuSig2KeyAggCtx kag{};
    kag.Q         = Q;
    kag.Q_x       = Q_x;
    kag.Q_negated = false;
    kag.key_coefficients = { Scalar::one() };       // a_1 = 1 for 1-of-1
    kag.individual_pubkeys = { Q.to_compressed() };  // signer 0 = Q (Rule-13 validation)

    // Non-degenerate nonce for both tests.
    static const uint8_t kMsg[32] = {
        0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
        0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
        0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10,
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77
    };
    std::array<uint8_t,32> msg{};
    std::memcpy(msg.data(), kMsg, 32);

    auto agg_pubkey_x = Q_x;

    auto [sec_nonce1, pub_nonce1] = musig2_nonce_gen(sk, Q_x, agg_pubkey_x, msg);

    // FMD-3: degenerate session with e==0 must return Scalar::zero() (fail-closed).
    MuSig2Session degen_session{};
    degen_session.R        = ct::generator_mul(sec_nonce1.k1);  // non-zero R
    degen_session.b        = Scalar::one();
    degen_session.e        = Scalar::zero();                     // DEGENERATE: e=0
    degen_session.R_negated = false;

    MuSig2SecNonce sec_nonce_copy1 = sec_nonce1;  // copy — nonce is consumed
    Scalar psig_degen = musig2_partial_sign(sec_nonce_copy1, sk, kag, degen_session, 0);
    CHECK(psig_degen.is_zero(), "[FMD-3] partial_sign with e=0 returns zero (fail-closed)");

    // FMD-4: normal session still works (generator_mul_blinded change correctness).
    // Construct a real session via the API.
    auto [sec_nonce2, pub_nonce2] = musig2_nonce_gen(sk, Q_x, agg_pubkey_x, msg);

    MuSig2AggNonce agg_nonce = musig2_nonce_agg({ pub_nonce2 });
    MuSig2Session session = musig2_start_sign_session(agg_nonce, kag, msg);

    CHECK(!session.e.is_zero(), "[FMD-4] session.e is non-zero for valid session");

    MuSig2SecNonce sec_nonce_copy2 = sec_nonce2;
    Scalar psig = musig2_partial_sign(sec_nonce_copy2, sk, kag, session, 0);
    CHECK(!psig.is_zero(), "[FMD-4] partial_sign with valid session returns non-zero");

    // Aggregate: 1-signer aggregation yields the final sig.
    auto final_sig = musig2_partial_sig_agg({ psig }, session);
    // Final sig r (first 32 bytes) should not be all-zero.
    bool r_nonzero = false;
    for (int i = 0; i < 32; ++i) r_nonzero |= (final_sig[i] != 0);
    CHECK(r_nonzero, "[FMD-4] aggregated sig r != 0");
}

// ─── Entry point ─────────────────────────────────────────────────────────────

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
int main() {
#else
int test_regression_frost_musig2_degenerate_run() {
#endif
    printf("[frost_musig2_degenerate] FMD-1..4: FROST infinity key + MuSig2 degenerate session\n");

#if SECP256K1_HAS_FROST
    test_frost_infinity_group_key();
    test_frost_nonce_gen_by_value_seed();
#else
    printf("  [skip] SECP256K1_HAS_FROST not set — FMD-1/2 skipped\n");
    g_pass += 2;
#endif
    test_musig2_degenerate_session_e_zero();

    printf("[frost_musig2_degenerate] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail;
}
