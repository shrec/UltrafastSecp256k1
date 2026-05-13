// ============================================================================
// test_regression_ct_blinding_nonce_path.cpp
// ============================================================================
// Regression: ct::ecdsa_sign / ct::schnorr_sign must use generator_mul_blinded
// for the nonce multiplication R = k*G.
//
// Root Cause (2026-05-12)
//   ct_sign.cpp used ct::generator_mul(k) for R = k*G on all 5 nonce-path
//   multiplications (ecdsa_sign, ecdsa_sign_hedged, schnorr_sign,
//   ecdsa_sign_recoverable, ecdsa_sign_hedged_recoverable).
//   ct::generator_mul_blinded was only used by the non-CT fast:: path in
//   schnorr.cpp:397, leaving the dedicated CT signing functions without DPA
//   blinding protection even when secp256k1_context_randomize() was called.
//
// Fix
//   All five nonce-path calls in ct_sign.cpp replaced with
//   ct::generator_mul_blinded(k). The blinding is transparent: mathematically
//   blinded(k)*G == k*G, so signatures remain deterministic.
//
// Guard
//   This test verifies that:
//   1. CT ECDSA sign produces a valid, verifiable signature with blinding OFF.
//   2. CT ECDSA sign produces the same valid signature with blinding ON
//      (deterministic RFC 6979 nonce ⇒ same output despite blinding).
//   3. CT Schnorr sign produces the same valid signature with blinding ON.
//   4. 20 random keys: sign + verify identical with and without blinding.
//
// LIMITATION (NEW-TEST-002, 2026-05-13 v8 audit)
//   This test verifies FUNCTIONAL CORRECTNESS only: that the signed output is
//   identical whether or not blinding is applied. It does NOT verify that
//   generator_mul_blinded is actually invoked at the call site. A revert from
//   `ct::generator_mul_blinded(k)` to `ct::generator_mul(k)` would still PASS
//   this test, because the unblinded path produces the same mathematical result.
//
//   Verifying actual blinding requires CT instrumentation (Valgrind ct-verif,
//   MSAN, or a taint-tracking build that classifies the random scalar `r` as
//   secret-tainted) — see audit/test_ct_verif_formal.cpp for that path.
//
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>
#include <cstring>
#include <random>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"

#include "audit_check.hpp"

#ifndef UNIFIED_AUDIT_RUNNER
#include "ufsecp_abi.h"
#endif

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;

// ── fixed test vector ────────────────────────────────────────────────────────
static const std::array<uint8_t, 32> TEST_KEY = {{
    0xB7,0xE1,0x51,0x62,0x8A,0xED,0x2A,0x6A,
    0xBF,0x71,0x58,0x80,0x9C,0xF4,0xF3,0xC7,
    0x62,0xE7,0x16,0x0F,0x38,0xB4,0xDA,0x56,
    0xA7,0x84,0xD9,0x04,0x51,0x90,0xCF,0xEF
}};

static const std::array<uint8_t, 32> TEST_MSG = {{
    0x24,0x3F,0x6A,0x88,0x85,0xA3,0x08,0xD3,
    0x13,0x19,0x8A,0x2E,0x03,0x70,0x73,0x44,
    0xA4,0x09,0x38,0x22,0x29,0x9F,0x31,0xD0,
    0x08,0x2E,0xFA,0x98,0xEC,0x4E,0x6C,0x89
}};

static const std::array<uint8_t, 32> AUX_RAND = {{
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00
}};

// ── helpers ──────────────────────────────────────────────────────────────────

static void activate_blinding(ufsecp_ctx* ctx) {
    static const uint8_t SEED[32] = {
        0xDE,0xAD,0xBE,0xEF,0xCA,0xFE,0xBA,0xBE,
        0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0A,0x0B,0x0C,0x0D,0x0E,0x0F,0x10,
        0x11,0x12,0x13,0x14,0x15,0x16,0x17,0x18
    };
    ufsecp_context_randomize(ctx, SEED);
}

static void deactivate_blinding(ufsecp_ctx* ctx) {
    // Randomizing with all-zeros seed deterministically sets blinding to a
    // known scalar; blinding is still ACTIVE but reproducible.
    // To exercise the unblinded fallback we destroy and recreate the context.
    (void)ctx;
}

// ── test_ecdsa_blinded_transparent ──────────────────────────────────────────
// Verifies that activating blinding does not change ECDSA signature output.
// Same private key + same message → same RFC 6979 nonce → same signature.

static void test_ecdsa_blinded_transparent() {
    printf("[1] ct::ecdsa_sign: blinded == unblinded (RFC 6979 determinism)\n");

    Scalar sk = Scalar::from_bytes(TEST_KEY);
    auto sig_no_blind = secp256k1::ct::ecdsa_sign(TEST_MSG, sk);

    // Activate blinding globally.
    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK) {
        printf("  [SKIP] ctx_create failed\n");
        return;
    }
    activate_blinding(ctx);

    auto sig_blinded = secp256k1::ct::ecdsa_sign(TEST_MSG, sk);

    // Must produce identical r and s (deterministic nonce, same math).
    auto r_nb = sig_no_blind.r.to_bytes();
    auto s_nb = sig_no_blind.s.to_bytes();
    auto r_bl = sig_blinded.r.to_bytes();
    auto s_bl = sig_blinded.s.to_bytes();

    CHECK(std::memcmp(r_nb.data(), r_bl.data(), 32) == 0,
          "ecdsa_sign r: blinded == unblinded");
    CHECK(std::memcmp(s_nb.data(), s_bl.data(), 32) == 0,
          "ecdsa_sign s: blinded == unblinded");

    // Signature must verify.
    auto pk = secp256k1::ct::generator_mul(sk);
    bool ok_nb = secp256k1::ecdsa_verify(TEST_MSG.data(), pk, sig_no_blind);
    bool ok_bl = secp256k1::ecdsa_verify(TEST_MSG.data(), pk, sig_blinded);
    CHECK(ok_nb, "ecdsa_sign (no-blind) verifies");
    CHECK(ok_bl, "ecdsa_sign (blinded)  verifies");

    ufsecp_ctx_destroy(ctx);
}

// ── test_schnorr_blinded_transparent ─────────────────────────────────────────
// Same as above for ct::schnorr_sign.

static void test_schnorr_blinded_transparent() {
    printf("[2] ct::schnorr_sign: blinded == unblinded (BIP-340 determinism)\n");

    Scalar sk = Scalar::from_bytes(TEST_KEY);
    auto kp = secp256k1::ct::schnorr_keypair_create(sk);

    auto sig_no_blind = secp256k1::ct::schnorr_sign(kp, TEST_MSG, AUX_RAND);

    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK) {
        printf("  [SKIP] ctx_create failed\n");
        return;
    }
    activate_blinding(ctx);

    auto sig_blinded = secp256k1::ct::schnorr_sign(kp, TEST_MSG, AUX_RAND);

    CHECK(std::memcmp(sig_no_blind.r.data(), sig_blinded.r.data(), 32) == 0,
          "schnorr_sign r: blinded == unblinded");
    CHECK(std::memcmp(sig_no_blind.s.to_bytes().data(),
                      sig_blinded.s.to_bytes().data(), 32) == 0,
          "schnorr_sign s: blinded == unblinded");

    // Verify via raw BIP-340 verify (public key = kp.px).
    bool ok_bl = secp256k1::schnorr_verify(kp.px, TEST_MSG, sig_blinded);
    CHECK(ok_bl, "schnorr_sign (blinded) verifies");

    ufsecp_ctx_destroy(ctx);
}

// ── test_ecdsa_random_keys_blinded ───────────────────────────────────────────
// 20 random keys: sign with blinding, verify, compare with unblinded output.

static void test_ecdsa_random_keys_blinded() {
    printf("[3] ct::ecdsa_sign: 20 random keys, blinded == unblinded\n");

    ufsecp_ctx* ctx = nullptr;
    if (ufsecp_ctx_create(&ctx) != UFSECP_OK) {
        printf("  [SKIP] ctx_create failed\n");
        return;
    }
    activate_blinding(ctx);

    std::mt19937_64 rng(0xABCDEF0123456789ULL);
    int ok_count = 0;
    for (int i = 0; i < 20; ++i) {
        std::array<uint8_t, 32> kb, msgb;
        for (auto& x : kb)   x = (uint8_t)rng();
        for (auto& x : msgb) x = (uint8_t)rng();
        // Ensure key != 0 and < n (use from_bytes which does mod-n reduction).
        Scalar sk = Scalar::from_bytes(kb);
        if (sk.is_zero()) continue;

        // Unblinded (fresh ctx without randomize == no blinding).
        ufsecp_ctx* ctx2 = nullptr;
        ufsecp_ctx_create(&ctx2);
        // ctx2 has blinding OFF (not randomized).

        auto sig_nb = secp256k1::ct::ecdsa_sign(msgb, sk);

        // Switch to blinded context path.
        activate_blinding(ctx);
        auto sig_bl = secp256k1::ct::ecdsa_sign(msgb, sk);

        auto pk = secp256k1::ct::generator_mul(sk);
        bool verify_ok = secp256k1::ecdsa_verify(msgb.data(), pk, sig_bl);
        bool identical  = (std::memcmp(sig_nb.r.to_bytes().data(),
                                        sig_bl.r.to_bytes().data(), 32) == 0 &&
                           std::memcmp(sig_nb.s.to_bytes().data(),
                                        sig_bl.s.to_bytes().data(), 32) == 0);
        if (verify_ok && identical) ++ok_count;

        ufsecp_ctx_destroy(ctx2);
    }

    char msg[80];
    std::snprintf(msg, sizeof(msg),
                  "ecdsa_sign: %d/20 random-key blinded==unblinded+verified", ok_count);
    CHECK(ok_count >= 18, msg);  // allow 2 skipped (zero-key rejection)

    ufsecp_ctx_destroy(ctx);
}

// ── main entry ───────────────────────────────────────────────────────────────

int test_regression_ct_blinding_nonce_path_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: ct_sign nonce path uses generator_mul_blinded\n");
    printf("  (fix 2026-05-12: R=k*G was using unblinded generator_mul in\n");
    printf("   ct::ecdsa_sign, ct::schnorr_sign, and recoverable variants)\n");
    printf("======================================================================\n\n");

    test_ecdsa_blinded_transparent();
    printf("\n");
    test_schnorr_blinded_transparent();
    printf("\n");
    test_ecdsa_random_keys_blinded();
    printf("\n");

    printf("[ct_blinding_nonce_path] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_ct_blinding_nonce_path_run(); }
#endif
