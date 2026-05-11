// ============================================================================
// test_regression_bip32_private_key_strict.cpp
// Regression: ExtendedKey::private_key() must reject keys >= n or == 0.
//
// Bug fixed: bip32.cpp used Scalar::from_bytes(key) which silently reduces
// mod n. A stored key == n becomes scalar 0; n+1 becomes 1. Fix: use
// parse_bytes_strict_nonzero — returns Scalar{} (zero) on invalid input.
//
// Tests:
//   BKS-1: valid key round-trips through ExtendedKey correctly
//   BKS-2: key == n produces zero scalar (strict rejection)
//   BKS-3: key == 0 produces zero scalar (strict rejection)
// ============================================================================

#include <cstdio>
#include <cstring>
#include <array>
static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/scalar.hpp"

// secp256k1 order n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
static const unsigned char kN[32] = {
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
};

// ── BKS-1: valid key is non-zero after round-trip ─────────────────────────
static void test_valid_key_roundtrip() {
    unsigned char sk[32] = {};
    sk[31] = 7;  // key = 7

    secp256k1::ExtendedKey xk{};
    xk.key = {};
    std::memcpy(xk.key.data(), sk, 32);
    xk.is_private = true;

    auto scalar = xk.private_key();
    CHECK(!scalar.is_zero(), "[BKS-1a] valid key (7) produces non-zero scalar");
    auto bytes = scalar.to_bytes();
    CHECK(bytes[31] == 7, "[BKS-1b] scalar value matches stored key");
}

// ── BKS-2: key == n produces zero scalar (strict rejection) ───────────────
static void test_key_equal_n_rejected() {
    secp256k1::ExtendedKey xk{};
    xk.key = {};
    std::memcpy(xk.key.data(), kN, 32);
    xk.is_private = true;

    auto scalar = xk.private_key();
    // With strict parsing: n >= n → rejected → returns Scalar{} == 0
    CHECK(scalar.is_zero(), "[BKS-2] key == n is rejected (strict), returns zero scalar");
}

// ── BKS-3: key == 0 produces zero scalar ──────────────────────────────────
static void test_zero_key_rejected() {
    secp256k1::ExtendedKey xk{};
    xk.key = {};  // all zeros
    xk.is_private = true;

    auto scalar = xk.private_key();
    CHECK(scalar.is_zero(), "[BKS-3] zero key is rejected (strict), returns zero scalar");
}

// ── _run() ─────────────────────────────────────────────────────────────────
int test_regression_bip32_private_key_strict_run() {
    g_pass = 0; g_fail = 0;
    std::printf("[regression_bip32_private_key_strict] strict parsing for BIP-32 private_key()\n");

    test_valid_key_roundtrip();
    test_key_equal_n_rejected();
    test_zero_key_rejected();

    std::printf("  pass=%d  fail=%d\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_bip32_private_key_strict_run(); }
#endif
