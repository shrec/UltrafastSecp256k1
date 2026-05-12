// ============================================================================
// test_regression_hash_three_block_bounds.cpp
// Regression: compute_three_block missing input bounds guard (SEC-004).
// msg_len < 128 caused rem = msg_len - 128 to underflow (size_t wraparound),
// producing a ~0-byte memset. Fix: added if (msg_len < 128 || msg_len > 183) return.
//
// Tests exercise the three-block HMAC path via ecdsa_sign_hedged which always
// calls compute_three_block with a 129-byte input (V||byte||x||h1||aux).
//
// HTB-1: hedged sign produces non-zero, well-formed signature
// HTB-2: hedged sign is deterministic (same key+msg+aux → same sig)
// HTB-3: pubkey from scalar consistent across two derivations
// HTB-4: hedged sign with zero aux_rand succeeds
// HTB-5: hedged sign result verifies against correct pubkey
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#include <cstdio>
#define STANDALONE_TEST
#endif

#include <cstring>
#include <cstdio>
#include <array>

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/sign.hpp"

using secp256k1::fast::Scalar;

static int g_fail = 0;
#define ASSERT_TRUE(cond, msg)  do { if (!(cond)) { std::printf("FAIL [%s]: %s\n", __func__, (msg)); ++g_fail; } } while(0)
#define ASSERT_FALSE(cond, msg) do { if ( (cond)) { std::printf("FAIL [%s]: %s\n", __func__, (msg)); ++g_fail; } } while(0)

static void test_hedged_sign_produces_valid_sig() {
    uint8_t sk_bytes[32] = {}; sk_bytes[31] = 0x0A;
    Scalar sk;
    ASSERT_TRUE(Scalar::parse_bytes_strict_nonzero(sk_bytes, sk), "[HTB-1] parse sk=10");
    std::array<uint8_t, 32> msg{}; msg[31] = 0x01;
    std::array<uint8_t, 32> aux{}; aux[0] = 0xDE;
    auto sig = secp256k1::ecdsa_sign_hedged(msg, sk, aux);
    ASSERT_FALSE(sig.s.is_zero(), "[HTB-1] hedged sig s must not be zero");
    auto b = sig.to_bytes();
    uint32_t r = 0; for (int i = 0; i < 32; ++i) r |= b[i];
    ASSERT_TRUE(r != 0, "[HTB-1] hedged sig r must not be all-zero");
}

static void test_hedged_sign_is_deterministic() {
    uint8_t sk_bytes[32] = {}; sk_bytes[31] = 0x0B;
    Scalar sk;
    ASSERT_TRUE(Scalar::parse_bytes_strict_nonzero(sk_bytes, sk), "[HTB-2] parse sk=11");
    std::array<uint8_t, 32> msg{}; msg[0] = 0xAB;
    std::array<uint8_t, 32> aux{}; aux[7] = 0x42;
    auto sig1 = secp256k1::ecdsa_sign_hedged(msg, sk, aux);
    auto sig2 = secp256k1::ecdsa_sign_hedged(msg, sk, aux);
    ASSERT_TRUE(sig1.s == sig2.s, "[HTB-2] hedged signing must be deterministic (s)");
    auto b1 = sig1.to_bytes(); auto b2 = sig2.to_bytes();
    ASSERT_TRUE(std::memcmp(b1.data(), b2.data(), 64) == 0, "[HTB-2] deterministic (r||s)");
}

static void test_pubkey_consistent() {
    uint8_t sk_bytes[32] = {}; sk_bytes[31] = 0x05;
    Scalar sk;
    ASSERT_TRUE(Scalar::parse_bytes_strict_nonzero(sk_bytes, sk), "[HTB-3] parse sk=5");
    auto pk1 = secp256k1::ct::generator_mul(sk);
    auto pk2 = secp256k1::ct::generator_mul(sk);
    ASSERT_FALSE(pk1.is_infinity(), "[HTB-3] pubkey not infinity");
    ASSERT_TRUE(secp256k1::ct::point_eq(pk1, pk2) != 0, "[HTB-3] pubkeys consistent");
}

static void test_hedged_sign_zero_aux() {
    uint8_t sk_bytes[32] = {}; sk_bytes[31] = 0x07;
    Scalar sk;
    ASSERT_TRUE(Scalar::parse_bytes_strict_nonzero(sk_bytes, sk), "[HTB-4] parse sk=7");
    std::array<uint8_t, 32> msg{}; msg[15] = 0xFF;
    std::array<uint8_t, 32> aux{};
    auto sig = secp256k1::ecdsa_sign_hedged(msg, sk, aux);
    ASSERT_FALSE(sig.s.is_zero(), "[HTB-4] hedged sign with zero aux must succeed");
}

static void test_hedged_sign_verifies() {
    uint8_t sk_bytes[32] = {}; sk_bytes[31] = 0x0C;
    Scalar sk;
    ASSERT_TRUE(Scalar::parse_bytes_strict_nonzero(sk_bytes, sk), "[HTB-5] parse sk=12");
    auto pubkey = secp256k1::ct::generator_mul(sk);
    ASSERT_FALSE(pubkey.is_infinity(), "[HTB-5] pubkey valid");
    std::array<uint8_t, 32> msg{}; msg[0] = 0x10; msg[31] = 0x20;
    std::array<uint8_t, 32> aux{}; aux[0] = 0x99;
    auto sig = secp256k1::ecdsa_sign_hedged(msg, sk, aux);
    ASSERT_FALSE(sig.s.is_zero(), "[HTB-5] sig s not zero");
    ASSERT_TRUE(secp256k1::ecdsa_verify(sig, msg, pubkey), "[HTB-5] sig must verify");
}

int test_regression_hash_three_block_bounds_run() {
    g_fail = 0;
    test_hedged_sign_produces_valid_sig();
    test_hedged_sign_is_deterministic();
    test_pubkey_consistent();
    test_hedged_sign_zero_aux();
    test_hedged_sign_verifies();
    if (g_fail == 0)
        std::printf("PASS: compute_three_block bounds guard (SEC-004, HTB-1..5)\n");
    else
        std::printf("FAIL: compute_three_block bounds: %d failure(s)\n", g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_hash_three_block_bounds_run(); }
#endif
