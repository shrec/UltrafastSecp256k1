// ============================================================================
// Test: ECDSA sign/verify + Schnorr BIP-340 sign/verify
// ============================================================================

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

#include <cstdio>
#include <cstring>
#include <array>

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

// -- SHA-256 Tests ------------------------------------------------------------

static void test_sha256() {
    printf("\n--- SHA-256 ---\n");

    // NIST test vector: SHA256("abc")
    auto h = SHA256::hash("abc", 3);
    uint8_t expected[] = {
        0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
        0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
        0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
        0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
    };
    CHECK(std::memcmp(h.data(), expected, 32) == 0, "SHA256(\"abc\") matches NIST vector");

    // Empty string
    auto h2 = SHA256::hash("", 0);
    uint8_t expected2[] = {
        0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
        0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
        0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
        0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55
    };
    CHECK(std::memcmp(h2.data(), expected2, 32) == 0, "SHA256(\"\") matches NIST vector");
}

// -- Scalar::inverse Tests ----------------------------------------------------

static void test_scalar_inverse() {
    printf("\n--- Scalar::inverse ---\n");

    auto a = Scalar::from_uint64(7);
    auto a_inv = a.inverse();
    auto product = a * a_inv;
    CHECK(product == Scalar::one(), "7 * 7^{-1} = 1 mod n");

    auto b = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto b_inv = b.inverse();
    auto product2 = b * b_inv;
    CHECK(product2 == Scalar::one(), "random * random^{-1} = 1 mod n");

    auto zero_inv = Scalar::zero().inverse();
    CHECK(zero_inv.is_zero(), "inverse(0) = 0");
}

// -- Scalar::negate Tests -----------------------------------------------------

static void test_scalar_negate() {
    printf("\n--- Scalar::negate ---\n");

    auto a = Scalar::from_uint64(42);
    auto neg_a = a.negate();
    auto sum = a + neg_a;
    CHECK(sum.is_zero(), "a + (-a) = 0");

    CHECK(Scalar::zero().negate().is_zero(), "negate(0) = 0");
}

// -- ECDSA Tests --------------------------------------------------------------

static void test_ecdsa_sign_verify() {
    printf("\n--- ECDSA Sign/Verify ---\n");

    // Key pair
    auto priv = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto pub = Point::generator().scalar_mul(priv);

    // Message hash = SHA256("Hello, secp256k1!")
    auto msg_hash = SHA256::hash("Hello, secp256k1!", 17);

    // Sign
    auto sig = ecdsa_sign(msg_hash, priv);
    CHECK(!sig.r.is_zero() && !sig.s.is_zero(), "signature is non-zero");
    CHECK(sig.is_low_s(), "signature has low-S (BIP-62)");

    // Verify
    bool const valid = ecdsa_verify(msg_hash, pub, sig);
    CHECK(valid, "verify(sign(msg, priv), pub) = true");

    // Wrong message
    auto wrong_hash = SHA256::hash("Wrong message", 13);
    bool const invalid_msg = ecdsa_verify(wrong_hash, pub, sig);
    CHECK(!invalid_msg, "verify with wrong message = false");

    // Wrong key
    auto wrong_pub = Point::generator().scalar_mul(Scalar::from_uint64(999));
    bool const invalid_key = ecdsa_verify(msg_hash, wrong_pub, sig);
    CHECK(!invalid_key, "verify with wrong pubkey = false");

    // Compact round-trip
    auto compact = sig.to_compact();
    auto sig2 = ECDSASignature::from_compact(compact);
    CHECK(sig2.r == sig.r && sig2.s == sig.s, "compact encoding round-trip");

    // DER encoding
    auto [der, der_len] = sig.to_der();
    CHECK(der_len > 0 && der[0] == 0x30, "DER encoding starts with SEQUENCE tag");
}

static void test_ecdsa_deterministic() {
    printf("\n--- ECDSA Deterministic (RFC 6979) ---\n");

    auto priv = Scalar::from_hex(
        "0000000000000000000000000000000000000000000000000000000000000001");
    auto msg = SHA256::hash("test", 4);

    // Sign twice with same key + message -> same signature (deterministic)
    auto sig1 = ecdsa_sign(msg, priv);
    auto sig2 = ecdsa_sign(msg, priv);
    CHECK(sig1.r == sig2.r && sig1.s == sig2.s,
          "same key + message -> same signature (deterministic)");

    // Different message -> different signature
    auto msg2 = SHA256::hash("test2", 5);
    auto sig3 = ecdsa_sign(msg2, priv);
    CHECK(sig3.r != sig1.r || sig3.s != sig1.s,
          "different message -> different signature");
}

// -- Schnorr Tests ------------------------------------------------------------

static void test_schnorr_sign_verify() {
    printf("\n--- Schnorr BIP-340 Sign/Verify ---\n");

    auto priv = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");

    // BIP-340 x-only pubkey
    auto pubkey_x = schnorr_pubkey(priv);
    {
        std::array<uint8_t, 32> const zero_arr{};
        CHECK(pubkey_x != zero_arr, "x-only pubkey is non-zero");
    }

    // Message
    std::array<uint8_t, 32> msg{};
    auto h = SHA256::hash("Hello Schnorr!", 14);
    std::memcpy(msg.data(), h.data(), 32);

    // Aux randomness (zeros for determinism)
    std::array<uint8_t, 32> const aux{};

    // Sign
    auto sig = schnorr_sign(priv, msg, aux);

    // Verify
    bool const valid = schnorr_verify(pubkey_x, msg, sig);
    CHECK(valid, "schnorr_verify(sign(msg, priv), pubkey) = true");

    // Wrong message
    std::array<uint8_t, 32> wrong_msg{};
    wrong_msg[0] = 0xFF;
    bool const invalid = schnorr_verify(pubkey_x, wrong_msg, sig);
    CHECK(!invalid, "schnorr_verify with wrong message = false");

    // Round-trip
    auto sig_bytes = sig.to_bytes();
    auto sig2 = SchnorrSignature::from_bytes(sig_bytes);
    CHECK(sig2.r == sig.r && sig2.s == sig.s, "schnorr signature round-trip");
}

static void test_tagged_hash() {
    printf("\n--- Tagged Hash (BIP-340) ---\n");

    auto h1 = tagged_hash("BIP0340/challenge", "test", 4);
    auto h2 = tagged_hash("BIP0340/challenge", "test", 4);
    CHECK(h1 == h2, "tagged_hash is deterministic");

    auto h3 = tagged_hash("BIP0340/aux", "test", 4);
    CHECK(h1 != h3, "different tags -> different hashes");
}

// -- Main ---------------------------------------------------------------------

int test_ecdsa_schnorr_run() {
    printf("================================================================\n");
    printf("  ECDSA + Schnorr (BIP-340) Test Suite\n");
    printf("================================================================\n");

    test_sha256();
    test_scalar_inverse();
    test_scalar_negate();
    test_ecdsa_sign_verify();
    test_ecdsa_deterministic();
    test_tagged_hash();
    test_schnorr_sign_verify();

    printf("\n================================================================\n");
    printf("  Results: %d / %d passed\n", tests_passed, tests_run);
    printf("================================================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}
