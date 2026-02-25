/**
 * ESP32 Audit Test - Minimal secp256k1 verification for Xtensa targets
 *
 * Tests: field_26 ops, scalar ops, point operations, KAT vectors
 * Targets: ESP32-S3 (Xtensa LX7), ESP32/PICO-D4 (Xtensa LX6)
 */
#include <cstdio>
#include <cstdint>
#include <cstring>

#ifdef ESP_PLATFORM
#include "esp_log.h"
#include "esp_system.h"
static const char* TAG = "secp256k1_audit";
#define LOG(fmt, ...) ESP_LOGI(TAG, fmt, ##__VA_ARGS__)
#else
#define LOG(fmt, ...) printf("[AUDIT] " fmt "\n", ##__VA_ARGS__)
#endif

// Include library headers
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"

using namespace secp256k1::fast;

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, name) do { \
    if (cond) { g_pass++; LOG("  PASS: %s", name); } \
    else { g_fail++; LOG("  FAIL: %s", name); } \
} while(0)

// -- Test 1: Field element basics ----------------------------------
static void test_field_basics() {
    LOG("=== Field Element Basics ===");

    auto zero = FieldElement::zero();
    auto one  = FieldElement::one();

    // zero + one = one
    auto result = FieldElement::add(zero, one);
    CHECK(result == one, "0 + 1 == 1");

    // one + one != one
    auto two = FieldElement::add(one, one);
    CHECK(!(two == one), "1 + 1 != 1");

    // one * one = one
    auto prod = FieldElement::mul(one, one);
    CHECK(prod == one, "1 * 1 == 1");

    // one - one = zero
    auto diff = FieldElement::sub(one, one);
    CHECK(diff == zero, "1 - 1 == 0");

    // Negation: -0 == 0
    auto neg_zero = FieldElement::negate(zero);
    CHECK(neg_zero == zero, "negate(0) == 0");

    // a + (-a) = 0
    auto neg_one = FieldElement::negate(one);
    auto sum = FieldElement::add(one, neg_one);
    CHECK(sum == zero, "1 + (-1) == 0");
}

// -- Test 2: Field multiplication properties -----------------------
static void test_field_mul_properties() {
    LOG("=== Field Multiplication Properties ===");

    auto a = FieldElement::from_uint64(0x123456789ABCDEF0ULL);
    auto b = FieldElement::from_uint64(0xFEDCBA9876543210ULL);
    auto one = FieldElement::one();

    // Commutativity: a*b == b*a
    auto ab = FieldElement::mul(a, b);
    auto ba = FieldElement::mul(b, a);
    CHECK(ab == ba, "mul commutativity: a*b == b*a");

    // Identity: a*1 == a
    auto a1 = FieldElement::mul(a, one);
    CHECK(a1 == a, "mul identity: a*1 == a");

    // Squaring: a*a == square(a)
    auto aa = FieldElement::mul(a, a);
    auto sq = FieldElement::square(a);
    CHECK(aa == sq, "a*a == square(a)");
}

// -- Test 3: Scalar basics ----------------------------------------
static void test_scalar_basics() {
    LOG("=== Scalar Basics ===");

    auto zero = Scalar::zero();
    auto one  = Scalar::one();

    CHECK(zero.is_zero(), "scalar zero is_zero");
    CHECK(!one.is_zero(), "scalar one is not zero");

    // one + zero = one
    auto sum = Scalar::add(one, zero);
    CHECK(sum == one, "scalar 1 + 0 == 1");

    // one - one = zero
    auto diff = Scalar::sub(one, one);
    CHECK(diff.is_zero(), "scalar 1 - 1 == 0");

    // one * one = one
    auto prod = Scalar::mul(one, one);
    CHECK(prod == one, "scalar 1 * 1 == 1");
}

// -- Test 4: Point on curve ---------------------------------------
static void test_point_basics() {
    LOG("=== Point Basics ===");

    auto G = Point::generator();
    CHECK(!G.is_infinity(), "generator is not infinity");

    auto inf = Point::infinity();
    CHECK(inf.is_infinity(), "infinity is infinity");

    // G + infinity = G
    auto sum = Point::add(G, inf);
    CHECK(!sum.is_infinity(), "G + inf is not infinity");

    // G - G = infinity
    auto neg_G = Point::negate(G);
    auto diff = Point::add(G, neg_G);
    CHECK(diff.is_infinity(), "G + (-G) == infinity");
}

// -- Test 5: Scalar multiplication KAT ----------------------------
static void test_scalar_mul_kat() {
    LOG("=== Scalar Multiplication KAT ===");

    auto G = Point::generator();
    auto one = Scalar::one();

    // 1*G == G
    auto P1 = Point::mul(G, one);
    auto G_bytes = G.to_compressed();
    auto P1_bytes = P1.to_compressed();
    CHECK(G_bytes == P1_bytes, "1*G == G");

    // 2*G
    auto two = Scalar::add(one, one);
    auto P2 = Point::mul(G, two);
    CHECK(!P2.is_infinity(), "2*G is not infinity");

    // 2*G == G+G
    auto G_plus_G = Point::add(G, G);
    auto P2_bytes = P2.to_compressed();
    auto GG_bytes = G_plus_G.to_compressed();
    CHECK(P2_bytes == GG_bytes, "2*G == G+G");

    // Known 2*G x-coordinate (first 4 bytes of compressed)
    // 02c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5
    CHECK(P2_bytes[0] == 0x02 || P2_bytes[0] == 0x03, "2*G compressed prefix valid");
    CHECK(P2_bytes[1] == 0xc6, "2*G x[0] == 0xc6");
    CHECK(P2_bytes[2] == 0x04, "2*G x[1] == 0x04");
    CHECK(P2_bytes[3] == 0x7f, "2*G x[2] == 0x7f");
}

// -- Test 6: ECDSA sign/verify ------------------------------------
static void test_ecdsa() {
    LOG("=== ECDSA Sign/Verify ===");

    // Private key (test vector)
    std::array<uint8_t, 32> privkey = {
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01
    };

    // Message hash
    std::array<uint8_t, 32> msg = {
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10,
        0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18,
        0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x20
    };

    auto sk = Scalar::from_bytes(privkey);
    CHECK(!sk.is_zero(), "private key is not zero");

    // Derive public key: pk = sk * G
    auto G = Point::generator();
    auto pk = Point::mul(G, sk);
    CHECK(!pk.is_infinity(), "public key is not infinity");

    // Sign
    auto sig = secp256k1::fast::ecdsa_sign(msg, sk);
    CHECK(sig.has_value(), "ECDSA sign succeeded");

    if (sig.has_value()) {
        // Verify
        bool valid = secp256k1::fast::ecdsa_verify(msg, sig.value(), pk);
        CHECK(valid, "ECDSA verify succeeded");

        // Tamper with message -> verify should fail
        std::array<uint8_t, 32> bad_msg = msg;
        bad_msg[0] ^= 0xFF;
        bool invalid = secp256k1::fast::ecdsa_verify(bad_msg, sig.value(), pk);
        CHECK(!invalid, "ECDSA verify rejects tampered message");
    }
}

#ifdef ESP_PLATFORM
extern "C" void app_main(void)
#else
int main()
#endif
{
    LOG("+==================================================+");
    LOG("|  UltrafastSecp256k1 ESP32 Audit Test v3.14.0    |");
    LOG("+==================================================+");
#ifdef CONFIG_IDF_TARGET_ESP32S3
    LOG("|  Target: ESP32-S3 (Xtensa LX7)                 |");
#elif defined(CONFIG_IDF_TARGET_ESP32)
    LOG("|  Target: ESP32 (Xtensa LX6 / PICO-D4)          |");
#else
    LOG("|  Target: Generic                                |");
#endif
    LOG("+==================================================+");
    LOG("");

    test_field_basics();
    test_field_mul_properties();
    test_scalar_basics();
    test_point_basics();
    test_scalar_mul_kat();
    test_ecdsa();

    LOG("");
    LOG("===================================================");
    LOG("  Results: %d PASSED, %d FAILED", g_pass, g_fail);
    LOG("  Status:  %s", g_fail == 0 ? "ALL PASS OK" : "FAILURES DETECTED X");
    LOG("===================================================");

#ifndef ESP_PLATFORM
    return g_fail > 0 ? 1 : 0;
#endif
}
