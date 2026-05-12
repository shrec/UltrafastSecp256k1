// ============================================================================
// Test: ECIES Encryption / Decryption
// ============================================================================
// Validates ECIES encrypt → decrypt roundtrip, tamper detection, wrong-key
// rejection, empty plaintext, and variable-length payloads.
// ============================================================================

#include "secp256k1/ecies.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

#include <cstdio>
#include <cstring>
#include <vector>
#include <array>

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; } \
    else { std::printf("  [FAIL] %s\n", msg); } \
} while(0)

// ============================================================================
// Test helpers
// ============================================================================

static Scalar make_key(std::uint64_t seed) {
    std::array<std::uint8_t, 32> buf{};
    for (int i = 0; i < 8; ++i)
        buf[static_cast<std::size_t>(24 + i)] = static_cast<std::uint8_t>((seed >> (8 * (7 - i))) & 0xFF);
    buf[0] = 0x01;  // ensure non-zero
    return Scalar::from_bytes(buf);
}

// ============================================================================
// Roundtrip: encrypt then decrypt recovers plaintext
// ============================================================================

static void test_roundtrip_basic() {
    std::printf("\n=== ECIES: Basic roundtrip ===\n");

    auto sk = make_key(0xEC1E5001ULL);
    auto pk = Point::generator().scalar_mul(sk);

    const std::uint8_t pt[] = "Hello, ECIES!";
    auto ct = ecies_encrypt(pk, pt, sizeof(pt) - 1);
    CHECK(!ct.empty(), "encrypt produces output");

    auto recovered = ecies_decrypt(sk, ct.data(), ct.size());
    CHECK(!recovered.empty(), "decrypt succeeds");
    CHECK(recovered.size() == sizeof(pt) - 1, "plaintext length matches");
    CHECK(std::memcmp(recovered.data(), pt, sizeof(pt) - 1) == 0,
          "plaintext content matches");
}

// ============================================================================
// Variable-length payloads: 1B, 256B, 4KB
// ============================================================================

static void test_variable_lengths() {
    std::printf("\n=== ECIES: Variable payload lengths ===\n");

    auto sk = make_key(0xEC1E5002ULL);
    auto pk = Point::generator().scalar_mul(sk);

    for (std::size_t len : {std::size_t{1}, std::size_t{256}, std::size_t{4096}}) {
        std::vector<std::uint8_t> pt(len, static_cast<std::uint8_t>(len & 0xFF));
        auto ct = ecies_encrypt(pk, pt.data(), pt.size());
        CHECK(!ct.empty(), "encrypt non-empty");
        CHECK(ct.size() > pt.size(), "ciphertext larger than plaintext");

        auto recovered = ecies_decrypt(sk, ct.data(), ct.size());
        char label[64];
        std::snprintf(label, sizeof(label), "roundtrip %zu bytes", len);
        CHECK(recovered.size() == len && std::memcmp(recovered.data(), pt.data(), len) == 0,
              label);
    }
}

// ============================================================================
// Empty plaintext
// ============================================================================

static void test_empty_plaintext() {
    std::printf("\n=== ECIES: Empty plaintext ===\n");

    auto sk = make_key(0xEC1E5003ULL);
    auto pk = Point::generator().scalar_mul(sk);

    auto ct = ecies_encrypt(pk, nullptr, 0);
    // Implementations may either produce an empty envelope or a valid one
    if (!ct.empty()) {
        auto recovered = ecies_decrypt(sk, ct.data(), ct.size());
        CHECK(recovered.empty(), "empty plaintext → empty recovery");
    } else {
        CHECK(ct.empty(), "encrypt(empty) returns empty (implementation choice)");
    }
}

// ============================================================================
// Wrong key: decryption must fail
// ============================================================================

static void test_wrong_key() {
    std::printf("\n=== ECIES: Wrong key rejection ===\n");

    auto sk1 = make_key(0xEC1E5010ULL);
    auto pk1 = Point::generator().scalar_mul(sk1);
    auto sk2 = make_key(0xEC1E5020ULL);  // different key

    const std::uint8_t pt[] = "secret data";
    auto ct = ecies_encrypt(pk1, pt, sizeof(pt) - 1);
    CHECK(!ct.empty(), "encrypt ok");

    auto bad = ecies_decrypt(sk2, ct.data(), ct.size());
    CHECK(bad.empty(), "wrong key → empty result (decryption fails)");
}

// ============================================================================
// Tamper detection: flipping a ciphertext byte must fail decryption
// ============================================================================

static void test_tamper_detection() {
    std::printf("\n=== ECIES: Tamper detection ===\n");

    auto sk = make_key(0xEC1E5030ULL);
    auto pk = Point::generator().scalar_mul(sk);

    const std::uint8_t pt[] = "integrity check payload";
    auto ct = ecies_encrypt(pk, pt, sizeof(pt) - 1);
    CHECK(!ct.empty(), "encrypt ok");

    // Flip a byte in the middle of the ciphertext
    auto tampered = ct;
    tampered[tampered.size() / 2] ^= 0xFF;
    auto bad = ecies_decrypt(sk, tampered.data(), tampered.size());
    CHECK(bad.empty(), "tampered ciphertext → decryption fails");
}

// ============================================================================
// Truncated envelope: short data must fail
// ============================================================================

static void test_truncated() {
    std::printf("\n=== ECIES: Truncated envelope ===\n");

    auto sk = make_key(0xEC1E5040ULL);
    auto pk = Point::generator().scalar_mul(sk);

    const std::uint8_t pt[] = "truncation test";
    auto ct = ecies_encrypt(pk, pt, sizeof(pt) - 1);
    CHECK(!ct.empty(), "encrypt ok");

    // Try decrypting with only half the envelope
    auto half = ecies_decrypt(sk, ct.data(), ct.size() / 2);
    CHECK(half.empty(), "truncated envelope → decryption fails");

    // Minimum overhead is 33 (pubkey) + 16 (IV) + 32 (HMAC) = 81 for CBC mode
    auto tiny = ecies_decrypt(sk, ct.data(), 10);
    CHECK(tiny.empty(), "tiny envelope → decryption fails");
}

// ============================================================================
// Determinism: two encryptions of same plaintext produce different ciphertext
// (due to ephemeral key)
// ============================================================================

static void test_nondeterminism() {
    std::printf("\n=== ECIES: Non-deterministic encryption ===\n");

    auto sk = make_key(0xEC1E5050ULL);
    auto pk = Point::generator().scalar_mul(sk);

    const std::uint8_t pt[] = "same message twice";
    auto ct1 = ecies_encrypt(pk, pt, sizeof(pt) - 1);
    auto ct2 = ecies_encrypt(pk, pt, sizeof(pt) - 1);
    CHECK(!ct1.empty() && !ct2.empty(), "both encryptions succeed");
    // Ciphertext must differ (ephemeral key randomness)
    CHECK(ct1 != ct2, "two encryptions of same plaintext differ");

    // But both decrypt to the same plaintext
    auto pt1 = ecies_decrypt(sk, ct1.data(), ct1.size());
    auto pt2 = ecies_decrypt(sk, ct2.data(), ct2.size());
    CHECK(pt1 == pt2, "both decrypt to same plaintext");
}

// ============================================================================
// Entry point
// ============================================================================

int test_ecies_run() {
    std::printf("\n========== ECIES Encryption Tests ==========\n");

    test_roundtrip_basic();
    test_variable_lengths();
    test_empty_plaintext();
    test_wrong_key();
    test_tamper_detection();
    test_truncated();
    test_nondeterminism();

    std::printf("\n  ECIES: %d/%d passed\n", tests_passed, tests_run);
    return tests_run - tests_passed;
}

#ifdef STANDALONE_TEST
int main() {
    return test_ecies_run();
}
#endif
