// ============================================================================
// Test: SHA-256 and SHA-512 Standalone
// ============================================================================
// NIST / RFC test vectors for SHA-256 and SHA-512.
// Covers empty input, short input, two-block input, and hash256 (double-SHA).
// ============================================================================

#include "secp256k1/sha256.hpp"
#include "secp256k1/sha512.hpp"

#include <cstdio>
#include <cstring>
#include <array>

using namespace secp256k1;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; } \
    else { std::printf("  [FAIL] %s\n", msg); } \
} while(0)

// Hex-decode helper (compile-time safe for known test vectors)
static std::array<std::uint8_t, 32> hex32(const char* hex) {
    std::array<std::uint8_t, 32> out{};
    for (int i = 0; i < 32; ++i) {
        auto nibble = [](char c) -> std::uint8_t {
            if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<std::uint8_t>(c - 'A' + 10);
            return 0;
        };
        out[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(
            (nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
    }
    return out;
}

static std::array<std::uint8_t, 64> hex64(const char* hex) {
    std::array<std::uint8_t, 64> out{};
    for (int i = 0; i < 64; ++i) {
        auto nibble = [](char c) -> std::uint8_t {
            if (c >= '0' && c <= '9') return static_cast<std::uint8_t>(c - '0');
            if (c >= 'a' && c <= 'f') return static_cast<std::uint8_t>(c - 'a' + 10);
            if (c >= 'A' && c <= 'F') return static_cast<std::uint8_t>(c - 'A' + 10);
            return 0;
        };
        out[static_cast<std::size_t>(i)] = static_cast<std::uint8_t>(
            (nibble(hex[2 * i]) << 4) | nibble(hex[2 * i + 1]));
    }
    return out;
}

// ============================================================================
// SHA-256 Test Vectors (NIST FIPS 180-4)
// ============================================================================

static void test_sha256_empty() {
    std::printf("\n=== SHA-256: Empty input ===\n");
    auto h = SHA256::hash(nullptr, 0);
    auto exp = hex32("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    CHECK(h == exp, "SHA-256(\"\") matches NIST");
}

static void test_sha256_abc() {
    std::printf("\n=== SHA-256: \"abc\" ===\n");
    const char* msg = "abc";
    auto h = SHA256::hash(msg, 3);
    auto exp = hex32("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(h == exp, "SHA-256(\"abc\") matches NIST");
}

static void test_sha256_two_block() {
    std::printf("\n=== SHA-256: Two-block message ===\n");
    // "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
    const char* msg = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    auto h = SHA256::hash(msg, 56);
    auto exp = hex32("248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
    CHECK(h == exp, "SHA-256(two-block) matches NIST");
}

static void test_sha256_hash256() {
    std::printf("\n=== SHA-256: Double-hash (hash256) ===\n");
    // Bitcoin's hash256 = SHA256(SHA256(x))
    const char* msg = "abc";
    auto h = SHA256::hash256(msg, 3);
    // SHA256(SHA256("abc")) = known value
    auto inner = SHA256::hash(msg, 3);
    auto expected = SHA256::hash(inner.data(), inner.size());
    CHECK(h == expected, "hash256 == SHA256(SHA256(data))");
}

static void test_sha256_incremental() {
    std::printf("\n=== SHA-256: Incremental update ===\n");
    // Hash "abc" as 'a' + 'bc'
    SHA256 ctx;
    ctx.update("a", 1);
    ctx.update("bc", 2);
    auto h = ctx.finalize();
    auto exp = hex32("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    CHECK(h == exp, "incremental SHA-256(\"abc\") matches one-shot");
}

// ============================================================================
// SHA-512 Test Vectors (NIST FIPS 180-4)
// ============================================================================

static void test_sha512_empty() {
    std::printf("\n=== SHA-512: Empty input ===\n");
    auto h = SHA512::hash(nullptr, 0);
    auto exp = hex64(
        "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
        "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
    CHECK(h == exp, "SHA-512(\"\") matches NIST");
}

static void test_sha512_abc() {
    std::printf("\n=== SHA-512: \"abc\" ===\n");
    const char* msg = "abc";
    auto h = SHA512::hash(msg, 3);
    auto exp = hex64(
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
        "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
    CHECK(h == exp, "SHA-512(\"abc\") matches NIST");
}

static void test_sha512_two_block() {
    std::printf("\n=== SHA-512: Two-block message ===\n");
    const char* msg = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
                      "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    auto h = SHA512::hash(msg, 112);
    auto exp = hex64(
        "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018"
        "501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909");
    CHECK(h == exp, "SHA-512(two-block) matches NIST");
}

static void test_sha512_incremental() {
    std::printf("\n=== SHA-512: Incremental update ===\n");
    SHA512 ctx;
    ctx.update("a", 1);
    ctx.update("bc", 2);
    auto h = ctx.finalize();
    auto exp = hex64(
        "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
        "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
    CHECK(h == exp, "incremental SHA-512(\"abc\") matches one-shot");
}

// ============================================================================
// Entry point
// ============================================================================

int test_sha_run() {
    std::printf("\n========== SHA-256 / SHA-512 Test Vectors ==========\n");

    test_sha256_empty();
    test_sha256_abc();
    test_sha256_two_block();
    test_sha256_hash256();
    test_sha256_incremental();

    test_sha512_empty();
    test_sha512_abc();
    test_sha512_two_block();
    test_sha512_incremental();

    std::printf("\n  SHA: %d/%d passed\n", tests_passed, tests_run);
    return tests_run - tests_passed;
}

#ifdef STANDALONE_TEST
int main() {
    return test_sha_run();
}
#endif
