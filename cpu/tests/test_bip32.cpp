// ============================================================================
// Test: BIP-32 HD Key Derivation
// ============================================================================

#include "secp256k1/bip32.hpp"
#include "secp256k1/sha256.hpp"

#include <cstdio>
#include <cstring>
#include <array>

using namespace secp256k1;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        unsigned int byte = 0;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        if (sscanf(hex + i * 2, "%02x", &byte) != 1) byte = 0;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
        out[i] = static_cast<uint8_t>(byte);
    }
}

// -- HMAC-SHA512 Test ---------------------------------------------------------

static void test_hmac_sha512() {
    printf("\n--- HMAC-SHA512 ---\n");

    // RFC 4231 Test Case 2
    // Key = "Jefe" (4 bytes)
    // Data = "what do ya want for nothing?" (28 bytes)
    const char* key_str = "Jefe";
    const char* data_str = "what do ya want for nothing?";
    auto result = hmac_sha512(
        reinterpret_cast<const uint8_t*>(key_str), 4,
        reinterpret_cast<const uint8_t*>(data_str), 28);

    // Expected: 164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea250554
    //           9758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737
    uint8_t expected[64];
    hex_to_bytes("164b7a7bfcf819e2e395fbe73b56e0a387bd64222e831fd610270cd7ea250554"
                 "9758bf75c05a994a6d034f65f8f0e6fdcaeab1a34d4a6b4b636e070a38bce737",
                 expected, 64);

    CHECK(std::memcmp(result.data(), expected, 64) == 0,
          "HMAC-SHA512 RFC 4231 TC2 matches");
}

// -- BIP-32 Master Key --------------------------------------------------------

static void test_bip32_master() {
    printf("\n--- BIP-32 Master Key ---\n");

    // BIP-32 Test Vector 1: Seed = 000102030405060708090a0b0c0d0e0f
    uint8_t seed[16];
    hex_to_bytes("000102030405060708090a0b0c0d0e0f", seed, 16);

    auto [master, ok] = bip32_master_key(seed, 16);
    CHECK(ok, "Master key generation succeeds");
    CHECK(master.is_private, "Master key is private");
    CHECK(master.depth == 0, "Master depth = 0");

    // Expected chain code from BIP-32 TV1:
    // 873dff81c02f525623fd1fe5167eac3a55a049de3d314bb42ee227ffed37d508
    uint8_t expected_cc[32];
    hex_to_bytes("873dff81c02f525623fd1fe5167eac3a55a049de3d314bb42ee227ffed37d508",
                 expected_cc, 32);
    CHECK(std::memcmp(master.chain_code.data(), expected_cc, 32) == 0,
          "Master chain code matches BIP-32 TV1");

    // Expected master private key:
    // e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35
    uint8_t expected_key[32];
    hex_to_bytes("e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35",
                 expected_key, 32);
    CHECK(std::memcmp(master.key.data(), expected_key, 32) == 0,
          "Master private key matches BIP-32 TV1");
}

// -- BIP-32 Child Derivation --------------------------------------------------

static void test_bip32_derive() {
    printf("\n--- BIP-32 Child Derivation ---\n");

    uint8_t seed[16];
    hex_to_bytes("000102030405060708090a0b0c0d0e0f", seed, 16);
    auto [master, ok] = bip32_master_key(seed, 16);
    CHECK(ok, "Master key OK for derivation test");

    // Derive m/0' (hardened)
    auto [child0h, ok0] = master.derive_hardened(0);
    CHECK(ok0, "m/0' derivation succeeds");
    CHECK(child0h.depth == 1, "m/0' depth = 1");
    CHECK(child0h.is_private, "m/0' is private");

    // Verify child chain code matches BIP-32 TV1 m/0':
    // 47fdacbd0f1097043b78c63c20c34ef4ed9a111d980047ad16282c7ae6236141
    uint8_t expected_cc[32];
    hex_to_bytes("47fdacbd0f1097043b78c63c20c34ef4ed9a111d980047ad16282c7ae6236141",
                 expected_cc, 32);
    CHECK(std::memcmp(child0h.chain_code.data(), expected_cc, 32) == 0,
          "m/0' chain code matches BIP-32 TV1");

    // Public key derivation: convert to public, then do normal derivation
    auto pub_key = child0h.to_public();
    CHECK(!pub_key.is_private, "to_public() returns public key");

    // Derive m/0'/1 (normal child from m/0')
    auto [child1, ok1] = child0h.derive_normal(1);
    CHECK(ok1, "m/0'/1 derivation succeeds");
    CHECK(child1.depth == 2, "m/0'/1 depth = 2");
}

// -- BIP-32 Path Derivation ---------------------------------------------------

static void test_bip32_path() {
    printf("\n--- BIP-32 Path Derivation ---\n");

    uint8_t seed[16];
    hex_to_bytes("000102030405060708090a0b0c0d0e0f", seed, 16);
    auto [master, ok] = bip32_master_key(seed, 16);
    CHECK(ok, "Master key OK for path test");

    // Derive m/0'/1
    auto [key, ok2] = bip32_derive_path(master, "m/0'/1");
    CHECK(ok2, "Path m/0'/1 succeeds");
    CHECK(key.depth == 2, "Path m/0'/1 depth = 2");

    // Derive m/0'/1/2' -- deeper path
    auto [key3, ok3] = bip32_derive_path(master, "m/0'/1/2'");
    CHECK(ok3, "Path m/0'/1/2' succeeds");
    CHECK(key3.depth == 3, "Path m/0'/1/2' depth = 3");

    // Invalid paths
    auto [_, ok_bad1] = bip32_derive_path(master, "");
    (void)_;
    CHECK(!ok_bad1, "Empty path fails");
    auto [_2, ok_bad2] = bip32_derive_path(master, "x/0");
    (void)_2;
    CHECK(!ok_bad2, "Path not starting with 'm' fails");
}

// -- Serialization ------------------------------------------------------------

static void test_bip32_serialize() {
    printf("\n--- BIP-32 Serialization ---\n");

    uint8_t seed[16];
    hex_to_bytes("000102030405060708090a0b0c0d0e0f", seed, 16);
    auto [master, ok] = bip32_master_key(seed, 16);
    (void)ok;

    auto ser = master.serialize();
    CHECK(ser.size() == 78, "Serialized key is 78 bytes");

    // Check version bytes (xprv = 0x0488ADE4)
    CHECK(ser[0] == 0x04 && ser[1] == 0x88 && ser[2] == 0xAD && ser[3] == 0xE4,
          "Serialized version = xprv");

    // Check depth
    CHECK(ser[4] == 0, "Serialized depth = 0");

    // Fingerprint
    auto fp = master.fingerprint();
    (void)fp;
    CHECK(fp.size() == 4, "Fingerprint is 4 bytes");
}

// -- Seed Length Validation ---------------------------------------------------

static void test_bip32_seed_validation() {
    printf("\n--- BIP-32 Seed Validation ---\n");

    uint8_t short_seed[8] = {1,2,3,4,5,6,7,8};
    auto [_, ok_short] = bip32_master_key(short_seed, 8);
    (void)_;
    CHECK(!ok_short, "Seed < 16 bytes rejected");

    uint8_t seed_16[16] = {};
    // Fill with non-zero to avoid zero key
    for (int i = 0; i < 16; ++i) seed_16[i] = static_cast<uint8_t>(i + 1);
    auto [k16, ok_16] = bip32_master_key(seed_16, 16);
    (void)k16;
    CHECK(ok_16, "16-byte seed accepted");

    uint8_t seed_64[64] = {};
    for (int i = 0; i < 64; ++i) seed_64[i] = static_cast<uint8_t>(i + 1);
    auto [k64, ok_64] = bip32_master_key(seed_64, 64);
    (void)k64;
    CHECK(ok_64, "64-byte seed accepted");
}

// -- Main ---------------------------------------------------------------------

int test_bip32_run() {
    printf("=== BIP-32 HD Key Derivation Tests ===\n");

    test_hmac_sha512();
    test_bip32_master();
    test_bip32_derive();
    test_bip32_path();
    test_bip32_serialize();
    test_bip32_seed_validation();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
