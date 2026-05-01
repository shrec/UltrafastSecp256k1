// ============================================================================
// Test: Ethereum Signing Layer (EIP-191, EIP-155, ecrecover)
// ============================================================================
// Tests:
//   1. Keccak-256 -- known test vectors
//   2. EIP-191 -- personal message hashing
//   3. EIP-155 -- chain ID encoding/decoding
//   4. eth_sign_hash -- ECDSA sign with recovery (v,r,s)
//   5. eth_personal_sign -- full MetaMask personal_sign flow
//   6. ecrecover -- address recovery from signature
//   7. eth_personal_verify -- verify personal_sign signature
//   8. Round-trip -- sign + recover + verify
//   9. Multi-chain -- EIP-155 with various chain IDs
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <array>

#include "secp256k1/coins/keccak256.hpp"
#include "secp256k1/coins/ethereum.hpp"
#include "secp256k1/coins/eth_signing.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using namespace secp256k1;
using namespace secp256k1::coins;
using fast::Scalar;
using fast::Point;

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { std::printf("  [TEST] %-55s ", name); } while(0)

#define PASS() \
    do { std::printf("PASS\n"); ++tests_passed; } while(0)

#define FAIL(msg) \
    do { std::printf("FAIL: %s\n", msg); ++tests_failed; } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while(0)

#define ASSERT_EQ(a, b, msg) \
    do { if ((a) != (b)) { FAIL(msg); return; } } while(0)

static void hex_to_bytes(const char* hex, uint8_t* out, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        char pair[3] = { hex[i * 2], hex[i * 2 + 1], '\0' };
        char* endptr = nullptr;
        const unsigned long val = std::strtoul(pair, &endptr, 16);
        out[i] = (endptr == pair + 2) ? static_cast<uint8_t>(val) : 0;
    }
}

static std::string bytes_to_hex(const uint8_t* data, size_t len) {
    static const char hex[] = "0123456789abcdef";
    std::string result(len * 2, '\0');
    for (size_t i = 0; i < len; ++i) {
        result[i * 2]     = hex[data[i] >> 4];
        result[i * 2 + 1] = hex[data[i] & 0x0F];
    }
    return result;
}

// ============================================================================
// 1. EIP-155 Chain ID encoding/decoding
// ============================================================================

static void test_eip155_encoding() {
    std::printf("\n--- EIP-155 Chain ID ---\n");

    // Legacy: v=27 -> recid=0, chain_id=0
    TEST("Legacy v=27 -> recid=0");
    ASSERT_EQ(eip155_recid(27), 0, "recid mismatch");
    ASSERT_EQ(eip155_chain_id(27), 0u, "chain_id mismatch");
    PASS();

    TEST("Legacy v=28 -> recid=1");
    ASSERT_EQ(eip155_recid(28), 1, "recid mismatch");
    ASSERT_EQ(eip155_chain_id(28), 0u, "chain_id mismatch");
    PASS();

    // Ethereum mainnet chain_id=1
    TEST("EIP-155 Ethereum mainnet (chain_id=1)");
    ASSERT_EQ(eip155_v(0, 1), 37u, "v=35+2*1+0=37");
    ASSERT_EQ(eip155_v(1, 1), 38u, "v=35+2*1+1=38");
    ASSERT_EQ(eip155_recid(37), 0, "recid from v=37");
    ASSERT_EQ(eip155_recid(38), 1, "recid from v=38");
    ASSERT_EQ(eip155_chain_id(37), 1u, "chain_id from v=37");
    ASSERT_EQ(eip155_chain_id(38), 1u, "chain_id from v=38");
    PASS();

    // BSC chain_id=56
    TEST("EIP-155 BSC (chain_id=56)");
    ASSERT_EQ(eip155_v(0, 56), 147u, "v=35+112+0=147");
    ASSERT_EQ(eip155_v(1, 56), 148u, "v=35+112+1=148");
    ASSERT_EQ(eip155_chain_id(147), 56u, "chain_id from v=147");
    PASS();

    // Polygon chain_id=137
    TEST("EIP-155 Polygon (chain_id=137)");
    ASSERT_EQ(eip155_v(0, 137), 309u, "v=35+274+0=309");
    ASSERT_EQ(eip155_chain_id(309), 137u, "chain_id from v=309");
    PASS();

    // Round-trip for many chains
    TEST("EIP-155 round-trip (10 chains)");
    uint64_t const chains[] = {1, 3, 4, 5, 42, 56, 97, 137, 250, 43114};
    for (auto cid : chains) {
        for (int recid = 0; recid < 2; ++recid) {
            uint64_t const v = eip155_v(recid, cid);
            ASSERT_EQ(eip155_recid(v), recid, "recid roundtrip fail");
            ASSERT_EQ(eip155_chain_id(v), cid, "chain_id roundtrip fail");
        }
    }
    PASS();
}

// ============================================================================
// 2. EIP-191 Personal Message Hash
// ============================================================================

static void test_eip191_hash() {
    std::printf("\n--- EIP-191 Personal Message Hash ---\n");

    // Known vector: "Hello, world!" with EIP-191 prefix
    // "\x19Ethereum Signed Message:\n13Hello, world!"
    TEST("EIP-191 hash 'Hello, world!'");
    const char* msg = "Hello, world!";
    auto hash = eip191_hash(reinterpret_cast<const uint8_t*>(msg), 13);
    // Hash should be 32 bytes, not all zeros
    bool all_zero = true;
    for (auto b : hash) { if (b != 0) all_zero = false; }
    ASSERT_TRUE(!all_zero, "hash should not be all zeros");
    PASS();

    // Empty message
    TEST("EIP-191 hash empty message");
    auto hash_empty = eip191_hash(nullptr, 0);
    // "\x19Ethereum Signed Message:\n0"
    all_zero = true;
    for (auto b : hash_empty) { if (b != 0) all_zero = false; }
    ASSERT_TRUE(!all_zero, "empty msg hash should not be zero");
    PASS();

    // Different messages produce different hashes
    TEST("EIP-191 different messages -> different hashes");
    const char* msg2 = "Hello, World!";  // capital W
    auto hash2 = eip191_hash(reinterpret_cast<const uint8_t*>(msg2), 13);
    ASSERT_TRUE(hash != hash2, "different msgs should produce different hashes");
    PASS();

    // Deterministic: same input -> same output
    TEST("EIP-191 deterministic");
    auto hash3 = eip191_hash(reinterpret_cast<const uint8_t*>(msg), 13);
    ASSERT_EQ(hash, hash3, "same input should give same hash");
    PASS();
}

// ============================================================================
// 3. eth_sign_hash -- ECDSA sign with recovery
// ============================================================================

static void test_eth_sign_hash() {
    std::printf("\n--- eth_sign_hash ---\n");

    // Private key: 1
    std::array<uint8_t, 32> sk_bytes{};
    sk_bytes[31] = 1;
    Scalar const sk = Scalar::from_bytes(sk_bytes);

    // Sign a known hash
    std::array<uint8_t, 32> hash{};
    hash[31] = 42;

    TEST("eth_sign_hash basic (legacy, chain_id=0)");
    auto sig = eth_sign_hash(hash, sk, 0);
    // r and s should be non-zero
    bool r_zero = true, s_zero = true;
    for (auto b : sig.r) { if (b != 0) r_zero = false; }
    for (auto b : sig.s) { if (b != 0) s_zero = false; }
    ASSERT_TRUE(!r_zero, "r should be non-zero");
    ASSERT_TRUE(!s_zero, "s should be non-zero");
    // v should be 27 or 28 for legacy
    {
        const bool v_ok = (sig.v == 27 || sig.v == 28);
        ASSERT_TRUE(v_ok, "legacy v should be 27 or 28");
    }
    PASS();

    // Sign with Ethereum mainnet chain ID
    TEST("eth_sign_hash with chain_id=1 (Ethereum)");
    auto sig2 = eth_sign_hash(hash, sk, 1);
    {
        const bool v2_ok = (sig2.v == 37 || sig2.v == 38);
        ASSERT_TRUE(v2_ok, "EIP-155 v should be 37 or 38");
    }
    PASS();

    // Same hash + key should give same r,s
    TEST("eth_sign_hash deterministic (RFC 6979)");
    auto sig3 = eth_sign_hash(hash, sk, 0);
    ASSERT_EQ(sig.r, sig3.r, "r should be deterministic");
    ASSERT_EQ(sig.s, sig3.s, "s should be deterministic");
    PASS();

    // Different hash should give different signature
    TEST("eth_sign_hash different hash -> different sig");
    std::array<uint8_t, 32> hash2{};
    hash2[31] = 43;
    auto sig4 = eth_sign_hash(hash2, sk, 0);
    ASSERT_TRUE(sig.r != sig4.r || sig.s != sig4.s, "different hash should give different sig");
    PASS();
}

// ============================================================================
// 4. ecrecover -- Round-trip sign + recover
// ============================================================================

static void test_ecrecover() {
    std::printf("\n--- ecrecover ---\n");

    // Generate a key pair
    std::array<uint8_t, 32> sk_bytes{};
    hex_to_bytes("c6b506e21f3c26dfe9b3a15a40d2dde0ab9ee4bb9e6f7e6e49f7ef9fd9b3a3d5",
                 sk_bytes.data(), 32);
    Scalar const sk = Scalar::from_bytes(sk_bytes);
    const Point pk = Point::generator().scalar_mul(sk);
    const auto expected_addr = ethereum_address_bytes(pk);

    // Hash a message
    std::array<uint8_t, 32> hash{};
    hex_to_bytes("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2",
                 hash.data(), 32);

    // Sign
    TEST("ecrecover round-trip");
    auto sig = eth_sign_hash(hash, sk, 0);

    // Recover
    auto [recovered_addr, ok] = ecrecover(hash, sig);
    ASSERT_TRUE(ok, "ecrecover should succeed");
    ASSERT_TRUE(recovered_addr == expected_addr, "recovered address should match");
    PASS();

    // ecrecover with EIP-155
    TEST("ecrecover with EIP-155 chain_id=1");
    auto sig2 = eth_sign_hash(hash, sk, 1);
    auto [addr2, ok2] = ecrecover(hash, sig2);
    ASSERT_TRUE(ok2, "ecrecover should succeed with EIP-155");
    ASSERT_TRUE(addr2 == expected_addr, "recovered address should match with EIP-155");
    PASS();

    // ecrecover with wrong hash should give different address
    TEST("ecrecover wrong hash -> wrong address");
    std::array<uint8_t, 32> wrong_hash{};
    wrong_hash[31] = 99;
    auto [wrong_addr, ok3] = ecrecover(wrong_hash, sig);
    if (ok3) {
        ASSERT_TRUE(wrong_addr != expected_addr, "wrong hash should give different address");
    }
    PASS();

    // ecrecover with invalid r=0 should fail
    TEST("ecrecover invalid r=0");
    const std::array<uint8_t, 32> zero{};
    auto [_, ok4] = ecrecover(hash, zero, sig.s, sig.v);
    ASSERT_TRUE(!ok4, "ecrecover with r=0 should fail");
    PASS();

    // ecrecover with invalid s=0 should fail
    TEST("ecrecover invalid s=0");
    auto [_2, ok5] = ecrecover(hash, sig.r, zero, sig.v);
    ASSERT_TRUE(!ok5, "ecrecover with s=0 should fail");
    PASS();
}

// ============================================================================
// 5. eth_personal_sign + eth_personal_verify
// ============================================================================

static void test_personal_sign() {
    std::printf("\n--- eth_personal_sign + verify ---\n");

    std::array<uint8_t, 32> sk_bytes{};
    hex_to_bytes("4c0883a69102937d6231471b5dbb6204fe512961708279f8f30ab5c5dbe3a2b7",
                 sk_bytes.data(), 32);
    Scalar const sk = Scalar::from_bytes(sk_bytes);
    const Point pk = Point::generator().scalar_mul(sk);
    auto addr = ethereum_address_bytes(pk);

    const char* msg = "I agree to the terms of service";
    size_t const msg_len = std::strlen(msg);

    TEST("personal_sign basic");
    auto sig = eth_personal_sign(reinterpret_cast<const uint8_t*>(msg), msg_len, sk);
    bool r_zero = true;
    for (auto b : sig.r) { if (b != 0) r_zero = false; }
    ASSERT_TRUE(!r_zero, "r should be non-zero");
    {
        const bool v_ok2 = (sig.v == 27 || sig.v == 28);
        ASSERT_TRUE(v_ok2, "v should be 27 or 28");
    }
    PASS();

    TEST("personal_verify valid");
    const bool valid = eth_personal_verify(
        reinterpret_cast<const uint8_t*>(msg), msg_len, sig, addr);
    ASSERT_TRUE(valid, "signature should verify");
    PASS();

    TEST("personal_verify wrong message");
    const char* wrong_msg = "I disagree to the terms of service";
    const bool wrong = eth_personal_verify(
        reinterpret_cast<const uint8_t*>(wrong_msg), std::strlen(wrong_msg), sig, addr);
    ASSERT_TRUE(!wrong, "wrong message should not verify");
    PASS();

    TEST("personal_verify wrong address");
    std::array<uint8_t, 20> wrong_addr{};
    wrong_addr[0] = 0xFF;
    const bool wrong2 = eth_personal_verify(
        reinterpret_cast<const uint8_t*>(msg), msg_len, sig, wrong_addr);
    ASSERT_TRUE(!wrong2, "wrong address should not verify");
    PASS();
}

// ============================================================================
// 6. Multi-chain round-trip
// ============================================================================

static void test_multi_chain() {
    std::printf("\n--- Multi-chain EIP-155 ---\n");

    std::array<uint8_t, 32> sk_bytes{};
    sk_bytes[31] = 7;
    Scalar const sk = Scalar::from_bytes(sk_bytes);
    const Point pk = Point::generator().scalar_mul(sk);
    const auto expected_addr = ethereum_address_bytes(pk);

    std::array<uint8_t, 32> hash{};
    hash[0] = 0xAB; hash[1] = 0xCD;

    struct ChainTest { uint64_t id; const char* name; };
    ChainTest const chains[] = {
        {1,     "Ethereum"},
        {56,    "BSC"},
        {137,   "Polygon"},
        {43114, "Avalanche"},
        {250,   "Fantom"},
        {42161, "Arbitrum"},
        {10,    "Optimism"},
    };

    for (auto& chain : chains) {
        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "Round-trip chain_id=%lu (%s)",
                     static_cast<unsigned long>(chain.id), chain.name);
        TEST(buf);

        auto sig = eth_sign_hash(hash, sk, chain.id);

        // v should encode correct chain ID
        ASSERT_EQ(eip155_chain_id(sig.v), chain.id, "chain_id mismatch in v");

        // Recover should give correct address
        auto [addr, ok] = ecrecover(hash, sig);
        ASSERT_TRUE(ok, "ecrecover should succeed");
        ASSERT_TRUE(addr == expected_addr, "recovered address should match");
        PASS();
    }
}

// ============================================================================
// 7. Keccak-256 test vectors
// ============================================================================

static void test_keccak256_vectors() {
    std::printf("\n--- Keccak-256 Test Vectors ---\n");

    // Empty string: Keccak-256("")
    TEST("Keccak-256 empty string");
    auto hash = keccak256(nullptr, 0);
    std::string hex = bytes_to_hex(hash.data(), 32);
    ASSERT_EQ(hex, std::string("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
              "empty Keccak-256 mismatch");
    PASS();

    // "abc"
    TEST("Keccak-256 'abc'");
    const char* abc = "abc";
    hash = keccak256(reinterpret_cast<const uint8_t*>(abc), 3);
    hex = bytes_to_hex(hash.data(), 32);
    ASSERT_EQ(hex, std::string("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
              "'abc' Keccak-256 mismatch");
    PASS();
}

// ============================================================================
// Main
// ============================================================================

#ifdef STANDALONE_TEST
int main() {
#else
int test_ethereum_run() {
#endif
    std::printf("\n========================================\n");
    std::printf("  Ethereum Signing Layer Tests\n");
    std::printf("========================================\n");

    test_eip155_encoding();
    test_eip191_hash();
    test_eth_sign_hash();
    test_ecrecover();
    test_personal_sign();
    test_multi_chain();
    test_keccak256_vectors();

    std::printf("\n========================================\n");
    std::printf("  Result: %d passed, %d failed (total %d)\n",
           tests_passed, tests_failed, tests_passed + tests_failed);
    std::printf("========================================\n");

#ifdef STANDALONE_TEST
    return tests_failed > 0 ? EXIT_FAILURE : EXIT_SUCCESS;
#else
    return tests_failed;
#endif
}
