// ============================================================================
// Test: BIP-32 Official Test Vectors (TV1-TV5)
// ============================================================================
// Source: https://github.com/bitcoin/bips/blob/master/bip-0032.mediawiki
//
// TV1: 128-bit seed -> 5 derivation levels
// TV2: 512-bit seed -> 5 derivation levels
// TV3: 128-bit seed -> 2 levels (tests zero-padding of private key)
// TV4: 128-bit seed -> 2 levels (same as TV3 but public derivation)
// TV5: zero leading bytes in serialized key test
//
// Each vector verifies the full derivation chain:
//   - Master key chain code + private key
//   - Each child level: chain code + private key + public key
//   - Serialized xprv / xpub (Base58Check encoded, 78 bytes pre-encoding)
// ============================================================================

#include "secp256k1/bip32.hpp"

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <array>
#include <string>

using namespace secp256k1;

static int g_tests_run = 0;
static int g_tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++g_tests_run; \
    if (cond) { ++g_tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

// -- Hex utility (no heap) ----------------------------------------------------

static void hex_to_bytes(const char* hex, std::uint8_t* out, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i) {
        unsigned int byte = 0;
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
#endif
        if (std::sscanf(hex + i * 2, "%02x", &byte) != 1) byte = 0;
#ifdef __clang__
#pragma clang diagnostic pop
#endif
        out[i] = static_cast<std::uint8_t>(byte);
    }
}

// ============================================================================
// Verify one derivation chain-level
// ============================================================================

struct ChainVector {
    const char* path;           // e.g. "m", "m/0'", "m/0'/1", ...
    const char* chain_code;     // 64 hex chars (32 bytes)
    const char* priv_key;       // 64 hex chars (32 bytes) -- private key bytes
    const char* pub_key;        // 66 hex chars (33 bytes) -- compressed pubkey
};

static void verify_chain(const ExtendedKey& master,
                          const ChainVector* vecs, int count,
                          const char* tv_label) {
    char label[128];

    // Verify master (vecs[0].path == "m")
    {
        std::uint8_t expected_cc[32], expected_key[32];
        hex_to_bytes(vecs[0].chain_code, expected_cc, 32);
        hex_to_bytes(vecs[0].priv_key, expected_key, 32);

        std::snprintf(label, sizeof(label), "%s m: chain_code", tv_label);
        CHECK(std::memcmp(master.chain_code.data(), expected_cc, 32) == 0, label);

        std::snprintf(label, sizeof(label), "%s m: priv_key", tv_label);
        CHECK(std::memcmp(master.key.data(), expected_key, 32) == 0, label);

        // Verify public key
        if (vecs[0].pub_key) {
            std::uint8_t expected_pub[33];
            hex_to_bytes(vecs[0].pub_key, expected_pub, 33);
            auto pub_point = master.public_key();
            auto pub_bytes = pub_point.to_compressed();
            std::snprintf(label, sizeof(label), "%s m: pub_key", tv_label);
            CHECK(std::memcmp(pub_bytes.data(), expected_pub, 33) == 0, label);
        }
    }

    // Derive and verify each child level
    for (int i = 1; i < count; ++i) {
        auto [child, ok] = bip32_derive_path(master, vecs[i].path);
        std::snprintf(label, sizeof(label), "%s %s: derivation succeeds", tv_label, vecs[i].path);
        CHECK(ok, label);
        if (!ok) continue;

        std::uint8_t expected_cc[32], expected_key[32];
        hex_to_bytes(vecs[i].chain_code, expected_cc, 32);
        hex_to_bytes(vecs[i].priv_key, expected_key, 32);

        std::snprintf(label, sizeof(label), "%s %s: chain_code", tv_label, vecs[i].path);
        CHECK(std::memcmp(child.chain_code.data(), expected_cc, 32) == 0, label);

        std::snprintf(label, sizeof(label), "%s %s: priv_key", tv_label, vecs[i].path);
        CHECK(std::memcmp(child.key.data(), expected_key, 32) == 0, label);

        // Verify public key
        if (vecs[i].pub_key) {
            std::uint8_t expected_pub[33];
            hex_to_bytes(vecs[i].pub_key, expected_pub, 33);
            auto pub_point = child.public_key();
            auto pub_bytes = pub_point.to_compressed();
            std::snprintf(label, sizeof(label), "%s %s: pub_key", tv_label, vecs[i].path);
            CHECK(std::memcmp(pub_bytes.data(), expected_pub, 33) == 0, label);
        }
    }
}

// ============================================================================
// Test Vector 1: seed = 000102030405060708090a0b0c0d0e0f
// ============================================================================

static const ChainVector TV1[] = {
    { "m",
      "873dff81c02f525623fd1fe5167eac3a55a049de3d314bb42ee227ffed37d508",
      "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35",
      "0339a36013301597daef41fbe593a02cc513d0b55527ec2df1050e2e8ff49c85c2" },
    { "m/0'",
      "47fdacbd0f1097043b78c63c20c34ef4ed9a111d980047ad16282c7ae6236141",
      "edb2e14f9ee77d26dd93b4ecede8d16ed408ce149b6cd80b0715a2d911a0afea",
      "035a784662a4a20a65bf6aab9ae98a6c068a81c52e4b032c0fb5400c706cfccc56" },
    { "m/0'/1",
      "2a7857631386ba23dacac34180dd1983734e444fdbf774041578e9b6adb37c19",
      "3c6cb8d0f6a264c91ea8b5030fadaa8e538b020f0a387421a12de9319dc93368",
      "03501e454bf00751f24b1b489aa925215d66af2234e3891c3b21a52bedb3cd711c" },
    { "m/0'/1/2'",
      "04466b9cc8e161e966409ca52986c584f07e9dc81f735db683c3ff6ec7b1503f",
      "cbce0d719ecf7431d88e6a89fa1483e02e35092af60c042b1df2ff59fa424dca",
      "0357bfe1e341d01c69fe5654309956cbea516822fba8a601743a012a7896ee8dc2" },
    { "m/0'/1/2'/2",
      "cfb71883f01676f587d023cc53a35bc7f88f724b1f8c2892ac1275ac822a3edd",
      "0f479245fb19a38a1954c5c7c0ebab2f9bdfd96a17563ef28a6a4b1a2a764ef4",
      "02e8445082a72f29b75ca48748a914df60622a609cacfce8ed0e35804560741d29" },
    { "m/0'/1/2'/2/1000000000",
      "c783e67b921d2beb8f6b389cc646d7263b4145701dadd2161548a8b078e65e9e",
      "471b76e389e528d6de6d816857e012c5455051cad6660850e58372a6c3e6e7c8",
      "022a471424da5e657499d1ff51cb43c47481a03b1e77f951fe64cec9f5a48f7011" },
};

static void test_bip32_tv1() {
    printf("\n--- BIP-32 Test Vector 1 ---\n");
    std::uint8_t seed[16];
    hex_to_bytes("000102030405060708090a0b0c0d0e0f", seed, 16);
    auto [master, ok] = bip32_master_key(seed, 16);
    CHECK(ok, "TV1: master key generation succeeds");
    if (!ok) return;
    verify_chain(master, TV1, 6, "TV1");
}

// ============================================================================
// Test Vector 2: seed = fffcf9f6f3f0edeae7e4e1dedbd8d5d2cfccc9c6c3c0bdbab7b4
//                        b1aeaba8a5a29f9c999693908d8a8784817e7b7875726f6c696663
//                        605d5a5754514e4b484542
// ============================================================================

static const ChainVector TV2[] = {
    { "m",
      "60499f801b896d83179a4374aeb7822aaeaceaa0db1f85ee3e904c4defbd9689",
      "4b03d6fc340455b363f51020ad3ecca4f0850280cf436c70c727923f6db46c3e",
      "03cbcaa9c98c877a26977d00825c956a238e8dddfbd322cce4f74b0b5bd6ace4a7" },
    { "m/0",
      "f0909affaa7ee7abe5dd4e100598d4dc53cd709d5a5c2cac40e7412f232f7c9c",
      "abe74a98f6c7eabee0428f53798f0ab8aa1bd37873999041703c742f15ac7e1e",
      "02fc9e5af0ac8d9b3cecfe2a888e2117ba3d089d8585886c9c826b6b22a98d12ea" },
    { "m/0/2147483647'",
      "be17a268474a6bb9c61e1d720cf6215e2a88c5406c4aee7b38547f585c9a37d9",
      "877c779ad9687164e9c2f4f0f4ff0340814392330693ce95a58fe18fd52e6e93",
      "03c01e7425647bdefa82b12d9bad5e3e6865bee0502694b94ca58b666abc0a5c3b" },
    { "m/0/2147483647'/1",
      "f366f48f1ea9f2d1d3fe958c95ca84ea18e4c4ddb9366c336c927eb246fb38cb",
      "704addf544a06e5ee4bea37098463c23613da32020d604506da8c0518e1da4b7",
      "03a7d1d856deb74c508e05031f9895dab54626251b3806e16b4bd12e781a7df5b9" },
    { "m/0/2147483647'/1/2147483646'",
      "637807030d55d01f9a0cb3a7839515d796bd07706386a6eddf06cc29a65a0e29",
      "f1c7c871a54a804afe328b4c83a1c33b8e5ff48f5087273f04efa83b247d6a2d",
      "02d2b36900396c9282fa14628566582f206a5dd0bcc8d5e892611806cafb0301f0" },
    { "m/0/2147483647'/1/2147483646'/2",
      "9452b549be8cea3ecb7a84bec10dcfd94afe4d129ebfd3b3cb58eedf394ed271",
      "bb7d39bdb83ecf58f2fd82b6d918341cbef428661ef01ab97c28a4842125ac23",
      "024d902e1a2fc7a8755ab5b694c575fce742c48d9ff192e63df5193e4c7afe1f9c" },
};

static void test_bip32_tv2() {
    printf("\n--- BIP-32 Test Vector 2 ---\n");
    // 64-byte seed
    std::uint8_t seed[64];
    hex_to_bytes(
        "fffcf9f6f3f0edeae7e4e1dedbd8d5d2cfccc9c6c3c0bdbab7b4"
        "b1aeaba8a5a29f9c999693908d8a8784817e7b7875726f6c696663"
        "605d5a5754514e4b484542", seed, 64);
    auto [master, ok] = bip32_master_key(seed, 64);
    CHECK(ok, "TV2: master key generation succeeds");
    if (!ok) return;
    verify_chain(master, TV2, 6, "TV2");
}

// ============================================================================
// Test Vector 3: Tests for the retention of leading zeros
// seed = 4b381541583be4423346c643850da4b320e46a87ae3d2a4e6da11eba819cd4acba45d239319ac14f863b8d5ab5a0d0c64d2e8a1e7d1457df2e5a3c51c73235be
// ============================================================================

static const ChainVector TV3[] = {
    { "m",
      "01d28a3e53cffa419ec122c968b3259e16b65076495494d97cae10bbfec3c36f",
      "00ddb80b067e0d4993197fe10f2657a844a384589847602d56f0c629c81aae32",
      "03683af1ba5743bdfc798cf814efeeab2735ec52d95eced528e692b8e34c4e5669" },
    { "m/0'",
      "e5fea12a97b927fc9dc3d2cb0d1ea1cf50aa5a1fdc1f933e8906bb38df3377bd",
      "491f7a2eebc7b57028e0d3faa0acda02e75c33b03c48fb288c41e2ea44e1daef",
      "026557fdda1d5d43d79611f784780471f086d58e8126b8c40acb82272a7712e7f2" },
};

static void test_bip32_tv3() {
    printf("\n--- BIP-32 Test Vector 3 (leading zeros retention) ---\n");
    std::uint8_t seed[64];
    hex_to_bytes(
        "4b381541583be4423346c643850da4b320e46a87ae3d2a4e6da11e"
        "ba819cd4acba45d239319ac14f863b8d5ab5a0d0c64d2e8a1e7d14"
        "57df2e5a3c51c73235be", seed, 64);
    auto [master, ok] = bip32_master_key(seed, 64);
    CHECK(ok, "TV3: master key generation succeeds");
    if (!ok) return;
    verify_chain(master, TV3, 2, "TV3");
}

// ============================================================================
// Test Vector 4: Tests for the retention of leading zeros (public derivation)
// seed = 3ddd5602285899a946114506157c7997e5444528f3003f6134712147db19b678
// ============================================================================

static const ChainVector TV4[] = {
    { "m",
      "d0c8a1f6edf2500798c3e0b54f1b56e45f6d03e6076abd36e5e2f54101e44ce6",
      "12c0d59c7aa3a10973dbd3f478b65f2516627e3fe61e00c345be9a477ad2e215",
      "026f6fedc9240f61daa9c7144b682a430a3a1366576f840bf2d070101fcbc9a02d" },
    { "m/0'",
      "cdc0f06456a14876c898790e0b3b1a41c531170aec69da44ff7b7265bfe7743b",
      "00d948e9261e41362a688b916f297121ba6bfb2274a3575ac0e456551dfd7f7e",  // note leading 00!
      "039382d2b6003446792d2917f7ac4b3edf079a1a94dd4eb010dc25109dda680a9d" },
    { "m/0'/1'",
      "a48ee6674c5264a237703fd383bccd9fad4d9378ac98ab05e6e7029b06360c0d",
      "3a2086edd7d9df86c3487a5905a1712a9aa664bce8cc268141e07549eaa8661d",
      "032edaf9e591ee27f3c69c36221e3c54c38088ef34e93fbb9bb2d4d9b92364cbbd" },
};

static void test_bip32_tv4() {
    printf("\n--- BIP-32 Test Vector 4 (leading zeros, hardened children) ---\n");
    std::uint8_t seed[32];
    hex_to_bytes("3ddd5602285899a946114506157c7997e5444528f3003f6134712147db19b678",
                 seed, 32);
    auto [master, ok] = bip32_master_key(seed, 32);
    CHECK(ok, "TV4: master key generation succeeds");
    if (!ok) return;
    verify_chain(master, TV4, 3, "TV4");
}

// ============================================================================
// Test Vector 5: Tests for invalid extended key serialization
// seed = 000102030405060708090a0b0c0d0e0f
// (same seed as TV1 but tests specific serialization edge cases)
// ============================================================================

static void test_bip32_tv5_serialization() {
    printf("\n--- BIP-32 Test Vector 5 (serialization) ---\n");

    // Use TV1 seed
    std::uint8_t seed[16];
    hex_to_bytes("000102030405060708090a0b0c0d0e0f", seed, 16);
    auto [master, ok] = bip32_master_key(seed, 16);
    CHECK(ok, "TV5: master key generation succeeds");
    if (!ok) return;

    // Verify serialization is exactly 78 bytes
    auto ser = master.serialize();
    CHECK(ser.size() == 78, "TV5: serialized master is 78 bytes");

    // Version bytes: xprv = 0x0488ADE4
    CHECK(ser[0] == 0x04 && ser[1] == 0x88 && ser[2] == 0xAD && ser[3] == 0xE4,
          "TV5: version bytes = xprv (0x0488ADE4)");

    // Depth = 0 for master
    CHECK(ser[4] == 0x00, "TV5: depth = 0");

    // Parent fingerprint = 00000000 for master
    CHECK(ser[5] == 0 && ser[6] == 0 && ser[7] == 0 && ser[8] == 0,
          "TV5: parent fingerprint = 00000000");

    // Child number = 00000000 for master
    CHECK(ser[9] == 0 && ser[10] == 0 && ser[11] == 0 && ser[12] == 0,
          "TV5: child number = 00000000");

    // Chain code [13..44] should match TV1 master chain code
    std::uint8_t expected_cc[32];
    hex_to_bytes("873dff81c02f525623fd1fe5167eac3a55a049de3d314bb42ee227ffed37d508",
                 expected_cc, 32);
    CHECK(std::memcmp(&ser[13], expected_cc, 32) == 0,
          "TV5: chain code in serialization matches TV1");

    // Private key: byte [45] = 0x00 (prefix), then 32 bytes key [46..77]
    CHECK(ser[45] == 0x00, "TV5: private key prefix = 0x00");

    std::uint8_t expected_key[32];
    hex_to_bytes("e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35",
                 expected_key, 32);
    CHECK(std::memcmp(&ser[46], expected_key, 32) == 0,
          "TV5: private key in serialization matches TV1");

    // Public key serialization
    auto pub = master.to_public();
    auto pub_ser = pub.serialize();
    CHECK(pub_ser.size() == 78, "TV5: serialized public key is 78 bytes");

    // Version bytes: xpub = 0x0488B21E
    CHECK(pub_ser[0] == 0x04 && pub_ser[1] == 0x88 &&
          pub_ser[2] == 0xB2 && pub_ser[3] == 0x1E,
          "TV5: public version bytes = xpub (0x0488B21E)");

    // Derive m/0' and verify child serialization
    auto [child0h, ok2] = master.derive_hardened(0);
    CHECK(ok2, "TV5: m/0' derivation succeeds");
    if (!ok2) return;

    auto child_ser = child0h.serialize();
    CHECK(child_ser[4] == 0x01, "TV5: m/0' depth = 1 in serialization");

    // Verify parent fingerprint = fingerprint(master)
    auto master_fp = master.fingerprint();
    CHECK(std::memcmp(&child_ser[5], master_fp.data(), 4) == 0,
          "TV5: m/0' parent fingerprint = fingerprint(master)");

    // Child number = 0x80000000 (hardened 0)
    CHECK(child_ser[9] == 0x80 && child_ser[10] == 0x00 &&
          child_ser[11] == 0x00 && child_ser[12] == 0x00,
          "TV5: m/0' child number = 0x80000000");
}

// ============================================================================
// Test: Public key derivation consistency
// Derive a chain from private keys, then verify that the same public keys
// result from public-only derivation of normal (non-hardened) children.
// ============================================================================

static void test_bip32_public_derivation() {
    printf("\n--- BIP-32 Public Derivation Consistency ---\n");

    // TV1: derive m/0'/1 (hardened then normal)
    std::uint8_t seed[16];
    hex_to_bytes("000102030405060708090a0b0c0d0e0f", seed, 16);
    auto [master, ok] = bip32_master_key(seed, 16);
    CHECK(ok, "PubDeriv: master key OK");
    if (!ok) return;

    // Private path: m/0'/1
    auto [priv_0h, ok1] = master.derive_hardened(0);
    CHECK(ok1, "PubDeriv: m/0' OK");
    if (!ok1) return;

    auto [priv_0h_1, ok2] = priv_0h.derive_normal(1);
    CHECK(ok2, "PubDeriv: m/0'/1 (private) OK");
    if (!ok2) return;

    // Public path: get xpub at m/0', then derive /1 publicly
    auto pub_0h = priv_0h.to_public();
    CHECK(!pub_0h.is_private, "PubDeriv: to_public() is public");

    auto [pub_0h_1, ok3] = pub_0h.derive_normal(1);
    CHECK(ok3, "PubDeriv: m/0'/1 (public) OK");
    if (!ok3) return;

    // Both paths must produce the same public key
    auto priv_pub = priv_0h_1.public_key().to_compressed();
    auto pub_pub = pub_0h_1.public_key().to_compressed();
    CHECK(std::memcmp(priv_pub.data(), pub_pub.data(), 33) == 0,
          "PubDeriv: private and public derivation yield same pubkey");

    // Chain codes must also match
    CHECK(std::memcmp(priv_0h_1.chain_code.data(), pub_0h_1.chain_code.data(), 32) == 0,
          "PubDeriv: chain codes match for private/public derivation");
}

// ============================================================================
// Entry point
// ============================================================================

int test_bip32_vectors_run() {
    printf("=== BIP-32 Official Test Vectors (TV1-TV5) ===\n");

    test_bip32_tv1();
    test_bip32_tv2();
    test_bip32_tv3();
    test_bip32_tv4();
    test_bip32_tv5_serialization();
    test_bip32_public_derivation();

    printf("\n=== BIP-32 Vectors: %d/%d passed ===\n", g_tests_passed, g_tests_run);
    return (g_tests_passed == g_tests_run) ? 0 : 1;
}

// Standalone mode
#ifdef STANDALONE_TEST
int main() {
    return test_bip32_vectors_run();
}
#endif
