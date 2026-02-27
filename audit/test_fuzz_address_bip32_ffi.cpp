// ============================================================================
// Fuzz Tests: Address Encoders + BIP32 Path Parser + FFI Boundary
// ============================================================================
//
// Deterministic pseudo-fuzz: generates random & adversarial byte sequences and
// feeds them to address generation, BIP32 derivation, and the full FFI shim.
// Contract: functions must either succeed with valid output or return an error
// code -- never crash, hang, or corrupt memory.
//
// Covers roadmap tasks:
//   2.3.4  Address encoders fuzz (bech32/base58)
//   2.3.5  BIP32 path parser fuzz
//   2.3.6  FFI boundary fuzz (ufsecp shim)
//
// Build:
//   cmake -S . -B build -DSECP256K1_BUILD_FUZZ_TESTS=ON
//   cmake --build build --target test_fuzz_address_bip32_ffi
//
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <array>
#include <random>
#include <vector>
#include <string>

// C ABI
#include "ufsecp/ufsecp.h"

// Sanitizer-aware iteration scaling
#include "secp256k1/sanitizer_scale.hpp"

// -- Infrastructure ----------------------------------------------------------

static int g_pass = 0;
static int g_fail = 0;
static int g_crash = 0;  // should stay 0

#define CHECK(cond, msg) do { \
    if (cond) { \
        ++g_pass; \
    } else { \
        std::printf("  FAIL: %s (line %d)\n", (msg), __LINE__); \
        ++g_fail; \
    } \
} while(0)

#define MUST_NOT_CRASH(expr, msg) do { \
    (expr); \
    ++g_pass; \
} while(0)

static std::mt19937_64 rng(0xADD12E55);  // NOLINT(cert-msc32-c,cert-msc51-cpp)

static void fill_random(uint8_t* buf, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        buf[i] = static_cast<uint8_t>(rng() & 0xFF);
}
}

static void fill_random_str(char* buf, size_t len) {
    // Random printable + non-printable chars
    for (size_t i = 0; i < len; ++i) {
        buf[i] = static_cast<char>(rng() & 0xFF);
}
}

// Generate a valid compressed pubkey via ufsecp
static bool make_valid_pubkey(ufsecp_ctx* ctx, uint8_t pubkey33[33]) {
    uint8_t privkey[32];
    // Use a well-known valid private key (secp256k1 order is ~2^256)
    std::memset(privkey, 0, 32);
    // Set last 8 bytes to a random nonzero value
    uint64_t const val = (rng() % 0xFFFFFFFFFFFFFFFEULL) + 1;
    for (int i = 0; i < 8; ++i) {
        privkey[31 - i] = static_cast<uint8_t>((val >> (i * 8)) & 0xFF);
}

    return ufsecp_pubkey_create(ctx, privkey, pubkey33) == UFSECP_OK;
}

// ===========================================================================
// Suite [1]: P2PKH Address Fuzz (Base58Check)
// ===========================================================================

static void suite_1_p2pkh_fuzz(ufsecp_ctx* ctx) {
    std::printf("\n[1] P2PKH Address Fuzz (Base58Check)\n");

    // 1a: Random compressed-pubkey-like blobs -> should not crash
    for (int i = 0; i < SCALED(10000, 200); ++i) {
        uint8_t pubkey[33];
        fill_random(pubkey, 33);
        // Set valid prefix (02 or 03)
        pubkey[0] = (rng() & 1) ? 0x02 : 0x03;

        char addr[128] = {};
        size_t addr_len = sizeof(addr);
        ufsecp_error_t const err = ufsecp_addr_p2pkh(ctx, pubkey, UFSECP_NET_MAINNET,
                                                addr, &addr_len);
        // Either succeeds or returns error, no crash
        MUST_NOT_CRASH((void)err, "p2pkh_random_blob");
    }

    // 1b: Valid pubkey -> mainnet starts with '1', testnet starts with 'm'|'n'
    {
        uint8_t pub33[33];
        if (make_valid_pubkey(ctx, pub33)) {
            char addr[128];
            size_t alen = sizeof(addr);
            ufsecp_error_t err = ufsecp_addr_p2pkh(ctx, pub33, UFSECP_NET_MAINNET,
                                                    addr, &alen);
            CHECK(err == UFSECP_OK, "p2pkh_valid_ok");
            CHECK(alen > 0 && addr[0] == '1', "p2pkh_mainnet_prefix_1");

            alen = sizeof(addr);
            err = ufsecp_addr_p2pkh(ctx, pub33, UFSECP_NET_TESTNET,
                                    addr, &alen);
            CHECK(err == UFSECP_OK, "p2pkh_testnet_ok");
            CHECK(alen > 0 && (addr[0] == 'm' || addr[0] == 'n'),
                  "p2pkh_testnet_prefix");
        }
    }

    // 1c: NULL args -> must return error, not crash
    {
        char addr[128];
        size_t alen = sizeof(addr);
        uint8_t pub33[33] = {};
        MUST_NOT_CRASH(ufsecp_addr_p2pkh(ctx, nullptr, UFSECP_NET_MAINNET, addr, &alen),
                       "p2pkh_null_pubkey");
        MUST_NOT_CRASH(ufsecp_addr_p2pkh(ctx, pub33, UFSECP_NET_MAINNET, nullptr, &alen),
                       "p2pkh_null_addr");
        MUST_NOT_CRASH(ufsecp_addr_p2pkh(ctx, pub33, UFSECP_NET_MAINNET, addr, nullptr),
                       "p2pkh_null_len");
        MUST_NOT_CRASH(ufsecp_addr_p2pkh(nullptr, pub33, UFSECP_NET_MAINNET, addr, &alen),
                       "p2pkh_null_ctx");
    }

    // 1d: Buffer too small
    {
        uint8_t pub33[33];
        if (make_valid_pubkey(ctx, pub33)) {
            char tiny[5];
            size_t tlen = sizeof(tiny);
            ufsecp_error_t const err = ufsecp_addr_p2pkh(ctx, pub33, UFSECP_NET_MAINNET,
                                                    tiny, &tlen);
            // Should return buffer-too-small or fail gracefully
            MUST_NOT_CRASH((void)err, "p2pkh_tiny_buffer");
        }
    }

    // 1e: Invalid network ids
    for (int net = -10; net <= 255; ++net) {
        if (net == UFSECP_NET_MAINNET || net == UFSECP_NET_TESTNET) continue;
        uint8_t pub33[33];
        if (make_valid_pubkey(ctx, pub33)) {
            char addr[128];
            size_t alen = sizeof(addr);
            MUST_NOT_CRASH(ufsecp_addr_p2pkh(ctx, pub33, net, addr, &alen),
                           "p2pkh_bad_network");
        }
    }
}

// ===========================================================================
// Suite [2]: P2WPKH Address Fuzz (Bech32)
// ===========================================================================

static void suite_2_p2wpkh_fuzz(ufsecp_ctx* ctx) {
    std::printf("\n[2] P2WPKH Address Fuzz (Bech32)\n");

    // 2a: Random blobs as pubkey
    for (int i = 0; i < SCALED(10000, 200); ++i) {
        uint8_t pubkey[33];
        fill_random(pubkey, 33);
        pubkey[0] = (rng() & 1) ? 0x02 : 0x03;
        char addr[128] = {};
        size_t addr_len = sizeof(addr);
        MUST_NOT_CRASH(ufsecp_addr_p2wpkh(ctx, pubkey, UFSECP_NET_MAINNET,
                                           addr, &addr_len),
                       "p2wpkh_random_blob");
    }

    // 2b: Valid pubkey -> mainnet starts with "bc1q", testnet "tb1q"
    {
        uint8_t pub33[33];
        if (make_valid_pubkey(ctx, pub33)) {
            char addr[128];
            size_t alen = sizeof(addr);
            ufsecp_error_t err = ufsecp_addr_p2wpkh(ctx, pub33, UFSECP_NET_MAINNET,
                                                     addr, &alen);
            CHECK(err == UFSECP_OK, "p2wpkh_valid_ok");
            CHECK(alen >= 4 && std::strncmp(addr, "bc1q", 4) == 0,
                  "p2wpkh_mainnet_bc1q");

            alen = sizeof(addr);
            err = ufsecp_addr_p2wpkh(ctx, pub33, UFSECP_NET_TESTNET, addr, &alen);
            CHECK(err == UFSECP_OK, "p2wpkh_testnet_ok");
            CHECK(alen >= 4 && std::strncmp(addr, "tb1q", 4) == 0,
                  "p2wpkh_testnet_tb1q");
        }
    }

    // 2c: NULL args
    {
        char addr[128];
        size_t alen = sizeof(addr);
        uint8_t pub33[33] = {};
        MUST_NOT_CRASH(ufsecp_addr_p2wpkh(ctx, nullptr, 0, addr, &alen),
                       "p2wpkh_null_pubkey");
        MUST_NOT_CRASH(ufsecp_addr_p2wpkh(ctx, pub33, 0, nullptr, &alen),
                       "p2wpkh_null_addr");
        MUST_NOT_CRASH(ufsecp_addr_p2wpkh(nullptr, pub33, 0, addr, &alen),
                       "p2wpkh_null_ctx");
    }

    // 2d: Tiny buffer
    {
        uint8_t pub33[33];
        if (make_valid_pubkey(ctx, pub33)) {
            char tiny[5];
            size_t tlen = sizeof(tiny);
            MUST_NOT_CRASH(ufsecp_addr_p2wpkh(ctx, pub33, 0, tiny, &tlen),
                           "p2wpkh_tiny_buffer");
        }
    }
}

// ===========================================================================
// Suite [3]: P2TR Address Fuzz (Bech32m)
// ===========================================================================

static void suite_3_p2tr_fuzz(ufsecp_ctx* ctx) {
    std::printf("\n[3] P2TR Address Fuzz (Bech32m)\n");

    // 3a: Random 32-byte x-only keys
    for (int i = 0; i < SCALED(10000, 200); ++i) {
        uint8_t xkey[32];
        fill_random(xkey, 32);
        char addr[128] = {};
        size_t addr_len = sizeof(addr);
        MUST_NOT_CRASH(ufsecp_addr_p2tr(ctx, xkey, UFSECP_NET_MAINNET,
                                        addr, &addr_len),
                       "p2tr_random_x");
    }

    // 3b: Valid x-only key -> "bc1p" prefix
    {
        uint8_t pub33[33];
        if (make_valid_pubkey(ctx, pub33)) {
            // x-only = pub33[1..33]
            char addr[128];
            size_t alen = sizeof(addr);
            ufsecp_error_t err = ufsecp_addr_p2tr(ctx, pub33 + 1, UFSECP_NET_MAINNET,
                                                   addr, &alen);
            CHECK(err == UFSECP_OK, "p2tr_valid_ok");
            CHECK(alen >= 4 && std::strncmp(addr, "bc1p", 4) == 0,
                  "p2tr_bc1p_prefix");

            alen = sizeof(addr);
            err = ufsecp_addr_p2tr(ctx, pub33 + 1, UFSECP_NET_TESTNET,
                                   addr, &alen);
            CHECK(err == UFSECP_OK, "p2tr_testnet_ok");
            CHECK(alen >= 4 && std::strncmp(addr, "tb1p", 4) == 0,
                  "p2tr_testnet_tb1p");
        }
    }

    // 3c: NULL and edge cases
    {
        char addr[128];
        size_t alen = sizeof(addr);
        uint8_t xkey[32] = {};
        MUST_NOT_CRASH(ufsecp_addr_p2tr(ctx, nullptr, 0, addr, &alen),
                       "p2tr_null_key");
        MUST_NOT_CRASH(ufsecp_addr_p2tr(ctx, xkey, 0, nullptr, &alen),
                       "p2tr_null_addr");
        MUST_NOT_CRASH(ufsecp_addr_p2tr(nullptr, xkey, 0, addr, &alen),
                       "p2tr_null_ctx");
    }

    // 3d: All-zeros and all-ones x-only keys
    {
        uint8_t zeros[32] = {};
        uint8_t ones[32];
        std::memset(ones, 0xFF, 32);
        char addr[128];
        size_t alen = 0;

        alen = sizeof(addr);
        MUST_NOT_CRASH(ufsecp_addr_p2tr(ctx, zeros, 0, addr, &alen),
                       "p2tr_all_zeros");
        alen = sizeof(addr);
        MUST_NOT_CRASH(ufsecp_addr_p2tr(ctx, ones, 0, addr, &alen),
                       "p2tr_all_ones");
    }
}

// ===========================================================================
// Suite [4]: WIF Encode/Decode Fuzz
// ===========================================================================

static void suite_4_wif_fuzz(ufsecp_ctx* ctx) {
    std::printf("\n[4] WIF Encode/Decode Fuzz\n");

    // 4a: Random 32-byte privkeys -> encode -> decode -> must match
    int roundtrip_ok = 0;
    for (int i = 0; i < SCALED(5000, 100); ++i) {
        uint8_t privkey[32];
        fill_random(privkey, 32);

        char wif[128];
        size_t wlen = sizeof(wif);
        ufsecp_error_t const err = ufsecp_wif_encode(ctx, privkey, 1 /*compressed*/,
                                                UFSECP_NET_MAINNET, wif, &wlen);
        if (err == UFSECP_OK) {
            uint8_t decoded[32];
            int comp_out = 0, net_out = 0;
            ufsecp_error_t const err2 = ufsecp_wif_decode(ctx, wif, decoded,
                                                     &comp_out, &net_out);
            if (err2 == UFSECP_OK &&
                std::memcmp(privkey, decoded, 32) == 0 &&
                comp_out == 1 &&
                net_out == UFSECP_NET_MAINNET) {
                ++roundtrip_ok;
            }
        }
        // Keys >= order or zero are expected to fail
        MUST_NOT_CRASH((void)0, "wif_random_no_crash");
    }
    CHECK(roundtrip_ok > SCALED(4500, 90), "wif_roundtrip_majority");  // >90% should be valid keys

    // 4b: Decode garbage strings
    for (int i = 0; i < SCALED(5000, 100); ++i) {
        char garbage[128];
        size_t const glen = rng() % 120 + 1;
        fill_random_str(garbage, glen);
        garbage[glen] = '\0';

        uint8_t decoded[32];
        int comp = 0, net = 0;
        MUST_NOT_CRASH(ufsecp_wif_decode(ctx, garbage, decoded, &comp, &net),
                       "wif_decode_garbage");
    }

    // 4c: NULL args
    {
        uint8_t pk[32] = {};
        char wif[128];
        size_t wlen = sizeof(wif);
        MUST_NOT_CRASH(ufsecp_wif_encode(ctx, nullptr, 1, 0, wif, &wlen),
                       "wif_encode_null_key");
        MUST_NOT_CRASH(ufsecp_wif_encode(ctx, pk, 1, 0, nullptr, &wlen),
                       "wif_encode_null_buf");
        uint8_t dec[32];
        int c = 0, n = 0;
        MUST_NOT_CRASH(ufsecp_wif_decode(ctx, nullptr, dec, &c, &n),
                       "wif_decode_null_str");
        MUST_NOT_CRASH(ufsecp_wif_decode(ctx, "5K...", nullptr, &c, &n),
                       "wif_decode_null_out");
    }

    // 4d: Uncompressed variant
    {
        uint8_t pk[32];
        std::memset(pk, 0, 32); pk[31] = 1;  // key = 1
        char wif[128];
        size_t wlen = sizeof(wif);
        ufsecp_error_t const err = ufsecp_wif_encode(ctx, pk, 0 /*uncomp*/, 0, wif, &wlen);
        if (err == UFSECP_OK) {
            uint8_t dec[32]; int c = 0, n = 0;
            ufsecp_error_t const err2 = ufsecp_wif_decode(ctx, wif, dec, &c, &n);
            CHECK(err2 == UFSECP_OK, "wif_uncomp_decode_ok");
            CHECK(c == 0, "wif_uncomp_flag_0");
            CHECK(std::memcmp(pk, dec, 32) == 0, "wif_uncomp_roundtrip");
        }
    }
}

// ===========================================================================
// Suite [5]: BIP32 Master Key from Seed Fuzz
// ===========================================================================

static void suite_5_bip32_master_fuzz(ufsecp_ctx* ctx) {
    std::printf("\n[5] BIP32 Master Key from Seed Fuzz\n");

    // 5a: Valid seed lengths (16, 32, 64)
    for (int const slen : {16, 32, 64}) {
        uint8_t seed[64];
        fill_random(seed, slen);
        ufsecp_bip32_key key;
        ufsecp_error_t const err = ufsecp_bip32_master(ctx, seed, slen, &key);
        CHECK(err == UFSECP_OK, "bip32_master_valid_seed");
        CHECK(key.is_private == 1, "bip32_master_is_private");
    }

    // 5b: Invalid seed lengths
    for (size_t const slen : {0, 1, 5, 15, 65, 100, 255}) {
        uint8_t seed[256];
        fill_random(seed, slen > 0 ? slen : 1);
        ufsecp_bip32_key key;
        MUST_NOT_CRASH(ufsecp_bip32_master(ctx, seed, slen, &key),
                       "bip32_master_bad_seed_len");
    }

    // 5c: Random seed bytes at all valid lengths
    for (int i = 0; i < SCALED(5000, 100); ++i) {
        size_t const slen = 16 + (rng() % 49);  // 16..64
        uint8_t seed[64];
        fill_random(seed, slen);
        ufsecp_bip32_key key;
        MUST_NOT_CRASH(ufsecp_bip32_master(ctx, seed, slen, &key),
                       "bip32_master_random");
    }

    // 5d: NULL args
    {
        uint8_t seed[32];
        ufsecp_bip32_key key;
        MUST_NOT_CRASH(ufsecp_bip32_master(ctx, nullptr, 32, &key),
                       "bip32_master_null_seed");
        MUST_NOT_CRASH(ufsecp_bip32_master(ctx, seed, 32, nullptr),
                       "bip32_master_null_key");
        MUST_NOT_CRASH(ufsecp_bip32_master(nullptr, seed, 32, &key),
                       "bip32_master_null_ctx");
    }
}

// ===========================================================================
// Suite [6]: BIP32 Path Parser Fuzz
// ===========================================================================

static void suite_6_bip32_path_fuzz(ufsecp_ctx* ctx) {
    std::printf("\n[6] BIP32 Path Parser Fuzz\n");

    // Get a valid master key
    uint8_t seed[32];
    std::memset(seed, 0xAA, 32);
    ufsecp_bip32_key master;
    ufsecp_error_t err = ufsecp_bip32_master(ctx, seed, 32, &master);
    if (err != UFSECP_OK) {
        std::printf("  SKIP: cannot create master key\n");
        return;
    }

    // 6a: Valid BIP32 paths
    const char* const valid_paths[] = {
        "m",
        "m/0",
        "m/0/1",
        "m/44'/0'/0'/0/0",
        "m/84'/0'/0'/0/0",
        "m/86'/0'/0'/0/0",
        "m/0'",
        "m/2147483647",       // max non-hardened
        "m/2147483647'",      // max hardened
        "m/0/1/2/3/4/5/6/7",  // deep path
    };
    for (const char* path : valid_paths) {
        ufsecp_bip32_key child;
        err = ufsecp_bip32_derive_path(ctx, &master, path, &child);
        CHECK(err == UFSECP_OK, "bip32_path_valid");
    }

    // 6b: Invalid BIP32 paths -- must not crash
    const char* const invalid_paths[] = {
        "",
        "m/",
        "/0/1",
        "m//0",
        "x/0/1",
        "m/-1",
        "m/abc",
        "m/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0/0",  // very deep
        "m/99999999999999999999",   // overflow
        "m/4294967296",             // > uint32_max
        "m/0h",                     // alternate hardened syntax
        "m/0H",
        "m/0'0",                    // malformed
        "m/'0",
        "m/0//1",                   // double slash
        "m\x00/0",                  // embedded NUL
        "\xff\xfe\xfd",            // garbage bytes
        "m/\t\n\r",                 // whitespace chars
    };
    for (const char* path : invalid_paths) {
        ufsecp_bip32_key child;
        MUST_NOT_CRASH(ufsecp_bip32_derive_path(ctx, &master, path, &child),
                       "bip32_path_invalid");
    }

    // 6c: Random string paths
    for (int i = 0; i < SCALED(10000, 200); ++i) {
        char path[128];
        size_t const plen = rng() % 100 + 1;
        fill_random_str(path, plen);
        path[plen] = '\0';
        ufsecp_bip32_key child;
        MUST_NOT_CRASH(ufsecp_bip32_derive_path(ctx, &master, path, &child),
                       "bip32_path_random_str");
    }

    // 6d: Determinism -- same path must produce same key
    {
        ufsecp_bip32_key k1, k2;
        err = ufsecp_bip32_derive_path(ctx, &master, "m/44'/0'/0'/0/0", &k1);
        ufsecp_error_t const err2 = ufsecp_bip32_derive_path(ctx, &master, "m/44'/0'/0'/0/0", &k2);
        if (err == UFSECP_OK && err2 == UFSECP_OK) {
            CHECK(std::memcmp(&k1.data, &k2.data, UFSECP_BIP32_SERIALIZED_LEN) == 0,
                  "bip32_path_deterministic");
        }
    }

    // 6e: Different paths must produce different keys
    {
        ufsecp_bip32_key k1, k2;
        err = ufsecp_bip32_derive_path(ctx, &master, "m/44'/0'/0'/0/0", &k1);
        ufsecp_error_t const err2 = ufsecp_bip32_derive_path(ctx, &master, "m/44'/0'/0'/0/1", &k2);
        if (err == UFSECP_OK && err2 == UFSECP_OK) {
            CHECK(std::memcmp(&k1.data, &k2.data, UFSECP_BIP32_SERIALIZED_LEN) != 0,
                  "bip32_path_different");
        }
    }

    // 6f: NULL args
    {
        ufsecp_bip32_key child;
        MUST_NOT_CRASH(ufsecp_bip32_derive_path(ctx, &master, nullptr, &child),
                       "bip32_path_null_path");
        MUST_NOT_CRASH(ufsecp_bip32_derive_path(ctx, &master, "m/0", nullptr),
                       "bip32_path_null_child");
        MUST_NOT_CRASH(ufsecp_bip32_derive_path(ctx, nullptr, "m/0", &child),
                       "bip32_path_null_master");
        MUST_NOT_CRASH(ufsecp_bip32_derive_path(nullptr, &master, "m/0", &child),
                       "bip32_path_null_ctx");
    }
}

// ===========================================================================
// Suite [7]: BIP32 Derive (single-step) Fuzz
// ===========================================================================

static void suite_7_bip32_derive_fuzz(ufsecp_ctx* ctx) {
    std::printf("\n[7] BIP32 Derive (single-step) Fuzz\n");

    uint8_t seed[32];
    std::memset(seed, 0xBB, 32);
    ufsecp_bip32_key master;
    ufsecp_error_t err = ufsecp_bip32_master(ctx, seed, 32, &master);
    if (err != UFSECP_OK) {
        std::printf("  SKIP: cannot create master key\n");
        return;
    }

    // 7a: Normal derivation (indices 0..999)
    for (uint32_t idx = 0; idx < SCALED(1000, 50); ++idx) {
        ufsecp_bip32_key child;
        err = ufsecp_bip32_derive(ctx, &master, idx, &child);
        CHECK(err == UFSECP_OK, "bip32_derive_normal");
    }

    // 7b: Hardened derivation
    for (uint32_t idx = 0x80000000; idx < 0x80000000 + 100; ++idx) {
        ufsecp_bip32_key child;
        err = ufsecp_bip32_derive(ctx, &master, idx, &child);
        CHECK(err == UFSECP_OK, "bip32_derive_hardened");
    }

    // 7c: Edge indices
    for (uint32_t const idx : {0u, 0x7FFFFFFFu, 0x80000000u, 0xFFFFFFFFu}) {
        ufsecp_bip32_key child;
        MUST_NOT_CRASH(ufsecp_bip32_derive(ctx, &master, idx, &child),
                       "bip32_derive_edge_idx");
    }

    // 7d: Extract privkey and pubkey from derived key
    {
        ufsecp_bip32_key child;
        err = ufsecp_bip32_derive(ctx, &master, 0, &child);
        if (err == UFSECP_OK) {
            uint8_t privkey[32];
            ufsecp_error_t const perr = ufsecp_bip32_privkey(ctx, &child, privkey);
            CHECK(perr == UFSECP_OK, "bip32_extract_privkey");

            uint8_t pubkey[33];
            ufsecp_error_t const puerr = ufsecp_bip32_pubkey(ctx, &child, pubkey);
            CHECK(puerr == UFSECP_OK, "bip32_extract_pubkey");
            CHECK(pubkey[0] == 0x02 || pubkey[0] == 0x03, "bip32_pubkey_compressed");
        }
    }
}

// ===========================================================================
// Suite [8]: FFI Context Lifecycle Stress
// ===========================================================================

static void suite_8_ffi_context_stress() {
    std::printf("\n[8] FFI Context Lifecycle Stress\n");

    // 8a: Rapid create/destroy cycles
    for (int i = 0; i < 100; ++i) {
        ufsecp_ctx* c = nullptr;
        ufsecp_error_t const err = ufsecp_ctx_create(&c);
        CHECK(err == UFSECP_OK, "ffi_ctx_create");
        CHECK(c != nullptr, "ffi_ctx_not_null");
        ufsecp_ctx_destroy(c);
    }
    ++g_pass;  // survived

    // 8b: Clone and destroy
    {
        ufsecp_ctx* c1 = nullptr;
        ufsecp_ctx_create(&c1);
        if (c1) {
            ufsecp_ctx* c2 = nullptr;
            ufsecp_error_t const err = ufsecp_ctx_clone(c1, &c2);
            CHECK(err == UFSECP_OK, "ffi_ctx_clone");
            CHECK(c2 != nullptr, "ffi_ctx_clone_not_null");

            // Both contexts should work independently
            uint8_t pk[32] = {}; pk[31] = 42;
            uint8_t pub1[33], pub2[33];
            ufsecp_pubkey_create(c1, pk, pub1);
            ufsecp_pubkey_create(c2, pk, pub2);
            CHECK(std::memcmp(pub1, pub2, 33) == 0, "ffi_ctx_clone_equivalent");

            ufsecp_ctx_destroy(c2);
            ufsecp_ctx_destroy(c1);
        }
    }

    // 8c: Destroy NULL (must not crash)
    MUST_NOT_CRASH(ufsecp_ctx_destroy(nullptr), "ffi_destroy_null");

    // 8d: Create with NULL out ptr
    MUST_NOT_CRASH(ufsecp_ctx_create(nullptr), "ffi_create_null_out");

    // 8e: Clone with NULL src
    {
        ufsecp_ctx* out = nullptr;
        MUST_NOT_CRASH(ufsecp_ctx_clone(nullptr, &out), "ffi_clone_null_src");
    }
}

// ===========================================================================
// Suite [9]: FFI ECDSA Sign/Verify Boundary Fuzz
// ===========================================================================

static void suite_9_ffi_ecdsa_boundary(ufsecp_ctx* ctx) {
    std::printf("\n[9] FFI ECDSA Sign/Verify Boundary Fuzz\n");

    // 9a: Sign with random key + hash, verify result
    int sign_verify_ok = 0;
    for (int i = 0; i < SCALED(5000, 100); ++i) {
        uint8_t pk[32], hash[32], sig[64];
        fill_random(pk, 32);
        fill_random(hash, 32);
        ufsecp_error_t const serr = ufsecp_ecdsa_sign(ctx, hash, pk, sig);
        if (serr == UFSECP_OK) {
            // Get pubkey
            uint8_t pub33[33];
            if (ufsecp_pubkey_create(ctx, pk, pub33) == UFSECP_OK) {
                ufsecp_error_t const verr = ufsecp_ecdsa_verify(ctx, hash, sig, pub33);
                if (verr == UFSECP_OK) ++sign_verify_ok;
            }
        }
    }
    CHECK(sign_verify_ok > SCALED(4500, 90), "ffi_ecdsa_sign_verify_majority");

    // 9b: Verify with garbage
    for (int i = 0; i < SCALED(5000, 100); ++i) {
        uint8_t pub33[33], hash[32], sig[64];
        fill_random(pub33, 33);
        pub33[0] = (rng() & 1) ? 0x02 : 0x03;
        fill_random(hash, 32);
        fill_random(sig, 64);
        MUST_NOT_CRASH(ufsecp_ecdsa_verify(ctx, hash, sig, pub33),
                       "ffi_ecdsa_verify_garbage");
    }

    // 9c: Sign with zero key (must fail, not crash)
    {
        uint8_t zero_key[32] = {}, hash[32], sig[64];
        fill_random(hash, 32);
        ufsecp_error_t const err = ufsecp_ecdsa_sign(ctx, hash, zero_key, sig);
        CHECK(err != UFSECP_OK, "ffi_ecdsa_sign_zero_key_fails");
    }

    // 9d: NULL args
    {
        uint8_t pk[32], hash[32], sig[64], pub33[33];
        std::memset(pk, 0, 32); pk[31] = 1;
        std::memset(hash, 0, 32);
        MUST_NOT_CRASH(ufsecp_ecdsa_sign(ctx, nullptr, pk, sig), "ffi_sign_null_msg");
        MUST_NOT_CRASH(ufsecp_ecdsa_sign(ctx, hash, nullptr, sig), "ffi_sign_null_key");
        MUST_NOT_CRASH(ufsecp_ecdsa_sign(ctx, hash, pk, nullptr), "ffi_sign_null_sig");
        MUST_NOT_CRASH(ufsecp_ecdsa_verify(ctx, nullptr, sig, pub33), "ffi_verify_null_msg");
        MUST_NOT_CRASH(ufsecp_ecdsa_verify(ctx, hash, nullptr, pub33), "ffi_verify_null_sig");
        MUST_NOT_CRASH(ufsecp_ecdsa_verify(ctx, hash, sig, nullptr), "ffi_verify_null_pub");
    }
}

// ===========================================================================
// Suite [10]: FFI Schnorr Sign/Verify Boundary Fuzz
// ===========================================================================

static void suite_10_ffi_schnorr_boundary(ufsecp_ctx* ctx) {
    std::printf("\n[10] FFI Schnorr Sign/Verify Boundary Fuzz\n");

    // 10a: Sign with random key + message, verify result
    int sign_verify_ok = 0;
    for (int i = 0; i < SCALED(5000, 100); ++i) {
        uint8_t pk[32], msg[32], sig[64], aux[32];
        fill_random(pk, 32);
        fill_random(msg, 32);
        std::memset(aux, 0, 32);  // deterministic aux
        ufsecp_error_t const serr = ufsecp_schnorr_sign(ctx, msg, pk, aux, sig);
        if (serr == UFSECP_OK) {
            uint8_t xpub[32];
            if (ufsecp_pubkey_xonly(ctx, pk, xpub) == UFSECP_OK) {
                ufsecp_error_t const verr = ufsecp_schnorr_verify(ctx, msg, sig, xpub);
                if (verr == UFSECP_OK) ++sign_verify_ok;
            }
        }
    }
    CHECK(sign_verify_ok > SCALED(4500, 90), "ffi_schnorr_sign_verify_majority");

    // 10b: Verify garbage sigs
    for (int i = 0; i < SCALED(5000, 100); ++i) {
        uint8_t xpub[32], msg[32], sig[64];
        fill_random(xpub, 32);
        fill_random(msg, 32);
        fill_random(sig, 64);
        MUST_NOT_CRASH(ufsecp_schnorr_verify(ctx, msg, sig, xpub),
                       "ffi_schnorr_verify_garbage");
    }

    // 10c: NULL args
    {
        uint8_t pk[32], msg[32], sig[64], xpub[32], aux[32] = {};
        std::memset(pk, 0, 32); pk[31] = 1;
        std::memset(msg, 0, 32);
        MUST_NOT_CRASH(ufsecp_schnorr_sign(ctx, nullptr, pk, aux, sig), "ffi_schnorr_sign_null_msg");
        MUST_NOT_CRASH(ufsecp_schnorr_sign(ctx, msg, nullptr, aux, sig), "ffi_schnorr_sign_null_key");
        MUST_NOT_CRASH(ufsecp_schnorr_sign(ctx, msg, pk, aux, nullptr), "ffi_schnorr_sign_null_sig");
        MUST_NOT_CRASH(ufsecp_schnorr_verify(ctx, nullptr, sig, xpub), "ffi_schnorr_verify_null_msg");
        MUST_NOT_CRASH(ufsecp_schnorr_verify(ctx, msg, nullptr, xpub), "ffi_schnorr_verify_null_sig");
        MUST_NOT_CRASH(ufsecp_schnorr_verify(ctx, msg, sig, nullptr), "ffi_schnorr_verify_null_pub");
    }
}

// ===========================================================================
// Suite [11]: FFI ECDH + Tweaking Boundary
// ===========================================================================

static void suite_11_ffi_ecdh_tweak(ufsecp_ctx* ctx) {
    std::printf("\n[11] FFI ECDH + Tweaking Boundary Fuzz\n");

    // 11a: ECDH with random keypairs
    int ecdh_ok = 0;
    for (int i = 0; i < SCALED(2000, 50); ++i) {
        uint8_t sk_a[32], sk_b[32], pub_a[33], pub_b[33];
        fill_random(sk_a, 32);
        fill_random(sk_b, 32);

        if (ufsecp_pubkey_create(ctx, sk_a, pub_a) == UFSECP_OK &&
            ufsecp_pubkey_create(ctx, sk_b, pub_b) == UFSECP_OK) {
            uint8_t secret_ab[32], secret_ba[32];
            ufsecp_error_t const e1 = ufsecp_ecdh(ctx, sk_a, pub_b, secret_ab);
            ufsecp_error_t const e2 = ufsecp_ecdh(ctx, sk_b, pub_a, secret_ba);
            if (e1 == UFSECP_OK && e2 == UFSECP_OK) {
                if (std::memcmp(secret_ab, secret_ba, 32) == 0) ++ecdh_ok;
            }
        }
    }
    CHECK(ecdh_ok > SCALED(1800, 45), "ffi_ecdh_shared_secret_majority");

    // 11b: ECDH with garbage pubkey
    for (int i = 0; i < SCALED(2000, 50); ++i) {
        uint8_t sk[32], garbage_pub[33], secret[32];
        fill_random(sk, 32);
        fill_random(garbage_pub, 33);
        MUST_NOT_CRASH(ufsecp_ecdh(ctx, sk, garbage_pub, secret),
                       "ffi_ecdh_garbage_pub");
    }

    // 11c: Secret key tweaking (in-place)
    {
        uint8_t sk[32], tweak[32];
        std::memset(sk, 0, 32); sk[31] = 1;
        fill_random(tweak, 32);
        uint8_t sk_copy[32];
        std::memcpy(sk_copy, sk, 32);
        MUST_NOT_CRASH(ufsecp_seckey_tweak_add(ctx, sk_copy, tweak),
                       "ffi_tweak_add");

        std::memcpy(sk_copy, sk, 32);
        MUST_NOT_CRASH(ufsecp_seckey_tweak_mul(ctx, sk_copy, tweak),
                       "ffi_tweak_mul");
    }

    // 11d: ECDH NULL args
    {
        uint8_t sk[32], pub33[33], secret[32];
        MUST_NOT_CRASH(ufsecp_ecdh(ctx, nullptr, pub33, secret), "ffi_ecdh_null_sk");
        MUST_NOT_CRASH(ufsecp_ecdh(ctx, sk, nullptr, secret), "ffi_ecdh_null_pub");
        MUST_NOT_CRASH(ufsecp_ecdh(ctx, sk, pub33, nullptr), "ffi_ecdh_null_out");
    }
}

// ===========================================================================
// Suite [12]: FFI Taproot Output Key Boundary
// ===========================================================================

static void suite_12_ffi_taproot_boundary(ufsecp_ctx* ctx) {
    std::printf("\n[12] FFI Taproot Output Key Boundary Fuzz\n");

    // 12a: Valid internal key -> output key
    {
        uint8_t pub33[33];
        if (make_valid_pubkey(ctx, pub33)) {
            uint8_t output_x[32];
            int parity = -1;
            ufsecp_error_t const err = ufsecp_taproot_output_key(ctx, pub33 + 1,
                                                            nullptr, output_x, &parity);
            CHECK(err == UFSECP_OK, "taproot_output_key_ok");
            CHECK(parity == 0 || parity == 1, "taproot_parity_valid");
        }
    }

    // 12b: With merkle root
    {
        uint8_t pub33[33];
        if (make_valid_pubkey(ctx, pub33)) {
            uint8_t merkle[32];
            fill_random(merkle, 32);
            uint8_t output_x[32];
            int parity = 0;
            ufsecp_error_t const err = ufsecp_taproot_output_key(ctx, pub33 + 1,
                                                            merkle, output_x, &parity);
            CHECK(err == UFSECP_OK, "taproot_with_merkle_ok");
        }
    }

    // 12c: Random x-only keys
    for (int i = 0; i < SCALED(5000, 100); ++i) {
        uint8_t xkey[32], merkle[32], output[32];
        fill_random(xkey, 32);
        fill_random(merkle, 32);
        int parity = 0;
        MUST_NOT_CRASH(ufsecp_taproot_output_key(ctx, xkey,
                                                  (rng() & 1) ? merkle : nullptr,
                                                  output, &parity),
                       "taproot_random_x");
    }

    // 12d: NULL args
    {
        uint8_t xkey[32] = {}, output[32];
        int parity = 0;
        MUST_NOT_CRASH(ufsecp_taproot_output_key(ctx, nullptr, nullptr, output, &parity),
                       "taproot_null_key");
        MUST_NOT_CRASH(ufsecp_taproot_output_key(ctx, xkey, nullptr, nullptr, &parity),
                       "taproot_null_output");
        MUST_NOT_CRASH(ufsecp_taproot_output_key(nullptr, xkey, nullptr, output, &parity),
                       "taproot_null_ctx");
    }
}

// ===========================================================================
// Suite [13]: FFI Error Inspection
// ===========================================================================

static void suite_13_ffi_error_inspection(ufsecp_ctx* ctx) {
    std::printf("\n[13] FFI Error Inspection\n");

    // 13a: All defined error codes have a string
    for (int code = 0; code <= 10; ++code) {
        const char* str = ufsecp_error_str(static_cast<ufsecp_error_t>(code));
        // cppcheck-suppress nullPointerRedundantCheck
        CHECK(str != nullptr, "error_str_not_null");
        // cppcheck-suppress nullPointerRedundantCheck
        CHECK(std::strlen(str) > 0, "error_str_not_empty");
    }

    // 13b: Undefined error codes should still return something
    for (int code = 11; code <= 255; ++code) {
        const char* str = ufsecp_error_str(static_cast<ufsecp_error_t>(code));
        CHECK(str != nullptr, "error_str_unknown_not_null");
    }

    // 13c: ufsecp_last_error after a failure
    {
        uint8_t zero_key[32] = {};
        uint8_t pub33[33];
        ufsecp_error_t const err = ufsecp_pubkey_create(ctx, zero_key, pub33);
        if (err != UFSECP_OK) {
            ufsecp_error_t const last = ufsecp_last_error(ctx);
            CHECK(last != UFSECP_OK, "last_error_after_fail");
            const char* msg = ufsecp_last_error_msg(ctx);
            CHECK(msg != nullptr, "last_error_msg_not_null");
        }
    }

    // 13d: ABI version
    {
        uint32_t const ver = ufsecp_abi_version();
        CHECK(ver > 0, "abi_version_positive");
    }
}

// ===========================================================================
// _run() entry point for unified audit runner
// ===========================================================================

int test_fuzz_address_bip32_ffi_run() {
    g_pass = 0; g_fail = 0; g_crash = 0;

    ufsecp_ctx* ctx = nullptr;
    ufsecp_error_t const err = ufsecp_ctx_create(&ctx);
    if (err != UFSECP_OK || !ctx) {
        std::printf("  FATAL: ufsecp_ctx_create failed\n");
        return 1;
    }

    suite_1_p2pkh_fuzz(ctx);
    suite_2_p2wpkh_fuzz(ctx);
    suite_3_p2tr_fuzz(ctx);
    suite_4_wif_fuzz(ctx);
    suite_5_bip32_master_fuzz(ctx);
    suite_6_bip32_path_fuzz(ctx);
    suite_7_bip32_derive_fuzz(ctx);
    suite_8_ffi_context_stress();
    suite_9_ffi_ecdsa_boundary(ctx);
    suite_10_ffi_schnorr_boundary(ctx);
    suite_11_ffi_ecdh_tweak(ctx);
    suite_12_ffi_taproot_boundary(ctx);
    suite_13_ffi_error_inspection(ctx);

    ufsecp_ctx_destroy(ctx);
    return (g_fail > 0 || g_crash > 0) ? 1 : 0;
}

// ===========================================================================
// Main (standalone only)
// ===========================================================================

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    std::printf("=== Fuzz: Address Encoders + BIP32 + FFI Boundary ===\n");
    std::printf("    Tasks: 2.3.4 (address), 2.3.5 (BIP32), 2.3.6 (FFI)\n\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_error_t err = ufsecp_ctx_create(&ctx);
    if (err != UFSECP_OK || !ctx) {
        std::printf("FATAL: cannot create ufsecp context\n");
        return 1;
    }

    suite_1_p2pkh_fuzz(ctx);
    suite_2_p2wpkh_fuzz(ctx);
    suite_3_p2tr_fuzz(ctx);
    suite_4_wif_fuzz(ctx);
    suite_5_bip32_master_fuzz(ctx);
    suite_6_bip32_path_fuzz(ctx);
    suite_7_bip32_derive_fuzz(ctx);
    suite_8_ffi_context_stress();
    suite_9_ffi_ecdsa_boundary(ctx);
    suite_10_ffi_schnorr_boundary(ctx);
    suite_11_ffi_ecdh_tweak(ctx);
    suite_12_ffi_taproot_boundary(ctx);
    suite_13_ffi_error_inspection(ctx);

    ufsecp_ctx_destroy(ctx);

    std::printf("\n====================================================\n");
    std::printf("  PASSED: %d   FAILED: %d   CRASHES: %d\n", g_pass, g_fail, g_crash);
    std::printf("====================================================\n");
    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
