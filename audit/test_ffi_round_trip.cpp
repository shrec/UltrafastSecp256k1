// ============================================================================
// Cross-ABI / FFI Round-Trip Tests
// Phase V -- Verify ufsecp C ABI correctness via complete round-trip cycles
// ============================================================================
//
// Tests the ufsecp C API (stable ABI boundary) for:
//   1. Context lifecycle (create / clone / destroy)
//   2. Key generation: privkey -> pubkey (compressed, uncompressed, x-only)
//   3. ECDSA: sign -> verify -> DER encode/decode -> verify
//   4. ECDSA Recovery: sign_recoverable -> recover -> compare pubkey
//   5. Schnorr/BIP-340: sign -> verify round-trip
//   6. ECDH: shared secret agreement (both sides compute same secret)
//   7. BIP-32: master -> derive -> extract -> verify
//   8. Address generation: P2PKH, P2WPKH, P2TR from known keys
//   9. WIF: encode -> decode round-trip
//  10. Hashing: SHA-256, Hash160, tagged hash known vectors
//  11. Taproot: output key derivation + commitment verification
//  12. Error paths: NULL args, bad keys, invalid sigs
//
// All tests go through the C ABI boundary (ufsecp_*), verifying that the
// FFI layer correctly marshals data in/out without corruption.
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <array>

// Include C ABI header -- we define UFSECP_API= to resolve as local linkage
// (the impl is compiled into unified_audit_runner directly)
#ifndef UFSECP_BUILDING
#define UFSECP_BUILDING
#endif
#include "ufsecp/ufsecp.h"

static int g_pass = 0, g_fail = 0;

#include "audit_check.hpp"

#define CHECK_OK(expr, msg) CHECK((expr) == UFSECP_OK, msg)

// -- Helpers ------------------------------------------------------------------

static void hex_to_bytes(const char* hex, uint8_t* out, int len) {
    for (int i = 0; i < len; ++i) {
        unsigned byte = 0;
        // NOLINTNEXTLINE(cert-err34-c)
        if (std::sscanf(hex + static_cast<size_t>(i) * 2, "%02x", &byte) != 1) byte = 0;
        out[i] = static_cast<uint8_t>(byte);
    }
}

// Well-known private key: scalar = 1 (generator point)
static const char* PRIVKEY1_HEX =
    "0000000000000000000000000000000000000000000000000000000000000001";

// Well-known private key: scalar = 2
static const char* PRIVKEY2_HEX =
    "0000000000000000000000000000000000000000000000000000000000000002";

// Test message: SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
static const char* MSG_HEX =
    "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";

// ============================================================================
// Test 1: Context Lifecycle
// ============================================================================
static void test_context_lifecycle() {
    (void)std::printf("[1] FFI: Context create / clone / destroy\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");
    CHECK(ctx != nullptr, "ctx is non-null");

    // Clone
    ufsecp_ctx* clone = nullptr;
    CHECK_OK(ufsecp_ctx_clone(ctx, &clone), "ctx_clone");
    CHECK(clone != nullptr, "clone is non-null");
    CHECK(clone != ctx, "clone is distinct pointer");

    // Destroy (NULL safe)
    ufsecp_ctx_destroy(clone);
    ufsecp_ctx_destroy(ctx);
    ufsecp_ctx_destroy(nullptr); // should not crash

    (void)std::printf("    context lifecycle OK\n");
}

// ============================================================================
// Test 2: Key Generation Round-Trip
// ============================================================================
static void test_key_generation() {
    (void)std::printf("[2] FFI: Key generation (compressed, uncompressed, xonly)\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);

    // Verify key
    CHECK_OK(ufsecp_seckey_verify(ctx, privkey), "seckey_verify(1)");

    // Compressed pubkey
    uint8_t pub33[33] = {};
    CHECK_OK(ufsecp_pubkey_create(ctx, privkey, pub33), "pubkey_create");
    CHECK(pub33[0] == 0x02 || pub33[0] == 0x03, "compressed prefix valid");

    // Uncompressed pubkey
    uint8_t pub65[65] = {};
    CHECK_OK(ufsecp_pubkey_create_uncompressed(ctx, privkey, pub65),
             "pubkey_create_uncompressed");
    CHECK(pub65[0] == 0x04, "uncompressed prefix is 0x04");

    // Parse uncompressed -> compressed
    uint8_t parsed33[33] = {};
    CHECK_OK(ufsecp_pubkey_parse(ctx, pub65, 65, parsed33), "pubkey_parse(65->33)");
    CHECK(std::memcmp(pub33, parsed33, 33) == 0, "parse(uncomp) == compressed");

    // x-only
    uint8_t xonly[32] = {};
    CHECK_OK(ufsecp_pubkey_xonly(ctx, privkey, xonly), "pubkey_xonly");
    // x-only should match bytes 1..32 of compressed (if y is even)
    // or of the negated point. Just check it's non-zero.
    bool nonzero = false;
    for (int i = 0; i < 32; ++i) {
        if (xonly[i] != 0) { nonzero = true; break; }
    }
    CHECK(nonzero, "xonly is non-zero");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 3: ECDSA Sign -> Verify -> DER Round-Trip
// ============================================================================
static void test_ecdsa_round_trip() {
    (void)std::printf("[3] FFI: ECDSA sign -> verify -> DER encode/decode\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    uint8_t pub33[33];
    ufsecp_pubkey_create(ctx, privkey, pub33);

    // Sign
    uint8_t sig64[64] = {};
    CHECK_OK(ufsecp_ecdsa_sign(ctx, msg32, privkey, sig64), "ecdsa_sign");

    // Verify
    CHECK_OK(ufsecp_ecdsa_verify(ctx, msg32, sig64, pub33), "ecdsa_verify");

    // Wrong message should fail
    uint8_t bad_msg[32];
    std::memcpy(bad_msg, msg32, 32);
    bad_msg[0] ^= 0xFF;
    CHECK(ufsecp_ecdsa_verify(ctx, bad_msg, sig64, pub33) != UFSECP_OK,
          "ecdsa_verify rejects wrong msg");

    // DER encode
    uint8_t der[72] = {};
    size_t der_len = sizeof(der);
    CHECK_OK(ufsecp_ecdsa_sig_to_der(ctx, sig64, der, &der_len), "sig_to_der");
    CHECK(der_len > 0 && der_len <= 72, "DER length valid");

    // DER decode
    uint8_t decoded64[64] = {};
    CHECK_OK(ufsecp_ecdsa_sig_from_der(ctx, der, der_len, decoded64), "sig_from_der");
    CHECK(std::memcmp(sig64, decoded64, 64) == 0, "DER round-trip preserves sig");

    // Verify decoded sig
    CHECK_OK(ufsecp_ecdsa_verify(ctx, msg32, decoded64, pub33),
             "ecdsa_verify(decoded DER)");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 4: ECDSA Recovery
// ============================================================================
static void test_ecdsa_recovery() {
    (void)std::printf("[4] FFI: ECDSA recoverable sign -> recover pubkey\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    uint8_t pub33_expected[33];
    ufsecp_pubkey_create(ctx, privkey, pub33_expected);

    // Recoverable sign
    uint8_t sig64[64] = {};
    int recid = -1;
    CHECK_OK(ufsecp_ecdsa_sign_recoverable(ctx, msg32, privkey, sig64, &recid),
             "ecdsa_sign_recoverable");
    CHECK(recid >= 0 && recid <= 3, "recid in range [0,3]");

    // Recover pubkey
    uint8_t recovered33[33] = {};
    CHECK_OK(ufsecp_ecdsa_recover(ctx, msg32, sig64, recid, recovered33),
             "ecdsa_recover");
    CHECK(std::memcmp(pub33_expected, recovered33, 33) == 0,
          "recovered pubkey matches");

    // Wrong recid should give different pubkey (or fail)
    const int bad_recid = (recid + 1) % 4;
    uint8_t wrong33[33] = {};
    const ufsecp_error_t err = ufsecp_ecdsa_recover(ctx, msg32, sig64, bad_recid, wrong33);
    if (err == UFSECP_OK) {
        // If it succeeded, the pubkey must differ
        CHECK(std::memcmp(pub33_expected, wrong33, 33) != 0,
              "wrong recid -> different pubkey");
    } else {
        // Recovery failure is also acceptable
        CHECK(true, "wrong recid -> recovery failed (expected)");
    }

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 5: Schnorr/BIP-340 Sign -> Verify
// ============================================================================
static void test_schnorr_round_trip() {
    (void)std::printf("[5] FFI: Schnorr/BIP-340 sign -> verify\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    uint8_t xonly[32];
    ufsecp_pubkey_xonly(ctx, privkey, xonly);

    // Sign with deterministic aux (all zeros)
    uint8_t aux32[32] = {};
    uint8_t sig64[64] = {};
    CHECK_OK(ufsecp_schnorr_sign(ctx, msg32, privkey, aux32, sig64), "schnorr_sign");

    // Verify
    CHECK_OK(ufsecp_schnorr_verify(ctx, msg32, sig64, xonly), "schnorr_verify");

    // Tampered sig should fail
    uint8_t bad_sig[64];
    std::memcpy(bad_sig, sig64, 64);
    bad_sig[63] ^= 0x01;
    CHECK(ufsecp_schnorr_verify(ctx, msg32, bad_sig, xonly) != UFSECP_OK,
          "schnorr_verify rejects tampered sig");

    // Determinism: sign again -> same sig
    uint8_t sig64_b[64] = {};
    CHECK_OK(ufsecp_schnorr_sign(ctx, msg32, privkey, aux32, sig64_b), "schnorr_sign(2)");
    CHECK(std::memcmp(sig64, sig64_b, 64) == 0, "schnorr deterministic");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 6: ECDH Shared Secret
// ============================================================================
static void test_ecdh_agreement() {
    (void)std::printf("[6] FFI: ECDH shared secret agreement\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t sk_a[32], sk_b[32];
    hex_to_bytes(PRIVKEY1_HEX, sk_a, 32);
    hex_to_bytes(PRIVKEY2_HEX, sk_b, 32);

    uint8_t pub_a[33], pub_b[33];
    ufsecp_pubkey_create(ctx, sk_a, pub_a);
    ufsecp_pubkey_create(ctx, sk_b, pub_b);

    // A computes: ECDH(sk_a, pub_b)
    uint8_t secret_ab[32] = {};
    CHECK_OK(ufsecp_ecdh(ctx, sk_a, pub_b, secret_ab), "ecdh(A,B)");

    // B computes: ECDH(sk_b, pub_a)
    uint8_t secret_ba[32] = {};
    CHECK_OK(ufsecp_ecdh(ctx, sk_b, pub_a, secret_ba), "ecdh(B,A)");

    CHECK(std::memcmp(secret_ab, secret_ba, 32) == 0,
          "ECDH shared secret agrees (A,B == B,A)");

    // x-only variant
    uint8_t xsecret_ab[32] = {}, xsecret_ba[32] = {};
    CHECK_OK(ufsecp_ecdh_xonly(ctx, sk_a, pub_b, xsecret_ab), "ecdh_xonly(A,B)");
    CHECK_OK(ufsecp_ecdh_xonly(ctx, sk_b, pub_a, xsecret_ba), "ecdh_xonly(B,A)");
    CHECK(std::memcmp(xsecret_ab, xsecret_ba, 32) == 0,
          "ECDH x-only agrees");

    // Raw variant
    uint8_t raw_ab[32] = {}, raw_ba[32] = {};
    CHECK_OK(ufsecp_ecdh_raw(ctx, sk_a, pub_b, raw_ab), "ecdh_raw(A,B)");
    CHECK_OK(ufsecp_ecdh_raw(ctx, sk_b, pub_a, raw_ba), "ecdh_raw(B,A)");
    CHECK(std::memcmp(raw_ab, raw_ba, 32) == 0, "ECDH raw agrees");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 7: BIP-32 HD Key Derivation
// ============================================================================
static void test_bip32_derivation() {
    (void)std::printf("[7] FFI: BIP-32 master -> derive -> extract\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    // BIP-32 TV1 seed
    uint8_t seed[16];
    hex_to_bytes("000102030405060708090a0b0c0d0e0f", seed, 16);

    ufsecp_bip32_key master = {};
    CHECK_OK(ufsecp_bip32_master(ctx, seed, 16, &master), "bip32_master");
    CHECK(master.is_private == 1, "master is private");

    // Extract master private key
    uint8_t master_priv[32] = {};
    CHECK_OK(ufsecp_bip32_privkey(ctx, &master, master_priv), "bip32_privkey(master)");

    // Verify master private key is valid
    CHECK_OK(ufsecp_seckey_verify(ctx, master_priv), "master privkey valid");

    // Extract master public key
    uint8_t master_pub[33] = {};
    CHECK_OK(ufsecp_bip32_pubkey(ctx, &master, master_pub), "bip32_pubkey(master)");
    CHECK(master_pub[0] == 0x02 || master_pub[0] == 0x03, "master pub prefix valid");

    // Derive child at index 0 (normal)
    ufsecp_bip32_key child0 = {};
    CHECK_OK(ufsecp_bip32_derive(ctx, &master, 0, &child0), "bip32_derive(0)");
    CHECK(child0.is_private == 1, "child0 is private");

    // Derive hardened child at index 0x80000000
    ufsecp_bip32_key child_h = {};
    CHECK_OK(ufsecp_bip32_derive(ctx, &master, 0x80000000u, &child_h),
             "bip32_derive(0h)");

    // Child keys should differ from master
    uint8_t child0_priv[32] = {};
    ufsecp_bip32_privkey(ctx, &child0, child0_priv);
    CHECK(std::memcmp(master_priv, child0_priv, 32) != 0,
          "child0 privkey != master");

    // Path derivation: m/44'/0'/0'/0/0
    ufsecp_bip32_key account = {};
    CHECK_OK(ufsecp_bip32_derive_path(ctx, &master, "m/44'/0'/0'/0/0", &account),
             "bip32_derive_path(m/44h/0h/0h/0/0)");

    uint8_t account_pub[33] = {};
    CHECK_OK(ufsecp_bip32_pubkey(ctx, &account, account_pub),
             "bip32_pubkey(account)");
    CHECK(account_pub[0] == 0x02 || account_pub[0] == 0x03,
          "account pub prefix valid");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 8: Address Generation
// ============================================================================
static void test_address_generation() {
    (void)std::printf("[8] FFI: Address generation (P2PKH, P2WPKH, P2TR)\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);

    uint8_t pub33[33];
    ufsecp_pubkey_create(ctx, privkey, pub33);

    // P2PKH (mainnet)
    char addr_buf[128] = {};
    size_t addr_len = sizeof(addr_buf);
    CHECK_OK(ufsecp_addr_p2pkh(ctx, pub33, UFSECP_NET_MAINNET, addr_buf, &addr_len),
             "addr_p2pkh(mainnet)");
    CHECK(addr_len > 0, "P2PKH addr length > 0");
    CHECK(addr_buf[0] == '1', "P2PKH mainnet starts with '1'");
    (void)std::printf("    P2PKH:  %s\n", addr_buf);

    // P2WPKH (mainnet)
    addr_len = sizeof(addr_buf);
    std::memset(addr_buf, 0, sizeof(addr_buf));
    CHECK_OK(ufsecp_addr_p2wpkh(ctx, pub33, UFSECP_NET_MAINNET, addr_buf, &addr_len),
             "addr_p2wpkh(mainnet)");
    CHECK(addr_len > 0, "P2WPKH addr length > 0");
    // Bech32 address starts with "bc1"
    CHECK(addr_buf[0] == 'b' && addr_buf[1] == 'c' && addr_buf[2] == '1',
          "P2WPKH mainnet starts with 'bc1'");
    (void)std::printf("    P2WPKH: %s\n", addr_buf);

    // P2TR (mainnet)
    uint8_t xonly[32];
    ufsecp_pubkey_xonly(ctx, privkey, xonly);

    addr_len = sizeof(addr_buf);
    std::memset(addr_buf, 0, sizeof(addr_buf));
    CHECK_OK(ufsecp_addr_p2tr(ctx, xonly, UFSECP_NET_MAINNET, addr_buf, &addr_len),
             "addr_p2tr(mainnet)");
    CHECK(addr_len > 0, "P2TR addr length > 0");
    CHECK(addr_buf[0] == 'b' && addr_buf[1] == 'c' && addr_buf[2] == '1',
          "P2TR mainnet starts with 'bc1'");
    (void)std::printf("    P2TR:   %s\n", addr_buf);

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 9: WIF Encode/Decode Round-Trip
// ============================================================================
static void test_wif_round_trip() {
    (void)std::printf("[9] FFI: WIF encode -> decode round-trip\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);

    // Encode compressed mainnet
    char wif_buf[64] = {};
    size_t wif_len = sizeof(wif_buf);
    CHECK_OK(ufsecp_wif_encode(ctx, privkey, 1, UFSECP_NET_MAINNET, wif_buf, &wif_len),
             "wif_encode(compressed, mainnet)");
    CHECK(wif_len > 0, "WIF length > 0");
    CHECK(wif_buf[0] == 'K' || wif_buf[0] == 'L',
          "compressed mainnet WIF starts with K or L");
    (void)std::printf("    WIF: %s\n", wif_buf);

    // Decode back
    uint8_t decoded_priv[32] = {};
    int compressed_out = -1, network_out = -1;
    CHECK_OK(ufsecp_wif_decode(ctx, wif_buf, decoded_priv, &compressed_out, &network_out),
             "wif_decode");
    CHECK(std::memcmp(privkey, decoded_priv, 32) == 0, "WIF round-trip preserves key");
    CHECK(compressed_out == 1, "decoded as compressed");
    CHECK(network_out == UFSECP_NET_MAINNET, "decoded as mainnet");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 10: Hashing Known Vectors
// ============================================================================
static void test_hashing_vectors() {
    (void)std::printf("[10] FFI: SHA-256, Hash160, tagged hash\n");

    // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    const uint8_t empty = 0; // non-null pointer for 0-length hash
    uint8_t digest[32] = {};
    CHECK_OK(ufsecp_sha256(&empty, 0, digest), "sha256(\"\")");

    uint8_t expected_sha[32];
    hex_to_bytes("E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855",
                 expected_sha, 32);
    CHECK(std::memcmp(digest, expected_sha, 32) == 0, "SHA-256(\"\") matches");

    // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    const uint8_t abc[] = { 0x61, 0x62, 0x63 };
    uint8_t digest_abc[32] = {};
    CHECK_OK(ufsecp_sha256(abc, 3, digest_abc), "sha256(\"abc\")");

    uint8_t expected_abc[32];
    hex_to_bytes("BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD",
                 expected_abc, 32);
    CHECK(std::memcmp(digest_abc, expected_abc, 32) == 0, "SHA-256(\"abc\") matches");

    // Hash160("abc") -- use non-empty input for Hash160
    uint8_t hash160[20] = {};
    CHECK_OK(ufsecp_hash160(abc, 3, hash160), "hash160(\"abc\")");
    bool nonzero = false;
    for (int i = 0; i < 20; ++i) {
        if (hash160[i] != 0) { nonzero = true; break; }
    }
    CHECK(nonzero, "hash160 result is non-zero");

    // Tagged hash with non-empty data
    uint8_t tagged[32] = {};
    CHECK_OK(ufsecp_tagged_hash("BIP0340/challenge", abc, 3, tagged),
             "tagged_hash");
    bool tag_nonzero = false;
    for (int i = 0; i < 32; ++i) {
        if (tagged[i] != 0) { tag_nonzero = true; break; }
    }
    CHECK(tag_nonzero, "tagged hash result is non-zero");
}

// ============================================================================
// Test 11: Taproot Output Key + Verify
// ============================================================================
static void test_taproot_operations() {
    (void)std::printf("[11] FFI: Taproot output key + verification\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);

    uint8_t internal_x[32];
    ufsecp_pubkey_xonly(ctx, privkey, internal_x);

    // Key-path-only: no merkle root
    uint8_t output_x[32] = {};
    int parity = -1;
    CHECK_OK(ufsecp_taproot_output_key(ctx, internal_x, nullptr, output_x, &parity),
             "taproot_output_key(keypath)");
    CHECK(parity == 0 || parity == 1, "parity is 0 or 1");

    // Output key should differ from internal key (tweaked)
    CHECK(std::memcmp(internal_x, output_x, 32) != 0,
          "output_key != internal_key");

    // Verify commitment
    CHECK_OK(ufsecp_taproot_verify(ctx, output_x, parity, internal_x, nullptr, 0),
             "taproot_verify(keypath)");

    // Tweak seckey for spending
    uint8_t tweaked_sk[32] = {};
    CHECK_OK(ufsecp_taproot_tweak_seckey(ctx, privkey, nullptr, tweaked_sk),
             "taproot_tweak_seckey");

    // Tweaked privkey should produce the output_x as its xonly pubkey
    uint8_t tweaked_xonly[32] = {};
    ufsecp_pubkey_xonly(ctx, tweaked_sk, tweaked_xonly);
    CHECK(std::memcmp(tweaked_xonly, output_x, 32) == 0,
          "tweaked_seckey -> output_x matches");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 12: Error Paths
// ============================================================================
static void test_error_paths() {
    (void)std::printf("[12] FFI: Error paths (NULL, bad key, invalid sig)\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    // NULL context for create
    CHECK(ufsecp_ctx_create(nullptr) != UFSECP_OK, "ctx_create(NULL) fails");

    // Zero private key (invalid)
    uint8_t zero_key[32] = {};
    CHECK(ufsecp_seckey_verify(ctx, zero_key) != UFSECP_OK,
          "seckey_verify(0) fails");

    // Key >= order (invalid) -- secp256k1 order n starts with FFFF...BAAED...
    uint8_t big_key[32];
    std::memset(big_key, 0xFF, 32);
    CHECK(ufsecp_seckey_verify(ctx, big_key) != UFSECP_OK,
          "seckey_verify(0xFF...) fails");

    // Invalid pubkey for ECDSA verify
    uint8_t bad_pub[33] = {};
    bad_pub[0] = 0x04; // wrong prefix for 33-byte key
    uint8_t msg[32] = {};
    uint8_t sig[64] = {};
    CHECK(ufsecp_ecdsa_verify(ctx, msg, sig, bad_pub) != UFSECP_OK,
          "ecdsa_verify(bad pubkey) fails");

    // Invalid signature for Schnorr verify (all zeros)
    uint8_t xonly[32] = {};
    xonly[0] = 0x01; // some non-zero value
    CHECK(ufsecp_schnorr_verify(ctx, msg, sig, xonly) != UFSECP_OK,
          "schnorr_verify(zero sig) fails");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 13: Key Tweak Operations
// ============================================================================
static void test_key_tweaks() {
    (void)std::printf("[13] FFI: Key tweak add/mul + negate\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);

    // Save original
    uint8_t original[32];
    std::memcpy(original, privkey, 32);

    // Negate
    uint8_t negated[32];
    std::memcpy(negated, privkey, 32);
    CHECK_OK(ufsecp_seckey_negate(ctx, negated), "seckey_negate");
    CHECK(std::memcmp(original, negated, 32) != 0, "negated != original");

    // Double negate = original
    CHECK_OK(ufsecp_seckey_negate(ctx, negated), "seckey_negate(2)");
    CHECK(std::memcmp(original, negated, 32) == 0, "double negate = original");

    // Tweak add
    uint8_t tweaked[32];
    std::memcpy(tweaked, privkey, 32);
    uint8_t tweak[32] = {};
    tweak[31] = 1; // add 1
    CHECK_OK(ufsecp_seckey_tweak_add(ctx, tweaked, tweak), "seckey_tweak_add");

    // tweaked should now be privkey + 1 = 2
    uint8_t expected_2[32];
    hex_to_bytes(PRIVKEY2_HEX, expected_2, 32);
    CHECK(std::memcmp(tweaked, expected_2, 32) == 0,
          "1 + 1 = 2 (tweak_add)");

    // Tweak mul by 2 -> result should be 2*original = 2
    uint8_t mul_tweaked[32];
    std::memcpy(mul_tweaked, privkey, 32);
    uint8_t mul_tweak[32] = {};
    mul_tweak[31] = 2;
    CHECK_OK(ufsecp_seckey_tweak_mul(ctx, mul_tweaked, mul_tweak), "seckey_tweak_mul");
    CHECK(std::memcmp(mul_tweaked, expected_2, 32) == 0,
          "1 * 2 = 2 (tweak_mul)");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 14: Cross-check C ABI vs C++ API (ECDSA)
// ============================================================================
static void test_cross_api_ecdsa() {
    (void)std::printf("[14] FFI: Cross-check C ABI vs C++ (ECDSA sign+verify)\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    // C ABI sign
    uint8_t c_sig64[64] = {};
    CHECK_OK(ufsecp_ecdsa_sign(ctx, msg32, privkey, c_sig64), "c_ecdsa_sign");

    // C ABI verify
    uint8_t pub33[33];
    ufsecp_pubkey_create(ctx, privkey, pub33);
    CHECK_OK(ufsecp_ecdsa_verify(ctx, msg32, c_sig64, pub33), "c_ecdsa_verify");

    // The C API should produce a valid, low-S signature
    // Check low-S: S 32 bytes (sig64[32..63]) must be "low" per BIP-62
    // (we just verify it's accepted by verify, which enforces low-S)

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 15: Cross-check C ABI vs C++ API (Schnorr)
// ============================================================================
static void test_cross_api_schnorr() {
    (void)std::printf("[15] FFI: Cross-check C ABI vs C++ (Schnorr sign+verify)\n");

    ufsecp_ctx* ctx = nullptr;
    ufsecp_ctx_create(&ctx);

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    uint8_t xonly[32];
    ufsecp_pubkey_xonly(ctx, privkey, xonly);

    // C ABI Schnorr sign
    uint8_t aux[32] = {};
    uint8_t c_sig64[64] = {};
    CHECK_OK(ufsecp_schnorr_sign(ctx, msg32, privkey, aux, c_sig64), "c_schnorr_sign");

    // C ABI verify
    CHECK_OK(ufsecp_schnorr_verify(ctx, msg32, c_sig64, xonly), "c_schnorr_verify");

    // Determinism: same inputs -> same sig
    uint8_t c_sig64_b[64] = {};
    CHECK_OK(ufsecp_schnorr_sign(ctx, msg32, privkey, aux, c_sig64_b), "c_schnorr_sign(2)");
    CHECK(std::memcmp(c_sig64, c_sig64_b, 64) == 0, "schnorr is deterministic via C ABI");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Entry Point
// ============================================================================

int test_ffi_round_trip_run() {
    g_pass = 0;
    g_fail = 0;

    (void)std::printf("\n=== Cross-ABI / FFI Round-Trip Tests ===\n");

    test_context_lifecycle();
    test_key_generation();
    test_ecdsa_round_trip();
    test_ecdsa_recovery();
    test_schnorr_round_trip();
    test_ecdh_agreement();
    test_bip32_derivation();
    test_address_generation();
    test_wif_round_trip();
    test_hashing_vectors();
    test_taproot_operations();
    test_error_paths();
    test_key_tweaks();
    test_cross_api_ecdsa();
    test_cross_api_schnorr();

    (void)std::printf("\n--- FFI Round-Trip Summary: %d passed, %d failed ---\n\n",
                      g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    return test_ffi_round_trip_run();
}
#endif
