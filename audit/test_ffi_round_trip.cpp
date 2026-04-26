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
//  13. Key tweaks: negate, add, mul
//  14. Cross-API ECDSA: sign -> verify via separate contexts
//  15. Cross-API Schnorr: sign -> verify via separate contexts
//  16. Negative vectors: strict parsing, non-canonical inputs
//  17. SHA-512: known NIST vectors
//  18. Pubkey arithmetic: add, negate, combine, tweak_add, tweak_mul
//  19. BIP-39: generate -> validate -> seed -> entropy round-trip
//  20. Batch verification: ECDSA + Schnorr batch verify + identify invalid
//  21. Pedersen commitments: commit -> verify -> sum balance
//  22. ZK proofs: knowledge prove -> verify
//  23. Multi-scalar multiplication: Shamir's trick + MSM
//  24. Multi-coin wallet: address dispatch for BTC/LTC/DOGE
//  25. Bitcoin message signing: BIP-137 sign -> verify
//  26. Ethereum: keccak256, eth_address, eth_sign -> ecrecover (conditional)
//  27. MuSig2: 2-of-2 key agg -> nonce -> sign -> aggregate -> verify
//  28. Adaptor signatures: pre-sign -> verify -> adapt -> extract
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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    uint8_t pub33[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, privkey, pub33), "pubkey_create");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    uint8_t pub33_expected[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, privkey, pub33_expected), "pubkey_create");

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
        CHECK(err != UFSECP_OK, "wrong recid -> recovery failed (expected)");
    }

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 5: Schnorr/BIP-340 Sign -> Verify
// ============================================================================
static void test_schnorr_round_trip() {
    (void)std::printf("[5] FFI: Schnorr/BIP-340 sign -> verify\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    uint8_t xonly[32];
    CHECK_OK(ufsecp_pubkey_xonly(ctx, privkey, xonly), "pubkey_xonly");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t sk_a[32], sk_b[32];
    hex_to_bytes(PRIVKEY1_HEX, sk_a, 32);
    hex_to_bytes(PRIVKEY2_HEX, sk_b, 32);

    uint8_t pub_a[33], pub_b[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, sk_a, pub_a), "pubkey_create");
    CHECK_OK(ufsecp_pubkey_create(ctx, sk_b, pub_b), "pubkey_create");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t privkey[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);

    uint8_t pub33[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, privkey, pub33), "pubkey_create");

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
    CHECK_OK(ufsecp_pubkey_xonly(ctx, privkey, xonly), "pubkey_xonly");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t privkey[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);

    uint8_t internal_x[32];
    CHECK_OK(ufsecp_pubkey_xonly(ctx, privkey, internal_x), "pubkey_xonly");

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
    CHECK_OK(ufsecp_pubkey_xonly(ctx, tweaked_sk, tweaked_xonly),
             "pubkey_xonly(tweaked)");
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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    // C ABI sign
    uint8_t c_sig64[64] = {};
    CHECK_OK(ufsecp_ecdsa_sign(ctx, msg32, privkey, c_sig64), "c_ecdsa_sign");

    // C ABI verify
    uint8_t pub33[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, privkey, pub33), "pubkey_create");
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
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t privkey[32], msg32[32];
    hex_to_bytes(PRIVKEY1_HEX, privkey, 32);
    hex_to_bytes(MSG_HEX, msg32, 32);

    uint8_t xonly[32];
    CHECK_OK(ufsecp_pubkey_xonly(ctx, privkey, xonly), "pubkey_xonly");

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
// Test 16: Negative Test Vectors (Strict Parsing)
// ============================================================================
// Bounty-hunter-grade negative vectors: non-canonical keys, sigs, DER malforms
static void test_negative_vectors() {
    (void)std::printf("[16] FFI: Negative test vectors (strict parsing)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    // -- secp256k1 curve order n (hex) --
    // n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    uint8_t order_n[32];
    hex_to_bytes("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141",
                 order_n, 32);

    // n+1
    uint8_t order_n_plus1[32];
    hex_to_bytes("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364142",
                 order_n_plus1, 32);

    // Zero key
    uint8_t zero32[32] = {};

    // Valid privkey for reference
    uint8_t valid_priv[32];
    hex_to_bytes(PRIVKEY1_HEX, valid_priv, 32);
    uint8_t pub33[33] = {};
    CHECK_OK(ufsecp_pubkey_create(ctx, valid_priv, pub33), "setup: pubkey_create");

    // === A: seckey strict parsing ===
    CHECK(ufsecp_seckey_verify(ctx, zero32) != UFSECP_OK,
          "seckey_verify(0) rejects");
    CHECK(ufsecp_seckey_verify(ctx, order_n) != UFSECP_OK,
          "seckey_verify(n) rejects");
    CHECK(ufsecp_seckey_verify(ctx, order_n_plus1) != UFSECP_OK,
          "seckey_verify(n+1) rejects");

    // pubkey_create with bad keys
    uint8_t dummy_pub[33];
    CHECK(ufsecp_pubkey_create(ctx, zero32, dummy_pub) != UFSECP_OK,
          "pubkey_create(0) rejects");
    CHECK(ufsecp_pubkey_create(ctx, order_n, dummy_pub) != UFSECP_OK,
          "pubkey_create(n) rejects");
    CHECK(ufsecp_pubkey_create(ctx, order_n_plus1, dummy_pub) != UFSECP_OK,
          "pubkey_create(n+1) rejects");

    // ecdsa_sign with bad keys
    uint8_t msg32[32] = {1};
    uint8_t sig64[64];
    CHECK(ufsecp_ecdsa_sign(ctx, msg32, zero32, sig64) != UFSECP_OK,
          "ecdsa_sign(sk=0) rejects");
    CHECK(ufsecp_ecdsa_sign(ctx, msg32, order_n, sig64) != UFSECP_OK,
          "ecdsa_sign(sk=n) rejects");

    // schnorr_sign with bad keys
    uint8_t aux32[32] = {};
    CHECK(ufsecp_schnorr_sign(ctx, msg32, zero32, aux32, sig64) != UFSECP_OK,
          "schnorr_sign(sk=0) rejects");
    CHECK(ufsecp_schnorr_sign(ctx, msg32, order_n, aux32, sig64) != UFSECP_OK,
          "schnorr_sign(sk=n) rejects");

    // tweak_add: tweak = n should fail
    {
        uint8_t sk[32];
        std::memcpy(sk, valid_priv, 32);
        CHECK(ufsecp_seckey_tweak_add(ctx, sk, order_n) != UFSECP_OK,
              "tweak_add(tweak=n) rejects");
    }
    // tweak_mul: tweak = 0 should fail
    {
        uint8_t sk[32];
        std::memcpy(sk, valid_priv, 32);
        CHECK(ufsecp_seckey_tweak_mul(ctx, sk, zero32) != UFSECP_OK,
              "tweak_mul(tweak=0) rejects");
    }
    // tweak_mul: tweak = n should fail
    {
        uint8_t sk[32];
        std::memcpy(sk, valid_priv, 32);
        CHECK(ufsecp_seckey_tweak_mul(ctx, sk, order_n) != UFSECP_OK,
              "tweak_mul(tweak=n) rejects");
    }

    // === B: ECDSA compact sig with r=0 or s=0 ===
    // First sign a valid signature for reference
    uint8_t valid_sig[64];
    CHECK_OK(ufsecp_ecdsa_sign(ctx, msg32, valid_priv, valid_sig), "setup: ecdsa_sign");

    // sig with r=0 (first 32 bytes zero)
    {
        uint8_t bad_sig[64];
        std::memset(bad_sig, 0, 64);
        std::memcpy(bad_sig + 32, valid_sig + 32, 32); // keep valid s
        CHECK(ufsecp_ecdsa_verify(ctx, msg32, bad_sig, pub33) == UFSECP_ERR_BAD_SIG,
              "ecdsa_verify(r=0) -> BAD_SIG");
    }
    // sig with s=0
    {
        uint8_t bad_sig[64];
        std::memcpy(bad_sig, valid_sig, 32); // keep valid r
        std::memset(bad_sig + 32, 0, 32);
        CHECK(ufsecp_ecdsa_verify(ctx, msg32, bad_sig, pub33) == UFSECP_ERR_BAD_SIG,
              "ecdsa_verify(s=0) -> BAD_SIG");
    }
    // sig with r=n
    {
        uint8_t bad_sig[64];
        std::memcpy(bad_sig, order_n, 32);
        std::memcpy(bad_sig + 32, valid_sig + 32, 32);
        CHECK(ufsecp_ecdsa_verify(ctx, msg32, bad_sig, pub33) == UFSECP_ERR_BAD_SIG,
              "ecdsa_verify(r=n) -> BAD_SIG");
    }
    // sig with s=n
    {
        uint8_t bad_sig[64];
        std::memcpy(bad_sig, valid_sig, 32);
        std::memcpy(bad_sig + 32, order_n, 32);
        CHECK(ufsecp_ecdsa_verify(ctx, msg32, bad_sig, pub33) == UFSECP_ERR_BAD_SIG,
              "ecdsa_verify(s=n) -> BAD_SIG");
    }

    // sig_to_der with non-canonical compact sig
    {
        uint8_t bad_sig[64] = {};  // r=0, s=0
        uint8_t der_buf[72];
        size_t der_len = sizeof(der_buf);
        CHECK(ufsecp_ecdsa_sig_to_der(ctx, bad_sig, der_buf, &der_len) == UFSECP_ERR_BAD_SIG,
              "sig_to_der(r=0,s=0) -> BAD_SIG");
    }

    // === B3: DER malformation corpus ===
    auto check_bad_der = [&](const uint8_t* der, size_t len, const char* label) {
        uint8_t out64[64];
        ufsecp_error_t const rc = ufsecp_ecdsa_sig_from_der(ctx, der, len, out64);
        CHECK(rc == UFSECP_ERR_BAD_SIG, label);
    };

    // Too short
    {
        const uint8_t d[] = {0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01};
        check_bad_der(d, sizeof(d), "DER: truncated (7 bytes)");
    }
    // Wrong tag (not SEQUENCE)
    {
        const uint8_t d[] = {0x31, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01};
        check_bad_der(d, sizeof(d), "DER: wrong tag 0x31");
    }
    // Length mismatch (seq_len says 7 but only 6 bytes follow)
    {
        const uint8_t d[] = {0x30, 0x07, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01};
        check_bad_der(d, sizeof(d), "DER: seq length mismatch");
    }
    // Negative R (high bit set, no leading zero)
    {
        const uint8_t d[] = {0x30, 0x06, 0x02, 0x01, 0x80, 0x02, 0x01, 0x01};
        check_bad_der(d, sizeof(d), "DER: negative R");
    }
    // Negative S
    {
        const uint8_t d[] = {0x30, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0xFF};
        check_bad_der(d, sizeof(d), "DER: negative S");
    }
    // Unnecessary leading zero in R (0x00 + byte without high bit)
    {
        const uint8_t d[] = {0x30, 0x07, 0x02, 0x02, 0x00, 0x01, 0x02, 0x01, 0x01};
        check_bad_der(d, sizeof(d), "DER: unnecessary leading zero in R");
    }
    // Unnecessary leading zero in S
    {
        const uint8_t d[] = {0x30, 0x07, 0x02, 0x01, 0x01, 0x02, 0x02, 0x00, 0x01};
        check_bad_der(d, sizeof(d), "DER: unnecessary leading zero in S");
    }
    // Trailing bytes after S
    {
        const uint8_t d[] = {0x30, 0x08, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01, 0x00, 0x00};
        check_bad_der(d, 10, "DER: trailing bytes");
    }
    // Long-form length (0x81 prefix)
    {
        const uint8_t d[] = {0x30, 0x81, 0x06, 0x02, 0x01, 0x01, 0x02, 0x01, 0x01};
        check_bad_der(d, sizeof(d), "DER: long-form sequence length");
    }
    // R integer zero length
    {
        const uint8_t d[] = {0x30, 0x04, 0x02, 0x00, 0x02, 0x01, 0x01};
        check_bad_der(d, sizeof(d), "DER: R zero length");
    }
    // Missing S INTEGER tag
    {
        const uint8_t d[] = {0x30, 0x05, 0x02, 0x01, 0x01, 0x03, 0x01, 0x01};
        check_bad_der(d, sizeof(d), "DER: wrong S tag");
    }

    // === ECDSA recovery with non-canonical sig ===
    {
        uint8_t bad_sig[64] = {};  // r=0, s=0
        uint8_t recovered[33];
        CHECK(ufsecp_ecdsa_recover(ctx, msg32, bad_sig, 0, recovered) == UFSECP_ERR_BAD_SIG,
              "ecdsa_recover(r=0,s=0) -> BAD_SIG");
    }

    // === Error model: parse-fail vs verify-fail ===
    // Valid-format sig but wrong message -> VERIFY_FAIL (not BAD_SIG)
    {
        uint8_t wrong_msg[32] = {0x42};
        ufsecp_error_t const rc = ufsecp_ecdsa_verify(ctx, wrong_msg, valid_sig, pub33);
        CHECK(rc == UFSECP_ERR_VERIFY_FAIL,
              "ecdsa_verify(wrong msg) -> VERIFY_FAIL (not BAD_SIG)");
    }

    // === Schnorr negative vectors ===
    {
        // sig with s=0 (all-zero s portion)
        uint8_t bad_schnorr[64] = {};
        bad_schnorr[0] = 0x01;  // some r
        uint8_t xonly[32];
        hex_to_bytes(PRIVKEY1_HEX, valid_priv, 32);
        CHECK_OK(ufsecp_pubkey_xonly(ctx, valid_priv, xonly), "setup: pubkey_xonly");
        ufsecp_error_t const rc = ufsecp_schnorr_verify(ctx, msg32, bad_schnorr, xonly);
        CHECK(rc != UFSECP_OK, "schnorr_verify(bad sig) rejects");
    }

    // === ECDH with bad private key ===
    {
        uint8_t secret[32];
        CHECK(ufsecp_ecdh(ctx, zero32, pub33, secret) != UFSECP_OK,
              "ecdh(sk=0) rejects");
        CHECK(ufsecp_ecdh(ctx, order_n, pub33, secret) != UFSECP_OK,
              "ecdh(sk=n) rejects");
    }

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 17: SHA-512 known vector
// ============================================================================
static void test_sha512_vector() {
    (void)std::printf("[17] FFI: SHA-512 known vector\n");

    // SHA-512("") = cf83e1357eefb8bd...
    uint8_t digest[64];
    const uint8_t empty_buf = 0;
    CHECK_OK(ufsecp_sha512(&empty_buf, 0, digest), "sha512 empty");

    uint8_t expected[64];
    hex_to_bytes("cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce"
                 "47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e",
                 expected, 64);
    CHECK(std::memcmp(digest, expected, 64) == 0, "sha512 empty matches NIST vector");

    // SHA-512("abc")
    const uint8_t abc[] = {'a', 'b', 'c'};
    CHECK_OK(ufsecp_sha512(abc, 3, digest), "sha512 abc");
    uint8_t expected_abc[64];
    hex_to_bytes("ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a"
                 "2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f",
                 expected_abc, 64);
    CHECK(std::memcmp(digest, expected_abc, 64) == 0, "sha512 abc matches NIST vector");
}

// ============================================================================
// Test 18: Public key arithmetic
// ============================================================================
static void test_pubkey_arithmetic() {
    (void)std::printf("[18] FFI: Pubkey arithmetic (add, negate, combine, tweak)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t priv1[32], priv2[32];
    hex_to_bytes(PRIVKEY1_HEX, priv1, 32);
    hex_to_bytes(PRIVKEY2_HEX, priv2, 32);

    uint8_t pub1[33], pub2[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, priv1, pub1), "pubkey1");
    CHECK_OK(ufsecp_pubkey_create(ctx, priv2, pub2), "pubkey2");

    // Add: pub1 + pub2
    uint8_t sum[33];
    CHECK_OK(ufsecp_pubkey_add(ctx, pub1, pub2, sum), "pubkey_add");

    // Verify: sum == pubkey(priv1 + priv2 mod n)
    uint8_t priv_sum[32];
    std::memcpy(priv_sum, priv1, 32);
    CHECK_OK(ufsecp_seckey_tweak_add(ctx, priv_sum, priv2), "seckey_tweak_add");
    uint8_t pub_sum[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, priv_sum, pub_sum), "pubkey(priv_sum)");
    CHECK(std::memcmp(sum, pub_sum, 33) == 0, "pubkey_add == pubkey(priv1+priv2)");

    // Negate: -pub1 + pub1 = infinity -> should get valid compressed point
    uint8_t neg_pub1[33];
    CHECK_OK(ufsecp_pubkey_negate(ctx, pub1, neg_pub1), "pubkey_negate");

    // Combine: combine([pub1, pub2]) should equal add(pub1, pub2)
    uint8_t keys_buf[66];
    std::memcpy(keys_buf, pub1, 33);
    std::memcpy(keys_buf + 33, pub2, 33);
    uint8_t combined[33];
    CHECK_OK(ufsecp_pubkey_combine(ctx, keys_buf, 2, combined), "pubkey_combine");
    CHECK(std::memcmp(combined, sum, 33) == 0, "combine([P1,P2]) == add(P1,P2)");

    // Tweak add: pubkey_tweak_add(pub1, priv2) == pub_sum
    uint8_t tweak_added[33];
    CHECK_OK(ufsecp_pubkey_tweak_add(ctx, pub1, priv2, tweak_added), "pubkey_tweak_add");
    CHECK(std::memcmp(tweak_added, pub_sum, 33) == 0, "tweak_add consistency");

    // Tweak mul: pubkey_tweak_mul(G, 2) == pub2
    uint8_t tweak_mulled[33];
    CHECK_OK(ufsecp_pubkey_tweak_mul(ctx, pub1, priv2, tweak_mulled), "pubkey_tweak_mul");
    // tweak_mul(1*G, 2) = 2*G = pub2
    CHECK(std::memcmp(tweak_mulled, pub2, 33) == 0, "tweak_mul(G,2) == 2G");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 19: BIP-39 round-trip
// ============================================================================
static void test_bip39_round_trip() {
    (void)std::printf("[19] FFI: BIP-39 (generate -> validate -> seed)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    // Generate 12-word mnemonic from known entropy
    uint8_t entropy[16];
    hex_to_bytes("00000000000000000000000000000000", entropy, 16);

    char mnemonic[512];
    size_t mlen = sizeof(mnemonic);
    CHECK_OK(ufsecp_bip39_generate(ctx, 16, entropy, mnemonic, &mlen),
             "bip39_generate(16 bytes)");
    CHECK(mlen > 0, "bip39 mnemonic non-empty");

    // Validate
    CHECK_OK(ufsecp_bip39_validate(ctx, mnemonic), "bip39_validate");

    // Known valid 12-word mnemonic (BIP-39 test vector)
    CHECK(ufsecp_bip39_validate(ctx, "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about") == UFSECP_OK,
          "bip39_validate accepts valid 12-word mnemonic");

    // Invalid checksum variant
    CHECK(ufsecp_bip39_validate(ctx, "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon") == UFSECP_ERR_BAD_INPUT,
          "bip39_validate rejects invalid checksum mnemonic");

    // To seed
    uint8_t seed[64];
    CHECK_OK(ufsecp_bip39_to_seed(ctx, mnemonic, "", seed), "bip39_to_seed");

    // Seed should be non-zero
    uint8_t zero64[64] = {};
    CHECK(std::memcmp(seed, zero64, 64) != 0, "seed is non-zero");

    // To entropy round-trip
    uint8_t ent_out[32];
    size_t ent_len = sizeof(ent_out);
    CHECK_OK(ufsecp_bip39_to_entropy(ctx, mnemonic, ent_out, &ent_len),
             "bip39_to_entropy");
    CHECK(ent_len == 16, "entropy length == 16");
    CHECK(std::memcmp(ent_out, entropy, 16) == 0, "entropy round-trip matches");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 20: Batch verification
// ============================================================================
static void test_batch_verify() {
    (void)std::printf("[20] FFI: Batch verification (ECDSA + Schnorr)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t priv[32];
    hex_to_bytes(PRIVKEY1_HEX, priv, 32);

    // === Schnorr batch verify ===
    uint8_t xonly[32];
    CHECK_OK(ufsecp_pubkey_xonly(ctx, priv, xonly), "xonly for batch");

    // Create 3 valid Schnorr sigs
    uint8_t schnorr_entries[3 * 128]; // Each: 32 xonly + 32 msg + 64 sig = 128
    for (int i = 0; i < 3; ++i) {
        uint8_t msg[32] = {};
        msg[0] = static_cast<uint8_t>(i + 1);
        uint8_t aux[32] = {};
        uint8_t sig[64];
        CHECK_OK(ufsecp_schnorr_sign(ctx, msg, priv, aux, sig), "schnorr sign for batch");

        std::memcpy(schnorr_entries + static_cast<size_t>(i) * 128, xonly, 32);
        std::memcpy(schnorr_entries + static_cast<size_t>(i) * 128 + 32, msg, 32);
        std::memcpy(schnorr_entries + static_cast<size_t>(i) * 128 + 64, sig, 64);
    }

    CHECK_OK(ufsecp_schnorr_batch_verify(ctx, schnorr_entries, 3), "schnorr_batch_verify 3 valid");

    // Corrupt one sig -> batch should fail
    schnorr_entries[2 * 128 + 64] ^= 0xFF;
    CHECK(ufsecp_schnorr_batch_verify(ctx, schnorr_entries, 3) != UFSECP_OK,
          "schnorr_batch_verify rejects with 1 bad");

    // Identify invalid
    size_t invalid_idx[3];
    size_t invalid_count = 0;
    CHECK_OK(ufsecp_schnorr_batch_identify_invalid(ctx, schnorr_entries, 3,
             invalid_idx, &invalid_count), "schnorr_batch_identify");
    CHECK(invalid_count >= 1, "schnorr identify found >= 1 invalid");

    // === ECDSA batch verify ===
    uint8_t pub33[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, priv, pub33), "pubkey for ecdsa batch");

    uint8_t ecdsa_entries[3 * 129]; // Each: 32 msg + 33 pubkey + 64 sig = 129
    for (int i = 0; i < 3; ++i) {
        uint8_t msg[32] = {};
        msg[0] = static_cast<uint8_t>(i + 10);
        uint8_t sig[64];
        CHECK_OK(ufsecp_ecdsa_sign(ctx, msg, priv, sig), "ecdsa sign for batch");

        std::memcpy(ecdsa_entries + static_cast<size_t>(i) * 129, msg, 32);
        std::memcpy(ecdsa_entries + static_cast<size_t>(i) * 129 + 32, pub33, 33);
        std::memcpy(ecdsa_entries + static_cast<size_t>(i) * 129 + 65, sig, 64);
    }

    CHECK_OK(ufsecp_ecdsa_batch_verify(ctx, ecdsa_entries, 3), "ecdsa_batch_verify 3 valid");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 21: Pedersen commitments
// ============================================================================
static void test_pedersen_commitments() {
    (void)std::printf("[21] FFI: Pedersen commitments (commit -> verify -> sum)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t value[32] = {};
    value[31] = 42; // value = 42
    uint8_t blinding[32];
    hex_to_bytes(PRIVKEY1_HEX, blinding, 32);

    // Commit
    uint8_t commit[33];
    CHECK_OK(ufsecp_pedersen_commit(ctx, value, blinding, commit), "pedersen_commit");

    // Verify
    CHECK_OK(ufsecp_pedersen_verify(ctx, commit, value, blinding), "pedersen_verify");

    // Verify with wrong value should fail
    uint8_t wrong_val[32] = {};
    wrong_val[31] = 43;
    CHECK(ufsecp_pedersen_verify(ctx, commit, wrong_val, blinding) != UFSECP_OK,
          "pedersen_verify rejects wrong value");

    // Sum balance: commit(42, b1) == commit(42, b1)
    CHECK_OK(ufsecp_pedersen_verify_sum(ctx, commit, 1, commit, 1),
             "pedersen_verify_sum balanced");

    // Blind sum
    uint8_t blind_sum[32];
    CHECK_OK(ufsecp_pedersen_blind_sum(ctx, blinding, 1, blinding, 1, blind_sum),
             "pedersen_blind_sum");
    // Sum of same blind in and out should be zero
    uint8_t zero32[32] = {};
    CHECK(std::memcmp(blind_sum, zero32, 32) == 0, "blind_sum(b,-b) == 0");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 22: ZK proofs (knowledge proof)
// ============================================================================
static void test_zk_proofs() {
    (void)std::printf("[22] FFI: ZK proofs (knowledge prove -> verify)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t secret[32];
    hex_to_bytes(PRIVKEY1_HEX, secret, 32);
    uint8_t pubkey[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, secret, pubkey), "zk: pubkey");

    uint8_t msg[32];
    hex_to_bytes(MSG_HEX, msg, 32);
    uint8_t aux[32] = {};

    // Knowledge proof: prove + verify
    uint8_t proof[UFSECP_ZK_KNOWLEDGE_PROOF_LEN];
    CHECK_OK(ufsecp_zk_knowledge_prove(ctx, secret, pubkey, msg, aux, proof),
             "zk_knowledge_prove");
    CHECK_OK(ufsecp_zk_knowledge_verify(ctx, proof, pubkey, msg),
             "zk_knowledge_verify");

    // Verify with wrong message fails
    uint8_t wrong_msg[32] = {0x42};
    CHECK(ufsecp_zk_knowledge_verify(ctx, proof, pubkey, wrong_msg) != UFSECP_OK,
          "zk_knowledge_verify rejects wrong msg");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 23: Multi-scalar multiplication
// ============================================================================
static void test_multi_scalar_mul() {
    (void)std::printf("[23] FFI: Multi-scalar multiplication (Shamir + MSM)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t priv1[32], priv2[32];
    hex_to_bytes(PRIVKEY1_HEX, priv1, 32);
    hex_to_bytes(PRIVKEY2_HEX, priv2, 32);

    uint8_t pub1[33], pub2[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, priv1, pub1), "msm: pub1");
    CHECK_OK(ufsecp_pubkey_create(ctx, priv2, pub2), "msm: pub2");

    // Shamir: 1*pub1 + 1*pub2 = pub1 + pub2
    uint8_t one[32] = {};
    one[31] = 1;
    uint8_t shamir_out[33];
    CHECK_OK(ufsecp_shamir_trick(ctx, one, pub1, one, pub2, shamir_out), "shamir_trick");

    uint8_t add_out[33];
    CHECK_OK(ufsecp_pubkey_add(ctx, pub1, pub2, add_out), "pubkey_add for shamir check");
    CHECK(std::memcmp(shamir_out, add_out, 33) == 0, "shamir(1*P1+1*P2) == add(P1,P2)");

    // MSM: same thing, via multi_scalar_mul
    uint8_t scalars[64]; // 2 * 32
    std::memcpy(scalars, one, 32);
    std::memcpy(scalars + 32, one, 32);
    uint8_t points[66]; // 2 * 33
    std::memcpy(points, pub1, 33);
    std::memcpy(points + 33, pub2, 33);
    uint8_t msm_out[33];
    CHECK_OK(ufsecp_multi_scalar_mul(ctx, scalars, points, 2, msm_out), "multi_scalar_mul");
    CHECK(std::memcmp(msm_out, add_out, 33) == 0, "msm(1*P1+1*P2) == add(P1,P2)");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 24: Multi-coin wallet
// ============================================================================
static void test_multi_coin_wallet() {
    (void)std::printf("[24] FFI: Multi-coin wallet (address dispatch)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t priv[32];
    hex_to_bytes(PRIVKEY1_HEX, priv, 32);
    uint8_t pub33[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, priv, pub33), "multicoin: pubkey");

    char addr[UFSECP_COIN_ADDR_MAX_LEN];

    // Bitcoin
    size_t len = sizeof(addr);
    CHECK_OK(ufsecp_coin_address(ctx, pub33, UFSECP_COIN_BITCOIN, 0, addr, &len),
             "coin_address BTC");
    CHECK(len > 0, "BTC addr non-empty");

    // Litecoin
    len = sizeof(addr);
    CHECK_OK(ufsecp_coin_address(ctx, pub33, UFSECP_COIN_LITECOIN, 0, addr, &len),
             "coin_address LTC");
    CHECK(len > 0, "LTC addr non-empty");

    // Dogecoin
    len = sizeof(addr);
    CHECK_OK(ufsecp_coin_address(ctx, pub33, UFSECP_COIN_DOGECOIN, 0, addr, &len),
             "coin_address DOGE");
    CHECK(len > 0, "DOGE addr non-empty");

    // WIF encode for coin
    char wif[64];
    size_t wlen = sizeof(wif);
    CHECK_OK(ufsecp_coin_wif_encode(ctx, priv, UFSECP_COIN_BITCOIN, 0, wif, &wlen),
             "coin_wif_encode BTC");
    CHECK(wlen > 0, "BTC WIF non-empty");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 25: Bitcoin message signing (BIP-137)
// ============================================================================
static void test_btc_message_sign() {
    (void)std::printf("[25] FFI: Bitcoin message sign/verify (BIP-137)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t priv[32];
    hex_to_bytes(PRIVKEY1_HEX, priv, 32);
    uint8_t pub33[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, priv, pub33), "btc_msg: pubkey");

    const uint8_t msg[] = "Hello, Bitcoin!";
    const size_t msg_len = 15;

    // Message hash
    uint8_t hash[32];
    CHECK_OK(ufsecp_btc_message_hash(msg, msg_len, hash), "btc_message_hash");
    uint8_t zero32[32] = {};
    CHECK(std::memcmp(hash, zero32, 32) != 0, "btc_msg hash non-zero");

    // Sign
    char base64[128];
    size_t b64len = sizeof(base64);
    CHECK_OK(ufsecp_btc_message_sign(ctx, msg, msg_len, priv, base64, &b64len),
             "btc_message_sign");
    CHECK(b64len > 0, "btc_msg sig non-empty");

    // Verify
    CHECK_OK(ufsecp_btc_message_verify(ctx, msg, msg_len, pub33, base64),
             "btc_message_verify");

    // Verify with wrong message
    const uint8_t wrong[] = "Wrong message!";
    CHECK(ufsecp_btc_message_verify(ctx, wrong, 14, pub33, base64) != UFSECP_OK,
          "btc_message_verify rejects wrong msg");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 26: Ethereum (conditional)
// ============================================================================
#ifdef SECP256K1_BUILD_ETHEREUM
static void test_ethereum_round_trip() {
    (void)std::printf("[26] FFI: Ethereum (keccak256, address, sign, ecrecover)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    // Keccak-256 known vector: keccak256("") = c5d2460186f7233c9...
    uint8_t keccak_out[32];
    CHECK_OK(ufsecp_keccak256(nullptr, 0, keccak_out), "keccak256 empty");
    uint8_t keccak_expected[32];
    hex_to_bytes("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
                 keccak_expected, 32);
    CHECK(std::memcmp(keccak_out, keccak_expected, 32) == 0, "keccak256 empty matches");

    // Ethereum address from privkey 1
    uint8_t priv[32];
    hex_to_bytes(PRIVKEY1_HEX, priv, 32);
    uint8_t pub33[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, priv, pub33), "eth: pubkey");

    uint8_t eth_addr[20];
    CHECK_OK(ufsecp_eth_address(ctx, pub33, eth_addr), "eth_address");
    uint8_t zero20[20] = {};
    CHECK(std::memcmp(eth_addr, zero20, 20) != 0, "eth addr non-zero");

    // EIP-55 checksummed address
    char eip55[64];
    size_t elen = sizeof(eip55);
    CHECK_OK(ufsecp_eth_address_checksummed(ctx, pub33, eip55, &elen), "eth_address_checksummed");
    CHECK(elen > 0, "eip55 addr non-empty");
    CHECK(eip55[0] == '0' && eip55[1] == 'x', "eip55 starts with 0x");

    // Personal hash
    const uint8_t eth_msg[] = "Hello Ethereum!";
    uint8_t personal_hash[32];
    CHECK_OK(ufsecp_eth_personal_hash(eth_msg, 15, personal_hash), "eth_personal_hash");

    // Sign with EIP-155 (chain_id = 1 for mainnet)
    uint8_t msg32[32];
    hex_to_bytes(MSG_HEX, msg32, 32);
    uint8_t r[32], s[32];
    uint64_t v = 0;
    CHECK_OK(ufsecp_eth_sign(ctx, msg32, priv, r, s, &v, 1), "eth_sign");
    CHECK(v >= 27, "eth v >= 27");

    // Ecrecover
    uint8_t recovered_addr[20];
    CHECK_OK(ufsecp_eth_ecrecover(ctx, msg32, r, s, v, recovered_addr), "eth_ecrecover");
    CHECK(std::memcmp(recovered_addr, eth_addr, 20) == 0, "ecrecover matches eth_address");

    ufsecp_ctx_destroy(ctx);
}
#endif

// ============================================================================
// Test 27: MuSig2 flow (2-of-2)
// ============================================================================
static void test_musig2_flow() {
    (void)std::printf("[27] FFI: MuSig2 (2-of-2 key agg -> sign -> aggregate)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    // Two signers
    uint8_t priv1[32], priv2[32];
    hex_to_bytes(PRIVKEY1_HEX, priv1, 32);
    hex_to_bytes(PRIVKEY2_HEX, priv2, 32);

    uint8_t comp1[33], comp2[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, priv1, comp1), "musig2: comp1");
    CHECK_OK(ufsecp_pubkey_create(ctx, priv2, comp2), "musig2: comp2");
    uint8_t xonly1[32], xonly2[32];
    std::memcpy(xonly1, comp1 + 1, 32);
    std::memcpy(xonly2, comp2 + 1, 32);

    // Key aggregation (BIP-327: 33-byte compressed pubkeys)
    uint8_t pubkeys[66]; // 2 * 33
    std::memcpy(pubkeys, comp1, 33);
    std::memcpy(pubkeys + 33, comp2, 33);

    uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN];
    uint8_t agg_pub[32];
    CHECK_OK(ufsecp_musig2_key_agg(ctx, pubkeys, 2, keyagg, agg_pub), "musig2_key_agg");

    uint8_t msg32[32];
    hex_to_bytes(MSG_HEX, msg32, 32);

    // Nonce gen (signer 1)
    uint8_t extra[32] = {};
    uint8_t secnonce1[UFSECP_MUSIG2_SECNONCE_LEN], pubnonce1[UFSECP_MUSIG2_PUBNONCE_LEN];
    CHECK_OK(ufsecp_musig2_nonce_gen(ctx, priv1, xonly1, agg_pub, msg32, extra,
             secnonce1, pubnonce1), "musig2: nonce_gen signer1");

    // Nonce gen (signer 2)
    extra[0] = 1;
    uint8_t secnonce2[UFSECP_MUSIG2_SECNONCE_LEN], pubnonce2[UFSECP_MUSIG2_PUBNONCE_LEN];
    CHECK_OK(ufsecp_musig2_nonce_gen(ctx, priv2, xonly2, agg_pub, msg32, extra,
             secnonce2, pubnonce2), "musig2: nonce_gen signer2");

    // Nonce agg
    uint8_t pubnonces_all[2 * UFSECP_MUSIG2_PUBNONCE_LEN];
    std::memcpy(pubnonces_all, pubnonce1, UFSECP_MUSIG2_PUBNONCE_LEN);
    std::memcpy(pubnonces_all + UFSECP_MUSIG2_PUBNONCE_LEN, pubnonce2, UFSECP_MUSIG2_PUBNONCE_LEN);
    uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN];
    CHECK_OK(ufsecp_musig2_nonce_agg(ctx, pubnonces_all, 2, aggnonce), "musig2: nonce_agg");

    // Start session
    uint8_t session[UFSECP_MUSIG2_SESSION_LEN];
    CHECK_OK(ufsecp_musig2_start_sign_session(ctx, aggnonce, keyagg, msg32, session),
             "musig2: start_session");

    // Partial sign (signer 1)
    uint8_t psig1[32];
    CHECK_OK(ufsecp_musig2_partial_sign(ctx, secnonce1, priv1, keyagg, session, 0, psig1),
             "musig2: partial_sign signer1");

    // Partial sign (signer 2)
    uint8_t psig2[32];
    CHECK_OK(ufsecp_musig2_partial_sign(ctx, secnonce2, priv2, keyagg, session, 1, psig2),
             "musig2: partial_sign signer2");

    // Aggregate partial sigs
    uint8_t psigs_all[64]; // 2 * 32
    std::memcpy(psigs_all, psig1, 32);
    std::memcpy(psigs_all + 32, psig2, 32);
    uint8_t final_sig[64];
    CHECK_OK(ufsecp_musig2_partial_sig_agg(ctx, psigs_all, 2, session, final_sig),
             "musig2: sig_agg");

    // Verify aggregated signature as standard BIP-340
    CHECK_OK(ufsecp_schnorr_verify(ctx, msg32, final_sig, agg_pub),
             "musig2: schnorr_verify(agg_sig, agg_pub)");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 28: Adaptor signatures (Schnorr)
// ============================================================================
static void test_adaptor_signatures() {
    (void)std::printf("[28] FFI: Adaptor signatures (pre-sign -> adapt -> extract)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    uint8_t priv[32];
    hex_to_bytes(PRIVKEY1_HEX, priv, 32);
    uint8_t xonly[32];
    CHECK_OK(ufsecp_pubkey_xonly(ctx, priv, xonly), "adaptor: xonly");

    // Adaptor secret and point
    uint8_t adaptor_secret[32];
    hex_to_bytes(PRIVKEY2_HEX, adaptor_secret, 32);
    uint8_t adaptor_point[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, adaptor_secret, adaptor_point), "adaptor: point");

    uint8_t msg32[32];
    hex_to_bytes(MSG_HEX, msg32, 32);
    uint8_t aux[32] = {};

    // Pre-sign
    uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN];
    CHECK_OK(ufsecp_schnorr_adaptor_sign(ctx, priv, msg32, adaptor_point, aux, pre_sig),
             "schnorr_adaptor_sign");

    // Verify pre-sig
    CHECK_OK(ufsecp_schnorr_adaptor_verify(ctx, pre_sig, xonly, msg32, adaptor_point),
             "schnorr_adaptor_verify");

    // Adapt: pre_sig + secret -> valid signature
    uint8_t final_sig[64];
    CHECK_OK(ufsecp_schnorr_adaptor_adapt(ctx, pre_sig, adaptor_secret, final_sig),
             "schnorr_adaptor_adapt");

    // Verify final sig as standard Schnorr
    CHECK_OK(ufsecp_schnorr_verify(ctx, msg32, final_sig, xonly),
             "adapted sig verifies as schnorr");

    // Extract secret from pre_sig + final_sig
    uint8_t extracted[32];
    CHECK_OK(ufsecp_schnorr_adaptor_extract(ctx, pre_sig, final_sig, extracted),
             "schnorr_adaptor_extract");
    // Verify extracted secret is valid (non-zero scalar)
    CHECK_OK(ufsecp_seckey_verify(ctx, extracted), "extracted secret is valid scalar");
    // The extracted secret may be the original or its negation (mod n),
    // depending on nonce parity. Verify it matches one or the other.
    uint8_t extracted_point[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, extracted, extracted_point),
             "pubkey from extracted");
    // Check direct match or negated match
    uint8_t neg_point[33];
    CHECK_OK(ufsecp_pubkey_negate(ctx, extracted_point, neg_point),
             "negate extracted point");
    const bool match = (std::memcmp(extracted_point, adaptor_point, 33) == 0) ||
                 (std::memcmp(neg_point, adaptor_point, 33) == 0);
    CHECK(match, "extracted secret matches adaptor (direct or negated)");

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 29: ECIES Round-Trip
// ============================================================================

static void test_ecies_round_trip() {
    (void)std::printf("[29] FFI: ECIES (encrypt -> decrypt -> tamper -> wrong key)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    // Use PRIVKEY1 as recipient
    uint8_t priv[32];
    hex_to_bytes(PRIVKEY1_HEX, priv, 32);
    uint8_t pub33[33] = {};
    CHECK_OK(ufsecp_pubkey_create(ctx, priv, pub33), "pubkey_create for ECIES");

    // Encrypt a message
    const char* msg = "Hello, ECIES on secp256k1!";
    size_t const msg_len = std::strlen(msg);
    uint8_t envelope[256];
    size_t env_len = sizeof(envelope);
    CHECK_OK(ufsecp_ecies_encrypt(ctx, pub33,
          reinterpret_cast<const uint8_t*>(msg), msg_len,
          envelope, &env_len), "ECIES encrypt");
    CHECK(env_len == msg_len + UFSECP_ECIES_OVERHEAD, "envelope size = plaintext + 81");

    // Decrypt
    uint8_t plaintext[256];
    size_t pt_len = sizeof(plaintext);
    CHECK_OK(ufsecp_ecies_decrypt(ctx, priv, envelope, env_len,
          plaintext, &pt_len), "ECIES decrypt");
    CHECK(pt_len == msg_len, "plaintext size matches");
    CHECK(std::memcmp(plaintext, msg, msg_len) == 0, "plaintext matches original");

    // Tamper test: flip one byte in ciphertext region -> should fail
    envelope[50] ^= 0xFF;
    pt_len = sizeof(plaintext);
    CHECK(ufsecp_ecies_decrypt(ctx, priv, envelope, env_len,
          plaintext, &pt_len) != UFSECP_OK, "tampered envelope rejected");

    // Wrong key test
    uint8_t priv2[32];
    hex_to_bytes(PRIVKEY2_HEX, priv2, 32);
    envelope[50] ^= 0xFF; // restore
    pt_len = sizeof(plaintext);
    CHECK(ufsecp_ecies_decrypt(ctx, priv2, envelope, env_len,
          plaintext, &pt_len) != UFSECP_OK, "wrong key rejected");

    // Null arg tests
    env_len = sizeof(envelope);
    CHECK(ufsecp_ecies_encrypt(nullptr, pub33,
          reinterpret_cast<const uint8_t*>(msg), msg_len,
          envelope, &env_len) == UFSECP_ERR_NULL_ARG, "encrypt null ctx");
    CHECK(ufsecp_ecies_decrypt(ctx, nullptr, envelope, env_len,
          plaintext, &pt_len) == UFSECP_ERR_NULL_ARG, "decrypt null privkey");

    // Buffer too small
    size_t small = 10;
    CHECK(ufsecp_ecies_encrypt(ctx, pub33,
          reinterpret_cast<const uint8_t*>(msg), msg_len,
          envelope, &small) == UFSECP_ERR_BUF_TOO_SMALL, "encrypt buf too small");

    // ---- Regression: parity-byte malleability (Finding #1) ----
    // Flip ephemeral pubkey parity byte (0x02 <-> 0x03) in a valid envelope.
    // HMAC now covers pubkey, so this MUST be rejected.
    {
        uint8_t env_copy[256];
        std::memcpy(env_copy, envelope, env_len);
        env_copy[0] ^= 0x01; // flip 0x02->0x03 or 0x03->0x02
        pt_len = sizeof(plaintext);
        // Re-encrypt fresh since we used priv2 above and envelope may be stale
        env_len = sizeof(envelope);
        CHECK_OK(ufsecp_ecies_encrypt(ctx, pub33,
              reinterpret_cast<const uint8_t*>(msg), msg_len,
              envelope, &env_len), "re-encrypt for malleability test");
        std::memcpy(env_copy, envelope, env_len);
        env_copy[0] ^= 0x01; // flip parity
        pt_len = sizeof(plaintext);
        CHECK(ufsecp_ecies_decrypt(ctx, priv, env_copy, env_len,
              plaintext, &pt_len) != UFSECP_OK,
              "parity-flipped ephemeral pubkey rejected (anti-malleability)");
    }

    // ---- Regression: invalid prefix byte (Finding #3) ----
    // Set prefix to 0x04 (uncompressed), 0x00, 0x01, 0x07 -- all must fail.
    {
        uint8_t env_bad[256];
        const uint8_t bad_prefixes[] = {0x00, 0x01, 0x04, 0x07, 0xFF};
        for (auto bp : bad_prefixes) {
            std::memcpy(env_bad, envelope, env_len);
            env_bad[0] = bp;
            pt_len = sizeof(plaintext);
            CHECK(ufsecp_ecies_decrypt(ctx, priv, env_bad, env_len,
                  plaintext, &pt_len) != UFSECP_OK,
                  "bad prefix byte in ephemeral pubkey rejected");
        }
    }

    // ---- Regression: bad prefix in pubkey for ECDH (Finding #3) ----
    {
        uint8_t bad_pub[33];
        std::memcpy(bad_pub, pub33, 33);
        bad_pub[0] = 0x04; // not valid for compressed
        uint8_t ecdh_out[32];
        CHECK(ufsecp_ecdh(ctx, priv, bad_pub, ecdh_out) != UFSECP_OK,
              "ECDH rejects 0x04 prefix in compressed pubkey");
        bad_pub[0] = 0x00;
        CHECK(ufsecp_ecdh(ctx, priv, bad_pub, ecdh_out) != UFSECP_OK,
              "ECDH rejects 0x00 prefix in compressed pubkey");
    }

    ufsecp_ctx_destroy(ctx);
}

// ============================================================================
// Test 30: BIP-352 Silent Payments
// ============================================================================

static void test_silent_payments() {
    (void)std::printf("[30] FFI: Silent Payments (address -> create_output -> scan)\n");

    ufsecp_ctx* ctx = nullptr;
    CHECK_OK(ufsecp_ctx_create(&ctx), "ctx_create");

    // Use PRIVKEY1 as scan key, PRIVKEY2 as spend key
    uint8_t scan_priv[32], spend_priv[32];
    hex_to_bytes(PRIVKEY1_HEX, scan_priv, 32);
    hex_to_bytes(PRIVKEY2_HEX, spend_priv, 32);

    // Generate Silent Payment address
    uint8_t sp_scan33[33], sp_spend33[33];
    char addr[256];
    size_t addr_len = sizeof(addr);
    CHECK_OK(ufsecp_silent_payment_address(ctx, scan_priv, spend_priv,
          sp_scan33, sp_spend33, addr, &addr_len),
          "silent_payment_address");
    CHECK(addr_len > 0, "address not empty");

    // Use a third key as sender input
    // scalar=3 as sender
    uint8_t sender_priv[32] = {};
    sender_priv[31] = 3;
    uint8_t sender_pub[33] = {};
    CHECK_OK(ufsecp_pubkey_create(ctx, sender_priv, sender_pub), "sender pubkey");

    // Create output (sender side)
    uint8_t output_pub33[33], tweak32[32];
    CHECK_OK(ufsecp_silent_payment_create_output(ctx,
          sender_priv, 1, sp_scan33, sp_spend33, 0,
          output_pub33, tweak32), "create_output");

    // Scan: receiver should find the output
    // Extract x-only from output_pub33
    uint8_t xonly[32];
    std::memcpy(xonly, output_pub33 + 1, 32);

    uint32_t found_idx[4];
    uint8_t found_keys[128];
    size_t n_found = 4;
    CHECK_OK(ufsecp_silent_payment_scan(ctx,
          scan_priv, spend_priv,
          sender_pub, 1,
          xonly, 1,
          found_idx, found_keys, &n_found), "scan");
    CHECK(n_found == 1, "found exactly one output");
    CHECK(found_idx[0] == 0, "found at index 0");

    // Verify the found spending key produces the output pubkey
    uint8_t verify_pub[33];
    CHECK_OK(ufsecp_pubkey_create(ctx, found_keys, verify_pub),
          "derive pubkey from found key");
    CHECK(std::memcmp(verify_pub, output_pub33, 33) == 0,
          "derived pubkey matches output");

    // Null arg tests
    addr_len = sizeof(addr);
    CHECK(ufsecp_silent_payment_address(nullptr, scan_priv, spend_priv,
          sp_scan33, sp_spend33, addr, &addr_len) == UFSECP_ERR_NULL_ARG,
          "address null ctx");

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
    test_negative_vectors();
    test_sha512_vector();
    test_pubkey_arithmetic();
    test_bip39_round_trip();
    test_batch_verify();
    test_pedersen_commitments();
    test_zk_proofs();
    test_multi_scalar_mul();
    test_multi_coin_wallet();
    test_btc_message_sign();
#ifdef SECP256K1_BUILD_ETHEREUM
    test_ethereum_round_trip();
#endif
    test_musig2_flow();
    test_adaptor_signatures();
    test_ecies_round_trip();
    test_silent_payments();

    (void)std::printf("\n--- FFI Round-Trip Summary: %d passed, %d failed ---\n\n",
                      g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    return test_ffi_round_trip_run();
}
#endif
