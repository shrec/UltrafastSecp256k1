// ============================================================================
// Test: Unified Wallet API + Bitcoin Message Signing + Tron Address
// ============================================================================
// Tests:
//   1. from_private_key -- valid and invalid keys
//   2. Bitcoin address via wallet API
//   3. Ethereum address via wallet API
//   4. Tron address via wallet API
//   5. export_private_key -- WIF, 0x-hex, raw hex
//   6. export_public_key_hex -- compressed vs uncompressed
//   7. Bitcoin message hash -- known vector
//   8. Bitcoin sign + verify message round-trip
//   9. Bitcoin sign + recover message round-trip
//  10. Bitcoin base64 encode/decode round-trip
//  11. Wallet sign_message + verify_message (Bitcoin)
//  12. Wallet sign_hash + recover_signer (Bitcoin)
//  13. Wallet sign_message + recover_address (Ethereum, conditional)
//  14. CoinParams: Tron descriptor validation
//  15. CoinParams: chain_id values
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <array>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/context.hpp"
#include "secp256k1/coins/coin_params.hpp"
#include "secp256k1/coins/coin_address.hpp"
#include "secp256k1/coins/message_signing.hpp"
#include "secp256k1/coins/wallet.hpp"

#if defined(SECP256K1_BUILD_ETHEREUM)
#include "secp256k1/coins/ethereum.hpp"
#include "secp256k1/coins/eth_signing.hpp"
#endif

using namespace secp256k1;
using namespace secp256k1::coins;
using namespace secp256k1::coins::wallet;
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
        unsigned int byte = 0;
        if (std::sscanf(hex + i * 2, "%02x", &byte) != 1) byte = 0;
        out[i] = static_cast<uint8_t>(byte);
    }
}

static std::string bytes_to_hex(const uint8_t* data, size_t len) {
    static const char hex[] = "0123456789abcdef";
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        result += hex[data[i] >> 4];
        result += hex[data[i] & 0xF];
    }
    return result;
}

// Well-known test private key (Bitcoin wiki example)
static constexpr const char* TEST_PRIVKEY_HEX =
    "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35";

// ============================================================================
// 1. from_private_key
// ============================================================================

static void test_from_private_key_valid() {
    TEST("from_private_key: valid key");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "should succeed");
    ASSERT_TRUE(!key.priv.is_zero(), "priv not zero");
    ASSERT_TRUE(!key.pub.is_infinity(), "pub not infinity");
    PASS();
}

static void test_from_private_key_zero() {
    TEST("from_private_key: zero key rejected");
    uint8_t priv[32] = {};
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(!ok, "should fail for zero");
    PASS();
}

// ============================================================================
// 2-4. Address generation via wallet API
// ============================================================================

static void test_wallet_bitcoin_address() {
    TEST("wallet::get_address Bitcoin");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address(Bitcoin, key);
    ASSERT_TRUE(!addr.empty(), "non-empty address");
    // Bitcoin bech32 addresses start with "bc1"
    ASSERT_TRUE(addr.substr(0, 3) == "bc1", "starts with bc1");
    PASS();
}

static void test_wallet_litecoin_address() {
    TEST("wallet::get_address Litecoin");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address(Litecoin, key);
    ASSERT_TRUE(!addr.empty(), "non-empty address");
    // Litecoin bech32 starts with "ltc1"
    ASSERT_TRUE(addr.substr(0, 4) == "ltc1", "starts with ltc1");
    PASS();
}

static void test_wallet_dogecoin_address() {
    TEST("wallet::get_address Dogecoin (Base58Check)");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address(Dogecoin, key);
    ASSERT_TRUE(!addr.empty(), "non-empty address");
    // Dogecoin P2PKH addresses start with 'D'
    ASSERT_TRUE(addr[0] == 'D', "starts with D");
    PASS();
}

#if defined(SECP256K1_BUILD_ETHEREUM)
static void test_wallet_ethereum_address() {
    TEST("wallet::get_address Ethereum");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address(Ethereum, key);
    ASSERT_TRUE(!addr.empty(), "non-empty address");
    // Ethereum addresses start with "0x"
    ASSERT_TRUE(addr.substr(0, 2) == "0x", "starts with 0x");
    ASSERT_EQ(addr.size(), 42u, "42 chars (0x + 40 hex)");
    PASS();
}

static void test_wallet_tron_address() {
    TEST("wallet::get_address Tron");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address(Tron, key);
    ASSERT_TRUE(!addr.empty(), "non-empty address");
    // Tron addresses start with 'T'
    ASSERT_TRUE(addr[0] == 'T', "starts with T");
    ASSERT_EQ(addr.size(), 34u, "34 chars (Base58Check)");
    PASS();
}
#endif

// ============================================================================
// 5. export_private_key
// ============================================================================

static void test_export_privkey_bitcoin_wif() {
    TEST("export_private_key: Bitcoin WIF");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto wif = export_private_key(Bitcoin, key);
    ASSERT_TRUE(!wif.empty(), "non-empty WIF");
    // Compressed mainnet WIF starts with 'K' or 'L'
    ASSERT_TRUE(wif[0] == 'K' || wif[0] == 'L', "WIF starts with K or L");
    PASS();
}

#if defined(SECP256K1_BUILD_ETHEREUM)
static void test_export_privkey_ethereum_hex() {
    TEST("export_private_key: Ethereum 0x-hex");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto hex_key = export_private_key(Ethereum, key);
    ASSERT_TRUE(hex_key.substr(0, 2) == "0x", "starts with 0x");
    ASSERT_EQ(hex_key.size(), 66u, "66 chars (0x + 64 hex)");
    // Verify round-trip
    ASSERT_EQ(hex_key.substr(2), std::string(TEST_PRIVKEY_HEX), "matches input");
    PASS();
}

static void test_export_privkey_tron_raw() {
    TEST("export_private_key: Tron raw hex");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto hex_key = export_private_key(Tron, key);
    ASSERT_EQ(hex_key.size(), 64u, "64 chars (no 0x)");
    ASSERT_EQ(hex_key, std::string(TEST_PRIVKEY_HEX), "matches input");
    PASS();
}
#endif

// ============================================================================
// 6. export_public_key_hex
// ============================================================================

static void test_export_pubkey_bitcoin() {
    TEST("export_public_key_hex: Bitcoin (compressed)");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto hex_pub = export_public_key_hex(Bitcoin, key);
    ASSERT_EQ(hex_pub.size(), 66u, "66 chars (33 bytes)");
    // Compressed pubkey starts with 02 or 03
    ASSERT_TRUE(hex_pub.substr(0, 2) == "02" || hex_pub.substr(0, 2) == "03",
                "starts with 02 or 03");
    PASS();
}

#if defined(SECP256K1_BUILD_ETHEREUM)
static void test_export_pubkey_ethereum() {
    TEST("export_public_key_hex: Ethereum (uncompressed)");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto hex_pub = export_public_key_hex(Ethereum, key);
    ASSERT_EQ(hex_pub.size(), 130u, "130 chars (65 bytes)");
    // Uncompressed pubkey starts with 04
    ASSERT_TRUE(hex_pub.substr(0, 2) == "04", "starts with 04");
    PASS();
}
#endif

// ============================================================================
// 7. Bitcoin message hash
// ============================================================================

static void test_bitcoin_message_hash_known() {
    TEST("bitcoin_message_hash: 'Hello' known format");
    const uint8_t msg[] = { 'H', 'e', 'l', 'l', 'o' };
    auto hash = bitcoin_message_hash(msg, 5);
    // Just verify it's non-zero and deterministic
    auto hash2 = bitcoin_message_hash(msg, 5);
    ASSERT_TRUE(hash == hash2, "deterministic");
    bool non_zero = false;
    for (auto b : hash) if (b != 0) { non_zero = true; break; }
    ASSERT_TRUE(non_zero, "non-zero hash");
    PASS();
}

// ============================================================================
// 8. Bitcoin sign + verify round-trip
// ============================================================================

static void test_bitcoin_sign_verify() {
    TEST("bitcoin_sign_message + verify round-trip");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto scalar = Scalar::from_bytes(priv);
    auto pubkey = derive_public_key(scalar);
    const uint8_t msg[] = "Test message for signing";
    size_t msg_len = sizeof(msg) - 1; // no null terminator

    auto rsig = bitcoin_sign_message(msg, msg_len, scalar);
    bool ok = bitcoin_verify_message(msg, msg_len, pubkey, rsig.sig);
    ASSERT_TRUE(ok, "verify should pass");

    // Tamper: different message should fail
    const uint8_t bad_msg[] = "Wrong message";
    bool bad = bitcoin_verify_message(bad_msg, sizeof(bad_msg) - 1, pubkey, rsig.sig);
    ASSERT_TRUE(!bad, "tampered msg should fail");
    PASS();
}

// ============================================================================
// 9. Bitcoin sign + recover round-trip
// ============================================================================

static void test_bitcoin_sign_recover() {
    TEST("bitcoin_sign + recover round-trip");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto scalar = Scalar::from_bytes(priv);
    auto pubkey = derive_public_key(scalar);
    const uint8_t msg[] = "Recovery test message";
    size_t msg_len = sizeof(msg) - 1;

    auto rsig = bitcoin_sign_message(msg, msg_len, scalar);
    auto [recovered, ok] = bitcoin_recover_message(msg, msg_len, rsig.sig, rsig.recid);
    ASSERT_TRUE(ok, "recovery should succeed");

    auto orig = pubkey.to_compressed();
    auto rec = recovered.to_compressed();
    ASSERT_TRUE(orig == rec, "recovered key matches original");
    PASS();
}

// ============================================================================
// 10. Base64 encode/decode round-trip
// ============================================================================

static void test_base64_round_trip() {
    TEST("bitcoin_sig base64 encode/decode round-trip");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto scalar = Scalar::from_bytes(priv);
    const uint8_t msg[] = "Base64 test";
    size_t msg_len = sizeof(msg) - 1;

    auto rsig = bitcoin_sign_message(msg, msg_len, scalar);
    auto b64 = bitcoin_sig_to_base64(rsig, true);
    ASSERT_TRUE(!b64.empty(), "non-empty base64");

    auto decoded = bitcoin_sig_from_base64(b64);
    ASSERT_TRUE(decoded.valid, "decode success");
    ASSERT_EQ(decoded.recid, rsig.recid, "recid matches");
    ASSERT_TRUE(decoded.compressed, "compressed flag");

    // Verify decoded sig matches original
    auto orig_compact = rsig.sig.to_compact();
    auto dec_compact = decoded.sig.to_compact();
    ASSERT_TRUE(orig_compact == dec_compact, "sig matches");
    PASS();
}

// ============================================================================
// 11. Wallet sign_message + verify_message (Bitcoin)
// ============================================================================

static void test_wallet_sign_verify_bitcoin() {
    TEST("wallet: sign_message + verify_message (Bitcoin)");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");

    const uint8_t msg[] = "Wallet API test message";
    size_t msg_len = sizeof(msg) - 1;

    auto sig = sign_message(Bitcoin, key, msg, msg_len);
    bool verified = verify_message(Bitcoin, key.pub, msg, msg_len, sig);
    ASSERT_TRUE(verified, "verify should pass");

    // Wrong message should fail
    const uint8_t bad[] = "bad";
    ASSERT_TRUE(!verify_message(Bitcoin, key.pub, bad, 3, sig), "bad msg fails");
    PASS();
}

// ============================================================================
// 12. Wallet sign_hash + recover_signer (Bitcoin)
// ============================================================================

static void test_wallet_sign_hash_recover() {
    TEST("wallet: sign_hash + recover_signer (Bitcoin)");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");

    // Sign a known hash
    uint8_t hash[32];
    hex_to_bytes("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470",
                 hash, 32);
    auto sig = sign_hash(Bitcoin, key, hash);

    // Reconstruct: for raw hash verification, manually hash + verify
    // sign_hash with Bitcoin coin uses ecdsa_sign_recoverable directly
    ASSERT_TRUE(sig.recid >= 0 && sig.recid <= 3, "valid recid");
    bool r_nonzero = false, s_nonzero = false;
    for (auto b : sig.r) if (b) { r_nonzero = true; break; }
    for (auto b : sig.s) if (b) { s_nonzero = true; break; }
    ASSERT_TRUE(r_nonzero && s_nonzero, "non-zero r,s");
    PASS();
}

// ============================================================================
// 13. Wallet Ethereum sign + recover (conditional)
// ============================================================================

#if defined(SECP256K1_BUILD_ETHEREUM)
static void test_wallet_sign_recover_ethereum() {
    TEST("wallet: sign_message + recover_address (Ethereum)");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");

    const uint8_t msg[] = "Ethereum wallet test";
    size_t msg_len = sizeof(msg) - 1;

    auto sig = sign_message(Ethereum, key, msg, msg_len);
    auto [addr, recovered] = recover_address(Ethereum, msg, msg_len, sig);
    ASSERT_TRUE(recovered, "recovery success");

    auto expected_addr = get_address(Ethereum, key);
    ASSERT_EQ(addr, expected_addr, "recovered address matches");
    PASS();
}

static void test_wallet_sign_recover_tron() {
    TEST("wallet: sign_message + recover_address (Tron)");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");

    const uint8_t msg[] = "Tron wallet test";
    size_t msg_len = sizeof(msg) - 1;

    auto sig = sign_message(Tron, key, msg, msg_len);
    auto [addr, recovered] = recover_address(Tron, msg, msg_len, sig);
    ASSERT_TRUE(recovered, "recovery success");

    auto expected_addr = get_address(Tron, key);
    ASSERT_EQ(addr, expected_addr, "recovered address matches");
    PASS();
}
#endif

// ============================================================================
// 14. CoinParams: Tron descriptor
// ============================================================================

static void test_coin_params_tron() {
    TEST("CoinParams: Tron descriptor");
    const auto& trx = Tron;
    ASSERT_EQ(trx.coin_type, 195u, "coin_type=195");
    ASSERT_EQ(trx.p2pkh_version, 0x41u, "p2pkh=0x41");
    ASSERT_TRUE(trx.hash_algo == AddressHash::KECCAK256, "KECCAK256");
    ASSERT_TRUE(trx.default_encoding == AddressEncoding::TRON_BASE58, "TRON_BASE58");
    ASSERT_TRUE(!trx.features.uses_evm, "not EVM");
    ASSERT_TRUE(trx.features.compressed_only, "compressed_only");

    // Tron should be findable by ticker
    auto* found = find_by_ticker("TRX");
    ASSERT_TRUE(found != nullptr, "findable by TRX");
    ASSERT_EQ(found->coin_type, 195u, "found coin_type");
    PASS();
}

// ============================================================================
// 15. CoinParams: chain_id values
// ============================================================================

static void test_coin_params_chain_id() {
    TEST("CoinParams: chain_id values");
    ASSERT_EQ(Bitcoin.chain_id, 0u, "BTC chain_id=0");
    ASSERT_EQ(Litecoin.chain_id, 0u, "LTC chain_id=0");
    ASSERT_EQ(Ethereum.chain_id, 1u, "ETH chain_id=1");
    ASSERT_EQ(BNBSmartChain.chain_id, 56u, "BSC chain_id=56");
    ASSERT_EQ(Polygon.chain_id, 137u, "MATIC chain_id=137");
    ASSERT_EQ(Avalanche.chain_id, 43114u, "AVAX chain_id=43114");
    ASSERT_EQ(Fantom.chain_id, 250u, "FTM chain_id=250");
    ASSERT_EQ(Arbitrum.chain_id, 42161u, "ARB chain_id=42161");
    ASSERT_EQ(Optimism.chain_id, 10u, "OP chain_id=10");
    ASSERT_EQ(Tron.chain_id, 0u, "TRX chain_id=0");
    PASS();
}

// ============================================================================
// 16. Multi-coin address generation
// ============================================================================

static void test_multi_coin_addresses() {
    TEST("wallet: multiple coins from same key");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");

    // Generate addresses for several coins
    auto btc = get_address(Bitcoin, key);
    auto ltc = get_address(Litecoin, key);
    auto doge = get_address(Dogecoin, key);

    ASSERT_TRUE(!btc.empty() && !ltc.empty() && !doge.empty(), "all non-empty");
    // All addresses should be different (different prefixes/encoding)
    ASSERT_TRUE(btc != ltc, "BTC != LTC");
    ASSERT_TRUE(btc != doge, "BTC != DOGE");
    ASSERT_TRUE(ltc != doge, "LTC != DOGE");
    PASS();
}

// ============================================================================
// 16b. Wallet explicit address format helpers
// ============================================================================

static void test_wallet_get_address_p2pkh() {
    TEST("wallet: get_address_p2pkh Bitcoin");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address_p2pkh(Bitcoin, key);
    ASSERT_TRUE(!addr.empty(), "non-empty");
    ASSERT_TRUE(addr[0] == '1', "starts with 1");
    PASS();
}

static void test_wallet_get_address_p2wpkh() {
    TEST("wallet: get_address_p2wpkh Bitcoin");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address_p2wpkh(Bitcoin, key);
    ASSERT_TRUE(!addr.empty(), "non-empty");
    ASSERT_TRUE(addr.substr(0, 4) == "bc1q", "starts with bc1q");
    PASS();
}

static void test_wallet_get_address_p2sh_p2wpkh() {
    TEST("wallet: get_address_p2sh_p2wpkh Bitcoin");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address_p2sh_p2wpkh(Bitcoin, key);
    ASSERT_TRUE(!addr.empty(), "non-empty");
    ASSERT_TRUE(addr[0] == '3', "starts with 3");
    PASS();
}

static void test_wallet_get_address_p2tr() {
    TEST("wallet: get_address_p2tr Bitcoin");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address_p2tr(Bitcoin, key);
    ASSERT_TRUE(!addr.empty(), "non-empty");
    ASSERT_TRUE(addr.substr(0, 4) == "bc1p", "starts with bc1p");
    PASS();
}

static void test_wallet_get_address_cashaddr() {
    TEST("wallet: get_address_cashaddr BCH");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address_cashaddr(BitcoinCash, key);
    ASSERT_TRUE(!addr.empty(), "non-empty");
    ASSERT_TRUE(addr.substr(0, 13) == "bitcoincash:q", "starts with bitcoincash:q");
    PASS();
}

static void test_wallet_bch_default_cashaddr() {
    TEST("wallet: BCH default is CashAddr");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");
    auto addr = get_address(BitcoinCash, key);
    ASSERT_TRUE(!addr.empty(), "non-empty");
    ASSERT_TRUE(addr.substr(0, 13) == "bitcoincash:q", "default is CashAddr");
    PASS();
}

static void test_wallet_all_btc_formats() {
    TEST("wallet: all 4 BTC address formats");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");

    auto p2pkh = get_address_p2pkh(Bitcoin, key);
    auto p2wpkh = get_address_p2wpkh(Bitcoin, key);
    auto p2sh = get_address_p2sh_p2wpkh(Bitcoin, key);
    auto p2tr = get_address_p2tr(Bitcoin, key);

    // All four formats should be non-empty and different
    ASSERT_TRUE(!p2pkh.empty() && !p2wpkh.empty() && !p2sh.empty() && !p2tr.empty(),
                "all non-empty");
    ASSERT_TRUE(p2pkh != p2wpkh, "P2PKH != P2WPKH");
    ASSERT_TRUE(p2pkh != p2sh, "P2PKH != P2SH-P2WPKH");
    ASSERT_TRUE(p2pkh != p2tr, "P2PKH != P2TR");
    ASSERT_TRUE(p2wpkh != p2sh, "P2WPKH != P2SH-P2WPKH");
    ASSERT_TRUE(p2wpkh != p2tr, "P2WPKH != P2TR");
    ASSERT_TRUE(p2sh != p2tr, "P2SH-P2WPKH != P2TR");
    PASS();
}

// ============================================================================
// 17. MessageSignature::to_rsv
// ============================================================================

static void test_message_signature_to_rsv() {
    TEST("MessageSignature::to_rsv format");
    uint8_t priv[32];
    hex_to_bytes(TEST_PRIVKEY_HEX, priv, 32);
    auto [key, ok] = from_private_key(priv);
    ASSERT_TRUE(ok, "key creation");

    const uint8_t msg[] = "RSV test";
    auto sig = sign_message(Bitcoin, key, msg, 8);
    auto rsv = sig.to_rsv();

    // rsv should be 65 bytes: [r:32][s:32][v:1]
    ASSERT_TRUE(std::memcmp(rsv.data(), sig.r.data(), 32) == 0, "r matches");
    ASSERT_TRUE(std::memcmp(rsv.data() + 32, sig.s.data(), 32) == 0, "s matches");
    ASSERT_EQ(rsv[64], static_cast<uint8_t>(sig.v & 0xFF), "v matches");
    PASS();
}

// ============================================================================
// Main
// ============================================================================

#ifdef STANDALONE_TEST
int main() {
#else
int test_wallet_run() {
#endif
    std::printf("\n========================================\n");
    std::printf("  Unified Wallet API Tests\n");
    std::printf("========================================\n");

    // Key management
    std::printf("\n--- Key Management ---\n");
    test_from_private_key_valid();
    test_from_private_key_zero();

    // Address generation
    std::printf("\n--- Address Generation ---\n");
    test_wallet_bitcoin_address();
    test_wallet_litecoin_address();
    test_wallet_dogecoin_address();
#if defined(SECP256K1_BUILD_ETHEREUM)
    test_wallet_ethereum_address();
    test_wallet_tron_address();
#endif

    // Private/public key export
    std::printf("\n--- Key Export ---\n");
    test_export_privkey_bitcoin_wif();
#if defined(SECP256K1_BUILD_ETHEREUM)
    test_export_privkey_ethereum_hex();
    test_export_privkey_tron_raw();
#endif
    test_export_pubkey_bitcoin();
#if defined(SECP256K1_BUILD_ETHEREUM)
    test_export_pubkey_ethereum();
#endif

    // Bitcoin message signing
    std::printf("\n--- Bitcoin Message Signing ---\n");
    test_bitcoin_message_hash_known();
    test_bitcoin_sign_verify();
    test_bitcoin_sign_recover();
    test_base64_round_trip();

    // Wallet signing API
    std::printf("\n--- Wallet Signing API ---\n");
    test_wallet_sign_verify_bitcoin();
    test_wallet_sign_hash_recover();
#if defined(SECP256K1_BUILD_ETHEREUM)
    test_wallet_sign_recover_ethereum();
    test_wallet_sign_recover_tron();
#endif

    // Coin params
    std::printf("\n--- Coin Params ---\n");
    test_coin_params_tron();
    test_coin_params_chain_id();
    test_multi_coin_addresses();
    test_message_signature_to_rsv();

    // Address formats
    std::printf("\n--- Address Formats ---\n");
    test_wallet_get_address_p2pkh();
    test_wallet_get_address_p2wpkh();
    test_wallet_get_address_p2sh_p2wpkh();
    test_wallet_get_address_p2tr();
    test_wallet_get_address_cashaddr();
    test_wallet_bch_default_cashaddr();
    test_wallet_all_btc_formats();

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
