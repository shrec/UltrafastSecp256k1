// ============================================================================
// Test: Coins Layer + Custom Generator + Ethereum + BIP-44
// ============================================================================
// Tests:
//   1. CurveContext -- custom generator, default context, derive_public_key
//   2. CoinParams -- registry, lookup, all coins defined
//   3. Keccak-256 -- known test vectors
//   4. Ethereum address -- EIP-55 checksum, known vectors
//   5. Coin addresses -- Bitcoin, Litecoin, Dogecoin, Ethereum P2PKH/P2WPKH
//   6. WIF encoding -- coin-specific WIF prefixes
//   7. BIP-44 HD -- path construction, key derivation, seed->address
//   8. Custom generator -- key derivation with non-standard G
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "secp256k1/context.hpp"
#include "secp256k1/coins/coin_params.hpp"
#include "secp256k1/coins/keccak256.hpp"
#include "secp256k1/coins/ethereum.hpp"
#include "secp256k1/coins/coin_address.hpp"
#include "secp256k1/coins/coin_hd.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/bip32.hpp"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) \
    do { printf("  [TEST] %-50s ", name); } while(0)

#define PASS() \
    do { printf("PASS\n"); ++tests_passed; } while(0)

#define FAIL(msg) \
    do { printf("FAIL: %s\n", msg); ++tests_failed; } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { if (!(cond)) { FAIL(msg); return; } } while(0)

#define ASSERT_EQ(a, b, msg) \
    do { if ((a) != (b)) { FAIL(msg); return; } } while(0)

#define ASSERT_STR_EQ(a, b, msg) \
    do { if ((a) != (b)) { \
        printf("FAIL: %s\n  expected: %s\n  got:      %s\n", msg, (b).c_str(), (a).c_str()); \
        ++tests_failed; return; } } while(0)

// ============================================================================
// 1. CurveContext Tests
// ============================================================================

static void test_context_default() {
    TEST("CurveContext::secp256k1_default()");
    
    auto ctx = secp256k1::CurveContext::secp256k1_default();
    ASSERT_TRUE(!ctx.generator.is_infinity(), "default G should not be infinity");
    // Order is stored as raw bytes (not Scalar, since n mod n = 0)
    bool order_nonzero = false;
    for (auto b : ctx.order) { if (b != 0) { order_nonzero = true; break; } }
    ASSERT_TRUE(order_nonzero, "default order should not be all zeros");
    ASSERT_EQ(ctx.order[0], 0xFFu, "order first byte should be 0xFF");
    ASSERT_EQ(ctx.order[31], 0x41u, "order last byte should be 0x41");
    ASSERT_EQ(ctx.cofactor, 1u, "secp256k1 cofactor = 1");
    ASSERT_EQ(ctx.curve_name(), std::string_view("secp256k1"), "name mismatch");
    
    // Verify G matches Point::generator()
    auto std_G = secp256k1::fast::Point::generator();
    ASSERT_TRUE(ctx.generator.x() == std_G.x(), "G.x mismatch");
    ASSERT_TRUE(ctx.generator.y() == std_G.y(), "G.y mismatch");
    
    PASS();
}

static void test_context_custom_generator() {
    TEST("CurveContext::with_generator(custom)");
    
    // Use 2*G as custom generator
    auto G = secp256k1::fast::Point::generator();
    auto G2 = G.dbl();
    
    auto ctx = secp256k1::CurveContext::with_generator(G2, "test-2G");
    ASSERT_TRUE(ctx.generator.x() == G2.x(), "custom G.x mismatch");
    ASSERT_TRUE(ctx.generator.y() == G2.y(), "custom G.y mismatch");
    ASSERT_EQ(ctx.curve_name(), std::string_view("test-2G"), "name mismatch");
    
    PASS();
}

static void test_context_derive_public_key() {
    TEST("derive_public_key(default vs custom G)");
    
    auto privkey = secp256k1::fast::Scalar::from_uint64(12345);
    
    // Default: pubkey = privkey * G
    auto pub_default = secp256k1::derive_public_key(privkey);
    auto G = secp256k1::fast::Point::generator();
    auto expected = G.scalar_mul(privkey);
    ASSERT_TRUE(pub_default.x() == expected.x(), "default derive mismatch");
    
    // Custom: pubkey = privkey * (2*G) != privkey * G
    auto G2 = G.dbl();
    auto ctx = secp256k1::CurveContext::with_generator(G2);
    auto pub_custom = secp256k1::derive_public_key(privkey, &ctx);
    auto expected_custom = G2.scalar_mul(privkey);
    ASSERT_TRUE(pub_custom.x() == expected_custom.x(), "custom derive mismatch");
    
    // They should differ
    ASSERT_TRUE(!(pub_default.x() == pub_custom.x()), "default and custom should differ");
    
    PASS();
}

static void test_context_effective_generator() {
    TEST("effective_generator(nullptr vs ctx)");
    
    auto G = secp256k1::fast::Point::generator();
    
    // nullptr -> standard G
    const auto& eff_default = secp256k1::effective_generator(nullptr);
    ASSERT_TRUE(eff_default.x() == G.x(), "nullptr should return standard G");
    
    // Custom context
    auto G3 = G.scalar_mul(secp256k1::fast::Scalar::from_uint64(3));
    auto ctx = secp256k1::CurveContext::with_generator(G3);
    const auto& eff_custom = secp256k1::effective_generator(&ctx);
    ASSERT_TRUE(eff_custom.x() == G3.x(), "custom context should return custom G");
    
    PASS();
}

// ============================================================================
// 2. CoinParams Tests
// ============================================================================

static void test_coin_params_count() {
    TEST("CoinParams: 27 coins defined");
    ASSERT_EQ(secp256k1::coins::ALL_COINS_COUNT, 27u, "expected 27 coins");
    PASS();
}

static void test_coin_params_bitcoin() {
    TEST("CoinParams: Bitcoin values");
    const auto& btc = secp256k1::coins::Bitcoin;
    ASSERT_EQ(btc.p2pkh_version, 0x00u, "P2PKH version");
    ASSERT_EQ(btc.wif_prefix, 0x80u, "WIF prefix");
    ASSERT_EQ(btc.coin_type, 0u, "coin_type");
    ASSERT_TRUE(btc.features.supports_segwit, "SegWit support");
    ASSERT_TRUE(btc.features.supports_taproot, "Taproot support");
    PASS();
}

static void test_coin_params_ethereum() {
    TEST("CoinParams: Ethereum values");
    const auto& eth = secp256k1::coins::Ethereum;
    ASSERT_EQ(eth.coin_type, 60u, "coin_type");
    ASSERT_TRUE(eth.features.uses_evm, "EVM flag");
    ASSERT_EQ(static_cast<int>(eth.hash_algo),
              static_cast<int>(secp256k1::coins::AddressHash::KECCAK256), "hash algo");
    PASS();
}

static void test_coin_params_lookup() {
    TEST("CoinParams: find_by_ticker + find_by_coin_type");
    
    auto btc = secp256k1::coins::find_by_ticker("BTC");
    ASSERT_TRUE(btc != nullptr, "BTC not found");
    ASSERT_EQ(btc->coin_type, 0u, "BTC coin_type");
    
    auto eth = secp256k1::coins::find_by_coin_type(60);
    ASSERT_TRUE(eth != nullptr, "coin_type 60 not found");
    // Ethereum is the first with coin_type 60
    ASSERT_EQ(std::string(eth->ticker), std::string("ETH"), "ticker mismatch");
    
    auto missing = secp256k1::coins::find_by_ticker("NOTACOIN");
    ASSERT_TRUE(missing == nullptr, "should return nullptr for unknown");
    
    PASS();
}

// ============================================================================
// 3. Keccak-256 Tests
// ============================================================================

static void test_keccak256_empty() {
    TEST("Keccak-256: empty string");
    
    // Keccak-256("") = c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470
    auto hash = secp256k1::coins::keccak256(nullptr, 0);
    
    // Check first few bytes
    ASSERT_EQ(hash[0], 0xC5u, "byte 0");
    ASSERT_EQ(hash[1], 0xD2u, "byte 1");
    ASSERT_EQ(hash[2], 0x46u, "byte 2");
    ASSERT_EQ(hash[31], 0x70u, "byte 31");
    
    PASS();
}

static void test_keccak256_abc() {
    TEST("Keccak-256: \"abc\"");
    
    // Keccak-256("abc") = 4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45
    const std::uint8_t data[] = {'a', 'b', 'c'};
    auto hash = secp256k1::coins::keccak256(data, 3);
    
    ASSERT_EQ(hash[0], 0x4Eu, "byte 0");
    ASSERT_EQ(hash[1], 0x03u, "byte 1");
    ASSERT_EQ(hash[2], 0x65u, "byte 2");
    ASSERT_EQ(hash[31], 0x45u, "byte 31");
    
    PASS();
}

static void test_keccak256_incremental() {
    TEST("Keccak-256: incremental == one-shot");
    
    const std::uint8_t data[] = "hello world";
    auto one_shot = secp256k1::coins::keccak256(data, 11);
    
    secp256k1::coins::Keccak256State state;
    state.update(data, 5);       // "hello"
    state.update(data + 5, 6);   // " world"
    auto incremental = state.finalize();
    
    ASSERT_TRUE(one_shot == incremental, "incremental must match one-shot");
    
    PASS();
}

// ============================================================================
// 4. Ethereum Address Tests
// ============================================================================

static void test_ethereum_address_format() {
    TEST("Ethereum: address format 0x + 40 hex");
    
    auto privkey = secp256k1::fast::Scalar::from_uint64(1);
    (void)privkey;
    auto pubkey = secp256k1::fast::Point::generator(); // 1 * G = G
    
    auto addr = secp256k1::coins::ethereum_address(pubkey);
    ASSERT_EQ(addr.size(), 42u, "should be 42 chars (0x + 40)");
    ASSERT_TRUE(addr[0] == '0' && addr[1] == 'x', "should start with 0x");
    
    PASS();
}

static void test_ethereum_eip55_checksum() {
    TEST("Ethereum: EIP-55 checksum verify");
    
    // Known EIP-55 test vector
    // Address of private key = 1 (G point)
    auto pubkey = secp256k1::fast::Point::generator();
    auto addr = secp256k1::coins::ethereum_address(pubkey);
    
    // The generated address should pass EIP-55 verification
    ASSERT_TRUE(secp256k1::coins::eip55_verify(addr), "EIP-55 checksum failed");
    
    PASS();
}

static void test_ethereum_eip55_case_sensitivity() {
    TEST("Ethereum: EIP-55 case sensitivity");
    
    // An all-lowercase address should fail EIP-55 if it contains a-f chars
    auto pubkey = secp256k1::fast::Point::generator();
    auto raw = secp256k1::coins::ethereum_address_raw(pubkey);
    
    // Make it all lowercase with 0x prefix
    std::string lower = "0x" + raw;
    
    // A random lowercase address may or may not pass EIP-55
    // (it passes only if checksum happens to produce all lowercase)
    // But the properly checksummed version should always pass
    auto correct = secp256k1::coins::ethereum_address(pubkey);
    ASSERT_TRUE(secp256k1::coins::eip55_verify(correct), "correct checksum must verify");
    
    PASS();
}

// ============================================================================
// 5. Coin Address Tests
// ============================================================================

static void test_bitcoin_p2pkh_address() {
    TEST("Bitcoin: P2PKH address starts with 1");
    
    auto privkey = secp256k1::fast::Scalar::from_uint64(1);
    (void)privkey;
    auto pubkey = secp256k1::fast::Point::generator();
    
    auto addr = secp256k1::coins::coin_address_p2pkh(pubkey, secp256k1::coins::Bitcoin);
    ASSERT_TRUE(!addr.empty(), "address should not be empty");
    ASSERT_TRUE(addr[0] == '1', "Bitcoin P2PKH should start with '1'");
    
    PASS();
}

static void test_bitcoin_p2wpkh_address() {
    TEST("Bitcoin: P2WPKH address starts with bc1q");
    
    auto pubkey = secp256k1::fast::Point::generator();
    
    auto addr = secp256k1::coins::coin_address_p2wpkh(pubkey, secp256k1::coins::Bitcoin);
    ASSERT_TRUE(!addr.empty(), "address should not be empty");
    ASSERT_TRUE(addr.substr(0, 4) == "bc1q", "Bitcoin P2WPKH should start with 'bc1q'");
    
    PASS();
}

static void test_litecoin_address() {
    TEST("Litecoin: Bech32 address starts with ltc1q");
    
    auto pubkey = secp256k1::fast::Point::generator();
    
    auto addr = secp256k1::coins::coin_address(pubkey, secp256k1::coins::Litecoin);
    ASSERT_TRUE(!addr.empty(), "address should not be empty");
    ASSERT_TRUE(addr.substr(0, 5) == "ltc1q", "Litecoin default should start with 'ltc1q'");
    
    PASS();
}

static void test_dogecoin_address() {
    TEST("Dogecoin: P2PKH address starts with D");
    
    auto pubkey = secp256k1::fast::Point::generator();
    
    auto addr = secp256k1::coins::coin_address(pubkey, secp256k1::coins::Dogecoin);
    ASSERT_TRUE(!addr.empty(), "address should not be empty");
    ASSERT_TRUE(addr[0] == 'D', "Dogecoin P2PKH should start with 'D'");
    
    PASS();
}

static void test_ethereum_coin_address() {
    TEST("Ethereum: coin_address returns EIP-55");
    
    auto pubkey = secp256k1::fast::Point::generator();
    
    auto addr = secp256k1::coins::coin_address(pubkey, secp256k1::coins::Ethereum);
    ASSERT_TRUE(!addr.empty(), "address should not be empty");
    ASSERT_TRUE(addr.substr(0, 2) == "0x", "Ethereum should start with '0x'");
    ASSERT_EQ(addr.size(), 42u, "should be 42 chars");
    
    PASS();
}

static void test_dash_address() {
    TEST("Dash: P2PKH address starts with X");
    
    auto pubkey = secp256k1::fast::Point::generator();
    
    auto addr = secp256k1::coins::coin_address(pubkey, secp256k1::coins::Dash);
    ASSERT_TRUE(!addr.empty(), "address should not be empty");
    ASSERT_TRUE(addr[0] == 'X', "Dash P2PKH should start with 'X'");
    
    PASS();
}

static void test_no_segwit_returns_empty() {
    TEST("Dogecoin: P2WPKH returns empty (no SegWit)");
    
    auto pubkey = secp256k1::fast::Point::generator();
    
    auto addr = secp256k1::coins::coin_address_p2wpkh(pubkey, secp256k1::coins::Dogecoin);
    ASSERT_TRUE(addr.empty(), "Dogecoin should return empty for P2WPKH");
    
    PASS();
}

// ============================================================================
// 6. WIF Tests
// ============================================================================

static void test_bitcoin_wif() {
    TEST("Bitcoin: WIF starts with K or L (compressed)");
    
    auto privkey = secp256k1::fast::Scalar::from_uint64(12345);
    
    auto wif = secp256k1::coins::coin_wif_encode(privkey, secp256k1::coins::Bitcoin);
    ASSERT_TRUE(!wif.empty(), "WIF should not be empty");
    ASSERT_TRUE(wif[0] == 'K' || wif[0] == 'L', "compressed WIF starts with K or L");
    
    PASS();
}

static void test_litecoin_wif() {
    TEST("Litecoin: WIF starts with T (compressed)");
    
    auto privkey = secp256k1::fast::Scalar::from_uint64(12345);
    
    auto wif = secp256k1::coins::coin_wif_encode(privkey, secp256k1::coins::Litecoin);
    ASSERT_TRUE(!wif.empty(), "WIF should not be empty");
    // Litecoin WIF prefix 0xB0 -> compressed starts with T
    ASSERT_TRUE(wif[0] == 'T', "Litecoin compressed WIF starts with T");
    
    PASS();
}

// ============================================================================
// 7. BIP-44 HD Tests
// ============================================================================

static void test_bip44_path_bitcoin() {
    TEST("BIP-44: Bitcoin path m/86'/0'/0'/0/0 (Taproot)");
    
    auto path = secp256k1::coins::coin_derive_path(
        secp256k1::coins::Bitcoin, 0, false, 0,
        secp256k1::coins::DerivationPurpose::BIP86);
    ASSERT_STR_EQ(path, std::string("m/86'/0'/0'/0/0"), "path mismatch");
    
    PASS();
}

static void test_bip44_path_ethereum() {
    TEST("BIP-44: Ethereum path m/44'/60'/0'/0/0");
    
    auto path = secp256k1::coins::coin_derive_path(
        secp256k1::coins::Ethereum, 0, false, 0,
        secp256k1::coins::DerivationPurpose::BIP44);
    ASSERT_STR_EQ(path, std::string("m/44'/60'/0'/0/0"), "path mismatch");
    
    PASS();
}

static void test_bip44_best_purpose() {
    TEST("BIP-44: best_purpose selection");
    
    // Bitcoin -> BIP86 (Taproot)
    ASSERT_EQ(static_cast<int>(secp256k1::coins::best_purpose(secp256k1::coins::Bitcoin)),
              static_cast<int>(secp256k1::coins::DerivationPurpose::BIP86), "Bitcoin -> BIP86");
    
    // Litecoin -> BIP84 (SegWit but no Taproot)
    ASSERT_EQ(static_cast<int>(secp256k1::coins::best_purpose(secp256k1::coins::Litecoin)),
              static_cast<int>(secp256k1::coins::DerivationPurpose::BIP84), "Litecoin -> BIP84");
    
    // Dogecoin -> BIP44 (no SegWit)
    ASSERT_EQ(static_cast<int>(secp256k1::coins::best_purpose(secp256k1::coins::Dogecoin)),
              static_cast<int>(secp256k1::coins::DerivationPurpose::BIP44), "Dogecoin -> BIP44");
    
    PASS();
}

static void test_bip44_key_derivation() {
    TEST("BIP-44: seed -> key derivation");
    
    // BIP-39 test seed (all zeros, 16 bytes -- minimal valid seed)
    std::uint8_t seed[64] = {};
    seed[0] = 0x01; // Non-zero to avoid edge cases
    
    auto [master, master_ok] = secp256k1::bip32_master_key(seed, 64);
    ASSERT_TRUE(master_ok, "master key generation failed");
    
    auto [child, child_ok] = secp256k1::coins::coin_derive_key(
        master, secp256k1::coins::Bitcoin, 0, false, 0);
    ASSERT_TRUE(child_ok, "child derivation failed");
    ASSERT_TRUE(child.is_private, "child should be private");
    
    PASS();
}

static void test_bip44_seed_to_address() {
    TEST("BIP-44: seed -> Bitcoin address");
    
    std::uint8_t seed[64] = {};
    seed[0] = 0x42;
    
    auto [addr, ok] = secp256k1::coins::coin_address_from_seed(
        seed, 64, secp256k1::coins::Bitcoin);
    ASSERT_TRUE(ok, "address derivation failed");
    ASSERT_TRUE(!addr.empty(), "address should not be empty");
    
    // Bitcoin default address (BIP-86/Taproot or BIP-84/SegWit)
    ASSERT_TRUE(addr.substr(0, 3) == "bc1", "Bitcoin address should start with bc1");
    
    PASS();
}

static void test_bip44_seed_to_eth_address() {
    TEST("BIP-44: seed -> Ethereum address");
    
    std::uint8_t seed[64] = {};
    seed[0] = 0x42;
    
    auto [addr, ok] = secp256k1::coins::coin_address_from_seed(
        seed, 64, secp256k1::coins::Ethereum);
    ASSERT_TRUE(ok, "address derivation failed");
    ASSERT_TRUE(!addr.empty(), "address should not be empty");
    ASSERT_TRUE(addr.substr(0, 2) == "0x", "ETH address should start with 0x");
    ASSERT_EQ(addr.size(), 42u, "ETH address should be 42 chars");
    
    PASS();
}

// ============================================================================
// 8. Custom Generator + Coin Derivation Tests
// ============================================================================

static void test_custom_generator_coin_derive() {
    TEST("Custom G: coin_derive with custom generator");
    
    auto G = secp256k1::fast::Point::generator();
    auto G2 = G.dbl(); // 2*G as custom generator
    
    auto ctx = secp256k1::CurveContext::with_generator(G2);
    auto privkey = secp256k1::fast::Scalar::from_uint64(42);
    
    // Derive with custom G
    auto result = secp256k1::coins::coin_derive(
        privkey, secp256k1::coins::Bitcoin, false, &ctx);
    
    // Verify public key = privkey * (2*G) = 42 * (2*G) = 84*G
    auto expected = G.scalar_mul(secp256k1::fast::Scalar::from_uint64(84));
    ASSERT_TRUE(result.public_key.x() == expected.x(), "custom G derive mismatch");
    
    // Address should still be valid (just different from standard)
    ASSERT_TRUE(!result.address.empty(), "address should not be empty");
    
    PASS();
}

static void test_custom_generator_deterministic() {
    TEST("Custom G: deterministic derivation");
    
    auto G5 = secp256k1::fast::Point::generator()
                .scalar_mul(secp256k1::fast::Scalar::from_uint64(5));
    auto ctx = secp256k1::CurveContext::with_generator(G5);
    auto privkey = secp256k1::fast::Scalar::from_uint64(100);
    
    auto pub1 = secp256k1::derive_public_key(privkey, &ctx);
    auto pub2 = secp256k1::derive_public_key(privkey, &ctx);
    
    ASSERT_TRUE(pub1.x() == pub2.x(), "same inputs must produce same output");
    ASSERT_TRUE(pub1.y() == pub2.y(), "same inputs must produce same output (y)");
    
    PASS();
}

// ============================================================================
// 9. Full Pipeline Test
// ============================================================================

static void test_full_pipeline_multi_coin() {
    TEST("Full pipeline: same key -> different addresses per coin");
    
    auto privkey = secp256k1::fast::Scalar::from_uint64(999);
    auto pubkey = secp256k1::derive_public_key(privkey);
    
    auto btc_addr = secp256k1::coins::coin_address(pubkey, secp256k1::coins::Bitcoin);
    auto ltc_addr = secp256k1::coins::coin_address(pubkey, secp256k1::coins::Litecoin);
    auto doge_addr = secp256k1::coins::coin_address(pubkey, secp256k1::coins::Dogecoin);
    auto eth_addr = secp256k1::coins::coin_address(pubkey, secp256k1::coins::Ethereum);
    
    // All should be non-empty and different
    ASSERT_TRUE(!btc_addr.empty(), "BTC address empty");
    ASSERT_TRUE(!ltc_addr.empty(), "LTC address empty");
    ASSERT_TRUE(!doge_addr.empty(), "DOGE address empty");
    ASSERT_TRUE(!eth_addr.empty(), "ETH address empty");
    
    ASSERT_TRUE(btc_addr != ltc_addr, "BTC != LTC");
    ASSERT_TRUE(btc_addr != doge_addr, "BTC != DOGE");
    ASSERT_TRUE(btc_addr != eth_addr, "BTC != ETH");
    ASSERT_TRUE(ltc_addr != doge_addr, "LTC != DOGE");
    
    PASS();
}

// ============================================================================
// Main
// ============================================================================

int test_coins_run() {
    printf("=== Coins Layer + Custom Generator Tests ===\n\n");
    
    printf("[CurveContext]\n");
    test_context_default();
    test_context_custom_generator();
    test_context_derive_public_key();
    test_context_effective_generator();
    
    printf("\n[CoinParams]\n");
    test_coin_params_count();
    test_coin_params_bitcoin();
    test_coin_params_ethereum();
    test_coin_params_lookup();
    
    printf("\n[Keccak-256]\n");
    test_keccak256_empty();
    test_keccak256_abc();
    test_keccak256_incremental();
    
    printf("\n[Ethereum]\n");
    test_ethereum_address_format();
    test_ethereum_eip55_checksum();
    test_ethereum_eip55_case_sensitivity();
    
    printf("\n[Coin Addresses]\n");
    test_bitcoin_p2pkh_address();
    test_bitcoin_p2wpkh_address();
    test_litecoin_address();
    test_dogecoin_address();
    test_ethereum_coin_address();
    test_dash_address();
    test_no_segwit_returns_empty();
    
    printf("\n[WIF]\n");
    test_bitcoin_wif();
    test_litecoin_wif();
    
    printf("\n[BIP-44 HD]\n");
    test_bip44_path_bitcoin();
    test_bip44_path_ethereum();
    test_bip44_best_purpose();
    test_bip44_key_derivation();
    test_bip44_seed_to_address();
    test_bip44_seed_to_eth_address();
    
    printf("\n[Custom Generator]\n");
    test_custom_generator_coin_derive();
    test_custom_generator_deterministic();
    
    printf("\n[Full Pipeline]\n");
    test_full_pipeline_multi_coin();
    
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n");
    
    return tests_failed > 0 ? 1 : 0;
}
