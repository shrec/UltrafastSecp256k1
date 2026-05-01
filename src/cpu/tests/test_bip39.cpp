// ============================================================================
// Test: BIP-39 Mnemonic Seed Phrases
// ============================================================================
// Test vectors from:
//   https://github.com/trezor/python-mnemonic/blob/master/vectors.json
//   (passphrase = "TREZOR" for all seed derivation tests)
// ============================================================================

#include "secp256k1/bip39.hpp"
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
        char pair[3] = { hex[2 * i], hex[2 * i + 1], '\0' };
        char* endptr = nullptr;
        const unsigned long val = std::strtoul(pair, &endptr, 16);
        out[i] = (endptr == pair + 2) ? static_cast<uint8_t>(val) : 0;
    }
}

static std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::string result;
    result.reserve(len * 2);
    for (size_t i = 0; i < len; ++i) {
        char buf[3];
        (void)std::snprintf(buf, sizeof(buf), "%02x", data[i]);
        result += buf;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Test: PBKDF2-HMAC-SHA512 against known vector
// ---------------------------------------------------------------------------
static void test_pbkdf2() {
    printf("\n--- PBKDF2-HMAC-SHA512 ---\n");

    // RFC 6070 style test: password="password", salt="salt", c=1, dkLen=64
    // (We verify our PBKDF2 produces repeatable output)
    const char* pwd = "password";
    const char* salt = "salt";
    uint8_t output[64] = {};
    pbkdf2_hmac_sha512(
        reinterpret_cast<const uint8_t*>(pwd), std::strlen(pwd),
        reinterpret_cast<const uint8_t*>(salt), std::strlen(salt),
        1, output, 64);

    // Known result for PBKDF2-HMAC-SHA512("password", "salt", 1, 64):
    // 867f70cf1ade02cff3752599a3a53dc4af34c7a669815ae5d513554e1c8cf252
    // c02d470a285a0501bad999bfe943c08f050235d7d68b1da55e63f73b60a57fce
    const char* expected_hex =
        "867f70cf1ade02cff3752599a3a53dc4af34c7a669815ae5d513554e1c8cf252"
        "c02d470a285a0501bad999bfe943c08f050235d7d68b1da55e63f73b60a57fce";
    uint8_t expected[64];
    hex_to_bytes(expected_hex, expected, 64);
    CHECK(std::memcmp(output, expected, 64) == 0, "PBKDF2-HMAC-SHA512 (c=1)");

    // PBKDF2 with 2 iterations
    pbkdf2_hmac_sha512(
        reinterpret_cast<const uint8_t*>(pwd), std::strlen(pwd),
        reinterpret_cast<const uint8_t*>(salt), std::strlen(salt),
        2, output, 64);
    // Known: e1d9c16aa681708a45f5c7c4e215ceb66e011a2e9f0040713f18aefdb866d53c
    //        f76cab2868a39b9f7840edce4fef5a82be67335c77a6068e04112754f27ccf4e
    const char* expected_2_hex =
        "e1d9c16aa681708a45f5c7c4e215ceb66e011a2e9f0040713f18aefdb866d53c"
        "f76cab2868a39b9f7840edce4fef5a82be67335c77a6068e04112754f27ccf4e";
    hex_to_bytes(expected_2_hex, expected, 64);
    CHECK(std::memcmp(output, expected, 64) == 0, "PBKDF2-HMAC-SHA512 (c=2)");
}

// ---------------------------------------------------------------------------
// Test: Wordlist access
// ---------------------------------------------------------------------------
static void test_wordlist() {
    printf("\n--- Wordlist ---\n");

    const char* const* wl = bip39_wordlist_english();
    CHECK(wl != nullptr, "wordlist not null");
    if (!wl) { return; }
    CHECK(std::strcmp(wl[0], "abandon") == 0, "first word = abandon");
    CHECK(std::strcmp(wl[2047], "zoo") == 0, "last word = zoo");
    CHECK(std::strcmp(wl[1], "ability") == 0, "word[1] = ability");
    CHECK(std::strcmp(wl[100], "arrive") == 0, "word[100] = arrive");
}

// ---------------------------------------------------------------------------
// Test: Entropy -> Mnemonic (Trezor official test vectors)
// ---------------------------------------------------------------------------
static void test_entropy_to_mnemonic() {
    printf("\n--- Entropy to Mnemonic ---\n");

    // Vector 1: 128-bit entropy (12 words)
    {
        uint8_t entropy[16];
        hex_to_bytes("00000000000000000000000000000000", entropy, 16);
        auto [mnemonic, ok] = bip39_generate(16, entropy);
        CHECK(ok, "TV1: generate ok");
        CHECK(mnemonic == "abandon abandon abandon abandon abandon abandon "
                          "abandon abandon abandon abandon abandon about",
              "TV1: 128-bit zero entropy -> correct mnemonic");
    }

    // Vector 2: 128-bit entropy
    {
        uint8_t entropy[16];
        hex_to_bytes("7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f", entropy, 16);
        auto [mnemonic, ok] = bip39_generate(16, entropy);
        CHECK(ok, "TV2: generate ok");
        CHECK(mnemonic == "legal winner thank year wave sausage worth useful "
                          "legal winner thank yellow",
              "TV2: 128-bit 0x7f entropy -> correct mnemonic");
    }

    // Vector 3: 128-bit entropy
    {
        uint8_t entropy[16];
        hex_to_bytes("80808080808080808080808080808080", entropy, 16);
        auto [mnemonic, ok] = bip39_generate(16, entropy);
        CHECK(ok, "TV3: generate ok");
        CHECK(mnemonic == "letter advice cage absurd amount doctor acoustic "
                          "avoid letter advice cage above",
              "TV3: 128-bit 0x80 entropy -> correct mnemonic");
    }

    // Vector 4: 128-bit entropy
    {
        uint8_t entropy[16];
        hex_to_bytes("ffffffffffffffffffffffffffffffff", entropy, 16);
        auto [mnemonic, ok] = bip39_generate(16, entropy);
        CHECK(ok, "TV4: generate ok");
        CHECK(mnemonic == "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong",
              "TV4: 128-bit 0xff entropy -> correct mnemonic");
    }

    // Vector 5: 256-bit entropy (24 words)
    {
        uint8_t entropy[32];
        hex_to_bytes("0000000000000000000000000000000000000000000000000000000000000000",
                     entropy, 32);
        auto [mnemonic, ok] = bip39_generate(32, entropy);
        CHECK(ok, "TV5: generate ok");
        CHECK(mnemonic == "abandon abandon abandon abandon abandon abandon "
                          "abandon abandon abandon abandon abandon abandon "
                          "abandon abandon abandon abandon abandon abandon "
                          "abandon abandon abandon abandon abandon art",
              "TV5: 256-bit zero entropy -> correct mnemonic");
    }

    // Vector 6: 256-bit entropy
    {
        uint8_t entropy[32];
        hex_to_bytes("7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f",
                     entropy, 32);
        auto [mnemonic, ok] = bip39_generate(32, entropy);
        CHECK(ok, "TV6: generate ok");
        CHECK(mnemonic == "legal winner thank year wave sausage worth useful "
                          "legal winner thank year wave sausage worth useful "
                          "legal winner thank year wave sausage worth title",
              "TV6: 256-bit 0x7f entropy -> correct mnemonic");
    }

    // Vector 7: 256-bit entropy (0xff)
    {
        uint8_t entropy[32];
        hex_to_bytes("ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                     entropy, 32);
        auto [mnemonic, ok] = bip39_generate(32, entropy);
        CHECK(ok, "TV7: generate ok");
        CHECK(mnemonic == "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo "
                          "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo vote",
              "TV7: 256-bit 0xff entropy -> correct mnemonic");
    }
}

// ---------------------------------------------------------------------------
// Test: Mnemonic validation
// ---------------------------------------------------------------------------
static void test_validate() {
    printf("\n--- Mnemonic Validation ---\n");

    CHECK(bip39_validate("abandon abandon abandon abandon abandon abandon "
                          "abandon abandon abandon abandon abandon about"),
          "valid 12-word mnemonic");

    CHECK(bip39_validate("zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo wrong"),
          "valid 12-word zoo mnemonic");

    CHECK(bip39_validate("zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo "
                          "zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo zoo vote"),
          "valid 24-word mnemonic");

    // Invalid: wrong word
    CHECK(!bip39_validate("abandon abandon abandon abandon abandon abandon "
                           "abandon abandon abandon abandon abandon xyzzy"),
          "invalid word rejected");

    // Invalid: wrong checksum
    CHECK(!bip39_validate("abandon abandon abandon abandon abandon abandon "
                           "abandon abandon abandon abandon abandon abandon"),
          "bad checksum rejected");

    // Invalid: wrong word count
    CHECK(!bip39_validate("abandon abandon abandon"), "3-word mnemonic rejected");
    CHECK(!bip39_validate(""), "empty mnemonic rejected");
}

// ---------------------------------------------------------------------------
// Test: Mnemonic -> Seed (Trezor official vectors, passphrase="TREZOR")
// ---------------------------------------------------------------------------
static void test_mnemonic_to_seed() {
    printf("\n--- Mnemonic to Seed ---\n");

    // TV1: 128-bit zero entropy, passphrase "TREZOR"
    {
        const char* mnemonic = "abandon abandon abandon abandon abandon abandon "
                               "abandon abandon abandon abandon abandon about";
        auto [seed, ok] = bip39_mnemonic_to_seed(mnemonic, "TREZOR");
        CHECK(ok, "TV1 seed: derivation ok");
        const std::string hex = bytes_to_hex(seed.data(), 64);
        CHECK(hex == "c55257c360c07c72029aebc1b53c05ed0362ada38ead3e3e9efa3708e5349553"
                     "1f09a6987599d18264c1e1c92f2cf141630c7a3c4ab7c81b2f001698e7463b04",
              "TV1 seed: matches Trezor vector");
    }

    // TV5: 256-bit zero entropy, passphrase "TREZOR"
    {
        const char* mnemonic = "abandon abandon abandon abandon abandon abandon "
                               "abandon abandon abandon abandon abandon abandon "
                               "abandon abandon abandon abandon abandon abandon "
                               "abandon abandon abandon abandon abandon art";
        auto [seed, ok] = bip39_mnemonic_to_seed(mnemonic, "TREZOR");
        CHECK(ok, "TV5 seed: derivation ok");
        const std::string hex = bytes_to_hex(seed.data(), 64);
        CHECK(hex == "bda85446c68413707090a52022edd26a1c9462295029f2e60cd7c4f2bbd30971"
                     "70af7a4d73245cafa9c3cca8d561a7c3de6f5d4a10be8ed2a5e608d68f92fcc8",
              "TV5 seed: matches Trezor vector");
    }

    // No passphrase test
    {
        const char* mnemonic = "abandon abandon abandon abandon abandon abandon "
                               "abandon abandon abandon abandon abandon about";
        auto [seed, ok] = bip39_mnemonic_to_seed(mnemonic, "");
        CHECK(ok, "no-passphrase seed: derivation ok");
        const std::string hex = bytes_to_hex(seed.data(), 64);
        // Known result with empty passphrase (salt = "mnemonic"):
        CHECK(hex == "5eb00bbddcf069084889a8ab9155568165f5c453ccb85e70811aaed6f6da5fc1"
                     "9a5ac40b389cd370d086206dec8aa6c43daea6690f20ad3d8d48b2d2ce9e38e4",
              "no-passphrase seed: correct");
    }
}

// ---------------------------------------------------------------------------
// Test: Mnemonic -> Entropy roundtrip
// ---------------------------------------------------------------------------
static void test_mnemonic_to_entropy() {
    printf("\n--- Mnemonic to Entropy ---\n");

    // TV1 roundtrip
    {
        uint8_t orig[16];
        hex_to_bytes("00000000000000000000000000000000", orig, 16);
        auto [mnemonic, ok1] = bip39_generate(16, orig);
        CHECK(ok1, "roundtrip: generate ok");
        auto [ent, ok2] = bip39_mnemonic_to_entropy(mnemonic);
        CHECK(ok2, "roundtrip: decode ok");
        CHECK(ent.length == 16, "roundtrip: entropy length = 16");
        CHECK(std::memcmp(ent.data.data(), orig, 16) == 0, "roundtrip: entropy matches");
    }

    // 256-bit roundtrip
    {
        uint8_t orig[32];
        hex_to_bytes("68a79eaca2324873eacc50cb9c6eca8cc68ea5d936f98787c60c7ebc74e6ce7c",
                     orig, 32);
        auto [mnemonic, ok1] = bip39_generate(32, orig);
        CHECK(ok1, "roundtrip-256: generate ok");
        auto [ent, ok2] = bip39_mnemonic_to_entropy(mnemonic);
        CHECK(ok2, "roundtrip-256: decode ok");
        CHECK(ent.length == 32, "roundtrip-256: entropy length = 32");
        CHECK(std::memcmp(ent.data.data(), orig, 32) == 0, "roundtrip-256: entropy matches");
    }
}

// ---------------------------------------------------------------------------
// Test: Random mnemonic generation
// ---------------------------------------------------------------------------
static void test_random_generation() {
    printf("\n--- Random Generation ---\n");

    // Generate with CSPRNG (no entropy_in)
    auto [m12, ok12] = bip39_generate(16);
    CHECK(ok12, "random 12-word generation ok");
    CHECK(bip39_validate(m12), "random 12-word validates");

    auto [m24, ok24] = bip39_generate(32);
    CHECK(ok24, "random 24-word generation ok");
    CHECK(bip39_validate(m24), "random 24-word validates");

    // Invalid sizes rejected
    auto [bad1, ok_bad1] = bip39_generate(15);
    CHECK(!ok_bad1, "15-byte entropy rejected");
    auto [bad2, ok_bad2] = bip39_generate(33);
    CHECK(!ok_bad2, "33-byte entropy rejected");
}

// ---------------------------------------------------------------------------
// Test: Edge cases
// ---------------------------------------------------------------------------
static void test_edge_cases() {
    printf("\n--- Edge Cases ---\n");

    // 160-bit entropy (15 words)
    {
        uint8_t entropy[20];
        hex_to_bytes("0000000000000000000000000000000000000000", entropy, 20);
        auto [mnemonic, ok] = bip39_generate(20, entropy);
        CHECK(ok, "160-bit entropy generates ok");
        // Count words
        int wc = 1;
        for (const char c : mnemonic) if (c == ' ') ++wc;
        CHECK(wc == 15, "160-bit entropy -> 15 words");
        CHECK(bip39_validate(mnemonic), "160-bit mnemonic validates");
    }

    // 192-bit entropy (18 words)
    {
        uint8_t entropy[24];
        hex_to_bytes("000000000000000000000000000000000000000000000000", entropy, 24);
        auto [mnemonic, ok] = bip39_generate(24, entropy);
        CHECK(ok, "192-bit entropy generates ok");
        int wc = 1;
        for (const char c : mnemonic) if (c == ' ') ++wc;
        CHECK(wc == 18, "192-bit entropy -> 18 words");
        CHECK(bip39_validate(mnemonic), "192-bit mnemonic validates");
    }

    // 224-bit entropy (21 words)
    {
        uint8_t entropy[28];
        hex_to_bytes("00000000000000000000000000000000000000000000000000000000",
                     entropy, 28);
        auto [mnemonic, ok] = bip39_generate(28, entropy);
        CHECK(ok, "224-bit entropy generates ok");
        int wc = 1;
        for (const char c : mnemonic) if (c == ' ') ++wc;
        CHECK(wc == 21, "224-bit entropy -> 21 words");
        CHECK(bip39_validate(mnemonic), "224-bit mnemonic validates");
    }
}

// -- Main ---------------------------------------------------------------------

int test_bip39_run() {
    printf("=== BIP-39 Mnemonic Seed Phrase Tests ===\n");

    test_pbkdf2();
    test_wordlist();
    test_entropy_to_mnemonic();
    test_validate();
    test_mnemonic_to_seed();
    test_mnemonic_to_entropy();
    test_random_generation();
    test_edge_cases();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_bip39_run();
}
#endif
