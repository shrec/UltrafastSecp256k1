// ============================================================================
// Test: RFC 6979 Deterministic ECDSA (secp256k1 + SHA-256)
// ============================================================================
// Source: bitcoinjs/bitcoinjs-lib test/fixtures/ecdsa.json
// These are well-established secp256k1-specific vectors that test:
//   1. RFC 6979 nonce generation (k value)
//   2. Full ECDSA signature (r, s)
//   3. Verification roundtrip
// ============================================================================

#include "secp256k1/ecdsa.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

#include <cstdio>
#include <cstring>
#include <array>

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

// -- Hex helpers (allocation-free) -------------------------------------------

static std::array<uint8_t, 32> h32(const char* hex) {
    std::array<uint8_t, 32> r{};
    for (size_t i = 0; i < 32; ++i) {
        unsigned v = 0;
        std::sscanf(hex + i * 2, "%02x", &v);
        r[i] = static_cast<uint8_t>(v);
    }
    return r;
}

static void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("    %s: ", label);
    for (size_t i = 0; i < len; ++i)
        printf("%02x", data[i]);
    printf("\n");
}

// -- SHA-256 of ASCII string --------------------------------------------------

static std::array<uint8_t, 32> sha256_str(const char* str) {
    return SHA256::hash(str, std::strlen(str));
}

// -- Scalar comparison helper (with diagnostics) ------------------------------

static bool scalar_eq_hex(const Scalar& got, const char* expected_hex,
                          const char* label) {
    auto expected = Scalar::from_hex(expected_hex);
    if (got == expected) return true;

    // Diagnostic on mismatch
    auto got_bytes = got.to_bytes();
    auto exp_bytes = expected.to_bytes();
    printf("    MISMATCH in %s:\n", label);
    print_hex("expected", exp_bytes.data(), 32);
    print_hex("got     ", got_bytes.data(), 32);
    return false;
}

// ============================================================================
// Section 1: RFC 6979 Nonce Generation Tests
// ============================================================================
// Test that rfc6979_nonce(privkey, SHA256(msg)) produces the expected k.
// Source: bitcoinjs ecdsa.json "valid.rfc6979" section
// ============================================================================

static void test_rfc6979_nonce_vectors() {
    printf("\n--- RFC 6979 Nonce Generation (secp256k1 + SHA-256) ---\n");

    // Vector 1: d=fee0...be1e, msg="test data"
    {
        auto d = Scalar::from_hex(
            "fee0a1f7afebf9d2a5a80c0c98a31c709681cce195cbcd06342b517970c0be1e");
        auto msg_hash = sha256_str("test data");
        auto k = rfc6979_nonce(d, msg_hash);
        CHECK(scalar_eq_hex(k,
            "fcce1de7a9bcd6b2d3defade6afa1913fb9229e3b7ddf4749b55c4848b2a196e",
            "k"),
            "RFC6979 nonce: d=fee0...be1e msg='test data'");
    }

    // Vector 2: d=1, msg="Everything should be made as simple as possible, but not simpler."
    {
        auto d = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto msg_hash = sha256_str(
            "Everything should be made as simple as possible, but not simpler.");
        auto k = rfc6979_nonce(d, msg_hash);
        CHECK(scalar_eq_hex(k,
            "ec633bd56a5774a0940cb97e27a9e4e51dc94af737596a0c5cbb3d30332d92a5",
            "k"),
            "RFC6979 nonce: d=1 msg='Everything should be...'");
    }

    // Vector 3: d=2, msg="Satoshi Nakamoto"
    {
        auto d = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000002");
        auto msg_hash = sha256_str("Satoshi Nakamoto");
        auto k = rfc6979_nonce(d, msg_hash);
        CHECK(scalar_eq_hex(k,
            "d3edc1b8224e953f6ee05c8bbf7ae228f461030e47caf97cde91430b4607405e",
            "k"),
            "RFC6979 nonce: d=2 msg='Satoshi Nakamoto'");
    }

    // Vector 4: d=7f7f...7f, msg="Diffie Hellman"
    {
        auto d = Scalar::from_hex(
            "7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f7f");
        auto msg_hash = sha256_str("Diffie Hellman");
        auto k = rfc6979_nonce(d, msg_hash);
        CHECK(scalar_eq_hex(k,
            "c378a41cb17dce12340788dd3503635f54f894c306d52f6e9bc4b8f18d27afcc",
            "k"),
            "RFC6979 nonce: d=7f7f...7f msg='Diffie Hellman'");
    }

    // Vector 5: d=8080...80, msg="Japan"
    {
        auto d = Scalar::from_hex(
            "8080808080808080808080808080808080808080808080808080808080808080");
        auto msg_hash = sha256_str("Japan");
        auto k = rfc6979_nonce(d, msg_hash);
        CHECK(scalar_eq_hex(k,
            "f471e61b51d2d8db78f3dae19d973616f57cdc54caaa81c269394b8c34edcf59",
            "k"),
            "RFC6979 nonce: d=8080...80 msg='Japan'");
    }

    // Vector 6: d=n-1, msg="Bitcoin"
    {
        auto d = Scalar::from_hex(
            "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140");
        auto msg_hash = sha256_str("Bitcoin");
        auto k = rfc6979_nonce(d, msg_hash);
        CHECK(scalar_eq_hex(k,
            "36c848ffb2cbecc5422c33a994955b807665317c1ce2a0f59c689321aaa631cc",
            "k"),
            "RFC6979 nonce: d=n-1 msg='Bitcoin'");
    }
}

// ============================================================================
// Section 2: Full ECDSA Signature Tests
// ============================================================================
// Test that ecdsa_sign produces the expected (r, s) values.
// Source: bitcoinjs ecdsa.json "valid.ecdsa" section
// ============================================================================

static void test_ecdsa_sign_vectors() {
    printf("\n--- ECDSA Signature Vectors (secp256k1 + SHA-256) ---\n");

    // Vector 1: d=1, msg="Everything should be made as simple..."
    {
        auto d = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto msg_hash = sha256_str(
            "Everything should be made as simple as possible, but not simpler.");
        auto sig = ecdsa_sign(msg_hash, d);

        CHECK(scalar_eq_hex(sig.r,
            "33a69cd2065432a30f3d1ce4eb0d59b8ab58c74f27c41a7fdb5696ad4e6108c9",
            "r"),
            "ECDSA sig r: d=1 (Einstein quote)");
        CHECK(scalar_eq_hex(sig.s,
            "6f807982866f785d3f6418d24163ddae117b7db4d5fdf0071de069fa54342262",
            "s"),
            "ECDSA sig s: d=1 (Einstein quote)");
    }

    // Vector 2: d=1, msg="How wonderful that we have met with a paradox..."
    {
        auto d = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000001");
        auto msg_hash = sha256_str(
            "How wonderful that we have met with a paradox. Now we have some hope of making progress.");
        auto sig = ecdsa_sign(msg_hash, d);

        CHECK(scalar_eq_hex(sig.r,
            "c0dafec8251f1d5010289d210232220b03202cba34ec11fec58b3e93a85b91d3",
            "r"),
            "ECDSA sig r: d=1 (Bohr quote)");
        CHECK(scalar_eq_hex(sig.s,
            "75afdc06b7d6322a590955bf264e7aaa155847f614d80078a90292fe205064d3",
            "s"),
            "ECDSA sig s: d=1 (Bohr quote)");
    }

    // Vector 3: d=n-1, msg="Equations are more important..."
    {
        auto d = Scalar::from_hex(
            "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140");
        auto msg_hash = sha256_str(
            "Equations are more important to me, because politics is for the present, but an equation is something for eternity.");
        auto sig = ecdsa_sign(msg_hash, d);

        CHECK(scalar_eq_hex(sig.r,
            "54c4a33c6423d689378f160a7ff8b61330444abb58fb470f96ea16d99d4a2fed",
            "r"),
            "ECDSA sig r: d=n-1 (Dirac quote)");
        CHECK(scalar_eq_hex(sig.s,
            "07082304410efa6b2943111b6a4e0aaa7b7db55a07e9861d1fb3cb1f421044a5",
            "s"),
            "ECDSA sig s: d=n-1 (Dirac quote)");
    }

    // Vector 4: d=n-1, msg="Not only is the Universe stranger..."
    {
        auto d = Scalar::from_hex(
            "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140");
        auto msg_hash = sha256_str(
            "Not only is the Universe stranger than we think, it is stranger than we can think.");
        auto sig = ecdsa_sign(msg_hash, d);

        CHECK(scalar_eq_hex(sig.r,
            "ff466a9f1b7b273e2f4c3ffe032eb2e814121ed18ef84665d0f515360dab3dd0",
            "r"),
            "ECDSA sig r: d=n-1 (Heisenberg quote)");
        CHECK(scalar_eq_hex(sig.s,
            "6fc95f5132e5ecfdc8e5e6e616cc77151455d46ed48f5589b7db7771a332b283",
            "s"),
            "ECDSA sig s: d=n-1 (Heisenberg quote)");
    }

    // Vector 5: d=69ec...ba64, msg="Computer science is no more about computers..."
    {
        auto d = Scalar::from_hex(
            "69ec59eaa1f4f2e36b639716b7c30ca86d9a5375c7b38d8918bd9c0ebc80ba64");
        auto msg_hash = sha256_str(
            "Computer science is no more about computers than astronomy is about telescopes.");
        auto sig = ecdsa_sign(msg_hash, d);

        CHECK(scalar_eq_hex(sig.r,
            "7186363571d65e084e7f02b0b77c3ec44fb1b257dee26274c38c928986fea45d",
            "r"),
            "ECDSA sig r: d=69ec (Dijkstra quote)");
        CHECK(scalar_eq_hex(sig.s,
            "0de0b38e06807e46bda1f1e293f4f6323e854c86d58abdd00c46c16441085df6",
            "s"),
            "ECDSA sig s: d=69ec (Dijkstra quote)");
    }

    // Vector 6: small d, msg="...if you aren't..."
    {
        auto d = Scalar::from_hex(
            "00000000000000000000000000007246174ab1e92e9149c6e446fe194d072637");
        auto msg_hash = sha256_str(
            "...if you aren't, at any given time, scandalized by code you wrote five or even three years ago, you're not learning anywhere near enough");
        auto sig = ecdsa_sign(msg_hash, d);

        CHECK(scalar_eq_hex(sig.r,
            "fbfe5076a15860ba8ed00e75e9bd22e05d230f02a936b653eb55b61c99dda487",
            "r"),
            "ECDSA sig r: small d (programming quote)");
        CHECK(scalar_eq_hex(sig.s,
            "0e68880ebb0050fe4312b1b1eb0899e1b82da89baa5b895f612619edf34cbd37",
            "s"),
            "ECDSA sig s: small d (programming quote)");
    }

    // Vector 7: small d, msg="The question of whether computers can think..."
    {
        auto d = Scalar::from_hex(
            "000000000000000000000000000000000000000000056916d0f9b31dc9b637f3");
        auto msg_hash = sha256_str(
            "The question of whether computers can think is like the question of whether submarines can swim.");
        auto sig = ecdsa_sign(msg_hash, d);

        CHECK(scalar_eq_hex(sig.r,
            "cde1302d83f8dd835d89aef803c74a119f561fbaef3eb9129e45f30de86abbf9",
            "r"),
            "ECDSA sig r: tiny d (Dijkstra submarine)");
        CHECK(scalar_eq_hex(sig.s,
            "06ce643f5049ee1f27890467b77a6a8e11ec4661cc38cd8badf90115fbd03cef",
            "s"),
            "ECDSA sig s: tiny d (Dijkstra submarine)");
    }
}

// ============================================================================
// Section 3: ECDSA Verify Roundtrip
// ============================================================================
// Verify that signatures produced by our signer are accepted by our verifier.
// ============================================================================

static void test_ecdsa_verify_roundtrip() {
    printf("\n--- ECDSA Verify Roundtrip ---\n");

    struct SignVerifyVec {
        const char* d_hex;
        const char* message;
    };

    static const SignVerifyVec vectors[] = {
        { "0000000000000000000000000000000000000000000000000000000000000001",
          "Everything should be made as simple as possible, but not simpler." },
        { "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140",
          "Equations are more important to me, because politics is for the present, but an equation is something for eternity." },
        { "69ec59eaa1f4f2e36b639716b7c30ca86d9a5375c7b38d8918bd9c0ebc80ba64",
          "Computer science is no more about computers than astronomy is about telescopes." },
        { "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35",
          "Hello, secp256k1!" },
        { "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
          "The quick brown fox jumps over the lazy dog" },
    };

    for (size_t i = 0; i < sizeof(vectors) / sizeof(vectors[0]); ++i) {
        auto priv = Scalar::from_hex(vectors[i].d_hex);
        auto pub = Point::generator().scalar_mul(priv);
        auto msg_hash = sha256_str(vectors[i].message);

        auto sig = ecdsa_sign(msg_hash, priv);

        // Verify correct
        bool valid = ecdsa_verify(msg_hash, pub, sig);
        char label[128];
        std::snprintf(label, sizeof(label), "verify roundtrip #%zu", i + 1);
        CHECK(valid, label);

        // Verify wrong message fails
        auto wrong_hash = sha256_str("wrong");
        bool invalid = ecdsa_verify(wrong_hash, pub, sig);
        std::snprintf(label, sizeof(label), "wrong msg rejects #%zu", i + 1);
        CHECK(!invalid, label);
    }
}

// ============================================================================
// Section 4: Determinism Check
// ============================================================================
// Same (key, msg) always produces same (r, s)
// ============================================================================

static void test_ecdsa_determinism() {
    printf("\n--- ECDSA Determinism ---\n");

    auto priv = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto msg_hash = sha256_str("determinism test message");

    auto sig1 = ecdsa_sign(msg_hash, priv);
    auto sig2 = ecdsa_sign(msg_hash, priv);

    CHECK(sig1.r == sig2.r && sig1.s == sig2.s,
          "same (key, msg) -> identical signature");

    // Different message -> different sig
    auto msg_hash2 = sha256_str("different message");
    auto sig3 = ecdsa_sign(msg_hash2, priv);
    CHECK(sig3.r != sig1.r || sig3.s != sig1.s,
          "different msg -> different signature");

    // All signatures normalized (low-S, BIP-62)
    CHECK(sig1.is_low_s(), "sig1 is low-S");
    CHECK(sig2.is_low_s(), "sig2 is low-S");
    CHECK(sig3.is_low_s(), "sig3 is low-S");
}

// ============================================================================
// Entry point
// ============================================================================

int test_rfc6979_vectors_run() {
    printf("================================================================\n");
    printf("  RFC 6979 Deterministic ECDSA Test Vectors (secp256k1)\n");
    printf("================================================================\n");

    test_rfc6979_nonce_vectors();
    test_ecdsa_sign_vectors();
    test_ecdsa_verify_roundtrip();
    test_ecdsa_determinism();

    printf("\n================================================================\n");
    printf("  Results: %d / %d passed\n", tests_passed, tests_run);
    printf("================================================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_rfc6979_vectors_run();
}
#endif
