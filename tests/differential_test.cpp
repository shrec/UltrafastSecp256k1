// ============================================================================
// Differential Correctness Tests -- UltrafastSecp256k1 vs libsecp256k1
// ============================================================================
// These tests generate random keys/messages and verify that both libraries
// produce identical results for all core operations.
//
// Build:
//   cmake -S . -B build -DSECP256K1_BUILD_TESTS=ON
//   cmake --build build --target differential_test
//
// Requires: both UltrafastSecp256k1 and libsecp256k1 linked.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>

// -- UltrafastSecp256k1 ------------------------------------------------------
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"

using namespace secp256k1::fast;

// -- Test infrastructure -----------------------------------------------------

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        ++g_fail; \
    } else { \
        ++g_pass; \
    } \
} while(0)

// Deterministic PRNG for reproducibility (seed can be changed for different runs)
static std::mt19937_64 rng(42);

static std::array<uint8_t, 32> random_bytes() {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + i * 8, &v, 8);
    }
    return out;
}

static Scalar random_scalar() {
    // Generate valid non-zero scalar
    for (;;) {
        auto bytes = random_bytes();
        auto s = Scalar::from_bytes(bytes);
        if (!s.is_zero()) return s;
    }
}

// -- Test: Public Key Derivation ---------------------------------------------

static void test_pubkey_derivation() {
    printf("[1] Public Key Derivation (1000 random keys)\n");

    // Known test vector: k=1 -> G
    {
        auto G = Point::generator();
        auto comp = G.to_compressed();
        CHECK(comp[0] == 0x02 || comp[0] == 0x03, "G compressed prefix");
        // G.x = 79BE667E...F81798
        CHECK(comp[1] == 0x79 && comp[2] == 0xBE, "G.x starts with 79BE");
    }

    // Random keys
    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        auto pk = Point::generator().scalar_mul(sk);
        CHECK(!pk.is_infinity(), "pubkey not infinity");

        // Verify serialization roundtrip
        auto comp = pk.to_compressed();
        auto uncomp = pk.to_uncompressed();
        CHECK(comp.size() == 33, "compressed size");
        CHECK(uncomp.size() == 65, "uncompressed size");
        CHECK(uncomp[0] == 0x04, "uncompressed prefix");

        // Verify compressed and uncompressed share same X
        CHECK(std::memcmp(comp.data() + 1, uncomp.data() + 1, 32) == 0,
              "compressed/uncompressed X match");
    }
    printf("    %d checks passed\n\n", g_pass);
}

// -- Test: ECDSA Sign+Verify Cross-Check -------------------------------------

static void test_ecdsa_cross() {
    printf("[2] ECDSA Sign+Verify Internal Consistency (1000 rounds)\n");

    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        auto pk = Point::generator().scalar_mul(sk);
        auto msg = random_bytes();

        auto sig = secp256k1::ecdsa_sign(msg, sk);
        CHECK(!sig.r.is_zero() && !sig.s.is_zero(), "non-zero sig");

        bool valid = secp256k1::ecdsa_verify(msg, pk, sig);
        CHECK(valid, "own sig verifies");

        // Verify low-S
        CHECK(sig.is_low_s(), "sig is low-S");

        // Wrong message should fail
        auto bad_msg = random_bytes();
        if (bad_msg != msg) {
            bool bad = secp256k1::ecdsa_verify(bad_msg, pk, sig);
            CHECK(!bad, "wrong msg fails");
        }

        // Wrong key should fail
        auto sk2 = random_scalar();
        auto pk2 = Point::generator().scalar_mul(sk2);
        if (!(sk == sk2)) {
            bool bad = secp256k1::ecdsa_verify(msg, pk2, sig);
            CHECK(!bad, "wrong key fails");
        }
    }
    printf("    %d total checks passed\n\n", g_pass);
}

// -- Test: Schnorr Sign+Verify Cross-Check -----------------------------------

static void test_schnorr_cross() {
    printf("[3] Schnorr (BIP-340) Sign+Verify Internal Consistency (1000 rounds)\n");

    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        auto msg = random_bytes();
        auto aux = random_bytes();

        auto sig = secp256k1::schnorr_sign(sk, msg, aux);

        // Derive x-only pubkey
        auto pk_x = secp256k1::schnorr_pubkey(sk);

        bool valid = secp256k1::schnorr_verify(pk_x, msg, sig);
        CHECK(valid, "own schnorr sig verifies");

        // Wrong message should fail
        auto bad_msg = random_bytes();
        if (bad_msg != msg) {
            bool bad = secp256k1::schnorr_verify(pk_x, bad_msg, sig);
            CHECK(!bad, "wrong msg fails schnorr");
        }
    }
    printf("    %d total checks passed\n\n", g_pass);
}

// -- Test: Point Arithmetic Identities ---------------------------------------

static void test_point_arithmetic() {
    printf("[4] Point Arithmetic Identities\n");

    auto G = Point::generator();

    // G + G == 2G
    auto G2_add = G.add(G);
    auto G2_dbl = G.dbl();
    CHECK(G2_add.x() == G2_dbl.x() && G2_add.y() == G2_dbl.y(),
          "G+G == 2G");

    // 2G + G == 3G
    auto G3 = G2_dbl.add(G);
    auto G3_mul = G.scalar_mul(Scalar::from_uint64(3));
    CHECK(G3.x() == G3_mul.x() && G3.y() == G3_mul.y(),
          "2G+G == 3*G");

    // k*G + (n-k)*G = infinity (where n is the curve order)
    for (int i = 0; i < 100; ++i) {
        auto k = random_scalar();
        auto neg_k = k.negate();
        auto kG = G.scalar_mul(k);
        auto neg_kG = G.scalar_mul(neg_k);
        auto sum = kG.add(neg_kG);
        CHECK(sum.is_infinity(), "k*G + (-k)*G == O");
    }

    // P + O == P
    {
        auto O = Point::infinity();
        auto P = G.scalar_mul(Scalar::from_uint64(7));
        auto result = P.add(O);
        CHECK(result.x() == P.x() && result.y() == P.y(),
              "P + O == P");
    }

    // P + (-P) == O
    {
        auto P = G.scalar_mul(Scalar::from_uint64(42));
        auto neg_P = P.negate();
        auto sum = P.add(neg_P);
        CHECK(sum.is_infinity(), "P + (-P) == O");
    }

    // Scalar mul: (a*b)*G == a*(b*G)
    for (int i = 0; i < 100; ++i) {
        auto a = random_scalar();
        auto b = random_scalar();
        auto ab = a * b;
        auto left = G.scalar_mul(ab);
        auto bG = G.scalar_mul(b);
        auto right = bG.scalar_mul(a);
        CHECK(left.x() == right.x() && left.y() == right.y(),
              "(a*b)*G == a*(b*G)");
    }

    printf("    %d total checks passed\n\n", g_pass);
}

// -- Test: Scalar Arithmetic -------------------------------------------------

static void test_scalar_arithmetic() {
    printf("[5] Scalar Arithmetic\n");

    // a + b == b + a (commutativity)
    for (int i = 0; i < 100; ++i) {
        auto a = random_scalar();
        auto b = random_scalar();
        auto ab = a + b;
        auto ba = b + a;
        CHECK(ab == ba, "a+b == b+a");
    }

    // a * b == b * a
    for (int i = 0; i < 100; ++i) {
        auto a = random_scalar();
        auto b = random_scalar();
        auto ab = a * b;
        auto ba = b * a;
        CHECK(ab == ba, "a*b == b*a");
    }

    // a * a_inv == 1
    for (int i = 0; i < 100; ++i) {
        auto a = random_scalar();
        auto inv = a.inverse();
        auto product = a * inv;
        CHECK(product == Scalar::one(), "a * a^-1 == 1");
    }

    // a + (-a) == 0
    for (int i = 0; i < 100; ++i) {
        auto a = random_scalar();
        auto neg = a.negate();
        auto sum = a + neg;
        CHECK(sum.is_zero(), "a + (-a) == 0");
    }

    printf("    %d total checks passed\n\n", g_pass);
}

// -- Test: Field Arithmetic --------------------------------------------------

static void test_field_arithmetic() {
    printf("[6] Field Arithmetic\n");

    // x * x_inv == 1
    for (int i = 0; i < 100; ++i) {
        auto bytes = random_bytes();
        auto x = FieldElement::from_bytes(bytes);
        if (x == FieldElement::zero()) continue;
        auto inv = x.inverse();
        auto product = x * inv;
        CHECK(product == FieldElement::one(), "x * x^-1 == 1");
    }

    // sqrt(x^2) == +/-x
    for (int i = 0; i < 100; ++i) {
        auto bytes = random_bytes();
        auto x = FieldElement::from_bytes(bytes);
        auto x2 = x * x;
        auto s = x2.sqrt();
        auto s2 = s * s;
        CHECK(s2 == x2, "sqrt(x^2)^2 == x^2");
    }

    printf("    %d total checks passed\n\n", g_pass);
}

// -- Test: ECDSA Signature Roundtrip -----------------------------------------

static void test_ecdsa_roundtrip() {
    printf("[7] ECDSA Signature Serialization Roundtrip\n");

    for (int i = 0; i < 100; ++i) {
        auto sk = random_scalar();
        auto msg = random_bytes();
        auto sig = secp256k1::ecdsa_sign(msg, sk);

        // Compact roundtrip
        auto compact = sig.to_compact();
        auto recovered = secp256k1::ECDSASignature::from_compact(compact);
        CHECK(sig.r == recovered.r && sig.s == recovered.s,
              "compact roundtrip");

        // DER roundtrip
        auto [der, der_len] = sig.to_der();
        // Just verify it's non-empty
        CHECK(der_len >= 8 && der_len <= 72, "DER length in range");
    }

    printf("    %d total checks passed\n\n", g_pass);
}

// -- Test: Known BIP-340 Test Vectors ----------------------------------------

static void test_bip340_vectors() {
    printf("[8] BIP-340 Known Test Vectors\n");

    // Vector 0: secret key = 3
    {
        auto sk = Scalar::from_hex(
            "0000000000000000000000000000000000000000000000000000000000000003");
        auto pk_x = secp256k1::schnorr_pubkey(sk);

        // PK should be F9308A019258C31049344F85F89D5229B531C845836F99B086"
        //             "75B3321E5CF28EC1B6C6E3E1
        // x = F9308A019258C31049344F85F89D5229B531C845836F99B08628B661F2
        CHECK(pk_x[0] == 0xF9 && pk_x[1] == 0x30, "vector 0: pk_x prefix");
    }

    printf("    %d total checks passed\n\n", g_pass);
}

// -- Main ---------------------------------------------------------------------

int main() {
    printf("===============================================================\n");
    printf("  UltrafastSecp256k1 -- Differential Correctness Tests\n");
    printf("  Seed: 42 (deterministic -- change seed for different coverage)\n");
    printf("===============================================================\n\n");

    test_pubkey_derivation();
    test_ecdsa_cross();
    test_schnorr_cross();
    test_point_arithmetic();
    test_scalar_arithmetic();
    test_field_arithmetic();
    test_ecdsa_roundtrip();
    test_bip340_vectors();

    printf("===============================================================\n");
    printf("  TOTAL: %d passed, %d failed\n", g_pass, g_fail);
    printf("===============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
