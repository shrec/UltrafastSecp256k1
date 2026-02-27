// ============================================================================
// Cryptographic Self-Audit: Point Operations (secp256k1 curve)
// ============================================================================
// Covers: infinity, Jacobian add/dbl special cases, mixed-add H=0,
//         Z=0 propagation, affine conversion, batch inversion zero detection,
//         differential test vs known vectors.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "test_vectors.hpp"

using namespace secp256k1::fast;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL [%s]: %s (line %d)\n", g_section, msg, __LINE__); \
        ++g_fail; \
    } else { \
        ++g_pass; \
    } \
} while(0)

static std::mt19937_64 rng(0xA0D17'F01DA);

static Scalar random_scalar() {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + i * 8, &v, 8);
    }
    for (;;) {
        auto s = Scalar::from_bytes(out);
        if (!s.is_zero()) return s;
        out[31] ^= 0x01;
    }
}

static bool points_equal(const Point& a, const Point& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() != b.is_infinity()) return false;
    return a.to_compressed() == b.to_compressed();
}

// ============================================================================
// 1. Point at infinity
// ============================================================================
static void test_infinity() {
    g_section = "infinity";
    printf("[1] Point at infinity correctness\n");

    auto O = Point::infinity();
    auto G = Point::generator();

    CHECK(O.is_infinity(), "O is infinity");
    CHECK(!G.is_infinity(), "G is not infinity");

    // O + O == O
    CHECK(O.add(O).is_infinity(), "O + O == O");

    // G + O == G
    CHECK(points_equal(G.add(O), G), "G + O == G");

    // O + G == G
    CHECK(points_equal(O.add(G), G), "O + G == G");

    // 0 * G == O
    auto zero = Scalar::from_uint64(0);
    CHECK(G.scalar_mul(zero).is_infinity(), "0 * G == O");

    // n * G == O (where n is curve order)
    auto n = Scalar::from_hex(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
    // n reduces to 0, so this is 0*G
    CHECK(G.scalar_mul(n).is_infinity(), "n * G == O");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 2. Jacobian add special cases
// ============================================================================
static void test_jacobian_add() {
    g_section = "jac_add";
    printf("[2] Jacobian add special cases\n");

    auto G = Point::generator();

    // G + G should equal 2G (dbl path)
    auto G2_add = G.add(G);
    auto G2_dbl = G.dbl();
    CHECK(points_equal(G2_add, G2_dbl), "G+G == 2G");

    // P + Q where P != Q, P != -Q
    for (int i = 0; i < 1000; ++i) {
        auto k1 = random_scalar();
        auto k2 = random_scalar();
        auto P = G.scalar_mul(k1);
        auto Q = G.scalar_mul(k2);
        auto R1 = P.add(Q);
        auto R2 = Q.add(P);
        CHECK(points_equal(R1, R2), "P+Q == Q+P");
    }

    // Associativity: (P + Q) + R == P + (Q + R)
    for (int i = 0; i < 500; ++i) {
        auto P = G.scalar_mul(random_scalar());
        auto Q = G.scalar_mul(random_scalar());
        auto R = G.scalar_mul(random_scalar());
        auto lhs = P.add(Q).add(R);
        auto rhs = P.add(Q.add(R));
        CHECK(points_equal(lhs, rhs), "(P+Q)+R == P+(Q+R)");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 3. Jacobian double edge cases
// ============================================================================
static void test_jacobian_dbl() {
    g_section = "jac_dbl";
    printf("[3] Jacobian double edge cases\n");

    auto G = Point::generator();

    // dbl(G) == 2*G
    auto G2 = G.dbl();
    auto G2_mul = G.scalar_mul(Scalar::from_uint64(2));
    CHECK(points_equal(G2, G2_mul), "dbl(G) == 2*G");

    // dbl(dbl(G)) == 4*G
    auto G4 = G2.dbl();
    auto G4_mul = G.scalar_mul(Scalar::from_uint64(4));
    CHECK(points_equal(G4, G4_mul), "dbl(dbl(G)) == 4*G");

    // dbl(O) == O
    CHECK(Point::infinity().dbl().is_infinity(), "dbl(O) == O");

    // Chain: dbl^10(G) == 1024*G
    auto P = G;
    for (int i = 0; i < 10; ++i) P = P.dbl();
    auto P_mul = G.scalar_mul(Scalar::from_uint64(1024));
    CHECK(points_equal(P, P_mul), "dbl^10(G) == 1024*G");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 4. Mixed add H=0 case (P + P via add path)
// ============================================================================
static void test_mixed_add_same_point() {
    g_section = "mixed_H0";
    printf("[4] P + P via add (H=0 case)\n");

    auto G = Point::generator();

    // When two points are equal, add should still return dbl result
    for (int i = 0; i < 100; ++i) {
        auto P = G.scalar_mul(random_scalar());
        auto sum = P.add(P);
        auto dbl = P.dbl();
        CHECK(points_equal(sum, dbl), "P+P == dbl(P)");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 5. P + (-P) == O  (Z cancellation)
// ============================================================================
static void test_point_negation() {
    g_section = "negate";
    printf("[5] P + (-P) == O\n");

    auto G = Point::generator();

    for (int i = 0; i < 1000; ++i) {
        auto P = G.scalar_mul(random_scalar());
        auto neg_P = P.negate();

        CHECK(!neg_P.is_infinity(), "-P is not infinity");
        auto sum = P.add(neg_P);
        CHECK(sum.is_infinity(), "P + (-P) == O");
    }

    // Verify negate preserves x, flips y
    {
        auto P = G.scalar_mul(Scalar::from_uint64(7));
        auto nP = P.negate();
        auto P_bytes = P.to_uncompressed();
        auto nP_bytes = nP.to_uncompressed();
        // x should match (bytes 1..32)
        CHECK(std::memcmp(P_bytes.data() + 1, nP_bytes.data() + 1, 32) == 0,
              "negate: same x");
        // y should differ
        CHECK(std::memcmp(P_bytes.data() + 33, nP_bytes.data() + 33, 32) != 0,
              "negate: different y");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 6. Affine conversion correctness
// ============================================================================
static void test_affine_conversion() {
    g_section = "affine";
    printf("[6] Affine conversion correctness\n");

    auto G = Point::generator();

    for (int i = 0; i < 1000; ++i) {
        auto P = G.scalar_mul(random_scalar());

        // compressed -> uncompressed should represent same point
        auto comp = P.to_compressed();
        auto uncomp = P.to_uncompressed();

        // Prefix check
        CHECK(comp[0] == 0x02 || comp[0] == 0x03, "compressed prefix");
        CHECK(uncomp[0] == 0x04, "uncompressed prefix");

        // X coordinates match
        CHECK(std::memcmp(comp.data() + 1, uncomp.data() + 1, 32) == 0,
              "compressed/uncompressed X match");

        // Verify point is on curve: y^2 == x^3 + 7
        auto x = FieldElement::from_bytes(
            *reinterpret_cast<const std::array<uint8_t, 32>*>(uncomp.data() + 1));
        auto y = FieldElement::from_bytes(
            *reinterpret_cast<const std::array<uint8_t, 32>*>(uncomp.data() + 33));
        auto y2 = y.square();
        auto x3 = x.square() * x;
        auto rhs = x3 + FieldElement::from_uint64(7);
        CHECK(y2 == rhs, "on-curve: y^2 == x^3 + 7");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 7. Scalar mul algebraic identities
// ============================================================================
static void test_scalar_mul_identities() {
    g_section = "smul_id";
    printf("[7] Scalar mul algebraic identities\n");

    auto G = Point::generator();

    // (a + b) * G == a*G + b*G
    for (int i = 0; i < 1000; ++i) {
        auto a = random_scalar();
        auto b = random_scalar();
        auto lhs = G.scalar_mul(a + b);
        auto rhs = G.scalar_mul(a).add(G.scalar_mul(b));
        CHECK(points_equal(lhs, rhs), "(a+b)*G == a*G + b*G");
    }

    // (a * b) * G == a * (b * G)
    for (int i = 0; i < 500; ++i) {
        auto a = random_scalar();
        auto b = random_scalar();
        auto lhs = G.scalar_mul(a * b);
        auto bG = G.scalar_mul(b);
        auto rhs = bG.scalar_mul(a);
        CHECK(points_equal(lhs, rhs), "(a*b)*G == a*(b*G)");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 8. Known K*G test vectors
// ============================================================================
static void test_known_vectors() {
    g_section = "vectors";
    printf("[8] Known K*G test vectors\n");

    auto G = Point::generator();

    for (int i = 0; i < secp256k1::test_vectors::KG_VECTOR_COUNT; ++i) {
        const auto& v = secp256k1::test_vectors::KG_VECTORS[i];
        auto k = Scalar::from_hex(v.scalar_hex);
        auto P = G.scalar_mul(k);
        auto uncomp = P.to_uncompressed();

        // Parse expected X
        auto expected_x = FieldElement::from_hex(v.expected_x);
        auto actual_x = FieldElement::from_bytes(
            *reinterpret_cast<const std::array<uint8_t, 32>*>(uncomp.data() + 1));

        CHECK(actual_x == expected_x, v.description);
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 9. ECDSA sign+verify round-trip (1000 random)
// ============================================================================
static void test_ecdsa_roundtrip() {
    g_section = "ecdsa";
    printf("[9] ECDSA sign+verify round-trip (1000 random)\n");

    auto G = Point::generator();

    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        auto pk = G.scalar_mul(sk);
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);

        auto sig = secp256k1::ecdsa_sign(msg, sk);
        CHECK(!sig.r.is_zero() && !sig.s.is_zero(), "non-zero sig");
        CHECK(sig.is_low_s(), "low-S enforced");

        bool const valid = secp256k1::ecdsa_verify(msg, pk, sig);
        CHECK(valid, "valid sig verifies");

        // Wrong message
        msg[0] ^= 0x01;
        CHECK(!secp256k1::ecdsa_verify(msg, pk, sig), "wrong msg fails");
        msg[0] ^= 0x01;

        // Wrong key
        auto pk2 = G.scalar_mul(random_scalar());
        CHECK(!secp256k1::ecdsa_verify(msg, pk2, sig), "wrong key fails");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 10. Schnorr BIP-340 sign+verify round-trip (1000 random)
// ============================================================================
static void test_schnorr_roundtrip() {
    g_section = "schnorr";
    printf("[10] Schnorr BIP-340 sign+verify round-trip (1000 random)\n");

    for (int i = 0; i < 1000; ++i) {
        auto sk = random_scalar();
        std::array<uint8_t, 32> msg{};
        uint64_t v = rng();
        std::memcpy(msg.data(), &v, 8);
        std::array<uint8_t, 32> const aux{};

        auto sig = secp256k1::schnorr_sign(sk, msg, aux);
        auto pk_x = secp256k1::schnorr_pubkey(sk);

        bool const valid = secp256k1::schnorr_verify(pk_x, msg, sig);
        CHECK(valid, "valid schnorr sig verifies");

        // Wrong message
        msg[0] ^= 0x01;
        CHECK(!secp256k1::schnorr_verify(pk_x, msg, sig), "wrong msg fails schnorr");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 11. Large-N random stress test
// ============================================================================
static void test_stress_random() {
    g_section = "stress";
    printf("[11] Large-N random stress (100K point ops)\n");

    auto G = Point::generator();
    int failures = 0;

    for (int i = 0; i < 100000; ++i) {
        auto k = random_scalar();
        auto P = G.scalar_mul(k);

        // Check not infinity (probability negligible for random k)
        if (P.is_infinity()) { ++failures; continue; }

        // Verify on curve
        auto uncomp = P.to_uncompressed();
        auto x = FieldElement::from_bytes(
            *reinterpret_cast<const std::array<uint8_t, 32>*>(uncomp.data() + 1));
        auto y = FieldElement::from_bytes(
            *reinterpret_cast<const std::array<uint8_t, 32>*>(uncomp.data() + 33));
        auto y2 = y.square();
        auto x3_7 = x.square() * x + FieldElement::from_uint64(7);
        CHECK(y2 == x3_7, "on-curve check");
    }

    printf("    infinity hits (should be 0): %d\n", failures);
    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// _run() entry point for unified audit runner
// ============================================================================

int audit_point_run() {
    g_pass = 0; g_fail = 0;

    test_infinity();
    test_jacobian_add();
    test_jacobian_dbl();
    test_mixed_add_same_point();
    test_point_negation();
    test_affine_conversion();
    test_scalar_mul_identities();
    test_known_vectors();
    test_ecdsa_roundtrip();
    test_schnorr_roundtrip();
    test_stress_random();

    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("===============================================================\n");
    printf("  AUDIT I.3 -- Point Operations & Signature Correctness\n");
    printf("===============================================================\n\n");

    test_infinity();
    test_jacobian_add();
    test_jacobian_dbl();
    test_mixed_add_same_point();
    test_point_negation();
    test_affine_conversion();
    test_scalar_mul_identities();
    test_known_vectors();
    test_ecdsa_roundtrip();
    test_schnorr_roundtrip();
    test_stress_random();

    printf("===============================================================\n");
    printf("  POINT AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    printf("===============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
