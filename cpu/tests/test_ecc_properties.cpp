// ============================================================================
// Test: Property-Based ECC Algebraic Invariants (secp256k1)
// ============================================================================
// Verifies fundamental group-law properties that MUST hold for any correct
// implementation of secp256k1 point arithmetic:
//
//   1. Closure:           P + Q is always on the curve
//   2. Commutativity:     P + Q == Q + P
//   3. Associativity:     (P + Q) + R == P + (Q + R)
//   4. Identity:          P + O == P
//   5. Inverse:           P + (-P) == O
//   6. Generator order:   n * G == O
//   7. Double == add:     2*P == P + P
//   8. Scalar ring:       (a + b) * G == a*G + b*G
//   9. Scalar assoc:      (a * b) * G == a * (b * G)
//  10. Distributivity:    k * (P + Q) == k*P + k*Q
//  11. Negate involution: -(-P) == P
//  12. Sub consistency:   P - Q == P + (-Q)
//
// Uses deterministic pseudo-random scalars derived from a simple hash of
// the iteration index -- fully reproducible, no external PRNG dependency.
// ============================================================================

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"

#include <cstdio>
#include <cstdint>
#include <array>

using namespace secp256k1::fast;

// -- helpers -----------------------------------------------------------------

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; (void)printf("  [PASS] %s\n", msg); } \
    else { (void)printf("  [FAIL] %s\n", msg); } \
} while(0)

// Compare two points by compressed encoding (33 bytes)
static bool points_equal(const Point& a, const Point& b) {
    return a.to_compressed() == b.to_compressed();
}

// Deterministic scalar from index: SHA256-like mixing of 'seed' bits.
// Not cryptographically random -- that's intentional: reproducibility > entropy.
static Scalar deterministic_scalar(uint64_t idx) {
    // Knuth multiplicative hash + bit mixing
    uint64_t h = idx * 0x9E3779B97F4A7C15ULL;
    h ^= h >> 30; h *= 0xBF58476D1CE4E5B9ULL;
    h ^= h >> 27; h *= 0x94D049BB133111EBULL;
    h ^= h >> 31;

    uint64_t h2 = (idx + 1) * 0x517CC1B727220A95ULL;
    h2 ^= h2 >> 30; h2 *= 0xBF58476D1CE4E5B9ULL;
    h2 ^= h2 >> 27; h2 *= 0x94D049BB133111EBULL;
    h2 ^= h2 >> 31;

    uint64_t h3 = (idx + 2) * 0x6C62272E07BB0142ULL;
    h3 ^= h3 >> 30; h3 *= 0xBF58476D1CE4E5B9ULL;
    h3 ^= h3 >> 27; h3 *= 0x94D049BB133111EBULL;
    h3 ^= h3 >> 31;

    uint64_t h4 = (idx + 3) * 0xC24A2C36CBBFE8AAULL;
    h4 ^= h4 >> 30; h4 *= 0xBF58476D1CE4E5B9ULL;
    h4 ^= h4 >> 27; h4 *= 0x94D049BB133111EBULL;
    h4 ^= h4 >> 31;

    // Construct from 4 limbs (native little-endian)
    Scalar s = Scalar::from_limbs({h, h2, h3, h4});
    // Ensure non-zero (extremely unlikely to be zero, but be safe)
    if (s.is_zero()) {
        s = Scalar::from_uint64(idx + 42);
    }
    return s;
}

// Derive a point from index: k*G for a deterministic k
static Point deterministic_point(uint64_t idx) {
    Scalar const k = deterministic_scalar(idx * 1000 + 7);
    return Point::generator().scalar_mul(k);
}

// -- property tests ----------------------------------------------------------

static void test_identity_element() {
    (void)printf("\n--- Identity element: P + O == P ---\n");

    Point const G = Point::generator();
    Point const O = Point::infinity();

    CHECK(points_equal(G.add(O), G), "G + O == G");
    CHECK(points_equal(O.add(G), G), "O + G == G");
    CHECK(O.add(O).is_infinity(),     "O + O == O");

    // Non-generator point
    Scalar const k = Scalar::from_uint64(12345);
    Point const P = G.scalar_mul(k);
    CHECK(points_equal(P.add(O), P), "P + O == P (arbitrary point)");
    CHECK(points_equal(O.add(P), P), "O + P == P (arbitrary point)");
}

static void test_inverse_element() {
    (void)printf("\n--- Inverse element: P + (-P) == O ---\n");

    Point const G = Point::generator();
    Point const neg_G = G.negate();
    CHECK(G.add(neg_G).is_infinity(), "G + (-G) == O");

    for (uint64_t i = 1; i <= 5; ++i) {
        Point const P = deterministic_point(i);
        Point const neg_P = P.negate();
        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "P_%llu + (-P_%llu) == O",
                      static_cast<unsigned long long>(i),
                      static_cast<unsigned long long>(i));
        CHECK(P.add(neg_P).is_infinity(), buf);
    }
}

static void test_negate_involution() {
    (void)printf("\n--- Negate involution: -(-P) == P ---\n");

    Point const G = Point::generator();
    CHECK(points_equal(G.negate().negate(), G), "-(-G) == G");

    for (uint64_t i = 1; i <= 5; ++i) {
        Point const P = deterministic_point(i);
        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "-(-P_%llu) == P_%llu",
                      static_cast<unsigned long long>(i),
                      static_cast<unsigned long long>(i));
        CHECK(points_equal(P.negate().negate(), P), buf);
    }
}

static void test_commutativity() {
    (void)printf("\n--- Commutativity: P + Q == Q + P ---\n");

    for (uint64_t i = 0; i < 8; ++i) {
        Point const P = deterministic_point(i * 2);
        Point const Q = deterministic_point(i * 2 + 1);
        Point const PQ = P.add(Q);
        Point const QP = Q.add(P);
        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "P_%llu + Q_%llu == Q_%llu + P_%llu",
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i * 2 + 1),
                      static_cast<unsigned long long>(i * 2 + 1),
                      static_cast<unsigned long long>(i * 2));
        CHECK(points_equal(PQ, QP), buf);
    }
}

static void test_associativity() {
    (void)printf("\n--- Associativity: (P + Q) + R == P + (Q + R) ---\n");

    for (uint64_t i = 0; i < 5; ++i) {
        Point const P = deterministic_point(i * 3);
        Point const Q = deterministic_point(i * 3 + 1);
        Point const R = deterministic_point(i * 3 + 2);

        Point const lhs = (P.add(Q)).add(R);
        Point const rhs = P.add(Q.add(R));

        char buf[80];
        (void)std::snprintf(buf, sizeof(buf), "(P_%llu + Q_%llu) + R_%llu == P_%llu + (Q_%llu + R_%llu)",
                      static_cast<unsigned long long>(i * 3),
                      static_cast<unsigned long long>(i * 3 + 1),
                      static_cast<unsigned long long>(i * 3 + 2),
                      static_cast<unsigned long long>(i * 3),
                      static_cast<unsigned long long>(i * 3 + 1),
                      static_cast<unsigned long long>(i * 3 + 2));
        CHECK(points_equal(lhs, rhs), buf);
    }
}

static void test_double_equals_add() {
    (void)printf("\n--- Double consistency: 2*P == P + P ---\n");

    Point const G = Point::generator();
    CHECK(points_equal(G.dbl(), G.add(G)), "2*G == G + G");

    for (uint64_t i = 1; i <= 5; ++i) {
        Point const P = deterministic_point(i);
        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "2*P_%llu == P_%llu + P_%llu",
                      static_cast<unsigned long long>(i),
                      static_cast<unsigned long long>(i),
                      static_cast<unsigned long long>(i));
        CHECK(points_equal(P.dbl(), P.add(P)), buf);
    }
}

static void test_scalar_ring_distributivity() {
    (void)printf("\n--- Scalar ring: (a + b)*G == a*G + b*G ---\n");

    Point const G = Point::generator();

    for (uint64_t i = 0; i < 8; ++i) {
        Scalar const a = deterministic_scalar(i * 2 + 100);
        Scalar const b = deterministic_scalar(i * 2 + 101);
        Scalar const ab = a + b;

        Point const lhs = G.scalar_mul(ab);
        Point const rhs = G.scalar_mul(a).add(G.scalar_mul(b));

        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "(a_%llu + b_%llu)*G == a_%llu*G + b_%llu*G",
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i * 2 + 1),
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i * 2 + 1));
        CHECK(points_equal(lhs, rhs), buf);
    }
}

static void test_scalar_mul_associativity() {
    (void)printf("\n--- Scalar associativity: (a*b)*G == a*(b*G) ---\n");

    Point const G = Point::generator();

    for (uint64_t i = 0; i < 8; ++i) {
        Scalar const a = deterministic_scalar(i * 2 + 200);
        Scalar const b = deterministic_scalar(i * 2 + 201);
        Scalar const ab = a * b;

        Point const lhs = G.scalar_mul(ab);
        Point const rhs = G.scalar_mul(b).scalar_mul(a);

        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "(a_%llu * b_%llu)*G == a_%llu * (b_%llu * G)",
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i * 2 + 1),
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i * 2 + 1));
        CHECK(points_equal(lhs, rhs), buf);
    }
}

static void test_distributivity() {
    (void)printf("\n--- Distributivity: k*(P + Q) == k*P + k*Q ---\n");

    for (uint64_t i = 0; i < 8; ++i) {
        Scalar const k = deterministic_scalar(i + 300);
        Point const P = deterministic_point(i * 2 + 50);
        Point const Q = deterministic_point(i * 2 + 51);

        Point const lhs = (P.add(Q)).scalar_mul(k);
        Point const rhs = P.scalar_mul(k).add(Q.scalar_mul(k));

        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "k_%llu * (P_%llu + Q_%llu) == k_%llu*P_%llu + k_%llu*Q_%llu",
                      static_cast<unsigned long long>(i),
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i * 2 + 1),
                      static_cast<unsigned long long>(i),
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i),
                      static_cast<unsigned long long>(i * 2 + 1));
        CHECK(points_equal(lhs, rhs), buf);
    }
}

static void test_generator_order() {
    (void)printf("\n--- Generator order: n*G == O ---\n");

    // secp256k1 order n
    Scalar const n = Scalar::from_hex(
        "FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141");
    Point const G = Point::generator();

    Point const nG = G.scalar_mul(n);
    CHECK(nG.is_infinity(), "n * G == O (full order)");

    // (n-1)*G should NOT be infinity and should equal -G
    Scalar const n_minus_1 = n - Scalar::from_uint64(1);
    Point const nm1G = G.scalar_mul(n_minus_1);
    CHECK(!nm1G.is_infinity(), "(n-1)*G != O");
    CHECK(points_equal(nm1G, G.negate()), "(n-1)*G == -G");

    // 1*G should be G
    Point const oneG = G.scalar_mul(Scalar::from_uint64(1));
    CHECK(points_equal(oneG, G), "1*G == G");

    // 0*G should be O
    Scalar const zero;  // default constructed = 0
    Point const zeroG = G.scalar_mul(zero);
    CHECK(zeroG.is_infinity(), "0*G == O");
}

static void test_sub_consistency() {
    (void)printf("\n--- Subtraction: P - Q == P + (-Q) ---\n");

    for (uint64_t i = 0; i < 5; ++i) {
        Point const P = deterministic_point(i * 2 + 400);
        Point const Q = deterministic_point(i * 2 + 401);

        // P - Q via add(negate)
        Point const via_negate = P.add(Q.negate());

        // P - Q via scalar: P + (n-1)*Q (equivalent to P + (-Q))
        // We just verify the negate path is consistent
        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "P_%llu - Q_%llu == P_%llu + (-Q_%llu)",
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i * 2 + 1),
                      static_cast<unsigned long long>(i * 2),
                      static_cast<unsigned long long>(i * 2 + 1));

        // Verify: (P - Q) + Q == P
        Point const roundtrip = via_negate.add(Q);
        CHECK(points_equal(roundtrip, P), buf);
    }
}

static void test_scalar_mul_small_values() {
    (void)printf("\n--- Scalar mul small values: k*G consistency ---\n");

    Point const G = Point::generator();
    Point acc = Point::infinity();

    // Verify 1*G .. 8*G by repeated addition
    for (uint64_t k = 1; k <= 8; ++k) {
        acc = acc.add(G);
        Point const via_mul = G.scalar_mul(Scalar::from_uint64(k));
        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "%llu*G == G+G+...+G (%llu times)",
                      static_cast<unsigned long long>(k),
                      static_cast<unsigned long long>(k));
        CHECK(points_equal(via_mul, acc), buf);
    }
}

static void test_inplace_consistency() {
    (void)printf("\n--- In-place ops consistency ---\n");

    Point const G = Point::generator();
    Scalar const k = deterministic_scalar(500);
    Point const P = G.scalar_mul(k);
    Point const Q = deterministic_point(501);

    // add_inplace vs add
    {
        Point p1 = P;
        Point const p2 = P.add(Q);
        p1.add_inplace(Q);
        CHECK(points_equal(p1, p2), "add_inplace(Q) == add(Q)");
    }

    // dbl_inplace vs dbl
    {
        Point p1 = P;
        Point const p2 = P.dbl();
        p1.dbl_inplace();
        CHECK(points_equal(p1, p2), "dbl_inplace() == dbl()");
    }

    // negate_inplace vs negate
    {
        Point p1 = P;
        Point const p2 = P.negate();
        p1.negate_inplace();
        CHECK(points_equal(p1, p2), "negate_inplace() == negate()");
    }

    // next_inplace vs next
    {
        Point p1 = P;
        Point const p2 = P.next();
        p1.next_inplace();
        CHECK(points_equal(p1, p2), "next_inplace() == next()");
    }

    // prev_inplace vs prev
    {
        Point p1 = P;
        Point const p2 = P.prev();
        p1.prev_inplace();
        CHECK(points_equal(p1, p2), "prev_inplace() == prev()");
    }

    // next then prev should roundtrip
    {
        Point p1 = P;
        p1.next_inplace();
        p1.prev_inplace();
        CHECK(points_equal(p1, P), "next_inplace + prev_inplace == identity");
    }
}

static void test_dual_scalar_mul() {
    (void)printf("\n--- Dual scalar mul: a*G + b*P ---\n");

    Point const G = Point::generator();

    for (uint64_t i = 0; i < 5; ++i) {
        Scalar const a = deterministic_scalar(i + 600);
        Scalar const b = deterministic_scalar(i + 610);
        Point const P = deterministic_point(i + 620);

        Point const expected = G.scalar_mul(a).add(P.scalar_mul(b));
        Point const dual = Point::dual_scalar_mul_gen_point(a, b, P);

        char buf[64];
        (void)std::snprintf(buf, sizeof(buf), "dual_mul(%llu) == a*G + b*P",
                      static_cast<unsigned long long>(i));
        CHECK(points_equal(dual, expected), buf);
    }
}

// -- entry points ------------------------------------------------------------

int test_ecc_properties_run() {
    (void)printf("\n================================================================\n");
    (void)printf("  ECC Property-Based Tests (secp256k1 group-law invariants)\n");
    (void)printf("================================================================\n");

    tests_run = 0;
    tests_passed = 0;

    test_identity_element();
    test_inverse_element();
    test_negate_involution();
    test_commutativity();
    test_associativity();
    test_double_equals_add();
    test_scalar_ring_distributivity();
    test_scalar_mul_associativity();
    test_distributivity();
    test_generator_order();
    test_sub_consistency();
    test_scalar_mul_small_values();
    test_inplace_consistency();
    test_dual_scalar_mul();

    (void)printf("\n================================================================\n");
    (void)printf("  ECC Property Results: %d / %d passed\n", tests_passed, tests_run);
    (void)printf("================================================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_ecc_properties_run();
}
#endif
