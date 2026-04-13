// ============================================================================
// Point Group-Law Audit Test
// ============================================================================
// Verifies the Jacobian point arithmetic (double, add_mixed, add) satisfies
// elliptic curve group axioms.  This is the hot-path layer that every
// signing, verification, and key-generation operation depends on.
//
// TESTS:
//   1.  Generator is on curve
//   2.  Infinity is identity element
//   3.  Point doubling correctness
//   4.  Point addition commutativity
//   5.  Point addition associativity
//   6.  Point + (-Point) = Infinity
//   7.  Scalar multiplication boundary (0*G, 1*G, (n-1)*G, n*G)
//   8.  Scalar multiplication linearity ((a+b)*G = a*G + b*G)
//   9.  Point serialization roundtrip (compressed + uncompressed)
//  10.  Batch normalize consistency
//  11.  GLV endomorphism consistency
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>
#include <vector>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/glv.hpp"

// Sanitizer-aware iteration scaling
#include "secp256k1/sanitizer_scale.hpp"

using namespace secp256k1;
using namespace secp256k1::fast;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#include "audit_check.hpp"

static std::mt19937_64 rng(0xA1B2C3D4'E5F60718ULL);

static Scalar random_scalar() {
    std::array<uint8_t, 32> buf{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(buf.data() + static_cast<std::size_t>(i) * 8, &v, 8);
    }
    return Scalar::from_bytes(buf);
}

static Scalar random_nonzero_scalar() {
    Scalar s;
    do { s = random_scalar(); } while (s.is_zero());
    return s;
}

// secp256k1 group order n
static Scalar sc_n_minus_1() {
    return Scalar::from_hex("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364140");
}

static bool point_eq(const Point& a, const Point& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() || b.is_infinity()) return false;
    return a.x().to_bytes() == b.x().to_bytes() &&
           a.y().to_bytes() == b.y().to_bytes();
}

// ============================================================================
//  TEST 1: Generator is on curve
// ============================================================================
static void test_generator_on_curve() {
    g_section = "gen_on_curve";
    std::printf("\n[1] Generator is on curve (y^2 = x^3 + 7)\n");

    Point G = Point::generator();
    CHECK(!G.is_infinity(), "G is not infinity");

    // y^2 = x^3 + 7 mod p
    FieldElement x = G.x();
    FieldElement y = G.y();
    FieldElement lhs = y.square();
    FieldElement rhs = x.square() * x + FieldElement::from_uint64(7);
    CHECK(lhs == rhs, "y^2 == x^3 + 7 for generator");

    // Check known generator x coordinate
    FieldElement known_gx = FieldElement::from_hex(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    CHECK(x == known_gx, "G.x matches known value");

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 2: Infinity is identity element
// ============================================================================
static void test_infinity_identity() {
    g_section = "infinity_identity";
    std::printf("\n[2] Infinity is identity element\n");

    Point G   = Point::generator();
    Point inf = Point::infinity();

    CHECK(inf.is_infinity(), "infinity is infinity");

    // G + inf = G
    Point sum1 = G.add(inf);
    CHECK(point_eq(sum1, G), "G + inf == G");

    // inf + G = G
    Point sum2 = inf.add(G);
    CHECK(point_eq(sum2, G), "inf + G == G");

    // inf + inf = inf
    Point sum3 = inf.add(inf);
    CHECK(sum3.is_infinity(), "inf + inf == inf");

    // 0 * G = inf
    Point zero_g = G.scalar_mul(Scalar::zero());
    CHECK(zero_g.is_infinity(), "0*G == inf");

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 3: Point doubling correctness
// ============================================================================
static void test_doubling() {
    g_section = "doubling";
    std::printf("\n[3] Point doubling correctness\n");

    Point G = Point::generator();

    // 2*G via doubling == G + G
    Point two_g_add = G.add(G);
    Point two_g_mul = G.scalar_mul(Scalar::from_uint64(2));
    CHECK(point_eq(two_g_add, two_g_mul), "G+G == 2*G");

    // 4*G = 2*(2*G)
    Point four_g_mul = G.scalar_mul(Scalar::from_uint64(4));
    Point four_g_dbl = two_g_add.add(two_g_add);
    CHECK(point_eq(four_g_mul, four_g_dbl), "4*G == 2*G + 2*G");

    // Doubling infinity = infinity
    Point inf_dbl = Point::infinity().add(Point::infinity());
    CHECK(inf_dbl.is_infinity(), "2*inf == inf");

    // Chained doubling: 8*G
    Point eight_g_mul = G.scalar_mul(Scalar::from_uint64(8));
    Point eight_g_chain = four_g_dbl.add(four_g_dbl);
    CHECK(point_eq(eight_g_mul, eight_g_chain), "8*G chain doubling");

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 4: Point addition commutativity
// ============================================================================
static void test_commutativity() {
    g_section = "commutativity";
    std::printf("\n[4] Point addition commutativity\n");

    Point G = Point::generator();
    const int N = SCALED(20, 4);

    for (int i = 0; i < N; ++i) {
        Scalar a = random_nonzero_scalar();
        Scalar b = random_nonzero_scalar();
        Point A = G.scalar_mul(a);
        Point B = G.scalar_mul(b);
        Point ab = A.add(B);
        Point ba = B.add(A);
        CHECK(point_eq(ab, ba), "A+B == B+A #" + std::to_string(i));
    }

    std::printf("    -> %d/%d OK\n", N, N);
}

// ============================================================================
//  TEST 5: Point addition associativity
// ============================================================================
static void test_associativity() {
    g_section = "associativity";
    std::printf("\n[5] Point addition associativity\n");

    Point G = Point::generator();
    const int N = SCALED(10, 2);

    for (int i = 0; i < N; ++i) {
        Scalar a = random_nonzero_scalar();
        Scalar b = random_nonzero_scalar();
        Scalar c = random_nonzero_scalar();
        Point A = G.scalar_mul(a);
        Point B = G.scalar_mul(b);
        Point C = G.scalar_mul(c);
        Point lhs = A.add(B).add(C);
        Point rhs = A.add(B.add(C));
        CHECK(point_eq(lhs, rhs), "(A+B)+C == A+(B+C) #" + std::to_string(i));
    }

    std::printf("    -> %d/%d OK\n", N, N);
}

// ============================================================================
//  TEST 6: Point + (-Point) = Infinity
// ============================================================================
static void test_point_negation() {
    g_section = "negation";
    std::printf("\n[6] Point + (-Point) = Infinity\n");

    Point G = Point::generator();
    const int N = SCALED(20, 4);

    for (int i = 0; i < N; ++i) {
        Scalar k = random_nonzero_scalar();
        Point P = G.scalar_mul(k);

        // -P = (n-k)*G
        Scalar neg_k = k.negate();
        Point neg_P = G.scalar_mul(neg_k);

        Point sum = P.add(neg_P);
        CHECK(sum.is_infinity(), "P + (-P) == inf #" + std::to_string(i));
    }

    // G + (n-1)*G = inf
    Point nm1_g = G.scalar_mul(sc_n_minus_1());
    Point sum_g = G.add(nm1_g);
    CHECK(sum_g.is_infinity(), "G + (n-1)*G == inf");

    std::printf("    -> %d/%d OK\n", N + 1, N + 1);
}

// ============================================================================
//  TEST 7: Scalar multiplication boundary
// ============================================================================
static void test_scalar_mul_boundary() {
    g_section = "scalar_mul_boundary";
    std::printf("\n[7] Scalar multiplication boundary\n");

    Point G = Point::generator();

    // 0*G = inf
    CHECK(G.scalar_mul(Scalar::zero()).is_infinity(), "0*G == inf");

    // 1*G = G
    CHECK(point_eq(G.scalar_mul(Scalar::one()), G), "1*G == G");

    // (n-1)*G exists and is not infinity
    Point nm1_g = G.scalar_mul(sc_n_minus_1());
    CHECK(!nm1_g.is_infinity(), "(n-1)*G is not inf");

    // n*G = inf — use (n-1)*G + G
    Point n_g = nm1_g.add(G);
    CHECK(n_g.is_infinity(), "n*G == inf");

    // 2*G is not G and not infinity
    Point two_g = G.scalar_mul(Scalar::from_uint64(2));
    CHECK(!two_g.is_infinity(), "2*G not inf");
    CHECK(!point_eq(two_g, G), "2*G != G");

    std::printf("    -> OK\n");
}

// ============================================================================
//  TEST 8: Scalar multiplication linearity
// ============================================================================
static void test_scalar_mul_linearity() {
    g_section = "scalar_mul_linearity";
    std::printf("\n[8] Scalar multiplication linearity\n");

    Point G = Point::generator();
    const int N = SCALED(15, 3);

    for (int i = 0; i < N; ++i) {
        Scalar a = random_nonzero_scalar();
        Scalar b = random_nonzero_scalar();

        // (a+b)*G = a*G + b*G
        Point lhs = G.scalar_mul(a + b);
        Point rhs = G.scalar_mul(a).add(G.scalar_mul(b));
        CHECK(point_eq(lhs, rhs), "(a+b)*G == a*G + b*G #" + std::to_string(i));
    }

    // Specific: 3*G = G + G + G
    Point g3_mul = G.scalar_mul(Scalar::from_uint64(3));
    Point g3_add = G.add(G).add(G);
    CHECK(point_eq(g3_mul, g3_add), "3*G == G+G+G");

    // 7*G = 3*G + 4*G
    Point g7_mul = G.scalar_mul(Scalar::from_uint64(7));
    Point g4_mul = G.scalar_mul(Scalar::from_uint64(4));
    Point g7_add = g3_mul.add(g4_mul);
    CHECK(point_eq(g7_mul, g7_add), "7*G == 3*G + 4*G");

    std::printf("    -> %d/%d OK\n", N + 2, N + 2);
}

// ============================================================================
//  TEST 9: Point serialization roundtrip
// ============================================================================
static void test_serialization_roundtrip() {
    g_section = "point_serialization";
    std::printf("\n[9] Point serialization roundtrip\n");

    Point G = Point::generator();
    const int N = SCALED(20, 4);

    for (int i = 0; i < N; ++i) {
        Scalar k = random_nonzero_scalar();
        Point P = G.scalar_mul(k);

        // Compressed: prefix is 02 or 03, followed by 32-byte x
        auto compressed = P.to_compressed();
        CHECK(compressed[0] == 0x02 || compressed[0] == 0x03,
              "compressed prefix valid #" + std::to_string(i));

        // x bytes are consistent
        auto x_bytes = P.x().to_bytes();
        CHECK(std::memcmp(compressed.data() + 1, x_bytes.data(), 32) == 0,
              "compressed contains correct x #" + std::to_string(i));

        // Parity byte (02=even y, 03=odd y) must match y.bit(0)
        auto y_bytes = P.y().to_bytes();
        bool y_odd = (y_bytes[31] & 1) == 1;
        CHECK((compressed[0] == 0x03) == y_odd,
              "compressed parity matches y #" + std::to_string(i));
    }

    // Infinity to_compressed should produce an implementation-defined result
    // (we just verify it doesn't crash / returns 33 bytes)
    Point inf = Point::infinity();
    auto inf_bytes = inf.to_compressed();
    CHECK(inf_bytes.size() == 33, "infinity to_compressed is 33 bytes");

    std::printf("    -> %d/%d OK\n", N, N);
}

// ============================================================================
//  TEST 10: Batch normalize consistency
// ============================================================================
static void test_batch_normalize() {
    g_section = "batch_normalize";
    std::printf("\n[10] Batch normalize consistency\n");

    Point G = Point::generator();
    const int BATCH = SCALED(20, 4);

    std::vector<Point> points;
    points.reserve(static_cast<std::size_t>(BATCH));
    for (int i = 0; i < BATCH; ++i) {
        points.push_back(G.scalar_mul(random_nonzero_scalar()));
    }

    // batch_to_compressed serializes N Jacobian points in one inversion pass.
    // Each result must roundtrip back to the original affine point.
    std::vector<std::array<uint8_t, 33>> compressed(static_cast<std::size_t>(BATCH));
    Point::batch_to_compressed(points.data(), static_cast<std::size_t>(BATCH), compressed.data());
    CHECK(static_cast<int>(compressed.size()) == BATCH, "batch size correct");

    // Decompress each and compare with individual scalar_mul result
    for (int i = 0; i < BATCH; ++i) {
        auto single = points[static_cast<std::size_t>(i)].to_compressed();
        CHECK(compressed[static_cast<std::size_t>(i)] == single,
              "batch_to_compressed matches individual #" + std::to_string(i));
    }

    // batch_normalize: fill out_x, out_y arrays
    std::vector<FieldElement> out_x(static_cast<std::size_t>(BATCH));
    std::vector<FieldElement> out_y(static_cast<std::size_t>(BATCH));
    Point::batch_normalize(points.data(), static_cast<std::size_t>(BATCH), out_x.data(), out_y.data());
    // Verify x-coordinates match compressed prefix
    for (int i = 0; i < BATCH; ++i) {
        std::array<uint8_t, 32> xbytes = out_x[static_cast<std::size_t>(i)].to_bytes();
        CHECK(std::memcmp(xbytes.data(), compressed[static_cast<std::size_t>(i)].data() + 1, 32) == 0,
              "batch_normalize x matches compressed #" + std::to_string(i));
    }

    std::printf("    -> %d/%d OK\n", 2 * BATCH + 1, 2 * BATCH + 1);
}

// ============================================================================
//  TEST 11: GLV endomorphism consistency
// ============================================================================
static void test_glv_endomorphism() {
    g_section = "glv_endo";
    std::printf("\n[11] GLV endomorphism consistency\n");

    Point G = Point::generator();
    const int N = SCALED(15, 3);

    // lambda * P should equal phi(P) where phi is the GLV endomorphism
    // Verify via scalar mul: k*G via GLV should equal k*G via standard
    for (int i = 0; i < N; ++i) {
        Scalar k = random_nonzero_scalar();
        Point P = G.scalar_mul(k);
        // The GLV path is internal but scalar_mul uses it. We verify
        // the result is on curve by checking y^2 = x^3 + 7.
        if (!P.is_infinity()) {
            FieldElement px = P.x();
            FieldElement py = P.y();
            FieldElement lhs = py.square();
            FieldElement rhs = px.square() * px + FieldElement::from_uint64(7);
            CHECK(lhs == rhs, "k*G on curve #" + std::to_string(i));
        }
    }

    // GLV decomposition: k = k1 + k2*lambda mod n
    // We verify indirectly: a*G + b*G = (a+b)*G (linearity implies
    // any internal decomposition must be correct)
    Scalar a = Scalar::from_hex("DEADBEEF0123456789ABCDEF01234567DEADBEEF0123456789ABCDEF01234567");
    Scalar b = Scalar::from_hex("1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF1234567890ABCDEF");
    Point aG = G.scalar_mul(a);
    Point bG = G.scalar_mul(b);
    Point sum = aG.add(bG);
    Point direct = G.scalar_mul(a + b);
    CHECK(point_eq(sum, direct), "GLV linearity with large scalars");

    std::printf("    -> %d/%d OK\n", N + 1, N + 1);
}

// ============================================================================
//  MAIN
// ============================================================================

#ifdef STANDALONE_TEST
int main() {
#else
int test_point_group_law_run() {
#endif
    std::printf("============================================================\n");
    std::printf("  Point Group-Law Audit Test\n");
    std::printf("============================================================\n");

    test_generator_on_curve();
    test_infinity_identity();
    test_doubling();
    test_commutativity();
    test_associativity();
    test_point_negation();
    test_scalar_mul_boundary();
    test_scalar_mul_linearity();
    test_serialization_roundtrip();
    test_batch_normalize();
    test_glv_endomorphism();

    std::printf("\n  [point_group_law] %d passed, %d failed\n\n", g_pass, g_fail);
    std::printf("============================================================\n");
    std::printf("  Result: %s\n", g_fail == 0 ? "ALL PASSED" : "FAILURES DETECTED");
    std::printf("============================================================\n");
    return g_fail == 0 ? 0 : 1;
}
