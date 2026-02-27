// ============================================================================
// Test: Point output function edge cases (infinity + Z=0 defensive guards)
// ============================================================================
// Exercises all Point output functions with:
//   - Point::infinity()       (the canonical infinity representation)
//   - P + (-P) = infinity     (computed infinity through group law)
//   - Normal point G          (baseline sanity check)
//
// Ensures to_compressed(), to_uncompressed(), x(), y(), has_even_y(),
// and x_bytes_and_parity() all behave correctly on edge-case inputs
// WITHOUT crashing or invoking undefined behavior.
// ============================================================================

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"

#include <cstdio>
#include <cstdint>
#include <array>

using namespace secp256k1::fast;

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, msg) do { \
    ++tests_run; \
    if (cond) { ++tests_passed; printf("  [PASS] %s\n", msg); } \
    else { printf("  [FAIL] %s\n", msg); } \
} while(0)

// Check that all 33 bytes are zero (infinity encoding)
static bool is_zero_33(const std::array<uint8_t, 33>& a) {
    for (auto b : a) if (b != 0) return false;
    return true;
}

// Check that all 65 bytes are zero (infinity encoding)
static bool is_zero_65(const std::array<uint8_t, 65>& a) {
    for (auto b : a) if (b != 0) return false;
    return true;
}

// All 32 bytes zero
static bool is_zero_32(const std::array<uint8_t, 32>& a) {
    for (auto b : a) if (b != 0) return false;
    return true;
}

// -- infinity tests ----------------------------------------------------------

static void test_infinity_outputs() {
    printf("\n=== Infinity point output functions ===\n");
    const Point inf = Point::infinity();

    // to_compressed: should return all zeros
    const auto comp = inf.to_compressed();
    CHECK(is_zero_33(comp), "infinity to_compressed -> all zeros");

    // to_uncompressed: should return all zeros
    const auto uncomp = inf.to_uncompressed();
    CHECK(is_zero_65(uncomp), "infinity to_uncompressed -> all zeros");

    // x(): should be zero field element
    const FieldElement xv = inf.x();
    CHECK(xv == FieldElement::zero(), "infinity x() -> zero");

    // y(): should be zero field element
    const FieldElement yv = inf.y();
    CHECK(yv == FieldElement::zero(), "infinity y() -> zero");

    // has_even_y: infinity -> false
    CHECK(inf.has_even_y() == false, "infinity has_even_y -> false");

    // x_bytes_and_parity: infinity -> (zeros, false)
    auto [xb, parity] = inf.x_bytes_and_parity();
    CHECK(is_zero_32(xb), "infinity x_bytes_and_parity x -> zeros");
    CHECK(parity == false, "infinity x_bytes_and_parity parity -> false");

    // is_infinity flag
    CHECK(inf.is_infinity(), "infinity is_infinity -> true");
}

// -- computed infinity via P + (-P) ------------------------------------------

static void test_computed_infinity() {
    printf("\n=== Computed infinity (P + (-P)) output functions ===\n");
    const Point G = Point::generator();
    const Point negG = G.negate();
    const Point sum = G.add(negG);  // should be infinity

    CHECK(sum.is_infinity(), "G + (-G) is infinity");

    const auto comp = sum.to_compressed();
    CHECK(is_zero_33(comp), "G+(-G) to_compressed -> all zeros");

    const auto uncomp = sum.to_uncompressed();
    CHECK(is_zero_65(uncomp), "G+(-G) to_uncompressed -> all zeros");

    const FieldElement xv = sum.x();
    CHECK(xv == FieldElement::zero(), "G+(-G) x() -> zero");

    const FieldElement yv = sum.y();
    CHECK(yv == FieldElement::zero(), "G+(-G) y() -> zero");

    CHECK(sum.has_even_y() == false, "G+(-G) has_even_y -> false");

    auto [xb, parity] = sum.x_bytes_and_parity();
    CHECK(is_zero_32(xb), "G+(-G) x_bytes_and_parity x -> zeros");
    CHECK(parity == false, "G+(-G) x_bytes_and_parity parity -> false");
}

// -- normal point G (sanity baseline) ----------------------------------------

static void test_generator_outputs() {
    printf("\n=== Generator point output functions ===\n");
    const Point G = Point::generator();

    CHECK(!G.is_infinity(), "G is not infinity");

    auto comp = G.to_compressed();
    // G compressed starts with 0x02 (even y)
    CHECK(comp[0] == 0x02 || comp[0] == 0x03, "G compressed prefix valid");
    CHECK(!is_zero_33(comp), "G compressed is nonzero");

    auto uncomp = G.to_uncompressed();
    CHECK(uncomp[0] == 0x04, "G uncompressed prefix 0x04");
    CHECK(!is_zero_65(uncomp), "G uncompressed is nonzero");

    const FieldElement xv = G.x();
    CHECK(!(xv == FieldElement::zero()), "G x() is nonzero");

    const FieldElement yv = G.y();
    CHECK(!(yv == FieldElement::zero()), "G y() is nonzero");

    // G has even y (known property of secp256k1 generator)
    CHECK(G.has_even_y() == true, "G has_even_y -> true");

    auto [xb, parity] = G.x_bytes_and_parity();
    CHECK(!is_zero_32(xb), "G x_bytes nonzero");
    // parity=false means even y for G
    CHECK(parity == false, "G x_bytes_and_parity parity -> false (even y)");
}

// -- scalar mul edge cases ---------------------------------------------------

static void test_scalar_mul_edge_cases() {
    printf("\n=== Scalar multiplication edge cases ===\n");
    const Point G = Point::generator();

    // 0 * G = infinity
    const Scalar zero_s = Scalar::from_uint64(0);
    const Point p0 = G.scalar_mul(zero_s);
    CHECK(p0.is_infinity(), "0*G is infinity");
    auto comp0 = p0.to_compressed();
    CHECK(is_zero_33(comp0), "0*G compressed -> zeros");

    // 1 * G = G
    const Scalar one_s = Scalar::from_uint64(1);
    const Point p1 = G.scalar_mul(one_s);
    CHECK(!p1.is_infinity(), "1*G is not infinity");
    CHECK(p1.to_compressed() == G.to_compressed(), "1*G == G");

    // n * G = infinity (group order)
    // n = FFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE BAAEDCE6 AF48A03B BFD25E8C D0364141
    const Scalar n = Scalar::from_limbs({
        0xBFD25E8CD0364141ULL,
        0xBAAEDCE6AF48A03BULL,
        0xFFFFFFFFFFFFFFFEULL,
        0xFFFFFFFFFFFFFFFFULL
    });
    const Point pn = G.scalar_mul(n);
    CHECK(pn.is_infinity(), "n*G is infinity");
    auto compn = pn.to_compressed();
    CHECK(is_zero_33(compn), "n*G compressed -> zeros");
    CHECK(pn.has_even_y() == false, "n*G has_even_y -> false");
}

// -- roundtrip: compressed encode consistency ----------------------------

static void test_roundtrip() {
    printf("\n=== Roundtrip compressed encoding ===\n");
    const Point G = Point::generator();

    // G compressed twice should give same bytes
    auto comp1 = G.to_compressed();
    auto comp2 = G.to_compressed();
    CHECK(comp1 == comp2, "G double-compress consistency");

    // 2*G = G + G
    const Point G2 = G.add(G);
    auto comp2a = G2.to_compressed();
    auto comp2b = G2.to_compressed();
    CHECK(comp2a == comp2b, "2*G double-compress consistency");

    // Uncompressed consistency
    auto uncomp1 = G.to_uncompressed();
    auto uncomp2 = G.to_uncompressed();
    CHECK(uncomp1 == uncomp2, "G double-uncompress consistency");

    // x() and y() consistency
    const FieldElement x1 = G.x();
    const FieldElement x2 = G.x();
    CHECK(x1 == x2, "G x() consistency");

    const FieldElement y1 = G.y();
    const FieldElement y2 = G.y();
    CHECK(y1 == y2, "G y() consistency");

    // has_even_y consistency
    const bool e1 = G.has_even_y();
    const bool e2 = G.has_even_y();
    CHECK(e1 == e2, "G has_even_y consistency");

    // x_bytes_and_parity consistency
    auto [xb1, p1] = G.x_bytes_and_parity();
    auto [xb2, p2] = G.x_bytes_and_parity();
    CHECK(xb1 == xb2, "G x_bytes consistency");
    CHECK(p1 == p2, "G parity consistency");
}

// ============================================================================

int main() {
    printf("Point edge-case tests\n");
    printf("=====================\n");

    test_infinity_outputs();
    test_computed_infinity();
    test_generator_outputs();
    test_scalar_mul_edge_cases();
    test_roundtrip();

    printf("\n-----\nResults: %d / %d passed\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
