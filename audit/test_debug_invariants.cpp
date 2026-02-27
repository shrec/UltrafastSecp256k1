// ============================================================================
// Debug Invariant Assertions Test
// Phase V, Task 5.3.3 -- Verify invariant checking works in debug builds
// ============================================================================
// Tests that:
//   1. is_normalized_field_element correctly identifies canonical FE
//   2. is_on_curve correctly validates curve membership
//   3. is_valid_scalar correctly checks range
//   4. All operations produce results that pass invariant checks
//   5. Debug counters accumulate correctly
// ============================================================================

// Force invariants ON even if NDEBUG is set (for testing purposes)
#define SECP256K1_FORCE_INVARIANTS 1
#undef NDEBUG

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/debug_invariants.hpp"

using namespace secp256k1::fast;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        (void)printf("  FAIL [%s]: %s (line %d)\n", g_section, msg, __LINE__); \
        ++g_fail; \
    } else { \
        ++g_pass; \
    } \
} while(0)

// ============================================================================
// 1. Field element normalization check
// ============================================================================
static void test_fe_normalization() {
    g_section = "fe_norm";
    (void)printf("[1] Field element normalization invariant\n");

    // Zero is normalized
    CHECK(debug::is_normalized_field_element(FieldElement::zero()),
          "zero must be canonical");

    // One is normalized
    CHECK(debug::is_normalized_field_element(FieldElement::one()),
          "one must be canonical");

    // Small values are normalized
    for (uint64_t v = 2; v < 100; ++v) {
        auto fe = FieldElement::from_uint64(v);
        CHECK(debug::is_normalized_field_element(fe),
              "small values must be canonical");
    }

    // random FE from proper constructors should always be normalized
    std::mt19937_64 rng(42);  // NOLINT(cert-msc32-c,cert-msc51-cpp)
    for (int i = 0; i < 100; ++i) {
        std::array<uint8_t, 32> bytes{};
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes.data() + static_cast<std::size_t>(j) * 8, &v, 8);
        }
        auto fe = FieldElement::from_bytes(bytes);
        CHECK(debug::is_normalized_field_element(fe),
              "from_bytes result must be canonical");
    }

    // Arithmetic results should be normalized
    auto a = FieldElement::from_uint64(0xDEADBEEF);
    auto b = FieldElement::from_uint64(0xCAFEBABE);
    CHECK(debug::is_normalized_field_element(a + b), "add result normalized");
    CHECK(debug::is_normalized_field_element(a - b), "sub result normalized");
    CHECK(debug::is_normalized_field_element(a * b), "mul result normalized");
    CHECK(debug::is_normalized_field_element(a.square()), "sqr result normalized");
    CHECK(debug::is_normalized_field_element(a.inverse()), "inv result normalized");

    (void)printf("    -> all FE normalization checks passed\n");
}

// ============================================================================
// 2. Point on-curve check
// ============================================================================
static void test_on_curve() {
    g_section = "on_curve";
    (void)printf("[2] Point on-curve invariant\n");

    // Generator must be on curve
    CHECK(debug::is_on_curve(Point::generator()), "G must be on curve");

    // Infinity is "on curve" by convention
    CHECK(debug::is_on_curve(Point::infinity()), "O must be on curve");

    // Random scalar multiples of G must be on curve
    std::mt19937_64 rng(123);  // NOLINT(cert-msc32-c,cert-msc51-cpp)
    for (int i = 0; i < 50; ++i) {
        std::array<uint8_t, 32> bytes{};
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes.data() + static_cast<std::size_t>(j) * 8, &v, 8);
        }
        auto s = Scalar::from_bytes(bytes);
        if (s.is_zero()) continue;
        Point const P = Point::generator().scalar_mul(s);
        CHECK(debug::is_on_curve(P), "kG must be on curve");
    }

    // Point addition results must be on curve
    auto s1 = Scalar::from_uint64(12345);
    auto s2 = Scalar::from_uint64(67890);
    Point const P1 = Point::generator().scalar_mul(s1);
    Point const P2 = Point::generator().scalar_mul(s2);
    Point const P3 = P1.add(P2);
    CHECK(debug::is_on_curve(P3), "P1+P2 must be on curve");

    Point const P4 = P1.dbl();
    CHECK(debug::is_on_curve(P4), "2*P must be on curve");

    Point const P5 = P1.negate();
    CHECK(debug::is_on_curve(P5), "-P must be on curve");

    (void)printf("    -> all on-curve checks passed\n");
}

// ============================================================================
// 3. Scalar validity check
// ============================================================================
static void test_scalar_valid() {
    g_section = "scalar_valid";
    (void)printf("[3] Scalar validity invariant\n");

    // Zero is NOT valid (for signing/mul purposes)
    CHECK(!debug::is_valid_scalar(Scalar::zero()), "zero must not be 'valid'");

    // One is valid
    CHECK(debug::is_valid_scalar(Scalar::one()), "one must be valid");

    // Random scalars from proper constructors
    std::mt19937_64 rng(456);  // NOLINT(cert-msc32-c,cert-msc51-cpp)
    for (int i = 0; i < 100; ++i) {
        std::array<uint8_t, 32> bytes{};
        for (int j = 0; j < 4; ++j) {
            uint64_t v = rng();
            std::memcpy(bytes.data() + static_cast<std::size_t>(j) * 8, &v, 8);
        }
        auto s = Scalar::from_bytes(bytes);
        if (s.is_zero()) continue;
        CHECK(debug::is_valid_scalar(s), "random scalar must be valid");
    }

    // Arithmetic results
    auto a = Scalar::from_uint64(42);
    auto b = Scalar::from_uint64(17);
    CHECK(debug::is_valid_scalar(a + b), "a+b must be valid");
    CHECK(debug::is_valid_scalar(a * b), "a*b must be valid");
    CHECK(debug::is_valid_scalar(a.inverse()), "a^-1 must be valid");
    CHECK(debug::is_valid_scalar(a.negate()), "-a must be valid");

    (void)printf("    -> all scalar validity checks passed\n");
}

// ============================================================================
// 4. Macro integration test (SECP_ASSERT_* macros)
// ============================================================================
static void test_macro_integration() {
    g_section = "macros";
    (void)printf("[4] Debug assertion macro integration\n");

    // Reset counter
    auto& c = debug::counters();
    uint64_t const prev = c.invariant_check_count;

    // These should all succeed (not abort)
    auto fe = FieldElement::from_uint64(42);
    SECP_ASSERT_NORMALIZED(fe);
    CHECK(c.invariant_check_count == prev + 1, "counter must increment");

    auto pt = Point::generator();
    SECP_ASSERT_ON_CURVE(pt);
    CHECK(c.invariant_check_count == prev + 2, "counter must increment");

    auto s = Scalar::from_uint64(7);
    SECP_ASSERT_SCALAR_VALID(s);
    CHECK(c.invariant_check_count == prev + 3, "counter must increment");

    SECP_ASSERT_NOT_INFINITY(pt);
    SECP_ASSERT(1 + 1 == 2);
    SECP_ASSERT_MSG(true, "this should not fail");

    (void)printf("    -> all macros work correctly\n");
}

// ============================================================================
// 5. Full computation chain with invariants
// ============================================================================
static void test_full_chain() {
    g_section = "full_chain";
    (void)printf("[5] Full computation chain with invariant checks\n");

    auto k = Scalar::from_uint64(0xABCDEF);
    SECP_ASSERT_SCALAR_VALID(k);

    Point const G = Point::generator();
    SECP_ASSERT_ON_CURVE(G);

    Point const P = G.scalar_mul(k);
    SECP_ASSERT_ON_CURVE(P);

    Point const P2 = P.dbl();
    SECP_ASSERT_ON_CURVE(P2);

    Point const P3 = P.add(P2);
    SECP_ASSERT_ON_CURVE(P3);

    Point const neg = P3.negate();
    SECP_ASSERT_ON_CURVE(neg);

    Point const should_be_inf = P3.add(neg);
    // P + (-P) = O
    CHECK(should_be_inf.is_infinity(), "P + (-P) must be infinity");

    FieldElement const x = P.x();
    FieldElement const y = P.y();

    // y^2 should equal x^3 + 7
    // operator== now normalizes both sides, so non-canonical intermediates
    // from mul_impl / square_impl are handled correctly.
    auto y2 = y.square();
    auto x3 = (x.square() * x) + FieldElement::from_uint64(7);
    CHECK(y2 == x3, "curve equation must hold");

    (void)printf("    -> full chain invariants passed\n");
}

// ============================================================================
// 6. Debug counter reporting
// ============================================================================
static void test_debug_counters() {
    g_section = "counters";
    (void)printf("[6] Debug counter accumulation\n");

    auto& c = debug::counters();
    CHECK(c.invariant_check_count > 0, "invariant counter must have accumulated");
    (void)printf("    -> %llu invariant checks performed so far\n",
           (unsigned long long)c.invariant_check_count);
}

// ============================================================================
// Exportable run function (for unified audit runner)
// ============================================================================
int test_debug_invariants_run() {
    g_pass = g_fail = 0;
    test_fe_normalization();
    test_on_curve();
    test_scalar_valid();
    test_macro_integration();
    test_full_chain();
    test_debug_counters();
    (void)printf("  [debug_invariants] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
// Main (standalone mode)
// ============================================================================
#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    (void)printf("============================================================\n");
    (void)printf("  Debug Invariant Assertions Test\n");
    (void)printf("  Phase V, Task 5.3.3\n");
    (void)printf("============================================================\n\n");

    test_fe_normalization();
    (void)printf("\n");
    test_on_curve();
    (void)printf("\n");
    test_scalar_valid();
    (void)printf("\n");
    test_macro_integration();
    (void)printf("\n");
    test_full_chain();
    (void)printf("\n");
    test_debug_counters();

    (void)printf("\n============================================================\n");
    (void)printf("  Summary: %d passed, %d failed\n", g_pass, g_fail);
    (void)printf("============================================================\n");

    // Print counter report
    SECP_DEBUG_COUNTER_REPORT();

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
