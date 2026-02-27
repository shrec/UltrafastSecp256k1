// ============================================================================
// Tests for FieldElement52 (5x52 Lazy-Reduction Field Arithmetic)
// ============================================================================
// Strategy: Every result is cross-checked against the proven FieldElement (4x64).
// If 5x52 and 4x64 agree on all operations, the implementation is correct.
// ============================================================================

#include "secp256k1/field.hpp"
#include "secp256k1/field_52.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

using namespace secp256k1::fast;

int g_tests_passed = 0;
int g_tests_failed = 0;

#define CHECK(cond, msg) do {                                      \
    if (!(cond)) {                                                 \
        (void)std::printf("  FAIL: %s (line %d)\n", msg, __LINE__);     \
        g_tests_failed++;                                          \
    } else {                                                       \
        g_tests_passed++;                                          \
    }                                                              \
} while(0)

// Compare FieldElement52 against FieldElement by converting to bytes
bool fe52_equals_fe64(const FieldElement52& a52, const FieldElement& a64) {
    FieldElement const converted = a52.to_fe();
    return converted == a64;
}

// Known test vectors for secp256k1 field
struct TestVector {
    const char* name;
    std::array<std::uint64_t, 4> limbs;
};

const TestVector VECTORS[] = {
    {"zero",  {0, 0, 0, 0}},
    {"one",   {1, 0, 0, 0}},
    {"two",   {2, 0, 0, 0}},
    {"small", {0xDEADBEEFULL, 0, 0, 0}},
    {"mid",   {0x123456789ABCDEF0ULL, 0xFEDCBA9876543210ULL, 0, 0}},
    {"large", {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
               0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL}},
    // Close to p
    {"p-1",   {0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
               0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL}},
    // G.x (generator x coordinate)
    {"Gx",    {0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
               0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL}},
    // G.y (generator y coordinate)
    {"Gy",    {0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL,
               0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL}},
    // Random-looking value
    {"rand1", {0xA5A5A5A5A5A5A5A5ULL, 0x5A5A5A5A5A5A5A5AULL,
               0x1234567890ABCDEFULL, 0xFEDCBA0987654321ULL}},
};
constexpr int NUM_VECTORS = sizeof(VECTORS) / sizeof(VECTORS[0]);

// ===========================================================================
// Test Sections
// ===========================================================================

void test_conversion_roundtrip() {
    (void)std::printf("-- Conversion Roundtrip (4x64 -> 5x52 -> 4x64) --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        FieldElement const fe = FieldElement::from_limbs(VECTORS[i].limbs);
        FieldElement52 const fe52 = FieldElement52::from_fe(fe);
        FieldElement const back = fe52.to_fe();
        CHECK(fe == back, VECTORS[i].name);
    }
}

void test_zero_one() {
    (void)std::printf("-- Zero / One --\n");
    FieldElement52 const z = FieldElement52::zero();
    FieldElement52 const o = FieldElement52::one();
    CHECK(fe52_equals_fe64(z, FieldElement::zero()), "zero matches");
    CHECK(fe52_equals_fe64(o, FieldElement::one()),  "one matches");
    CHECK(z.is_zero(), "zero is_zero");
    CHECK(!o.is_zero(), "one is not zero");
}

void test_addition() {
    (void)std::printf("-- Addition --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int j = 0; j < NUM_VECTORS; ++j) {
            FieldElement const a = FieldElement::from_limbs(VECTORS[i].limbs);
            FieldElement const b = FieldElement::from_limbs(VECTORS[j].limbs);
            FieldElement const sum64 = a + b;

            FieldElement52 const a52 = FieldElement52::from_fe(a);
            FieldElement52 const b52 = FieldElement52::from_fe(b);
            FieldElement52 const sum52 = a52 + b52;

            // Must normalize before comparing (addition is lazy!)
            bool const ok = fe52_equals_fe64(sum52, sum64);
            if (!ok) {
                char buf[128];
                (void)std::snprintf(buf, sizeof(buf), "add(%s, %s)",
                             VECTORS[i].name, VECTORS[j].name);
                CHECK(false, buf);
            } else {
                g_tests_passed++;
            }
        }
    }
    (void)std::printf("  %d addition pairs tested\n", NUM_VECTORS * NUM_VECTORS);
}

void test_lazy_chain() {
    (void)std::printf("-- Lazy Addition Chain (accumulate without normalize) --\n");

    FieldElement const a = FieldElement::from_limbs(VECTORS[7].limbs);  // Gx
    FieldElement const b = FieldElement::from_limbs(VECTORS[8].limbs);  // Gy
    FieldElement52 const a52 = FieldElement52::from_fe(a);
    FieldElement52 const b52 = FieldElement52::from_fe(b);

    // Accumulate 100 additions without normalizing
    FieldElement52 chain52 = a52;
    FieldElement chain64 = a;
    for (int i = 0; i < 100; ++i) {
        chain52.add_assign(b52);
        chain64 += b;
    }
    CHECK(fe52_equals_fe64(chain52, chain64), "100 lazy adds match");

    // Accumulate 1000 additions
    chain52 = a52;
    chain64 = a;
    for (int i = 0; i < 1000; ++i) {
        chain52.add_assign(b52);
        chain64 += b;
    }
    CHECK(fe52_equals_fe64(chain52, chain64), "1000 lazy adds match");
}

void test_negate() {
    (void)std::printf("-- Negate --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        FieldElement const fe = FieldElement::from_limbs(VECTORS[i].limbs);
        FieldElement const neg64 = FieldElement::zero() - fe;

        FieldElement52 const fe52 = FieldElement52::from_fe(fe);
        FieldElement52 const neg52 = fe52.negate(1);

        bool const ok = fe52_equals_fe64(neg52, neg64);
        if (!ok) {
            char buf[128];
            (void)std::snprintf(buf, sizeof(buf), "negate(%s)", VECTORS[i].name);
            CHECK(false, buf);
        } else {
            g_tests_passed++;
        }
    }
}

void test_multiplication() {
    (void)std::printf("-- Multiplication --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        for (int j = 0; j < NUM_VECTORS; ++j) {
            FieldElement const a = FieldElement::from_limbs(VECTORS[i].limbs);
            FieldElement const b = FieldElement::from_limbs(VECTORS[j].limbs);
            FieldElement const prod64 = a * b;

            FieldElement52 const a52 = FieldElement52::from_fe(a);
            FieldElement52 const b52 = FieldElement52::from_fe(b);
            FieldElement52 const prod52 = a52 * b52;

            bool const ok = fe52_equals_fe64(prod52, prod64);
            if (!ok) {
                char buf[128];
                (void)std::snprintf(buf, sizeof(buf), "mul(%s, %s)",
                             VECTORS[i].name, VECTORS[j].name);
                CHECK(false, buf);
            } else {
                g_tests_passed++;
            }
        }
    }
    (void)std::printf("  %d multiplication pairs tested\n", NUM_VECTORS * NUM_VECTORS);
}

void test_squaring() {
    (void)std::printf("-- Squaring --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        FieldElement const a = FieldElement::from_limbs(VECTORS[i].limbs);
        FieldElement const sq64 = a.square();

        FieldElement52 const a52 = FieldElement52::from_fe(a);
        FieldElement52 const sq52 = a52.square();

        bool const ok = fe52_equals_fe64(sq52, sq64);
        if (!ok) {
            char buf[128];
            (void)std::snprintf(buf, sizeof(buf), "square(%s)", VECTORS[i].name);
            CHECK(false, buf);
        } else {
            g_tests_passed++;
        }

        // Also verify square == mul(a, a)
        FieldElement52 const prod52 = a52 * a52;
        CHECK(sq52 == prod52, "square == mul(a,a)");
    }
}

void test_mul_chain() {
    (void)std::printf("-- Multiplication Chain (repeated squaring) --\n");
    FieldElement fe = FieldElement::from_limbs(VECTORS[7].limbs);  // Gx
    FieldElement52 fe52 = FieldElement52::from_fe(fe);

    // 100 repeated squarings
    for (int i = 0; i < 100; ++i) {
        fe = fe.square();
        fe52.square_inplace();
    }
    CHECK(fe52_equals_fe64(fe52, fe), "100 repeated squarings match");

    // Continue to 256 total
    for (int i = 0; i < 156; ++i) {
        fe = fe.square();
        fe52.square_inplace();
    }
    CHECK(fe52_equals_fe64(fe52, fe), "256 repeated squarings match");
}

void test_mixed_operations() {
    (void)std::printf("-- Mixed Operations (add + mul + square chains) --\n");
    FieldElement const a = FieldElement::from_limbs(VECTORS[7].limbs);  // Gx
    FieldElement const b = FieldElement::from_limbs(VECTORS[8].limbs);  // Gy
    FieldElement52 const a52 = FieldElement52::from_fe(a);
    FieldElement52 const b52 = FieldElement52::from_fe(b);

    // Compute: ((a + b) * a) + (b * b) - simulate ECC-like operation
    FieldElement const r64 = (a + b) * a + b.square();
    FieldElement52 sum52 = a52 + b52;
    sum52.normalize_weak();  // normalize before mul to keep within range
    FieldElement52 r52 = sum52 * a52;
    FieldElement52 const bsq52 = b52.square();
    r52 = r52 + bsq52;
    CHECK(fe52_equals_fe64(r52, r64), "((a+b)*a + b^2) matches");

    // Longer chain simulating point doubling operations
    FieldElement t64 = a;
    FieldElement52 t52 = a52;
    for (int i = 0; i < 50; ++i) {
        // t = t^2 + Gy * t + Gx
        FieldElement const tsq64 = t64.square();
        FieldElement52 const tsq52 = t52.square();

        FieldElement const bt64 = b * t64;
        FieldElement52 const bt52 = b52 * t52;

        t64 = tsq64 + bt64 + a;
        t52 = tsq52 + bt52 + a52;
    }
    CHECK(fe52_equals_fe64(t52, t64), "50 mixed ops chain matches");
}

void test_half() {
    (void)std::printf("-- Half --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        FieldElement const a = FieldElement::from_limbs(VECTORS[i].limbs);
        FieldElement52 const a52 = FieldElement52::from_fe(a);

        FieldElement52 const h = a52.half();

        // Verify: h + h == a (or h * 2 == a)
        FieldElement52 const dbl = h + h;
        CHECK(dbl == a52, VECTORS[i].name);
    }
}

void test_normalize_edge() {
    (void)std::printf("-- Normalization Edge Cases --\n");

    // Test value = p (should normalize to 0)
    FieldElement52 p_val;
    p_val.n[0] = fe52_constants::P0;
    p_val.n[1] = fe52_constants::P1;
    p_val.n[2] = fe52_constants::P2;
    p_val.n[3] = fe52_constants::P3;
    p_val.n[4] = fe52_constants::P4;
    p_val.normalize();
    CHECK(p_val.is_zero(), "p normalizes to 0");

    // Test value = 2p (should normalize to 0)
    FieldElement52 two_p;
    two_p.n[0] = fe52_constants::P0 * 2;
    two_p.n[1] = fe52_constants::P1 * 2;
    two_p.n[2] = fe52_constants::P2 * 2;
    two_p.n[3] = fe52_constants::P3 * 2;
    two_p.n[4] = fe52_constants::P4 * 2;
    two_p.normalize();
    CHECK(two_p.is_zero(), "2p normalizes to 0");

    // Test p-1 normalizes correctly
    FieldElement52 pm1;
    pm1.n[0] = fe52_constants::P0 - 1;
    pm1.n[1] = fe52_constants::P1;
    pm1.n[2] = fe52_constants::P2;
    pm1.n[3] = fe52_constants::P3;
    pm1.n[4] = fe52_constants::P4;
    pm1.normalize();
    FieldElement const pm1_64 = FieldElement::from_limbs(VECTORS[6].limbs);  // p-1 vector
    CHECK(fe52_equals_fe64(pm1, pm1_64), "p-1 normalizes correctly");
}

void test_commutativity_associativity() {
    (void)std::printf("-- Commutativity & Associativity --\n");
    FieldElement52 const a = FieldElement52::from_fe(FieldElement::from_limbs(VECTORS[7].limbs));
    FieldElement52 const b = FieldElement52::from_fe(FieldElement::from_limbs(VECTORS[8].limbs));
    FieldElement52 const c = FieldElement52::from_fe(FieldElement::from_limbs(VECTORS[9].limbs));

    // Commutative addition
    FieldElement52 const ab = a + b;
    FieldElement52 const ba = b + a;
    CHECK(ab == ba, "a+b == b+a");

    // Commutative multiplication
    FieldElement52 const a_times_b = a * b;
    FieldElement52 const b_times_a = b * a;
    CHECK(a_times_b == b_times_a, "a*b == b*a");

    // Associative multiplication
    FieldElement52 const ab_c = (a * b) * c;
    FieldElement52 const a_bc = a * (b * c);
    CHECK(ab_c == a_bc, "(a*b)*c == a*(b*c)");

    // Distributive: a * (b + c) == a*b + a*c
    FieldElement52 bc_sum = b + c;
    bc_sum.normalize_weak();
    FieldElement52 const dist_left = a * bc_sum;
    FieldElement52 const dist_right = (a * b) + (a * c);
    CHECK(dist_left == dist_right, "a*(b+c) == a*b + a*c");
}

} // anonymous namespace

#ifdef STANDALONE_TEST
int main() {
#else
int test_field_52_main() {
#endif
    (void)std::printf("\n=== FieldElement52 (5x52 Lazy-Reduction) Tests ===\n\n");

    test_conversion_roundtrip();
    test_zero_one();
    test_addition();
    test_lazy_chain();
    test_negate();
    test_multiplication();
    test_squaring();
    test_mul_chain();
    test_mixed_operations();
    test_half();
    test_normalize_edge();
    test_commutativity_associativity();

    (void)std::printf("\n=== Results: %d passed, %d failed ===\n",
               g_tests_passed, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
