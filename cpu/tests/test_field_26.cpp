// ============================================================================
// Tests for FieldElement26 (10x26 Lazy-Reduction Field Arithmetic)
// ============================================================================
// Strategy: Every result is cross-checked against the proven FieldElement (4x64).
// If 10x26 and 4x64 agree on all operations, the implementation is correct.
// ============================================================================

#include "secp256k1/field.hpp"
#include "secp256k1/field_26.hpp"

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

// Compare FieldElement26 against FieldElement by converting to bytes
bool fe26_equals_fe64(const FieldElement26& a26, const FieldElement& a64) {
    FieldElement const converted = a26.to_fe();
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
    (void)std::printf("-- Conversion Roundtrip (4x64 -> 10x26 -> 4x64) --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        FieldElement const fe = FieldElement::from_limbs(VECTORS[i].limbs);
        FieldElement26 const fe26 = FieldElement26::from_fe(fe);
        FieldElement const back = fe26.to_fe();
        CHECK(fe == back, VECTORS[i].name);
    }
}

void test_zero_one() {
    (void)std::printf("-- Zero / One --\n");
    FieldElement26 const z = FieldElement26::zero();
    FieldElement26 const o = FieldElement26::one();
    CHECK(fe26_equals_fe64(z, FieldElement::zero()), "zero matches");
    CHECK(fe26_equals_fe64(o, FieldElement::one()),  "one matches");
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

            FieldElement26 const a26 = FieldElement26::from_fe(a);
            FieldElement26 const b26 = FieldElement26::from_fe(b);
            FieldElement26 const sum26 = a26 + b26;

            bool const ok = fe26_equals_fe64(sum26, sum64);
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
    FieldElement26 const a26 = FieldElement26::from_fe(a);
    FieldElement26 const b26 = FieldElement26::from_fe(b);

    // Accumulate 50 additions without normalizing (within 6-bit headroom)
    FieldElement26 chain26 = a26;
    FieldElement chain64 = a;
    for (int i = 0; i < 50; ++i) {
        chain26.add_assign(b26);
        chain64 += b;
    }
    CHECK(fe26_equals_fe64(chain26, chain64), "50 lazy adds match");

    // With normalize_weak every 50 adds, do 500 total
    chain26 = a26;
    chain64 = a;
    for (int i = 0; i < 500; ++i) {
        chain26.add_assign(b26);
        chain64 += b;
        if ((i + 1) % 50 == 0) chain26.normalize_weak();
    }
    CHECK(fe26_equals_fe64(chain26, chain64), "500 adds (norm every 50) match");
}

void test_negate() {
    (void)std::printf("-- Negate --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        FieldElement const fe = FieldElement::from_limbs(VECTORS[i].limbs);
        FieldElement const neg64 = FieldElement::zero() - fe;

        FieldElement26 const fe26 = FieldElement26::from_fe(fe);
        FieldElement26 const neg26 = fe26.negate(1);

        bool const ok = fe26_equals_fe64(neg26, neg64);
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

            FieldElement26 const a26 = FieldElement26::from_fe(a);
            FieldElement26 const b26 = FieldElement26::from_fe(b);
            FieldElement26 const prod26 = a26 * b26;

            bool const ok = fe26_equals_fe64(prod26, prod64);
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

        FieldElement26 const a26 = FieldElement26::from_fe(a);
        FieldElement26 const sq26 = a26.square();

        bool const ok = fe26_equals_fe64(sq26, sq64);
        if (!ok) {
            char buf[128];
            (void)std::snprintf(buf, sizeof(buf), "square(%s)", VECTORS[i].name);
            CHECK(false, buf);
        } else {
            g_tests_passed++;
        }

        // Also verify square == mul(a, a)
        FieldElement26 const prod26 = a26 * a26;
        CHECK(sq26 == prod26, "square == mul(a,a)");
    }
}

void test_mul_chain() {
    (void)std::printf("-- Multiplication Chain (repeated squaring) --\n");
    FieldElement fe = FieldElement::from_limbs(VECTORS[7].limbs);  // Gx
    FieldElement26 fe26 = FieldElement26::from_fe(fe);

    // 100 repeated squarings
    for (int i = 0; i < 100; ++i) {
        fe = fe.square();
        fe26.square_inplace();
    }
    CHECK(fe26_equals_fe64(fe26, fe), "100 repeated squarings match");

    // Continue to 256 total
    for (int i = 0; i < 156; ++i) {
        fe = fe.square();
        fe26.square_inplace();
    }
    CHECK(fe26_equals_fe64(fe26, fe), "256 repeated squarings match");
}

void test_mixed_operations() {
    (void)std::printf("-- Mixed Operations (add + mul + square chains) --\n");
    FieldElement const a = FieldElement::from_limbs(VECTORS[7].limbs);  // Gx
    FieldElement const b = FieldElement::from_limbs(VECTORS[8].limbs);  // Gy
    FieldElement26 const a26 = FieldElement26::from_fe(a);
    FieldElement26 const b26 = FieldElement26::from_fe(b);

    // Compute: ((a + b) * a) + (b * b)
    FieldElement const r64 = (a + b) * a + b.square();
    FieldElement26 sum26 = a26 + b26;
    sum26.normalize_weak();
    FieldElement26 r26 = sum26 * a26;
    FieldElement26 const bsq26 = b26.square();
    r26 = r26 + bsq26;
    CHECK(fe26_equals_fe64(r26, r64), "((a+b)*a + b^2) matches");

    // Longer chain simulating point doubling operations
    FieldElement t64 = a;
    FieldElement26 t26 = a26;
    for (int i = 0; i < 50; ++i) {
        FieldElement const tsq64 = t64.square();
        FieldElement26 const tsq26 = t26.square();

        FieldElement const bt64 = b * t64;
        FieldElement26 const bt26 = b26 * t26;

        t64 = tsq64 + bt64 + a;
        t26 = tsq26 + bt26 + a26;
    }
    CHECK(fe26_equals_fe64(t26, t64), "50 mixed ops chain matches");
}

void test_half() {
    (void)std::printf("-- Half --\n");
    for (int i = 0; i < NUM_VECTORS; ++i) {
        FieldElement const a = FieldElement::from_limbs(VECTORS[i].limbs);
        FieldElement26 const a26 = FieldElement26::from_fe(a);

        FieldElement26 const h = a26.half();

        // Verify: h + h == a
        FieldElement26 const dbl = h + h;
        CHECK(dbl == a26, VECTORS[i].name);
    }
}

void test_normalize_edge() {
    (void)std::printf("-- Normalization Edge Cases --\n");
    using namespace fe26_constants;

    // Test value = p (should normalize to 0)
    FieldElement26 p_val;
    p_val.n[0] = P0; p_val.n[1] = P1; p_val.n[2] = P2; p_val.n[3] = P3;
    p_val.n[4] = P4; p_val.n[5] = P5; p_val.n[6] = P6; p_val.n[7] = P7;
    p_val.n[8] = P8; p_val.n[9] = P9;
    p_val.normalize();
    CHECK(p_val.is_zero(), "p normalizes to 0");

    // Test value = 2p (should normalize to 0)
    FieldElement26 two_p;
    two_p.n[0] = P0*2; two_p.n[1] = P1*2; two_p.n[2] = P2*2; two_p.n[3] = P3*2;
    two_p.n[4] = P4*2; two_p.n[5] = P5*2; two_p.n[6] = P6*2; two_p.n[7] = P7*2;
    two_p.n[8] = P8*2; two_p.n[9] = P9*2;
    two_p.normalize();
    CHECK(two_p.is_zero(), "2p normalizes to 0");

    // Test p-1 normalizes correctly
    FieldElement26 pm1;
    pm1.n[0] = P0 - 1; pm1.n[1] = P1; pm1.n[2] = P2; pm1.n[3] = P3;
    pm1.n[4] = P4; pm1.n[5] = P5; pm1.n[6] = P6; pm1.n[7] = P7;
    pm1.n[8] = P8; pm1.n[9] = P9;
    pm1.normalize();
    FieldElement const pm1_64 = FieldElement::from_limbs(VECTORS[6].limbs);  // p-1
    CHECK(fe26_equals_fe64(pm1, pm1_64), "p-1 normalizes correctly");
}

void test_commutativity_associativity() {
    (void)std::printf("-- Commutativity & Associativity --\n");
    FieldElement26 const a = FieldElement26::from_fe(FieldElement::from_limbs(VECTORS[7].limbs));
    FieldElement26 const b = FieldElement26::from_fe(FieldElement::from_limbs(VECTORS[8].limbs));
    FieldElement26 const c = FieldElement26::from_fe(FieldElement::from_limbs(VECTORS[9].limbs));

    // Commutative addition
    FieldElement26 const ab = a + b;
    FieldElement26 const ba = b + a;
    CHECK(ab == ba, "a+b == b+a");

    // Commutative multiplication
    FieldElement26 const a_times_b = a * b;
    FieldElement26 const b_times_a = b * a;
    CHECK(a_times_b == b_times_a, "a*b == b*a");

    // Associative multiplication
    FieldElement26 const ab_c = (a * b) * c;
    FieldElement26 const a_bc = a * (b * c);
    CHECK(ab_c == a_bc, "(a*b)*c == a*(b*c)");

    // Distributive: a * (b + c) == a*b + a*c
    FieldElement26 bc_sum = b + c;
    bc_sum.normalize_weak();
    FieldElement26 const dist_left = a * bc_sum;
    FieldElement26 const dist_right = (a * b) + (a * c);
    CHECK(dist_left == dist_right, "a*(b+c) == a*b + a*c");
}

void test_mul_after_lazy_add() {
    (void)std::printf("-- Mul After Lazy Additions --\n");
    // Test that mul works correctly on inputs with high magnitude
    // (limb values > 26 bits due to lazy additions)
    FieldElement const a = FieldElement::from_limbs(VECTORS[7].limbs);
    FieldElement const b = FieldElement::from_limbs(VECTORS[8].limbs);

    FieldElement26 const a26 = FieldElement26::from_fe(a);
    FieldElement26 const b26 = FieldElement26::from_fe(b);

    // Accumulate 10 lazy adds, then multiply
    FieldElement26 sum26 = a26;
    FieldElement sum64 = a;
    for (int i = 0; i < 10; ++i) {
        sum26.add_assign(b26);
        sum64 += b;
    }

    // Multiply the accumulated value by b
    FieldElement const prod64 = sum64 * b;
    FieldElement26 const prod26 = sum26 * b26;
    CHECK(fe26_equals_fe64(prod26, prod64), "mul after 10 lazy adds");

    // Multiply two accumulated values
    FieldElement26 sum2_26 = b26;
    FieldElement sum2_64 = b;
    for (int i = 0; i < 5; ++i) {
        sum2_26.add_assign(a26);
        sum2_64 += a;
    }

    FieldElement const prod2_64 = sum64 * sum2_64;
    FieldElement26 const prod2_26 = sum26 * sum2_26;
    CHECK(fe26_equals_fe64(prod2_26, prod2_64), "mul of two accumulated values");
}

} // anonymous namespace

#ifdef STANDALONE_TEST
int main() {
#else
int test_field_26_main() {
#endif
    (void)std::printf("\n=== FieldElement26 (10x26 Lazy-Reduction) Tests ===\n\n");

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
    test_mul_after_lazy_add();

    (void)std::printf("\n=== Results: %d passed, %d failed ===\n",
               g_tests_passed, g_tests_failed);

    return g_tests_failed > 0 ? 1 : 0;
}
