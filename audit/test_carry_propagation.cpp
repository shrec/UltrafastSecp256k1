// ============================================================================
// Carry Propagation Stress Test
// Tests field arithmetic at limb boundaries near 2^256 - 1
// ============================================================================
// Exercises the most dangerous corner cases for carry chain bugs:
//   1. All-ones limbs (max carry pressure)
//   2. Single-limb overflow patterns (0xFFFFFFFFFFFFFFFF in one limb)
//   3. Cascading carry across all limbs
//   4. Values near p that trigger final reduction
//   5. Products that produce maximum intermediate values
//   6. Cross-limb boundary patterns (bit 63->64, 127->128, 191->192)
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

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

// secp256k1 prime p
static const std::array<uint8_t, 32> P_BYTES = {
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFE, 0xFF,0xFF,0xFC,0x2F
};

[[maybe_unused]]
static FieldElement fe_from_hex(const char* hex64) {
    std::array<uint8_t, 32> bytes{};
    for (int i = 0; i < 32; ++i) {
        unsigned hi, lo;
        sscanf(hex64 + i * 2, "%1x", &hi);
        sscanf(hex64 + i * 2 + 1, "%1x", &lo);
        bytes[i] = static_cast<uint8_t>((hi << 4) | lo);
    }
    return FieldElement::from_bytes(bytes);
}

static FieldElement fe_from_limbs4(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) {
    return FieldElement::from_limbs({l0, l1, l2, l3});
}

// ============================================================================
// 1. All-ones pattern: 0xFFFFFFFF...FF (= 2^256 - 1)
//    This is the maximum possible 256-bit value.
//    After reduction mod p: 2^256 - 1 - p = 2^32 + 976 = 0x1000003D0
// ============================================================================
static void test_all_ones() {
    g_section = "all_ones";
    printf("[1] All-ones limb pattern (2^256 - 1)\n");

    auto max_val = fe_from_limbs4(
        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF
    );
    auto zero = FieldElement::zero();
    auto one  = FieldElement::one();
    CHECK(zero.to_bytes() != one.to_bytes(), "zero != one");

    // (2^256 - 1) mod p = 0x1000003D0
    auto expected = FieldElement::from_uint64(0x1000003D0ULL);
    CHECK(max_val.to_bytes() == expected.to_bytes(), "2^256-1 reduces to 0x1000003D0");

    // max * max should be consistent
    auto sq = max_val.square();
    auto mul_self = max_val * max_val;
    CHECK(sq.to_bytes() == mul_self.to_bytes(), "max^2 == max*max");

    // max + 1 should = (0x1000003D0 + 1)
    auto max_plus_1 = max_val + one;
    auto expected2 = FieldElement::from_uint64(0x1000003D1ULL);
    CHECK(max_plus_1.to_bytes() == expected2.to_bytes(), "max+1 = 0x1000003D1");

    // max * 2 = 2 * 0x1000003D0 = 0x20000007A0
    auto two = FieldElement::from_uint64(2);
    auto dbl = max_val + max_val;
    auto dbl2 = max_val * two;
    CHECK(dbl.to_bytes() == dbl2.to_bytes(), "max+max == max*2");
}

// ============================================================================
// 2. Single-limb overflow patterns
// ============================================================================
static void test_single_limb_max() {
    g_section = "single_limb_max";
    printf("[2] Single-limb maximum patterns\n");

    auto one  = FieldElement::one();
    auto zero = FieldElement::zero();

    // Pattern: 0xFFFFFFFFFFFFFFFF in each limb position, 0 elsewhere
    for (int pos = 0; pos < 4; ++pos) {
        uint64_t limbs[4] = {0, 0, 0, 0};
        limbs[pos] = 0xFFFFFFFFFFFFFFFF;
        auto fe = FieldElement::from_limbs({limbs[0], limbs[1], limbs[2], limbs[3]});

        // Round-trip: serialization must be consistent
        auto bytes = fe.to_bytes();
        auto fe2 = FieldElement::from_bytes(bytes);
        CHECK(fe2.to_bytes() == bytes, "single-limb-max round-trip");

        // Squaring consistency
        auto sq1 = fe.square();
        auto sq2 = fe * fe;
        CHECK(sq1.to_bytes() == sq2.to_bytes(), "single-limb-max sqr==mul");

        // Inverse round-trip: a * a^(-1) == 1
        if (fe.to_bytes() != zero.to_bytes()) {
            auto inv = fe.inverse();
            auto product = fe * inv;
            CHECK(product.to_bytes() == one.to_bytes(), "single-limb-max inverse");
        }
    }
}

// ============================================================================
// 3. Cross-limb boundary carry patterns
//    Set bit 63, 127, 191, 255 (the high bit of each 64-bit limb)
// ============================================================================
static void test_cross_limb_carry() {
    g_section = "cross_limb_carry";
    printf("[3] Cross-limb boundary carry patterns\n");

    auto one = FieldElement::one();

    // Create values with carry-prone bit patterns
    struct Pattern {
        uint64_t l0, l1, l2, l3;
    };

    Pattern patterns[] = {
        // Bit 63 set: carry from limb0 -> limb1
        {0x8000000000000000ULL, 0, 0, 0},
        // Bit 127 set: carry from limb1 -> limb2
        {0, 0x8000000000000000ULL, 0, 0},
        // Bit 191 set: carry from limb2 -> limb3
        {0, 0, 0x8000000000000000ULL, 0},
        // Bit 255 set: carry from limb3 -> reduction
        {0, 0, 0, 0x8000000000000000ULL},
        // All high-bits set
        {0x8000000000000000ULL, 0x8000000000000000ULL,
         0x8000000000000000ULL, 0x8000000000000000ULL},
        // Alternating: 0xAAAA... pattern
        {0xAAAAAAAAAAAAAAAAULL, 0xAAAAAAAAAAAAAAAAULL,
         0xAAAAAAAAAAAAAAAAULL, 0xAAAAAAAAAAAAAAAAULL},
        // Alternating: 0x5555... pattern
        {0x5555555555555555ULL, 0x5555555555555555ULL,
         0x5555555555555555ULL, 0x5555555555555555ULL},
    };

    for (int i = 0; i < (int)(sizeof(patterns) / sizeof(patterns[0])); ++i) {
        auto& p = patterns[i];
        auto a = fe_from_limbs4(p.l0, p.l1, p.l2, p.l3);
        auto zero = FieldElement::zero();

        // Basic consistency: serialize round-trip
        auto bytes = a.to_bytes();
        auto a2 = FieldElement::from_bytes(bytes);
        CHECK(a2.to_bytes() == bytes, "carry pattern round-trip");

        // a * a must equal a.square()
        CHECK((a * a).to_bytes() == a.square().to_bytes(), "carry pattern sqr==mul*");

        // (a + a) must equal a * 2
        auto two = FieldElement::from_uint64(2);
        CHECK((a + a).to_bytes() == (a * two).to_bytes(), "carry pattern add==mul2");

        // a - a must equal 0
        CHECK((a - a).to_bytes() == zero.to_bytes(), "carry pattern sub-self");

        // Inverse (if non-zero)
        if (bytes != zero.to_bytes()) {
            auto inv = a.inverse();
            CHECK((a * inv).to_bytes() == one.to_bytes(), "carry pattern inverse");
        }
    }
}

// ============================================================================
// 4. Values near p (reduction boundary)
// ============================================================================
static void test_near_prime() {
    g_section = "near_prime";
    printf("[4] Values near the prime p (reduction boundary)\n");

    auto p_m1 = FieldElement::from_bytes(P_BYTES) - FieldElement::one();
    auto zero = FieldElement::zero();
    auto one  = FieldElement::one();

    // p itself should reduce to 0
    auto p_val = FieldElement::from_bytes(P_BYTES);
    CHECK(p_val.to_bytes() == zero.to_bytes(), "p reduces to 0");

    // p + 1 should reduce to 1
    // (but from_bytes reduces on load, so p -> 0, then 0 + 1 = 1)
    auto p_plus_1 = p_val + one;
    CHECK(p_plus_1.to_bytes() == one.to_bytes(), "p + 1 reduces to 1");

    // (p-1) + 1 = 0
    CHECK((p_m1 + one).to_bytes() == zero.to_bytes(), "(p-1)+1 == 0");

    // (p-1)^2 == 1 (since p-1 == -1 mod p)
    CHECK(p_m1.square().to_bytes() == one.to_bytes(), "(p-1)^2 == 1");

    // (p-1) * (p-1) == 1
    CHECK((p_m1 * p_m1).to_bytes() == one.to_bytes(), "(p-1)*(p-1) == 1");

    // Test values: p-2, p-3, ..., p-16
    for (uint64_t delta = 2; delta <= 16; ++delta) {
        auto d = FieldElement::from_uint64(delta);
        auto val = p_m1 - d + one; // = p - delta

        // val + delta should == 0 (since val = p - delta == -delta)
        auto sum = val + d;
        CHECK(sum.to_bytes() == zero.to_bytes(), "p-delta + delta == 0");

        // val * val should be consistent with square
        CHECK((val * val).to_bytes() == val.square().to_bytes(), "near-p sqr consistency");
    }
}

// ============================================================================
// 5. Maximum intermediate values during multiplication
//    These trigger the widest possible products in the schoolbook/Karatsuba mul
// ============================================================================
static void test_max_intermediate() {
    g_section = "max_intermediate";
    printf("[5] Maximum intermediate values (carry chain stress)\n");

    auto one  = FieldElement::one();

    // Construct two values designed to maximize partial products:
    // a = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2E (= p-1)
    // b = 0xFFFFFFFF FFFFFFFF FFFFFFFF FFFFFFFE FFFFFC2D (= p-2)
    // a * b = (-1) * (-2) = 2  (mod p)
    auto p_m1 = FieldElement::from_bytes(P_BYTES) - one;
    auto p_m2 = p_m1 - one;
    auto two = FieldElement::from_uint64(2);

    auto prod = p_m1 * p_m2;
    CHECK(prod.to_bytes() == two.to_bytes(), "(p-1)*(p-2) == 2");

    // Chained: ((p-1)^2)^2 = 1^2 = 1
    auto r1 = p_m1.square().square();
    CHECK(r1.to_bytes() == one.to_bytes(), "((p-1)^2)^2 == 1");

    // Alternating max: (p-1) * (p-1) * (p-1) = (p-1) (since 1 * (p-1) = (p-1))
    auto r2 = (p_m1 * p_m1) * p_m1;
    CHECK(r2.to_bytes() == p_m1.to_bytes(), "(p-1)^3 == p-1");

    // Large fan-out: multiply several near-p values
    auto a = p_m1;
    for (int i = 0; i < 100; ++i) {
        auto b = p_m1 - FieldElement::from_uint64(static_cast<uint64_t>(i));
        auto product = a * b;
        auto inv_b = b.inverse();
        auto recovered = product * inv_b;
        CHECK(recovered.to_bytes() == a.to_bytes(), "large multiply recovery");
    }
}

// ============================================================================
// 6. Scalar carry propagation (group order n)
// ============================================================================
static void test_scalar_carry() {
    g_section = "scalar_carry";
    printf("[6] Scalar carry propagation near group order n\n");

    // n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    auto n_m1_bytes = std::array<uint8_t, 32>{
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x40
    };
    auto n_m1 = Scalar::from_bytes(n_m1_bytes);
    auto one  = Scalar::from_bytes({0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1});

    // (n-1) + 1 = 0 mod n
    auto sum = n_m1 + one;
    CHECK(sum.is_zero(), "(n-1)+1 == 0");

    // (n-1)^2 == 1 mod n
    auto sq = n_m1 * n_m1;
    CHECK(sq.to_bytes() == one.to_bytes(), "(n-1)^2 == 1");

    // Scalar inverse of 2
    auto two = Scalar::from_bytes({0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2});
    auto two_inv = two.inverse();
    CHECK((two * two_inv).to_bytes() == one.to_bytes(), "2 * 2^(-1) == 1");

    // Chain: multiply many near-n values and verify consistency
    auto s = n_m1;
    for (int i = 0; i < 50; ++i) {
        auto t = n_m1 - Scalar::from_bytes({0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                                             static_cast<uint8_t>(i + 2)});
        auto product = s * t;
        auto inv_t = t.inverse();
        auto recovered = product * inv_t;
        CHECK(recovered.to_bytes() == s.to_bytes(), "scalar carry chain recovery");
    }
}

// ============================================================================
// 7. Point arithmetic at carry boundaries
// ============================================================================
static void test_point_carry() {
    g_section = "point_carry";
    printf("[7] Point arithmetic carry propagation\n");

    auto G = Point::generator();

    // Scalar mul with near-n value: (n-1)*G should equal -G
    auto n_m1 = Scalar::from_bytes({
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x40
    });

    auto P = G.scalar_mul(n_m1);

    // (n-1)*G should have same x as G
    CHECK(P.x().to_bytes() == G.x().to_bytes(), "(n-1)G has same x as G");

    // (n-1)*G + G should be infinity
    auto sum = P.add(G);
    CHECK(sum.is_infinity(), "(n-1)G + G = O");

    // Double-and-add vs scalar_mul consistency at carry boundary
    auto k = Scalar::from_bytes({
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x01
    });
    auto R1 = G.scalar_mul(k);
    // Same computation via splitting: not possible to do naively, but
    // verify that the result lies on the curve
    auto comp = R1.to_compressed();
    auto uncomp = R1.to_uncompressed();
    CHECK(comp[0] == 0x02 || comp[0] == 0x03, "carry-boundary point is valid (compressed)");
    CHECK(uncomp[0] == 0x04, "carry-boundary point is valid (uncompressed)");

    // Verify y^2 == x^3 + 7 (curve equation)
    auto x = R1.x();
    auto y = R1.y();
    auto y_sq = y.square();
    auto x_cubed = x * x * x;
    auto seven = FieldElement::from_uint64(7);
    auto rhs = x_cubed + seven;
    CHECK(y_sq.to_bytes() == rhs.to_bytes(), "carry result on curve: y^2==x^3+7");
}

// ============================================================================
// Exportable run function (for unified audit runner)
// ============================================================================
int test_carry_propagation_run() {
    g_pass = g_fail = 0;
    test_all_ones();
    test_single_limb_max();
    test_cross_limb_carry();
    test_near_prime();
    test_max_intermediate();
    test_scalar_carry();
    test_point_carry();
    printf("  [carry_propagation] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
// Main (standalone mode)
// ============================================================================
#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("============================================================\n");
    printf("  Carry Propagation Stress Test\n");
    printf("  Arithmetic boundary & limb carry-chain verification\n");
    printf("============================================================\n\n");

    test_all_ones();          printf("\n");
    test_single_limb_max();   printf("\n");
    test_cross_limb_carry();  printf("\n");
    test_near_prime();        printf("\n");
    test_max_intermediate();  printf("\n");
    test_scalar_carry();      printf("\n");
    test_point_carry();

    printf("\n============================================================\n");
    printf("  Summary: %d passed, %d failed\n", g_pass, g_fail);
    printf("============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
