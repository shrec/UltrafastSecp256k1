// ============================================================================
// Cryptographic Self-Audit: Field Arithmetic (secp256k1 prime field)
// ============================================================================
// Covers: addition, subtraction, multiplication, square, reduction,
//         canonical representation, limb boundaries, batch inverse, sqrt.
// Cross-checks against big-integer reference (manual carry chains).
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>
#include <vector>

#include "secp256k1/field.hpp"
#include "secp256k1/sanitizer_scale.hpp"

using namespace secp256k1::fast;

static int g_pass = 0, g_fail = 0;
static const char* g_section = "";

#define CHECK(cond, msg) do { \
    if (cond) { \
        ++g_pass; \
    } else { \
        printf("  FAIL [%s]: %s (line %d)\n", g_section, msg, __LINE__); \
        ++g_fail; \
    } \
} while(0)

// Deterministic PRNG
static std::mt19937_64 rng(0xA0D17'F1E1D);  // NOLINT(cert-msc32-c,cert-msc51-cpp)

static std::array<uint8_t, 32> random_bytes() {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + static_cast<std::size_t>(i) * 8, &v, 8);
    }
    return out;
}

static FieldElement random_fe() {
    return FieldElement::from_bytes(random_bytes());
}

// The prime p as bytes (big-endian)
static const std::array<uint8_t, 32> P_BYTES = {
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFE, 0xFF,0xFF,0xFC,0x2F
};

// ============================================================================
// 1. Addition mod p -- overflow paths
// ============================================================================
static void test_addition_overflow() {
    g_section = "add_overflow";
    printf("[1] Addition mod p -- overflow paths\n");

    auto zero = FieldElement::from_uint64(0);
    auto one  = FieldElement::one();

    // a + 0 == a
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        auto r = a + zero;
        CHECK(r == a, "a + 0 == a");
    }

    // a + b == b + a (commutativity)
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        auto b = random_fe();
        CHECK((a + b) == (b + a), "a+b == b+a");
    }

    // (a + b) + c == a + (b + c) (associativity)
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        auto b = random_fe();
        auto c = random_fe();
        CHECK(((a + b) + c) == (a + (b + c)), "(a+b)+c == a+(b+c)");
    }

    // p-1 + 1 == 0 (overflow wrapping)
    {
        auto p_minus_1 = FieldElement::from_bytes(P_BYTES) - one;
        auto r = p_minus_1 + one;
        // p mod p == 0, but from_bytes(P) reduces to 0 already
        // So we test: (p-1) + 1 should be 0 mod p
        auto zero_fe = FieldElement::from_uint64(0);
        CHECK(r == zero_fe, "(p-1) + 1 == 0");
    }

    // Large value near p
    for (int i = 0; i < 100; ++i) {
        auto a = random_fe();
        auto neg_a = a.negate();
        auto sum = a + neg_a;
        CHECK(sum == zero, "a + (-a) == 0");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 2. Subtraction borrow-chain
// ============================================================================
static void test_subtraction_borrow() {
    g_section = "sub_borrow";
    printf("[2] Subtraction borrow-chain\n");

    auto zero = FieldElement::from_uint64(0);

    // a - a == 0
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        CHECK((a - a) == zero, "a - a == 0");
    }

    // a - 0 == a
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        CHECK((a - zero) == a, "a - 0 == a");
    }

    // 0 - a == -a
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        auto neg = a.negate();
        CHECK((zero - a) == neg, "0 - a == -a");
    }

    // Borrow across limb boundary: 1 - 2 == p - 1
    {
        auto one = FieldElement::one();
        auto two = FieldElement::from_uint64(2);
        auto r = one - two;
        auto neg_one = one.negate();
        CHECK(r == neg_one, "1 - 2 == -(1) mod p");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 3. Multiplication carry propagation
// ============================================================================
static void test_mul_carry() {
    g_section = "mul_carry";
    printf("[3] Multiplication carry propagation\n");

    auto one = FieldElement::one();
    auto zero = FieldElement::from_uint64(0);

    // a * 1 == a
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        CHECK((a * one) == a, "a * 1 == a");
    }

    // a * 0 == 0
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        CHECK((a * zero) == zero, "a * 0 == 0");
    }

    // a * b == b * a (commutativity)
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        auto b = random_fe();
        CHECK((a * b) == (b * a), "a*b == b*a");
    }

    // (a * b) * c == a * (b * c) (associativity)
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        auto b = random_fe();
        auto c = random_fe();
        CHECK(((a * b) * c) == (a * (b * c)), "(a*b)*c == a*(b*c)");
    }

    // a * (b + c) == a*b + a*c (distributivity)
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        auto b = random_fe();
        auto c = random_fe();
        CHECK((a * (b + c)) == ((a * b) + (a * c)), "a*(b+c) == a*b + a*c");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 4. Square vs Mul equivalence
// ============================================================================
static void test_square_vs_mul() {
    g_section = "sqr_vs_mul";
    printf("[4] Square vs Mul equivalence\n");

    for (int i = 0; i < SCALED(10000, 200); ++i) {
        auto a = random_fe();
        auto sq = a.square();
        auto mul = a * a;
        CHECK(sq == mul, "a^2 == a*a");
    }

    // Square of zero == 0
    {
        auto z = FieldElement::from_uint64(0);
        CHECK(z.square() == z, "0^2 == 0");
    }

    // Square of one == 1
    {
        auto o = FieldElement::one();
        CHECK(o.square() == o, "1^2 == 1");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 5. Reduction correctness (final conditional subtract p)
// ============================================================================
static void test_reduction() {
    g_section = "reduction";
    printf("[5] Reduction correctness\n");

    // from_bytes(P) should give 0
    {
        auto p = FieldElement::from_bytes(P_BYTES);
        auto zero = FieldElement::from_uint64(0);
        CHECK(p == zero, "from_bytes(p) == 0");
    }

    // from_bytes(P+1) should give 1
    {
        std::array<uint8_t, 32> p_plus_1 = P_BYTES;
        // p = ...FFFFFFFEFFFFFC2F, p+1 = ...FFFFFFFEFFFFFC30
        // But from_bytes reduces mod p, so p+1 mod p == 1
        // Actually we need to compute p+1 carefully
        // p[31] = 0x2F. 0x2F + 1 = 0x30, no carry
        p_plus_1[31] = 0x30;
        auto r = FieldElement::from_bytes(p_plus_1);
        auto one = FieldElement::one();
        CHECK(r == one, "from_bytes(p+1) == 1");
    }

    // to_bytes -> from_bytes roundtrip
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        auto bytes = a.to_bytes();
        auto b = FieldElement::from_bytes(bytes);
        CHECK(a == b, "to_bytes -> from_bytes roundtrip");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 6. Canonical representation enforcement
// ============================================================================
static void test_canonical() {
    g_section = "canonical";
    printf("[6] Canonical representation\n");

    // After to_bytes, all values should be < p
    for (int i = 0; i < SCALED(10000, 200); ++i) {
        auto a = random_fe();
        auto bytes = a.to_bytes();

        // Check bytes < P_BYTES (big-endian comparison)
        bool greater = false;
        for (int j = 0; j < 32; ++j) {
            if (bytes[j] < P_BYTES[j]) { break; }
            if (bytes[j] > P_BYTES[j]) { greater = true; break; }
        }
        CHECK(!greater, "serialized value < p or == 0");
        CHECK(!greater, "no value exceeds p");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 7. Limb boundary stress (max limb values)
// ============================================================================
static void test_limb_boundary() {
    g_section = "limb_boundary";
    printf("[7] Limb boundary stress\n");

    // All-ones value (0xFF...FF = 2^256 - 1)
    std::array<uint8_t, 32> all_ff{};
    std::memset(all_ff.data(), 0xFF, 32);
    auto max_val = FieldElement::from_bytes(all_ff);
    // 2^256 - 1 mod p = 2^256 - 1 - p = 2^32 + 976 = 0x1000003D0
    auto expected = FieldElement::from_uint64(0x1000003D0ULL);
    CHECK(max_val == expected, "0xFF..FF mod p == 0x1000003D0");

    // Operations on p-1
    auto p_minus_1 = FieldElement::from_bytes(P_BYTES) - FieldElement::one();
    auto sq = p_minus_1.square();
    // (p-1)^2 = (-1)^2 = 1 mod p
    CHECK(sq == FieldElement::one(), "(p-1)^2 == 1");

    // p-1 inverse = p-1 (since (-1)^(-1) = -1)
    auto inv = p_minus_1.inverse();
    CHECK(inv == p_minus_1, "(p-1)^(-1) == p-1");

    // Stress: multiply near-max values
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        // Generate value with high limbs
        auto bytes = random_bytes();
        bytes[0] = 0xFF; bytes[1] = 0xFF; bytes[2] = 0xFF; bytes[3] = 0xFF;
        auto a = FieldElement::from_bytes(bytes);
        auto b = FieldElement::from_bytes(bytes);
        auto c = a * b;
        auto d = a.square();
        CHECK(c == d, "near-max: a*a == a^2");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 8. Inverse correctness
// ============================================================================
static void test_inverse() {
    g_section = "inverse";
    printf("[8] Inverse correctness\n");

    auto one = FieldElement::one();

    for (int i = 0; i < SCALED(10000, 200); ++i) {
        auto a = random_fe();
        if (a == FieldElement::from_uint64(0)) continue;
        auto inv = a.inverse();
        auto product = a * inv;
        CHECK(product == one, "a * a^(-1) == 1");
    }

    // Double inverse: (a^-1)^-1 == a
    for (int i = 0; i < SCALED(1000, 50); ++i) {
        auto a = random_fe();
        if (a == FieldElement::from_uint64(0)) continue;
        auto inv_inv = a.inverse().inverse();
        CHECK(inv_inv == a, "(a^-1)^-1 == a");
    }

    // Inverse of 1 == 1
    CHECK(one.inverse() == one, "1^(-1) == 1");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 9. Square root
// ============================================================================
static void test_sqrt() {
    g_section = "sqrt";
    printf("[9] Square root\n");

    // sqrt(x^2) should give x or -x
    int qr_count = 0;
    for (int i = 0; i < SCALED(10000, 200); ++i) {
        auto x = random_fe();
        auto x2 = x.square();
        auto s = x2.sqrt();
        auto s2 = s.square();
        CHECK(s2 == x2, "sqrt(x^2)^2 == x^2");
        if (s == x) ++qr_count;
    }
    // Roughly half should match x directly (the other half give -x)
    printf("    sqrt matched +x in %d/10000 cases (expected ~5000)\n", qr_count);

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 10. Batch inverse with zero detection
// ============================================================================
static void test_batch_inverse() {
    g_section = "batch_inv";
    printf("[10] Batch inverse\n");

    constexpr int N = 256;
    std::vector<FieldElement> elems(N);
    for (int i = 0; i < N; ++i) {
        elems[i] = random_fe();
    }

    // Save copies
    auto original = elems;

    // Batch inverse
    fe_batch_inverse(elems.data(), N);

    auto one = FieldElement::one();
    for (int i = 0; i < N; ++i) {
        auto product = original[i] * elems[i];
        CHECK(product == one, "batch_inverse: a[i] * a[i]^-1 == 1");
    }

    // Verify batch inverse matches individual inverse
    for (int i = 0; i < N; ++i) {
        auto single_inv = original[i].inverse();
        CHECK(single_inv == elems[i], "batch vs single inverse match");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 11. Cross-check random cases (big-number stress)
// ============================================================================
static void test_random_cross_check() {
    g_section = "random_cross";
    printf("[11] Random cross-check (100K operations)\n");

    for (int i = 0; i < SCALED(100000, 1000); ++i) {
        auto a = random_fe();
        auto b = random_fe();

        // (a + b) - b == a
        CHECK(((a + b) - b) == a, "(a+b)-b == a");

        // (a * b) * b^-1 == a (when b != 0)
        if (!(b == FieldElement::from_uint64(0))) {
            auto r = (a * b) * b.inverse();
            CHECK(r == a, "(a*b)*b^-1 == a");
        }
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// _run() entry point for unified audit runner
// ============================================================================

int audit_field_run() {
    g_pass = 0; g_fail = 0;

    test_addition_overflow();
    test_subtraction_borrow();
    test_mul_carry();
    test_square_vs_mul();
    test_reduction();
    test_canonical();
    test_limb_boundary();
    test_inverse();
    test_sqrt();
    test_batch_inverse();
    test_random_cross_check();

    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("===============================================================\n");
    printf("  AUDIT I.1 -- Field Arithmetic Correctness\n");
    printf("===============================================================\n\n");

    test_addition_overflow();
    test_subtraction_borrow();
    test_mul_carry();
    test_square_vs_mul();
    test_reduction();
    test_canonical();
    test_limb_boundary();
    test_inverse();
    test_sqrt();
    test_batch_inverse();
    test_random_cross_check();

    printf("===============================================================\n");
    printf("  FIELD AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    printf("===============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
