// ============================================================================
// Cryptographic Self-Audit: Scalar Arithmetic (secp256k1 order field)
// ============================================================================
// Covers: mod n reduction, overflow normalization, edge scalars (0, n, n-1),
//         GLV split correctness, 2^255 boundary, random fuzz.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

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

static std::mt19937_64 rng(0xA0D17'5CA1A);  // NOLINT(cert-msc32-c,cert-msc51-cpp)

static std::array<uint8_t, 32> random_bytes() {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + static_cast<std::size_t>(i) * 8, &v, 8);
    }
    return out;
}

static Scalar random_scalar() {
    for (;;) {
        auto s = Scalar::from_bytes(random_bytes());
        if (!s.is_zero()) return s;
    }
}

// ============================================================================
// 1. mod n reduction
// ============================================================================
static void test_mod_n_reduction() {
    g_section = "mod_n";
    printf("[1] Scalar mod n reduction\n");

    // n itself should reduce to 0
    auto n = Scalar::from_hex(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");
    CHECK(n.is_zero(), "n mod n == 0");

    // n+1 should reduce to 1
    auto n_plus_1 = Scalar::from_hex(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364142");
    CHECK(n_plus_1 == Scalar::one(), "(n+1) mod n == 1");

    // 2n should reduce to 0
    // 2n = 1FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDD5DB9CD5E9140777FA4BD19A06C8282
    // But from_bytes takes 32 bytes, 2n overflows 256 bits -> gets truncated+reduced
    // Instead test: n-1 + 1 == 0
    auto n_minus_1 = Scalar::from_hex(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140");
    auto sum = n_minus_1 + Scalar::one();
    CHECK(sum.is_zero(), "(n-1) + 1 == 0");

    // to_bytes -> from_bytes roundtrip
    for (int i = 0; i < 10000; ++i) {
        auto s = random_scalar();
        auto bytes = s.to_bytes();
        auto s2 = Scalar::from_bytes(bytes);
        CHECK(s == s2, "roundtrip");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 2. Scalar overflow normalization
// ============================================================================
static void test_overflow_normalization() {
    g_section = "overflow";
    printf("[2] Scalar overflow normalization\n");

    // Any 32-byte input from from_bytes should produce a valid scalar
    // n in big-endian (raw bytes, NOT parsed through Scalar which would reduce)
    const std::array<uint8_t, 32> n_bytes = {
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6,0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C,0xD0,0x36,0x41,0x41
    };

    for (int i = 0; i < 10000; ++i) {
        auto bytes = random_bytes();
        auto s = Scalar::from_bytes(bytes);
        // Verify serialized form is < n
        auto out = s.to_bytes();
        bool less = false;
        for (int j = 0; j < 32; ++j) {
            if (out[j] < n_bytes[j]) { less = true; break; }
            if (out[j] > n_bytes[j]) { CHECK(false, "scalar >= n"); break; }
        }
        // If we didn't find it less, it must be exactly n (which would be 0)
        if (!less) {
            // All equal means it's n, which reduces to 0
            bool all_zero = true;
            for (int j = 0; j < 32; ++j) {
                if (out[j] != 0) { all_zero = false; break; }
            }
            CHECK(all_zero, "if == n, must serialize as 0");
        }
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 3. Edge scalars
// ============================================================================
static void test_edge_scalars() {
    g_section = "edge";
    printf("[3] Edge scalar handling (0, 1, n-1, n, n+1)\n");

    auto zero = Scalar::from_uint64(0);
    auto one  = Scalar::one();
    auto n_minus_1 = Scalar::from_hex(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140");
    auto n = Scalar::from_hex(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141");

    // Zero
    CHECK(zero.is_zero(), "0 is zero");
    CHECK(!(one.is_zero()), "1 is not zero");

    // n reduces to 0
    CHECK(n.is_zero(), "n == 0");

    // n-1 + 1 == 0
    CHECK((n_minus_1 + one).is_zero(), "(n-1) + 1 == 0");

    // n-1 * n-1 == 1 (since (-1)^2 = 1)
    auto prod = n_minus_1 * n_minus_1;
    CHECK(prod == one, "(n-1)^2 == 1");

    // 0 * anything == 0
    for (int i = 0; i < 100; ++i) {
        auto s = random_scalar();
        CHECK((zero * s).is_zero(), "0 * s == 0");
    }

    // 1 * anything == anything  
    for (int i = 0; i < 100; ++i) {
        auto s = random_scalar();
        CHECK((one * s) == s, "1 * s == s");
    }

    // negate(n-1) == 1
    CHECK(n_minus_1.negate() == one, "-(n-1) == 1");

    // negate(1) == n-1
    CHECK(one.negate() == n_minus_1, "-(1) == n-1");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 4. Scalar arithmetic laws (10K random)
// ============================================================================
static void test_scalar_laws() {
    g_section = "laws";
    printf("[4] Scalar arithmetic laws (10K random)\n");

    for (int i = 0; i < 10000; ++i) {
        auto a = random_scalar();
        auto b = random_scalar();
        auto c = random_scalar();

        // Commutativity
        CHECK((a + b) == (b + a), "a+b == b+a");
        CHECK((a * b) == (b * a), "a*b == b*a");

        // Associativity
        CHECK(((a + b) + c) == (a + (b + c)), "(a+b)+c == a+(b+c)");
        CHECK(((a * b) * c) == (a * (b * c)), "(a*b)*c == a*(b*c)");

        // Distributivity
        CHECK((a * (b + c)) == ((a * b) + (a * c)), "a*(b+c) == a*b + a*c");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 5. Inverse
// ============================================================================
static void test_scalar_inverse() {
    g_section = "inverse";
    printf("[5] Scalar inverse\n");

    auto one = Scalar::one();

    for (int i = 0; i < 10000; ++i) {
        auto a = random_scalar();
        auto inv = a.inverse();
        CHECK((a * inv) == one, "a * a^-1 == 1");
    }

    // Double inverse
    for (int i = 0; i < 1000; ++i) {
        auto a = random_scalar();
        CHECK(a.inverse().inverse() == a, "(a^-1)^-1 == a");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 6. GLV split correctness
// ============================================================================
static void test_glv_split() {
    g_section = "glv";
    printf("[6] GLV split correctness via point arithmetic\n");

    auto G = Point::generator();

    // For random k, verify k*G computed via GLV matches direct scalar_mul
    // (scalar_mul internally uses GLV, but we verify algebraic identity)
    for (int i = 0; i < 1000; ++i) {
        auto k = random_scalar();

        // Compute k*G
        auto P = G.scalar_mul(k);
        CHECK(!P.is_infinity() || k.is_zero(), "k*G consistency");

        // Verify: (k+1)*G == k*G + G
        auto k_plus_1 = k + Scalar::one();
        auto P2 = G.scalar_mul(k_plus_1);
        auto P3 = P.add(G);
        CHECK(P2.to_compressed() == P3.to_compressed(), "(k+1)*G == k*G + G");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 7. 2^255 boundary
// ============================================================================
static void test_high_bits() {
    g_section = "high_bits";
    printf("[7] High-bit boundary (2^255 region)\n");

    // 2^255
    std::array<uint8_t, 32> bytes{};
    bytes[0] = 0x80;  // big-endian: bit 255 set
    auto s = Scalar::from_bytes(bytes);
    // This should be a valid scalar (2^255 < n)
    CHECK(!s.is_zero(), "2^255 is non-zero mod n");

    // 2^255 - 1
    bytes[0] = 0x7F;
    for (int i = 1; i < 32; ++i) bytes[i] = 0xFF;
    auto s2 = Scalar::from_bytes(bytes);
    CHECK(!s2.is_zero(), "2^255 - 1 is non-zero mod n");

    // Operations near 2^255
    auto sum = s + s2;
    CHECK(!sum.is_zero(), "2^255 + (2^255-1) non-zero");

    // Multiplication near boundary
    auto product = s * s2;
    auto product2 = s2 * s;
    CHECK(product == product2, "commutative near 2^255");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 8. Negate self-consistency
// ============================================================================
static void test_negate() {
    g_section = "negate";
    printf("[8] Negate self-consistency\n");

    for (int i = 0; i < 10000; ++i) {
        auto a = random_scalar();
        auto neg = a.negate();
        CHECK((a + neg).is_zero(), "a + (-a) == 0");
        CHECK(neg.negate() == a, "-(-a) == a");
    }

    // negate(0) == 0
    CHECK(Scalar::from_uint64(0).negate().is_zero(), "-(0) == 0");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// _run() entry point for unified audit runner
// ============================================================================

int audit_scalar_run() {
    g_pass = 0; g_fail = 0;

    test_mod_n_reduction();
    test_overflow_normalization();
    test_edge_scalars();
    test_scalar_laws();
    test_scalar_inverse();
    test_glv_split();
    test_high_bits();
    test_negate();

    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("===============================================================\n");
    printf("  AUDIT I.2 -- Scalar Arithmetic Correctness\n");
    printf("===============================================================\n\n");

    test_mod_n_reduction();
    test_overflow_normalization();
    test_edge_scalars();
    test_scalar_laws();
    test_scalar_inverse();
    test_glv_split();
    test_high_bits();
    test_negate();

    printf("===============================================================\n");
    printf("  SCALAR AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    printf("===============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
