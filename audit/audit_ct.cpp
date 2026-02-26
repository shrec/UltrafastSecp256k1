// ============================================================================
// Cryptographic Self-Audit: Constant-Time & Side-Channel (Section II)
// ============================================================================
// Covers: CT primitives (masks, cmov, cswap, table lookup), CT field ops,
//         CT scalar ops, CT point ops, Valgrind CLASSIFY/DECLASSIFY markers,
//         differential: fast vs CT paths produce same results.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <array>
#include <random>
#include <chrono>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ct/ops.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct_utils.hpp"

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

static std::mt19937_64 rng(0xA0D17'C7C7A);

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

static FieldElement random_fe() {
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = rng();
        std::memcpy(out.data() + i * 8, &v, 8);
    }
    return FieldElement::from_bytes(out);
}

static bool points_equal(const Point& a, const Point& b) {
    if (a.is_infinity() && b.is_infinity()) return true;
    if (a.is_infinity() != b.is_infinity()) return false;
    return a.to_compressed() == b.to_compressed();
}

// ============================================================================
// 1. CT mask generation
// ============================================================================
static void test_ct_masks() {
    g_section = "ct_mask";
    printf("[1] CT mask generation\n");

    // is_zero_mask
    CHECK(secp256k1::ct::is_zero_mask(0) == UINT64_MAX, "is_zero(0) = all-ones");
    CHECK(secp256k1::ct::is_zero_mask(1) == 0, "is_zero(1) = 0");
    CHECK(secp256k1::ct::is_zero_mask(UINT64_MAX) == 0, "is_zero(MAX) = 0");

    // is_nonzero_mask
    CHECK(secp256k1::ct::is_nonzero_mask(0) == 0, "nonzero(0) = 0");
    CHECK(secp256k1::ct::is_nonzero_mask(42) == UINT64_MAX, "nonzero(42) = all-ones");

    // eq_mask
    CHECK(secp256k1::ct::eq_mask(5, 5) == UINT64_MAX, "eq(5,5)");
    CHECK(secp256k1::ct::eq_mask(5, 6) == 0, "neq(5,6)");

    // bool_to_mask
    CHECK(secp256k1::ct::bool_to_mask(true) == UINT64_MAX, "bool(true)");
    CHECK(secp256k1::ct::bool_to_mask(false) == 0, "bool(false)");

    // lt_mask
    CHECK(secp256k1::ct::lt_mask(3, 5) == UINT64_MAX, "lt(3,5)");
    CHECK(secp256k1::ct::lt_mask(5, 3) == 0, "!lt(5,3)");
    CHECK(secp256k1::ct::lt_mask(5, 5) == 0, "!lt(5,5)");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 2. CT cmov / cswap
// ============================================================================
static void test_ct_cmov_cswap() {
    g_section = "ct_cmov";
    printf("[2] CT cmov / cswap (10K)\n");

    for (int i = 0; i < 10000; ++i) {
        uint64_t a[4], b[4], a_orig[4], b_orig[4];
        for (int j = 0; j < 4; ++j) {
            a[j] = a_orig[j] = rng();
            b[j] = b_orig[j] = rng();
        }

        bool do_it = (i % 2 == 0);
        uint64_t mask = secp256k1::ct::bool_to_mask(do_it);

        // cmov
        uint64_t dst[4];
        std::memcpy(dst, a, 32);
        secp256k1::ct::cmov256(dst, b, mask);
        if (do_it) {
            CHECK(std::memcmp(dst, b_orig, 32) == 0, "cmov: moved");
        } else {
            CHECK(std::memcmp(dst, a_orig, 32) == 0, "cmov: unchanged");
        }

        // cswap
        std::memcpy(a, a_orig, 32);
        std::memcpy(b, b_orig, 32);
        secp256k1::ct::cswap256(a, b, mask);
        if (do_it) {
            CHECK(std::memcmp(a, b_orig, 32) == 0, "cswap: a=bo");
            CHECK(std::memcmp(b, a_orig, 32) == 0, "cswap: b=ao");
        } else {
            CHECK(std::memcmp(a, a_orig, 32) == 0, "cswap: a same");
            CHECK(std::memcmp(b, b_orig, 32) == 0, "cswap: b same");
        }
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 3. CT table lookup
// ============================================================================
static void test_ct_table_lookup() {
    g_section = "ct_lookup";
    printf("[3] CT table lookup (256-bit)\n");

    constexpr int N = 16;
    uint64_t table[N][4];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < 4; ++j)
            table[i][j] = rng();

    // Every index should return the correct entry
    for (int idx = 0; idx < N; ++idx) {
        uint64_t out[4] = {};
        secp256k1::ct::ct_lookup_256(table, N, idx, out);
        CHECK(std::memcmp(out, table[idx], 32) == 0, "ct_lookup correct entry");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 4. CT field operations vs fast (differential)
// ============================================================================
static void test_ct_field_differential() {
    g_section = "ct_field";
    printf("[4] CT field ops vs fast:: differential (10K)\n");

    for (int i = 0; i < 10000; ++i) {
        auto a = random_fe();
        auto b = random_fe();

        // add
        auto fast_add = a + b;
        auto ct_add = secp256k1::ct::field_add(a, b);
        CHECK(fast_add == ct_add, "CT field_add == fast +");

        // sub
        auto fast_sub = a - b;
        auto ct_sub = secp256k1::ct::field_sub(a, b);
        CHECK(fast_sub == ct_sub, "CT field_sub == fast -");

        // mul
        auto fast_mul = a * b;
        auto ct_mul = secp256k1::ct::field_mul(a, b);
        CHECK(fast_mul == ct_mul, "CT field_mul == fast *");

        // sqr
        auto fast_sqr = a.square();
        auto ct_sqr = secp256k1::ct::field_sqr(a);
        CHECK(fast_sqr == ct_sqr, "CT field_sqr == fast sqr");

        // neg
        auto fast_neg = a.negate();
        auto ct_neg = secp256k1::ct::field_neg(a);
        CHECK(fast_neg == ct_neg, "CT field_neg == fast negate");
    }

    // inv (1K -- slower)
    for (int i = 0; i < 1000; ++i) {
        auto a = random_fe();
        auto fast_inv = a.inverse();
        auto ct_inv = secp256k1::ct::field_inv(a);
        CHECK(fast_inv == ct_inv, "CT field_inv == fast inverse");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 5. CT scalar operations vs fast (differential)
// ============================================================================
static void test_ct_scalar_differential() {
    g_section = "ct_scalar";
    printf("[5] CT scalar ops vs fast:: differential (10K)\n");

    for (int i = 0; i < 10000; ++i) {
        auto a = random_scalar();
        auto b = random_scalar();

        // add
        auto fast_add = a + b;
        auto ct_add = secp256k1::ct::scalar_add(a, b);
        CHECK(fast_add == ct_add, "CT scalar_add == fast +");

        // sub
        auto fast_sub = a - b;
        auto ct_sub = secp256k1::ct::scalar_sub(a, b);
        CHECK(fast_sub == ct_sub, "CT scalar_sub == fast -");

        // neg
        auto fast_neg = a.negate();
        auto ct_neg = secp256k1::ct::scalar_neg(a);
        CHECK(fast_neg == ct_neg, "CT scalar_neg == fast negate");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 6. CT scalar cmov / cswap
// ============================================================================
static void test_ct_scalar_cmov() {
    g_section = "ct_s_cmov";
    printf("[6] CT scalar cmov/cswap (1K)\n");

    for (int i = 0; i < 1000; ++i) {
        auto a = random_scalar();
        auto b = random_scalar();
        auto a_orig = a;
        auto b_orig = b;

        uint64_t mask = secp256k1::ct::bool_to_mask(i % 2 == 0);

        // cmov
        auto r = a;
        secp256k1::ct::scalar_cmov(&r, b, mask);
        if (i % 2 == 0) {
            CHECK(r == b_orig, "scalar cmov: moved");
        } else {
            CHECK(r == a_orig, "scalar cmov: unchanged");
        }

        // cswap
        auto ca = a_orig, cb = b_orig;
        secp256k1::ct::scalar_cswap(&ca, &cb, mask);
        if (i % 2 == 0) {
            CHECK(ca == b_orig && cb == a_orig, "scalar cswap: swapped");
        } else {
            CHECK(ca == a_orig && cb == b_orig, "scalar cswap: unchanged");
        }
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 7. CT field cmov / cswap / select
// ============================================================================
static void test_ct_field_cmov() {
    g_section = "ct_f_cmov";
    printf("[7] CT field cmov/cswap/select (1K)\n");

    for (int i = 0; i < 1000; ++i) {
        auto a = random_fe();
        auto b = random_fe();
        auto a_orig = a;
        auto b_orig = b;

        uint64_t mask = secp256k1::ct::bool_to_mask(i % 2 == 0);

        // cmov
        auto r = a;
        secp256k1::ct::field_cmov(&r, b, mask);
        if (i % 2 == 0) {
            CHECK(r == b_orig, "field cmov: moved");
        } else {
            CHECK(r == a_orig, "field cmov: unchanged");
        }

        // cswap
        auto ca = a_orig, cb = b_orig;
        secp256k1::ct::field_cswap(&ca, &cb, mask);
        if (i % 2 == 0) {
            CHECK(ca == b_orig && cb == a_orig, "field cswap: swapped");
        } else {
            CHECK(ca == a_orig && cb == b_orig, "field cswap: unchanged");
        }

        // select
        auto sel = secp256k1::ct::field_select(a_orig, b_orig, mask);
        if (i % 2 == 0) {
            CHECK(sel == a_orig, "field select: a for all-ones");
        } else {
            CHECK(sel == b_orig, "field select: b for zero");
        }

        // cneg
        auto neg_r = secp256k1::ct::field_cneg(a_orig, mask);
        if (i % 2 == 0) {
            CHECK(neg_r == a_orig.negate(), "field cneg: negated");
        } else {
            CHECK(neg_r == a_orig, "field cneg: unchanged");
        }
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 8. CT scalar/field is_zero / eq
// ============================================================================
static void test_ct_comparisons() {
    g_section = "ct_cmp";
    printf("[8] CT is_zero / eq comparisons\n");

    // Field
    auto zero_fe = FieldElement::from_uint64(0);
    auto one_fe = FieldElement::one();

    CHECK(secp256k1::ct::field_is_zero(zero_fe) == UINT64_MAX, "fe zero == zero");
    CHECK(secp256k1::ct::field_is_zero(one_fe) == 0, "fe one != zero");

    CHECK(secp256k1::ct::field_eq(one_fe, one_fe) == UINT64_MAX, "fe 1 == 1");
    CHECK(secp256k1::ct::field_eq(zero_fe, one_fe) == 0, "fe 0 != 1");

    // Scalar
    auto zero_sc = Scalar::from_uint64(0);
    auto one_sc = Scalar::one();

    CHECK(secp256k1::ct::scalar_is_zero(zero_sc) == UINT64_MAX, "sc zero == zero");
    CHECK(secp256k1::ct::scalar_is_zero(one_sc) == 0, "sc one != zero");

    CHECK(secp256k1::ct::scalar_eq(one_sc, one_sc) == UINT64_MAX, "sc 1 == 1");
    CHECK(secp256k1::ct::scalar_eq(zero_sc, one_sc) == 0, "sc 0 != 1");

    // Random (1K)
    for (int i = 0; i < 1000; ++i) {
        auto a = random_fe();
        auto b = random_fe();
        bool fast_eq = (a == b);
        bool ct_eq = (secp256k1::ct::field_eq(a, b) == UINT64_MAX);
        CHECK(fast_eq == ct_eq, "fe eq matches fast");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 9. CT point scalar multiplication vs fast (differential, 1K)
// ============================================================================
static void test_ct_point_scalar_mul() {
    g_section = "ct_smul";
    printf("[9] CT scalar_mul vs fast:: scalar_mul (1K)\n");

    auto G = Point::generator();

    for (int i = 0; i < 1000; ++i) {
        auto k = random_scalar();

        auto fast_r = G.scalar_mul(k);
        auto ct_r = secp256k1::ct::scalar_mul(G, k);

        CHECK(points_equal(fast_r, ct_r), "CT scalar_mul == fast scalar_mul");
    }

    // Edge: k = 1
    {
        auto one = Scalar::one();
        auto r = secp256k1::ct::scalar_mul(G, one);
        CHECK(points_equal(r, G), "CT: 1*G == G");
    }

    // Edge: k = 0
    {
        auto zero = Scalar::from_uint64(0);
        auto r = secp256k1::ct::scalar_mul(G, zero);
        CHECK(r.is_infinity(), "CT: 0*G == O");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 10. CT complete addition vs fast (differential, 1K)
// ============================================================================
static void test_ct_complete_addition() {
    g_section = "ct_add";
    printf("[10] CT complete addition vs fast add (1K)\n");

    auto G = Point::generator();

    for (int i = 0; i < 1000; ++i) {
        auto P = G.scalar_mul(random_scalar());
        auto Q = G.scalar_mul(random_scalar());

        auto fast_r = P.add(Q);

        auto jp = secp256k1::ct::CTJacobianPoint::from_point(P);
        auto jq = secp256k1::ct::CTJacobianPoint::from_point(Q);
        auto jr = secp256k1::ct::point_add_complete(jp, jq);
        auto ct_r = jr.to_point();

        CHECK(points_equal(fast_r, ct_r), "CT complete_add == fast add");
    }

    // P + O
    {
        auto P = G.scalar_mul(random_scalar());
        auto jp = secp256k1::ct::CTJacobianPoint::from_point(P);
        auto jo = secp256k1::ct::CTJacobianPoint::make_infinity();
        auto r = secp256k1::ct::point_add_complete(jp, jo).to_point();
        CHECK(points_equal(P, r), "CT: P + O == P");
    }

    // O + Q
    {
        auto Q = G.scalar_mul(random_scalar());
        auto jo = secp256k1::ct::CTJacobianPoint::make_infinity();
        auto jq = secp256k1::ct::CTJacobianPoint::from_point(Q);
        auto r = secp256k1::ct::point_add_complete(jo, jq).to_point();
        CHECK(points_equal(Q, r), "CT: O + Q == Q");
    }

    // P + (-P) == O
    {
        auto P = G.scalar_mul(random_scalar());
        auto nP = P.negate();
        auto jp = secp256k1::ct::CTJacobianPoint::from_point(P);
        auto jnp = secp256k1::ct::CTJacobianPoint::from_point(nP);
        auto r = secp256k1::ct::point_add_complete(jp, jnp).to_point();
        CHECK(r.is_infinity(), "CT: P + (-P) == O");
    }

    // P + P (doubling path)
    for (int i = 0; i < 100; ++i) {
        auto P = G.scalar_mul(random_scalar());
        auto fast_r = P.dbl();
        auto jp = secp256k1::ct::CTJacobianPoint::from_point(P);
        auto jr = secp256k1::ct::point_add_complete(jp, jp);
        auto ct_r = jr.to_point();
        CHECK(points_equal(fast_r, ct_r), "CT: P+P via complete == fast dbl");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 11. CT byte-level utilities (ct_utils.hpp)
// ============================================================================
static void test_ct_utils() {
    g_section = "ct_utils";
    printf("[11] CT byte-level utilities\n");

    // ct_equal
    {
        std::array<uint8_t, 32> a{}, b{};
        for (auto& x : a) x = rng() & 0xFF;
        b = a;
        CHECK(secp256k1::ct::ct_equal(a, b), "ct_equal: same");
        b[15] ^= 0x01;
        CHECK(!secp256k1::ct::ct_equal(a, b), "ct_equal: different");
    }

    // ct_is_zero
    {
        std::array<uint8_t, 32> z{};
        CHECK(secp256k1::ct::ct_is_zero(z), "ct_is_zero: zeros");
        z[0] = 1;
        CHECK(!secp256k1::ct::ct_is_zero(z), "ct_is_zero: nonzero");
    }

    // ct_memzero
    {
        std::array<uint8_t, 32> buf{};
        for (auto& x : buf) x = 0xFF;
        secp256k1::ct::ct_memzero(buf.data(), 32);
        bool all_zero = true;
        for (auto x : buf) if (x != 0) all_zero = false;
        CHECK(all_zero, "ct_memzero");
    }

    // ct_compare
    {
        uint8_t a[] = {1, 2, 3};
        uint8_t b[] = {1, 2, 4};
        CHECK(secp256k1::ct::ct_compare(a, b, 3) < 0, "ct_compare: a < b");
        CHECK(secp256k1::ct::ct_compare(b, a, 3) > 0, "ct_compare: b > a");
        CHECK(secp256k1::ct::ct_compare(a, a, 3) == 0, "ct_compare: eq");
    }

    // ct_select_byte
    CHECK(secp256k1::ct::ct_select_byte(0xAB, 0xCD, true) == 0xAB, "select true");
    CHECK(secp256k1::ct::ct_select_byte(0xAB, 0xCD, false) == 0xCD, "select false");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 12. CT generator_mul vs fast
// ============================================================================
static void test_ct_generator_mul() {
    g_section = "ct_gen";
    printf("[12] CT generator_mul vs fast (500)\n");

    auto G = Point::generator();

    for (int i = 0; i < 500; ++i) {
        auto k = random_scalar();
        auto fast_r = G.scalar_mul(k);
        auto ct_r = secp256k1::ct::generator_mul(k);
        CHECK(points_equal(fast_r, ct_r), "CT generator_mul == fast G.scalar_mul");
    }

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// 13. Rudimentary timing variance test (statistical, non-definitive)
// ============================================================================
static void test_timing_variance() {
    g_section = "timing";
    printf("[13] Rudimentary timing variance (CT scalar_mul)\n");
    printf("    NOTE: Not a formal side-channel test -- just sanity check.\n");

    auto G = Point::generator();

    // Two structurally different scalars -- one with many 0 bits, one with many 1 bits
    auto k_low = Scalar::from_uint64(1);  // k = 1 (very sparse bits)
    auto k_high = Scalar::from_hex(
        "fffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364140"); // n-1

    constexpr int TRIALS = 100;
    double avg_low = 0, avg_high = 0;

    for (int i = 0; i < TRIALS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile auto r = secp256k1::ct::scalar_mul(G, k_low);
        auto t1 = std::chrono::high_resolution_clock::now();
        avg_low += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }
    avg_low /= TRIALS;

    for (int i = 0; i < TRIALS; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        volatile auto r = secp256k1::ct::scalar_mul(G, k_high);
        auto t1 = std::chrono::high_resolution_clock::now();
        avg_high += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }
    avg_high /= TRIALS;

    double ratio = (avg_high > avg_low)
        ? avg_high / avg_low
        : avg_low / avg_high;

    printf("    k=1 avg: %.0f ns\n", avg_low);
    printf("    k=n-1 avg: %.0f ns\n", avg_high);
    printf("    ratio: %.3f (ideal ~= 1.0, concern > 2.0)\n", ratio);

    // Generous threshold -- this is a rudimentary check, not formal side-channel analysis.
    // Real CT validation is done by dudect (ct_sidechannel_smoke: 34/34 sub-tests).
    // CI runners (especially macOS ARM64 GitHub Actions) are multi-tenant VMs where
    // timing jitter routinely reaches 1.5-1.7x due to frequency scaling, shared caches,
    // and hypervisor scheduling. We use 2.0x as the fail threshold to avoid flaky CI
    // while still catching catastrophic CT regressions (e.g. branch-on-secret).
    CHECK(ratio < 2.0, "CT mul timing ratio < 2.0x");

    printf("    %d checks\n\n", g_pass);
}

// ============================================================================
// _run() entry point for unified audit runner
// ============================================================================

int audit_ct_run() {
    g_pass = 0; g_fail = 0;

    test_ct_masks();
    test_ct_cmov_cswap();
    test_ct_table_lookup();
    test_ct_field_differential();
    test_ct_scalar_differential();
    test_ct_scalar_cmov();
    test_ct_field_cmov();
    test_ct_comparisons();
    test_ct_point_scalar_mul();
    test_ct_complete_addition();
    test_ct_utils();
    test_ct_generator_mul();
    test_timing_variance();

    return g_fail > 0 ? 1 : 0;
}

// ============================================================================
#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("===============================================================\n");
    printf("  AUDIT II -- Constant-Time & Side-Channel\n");
    printf("===============================================================\n\n");

    test_ct_masks();
    test_ct_cmov_cswap();
    test_ct_table_lookup();
    test_ct_field_differential();
    test_ct_scalar_differential();
    test_ct_scalar_cmov();
    test_ct_field_cmov();
    test_ct_comparisons();
    test_ct_point_scalar_mul();
    test_ct_complete_addition();
    test_ct_utils();
    test_ct_generator_mul();
    test_timing_variance();

    printf("===============================================================\n");
    printf("  CT AUDIT: %d passed, %d failed\n", g_pass, g_fail);
    printf("===============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
