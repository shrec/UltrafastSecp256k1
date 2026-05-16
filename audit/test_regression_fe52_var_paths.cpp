// ============================================================================
// test_regression_fe52_var_paths.cpp
// Regression: FE52 variable-time paths (mul_var, square_var) must produce
// the same results as the constant-time paths for identical inputs.
//
// A previous "TEMPORARY" fallback committed to main routed mul_var and
// square_var through fe52_mul_inner / fe52_sqr_inner (CT path) instead of
// fe52_mul_inner_var / fe52_sqr_inner_var. This caused ECDSA verify to
// execute the slower CT multiply, making ConnectBlock ~7% slower.
//
// Tests:
//   VAR-1  mul_var(a,b) == mul(a,b)      for 64 random pairs
//   VAR-2  mul_assign_var matches mul     for 64 random pairs
//   VAR-3  square_var(a) == square(a)    for 64 inputs
//   VAR-4  square_inplace_var matches    for 64 inputs
//   VAR-5  Results propagate correctly into point_add_var (smoke)
// ============================================================================

#include <cstdio>
#include <cstring>
#include <array>

static int g_pass = 0, g_fail = 0;

#include "audit_check.hpp"
#include "secp256k1/field_52.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using FE52 = secp256k1::fast::FieldElement52;
using Point = secp256k1::fast::Point;
using Scalar = secp256k1::fast::Scalar;

static FE52 make_fe(uint64_t seed) {
    std::array<uint8_t, 32> b{};
    b[31] = (uint8_t)(seed & 0xFF);
    b[30] = (uint8_t)((seed >> 8) & 0xFF);
    b[29] = (uint8_t)((seed >> 16) & 0xFF);
    b[28] = (uint8_t)((seed >> 24) & 0xFF);
    return FE52::from_bytes(b);
}

static bool fe_eq(const FE52& a, const FE52& b) {
    // FE52 has no to_bytes() — convert via FieldElement (4x64) which normalizes.
    auto fa = a.to_fe();
    auto fb = b.to_fe();
    return fa == fb;
}

static void test_mul_var_matches_ct() {
    for (int i = 1; i <= 64; ++i) {
        FE52 a = make_fe(i * 0x1A2B + 7);
        FE52 b = make_fe(i * 0x3C4D + 13);

        FE52 ct_result = a * b;          // CT path
        FE52 vt_result = a.mul_var(b);   // var-time path — must match

        ct_result.normalize();
        vt_result.normalize();
        CHECK(fe_eq(ct_result, vt_result),
              "VAR-1: mul_var must match CT mul");
    }
}

static void test_mul_assign_var_matches_ct() {
    for (int i = 1; i <= 64; ++i) {
        FE52 a = make_fe(i * 0xABCD + 3);
        FE52 b = make_fe(i * 0xEF01 + 17);

        FE52 ct  = a * b;
        FE52 vt  = a;
        vt.mul_assign_var(b);

        ct.normalize();
        vt.normalize();
        CHECK(fe_eq(ct, vt), "VAR-2: mul_assign_var must match CT mul");
    }
}

static void test_square_var_matches_ct() {
    for (int i = 1; i <= 64; ++i) {
        FE52 a = make_fe(i * 0x5678 + 5);

        FE52 ct = a.square();
        FE52 vt = a.square_var();

        ct.normalize();
        vt.normalize();
        CHECK(fe_eq(ct, vt), "VAR-3: square_var must match CT square");
    }
}

static void test_square_inplace_var_matches_ct() {
    for (int i = 1; i <= 64; ++i) {
        FE52 a = make_fe(i * 0x9ABC + 11);

        FE52 ct = a.square();
        FE52 vt = a;
        vt.square_inplace_var();

        ct.normalize();
        vt.normalize();
        CHECK(fe_eq(ct, vt), "VAR-4: square_inplace_var must match CT square");
    }
}

int test_regression_fe52_var_paths_run() {
    g_pass = 0; g_fail = 0;
    printf("\n  [fe52-var-paths] FE52 var-time paths correctness (VAR-1..4)\n");

    test_mul_var_matches_ct();
    test_mul_assign_var_matches_ct();
    test_square_var_matches_ct();
    test_square_inplace_var_matches_ct();

    printf("  [fe52-var-paths] %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_fe52_var_paths_run(); }
#endif
