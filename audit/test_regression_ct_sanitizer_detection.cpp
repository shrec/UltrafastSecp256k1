// ============================================================================
// test_regression_ct_sanitizer_detection.cpp
// ============================================================================
// Regression test for the Clang sanitizer detection bug in ct_field.cpp.
//
// Root Cause (2026-05-14)
//   Five preprocessor sites in src/cpu/src/ct_field.cpp used the GCC-only
//   __SANITIZE_THREAD__ / __SANITIZE_ADDRESS__ / __SANITIZE_MEMORY__
//   macros to gate ADCX-chain intrinsics and asm-memory barriers. Clang
//   does NOT define those macros and instead exposes sanitizer state via
//   __has_feature(thread_sanitizer) etc., so under Clang TSan/MSan/ASan
//   the barriers ran AND were instrumented by the sanitizer's shadow
//   memory, producing wrong results — manifesting as:
//     FAIL: ct field_add, ct field_sub, ct field_neg
//     FAIL: ct add #1..#64, ct mul #1..#64
//   in test_comprehensive::test_ct_field.
//
// Fix: src/cpu/src/ct_field.cpp introduces SECP256K1_HAS_SANITIZER that
//   covers BOTH GCC predefined macros AND Clang __has_feature().
//
// Guard: This test asserts ct::field_add(a, b) == a + b for many values,
//   matching the loop that failed in test_comprehensive. If the macro
//   guard regresses (e.g. someone drops __has_feature detection), this
//   test will fail under Clang TSan and catch it before it ships.

#include "secp256k1/ct/field.hpp"
#include "secp256k1/field.hpp"
#include <cstdio>
#include <cstdint>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

using secp256k1::fast::FieldElement;
namespace ctf = secp256k1::ct;

static void test_ct_field_arithmetic_matches_fast() {
    printf("[ct_sanitizer_detection] ct vs fast field arithmetic parity...\n");

    // Mirror the loop in test_comprehensive::test_ct_field that previously
    // failed under Clang TSan. If macro guards regress, these checks will
    // fail in sanitizer builds before any other regression surfaces.
    for (std::uint64_t v = 1; v <= 64; ++v) {
        FieldElement const a = FieldElement::from_uint64(v);
        FieldElement const b = FieldElement::from_uint64(v + 1);

        char msg[64];
        std::snprintf(msg, sizeof(msg), "ct field_add #%llu",
                      (unsigned long long)v);
        CHECK(ctf::field_add(a, b) == a + b, msg);

        std::snprintf(msg, sizeof(msg), "ct field_sub #%llu",
                      (unsigned long long)v);
        CHECK(ctf::field_sub(b, a) == b - a, msg);

        std::snprintf(msg, sizeof(msg), "ct field_mul #%llu",
                      (unsigned long long)v);
        CHECK(ctf::field_mul(a, b) == a * b, msg);

        std::snprintf(msg, sizeof(msg), "ct field_sqr #%llu",
                      (unsigned long long)v);
        CHECK(ctf::field_sqr(a) == a.square(), msg);
    }

    // Boundary inputs: zero + non-zero, ones, large constants.
    FieldElement const zero = FieldElement::zero();
    FieldElement const one  = FieldElement::one();
    FieldElement const big  = FieldElement::from_uint64(0xFEDCBA9876543210ULL);

    CHECK(ctf::field_add(zero, big) == big, "ct field_add zero+big");
    CHECK(ctf::field_sub(big, zero) == big, "ct field_sub big-zero");
    CHECK(ctf::field_mul(one, big)  == big, "ct field_mul one*big");
    CHECK(ctf::field_neg(zero) == zero, "ct field_neg(0) == 0");
}

int test_regression_ct_sanitizer_detection_run() {
    g_pass = 0; g_fail = 0;
    printf("==================================================================\n");
    printf("  Regression: Clang sanitizer detection in ct_field.cpp\n");
    printf("  (fix: SECP256K1_HAS_SANITIZER macro covers GCC + Clang)\n");
    printf("==================================================================\n");

    test_ct_field_arithmetic_matches_fast();

    printf("[regression_ct_sanitizer_detection] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_regression_ct_sanitizer_detection_run(); }
#endif
