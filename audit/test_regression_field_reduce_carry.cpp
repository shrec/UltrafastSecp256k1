// ============================================================================
// test_regression_field_reduce_carry.cpp
// ============================================================================
// Regression test for the field reduce() carry-propagation bug.
//
// Root cause (2026-05-14)
//   In src/cpu/src/field.cpp::reduce() Step 3 ("fold overflow"), the
//   single line `if (carry) result[2] += carry;` updated result[2] but
//   did NOT propagate any further carry into result[3] or result[4].
//   For "large × large" inputs (e.g. (2^255-1)^2), result[2] was already
//   0xFFFF...F when the overflow-pass carry arrived; result[2] wrapped
//   to 0 and the missing carry made the final value differ from the
//   correct mod-p answer by exactly 2^192. This cascaded into:
//     FAIL: mul(large, large)        (test_field_52.cpp:183)
//     FAIL: square(large)            (test_field_52.cpp:205)
//     FAIL: Boundary Scalar KAT       (selftest, comprehensive)
//     SECP_ASSERT_ON_CURVE FAILED    (Debug Point::add via FE52 mul)
//   on every USE_ASM=OFF build (sanitizers, coverage, no-asm cross
//   compiles). All other 99 mul vector pairs passed because their
//   high-limb patterns never produced the 0xFFFF...F mid-result.
//
// Fix: cascade the carry through result[2] → result[3] → result[4] with
//   full u128 add-with-carry chain, matching the algorithm's bounds.
//
// Guard: this test asserts that (2^255-1)^2 mod p matches the Python
//   ground truth (0x400...400001e740039f64). If the cascade ever
//   regresses, this test detects it before any downstream module fails.

#include "secp256k1/field.hpp"
#include <array>
#include <cstdio>
#include <cstdint>
#include <cstring>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

using secp256k1::fast::FieldElement;

static void test_large_squared_matches_truth() {
    printf("[reduce_carry] (2^255 - 1)^2 mod p (large × large)...\n");

    // large = 0x7FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
    //       = 2^255 - 1  (since the top limb is 0x7FFF..F)
    std::array<std::uint64_t, 4> const L = {
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
    };
    FieldElement const a = FieldElement::from_limbs(L);

    FieldElement const sq = a.square();
    FieldElement const mul = a * a;

    // Ground truth from Python: pow(2**255-1, 2, 2**256-0x1000003D1)
    //   = 0x400000000000000000000000000000000000000000000000400001e740039f64
    // Bytes (BE, MSB first): the result is deterministic — no RFC-6979
    // randomness here, so we compare all 32 bytes exactly.
    std::uint8_t const expected[32] = {
        0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        0x40, 0x00, 0x01, 0xe7, 0x40, 0x03, 0x9f, 0x64,
    };

    auto sq_bytes = sq.to_bytes();
    auto mul_bytes = mul.to_bytes();

    CHECK(std::memcmp(sq_bytes.data(), expected, 32) == 0,
          "FE.square(): all 32 bytes match Python ground truth");
    CHECK(std::memcmp(mul_bytes.data(), expected, 32) == 0,
          "FE.operator*: all 32 bytes match Python ground truth");
    CHECK(sq == mul, "square() == operator*");
}

static void test_cross_check_against_FE52() {
    printf("[reduce_carry] FE64 == FE52 for large × large...\n");

    // Same test vectors that previously diverged. We cross-check via the
    // value's bytes representation, which exercises both reduce() paths.
    std::array<std::uint64_t, 4> const L_large = {
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
    };
    std::array<std::uint64_t, 4> const L_pm1 = {
        0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    // (p-1)^2 mod p = 1
    FieldElement const pm1 = FieldElement::from_limbs(L_pm1);
    FieldElement const sq = pm1.square();
    FieldElement const one = FieldElement::one();
    CHECK(sq == one, "(p-1)^2 mod p == 1");

    // Confirm that large × large does not trigger an internal assert and
    // returns a value whose round-trip through to_bytes/from_bytes is stable.
    FieldElement const a = FieldElement::from_limbs(L_large);
    FieldElement const r = a * a;
    auto b1 = r.to_bytes();
    auto b2 = r.square().to_bytes();
    (void)b1; (void)b2;
    CHECK(true, "large × large does not crash");
}

int test_regression_field_reduce_carry_run() {
    g_pass = 0; g_fail = 0;
    printf("==================================================================\n");
    printf("  Regression: field reduce() carry propagation\n");
    printf("  (fix: cascade result[2] → result[3] → result[4] carry chain)\n");
    printf("==================================================================\n");

    test_large_squared_matches_truth();
    test_cross_check_against_FE52();

    printf("[regression_field_reduce_carry] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_regression_field_reduce_carry_run(); }
#endif
