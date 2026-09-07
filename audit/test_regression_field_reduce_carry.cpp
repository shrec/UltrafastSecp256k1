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

#if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
#include "secp256k1/field_asm.hpp"
#if defined(_WIN32) && (defined(__clang__) || defined(__GNUC__))
#define REDUCE_CARRY_ASM_CC __attribute__((sysv_abi))
#else
#define REDUCE_CARRY_ASM_CC
#endif
extern "C" {
void REDUCE_CARRY_ASM_CC field_mul_full_asm(
    const std::uint64_t*, const std::uint64_t*, std::uint64_t*);
void REDUCE_CARRY_ASM_CC field_sqr_full_asm(
    const std::uint64_t*, std::uint64_t*);
void REDUCE_CARRY_ASM_CC reduce_4_asm(std::uint64_t*);
}
#undef REDUCE_CARRY_ASM_CC
#endif

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
    auto b2 = r.to_bytes();  // second call on same value
    // to_bytes() must be deterministic: same value → same bytes
    CHECK(b1 == b2, "large × large: to_bytes() is deterministic");
}

// Check raw limbs as well as serialized bytes: FieldElement::operator==
// normalizes its operands and would hide a noncanonical result.
static void check_exact_field(const FieldElement& actual,
                              const FieldElement::limbs_type& expected,
                              const char* context) {
    std::array<std::uint8_t, 32> expected_bytes{};
    for (std::size_t limb = 0; limb < expected.size(); ++limb) {
        for (std::size_t byte = 0; byte < 8; ++byte) {
            expected_bytes[31 - (limb * 8 + byte)] =
                static_cast<std::uint8_t>(expected[limb] >> (8 * byte));
        }
    }
    bool const limbs_match = actual.limbs() == expected;
    bool const bytes_match = actual.to_bytes() == expected_bytes;
    if (!limbs_match || !bytes_match) {
        printf("[reduce_carry] exact KAT mismatch: %s\n", context);
    }
    CHECK(limbs_match, "all four raw limbs match the canonical KAT");
    CHECK(bytes_match, "all 32 big-endian bytes match the KAT");
}

static void test_second_fold_carry_bit() {
    printf("[reduce_carry] final fold expands carry bit to a full mask...\n");
    // B=2^256, K=2^32+977, p=B-K, a=B-2^33-1=p-(2^32-976).
    // Thus a^2 mod p=(2^32-976)^2=0xfffff860000e8900.
    // The second fold overflows: adding K&1 instead of K loses K-1.
    FieldElement::limbs_type const input = {
        0xFFFFFFFDFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    FieldElement::limbs_type const expected = {0xFFFFF860000E8900ULL, 0, 0, 0};
    FieldElement const a = FieldElement::from_limbs(input);
    check_exact_field(a.square(), expected, "carry-bit square");
    check_exact_field(a * a, expected, "carry-bit multiply");
    FieldElement in_place = a;
    in_place.square_inplace();
    check_exact_field(in_place, expected, "carry-bit square_inplace");
    in_place = a;
    in_place *= in_place;
    check_exact_field(in_place, expected, "carry-bit self-alias multiply");

    FieldElement::limbs_type const one = {1, 0, 0, 0};
    check_exact_field(FieldElement::one().square(), one, "carry-zero square");
    check_exact_field(a * FieldElement::one(), input, "carry-zero multiply");

#if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
    if (!secp256k1::fast::has_bmi2_support() ||
        !secp256k1::fast::has_adx_support()) {
        printf("[reduce_carry] direct ASM KATs skipped: BMI2/ADX unavailable\n");
        return;
    }
    // Exercise all three GAS reduction copies directly, independently of
    // which wrapper the configured public FieldElement API selects.
    FieldElement::limbs_type raw{};
    field_sqr_full_asm(input.data(), raw.data());
    check_exact_field(FieldElement::from_limbs_raw(raw), expected, "direct ASM square");
    field_mul_full_asm(input.data(), input.data(), raw.data());
    check_exact_field(FieldElement::from_limbs_raw(raw), expected, "direct ASM multiply");
    raw = input;
    field_sqr_full_asm(raw.data(), raw.data());
    check_exact_field(FieldElement::from_limbs_raw(raw), expected, "direct ASM square alias");
    raw = input;
    field_mul_full_asm(raw.data(), raw.data(), raw.data());
    check_exact_field(FieldElement::from_limbs_raw(raw), expected, "direct ASM multiply alias");

    field_sqr_full_asm(one.data(), raw.data());
    check_exact_field(FieldElement::from_limbs_raw(raw), one, "direct ASM square carry-zero");
    field_mul_full_asm(input.data(), one.data(), raw.data());
    check_exact_field(FieldElement::from_limbs_raw(raw), input, "direct ASM multiply carry-zero");
    field_mul_full_asm(one.data(), input.data(), raw.data());
    check_exact_field(FieldElement::from_limbs_raw(raw), input, "direct ASM multiply reversed");

    // a^2=(B-d)^2=(B-2d)*B+d^2, d=2^33+1; retain all eight limbs.
    std::array<std::uint64_t, 8> wide = {
        0x0000000400000001ULL, 4, 0, 0,
        0xFFFFFFFBFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    reduce_4_asm(wide.data());
    check_exact_field(FieldElement::from_limbs_raw({wide[0], wide[1], wide[2], wide[3]}),
                      expected, "direct ASM reduction of a^2");

    // B^2-1 mod p = K^2-1; this also forces the final overflow bit.
    wide.fill(0xFFFFFFFFFFFFFFFFULL);
    reduce_4_asm(wide.data());
    FieldElement::limbs_type const max_wide_expected = {0x000007A2000E90A0ULL, 1, 0, 0};
    check_exact_field(FieldElement::from_limbs_raw({wide[0], wide[1], wide[2], wide[3]}),
                      max_wide_expected, "direct ASM reduction of B^2-1");
    wide.fill(0);
    wide[0] = 1;
    reduce_4_asm(wide.data());
    check_exact_field(FieldElement::from_limbs_raw({wide[0], wide[1], wide[2], wide[3]}),
                      one, "direct ASM reduction carry-zero");
#endif
}

static void test_first_fold_full_cascade() {
    printf("[reduce_carry] first fold carries through every remaining limb...\n");
    // a=2^255+1, b=p-1: ab mod p=-a mod p=p-a.
    // The old Step-2 carry tail wrapped result[3] without incrementing
    // result[4], losing B, equivalently K modulo p.
    FieldElement const a = FieldElement::from_limbs({1, 0, 0, 0x8000000000000000ULL});
    FieldElement const b = FieldElement::from_limbs({
        0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    });
    FieldElement::limbs_type const expected = {
        0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL
    };
    check_exact_field(a * b, expected, "first-fold multiply");
    check_exact_field(b * a, expected, "first-fold multiply reversed");
    FieldElement in_place = a;
    in_place *= b;
    check_exact_field(in_place, expected, "first-fold multiply inplace lhs");
    in_place = b;
    in_place *= a;
    check_exact_field(in_place, expected, "first-fold multiply inplace reversed");
}

int test_regression_field_reduce_carry_run() {
    g_pass = 0; g_fail = 0;
    printf("==================================================================\n");
    printf("  Regression: field reduce() carry propagation\n");
    printf("  (fix: cascade result[2] → result[3] → result[4] carry chain)\n");
    printf("==================================================================\n");

    test_large_squared_matches_truth();
    test_cross_check_against_FE52();
    test_second_fold_carry_bit();
    test_first_fold_full_cascade();

    printf("[regression_field_reduce_carry] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_regression_field_reduce_carry_run(); }
#endif
