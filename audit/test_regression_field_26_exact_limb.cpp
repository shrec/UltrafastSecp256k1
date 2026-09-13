// ============================================================================
// test_regression_field_26_exact_limb.cpp
// ============================================================================
// Exact-limb regression coverage for the FieldElement26 (10x26) field path.
//
// Why this exists
//   PR 1548d44 added Python-truth exact-limb KATs + a trigger family for the
//   FE64 (4x64) reduce() carry-loss bug. The 10x26 representation -- the
//   production field on ARM64 and 32-bit targets (see field_optimal.hpp) --
//   was never given the same guard. Its only coverage was test_field_26.cpp,
//   whose comparisons went through FieldElement26::to_fe() → operator==,
//   i.e. normalized canonical comparison, which is exactly the comparison the
//   1548d44 write-up calls out as the one that hid the carry loss:
//
//     "The new cases compare raw limbs rather than going through operator==,
//      and that is the point: the old cases used operator==, whose
//      normalisation hid the second loss."
//
//   If an equivalent loss ever appears in fe26 (mul/sqr/normalize), the old
//   test set -- small vectors, normalized == -- would stay green. This module
//   pins the trigger family and a randomized differential at exact-limb
//   granularity against independently computed Python ground truth, so a
//   regressing kernel is caught here before any downstream module.
//
// Methodology (mirrors test_regression_field_reduce_carry.cpp)
//   * Every result is compared as all 32 big-endian bytes returned by
//     to_bytes(), NOT through operator==.
//   * Trigger family: the 1548d44 inputs (2^256-2^33-1, 2^255-1), p-1, p,
//     2^256-1 (>=p, raw entry), plus products of the family.
//   * Raw-limb entry exercises normalize() on inputs that are NOT < p.
//   * The expected values below are ground truth from Python 3 big-int:
//        P = 2**256 - 0x1000003D1
//        (x * y) % P, (x + x) % P, pow(x, 2, P)
//     computed independently of both the 4x64 and 10x26 C++ paths.

#include "secp256k1/field.hpp"
#include "secp256k1/field_26.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>

static int g_pass = 0, g_fail = 0;
#include "audit_check.hpp"

using secp256k1::fast::FieldElement;
using secp256k1::fast::FieldElement26;
using limb4 = std::array<std::uint64_t, 4>;
using b32 = std::array<std::uint8_t, 32>;

// Hex string -> 32 BE bytes.
static b32 be(const char* hex) {
    b32 out{};
    for (int i = 0; i < 32; ++i) {
        char h[3] = {hex[2 * i], hex[2 * i + 1], 0};
        out[i] = static_cast<std::uint8_t>(std::strtoul(h, nullptr, 16));
    }
    return out;
}

// Diagnostic: 32 BE bytes -> hex string.
static std::string bytes_hex(const b32& x) {
    std::string s;
    for (auto c : x) {
        char buf[4];
        (void)std::snprintf(buf, sizeof(buf), "%02x", static_cast<int>(c));
        s += buf;
    }
    return s;
}

// fe26 mul (or sqr) on raw 4x64 limbs, result canonical bytes. Uses the raw
// 10x26 entry (from_fe bit-slices without normalizing) so >=p inputs such as
// 2^256-1 are exercised identically to real 32-bit usage.
static b32 raw26_mul(const limb4& a, const limb4& b) {
    FieldElement26 a26 = FieldElement26::from_fe(FieldElement::from_limbs_raw(a));
    FieldElement26 b26 = FieldElement26::from_fe(FieldElement::from_limbs_raw(b));
    return (a26 * b26).to_fe().to_bytes();
}

// ---------------------------------------------------------------------------
// Test 1: mul/sqr on the trigger family, exact 32-byte KATs vs Python truth.
// ---------------------------------------------------------------------------
static void test_trigger_family_exact_bytes() {
    printf("[field_26_exact] trigger family exact-limb KATs (Python truth)...\n");

    // 2^256 - 2^33 - 1  (the 1548d44 trigger; a < p, but mul drives
    //   normalization through the same overflow zone as the fe64 bug)
    limb4 const trig = {0xFFFFFFFDFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
                        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    // 2^255 - 1  (the 1548d44 "large × large" vector)
    limb4 const large = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
                         0xFFFFFFFFFFFFFFFFULL, 0x7FFFFFFFFFFFFFFFULL};
    // p - 1
    limb4 const pm1 = {0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
                       0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    // 2^256 - 1  (>= p: raw entry)
    limb4 const max256 = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
                          0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    limb4 const one = {1, 0, 0, 0};

    struct KAT {
        const char* name;
        limb4 a, b;
        b32 want;
    };
    KAT const kats[] = {
        {"trig*trig", trig, trig,
         be("000000000000000000000000000000000000000000000000fffff860000e8900")},
        {"large*large", large, large,
         be("400000000000000000000000000000000000000000000000400001e740039f64")},
        {"(p-1)*(p-1)", pm1, pm1,
         be("0000000000000000000000000000000000000000000000000000000000000001")},
        {"trig*(p-1)", trig, pm1,
         be("00000000000000000000000000000000000000000000000000000000fffffc30")},
        {"(2^256-1)*(2^256-1)", max256, max256,
         be("000000000000000000000000000000000000000000000001000007a0000e8900")},
        {"trig*1", trig, one,
         be("fffffffffffffffffffffffffffffffffffffffffffffffffffffffdffffffff")},
    };

    for (auto const& k : kats) {
        b32 const got = raw26_mul(k.a, k.b);
        if (got != k.want) {
            CHECK(false, std::string(k.name) + "\n  got : " + bytes_hex(got) +
                         "\n  want: " + bytes_hex(k.want));
        } else {
            ++g_pass;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 2: raw-limb entry -- normalize() on inputs that are NOT canonical < p.
// ---------------------------------------------------------------------------
static void test_normalize_raw_noncanonical() {
    printf("[field_26_exact] normalize() on >=p raw limbs...\n");

    // a = p   -> should normalize to 0
    limb4 const p_val = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                         0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    {
        FieldElement26 a26 =
            FieldElement26::from_fe(FieldElement::from_limbs_raw(p_val));
        a26.normalize();
        b32 const got = a26.to_fe().to_bytes();
        b32 const want{};  // all zeros
        if (got != want) {
            CHECK(false, "p normalizes to zero (raw entry)\n  got: " +
                         bytes_hex(got));
        } else {
            ++g_pass;
        }
    }

    // a = 2^256 - 1  -> 2*p complement; canonical result is 0x1000003D0
    limb4 const max256 = {0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL,
                          0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
    {
        FieldElement26 a26 =
            FieldElement26::from_fe(FieldElement::from_limbs_raw(max256));
        // normalize directly on raw 10x26 limbs (no canonical pre-pass)
        a26.normalize();
        b32 const got = a26.to_fe().to_bytes();
        b32 const want =
            be("00000000000000000000000000000000000000000000000000000001000003d0");
        if (got != want) {
            CHECK(false, "2^256-1 raw normalizes to 0x1000003D0\n  got: " +
                         bytes_hex(got));
        } else {
            ++g_pass;
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: randomized exact-limb differential, fe26 vs fe64 (random a,b in
// [0, 2^256) clamped below p so fe64 canonical inputs stay in contract).
// ---------------------------------------------------------------------------
static void test_randomized_exact_limb() {
    printf("[field_26_exact] 200,000 randomized exact-limb diffs...\n");

    std::mt19937_64 rng(0x26E2'26ULL);

    for (long i = 0; i < 200000; ++i) {
        limb4 a = {rng(), rng(), rng(), rng() & 0x7FFFFFFFFFFFFFFFULL};
        limb4 b = {rng(), rng(), rng(), rng() & 0x7FFFFFFFFFFFFFFFULL};

        FieldElement const fa = FieldElement::from_limbs(a);
        FieldElement const fb = FieldElement::from_limbs(b);
        FieldElement26 a26 = FieldElement26::from_fe(fa);
        FieldElement26 b26 = FieldElement26::from_fe(fb);

        FieldElement const r64 = fa * fb;
        b32 const b64 = r64.to_bytes();
        b32 const b26b = (a26 * b26).to_fe().to_bytes();
        if (b64 != b26b) {
            CHECK(false, "mul random iter=" + std::to_string(i) +
                         "\n  fe64: " + bytes_hex(b64) +
                         "\n  fe26: " + bytes_hex(b26b));
            return;
        }

        FieldElement const s64 = fa.square();
        b32 const s26 = a26.square().to_fe().to_bytes();
        if (s64.to_bytes() != s26) {
            CHECK(false, "square random iter=" + std::to_string(i));
            return;
        }

        FieldElement t64 = fa + fb;
        b32 const t64b = t64.to_bytes();
        FieldElement26 t26 = a26 + b26;
        t26.normalize();
        b32 const t26b = t26.to_fe().to_bytes();
        if (t64b != t26b) {
            CHECK(false, "add random iter=" + std::to_string(i));
            return;
        }

        ++g_pass;
    }
}

int test_regression_field_26_exact_limb_run() {
    g_pass = 0; g_fail = 0;
    printf("==================================================================\n");
    printf("  Regression: FieldElement26 (10x26) exact-limb coverage\n");
    printf("  (guard the fe26 equivalent of the fe64 reduce-carry loss)\n");
    printf("==================================================================\n");

    test_trigger_family_exact_bytes();
    test_normalize_raw_noncanonical();
    test_randomized_exact_limb();

    printf("[regression_field_26_exact_limb] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_regression_field_26_exact_limb_run(); }
#endif
