// ============================================================================
// test_regression_mul128_portability.cpp
// Regression: the Windows-ARM64 (clang-cl) port routes the 64x64->128 multiply
// off the x86-only MSVC intrinsics and onto the portable `unsigned __int128` path.
// Three sites carry this multiply (all -> __int128 on clang-cl/ARM64):
//   - precompute.cpp `_umul128`        (clang-cl/ARM64 -> portable __int128)
//   - detail/arith64.hpp `mulhi64`     (clang-cl -> __int128)
//   - field_asm.hpp `mulx64`           (clang-cl/ARM64 -> __uint128_t, off _mulx_u64)
// If the __int128 path ever disagreed with the x86 path, ARM64 results would
// silently diverge from x86 with no runtime test on ARM64 (the cross-built exe
// cannot execute on the x64 CI host). This test pins, on every host, that the
// portable 32-bit schoolbook multiply, the `unsigned __int128` multiply, and
// `detail::mulhi64` all agree on hi/lo of a*b over edge + random inputs.
// ============================================================================

#include <cstdio>
#include <cstdint>

static int g_pass = 0, g_fail = 0;

#include "audit_check.hpp"
#include "secp256k1/detail/arith64.hpp"

// These checks compare against `unsigned __int128` — which IS exactly the path the
// Windows-ARM64 clang-cl port routes to. MSVC `cl` has no __int128 (error C4235) and
// never takes that path, so the whole comparison is gated on __int128 availability;
// on cl the module is a no-op. clang-cl, GCC and Clang all define __SIZEOF_INT128__.
#if defined(__SIZEOF_INT128__)

// Portable 32-bit schoolbook 64x64->128 — byte-for-byte the SECP256K1_NO_INT128
// branch of precompute.cpp::_umul128. Kept here independently so the test does
// not depend on which path the engine was compiled with.
static std::uint64_t umul128_portable(std::uint64_t a, std::uint64_t b, std::uint64_t* hi) {
    std::uint32_t a_lo = (std::uint32_t)a, a_hi = (std::uint32_t)(a >> 32);
    std::uint32_t b_lo = (std::uint32_t)b, b_hi = (std::uint32_t)(b >> 32);
    std::uint64_t p0 = (std::uint64_t)a_lo * b_lo;
    std::uint64_t p1 = (std::uint64_t)a_lo * b_hi;
    std::uint64_t p2 = (std::uint64_t)a_hi * b_lo;
    std::uint64_t p3 = (std::uint64_t)a_hi * b_hi;
    std::uint64_t mid = p1 + (p0 >> 32);
    mid += p2;
    if (mid < p2) p3 += 0x100000000ULL;
    *hi = p3 + (mid >> 32);
    return (mid << 32) | (std::uint32_t)p0;
}

static void check_mul(std::uint64_t a, std::uint64_t b) {
    std::uint64_t hp = 0, lp = umul128_portable(a, b, &hp);
    unsigned __int128 const r = (unsigned __int128)a * b;
    std::uint64_t const hi128 = (std::uint64_t)(r >> 64);
    std::uint64_t const lo128 = (std::uint64_t)r;
    CHECK(lp == lo128 && hp == hi128, "umul128 portable schoolbook == __int128 multiply");
    CHECK(secp256k1::detail::mulhi64(a, b) == hi128, "detail::mulhi64 == high 64 of __int128 multiply");
}

// Subtract-with-borrow: field_asm.cpp subborrow64 + arith64.hpp sub64 route clang-cl/
// ARM64 onto the __int128 path (off the x86-only _subborrow_u64). Pin __int128 == portable.
static void check_subborrow(std::uint64_t a, std::uint64_t b, std::uint8_t bin) {
    unsigned __int128 const d = (unsigned __int128)a - b - bin;
    std::uint64_t const res128 = (std::uint64_t)d;
    std::uint8_t const bout128 = (std::uint8_t)((d >> 127) & 1);
    std::uint64_t const t = a - bin;       std::uint8_t b1 = (a < bin);
    std::uint64_t const resp = t - b;      std::uint8_t b2 = (t < b);
    CHECK(resp == res128 && (std::uint8_t)(b1 | b2) == bout128, "subborrow portable == __int128");
}

// Add-with-carry: field_asm.cpp adcx64 (#else) + arith64.hpp add64 route clang-cl/ARM64
// onto the __int128 path (off the x86-only _addcarry_u64). Pin __int128 == portable.
static void check_addcarry(std::uint64_t a, std::uint64_t b, std::uint8_t cin) {
    unsigned __int128 const s = (unsigned __int128)a + b + cin;
    std::uint64_t const res128 = (std::uint64_t)s;
    std::uint8_t const cout128 = (std::uint8_t)(s >> 64);
    std::uint64_t const sum = a + b;       std::uint8_t c1 = (sum < a);
    std::uint64_t const resp = sum + cin;  std::uint8_t c2 = (resp < sum);
    CHECK(resp == res128 && (std::uint8_t)(c1 | c2) == cout128, "addcarry portable == __int128");
}

#endif // __SIZEOF_INT128__

int test_regression_mul128_portability_run() {
    g_pass = 0; g_fail = 0;
    printf("\n  [mul128-portability] 64x64->128 multiply path equivalence (Windows-ARM64 clang-cl port)\n");

#if !defined(__SIZEOF_INT128__)
    printf("  [mul128-portability] SKIP — __int128 unavailable (e.g. MSVC cl); the __int128 paths under test are not used by this compiler\n");
#else
    static const std::uint64_t edges[] = {
        0ULL, 1ULL, 2ULL, 0xFFFFFFFFULL, 0x100000000ULL,
        0xFFFFFFFFFFFFFFFFULL, 0x8000000000000000ULL, 0x9E3779B97F4A7C15ULL,
        0x0123456789ABCDEFULL, 0xFEDCBA9876543210ULL,
    };
    for (std::uint64_t a : edges)
        for (std::uint64_t b : edges)
            check_mul(a, b);

    for (std::uint64_t a : edges)
        for (std::uint64_t b : edges)
            for (std::uint8_t c = 0; c <= 1; ++c) { check_subborrow(a, b, c); check_addcarry(a, b, c); }

    std::uint64_t s = 0xDEADBEEF12345678ULL;
    auto rnd = [&]() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return s; };
    for (int i = 0; i < 20000; ++i) {
        std::uint64_t a = rnd(), b = rnd();
        check_mul(a, b);
        check_subborrow(a, b, (std::uint8_t)(rnd() & 1));
        check_addcarry(a, b, (std::uint8_t)(rnd() & 1));
    }
#endif // __SIZEOF_INT128__

    printf("  [mul128-portability] %d passed, %d failed\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_mul128_portability_run(); }
#endif
