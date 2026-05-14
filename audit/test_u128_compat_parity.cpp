// ============================================================================
// test_u128_compat_parity.cpp -- portable u128 struct parity with __int128
// ============================================================================
//
// Validates that secp256k1::detail::u128_compat (the portable 128-bit struct
// used on wasm32 and any target where SECP256K1_NO_INT128 is defined) produces
// byte-identical results to native unsigned __int128 for every operation
// pattern used by src/cpu/include/secp256k1/field_52_impl.hpp:
//
//   - 64x64 -> 128 multiplication: `(u128)x * y`
//   - composition from two u64s: `((u128)hi << 64) | lo`
//   - addition: u128 += u128, u128 += u64
//   - right shift: 0..127 bits
//   - left shift: 0..127 bits
//   - bitwise AND with u64
//
// This test runs both on the native __int128 path AND when forced to use the
// portable struct (via PARITY_FORCE_STRUCT). Comparing the two builds reveals
// any divergence in the portable implementation.
//
// Coverage: 10,000 deterministic random vectors per operation. Failures abort
// after the first 20 with full operand dump for diagnosis.
//
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#define STANDALONE_TEST
#endif

#include <cstdio>
#include <cstdint>
#include <initializer_list>

// We always test the portable struct (under a forced rename), regardless of
// whether the engine itself was built with NO_INT128 — this test verifies the
// struct in isolation.
#define SECP256K1_NO_INT128 1

#include "secp256k1/u128_compat.hpp"

// MSVC has neither __int128 nor the GCC/Clang extension we need for the
// parity comparison. On those targets the test is a no-op stub that returns
// SUCCESS — the portable struct is exercised in the engine's regular FE52
// tests instead.
#if !defined(__SIZEOF_INT128__)

#if defined(_MSC_VER)
#pragma message("test_u128_compat_parity: __int128 unavailable — stubbing test")
#endif

extern "C" {
}  // keep file non-empty under MSVC

int test_u128_compat_parity_run() {
    std::printf("[u128_compat_parity] skipped — target lacks __int128 (MSVC / 32-bit)\n");
    return 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() { return test_u128_compat_parity_run(); }
#endif

#else  // __SIZEOF_INT128__ present below

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

using u128_struct = ::secp256k1::detail::u128_compat;
using u128_native = unsigned __int128;

namespace {

int g_fail = 0;
int g_pass = 0;

template <typename T>
std::uint64_t u128_lo(T x) noexcept { return static_cast<std::uint64_t>(x); }

template <typename T>
std::uint64_t u128_hi(T x) noexcept { return static_cast<std::uint64_t>(x >> 64); }

template <>
std::uint64_t u128_lo<u128_struct>(u128_struct x) noexcept { return x.lo; }

template <>
std::uint64_t u128_hi<u128_struct>(u128_struct x) noexcept { return x.hi; }

#define CHECK_EQ(label, A, B) do {                                              \
    if (u128_lo(A) != u128_lo(B) || u128_hi(A) != u128_hi(B)) {                 \
        std::fprintf(stderr,                                                    \
            "[FAIL] %s: native={hi=%016lx lo=%016lx} struct={hi=%016lx lo=%016lx}\n", \
            (label),                                                            \
            (unsigned long)u128_hi(A), (unsigned long)u128_lo(A),               \
            (unsigned long)u128_hi(B), (unsigned long)u128_lo(B));              \
        if (++g_fail >= 20) {                                                   \
            std::fprintf(stderr, "  aborting after 20 divergences\n");          \
            return g_fail;                                                      \
        }                                                                       \
    } else {                                                                    \
        ++g_pass;                                                               \
    }                                                                           \
} while (0)

std::uint64_t pcg64(std::uint64_t& s) noexcept {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return s;
}

} // namespace

int test_u128_compat_parity_run() {
    g_pass = 0;
    g_fail = 0;
    std::uint64_t seed = 0xCAFEBABE12345678ULL;

    for (int iter = 0; iter < 10000; ++iter) {
        const std::uint64_t a = pcg64(seed);
        const std::uint64_t b = pcg64(seed);
        const std::uint64_t c = pcg64(seed);

        // -- 64x64 -> 128 multiplication --------------------------------------
        {
            u128_native nm = static_cast<u128_native>(a) * b;
            u128_struct  sm = u128_struct(a) * b;
            CHECK_EQ("mul64x64", nm, sm);
        }

        // -- compose from two u64s --------------------------------------------
        u128_native nc = (static_cast<u128_native>(a) << 64) | b;
        u128_struct  sc = (u128_struct(a) << 64) | b;
        CHECK_EQ("compose", nc, sc);

        // -- u128 + u128 -------------------------------------------------------
        {
            u128_native na = nc + static_cast<u128_native>(c);
            u128_struct  sa = sc + u128_struct(c);
            CHECK_EQ("add_u128", na, sa);
        }

        // -- u128 += u64 -------------------------------------------------------
        {
            u128_native ne = nc; ne += c;
            u128_struct  se = sc; se += c;
            CHECK_EQ("add_u64", ne, se);
        }

        // -- u128 += (u128 product) -------------------------------------------
        {
            u128_native np = nc; np += static_cast<u128_native>(a) * b;
            u128_struct  sp = sc; sp += u128_struct(a) * b;
            CHECK_EQ("add_prod", np, sp);
        }

        // -- right shift across boundary cases --------------------------------
        for (unsigned n : {0u, 1u, 12u, 32u, 52u, 63u, 64u, 100u, 127u}) {
            u128_native nrr = nc >> n;
            u128_struct  srr = sc >> n;
            if (u128_lo(nrr) != u128_lo(srr) || u128_hi(nrr) != u128_hi(srr)) {
                std::fprintf(stderr,
                    "[FAIL] >>%u: iter=%d native={hi=%016lx lo=%016lx} struct={hi=%016lx lo=%016lx}\n",
                    n, iter,
                    (unsigned long)u128_hi(nrr), (unsigned long)u128_lo(nrr),
                    (unsigned long)u128_hi(srr), (unsigned long)u128_lo(srr));
                if (++g_fail >= 20) return g_fail;
            } else {
                ++g_pass;
            }
        }

        // -- left shift across boundary cases ---------------------------------
        for (unsigned n : {0u, 1u, 12u, 32u, 52u, 63u, 64u, 100u, 127u}) {
            u128_native nrr = nc << n;
            u128_struct  srr = sc << n;
            if (u128_lo(nrr) != u128_lo(srr) || u128_hi(nrr) != u128_hi(srr)) {
                std::fprintf(stderr,
                    "[FAIL] <<%u: iter=%d native={hi=%016lx lo=%016lx} struct={hi=%016lx lo=%016lx}\n",
                    n, iter,
                    (unsigned long)u128_hi(nrr), (unsigned long)u128_lo(nrr),
                    (unsigned long)u128_hi(srr), (unsigned long)u128_lo(srr));
                if (++g_fail >= 20) return g_fail;
            } else {
                ++g_pass;
            }
        }

        // -- mask: (u64)dv & u64_const ----------------------------------------
        {
            constexpr std::uint64_t M52 = 0xFFFFFFFFFFFFFULL;
            std::uint64_t nmask = static_cast<std::uint64_t>(nc) & M52;
            std::uint64_t smask = static_cast<std::uint64_t>(sc) & M52;
            if (nmask != smask) {
                std::fprintf(stderr,
                    "[FAIL] mask_u64: iter=%d native=%016lx struct=%016lx\n",
                    iter, (unsigned long)nmask, (unsigned long)smask);
                if (++g_fail >= 20) return g_fail;
            } else {
                ++g_pass;
            }
        }
    }

    std::printf("[u128_compat_parity] pass=%d fail=%d\n", g_pass, g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() { return test_u128_compat_parity_run() == 0 ? 0 : 1; }
#endif

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif  // __SIZEOF_INT128__ guard around the native-comparison body
