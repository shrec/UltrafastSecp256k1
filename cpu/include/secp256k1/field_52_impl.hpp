// ============================================================================
// 5x52 Field Element -- Inline Hot-Path Implementations
// ============================================================================
//
// All performance-critical 5x52 operations are FORCE-INLINED to eliminate
// function-call overhead in ECC point operations (the #1 bottleneck).
//
// On x86-64 with -march=native, Clang/GCC generate MULX (BMI2) assembly
// from the __int128 C code -- identical to hand-written assembly but with
// superior register allocation (no callee-save push/pop overhead).
//
// This matches the strategy of bitcoin-core/secp256k1, which uses
// SECP256K1_INLINE static in field_5x52_int128_impl.h.
//
// Impact: eliminates ~2-3ns per field-mul call -> cumulative ~30-50ns
// savings per point double/add (which has 7+ field mul/sqr calls).
//
// Adaptation from bitcoin-core/secp256k1 field_5x52_int128_impl.h
// (MIT license, Copyright (c) 2013-2024 Pieter Wuille and contributors)
// ============================================================================

#ifndef SECP256K1_FIELD_52_IMPL_HPP
#define SECP256K1_FIELD_52_IMPL_HPP
#pragma once

#include <cstdint>

// Guard: __int128 required for the 5x52 kernels
// __SIZEOF_INT128__ is the canonical check -- defined on 64-bit GCC/Clang,
// NOT on 32-bit (ESP32 Xtensa, Cortex-M, etc.) even though __GNUC__ is set.
#if defined(__SIZEOF_INT128__)

// Suppress GCC -Wpedantic for __int128 (universally supported on 64-bit GCC/Clang)
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

// -- ARM64 optimized field kernels ----------------------------------------
// With -mcpu=cortex-a76, Clang already knows the A76 pipeline and schedules
// MUL/UMULH interleaving optimally from __int128 C code. The hand-scheduled
// ARM64 v2 ASM (field_asm52_arm64_v2.cpp) is kept for reference but disabled:
// separate asm volatile blocks act as compiler barriers that PREVENT optimal
// scheduling, and cross-TU calls add function-call overhead.
//
// To re-enable: uncomment the #define below.
// #define SECP256K1_ARM64_FE52_V2 1
#if 0 && defined(__aarch64__) && defined(SECP256K1_HAS_ARM64_FE52_ASM)
namespace secp256k1::fast::arm64_v2 {
    void fe52_mul_arm64_v2(std::uint64_t* __restrict r,
                           const std::uint64_t* __restrict a,
                           const std::uint64_t* __restrict b) noexcept;
    void fe52_sqr_arm64_v2(std::uint64_t* __restrict r,
                           const std::uint64_t* __restrict a) noexcept;
} // namespace secp256k1::fast::arm64_v2
#endif

// -- RISC-V 64-bit optimized FE52 kernels ---------------------------------
// On SiFive U74 (in-order dual-issue), hand-scheduled MUL/MULHU assembly
// for 5x52 Comba multiply with integrated secp256k1 reduction outperforms
// __int128 C++ code because:
//   1) Explicit register allocation avoids spills (25+ MUL ops)
//   2) Carry chain scheduling hides MUL latency on in-order pipeline
//   3) Branchless reduction integrated without separate passes
//
// Enabled by default when SECP256K1_HAS_RISCV_FE52_ASM is set by CMake.
// To disable and fall back to __int128 C++: -DSECP256K1_RISCV_FE52_DISABLE=1
#if defined(__riscv) && (__riscv_xlen == 64) && defined(SECP256K1_HAS_RISCV_FE52_ASM) \
    && !defined(SECP256K1_RISCV_FE52_DISABLE)
  #define SECP256K1_RISCV_FE52_V1 1
extern "C" {
    void fe52_mul_inner_riscv64(std::uint64_t* r, const std::uint64_t* a, const std::uint64_t* b);
    void fe52_sqr_inner_riscv64(std::uint64_t* r, const std::uint64_t* a);
}
#endif

// Force-inline attribute -- ensures zero call overhead for field ops.
// The compiler generates MULX assembly automatically with -mbmi2.
#if defined(__GNUC__) || defined(__clang__)
  #define SECP256K1_FE52_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
  #define SECP256K1_FE52_FORCE_INLINE __forceinline
#else
  #define SECP256K1_FE52_FORCE_INLINE inline
#endif

namespace secp256k1::fast {

using namespace fe52_constants;

// ===========================================================================
// Core Multiplication Kernel
// ===========================================================================
//
// 5x52 field multiplication with inline secp256k1 reduction.
// p = 2^256 - 0x1000003D1, so 2^260 == R = 0x1000003D10 (mod p).
//
// Product columns 5-8 are reduced by multiplying by R (or R>>4, R<<12)
// and adding to columns 0-3. Columns processed out of order (3,4,0,1,2)
// to keep 128-bit accumulators from overflowing.
//
// With -mbmi2 -O3: compiles to MULX + ADD/ADC chains (verified).
// With always_inline: zero function-call overhead.

SECP256K1_FE52_FORCE_INLINE
void fe52_mul_inner(std::uint64_t* __restrict__ r,
                    const std::uint64_t* __restrict__ a,
                    const std::uint64_t* __restrict__ b) noexcept {
#if defined(SECP256K1_ARM64_FE52_V2)
    // ARM64: hand-scheduled MUL/UMULH interleaving for Cortex-A76 class cores
    arm64_v2::fe52_mul_arm64_v2(r, a, b);
#elif defined(SECP256K1_RISCV_FE52_V1)
    // RISC-V: Comba 5x52 multiply with integrated reduction in asm.
    // On U74 in-order core, explicit register scheduling + carry hiding
    // outperforms __int128 C++ (which Clang compiles to MUL/MULHU pairs
    // with suboptimal register allocation for 25+ multiplications).
    fe52_mul_inner_riscv64(r, a, b);
#else
    using u128 = unsigned __int128;
    u128 c, d;
    std::uint64_t t3, t4, tx, u0;
    const std::uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];

    // -- Column 3 + reduced column 8 ---------------------------------
    d  = (u128)a0 * b[3]
       + (u128)a1 * b[2]
       + (u128)a2 * b[1]
       + (u128)a3 * b[0];
    c  = (u128)a4 * b[4];
    d += (u128)R52 * (std::uint64_t)c;
    c >>= 64;
    t3 = (std::uint64_t)d & M52;
    d >>= 52;

    // -- Column 4 + column 8 carry -----------------------------------
    d += (u128)a0 * b[4]
       + (u128)a1 * b[3]
       + (u128)a2 * b[2]
       + (u128)a3 * b[1]
       + (u128)a4 * b[0];
    d += (u128)(R52 << 12) * (std::uint64_t)c;
    t4 = (std::uint64_t)d & M52;
    d >>= 52;
    tx = (t4 >> 48); t4 &= (M52 >> 4);

    // -- Column 0 + reduced column 5 ---------------------------------
    c  = (u128)a0 * b[0];
    d += (u128)a1 * b[4]
       + (u128)a2 * b[3]
       + (u128)a3 * b[2]
       + (u128)a4 * b[1];
    u0 = (std::uint64_t)d & M52;
    d >>= 52;
    u0 = (u0 << 4) | tx;
    c += (u128)u0 * (R52 >> 4);
    r[0] = (std::uint64_t)c & M52;
    c >>= 52;

    // -- Column 1 + reduced column 6 ---------------------------------
    c += (u128)a0 * b[1]
       + (u128)a1 * b[0];
    d += (u128)a2 * b[4]
       + (u128)a3 * b[3]
       + (u128)a4 * b[2];
    c += (u128)((std::uint64_t)d & M52) * R52;
    d >>= 52;
    r[1] = (std::uint64_t)c & M52;
    c >>= 52;

    // -- Column 2 + reduced column 7 ---------------------------------
    c += (u128)a0 * b[2]
       + (u128)a1 * b[1]
       + (u128)a2 * b[0];
    d += (u128)a3 * b[4]
       + (u128)a4 * b[3];
    c += (u128)R52 * (std::uint64_t)d;
    d >>= 64;
    r[2] = (std::uint64_t)c & M52;
    c >>= 52;

    // -- Finalize columns 3 and 4 ------------------------------------
    c += (u128)(R52 << 12) * (std::uint64_t)d;
    c += t3;
    r[3] = (std::uint64_t)c & M52;
    c >>= 52;
    c += t4;
    r[4] = (std::uint64_t)c;
#endif // ARM64_FE52 / RISCV_FE52 / generic (mul)
}

// ===========================================================================
// Core Squaring Kernel (symmetry-optimized)
// ===========================================================================
//
// Uses a[i]*a[j] == a[j]*a[i] symmetry to halve cross-product count.
// Cross-products computed once and doubled via (a[i]*2) trick.

SECP256K1_FE52_FORCE_INLINE
void fe52_sqr_inner(std::uint64_t* __restrict__ r,
                    const std::uint64_t* __restrict__ a) noexcept {
#if defined(SECP256K1_ARM64_FE52_V2)
    arm64_v2::fe52_sqr_arm64_v2(r, a);
#elif defined(SECP256K1_RISCV_FE52_V1)
    // RISC-V: Symmetry-optimized squaring in asm.
    // Cross-products doubled via shift, halving multiplication count.
    fe52_sqr_inner_riscv64(r, a);
#else
    using u128 = unsigned __int128;
    u128 c, d;
    std::uint64_t t3, t4, tx, u0;
    const std::uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];

    // -- Column 3 + reduced column 8 ---------------------------------
    d  = (u128)(a0 * 2) * a3
       + (u128)(a1 * 2) * a2;
    c  = (u128)a4 * a4;
    d += (u128)R52 * (std::uint64_t)c;
    c >>= 64;
    t3 = (std::uint64_t)d & M52;
    d >>= 52;

    // -- Column 4 ----------------------------------------------------
    d += (u128)(a0 * 2) * a4
       + (u128)(a1 * 2) * a3
       + (u128)a2 * a2;
    d += (u128)(R52 << 12) * (std::uint64_t)c;
    t4 = (std::uint64_t)d & M52;
    d >>= 52;
    tx = (t4 >> 48); t4 &= (M52 >> 4);

    // -- Column 0 + reduced column 5 ---------------------------------
    c  = (u128)a0 * a0;
    d += (u128)(a1 * 2) * a4
       + (u128)(a2 * 2) * a3;
    u0 = (std::uint64_t)d & M52;
    d >>= 52;
    u0 = (u0 << 4) | tx;
    c += (u128)u0 * (R52 >> 4);
    r[0] = (std::uint64_t)c & M52;
    c >>= 52;

    // -- Column 1 + reduced column 6 ---------------------------------
    c += (u128)(a0 * 2) * a1;
    d += (u128)(a2 * 2) * a4
       + (u128)a3 * a3;
    c += (u128)((std::uint64_t)d & M52) * R52;
    d >>= 52;
    r[1] = (std::uint64_t)c & M52;
    c >>= 52;

    // -- Column 2 + reduced column 7 ---------------------------------
    c += (u128)(a0 * 2) * a2
       + (u128)a1 * a1;
    d += (u128)(a3 * 2) * a4;
    c += (u128)R52 * (std::uint64_t)d;
    d >>= 64;
    r[2] = (std::uint64_t)c & M52;
    c >>= 52;

    // -- Finalize columns 3 and 4 ------------------------------------
    c += (u128)(R52 << 12) * (std::uint64_t)d;
    c += t3;
    r[3] = (std::uint64_t)c & M52;
    c >>= 52;
    c += t4;
    r[4] = (std::uint64_t)c;
#endif // ARM64_FE52 / RISCV_FE52 / generic (sqr)
}

// ===========================================================================
// Weak Normalization (inline for half() hot path)
// ===========================================================================

SECP256K1_FE52_FORCE_INLINE
void fe52_normalize_weak(std::uint64_t* r) noexcept {
    std::uint64_t t0 = r[0], t1 = r[1], t2 = r[2], t3 = r[3], t4 = r[4];
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;
    std::uint64_t x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;
    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3; r[4] = t4;
}

// ===========================================================================
// FieldElement52 Method Implementations (all force-inlined)
// ===========================================================================

// -- Multiplication -------------------------------------------------------

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::operator*(const FieldElement52& rhs) const noexcept {
    FieldElement52 r;
    fe52_mul_inner(r.n, n, rhs.n);
    return r;
}

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::square() const noexcept {
    FieldElement52 r;
    fe52_sqr_inner(r.n, n);
    return r;
}

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::mul_assign(const FieldElement52& rhs) noexcept {
    std::uint64_t tmp[5];
    fe52_mul_inner(tmp, n, rhs.n);
    n[0] = tmp[0]; n[1] = tmp[1]; n[2] = tmp[2]; n[3] = tmp[3]; n[4] = tmp[4];
}

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::square_inplace() noexcept {
    std::uint64_t tmp[5];
    fe52_sqr_inner(tmp, n);
    n[0] = tmp[0]; n[1] = tmp[1]; n[2] = tmp[2]; n[3] = tmp[3]; n[4] = tmp[4];
}

// -- Lazy Addition (NO carry propagation!) --------------------------------

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::operator+(const FieldElement52& rhs) const noexcept {
    FieldElement52 r;
    r.n[0] = n[0] + rhs.n[0];
    r.n[1] = n[1] + rhs.n[1];
    r.n[2] = n[2] + rhs.n[2];
    r.n[3] = n[3] + rhs.n[3];
    r.n[4] = n[4] + rhs.n[4];
    return r;
}

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::add_assign(const FieldElement52& rhs) noexcept {
    n[0] += rhs.n[0];
    n[1] += rhs.n[1];
    n[2] += rhs.n[2];
    n[3] += rhs.n[3];
    n[4] += rhs.n[4];
}

// -- Negate: (M+1)*p - a -------------------------------------------------

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::negate(unsigned magnitude) const noexcept {
    FieldElement52 r = *this;
    r.negate_assign(magnitude);
    return r;
}

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::negate_assign(unsigned magnitude) noexcept {
    const std::uint64_t m1 = static_cast<std::uint64_t>(magnitude) + 1ULL;
    n[0] = m1 * P0 - n[0];
    n[1] = m1 * P1 - n[1];
    n[2] = m1 * P2 - n[2];
    n[3] = m1 * P3 - n[3];
    n[4] = m1 * P4 - n[4];
}

// -- Weak Normalization (member) ------------------------------------------

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::normalize_weak() noexcept {
    fe52_normalize_weak(n);
}

// -- Fast Variable-time Zero Check ----------------------------------------
// Checks if value reduces to zero mod p WITHOUT full normalization.
// Variable-time: safe for non-secret values (point coordinates in ECC).

SECP256K1_FE52_FORCE_INLINE
bool FieldElement52::normalizes_to_zero() const noexcept {
    std::uint64_t t0 = n[0], t1 = n[1], t2 = n[2], t3 = n[3], t4 = n[4];

    // Carry propagation
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Overflow reduction
    std::uint64_t x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;

    // Second carry propagation
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Check zero or equal to p
    std::uint64_t z0 = t0 | t1 | t2 | t3 | t4;
    std::uint64_t zp = (t0 ^ P0) | (t1 ^ P1) | (t2 ^ P2) | (t3 ^ P3) | (t4 ^ P4);

    return (z0 == 0) | (zp == 0);
}

// -- Half (a/2 mod p) -- branchless ---------------------------------------

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::half() const noexcept {
    // Note: no normalize_weak needed -- parity of n[0] is the parity of the
    // value regardless of carry overflow, and conditional p-add + carry + shift
    // are safe for magnitudes up to ~40000 (limbs < 2^58).
    const std::uint64_t* src = n;

    // mask = 0 if even, all-ones if odd
    std::uint64_t mask = -(src[0] & 1ULL);

    // Conditionally add p
    std::uint64_t t0 = src[0] + (P0 & mask);
    std::uint64_t t1 = src[1] + (P1 & mask);
    std::uint64_t t2 = src[2] + (P2 & mask);
    std::uint64_t t3 = src[3] + (P3 & mask);
    std::uint64_t t4 = src[4] + (P4 & mask);

    // Carry propagation
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Right shift by 1 (divide by 2)
    FieldElement52 r;
    r.n[0] = (t0 >> 1) | ((t1 & 1ULL) << 51);
    r.n[1] = (t1 >> 1) | ((t2 & 1ULL) << 51);
    r.n[2] = (t2 >> 1) | ((t3 & 1ULL) << 51);
    r.n[3] = (t3 >> 1) | ((t4 & 1ULL) << 51);
    r.n[4] = (t4 >> 1);

    return r;
}

// -- Full Normalization: canonical result in [0, p) ----------------------

SECP256K1_FE52_FORCE_INLINE
static void fe52_normalize_inline(std::uint64_t* r) noexcept {
    std::uint64_t t0 = r[0], t1 = r[1], t2 = r[2], t3 = r[3], t4 = r[4];

    // First pass: carry propagation + overflow reduction
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    std::uint64_t x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;

    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Second overflow reduction
    x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Branchless conditional subtraction of p if t >= p
    std::uint64_t u0 = t0 + 0x1000003D1ULL;
    std::uint64_t u1 = t1 + (u0 >> 52); u0 &= M52;
    std::uint64_t u2 = t2 + (u1 >> 52); u1 &= M52;
    std::uint64_t u3 = t3 + (u2 >> 52); u2 &= M52;
    std::uint64_t u4 = t4 + (u3 >> 52); u3 &= M52;

    std::uint64_t overflow = u4 >> 48;
    u4 &= M48;

    std::uint64_t mask = -overflow;
    r[0] = (u0 & mask) | (t0 & ~mask);
    r[1] = (u1 & mask) | (t1 & ~mask);
    r[2] = (u2 & mask) | (t2 & ~mask);
    r[3] = (u3 & mask) | (t3 & ~mask);
    r[4] = (u4 & mask) | (t4 & ~mask);
}

// -- Inline Normalization Method -----------------------------------------

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::normalize() noexcept {
    fe52_normalize_inline(n);
}

// -- Conversion: 4x64 -> 5x52 (inline) -----------------------------------

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::from_fe(const FieldElement& fe) noexcept {
    const auto& L = fe.limbs();
    FieldElement52 r;
    r.n[0] =  L[0]                           & M52;
    r.n[1] = (L[0] >> 52) | ((L[1] & 0xFFFFFFFFFFULL) << 12);
    r.n[2] = (L[1] >> 40) | ((L[2] & 0xFFFFFFFULL)    << 24);
    r.n[3] = (L[2] >> 28) | ((L[3] & 0xFFFFULL)       << 36);
    r.n[4] =  L[3] >> 16;
    return r;
}

// -- Conversion: 5x52 -> 4x64 (inline, includes full normalize) ----------

SECP256K1_FE52_FORCE_INLINE
FieldElement FieldElement52::to_fe() const noexcept {
    FieldElement52 tmp = *this;
    fe52_normalize_inline(tmp.n);

    FieldElement::limbs_type L;
    L[0] =  tmp.n[0]        | (tmp.n[1] << 52);
    L[1] = (tmp.n[1] >> 12) | (tmp.n[2] << 40);
    L[2] = (tmp.n[2] >> 24) | (tmp.n[3] << 28);
    L[3] = (tmp.n[3] >> 36) | (tmp.n[4] << 16);
    return FieldElement::from_limbs_raw(L);  // already canonical -- skip redundant normalize
}

// -- Direct 4x64 limbs -> 5x52 (no FieldElement construction) -------------
// Same bit-slicing as from_fe but takes raw uint64_t[4] pointer.
// Avoids FieldElement copy + normalization when caller knows value < p.

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::from_4x64_limbs(const std::uint64_t* L) noexcept {
    FieldElement52 r;
    r.n[0] =  L[0]                           & M52;
    r.n[1] = (L[0] >> 52) | ((L[1] & 0xFFFFFFFFFFULL) << 12);
    r.n[2] = (L[1] >> 40) | ((L[2] & 0xFFFFFFFULL)    << 24);
    r.n[3] = (L[2] >> 28) | ((L[3] & 0xFFFFULL)       << 36);
    r.n[4] =  L[3] >> 16;
    return r;
}

// -- Direct bytes (big-endian) -> 5x52 conversion ------------------------
// Combines FieldElement::from_bytes + from_fe into a single step.

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::from_bytes(const std::uint8_t* bytes) noexcept {
    // Read 4 uint64_t limbs from big-endian bytes (same layout as FieldElement::from_bytes)
    std::uint64_t L[4];
    for (int i = 0; i < 4; ++i) {
        std::uint64_t limb = 0;
        for (int j = 0; j < 8; ++j) {
            limb = (limb << 8) | static_cast<std::uint64_t>(bytes[i * 8 + j]);
        }
        L[3 - i] = limb;
    }
    // Reduce mod p if value >= p.
    // p = {0xFFFFFFFEFFFFFC2F, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF}
    static constexpr std::uint64_t P[4] = {
        0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
    };
    // ge(L, P): check L >= P lexicographically from high limb
    bool ge_p = true;
    for (int i = 3; i >= 0; --i) {
        if (L[i] < P[i]) { ge_p = false; break; }
        if (L[i] > P[i]) { break; }
    }
    if (ge_p) {
        // L -= P (with borrow)
        unsigned __int128 acc = static_cast<unsigned __int128>(L[0]) + (~P[0]) + 1;
        L[0] = static_cast<std::uint64_t>(acc);
        acc = static_cast<unsigned __int128>(L[1]) + (~P[1]) + (acc >> 64);
        L[1] = static_cast<std::uint64_t>(acc);
        acc = static_cast<unsigned __int128>(L[2]) + (~P[2]) + (acc >> 64);
        L[2] = static_cast<std::uint64_t>(acc);
        L[3] = L[3] + (~P[3]) + static_cast<std::uint64_t>(acc >> 64);
    }
    return from_4x64_limbs(L);
}

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::from_bytes(const std::array<std::uint8_t, 32>& bytes) noexcept {
    return from_bytes(bytes.data());
}

// -- Inverse via safegcd (4x64 round-trip, single wrapper) ---------------
// Replaces the common pattern: FieldElement52::from_fe(x.to_fe().inverse())

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::inverse_safegcd() const noexcept {
    return from_fe(to_fe().inverse());
}

} // namespace secp256k1::fast

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif // __int128 guard

#undef SECP256K1_FE52_FORCE_INLINE

#endif // SECP256K1_FIELD_52_IMPL_HPP
