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

// -- 4x64 assembly bridge for boundary-level FE52 optimizations ---------------
// Provides access to 4x64 ADCX/ADOX field_mul/sqr assembly from FE52 code.
// Used for pure sqr/mul chains (inverse, sqrt) where conversion at boundaries
// is negligible (~6ns) compared to per-op savings (~2ns x 269 ops = ~538ns).
// NOT used per-mul/sqr (GPU-style hybrid: same pointer, no conversion).
// Requires: SECP256K1_HAS_ASM + x86-64 (4x64 assembly always linked)
#if defined(SECP256K1_HAS_ASM) && (defined(__x86_64__) || defined(_M_X64))
  #define SECP256K1_HYBRID_4X64_ACTIVE 1
  #if defined(_WIN32)
    extern "C" __attribute__((sysv_abi)) void field_mul_full_asm(
        const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* result);
    extern "C" __attribute__((sysv_abi)) void field_sqr_full_asm(
        const std::uint64_t* a, std::uint64_t* result);
  #else
    extern "C" {
        void field_mul_full_asm(
            const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* result);
        void field_sqr_full_asm(
            const std::uint64_t* a, std::uint64_t* result);
    }
  #endif
#endif // SECP256K1_HAS_ASM && x86-64

// Force-inline attribute -- ensures zero call overhead for field ops.
// The compiler generates MULX assembly automatically with -mbmi2.
#if defined(__GNUC__) || defined(__clang__)
  #define SECP256K1_FE52_FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
  #define SECP256K1_FE52_FORCE_INLINE __forceinline
#else
  #define SECP256K1_FE52_FORCE_INLINE inline
#endif

// -- Hybrid 4x64 helper functions (placed after SECP256K1_FE52_FORCE_INLINE) --
#if defined(SECP256K1_HYBRID_4X64_ACTIVE)

  // Fused normalize_weak + 5x52->4x64 pack (single function, minimal overhead)
  SECP256K1_FE52_FORCE_INLINE
  void fe52_normalize_and_pack_4x64(const std::uint64_t* n, std::uint64_t* out) noexcept {
      constexpr std::uint64_t M = 0xFFFFFFFFFFFFFULL;   // 52-bit mask
      constexpr std::uint64_t M48v = 0xFFFFFFFFFFFFULL;  // 48-bit mask
      std::uint64_t t0 = n[0], t1 = n[1], t2 = n[2], t3 = n[3], t4 = n[4];
      // Pass 1: carry propagation
      t1 += (t0 >> 52); t0 &= M;
      t2 += (t1 >> 52); t1 &= M;
      t3 += (t2 >> 52); t2 &= M;
      t4 += (t3 >> 52); t3 &= M;
      // Overflow fold: x * 2^256 == x * 0x1000003D1 (mod p)
      std::uint64_t const x = t4 >> 48;
      t4 &= M48v;
      t0 += x * 0x1000003D1ULL;
      // Pass 2: re-propagate carry from fold
      t1 += (t0 >> 52); t0 &= M;
      t2 += (t1 >> 52); t1 &= M;
      t3 += (t2 >> 52); t2 &= M;
      t4 += (t3 >> 52); t3 &= M;
      // Pack to 4x64
      out[0] = t0 | (t1 << 52);
      out[1] = (t1 >> 12) | (t2 << 40);
      out[2] = (t2 >> 24) | (t3 << 28);
      out[3] = (t3 >> 36) | (t4 << 16);
  }

  // 4x64 -> 5x52 unpack (no normalization needed, output is magnitude 1)
  SECP256K1_FE52_FORCE_INLINE
  void fe64_unpack_to_fe52(const std::uint64_t* L, std::uint64_t* r) noexcept {
      constexpr std::uint64_t M = 0xFFFFFFFFFFFFFULL;
      r[0] =  L[0]                                      & M;
      r[1] = (L[0] >> 52) | ((L[1] & 0xFFFFFFFFFFULL)  << 12);
      r[2] = (L[1] >> 40) | ((L[2] & 0xFFFFFFFULL)     << 24);
      r[3] = (L[2] >> 28) | ((L[3] & 0xFFFFULL)        << 36);
      r[4] =  L[3] >> 16;
  }

#endif // SECP256K1_HYBRID_4X64_ACTIVE

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
void fe52_mul_inner(std::uint64_t* r,
                    const std::uint64_t* a,
                    const std::uint64_t* b) noexcept {
#if defined(SECP256K1_RISCV_FE52_V1)
    // RISC-V: Comba 5x52 multiply with integrated reduction in asm.
    // On U74 in-order core, explicit register scheduling + carry hiding
    // outperforms __int128 C++ (which Clang compiles to MUL/MULHU pairs
    // with suboptimal register allocation for 25+ multiplications).
    fe52_mul_inner_riscv64(r, a, b);
#elif 0 // INLINE_ADX disabled: asm barriers prevent ILP, __int128 is 6% faster
    // ------------------------------------------------------------------
    // x86-64 inline MULX + ADCX/ADOX dual carry chain path (OPT-IN)
    // NOTE: opt-in only. In benchmarks, the overhead of asm-block
    // optimization barriers outweighs the ADCX/ADOX parallel benefit.
    // The __int128 fallback lets the compiler schedule across column
    // boundaries, giving ~6% better throughput on Rocket Lake.
    // ------------------------------------------------------------------
    // ADCX uses CF flag, ADOX uses OF flag -- truly independent chains.
    // When both c and d accumulators accumulate products in the same
    // column, we interleave ADCX (d) and ADOX (c) to overlap execution.
    //
    // High-word carry invariant: sum of N products where each product
    // < 2^104 (52x52 bits) gives total < N*2^104. For N<=5:
    // 5*2^104 < 2^107 < 2^128. The 64-bit high word never overflows,
    // so carry-out from adcx/adox on the high part is always 0.
    // This keeps the continuous flag chain correct.
    //
    // Reduction multiplies between columns use __int128 C code (single
    // MULX+ADD+ADC pair, compiler-optimal for isolated operations).
    // ------------------------------------------------------------------
    using u128 = unsigned __int128;
    std::uint64_t d_lo = 0, d_hi = 0;
    std::uint64_t c_lo = 0, c_hi = 0;
    std::uint64_t t3, t4, tx, u0;
    std::uint64_t sl, sh;
    const std::uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];

    // -- Column 3 + reduced column 8 ---------------------------------
    // d = a0*b3 + a1*b2 + a2*b1 + a3*b0  (4 products, ADCX/CF)
    // c = a4*b4                            (1 product, ADOX/OF)
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "mov %[a0], %%rdx\n\t"
        "mulxq 24(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a4], %%rdx\n\t"
        "mulxq 32(%[bp]), %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "mov %[a1], %%rdx\n\t"
        "mulxq 16(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a2], %%rdx\n\t"
        "mulxq 8(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a3], %%rdx\n\t"
        "mulxq (%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [cl] "+&r"(c_lo), [ch] "+&r"(c_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [bp] "r"(b)
        : "rdx", "rcx", "cc"
    );
    // d += R52 * (uint64_t)c
    { u128 dv = ((u128)d_hi << 64) | d_lo;
      dv += (u128)R52 * c_lo;
      d_lo = (std::uint64_t)dv; d_hi = (std::uint64_t)(dv >> 64); }
    c_lo = c_hi; c_hi = 0;
    t3 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12); d_hi >>= 52;

    // -- Column 4 + column 8 carry -----------------------------------
    // d += a0*b4 + a1*b3 + a2*b2 + a3*b1 + a4*b0  (5 products, ADCX only)
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "mov %[a0], %%rdx\n\t"
        "mulxq 32(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a1], %%rdx\n\t"
        "mulxq 24(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a2], %%rdx\n\t"
        "mulxq 16(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a3], %%rdx\n\t"
        "mulxq 8(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a4], %%rdx\n\t"
        "mulxq (%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [bp] "r"(b)
        : "rdx", "rcx", "cc"
    );
    // d += (R52 << 12) * c_lo  (c_lo carries column 3's c_hi)
    { u128 dv = ((u128)d_hi << 64) | d_lo;
      dv += (u128)(R52 << 12) * c_lo;
      d_lo = (std::uint64_t)dv; d_hi = (std::uint64_t)(dv >> 64); }
    t4 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12); d_hi >>= 52;
    tx = (t4 >> 48); t4 &= (M52 >> 4);

    // -- Column 0 + reduced column 5 ---------------------------------
    // c = a0*b0                            (1 product, ADOX/OF)
    // d += a1*b4 + a2*b3 + a3*b2 + a4*b1  (4 products, ADCX/CF)
    c_lo = 0; c_hi = 0;
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "mov %[a0], %%rdx\n\t"
        "mulxq (%[bp]), %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "mov %[a1], %%rdx\n\t"
        "mulxq 32(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a2], %%rdx\n\t"
        "mulxq 24(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a3], %%rdx\n\t"
        "mulxq 16(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a4], %%rdx\n\t"
        "mulxq 8(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [cl] "+&r"(c_lo), [ch] "+&r"(c_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [bp] "r"(b)
        : "rdx", "rcx", "cc"
    );
    u0 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12); d_hi >>= 52;
    u0 = (u0 << 4) | tx;
    // c += u0 * (R52 >> 4)
    { u128 cv = ((u128)c_hi << 64) | c_lo;
      cv += (u128)u0 * (R52 >> 4);
      c_lo = (std::uint64_t)cv; c_hi = (std::uint64_t)(cv >> 64); }
    r[0] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12); c_hi >>= 52;

    // -- Column 1 + reduced column 6 ---------------------------------
    // c += a0*b1 + a1*b0              (2 products, ADOX/OF)
    // d += a2*b4 + a3*b3 + a4*b2      (3 products, ADCX/CF)
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "mov %[a0], %%rdx\n\t"
        "mulxq 8(%[bp]), %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "mov %[a2], %%rdx\n\t"
        "mulxq 32(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a1], %%rdx\n\t"
        "mulxq (%[bp]), %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "mov %[a3], %%rdx\n\t"
        "mulxq 24(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a4], %%rdx\n\t"
        "mulxq 16(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [cl] "+&r"(c_lo), [ch] "+&r"(c_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [bp] "r"(b)
        : "rdx", "rcx", "cc"
    );
    // c += ((uint64_t)d & M52) * R52
    { std::uint64_t d_masked = d_lo & M52;
      u128 cv = ((u128)c_hi << 64) | c_lo;
      cv += (u128)d_masked * R52;
      c_lo = (std::uint64_t)cv; c_hi = (std::uint64_t)(cv >> 64); }
    d_lo = (d_lo >> 52) | (d_hi << 12); d_hi >>= 52;
    r[1] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12); c_hi >>= 52;

    // -- Column 2 + reduced column 7 ---------------------------------
    // c += a0*b2 + a1*b1 + a2*b0      (3 products, ADOX/OF)
    // d += a3*b4 + a4*b3              (2 products, ADCX/CF)
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "mov %[a0], %%rdx\n\t"
        "mulxq 16(%[bp]), %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "mov %[a3], %%rdx\n\t"
        "mulxq 32(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a1], %%rdx\n\t"
        "mulxq 8(%[bp]), %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "mov %[a4], %%rdx\n\t"
        "mulxq 24(%[bp]), %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a2], %%rdx\n\t"
        "mulxq (%[bp]), %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [cl] "+&r"(c_lo), [ch] "+&r"(c_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4),
          [bp] "r"(b)
        : "rdx", "rcx", "cc"
    );
    // c += R52 * (uint64_t)d
    { u128 cv = ((u128)c_hi << 64) | c_lo;
      cv += (u128)R52 * d_lo;
      c_lo = (std::uint64_t)cv; c_hi = (std::uint64_t)(cv >> 64); }
    d_lo = d_hi; d_hi = 0;   // d >>= 64
    r[2] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12); c_hi >>= 52;

    // -- Finalize columns 3 and 4 ------------------------------------
    { u128 cv = ((u128)c_hi << 64) | c_lo;
      cv += (u128)(R52 << 12) * d_lo;
      cv += t3;
      c_lo = (std::uint64_t)cv; c_hi = (std::uint64_t)(cv >> 64); }
    r[3] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12); c_hi >>= 52;
    c_lo += t4;
    r[4] = c_lo;
#else
    using u128 = unsigned __int128;
    u128 c = 0, d = 0;
    std::uint64_t t3 = 0, t4 = 0, tx = 0, u0 = 0;
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
#endif // ARM64_FE52 / RISCV_FE52 / x64_ADX / generic (mul)
}

// ===========================================================================
// Core Squaring Kernel (symmetry-optimized)
// ===========================================================================
//
// Uses a[i]*a[j] == a[j]*a[i] symmetry to halve cross-product count.
// Cross-products computed once and doubled via (a[i]*2) trick.

SECP256K1_FE52_FORCE_INLINE
void fe52_sqr_inner(std::uint64_t* r,
                    const std::uint64_t* a) noexcept {
#if defined(SECP256K1_RISCV_FE52_V1)
    // RISC-V: Symmetry-optimized squaring in asm.
    // Cross-products doubled via shift, halving multiplication count.
    fe52_sqr_inner_riscv64(r, a);
#elif 0 // INLINE_ADX disabled: asm barriers prevent ILP, __int128 is 6% faster
    // ------------------------------------------------------------------
    // x86-64 inline MULX + ADCX/ADOX squaring (OPT-IN) -- see mul note
    // ------------------------------------------------------------------
    // Cross-products doubled via LEA (flags-neutral) then accumulated
    // with ADCX/ADOX dual carry chains. Square terms use plain MULX.
    // Same high-word carry invariant as fe52_mul_inner (sum < 2^128).
    // ------------------------------------------------------------------
    using u128 = unsigned __int128;
    std::uint64_t d_lo = 0, d_hi = 0;
    std::uint64_t c_lo = 0, c_hi = 0;
    std::uint64_t t3, t4, tx, u0;
    std::uint64_t sl, sh;
    const std::uint64_t a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];

    // -- Column 3 + reduced column 8 ---------------------------------
    // d = (a0*2)*a3 + (a1*2)*a2   (2 cross-products, ADCX/CF)
    // c = a4*a4                    (1 square, ADOX/OF)
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "lea (%[a0], %[a0]), %%rdx\n\t"
        "mulxq %[a3], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a4], %%rdx\n\t"
        "mulxq %[a4], %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "lea (%[a1], %[a1]), %%rdx\n\t"
        "mulxq %[a2], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [cl] "+&r"(c_lo), [ch] "+&r"(c_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4)
        : "rdx", "rcx", "cc"
    );
    { u128 dv = ((u128)d_hi << 64) | d_lo;
      dv += (u128)R52 * c_lo;
      d_lo = (std::uint64_t)dv; d_hi = (std::uint64_t)(dv >> 64); }
    c_lo = c_hi; c_hi = 0;
    t3 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12); d_hi >>= 52;

    // -- Column 4 ----------------------------------------------------
    // d += (a0*2)*a4 + (a1*2)*a3 + a2*a2  (3 products, ADCX only)
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "lea (%[a0], %[a0]), %%rdx\n\t"
        "mulxq %[a4], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "lea (%[a1], %[a1]), %%rdx\n\t"
        "mulxq %[a3], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a2], %%rdx\n\t"
        "mulxq %[a2], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4)
        : "rdx", "rcx", "cc"
    );
    { u128 dv = ((u128)d_hi << 64) | d_lo;
      dv += (u128)(R52 << 12) * c_lo;
      d_lo = (std::uint64_t)dv; d_hi = (std::uint64_t)(dv >> 64); }
    t4 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12); d_hi >>= 52;
    tx = (t4 >> 48); t4 &= (M52 >> 4);

    // -- Column 0 + reduced column 5 ---------------------------------
    // c = a0*a0                      (1 square, ADOX/OF)
    // d += (a1*2)*a4 + (a2*2)*a3     (2 cross-products, ADCX/CF)
    c_lo = 0; c_hi = 0;
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "mov %[a0], %%rdx\n\t"
        "mulxq %[a0], %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "lea (%[a1], %[a1]), %%rdx\n\t"
        "mulxq %[a4], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "lea (%[a2], %[a2]), %%rdx\n\t"
        "mulxq %[a3], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [cl] "+&r"(c_lo), [ch] "+&r"(c_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4)
        : "rdx", "rcx", "cc"
    );
    u0 = d_lo & M52;
    d_lo = (d_lo >> 52) | (d_hi << 12); d_hi >>= 52;
    u0 = (u0 << 4) | tx;
    { u128 cv = ((u128)c_hi << 64) | c_lo;
      cv += (u128)u0 * (R52 >> 4);
      c_lo = (std::uint64_t)cv; c_hi = (std::uint64_t)(cv >> 64); }
    r[0] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12); c_hi >>= 52;

    // -- Column 1 + reduced column 6 ---------------------------------
    // c += (a0*2)*a1                  (1 cross-product, ADOX/OF)
    // d += (a2*2)*a4 + a3*a3          (2 products, ADCX/CF)
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "lea (%[a0], %[a0]), %%rdx\n\t"
        "mulxq %[a1], %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "lea (%[a2], %[a2]), %%rdx\n\t"
        "mulxq %[a4], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a3], %%rdx\n\t"
        "mulxq %[a3], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [cl] "+&r"(c_lo), [ch] "+&r"(c_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4)
        : "rdx", "rcx", "cc"
    );
    { std::uint64_t d_masked = d_lo & M52;
      u128 cv = ((u128)c_hi << 64) | c_lo;
      cv += (u128)d_masked * R52;
      c_lo = (std::uint64_t)cv; c_hi = (std::uint64_t)(cv >> 64); }
    d_lo = (d_lo >> 52) | (d_hi << 12); d_hi >>= 52;
    r[1] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12); c_hi >>= 52;

    // -- Column 2 + reduced column 7 ---------------------------------
    // c += (a0*2)*a2 + a1*a1          (2 products, ADOX/OF)
    // d += (a3*2)*a4                  (1 cross-product, ADCX/CF)
    __asm__ __volatile__(
        "xor %%ecx, %%ecx\n\t"
        "lea (%[a0], %[a0]), %%rdx\n\t"
        "mulxq %[a2], %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        "lea (%[a3], %[a3]), %%rdx\n\t"
        "mulxq %[a4], %[sl], %[sh]\n\t"
        "adcx %[sl], %[dl]\n\t"
        "adcx %[sh], %[dh]\n\t"
        "mov %[a1], %%rdx\n\t"
        "mulxq %[a1], %[sl], %[sh]\n\t"
        "adox %[sl], %[cl]\n\t"
        "adox %[sh], %[ch]\n\t"
        : [dl] "+&r"(d_lo), [dh] "+&r"(d_hi),
          [cl] "+&r"(c_lo), [ch] "+&r"(c_hi),
          [sl] "=&r"(sl), [sh] "=&r"(sh)
        : [a0] "r"(a0), [a1] "r"(a1), [a2] "r"(a2), [a3] "r"(a3), [a4] "r"(a4)
        : "rdx", "rcx", "cc"
    );
    { u128 cv = ((u128)c_hi << 64) | c_lo;
      cv += (u128)R52 * d_lo;
      c_lo = (std::uint64_t)cv; c_hi = (std::uint64_t)(cv >> 64); }
    d_lo = d_hi; d_hi = 0;
    r[2] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12); c_hi >>= 52;

    // -- Finalize columns 3 and 4 ------------------------------------
    { u128 cv = ((u128)c_hi << 64) | c_lo;
      cv += (u128)(R52 << 12) * d_lo;
      cv += t3;
      c_lo = (std::uint64_t)cv; c_hi = (std::uint64_t)(cv >> 64); }
    r[3] = c_lo & M52;
    c_lo = (c_lo >> 52) | (c_hi << 12); c_hi >>= 52;
    c_lo += t4;
    r[4] = c_lo;
#else
    using u128 = unsigned __int128;
    u128 c = 0, d = 0;
    std::uint64_t t3 = 0, t4 = 0, tx = 0, u0 = 0;
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
#endif // ARM64_FE52 / RISCV_FE52 / x64_ADX / generic (sqr)
}

// ===========================================================================
// Weak Normalization (inline for half() hot path)
// ===========================================================================

SECP256K1_FE52_FORCE_INLINE
void fe52_normalize_weak(std::uint64_t* r) noexcept {
    std::uint64_t t0 = r[0], t1 = r[1], t2 = r[2], t3 = r[3], t4 = r[4];
    // Pass 1: propagate carries bottom-to-top to get true t4 value.
    // Required because our negate convention (1*(m+1)*P, not 2*(m+1)*P)
    // allows lower-limb carries that propagate to t4.
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;
    // Fold t4 overflow: x * 2^256 == x * R (mod p)
    std::uint64_t const x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;
    // Pass 2: re-propagate carry from fold
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
    fe52_mul_inner(n, n, rhs.n);
}

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::square_inplace() noexcept {
    fe52_sqr_inner(n, n);
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

// -- Branchless conditional negate (magnitude 1) --------------------------
// sign_mask: 0 = keep original, -1 (0xFFFFFFFF) = negate.
// Uses XOR-select to avoid branches on unpredictable sign bits.
SECP256K1_FE52_FORCE_INLINE
void FieldElement52::conditional_negate_assign(std::int32_t sign_mask) noexcept {
    const std::uint64_t mask = static_cast<std::uint64_t>(static_cast<std::int64_t>(sign_mask));
    // Compute negated limbs (magnitude 1: 2*P - n)
    const std::uint64_t neg0 = 2ULL * P0 - n[0];
    const std::uint64_t neg1 = 2ULL * P1 - n[1];
    const std::uint64_t neg2 = 2ULL * P2 - n[2];
    const std::uint64_t neg3 = 2ULL * P3 - n[3];
    const std::uint64_t neg4 = 2ULL * P4 - n[4];
    // Branchless select: mask=0 → keep n[i]; mask=~0 → use neg[i]
    n[0] ^= (n[0] ^ neg0) & mask;
    n[1] ^= (n[1] ^ neg1) & mask;
    n[2] ^= (n[2] ^ neg2) & mask;
    n[3] ^= (n[3] ^ neg3) & mask;
    n[4] ^= (n[4] ^ neg4) & mask;
}

// -- Weak Normalization (member) ------------------------------------------

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::normalize_weak() noexcept {
    fe52_normalize_weak(n);
}

// -- Half (a/2 mod p) -- branchless ---------------------------------------
// libsecp-style: mask trick avoids carry propagation entirely.
// If odd, add p; then right-shift by 1.  The mask is (-(t0 & 1)) >> 12
// which produces a 52-bit all-ones mask (0xFFFFFFFFFFFFF) when odd, 0 when even.
// Since P1=P2=P3 = M52 = 0xFFFFFFFFFFFFF, and the mask has exactly 52 set bits,
// adding mask to P1..P3 limbs can never exceed 2*M52 < 2^53 (fits in 64 bits).
// No carry propagation needed!

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::half() const noexcept {
    const std::uint64_t* src = n;
    const std::uint64_t one = 1ULL;
    const std::uint64_t mask = (0ULL - (src[0] & one)) >> 12;  // 52-bit mask if odd

    // Conditionally add p (limb-wise, no carry propagation needed)
    std::uint64_t t0 = src[0] + (0xFFFFEFFFFFC2FULL & mask);
    std::uint64_t t1 = src[1] + mask;       // P1 = M52 = mask
    std::uint64_t t2 = src[2] + mask;       // P2 = M52 = mask
    std::uint64_t t3 = src[3] + mask;       // P3 = M52 = mask
    std::uint64_t t4 = src[4] + (mask >> 4); // P4 = 48-bit

    // Right shift by 1 (divide by 2)
    // MUST use + (not |): without carry propagation, t_i can exceed M52,
    // so bit 51 of (t_i >> 1) can be set, overlapping with (t_{i+1} & 1) << 51.
    // Addition correctly carries; OR would silently drop the carry.
    FieldElement52 r;
    r.n[0] = (t0 >> 1) + ((t1 & one) << 51);
    r.n[1] = (t1 >> 1) + ((t2 & one) << 51);
    r.n[2] = (t2 >> 1) + ((t3 & one) << 51);
    r.n[3] = (t3 >> 1) + ((t4 & one) << 51);
    r.n[4] = (t4 >> 1);
    return r;
}

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::half_assign() noexcept {
    const std::uint64_t one = 1ULL;
    const std::uint64_t mask = (0ULL - (n[0] & one)) >> 12;

    std::uint64_t t0 = n[0] + (0xFFFFEFFFFFC2FULL & mask);
    std::uint64_t t1 = n[1] + mask;
    std::uint64_t t2 = n[2] + mask;
    std::uint64_t t3 = n[3] + mask;
    std::uint64_t t4 = n[4] + (mask >> 4);

    // MUST use + (not |): see half() comment above.
    n[0] = (t0 >> 1) + ((t1 & one) << 51);
    n[1] = (t1 >> 1) + ((t2 & one) << 51);
    n[2] = (t2 >> 1) + ((t3 & one) << 51);
    n[3] = (t3 >> 1) + ((t4 & one) << 51);
    n[4] = (t4 >> 1);
}

// -- Multiply by small integer (no carry propagation) ---------------------
// Each limb is multiplied by a (scalar <= 32).
// Safe as long as magnitude * a * 2^52 < 2^64, i.e. magnitude * a < 4096.

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::mul_int_assign(std::uint32_t a) noexcept {
    n[0] *= a;
    n[1] *= a;
    n[2] *= a;
    n[3] *= a;
    n[4] *= a;
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

    const std::uint64_t overflow = u4 >> 48;
    u4 &= M48;

    const std::uint64_t mask = 0ULL - overflow;
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

// -- Variable-time Zero Check (full normalize) ----------------------------
// Uses fe52_normalize_inline (TWO overflow-reduction passes + conditional
// p-subtraction) then checks canonical zero.  The previous single-pass
// implementation could produce false negatives at magnitude >= 25
// (e.g. h = u2 + negate(23) in mixed-add) because one pass can leave
// the value in [p, 2p) -- neither raw-0 nor raw-p.
//
// Variable-time: safe for non-secret values (point coordinates in ECC).

SECP256K1_FE52_FORCE_INLINE
bool FieldElement52::normalizes_to_zero() const noexcept {
    std::uint64_t t[5] = {n[0], n[1], n[2], n[3], n[4]};
    fe52_normalize_inline(t);
    return (t[0] | t[1] | t[2] | t[3] | t[4]) == 0;
}

// -- Variable-time Zero Check with Early Exit ------------------------------
// Performs a single normalize_weak pass (carry + overflow reduction + carry),
// then checks for raw-zero and p.  Avoids the expensive conditional
// p-subtraction + branchless-select of fe52_normalize_inline.
//
// After one normalize_weak pass at any magnitude <= ~4000, the value is
// in [0, 2p).  The only representations of 0 mod p in [0, 2p) are
// raw-zero (all limbs 0) and p itself.
//
// In the ecmult hot loop, h == 0 occurs with probability ~2^-256,
// so the fast non-zero path fires in essentially 100% of calls.
// This replaces the old normalize_weak() + normalizes_to_zero() pair
// in jac52_add_mixed*, saving ~40 limb ops per mixed add.

SECP256K1_FE52_FORCE_INLINE
bool FieldElement52::normalizes_to_zero_var() const noexcept {
    using namespace fe52_constants;
    std::uint64_t t0 = n[0], t1 = n[1], t2 = n[2], t3 = n[3], t4 = n[4];

    // Pass 1: carry propagation
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Overflow reduction: fold (t4 >> 48) * R back into t0
    std::uint64_t x = t4 >> 48;
    t4 &= M48;
    t0 += x * 0x1000003D1ULL;

    // Pass 2: propagate injection carry
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;

    // Second overflow reduction (handles magnitude > ~25)
    x = t4 >> 48;
    t4 &= M48;
    if (SECP256K1_UNLIKELY(x != 0)) {
        t0 += x * 0x1000003D1ULL;
        t1 += (t0 >> 52); t0 &= M52;
        t2 += (t1 >> 52); t1 &= M52;
        t3 += (t2 >> 52); t2 &= M52;
        t4 += (t3 >> 52); t3 &= M52;
        t4 &= M48;
    }

    // Fast path: raw-zero check (fires ~100% of the time for non-zero h)
    if ((t0 | t1 | t2 | t3 | t4) == 0) return true;

    // Value is in [1, p].  Check if it equals p.
    // p = {P0, M52, M52, M52, M48}  where P0 = 0xFFFFEFFFFFC2F
    // Quick exit: if any of t1..t3 != M52, it cannot be p.
    if ((t1 & t2 & t3) != M52 || t4 != M48) return false;

    return t0 == P0;
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
    // ge(L, P): check L >= P lexicographically from high limb.
    // NOTE: Variable-time comparison -- acceptable because input bytes
    // are public data (from wire / serialized keys), not secret.
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
// Returns zero for zero input (consistent with noexcept contract + embedded).

SECP256K1_FE52_FORCE_INLINE
FieldElement52 FieldElement52::inverse_safegcd() const noexcept {
    if (SECP256K1_UNLIKELY(normalizes_to_zero())) {
        return FieldElement52::zero();
    }
    return from_fe(to_fe().inverse());
}

} // namespace secp256k1::fast

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif // __int128 guard

#undef SECP256K1_FE52_FORCE_INLINE

#endif // SECP256K1_FIELD_52_IMPL_HPP
