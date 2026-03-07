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
#elif 0 // MONOLITHIC_MUL_ASM: Disabled -- with -march=native, Clang __int128
      // already emits optimal MULX+ADCX/ADOX code. The inline asm prevents
      // cross-operation scheduling and increases register pressure (~30% slower
      // point_add when enabled). Kept for reference/non-native builds.
    // ========================================================================
    // Monolithic x86-64 MULX + ADCX/ADOX field multiply (single asm block)
    // ========================================================================
    // Single asm block = ZERO optimization barriers between columns.
    // ADCX/ADOX dual carry chains enable ILP in columns 1 & 2.
    //
    // Register layout — ALL clobbered registers are volatile on Win64:
    //   [a0]-[a4] = a limbs (read-only inputs, compiler picks registers)
    //   [bp]      = b pointer (read-only input)
    //   r8        = d_lo accumulator (volatile, clobbered)
    //   r9        = d_hi accumulator (volatile, clobbered)
    //   r10       = c_lo accumulator (volatile, clobbered)
    //   r11       = c_hi accumulator (volatile, clobbered)
    //   rdx       = MULX source (volatile, clobbered)
    //   rax       = MULX low / scratch (volatile, clobbered)
    //   rcx       = MULX high / scratch (volatile, clobbered)
    //   rbp,rsi,rbx,rdi,r12-r15 = FREE for compiler (not touched by asm)
    // ========================================================================
    std::uint64_t out0, out1, out2, out3 = 0, out4 = 0;
    const std::uint64_t a0_v = a[0], a1_v = a[1], a2_v = a[2];
    const std::uint64_t a3_v = a[3], a4_v = a[4];
    __asm__ __volatile__ (
        // ---- Column 3 + reduced column 8 ----
        "xorl %%r8d, %%r8d\n\t"                // d_lo=0, clears CF+OF
        "xorl %%r9d, %%r9d\n\t"                // d_hi=0

        "movq %[a0], %%rdx\n\t"
        "mulxq 24(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a1], %%rdx\n\t"
        "mulxq 16(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a2], %%rdx\n\t"
        "mulxq 8(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a3], %%rdx\n\t"
        "mulxq (%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"

        // c = a4*b4
        "movq %[a4], %%rdx\n\t"
        "mulxq 32(%[bp]), %%r10, %%r11\n\t"

        // d += R52 * c_lo
        "movabsq $0x1000003D10, %%rdx\n\t"
        "mulxq %%r10, %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        // c >>= 64
        "movq %%r11, %%r10\n\t"

        // t3 = d & M52 → store to out3; d >>= 52
        "movq %%r8, %%rax\n\t"
        "movq $0xFFFFFFFFFFFFF, %%rcx\n\t"
        "andq %%rcx, %%rax\n\t"
        "movq %%rax, %[o3]\n\t"
        "shrdq $52, %%r9, %%r8\n\t"
        "shrq $52, %%r9\n\t"

        // ---- Column 4 + column 8 carry ----
        "xorl %%eax, %%eax\n\t"
        "movq %[a0], %%rdx\n\t"
        "mulxq 32(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a1], %%rdx\n\t"
        "mulxq 24(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a2], %%rdx\n\t"
        "mulxq 16(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a3], %%rdx\n\t"
        "mulxq 8(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a4], %%rdx\n\t"
        "mulxq (%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"

        // d += (R52 << 12) * c_lo
        "movabsq $0x1000003D10000, %%rdx\n\t"
        "mulxq %%r10, %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"

        // t4_full = d & M52 → r10; d >>= 52
        "movq %%r8, %%r10\n\t"
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "andq %%rax, %%r10\n\t"
        "shrdq $52, %%r9, %%r8\n\t"
        "shrq $52, %%r9\n\t"

        // ---- Column 0 + reduced column 5 ----
        "xorl %%eax, %%eax\n\t"
        "movq %[a1], %%rdx\n\t"
        "mulxq 32(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a2], %%rdx\n\t"
        "mulxq 24(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a3], %%rdx\n\t"
        "mulxq 16(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a4], %%rdx\n\t"
        "mulxq 8(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"

        // u0 = ((d & M52) << 4) | (t4_full >> 48)
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r8, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "shrdq $52, %%r9, %%r8\n\t"
        "shrq $52, %%r9\n\t"
        "shlq $4, %%rcx\n\t"
        "movq %%r10, %%rax\n\t"
        "shrq $48, %%rax\n\t"
        "orq %%rax, %%rcx\n\t"
        // t4 = t4_full & M48 → store to out4
        "movq $0xFFFFFFFFFFFF, %%rax\n\t"
        "andq %%rax, %%r10\n\t"
        "movq %%r10, %[o4]\n\t"

        // c = a0*b0 + u0 * (R52 >> 4)
        "movq %[a0], %%rdx\n\t"
        "mulxq (%[bp]), %%r10, %%r11\n\t"
        "movabsq $0x1000003D1, %%rdx\n\t"
        "mulxq %%rcx, %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        // r[0] = c & M52; c >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r10, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "movq %%rcx, %[o0]\n\t"
        "shrdq $52, %%r11, %%r10\n\t"
        "shrq $52, %%r11\n\t"

        // ---- Column 1 + reduced column 6 (dual chain) ----
        "xorl %%eax, %%eax\n\t"
        "movq %[a0], %%rdx\n\t"
        "mulxq 8(%[bp]), %%rax, %%rcx\n\t"
        "adoxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "movq %[a2], %%rdx\n\t"
        "mulxq 32(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a1], %%rdx\n\t"
        "mulxq (%[bp]), %%rax, %%rcx\n\t"
        "adoxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "movq %[a3], %%rdx\n\t"
        "mulxq 24(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a4], %%rdx\n\t"
        "mulxq 16(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"

        // c += (d & M52) * R52; d >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r8, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "shrdq $52, %%r9, %%r8\n\t"
        "shrq $52, %%r9\n\t"
        "movabsq $0x1000003D10, %%rdx\n\t"
        "mulxq %%rcx, %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        // r[1] = c & M52; c >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r10, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "movq %%rcx, %[o1]\n\t"
        "shrdq $52, %%r11, %%r10\n\t"
        "shrq $52, %%r11\n\t"

        // ---- Column 2 + reduced column 7 (dual chain) ----
        "xorl %%eax, %%eax\n\t"
        "movq %[a0], %%rdx\n\t"
        "mulxq 16(%[bp]), %%rax, %%rcx\n\t"
        "adoxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "movq %[a3], %%rdx\n\t"
        "mulxq 32(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a1], %%rdx\n\t"
        "mulxq 8(%[bp]), %%rax, %%rcx\n\t"
        "adoxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "movq %[a4], %%rdx\n\t"
        "mulxq 24(%[bp]), %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a2], %%rdx\n\t"
        "mulxq (%[bp]), %%rax, %%rcx\n\t"
        "adoxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"

        // c += R52 * d_lo; d = d_hi
        "movabsq $0x1000003D10, %%rdx\n\t"
        "mulxq %%r8, %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"
        "movq %%r9, %%r8\n\t"
        "xorl %%r9d, %%r9d\n\t"

        // r[2] = c & M52; c >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r10, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "movq %%rcx, %[o2]\n\t"
        "shrdq $52, %%r11, %%r10\n\t"
        "shrq $52, %%r11\n\t"

        // ---- Finalize columns 3 and 4 ----
        "movabsq $0x1000003D10000, %%rdx\n\t"
        "mulxq %%r8, %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"
        "addq %[o3], %%r10\n\t"
        "adcq $0, %%r11\n\t"

        // r[3] = c & M52; c >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r10, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "movq %%rcx, %[o3]\n\t"
        "shrdq $52, %%r11, %%r10\n\t"

        // r[4] = c + t4
        "addq %[o4], %%r10\n\t"
        "movq %%r10, %[o4]\n\t"

        : [o0] "=m"(out0), [o1] "=m"(out1), [o2] "=m"(out2),
          [o3] "+m"(out3), [o4] "+m"(out4)
        : [a0] "r"(a0_v), [a1] "r"(a1_v), [a2] "r"(a2_v),
          [a3] "r"(a3_v), [a4] "r"(a4_v), [bp] "r"(b)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "cc", "memory"
    );
    r[0] = out0; r[1] = out1; r[2] = out2; r[3] = out3; r[4] = out4;
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
#elif 0 // MONOLITHIC_SQR_ASM: Disabled -- same rationale as mul ASM above.
    // ========================================================================
    // Monolithic x86-64 MULX + ADCX/ADOX field squaring (single asm block)
    // ========================================================================
    // Cross-products doubled via LEA (flags-neutral), only 15 MULXes vs 25.
    // ALL clobbered registers are volatile on Win64 (r8-r11, rax, rcx, rdx).
    // ========================================================================
    std::uint64_t out0, out1, out2, out3 = 0, out4 = 0;
    const std::uint64_t a0_v = a[0], a1_v = a[1], a2_v = a[2];
    const std::uint64_t a3_v = a[3], a4_v = a[4];
    __asm__ __volatile__ (
        // ---- Column 3 + reduced column 8 ----
        // d = 2*a0*a3 + 2*a1*a2; c = a4^2
        "xorl %%r8d, %%r8d\n\t"
        "xorl %%r9d, %%r9d\n\t"

        "leaq (%[a0], %[a0]), %%rdx\n\t"
        "mulxq %[a3], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "leaq (%[a1], %[a1]), %%rdx\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"

        // c = a4*a4
        "movq %[a4], %%rdx\n\t"
        "mulxq %[a4], %%r10, %%r11\n\t"

        // d += R52 * c_lo
        "movabsq $0x1000003D10, %%rdx\n\t"
        "mulxq %%r10, %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"
        // c >>= 64
        "movq %%r11, %%r10\n\t"

        // t3 = d & M52; d >>= 52
        "movq %%r8, %%rax\n\t"
        "movq $0xFFFFFFFFFFFFF, %%rcx\n\t"
        "andq %%rcx, %%rax\n\t"
        "movq %%rax, %[o3]\n\t"
        "shrdq $52, %%r9, %%r8\n\t"
        "shrq $52, %%r9\n\t"

        // ---- Column 4 ----
        // d += 2*a0*a4 + 2*a1*a3 + a2^2
        "xorl %%eax, %%eax\n\t"
        "leaq (%[a0], %[a0]), %%rdx\n\t"
        "mulxq %[a4], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "leaq (%[a1], %[a1]), %%rdx\n\t"
        "mulxq %[a3], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a2], %%rdx\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"

        // d += (R52 << 12) * c_lo
        "movabsq $0x1000003D10000, %%rdx\n\t"
        "mulxq %%r10, %%rax, %%rcx\n\t"
        "addq %%rax, %%r8\n\t"
        "adcq %%rcx, %%r9\n\t"

        // t4_full = d & M52; d >>= 52
        "movq %%r8, %%r10\n\t"
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "andq %%rax, %%r10\n\t"
        "shrdq $52, %%r9, %%r8\n\t"
        "shrq $52, %%r9\n\t"

        // ---- Column 0 + reduced column 5 ----
        // c = a0^2; d += 2*a1*a4 + 2*a2*a3
        "xorl %%eax, %%eax\n\t"
        "leaq (%[a1], %[a1]), %%rdx\n\t"
        "mulxq %[a4], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "leaq (%[a2], %[a2]), %%rdx\n\t"
        "mulxq %[a3], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"

        // u0 = ((d & M52) << 4) | (t4_full >> 48)
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r8, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "shrdq $52, %%r9, %%r8\n\t"
        "shrq $52, %%r9\n\t"
        "shlq $4, %%rcx\n\t"
        "movq %%r10, %%rax\n\t"
        "shrq $48, %%rax\n\t"
        "orq %%rax, %%rcx\n\t"
        // t4 = t4_full & M48
        "movq $0xFFFFFFFFFFFF, %%rax\n\t"
        "andq %%rax, %%r10\n\t"
        "movq %%r10, %[o4]\n\t"

        // c = a0*a0 + u0 * (R52 >> 4)
        "movq %[a0], %%rdx\n\t"
        "mulxq %[a0], %%r10, %%r11\n\t"
        "movabsq $0x1000003D1, %%rdx\n\t"
        "mulxq %%rcx, %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        // r[0] = c & M52; c >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r10, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "movq %%rcx, %[o0]\n\t"
        "shrdq $52, %%r11, %%r10\n\t"
        "shrq $52, %%r11\n\t"

        // ---- Column 1 + reduced column 6 (dual chain) ----
        // c += 2*a0*a1 (ADOX); d += 2*a2*a4 + a3^2 (ADCX)
        "xorl %%eax, %%eax\n\t"
        "leaq (%[a0], %[a0]), %%rdx\n\t"
        "mulxq %[a1], %%rax, %%rcx\n\t"
        "adoxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "leaq (%[a2], %[a2]), %%rdx\n\t"
        "mulxq %[a4], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a3], %%rdx\n\t"
        "mulxq %[a3], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"

        // c += (d & M52) * R52; d >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r8, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "shrdq $52, %%r9, %%r8\n\t"
        "shrq $52, %%r9\n\t"
        "movabsq $0x1000003D10, %%rdx\n\t"
        "mulxq %%rcx, %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"

        // r[1] = c & M52; c >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r10, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "movq %%rcx, %[o1]\n\t"
        "shrdq $52, %%r11, %%r10\n\t"
        "shrq $52, %%r11\n\t"

        // ---- Column 2 + reduced column 7 (dual chain) ----
        // c += 2*a0*a2 + a1^2 (ADOX); d += 2*a3*a4 (ADCX)
        "xorl %%eax, %%eax\n\t"
        "leaq (%[a0], %[a0]), %%rdx\n\t"
        "mulxq %[a2], %%rax, %%rcx\n\t"
        "adoxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"
        "leaq (%[a3], %[a3]), %%rdx\n\t"
        "mulxq %[a4], %%rax, %%rcx\n\t"
        "adcxq %%rax, %%r8\n\t"
        "adcxq %%rcx, %%r9\n\t"
        "movq %[a1], %%rdx\n\t"
        "mulxq %[a1], %%rax, %%rcx\n\t"
        "adoxq %%rax, %%r10\n\t"
        "adoxq %%rcx, %%r11\n\t"

        // c += R52 * d_lo; d = d_hi
        "movabsq $0x1000003D10, %%rdx\n\t"
        "mulxq %%r8, %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"
        "movq %%r9, %%r8\n\t"
        "xorl %%r9d, %%r9d\n\t"

        // r[2] = c & M52; c >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r10, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "movq %%rcx, %[o2]\n\t"
        "shrdq $52, %%r11, %%r10\n\t"
        "shrq $52, %%r11\n\t"

        // ---- Finalize columns 3 and 4 ----
        "movabsq $0x1000003D10000, %%rdx\n\t"
        "mulxq %%r8, %%rax, %%rcx\n\t"
        "addq %%rax, %%r10\n\t"
        "adcq %%rcx, %%r11\n\t"
        "addq %[o3], %%r10\n\t"
        "adcq $0, %%r11\n\t"

        // r[3] = c & M52; c >>= 52
        "movq $0xFFFFFFFFFFFFF, %%rax\n\t"
        "movq %%r10, %%rcx\n\t"
        "andq %%rax, %%rcx\n\t"
        "movq %%rcx, %[o3]\n\t"
        "shrdq $52, %%r11, %%r10\n\t"

        // r[4] = c + t4
        "addq %[o4], %%r10\n\t"
        "movq %%r10, %[o4]\n\t"

        : [o0] "=m"(out0), [o1] "=m"(out1), [o2] "=m"(out2),
          [o3] "+m"(out3), [o4] "+m"(out4)
        : [a0] "r"(a0_v), [a1] "r"(a1_v), [a2] "r"(a2_v),
          [a3] "r"(a3_v), [a4] "r"(a4_v)
        : "rax", "rcx", "rdx", "r8", "r9", "r10", "r11", "cc", "memory"
    );
    r[0] = out0; r[1] = out1; r[2] = out2; r[3] = out3; r[4] = out4;
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
    using namespace fe52_constants;
    FieldElement52 r;
    const std::uint64_t m1 = static_cast<std::uint64_t>(magnitude) + 1ULL;
    r.n[0] = m1 * P0 - n[0];
    r.n[1] = m1 * P1 - n[1];
    r.n[2] = m1 * P2 - n[2];
    r.n[3] = m1 * P3 - n[3];
    r.n[4] = m1 * P4 - n[4];
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
// Fold-first approach (matches libsecp256k1): fold t4 overflow BEFORE carry
// propagation so only 2 carry chains are needed instead of 3.

SECP256K1_FE52_FORCE_INLINE
static void fe52_normalize_inline(std::uint64_t* r) noexcept {
    std::uint64_t t0 = r[0], t1 = r[1], t2 = r[2], t3 = r[3], t4 = r[4];

    // Reduce t4 overflow first (before carry propagation).
    // This ensures at most a single carry from the first pass.
    std::uint64_t m = 0;
    std::uint64_t x = t4 >> 48; t4 &= M48;

    // Single carry propagation pass with m accumulation for >= p check
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52; m = t1;
    t3 += (t2 >> 52); t2 &= M52; m &= t2;
    t4 += (t3 >> 52); t3 &= M52; m &= t3;

    // At most a single bit of overflow at bit 48 of t4 (bit 256 of value).
    // Check if result >= p:
    //   bit 48 of t4 set (value >= 2^256), OR
    //   all limbs at max (t1&t2&t3 == M52, t4 == M48, t0 >= p's low 52 bits)
    x = (t4 >> 48) | ((t4 == M48) & (m == M52)
        & (t0 >= 0xFFFFEFFFFFC2FULL));

    // Conditional final reduction (always executed for constant-time)
    t0 += x * 0x1000003D1ULL;
    t1 += (t0 >> 52); t0 &= M52;
    t2 += (t1 >> 52); t1 &= M52;
    t3 += (t2 >> 52); t2 &= M52;
    t4 += (t3 >> 52); t3 &= M52;
    t4 &= M48;

    r[0] = t0; r[1] = t1; r[2] = t2; r[3] = t3; r[4] = t4;
}

// -- Inline Normalization Method -----------------------------------------

SECP256K1_FE52_FORCE_INLINE
void FieldElement52::normalize() noexcept {
    fe52_normalize_inline(n);
}

// -- Variable-time Zero Check (full normalize) ----------------------------
// Uses fe52_normalize_inline (fold-first carry + conditional p-subtraction)
// then checks canonical zero.  The previous single-pass implementation
// could produce false negatives at magnitude >= 25 (e.g. h = u2 +
// negate(23) in mixed-add) because one pass can leave the value in
// [p, 2p) -- neither raw-0 nor raw-p.
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
    std::uint64_t t0 = n[0], t4 = n[4];

    // Reduce t4 overflow into t0 first (at most one carry fold).
    // This ensures the first full carry pass has at most one carry
    // propagation step from the injected overflow.
    const std::uint64_t x = t4 >> 48;
    t0 += x * 0x1000003D1ULL;

    // z0 tracks "could be raw zero", z1 tracks "could be p".
    // If the low 52 bits of t0 are clearly non-zero AND don't match P0,
    // the full value is neither 0 nor p -- early exit without touching n[1..3].
    std::uint64_t z0 = t0 & M52;
    std::uint64_t z1 = z0 ^ 0x1000003D0ULL;

    // Fast return: catches ~100% of cases using only t0 and t4.
    if ((z0 != 0ULL) & (z1 != M52)) {
        return false;
    }

    // Slow path: full carry propagation for the remaining cases.
    std::uint64_t t1 = n[1], t2 = n[2], t3 = n[3];
    t4 &= M48;

    t1 += (t0 >> 52);
    t2 += (t1 >> 52); t1 &= M52; z0 |= t1; z1 &= t1;
    t3 += (t2 >> 52); t2 &= M52; z0 |= t2; z1 &= t2;
    t4 += (t3 >> 52); t3 &= M52; z0 |= t3; z1 &= t3;
                                  z0 |= t4; z1 &= t4 ^ 0xF000000000000ULL;

    return (z0 == 0) | (z1 == M52);
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

// Convenience serialization: FE52 -> bytes in one call
SECP256K1_FE52_FORCE_INLINE
void FieldElement52::to_bytes_into(std::uint8_t* out) const noexcept {
    // Direct 5x52 -> 32 big-endian bytes (skip intermediate 4x64 conversion).
    // Same approach as libsecp256k1's secp256k1_fe_impl_get_b32.
    FieldElement52 tmp = *this;
    fe52_normalize_inline(tmp.n);

    out[ 0] = static_cast<std::uint8_t>(tmp.n[4] >> 40);
    out[ 1] = static_cast<std::uint8_t>(tmp.n[4] >> 32);
    out[ 2] = static_cast<std::uint8_t>(tmp.n[4] >> 24);
    out[ 3] = static_cast<std::uint8_t>(tmp.n[4] >> 16);
    out[ 4] = static_cast<std::uint8_t>(tmp.n[4] >>  8);
    out[ 5] = static_cast<std::uint8_t>(tmp.n[4]      );
    out[ 6] = static_cast<std::uint8_t>(tmp.n[3] >> 44);
    out[ 7] = static_cast<std::uint8_t>(tmp.n[3] >> 36);
    out[ 8] = static_cast<std::uint8_t>(tmp.n[3] >> 28);
    out[ 9] = static_cast<std::uint8_t>(tmp.n[3] >> 20);
    out[10] = static_cast<std::uint8_t>(tmp.n[3] >> 12);
    out[11] = static_cast<std::uint8_t>(tmp.n[3] >>  4);
    out[12] = static_cast<std::uint8_t>(((tmp.n[2] >> 48) & 0xF) | ((tmp.n[3] & 0xF) << 4));
    out[13] = static_cast<std::uint8_t>(tmp.n[2] >> 40);
    out[14] = static_cast<std::uint8_t>(tmp.n[2] >> 32);
    out[15] = static_cast<std::uint8_t>(tmp.n[2] >> 24);
    out[16] = static_cast<std::uint8_t>(tmp.n[2] >> 16);
    out[17] = static_cast<std::uint8_t>(tmp.n[2] >>  8);
    out[18] = static_cast<std::uint8_t>(tmp.n[2]      );
    out[19] = static_cast<std::uint8_t>(tmp.n[1] >> 44);
    out[20] = static_cast<std::uint8_t>(tmp.n[1] >> 36);
    out[21] = static_cast<std::uint8_t>(tmp.n[1] >> 28);
    out[22] = static_cast<std::uint8_t>(tmp.n[1] >> 20);
    out[23] = static_cast<std::uint8_t>(tmp.n[1] >> 12);
    out[24] = static_cast<std::uint8_t>(tmp.n[1] >>  4);
    out[25] = static_cast<std::uint8_t>(((tmp.n[0] >> 48) & 0xF) | ((tmp.n[1] & 0xF) << 4));
    out[26] = static_cast<std::uint8_t>(tmp.n[0] >> 40);
    out[27] = static_cast<std::uint8_t>(tmp.n[0] >> 32);
    out[28] = static_cast<std::uint8_t>(tmp.n[0] >> 24);
    out[29] = static_cast<std::uint8_t>(tmp.n[0] >> 16);
    out[30] = static_cast<std::uint8_t>(tmp.n[0] >>  8);
    out[31] = static_cast<std::uint8_t>(tmp.n[0]      );
}

// Fast serialize for pre-normalized limbs: 5x52 -> 32 big-endian bytes.
// Skips fe52_normalize_inline (saves ~10 ns per call on pre-normalized inputs).
// Uses bit-slicing to 4x64 intermediary + byte-swap stores.
SECP256K1_FE52_FORCE_INLINE
void FieldElement52::store_b32_prenorm(std::uint8_t* out) const noexcept {
    // 5x52 -> 4x64 bit-slicing (same logic as to_fe, no normalize)
    std::uint64_t L0 =  n[0]        | (n[1] << 52);
    std::uint64_t L1 = (n[1] >> 12) | (n[2] << 40);
    std::uint64_t L2 = (n[2] >> 24) | (n[3] << 28);
    std::uint64_t L3 = (n[3] >> 36) | (n[4] << 16);

    // Big-endian store: 4 bswap + 4 unaligned writes (L3 = MSB)
#if defined(__GNUC__) || defined(__clang__)
    L3 = __builtin_bswap64(L3); L2 = __builtin_bswap64(L2);
    L1 = __builtin_bswap64(L1); L0 = __builtin_bswap64(L0);
#elif defined(_MSC_VER)
    L3 = _byteswap_uint64(L3); L2 = _byteswap_uint64(L2);
    L1 = _byteswap_uint64(L1); L0 = _byteswap_uint64(L0);
#else
    // Portable bswap fallback
    auto bswap64 = [](std::uint64_t v) -> std::uint64_t {
        v = ((v >> 8) & 0x00FF00FF00FF00FFULL) | ((v & 0x00FF00FF00FF00FFULL) << 8);
        v = ((v >> 16) & 0x0000FFFF0000FFFFULL) | ((v & 0x0000FFFF0000FFFFULL) << 16);
        return (v >> 32) | (v << 32);
    };
    L3 = bswap64(L3); L2 = bswap64(L2);
    L1 = bswap64(L1); L0 = bswap64(L0);
#endif
    std::memcpy(out,      &L3, 8);
    std::memcpy(out + 8,  &L2, 8);
    std::memcpy(out + 16, &L1, 8);
    std::memcpy(out + 24, &L0, 8);
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
    if (SECP256K1_UNLIKELY(normalizes_to_zero_var())) {
        return FieldElement52::zero();
    }
    // Direct 5x52 → signed62 → SafeGCD → signed62 → 5x52.
    // Bypasses the old FE52→to_fe()→inverse()→from_fe() chain
    // which had 4 intermediate format conversions (5x52↔4x64↔signed62).
    FieldElement52 tmp = *this;
    fe52_normalize_inline(tmp.n);
    FieldElement52 r;
    fe52_inverse_safegcd_var(tmp.n, r.n);
    return r;
}

} // namespace secp256k1::fast

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#endif // __int128 guard

#undef SECP256K1_FE52_FORCE_INLINE

#endif // SECP256K1_FIELD_52_IMPL_HPP
