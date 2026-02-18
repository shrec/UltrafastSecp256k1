#ifndef SECP256K1_FIELD_OPTIMAL_HPP
#define SECP256K1_FIELD_OPTIMAL_HPP
#pragma once

// ============================================================================
// Compile-Time Optimal Field Element Selection
// ============================================================================
//
// Selects the best FieldElement representation per platform based on
// measured benchmarks (Release builds, Feb 2026).
//
// KEY PRINCIPLE: Only activate alternative representations where they
// actually beat the native FieldElement (4×64 + platform asm).
// Where hand-tuned asm already exists and is faster, stay native.
//
// Decision matrix (measured, all verified Feb 2026):
//
//   Platform              Native asm?   Optimal         Mul Winner (verified)
//   ─────────────────────────────────────────────────────────────────────────
//   x86-64 (BMI2/ADX)    YES (GAS/MASM) FieldElement52  5×52 21ns  vs native 44ns  (+2.1×)
//   ARM64  (RK3588)      YES (MUL/UMULH) FieldElement26  10×26 73ns vs native 85ns  (+1.16×)
//   RISC-V 64 (RVV)      YES (RV asm)   FieldElement52  5×52 142ns vs native 173ns (+1.22×)
//   STM32  (Cortex-M4)   YES (Comba)    FieldElement    native 15.4μs vs 10×26 18.1μs
//   ESP32-S3 (LX7)       NO             FieldElement    native 6.0μs  vs 10×26 8.6μs
//   ESP32-PICO (LX6)     NO             FieldElement26  10×26 6.4μs  vs native 7.1μs (+1.11×)
//   MSVC x64             varies         FieldElement    no __int128
//   ARM32 generic        NO             FieldElement26  10×26 expected
//
// USAGE:
//   #include "secp256k1/field_optimal.hpp"
//   using FE = secp256k1::fast::OptimalFieldElement;
//
//   FE a = secp256k1::fast::to_optimal(some_4x64_element);
//   FE c = a * b;                          // uses best mul for this platform
//   FieldElement result = secp256k1::fast::from_optimal(c);
//
// API contract (all representations provide):
//   operator*, square(), operator+, add_assign(), negate(), half(),
//   normalize(), normalize_weak(), is_zero(), operator==,
//   from_fe()/to_fe() for FieldElement ↔ optimal conversion.
// ============================================================================

#include "secp256k1/config.hpp"
#include "secp256k1/field.hpp"

// ── Step 1: Platforms where native asm already beats alternatives ────────────
//
// On these platforms, hand-tuned assembly in FieldElement is already the
// fastest option. Do NOT override with 5×52 or 10×26 — it would be slower.
//
//   SECP256K1_OPTIMAL_TIER_52  →  5×52 limbs  (64-bit + __int128)
//   SECP256K1_OPTIMAL_TIER_26  → 10×26 limbs  (32-bit, native 32×32→64)
//   SECP256K1_OPTIMAL_TIER_64  →  4×64 limbs  (native asm / fallback)
//

#if defined(SECP256K1_PLATFORM_STM32)
    // ── STM32 Cortex-M with Comba asm: native FieldElement wins ──
    // Measured: native Mul 15.4μs vs 10×26 Mul 18.1μs (-18%)
    // Native 32-bit Comba (esp32_mul_mod) is already optimal.
    #define SECP256K1_OPTIMAL_TIER_64 1

#elif defined(SECP256K1_HAS_ARM64_ASM) || defined(__aarch64__) || defined(_M_ARM64)
    // ── ARM64 (MUL/UMULH asm): 10×26 wins everything ──
    // Verified on RK3588 (Cortex-A55/A76), Feb 2026:
    //   10×26 Mul  73ns vs native  85ns vs 5×52 100ns  → 10×26 best
    //   10×26 Sqr  53ns vs native  66ns vs 5×52  72ns  → 10×26 best
    //   10×26 Add   7ns vs native  19ns vs 5×52  11ns  → 10×26 best
    // Despite having 64-bit MUL, the 10×26 two-accumulator algorithm
    // wins because it avoids __int128 overhead on AArch64 compilers.
    #define SECP256K1_OPTIMAL_TIER_26 1
    #define SECP256K1_OPTIMAL_ARM64   1
    #include "secp256k1/field_26.hpp"

// ── Step 2: Platforms where 5×52 wins ───────────────────────────────────────

#elif !defined(SECP256K1_NO_INT128) && \
    (defined(__SIZEOF_INT128__) || \
     (defined(__GNUC__) && !defined(__i386__) && !defined(__arm__) && !defined(__xtensa__)))
    // ── 64-bit with __int128 (x86-64, RISC-V 64): 5×52 wins ──
    // Verified Feb 2026:
    //   x86-64:  Mul  21ns vs native  44ns (+2.1×), Sqr 15ns vs 40ns (+2.7×)
    //   RISC-V:  Mul 142ns vs native 173ns (+1.2×), Sqr 102ns vs 160ns (+1.6×)
    // 5×52 lazy-reduction with __int128 beats native asm.
    #define SECP256K1_OPTIMAL_TIER_52 1
    #include "secp256k1/field_52.hpp"

#elif defined(SECP256K1_32BIT)
    // 32-bit platform — per-MCU selection
    #if defined(CONFIG_IDF_TARGET_ESP32S3) || defined(SECP256K1_FORCE_TIER_64)
        // ── ESP32-S3 (LX7): native 4×64 emulated mul wins ──
        // Verified: Mul 5,818ns vs 10×26 8,639ns (+1.49×)
        //           Sqr 4,799ns vs 10×26 5,229ns (+1.09×)
        #define SECP256K1_OPTIMAL_TIER_64 1
    #else
        // ── ESP32-PICO (LX6), generic ARM32, RISC-V 32 ──
        // Verified PICO: Mul 6,405ns vs native 7,120ns (+1.11×)
        //                Sqr 3,972ns vs native 6,366ns (+1.60×)
        //                Add   408ns vs native   980ns (+2.40×)
        #define SECP256K1_OPTIMAL_TIER_26 1
        #include "secp256k1/field_26.hpp"
    #endif

#else
    // 64-bit without __int128 (MSVC x64, or explicit SECP256K1_NO_INT128)
    #define SECP256K1_OPTIMAL_TIER_64 1
#endif


namespace secp256k1::fast {

// ── The type alias ──────────────────────────────────────────────────────────
#if defined(SECP256K1_OPTIMAL_TIER_52)
    using OptimalFieldElement = FieldElement52;
#elif defined(SECP256K1_OPTIMAL_TIER_26)
    using OptimalFieldElement = FieldElement26;
#else
    using OptimalFieldElement = FieldElement;
#endif

// ── Compile-time tag ────────────────────────────────────────────────────────
enum class FieldTier { FE64, FE52, FE26 };

#if defined(SECP256K1_OPTIMAL_TIER_52)
    inline constexpr FieldTier kOptimalTier = FieldTier::FE52;
    inline constexpr const char* kOptimalTierName = "5x52 (64-bit, __int128)";
#elif defined(SECP256K1_OPTIMAL_TIER_26)
    inline constexpr FieldTier kOptimalTier = FieldTier::FE26;
    #if defined(SECP256K1_OPTIMAL_ARM64)
        inline constexpr const char* kOptimalTierName = "10x26 (ARM64 — wins all field ops)";
    #else
        inline constexpr const char* kOptimalTierName = "10x26 (32-bit native)";
    #endif
#else
    inline constexpr FieldTier kOptimalTier = FieldTier::FE64;
    #if defined(SECP256K1_PLATFORM_STM32)
        inline constexpr const char* kOptimalTierName = "4x64 (STM32 Comba — already optimal)";
    #elif defined(CONFIG_IDF_TARGET_ESP32S3)
        inline constexpr const char* kOptimalTierName = "4x64 (ESP32-S3 — native wins mul)";
    #else
        inline constexpr const char* kOptimalTierName = "4x64 (baseline)";
    #endif
#endif

// ── Conversion helpers ──────────────────────────────────────────────────────
// Zero-cost when OptimalFieldElement == FieldElement (identity).
// Otherwise, bit-rearrangement (cheap compared to any mul/sqr).

inline OptimalFieldElement to_optimal(const FieldElement& fe) noexcept {
#if defined(SECP256K1_OPTIMAL_TIER_52)
    return FieldElement52::from_fe(fe);
#elif defined(SECP256K1_OPTIMAL_TIER_26)
    return FieldElement26::from_fe(fe);
#else
    return fe;
#endif
}

inline FieldElement from_optimal(const OptimalFieldElement& ofe) noexcept {
#if defined(SECP256K1_OPTIMAL_TIER_52)
    return ofe.to_fe();
#elif defined(SECP256K1_OPTIMAL_TIER_26)
    return ofe.to_fe();
#else
    return ofe;
#endif
}

} // namespace secp256k1::fast

#endif // SECP256K1_FIELD_OPTIMAL_HPP
