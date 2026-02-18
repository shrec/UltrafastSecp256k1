#ifndef SECP256K1_FIELD_26_HPP
#define SECP256K1_FIELD_26_HPP
#pragma once

// ============================================================================
// 10×26-bit Field Element for secp256k1 (Lazy-Reduction for 32-bit Platforms)
// ============================================================================
//
// Alternative representation: 10 limbs × 26 bits each in uint32_t[10].
// The upper 6 bits per limb provide "headroom" for lazy reduction:
//
//   Addition  = 10 plain adds (NO carry propagation!)
//   Sub       = 10 adds (pre-add 2p to avoid underflow, NO borrow!)
//   Mul/Sqr   = native 10×26 with inline secp256k1 reduction
//
// Headroom budget: 6 bits → up to 64 additions without normalization.
// In practice, ECC point operations need ≤50 chained additions.
//
// This is the 32-bit counterpart of FieldElement52 (5×52 for 64-bit CPUs).
// Targets: ESP32 (Xtensa LX6/LX7), STM32 (Cortex-M3/M4), any 32-bit CPU.
//
//   FieldElement   (4×64) → optimal for: 64-bit CPUs, serialization, I/O
//   FieldElement52 (5×52) → optimal for: 64-bit ECC point ops (lazy adds)
//   FieldElement26 (10×26)→ optimal for: 32-bit ECC point ops (lazy adds)
//
// Multiplication/squaring adapted from bitcoin-core/secp256k1 field_10x26_impl.h
// (MIT license, Copyright (c) 2013-2024 Pieter Wuille and contributors)
// ============================================================================

#include <cstdint>
#include <array>
#include "secp256k1/field.hpp"

namespace secp256k1::fast {

// ── Constants ────────────────────────────────────────────────────────────────
namespace fe26_constants {
    // Mask for 26 bits
    constexpr std::uint32_t M26 = 0x3FFFFFFU;  // (1 << 26) - 1
    // Mask for 22 bits (top limb)
    constexpr std::uint32_t M22 = 0x3FFFFFU;    // (1 << 22) - 1

    // p in 10×26 representation
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    //
    // Split into 26-bit chunks from LSB:
    // bits  [0..25]   = 0x3FFFC2F
    // bits  [26..51]  = 0x3FFFFBF  (note: FFFFFFEFFFFC2F → ...03FFFFBF at this position)
    // bits  [52..77]  = 0x3FFFFFF
    // ...
    // bits  [234..255] = 0x3FFFFF  (22 bits)
    //
    // Actually: p = 2^256 - 0x1000003D1
    //   limb 0: bits[0..25]     → lower 26 of 0xFFFFFC2F = 0x3FFFC2F
    //   limb 1: bits[26..51]    → (0xFFFFFFFEFFFFFC2F >> 26) & M26 = 0x3FFFFBF
    //   limb 2: bits[52..77]    → all 1s = 0x3FFFFFF
    //   limb 3..8: all 1s       = 0x3FFFFFF
    //   limb 9: bits[234..255]  → 22 bits, all 1s = 0x3FFFFF
    constexpr std::uint32_t P0 = 0x3FFFC2FU;
    constexpr std::uint32_t P1 = 0x3FFFFBFU;
    constexpr std::uint32_t P2 = 0x3FFFFFFU;
    constexpr std::uint32_t P3 = 0x3FFFFFFU;
    constexpr std::uint32_t P4 = 0x3FFFFFFU;
    constexpr std::uint32_t P5 = 0x3FFFFFFU;
    constexpr std::uint32_t P6 = 0x3FFFFFFU;
    constexpr std::uint32_t P7 = 0x3FFFFFFU;
    constexpr std::uint32_t P8 = 0x3FFFFFFU;
    constexpr std::uint32_t P9 = 0x3FFFFFU;        // 22 bits
}

// ── 10×26 Field Element ──────────────────────────────────────────────────────
struct alignas(4) FieldElement26 {
    std::uint32_t n[10];  // Each limb holds ≤26 bits when normalized (magnitude=1)

    // ── Construction ─────────────────────────────────────────────────
    static FieldElement26 zero() noexcept;
    static FieldElement26 one() noexcept;

    // ── Conversion (4×64 ↔ 10×26) ───────────────────────────────────
    //
    // 4×64 bit layout:  [bits 0..63] [64..127] [128..191] [192..255]
    // 10×26 bit layout: [bits 0..25] [26..51] [52..77] [78..103]
    //                   [104..129] [130..155] [156..181] [182..207]
    //                   [208..233] [234..255]
    //
    static FieldElement26 from_fe(const FieldElement& fe) noexcept;
    FieldElement to_fe() const noexcept;  // Normalizes first!

    // ── Normalization ────────────────────────────────────────────────
    // Weak: carry-propagate so each limb ≤ 26 bits, but result may be ≥ p
    void normalize_weak() noexcept;
    // Full: canonical result in [0, p)
    void normalize() noexcept;

    // ── Lazy Arithmetic (NO carry propagation!) ──────────────────────
    // These just do 10 plain adds per operation. Caller is responsible
    // for normalizing before limbs would exceed 32 bits (after ~64 adds).

    FieldElement26 operator+(const FieldElement26& rhs) const noexcept;
    void add_assign(const FieldElement26& rhs) noexcept;

    // ── Negate ───────────────────────────────────────────────────────
    // Computes (magnitude+1)*p - a to ensure positive result.
    FieldElement26 negate(unsigned magnitude) const noexcept;
    void negate_assign(unsigned magnitude) noexcept;

    // ── Fully-Reduced Arithmetic ─────────────────────────────────────
    // Multiplication and squaring produce normalized output (magnitude=1).
    FieldElement26 operator*(const FieldElement26& rhs) const noexcept;
    FieldElement26 square() const noexcept;

    // In-place variants
    void mul_assign(const FieldElement26& rhs) noexcept;
    void square_inplace() noexcept;

    // ── Comparison (requires normalized inputs!) ─────────────────────
    bool is_zero() const noexcept;
    bool operator==(const FieldElement26& rhs) const noexcept;
    bool operator!=(const FieldElement26& rhs) const noexcept;

    // ── Half ─────────────────────────────────────────────────────────
    // Computes a/2 mod p. Branchless.
    FieldElement26 half() const noexcept;
};

// ── Free Functions (hot-path, avoid vtable/method overhead) ──────────────────
void fe26_mul_inner(std::uint32_t* r, const std::uint32_t* a,
                    const std::uint32_t* b) noexcept;
void fe26_sqr_inner(std::uint32_t* r, const std::uint32_t* a) noexcept;
void fe26_normalize(std::uint32_t* r) noexcept;
void fe26_normalize_weak(std::uint32_t* r) noexcept;

// Compile-time layout verification
static_assert(sizeof(FieldElement26) == 40, "FieldElement26 must be 40 bytes (10×4)");

} // namespace secp256k1::fast

#endif // SECP256K1_FIELD_26_HPP
