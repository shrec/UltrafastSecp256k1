#ifndef SECP256K1_FIELD_52_HPP
#define SECP256K1_FIELD_52_HPP
#pragma once

// ============================================================================
// 5×52-bit Field Element for secp256k1 (Hybrid Lazy-Reduction Scheme)
// ============================================================================
//
// Alternative representation: 5 limbs × 52 bits each in uint64_t[5].
// The upper 12 bits per limb provide "headroom" for lazy reduction:
//
//   Addition  = 5 plain adds (NO carry propagation!)
//   Sub       = 5 adds (pre-add 2p to avoid underflow, NO borrow!)
//   Mul/Sqr   = native 5×52 with inline secp256k1 reduction
//
// Headroom budget: 12 bits → up to 4096 additions without normalization.
// In practice, ECC point operations need ≤50 chained additions.
//
// Hybrid approach: convert between FieldElement (4×64) ↔ FieldElement52 (5×52)
// to use whichever representation is optimal for each code path.
//
//   FieldElement   (4×64) → optimal for: scalar multiply, serialization, I/O
//   FieldElement52 (5×52) → optimal for: ECC point ops (add-heavy chains)
//
// Based on bitcoin-core/secp256k1 field_5x52 representation.
// Multiplication/squaring adapted from bitcoin-core's field_5x52_int128_impl.h
// (MIT license, Copyright (c) 2013-2024 Pieter Wuille and contributors)
// ============================================================================

#include <cstdint>
#include <array>
#include "secp256k1/field.hpp"

namespace secp256k1::fast {

// ── Constants ────────────────────────────────────────────────────────────────
namespace fe52_constants {
    // Mask for 52 bits
    constexpr std::uint64_t M52 = 0xFFFFFFFFFFFFFULL;  // (1 << 52) - 1
    // Mask for 48 bits (top limb)
    constexpr std::uint64_t M48 = 0xFFFFFFFFFFFFULL;   // (1 << 48) - 1
    // 2^260 mod p = 2^4 * (2^256 mod p) = 16 * 0x1000003D1 = 0x1000003D10
    constexpr std::uint64_t R52 = 0x1000003D10ULL;

    // p in 5×52 representation
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    constexpr std::uint64_t P0 = 0xFFFFEFFFFFC2FULL;
    constexpr std::uint64_t P1 = 0xFFFFFFFFFFFFFULL;
    constexpr std::uint64_t P2 = 0xFFFFFFFFFFFFFULL;
    constexpr std::uint64_t P3 = 0xFFFFFFFFFFFFFULL;
    constexpr std::uint64_t P4 = 0xFFFFFFFFFFFFULL;     // 48 bits
}

// ── 5×52 Field Element ───────────────────────────────────────────────────────
struct alignas(8) FieldElement52 {
    std::uint64_t n[5];  // Each limb holds ≤52 bits when normalized (magnitude=1)

    // ── Construction ─────────────────────────────────────────────────
    static FieldElement52 zero() noexcept;
    static FieldElement52 one() noexcept;

    // ── Conversion (4×64 ↔ 5×52) ────────────────────────────────────
    //
    // 4×64 bit layout:  [bits 0..63] [64..127] [128..191] [192..255]
    // 5×52 bit layout:  [bits 0..51] [52..103] [104..155] [156..207] [208..255]
    //
    static FieldElement52 from_fe(const FieldElement& fe) noexcept;
    FieldElement to_fe() const noexcept;  // Normalizes first!

    // ── Normalization ────────────────────────────────────────────────
    // Weak: carry-propagate so each limb ≤ 52 bits, but result may be ≥ p
    void normalize_weak() noexcept;
    // Full: canonical result in [0, p)
    void normalize() noexcept;

    // ── Lazy Arithmetic (NO carry propagation!) ──────────────────────
    // These just do 5 plain adds per operation. Caller is responsible
    // for normalizing before limbs would exceed 64 bits (after ~4096 adds).

    // add: r[i] = a[i] + b[i]  (magnitude += 1)
    FieldElement52 operator+(const FieldElement52& rhs) const noexcept;
    void add_assign(const FieldElement52& rhs) noexcept;

    // ── Negate ───────────────────────────────────────────────────────
    // Computes (magnitude+1)*p - a to ensure positive result.
    // 'magnitude' must be ≥ the current maximum magnitude of this element.
    FieldElement52 negate(unsigned magnitude) const noexcept;
    void negate_assign(unsigned magnitude) noexcept;

    // ── Fully-Reduced Arithmetic ─────────────────────────────────────
    // Multiplication and squaring produce normalized output (magnitude=1).
    FieldElement52 operator*(const FieldElement52& rhs) const noexcept;
    FieldElement52 square() const noexcept;

    // In-place variants
    void mul_assign(const FieldElement52& rhs) noexcept;
    void square_inplace() noexcept;

    // ── Comparison (requires normalized inputs!) ─────────────────────
    bool is_zero() const noexcept;
    bool operator==(const FieldElement52& rhs) const noexcept;
    bool operator!=(const FieldElement52& rhs) const noexcept;

    // ── Half ─────────────────────────────────────────────────────────
    // Computes a/2 mod p. Branchless.
    FieldElement52 half() const noexcept;
};

// ── Free Functions (hot-path, avoid vtable/method overhead) ──────────────────
//
// These operate on raw uint64_t[5] arrays for maximum performance.
// Use these in inner loops where every nanosecond counts.

void fe52_mul_inner(std::uint64_t* r, const std::uint64_t* a,
                    const std::uint64_t* b) noexcept;
void fe52_sqr_inner(std::uint64_t* r, const std::uint64_t* a) noexcept;
void fe52_normalize(std::uint64_t* r) noexcept;
void fe52_normalize_weak(std::uint64_t* r) noexcept;

// Compile-time layout verification
static_assert(sizeof(FieldElement52) == 40, "FieldElement52 must be 40 bytes (5×8)");

} // namespace secp256k1::fast

#endif // SECP256K1_FIELD_52_HPP
