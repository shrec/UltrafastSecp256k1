#ifndef SECP256K1_FIELD_52_HPP
#define SECP256K1_FIELD_52_HPP
#pragma once

// ============================================================================
// 5x52-bit Field Element for secp256k1 (Hybrid Lazy-Reduction Scheme)
// ============================================================================
//
// Alternative representation: 5 limbs x 52 bits each in uint64_t[5].
// The upper 12 bits per limb provide "headroom" for lazy reduction:
//
//   Addition  = 5 plain adds (NO carry propagation!)
//   Sub       = 5 adds (pre-add 2p to avoid underflow, NO borrow!)
//   Mul/Sqr   = native 5x52 with inline secp256k1 reduction
//
// Headroom budget: 12 bits -> up to 4096 additions without normalization.
// In practice, ECC point operations need <=50 chained additions.
//
// Hybrid approach: convert between FieldElement (4x64) <-> FieldElement52 (5x52)
// to use whichever representation is optimal for each code path.
//
//   FieldElement   (4x64) -> optimal for: scalar multiply, serialization, I/O
//   FieldElement52 (5x52) -> optimal for: ECC point ops (add-heavy chains)
//
// Based on bitcoin-core/secp256k1 field_5x52 representation.
// Multiplication/squaring adapted from bitcoin-core's field_5x52_int128_impl.h
// (MIT license, Copyright (c) 2013-2024 Pieter Wuille and contributors)
// ============================================================================

#include <cstdint>
#include <array>
#include "secp256k1/field.hpp"

namespace secp256k1::fast {

// -- Constants ----------------------------------------------------------------
namespace fe52_constants {
    // Mask for 52 bits
    constexpr std::uint64_t M52 = 0xFFFFFFFFFFFFFULL;  // (1 << 52) - 1
    // Mask for 48 bits (top limb)
    constexpr std::uint64_t M48 = 0xFFFFFFFFFFFFULL;   // (1 << 48) - 1
    // 2^260 mod p = 2^4 * (2^256 mod p) = 16 * 0x1000003D1 = 0x1000003D10
    constexpr std::uint64_t R52 = 0x1000003D10ULL;

    // p in 5x52 representation
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    constexpr std::uint64_t P0 = 0xFFFFEFFFFFC2FULL;
    constexpr std::uint64_t P1 = 0xFFFFFFFFFFFFFULL;
    constexpr std::uint64_t P2 = 0xFFFFFFFFFFFFFULL;
    constexpr std::uint64_t P3 = 0xFFFFFFFFFFFFFULL;
    constexpr std::uint64_t P4 = 0xFFFFFFFFFFFFULL;     // 48 bits
}

// -- 5x52 Field Element -------------------------------------------------------
struct alignas(8) FieldElement52 {
    std::uint64_t n[5];  // Each limb holds <=52 bits when normalized (magnitude=1)

    // -- Construction -------------------------------------------------
    static FieldElement52 zero() noexcept;
    static FieldElement52 one() noexcept;

    // -- Conversion (4x64 <-> 5x52) ------------------------------------
    //
    // 4x64 bit layout:  [bits 0..63] [64..127] [128..191] [192..255]
    // 5x52 bit layout:  [bits 0..51] [52..103] [104..155] [156..207] [208..255]
    //
    static FieldElement52 from_fe(const FieldElement& fe) noexcept;
    FieldElement to_fe() const noexcept;  // Normalizes first!

    // Direct 4x64 limbs -> 5x52 conversion (zero-copy, no FieldElement construction).
    // Input: 4 little-endian uint64_t limbs representing a value < p.
    // Use for Scalar->FE52 where we know value < n < p.
    static FieldElement52 from_4x64_limbs(const std::uint64_t* limbs) noexcept;

    // Direct bytes (big-endian) -> 5x52 conversion (no FieldElement construction).
    // Reads 32 BE bytes, reduces mod p if needed, converts to 5x52 limbs.
    // Replaces the common pattern: FE52::from_fe(FieldElement::from_bytes(data))
    static FieldElement52 from_bytes(const std::array<std::uint8_t, 32>& bytes) noexcept;
    static FieldElement52 from_bytes(const std::uint8_t* bytes) noexcept;

    // Inverse via safegcd (4x64): FE52 -> FE(4x64) -> safegcd -> FE52
    // Single call replaces the common pattern: from_fe(to_fe().inverse())
    FieldElement52 inverse_safegcd() const noexcept;

    // -- Normalization ------------------------------------------------
    // Weak: carry-propagate so each limb <= 52 bits, but result may be >= p
    void normalize_weak() noexcept;
    // Full: canonical result in [0, p)
    void normalize() noexcept;

    // -- Lazy Arithmetic (NO carry propagation!) ----------------------
    // These just do 5 plain adds per operation. Caller is responsible
    // for normalizing before limbs would exceed 64 bits (after ~4096 adds).

    // add: result limb i equals sum of operand limbs (magnitude increases by 1)
    FieldElement52 operator+(const FieldElement52& rhs) const noexcept;
    void add_assign(const FieldElement52& rhs) noexcept;

    // -- Negate -------------------------------------------------------
    // Computes (magnitude+1)*p - a to ensure positive result.
    // 'magnitude' must be >= the current maximum magnitude of this element.
    FieldElement52 negate(unsigned magnitude) const noexcept;
    void negate_assign(unsigned magnitude) noexcept;

    // -- Fully-Reduced Arithmetic -------------------------------------
    // Multiplication and squaring produce normalized output (magnitude=1).
    FieldElement52 operator*(const FieldElement52& rhs) const noexcept;
    FieldElement52 square() const noexcept;

    // In-place variants
    void mul_assign(const FieldElement52& rhs) noexcept;
    void square_inplace() noexcept;

    // -- Comparison (requires normalized inputs!) ---------------------
    bool is_zero() const noexcept;
    bool operator==(const FieldElement52& rhs) const noexcept;

    // -- Fast Variable-time Zero Check --------------------------------
    // Checks if value reduces to zero mod p WITHOUT full normalization.
    // Much cheaper than normalize()+is_zero() -- no copy, no canonical form.
    // Variable-time: safe for non-secret values (point coordinates in ECC).
    bool normalizes_to_zero() const noexcept;

    // Variable-time zero check with early exit (verify hot path).
    // Combined normalize_weak + zero check in a single pass.
    // Safe for magnitudes up to ~4000.  99.99..% of calls exit after
    // a single OR-reduction (non-zero fast path).  Only values that
    // happen to be 0 or p fall through to the full limb comparison.
    bool normalizes_to_zero_var() const noexcept;

    // -- Half ---------------------------------------------------------
    // Computes a/2 mod p. Branchless.
    FieldElement52 half() const noexcept;

    // -- Inverse (Fermat) ---------------------------------------------
    // a^(p-2) mod p. 255 squarings + 14 multiplications in native FE52.
    // ~1.6us -- faster than binary GCD (~3us) or SafeGCD (~3-5us in 4x64).
    FieldElement52 inverse() const noexcept;

    // -- Square Root --------------------------------------------------
    // a^((p+1)/4) mod p. 253 squarings + 13 multiplications in native FE52.
    // Returns a candidate y such that y^2 == a (mod p). Caller must verify.
    FieldElement52 sqrt() const noexcept;
};

// -- Free Functions ------------------------------------------------------------
//
// Hot-path functions (fe52_mul_inner, fe52_sqr_inner, fe52_normalize_weak,
// fe52_normalize, from_fe, to_fe) are defined INLINE in field_52_impl.hpp
// for zero call overhead.

// Out-of-line fe52_normalize kept for backward compatibility with code
// that calls it outside hot paths. Actual implementation is inline.
void fe52_normalize(std::uint64_t* r) noexcept;

// Compile-time layout verification
static_assert(sizeof(FieldElement52) == 40, "FieldElement52 must be 40 bytes (5x8)");

} // namespace secp256k1::fast

// Inline implementations of all hot-path 5x52 operations.
// Must come AFTER the FieldElement52 class definition.
#include "secp256k1/field_52_impl.hpp"

#endif // SECP256K1_FIELD_52_HPP
