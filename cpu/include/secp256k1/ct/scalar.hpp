#ifndef SECP256K1_CT_SCALAR_HPP
#define SECP256K1_CT_SCALAR_HPP

// ============================================================================
// Constant-Time Scalar Arithmetic
// ============================================================================
// Side-channel resistant scalar operations for secp256k1 curve order.
// Operates on secp256k1::fast::Scalar — same data type, CT execution.
// ============================================================================

#include <cstdint>
#include "secp256k1/scalar.hpp"
#include "secp256k1/ct/ops.hpp"

namespace secp256k1::ct {

using Scalar = secp256k1::fast::Scalar;

// ─── CT Scalar Arithmetic ────────────────────────────────────────────────────

// CT modular addition: r = (a + b) mod n
Scalar scalar_add(const Scalar& a, const Scalar& b) noexcept;

// CT modular subtraction: r = (a - b) mod n
Scalar scalar_sub(const Scalar& a, const Scalar& b) noexcept;

// CT modular negation: r = -a mod n
Scalar scalar_neg(const Scalar& a) noexcept;

// CT modular halving: r = a/2 mod n  (if a is odd: (a+n)/2, else a/2)
Scalar scalar_half(const Scalar& a) noexcept;

// ─── CT Conditional Operations ───────────────────────────────────────────────

// CT conditional move: if (mask) r = a; else r unchanged
void scalar_cmov(Scalar* r, const Scalar& a, std::uint64_t mask) noexcept;

// CT conditional swap: if (mask) swap(a, b)
void scalar_cswap(Scalar* a, Scalar* b, std::uint64_t mask) noexcept;

// CT select: returns a if mask==all-ones, else b
Scalar scalar_select(const Scalar& a, const Scalar& b,
                     std::uint64_t mask) noexcept;

// CT conditional negate: if (mask) r = -a; else r = a
Scalar scalar_cneg(const Scalar& a, std::uint64_t mask) noexcept;

// ─── CT Comparison ───────────────────────────────────────────────────────────

// Returns all-ones mask if a == 0, else 0.
std::uint64_t scalar_is_zero(const Scalar& a) noexcept;

// Returns all-ones mask if a == b, else 0.
std::uint64_t scalar_eq(const Scalar& a, const Scalar& b) noexcept;

// ─── CT Bit Access ───────────────────────────────────────────────────────────

// Returns bit at position 'index' (0 = LSB). CT (always same memory access).
std::uint64_t scalar_bit(const Scalar& a, std::size_t index) noexcept;

// Returns w-bit window at position 'pos' (0 = LSB). CT.
std::uint64_t scalar_window(const Scalar& a, std::size_t pos,
                            unsigned width) noexcept;

} // namespace secp256k1::ct

#endif // SECP256K1_CT_SCALAR_HPP
