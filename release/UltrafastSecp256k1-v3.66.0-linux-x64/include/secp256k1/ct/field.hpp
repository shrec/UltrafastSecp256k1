#ifndef SECP256K1_CT_FIELD_HPP
#define SECP256K1_CT_FIELD_HPP

// ============================================================================
// Constant-Time Field Arithmetic
// ============================================================================
// Side-channel resistant field operations for secp256k1.
// Operates on secp256k1::fast::FieldElement -- same data type, CT execution.
//
// Guarantees:
//   - No secret-dependent branches
//   - No secret-dependent memory access patterns
//   - Fixed instruction count for all inputs
//
// Usage:
//   using FE = secp256k1::fast::FieldElement;
//   FE a = ..., b = ...;
//   FE r = secp256k1::ct::field_add(a, b);   // CT modular addition
//   secp256k1::ct::field_cmov(&r, &a, flag); // CT conditional move
//
// Mixing with fast::
//   FE x = secp256k1::fast::FieldElement::from_hex("...");  // public data
//   FE secret = ...; // secret scalar, etc.
//   FE r = secp256k1::ct::field_mul(x, secret);  // CT for secret operand
// ============================================================================

#include <cstdint>
#include "secp256k1/field.hpp"
#include "secp256k1/ct/ops.hpp"

namespace secp256k1::ct {

using FieldElement = secp256k1::fast::FieldElement;

// --- CT Modular Arithmetic ---------------------------------------------------
// mul and square are inherently constant-time (fixed mul count).
// add/sub/normalize need CT reduction.

// CT modular addition: r = (a + b) mod p
FieldElement field_add(const FieldElement& a, const FieldElement& b) noexcept;

// CT modular subtraction: r = (a - b) mod p
FieldElement field_sub(const FieldElement& a, const FieldElement& b) noexcept;

// CT modular multiplication: r = (a * b) mod p
// Note: The underlying mul is already CT (fixed Comba/schoolbook).
// This wrapper ensures CT reduction.
FieldElement field_mul(const FieldElement& a, const FieldElement& b) noexcept;

// CT modular squaring: r = a^2 mod p
FieldElement field_sqr(const FieldElement& a) noexcept;

// CT modular negation: r = -a mod p = p - a (if a != 0), 0 (if a == 0)
FieldElement field_neg(const FieldElement& a) noexcept;

// CT modular half: r = a/2 mod p
// If a is odd: r = (a + p) / 2; if even: r = a / 2. Branchless.
FieldElement field_half(const FieldElement& a) noexcept;

// CT modular inverse: r = a^-^1 mod p  (via Fermat: a^(p-2))
// Fixed add-chain: always same number of mul+sqr regardless of input
FieldElement field_inv(const FieldElement& a) noexcept;

// --- CT Conditional Operations -----------------------------------------------

// CT conditional move: if (mask == all-ones) r = a; else r unchanged
// mask MUST be 0x0000000000000000 or 0xFFFFFFFFFFFFFFFF
void field_cmov(FieldElement* r, const FieldElement& a,
                std::uint64_t mask) noexcept;

// CT conditional swap: when mask is all-ones, swaps a and b; otherwise unchanged
void field_cswap(FieldElement* a, FieldElement* b,
                 std::uint64_t mask) noexcept;

// CT select: returns a if mask==all-ones, else b
FieldElement field_select(const FieldElement& a, const FieldElement& b,
                          std::uint64_t mask) noexcept;

// CT conditional negate: if (mask == all-ones) r = -a; else r = a
FieldElement field_cneg(const FieldElement& a, std::uint64_t mask) noexcept;

// --- CT Comparison -----------------------------------------------------------

// Returns all-ones mask if a == 0, else 0. CT (no branch).
std::uint64_t field_is_zero(const FieldElement& a) noexcept;

// Returns all-ones mask if a == b, else 0. CT (no branch).
std::uint64_t field_eq(const FieldElement& a, const FieldElement& b) noexcept;

// --- CT Normalize ------------------------------------------------------------
// Reduces a field element to canonical form [0, p) without branches.
FieldElement field_normalize(const FieldElement& a) noexcept;

} // namespace secp256k1::ct

#endif // SECP256K1_CT_FIELD_HPP
