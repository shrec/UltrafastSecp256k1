#ifndef SECP256K1_FIELD_BRANCHLESS_HPP
#define SECP256K1_FIELD_BRANCHLESS_HPP

#include <cstdint>
#include "field.hpp"

namespace secp256k1::fast {

// ============================================================================
// Branchless Operations (GPU-inspired, CPU-optimized)
// ============================================================================
// These functions use conditional moves (CMOV) instead of branches for
// predictable performance. Critical for hot paths and batch operations.
//
// Performance impact:
//   - Eliminates branch mispredictions (~10-20 cycles penalty)
//   - Enables better pipelining and instruction-level parallelism
//   - 5-10% faster in tight loops with unpredictable conditions
//
// Usage:
//   - field_cmov: Conditional move entire field element
//   - field_cmovznz: Move based on zero/nonzero flag
//   - field_select: Select between two field elements
//
// ============================================================================

// Branchless conditional move: if (flag) *r = *a; else *r = *b;
// Uses CMOV instruction when available (x86_64, ARM64)
// flag MUST be 0 or 1 (behavior undefined otherwise)
inline void field_cmov(FieldElement* r, const FieldElement* a, 
                       const FieldElement* b, bool flag) noexcept {
    // Convert flag to mask: 0 -> 0x0000000000000000, 1 -> 0xFFFFFFFFFFFFFFFF
    std::uint64_t const mask = 0ULL - static_cast<std::uint64_t>(flag);
    
    auto& r_limbs = const_cast<std::array<std::uint64_t, 4>&>(r->limbs());
    const auto& a_limbs = a->limbs();
    const auto& b_limbs = b->limbs();
    
    // Branchless selection: r = (a & mask) | (b & ~mask)
    r_limbs[0] = (a_limbs[0] & mask) | (b_limbs[0] & ~mask);
    r_limbs[1] = (a_limbs[1] & mask) | (b_limbs[1] & ~mask);
    r_limbs[2] = (a_limbs[2] & mask) | (b_limbs[2] & ~mask);
    r_limbs[3] = (a_limbs[3] & mask) | (b_limbs[3] & ~mask);
}

// Conditional move zero/nonzero: if (flag != 0) *r = *a; else *r = *b;
// More flexible than field_cmov: flag can be any value (not just 0/1)
inline void field_cmovznz(FieldElement* r, const FieldElement* a, 
                          const FieldElement* b, std::uint64_t flag) noexcept {
    // Convert flag to mask: 0 -> 0x0000000000000000, nonzero -> 0xFFFFFFFFFFFFFFFF
    // Use bitwise OR reduction across all bits to detect nonzero
    std::uint64_t mask = (flag | (0ULL - flag)) >> 63; // Branchless nonzero test
    mask = 0ULL - mask; // Expand to full 64-bit mask
    
    auto& r_limbs = const_cast<std::array<std::uint64_t, 4>&>(r->limbs());
    const auto& a_limbs = a->limbs();
    const auto& b_limbs = b->limbs();
    
    r_limbs[0] = (a_limbs[0] & mask) | (b_limbs[0] & ~mask);
    r_limbs[1] = (a_limbs[1] & mask) | (b_limbs[1] & ~mask);
    r_limbs[2] = (a_limbs[2] & mask) | (b_limbs[2] & ~mask);
    r_limbs[3] = (a_limbs[3] & mask) | (b_limbs[3] & ~mask);
}

// Select between two field elements: return (flag ? a : b)
// Returns NEW FieldElement (no in-place mutation)
inline FieldElement field_select(const FieldElement& a, const FieldElement& b, 
                                   bool flag) noexcept {
    std::uint64_t const mask = 0ULL - static_cast<std::uint64_t>(flag);
    
    const auto& a_limbs = a.limbs();
    const auto& b_limbs = b.limbs();
    
    return FieldElement::from_limbs({
        (a_limbs[0] & mask) | (b_limbs[0] & ~mask),
        (a_limbs[1] & mask) | (b_limbs[1] & ~mask),
        (a_limbs[2] & mask) | (b_limbs[2] & ~mask),
        (a_limbs[3] & mask) | (b_limbs[3] & ~mask)
    });
}

// Check if field element is zero (branchless)
// Returns 1 if zero, 0 if nonzero
inline std::uint64_t field_is_zero(const FieldElement& a) noexcept {
    const auto& limbs = a.limbs();
    std::uint64_t const z = limbs[0] | limbs[1] | limbs[2] | limbs[3];
    // Branchless zero check: z==0 ? 1 : 0
    return (z | (0ULL - z)) >> 63 ^ 1;
}

// Check if two field elements are equal (branchless)
// Returns 1 if equal, 0 if different
inline std::uint64_t field_eq(const FieldElement& a, const FieldElement& b) noexcept {
    const auto& a_limbs = a.limbs();
    const auto& b_limbs = b.limbs();
    
    std::uint64_t const diff = (a_limbs[0] ^ b_limbs[0]) | 
                         (a_limbs[1] ^ b_limbs[1]) |
                         (a_limbs[2] ^ b_limbs[2]) | 
                         (a_limbs[3] ^ b_limbs[3]);
    
    // Branchless equality check: diff==0 ? 1 : 0
    return (diff | (0ULL - diff)) >> 63 ^ 1;
}

// Conditional negate: if (flag) *r = -a; else *r = a;
// Used in GLV scalar decomposition and point operations
inline void field_cneg(FieldElement* r, const FieldElement& a, bool flag) noexcept {
    FieldElement const negated = FieldElement::zero() - a;
    field_cmov(r, &negated, &a, flag);
}

// Conditional addition: if (flag) *r = a + b; else *r = a;
// Avoids branch in tight loops
inline void field_cadd(FieldElement* r, const FieldElement& a, 
                       const FieldElement& b, bool flag) noexcept {
    FieldElement const sum = a + b;
    field_cmov(r, &sum, &a, flag);
}

// Conditional subtraction: sets *r to (a - b) when flag is true, otherwise keeps a
inline void field_csub(FieldElement* r, const FieldElement& a, 
                       const FieldElement& b, bool flag) noexcept {
    FieldElement const diff = a - b;
    field_cmov(r, &diff, &a, flag);
}

} // namespace secp256k1::fast

#endif // SECP256K1_FIELD_BRANCHLESS_HPP
