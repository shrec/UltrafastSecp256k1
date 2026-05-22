// ============================================================================
// Private: Constant-Time Carry/Borrow Helpers
// ============================================================================
// Shared between ct_field.cpp and ct_scalar.cpp to avoid duplicate static
// definitions that cause redefinition errors under SECP256K1_UNITY_BUILD.
//
// NOT part of the public API. Do not include from headers.
// ============================================================================

#pragma once
#ifndef SECP256K1_CT_CARRY_HELPERS_P_HPP
#define SECP256K1_CT_CARRY_HELPERS_P_HPP

#include <cstdint>

namespace secp256k1::ct {

// Returns the carry out of (a + b) = sum, as 0 or 1.
// Purely bitwise — no branches, no memory access patterns.
[[maybe_unused]] inline std::uint64_t add_carry_u64(std::uint64_t a,
                                                     std::uint64_t b,
                                                     std::uint64_t sum) noexcept {
    return ((a & b) | ((a | b) & ~sum)) >> 63;
}

// Returns the borrow out of (a - b) = diff, as 0 or 1.
// Purely bitwise — no branches, no memory access patterns.
[[maybe_unused]] inline std::uint64_t sub_borrow_u64(std::uint64_t a,
                                                      std::uint64_t b,
                                                      std::uint64_t diff) noexcept {
    return (a ^ ((a ^ b) | (diff ^ a))) >> 63;
}

} // namespace secp256k1::ct

#endif // SECP256K1_CT_CARRY_HELPERS_P_HPP
