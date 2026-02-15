#ifndef SECP256K1_PEDERSEN_HPP
#define SECP256K1_PEDERSEN_HPP
#pragma once

// ============================================================================
// Pedersen Commitments for secp256k1
// ============================================================================
// Homomorphic commitments: C = v*H + r*G
//   - v: committed value (scalar)
//   - r: blinding factor (scalar)
//   - G: generator (standard secp256k1 generator)
//   - H: alternate generator (nothing-up-my-sleeve construction)
//
// Properties:
//   - Hiding: C reveals nothing about v (given random r)
//   - Binding: cannot open C to a different (v', r')
//   - Homomorphic: C1 + C2 = commit(v1+v2, r1+r2)
//
// Used in: Confidential Transactions, Mimblewimble, Liquid, Bulletproofs
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// ── Pedersen Commitment ──────────────────────────────────────────────────────

struct PedersenCommitment {
    fast::Point point;   // The commitment point C = v*H + r*G

    // Serialize to 33 bytes (compressed point)
    std::array<std::uint8_t, 33> to_compressed() const;

    // Add two commitments: C1 + C2 (homomorphic addition)
    PedersenCommitment operator+(const PedersenCommitment& rhs) const;

    // Verify that this commitment equals commit(value, blinding)
    bool verify(const fast::Scalar& value, const fast::Scalar& blinding) const;
};

// ── Alternate Generator H ────────────────────────────────────────────────────

// Get alternate generator H (nothing-up-my-sleeve: H = lift_x(SHA256("Pedersen_H")))
// Cached after first call.
const fast::Point& pedersen_generator_H();

// ── Commit / Open ────────────────────────────────────────────────────────────

// Create Pedersen commitment: C = v*H + r*G
// value: the committed value
// blinding: random blinding factor (must be secret)
PedersenCommitment pedersen_commit(const fast::Scalar& value,
                                   const fast::Scalar& blinding);

// Verify commitment opens to (value, blinding):
// C == v*H + r*G
bool pedersen_verify(const PedersenCommitment& commitment,
                     const fast::Scalar& value,
                     const fast::Scalar& blinding);

// ── Homomorphic Operations ───────────────────────────────────────────────────

// Verify that commitments sum to zero (for balance proofs):
// sum(commitments) + excess*G == 0
// This checks: sum(v_i)*H + sum(r_i)*G == 0
// In practice: sum(output_v) - sum(input_v) = 0 and excess = sum(output_r) - sum(input_r)
bool pedersen_verify_sum(const PedersenCommitment* commitments_pos,
                         std::size_t n_pos,
                         const PedersenCommitment* commitments_neg,
                         std::size_t n_neg);

// Compute blinding factor that balances a set of commitments:
// Given input blindings and output blindings (except last),
// compute the last output blinding so the sum balances.
// blind_out = sum(blind_in) - sum(blind_out_partial)
fast::Scalar pedersen_blind_sum(const fast::Scalar* blinds_in,
                                std::size_t n_in,
                                const fast::Scalar* blinds_out,
                                std::size_t n_out);

// ── Switch Commitment (Mimblewimble) ─────────────────────────────────────────

// Create switch commitment: C = v*H + r*G + switch_blind*J
// J is a third generator for switch commitments
const fast::Point& pedersen_generator_J();

PedersenCommitment pedersen_switch_commit(const fast::Scalar& value,
                                          const fast::Scalar& blinding,
                                          const fast::Scalar& switch_blind);

} // namespace secp256k1

#endif // SECP256K1_PEDERSEN_HPP
