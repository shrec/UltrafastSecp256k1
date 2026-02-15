#ifndef SECP256K1_FROST_HPP
#define SECP256K1_FROST_HPP
#pragma once

// ============================================================================
// FROST Threshold Signatures for secp256k1
// ============================================================================
// Flexible Round-Optimized Schnorr Threshold signatures (FROST)
// t-of-n distributed key generation and signing.
//
// Protocol overview:
//   1. KeyGen (DKG): Each participant generates a polynomial commitment
//      and secret shares. Participants exchange commitments and shares
//      to derive the group public key and individual signing shares.
//   2. Signing Round 1: Each signer generates nonce commitments (D_i, E_i)
//   3. Signing Round 2: Each signer produces a partial signature s_i
//   4. Aggregation: Coordinator combines partial signatures into final sig
//
// The final signature is a standard BIP-340 Schnorr signature,
// verifiable with schnorr_verify().
//
// Reference: https://eprint.iacr.org/2020/852
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <vector>
#include <utility>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"

namespace secp256k1 {

// ── Types ────────────────────────────────────────────────────────────────────

// Participant index (1-based, as per Shamir's secret sharing)
using ParticipantId = std::uint32_t;

// Secret share for a participant
struct FrostShare {
    ParticipantId from;       // Who generated this share
    ParticipantId id;         // Participant index (target, 1..n)
    fast::Scalar value;       // The share value f_from(id)
};

// Public polynomial commitment (Feldman VSS)
struct FrostCommitment {
    ParticipantId from;                  // Who generated this
    std::vector<fast::Point> coeffs;     // A_0 = s_i*G, A_1 = a_1*G, ...
};

// A participant's long-lived key material after DKG
struct FrostKeyPackage {
    ParticipantId id;                    // This participant's index
    fast::Scalar signing_share;          // Private signing share
    fast::Point verification_share;      // Public verification share (Y_i)
    fast::Point group_public_key;        // Group public key (Y)
    std::uint32_t threshold;             // t (minimum signers)
    std::uint32_t num_participants;      // n (total participants)
};

// Nonce pair for a signing round
struct FrostNonce {
    fast::Scalar hiding_nonce;    // d_i (secret)
    fast::Scalar binding_nonce;   // e_i (secret)
};

// Public nonce commitment
struct FrostNonceCommitment {
    ParticipantId id;
    fast::Point hiding_point;     // D_i = d_i * G
    fast::Point binding_point;    // E_i = e_i * G
};

// Partial signature from one signer
struct FrostPartialSig {
    ParticipantId id;
    fast::Scalar z_i;             // Partial signature scalar
};

// ── DKG (Distributed Key Generation) ─────────────────────────────────────────

// Round 1: Generate secret polynomial and commitment
// Returns: (commitment to broadcast, shares to send privately)
// participant_id: this participant's 1-based index
// threshold: minimum signers required (t)
// num_participants: total participants (n)
std::pair<FrostCommitment, std::vector<FrostShare>>
frost_keygen_begin(ParticipantId participant_id,
                   std::uint32_t threshold,
                   std::uint32_t num_participants,
                   const std::array<std::uint8_t, 32>& secret_seed);

// Round 2: Verify received commitments and shares, compute signing key
// commitments: all participants' polynomial commitments (including own)
// received_shares: shares received from other participants (one per participant)
// own_share: this participant's share of their own polynomial
// Returns: {key_package, success}
std::pair<FrostKeyPackage, bool>
frost_keygen_finalize(ParticipantId participant_id,
                      const std::vector<FrostCommitment>& commitments,
                      const std::vector<FrostShare>& received_shares,
                      std::uint32_t threshold,
                      std::uint32_t num_participants);

// ── Signing ──────────────────────────────────────────────────────────────────

// Generate nonces for a signing round
// Returns: (secret nonce pair, public commitment)
std::pair<FrostNonce, FrostNonceCommitment>
frost_sign_nonce_gen(ParticipantId participant_id,
                     const std::array<std::uint8_t, 32>& nonce_seed);

// Compute partial signature
// key_pkg: this signer's key package
// nonce: this signer's secret nonce
// msg: 32-byte message to sign
// nonce_commitments: all participating signers' nonce commitments
FrostPartialSig
frost_sign(const FrostKeyPackage& key_pkg,
           const FrostNonce& nonce,
           const std::array<std::uint8_t, 32>& msg,
           const std::vector<FrostNonceCommitment>& nonce_commitments);

// Verify a partial signature (optional, for robustness)
bool frost_verify_partial(const FrostPartialSig& partial_sig,
                          const FrostNonceCommitment& signer_commitment,
                          const fast::Point& verification_share,
                          const std::array<std::uint8_t, 32>& msg,
                          const std::vector<FrostNonceCommitment>& nonce_commitments,
                          const fast::Point& group_public_key);

// Aggregate partial signatures into final Schnorr signature
// The result is a standard BIP-340 Schnorr signature
SchnorrSignature
frost_aggregate(const std::vector<FrostPartialSig>& partial_sigs,
                const std::vector<FrostNonceCommitment>& nonce_commitments,
                const fast::Point& group_public_key,
                const std::array<std::uint8_t, 32>& msg);

// ── Lagrange Coefficients ────────────────────────────────────────────────────

// Compute Lagrange coefficient λ_i for participant i in signer set S
// λ_i = Π_{j∈S, j≠i} (j / (j - i)) mod n
fast::Scalar frost_lagrange_coefficient(ParticipantId i,
                                         const std::vector<ParticipantId>& signer_ids);

} // namespace secp256k1

#endif // SECP256K1_FROST_HPP
