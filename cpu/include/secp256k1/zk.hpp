#ifndef SECP256K1_ZK_HPP
#define SECP256K1_ZK_HPP
#pragma once

// ============================================================================
// Zero-Knowledge Proof Layer for secp256k1
// ============================================================================
// Implements ZK proof primitives over the secp256k1 curve:
//
//   1. Schnorr Knowledge Proof (sigma protocol)
//      - Non-interactive proof of knowledge of discrete log
//      - Prove: "I know x such that P = x*G" without revealing x
//
//   2. DLEQ Proof (Discrete Log Equality)
//      - Prove: log_G(P) == log_H(Q) without revealing the secret
//      - Used in: VRFs, adaptor signatures, ECDH proofs, atomic swaps
//
//   3. Bulletproof Range Proof
//      - Prove: committed value v in [0, 2^n) without revealing v
//      - Logarithmic proof size via inner product argument
//      - Used in: Confidential Transactions, Mimblewimble, Liquid
//
// Security: All proving operations use CT layer (constant-time).
//           Verification uses fast layer (variable-time, public data).
//
// Fiat-Shamir: All proofs are non-interactive via tagged SHA-256 hashing.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/pedersen.hpp"

namespace secp256k1 {
namespace zk {

// ============================================================================
// 1. Schnorr Knowledge Proof (Sigma Protocol)
// ============================================================================
// Non-interactive proof of knowledge of discrete log.
// Proves: "I know x such that P = x*G" (or P = x*B for arbitrary base B).
//
// Protocol (Fiat-Shamir):
//   Prover: k <- random, R = k*G, e = H("ZK/knowledge" || R || P || msg), s = k + e*x
//   Verifier: s*G == R + e*P
//
// Proof size: 64 bytes (R_compressed[33] + s[32] -> optimized to R.x[32] + s[32])

struct KnowledgeProof {
    std::array<std::uint8_t, 32> rx;  // R.x (x-coordinate of nonce point)
    fast::Scalar s;                    // response scalar

    std::array<std::uint8_t, 64> serialize() const;
    static bool deserialize(const std::uint8_t* data64, KnowledgeProof& out);
};

// Prove knowledge of secret x such that pubkey = x*G
// msg: optional binding message (32 bytes, can be all-zero)
// aux_rand: 32 bytes entropy for nonce hedging
KnowledgeProof knowledge_prove(const fast::Scalar& secret,
                                const fast::Point& pubkey,
                                const std::array<std::uint8_t, 32>& msg,
                                const std::array<std::uint8_t, 32>& aux_rand);

// Verify knowledge proof against public key and message
bool knowledge_verify(const KnowledgeProof& proof,
                      const fast::Point& pubkey,
                      const std::array<std::uint8_t, 32>& msg);

// Prove knowledge of secret x such that point = x*base (arbitrary base)
KnowledgeProof knowledge_prove_base(const fast::Scalar& secret,
                                     const fast::Point& point,
                                     const fast::Point& base,
                                     const std::array<std::uint8_t, 32>& msg,
                                     const std::array<std::uint8_t, 32>& aux_rand);

// Verify knowledge proof against arbitrary base
bool knowledge_verify_base(const KnowledgeProof& proof,
                           const fast::Point& point,
                           const fast::Point& base,
                           const std::array<std::uint8_t, 32>& msg);


// ============================================================================
// 2. DLEQ Proof (Discrete Log Equality)
// ============================================================================
// Proves: log_G(P) == log_H(Q), i.e., P = x*G and Q = x*H for same x.
//
// Protocol (Fiat-Shamir):
//   Prover: k <- random, R1 = k*G, R2 = k*H
//           e = H("ZK/dleq" || G || H || P || Q || R1 || R2)
//           s = k + e*x
//   Verifier: s*G == R1 + e*P  AND  s*H == R2 + e*Q
//
// Used in VRFs, DLEQ-based adaptor signatures, provable ECDH.
// Proof size: 64 bytes (e[32] + s[32])

struct DLEQProof {
    fast::Scalar e;  // challenge
    fast::Scalar s;  // response

    std::array<std::uint8_t, 64> serialize() const;
    static bool deserialize(const std::uint8_t* data64, DLEQProof& out);
};

// Prove that log_G(P) == log_H(Q) where P = secret*G and Q = secret*H
// aux_rand: 32 bytes entropy for nonce hedging
DLEQProof dleq_prove(const fast::Scalar& secret,
                      const fast::Point& G,
                      const fast::Point& H,
                      const fast::Point& P,
                      const fast::Point& Q,
                      const std::array<std::uint8_t, 32>& aux_rand);

// Verify DLEQ proof
bool dleq_verify(const DLEQProof& proof,
                 const fast::Point& G,
                 const fast::Point& H,
                 const fast::Point& P,
                 const fast::Point& Q);


// ============================================================================
// 3. Bulletproof Range Proof
// ============================================================================
// Proves that a Pedersen commitment C = v*H + r*G commits to v in [0, 2^n).
// Based on Bulletproofs (Bunz et al., 2018).
//
// Proof structure:
//   - A, S: vector commitment points (2 group elements)
//   - T1, T2: polynomial commitment points (2 group elements)
//   - tau_x, mu, t_hat: scalar values (3 scalars)
//   - L[], R[]: inner product argument (2*log2(n) group elements)
//   - a, b: final inner product scalars (2 scalars)
//
// For n=64 bits: 2*log2(64) = 12 group elements + 7 scalars = ~620 bytes
// Verification: O(n) multi-exp (can batch across multiple proofs)

static constexpr std::size_t RANGE_PROOF_BITS = 64;
static constexpr std::size_t RANGE_PROOF_LOG2 = 6;  // log2(64)

struct RangeProof {
    // Vector commitments
    fast::Point A;   // commitment to bits vector
    fast::Point S;   // commitment to blinding vectors

    // Polynomial commitments
    fast::Point T1;  // commitment to t_1 coefficient
    fast::Point T2;  // commitment to t_2 coefficient

    // Scalar responses
    fast::Scalar tau_x;   // blinding for polynomial eval
    fast::Scalar mu;      // aggregate blinding
    fast::Scalar t_hat;   // polynomial evaluation at challenge

    // Inner product argument (log2(n) rounds)
    std::array<fast::Point, RANGE_PROOF_LOG2> L;
    std::array<fast::Point, RANGE_PROOF_LOG2> R;

    // Final scalars
    fast::Scalar a;
    fast::Scalar b;
};

// Generate range proof for a Pedersen commitment
// value: the committed value (must be in [0, 2^64))
// blinding: the blinding factor used in the commitment
// commitment: the Pedersen commitment C = value*H + blinding*G
// aux_rand: 32 bytes of entropy
RangeProof range_prove(std::uint64_t value,
                        const fast::Scalar& blinding,
                        const PedersenCommitment& commitment,
                        const std::array<std::uint8_t, 32>& aux_rand);

// Verify range proof for a Pedersen commitment
// Returns true if the proof is valid (committed value is in [0, 2^64))
bool range_verify(const PedersenCommitment& commitment,
                  const RangeProof& proof);


// ============================================================================
// Generator Vectors (for Bulletproofs)
// ============================================================================
// Nothing-up-my-sleeve generators: G_i = H("BP_G" || LE32(i)), H_i = H("BP_H" || LE32(i))
// Cached after first computation.

struct GeneratorVectors {
    std::array<fast::Point, RANGE_PROOF_BITS> G;
    std::array<fast::Point, RANGE_PROOF_BITS> H;
};

const GeneratorVectors& get_generator_vectors();


// ============================================================================
// Batch Operations
// ============================================================================

// Batch-verify multiple range proofs (more efficient than individual verification)
// Returns true only if ALL proofs are valid.
bool batch_range_verify(const PedersenCommitment* commitments,
                        const RangeProof* proofs,
                        std::size_t count);

// Batch-create Pedersen commitments (performance optimization)
// values[count], blindings[count] -> commitments_out[count]
void batch_commit(const fast::Scalar* values,
                  const fast::Scalar* blindings,
                  PedersenCommitment* commitments_out,
                  std::size_t count);

} // namespace zk
} // namespace secp256k1

#endif // SECP256K1_ZK_HPP
