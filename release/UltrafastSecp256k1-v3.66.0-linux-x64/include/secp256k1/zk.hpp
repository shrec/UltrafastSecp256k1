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


// ============================================================================
// 4. ECDSA-in-SNARK Foreign-Field Witness  (eprint 2025/695)
// ============================================================================
// Generates all intermediate values needed by a PLONK circuit prover to
// verify an ECDSA signature over secp256k1 with foreign-field arithmetic.
//
// Background (eprint 2025/695, Ambrona, Firsov, Querejeta-Azurmendi):
//   secp256k1 p and n are both larger than common SNARK scalar fields (BN254 r,
//   BLS12-381 r).  A PLONK circuit must therefore encode every secp256k1 field
//   element as multiple "limbs" — the foreign-field representation.  Using 5×52-
//   bit limbs with tight range bounds reduces the gate count for one ECDSA
//   verification from ~50 000 constraints to ~5 000 (≈10× improvement).
//
//   This function is the **host-side witness generator**: it computes every
//   intermediate value the PLONK prover needs as private inputs, and returns
//   them in both canonical 32-byte encoding AND in 5×52-bit limb form so that
//   the caller can feed them directly into a PLONK framework (Halo2, Plonky3,
//   Circom, etc.) without any additional decomposition step.

// ── Limb container ──────────────────────────────────────────────────────────
// Per-value foreign-field representation for PLONK circuits.
// 5 limbs × 52 bits = 260 bits total, covering the 256-bit secp256k1 prime.
// Top limb (limbs[4]) uses at most 48 bits when the value is < p (or < n).
// Each limb fits in uint64_t without overflow — no masking needed at capture.
struct ForeignFieldLimbs {
    std::uint64_t limbs[5];  // little-endian 52-bit limbs
};

// ── Witness struct ───────────────────────────────────────────────────────────
// Complete PLONK prover witness for one secp256k1 ECDSA verification.
// All scalar/field values are provided in both canonical form (Scalar/Point)
// AND as 5×52-bit limbs ready for PLONK gate wiring.
//
// ECDSA verify steps (Fp = secp256k1 field, Fr = secp256k1 scalar field):
//   1. s_inv   = sig_s^{-1}              mod n   (Fr)
//   2. u1      = msg_hash * s_inv         mod n   (Fr)
//   3. u2      = sig_r * s_inv            mod n   (Fr)
//   4. R       = u1*G + u2*pubkey                 (Fp^2)
//   5. valid   = (R ≠ ∞) AND (R.x mod n == sig_r)
struct EcdsaSnarkWitness {
    // ── public inputs ──────────────────────────────────────────────────────
    ForeignFieldLimbs msg;       // message hash mod n         (Fr)
    ForeignFieldLimbs sig_r;     // signature r                (Fr)
    ForeignFieldLimbs sig_s;     // signature s                (Fr)
    ForeignFieldLimbs pub_x;     // public key P.x             (Fp)
    ForeignFieldLimbs pub_y;     // public key P.y             (Fp)

    // ── private witness (circuit signals) ─────────────────────────────────
    ForeignFieldLimbs s_inv;             // s^{-1} mod n       (Fr)
    ForeignFieldLimbs u1;                // e * s^{-1} mod n   (Fr)
    ForeignFieldLimbs u2;                // r * s^{-1} mod n   (Fr)
    ForeignFieldLimbs result_x;          // R.x                (Fp)
    ForeignFieldLimbs result_y;          // R.y                (Fp)
    ForeignFieldLimbs result_x_mod_n;    // R.x mod n          (Fr)

    // ── canonical byte encodings (big-endian) ─────────────────────────────
    std::array<std::uint8_t, 32> bytes_s_inv;
    std::array<std::uint8_t, 32> bytes_u1;
    std::array<std::uint8_t, 32> bytes_u2;
    std::array<std::uint8_t, 32> bytes_result_x;
    std::array<std::uint8_t, 32> bytes_result_y;
    std::array<std::uint8_t, 32> bytes_result_x_mod_n;

    // ── verdict ───────────────────────────────────────────────────────────
    bool valid;  // true iff the ECDSA signature is valid
};

// Compute the ECDSA-in-SNARK foreign-field witness.
//
// msg_hash : 32-byte big-endian message hash (hash of the signed message)
// pubkey   : uncompressed public key point P = d*G
// sig_r    : ECDSA signature r-scalar (must be in [1, n-1])
// sig_s    : ECDSA signature s-scalar (must be in [1, n-1]; accepts high-S)
//
// Returns a fully populated EcdsaSnarkWitness.
// If the signature is invalid, `valid` is false but witness values are still
// populated (to allow the prover to build a failing-path proof if needed).
EcdsaSnarkWitness ecdsa_snark_witness(
    const std::array<std::uint8_t, 32>& msg_hash,
    const fast::Point& pubkey,
    const fast::Scalar& sig_r,
    const fast::Scalar& sig_s);


// ============================================================================
// 5. BIP340 Schnorr-in-SNARK Foreign-Field Witness
// ============================================================================
// Generates all intermediate values needed by a PLONK circuit prover to
// verify a BIP-340 Schnorr signature over secp256k1.
//
// BIP-340 verification in a circuit:
//   1. Lift R (even Y) from 32-byte R.x
//   2. Lift P (even Y) from 32-byte x-only pubkey
//   3. e = H("BIP0340/challenge" || R.x || P.x || msg) mod n
//   4. R' = s*G - e*P
//   5. valid = (R' == R) => equivalently, s*G == R + e*P
//
// Schnorr is simpler than ECDSA for circuits: no modular inverse, only
// one multi-scalar multiplication, and the challenge is a plain hash.
//
// Like EcdsaSnarkWitness, all values are returned in both canonical
// 32-byte form and 5×52-bit ForeignFieldLimbs for PLONK/Halo2/Circom.

struct SchnorrSnarkWitness {
    // ── public inputs ──────────────────────────────────────────────────
    ForeignFieldLimbs msg;       // message (32 bytes)            (Fr)
    ForeignFieldLimbs sig_r;     // R.x (nonce x-coordinate)     (Fp)
    ForeignFieldLimbs sig_s;     // s scalar                     (Fr)
    ForeignFieldLimbs pub_x;     // P.x (x-only pubkey)          (Fp)

    // ── private witness (circuit signals) ─────────────────────────────
    ForeignFieldLimbs r_y;       // R.y (lifted, even Y)         (Fp)
    ForeignFieldLimbs pub_y;     // P.y (lifted, even Y)         (Fp)
    ForeignFieldLimbs e;         // challenge scalar             (Fr)

    // ── canonical byte encodings (big-endian) ─────────────────────────
    std::array<std::uint8_t, 32> bytes_r_y;
    std::array<std::uint8_t, 32> bytes_pub_y;
    std::array<std::uint8_t, 32> bytes_e;

    // ── verdict ───────────────────────────────────────────────────────
    bool valid;  // true iff the BIP-340 signature is valid
};

// Compute the BIP340 Schnorr-in-SNARK foreign-field witness.
//
// msg       : 32-byte message (per BIP-340, this is the message, not a hash)
// pubkey_x  : 32-byte x-only public key (big-endian)
// sig_r     : 32-byte R.x from signature (big-endian)
// sig_s     : s scalar from signature
//
// Returns a fully populated SchnorrSnarkWitness.
// If the signature is invalid, `valid` is false but witness values are still
// populated (to allow the prover to build a failing-path proof if needed).
SchnorrSnarkWitness schnorr_snark_witness(
    const std::array<std::uint8_t, 32>& msg,
    const std::array<std::uint8_t, 32>& pubkey_x,
    const std::array<std::uint8_t, 32>& sig_r,
    const fast::Scalar& sig_s);

} // namespace zk
} // namespace secp256k1

#endif // SECP256K1_ZK_HPP
