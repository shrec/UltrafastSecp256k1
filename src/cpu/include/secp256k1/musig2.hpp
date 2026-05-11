#ifndef SECP256K1_MUSIG2_HPP
#define SECP256K1_MUSIG2_HPP
#pragma once

// ============================================================================
// MuSig2: Two-Round Multi-Signature Scheme for secp256k1
// ============================================================================
// Implements MuSig2 (BIP-327) -- a Schnorr-based multi-signature protocol
// where N signers produce a single signature verifiable against an
// aggregated public key.
//
// Protocol:
//   1. Key Aggregation: Q = KeyAgg(pk_1, ..., pk_n)
//   2. Nonce Generation: Each signer generates nonce pair (R_1, R_2)
//   3. Nonce Aggregation: Combine all nonces via deterministic coefficient
//   4. Partial Signing: Each signer produces partial signature s_i
//   5. Signature Aggregation: s = sum(s_i), final sig = (R, s)
//
// The final signature is a standard BIP-340 Schnorr signature.
//
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0327.mediawiki
// ============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1 {

// -- Key Aggregation Context --------------------------------------------------

struct MuSig2KeyAggCtx {
    fast::Point Q;                              // Aggregated public key
    std::array<std::uint8_t, 32> Q_x;          // X-only aggregated pubkey
    std::vector<fast::Scalar> key_coefficients; // a_i for each signer
    bool Q_negated;                              // Whether Q was negated for even-Y
    // Individual compressed pubkeys (33 bytes each), populated by musig2_key_agg.
    // Empty when the context was deserialized from the ABI blob (KEYAGG_LEN does not
    // carry individual pubkeys). When non-empty, musig2_partial_sign validates that
    // the caller's secret_key corresponds to the claimed signer_index (Rule 13).
    // Full ABI-layer fix scheduled for v2 (MED-3 / KEYAGG_LEN expansion).
    std::vector<std::array<std::uint8_t, 33>> individual_pubkeys;
};

// Aggregate public keys (KeyAgg from BIP-327).
// pubkeys: array of compressed public keys (33 bytes each, 02/03 prefix)
// Returns aggregation context with combined key and coefficients.
// If ANY pubkey is invalid (x >= p or not on curve), returns ctx with Q = infinity.
MuSig2KeyAggCtx musig2_key_agg(const std::vector<std::array<std::uint8_t, 33>>& pubkeys);

// -- Nonce --------------------------------------------------------------------

struct MuSig2SecNonce {
    fast::Scalar k1;  // First secret nonce
    fast::Scalar k2;  // Second secret nonce
};

struct MuSig2PubNonce {
    std::array<std::uint8_t, 33> R1;  // First public nonce (compressed)
    std::array<std::uint8_t, 33> R2;  // Second public nonce (compressed)

    // Serialize to 66 bytes
    std::array<std::uint8_t, 66> serialize() const;
    static MuSig2PubNonce deserialize(const std::array<std::uint8_t, 66>& data);
};

// Generate a nonce pair deterministically.
// secret_key: signer's private key
// pub_key: signer's public key (X-only)
// agg_pub_key: aggregated public key (X-only)
// msg: 32-byte message
// extra_input: optional extra randomness (32 bytes, or nullptr)
//
// Returns {secret_nonce, public_nonce}
std::pair<MuSig2SecNonce, MuSig2PubNonce> musig2_nonce_gen(
    const fast::Scalar& secret_key,
    const std::array<std::uint8_t, 32>& pub_key,
    const std::array<std::uint8_t, 32>& agg_pub_key,
    const std::array<std::uint8_t, 32>& msg,
    const std::uint8_t* extra_input = nullptr);

// -- Nonce Aggregation --------------------------------------------------------

struct MuSig2AggNonce {
    fast::Point R1;  // Aggregated first nonce point
    fast::Point R2;  // Aggregated second nonce point
};

// Aggregate all signers' public nonces.
MuSig2AggNonce musig2_nonce_agg(const std::vector<MuSig2PubNonce>& pub_nonces);

// -- Session (Signing) --------------------------------------------------------

struct MuSig2Session {
    fast::Point R;                            // Final nonce point
    fast::Scalar b;                           // Nonce coefficient
    fast::Scalar e;                           // Challenge
    bool R_negated;                            // Whether R was negated
};

// Start a signing session: compute the effective R, b, and challenge e.
MuSig2Session musig2_start_sign_session(
    const MuSig2AggNonce& agg_nonce,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const std::array<std::uint8_t, 32>& msg);

// -- Partial Signing ----------------------------------------------------------

// Produce a partial signature.
// s_i = k1 + b*k2 + e * a_i * d_i  (mod n)
// where d_i is adjusted for Q/R negation.
// sec_nonce is CONSUMED — both k1 and k2 are zeroized before return
// to enforce single-use (M-03 nonce-reuse prevention).
fast::Scalar musig2_partial_sign(
    MuSig2SecNonce& sec_nonce,
    const fast::Scalar& secret_key,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const MuSig2Session& session,
    std::size_t signer_index);

// Verify a partial signature (optional, for honest-signer detection).
bool musig2_partial_verify(
    const fast::Scalar& partial_sig,
    const MuSig2PubNonce& pub_nonce,
    const std::array<std::uint8_t, 32>& pubkey,
    const MuSig2KeyAggCtx& key_agg_ctx,
    const MuSig2Session& session,
    std::size_t signer_index);

// -- Signature Aggregation ----------------------------------------------------

// Aggregate partial signatures into a final BIP-340 Schnorr signature.
// Returns {r (32 bytes), s (32 bytes)} = standard 64-byte Schnorr sig.
std::array<std::uint8_t, 64> musig2_partial_sig_agg(
    const std::vector<fast::Scalar>& partial_sigs,
    const MuSig2Session& session);

} // namespace secp256k1

#endif // SECP256K1_MUSIG2_HPP
