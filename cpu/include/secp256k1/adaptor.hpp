#ifndef SECP256K1_ADAPTOR_HPP
#define SECP256K1_ADAPTOR_HPP
#pragma once

// ============================================================================
// Adaptor Signatures for secp256k1
// ============================================================================
// Adaptor signatures enable atomic protocols like:
//   - Atomic swaps (cross-chain)
//   - Discreet Log Contracts (DLCs)
//   - Payment channel protocols
//
// Protocol:
//   1. Signer creates pre-signature σ̃ w.r.t. adaptor point T = t*G
//   2. Verifier checks pre-signature validity against T
//   3. Once signer learns secret t, they adapt σ̃ → valid signature σ
//   4. Verifier extracts t from (σ̃, σ)
//
// Works with both ECDSA and Schnorr signatures.
// ============================================================================

#include <array>
#include <cstdint>
#include <utility>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdsa.hpp"

namespace secp256k1 {

// ── Schnorr Adaptor Signatures ───────────────────────────────────────────────

// Pre-signature (adaptor signature) for Schnorr
struct SchnorrAdaptorSig {
    fast::Point R_hat;    // R̂ = k*G (before adapting with T)
    fast::Scalar s_hat;   // ŝ = k - e*x (partial, needs adapting)
    bool needs_negation;  // Whether R̂+T has even y (BIP-340)
};

// Create Schnorr adaptor pre-signature
// Signs message but locks the result: needs adaptor secret t to complete.
// adaptor_point: T = t*G (public adaptor point)
// private_key: signer's private key
// msg: 32-byte message
// aux_rand: 32 bytes auxiliary randomness
SchnorrAdaptorSig
schnorr_adaptor_sign(const fast::Scalar& private_key,
                     const std::array<std::uint8_t, 32>& msg,
                     const fast::Point& adaptor_point,
                     const std::array<std::uint8_t, 32>& aux_rand);

// Verify a Schnorr adaptor pre-signature
// Checks: ŝ*G == R̂ - e*P (where e = H(R̂+T, P, m))
bool schnorr_adaptor_verify(const SchnorrAdaptorSig& pre_sig,
                            const std::array<std::uint8_t, 32>& pubkey_x,
                            const std::array<std::uint8_t, 32>& msg,
                            const fast::Point& adaptor_point);

// Adapt pre-signature with secret t to produce valid Schnorr signature
// σ = (R̂+T, ŝ+t)
SchnorrSignature
schnorr_adaptor_adapt(const SchnorrAdaptorSig& pre_sig,
                      const fast::Scalar& adaptor_secret);

// Extract adaptor secret t from pre-signature and completed signature
// t = s - ŝ (mod n)
std::pair<fast::Scalar, bool>
schnorr_adaptor_extract(const SchnorrAdaptorSig& pre_sig,
                        const SchnorrSignature& sig);

// ── ECDSA Adaptor Signatures ─────────────────────────────────────────────────

// Pre-signature for ECDSA adaptor
struct ECDSAAdaptorSig {
    fast::Point R_hat;    // R̂ = k*G
    fast::Scalar s_hat;   // Encrypted signature scalar
    fast::Scalar r;       // r = x-coord of (R̂ + T)
};

// Create ECDSA adaptor pre-signature
ECDSAAdaptorSig
ecdsa_adaptor_sign(const fast::Scalar& private_key,
                   const std::array<std::uint8_t, 32>& msg_hash,
                   const fast::Point& adaptor_point);

// Verify ECDSA adaptor pre-signature
bool ecdsa_adaptor_verify(const ECDSAAdaptorSig& pre_sig,
                          const fast::Point& public_key,
                          const std::array<std::uint8_t, 32>& msg_hash,
                          const fast::Point& adaptor_point);

// Adapt ECDSA pre-signature to valid signature with adaptor secret
ECDSASignature
ecdsa_adaptor_adapt(const ECDSAAdaptorSig& pre_sig,
                    const fast::Scalar& adaptor_secret);

// Extract adaptor secret from ECDSA pre-sig and completed sig
std::pair<fast::Scalar, bool>
ecdsa_adaptor_extract(const ECDSAAdaptorSig& pre_sig,
                      const ECDSASignature& sig);

} // namespace secp256k1

#endif // SECP256K1_ADAPTOR_HPP
