#ifndef SECP256K1_CT_SIGN_HPP
#define SECP256K1_CT_SIGN_HPP

// ============================================================================
// Constant-Time Signing & Key Generation
// ============================================================================
// Drop-in CT replacements for secp256k1::ecdsa_sign() and schnorr_sign().
// These use ct::generator_mul() (data-independent execution trace) instead of
// the fast variable-time scalar_mul on the generator.
//
// For production signing where the private key/nonce must remain secret,
// ALWAYS use these functions instead of the fast:: variants.
//
// Usage:
//   #include <secp256k1/ct/sign.hpp>
//   auto sig = secp256k1::ct::ecdsa_sign(msg_hash, privkey);
//   auto schnorr_sig = secp256k1::ct::schnorr_sign(keypair, msg, aux);
//
// Compile with -DSECP256K1_REQUIRE_CT=1 to deprecate non-CT sign functions.
// ============================================================================

#include <array>
#include <cstdint>
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/private_key.hpp"
#include "secp256k1/ct/point.hpp"

namespace secp256k1::ct {

// -- CT ECDSA Sign ------------------------------------------------------------
// Equivalent to secp256k1::ecdsa_sign() but uses ct::generator_mul() for R=k*G.
// RFC 6979 deterministic nonce. Returns normalized (low-S) signature.
ECDSASignature ecdsa_sign(const std::array<std::uint8_t, 32>& msg_hash,
                          const fast::Scalar& private_key);

// -- CT ECDSA Sign + Verify (fault attack countermeasure) ---------------------
// Signs and then verifies (FIPS 186-4 fault countermeasure).
ECDSASignature ecdsa_sign_verified(const std::array<std::uint8_t, 32>& msg_hash,
                                   const fast::Scalar& private_key);

// -- CT ECDSA Sign (hedged, with extra entropy) --------------------------------
// RFC 6979 Section 3.6: aux_rand mixed into HMAC-DRBG. CT generator_mul for R.
ECDSASignature ecdsa_sign_hedged(const std::array<std::uint8_t, 32>& msg_hash,
                                  const fast::Scalar& private_key,
                                  const std::array<std::uint8_t, 32>& aux_rand);

// -- CT ECDSA Sign Hedged + Verify (fault attack countermeasure) ---------------
ECDSASignature ecdsa_sign_hedged_verified(const std::array<std::uint8_t, 32>& msg_hash,
                                          const fast::Scalar& private_key,
                                          const std::array<std::uint8_t, 32>& aux_rand);

// -- CT ECDSA Sign with Recovery ID -------------------------------------------
// Like ecdsa_sign() but also returns the recovery ID (0-3) needed for public key
// recovery. Uses ct::generator_mul() for R=k*G and ct::scalar_inverse() for
// k^{-1}: the private key and nonce remain constant-time throughout.
//
// Replaces the variable-time ::ecdsa_sign_recoverable() in all signing contexts
// where a private key is involved (bitcoin_sign_message, Ethereum personal_sign,
// and any Sparrow Wallet / ECIES integration using this library).
//
// Recovery ID extraction reads R.y parity from FieldElement::limbs()[0]&1 and
// checks R.x overflow with a byte comparison -- neither branches on secret data.
RecoverableSignature ecdsa_sign_recoverable(
    const std::array<std::uint8_t, 32>& msg_hash,
    const fast::Scalar& private_key);

// PrivateKey overload.
inline RecoverableSignature ecdsa_sign_recoverable(
    const std::array<std::uint8_t, 32>& msg_hash,
    const PrivateKey& private_key) {
    return ecdsa_sign_recoverable(msg_hash, private_key.scalar());
}

// -- CT ECDSA Sign (PrivateKey overload) --------------------------------------
// Preferred overload: accepts strong-typed PrivateKey for compile-time safety.
inline ECDSASignature ecdsa_sign(const std::array<std::uint8_t, 32>& msg_hash,
                                  const PrivateKey& private_key) {
    return ecdsa_sign(msg_hash, private_key.scalar());
}

inline ECDSASignature ecdsa_sign_verified(const std::array<std::uint8_t, 32>& msg_hash,
                                          const PrivateKey& private_key) {
    return ecdsa_sign_verified(msg_hash, private_key.scalar());
}

// -- CT ECDSA Sign Hedged (PrivateKey overload) --------------------------------
inline ECDSASignature ecdsa_sign_hedged(const std::array<std::uint8_t, 32>& msg_hash,
                                         const PrivateKey& private_key,
                                         const std::array<std::uint8_t, 32>& aux_rand) {
    return ecdsa_sign_hedged(msg_hash, private_key.scalar(), aux_rand);
}

inline ECDSASignature ecdsa_sign_hedged_verified(const std::array<std::uint8_t, 32>& msg_hash,
                                                 const PrivateKey& private_key,
                                                 const std::array<std::uint8_t, 32>& aux_rand) {
    return ecdsa_sign_hedged_verified(msg_hash, private_key.scalar(), aux_rand);
}

// -- CT Schnorr Pubkey --------------------------------------------------------
// X-only public key derivation using ct::generator_mul().
std::array<std::uint8_t, 32> schnorr_pubkey(const fast::Scalar& private_key);

// PrivateKey overload.
inline std::array<std::uint8_t, 32> schnorr_pubkey(const PrivateKey& pk) {
    return schnorr_pubkey(pk.scalar());
}

// -- CT Schnorr Keypair Create ------------------------------------------------
// Creates a BIP-340 keypair using ct::generator_mul().
SchnorrKeypair schnorr_keypair_create(const fast::Scalar& private_key);

// PrivateKey overload.
inline SchnorrKeypair schnorr_keypair_create(const PrivateKey& pk) {
    return schnorr_keypair_create(pk.scalar());
}

// -- CT Schnorr Sign (keypair variant) ----------------------------------------
// BIP-340 signing using ct::generator_mul() for the nonce point R = k'*G.
//
// aux_rand: MUST be 32 bytes of fresh cryptographic randomness (e.g. from
//   OS CSPRNG). Provides synthetic nonce hedging per BIP-340. All-zeros is
//   safe against nonce reuse but not against fault injection.
//   See secp256k1/schnorr.hpp for full entropy contract.
SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<std::uint8_t, 32>& msg,
                              const std::array<std::uint8_t, 32>& aux_rand);

// -- CT Schnorr Sign + Verify (fault attack countermeasure) --------------------
// Signs and then verifies (FIPS 186-4 fault countermeasure).
SchnorrSignature schnorr_sign_verified(const SchnorrKeypair& kp,
                                       const std::array<std::uint8_t, 32>& msg,
                                       const std::array<std::uint8_t, 32>& aux_rand);

} // namespace secp256k1::ct

// ============================================================================
// FAST guardrail: when SECP256K1_REQUIRE_CT is defined, mark the non-CT
// sign functions as deprecated so callers get a compile-time warning.
// ============================================================================
#if defined(SECP256K1_REQUIRE_CT) && SECP256K1_REQUIRE_CT

// GCC/Clang [[deprecated]] on existing declarations (re-declaration is valid)
namespace secp256k1 {

[[deprecated("Non-CT signing: use secp256k1::ct::ecdsa_sign() for production. "
             "Define SECP256K1_ALLOW_FAST_SIGN to suppress.")]]
ECDSASignature ecdsa_sign(const std::array<std::uint8_t, 32>& msg_hash,
                          const fast::Scalar& private_key);

[[deprecated("Non-CT signing: use secp256k1::ct::schnorr_sign() for production. "
             "Define SECP256K1_ALLOW_FAST_SIGN to suppress.")]]
SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<std::uint8_t, 32>& msg,
                              const std::array<std::uint8_t, 32>& aux_rand);

[[deprecated("Non-CT key gen: use secp256k1::ct::schnorr_keypair_create().")]]
SchnorrKeypair schnorr_keypair_create(const fast::Scalar& private_key);

[[deprecated("Non-CT key gen: use secp256k1::ct::schnorr_pubkey().")]]
std::array<std::uint8_t, 32> schnorr_pubkey(const fast::Scalar& private_key);

} // namespace secp256k1

#endif // SECP256K1_REQUIRE_CT

#endif // SECP256K1_CT_SIGN_HPP
