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
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ct/point.hpp"

namespace secp256k1::ct {

// -- CT ECDSA Sign ------------------------------------------------------------
// Equivalent to secp256k1::ecdsa_sign() but uses ct::generator_mul() for R=k*G.
// RFC 6979 deterministic nonce. Returns normalized (low-S) signature.
ECDSASignature ecdsa_sign(const std::array<std::uint8_t, 32>& msg_hash,
                          const fast::Scalar& private_key);

// -- CT Schnorr Pubkey --------------------------------------------------------
// X-only public key derivation using ct::generator_mul().
std::array<std::uint8_t, 32> schnorr_pubkey(const fast::Scalar& private_key);

// -- CT Schnorr Keypair Create ------------------------------------------------
// Creates a BIP-340 keypair using ct::generator_mul().
SchnorrKeypair schnorr_keypair_create(const fast::Scalar& private_key);

// -- CT Schnorr Sign (keypair variant) ----------------------------------------
// BIP-340 signing using ct::generator_mul() for the nonce point R = k'*G.
SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
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
