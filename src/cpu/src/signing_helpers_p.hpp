// ============================================================================
// Private: Signing Helper Utilities
// ============================================================================
// Shared between ecdsa.cpp and recovery.cpp to avoid duplicate static function
// definitions that cause redefinition errors under SECP256K1_UNITY_BUILD.
//
// NOT part of the public API. Do not include from headers.
// ============================================================================

#pragma once
#ifndef SECP256K1_SIGNING_HELPERS_P_HPP
#define SECP256K1_SIGNING_HELPERS_P_HPP

#include "secp256k1/ct/point.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

// CT generator multiply for signing paths: uses ct::generator_mul_blinded
// so that neither k nor privkey timing is observable.
// Used by ecdsa_sign*, ecdsa_sign_recoverable* and any other signing path
// that must remain constant-time on the secret scalar.
[[maybe_unused]] inline fast::Point signing_generator_mul(const fast::Scalar& scalar) {
    return ct::generator_mul_blinded(scalar);
}

} // namespace secp256k1

#endif // SECP256K1_SIGNING_HELPERS_P_HPP
