#ifndef SECP256K1_HKDF_HPP
#define SECP256K1_HKDF_HPP
#pragma once

// ============================================================================
// HKDF-SHA256 (RFC 5869) + HMAC-SHA256
// ============================================================================
// Used by BIP-324 for key derivation from the ECDH shared secret.
// Built on the existing SHA256 class — zero external dependencies.
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>

namespace secp256k1 {

// -- HMAC-SHA256 --------------------------------------------------------------
std::array<std::uint8_t, 32> hmac_sha256(
    const std::uint8_t* key, std::size_t key_len,
    const std::uint8_t* data, std::size_t data_len) noexcept;

// -- HKDF-SHA256 Extract (RFC 5869 Section 2.2) ------------------------------
// PRK = HMAC-SHA256(salt, IKM)
// If salt is nullptr, uses a 32-byte zero string.
std::array<std::uint8_t, 32> hkdf_sha256_extract(
    const std::uint8_t* salt, std::size_t salt_len,
    const std::uint8_t* ikm, std::size_t ikm_len) noexcept;

// -- HKDF-SHA256 Expand (RFC 5869 Section 2.3) -------------------------------
// OKM = T(1) || T(2) || ... truncated to out_len bytes
// T(i) = HMAC-SHA256(PRK, T(i-1) || info || i)
// out_len must be <= 255 * 32 = 8160 bytes.
// Returns false if out_len is too large.
bool hkdf_sha256_expand(
    const std::uint8_t prk[32],
    const std::uint8_t* info, std::size_t info_len,
    std::uint8_t* out, std::size_t out_len) noexcept;

} // namespace secp256k1

#endif // SECP256K1_HKDF_HPP
