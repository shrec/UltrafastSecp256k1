#pragma once
// ============================================================================
// ECDH -- Elliptic Curve Diffie-Hellman (CUDA device)
// ============================================================================
// Computes shared secret from private key + peer public key.
// Three variants:
//   - ecdh_compute:       SHA-256(0x02|x) standard compressed hash
//   - ecdh_compute_xonly: SHA-256(x) x-only hash
//   - ecdh_compute_raw:   raw 32-byte x-coordinate (no hash)
//
// 64-bit limb mode only.
// ============================================================================

#include "ecdsa.cuh"   // for SHA256Ctx, sha256_*, scalar_from_bytes, field_to_bytes

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// -- ECDH: compute raw x-coordinate ------------------------------------------
// shared_secret = x-coordinate of sk * PK (32 bytes, big-endian)
// Returns false if result is point at infinity.

__device__ inline bool ecdh_compute_raw(
    const Scalar* private_key,
    const JacobianPoint* peer_pubkey,
    uint8_t out[32])
{
    JacobianPoint shared;
    scalar_mul(peer_pubkey, private_key, &shared);
    if (shared.infinity) return false;

    // Convert to affine x-coordinate
    FieldElement z_inv, z_inv2, x_aff;
    field_inv(&shared.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&shared.x, &z_inv2, &x_aff);

    field_to_bytes(&x_aff, out);
    return true;
}

// -- ECDH: compute x-only hash -----------------------------------------------
// shared_secret = SHA-256(x) where x = x-coordinate of sk * PK.

__device__ inline bool ecdh_compute_xonly(
    const Scalar* private_key,
    const JacobianPoint* peer_pubkey,
    uint8_t out[32])
{
    uint8_t x_bytes[32];
    if (!ecdh_compute_raw(private_key, peer_pubkey, x_bytes))
        return false;

    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, x_bytes, 32);
    sha256_final(&ctx, out);
    return true;
}

// -- ECDH: compute standard compressed hash ----------------------------------
// shared_secret = SHA-256(compressed_point) matching CPU ecdh_compute.
// Prefix is 0x02 (even y) or 0x03 (odd y).

__device__ inline bool ecdh_compute(
    const Scalar* private_key,
    const JacobianPoint* peer_pubkey,
    uint8_t out[32])
{
    JacobianPoint shared;
    scalar_mul(peer_pubkey, private_key, &shared);
    if (shared.infinity) return false;

    // Convert to affine x and y
    FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
    field_inv(&shared.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&shared.x, &z_inv2, &x_aff);
    field_mul(&shared.y, &z_inv3, &y_aff);

    uint8_t x_bytes[32];
    field_to_bytes(&x_aff, x_bytes);

    // Determine y parity for prefix
    uint8_t y_bytes[32];
    field_to_bytes(&y_aff, y_bytes);
    uint8_t prefix = (y_bytes[31] & 1) ? 0x03 : 0x02;

    // SHA-256(prefix || x)
    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, &prefix, 1);
    sha256_update(&ctx, x_bytes, 32);
    sha256_final(&ctx, out);
    return true;
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
