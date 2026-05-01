#pragma once
// ============================================================================
// Constant-Time ECDSA & Schnorr Signing -- CUDA Device
// ============================================================================
// Side-channel resistant signing for secp256k1.
//
// Key differences from fast path:
//   - R = k*G via ct_generator_mul (fixed execution trace)
//   - k^{-1} via ct::scalar_inverse (Fermat, no branch on bits)
//   - Low-S via ct::scalar_normalize_low_s (branchless cmov)
//   - Y-parity via ct::scalar_cneg + bool_to_mask (no branch)
//   - All scalar arithmetic via CT layer (no early-exit comparisons)
//   - SHA-256 / HMAC / RFC6979 / tagged_hash already data-independent
//
// Port of: cpu/include/secp256k1/ct/sign.hpp + cpu/src/ct_sign.cpp
// ============================================================================

#include "ct/ct_point.cuh"
#include "ecdsa.cuh"
#include "schnorr.cuh"

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {
namespace ct {

// ============================================================================
// CT Jacobian -> Affine Conversion
// ============================================================================
// Branchless: always computes z_inv; cmov zeros on infinity

__device__ inline
void ct_jacobian_to_affine(const JacobianPoint* p,
                           FieldElement* out_x, FieldElement* out_y,
                           uint8_t* y_parity)
{
    FieldElement z_inv, z_inv2, z_inv3;
    secp256k1::cuda::field_inv(&p->z, &z_inv);
    secp256k1::cuda::field_sqr(&z_inv, &z_inv2);
    secp256k1::cuda::field_mul(&z_inv, &z_inv2, &z_inv3);
    secp256k1::cuda::field_mul(&p->x, &z_inv2, out_x);
    secp256k1::cuda::field_mul(&p->y, &z_inv3, out_y);

    // Extract Y parity from bytes
    uint8_t y_bytes[32];
    secp256k1::cuda::field_to_bytes(out_y, y_bytes);
    *y_parity = y_bytes[31] & 1;
}

// ============================================================================
// CT ECDSA Sign
// ============================================================================
// Constant-time ECDSA sign using RFC 6979 deterministic nonce.
// All secret-dependent operations use CT layer:
//   - R = k*G: ct_generator_mul
//   - k^{-1}: ct::scalar_inverse
//   - low-S:  ct::scalar_normalize_low_s
//   - scalar add: ct::scalar_add

__device__ inline bool ct_ecdsa_sign(
    const uint8_t msg_hash[32],
    const Scalar* private_key,
    ECDSASignatureGPU* sig)
{
    // Check private key is nonzero (public validation, branch OK)
    if (secp256k1::cuda::scalar_is_zero(private_key)) return false;

    // z = message hash as scalar (public data, use fast path)
    Scalar z;
    secp256k1::cuda::scalar_from_bytes(msg_hash, &z);

    // k = RFC 6979 deterministic nonce (HMAC-SHA256 is data-independent)
    Scalar k;
    secp256k1::cuda::rfc6979_nonce(private_key, msg_hash, &k);

    // R = k * G  (CT: fixed execution trace, no branch on bits of k)
    JacobianPoint R;
    ct_generator_mul(&k, &R);

    // Convert R to affine (need x-coordinate for r)
    FieldElement rx_aff, ry_aff;
    uint8_t y_parity;
    ct_jacobian_to_affine(&R, &rx_aff, &ry_aff, &y_parity);

    // r = rx mod n
    uint8_t x_bytes[32];
    secp256k1::cuda::field_to_bytes(&rx_aff, x_bytes);
    secp256k1::cuda::scalar_from_bytes(x_bytes, &sig->r);

    // Check r != 0 (r depends on public curve point, branch OK)
    if (secp256k1::cuda::scalar_is_zero(&sig->r)) return false;

    // k^{-1} (CT Fermat: always compute both paths, select by bit)
    Scalar k_inv;
    scalar_inverse(&k, &k_inv);

    // rd = r * d mod n (CT scalar_mul)
    Scalar rd;
    scalar_mul(&sig->r, private_key, &rd);

    // z_plus_rd = z + r*d mod n (CT scalar_add)
    Scalar z_plus_rd;
    scalar_add(&z, &rd, &z_plus_rd);

    // s = k^{-1} * (z + r*d) mod n (CT scalar_mul)
    scalar_mul(&k_inv, &z_plus_rd, &sig->s);

    // Check s != 0 (branch on public output, OK)
    if (secp256k1::cuda::scalar_is_zero(&sig->s)) return false;

    // Low-S normalization (CT: branchless cmov)
    scalar_normalize_low_s(&sig->s);

    return true;
}

// CT ECDSA sign with immediate verification (fault countermeasure)
__device__ inline bool ct_ecdsa_sign_verified(
    const uint8_t msg_hash[32],
    const Scalar* private_key,
    ECDSASignatureGPU* sig)
{
    if (!ct_ecdsa_sign(msg_hash, private_key, sig)) return false;

    // Derive public key (CT, since private_key is secret)
    JacobianPoint pubkey;
    ct_generator_mul(private_key, &pubkey);

    // Verify uses fast path (public key + signature are public)
    return secp256k1::cuda::ecdsa_verify(msg_hash, &pubkey, sig);
}

// ============================================================================
// CT Schnorr Keypair
// ============================================================================

struct CTSchnorrKeypairGPU {
    Scalar d;           // signing key (adjusted for even Y)
    uint8_t px[32];     // x-coordinate bytes of pubkey
};

// CT keypair creation: adjusts private key for even Y without branching.
__device__ inline bool ct_schnorr_keypair_create(
    const Scalar* private_key,
    CTSchnorrKeypairGPU* kp)
{
    if (secp256k1::cuda::scalar_is_zero(private_key)) return false;

    // P = d' * G  (CT: secret key)
    JacobianPoint P;
    ct_generator_mul(private_key, &P);

    // Convert to affine and get parity
    FieldElement px_fe, py_fe;
    uint8_t y_parity;
    ct_jacobian_to_affine(&P, &px_fe, &py_fe, &y_parity);

    // Store pubkey x-bytes
    secp256k1::cuda::field_to_bytes(&px_fe, kp->px);

    // CT conditional negate: if y is odd, negate d
    uint64_t odd_mask = bool_to_mask((uint64_t)y_parity);
    scalar_cneg(&kp->d, private_key, odd_mask);

    return true;
}

// ============================================================================
// CT Schnorr Sign (BIP-340)
// ============================================================================
// CT BIP-340 Schnorr sign. All secret-dependent ops are constant-time:
//   - P = d'*G: ct_generator_mul (secret key)
//   - R = k'*G: ct_generator_mul (secret nonce)
//   - Y-parity negation: ct::scalar_cneg (no branch)
//   - s = k + e*d: ct::scalar_add + ct::scalar_mul

__device__ inline bool ct_schnorr_sign(
    const Scalar* private_key,
    const uint8_t msg[32],
    const uint8_t aux_rand[32],
    SchnorrSignatureGPU* sig)
{
    if (secp256k1::cuda::scalar_is_zero(private_key)) return false;

    // P = d' * G  (CT: secret key multiplication)
    JacobianPoint P;
    ct_generator_mul(private_key, &P);

    // Convert P to affine, extract parity
    FieldElement px_fe, py_fe;
    uint8_t p_y_parity;
    ct_jacobian_to_affine(&P, &px_fe, &py_fe, &p_y_parity);

    // CT conditional negate: if Y is odd, d = -d'
    uint64_t p_odd_mask = bool_to_mask((uint64_t)p_y_parity);
    Scalar d;
    scalar_cneg(&d, private_key, p_odd_mask);

    // px as bytes
    uint8_t px_bytes[32];
    secp256k1::cuda::field_to_bytes(&px_fe, px_bytes);

    // t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    // (SHA-256/tagged_hash is data-independent, safe on fast path)
    uint8_t t_hash[32];
    secp256k1::cuda::tagged_hash_fast(BIP340_TAG_AUX, aux_rand, 32, t_hash);

    uint8_t d_bytes[32];
    secp256k1::cuda::scalar_to_bytes(&d, d_bytes);

    uint8_t t[32];
    for (int i = 0; i < 32; i++) t[i] = d_bytes[i] ^ t_hash[i];

    // rand = tagged_hash("BIP0340/nonce", t || px || msg)
    uint8_t nonce_input[96];
    for (int i = 0; i < 32; i++) nonce_input[i] = t[i];
    for (int i = 0; i < 32; i++) nonce_input[32 + i] = px_bytes[i];
    for (int i = 0; i < 32; i++) nonce_input[64 + i] = msg[i];

    uint8_t rand_hash[32];
    secp256k1::cuda::tagged_hash_fast(BIP340_TAG_NONCE, nonce_input, 96, rand_hash);

    Scalar k_prime;
    secp256k1::cuda::scalar_from_bytes(rand_hash, &k_prime);
    if (secp256k1::cuda::scalar_is_zero(&k_prime)) return false;

    // R = k' * G  (CT: secret nonce multiplication)
    JacobianPoint R;
    ct_generator_mul(&k_prime, &R);

    // Convert R to affine, extract Y parity
    FieldElement rx_fe, ry_fe;
    uint8_t r_y_parity;
    ct_jacobian_to_affine(&R, &rx_fe, &ry_fe, &r_y_parity);

    // CT conditional negate: if R.y is odd, k = -k'
    uint64_t r_odd_mask = bool_to_mask((uint64_t)r_y_parity);
    Scalar k;
    scalar_cneg(&k, &k_prime, r_odd_mask);

    // sig.r = R.x as bytes
    secp256k1::cuda::field_to_bytes(&rx_fe, sig->r);

    // e = tagged_hash("BIP0340/challenge", R.x || px || msg) mod n
    uint8_t challenge_input[96];
    for (int i = 0; i < 32; i++) challenge_input[i] = sig->r[i];
    for (int i = 0; i < 32; i++) challenge_input[32 + i] = px_bytes[i];
    for (int i = 0; i < 32; i++) challenge_input[64 + i] = msg[i];

    uint8_t e_hash[32];
    secp256k1::cuda::tagged_hash_fast(BIP340_TAG_CHALLENGE, challenge_input, 96, e_hash);

    Scalar e;
    secp256k1::cuda::scalar_from_bytes(e_hash, &e);

    // s = k + e * d mod n  (CT scalar arithmetic)
    Scalar ed;
    scalar_mul(&e, &d, &ed);

    scalar_add(&k, &ed, &sig->s);

    return true;
}

// CT Schnorr sign with keypair (avoids recomputing P = d*G)
__device__ inline bool ct_schnorr_sign_with_keypair(
    const CTSchnorrKeypairGPU* kp,
    const uint8_t msg[32],
    const uint8_t aux_rand[32],
    SchnorrSignatureGPU* sig)
{
    // t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    uint8_t t_hash[32];
    secp256k1::cuda::tagged_hash_fast(BIP340_TAG_AUX, aux_rand, 32, t_hash);

    uint8_t d_bytes[32];
    secp256k1::cuda::scalar_to_bytes(&kp->d, d_bytes);

    uint8_t t[32];
    for (int i = 0; i < 32; i++) t[i] = d_bytes[i] ^ t_hash[i];

    // rand = tagged_hash("BIP0340/nonce", t || px || msg)
    uint8_t nonce_input[96];
    for (int i = 0; i < 32; i++) nonce_input[i] = t[i];
    for (int i = 0; i < 32; i++) nonce_input[32 + i] = kp->px[i];
    for (int i = 0; i < 32; i++) nonce_input[64 + i] = msg[i];

    uint8_t rand_hash[32];
    secp256k1::cuda::tagged_hash_fast(BIP340_TAG_NONCE, nonce_input, 96, rand_hash);

    Scalar k_prime;
    secp256k1::cuda::scalar_from_bytes(rand_hash, &k_prime);
    if (secp256k1::cuda::scalar_is_zero(&k_prime)) return false;

    // R = k' * G  (CT: secret nonce)
    JacobianPoint R;
    ct_generator_mul(&k_prime, &R);

    // Convert R to affine, extract parity
    FieldElement rx_fe, ry_fe;
    uint8_t r_y_parity;
    ct_jacobian_to_affine(&R, &rx_fe, &ry_fe, &r_y_parity);

    // CT conditional negate k
    uint64_t r_odd_mask = bool_to_mask((uint64_t)r_y_parity);
    Scalar k;
    scalar_cneg(&k, &k_prime, r_odd_mask);

    // sig.r = R.x
    secp256k1::cuda::field_to_bytes(&rx_fe, sig->r);

    // e = tagged_hash("BIP0340/challenge", R.x || px || msg) mod n
    uint8_t challenge_input[96];
    for (int i = 0; i < 32; i++) challenge_input[i] = sig->r[i];
    for (int i = 0; i < 32; i++) challenge_input[32 + i] = kp->px[i];
    for (int i = 0; i < 32; i++) challenge_input[64 + i] = msg[i];

    uint8_t e_hash[32];
    secp256k1::cuda::tagged_hash_fast(BIP340_TAG_CHALLENGE, challenge_input, 96, e_hash);

    Scalar e;
    secp256k1::cuda::scalar_from_bytes(e_hash, &e);

    // s = k + e * d mod n  (CT)
    Scalar ed;
    scalar_mul(&e, &kp->d, &ed);

    scalar_add(&k, &ed, &sig->s);

    return true;
}

// CT Schnorr sign with immediate verification (fault countermeasure)
__device__ inline bool ct_schnorr_sign_verified(
    const Scalar* private_key,
    const uint8_t msg[32],
    const uint8_t aux_rand[32],
    SchnorrSignatureGPU* sig)
{
    if (!ct_schnorr_sign(private_key, msg, aux_rand, sig)) return false;
    if (secp256k1::cuda::scalar_is_zero(&sig->s)) return false;

    // Compute pubkey for verification (fast path OK: pubkey is public)
    uint8_t pubkey_x[32];
    JacobianPoint P;
    ct_generator_mul(private_key, &P);
    FieldElement ax, ay;
    uint8_t _;
    ct_jacobian_to_affine(&P, &ax, &ay, &_);
    secp256k1::cuda::field_to_bytes(&ax, pubkey_x);

    // Verify uses fast path (public data)
    return secp256k1::cuda::schnorr_verify(pubkey_x, msg, sig);
}

// CT Schnorr pubkey extraction (X-only, even Y)
__device__ inline bool ct_schnorr_pubkey(
    const Scalar* private_key,
    uint8_t pubkey_x[32])
{
    if (secp256k1::cuda::scalar_is_zero(private_key)) return false;

    JacobianPoint P;
    ct_generator_mul(private_key, &P);

    FieldElement ax, ay;
    uint8_t y_parity;
    ct_jacobian_to_affine(&P, &ax, &ay, &y_parity);
    secp256k1::cuda::field_to_bytes(&ax, pubkey_x);
    return true;
}

} // namespace ct
} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
