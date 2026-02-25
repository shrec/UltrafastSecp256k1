#pragma once
// ============================================================================
// Schnorr Signatures (BIP-340) -- CUDA device implementation
// ============================================================================
// - Tagged hash: H_tag(msg) = SHA256(SHA256(tag) || SHA256(tag) || msg)
// - Schnorr sign (BIP-340): X-only pubkeys, deterministic nonce
// - Schnorr verify (BIP-340): lift_x + verification equation
// - Field square root: needed by lift_x to recover Y from X
//
// 64-bit limb mode only.
// ============================================================================

#include "ecdsa.cuh"   // for SHA256Ctx, sha256_*, scalar_from_bytes, etc.

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// -- Tagged Hash (BIP-340) ----------------------------------------------------
// H_tag(msg) = SHA256(SHA256(tag) || SHA256(tag) || msg)

__device__ inline void tagged_hash(
    const char* tag, size_t tag_len,
    const uint8_t* data, size_t data_len,
    uint8_t out[32])
{
    // Precompute SHA256(tag)
    uint8_t tag_hash[32];
    {
        SHA256Ctx ctx; sha256_init(&ctx);
        sha256_update(&ctx, (const uint8_t*)tag, tag_len);
        sha256_final(&ctx, tag_hash);
    }

    // H_tag(msg) = SHA256(tag_hash || tag_hash || msg)
    SHA256Ctx ctx; sha256_init(&ctx);
    sha256_update(&ctx, tag_hash, 32);
    sha256_update(&ctx, tag_hash, 32);
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

// -- Precomputed Tagged Hash Midstates (BIP-340) -----------------------------
// SHA256 state after processing SHA256(tag)||SHA256(tag) (one 64-byte block).
// Saves 2 SHA-256 block compressions per tagged_hash call (6 total per sign/verify).
// Each midstate: h[8] = SHA256 state, total = 64 bytes processed, buf_len = 0.

// Tag index constants for midstate lookup
#define BIP340_TAG_AUX       0
#define BIP340_TAG_NONCE     1
#define BIP340_TAG_CHALLENGE 2

__device__ __constant__ static const uint32_t BIP340_MIDSTATES[3][8] = {
    // "BIP0340/aux" midstate
    {0x24dd3219U, 0x4eba7e70U, 0xca0fabb9U, 0x0fa3166dU,
     0x3afbe4b1U, 0x4c44df97U, 0x4aac2739U, 0x249e850aU},
    // "BIP0340/nonce" midstate
    {0x46615b35U, 0xf4bfbff7U, 0x9f8dc671U, 0x83627ab3U,
     0x60217180U, 0x57358661U, 0x21a29e54U, 0x68b07b4cU},
    // "BIP0340/challenge" midstate
    {0x9cecba11U, 0x23925381U, 0x11679112U, 0xd1627e0fU,
     0x97c87550U, 0x003cc765U, 0x90f61164U, 0x33e9b66aU},
};

// Fast tagged hash using precomputed midstate (saves 2 SHA-256 compressions).
// tag_idx: BIP340_TAG_AUX=0, BIP340_TAG_NONCE=1, BIP340_TAG_CHALLENGE=2
__device__ inline void tagged_hash_fast(
    int tag_idx,
    const uint8_t* data, size_t data_len,
    uint8_t out[32])
{
    SHA256Ctx ctx;
    // Load precomputed midstate
    for (int i = 0; i < 8; i++) ctx.h[i] = BIP340_MIDSTATES[tag_idx][i];
    ctx.buf_len = 0;
    ctx.total = 64;
    // Continue with data
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

// Helper: strlen for device strings
__device__ inline size_t dev_strlen(const char* s) {
    size_t n = 0;
    while (s[n]) n++;
    return n;
}

// -- Lift X (BIP-340): recover Y from X-only pubkey --------------------------
// Given 32-byte x coordinate, compute the point with even Y.
// Returns false if x is not on the curve.

__device__ inline bool lift_x(
    const uint8_t x_bytes[32],
    JacobianPoint* p)
{
    // Parse x as field element
    FieldElement x;
    for (int i = 0; i < 4; i++) {
        uint64_t limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++) limb = (limb << 8) | x_bytes[base + j];
        x.limbs[i] = limb;
    }

    // y^2 = x^3 + 7
    FieldElement x2, x3, y2, seven, y;
    field_sqr(&x, &x2);
    field_mul(&x2, &x, &x3);

    // seven = 7 (field element)
    field_set_zero(&seven);
    seven.limbs[0] = 7;

    field_add(&x3, &seven, &y2);

    // y = sqrt(y^2) = y2^((p+1)/4)
    field_sqrt(&y2, &y);

    // Verify: y^2 == y2 (compare via normalized bytes to handle unreduced limbs)
    FieldElement y_check;
    field_sqr(&y, &y_check);
    uint8_t y_check_bytes[32], y2_bytes[32];
    field_to_bytes(&y_check, y_check_bytes);
    field_to_bytes(&y2, y2_bytes);
    bool valid = true;
    for (int i = 0; i < 32; i++) {
        if (y_check_bytes[i] != y2_bytes[i]) valid = false;
    }
    if (!valid) return false;

    // BIP-340: ensure y is even (y_bytes[31] & 1 == 0)
    uint8_t y_bytes[32];
    field_to_bytes(&y, y_bytes);
    if (y_bytes[31] & 1) {
        // Negate y: y = p - y
        FieldElement zero;
        field_set_zero(&zero);
        field_sub(&zero, &y, &y);
    }

    p->x = x;
    p->y = y;
    field_set_one(&p->z);
    p->infinity = false;
    return true;
}

// -- Schnorr Signature Struct -------------------------------------------------

struct SchnorrSignatureGPU {
    uint8_t r[32];   // R.x (x-coordinate of nonce point)
    Scalar s;         // scalar s
};

// -- BIP-340 Schnorr Sign -----------------------------------------------------
// Signs a 32-byte message with a private key using BIP-340.
// aux_rand: 32 bytes of auxiliary randomness (can be zeros for deterministic).
// Returns false on failure.

__device__ inline bool schnorr_sign(
    const Scalar* private_key,
    const uint8_t msg[32],
    const uint8_t aux_rand[32],
    SchnorrSignatureGPU* sig)
{
    if (scalar_is_zero(private_key)) return false;

    // P = d' * G  (use precomputed constant table for generator)
    JacobianPoint P;
    scalar_mul_generator_const(private_key, &P);
    if (P.infinity) return false;

    // Convert to affine
    FieldElement z_inv, z_inv2, z_inv3, px, py;
    field_inv(&P.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&P.x, &z_inv2, &px);
    field_mul(&P.y, &z_inv3, &py);

    // Check parity of Y: if odd, negate d
    uint8_t py_bytes[32];
    field_to_bytes(&py, py_bytes);
    Scalar d;
    if (py_bytes[31] & 1) {
        scalar_negate(private_key, &d);
    } else {
        d = *private_key;
    }

    // px as bytes (for tagged hashes)
    uint8_t px_bytes[32];
    field_to_bytes(&px, px_bytes);

    // t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    uint8_t t_hash[32];
    tagged_hash_fast(BIP340_TAG_AUX, aux_rand, 32, t_hash);

    uint8_t d_bytes[32];
    scalar_to_bytes(&d, d_bytes);

    uint8_t t[32];
    for (int i = 0; i < 32; i++) t[i] = d_bytes[i] ^ t_hash[i];

    // rand = tagged_hash("BIP0340/nonce", t || px || msg)
    uint8_t nonce_input[96];
    for (int i = 0; i < 32; i++) nonce_input[i] = t[i];
    for (int i = 0; i < 32; i++) nonce_input[32 + i] = px_bytes[i];
    for (int i = 0; i < 32; i++) nonce_input[64 + i] = msg[i];

    uint8_t rand_hash[32];
    tagged_hash_fast(BIP340_TAG_NONCE, nonce_input, 96, rand_hash);

    Scalar k_prime;
    scalar_from_bytes(rand_hash, &k_prime);
    if (scalar_is_zero(&k_prime)) return false;

    // R = k' * G  (use precomputed constant table for generator)
    JacobianPoint R;
    scalar_mul_generator_const(&k_prime, &R);

    // Convert R to affine
    FieldElement rz_inv, rz_inv2, rz_inv3, rx, ry;
    field_inv(&R.z, &rz_inv);
    field_sqr(&rz_inv, &rz_inv2);
    field_mul(&rz_inv, &rz_inv2, &rz_inv3);
    field_mul(&R.x, &rz_inv2, &rx);
    field_mul(&R.y, &rz_inv3, &ry);

    // If R.y is odd, negate k
    uint8_t ry_bytes[32];
    field_to_bytes(&ry, ry_bytes);
    Scalar k;
    if (ry_bytes[31] & 1) {
        scalar_negate(&k_prime, &k);
    } else {
        k = k_prime;
    }

    // sig.r = R.x as bytes
    field_to_bytes(&rx, sig->r);

    // e = tagged_hash("BIP0340/challenge", R.x || px || msg) mod n
    uint8_t challenge_input[96];
    for (int i = 0; i < 32; i++) challenge_input[i] = sig->r[i];
    for (int i = 0; i < 32; i++) challenge_input[32 + i] = px_bytes[i];
    for (int i = 0; i < 32; i++) challenge_input[64 + i] = msg[i];

    uint8_t e_hash[32];
    tagged_hash_fast(BIP340_TAG_CHALLENGE, challenge_input, 96, e_hash);

    Scalar e;
    scalar_from_bytes(e_hash, &e);

    // s = k + e * d mod n
    Scalar ed;
    scalar_mul_mod_n(&e, &d, &ed);

    // s = k + ed mod n (addition with reduction)
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 sum = (unsigned __int128)k.limbs[i] + ed.limbs[i] + carry;
        sig->s.limbs[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
    // Reduce mod n
    uint64_t borrow = 0;
    uint64_t tmp[4];
    for (int i = 0; i < 4; i++) {
        unsigned __int128 diff = (unsigned __int128)sig->s.limbs[i] - ORDER[i] - borrow;
        tmp[i] = (uint64_t)diff;
        borrow = (uint64_t)(-(int64_t)(diff >> 64));
    }
    uint64_t mask = -(uint64_t)(borrow == 0 || carry);
    for (int i = 0; i < 4; i++) {
        sig->s.limbs[i] = (tmp[i] & mask) | (sig->s.limbs[i] & ~mask);
    }

    return true;
}

// -- BIP-340 Schnorr Verify ---------------------------------------------------
// Verifies a BIP-340 Schnorr signature.

__device__ inline bool schnorr_verify(
    const uint8_t pubkey_x[32],
    const uint8_t msg[32],
    const SchnorrSignatureGPU* sig)
{
    if (scalar_is_zero(&sig->s)) return false;

    // Lift pubkey x-only to full point
    JacobianPoint P;
    if (!lift_x(pubkey_x, &P)) return false;

    // e = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg) mod n
    uint8_t challenge_input[96];
    for (int i = 0; i < 32; i++) challenge_input[i] = sig->r[i];
    for (int i = 0; i < 32; i++) challenge_input[32 + i] = pubkey_x[i];
    for (int i = 0; i < 32; i++) challenge_input[64 + i] = msg[i];

    uint8_t e_hash[32];
    tagged_hash_fast(BIP340_TAG_CHALLENGE, challenge_input, 96, e_hash);

    Scalar e;
    scalar_from_bytes(e_hash, &e);

    // R = s*G - e*P = s*G + neg_e*P  (Shamir's trick with GLV: ~128 doublings)
    Scalar neg_e;
    scalar_negate(&e, &neg_e);

    JacobianPoint R;
    shamir_double_mul_glv(&GENERATOR_JACOBIAN, &sig->s, &P, &neg_e, &R);

    if (R.infinity) return false;

    // Convert R to affine
    FieldElement rz_inv, rz_inv2, rz_inv3, rx_aff, ry_aff;
    field_inv(&R.z, &rz_inv);
    field_sqr(&rz_inv, &rz_inv2);
    field_mul(&rz_inv, &rz_inv2, &rz_inv3);
    field_mul(&R.x, &rz_inv2, &rx_aff);
    field_mul(&R.y, &rz_inv3, &ry_aff);

    // Check R has even y
    uint8_t ry_bytes[32];
    field_to_bytes(&ry_aff, ry_bytes);
    if (ry_bytes[31] & 1) return false;

    // Check R.x == sig.r
    uint8_t rx_bytes[32];
    field_to_bytes(&rx_aff, rx_bytes);
    for (int i = 0; i < 32; i++) {
        if (rx_bytes[i] != sig->r[i]) return false;
    }

    return true;
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
