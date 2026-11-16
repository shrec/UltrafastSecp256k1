#pragma once
// ============================================================================
// Bitcoin Cash 2019 Schnorr verification -- CUDA device implementation
// ============================================================================
// NOT BIP-340. Spec: https://upgradespecs.bitcoincashnode.org/2019-05-15-schnorr/
// CPU reference: secp256k1::bch_schnorr_verify (src/cpu/src/bch_schnorr.cpp).
//
//   sig = r[32] || s[32] (big-endian), pubkey = 33-byte SEC1 compressed
//   fail if r >= p, s >= n or s == 0, or the key does not decompress
//   e = SHA256(r || pubkey33 || msg32) mod n       (97 bytes, two blocks)
//   R = s*G - e*P                                  (GLV Shamir)
//   fail if R = O, if R.y is not a quadratic residue, or if x(R) != r
//
// No field inversion: R = (X, Y, Z) Jacobian, y = Y/Z^3 = (Y*Z)/Z^4 and Z^4 is
// a square, so QR(y) == QR(Y*Z); and x(R) == r  <=>  X == r*Z^2 (r < p).
//
// VARIABLE TIME -- PUBLIC DATA ONLY. Every input of a verification is public
// (CLAUDE.md CT-VERIFY); ct:: primitives here would be a performance bug.
//
// 64-bit limb mode only.
// ============================================================================

#include "ecdsa.cuh"   // sha256_compress, scalar/field byte helpers, GLV Shamir

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// e = SHA256(r32 || pub33 || msg32), 97 bytes = two SHA-256 blocks, no
// streaming context. Bit length 97 * 8 = 776 = 0x0308.
__device__ inline void bch_schnorr_challenge_hash(
    const uint8_t r32[32], const uint8_t pub33[33], const uint8_t msg32[32],
    uint8_t out32[32])
{
    uint32_t h[8] = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U
    };
    uint8_t block[64];
    for (int i = 0; i < 32; ++i) block[i]      = r32[i];
    for (int i = 0; i < 32; ++i) block[32 + i] = pub33[i];
    sha256_compress(h, block);

    block[0] = pub33[32];
    for (int i = 0; i < 32; ++i) block[1 + i] = msg32[i];
    block[33] = 0x80u;
    for (int i = 34; i < 62; ++i) block[i] = 0u;
    block[62] = 0x03u;
    block[63] = 0x08u;
    sha256_compress(h, block);

    for (int i = 0; i < 8; ++i) {
        out32[i * 4    ] = (uint8_t)(h[i] >> 24);
        out32[i * 4 + 1] = (uint8_t)(h[i] >> 16);
        out32[i * 4 + 2] = (uint8_t)(h[i] >>  8);
        out32[i * 4 + 3] = (uint8_t)(h[i]);
    }
}

// a and b compared as canonical field elements (both normalised by field_to_bytes).
__device__ inline bool bch_field_eq_var(const FieldElement* a, const FieldElement* b) {
    uint8_t ab[32], bb[32];
    field_to_bytes(a, ab);
    field_to_bytes(b, bb);
    unsigned diff = 0;
    for (int i = 0; i < 32; ++i) diff |= (unsigned)(ab[i] ^ bb[i]);
    return diff == 0;
}

// Quadratic-residue test of a nonzero field element: sqrt(v)^2 == v, where
// field_sqrt is v^((p+1)/4) (the same exponentiation cost as Euler's criterion).
__device__ inline bool bch_field_is_qr_var(const FieldElement* v) {
    FieldElement root, sq;
    field_sqrt(v, &root);
    field_sqr(&root, &sq);
    return bch_field_eq_var(&sq, v);
}

// Full verification of one BCH 2019 Schnorr signature on wire-form inputs.
__device__ inline bool bch_schnorr_verify_device(
    const uint8_t sig64[64], const uint8_t msg32[32], const uint8_t pub33[33])
{
    // r < p (the x check compares r as a field element).
    FieldElement r_fe;
    if (!field_from_bytes_strict(sig64, &r_fe)) return false;

    // 0 < s < n.
    Scalar s;
    if (!scalar_from_bytes_strict_nonzero(sig64 + 32, &s)) return false;

    // Canonical compressed key: prefix 02/03, x < p, x on the curve.
    if (pub33[0] != 0x02u && pub33[0] != 0x03u) return false;
    {
        FieldElement x_chk;
        if (!field_from_bytes_strict(pub33 + 1, &x_chk)) return false;
    }
    JacobianPoint P;
    if (!point_from_compressed(pub33, &P)) return false;

    // e = SHA256(r || pubkey33 || msg32) mod n.
    uint8_t e_hash[32];
    bch_schnorr_challenge_hash(sig64, pub33, msg32, e_hash);
    Scalar e, neg_e;
    scalar_from_bytes(e_hash, &e);
    scalar_negate(&e, &neg_e);

    // R = s*G - e*P.
    JacobianPoint R;
    shamir_double_mul_glv(&GENERATOR_JACOBIAN, &s, &P, &neg_e, &R);
    if (R.infinity) return false;

    // R.y must be a quadratic residue: Jacobi(Y*Z), no affine conversion.
    // Y != 0 for every finite point (the group has odd prime order).
    FieldElement yz;
    field_mul(&R.y, &R.z, &yz);
    if (!bch_field_is_qr_var(&yz)) return false;

    // x(R) == r  <=>  X == r*Z^2.
    FieldElement zz, r_dom, rzz;
    field_sqr(&R.z, &zz);
#if SECP256K1_CUDA_USE_MONTGOMERY
    field_to_mont(&r_fe, &r_dom);     // same domain as R's coordinates
#else
    r_dom = r_fe;
#endif
    field_mul(&r_dom, &zz, &rzz);
    return bch_field_eq_var(&rzz, &R.x);
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
