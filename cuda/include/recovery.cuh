#pragma once
// ============================================================================
// ECDSA Key Recovery — CUDA device implementation
// ============================================================================
// - ecdsa_sign_recoverable: ECDSA sign with recovery ID (recid 0-3)
// - ecdsa_recover: recover public key from signature + recid
//
// Recid bits:
//   bit 0: parity of R.y (0=even, 1=odd)
//   bit 1: R.x overflow (R.x >= n, extremely rare ~2^-128)
//
// 64-bit limb mode only.
// ============================================================================

#include "ecdsa.cuh"   // ECDSASignatureGPU, SHA256, RFC 6979, scalar ops

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// ── Recoverable Signature ────────────────────────────────────────────────────

struct RecoverableSignatureGPU {
    ECDSASignatureGPU sig;
    int recid;  // 0-3
};

// ── Lift x-coordinate to curve point ─────────────────────────────────────────
// Given x as FieldElement, compute point with y parity matching `parity`.
// Returns false if x is not on the curve.

__device__ inline bool lift_x_field(
    const FieldElement* x_fe,
    int parity,
    JacobianPoint* p)
{
    // y² = x³ + 7
    FieldElement x2, x3, y2, seven, y;
    field_sqr(x_fe, &x2);
    field_mul(&x2, x_fe, &x3);

    field_set_zero(&seven);
    seven.limbs[0] = 7;

    field_add(&x3, &seven, &y2);

    // y = sqrt(y²) = y2^((p+1)/4)
    field_sqrt(&y2, &y);

    // Verify: y² == y2
    FieldElement y_check;
    field_sqr(&y, &y_check);
    bool valid = true;
    for (int i = 0; i < 4; i++) {
        if (y_check.limbs[i] != y2.limbs[i]) valid = false;
    }
    if (!valid) return false;

    // Check parity and adjust
    uint8_t y_bytes[32];
    field_to_bytes(&y, y_bytes);
    bool y_is_odd = (y_bytes[31] & 1) != 0;
    if ((parity != 0) != y_is_odd) {
        FieldElement zero;
        field_set_zero(&zero);
        field_sub(&zero, &y, &y);
    }

    p->x = *x_fe;
    p->y = y;
    field_set_one(&p->z);
    p->infinity = false;
    return true;
}

// ── ECDSA Sign with Recovery ID ──────────────────────────────────────────────

__device__ inline bool ecdsa_sign_recoverable(
    const uint8_t msg_hash[32],
    const Scalar* private_key,
    RecoverableSignatureGPU* rsig)
{
    if (scalar_is_zero(private_key)) return false;

    Scalar z;
    scalar_from_bytes(msg_hash, &z);

    // k = RFC 6979 nonce
    Scalar k;
    rfc6979_nonce(private_key, msg_hash, &k);
    if (scalar_is_zero(&k)) return false;

    // R = k * G
    JacobianPoint R;
    scalar_mul(&GENERATOR_JACOBIAN, &k, &R);
    if (R.infinity) return false;

    // Convert R to affine
    FieldElement z_inv, z_inv2, z_inv3, rx_aff, ry_aff;
    field_inv(&R.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&R.x, &z_inv2, &rx_aff);
    field_mul(&R.y, &z_inv3, &ry_aff);

    // r = R.x mod n (as scalar)
    uint8_t rx_bytes[32];
    field_to_bytes(&rx_aff, rx_bytes);
    Scalar r;
    scalar_from_bytes(rx_bytes, &r);
    if (scalar_is_zero(&r)) return false;

    // Determine recovery ID
    int recid = 0;

    // bit 0: parity of R.y
    uint8_t ry_bytes[32];
    field_to_bytes(&ry_aff, ry_bytes);
    if (ry_bytes[31] & 1) recid |= 1;

    // bit 1: R.x >= n (overflow)
    // Compare rx_bytes (big-endian) with ORDER_BYTES
    // ORDER in big-endian:
    static const uint8_t ORDER_BE[32] = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
    };
    bool overflow = false;
    for (int i = 0; i < 32; i++) {
        if (rx_bytes[i] < ORDER_BE[i]) break;
        if (rx_bytes[i] > ORDER_BE[i]) { overflow = true; break; }
    }
    if (overflow) recid |= 2;

    // s = k⁻¹ * (z + r*d) mod n
    Scalar k_inv;
    scalar_inverse(&k, &k_inv);

    Scalar rd;
    scalar_mul_mod_n(&r, private_key, &rd);

    // z + rd mod n
    Scalar z_plus_rd;
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        unsigned __int128 sum = (unsigned __int128)z.limbs[i] + rd.limbs[i] + carry;
        z_plus_rd.limbs[i] = (uint64_t)sum;
        carry = (uint64_t)(sum >> 64);
    }
    {
        uint64_t borrow = 0;
        uint64_t tmp[4];
        for (int i = 0; i < 4; i++) {
            unsigned __int128 diff = (unsigned __int128)z_plus_rd.limbs[i] - ORDER[i] - borrow;
            tmp[i] = (uint64_t)diff;
            borrow = (uint64_t)(-(int64_t)(diff >> 64));
        }
        uint64_t mask = -(uint64_t)(borrow == 0 || carry);
        for (int i = 0; i < 4; i++) {
            z_plus_rd.limbs[i] = (tmp[i] & mask) | (z_plus_rd.limbs[i] & ~mask);
        }
    }

    // s = k_inv * (z + rd)
    Scalar s;
    scalar_mul_mod_n(&k_inv, &z_plus_rd, &s);
    if (scalar_is_zero(&s)) return false;

    // Normalize to low-S (BIP-62)
    if (!scalar_is_low_s(&s)) {
        scalar_negate(&s, &s);
        recid ^= 1;
    }

    rsig->sig.r = r;
    rsig->sig.s = s;
    rsig->recid = recid;
    return true;
}

// ── ECDSA Public Key Recovery ────────────────────────────────────────────────
// Q = r⁻¹ * (s*R - z*G)

__device__ inline bool ecdsa_recover(
    const uint8_t msg_hash[32],
    const ECDSASignatureGPU* sig,
    int recid,
    JacobianPoint* Q)
{
    if (recid < 0 || recid > 3) return false;
    if (scalar_is_zero(&sig->r) || scalar_is_zero(&sig->s)) return false;

    // Step 1: Reconstruct R.x field element
    // If recid bit 1 is set, R.x = r + n (extremely rare)
    FieldElement rx_fe;
    {
        uint8_t r_bytes[32];
        scalar_to_bytes(&sig->r, r_bytes);

        // Parse r as field element
        for (int i = 0; i < 4; i++) {
            uint64_t limb = 0;
            int base = (3 - i) * 8;
            for (int j = 0; j < 8; j++) limb = (limb << 8) | r_bytes[base + j];
            rx_fe.limbs[i] = limb;
        }

        if (recid & 2) {
            // Add n to rx_fe (field addition — n as field element)
            FieldElement n_fe;
            n_fe.limbs[0] = ORDER[0];
            n_fe.limbs[1] = ORDER[1];
            n_fe.limbs[2] = ORDER[2];
            n_fe.limbs[3] = ORDER[3];
            field_add(&rx_fe, &n_fe, &rx_fe);
        }
    }

    // Step 2: Lift x to curve point R with correct y parity
    int y_parity = recid & 1;
    JacobianPoint R;
    if (!lift_x_field(&rx_fe, y_parity, &R)) return false;

    // Step 3: Recover public key Q = r⁻¹ * (s*R - z*G)
    Scalar z;
    scalar_from_bytes(msg_hash, &z);

    Scalar r_inv;
    scalar_inverse(&sig->r, &r_inv);

    // s * R
    JacobianPoint sR;
    scalar_mul(&R, &sig->s, &sR);

    // z * G
    JacobianPoint zG;
    scalar_mul(&GENERATOR_JACOBIAN, &z, &zG);

    // Negate zG (negate Y)
    FieldElement fzero;
    field_set_zero(&fzero);
    field_sub(&fzero, &zG.y, &zG.y);

    // sR - zG = sR + (-zG)
    JacobianPoint sR_minus_zG;
    jacobian_add(&sR, &zG, &sR_minus_zG);

    // Q = r_inv * (sR - zG)
    scalar_mul(&sR_minus_zG, &r_inv, Q);

    if (Q->infinity) return false;
    return true;
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
