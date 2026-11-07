#pragma once
// ============================================================================
// ECDSA Key Recovery -- CUDA device implementation
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

#include "ecdsa.cuh"         // ECDSASignatureGPU, SHA256, RFC 6979, scalar ops
#include "ct/ct_point.cuh"   // BUG-H2 FIX: CT generator mul for secret nonce k

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// -- Recoverable Signature ----------------------------------------------------

struct RecoverableSignatureGPU {
    ECDSASignatureGPU sig;
    int recid;  // 0-3
};

// -- Lift x-coordinate to curve point -----------------------------------------
// Given x as FieldElement, compute point with y parity matching `parity`.
// Returns false if x is not on the curve.

__device__ inline bool lift_x_field(
    const FieldElement* x_fe,
    int parity,
    JacobianPoint* p)
{
    // y^2 = x^3 + 7
    FieldElement x2, x3, y2, seven, y;
    field_sqr(x_fe, &x2);
    field_mul(&x2, x_fe, &x3);

    field_set_zero(&seven);
    seven.limbs[0] = 7;

    field_add(&x3, &seven, &y2);

    // y = sqrt(y^2) = y2^((p+1)/4)
    field_sqrt(&y2, &y);

    // Verify: y^2 == y2 (compare via normalized bytes to handle unreduced limbs)
    FieldElement y_check;
    field_sqr(&y, &y_check);
    uint8_t y_check_bytes[32], y2_bytes_cmp[32];
    field_to_bytes(&y_check, y_check_bytes);
    field_to_bytes(&y2, y2_bytes_cmp);
    bool valid = true;
    for (int i = 0; i < 32; i++) {
        if (y_check_bytes[i] != y2_bytes_cmp[i]) valid = false;
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

// -- ECDSA Sign with Recovery ID ----------------------------------------------

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

    // BUG-H2 FIX: CT generator multiply — no variable-time wNAF on secret nonce k.
    // Previously: scalar_mul (windowed wNAF, secret-dependent access) leaked k via timing.
    JacobianPoint R_ct;
    ct::ct_generator_mul(&k, &R_ct);
    if (R_ct.infinity) return false;

    // Affine conversion: these field ops are data-independent (Z is not secret)
    FieldElement z_inv, z_inv2, z_inv3, rx_aff, ry_aff;
    field_inv(&R_ct.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&R_ct.x, &z_inv2, &rx_aff);
    field_mul(&R_ct.y, &z_inv3, &ry_aff);

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
    // CT-007: branchless comparison — no early-exit on nonce-derived rx_bytes.
    // Mask-accumulation pattern: gt tracks if rx > order seen so far (MSB first).
    static const uint8_t ORDER_BE[32] = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
    };
    uint32_t gt = 0u, eq_acc = 0xFFFFFFFFu;
    for (int i = 0; i < 32; i++) {
        const uint32_t a = (uint32_t)rx_bytes[i];
        const uint32_t b = (uint32_t)ORDER_BE[i];
        const uint32_t byte_gt = (uint32_t)(-(int32_t)(a > b));
        const uint32_t byte_lt = (uint32_t)(-(int32_t)(a < b));
        gt |= (byte_gt & eq_acc);
        eq_acc &= ~(byte_gt | byte_lt);
    }
    const bool overflow = (gt != 0u);
    if (overflow) recid |= 2;

    // BUG-H2 FIX: CT Fermat inverse — fixed 256-iteration trace, no secret-dependent branches.
    // Previously: scalar_inverse (extended-GCD, variable-time) leaked nonce bits via timing.
    Scalar k_inv;
    ct::scalar_inverse(&k, &k_inv);

    Scalar rd;
    scalar_mul_mod_n(&r, private_key, &rd);

    // z + rd mod n
    Scalar z_plus_rd;
    uint64_t carry = 0;
    for (int i = 0; i < 4; i++) {
        const uint64_t sum = z.limbs[i] + rd.limbs[i];
        const uint64_t carry0 = (sum < z.limbs[i]) ? 1ULL : 0ULL;
        z_plus_rd.limbs[i] = sum + carry;
        carry = carry0 | ((z_plus_rd.limbs[i] < sum) ? 1ULL : 0ULL);
    }
    {
        uint64_t borrow = 0;
        uint64_t tmp[4];
        for (int i = 0; i < 4; i++) {
            const uint64_t diff = z_plus_rd.limbs[i] - ORDER[i];
            const uint64_t borrow0 = (z_plus_rd.limbs[i] < ORDER[i]) ? 1ULL : 0ULL;
            tmp[i] = diff - borrow;
            borrow = borrow0 | ((diff < borrow) ? 1ULL : 0ULL);
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

// -- ECDSA Public Key Recovery ------------------------------------------------
// Q = r^-^1 * (s*R - z*G)

__device__ inline bool ecdsa_recover(
    const uint8_t msg_hash[32],
    const ECDSASignatureGPU* sig,
    int recid,
    JacobianPoint* Q)
{
    if (recid < 0 || recid > 3) return false;
    if (scalar_is_zero(&sig->r) || scalar_is_zero(&sig->s)) return false;
    // r >= n or s >= n is not a valid compact signature (the CPU parser and
    // ecdsa_verify reject it); the host hands raw limbs to this kernel
    // (bytes_to_ecdsa_sig, no parse), so reject here instead of reducing mod n.
    if (scalar_ge(&sig->r, ORDER) || scalar_ge(&sig->s, ORDER)) return false;

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
            // bbhunt-001: reject r >= (p - n). Otherwise r + n overflows the field
            // prime and field_add reduces it mod p to (r + n - p) -- a DIFFERENT
            // x-coordinate -- which could lift to a bogus pubkey reported as success
            // where upstream libsecp256k1 returns 0. Match upstream for cross-backend
            // consensus parity (recid/sig are public -> data-dependent branch is fine).
            // p - n little-endian 64-bit limbs:
            const uint64_t PMN0 = 0x402DA1722FC9BAEEULL;
            const uint64_t PMN1 = 0x4551231950B75FC4ULL;
            const uint64_t PMN2 = 0x0000000000000001ULL;
            const uint64_t PMN3 = 0x0000000000000000ULL;
            bool r_ge_pmn;
            if      (rx_fe.limbs[3] != PMN3) r_ge_pmn = (rx_fe.limbs[3] > PMN3);
            else if (rx_fe.limbs[2] != PMN2) r_ge_pmn = (rx_fe.limbs[2] > PMN2);
            else if (rx_fe.limbs[1] != PMN1) r_ge_pmn = (rx_fe.limbs[1] > PMN1);
            else                             r_ge_pmn = (rx_fe.limbs[0] >= PMN0);
            if (r_ge_pmn) return false;

            // Add n to rx_fe (field addition -- n as field element)
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

    // Step 3: Q = r^-1 * (s*R - z*G) = u1*G + u2*R with u1 = -z*r^-1 and
    // u2 = s*r^-1. Every input (r, s, recid, message hash) is public and R is
    // lifted from r, so r^-1, the Shamir table inversion and the final Q.z
    // inversion are variable time (same contract as ecdsa_verify).
    Scalar z;
    scalar_from_bytes(msg_hash, &z);

    Scalar r_inv;
    scalar_inv_var(&sig->r, &r_inv);

    Scalar zr, u1, u2;
    scalar_mul_mod_n(&z, &r_inv, &zr);
    scalar_negate(&zr, &u1);
    scalar_mul_mod_n(&sig->s, &r_inv, &u2);

    // One GLV Shamir dual multiplication (R has Z = 1 from lift_x_field).
    JacobianPoint Qj;
    shamir_double_mul_glv(&GENERATOR_JACOBIAN, &u1, &R, &u2, &Qj);
    if (Qj.infinity) { *Q = Qj; return false; }

    // The Shamir loop adds without the P == +-Q checks. An input whose true
    // result is the point at infinity (R = c*G with z = c*s, crafted from
    // public data) or an exceptional doubling leaves Z = 0 with the infinity
    // flag clear, and Z = 0 is sticky. Recompute exactly with two single
    // multiplications and the complete jacobian_add in that case only.
    bool z_is_zero = true;
    {
        uint8_t zb[32];
        field_to_bytes(&Qj.z, zb);
        for (int i = 0; i < 32; i++) if (zb[i] != 0) z_is_zero = false;
    }
    if (z_is_zero) {
        JacobianPoint uR;
        scalar_mul(&R, &u2, &uR);
        if (scalar_is_zero(&u1)) {
            Qj = uR;
        } else {
            JacobianPoint uG;
            scalar_mul(&GENERATOR_JACOBIAN, &u1, &uG);
            jacobian_add(&uG, &uR, &Qj);
        }
        if (Qj.infinity) { *Q = Qj; return false; }
    }

    // One affine conversion; Q is returned with Z = 1.
    FieldElement qx, qy;
    if (!jacobian_to_affine_var(&Qj, &qx, &qy)) return false;
    Q->x = qx;
    Q->y = qy;
    field_set_one(&Q->z);
    Q->infinity = false;
    return true;
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
