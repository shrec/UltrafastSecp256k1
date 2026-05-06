// =============================================================================
// secp256k1_recovery.h -- ECDSA Public Key Recovery for Metal
// =============================================================================
// - ecdsa_sign_recoverable_metal: sign with recovery ID
// - ecdsa_recover_metal: recover public key from signature + recid
// =============================================================================

#ifndef SECP256K1_RECOVERY_H
#define SECP256K1_RECOVERY_H

// Required for ct_generator_mul_metal and ct_jacobian_to_affine_metal
#include "secp256k1_ct_point.h"
#include "secp256k1_ct_sign.h"

struct RecoverableSignatureMetal {
    Scalar256 r;
    Scalar256 s;
    int recid;
};

// -- lift_x to point with given parity (8x32) ---------------------------------
inline bool lift_x_parity_metal(thread const FieldElement& x_fe, int parity,
                                 thread JacobianPoint& p) {
    FieldElement x2 = field_sqr(x_fe);
    FieldElement x3 = field_mul(x2, x_fe);
    FieldElement seven = field_zero();
    seven.limbs[0] = 7;
    FieldElement y2 = field_add(x3, seven);
    FieldElement y = field_sqrt(y2);

    // Verify
    FieldElement check = field_sqr(y);
    uchar cb[32], yb[32];
    for (int i = 0; i < 8; ++i) {
        uint v1 = check.limbs[7-i], v2 = y2.limbs[7-i];
        cb[i*4+0]=(uchar)(v1>>24); cb[i*4+1]=(uchar)(v1>>16);
        cb[i*4+2]=(uchar)(v1>>8);  cb[i*4+3]=(uchar)(v1);
        yb[i*4+0]=(uchar)(v2>>24); yb[i*4+1]=(uchar)(v2>>16);
        yb[i*4+2]=(uchar)(v2>>8);  yb[i*4+3]=(uchar)(v2);
    }
    for (int i = 0; i < 32; ++i)
        if (cb[i] != yb[i]) return false;

    // Adjust parity
    uchar y_bytes[32];
    for (int i = 0; i < 8; ++i) {
        uint v = y.limbs[7-i];
        y_bytes[i*4+0]=(uchar)(v>>24); y_bytes[i*4+1]=(uchar)(v>>16);
        y_bytes[i*4+2]=(uchar)(v>>8);  y_bytes[i*4+3]=(uchar)(v);
    }
    if ((parity != 0) != ((y_bytes[31] & 1) != 0))
        y = field_negate(y);

    p.x = x_fe;
    p.y = y;
    p.z = field_one();
    p.infinity = 0;
    return true;
}

// ORDER big-endian for overflow check
constant uchar RECOVERY_ORDER_BE_METAL[32] = {
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
};

// -- ECDSA sign recoverable ---------------------------------------------------
inline RecoverableSignatureMetal ecdsa_sign_recoverable_metal(
    thread const uchar* msg_hash,
    thread const Scalar256& private_key)
{
    RecoverableSignatureMetal rsig;
    rsig.recid = -1; // indicates failure

    bool is_zero = true;
    for (int j = 0; j < 8; ++j)
        if (private_key.limbs[j] != 0) is_zero = false;
    if (is_zero) return rsig;

    Scalar256 z = scalar_from_bytes(msg_hash);
    Scalar256 k = rfc6979_nonce(private_key, msg_hash);

    {
        bool kz = true;
        for (int j = 0; j < 8; ++j)
            if (k.limbs[j] != 0) kz = false;
        if (kz) return rsig;
    }

    // CT-FIX (C2-class): use ct_generator_mul_metal for secret nonce k — prevents
    // timing side-channel. scalar_mul_generator is variable-time on the scalar.
    CTJacobianPoint R_ct = ct_generator_mul_metal(k);
    if (R_ct.infinity != 0) return rsig;

    FieldElement rx_aff, ry_aff;
    ct_jacobian_to_affine_metal(R_ct, rx_aff, ry_aff);

    uchar rx_bytes[32];
    for (int i = 0; i < 8; ++i) {
        uint v = rx_aff.limbs[7-i];
        rx_bytes[i*4+0]=(uchar)(v>>24); rx_bytes[i*4+1]=(uchar)(v>>16);
        rx_bytes[i*4+2]=(uchar)(v>>8);  rx_bytes[i*4+3]=(uchar)(v);
    }

    Scalar256 r = scalar_from_bytes(rx_bytes);
    {
        bool rz = true;
        for (int j = 0; j < 8; ++j)
            if (r.limbs[j] != 0) rz = false;
        if (rz) return rsig;
    }

    int recid = 0;
    uchar ry_bytes[32];
    for (int i = 0; i < 8; ++i) {
        uint v = ry_aff.limbs[7-i];
        ry_bytes[i*4+0]=(uchar)(v>>24); ry_bytes[i*4+1]=(uchar)(v>>16);
        ry_bytes[i*4+2]=(uchar)(v>>8);  ry_bytes[i*4+3]=(uchar)(v);
    }
    if (ry_bytes[31] & 1) recid |= 1;

    // CT-FIX (C3-class): branchless overflow check — no early exit on nonce-derived bytes.
    // Early `break` leaks nonce bits via iteration count (timing side-channel).
    {
        uint gt = 0u, eq_run = 1u;
        for (int i = 0; i < 32; ++i) {
            uint rb = (uint)rx_bytes[i];
            uint ob = (uint)RECOVERY_ORDER_BE_METAL[i];
            uint byte_gt = ((ob - rb) >> 31) & 1u;
            uint byte_lt = ((rb - ob) >> 31) & 1u;
            gt     = gt | (eq_run & byte_gt);
            eq_run = eq_run & (1u - byte_gt) & (1u - byte_lt);
        }
        if (gt != 0) recid |= 2;
    }

    Scalar256 k_inv = scalar_inverse(k);
    Scalar256 rd = scalar_mul_mod_n(r, private_key);
    Scalar256 z_plus_rd = scalar_add_mod_n(z, rd);
    Scalar256 s = scalar_mul_mod_n(k_inv, z_plus_rd);

    {
        bool sz = true;
        for (int j = 0; j < 8; ++j)
            if (s.limbs[j] != 0) sz = false;
        if (sz) return rsig;
    }

    // Low-S normalization: correct half-order comparison (not odd/even parity)
    if (!scalar_is_low_s(s)) {
        s = scalar_negate(s);
        recid ^= 1;
    }

    rsig.r = r;
    rsig.s = s;
    rsig.recid = recid;
    return rsig;
}

// -- ECDSA recover: Q = r^-1 * (s*R - z*G) -----------------------------------
inline bool ecdsa_recover_metal(thread const uchar* msg_hash,
                                 thread const Scalar256& sig_r,
                                 thread const Scalar256& sig_s,
                                 int recid,
                                 thread JacobianPoint& Q)
{
    if (recid < 0 || recid > 3) return false;

    bool rz = true, sz = true;
    for (int j = 0; j < 8; ++j) {
        if (sig_r.limbs[j] != 0) rz = false;
        if (sig_s.limbs[j] != 0) sz = false;
    }
    if (rz || sz) return false;

    // Reconstruct R.x
    FieldElement rx_fe;
    {
        uchar r_bytes[32];
        scalar_to_bytes(sig_r, r_bytes);
        for (int i = 0; i < 8; ++i) {
            uint limb = 0;
            int base = (7 - i) * 4;
            limb = ((uint)r_bytes[base] << 24) | ((uint)r_bytes[base+1] << 16) |
                   ((uint)r_bytes[base+2] << 8) | (uint)r_bytes[base+3];
            rx_fe.limbs[i] = limb;
        }
        if (recid & 2) {
            FieldElement n_fe;
            for (int j = 0; j < 8; ++j) n_fe.limbs[j] = SECP256K1_N[j];
            rx_fe = field_add(rx_fe, n_fe);
        }
    }

    int y_parity = recid & 1;
    JacobianPoint R;
    if (!lift_x_parity_metal(rx_fe, y_parity, R)) return false;

    Scalar256 z_msg = scalar_from_bytes(msg_hash);
    Scalar256 r_inv = scalar_inverse(sig_r);

    JacobianPoint sR = scalar_mul(R, sig_s);
    JacobianPoint zG = scalar_mul_generator(z_msg);

    // Negate zG
    zG.y = field_negate(zG.y);

    JacobianPoint sR_minus_zG = point_add(sR, zG);
    Q = scalar_mul(sR_minus_zG, r_inv);

    if (Q.infinity) return false;
    return true;
}

#endif // SECP256K1_RECOVERY_H
