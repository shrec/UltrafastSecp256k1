// =============================================================================
// secp256k1_recovery.cl -- ECDSA Public Key Recovery for OpenCL
// =============================================================================
// - ecdsa_sign_recoverable_impl: sign with recovery ID
// - ecdsa_recover_impl: recover public key from signature + recid
// =============================================================================

#ifndef SECP256K1_RECOVERY_CL
#define SECP256K1_RECOVERY_CL

typedef struct {
    Scalar r;
    Scalar s;
    int recid;
} RecoverableSignatureOCL;

// -- lift_x to curve point with given parity ----------------------------------
inline int lift_x_parity_impl(const FieldElement* x_fe, int parity, JacobianPoint* p) {
    FieldElement x2, x3, y2, y;
    field_sqr_impl(&x2, x_fe);
    field_mul_impl(&x3, &x2, x_fe);

    FieldElement seven;
    for (int i = 0; i < 4; ++i) seven.limbs[i] = 0;
    seven.limbs[0] = 7;
    field_add_impl(&y2, &x3, &seven);

    field_sqrt_impl(&y, &y2);

    // Verify
    FieldElement check;
    field_sqr_impl(&check, &y);
    uchar cb[32], yb[32];
    for (int i = 0; i < 4; ++i) {
        ulong v1 = check.limbs[3-i], v2 = y2.limbs[3-i];
        for (int j = 0; j < 8; ++j) {
            cb[i*8+j] = (uchar)(v1 >> (56-j*8));
            yb[i*8+j] = (uchar)(v2 >> (56-j*8));
        }
    }
    for (int i = 0; i < 32; ++i)
        if (cb[i] != yb[i]) return 0;

    // Adjust parity
    uchar y_bytes[32];
    for (int i = 0; i < 4; ++i) {
        ulong v = y.limbs[3-i];
        for (int j = 0; j < 8; ++j)
            y_bytes[i*8+j] = (uchar)(v >> (56-j*8));
    }
    int y_is_odd = (y_bytes[31] & 1) != 0;
    if ((parity != 0) != y_is_odd) {
        FieldElement neg_y;
        field_neg_impl(&neg_y, &y);
        y = neg_y;
    }

    p->x = *x_fe;
    p->y = y;
    p->z.limbs[0] = 1; p->z.limbs[1] = 0;
    p->z.limbs[2] = 0; p->z.limbs[3] = 0;
    p->infinity = 0;
    return 1;
}

// ORDER in big-endian for comparison
__constant uchar RECOVERY_ORDER_BE[32] = {
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
};

// -- ECDSA sign recoverable ---------------------------------------------------
inline int ecdsa_sign_recoverable_impl(const uchar msg_hash[32],
                                        const Scalar* private_key,
                                        RecoverableSignatureOCL* rsig)
{
    // Check private key nonzero
    int is_zero = 1;
    for (int j = 0; j < 4; ++j)
        if (private_key->limbs[j] != 0) is_zero = 0;
    if (is_zero) return 0;

    Scalar z;
    scalar_from_bytes_impl(msg_hash, &z);

    Scalar k;
    rfc6979_nonce_impl(private_key, msg_hash, &k);
    {
        int kz = 1;
        for (int j = 0; j < 4; ++j)
            if (k.limbs[j] != 0) kz = 0;
        if (kz) return 0;
    }

    // R = k * G
    JacobianPoint R;
    scalar_mul_impl(&R, &GENERATOR_POINT, &k);
    if (R.infinity) return 0;

    FieldElement z_inv, z_inv2, z_inv3, rx_aff, ry_aff;
    field_inv_impl(&z_inv, &R.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv, &z_inv2);
    field_mul_impl(&rx_aff, &R.x, &z_inv2);
    field_mul_impl(&ry_aff, &R.y, &z_inv3);

    uchar rx_bytes[32];
    for (int i = 0; i < 4; ++i) {
        ulong v = rx_aff.limbs[3-i];
        for (int j = 0; j < 8; ++j)
            rx_bytes[i*8+j] = (uchar)(v >> (56-j*8));
    }

    Scalar r;
    scalar_from_bytes_impl(rx_bytes, &r);
    {
        int rz = 1;
        for (int j = 0; j < 4; ++j)
            if (r.limbs[j] != 0) rz = 0;
        if (rz) return 0;
    }

    // Recovery ID
    int recid = 0;
    uchar ry_bytes[32];
    for (int i = 0; i < 4; ++i) {
        ulong v = ry_aff.limbs[3-i];
        for (int j = 0; j < 8; ++j)
            ry_bytes[i*8+j] = (uchar)(v >> (56-j*8));
    }
    if (ry_bytes[31] & 1) recid |= 1;

    // Check overflow (R.x >= n)
    int overflow = 0;
    for (int i = 0; i < 32; ++i) {
        if (rx_bytes[i] < RECOVERY_ORDER_BE[i]) break;
        if (rx_bytes[i] > RECOVERY_ORDER_BE[i]) { overflow = 1; break; }
    }
    if (overflow) recid |= 2;

    // s = k^-1 * (z + r*d) mod n
    Scalar k_inv;
    scalar_inverse_impl(&k_inv, &k);

    Scalar rd;
    scalar_mul_mod_n_impl(&rd, &r, private_key);

    Scalar z_plus_rd;
    scalar_add_mod_n_impl(&z_plus_rd, &z, &rd);

    Scalar s;
    scalar_mul_mod_n_impl(&s, &k_inv, &z_plus_rd);

    {
        int sz = 1;
        for (int j = 0; j < 4; ++j)
            if (s.limbs[j] != 0) sz = 0;
        if (sz) return 0;
    }

    // Normalize low-S
    if (!scalar_is_even_impl(&s)) {
        scalar_negate_impl(&s, &s);
        recid ^= 1;
    }

    rsig->r = r;
    rsig->s = s;
    rsig->recid = recid;
    return 1;
}

// -- ECDSA public key recovery ------------------------------------------------
// Q = r^-1 * (s*R - z*G)
inline int ecdsa_recover_impl(const uchar msg_hash[32],
                               const Scalar* sig_r,
                               const Scalar* sig_s,
                               int recid,
                               JacobianPoint* Q)
{
    if (recid < 0 || recid > 3) return 0;

    // Check r, s nonzero
    {
        int rz = 1, sz = 1;
        for (int j = 0; j < 4; ++j) {
            if (sig_r->limbs[j] != 0) rz = 0;
            if (sig_s->limbs[j] != 0) sz = 0;
        }
        if (rz || sz) return 0;
    }

    // Reconstruct R.x
    FieldElement rx_fe;
    {
        uchar r_bytes[32];
        scalar_to_bytes_impl(sig_r, r_bytes);
        for (int i = 0; i < 4; ++i) {
            ulong limb = 0;
            int base = (3 - i) * 8;
            for (int j = 0; j < 8; ++j)
                limb = (limb << 8) | (ulong)r_bytes[base + j];
            rx_fe.limbs[i] = limb;
        }
        if (recid & 2) {
            FieldElement n_fe;
            n_fe.limbs[0] = SECP256K1_N0;
            n_fe.limbs[1] = SECP256K1_N1;
            n_fe.limbs[2] = SECP256K1_N2;
            n_fe.limbs[3] = SECP256K1_N3;
            FieldElement tmp;
            field_add_impl(&tmp, &rx_fe, &n_fe);
            rx_fe = tmp;
        }
    }

    // Lift x to R
    int y_parity = recid & 1;
    JacobianPoint R;
    if (!lift_x_parity_impl(&rx_fe, y_parity, &R)) return 0;

    // Q = r^-1 * (s*R - z*G)
    Scalar z_msg;
    scalar_from_bytes_impl(msg_hash, &z_msg);

    Scalar r_inv;
    scalar_inverse_impl(&r_inv, sig_r);

    JacobianPoint sR;
    scalar_mul_impl(&sR, &R, sig_s);

    JacobianPoint zG;
    scalar_mul_impl(&zG, &GENERATOR_POINT, &z_msg);

    // Negate zG
    FieldElement neg_zy;
    field_neg_impl(&neg_zy, &zG.y);
    zG.y = neg_zy;

    // sR + (-zG)
    JacobianPoint sR_minus_zG;
    point_add_impl(&sR_minus_zG, &sR, &zG);

    // Q = r_inv * (sR - zG)
    scalar_mul_impl(Q, &sR_minus_zG, &r_inv);

    if (Q->infinity) return 0;
    return 1;
}

#endif // SECP256K1_RECOVERY_CL
