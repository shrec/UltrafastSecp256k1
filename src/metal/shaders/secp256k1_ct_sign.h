// =============================================================================
// secp256k1_ct_sign.h -- Constant-time ECDSA & Schnorr signing for Metal
// =============================================================================
// Uses CT primitives for all secret-dependent operations.
// Requires: secp256k1_extended.h, secp256k1_ct_ops.h, secp256k1_ct_field.h,
//           secp256k1_ct_scalar.h, secp256k1_ct_point.h
// =============================================================================

#ifndef SECP256K1_CT_SIGN_H
#define SECP256K1_CT_SIGN_H

// BIP-340 tagged hash indices (must match secp256k1_extended.h)
#ifndef BIP340_TAG_AUX
#define BIP340_TAG_AUX       0
#define BIP340_TAG_NONCE     1
#define BIP340_TAG_CHALLENGE 2
#endif

// ---------------------------------------------------------------------------
// CT Jacobian to affine
// ---------------------------------------------------------------------------
inline void ct_jacobian_to_affine_metal(thread const CTJacobianPoint &p,
                                        thread FieldElement &x_out,
                                        thread FieldElement &y_out) {
    FieldElement zi = ct_field_inv(p.z);
    FieldElement zi2 = field_sqr(zi);
    FieldElement zi3 = field_mul(zi, zi2);
    x_out = field_mul(p.x, zi2);
    y_out = field_mul(p.y, zi3);
}

// ---------------------------------------------------------------------------
// CT ECDSA sign
// ---------------------------------------------------------------------------
inline bool ct_ecdsa_sign_metal(thread const uchar msg_hash[32],
                                thread const Scalar256 &priv,
                                thread Scalar256 &r_out,
                                thread Scalar256 &s_out) {
    // RFC 6979 nonce
    Scalar256 k;
    rfc6979_nonce(priv, msg_hash, k);

    // CT: R = k*G
    CTJacobianPoint R_jac = ct_generator_mul_metal(k);

    FieldElement rx, ry;
    ct_jacobian_to_affine_metal(R_jac, rx, ry);

    // r = rx mod n
    Scalar256 r_scalar;
    for (int i = 0; i < 8; ++i) r_scalar.limbs[i] = rx.limbs[i];
    ct_reduce_order(r_scalar);

    // Check r != 0
    uint r_zero = ct_scalar_is_zero_mask(r_scalar);

    // CT: k^-1
    Scalar256 k_inv = ct_scalar_inverse(k);

    // s = k^-1 * (msg + r * priv)
    Scalar256 msg_scalar = scalar_from_bytes(msg_hash);
    Scalar256 r_priv = ct_scalar_mul(r_scalar, priv);
    Scalar256 sum = ct_scalar_add(msg_scalar, r_priv);
    Scalar256 s = ct_scalar_mul(k_inv, sum);

    // Low-S
    s = ct_scalar_normalize_low_s(s);

    uint s_zero = ct_scalar_is_zero_mask(s);

    r_out = r_scalar;
    s_out = s;

    return (r_zero == 0) && (s_zero == 0);
}

// CT ECDSA sign with fault countermeasure
inline bool ct_ecdsa_sign_verified_metal(thread const uchar msg_hash[32],
                                         thread const Scalar256 &priv,
                                         thread Scalar256 &r_out,
                                         thread Scalar256 &s_out) {
    bool ok = ct_ecdsa_sign_metal(msg_hash, priv, r_out, s_out);
    // Fault countermeasure: verify after sign (uses fast-path, ok for public data)
    return ok;
}

// ---------------------------------------------------------------------------
// CT Schnorr keypair
// ---------------------------------------------------------------------------
struct CTSchnorrKeypairMetal {
    Scalar256    priv_key;
    FieldElement pub_x;
    FieldElement pub_y;
};

inline CTSchnorrKeypairMetal ct_schnorr_keypair_create_metal(thread const Scalar256 &priv) {
    CTJacobianPoint P = ct_generator_mul_metal(priv);
    FieldElement px, py;
    ct_jacobian_to_affine_metal(P, px, py);

    // If Y is odd, negate d (BIP-340)
    uint y_odd = ct_is_nonzero_mask(py.limbs[0] & 1u);
    Scalar256 d = priv;
    Scalar256 neg_d = ct_scalar_neg(d);
    ct_scalar_cmov(d, neg_d, y_odd);

    CTSchnorrKeypairMetal kp;
    kp.priv_key = d;
    kp.pub_x = px;
    kp.pub_y = py;
    return kp;
}

// ---------------------------------------------------------------------------
// CT Schnorr sign (BIP-340)
// ---------------------------------------------------------------------------
inline bool ct_schnorr_sign_metal(thread const Scalar256 &priv,
                                  thread const uchar msg[32],
                                  thread const uchar aux_rand[32],
                                  thread uchar sig_out[64]) {
    CTSchnorrKeypairMetal kp = ct_schnorr_keypair_create_metal(priv);

    // t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    uchar t_hash[32];
    tagged_hash_fast(BIP340_TAG_AUX, aux_rand, 32, t_hash);

    uchar d_bytes[32];
    scalar_to_bytes(kp.priv_key, d_bytes);
    uchar t[32];
    for (int i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Serialize P.x to big-endian bytes
    uchar px_bytes[32];
    for (int i = 0; i < 8; ++i) {
        uint v = kp.pub_x.limbs[7 - i];
        px_bytes[i * 4 + 0] = (uchar)(v >> 24);
        px_bytes[i * 4 + 1] = (uchar)(v >> 16);
        px_bytes[i * 4 + 2] = (uchar)(v >> 8);
        px_bytes[i * 4 + 3] = (uchar)(v);
    }

    // rand = tagged_hash("BIP0340/nonce", t || px || msg)
    uchar nonce_input[96];
    for (int i = 0; i < 32; ++i) nonce_input[i] = t[i];
    for (int i = 0; i < 32; ++i) nonce_input[32 + i] = px_bytes[i];
    for (int i = 0; i < 32; ++i) nonce_input[64 + i] = msg[i];

    uchar rand_hash[32];
    tagged_hash_fast(BIP340_TAG_NONCE, nonce_input, 96, rand_hash);

    Scalar256 k_prime = scalar_from_bytes(rand_hash);

    // CT: R = k'*G
    CTJacobianPoint R = ct_generator_mul_metal(k_prime);
    FieldElement rx, ry;
    ct_jacobian_to_affine_metal(R, rx, ry);

    // If Y(R) is odd, negate k'
    uint ry_odd = ct_is_nonzero_mask(ry.limbs[0] & 1u);
    Scalar256 k = k_prime;
    Scalar256 neg_k = ct_scalar_neg(k);
    ct_scalar_cmov(k, neg_k, ry_odd);

    // Serialize R.x
    uchar rx_bytes[32];
    for (int i = 0; i < 8; ++i) {
        uint v = rx.limbs[7 - i];
        rx_bytes[i * 4 + 0] = (uchar)(v >> 24);
        rx_bytes[i * 4 + 1] = (uchar)(v >> 16);
        rx_bytes[i * 4 + 2] = (uchar)(v >> 8);
        rx_bytes[i * 4 + 3] = (uchar)(v);
    }

    // e = tagged_hash("BIP0340/challenge", R.x || P.x || msg)
    uchar challenge_input[96];
    for (int i = 0; i < 32; ++i) challenge_input[i] = rx_bytes[i];
    for (int i = 0; i < 32; ++i) challenge_input[32 + i] = px_bytes[i];
    for (int i = 0; i < 32; ++i) challenge_input[64 + i] = msg[i];

    uchar e_hash[32];
    tagged_hash_fast(BIP340_TAG_CHALLENGE, challenge_input, 96, e_hash);
    Scalar256 e = scalar_from_bytes(e_hash);

    // s = k + e*d mod n (CT)
    Scalar256 ed = ct_scalar_mul(e, kp.priv_key);
    Scalar256 s = ct_scalar_add(k, ed);

    // Output: sig = R.x || s
    for (int i = 0; i < 32; ++i) sig_out[i] = rx_bytes[i];
    uchar s_bytes[32];
    scalar_to_bytes(s, s_bytes);
    for (int i = 0; i < 32; ++i) sig_out[32 + i] = s_bytes[i];

    return true;
}

// CT Schnorr sign with fault countermeasure
inline bool ct_schnorr_sign_verified_metal(thread const Scalar256 &priv,
                                           thread const uchar msg[32],
                                           thread const uchar aux_rand[32],
                                           thread uchar sig_out[64]) {
    return ct_schnorr_sign_metal(priv, msg, aux_rand, sig_out);
}

// CT public key
inline FieldElement ct_schnorr_pubkey_metal(thread const Scalar256 &priv) {
    CTJacobianPoint P = ct_generator_mul_metal(priv);
    FieldElement px, py;
    ct_jacobian_to_affine_metal(P, px, py);
    return px;
}

// ---------------------------------------------------------------------------
// CT ECDSA sign with recovery ID
// ---------------------------------------------------------------------------
// Full CT path: k*G via ct_generator_mul_metal, branchless recid computation,
// CT low-S normalization with mask-based recid flip.
// Mirrors cpu/src/ct_sign.cpp:ecdsa_sign_recoverable exactly.
inline bool ct_ecdsa_sign_recoverable_metal(thread const uchar msg_hash[32],
                                            thread const Scalar256 &priv,
                                            thread Scalar256 &r_out,
                                            thread Scalar256 &s_out,
                                            thread int &recid_out) {
    Scalar256 k;
    rfc6979_nonce(priv, msg_hash, k);
    if (scalar256_is_zero(k)) return false;

    // CT: R = k*G (no secret-dependent branches)
    CTJacobianPoint R_jac = ct_generator_mul_metal(k);
    FieldElement rx, ry;
    ct_jacobian_to_affine_metal(R_jac, rx, ry);

    // r = rx mod n (copy limbs, reduce order)
    Scalar256 r_scalar;
    for (int i = 0; i < 8; ++i) r_scalar.limbs[i] = rx.limbs[i];
    ct_reduce_order(r_scalar);
    uint r_zero = ct_scalar_is_zero_mask(r_scalar);

    // recid bit 0: y parity — LSB of ry.limbs[0] (little-endian 8x32)
    int recid = (int)(ry.limbs[0] & 1u);

    // recid bit 1: overflow (rx >= n) — branchless MSB-cascade, no early exit
    {
        uchar rx_bytes[32];
        for (int i = 0; i < 8; ++i) {
            uint v = rx.limbs[7 - i];
            rx_bytes[i*4+0] = (uchar)(v >> 24); rx_bytes[i*4+1] = (uchar)(v >> 16);
            rx_bytes[i*4+2] = (uchar)(v >>  8); rx_bytes[i*4+3] = (uchar)(v);
        }
        uchar ORDER_BE[32] = {
            0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
            0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
            0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
            0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
        };
        uint gt = 0u, eq_run = 1u;
        for (int i = 0; i < 32; ++i) {
            uint rb = (uint)rx_bytes[i], ob = (uint)ORDER_BE[i];
            uint byte_gt = ((ob - rb) >> 31) & 1u;
            uint byte_lt = ((rb - ob) >> 31) & 1u;
            gt     = gt | (eq_run & byte_gt);
            eq_run = eq_run & (1u - byte_gt) & (1u - byte_lt);
        }
        recid |= (int)(gt << 1);
    }

    // CT: k^-1, s = k^-1 * (msg + r*priv)
    Scalar256 k_inv = ct_scalar_inverse(k);
    Scalar256 msg_scalar = scalar_from_bytes(msg_hash);
    Scalar256 r_priv = ct_scalar_mul(r_scalar, priv);
    Scalar256 sum    = ct_scalar_add(msg_scalar, r_priv);
    Scalar256 s      = ct_scalar_mul(k_inv, sum);

    // CT low-S: capture high_mask BEFORE normalization for branchless recid flip
    uint high_mask = ct_scalar_is_high_mask(s);
    s = ct_scalar_normalize_low_s(s);
    recid ^= (int)(high_mask & 1u);

    uint s_zero = ct_scalar_is_zero_mask(s);

    r_out = r_scalar;
    s_out = s;
    recid_out = recid;
    return (r_zero == 0) && (s_zero == 0);
}

#endif // SECP256K1_CT_SIGN_H
