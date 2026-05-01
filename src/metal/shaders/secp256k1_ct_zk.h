// =============================================================================
// secp256k1_ct_zk.h -- Constant-time ZK proof operations for Metal
// =============================================================================
// CT knowledge proofs, DLEQ proofs, range proof (single-thread portable).
// All secret-dependent scalar/point ops use CT primitives (8x32 limbs).
// Requires: secp256k1_zk.h, secp256k1_ct_ops.h, secp256k1_ct_field.h,
//           secp256k1_ct_scalar.h, secp256k1_ct_point.h, secp256k1_ct_sign.h
// =============================================================================

#ifndef SECP256K1_CT_ZK_H
#define SECP256K1_CT_ZK_H

// ---------------------------------------------------------------------------
// Helper: CT Jacobian to compressed bytes
// ---------------------------------------------------------------------------
inline void ct_jac_to_compressed_metal(thread const CTJacobianPointMetal& p,
                                       thread uchar* out) {
    auto af = ct_jacobian_to_affine_metal(p);
    uint y_odd_bit = af.second.limbs[0] & 1;
    out[0] = y_odd_bit ? 0x03 : 0x02;
    // Serialize x in big-endian (8x32 LE -> big-endian bytes)
    for (int i = 0; i < 8; ++i) {
        uint v = af.first.limbs[7 - i];
        out[1 + i * 4 + 0] = (uchar)(v >> 24);
        out[1 + i * 4 + 1] = (uchar)(v >> 16);
        out[1 + i * 4 + 2] = (uchar)(v >> 8);
        out[1 + i * 4 + 3] = (uchar)(v);
    }
}

// ---------------------------------------------------------------------------
// CT nonce derivation for ZK proofs
// ---------------------------------------------------------------------------
inline Scalar256 ct_zk_derive_nonce_metal(thread const Scalar256& secret,
                                          thread const CTJacobianPointMetal& pubkey,
                                          thread const uchar* msg,
                                          thread const uchar* aux) {
    uchar sec_bytes[32];
    scalar_to_bytes(secret, sec_bytes);

    uchar pk_comp[33];
    ct_jac_to_compressed_metal(pubkey, pk_comp);

    uchar buf[129]; // 32+33+32+32
    for (int i = 0; i < 32; ++i) buf[i] = sec_bytes[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = pk_comp[i];
    for (int i = 0; i < 32; ++i) buf[65 + i] = msg[i];
    for (int i = 0; i < 32; ++i) buf[97 + i] = aux[i];

    uchar hash[32];
    uchar tag[] = {'Z','K','/','n','o','n','c','e'};
    zk_tagged_hash(tag, 8, buf, 129, hash);
    return scalar_from_bytes(hash);
}

// ---------------------------------------------------------------------------
// CT Knowledge Proof: proves knowledge of s such that P = s*B
// ---------------------------------------------------------------------------
struct CTZKKnowledgeProof {
    uchar rx[32];
    Scalar256 s;
};

inline CTZKKnowledgeProof ct_knowledge_prove_metal(
    thread const Scalar256& secret,
    thread const CTJacobianPointMetal& pubkey,
    thread const CTJacobianPointMetal& base,
    thread const uchar* msg,
    thread const uchar* aux)
{
    CTZKKnowledgeProof proof;

    // Deterministic nonce
    Scalar256 k = ct_zk_derive_nonce_metal(secret, pubkey, msg, aux);

    // R = k * base (CT)
    CTJacobianPointMetal R = ct_scalar_mul_point(base, k);

    // Convert R to affine
    auto r_af = ct_jacobian_to_affine_metal(R);
    FieldElement rx_fe = r_af.first;
    FieldElement ry_fe = r_af.second;

    // Even Y: negate k if Y is odd (branchless)
    uint ry_odd = ry_fe.limbs[0] & 1;
    uint mask = ct_bool_to_mask(ry_odd);
    Scalar256 neg_k = ct_scalar_negate_metal(k);
    k = ct_scalar_select_metal(k, neg_k, mask);

    // Serialize R.x in big-endian
    for (int i = 0; i < 8; ++i) {
        uint v = rx_fe.limbs[7 - i];
        proof.rx[i * 4 + 0] = (uchar)(v >> 24);
        proof.rx[i * 4 + 1] = (uchar)(v >> 16);
        proof.rx[i * 4 + 2] = (uchar)(v >> 8);
        proof.rx[i * 4 + 3] = (uchar)(v);
    }

    // Challenge: e = H("ZK/knowledge" || R.x || P_comp || B_comp || msg)
    uchar p_comp[33], b_comp[33];
    ct_jac_to_compressed_metal(pubkey, p_comp);
    ct_jac_to_compressed_metal(base, b_comp);

    uchar buf[130]; // 32+33+33+32
    for (int i = 0; i < 32; ++i) buf[i] = proof.rx[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[65 + i] = b_comp[i];
    for (int i = 0; i < 32; ++i) buf[98 + i] = msg[i];

    uchar e_hash[32];
    uchar tag[] = {'Z','K','/','k','n','o','w','l','e','d','g','e'};
    zk_tagged_hash(tag, 12, buf, 130, e_hash);
    Scalar256 e = scalar_from_bytes(e_hash);

    // s = k + e * secret (CT)
    Scalar256 e_sec = ct_scalar_mul_metal(e, secret);
    proof.s = ct_scalar_add_metal(k, e_sec);

    return proof;
}

// CT Knowledge Proof for generator G
inline CTZKKnowledgeProof ct_knowledge_prove_generator_metal(
    thread const Scalar256& secret,
    thread const uchar* msg,
    thread const uchar* aux)
{
    CTJacobianPointMetal P = ct_generator_mul_metal(secret);

    // G as CTJacobianPointMetal
    CTJacobianPointMetal G;
    G.x = SECP256K1_GX;
    G.y = SECP256K1_GY;
    G.z = field_one();
    G.infinity = 0;

    return ct_knowledge_prove_metal(secret, P, G, msg, aux);
}

// ---------------------------------------------------------------------------
// CT DLEQ Proof: proves same discrete log across two bases
// ---------------------------------------------------------------------------
struct CTZKDLEQProof {
    uchar r1x[32];
    uchar r2x[32];
    Scalar256 s;
};

inline CTZKDLEQProof ct_dleq_prove_metal(
    thread const Scalar256& secret,
    thread const CTJacobianPointMetal& base_g,
    thread const CTJacobianPointMetal& base_h,
    thread const CTJacobianPointMetal& pub_g,
    thread const CTJacobianPointMetal& pub_h,
    thread const uchar* msg,
    thread const uchar* aux)
{
    CTZKDLEQProof proof;

    Scalar256 k = ct_zk_derive_nonce_metal(secret, pub_g, msg, aux);

    CTJacobianPointMetal R1 = ct_scalar_mul_point(base_g, k);
    CTJacobianPointMetal R2 = ct_scalar_mul_point(base_h, k);

    auto r1_af = ct_jacobian_to_affine_metal(R1);
    auto r2_af = ct_jacobian_to_affine_metal(R2);

    // Serialize R1.x, R2.x
    for (int i = 0; i < 8; ++i) {
        uint v1 = r1_af.first.limbs[7 - i];
        proof.r1x[i*4+0] = (uchar)(v1 >> 24);
        proof.r1x[i*4+1] = (uchar)(v1 >> 16);
        proof.r1x[i*4+2] = (uchar)(v1 >> 8);
        proof.r1x[i*4+3] = (uchar)(v1);

        uint v2 = r2_af.first.limbs[7 - i];
        proof.r2x[i*4+0] = (uchar)(v2 >> 24);
        proof.r2x[i*4+1] = (uchar)(v2 >> 16);
        proof.r2x[i*4+2] = (uchar)(v2 >> 8);
        proof.r2x[i*4+3] = (uchar)(v2);
    }

    // Challenge
    uchar g_comp[33], h_comp[33], p_comp[33], q_comp[33];
    ct_jac_to_compressed_metal(base_g, g_comp);
    ct_jac_to_compressed_metal(base_h, h_comp);
    ct_jac_to_compressed_metal(pub_g, p_comp);
    ct_jac_to_compressed_metal(pub_h, q_comp);

    uchar buf[228]; // 32+32+33*4+32
    int off = 0;
    for (int i = 0; i < 32; ++i) buf[off++] = proof.r1x[i];
    for (int i = 0; i < 32; ++i) buf[off++] = proof.r2x[i];
    for (int i = 0; i < 33; ++i) buf[off++] = g_comp[i];
    for (int i = 0; i < 33; ++i) buf[off++] = h_comp[i];
    for (int i = 0; i < 33; ++i) buf[off++] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[off++] = q_comp[i];
    for (int i = 0; i < 32; ++i) buf[off++] = msg[i];

    uchar e_hash[32];
    uchar tag[] = {'Z','K','/','d','l','e','q'};
    zk_tagged_hash(tag, 7, buf, off, e_hash);
    Scalar256 e = scalar_from_bytes(e_hash);

    Scalar256 e_sec = ct_scalar_mul_metal(e, secret);
    proof.s = ct_scalar_add_metal(k, e_sec);

    return proof;
}

// CT DLEQ for generator G
inline CTZKDLEQProof ct_dleq_prove_generator_metal(
    thread const Scalar256& secret,
    thread const CTJacobianPointMetal& base_h,
    thread const CTJacobianPointMetal& pub_h,
    thread const uchar* msg,
    thread const uchar* aux)
{
    CTJacobianPointMetal P1 = ct_generator_mul_metal(secret);
    CTJacobianPointMetal G;
    G.x = SECP256K1_GX;
    G.y = SECP256K1_GY;
    G.z = field_one();
    G.infinity = 0;
    return ct_dleq_prove_metal(secret, G, base_h, P1, pub_h, msg, aux);
}

// ---------------------------------------------------------------------------
// CT Range Prove (single-thread, portable)
// Produces a full Bulletproof range proof for a committed value.
// Computationally expensive but fully constant-time.
// ---------------------------------------------------------------------------
struct CTRangeProofMetal {
    AffinePoint A;
    AffinePoint S;
    AffinePoint T1;
    AffinePoint T2;
    Scalar256 t_hat;
    Scalar256 tau_x;
    Scalar256 mu;
    AffinePoint L[6];
    AffinePoint R[6];
    Scalar256 a;
    Scalar256 b;
};

inline CTRangeProofMetal ct_range_prove_metal(
    ulong value,
    thread const Scalar256& blinding,
    thread const AffinePoint& commitment,
    thread const AffinePoint& H_gen,
    thread const uchar* aux,
    const device AffinePoint* bp_G,
    const device AffinePoint* bp_H)
{
    CTRangeProofMetal proof;

    Scalar256 ONE_S = scalar_zero();
    ONE_S.limbs[0] = 1;
    Scalar256 ZERO_S = scalar_zero();

    // Bit decomposition (CT)
    Scalar256 a_L[64], a_R[64];
    for (int i = 0; i < 64; ++i) {
        uint bit = (uint)((value >> i) & 1);
        uint mask = ct_bool_to_mask(bit);
        a_L[i] = ct_scalar_select_metal(ZERO_S, ONE_S, mask);
        a_R[i] = ct_scalar_sub_metal(a_L[i], ONE_S);
    }

    // Alpha, rho derivation
    uchar blind_bytes[32];
    scalar_to_bytes(blinding, blind_bytes);
    uchar v_comp[33];
    // Commitment to compressed
    {
        uint y_odd = commitment.y.limbs[0] & 1;
        v_comp[0] = y_odd ? 0x03 : 0x02;
        for (int i = 0; i < 8; ++i) {
            uint v = commitment.x.limbs[7-i];
            v_comp[1+i*4+0] = (uchar)(v>>24);
            v_comp[1+i*4+1] = (uchar)(v>>16);
            v_comp[1+i*4+2] = (uchar)(v>>8);
            v_comp[1+i*4+3] = (uchar)(v);
        }
    }

    uchar alpha_buf[97];
    for (int i = 0; i < 32; ++i) alpha_buf[i] = blind_bytes[i];
    for (int i = 0; i < 33; ++i) alpha_buf[32+i] = v_comp[i];
    for (int i = 0; i < 32; ++i) alpha_buf[65+i] = aux[i];

    uchar alpha_hash[32];
    SHA256Ctx sha_ctx = sha256_init();
    sha_ctx = sha256_update(sha_ctx, alpha_buf, 97);
    sha256_final(sha_ctx, alpha_hash);

    Scalar256 alpha = scalar_from_bytes(alpha_hash);

    uchar rho_hash[32];
    sha_ctx = sha256_init();
    sha_ctx = sha256_update(sha_ctx, alpha_hash, 32);
    sha256_final(sha_ctx, rho_hash);
    Scalar256 rho = scalar_from_bytes(rho_hash);

    // Blinding vectors
    Scalar256 s_L[64], s_R[64];
    for (int i = 0; i < 64; ++i) {
        uchar buf[34];
        for (int j = 0; j < 32; ++j) buf[j] = alpha_hash[j];
        buf[32] = (uchar)i;
        buf[33] = 'L';
        uchar h[32];
        sha_ctx = sha256_init();
        sha_ctx = sha256_update(sha_ctx, buf, 34);
        sha256_final(sha_ctx, h);
        s_L[i] = scalar_from_bytes(h);
        buf[33] = 'R';
        sha_ctx = sha256_init();
        sha_ctx = sha256_update(sha_ctx, buf, 34);
        sha256_final(sha_ctx, h);
        s_R[i] = scalar_from_bytes(h);
    }

    // A = alpha*G + sum(a_L[i]*G_i + a_R[i]*H_i)
    CTJacobianPointMetal A_pt = ct_generator_mul_metal(alpha);
    for (int i = 0; i < 64; ++i) {
        CTJacobianPointMetal Gi_jac;
        Gi_jac.x = bp_G[i].x; Gi_jac.y = bp_G[i].y;
        Gi_jac.z = field_one(); Gi_jac.infinity = 0;

        CTJacobianPointMetal aGi = ct_scalar_mul_point(Gi_jac, a_L[i]);
        JacobianPoint tmp_a = ct_point_to_jacobian(A_pt);
        JacobianPoint tmp_b = ct_point_to_jacobian(aGi);
        JacobianPoint tmp_r = point_add(tmp_a, tmp_b);
        A_pt = ct_point_from_jacobian(tmp_r);

        CTJacobianPointMetal Hi_jac;
        Hi_jac.x = bp_H[i].x; Hi_jac.y = bp_H[i].y;
        Hi_jac.z = field_one(); Hi_jac.infinity = 0;

        CTJacobianPointMetal aHi = ct_scalar_mul_point(Hi_jac, a_R[i]);
        tmp_a = ct_point_to_jacobian(A_pt);
        tmp_b = ct_point_to_jacobian(aHi);
        tmp_r = point_add(tmp_a, tmp_b);
        A_pt = ct_point_from_jacobian(tmp_r);
    }
    {
        auto af = ct_jacobian_to_affine_metal(A_pt);
        proof.A.x = af.first; proof.A.y = af.second;
    }

    // S = rho*G + sum(s_L[i]*G_i + s_R[i]*H_i)
    CTJacobianPointMetal S_pt = ct_generator_mul_metal(rho);
    for (int i = 0; i < 64; ++i) {
        CTJacobianPointMetal Gi_jac;
        Gi_jac.x = bp_G[i].x; Gi_jac.y = bp_G[i].y;
        Gi_jac.z = field_one(); Gi_jac.infinity = 0;

        CTJacobianPointMetal sGi = ct_scalar_mul_point(Gi_jac, s_L[i]);
        JacobianPoint tmp_a = ct_point_to_jacobian(S_pt);
        JacobianPoint tmp_b = ct_point_to_jacobian(sGi);
        JacobianPoint tmp_r = point_add(tmp_a, tmp_b);
        S_pt = ct_point_from_jacobian(tmp_r);

        CTJacobianPointMetal Hi_jac;
        Hi_jac.x = bp_H[i].x; Hi_jac.y = bp_H[i].y;
        Hi_jac.z = field_one(); Hi_jac.infinity = 0;

        CTJacobianPointMetal sHi = ct_scalar_mul_point(Hi_jac, s_R[i]);
        tmp_a = ct_point_to_jacobian(S_pt);
        tmp_b = ct_point_to_jacobian(sHi);
        tmp_r = point_add(tmp_a, tmp_b);
        S_pt = ct_point_from_jacobian(tmp_r);
    }
    {
        auto af = ct_jacobian_to_affine_metal(S_pt);
        proof.S.x = af.first; proof.S.y = af.second;
    }

    // Fiat-Shamir: y, z  (same midstate pattern as fast-path)
    uchar a_comp[33], s_comp[33];
    {
        uint yo = proof.A.y.limbs[0] & 1;
        a_comp[0] = yo ? 0x03 : 0x02;
        for (int i = 0; i < 8; ++i) {
            uint v = proof.A.x.limbs[7-i];
            a_comp[1+i*4+0] = (uchar)(v>>24);
            a_comp[1+i*4+1] = (uchar)(v>>16);
            a_comp[1+i*4+2] = (uchar)(v>>8);
            a_comp[1+i*4+3] = (uchar)(v);
        }
    }
    {
        uint yo = proof.S.y.limbs[0] & 1;
        s_comp[0] = yo ? 0x03 : 0x02;
        for (int i = 0; i < 8; ++i) {
            uint v = proof.S.x.limbs[7-i];
            s_comp[1+i*4+0] = (uchar)(v>>24);
            s_comp[1+i*4+1] = (uchar)(v>>16);
            s_comp[1+i*4+2] = (uchar)(v>>8);
            s_comp[1+i*4+3] = (uchar)(v);
        }
    }

    uchar fs_buf[99];
    for (int i = 0; i < 33; ++i) {
        fs_buf[i]      = a_comp[i];
        fs_buf[33 + i] = s_comp[i];
        fs_buf[66 + i] = v_comp[i];
    }

    uchar y_hash[32], z_hash[32];
    {
        uchar y_tag[] = {'B','P','/','y'};
        zk_tagged_hash(y_tag, 4, fs_buf, 99, y_hash);
        uchar z_tag[] = {'B','P','/','z'};
        zk_tagged_hash(z_tag, 4, fs_buf, 99, z_hash);
    }
    Scalar256 y = scalar_from_bytes(y_hash);
    Scalar256 z = scalar_from_bytes(z_hash);

    // y powers, z^2, 2^i
    Scalar256 y_powers[64];
    y_powers[0] = ONE_S;
    for (int i = 1; i < 64; ++i)
        y_powers[i] = scalar_mul_mod_n(y_powers[i-1], y);

    Scalar256 z2 = scalar_mul_mod_n(z, z);

    Scalar256 two_powers[64];
    two_powers[0] = ONE_S;
    for (int i = 1; i < 64; ++i)
        two_powers[i] = scalar_add_mod_n(two_powers[i-1], two_powers[i-1]);

    // t1, t2 polynomial coefficients
    Scalar256 t1 = ZERO_S, t2 = ZERO_S;
    for (int i = 0; i < 64; ++i) {
        Scalar256 l0_i = ct_scalar_sub_metal(a_L[i], z);
        Scalar256 aR_z = ct_scalar_add_metal(a_R[i], z);
        Scalar256 yi_aRz = ct_scalar_mul_metal(y_powers[i], aR_z);
        Scalar256 z2_2i = ct_scalar_mul_metal(z2, two_powers[i]);
        Scalar256 r0_i = ct_scalar_add_metal(yi_aRz, z2_2i);
        Scalar256 r1_i = scalar_mul_mod_n(y_powers[i], s_R[i]);

        Scalar256 cross1 = ct_scalar_mul_metal(l0_i, r1_i);
        Scalar256 cross2 = ct_scalar_mul_metal(s_L[i], r0_i);
        t1 = ct_scalar_add_metal(t1, ct_scalar_add_metal(cross1, cross2));
        t2 = ct_scalar_add_metal(t2, ct_scalar_mul_metal(s_L[i], r1_i));
    }

    // tau1, tau2
    uchar tau1_hash[32], tau2_hash[32];
    sha_ctx = sha256_init();
    sha_ctx = sha256_update(sha_ctx, rho_hash, 32);
    sha256_final(sha_ctx, tau1_hash);
    sha_ctx = sha256_init();
    sha_ctx = sha256_update(sha_ctx, tau1_hash, 32);
    sha256_final(sha_ctx, tau2_hash);
    Scalar256 tau1 = scalar_from_bytes(tau1_hash);
    Scalar256 tau2 = scalar_from_bytes(tau2_hash);

    // T1 = t1*H + tau1*G, T2 = t2*H + tau2*G
    CTJacobianPointMetal H_jac;
    H_jac.x = H_gen.x; H_jac.y = H_gen.y;
    H_jac.z = field_one(); H_jac.infinity = 0;

    {
        CTJacobianPointMetal t1H = ct_scalar_mul_point(H_jac, t1);
        CTJacobianPointMetal tau1G = ct_generator_mul_metal(tau1);
        JacobianPoint ja = ct_point_to_jacobian(t1H);
        JacobianPoint jb = ct_point_to_jacobian(tau1G);
        JacobianPoint jr = point_add(ja, jb);
        CTJacobianPointMetal T1 = ct_point_from_jacobian(jr);
        auto af = ct_jacobian_to_affine_metal(T1);
        proof.T1.x = af.first; proof.T1.y = af.second;
    }
    {
        CTJacobianPointMetal t2H = ct_scalar_mul_point(H_jac, t2);
        CTJacobianPointMetal tau2G = ct_generator_mul_metal(tau2);
        JacobianPoint ja = ct_point_to_jacobian(t2H);
        JacobianPoint jb = ct_point_to_jacobian(tau2G);
        JacobianPoint jr = point_add(ja, jb);
        CTJacobianPointMetal T2 = ct_point_from_jacobian(jr);
        auto af = ct_jacobian_to_affine_metal(T2);
        proof.T2.x = af.first; proof.T2.y = af.second;
    }

    // Fiat-Shamir: x
    uchar t1_comp[33], t2_comp[33];
    {
        uint yo = proof.T1.y.limbs[0] & 1;
        t1_comp[0] = yo ? 0x03 : 0x02;
        for (int i = 0; i < 8; ++i) {
            uint v = proof.T1.x.limbs[7-i];
            t1_comp[1+i*4+0] = (uchar)(v>>24);
            t1_comp[1+i*4+1] = (uchar)(v>>16);
            t1_comp[1+i*4+2] = (uchar)(v>>8);
            t1_comp[1+i*4+3] = (uchar)(v);
        }
    }
    {
        uint yo = proof.T2.y.limbs[0] & 1;
        t2_comp[0] = yo ? 0x03 : 0x02;
        for (int i = 0; i < 8; ++i) {
            uint v = proof.T2.x.limbs[7-i];
            t2_comp[1+i*4+0] = (uchar)(v>>24);
            t2_comp[1+i*4+1] = (uchar)(v>>16);
            t2_comp[1+i*4+2] = (uchar)(v>>8);
            t2_comp[1+i*4+3] = (uchar)(v);
        }
    }

    uchar x_buf[130];
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33+i] = t2_comp[i]; }
    scalar_to_bytes(y, x_buf + 66);
    scalar_to_bytes(z, x_buf + 98);

    uchar x_hash[32];
    {
        uchar x_tag[] = {'B','P','/','x'};
        zk_tagged_hash(x_tag, 4, x_buf, 130, x_hash);
    }
    Scalar256 xx = scalar_from_bytes(x_hash);

    // Evaluate l(x), r(x), t_hat
    Scalar256 t_hat = ZERO_S;
    Scalar256 l_x[64], r_x[64];
    for (int i = 0; i < 64; ++i) {
        Scalar256 aL_z = ct_scalar_sub_metal(a_L[i], z);
        Scalar256 sL_x = ct_scalar_mul_metal(s_L[i], xx);
        l_x[i] = ct_scalar_add_metal(aL_z, sL_x);

        Scalar256 aR_z = ct_scalar_add_metal(a_R[i], z);
        Scalar256 sR_x = ct_scalar_mul_metal(s_R[i], xx);
        Scalar256 aR_z_sR_x = ct_scalar_add_metal(aR_z, sR_x);
        Scalar256 yi_term = ct_scalar_mul_metal(y_powers[i], aR_z_sR_x);
        Scalar256 z2_2i = ct_scalar_mul_metal(z2, two_powers[i]);
        r_x[i] = ct_scalar_add_metal(yi_term, z2_2i);

        t_hat = ct_scalar_add_metal(t_hat, ct_scalar_mul_metal(l_x[i], r_x[i]));
    }
    proof.t_hat = t_hat;

    // tau_x = tau2*x^2 + tau1*x + z^2*blinding
    Scalar256 xx2 = ct_scalar_mul_metal(xx, xx);
    Scalar256 tau_x = ct_scalar_add_metal(
        ct_scalar_add_metal(ct_scalar_mul_metal(tau2, xx2),
                            ct_scalar_mul_metal(tau1, xx)),
        ct_scalar_mul_metal(z2, blinding));
    proof.tau_x = tau_x;

    // mu = alpha + rho*x
    proof.mu = ct_scalar_add_metal(alpha, ct_scalar_mul_metal(rho, xx));

    // Inner Product Argument (6 rounds for n=64)
    Scalar256 a_vec[64], b_vec[64];
    for (int i = 0; i < 64; ++i) { a_vec[i] = l_x[i]; b_vec[i] = r_x[i]; }

    // Modified generators: H'_i = y^{-i} * H_i
    Scalar256 y_inv = ct_scalar_inverse_metal(y);
    Scalar256 y_inv_pow = ONE_S;

    JacobianPoint G_vec[64], H_vec_mod[64];
    for (int i = 0; i < 64; ++i) {
        G_vec[i].x = bp_G[i].x;
        G_vec[i].y = bp_G[i].y;
        G_vec[i].z = field_one();
        G_vec[i].infinity = 0;

        JacobianPoint Hi_jac;
        Hi_jac.x = bp_H[i].x; Hi_jac.y = bp_H[i].y;
        Hi_jac.z = field_one(); Hi_jac.infinity = 0;
        H_vec_mod[i] = scalar_mul(Hi_jac, y_inv_pow);
        y_inv_pow = scalar_mul_mod_n(y_inv_pow, y_inv);
    }

    int n = 64;
    for (int round = 0; round < 6; ++round) {
        n /= 2;
        Scalar256 c_L = ZERO_S, c_R = ZERO_S;
        JacobianPoint L_pt, R_pt;
        L_pt.infinity = 1; L_pt.z = field_one();
        R_pt = L_pt;

        for (int i = 0; i < n; ++i) {
            JacobianPoint aG = scalar_mul(G_vec[n+i], a_vec[i]);
            L_pt = point_add(L_pt, aG);
            JacobianPoint bH = scalar_mul(H_vec_mod[i], b_vec[n+i]);
            L_pt = point_add(L_pt, bH);
            c_L = scalar_add_mod_n(c_L, scalar_mul_mod_n(a_vec[i], b_vec[n+i]));

            aG = scalar_mul(G_vec[i], a_vec[n+i]);
            R_pt = point_add(R_pt, aG);
            bH = scalar_mul(H_vec_mod[n+i], b_vec[i]);
            R_pt = point_add(R_pt, bH);
            c_R = scalar_add_mod_n(c_R, scalar_mul_mod_n(a_vec[n+i], b_vec[i]));
        }

        JacobianPoint H_jac_std;
        H_jac_std.x = H_gen.x; H_jac_std.y = H_gen.y;
        H_jac_std.z = field_one(); H_jac_std.infinity = 0;

        L_pt = point_add(L_pt, scalar_mul(H_jac_std, c_L));
        R_pt = point_add(R_pt, scalar_mul(H_jac_std, c_R));

        // L,R to affine -> store in proof
        {
            FieldElement zi = field_inv(L_pt.z);
            FieldElement zi2 = field_sqr(zi);
            FieldElement zi3 = field_mul(zi, zi2);
            proof.L[round].x = field_mul(L_pt.x, zi2);
            proof.L[round].y = field_mul(L_pt.y, zi3);
        }
        {
            FieldElement zi = field_inv(R_pt.z);
            FieldElement zi2 = field_sqr(zi);
            FieldElement zi3 = field_mul(zi, zi2);
            proof.R[round].x = field_mul(R_pt.x, zi2);
            proof.R[round].y = field_mul(R_pt.y, zi3);
        }

        // Fiat-Shamir x_round
        uchar l_comp[33], r_comp[33];
        {
            uint yo = proof.L[round].y.limbs[0] & 1;
            l_comp[0] = yo ? 0x03 : 0x02;
            for (int ii = 0; ii < 8; ++ii) {
                uint v = proof.L[round].x.limbs[7-ii];
                l_comp[1+ii*4+0] = (uchar)(v>>24);
                l_comp[1+ii*4+1] = (uchar)(v>>16);
                l_comp[1+ii*4+2] = (uchar)(v>>8);
                l_comp[1+ii*4+3] = (uchar)(v);
            }
        }
        {
            uint yo = proof.R[round].y.limbs[0] & 1;
            r_comp[0] = yo ? 0x03 : 0x02;
            for (int ii = 0; ii < 8; ++ii) {
                uint v = proof.R[round].x.limbs[7-ii];
                r_comp[1+ii*4+0] = (uchar)(v>>24);
                r_comp[1+ii*4+1] = (uchar)(v>>16);
                r_comp[1+ii*4+2] = (uchar)(v>>8);
                r_comp[1+ii*4+3] = (uchar)(v);
            }
        }

        uchar ip_buf[66];
        for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33+i] = r_comp[i]; }

        uchar xr_hash[32];
        uchar ip_tag[] = {'B','P','/','i','p'};
        zk_tagged_hash(ip_tag, 5, ip_buf, 66, xr_hash);
        Scalar256 x_r = scalar_from_bytes(xr_hash);
        Scalar256 x_r_inv = ct_scalar_inverse_metal(x_r);

        // Fold vectors
        for (int i = 0; i < n; ++i) {
            Scalar256 a_new = ct_scalar_add_metal(
                ct_scalar_mul_metal(a_vec[i], x_r),
                ct_scalar_mul_metal(a_vec[n+i], x_r_inv));
            Scalar256 b_new = ct_scalar_add_metal(
                ct_scalar_mul_metal(b_vec[i], x_r_inv),
                ct_scalar_mul_metal(b_vec[n+i], x_r));
            a_vec[i] = a_new;
            b_vec[i] = b_new;
        }
        for (int i = 0; i < n; ++i) {
            JacobianPoint G_new = point_add(
                scalar_mul(G_vec[i], x_r_inv),
                scalar_mul(G_vec[n+i], x_r));
            JacobianPoint H_new = point_add(
                scalar_mul(H_vec_mod[i], x_r),
                scalar_mul(H_vec_mod[n+i], x_r_inv));
            G_vec[i] = G_new;
            H_vec_mod[i] = H_new;
        }
    }

    proof.a = a_vec[0];
    proof.b = b_vec[0];
    return proof;
}

#endif // SECP256K1_CT_ZK_H
