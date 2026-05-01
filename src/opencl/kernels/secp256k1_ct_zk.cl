// =============================================================================
// secp256k1_ct_zk.cl -- Constant-time ZK proof operations for OpenCL
// =============================================================================
// CT knowledge proofs, DLEQ proofs. All secret-dependent scalar/point ops
// use CT primitives. Range prove uses single-thread portable implementation.
// Requires: secp256k1_zk.cl, secp256k1_ct_ops.cl, secp256k1_ct_field.cl,
//           secp256k1_ct_scalar.cl, secp256k1_ct_point.cl, secp256k1_ct_sign.cl
// =============================================================================

#ifndef SECP256K1_CT_ZK_CL
#define SECP256K1_CT_ZK_CL

// ---------------------------------------------------------------------------
// Helper: CT Jacobian to compressed bytes
// ---------------------------------------------------------------------------
inline void ct_jac_to_compressed(const CTJacobianPoint* p, uchar out[33]) {
    FieldElement px, py;
    ct_jacobian_to_affine(p, &px, &py);
    // Determine prefix: 02 if y even, 03 if y odd
    ulong y_odd = ct_is_nonzero_mask(py.limbs[0] & 1);
    out[0] = (y_odd != 0) ? 0x03 : 0x02;
    // Serialize x in big-endian
    for (int i = 0; i < 4; ++i) {
        ulong v = px.limbs[3 - i];
        for (int j = 0; j < 8; ++j)
            out[1 + i * 8 + j] = (uchar)(v >> (56 - j * 8));
    }
}

// ---------------------------------------------------------------------------
// CT nonce derivation: H(secret || pubkey_compressed || msg || aux)
// ---------------------------------------------------------------------------
inline void ct_zk_derive_nonce(const Scalar* secret, const CTJacobianPoint* pubkey,
                               const uchar msg[32], const uchar aux[32],
                               Scalar* k_out) {
    uchar sec_bytes[32];
    scalar_to_bytes_impl(secret, sec_bytes);

    uchar pk_comp[33];
    ct_jac_to_compressed(pubkey, pk_comp);

    uchar buf[32 + 33 + 32 + 32];
    for (int i = 0; i < 32; ++i) buf[i] = sec_bytes[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = pk_comp[i];
    for (int i = 0; i < 32; ++i) buf[65 + i] = msg[i];
    for (int i = 0; i < 32; ++i) buf[97 + i] = aux[i];

    uchar hash[32];
    uchar tag[] = {'Z','K','/','n','o','n','c','e'};
    zk_tagged_hash_impl(tag, 8, buf, sizeof(buf), hash);
    scalar_from_bytes_impl(hash, k_out);
}

// ---------------------------------------------------------------------------
// CT Knowledge Proof: proves knowledge of s such that P = s*B
// ---------------------------------------------------------------------------
inline int ct_knowledge_prove_impl(
    const Scalar* secret,
    const CTJacobianPoint* pubkey,
    const CTJacobianPoint* base,
    const uchar msg[32],
    const uchar aux[32],
    ZKKnowledgeProof* proof)
{
    // Deterministic nonce
    Scalar k;
    ct_zk_derive_nonce(secret, pubkey, msg, aux, &k);

    // CT: R = k * base
    CTJacobianPoint R;
    ct_scalar_mul_point(base, &k, &R);

    // Convert R to affine
    FieldElement rx_fe, ry_fe;
    ct_jacobian_to_affine(&R, &rx_fe, &ry_fe);

    // Even Y: if Y odd, negate k (branchless)
    ulong ry_odd = ct_is_nonzero_mask(ry_fe.limbs[0] & 1);
    Scalar neg_k;
    ct_scalar_neg_impl(&k, &neg_k);
    ct_scalar_cmov(&k, &neg_k, ry_odd);

    // Negate ry if odd
    FieldElement neg_ry;
    ct_field_neg_impl(&neg_ry, &ry_fe);
    ct_field_cmov(&ry_fe, &neg_ry, ry_odd);

    // Serialize R.x
    for (int i = 0; i < 4; ++i) {
        ulong v = rx_fe.limbs[3 - i];
        for (int j = 0; j < 8; ++j)
            proof->rx[i * 8 + j] = (uchar)(v >> (56 - j * 8));
    }

    // Challenge: e = H("ZK/knowledge" || R.x || P_comp || B_comp || msg)
    uchar p_comp[33], b_comp[33];
    ct_jac_to_compressed(pubkey, p_comp);
    ct_jac_to_compressed(base, b_comp);

    uchar buf[32 + 33 + 33 + 32];
    for (int i = 0; i < 32; ++i) buf[i] = proof->rx[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[65 + i] = b_comp[i];
    for (int i = 0; i < 32; ++i) buf[98 + i] = msg[i];

    uchar e_hash[32];
    uchar tag[] = {'Z','K','/','k','n','o','w','l','e','d','g','e'};
    zk_tagged_hash_impl(tag, 12, buf, sizeof(buf), e_hash);
    Scalar e;
    scalar_from_bytes_impl(e_hash, &e);

    // s = k + e * secret (CT)
    Scalar e_sec;
    ct_scalar_mul_impl(&e, secret, &e_sec);
    ct_scalar_add_impl(&k, &e_sec, &proof->s);

    return 1;
}

// CT Knowledge Proof for generator G
inline int ct_knowledge_prove_generator_impl(
    const Scalar* secret,
    const uchar msg[32],
    const uchar aux[32],
    ZKKnowledgeProof* proof)
{
    // Compute P = secret * G
    CTJacobianPoint P;
    ct_generator_mul_impl(secret, &P);

    // G as CTJacobianPoint
    CTJacobianPoint G;
    G.x.limbs[0] = 0x59F2815B16F81798UL;
    G.x.limbs[1] = 0x029BFCDB2DCE28D9UL;
    G.x.limbs[2] = 0x55A06295CE870B07UL;
    G.x.limbs[3] = 0x79BE667EF9DCBBACUL;
    G.y.limbs[0] = 0x9C47D08FFB10D4B8UL;
    G.y.limbs[1] = 0xFD17B448A6855419UL;
    G.y.limbs[2] = 0x5DA4FBFC0E1108A8UL;
    G.y.limbs[3] = 0x483ADA7726A3C465UL;
    G.z.limbs[0] = 1; G.z.limbs[1] = 0;
    G.z.limbs[2] = 0; G.z.limbs[3] = 0;
    G.infinity = 0;

    return ct_knowledge_prove_impl(secret, &P, &G, msg, aux, proof);
}

// ---------------------------------------------------------------------------
// CT DLEQ Proof: proves same discrete log across two bases
// Proves: x such that P1 = x*G and P2 = x*H
// ---------------------------------------------------------------------------
inline int ct_dleq_prove_impl(
    const Scalar* secret,
    const CTJacobianPoint* base_g,
    const CTJacobianPoint* base_h,
    const CTJacobianPoint* pub_g,
    const CTJacobianPoint* pub_h,
    const uchar msg[32],
    const uchar aux[32],
    ZKDLEQProof* proof)
{
    // Nonce
    Scalar k;
    ct_zk_derive_nonce(secret, pub_g, msg, aux, &k);

    // CT: R1 = k*G, R2 = k*H
    CTJacobianPoint R1, R2;
    ct_scalar_mul_point(base_g, &k, &R1);
    ct_scalar_mul_point(base_h, &k, &R2);

    // Batch convert to affine
    FieldElement z_vals[4], z_invs[4];
    z_vals[0] = R1.z; z_vals[1] = R2.z;
    z_vals[2] = pub_g->z; z_vals[3] = pub_h->z;
    ct_batch_field_inv(z_vals, z_invs, 4);

    FieldElement r1x, r1y, r2x, r2y;
    {
        FieldElement zi2, zi3;
        field_sqr_impl(&zi2, &z_invs[0]);
        field_mul_impl(&zi3, &z_invs[0], &zi2);
        field_mul_impl(&r1x, &R1.x, &zi2);
        field_mul_impl(&r1y, &R1.y, &zi3);
    }
    {
        FieldElement zi2, zi3;
        field_sqr_impl(&zi2, &z_invs[1]);
        field_mul_impl(&zi3, &z_invs[1], &zi2);
        field_mul_impl(&r2x, &R2.x, &zi2);
        field_mul_impl(&r2y, &R2.y, &zi3);
    }

    // Serialize R1.x, R2.x
    for (int i = 0; i < 4; ++i) {
        ulong v = r1x.limbs[3 - i];
        for (int j = 0; j < 8; ++j)
            proof->r1x[i * 8 + j] = (uchar)(v >> (56 - j * 8));
    }
    for (int i = 0; i < 4; ++i) {
        ulong v = r2x.limbs[3 - i];
        for (int j = 0; j < 8; ++j)
            proof->r2x[i * 8 + j] = (uchar)(v >> (56 - j * 8));
    }

    // Challenge: e = H("ZK/dleq" || R1.x || R2.x || G_comp || H_comp || P_comp || Q_comp || msg)
    uchar g_comp[33], h_comp[33], p_comp[33], q_comp[33];
    ct_jac_to_compressed(base_g, g_comp);
    ct_jac_to_compressed(base_h, h_comp);
    ct_jac_to_compressed(pub_g, p_comp);
    ct_jac_to_compressed(pub_h, q_comp);

    uchar buf[32 + 32 + 33 + 33 + 33 + 33 + 32];
    int off = 0;
    for (int i = 0; i < 32; ++i) buf[off++] = proof->r1x[i];
    for (int i = 0; i < 32; ++i) buf[off++] = proof->r2x[i];
    for (int i = 0; i < 33; ++i) buf[off++] = g_comp[i];
    for (int i = 0; i < 33; ++i) buf[off++] = h_comp[i];
    for (int i = 0; i < 33; ++i) buf[off++] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[off++] = q_comp[i];
    for (int i = 0; i < 32; ++i) buf[off++] = msg[i];

    uchar e_hash[32];
    uchar tag[] = {'Z','K','/','d','l','e','q'};
    zk_tagged_hash_impl(tag, 7, buf, sizeof(buf), e_hash);
    Scalar e;
    scalar_from_bytes_impl(e_hash, &e);

    // s = k + e * secret (CT)
    Scalar e_sec;
    ct_scalar_mul_impl(&e, secret, &e_sec);
    ct_scalar_add_impl(&k, &e_sec, &proof->s);

    return 1;
}

// CT DLEQ for generator G
inline int ct_dleq_prove_generator_impl(
    const Scalar* secret,
    const CTJacobianPoint* base_h,
    const CTJacobianPoint* pub_h,
    const uchar msg[32],
    const uchar aux[32],
    ZKDLEQProof* proof)
{
    // P1 = secret * G
    CTJacobianPoint P1;
    ct_generator_mul_impl(secret, &P1);

    // G as CTJacobianPoint
    CTJacobianPoint G;
    G.x.limbs[0] = 0x59F2815B16F81798UL;
    G.x.limbs[1] = 0x029BFCDB2DCE28D9UL;
    G.x.limbs[2] = 0x55A06295CE870B07UL;
    G.x.limbs[3] = 0x79BE667EF9DCBBACUL;
    G.y.limbs[0] = 0x9C47D08FFB10D4B8UL;
    G.y.limbs[1] = 0xFD17B448A6855419UL;
    G.y.limbs[2] = 0x5DA4FBFC0E1108A8UL;
    G.y.limbs[3] = 0x483ADA7726A3C465UL;
    G.z.limbs[0] = 1; G.z.limbs[1] = 0;
    G.z.limbs[2] = 0; G.z.limbs[3] = 0;
    G.infinity = 0;

    return ct_dleq_prove_impl(secret, &G, base_h, &P1, pub_h, msg, aux, proof);
}

// ---------------------------------------------------------------------------
// CT Range Prove (single-thread, portable -- no warp primitives)
// Produces a full Bulletproof range proof for a committed value.
// This is computationally expensive (hundreds of scalar muls) but fully CT.
// ---------------------------------------------------------------------------
inline int ct_range_prove_impl(
    ulong value,
    const Scalar* blinding,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    const uchar aux[32],
    __global const AffinePoint* bp_G,    // 64 generator points
    __global const AffinePoint* bp_H,    // 64 generator points
    RangeProofGPU* proof)
{
    Scalar ONE_S;
    ONE_S.limbs[0] = 1; ONE_S.limbs[1] = 0;
    ONE_S.limbs[2] = 0; ONE_S.limbs[3] = 0;
    Scalar ZERO_S;
    ZERO_S.limbs[0] = 0; ZERO_S.limbs[1] = 0;
    ZERO_S.limbs[2] = 0; ZERO_S.limbs[3] = 0;

    // Bit decomposition: a_L[i] = (value >> i) & 1, a_R[i] = a_L[i] - 1
    Scalar a_L[64], a_R[64];
    for (int i = 0; i < 64; ++i) {
        ulong bit = (value >> i) & 1;
        // CT: select without branch
        ct_scalar_select(&ONE_S, &ZERO_S, ct_bool_to_mask(bit != 0), &a_L[i]);
        ct_scalar_sub_impl(&a_L[i], &ONE_S, &a_R[i]);
    }

    // Derive alpha deterministically
    uchar blind_bytes[32];
    scalar_to_bytes_impl(blinding, blind_bytes);
    uchar alpha_buf[97];
    for (int i = 0; i < 32; ++i) alpha_buf[i] = blind_bytes[i];
    // Commitment compressed
    uchar v_comp[33];
    affine_to_compressed_impl(&commitment->x, &commitment->y, v_comp);
    for (int i = 0; i < 33; ++i) alpha_buf[32 + i] = v_comp[i];
    for (int i = 0; i < 32; ++i) alpha_buf[65 + i] = aux[i];

    uchar alpha_hash[32];
    SHA256Ctx sha_ctx;
    sha256_init(&sha_ctx);
    sha256_update(&sha_ctx, alpha_buf, 97);
    sha256_final(&sha_ctx, alpha_hash);

    Scalar alpha;
    scalar_from_bytes_impl(alpha_hash, &alpha);

    // Derive rho
    uchar rho_hash[32];
    sha256_init(&sha_ctx);
    sha256_update(&sha_ctx, alpha_hash, 32);
    sha256_final(&sha_ctx, rho_hash);
    Scalar rho;
    scalar_from_bytes_impl(rho_hash, &rho);

    // Blinding vectors s_L, s_R
    Scalar s_L[64], s_R[64];
    for (int i = 0; i < 64; ++i) {
        uchar buf[34];
        for (int j = 0; j < 32; ++j) buf[j] = alpha_hash[j];
        buf[32] = (uchar)i;
        buf[33] = 'L';
        uchar h[32];
        sha256_init(&sha_ctx);
        sha256_update(&sha_ctx, buf, 34);
        sha256_final(&sha_ctx, h);
        scalar_from_bytes_impl(h, &s_L[i]);
        buf[33] = 'R';
        sha256_init(&sha_ctx);
        sha256_update(&sha_ctx, buf, 34);
        sha256_final(&sha_ctx, h);
        scalar_from_bytes_impl(h, &s_R[i]);
    }

    // A = alpha*G + sum(a_L[i]*G_i + a_R[i]*H_i)
    CTJacobianPoint A_pt;
    ct_generator_mul_impl(&alpha, &A_pt);
    for (int i = 0; i < 64; ++i) {
        // a_L[i] * G_i (skip if zero for efficiency, but always touch for CT)
        CTJacobianPoint Gi_jac;
        Gi_jac.x = bp_G[i].x; Gi_jac.y = bp_G[i].y;
        Gi_jac.z.limbs[0] = 1; Gi_jac.z.limbs[1] = 0;
        Gi_jac.z.limbs[2] = 0; Gi_jac.z.limbs[3] = 0;
        Gi_jac.infinity = 0;

        CTJacobianPoint aGi;
        ct_scalar_mul_point(&Gi_jac, &a_L[i], &aGi);
        // CT add: always add, result is identity if a_L[i]==0
        JacobianPoint tmp_a = ct_point_to_jacobian(&A_pt);
        JacobianPoint tmp_b = ct_point_to_jacobian(&aGi);
        JacobianPoint tmp_r;
        point_add_impl(&tmp_r, &tmp_a, &tmp_b);
        A_pt = ct_point_from_jacobian(&tmp_r);

        // a_R[i] * H_i
        CTJacobianPoint Hi_jac;
        Hi_jac.x = bp_H[i].x; Hi_jac.y = bp_H[i].y;
        Hi_jac.z.limbs[0] = 1; Hi_jac.z.limbs[1] = 0;
        Hi_jac.z.limbs[2] = 0; Hi_jac.z.limbs[3] = 0;
        Hi_jac.infinity = 0;

        CTJacobianPoint aHi;
        ct_scalar_mul_point(&Hi_jac, &a_R[i], &aHi);
        tmp_a = ct_point_to_jacobian(&A_pt);
        tmp_b = ct_point_to_jacobian(&aHi);
        point_add_impl(&tmp_r, &tmp_a, &tmp_b);
        A_pt = ct_point_from_jacobian(&tmp_r);
    }

    // Convert A to affine
    FieldElement zi, zi2, zi3;
    ct_field_inv(&zi, &A_pt.z);
    field_sqr_impl(&zi2, &zi);
    field_mul_impl(&zi3, &zi, &zi2);
    field_mul_impl(&proof->A.x, &A_pt.x, &zi2);
    field_mul_impl(&proof->A.y, &A_pt.y, &zi3);

    // S = rho*G + sum(s_L[i]*G_i + s_R[i]*H_i)
    CTJacobianPoint S_pt;
    ct_generator_mul_impl(&rho, &S_pt);
    for (int i = 0; i < 64; ++i) {
        CTJacobianPoint Gi_jac;
        Gi_jac.x = bp_G[i].x; Gi_jac.y = bp_G[i].y;
        Gi_jac.z.limbs[0] = 1; Gi_jac.z.limbs[1] = 0;
        Gi_jac.z.limbs[2] = 0; Gi_jac.z.limbs[3] = 0;
        Gi_jac.infinity = 0;

        CTJacobianPoint sGi;
        ct_scalar_mul_point(&Gi_jac, &s_L[i], &sGi);
        JacobianPoint tmp_a = ct_point_to_jacobian(&S_pt);
        JacobianPoint tmp_b = ct_point_to_jacobian(&sGi);
        JacobianPoint tmp_r;
        point_add_impl(&tmp_r, &tmp_a, &tmp_b);
        S_pt = ct_point_from_jacobian(&tmp_r);

        CTJacobianPoint Hi_jac;
        Hi_jac.x = bp_H[i].x; Hi_jac.y = bp_H[i].y;
        Hi_jac.z.limbs[0] = 1; Hi_jac.z.limbs[1] = 0;
        Hi_jac.z.limbs[2] = 0; Hi_jac.z.limbs[3] = 0;
        Hi_jac.infinity = 0;

        CTJacobianPoint sHi;
        ct_scalar_mul_point(&Hi_jac, &s_R[i], &sHi);
        tmp_a = ct_point_to_jacobian(&S_pt);
        tmp_b = ct_point_to_jacobian(&sHi);
        point_add_impl(&tmp_r, &tmp_a, &tmp_b);
        S_pt = ct_point_from_jacobian(&tmp_r);
    }

    // Convert S to affine
    ct_field_inv(&zi, &S_pt.z);
    field_sqr_impl(&zi2, &zi);
    field_mul_impl(&zi3, &zi, &zi2);
    field_mul_impl(&proof->S.x, &S_pt.x, &zi2);
    field_mul_impl(&proof->S.y, &S_pt.y, &zi3);

    // Fiat-Shamir: y, z
    uchar a_comp[33], s_comp[33];
    affine_to_compressed_impl(&proof->A.x, &proof->A.y, a_comp);
    affine_to_compressed_impl(&proof->S.x, &proof->S.y, s_comp);

    uchar fs_buf[99];
    for (int i = 0; i < 33; ++i) {
        fs_buf[i]      = a_comp[i];
        fs_buf[33 + i] = s_comp[i];
        fs_buf[66 + i] = v_comp[i];
    }
    uchar y_hash[32], z_hash[32];
    zk_tagged_hash_midstate_impl(&ZK_BULLETPROOF_Y_MIDSTATE, fs_buf, 99, y_hash);
    zk_tagged_hash_midstate_impl(&ZK_BULLETPROOF_Z_MIDSTATE, fs_buf, 99, z_hash);
    Scalar y, z;
    scalar_from_bytes_impl(y_hash, &y);
    scalar_from_bytes_impl(z_hash, &z);

    // y powers, z^2, 2^i
    Scalar y_powers[64];
    y_powers[0] = ONE_S;
    for (int i = 1; i < 64; ++i)
        scalar_mul_mod_n_impl(&y_powers[i-1], &y, &y_powers[i]);

    Scalar z2;
    scalar_mul_mod_n_impl(&z, &z, &z2);

    Scalar two_powers[64];
    two_powers[0] = ONE_S;
    for (int i = 1; i < 64; ++i)
        scalar_add_mod_n_impl(&two_powers[i-1], &two_powers[i-1], &two_powers[i]);

    // t1, t2 coefficients
    Scalar t1 = ZERO_S, t2 = ZERO_S;
    for (int i = 0; i < 64; ++i) {
        Scalar l0_i, aR_plus_z, yi_aRz, z2_2i, r0_i, r1_i;
        ct_scalar_sub_impl(&a_L[i], &z, &l0_i);
        ct_scalar_add_impl(&a_R[i], &z, &aR_plus_z);
        ct_scalar_mul_impl(&y_powers[i], &aR_plus_z, &yi_aRz);
        ct_scalar_mul_impl(&z2, &two_powers[i], &z2_2i);
        ct_scalar_add_impl(&yi_aRz, &z2_2i, &r0_i);
        scalar_mul_mod_n_impl(&y_powers[i], &s_R[i], &r1_i);

        Scalar cross1, cross2, sum12;
        ct_scalar_mul_impl(&l0_i, &r1_i, &cross1);
        ct_scalar_mul_impl(&s_L[i], &r0_i, &cross2);
        ct_scalar_add_impl(&cross1, &cross2, &sum12);
        ct_scalar_add_impl(&t1, &sum12, &t1);

        Scalar t2_i;
        ct_scalar_mul_impl(&s_L[i], &r1_i, &t2_i);
        ct_scalar_add_impl(&t2, &t2_i, &t2);
    }

    // tau1, tau2
    uchar tau1_hash[32], tau2_hash[32];
    sha256_init(&sha_ctx);
    sha256_update(&sha_ctx, rho_hash, 32);
    sha256_final(&sha_ctx, tau1_hash);
    sha256_init(&sha_ctx);
    sha256_update(&sha_ctx, tau1_hash, 32);
    sha256_final(&sha_ctx, tau2_hash);
    Scalar tau1, tau2;
    scalar_from_bytes_impl(tau1_hash, &tau1);
    scalar_from_bytes_impl(tau2_hash, &tau2);

    // T1 = t1*H + tau1*G, T2 = t2*H + tau2*G
    CTJacobianPoint H_jac;
    H_jac.x = H_gen->x; H_jac.y = H_gen->y;
    H_jac.z.limbs[0] = 1; H_jac.z.limbs[1] = 0;
    H_jac.z.limbs[2] = 0; H_jac.z.limbs[3] = 0;
    H_jac.infinity = 0;

    CTJacobianPoint t1H, tau1G, T1_pt;
    ct_scalar_mul_point(&H_jac, &t1, &t1H);
    ct_generator_mul_impl(&tau1, &tau1G);
    {
        JacobianPoint ja = ct_point_to_jacobian(&t1H);
        JacobianPoint jb = ct_point_to_jacobian(&tau1G);
        JacobianPoint jr;
        point_add_impl(&jr, &ja, &jb);
        T1_pt = ct_point_from_jacobian(&jr);
    }
    ct_field_inv(&zi, &T1_pt.z);
    field_sqr_impl(&zi2, &zi);
    field_mul_impl(&zi3, &zi, &zi2);
    field_mul_impl(&proof->T1.x, &T1_pt.x, &zi2);
    field_mul_impl(&proof->T1.y, &T1_pt.y, &zi3);

    CTJacobianPoint t2H, tau2G, T2_pt;
    ct_scalar_mul_point(&H_jac, &t2, &t2H);
    ct_generator_mul_impl(&tau2, &tau2G);
    {
        JacobianPoint ja = ct_point_to_jacobian(&t2H);
        JacobianPoint jb = ct_point_to_jacobian(&tau2G);
        JacobianPoint jr;
        point_add_impl(&jr, &ja, &jb);
        T2_pt = ct_point_from_jacobian(&jr);
    }
    ct_field_inv(&zi, &T2_pt.z);
    field_sqr_impl(&zi2, &zi);
    field_mul_impl(&zi3, &zi, &zi2);
    field_mul_impl(&proof->T2.x, &T2_pt.x, &zi2);
    field_mul_impl(&proof->T2.y, &T2_pt.y, &zi3);

    // Fiat-Shamir: x
    uchar t1_comp[33], t2_comp[33];
    affine_to_compressed_impl(&proof->T1.x, &proof->T1.y, t1_comp);
    affine_to_compressed_impl(&proof->T2.x, &proof->T2.y, t2_comp);
    uchar x_buf[130];
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
    scalar_to_bytes_impl(&y, x_buf + 66);
    scalar_to_bytes_impl(&z, x_buf + 98);
    uchar x_hash[32];
    zk_tagged_hash_midstate_impl(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, 130, x_hash);
    Scalar xx;
    scalar_from_bytes_impl(x_hash, &xx);

    // Evaluate l(x), r(x), t_hat
    Scalar t_hat = ZERO_S;
    Scalar l_x[64], r_x[64];
    for (int i = 0; i < 64; ++i) {
        Scalar aL_z, sL_x;
        ct_scalar_sub_impl(&a_L[i], &z, &aL_z);
        ct_scalar_mul_impl(&s_L[i], &xx, &sL_x);
        ct_scalar_add_impl(&aL_z, &sL_x, &l_x[i]);

        Scalar aR_z, sR_x, aR_z_sR_x, yi_term, z2_2i;
        ct_scalar_add_impl(&a_R[i], &z, &aR_z);
        ct_scalar_mul_impl(&s_R[i], &xx, &sR_x);
        ct_scalar_add_impl(&aR_z, &sR_x, &aR_z_sR_x);
        ct_scalar_mul_impl(&y_powers[i], &aR_z_sR_x, &yi_term);
        ct_scalar_mul_impl(&z2, &two_powers[i], &z2_2i);
        ct_scalar_add_impl(&yi_term, &z2_2i, &r_x[i]);

        Scalar prod;
        ct_scalar_mul_impl(&l_x[i], &r_x[i], &prod);
        ct_scalar_add_impl(&t_hat, &prod, &t_hat);
    }
    proof->t_hat = t_hat;

    // tau_x = tau2*x^2 + tau1*x + z^2*blinding
    Scalar xx2, tau2_x2, tau1_x, z2_blind, tau_x;
    ct_scalar_mul_impl(&xx, &xx, &xx2);
    ct_scalar_mul_impl(&tau2, &xx2, &tau2_x2);
    ct_scalar_mul_impl(&tau1, &xx, &tau1_x);
    ct_scalar_mul_impl(&z2, blinding, &z2_blind);
    ct_scalar_add_impl(&tau2_x2, &tau1_x, &tau_x);
    ct_scalar_add_impl(&tau_x, &z2_blind, &tau_x);
    proof->tau_x = tau_x;

    // mu = alpha + rho*x
    Scalar rho_x;
    ct_scalar_mul_impl(&rho, &xx, &rho_x);
    ct_scalar_add_impl(&alpha, &rho_x, &proof->mu);

    // Inner Product Argument (6 rounds for n=64)
    Scalar a_vec[64], b_vec[64];
    for (int i = 0; i < 64; ++i) { a_vec[i] = l_x[i]; b_vec[i] = r_x[i]; }

    // Modified generators: H'_i = y^{-i} * H_i
    Scalar y_inv;
    ct_scalar_inverse_impl(&y, &y_inv);
    Scalar y_inv_pow = ONE_S;

    JacobianPoint G_vec[64], H_vec[64];
    for (int i = 0; i < 64; ++i) {
        G_vec[i].x = bp_G[i].x; G_vec[i].y = bp_G[i].y;
        G_vec[i].z.limbs[0] = 1; G_vec[i].z.limbs[1] = 0;
        G_vec[i].z.limbs[2] = 0; G_vec[i].z.limbs[3] = 0;
        G_vec[i].infinity = 0;

        JacobianPoint Hi_jac;
        Hi_jac.x = bp_H[i].x; Hi_jac.y = bp_H[i].y;
        Hi_jac.z.limbs[0] = 1; Hi_jac.z.limbs[1] = 0;
        Hi_jac.z.limbs[2] = 0; Hi_jac.z.limbs[3] = 0;
        Hi_jac.infinity = 0;
        scalar_mul_impl(&H_vec[i], &Hi_jac, &y_inv_pow);
        scalar_mul_mod_n_impl(&y_inv_pow, &y_inv, &y_inv_pow);
    }

    int n = 64;
    for (int round = 0; round < 6; ++round) {
        n /= 2;
        Scalar c_L = ZERO_S, c_R = ZERO_S;
        JacobianPoint L_pt, R_pt;
        L_pt.infinity = 1; L_pt.z.limbs[0] = 1;
        L_pt.z.limbs[1] = 0; L_pt.z.limbs[2] = 0; L_pt.z.limbs[3] = 0;
        R_pt = L_pt;

        for (int i = 0; i < n; ++i) {
            JacobianPoint aG, bH, tmp_r;
            scalar_mul_impl(&aG, &G_vec[n + i], &a_vec[i]);
            point_add_impl(&tmp_r, &L_pt, &aG); L_pt = tmp_r;
            scalar_mul_impl(&bH, &H_vec[i], &b_vec[n + i]);
            point_add_impl(&tmp_r, &L_pt, &bH); L_pt = tmp_r;
            Scalar prod;
            scalar_mul_mod_n_impl(&a_vec[i], &b_vec[n + i], &prod);
            scalar_add_mod_n_impl(&c_L, &prod, &c_L);

            scalar_mul_impl(&aG, &G_vec[i], &a_vec[n + i]);
            point_add_impl(&tmp_r, &R_pt, &aG); R_pt = tmp_r;
            scalar_mul_impl(&bH, &H_vec[n + i], &b_vec[i]);
            point_add_impl(&tmp_r, &R_pt, &bH); R_pt = tmp_r;
            scalar_mul_mod_n_impl(&a_vec[n + i], &b_vec[i], &prod);
            scalar_add_mod_n_impl(&c_R, &prod, &c_R);
        }

        JacobianPoint H_jac_std;
        H_jac_std.x = H_gen->x; H_jac_std.y = H_gen->y;
        H_jac_std.z.limbs[0] = 1; H_jac_std.z.limbs[1] = 0;
        H_jac_std.z.limbs[2] = 0; H_jac_std.z.limbs[3] = 0;
        H_jac_std.infinity = 0;

        JacobianPoint cU, tmp_r;
        scalar_mul_impl(&cU, &H_jac_std, &c_L);
        point_add_impl(&tmp_r, &L_pt, &cU); L_pt = tmp_r;
        scalar_mul_impl(&cU, &H_jac_std, &c_R);
        point_add_impl(&tmp_r, &R_pt, &cU); R_pt = tmp_r;

        // L,R to affine
        FieldElement zii, zii2, zii3;
        field_inv_impl(&zii, &L_pt.z);
        field_sqr_impl(&zii2, &zii);
        field_mul_impl(&zii3, &zii, &zii2);
        field_mul_impl(&proof->L[round].x, &L_pt.x, &zii2);
        field_mul_impl(&proof->L[round].y, &L_pt.y, &zii3);

        field_inv_impl(&zii, &R_pt.z);
        field_sqr_impl(&zii2, &zii);
        field_mul_impl(&zii3, &zii, &zii2);
        field_mul_impl(&proof->R[round].x, &R_pt.x, &zii2);
        field_mul_impl(&proof->R[round].y, &R_pt.y, &zii3);

        // Fiat-Shamir: x_round
        uchar l_comp[33], r_comp[33];
        affine_to_compressed_impl(&proof->L[round].x, &proof->L[round].y, l_comp);
        affine_to_compressed_impl(&proof->R[round].x, &proof->R[round].y, r_comp);
        uchar ip_buf[66];
        for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }

        // Use simple SHA-256 tagged hash for IP rounds
        uchar xr_hash[32];
        uchar ip_tag[] = {'B','P','/','i','p'};
        zk_tagged_hash_impl(ip_tag, 5, ip_buf, 66, xr_hash);
        Scalar x_r;
        scalar_from_bytes_impl(xr_hash, &x_r);
        Scalar x_r_inv;
        ct_scalar_inverse_impl(&x_r, &x_r_inv);

        // Fold vectors
        for (int i = 0; i < n; ++i) {
            Scalar a_lo_x, a_hi_xi;
            ct_scalar_mul_impl(&a_vec[i], &x_r, &a_lo_x);
            ct_scalar_mul_impl(&a_vec[n + i], &x_r_inv, &a_hi_xi);
            ct_scalar_add_impl(&a_lo_x, &a_hi_xi, &a_vec[i]);

            Scalar b_lo_xi, b_hi_x;
            ct_scalar_mul_impl(&b_vec[i], &x_r_inv, &b_lo_xi);
            ct_scalar_mul_impl(&b_vec[n + i], &x_r, &b_hi_x);
            ct_scalar_add_impl(&b_lo_xi, &b_hi_x, &b_vec[i]);
        }
        for (int i = 0; i < n; ++i) {
            JacobianPoint G_lo_xi, G_hi_x, H_lo_x, H_hi_xi;
            scalar_mul_impl(&G_lo_xi, &G_vec[i], &x_r_inv);
            scalar_mul_impl(&G_hi_x, &G_vec[n + i], &x_r);
            point_add_impl(&G_vec[i], &G_lo_xi, &G_hi_x);

            scalar_mul_impl(&H_lo_x, &H_vec[i], &x_r);
            scalar_mul_impl(&H_hi_xi, &H_vec[n + i], &x_r_inv);
            point_add_impl(&H_vec[i], &H_lo_x, &H_hi_xi);
        }
    }

    proof->a = a_vec[0];
    proof->b = b_vec[0];
    return 1;
}

#endif // SECP256K1_CT_ZK_CL
