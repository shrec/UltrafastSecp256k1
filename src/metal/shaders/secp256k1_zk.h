#pragma once
// =============================================================================
// UltrafastSecp256k1 Metal -- Zero-Knowledge Proof Operations
// =============================================================================
// ZK proof primitives for Metal (Apple Silicon GPU):
//   1. Knowledge proof (Schnorr sigma protocol) -- prove + verify
//   2. DLEQ proof (discrete log equality) -- prove + verify
//   3. Bulletproof range proof (64-bit) -- verify
//
// Depends on: secp256k1_extended.h (includes field, point, scalar, SHA-256)
// Uses 8x32-bit limbs (uint) -- matching existing Metal convention.
// =============================================================================

#include "secp256k1_extended.h"

// =============================================================================
// ZK Proof Structures
// =============================================================================

struct ZKKnowledgeProof {
    uchar rx[32];    // R.x (x-coordinate of nonce point, even Y)
    Scalar256 s;     // response scalar
};

struct ZKDLEQProof {
    Scalar256 e;     // challenge
    Scalar256 s;     // response
};

// =============================================================================
// Helper: Point to Compressed (33 bytes: prefix || x)
// =============================================================================

inline void point_to_compressed(thread const JacobianPoint &p,
                                 thread uchar out[33]) {
    if (p.infinity != 0) {
        for (int i = 0; i < 33; ++i) out[i] = 0;
        return;
    }
    AffinePoint aff = jacobian_to_affine(p);

    uchar y_bytes[32];
    field_to_bytes(aff.y, y_bytes);
    out[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
    field_to_bytes(aff.x, out + 1);
}

// =============================================================================
// ZK Nonce Derivation
// =============================================================================
// k = H("ZK/nonce" || (secret XOR H(aux)) || point_compressed || msg || aux)

inline Scalar256 zk_derive_nonce(thread const Scalar256 &secret,
                                  thread const JacobianPoint &point,
                                  thread const uchar msg[32],
                                  thread const uchar aux[32]) {
    // Hash aux for XOR hedging
    uchar aux_hash[32];
    SHA256Ctx ctx; sha256_init(ctx);
    sha256_update(ctx, aux, 32);
    sha256_final(ctx, aux_hash);

    // masked = secret_bytes XOR aux_hash
    uchar sec_bytes[32];
    scalar_to_bytes(secret, sec_bytes);
    uchar masked[32];
    for (int i = 0; i < 32; ++i) masked[i] = sec_bytes[i] ^ aux_hash[i];

    // Compress point
    uchar pt_comp[33];
    point_to_compressed(point, pt_comp);

    // buf = masked[32] || pt_comp[33] || msg[32] || aux[32] = 129 bytes
    uchar buf[129];
    for (int i = 0; i < 32; ++i) buf[i] = masked[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = pt_comp[i];
    for (int i = 0; i < 32; ++i) buf[65 + i] = msg[i];
    for (int i = 0; i < 32; ++i) buf[97 + i] = aux[i];

    uchar hash[32];
    uchar tag[] = {'Z','K','/','n','o','n','c','e'};
    tagged_hash(tag, 8, buf, 129, hash);
    return scalar_from_bytes(hash);
}

// =============================================================================
// 1a. Knowledge Proof -- Proving
// =============================================================================
// Proves knowledge of secret s such that P = s * B for arbitrary base B.

inline bool zk_knowledge_prove(thread const Scalar256 &secret,
                                thread const JacobianPoint &pubkey,
                                thread const AffinePoint &base,
                                thread const uchar msg[32],
                                thread const uchar aux[32],
                                thread ZKKnowledgeProof &proof) {
    // k = deterministic nonce
    Scalar256 k = zk_derive_nonce(secret, pubkey, msg, aux);
    if (scalar256_is_zero(k)) return false;

    // R = k * base
    JacobianPoint R = scalar_mul(base, k);

    // Convert R to affine
    AffinePoint R_aff = jacobian_to_affine(R);

    // Ensure even Y: if Y is odd, negate k
    uchar ry_bytes[32];
    field_to_bytes(R_aff.y, ry_bytes);
    Scalar256 k_eff;
    if (ry_bytes[31] & 1) {
        k_eff = scalar_negate(k);
        R_aff.y = field_negate(R_aff.y);
    } else {
        k_eff = k;
    }

    // Store R.x
    field_to_bytes(R_aff.x, proof.rx);

    // e = H("ZK/knowledge" || R.x || P_comp || B_comp || msg)
    uchar p_comp[33], b_comp[33];
    point_to_compressed(pubkey, p_comp);
    JacobianPoint base_jac;
    base_jac.x = base.x; base_jac.y = base.y; base_jac.z = field_one(); base_jac.infinity = 0;
    point_to_compressed(base_jac, b_comp);

    uchar buf[130]; // 32 + 33 + 33 + 32
    for (int i = 0; i < 32; ++i) buf[i] = proof.rx[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[65 + i] = b_comp[i];
    for (int i = 0; i < 32; ++i) buf[98 + i] = msg[i];

    uchar e_hash[32];
    uchar tag[] = {'Z','K','/','k','n','o','w','l','e','d','g','e'};
    tagged_hash(tag, 12, buf, 130, e_hash);

    Scalar256 e = scalar_from_bytes(e_hash);

    // s = k_eff + e * secret
    Scalar256 e_sec = scalar_mul_mod_n(e, secret);
    proof.s = scalar_add_mod_n(k_eff, e_sec);

    return true;
}

// Convenience: prove with generator G
inline bool zk_knowledge_prove_generator(thread const Scalar256 &secret,
                                          thread const JacobianPoint &pubkey,
                                          thread const uchar msg[32],
                                          thread const uchar aux[32],
                                          thread ZKKnowledgeProof &proof) {
    AffinePoint G = generator_affine();
    return zk_knowledge_prove(secret, pubkey, G, msg, aux, proof);
}

// =============================================================================
// 1b. Knowledge Proof -- Verification
// =============================================================================

inline bool zk_knowledge_verify(thread const ZKKnowledgeProof &proof,
                                 thread const JacobianPoint &pubkey,
                                 thread const AffinePoint &base,
                                 thread const uchar msg[32]) {
    // Reconstruct challenge e
    uchar p_comp[33], b_comp[33];
    point_to_compressed(pubkey, p_comp);
    JacobianPoint base_jac;
    base_jac.x = base.x; base_jac.y = base.y; base_jac.z = field_one(); base_jac.infinity = 0;
    point_to_compressed(base_jac, b_comp);

    uchar buf[130];
    for (int i = 0; i < 32; ++i) buf[i] = proof.rx[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[65 + i] = b_comp[i];
    for (int i = 0; i < 32; ++i) buf[98 + i] = msg[i];

    uchar e_hash[32];
    uchar tag[] = {'Z','K','/','k','n','o','w','l','e','d','g','e'};
    tagged_hash(tag, 12, buf, 130, e_hash);

    Scalar256 e = scalar_from_bytes(e_hash);

    // Verify: s*B == R + e*P
    JacobianPoint sB = scalar_mul(base, proof.s);

    AffinePoint pk_aff = jacobian_to_affine(pubkey);
    JacobianPoint eP = scalar_mul(pk_aff, e);

    // R = lift_x(proof.rx) with even Y
    JacobianPoint R_pt;
    if (!lift_x(proof.rx, R_pt)) return false;

    // R + e*P
    AffinePoint eP_aff = jacobian_to_affine(eP);
    AffinePoint R_aff = jacobian_to_affine(R_pt);
    JacobianPoint R_jac;
    R_jac.x = R_aff.x; R_jac.y = R_aff.y; R_jac.z = field_one(); R_jac.infinity = 0;
    JacobianPoint R_plus_eP = jacobian_add_mixed(R_jac, eP_aff);

    // Compare s*B == R + e*P via compressed form
    uchar comp1[33], comp2[33];
    point_to_compressed(sB, comp1);
    point_to_compressed(R_plus_eP, comp2);

    for (int i = 0; i < 33; ++i)
        if (comp1[i] != comp2[i]) return false;
    return true;
}

// =============================================================================
// 2a. DLEQ Proof -- Proving
// =============================================================================
// Proves log_G(P) == log_H(Q) without revealing the discrete log.

inline bool zk_dleq_prove(thread const Scalar256 &secret,
                            thread const AffinePoint &G,
                            thread const AffinePoint &H,
                            thread const JacobianPoint &P,
                            thread const JacobianPoint &Q,
                            thread const uchar aux[32],
                            thread ZKDLEQProof &proof) {
    // Derive nonce using Q_compressed as msg
    uchar q_comp[33];
    point_to_compressed(Q, q_comp);

    Scalar256 k = zk_derive_nonce(secret, P, q_comp, aux);
    if (scalar256_is_zero(k)) return false;

    // R1 = k * G, R2 = k * H
    JacobianPoint R1 = scalar_mul(G, k);
    JacobianPoint R2 = scalar_mul(H, k);

    // Serialize all 6 points
    uchar g_comp[33], h_comp[33], p_comp[33];
    uchar r1_comp[33], r2_comp[33];
    JacobianPoint G_jac, H_jac;
    G_jac.x = G.x; G_jac.y = G.y; G_jac.z = field_one(); G_jac.infinity = 0;
    H_jac.x = H.x; H_jac.y = H.y; H_jac.z = field_one(); H_jac.infinity = 0;
    point_to_compressed(G_jac, g_comp);
    point_to_compressed(H_jac, h_comp);
    point_to_compressed(P, p_comp);
    point_to_compressed(R1, r1_comp);
    point_to_compressed(R2, r2_comp);

    // e = H("ZK/dleq" || G || H || P || Q || R1 || R2)
    uchar buf[198]; // 33 * 6
    for (int i = 0; i < 33; ++i) {
        buf[i]       = g_comp[i];
        buf[33 + i]  = h_comp[i];
        buf[66 + i]  = p_comp[i];
        buf[99 + i]  = q_comp[i];
        buf[132 + i] = r1_comp[i];
        buf[165 + i] = r2_comp[i];
    }

    uchar e_hash[32];
    uchar tag[] = {'Z','K','/','d','l','e','q'};
    tagged_hash(tag, 7, buf, 198, e_hash);
    proof.e = scalar_from_bytes(e_hash);

    // s = k + e * secret
    Scalar256 e_sec = scalar_mul_mod_n(proof.e, secret);
    proof.s = scalar_add_mod_n(k, e_sec);

    return true;
}

// =============================================================================
// 2b. DLEQ Proof -- Verification
// =============================================================================

inline bool zk_dleq_verify(thread const ZKDLEQProof &proof,
                             thread const AffinePoint &G,
                             thread const AffinePoint &H,
                             thread const JacobianPoint &P,
                             thread const JacobianPoint &Q) {
    // R1 = s*G - e*P, R2 = s*H - e*Q
    JacobianPoint sG = scalar_mul(G, proof.s);
    AffinePoint P_aff = jacobian_to_affine(P);
    JacobianPoint eP = scalar_mul(P_aff, proof.e);

    JacobianPoint sH = scalar_mul(H, proof.s);
    AffinePoint Q_aff = jacobian_to_affine(Q);
    JacobianPoint eQ = scalar_mul(Q_aff, proof.e);

    // Negate eP and eQ
    eP.y = field_negate(eP.y);
    eQ.y = field_negate(eQ.y);

    AffinePoint eP_aff = jacobian_to_affine(eP);
    AffinePoint eQ_aff = jacobian_to_affine(eQ);
    JacobianPoint R1 = jacobian_add_mixed(sG, eP_aff);
    JacobianPoint R2 = jacobian_add_mixed(sH, eQ_aff);

    // Serialize all 6 points for challenge recomputation
    uchar g_comp[33], h_comp[33], p_comp[33], q_comp[33];
    uchar r1_comp[33], r2_comp[33];
    JacobianPoint G_jac, H_jac;
    G_jac.x = G.x; G_jac.y = G.y; G_jac.z = field_one(); G_jac.infinity = 0;
    H_jac.x = H.x; H_jac.y = H.y; H_jac.z = field_one(); H_jac.infinity = 0;
    point_to_compressed(G_jac, g_comp);
    point_to_compressed(H_jac, h_comp);
    point_to_compressed(P, p_comp);
    point_to_compressed(Q, q_comp);
    point_to_compressed(R1, r1_comp);
    point_to_compressed(R2, r2_comp);

    uchar buf[198];
    for (int i = 0; i < 33; ++i) {
        buf[i]       = g_comp[i];
        buf[33 + i]  = h_comp[i];
        buf[66 + i]  = p_comp[i];
        buf[99 + i]  = q_comp[i];
        buf[132 + i] = r1_comp[i];
        buf[165 + i] = r2_comp[i];
    }

    uchar e_hash[32];
    uchar tag[] = {'Z','K','/','d','l','e','q'};
    tagged_hash(tag, 7, buf, 198, e_hash);

    Scalar256 e_check = scalar_from_bytes(e_hash);
    return scalar256_eq(proof.e, e_check);
}

// =============================================================================
// 3. Bulletproof Range Proof (64-bit) -- Verification
// =============================================================================
// Full Bulletproof range proof verification on Metal.
// Ported from CUDA implementation (commit 02ac59d).
//
// Architecture:
//   - Single thread per proof (no SIMD group cooperation)
//   - Generator tables stored in device memory
//   - Tagged hash midstates for Fiat-Shamir (precomputed once)
// =============================================================================

#define BP_BITS  64
#define BP_LOG2  6

// -- Tagged Hash Midstate --
// SHA256 state after processing SHA256(tag)||SHA256(tag) (1 block = 64 bytes).
struct ZKTagMidstate {
    uint h[8];
};

// Precomputed midstates (same as CUDA)
constant const ZKTagMidstate ZK_BULLETPROOF_Y_MIDSTATE = {{
    0x770918af, 0xa4791204, 0x3c076a40, 0x5fb23056,
    0x902acdb9, 0x1d85371b, 0x10f624c4, 0x9048ba46
}};

constant const ZKTagMidstate ZK_BULLETPROOF_Z_MIDSTATE = {{
    0x22be001a, 0x3c79431b, 0xe60a9432, 0xfd965d54,
    0x84df949f, 0x62937cee, 0x20924a62, 0x99f23a35
}};

constant const ZKTagMidstate ZK_BULLETPROOF_X_MIDSTATE = {{
    0x1378a3c8, 0x2e8ad1b2, 0xa47ce2e2, 0x143037a2,
    0xbaec0bd8, 0x40cb0ed7, 0xd1b23b65, 0x43871df4
}};

// Tagged hash using precomputed midstate
inline void zk_tagged_hash_midstate(thread const ZKTagMidstate &midstate,
                                     thread const uchar* data, uint data_len,
                                     thread uchar out[32]) {
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = midstate.h[i];
    ctx.buf_len = 0;
    ctx.total_len_lo = 64;
    ctx.total_len_hi = 0;
    sha256_update(ctx, data, data_len);
    sha256_final(ctx, out);
}

// Overload for constant address space midstates (e.g. global constants)
inline void zk_tagged_hash_midstate(constant const ZKTagMidstate &midstate,
                                     thread const uchar* data, uint data_len,
                                     thread uchar out[32]) {
    ZKTagMidstate local_ms;
    for (int i = 0; i < 8; i++) local_ms.h[i] = midstate.h[i];
    zk_tagged_hash_midstate(local_ms, data, data_len, out);
}

// field_from_bytes: big-endian bytes -> 8x32-bit limbs
inline FieldElement field_from_bytes(thread const uchar bytes[32]) {
    FieldElement r;
    for (int i = 0; i < 8; i++) {
        uint limb = 0;
        int base = (7 - i) * 4;
        for (int j = 0; j < 4; j++)
            limb = (limb << 8) | (uint)bytes[base + j];
        r.limbs[i] = limb;
    }
    return r;
}

// lift_x with even Y from FieldElement (not bytes)
inline bool lift_x_field_even(thread const FieldElement &x, thread AffinePoint &out) {
    FieldElement x2 = field_sqr(x);
    FieldElement x3 = field_mul(x2, x);
    FieldElement seven;
    for (int i = 0; i < 8; i++) seven.limbs[i] = 0;
    seven.limbs[0] = 7;
    FieldElement y2 = field_add(x3, seven);
    FieldElement y = field_sqrt(y2);

    // Verify: y^2 == x^3 + 7
    FieldElement y_check = field_sqr(y);
    uchar yc_b[32], y2_b[32];
    field_to_bytes(y_check, yc_b);
    field_to_bytes(y2, y2_b);
    for (int i = 0; i < 32; i++)
        if (yc_b[i] != y2_b[i]) return false;

    // Ensure even Y
    uchar y_bytes[32];
    field_to_bytes(y, y_bytes);
    if (y_bytes[31] & 1) y = field_negate(y);

    out.x = x;
    out.y = y;
    return true;
}

// Try-and-increment: find point on curve starting from x
inline AffinePoint hash_to_point_increment(thread FieldElement &x) {
    AffinePoint out;
    FieldElement one_fe;
    for (int i = 0; i < 8; i++) one_fe.limbs[i] = 0;
    one_fe.limbs[0] = 1;
    for (int attempt = 0; attempt < 256; ++attempt) {
        if (lift_x_field_even(x, out)) return out;
        x = field_add(x, one_fe);
    }
    return out;
}

// Affine point to 33-byte compressed (prefix || x_bytes)
inline void affine_to_compressed(thread const AffinePoint &p, thread uchar out[33]) {
    uchar y_bytes[32];
    field_to_bytes(p.y, y_bytes);
    out[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
    field_to_bytes(p.x, out + 1);
}

// -- Bulletproof Range Proof Structure --
struct RangeProofGPU {
    AffinePoint A, S;          // vector commitments
    AffinePoint T1, T2;        // polynomial commitments
    Scalar256 tau_x, mu, t_hat; // blinding, aggregate blinding, poly eval
    Scalar256 a, b;             // final IPA scalars
    AffinePoint L[6], R[6];    // inner product argument rounds (log2(64)=6)
};

// =============================================================================
// Bulletproof Full Verify (single thread per proof)
// =============================================================================

inline bool range_verify_full(thread const RangeProofGPU &proof,
                               thread const AffinePoint &commitment,
                               thread const AffinePoint &H_ped,
                               device const AffinePoint* bp_G,
                               device const AffinePoint* bp_H,
                               thread const ZKTagMidstate &bp_ip_midstate)
{
    // ---- Fiat-Shamir: recompute y, z, x ----
    uchar a_comp[33], s_comp[33], v_comp[33];
    affine_to_compressed(proof.A, a_comp);
    affine_to_compressed(proof.S, s_comp);
    affine_to_compressed(commitment, v_comp);

    uchar fs_buf[99];
    for (int i = 0; i < 33; ++i) {
        fs_buf[i]      = a_comp[i];
        fs_buf[33 + i] = s_comp[i];
        fs_buf[66 + i] = v_comp[i];
    }

    uchar y_hash[32], z_hash[32];
    zk_tagged_hash_midstate(ZK_BULLETPROOF_Y_MIDSTATE, fs_buf, 99, y_hash);
    zk_tagged_hash_midstate(ZK_BULLETPROOF_Z_MIDSTATE, fs_buf, 99, z_hash);

    Scalar256 y = scalar_from_bytes(y_hash);
    Scalar256 z = scalar_from_bytes(z_hash);

    uchar t1_comp[33], t2_comp[33];
    affine_to_compressed(proof.T1, t1_comp);
    affine_to_compressed(proof.T2, t2_comp);

    uchar x_buf[130];
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
    scalar_to_bytes(y, x_buf + 66);
    scalar_to_bytes(z, x_buf + 98);

    uchar x_hash[32];
    zk_tagged_hash_midstate(ZK_BULLETPROOF_X_MIDSTATE, x_buf, 130, x_hash);
    Scalar256 x = scalar_from_bytes(x_hash);

    // ---- Compute delta(y,z) ----
    Scalar256 z2 = scalar_mul_mod_n(z, z);
    Scalar256 z3 = scalar_mul_mod_n(z2, z);
    Scalar256 x2 = scalar_mul_mod_n(x, x);

    // sum(y^i) for i in [0, 64)
    Scalar256 sum_y;
    for (int i = 0; i < 8; i++) sum_y.limbs[i] = 0;
    sum_y.limbs[0] = 1;
    Scalar256 y_pow = y;
    for (int i = 1; i < BP_BITS; ++i) {
        sum_y = scalar_add_mod_n(sum_y, y_pow);
        y_pow = scalar_mul_mod_n(y_pow, y);
    }

    // sum(2^i) for i in [0, 64) = 2^64 - 1
    Scalar256 sum_2;
    for (int i = 0; i < 8; i++) sum_2.limbs[i] = 0;
    sum_2.limbs[0] = 0xFFFFFFFF;
    sum_2.limbs[1] = 0xFFFFFFFF;

    Scalar256 z_minus_z2 = scalar_sub_mod_n(z, z2);
    Scalar256 term1 = scalar_mul_mod_n(z_minus_z2, sum_y);
    Scalar256 term2 = scalar_mul_mod_n(z3, sum_2);
    Scalar256 delta = scalar_sub_mod_n(term1, term2);

    // ---- Polynomial check ----
    // LHS = t_hat * H + tau_x * G
    JacobianPoint tH = scalar_mul(H_ped, proof.t_hat);
    JacobianPoint tauG = scalar_mul_generator_windowed(proof.tau_x);
    JacobianPoint LHS = jacobian_add(tH, tauG);

    // RHS = z^2 * V + delta * H + x * T1 + x^2 * T2
    JacobianPoint z2V = scalar_mul(commitment, z2);
    JacobianPoint deltaH = scalar_mul(H_ped, delta);
    JacobianPoint xT1 = scalar_mul(proof.T1, x);
    JacobianPoint x2T2 = scalar_mul(proof.T2, x2);

    JacobianPoint tmp_rhs = jacobian_add(z2V, deltaH);
    JacobianPoint RHS = jacobian_add(tmp_rhs, xT1);
    tmp_rhs = jacobian_add(RHS, x2T2);

    // Compare LHS == RHS via Jacobian cross-multiply (0 field_inv)
    {
        FieldElement z1sq = field_sqr(LHS.z);
        FieldElement z2sq = field_sqr(tmp_rhs.z);
        FieldElement z1cu = field_mul(z1sq, LHS.z);
        FieldElement z2cu = field_mul(z2sq, tmp_rhs.z);

        FieldElement lx = field_mul(LHS.x, z2sq);
        FieldElement rx_cmp = field_mul(tmp_rhs.x, z1sq);
        FieldElement ly = field_mul(LHS.y, z2cu);
        FieldElement ry = field_mul(tmp_rhs.y, z1cu);

        FieldElement dx = field_sub(lx, rx_cmp);
        FieldElement dy = field_sub(ly, ry);

        uchar dx_b[32], dy_b[32];
        field_to_bytes(dx, dx_b);
        field_to_bytes(dy, dy_b);
        for (int i = 0; i < 32; i++)
            if (dx_b[i] != 0 || dy_b[i] != 0) return false;
    }

    // ---- Inner Product Argument verification ----
    Scalar256 x_rounds[BP_LOG2];
    for (int round = 0; round < BP_LOG2; ++round) {
        uchar l_comp[33], r_comp[33];
        affine_to_compressed(proof.L[round], l_comp);
        affine_to_compressed(proof.R[round], r_comp);
        uchar ip_buf[66];
        for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
        uchar xr_hash[32];
        zk_tagged_hash_midstate(bp_ip_midstate, ip_buf, 66, xr_hash);
        x_rounds[round] = scalar_from_bytes(xr_hash);
    }

    // Batch inversion of x_rounds
    Scalar256 x_inv_rounds[BP_LOG2];
    {
        Scalar256 acc[BP_LOG2];
        acc[0] = x_rounds[0];
        for (int j = 1; j < BP_LOG2; ++j) acc[j] = scalar_mul_mod_n(acc[j-1], x_rounds[j]);
        Scalar256 inv_acc = scalar_inverse(acc[BP_LOG2 - 1]);
        for (int j = BP_LOG2 - 1; j >= 1; --j) {
            x_inv_rounds[j] = scalar_mul_mod_n(inv_acc, acc[j-1]);
            inv_acc = scalar_mul_mod_n(inv_acc, x_rounds[j]);
        }
        x_inv_rounds[0] = inv_acc;
    }

    // y_inv and y_inv_powers
    Scalar256 y_inv = scalar_inverse(y);
    Scalar256 y_inv_powers[BP_BITS];
    for (int i = 0; i < 8; i++) y_inv_powers[0].limbs[i] = 0;
    y_inv_powers[0].limbs[0] = 1;
    for (int i = 1; i < BP_BITS; ++i)
        y_inv_powers[i] = scalar_mul_mod_n(y_inv_powers[i-1], y_inv);

    // s_coeff: product tree of x_rounds / x_inv_rounds
    Scalar256 s_coeff[BP_BITS];
    for (int i = 0; i < 8; i++) s_coeff[0].limbs[i] = 0;
    s_coeff[0].limbs[0] = 1;
    for (int j = 0; j < BP_LOG2; ++j)
        s_coeff[0] = scalar_mul_mod_n(s_coeff[0], x_inv_rounds[j]);
    for (int i = 1; i < BP_BITS; ++i) {
        for (int ii = 0; ii < 8; ii++) s_coeff[i].limbs[ii] = 0;
        s_coeff[i].limbs[0] = 1;
        for (int jj = 0; jj < BP_LOG2; ++jj) {
            if ((i >> (BP_LOG2 - 1 - jj)) & 1)
                s_coeff[i] = scalar_mul_mod_n(s_coeff[i], x_rounds[jj]);
            else
                s_coeff[i] = scalar_mul_mod_n(s_coeff[i], x_inv_rounds[jj]);
        }
    }

    // Batch inversion for s_inv
    Scalar256 s_inv[BP_BITS];
    {
        Scalar256 acc[BP_BITS];
        acc[0] = s_coeff[0];
        for (int i = 1; i < BP_BITS; ++i) acc[i] = scalar_mul_mod_n(acc[i-1], s_coeff[i]);
        Scalar256 inv_acc = scalar_inverse(acc[BP_BITS - 1]);
        for (int i = BP_BITS - 1; i >= 1; --i) {
            s_inv[i] = scalar_mul_mod_n(inv_acc, acc[i-1]);
            inv_acc = scalar_mul_mod_n(inv_acc, s_coeff[i]);
        }
        s_inv[0] = inv_acc;
    }

    // two_powers: 2^i
    Scalar256 two_powers[BP_BITS];
    for (int i = 0; i < 8; i++) two_powers[0].limbs[i] = 0;
    two_powers[0].limbs[0] = 1;
    for (int i = 1; i < BP_BITS; ++i)
        two_powers[i] = scalar_add_mod_n(two_powers[i-1], two_powers[i-1]);

    // ---- Build MSM ----
    Scalar256 neg_z = scalar_negate(z);
    Scalar256 ab = scalar_mul_mod_n(proof.a, proof.b);

    JacobianPoint msm_acc = point_at_infinity();

    // A (coefficient 1)
    msm_acc = jacobian_add_mixed(msm_acc, proof.A);

    // x * S
    {
        JacobianPoint xS = scalar_mul(proof.S, x);
        msm_acc = jacobian_add(msm_acc, xS);
    }

    // G_i and H_i contributions
    for (int i = 0; i < BP_BITS; ++i) {
        // G_i: (-z - a*s_i)
        Scalar256 a_si = scalar_mul_mod_n(proof.a, s_coeff[i]);
        Scalar256 g_coeff = scalar_sub_mod_n(neg_z, a_si);

        JacobianPoint g_term = scalar_mul(bp_G[i], g_coeff);
        msm_acc = jacobian_add(msm_acc, g_term);

        // H_i: (z + z2*2^i*y_inv^i) - b*s_inv[i]*y_inv^i
        Scalar256 z2_2i = scalar_mul_mod_n(z2, two_powers[i]);
        Scalar256 z2_2i_yi = scalar_mul_mod_n(z2_2i, y_inv_powers[i]);
        Scalar256 h_pcheck = scalar_add_mod_n(z, z2_2i_yi);

        Scalar256 b_si = scalar_mul_mod_n(proof.b, s_inv[i]);
        Scalar256 b_si_yi = scalar_mul_mod_n(b_si, y_inv_powers[i]);
        Scalar256 h_coeff = scalar_sub_mod_n(h_pcheck, b_si_yi);

        JacobianPoint h_term = scalar_mul(bp_H[i], h_coeff);
        msm_acc = jacobian_add(msm_acc, h_term);
    }

    // -mu * G
    {
        Scalar256 neg_mu = scalar_negate(proof.mu);
        JacobianPoint muG = scalar_mul_generator_windowed(neg_mu);
        msm_acc = jacobian_add(msm_acc, muG);
    }

    // (t_hat - a*b) * U (H_ped)
    {
        Scalar256 t_ab = scalar_sub_mod_n(proof.t_hat, ab);
        JacobianPoint tU = scalar_mul(H_ped, t_ab);
        msm_acc = jacobian_add(msm_acc, tU);
    }

    // L_j and R_j contributions
    for (int j = 0; j < BP_LOG2; ++j) {
        Scalar256 xj2 = scalar_mul_mod_n(x_rounds[j], x_rounds[j]);
        Scalar256 xj_inv2 = scalar_mul_mod_n(x_inv_rounds[j], x_inv_rounds[j]);

        JacobianPoint lterm = scalar_mul(proof.L[j], xj2);
        JacobianPoint rterm = scalar_mul(proof.R[j], xj_inv2);
        msm_acc = jacobian_add(msm_acc, lterm);
        msm_acc = jacobian_add(msm_acc, rterm);
    }

    // Check: msm_acc should be identity
    if (msm_acc.infinity) return true;

    // Check Z == 0 via bytes (handles unreduced limbs)
    uchar z_bytes[32];
    field_to_bytes(msm_acc.z, z_bytes);
    for (int i = 0; i < 32; i++)
        if (z_bytes[i] != 0) return false;
    return true;
}

// =============================================================================
// 4. Bulletproof Polynomial Check (fast partial verification)
// =============================================================================
// Verifies the polynomial commitment part only (no IPA).
// Checks: t_hat * H + tau_x * G == z^2 * V + delta * H + x * T1 + x^2 * T2
// Ported from CUDA range_proof_poly_check_device (zk.cuh).

struct RangeProofPolyGPU {
    AffinePoint A;       // vector commitment A
    AffinePoint S;       // vector commitment S
    AffinePoint T1;      // polynomial commitment T1
    AffinePoint T2;      // polynomial commitment T2
    Scalar256 tau_x;     // blinding for polynomial eval
    Scalar256 t_hat;     // polynomial evaluation
};

inline bool range_proof_poly_check(thread const RangeProofPolyGPU &proof,
                                    thread const AffinePoint &commitment,
                                    thread const AffinePoint &H_gen)
{
    // Serialize A, S, V
    uchar a_comp[33], s_comp[33], v_comp[33];
    affine_to_compressed(proof.A, a_comp);
    affine_to_compressed(proof.S, s_comp);
    affine_to_compressed(commitment, v_comp);

    uchar fs_buf[99];
    for (int i = 0; i < 33; ++i) {
        fs_buf[i]      = a_comp[i];
        fs_buf[33 + i] = s_comp[i];
        fs_buf[66 + i] = v_comp[i];
    }

    uchar y_hash[32], z_hash[32];
    zk_tagged_hash_midstate(ZK_BULLETPROOF_Y_MIDSTATE, fs_buf, 99, y_hash);
    zk_tagged_hash_midstate(ZK_BULLETPROOF_Z_MIDSTATE, fs_buf, 99, z_hash);

    Scalar256 y = scalar_from_bytes(y_hash);
    Scalar256 z = scalar_from_bytes(z_hash);

    // Serialize T1, T2
    uchar t1_comp[33], t2_comp[33];
    affine_to_compressed(proof.T1, t1_comp);
    affine_to_compressed(proof.T2, t2_comp);

    uchar x_buf[130];
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
    scalar_to_bytes(y, x_buf + 66);
    scalar_to_bytes(z, x_buf + 98);

    uchar x_hash[32];
    zk_tagged_hash_midstate(ZK_BULLETPROOF_X_MIDSTATE, x_buf, 130, x_hash);
    Scalar256 x = scalar_from_bytes(x_hash);

    // Compute delta(y,z) = (z - z^2) * sum(y^i) - z^3 * sum(2^i)
    Scalar256 z2 = scalar_mul_mod_n(z, z);
    Scalar256 z3 = scalar_mul_mod_n(z2, z);

    // sum(y^i) for i in [0, 64)
    Scalar256 sum_y;
    for (int i = 0; i < 8; i++) sum_y.limbs[i] = 0;
    sum_y.limbs[0] = 1;
    Scalar256 y_pow = y;
    for (int i = 1; i < 64; ++i) {
        sum_y = scalar_add_mod_n(sum_y, y_pow);
        y_pow = scalar_mul_mod_n(y_pow, y);
    }

    // sum(2^i) for i in [0, 64) = 2^64 - 1
    Scalar256 sum_2;
    for (int i = 0; i < 8; i++) sum_2.limbs[i] = 0;
    sum_2.limbs[0] = 0xFFFFFFFF;
    sum_2.limbs[1] = 0xFFFFFFFF;

    Scalar256 z_minus_z2 = scalar_sub_mod_n(z, z2);
    Scalar256 term1 = scalar_mul_mod_n(z_minus_z2, sum_y);
    Scalar256 term2 = scalar_mul_mod_n(z3, sum_2);
    Scalar256 delta = scalar_sub_mod_n(term1, term2);

    // LHS = t_hat * H + tau_x * G
    JacobianPoint tH = scalar_mul(H_gen, proof.t_hat);
    JacobianPoint tauG = scalar_mul_generator_windowed(proof.tau_x);
    JacobianPoint LHS = jacobian_add(tH, tauG);

    // RHS = z^2 * V + delta * H + x * T1 + x^2 * T2
    Scalar256 x2 = scalar_mul_mod_n(x, x);

    JacobianPoint z2V = scalar_mul(commitment, z2);
    JacobianPoint deltaH = scalar_mul(H_gen, delta);
    JacobianPoint xT1 = scalar_mul(proof.T1, x);
    JacobianPoint x2T2 = scalar_mul(proof.T2, x2);

    JacobianPoint tmp_rhs = jacobian_add(z2V, deltaH);
    JacobianPoint RHS = jacobian_add(tmp_rhs, xT1);
    tmp_rhs = jacobian_add(RHS, x2T2);

    // Compare LHS == RHS via Jacobian cross-multiply (0 field_inv)
    {
        FieldElement z1sq = field_sqr(LHS.z);
        FieldElement z2sq = field_sqr(tmp_rhs.z);
        FieldElement z1cu = field_mul(z1sq, LHS.z);
        FieldElement z2cu = field_mul(z2sq, tmp_rhs.z);

        FieldElement lx = field_mul(LHS.x, z2sq);
        FieldElement rx_cmp = field_mul(tmp_rhs.x, z1sq);
        FieldElement ly = field_mul(LHS.y, z2cu);
        FieldElement ry = field_mul(tmp_rhs.y, z1cu);

        FieldElement dx = field_sub(lx, rx_cmp);
        FieldElement dy = field_sub(ly, ry);

        uchar dx_b[32], dy_b[32];
        field_to_bytes(dx, dx_b);
        field_to_bytes(dy, dy_b);
        for (int i = 0; i < 32; i++)
            if (dx_b[i] != 0 || dy_b[i] != 0) return false;
        return true;
    }
}

// =============================================================================
// 5. Pedersen Commitments
// =============================================================================
// Batch Pedersen commitment generation: C_i = v_i * H + r_i * G
// Ported from CUDA pedersen.cuh.

inline JacobianPoint pedersen_commit(thread const Scalar256 &value,
                                      thread const Scalar256 &blinding,
                                      thread const AffinePoint &H)
{
    JacobianPoint vH = scalar_mul(H, value);
    JacobianPoint rG = scalar_mul_generator_windowed(blinding);
    return jacobian_add(vH, rG);
}
