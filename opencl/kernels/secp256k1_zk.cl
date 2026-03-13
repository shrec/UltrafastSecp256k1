// =============================================================================
// UltrafastSecp256k1 OpenCL -- Zero-Knowledge Proof Kernels
// =============================================================================
// ZK proof primitives for OpenCL:
//   1. Knowledge proof (Schnorr sigma protocol) -- prove + verify
//   2. DLEQ proof (discrete log equality) -- prove + verify
//   3. Bulletproof range proof (64-bit) -- verify
//
// Depends on: secp256k1_extended.cl (which includes field, point, scalar, SHA-256)
// Uses 4x64-bit limbs (ulong) -- matching existing OpenCL convention.
// =============================================================================

#include "secp256k1_extended.cl"

// Alias: existing code uses jacobian_add_impl, point.cl defines point_add_impl
#define jacobian_add_impl point_add_impl

// =============================================================================
// ZK Proof Structures
// =============================================================================

typedef struct {
    uchar rx[32];  // R.x (x-coordinate of nonce point, even Y)
    Scalar s;       // response scalar
} ZKKnowledgeProof;

typedef struct {
    Scalar e;  // challenge
    Scalar s;  // response
} ZKDLEQProof;

// =============================================================================
// Helper: Jacobian to Affine (inline)
// =============================================================================

inline void jacobian_to_affine_impl(const JacobianPoint* p,
                                     FieldElement* ax, FieldElement* ay) {
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv_impl(&z_inv, &p->z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv2, &z_inv);
    field_mul_impl(ax, &p->x, &z_inv2);
    field_mul_impl(ay, &p->y, &z_inv3);
}

// =============================================================================
// Helper: Point to Compressed (33 bytes: prefix || x)
// =============================================================================

inline void point_to_compressed_impl(const JacobianPoint* p,
                                      uchar out[33]) {
    if (point_is_infinity(p)) {
        for (int i = 0; i < 33; ++i) out[i] = 0;
        return;
    }
    FieldElement ax, ay;
    jacobian_to_affine_impl(p, &ax, &ay);

    uchar y_bytes[32];
    field_to_bytes_impl(&ay, y_bytes);
    out[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
    field_to_bytes_impl(&ax, out + 1);
}

// =============================================================================
// ZK Tagged Hash
// =============================================================================

inline void zk_tagged_hash_impl(const uchar* tag, uint tag_len,
                                 const uchar* data, uint data_len,
                                 uchar out[32]) {
    tagged_hash_impl(tag, tag_len, data, data_len, out);
}

// =============================================================================
// ZK Nonce Derivation
// =============================================================================
// k = H("ZK/nonce" || (secret XOR H(aux)) || point_compressed || msg || aux)

inline void zk_derive_nonce_impl(const Scalar* secret,
                                  const JacobianPoint* point,
                                  const uchar msg[32],
                                  const uchar aux[32],
                                  Scalar* k_out) {
    // Hash aux for XOR hedging
    uchar aux_hash[32];
    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, aux, 32);
    sha256_final(&ctx, aux_hash);

    // masked = secret_bytes XOR aux_hash
    uchar sec_bytes[32];
    scalar_to_bytes_impl(secret, sec_bytes);
    uchar masked[32];
    for (int i = 0; i < 32; ++i) masked[i] = sec_bytes[i] ^ aux_hash[i];

    // Compress point
    uchar pt_comp[33];
    JacobianPoint pt_copy = *point;
    point_to_compressed_impl(&pt_copy, pt_comp);

    // buf = masked[32] || pt_comp[33] || msg[32] || aux[32] = 129 bytes
    uchar buf[32 + 33 + 32 + 32];
    for (int i = 0; i < 32; ++i) buf[i] = masked[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = pt_comp[i];
    for (int i = 0; i < 32; ++i) buf[65 + i] = msg[i];
    for (int i = 0; i < 32; ++i) buf[97 + i] = aux[i];

    uchar hash[32];
    uchar tag[] = {'Z','K','/','n','o','n','c','e'};
    zk_tagged_hash_impl(tag, 8, buf, sizeof(buf), hash);
    scalar_from_bytes_impl(hash, k_out);
}

// =============================================================================
// 1a. Knowledge Proof -- Proving
// =============================================================================
// Proves knowledge of secret s such that P = s * B for arbitrary base B.

inline int zk_knowledge_prove_impl(
    const Scalar* secret,
    const JacobianPoint* pubkey,
    const JacobianPoint* base,
    const uchar msg[32],
    const uchar aux[32],
    ZKKnowledgeProof* proof)
{
    // k = deterministic nonce
    Scalar k;
    zk_derive_nonce_impl(secret, pubkey, msg, aux, &k);
    if (scalar_is_zero(&k)) return 0;

    // R = k * base
    JacobianPoint R;
    JacobianPoint base_copy = *base;
    scalar_mul_impl(&R, &base_copy, &k);

    // Convert R to affine
    FieldElement rx_fe, ry_fe;
    jacobian_to_affine_impl(&R, &rx_fe, &ry_fe);

    // Ensure even Y: if Y is odd, negate k
    uchar ry_bytes[32];
    field_to_bytes_impl(&ry_fe, ry_bytes);
    Scalar k_eff;
    if (ry_bytes[31] & 1) {
        scalar_negate_impl(&k, &k_eff);
        field_neg_impl(&ry_fe, &ry_fe);
    } else {
        k_eff = k;
    }

    // Store R.x
    field_to_bytes_impl(&rx_fe, proof->rx);

    // e = H("ZK/knowledge" || R.x || P_comp || B_comp || msg)
    uchar p_comp[33], b_comp[33];
    JacobianPoint pk_copy = *pubkey;
    JacobianPoint base_copy2 = *base;
    point_to_compressed_impl(&pk_copy, p_comp);
    point_to_compressed_impl(&base_copy2, b_comp);

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

    // s = k_eff + e * secret
    Scalar e_sec;
    scalar_mul_mod_n_impl(&e, secret, &e_sec);
    scalar_add_mod_n_impl(&k_eff, &e_sec, &proof->s);

    return 1;
}

// =============================================================================
// 1b. Knowledge Proof -- Verification
// =============================================================================

inline int zk_knowledge_verify_impl(
    const ZKKnowledgeProof* proof,
    const JacobianPoint* pubkey,
    const JacobianPoint* base,
    const uchar msg[32])
{
    // Reconstruct challenge e
    uchar p_comp[33], b_comp[33];
    JacobianPoint pk_copy = *pubkey;
    JacobianPoint base_copy = *base;
    point_to_compressed_impl(&pk_copy, p_comp);
    point_to_compressed_impl(&base_copy, b_comp);

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

    // Verify: s*B == R + e*P
    JacobianPoint sB;
    JacobianPoint base_copy2 = *base;
    scalar_mul_impl(&sB, &base_copy2, &proof->s);

    JacobianPoint eP;
    JacobianPoint pk_copy2 = *pubkey;
    scalar_mul_impl(&eP, &pk_copy2, &e);

    // R = lift_x(proof->rx) with even Y
    JacobianPoint R_pt;
    if (!lift_x_impl(proof->rx, &R_pt)) return 0;

    // R + e*P
    JacobianPoint R_plus_eP;
    jacobian_add_impl(&R_plus_eP, &R_pt, &eP);

    // Compare s*B == R + e*P via compressed form
    uchar comp1[33], comp2[33];
    point_to_compressed_impl(&sB, comp1);
    point_to_compressed_impl(&R_plus_eP, comp2);

    for (int i = 0; i < 33; ++i)
        if (comp1[i] != comp2[i]) return 0;
    return 1;
}

// =============================================================================
// 2a. DLEQ Proof -- Proving
// =============================================================================
// Proves log_G(P) == log_H(Q) without revealing the discrete log.

inline int zk_dleq_prove_impl(
    const Scalar* secret,
    const JacobianPoint* G_pt,
    const JacobianPoint* H_pt,
    const JacobianPoint* P_pt,
    const JacobianPoint* Q_pt,
    const uchar aux[32],
    ZKDLEQProof* proof)
{
    // Derive nonce using Q_compressed as msg
    uchar q_comp[33];
    JacobianPoint q_copy = *Q_pt;
    point_to_compressed_impl(&q_copy, q_comp);

    Scalar k;
    zk_derive_nonce_impl(secret, P_pt, q_comp, aux, &k);
    if (scalar_is_zero(&k)) return 0;

    // R1 = k * G, R2 = k * H
    JacobianPoint R1, R2;
    JacobianPoint g_copy = *G_pt, h_copy = *H_pt;
    scalar_mul_impl(&R1, &g_copy, &k);
    scalar_mul_impl(&R2, &h_copy, &k);

    // Serialize all 6 points
    uchar g_comp[33], h_comp[33], p_comp[33];
    uchar r1_comp[33], r2_comp[33];
    JacobianPoint g_copy2 = *G_pt, h_copy2 = *H_pt, p_copy = *P_pt;
    point_to_compressed_impl(&g_copy2, g_comp);
    point_to_compressed_impl(&h_copy2, h_comp);
    point_to_compressed_impl(&p_copy, p_comp);
    point_to_compressed_impl(&R1, r1_comp);
    point_to_compressed_impl(&R2, r2_comp);

    // e = H("ZK/dleq" || G || H || P || Q || R1 || R2)
    uchar buf[33 * 6];
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
    zk_tagged_hash_impl(tag, 7, buf, sizeof(buf), e_hash);
    scalar_from_bytes_impl(e_hash, &proof->e);

    // s = k + e * secret
    Scalar e_sec;
    scalar_mul_mod_n_impl(&proof->e, secret, &e_sec);
    scalar_add_mod_n_impl(&k, &e_sec, &proof->s);

    return 1;
}

// =============================================================================
// 2b. DLEQ Proof -- Verification
// =============================================================================

inline int zk_dleq_verify_impl(
    const ZKDLEQProof* proof,
    const JacobianPoint* G_pt,
    const JacobianPoint* H_pt,
    const JacobianPoint* P_pt,
    const JacobianPoint* Q_pt)
{
    // R1 = s*G - e*P, R2 = s*H - e*Q
    JacobianPoint sG, eP, sH, eQ;
    JacobianPoint g_copy = *G_pt, h_copy = *H_pt;
    JacobianPoint p_copy = *P_pt, q_copy = *Q_pt;
    scalar_mul_impl(&sG, &g_copy, &proof->s);
    scalar_mul_impl(&eP, &p_copy, &proof->e);
    scalar_mul_impl(&sH, &h_copy, &proof->s);
    scalar_mul_impl(&eQ, &q_copy, &proof->e);

    // Negate eP and eQ
    field_neg_impl(&eP.y, &eP.y);
    field_neg_impl(&eQ.y, &eQ.y);

    JacobianPoint R1, R2;
    jacobian_add_impl(&R1, &sG, &eP);
    jacobian_add_impl(&R2, &sH, &eQ);

    // Serialize all 6 points for challenge recomputation
    uchar g_comp[33], h_comp[33], p_comp[33], q_comp[33];
    uchar r1_comp[33], r2_comp[33];
    JacobianPoint g_copy2 = *G_pt, h_copy2 = *H_pt;
    JacobianPoint p_copy2 = *P_pt, q_copy2 = *Q_pt;
    point_to_compressed_impl(&g_copy2, g_comp);
    point_to_compressed_impl(&h_copy2, h_comp);
    point_to_compressed_impl(&p_copy2, p_comp);
    point_to_compressed_impl(&q_copy2, q_comp);
    point_to_compressed_impl(&R1, r1_comp);
    point_to_compressed_impl(&R2, r2_comp);

    uchar buf[33 * 6];
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
    zk_tagged_hash_impl(tag, 7, buf, sizeof(buf), e_hash);

    Scalar e_check;
    scalar_from_bytes_impl(e_hash, &e_check);

    // e must match
    return scalar_eq_impl(&proof->e, &e_check);
}

// =============================================================================
// Batch Kernel Entry Points
// =============================================================================

__kernel void zk_knowledge_prove_batch(
    __global const Scalar* secrets,
    __global const JacobianPoint* pubkeys,
    __global const JacobianPoint* bases,
    __global const uchar* messages,
    __global const uchar* aux_rands,
    __global ZKKnowledgeProof* proofs,
    __global int* results,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    Scalar sec = secrets[gid];
    JacobianPoint pk = pubkeys[gid];
    JacobianPoint base = bases[gid];

    uchar msg[32], aux[32];
    for (int i = 0; i < 32; ++i) { msg[i] = messages[gid * 32 + i]; aux[i] = aux_rands[gid * 32 + i]; }

    ZKKnowledgeProof proof;
    results[gid] = zk_knowledge_prove_impl(&sec, &pk, &base, msg, aux, &proof);
    proofs[gid] = proof;
}

__kernel void zk_knowledge_verify_batch(
    __global const ZKKnowledgeProof* proofs,
    __global const JacobianPoint* pubkeys,
    __global const JacobianPoint* bases,
    __global const uchar* messages,
    __global int* results,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    ZKKnowledgeProof proof = proofs[gid];
    JacobianPoint pk = pubkeys[gid];
    JacobianPoint base = bases[gid];

    uchar msg[32];
    for (int i = 0; i < 32; ++i) msg[i] = messages[gid * 32 + i];

    results[gid] = zk_knowledge_verify_impl(&proof, &pk, &base, msg);
}

__kernel void zk_dleq_prove_batch(
    __global const Scalar* secrets,
    __global const JacobianPoint* G_pts,
    __global const JacobianPoint* H_pts,
    __global const JacobianPoint* P_pts,
    __global const JacobianPoint* Q_pts,
    __global const uchar* aux_rands,
    __global ZKDLEQProof* proofs,
    __global int* results,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    Scalar sec = secrets[gid];
    JacobianPoint G = G_pts[gid], H = H_pts[gid];
    JacobianPoint P = P_pts[gid], Q = Q_pts[gid];

    uchar aux[32];
    for (int i = 0; i < 32; ++i) aux[i] = aux_rands[gid * 32 + i];

    ZKDLEQProof proof;
    results[gid] = zk_dleq_prove_impl(&sec, &G, &H, &P, &Q, aux, &proof);
    proofs[gid] = proof;
}

__kernel void zk_dleq_verify_batch(
    __global const ZKDLEQProof* proofs,
    __global const JacobianPoint* G_pts,
    __global const JacobianPoint* H_pts,
    __global const JacobianPoint* P_pts,
    __global const JacobianPoint* Q_pts,
    __global int* results,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    ZKDLEQProof proof = proofs[gid];
    JacobianPoint G = G_pts[gid], H = H_pts[gid];
    JacobianPoint P = P_pts[gid], Q = Q_pts[gid];

    results[gid] = zk_dleq_verify_impl(&proof, &G, &H, &P, &Q);
}

// =============================================================================
// 3. Bulletproof Range Proof (64-bit)
// =============================================================================
// Full Bulletproof range proof verification on OpenCL.
// Ported from CUDA implementation (commit 02ac59d).
//
// Architecture:
//   - Single work-item per proof (no subgroup cooperation)
//   - Generator tables stored in __global device memory
//   - Tagged hash midstates for Fiat-Shamir (precomputed once)
// =============================================================================

#define BP_BITS  64
#define BP_LOG2  6

// -- Tagged Hash Midstate --
// SHA256 state after processing SHA256(tag)||SHA256(tag) (1 block = 64 bytes).
typedef struct {
    uint h[8];
} ZKTagMidstate;

// Precomputed midstates (same as CUDA)
__constant const ZKTagMidstate ZK_BULLETPROOF_Y_MIDSTATE = {{
    0x770918afU, 0xa4791204U, 0x3c076a40U, 0x5fb23056U,
    0x902acdb9U, 0x1d85371bU, 0x10f624c4U, 0x9048ba46U
}};

__constant const ZKTagMidstate ZK_BULLETPROOF_Z_MIDSTATE = {{
    0x22be001aU, 0x3c79431bU, 0xe60a9432U, 0xfd965d54U,
    0x84df949fU, 0x62937ceeU, 0x20924a62U, 0x99f23a35U
}};

__constant const ZKTagMidstate ZK_BULLETPROOF_X_MIDSTATE = {{
    0x1378a3c8U, 0x2e8ad1b2U, 0xa47ce2e2U, 0x143037a2U,
    0xbaec0bd8U, 0x40cb0ed7U, 0xd1b23b65U, 0x43871df4U
}};

// Tagged hash using precomputed midstate
inline void zk_tagged_hash_midstate_impl(const ZKTagMidstate* midstate,
                                          const uchar* data, uint data_len,
                                          uchar out[32]) {
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = midstate->h[i];
    ctx.buf_len = 0;
    ctx.total = 64;  // already processed 1 block (tag_hash||tag_hash)
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

// Inline field_from_bytes (big-endian bytes -> 4x64 limbs, no modular reduction)
inline void field_from_bytes_impl(const uchar bytes[32], FieldElement* out) {
    for (int i = 0; i < 4; i++) {
        ulong limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++)
            limb = (limb << 8) | (ulong)bytes[base + j];
        out->limbs[i] = limb;
    }
}

// lift_x with even Y from FieldElement (not bytes)
inline int lift_x_field_even_impl(const FieldElement* x, AffinePoint* out) {
    FieldElement x2, x3, y2, seven, y;
    field_sqr_impl(&x2, x);
    field_mul_impl(&x3, &x2, x);
    seven.limbs[0] = 7; seven.limbs[1] = 0; seven.limbs[2] = 0; seven.limbs[3] = 0;
    field_add_impl(&y2, &x3, &seven);
    field_sqrt_impl(&y2, &y);

    // Verify: y^2 == x^3 + 7
    FieldElement y_check;
    field_sqr_impl(&y_check, &y);
    uchar yc_bytes[32], y2_bytes[32];
    field_to_bytes_impl(&y_check, yc_bytes);
    field_to_bytes_impl(&y2, y2_bytes);
    int valid = 1;
    for (int i = 0; i < 32; i++)
        if (yc_bytes[i] != y2_bytes[i]) valid = 0;
    if (!valid) return 0;

    // Ensure even Y
    uchar y_bytes[32];
    field_to_bytes_impl(&y, y_bytes);
    if (y_bytes[31] & 1) field_neg_impl(&y, &y);

    out->x = *x;
    out->y = y;
    return 1;
}

// Try-and-increment: find point on curve starting from x
inline void hash_to_point_increment_impl(FieldElement* x, AffinePoint* out) {
    for (int attempt = 0; attempt < 256; ++attempt) {
        if (lift_x_field_even_impl(x, out)) return;
        // x += 1 (field addition with constant 1)
        FieldElement one;
        one.limbs[0] = 1; one.limbs[1] = 0; one.limbs[2] = 0; one.limbs[3] = 0;
        field_add_impl(x, x, &one);
    }
}

// Affine point to 33-byte compressed (prefix || x_bytes)
inline void affine_to_compressed_impl(const FieldElement* x, const FieldElement* y,
                                       uchar out[33]) {
    uchar y_bytes[32];
    field_to_bytes_impl(y, y_bytes);
    out[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
    field_to_bytes_impl(x, out + 1);
}

// -- Bulletproof Range Proof Structure --
typedef struct {
    AffinePoint A, S;         // vector commitments
    AffinePoint T1, T2;       // polynomial commitments
    Scalar tau_x, mu, t_hat;  // blinding, aggregate blinding, poly eval
    Scalar a, b;              // final IPA scalars
    AffinePoint L[6], R[6];   // inner product argument rounds (log2(64)=6)
} RangeProofGPU;

// =============================================================================
// Bulletproof Generator Init (called once)
// =============================================================================
// Computes 128 generator points (G_0..G_63, H_0..H_63) via tagged hash +
// try-and-increment. Also computes "Bulletproof/ip" midstate for IPA hashes.
// Generators are stored in __global buffers provided by host.

__kernel void bulletproof_init_kernel(
    __global AffinePoint* bp_G,       // output: 64 G_i generators
    __global AffinePoint* bp_H,       // output: 64 H_i generators
    __global ZKTagMidstate* bp_ip_midstate)  // output: "Bulletproof/ip" midstate
{
    uint gid = get_global_id(0);
    if (gid != 0) return;

    // Compute "Bulletproof/ip" midstate
    {
        uchar tag[14]; tag[0]='B'; tag[1]='u'; tag[2]='l'; tag[3]='l'; tag[4]='e';
        tag[5]='t'; tag[6]='p'; tag[7]='r'; tag[8]='o'; tag[9]='o'; tag[10]='f';
        tag[11]='/'; tag[12]='i'; tag[13]='p';
        uchar tag_hash[32];
        SHA256Ctx ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, tag, 14);
        sha256_final(&ctx, tag_hash);
        sha256_init(&ctx);
        sha256_update(&ctx, tag_hash, 32);
        sha256_update(&ctx, tag_hash, 32);
        for (int i = 0; i < 8; i++) bp_ip_midstate->h[i] = ctx.h[i];
    }

    // Compute "Bulletproof/gen" midstate
    ZKTagMidstate gen_midstate;
    {
        uchar tag[15]; tag[0]='B'; tag[1]='u'; tag[2]='l'; tag[3]='l'; tag[4]='e';
        tag[5]='t'; tag[6]='p'; tag[7]='r'; tag[8]='o'; tag[9]='o'; tag[10]='f';
        tag[11]='/'; tag[12]='g'; tag[13]='e'; tag[14]='n';
        uchar tag_hash[32];
        SHA256Ctx ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, tag, 15);
        sha256_final(&ctx, tag_hash);
        sha256_init(&ctx);
        sha256_update(&ctx, tag_hash, 32);
        sha256_update(&ctx, tag_hash, 32);
        for (int i = 0; i < 8; i++) gen_midstate.h[i] = ctx.h[i];
    }

    // Generate 64 G_i and 64 H_i
    for (int i = 0; i < 64; i++) {
        uchar buf[5];
        buf[1] = (uchar)(i & 0xFF);
        buf[2] = (uchar)((i >> 8) & 0xFF);
        buf[3] = (uchar)((i >> 16) & 0xFF);
        buf[4] = (uchar)((i >> 24) & 0xFF);

        uchar hash[32];
        FieldElement x;

        // G_i
        buf[0] = 'G';
        zk_tagged_hash_midstate_impl(&gen_midstate, buf, 5, hash);
        field_from_bytes_impl(hash, &x);
        hash_to_point_increment_impl(&x, &bp_G[i]);

        // H_i
        buf[0] = 'H';
        zk_tagged_hash_midstate_impl(&gen_midstate, buf, 5, hash);
        field_from_bytes_impl(hash, &x);
        hash_to_point_increment_impl(&x, &bp_H[i]);
    }
}

// =============================================================================
// Bulletproof Full Verify (single work-item per proof)
// =============================================================================

inline int range_verify_full_impl(
    const RangeProofGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    const AffinePoint* bp_G,     // 64 G_i generators
    const AffinePoint* bp_H,     // 64 H_i generators
    const ZKTagMidstate* bp_ip_midstate)
{
    // ---- Fiat-Shamir: recompute y, z, x ----
    uchar a_comp[33], s_comp[33], v_comp[33];
    affine_to_compressed_impl(&proof->A.x, &proof->A.y, a_comp);
    affine_to_compressed_impl(&proof->S.x, &proof->S.y, s_comp);
    affine_to_compressed_impl(&commitment->x, &commitment->y, v_comp);

    uchar fs_buf[99]; // 33 + 33 + 33
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

    uchar t1_comp[33], t2_comp[33];
    affine_to_compressed_impl(&proof->T1.x, &proof->T1.y, t1_comp);
    affine_to_compressed_impl(&proof->T2.x, &proof->T2.y, t2_comp);

    uchar x_buf[130]; // 33 + 33 + 32 + 32
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
    scalar_to_bytes_impl(&y, x_buf + 66);
    scalar_to_bytes_impl(&z, x_buf + 98);

    uchar x_hash[32];
    zk_tagged_hash_midstate_impl(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, 130, x_hash);
    Scalar x;
    scalar_from_bytes_impl(x_hash, &x);

    // ---- Compute delta(y,z) ----
    Scalar z2, z3, x2;
    scalar_mul_mod_n_impl(&z, &z, &z2);
    scalar_mul_mod_n_impl(&z2, &z, &z3);
    scalar_mul_mod_n_impl(&x, &x, &x2);

    // sum(y^i) for i in [0, 64)
    Scalar sum_y;
    sum_y.limbs[0] = 1; sum_y.limbs[1] = 0; sum_y.limbs[2] = 0; sum_y.limbs[3] = 0;
    Scalar y_pow = y;
    for (int i = 1; i < BP_BITS; ++i) {
        scalar_add_mod_n_impl(&sum_y, &y_pow, &sum_y);
        scalar_mul_mod_n_impl(&y_pow, &y, &y_pow);
    }

    // sum(2^i) for i in [0, 64) = 2^64 - 1
    Scalar sum_2;
    sum_2.limbs[0] = 0xFFFFFFFFFFFFFFFFUL;
    sum_2.limbs[1] = 0; sum_2.limbs[2] = 0; sum_2.limbs[3] = 0;

    Scalar z_minus_z2, term1, term2, delta;
    scalar_sub_mod_n_impl(&z, &z2, &z_minus_z2);
    scalar_mul_mod_n_impl(&z_minus_z2, &sum_y, &term1);
    scalar_mul_mod_n_impl(&z3, &sum_2, &term2);
    scalar_sub_mod_n_impl(&term1, &term2, &delta);

    // ---- Polynomial check ----
    // LHS = t_hat * H + tau_x * G
    JacobianPoint tH, tauG, LHS;
    scalar_mul_impl(&tH, &proof->t_hat, H_gen);
    scalar_mul_generator_impl(&tauG, &proof->tau_x);
    point_add_impl(&LHS, &tH, &tauG);

    // RHS = z^2 * V + delta * H + x * T1 + x^2 * T2
    JacobianPoint z2V, deltaH, xT1, x2T2;
    scalar_mul_impl(&z2V, &z2, commitment);
    scalar_mul_impl(&deltaH, &delta, H_gen);
    scalar_mul_impl(&xT1, &x, &proof->T1);
    scalar_mul_impl(&x2T2, &x2, &proof->T2);

    JacobianPoint RHS, tmp_rhs;
    point_add_impl(&tmp_rhs, &z2V, &deltaH);
    point_add_impl(&RHS, &tmp_rhs, &xT1);
    point_add_impl(&tmp_rhs, &RHS, &x2T2);

    // Compare LHS == RHS via Jacobian cross-multiply (0 field_inv)
    {
        FieldElement z1sq, z2sq, z1cu, z2cu;
        field_sqr_impl(&z1sq, &LHS.z);
        field_sqr_impl(&z2sq, &tmp_rhs.z);
        field_mul_impl(&z1cu, &z1sq, &LHS.z);
        field_mul_impl(&z2cu, &z2sq, &tmp_rhs.z);

        FieldElement lx, rx_cmp, ly, ry;
        field_mul_impl(&lx, &LHS.x, &z2sq);
        field_mul_impl(&rx_cmp, &tmp_rhs.x, &z1sq);
        field_mul_impl(&ly, &LHS.y, &z2cu);
        field_mul_impl(&ry, &tmp_rhs.y, &z1cu);

        FieldElement dx, dy;
        field_sub_impl(&dx, &lx, &rx_cmp);
        field_sub_impl(&dy, &ly, &ry);

        uchar dx_b[32], dy_b[32];
        field_to_bytes_impl(&dx, dx_b);
        field_to_bytes_impl(&dy, dy_b);
        int all_zero = 1;
        for (int i = 0; i < 32; i++)
            if (dx_b[i] != 0 || dy_b[i] != 0) all_zero = 0;
        if (!all_zero) return 0;
    }

    // ---- Inner Product Argument verification ----
    Scalar x_rounds[BP_LOG2];
    for (int round = 0; round < BP_LOG2; ++round) {
        uchar l_comp[33], r_comp[33];
        affine_to_compressed_impl(&proof->L[round].x, &proof->L[round].y, l_comp);
        affine_to_compressed_impl(&proof->R[round].x, &proof->R[round].y, r_comp);
        uchar ip_buf[66]; // 33 + 33
        for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
        uchar xr_hash[32];
        zk_tagged_hash_midstate_impl(bp_ip_midstate, ip_buf, 66, xr_hash);
        scalar_from_bytes_impl(xr_hash, &x_rounds[round]);
    }

    // Batch inversion of x_rounds
    Scalar x_inv_rounds[BP_LOG2];
    {
        Scalar acc[BP_LOG2];
        acc[0] = x_rounds[0];
        for (int j = 1; j < BP_LOG2; ++j) scalar_mul_mod_n_impl(&acc[j-1], &x_rounds[j], &acc[j]);
        Scalar inv_acc;
        scalar_inverse_impl(&acc[BP_LOG2 - 1], &inv_acc);
        for (int j = BP_LOG2 - 1; j >= 1; --j) {
            scalar_mul_mod_n_impl(&inv_acc, &acc[j-1], &x_inv_rounds[j]);
            scalar_mul_mod_n_impl(&inv_acc, &x_rounds[j], &inv_acc);
        }
        x_inv_rounds[0] = inv_acc;
    }

    // y_inv and y_inv_powers
    Scalar y_inv;
    scalar_inverse_impl(&y, &y_inv);
    Scalar y_inv_powers[BP_BITS];
    y_inv_powers[0].limbs[0] = 1; y_inv_powers[0].limbs[1] = 0;
    y_inv_powers[0].limbs[2] = 0; y_inv_powers[0].limbs[3] = 0;
    for (int i = 1; i < BP_BITS; ++i)
        scalar_mul_mod_n_impl(&y_inv_powers[i-1], &y_inv, &y_inv_powers[i]);

    // s_coeff: product tree of x_rounds / x_inv_rounds
    Scalar s_coeff[BP_BITS];
    s_coeff[0].limbs[0] = 1; s_coeff[0].limbs[1] = 0;
    s_coeff[0].limbs[2] = 0; s_coeff[0].limbs[3] = 0;
    for (int j = 0; j < BP_LOG2; ++j)
        scalar_mul_mod_n_impl(&s_coeff[0], &x_inv_rounds[j], &s_coeff[0]);
    for (int i = 1; i < BP_BITS; ++i) {
        s_coeff[i].limbs[0] = 1; s_coeff[i].limbs[1] = 0;
        s_coeff[i].limbs[2] = 0; s_coeff[i].limbs[3] = 0;
        for (int jj = 0; jj < BP_LOG2; ++jj) {
            if ((i >> (BP_LOG2 - 1 - jj)) & 1)
                scalar_mul_mod_n_impl(&s_coeff[i], &x_rounds[jj], &s_coeff[i]);
            else
                scalar_mul_mod_n_impl(&s_coeff[i], &x_inv_rounds[jj], &s_coeff[i]);
        }
    }

    // Batch inversion for s_inv
    Scalar s_inv[BP_BITS];
    {
        Scalar acc[BP_BITS];
        acc[0] = s_coeff[0];
        for (int i = 1; i < BP_BITS; ++i) scalar_mul_mod_n_impl(&acc[i-1], &s_coeff[i], &acc[i]);
        Scalar inv_acc;
        scalar_inverse_impl(&acc[BP_BITS - 1], &inv_acc);
        for (int i = BP_BITS - 1; i >= 1; --i) {
            scalar_mul_mod_n_impl(&inv_acc, &acc[i-1], &s_inv[i]);
            scalar_mul_mod_n_impl(&inv_acc, &s_coeff[i], &inv_acc);
        }
        s_inv[0] = inv_acc;
    }

    // two_powers: 2^i
    Scalar two_powers[BP_BITS];
    two_powers[0].limbs[0] = 1; two_powers[0].limbs[1] = 0;
    two_powers[0].limbs[2] = 0; two_powers[0].limbs[3] = 0;
    for (int i = 1; i < BP_BITS; ++i)
        scalar_add_mod_n_impl(&two_powers[i-1], &two_powers[i-1], &two_powers[i]);

    // ---- Build MSM ----
    Scalar neg_z;
    scalar_negate_impl(&z, &neg_z);
    Scalar ab;
    scalar_mul_mod_n_impl(&proof->a, &proof->b, &ab);

    JacobianPoint msm_acc;
    msm_acc.infinity = 1;
    msm_acc.z.limbs[0] = 1; msm_acc.z.limbs[1] = 0;
    msm_acc.z.limbs[2] = 0; msm_acc.z.limbs[3] = 0;

    // A (coefficient 1)
    {
        point_add_mixed_impl(&msm_acc, &msm_acc, &proof->A);
    }

    // x * S
    {
        JacobianPoint xS;
        scalar_mul_impl(&xS, &x, &proof->S);
        point_add_impl(&msm_acc, &msm_acc, &xS);
    }

    // G_i and H_i contributions
    for (int i = 0; i < BP_BITS; ++i) {
        // G_i: (-z - a*s_i)
        Scalar a_si, g_coeff;
        scalar_mul_mod_n_impl(&proof->a, &s_coeff[i], &a_si);
        scalar_sub_mod_n_impl(&neg_z, &a_si, &g_coeff);

        JacobianPoint g_term;
        scalar_mul_impl(&g_term, &g_coeff, &bp_G[i]);
        point_add_impl(&msm_acc, &msm_acc, &g_term);

        // H_i: (z + z2*2^i*y_inv^i) - b*s_inv[i]*y_inv^i
        Scalar z2_2i, z2_2i_yi, h_pcheck;
        scalar_mul_mod_n_impl(&z2, &two_powers[i], &z2_2i);
        scalar_mul_mod_n_impl(&z2_2i, &y_inv_powers[i], &z2_2i_yi);
        scalar_add_mod_n_impl(&z, &z2_2i_yi, &h_pcheck);

        Scalar b_si, b_si_yi, h_coeff;
        scalar_mul_mod_n_impl(&proof->b, &s_inv[i], &b_si);
        scalar_mul_mod_n_impl(&b_si, &y_inv_powers[i], &b_si_yi);
        scalar_sub_mod_n_impl(&h_pcheck, &b_si_yi, &h_coeff);

        JacobianPoint h_term;
        scalar_mul_impl(&h_term, &h_coeff, &bp_H[i]);
        point_add_impl(&msm_acc, &msm_acc, &h_term);
    }

    // -mu * G
    {
        Scalar neg_mu;
        scalar_negate_impl(&proof->mu, &neg_mu);
        JacobianPoint muG;
        scalar_mul_generator_impl(&muG, &neg_mu);
        point_add_impl(&msm_acc, &msm_acc, &muG);
    }

    // (t_hat - a*b) * U (H_ped)
    {
        Scalar t_ab;
        scalar_sub_mod_n_impl(&proof->t_hat, &ab, &t_ab);
        JacobianPoint tU;
        scalar_mul_impl(&tU, &t_ab, H_gen);
        point_add_impl(&msm_acc, &msm_acc, &tU);
    }

    // L_j and R_j contributions
    for (int j = 0; j < BP_LOG2; ++j) {
        Scalar xj2, xj_inv2;
        scalar_mul_mod_n_impl(&x_rounds[j], &x_rounds[j], &xj2);
        scalar_mul_mod_n_impl(&x_inv_rounds[j], &x_inv_rounds[j], &xj_inv2);

        JacobianPoint lterm, rterm;
        scalar_mul_impl(&lterm, &xj2, &proof->L[j]);
        scalar_mul_impl(&rterm, &xj_inv2, &proof->R[j]);
        point_add_impl(&msm_acc, &msm_acc, &lterm);
        point_add_impl(&msm_acc, &msm_acc, &rterm);
    }

    // Check: msm_acc should be identity
    if (msm_acc.infinity) return 1;

    // Check Z == 0 via bytes (handles unreduced limbs)
    uchar z_bytes[32];
    field_to_bytes_impl(&msm_acc.z, z_bytes);
    int z_zero = 1;
    for (int i = 0; i < 32; i++)
        if (z_bytes[i] != 0) z_zero = 0;
    return z_zero;
}

// =============================================================================
// Bulletproof Batch Verify Kernel
// =============================================================================

__kernel void bulletproof_verify_batch(
    __global const RangeProofGPU* proofs,
    __global const AffinePoint* commitments,
    __global const AffinePoint* H_gen,        // single Pedersen H
    __global const AffinePoint* bp_G,          // 64 G_i generators
    __global const AffinePoint* bp_H,          // 64 H_i generators
    __global const ZKTagMidstate* bp_ip_midstate,
    __global int* results,
    const uint count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy proof to private memory
    RangeProofGPU proof = proofs[gid];
    AffinePoint commit = commitments[gid];
    AffinePoint h_ped = H_gen[0];
    ZKTagMidstate ip_mid = bp_ip_midstate[0];

    results[gid] = range_verify_full_impl(&proof, &commit, &h_ped,
                                           bp_G, bp_H, &ip_mid);
}
