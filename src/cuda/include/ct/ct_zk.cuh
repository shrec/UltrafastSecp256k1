#pragma once
// ============================================================================
// Constant-Time Zero-Knowledge Proof Kernels -- CUDA Device
// ============================================================================
// GPU-accelerated ZK proof PROVING using CT layer:
//   1. CT knowledge proof (Schnorr sigma protocol)
//   2. CT DLEQ proof (discrete log equality)
//
// All proving operations use the CT layer for side-channel resistance.
// Verification uses the fast path (public data) -- see zk.cuh.
//
// 64-bit limb mode only.
// ============================================================================

#include "ct/ct_point.cuh"
#include "ct/ct_scalar.cuh"
#include "ct/ct_sign.cuh"
#include "zk.cuh"        // KnowledgeProofGPU, DLEQProofGPU, zk_tagged_hash
#include "pedersen.cuh"   // lift_x_even

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {
namespace ct {

// ============================================================================
// Helpers
// ============================================================================

// Compress Jacobian point to 33 bytes (1x field_inv)
__device__ inline void jac_to_compressed(
    const JacobianPoint* p, uint8_t out[33])
{
    FieldElement ax, ay;
    secp256k1::cuda::jacobian_to_affine(p, &ax, &ay);
    affine_to_compressed(&ax, &ay, out);
}

// Compress Jacobian point using pre-computed Z inverse (no field_inv)
// Used with ct_batch_field_inv to amortize inversion cost across N points
__device__ inline void jac_to_compressed_with_zinv(
    const JacobianPoint* p, const FieldElement* z_inv, uint8_t out[33])
{
    FieldElement z_inv2, z_inv3, ax, ay;
    secp256k1::cuda::field_sqr(z_inv, &z_inv2);
    secp256k1::cuda::field_mul(z_inv, &z_inv2, &z_inv3);
    secp256k1::cuda::field_mul(&p->x, &z_inv2, &ax);
    secp256k1::cuda::field_mul(&p->y, &z_inv3, &ay);
    affine_to_compressed(&ax, &ay, out);
}

// ============================================================================
// Deterministic Nonce Derivation (CT)
// ============================================================================
// k = H("ZK/nonce" || (secret XOR H(aux)) || point_compressed || msg || aux)
// Takes pre-compressed point to avoid redundant field_inv.

__device__ inline void ct_zk_derive_nonce(
    const Scalar* secret,
    const uint8_t pt_comp[33],
    const uint8_t msg[32],
    const uint8_t aux[32],
    Scalar* k_out)
{
    // Hash aux for XOR hedging
    uint8_t aux_hash[32];
    sha256_hash(aux, 32, aux_hash);

    // masked = secret_bytes XOR aux_hash (4x uint64 XOR vs 32 byte XOR)
    uint8_t sec_bytes[32];
    secp256k1::cuda::scalar_to_bytes(secret, sec_bytes);
    uint8_t masked[32];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        reinterpret_cast<uint64_t*>(masked)[i] =
            reinterpret_cast<const uint64_t*>(sec_bytes)[i] ^
            reinterpret_cast<const uint64_t*>(aux_hash)[i];

    // buf = masked[32] || pt_comp[33] || msg[32] || aux[32] = 129 bytes
    uint8_t buf[32 + 33 + 32 + 32];
    // Copy masked (32 bytes) via uint64
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        reinterpret_cast<uint64_t*>(buf)[i] =
            reinterpret_cast<const uint64_t*>(masked)[i];
    // pt_comp (33 bytes) -- not 8-byte aligned destination, byte copy
    for (int i = 0; i < 33; ++i) buf[32 + i] = pt_comp[i];
    // msg (32 bytes) at offset 65 -- not 8-byte aligned, byte copy
    for (int i = 0; i < 32; ++i) buf[65 + i] = msg[i];
    // aux (32 bytes) at offset 97 -- not 8-byte aligned, byte copy
    for (int i = 0; i < 32; ++i) buf[97 + i] = aux[i];

    uint8_t hash[32];
    zk_tagged_hash_midstate(&ZK_NONCE_MIDSTATE, buf, sizeof(buf), hash);
    secp256k1::cuda::scalar_from_bytes(hash, k_out);
}

// ============================================================================
// 1. CT Knowledge Proof -- Proving (Schnorr sigma protocol)
// ============================================================================
// Proves knowledge of secret s such that P = s * B for arbitrary base B.
// Protocol:
//   k = derive_nonce(secret, P, msg, aux)
//   R = k * B          (CT scalar mul: k is secret)
//   Ensure even Y for R (BIP-340 style)
//   e = H("ZK/knowledge" || R.x || P_comp || B_comp || msg)
//   s = k_eff + e * secret

__device__ inline bool ct_knowledge_prove_device(
    const Scalar* secret,
    const JacobianPoint* pubkey,
    const JacobianPoint* base,
    const uint8_t msg[32],
    const uint8_t aux[32],
    KnowledgeProofGPU* proof)
{
    // Pre-compress pubkey and base (batch invert 2 Z coords: 1 field_inv + 3 field_mul)
    uint8_t p_comp[33], b_comp[33];
    {
        FieldElement z_in[2] = { pubkey->z, base->z };
        FieldElement z_inv[2];
        ct_batch_field_inv(z_in, z_inv, 2);
        jac_to_compressed_with_zinv(pubkey, &z_inv[0], p_comp);
        jac_to_compressed_with_zinv(base, &z_inv[1], b_comp);
    }

    // k = deterministic nonce (uses pre-compressed pubkey)
    Scalar k;
    ct_zk_derive_nonce(secret, p_comp, msg, aux, &k);
    if (secp256k1::cuda::scalar_is_zero(&k)) return false;

    // R = k * base (CT: k is secret)
    JacobianPoint R;
    ct_scalar_mul(base, &k, &R);

    // Convert R to affine, get Y parity
    FieldElement rx_fe, ry_fe;
    uint8_t r_y_parity;
    ct_jacobian_to_affine(&R, &rx_fe, &ry_fe, &r_y_parity);

    // CT conditional negate k if R has odd Y (BIP-340 style)
    uint64_t odd_mask = bool_to_mask((uint64_t)r_y_parity);
    Scalar k_eff;
    scalar_cneg(&k_eff, &k, odd_mask);

    // Store R.x in proof
    secp256k1::cuda::field_to_bytes(&rx_fe, proof->rx);

    // e = H("ZK/knowledge" || R.x || P_comp || B_comp || msg)
    // Reuse pre-compressed p_comp and b_comp (no extra field_inv)
    uint8_t buf[32 + 33 + 33 + 32]; // rx || P || B || msg
    // rx (32 bytes at offset 0) -- aligned, use uint64
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        reinterpret_cast<uint64_t*>(buf)[i] =
            reinterpret_cast<const uint64_t*>(proof->rx)[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[65 + i] = b_comp[i];
    for (int i = 0; i < 32; ++i) buf[98 + i] = msg[i];

    uint8_t e_hash[32];
    zk_tagged_hash_midstate(&ZK_KNOWLEDGE_MIDSTATE, buf, sizeof(buf), e_hash);

    Scalar e;
    secp256k1::cuda::scalar_from_bytes(e_hash, &e);

    // s = k_eff + e * secret (CT scalar arithmetic)
    Scalar e_sec;
    scalar_mul(&e, secret, &e_sec);
    scalar_add(&k_eff, &e_sec, &proof->s);

    return true;
}

// Convenience: prove knowledge of secret for G (generator)
// Uses precomputed G_COMPRESSED to avoid 1 field_inv for base compression
__device__ inline bool ct_knowledge_prove_generator_device(
    const Scalar* secret,
    const JacobianPoint* pubkey,
    const uint8_t msg[32],
    const uint8_t aux[32],
    KnowledgeProofGPU* proof)
{
    // Pre-compress pubkey once
    uint8_t p_comp[33];
    jac_to_compressed(pubkey, p_comp);

    // k = deterministic nonce (uses pre-compressed pubkey)
    Scalar k;
    ct_zk_derive_nonce(secret, p_comp, msg, aux, &k);
    if (secp256k1::cuda::scalar_is_zero(&k)) return false;

    // R = k * G (CT: k is secret, precomputed generator table)
    JacobianPoint R;
    ct_generator_mul(&k, &R);

    // Convert R to affine, get Y parity
    FieldElement rx_fe, ry_fe;
    uint8_t r_y_parity;
    ct_jacobian_to_affine(&R, &rx_fe, &ry_fe, &r_y_parity);

    // CT conditional negate k if R has odd Y (BIP-340 style)
    uint64_t odd_mask = bool_to_mask((uint64_t)r_y_parity);
    Scalar k_eff;
    scalar_cneg(&k_eff, &k, odd_mask);

    // Store R.x in proof
    secp256k1::cuda::field_to_bytes(&rx_fe, proof->rx);

    // e = H("ZK/knowledge" || R.x || P_comp || G_comp || msg)
    // Uses precomputed G_COMPRESSED -- no field_inv for G
    uint8_t buf[32 + 33 + 33 + 32];
    // rx (32 bytes at offset 0) -- aligned, use uint64
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        reinterpret_cast<uint64_t*>(buf)[i] =
            reinterpret_cast<const uint64_t*>(proof->rx)[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[65 + i] = G_COMPRESSED[i];
    for (int i = 0; i < 32; ++i) buf[98 + i] = msg[i];

    uint8_t e_hash[32];
    zk_tagged_hash_midstate(&ZK_KNOWLEDGE_MIDSTATE, buf, sizeof(buf), e_hash);

    Scalar e;
    secp256k1::cuda::scalar_from_bytes(e_hash, &e);

    // s = k_eff + e * secret
    Scalar e_sec;
    scalar_mul(&e, secret, &e_sec);
    scalar_add(&k_eff, &e_sec, &proof->s);

    return true;
}

// Batch kernel: one thread proves one knowledge proof
__global__ void ct_knowledge_prove_batch_kernel(
    const Scalar* __restrict__ secrets,
    const JacobianPoint* __restrict__ pubkeys,
    const JacobianPoint* __restrict__ bases,
    const uint8_t* __restrict__ messages,    // N * 32 bytes
    const uint8_t* __restrict__ aux_rands,   // N * 32 bytes
    KnowledgeProofGPU* __restrict__ proofs,
    bool* __restrict__ results,
    uint32_t count)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = ct_knowledge_prove_device(
        &secrets[idx], &pubkeys[idx], &bases[idx],
        &messages[idx * 32], &aux_rands[idx * 32],
        &proofs[idx]);
}

// Batch kernel: prove with generator G
__global__ void ct_knowledge_prove_generator_batch_kernel(
    const Scalar* __restrict__ secrets,
    const JacobianPoint* __restrict__ pubkeys,
    const uint8_t* __restrict__ messages,
    const uint8_t* __restrict__ aux_rands,
    KnowledgeProofGPU* __restrict__ proofs,
    bool* __restrict__ results,
    uint32_t count)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = ct_knowledge_prove_generator_device(
        &secrets[idx], &pubkeys[idx],
        &messages[idx * 32], &aux_rands[idx * 32],
        &proofs[idx]);
}

// ============================================================================
// 2. CT DLEQ Proof -- Proving (Discrete Log Equality)
// ============================================================================
// Proves that log_G(P) == log_H(Q) without revealing the discrete log.
// Given: secret s, bases G,H, points P=sG, Q=sH
// Protocol:
//   k = derive_nonce(secret, P, Q_comp, aux)
//   R1 = k * G, R2 = k * H   (CT: k is secret)
//   e = H("ZK/dleq" || G_comp || H_comp || P_comp || Q_comp || R1_comp || R2_comp)
//   s = k + e * secret

__device__ inline bool ct_dleq_prove_device(
    const Scalar* secret,
    const JacobianPoint* G,
    const JacobianPoint* H,
    const JacobianPoint* P,
    const JacobianPoint* Q,
    const uint8_t aux[32],
    DLEQProofGPU* proof)
{
    // Batch invert 4 input Z coords (1 field_inv + 9 field_mul vs 4 field_inv)
    uint8_t g_comp[33], h_comp[33], p_comp[33], q_comp[33];
    {
        FieldElement z_in[4] = { Q->z, G->z, H->z, P->z };
        FieldElement z_inv[4];
        ct_batch_field_inv(z_in, z_inv, 4);
        jac_to_compressed_with_zinv(Q, &z_inv[0], q_comp);
        jac_to_compressed_with_zinv(G, &z_inv[1], g_comp);
        jac_to_compressed_with_zinv(H, &z_inv[2], h_comp);
        jac_to_compressed_with_zinv(P, &z_inv[3], p_comp);
    }

    // Derive nonce using pre-compressed P as pubkey, Q as msg
    Scalar k;
    ct_zk_derive_nonce(secret, p_comp, q_comp, aux, &k);
    if (secp256k1::cuda::scalar_is_zero(&k)) return false;

    // R1 = k * G, R2 = k * H (CT: k is secret)
    JacobianPoint R1, R2;
    ct_scalar_mul(G, &k, &R1);
    ct_scalar_mul(H, &k, &R2);

    // Batch invert R1, R2 Z coords (1 field_inv + 3 field_mul vs 2 field_inv)
    uint8_t r1_comp[33], r2_comp[33];
    {
        FieldElement rz_in[2] = { R1.z, R2.z };
        FieldElement rz_inv[2];
        ct_batch_field_inv(rz_in, rz_inv, 2);
        jac_to_compressed_with_zinv(&R1, &rz_inv[0], r1_comp);
        jac_to_compressed_with_zinv(&R2, &rz_inv[1], r2_comp);
    }

    // e = H("ZK/dleq" || G || H || P || Q || R1 || R2)
    // Reuse pre-compressed g_comp, h_comp, p_comp, q_comp
    uint8_t buf[33 * 6];
    for (int i = 0; i < 33; ++i) {
        buf[i]       = g_comp[i];
        buf[33 + i]  = h_comp[i];
        buf[66 + i]  = p_comp[i];
        buf[99 + i]  = q_comp[i];
        buf[132 + i] = r1_comp[i];
        buf[165 + i] = r2_comp[i];
    }

    uint8_t e_hash[32];
    zk_tagged_hash_midstate(&ZK_DLEQ_MIDSTATE, buf, sizeof(buf), e_hash);
    secp256k1::cuda::scalar_from_bytes(e_hash, &proof->e);

    // s = k + e * secret (CT scalar arithmetic)
    Scalar e_sec;
    scalar_mul(&proof->e, secret, &e_sec);
    scalar_add(&k, &e_sec, &proof->s);

    return true;
}

// Batch kernel: one thread proves one DLEQ proof
__global__ void ct_dleq_prove_batch_kernel(
    const Scalar* __restrict__ secrets,
    const JacobianPoint* __restrict__ G_pts,
    const JacobianPoint* __restrict__ H_pts,
    const JacobianPoint* __restrict__ P_pts,
    const JacobianPoint* __restrict__ Q_pts,
    const uint8_t* __restrict__ aux_rands,
    DLEQProofGPU* __restrict__ proofs,
    bool* __restrict__ results,
    uint32_t count)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = ct_dleq_prove_device(
        &secrets[idx], &G_pts[idx], &H_pts[idx],
        &P_pts[idx], &Q_pts[idx],
        &aux_rands[idx * 32], &proofs[idx]);
}

// ============================================================================
// 3. CT DLEQ Proof -- Generator Specialized
// ============================================================================
// Optimized DLEQ prove when G = standard secp256k1 generator.
// Uses precomputed G_COMPRESSED + ct_generator_mul for R1 = k*G.
// Batch inverts H,P,Q Z coords (3 pts -> 1 field_inv + 6 field_mul).

__device__ inline bool ct_dleq_prove_generator_device(
    const Scalar* secret,
    const JacobianPoint* H,
    const JacobianPoint* P,
    const JacobianPoint* Q,
    const uint8_t aux[32],
    DLEQProofGPU* proof)
{
    // Batch invert 3 input Z coords (1 field_inv + 6 field_mul vs 3 field_inv)
    // G uses precomputed G_COMPRESSED (0 field_inv)
    uint8_t h_comp[33], p_comp[33], q_comp[33];
    {
        FieldElement z_in[3] = { H->z, P->z, Q->z };
        FieldElement z_inv[3];
        ct_batch_field_inv(z_in, z_inv, 3);
        jac_to_compressed_with_zinv(H, &z_inv[0], h_comp);
        jac_to_compressed_with_zinv(P, &z_inv[1], p_comp);
        jac_to_compressed_with_zinv(Q, &z_inv[2], q_comp);
    }

    // Derive nonce using pre-compressed P as pubkey, Q as msg
    Scalar k;
    ct_zk_derive_nonce(secret, p_comp, q_comp, aux, &k);
    if (secp256k1::cuda::scalar_is_zero(&k)) return false;

    // R1 = k * G (CT: precomputed generator table -- 41% faster than ct_scalar_mul)
    // R2 = k * H (CT: arbitrary base)
    JacobianPoint R1, R2;
    ct_generator_mul(&k, &R1);
    ct_scalar_mul(H, &k, &R2);

    // Batch invert R1, R2 Z coords (1 field_inv + 3 field_mul vs 2 field_inv)
    uint8_t r1_comp[33], r2_comp[33];
    {
        FieldElement rz_in[2] = { R1.z, R2.z };
        FieldElement rz_inv[2];
        ct_batch_field_inv(rz_in, rz_inv, 2);
        jac_to_compressed_with_zinv(&R1, &rz_inv[0], r1_comp);
        jac_to_compressed_with_zinv(&R2, &rz_inv[1], r2_comp);
    }

    // e = H("ZK/dleq" || G || H || P || Q || R1 || R2)
    uint8_t buf[33 * 6];
    for (int i = 0; i < 33; ++i) {
        buf[i]       = G_COMPRESSED[i];
        buf[33 + i]  = h_comp[i];
        buf[66 + i]  = p_comp[i];
        buf[99 + i]  = q_comp[i];
        buf[132 + i] = r1_comp[i];
        buf[165 + i] = r2_comp[i];
    }

    uint8_t e_hash[32];
    zk_tagged_hash_midstate(&ZK_DLEQ_MIDSTATE, buf, sizeof(buf), e_hash);
    secp256k1::cuda::scalar_from_bytes(e_hash, &proof->e);

    // s = k + e * secret (CT scalar arithmetic)
    Scalar e_sec;
    scalar_mul(&proof->e, secret, &e_sec);
    scalar_add(&k, &e_sec, &proof->s);

    return true;
}

// Batch kernel: DLEQ prove with standard generator G
__global__ void ct_dleq_prove_generator_batch_kernel(
    const Scalar* __restrict__ secrets,
    const JacobianPoint* __restrict__ H_pts,
    const JacobianPoint* __restrict__ P_pts,
    const JacobianPoint* __restrict__ Q_pts,
    const uint8_t* __restrict__ aux_rands,
    DLEQProofGPU* __restrict__ proofs,
    bool* __restrict__ results,
    uint32_t count)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = ct_dleq_prove_generator_device(
        &secrets[idx], &H_pts[idx],
        &P_pts[idx], &Q_pts[idx],
        &aux_rands[idx * 32], &proofs[idx]);
}

// ============================================================================
// CT Bulletproof Range Prove (constant-time for secret value/blinding)
// ============================================================================
// Prerequisites: bulletproof_init_kernel() must have been called.

static __device__ __noinline__ bool ct_range_prove_device(
    uint64_t value,
    const Scalar* blinding,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    const uint8_t aux[32],
    RangeProofGPU* proof)
{
    // Bit decomposition: a_L[i] = (value >> i) & 1, a_R[i] = a_L[i] - 1
    Scalar a_L[64], a_R[64];
    Scalar ONE_S;
    ONE_S.limbs[0] = 1; ONE_S.limbs[1] = 0; ONE_S.limbs[2] = 0; ONE_S.limbs[3] = 0;
    Scalar ZERO_S;
    ZERO_S.limbs[0] = 0; ZERO_S.limbs[1] = 0; ZERO_S.limbs[2] = 0; ZERO_S.limbs[3] = 0;

    for (int i = 0; i < 64; ++i) {
        uint64_t bit = (value >> i) & 1;
        a_L[i] = bit ? ONE_S : ZERO_S;
        scalar_sub(&a_L[i], &ONE_S, &a_R[i]);
    }

    // Derive alpha deterministically: H(blinding || commitment || aux)
    uint8_t blind_bytes[32];
    scalar_to_bytes(blinding, blind_bytes);
    uint8_t alpha_buf[32 + 33 + 32];
    for (int i = 0; i < 32; ++i) alpha_buf[i] = blind_bytes[i];
    uint8_t v_comp[33];
    affine_to_compressed(&commitment->x, &commitment->y, v_comp);
    for (int i = 0; i < 33; ++i) alpha_buf[32 + i] = v_comp[i];
    for (int i = 0; i < 32; ++i) alpha_buf[65 + i] = aux[i];

    uint8_t alpha_hash[32];
    sha256_hash(alpha_buf, sizeof(alpha_buf), alpha_hash);
    Scalar alpha;
    scalar_from_bytes(alpha_hash, &alpha);

    // Derive rho = H(alpha)
    uint8_t rho_hash[32];
    sha256_hash(alpha_hash, 32, rho_hash);
    Scalar rho;
    scalar_from_bytes(rho_hash, &rho);

    // Random blinding vectors s_L, s_R
    Scalar s_L[64], s_R[64];
    for (int i = 0; i < 64; ++i) {
        uint8_t buf[34];
        for (int j = 0; j < 32; ++j) buf[j] = alpha_hash[j];
        buf[32] = (uint8_t)i;
        buf[33] = 'L';
        uint8_t h[32];
        sha256_hash(buf, 34, h);
        scalar_from_bytes(h, &s_L[i]);
        buf[33] = 'R';
        sha256_hash(buf, 34, h);
        scalar_from_bytes(h, &s_R[i]);
    }

    // A = alpha*G + sum(a_L[i]*G_i + a_R[i]*H_i)
    JacobianPoint A_pt;
    ct_generator_mul(&alpha, &A_pt);
    for (int i = 0; i < 64; ++i) {
        if (a_L[i].limbs[0] != 0) {
            JacobianPoint Gi_jac, aGi;
            Gi_jac.x = g_bulletproof_G[i].x; Gi_jac.y = g_bulletproof_G[i].y;
            Gi_jac.z = FIELD_ONE; Gi_jac.infinity = false;
            ct_scalar_mul(&Gi_jac, &a_L[i], &aGi);
            jacobian_add(&A_pt, &aGi, &A_pt);
        }
        JacobianPoint Hi_jac, aHi;
        Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
        Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
        ct_scalar_mul(&Hi_jac, &a_R[i], &aHi);
        jacobian_add(&A_pt, &aHi, &A_pt);
    }

    // Convert A to affine
    FieldElement zi, zi2, zi3;
    field_inv(&A_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
    field_mul(&A_pt.x, &zi2, &proof->A.x);
    field_mul(&A_pt.y, &zi3, &proof->A.y);

    // S = rho*G + sum(s_L[i]*G_i + s_R[i]*H_i)
    JacobianPoint S_pt;
    ct_generator_mul(&rho, &S_pt);
    for (int i = 0; i < 64; ++i) {
        JacobianPoint Gi_jac, sGi, Hi_jac, sHi;
        Gi_jac.x = g_bulletproof_G[i].x; Gi_jac.y = g_bulletproof_G[i].y;
        Gi_jac.z = FIELD_ONE; Gi_jac.infinity = false;
        ct_scalar_mul(&Gi_jac, &s_L[i], &sGi);
        jacobian_add(&S_pt, &sGi, &S_pt);

        Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
        Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
        ct_scalar_mul(&Hi_jac, &s_R[i], &sHi);
        jacobian_add(&S_pt, &sHi, &S_pt);
    }

    // Convert S to affine
    field_inv(&S_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
    field_mul(&S_pt.x, &zi2, &proof->S.x);
    field_mul(&S_pt.y, &zi3, &proof->S.y);

    // ---- Fiat-Shamir: y, z ----
    uint8_t a_comp[33], s_comp[33];
    affine_to_compressed(&proof->A.x, &proof->A.y, a_comp);
    affine_to_compressed(&proof->S.x, &proof->S.y, s_comp);

    uint8_t fs_buf[33 + 33 + 33];
    for (int i = 0; i < 33; ++i) {
        fs_buf[i]      = a_comp[i];
        fs_buf[33 + i] = s_comp[i];
        fs_buf[66 + i] = v_comp[i];
    }
    uint8_t y_hash[32], z_hash[32];
    zk_tagged_hash_midstate(&ZK_BULLETPROOF_Y_MIDSTATE, fs_buf, sizeof(fs_buf), y_hash);
    zk_tagged_hash_midstate(&ZK_BULLETPROOF_Z_MIDSTATE, fs_buf, sizeof(fs_buf), z_hash);
    Scalar y, z;
    scalar_from_bytes(y_hash, &y);
    scalar_from_bytes(z_hash, &z);

    // Compute y powers, z^2, 2^i
    Scalar y_powers[64];
    y_powers[0] = ONE_S;
    for (int i = 1; i < 64; ++i) scalar_mul_mod_n(&y_powers[i-1], &y, &y_powers[i]);

    Scalar z2;
    scalar_mul_mod_n(&z, &z, &z2);

    Scalar two_powers[64];
    two_powers[0] = ONE_S;
    for (int i = 1; i < 64; ++i) scalar_add(&two_powers[i-1], &two_powers[i-1], &two_powers[i]);

    // t1, t2 coefficients
    Scalar t1 = ZERO_S, t2 = ZERO_S;
    for (int i = 0; i < 64; ++i) {
        Scalar l0_i, r0_i, l1_i, r1_i;
        scalar_sub(&a_L[i], &z, &l0_i);
        Scalar aR_plus_z;
        scalar_add(&a_R[i], &z, &aR_plus_z);
        Scalar yi_aRz;
        scalar_mul_mod_n(&y_powers[i], &aR_plus_z, &yi_aRz);
        Scalar z2_2i;
        scalar_mul_mod_n(&z2, &two_powers[i], &z2_2i);
        scalar_add(&yi_aRz, &z2_2i, &r0_i);
        l1_i = s_L[i];
        scalar_mul_mod_n(&y_powers[i], &s_R[i], &r1_i);

        Scalar cross1, cross2;
        scalar_mul_mod_n(&l0_i, &r1_i, &cross1);
        scalar_mul_mod_n(&l1_i, &r0_i, &cross2);
        Scalar sum12;
        scalar_add(&cross1, &cross2, &sum12);
        scalar_add(&t1, &sum12, &t1);

        Scalar t2_i;
        scalar_mul_mod_n(&l1_i, &r1_i, &t2_i);
        scalar_add(&t2, &t2_i, &t2);
    }

    // Derive tau1, tau2
    uint8_t tau1_hash[32], tau2_hash[32];
    sha256_hash(rho_hash, 32, tau1_hash);
    sha256_hash(tau1_hash, 32, tau2_hash);
    Scalar tau1, tau2;
    scalar_from_bytes(tau1_hash, &tau1);
    scalar_from_bytes(tau2_hash, &tau2);

    // T1 = t1*H + tau1*G, T2 = t2*H + tau2*G
    JacobianPoint H_jac;
    H_jac.x = H_gen->x; H_jac.y = H_gen->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;

    JacobianPoint t1H, tau1G, T1_pt;
    ct_scalar_mul(&H_jac, &t1, &t1H);
    ct_generator_mul(&tau1, &tau1G);
    jacobian_add(&t1H, &tau1G, &T1_pt);
    field_inv(&T1_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
    field_mul(&T1_pt.x, &zi2, &proof->T1.x);
    field_mul(&T1_pt.y, &zi3, &proof->T1.y);

    JacobianPoint t2H, tau2G, T2_pt;
    ct_scalar_mul(&H_jac, &t2, &t2H);
    ct_generator_mul(&tau2, &tau2G);
    jacobian_add(&t2H, &tau2G, &T2_pt);
    field_inv(&T2_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
    field_mul(&T2_pt.x, &zi2, &proof->T2.x);
    field_mul(&T2_pt.y, &zi3, &proof->T2.y);

    // ---- Fiat-Shamir: x ----
    uint8_t t1_comp[33], t2_comp[33];
    affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
    affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);
    uint8_t x_buf[33 + 33 + 32 + 32];
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
    scalar_to_bytes(&y, x_buf + 66);
    scalar_to_bytes(&z, x_buf + 98);
    uint8_t x_hash[32];
    zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);
    Scalar xx;
    scalar_from_bytes(x_hash, &xx);

    // Evaluate l(x), r(x), t_hat
    Scalar l_x[64], r_x[64];
    Scalar t_hat = ZERO_S;
    for (int i = 0; i < 64; ++i) {
        Scalar aL_z, sL_x;
        scalar_sub(&a_L[i], &z, &aL_z);
        scalar_mul_mod_n(&s_L[i], &xx, &sL_x);
        scalar_add(&aL_z, &sL_x, &l_x[i]);

        Scalar aR_z, sR_x, aR_z_sR_x, yi_term, z2_2i;
        scalar_add(&a_R[i], &z, &aR_z);
        scalar_mul_mod_n(&s_R[i], &xx, &sR_x);
        scalar_add(&aR_z, &sR_x, &aR_z_sR_x);
        scalar_mul_mod_n(&y_powers[i], &aR_z_sR_x, &yi_term);
        scalar_mul_mod_n(&z2, &two_powers[i], &z2_2i);
        scalar_add(&yi_term, &z2_2i, &r_x[i]);

        Scalar prod;
        scalar_mul_mod_n(&l_x[i], &r_x[i], &prod);
        scalar_add(&t_hat, &prod, &t_hat);
    }
    proof->t_hat = t_hat;

    // tau_x = tau2*x^2 + tau1*x + z^2*blinding
    Scalar xx2;
    scalar_mul_mod_n(&xx, &xx, &xx2);
    Scalar tau2_x2, tau1_x, z2_blind;
    scalar_mul_mod_n(&tau2, &xx2, &tau2_x2);
    scalar_mul_mod_n(&tau1, &xx, &tau1_x);
    scalar_mul_mod_n(&z2, blinding, &z2_blind);
    Scalar tau_x;
    scalar_add(&tau2_x2, &tau1_x, &tau_x);
    scalar_add(&tau_x, &z2_blind, &tau_x);
    proof->tau_x = tau_x;

    // mu = alpha + rho*x
    Scalar rho_x;
    scalar_mul_mod_n(&rho, &xx, &rho_x);
    scalar_add(&alpha, &rho_x, &proof->mu);

    // ---- Inner Product Argument ----
    Scalar a_vec[64], b_vec[64];
    for (int i = 0; i < 64; ++i) { a_vec[i] = l_x[i]; b_vec[i] = r_x[i]; }

    // Modified generator points: H'_i = y^{-i} * H_i
    Scalar y_inv;
    scalar_inverse(&y, &y_inv);
    Scalar y_inv_pow = ONE_S;

    // Precompute H' as Jacobian (y_inv^i * H_i)
    JacobianPoint G_vec[64], H_vec[64];
    for (int i = 0; i < 64; ++i) {
        G_vec[i].x = g_bulletproof_G[i].x; G_vec[i].y = g_bulletproof_G[i].y;
        G_vec[i].z = FIELD_ONE; G_vec[i].infinity = false;

        JacobianPoint Hi_jac;
        Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
        Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
        scalar_mul(&Hi_jac, &y_inv_pow, &H_vec[i]);
        scalar_mul_mod_n(&y_inv_pow, &y_inv, &y_inv_pow);
    }

    int n = 64;
    for (int round = 0; round < 6; ++round) {
        n /= 2;

        Scalar c_L = ZERO_S, c_R = ZERO_S;
        JacobianPoint L_pt, R_pt;
        L_pt.infinity = true; L_pt.z = FIELD_ONE;
        R_pt.infinity = true; R_pt.z = FIELD_ONE;

        for (int i = 0; i < n; ++i) {
            // L = sum(a_lo[i]*G_hi[i]) + sum(b_hi[i]*H'_lo[i]) + c_L*U
            JacobianPoint aG, bH;
            scalar_mul(&G_vec[n + i], &a_vec[i], &aG);
            jacobian_add(&L_pt, &aG, &L_pt);
            scalar_mul(&H_vec[i], &b_vec[n + i], &bH);
            jacobian_add(&L_pt, &bH, &L_pt);
            Scalar prod;
            scalar_mul_mod_n(&a_vec[i], &b_vec[n + i], &prod);
            scalar_add(&c_L, &prod, &c_L);

            // R = sum(a_hi[i]*G_lo[i]) + sum(b_lo[i]*H'_hi[i]) + c_R*U
            scalar_mul(&G_vec[i], &a_vec[n + i], &aG);
            jacobian_add(&R_pt, &aG, &R_pt);
            scalar_mul(&H_vec[n + i], &b_vec[i], &bH);
            jacobian_add(&R_pt, &bH, &R_pt);
            scalar_mul_mod_n(&a_vec[n + i], &b_vec[i], &prod);
            scalar_add(&c_R, &prod, &c_R);
        }
        // Add c_L*U and c_R*U (U = H_ped)
        JacobianPoint cU;
        scalar_mul(&H_jac, &c_L, &cU);
        jacobian_add(&L_pt, &cU, &L_pt);
        scalar_mul(&H_jac, &c_R, &cU);
        jacobian_add(&R_pt, &cU, &R_pt);

        // Convert L, R to affine for proof
        field_inv(&L_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&L_pt.x, &zi2, &proof->L[round].x);
        field_mul(&L_pt.y, &zi3, &proof->L[round].y);

        field_inv(&R_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&R_pt.x, &zi2, &proof->R[round].x);
        field_mul(&R_pt.y, &zi3, &proof->R[round].y);

        // Fiat-Shamir: x_round = H("Bulletproof/ip" || L || R)
        uint8_t l_comp[33], r_comp[33];
        affine_to_compressed(&proof->L[round].x, &proof->L[round].y, l_comp);
        affine_to_compressed(&proof->R[round].x, &proof->R[round].y, r_comp);
        uint8_t ip_buf[33 + 33];
        for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
        uint8_t xr_hash[32];
        zk_tagged_hash_midstate(&g_bp_ip_midstate, ip_buf, sizeof(ip_buf), xr_hash);
        Scalar x_r;
        scalar_from_bytes(xr_hash, &x_r);
        Scalar x_r_inv;
        scalar_inverse(&x_r, &x_r_inv);

        // Fold vectors
        for (int i = 0; i < n; ++i) {
            Scalar a_lo_x, a_hi_xi;
            scalar_mul_mod_n(&a_vec[i], &x_r, &a_lo_x);
            scalar_mul_mod_n(&a_vec[n + i], &x_r_inv, &a_hi_xi);
            scalar_add(&a_lo_x, &a_hi_xi, &a_vec[i]);

            Scalar b_lo_xi, b_hi_x;
            scalar_mul_mod_n(&b_vec[i], &x_r_inv, &b_lo_xi);
            scalar_mul_mod_n(&b_vec[n + i], &x_r, &b_hi_x);
            scalar_add(&b_lo_xi, &b_hi_x, &b_vec[i]);
        }
        for (int i = 0; i < n; ++i) {
            JacobianPoint G_lo_xi, G_hi_x;
            scalar_mul(&G_vec[i], &x_r_inv, &G_lo_xi);
            scalar_mul(&G_vec[n + i], &x_r, &G_hi_x);
            jacobian_add(&G_lo_xi, &G_hi_x, &G_vec[i]);

            JacobianPoint H_lo_x, H_hi_xi;
            scalar_mul(&H_vec[i], &x_r, &H_lo_x);
            scalar_mul(&H_vec[n + i], &x_r_inv, &H_hi_xi);
            jacobian_add(&H_lo_x, &H_hi_xi, &H_vec[i]);
        }
    }

    proof->a = a_vec[0];
    proof->b = b_vec[0];
    return true;
}

// ============================================================================
// Warp-Cooperative Bulletproof Range Prover
// ============================================================================
// 32 threads (1 warp) cooperate on a single proof.
// Parallelizes the A and S vector commitment MSMs (128 ct_scalar_mul each).
// Lane 0 handles all Fiat-Shamir hashes and IPA rounds (data-dependent).
//
// Shared memory layout per warp:
//   a_L[64], a_R[64]  -- bit decomposition vectors
//   s_L[64], s_R[64]  -- blinding vectors
//   A_pt, S_pt         -- accumulated Jacobian results from parallel MSM
//   alpha, rho         -- derived blindings for IPA
// ============================================================================

struct BPProveWarpShared {
    Scalar a_L[BP_BITS];          // 2048 bytes
    Scalar a_R[BP_BITS];          // 2048
    Scalar s_L[BP_BITS];          // 2048
    Scalar s_R[BP_BITS];          // 2048
    Scalar alpha;
    Scalar rho;
    uint8_t alpha_hash[32];       // for s_L/s_R derivation reproducibility
    uint8_t v_comp[33];           // compressed commitment
    JacobianPoint accum;          // result of warp MSM reduction
    bool ready;                   // synchronization flag
};

// Warp tree reduction for JacobianPoint (same pattern as verifier)
__device__ inline void warp_reduce_jacobian(JacobianPoint* local_acc) {
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        JacobianPoint partner;
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc->x.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc->x.limbs[w] >> 32), offset);
            partner.x.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc->y.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc->y.limbs[w] >> 32), offset);
            partner.y.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc->z.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc->z.limbs[w] >> 32), offset);
            partner.z.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        uint32_t inf_raw = __shfl_down_sync(FULL_MASK, (uint32_t)local_acc->infinity, offset);
        partner.infinity = (bool)inf_raw;

        int lane = threadIdx.x & 31;
        if (lane + offset < 32) {
            jacobian_add(local_acc, &partner, local_acc);
        }
    }
}

static __device__ __noinline__ bool ct_range_prove_warp_device(
    uint64_t value,
    const Scalar* blinding,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    const uint8_t aux[32],
    RangeProofGPU* proof,
    BPProveWarpShared* smem)
{
    const int lane = threadIdx.x & 31;
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

    Scalar ONE_S;
    ONE_S.limbs[0] = 1; ONE_S.limbs[1] = 0; ONE_S.limbs[2] = 0; ONE_S.limbs[3] = 0;
    Scalar ZERO_S;
    ZERO_S.limbs[0] = 0; ZERO_S.limbs[1] = 0; ZERO_S.limbs[2] = 0; ZERO_S.limbs[3] = 0;

    // ====================================================================
    // Phase 1: Lane 0 derives all scalar data and stores in shared mem
    // ====================================================================
    if (lane == 0) {
        // Bit decomposition
        for (int i = 0; i < 64; ++i) {
            uint64_t bit = (value >> i) & 1;
            smem->a_L[i] = bit ? ONE_S : ZERO_S;
            scalar_sub(&smem->a_L[i], &ONE_S, &smem->a_R[i]);
        }

        // Derive alpha
        uint8_t blind_bytes[32];
        scalar_to_bytes(blinding, blind_bytes);
        uint8_t alpha_buf[32 + 33 + 32];
        for (int i = 0; i < 32; ++i) alpha_buf[i] = blind_bytes[i];
        affine_to_compressed(&commitment->x, &commitment->y, smem->v_comp);
        for (int i = 0; i < 33; ++i) alpha_buf[32 + i] = smem->v_comp[i];
        for (int i = 0; i < 32; ++i) alpha_buf[65 + i] = aux[i];
        sha256_hash(alpha_buf, sizeof(alpha_buf), smem->alpha_hash);
        scalar_from_bytes(smem->alpha_hash, &smem->alpha);

        // Derive rho
        uint8_t rho_hash[32];
        sha256_hash(smem->alpha_hash, 32, rho_hash);
        scalar_from_bytes(rho_hash, &smem->rho);

        // s_L, s_R blinding vectors
        for (int i = 0; i < 64; ++i) {
            uint8_t buf[34];
            for (int j = 0; j < 32; ++j) buf[j] = smem->alpha_hash[j];
            buf[32] = (uint8_t)i;
            buf[33] = 'L';
            uint8_t h[32];
            sha256_hash(buf, 34, h);
            scalar_from_bytes(h, &smem->s_L[i]);
            buf[33] = 'R';
            sha256_hash(buf, 34, h);
            scalar_from_bytes(h, &smem->s_R[i]);
        }

        smem->ready = true;
    }
    __syncwarp(FULL_MASK);

    // ====================================================================
    // Phase 2: All 32 lanes compute A commitment MSM in parallel
    // ====================================================================
    // A = alpha*G + sum(a_L[i]*G_i + a_R[i]*H_i)
    // Each lane handles 2 of the 64 generators

    JacobianPoint local_acc;
    local_acc.infinity = true;
    local_acc.z = FIELD_ONE;

    {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            // a_L[i] * G_i  (only if bit is set, a_L = 0 or 1)
            if (smem->a_L[i].limbs[0] != 0) {
                JacobianPoint Gi_jac, aGi;
                Gi_jac.x = g_bulletproof_G[i].x; Gi_jac.y = g_bulletproof_G[i].y;
                Gi_jac.z = FIELD_ONE; Gi_jac.infinity = false;
                ct_scalar_mul(&Gi_jac, &smem->a_L[i], &aGi);
                jacobian_add(&local_acc, &aGi, &local_acc);
            }

            // a_R[i] * H_i  (a_R = bit - 1, always -1 or 0)
            JacobianPoint Hi_jac, aHi;
            Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
            Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
            ct_scalar_mul(&Hi_jac, &smem->a_R[i], &aHi);
            jacobian_add(&local_acc, &aHi, &local_acc);
        }
    }

    // Warp tree reduction
    warp_reduce_jacobian(&local_acc);

    // Lane 0: add alpha*G and convert A to affine
    if (lane == 0) {
        JacobianPoint alphaG;
        ct_generator_mul(&smem->alpha, &alphaG);
        jacobian_add(&local_acc, &alphaG, &local_acc);

        FieldElement zi, zi2, zi3;
        field_inv(&local_acc.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&local_acc.x, &zi2, &proof->A.x);
        field_mul(&local_acc.y, &zi3, &proof->A.y);
    }
    __syncwarp(FULL_MASK);

    // ====================================================================
    // Phase 3: All 32 lanes compute S commitment MSM in parallel
    // ====================================================================
    // S = rho*G + sum(s_L[i]*G_i + s_R[i]*H_i)

    local_acc.infinity = true;
    local_acc.z = FIELD_ONE;

    {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            JacobianPoint Gi_jac, sGi;
            Gi_jac.x = g_bulletproof_G[i].x; Gi_jac.y = g_bulletproof_G[i].y;
            Gi_jac.z = FIELD_ONE; Gi_jac.infinity = false;
            ct_scalar_mul(&Gi_jac, &smem->s_L[i], &sGi);
            jacobian_add(&local_acc, &sGi, &local_acc);

            JacobianPoint Hi_jac, sHi;
            Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
            Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
            ct_scalar_mul(&Hi_jac, &smem->s_R[i], &sHi);
            jacobian_add(&local_acc, &sHi, &local_acc);
        }
    }

    // Warp tree reduction
    warp_reduce_jacobian(&local_acc);

    // Lane 0: add rho*G and convert S to affine
    if (lane == 0) {
        JacobianPoint rhoG;
        ct_generator_mul(&smem->rho, &rhoG);
        jacobian_add(&local_acc, &rhoG, &local_acc);

        FieldElement zi, zi2, zi3;
        field_inv(&local_acc.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&local_acc.x, &zi2, &proof->S.x);
        field_mul(&local_acc.y, &zi3, &proof->S.y);
    }
    __syncwarp(FULL_MASK);

    // ====================================================================
    // Phase 4: Lane 0 handles Fiat-Shamir, t1/t2, T1/T2, IPA (serial)
    // ====================================================================
    if (lane == 0) {
        // ---- Fiat-Shamir: y, z ----
        uint8_t a_comp[33], s_comp[33];
        affine_to_compressed(&proof->A.x, &proof->A.y, a_comp);
        affine_to_compressed(&proof->S.x, &proof->S.y, s_comp);

        uint8_t fs_buf[33 + 33 + 33];
        for (int i = 0; i < 33; ++i) {
            fs_buf[i]      = a_comp[i];
            fs_buf[33 + i] = s_comp[i];
            fs_buf[66 + i] = smem->v_comp[i];
        }
        uint8_t y_hash[32], z_hash[32];
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_Y_MIDSTATE, fs_buf, sizeof(fs_buf), y_hash);
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_Z_MIDSTATE, fs_buf, sizeof(fs_buf), z_hash);
        Scalar y, z;
        scalar_from_bytes(y_hash, &y);
        scalar_from_bytes(z_hash, &z);

        Scalar y_powers[64];
        y_powers[0] = ONE_S;
        for (int i = 1; i < 64; ++i) scalar_mul_mod_n(&y_powers[i-1], &y, &y_powers[i]);

        Scalar z2;
        scalar_mul_mod_n(&z, &z, &z2);

        Scalar two_powers[64];
        two_powers[0] = ONE_S;
        for (int i = 1; i < 64; ++i) scalar_add(&two_powers[i-1], &two_powers[i-1], &two_powers[i]);

        // t1, t2
        Scalar t1 = ZERO_S, t2 = ZERO_S;
        for (int i = 0; i < 64; ++i) {
            Scalar l0_i, r0_i, l1_i, r1_i;
            scalar_sub(&smem->a_L[i], &z, &l0_i);
            Scalar aR_plus_z;
            scalar_add(&smem->a_R[i], &z, &aR_plus_z);
            Scalar yi_aRz;
            scalar_mul_mod_n(&y_powers[i], &aR_plus_z, &yi_aRz);
            Scalar z2_2i;
            scalar_mul_mod_n(&z2, &two_powers[i], &z2_2i);
            scalar_add(&yi_aRz, &z2_2i, &r0_i);
            l1_i = smem->s_L[i];
            scalar_mul_mod_n(&y_powers[i], &smem->s_R[i], &r1_i);

            Scalar cross1, cross2;
            scalar_mul_mod_n(&l0_i, &r1_i, &cross1);
            scalar_mul_mod_n(&l1_i, &r0_i, &cross2);
            Scalar sum12;
            scalar_add(&cross1, &cross2, &sum12);
            scalar_add(&t1, &sum12, &t1);

            Scalar t2_i;
            scalar_mul_mod_n(&l1_i, &r1_i, &t2_i);
            scalar_add(&t2, &t2_i, &t2);
        }

        // Derive tau1, tau2 from rho chain
        uint8_t rho_hash[32];
        sha256_hash(smem->alpha_hash, 32, rho_hash);
        uint8_t tau1_hash[32], tau2_hash[32];
        sha256_hash(rho_hash, 32, tau1_hash);
        sha256_hash(tau1_hash, 32, tau2_hash);
        Scalar tau1, tau2;
        scalar_from_bytes(tau1_hash, &tau1);
        scalar_from_bytes(tau2_hash, &tau2);

        // T1 = t1*H + tau1*G, T2 = t2*H + tau2*G
        JacobianPoint H_jac;
        H_jac.x = H_gen->x; H_jac.y = H_gen->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;

        JacobianPoint t1H, tau1G, T1_pt;
        ct_scalar_mul(&H_jac, &t1, &t1H);
        ct_generator_mul(&tau1, &tau1G);
        jacobian_add(&t1H, &tau1G, &T1_pt);
        FieldElement zi, zi2, zi3;
        field_inv(&T1_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&T1_pt.x, &zi2, &proof->T1.x);
        field_mul(&T1_pt.y, &zi3, &proof->T1.y);

        JacobianPoint t2H, tau2G, T2_pt;
        ct_scalar_mul(&H_jac, &t2, &t2H);
        ct_generator_mul(&tau2, &tau2G);
        jacobian_add(&t2H, &tau2G, &T2_pt);
        field_inv(&T2_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&T2_pt.x, &zi2, &proof->T2.x);
        field_mul(&T2_pt.y, &zi3, &proof->T2.y);

        // ---- Fiat-Shamir: x ----
        uint8_t t1_comp[33], t2_comp[33];
        affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
        affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);
        uint8_t x_buf[33 + 33 + 32 + 32];
        for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
        scalar_to_bytes(&y, x_buf + 66);
        scalar_to_bytes(&z, x_buf + 98);
        uint8_t x_hash[32];
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);
        Scalar xx;
        scalar_from_bytes(x_hash, &xx);

        // l(x), r(x), t_hat
        Scalar l_x[64], r_x[64];
        Scalar t_hat = ZERO_S;
        for (int i = 0; i < 64; ++i) {
            Scalar aL_z, sL_x;
            scalar_sub(&smem->a_L[i], &z, &aL_z);
            scalar_mul_mod_n(&smem->s_L[i], &xx, &sL_x);
            scalar_add(&aL_z, &sL_x, &l_x[i]);

            Scalar aR_z, sR_x, aR_z_sR_x, yi_term, z2_2i;
            scalar_add(&smem->a_R[i], &z, &aR_z);
            scalar_mul_mod_n(&smem->s_R[i], &xx, &sR_x);
            scalar_add(&aR_z, &sR_x, &aR_z_sR_x);
            scalar_mul_mod_n(&y_powers[i], &aR_z_sR_x, &yi_term);
            scalar_mul_mod_n(&z2, &two_powers[i], &z2_2i);
            scalar_add(&yi_term, &z2_2i, &r_x[i]);

            Scalar prod;
            scalar_mul_mod_n(&l_x[i], &r_x[i], &prod);
            scalar_add(&t_hat, &prod, &t_hat);
        }
        proof->t_hat = t_hat;

        // tau_x, mu
        Scalar xx2;
        scalar_mul_mod_n(&xx, &xx, &xx2);
        Scalar tau2_x2, tau1_x, z2_blind;
        scalar_mul_mod_n(&tau2, &xx2, &tau2_x2);
        scalar_mul_mod_n(&tau1, &xx, &tau1_x);
        scalar_mul_mod_n(&z2, blinding, &z2_blind);
        Scalar tau_x;
        scalar_add(&tau2_x2, &tau1_x, &tau_x);
        scalar_add(&tau_x, &z2_blind, &tau_x);
        proof->tau_x = tau_x;

        Scalar rho_s;
        scalar_from_bytes(rho_hash, &rho_s);
        Scalar rho_x;
        scalar_mul_mod_n(&rho_s, &xx, &rho_x);
        scalar_add(&smem->alpha, &rho_x, &proof->mu);

        // ---- Inner Product Argument (serial) ----
        Scalar a_vec[64], b_vec[64];
        for (int i = 0; i < 64; ++i) { a_vec[i] = l_x[i]; b_vec[i] = r_x[i]; }

        Scalar y_inv;
        scalar_inverse(&y, &y_inv);
        Scalar y_inv_pow = ONE_S;

        JacobianPoint G_vec[64], H_vec[64];
        for (int i = 0; i < 64; ++i) {
            G_vec[i].x = g_bulletproof_G[i].x; G_vec[i].y = g_bulletproof_G[i].y;
            G_vec[i].z = FIELD_ONE; G_vec[i].infinity = false;

            JacobianPoint Hi_jac;
            Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
            Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
            scalar_mul(&Hi_jac, &y_inv_pow, &H_vec[i]);
            scalar_mul_mod_n(&y_inv_pow, &y_inv, &y_inv_pow);
        }

        int n = 64;
        for (int round = 0; round < 6; ++round) {
            n /= 2;

            Scalar c_L = ZERO_S, c_R = ZERO_S;
            JacobianPoint L_pt, R_pt;
            L_pt.infinity = true; L_pt.z = FIELD_ONE;
            R_pt.infinity = true; R_pt.z = FIELD_ONE;

            for (int i = 0; i < n; ++i) {
                JacobianPoint aG, bH;
                scalar_mul(&G_vec[n + i], &a_vec[i], &aG);
                jacobian_add(&L_pt, &aG, &L_pt);
                scalar_mul(&H_vec[i], &b_vec[n + i], &bH);
                jacobian_add(&L_pt, &bH, &L_pt);
                Scalar prod;
                scalar_mul_mod_n(&a_vec[i], &b_vec[n + i], &prod);
                scalar_add(&c_L, &prod, &c_L);

                scalar_mul(&G_vec[i], &a_vec[n + i], &aG);
                jacobian_add(&R_pt, &aG, &R_pt);
                scalar_mul(&H_vec[n + i], &b_vec[i], &bH);
                jacobian_add(&R_pt, &bH, &R_pt);
                scalar_mul_mod_n(&a_vec[n + i], &b_vec[i], &prod);
                scalar_add(&c_R, &prod, &c_R);
            }
            JacobianPoint cU;
            scalar_mul(&H_jac, &c_L, &cU);
            jacobian_add(&L_pt, &cU, &L_pt);
            scalar_mul(&H_jac, &c_R, &cU);
            jacobian_add(&R_pt, &cU, &R_pt);

            field_inv(&L_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
            field_mul(&L_pt.x, &zi2, &proof->L[round].x);
            field_mul(&L_pt.y, &zi3, &proof->L[round].y);

            field_inv(&R_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
            field_mul(&R_pt.x, &zi2, &proof->R[round].x);
            field_mul(&R_pt.y, &zi3, &proof->R[round].y);

            uint8_t l_comp[33], r_comp[33];
            affine_to_compressed(&proof->L[round].x, &proof->L[round].y, l_comp);
            affine_to_compressed(&proof->R[round].x, &proof->R[round].y, r_comp);
            uint8_t ip_buf[33 + 33];
            for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
            uint8_t xr_hash[32];
            zk_tagged_hash_midstate(&g_bp_ip_midstate, ip_buf, sizeof(ip_buf), xr_hash);
            Scalar x_r;
            scalar_from_bytes(xr_hash, &x_r);
            Scalar x_r_inv;
            scalar_inverse(&x_r, &x_r_inv);

            for (int i = 0; i < n; ++i) {
                Scalar a_lo_x, a_hi_xi;
                scalar_mul_mod_n(&a_vec[i], &x_r, &a_lo_x);
                scalar_mul_mod_n(&a_vec[n + i], &x_r_inv, &a_hi_xi);
                scalar_add(&a_lo_x, &a_hi_xi, &a_vec[i]);

                Scalar b_lo_xi, b_hi_x;
                scalar_mul_mod_n(&b_vec[i], &x_r_inv, &b_lo_xi);
                scalar_mul_mod_n(&b_vec[n + i], &x_r, &b_hi_x);
                scalar_add(&b_lo_xi, &b_hi_x, &b_vec[i]);
            }
            for (int i = 0; i < n; ++i) {
                JacobianPoint G_lo_xi, G_hi_x;
                scalar_mul(&G_vec[i], &x_r_inv, &G_lo_xi);
                scalar_mul(&G_vec[n + i], &x_r, &G_hi_x);
                jacobian_add(&G_lo_xi, &G_hi_x, &G_vec[i]);

                JacobianPoint H_lo_x, H_hi_xi;
                scalar_mul(&H_vec[i], &x_r, &H_lo_x);
                scalar_mul(&H_vec[n + i], &x_r_inv, &H_hi_xi);
                jacobian_add(&H_lo_x, &H_hi_xi, &H_vec[i]);
            }
        }

        proof->a = a_vec[0];
        proof->b = b_vec[0];
    }

    __syncwarp(FULL_MASK);
    return true;
}

// ============================================================================
// Bulletproof Range Prove -- Warp-Cooperative with Positional LUT4
// ============================================================================
// Same as ct_range_prove_warp_device but uses zero-doubling positional LUT4
// for the 128 fixed Bulletproof generators (G[64]+H[64]).
// Changes:
//   Phase 2 (A commitment): LUT4 for a_L[i]*G_i, a_R[i]*H_i
//   Phase 3 (S commitment): LUT4 for s_L[i]*G_i, s_R[i]*H_i
//   Phase 4 IPA round 0:    LUT4 for G_vec ops (generators still original)
//   IPA rounds 1-5:         standard scalar_mul (generators recombined)
// ============================================================================

static __device__ __noinline__ bool ct_range_prove_warp_lut4_device(
    uint64_t value,
    const Scalar* blinding,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    const uint8_t aux[32],
    RangeProofGPU* proof,
    BPProveWarpShared* smem)
{
    const int lane = threadIdx.x & 31;
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

    Scalar ONE_S;
    ONE_S.limbs[0] = 1; ONE_S.limbs[1] = 0; ONE_S.limbs[2] = 0; ONE_S.limbs[3] = 0;
    Scalar ZERO_S;
    ZERO_S.limbs[0] = 0; ZERO_S.limbs[1] = 0; ZERO_S.limbs[2] = 0; ZERO_S.limbs[3] = 0;

    // ====================================================================
    // Phase 1: Lane 0 derives all scalar data (IDENTICAL to original)
    // ====================================================================
    if (lane == 0) {
        for (int i = 0; i < 64; ++i) {
            uint64_t bit = (value >> i) & 1;
            smem->a_L[i] = bit ? ONE_S : ZERO_S;
            scalar_sub(&smem->a_L[i], &ONE_S, &smem->a_R[i]);
        }

        uint8_t blind_bytes[32];
        scalar_to_bytes(blinding, blind_bytes);
        uint8_t alpha_buf[32 + 33 + 32];
        for (int i = 0; i < 32; ++i) alpha_buf[i] = blind_bytes[i];
        affine_to_compressed(&commitment->x, &commitment->y, smem->v_comp);
        for (int i = 0; i < 33; ++i) alpha_buf[32 + i] = smem->v_comp[i];
        for (int i = 0; i < 32; ++i) alpha_buf[65 + i] = aux[i];
        sha256_hash(alpha_buf, sizeof(alpha_buf), smem->alpha_hash);
        scalar_from_bytes(smem->alpha_hash, &smem->alpha);

        uint8_t rho_hash[32];
        sha256_hash(smem->alpha_hash, 32, rho_hash);
        scalar_from_bytes(rho_hash, &smem->rho);

        for (int i = 0; i < 64; ++i) {
            uint8_t buf[34];
            for (int j = 0; j < 32; ++j) buf[j] = smem->alpha_hash[j];
            buf[32] = (uint8_t)i;
            buf[33] = 'L';
            uint8_t h[32];
            sha256_hash(buf, 34, h);
            scalar_from_bytes(h, &smem->s_L[i]);
            buf[33] = 'R';
            sha256_hash(buf, 34, h);
            scalar_from_bytes(h, &smem->s_R[i]);
        }

        smem->ready = true;
    }
    __syncwarp(FULL_MASK);

    // ====================================================================
    // Phase 2: A commitment MSM -- POSITIONAL LUT4 (zero doublings)
    // ====================================================================
    JacobianPoint local_acc;
    local_acc.infinity = true;
    local_acc.z = FIELD_ONE;

    {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            // a_L[i] * G_i via LUT4 (scalar is 0 or 1, LUT4 handles both)
            JacobianPoint aGi;
            scalar_mul_bp_lut4(&g_bp_lut4[i * BP_LUT4_GEN_STRIDE],
                               &smem->a_L[i], &aGi);
            jacobian_add(&local_acc, &aGi, &local_acc);

            // a_R[i] * H_i via LUT4 (scalar is 0 or -1)
            JacobianPoint aHi;
            scalar_mul_bp_lut4(&g_bp_lut4[(64 + i) * BP_LUT4_GEN_STRIDE],
                               &smem->a_R[i], &aHi);
            jacobian_add(&local_acc, &aHi, &local_acc);
        }
    }

    warp_reduce_jacobian(&local_acc);

    if (lane == 0) {
        JacobianPoint alphaG;
        ct_generator_mul(&smem->alpha, &alphaG);
        jacobian_add(&local_acc, &alphaG, &local_acc);

        FieldElement zi, zi2, zi3;
        field_inv(&local_acc.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&local_acc.x, &zi2, &proof->A.x);
        field_mul(&local_acc.y, &zi3, &proof->A.y);
    }
    __syncwarp(FULL_MASK);

    // ====================================================================
    // Phase 3: S commitment MSM -- POSITIONAL LUT4 (zero doublings)
    // ====================================================================
    local_acc.infinity = true;
    local_acc.z = FIELD_ONE;

    {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            // s_L[i] * G_i via LUT4 (full random scalar, big win)
            JacobianPoint sGi;
            scalar_mul_bp_lut4(&g_bp_lut4[i * BP_LUT4_GEN_STRIDE],
                               &smem->s_L[i], &sGi);
            jacobian_add(&local_acc, &sGi, &local_acc);

            // s_R[i] * H_i via LUT4
            JacobianPoint sHi;
            scalar_mul_bp_lut4(&g_bp_lut4[(64 + i) * BP_LUT4_GEN_STRIDE],
                               &smem->s_R[i], &sHi);
            jacobian_add(&local_acc, &sHi, &local_acc);
        }
    }

    warp_reduce_jacobian(&local_acc);

    if (lane == 0) {
        JacobianPoint rhoG;
        ct_generator_mul(&smem->rho, &rhoG);
        jacobian_add(&local_acc, &rhoG, &local_acc);

        FieldElement zi, zi2, zi3;
        field_inv(&local_acc.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&local_acc.x, &zi2, &proof->S.x);
        field_mul(&local_acc.y, &zi3, &proof->S.y);
    }
    __syncwarp(FULL_MASK);

    // ====================================================================
    // Phase 4: Lane 0 handles Fiat-Shamir, T1/T2, IPA (serial)
    // IPA round 0 uses LUT4 for G_vec (still original generators)
    // ====================================================================
    if (lane == 0) {
        // ---- Fiat-Shamir: y, z ----
        uint8_t a_comp[33], s_comp[33];
        affine_to_compressed(&proof->A.x, &proof->A.y, a_comp);
        affine_to_compressed(&proof->S.x, &proof->S.y, s_comp);

        uint8_t fs_buf[33 + 33 + 33];
        for (int i = 0; i < 33; ++i) {
            fs_buf[i]      = a_comp[i];
            fs_buf[33 + i] = s_comp[i];
            fs_buf[66 + i] = smem->v_comp[i];
        }
        uint8_t y_hash[32], z_hash[32];
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_Y_MIDSTATE, fs_buf, sizeof(fs_buf), y_hash);
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_Z_MIDSTATE, fs_buf, sizeof(fs_buf), z_hash);
        Scalar y, z;
        scalar_from_bytes(y_hash, &y);
        scalar_from_bytes(z_hash, &z);

        Scalar y_powers[64];
        y_powers[0] = ONE_S;
        for (int i = 1; i < 64; ++i) scalar_mul_mod_n(&y_powers[i-1], &y, &y_powers[i]);

        Scalar z2;
        scalar_mul_mod_n(&z, &z, &z2);

        Scalar two_powers[64];
        two_powers[0] = ONE_S;
        for (int i = 1; i < 64; ++i) scalar_add(&two_powers[i-1], &two_powers[i-1], &two_powers[i]);

        // t1, t2
        Scalar t1 = ZERO_S, t2 = ZERO_S;
        for (int i = 0; i < 64; ++i) {
            Scalar l0_i, r0_i, l1_i, r1_i;
            scalar_sub(&smem->a_L[i], &z, &l0_i);
            Scalar aR_plus_z;
            scalar_add(&smem->a_R[i], &z, &aR_plus_z);
            Scalar yi_aRz;
            scalar_mul_mod_n(&y_powers[i], &aR_plus_z, &yi_aRz);
            Scalar z2_2i;
            scalar_mul_mod_n(&z2, &two_powers[i], &z2_2i);
            scalar_add(&yi_aRz, &z2_2i, &r0_i);
            l1_i = smem->s_L[i];
            scalar_mul_mod_n(&y_powers[i], &smem->s_R[i], &r1_i);

            Scalar cross1, cross2;
            scalar_mul_mod_n(&l0_i, &r1_i, &cross1);
            scalar_mul_mod_n(&l1_i, &r0_i, &cross2);
            Scalar sum12;
            scalar_add(&cross1, &cross2, &sum12);
            scalar_add(&t1, &sum12, &t1);

            Scalar t2_i;
            scalar_mul_mod_n(&l1_i, &r1_i, &t2_i);
            scalar_add(&t2, &t2_i, &t2);
        }

        // Derive tau1, tau2 from rho chain
        uint8_t rho_hash[32];
        sha256_hash(smem->alpha_hash, 32, rho_hash);
        uint8_t tau1_hash[32], tau2_hash[32];
        sha256_hash(rho_hash, 32, tau1_hash);
        sha256_hash(tau1_hash, 32, tau2_hash);
        Scalar tau1, tau2;
        scalar_from_bytes(tau1_hash, &tau1);
        scalar_from_bytes(tau2_hash, &tau2);

        // T1 = t1*H + tau1*G, T2 = t2*H + tau2*G
        JacobianPoint H_jac;
        H_jac.x = H_gen->x; H_jac.y = H_gen->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;

        JacobianPoint t1H, tau1G, T1_pt;
        ct_scalar_mul(&H_jac, &t1, &t1H);
        ct_generator_mul(&tau1, &tau1G);
        jacobian_add(&t1H, &tau1G, &T1_pt);
        FieldElement zi, zi2, zi3;
        field_inv(&T1_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&T1_pt.x, &zi2, &proof->T1.x);
        field_mul(&T1_pt.y, &zi3, &proof->T1.y);

        JacobianPoint t2H, tau2G, T2_pt;
        ct_scalar_mul(&H_jac, &t2, &t2H);
        ct_generator_mul(&tau2, &tau2G);
        jacobian_add(&t2H, &tau2G, &T2_pt);
        field_inv(&T2_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
        field_mul(&T2_pt.x, &zi2, &proof->T2.x);
        field_mul(&T2_pt.y, &zi3, &proof->T2.y);

        // ---- Fiat-Shamir: x ----
        uint8_t t1_comp[33], t2_comp[33];
        affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
        affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);
        uint8_t x_buf[33 + 33 + 32 + 32];
        for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
        scalar_to_bytes(&y, x_buf + 66);
        scalar_to_bytes(&z, x_buf + 98);
        uint8_t x_hash[32];
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);
        Scalar xx;
        scalar_from_bytes(x_hash, &xx);

        // l(x), r(x), t_hat
        Scalar l_x[64], r_x[64];
        Scalar t_hat = ZERO_S;
        for (int i = 0; i < 64; ++i) {
            Scalar aL_z, sL_x;
            scalar_sub(&smem->a_L[i], &z, &aL_z);
            scalar_mul_mod_n(&smem->s_L[i], &xx, &sL_x);
            scalar_add(&aL_z, &sL_x, &l_x[i]);

            Scalar aR_z, sR_x, aR_z_sR_x, yi_term, z2_2i;
            scalar_add(&smem->a_R[i], &z, &aR_z);
            scalar_mul_mod_n(&smem->s_R[i], &xx, &sR_x);
            scalar_add(&aR_z, &sR_x, &aR_z_sR_x);
            scalar_mul_mod_n(&y_powers[i], &aR_z_sR_x, &yi_term);
            scalar_mul_mod_n(&z2, &two_powers[i], &z2_2i);
            scalar_add(&yi_term, &z2_2i, &r_x[i]);

            Scalar prod;
            scalar_mul_mod_n(&l_x[i], &r_x[i], &prod);
            scalar_add(&t_hat, &prod, &t_hat);
        }
        proof->t_hat = t_hat;

        // tau_x, mu
        Scalar xx2;
        scalar_mul_mod_n(&xx, &xx, &xx2);
        Scalar tau2_x2, tau1_x, z2_blind;
        scalar_mul_mod_n(&tau2, &xx2, &tau2_x2);
        scalar_mul_mod_n(&tau1, &xx, &tau1_x);
        scalar_mul_mod_n(&z2, blinding, &z2_blind);
        Scalar tau_x;
        scalar_add(&tau2_x2, &tau1_x, &tau_x);
        scalar_add(&tau_x, &z2_blind, &tau_x);
        proof->tau_x = tau_x;

        Scalar rho_s;
        scalar_from_bytes(rho_hash, &rho_s);
        Scalar rho_x;
        scalar_mul_mod_n(&rho_s, &xx, &rho_x);
        scalar_add(&smem->alpha, &rho_x, &proof->mu);

        // ---- Inner Product Argument ----
        // Round 0 uses LUT4 for G_vec (original generators).
        // Rounds 1-5 use standard scalar_mul (generators recombined).
        Scalar a_vec[64], b_vec[64];
        for (int i = 0; i < 64; ++i) { a_vec[i] = l_x[i]; b_vec[i] = r_x[i]; }

        Scalar y_inv;
        scalar_inverse(&y, &y_inv);
        Scalar y_inv_pow = ONE_S;

        JacobianPoint G_vec[64], H_vec[64];
        for (int i = 0; i < 64; ++i) {
            G_vec[i].x = g_bulletproof_G[i].x; G_vec[i].y = g_bulletproof_G[i].y;
            G_vec[i].z = FIELD_ONE; G_vec[i].infinity = false;

            JacobianPoint Hi_jac;
            Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
            Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
            scalar_mul(&Hi_jac, &y_inv_pow, &H_vec[i]);
            scalar_mul_mod_n(&y_inv_pow, &y_inv, &y_inv_pow);
        }

        int n = 64;
        for (int round = 0; round < 6; ++round) {
            n /= 2;

            Scalar c_L = ZERO_S, c_R = ZERO_S;
            JacobianPoint L_pt, R_pt;
            L_pt.infinity = true; L_pt.z = FIELD_ONE;
            R_pt.infinity = true; R_pt.z = FIELD_ONE;

            for (int i = 0; i < n; ++i) {
                JacobianPoint aG, bH;
                // G_vec: LUT4 in round 0 (original generators), scalar_mul after
                if (round == 0)
                    scalar_mul_bp_lut4(&g_bp_lut4[(n + i) * BP_LUT4_GEN_STRIDE], &a_vec[i], &aG);
                else
                    scalar_mul(&G_vec[n + i], &a_vec[i], &aG);
                jacobian_add(&L_pt, &aG, &L_pt);
                scalar_mul(&H_vec[i], &b_vec[n + i], &bH);
                jacobian_add(&L_pt, &bH, &L_pt);
                Scalar prod;
                scalar_mul_mod_n(&a_vec[i], &b_vec[n + i], &prod);
                scalar_add(&c_L, &prod, &c_L);

                if (round == 0)
                    scalar_mul_bp_lut4(&g_bp_lut4[i * BP_LUT4_GEN_STRIDE], &a_vec[n + i], &aG);
                else
                    scalar_mul(&G_vec[i], &a_vec[n + i], &aG);
                jacobian_add(&R_pt, &aG, &R_pt);
                scalar_mul(&H_vec[n + i], &b_vec[i], &bH);
                jacobian_add(&R_pt, &bH, &R_pt);
                scalar_mul_mod_n(&a_vec[n + i], &b_vec[i], &prod);
                scalar_add(&c_R, &prod, &c_R);
            }
            JacobianPoint cU;
            scalar_mul(&H_jac, &c_L, &cU);
            jacobian_add(&L_pt, &cU, &L_pt);
            scalar_mul(&H_jac, &c_R, &cU);
            jacobian_add(&R_pt, &cU, &R_pt);

            field_inv(&L_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
            field_mul(&L_pt.x, &zi2, &proof->L[round].x);
            field_mul(&L_pt.y, &zi3, &proof->L[round].y);

            field_inv(&R_pt.z, &zi); field_sqr(&zi, &zi2); field_mul(&zi2, &zi, &zi3);
            field_mul(&R_pt.x, &zi2, &proof->R[round].x);
            field_mul(&R_pt.y, &zi3, &proof->R[round].y);

            uint8_t l_comp[33], r_comp[33];
            affine_to_compressed(&proof->L[round].x, &proof->L[round].y, l_comp);
            affine_to_compressed(&proof->R[round].x, &proof->R[round].y, r_comp);
            uint8_t ip_buf[33 + 33];
            for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
            uint8_t xr_hash[32];
            zk_tagged_hash_midstate(&g_bp_ip_midstate, ip_buf, sizeof(ip_buf), xr_hash);
            Scalar x_r;
            scalar_from_bytes(xr_hash, &x_r);
            Scalar x_r_inv;
            scalar_inverse(&x_r, &x_r_inv);

            for (int i = 0; i < n; ++i) {
                Scalar a_lo_x, a_hi_xi;
                scalar_mul_mod_n(&a_vec[i], &x_r, &a_lo_x);
                scalar_mul_mod_n(&a_vec[n + i], &x_r_inv, &a_hi_xi);
                scalar_add(&a_lo_x, &a_hi_xi, &a_vec[i]);

                Scalar b_lo_xi, b_hi_x;
                scalar_mul_mod_n(&b_vec[i], &x_r_inv, &b_lo_xi);
                scalar_mul_mod_n(&b_vec[n + i], &x_r, &b_hi_x);
                scalar_add(&b_lo_xi, &b_hi_x, &b_vec[i]);
            }
            for (int i = 0; i < n; ++i) {
                JacobianPoint G_lo_xi, G_hi_x;
                // G_vec update: LUT4 in round 0
                if (round == 0) {
                    scalar_mul_bp_lut4(&g_bp_lut4[i * BP_LUT4_GEN_STRIDE], &x_r_inv, &G_lo_xi);
                    scalar_mul_bp_lut4(&g_bp_lut4[(n + i) * BP_LUT4_GEN_STRIDE], &x_r, &G_hi_x);
                } else {
                    scalar_mul(&G_vec[i], &x_r_inv, &G_lo_xi);
                    scalar_mul(&G_vec[n + i], &x_r, &G_hi_x);
                }
                jacobian_add(&G_lo_xi, &G_hi_x, &G_vec[i]);

                JacobianPoint H_lo_x, H_hi_xi;
                scalar_mul(&H_vec[i], &x_r, &H_lo_x);
                scalar_mul(&H_vec[n + i], &x_r_inv, &H_hi_xi);
                jacobian_add(&H_lo_x, &H_hi_xi, &H_vec[i]);
            }
        }

        proof->a = a_vec[0];
        proof->b = b_vec[0];
    }

    __syncwarp(FULL_MASK);
    return true;
}

} // namespace ct
} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
