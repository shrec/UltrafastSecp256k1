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

} // namespace ct
} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
