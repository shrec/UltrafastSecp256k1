#pragma once
// ============================================================================
// Zero-Knowledge Proof Kernels -- CUDA device implementation
// ============================================================================
// GPU-accelerated ZK proof primitives:
//   1. Batch Schnorr knowledge proof verification
//   2. Batch DLEQ proof verification
//   3. Batch Bulletproof range proof verification (polynomial check)
//   4. Inner product argument batch verification
//
// All kernels operate on public data (verification only).
// Proving uses CT layer and stays on CPU.
//
// 64-bit limb mode only.
// ============================================================================

#include "secp256k1.cuh"
#include "pedersen.cuh"
#include "ecdsa.cuh"  // SHA256Ctx, sha256_*

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// ============================================================================
// Tagged hash device utilities
// ============================================================================

// Compute SHA256(data, len) on device
__device__ inline void sha256_hash(const uint8_t* data, size_t len, uint8_t out[32]) {
    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, data, len);
    sha256_final(&ctx, out);
}

// Tagged hash: H(SHA256(tag) || SHA256(tag) || data)
__device__ inline void zk_tagged_hash(
    const char* tag, size_t tag_len,
    const uint8_t* data, size_t data_len,
    uint8_t out[32])
{
    uint8_t tag_hash[32];
    sha256_hash(reinterpret_cast<const uint8_t*>(tag), tag_len, tag_hash);

    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, tag_hash, 32);
    sha256_update(&ctx, tag_hash, 32);
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

// ============================================================================
// Precomputed Tagged Hash Midstates
// ============================================================================
// SHA256 state after processing SHA256(tag)||SHA256(tag) (1 block = 64 bytes).
// Eliminates 2 SHA256 compressions per tagged hash call.

struct ZKTagMidstate {
    uint32_t h[8];
};

// Midstate for "ZK/nonce" (8 bytes)
__constant__ const ZKTagMidstate ZK_NONCE_MIDSTATE = {{
    0x9dc14780U, 0xff35a050U, 0x1ee52bf5U, 0xdb8cf3f4U,
    0x08d70a7bU, 0x195b809dU, 0x70f3d011U, 0x7c124c01U
}};

// Midstate for "ZK/knowledge" (12 bytes)
__constant__ const ZKTagMidstate ZK_KNOWLEDGE_MIDSTATE = {{
    0xd88665c6U, 0x57be7980U, 0x8fb37fd8U, 0x485fc7f8U,
    0x82a5716bU, 0x1db3ed4dU, 0x9dacd635U, 0xea1cfaa4U
}};

// Midstate for "ZK/dleq" (7 bytes)
__constant__ const ZKTagMidstate ZK_DLEQ_MIDSTATE = {{
    0xad61ec8eU, 0x5a747086U, 0x1dd98eefU, 0xe172f2ffU,
    0x9b119897U, 0x02f290ddU, 0x21ffc089U, 0x0a5520b9U
}};

// Tagged hash using precomputed midstate -- skips tag hashing + first block
__device__ inline void zk_tagged_hash_midstate(
    const ZKTagMidstate* midstate,
    const uint8_t* data, size_t data_len,
    uint8_t out[32])
{
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = midstate->h[i];
    ctx.buf_len = 0;
    ctx.total = 64;  // already processed 1 block (tag_hash || tag_hash)
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

// ============================================================================
// Proof structures (GPU-compatible POD)
// ============================================================================

struct KnowledgeProofGPU {
    uint8_t rx[32];  // R.x (x-coordinate of nonce point)
    Scalar s;         // response scalar
};

struct DLEQProofGPU {
    Scalar e;  // challenge
    Scalar s;  // response
};

// ============================================================================
// 1. Batch Knowledge Proof Verification
// ============================================================================
// Verifies N Schnorr knowledge proofs in parallel.
// Each thread verifies one proof: s*G == R + e*P

__device__ inline bool knowledge_verify_device(
    const KnowledgeProofGPU* proof,
    const AffinePoint* pubkey,
    const uint8_t msg[32])
{
    // Reconstruct challenge: e = H("ZK/knowledge" || R.x || P_comp || G_comp || msg)
    // Serialize pubkey to compressed format
    uint8_t p_comp[33];
    JacobianPoint p_jac;
    p_jac.x = pubkey->x;
    p_jac.y = pubkey->y;
    p_jac.z = FIELD_ONE;
    p_jac.infinity = false;
    point_to_compressed(&p_jac, p_comp);

    // Serialize generator
    uint8_t g_comp[33];
    JacobianPoint g_jac;
    g_jac.x = {{GENERATOR_X[0], GENERATOR_X[1], GENERATOR_X[2], GENERATOR_X[3]}};
    g_jac.y = {{GENERATOR_Y[0], GENERATOR_Y[1], GENERATOR_Y[2], GENERATOR_Y[3]}};
    g_jac.z = FIELD_ONE;
    g_jac.infinity = false;
    point_to_compressed(&g_jac, g_comp);

    // Build hash input: rx[32] || P_comp[33] || G_comp[33] || msg[32]
    uint8_t buf[32 + 33 + 33 + 32];
    for (int i = 0; i < 32; ++i) buf[i] = proof->rx[i];
    for (int i = 0; i < 33; ++i) buf[32 + i] = p_comp[i];
    for (int i = 0; i < 33; ++i) buf[65 + i] = g_comp[i];
    for (int i = 0; i < 32; ++i) buf[98 + i] = msg[i];

    uint8_t e_hash[32];
    zk_tagged_hash("ZK/knowledge", 12, buf, sizeof(buf), e_hash);

    Scalar e;
    scalar_from_bytes(e_hash, &e);

    // Verify: s*G == R + e*P
    // Compute s*G
    JacobianPoint sG;
    scalar_mul_generator_const(&proof->s, &sG);

    // Compute e*P
    JacobianPoint eP;
    scalar_mul(&p_jac, &e, &eP);

    // Compute R from rx (lift_x with even Y)
    FieldElement rx_fe;
    uint8_t rx_bytes[32];
    for (int i = 0; i < 32; ++i) rx_bytes[i] = proof->rx[i];

    // Parse big-endian rx to field element
    rx_fe.limbs[3] = ((uint64_t)rx_bytes[0] << 56) | ((uint64_t)rx_bytes[1] << 48) |
                 ((uint64_t)rx_bytes[2] << 40) | ((uint64_t)rx_bytes[3] << 32) |
                 ((uint64_t)rx_bytes[4] << 24) | ((uint64_t)rx_bytes[5] << 16) |
                 ((uint64_t)rx_bytes[6] << 8)  | (uint64_t)rx_bytes[7];
    rx_fe.limbs[2] = ((uint64_t)rx_bytes[8] << 56) | ((uint64_t)rx_bytes[9] << 48) |
                 ((uint64_t)rx_bytes[10] << 40) | ((uint64_t)rx_bytes[11] << 32) |
                 ((uint64_t)rx_bytes[12] << 24) | ((uint64_t)rx_bytes[13] << 16) |
                 ((uint64_t)rx_bytes[14] << 8)  | (uint64_t)rx_bytes[15];
    rx_fe.limbs[1] = ((uint64_t)rx_bytes[16] << 56) | ((uint64_t)rx_bytes[17] << 48) |
                 ((uint64_t)rx_bytes[18] << 40) | ((uint64_t)rx_bytes[19] << 32) |
                 ((uint64_t)rx_bytes[20] << 24) | ((uint64_t)rx_bytes[21] << 16) |
                 ((uint64_t)rx_bytes[22] << 8)  | (uint64_t)rx_bytes[23];
    rx_fe.limbs[0] = ((uint64_t)rx_bytes[24] << 56) | ((uint64_t)rx_bytes[25] << 48) |
                 ((uint64_t)rx_bytes[26] << 40) | ((uint64_t)rx_bytes[27] << 32) |
                 ((uint64_t)rx_bytes[28] << 24) | ((uint64_t)rx_bytes[29] << 16) |
                 ((uint64_t)rx_bytes[30] << 8)  | (uint64_t)rx_bytes[31];

    AffinePoint R_affine;
    if (!lift_x_even(&rx_fe, &R_affine)) return false;

    // R + e*P
    JacobianPoint R_plus_eP;
    jacobian_add_mixed(&eP, &R_affine, &R_plus_eP);

    // Compare s*G == R + e*P by converting both to compressed form
    uint8_t comp1[33], comp2[33];
    if (!point_to_compressed(&sG, comp1)) return false;
    if (!point_to_compressed(&R_plus_eP, comp2)) return false;

    for (int i = 0; i < 33; ++i)
        if (comp1[i] != comp2[i]) return false;
    return true;
}

__global__ void knowledge_verify_batch_kernel(
    const KnowledgeProofGPU* __restrict__ proofs,
    const AffinePoint* __restrict__ pubkeys,
    const uint8_t* __restrict__ messages,  // N * 32 bytes
    bool* __restrict__ results,
    uint32_t count)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = knowledge_verify_device(
        &proofs[idx], &pubkeys[idx], &messages[idx * 32]);
}

// ============================================================================
// 2. Batch DLEQ Proof Verification
// ============================================================================

__device__ inline bool dleq_verify_device(
    const DLEQProofGPU* proof,
    const AffinePoint* G,
    const AffinePoint* H,
    const AffinePoint* P,
    const AffinePoint* Q)
{
    // R1 = s*G - e*P
    JacobianPoint G_jac, H_jac, P_jac, Q_jac;
    G_jac.x = G->x; G_jac.y = G->y; G_jac.z = FIELD_ONE; G_jac.infinity = false;
    H_jac.x = H->x; H_jac.y = H->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;
    P_jac.x = P->x; P_jac.y = P->y; P_jac.z = FIELD_ONE; P_jac.infinity = false;
    Q_jac.x = Q->x; Q_jac.y = Q->y; Q_jac.z = FIELD_ONE; Q_jac.infinity = false;

    JacobianPoint sG, eP, sH, eQ;
    scalar_mul(&G_jac, &proof->s, &sG);
    scalar_mul(&P_jac, &proof->e, &eP);
    scalar_mul(&H_jac, &proof->s, &sH);
    scalar_mul(&Q_jac, &proof->e, &eQ);

    // R1 = sG - eP (negate eP.y)
    field_negate(&eP.y, &eP.y);
    JacobianPoint R1;
    jacobian_add(&sG, &eP, &R1);

    // R2 = sH - eQ (negate eQ.y)
    field_negate(&eQ.y, &eQ.y);
    JacobianPoint R2;
    jacobian_add(&sH, &eQ, &R2);

    // Serialize all 6 points for challenge recomputation
    uint8_t g_comp[33], h_comp[33], p_comp[33], q_comp[33], r1_comp[33], r2_comp[33];
    point_to_compressed(&G_jac, g_comp);
    point_to_compressed(&H_jac, h_comp);

    // Restore P and Q (re-init since we may have mutated via negate above)
    P_jac.x = P->x; P_jac.y = P->y; P_jac.z = FIELD_ONE; P_jac.infinity = false;
    Q_jac.x = Q->x; Q_jac.y = Q->y; Q_jac.z = FIELD_ONE; Q_jac.infinity = false;
    point_to_compressed(&P_jac, p_comp);
    point_to_compressed(&Q_jac, q_comp);
    point_to_compressed(&R1, r1_comp);
    point_to_compressed(&R2, r2_comp);

    // e_check = H("ZK/dleq" || G || H || P || Q || R1 || R2)
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
    zk_tagged_hash("ZK/dleq", 7, buf, sizeof(buf), e_hash);

    Scalar e_check;
    scalar_from_bytes(e_hash, &e_check);

    // Compare e == e_check
    return (proof->e.limbs[0] == e_check.limbs[0] &&
            proof->e.limbs[1] == e_check.limbs[1] &&
            proof->e.limbs[2] == e_check.limbs[2] &&
            proof->e.limbs[3] == e_check.limbs[3]);
}

__global__ void dleq_verify_batch_kernel(
    const DLEQProofGPU* __restrict__ proofs,
    const AffinePoint* __restrict__ G_pts,
    const AffinePoint* __restrict__ H_pts,
    const AffinePoint* __restrict__ P_pts,
    const AffinePoint* __restrict__ Q_pts,
    bool* __restrict__ results,
    uint32_t count)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = dleq_verify_device(
        &proofs[idx], &G_pts[idx], &H_pts[idx], &P_pts[idx], &Q_pts[idx]);
}

// ============================================================================
// 3. Batch Bulletproof Polynomial Check
// ============================================================================
// Verifies the polynomial commitment part of Bulletproof range proofs.
// Each thread checks: t_hat * H + tau_x * G == z^2 * V + delta * H + x * T1 + x^2 * T2

// Compact range proof data for GPU (minimal for polynomial check)
struct RangeProofPolyGPU {
    AffinePoint A;      // vector commitment A
    AffinePoint S;      // vector commitment S
    AffinePoint T1;     // polynomial commitment T1
    AffinePoint T2;     // polynomial commitment T2
    Scalar tau_x;       // blinding for polynomial eval
    Scalar t_hat;       // polynomial evaluation
};

__device__ inline bool range_proof_poly_check_device(
    const RangeProofPolyGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen)
{
    // Compute Fiat-Shamir challenges y, z, x
    // Serialize A, S, V for y/z challenge
    uint8_t a_comp[33], s_comp[33], v_comp[33];
    JacobianPoint tmp;

    tmp.x = proof->A.x; tmp.y = proof->A.y; tmp.z = FIELD_ONE; tmp.infinity = false;
    point_to_compressed(&tmp, a_comp);

    tmp.x = proof->S.x; tmp.y = proof->S.y; tmp.z = FIELD_ONE; tmp.infinity = false;
    point_to_compressed(&tmp, s_comp);

    tmp.x = commitment->x; tmp.y = commitment->y; tmp.z = FIELD_ONE; tmp.infinity = false;
    point_to_compressed(&tmp, v_comp);

    uint8_t fs_buf[33 + 33 + 33];
    for (int i = 0; i < 33; ++i) {
        fs_buf[i]      = a_comp[i];
        fs_buf[33 + i] = s_comp[i];
        fs_buf[66 + i] = v_comp[i];
    }

    uint8_t y_hash[32], z_hash[32];
    zk_tagged_hash("Bulletproof/y", 13, fs_buf, sizeof(fs_buf), y_hash);
    zk_tagged_hash("Bulletproof/z", 13, fs_buf, sizeof(fs_buf), z_hash);

    Scalar y, z;
    scalar_from_bytes(y_hash, &y);
    scalar_from_bytes(z_hash, &z);

    // Compute x from T1, T2, y, z
    uint8_t t1_comp[33], t2_comp[33];
    tmp.x = proof->T1.x; tmp.y = proof->T1.y; tmp.z = FIELD_ONE; tmp.infinity = false;
    point_to_compressed(&tmp, t1_comp);
    tmp.x = proof->T2.x; tmp.y = proof->T2.y; tmp.z = FIELD_ONE; tmp.infinity = false;
    point_to_compressed(&tmp, t2_comp);

    uint8_t y_bytes[32], z_bytes[32];
    scalar_to_bytes(&y, y_bytes);
    scalar_to_bytes(&z, z_bytes);

    uint8_t x_buf[33 + 33 + 32 + 32];
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
    for (int i = 0; i < 32; ++i) { x_buf[66 + i] = y_bytes[i]; x_buf[98 + i] = z_bytes[i]; }

    uint8_t x_hash[32];
    zk_tagged_hash("Bulletproof/x", 13, x_buf, sizeof(x_buf), x_hash);

    Scalar x;
    scalar_from_bytes(x_hash, &x);

    // Compute delta(y,z) = (z - z^2) * sum(y^i) - z^3 * sum(2^i)
    Scalar z2, z3;
    scalar_mul_mod_n(&z, &z, &z2);
    scalar_mul_mod_n(&z2, &z, &z3);

    // sum(y^i) for i in [0, 64)
    Scalar sum_y;
    sum_y.limbs[0] = 1; sum_y.limbs[1] = 0; sum_y.limbs[2] = 0; sum_y.limbs[3] = 0;
    Scalar y_pow = y;
    for (int i = 1; i < 64; ++i) {
        scalar_add(&sum_y, &y_pow, &sum_y);
        scalar_mul_mod_n(&y_pow, &y, &y_pow);
    }

    // sum(2^i) for i in [0, 64) = 2^64 - 1
    Scalar sum_2;
    sum_2.limbs[0] = 0xFFFFFFFFFFFFFFFFULL;
    sum_2.limbs[1] = 0; sum_2.limbs[2] = 0; sum_2.limbs[3] = 0;

    Scalar z_minus_z2;
    scalar_sub(&z, &z2, &z_minus_z2);

    Scalar term1, term2, delta;
    scalar_mul_mod_n(&z_minus_z2, &sum_y, &term1);
    scalar_mul_mod_n(&z3, &sum_2, &term2);
    scalar_sub(&term1, &term2, &delta);

    // LHS = t_hat * H + tau_x * G
    JacobianPoint H_jac;
    H_jac.x = H_gen->x; H_jac.y = H_gen->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;

    JacobianPoint tH, tauG, LHS;
    scalar_mul(&H_jac, &proof->t_hat, &tH);
    scalar_mul_generator_const(&proof->tau_x, &tauG);
    jacobian_add(&tH, &tauG, &LHS);

    // RHS = z^2 * V + delta * H + x * T1 + x^2 * T2
    Scalar x2;
    scalar_mul_mod_n(&x, &x, &x2);

    tmp.x = commitment->x; tmp.y = commitment->y; tmp.z = FIELD_ONE; tmp.infinity = false;
    JacobianPoint z2V, deltaH, xT1, x2T2;
    scalar_mul(&tmp, &z2, &z2V);
    scalar_mul(&H_jac, &delta, &deltaH);

    tmp.x = proof->T1.x; tmp.y = proof->T1.y; tmp.z = FIELD_ONE; tmp.infinity = false;
    scalar_mul(&tmp, &x, &xT1);

    tmp.x = proof->T2.x; tmp.y = proof->T2.y; tmp.z = FIELD_ONE; tmp.infinity = false;
    scalar_mul(&tmp, &x2, &x2T2);

    JacobianPoint RHS;
    jacobian_add(&z2V, &deltaH, &RHS);
    jacobian_add(&RHS, &xT1, &RHS);
    jacobian_add(&RHS, &x2T2, &RHS);

    // Compare LHS == RHS
    uint8_t lhs_comp[33], rhs_comp[33];
    point_to_compressed(&LHS, lhs_comp);
    point_to_compressed(&RHS, rhs_comp);

    for (int i = 0; i < 33; ++i)
        if (lhs_comp[i] != rhs_comp[i]) return false;
    return true;
}

__global__ void range_proof_poly_batch_kernel(
    const RangeProofPolyGPU* __restrict__ proofs,
    const AffinePoint* __restrict__ commitments,
    const AffinePoint* __restrict__ H_gen,
    bool* __restrict__ results,
    uint32_t count)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    results[idx] = range_proof_poly_check_device(
        &proofs[idx], &commitments[idx], H_gen);
}

// ============================================================================
// Host-side helpers
// ============================================================================

// Compute Pedersen generator H on device side (call once at init)
// Host code should:
//   1. Compute H on CPU using pedersen_generator_H()
//   2. Convert to AffinePointData
//   3. cudaMemcpy to device global g_pedersen_H
// This avoids device-side SHA-256 generator computation.

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
