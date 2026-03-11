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

// Format affine coordinates as 33-byte compressed point (no field_inv needed)
__device__ inline void affine_to_compressed(
    const FieldElement* x, const FieldElement* y, uint8_t out[33])
{
    // Extract Y parity from normalized limbs[0] bit 0 -- avoids full field_to_bytes(y)
    // Normalize Y mod p: try y - p, if no borrow then y >= p so use y - p
    constexpr uint64_t P0 = 0xFFFFFFFEFFFFFC2FULL;
    uint64_t borrow = 0;
    unsigned __int128 d0 = (unsigned __int128)y->limbs[0] - P0;
    uint64_t r0 = (uint64_t)d0;
    borrow = (uint64_t)(-(int64_t)(d0 >> 64));
    for (int i = 1; i < 4; i++) {
        unsigned __int128 di = (unsigned __int128)y->limbs[i] - 0xFFFFFFFFFFFFFFFFULL - borrow;
        borrow = (uint64_t)(-(int64_t)(di >> 64));
    }
    // If borrow==0, y >= p, use reduced r0; otherwise use original limbs[0]
    uint64_t use_reduced = -(uint64_t)(borrow == 0); // branchless mask
    uint64_t y_low = (r0 & use_reduced) | (y->limbs[0] & ~use_reduced);
    out[0] = 0x02 | (uint8_t)(y_low & 1);
    field_to_bytes(x, out + 1);
}

// Precomputed generator G compressed form (02 || Gx)
__constant__ const uint8_t G_COMPRESSED[33] = {
    0x02,
    0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
    0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
    0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
    0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
};

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

// Midstate for "Bulletproof/y" (13 bytes)
__constant__ const ZKTagMidstate ZK_BULLETPROOF_Y_MIDSTATE = {{
    0x770918afU, 0xa4791204U, 0x3c076a40U, 0x5fb23056U,
    0x902acdb9U, 0x1d85371bU, 0x10f624c4U, 0x9048ba46U
}};

// Midstate for "Bulletproof/z" (13 bytes)
__constant__ const ZKTagMidstate ZK_BULLETPROOF_Z_MIDSTATE = {{
    0x22be001aU, 0x3c79431bU, 0xe60a9432U, 0xfd965d54U,
    0x84df949fU, 0x62937ceeU, 0x20924a62U, 0x99f23a35U
}};

// Midstate for "Bulletproof/x" (13 bytes)
__constant__ const ZKTagMidstate ZK_BULLETPROOF_X_MIDSTATE = {{
    0x1378a3c8U, 0x2e8ad1b2U, 0xa47ce2e2U, 0x143037a2U,
    0xbaec0bd8U, 0x40cb0ed7U, 0xd1b23b65U, 0x43871df4U
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
    // Compress pubkey directly from affine (no field_inv needed)
    uint8_t p_comp[33];
    affine_to_compressed(&pubkey->x, &pubkey->y, p_comp);

    // Build hash input: rx[32] || P_comp[33] || G_COMPRESSED[33] || msg[32]
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
    scalar_from_bytes(e_hash, &e);

    // Verify: s*G == R + e*P
    JacobianPoint sG;
    scalar_mul_generator_const(&proof->s, &sG);

    // Compute e*P
    JacobianPoint p_jac;
    p_jac.x = pubkey->x;
    p_jac.y = pubkey->y;
    p_jac.z = FIELD_ONE;
    p_jac.infinity = false;
    JacobianPoint eP;
    scalar_mul(&p_jac, &e, &eP);

    // Compute R from rx (lift_x with even Y)
    FieldElement rx_fe;
    field_from_bytes(proof->rx, &rx_fe);

    AffinePoint R_affine;
    if (!lift_x_even(&rx_fe, &R_affine)) return false;

    // R + e*P
    JacobianPoint R_plus_eP;
    jacobian_add_mixed(&eP, &R_affine, &R_plus_eP);

    // Compare s*G == R + e*P via Jacobian cross-multiply (0 field_inv)
    // Two Jacobian points (X1:Y1:Z1) == (X2:Y2:Z2) iff:
    //   X1 * Z2^2 == X2 * Z1^2  AND  Y1 * Z2^3 == Y2 * Z1^3
    {
        FieldElement z1sq, z2sq, z1cu, z2cu;
        field_sqr(&sG.z, &z1sq);
        field_sqr(&R_plus_eP.z, &z2sq);
        field_mul(&z1sq, &sG.z, &z1cu);
        field_mul(&z2sq, &R_plus_eP.z, &z2cu);

        FieldElement lx, rx_cmp, ly, ry;
        field_mul(&sG.x, &z2sq, &lx);
        field_mul(&R_plus_eP.x, &z1sq, &rx_cmp);
        field_mul(&sG.y, &z2cu, &ly);
        field_mul(&R_plus_eP.y, &z1cu, &ry);

        // Branchless compare: subtract and check limbs directly
        // field_sub outputs in [0, p) so zero check on limbs is exact
        FieldElement dx, dy;
        field_sub(&lx, &rx_cmp, &dx);
        field_sub(&ly, &ry, &dy);
        uint64_t acc = dx.limbs[0] | dx.limbs[1] | dx.limbs[2] | dx.limbs[3]
                     | dy.limbs[0] | dy.limbs[1] | dy.limbs[2] | dy.limbs[3];
        return acc == 0;
    }
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

    // Serialize input points directly from affine (no field_inv needed -- Z=1)
    uint8_t g_comp[33], h_comp[33], p_comp[33], q_comp[33], r1_comp[33], r2_comp[33];
    affine_to_compressed(&G->x, &G->y, g_comp);
    affine_to_compressed(&H->x, &H->y, h_comp);
    affine_to_compressed(&P->x, &P->y, p_comp);
    affine_to_compressed(&Q->x, &Q->y, q_comp);
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
    zk_tagged_hash_midstate(&ZK_DLEQ_MIDSTATE, buf, sizeof(buf), e_hash);

    Scalar e_check;
    scalar_from_bytes(e_hash, &e_check);

    // Compare e == e_check
    return scalar_eq(&proof->e, &e_check);
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
    // Serialize A, S, V directly from affine (Z=1, no field_inv)
    uint8_t a_comp[33], s_comp[33], v_comp[33];
    affine_to_compressed(&proof->A.x, &proof->A.y, a_comp);
    affine_to_compressed(&proof->S.x, &proof->S.y, s_comp);
    affine_to_compressed(&commitment->x, &commitment->y, v_comp);

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

    // Serialize T1, T2 directly from affine
    uint8_t t1_comp[33], t2_comp[33];
    affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
    affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);

    uint8_t x_buf[33 + 33 + 32 + 32];
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
    // Write scalar bytes directly into x_buf (no intermediate y_bytes/z_bytes)
    scalar_to_bytes(&y, x_buf + 66);
    scalar_to_bytes(&z, x_buf + 98);

    uint8_t x_hash[32];
    zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);

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

    JacobianPoint V_jac;
    V_jac.x = commitment->x; V_jac.y = commitment->y; V_jac.z = FIELD_ONE; V_jac.infinity = false;
    JacobianPoint z2V, deltaH, xT1, x2T2;
    scalar_mul(&V_jac, &z2, &z2V);
    scalar_mul(&H_jac, &delta, &deltaH);

    JacobianPoint T1_jac;
    T1_jac.x = proof->T1.x; T1_jac.y = proof->T1.y; T1_jac.z = FIELD_ONE; T1_jac.infinity = false;
    scalar_mul(&T1_jac, &x, &xT1);

    JacobianPoint T2_jac;
    T2_jac.x = proof->T2.x; T2_jac.y = proof->T2.y; T2_jac.z = FIELD_ONE; T2_jac.infinity = false;
    scalar_mul(&T2_jac, &x2, &x2T2);

    JacobianPoint RHS;
    jacobian_add(&z2V, &deltaH, &RHS);
    jacobian_add(&RHS, &xT1, &RHS);
    jacobian_add(&RHS, &x2T2, &RHS);

    // Compare LHS == RHS via Jacobian cross-multiply (0 field_inv)
    FieldElement z1sq, z2sq, z1cu, z2cu;
    field_sqr(&LHS.z, &z1sq);
    field_sqr(&RHS.z, &z2sq);
    field_mul(&z1sq, &LHS.z, &z1cu);
    field_mul(&z2sq, &RHS.z, &z2cu);

    FieldElement lx, rx_cmp, ly, ry;
    field_mul(&LHS.x, &z2sq, &lx);
    field_mul(&RHS.x, &z1sq, &rx_cmp);
    field_mul(&LHS.y, &z2cu, &ly);
    field_mul(&RHS.y, &z1cu, &ry);

    // field_sub outputs in [0, p) so zero check on limbs is exact
    FieldElement dx, dy;
    field_sub(&lx, &rx_cmp, &dx);
    field_sub(&ly, &ry, &dy);
    uint64_t acc = dx.limbs[0] | dx.limbs[1] | dx.limbs[2] | dx.limbs[3]
                 | dy.limbs[0] | dy.limbs[1] | dy.limbs[2] | dy.limbs[3];
    return acc == 0;
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
