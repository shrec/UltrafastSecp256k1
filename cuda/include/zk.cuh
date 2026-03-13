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
#include "bp_gen_table.cuh"

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

// ============================================================================
// 4. Full Bulletproof Range Proof (verify + prove structs)
// ============================================================================

static constexpr int BP_BITS = 64;
static constexpr int BP_LOG2 = 6;

// Bulletproof generator vectors -- set by bulletproof_init_kernel()
__device__ AffinePoint g_bulletproof_G[64];
__device__ AffinePoint g_bulletproof_H[64];

// Runtime-computed midstates for "Bulletproof/ip" and "Bulletproof/gen"
__device__ ZKTagMidstate g_bp_ip_midstate;
__device__ ZKTagMidstate g_bp_gen_midstate;
__device__ bool g_bulletproof_init_done = false;

// Full range proof structure (GPU-compatible POD)
struct RangeProofGPU {
    AffinePoint A, S;           // vector commitments
    AffinePoint T1, T2;        // polynomial commitments
    Scalar tau_x, mu, t_hat;   // blinding, aggregate blinding, poly eval
    Scalar a, b;               // final IPA scalars
    AffinePoint L[6], R[6];   // inner product argument rounds
};

// -- Bulletproof init kernel --------------------------------------------------
// Computes tagged hash midstates + 128 generator points on device.
// Launch once with <<<1, 1>>> before any Bulletproof operations.

__global__ void bulletproof_init_kernel() {
    if (threadIdx.x != 0) return;

    // Compute "Bulletproof/ip" midstate (14 bytes)
    {
        const uint8_t tag[] = { 'B','u','l','l','e','t','p','r','o','o','f','/','i','p' };
        uint8_t tag_hash[32];
        sha256_hash(tag, 14, tag_hash);
        SHA256Ctx ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, tag_hash, 32);
        sha256_update(&ctx, tag_hash, 32);
        for (int i = 0; i < 8; i++) g_bp_ip_midstate.h[i] = ctx.h[i];
    }

    // Compute "Bulletproof/gen" midstate (15 bytes)
    {
        const uint8_t tag[] = { 'B','u','l','l','e','t','p','r','o','o','f','/','g','e','n' };
        uint8_t tag_hash[32];
        sha256_hash(tag, 15, tag_hash);
        SHA256Ctx ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, tag_hash, 32);
        sha256_update(&ctx, tag_hash, 32);
        for (int i = 0; i < 8; i++) g_bp_gen_midstate.h[i] = ctx.h[i];
    }

    // Compute generator vectors with try-and-increment
    for (int i = 0; i < 64; i++) {
        uint8_t buf[5];
        buf[1] = (uint8_t)(i & 0xFF);
        buf[2] = (uint8_t)((i >> 8) & 0xFF);
        buf[3] = (uint8_t)((i >> 16) & 0xFF);
        buf[4] = (uint8_t)((i >> 24) & 0xFF);

        // G_i = lift_x(H("Bulletproof/gen" || "G" || LE32(i)))
        buf[0] = 'G';
        uint8_t hash[32];
        zk_tagged_hash_midstate(&g_bp_gen_midstate, buf, 5, hash);
        FieldElement x;
        field_from_bytes(hash, &x);
        hash_to_point_increment(&x, &g_bulletproof_G[i]);

        // H_i = lift_x(H("Bulletproof/gen" || "H" || LE32(i)))
        buf[0] = 'H';
        zk_tagged_hash_midstate(&g_bp_gen_midstate, buf, 5, hash);
        field_from_bytes(hash, &x);
        hash_to_point_increment(&x, &g_bulletproof_H[i]);
    }

    g_bulletproof_init_done = true;
}

// ============================================================================
// 5. Full Bulletproof Verify Device Function
// ============================================================================
// Verifies polynomial commitment + inner product argument in a single function.

__device__ inline bool range_verify_full_device(
    const RangeProofGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen)
{
    // ---- Fiat-Shamir: recompute y, z, x ----
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

    uint8_t t1_comp[33], t2_comp[33];
    affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
    affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);

    uint8_t x_buf[33 + 33 + 32 + 32];
    for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
    scalar_to_bytes(&y, x_buf + 66);
    scalar_to_bytes(&z, x_buf + 98);

    uint8_t x_hash[32];
    zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);
    Scalar x;
    scalar_from_bytes(x_hash, &x);

    // ---- Compute delta(y,z) ----
    Scalar z2, z3, x2;
    scalar_mul_mod_n(&z, &z, &z2);
    scalar_mul_mod_n(&z2, &z, &z3);
    scalar_mul_mod_n(&x, &x, &x2);

    Scalar sum_y;
    sum_y.limbs[0] = 1; sum_y.limbs[1] = 0; sum_y.limbs[2] = 0; sum_y.limbs[3] = 0;
    Scalar y_pow = y;
    for (int i = 1; i < BP_BITS; ++i) {
        scalar_add(&sum_y, &y_pow, &sum_y);
        scalar_mul_mod_n(&y_pow, &y, &y_pow);
    }

    Scalar sum_2;
    sum_2.limbs[0] = 0xFFFFFFFFFFFFFFFFULL;
    sum_2.limbs[1] = 0; sum_2.limbs[2] = 0; sum_2.limbs[3] = 0;

    Scalar z_minus_z2, term1, term2, delta;
    scalar_sub(&z, &z2, &z_minus_z2);
    scalar_mul_mod_n(&z_minus_z2, &sum_y, &term1);
    scalar_mul_mod_n(&z3, &sum_2, &term2);
    scalar_sub(&term1, &term2, &delta);

    // ---- Polynomial check ----
    // (t_hat - delta)*H + tau_x*G - z^2*V - x*T1 - x^2*T2 == 0
    Scalar t_hat_minus_delta;
    scalar_sub(&proof->t_hat, &delta, &t_hat_minus_delta);

    JacobianPoint H_jac;
    H_jac.x = H_gen->x; H_jac.y = H_gen->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;

    JacobianPoint tH, tauG, LHS;
    scalar_mul(&H_jac, &proof->t_hat, &tH);
    scalar_mul_generator_const(&proof->tau_x, &tauG);
    jacobian_add(&tH, &tauG, &LHS);

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

    // Compare LHS == RHS via cross-multiply
    {
        FieldElement z1sq, z2sq, z1cu, z2cu;
        field_sqr(&LHS.z, &z1sq);
        field_sqr(&RHS.z, &z2sq);
        field_mul(&z1sq, &LHS.z, &z1cu);
        field_mul(&z2sq, &RHS.z, &z2cu);

        FieldElement lx, rx_val, ly, ry;
        field_mul(&LHS.x, &z2sq, &lx);
        field_mul(&RHS.x, &z1sq, &rx_val);
        field_mul(&LHS.y, &z2cu, &ly);
        field_mul(&RHS.y, &z1cu, &ry);

        FieldElement dx, dy;
        field_sub(&lx, &rx_val, &dx);
        field_sub(&ly, &ry, &dy);
        uint64_t acc = dx.limbs[0] | dx.limbs[1] | dx.limbs[2] | dx.limbs[3]
                     | dy.limbs[0] | dy.limbs[1] | dy.limbs[2] | dy.limbs[3];
        if (acc != 0) return false;
    }

    // ---- Inner Product Argument verification ----
    // Reconstruct challenges from L, R pairs
    Scalar x_rounds[BP_LOG2];
    for (int round = 0; round < BP_LOG2; ++round) {
        uint8_t l_comp[33], r_comp[33];
        affine_to_compressed(&proof->L[round].x, &proof->L[round].y, l_comp);
        affine_to_compressed(&proof->R[round].x, &proof->R[round].y, r_comp);
        uint8_t ip_buf[33 + 33];
        for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
        uint8_t xr_hash[32];
        zk_tagged_hash_midstate(&g_bp_ip_midstate, ip_buf, sizeof(ip_buf), xr_hash);
        scalar_from_bytes(xr_hash, &x_rounds[round]);
    }

    // Compute x_inv_rounds via batch inversion
    Scalar x_inv_rounds[BP_LOG2];
    {
        Scalar acc[BP_LOG2];
        acc[0] = x_rounds[0];
        for (int j = 1; j < BP_LOG2; ++j) scalar_mul_mod_n(&acc[j-1], &x_rounds[j], &acc[j]);
        Scalar inv_acc;
        scalar_inverse(&acc[BP_LOG2 - 1], &inv_acc);
        for (int j = BP_LOG2 - 1; j >= 1; --j) {
            scalar_mul_mod_n(&inv_acc, &acc[j-1], &x_inv_rounds[j]);
            scalar_mul_mod_n(&inv_acc, &x_rounds[j], &inv_acc);
        }
        x_inv_rounds[0] = inv_acc;
    }

    // Compute y_inv and y_inv powers
    Scalar y_inv;
    scalar_inverse(&y, &y_inv);
    Scalar y_inv_powers[BP_BITS];
    y_inv_powers[0].limbs[0] = 1; y_inv_powers[0].limbs[1] = 0;
    y_inv_powers[0].limbs[2] = 0; y_inv_powers[0].limbs[3] = 0;
    for (int i = 1; i < BP_BITS; ++i)
        scalar_mul_mod_n(&y_inv_powers[i-1], &y_inv, &y_inv_powers[i]);

    // Compute s_coeff[i] = product tree of x_rounds / x_inv_rounds
    Scalar s_coeff[BP_BITS];
    s_coeff[0].limbs[0] = 1; s_coeff[0].limbs[1] = 0;
    s_coeff[0].limbs[2] = 0; s_coeff[0].limbs[3] = 0;
    for (int j = 0; j < BP_LOG2; ++j)
        scalar_mul_mod_n(&s_coeff[0], &x_inv_rounds[j], &s_coeff[0]);
    for (int i = 1; i < BP_BITS; ++i) {
        s_coeff[i].limbs[0] = 1; s_coeff[i].limbs[1] = 0;
        s_coeff[i].limbs[2] = 0; s_coeff[i].limbs[3] = 0;
        for (int jj = 0; jj < BP_LOG2; ++jj) {
            if ((i >> (BP_LOG2 - 1 - jj)) & 1)
                scalar_mul_mod_n(&s_coeff[i], &x_rounds[jj], &s_coeff[i]);
            else
                scalar_mul_mod_n(&s_coeff[i], &x_inv_rounds[jj], &s_coeff[i]);
        }
    }

    // Batch inversion of s_coeff for s_inv
    Scalar s_inv[BP_BITS];
    {
        Scalar acc[BP_BITS];
        acc[0] = s_coeff[0];
        for (int i = 1; i < BP_BITS; ++i) scalar_mul_mod_n(&acc[i-1], &s_coeff[i], &acc[i]);
        Scalar inv_acc;
        scalar_inverse(&acc[BP_BITS - 1], &inv_acc);
        for (int i = BP_BITS - 1; i >= 1; --i) {
            scalar_mul_mod_n(&inv_acc, &acc[i-1], &s_inv[i]);
            scalar_mul_mod_n(&inv_acc, &s_coeff[i], &inv_acc);
        }
        s_inv[0] = inv_acc;
    }

    // Two powers: 2^i
    Scalar two_powers[BP_BITS];
    two_powers[0].limbs[0] = 1; two_powers[0].limbs[1] = 0;
    two_powers[0].limbs[2] = 0; two_powers[0].limbs[3] = 0;
    for (int i = 1; i < BP_BITS; ++i)
        scalar_add(&two_powers[i-1], &two_powers[i-1], &two_powers[i]);

    // Build merged MSM: P_check
    // P = A + x*S + sum(g_i*G_i) + sum(h_i*H_i) - mu*G + (t_hat-a*b)*U + sum(x_j^2*L_j + x_j^{-2}*R_j)
    Scalar neg_z;
    scalar_negate(&z, &neg_z);
    Scalar ab;
    scalar_mul_mod_n(&proof->a, &proof->b, &ab);

    // Start accumulator at identity
    JacobianPoint msm_acc;
    msm_acc.infinity = true;
    msm_acc.z = FIELD_ONE;

    // A (coefficient 1)
    {
        JacobianPoint A_jac;
        A_jac.x = proof->A.x; A_jac.y = proof->A.y; A_jac.z = FIELD_ONE; A_jac.infinity = false;
        jacobian_add(&msm_acc, &A_jac, &msm_acc);
    }

    // x * S
    {
        JacobianPoint S_jac, xS;
        S_jac.x = proof->S.x; S_jac.y = proof->S.y; S_jac.z = FIELD_ONE; S_jac.infinity = false;
        scalar_mul(&S_jac, &x, &xS);
        jacobian_add(&msm_acc, &xS, &msm_acc);
    }

    // G_i and H_i contributions
    for (int i = 0; i < BP_BITS; ++i) {
        // G_i: (-z - a*s_i)
        Scalar a_si, g_coeff;
        scalar_mul_mod_n(&proof->a, &s_coeff[i], &a_si);
        scalar_sub(&neg_z, &a_si, &g_coeff);

        JacobianPoint Gi_jac, g_term;
        Gi_jac.x = g_bulletproof_G[i].x; Gi_jac.y = g_bulletproof_G[i].y;
        Gi_jac.z = FIELD_ONE; Gi_jac.infinity = false;
        scalar_mul(&Gi_jac, &g_coeff, &g_term);
        jacobian_add(&msm_acc, &g_term, &msm_acc);

        // H_i: (z + z2*2^i*y_inv^i) - b*s_inv[i]*y_inv^i
        Scalar z2_2i, z2_2i_yi, h_pcheck;
        scalar_mul_mod_n(&z2, &two_powers[i], &z2_2i);
        scalar_mul_mod_n(&z2_2i, &y_inv_powers[i], &z2_2i_yi);
        scalar_add(&z, &z2_2i_yi, &h_pcheck);

        Scalar b_si, b_si_yi, h_coeff;
        scalar_mul_mod_n(&proof->b, &s_inv[i], &b_si);
        scalar_mul_mod_n(&b_si, &y_inv_powers[i], &b_si_yi);
        scalar_sub(&h_pcheck, &b_si_yi, &h_coeff);

        JacobianPoint Hi_jac, h_term;
        Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
        Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
        scalar_mul(&Hi_jac, &h_coeff, &h_term);
        jacobian_add(&msm_acc, &h_term, &msm_acc);
    }

    // -mu * G
    {
        Scalar neg_mu;
        scalar_negate(&proof->mu, &neg_mu);
        JacobianPoint muG;
        scalar_mul_generator_const(&neg_mu, &muG);
        jacobian_add(&msm_acc, &muG, &msm_acc);
    }

    // (t_hat - a*b) * U (H_ped)
    {
        Scalar t_ab;
        scalar_sub(&proof->t_hat, &ab, &t_ab);
        JacobianPoint tU;
        scalar_mul(&H_jac, &t_ab, &tU);
        jacobian_add(&msm_acc, &tU, &msm_acc);
    }

    // L_j and R_j contributions
    for (int j = 0; j < BP_LOG2; ++j) {
        Scalar xj2, xj_inv2;
        scalar_mul_mod_n(&x_rounds[j], &x_rounds[j], &xj2);
        scalar_mul_mod_n(&x_inv_rounds[j], &x_inv_rounds[j], &xj_inv2);

        JacobianPoint Lj, Rj, lterm, rterm;
        Lj.x = proof->L[j].x; Lj.y = proof->L[j].y; Lj.z = FIELD_ONE; Lj.infinity = false;
        Rj.x = proof->R[j].x; Rj.y = proof->R[j].y; Rj.z = FIELD_ONE; Rj.infinity = false;
        scalar_mul(&Lj, &xj2, &lterm);
        scalar_mul(&Rj, &xj_inv2, &rterm);
        jacobian_add(&msm_acc, &lterm, &msm_acc);
        jacobian_add(&msm_acc, &rterm, &msm_acc);
    }

    // Check: msm_acc should be identity (infinity)
    if (msm_acc.infinity) return true;
    return field_is_zero(&msm_acc.z);
}

// ============================================================================
// 6. Warp-Cooperative Bulletproof Verify
// ============================================================================
// 32 threads (1 warp) cooperate on a single proof.
// Lane 0 handles Fiat-Shamir + polynomial check + scalar precomputation.
// All 32 lanes share the MSM work over 64 G_i + 64 H_i generators.
// Tree reduction combines partial Jacobian accumulators.
//
// Shared memory layout per warp (index = warpId within block):
//   s_coeff[64], s_inv[64], y_inv_powers[64], two_powers[64]  (scalar precomputation)
//   proof_data: neg_z, z, z2, a_proof, b_proof, t_hat, mu, x, x2, ab  (10 Scalars)
//   H_jac (1 JacobianPoint): Pedersen H as Jacobian
//   reduce_buf[32] JacobianPoints: for warp tree reduction
//   poly_ok: bool
// ============================================================================

// Shared memory struct for one warp's bulletproof verify work
struct BPWarpShared {
    Scalar s_coeff[BP_BITS];       // 64 * 32 = 2048 bytes
    Scalar s_inv[BP_BITS];         // 2048
    Scalar y_inv_powers[BP_BITS];  // 2048
    Scalar two_powers[BP_BITS];    // 2048
    // Proof-derived scalars (broadcast by lane 0)
    Scalar neg_z, z, z2, a_proof, b_proof;
    // IPA round challenges
    Scalar x_rounds[BP_LOG2];      // 6 * 32 = 192
    Scalar x_inv_rounds[BP_LOG2];  // 192
    // Pre-accumulated MSM terms (non-GiHi) computed by lane 0
    JacobianPoint msm_base;        // A + xS - muG + (t-ab)U + L/R terms + poly check
    bool poly_ok;
    // Phase 1 parallel: scalars broadcast for parallel point scalar_muls
    Scalar y_val;
    Scalar x_val, x2_val;
    Scalar delta_val;
    Scalar t_ab_val;               // t_hat - a*b
    Scalar xj_sq[BP_LOG2];         // x_rounds[j]^2
    Scalar xj_inv_sq[BP_LOG2];     // x_inv_rounds[j]^2
    Scalar y_inv_pow2[BP_LOG2];    // y_inv^(2^j) for j=0..5, used for parallel y_inv_powers
    // Phase profiling (clock64 cycle stamps, lane 0 only)
    long long phase_ts[6];             // T0..T5 for P1a/P1b/P1c/P2/P3
};

__device__ inline bool range_verify_warp_device(
    const RangeProofGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    BPWarpShared* smem)         // caller provides per-warp shared memory
{
    const int lane = threadIdx.x & 31;
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

    // ====================================================================
    // Phase 1: lane 0 does Fiat-Shamir, polynomial check, scalar precomp
    // ====================================================================
    if (lane == 0) {
        smem->poly_ok = false;  // pessimistic default

        // ---- Fiat-Shamir: recompute y, z, x ----
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

        Scalar y, z_val;
        scalar_from_bytes(y_hash, &y);
        scalar_from_bytes(z_hash, &z_val);

        uint8_t t1_comp[33], t2_comp[33];
        affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
        affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);

        uint8_t x_buf[33 + 33 + 32 + 32];
        for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
        scalar_to_bytes(&y, x_buf + 66);
        scalar_to_bytes(&z_val, x_buf + 98);

        uint8_t x_hash[32];
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);
        Scalar x;
        scalar_from_bytes(x_hash, &x);

        // ---- delta(y,z) ----
        Scalar z2_val, z3, x2;
        scalar_mul_mod_n(&z_val, &z_val, &z2_val);
        scalar_mul_mod_n(&z2_val, &z_val, &z3);
        scalar_mul_mod_n(&x, &x, &x2);

        Scalar sum_y;
        sum_y.limbs[0] = 1; sum_y.limbs[1] = 0; sum_y.limbs[2] = 0; sum_y.limbs[3] = 0;
        Scalar y_pow = y;
        for (int i = 1; i < BP_BITS; ++i) {
            scalar_add(&sum_y, &y_pow, &sum_y);
            scalar_mul_mod_n(&y_pow, &y, &y_pow);
        }

        Scalar sum_2;
        sum_2.limbs[0] = 0xFFFFFFFFFFFFFFFFULL;
        sum_2.limbs[1] = 0; sum_2.limbs[2] = 0; sum_2.limbs[3] = 0;

        Scalar z_minus_z2, term1, term2, delta;
        scalar_sub(&z_val, &z2_val, &z_minus_z2);
        scalar_mul_mod_n(&z_minus_z2, &sum_y, &term1);
        scalar_mul_mod_n(&z3, &sum_2, &term2);
        scalar_sub(&term1, &term2, &delta);

        // ---- Polynomial check ----
        Scalar t_hat_minus_delta;
        scalar_sub(&proof->t_hat, &delta, &t_hat_minus_delta);

        JacobianPoint H_jac;
        H_jac.x = H_gen->x; H_jac.y = H_gen->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;

        JacobianPoint tH, tauG, LHS;
        scalar_mul(&H_jac, &proof->t_hat, &tH);
        scalar_mul_generator_const(&proof->tau_x, &tauG);
        jacobian_add(&tH, &tauG, &LHS);

        JacobianPoint V_jac;
        V_jac.x = commitment->x; V_jac.y = commitment->y; V_jac.z = FIELD_ONE; V_jac.infinity = false;
        JacobianPoint z2V, deltaH, xT1, x2T2;
        scalar_mul(&V_jac, &z2_val, &z2V);
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

        // Compare LHS == RHS
        {
            FieldElement z1sq, z2sq, z1cu, z2cu;
            field_sqr(&LHS.z, &z1sq); field_sqr(&RHS.z, &z2sq);
            field_mul(&z1sq, &LHS.z, &z1cu); field_mul(&z2sq, &RHS.z, &z2cu);
            FieldElement lx, rx_val, ly, ry;
            field_mul(&LHS.x, &z2sq, &lx); field_mul(&RHS.x, &z1sq, &rx_val);
            field_mul(&LHS.y, &z2cu, &ly); field_mul(&RHS.y, &z1cu, &ry);
            FieldElement dx, dy;
            field_sub(&lx, &rx_val, &dx); field_sub(&ly, &ry, &dy);
            uint64_t acc = dx.limbs[0] | dx.limbs[1] | dx.limbs[2] | dx.limbs[3]
                         | dy.limbs[0] | dy.limbs[1] | dy.limbs[2] | dy.limbs[3];
            if (acc != 0) return false;  // poly check failed -- all lanes return below
        }

        // ---- IPA challenge reconstruction ----
        for (int round = 0; round < BP_LOG2; ++round) {
            uint8_t l_comp[33], r_comp[33];
            affine_to_compressed(&proof->L[round].x, &proof->L[round].y, l_comp);
            affine_to_compressed(&proof->R[round].x, &proof->R[round].y, r_comp);
            uint8_t ip_buf[33 + 33];
            for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
            uint8_t xr_hash[32];
            zk_tagged_hash_midstate(&g_bp_ip_midstate, ip_buf, sizeof(ip_buf), xr_hash);
            scalar_from_bytes(xr_hash, &smem->x_rounds[round]);
        }

        // Batch inversion of x_rounds
        {
            Scalar acc[BP_LOG2];
            acc[0] = smem->x_rounds[0];
            for (int j = 1; j < BP_LOG2; ++j) scalar_mul_mod_n(&acc[j-1], &smem->x_rounds[j], &acc[j]);
            Scalar inv_acc;
            scalar_inverse(&acc[BP_LOG2 - 1], &inv_acc);
            for (int j = BP_LOG2 - 1; j >= 1; --j) {
                scalar_mul_mod_n(&inv_acc, &acc[j-1], &smem->x_inv_rounds[j]);
                scalar_mul_mod_n(&inv_acc, &smem->x_rounds[j], &inv_acc);
            }
            smem->x_inv_rounds[0] = inv_acc;
        }

        // y_inv and y_inv_powers
        Scalar y_inv;
        scalar_inverse(&y, &y_inv);
        smem->y_inv_powers[0].limbs[0] = 1; smem->y_inv_powers[0].limbs[1] = 0;
        smem->y_inv_powers[0].limbs[2] = 0; smem->y_inv_powers[0].limbs[3] = 0;
        for (int i = 1; i < BP_BITS; ++i)
            scalar_mul_mod_n(&smem->y_inv_powers[i-1], &y_inv, &smem->y_inv_powers[i]);

        // s_coeff[i] = product tree
        smem->s_coeff[0].limbs[0] = 1; smem->s_coeff[0].limbs[1] = 0;
        smem->s_coeff[0].limbs[2] = 0; smem->s_coeff[0].limbs[3] = 0;
        for (int j = 0; j < BP_LOG2; ++j)
            scalar_mul_mod_n(&smem->s_coeff[0], &smem->x_inv_rounds[j], &smem->s_coeff[0]);
        for (int i = 1; i < BP_BITS; ++i) {
            smem->s_coeff[i].limbs[0] = 1; smem->s_coeff[i].limbs[1] = 0;
            smem->s_coeff[i].limbs[2] = 0; smem->s_coeff[i].limbs[3] = 0;
            for (int jj = 0; jj < BP_LOG2; ++jj) {
                if ((i >> (BP_LOG2 - 1 - jj)) & 1)
                    scalar_mul_mod_n(&smem->s_coeff[i], &smem->x_rounds[jj], &smem->s_coeff[i]);
                else
                    scalar_mul_mod_n(&smem->s_coeff[i], &smem->x_inv_rounds[jj], &smem->s_coeff[i]);
            }
        }

        // Batch inversion for s_inv
        {
            Scalar acc[BP_BITS];
            acc[0] = smem->s_coeff[0];
            for (int i = 1; i < BP_BITS; ++i) scalar_mul_mod_n(&acc[i-1], &smem->s_coeff[i], &acc[i]);
            Scalar inv_acc;
            scalar_inverse(&acc[BP_BITS - 1], &inv_acc);
            for (int i = BP_BITS - 1; i >= 1; --i) {
                scalar_mul_mod_n(&inv_acc, &acc[i-1], &smem->s_inv[i]);
                scalar_mul_mod_n(&inv_acc, &smem->s_coeff[i], &inv_acc);
            }
            smem->s_inv[0] = inv_acc;
        }

        // two_powers
        smem->two_powers[0].limbs[0] = 1; smem->two_powers[0].limbs[1] = 0;
        smem->two_powers[0].limbs[2] = 0; smem->two_powers[0].limbs[3] = 0;
        for (int i = 1; i < BP_BITS; ++i)
            scalar_add(&smem->two_powers[i-1], &smem->two_powers[i-1], &smem->two_powers[i]);

        // Store broadcast scalars
        Scalar neg_z_val;
        scalar_negate(&z_val, &neg_z_val);
        smem->neg_z = neg_z_val;
        smem->z     = z_val;
        smem->z2    = z2_val;
        smem->a_proof = proof->a;
        smem->b_proof = proof->b;

        Scalar ab;
        scalar_mul_mod_n(&proof->a, &proof->b, &ab);

        // Build non-GiHi MSM terms
        JacobianPoint msm;
        msm.infinity = true; msm.z = FIELD_ONE;

        // A (coefficient 1)
        {
            JacobianPoint A_jac;
            A_jac.x = proof->A.x; A_jac.y = proof->A.y; A_jac.z = FIELD_ONE; A_jac.infinity = false;
            jacobian_add(&msm, &A_jac, &msm);
        }
        // x * S
        {
            JacobianPoint S_jac, xS;
            S_jac.x = proof->S.x; S_jac.y = proof->S.y; S_jac.z = FIELD_ONE; S_jac.infinity = false;
            scalar_mul(&S_jac, &x, &xS);
            jacobian_add(&msm, &xS, &msm);
        }
        // -mu * G
        {
            Scalar neg_mu;
            scalar_negate(&proof->mu, &neg_mu);
            JacobianPoint muG;
            scalar_mul_generator_const(&neg_mu, &muG);
            jacobian_add(&msm, &muG, &msm);
        }
        // (t_hat - a*b) * U (H_ped)
        {
            Scalar t_ab;
            scalar_sub(&proof->t_hat, &ab, &t_ab);
            JacobianPoint tU;
            scalar_mul(&H_jac, &t_ab, &tU);
            jacobian_add(&msm, &tU, &msm);
        }
        // L_j and R_j contributions
        for (int j = 0; j < BP_LOG2; ++j) {
            Scalar xj2, xj_inv2;
            scalar_mul_mod_n(&smem->x_rounds[j], &smem->x_rounds[j], &xj2);
            scalar_mul_mod_n(&smem->x_inv_rounds[j], &smem->x_inv_rounds[j], &xj_inv2);
            JacobianPoint Lj, Rj, lterm, rterm;
            Lj.x = proof->L[j].x; Lj.y = proof->L[j].y; Lj.z = FIELD_ONE; Lj.infinity = false;
            Rj.x = proof->R[j].x; Rj.y = proof->R[j].y; Rj.z = FIELD_ONE; Rj.infinity = false;
            scalar_mul(&Lj, &xj2, &lterm);
            scalar_mul(&Rj, &xj_inv2, &rterm);
            jacobian_add(&msm, &lterm, &msm);
            jacobian_add(&msm, &rterm, &msm);
        }

        smem->msm_base = msm;
        smem->poly_ok = true;
    }

    // Synchronize warp -- ensure smem is visible to all lanes
    __syncwarp(FULL_MASK);

    // If polynomial check failed (lane 0 returned false above, but only for lane 0),
    // need to broadcast the result
    if (!smem->poly_ok) return false;

    // ====================================================================
    // Phase 2: All 32 lanes compute partial MSM over G_i + H_i
    // ====================================================================
    // 64 generators = 2 per lane for G, 2 per lane for H
    // lane k handles G[2k], G[2k+1], H[2k], H[2k+1]

    JacobianPoint local_acc;
    local_acc.infinity = true;
    local_acc.z = FIELD_ONE;

    if (lane < 32) {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            // G_i: (-z - a*s_i)
            Scalar a_si, g_coeff;
            scalar_mul_mod_n(&smem->a_proof, &smem->s_coeff[i], &a_si);
            scalar_sub(&smem->neg_z, &a_si, &g_coeff);

            JacobianPoint Gi_jac, g_term;
            Gi_jac.x = g_bulletproof_G[i].x; Gi_jac.y = g_bulletproof_G[i].y;
            Gi_jac.z = FIELD_ONE; Gi_jac.infinity = false;
            scalar_mul(&Gi_jac, &g_coeff, &g_term);
            jacobian_add(&local_acc, &g_term, &local_acc);

            // H_i: (z + z2*2^i*y_inv^i) - b*s_inv[i]*y_inv^i
            Scalar z2_2i, z2_2i_yi, h_pcheck;
            scalar_mul_mod_n(&smem->z2, &smem->two_powers[i], &z2_2i);
            scalar_mul_mod_n(&z2_2i, &smem->y_inv_powers[i], &z2_2i_yi);
            scalar_add(&smem->z, &z2_2i_yi, &h_pcheck);

            Scalar b_si, b_si_yi, h_coeff;
            scalar_mul_mod_n(&smem->b_proof, &smem->s_inv[i], &b_si);
            scalar_mul_mod_n(&b_si, &smem->y_inv_powers[i], &b_si_yi);
            scalar_sub(&h_pcheck, &b_si_yi, &h_coeff);

            JacobianPoint Hi_jac, h_term;
            Hi_jac.x = g_bulletproof_H[i].x; Hi_jac.y = g_bulletproof_H[i].y;
            Hi_jac.z = FIELD_ONE; Hi_jac.infinity = false;
            scalar_mul(&Hi_jac, &h_coeff, &h_term);
            jacobian_add(&local_acc, &h_term, &local_acc);
        }
    }

    // ====================================================================
    // Phase 3: Warp tree reduction of partial Jacobian accumulators
    // ====================================================================
    // 5 rounds of shuffle-based reduction (32 -> 16 -> 8 -> 4 -> 2 -> 1)
    // We use shared memory since JacobianPoint is too large for __shfl_sync

    // Declare shared memory for reduction (reuse smem region after scalar data is consumed)
    // We need 32 JacobianPoint slots for the reduction tree
    // But BPWarpShared doesn't have this -- use a separate shared buffer
    // Strategy: write local_acc to smem array, then tree-reduce in-place
    // We need the caller to also provide a reduction buffer.
    // Alternative: use __shfl_sync on individual uint32_t words (JacobianPoint = 136 bytes = 34 words)
    // That's 34 shuffles per round * 5 rounds = 170 shuffles -- acceptable

    // Shuffle-based tree reduction for JacobianPoint
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        // Receive partner's point via __shfl_down_sync (limb by limb)
        JacobianPoint partner;
        // x: 4 limbs * 2 words each = 8 words
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.x.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.x.limbs[w] >> 32), offset);
            partner.x.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        // y: 4 limbs
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.y.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.y.limbs[w] >> 32), offset);
            partner.y.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        // z: 4 limbs
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.z.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.z.limbs[w] >> 32), offset);
            partner.z.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        // infinity flag
        uint32_t inf_raw = __shfl_down_sync(FULL_MASK, (uint32_t)local_acc.infinity, offset);
        partner.infinity = (bool)inf_raw;

        if (lane + offset < 32) {
            jacobian_add(&local_acc, &partner, &local_acc);
        }
    }

    // Lane 0 now has the combined GiHi MSM result
    // Add the base MSM (A + xS - muG + ... + L/R terms)
    if (lane == 0) {
        jacobian_add(&local_acc, &smem->msm_base, &local_acc);
        // Check identity
        bool is_id = local_acc.infinity || field_is_zero(&local_acc.z);
        // Store result via __shfl_sync broadcast below
        smem->poly_ok = is_id;  // reuse poly_ok for final result
    }
    __syncwarp(FULL_MASK);

    return smem->poly_ok;
}

// ============================================================================
// 7. Precomputed-Table Warp-Cooperative Bulletproof Verify
// ============================================================================
// Same as range_verify_warp_device but uses precomputed fixed-window tables
// for the G_i/H_i scalar multiplications (Phase 2).
// WIN_BITS=4: 128 KB table, 64 windows, up to 64 mixed adds per scalar_mul
// WIN_BITS=8: 2 MB table,  32 windows, up to 32 mixed adds per scalar_mul
//
// Phase 1 (Fiat-Shamir + poly check + scalar precomp) is IDENTICAL.
// Phase 3 (warp reduction + finalize) is IDENTICAL.
// Only Phase 2 MSM uses the precomputed tables.

template<int WIN_BITS>
__device__ inline bool range_verify_warp_precomp_impl(
    const RangeProofGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    BPWarpShared* smem)
{
    static_assert(WIN_BITS == 4 || WIN_BITS == 8, "Only w=4 and w=8 supported");
    constexpr int TABLE_SIZE = (1 << WIN_BITS);

    const int lane = threadIdx.x & 31;
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

    // ====================================================================
    // Phase 1: lane 0 does Fiat-Shamir, polynomial check, scalar precomp
    // (IDENTICAL to range_verify_warp_device)
    // ====================================================================
    if (lane == 0) {
        smem->poly_ok = false;

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

        Scalar y, z_val;
        scalar_from_bytes(y_hash, &y);
        scalar_from_bytes(z_hash, &z_val);

        uint8_t t1_comp[33], t2_comp[33];
        affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
        affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);

        uint8_t x_buf[33 + 33 + 32 + 32];
        for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
        scalar_to_bytes(&y, x_buf + 66);
        scalar_to_bytes(&z_val, x_buf + 98);

        uint8_t x_hash[32];
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);
        Scalar x;
        scalar_from_bytes(x_hash, &x);

        Scalar z2_val, z3, x2;
        scalar_mul_mod_n(&z_val, &z_val, &z2_val);
        scalar_mul_mod_n(&z2_val, &z_val, &z3);
        scalar_mul_mod_n(&x, &x, &x2);

        Scalar sum_y;
        sum_y.limbs[0] = 1; sum_y.limbs[1] = 0; sum_y.limbs[2] = 0; sum_y.limbs[3] = 0;
        Scalar y_pow = y;
        for (int i = 1; i < BP_BITS; ++i) {
            scalar_add(&sum_y, &y_pow, &sum_y);
            scalar_mul_mod_n(&y_pow, &y, &y_pow);
        }

        Scalar sum_2;
        sum_2.limbs[0] = 0xFFFFFFFFFFFFFFFFULL;
        sum_2.limbs[1] = 0; sum_2.limbs[2] = 0; sum_2.limbs[3] = 0;

        Scalar z_minus_z2, term1, term2, delta;
        scalar_sub(&z_val, &z2_val, &z_minus_z2);
        scalar_mul_mod_n(&z_minus_z2, &sum_y, &term1);
        scalar_mul_mod_n(&z3, &sum_2, &term2);
        scalar_sub(&term1, &term2, &delta);

        Scalar t_hat_minus_delta;
        scalar_sub(&proof->t_hat, &delta, &t_hat_minus_delta);

        JacobianPoint H_jac;
        H_jac.x = H_gen->x; H_jac.y = H_gen->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;

        JacobianPoint tH, tauG, LHS;
        scalar_mul(&H_jac, &proof->t_hat, &tH);
        scalar_mul_generator_const(&proof->tau_x, &tauG);
        jacobian_add(&tH, &tauG, &LHS);

        JacobianPoint V_jac;
        V_jac.x = commitment->x; V_jac.y = commitment->y; V_jac.z = FIELD_ONE; V_jac.infinity = false;
        JacobianPoint z2V, deltaH, xT1, x2T2;
        scalar_mul(&V_jac, &z2_val, &z2V);
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

        {
            FieldElement z1sq, z2sq, z1cu, z2cu;
            field_sqr(&LHS.z, &z1sq); field_sqr(&RHS.z, &z2sq);
            field_mul(&z1sq, &LHS.z, &z1cu); field_mul(&z2sq, &RHS.z, &z2cu);
            FieldElement lx, rx_val, ly, ry;
            field_mul(&LHS.x, &z2sq, &lx); field_mul(&RHS.x, &z1sq, &rx_val);
            field_mul(&LHS.y, &z2cu, &ly); field_mul(&RHS.y, &z1cu, &ry);
            FieldElement dx, dy;
            field_sub(&lx, &rx_val, &dx); field_sub(&ly, &ry, &dy);
            uint64_t acc = dx.limbs[0] | dx.limbs[1] | dx.limbs[2] | dx.limbs[3]
                         | dy.limbs[0] | dy.limbs[1] | dy.limbs[2] | dy.limbs[3];
            if (acc != 0) return false;
        }

        for (int round = 0; round < BP_LOG2; ++round) {
            uint8_t l_comp[33], r_comp[33];
            affine_to_compressed(&proof->L[round].x, &proof->L[round].y, l_comp);
            affine_to_compressed(&proof->R[round].x, &proof->R[round].y, r_comp);
            uint8_t ip_buf[33 + 33];
            for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
            uint8_t xr_hash[32];
            zk_tagged_hash_midstate(&g_bp_ip_midstate, ip_buf, sizeof(ip_buf), xr_hash);
            scalar_from_bytes(xr_hash, &smem->x_rounds[round]);
        }

        {
            Scalar acc[BP_LOG2];
            acc[0] = smem->x_rounds[0];
            for (int j = 1; j < BP_LOG2; ++j) scalar_mul_mod_n(&acc[j-1], &smem->x_rounds[j], &acc[j]);
            Scalar inv_acc;
            scalar_inverse(&acc[BP_LOG2 - 1], &inv_acc);
            for (int j = BP_LOG2 - 1; j >= 1; --j) {
                scalar_mul_mod_n(&inv_acc, &acc[j-1], &smem->x_inv_rounds[j]);
                scalar_mul_mod_n(&inv_acc, &smem->x_rounds[j], &inv_acc);
            }
            smem->x_inv_rounds[0] = inv_acc;
        }

        Scalar y_inv;
        scalar_inverse(&y, &y_inv);
        smem->y_inv_powers[0].limbs[0] = 1; smem->y_inv_powers[0].limbs[1] = 0;
        smem->y_inv_powers[0].limbs[2] = 0; smem->y_inv_powers[0].limbs[3] = 0;
        for (int i = 1; i < BP_BITS; ++i)
            scalar_mul_mod_n(&smem->y_inv_powers[i-1], &y_inv, &smem->y_inv_powers[i]);

        smem->s_coeff[0].limbs[0] = 1; smem->s_coeff[0].limbs[1] = 0;
        smem->s_coeff[0].limbs[2] = 0; smem->s_coeff[0].limbs[3] = 0;
        for (int j = 0; j < BP_LOG2; ++j)
            scalar_mul_mod_n(&smem->s_coeff[0], &smem->x_inv_rounds[j], &smem->s_coeff[0]);
        for (int i = 1; i < BP_BITS; ++i) {
            smem->s_coeff[i].limbs[0] = 1; smem->s_coeff[i].limbs[1] = 0;
            smem->s_coeff[i].limbs[2] = 0; smem->s_coeff[i].limbs[3] = 0;
            for (int jj = 0; jj < BP_LOG2; ++jj) {
                if ((i >> (BP_LOG2 - 1 - jj)) & 1)
                    scalar_mul_mod_n(&smem->s_coeff[i], &smem->x_rounds[jj], &smem->s_coeff[i]);
                else
                    scalar_mul_mod_n(&smem->s_coeff[i], &smem->x_inv_rounds[jj], &smem->s_coeff[i]);
            }
        }

        {
            Scalar acc[BP_BITS];
            acc[0] = smem->s_coeff[0];
            for (int i = 1; i < BP_BITS; ++i) scalar_mul_mod_n(&acc[i-1], &smem->s_coeff[i], &acc[i]);
            Scalar inv_acc;
            scalar_inverse(&acc[BP_BITS - 1], &inv_acc);
            for (int i = BP_BITS - 1; i >= 1; --i) {
                scalar_mul_mod_n(&inv_acc, &acc[i-1], &smem->s_inv[i]);
                scalar_mul_mod_n(&inv_acc, &smem->s_coeff[i], &inv_acc);
            }
            smem->s_inv[0] = inv_acc;
        }

        smem->two_powers[0].limbs[0] = 1; smem->two_powers[0].limbs[1] = 0;
        smem->two_powers[0].limbs[2] = 0; smem->two_powers[0].limbs[3] = 0;
        for (int i = 1; i < BP_BITS; ++i)
            scalar_add(&smem->two_powers[i-1], &smem->two_powers[i-1], &smem->two_powers[i]);

        Scalar neg_z_val;
        scalar_negate(&z_val, &neg_z_val);
        smem->neg_z = neg_z_val;
        smem->z     = z_val;
        smem->z2    = z2_val;
        smem->a_proof = proof->a;
        smem->b_proof = proof->b;

        Scalar ab;
        scalar_mul_mod_n(&proof->a, &proof->b, &ab);

        JacobianPoint msm;
        msm.infinity = true; msm.z = FIELD_ONE;

        {
            JacobianPoint A_jac;
            A_jac.x = proof->A.x; A_jac.y = proof->A.y; A_jac.z = FIELD_ONE; A_jac.infinity = false;
            jacobian_add(&msm, &A_jac, &msm);
        }
        {
            JacobianPoint S_jac, xS;
            S_jac.x = proof->S.x; S_jac.y = proof->S.y; S_jac.z = FIELD_ONE; S_jac.infinity = false;
            scalar_mul(&S_jac, &x, &xS);
            jacobian_add(&msm, &xS, &msm);
        }
        {
            Scalar neg_mu;
            scalar_negate(&proof->mu, &neg_mu);
            JacobianPoint muG;
            scalar_mul_generator_const(&neg_mu, &muG);
            jacobian_add(&msm, &muG, &msm);
        }
        {
            Scalar t_ab;
            scalar_sub(&proof->t_hat, &ab, &t_ab);
            JacobianPoint tU;
            scalar_mul(&H_jac, &t_ab, &tU);
            jacobian_add(&msm, &tU, &msm);
        }
        for (int j = 0; j < BP_LOG2; ++j) {
            Scalar xj2, xj_inv2;
            scalar_mul_mod_n(&smem->x_rounds[j], &smem->x_rounds[j], &xj2);
            scalar_mul_mod_n(&smem->x_inv_rounds[j], &smem->x_inv_rounds[j], &xj_inv2);
            JacobianPoint Lj, Rj, lterm, rterm;
            Lj.x = proof->L[j].x; Lj.y = proof->L[j].y; Lj.z = FIELD_ONE; Lj.infinity = false;
            Rj.x = proof->R[j].x; Rj.y = proof->R[j].y; Rj.z = FIELD_ONE; Rj.infinity = false;
            scalar_mul(&Lj, &xj2, &lterm);
            scalar_mul(&Rj, &xj_inv2, &rterm);
            jacobian_add(&msm, &lterm, &msm);
            jacobian_add(&msm, &rterm, &msm);
        }

        smem->msm_base = msm;
        smem->poly_ok = true;
    }

    __syncwarp(FULL_MASK);
    if (!smem->poly_ok) return false;

    // ====================================================================
    // Phase 2: All 32 lanes compute partial MSM over G_i + H_i
    // PRECOMPUTED TABLE VERSION -- uses fixed-window lookup instead of scalar_mul
    // ====================================================================
    JacobianPoint local_acc;
    local_acc.infinity = true;
    local_acc.z = FIELD_ONE;

    if (lane < 32) {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            Scalar a_si, g_coeff;
            scalar_mul_mod_n(&smem->a_proof, &smem->s_coeff[i], &a_si);
            scalar_sub(&smem->neg_z, &a_si, &g_coeff);

            JacobianPoint g_term;
            if constexpr (WIN_BITS == 4)
                scalar_mul_bp_fixed_w4(&g_bp_gen_table_w4[i * TABLE_SIZE], &g_coeff, &g_term);
            else
                scalar_mul_bp_fixed_w8(&g_bp_gen_table_w8[i * TABLE_SIZE], &g_coeff, &g_term);
            jacobian_add(&local_acc, &g_term, &local_acc);

            Scalar z2_2i, z2_2i_yi, h_pcheck;
            scalar_mul_mod_n(&smem->z2, &smem->two_powers[i], &z2_2i);
            scalar_mul_mod_n(&z2_2i, &smem->y_inv_powers[i], &z2_2i_yi);
            scalar_add(&smem->z, &z2_2i_yi, &h_pcheck);

            Scalar b_si, b_si_yi, h_coeff;
            scalar_mul_mod_n(&smem->b_proof, &smem->s_inv[i], &b_si);
            scalar_mul_mod_n(&b_si, &smem->y_inv_powers[i], &b_si_yi);
            scalar_sub(&h_pcheck, &b_si_yi, &h_coeff);

            JacobianPoint h_term;
            if constexpr (WIN_BITS == 4)
                scalar_mul_bp_fixed_w4(&g_bp_gen_table_w4[(64 + i) * TABLE_SIZE], &h_coeff, &h_term);
            else
                scalar_mul_bp_fixed_w8(&g_bp_gen_table_w8[(64 + i) * TABLE_SIZE], &h_coeff, &h_term);
            jacobian_add(&local_acc, &h_term, &local_acc);
        }
    }

    // ====================================================================
    // Phase 3: Warp tree reduction (IDENTICAL to original)
    // ====================================================================
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        JacobianPoint partner;
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.x.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.x.limbs[w] >> 32), offset);
            partner.x.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.y.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.y.limbs[w] >> 32), offset);
            partner.y.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.z.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.z.limbs[w] >> 32), offset);
            partner.z.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        uint32_t inf_raw = __shfl_down_sync(FULL_MASK, (uint32_t)local_acc.infinity, offset);
        partner.infinity = (bool)inf_raw;

        if (lane + offset < 32) {
            jacobian_add(&local_acc, &partner, &local_acc);
        }
    }

    if (lane == 0) {
        jacobian_add(&local_acc, &smem->msm_base, &local_acc);
        bool is_id = local_acc.infinity || field_is_zero(&local_acc.z);
        smem->poly_ok = is_id;
    }
    __syncwarp(FULL_MASK);

    return smem->poly_ok;
}

// Public wrappers for precomputed table warp verify
__device__ inline bool range_verify_warp_precomp_w4_device(
    const RangeProofGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    BPWarpShared* smem)
{
    return range_verify_warp_precomp_impl<4>(proof, commitment, H_gen, smem);
}

__device__ inline bool range_verify_warp_precomp_w8_device(
    const RangeProofGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    BPWarpShared* smem)
{
    return range_verify_warp_precomp_impl<8>(proof, commitment, H_gen, smem);
}

// ============================================================================
// 8. Positional LUT4 Warp-Cooperative Verify (ZERO DOUBLINGS)
// ============================================================================
// Uses g_bp_lut4: positional tables where
//   lut[gen][window_j][digit_d] = d * 2^(4*j) * P
// Each scalar_mul becomes 63 mixed additions with 0 doublings.
// Phase 1 (Fiat-Shamir) is identical. Only Phase 2 MSM changes.

__device__ inline bool range_verify_warp_lut4_device(
    const RangeProofGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    BPWarpShared* smem)
{
    const int lane = threadIdx.x & 31;
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

    // Phase 1: lane 0 does Fiat-Shamir, polynomial check, scalar precomp
    // (IDENTICAL to all other warp verify variants)
    if (lane == 0) {
        smem->poly_ok = false;

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

        Scalar y, z_val;
        scalar_from_bytes(y_hash, &y);
        scalar_from_bytes(z_hash, &z_val);

        uint8_t t1_comp[33], t2_comp[33];
        affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
        affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);

        uint8_t x_buf[33 + 33 + 32 + 32];
        for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
        scalar_to_bytes(&y, x_buf + 66);
        scalar_to_bytes(&z_val, x_buf + 98);

        uint8_t x_hash[32];
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);
        Scalar x;
        scalar_from_bytes(x_hash, &x);

        Scalar z2_val, z3, x2;
        scalar_mul_mod_n(&z_val, &z_val, &z2_val);
        scalar_mul_mod_n(&z2_val, &z_val, &z3);
        scalar_mul_mod_n(&x, &x, &x2);

        Scalar sum_y;
        sum_y.limbs[0] = 1; sum_y.limbs[1] = 0; sum_y.limbs[2] = 0; sum_y.limbs[3] = 0;
        Scalar y_pow = y;
        for (int i = 1; i < BP_BITS; ++i) {
            scalar_add(&sum_y, &y_pow, &sum_y);
            scalar_mul_mod_n(&y_pow, &y, &y_pow);
        }

        Scalar sum_2;
        sum_2.limbs[0] = 0xFFFFFFFFFFFFFFFFULL;
        sum_2.limbs[1] = 0; sum_2.limbs[2] = 0; sum_2.limbs[3] = 0;

        Scalar z_minus_z2, term1, term2, delta;
        scalar_sub(&z_val, &z2_val, &z_minus_z2);
        scalar_mul_mod_n(&z_minus_z2, &sum_y, &term1);
        scalar_mul_mod_n(&z3, &sum_2, &term2);
        scalar_sub(&term1, &term2, &delta);

        Scalar t_hat_minus_delta;
        scalar_sub(&proof->t_hat, &delta, &t_hat_minus_delta);

        JacobianPoint H_jac;
        H_jac.x = H_gen->x; H_jac.y = H_gen->y; H_jac.z = FIELD_ONE; H_jac.infinity = false;

        JacobianPoint tH, tauG, LHS;
        scalar_mul(&H_jac, &proof->t_hat, &tH);
        scalar_mul_generator_const(&proof->tau_x, &tauG);
        jacobian_add(&tH, &tauG, &LHS);

        JacobianPoint V_jac;
        V_jac.x = commitment->x; V_jac.y = commitment->y; V_jac.z = FIELD_ONE; V_jac.infinity = false;
        JacobianPoint z2V, deltaH, xT1, x2T2;
        scalar_mul(&V_jac, &z2_val, &z2V);
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

        {
            FieldElement z1sq, z2sq, z1cu, z2cu;
            field_sqr(&LHS.z, &z1sq); field_sqr(&RHS.z, &z2sq);
            field_mul(&z1sq, &LHS.z, &z1cu); field_mul(&z2sq, &RHS.z, &z2cu);
            FieldElement lx, rx_val, ly, ry;
            field_mul(&LHS.x, &z2sq, &lx); field_mul(&RHS.x, &z1sq, &rx_val);
            field_mul(&LHS.y, &z2cu, &ly); field_mul(&RHS.y, &z1cu, &ry);
            FieldElement dx, dy;
            field_sub(&lx, &rx_val, &dx); field_sub(&ly, &ry, &dy);
            uint64_t acc = dx.limbs[0] | dx.limbs[1] | dx.limbs[2] | dx.limbs[3]
                         | dy.limbs[0] | dy.limbs[1] | dy.limbs[2] | dy.limbs[3];
            if (acc != 0) return false;
        }

        for (int round = 0; round < BP_LOG2; ++round) {
            uint8_t l_comp[33], r_comp[33];
            affine_to_compressed(&proof->L[round].x, &proof->L[round].y, l_comp);
            affine_to_compressed(&proof->R[round].x, &proof->R[round].y, r_comp);
            uint8_t ip_buf[33 + 33];
            for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
            uint8_t xr_hash[32];
            zk_tagged_hash_midstate(&g_bp_ip_midstate, ip_buf, sizeof(ip_buf), xr_hash);
            scalar_from_bytes(xr_hash, &smem->x_rounds[round]);
        }

        {
            Scalar acc[BP_LOG2];
            acc[0] = smem->x_rounds[0];
            for (int j = 1; j < BP_LOG2; ++j) scalar_mul_mod_n(&acc[j-1], &smem->x_rounds[j], &acc[j]);
            Scalar inv_acc;
            scalar_inverse(&acc[BP_LOG2 - 1], &inv_acc);
            for (int j = BP_LOG2 - 1; j >= 1; --j) {
                scalar_mul_mod_n(&inv_acc, &acc[j-1], &smem->x_inv_rounds[j]);
                scalar_mul_mod_n(&inv_acc, &smem->x_rounds[j], &inv_acc);
            }
            smem->x_inv_rounds[0] = inv_acc;
        }

        Scalar y_inv;
        scalar_inverse(&y, &y_inv);
        smem->y_inv_powers[0].limbs[0] = 1; smem->y_inv_powers[0].limbs[1] = 0;
        smem->y_inv_powers[0].limbs[2] = 0; smem->y_inv_powers[0].limbs[3] = 0;
        for (int i = 1; i < BP_BITS; ++i)
            scalar_mul_mod_n(&smem->y_inv_powers[i-1], &y_inv, &smem->y_inv_powers[i]);

        smem->s_coeff[0].limbs[0] = 1; smem->s_coeff[0].limbs[1] = 0;
        smem->s_coeff[0].limbs[2] = 0; smem->s_coeff[0].limbs[3] = 0;
        for (int j = 0; j < BP_LOG2; ++j)
            scalar_mul_mod_n(&smem->s_coeff[0], &smem->x_inv_rounds[j], &smem->s_coeff[0]);
        for (int i = 1; i < BP_BITS; ++i) {
            smem->s_coeff[i].limbs[0] = 1; smem->s_coeff[i].limbs[1] = 0;
            smem->s_coeff[i].limbs[2] = 0; smem->s_coeff[i].limbs[3] = 0;
            for (int jj = 0; jj < BP_LOG2; ++jj) {
                if ((i >> (BP_LOG2 - 1 - jj)) & 1)
                    scalar_mul_mod_n(&smem->s_coeff[i], &smem->x_rounds[jj], &smem->s_coeff[i]);
                else
                    scalar_mul_mod_n(&smem->s_coeff[i], &smem->x_inv_rounds[jj], &smem->s_coeff[i]);
            }
        }

        {
            Scalar acc[BP_BITS];
            acc[0] = smem->s_coeff[0];
            for (int i = 1; i < BP_BITS; ++i) scalar_mul_mod_n(&acc[i-1], &smem->s_coeff[i], &acc[i]);
            Scalar inv_acc;
            scalar_inverse(&acc[BP_BITS - 1], &inv_acc);
            for (int i = BP_BITS - 1; i >= 1; --i) {
                scalar_mul_mod_n(&inv_acc, &acc[i-1], &smem->s_inv[i]);
                scalar_mul_mod_n(&inv_acc, &smem->s_coeff[i], &inv_acc);
            }
            smem->s_inv[0] = inv_acc;
        }

        smem->two_powers[0].limbs[0] = 1; smem->two_powers[0].limbs[1] = 0;
        smem->two_powers[0].limbs[2] = 0; smem->two_powers[0].limbs[3] = 0;
        for (int i = 1; i < BP_BITS; ++i)
            scalar_add(&smem->two_powers[i-1], &smem->two_powers[i-1], &smem->two_powers[i]);

        Scalar neg_z_val;
        scalar_negate(&z_val, &neg_z_val);
        smem->neg_z = neg_z_val;
        smem->z     = z_val;
        smem->z2    = z2_val;
        smem->a_proof = proof->a;
        smem->b_proof = proof->b;

        Scalar ab;
        scalar_mul_mod_n(&proof->a, &proof->b, &ab);

        JacobianPoint msm;
        msm.infinity = true; msm.z = FIELD_ONE;

        {
            JacobianPoint A_jac;
            A_jac.x = proof->A.x; A_jac.y = proof->A.y; A_jac.z = FIELD_ONE; A_jac.infinity = false;
            jacobian_add(&msm, &A_jac, &msm);
        }
        {
            JacobianPoint S_jac, xS;
            S_jac.x = proof->S.x; S_jac.y = proof->S.y; S_jac.z = FIELD_ONE; S_jac.infinity = false;
            scalar_mul(&S_jac, &x, &xS);
            jacobian_add(&msm, &xS, &msm);
        }
        {
            Scalar neg_mu;
            scalar_negate(&proof->mu, &neg_mu);
            JacobianPoint muG;
            scalar_mul_generator_const(&neg_mu, &muG);
            jacobian_add(&msm, &muG, &msm);
        }
        {
            Scalar t_ab;
            scalar_sub(&proof->t_hat, &ab, &t_ab);
            JacobianPoint tU;
            scalar_mul(&H_jac, &t_ab, &tU);
            jacobian_add(&msm, &tU, &msm);
        }
        for (int j = 0; j < BP_LOG2; ++j) {
            Scalar xj2, xj_inv2;
            scalar_mul_mod_n(&smem->x_rounds[j], &smem->x_rounds[j], &xj2);
            scalar_mul_mod_n(&smem->x_inv_rounds[j], &smem->x_inv_rounds[j], &xj_inv2);
            JacobianPoint Lj, Rj, lterm, rterm;
            Lj.x = proof->L[j].x; Lj.y = proof->L[j].y; Lj.z = FIELD_ONE; Lj.infinity = false;
            Rj.x = proof->R[j].x; Rj.y = proof->R[j].y; Rj.z = FIELD_ONE; Rj.infinity = false;
            scalar_mul(&Lj, &xj2, &lterm);
            scalar_mul(&Rj, &xj_inv2, &rterm);
            jacobian_add(&msm, &lterm, &msm);
            jacobian_add(&msm, &rterm, &msm);
        }

        smem->msm_base = msm;
        smem->poly_ok = true;
    }

    __syncwarp(FULL_MASK);
    if (!smem->poly_ok) return false;

    // ====================================================================
    // Phase 2: POSITIONAL LUT4 -- ZERO DOUBLINGS
    // scalar_mul_bp_lut4 does 63 mixed additions, 0 doublings per scalar
    // ====================================================================
    JacobianPoint local_acc;
    local_acc.infinity = true;
    local_acc.z = FIELD_ONE;

    if (lane < 32) {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            Scalar a_si, g_coeff;
            scalar_mul_mod_n(&smem->a_proof, &smem->s_coeff[i], &a_si);
            scalar_sub(&smem->neg_z, &a_si, &g_coeff);

            JacobianPoint g_term;
            scalar_mul_bp_lut4(&g_bp_lut4[i * BP_LUT4_GEN_STRIDE], &g_coeff, &g_term);
            jacobian_add(&local_acc, &g_term, &local_acc);

            Scalar z2_2i, z2_2i_yi, h_pcheck;
            scalar_mul_mod_n(&smem->z2, &smem->two_powers[i], &z2_2i);
            scalar_mul_mod_n(&z2_2i, &smem->y_inv_powers[i], &z2_2i_yi);
            scalar_add(&smem->z, &z2_2i_yi, &h_pcheck);

            Scalar b_si, b_si_yi, h_coeff;
            scalar_mul_mod_n(&smem->b_proof, &smem->s_inv[i], &b_si);
            scalar_mul_mod_n(&b_si, &smem->y_inv_powers[i], &b_si_yi);
            scalar_sub(&h_pcheck, &b_si_yi, &h_coeff);

            JacobianPoint h_term;
            scalar_mul_bp_lut4(&g_bp_lut4[(64 + i) * BP_LUT4_GEN_STRIDE], &h_coeff, &h_term);
            jacobian_add(&local_acc, &h_term, &local_acc);
        }
    }

    // Phase 3: Warp tree reduction (IDENTICAL)
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        JacobianPoint partner;
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.x.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.x.limbs[w] >> 32), offset);
            partner.x.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.y.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.y.limbs[w] >> 32), offset);
            partner.y.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.z.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.z.limbs[w] >> 32), offset);
            partner.z.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        uint32_t inf_raw = __shfl_down_sync(FULL_MASK, (uint32_t)local_acc.infinity, offset);
        partner.infinity = (bool)inf_raw;

        if (lane + offset < 32) {
            jacobian_add(&local_acc, &partner, &local_acc);
        }
    }

    if (lane == 0) {
        jacobian_add(&local_acc, &smem->msm_base, &local_acc);
        bool is_id = local_acc.infinity || field_is_zero(&local_acc.z);
        smem->poly_ok = is_id;
    }
    __syncwarp(FULL_MASK);

    return smem->poly_ok;
}

// ============================================================================
// 9. Phase-1-Parallel Positional LUT4 Warp-Cooperative Verify
// ============================================================================
// Optimizations over range_verify_warp_lut4_device:
//   Phase 1a: Lanes 1-6 compute IPA challenge hashes in parallel with
//             lane 0's Fiat-Shamir derivation.
//   Phase 1b: Lane 0 does scalar prep (batch inversions, s_coeff, etc.)
//   Phase 1c: 22 lanes compute point scalar_muls in parallel. Poly check
//             is folded into MSM base (LHS - RHS + MSM) for single warp
//             reduction. If poly passes, LHS-RHS=0 → combined = MSM.
//   Phase 2:  Identical LUT4 zero-doubling MSM.
//   Phase 3:  Identical warp tree reduction + identity check.
// ============================================================================

__device__ inline bool range_verify_warp_lut4_p1par_device(
    const RangeProofGPU* proof,
    const AffinePoint* commitment,
    const AffinePoint* H_gen,
    BPWarpShared* smem)
{
    const int lane = threadIdx.x & 31;
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

    // T0: start
    if (lane == 0) smem->phase_ts[0] = clock64();

    // ================================================================
    // Phase 1a: PARALLEL Fiat-Shamir + IPA Challenge Derivation
    // ================================================================
    // Lane 0: Fiat-Shamir → y,z,x + sum_y → delta + t_ab
    // Lanes 1-6: IPA challenge hashes (independent of y,z,x)

    if (lane == 0) {
        smem->poly_ok = false;

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

        Scalar y, z_val;
        scalar_from_bytes(y_hash, &y);
        scalar_from_bytes(z_hash, &z_val);

        Scalar z2_val, z3;
        scalar_mul_mod_n(&z_val, &z_val, &z2_val);
        scalar_mul_mod_n(&z2_val, &z_val, &z3);

        uint8_t t1_comp[33], t2_comp[33];
        affine_to_compressed(&proof->T1.x, &proof->T1.y, t1_comp);
        affine_to_compressed(&proof->T2.x, &proof->T2.y, t2_comp);

        uint8_t x_buf[33 + 33 + 32 + 32];
        for (int i = 0; i < 33; ++i) { x_buf[i] = t1_comp[i]; x_buf[33 + i] = t2_comp[i]; }
        scalar_to_bytes(&y, x_buf + 66);
        scalar_to_bytes(&z_val, x_buf + 98);

        uint8_t x_hash[32];
        zk_tagged_hash_midstate(&ZK_BULLETPROOF_X_MIDSTATE, x_buf, sizeof(x_buf), x_hash);
        Scalar x;
        scalar_from_bytes(x_hash, &x);

        Scalar x2;
        scalar_mul_mod_n(&x, &x, &x2);

        // sum_y → delta
        Scalar sum_y;
        sum_y.limbs[0] = 1; sum_y.limbs[1] = 0; sum_y.limbs[2] = 0; sum_y.limbs[3] = 0;
        Scalar y_pow = y;
        for (int i = 1; i < BP_BITS; ++i) {
            scalar_add(&sum_y, &y_pow, &sum_y);
            scalar_mul_mod_n(&y_pow, &y, &y_pow);
        }

        Scalar sum_2;
        sum_2.limbs[0] = 0xFFFFFFFFFFFFFFFFULL;
        sum_2.limbs[1] = 0; sum_2.limbs[2] = 0; sum_2.limbs[3] = 0;

        Scalar z_minus_z2, term1, term2, delta;
        scalar_sub(&z_val, &z2_val, &z_minus_z2);
        scalar_mul_mod_n(&z_minus_z2, &sum_y, &term1);
        scalar_mul_mod_n(&z3, &sum_2, &term2);
        scalar_sub(&term1, &term2, &delta);

        // t_ab = t_hat - a*b
        Scalar ab, t_ab;
        scalar_mul_mod_n(&proof->a, &proof->b, &ab);
        scalar_sub(&proof->t_hat, &ab, &t_ab);

        Scalar neg_z_val;
        scalar_negate(&z_val, &neg_z_val);

        // Broadcast to shared memory
        smem->y_val = y;
        smem->z = z_val;
        smem->z2 = z2_val;
        smem->x_val = x;
        smem->x2_val = x2;
        smem->delta_val = delta;
        smem->t_ab_val = t_ab;
        smem->neg_z = neg_z_val;
        smem->a_proof = proof->a;
        smem->b_proof = proof->b;

    } else if (lane >= 1 && lane <= BP_LOG2) {
        // Lanes 1-6: IPA challenge hashes (independent of y,z,x)
        int round = lane - 1;
        uint8_t l_comp[33], r_comp[33];
        affine_to_compressed(&proof->L[round].x, &proof->L[round].y, l_comp);
        affine_to_compressed(&proof->R[round].x, &proof->R[round].y, r_comp);
        uint8_t ip_buf[33 + 33];
        for (int i = 0; i < 33; ++i) { ip_buf[i] = l_comp[i]; ip_buf[33 + i] = r_comp[i]; }
        uint8_t xr_hash[32];
        zk_tagged_hash_midstate(&g_bp_ip_midstate, ip_buf, sizeof(ip_buf), xr_hash);
        scalar_from_bytes(xr_hash, &smem->x_rounds[round]);
    }

    __syncwarp(FULL_MASK);

    // T1: after Phase 1a
    if (lane == 0) smem->phase_ts[1] = clock64();

    // ================================================================
    // Phase 1b: Scalar prep — fully parallelized
    //   1b-serial (lane 0): batch inversion x_rounds+y, xj_sq, y_inv_pow2
    //   1b-parallel (all 32 lanes): s_coeff + s_inv + y_inv_powers + two_powers
    //   No batch inversion for s_inv (uses flipped bit logic)
    // ================================================================

    // --- 1b-serial: lane 0 computes inversions and y_inv_pow2 basis ---
    if (lane == 0) {
        // Combined batch inversion: x_rounds[0..5] + y → 7 scalars, 1 inverse
        Scalar acc[BP_LOG2 + 1];
        acc[0] = smem->x_rounds[0];
        for (int j = 1; j < BP_LOG2; ++j)
            scalar_mul_mod_n(&acc[j-1], &smem->x_rounds[j], &acc[j]);
        scalar_mul_mod_n(&acc[BP_LOG2 - 1], &smem->y_val, &acc[BP_LOG2]);

        Scalar inv_acc;
        scalar_inverse(&acc[BP_LOG2], &inv_acc);

        // Extract y_inv (last element)
        Scalar y_inv;
        scalar_mul_mod_n(&inv_acc, &acc[BP_LOG2 - 1], &y_inv);
        scalar_mul_mod_n(&inv_acc, &smem->y_val, &inv_acc);

        // Extract x_inv_rounds
        for (int j = BP_LOG2 - 1; j >= 1; --j) {
            scalar_mul_mod_n(&inv_acc, &acc[j-1], &smem->x_inv_rounds[j]);
            scalar_mul_mod_n(&inv_acc, &smem->x_rounds[j], &inv_acc);
        }
        smem->x_inv_rounds[0] = inv_acc;

        // xj^2 and xj_inv^2 for Phase 1c L/R terms
        for (int j = 0; j < BP_LOG2; ++j) {
            scalar_mul_mod_n(&smem->x_rounds[j], &smem->x_rounds[j], &smem->xj_sq[j]);
            scalar_mul_mod_n(&smem->x_inv_rounds[j], &smem->x_inv_rounds[j], &smem->xj_inv_sq[j]);
        }

        // y_inv_pow2[j] = y_inv^(2^j) for binary decomposition
        smem->y_inv_pow2[0] = y_inv;  // y_inv^1
        for (int j = 1; j < BP_LOG2; ++j)
            scalar_mul_mod_n(&smem->y_inv_pow2[j-1], &smem->y_inv_pow2[j-1], &smem->y_inv_pow2[j]);

        smem->poly_ok = true;
    }

    __syncwarp(FULL_MASK);
    if (!smem->poly_ok) return false;

    // --- 1b-parallel: all 32 lanes compute s_coeff, s_inv, y_inv_powers, two_powers ---
    {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            // s_coeff[i] and s_inv[i] simultaneously (flipped bit logic)
            Scalar sc, si;
            sc.limbs[0] = 1; sc.limbs[1] = 0; sc.limbs[2] = 0; sc.limbs[3] = 0;
            si.limbs[0] = 1; si.limbs[1] = 0; si.limbs[2] = 0; si.limbs[3] = 0;
            for (int jj = 0; jj < BP_LOG2; ++jj) {
                if ((i >> (BP_LOG2 - 1 - jj)) & 1) {
                    scalar_mul_mod_n(&sc, &smem->x_rounds[jj], &sc);
                    scalar_mul_mod_n(&si, &smem->x_inv_rounds[jj], &si);
                } else {
                    scalar_mul_mod_n(&sc, &smem->x_inv_rounds[jj], &sc);
                    scalar_mul_mod_n(&si, &smem->x_rounds[jj], &si);
                }
            }
            smem->s_coeff[i] = sc;
            smem->s_inv[i] = si;

            // y_inv_powers[i] = y_inv^i via binary decomposition of i
            Scalar yp;
            yp.limbs[0] = 1; yp.limbs[1] = 0; yp.limbs[2] = 0; yp.limbs[3] = 0;
            for (int jj = 0; jj < BP_LOG2; ++jj) {
                if ((i >> jj) & 1)
                    scalar_mul_mod_n(&yp, &smem->y_inv_pow2[jj], &yp);
            }
            smem->y_inv_powers[i] = yp;

            // two_powers[i] = 2^i (direct bit shift, no arithmetic)
            Scalar tp;
            tp.limbs[0] = (i < 64) ? (1ULL << i) : 0;
            tp.limbs[1] = 0;
            tp.limbs[2] = 0;
            tp.limbs[3] = 0;
            smem->two_powers[i] = tp;
        }
    }

    __syncwarp(FULL_MASK);

    // T2: after Phase 1b
    if (lane == 0) smem->phase_ts[2] = clock64();

    // ================================================================
    // Phase 1c: UNIFORM GLV scalar_muls — WARP-DIVERGENCE-FREE
    // ================================================================
    // All active lanes call scalar_mul_glv in lockstep with lane-specific data.
    // Eliminates 14x warp divergence from the previous if/else chain.
    //
    // H terms merged: (t_hat + t_ab - delta) * H  (was 3 separate lanes)
    // G terms merged: (tau_x - mu) * G            (was 2 separate lanes)
    //
    // Lane assignments (19 active, 13 idle):
    //  0: combined_H * H    7: L[0]*xj_sq[0]     14: R[3]*xj_inv_sq[3]
    //  1: combined_G * G    8: R[0]*xj_inv_sq[0]  15: L[4]*xj_sq[4]
    //  2: -z2 * V           9: L[1]*xj_sq[1]     16: R[4]*xj_inv_sq[4]
    //  3: -x * T1          10: R[1]*xj_inv_sq[1]  17: L[5]*xj_sq[5]
    //  4: -x2 * T2         11: L[2]*xj_sq[2]     18: R[5]*xj_inv_sq[5]
    //  5: 1 * A             12: R[2]*xj_inv_sq[2]  19-31: identity
    //  6: x * S             13: L[3]*xj_sq[3]

    // --- Data loading (minor divergence, register ops only) ---
    JacobianPoint my_jac;
    my_jac.z = FIELD_ONE;
    my_jac.infinity = false;
    Scalar my_scalar;

    switch (lane) {
    case 0: {
        // Combined H: (t_hat + t_ab - delta) * H
        my_jac.x = H_gen->x; my_jac.y = H_gen->y;
        scalar_add(&proof->t_hat, &smem->t_ab_val, &my_scalar);
        scalar_sub(&my_scalar, &smem->delta_val, &my_scalar);
    } break;
    case 1: {
        // Combined G: (tau_x - mu) * G
        my_jac.x = GENERATOR_TABLE_AFFINE[1].x;
        my_jac.y = GENERATOR_TABLE_AFFINE[1].y;
        scalar_sub(&proof->tau_x, &proof->mu, &my_scalar);
    } break;
    case 2:
        my_jac.x = commitment->x; my_jac.y = commitment->y;
        scalar_negate(&smem->z2, &my_scalar);
        break;
    case 3:
        my_jac.x = proof->T1.x; my_jac.y = proof->T1.y;
        scalar_negate(&smem->x_val, &my_scalar);
        break;
    case 4:
        my_jac.x = proof->T2.x; my_jac.y = proof->T2.y;
        scalar_negate(&smem->x2_val, &my_scalar);
        break;
    case 5:
        my_jac.x = proof->A.x; my_jac.y = proof->A.y;
        my_scalar.limbs[0] = 1; my_scalar.limbs[1] = 0;
        my_scalar.limbs[2] = 0; my_scalar.limbs[3] = 0;
        break;
    case 6:
        my_jac.x = proof->S.x; my_jac.y = proof->S.y;
        my_scalar = smem->x_val;
        break;
    default:
        if (lane >= 7 && lane <= 18) {
            int idx = lane - 7;
            int j = idx >> 1;
            if (idx & 1) {
                my_jac.x = proof->R[j].x; my_jac.y = proof->R[j].y;
                my_scalar = smem->xj_inv_sq[j];
            } else {
                my_jac.x = proof->L[j].x; my_jac.y = proof->L[j].y;
                my_scalar = smem->xj_sq[j];
            }
        } else {
            my_jac.infinity = true;
            my_scalar.limbs[0] = 0; my_scalar.limbs[1] = 0;
            my_scalar.limbs[2] = 0; my_scalar.limbs[3] = 0;
        }
        break;
    }

    // --- UNIFORM scalar_mul_glv: all active lanes execute in lockstep ---
    JacobianPoint local_pt;
    local_pt.infinity = true;
    local_pt.z = FIELD_ONE;

    if (lane <= 18)
        scalar_mul_glv(&my_jac, &my_scalar, &local_pt);
    // lanes 19-31: local_pt stays identity (contributes nothing)

    // Warp tree reduction for combined_base
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        JacobianPoint partner;
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_pt.x.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_pt.x.limbs[w] >> 32), offset);
            partner.x.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_pt.y.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_pt.y.limbs[w] >> 32), offset);
            partner.y.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_pt.z.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_pt.z.limbs[w] >> 32), offset);
            partner.z.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        uint32_t inf_raw = __shfl_down_sync(FULL_MASK, (uint32_t)local_pt.infinity, offset);
        partner.infinity = (bool)inf_raw;

        if (lane + offset < 32)
            jacobian_add(&local_pt, &partner, &local_pt);
    }

    if (lane == 0)
        smem->msm_base = local_pt;
    __syncwarp(FULL_MASK);

    // T3: after Phase 1c
    if (lane == 0) smem->phase_ts[3] = clock64();

    // ================================================================
    // Phase 2: POSITIONAL LUT4 MSM (IDENTICAL to lut4 version)
    // ================================================================
    JacobianPoint local_acc;
    local_acc.infinity = true;
    local_acc.z = FIELD_ONE;

    if (lane < 32) {
        const int base_i = lane * 2;
        for (int off = 0; off < 2 && (base_i + off) < BP_BITS; ++off) {
            const int i = base_i + off;

            Scalar a_si, g_coeff;
            scalar_mul_mod_n(&smem->a_proof, &smem->s_coeff[i], &a_si);
            scalar_sub(&smem->neg_z, &a_si, &g_coeff);

            JacobianPoint g_term;
            scalar_mul_bp_lut4(&g_bp_lut4[i * BP_LUT4_GEN_STRIDE], &g_coeff, &g_term);
            jacobian_add(&local_acc, &g_term, &local_acc);

            // h_coeff = z + y_inv^i * (z2*2^i - b*s_inv[i])
            Scalar z2_2i, b_si, inner, h_inner_yi, h_coeff;
            scalar_mul_mod_n(&smem->z2, &smem->two_powers[i], &z2_2i);
            scalar_mul_mod_n(&smem->b_proof, &smem->s_inv[i], &b_si);
            scalar_sub(&z2_2i, &b_si, &inner);
            scalar_mul_mod_n(&inner, &smem->y_inv_powers[i], &h_inner_yi);
            scalar_add(&smem->z, &h_inner_yi, &h_coeff);

            JacobianPoint h_term;
            scalar_mul_bp_lut4(&g_bp_lut4[(64 + i) * BP_LUT4_GEN_STRIDE], &h_coeff, &h_term);
            jacobian_add(&local_acc, &h_term, &local_acc);
        }
    }

    // T4: after Phase 2 MSM
    if (lane == 0) smem->phase_ts[4] = clock64();

    // Phase 3: Warp tree reduction + final identity check
    #pragma unroll
    for (int offset = 16; offset >= 1; offset >>= 1) {
        JacobianPoint partner;
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.x.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.x.limbs[w] >> 32), offset);
            partner.x.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.y.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.y.limbs[w] >> 32), offset);
            partner.y.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        for (int w = 0; w < 4; ++w) {
            uint32_t lo = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.z.limbs[w]),       offset);
            uint32_t hi = __shfl_down_sync(FULL_MASK, (uint32_t)(local_acc.z.limbs[w] >> 32), offset);
            partner.z.limbs[w] = ((uint64_t)hi << 32) | lo;
        }
        uint32_t inf_raw = __shfl_down_sync(FULL_MASK, (uint32_t)local_acc.infinity, offset);
        partner.infinity = (bool)inf_raw;

        if (lane + offset < 32)
            jacobian_add(&local_acc, &partner, &local_acc);
    }

    if (lane == 0) {
        jacobian_add(&local_acc, &smem->msm_base, &local_acc);
        bool is_id = local_acc.infinity || field_is_zero(&local_acc.z);
        smem->poly_ok = is_id;
        // T5: end
        smem->phase_ts[5] = clock64();
    }
    __syncwarp(FULL_MASK);

    return smem->poly_ok;
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
