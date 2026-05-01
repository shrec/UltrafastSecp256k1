#pragma once
// ============================================================================
// Pedersen Commitments -- CUDA device implementation
// ============================================================================
// Batch Pedersen commitment generation on GPU:
//   C_i = v_i * H + r_i * G
//
// Provides:
//   - pedersen_commit_device: single commitment on device
//   - pedersen_commit_batch_kernel: batch kernel for N commitments
//   - pedersen_verify_sum_device: homomorphic sum verification
//
// 64-bit limb mode only.
// ============================================================================

#include "secp256k1.cuh"
#include "ecdsa.cuh"  // SHA256Ctx, sha256_*

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// -- Generator H (nothing-up-my-sleeve) ---------------------------------------
// H = lift_x(SHA256("Pedersen_generator_H"))
// Pre-computed, stored in constant memory.

__device__ __constant__ static const AffinePoint PEDERSEN_H = {
    // Pre-computed x-coordinate of H = lift_x(SHA256("Pedersen_generator_H"))
    // This must match the CPU implementation exactly.
    // Computed at init time by pedersen_init_generators().
    {{0, 0, 0, 0}},
    {{0, 0, 0, 0}}
};

// Runtime-initialized H in global memory (set by host before kernel launch)
__device__ static AffinePoint g_pedersen_H;
__device__ static AffinePoint g_pedersen_J;
__device__ static bool g_pedersen_init = false;

// -- Field sqrt (needed for lift_x) -------------------------------------------

// -- Field sqrt is defined in secp256k1.cuh -- no redefinition needed ---------

// -- lift_x: find point with given x-coordinate and even y --------------------

__device__ inline bool lift_x_even(const FieldElement* x, AffinePoint* out) {
    // y^2 = x^3 + 7
    FieldElement x2, x3, y2, y;
    field_sqr(x, &x2);
    field_mul(&x2, x, &x3);

    FieldElement seven = FIELD_ONE;
    seven.limbs[0] = 7;
    field_add(&x3, &seven, &y2);

    field_sqrt(&y2, &y);

    // Verify: y^2 == y2
    FieldElement check;
    field_sqr(&y, &check);
    field_sub(&check, &y2, &check);

    if (!field_is_zero(&check)) return false;

    // Ensure even y
    if (y.limbs[0] & 1) {
        field_negate(&y, &y);
    }

    out->x = *x;
    out->y = y;
    return true;
}

// -- hash_to_point_increment: try-and-increment for hash-to-curve -------------
// Tries x, x+1, x+2, ... until lift_x_even succeeds.
// Used for nothing-up-my-sleeve generator derivation (NOT for signature verification).

__device__ inline bool hash_to_point_increment(const FieldElement* x_in, AffinePoint* out) {
    FieldElement x = *x_in;
    FieldElement one;
    field_set_one(&one);
    for (int attempt = 0; attempt < 256; ++attempt) {
        if (lift_x_even(&x, out)) return true;
        field_add(&x, &one, &x);
    }
    return false;
}

// -- Single commitment on device ----------------------------------------------

__device__ inline void pedersen_commit_device(
    const Scalar* value,
    const Scalar* blinding,
    const AffinePoint* H,
    JacobianPoint* out)
{
    // C = v*H + r*G
    JacobianPoint vH, rG;

    // v*H (arbitrary point mul)
    JacobianPoint H_jac;
    H_jac.x = H->x;
    H_jac.y = H->y;
    H_jac.z = FIELD_ONE;
    H_jac.infinity = false;

    scalar_mul(&H_jac, value, &vH);

    // r*G (generator mul, fast path with precomputed tables)
    scalar_mul_generator_w8(blinding, &rG);

    // C = vH + rG
    jacobian_add(&vH, &rG, out);
}

// -- Batch Pedersen commitment kernel -----------------------------------------

__global__ void pedersen_commit_batch_kernel(
    const Scalar* __restrict__ values,
    const Scalar* __restrict__ blindings,
    const AffinePoint* __restrict__ H_gen,
    AffinePoint* __restrict__ commitments_out,
    uint32_t count)
{
    uint32_t const idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    JacobianPoint result;
    pedersen_commit_device(&values[idx], &blindings[idx], H_gen, &result);

    // Convert Jacobian to affine
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv(&result.z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv2, &z_inv, &z_inv3);

    field_mul(&result.x, &z_inv2, &commitments_out[idx].x);
    field_mul(&result.y, &z_inv3, &commitments_out[idx].y);
}

// -- Batch Pedersen verify sum kernel -----------------------------------------
// Checks that sum(pos[i]) - sum(neg[j]) == point-at-infinity

__global__ void pedersen_verify_sum_kernel(
    const AffinePoint* __restrict__ pos,
    uint32_t n_pos,
    const AffinePoint* __restrict__ neg,
    uint32_t n_neg,
    bool* __restrict__ result)
{
    // Single-thread kernel for sum verification
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    JacobianPoint sum;
    sum.x = FIELD_ONE;
    sum.y = FIELD_ONE;
    sum.z = {{0, 0, 0, 0}};
    sum.infinity = true;

    for (uint32_t i = 0; i < n_pos; ++i) {
        jacobian_add_mixed(&sum, &pos[i], &sum);
    }

    for (uint32_t i = 0; i < n_neg; ++i) {
        AffinePoint neg_pt = neg[i];
        // Negate Y coordinate
        field_negate(&neg_pt.y, &neg_pt.y);
        jacobian_add_mixed(&sum, &neg_pt, &sum);
    }

    // Check if sum is infinity
    *result = (sum.infinity || field_is_zero(&sum.z));
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
