#include "secp256k1.cuh"
#include "ecdsa.cuh"
#include "schnorr.cuh"
#include "recovery.cuh"
#include "ct/ct_sign.cuh"

namespace secp256k1 {
namespace cuda {

// Field operation kernels -- lightweight, high-occupancy targets.
// 256 threads/block, min 4 blocks/SM for register pressure balance.

__global__ __launch_bounds__(256, 4)
void field_mul_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        field_mul(&a[idx], &b[idx], &r[idx]);
    }
}

__global__ __launch_bounds__(256, 4)
void field_add_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        field_add(&a[idx], &b[idx], &r[idx]);
    }
}

__global__ __launch_bounds__(256, 4)
void field_sub_kernel(const FieldElement* a, const FieldElement* b, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        field_sub(&a[idx], &b[idx], &r[idx]);
    }
}

__global__ __launch_bounds__(256, 4)
void field_inv_kernel(const FieldElement* a, FieldElement* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        field_inv(&a[idx], &r[idx]);
    }
}

// Scalar multiplication kernels -- register-heavy, lower occupancy acceptable.
// 128 threads/block, min 2 blocks/SM to balance register pressure vs. latency hiding.

__global__ __launch_bounds__(128, 2)
void scalar_mul_batch_kernel(const JacobianPoint* points, const Scalar* scalars, 
                                         JacobianPoint* results, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        scalar_mul(&points[idx], &scalars[idx], &results[idx]);
    }
}

__global__ __launch_bounds__(128, 2)
void generator_mul_batch_kernel(const Scalar* scalars, JacobianPoint* results, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        // Compute G * scalar using pre-loaded constant generator (reduces per-thread setup)
        scalar_mul(&GENERATOR_JACOBIAN, &scalars[idx], &results[idx]);
    }
}

#ifndef SECP256K1_CUDA_LIMBS_32
// Windowed generator multiplication kernel (w=4, shared-memory precomputed table)
// Table[0..15] = i*G is built once per block by thread 0, then reused by all threads.
// ~30-40% faster than plain double-and-add.
__global__ __launch_bounds__(128, 2)
void generator_mul_windowed_batch_kernel(const Scalar* scalars, JacobianPoint* results, int count) {
    __shared__ JacobianPoint gen_table[16];  // ~1.6 KB shared memory

    if (threadIdx.x == 0) {
        build_generator_table(gen_table);
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        scalar_mul_generator_windowed(gen_table, &scalars[idx], &results[idx]);
    }
}
#endif // !SECP256K1_CUDA_LIMBS_32

__global__ __launch_bounds__(256, 4)
void point_add_kernel(const JacobianPoint* a, const JacobianPoint* b, JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        jacobian_add(&a[idx], &b[idx], &r[idx]);
    }
}

__global__ __launch_bounds__(256, 4)
void point_dbl_kernel(const JacobianPoint* a, JacobianPoint* r, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        jacobian_double(&a[idx], &r[idx]);
    }
}

__global__ __launch_bounds__(256, 4)
void hash160_pubkey_kernel(const uint8_t* pubkeys, int pubkey_len, uint8_t* out_hashes, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const uint8_t* pk = pubkeys + static_cast<size_t>(idx) * static_cast<size_t>(pubkey_len);
        uint8_t* out = out_hashes + static_cast<size_t>(idx) * 20U;
        hash160_pubkey(pk, static_cast<size_t>(pubkey_len), out);
    }
}

// ============================================================================
// ECDSA / Schnorr batch kernels (64-bit limb mode only)
// ============================================================================
#if !SECP256K1_CUDA_LIMBS_32

// ECDSA Sign batch -- uses ct_ecdsa_sign (explicit CT path) for benchmark + audit.
// Production signing goes through the CPU CT path (ufsecp_ecdsa_sign_batch → ct::ecdsa_sign).
__global__ __launch_bounds__(128, 2)
void ecdsa_sign_batch_kernel(
    const uint8_t* __restrict__ msg_hashes,   // count x 32 bytes
    const Scalar*  __restrict__ private_keys,
    ECDSASignatureGPU* __restrict__ sigs,
    bool*          __restrict__ results,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const uint8_t* msg = msg_hashes + static_cast<size_t>(idx) * 32;
        sigs[idx] = {};  // zero-init before sign so failure leaves no stale data
        results[idx] = ct_ecdsa_sign(msg, &private_keys[idx], &sigs[idx]);
    }
}

// ECDSA Verify batch -- each thread verifies one signature
__global__ __launch_bounds__(128, 2)
void ecdsa_verify_batch_kernel(
    const uint8_t* __restrict__ msg_hashes,
    const JacobianPoint* __restrict__ public_keys,
    const ECDSASignatureGPU* __restrict__ sigs,
    bool*          __restrict__ results,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const uint8_t* msg = msg_hashes + static_cast<size_t>(idx) * 32;
        results[idx] = ecdsa_verify(msg, &public_keys[idx], &sigs[idx]);
    }
}

// ECDSA SNARK witness batch (eprint 2025/695) -- each thread fills one witness
__global__ __launch_bounds__(128, 2)
void ecdsa_snark_witness_batch_kernel(
    const uint8_t*            __restrict__ msg_hashes,
    const JacobianPoint*      __restrict__ public_keys,
    const ECDSASignatureGPU*  __restrict__ sigs,
    EcdsaSnarkWitnessFlat*    __restrict__ out,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        ecdsa_snark_witness_device(
            msg_hashes + (size_t)idx * 32,
            &public_keys[idx],
            &sigs[idx],
            &out[idx]);
    }
}

// Schnorr SNARK witness batch (eprint 2025/695) -- each thread fills one witness
__global__ __launch_bounds__(128, 2)
void schnorr_snark_witness_batch_kernel(
    const uint8_t*             __restrict__ msgs32,
    const uint8_t*             __restrict__ pubkeys_x32,
    const uint8_t*             __restrict__ sigs64,
    SchnorrSnarkWitnessFlat*   __restrict__ out,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        schnorr_snark_witness_device(
            msgs32       + (size_t)idx * 32,
            pubkeys_x32  + (size_t)idx * 32,
            sigs64       + (size_t)idx * 64,
            &out[idx]);
    }
}

// Schnorr Sign batch -- each thread signs one message
__global__ __launch_bounds__(128, 2)
void schnorr_sign_batch_kernel(
    const Scalar*  __restrict__ private_keys,
    const uint8_t* __restrict__ msgs,         // count x 32 bytes
    const uint8_t* __restrict__ aux_rands,    // count x 32 bytes
    SchnorrSignatureGPU* __restrict__ sigs,
    bool*          __restrict__ results,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const uint8_t* msg = msgs + static_cast<size_t>(idx) * 32;
        const uint8_t* aux = aux_rands + static_cast<size_t>(idx) * 32;
        sigs[idx] = {};  // zero-init before sign so failure leaves no stale data
        results[idx] = ct_schnorr_sign(&private_keys[idx], msg, aux, &sigs[idx]);
    }
}

// Schnorr Verify batch -- each thread verifies one signature
__global__ __launch_bounds__(128, 2)
void schnorr_verify_batch_kernel(
    const uint8_t* __restrict__ pubkeys_x,    // count x 32 bytes (x-only)
    const uint8_t* __restrict__ msgs,
    const SchnorrSignatureGPU* __restrict__ sigs,
    bool*          __restrict__ results,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const uint8_t* pk  = pubkeys_x + static_cast<size_t>(idx) * 32;
        const uint8_t* msg = msgs + static_cast<size_t>(idx) * 32;
        results[idx] = schnorr_verify(pk, msg, &sigs[idx]);
    }
}

// ECDSA Sign Recoverable batch
__global__ __launch_bounds__(128, 2)
void ecdsa_sign_recoverable_batch_kernel(
    const uint8_t* __restrict__ msg_hashes,
    const Scalar*  __restrict__ private_keys,
    RecoverableSignatureGPU* __restrict__ rsigs,
    bool*          __restrict__ results,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const uint8_t* msg = msg_hashes + static_cast<size_t>(idx) * 32;
        results[idx] = ecdsa_sign_recoverable(msg, &private_keys[idx], &rsigs[idx]);
    }
}

// ECDSA Recover batch
__global__ __launch_bounds__(128, 2)
void ecdsa_recover_batch_kernel(
    const uint8_t* __restrict__ msg_hashes,
    const ECDSASignatureGPU* __restrict__ sigs,
    const int*     __restrict__ recids,
    JacobianPoint* __restrict__ recovered_keys,
    bool*          __restrict__ results,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        const uint8_t* msg = msg_hashes + static_cast<size_t>(idx) * 32;
        results[idx] = ecdsa_recover(msg, &sigs[idx], recids[idx], &recovered_keys[idx]);
    }
}

#endif // !SECP256K1_CUDA_LIMBS_32

// ============================================================================
// FROST Partial Signature Verification batch kernel
// ============================================================================
#ifndef SECP256K1_CUDA_LIMBS_32
__global__ __launch_bounds__(128, 2)
void frost_verify_partial_batch_kernel(
    const uint8_t* __restrict__ z_i_bytes,
    const uint8_t* __restrict__ D_i_bytes,
    const uint8_t* __restrict__ E_i_bytes,
    const uint8_t* __restrict__ Y_i_bytes,
    const uint8_t* __restrict__ rho_i_bytes,
    const uint8_t* __restrict__ lambda_i_e_bytes,
    const uint8_t* __restrict__ negate_R,
    const uint8_t* __restrict__ negate_key,
    uint8_t*       __restrict__ results,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    results[idx] = frost_verify_partial_device(
        z_i_bytes        + idx * 32,
        D_i_bytes        + idx * 33,
        E_i_bytes        + idx * 33,
        Y_i_bytes        + idx * 33,
        rho_i_bytes      + idx * 32,
        lambda_i_e_bytes + idx * 32,
        negate_R [idx],
        negate_key[idx]);
}

// ============================================================================
// Batch Jacobian -> Compressed (33-byte SEC1) batch kernel
// ============================================================================
__global__ __launch_bounds__(128, 2)
void batch_jacobian_to_compressed_kernel(
    const JacobianPoint* __restrict__ points,
    uint8_t*             __restrict__ out33,
    int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    jacobian_to_compressed_device(&points[idx], out33 + idx * 33);
}
#endif // !SECP256K1_CUDA_LIMBS_32

} // namespace cuda
} // namespace secp256k1
