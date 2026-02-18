#include "secp256k1.cuh"

namespace secp256k1 {
namespace cuda {

// Field operation kernels — lightweight, high-occupancy targets.
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

// Scalar multiplication kernels — register-heavy, lower occupancy acceptable.
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

} // namespace cuda
} // namespace secp256k1
