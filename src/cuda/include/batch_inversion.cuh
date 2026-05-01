#pragma once
#include "secp256k1.cuh"

namespace secp256k1 {
namespace cuda {

// ===================================================================
// HELPER FUNCTIONS
// ===================================================================

// Optimal addition chain for Secp256k1 modular inverse (P-2)
// P = 2^256 - 2^32 - 977
// P-2 = 2^256 - 2^32 - 979
// Chain length: 281 squarings, 19 multiplications
// Optimized for register usage (6 temporaries + t)
__device__ inline void field_inv_fermat_chain(const FieldElement* a, FieldElement* r) {
    FieldElement x_0, x_1, x_2, x_3, x_4, x_5;
    FieldElement t;

    // 1. x2 = x^2 * x (2 ones) -> x_0
    field_sqr(a, &x_0);
    field_mul(&x_0, a, &x_0);

    // 2. x3 = x2^2 * x (3 ones) -> x_1
    field_sqr(&x_0, &x_1);
    field_mul(&x_1, a, &x_1);

    // 3. x6 = x3^(2^3) * x3 (6 ones) -> x_2
    field_sqr(&x_1, &x_2);
    field_sqr(&x_2, &x_2);
    field_sqr(&x_2, &x_2);
    field_mul(&x_2, &x_1, &x_2);

    // 4. x12 = x6^(2^6) * x6 (12 ones) -> x_3
    field_sqr(&x_2, &x_3);
    field_sqr_n(&x_3, 5);
    field_mul(&x_3, &x_2, &x_3);

    // 5. x24 = x12^(2^12) * x12 (24 ones) -> x_3 (reuse)
    field_sqr(&x_3, &t); // Use t as temp for squaring start
    // Original: field_sqr(&x12, &x24); for(11) sqr; mul.
    // Here x_3 is x12. We want x24 in x_3.
    // We can't overwrite x_3 immediately if we need it for mul.
    // So compute into t, then mul x_3 into t, then store in x_3.
    // Wait, x12 is x_3.
    // x24 = x12^(2^12) * x12.
    // t = x_3.
    // for 12 times: sqr(t).
    // t = t * x_3.
    // x_3 = t.
    t = x_3;
    field_sqr_n(&t, 12);
    field_mul(&t, &x_3, &x_3);

    // 6. x48 = x24^(2^24) * x24 (48 ones) -> x_4
    // x_3 is x24.
    t = x_3;
    field_sqr_n(&t, 24);
    field_mul(&t, &x_3, &x_4);

    // 7. x96 = x48^(2^48) * x48 (96 ones) -> x_4 (reuse)
    // x_4 is x48.
    t = x_4;
    field_sqr_n(&t, 48);
    field_mul(&t, &x_4, &x_4);

    // 8. x192 = x96^(2^96) * x96 (192 ones) -> x_4 (reuse)
    // x_4 is x96.
    t = x_4;
    field_sqr_n(&t, 96);
    field_mul(&t, &x_4, &x_4);

    // 9. x7 = x6^2 * x (7 ones) -> x_5
    // x_2 is x6.
    field_sqr(&x_2, &x_5);
    field_mul(&x_5, a, &x_5);

    // 10. x31 = x24^(2^7) * x7 (31 ones) -> x_5 (reuse)
    // x_3 is x24. x_5 is x7.
    // We need to compute x24^(2^7) * x7.
    // Store in x_5.
    t = x_3;
    field_sqr_n(&t, 7);
    field_mul(&t, &x_5, &x_5);

    // 11. x223 = x192^(2^31) * x31 (223 ones) -> x_5 (reuse)
    // x_4 is x192. x_5 is x31.
    t = x_4;
    field_sqr_n(&t, 31);
    field_mul(&t, &x_5, &x_5);

    // 12. x5 = x3^(2^2) * x2 (5 ones) -> x_0 (reuse x2 slot)
    // x_1 is x3. x_0 is x2.
    // We need x3^(2^2) * x2.
    // Store in x_0.
    t = x_1;
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_mul(&t, &x_0, &x_0); // x_0 is now x5

    // 13. x11 = x6^(2^5) * x5 (11 ones) -> x_1 (reuse x3 slot)
    // x_2 is x6. x_0 is x5.
    t = x_2;
    field_sqr_n(&t, 5);
    field_mul(&t, &x_0, &x_1); // x_1 is now x11

    // 14. x22 = x11^(2^11) * x11 (22 ones) -> x_1 (reuse)
    // x_1 is x11.
    t = x_1;
    field_sqr_n(&t, 11);
    field_mul(&t, &x_1, &x_1); // x_1 is now x22

    // 15. t = x223^2 (bit 32 is 0)
    // x_5 is x223.
    field_sqr(&x_5, &t);

    // 16. t = t^(2^22) * x22 (append 22 ones)
    // x_1 is x22.
    field_sqr_n(&t, 22);
    field_mul(&t, &x_1, &t);

    // 17. t = t^(2^4) (bits 9,8,7,6 are 0)
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);
    field_sqr(&t, &t);

    // 18. Process remaining 6 bits: 101101
    // bit 5: 1
    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    // bit 4: 0
    field_sqr(&t, &t);
    // bit 3: 1
    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    // bit 2: 1
    field_sqr(&t, &t);
    field_mul(&t, a, &t);
    // bit 1: 0
    field_sqr(&t, &t);
    // bit 0: 1
    field_sqr(&t, &t);
    field_mul(&t, a, r);
}

// Hillis-Steele Parallel Prefix Scan (Inclusive)
__device__ inline void block_prefix_mul(const FieldElement* in, FieldElement* out, int n) {
    int tid = threadIdx.x;
    
    if (tid < n) {
        out[tid] = in[tid];
    }
    __syncthreads();

    FieldElement val;
    for (int offset = 1; offset < n; offset *= 2) {
        if (tid >= offset && tid < n) {
            field_mul(&out[tid - offset], &out[tid], &val);
        }
        __syncthreads();
        if (tid >= offset && tid < n) {
            out[tid] = val;
        }
        __syncthreads();
    }
}

// Hillis-Steele Parallel Suffix Scan (Inclusive)
__device__ inline void block_suffix_mul(const FieldElement* in, FieldElement* out, int n) {
    int tid = threadIdx.x;
    
    if (tid < n) {
        out[tid] = in[tid];
    }
    __syncthreads();

    FieldElement val;
    for (int offset = 1; offset < n; offset *= 2) {
        if (tid + offset < n) {
            field_mul(&out[tid + offset], &out[tid], &val);
        }
        __syncthreads();
        if (tid + offset < n) {
            out[tid] = val;
        }
        __syncthreads();
    }
}

// ===================================================================
// METHOD 1: Montgomery Batch Trick (prefix/suffix scan)
// ===================================================================
__global__ void batch_inverse_montgomery(const FieldElement* input, FieldElement* output, int count) {
    extern __shared__ FieldElement shared_mem[];
    FieldElement* L = shared_mem;
    FieldElement* R = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    FieldElement x;
    if (idx < count) {
        x = input[idx];
    }

    int valid_in_block = count - blockIdx.x * blockDim.x;
    if (valid_in_block > blockDim.x) valid_in_block = blockDim.x;
    
    if (idx < count) {
        L[tid] = x;
        R[tid] = x;
    }
    __syncthreads();

    block_prefix_mul(L, L, valid_in_block);
    block_suffix_mul(R, R, valid_in_block);
    
    FieldElement total_prod;
    if (valid_in_block > 0) {
        total_prod = L[valid_in_block - 1];
    }
    
    __shared__ FieldElement total_inv;
    if (tid == 0 && valid_in_block > 0) {
        field_inv(&total_prod, &total_inv);
    }
    __syncthreads();
    
    if (idx < count) {
        FieldElement res = total_inv;
        
        if (tid > 0) {
            field_mul(&res, &L[tid-1], &res);
        }
        
        if (tid < valid_in_block - 1) {
            field_mul(&res, &R[tid+1], &res);
        }
        
        output[idx] = res;
    }
}

// ===================================================================
// METHOD 2: Fermat's Little Theorem (branch-free)
// ===================================================================
__global__ void batch_inverse_fermat(const FieldElement* input, FieldElement* output, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    field_inv_fermat_chain(&input[idx], &output[idx]);
}

// ===================================================================
// METHOD 3: Naive per-element
// ===================================================================
__global__ void batch_inverse_naive(const FieldElement* input, FieldElement* output, int count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    if (!field_is_zero(&input[idx])) {
        field_inv(&input[idx], &output[idx]);
    } else {
        output[idx].limbs[0] = 0;
        output[idx].limbs[1] = 0;
        output[idx].limbs[2] = 0;
        output[idx].limbs[3] = 0;
    }
}

// ===================================================================
// DEFAULT KERNEL - Montgomery (proven working)
// ===================================================================
// Do not force 4-block residency here; the Fermat inversion path can need
// close to the full per-thread register budget on current CUDA toolchains.
__global__ __launch_bounds__(256) void batch_inverse_kernel(const FieldElement* input, FieldElement* output, int count) {
    extern __shared__ FieldElement shared_mem[];
    FieldElement* L = shared_mem;
    FieldElement* R = shared_mem + blockDim.x;

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    FieldElement x;
    if (idx < count) {
        x = input[idx];
    }

    int valid_in_block = count - blockIdx.x * blockDim.x;
    if (valid_in_block > blockDim.x) valid_in_block = blockDim.x;
    
    if (idx < count) {
        L[tid] = x;
        R[tid] = x;
    }
    __syncthreads();

    block_prefix_mul(L, L, valid_in_block);
    block_suffix_mul(R, R, valid_in_block);
    
    FieldElement total_prod;
    if (valid_in_block > 0) {
        total_prod = L[valid_in_block - 1];
    }
    
    __shared__ FieldElement total_inv;
    if (tid == 0 && valid_in_block > 0) {
        field_inv(&total_prod, &total_inv);
    }
    __syncthreads();
    
    if (idx < count) {
        FieldElement res = total_inv;
        
        if (tid > 0) {
            field_mul(&res, &L[tid-1], &res);
        }
        
        if (tid < valid_in_block - 1) {
            field_mul(&res, &R[tid+1], &res);
        }
        
        output[idx] = res;
    }
}

} // namespace cuda
} // namespace secp256k1
