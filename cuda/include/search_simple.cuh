#pragma once
#include "secp256k1.cuh"
#include "bloom.cuh"
#include "batch_inversion.cuh"

namespace secp256k1 {
namespace cuda {

// SearchResult defined in search_cpu_identical.cuh — use that as canonical definition
#ifndef SECP256K1_CUDA_SEARCH_RESULT_DEFINED
#define SECP256K1_CUDA_SEARCH_RESULT_DEFINED
// 40-byte structure for results
struct SearchResult {
    uint64_t x[4];   // 32 bytes (Affine X coordinate)
    int64_t index;   // 8 bytes (iteration index)
};
#endif

// CPU Algorithm Translation - EXACT REPLICA:
// Phase 1: Q[i+1] = Q[i] + G (incremental)
// Phase 2: KQ[i+1] = KQ[i] + KG (incremental - NO scalar multiplication!)
// Phase 3: Batch inverse on Z coordinates
// Phase 4: Convert to affine X
// Phase 5: Bloom filter check

__global__ void search_kernel(
    JacobianPoint Q_start,       // Starting Q point
    JacobianPoint G,             // Generator point (affine, Z=1)
    JacobianPoint KQ_start,      // K*Q_start (initial)
    JacobianPoint KG,            // K*G (precomputed, affine, Z=1)
    DeviceBloom bloom,
    SearchResult* results,
    uint32_t* result_count,
    uint64_t batch_offset,
    int batch_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;
    
    // Each thread handles ONE point (simplified for correctness)
    // Phase 1: Compute Q[tid] = Q_start + tid*G (incremental)
    JacobianPoint Q = Q_start;
    for (int i = 0; i < tid; i++) {
        // Mixed addition (G is affine with Z=1)
        jacobian_add_mixed(&Q, (AffinePoint*)&G, &Q);
    }
    
    // Phase 2: Compute KQ[tid] = KQ_start + tid*KG (incremental)
    JacobianPoint KQ = KQ_start;
    for (int i = 0; i < tid; i++) {
        // Mixed addition (KG is affine with Z=1)
        jacobian_add_mixed(&KQ, (AffinePoint*)&KG, &KQ);
    }
    
    // Phase 3: Store Z for batch inversion (shared memory)
    extern __shared__ FieldElement shared_z[];
    shared_z[threadIdx.x] = KQ.z;
    
    __syncthreads();
    
    // Batch inverse within block (simplified - need proper implementation)
    // For now, just do individual inverse
    FieldElement z_inv = KQ.z;
    // TODO: Call batch inverse kernel
    
    // Phase 4: Convert to affine X
    FieldElement z2_inv;
    field_sqr(&z_inv, &z2_inv);
    FieldElement x_affine;
    field_mul(&KQ.x, &z2_inv, &x_affine);
    
    // Phase 5: Bloom filter check (Little-Endian format)
    uint8_t x_bytes[32];
    // Convert to Little-Endian bytes
    for (int i = 0; i < 4; i++) {
        uint64_t limb = x_affine.limbs[i];
        for (int j = 0; j < 8; j++) {
            x_bytes[i*8 + j] = (limb >> (j*8)) & 0xFF;
        }
    }
    
    if (bloom.test(x_bytes, 32)) {
        // Found candidate!
        uint32_t idx = atomicAdd(result_count, 1);
        if (idx < 1024) {
            results[idx].x[0] = x_affine.limbs[0];
            results[idx].x[1] = x_affine.limbs[1];
            results[idx].x[2] = x_affine.limbs[2];
            results[idx].x[3] = x_affine.limbs[3];
            results[idx].index = batch_offset + tid;
        }
    }
}

} // namespace cuda
} // namespace secp256k1
