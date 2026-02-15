#pragma once
#include "secp256k1.cuh"
#include "bloom.cuh"
#include "batch_inversion.cuh"

namespace secp256k1 {
namespace cuda {

// Constant memory for K*G and -K*G (CPU-identical algorithm)
__constant__ AffinePoint c_kg_affine;       // K*G for forward direction
__constant__ AffinePoint c_neg_kg_affine;   // -K*G for backward direction

#ifndef SECP256K1_CUDA_SEARCH_RESULT_DEFINED
#define SECP256K1_CUDA_SEARCH_RESULT_DEFINED
struct SearchResult {
    uint64_t x[4];
    int64_t index;
};
#endif

// CPU-IDENTICAL Algorithm: Incremental Addition (like CPU megabatch)
// Each batch iteration does: P[i+1] = P[i] + K*G (one addition per iteration)
__global__ void incremental_add_batch_kernel(
    JacobianPoint* points,
    int batch_size,
    int direction  // 1 = forward (+K*G), -1 = backward (-K*G)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    JacobianPoint P = points[idx];
    
    const AffinePoint* step = (direction == 1) ? &c_kg_affine : &c_neg_kg_affine;
    jacobian_add_mixed(&P, step, &P);
    
    points[idx] = P;
}

// Initialize batch: P[i] = Q_div (all threads start from same point)
__global__ void init_batch_kernel(
    JacobianPoint* points,
    const JacobianPoint Q_div,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    points[idx] = Q_div;
}

// Extract Z coordinates for batch inversion
__global__ void extract_z_kernel(
    const JacobianPoint* points,
    FieldElement* zs,
    int batch_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    zs[idx] = points[idx].z;
}

// Convert to affine and check Bloom filter
__global__ void affine_and_bloom_kernel(
    const JacobianPoint* points,
    const FieldElement* inv_zs,
    const DeviceBloom filter,
    SearchResult* results,
    uint32_t* result_count,
    int batch_size,
    uint64_t batch_offset
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    const JacobianPoint& P = points[idx];
    if (P.infinity) return;
    
    // Convert to affine: x_aff = x_jac * z_inv^2
    FieldElement z_inv = inv_zs[idx];
    FieldElement z2;
    field_sqr(&z_inv, &z2);
    
    FieldElement x_aff;
    field_mul(&P.x, &z2, &x_aff);
    
    // Prepare for Bloom filter (Little-Endian limbs)
    uint8_t key_bytes[32];
    for (int i = 0; i < 4; i++) {
        uint64_t limb = x_aff.limbs[i];
        for (int j = 0; j < 8; j++) {
            key_bytes[i * 8 + j] = (uint8_t)(limb >> (j * 8));
        }
    }
    
    // Bloom filter test
    if (filter.test(key_bytes, 32)) {
        uint32_t slot = atomicAdd(result_count, 1);
        if (slot < 1024) {
            results[slot].x[0] = x_aff.limbs[0];
            results[slot].x[1] = x_aff.limbs[1];
            results[slot].x[2] = x_aff.limbs[2];
            results[slot].x[3] = x_aff.limbs[3];
            results[slot].index = batch_offset + idx;
        }
    }
}

// CPU-identical search runner
inline void run_cpu_identical_search(
    JacobianPoint* d_points_fwd,
    JacobianPoint* d_points_bwd,
    FieldElement* d_zs,
    FieldElement* d_inv_zs,
    SearchResult* d_results,
    uint32_t* d_result_count,
    const DeviceBloom& filter,
    const JacobianPoint& Q_div,
    int batch_size,
    int num_batches
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;
    size_t shared_mem = 2 * threads * sizeof(FieldElement);
    
    // Initialize: all threads start from Q_div
    init_batch_kernel<<<blocks, threads>>>(d_points_fwd, Q_div, batch_size);
    init_batch_kernel<<<blocks, threads>>>(d_points_bwd, Q_div, batch_size);
    cudaDeviceSynchronize();
    
    for (int batch = 0; batch < num_batches; batch++) {
        // CPU-style: loop batch_size times, each time P[i] = P[i] + K*G
        for (int iter = 0; iter < batch_size; iter++) {
            // Forward: P = P + K*G
            incremental_add_batch_kernel<<<blocks, threads>>>(d_points_fwd, batch_size, 1);
            
            // Backward: P = P - K*G
            incremental_add_batch_kernel<<<blocks, threads>>>(d_points_bwd, batch_size, -1);
        }
        cudaDeviceSynchronize();
        
        // Extract Z coordinates
        extract_z_kernel<<<blocks, threads>>>(d_points_fwd, d_zs, batch_size);
        cudaDeviceSynchronize();
        
        // Batch inverse
        batch_inverse_kernel<<<blocks, threads, shared_mem>>>(d_zs, d_inv_zs, batch_size);
        cudaDeviceSynchronize();
        
        // Check forward
        cudaMemset(d_result_count, 0, sizeof(uint32_t));
        affine_and_bloom_kernel<<<blocks, threads>>>(
            d_points_fwd, d_inv_zs, filter, d_results, d_result_count, 
            batch_size, (uint64_t)batch * batch_size
        );
        cudaDeviceSynchronize();
        
        // Extract Z for backward
        extract_z_kernel<<<blocks, threads>>>(d_points_bwd, d_zs, batch_size);
        cudaDeviceSynchronize();
        
        // Batch inverse
        batch_inverse_kernel<<<blocks, threads, shared_mem>>>(d_zs, d_inv_zs, batch_size);
        cudaDeviceSynchronize();
        
        // Check backward
        affine_and_bloom_kernel<<<blocks, threads>>>(
            d_points_bwd, d_inv_zs, filter, d_results, d_result_count,
            batch_size, (uint64_t)batch * batch_size
        );
        cudaDeviceSynchronize();
    }
}

} // namespace cuda
} // namespace secp256k1
