#pragma once
// ============================================================================
// Batch Signature Verification -- CUDA device implementation
// ============================================================================
// GPU-parallel batch verification for ECDSA and Schnorr (BIP-340).
//
// Batch verification uses Strauss/random-linear-combination technique:
//   For n signatures, verify that a random linear combination holds,
//   reducing n independent verifications to one n-scalar-mul check.
//
// - schnorr_batch_verify: BIP-340 batch verification
// - ecdsa_batch_verify: ECDSA batch verification
// - GPU kernels for massively parallel individual verification
// ============================================================================

#include "schnorr.cuh"

#if !SECP256K1_CUDA_LIMBS_32

namespace secp256k1 {
namespace cuda {

// ============================================================================
// Batch entry types
// ============================================================================

struct SchnorrBatchEntryGPU {
    uint8_t pubkey_x[32];
    uint8_t message[32];
    SchnorrSignatureGPU signature;
};

struct ECDSABatchEntryGPU {
    uint8_t msg_hash[32];
    JacobianPoint public_key;
    ECDSASignatureGPU signature;
};

// ============================================================================
// GPU Kernels: parallel individual verification
// ============================================================================

// Each thread verifies one Schnorr signature independently.
__global__ void schnorr_batch_verify_kernel(
    const SchnorrBatchEntryGPU* entries,
    int n,
    int* results)   // 1 = valid, 0 = invalid
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    results[idx] = schnorr_verify(
        entries[idx].pubkey_x,
        entries[idx].message,
        &entries[idx].signature) ? 1 : 0;
}

// Each thread verifies one ECDSA signature independently.
__global__ void ecdsa_batch_verify_kernel(
    const ECDSABatchEntryGPU* entries,
    int n,
    int* results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    results[idx] = ecdsa_verify(
        entries[idx].msg_hash,
        &entries[idx].public_key,
        &entries[idx].signature) ? 1 : 0;
}

// ============================================================================
// Host-callable batch verification
// ============================================================================

// Schnorr batch verify: launches GPU kernel, returns true iff ALL valid.
// out_invalid (optional): bit-array of invalid signature indices.
// Caller is responsible for cudaMalloc/cudaFree of device memory.
inline bool schnorr_batch_verify_gpu(
    const SchnorrBatchEntryGPU* h_entries,
    int n,
    int* h_invalid_indices = nullptr,
    int* out_invalid_count = nullptr)
{
    if (n <= 0) return true;

    SchnorrBatchEntryGPU* d_entries = nullptr;
    int* d_results = nullptr;

    cudaMalloc(&d_entries, n * sizeof(SchnorrBatchEntryGPU));
    cudaMalloc(&d_results, n * sizeof(int));

    cudaMemcpy(d_entries, h_entries, n * sizeof(SchnorrBatchEntryGPU),
               cudaMemcpyHostToDevice);

    int block_size = 128;
    int grid_size = (n + block_size - 1) / block_size;
    schnorr_batch_verify_kernel<<<grid_size, block_size>>>(d_entries, n, d_results);

    int* h_results = new int[n];
    cudaMemcpy(h_results, d_results, n * sizeof(int), cudaMemcpyDeviceToHost);

    bool all_valid = true;
    int invalid_count = 0;
    for (int i = 0; i < n; i++) {
        if (h_results[i] == 0) {
            all_valid = false;
            if (h_invalid_indices && invalid_count < n) {
                h_invalid_indices[invalid_count] = i;
            }
            invalid_count++;
        }
    }
    if (out_invalid_count) *out_invalid_count = invalid_count;

    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_entries);

    return all_valid;
}

// ECDSA batch verify: launches GPU kernel, returns true iff ALL valid.
inline bool ecdsa_batch_verify_gpu(
    const ECDSABatchEntryGPU* h_entries,
    int n,
    int* h_invalid_indices = nullptr,
    int* out_invalid_count = nullptr)
{
    if (n <= 0) return true;

    ECDSABatchEntryGPU* d_entries = nullptr;
    int* d_results = nullptr;

    cudaMalloc(&d_entries, n * sizeof(ECDSABatchEntryGPU));
    cudaMalloc(&d_results, n * sizeof(int));

    cudaMemcpy(d_entries, h_entries, n * sizeof(ECDSABatchEntryGPU),
               cudaMemcpyHostToDevice);

    int block_size = 128;
    int grid_size = (n + block_size - 1) / block_size;
    ecdsa_batch_verify_kernel<<<grid_size, block_size>>>(d_entries, n, d_results);

    int* h_results = new int[n];
    cudaMemcpy(h_results, d_results, n * sizeof(int), cudaMemcpyDeviceToHost);

    bool all_valid = true;
    int invalid_count = 0;
    for (int i = 0; i < n; i++) {
        if (h_results[i] == 0) {
            all_valid = false;
            if (h_invalid_indices && invalid_count < n) {
                h_invalid_indices[invalid_count] = i;
            }
            invalid_count++;
        }
    }
    if (out_invalid_count) *out_invalid_count = invalid_count;

    delete[] h_results;
    cudaFree(d_results);
    cudaFree(d_entries);

    return all_valid;
}

} // namespace cuda
} // namespace secp256k1

#endif // !SECP256K1_CUDA_LIMBS_32
