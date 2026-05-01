// ============================================================================
// GPU Compatibility Layer -- CUDA <-> HIP (ROCm) Runtime API Mapping
// ============================================================================
// Allows the same .cu source to compile with both NVIDIA CUDA and AMD ROCm.
//
// When compiling with HIP (hipcc / __HIP_PLATFORM_AMD__):
//   - CUDA runtime calls are mapped to HIP equivalents
//   - PTX inline asm is replaced with portable __int128 math
//   - __constant__ memory, atomics, __syncthreads() work natively
//
// When compiling with CUDA (nvcc / __CUDA_ARCH__):
//   - This header is a no-op passthrough
//   - Native PTX inline asm is used for maximum performance
//
// Usage: #include "gpu_compat.h" at the top of every .cu/.cuh file
// ============================================================================

#pragma once

// -- Platform Detection ------------------------------------------------------
#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
    #define SECP256K1_GPU_HIP 1
    #define SECP256K1_GPU_CUDA 0
    #define SECP256K1_USE_PTX 0
    #include <hip/hip_runtime.h>
#else
    #define SECP256K1_GPU_HIP 0
    #define SECP256K1_GPU_CUDA 1
    #define SECP256K1_USE_PTX 1
    #include <cuda_runtime.h>
#endif

// -- Runtime API Mapping (CUDA -> HIP) ---------------------------------------
#if SECP256K1_GPU_HIP

// Memory management
#define cudaMalloc              hipMalloc
#define cudaFree                hipFree
#define cudaMemcpy              hipMemcpy
#define cudaMemset              hipMemset
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice

// Synchronization
#define cudaDeviceSynchronize   hipDeviceSynchronize

// Events (timing)
#define cudaEvent_t             hipEvent_t
#define cudaEventCreate         hipEventCreate
#define cudaEventRecord         hipEventRecord
#define cudaEventSynchronize    hipEventSynchronize
#define cudaEventElapsedTime    hipEventElapsedTime
#define cudaEventDestroy        hipEventDestroy

// Error handling
#define cudaError_t             hipError_t
#define cudaSuccess             hipSuccess
#define cudaGetLastError        hipGetLastError
#define cudaGetErrorString      hipGetErrorString

// Device management
#define cudaSetDevice           hipSetDevice
#define cudaGetDevice           hipGetDevice
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceProp          hipDeviceProp_t

// Stream
#define cudaStream_t            hipStream_t
#define cudaStreamCreate        hipStreamCreate
#define cudaStreamDestroy       hipStreamDestroy
#define cudaStreamSynchronize   hipStreamSynchronize

// Launch configuration
#define cudaOccupancyMaxPotentialBlockSize hipOccupancyMaxPotentialBlockSize

#endif // SECP256K1_GPU_HIP
