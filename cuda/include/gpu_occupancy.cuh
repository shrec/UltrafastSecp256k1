#pragma once
// ============================================================================
// gpu_occupancy.cuh -- CUDA Occupancy Auto-Tuning Utilities
// ============================================================================
// Provides optimal launch configuration helpers that use the CUDA occupancy
// API to maximize SM utilization. Eliminates manual block-size guessing.
//
// Usage:
//   auto [grid, block] = optimal_launch_1d(my_kernel, count);
//   my_kernel<<<grid, block>>>(args...);
//
// Or with shared memory:
//   auto [grid, block] = optimal_launch_1d(my_kernel, count, smem_per_block);
// ============================================================================

#include "gpu_compat.h"
#include <cstdio>
#include <utility>

namespace secp256k1 {
namespace cuda {

// -- Optimal 1D launch configuration --------------------------------------

/// Compute optimal (grid, block) for a 1D kernel launch.
/// Uses cudaOccupancyMaxPotentialBlockSize to find the block size that
/// maximizes occupancy, then derives grid size from element count.
///
/// @param kernel       Device function pointer
/// @param count        Total number of elements to process
/// @param smem_bytes   Dynamic shared memory per block (default: 0)
/// @return             {gridDim, blockDim} suitable for kernel<<<grid, block>>>
template <typename KernelFunc>
__host__ inline std::pair<dim3, dim3> optimal_launch_1d(
    KernelFunc kernel,
    int count,
    size_t smem_bytes = 0)
{
    int min_grid_size = 0;
    int block_size = 256;  // fallback

#ifdef __CUDACC__
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                       kernel,
                                       smem_bytes, count);
#elif defined(__HIP_PLATFORM_AMD__)
    hipOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
                                      kernel,
                                      smem_bytes, count);
#endif

    int grid_size = (count + block_size - 1) / block_size;
    return {dim3(grid_size), dim3(block_size)};
}

// -- Query achievable occupancy -------------------------------------------

/// Query how many blocks of a given kernel can run concurrently per SM.
/// Useful for diagnostic/observability prints at startup.
template <typename KernelFunc>
__host__ inline int query_occupancy(
    KernelFunc kernel,
    int block_size,
    size_t smem_bytes = 0)
{
    int active_blocks = 0;
#ifdef __CUDACC__
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                   kernel,
                                                   block_size,
                                                   smem_bytes);
#elif defined(__HIP_PLATFORM_AMD__)
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks,
                                                  kernel,
                                                  block_size,
                                                  smem_bytes);
#endif
    return active_blocks;
}

// -- Startup diagnostics --------------------------------------------------

/// Print GPU device info and kernel occupancy for a set of key kernels.
/// Call once at application startup for observability.
__host__ inline void print_device_info(int device_id = 0) {
#ifdef __CUDACC__
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device_id);
    printf("GPU Device %d: %s\n", device_id, prop.name);
    printf("  Compute:          %d.%d\n", prop.major, prop.minor);
    printf("  SMs:              %d\n", prop.multiProcessorCount);
    printf("  Max threads/SM:   %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max threads/blk:  %d\n", prop.maxThreadsPerBlock);
    printf("  Shared mem/SM:    %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  Shared mem/blk:   %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Registers/SM:     %d\n", prop.regsPerMultiprocessor);
    printf("  Warp size:        %d\n", prop.warpSize);
    printf("  Global memory:    %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  L2 cache:         %d KB\n", prop.l2CacheSize / 1024);
    printf("  Clock:            %d MHz\n", prop.clockRate / 1000);
    printf("  Memory clock:     %d MHz\n", prop.memoryClockRate / 1000);
    printf("  Memory bus:       %d-bit\n", prop.memoryBusWidth);
#elif defined(__HIP_PLATFORM_AMD__)
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, device_id);
    printf("GPU Device %d: %s\n", device_id, prop.name);
    printf("  Compute:      %d.%d\n", prop.major, prop.minor);
    printf("  CUs:          %d\n", prop.multiProcessorCount);
    printf("  Max threads:  %d/CU\n", prop.maxThreadsPerMultiProcessor);
    printf("  Global mem:   %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
#endif
}

// -- Warp-level reduction primitives --------------------------------------

/// Warp-wide sum reduction using shuffle-down.
/// All lanes in the warp participate; result is valid in lane 0.
__device__ __forceinline__ uint32_t warp_reduce_sum(uint32_t val) {
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return val;
}

/// Warp-wide sum reduction for 64-bit values.
__device__ __forceinline__ uint64_t warp_reduce_sum64(uint64_t val) {
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;
    val += __shfl_down_sync(FULL_MASK, val, 16);
    val += __shfl_down_sync(FULL_MASK, val, 8);
    val += __shfl_down_sync(FULL_MASK, val, 4);
    val += __shfl_down_sync(FULL_MASK, val, 2);
    val += __shfl_down_sync(FULL_MASK, val, 1);
    return val;
}

/// Warp-wide OR reduction (useful for bloom filter checks).
__device__ __forceinline__ uint32_t warp_reduce_or(uint32_t val) {
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;
    val |= __shfl_down_sync(FULL_MASK, val, 16);
    val |= __shfl_down_sync(FULL_MASK, val, 8);
    val |= __shfl_down_sync(FULL_MASK, val, 4);
    val |= __shfl_down_sync(FULL_MASK, val, 2);
    val |= __shfl_down_sync(FULL_MASK, val, 1);
    return val;
}

/// Warp-wide broadcast: lane 0 broadcasts its value to all lanes.
__device__ __forceinline__ uint32_t warp_broadcast(uint32_t val) {
    return __shfl_sync(0xFFFFFFFF, val, 0);
}

/// Warp-aggregated atomicAdd: one atomic per warp instead of per-thread.
/// Returns the global offset for each participating thread.
__device__ __forceinline__ uint32_t warp_aggregated_atomic_add(
    uint32_t* counter,
    uint32_t increment)
{
    constexpr uint32_t FULL_MASK = 0xFFFFFFFF;
    uint32_t active = __ballot_sync(FULL_MASK, increment > 0);
    if (active == 0) return 0;

    int lane = threadIdx.x & 31;
    int leader = __ffs(active) - 1;

    uint32_t warp_total = __popc(active);
    uint32_t warp_offset = 0;

    if (lane == leader) {
        warp_offset = atomicAdd(counter, warp_total);
    }
    warp_offset = __shfl_sync(FULL_MASK, warp_offset, leader);

    // Each thread gets its unique position
    uint32_t prefix = __popc(active & ((1u << lane) - 1));
    return warp_offset + prefix;
}

} // namespace cuda
} // namespace secp256k1
