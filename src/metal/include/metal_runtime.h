// =============================================================================
// UltrafastSecp256k1 Metal -- Host-side Metal Runtime (metal_runtime.h)
// =============================================================================
// C++ wrapper for Apple Metal compute pipeline.
// Manages device, command queues, pipeline states, and buffer I/O.
//
// Usage:
//   MetalRuntime runtime;
//   runtime.init();
//   runtime.load_shader("secp256k1_kernels");
//   auto pipeline = runtime.make_pipeline("scalar_mul_batch");
//   auto buf = runtime.alloc_buffer(size);
//   runtime.dispatch(pipeline, grid_size, threadgroup_size);
// =============================================================================

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>

// Forward declarations -- actual Metal types are Objective-C objects
// We use opaque pointers in the C++ header to avoid requiring .mm everywhere
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#else
// Opaque forward declarations for pure C++ headers
typedef void* MTLDevicePtr;
typedef void* MTLCommandQueuePtr;
typedef void* MTLLibraryPtr;
typedef void* MTLComputePipelineStatePtr;
typedef void* MTLBufferPtr;
typedef void* MTLCommandBufferPtr;
typedef void* MTLComputeCommandEncoderPtr;
#endif

namespace secp256k1 {
namespace metal {

// =============================================================================
// Device Info
// =============================================================================

struct DeviceInfo {
    std::string name;
    uint64_t max_buffer_length;      // Max single buffer size (bytes)
    uint64_t recommended_working_set; // Recommended total GPU memory
    uint32_t max_threads_per_threadgroup;
    bool supports_family_apple7;     // M1+
    bool supports_family_apple8;     // M2+
    bool supports_family_apple9;     // M3+
    bool unified_memory;             // Always true on Apple Silicon
};

// =============================================================================
// Buffer Wrapper -- RAII wrapper for MTLBuffer
// =============================================================================

class MetalBuffer {
public:
    MetalBuffer() = default;
    MetalBuffer(MetalBuffer&& other) noexcept;
    MetalBuffer& operator=(MetalBuffer&& other) noexcept;
    ~MetalBuffer();

    // Non-copyable
    MetalBuffer(const MetalBuffer&) = delete;
    MetalBuffer& operator=(const MetalBuffer&) = delete;

    // Data access (unified memory -- no explicit copy needed!)
    void* contents();
    const void* contents() const;
    size_t length() const;

    // Type-safe upload/download
    template<typename T>
    void write(const T* data, size_t count, size_t offset = 0) {
        memcpy(static_cast<uint8_t*>(contents()) + offset, data, count * sizeof(T));
    }

    template<typename T>
    void read(T* data, size_t count, size_t offset = 0) const {
        memcpy(data, static_cast<const uint8_t*>(contents()) + offset, count * sizeof(T));
    }

    bool valid() const;

#ifdef __OBJC__
    id<MTLBuffer> native() const { return buffer_; }
    explicit MetalBuffer(id<MTLBuffer> buf) : buffer_(buf) {}
#else
    void* native() const { return buffer_; }
    explicit MetalBuffer(void* buf) : buffer_(buf) {}
#endif

private:
#ifdef __OBJC__
    id<MTLBuffer> buffer_ = nil;
#else
    void* buffer_ = nullptr;
#endif
};

// =============================================================================
// ComputePipeline -- Wrapper for MTLComputePipelineState
// =============================================================================

class ComputePipeline {
public:
    ComputePipeline() = default;
    
    uint32_t max_total_threads_per_threadgroup() const;
    uint32_t threadExecutionWidth() const;
    std::string label() const;
    bool valid() const;

#ifdef __OBJC__
    id<MTLComputePipelineState> native() const { return pipeline_; }
    explicit ComputePipeline(id<MTLComputePipelineState> p, NSString* name)
        : pipeline_(p), name_([name UTF8String]) {}
#else
    void* native() const { return pipeline_; }
#endif

private:
#ifdef __OBJC__
    id<MTLComputePipelineState> pipeline_ = nil;
#else
    void* pipeline_ = nullptr;
#endif
    std::string name_;
};

// =============================================================================
// MetalRuntime -- Main interface for GPU compute
// =============================================================================

class MetalRuntime {
public:
    MetalRuntime();
    ~MetalRuntime();

    // Initialize -- select device (default = system default GPU)
    bool init(int device_id = 0);

    // Device info
    DeviceInfo device_info() const;
    void print_device_info() const;

    // Shader management
    // Load compiled metallib from path, or compile .metal source at runtime
    bool load_library_from_path(const std::string& metallib_path);
    bool load_library_from_source(const std::string& source_code);
    
    // Create compute pipeline for a kernel function
    ComputePipeline make_pipeline(const std::string& function_name);

    // Buffer management (Apple Silicon unified memory -- zero-copy!)
    MetalBuffer alloc_buffer(size_t bytes);
    MetalBuffer alloc_buffer_shared(size_t bytes);  // Explicit shared mode
    MetalBuffer alloc_buffer_private(size_t bytes);  // GPU-only (faster for intermediates)

    // Command submission
    // Simple 1D dispatch
    void dispatch_1d(const ComputePipeline& pipeline,
                     uint32_t grid_size,
                     const std::vector<MetalBuffer*>& buffers);
    
    // dispatch with explicit threadgroup size
    void dispatch_1d(const ComputePipeline& pipeline,
                     uint32_t grid_size,
                     uint32_t threadgroup_size,
                     const std::vector<MetalBuffer*>& buffers);

    // Synchronous dispatch -- blocks until GPU completes
    void dispatch_sync(const ComputePipeline& pipeline,
                       uint32_t grid_size,
                       uint32_t threadgroup_size,
                       const std::vector<MetalBuffer*>& buffers);

    // Synchronous dispatch that PROPAGATES command-buffer failure to the
    // caller. dispatch_sync() above only inspects `cmd_buf.error` internally
    // and logs it to stderr -- it returns void, so a fire-and-forget call
    // site silently falls through "as if" a failed GPU dispatch (device
    // lost, runtime shader fault, driver timeout) succeeded, and the caller
    // proceeds to read back stale/undefined buffer contents and reports
    // GpuError::Ok.
    //
    // dispatch_sync_checked() returns false iff the command buffer completed
    // with a non-nil error, so security/correctness-critical callers can
    // fail closed (zero their output buffer, return a real GpuError) instead
    // of reporting corrupted output as success. Added 2026-07 (GitHub issue
    // #335 acceptance repair, round 2), then migrated repo-wide (round 8):
    // gpu_backend_metal.mm now calls dispatch_sync_checked() at all 37 of
    // its GpuBackend virtual-method dispatch sites and has zero remaining
    // bare dispatch_sync() calls -- see check_gpu_backend_parity.py / the
    // per-method fail-closed patterns in that file for the current source of
    // truth. The bare dispatch_sync() overload above is retained only for
    // non-shipped, Apple-only dev tools (src/metal/app/bench_metal.mm,
    // src/metal/app/metal_test.mm) that are not part of the production C-ABI
    // surface and are gated `if(NOT APPLE) return()` in
    // src/metal/CMakeLists.txt. This doc comment was stale after the round-8
    // migration landed -- keep it in sync with gpu_backend_metal.mm.
    bool dispatch_sync_checked(const ComputePipeline& pipeline,
                               uint32_t grid_size,
                               uint32_t threadgroup_size,
                               const std::vector<MetalBuffer*>& buffers);

    // Wait for all pending work
    void synchronize();

    // Timing
    double last_kernel_time_ms() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace metal
} // namespace secp256k1
