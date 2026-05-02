// =============================================================================
// UltrafastSecp256k1 Metal — Host-side Metal Runtime Implementation
// =============================================================================
// Objective-C++ implementation of the MetalRuntime C++ interface.
// Must be compiled as .mm (Objective-C++) on macOS.
// =============================================================================

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "metal_runtime.h"
#include <iostream>
#include <chrono>
#include <cassert>

namespace secp256k1 {
namespace metal {

// =============================================================================
// MetalBuffer Implementation
// =============================================================================

MetalBuffer::MetalBuffer(MetalBuffer&& other) noexcept : buffer_(other.buffer_) {
    other.buffer_ = nil;
}

MetalBuffer& MetalBuffer::operator=(MetalBuffer&& other) noexcept {
    if (this != &other) {
        buffer_ = other.buffer_;
        other.buffer_ = nil;
    }
    return *this;
}

MetalBuffer::~MetalBuffer() {
    // Rule 10: zero buffer contents before ARC dealloc (may hold private key material)
    if (buffer_ && [buffer_ contents] && [buffer_ length] > 0) {
        memset([buffer_ contents], 0, [buffer_ length]);
        [buffer_ didModifyRange:NSMakeRange(0, [buffer_ length])];
    }
    buffer_ = nil;
}

void* MetalBuffer::contents() {
    return [buffer_ contents];
}

const void* MetalBuffer::contents() const {
    return [buffer_ contents];
}

size_t MetalBuffer::length() const {
    return buffer_ ? [buffer_ length] : 0;
}

bool MetalBuffer::valid() const {
    return buffer_ != nil;
}

// =============================================================================
// ComputePipeline Implementation
// =============================================================================

uint32_t ComputePipeline::max_total_threads_per_threadgroup() const {
    return pipeline_ ? (uint32_t)[pipeline_ maxTotalThreadsPerThreadgroup] : 0;
}

uint32_t ComputePipeline::threadExecutionWidth() const {
    return pipeline_ ? (uint32_t)[pipeline_ threadExecutionWidth] : 0;
}

std::string ComputePipeline::label() const {
    return name_;
}

bool ComputePipeline::valid() const {
    return pipeline_ != nil;
}

// =============================================================================
// MetalRuntime::Impl — Private implementation (PIMPL)
// =============================================================================

struct MetalRuntime::Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> command_queue = nil;
    id<MTLLibrary> library = nil;
    
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;
    
    double last_kernel_time_ms_ = 0.0;
    
    ~Impl() {
        pipeline_cache.clear();
        library = nil;
        command_queue = nil;
        device = nil;
    }
};

// =============================================================================
// MetalRuntime Implementation
// =============================================================================

MetalRuntime::MetalRuntime() : impl_(std::make_unique<Impl>()) {}

MetalRuntime::~MetalRuntime() = default;

bool MetalRuntime::init(int device_id) {
    @autoreleasepool {
        NSArray<id<MTLDevice>>* devices = MTLCopyAllDevices();
        if (!devices || devices.count == 0) {
            // Fallback to default device
            impl_->device = MTLCreateSystemDefaultDevice();
            if (!impl_->device) {
                std::cerr << "[Metal] ERROR: No Metal-capable GPU found\n";
                return false;
            }
        } else {
            if (device_id >= 0 && (NSUInteger)device_id < devices.count) {
                impl_->device = devices[device_id];
            } else {
                impl_->device = devices[0];
            }
        }
        
        impl_->command_queue = [impl_->device newCommandQueue];
        if (!impl_->command_queue) {
            std::cerr << "[Metal] ERROR: Failed to create command queue\n";
            return false;
        }
        
        return true;
    }
}

DeviceInfo MetalRuntime::device_info() const {
    DeviceInfo info;
    if (!impl_->device) return info;
    
    info.name = [[impl_->device name] UTF8String];
    info.max_buffer_length = [impl_->device maxBufferLength];
    info.recommended_working_set = [impl_->device recommendedMaxWorkingSetSize];
    // Query device max threads per threadgroup (not a constant — varies by family).
    info.max_threads_per_threadgroup = static_cast<uint32_t>(
        [impl_->device maxThreadsPerThreadgroup].width);
    info.supports_family_apple7 = [impl_->device supportsFamily:MTLGPUFamilyApple7];
    info.supports_family_apple8 = [impl_->device supportsFamily:MTLGPUFamilyApple8];
    // Apple9 check — MTLGPUFamilyApple9 is an enum (not a macro),
    // so #if defined() doesn't work. Use SDK version guard instead.
#if (__MAC_OS_X_VERSION_MAX_ALLOWED >= 140000)
    if (@available(macOS 14.0, *)) {
        info.supports_family_apple9 = [impl_->device supportsFamily:MTLGPUFamilyApple9];
    } else {
        info.supports_family_apple9 = false;
    }
#else
    info.supports_family_apple9 = false;
#endif
    info.unified_memory = true; // Always true on Apple Silicon
    
    return info;
}

void MetalRuntime::print_device_info() const {
    auto info = device_info();
    std::cout << "=== Metal GPU Device ===\n"
              << "  Name: " << info.name << "\n"
              << "  Max buffer: " << (info.max_buffer_length / (1024*1024)) << " MB\n"
              << "  Recommended VRAM: " << (info.recommended_working_set / (1024*1024)) << " MB\n"
              << "  Max threads/group: " << info.max_threads_per_threadgroup << "\n"
              << "  Apple7+ (M1): " << (info.supports_family_apple7 ? "Yes" : "No") << "\n"
              << "  Apple8+ (M2): " << (info.supports_family_apple8 ? "Yes" : "No") << "\n"
              << "  Apple9+ (M3): " << (info.supports_family_apple9 ? "Yes" : "No") << "\n"
              << "  Unified Memory: " << (info.unified_memory ? "Yes" : "No") << "\n"
              << "========================\n";
}

bool MetalRuntime::load_library_from_path(const std::string& metallib_path) {
    @autoreleasepool {
        NSError* error = nil;
        NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        
        impl_->library = [impl_->device newLibraryWithURL:url error:&error];
        if (!impl_->library) {
            std::cerr << "[Metal] ERROR: Failed to load metallib: "
                      << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }
        return true;
    }
}

bool MetalRuntime::load_library_from_source(const std::string& source_code) {
    @autoreleasepool {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:source_code.c_str()];
        
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = YES;
        options.languageVersion = MTLLanguageVersion2_4;
        
        impl_->library = [impl_->device newLibraryWithSource:src options:options error:&error];
        if (!impl_->library) {
            std::cerr << "[Metal] ERROR: Failed to compile shader source: "
                      << [[error localizedDescription] UTF8String] << "\n";
            return false;
        }
        return true;
    }
}

ComputePipeline MetalRuntime::make_pipeline(const std::string& function_name) {
    @autoreleasepool {
        // Check cache
        auto it = impl_->pipeline_cache.find(function_name);
        if (it != impl_->pipeline_cache.end()) {
            NSString* name = [NSString stringWithUTF8String:function_name.c_str()];
            return ComputePipeline(it->second, name);
        }
        
        NSString* fname = [NSString stringWithUTF8String:function_name.c_str()];
        id<MTLFunction> func = [impl_->library newFunctionWithName:fname];
        if (!func) {
            std::cerr << "[Metal] ERROR: Function not found: " << function_name << "\n";
            return ComputePipeline();
        }
        
        NSError* error = nil;
        id<MTLComputePipelineState> pipeline =
            [impl_->device newComputePipelineStateWithFunction:func error:&error];
        if (!pipeline) {
            std::cerr << "[Metal] ERROR: Failed to create pipeline for "
                      << function_name << ": "
                      << [[error localizedDescription] UTF8String] << "\n";
            return ComputePipeline();
        }
        
        impl_->pipeline_cache[function_name] = pipeline;
        return ComputePipeline(pipeline, fname);
    }
}

MetalBuffer MetalRuntime::alloc_buffer(size_t bytes) {
    return alloc_buffer_shared(bytes);
}

MetalBuffer MetalRuntime::alloc_buffer_shared(size_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> buf = [impl_->device newBufferWithLength:bytes
                                                       options:MTLResourceStorageModeShared];
        if (!buf) {
            std::cerr << "[Metal] ERROR: Failed to allocate " << bytes << " bytes (shared)\n";
            return MetalBuffer();
        }
        return MetalBuffer(buf);
    }
}

MetalBuffer MetalRuntime::alloc_buffer_private(size_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> buf = [impl_->device newBufferWithLength:bytes
                                                       options:MTLResourceStorageModePrivate];
        if (!buf) {
            std::cerr << "[Metal] ERROR: Failed to allocate " << bytes << " bytes (private)\n";
            return MetalBuffer();
        }
        return MetalBuffer(buf);
    }
}

void MetalRuntime::dispatch_1d(const ComputePipeline& pipeline,
                                uint32_t grid_size,
                                const std::vector<MetalBuffer*>& buffers) {
    uint32_t tg_size = pipeline.threadExecutionWidth();
    if (tg_size == 0) tg_size = 256;
    dispatch_1d(pipeline, grid_size, tg_size, buffers);
}

void MetalRuntime::dispatch_1d(const ComputePipeline& pipeline,
                                uint32_t grid_size,
                                uint32_t threadgroup_size,
                                const std::vector<MetalBuffer*>& buffers) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [impl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline.native()];
        
        for (size_t i = 0; i < buffers.size(); i++) {
            [encoder setBuffer:buffers[i]->native() offset:0 atIndex:i];
        }
        
        MTLSize grid = MTLSizeMake(grid_size, 1, 1);
        MTLSize tg = MTLSizeMake(threadgroup_size, 1, 1);
        
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        [encoder endEncoding];
        [cmd_buf commit];
    }
}

void MetalRuntime::dispatch_sync(const ComputePipeline& pipeline,
                                  uint32_t grid_size,
                                  uint32_t threadgroup_size,
                                  const std::vector<MetalBuffer*>& buffers) {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [impl_->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
        
        [encoder setComputePipelineState:pipeline.native()];
        
        for (size_t i = 0; i < buffers.size(); i++) {
            [encoder setBuffer:buffers[i]->native() offset:0 atIndex:i];
        }
        
        MTLSize grid = MTLSizeMake(grid_size, 1, 1);
        MTLSize tg = MTLSizeMake(threadgroup_size, 1, 1);
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        [encoder endEncoding];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
        
        auto t1 = std::chrono::high_resolution_clock::now();
        impl_->last_kernel_time_ms_ =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        
        if (cmd_buf.error) {
            std::cerr << "[Metal] Kernel error: "
                      << [[cmd_buf.error localizedDescription] UTF8String] << "\n";
        }
    }
}

void MetalRuntime::synchronize() {
    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [impl_->command_queue commandBuffer];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

double MetalRuntime::last_kernel_time_ms() const {
    return impl_->last_kernel_time_ms_;
}

} // namespace metal
} // namespace secp256k1
