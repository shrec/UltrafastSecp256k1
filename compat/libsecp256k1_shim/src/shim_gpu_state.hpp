// ============================================================================
// shim_gpu_state.hpp -- GPU acceleration state for the libsecp256k1 shim
// ============================================================================
// Internal header. Do NOT include from public shim headers.
//
// When SECP256K1_SHIM_GPU is defined (GPU backend compiled in), this module
// manages a process-wide GPU context used for non-CT verify acceleration.
//
// CT signing is NEVER dispatched to GPU — always uses the CPU CT layer.
// ============================================================================
#pragma once

#ifdef SECP256K1_SHIM_GPU

#include "ufsecp/ufsecp_gpu.h"
#include <mutex>
#include <cstdint>

struct ShimGpuState {
    ufsecp_gpu_ctx* ctx      = nullptr;
    uint32_t        backend  = UFSECP_GPU_BACKEND_NONE;
    uint32_t        device   = 0;    // device index within the backend
    bool            enabled  = false;
    std::mutex      mu;   // protects ctx for concurrent single-item dispatches
};

// Returns the process-wide GPU state singleton.
ShimGpuState& shim_gpu_state() noexcept;

// Initialise GPU from a config.ini path (reads [gpu] section).
// Safe to call multiple times — initialises only once (std::once_flag).
void shim_gpu_init(const char* config_ini_path);

// Destroy the GPU context (called on last context_destroy or atexit).
void shim_gpu_shutdown();

#endif // SECP256K1_SHIM_GPU
