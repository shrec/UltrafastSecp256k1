// =============================================================================
// UltrafastSecp256k1 Metal -- GPU Compatibility Layer (gpu_compat_metal.h)
// =============================================================================
// Platform detection macros matching the CUDA gpu_compat.h pattern.
// Identifies Metal as the active GPU backend for conditional compilation.
//
// Unlike CUDA<->HIP (which share API structure), Metal has a fundamentally
// different dispatch model (Objective-C++, command encoders). This header
// provides detection macros only -- actual Metal API wrapping lives in
// metal_runtime.h / metal_runtime.mm.
// =============================================================================

#pragma once

// -------------------------------------------------------------
// Platform detection (matches gpu_compat.h contract)
// -------------------------------------------------------------
#define SECP256K1_GPU_METAL  1
#define SECP256K1_GPU_CUDA   0
#define SECP256K1_GPU_HIP    0
#define SECP256K1_USE_PTX    0

// Backend identification
#define SECP256K1_GPU_BACKEND_NAME  "Metal"
#define SECP256K1_GPU_UNIFIED_MEMORY 1   // Apple Silicon -- always unified

// -------------------------------------------------------------
// Metal framework includes (Objective-C++ only)
// -------------------------------------------------------------
#ifdef __OBJC__
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#endif

#include <cstdint>
#include <cstddef>
