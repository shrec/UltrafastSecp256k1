// ============================================================================
// SPM/CocoaPods compatibility forwarding header
// ============================================================================
// Canonical source: include/secp256k1/types.hpp (shared by CPU/CUDA/OpenCL)
//
// This forwarding header exists because SPM requires all public headers
// to be reachable from the target's publicHeadersPath (cpu/include/).
// The canonical types.hpp lives in the root include/ directory, which is
// shared across all backends. When building with CMake, both include paths
// are set via target_include_directories. When building with SPM/CocoaPods,
// the headerSearchPath("../include") cxxSetting resolves this — but
// downstream consumers also need the header accessible from publicHeadersPath.
//
// DO NOT EDIT — edit the canonical version at include/secp256k1/types.hpp
// ============================================================================

#pragma once
#include "../../../include/secp256k1/types.hpp"
