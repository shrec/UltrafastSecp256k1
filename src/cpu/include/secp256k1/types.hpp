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
// the headerSearchPath("../include") cxxSetting resolves this -- but
// downstream consumers also need the header accessible from publicHeadersPath.
//
// Portability note: this header provides the type definitions inline using a
// guard (SECP256K1_TYPES_HPP_INCLUDED_) rather than relying on a relative
// include path that may not resolve correctly on all build systems (e.g. MSVC
// with certain CMake generators). The canonical include/secp256k1/types.hpp
// uses the same guard, so whichever file is included first wins and the other
// is silently skipped — no redefinition, no missing types.
//
// DO NOT diverge from include/secp256k1/types.hpp — keep both files in sync.
// ============================================================================

#pragma once

#ifndef SECP256K1_TYPES_HPP_INCLUDED_
#define SECP256K1_TYPES_HPP_INCLUDED_

#include <cstdint>
#include <cstddef>

namespace secp256k1 {

struct FieldElementData {
    uint64_t limbs[4];  // Little-endian: limbs[0] = bits [0..63]
};

struct MidFieldElementData {
    uint32_t limbs[8];  // Little-endian: limbs[0] = bits [0..31]
};

struct ScalarData {
    uint64_t limbs[4];  // Little-endian: limbs[0] = bits [0..63]
};

struct AffinePointData {
    FieldElementData x;
    FieldElementData y;
};

struct JacobianPointData {
    FieldElementData x;
    FieldElementData y;
    FieldElementData z;
    uint32_t infinity;  // 1 = point at infinity, 0 = normal point
};

static_assert(sizeof(FieldElementData)    == 32, "FieldElement must be 256 bits");
static_assert(sizeof(MidFieldElementData) == 32, "MidFieldElement must be 256 bits");
static_assert(sizeof(ScalarData)          == 32, "Scalar must be 256 bits");
static_assert(sizeof(AffinePointData)     == 64, "AffinePoint must be 512 bits");
static_assert(sizeof(FieldElementData) == sizeof(MidFieldElementData),
              "FieldElementData and MidFieldElementData must be same size");
static_assert(offsetof(AffinePointData, x)    == 0,  "AffinePoint.x at offset 0");
static_assert(offsetof(AffinePointData, y)    == 32, "AffinePoint.y at offset 32");
static_assert(offsetof(JacobianPointData, x)  == 0,  "JacobianPoint.x at offset 0");
static_assert(offsetof(JacobianPointData, y)  == 32, "JacobianPoint.y at offset 32");
static_assert(offsetof(JacobianPointData, z)  == 64, "JacobianPoint.z at offset 64");

inline FieldElementData* fe_to_data(void* fe) noexcept {
    return static_cast<FieldElementData*>(fe);
}
inline const FieldElementData* fe_to_data(const void* fe) noexcept {
    return static_cast<const FieldElementData*>(fe);
}
inline ScalarData* sc_to_data(void* sc) noexcept {
    return static_cast<ScalarData*>(sc);
}
inline const ScalarData* sc_to_data(const void* sc) noexcept {
    return static_cast<const ScalarData*>(sc);
}

} // namespace secp256k1

#endif // SECP256K1_TYPES_HPP_INCLUDED_
