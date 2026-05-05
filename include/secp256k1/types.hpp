// =============================================================================
// UltrafastSecp256k1 -- Shared POD Data Types
// =============================================================================
// Canonical data layouts for secp256k1 field elements, scalars, and points.
// These types define the MEMORY LAYOUT contract between all backends
// (CPU, CUDA, OpenCL). Each backend may extend with its own methods,
// alignment, or __device__ qualifiers, but must remain layout-compatible.
//
// Design principles:
//   - Pure POD: no constructors, no virtual methods, no inheritance
//   - Zero overhead: reinterpret_cast between backend and shared types
//   - Little-endian limbs: limbs[0] is the least significant
//
// Backend usage:
//   - CUDA:   using FieldElement = secp256k1::FieldElementData;  (direct alias)
//   - OpenCL: struct FieldElement { ... };  + static_assert layout match
//   - CPU:    class FieldElement { ... };   + data()/from_data() accessors
// =============================================================================

#pragma once

// Guard matches src/cpu/include/secp256k1/types.hpp (forwarding header).
// Whichever file is included first defines the types; the other is skipped.
#ifndef SECP256K1_TYPES_HPP_INCLUDED_
#define SECP256K1_TYPES_HPP_INCLUDED_

#include <cstdint>
#include <cstddef>

namespace secp256k1 {

// -----------------------------------------------------------------------------
// Field element: 256-bit integer mod p (secp256k1 prime)
// p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
// -----------------------------------------------------------------------------
struct FieldElementData {
    uint64_t limbs[4];  // Little-endian: limbs[0] = bits [0..63]
};

// -----------------------------------------------------------------------------
// 32-bit view of field element (same 256 bits, different interpretation)
// Memory layout is IDENTICAL to FieldElementData -- safe to reinterpret_cast
// -----------------------------------------------------------------------------
struct MidFieldElementData {
    uint32_t limbs[8];  // Little-endian: limbs[0] = bits [0..31]
};

// -----------------------------------------------------------------------------
// Scalar: 256-bit integer mod n (curve order)
// n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
// -----------------------------------------------------------------------------
struct ScalarData {
    uint64_t limbs[4];  // Little-endian: limbs[0] = bits [0..63]
};

// -----------------------------------------------------------------------------
// Affine point: (x, y) on the curve y^2 = x^3 + 7
// -----------------------------------------------------------------------------
struct AffinePointData {
    FieldElementData x;
    FieldElementData y;
};

// -----------------------------------------------------------------------------
// Jacobian point: (X, Y, Z) where affine (x, y) = (X/Z^2, Y/Z^3)
//
// NOTE: The infinity flag representation varies by backend:
//   - CPU:    bool   (1 byte)  -- inside Point class
//   - CUDA:   bool   (1 byte)  -- in JacobianPoint struct
//   - OpenCL: uint32_t (4 bytes, GPU-friendly with padding)
//
// This reference layout uses uint32_t for maximum portability.
// Backends may define their own JacobianPoint with a different infinity type
// as long as the x/y/z field offsets match.
// -----------------------------------------------------------------------------
struct JacobianPointData {
    FieldElementData x;
    FieldElementData y;
    FieldElementData z;
    uint32_t infinity;  // 1 = point at infinity, 0 = normal point
};

// =============================================================================
// Layout Guarantees
// =============================================================================
static_assert(sizeof(FieldElementData)    == 32, "FieldElement must be 256 bits");
static_assert(sizeof(MidFieldElementData) == 32, "MidFieldElement must be 256 bits");
static_assert(sizeof(ScalarData)          == 32, "Scalar must be 256 bits");
static_assert(sizeof(AffinePointData)     == 64, "AffinePoint must be 512 bits");

// FieldElementData and MidFieldElementData are reinterpret_cast-compatible
static_assert(sizeof(FieldElementData) == sizeof(MidFieldElementData),
              "FieldElementData and MidFieldElementData must be same size");

// Field offsets for cross-backend compatibility checks
static_assert(offsetof(AffinePointData, x) == 0,  "AffinePoint.x at offset 0");
static_assert(offsetof(AffinePointData, y) == 32, "AffinePoint.y at offset 32");

static_assert(offsetof(JacobianPointData, x) == 0,  "JacobianPoint.x at offset 0");
static_assert(offsetof(JacobianPointData, y) == 32, "JacobianPoint.y at offset 32");
static_assert(offsetof(JacobianPointData, z) == 64, "JacobianPoint.z at offset 64");

// =============================================================================
// Zero-cost Conversion Utilities
// =============================================================================
// These are provided for convenience in wrapper/bridge code.
// Within each backend, prefer using the backend's native type directly.

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
