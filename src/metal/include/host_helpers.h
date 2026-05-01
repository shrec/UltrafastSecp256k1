// =============================================================================
// UltrafastSecp256k1 Metal -- Host Helpers (host_helpers.h)
// =============================================================================
// Host-side utility functions for field elements, hex conversion, etc.
// Matches the CUDA host_helpers.cuh API pattern exactly.
//
// Uses uint64_t limbs[4] (little-endian) -- same as shared types.hpp.
// Memory is reinterpret_cast-compatible with MidFieldElementData{uint32_t[8]}
// used by Metal shaders, so buffer I/O is zero-cost on little-endian hosts.
// =============================================================================

#pragma once

#include "secp256k1/types.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <array>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>

namespace secp256k1 {
namespace metal {

// ============================================================
// Host Helper Functions (matching CUDA host_helpers.cuh)
// ============================================================

// Helper: Hex string -> big-endian byte array (32 bytes)
inline std::array<uint8_t, 32> hex_to_bytes(const char* hex) {
    std::array<uint8_t, 32> bytes{};
    size_t len = strlen(hex);
    if (len > 64) len = 64;

    char c;
    uint8_t val;
    size_t byte_idx;

    for (size_t i = 0; i < len; i++) {
        c = hex[i];
        val = 0;
        if (c >= '0' && c <= '9') val = static_cast<uint8_t>(c - '0');
        else if (c >= 'a' && c <= 'f') val = static_cast<uint8_t>(c - 'a' + 10);
        else if (c >= 'A' && c <= 'F') val = static_cast<uint8_t>(c - 'A' + 10);

        byte_idx = (len - 1 - i) / 2;
        if ((len - 1 - i) % 2 == 0) {
            bytes[31 - byte_idx] |= val;
        } else {
            bytes[31 - byte_idx] |= (val << 4);
        }
    }
    return bytes;
}

// Helper: Byte array -> hex string
inline std::string bytes_to_hex(const uint8_t* bytes, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << (int)bytes[i];
    }
    return ss.str();
}

// Helper: Case-insensitive hex string comparison
inline bool hex_equal(const std::string& a, const char* b) {
    if (a.length() != strlen(b)) return false;
    char ca, cb;
    for (size_t i = 0; i < a.length(); i++) {
        ca = a[i]; cb = b[i];
        if (ca >= 'A' && ca <= 'F') ca += 32;
        if (cb >= 'A' && cb <= 'F') cb += 32;
        if (ca != cb) return false;
    }
    return true;
}

// ============================================================
// Host Scalar -- uint64_t limbs[4], matching ScalarData
// ============================================================

struct HostScalar {
    uint64_t limbs[4];

    HostScalar() { memset(limbs, 0, sizeof(limbs)); }

    static HostScalar from_bytes(const std::array<uint8_t, 32>& bytes) {
        HostScalar s;
        uint64_t limb;
        for (size_t i = 0; i < 4; ++i) {
            limb = 0;
            for (size_t j = 0; j < 8; ++j) {
                limb |= (uint64_t)bytes[31 - (i * 8 + j)] << (j * 8);
            }
            s.limbs[i] = limb;
        }
        return s;
    }

    static HostScalar from_hex(const char* hex) {
        return from_bytes(hex_to_bytes(hex));
    }

    static HostScalar from_uint64(uint64_t v) {
        HostScalar s;
        s.limbs[0] = v;
        return s;
    }

    static HostScalar zero() { return HostScalar(); }
    static HostScalar one()  { return from_uint64(1); }

    std::array<uint8_t, 32> to_bytes() const {
        std::array<uint8_t, 32> bytes;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                bytes[31 - (i * 8 + j)] = static_cast<uint8_t>((limbs[i] >> (j * 8)) & 0xFF);
            }
        }
        return bytes;
    }

    std::string to_hex() const {
        return bytes_to_hex(to_bytes().data(), 32);
    }

    // Bridge to shared POD type (zero-cost -- identical layout)
    ScalarData to_data() const {
        ScalarData d;
        for (int i = 0; i < 4; i++) d.limbs[i] = limbs[i];
        return d;
    }

    static HostScalar from_data(const ScalarData& d) {
        HostScalar s;
        for (int i = 0; i < 4; i++) s.limbs[i] = d.limbs[i];
        return s;
    }

    bool operator==(const HostScalar& other) const {
        for (int i = 0; i < 4; i++) if (limbs[i] != other.limbs[i]) return false;
        return true;
    }

    bool operator!=(const HostScalar& other) const { return !(*this == other); }
};

// ============================================================
// Host Field Element -- uint64_t limbs[4], matching FieldElementData
// ============================================================

struct HostFieldElement {
    uint64_t limbs[4];

    HostFieldElement() { memset(limbs, 0, sizeof(limbs)); }

    HostFieldElement(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3) {
        limbs[0] = l0; limbs[1] = l1; limbs[2] = l2; limbs[3] = l3;
    }

    static HostFieldElement from_bytes(const std::array<uint8_t, 32>& bytes) {
        HostFieldElement f;
        uint64_t limb;
        for (size_t i = 0; i < 4; ++i) {
            limb = 0;
            for (size_t j = 0; j < 8; ++j) {
                limb |= (uint64_t)bytes[31 - (i * 8 + j)] << (j * 8);
            }
            f.limbs[i] = limb;
        }
        return f;
    }

    static HostFieldElement from_hex(const char* hex) {
        return from_bytes(hex_to_bytes(hex));
    }

    static HostFieldElement from_uint64(uint64_t v) {
        HostFieldElement f;
        f.limbs[0] = v;
        return f;
    }

    static HostFieldElement zero() { return HostFieldElement(); }
    static HostFieldElement one()  { return from_uint64(1); }

    std::array<uint8_t, 32> to_bytes() const {
        std::array<uint8_t, 32> bytes;
        for (size_t i = 0; i < 4; ++i) {
            for (size_t j = 0; j < 8; ++j) {
                bytes[31 - (i * 8 + j)] = static_cast<uint8_t>((limbs[i] >> (j * 8)) & 0xFF);
            }
        }
        return bytes;
    }

    std::string to_hex() const {
        return bytes_to_hex(to_bytes().data(), 32);
    }

    bool is_zero() const {
        for (int i = 0; i < 4; i++) if (limbs[i] != 0) return false;
        return true;
    }

    // Bridge to shared POD type (zero-cost -- identical layout)
    FieldElementData to_data() const {
        FieldElementData d;
        for (int i = 0; i < 4; i++) d.limbs[i] = limbs[i];
        return d;
    }

    static HostFieldElement from_data(const FieldElementData& d) {
        HostFieldElement f;
        for (int i = 0; i < 4; i++) f.limbs[i] = d.limbs[i];
        return f;
    }

    bool operator==(const HostFieldElement& other) const {
        for (int i = 0; i < 4; i++) if (limbs[i] != other.limbs[i]) return false;
        return true;
    }

    bool operator!=(const HostFieldElement& other) const { return !(*this == other); }
};

// ============================================================
// Host Affine Point -- (x, y), matching AffinePointData layout
// ============================================================

struct HostAffinePoint {
    HostFieldElement x;
    HostFieldElement y;

    AffinePointData to_data() const {
        AffinePointData d;
        d.x = x.to_data();
        d.y = y.to_data();
        return d;
    }

    static HostAffinePoint from_data(const AffinePointData& d) {
        HostAffinePoint p;
        p.x = HostFieldElement::from_data(d.x);
        p.y = HostFieldElement::from_data(d.y);
        return p;
    }
};

// ============================================================
// Host Jacobian Point -- (X, Y, Z, infinity)
// ============================================================

struct HostJacobianPoint {
    HostFieldElement x;
    HostFieldElement y;
    HostFieldElement z;
    uint32_t infinity = 0;

    JacobianPointData to_data() const {
        JacobianPointData d;
        d.x = x.to_data();
        d.y = y.to_data();
        d.z = z.to_data();
        d.infinity = infinity;
        return d;
    }

    static HostJacobianPoint from_data(const JacobianPointData& d) {
        HostJacobianPoint p;
        p.x = HostFieldElement::from_data(d.x);
        p.y = HostFieldElement::from_data(d.y);
        p.z = HostFieldElement::from_data(d.z);
        p.infinity = d.infinity;
        return p;
    }
};

// ============================================================
// Host Point -- Jacobian with factory methods (matching CUDA HostPoint)
// ============================================================
// NOTE: GPU arithmetic (scalar_mul, add, dbl, normalize) is done through
// MetalRuntime dispatch -- not embedded in the type. This matches the
// OpenCL host type pattern (explicit dispatch model).

struct HostPoint {
    HostFieldElement x_fe;
    HostFieldElement y_fe;
    HostFieldElement z_fe;
    bool infinity;

    HostPoint() : infinity(true) {
        z_fe = HostFieldElement::one();
    }

    static HostPoint generator() {
        HostPoint p;
        p.infinity = false;
        p.x_fe.limbs[0] = 0x59F2815B16F81798ULL; p.x_fe.limbs[1] = 0x029BFCDB2DCE28D9ULL;
        p.x_fe.limbs[2] = 0x55A06295CE870B07ULL; p.x_fe.limbs[3] = 0x79BE667EF9DCBBACULL;
        p.y_fe.limbs[0] = 0x9C47D08FFB10D4B8ULL; p.y_fe.limbs[1] = 0xFD17B448A6855419ULL;
        p.y_fe.limbs[2] = 0x5DA4FBFC0E1108A8ULL; p.y_fe.limbs[3] = 0x483ADA7726A3C465ULL;
        p.z_fe = HostFieldElement::one();
        return p;
    }

    static HostPoint from_affine(const HostFieldElement& x, const HostFieldElement& y) {
        HostPoint p;
        p.infinity = false;
        p.x_fe = x;
        p.y_fe = y;
        p.z_fe = HostFieldElement::one();
        return p;
    }

    static HostPoint from_host_affine(const HostAffinePoint& ap) {
        return from_affine(ap.x, ap.y);
    }

    static HostPoint infinity_point() {
        return HostPoint();
    }

    HostAffinePoint to_affine() const {
        // Assumes already normalized (Z == 1)
        HostAffinePoint ap;
        ap.x = x_fe;
        ap.y = y_fe;
        return ap;
    }

    JacobianPointData to_data() const {
        JacobianPointData d;
        d.x = x_fe.to_data();
        d.y = y_fe.to_data();
        d.z = z_fe.to_data();
        d.infinity = infinity ? 1 : 0;
        return d;
    }

    static HostPoint from_data(const JacobianPointData& d) {
        HostPoint p;
        p.x_fe = HostFieldElement::from_data(d.x);
        p.y_fe = HostFieldElement::from_data(d.y);
        p.z_fe = HostFieldElement::from_data(d.z);
        p.infinity = (d.infinity != 0);
        return p;
    }

    bool is_infinity() const { return infinity; }

    HostPoint negate() const {
        HostPoint p = *this;
        if (p.infinity) return p;
        // Negate Y: field negate (p - y)
        // P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        uint64_t P[4] = {
            0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
            0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
        };
        HostFieldElement neg_y;
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t diff = P[i] - p.y_fe.limbs[i] - borrow;
            borrow = (P[i] < p.y_fe.limbs[i] + borrow) ? 1 : 0;
            neg_y.limbs[i] = diff;
        }
        p.y_fe = neg_y;
        return p;
    }

    // to_compressed/to_uncompressed assume normalized (Z == 1) point
    std::vector<uint8_t> to_compressed() const {
        std::vector<uint8_t> res(33);
        auto x_bytes = x_fe.to_bytes();
        auto y_bytes = y_fe.to_bytes();
        res[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
        std::copy(x_bytes.begin(), x_bytes.end(), res.begin() + 1);
        return res;
    }

    std::vector<uint8_t> to_uncompressed() const {
        std::vector<uint8_t> res(65);
        auto x_bytes = x_fe.to_bytes();
        auto y_bytes = y_fe.to_bytes();
        res[0] = 0x04;
        std::copy(x_bytes.begin(), x_bytes.end(), res.begin() + 1);
        std::copy(y_bytes.begin(), y_bytes.end(), res.begin() + 33);
        return res;
    }
};

// =============================================================================
// Known Test Vectors
// =============================================================================

// Generator point G
inline HostAffinePoint generator_point() {
    HostAffinePoint g;
    g.x = HostFieldElement::from_hex(
        "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798");
    g.y = HostFieldElement::from_hex(
        "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8");
    return g;
}

// 2*G
inline HostAffinePoint two_g_point() {
    HostAffinePoint p;
    p.x = HostFieldElement::from_hex(
        "C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5");
    p.y = HostFieldElement::from_hex(
        "1AE168FEA63DC339A3C58419466CEAEEF7F632653266D0E1236431A950CFE52A");
    return p;
}

// 3*G
inline HostAffinePoint three_g_point() {
    HostAffinePoint p;
    p.x = HostFieldElement::from_hex(
        "F9308A019258C31049344F85F89D5229B531C845836F99B08601F113BCE036F9");
    p.y = HostFieldElement::from_hex(
        "388F7B0F632DE8140FE337E62A37F3566500A99934C2231B6CB9FD7584B8E672");
    return p;
}

// =============================================================================
// Pretty Print
// =============================================================================

inline void print_point(const char* label, const HostAffinePoint& p) {
    std::cout << label << ":\n"
              << "  X = " << p.x.to_hex() << "\n"
              << "  Y = " << p.y.to_hex() << "\n";
}

inline void print_field(const char* label, const HostFieldElement& f) {
    std::cout << label << " = " << f.to_hex() << "\n";
}

// =============================================================================
// Layout Guarantees -- Cross-backend compatibility with shared types.hpp
// =============================================================================
// FieldElementData{uint64_t[4]} and MidFieldElementData{uint32_t[8]} are
// reinterpret_cast-compatible (same 32 bytes, little-endian).
// Metal shaders use 8x32-bit but host stores 4x64-bit -- buffer I/O is
// zero-cost since the byte layout is identical on little-endian platforms.

static_assert(sizeof(HostFieldElement) == sizeof(FieldElementData),
              "HostFieldElement must match FieldElementData layout (32 bytes)");
static_assert(sizeof(HostFieldElement) == sizeof(MidFieldElementData),
              "HostFieldElement must match MidFieldElementData layout (32 bytes)");
static_assert(sizeof(HostScalar) == sizeof(ScalarData),
              "HostScalar must match ScalarData layout (32 bytes)");
static_assert(sizeof(HostAffinePoint) == sizeof(AffinePointData),
              "HostAffinePoint must match AffinePointData layout (64 bytes)");

} // namespace metal
} // namespace secp256k1
