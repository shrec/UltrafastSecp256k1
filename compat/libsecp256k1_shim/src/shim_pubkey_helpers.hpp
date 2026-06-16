// ============================================================================
// shim_pubkey_helpers.hpp — Shared inline helpers for shim_*.cpp files
// ============================================================================
// `#pragma once` ensures Unity build compiles each helper exactly once even
// when all shim_*.cpp files are merged into a single translation unit.
// All functions are `inline` in the secp256k1_shim_internal namespace to avoid
// duplicate-symbol errors when multiple shim files are compiled together.
// ============================================================================
#pragma once
#include <array>
#include <cstring>
#include "secp256k1/field.hpp"
#include "secp256k1/point.hpp"

namespace secp256k1_shim_internal {

using secp256k1::fast::Point;
using secp256k1::fast::FieldElement;

// Write X[0..31] || Y[32..63] from a Point into a 64-byte opaque buffer.
// Fast path (Z=1): extract X and Y field elements directly — no allocation.
// Fallback (Jacobian): copy the point and normalize (one field inversion) then
// extract, avoiding the 65-byte to_uncompressed() heap allocation + memcpy.
inline void point_to_pubkey_data(const Point& pt, unsigned char data[64]) noexcept {
    if (pt.is_normalized()) {
        pt.x_raw().to_bytes_into(reinterpret_cast<uint8_t*>(data));
        pt.y_raw().to_bytes_into(reinterpret_cast<uint8_t*>(data) + 32);
    } else {
        Point n = pt;
        n.normalize();
        n.x_raw().to_bytes_into(reinterpret_cast<uint8_t*>(data));
        n.y_raw().to_bytes_into(reinterpret_cast<uint8_t*>(data) + 32);
    }
}

// Reconstruct a Point from a 64-byte opaque pubkey buffer (X || Y).
// TRUST CONTRACT (matches libsecp256k1 + PERF-002 / commit c67edc1c): EC curve
// membership (y²=x³+7) is validated exactly ONCE at ec_pubkey_parse /
// ec_pubkey_create. The verify, batch, tweak and combine paths trust the opaque
// secp256k1_pubkey struct — they do NOT re-run the per-call curve equation. The
// y²=x³+7 re-check re-added by d0b1435e ("Harden …") regressed that decision
// (see kb PERF-002-FIXED / SHIM-A10-TEST-TRUST-CONTRACT: "do not re-add"); it is
// removed again here. We keep the cheap field-range guard (reject x/y >= p) so
// from_affine only ever sees canonical limbs — that is range validation, not the
// EC curve equation, and it is what keeps off-curve handling deterministic.
[[nodiscard]] inline Point pubkey_data_to_point(const unsigned char data[64]) noexcept {
    const auto& xb = *reinterpret_cast<const std::array<uint8_t, 32>*>(data);
    const auto& yb = *reinterpret_cast<const std::array<uint8_t, 32>*>(data + 32);
    FieldElement x, y;
    if (!FieldElement::parse_bytes_strict(xb, x)) return Point::infinity();
    if (!FieldElement::parse_bytes_strict(yb, y)) return Point::infinity();
    return Point::from_affine(x, y);
}

// Like pubkey_data_to_point, but ADDITIONALLY enforces EC curve membership
// (y² == x³ + 7) and returns infinity for an off-curve (incl. all-zero) opaque
// pubkey. Used by the pubkey MANIPULATION / serialization / derivation paths
// (negate, serialize, tweak_add/mul, combine, x-only / keypair extraction) so they
// stay FAIL-CLOSED on a malformed `secp256k1_pubkey` struct — a deliberately
// libsecp-stricter guardrail (see audit/test_regression_p2_ct_shim_fixes.cpp:
// SHIM-FC / NEG-002). The HOT verify paths (ecdsa/schnorr verify + batch) keep the
// unchecked pubkey_data_to_point: curve membership is validated once at parse and
// the per-call re-check there is pure overhead (PERF-002 / c67edc1c). The curve
// equation costs ~1S+2M, negligible off the verify hot loop.
[[nodiscard]] inline Point pubkey_data_to_point_checked(const unsigned char data[64]) noexcept {
    const auto& xb = *reinterpret_cast<const std::array<uint8_t, 32>*>(data);
    const auto& yb = *reinterpret_cast<const std::array<uint8_t, 32>*>(data + 32);
    FieldElement x, y;
    if (!FieldElement::parse_bytes_strict(xb, x)) return Point::infinity();
    if (!FieldElement::parse_bytes_strict(yb, y)) return Point::infinity();
    FieldElement const b7 = FieldElement::from_uint64(7);
    if (y * y != x * x * x + b7) return Point::infinity();   // off-curve / (0,0)
    return Point::from_affine(x, y);
}

} // namespace secp256k1_shim_internal
