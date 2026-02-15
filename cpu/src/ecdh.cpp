#include "secp256k1/ecdh.hpp"
#include "secp256k1/sha256.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// ── ECDH: SHA-256(compressed point) ──────────────────────────────────────────

std::array<std::uint8_t, 32> ecdh_compute(
    const Scalar& private_key,
    const Point& public_key) {

    if (private_key.is_zero()) return {};

    auto shared_point = public_key.scalar_mul(private_key);
    if (shared_point.is_infinity()) return {};

    // Serialize as compressed point (33 bytes: 02/03 prefix + x)
    auto compressed = shared_point.to_compressed();

    // Hash with SHA-256
    return SHA256::hash(compressed.data(), compressed.size());
}

// ── ECDH: SHA-256(x-coordinate) ──────────────────────────────────────────────

std::array<std::uint8_t, 32> ecdh_compute_xonly(
    const Scalar& private_key,
    const Point& public_key) {

    if (private_key.is_zero()) return {};

    auto shared_point = public_key.scalar_mul(private_key);
    if (shared_point.is_infinity()) return {};

    // x-coordinate only
    auto x_bytes = shared_point.x().to_bytes();

    return SHA256::hash(x_bytes.data(), x_bytes.size());
}

// ── ECDH: Raw x-coordinate ──────────────────────────────────────────────────

std::array<std::uint8_t, 32> ecdh_compute_raw(
    const Scalar& private_key,
    const Point& public_key) {

    if (private_key.is_zero()) return {};

    auto shared_point = public_key.scalar_mul(private_key);
    if (shared_point.is_infinity()) return {};

    return shared_point.x().to_bytes();
}

} // namespace secp256k1
