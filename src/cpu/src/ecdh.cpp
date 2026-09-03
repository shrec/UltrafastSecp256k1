#include "secp256k1/ecdh.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- ECDH: SHA-256(compressed point) ------------------------------------------

std::array<std::uint8_t, 32> ecdh_compute(
    const Scalar& private_key,
    const Point& public_key) {

    if (private_key.is_zero_ct()) return {};

    // SEC-005: reject off-curve pubkeys (invalid-curve attack defense).
    if (public_key.is_infinity()) return {};
    {
        // x() and y() each invert Z, so a Jacobian pubkey pays two inversions
        // for one curve check. Normalize a local copy once and read both
        // coordinates off the affine result.
        Point affine = public_key;
        affine.normalize();
        auto px = affine.x(), py = affine.y();
        auto rhs = px.square() * px + FieldElement::from_uint64(7);
        if (!(py.square() == rhs)) return {};
    }

    auto shared_point = ct::scalar_mul(public_key, private_key);
    if (shared_point.is_infinity()) return {};

    // Serialize as compressed point (33 bytes: 02/03 prefix + x)
    auto compressed = shared_point.to_compressed();

    // Hash with SHA-256
    auto result = SHA256::hash(compressed.data(), compressed.size());
    secp256k1::detail::secure_erase(compressed.data(), compressed.size());
    secp256k1::detail::secure_erase(&shared_point, sizeof(shared_point));
    return result;
}

// -- ECDH: SHA-256(x-coordinate) ----------------------------------------------

std::array<std::uint8_t, 32> ecdh_compute_xonly(
    const Scalar& private_key,
    const Point& public_key) {

    if (private_key.is_zero_ct()) return {};

    // SEC-005: reject off-curve pubkeys (invalid-curve attack defense).
    if (public_key.is_infinity()) return {};
    {
        // x() and y() each invert Z, so a Jacobian pubkey pays two inversions
        // for one curve check. Normalize a local copy once and read both
        // coordinates off the affine result.
        Point affine = public_key;
        affine.normalize();
        auto px = affine.x(), py = affine.y();
        auto rhs = px.square() * px + FieldElement::from_uint64(7);
        if (!(py.square() == rhs)) return {};
    }

    auto shared_point = ct::scalar_mul(public_key, private_key);
    if (shared_point.is_infinity()) return {};

    // x-coordinate only
    auto x_bytes = shared_point.x().to_bytes();

    auto result = SHA256::hash(x_bytes.data(), x_bytes.size());
    secp256k1::detail::secure_erase(x_bytes.data(), x_bytes.size());
    secp256k1::detail::secure_erase(&shared_point, sizeof(shared_point));
    return result;
}

// -- ECDH: Raw x-coordinate --------------------------------------------------

std::array<std::uint8_t, 32> ecdh_compute_raw(
    const Scalar& private_key,
    const Point& public_key) {

    if (private_key.is_zero_ct()) return {};

    // SEC-005: reject off-curve pubkeys (invalid-curve attack defense).
    if (public_key.is_infinity()) return {};
    {
        // x() and y() each invert Z, so a Jacobian pubkey pays two inversions
        // for one curve check. Normalize a local copy once and read both
        // coordinates off the affine result.
        Point affine = public_key;
        affine.normalize();
        auto px = affine.x(), py = affine.y();
        auto rhs = px.square() * px + FieldElement::from_uint64(7);
        if (!(py.square() == rhs)) return {};
    }

    auto shared_point = ct::scalar_mul(public_key, private_key);
    if (shared_point.is_infinity()) return {};

    auto result = shared_point.x().to_bytes();
    secp256k1::detail::secure_erase(&shared_point, sizeof(shared_point));
    return result;
}

} // namespace secp256k1
