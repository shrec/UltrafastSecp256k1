// ============================================================================
// Pedersen Commitments — Implementation
// ============================================================================

#include "secp256k1/pedersen.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/field.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Point;
using fast::Scalar;
using fast::FieldElement;

// ── Nothing-up-my-sleeve generators ──────────────────────────────────────────

// Modular sqrt: y = a^((p+1)/4) mod p, valid since p ≡ 3 (mod 4)
static FieldElement field_sqrt(const FieldElement& a) {
    auto exp = FieldElement::from_hex(
        "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");
    auto y = FieldElement::one();
    auto base = a;
    auto exp_bytes = exp.to_bytes();
    for (int i = 0; i < 256; ++i) {
        y = y.square();
        int byte_idx = i / 8;
        int bit_idx = 7 - (i % 8);
        if ((exp_bytes[byte_idx] >> bit_idx) & 1) {
            y = y * base;
        }
    }
    return y;
}

// lift_x helper: find point with given x-coordinate and even y
// Uses try-and-increment if x is not a valid x-coordinate on the curve
static Point lift_x_even(const FieldElement& x_in) {
    FieldElement x = x_in;
    for (int attempt = 0; attempt < 256; ++attempt) {
        // y² = x³ + 7
        FieldElement x2 = x * x;
        FieldElement x3 = x2 * x;
        FieldElement rhs = x3 + FieldElement::from_uint64(7);

        FieldElement y = field_sqrt(rhs);

        // Verify: y² == rhs (check that sqrt succeeded)
        if (y.square() == rhs) {
            // Ensure even y
            auto y_bytes = y.to_bytes();
            if (y_bytes[31] & 1) {
                y = FieldElement::zero() - y;
            }
            return Point::from_affine(x, y);
        }

        // Try next x: x = x + 1
        x = x + FieldElement::one();
    }
    // Should never happen — ~50% chance each attempt, 256 attempts
    return Point::infinity();
}

const Point& pedersen_generator_H() {
    static const Point H = []() {
        // H = lift_x(SHA256("Pedersen_generator_H"))
        SHA256 hasher;
        const char* tag = "Pedersen_generator_H";
        hasher.update(tag, std::strlen(tag));
        auto hash = hasher.finalize();
        FieldElement x = FieldElement::from_bytes(hash);
        return lift_x_even(x);
    }();
    return H;
}

const Point& pedersen_generator_J() {
    static const Point J = []() {
        // J = lift_x(SHA256("Pedersen_switch_J"))
        SHA256 hasher;
        const char* tag = "Pedersen_switch_J";
        hasher.update(tag, std::strlen(tag));
        auto hash = hasher.finalize();
        FieldElement x = FieldElement::from_bytes(hash);
        return lift_x_even(x);
    }();
    return J;
}

// ── PedersenCommitment methods ───────────────────────────────────────────────

std::array<std::uint8_t, 33> PedersenCommitment::to_compressed() const {
    return point.to_compressed();
}

PedersenCommitment PedersenCommitment::operator+(const PedersenCommitment& rhs) const {
    return PedersenCommitment{point.add(rhs.point)};
}

bool PedersenCommitment::verify(const Scalar& value, const Scalar& blinding) const {
    return pedersen_verify(*this, value, blinding);
}

// ── Commit / Open ────────────────────────────────────────────────────────────

PedersenCommitment pedersen_commit(const Scalar& value, const Scalar& blinding) {
    // C = v*H + r*G
    const Point& H = pedersen_generator_H();
    Point vH = H.scalar_mul(value);
    Point rG = Point::generator().scalar_mul(blinding);
    return PedersenCommitment{vH.add(rG)};
}

bool pedersen_verify(const PedersenCommitment& commitment,
                     const Scalar& value,
                     const Scalar& blinding) {
    PedersenCommitment expected = pedersen_commit(value, blinding);
    auto c1 = commitment.to_compressed();
    auto c2 = expected.to_compressed();
    return c1 == c2;
}

// ── Homomorphic Operations ───────────────────────────────────────────────────

bool pedersen_verify_sum(const PedersenCommitment* commitments_pos,
                         std::size_t n_pos,
                         const PedersenCommitment* commitments_neg,
                         std::size_t n_neg) {
    Point sum = Point::infinity();

    for (std::size_t i = 0; i < n_pos; ++i) {
        sum = sum.add(commitments_pos[i].point);
    }
    for (std::size_t i = 0; i < n_neg; ++i) {
        sum = sum.add(commitments_neg[i].point.negate());
    }

    return sum.is_infinity();
}

Scalar pedersen_blind_sum(const Scalar* blinds_in,
                          std::size_t n_in,
                          const Scalar* blinds_out,
                          std::size_t n_out) {
    Scalar sum = Scalar::zero();
    for (std::size_t i = 0; i < n_in; ++i) {
        sum = sum + blinds_in[i];
    }
    for (std::size_t i = 0; i < n_out; ++i) {
        sum = sum - blinds_out[i];
    }
    return sum;
}

// ── Switch Commitment ────────────────────────────────────────────────────────

PedersenCommitment pedersen_switch_commit(const Scalar& value,
                                          const Scalar& blinding,
                                          const Scalar& switch_blind) {
    // C = v*H + r*G + s*J
    const Point& H = pedersen_generator_H();
    const Point& J = pedersen_generator_J();
    Point vH = H.scalar_mul(value);
    Point rG = Point::generator().scalar_mul(blinding);
    Point sJ = J.scalar_mul(switch_blind);
    return PedersenCommitment{vH.add(rG).add(sJ)};
}

} // namespace secp256k1
