// ============================================================================
// Affine Batch Addition -- Fastest CPU pipeline for sequential ECC search
// ============================================================================
// Algorithm: Given base point P (affine) and N offset points T[i] (affine),
// compute P + T[i] using Montgomery batch inversion on dx values.
//
// Cost per point: ~6M + 1S ~= 150 ns (vs 463 ns for Jacobian pipeline)
// Performance: 3x faster than Jacobian mixed-add + batch Z-inverse
// ============================================================================

#include "secp256k1/batch_add_affine.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/precompute.hpp"

#include <cstring>

namespace secp256k1::fast {

// ============================================================================
// Core implementation: batch_add_affine_x
// ============================================================================

void batch_add_affine_x(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    std::size_t count,
    std::vector<FieldElement>& scratch)
{
    if (count == 0) return;

    // scratch is used for dx values -> then inverted in-place
    if (scratch.size() < count) {
        scratch.resize(count);
    }

    const FieldElement zero = FieldElement::zero();

    // Phase 1: Compute dx[i] = x_T[i] - x_base
    //          Replace any zeros with ONE to avoid corrupting batch inverse chain.
    //          Zero dx means P == +/-T[i] (astronomically rare in search; ~2^{-128}).
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx = offsets[i].x - base_x;
        // Branchless: if dx==0, substitute ONE so batch inverse stays valid.
        // We detect and sentinel the output in Phase 3.
        bool const is_zero = (dx == zero);
        scratch[i] = is_zero ? FieldElement::one() : dx;
    }

    // Phase 2: Montgomery batch inversion on dx values
    // Cost: 3*(N-1) multiplications + 1 inversion
    fe_batch_inverse(scratch.data(), count);

    // Phase 3: Compute affine addition for each offset
    // lambda[i] = (y_T[i] - y_base) * dx_inv[i]
    // x3[i] = lambda^2 - x_base - x_T[i]
    for (std::size_t i = 0; i < count; ++i) {
        // Handle degenerate case: original dx was zero (P == T[i] or P == -T[i])
        FieldElement const dx_original = offsets[i].x - base_x;
        if (dx_original == zero) {
            out_x[i] = zero;  // Sentinel: never a valid curve X
            continue;
        }

        FieldElement const dy = offsets[i].y - base_y;      // dy = y_T - y_base
        FieldElement const lambda = dy * scratch[i];         // lambda = dy / dx  [1M]
        FieldElement lambda_sq = lambda;
        lambda_sq.square_inplace();                    // lambda^2            [1S]
        out_x[i] = lambda_sq - base_x - offsets[i].x; // x3 = lambda^2 - x_P - x_T
    }
}

// ============================================================================
// Full XY output variant
// ============================================================================

void batch_add_affine_xy(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    FieldElement* out_y,
    std::size_t count,
    std::vector<FieldElement>& scratch)
{
    if (count == 0) return;

    if (scratch.size() < count) {
        scratch.resize(count);
    }

    const FieldElement zero = FieldElement::zero();

    // Phase 1: dx[i] = x_T[i] - x_base (zero-safe: replace 0 -> 1)
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx = offsets[i].x - base_x;
        bool const is_zero = (dx == zero);
        scratch[i] = is_zero ? FieldElement::one() : dx;
    }

    // Phase 2: Batch inverse
    fe_batch_inverse(scratch.data(), count);

    // Phase 3: Full affine addition
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx_original = offsets[i].x - base_x;
        if (dx_original == zero) {
            out_x[i] = zero;
            out_y[i] = zero;
            continue;
        }

        FieldElement const dy = offsets[i].y - base_y;
        FieldElement const lambda = dy * scratch[i];         // lambda = dy/dx     [1M]
        FieldElement lambda_sq = lambda;
        lambda_sq.square_inplace();                    // lambda^2            [1S]
        FieldElement const x3 = lambda_sq - base_x - offsets[i].x;
        FieldElement const y3 = lambda * (base_x - x3) - base_y;  // [2M]

        out_x[i] = x3;
        out_y[i] = y3;
    }
}

// ============================================================================
// Convenience wrapper (internal scratch)
// ============================================================================

void batch_add_affine_x(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    std::size_t count)
{
    std::vector<FieldElement> scratch(count);
    batch_add_affine_x(base_x, base_y, offsets, out_x, count, scratch);
}

// ============================================================================
// Y-parity extraction
// ============================================================================

void batch_add_affine_x_with_parity(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    uint8_t* out_parity,
    std::size_t count,
    std::vector<FieldElement>& scratch)
{
    if (count == 0) return;

    if (scratch.size() < count) {
        scratch.resize(count);
    }

    const FieldElement zero = FieldElement::zero();

    // Phase 1: dx (zero-safe: replace 0 -> 1)
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx = offsets[i].x - base_x;
        bool const is_zero = (dx == zero);
        scratch[i] = is_zero ? FieldElement::one() : dx;
    }

    // Phase 2: Batch inverse
    fe_batch_inverse(scratch.data(), count);

    // Phase 3: Addition + Y parity
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx_original = offsets[i].x - base_x;
        if (dx_original == zero) {
            out_x[i] = zero;
            out_parity[i] = 0;
            continue;
        }

        FieldElement const dy = offsets[i].y - base_y;
        FieldElement const lambda = dy * scratch[i];
        FieldElement lambda_sq = lambda;
        lambda_sq.square_inplace();
        FieldElement const x3 = lambda_sq - base_x - offsets[i].x;
        FieldElement const y3 = lambda * (base_x - x3) - base_y;

        out_x[i] = x3;

        // Y parity: lowest bit of y (big-endian byte 31)
        auto y_bytes = y3.to_bytes();
        out_parity[i] = y_bytes[31] & 1;
    }
}

// ============================================================================
// Bidirectional batch add
// ============================================================================

void batch_add_affine_x_bidirectional(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets_fwd,
    const AffinePointCompact* offsets_bwd,
    FieldElement* out_x_fwd,
    FieldElement* out_x_bwd,
    std::size_t count,
    std::vector<FieldElement>& scratch)
{
    if (count == 0) return;

    // Need 2*count scratch space: [0..count-1] for fwd, [count..2count-1] for bwd
    const std::size_t total = count * 2;
    if (scratch.size() < total) {
        scratch.resize(total);
    }

    const FieldElement zero = FieldElement::zero();

    // Phase 1: dx for both directions (zero-safe: replace 0 -> 1)
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx_fwd = offsets_fwd[i].x - base_x;
        FieldElement const dx_bwd = offsets_bwd[i].x - base_x;
        scratch[i]         = (dx_fwd == zero) ? FieldElement::one() : dx_fwd;
        scratch[count + i] = (dx_bwd == zero) ? FieldElement::one() : dx_bwd;
    }

    // Phase 2: Single batch inverse over all 2*count dx values
    fe_batch_inverse(scratch.data(), total);

    // Phase 3: Forward results
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx_original = offsets_fwd[i].x - base_x;
        if (dx_original == zero) {
            out_x_fwd[i] = zero;
            continue;
        }
        FieldElement const dy = offsets_fwd[i].y - base_y;
        FieldElement const lambda = dy * scratch[i];
        FieldElement lambda_sq = lambda;
        lambda_sq.square_inplace();
        out_x_fwd[i] = lambda_sq - base_x - offsets_fwd[i].x;
    }

    // Phase 4: Backward results
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx_original = offsets_bwd[i].x - base_x;
        if (dx_original == zero) {
            out_x_bwd[i] = zero;
            continue;
        }
        FieldElement const dy = offsets_bwd[i].y - base_y;
        FieldElement const lambda = dy * scratch[count + i];
        FieldElement lambda_sq = lambda;
        lambda_sq.square_inplace();
        out_x_bwd[i] = lambda_sq - base_x - offsets_bwd[i].x;
    }
}

// ============================================================================
// Precomputed Generator Table
// ============================================================================

std::vector<AffinePointCompact> precompute_g_multiples(std::size_t count) {
    if (count == 0) return {};

    std::vector<AffinePointCompact> table(count);

    // Compute [G, 2G, 3G, ..., count*G] in Jacobian, then batch-convert to affine
    // Use existing library: scalar_mul_generator for first point, then add G
    Point current = Point::generator();  // 1*G

    // Collect Jacobian Z-coordinates for batch inverse
    std::vector<FieldElement> jac_x(count);
    std::vector<FieldElement> jac_y(count);
    std::vector<FieldElement> jac_z(count);

    jac_x[0] = current.X();
    jac_y[0] = current.Y();
    jac_z[0] = current.z();

    for (std::size_t i = 1; i < count; ++i) {
        current.next_inplace();  // (i+1)*G
        jac_x[i] = current.X();
        jac_y[i] = current.Y();
        jac_z[i] = current.z();
    }

    // Batch inverse all Z coordinates
    fe_batch_inverse(jac_z.data(), count);

    // Convert Jacobian -> Affine: x_aff = X * Z^{-2}, y_aff = Y * Z^{-3}
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const z_inv = jac_z[i];
        FieldElement z_inv2 = z_inv;
        z_inv2.square_inplace();        // Z^{-2}
        FieldElement const z_inv3 = z_inv2 * z_inv;  // Z^{-3}

        table[i].x = jac_x[i] * z_inv2;
        table[i].y = jac_y[i] * z_inv3;
    }

    return table;
}

std::vector<AffinePointCompact> precompute_point_multiples(
    const FieldElement& qx, const FieldElement& qy, std::size_t count)
{
    if (count == 0) return {};

    std::vector<AffinePointCompact> table(count);

    // Start with Q as affine, convert to Jacobian for additions
    Point current = Point::from_affine(qx, qy);

    std::vector<FieldElement> jac_x(count);
    std::vector<FieldElement> jac_y(count);
    std::vector<FieldElement> jac_z(count);

    jac_x[0] = current.X();
    jac_y[0] = current.Y();
    jac_z[0] = current.z();

    // Q is affine, so we do mixed-add with Q for 2Q, 3Q, ...
    for (std::size_t i = 1; i < count; ++i) {
        current.add_mixed_inplace(qx, qy);  // += Q
        jac_x[i] = current.X();
        jac_y[i] = current.Y();
        jac_z[i] = current.z();
    }

    // Batch inverse Z
    fe_batch_inverse(jac_z.data(), count);

    // Convert to affine
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const z_inv = jac_z[i];
        FieldElement z_inv2 = z_inv;
        z_inv2.square_inplace();
        FieldElement const z_inv3 = z_inv2 * z_inv;

        table[i].x = jac_x[i] * z_inv2;
        table[i].y = jac_y[i] * z_inv3;
    }

    return table;
}

// ============================================================================
// Negate table
// ============================================================================

std::vector<AffinePointCompact> negate_affine_table(
    const AffinePointCompact* table, std::size_t count)
{
    std::vector<AffinePointCompact> neg(count);
    const FieldElement zero = FieldElement::zero();

    for (std::size_t i = 0; i < count; ++i) {
        neg[i].x = table[i].x;
        neg[i].y = zero - table[i].y;  // -y mod p
    }

    return neg;
}

} // namespace secp256k1::fast
