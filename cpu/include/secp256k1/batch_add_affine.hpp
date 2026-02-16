#ifndef SECP256K1_BATCH_ADD_AFFINE_HPP
#define SECP256K1_BATCH_ADD_AFFINE_HPP

#include "field.hpp"
#include <cstddef>
#include <vector>

namespace secp256k1::fast {

// ============================================================================
// AFFINE BATCH ADDITION — Fastest CPU pipeline for sequential ECC search
// ============================================================================
//
// ## THE IDEA
// Given a base point P = (x_base, y_base) in AFFINE coordinates and N
// precomputed offsets T[0..N-1] = (x_i, y_i) also in AFFINE, compute
// all N sums: result[i] = P + T[i], returning AFFINE X-coordinates.
//
// ## WHY THIS IS FAST
// Standard Jacobian mixed-add: P += G costs 7M + 4S = ~354 ns/point
// with subsequent batch Z-inverse adding ~62 ns → total ~463 ns/point.
//
// Affine batch add uses Montgomery batch inversion on dx values:
//   dx[i] = x_T[i] - x_base               → 1 sub
//   batch_inverse(dx, N)                    → 3(N-1) mul + 1 inv ≈ 69 ns/pt
//   λ[i]  = (y_T[i] - y_base) * dx_inv[i]  → 1 mul
//   x3[i] = λ[i]² - x_base - x_T[i]        → 1 sqr + 2 sub
//   y3[i] = λ[i] * (x_base - x3[i]) - y_base → 2 mul + 1 sub
//
// Cost: ~6M + 1S per point ≈ 150 ns/point → 3× faster than Jacobian!
//
// ## USE CASE
// Satoshi puzzle / vanity search: walk through key range sequentially.
// Pre-compute T[i] = i*G in affine, then for each batch:
//   1. P_base = start_scalar * G (compute once per batch)
//   2. result[i] = P_base + T[i]  (affine batch add — this function!)
//   3. Check result X-coordinates against target(s)
//   4. P_base += B*G (advance by batch size)
//
// ============================================================================

/// Compact affine point for table storage (no Z coordinate, no infinity flag)
/// POD type: trivially copyable, cache-line friendly (64 bytes = 1 cache line)
struct alignas(64) AffinePointCompact {
    FieldElement x;
    FieldElement y;
};

// ============================================================================
// Core batch addition API
// ============================================================================

/// Compute P + T[i] for all i ∈ [0, count), returning affine X-coordinates.
///
/// @param base_x     Base point X coordinate (affine)
/// @param base_y     Base point Y coordinate (affine)
/// @param offsets    Array of precomputed affine offset points T[0..count-1]
/// @param out_x      Output: X-coordinates of P + T[i] (caller-allocated, size ≥ count)
/// @param count      Number of offset points
/// @param scratch    Reusable scratch buffer (avoids allocation; resized internally if needed)
///
/// @note Edge cases (P == T[i] or P == -T[i]) are handled via branchless sentinel:
///       output X is set to FieldElement::zero() which is never a valid curve X.
///       Caller should skip zero results if exactness matters (astronomically rare in search).
///
/// Hot-path contract: No heap allocation when scratch is pre-sized ≥ count.
void batch_add_affine_x(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    std::size_t count,
    std::vector<FieldElement>& scratch);

/// Same as above but also outputs Y-coordinates.
///
/// @param out_y  Output: Y-coordinates of P + T[i] (caller-allocated, size ≥ count)
void batch_add_affine_xy(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    FieldElement* out_y,
    std::size_t count,
    std::vector<FieldElement>& scratch);

/// Convenience: wraps batch_add_affine_x with internal scratch buffer.
/// Slightly slower due to potential reallocation; prefer the scratch version for hot loops.
void batch_add_affine_x(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    std::size_t count);

// ============================================================================
// Precomputed Generator Table
// ============================================================================

/// Build a table of G-multiples: T[i] = (i+1)*G in affine coordinates.
/// T[0] = 1*G, T[1] = 2*G, ..., T[count-1] = count*G
///
/// @param count  Number of multiples to precompute (= batch size B)
/// @return Vector of affine G-multiples
///
/// @note This is a one-time cost at startup. For batch_size=1024:
///       ~370 μs (1024 point additions + 1 batch inverse).
///       Table size: 1024 * 64 = 64 KB (fits in L1 cache!).
std::vector<AffinePointCompact> precompute_g_multiples(std::size_t count);

/// Build a table of multiples of an arbitrary affine point Q:
/// T[i] = (i+1)*Q in affine. Useful for non-generator walks.
std::vector<AffinePointCompact> precompute_point_multiples(
    const FieldElement& qx, const FieldElement& qy, std::size_t count);

// ============================================================================
// Full search pipeline helpers
// ============================================================================

/// Bidirectional affine batch add: compute both forward and backward offsets.
/// Forward:  result_fwd[i] = P + T[i]       (i = 0..count-1)
/// Backward: result_bwd[i] = P - T[i]       (using negated Y)
///
/// @param neg_offsets  Precomputed T[i] with negated Y (T_neg[i].y = -T[i].y)
///
/// This doubles throughput by checking 2*count keys per batch with minimal
/// extra cost (the negated table is precomputed once at startup).
void batch_add_affine_x_bidirectional(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets_fwd,
    const AffinePointCompact* offsets_bwd,
    FieldElement* out_x_fwd,
    FieldElement* out_x_bwd,
    std::size_t count,
    std::vector<FieldElement>& scratch);

/// Build negated table: T_neg[i] = (T[i].x, -T[i].y)
/// One-time cost, returns new vector.
std::vector<AffinePointCompact> negate_affine_table(
    const AffinePointCompact* table, std::size_t count);

// ============================================================================
// Y-parity extraction (for compressed pubkey byte without full Y)
// ============================================================================

/// Extract Y-parity bit for each result point.
/// parity[i] = lowest bit of Y-coordinate (0x02 or 0x03 prefix).
/// Use when you need compressed pubkey prefix without computing full Y.
///
/// @param out_parity  Output: 0 for even Y, 1 for odd Y (caller-allocated, size ≥ count)
void batch_add_affine_x_with_parity(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    uint8_t* out_parity,
    std::size_t count,
    std::vector<FieldElement>& scratch);

} // namespace secp256k1::fast

#endif // SECP256K1_BATCH_ADD_AFFINE_HPP
