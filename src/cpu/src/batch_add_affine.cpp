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
#include "secp256k1/config.hpp"
#if defined(SECP256K1_FAST_52BIT)
#include "secp256k1/field_52.hpp"
#endif

#include <array>
#include <cstring>

namespace secp256k1::fast {

namespace {

constexpr std::size_t kSmallPrecomputeTable = 64;
constexpr std::size_t kSmallBatchAddScratch = 64;

struct PrecomputeBuffers {
    std::array<FieldElement, kSmallPrecomputeTable> jac_x_stack{};
    std::array<FieldElement, kSmallPrecomputeTable> jac_y_stack{};
    std::array<FieldElement, kSmallPrecomputeTable> jac_z_stack{};
    std::vector<FieldElement> jac_x_heap;
    std::vector<FieldElement> jac_y_heap;
    std::vector<FieldElement> jac_z_heap;

    FieldElement* x(std::size_t count) {
        if (count <= kSmallPrecomputeTable) return jac_x_stack.data();
        jac_x_heap.resize(count);
        return jac_x_heap.data();
    }

    FieldElement* y(std::size_t count) {
        if (count <= kSmallPrecomputeTable) return jac_y_stack.data();
        jac_y_heap.resize(count);
        return jac_y_heap.data();
    }

    FieldElement* z(std::size_t count) {
        if (count <= kSmallPrecomputeTable) return jac_z_stack.data();
        jac_z_heap.resize(count);
        return jac_z_heap.data();
    }
};

void batch_add_affine_x_impl(
    const FieldElement& base_x,
    const FieldElement& base_y,
    const AffinePointCompact* offsets,
    FieldElement* out_x,
    std::size_t count,
    FieldElement* scratch)
{
    // First loop: validate all peer pubkeys BEFORE loading dx into scratch.
    // Use a bool array to record which entries are degenerate (dx == 0),
    // avoiding the second full field subtraction in the output loop.
    const FieldElement zero = FieldElement::zero();
    const FieldElement one  = FieldElement::one();

    // Stack array for typical small counts; heap for large batches.
    uint8_t dx_zero_stack[64] = {};
    static thread_local std::vector<uint8_t> dx_zero_heap;
    uint8_t* dx_zero = dx_zero_stack;
    if (SECP256K1_UNLIKELY(count > 64)) {
        if (dx_zero_heap.size() < count) dx_zero_heap.resize(count);
        dx_zero = dx_zero_heap.data();
    }

    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx = offsets[i].x - base_x;
        dx_zero[i] = static_cast<uint8_t>(dx == zero);
        scratch[i] = dx_zero[i] ? one : dx;
    }

    // scratch[i] is guaranteed nonzero: zero dx slots replaced with 1 above.
    fe_batch_inverse_nonzero(scratch, count);

    // Phase 2: lambda = dy * inv_dx; x3 = lambda^2 - base_x - ox.
    // FE52 path: mul ~23 ns + sqr ~17 ns vs 4x64: mul ~50 ns + sqr ~39 ns.
    // base_x/y converted once before loop; offsets/scratch converted per entry.
#if defined(SECP256K1_FAST_52BIT)
    using FE52 = FieldElement52;
    FE52 const neg_bx = FE52::from_fe(base_x).negate(1);  // -base_x, mag 2
    FE52 const neg_by = FE52::from_fe(base_y).negate(1);  // -base_y, mag 2
    for (std::size_t i = 0; i < count; ++i) {
        if (SECP256K1_UNLIKELY(dx_zero[i])) { out_x[i] = zero; continue; }
        FE52 const inv52 = FE52::from_fe(scratch[i]);
        FE52 const ox52  = FE52::from_fe(offsets[i].x);
        FE52 const dy52  = FE52::from_fe(offsets[i].y) + neg_by;  // mag 3
        FE52 const lam   = dy52 * inv52;                           // 1M
        FE52 const lsq   = lam.square();                           // 1S
        FE52 x3 = lsq + neg_bx + ox52.negate(1);                  // mag 4
        out_x[i] = x3.to_fe();
    }
#else
    for (std::size_t i = 0; i < count; ++i) {
        if (SECP256K1_UNLIKELY(dx_zero[i])) { out_x[i] = zero; continue; }
        FieldElement const dy = offsets[i].y - base_y;
        FieldElement const lambda = dy * scratch[i];
        FieldElement lambda_sq = lambda;
        lambda_sq.square_inplace();
        out_x[i] = lambda_sq - base_x - offsets[i].x;
    }
#endif
}

} // namespace

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

    if (scratch.size() < count) {
        scratch.resize(count);
    }

    batch_add_affine_x_impl(base_x, base_y, offsets, out_x, count,
                            scratch.data());
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
    const FieldElement one  = FieldElement::one();

    // Stack array for typical small counts; heap for large batches.
    uint8_t dx_zero_stack[64] = {};
    static thread_local std::vector<uint8_t> dx_zero_heap_xy;
    uint8_t* dx_zero = dx_zero_stack;
    if (SECP256K1_UNLIKELY(count > 64)) {
        if (dx_zero_heap_xy.size() < count) dx_zero_heap_xy.resize(count);
        dx_zero = dx_zero_heap_xy.data();
    }

    // Phase 1: dx[i] = x_T[i] - x_base (zero-safe: replace 0 -> 1)
    // Save is-zero flag to avoid recomputing dx in Phase 3.
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx = offsets[i].x - base_x;
        dx_zero[i] = static_cast<uint8_t>(dx == zero);
        scratch[i] = dx_zero[i] ? one : dx;
    }

    // Phase 2: Batch inverse (scratch guaranteed nonzero: zero slots replaced with 1 above)
    fe_batch_inverse_nonzero(scratch.data(), count);

    // Phase 3: Full affine addition — reuse dx_zero flags, no dx recomputation.
    // FE52: mul ~23 ns + sqr ~17 ns + extra mul ~23 ns vs 4x64: ~50+39+50 = ~139 ns.
    // Magnitude tracking: lsq(1) + neg_bx(2) + ox.negate(1)(2) = x3 mag 5.
    // y3 uses x3.negate(5) = 6p - x3 (not negate(1) = 2p - x3 which would underflow).
#if defined(SECP256K1_FAST_52BIT)
    {
        using FE52 = FieldElement52;
        FE52 const bx52   = FE52::from_fe(base_x);
        FE52 const neg_bx = bx52.negate(1);              // mag 2
        FE52 const neg_by = FE52::from_fe(base_y).negate(1);
        for (std::size_t i = 0; i < count; ++i) {
            if (SECP256K1_UNLIKELY(dx_zero[i])) {
                out_x[i] = zero; out_y[i] = zero; continue;
            }
            FE52 const inv52 = FE52::from_fe(scratch[i]);
            FE52 const ox52  = FE52::from_fe(offsets[i].x);
            FE52 const dy52  = FE52::from_fe(offsets[i].y) + neg_by;
            FE52 const lam   = dy52 * inv52;                         // 1M
            FE52 const lsq   = lam.square();                         // 1S
            FE52 x3 = lsq + neg_bx + ox52.negate(1);                // mag 5
            // x3.negate(5) = 6p - x3, correct for mag-5 x3
            FE52 const y3 = lam * (bx52 + x3.negate(5)) + neg_by;   // 1M
            out_x[i] = x3.to_fe();
            out_y[i] = y3.to_fe();
        }
    }
#else
    for (std::size_t i = 0; i < count; ++i) {
        if (SECP256K1_UNLIKELY(dx_zero[i])) {
            out_x[i] = zero; out_y[i] = zero; continue;
        }
        FieldElement const dy = offsets[i].y - base_y;
        FieldElement const lambda = dy * scratch[i];
        FieldElement lambda_sq = lambda;
        lambda_sq.square_inplace();
        FieldElement const x3 = lambda_sq - base_x - offsets[i].x;
        FieldElement const y3 = lambda * (base_x - x3) - base_y;
        out_x[i] = x3;
        out_y[i] = y3;
    }
#endif
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
    if (count == 0) return;

    if (count <= kSmallBatchAddScratch) {
        std::array<FieldElement, kSmallBatchAddScratch> scratch{};
        batch_add_affine_x_impl(base_x, base_y, offsets, out_x, count,
                                scratch.data());
        return;
    }

    static thread_local std::vector<FieldElement> tls_scratch;
    if (tls_scratch.size() < count) tls_scratch.resize(count);
    batch_add_affine_x_impl(base_x, base_y, offsets, out_x, count,
                            tls_scratch.data());
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
    const FieldElement one  = FieldElement::one();

    // Stack array for typical small counts; heap for large batches.
    uint8_t dx_zero_stack[64] = {};
    static thread_local std::vector<uint8_t> dx_zero_heap_parity;
    uint8_t* dx_zero = dx_zero_stack;
    if (SECP256K1_UNLIKELY(count > 64)) {
        if (dx_zero_heap_parity.size() < count) dx_zero_heap_parity.resize(count);
        dx_zero = dx_zero_heap_parity.data();
    }

    // Phase 1: dx (zero-safe: replace 0 -> 1)
    // Save is-zero flag to avoid recomputing dx in Phase 3.
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx = offsets[i].x - base_x;
        dx_zero[i] = static_cast<uint8_t>(dx == zero);
        scratch[i] = dx_zero[i] ? one : dx;
    }

    // Phase 2: Batch inverse (scratch guaranteed nonzero: zero slots replaced with 1 above)
    fe_batch_inverse_nonzero(scratch.data(), count);

    // Phase 3: Addition + Y parity — reuse dx_zero flags, no dx recomputation.
    // Same magnitude fix as XY variant: x3.negate(5) for correct 6p - x3.
#if defined(SECP256K1_FAST_52BIT)
    {
        using FE52 = FieldElement52;
        FE52 const bx52   = FE52::from_fe(base_x);
        FE52 const neg_bx = bx52.negate(1);
        FE52 const neg_by = FE52::from_fe(base_y).negate(1);
        for (std::size_t i = 0; i < count; ++i) {
            if (SECP256K1_UNLIKELY(dx_zero[i])) {
                out_x[i] = zero; out_parity[i] = 0; continue;
            }
            FE52 const inv52 = FE52::from_fe(scratch[i]);
            FE52 const ox52  = FE52::from_fe(offsets[i].x);
            FE52 const dy52  = FE52::from_fe(offsets[i].y) + neg_by;
            FE52 const lam   = dy52 * inv52;
            FE52 const lsq   = lam.square();
            FE52 x3 = lsq + neg_bx + ox52.negate(1);                // mag 5
            FE52 y3 = lam * (bx52 + x3.negate(5)) + neg_by;         // x3.negate(5) = 6p - x3
            y3.normalize();
            out_x[i]      = x3.to_fe();
            out_parity[i] = static_cast<uint8_t>(y3.n[0] & 1U);
        }
    }
#else
    for (std::size_t i = 0; i < count; ++i) {
        if (SECP256K1_UNLIKELY(dx_zero[i])) {
            out_x[i] = zero; out_parity[i] = 0; continue;
        }
        FieldElement const dy = offsets[i].y - base_y;
        FieldElement const lambda = dy * scratch[i];
        FieldElement lambda_sq = lambda;
        lambda_sq.square_inplace();
        FieldElement const x3 = lambda_sq - base_x - offsets[i].x;
        FieldElement const y3 = lambda * (base_x - x3) - base_y;
        out_x[i] = x3;
        out_parity[i] = static_cast<uint8_t>(y3.limbs()[0] & 1U);
    }
#endif
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
    const FieldElement one  = FieldElement::one();

    // Stack arrays for is-zero flags: fwd in [0..count-1], bwd in [count..2*count-1].
    // Avoids recomputing dx in Phases 3 and 4.
    uint8_t dx_zero_stack[128] = {};
    static thread_local std::vector<uint8_t> dx_zero_heap_bidir;
    uint8_t* dx_zero = dx_zero_stack;
    if (SECP256K1_UNLIKELY(total > 128)) {
        if (dx_zero_heap_bidir.size() < total) dx_zero_heap_bidir.resize(total);
        dx_zero = dx_zero_heap_bidir.data();
    }

    // Phase 1: dx for both directions (zero-safe: replace 0 -> 1)
    // Save is-zero flags to avoid recomputing dx in Phases 3 and 4.
    for (std::size_t i = 0; i < count; ++i) {
        FieldElement const dx_fwd = offsets_fwd[i].x - base_x;
        FieldElement const dx_bwd = offsets_bwd[i].x - base_x;
        dx_zero[i]         = static_cast<uint8_t>(dx_fwd == zero);
        dx_zero[count + i] = static_cast<uint8_t>(dx_bwd == zero);
        scratch[i]         = dx_zero[i]         ? one : dx_fwd;
        scratch[count + i] = dx_zero[count + i] ? one : dx_bwd;
    }

    // Phase 2: Single batch inverse (all slots nonzero: zero dx replaced with 1 above)
    fe_batch_inverse_nonzero(scratch.data(), total);

    // Phases 3+4: Forward and backward results (FE52 path: ~26% faster arithmetic).
#if defined(SECP256K1_FAST_52BIT)
    {
        using FE52 = FieldElement52;
        FE52 const neg_bx = FE52::from_fe(base_x).negate(1);
        FE52 const neg_by = FE52::from_fe(base_y).negate(1);
        for (std::size_t i = 0; i < count; ++i) {
            if (SECP256K1_UNLIKELY(dx_zero[i])) { out_x_fwd[i] = zero; }
            else {
                FE52 const inv52 = FE52::from_fe(scratch[i]);
                FE52 const ox52  = FE52::from_fe(offsets_fwd[i].x);
                FE52 const dy52  = FE52::from_fe(offsets_fwd[i].y) + neg_by;
                FE52 const lam   = dy52 * inv52;
                FE52 x3 = lam.square() + neg_bx + ox52.negate(1);
                out_x_fwd[i] = x3.to_fe();
            }
            if (SECP256K1_UNLIKELY(dx_zero[count + i])) { out_x_bwd[i] = zero; }
            else {
                FE52 const inv52 = FE52::from_fe(scratch[count + i]);
                FE52 const ox52  = FE52::from_fe(offsets_bwd[i].x);
                FE52 const dy52  = FE52::from_fe(offsets_bwd[i].y) + neg_by;
                FE52 const lam   = dy52 * inv52;
                FE52 x3 = lam.square() + neg_bx + ox52.negate(1);
                out_x_bwd[i] = x3.to_fe();
            }
        }
    }
#else
    for (std::size_t i = 0; i < count; ++i) {
        if (SECP256K1_UNLIKELY(dx_zero[i])) { out_x_fwd[i] = zero; }
        else {
            FieldElement const dy = offsets_fwd[i].y - base_y;
            FieldElement const lambda = dy * scratch[i];
            FieldElement lambda_sq = lambda; lambda_sq.square_inplace();
            out_x_fwd[i] = lambda_sq - base_x - offsets_fwd[i].x;
        }
    }
    for (std::size_t i = 0; i < count; ++i) {
        if (SECP256K1_UNLIKELY(dx_zero[count + i])) { out_x_bwd[i] = zero; }
        else {
            FieldElement const dy = offsets_bwd[i].y - base_y;
            FieldElement const lambda = dy * scratch[count + i];
            FieldElement lambda_sq = lambda; lambda_sq.square_inplace();
            out_x_bwd[i] = lambda_sq - base_x - offsets_bwd[i].x;
        }
    }
#endif
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
    PrecomputeBuffers bufs;
    FieldElement* jac_x = bufs.x(count);
    FieldElement* jac_y = bufs.y(count);
    FieldElement* jac_z = bufs.z(count);

    jac_x[0] = current.X();
    jac_y[0] = current.Y();
    jac_z[0] = current.z();

    for (std::size_t i = 1; i < count; ++i) {
        current.next_inplace();  // (i+1)*G
        jac_x[i] = current.X();
        jac_y[i] = current.Y();
        jac_z[i] = current.z();
    }

    // Batch inverse all Z coordinates (nonzero: ECC points on prime-order curve)
    fe_batch_inverse_nonzero(jac_z, count);

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

    PrecomputeBuffers bufs;
    FieldElement* jac_x = bufs.x(count);
    FieldElement* jac_y = bufs.y(count);
    FieldElement* jac_z = bufs.z(count);

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

    // Batch inverse Z (nonzero: ECC points on prime-order curve)
    fe_batch_inverse_nonzero(jac_z, count);

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
