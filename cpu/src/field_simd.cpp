#include "secp256k1/field_simd.hpp"
#include <cstring>

namespace secp256k1::simd {

using fast::FieldElement;

// ══════════════════════════════════════════════════════════════════════════════
// Scalar Fallback (always available, any platform)
// ══════════════════════════════════════════════════════════════════════════════

namespace detail {

void batch_field_add_scalar(FieldElement* out, const FieldElement* a,
                            const FieldElement* b, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

void batch_field_sub_scalar(FieldElement* out, const FieldElement* a,
                            const FieldElement* b, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = a[i] - b[i];
    }
}

void batch_field_mul_scalar(FieldElement* out, const FieldElement* a,
                            const FieldElement* b, std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

void batch_field_sqr_scalar(FieldElement* out, const FieldElement* a,
                            std::size_t count) {
    for (std::size_t i = 0; i < count; ++i) {
        out[i] = a[i].square();
    }
}

} // namespace detail

// ══════════════════════════════════════════════════════════════════════════════
// Auto-Dispatching Batch Operations
// ══════════════════════════════════════════════════════════════════════════════
// Current implementation: scalar-only with architecture detection.
// SIMD kernels (AVX2/AVX-512) operate on the 4×uint64 representation.
//
// For field multiplication, SIMD doesn't help much because secp256k1
// modular reduction is inherently serial (carry propagation).
// The main benefit is for add/sub which are carry-chain operations
// that can be partially parallelized, and for batch inverse
// (Montgomery's trick) which is inherently parallelizable.

void batch_field_add(FieldElement* out,
                     const FieldElement* a,
                     const FieldElement* b,
                     std::size_t count) {
    // Note: For secp256k1 field arithmetic, individual add operations
    // are already very fast (~2ns). The auto-vectorizer with -ftree-vectorize
    // + -march=native typically handles this well. Explicit SIMD intrinsics
    // would add complexity without measurable benefit for add/sub.
    detail::batch_field_add_scalar(out, a, b, count);
}

void batch_field_sub(FieldElement* out,
                     const FieldElement* a,
                     const FieldElement* b,
                     std::size_t count) {
    detail::batch_field_sub_scalar(out, a, b, count);
}

void batch_field_mul(FieldElement* out,
                     const FieldElement* a,
                     const FieldElement* b,
                     std::size_t count) {
    detail::batch_field_mul_scalar(out, a, b, count);
}

void batch_field_sqr(FieldElement* out,
                     const FieldElement* a,
                     std::size_t count) {
    detail::batch_field_sqr_scalar(out, a, count);
}

// ══════════════════════════════════════════════════════════════════════════════
// Batch Modular Inverse (Montgomery's Trick)
// ══════════════════════════════════════════════════════════════════════════════
// Computes n inversions with only 1 actual field inversion.
// Algorithm:
//   1. Compute running products: prod[i] = a[0] * a[1] * ... * a[i]
//   2. Invert the final product: inv_all = prod[n-1]^(-1)
//   3. Back-propagate: out[i] = inv_all * prod[i-1], inv_all *= a[i]
//
// Cost: 1 inversion + 3(n-1) multiplications
// vs. n inversions naively (~250x faster for n=256)

void batch_field_inv(FieldElement* out,
                     const FieldElement* a,
                     std::size_t count,
                     FieldElement* scratch) {
    if (count == 0) return;
    if (count == 1) {
        out[0] = a[0].inverse();
        return;
    }

    // Use scratch if provided, otherwise use output as scratch
    // (we'll overwrite it anyway)
    FieldElement* products = scratch ? scratch : out;

    // Step 1: Forward pass — compute running products
    products[0] = a[0];
    for (std::size_t i = 1; i < count; ++i) {
        products[i] = products[i - 1] * a[i];
    }

    // Step 2: Single inversion
    auto inv = products[count - 1].inverse();

    // Step 3: Backward pass — distribute the inverse
    for (std::size_t i = count - 1; i > 0; --i) {
        out[i] = inv * products[i - 1];
        inv = inv * a[i];
    }
    out[0] = inv;
}

} // namespace secp256k1::simd
