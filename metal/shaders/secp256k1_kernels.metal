// =============================================================================
// UltrafastSecp256k1 Metal — GPU Compute Kernels (Accelerated)
// =============================================================================
// Metal compute shaders for secp256k1 batch operations on Apple Silicon.
//
// ACCELERATION TECHNIQUES:
//   - Comba product scanning in field_mul/field_sqr (see secp256k1_field.h)
//   - 4-bit windowed scalar_mul (see secp256k1_point.h)
//   - Proper O(1) per-thread offset via scalar multiplication
//   - Chunked batch inverse — one threadgroup per chunk
//   - Branchless bloom check with coalesced memory access
//
// Kernels:
//   1.  search_kernel          — Batch ECC search (O(1) offset + bloom check)
//   2.  scalar_mul_batch       — Batch scalar multiplication
//   3.  generator_mul_batch    — Batch G×k multiplication
//   4.  field_mul_bench        — Field multiplication benchmark
//   5.  field_sqr_bench        — Field squaring benchmark
//   5b. field_add_bench        — Field addition benchmark
//   5c. field_sub_bench        — Field subtraction benchmark
//   5d. field_inv_bench        — Field inversion benchmark
//   6.  batch_inverse          — Chunked Montgomery batch inverse
//   7.  point_add_kernel       — Point addition (testing)
//   8.  point_double_kernel    — Point doubling (testing)
// =============================================================================

#include <metal_stdlib>
#include "secp256k1_field.h"
#include "secp256k1_point.h"
#include "secp256k1_bloom.h"

using namespace metal;

// =============================================================================
// Search Result — 40 bytes, matching CUDA layout
// =============================================================================

struct SearchResult {
    uint x[8];       // 32 bytes: affine X coordinate
    uint index_lo;   // iteration index (low 32)
    uint index_hi;   // iteration index (high 32)
};

// =============================================================================
// Kernel Parameters — passed as uniforms (constant buffer)
// =============================================================================

struct SearchParams {
    uint batch_size;
    uint batch_offset_lo;
    uint batch_offset_hi;
    uint max_results;
};

// =============================================================================
// Kernel 1: ECC Search — O(1) per-thread offset + bloom filter
// =============================================================================
// ACCELERATION: Instead of O(tid) incremental additions, each thread
// computes its scalar offset k = batch_start + tid and does a single
// scalar_mul(G, k). The 4-bit windowed scalar_mul makes this fast.
//
// For SUBTRACTION search (Q - kG = target?), host pre-computes:
//   Q_start = target point
//   KQ_start_scalar = base scalar offset
// Thread tid checks: k = KQ_start_scalar + tid
// =============================================================================

kernel void search_kernel(
    constant JacobianPoint &Q_start    [[buffer(0)]],
    constant AffinePoint &G_affine     [[buffer(1)]],
    constant JacobianPoint &KQ_start   [[buffer(2)]],
    constant AffinePoint &KG_affine    [[buffer(3)]],
    device const uint *bloom_bitwords  [[buffer(4)]],
    constant BloomParams &bloom_params [[buffer(5)]],
    device SearchResult *results       [[buffer(6)]],
    device atomic_uint *result_count   [[buffer(7)]],
    constant SearchParams &params      [[buffer(8)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= params.batch_size) return;

    // Compute scalar offset: k = tid as Scalar256
    Scalar256 k_offset;
    k_offset.limbs[0] = tid;
    for (int i = 1; i < 8; i++) k_offset.limbs[i] = 0;

    // Copy constant-address-space args to thread-local (MSL address space requirement)
    AffinePoint kg_local = KG_affine;
    JacobianPoint kq_local = KQ_start;

    // Compute tid*KG via scalar_mul (4-bit windowed, efficient for small scalars)
    JacobianPoint offset_point = scalar_mul(kg_local, k_offset);

    // KQ = KQ_start + tid*KG
    JacobianPoint KQ = jacobian_add(kq_local, offset_point);

    // Convert to affine X
    AffinePoint kq_aff = jacobian_to_affine(KQ);

    // Convert X to little-endian bytes for bloom test
    uint8_t x_bytes[32];
    for (int i = 0; i < 8; i++) {
        uint limb = kq_aff.x.limbs[i];
        x_bytes[i * 4 + 0] = uint8_t(limb & 0xFFu);
        x_bytes[i * 4 + 1] = uint8_t((limb >> 8) & 0xFFu);
        x_bytes[i * 4 + 2] = uint8_t((limb >> 16) & 0xFFu);
        x_bytes[i * 4 + 3] = uint8_t((limb >> 24) & 0xFFu);
    }

    // Bloom filter check — branchless
    if (bloom_test(bloom_bitwords, bloom_params, x_bytes, 32)) {
        uint idx = atomic_fetch_add_explicit(result_count, 1u, memory_order_relaxed);
        if (idx < params.max_results) {
            for (int i = 0; i < 8; i++) results[idx].x[i] = kq_aff.x.limbs[i];

            // 64-bit index = batch_offset + tid
            ulong full_idx = (ulong(params.batch_offset_hi) << 32) | ulong(params.batch_offset_lo);
            full_idx += tid;
            results[idx].index_lo = uint(full_idx);
            results[idx].index_hi = uint(full_idx >> 32);
        }
    }
}

// =============================================================================
// Kernel 2: Batch Scalar Multiplication — P × k for N points
// =============================================================================

kernel void scalar_mul_batch(
    device const AffinePoint *bases    [[buffer(0)]],
    device const Scalar256 *scalars    [[buffer(1)]],
    device AffinePoint *results        [[buffer(2)]],
    constant uint &count               [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    AffinePoint base = bases[tid];
    Scalar256 k = scalars[tid];

    JacobianPoint jac = scalar_mul(base, k);
    results[tid] = jacobian_to_affine(jac);
}

// =============================================================================
// Kernel 3: Batch Generator Multiplication — G × k for N scalars
// =============================================================================

kernel void generator_mul_batch(
    device const Scalar256 *scalars    [[buffer(0)]],
    device AffinePoint *results        [[buffer(1)]],
    constant uint &count               [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    AffinePoint gen = generator_affine();
    Scalar256 k = scalars[tid];

    JacobianPoint jac = scalar_mul(gen, k);
    results[tid] = jacobian_to_affine(jac);
}

// =============================================================================
// Kernel 4: Field Multiplication Benchmark
// =============================================================================

kernel void field_mul_bench(
    device const FieldElement *a_arr  [[buffer(0)]],
    device const FieldElement *b_arr  [[buffer(1)]],
    device FieldElement *r_arr        [[buffer(2)]],
    constant uint &count              [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    FieldElement a = a_arr[tid];
    FieldElement b = b_arr[tid];
    r_arr[tid] = field_mul(a, b);
}

// =============================================================================
// Kernel 5: Field Squaring Benchmark
// =============================================================================

kernel void field_sqr_bench(
    device const FieldElement *a_arr  [[buffer(0)]],
    device FieldElement *r_arr        [[buffer(1)]],
    constant uint &count              [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    FieldElement a = a_arr[tid];
    r_arr[tid] = field_sqr(a);
}

// =============================================================================
// Kernel 5b: Field Addition Benchmark
// =============================================================================

kernel void field_add_bench(
    device const FieldElement *a_arr  [[buffer(0)]],
    device const FieldElement *b_arr  [[buffer(1)]],
    device FieldElement *r_arr        [[buffer(2)]],
    constant uint &count              [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    FieldElement a = a_arr[tid];
    FieldElement b = b_arr[tid];
    r_arr[tid] = field_add(a, b);
}

// =============================================================================
// Kernel 5c: Field Subtraction Benchmark
// =============================================================================

kernel void field_sub_bench(
    device const FieldElement *a_arr  [[buffer(0)]],
    device const FieldElement *b_arr  [[buffer(1)]],
    device FieldElement *r_arr        [[buffer(2)]],
    constant uint &count              [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    FieldElement a = a_arr[tid];
    FieldElement b = b_arr[tid];
    r_arr[tid] = field_sub(a, b);
}

// =============================================================================
// Kernel 5d: Field Inversion Benchmark
// =============================================================================

kernel void field_inv_bench(
    device const FieldElement *a_arr  [[buffer(0)]],
    device FieldElement *r_arr        [[buffer(1)]],
    constant uint &count              [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    FieldElement a = a_arr[tid];
    r_arr[tid] = field_inv(a);
}

// =============================================================================
// Kernel 6: Chunked Batch Field Inverse
// =============================================================================
// ACCELERATION: Each threadgroup handles one chunk of the array.
// The chunk_size is passed as uniform — each threadgroup uses Montgomery's
// trick internally (1 inversion per chunk).
//
// For N elements with chunk_size C:
//   - Launch N/C threadgroups, each with 1 thread
//   - Total inversions: N/C instead of 1 (slightly more, but parallel!)
//
// With 256 chunks on Apple Silicon (256 threadgroups), this saturates the
// GPU while keeping inversion count manageable.
// =============================================================================

struct BatchInvParams {
    uint total_count;
    uint chunk_size;
};

kernel void batch_inverse(
    device FieldElement *elements     [[buffer(0)]],
    device FieldElement *scratch      [[buffer(1)]],
    constant BatchInvParams &params   [[buffer(2)]],
    uint tgid [[threadgroup_position_in_grid]]
) {
    uint start = tgid * params.chunk_size;
    uint end = min(start + params.chunk_size, params.total_count);
    if (start >= end) return;
    uint count = end - start;

    // Forward pass: prefix products (copy device→thread for field_mul)
    FieldElement acc = elements[start];
    scratch[start] = acc;
    for (uint i = 1; i < count; i++) {
        FieldElement el = elements[start + i];
        acc = field_mul(acc, el);
        scratch[start + i] = acc;
    }

    // Single inversion of the chunk product
    FieldElement s_last = scratch[start + count - 1];
    FieldElement inv = field_inv(s_last);

    // Backward pass: recover individual inverses
    for (uint i = count - 1; i > 0; i--) {
        FieldElement s_prev = scratch[start + i - 1];
        FieldElement el_i = elements[start + i];
        FieldElement tmp = field_mul(inv, s_prev);
        inv = field_mul(inv, el_i);
        elements[start + i] = tmp;
    }
    elements[start] = inv;
}

// =============================================================================
// Kernel 7: Point Addition Kernel (for testing)
// =============================================================================

kernel void point_add_kernel(
    device const JacobianPoint *a_arr  [[buffer(0)]],
    device const JacobianPoint *b_arr  [[buffer(1)]],
    device JacobianPoint *r_arr        [[buffer(2)]],
    constant uint &count               [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    // Copy device→thread for address space compatibility
    JacobianPoint a_local = a_arr[tid];
    JacobianPoint b_local = b_arr[tid];
    r_arr[tid] = jacobian_add(a_local, b_local);
}

// =============================================================================
// Kernel 8: Point Doubling Kernel (for testing)
// =============================================================================

kernel void point_double_kernel(
    device const JacobianPoint *a_arr  [[buffer(0)]],
    device JacobianPoint *r_arr        [[buffer(1)]],
    constant uint &count               [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;
    JacobianPoint a_local = a_arr[tid];
    r_arr[tid] = jacobian_double(a_local);
}
