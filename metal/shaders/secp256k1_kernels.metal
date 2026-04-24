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
//
// Kernels:
//   1.  scalar_mul_batch       — Batch scalar multiplication
//   2.  generator_mul_batch    — Batch G×k multiplication
//   3.  field_mul_bench        — Field multiplication benchmark
//   4.  field_sqr_bench        — Field squaring benchmark
//   4b. field_add_bench        — Field addition benchmark
//   4c. field_sub_bench        — Field subtraction benchmark
//   4d. field_inv_bench        — Field inversion benchmark
//   5.  batch_inverse          — Chunked Montgomery batch inverse
//   6.  point_add_kernel       — Point addition (testing)
//   7.  point_double_kernel    — Point doubling (testing)
// =============================================================================

#include <metal_stdlib>
#include "secp256k1_field.h"
#include "secp256k1_point.h"
#include "secp256k1_extended.h"
#include "secp256k1_hash160.h"
#include "secp256k1_zk.h"
#include "secp256k1_bip324.h"
#include "secp256k1_ct_ops.h"
#include "secp256k1_ct_field.h"
#include "secp256k1_ct_scalar.h"
#include "secp256k1_ct_point.h"
#include "secp256k1_ct_sign.h"

using namespace metal;

// =============================================================================
// Kernel 1: Batch Scalar Multiplication — P × k for N points
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

    JacobianPoint jac = scalar_mul_glv(base, k);
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

    JacobianPoint jac = scalar_mul_glv(gen, k);
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

// =============================================================================
// Kernel 9: Batch ECDSA Sign
// =============================================================================

kernel void ecdsa_sign_batch(
    device const uchar *msg_hashes     [[buffer(0)]],   // N × 32 bytes
    device const uchar *privkeys       [[buffer(1)]],   // N × 32 bytes
    device uchar *signatures           [[buffer(2)]],   // N × 64 bytes (r ∥ s)
    constant uint &count               [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 msg, sec;
    for (int i = 0; i < 8; i++) {
        uint idx = tid * 32 + i * 4;
        msg.limbs[7 - i] = ((uint)msg_hashes[idx] << 24) |
                            ((uint)msg_hashes[idx+1] << 16) |
                            ((uint)msg_hashes[idx+2] << 8) |
                            ((uint)msg_hashes[idx+3]);
        sec.limbs[7 - i] = ((uint)privkeys[idx] << 24) |
                            ((uint)privkeys[idx+1] << 16) |
                            ((uint)privkeys[idx+2] << 8) |
                            ((uint)privkeys[idx+3]);
    }

    Scalar256 r_sig, s_sig;
    ecdsa_sign(msg, sec, r_sig, s_sig);

    // Write r ∥ s as big-endian
    uint out_off = tid * 64;
    for (int i = 0; i < 8; i++) {
        uint rv = r_sig.limbs[7 - i];
        signatures[out_off + i*4 + 0] = (uchar)(rv >> 24);
        signatures[out_off + i*4 + 1] = (uchar)(rv >> 16);
        signatures[out_off + i*4 + 2] = (uchar)(rv >> 8);
        signatures[out_off + i*4 + 3] = (uchar)(rv);
    }
    for (int i = 0; i < 8; i++) {
        uint sv = s_sig.limbs[7 - i];
        signatures[out_off + 32 + i*4 + 0] = (uchar)(sv >> 24);
        signatures[out_off + 32 + i*4 + 1] = (uchar)(sv >> 16);
        signatures[out_off + 32 + i*4 + 2] = (uchar)(sv >> 8);
        signatures[out_off + 32 + i*4 + 3] = (uchar)(sv);
    }
}

// =============================================================================
// Kernel 10: Batch ECDSA Verify
// =============================================================================

kernel void ecdsa_verify_batch(
    device const uchar *msg_hashes     [[buffer(0)]],   // N × 32
    device const uchar *pubkeys        [[buffer(1)]],   // N × 64 (x ∥ y, uncompressed coords)
    device const uchar *signatures     [[buffer(2)]],   // N × 64 (r ∥ s)
    device uint *results               [[buffer(3)]],   // N × 1 (0=invalid, 1=valid)
    constant uint &count               [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 msg, r_sig, s_sig;
    AffinePoint pub;

    uint base_msg = tid * 32;
    uint base_pub = tid * 64;
    uint base_sig = tid * 64;

    for (int i = 0; i < 8; i++) {
        msg.limbs[7 - i] = ((uint)msg_hashes[base_msg + i*4] << 24) |
                            ((uint)msg_hashes[base_msg + i*4+1] << 16) |
                            ((uint)msg_hashes[base_msg + i*4+2] << 8) |
                            ((uint)msg_hashes[base_msg + i*4+3]);

        pub.x.limbs[7 - i] = ((uint)pubkeys[base_pub + i*4] << 24) |
                              ((uint)pubkeys[base_pub + i*4+1] << 16) |
                              ((uint)pubkeys[base_pub + i*4+2] << 8) |
                              ((uint)pubkeys[base_pub + i*4+3]);
        pub.y.limbs[7 - i] = ((uint)pubkeys[base_pub + 32 + i*4] << 24) |
                              ((uint)pubkeys[base_pub + 32 + i*4+1] << 16) |
                              ((uint)pubkeys[base_pub + 32 + i*4+2] << 8) |
                              ((uint)pubkeys[base_pub + 32 + i*4+3]);

        r_sig.limbs[7 - i] = ((uint)signatures[base_sig + i*4] << 24) |
                              ((uint)signatures[base_sig + i*4+1] << 16) |
                              ((uint)signatures[base_sig + i*4+2] << 8) |
                              ((uint)signatures[base_sig + i*4+3]);
        s_sig.limbs[7 - i] = ((uint)signatures[base_sig + 32 + i*4] << 24) |
                              ((uint)signatures[base_sig + 32 + i*4+1] << 16) |
                              ((uint)signatures[base_sig + 32 + i*4+2] << 8) |
                              ((uint)signatures[base_sig + 32 + i*4+3]);
    }

    results[tid] = ecdsa_verify(msg, pub, r_sig, s_sig) ? 1u : 0u;
}

// =============================================================================
// Kernel 11: Batch Schnorr Sign (BIP-340)
// =============================================================================

kernel void schnorr_sign_batch(
    device const uchar *msg_hashes     [[buffer(0)]],   // N × 32
    device const uchar *privkeys       [[buffer(1)]],   // N × 32
    device uchar *signatures           [[buffer(2)]],   // N × 64 (R.x ∥ s)
    constant uint &count               [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 msg, sec;
    for (int i = 0; i < 8; i++) {
        uint idx = tid * 32 + i * 4;
        msg.limbs[7 - i] = ((uint)msg_hashes[idx] << 24) |
                            ((uint)msg_hashes[idx+1] << 16) |
                            ((uint)msg_hashes[idx+2] << 8) |
                            ((uint)msg_hashes[idx+3]);
        sec.limbs[7 - i] = ((uint)privkeys[idx] << 24) |
                            ((uint)privkeys[idx+1] << 16) |
                            ((uint)privkeys[idx+2] << 8) |
                            ((uint)privkeys[idx+3]);
    }

    Scalar256 sig_rx, sig_s;
    schnorr_sign(msg, sec, sig_rx, sig_s);

    uint out_off = tid * 64;
    for (int i = 0; i < 8; i++) {
        uint rv = sig_rx.limbs[7 - i];
        signatures[out_off + i*4 + 0] = (uchar)(rv >> 24);
        signatures[out_off + i*4 + 1] = (uchar)(rv >> 16);
        signatures[out_off + i*4 + 2] = (uchar)(rv >> 8);
        signatures[out_off + i*4 + 3] = (uchar)(rv);
    }
    for (int i = 0; i < 8; i++) {
        uint sv = sig_s.limbs[7 - i];
        signatures[out_off + 32 + i*4 + 0] = (uchar)(sv >> 24);
        signatures[out_off + 32 + i*4 + 1] = (uchar)(sv >> 16);
        signatures[out_off + 32 + i*4 + 2] = (uchar)(sv >> 8);
        signatures[out_off + 32 + i*4 + 3] = (uchar)(sv);
    }
}

// =============================================================================
// Kernel 12: Batch Schnorr Verify (BIP-340)
// =============================================================================

kernel void schnorr_verify_batch(
    device const uchar *msg_hashes     [[buffer(0)]],   // N × 32
    device const uchar *pubkeys_x      [[buffer(1)]],   // N × 32 (x-only)
    device const uchar *signatures     [[buffer(2)]],   // N × 64 (R.x ∥ s)
    device uint *results               [[buffer(3)]],
    constant uint &count               [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 msg, pub_x, sig_rx, sig_s;
    for (int i = 0; i < 8; i++) {
        uint mi = tid * 32 + i * 4;
        msg.limbs[7 - i] = ((uint)msg_hashes[mi] << 24) |
                            ((uint)msg_hashes[mi+1] << 16) |
                            ((uint)msg_hashes[mi+2] << 8) |
                            ((uint)msg_hashes[mi+3]);

        pub_x.limbs[7 - i] = ((uint)pubkeys_x[mi] << 24) |
                              ((uint)pubkeys_x[mi+1] << 16) |
                              ((uint)pubkeys_x[mi+2] << 8) |
                              ((uint)pubkeys_x[mi+3]);

        uint si = tid * 64 + i * 4;
        sig_rx.limbs[7 - i] = ((uint)signatures[si] << 24) |
                               ((uint)signatures[si+1] << 16) |
                               ((uint)signatures[si+2] << 8) |
                               ((uint)signatures[si+3]);
        sig_s.limbs[7 - i] = ((uint)signatures[si + 32] << 24) |
                              ((uint)signatures[si + 32 +1] << 16) |
                              ((uint)signatures[si + 32 +2] << 8) |
                              ((uint)signatures[si + 32 +3]);
    }

    // Convert pub_x to FieldElement for schnorr_verify
    FieldElement px;
    for (int i = 0; i < 8; i++) px.limbs[i] = pub_x.limbs[i];

    results[tid] = schnorr_verify(msg, px, sig_rx, sig_s) ? 1u : 0u;
}

// =============================================================================
// Kernel 13: Batch ECDH Shared Secret
// =============================================================================

kernel void ecdh_batch(
    device const uchar *privkeys       [[buffer(0)]],   // N × 32
    device const uchar *pubkeys        [[buffer(1)]],   // N × 64 (x ∥ y)
    device uchar *shared_secrets       [[buffer(2)]],   // N × 32 (x-only)
    constant uint &count               [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 sec;
    AffinePoint pub;
    for (int i = 0; i < 8; i++) {
        uint ki = tid * 32 + i * 4;
        sec.limbs[7 - i] = ((uint)privkeys[ki] << 24) |
                            ((uint)privkeys[ki+1] << 16) |
                            ((uint)privkeys[ki+2] << 8) |
                            ((uint)privkeys[ki+3]);

        uint pi = tid * 64 + i * 4;
        pub.x.limbs[7 - i] = ((uint)pubkeys[pi] << 24) |
                              ((uint)pubkeys[pi+1] << 16) |
                              ((uint)pubkeys[pi+2] << 8) |
                              ((uint)pubkeys[pi+3]);
        uint yi = tid * 64 + 32 + i * 4;
        pub.y.limbs[7 - i] = ((uint)pubkeys[yi] << 24) |
                              ((uint)pubkeys[yi+1] << 16) |
                              ((uint)pubkeys[yi+2] << 8) |
                              ((uint)pubkeys[yi+3]);
    }

    FieldElement shared_x = ecdh_shared_secret_xonly(sec, pub);

    // Output x as big-endian
    uint out_off = tid * 32;
    for (int i = 0; i < 8; i++) {
        uint v = shared_x.limbs[7 - i];
        shared_secrets[out_off + i*4 + 0] = (uchar)(v >> 24);
        shared_secrets[out_off + i*4 + 1] = (uchar)(v >> 16);
        shared_secrets[out_off + i*4 + 2] = (uchar)(v >> 8);
        shared_secrets[out_off + i*4 + 3] = (uchar)(v);
    }
}

// =============================================================================
// Kernel 14: Batch Hash160 of public keys
// =============================================================================

kernel void hash160_batch(
    device const uchar *pubkeys        [[buffer(0)]],
    device uchar *hashes               [[buffer(1)]],   // N × 20
    constant uint &stride              [[buffer(2)]],   // 33 or 65
    constant uint &count               [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uchar pk[65];
    const uint pk_len = (stride <= 65u) ? stride : 65u;
    for (uint i = 0; i < pk_len; ++i) {
        pk[i] = pubkeys[tid * stride + i];
    }

    uchar h160[20];
    hash160_pubkey(pk, pk_len, h160);

    for (int i = 0; i < 20; ++i) {
        hashes[tid * 20 + i] = h160[i];
    }
}

// =============================================================================
// Kernel 15: Batch Key Recovery
// =============================================================================

kernel void ecrecover_batch(
    device const uchar *msg_hashes     [[buffer(0)]],   // N × 32
    device const uchar *signatures     [[buffer(1)]],   // N × 64 (r ∥ s)
    device const uint *recids          [[buffer(2)]],    // N × 1 (recovery id 0-3)
    device uchar *pubkeys              [[buffer(3)]],    // N × 64 (x ∥ y)
    device uint *valid                 [[buffer(4)]],    // N × 1 (0=fail, 1=ok)
    constant uint &count               [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 msg, r_sig, s_sig;
    for (int i = 0; i < 8; i++) {
        uint mi = tid * 32 + i * 4;
        msg.limbs[7 - i] = ((uint)msg_hashes[mi] << 24) |
                            ((uint)msg_hashes[mi+1] << 16) |
                            ((uint)msg_hashes[mi+2] << 8) |
                            ((uint)msg_hashes[mi+3]);

        uint si = tid * 64 + i * 4;
        r_sig.limbs[7 - i] = ((uint)signatures[si] << 24) |
                              ((uint)signatures[si+1] << 16) |
                              ((uint)signatures[si+2] << 8) |
                              ((uint)signatures[si+3]);
        s_sig.limbs[7 - i] = ((uint)signatures[si + 32] << 24) |
                              ((uint)signatures[si + 32 +1] << 16) |
                              ((uint)signatures[si + 32 +2] << 8) |
                              ((uint)signatures[si + 32 +3]);
    }

    AffinePoint recovered;
    bool ok = ecdsa_recover(msg, r_sig, s_sig, recids[tid], recovered);
    valid[tid] = ok ? 1u : 0u;

    if (ok) {
        uint out_off = tid * 64;
        for (int i = 0; i < 8; i++) {
            uint xv = recovered.x.limbs[7 - i];
            pubkeys[out_off + i*4 + 0] = (uchar)(xv >> 24);
            pubkeys[out_off + i*4 + 1] = (uchar)(xv >> 16);
            pubkeys[out_off + i*4 + 2] = (uchar)(xv >> 8);
            pubkeys[out_off + i*4 + 3] = (uchar)(xv);
        }
        for (int i = 0; i < 8; i++) {
            uint yv = recovered.y.limbs[7 - i];
            pubkeys[out_off + 32 + i*4 + 0] = (uchar)(yv >> 24);
            pubkeys[out_off + 32 + i*4 + 1] = (uchar)(yv >> 16);
            pubkeys[out_off + 32 + i*4 + 2] = (uchar)(yv >> 8);
            pubkeys[out_off + 32 + i*4 + 3] = (uchar)(yv);
        }
    }
}

// =============================================================================
// Kernel 16: SHA-256 Benchmark
// =============================================================================

kernel void sha256_bench(
    device const uchar *inputs         [[buffer(0)]],   // N × 64 bytes
    device uchar *outputs              [[buffer(1)]],   // N × 32 bytes
    constant uint &count               [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uchar data[64];
    for (int i = 0; i < 64; i++) data[i] = inputs[tid * 64 + i];

    uchar hash[32];
    sha256_oneshot(data, 64, hash);

    for (int i = 0; i < 32; i++) outputs[tid * 32 + i] = hash[i];
}

// =============================================================================
// Kernel 17: Hash160 Benchmark
// =============================================================================

kernel void hash160_bench(
    device const uchar *inputs         [[buffer(0)]],   // N × 33 bytes (compressed pubkeys)
    device uchar *outputs              [[buffer(1)]],   // N × 20 bytes
    constant uint &count               [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uchar pk[33];
    for (int i = 0; i < 33; i++) pk[i] = inputs[tid * 33 + i];

    uchar h160[20];
    hash160_pubkey(pk, 33, h160);

    for (int i = 0; i < 20; i++) outputs[tid * 20 + i] = h160[i];
}

// =============================================================================
// Kernel 18: ECDSA Sign Benchmark (sign + verify round-trip)
// =============================================================================

kernel void ecdsa_bench(
    device const uchar *msg_hashes     [[buffer(0)]],
    device const uchar *privkeys       [[buffer(1)]],
    device uint *results               [[buffer(2)]],
    constant uint &count               [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 msg, sec;
    for (int i = 0; i < 8; i++) {
        uint idx = tid * 32 + i * 4;
        msg.limbs[7 - i] = ((uint)msg_hashes[idx] << 24) |
                            ((uint)msg_hashes[idx+1] << 16) |
                            ((uint)msg_hashes[idx+2] << 8) |
                            ((uint)msg_hashes[idx+3]);
        sec.limbs[7 - i] = ((uint)privkeys[idx] << 24) |
                            ((uint)privkeys[idx+1] << 16) |
                            ((uint)privkeys[idx+2] << 8) |
                            ((uint)privkeys[idx+3]);
    }

    // Sign
    Scalar256 r_sig, s_sig;
    ecdsa_sign(msg, sec, r_sig, s_sig);

    // Derive public key
    AffinePoint gen = generator_affine();
    JacobianPoint pub_jac = scalar_mul_glv(gen, sec);
    AffinePoint pub_aff = jacobian_to_affine(pub_jac);

    // Verify
    results[tid] = ecdsa_verify(msg, pub_aff, r_sig, s_sig) ? 1u : 0u;
}

// =============================================================================
// Kernel 19: ZK Knowledge Proof -- Batch Prove
// =============================================================================

kernel void zk_knowledge_prove_batch(
    device const uchar *secrets        [[buffer(0)]],
    device const uchar *pubkeys        [[buffer(1)]],
    device const uchar *messages       [[buffer(2)]],
    device const uchar *aux_rands      [[buffer(3)]],
    device uchar *proof_rx_out         [[buffer(4)]],
    device uchar *proof_s_out          [[buffer(5)]],
    device uint *results               [[buffer(6)]],
    constant uint &count               [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 sec = scalar_from_bytes(secrets + tid * 32);
    JacobianPoint pk = scalar_mul_generator_windowed(sec);

    uchar msg[32], aux[32];
    for (int i = 0; i < 32; ++i) { msg[i] = messages[tid * 32 + i]; aux[i] = aux_rands[tid * 32 + i]; }

    AffinePoint G = generator_affine();
    ZKKnowledgeProof proof;
    bool ok = zk_knowledge_prove(sec, pk, G, msg, aux, proof);

    for (int i = 0; i < 32; ++i) proof_rx_out[tid * 32 + i] = proof.rx[i];
    uchar s_bytes[32];
    scalar_to_bytes(proof.s, s_bytes);
    for (int i = 0; i < 32; ++i) proof_s_out[tid * 32 + i] = s_bytes[i];
    results[tid] = ok ? 1u : 0u;
}

// =============================================================================
// Kernel 20: ZK Knowledge Proof -- Batch Verify
// =============================================================================

kernel void zk_knowledge_verify_batch(
    device const uchar *proof_rx_in    [[buffer(0)]],
    device const uchar *proof_s_in     [[buffer(1)]],
    device const uchar *pubkeys        [[buffer(2)]],
    device const uchar *messages       [[buffer(3)]],
    device uint *results               [[buffer(4)]],
    constant uint &count               [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    ZKKnowledgeProof proof;
    for (int i = 0; i < 32; ++i) proof.rx[i] = proof_rx_in[tid * 32 + i];
    uchar s_bytes[32];
    for (int i = 0; i < 32; ++i) s_bytes[i] = proof_s_in[tid * 32 + i];
    proof.s = scalar_from_bytes(s_bytes);

    // pubkeys contains 32-byte x-coordinates; use lift_x to recover the full point.
    JacobianPoint pk;
    lift_x(pubkeys + tid * 32, pk);

    uchar msg[32];
    for (int i = 0; i < 32; ++i) msg[i] = messages[tid * 32 + i];

    AffinePoint G = generator_affine();
    results[tid] = zk_knowledge_verify(proof, pk, G, msg) ? 1u : 0u;
}

// =============================================================================
// Kernel 21: ZK DLEQ Proof -- Batch Prove
// =============================================================================

kernel void zk_dleq_prove_batch(
    device const uchar *secrets        [[buffer(0)]],
    device const uchar *aux_rands      [[buffer(1)]],
    device uchar *proof_e_out          [[buffer(2)]],
    device uchar *proof_s_out          [[buffer(3)]],
    device uint *results               [[buffer(4)]],
    constant uint &count               [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 sec = scalar_from_bytes(secrets + tid * 32);
    uchar aux[32];
    for (int i = 0; i < 32; ++i) aux[i] = aux_rands[tid * 32 + i];

    AffinePoint G = generator_affine();
    // H = second generator (deterministic derivation)
    uchar h_tag[] = {'Z','K','/','d','l','e','q','/','H'};
    uchar h_hash[32];
    tagged_hash(h_tag, 9, h_tag, 9, h_hash);
    JacobianPoint H_jac;
    lift_x(h_hash, H_jac);
    AffinePoint H = jacobian_to_affine(H_jac);

    JacobianPoint P = scalar_mul(G, sec);
    JacobianPoint Q = scalar_mul(H, sec);

    ZKDLEQProof proof;
    bool ok = zk_dleq_prove(sec, G, H, P, Q, aux, proof);

    uchar e_bytes[32], s_bytes[32];
    scalar_to_bytes(proof.e, e_bytes);
    scalar_to_bytes(proof.s, s_bytes);
    for (int i = 0; i < 32; ++i) { proof_e_out[tid * 32 + i] = e_bytes[i]; proof_s_out[tid * 32 + i] = s_bytes[i]; }
    results[tid] = ok ? 1u : 0u;
}

// =============================================================================
// Kernel 22: ZK DLEQ Proof -- Batch Verify
// =============================================================================

kernel void zk_dleq_verify_batch(
    device const uchar *proof_e_in     [[buffer(0)]],
    device const uchar *proof_s_in     [[buffer(1)]],
    device const uchar *pubkeys_P      [[buffer(2)]],
    device const uchar *pubkeys_Q      [[buffer(3)]],
    device uint *results               [[buffer(4)]],
    constant uint &count               [[buffer(5)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    ZKDLEQProof proof;
    uchar e_bytes[32], s_bytes[32];
    for (int i = 0; i < 32; ++i) { e_bytes[i] = proof_e_in[tid * 32 + i]; s_bytes[i] = proof_s_in[tid * 32 + i]; }
    proof.e = scalar_from_bytes(e_bytes);
    proof.s = scalar_from_bytes(s_bytes);

    AffinePoint G = generator_affine();
    uchar h_tag[] = {'Z','K','/','d','l','e','q','/','H'};
    uchar h_hash[32];
    tagged_hash(h_tag, 9, h_tag, 9, h_hash);
    JacobianPoint H_jac;
    lift_x(h_hash, H_jac);
    AffinePoint H = jacobian_to_affine(H_jac);

    // Reconstruct P and Q from pubkey bytes
    JacobianPoint P, Q;
    lift_x(pubkeys_P + tid * 32, P);
    lift_x(pubkeys_Q + tid * 32, Q);

    results[tid] = zk_dleq_verify(proof, G, H, P, Q) ? 1u : 0u;
}

// =============================================================================
// Kernel 23: Bulletproof Init (generator computation)
// =============================================================================

kernel void bulletproof_init_kernel(
    device AffinePoint *bp_G           [[buffer(0)]],
    device AffinePoint *bp_H           [[buffer(1)]],
    device ZKTagMidstate *bp_ip_midstate [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    // Compute "Bulletproof/ip" midstate
    {
        uchar tag[14] = {'B','u','l','l','e','t','p','r','o','o','f','/','i','p'};
        uchar tag_hash[32];
        SHA256Ctx ctx; sha256_init(ctx);
        sha256_update(ctx, tag, 14);
        sha256_final(ctx, tag_hash);
        sha256_init(ctx);
        sha256_update(ctx, tag_hash, 32);
        sha256_update(ctx, tag_hash, 32);
        for (int i = 0; i < 8; i++) bp_ip_midstate[0].h[i] = ctx.h[i];
    }

    // Compute "Bulletproof/gen" midstate
    ZKTagMidstate gen_midstate;
    {
        uchar tag[15] = {'B','u','l','l','e','t','p','r','o','o','f','/','g','e','n'};
        uchar tag_hash[32];
        SHA256Ctx ctx; sha256_init(ctx);
        sha256_update(ctx, tag, 15);
        sha256_final(ctx, tag_hash);
        sha256_init(ctx);
        sha256_update(ctx, tag_hash, 32);
        sha256_update(ctx, tag_hash, 32);
        for (int i = 0; i < 8; i++) gen_midstate.h[i] = ctx.h[i];
    }

    // Generate 64 G_i and 64 H_i
    for (int i = 0; i < 64; i++) {
        uchar buf[5];
        buf[1] = (uchar)(i & 0xFF);
        buf[2] = (uchar)((i >> 8) & 0xFF);
        buf[3] = (uchar)((i >> 16) & 0xFF);
        buf[4] = (uchar)((i >> 24) & 0xFF);

        uchar hash[32];

        // G_i
        buf[0] = 'G';
        zk_tagged_hash_midstate(gen_midstate, buf, 5, hash);
        FieldElement gx = field_from_bytes(hash);
        bp_G[i] = hash_to_point_increment(gx);

        // H_i
        buf[0] = 'H';
        zk_tagged_hash_midstate(gen_midstate, buf, 5, hash);
        FieldElement hx = field_from_bytes(hash);
        bp_H[i] = hash_to_point_increment(hx);
    }
}

// =============================================================================
// Kernel 24: Bulletproof Batch Verify
// =============================================================================

kernel void bulletproof_verify_batch(
    device const RangeProofGPU *proofs     [[buffer(0)]],
    device const AffinePoint *commitments  [[buffer(1)]],
    device const AffinePoint *H_gen        [[buffer(2)]],
    device const AffinePoint *bp_G         [[buffer(3)]],
    device const AffinePoint *bp_H         [[buffer(4)]],
    device const ZKTagMidstate *bp_ip_midstate [[buffer(5)]],
    device uint *results                   [[buffer(6)]],
    constant uint &count                   [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    RangeProofGPU proof = proofs[tid];
    AffinePoint commit = commitments[tid];
    AffinePoint h_ped = H_gen[0];
    ZKTagMidstate ip_mid = bp_ip_midstate[0];

    results[tid] = range_verify_full(proof, commit, h_ped,
                                      bp_G, bp_H, ip_mid) ? 1u : 0u;
}

// =============================================================================
// Kernel 25: Range Proof Polynomial Check (batch)
// =============================================================================

kernel void range_proof_poly_batch(
    device const RangeProofPolyGPU *proofs   [[buffer(0)]],
    device const AffinePoint *commitments    [[buffer(1)]],
    device const AffinePoint *H_gen          [[buffer(2)]],
    device uint *results                     [[buffer(3)]],
    constant uint &count                     [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    RangeProofPolyGPU proof = proofs[tid];
    AffinePoint commit = commitments[tid];
    AffinePoint h_ped = H_gen[0];

    results[tid] = range_proof_poly_check(proof, commit, h_ped) ? 1u : 0u;
}

// =============================================================================
// Kernel 26: Pedersen Commit Batch
// =============================================================================

kernel void pedersen_commit_batch(
    device const uchar *values_in          [[buffer(0)]],
    device const uchar *blindings_in       [[buffer(1)]],
    device const AffinePoint *H_gen        [[buffer(2)]],
    device uchar *commitments_out          [[buffer(3)]],
    constant uint &count                   [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    Scalar256 val = scalar_from_bytes(values_in + tid * 32);
    Scalar256 blind = scalar_from_bytes(blindings_in + tid * 32);
    AffinePoint h_ped = H_gen[0];

    JacobianPoint result = pedersen_commit(val, blind, h_ped);

    // Convert Jacobian to affine and output as bytes (x || y, 64 bytes)
    AffinePoint aff = jacobian_to_affine(result);
    field_to_bytes(aff.x, commitments_out + tid * 64);
    field_to_bytes(aff.y, commitments_out + tid * 64 + 32);
}

// =============================================================================
// Kernel 27: Pedersen Verify Sum (homomorphic)
// =============================================================================

kernel void pedersen_verify_sum(
    device const AffinePoint *pos          [[buffer(0)]],
    constant uint &n_pos                   [[buffer(1)]],
    device const AffinePoint *neg          [[buffer(2)]],
    constant uint &n_neg                   [[buffer(3)]],
    device uint *result                    [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;

    JacobianPoint sum = point_at_infinity();

    for (uint i = 0; i < n_pos; ++i) {
        sum = jacobian_add_mixed(sum, pos[i]);
    }

    for (uint i = 0; i < n_neg; ++i) {
        AffinePoint neg_pt = neg[i];
        neg_pt.y = field_negate(neg_pt.y);
        sum = jacobian_add_mixed(sum, neg_pt);
    }

    // Check if sum is infinity (Z == 0)
    if (sum.infinity) { result[0] = 1u; return; }

    uchar z_bytes[32];
    field_to_bytes(sum.z, z_bytes);
    uint z_zero = 1u;
    for (int i = 0; i < 32; i++)
        if (z_bytes[i] != 0) z_zero = 0u;
    result[0] = z_zero;
}

// =============================================================================
// Kernel 20: FROST Partial Signature Verification
// =============================================================================
// Verifies one FROST-secp256k1 partial signature per thread.
//
// Algorithm (per thread i):
//   R_i = D_i + rho_i * E_i           (nonce combination)
//   lhs = z_i * G                     (partial sig on generator)
//   rhs = R_i + lambda_i_e * Y_i      (weighted verification share)
//   valid = (lhs == rhs)
//
// Optionally negate R_i (negate_R[i] != 0) and/or Y_i (negate_key[i] != 0)
// to handle even-y conventions.
// =============================================================================

// ---- Internal helpers -------------------------------------------------------

// Decompress a 33-byte SEC1 compressed point to a JacobianPoint (Z=1).
// Returns false on invalid prefix or non-square y.
inline bool frost_decompress_sec1(device const uchar* sec1_33,
                                   thread JacobianPoint &out)
{
    uchar prefix = sec1_33[0];
    if (prefix != 0x02u && prefix != 0x03u) return false;
    int parity = (prefix == 0x03u) ? 1 : 0;

    FieldElement x;
    for (int i = 0; i < 8; i++) {
        uint off = (uint)(1 + (7 - i) * 4);
        x.limbs[i] = ((uint)sec1_33[off]   << 24) |
                     ((uint)sec1_33[off+1]  << 16) |
                     ((uint)sec1_33[off+2]  << 8)  |
                     ((uint)sec1_33[off+3]);
    }
    return lift_x_field(x, parity, out);
}

// Read a 32-byte big-endian scalar from a device buffer at byte offset.
inline Scalar256 frost_read_scalar(device const uchar* buf, uint tid)
{
    Scalar256 s;
    uint off = tid * 32;
    for (int i = 0; i < 8; i++) {
        uint b = off + (uint)(7 - i) * 4;
        s.limbs[i] = ((uint)buf[b]   << 24) |
                     ((uint)buf[b+1] << 16) |
                     ((uint)buf[b+2] << 8)  |
                     ((uint)buf[b+3]);
    }
    return s;
}

// Compare two JacobianPoints for equality via affine x bytes + y parity.
// Returns true if both represent the same curve point.
inline bool frost_jac_equal(thread const JacobianPoint &a_in,
                             thread const JacobianPoint &b_in)
{
    if (a_in.infinity && b_in.infinity) return true;
    if (a_in.infinity || b_in.infinity) return false;

    AffinePoint a_aff = jacobian_to_affine(a_in);
    AffinePoint b_aff = jacobian_to_affine(b_in);

    uchar ax[32], ay[32], bx[32], by[32];
    field_to_bytes(a_aff.x, ax);
    field_to_bytes(a_aff.y, ay);
    field_to_bytes(b_aff.x, bx);
    field_to_bytes(b_aff.y, by);

    for (int i = 0; i < 32; i++)
        if (ax[i] != bx[i] || ay[i] != by[i]) return false;
    return true;
}

// ---- Kernel -----------------------------------------------------------------

kernel void frost_verify_partial_batch(
    device const uchar *z_i32         [[buffer(0)]],   // N × 32  partial sig scalar
    device const uchar *D_i33         [[buffer(1)]],   // N × 33  hiding nonce
    device const uchar *E_i33         [[buffer(2)]],   // N × 33  binding nonce
    device const uchar *Y_i33         [[buffer(3)]],   // N × 33  verification share
    device const uchar *rho_i32       [[buffer(4)]],   // N × 32  binding factor
    device const uchar *lambda_ie32   [[buffer(5)]],   // N × 32  lambda_i * e
    device const uchar *negate_R      [[buffer(6)]],   // N × 1   negate R flag
    device const uchar *negate_key    [[buffer(7)]],   // N × 1   negate Y flag
    device uint        *results       [[buffer(8)]],   // N × 1   output (1=valid)
    constant uint      &count         [[buffer(9)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) { return; }

    /* Parse scalars */
    Scalar256 z_i      = frost_read_scalar(z_i32,      tid);
    Scalar256 rho_i    = frost_read_scalar(rho_i32,    tid);
    Scalar256 lambda_ie = frost_read_scalar(lambda_ie32, tid);

    /* Decompress points */
    JacobianPoint D_jac, E_jac, Y_jac;
    if (!frost_decompress_sec1(D_i33 + tid * 33u, D_jac)) { results[tid] = 0u; return; }
    if (!frost_decompress_sec1(E_i33 + tid * 33u, E_jac)) { results[tid] = 0u; return; }
    if (!frost_decompress_sec1(Y_i33 + tid * 33u, Y_jac)) { results[tid] = 0u; return; }

    /* Extract affine for scalar_mul_glv (Z=1 after decompress) */
    AffinePoint E_aff = { E_jac.x, E_jac.y };
    AffinePoint Y_aff = { Y_jac.x, Y_jac.y };

    /* Optionally negate Y_i */
    if (negate_key[tid]) {
        Y_aff.y = field_negate(Y_aff.y);
    }

    /* R_i = D_i + rho_i * E_i */
    JacobianPoint rho_E = scalar_mul_glv(E_aff, rho_i);
    JacobianPoint R_i   = jacobian_add(D_jac, rho_E);

    /* Optionally negate R_i */
    if (negate_R[tid]) {
        R_i.y = field_negate(R_i.y);
    }

    /* lhs = z_i * G */
    JacobianPoint lhs = scalar_mul_generator_windowed(z_i);

    /* rhs = R_i + lambda_i_e * Y_i */
    JacobianPoint lambda_Y = scalar_mul_glv(Y_aff, lambda_ie);
    JacobianPoint rhs      = jacobian_add(R_i, lambda_Y);

    /* Compare */
    results[tid] = frost_jac_equal(lhs, rhs) ? 1u : 0u;
}

// =============================================================================
// BIP-324 Kernels — Batch ChaCha20-Poly1305 AEAD
// =============================================================================

// Kernel: batch ChaCha20 block (throughput ceiling)
kernel void kernel_bip324_chacha20_block_batch(
    device const uchar* keys    [[buffer(0)]],   // N * 32
    device const uchar* nonces  [[buffer(1)]],   // N * 12
    device uchar*       out     [[buffer(2)]],   // N * 64
    constant uint&      count   [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint state[16];
    bip324_chacha20_setup(state, keys + tid * 32, nonces + tid * 12, 0);

    uchar block[64];
    bip324_chacha20_block(state, block);

    for (int i = 0; i < 64; ++i)
        out[tid * 64 + i] = block[i];
}

// Kernel: batch AEAD encrypt
kernel void kernel_bip324_aead_encrypt(
    device const uchar* keys        [[buffer(0)]],   // N * 32
    device const uchar* nonces      [[buffer(1)]],   // N * 12
    device const uchar* plaintexts  [[buffer(2)]],   // N * max_payload
    device const uint*  sizes       [[buffer(3)]],   // N payload sizes
    device uchar*       wire_out    [[buffer(4)]],   // N * (max_payload + 19)
    constant uint&      max_payload [[buffer(5)]],
    constant uint&      count       [[buffer(6)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint payload_sz = sizes[tid];
    device const uchar* key   = keys + tid * 32;
    device const uchar* nonce = nonces + tid * 12;
    device const uchar* pt    = plaintexts + (ulong)tid * max_payload;

    ulong wire_stride = max_payload + BIP324_OVERHEAD;
    device uchar* wire = wire_out + (ulong)tid * wire_stride;

    // BIP-324: 3-byte LE length header (simplified for benchmark)
    wire[0] = (uchar)(payload_sz);
    wire[1] = (uchar)(payload_sz >> 8);
    wire[2] = (uchar)(payload_sz >> 16);

    bip324_aead_encrypt(key, nonce, pt, payload_sz,
                        wire + 3, wire + 3 + payload_sz);
}

// Kernel: batch AEAD decrypt
kernel void kernel_bip324_aead_decrypt(
    device const uchar* keys         [[buffer(0)]],
    device const uchar* nonces       [[buffer(1)]],
    device const uchar* wire_in      [[buffer(2)]],   // N * (max_payload + 19)
    device const uint*  sizes        [[buffer(3)]],
    device uchar*       plaintext_out[[buffer(4)]],   // N * max_payload
    device uint*        ok           [[buffer(5)]],   // N success flags
    constant uint&      max_payload  [[buffer(6)]],
    constant uint&      count        [[buffer(7)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    uint payload_sz = sizes[tid];
    device const uchar* key   = keys + tid * 32;
    device const uchar* nonce = nonces + tid * 12;

    ulong wire_stride = max_payload + BIP324_OVERHEAD;
    device const uchar* wire = wire_in + (ulong)tid * wire_stride;
    device uchar* pt_out = plaintext_out + (ulong)tid * max_payload;

    device const uchar* ct  = wire + 3;
    device const uchar* tag = wire + 3 + payload_sz;

    ok[tid] = bip324_aead_decrypt(key, nonce, ct, payload_sz, tag, pt_out) ? 1u : 0u;
}

// =============================================================================
// MSM Block Sum Kernel — GPU second pass for msm()
// Each thread (one per block) sums 256 scalar_mul AffinePoint results into one
// JacobianPoint without field inversions (uses jacobian_add_mixed internally).
// dispatch: grid_size=n_blocks, threadgroup_size=1
// =============================================================================
kernel void msm_block_sum_kernel(
    device const AffinePoint*  partials      [[buffer(0)]],
    constant uint&             n             [[buffer(1)]],
    device       JacobianPoint* block_results [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint start = gid * 256u;
    if (start >= n) { block_results[gid] = point_at_infinity(); return; }
    uint end = (start + 256u <= n) ? start + 256u : n;
    JacobianPoint acc = point_at_infinity();
    for (uint i = start; i < end; i++) {
        acc = jacobian_add_mixed(acc, partials[i]);
    }
    block_results[gid] = acc;
}

// =============================================================================
// BIP-352 Silent Payment scan pipeline
// =============================================================================
// Pipeline per thread:
//   1. shared  = scan_scalar × tweak_points[tid]   (scalar_mul_glv)
//   2. ser[37] = compress(shared) ++ [0,0,0,0]     (SEC1 + k=0 index)
//   3. hash    = SHA256("BIP0352/SharedSecret" || ser)  (tagged hash midstate)
//   4. out_pt  = hash × G                          (scalar_mul_generator_windowed)
//   5. cand    = out_pt + spend_point              (jacobian_add_mixed)
//   6. prefix  = upper 64 bits of cand.x           (Jacobian → affine → extract)
//
// Buffers:
//   0: device const AffinePoint* tweak_points  — N decompressed tweak points
//   1: constant Scalar256&       scan_scalar   — scan private key as Scalar256
//   2: device const AffinePoint* spend_point   — spend public key (1 point)
//   3: device ulong*             prefixes      — N × 8-byte output prefix
//   4: constant uint&            count         — N
// =============================================================================

constant uint BIP352_SHAREDSECRET_MIDSTATE[8] = {
    0x88831537U, 0x5127079bU, 0x69c2137bU, 0xab0303e6U,
    0x98fa21faU, 0x4a888523U, 0xbd99daabU, 0xf25e5e0aU
};

kernel void bip352_scan_pipeline(
    device const AffinePoint* tweak_points [[buffer(0)]],
    constant Scalar256&       scan_scalar  [[buffer(1)]],
    device const AffinePoint* spend_point  [[buffer(2)]],
    device ulong*             prefixes     [[buffer(3)]],
    constant uint&            count        [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    // Phase 1: shared_secret = scan_scalar × tweak_points[tid]
    // Copy constant-address-space scalar to thread-local so scalar_mul_glv
    // (which takes `thread const Scalar256&`) can accept it.
    AffinePoint tweak = tweak_points[tid];
    thread Scalar256 local_scan_scalar = scan_scalar;
    JacobianPoint shared = scalar_mul_glv(tweak, local_scan_scalar);
    if (shared.infinity != 0) { prefixes[tid] = 0; return; }

    // Phase 2: serialize to SEC1 compressed + 4 zero bytes (output index k=0)
    AffinePoint shared_aff = jacobian_to_affine(shared);
    uchar x_bytes[32], y_bytes[32];
    field_to_bytes(shared_aff.x, x_bytes);
    field_to_bytes(shared_aff.y, y_bytes);

    uchar ser[37];
    ser[0] = (y_bytes[31] & 1u) ? 0x03 : 0x02;
    for (int i = 0; i < 32; i++) ser[1 + i] = x_bytes[i];
    ser[33] = 0; ser[34] = 0; ser[35] = 0; ser[36] = 0;

    // Phase 3: tagged SHA-256 with precomputed BIP0352/SharedSecret midstate
    SHA256Ctx sha_ctx;
    for (int i = 0; i < 8; i++) sha_ctx.h[i] = BIP352_SHAREDSECRET_MIDSTATE[i];
    sha_ctx.buf_len = 0;
    sha_ctx.total_len_lo = 64; // midstate already consumed one 64-byte tag block
    sha_ctx.total_len_hi = 0;
    sha256_update(sha_ctx, ser, 37);
    uchar hash[32];
    sha256_final(sha_ctx, hash);

    // Phase 4: hash_scalar × G
    Scalar256 hs = scalar_from_bytes(hash);
    JacobianPoint out_pt = scalar_mul_generator_windowed(hs);

    // Phase 5: cand = out_pt + spend_point[0]
    AffinePoint spend = spend_point[0];
    JacobianPoint cand = jacobian_add_mixed(out_pt, spend);
    if (cand.infinity != 0) { prefixes[tid] = 0; return; }

    // Phase 6: extract upper 64 bits of cand.x (Jacobian → affine → BE bytes)
    AffinePoint cand_aff = jacobian_to_affine(cand);
    uchar cx[32];
    field_to_bytes(cand_aff.x, cx);
    ulong prefix = 0;
    for (int i = 0; i < 8; i++) prefix = (prefix << 8) | ulong(cx[i]);
    prefixes[tid] = prefix;
}

// =============================================================================
// ECDSA SNARK Witness Batch
// =============================================================================
// Computes one 760-byte foreign-field PLONK witness record per thread.
// Record layout (identical to CUDA and OpenCL):
//   11 × 32 bytes (msg, r, s, pub_x, pub_y, s_inv, u1, u2, rx, ry, rx_mod_n)
//   10 × 5 × 8 bytes (52-bit foreign-field limbs for each of the above scalars)
//   1 × 4 bytes (valid flag) + 1 × 4 bytes (padding) = 760 bytes total
//
// Buffers:
//   0: device const uchar* msg_hashes  — N × 32 bytes (BE SHA-256)
//   1: device const uchar* pubkeys64   — N × 64 bytes (x‖y big-endian, decompressed)
//   2: device const uchar* sigs64      — N × 64 bytes (r‖s big-endian)
//   3: device uchar*       out_flat    — N × 760 bytes
//   4: constant uint&      count       — N
// =============================================================================

// Helper: Scalar256 → 5 × 52-bit foreign-field limbs (ulong each).
inline void scalar_to_ff_limbs(thread const Scalar256 &s, thread ulong out[5]) {
    // Reconstruct 4 × ulong LE from 8 × uint LE Metal limbs
    ulong w0 = ulong(s.limbs[0]) | (ulong(s.limbs[1]) << 32);
    ulong w1 = ulong(s.limbs[2]) | (ulong(s.limbs[3]) << 32);
    ulong w2 = ulong(s.limbs[4]) | (ulong(s.limbs[5]) << 32);
    ulong w3 = ulong(s.limbs[6]) | (ulong(s.limbs[7]) << 32);
    const ulong M52 = (1UL << 52) - 1UL;
    out[0] =  w0                          & M52;
    out[1] = ((w0 >> 52) | (w1 << 12))   & M52;
    out[2] = ((w1 >> 40) | (w2 << 24))   & M52;
    out[3] = ((w2 >> 28) | (w3 << 36))   & M52;
    out[4] =   w3 >> 16;
}

// Helper: 32 BE bytes → 5 × 52-bit foreign-field limbs (ulong each).
inline void be32_to_ff_limbs(thread const uchar be[32], thread ulong out[5]) {
    ulong w[4];
    for (int i = 0; i < 4; i++) {
        ulong v = 0;
        int base = (3 - i) * 8;
        for (int b = 0; b < 8; b++) v = (v << 8) | ulong(be[base + b]);
        w[i] = v;
    }
    const ulong M52 = (1UL << 52) - 1UL;
    out[0] =  w[0]                          & M52;
    out[1] = ((w[0] >> 52) | (w[1] << 12)) & M52;
    out[2] = ((w[1] >> 40) | (w[2] << 24)) & M52;
    out[3] = ((w[2] >> 28) | (w[3] << 36)) & M52;
    out[4] =   w[3] >> 16;
}

// Helper: write uint at a byte offset into a device buffer.
inline void write_u32_le(device uchar* buf, uint offset, int val) {
    uint v = uint(val);
    buf[offset+0] = uchar(v & 0xFFu);
    buf[offset+1] = uchar((v >> 8) & 0xFFu);
    buf[offset+2] = uchar((v >> 16) & 0xFFu);
    buf[offset+3] = uchar((v >> 24) & 0xFFu);
}

// Helper: write ulong FF limbs at byte offset (LE 8 bytes each limb).
inline void write_ff_limbs(device uchar* buf, uint offset, thread ulong limbs[5]) {
    for (int i = 0; i < 5; i++) {
        ulong v = limbs[i];
        for (int b = 0; b < 8; b++) buf[offset + i*8 + b] = uchar((v >> (b*8)) & 0xFFu);
    }
}

kernel void ecdsa_snark_witness_batch(
    device const uchar* msg_hashes  [[buffer(0)]],
    device const uchar* pubkeys64   [[buffer(1)]],
    device const uchar* sigs64      [[buffer(2)]],
    device       uchar* out_flat    [[buffer(3)]],
    constant uint&      count       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    // Output record: 760 bytes starting at tid*760
    const uint rec_off = tid * 760u;

    // -- zero the record (valid=0 by default) --
    for (uint i = 0; i < 760u; i++) out_flat[rec_off + i] = 0;

    // -- load message hash (32 BE bytes) --
    uchar msg[32];
    for (int i = 0; i < 32; i++) msg[i] = msg_hashes[tid * 32 + i];

    // -- load sig r and s (each 32 BE bytes) --
    uchar r_be[32], s_be[32];
    for (int i = 0; i < 32; i++) {
        r_be[i] = sigs64[tid * 64 + i];
        s_be[i] = sigs64[tid * 64 + 32 + i];
    }
    Scalar256 sig_r = scalar_from_bytes(r_be);
    Scalar256 sig_s = scalar_from_bytes(s_be);
    if (scalar256_is_zero(sig_r) || scalar256_is_zero(sig_s)) return;

    // -- load decompressed pubkey (64 BE bytes: x‖y) --
    uchar pub_x_be[32], pub_y_be[32];
    for (int i = 0; i < 32; i++) {
        pub_x_be[i] = pubkeys64[tid * 64 + i];
        pub_y_be[i] = pubkeys64[tid * 64 + 32 + i];
    }
    AffinePoint pub_aff;
    // Load x limbs from BE bytes (LE limb order: limbs[0]=LSW, limbs[7]=MSW)
    for (int i = 0; i < 8; i++) {
        int base = (7 - i) * 4;
        pub_aff.x.limbs[i] = (uint(pub_x_be[base])   << 24) |
                              (uint(pub_x_be[base+1]) << 16) |
                              (uint(pub_x_be[base+2]) << 8)  |
                              (uint(pub_x_be[base+3]));
        pub_aff.y.limbs[i] = (uint(pub_y_be[base])   << 24) |
                              (uint(pub_y_be[base+1]) << 16) |
                              (uint(pub_y_be[base+2]) << 8)  |
                              (uint(pub_y_be[base+3]));
    }
    // pub_aff is a valid non-infinity point (AffinePoint has no infinity field).

    // -- witness scalars: z, s_inv, u1, u2 --
    Scalar256 z     = scalar_from_bytes(msg);
    Scalar256 s_inv = scalar_inverse(sig_s);
    Scalar256 u1    = scalar_mul_mod_n(z,     s_inv);
    Scalar256 u2    = scalar_mul_mod_n(sig_r, s_inv);

    // -- R = u1·G + u2·Q using separate muls + jacobian_add --
    AffinePoint G  = generator_affine();
    JacobianPoint u1G = scalar_mul_glv(G,       u1);
    JacobianPoint u2Q = scalar_mul_glv(pub_aff, u2);
    JacobianPoint R   = jacobian_add(u1G, u2Q);
    if (R.infinity != 0) return;

    // -- R affine --
    AffinePoint R_aff = jacobian_to_affine(R);
    uchar rx[32], ry[32];
    field_to_bytes(R_aff.x, rx);
    field_to_bytes(R_aff.y, ry);

    // -- result_x mod n = scalar parse of rx --
    Scalar256 v = scalar_from_bytes(rx);
    uchar rx_mod_n[32];
    scalar_to_bytes(v, rx_mod_n);

    // -- validity: v == sig_r --
    bool valid = scalar256_eq(v, sig_r);

    // ---- serialize into flat 760-byte record ----
    // bytes 0-31:   msg
    for (int i = 0; i < 32; i++) out_flat[rec_off +   0 + i] = msg[i];
    // bytes 32-63:  sig_r
    for (int i = 0; i < 32; i++) out_flat[rec_off +  32 + i] = r_be[i];
    // bytes 64-95:  sig_s
    for (int i = 0; i < 32; i++) out_flat[rec_off +  64 + i] = s_be[i];
    // bytes 96-127: pub_x
    for (int i = 0; i < 32; i++) out_flat[rec_off +  96 + i] = pub_x_be[i];
    // bytes 128-159: pub_y
    for (int i = 0; i < 32; i++) out_flat[rec_off + 128 + i] = pub_y_be[i];
    // bytes 160-191: s_inv
    uchar s_inv_bytes[32]; scalar_to_bytes(s_inv, s_inv_bytes);
    for (int i = 0; i < 32; i++) out_flat[rec_off + 160 + i] = s_inv_bytes[i];
    // bytes 192-223: u1
    uchar u1_bytes[32]; scalar_to_bytes(u1, u1_bytes);
    for (int i = 0; i < 32; i++) out_flat[rec_off + 192 + i] = u1_bytes[i];
    // bytes 224-255: u2
    uchar u2_bytes[32]; scalar_to_bytes(u2, u2_bytes);
    for (int i = 0; i < 32; i++) out_flat[rec_off + 224 + i] = u2_bytes[i];
    // bytes 256-287: result_x
    for (int i = 0; i < 32; i++) out_flat[rec_off + 256 + i] = rx[i];
    // bytes 288-319: result_y
    for (int i = 0; i < 32; i++) out_flat[rec_off + 288 + i] = ry[i];
    // bytes 320-351: result_x_mod_n
    for (int i = 0; i < 32; i++) out_flat[rec_off + 352 - 32 + i] = rx_mod_n[i];

    // 10 foreign-field limb arrays × 5 × 8 bytes = 400 bytes, offset 352..751
    // Each array is exactly 40 bytes (5 × ulong) with step = 40.
    ulong lmb[5];
    scalar_to_ff_limbs(sig_r,  lmb); write_ff_limbs(out_flat, rec_off + 352, lmb); // [0]
    scalar_to_ff_limbs(sig_s,  lmb); write_ff_limbs(out_flat, rec_off + 392, lmb); // [1]
    be32_to_ff_limbs(pub_x_be, lmb); write_ff_limbs(out_flat, rec_off + 432, lmb); // [2]
    be32_to_ff_limbs(pub_y_be, lmb); write_ff_limbs(out_flat, rec_off + 472, lmb); // [3]
    scalar_to_ff_limbs(s_inv,  lmb); write_ff_limbs(out_flat, rec_off + 512, lmb); // [4]
    scalar_to_ff_limbs(u1,     lmb); write_ff_limbs(out_flat, rec_off + 552, lmb); // [5]
    scalar_to_ff_limbs(u2,     lmb); write_ff_limbs(out_flat, rec_off + 592, lmb); // [6]
    be32_to_ff_limbs(rx,       lmb); write_ff_limbs(out_flat, rec_off + 632, lmb); // [7]
    be32_to_ff_limbs(ry,       lmb); write_ff_limbs(out_flat, rec_off + 672, lmb); // [8]
    be32_to_ff_limbs(rx_mod_n, lmb); write_ff_limbs(out_flat, rec_off + 712, lmb); // [9]
    // bytes 752-755: valid (int32 LE)
    write_u32_le(out_flat, rec_off + 752, valid ? 1 : 0);
    // bytes 756-759: _pad (already zero)
    // Total record size: 352 + 400 + 8 = 760 bytes ✓
}

// =============================================================================
// BIP-340 Schnorr SNARK witness batch (eprint 2025/695)
// Output: 472-byte flat records, one per thread.
// Layout: msg[32] sig_r[32] sig_s[32] pub_x[32] r_y[32] pub_y[32] e[32]
//         + 6×40-byte FF-limb blocks + valid(4) + _pad(4)
// =============================================================================

kernel void schnorr_snark_witness_batch(
    device const uchar* msgs32      [[buffer(0)]],  // count × 32 bytes
    device const uchar* pubkeys_x32 [[buffer(1)]],  // count × 32 bytes (x-only)
    device const uchar* sigs64      [[buffer(2)]],  // count × 64 bytes (r||s)
    device       uchar* out_flat    [[buffer(3)]],
    constant uint&      count       [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) return;

    const uint rec_off = tid * 472u;
    for (uint i = 0; i < 472u; i++) out_flat[rec_off + i] = 0;

    // Load inputs
    uchar msg[32], pub_x[32], r_be[32], s_be[32];
    for (int i = 0; i < 32; i++) msg[i]   = msgs32[tid * 32 + i];
    for (int i = 0; i < 32; i++) pub_x[i] = pubkeys_x32[tid * 32 + i];
    for (int i = 0; i < 32; i++) r_be[i]  = sigs64[tid * 64 + i];
    for (int i = 0; i < 32; i++) s_be[i]  = sigs64[tid * 64 + 32 + i];

    // Validate r < p (secp256k1 field prime)
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    const uchar P_BYTES[32] = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFE, 0xFF,0xFF,0xFC,0x2F
    };
    bool r_valid = false;
    for (int i = 0; i < 32; i++) {
        if (r_be[i] < P_BYTES[i]) { r_valid = true;  break; }
        if (r_be[i] > P_BYTES[i]) { r_valid = false; break; }
    }
    if (!r_valid) return;

    // Validate s in [1, n-1] (secp256k1 group order)
    // n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
    bool s_nonzero = false;
    for (int i = 0; i < 32; i++) if (s_be[i] != 0) { s_nonzero = true; break; }
    if (!s_nonzero) return;

    const uchar N_BYTES[32] = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
    };
    bool s_valid = false;
    for (int i = 0; i < 32; i++) {
        if (s_be[i] < N_BYTES[i]) { s_valid = true;  break; }
        if (s_be[i] > N_BYTES[i]) { s_valid = false; break; }
    }
    if (!s_valid) return;

    // lift_x(pub_x) → pub_y (even Y, BIP-340)
    JacobianPoint P;
    if (!lift_x(pub_x, P)) return;
    uchar pub_y[32];
    field_to_bytes(P.y, pub_y);

    // lift_x(sig_r) → r_y (even Y, BIP-340)
    JacobianPoint R;
    if (!lift_x(r_be, R)) return;
    uchar r_y[32];
    field_to_bytes(R.y, r_y);

    // Challenge e = tagged_hash("BIP0340/challenge", r_be || pub_x || msg)
    uchar challenge_in[96];
    for (int i = 0; i < 32; i++) challenge_in[i]      = r_be[i];
    for (int i = 0; i < 32; i++) challenge_in[32 + i] = pub_x[i];
    for (int i = 0; i < 32; i++) challenge_in[64 + i] = msg[i];
    uchar e_hash[32];
    tagged_hash_fast(2 /* BIP340_TAG_CHALLENGE */, challenge_in, 96, e_hash);

    // Serialize 472-byte flat record
    // bytes 0-31: msg
    for (int i = 0; i < 32; i++) out_flat[rec_off +   0 + i] = msg[i];
    // bytes 32-63: sig_r
    for (int i = 0; i < 32; i++) out_flat[rec_off +  32 + i] = r_be[i];
    // bytes 64-95: sig_s
    for (int i = 0; i < 32; i++) out_flat[rec_off +  64 + i] = s_be[i];
    // bytes 96-127: pub_x
    for (int i = 0; i < 32; i++) out_flat[rec_off +  96 + i] = pub_x[i];
    // bytes 128-159: r_y
    for (int i = 0; i < 32; i++) out_flat[rec_off + 128 + i] = r_y[i];
    // bytes 160-191: pub_y
    for (int i = 0; i < 32; i++) out_flat[rec_off + 160 + i] = pub_y[i];
    // bytes 192-223: e
    for (int i = 0; i < 32; i++) out_flat[rec_off + 192 + i] = e_hash[i];

    // 6 foreign-field limb arrays × 40 bytes = 240 bytes (bytes 224..463)
    Scalar256 sig_s = scalar_from_bytes(s_be);
    ulong lmb[5];
    be32_to_ff_limbs(r_be,    lmb); write_ff_limbs(out_flat, rec_off + 224, lmb); // sig_r
    scalar_to_ff_limbs(sig_s, lmb); write_ff_limbs(out_flat, rec_off + 264, lmb); // sig_s
    be32_to_ff_limbs(pub_x,   lmb); write_ff_limbs(out_flat, rec_off + 304, lmb); // pub_x
    be32_to_ff_limbs(r_y,     lmb); write_ff_limbs(out_flat, rec_off + 344, lmb); // r_y
    be32_to_ff_limbs(pub_y,   lmb); write_ff_limbs(out_flat, rec_off + 384, lmb); // pub_y
    be32_to_ff_limbs(e_hash,  lmb); write_ff_limbs(out_flat, rec_off + 424, lmb); // e

    // bytes 464-467: valid = 1 (LE int32)
    write_u32_le(out_flat, rec_off + 464, 1);
    // bytes 468-471: _pad (already zero)
    // Total record size: 7×32 + 6×40 + 8 = 224 + 240 + 8 = 472 bytes ✓
}

// =============================================================================
// CT Smoke Kernel -- Branchless Constant-Time Layer (Section 11)
// =============================================================================
// Exercises CT primitives, CT field/scalar/point ops, CT ECDSA sign, CT Schnorr
// sign.  All subtests run on thread 0; other threads return immediately.
//
// Output layout (out[0..3]):
//   out[0]: CT mask primitives (0 = pass)
//   out[1]: CT cmov / cswap    (0 = pass)
//   out[2]: CT ECDSA sign+verify with privkey=1 (0 = pass)
//   out[3]: CT Schnorr sign+verify with privkey=1 (0 = pass)
// =============================================================================

// SHA-256("test")
constant uchar CT_SMOKE_MSG[32] = {
    0x9f, 0x86, 0xd0, 0x81, 0x88, 0x4c, 0x7d, 0x65,
    0x9a, 0x2f, 0xea, 0xa0, 0xc5, 0x5a, 0xd0, 0x15,
    0xa3, 0xbf, 0x4f, 0x1b, 0x2b, 0x0b, 0x82, 0x2c,
    0xd1, 0x5d, 0x6c, 0x15, 0xb0, 0xf0, 0x0a, 0x08
};

constant uchar CT_SMOKE_AUX[32] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

kernel void ct_smoke_kernel(device int* out [[buffer(0)]],
                             uint gid [[thread_position_in_grid]]) {
    if (gid != 0) return;

    // ---- Subtest 0: CT mask primitives ----
    {
        int fail = 0;
        uint zm0 = ct_is_zero_mask(0u);
        uint zm1 = ct_is_zero_mask(1u);
        if (zm0 != 0xFFFFFFFFu) fail |= 1;
        if (zm1 != 0u)          fail |= 1;

        uint nz0 = ct_is_nonzero_mask(0u);
        uint nz1 = ct_is_nonzero_mask(42u);
        if (nz0 != 0u)          fail |= 2;
        if (nz1 != 0xFFFFFFFFu) fail |= 2;

        uint eq0 = ct_eq_mask(7u, 7u);
        uint eq1 = ct_eq_mask(7u, 8u);
        if (eq0 != 0xFFFFFFFFu) fail |= 4;
        if (eq1 != 0u)          fail |= 4;

        if (ct_bool_to_mask(true)  != 0xFFFFFFFFu) fail |= 8;
        if (ct_bool_to_mask(false) != 0u)          fail |= 16;

        out[0] = fail;
    }

    // ---- Subtest 1: CT cmov / cswap on 8-limb arrays ----
    {
        int fail = 0;
        uint all1 = 0xFFFFFFFFu, all0 = 0u;

        uint a[8] = {1,2,3,4,5,6,7,8};
        uint b[8] = {9,10,11,12,13,14,15,16};

        // cmov: mask=all-ones -> copy b into r
        uint r[8] = {1,2,3,4,5,6,7,8};
        ct_cmov_8limb(r, b, all1);
        for (int i = 0; i < 8; ++i) if (r[i] != b[i]) { fail |= 1; break; }

        // cmov: mask=0 -> keep a
        uint r2[8] = {1,2,3,4,5,6,7,8};
        ct_cmov_8limb(r2, b, all0);
        for (int i = 0; i < 8; ++i) if (r2[i] != a[i]) { fail |= 2; break; }

        // cswap: mask=all-ones -> swap
        uint sa[8] = {1,2,3,4,5,6,7,8};
        uint sb[8] = {9,10,11,12,13,14,15,16};
        ct_cswap_8limb(sa, sb, all1);
        for (int i = 0; i < 8; ++i) {
            if (sa[i] != b[i] || sb[i] != a[i]) { fail |= 4; break; }
        }

        // cswap: mask=0 -> no swap
        uint ca[8] = {1,2,3,4,5,6,7,8};
        uint cb[8] = {9,10,11,12,13,14,15,16};
        ct_cswap_8limb(ca, cb, all0);
        for (int i = 0; i < 8; ++i) {
            if (ca[i] != a[i] || cb[i] != b[i]) { fail |= 8; break; }
        }

        out[1] = fail;
    }

    // ---- Subtest 2: CT ECDSA sign (privkey=1) + fast-path verify ----
    {
        // privkey = scalar 1 (little-endian 8x32: limbs[0]=1, rest 0)
        Scalar256 priv;
        for (int i = 0; i < 8; ++i) priv.limbs[i] = 0;
        priv.limbs[0] = 1u;

        uchar msg_hash[32];
        for (int i = 0; i < 32; ++i) msg_hash[i] = CT_SMOKE_MSG[i];

        Scalar256 r_out, s_out;
        bool sign_ok = ct_ecdsa_sign_metal(msg_hash, priv, r_out, s_out);
        if (!sign_ok) { out[2] = 1; }
        else {
            // Derive pubkey = 1*G via CT
            CTJacobianPoint pub_ct = ct_generator_mul_metal(priv);
            JacobianPoint pub_jac = ct_to_jacobian(pub_ct);
            AffinePoint pub = jacobian_to_affine(pub_jac);

            // Verify via fast path
            Scalar256 msg_scalar = scalar_from_bytes(msg_hash);
            bool ok = ecdsa_verify(msg_scalar, pub, r_out, s_out);
            out[2] = ok ? 0 : 2;
        }
    }

    // ---- Subtest 3: CT Schnorr sign (privkey=1) + fast-path verify ----
    {
        Scalar256 priv;
        for (int i = 0; i < 8; ++i) priv.limbs[i] = 0;
        priv.limbs[0] = 1u;

        uchar msg_bytes[32], aux_bytes[32];
        for (int i = 0; i < 32; ++i) msg_bytes[i] = CT_SMOKE_MSG[i];
        for (int i = 0; i < 32; ++i) aux_bytes[i] = CT_SMOKE_AUX[i];

        uchar sig64[64];
        bool sign_ok = ct_schnorr_sign_metal(priv, msg_bytes, aux_bytes, sig64);
        if (!sign_ok) { out[3] = 1; }
        else {
            // xonly pubkey via CT
            FieldElement pub_x = ct_schnorr_pubkey_metal(priv);
            uchar pub_x_bytes[32];
            field_to_bytes(pub_x, pub_x_bytes);

            // Reconstruct SchnorrSignature from raw bytes
            SchnorrSignature ssig;
            for (int i = 0; i < 32; ++i) ssig.r[i] = sig64[i];
            uchar s_bytes[32];
            for (int i = 0; i < 32; ++i) s_bytes[i] = sig64[32 + i];
            ssig.s = scalar_from_bytes(s_bytes);

            bool ok = schnorr_verify(pub_x_bytes, msg_bytes, ssig);
            out[3] = ok ? 0 : 2;
        }
    }
}

