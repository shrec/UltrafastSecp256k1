#ifndef SECP256K1_BIP352_CL
#define SECP256K1_BIP352_CL

#include "secp256k1_extended.cl"

// BIP352ScanKeyGlv: precomputed GLV scan-key plan uploaded to __constant memory.
// wNAF digits are computed on the CPU host and read directly in the kernel,
// avoiding the GPU call to scalar_to_wnaf and eliminating 1040 bytes of
// private-stack pressure (int wnaf1[130] + int wnaf2[130]).
// Layout must match the host-side BIP352ScanKeyGlv struct exactly.
typedef struct {
    char wnaf1[130]; // wNAF digits for k1 half-scalar (precomputed, range [-15..15])
    char wnaf2[130]; // wNAF digits for k2 half-scalar (precomputed, range [-15..15])
    uchar k1_neg;    // 1 if k1 was negative: negate base.y before table build
    uchar flip_phi;  // 1 if phi table y-coordinate should be negated
    uchar pad0;
    uchar pad1;
} BIP352ScanKeyGlv;

// SHA256("BIP0352/SharedSecret") || SHA256("BIP0352/SharedSecret")
__constant uint BIP352_SHAREDSECRET_MIDSTATE[8] = {
    0x88831537U, 0x5127079bU, 0x69c2137bU, 0xab0303e6U,
    0x98fa21faU, 0x4a888523U, 0xbd99daabU, 0xf25e5e0aU
};

inline void bip352_tagged_sha256_impl(const uchar* data, uint data_len, uchar out[32]) {
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = BIP352_SHAREDSECRET_MIDSTATE[i];
    ctx.buf_len = 0;
    ctx.total_len = 64;
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

inline void bip352_shared_secret_input_impl(const JacobianPoint* p, uchar ser[37]) {
    FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
    field_inv_impl(&z_inv, &p->z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv2, &z_inv);
    field_mul_impl(&x_aff, &p->x, &z_inv2);
    field_mul_impl(&y_aff, &p->y, &z_inv3);

    uchar x_bytes[32], y_bytes[32];
    field_to_bytes_impl(&x_aff, x_bytes);
    field_to_bytes_impl(&y_aff, y_bytes);

    ser[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
    for (int i = 0; i < 32; i++) ser[1 + i] = x_bytes[i];
    ser[33] = 0;
    ser[34] = 0;
    ser[35] = 0;
    ser[36] = 0;
}

inline ulong point_prefix64_impl(const JacobianPoint* p) {
    FieldElement z_inv, z_inv2, x_aff;
    field_inv_impl(&z_inv, &p->z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&x_aff, &p->x, &z_inv2);

    uchar x_bytes[32];
    field_to_bytes_impl(&x_aff, x_bytes);

    ulong prefix = 0;
    for (int i = 0; i < 8; i++) {
        prefix = (prefix << 8) | (ulong)x_bytes[i];
    }
    return prefix;
}

// Optimized GLV scalar multiply with pre-decomposed scan key.
// Uses build_wnaf_table_zr_impl (Z-trick affine table) + derive_endo_table_impl
// instead of the old Jacobian-Jacobian table -- eliminates 6 J-J adds per half,
// replaces with 7 mixed (J+A) adds and 1 field_inv shared across 8 entries.
// This matches the quality of scalar_mul_glv_impl in secp256k1_extended.cl.
inline void scalar_mul_glv_predecomp_impl(
    JacobianPoint* r,
    const AffinePoint* p,
    __constant const BIP352ScanKeyGlv* scan
) {
    AffinePoint base = *p;
    if (scan->k1_neg) field_negate_impl(&base.y, &base.y);

    // Build affine table[0..7] = {P, 3P, 5P, 7P, 9P, 11P, 13P, 15P} via Z-trick.
    // One field_inv for the whole table instead of per-point.
    AffinePoint table[8];
    FieldElement globalz;
    build_wnaf_table_zr_impl(&base, table, &globalz);

    // Endomorphism table: endo_table[i] = phi(table[i]) with optional Y-negate.
    AffinePoint endo_table[8];
    derive_endo_table_impl(table, endo_table, scan->flip_phi);

    // Shamir interleaved double-and-add with mixed (J+A) additions.
    // wNAF digits are read directly from __constant memory (precomputed on CPU host),
    // eliminating the GPU scalar_to_wnaf call and 1040 bytes of private stack.
    // Unchecked variants: r->infinity is guarded by the if(!point_is_infinity)
    // check above each call, mirroring the OpenCL fc378bdc optimization.
    point_set_infinity(r);
    for (int i = 129; i >= 0; --i) {
        if (!point_is_infinity(r)) point_double_unchecked(r, r);

        int d1 = (int)scan->wnaf1[i];
        if (d1 != 0) {
            int idx = (((d1 > 0) ? d1 : -d1) - 1) >> 1;
            AffinePoint pt = table[idx];
            if (d1 < 0) field_negate_impl(&pt.y, &pt.y);
            if (point_is_infinity(r)) { point_from_affine(r, &pt); }
            else { point_add_mixed_unchecked(r, r, &pt); }
        }

        int d2 = (int)scan->wnaf2[i];
        if (d2 != 0) {
            int idx = (((d2 > 0) ? d2 : -d2) - 1) >> 1;
            AffinePoint pt = endo_table[idx];
            if (d2 < 0) field_negate_impl(&pt.y, &pt.y);
            if (point_is_infinity(r)) { point_from_affine(r, &pt); }
            else { point_add_mixed_unchecked(r, r, &pt); }
        }
    }

    // Correct accumulated Z by the shared table Z factor.
    if (!point_is_infinity(r)) {
        FieldElement corrected_z;
        field_mul_impl(&corrected_z, &r->z, &globalz);
        r->z = corrected_z;
    }
}

__kernel void bip352_pipeline_kernel(
    __global const AffinePoint* tweak_points,
    __constant const BIP352ScanKeyGlv* scan_key,
    __global const AffinePoint* spend_point,
    __global ulong* prefixes,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    AffinePoint tweak = tweak_points[gid];
    AffinePoint spend = spend_point[0];

    JacobianPoint shared;
    scalar_mul_glv_predecomp_impl(&shared, &tweak, scan_key);
    if (point_is_infinity(&shared)) {
        prefixes[gid] = 0;
        return;
    }

    uchar ser[37];
    bip352_shared_secret_input_impl(&shared, ser);

    uchar hash[32];
    bip352_tagged_sha256_impl(ser, 37, hash);

    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);

    JacobianPoint out;
    scalar_mul_generator_windowed_impl(&out, &hs);

    JacobianPoint cand;
    point_add_mixed_impl(&cand, &out, &spend);
    prefixes[gid] = point_prefix64_impl(&cand);
}

__kernel void bip352_pipeline_kernel_lut(
    __global const AffinePoint* tweak_points,
    __constant const BIP352ScanKeyGlv* scan_key,
    __global const AffinePoint* spend_point,
    __global const AffinePoint* gen_lut,
    __global ulong* prefixes,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    AffinePoint tweak = tweak_points[gid];
    AffinePoint spend = spend_point[0];

    JacobianPoint shared;
    scalar_mul_glv_predecomp_impl(&shared, &tweak, scan_key);
    if (point_is_infinity(&shared)) {
        prefixes[gid] = 0;
        return;
    }

    uchar ser[37];
    bip352_shared_secret_input_impl(&shared, ser);

    uchar hash[32];
    bip352_tagged_sha256_impl(ser, 37, hash);

    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);

    JacobianPoint out;
    scalar_mul_generator_lut_impl(&out, &hs, gen_lut);

    JacobianPoint cand;
    point_add_mixed_impl(&cand, &out, &spend);
    prefixes[gid] = point_prefix64_impl(&cand);
}

#endif
