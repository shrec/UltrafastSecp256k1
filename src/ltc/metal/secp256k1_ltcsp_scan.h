// =============================================================================
// secp256k1_ltcsp_scan.h — Litecoin Silent Payments GPU Scanner (Metal)
// =============================================================================
// Metal Shading Language port of secp256k1_ltcsp_scan.cl.
// Identical algorithm to BIP-352 scanner; only LTCSP/ tagged hash differs.
//
// Throughput: not yet measured on Apple Silicon — benchmark required.
// Architecture differs significantly from NVIDIA/AMD: tile-based, shared CPU/GPU memory,
// different occupancy model. Do not extrapolate from OpenCL/CUDA numbers.
//
// Include order (in secp256k1_kernels.metal or similar):
//   #include "secp256k1_extended.h"
//   #include "secp256k1_ltcsp_scan.h"
// =============================================================================

#pragma once
#include <metal_stdlib>
using namespace metal;

// Precomputed scan-key GLV plan (host fills on CPU, passed as constant buffer).
struct LTCSPScanKeyGlv {
    char  wnaf1[130];
    char  wnaf2[130];
    uchar k1_neg;
    uchar flip_phi;
    uchar pad0;
    uchar pad1;
};

// SHA256("LTCSP/SharedSecret") midstate (precomputed):
//   python3 -c "import hashlib,struct;h=hashlib.sha256(b'LTCSP/SharedSecret').digest();
//               print([hex(w) for w in struct.unpack('>8I',h)])"
constant uint LTCSP_SHAREDSECRET_MIDSTATE[8] = {
    0x723B685Du,
    0x30E5495Fu,
    0x0A46E767u,
    0x1BD157DAu,
    0x2A9D6545u,
    0x4B2CC0B6u,
    0xBB6A9099u,
    0xECB73D0Bu
};

// H_LTCSP/SharedSecret(data) — Metal version
inline void ltcsp_tagged_sha256_metal(const uchar* data, uint data_len, thread uchar* out) {
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = LTCSP_SHAREDSECRET_MIDSTATE[i];
    ctx.buf_len   = 0;
    ctx.total_len = 64;
    sha256_update_metal(&ctx, data, data_len);
    sha256_final_metal(&ctx, out);
}

// Serialize Jacobian point → 33-byte compressed + 4 zero padding bytes
inline void ltcsp_shared_secret_input_metal(thread const JacobianPoint* p,
                                             thread uchar* ser) {
    FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
    field_inv_metal(&z_inv, &p->z);
    field_sqr_metal(&z_inv2, &z_inv);
    field_mul_metal(&z_inv3, &z_inv2, &z_inv);
    field_mul_metal(&x_aff, &p->x, &z_inv2);
    field_mul_metal(&y_aff, &p->y, &z_inv3);

    uchar x_bytes[32], y_bytes[32];
    field_to_bytes_metal(&x_aff, x_bytes);
    field_to_bytes_metal(&y_aff, y_bytes);

    ser[0] = (y_bytes[31] & 1) ? 0x03 : 0x02;
    for (int i = 0; i < 32; i++) ser[1 + i] = x_bytes[i];
    ser[33] = ser[34] = ser[35] = ser[36] = 0;
}

inline ulong ltcsp_point_prefix64_metal(thread const JacobianPoint* p) {
    FieldElement z_inv, z_inv2, x_aff;
    field_inv_metal(&z_inv, &p->z);
    field_sqr_metal(&z_inv2, &z_inv);
    field_mul_metal(&x_aff, &p->x, &z_inv2);

    uchar x_bytes[32];
    field_to_bytes_metal(&x_aff, x_bytes);

    ulong prefix = 0;
    for (int i = 0; i < 8; i++)
        prefix = (prefix << 8) | (ulong)x_bytes[i];
    return prefix;
}

inline void ltcsp_scalar_mul_glv_predecomp_metal(
    thread JacobianPoint*             r,
    thread const AffinePoint*         p,
    constant const LTCSPScanKeyGlv*   scan)
{
    AffinePoint base = *p;
    if (scan->k1_neg) field_negate_metal(&base.y, &base.y);

    AffinePoint table[8];
    FieldElement globalz;
    build_wnaf_table_zr_metal(&base, table, &globalz);

    AffinePoint endo_table[8];
    derive_endo_table_metal(table, endo_table, scan->flip_phi);

    point_set_infinity_metal(r);
    for (int i = 129; i >= 0; --i) {
        if (!point_is_infinity_metal(r)) point_double_unchecked_metal(r, r);

        int d1 = (int)scan->wnaf1[i];
        if (d1 != 0) {
            int idx = (((d1 > 0) ? d1 : -d1) - 1) >> 1;
            AffinePoint pt = table[idx];
            if (d1 < 0) field_negate_metal(&pt.y, &pt.y);
            if (point_is_infinity_metal(r)) point_from_affine_metal(r, &pt);
            else                             point_add_mixed_unchecked_metal(r, r, &pt);
        }

        int d2 = (int)scan->wnaf2[i];
        if (d2 != 0) {
            int idx = (((d2 > 0) ? d2 : -d2) - 1) >> 1;
            AffinePoint pt = endo_table[idx];
            if (d2 < 0) field_negate_metal(&pt.y, &pt.y);
            if (point_is_infinity_metal(r)) point_from_affine_metal(r, &pt);
            else                             point_add_mixed_unchecked_metal(r, r, &pt);
        }
    }

    if (!point_is_infinity_metal(r)) {
        FieldElement cz;
        field_mul_metal(&cz, &r->z, &globalz);
        r->z = cz;
    }
}

// =============================================================================
// Kernel: ltcsp_pipeline_kernel (GLV — no LUT)
// =============================================================================
kernel void ltcsp_pipeline_kernel(
    device const AffinePoint*       tweak_points  [[buffer(0)]],
    constant const LTCSPScanKeyGlv* scan_key      [[buffer(1)]],
    device const AffinePoint*       spend_point   [[buffer(2)]],
    device ulong*                   prefixes      [[buffer(3)]],
    constant uint&                  count         [[buffer(4)]],
    uint                            gid           [[thread_position_in_grid]])
{
    if (gid >= count) return;

    AffinePoint tweak = tweak_points[gid];
    AffinePoint spend = spend_point[0];

    JacobianPoint shared;
    ltcsp_scalar_mul_glv_predecomp_metal(&shared, &tweak, scan_key);
    if (point_is_infinity_metal(&shared)) { prefixes[gid] = 0; return; }

    uchar ser[37];
    ltcsp_shared_secret_input_metal(&shared, ser);

    uchar hash[32];
    ltcsp_tagged_sha256_metal(ser, 37, hash);

    Scalar hs;
    scalar_from_bytes_metal(hash, &hs);

    JacobianPoint out;
    scalar_mul_generator_windowed_metal(&out, &hs);

    JacobianPoint cand;
    point_add_mixed_metal(&cand, &out, &spend);
    prefixes[gid] = ltcsp_point_prefix64_metal(&cand);
}

// =============================================================================
// Kernel: ltcsp_pipeline_kernel_lut (LUT — ~2x faster on Apple Silicon)
// =============================================================================
kernel void ltcsp_pipeline_kernel_lut(
    device const AffinePoint*       tweak_points  [[buffer(0)]],
    constant const LTCSPScanKeyGlv* scan_key      [[buffer(1)]],
    device const AffinePoint*       spend_point   [[buffer(2)]],
    device const AffinePoint*       gen_lut       [[buffer(3)]],
    device ulong*                   prefixes      [[buffer(4)]],
    constant uint&                  count         [[buffer(5)]],
    uint                            gid           [[thread_position_in_grid]])
{
    if (gid >= count) return;

    AffinePoint tweak = tweak_points[gid];
    AffinePoint spend = spend_point[0];

    JacobianPoint shared;
    ltcsp_scalar_mul_glv_predecomp_metal(&shared, &tweak, scan_key);
    if (point_is_infinity_metal(&shared)) { prefixes[gid] = 0; return; }

    uchar ser[37];
    ltcsp_shared_secret_input_metal(&shared, ser);

    uchar hash[32];
    ltcsp_tagged_sha256_metal(ser, 37, hash);

    Scalar hs;
    scalar_from_bytes_metal(hash, &hs);

    JacobianPoint out;
    scalar_mul_generator_lut_metal(&out, &hs, gen_lut);

    JacobianPoint cand;
    point_add_mixed_metal(&cand, &out, &spend);
    prefixes[gid] = ltcsp_point_prefix64_metal(&cand);
}
