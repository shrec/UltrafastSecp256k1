// =============================================================================
// secp256k1_ltcsp_scan.cl — Litecoin Silent Payments GPU Scanner (OpenCL)
// =============================================================================
// Port of secp256k1_bip352.cl with LTCSP/ tagged hash domain.
// Identical algorithm; only the SHA256 midstate constant differs.
//
// Expected throughput: ~5.4 M tx/s (RTX 5060 Ti GLV kernel)
//                      ~11 M tx/s (LUT kernel, pre-built generator table)
// Same as BIP-352 OpenCL scanner — algorithm is identical.
//
// Depends on: secp256k1_extended.cl (included by build system)
// =============================================================================

#ifndef SECP256K1_LTCSP_SCAN_CL
#define SECP256K1_LTCSP_SCAN_CL

// secp256k1_extended.cl loaded by host bench before this file
// (same pattern as secp256k1_bip352.cl — host resolves the dependency chain)

// Precomputed scan-key GLV plan (same layout as BIP352ScanKeyGlv).
// Uploaded from CPU host; eliminates GPU scalar_to_wnaf and 1040B stack pressure.
typedef struct {
    char  wnaf1[130];
    char  wnaf2[130];
    uchar k1_neg;
    uchar flip_phi;
    uchar pad0;
    uchar pad1;
} LTCSPScanKeyGlv;

// SHA256("LTCSP/SharedSecret") — used as tagged hash midstate.
// Precomputed: python3 -c "import hashlib,struct; h=hashlib.sha256(b'LTCSP/SharedSecret').digest();
//              print([hex(w) for w in struct.unpack('>8I',h)])"
__constant uint LTCSP_SHAREDSECRET_MIDSTATE[8] = {
    0x723B685DU,
    0x30E5495FU,
    0x0A46E767U,
    0x1BD157DAU,
    0x2A9D6545U,
    0x4B2CC0B6U,
    0xBB6A9099U,
    0xECB73D0BU
};

// H_LTCSP/SharedSecret(data) = SHA256(tag || tag || data)
// Midstate already includes the two copies of SHA256(tag) per BIP-340 §D.
inline void ltcsp_tagged_sha256_impl(const uchar* data, uint data_len, uchar out[32]) {
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = LTCSP_SHAREDSECRET_MIDSTATE[i];
    ctx.buf_len  = 0;
    ctx.total_len = 64;
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

// Serialize Jacobian point to 33-byte compressed form + 4 zero bytes (37 total).
// Matches bip352_shared_secret_input_impl — same layout for the hash input.
inline void ltcsp_shared_secret_input_impl(const JacobianPoint* p, uchar ser[37]) {
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
    ser[33] = ser[34] = ser[35] = ser[36] = 0;
}

// Extract first 8 bytes of affine X coordinate as uint64 prefix (for fast filtering).
inline ulong ltcsp_point_prefix64_impl(const JacobianPoint* p) {
    FieldElement z_inv, z_inv2, x_aff;
    field_inv_impl(&z_inv, &p->z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&x_aff, &p->x, &z_inv2);

    uchar x_bytes[32];
    field_to_bytes_impl(&x_aff, x_bytes);

    ulong prefix = 0;
    for (int i = 0; i < 8; i++)
        prefix = (prefix << 8) | (ulong)x_bytes[i];
    return prefix;
}

// Optimised GLV scalar-multiply with pre-decomposed scan key (same as BIP-352).
// Precomputed wNAF digits in __constant memory; one field_inv shared across table.
inline void ltcsp_scalar_mul_glv_predecomp_impl(
    JacobianPoint*                  r,
    const AffinePoint*              p,
    __constant const LTCSPScanKeyGlv* scan)
{
    AffinePoint base = *p;
    if (scan->k1_neg) field_negate_impl(&base.y, &base.y);

    AffinePoint table[8];
    FieldElement globalz;
    build_wnaf_table_zr_impl(&base, table, &globalz);

    AffinePoint endo_table[8];
    derive_endo_table_impl(table, endo_table, scan->flip_phi);

    point_set_infinity(r);
    for (int i = 129; i >= 0; --i) {
        if (!point_is_infinity(r)) point_double_unchecked(r, r);

        int d1 = (int)scan->wnaf1[i];
        if (d1 != 0) {
            int idx = (((d1 > 0) ? d1 : -d1) - 1) >> 1;
            AffinePoint pt = table[idx];
            if (d1 < 0) field_negate_impl(&pt.y, &pt.y);
            if (point_is_infinity(r)) point_from_affine(r, &pt);
            else                      point_add_mixed_unchecked(r, r, &pt);
        }

        int d2 = (int)scan->wnaf2[i];
        if (d2 != 0) {
            int idx = (((d2 > 0) ? d2 : -d2) - 1) >> 1;
            AffinePoint pt = endo_table[idx];
            if (d2 < 0) field_negate_impl(&pt.y, &pt.y);
            if (point_is_infinity(r)) point_from_affine(r, &pt);
            else                      point_add_mixed_unchecked(r, r, &pt);
        }
    }

    if (!point_is_infinity(r)) {
        FieldElement cz;
        field_mul_impl(&cz, &r->z, &globalz);
        r->z = cz;
    }
}

// =============================================================================
// Kernel: ltcsp_pipeline_kernel (GLV — no LUT)
// =============================================================================
// Each work-item processes one transaction:
//   tweak_points[gid] = sender's input public key (A_i, aggregated by host)
//   scan_key           = scan_privkey GLV plan (precomputed on CPU, __constant)
//   spend_point[0]     = receiver's spend pubkey (B_spend, __constant via global)
//   prefixes[gid]      = first 8 bytes of candidate output X (for host filtering)
__kernel void ltcsp_pipeline_kernel(
    __global const AffinePoint*       tweak_points,
    __constant const LTCSPScanKeyGlv* scan_key,
    __global const AffinePoint*       spend_point,
    __global ulong*                   prefixes,
    const uint                        count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    AffinePoint tweak = tweak_points[gid];
    AffinePoint spend = spend_point[0];

    // S = scan_privkey × tweak  (ECDH shared secret)
    JacobianPoint shared;
    ltcsp_scalar_mul_glv_predecomp_impl(&shared, &tweak, scan_key);
    if (point_is_infinity(&shared)) { prefixes[gid] = 0; return; }

    // Hash: t_k = H_LTCSP/SharedSecret(ser(S) || ser32(k=0))
    uchar ser[37];
    ltcsp_shared_secret_input_impl(&shared, ser);

    uchar hash[32];
    ltcsp_tagged_sha256_impl(ser, 37, hash);

    // P_out = B_spend + t_k * G
    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);

    JacobianPoint out;
    scalar_mul_generator_windowed_impl(&out, &hs);

    JacobianPoint cand;
    point_add_mixed_impl(&cand, &out, &spend);
    prefixes[gid] = ltcsp_point_prefix64_impl(&cand);
}

// =============================================================================
// Kernel: ltcsp_pipeline_kernel_lut (LUT — pre-built generator table, ~2x faster)
// =============================================================================
__kernel void ltcsp_pipeline_kernel_lut(
    __global const AffinePoint*       tweak_points,
    __constant const LTCSPScanKeyGlv* scan_key,
    __global const AffinePoint*       spend_point,
    __global const AffinePoint*       gen_lut,
    __global ulong*                   prefixes,
    const uint                        count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    AffinePoint tweak = tweak_points[gid];
    AffinePoint spend = spend_point[0];

    JacobianPoint shared;
    ltcsp_scalar_mul_glv_predecomp_impl(&shared, &tweak, scan_key);
    if (point_is_infinity(&shared)) { prefixes[gid] = 0; return; }

    uchar ser[37];
    ltcsp_shared_secret_input_impl(&shared, ser);

    uchar hash[32];
    ltcsp_tagged_sha256_impl(ser, 37, hash);

    Scalar hs;
    scalar_from_bytes_impl(hash, &hs);

    JacobianPoint out;
    scalar_mul_generator_lut_impl(&out, &hs, gen_lut);

    JacobianPoint cand;
    point_add_mixed_impl(&cand, &out, &spend);
    prefixes[gid] = ltcsp_point_prefix64_impl(&cand);
}

#endif // SECP256K1_LTCSP_SCAN_CL
