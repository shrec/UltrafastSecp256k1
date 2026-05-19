// =============================================================================
// secp256k1_bch_scan.cl — BCH RPA GPU Scanner (OpenCL)
// =============================================================================
// BCH Reusable Payment Address scan pipeline per work-item:
//   1. S = scan_privkey × tweak_point     (ECDH, GLV with precomputed scan key)
//   2. c = SHA256(SHA256(ser(S)) || outpoint[36])   (double-SHA256 + outpoint)
//   3. t_k = SHA256(spend_pubkey[33] || c[32] || ser32(k))
//   4. P_k = spend_point + t_k × G       (LUT generator mul)
//   5. prefixes[gid] = P_k.x[0..7]       (64-bit prefix for host filtering)
//
// Hash differs from BIP-352/LTC-SP: double-SHA256 + outpoint, not tagged hash.
//
// Throughput: not yet measured — benchmark required.
// =============================================================================

#ifndef SECP256K1_BCH_SCAN_CL
#define SECP256K1_BCH_SCAN_CL

#include "secp256k1_extended.cl"
#include "secp256k1_hash160.cl"

// BCH RPA scan-key GLV plan (same layout as LTCSPScanKeyGlv / BIP352ScanKeyGlv)
typedef struct {
    char  wnaf1[130];
    char  wnaf2[130];
    uchar k1_neg;
    uchar flip_phi;
    uchar pad0;
    uchar pad1;
} BCHScanKeyGlv;

__constant BCHScanKeyGlv  BCH_SCANKEY_WNAF;
__constant AffinePoint    BCH_SPEND_AFFINE;

// Double-SHA256: SHA256(SHA256(data))
inline void bch_double_sha256(const uchar* data, uint len, uchar out[32]) {
    uchar inner[32];
    sha256_oneshot_impl(data, len, inner);
    sha256_oneshot_impl(inner, 32, out);
}

// Shared secret hash: c = SHA256(SHA256(ser(S)) || outpoint[36])
inline void bch_shared_secret_hash(
    const uchar ser_S[33],          // compressed S
    const uchar outpoint[36],       // txid[32] + vout[4]
    uchar c_out[32])
{
    uchar inner[32];
    sha256_oneshot_impl(ser_S, 33, inner);   // SHA256(S_comp)

    // SHA256(inner[32] || outpoint[36]) = SHA256(68-byte input)
    uchar buf[68];
    for (int i = 0; i < 32; i++) buf[i] = inner[i];
    for (int i = 0; i < 36; i++) buf[32 + i] = outpoint[i];
    sha256_oneshot_impl(buf, 68, c_out);
}

// Payment key hash: t_k = SHA256(spend_pubkey[33] || c[32] || ser32(k)[4])
inline void bch_payment_key_hash(
    const uchar spend_pubkey[33],
    const uchar c[32],
    uint k,
    uchar t_out[32])
{
    uchar buf[69];
    for (int i = 0; i < 33; i++) buf[i] = spend_pubkey[i];
    for (int i = 0; i < 32; i++) buf[33 + i] = c[i];
    buf[65] = (uchar)(k >> 24);
    buf[66] = (uchar)(k >> 16);
    buf[67] = (uchar)(k >> 8);
    buf[68] = (uchar)(k);
    sha256_oneshot_impl(buf, 69, t_out);
}

// GLV scalar multiply with precomputed scan key (same as LTC-SP/BIP-352)
inline void bch_scalar_mul_glv_predecomp(
    JacobianPoint*                r,
    const AffinePoint*            p,
    constant const BCHScanKeyGlv* scan)
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
        if (d1) {
            int idx = (((d1 > 0) ? d1 : -d1) - 1) >> 1;
            AffinePoint pt = table[idx];
            if (d1 < 0) field_negate_impl(&pt.y, &pt.y);
            if (point_is_infinity(r)) point_from_affine(r, &pt);
            else                      point_add_mixed_unchecked(r, r, &pt);
        }
        int d2 = (int)scan->wnaf2[i];
        if (d2) {
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

// Extract 64-bit x-coordinate prefix
inline ulong bch_point_prefix64(const JacobianPoint* p) {
    FieldElement z_inv, z_inv2, x_aff;
    field_inv_impl(&z_inv, &p->z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&x_aff, &p->x, &z_inv2);
    uchar xb[32];
    field_to_bytes_impl(&x_aff, xb);
    ulong prefix = 0;
    for (int i = 0; i < 8; i++) prefix = (prefix << 8) | (ulong)xb[i];
    return prefix;
}

// =============================================================================
// Kernel: bch_scan_kernel_glv — GLV, no LUT
// =============================================================================
// Inputs per work-item:
//   tweak_points[gid] = sender's input_pubkey (AffinePoint)
//   outpoints[36*gid] = txid[32] || vout[4]
//   spend_pubkey33[33] = compressed spend pubkey (constant)
__kernel void bch_scan_kernel_glv(
    __global const AffinePoint*      tweak_points,
    __constant const BCHScanKeyGlv*  scan_key,
    __global const uchar*            outpoints,      // N × 36 bytes
    __global const uchar*            spend_pubkey33, // 33 bytes (same for all)
    __global ulong*                  prefixes,
    const uint                       count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    AffinePoint tweak = tweak_points[gid];

    // 1. ECDH: S = scan_privkey × tweak_point
    JacobianPoint shared;
    bch_scalar_mul_glv_predecomp(&shared, &tweak, scan_key);
    if (point_is_infinity(&shared)) { prefixes[gid] = 0; return; }

    // 2. Serialize S to 33-byte compressed
    uchar S_comp[33];
    {
        FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
        field_inv_impl(&z_inv, &shared.z);
        field_sqr_impl(&z_inv2, &z_inv);
        field_mul_impl(&z_inv3, &z_inv2, &z_inv);
        field_mul_impl(&x_aff, &shared.x, &z_inv2);
        field_mul_impl(&y_aff, &shared.y, &z_inv3);
        uchar xb[32], yb[32];
        field_to_bytes_impl(&x_aff, xb);
        field_to_bytes_impl(&y_aff, yb);
        S_comp[0] = (yb[31] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) S_comp[1+i] = xb[i];
    }

    // 3. c = SHA256(SHA256(S_comp) || outpoint[36])
    const __global uchar* outpoint = outpoints + (ulong)gid * 36;
    uchar outpoint_local[36];
    for (int i = 0; i < 36; i++) outpoint_local[i] = outpoint[i];

    uchar c[32];
    bch_shared_secret_hash(S_comp, outpoint_local, c);

    // 4. t_0 = SHA256(spend_pubkey[33] || c[32] || 0x00000000[4])
    uchar spend_local[33];
    for (int i = 0; i < 33; i++) spend_local[i] = spend_pubkey33[i];

    uchar t_k[32];
    bch_payment_key_hash(spend_local, c, 0, t_k);

    // 5. P_0 = spend_point + t_0*G  (wNAF generator mul)
    Scalar hs;
    scalar_from_bytes_impl(t_k, &hs);

    JacobianPoint out;
    scalar_mul_generator_windowed_impl(&out, &hs);

    JacobianPoint cand;
    point_add_mixed_impl(&cand, &out, &BCH_SPEND_AFFINE);
    prefixes[gid] = bch_point_prefix64(&cand);
}

// =============================================================================
// Kernel: bch_scan_kernel_lut — LUT generator table, ~2x faster k*G
// =============================================================================
__kernel void bch_scan_kernel_lut(
    __global const AffinePoint*      tweak_points,
    __constant const BCHScanKeyGlv*  scan_key,
    __global const uchar*            outpoints,
    __global const uchar*            spend_pubkey33,
    __global const AffinePoint*      gen_lut,
    __global ulong*                  prefixes,
    const uint                       count)
{
    uint gid = get_global_id(0);
    if (gid >= count) return;

    AffinePoint tweak = tweak_points[gid];

    JacobianPoint shared;
    bch_scalar_mul_glv_predecomp(&shared, &tweak, scan_key);
    if (point_is_infinity(&shared)) { prefixes[gid] = 0; return; }

    uchar S_comp[33];
    {
        FieldElement z_inv, z_inv2, z_inv3, x_aff, y_aff;
        field_inv_impl(&z_inv, &shared.z);
        field_sqr_impl(&z_inv2, &z_inv);
        field_mul_impl(&z_inv3, &z_inv2, &z_inv);
        field_mul_impl(&x_aff, &shared.x, &z_inv2);
        field_mul_impl(&y_aff, &shared.y, &z_inv3);
        uchar xb[32], yb[32];
        field_to_bytes_impl(&x_aff, xb);
        field_to_bytes_impl(&y_aff, yb);
        S_comp[0] = (yb[31] & 1) ? 0x03 : 0x02;
        for (int i = 0; i < 32; i++) S_comp[1+i] = xb[i];
    }

    const __global uchar* outpoint = outpoints + (ulong)gid * 36;
    uchar outpoint_local[36];
    for (int i = 0; i < 36; i++) outpoint_local[i] = outpoint[i];

    uchar c[32];
    bch_shared_secret_hash(S_comp, outpoint_local, c);

    uchar spend_local[33];
    for (int i = 0; i < 33; i++) spend_local[i] = spend_pubkey33[i];

    uchar t_k[32];
    bch_payment_key_hash(spend_local, c, 0, t_k);

    Scalar hs;
    scalar_from_bytes_impl(t_k, &hs);

    JacobianPoint out;
    scalar_mul_generator_lut_impl(&out, &hs, gen_lut);  // LUT path

    JacobianPoint cand;
    point_add_mixed_impl(&cand, &out, &BCH_SPEND_AFFINE);
    prefixes[gid] = bch_point_prefix64(&cand);
}

#endif // SECP256K1_BCH_SCAN_CL
