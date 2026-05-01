// =============================================================================
// secp256k1_ct_point.h -- Constant-time point operations for Metal
// =============================================================================
// 8x32-bit limbs. Brier-Joye complete addition. CT scalar mul with GLV.
// Requires: secp256k1_point.h, secp256k1_ct_ops.h, secp256k1_ct_field.h,
//           secp256k1_ct_scalar.h
// =============================================================================

#ifndef SECP256K1_CT_POINT_H
#define SECP256K1_CT_POINT_H

// ---------------------------------------------------------------------------
// CT point types: infinity is a uint mask (0 or 0xFFFFFFFF)
// ---------------------------------------------------------------------------
struct CTJacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    uint infinity;  // 0 = not infinity, 0xFFFFFFFF = infinity
};

struct CTAffinePoint {
    FieldElement x;
    FieldElement y;
    uint infinity;  // 0 = not infinity, 0xFFFFFFFF = infinity
};

// ---------------------------------------------------------------------------
// Conversion utilities
// ---------------------------------------------------------------------------
inline CTJacobianPoint ct_point_infinity() {
    CTJacobianPoint p;
    p.x = field_zero();
    p.y = field_zero();
    p.z = field_one();
    p.infinity = 0xFFFFFFFFu;
    return p;
}

inline JacobianPoint ct_to_jacobian(thread const CTJacobianPoint &p) {
    JacobianPoint r;
    r.x = p.x; r.y = p.y; r.z = p.z;
    r.infinity = (p.infinity != 0) ? 1 : 0;
    return r;
}

inline CTJacobianPoint ct_from_jacobian(thread const JacobianPoint &p) {
    CTJacobianPoint r;
    r.x = p.x; r.y = p.y; r.z = p.z;
    r.infinity = p.infinity ? 0xFFFFFFFFu : 0;
    return r;
}

// ---------------------------------------------------------------------------
// CT conditional ops on points
// ---------------------------------------------------------------------------
inline void ct_point_cmov(thread CTJacobianPoint &r, thread const CTJacobianPoint &a, uint mask) {
    ct_field_cmov(r.x, a.x, mask);
    ct_field_cmov(r.y, a.y, mask);
    ct_field_cmov(r.z, a.z, mask);
    r.infinity = (r.infinity & ~mask) | (a.infinity & mask);
}

inline void ct_aff_cmov(thread CTAffinePoint &r, thread const CTAffinePoint &a, uint mask) {
    ct_field_cmov(r.x, a.x, mask);
    ct_field_cmov(r.y, a.y, mask);
    r.infinity = (r.infinity & ~mask) | (a.infinity & mask);
}

inline void ct_point_cneg_y(thread CTAffinePoint &p, uint mask) {
    FieldElement neg = ct_field_neg(p.y);
    p.y = ct_field_select(neg, p.y, mask);
}

// CT lookup from affine table
inline CTAffinePoint ct_affine_table_lookup(thread const CTAffinePoint* table,
                                            int table_size, int index) {
    CTAffinePoint out = table[0];
    for (int i = 1; i < table_size; ++i) {
        uint mask = ct_eq_mask((uint)i, (uint)index);
        ct_aff_cmov(out, table[i], mask);
    }
    return out;
}

// ---------------------------------------------------------------------------
// CT point doubling (4M+4S)
// ---------------------------------------------------------------------------
inline CTJacobianPoint ct_point_dbl(thread const CTJacobianPoint &p) {
    FieldElement a = field_sqr(p.x);            // x^2
    FieldElement b = field_sqr(p.y);            // y^2
    FieldElement c = field_sqr(b);              // y^4

    FieldElement xb = field_add(p.x, b);
    FieldElement d = field_sqr(xb);
    d = field_sub(d, a);
    d = field_sub(d, c);
    d = field_add(d, d);                        // 2*((x+b)^2-a-c)

    FieldElement e = field_add(a, field_add(a, a)); // 3*a
    FieldElement f = field_sqr(e);              // e^2

    CTJacobianPoint r;
    FieldElement d2 = field_add(d, d);
    r.x = field_sub(f, d2);

    r.z = field_mul(p.y, p.z);
    r.z = field_add(r.z, r.z);                 // 2*y*z

    FieldElement dx = field_sub(d, r.x);
    r.y = field_mul(e, dx);
    FieldElement c8 = field_add(c, c);
    c8 = field_add(c8, c8);
    c8 = field_add(c8, c8);
    r.y = field_sub(r.y, c8);

    r.infinity = p.infinity;
    return r;
}

// ---------------------------------------------------------------------------
// CT point add mixed (Brier-Joye complete: Jacobian + Affine -> Jacobian)
// ---------------------------------------------------------------------------
inline CTJacobianPoint ct_point_add_mixed(thread const CTJacobianPoint &p,
                                          thread const CTAffinePoint &q) {
    FieldElement z2 = field_sqr(p.z);
    FieldElement u2 = field_mul(q.x, z2);
    FieldElement z3 = field_mul(p.z, z2);
    FieldElement s2 = field_mul(q.y, z3);

    FieldElement h = field_sub(u2, p.x);
    FieldElement hh = field_sqr(h);
    FieldElement i_val = field_add(hh, hh);
    i_val = field_add(i_val, i_val);           // 4*h^2
    FieldElement j = field_mul(h, i_val);
    FieldElement rr = field_sub(s2, p.y);
    rr = field_add(rr, rr);                    // 2*(s2-y1)
    FieldElement v = field_mul(p.x, i_val);

    // x3 = r^2 - j - 2*v
    CTJacobianPoint result;
    FieldElement rr2 = field_sqr(rr);
    result.x = field_sub(rr2, j);
    FieldElement v2 = field_add(v, v);
    result.x = field_sub(result.x, v2);

    // y3 = r*(v-x3) - 2*y1*j
    FieldElement vx = field_sub(v, result.x);
    result.y = field_mul(rr, vx);
    FieldElement y1j = field_mul(p.y, j);
    y1j = field_add(y1j, y1j);
    result.y = field_sub(result.y, y1j);

    // z3 = 2*z1*h
    result.z = field_mul(p.z, h);
    result.z = field_add(result.z, result.z);

    // Degenerate case: h==0 && rr==0 -> P==Q, double
    uint h_zero = ct_field_is_zero_mask(h);
    uint rr_zero = ct_field_is_zero_mask(rr);
    uint same_point = h_zero & rr_zero;

    CTJacobianPoint p_dbl = ct_point_dbl(p);
    ct_point_cmov(result, p_dbl, same_point & ~p.infinity & ~q.infinity);

    // P is infinity -> result = Q
    CTJacobianPoint q_jac;
    q_jac.x = q.x; q_jac.y = q.y;
    q_jac.z = field_one();
    q_jac.infinity = q.infinity;
    ct_point_cmov(result, q_jac, p.infinity);

    // Q is infinity -> result = P
    ct_point_cmov(result, p, q.infinity);

    result.infinity = p.infinity & q.infinity;
    return result;
}

// ---------------------------------------------------------------------------
// CT batch field inverse (Montgomery trick)
// ---------------------------------------------------------------------------
inline void ct_batch_field_inv_metal(thread FieldElement* vals,
                                     thread FieldElement* invs, int n) {
    if (n <= 0) return;
    FieldElement acc[16];
    acc[0] = vals[0];
    for (int i = 1; i < n; ++i)
        acc[i] = field_mul(acc[i - 1], vals[i]);
    FieldElement inv_val = ct_field_inv(acc[n - 1]);
    for (int i = n - 1; i > 0; --i) {
        invs[i] = field_mul(inv_val, acc[i - 1]);
        inv_val = field_mul(inv_val, vals[i]);
    }
    invs[0] = inv_val;
}

// ---------------------------------------------------------------------------
// Precomputed G tables in 8x32 LE format
// Each entry: 8 limbs for X (LE) + 8 limbs for Y (LE)
// ---------------------------------------------------------------------------
constant uint CT_G_TABLE_A_METAL[15][16] = {
    // 1G
    { 0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
      0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu,
      0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
      0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u },
    // 2G
    { 0x5C709EE5u, 0xABAC09B9u, 0x8CEF3CA7u, 0x5C778E4Bu,
      0x95C07CD8u, 0x3045406Eu, 0x41ED7D6Du, 0xC6047F94u,
      0x50CFE52Au, 0x236431A9u, 0x3266D0E1u, 0xF7F63265u,
      0x466CEAEEu, 0xA3C58419u, 0xA63DC339u, 0x1AE168FEu },
    // 3G
    { 0xBCE036F9u, 0x8601F113u, 0x836F99B0u, 0xB531C845u,
      0xF89D5229u, 0x49344F85u, 0x9258C310u, 0xF9308A01u,
      0x84B8E672u, 0x6CB9FD75u, 0x34C2231Bu, 0x6500A999u,
      0x2A37F356u, 0x0FE337E6u, 0x632DE814u, 0x388F7B0Fu },
    // 4G
    { 0xE8C4CD13u, 0x74FA94ABu, 0x0EE07584u, 0xCC6C1390u,
      0x930B1404u, 0x581E4904u, 0xC10D80F3u, 0xE493DBF1u,
      0x47739922u, 0xCFE97BDCu, 0xBFBDFE40u, 0xD967AE33u,
      0x8EA51448u, 0x5642E209u, 0xA0D455B7u, 0x51ED993Eu },
    // 5G
    { 0xB240EFE4u, 0xCBA8D569u, 0xDC619AB7u, 0xE88B84BDu,
      0x0A5C5128u, 0x55B4A725u, 0x1A072093u, 0x2F8BDE4Du,
      0xA6AC62D6u, 0xDCA87D3Au, 0xAB0D6840u, 0xF788271Bu,
      0xA6C9C426u, 0xD4DBA9DDu, 0x36E5E3D6u, 0xD8AC2226u },
    // 6G
    { 0x60297556u, 0x2F057A14u, 0x8568A18Bu, 0x82F6472Fu,
      0x355235D3u, 0x20453A14u, 0x755EEEA4u, 0xFFF97BD5u,
      0xB075F297u, 0x3C870C36u, 0x518FE4A0u, 0xDE80F0F6u,
      0x7F45C560u, 0xF3BE9601u, 0xACFBB620u, 0xAE12777Au },
    // 7G
    { 0xCAC4F9BCu, 0xE92BDDEDu, 0x0330E39Cu, 0x3D419B7Eu,
      0xF2EA7A0Eu, 0xA398F365u, 0x6E5DB4EAu, 0x5CBDF064u,
      0x087264DAu, 0xA5082628u, 0x13FDE7B5u, 0xA813D0B8u,
      0x861A54DBu, 0xA3178D6Du, 0xBA255960u, 0x6AEBCA40u },
    // 8G
    { 0xE10A2A01u, 0x67784EF3u, 0xE5AF888Au, 0x0A1BDD05u,
      0xB70F3C2Fu, 0xAFF3843Fu, 0x5CCA351Du, 0x2F01E5E1u,
      0x6CBDE904u, 0xB5DA2CB7u, 0xBA5B7617u, 0xC2E213D6u,
      0x132D13B4u, 0x293D082Au, 0x41539949u, 0x5C4DA8A7u },
    // 9G
    { 0xFC27CCBEu, 0xC35F110Du, 0x4C57E714u, 0xE0979697u,
      0x9F559ABDu, 0x09AD178Au, 0xF0C7F653u, 0xACD484E2u,
      0xC64F9C37u, 0x05CC262Au, 0x375F8E0Fu, 0xADD888A4u,
      0x763B61E9u, 0x64380971u, 0xB0A7D9FDu, 0xCC338921u },
    // 10G
    { 0x47E247C7u, 0x52A68E2Au, 0x1943C2B7u, 0x3442D49Bu,
      0x1AE6AE5Du, 0x35477C7Bu, 0x47F3C862u, 0xA0434D9Eu,
      0x037368D7u, 0x3CBEE53Bu, 0xD877A159u, 0x6F794C2Eu,
      0x93A24C69u, 0xA3B6C7E6u, 0x5419BC27u, 0x893ABA42u },
    // 11G
    { 0x5DA008CBu, 0xBBEC1789u, 0xE5C17891u, 0x5649980Bu,
      0x70C65AACu, 0x5EF4246Bu, 0x58A9411Eu, 0x774AE7F8u,
      0xC953C61Bu, 0x301D74C9u, 0xDFF9D6A8u, 0x372DB1E2u,
      0xD7B7B365u, 0x0243DD56u, 0xEB6B5E19u, 0xD984A032u },
    // 12G
    { 0x70AFE85Au, 0xC5B0F470u, 0x9620095Bu, 0x687CF441u,
      0x4D734633u, 0x15C38F00u, 0x48E7561Bu, 0xD01115D5u,
      0xF4062327u, 0x6B051B13u, 0xD9A86D52u, 0x79238C5Du,
      0xE17BD815u, 0xA8B64537u, 0xC815E0D7u, 0xA9F34FFDu },
    // 13G
    { 0x19405AA8u, 0xDEEDDF8Fu, 0x610E58CDu, 0xB075FBC6u,
      0xC3748651u, 0xC7D1D205u, 0xD975288Bu, 0xF28773C2u,
      0xDB03ED81u, 0x29B5CB52u, 0x521FA91Fu, 0x3A1A06DAu,
      0x65CDAF47u, 0x758212EBu, 0x8D880A89u, 0x0AB0902Eu },
    // 14G
    { 0x60E823E4u, 0xE49B241Au, 0x678949E6u, 0x26AA7B63u,
      0x07D38E32u, 0xFD64E67Fu, 0x895E719Cu, 0x499FDF9Eu,
      0x03A13F5Bu, 0xC65F40D4u, 0xA3F95BCu, 0x464279C2u,
      0xA7B3D464u, 0x90F044E4u, 0xB54E8551u, 0xCAC2F6C4u },
    // 15G
    { 0xE27E080Eu, 0x44ADBCF8u, 0x3C85F79Eu, 0x31E5946Fu,
      0x095FF411u, 0x5A465AE3u, 0x7D43EA96u, 0xD7924D4Fu,
      0xF6A26B58u, 0xC504DC9Fu, 0xD896D3A5u, 0xEA40AF2Bu,
      0x28CC6DEFu, 0x83842EC2u, 0xA86C72A6u, 0x581E2872u },
};

constant uint CT_G_TABLE_B_METAL[15][16] = {
    // phi(1G)
    { 0x00B88FCBu, 0xA7BBA044u, 0x7F15E98Du, 0x87284406u,
      0x96902325u, 0xAB0102B6u, 0x9DA01887u, 0xBCACE2E9u,
      0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
      0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u },
    // phi(2G)
    { 0xD89250E1u, 0x3E995B6Eu, 0xE43837EFu, 0xD2FAD8CCu,
      0x59F87B33u, 0x4135EE7Du, 0xB34CE6DFu, 0xC360A6D0u,
      0x50CFE52Au, 0x236431A9u, 0x3266D0E1u, 0xF7F63265u,
      0x466CEAEEu, 0xA3C58419u, 0xA63DC339u, 0x1AE168FEu },
    // phi(3G)
    { 0x77206B2Fu, 0xF7F0728Cu, 0xC6DC8E1Cu, 0x8AF1E022u,
      0x2A28FA2Fu, 0x8DCD8DCFu, 0x731F9B4Bu, 0xDF6EDF03u,
      0x84B8E672u, 0x6CB9FD75u, 0x34C2231Bu, 0x6500A999u,
      0x2A37F356u, 0x0FE337E6u, 0x632DE814u, 0x388F7B0Fu },
    // phi(4G)
    { 0x3B306100u, 0x5BDE5B33u, 0xAB487127u, 0x714C30B5u,
      0xB90E324Bu, 0x5C45FAF8u, 0x0D382907u, 0x1B77921Fu,
      0x47739922u, 0xCFE97BDCu, 0xBFBDFE40u, 0xD967AE33u,
      0x8EA51448u, 0x5642E209u, 0xA0D455B7u, 0x51ED993Eu },
    // phi(5G)
    { 0x95A83668u, 0x138C6946u, 0xE0D097CCu, 0xA045693Eu,
      0xCCB94671u, 0xF79F54FBu, 0xACDA49DFu, 0x337B52E3u,
      0xA6AC62D6u, 0xDCA87D3Au, 0xAB0D6840u, 0xF788271Bu,
      0xA6C9C426u, 0xD4DBA9DDu, 0x36E5E3D6u, 0xD8AC2226u },
    // phi(6G)
    { 0x78F38045u, 0x47AAF280u, 0x56A15A68u, 0x86649D3Eu,
      0xE3E8BED7u, 0x5E3AA731u, 0xAA535FC6u, 0xE63BCDD9u,
      0xB075F297u, 0x3C870C36u, 0x518FE4A0u, 0xDE80F0F6u,
      0x7F45C560u, 0xF3BE9601u, 0xACFBB620u, 0xAE12777Au },
    // phi(7G)
    { 0x4E53BC94u, 0x3BC4686Eu, 0x0FAF7AAAu, 0x0D3B20E2u,
      0xC095C06Eu, 0xA4FEC4D1u, 0x4BEA0B77u, 0x13F26E75u,
      0x087264DAu, 0xA5082628u, 0x13FDE7B5u, 0xA813D0B8u,
      0x861A54DBu, 0xA3178D6Du, 0xBA255960u, 0x6AEBCA40u },
    // phi(8G)
    { 0x2446CC73u, 0x03E94774u, 0x24257657u, 0xB4FF7715u,
      0x29E24892u, 0xAA77840Fu, 0x42D401A7u, 0x47AB6503u,
      0x6CBDE904u, 0xB5DA2CB7u, 0xBA5B7617u, 0xC2E213D6u,
      0x132D13B4u, 0x293D082Au, 0x41539949u, 0x5C4DA8A7u },
    // phi(9G)
    { 0x65953A52u, 0x20CD912Eu, 0xEF6D44E1u, 0xB565CDF5u,
      0xEC58AB20u, 0x7B6558AFu, 0x7E44E819u, 0x87B40403u,
      0xC64F9C37u, 0x05CC262Au, 0x375F8E0Fu, 0xADD888A4u,
      0x763B61E9u, 0x64380971u, 0xB0A7D9FDu, 0xCC338921u },
    // phi(10G)
    { 0x741AFE29u, 0xBDB3E957u, 0x083762E4u, 0xC1938D8Eu,
      0x46813990u, 0xA136EBB2u, 0xF7A397B1u, 0x26CE269Bu,
      0x037368D7u, 0x3CBEE53Bu, 0xD877A159u, 0x6F794C2Eu,
      0x93A24C69u, 0xA3B6C7E6u, 0x5419BC27u, 0x893ABA42u },
    // phi(11G)
    { 0xBB209CE7u, 0xC5FF4334u, 0x0B5FF620u, 0x79859BB7u,
      0xBEBF1A26u, 0x8D897C41u, 0x171DAC1Du, 0x51F4D3D1u,
      0xC953C61Bu, 0x301D74C9u, 0xDFF9D6A8u, 0x372DB1E2u,
      0xD7B7B365u, 0x0243DD56u, 0xEB6B5E19u, 0xD984A032u },
    // phi(12G)
    { 0x042295E5u, 0x4A3EB52Cu, 0xC9535355u, 0xF9482837u,
      0x2EAC82ADu, 0xAC154842u, 0x953AAC41u, 0x88591BFDu,
      0xF4062327u, 0x6B051B13u, 0xD9A86D52u, 0x79238C5Du,
      0xE17BD815u, 0xA8B64537u, 0xC815E0D7u, 0xA9F34FFDu },
    // phi(13G)
    { 0x475FB678u, 0x60AAEE6Au, 0x4A3D0562u, 0x32907ED7u,
      0x78FC783Bu, 0x07046C45u, 0x4BB890A2u, 0xF14D5837u,
      0xDB03ED81u, 0x29B5CB52u, 0x521FA91Fu, 0x3A1A06DAu,
      0x65CDAF47u, 0x758212EBu, 0x8D880A89u, 0x0AB0902Eu },
    // phi(14G)
    { 0x20A0B458u, 0x0E6AB7EEu, 0x27C529F6u, 0x580656A6u,
      0x87C37384u, 0x1548F0DCu, 0x7810048Au, 0x7B125217u,
      0x03A13F5Bu, 0xC65F40D4u, 0xA3F95BCu, 0x464279C2u,
      0xA7B3D464u, 0x90F044E4u, 0xB54E8551u, 0xCAC2F6C4u },
    // phi(15G)
    { 0x71B1B3B4u, 0x3AC0A40Cu, 0xC1C0A639u, 0x05CC3BC9u,
      0x512B6948u, 0x0E1B4825u, 0xF5F9454Au, 0x805F1105u,
      0xF6A26B58u, 0xC504DC9Fu, 0xD896D3A5u, 0xEA40AF2Bu,
      0x28CC6DEFu, 0x83842EC2u, 0xA86C72A6u, 0x581E2872u },
};

// ---------------------------------------------------------------------------
// CT scalar multiplication: k*P (GLV + 4-bit windowed)
// ---------------------------------------------------------------------------
inline CTJacobianPoint ct_scalar_mul_point(thread const CTJacobianPoint &p,
                                           thread const Scalar256 &k) {
    CTGLVDecompositionMetal glv = ct_glv_decompose(k);

    constexpr int TABLE_SIZE = 16;
    CTAffinePoint table_a[TABLE_SIZE];
    CTAffinePoint table_b[TABLE_SIZE];

    // Identity at 0
    table_a[0].x = field_zero();
    table_a[0].y = field_zero();
    table_a[0].infinity = 0xFFFFFFFFu;

    // Convert P to affine for table[1]
    FieldElement z_inv0 = ct_field_inv(p.z);
    FieldElement z_inv2_0 = field_sqr(z_inv0);
    FieldElement z_inv3_0 = field_mul(z_inv0, z_inv2_0);
    table_a[1].x = field_mul(p.x, z_inv2_0);
    table_a[1].y = field_mul(p.y, z_inv3_0);
    table_a[1].infinity = p.infinity;

    // Build 2P..15P via sequential CT adds
    CTJacobianPoint jac_pts[15];
    jac_pts[0].x = p.x; jac_pts[0].y = p.y;
    jac_pts[0].z = p.z; jac_pts[0].infinity = p.infinity;
    for (int i = 1; i < 15; ++i)
        jac_pts[i] = ct_point_add_mixed(jac_pts[i - 1], table_a[1]);

    // Batch invert Z -> affine
    FieldElement z_vals[15], z_inv_vals[15];
    for (int i = 0; i < 15; ++i) z_vals[i] = jac_pts[i].z;
    ct_batch_field_inv_metal(z_vals, z_inv_vals, 15);
    for (int i = 0; i < 15; ++i) {
        FieldElement zi2 = field_sqr(z_inv_vals[i]);
        FieldElement zi3 = field_mul(z_inv_vals[i], zi2);
        table_a[i + 1].x = field_mul(jac_pts[i].x, zi2);
        table_a[i + 1].y = field_mul(jac_pts[i].y, zi3);
        table_a[i + 1].infinity = jac_pts[i].infinity;
    }

    // Endomorphism table: phi(P) = (beta*x, y)
    const uint BETA_METAL[8] = {
        0x57C0710u, 0x7AE96A2Bu, 0xEB4C3F40u, 0x6584D3F6u,
        0x0E46AB35u, 0x7F09A368u, 0x9A83F8EFu, 0x851695D4u
    };
    FieldElement beta;
    for (int i = 0; i < 8; ++i) beta.limbs[i] = BETA_METAL[i];
    table_b[0] = table_a[0];
    for (int i = 1; i < TABLE_SIZE; ++i) {
        table_b[i].x = field_mul(table_a[i].x, beta);
        table_b[i].y = table_a[i].y;
        table_b[i].infinity = table_a[i].infinity;
    }

    // Conditionally negate
    for (int i = 1; i < TABLE_SIZE; ++i) {
        ct_point_cneg_y(table_a[i], glv.k1_neg);
        ct_point_cneg_y(table_b[i], glv.k2_neg);
    }

    // Windowed loop
    CTJacobianPoint result = ct_point_infinity();
    for (int w = 32; w >= 0; --w) {
        result = ct_point_dbl(result);
        result = ct_point_dbl(result);
        result = ct_point_dbl(result);
        result = ct_point_dbl(result);

        int bit_pos = w * 4;
        int limb_idx = bit_pos >> 5;
        int bit_off = bit_pos & 31;
        int d1 = (int)((glv.k1.limbs[limb_idx] >> bit_off) & 0xFu);
        int d2 = (int)((glv.k2.limbs[limb_idx] >> bit_off) & 0xFu);

        CTAffinePoint entry1 = ct_affine_table_lookup(table_a, TABLE_SIZE, d1);
        result = ct_point_add_mixed(result, entry1);

        CTAffinePoint entry2 = ct_affine_table_lookup(table_b, TABLE_SIZE, d2);
        result = ct_point_add_mixed(result, entry2);
    }
    return result;
}

// ---------------------------------------------------------------------------
// CT generator multiplication: k*G (fixed-base, precomputed)
// ---------------------------------------------------------------------------
inline CTJacobianPoint ct_generator_mul_metal(thread const Scalar256 &k) {
    CTGLVDecompositionMetal glv = ct_glv_decompose(k);

    constexpr int TABLE_SIZE = 16;
    CTAffinePoint table_a[TABLE_SIZE];
    CTAffinePoint table_b[TABLE_SIZE];

    table_a[0].x = field_zero();
    table_a[0].y = field_zero();
    table_a[0].infinity = 0xFFFFFFFFu;
    table_b[0] = table_a[0];

    // Load from constant tables (8x32 format)
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 8; ++j) {
            table_a[i + 1].x.limbs[j] = CT_G_TABLE_A_METAL[i][j];
            table_a[i + 1].y.limbs[j] = CT_G_TABLE_A_METAL[i][j + 8];
            table_b[i + 1].x.limbs[j] = CT_G_TABLE_B_METAL[i][j];
            table_b[i + 1].y.limbs[j] = CT_G_TABLE_B_METAL[i][j + 8];
        }
        table_a[i + 1].infinity = 0;
        table_b[i + 1].infinity = 0;
    }

    for (int i = 1; i < TABLE_SIZE; ++i) {
        ct_point_cneg_y(table_a[i], glv.k1_neg);
        ct_point_cneg_y(table_b[i], glv.k2_neg);
    }

    CTJacobianPoint result = ct_point_infinity();
    for (int w = 32; w >= 0; --w) {
        result = ct_point_dbl(result);
        result = ct_point_dbl(result);
        result = ct_point_dbl(result);
        result = ct_point_dbl(result);

        int bit_pos = w * 4;
        int limb_idx = bit_pos >> 5;
        int bit_off = bit_pos & 31;
        int d1 = (int)((glv.k1.limbs[limb_idx] >> bit_off) & 0xFu);
        int d2 = (int)((glv.k2.limbs[limb_idx] >> bit_off) & 0xFu);

        CTAffinePoint entry1 = ct_affine_table_lookup(table_a, TABLE_SIZE, d1);
        result = ct_point_add_mixed(result, entry1);

        CTAffinePoint entry2 = ct_affine_table_lookup(table_b, TABLE_SIZE, d2);
        result = ct_point_add_mixed(result, entry2);
    }
    return result;
}

#endif // SECP256K1_CT_POINT_H
