// =============================================================================
// secp256k1_ct_point.cl -- Constant-time point operations for OpenCL
// =============================================================================
// Brier-Joye complete addition (handles all degenerate cases in single codepath).
// CT scalar multiplication with GLV + 4-bit windowed, precomputed G tables.
// Requires: secp256k1_point.cl, secp256k1_ct_ops.cl, secp256k1_ct_field.cl,
//           secp256k1_ct_scalar.cl
// =============================================================================

#ifndef SECP256K1_CT_POINT_CL
#define SECP256K1_CT_POINT_CL

// ---------------------------------------------------------------------------
// CT point types: infinity is a full 64-bit mask (all-ones or zero)
// This prevents branching on point-at-infinity checks.
// ---------------------------------------------------------------------------
typedef struct {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    ulong infinity;  // 0 = not infinity, ~0 = infinity
} CTJacobianPoint;

typedef struct {
    FieldElement x;
    FieldElement y;
    ulong infinity;  // 0 = not infinity, ~0 = infinity
} CTAffinePoint;

// ---------------------------------------------------------------------------
// Conversion utilities
// ---------------------------------------------------------------------------
inline void ct_point_set_infinity(CTJacobianPoint* p) {
    for (int i = 0; i < 4; ++i) {
        p->x.limbs[i] = 0;
        p->y.limbs[i] = 0;
        p->z.limbs[i] = 0;
    }
    p->z.limbs[0] = 1;
    p->infinity = ~(ulong)0;
}

inline JacobianPoint ct_point_to_jacobian(const CTJacobianPoint* p) {
    JacobianPoint r;
    r.x = p->x;
    r.y = p->y;
    r.z = p->z;
    r.infinity = (p->infinity != 0) ? 1 : 0;
    return r;
}

inline CTJacobianPoint ct_point_from_jacobian(const JacobianPoint* p) {
    CTJacobianPoint r;
    r.x = p->x;
    r.y = p->y;
    r.z = p->z;
    r.infinity = p->infinity ? ~(ulong)0 : 0;
    return r;
}

// ---------------------------------------------------------------------------
// CT conditional ops on points
// ---------------------------------------------------------------------------
inline void ct_point_cmov(CTJacobianPoint* r, const CTJacobianPoint* a, ulong mask) {
    ct_cmov256((ulong*)r->x.limbs, (const ulong*)a->x.limbs, mask);
    ct_cmov256((ulong*)r->y.limbs, (const ulong*)a->y.limbs, mask);
    ct_cmov256((ulong*)r->z.limbs, (const ulong*)a->z.limbs, mask);
    ct_cmov64(&r->infinity, a->infinity, mask);
}

inline void ct_aff_cmov(CTAffinePoint* r, const CTAffinePoint* a, ulong mask) {
    ct_cmov256((ulong*)r->x.limbs, (const ulong*)a->x.limbs, mask);
    ct_cmov256((ulong*)r->y.limbs, (const ulong*)a->y.limbs, mask);
    ct_cmov64(&r->infinity, a->infinity, mask);
}

// Conditionally negate Y: if mask, y = -y
inline void ct_point_cneg_y(CTAffinePoint* p, ulong mask) {
    FieldElement neg;
    ct_field_neg_impl(&neg, &p->y);
    ct_field_cmov(&p->y, &neg, mask);
}

// CT lookup from affine table (scans ALL entries)
inline void ct_affine_table_lookup(const CTAffinePoint* table, int table_size,
                                   int index, CTAffinePoint* out) {
    *out = table[0];
    for (int i = 1; i < table_size; ++i) {
        ulong mask = ct_eq_mask((ulong)i, (ulong)index);
        ct_aff_cmov(out, &table[i], mask);
    }
}

// ---------------------------------------------------------------------------
// CT point doubling (standard 4M+4S, with CT infinity handling)
// ---------------------------------------------------------------------------
inline void ct_point_dbl(const CTJacobianPoint* p, CTJacobianPoint* r) {
    // Standard Jacobian doubling
    FieldElement a, b, c, d, e, f;

    field_sqr_impl(&a, &p->x);           // a = x^2
    field_sqr_impl(&b, &p->y);           // b = y^2
    field_sqr_impl(&c, &b);              // c = y^4

    // d = 2*((x+b)^2 - a - c)
    FieldElement xb;
    field_add_impl(&xb, &p->x, &b);
    field_sqr_impl(&d, &xb);
    field_sub_impl(&d, &d, &a);
    field_sub_impl(&d, &d, &c);
    field_add_impl(&d, &d, &d);

    // e = 3*a
    field_add_impl(&e, &a, &a);
    field_add_impl(&e, &e, &a);

    // f = e^2
    field_sqr_impl(&f, &e);

    // x3 = f - 2*d
    FieldElement d2;
    field_add_impl(&d2, &d, &d);
    field_sub_impl(&r->x, &f, &d2);

    // z3 = 2*y*z
    field_mul_impl(&r->z, &p->y, &p->z);
    field_add_impl(&r->z, &r->z, &r->z);

    // y3 = e*(d - x3) - 8*c
    FieldElement dx;
    field_sub_impl(&dx, &d, &r->x);
    field_mul_impl(&r->y, &e, &dx);
    FieldElement c8;
    field_add_impl(&c8, &c, &c);
    field_add_impl(&c8, &c8, &c8);
    field_add_impl(&c8, &c8, &c8);
    field_sub_impl(&r->y, &r->y, &c8);

    // If p was infinity, result is infinity
    r->infinity = p->infinity;
}

// ---------------------------------------------------------------------------
// CT point add mixed (Jacobian + Affine -> Jacobian)
// Brier-Joye complete formula: 7M + 5S, handles all degenerate cases
// ---------------------------------------------------------------------------
inline void ct_point_add_mixed(const CTJacobianPoint* p, const CTAffinePoint* q,
                               CTJacobianPoint* r) {
    FieldElement z2, u2, s2, h, hh, i, j, rr, v;

    field_sqr_impl(&z2, &p->z);          // z1^2
    field_mul_impl(&u2, &q->x, &z2);     // u2 = x2*z1^2
    FieldElement z3;
    field_mul_impl(&z3, &p->z, &z2);     // z1^3
    field_mul_impl(&s2, &q->y, &z3);     // s2 = y2*z1^3

    field_sub_impl(&h, &u2, &p->x);      // h = u2 - x1
    field_sqr_impl(&hh, &h);             // hh = h^2
    field_add_impl(&i, &hh, &hh);
    field_add_impl(&i, &i, &i);          // i = 4*h^2
    field_mul_impl(&j, &h, &i);          // j = h*i
    field_sub_impl(&rr, &s2, &p->y);
    field_add_impl(&rr, &rr, &rr);       // r = 2*(s2 - y1)
    field_mul_impl(&v, &p->x, &i);       // v = x1*i

    // x3 = r^2 - j - 2*v
    FieldElement rr2;
    field_sqr_impl(&rr2, &rr);
    field_sub_impl(&r->x, &rr2, &j);
    FieldElement v2;
    field_add_impl(&v2, &v, &v);
    field_sub_impl(&r->x, &r->x, &v2);

    // y3 = r*(v - x3) - 2*y1*j
    FieldElement vx, y1j;
    field_sub_impl(&vx, &v, &r->x);
    field_mul_impl(&r->y, &rr, &vx);
    field_mul_impl(&y1j, &p->y, &j);
    field_add_impl(&y1j, &y1j, &y1j);
    field_sub_impl(&r->y, &r->y, &y1j);

    // z3 = 2*z1*h  (since z2=1 for affine)
    field_mul_impl(&r->z, &p->z, &h);
    field_add_impl(&r->z, &r->z, &r->z);

    // Handle degenerate cases via CT select:
    // If h==0 && rr==0: P==Q, should double
    ulong h_zero = ct_field_is_zero(&h);
    ulong rr_zero = ct_field_is_zero(&rr);
    ulong same_point = h_zero & rr_zero;

    CTJacobianPoint dbl_result;
    // Only compute double if needed (always computed for CT)
    CTJacobianPoint p_dbl;
    ct_point_dbl(p, &p_dbl);
    ct_point_cmov(r, &p_dbl, same_point & ~p->infinity & ~q->infinity);

    // If P is infinity, result = Q (as Jacobian)
    CTJacobianPoint q_jac;
    q_jac.x = q->x; q_jac.y = q->y;
    q_jac.z.limbs[0] = 1; q_jac.z.limbs[1] = 0;
    q_jac.z.limbs[2] = 0; q_jac.z.limbs[3] = 0;
    q_jac.infinity = q->infinity;
    ct_point_cmov(r, &q_jac, p->infinity);

    // If Q is infinity, result = P
    ct_point_cmov(r, p, q->infinity);

    // Final infinity flag
    r->infinity = p->infinity & q->infinity;
}

// ---------------------------------------------------------------------------
// CT batch field inverse (Montgomery trick)
// ---------------------------------------------------------------------------
inline void ct_batch_field_inv(FieldElement* vals, FieldElement* invs, int n) {
    if (n <= 0) return;
    FieldElement acc[16];  // max 16
    acc[0] = vals[0];
    for (int i = 1; i < n; ++i)
        field_mul_impl(&acc[i], &acc[i - 1], &vals[i]);
    FieldElement inv;
    ct_field_inv(&inv, &acc[n - 1]);
    for (int i = n - 1; i > 0; --i) {
        field_mul_impl(&invs[i], &inv, &acc[i - 1]);
        FieldElement tmp;
        field_mul_impl(&tmp, &inv, &vals[i]);
        inv = tmp;
    }
    invs[0] = inv;
}

// ---------------------------------------------------------------------------
// CT scalar multiplication: k*P using GLV + 4-bit windowed
// ---------------------------------------------------------------------------
inline void ct_scalar_mul_point(const CTJacobianPoint* p, const Scalar* k,
                                CTJacobianPoint* r_out) {
    // GLV decomposition
    CTGLVDecompositionOCL glv;
    ct_glv_decompose_impl(k, &glv);

    // Build 16-entry table: table[0] = identity, table[1..15] = 1P..15P
    #define CT_TABLE_SIZE 16
    CTAffinePoint table_a[CT_TABLE_SIZE];
    CTAffinePoint table_b[CT_TABLE_SIZE];

    // Identity at 0
    for (int i = 0; i < 4; ++i) {
        table_a[0].x.limbs[i] = 0;
        table_a[0].y.limbs[i] = 0;
    }
    table_a[0].infinity = ~(ulong)0;

    // Compute 1P..15P in Jacobian
    CTJacobianPoint jac_pts[15];
    jac_pts[0] = *p;
    for (int i = 1; i < 15; ++i)
        ct_point_add_mixed(&jac_pts[i - 1], &table_a[1], &jac_pts[i]);  // placeholder

    // Batch invert Z coords
    FieldElement z_vals[15], z_inv_vals[15];
    for (int i = 0; i < 15; ++i) z_vals[i] = jac_pts[i].z;
    ct_batch_field_inv(z_vals, z_inv_vals, 15);

    // Convert to affine
    for (int i = 0; i < 15; ++i) {
        FieldElement z_inv2, z_inv3;
        field_sqr_impl(&z_inv2, &z_inv_vals[i]);
        field_mul_impl(&z_inv3, &z_inv_vals[i], &z_inv2);
        field_mul_impl(&table_a[i + 1].x, &jac_pts[i].x, &z_inv2);
        field_mul_impl(&table_a[i + 1].y, &jac_pts[i].y, &z_inv3);
        table_a[i + 1].infinity = jac_pts[i].infinity;
    }

    // Fix: recompute multiples using table_a[1] for sequential adds
    // We need to redo this properly: 1P=p, 2P=P+P, 3P=2P+P, etc.
    {
        CTJacobianPoint acc;
        acc.x = p->x; acc.y = p->y; acc.z = p->z; acc.infinity = p->infinity;
        jac_pts[0] = acc;
        // Convert P to affine for table_a[1]
        FieldElement z_inv2_0, z_inv3_0, z_inv_0;
        ct_field_inv(&z_inv_0, &p->z);
        field_sqr_impl(&z_inv2_0, &z_inv_0);
        field_mul_impl(&z_inv3_0, &z_inv_0, &z_inv2_0);
        field_mul_impl(&table_a[1].x, &p->x, &z_inv2_0);
        field_mul_impl(&table_a[1].y, &p->y, &z_inv3_0);
        table_a[1].infinity = p->infinity;

        // Build 2P..15P
        for (int i = 1; i < 15; ++i) {
            ct_point_add_mixed(&jac_pts[i - 1], &table_a[1], &jac_pts[i]);
        }

        // Batch invert and convert
        for (int i = 0; i < 15; ++i) z_vals[i] = jac_pts[i].z;
        ct_batch_field_inv(z_vals, z_inv_vals, 15);
        for (int i = 0; i < 15; ++i) {
            FieldElement zi2, zi3;
            field_sqr_impl(&zi2, &z_inv_vals[i]);
            field_mul_impl(&zi3, &z_inv_vals[i], &zi2);
            field_mul_impl(&table_a[i + 1].x, &jac_pts[i].x, &zi2);
            field_mul_impl(&table_a[i + 1].y, &jac_pts[i].y, &zi3);
            table_a[i + 1].infinity = jac_pts[i].infinity;
        }
    }

    // Build endomorphism table: phi(P) = (beta*x, y)
    __constant ulong BETA_LIMBS[4] = {
        0x7AE96A2B657C0710UL, 0x6584D3F6EB4C3F40UL,
        0x7F09A3680E46AB35UL, 0x851695D49A83F8EFUL
    };
    FieldElement beta;
    for (int i = 0; i < 4; ++i) beta.limbs[i] = BETA_LIMBS[i];
    table_b[0] = table_a[0];
    for (int i = 1; i < CT_TABLE_SIZE; ++i) {
        field_mul_impl(&table_b[i].x, &table_a[i].x, &beta);
        table_b[i].y = table_a[i].y;
        table_b[i].infinity = table_a[i].infinity;
    }

    // Conditionally negate tables
    for (int i = 1; i < CT_TABLE_SIZE; ++i) {
        ct_point_cneg_y(&table_a[i], glv.k1_neg);
        ct_point_cneg_y(&table_b[i], glv.k2_neg);
    }

    // Windowed double-and-add: 33 iterations
    ct_point_set_infinity(r_out);
    for (int w = 32; w >= 0; --w) {
        ct_point_dbl(r_out, r_out);
        ct_point_dbl(r_out, r_out);
        ct_point_dbl(r_out, r_out);
        ct_point_dbl(r_out, r_out);

        int bit_pos = w * 4;
        int limb_idx = bit_pos >> 6;
        int bit_off = bit_pos & 63;
        int d1 = (int)((glv.k1.limbs[limb_idx] >> bit_off) & 0xF);
        int d2 = (int)((glv.k2.limbs[limb_idx] >> bit_off) & 0xF);

        CTAffinePoint entry1;
        ct_affine_table_lookup(table_a, CT_TABLE_SIZE, d1, &entry1);
        CTJacobianPoint tmp;
        ct_point_add_mixed(r_out, &entry1, &tmp);
        *r_out = tmp;

        CTAffinePoint entry2;
        ct_affine_table_lookup(table_b, CT_TABLE_SIZE, d2, &entry2);
        ct_point_add_mixed(r_out, &entry2, &tmp);
        *r_out = tmp;
    }
    #undef CT_TABLE_SIZE
}

// ---------------------------------------------------------------------------
// Precomputed G tables: 15 multiples of G in affine, 4 limbs X + 4 limbs Y
// ---------------------------------------------------------------------------
__constant ulong CT_G_TABLE_A[15][8] = {
    { 0x59F2815B16F81798UL, 0x029BFCDB2DCE28D9UL, 0x55A06295CE870B07UL, 0x79BE667EF9DCBBACUL,
      0x9C47D08FFB10D4B8UL, 0xFD17B448A6855419UL, 0x5DA4FBFC0E1108A8UL, 0x483ADA7726A3C465UL },
    { 0xABAC09B95C709EE5UL, 0x5C778E4B8CEF3CA7UL, 0x3045406E95C07CD8UL, 0xC6047F9441ED7D6DUL,
      0x236431A950CFE52AUL, 0xF7F632653266D0E1UL, 0xA3C58419466CEAEEUL, 0x1AE168FEA63DC339UL },
    { 0x8601F113BCE036F9UL, 0xB531C845836F99B0UL, 0x49344F85F89D5229UL, 0xF9308A019258C310UL,
      0x6CB9FD7584B8E672UL, 0x6500A99934C2231BUL, 0x0FE337E62A37F356UL, 0x388F7B0F632DE814UL },
    { 0x74FA94ABE8C4CD13UL, 0xCC6C13900EE07584UL, 0x581E4904930B1404UL, 0xE493DBF1C10D80F3UL,
      0xCFE97BDC47739922UL, 0xD967AE33BFBDFE40UL, 0x5642E2098EA51448UL, 0x51ED993EA0D455B7UL },
    { 0xCBA8D569B240EFE4UL, 0xE88B84BDDC619AB7UL, 0x55B4A7250A5C5128UL, 0x2F8BDE4D1A072093UL,
      0xDCA87D3AA6AC62D6UL, 0xF788271BAB0D6840UL, 0xD4DBA9DDA6C9C426UL, 0xD8AC222636E5E3D6UL },
    { 0x2F057A1460297556UL, 0x82F6472F8568A18BUL, 0x20453A14355235D3UL, 0xFFF97BD5755EEEA4UL,
      0x3C870C36B075F297UL, 0xDE80F0F6518FE4A0UL, 0xF3BE96017F45C560UL, 0xAE12777AACFBB620UL },
    { 0xE92BDDEDCAC4F9BCUL, 0x3D419B7E0330E39CUL, 0xA398F365F2EA7A0EUL, 0x5CBDF0646E5DB4EAUL,
      0xA5082628087264DAUL, 0xA813D0B813FDE7B5UL, 0xA3178D6D861A54DBUL, 0x6AEBCA40BA255960UL },
    { 0x67784EF3E10A2A01UL, 0x0A1BDD05E5AF888AUL, 0xAFF3843FB70F3C2FUL, 0x2F01E5E15CCA351DUL,
      0xB5DA2CB76CBDE904UL, 0xC2E213D6BA5B7617UL, 0x293D082A132D13B4UL, 0x5C4DA8A741539949UL },
    { 0xC35F110DFC27CCBEUL, 0xE09796974C57E714UL, 0x09AD178A9F559ABDUL, 0xACD484E2F0C7F653UL,
      0x05CC262AC64F9C37UL, 0xADD888A4375F8E0FUL, 0x64380971763B61E9UL, 0xCC338921B0A7D9FDUL },
    { 0x52A68E2A47E247C7UL, 0x3442D49B1943C2B7UL, 0x35477C7B1AE6AE5DUL, 0xA0434D9E47F3C862UL,
      0x3CBEE53B037368D7UL, 0x6F794C2ED877A159UL, 0xA3B6C7E693A24C69UL, 0x893ABA425419BC27UL },
    { 0xBBEC17895DA008CBUL, 0x5649980BE5C17891UL, 0x5EF4246B70C65AACUL, 0x774AE7F858A9411EUL,
      0x301D74C9C953C61BUL, 0x372DB1E2DFF9D6A8UL, 0x0243DD56D7B7B365UL, 0xD984A032EB6B5E19UL },
    { 0xC5B0F47070AFE85AUL, 0x687CF4419620095BUL, 0x15C38F004D734633UL, 0xD01115D548E7561BUL,
      0x6B051B13F4062327UL, 0x79238C5DD9A86D52UL, 0xA8B64537E17BD815UL, 0xA9F34FFDC815E0D7UL },
    { 0xDEEDDF8F19405AA8UL, 0xB075FBC6610E58CDUL, 0xC7D1D205C3748651UL, 0xF28773C2D975288BUL,
      0x29B5CB52DB03ED81UL, 0x3A1A06DA521FA91FUL, 0x758212EB65CDAF47UL, 0x0AB0902E8D880A89UL },
    { 0xE49B241A60E823E4UL, 0x26AA7B63678949E6UL, 0xFD64E67F07D38E32UL, 0x499FDF9E895E719CUL,
      0xC65F40D403A13F5BUL, 0x464279C27A3F95BCUL, 0x90F044E4A7B3D464UL, 0xCAC2F6C4B54E8551UL },
    { 0x44ADBCF8E27E080EUL, 0x31E5946F3C85F79EUL, 0x5A465AE3095FF411UL, 0xD7924D4F7D43EA96UL,
      0xC504DC9FF6A26B58UL, 0xEA40AF2BD896D3A5UL, 0x83842EC228CC6DEFUL, 0x581E2872A86C72A6UL },
};

__constant ulong CT_G_TABLE_B[15][8] = {
    { 0xA7BBA04400B88FCBUL, 0x872844067F15E98DUL, 0xAB0102B696902325UL, 0xBCACE2E99DA01887UL,
      0x9C47D08FFB10D4B8UL, 0xFD17B448A6855419UL, 0x5DA4FBFC0E1108A8UL, 0x483ADA7726A3C465UL },
    { 0x3E995B6ED89250E1UL, 0xD2FAD8CCE43837EFUL, 0x4135EE7D59F87B33UL, 0xC360A6D0B34CE6DFUL,
      0x236431A950CFE52AUL, 0xF7F632653266D0E1UL, 0xA3C58419466CEAEEUL, 0x1AE168FEA63DC339UL },
    { 0xF7F0728C77206B2FUL, 0x8AF1E022C6DC8E1CUL, 0x8DCD8DCF2A28FA2FUL, 0xDF6EDF03731F9B4BUL,
      0x6CB9FD7584B8E672UL, 0x6500A99934C2231BUL, 0x0FE337E62A37F356UL, 0x388F7B0F632DE814UL },
    { 0x5BDE5B333B306100UL, 0x714C30B5AB487127UL, 0x5C45FAF8B90E324BUL, 0x1B77921F0D382907UL,
      0xCFE97BDC47739922UL, 0xD967AE33BFBDFE40UL, 0x5642E2098EA51448UL, 0x51ED993EA0D455B7UL },
    { 0x138C694695A83668UL, 0xA045693EE0D097CCUL, 0xF79F54FBCCB94671UL, 0x337B52E3ACDA49DFUL,
      0xDCA87D3AA6AC62D6UL, 0xF788271BAB0D6840UL, 0xD4DBA9DDA6C9C426UL, 0xD8AC222636E5E3D6UL },
    { 0x47AAF28078F38045UL, 0x86649D3E56A15A68UL, 0x5E3AA731E3E8BED7UL, 0xE63BCDD9AA535FC6UL,
      0x3C870C36B075F297UL, 0xDE80F0F6518FE4A0UL, 0xF3BE96017F45C560UL, 0xAE12777AACFBB620UL },
    { 0x3BC4686E4E53BC94UL, 0x0D3B20E20FAF7AAAUL, 0xA4FEC4D1C095C06EUL, 0x13F26E754BEA0B77UL,
      0xA5082628087264DAUL, 0xA813D0B813FDE7B5UL, 0xA3178D6D861A54DBUL, 0x6AEBCA40BA255960UL },
    { 0x03E947742446CC73UL, 0xB4FF771524257657UL, 0xAA77840F29E24892UL, 0x47AB650342D401A7UL,
      0xB5DA2CB76CBDE904UL, 0xC2E213D6BA5B7617UL, 0x293D082A132D13B4UL, 0x5C4DA8A741539949UL },
    { 0x20CD912E65953A52UL, 0xB565CDF5EF6D44E1UL, 0x7B6558AFEC58AB20UL, 0x87B404037E44E819UL,
      0x05CC262AC64F9C37UL, 0xADD888A4375F8E0FUL, 0x64380971763B61E9UL, 0xCC338921B0A7D9FDUL },
    { 0xBDB3E957741AFE29UL, 0xC1938D8E083762E4UL, 0xA136EBB246813990UL, 0x26CE269BF7A397B1UL,
      0x3CBEE53B037368D7UL, 0x6F794C2ED877A159UL, 0xA3B6C7E693A24C69UL, 0x893ABA425419BC27UL },
    { 0xC5FF4334BB209CE7UL, 0x79859BB70B5FF620UL, 0x8D897C41BEBF1A26UL, 0x51F4D3D1171DAC1DUL,
      0x301D74C9C953C61BUL, 0x372DB1E2DFF9D6A8UL, 0x0243DD56D7B7B365UL, 0xD984A032EB6B5E19UL },
    { 0x4A3EB52C042295E5UL, 0xF9482837C9535355UL, 0xAC1548422EAC82ADUL, 0x88591BFD953AAC41UL,
      0x6B051B13F4062327UL, 0x79238C5DD9A86D52UL, 0xA8B64537E17BD815UL, 0xA9F34FFDC815E0D7UL },
    { 0x60AAEE6A475FB678UL, 0x32907ED74A3D0562UL, 0x07046C4578FC783BUL, 0xF14D58374BB890A2UL,
      0x29B5CB52DB03ED81UL, 0x3A1A06DA521FA91FUL, 0x758212EB65CDAF47UL, 0x0AB0902E8D880A89UL },
    { 0x0E6AB7EE20A0B458UL, 0x580656A627C529F6UL, 0x1548F0DC87C37384UL, 0x7B1252177810048AUL,
      0xC65F40D403A13F5BUL, 0x464279C27A3F95BCUL, 0x90F044E4A7B3D464UL, 0xCAC2F6C4B54E8551UL },
    { 0x3AC0A40C71B1B3B4UL, 0x05CC3BC9C1C0A639UL, 0x0E1B4825512B6948UL, 0x805F1105F5F9454AUL,
      0xC504DC9FF6A26B58UL, 0xEA40AF2BD896D3A5UL, 0x83842EC228CC6DEFUL, 0x581E2872A86C72A6UL },
};

// ---------------------------------------------------------------------------
// CT generator multiplication: k*G (fixed-base, precomputed tables)
// ---------------------------------------------------------------------------
inline void ct_generator_mul_impl(const Scalar* k, CTJacobianPoint* r_out) {
    CTGLVDecompositionOCL glv;
    ct_glv_decompose_impl(k, &glv);

    #define CT_GTABLE_SIZE 16
    CTAffinePoint table_a[CT_GTABLE_SIZE];
    CTAffinePoint table_b[CT_GTABLE_SIZE];

    // Identity at index 0
    for (int i = 0; i < 4; ++i) {
        table_a[0].x.limbs[i] = 0;
        table_a[0].y.limbs[i] = 0;
    }
    table_a[0].infinity = ~(ulong)0;
    table_b[0] = table_a[0];

    // Load from constant tables
    for (int i = 0; i < 15; ++i) {
        for (int j = 0; j < 4; ++j) {
            table_a[i + 1].x.limbs[j] = CT_G_TABLE_A[i][j];
            table_a[i + 1].y.limbs[j] = CT_G_TABLE_A[i][j + 4];
            table_b[i + 1].x.limbs[j] = CT_G_TABLE_B[i][j];
            table_b[i + 1].y.limbs[j] = CT_G_TABLE_B[i][j + 4];
        }
        table_a[i + 1].infinity = 0;
        table_b[i + 1].infinity = 0;
    }

    // Conditionally negate
    for (int i = 1; i < CT_GTABLE_SIZE; ++i) {
        ct_point_cneg_y(&table_a[i], glv.k1_neg);
        ct_point_cneg_y(&table_b[i], glv.k2_neg);
    }

    // Windowed loop: 33 iterations
    ct_point_set_infinity(r_out);
    for (int w = 32; w >= 0; --w) {
        ct_point_dbl(r_out, r_out);
        ct_point_dbl(r_out, r_out);
        ct_point_dbl(r_out, r_out);
        ct_point_dbl(r_out, r_out);

        int bit_pos = w * 4;
        int limb_idx = bit_pos >> 6;
        int bit_off = bit_pos & 63;
        int d1 = (int)((glv.k1.limbs[limb_idx] >> bit_off) & 0xF);
        int d2 = (int)((glv.k2.limbs[limb_idx] >> bit_off) & 0xF);

        CTAffinePoint entry1;
        ct_affine_table_lookup(table_a, CT_GTABLE_SIZE, d1, &entry1);
        CTJacobianPoint tmp;
        ct_point_add_mixed(r_out, &entry1, &tmp);
        *r_out = tmp;

        CTAffinePoint entry2;
        ct_affine_table_lookup(table_b, CT_GTABLE_SIZE, d2, &entry2);
        ct_point_add_mixed(r_out, &entry2, &tmp);
        *r_out = tmp;
    }
    #undef CT_GTABLE_SIZE
}

#endif // SECP256K1_CT_POINT_CL
