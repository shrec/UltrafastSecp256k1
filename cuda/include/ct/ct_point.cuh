#pragma once
// ============================================================================
// Constant-Time Point Arithmetic -- CUDA Device
// ============================================================================
// Side-channel resistant point operations for secp256k1.
//
// Key features:
//   - Brier-Joye complete addition (11M+6S Jac+Jac, 7M+5S Jac+Aff)
//   - Handles ALL cases: P+Q, P+P, P+O, O+Q, P+(-P)=O
//   - Mask-based infinity flag (uint64_t, not bool)
//   - CT table lookup (scans ALL entries)
//   - CT generator_mul (fixed-trace, signed-digit comb)
//   - CT scalar_mul (GLV + Hamburg wNAF, fixed-trace)
//
// Port of: cpu/include/secp256k1/ct/point.hpp + cpu/src/ct_point.cpp
// ============================================================================

#include "ct/ct_scalar.cuh"

namespace secp256k1 {
namespace cuda {
namespace ct {

// --- CT Point Types ----------------------------------------------------------

// Jacobian point with mask-based infinity flag (not bool)
struct CTJacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    uint64_t infinity;  // 0 = normal, 0xFFFF... = point at infinity
};

// Affine point with mask-based infinity flag
struct CTAffinePoint {
    FieldElement x;
    FieldElement y;
    uint64_t infinity;  // 0 = normal, 0xFFFF... = point at infinity
};

// --- CT Point Utilities ------------------------------------------------------

__device__ __forceinline__
void ct_point_set_infinity(CTJacobianPoint* p) {
    field_set_zero(&p->x);
    field_set_zero(&p->y);
    field_set_zero(&p->z);
    p->infinity = ~(uint64_t)0;
}

__device__ __forceinline__
CTJacobianPoint ct_point_from_jacobian(const JacobianPoint* p) {
    CTJacobianPoint r;
    r.x = p->x; r.y = p->y; r.z = p->z;
    r.infinity = bool_to_mask((uint64_t)p->infinity);
    return r;
}

__device__ __forceinline__
JacobianPoint ct_point_to_jacobian(const CTJacobianPoint* p) {
    JacobianPoint r;
    r.x = p->x; r.y = p->y; r.z = p->z;
    r.infinity = (p->infinity != 0);
    return r;
}

__device__ __forceinline__
CTJacobianPoint ct_point_from_affine(const CTAffinePoint* a) {
    CTJacobianPoint r;
    r.x = a->x; r.y = a->y;
    field_set_one(&r.z);
    r.infinity = a->infinity;
    return r;
}

// CT conditional move for CT Jacobian points
__device__ __forceinline__
void ct_point_cmov(CTJacobianPoint* r, const CTJacobianPoint* a, uint64_t mask) {
    cmov256(r->x.limbs, a->x.limbs, mask);
    cmov256(r->y.limbs, a->y.limbs, mask);
    cmov256(r->z.limbs, a->z.limbs, mask);
    cmov64(&r->infinity, a->infinity, mask);
}

// CT conditional move for CT Affine points
__device__ __forceinline__
void ct_aff_cmov(CTAffinePoint* r, const CTAffinePoint* a, uint64_t mask) {
    cmov256(r->x.limbs, a->x.limbs, mask);
    cmov256(r->y.limbs, a->y.limbs, mask);
    cmov64(&r->infinity, a->infinity, mask);
}

// CT conditional negate Y
__device__ __forceinline__
void ct_point_cneg_y(CTAffinePoint* p, uint64_t mask) {
    FieldElement neg_y;
    secp256k1::cuda::field_negate(&p->y, &neg_y);
    cmov256(p->y.limbs, neg_y.limbs, mask);
}

// CT table lookup for affine points: scans ALL entries
__device__ inline
void ct_affine_table_lookup(const CTAffinePoint* table, int count,
                            int index, CTAffinePoint* out) {
    out->x.limbs[0] = 0; out->x.limbs[1] = 0;
    out->x.limbs[2] = 0; out->x.limbs[3] = 0;
    out->y.limbs[0] = 0; out->y.limbs[1] = 0;
    out->y.limbs[2] = 0; out->y.limbs[3] = 0;
    out->infinity = 0;

    for (int i = 0; i < count; i++) {
        uint64_t m = eq_mask((uint64_t)i, (uint64_t)index);
        out->x.limbs[0] |= (table[i].x.limbs[0] & m);
        out->x.limbs[1] |= (table[i].x.limbs[1] & m);
        out->x.limbs[2] |= (table[i].x.limbs[2] & m);
        out->x.limbs[3] |= (table[i].x.limbs[3] & m);
        out->y.limbs[0] |= (table[i].y.limbs[0] & m);
        out->y.limbs[1] |= (table[i].y.limbs[1] & m);
        out->y.limbs[2] |= (table[i].y.limbs[2] & m);
        out->y.limbs[3] |= (table[i].y.limbs[3] & m);
        out->infinity   |= (table[i].infinity & m);
    }
}

// --- CT Point Doubling (4M + 4S + 2add + 2sub) --------------------------------
// Standard Jacobian doubling (same formula as fast path's jacobian_double).
// Handles identity via cmov.

__device__ inline
void ct_point_dbl(const CTJacobianPoint* p, CTJacobianPoint* r) {
    using namespace secp256k1::cuda;

    FieldElement S, M, X3, Y3, Z3, YY, YYYY, t1;

    // YY = Y^2
    field_sqr(&p->y, &YY);

    // S = 4*X*Y^2
    field_mul(&p->x, &YY, &S);
    field_add(&S, &S, &S);
    field_add(&S, &S, &S);

    // M = 3*X^2
    field_sqr(&p->x, &M);
    field_add(&M, &M, &t1);     // t1 = 2*X^2
    field_add(&M, &t1, &M);     // M = 3*X^2

    // X3 = M^2 - 2*S
    field_sqr(&M, &X3);
    field_add(&S, &S, &t1);     // t1 = 2*S
    field_sub(&X3, &t1, &X3);

    // YYYY = Y^4
    field_sqr(&YY, &YYYY);

    // Y3 = M*(S - X3) - 8*Y^4
    field_add(&YYYY, &YYYY, &t1);   // 2*Y^4
    field_add(&t1, &t1, &t1);       // 4*Y^4
    field_add(&t1, &t1, &t1);       // 8*Y^4
    field_sub(&S, &X3, &S);         // S - X3 (reuse S)
    field_mul(&M, &S, &Y3);         // M*(S - X3)
    field_sub(&Y3, &t1, &Y3);       // Y3 final

    // Z3 = 2*Y*Z
    field_mul(&p->y, &p->z, &Z3);
    field_add(&Z3, &Z3, &Z3);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = p->infinity;

    // If input was infinity, cmov to identity
    CTJacobianPoint inf;
    ct_point_set_infinity(&inf);
    ct_point_cmov(r, &inf, p->infinity);
}

// --- CT Complete Addition (Jac + Aff, Brier-Joye, 7M + 5S) ------------------
// Handles ALL cases in ONE codepath: P+Q, P+P (doubling), P+O, O+Q, P+(-P)=O

__device__ inline
void ct_point_add_mixed(const CTJacobianPoint* p, const CTAffinePoint* q,
                        CTJacobianPoint* r) {
    using namespace secp256k1::cuda;

    // ZZ = Z1^2
    FieldElement zz;
    field_sqr(&p->z, &zz);

    // U1 = X1, U2 = q.x * ZZ
    FieldElement u1, u2;
    u1 = p->x;
    field_mul(&q->x, &zz, &u2);

    // S1 = Y1, S2 = q.y * ZZ * Z1
    FieldElement s1, s2;
    s1 = p->y;
    FieldElement zzz;
    field_mul(&zz, &p->z, &zzz);
    field_mul(&q->y, &zzz, &s2);

    // T = U1 + U2
    FieldElement t;
    field_add(&u1, &u2, &t);

    // M = S1 + S2
    FieldElement m;
    field_add(&s1, &s2, &m);

    // R = T^2 - U1*U2
    FieldElement t_sq, u1u2, rr;
    field_sqr(&t, &t_sq);
    field_mul(&u1, &u2, &u1u2);
    field_sub(&t_sq, &u1u2, &rr);

    // Degenerate check: M == 0 means P == -Q (doubling case)
    uint64_t m_is_zero = ct::field_is_zero(&m);

    // Ralt = degen ? 2*S1 : R
    FieldElement ralt, s1_2;
    field_add(&s1, &s1, &s1_2);
    ct_select256(ralt.limbs, s1_2.limbs, rr.limbs, m_is_zero);

    // Malt = degen ? U1-U2 : M
    FieldElement malt, u1_u2;
    field_sub(&u1, &u2, &u1_u2);
    ct_select256(malt.limbs, u1_u2.limbs, m.limbs, m_is_zero);

    // N = Malt^2
    FieldElement n;
    field_sqr(&malt, &n);

    // Q_ = -T * N
    FieldElement q_;
    field_mul(&t, &n, &q_);
    field_negate(&q_, &q_);

    // N = N^2 (reuse N)
    FieldElement nn;
    field_sqr(&n, &nn);

    // X3 = Ralt^2 + Q_
    FieldElement x3;
    field_sqr(&ralt, &x3);
    field_add(&x3, &q_, &x3);

    // Z3 = Malt * Z1
    FieldElement z3;
    field_mul(&malt, &p->z, &z3);

    // Y3 = -(Ralt * (2*X3 + Q_) + N) / 2
    FieldElement x3_2, y3_tmp, y3;
    field_add(&x3, &x3, &x3_2);
    field_add(&x3_2, &q_, &y3_tmp);
    field_mul(&ralt, &y3_tmp, &y3);
    field_add(&y3, &nn, &y3);
    field_negate(&y3, &y3);
    field_half(&y3, &y3);

    r->x = x3;
    r->y = y3;
    r->z = z3;
    r->infinity = 0;

    // Check Z3 == 0 (means P + (-P) = O)
    uint64_t z3_zero = ct::field_is_zero(&z3);
    r->infinity = z3_zero;

    // If P was infinity, result = Q (as Jacobian)
    CTJacobianPoint q_jac;
    q_jac.x = q->x; q_jac.y = q->y;
    field_set_one(&q_jac.z);
    q_jac.infinity = q->infinity;
    ct_point_cmov(r, &q_jac, p->infinity);

    // If Q was infinity, result = P
    ct_point_cmov(r, p, q->infinity);
}

// --- CT Complete Addition (Jac + Jac, Brier-Joye, 11M + 6S) -----------------

__device__ inline
void ct_point_add(const CTJacobianPoint* p, const CTJacobianPoint* q,
                  CTJacobianPoint* r) {
    using namespace secp256k1::cuda;

    // Z1Z1 = Z1^2, Z2Z2 = Z2^2
    FieldElement z1z1, z2z2;
    field_sqr(&p->z, &z1z1);
    field_sqr(&q->z, &z2z2);

    // U1 = X1 * Z2Z2, U2 = X2 * Z1Z1
    FieldElement u1, u2;
    field_mul(&p->x, &z2z2, &u1);
    field_mul(&q->x, &z1z1, &u2);

    // S1 = Y1 * Z2Z2 * Z2, S2 = Y2 * Z1Z1 * Z1
    FieldElement s1, s2, z2z2z2, z1z1z1;
    field_mul(&z2z2, &q->z, &z2z2z2);
    field_mul(&p->y, &z2z2z2, &s1);
    field_mul(&z1z1, &p->z, &z1z1z1);
    field_mul(&q->y, &z1z1z1, &s2);

    // Z = Z1 * Z2
    FieldElement z;
    field_mul(&p->z, &q->z, &z);

    // T = U1 + U2
    FieldElement t;
    field_add(&u1, &u2, &t);

    // M = S1 + S2
    FieldElement m;
    field_add(&s1, &s2, &m);

    // R = T^2 - U1*U2
    FieldElement t_sq, u1u2, rr;
    field_sqr(&t, &t_sq);
    field_mul(&u1, &u2, &u1u2);
    field_sub(&t_sq, &u1u2, &rr);

    // Degenerate check
    uint64_t m_is_zero = ct::field_is_zero(&m);

    FieldElement ralt, s1_2;
    field_add(&s1, &s1, &s1_2);
    ct_select256(ralt.limbs, s1_2.limbs, rr.limbs, m_is_zero);

    FieldElement malt, u1_u2;
    field_sub(&u1, &u2, &u1_u2);
    ct_select256(malt.limbs, u1_u2.limbs, m.limbs, m_is_zero);

    FieldElement n;
    field_sqr(&malt, &n);

    FieldElement q_;
    field_mul(&t, &n, &q_);
    field_negate(&q_, &q_);

    FieldElement nn;
    field_sqr(&n, &nn);

    FieldElement x3;
    field_sqr(&ralt, &x3);
    field_add(&x3, &q_, &x3);

    // Z3 = Z * Malt  (note: Z = Z1*Z2, not just Z1)
    FieldElement z3;
    field_mul(&z, &malt, &z3);

    FieldElement x3_2, y3_tmp, y3;
    field_add(&x3, &x3, &x3_2);
    field_add(&x3_2, &q_, &y3_tmp);
    field_mul(&ralt, &y3_tmp, &y3);
    field_add(&y3, &nn, &y3);
    field_negate(&y3, &y3);
    field_half(&y3, &y3);

    r->x = x3;
    r->y = y3;
    r->z = z3;
    r->infinity = 0;

    uint64_t z3_zero = ct::field_is_zero(&z3);
    r->infinity = z3_zero;

    ct_point_cmov(r, q, p->infinity);
    ct_point_cmov(r, p, q->infinity);
}

// --- CT Point Negation -------------------------------------------------------

__device__ __forceinline__
void ct_point_neg(const CTJacobianPoint* p, CTJacobianPoint* r) {
    r->x = p->x;
    secp256k1::cuda::field_negate(&p->y, &r->y);
    r->z = p->z;
    r->infinity = p->infinity;
}

// --- Montgomery Batch Field Inversion ----------------------------------------
// Inverts N field elements using only 1 field_inv + 3*(N-1) field_mul.
// inputs[i] and outputs[i] may alias.

__device__ inline
void ct_batch_field_inv(const FieldElement* inputs, FieldElement* outputs, int count) {
    using namespace secp256k1::cuda;

    if (count == 0) return;
    if (count == 1) { field_inv(&inputs[0], &outputs[0]); return; }

    // Running products: prod[i] = inputs[0] * inputs[1] * ... * inputs[i]
    FieldElement products[16];  // max supported count
    products[0] = inputs[0];
    for (int i = 1; i < count; i++) {
        field_mul(&products[i - 1], &inputs[i], &products[i]);
    }

    // Single inversion of the full product
    FieldElement inv_all;
    field_inv(&products[count - 1], &inv_all);

    // Recover individual inverses right-to-left
    for (int i = count - 1; i > 0; i--) {
        field_mul(&inv_all, &products[i - 1], &outputs[i]);
        field_mul(&inv_all, &inputs[i], &inv_all);
    }
    outputs[0] = inv_all;
}

// --- CT Scalar Multiplication: k*P (GLV + 4-bit windowed, CT) ----------------
// Uses CT complete addition, 16-entry table [0P..15P], CT table lookups.
// Cost: ~132 doublings + ~66 mixed additions (vs 128 dbl + 256 add bit-by-bit)

__device__ inline
void ct_scalar_mul(const JacobianPoint* p_in, const Scalar* k,
                   JacobianPoint* r_out) {
    using namespace secp256k1::cuda;

    // Convert to CT types
    CTJacobianPoint p = ct_point_from_jacobian(p_in);

    // GLV decompose (CT version)
    CTGLVDecomposition glv = ct_glv_decompose(k);

    // Build precomputed table: [0: identity, 1: 1P, 2: 2P, ..., 15: 15P]
    // 4-bit window digit directly indexes the table
    constexpr int TABLE_SIZE = 16;
    CTAffinePoint table_a[TABLE_SIZE];
    CTAffinePoint table_b[TABLE_SIZE];

    // Entry 0: identity (infinity)
    field_set_zero(&table_a[0].x);
    field_set_zero(&table_a[0].y);
    table_a[0].infinity = ~(uint64_t)0;

    // Compute 1P through 15P in Jacobian via sequential Jac+Jac additions
    CTJacobianPoint jac_pts[15];
    jac_pts[0] = p;  // 1P
    for (int i = 1; i < 15; i++) {
        ct_point_add(&jac_pts[i - 1], &p, &jac_pts[i]);  // (i+1)P
    }

    // Batch invert all 15 Z coordinates (1 field_inv + 42 field_mul)
    FieldElement z_vals[15], z_inv_vals[15];
    for (int i = 0; i < 15; i++) z_vals[i] = jac_pts[i].z;
    ct_batch_field_inv(z_vals, z_inv_vals, 15);

    // Convert to affine using batch-inverted Z values
    for (int i = 0; i < 15; i++) {
        FieldElement z_inv2, z_inv3;
        field_sqr(&z_inv_vals[i], &z_inv2);
        field_mul(&z_inv_vals[i], &z_inv2, &z_inv3);
        field_mul(&jac_pts[i].x, &z_inv2, &table_a[i + 1].x);
        field_mul(&jac_pts[i].y, &z_inv3, &table_a[i + 1].y);
        table_a[i + 1].infinity = jac_pts[i].infinity;
    }

    // Build endomorphism table: phi(P) = (beta*x, y) on secp256k1
    FieldElement beta;
    for (int i = 0; i < 4; i++) beta.limbs[i] = BETA[i];
    table_b[0] = table_a[0];  // identity
    for (int i = 1; i < TABLE_SIZE; i++) {
        field_mul(&table_a[i].x, &beta, &table_b[i].x);
        table_b[i].y = table_a[i].y;
        table_b[i].infinity = table_a[i].infinity;
    }

    // Conditionally negate tables based on GLV sign (skip identity at 0)
    for (int i = 1; i < TABLE_SIZE; i++) {
        ct_point_cneg_y(&table_a[i], glv.k1_neg);
        ct_point_cneg_y(&table_b[i], glv.k2_neg);
    }

    // Windowed double-and-add: process 4 bits at a time
    // k1, k2 fit in ~129 bits; ceil(129/4) = 33 windows
    CTJacobianPoint result;
    ct_point_set_infinity(&result);

    for (int w = 32; w >= 0; --w) {
        // 4 doublings (multiply accumulator by 16)
        ct_point_dbl(&result, &result);
        ct_point_dbl(&result, &result);
        ct_point_dbl(&result, &result);
        ct_point_dbl(&result, &result);

        // Extract 4-bit digits from k1, k2 at window position w
        int bit_pos  = w * 4;
        int limb_idx = bit_pos >> 6;   // bit_pos / 64
        int bit_off  = bit_pos & 63;   // bit_pos % 64
        int d1 = (int)((glv.k1.limbs[limb_idx] >> bit_off) & 0xF);
        int d2 = (int)((glv.k2.limbs[limb_idx] >> bit_off) & 0xF);

        // CT table lookup + mixed add for k1 component
        CTAffinePoint entry1;
        ct_affine_table_lookup(table_a, TABLE_SIZE, d1, &entry1);
        CTJacobianPoint tmp;
        ct_point_add_mixed(&result, &entry1, &tmp);
        result = tmp;

        // CT table lookup + mixed add for k2 component
        CTAffinePoint entry2;
        ct_affine_table_lookup(table_b, TABLE_SIZE, d2, &entry2);
        ct_point_add_mixed(&result, &entry2, &tmp);
        result = tmp;
    }

    *r_out = ct_point_to_jacobian(&result);
}

// --- CT Generator Multiplication: k*G (fixed-base, 4-bit windowed, CT) -------
// Uses precomputed tables in __constant__ memory: 1G through 15G + identity.
// Same windowed algorithm as ct_scalar_mul but zero-cost table build.

// Precomputed table: multiples of G in affine (1G, 2G, ..., 15G)
// Layout: [15][8] = 15 points x (4 limbs X + 4 limbs Y)
__constant__ const uint64_t G_TABLE_A[15][8] = {
    // 1G
    { 0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL,
      0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL },
    // 2G
    { 0xABAC09B95C709EE5ULL, 0x5C778E4B8CEF3CA7ULL, 0x3045406E95C07CD8ULL, 0xC6047F9441ED7D6DULL,
      0x236431A950CFE52AULL, 0xF7F632653266D0E1ULL, 0xA3C58419466CEAEEULL, 0x1AE168FEA63DC339ULL },
    // 3G
    { 0x8601F113BCE036F9ULL, 0xB531C845836F99B0ULL, 0x49344F85F89D5229ULL, 0xF9308A019258C310ULL,
      0x6CB9FD7584B8E672ULL, 0x6500A99934C2231BULL, 0x0FE337E62A37F356ULL, 0x388F7B0F632DE814ULL },
    // 4G
    { 0x74FA94ABE8C4CD13ULL, 0xCC6C13900EE07584ULL, 0x581E4904930B1404ULL, 0xE493DBF1C10D80F3ULL,
      0xCFE97BDC47739922ULL, 0xD967AE33BFBDFE40ULL, 0x5642E2098EA51448ULL, 0x51ED993EA0D455B7ULL },
    // 5G
    { 0xCBA8D569B240EFE4ULL, 0xE88B84BDDC619AB7ULL, 0x55B4A7250A5C5128ULL, 0x2F8BDE4D1A072093ULL,
      0xDCA87D3AA6AC62D6ULL, 0xF788271BAB0D6840ULL, 0xD4DBA9DDA6C9C426ULL, 0xD8AC222636E5E3D6ULL },
    // 6G
    { 0x2F057A1460297556ULL, 0x82F6472F8568A18BULL, 0x20453A14355235D3ULL, 0xFFF97BD5755EEEA4ULL,
      0x3C870C36B075F297ULL, 0xDE80F0F6518FE4A0ULL, 0xF3BE96017F45C560ULL, 0xAE12777AACFBB620ULL },
    // 7G
    { 0xE92BDDEDCAC4F9BCULL, 0x3D419B7E0330E39CULL, 0xA398F365F2EA7A0EULL, 0x5CBDF0646E5DB4EAULL,
      0xA5082628087264DAULL, 0xA813D0B813FDE7B5ULL, 0xA3178D6D861A54DBULL, 0x6AEBCA40BA255960ULL },
    // 8G
    { 0x67784EF3E10A2A01ULL, 0x0A1BDD05E5AF888AULL, 0xAFF3843FB70F3C2FULL, 0x2F01E5E15CCA351DULL,
      0xB5DA2CB76CBDE904ULL, 0xC2E213D6BA5B7617ULL, 0x293D082A132D13B4ULL, 0x5C4DA8A741539949ULL },
    // 9G
    { 0xC35F110DFC27CCBEULL, 0xE09796974C57E714ULL, 0x09AD178A9F559ABDULL, 0xACD484E2F0C7F653ULL,
      0x05CC262AC64F9C37ULL, 0xADD888A4375F8E0FULL, 0x64380971763B61E9ULL, 0xCC338921B0A7D9FDULL },
    // 10G
    { 0x52A68E2A47E247C7ULL, 0x3442D49B1943C2B7ULL, 0x35477C7B1AE6AE5DULL, 0xA0434D9E47F3C862ULL,
      0x3CBEE53B037368D7ULL, 0x6F794C2ED877A159ULL, 0xA3B6C7E693A24C69ULL, 0x893ABA425419BC27ULL },
    // 11G
    { 0xBBEC17895DA008CBULL, 0x5649980BE5C17891ULL, 0x5EF4246B70C65AACULL, 0x774AE7F858A9411EULL,
      0x301D74C9C953C61BULL, 0x372DB1E2DFF9D6A8ULL, 0x0243DD56D7B7B365ULL, 0xD984A032EB6B5E19ULL },
    // 12G
    { 0xC5B0F47070AFE85AULL, 0x687CF4419620095BULL, 0x15C38F004D734633ULL, 0xD01115D548E7561BULL,
      0x6B051B13F4062327ULL, 0x79238C5DD9A86D52ULL, 0xA8B64537E17BD815ULL, 0xA9F34FFDC815E0D7ULL },
    // 13G
    { 0xDEEDDF8F19405AA8ULL, 0xB075FBC6610E58CDULL, 0xC7D1D205C3748651ULL, 0xF28773C2D975288BULL,
      0x29B5CB52DB03ED81ULL, 0x3A1A06DA521FA91FULL, 0x758212EB65CDAF47ULL, 0x0AB0902E8D880A89ULL },
    // 14G
    { 0xE49B241A60E823E4ULL, 0x26AA7B63678949E6ULL, 0xFD64E67F07D38E32ULL, 0x499FDF9E895E719CULL,
      0xC65F40D403A13F5BULL, 0x464279C27A3F95BCULL, 0x90F044E4A7B3D464ULL, 0xCAC2F6C4B54E8551ULL },
    // 15G
    { 0x44ADBCF8E27E080EULL, 0x31E5946F3C85F79EULL, 0x5A465AE3095FF411ULL, 0xD7924D4F7D43EA96ULL,
      0xC504DC9FF6A26B58ULL, 0xEA40AF2BD896D3A5ULL, 0x83842EC228CC6DEFULL, 0x581E2872A86C72A6ULL },
};

// Endomorphism table: (beta*x, y) for 1G through 15G
__constant__ const uint64_t G_TABLE_B[15][8] = {
    // phi(1G)
    { 0xA7BBA04400B88FCBULL, 0x872844067F15E98DULL, 0xAB0102B696902325ULL, 0xBCACE2E99DA01887ULL,
      0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL },
    // phi(2G)
    { 0x3E995B6ED89250E1ULL, 0xD2FAD8CCE43837EFULL, 0x4135EE7D59F87B33ULL, 0xC360A6D0B34CE6DFULL,
      0x236431A950CFE52AULL, 0xF7F632653266D0E1ULL, 0xA3C58419466CEAEEULL, 0x1AE168FEA63DC339ULL },
    // phi(3G)
    { 0xF7F0728C77206B2FULL, 0x8AF1E022C6DC8E1CULL, 0x8DCD8DCF2A28FA2FULL, 0xDF6EDF03731F9B4BULL,
      0x6CB9FD7584B8E672ULL, 0x6500A99934C2231BULL, 0x0FE337E62A37F356ULL, 0x388F7B0F632DE814ULL },
    // phi(4G)
    { 0x5BDE5B333B306100ULL, 0x714C30B5AB487127ULL, 0x5C45FAF8B90E324BULL, 0x1B77921F0D382907ULL,
      0xCFE97BDC47739922ULL, 0xD967AE33BFBDFE40ULL, 0x5642E2098EA51448ULL, 0x51ED993EA0D455B7ULL },
    // phi(5G)
    { 0x138C694695A83668ULL, 0xA045693EE0D097CCULL, 0xF79F54FBCCB94671ULL, 0x337B52E3ACDA49DFULL,
      0xDCA87D3AA6AC62D6ULL, 0xF788271BAB0D6840ULL, 0xD4DBA9DDA6C9C426ULL, 0xD8AC222636E5E3D6ULL },
    // phi(6G)
    { 0x47AAF28078F38045ULL, 0x86649D3E56A15A68ULL, 0x5E3AA731E3E8BED7ULL, 0xE63BCDD9AA535FC6ULL,
      0x3C870C36B075F297ULL, 0xDE80F0F6518FE4A0ULL, 0xF3BE96017F45C560ULL, 0xAE12777AACFBB620ULL },
    // phi(7G)
    { 0x3BC4686E4E53BC94ULL, 0x0D3B20E20FAF7AAAULL, 0xA4FEC4D1C095C06EULL, 0x13F26E754BEA0B77ULL,
      0xA5082628087264DAULL, 0xA813D0B813FDE7B5ULL, 0xA3178D6D861A54DBULL, 0x6AEBCA40BA255960ULL },
    // phi(8G)
    { 0x03E947742446CC73ULL, 0xB4FF771524257657ULL, 0xAA77840F29E24892ULL, 0x47AB650342D401A7ULL,
      0xB5DA2CB76CBDE904ULL, 0xC2E213D6BA5B7617ULL, 0x293D082A132D13B4ULL, 0x5C4DA8A741539949ULL },
    // phi(9G)
    { 0x20CD912E65953A52ULL, 0xB565CDF5EF6D44E1ULL, 0x7B6558AFEC58AB20ULL, 0x87B404037E44E819ULL,
      0x05CC262AC64F9C37ULL, 0xADD888A4375F8E0FULL, 0x64380971763B61E9ULL, 0xCC338921B0A7D9FDULL },
    // phi(10G)
    { 0xBDB3E957741AFE29ULL, 0xC1938D8E083762E4ULL, 0xA136EBB246813990ULL, 0x26CE269BF7A397B1ULL,
      0x3CBEE53B037368D7ULL, 0x6F794C2ED877A159ULL, 0xA3B6C7E693A24C69ULL, 0x893ABA425419BC27ULL },
    // phi(11G)
    { 0xC5FF4334BB209CE7ULL, 0x79859BB70B5FF620ULL, 0x8D897C41BEBF1A26ULL, 0x51F4D3D1171DAC1DULL,
      0x301D74C9C953C61BULL, 0x372DB1E2DFF9D6A8ULL, 0x0243DD56D7B7B365ULL, 0xD984A032EB6B5E19ULL },
    // phi(12G)
    { 0x4A3EB52C042295E5ULL, 0xF9482837C9535355ULL, 0xAC1548422EAC82ADULL, 0x88591BFD953AAC41ULL,
      0x6B051B13F4062327ULL, 0x79238C5DD9A86D52ULL, 0xA8B64537E17BD815ULL, 0xA9F34FFDC815E0D7ULL },
    // phi(13G)
    { 0x60AAEE6A475FB678ULL, 0x32907ED74A3D0562ULL, 0x07046C4578FC783BULL, 0xF14D58374BB890A2ULL,
      0x29B5CB52DB03ED81ULL, 0x3A1A06DA521FA91FULL, 0x758212EB65CDAF47ULL, 0x0AB0902E8D880A89ULL },
    // phi(14G)
    { 0x0E6AB7EE20A0B458ULL, 0x580656A627C529F6ULL, 0x1548F0DC87C37384ULL, 0x7B1252177810048AULL,
      0xC65F40D403A13F5BULL, 0x464279C27A3F95BCULL, 0x90F044E4A7B3D464ULL, 0xCAC2F6C4B54E8551ULL },
    // phi(15G)
    { 0x3AC0A40C71B1B3B4ULL, 0x05CC3BC9C1C0A639ULL, 0x0E1B4825512B6948ULL, 0x805F1105F5F9454AULL,
      0xC504DC9FF6A26B58ULL, 0xEA40AF2BD896D3A5ULL, 0x83842EC228CC6DEFULL, 0x581E2872A86C72A6ULL },
};

__device__ inline
void ct_generator_mul(const Scalar* k, JacobianPoint* r_out) {
    using namespace secp256k1::cuda;

    // GLV decompose
    CTGLVDecomposition glv = ct_glv_decompose(k);

    // Load precomputed tables from __constant__ memory
    // Entry 0: identity, entries 1-15: 1G through 15G
    constexpr int TABLE_SIZE = 16;
    CTAffinePoint table_a[TABLE_SIZE];
    CTAffinePoint table_b[TABLE_SIZE];

    // Identity at index 0
    field_set_zero(&table_a[0].x);
    field_set_zero(&table_a[0].y);
    table_a[0].infinity = ~(uint64_t)0;
    table_b[0] = table_a[0];

    // Load 1G..15G from __constant__
    for (int i = 0; i < 15; i++) {
        for (int j = 0; j < 4; j++) {
            table_a[i + 1].x.limbs[j] = G_TABLE_A[i][j];
            table_a[i + 1].y.limbs[j] = G_TABLE_A[i][j + 4];
            table_b[i + 1].x.limbs[j] = G_TABLE_B[i][j];
            table_b[i + 1].y.limbs[j] = G_TABLE_B[i][j + 4];
        }
        table_a[i + 1].infinity = 0;
        table_b[i + 1].infinity = 0;
    }

    // Conditionally negate tables based on GLV sign (skip identity at 0)
    for (int i = 1; i < TABLE_SIZE; i++) {
        ct_point_cneg_y(&table_a[i], glv.k1_neg);
        ct_point_cneg_y(&table_b[i], glv.k2_neg);
    }

    // Windowed main loop: 33 iterations of (4 dbl + 2 CT lookup + 2 mixed add)
    CTJacobianPoint result;
    ct_point_set_infinity(&result);

    for (int w = 32; w >= 0; --w) {
        ct_point_dbl(&result, &result);
        ct_point_dbl(&result, &result);
        ct_point_dbl(&result, &result);
        ct_point_dbl(&result, &result);

        int bit_pos  = w * 4;
        int limb_idx = bit_pos >> 6;
        int bit_off  = bit_pos & 63;
        int d1 = (int)((glv.k1.limbs[limb_idx] >> bit_off) & 0xF);
        int d2 = (int)((glv.k2.limbs[limb_idx] >> bit_off) & 0xF);

        CTAffinePoint entry1;
        ct_affine_table_lookup(table_a, TABLE_SIZE, d1, &entry1);
        CTJacobianPoint tmp;
        ct_point_add_mixed(&result, &entry1, &tmp);
        result = tmp;

        CTAffinePoint entry2;
        ct_affine_table_lookup(table_b, TABLE_SIZE, d2, &entry2);
        ct_point_add_mixed(&result, &entry2, &tmp);
        result = tmp;
    }

    *r_out = ct_point_to_jacobian(&result);
}

} // namespace ct
} // namespace cuda
} // namespace secp256k1
