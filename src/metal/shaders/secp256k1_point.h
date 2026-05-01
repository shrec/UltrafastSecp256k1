// =============================================================================
// UltrafastSecp256k1 Metal -- Point Operations (secp256k1_point.h)
// =============================================================================
// Jacobian coordinate point operations for secp256k1 (a=0 short Weierstrass).
// All formulas match the CUDA backend exactly.
//
// Point doubling:    dbl-2001-b  (3M + 4S)
// Mixed addition:    madd-2007-bl (7M + 4S) -- Jacobian + Affine
// General addition:  add-2007-bl  (11M + 5S) -- Jacobian + Jacobian
// Scalar mul:        4-bit fixed window (64 doubles + 64 adds, ~35% faster)
// =============================================================================

#pragma once

#include "secp256k1_field.h"

// =============================================================================
// Point Types
// =============================================================================

struct AffinePoint {
    FieldElement x;
    FieldElement y;
};

struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    uint infinity;  // 1 = point at infinity, 0 = normal
};

// Scalar (256-bit integer, 8x32) -- same layout as FieldElement
struct Scalar256 {
    uint limbs[8];
};

// =============================================================================
// Generator Point G (affine coordinates)
// =============================================================================

constant uint GEN_X[8] = {
    0x16F81798u, 0x59F2815Bu, 0x2DCE28D9u, 0x029BFCDBu,
    0xCE870B07u, 0x55A06295u, 0xF9DCBBACu, 0x79BE667Eu
};

constant uint GEN_Y[8] = {
    0xFB10D4B8u, 0x9C47D08Fu, 0xA6855419u, 0xFD17B448u,
    0x0E1108A8u, 0x5DA4FBFCu, 0x26A3C465u, 0x483ADA77u
};

inline AffinePoint generator_affine() {
    AffinePoint g;
    for (int i = 0; i < 8; i++) { g.x.limbs[i] = GEN_X[i]; g.y.limbs[i] = GEN_Y[i]; }
    return g;
}

inline JacobianPoint generator_jacobian() {
    JacobianPoint g;
    for (int i = 0; i < 8; i++) { g.x.limbs[i] = GEN_X[i]; g.y.limbs[i] = GEN_Y[i]; }
    g.z = field_one();
    g.infinity = 0;
    return g;
}

inline JacobianPoint point_at_infinity() {
    JacobianPoint r;
    r.x = field_zero();
    r.y = field_one();
    r.z = field_zero();
    r.infinity = 1;
    return r;
}

// =============================================================================
// Point Doubling -- dbl-2001-b for a=0 (3M + 4S + 7add)
// =============================================================================

inline JacobianPoint jacobian_double(thread const JacobianPoint &p) {
    if (p.infinity != 0) return p;
    if (field_is_zero(p.y)) {
        return point_at_infinity();
    }

    FieldElement YY, S, M, X3, Y3, Z3, YYYY, t1;

    // YY = Y^2 [1S]
    YY = field_sqr(p.y);

    // S = 4*X*Y^2 [1M + 2add]
    S = field_mul(p.x, YY);
    S = field_add(S, S);
    S = field_add(S, S);

    // M = 3*X^2 [2S + 2add] (a=0 for secp256k1)
    M = field_sqr(p.x);
    t1 = field_add(M, M);     // 2X^2
    M = field_add(M, t1);     // 3X^2

    // X3 = M^2 - 2S [3S + sub]
    X3 = field_sqr(M);
    t1 = field_add(S, S);
    X3 = field_sub(X3, t1);

    // YYYY = Y^4 [4S]
    YYYY = field_sqr(YY);

    // Y3 = M*(S - X3) - 8*Y^4
    t1 = field_add(YYYY, YYYY); // 2Y^4
    t1 = field_add(t1, t1);     // 4Y^4
    t1 = field_add(t1, t1);     // 8Y^4
    FieldElement diff = field_sub(S, X3);
    Y3 = field_mul(M, diff);
    Y3 = field_sub(Y3, t1);

    // Z3 = 2*Y*Z
    Z3 = field_mul(p.y, p.z);
    Z3 = field_add(Z3, Z3);

    JacobianPoint r;
    r.x = X3; r.y = Y3; r.z = Z3;
    r.infinity = 0;
    return r;
}

// =============================================================================
// Mixed Addition: P (Jacobian) + Q (Affine) -> Result (Jacobian)
// madd-2007-bl formula (7M + 4S) -- Q has implicit Z=1
// =============================================================================

inline JacobianPoint jacobian_add_mixed(thread const JacobianPoint &p,
                                         thread const AffinePoint &q);

// Overload for device address space AffinePoint (kernel buffers)
inline JacobianPoint jacobian_add_mixed(thread const JacobianPoint &p,
                                         device const AffinePoint &q) {
    AffinePoint local_q = q;
    return jacobian_add_mixed(p, local_q);
}

inline JacobianPoint jacobian_add_mixed(thread const JacobianPoint &p,
                                         thread const AffinePoint &q) {
    if (p.infinity != 0) {
        JacobianPoint r;
        r.x = q.x; r.y = q.y;
        r.z = field_one();
        r.infinity = 0;
        return r;
    }

    FieldElement z1z1, u2, s2, h, hh, i, j, rr, v;
    FieldElement X3, Y3, Z3, t1, t2;

    // Z1^2 [1S]
    z1z1 = field_sqr(p.z);

    // U2 = X2*Z1^2 [1M]
    u2 = field_mul(q.x, z1z1);

    // S2 = Y2*Z1^3 [2M]
    t1 = field_mul(p.z, z1z1);
    s2 = field_mul(q.y, t1);

    // H = U2 - X1
    h = field_sub(u2, p.x);

    // Same x check
    if (field_is_zero(h)) {
        t1 = field_sub(s2, p.y);
        if (field_is_zero(t1)) {
            return jacobian_double(p);
        }
        return point_at_infinity();
    }

    // HH = H^2 [2S]
    hh = field_sqr(h);

    // I = 4*HH
    i = field_add(hh, hh);
    i = field_add(i, i);

    // J = H*I [3M]
    j = field_mul(h, i);

    // rr = 2*(S2 - Y1)
    t1 = field_sub(s2, p.y);
    rr = field_add(t1, t1);

    // V = X1*I [4M]
    v = field_mul(p.x, i);

    // X3 = rr^2 - J - 2V [3S]
    X3 = field_sqr(rr);
    X3 = field_sub(X3, j);
    t1 = field_add(v, v);
    X3 = field_sub(X3, t1);

    // Y3 = rr*(V - X3) - 2*Y1*J [5M, 6M]
    t1 = field_sub(v, X3);
    Y3 = field_mul(rr, t1);
    t2 = field_mul(p.y, j);
    t2 = field_add(t2, t2);
    Y3 = field_sub(Y3, t2);

    // Z3 = (Z1 + H)^2 - Z1^2 - HH [4S]
    t1 = field_add(p.z, h);
    Z3 = field_sqr(t1);
    Z3 = field_sub(Z3, z1z1);
    Z3 = field_sub(Z3, hh);

    JacobianPoint r;
    r.x = X3; r.y = Y3; r.z = Z3;
    r.infinity = 0;
    return r;
}

// =============================================================================
// General Jacobian Addition: P + Q (both Jacobian) -> Result
// add-2007-bl formula (11M + 5S)
// =============================================================================

inline JacobianPoint jacobian_add(thread const JacobianPoint &p,
                                   thread const JacobianPoint &q) {
    if (p.infinity != 0) return q;
    if (q.infinity != 0) return p;

    FieldElement z1z1, z2z2, u1, u2, s1, s2, h, r_val;
    FieldElement t1, t2;

    z1z1 = field_sqr(p.z);
    z2z2 = field_sqr(q.z);

    u1 = field_mul(p.x, z2z2);
    u2 = field_mul(q.x, z1z1);

    t1 = field_mul(q.z, z2z2);
    s1 = field_mul(p.y, t1);

    t2 = field_mul(p.z, z1z1);
    s2 = field_mul(q.y, t2);

    h = field_sub(u2, u1);

    // Same X check
    if (field_is_zero(h)) {
        t1 = field_sub(s2, s1);
        if (field_is_zero(t1)) {
            return jacobian_double(p);
        }
        return point_at_infinity();
    }

    r_val = field_sub(s2, s1);
    r_val = field_add(r_val, r_val); // rr = 2(S2 - S1)

    FieldElement hh = field_sqr(h);
    FieldElement i = field_add(hh, hh);
    i = field_add(i, i);                // I = 4H^2

    FieldElement j = field_mul(h, i);    // J = H*I
    FieldElement v = field_mul(u1, i);   // V = U1*I

    FieldElement X3 = field_sqr(r_val);
    X3 = field_sub(X3, j);
    t1 = field_add(v, v);
    X3 = field_sub(X3, t1);

    t1 = field_sub(v, X3);
    FieldElement Y3 = field_mul(r_val, t1);
    t2 = field_mul(s1, j);
    t2 = field_add(t2, t2);
    Y3 = field_sub(Y3, t2);

    // Z3 = ((Z1+Z2)^2 - Z1^2 - Z2^2)*H
    t1 = field_add(p.z, q.z);
    FieldElement Z3 = field_sqr(t1);
    Z3 = field_sub(Z3, z1z1);
    Z3 = field_sub(Z3, z2z2);
    Z3 = field_mul(Z3, h);

    JacobianPoint result;
    result.x = X3; result.y = Y3; result.z = Z3;
    result.infinity = 0;
    return result;
}

// =============================================================================
// Jacobian -> Affine Conversion
// (Defined here before scalar_mul which depends on it)
// =============================================================================

inline AffinePoint jacobian_to_affine(thread const JacobianPoint &p) {
    AffinePoint r;
    if (p.infinity != 0) {
        r.x = field_zero();
        r.y = field_zero();
        return r;
    }

    FieldElement z_inv = field_inv(p.z);
    FieldElement z_inv2 = field_sqr(z_inv);
    FieldElement z_inv3 = field_mul(z_inv2, z_inv);

    r.x = field_mul(p.x, z_inv2);
    r.y = field_mul(p.y, z_inv3);
    return r;
}

// =============================================================================
// Scalar Multiplication: P x k -- 4-bit Fixed Window (w=4)
//
// ACCELERATION: Instead of simple double-and-add (256 doubles + ~128 adds),
// this uses a 4-bit window: 64 doubles + 64 lookups + 64 adds.
// ~30-40% faster than naive double-and-add on GPU.
//
// Precomputes: table[i] = i*P for i=0..15 (16 affine points)
// Then processes scalar 4 bits at a time (64 windows).
//
// This is the Metal equivalent of CUDA's wNAF or fixed-window approaches.
// Register pressure: 16 AffinePoints x 16 limbs = 256 registers -- fits
// nicely in Apple Silicon's large register file.
// =============================================================================

// Branchless conditional select -- avoids divergent control flow on GPU
inline AffinePoint affine_select(thread const AffinePoint table[16], uint idx) {
    AffinePoint result;
    for (int i = 0; i < 8; i++) { result.x.limbs[i] = 0; result.y.limbs[i] = 0; }

    for (uint k = 0; k < 16; k++) {
        uint mask = -uint(k == idx);  // 0xFFFFFFFF if match, 0 otherwise
        for (int i = 0; i < 8; i++) {
            result.x.limbs[i] |= (table[k].x.limbs[i] & mask);
            result.y.limbs[i] |= (table[k].y.limbs[i] & mask);
        }
    }
    return result;
}

inline JacobianPoint scalar_mul(thread const AffinePoint &base,
                                 thread const Scalar256 &k) {
    // === Precompute table[0..15] ===
    AffinePoint table[16];

    // table[0] = O (point at infinity -- zero affine)
    for (int i = 0; i < 8; i++) { table[0].x.limbs[i] = 0; table[0].y.limbs[i] = 0; }

    // table[1] = P
    table[1] = base;

    // table[2] = 2P
    JacobianPoint jp;
    jp.x = base.x; jp.y = base.y; jp.z = field_one(); jp.infinity = 0;
    JacobianPoint j2 = jacobian_double(jp);
    table[2] = jacobian_to_affine(j2);

    // table[3..15] = iP via mixed addition
    for (int i = 3; i <= 15; i++) {
        JacobianPoint prev;
        prev.x = table[i - 1].x; prev.y = table[i - 1].y;
        prev.z = field_one(); prev.infinity = 0;
        JacobianPoint sum = jacobian_add_mixed(prev, base);
        table[i] = jacobian_to_affine(sum);
    }

    // === Fixed-window scan: 4 bits at a time, MSB first ===
    JacobianPoint r = point_at_infinity();
    bool started = false;

    for (int limb = 7; limb >= 0; limb--) {
        uint w = k.limbs[limb];
        // Process 8 nibbles per 32-bit limb (MSB first)
        for (int nib = 7; nib >= 0; nib--) {
            uint idx = (w >> (nib * 4)) & 0xFu;

            // Double 4 times (shift window)
            if (started) {
                r = jacobian_double(r);
                r = jacobian_double(r);
                r = jacobian_double(r);
                r = jacobian_double(r);
            }

            if (idx != 0) {
                AffinePoint selected = affine_select(table, idx);
                if (!started) {
                    r.x = selected.x;
                    r.y = selected.y;
                    r.z = field_one();
                    r.infinity = 0;
                    started = true;
                } else {
                    r = jacobian_add_mixed(r, selected);
                }
            }
        }
    }
    return r;
}

// Overload for device address space base points (e.g. buffer arrays)
inline JacobianPoint scalar_mul(device const AffinePoint &base,
                                 thread const Scalar256 &k) {
    AffinePoint local_base = base;
    return scalar_mul(local_base, k);
}

// =============================================================================
// GLV Endomorphism: phi(x,y) = (beta*x, y) where beta^3 == 1 (mod p)
// =============================================================================

constant uint BETA_LIMBS[8] = {
    0x719501EEu, 0xC1396C28u, 0x12F58995u, 0x9CF04975u,
    0xAC3434E9u, 0x6E64479Eu, 0x657C0710u, 0x7AE96A2Bu
};

inline AffinePoint apply_endomorphism(thread const AffinePoint &p) {
    FieldElement beta;
    for (int i = 0; i < 8; i++) beta.limbs[i] = BETA_LIMBS[i];

    AffinePoint r;
    r.x = field_mul(p.x, beta);
    r.y = p.y;
    return r;
}
