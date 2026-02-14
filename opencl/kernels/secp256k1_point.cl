// =============================================================================
// UltrafastSecp256k1 OpenCL Kernels - Point Operations
// =============================================================================
// Elliptic curve point operations on secp256k1: y² = x³ + 7
// Jacobian coordinates for efficient operations
// =============================================================================

// Include field arithmetic
#include "secp256k1_field.cl"

// =============================================================================
// Curve Constants
// =============================================================================

// Generator point G (affine coordinates)
// Gx = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
// Gy = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8

#define SECP256K1_GX0 0x59F2815B16F81798UL
#define SECP256K1_GX1 0x029BFCDB2DCE28D9UL
#define SECP256K1_GX2 0x55A06295CE870B07UL
#define SECP256K1_GX3 0x79BE667EF9DCBBACUL

#define SECP256K1_GY0 0x9C47D08FFB10D4B8UL
#define SECP256K1_GY1 0xFD17B448A6855419UL
#define SECP256K1_GY2 0x5DA4FBFC0E1108A8UL
#define SECP256K1_GY3 0x483ADA7726A3C465UL

// Curve order n
#define SECP256K1_N0 0xBFD25E8CD0364141UL
#define SECP256K1_N1 0xBAAEDCE6AF48A03BUL
#define SECP256K1_N2 0xFFFFFFFFFFFFFFFEUL
#define SECP256K1_N3 0xFFFFFFFFFFFFFFFFUL

// =============================================================================
// Point Types
// =============================================================================

typedef struct {
    FieldElement x;
    FieldElement y;
} AffinePoint;

typedef struct {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    uint infinity;  // 1 if point at infinity
} JacobianPoint;

typedef struct {
    ulong limbs[4];
} Scalar;

// =============================================================================
// Point Utilities
// =============================================================================

inline void point_set_infinity(JacobianPoint* p) {
    p->x.limbs[0] = 0; p->x.limbs[1] = 0; p->x.limbs[2] = 0; p->x.limbs[3] = 0;
    p->y.limbs[0] = 1; p->y.limbs[1] = 0; p->y.limbs[2] = 0; p->y.limbs[3] = 0;
    p->z.limbs[0] = 0; p->z.limbs[1] = 0; p->z.limbs[2] = 0; p->z.limbs[3] = 0;
    p->infinity = 1;
}

inline int point_is_infinity(const JacobianPoint* p) {
    return p->infinity ||
           ((p->z.limbs[0] | p->z.limbs[1] | p->z.limbs[2] | p->z.limbs[3]) == 0);
}

inline void point_from_affine(JacobianPoint* j, const AffinePoint* a) {
    j->x = a->x;
    j->y = a->y;
    j->z.limbs[0] = 1; j->z.limbs[1] = 0; j->z.limbs[2] = 0; j->z.limbs[3] = 0;
    j->infinity = 0;
}

inline void get_generator(AffinePoint* g) {
    g->x.limbs[0] = SECP256K1_GX0;
    g->x.limbs[1] = SECP256K1_GX1;
    g->x.limbs[2] = SECP256K1_GX2;
    g->x.limbs[3] = SECP256K1_GX3;

    g->y.limbs[0] = SECP256K1_GY0;
    g->y.limbs[1] = SECP256K1_GY1;
    g->y.limbs[2] = SECP256K1_GY2;
    g->y.limbs[3] = SECP256K1_GY3;
}

// =============================================================================
// Point Doubling: R = 2*P (Jacobian coordinates)
// Using standard doubling formula for a = 0 curves (secp256k1)
// =============================================================================

inline void point_double_impl(JacobianPoint* r, const JacobianPoint* p) {
    if (point_is_infinity(p)) {
        point_set_infinity(r);
        return;
    }

    // Check if Y = 0 (point of order 2, but secp256k1 doesn't have one)
    if ((p->y.limbs[0] | p->y.limbs[1] | p->y.limbs[2] | p->y.limbs[3]) == 0) {
        point_set_infinity(r);
        return;
    }

    FieldElement S, M, X3, Y3, Z3, YY, YYYY, ZZ, t1, t2;

    // S = 4*X*Y^2
    field_sqr_impl(&YY, &p->y);           // YY = Y^2
    field_mul_impl(&S, &p->x, &YY);       // S = X * Y^2
    field_add_impl(&S, &S, &S);           // S = 2*X*Y^2
    field_add_impl(&S, &S, &S);           // S = 4*X*Y^2

    // M = 3*X^2 (since a=0 for secp256k1)
    field_sqr_impl(&M, &p->x);            // M = X^2
    field_add_impl(&t1, &M, &M);          // t1 = 2*X^2
    field_add_impl(&M, &M, &t1);          // M = 3*X^2

    // X3 = M^2 - 2*S
    field_sqr_impl(&X3, &M);              // X3 = M^2
    field_add_impl(&t1, &S, &S);          // t1 = 2*S
    field_sub_impl(&X3, &X3, &t1);        // X3 = M^2 - 2*S

    // Y3 = M*(S - X3) - 8*Y^4
    field_sqr_impl(&YYYY, &YY);           // YYYY = Y^4
    field_add_impl(&t1, &YYYY, &YYYY);    // t1 = 2*Y^4
    field_add_impl(&t1, &t1, &t1);        // t1 = 4*Y^4
    field_add_impl(&t1, &t1, &t1);        // t1 = 8*Y^4
    field_sub_impl(&t2, &S, &X3);         // t2 = S - X3
    field_mul_impl(&Y3, &M, &t2);         // Y3 = M*(S - X3)
    field_sub_impl(&Y3, &Y3, &t1);        // Y3 = M*(S - X3) - 8*Y^4

    // Z3 = 2*Y*Z
    field_mul_impl(&Z3, &p->y, &p->z);    // Z3 = Y*Z
    field_add_impl(&Z3, &Z3, &Z3);        // Z3 = 2*Y*Z

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = 0;
}

// =============================================================================
// Point Addition: R = P + Q (Jacobian + Jacobian)
// Complete addition formula
// =============================================================================

inline void point_add_impl(JacobianPoint* r, const JacobianPoint* p, const JacobianPoint* q) {
    // Handle infinity cases
    if (point_is_infinity(p)) {
        *r = *q;
        return;
    }
    if (point_is_infinity(q)) {
        *r = *p;
        return;
    }

    FieldElement U1, U2, S1, S2, H, I, J, rr, V, X3, Y3, Z3;
    FieldElement Z1Z1, Z2Z2, t1, t2;

    // Z1Z1 = Z1^2
    field_sqr_impl(&Z1Z1, &p->z);

    // Z2Z2 = Z2^2
    field_sqr_impl(&Z2Z2, &q->z);

    // U1 = X1*Z2Z2
    field_mul_impl(&U1, &p->x, &Z2Z2);

    // U2 = X2*Z1Z1
    field_mul_impl(&U2, &q->x, &Z1Z1);

    // S1 = Y1*Z2*Z2Z2
    field_mul_impl(&t1, &p->y, &q->z);
    field_mul_impl(&S1, &t1, &Z2Z2);

    // S2 = Y2*Z1*Z1Z1
    field_mul_impl(&t1, &q->y, &p->z);
    field_mul_impl(&S2, &t1, &Z1Z1);

    // H = U2 - U1
    field_sub_impl(&H, &U2, &U1);

    // Check if H = 0 (points have same X coordinate)
    if ((H.limbs[0] | H.limbs[1] | H.limbs[2] | H.limbs[3]) == 0) {
        // Check if S1 == S2 (same point, do doubling)
        field_sub_impl(&t1, &S2, &S1);
        if ((t1.limbs[0] | t1.limbs[1] | t1.limbs[2] | t1.limbs[3]) == 0) {
            point_double_impl(r, p);
            return;
        }
        // Points are negatives, result is infinity
        point_set_infinity(r);
        return;
    }

    // I = (2*H)^2
    field_add_impl(&I, &H, &H);           // I = 2*H
    field_sqr_impl(&I, &I);               // I = (2*H)^2

    // J = H*I
    field_mul_impl(&J, &H, &I);

    // r = 2*(S2 - S1)
    field_sub_impl(&rr, &S2, &S1);
    field_add_impl(&rr, &rr, &rr);

    // V = U1*I
    field_mul_impl(&V, &U1, &I);

    // X3 = r^2 - J - 2*V
    field_sqr_impl(&X3, &rr);
    field_sub_impl(&X3, &X3, &J);
    field_add_impl(&t1, &V, &V);
    field_sub_impl(&X3, &X3, &t1);

    // Y3 = r*(V - X3) - 2*S1*J
    field_sub_impl(&t1, &V, &X3);
    field_mul_impl(&Y3, &rr, &t1);
    field_mul_impl(&t2, &S1, &J);
    field_add_impl(&t2, &t2, &t2);
    field_sub_impl(&Y3, &Y3, &t2);

    // Z3 = ((Z1 + Z2)^2 - Z1Z1 - Z2Z2) * H
    field_add_impl(&t1, &p->z, &q->z);
    field_sqr_impl(&t1, &t1);
    field_sub_impl(&t1, &t1, &Z1Z1);
    field_sub_impl(&t1, &t1, &Z2Z2);
    field_mul_impl(&Z3, &t1, &H);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = 0;
}

// =============================================================================
// Mixed Addition: R = P + Q (Jacobian + Affine)
// More efficient when one point is affine (Z = 1)
// =============================================================================

inline void point_add_mixed_impl(JacobianPoint* r, const JacobianPoint* p, const AffinePoint* q) {
    if (point_is_infinity(p)) {
        point_from_affine(r, q);
        return;
    }

    FieldElement Z1Z1, U2, S2, H, HH, I, J, rr, V, X3, Y3, Z3, t1, t2;

    // Z1Z1 = Z1^2
    field_sqr_impl(&Z1Z1, &p->z);

    // U2 = X2*Z1Z1 (U1 = X1 since Z2 = 1)
    field_mul_impl(&U2, &q->x, &Z1Z1);

    // S2 = Y2*Z1*Z1Z1 (S1 = Y1 since Z2 = 1)
    field_mul_impl(&t1, &q->y, &p->z);
    field_mul_impl(&S2, &t1, &Z1Z1);

    // H = U2 - X1
    field_sub_impl(&H, &U2, &p->x);

    // Check if points are same or negatives
    if ((H.limbs[0] | H.limbs[1] | H.limbs[2] | H.limbs[3]) == 0) {
        field_sub_impl(&t1, &S2, &p->y);
        if ((t1.limbs[0] | t1.limbs[1] | t1.limbs[2] | t1.limbs[3]) == 0) {
            point_double_impl(r, p);
            return;
        }
        point_set_infinity(r);
        return;
    }

    // HH = H^2
    field_sqr_impl(&HH, &H);

    // I = 4*HH
    field_add_impl(&I, &HH, &HH);
    field_add_impl(&I, &I, &I);

    // J = H*I
    field_mul_impl(&J, &H, &I);

    // r = 2*(S2 - Y1)
    field_sub_impl(&rr, &S2, &p->y);
    field_add_impl(&rr, &rr, &rr);

    // V = X1*I
    field_mul_impl(&V, &p->x, &I);

    // X3 = r^2 - J - 2*V
    field_sqr_impl(&X3, &rr);
    field_sub_impl(&X3, &X3, &J);
    field_add_impl(&t1, &V, &V);
    field_sub_impl(&X3, &X3, &t1);

    // Y3 = r*(V - X3) - 2*Y1*J
    field_sub_impl(&t1, &V, &X3);
    field_mul_impl(&Y3, &rr, &t1);
    field_mul_impl(&t2, &p->y, &J);
    field_add_impl(&t2, &t2, &t2);
    field_sub_impl(&Y3, &Y3, &t2);

    // Z3 = (Z1 + H)^2 - Z1Z1 - HH
    field_add_impl(&t1, &p->z, &H);
    field_sqr_impl(&Z3, &t1);
    field_sub_impl(&Z3, &Z3, &Z1Z1);
    field_sub_impl(&Z3, &Z3, &HH);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = 0;
}

// =============================================================================
// Scalar Multiplication: R = k * P
// Double-and-add algorithm (left-to-right)
// =============================================================================

inline void scalar_mul_impl(JacobianPoint* r, const Scalar* k, const AffinePoint* p) {
    // Check for zero scalar
    if ((k->limbs[0] | k->limbs[1] | k->limbs[2] | k->limbs[3]) == 0) {
        point_set_infinity(r);
        return;
    }

    JacobianPoint R;
    point_set_infinity(&R);

    JacobianPoint P;
    point_from_affine(&P, p);

    // Find highest set bit
    int start_limb = 3;
    int start_bit = 63;

    while (start_limb >= 0) {
        if (k->limbs[start_limb] != 0) {
            // Find highest bit in this limb
            ulong v = k->limbs[start_limb];
            start_bit = 63;
            while (start_bit >= 0 && !((v >> start_bit) & 1)) {
                start_bit--;
            }
            break;
        }
        start_limb--;
    }

    // Double-and-add from highest to lowest bit
    for (int limb = start_limb; limb >= 0; limb--) {
        int end_bit = (limb == start_limb) ? start_bit : 63;
        for (int bit = end_bit; bit >= 0; bit--) {
            point_double_impl(&R, &R);

            if ((k->limbs[limb] >> bit) & 1) {
                point_add_mixed_impl(&R, &R, p);
            }
        }
    }

    *r = R;
}

// =============================================================================
// Scalar Multiplication with Generator: R = k * G
// =============================================================================

inline void scalar_mul_generator_impl(JacobianPoint* r, const Scalar* k) {
    AffinePoint G;
    get_generator(&G);
    scalar_mul_impl(r, k, &G);
}

// =============================================================================
// OpenCL Kernels - Point Operations
// =============================================================================

__kernel void point_double(
    __global const JacobianPoint* points,
    __global JacobianPoint* results,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    JacobianPoint p_local = points[gid];
    JacobianPoint r;
    point_double_impl(&r, &p_local);
    results[gid] = r;
}

__kernel void point_add(
    __global const JacobianPoint* p,
    __global const JacobianPoint* q,
    __global JacobianPoint* results,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    JacobianPoint p_local = p[gid];
    JacobianPoint q_local = q[gid];
    JacobianPoint r;
    point_add_impl(&r, &p_local, &q_local);
    results[gid] = r;
}

__kernel void scalar_mul(
    __global const Scalar* scalars,
    __global const AffinePoint* points,
    __global JacobianPoint* results,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    Scalar k_local = scalars[gid];
    AffinePoint p_local = points[gid];
    JacobianPoint r;
    scalar_mul_impl(&r, &k_local, &p_local);
    results[gid] = r;
}

__kernel void scalar_mul_generator(
    __global const Scalar* scalars,
    __global JacobianPoint* results,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    // Copy from global to private memory
    Scalar k_local = scalars[gid];
    JacobianPoint r;
    scalar_mul_generator_impl(&r, &k_local);
    results[gid] = r;
}

