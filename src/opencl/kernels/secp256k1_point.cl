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
    uint pad[7];    // Match host alignas(128) layout — sizeof = 128 bytes
} JacobianPoint;

typedef struct {
    ulong limbs[4];
} Scalar;

// =============================================================================
// Point Utilities
// =============================================================================

FORCE_INLINE void point_set_infinity(JacobianPoint* p) {
    p->x.limbs[0] = 0; p->x.limbs[1] = 0; p->x.limbs[2] = 0; p->x.limbs[3] = 0;
    p->y.limbs[0] = 1; p->y.limbs[1] = 0; p->y.limbs[2] = 0; p->y.limbs[3] = 0;
    p->z.limbs[0] = 0; p->z.limbs[1] = 0; p->z.limbs[2] = 0; p->z.limbs[3] = 0;
    p->infinity = 1;
}

FORCE_INLINE int point_is_infinity(const JacobianPoint* p) {
    return p->infinity ||
           ((p->z.limbs[0] | p->z.limbs[1] | p->z.limbs[2] | p->z.limbs[3]) == 0);
}

FORCE_INLINE void point_from_affine(JacobianPoint* j, const AffinePoint* a) {
    j->x = a->x;
    j->y = a->y;
    j->z.limbs[0] = 1; j->z.limbs[1] = 0; j->z.limbs[2] = 0; j->z.limbs[3] = 0;
    j->infinity = 0;
}

FORCE_INLINE void get_generator(AffinePoint* g) {
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

FORCE_INLINE void point_double_impl(JacobianPoint* r, const JacobianPoint* p) {
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

// Unchecked doubling: skips infinity and Y==0 checks.
// Precondition: p is a valid, non-infinity point with Y != 0.
FORCE_INLINE void point_double_unchecked(JacobianPoint* r, const JacobianPoint* p) {
    FieldElement S, M, X3, Y3, Z3, YY, YYYY, t1, t2;

    field_sqr_impl(&YY, &p->y);
    field_mul_impl(&S, &p->x, &YY);
    field_add_impl(&S, &S, &S);
    field_add_impl(&S, &S, &S);

    field_sqr_impl(&M, &p->x);
    field_add_impl(&t1, &M, &M);
    field_add_impl(&M, &M, &t1);

    field_sqr_impl(&X3, &M);
    field_add_impl(&t1, &S, &S);
    field_sub_impl(&X3, &X3, &t1);

    field_sqr_impl(&YYYY, &YY);
    field_add_impl(&t1, &YYYY, &YYYY);
    field_add_impl(&t1, &t1, &t1);
    field_add_impl(&t1, &t1, &t1);
    field_sub_impl(&t2, &S, &X3);
    field_mul_impl(&Y3, &M, &t2);
    field_sub_impl(&Y3, &Y3, &t1);

    field_mul_impl(&Z3, &p->y, &p->z);
    field_add_impl(&Z3, &Z3, &Z3);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
    r->infinity = 0;
}

// Unchecked mixed addition: skips p->infinity check.
// Precondition: p is a valid, non-infinity Jacobian point.
// Keeps the H==0 check for algebraic completeness.
FORCE_INLINE void point_add_mixed_unchecked(JacobianPoint* r, const JacobianPoint* p, const AffinePoint* q) {
    FieldElement Z1Z1, U2, S2, H, HH, I, J, rr, V, X3, Y3, Z3, t1, t2;

    field_sqr_impl(&Z1Z1, &p->z);
    field_mul_impl(&U2, &q->x, &Z1Z1);
    field_mul_impl(&t1, &q->y, &p->z);
    field_mul_impl(&S2, &t1, &Z1Z1);
    field_sub_impl(&H, &U2, &p->x);

    if ((H.limbs[0] | H.limbs[1] | H.limbs[2] | H.limbs[3]) == 0) {
        field_sub_impl(&t1, &S2, &p->y);
        if ((t1.limbs[0] | t1.limbs[1] | t1.limbs[2] | t1.limbs[3]) == 0) {
            point_double_unchecked(r, p);
            return;
        }
        point_set_infinity(r);
        return;
    }

    field_sqr_impl(&HH, &H);
    field_add_impl(&I, &HH, &HH);
    field_add_impl(&I, &I, &I);
    field_mul_impl(&J, &H, &I);
    field_sub_impl(&rr, &S2, &p->y);
    field_add_impl(&rr, &rr, &rr);
    field_mul_impl(&V, &p->x, &I);

    field_sqr_impl(&X3, &rr);
    field_sub_impl(&X3, &X3, &J);
    field_add_impl(&t1, &V, &V);
    field_sub_impl(&X3, &X3, &t1);

    field_sub_impl(&t1, &V, &X3);
    field_mul_impl(&Y3, &rr, &t1);
    field_mul_impl(&t2, &p->y, &J);
    field_add_impl(&t2, &t2, &t2);
    field_sub_impl(&Y3, &Y3, &t2);

    field_add_impl(&t1, &p->z, &H);
    field_sqr_impl(&Z3, &t1);
    field_sub_impl(&Z3, &Z3, &Z1Z1);
    field_sub_impl(&Z3, &Z3, &HH);

    r->x = X3;
    r->y = Y3;
    r->z = Z3;
}

// =============================================================================
// Point Addition: R = P + Q (Jacobian + Jacobian)
// Complete addition formula
// =============================================================================

FORCE_INLINE void point_add_impl(JacobianPoint* r, const JacobianPoint* p, const JacobianPoint* q) {
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

FORCE_INLINE void point_add_mixed_impl(JacobianPoint* r, const JacobianPoint* p, const AffinePoint* q) {
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

// Mixed Jacobian+affine addition with H output for batch inversion.
// h_out receives H = U2 - X1 (the Z-coordinate ratio).
// For degenerate cases (infinity, doubling, negation), h_out = ONE.
FORCE_INLINE void point_add_mixed_h_impl(JacobianPoint* r, const JacobianPoint* p,
                                   const AffinePoint* q, FieldElement* h_out) {
    h_out->limbs[0] = 1UL; h_out->limbs[1] = 0; h_out->limbs[2] = 0; h_out->limbs[3] = 0;

    if (point_is_infinity(p)) {
        point_from_affine(r, q);
        return;
    }

    FieldElement Z1Z1, U2, S2, H, HH, I, J, rr, V, X3, Y3, Z3, t1, t2;

    field_sqr_impl(&Z1Z1, &p->z);
    field_mul_impl(&U2, &q->x, &Z1Z1);
    field_mul_impl(&t1, &q->y, &p->z);
    field_mul_impl(&S2, &t1, &Z1Z1);

    field_sub_impl(&H, &U2, &p->x);

    if ((H.limbs[0] | H.limbs[1] | H.limbs[2] | H.limbs[3]) == 0) {
        field_sub_impl(&t1, &S2, &p->y);
        if ((t1.limbs[0] | t1.limbs[1] | t1.limbs[2] | t1.limbs[3]) == 0) {
            point_double_impl(r, p);
            return;
        }
        point_set_infinity(r);
        return;
    }

    // Z3 = (Z1+H)^2 - Z1Z1 - HH = 2*Z1*H, so Z-ratio is 2*H
    field_add_impl(h_out, &H, &H);

    field_sqr_impl(&HH, &H);
    field_add_impl(&I, &HH, &HH);
    field_add_impl(&I, &I, &I);
    field_mul_impl(&J, &H, &I);
    field_sub_impl(&rr, &S2, &p->y);
    field_add_impl(&rr, &rr, &rr);
    field_mul_impl(&V, &p->x, &I);

    field_sqr_impl(&X3, &rr);
    field_sub_impl(&X3, &X3, &J);
    field_add_impl(&t1, &V, &V);
    field_sub_impl(&X3, &X3, &t1);

    field_sub_impl(&t1, &V, &X3);
    field_mul_impl(&Y3, &rr, &t1);
    field_mul_impl(&t2, &p->y, &J);
    field_add_impl(&t2, &t2, &t2);
    field_sub_impl(&Y3, &Y3, &t2);

    field_add_impl(&t1, &p->z, &H);
    field_sqr_impl(&Z3, &t1);
    field_sub_impl(&Z3, &Z3, &Z1Z1);
    field_sub_impl(&Z3, &Z3, &HH);

    r->x = X3; r->y = Y3; r->z = Z3; r->infinity = 0;
}

// =============================================================================
// Scalar Utilities for wNAF
// =============================================================================

FORCE_INLINE int scalar_is_zero(const Scalar* k) {
    return (k->limbs[0] | k->limbs[1] | k->limbs[2] | k->limbs[3]) == 0;
}

FORCE_INLINE int scalar_bit(const Scalar* k, int pos) {
    int limb = pos / 64;
    int bit = pos % 64;
    return (int)((k->limbs[limb] >> bit) & 1UL);
}

FORCE_INLINE void scalar_sub_u64(Scalar* a, ulong val, Scalar* r) {
    *r = *a;
    ulong old = r->limbs[0];
    r->limbs[0] -= val;
    if (r->limbs[0] > old) { // borrow
        for (int i = 1; i < 4; i++) {
            r->limbs[i] -= 1;
            if (r->limbs[i] != ~0UL) break; // no further borrow
        }
    }
}

FORCE_INLINE void scalar_add_u64(Scalar* a, ulong val, Scalar* r) {
    *r = *a;
    ulong old = r->limbs[0];
    r->limbs[0] += val;
    if (r->limbs[0] < old) { // carry
        for (int i = 1; i < 4; i++) {
            r->limbs[i] += 1;
            if (r->limbs[i] != 0) break; // no further carry
        }
    }
}

// Convert scalar to wNAF representation (window width 5)
// Returns length of wNAF representation
static inline int scalar_to_wnaf(const Scalar* k, int wnaf[260]) {
    Scalar temp = *k;
    int len = 0;
    const int window_size = 32;   // 2^5
    const int window_mask = 31;   // 2^5 - 1
    const int window_half = 16;   // 2^(5-1)
    
    int digit;
    ulong limb;

    while (!scalar_is_zero(&temp) && len < 260) {
        if (scalar_bit(&temp, 0) == 1) { // temp is odd
            digit = (int)(temp.limbs[0] & window_mask);
            
            if (digit >= window_half) {
                digit -= window_size;
                scalar_add_u64(&temp, (ulong)(-digit), &temp);
            } else {
                scalar_sub_u64(&temp, (ulong)digit, &temp);
            }
            
            wnaf[len] = digit;
        } else {
            wnaf[len] = 0;
        }
        
        // Right shift by 1
        limb = temp.limbs[3];
        temp.limbs[3] = (limb >> 1);
        ulong carry = limb & 1;
        
        limb = temp.limbs[2];
        temp.limbs[2] = (limb >> 1) | (carry << 63);
        carry = limb & 1;
        
        limb = temp.limbs[1];
        temp.limbs[1] = (limb >> 1) | (carry << 63);
        carry = limb & 1;
        
        limb = temp.limbs[0];
        temp.limbs[0] = (limb >> 1) | (carry << 63);
        
        len++;
    }
    
    return len;
}

// Negate Y coordinate of Jacobian point
FORCE_INLINE void point_negate_y(JacobianPoint* p) {
    FieldElement zero;
    zero.limbs[0] = 0; zero.limbs[1] = 0;
    zero.limbs[2] = 0; zero.limbs[3] = 0;
    field_neg_impl(&p->y, &p->y);
}

// =============================================================================
// Scalar Multiplication: R = k * P
// wNAF (window width 5) — matches CUDA's scalar_mul
// =============================================================================

FORCE_INLINE void scalar_mul_impl(JacobianPoint* r, const Scalar* k, const AffinePoint* p) {
    // Check for zero scalar
    if (scalar_is_zero(k)) {
        point_set_infinity(r);
        return;
    }

    // Convert scalar to wNAF representation
    int wnaf[260];
    int wnaf_len = scalar_to_wnaf(k, wnaf);

    // Precompute table: [P, 3P, 5P, ..., 15P] (8 entries)
    JacobianPoint table[8];
    JacobianPoint double_p;
    
    point_from_affine(&table[0], p);
    point_double_impl(&double_p, &table[0]);
    
    for (int i = 1; i < 8; i++) {
        point_add_impl(&table[i], &table[i-1], &double_p);
    }

    // Initialize result as infinity
    point_set_infinity(r);

    int digit;
    int idx;

    // Process wNAF from MSB to LSB
    for (int i = wnaf_len - 1; i >= 0; --i) {
        point_double_impl(r, r);

        digit = wnaf[i];
        if (digit > 0) {
            idx = (digit - 1) / 2;
            point_add_impl(r, r, &table[idx]);
        } else if (digit < 0) {
            idx = (-digit - 1) / 2;
            JacobianPoint neg_point = table[idx];
            point_negate_y(&neg_point);
            point_add_impl(r, r, &neg_point);
        }
    }
}

// =============================================================================
// Scalar Multiplication with Generator: R = k * G
// Fixed-window w=4 with precomputed affine table of {0G..15G}.
// Uses mixed J+A additions and unchecked variants for maximum throughput.
// =============================================================================

FORCE_INLINE void scalar_mul_generator_impl(JacobianPoint* r, const Scalar* k) {
    // Precomputed affine table: table[i] = i*G for i = 0..15.
    // table[0] is the point at infinity (unused except as sentinel).
    AffinePoint table[16];
    table[0].x.limbs[0] = 0; table[0].x.limbs[1] = 0; table[0].x.limbs[2] = 0; table[0].x.limbs[3] = 0;
    table[0].y.limbs[0] = 0; table[0].y.limbs[1] = 0; table[0].y.limbs[2] = 0; table[0].y.limbs[3] = 0;
    // 1*G
    table[1].x.limbs[0] = 0x59F2815B16F81798UL; table[1].x.limbs[1] = 0x029BFCDB2DCE28D9UL;
    table[1].x.limbs[2] = 0x55A06295CE870B07UL; table[1].x.limbs[3] = 0x79BE667EF9DCBBACUL;
    table[1].y.limbs[0] = 0x9C47D08FFB10D4B8UL; table[1].y.limbs[1] = 0xFD17B448A6855419UL;
    table[1].y.limbs[2] = 0x5DA4FBFC0E1108A8UL; table[1].y.limbs[3] = 0x483ADA7726A3C465UL;
    // 2*G
    table[2].x.limbs[0] = 0xABAC09B95C709EE5UL; table[2].x.limbs[1] = 0x5C778E4B8CEF3CA7UL;
    table[2].x.limbs[2] = 0x3045406E95C07CD8UL; table[2].x.limbs[3] = 0xC6047F9441ED7D6DUL;
    table[2].y.limbs[0] = 0x236431A950CFE52AUL; table[2].y.limbs[1] = 0xF7F632653266D0E1UL;
    table[2].y.limbs[2] = 0xA3C58419466CEAEEUL; table[2].y.limbs[3] = 0x1AE168FEA63DC339UL;
    // 3*G
    table[3].x.limbs[0] = 0x8601F113BCE036F9UL; table[3].x.limbs[1] = 0xB531C845836F99B0UL;
    table[3].x.limbs[2] = 0x49344F85F89D5229UL; table[3].x.limbs[3] = 0xF9308A019258C310UL;
    table[3].y.limbs[0] = 0x6CB9FD7584B8E672UL; table[3].y.limbs[1] = 0x6500A99934C2231BUL;
    table[3].y.limbs[2] = 0x0FE337E62A37F356UL; table[3].y.limbs[3] = 0x388F7B0F632DE814UL;
    // 4*G
    table[4].x.limbs[0] = 0x74FA94ABE8C4CD13UL; table[4].x.limbs[1] = 0xCC6C13900EE07584UL;
    table[4].x.limbs[2] = 0x581E4904930B1404UL; table[4].x.limbs[3] = 0xE493DBF1C10D80F3UL;
    table[4].y.limbs[0] = 0xCFE97BDC47739922UL; table[4].y.limbs[1] = 0xD967AE33BFBDFE40UL;
    table[4].y.limbs[2] = 0x5642E2098EA51448UL; table[4].y.limbs[3] = 0x51ED993EA0D455B7UL;
    // 5*G
    table[5].x.limbs[0] = 0xCBA8D569B240EFE4UL; table[5].x.limbs[1] = 0xE88B84BDDC619AB7UL;
    table[5].x.limbs[2] = 0x55B4A7250A5C5128UL; table[5].x.limbs[3] = 0x2F8BDE4D1A072093UL;
    table[5].y.limbs[0] = 0xDCA87D3AA6AC62D6UL; table[5].y.limbs[1] = 0xF788271BAB0D6840UL;
    table[5].y.limbs[2] = 0xD4DBA9DDA6C9C426UL; table[5].y.limbs[3] = 0xD8AC222636E5E3D6UL;
    // 6*G
    table[6].x.limbs[0] = 0x2F057A1460297556UL; table[6].x.limbs[1] = 0x82F6472F8568A18BUL;
    table[6].x.limbs[2] = 0x20453A14355235D3UL; table[6].x.limbs[3] = 0xFFF97BD5755EEEA4UL;
    table[6].y.limbs[0] = 0x3C870C36B075F297UL; table[6].y.limbs[1] = 0xDE80F0F6518FE4A0UL;
    table[6].y.limbs[2] = 0xF3BE96017F45C560UL; table[6].y.limbs[3] = 0xAE12777AACFBB620UL;
    // 7*G
    table[7].x.limbs[0] = 0xE92BDDEDCAC4F9BCUL; table[7].x.limbs[1] = 0x3D419B7E0330E39CUL;
    table[7].x.limbs[2] = 0xA398F365F2EA7A0EUL; table[7].x.limbs[3] = 0x5CBDF0646E5DB4EAUL;
    table[7].y.limbs[0] = 0xA5082628087264DAUL; table[7].y.limbs[1] = 0xA813D0B813FDE7B5UL;
    table[7].y.limbs[2] = 0xA3178D6D861A54DBUL; table[7].y.limbs[3] = 0x6AEBCA40BA255960UL;
    // 8*G
    table[8].x.limbs[0] = 0x67784EF3E10A2A01UL; table[8].x.limbs[1] = 0x0A1BDD05E5AF888AUL;
    table[8].x.limbs[2] = 0xAFF3843FB70F3C2FUL; table[8].x.limbs[3] = 0x2F01E5E15CCA351DUL;
    table[8].y.limbs[0] = 0xB5DA2CB76CBDE904UL; table[8].y.limbs[1] = 0xC2E213D6BA5B7617UL;
    table[8].y.limbs[2] = 0x293D082A132D13B4UL; table[8].y.limbs[3] = 0x5C4DA8A741539949UL;
    // 9*G
    table[9].x.limbs[0] = 0xC35F110DFC27CCBEUL; table[9].x.limbs[1] = 0xE09796974C57E714UL;
    table[9].x.limbs[2] = 0x09AD178A9F559ABDUL; table[9].x.limbs[3] = 0xACD484E2F0C7F653UL;
    table[9].y.limbs[0] = 0x05CC262AC64F9C37UL; table[9].y.limbs[1] = 0xADD888A4375F8E0FUL;
    table[9].y.limbs[2] = 0x64380971763B61E9UL; table[9].y.limbs[3] = 0xCC338921B0A7D9FDUL;
    // 10*G
    table[10].x.limbs[0] = 0x52A68E2A47E247C7UL; table[10].x.limbs[1] = 0x3442D49B1943C2B7UL;
    table[10].x.limbs[2] = 0x35477C7B1AE6AE5DUL; table[10].x.limbs[3] = 0xA0434D9E47F3C862UL;
    table[10].y.limbs[0] = 0x3CBEE53B037368D7UL; table[10].y.limbs[1] = 0x6F794C2ED877A159UL;
    table[10].y.limbs[2] = 0xA3B6C7E693A24C69UL; table[10].y.limbs[3] = 0x893ABA425419BC27UL;
    // 11*G
    table[11].x.limbs[0] = 0xBBEC17895DA008CBUL; table[11].x.limbs[1] = 0x5649980BE5C17891UL;
    table[11].x.limbs[2] = 0x5EF4246B70C65AACUL; table[11].x.limbs[3] = 0x774AE7F858A9411EUL;
    table[11].y.limbs[0] = 0x301D74C9C953C61BUL; table[11].y.limbs[1] = 0x372DB1E2DFF9D6A8UL;
    table[11].y.limbs[2] = 0x0243DD56D7B7B365UL; table[11].y.limbs[3] = 0xD984A032EB6B5E19UL;
    // 12*G
    table[12].x.limbs[0] = 0xC5B0F47070AFE85AUL; table[12].x.limbs[1] = 0x687CF4419620095BUL;
    table[12].x.limbs[2] = 0x15C38F004D734633UL; table[12].x.limbs[3] = 0xD01115D548E7561BUL;
    table[12].y.limbs[0] = 0x6B051B13F4062327UL; table[12].y.limbs[1] = 0x79238C5DD9A86D52UL;
    table[12].y.limbs[2] = 0xA8B64537E17BD815UL; table[12].y.limbs[3] = 0xA9F34FFDC815E0D7UL;
    // 13*G
    table[13].x.limbs[0] = 0xDEEDDF8F19405AA8UL; table[13].x.limbs[1] = 0xB075FBC6610E58CDUL;
    table[13].x.limbs[2] = 0xC7D1D205C3748651UL; table[13].x.limbs[3] = 0xF28773C2D975288BUL;
    table[13].y.limbs[0] = 0x29B5CB52DB03ED81UL; table[13].y.limbs[1] = 0x3A1A06DA521FA91FUL;
    table[13].y.limbs[2] = 0x758212EB65CDAF47UL; table[13].y.limbs[3] = 0x0AB0902E8D880A89UL;
    // 14*G
    table[14].x.limbs[0] = 0xE49B241A60E823E4UL; table[14].x.limbs[1] = 0x26AA7B63678949E6UL;
    table[14].x.limbs[2] = 0xFD64E67F07D38E32UL; table[14].x.limbs[3] = 0x499FDF9E895E719CUL;
    table[14].y.limbs[0] = 0xC65F40D403A13F5BUL; table[14].y.limbs[1] = 0x464279C27A3F95BCUL;
    table[14].y.limbs[2] = 0x90F044E4A7B3D464UL; table[14].y.limbs[3] = 0xCAC2F6C4B54E8551UL;
    // 15*G
    table[15].x.limbs[0] = 0x44ADBCF8E27E080EUL; table[15].x.limbs[1] = 0x31E5946F3C85F79EUL;
    table[15].x.limbs[2] = 0x5A465AE3095FF411UL; table[15].x.limbs[3] = 0xD7924D4F7D43EA96UL;
    table[15].y.limbs[0] = 0xC504DC9FF6A26B58UL; table[15].y.limbs[1] = 0xEA40AF2BD896D3A5UL;
    table[15].y.limbs[2] = 0x83842EC228CC6DEFUL; table[15].y.limbs[3] = 0x581E2872A86C72A6UL;

    // Process scalar 4 bits at a time (MSB first)
    point_set_infinity(r);
    int started = 0;

    for (int limb = 3; limb >= 0; limb--) {
        ulong w = k->limbs[limb];
        for (int nib = 15; nib >= 0; nib--) {
            uint idx = (uint)((w >> (nib * 4)) & 0xFUL);

            if (started) {
                point_double_unchecked(r, r);
                point_double_unchecked(r, r);
                point_double_unchecked(r, r);
                point_double_unchecked(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    point_from_affine(r, &table[idx]);
                    started = 1;
                } else {
                    point_add_mixed_unchecked(r, r, &table[idx]);
                }
            }
        }
    }
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

