// =============================================================================
// UltrafastSecp256k1 OpenCL — Extended Scalar, Crypto & MSM Operations
// =============================================================================
// This file extends the OpenCL kernels with all missing functionality:
//
// Layer 1: Serialization (scalar_from_bytes, scalar_to_bytes, field_to_bytes)
//          + field_sqrt (modular square root)
// Layer 2: Scalar mod-n algebra (negate, mul, inverse, add, sub, eq, ge, etc.)
//          + GLV endomorphism (decompose + dual scalar mul)
// Layer 3: SHA-256 streaming + HMAC-SHA256 + RFC 6979
// Layer 4: ECDSA sign/verify + precomputed generator mul
// Layer 5: Schnorr BIP-340 + ECDH + Key Recovery + MSM/Pippenger
//
// Depends on: secp256k1_point.cl (which includes secp256k1_field.cl)
// Uses 4×64-bit limbs (ulong) — matching existing OpenCL convention.
// =============================================================================

#include "secp256k1_point.cl"

// =============================================================================
// Constants
// =============================================================================

// Order n in 64-bit LE limbs
#define ORDER_N0 0xBFD25E8CD0364141UL
#define ORDER_N1 0xBAAEDCE6AF48A03BUL
#define ORDER_N2 0xFFFFFFFFFFFFFFFEUL
#define ORDER_N3 0xFFFFFFFFFFFFFFFFUL

// n/2 (half order for low-S check) LE limbs
#define HALF_ORDER_0 0xDFE92F46681B20A0UL
#define HALF_ORDER_1 0x5D576E7357A4501DUL
#define HALF_ORDER_2 0xFFFFFFFFFFFFFFFFUL
#define HALF_ORDER_3 0x7FFFFFFFFFFFFFFFUL

// n - 2 (for Fermat little theorem inversion mod n)
#define ORDER_N_MINUS2_0 0xBFD25E8CD036413FUL
#define ORDER_N_MINUS2_1 0xBAAEDCE6AF48A03BUL
#define ORDER_N_MINUS2_2 0xFFFFFFFFFFFFFFFEUL
#define ORDER_N_MINUS2_3 0xFFFFFFFFFFFFFFFFUL

// Barrett constant mu = floor(2^512 / n), 5×64-bit LE
#define BARRETT_MU0 0x402DA1732FC9BEC0UL
#define BARRETT_MU1 0x4551231950B75FC4UL
#define BARRETT_MU2 0x0000000000000001UL
#define BARRETT_MU3 0x0000000000000000UL
#define BARRETT_MU4 0x0000000000000001UL

// beta (GLV endomorphism field constant: cube root of unity in Fp)
// BETA = 0x7ae96a2b657c0710_6e64479eac3434e9_9cf0497512f58995_c1396c28719501ee
// Stored in LE limb order (limb[0] = LSW)
#define GLV_BETA0 0xC1396C28719501EEUL
#define GLV_BETA1 0x9CF0497512F58995UL
#define GLV_BETA2 0x6E64479EAC3434E9UL
#define GLV_BETA3 0x7AE96A2B657C0710UL

// lambda (GLV endomorphism scalar: [lambda]*P = (beta*x, y) for any point P on secp256k1)
// LAMBDA = 0x5363ad4cc05c30e0_a5261c028812645a_122e22ea20816678_df02967c1b23bd72
// Stored in LE limb order (limb[0] = LSW)
#define GLV_LAMBDA0 0xDF02967C1B23BD72UL
#define GLV_LAMBDA1 0x122E22EA20816678UL
#define GLV_LAMBDA2 0xA5261C028812645AUL
#define GLV_LAMBDA3 0x5363AD4CC05C30E0UL

// GLV lattice vectors g1, g2 (full 256-bit, for mul_shift_384)
__constant ulong GLV_G1[4] = {
    0xE893209A45DBB031UL, 0x3DAA8A1471E8CA7FUL,
    0xE86C90E49284EB15UL, 0x3086D221A7D46BCDUL
};
__constant ulong GLV_G2[4] = {
    0x1571B4AE8AC47F71UL, 0x221208AC9DF506C6UL,
    0x6F547FA90ABFE4C4UL, 0xE4437ED6010E8828UL
};
// -b1 and -b2 vectors (full 256-bit)
__constant ulong GLV_MINUS_B1[4] = {
    0x6F547FA90ABFE4C3UL, 0xE4437ED6010E8828UL, 0x0UL, 0x0UL
};
__constant ulong GLV_MINUS_B2[4] = {
    0xD765CDA83DB1562CUL, 0x8A280AC50774346DUL,
    0xFFFFFFFFFFFFFFFEUL, 0xFFFFFFFFFFFFFFFFUL
};

// SHA-256 round constants
__constant uint K256[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

// =============================================================================
// LAYER 1: Serialization + field_sqrt
// =============================================================================

// BE 32 bytes → Scalar (4×64 LE limbs) with branchless mod n reduction
inline void scalar_from_bytes_impl(const uchar bytes[32], Scalar* out) {
    for (int i = 0; i < 4; i++) {
        ulong limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++)
            limb = (limb << 8) | (ulong)bytes[base + j];
        out->limbs[i] = limb;
    }
    // Branchless reduction: if scalar >= n, subtract n
    ulong borrow = 0, tmp[4];
    ulong n[4] = { ORDER_N0, ORDER_N1, ORDER_N2, ORDER_N3 };
    for (int i = 0; i < 4; i++)
        tmp[i] = sub_with_borrow(out->limbs[i], n[i], borrow, &borrow);
    ulong mask = -(ulong)(borrow == 0); // if no borrow, scalar >= n
    for (int i = 0; i < 4; i++)
        out->limbs[i] = (tmp[i] & mask) | (out->limbs[i] & ~mask);
}

// Scalar → BE 32 bytes
inline void scalar_to_bytes_impl(const Scalar* s, uchar out[32]) {
    for (int i = 0; i < 4; i++) {
        ulong limb = s->limbs[3 - i];
        for (int j = 0; j < 8; j++)
            out[i * 8 + j] = (uchar)(limb >> (56 - j * 8));
    }
}

// FieldElement → BE 32 bytes (normalizes mod p before serialization)
inline void field_to_bytes_impl(const FieldElement* f, uchar out[32]) {
    // p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
    const ulong P[4] = {
        0xFFFFFFFEFFFFFC2FUL, 0xFFFFFFFFFFFFFFFFUL,
        0xFFFFFFFFFFFFFFFFUL, 0xFFFFFFFFFFFFFFFFUL
    };
    // Branchless: subtract p, keep result only if no borrow (fe >= p)
    ulong tmp[4];
    ulong borrow = 0;
    for (int i = 0; i < 4; i++) {
        ulong a = f->limbs[i];
        ulong b = P[i] + borrow;
        ulong underflow = (borrow && b == 0) ? 1UL : 0UL;
        tmp[i] = a - b;
        borrow = (a < b || underflow) ? 1UL : 0UL;
    }
    // borrow==0 → fe >= p → use tmp; borrow==1 → fe < p → use original
    ulong mask = (borrow == 0) ? ~0UL : 0UL;
    ulong norm[4];
    for (int i = 0; i < 4; i++)
        norm[i] = (tmp[i] & mask) | (f->limbs[i] & ~mask);

    for (int i = 0; i < 4; i++) {
        ulong limb = norm[3 - i];
        for (int j = 0; j < 8; j++)
            out[i * 8 + j] = (uchar)(limb >> (56 - j * 8));
    }
}

// Field square root: a^((p+1)/4) via optimized addition chain (269 ops)
// p ≡ 3 (mod 4) ⇒ sqrt(a) = a^((p+1)/4)
inline void field_sqrt_impl(const FieldElement* a, FieldElement* r) {
    FieldElement x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x222, t;

    // x2 = a^(2^2-1)
    field_sqr_impl(&x2, a);         // a^2
    field_mul_impl(&x2, &x2, a);    // a^3 = a^(2^2-1)

    // x3 = a^(2^3-1)
    field_sqr_impl(&x3, &x2);       // a^6
    field_mul_impl(&x3, &x3, a);    // a^7 = a^(2^3-1)

    // x6 = a^(2^6-1)
    t = x3;
    field_sqr_n_impl(&t, 3);
    field_mul_impl(&x6, &t, &x3);

    // x9 = a^(2^9-1)
    t = x6;
    field_sqr_n_impl(&t, 3);
    field_mul_impl(&x9, &t, &x3);

    // x11 = a^(2^11-1)
    t = x9;
    field_sqr_n_impl(&t, 2);
    field_mul_impl(&x11, &t, &x2);

    // x22 = a^(2^22-1)
    t = x11;
    field_sqr_n_impl(&t, 11);
    field_mul_impl(&x22, &t, &x11);

    // x44 = a^(2^44-1)
    t = x22;
    field_sqr_n_impl(&t, 22);
    field_mul_impl(&x44, &t, &x22);

    // x88 = a^(2^88-1)
    t = x44;
    field_sqr_n_impl(&t, 44);
    field_mul_impl(&x88, &t, &x44);

    // x176 = a^(2^176-1)
    t = x88;
    field_sqr_n_impl(&t, 88);
    field_mul_impl(&x176, &t, &x88);

    // x220 = a^(2^220-1)
    t = x176;
    field_sqr_n_impl(&t, 44);
    field_mul_impl(&x220, &t, &x44);

    // x222 = a^(2^222-1)
    t = x220;
    field_sqr_n_impl(&t, 2);
    field_mul_impl(&x222, &t, &x2);

    // Tail: bits 10 1^22 0000 11 00
    // x223 = x222 * a after one squaring
    t = x222;
    field_sqr_impl(&t, &t);
    field_mul_impl(&t, &t, a);

    // shift left 1 (just square)
    field_sqr_impl(&t, &t);

    // shift left 22, multiply by x22
    field_sqr_n_impl(&t, 22);
    field_mul_impl(&t, &t, &x22);

    // compute a^12 = (a^3)^4 = x2 squared twice
    FieldElement a12;
    field_sqr_impl(&a12, &x2);
    field_sqr_impl(&a12, &a12);

    // shift left 8, multiply by a^12
    field_sqr_n_impl(&t, 8);
    field_mul_impl(r, &t, &a12);
}

// =============================================================================
// LAYER 2: Scalar mod-n Algebra
// =============================================================================

// Scalar negate: r = n - a (if a != 0)
inline void scalar_negate_impl(const Scalar* a, Scalar* r) {
    ulong n[4] = { ORDER_N0, ORDER_N1, ORDER_N2, ORDER_N3 };
    int is_zero_flag = scalar_is_zero(a);

    ulong borrow = 0;
    for (int i = 0; i < 4; i++)
        r->limbs[i] = sub_with_borrow(n[i], a->limbs[i], borrow, &borrow);
    // If a was zero, result should be zero too
    ulong mask = -(ulong)(!is_zero_flag);
    for (int i = 0; i < 4; i++) r->limbs[i] &= mask;
}

// Helper: branchless conditional subtract n (r -= n if r >= n)
inline void scalar_cond_sub_n(Scalar* r) {
    ulong n[4] = { ORDER_N0, ORDER_N1, ORDER_N2, ORDER_N3 };
    ulong borrow = 0;
    ulong tmp[4];
    for (int i = 0; i < 4; i++)
        tmp[i] = sub_with_borrow(r->limbs[i], n[i], borrow, &borrow);
    // borrow==0 means r >= n, use subtracted result
    ulong mask = -(ulong)(borrow == 0);
    for (int i = 0; i < 4; i++)
        r->limbs[i] = (tmp[i] & mask) | (r->limbs[i] & ~mask);
}

// Scalar add mod n: r = (a + b) mod n
inline void scalar_add_mod_n_impl(const Scalar* a, const Scalar* b, Scalar* r) {
    ulong carry = 0;
    for (int i = 0; i < 4; i++)
        r->limbs[i] = add_with_carry(a->limbs[i], b->limbs[i], carry, &carry);
    // If carry, definitely >= n; otherwise check and conditionally subtract
    if (carry) {
        // r + 2^256 - n: since carry=1, effectively subtract (n - 2^256) = subtract n, add 2^256
        // which is same as: result = r - n (the carry absorbed the 2^256)
        ulong n[4] = { ORDER_N0, ORDER_N1, ORDER_N2, ORDER_N3 };
        ulong borrow = 0;
        for (int i = 0; i < 4; i++)
            r->limbs[i] = sub_with_borrow(r->limbs[i], n[i], borrow, &borrow);
    } else {
        scalar_cond_sub_n(r);
    }
}

// Scalar sub mod n: r = (a - b) mod n
inline void scalar_sub_mod_n_impl(const Scalar* a, const Scalar* b, Scalar* r) {
    ulong borrow = 0;
    for (int i = 0; i < 4; i++)
        r->limbs[i] = sub_with_borrow(a->limbs[i], b->limbs[i], borrow, &borrow);
    // If borrow, add n back
    if (borrow) {
        ulong n[4] = { ORDER_N0, ORDER_N1, ORDER_N2, ORDER_N3 };
        ulong carry2 = 0;
        for (int i = 0; i < 4; i++)
            r->limbs[i] = add_with_carry(r->limbs[i], n[i], carry2, &carry2);
    }
}

// Scalar multiply mod n: r = (a * b) mod n
// Uses 2^256 ≡ NC (mod n) reduction where NC = 2^256 - n
inline void scalar_mul_mod_n_impl(const Scalar* a, const Scalar* b, Scalar* r) {
    // NC = 2^256 - n = {0x402DA1732FC9BEBF, 0x4551231950B75FC4, 1, 0}
    ulong NC[3] = { 0x402DA1732FC9BEBFUL, 0x4551231950B75FC4UL, 0x1UL };

    // Step 1: Full 512-bit schoolbook multiplication
    ulong prod[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            ulong2 full = mul64_full(a->limbs[i], b->limbs[j]);
            ulong c1, c2;
            ulong s = add_with_carry(full.x, prod[i+j], 0, &c1);
            s = add_with_carry(s, carry, 0, &c2);
            prod[i+j] = s;
            carry = full.y + c1 + c2;
        }
        prod[i+4] = carry;
    }

    // Step 2: Reduce high 256 bits. acc = prod[0..3] + prod[4..7] * NC
    // prod[4..7] * NC has at most 256+129 = 385 bits
    ulong acc[7] = {prod[0], prod[1], prod[2], prod[3], 0, 0, 0};
    for (int i = 0; i < 4; i++) {
        if (prod[4+i] == 0) continue;
        ulong carry = 0;
        for (int j = 0; j < 3; j++) {
            ulong2 full = mul64_full(prod[4+i], NC[j]);
            ulong c1, c2;
            ulong s = add_with_carry(full.x, acc[i+j], 0, &c1);
            s = add_with_carry(s, carry, 0, &c2);
            acc[i+j] = s;
            carry = full.y + c1 + c2;
        }
        // Propagate remaining carry
        for (int k = i+3; k < 7 && carry; k++) {
            acc[k] = add_with_carry(acc[k], carry, 0, &carry);
        }
    }

    // Step 3: Reduce again. res = acc[0..3] + acc[4..6] * NC
    ulong res[5] = {acc[0], acc[1], acc[2], acc[3], 0};
    for (int i = 0; i < 3; i++) {
        if (acc[4+i] == 0) continue;
        ulong carry = 0;
        for (int j = 0; j < 3; j++) {
            if (i+j >= 5) break;
            ulong2 full = mul64_full(acc[4+i], NC[j]);
            ulong c1, c2;
            ulong s = add_with_carry(full.x, res[i+j], 0, &c1);
            s = add_with_carry(s, carry, 0, &c2);
            res[i+j] = s;
            carry = full.y + c1 + c2;
        }
        for (int k = i+3; k < 5 && carry; k++) {
            res[k] = add_with_carry(res[k], carry, 0, &carry);
        }
    }

    // Step 4: Handle res[4] overflow
    r->limbs[0] = res[0]; r->limbs[1] = res[1];
    r->limbs[2] = res[2]; r->limbs[3] = res[3];
    if (res[4] != 0) {
        ulong carry = 0;
        for (int j = 0; j < 3; j++) {
            ulong2 full = mul64_full(res[4], NC[j]);
            ulong c1, c2;
            ulong s = add_with_carry(full.x, r->limbs[j], 0, &c1);
            s = add_with_carry(s, carry, 0, &c2);
            r->limbs[j] = s;
            carry = full.y + c1 + c2;
        }
        r->limbs[3] += carry;
    }

    // Step 5: Conditional subtract n (at most 3 times to ensure < n)
    scalar_cond_sub_n(r);
    scalar_cond_sub_n(r);
    scalar_cond_sub_n(r);
}

// Scalar inverse mod n via binary exponentiation: a^(n-2) mod n
inline void scalar_inverse_impl(const Scalar* a, Scalar* r) {
    ulong exp[4] = { ORDER_N_MINUS2_0, ORDER_N_MINUS2_1, ORDER_N_MINUS2_2, ORDER_N_MINUS2_3 };
    Scalar base = *a;
    Scalar result;
    result.limbs[0] = 1; result.limbs[1] = 0;
    result.limbs[2] = 0; result.limbs[3] = 0;

    for (int i = 0; i < 4; i++) {
        for (int bit = 0; bit < 64; bit++) {
            if ((exp[i] >> bit) & 1UL) {
                scalar_mul_mod_n_impl(&result, &base, &result);
            }
            scalar_mul_mod_n_impl(&base, &base, &base);
        }
    }
    *r = result;
}

// Scalar is even: test bit 0
inline int scalar_is_even_impl(const Scalar* s) {
    return (s->limbs[0] & 1UL) == 0;
}

// Scalar equality
inline int scalar_eq_impl(const Scalar* a, const Scalar* b) {
    ulong diff = 0;
    for (int i = 0; i < 4; i++) diff |= (a->limbs[i] ^ b->limbs[i]);
    return diff == 0;
}

// Scalar bit length
inline int scalar_bitlen_impl(const Scalar* s) {
    for (int i = 3; i >= 0; i--) {
        if (s->limbs[i] != 0) {
            int bits = 64;
            ulong v = s->limbs[i];
            while (!(v >> 63)) { v <<= 1; bits--; }
            return i * 64 + bits;
        }
    }
    return 0;
}

// Scalar greater-or-equal
inline int scalar_ge_impl(const Scalar* a, const Scalar* b) {
    for (int i = 3; i >= 0; i--) {
        if (a->limbs[i] > b->limbs[i]) return 1;
        if (a->limbs[i] < b->limbs[i]) return 0;
    }
    return 1; // equal
}

// low-S check (BIP-62)
inline int scalar_is_low_s_impl(const Scalar* s) {
    Scalar half_n;
    half_n.limbs[0] = HALF_ORDER_0; half_n.limbs[1] = HALF_ORDER_1;
    half_n.limbs[2] = HALF_ORDER_2; half_n.limbs[3] = HALF_ORDER_3;

    for (int i = 3; i >= 0; i--) {
        if (s->limbs[i] > half_n.limbs[i]) return 0;
        if (s->limbs[i] < half_n.limbs[i]) return 1;
    }
    return 1; // equal = low
}

// =============================================================================
// GLV Endomorphism
// =============================================================================

inline void apply_endomorphism_impl(const JacobianPoint* p, JacobianPoint* r) {
    FieldElement beta;
    beta.limbs[0] = 0x7AE96A2B657C0710UL;
    beta.limbs[1] = 0x6E64479EAC3434E9UL;
    beta.limbs[2] = 0x9CF0497512F58995UL;
    beta.limbs[3] = 0xC1396C28719501EEUL;

    field_mul_impl(&r->x, &p->x, &beta);
    r->y = p->y;
    r->z = p->z;
    r->infinity = p->infinity;
}

// Field negation: r = p - a (mod p)
inline void field_negate_impl(FieldElement* r, const FieldElement* a) {
    FieldElement zero;
    zero.limbs[0] = 0; zero.limbs[1] = 0; zero.limbs[2] = 0; zero.limbs[3] = 0;
    field_sub_impl(r, &zero, a);
}

// GLV decomposition: k = k1 + k2*lambda (mod n), |k1|,|k2| ~ 128 bits
// Uses full lattice-based decomposition with Babai rounding.

// (a * b) >> 384 with rounding (bit 383)
inline void mul_shift_384_impl(const ulong a[4], __constant const ulong b[4], ulong result[4]) {
    ulong prod[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; i++) {
        ulong carry = 0;
        for (int j = 0; j < 4; j++) {
            ulong2 full = mul64_full(a[i], b[j]);
            ulong c1 = 0, c2 = 0;
            ulong s = add_with_carry(full.x, prod[i+j], 0, &c1);
            s = add_with_carry(s, carry, 0, &c2);
            prod[i+j] = s;
            carry = full.y + c1 + c2;
        }
        prod[i+4] = carry;
    }
    result[0] = prod[6];  result[1] = prod[7];
    result[2] = 0;        result[3] = 0;
    if (prod[5] >> 63) {  // rounding bit 383
        result[0]++;
        if (result[0] == 0) result[1]++;
    }
}

inline void glv_decompose_impl(const Scalar* k, Scalar* k1, Scalar* k2,
                                 int* k1_neg, int* k2_neg) {
    // c1 = round(k * g1 / 2^384), c2 = round(k * g2 / 2^384)
    ulong c1_limbs[4], c2_limbs[4];
    mul_shift_384_impl(k->limbs, GLV_G1, c1_limbs);
    mul_shift_384_impl(k->limbs, GLV_G2, c2_limbs);

    Scalar c1, c2;
    for (int i = 0; i < 4; i++) { c1.limbs[i] = c1_limbs[i]; c2.limbs[i] = c2_limbs[i]; }

    // Reduce c1, c2 mod n if needed
    Scalar order;
    order.limbs[0] = ORDER_N0; order.limbs[1] = ORDER_N1;
    order.limbs[2] = ORDER_N2; order.limbs[3] = ORDER_N3;
    if (scalar_ge_impl(&c1, &order)) scalar_sub_mod_n_impl(&c1, &order, &c1);
    if (scalar_ge_impl(&c2, &order)) scalar_sub_mod_n_impl(&c2, &order, &c2);

    // k2_mod = c1*(-b1) + c2*(-b2) mod n
    Scalar minus_b1, minus_b2;
    for (int i = 0; i < 4; i++) {
        minus_b1.limbs[i] = GLV_MINUS_B1[i];
        minus_b2.limbs[i] = GLV_MINUS_B2[i];
    }
    Scalar t1, t2, k2_mod;
    scalar_mul_mod_n_impl(&c1, &minus_b1, &t1);
    scalar_mul_mod_n_impl(&c2, &minus_b2, &t2);
    scalar_add_mod_n_impl(&t1, &t2, &k2_mod);

    // Pick shorter k2: compare |k2_mod| vs |n - k2_mod|
    Scalar k2_neg_val;
    scalar_negate_impl(&k2_mod, &k2_neg_val);
    int k2_is_neg = (scalar_bitlen_impl(&k2_neg_val) < scalar_bitlen_impl(&k2_mod));
    Scalar k2_abs = k2_is_neg ? k2_neg_val : k2_mod;

    // For computing k1: need the signed k2
    Scalar k2_signed;
    if (k2_is_neg) { scalar_negate_impl(&k2_abs, &k2_signed); }
    else           { k2_signed = k2_abs; }

    // k1 = k - lambda*k2_signed mod n
    Scalar lambda_s;
    lambda_s.limbs[0] = GLV_LAMBDA0; lambda_s.limbs[1] = GLV_LAMBDA1;
    lambda_s.limbs[2] = GLV_LAMBDA2; lambda_s.limbs[3] = GLV_LAMBDA3;
    Scalar lk2;
    scalar_mul_mod_n_impl(&lambda_s, &k2_signed, &lk2);
    Scalar k1_mod;
    scalar_sub_mod_n_impl(k, &lk2, &k1_mod);

    // Pick shorter k1
    Scalar k1_neg_val;
    scalar_negate_impl(&k1_mod, &k1_neg_val);
    int k1_is_neg = (scalar_bitlen_impl(&k1_neg_val) < scalar_bitlen_impl(&k1_mod));
    Scalar k1_abs = k1_is_neg ? k1_neg_val : k1_mod;

    *k1 = k1_abs;  *k2 = k2_abs;
    *k1_neg = k1_is_neg;  *k2_neg = k2_is_neg;
}

// GLV-accelerated scalar multiplication: k*P using Shamir's trick
// with endomorphism phi(P) = (beta*x, y) where phi corresponds to lambda.
// Uses interleaved wNAF w=5 for both half-scalars k1, k2.
inline void scalar_mul_glv_impl(JacobianPoint* r, const Scalar* k, const AffinePoint* p) {
    Scalar k1, k2;
    int k1_neg, k2_neg;
    glv_decompose_impl(k, &k1, &k2, &k1_neg, &k2_neg);

    // Build base point, negate if k1 is negative
    AffinePoint base = *p;
    if (k1_neg) field_negate_impl(&base.y, &base.y);

    // Build P precomp table: [P, 3P, 5P, ..., 15P] (8 entries, w=5)
    JacobianPoint tbl_jac[8];
    JacobianPoint dbl;
    point_from_affine(&tbl_jac[0], &base);
    point_double_impl(&dbl, &tbl_jac[0]);
    for (int i = 1; i < 8; i++)
        point_add_impl(&tbl_jac[i], &tbl_jac[i-1], &dbl);

    // Build phi(P) table: apply endomorphism, flip y if signs differ
    AffinePoint endo_base;
    FieldElement beta;
    beta.limbs[0] = GLV_BETA0; beta.limbs[1] = GLV_BETA1;
    beta.limbs[2] = GLV_BETA2; beta.limbs[3] = GLV_BETA3;
    field_mul_impl(&endo_base.x, &base.x, &beta);
    endo_base.y = base.y;
    int flip_phi = (k1_neg != k2_neg);
    if (flip_phi) field_negate_impl(&endo_base.y, &endo_base.y);

    JacobianPoint tbl2_jac[8];
    point_from_affine(&tbl2_jac[0], &endo_base);
    JacobianPoint dbl2;
    point_double_impl(&dbl2, &tbl2_jac[0]);
    for (int i = 1; i < 8; i++)
        point_add_impl(&tbl2_jac[i], &tbl2_jac[i-1], &dbl2);

    // wNAF encode both half-width scalars
    int wnaf1[260], wnaf2[260];
    int len1 = scalar_to_wnaf(&k1, wnaf1);
    int len2 = scalar_to_wnaf(&k2, wnaf2);
    int max_len = (len1 > len2) ? len1 : len2;

    // Shamir interleaved loop
    point_set_infinity(r);
    for (int i = max_len - 1; i >= 0; --i) {
        if (!point_is_infinity(r)) point_double_impl(r, r);

        int d1 = (i < len1) ? wnaf1[i] : 0;
        if (d1 != 0) {
            int idx = ((d1 > 0) ? d1 : -d1) >> 1;
            if (idx >= 8) idx = 7;
            JacobianPoint pt = tbl_jac[idx];
            if (d1 < 0) field_negate_impl(&pt.y, &pt.y);
            if (point_is_infinity(r)) { *r = pt; }
            else { JacobianPoint tmp; point_add_impl(&tmp, r, &pt); *r = tmp; }
        }

        int d2 = (i < len2) ? wnaf2[i] : 0;
        if (d2 != 0) {
            int idx = ((d2 > 0) ? d2 : -d2) >> 1;
            if (idx >= 8) idx = 7;
            JacobianPoint pt = tbl2_jac[idx];
            if (d2 < 0) field_negate_impl(&pt.y, &pt.y);
            if (point_is_infinity(r)) { *r = pt; }
            else { JacobianPoint tmp; point_add_impl(&tmp, r, &pt); *r = tmp; }
        }
    }
}

// Precomputed generator multiplication using fixed window w=4
inline void scalar_mul_generator_windowed_impl(JacobianPoint* r, const Scalar* k) {
    // Build 16-entry table: table[i] = i*G
    AffinePoint G;
    get_generator(&G);

    JacobianPoint table[16];
    point_set_infinity(&table[0]);

    point_from_affine(&table[1], &G);

    for (int i = 2; i < 16; i++) {
        if (i == 2) {
            point_double_impl(&table[2], &table[1]);
        } else {
            point_add_mixed_impl(&table[i], &table[i-1], &G);
        }
    }

    // Process scalar 4 bits at a time (MSB first)
    point_set_infinity(r);
    int started = 0;

    for (int limb = 3; limb >= 0; limb--) {
        ulong w = k->limbs[limb];
        for (int nib = 15; nib >= 0; nib--) {
            uint idx = (uint)((w >> (nib * 4)) & 0xFUL);

            if (started) {
                point_double_impl(r, r);
                point_double_impl(r, r);
                point_double_impl(r, r);
                point_double_impl(r, r);
            }

            if (idx != 0) {
                if (!started) {
                    *r = table[idx];
                    started = 1;
                } else {
                    JacobianPoint tmp;
                    point_add_impl(&tmp, r, &table[idx]);
                    *r = tmp;
                }
            }
        }
    }
}

// Generator multiplication via precomputed LUT in __global memory.
// lut: 16 slices x 65536 AffinePoints. lut[win*65536+idx] = idx * 2^(16*win) * G.
// 15 mixed additions, 0 doublings.
inline void scalar_mul_generator_lut_impl(JacobianPoint* r, const Scalar* k,
                                          __global const AffinePoint* lut) {
    point_set_infinity(r);
    for (int win = 0; win < 16; win++) {
        uint idx = (uint)((k->limbs[win >> 2] >> ((win & 3) * 16)) & 0xFFFFUL);
        if (idx != 0) {
            AffinePoint pt = lut[(uint)win * 65536 + idx];
            if (point_is_infinity(r)) {
                point_from_affine(r, &pt);
            } else {
                point_add_mixed_impl(r, r, &pt);
            }
        }
    }
}

// =============================================================================
// LAYER 3: SHA-256 Streaming + HMAC + RFC 6979
// =============================================================================

typedef struct {
    uint h[8];
    uchar buf[64];
    uint buf_len;
    ulong total_len;
} SHA256Ctx;

inline uint sha256_rotr(uint x, uint n) { return (x >> n) | (x << (32 - n)); }
inline uint sha256_ch(uint x, uint y, uint z)  { return (x & y) ^ (~x & z); }
inline uint sha256_maj(uint x, uint y, uint z) { return (x & y) ^ (x & z) ^ (y & z); }
inline uint sha256_bsig0(uint x) { return sha256_rotr(x,2) ^ sha256_rotr(x,13) ^ sha256_rotr(x,22); }
inline uint sha256_bsig1(uint x) { return sha256_rotr(x,6) ^ sha256_rotr(x,11) ^ sha256_rotr(x,25); }
inline uint sha256_ssig0(uint x) { return sha256_rotr(x,7) ^ sha256_rotr(x,18) ^ (x >> 3); }
inline uint sha256_ssig1(uint x) { return sha256_rotr(x,17) ^ sha256_rotr(x,19) ^ (x >> 10); }

inline void sha256_compress(SHA256Ctx* ctx, const uchar block[64]) {
    uint w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint)block[i*4] << 24) | ((uint)block[i*4+1] << 16)
             | ((uint)block[i*4+2] << 8) | (uint)block[i*4+3];
    for (int i = 16; i < 64; i++)
        w[i] = sha256_ssig1(w[i-2]) + w[i-7] + sha256_ssig0(w[i-15]) + w[i-16];

    uint a=ctx->h[0], b=ctx->h[1], c=ctx->h[2], d=ctx->h[3];
    uint e=ctx->h[4], f=ctx->h[5], g=ctx->h[6], h=ctx->h[7];

    for (int i = 0; i < 64; i++) {
        uint t1 = h + sha256_bsig1(e) + sha256_ch(e,f,g) + K256[i] + w[i];
        uint t2 = sha256_bsig0(a) + sha256_maj(a,b,c);
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }

    ctx->h[0]+=a; ctx->h[1]+=b; ctx->h[2]+=c; ctx->h[3]+=d;
    ctx->h[4]+=e; ctx->h[5]+=f; ctx->h[6]+=g; ctx->h[7]+=h;
}

inline void sha256_init(SHA256Ctx* ctx) {
    ctx->h[0]=0x6a09e667u; ctx->h[1]=0xbb67ae85u;
    ctx->h[2]=0x3c6ef372u; ctx->h[3]=0xa54ff53au;
    ctx->h[4]=0x510e527fu; ctx->h[5]=0x9b05688cu;
    ctx->h[6]=0x1f83d9abu; ctx->h[7]=0x5be0cd19u;
    ctx->buf_len = 0; ctx->total_len = 0;
}

inline void sha256_update(SHA256Ctx* ctx, const uchar* data, uint len) {
    ctx->total_len += len;
    uint i = 0;
    if (ctx->buf_len > 0) {
        while (ctx->buf_len < 64 && i < len) ctx->buf[ctx->buf_len++] = data[i++];
        if (ctx->buf_len == 64) { sha256_compress(ctx, ctx->buf); ctx->buf_len = 0; }
    }
    while (i + 64 <= len) { sha256_compress(ctx, data + i); i += 64; }
    while (i < len) ctx->buf[ctx->buf_len++] = data[i++];
}

inline void sha256_final(SHA256Ctx* ctx, uchar out[32]) {
    ulong bits = ctx->total_len * 8;
    uchar pad = 0x80;
    sha256_update(ctx, &pad, 1);
    uchar zero = 0;
    while (ctx->buf_len != 56) sha256_update(ctx, &zero, 1);
    uchar len_bytes[8];
    for (int i = 0; i < 8; i++) len_bytes[i] = (uchar)(bits >> (56 - i*8));
    sha256_update(ctx, len_bytes, 8);

    for (int i = 0; i < 8; i++) {
        out[i*4+0] = (uchar)(ctx->h[i] >> 24);
        out[i*4+1] = (uchar)(ctx->h[i] >> 16);
        out[i*4+2] = (uchar)(ctx->h[i] >> 8);
        out[i*4+3] = (uchar)(ctx->h[i]);
    }
}

inline void hmac_sha256_impl(const uchar* key, uint key_len,
                              const uchar* msg, uint msg_len,
                              uchar out[32]) {
    uchar k_pad[64];
    // If key > 64 bytes, hash it
    uchar key_hash[32];
    if (key_len > 64) {
        SHA256Ctx kctx; sha256_init(&kctx);
        sha256_update(&kctx, key, key_len);
        sha256_final(&kctx, key_hash);
        key = key_hash; key_len = 32;
    }

    // ipad = 0x36, opad = 0x5c
    for (uint i = 0; i < 64; i++) k_pad[i] = (i < key_len ? key[i] : 0) ^ 0x36;

    SHA256Ctx ictx; sha256_init(&ictx);
    sha256_update(&ictx, k_pad, 64);
    sha256_update(&ictx, msg, msg_len);
    uchar inner[32];
    sha256_final(&ictx, inner);

    for (uint i = 0; i < 64; i++) k_pad[i] = (i < key_len ? key[i] : 0) ^ 0x5c;

    SHA256Ctx octx; sha256_init(&octx);
    sha256_update(&octx, k_pad, 64);
    sha256_update(&octx, inner, 32);
    sha256_final(&octx, out);
}

inline void rfc6979_nonce_impl(const Scalar* priv, const uchar msg_hash[32], Scalar* k_out) {
    uchar priv_bytes[32];
    scalar_to_bytes_impl(priv, priv_bytes);

    // V = 0x01 * 32, K = 0x00 * 32
    uchar V[32], K_[32];
    for (int i = 0; i < 32; i++) { V[i] = 0x01; K_[i] = 0x00; }

    // K = HMAC_K(V || 0x00 || x || h)
    uchar hmac_input[97];
    for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
    hmac_input[32] = 0x00;
    for (int i = 0; i < 32; i++) hmac_input[33+i] = priv_bytes[i];
    for (int i = 0; i < 32; i++) hmac_input[65+i] = msg_hash[i];
    hmac_sha256_impl(K_, 32, hmac_input, 97, K_);

    // V = HMAC_K(V)
    hmac_sha256_impl(K_, 32, V, 32, V);

    // K = HMAC_K(V || 0x01 || x || h)
    for (int i = 0; i < 32; i++) hmac_input[i] = V[i];
    hmac_input[32] = 0x01;
    hmac_sha256_impl(K_, 32, hmac_input, 97, K_);

    // V = HMAC_K(V)
    hmac_sha256_impl(K_, 32, V, 32, V);

    // Generate k
    for (int attempt = 0; attempt < 100; attempt++) {
        hmac_sha256_impl(K_, 32, V, 32, V);
        scalar_from_bytes_impl(V, k_out);
        if (!scalar_is_zero(k_out)) {
            Scalar order;
            order.limbs[0] = ORDER_N0; order.limbs[1] = ORDER_N1;
            order.limbs[2] = ORDER_N2; order.limbs[3] = ORDER_N3;
            if (!scalar_ge_impl(k_out, &order)) return;
        }
        // Retry: K = HMAC_K(V || 0x00), V = HMAC_K(V)
        uchar retry_input[33];
        for (int i = 0; i < 32; i++) retry_input[i] = V[i];
        retry_input[32] = 0x00;
        hmac_sha256_impl(K_, 32, retry_input, 33, K_);
        hmac_sha256_impl(K_, 32, V, 32, V);
    }
}

// =============================================================================
// LAYER 4: ECDSA Sign / Verify
// =============================================================================

typedef struct {
    Scalar r;
    Scalar s;
} ECDSASignature;

inline int ecdsa_sign_impl(const uchar msg_hash[32], const Scalar* priv, ECDSASignature* sig) {
    if (scalar_is_zero(priv)) return 0;

    Scalar z;
    scalar_from_bytes_impl(msg_hash, &z);

    Scalar k;
    rfc6979_nonce_impl(priv, msg_hash, &k);
    if (scalar_is_zero(&k)) return 0;

    JacobianPoint R;
    scalar_mul_generator_impl(&R, &k);
    if (point_is_infinity(&R)) return 0;

    // r = R.x mod n
    FieldElement z_inv, z_inv2, rx_aff;
    field_inv_impl(&z_inv, &R.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&rx_aff, &R.x, &z_inv2);

    uchar rx_bytes[32];
    field_to_bytes_impl(&rx_aff, rx_bytes);
    scalar_from_bytes_impl(rx_bytes, &sig->r);
    if (scalar_is_zero(&sig->r)) return 0;

    // s = k⁻¹ * (z + r*d) mod n
    Scalar k_inv;
    scalar_inverse_impl(&k, &k_inv);

    Scalar rd;
    scalar_mul_mod_n_impl(&sig->r, priv, &rd);

    Scalar z_plus_rd;
    scalar_add_mod_n_impl(&z, &rd, &z_plus_rd);

    scalar_mul_mod_n_impl(&k_inv, &z_plus_rd, &sig->s);
    if (scalar_is_zero(&sig->s)) return 0;

    // Low-S normalization
    if (!scalar_is_low_s_impl(&sig->s))
        scalar_negate_impl(&sig->s, &sig->s);

    return 1;
}

inline int ecdsa_verify_impl(const uchar msg_hash[32], const JacobianPoint* pubkey, const ECDSASignature* sig) {
    if (scalar_is_zero(&sig->r) || scalar_is_zero(&sig->s)) return 0;

    Scalar z;
    scalar_from_bytes_impl(msg_hash, &z);

    Scalar s_inv;
    scalar_inverse_impl(&sig->s, &s_inv);

    Scalar u1, u2;
    scalar_mul_mod_n_impl(&z, &s_inv, &u1);
    scalar_mul_mod_n_impl(&sig->r, &s_inv, &u2);

    AffinePoint G; get_generator(&G);
    JacobianPoint u1G, u2Q;
    scalar_mul_impl(&u1G, &u1, &G);

    // Convert pubkey to affine for scalar_mul_impl
    FieldElement pz_inv, pz_inv2, pz_inv3, px_aff, py_aff;
    field_inv_impl(&pz_inv, &pubkey->z);
    field_sqr_impl(&pz_inv2, &pz_inv);
    field_mul_impl(&pz_inv3, &pz_inv2, &pz_inv);
    field_mul_impl(&px_aff, &pubkey->x, &pz_inv2);
    field_mul_impl(&py_aff, &pubkey->y, &pz_inv3);

    AffinePoint pub_aff;
    pub_aff.x = px_aff; pub_aff.y = py_aff;

    scalar_mul_impl(&u2Q, &u2, &pub_aff);

    JacobianPoint R;
    point_add_impl(&R, &u1G, &u2Q);
    if (point_is_infinity(&R)) return 0;

    // Check R.x mod n == r
    FieldElement rz_inv, rz_inv2, rx_aff;
    field_inv_impl(&rz_inv, &R.z);
    field_sqr_impl(&rz_inv2, &rz_inv);
    field_mul_impl(&rx_aff, &R.x, &rz_inv2);

    uchar rx_bytes[32];
    field_to_bytes_impl(&rx_aff, rx_bytes);
    Scalar rx_scalar;
    scalar_from_bytes_impl(rx_bytes, &rx_scalar);

    return scalar_eq_impl(&rx_scalar, &sig->r);
}

// =============================================================================
// LAYER 5a: Tagged Hash + Schnorr BIP-340
// =============================================================================

inline void tagged_hash_impl(const uchar* tag, uint tag_len,
                              const uchar* data, uint data_len,
                              uchar out[32]) {
    // H_tag(msg) = SHA256(SHA256(tag) || SHA256(tag) || msg)
    uchar tag_hash[32];
    SHA256Ctx ctx;
    sha256_init(&ctx);
    sha256_update(&ctx, tag, tag_len);
    sha256_final(&ctx, tag_hash);

    sha256_init(&ctx);
    sha256_update(&ctx, tag_hash, 32);
    sha256_update(&ctx, tag_hash, 32);
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

// BIP-340 tagged hash midstate precomputation.
// SHA256(tag||tag) for each BIP-340 tag is exactly 64 bytes (one block).
// These midstates are the SHA-256 internal state after compressing that block,
// saving 2 compressions per tagged_hash call.
#define BIP340_TAG_AUX       0
#define BIP340_TAG_NONCE     1
#define BIP340_TAG_CHALLENGE 2

__constant uint BIP340_MIDSTATES[3][8] = {
    // "BIP0340/aux"
    {0x24dd3219U, 0x4eba7e70U, 0xca0fabb9U, 0x0fa3166dU,
     0x3afbe4b1U, 0x4c44df97U, 0x4aac2739U, 0x249e850aU},
    // "BIP0340/nonce"
    {0x46615b35U, 0xf4bfbff7U, 0x9f8dc671U, 0x83627ab3U,
     0x60217180U, 0x57358661U, 0x21a29e54U, 0x68b07b4cU},
    // "BIP0340/challenge"
    {0x9cecba11U, 0x23925381U, 0x11679112U, 0xd1627e0fU,
     0x97c87550U, 0x003cc765U, 0x90f61164U, 0x33e9b66aU},
};

inline void tagged_hash_fast_impl(int tag_idx,
                                  const uchar* data, uint data_len,
                                  uchar out[32]) {
    SHA256Ctx ctx;
    for (int i = 0; i < 8; i++) ctx.h[i] = BIP340_MIDSTATES[tag_idx][i];
    ctx.buf_len = 0;
    ctx.total_len = 64;  // 64 bytes already processed (tag_hash||tag_hash)
    sha256_update(&ctx, data, data_len);
    sha256_final(&ctx, out);
}

// Lift x to curve point with even Y
inline int lift_x_impl(const uchar x_bytes[32], JacobianPoint* p) {
    FieldElement x;
    for (int i = 0; i < 4; i++) {
        ulong limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++) limb = (limb << 8) | (ulong)x_bytes[base + j];
        x.limbs[i] = limb;
    }

    FieldElement x2, x3, y2, seven, y;
    field_sqr_impl(&x2, &x);
    field_mul_impl(&x3, &x2, &x);
    seven.limbs[0] = 7; seven.limbs[1] = 0; seven.limbs[2] = 0; seven.limbs[3] = 0;
    field_add_impl(&y2, &x3, &seven);

    field_sqrt_impl(&y2, &y);

    // Verify: y² == y2 (compare via normalized bytes to handle unreduced limbs)
    FieldElement y_check;
    field_sqr_impl(&y_check, &y);
    uchar yc_bytes[32], y2_bytes[32];
    field_to_bytes_impl(&y_check, yc_bytes);
    field_to_bytes_impl(&y2, y2_bytes);
    int valid = 1;
    for (int i = 0; i < 32; i++)
        if (yc_bytes[i] != y2_bytes[i]) valid = 0;
    if (!valid) return 0;

    // Ensure even Y
    uchar y_bytes[32];
    field_to_bytes_impl(&y, y_bytes);
    if (y_bytes[31] & 1) {
        field_neg_impl(&y, &y);
    }

    p->x = x; p->y = y;
    p->z.limbs[0] = 1; p->z.limbs[1] = 0; p->z.limbs[2] = 0; p->z.limbs[3] = 0;
    p->infinity = 0;
    return 1;
}

typedef struct {
    uchar r[32];
    Scalar s;
} SchnorrSignature;

inline int schnorr_sign_impl(const Scalar* priv, const uchar msg[32],
                               const uchar aux_rand[32], SchnorrSignature* sig) {
    if (scalar_is_zero(priv)) return 0;

    // P = d' * G
    JacobianPoint P;
    scalar_mul_generator_impl(&P, priv);
    if (point_is_infinity(&P)) return 0;

    // Convert to affine
    FieldElement z_inv, z_inv2, z_inv3, px, py;
    field_inv_impl(&z_inv, &P.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&z_inv3, &z_inv2, &z_inv);
    field_mul_impl(&px, &P.x, &z_inv2);
    field_mul_impl(&py, &P.y, &z_inv3);

    // If Y is odd, negate d
    uchar py_bytes[32];
    field_to_bytes_impl(&py, py_bytes);
    Scalar d;
    if (py_bytes[31] & 1) {
        scalar_negate_impl(priv, &d);
    } else { d = *priv; }

    uchar px_bytes[32];
    field_to_bytes_impl(&px, px_bytes);

    // t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    uchar t_hash[32];
    tagged_hash_fast_impl(BIP340_TAG_AUX, aux_rand, 32, t_hash);

    uchar d_bytes[32];
    scalar_to_bytes_impl(&d, d_bytes);

    uchar t[32];
    for (int i = 0; i < 32; i++) t[i] = d_bytes[i] ^ t_hash[i];

    // rand = tagged_hash("BIP0340/nonce", t || px || msg)
    uchar nonce_input[96];
    for (int i = 0; i < 32; i++) nonce_input[i] = t[i];
    for (int i = 0; i < 32; i++) nonce_input[32+i] = px_bytes[i];
    for (int i = 0; i < 32; i++) nonce_input[64+i] = msg[i];

    uchar rand_hash[32];
    tagged_hash_fast_impl(BIP340_TAG_NONCE, nonce_input, 96, rand_hash);

    Scalar k_prime;
    scalar_from_bytes_impl(rand_hash, &k_prime);
    if (scalar_is_zero(&k_prime)) return 0;

    // R = k' * G
    JacobianPoint R;
    scalar_mul_generator_impl(&R, &k_prime);

    FieldElement rz_inv, rz_inv2, rz_inv3, rx, ry;
    field_inv_impl(&rz_inv, &R.z);
    field_sqr_impl(&rz_inv2, &rz_inv);
    field_mul_impl(&rz_inv3, &rz_inv2, &rz_inv);
    field_mul_impl(&rx, &R.x, &rz_inv2);
    field_mul_impl(&ry, &R.y, &rz_inv3);

    uchar ry_bytes[32];
    field_to_bytes_impl(&ry, ry_bytes);
    Scalar k;
    if (ry_bytes[31] & 1) { scalar_negate_impl(&k_prime, &k); }
    else { k = k_prime; }

    field_to_bytes_impl(&rx, sig->r);

    // e = tagged_hash("BIP0340/challenge", R.x || px || msg) mod n
    uchar challenge_input[96];
    for (int i = 0; i < 32; i++) challenge_input[i] = sig->r[i];
    for (int i = 0; i < 32; i++) challenge_input[32+i] = px_bytes[i];
    for (int i = 0; i < 32; i++) challenge_input[64+i] = msg[i];

    uchar e_hash[32];
    tagged_hash_fast_impl(BIP340_TAG_CHALLENGE, challenge_input, 96, e_hash);

    Scalar e;
    scalar_from_bytes_impl(e_hash, &e);

    // s = k + e * d mod n
    Scalar ed;
    scalar_mul_mod_n_impl(&e, &d, &ed);
    scalar_add_mod_n_impl(&k, &ed, &sig->s);

    return 1;
}

inline int schnorr_verify_impl(const uchar pubkey_x[32], const uchar msg[32],
                                 const SchnorrSignature* sig) {
    if (scalar_is_zero(&sig->s)) return 0;

    JacobianPoint P;
    if (!lift_x_impl(pubkey_x, &P)) return 0;

    uchar challenge_input[96];
    for (int i = 0; i < 32; i++) challenge_input[i] = sig->r[i];
    for (int i = 0; i < 32; i++) challenge_input[32+i] = pubkey_x[i];
    for (int i = 0; i < 32; i++) challenge_input[64+i] = msg[i];

    uchar e_hash[32];
    tagged_hash_fast_impl(BIP340_TAG_CHALLENGE, challenge_input, 96, e_hash);

    Scalar e;
    scalar_from_bytes_impl(e_hash, &e);

    // R = s*G - e*P
    AffinePoint G; get_generator(&G);
    JacobianPoint sG, eP;
    scalar_mul_glv_impl(&sG, &sig->s, &G);

    // Convert P to affine for scalar_mul_impl
    FieldElement pz_inv, pz_inv2, pz_inv3, px_aff, py_aff;
    field_inv_impl(&pz_inv, &P.z);
    field_sqr_impl(&pz_inv2, &pz_inv);
    field_mul_impl(&pz_inv3, &pz_inv2, &pz_inv);
    field_mul_impl(&px_aff, &P.x, &pz_inv2);
    field_mul_impl(&py_aff, &P.y, &pz_inv3);
    AffinePoint p_aff; p_aff.x = px_aff; p_aff.y = py_aff;

    scalar_mul_glv_impl(&eP, &e, &p_aff);

    point_negate_y(&eP);

    JacobianPoint Rpt;
    point_add_impl(&Rpt, &sG, &eP);
    if (point_is_infinity(&Rpt)) return 0;

    FieldElement rz_inv, rz_inv2, rz_inv3, rx_aff, ry_aff;
    field_inv_impl(&rz_inv, &Rpt.z);
    field_sqr_impl(&rz_inv2, &rz_inv);
    field_mul_impl(&rz_inv3, &rz_inv2, &rz_inv);
    field_mul_impl(&rx_aff, &Rpt.x, &rz_inv2);
    field_mul_impl(&ry_aff, &Rpt.y, &rz_inv3);

    uchar ry_bytes[32];
    field_to_bytes_impl(&ry_aff, ry_bytes);
    if (ry_bytes[31] & 1) return 0; // must have even Y

    uchar rx_bytes[32];
    field_to_bytes_impl(&rx_aff, rx_bytes);
    for (int i = 0; i < 32; i++)
        if (rx_bytes[i] != sig->r[i]) return 0;

    return 1;
}

// =============================================================================
// LAYER 5b: ECDH
// =============================================================================

inline int ecdh_compute_raw_impl(const Scalar* priv, const AffinePoint* peer, uchar out[32]) {
    JacobianPoint shared;
    scalar_mul_glv_impl(&shared, priv, peer);
    if (point_is_infinity(&shared)) return 0;

    FieldElement z_inv, z_inv2, x_aff;
    field_inv_impl(&z_inv, &shared.z);
    field_sqr_impl(&z_inv2, &z_inv);
    field_mul_impl(&x_aff, &shared.x, &z_inv2);
    field_to_bytes_impl(&x_aff, out);
    return 1;
}

inline int ecdh_compute_xonly_impl(const Scalar* priv, const AffinePoint* peer, uchar out[32]) {
    uchar x_bytes[32];
    if (!ecdh_compute_raw_impl(priv, peer, x_bytes)) return 0;

    SHA256Ctx ctx; sha256_init(&ctx);
    sha256_update(&ctx, x_bytes, 32);
    sha256_final(&ctx, out);
    return 1;
}

inline int ecdh_compute_impl(const Scalar* priv, const AffinePoint* peer, uchar out[32]) {
    uchar x_bytes[32];
    if (!ecdh_compute_raw_impl(priv, peer, x_bytes)) return 0;

    SHA256Ctx ctx; sha256_init(&ctx);
    uchar prefix = 0x02;
    sha256_update(&ctx, &prefix, 1);
    sha256_update(&ctx, x_bytes, 32);
    sha256_final(&ctx, out);
    return 1;
}

// =============================================================================
// LAYER 5c: Key Recovery
// =============================================================================

typedef struct {
    ECDSASignature sig;
    int recid;
} RecoverableSignature;

inline int ecdsa_sign_recoverable_impl(const uchar msg_hash[32], const Scalar* priv,
                                         RecoverableSignature* rsig) {
    if (scalar_is_zero(priv)) return 0;

    Scalar z;
    scalar_from_bytes_impl(msg_hash, &z);

    Scalar k;
    rfc6979_nonce_impl(priv, msg_hash, &k);
    if (scalar_is_zero(&k)) return 0;

    JacobianPoint R;
    scalar_mul_generator_impl(&R, &k);
    if (point_is_infinity(&R)) return 0;

    FieldElement zinv, zinv2, zinv3, rx_aff, ry_aff;
    field_inv_impl(&zinv, &R.z);
    field_sqr_impl(&zinv2, &zinv);
    field_mul_impl(&zinv3, &zinv2, &zinv);
    field_mul_impl(&rx_aff, &R.x, &zinv2);
    field_mul_impl(&ry_aff, &R.y, &zinv3);

    uchar rx_bytes[32];
    field_to_bytes_impl(&rx_aff, rx_bytes);
    scalar_from_bytes_impl(rx_bytes, &rsig->sig.r);
    if (scalar_is_zero(&rsig->sig.r)) return 0;

    int recid = 0;
    uchar ry_bytes[32];
    field_to_bytes_impl(&ry_aff, ry_bytes);
    if (ry_bytes[31] & 1) recid |= 1;

    // Check overflow (R.x >= n)
    uchar order_be[32] = {
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
        0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
        0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
        0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
    };
    int overflow = 0;
    for (int i = 0; i < 32; i++) {
        if (rx_bytes[i] < order_be[i]) break;
        if (rx_bytes[i] > order_be[i]) { overflow = 1; break; }
    }
    if (overflow) recid |= 2;

    // s = k⁻¹(z + r*d) mod n
    Scalar k_inv;
    scalar_inverse_impl(&k, &k_inv);
    Scalar rd;
    scalar_mul_mod_n_impl(&rsig->sig.r, priv, &rd);
    Scalar z_plus_rd;
    scalar_add_mod_n_impl(&z, &rd, &z_plus_rd);
    scalar_mul_mod_n_impl(&k_inv, &z_plus_rd, &rsig->sig.s);
    if (scalar_is_zero(&rsig->sig.s)) return 0;

    if (!scalar_is_low_s_impl(&rsig->sig.s)) {
        scalar_negate_impl(&rsig->sig.s, &rsig->sig.s);
        recid ^= 1;
    }

    rsig->recid = recid;
    return 1;
}

// Lift x as FieldElement with parity control
inline int lift_x_field_impl(const FieldElement* x_fe, int parity, JacobianPoint* p) {
    FieldElement x2, x3, y2, seven, y;
    field_sqr_impl(&x2, x_fe);
    field_mul_impl(&x3, &x2, x_fe);
    seven.limbs[0]=7; seven.limbs[1]=0; seven.limbs[2]=0; seven.limbs[3]=0;
    field_add_impl(&y2, &x3, &seven);
    field_sqrt_impl(&y2, &y);

    // Verify: y² == y2 (compare via normalized bytes to handle unreduced limbs)
    FieldElement y_check;
    field_sqr_impl(&y_check, &y);
    uchar yc_bytes2[32], y2_bytes2[32];
    field_to_bytes_impl(&y_check, yc_bytes2);
    field_to_bytes_impl(&y2, y2_bytes2);
    int valid = 1;
    for (int i = 0; i < 32; i++)
        if (yc_bytes2[i] != y2_bytes2[i]) valid = 0;
    if (!valid) return 0;

    uchar y_bytes[32];
    field_to_bytes_impl(&y, y_bytes);
    int y_is_odd = (y_bytes[31] & 1) != 0;
    if ((parity != 0) != y_is_odd) {
        field_neg_impl(&y, &y);
    }

    p->x = *x_fe; p->y = y;
    p->z.limbs[0]=1; p->z.limbs[1]=0; p->z.limbs[2]=0; p->z.limbs[3]=0;
    p->infinity = 0;
    return 1;
}

inline int ecdsa_recover_impl(const uchar msg_hash[32], const ECDSASignature* sig,
                                int recid, JacobianPoint* Q) {
    if (recid < 0 || recid > 3) return 0;
    if (scalar_is_zero(&sig->r) || scalar_is_zero(&sig->s)) return 0;

    // Reconstruct R.x
    FieldElement rx_fe;
    uchar r_bytes[32];
    scalar_to_bytes_impl(&sig->r, r_bytes);
    for (int i = 0; i < 4; i++) {
        ulong limb = 0;
        int base = (3 - i) * 8;
        for (int j = 0; j < 8; j++) limb = (limb << 8) | (ulong)r_bytes[base + j];
        rx_fe.limbs[i] = limb;
    }

    if (recid & 2) {
        FieldElement n_fe;
        n_fe.limbs[0] = ORDER_N0; n_fe.limbs[1] = ORDER_N1;
        n_fe.limbs[2] = ORDER_N2; n_fe.limbs[3] = ORDER_N3;
        field_add_impl(&rx_fe, &rx_fe, &n_fe);
    }

    int y_parity = recid & 1;
    JacobianPoint Rpt;
    if (!lift_x_field_impl(&rx_fe, y_parity, &Rpt)) return 0;

    // Q = r⁻¹ * (s*R - z*G)
    Scalar z;
    scalar_from_bytes_impl(msg_hash, &z);

    Scalar r_inv;
    scalar_inverse_impl(&sig->r, &r_inv);

    // s*R  (need affine for scalar_mul_impl)
    FieldElement pz_inv, pz_inv2, pz_inv3;
    field_inv_impl(&pz_inv, &Rpt.z);
    field_sqr_impl(&pz_inv2, &pz_inv);
    field_mul_impl(&pz_inv3, &pz_inv2, &pz_inv);
    AffinePoint r_aff;
    field_mul_impl(&r_aff.x, &Rpt.x, &pz_inv2);
    field_mul_impl(&r_aff.y, &Rpt.y, &pz_inv3);

    JacobianPoint sR;
    scalar_mul_glv_impl(&sR, &sig->s, &r_aff);

    AffinePoint G; get_generator(&G);
    JacobianPoint zG;
    scalar_mul_glv_impl(&zG, &z, &G);

    point_negate_y(&zG);
    JacobianPoint sR_minus_zG;
    point_add_impl(&sR_minus_zG, &sR, &zG);

    // Convert to affine for final scalar_mul
    FieldElement qz_inv, qz_inv2, qz_inv3;
    field_inv_impl(&qz_inv, &sR_minus_zG.z);
    field_sqr_impl(&qz_inv2, &qz_inv);
    field_mul_impl(&qz_inv3, &qz_inv2, &qz_inv);
    AffinePoint diff_aff;
    field_mul_impl(&diff_aff.x, &sR_minus_zG.x, &qz_inv2);
    field_mul_impl(&diff_aff.y, &sR_minus_zG.y, &qz_inv3);

    scalar_mul_glv_impl(Q, &r_inv, &diff_aff);

    if (point_is_infinity(Q)) return 0;
    return 1;
}

// =============================================================================
// LAYER 5d: MSM (Multi-Scalar Multiplication)
// =============================================================================

// Extract c-bit window from scalar
inline uint scalar_get_window_impl(const Scalar* s, int window_idx, int c) {
    int bit_offset = window_idx * c;
    int limb_idx = bit_offset / 64;
    int bit_idx = bit_offset % 64;
    if (limb_idx >= 4) return 0;

    uint val = (uint)((s->limbs[limb_idx] >> bit_idx) & ((1UL << c) - 1));
    int bits_from_first = 64 - bit_idx;
    if (bits_from_first < c && limb_idx + 1 < 4) {
        int remaining = c - bits_from_first;
        val |= (uint)(s->limbs[limb_idx+1] & ((1UL << remaining) - 1)) << bits_from_first;
    }
    return val;
}

// Naive MSM: sum of individual scalar multiplications
inline void msm_naive_impl(const Scalar* scalars, const AffinePoint* points,
                             int n, JacobianPoint* result) {
    point_set_infinity(result);
    for (int i = 0; i < n; i++) {
        if (scalar_is_zero(&scalars[i])) continue;
        JacobianPoint tmp;
        scalar_mul_glv_impl(&tmp, &scalars[i], &points[i]);
        if (point_is_infinity(result)) {
            *result = tmp;
        } else {
            JacobianPoint sum;
            point_add_impl(&sum, result, &tmp);
            *result = sum;
        }
    }
}

// Pippenger bucket MSM
inline void msm_pippenger_impl(const Scalar* scalars, const AffinePoint* points,
                                 int n, JacobianPoint* result,
                                 JacobianPoint* buckets, int c) {
    int num_buckets = 1 << c;
    int num_windows = (256 + c - 1) / c;

    point_set_infinity(result);

    for (int w = num_windows - 1; w >= 0; w--) {
        if (!point_is_infinity(result)) {
            for (int d = 0; d < c; d++) {
                JacobianPoint dbl;
                point_double_impl(&dbl, result);
                *result = dbl;
            }
        }

        for (int b = 0; b < num_buckets; b++) point_set_infinity(&buckets[b]);

        for (int i = 0; i < n; i++) {
            uint digit = scalar_get_window_impl(&scalars[i], w, c);
            if (digit == 0) continue;
            if (point_is_infinity(&buckets[digit])) {
                point_from_affine(&buckets[digit], &points[i]);
            } else {
                JacobianPoint sum;
                point_add_mixed_impl(&sum, &buckets[digit], &points[i]);
                buckets[digit] = sum;
            }
        }

        JacobianPoint running_sum, partial_sum;
        point_set_infinity(&running_sum);
        point_set_infinity(&partial_sum);

        for (int b = num_buckets - 1; b >= 1; b--) {
            if (!point_is_infinity(&buckets[b])) {
                if (point_is_infinity(&running_sum)) {
                    running_sum = buckets[b];
                } else {
                    JacobianPoint sum;
                    point_add_impl(&sum, &running_sum, &buckets[b]);
                    running_sum = sum;
                }
            }
            if (!point_is_infinity(&running_sum)) {
                if (point_is_infinity(&partial_sum)) {
                    partial_sum = running_sum;
                } else {
                    JacobianPoint sum;
                    point_add_impl(&sum, &partial_sum, &running_sum);
                    partial_sum = sum;
                }
            }
        }

        if (!point_is_infinity(&partial_sum)) {
            if (point_is_infinity(result)) {
                *result = partial_sum;
            } else {
                JacobianPoint sum;
                point_add_impl(&sum, result, &partial_sum);
                *result = sum;
            }
        }
    }
}

// =============================================================================
// OpenCL Dispatch Kernels — Extended Operations
// =============================================================================

__kernel void ecdsa_sign(
    __global const uchar* msg_hashes,       // n * 32 bytes
    __global const Scalar* private_keys,
    __global ECDSASignature* signatures,
    __global int* success_flags,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uchar msg[32];
    for (int i = 0; i < 32; i++) msg[i] = msg_hashes[gid * 32 + i];

    Scalar priv = private_keys[gid];
    ECDSASignature sig;
    success_flags[gid] = ecdsa_sign_impl(msg, &priv, &sig);
    signatures[gid] = sig;
}

__kernel void ecdsa_verify(
    __global const uchar* msg_hashes,
    __global const JacobianPoint* pubkeys,
    __global const ECDSASignature* signatures,
    __global int* results,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uchar msg[32];
    for (int i = 0; i < 32; i++) msg[i] = msg_hashes[gid * 32 + i];

    JacobianPoint pub = pubkeys[gid];
    ECDSASignature sig = signatures[gid];
    results[gid] = ecdsa_verify_impl(msg, &pub, &sig);
}

__kernel void schnorr_sign(
    __global const uchar* messages,
    __global const Scalar* private_keys,
    __global const uchar* aux_rands,
    __global SchnorrSignature* signatures,
    __global int* success_flags,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uchar msg[32], aux[32];
    for (int i = 0; i < 32; i++) { msg[i] = messages[gid*32+i]; aux[i] = aux_rands[gid*32+i]; }

    Scalar priv = private_keys[gid];
    SchnorrSignature sig;
    success_flags[gid] = schnorr_sign_impl(&priv, msg, aux, &sig);
    signatures[gid] = sig;
}

__kernel void schnorr_verify(
    __global const uchar* pubkeys_x,
    __global const uchar* messages,
    __global const SchnorrSignature* signatures,
    __global int* results,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    uchar pk[32], msg[32];
    for (int i = 0; i < 32; i++) { pk[i] = pubkeys_x[gid*32+i]; msg[i] = messages[gid*32+i]; }

    SchnorrSignature sig = signatures[gid];
    results[gid] = schnorr_verify_impl(pk, msg, &sig);
}

__kernel void generator_mul_windowed(
    __global const Scalar* scalars,
    __global JacobianPoint* results,
    const uint count
) {
    uint gid = get_global_id(0);
    if (gid >= count) return;

    Scalar k = scalars[gid];
    JacobianPoint r;
    scalar_mul_generator_windowed_impl(&r, &k);
    results[gid] = r;
}
