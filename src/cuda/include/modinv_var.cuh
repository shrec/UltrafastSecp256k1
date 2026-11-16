// =============================================================================
// Variable-time modular inverse for secp256k1 (CUDA device functions)
// =============================================================================
//   field_inv_var(a, r)    r = a^-1 mod p   (field prime)
//   scalar_inv_var(a, r)   r = a^-1 mod n   (group order)
//   jacobian_to_affine_var(p, x, y)   affine form of a PUBLIC point
//
// !! VARIABLE TIME -- PUBLIC DATA ONLY !!
// The number of loop iterations depends on the VALUE being inverted. Never
// call these from a constant-time / secret-bearing path: not on a private key,
// a nonce, a signing share, an ECDH scalar, and not on the Z coordinate of a
// point produced by a secret scalar multiplication (k*G, sk*G, sk*P).
// Secret paths keep using field_inv / ct::field_inv / ct::scalar_inverse.
// The ct/ headers do see these functions (they include secp256k1.cuh, which
// includes this header), but nothing in namespace ct may call them: the
// deleted declarations at the end of this file make an unqualified call from
// ct code a compile error.
// Legitimate callers: signature and proof verification (ecdsa_verify,
// schnorr_verify, ecdsa_recover, the ECDSA SNARK witness, DLEQ / FROST partial
// / Bulletproof range verification) and affine conversion of public points.
// A new caller needs its own proof that no caller chain brings a secret, or a
// value derived from one, into the inversion.
//
// Contract (same results as field_inv / scalar_inverse, bit for bit):
//   - input: any 256-bit value, reduced or not; a == r aliasing is allowed
//   - output: canonical, in [0, modulus); 0 when the input is 0 mod modulus
//   - always terminates: at most MODINV30_MAX_ITERS outer iterations, each of
//     at most 30 inner passes; if the cap were ever reached the result is 0
//     (fail closed) -- unreachable for a 256-bit input, see MODINV30_MAX_ITERS
//
// Algorithm: Bernstein-Yang safegcd ("Fast constant-time gcd computation and
// modular inversion", 2019), variable-time form, 30 divsteps per batch on
// 9 signed 30-bit limbs with 64-bit accumulators.
// Provenance: port of this library's own CPU code (src/cpu/src/field.cpp
// namespace field_safegcd30, src/cpu/src/scalar.cpp namespace
// scalar_safegcd30), which is adapted from bitcoin-core/secp256k1
// secp256k1_modinv32_var (src/modinv32_impl.h):
//   Copyright (c) 2020 Peter Dettman
//   Distributed under the MIT software license, see
//   https://www.opensource.org/licenses/mit-license.php.
// Differences from the CPU code, all for the GPU:
//   - hard cap on the outer loop (the CPU code loops until g == 0)
//   - table-free 6-bit inner step w = f*g*(f*f-2) (as in the CPU 62-bit
//     divsteps) instead of the 128-byte table: no memory load per pass
//   - f and g always keep 9 limbs: every array index is a compile-time
//     constant, so the whole state stays in registers
//   - the input is reduced first, so unreduced values are accepted
//   - signed 32x32->64 products are forced to one mul.wide.s32 each
//
// Not available as SafeGCD in two configurations, where both functions simply
// forward to the existing Fermat inverses: 32-bit limbs (different element
// layout) and, for the field only, Montgomery mode (field_inv there returns
// the Montgomery-domain result, which this code does not produce).
// =============================================================================

#pragma once
#include "secp256k1.cuh"

namespace secp256k1 {
namespace cuda {

#if SECP256K1_CUDA_LIMBS_32

__device__ inline void field_inv_var(const FieldElement* a, FieldElement* r) { field_inv(a, r); }
__device__ inline void scalar_inv_var(const Scalar* a, Scalar* r) { scalar_inverse(a, r); }

#else // 4 x 64-bit limbs

namespace modinv30 {

static_assert((-1 >> 1) == -1, "modinv30 needs an arithmetic right shift on int32_t");
static_assert((int64_t(-1) >> 30) == -1, "modinv30 needs an arithmetic right shift on int64_t");

// 25 * 30 = 750 divsteps. 741 divsteps are proven sufficient for any 256-bit
// input with delta starting at 1 (Bernstein-Yang, Theorem 11.2); 724 by
// sipa/safegcd-bounds. bitcoin-core/secp256k1 modinv32_var asserts the same 25.
static constexpr int MODINV30_MAX_ITERS = 25;
static_assert(MODINV30_MAX_ITERS * 30 >= 741,
              "modinv30: the iteration cap must cover the 741 divsteps proven sufficient");

// Modulus as 9 signed 30-bit limbs (value = sum limb(i) * 2^(30*i)), its
// inverse mod 2^30, and the same modulus as 4 x 64-bit limbs. Constants are
// functions so that nvcc folds them (zero limbs vanish, powers of two become
// shifts) instead of loading a __constant__ array.
struct ModP {   // p = 2^256 - 2^32 - 977
    __host__ __device__ static constexpr int32_t limb(int i) {
        return i == 0 ? -0x3D1 : i == 1 ? -4 : i == 8 ? 65536 : 0;
    }
    __host__ __device__ static constexpr uint32_t inv30() { return 0x2DDACACFu; }
    __host__ __device__ static constexpr uint64_t m64(int i) {
        return i == 0 ? 0xFFFFFFFEFFFFFC2FULL : 0xFFFFFFFFFFFFFFFFULL;
    }
};
struct ModN {   // n = group order
    __host__ __device__ static constexpr int32_t limb(int i) {
        return i == 0 ? 0x10364141 : i == 1 ? 0x3F497A33 : i == 2 ? 0x348A03BB
             : i == 3 ? 0x2BB739AB : i == 4 ? -0x146 : i == 8 ? 65536 : 0;
    }
    __host__ __device__ static constexpr uint32_t inv30() { return 0x2A774EC1u; }
    __host__ __device__ static constexpr uint64_t m64(int i) {
        return i == 0 ? 0xBFD25E8CD0364141ULL : i == 1 ? 0xBAAEDCE6AF48A03BULL
             : i == 2 ? 0xFFFFFFFFFFFFFFFEULL : 0xFFFFFFFFFFFFFFFFULL;
    }
};

// Word w (0..4) of sum(limb(i) * 2^(30*i)), accumulated as a 320-bit
// two's-complement value (the limbs are signed). Compile-time only.
template <class M>
__host__ __device__ constexpr uint64_t limb_sum_word(int w) {
    uint64_t acc[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 9; ++i) {
        const uint64_t v = (uint64_t)(int64_t)M::limb(i);        // sign-extended
        const uint64_t ext = M::limb(i) < 0 ? ~0ULL : 0ULL;
        const int word = (30 * i) / 64, bit = (30 * i) % 64;
        uint64_t carry = 0;
        for (int k = word; k < 5; ++k) {
            const uint64_t hi = k == word ? v : ext;
            const uint64_t lo = k == word ? 0ULL : (k == word + 1 ? v : ext);
            const uint64_t t = (hi << bit) | (bit != 0 ? lo >> (64 - bit) : 0ULL);
            const uint64_t s = acc[k] + t;
            const uint64_t s2 = s + carry;
            carry = (uint64_t)(s < t) | (uint64_t)(s2 < s);
            acc[k] = s2;
        }
    }
    return acc[w];
}

// Constant self-checks: modulus * inv30 == 1 (mod 2^30), and the signed-30
// form equals the 4 x 64-bit form exactly: all nine limbs against all four
// words, with nothing left above bit 255.
template <class M>
__host__ __device__ constexpr bool mod_constants_ok() {
    return ((M::inv30() * (uint32_t)M::limb(0)) & 0x3FFFFFFFu) == 1u
        && limb_sum_word<M>(0) == M::m64(0) && limb_sum_word<M>(1) == M::m64(1)
        && limb_sum_word<M>(2) == M::m64(2) && limb_sum_word<M>(3) == M::m64(3)
        && limb_sum_word<M>(4) == 0;
}
static_assert(mod_constants_ok<ModP>(), "modinv30: bad field modulus constants");
static_assert(mod_constants_ok<ModN>(), "modinv30: bad group order constants");

// acc + a*b, signed 32x32->64. Written as PTX because the compiler otherwise
// widens the operands and emits a 64x64 multiply (three multiplies per product).
__device__ __forceinline__ int64_t mul_add_wide(int64_t acc, int32_t a, int32_t b) {
#if SECP256K1_USE_PTX
    int64_t p;
    asm volatile("mul.wide.s32 %0, %1, %2;" : "=l"(p) : "r"(a), "r"(b));
    return acc + p;
#else
    return acc + (int64_t)a * b;
#endif
}

// 30 divsteps on the low words of f and g. Returns the new eta (= -delta) and
// the transition matrix (u, v, q, r), scaled by 2^30.
// Bounded for ANY f, g: each pass after the first removes at least one of the
// 30 steps (f is forced odd by the caller, and g is odd after the shift).
__device__ __forceinline__ int32_t divsteps_30_var(int32_t eta, uint32_t f, uint32_t g,
                                                   int32_t& tu, int32_t& tv,
                                                   int32_t& tq, int32_t& tr) {
    uint32_t u = 1, v = 0, q = 0, r = 1;
    int i = 30;
    for (;;) {
        // strip zero bits of g; the sentinel at bit i stops the count at i
        const int zeros = __ffs((int)(g | (0xFFFFFFFFu << i))) - 1;
        g >>= zeros; u <<= zeros; v <<= zeros; eta -= zeros; i -= zeros;
        if (i == 0) break;
        // eta < 0: (f, g) <- (g, -f), same for the matrix rows (selects, no branch)
        const bool swap = eta < 0;
        const uint32_t f0 = f, u0 = u, v0 = v;
        eta = swap ? -eta : eta;
        f = swap ? g : f0;   g = swap ? (0u - f0) : g;
        u = swap ? q : u0;   q = swap ? (0u - u0) : q;
        v = swap ? r : v0;   r = swap ? (0u - v0) : r;
        // cancel up to 6 low bits of g with a multiple of f (f odd):
        // f * (f*(f*f-2)) == -1 (mod 64)
        const int limit = (eta + 1) < i ? (eta + 1) : i;
        const uint32_t m = (0xFFFFFFFFu >> (32 - limit)) & 63u;
        const uint32_t w = ((f * g) * (f * f - 2u)) & m;
        g += f * w; q += u * w; r += v * w;
    }
    tu = (int32_t)u; tv = (int32_t)v; tq = (int32_t)q; tr = (int32_t)r;
    return eta;
}

// r4 = (a4 mod M)^-1 mod M, canonical; 0 if a4 == 0 mod M or if max_iters is
// reached. Returns the number of outer iterations used. a4 == r4 is allowed.
// max_iters is a parameter only so that tests can exercise the fail-closed
// path; production callers pass MODINV30_MAX_ITERS.
template <class M>
__device__ __forceinline__ int inverse_var(const uint64_t* a4, uint64_t* r4, int max_iters) {
    const uint32_t M30 = 0x3FFFFFFFu;

    // x = a >= M ? a - M : a   (M < 2^256 < 2M, so one subtraction is enough)
    uint64_t x0 = a4[0], x1 = a4[1], x2 = a4[2], x3 = a4[3];
    {
        uint64_t borrow = 0;
        const uint64_t t0 = sub_cc(x0, M::m64(0), borrow);
        const uint64_t t1 = sub_cc(x1, M::m64(1), borrow);
        const uint64_t t2 = sub_cc(x2, M::m64(2), borrow);
        const uint64_t t3 = sub_cc(x3, M::m64(3), borrow);
        const uint64_t keep = 0ULL - borrow;          // all-ones when a < M
        x0 = (x0 & keep) | (t0 & ~keep);
        x1 = (x1 & keep) | (t1 & ~keep);
        x2 = (x2 & keep) | (t2 & ~keep);
        x3 = (x3 & keep) | (t3 & ~keep);
    }

    // d = 0, e = 1, f = M, g = x
    int32_t d[9], e[9], f[9], g[9];
    #pragma unroll
    for (int i = 0; i < 9; ++i) { d[i] = 0; e[i] = 0; f[i] = M::limb(i); }
    e[0] = 1;
    g[0] = (int32_t)((uint32_t)x0 & M30);
    g[1] = (int32_t)((uint32_t)(x0 >> 30) & M30);
    g[2] = (int32_t)((uint32_t)((x0 >> 60) | (x1 << 4)) & M30);
    g[3] = (int32_t)((uint32_t)(x1 >> 26) & M30);
    g[4] = (int32_t)((uint32_t)((x1 >> 56) | (x2 << 8)) & M30);
    g[5] = (int32_t)((uint32_t)(x2 >> 22) & M30);
    g[6] = (int32_t)((uint32_t)((x2 >> 52) | (x3 << 12)) & M30);
    g[7] = (int32_t)((uint32_t)(x3 >> 18) & M30);
    g[8] = (int32_t)(x3 >> 48);

    int32_t eta = -1;
    bool done = false;
    int it = 0;
    while (it < max_iters) {
        ++it;
        int32_t u, v, q, r;
        // "| 1u": f is always odd, the OR only makes the inner loop bound
        // independent of that invariant
        eta = divsteps_30_var(eta, (uint32_t)f[0] | 1u, (uint32_t)g[0], u, v, q, r);

        // (d, e) <- (matrix / 2^30) * (d, e) mod M
        {
            const int32_t sd = d[8] >> 31, se = e[8] >> 31;
            int32_t md = (u & sd) + (v & se);
            int32_t me = (q & sd) + (r & se);
            int64_t cd = mul_add_wide(mul_add_wide(0, u, d[0]), v, e[0]);
            int64_t ce = mul_add_wide(mul_add_wide(0, q, d[0]), r, e[0]);
            md -= (int32_t)((M::inv30() * (uint32_t)cd + (uint32_t)md) & M30);
            me -= (int32_t)((M::inv30() * (uint32_t)ce + (uint32_t)me) & M30);
            cd += (int64_t)M::limb(0) * md;
            ce += (int64_t)M::limb(0) * me;
            cd >>= 30; ce >>= 30;                    // low 30 bits are zero
            #pragma unroll
            for (int i = 1; i < 9; ++i) {
                cd = mul_add_wide(mul_add_wide(cd, u, d[i]), v, e[i]);
                ce = mul_add_wide(mul_add_wide(ce, q, d[i]), r, e[i]);
                if (M::limb(i) != 0) {               // compile-time constant
                    cd += (int64_t)M::limb(i) * md;
                    ce += (int64_t)M::limb(i) * me;
                }
                d[i - 1] = (int32_t)((uint32_t)cd & M30); cd >>= 30;
                e[i - 1] = (int32_t)((uint32_t)ce & M30); ce >>= 30;
            }
            d[8] = (int32_t)cd; e[8] = (int32_t)ce;
        }

        // (f, g) <- (matrix / 2^30) * (f, g), exact; g_nonzero = OR of the new g
        int32_t g_nonzero;
        {
            int64_t cf = mul_add_wide(mul_add_wide(0, u, f[0]), v, g[0]);
            int64_t cg = mul_add_wide(mul_add_wide(0, q, f[0]), r, g[0]);
            cf >>= 30; cg >>= 30;                    // low 30 bits are zero
            g_nonzero = 0;
            #pragma unroll
            for (int i = 1; i < 9; ++i) {
                cf = mul_add_wide(mul_add_wide(cf, u, f[i]), v, g[i]);
                cg = mul_add_wide(mul_add_wide(cg, q, f[i]), r, g[i]);
                f[i - 1] = (int32_t)((uint32_t)cf & M30); cf >>= 30;
                g[i - 1] = (int32_t)((uint32_t)cg & M30); cg >>= 30;
                g_nonzero |= g[i - 1];
            }
            f[8] = (int32_t)cf; g[8] = (int32_t)cg;
            g_nonzero |= g[8];
        }
        if (g_nonzero == 0) { done = true; break; }
    }

    // normalize d to [0, M), negating first when f == -1
    {
        int32_t cond_add = d[8] >> 31;
        #pragma unroll
        for (int i = 0; i < 9; ++i) d[i] += M::limb(i) & cond_add;
        const int32_t cond_neg = f[8] >> 31;
        #pragma unroll
        for (int i = 0; i < 9; ++i) d[i] = (d[i] ^ cond_neg) - cond_neg;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { d[i + 1] += d[i] >> 30; d[i] &= (int32_t)M30; }
        cond_add = d[8] >> 31;
        #pragma unroll
        for (int i = 0; i < 9; ++i) d[i] += M::limb(i) & cond_add;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { d[i + 1] += d[i] >> 30; d[i] &= (int32_t)M30; }
    }

    const uint64_t ok = 0ULL - (uint64_t)done;      // cap reached -> 0 (fail closed)
    const uint64_t d0 = (uint32_t)d[0], d1 = (uint32_t)d[1], d2 = (uint32_t)d[2],
                   d3 = (uint32_t)d[3], d4 = (uint32_t)d[4], d5 = (uint32_t)d[5],
                   d6 = (uint32_t)d[6], d7 = (uint32_t)d[7], d8 = (uint32_t)d[8];
    r4[0] = ok & (d0 | (d1 << 30) | (d2 << 60));
    r4[1] = ok & ((d2 >> 4) | (d3 << 26) | (d4 << 56));
    r4[2] = ok & ((d4 >> 8) | (d5 << 22) | (d6 << 52));
    r4[3] = ok & ((d6 >> 12) | (d7 << 18) | (d8 << 48));
    return it;
}

} // namespace modinv30

// r = a^-1 mod p. VARIABLE TIME, PUBLIC DATA ONLY (see the file header).
__device__ inline void field_inv_var(const FieldElement* a, FieldElement* r) {
#if SECP256K1_CUDA_USE_MONTGOMERY
    field_inv(a, r);
#else
    (void)modinv30::inverse_var<modinv30::ModP>(a->limbs, r->limbs, modinv30::MODINV30_MAX_ITERS);
#endif
}

// r = a^-1 mod n. VARIABLE TIME, PUBLIC DATA ONLY (see the file header).
__device__ inline void scalar_inv_var(const Scalar* a, Scalar* r) {
    (void)modinv30::inverse_var<modinv30::ModN>(a->limbs, r->limbs, modinv30::MODINV30_MAX_ITERS);
}

#endif // SECP256K1_CUDA_LIMBS_32

// Jacobian (X, Y, Z) -> affine (x, y) = (X/Z^2, Y/Z^3). Same contract and same
// results as jacobian_to_affine: for infinity it sets out to (0, 0) and returns
// false.
// VARIABLE TIME, PUBLIC POINTS ONLY (see the file header). The point must be
// public AND must not be the output of a secret scalar multiplication (k*G,
// sk*G, sk*P): the Z of such a point depends on the secret. Called from
// verification code only. Every shared helper (jacobian_to_affine and the
// point_to_compressed family) and every secret-bearing path keeps field_inv.
__device__ inline bool jacobian_to_affine_var(const JacobianPoint* p,
                                              FieldElement* out_x, FieldElement* out_y) {
    if (p->infinity) {
        field_set_zero(out_x);
        field_set_zero(out_y);
        return false;
    }
    FieldElement z_inv, z_inv2, z_inv3;
    field_inv_var(&p->z, &z_inv);
    field_sqr(&z_inv, &z_inv2);
    field_mul(&z_inv, &z_inv2, &z_inv3);
    field_mul(&p->x, &z_inv2, out_x);
    field_mul(&p->y, &z_inv3, out_y);
    return true;
}

// Constant-time guard. Code in namespace ct that calls field_inv_var,
// scalar_inv_var or jacobian_to_affine_var unqualified finds these deleted
// declarations and does not compile. A qualified secp256k1::cuda:: call is not
// caught by the compiler; it stays forbidden by the rule in the file header.
namespace ct {
void field_inv_var(const FieldElement*, FieldElement*) = delete;
void scalar_inv_var(const Scalar*, Scalar*) = delete;
bool jacobian_to_affine_var(const JacobianPoint*, FieldElement*, FieldElement*) = delete;
} // namespace ct

} // namespace cuda
} // namespace secp256k1
