// ============================================================================
// add_cc native-vs-portable carry-chain A/B harness   (NF-2026-00031)
// ============================================================================
// Context: an external report (VanityPlate, Discussion #462) stayed on commit
// 552f86dd because a later "portable" carry change in the CUDA kernels ran
// slower than the native path.
//
// Finding (source reading, no measurement behind it):
//   * At 552f86dd, add_cc / sub_cc in src/cuda/include/secp256k1.cuh were
//     written with `unsigned __int128`, which nvcc lowers to a PTX
//     add.cc / addc carry chain.
//   * Commit 8af7320c ("Harden audit and fix Windows CUDA build") replaced
//     them, unconditionally and for nvcc too, with a compare-based form
//     (`sum < a`) so that MSVC host parsing and HIP could build.
//   * The hot field core on nvcc (field_add, field_sub, reduce_512_to_256,
//     mont_reduce_512, mul64, hybrid 32-bit field_mul/field_sqr) sits under
//     `#if SECP256K1_USE_PTX` and is inline PTX in both revisions; it never
//     called add_cc. The compare-based form therefore reached nvcc builds
//     through the scalar-order arithmetic (scalar_add, scalar_add_u64,
//     scalar_sub, scalar_mul_mod_n, glv_decompose) and field_mul_small, which
//     is the part of a vanity search that steps a private key.
//   * Fix: secp256k1.cuh now dispatches add_cc / sub_cc / muladd64 on
//     SECP256K1_CUDA_PORTABLE_CARRY. Default is the native __int128 form when
//     the build is nvcc (SECP256K1_USE_PTX) and the host compiler has
//     __int128; HIP/ROCm and MSVC hosts keep the portable form. Defining
//     SECP256K1_CUDA_PORTABLE_CARRY=1 forces the portable form on nvcc.
//     Both forms are exposed by name (add_cc_native / add_cc_portable, ...).
//
// What this program measures, in ONE binary (template instantiation):
//   NATIVE   : the production code nvcc compiles by default -- PTX field core
//              plus the __int128 add_cc/sub_cc in the scalar ops.
//   PORTABLE : the same operations with the compare-based add_cc / sub_cc /
//              muladd64 (this is the code path the HIP build takes) applied
//              to the field add/sub/reduce and the scalar add as well.
//              Multiplication itself is the production hybrid 32-bit Comba in
//              both forms, so only the carry-chain form differs.
// The generator multiply and the vanity kernel are a plain double-and-add on
// Jacobian coordinates written once as a template over the form; they are not
// the tuned production w8 table path. Their native result is additionally
// checked against production scalar_mul_generator_w8.
//
// Outputs of both forms are compared bit-for-bit; a mismatch exits non-zero.
// Timing uses cudaEvent only. No GPU is needed to compile; with no CUDA
// device the program prints a notice and exits 0 without measuring.
//
// Build (from libs/UltrafastSecp256k1, nvcc 13.x, RTX 5060 Ti = sm_120):
//   nvcc -O3 -std=c++17 -gencode arch=compute_120,code=sm_120 \
//        -I src/cuda/include -I include \
//        benchmarks/gpu/cuda/addcc_ab_bench.cu -o addcc_ab_bench
// Run:  ./addcc_ab_bench [reps=11]
//
// Do NOT build this with -DSECP256K1_CUDA_PORTABLE_CARRY=1: the NATIVE side
// calls the production code and would silently become the portable form.
// Performance numbers are intentionally not recorded in this file.
// ============================================================================

#include "secp256k1.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#if SECP256K1_CUDA_PORTABLE_CARRY
#error "addcc_ab_bench needs the default carry form (native __int128) for its NATIVE side; do not define SECP256K1_CUDA_PORTABLE_CARRY=1."
#endif
#if SECP256K1_CUDA_USE_MONTGOMERY
#error "addcc_ab_bench assumes the standard (non-Montgomery) field domain."
#endif

using namespace secp256k1::cuda;

#define CUDA_CHECK(c) do { cudaError_t e_ = (c); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error '%s' at line %d\n", cudaGetErrorString(e_), __LINE__); \
    std::exit(2); } } while (0)

// ---------------------------------------------------------------------------
// Carry policies
// ---------------------------------------------------------------------------
struct NativeCarry {
    __device__ static __forceinline__ uint64_t add(uint64_t a, uint64_t b, uint64_t& c) { return add_cc_native(a, b, c); }
    __device__ static __forceinline__ uint64_t sub(uint64_t a, uint64_t b, uint64_t& c) { return sub_cc_native(a, b, c); }
    __device__ static __forceinline__ uint64_t muladd(uint64_t a, uint64_t b, uint64_t d, uint64_t& c) { return muladd64_native(a, b, d, c); }
};
struct PortableCarry {
    __device__ static __forceinline__ uint64_t add(uint64_t a, uint64_t b, uint64_t& c) { return add_cc_portable(a, b, c); }
    __device__ static __forceinline__ uint64_t sub(uint64_t a, uint64_t b, uint64_t& c) { return sub_cc_portable(a, b, c); }
    __device__ static __forceinline__ uint64_t muladd(uint64_t a, uint64_t b, uint64_t d, uint64_t& c) { return muladd64_portable(a, b, d, c); }
};

// ---------------------------------------------------------------------------
// Field/scalar operation sets
// ---------------------------------------------------------------------------
// NATIVE: exactly what an nvcc build of the library compiles.
struct NativeOps {
    __device__ static __forceinline__ void add(const FieldElement* a, const FieldElement* b, FieldElement* r) { field_add(a, b, r); }
    __device__ static __forceinline__ void sub(const FieldElement* a, const FieldElement* b, FieldElement* r) { field_sub(a, b, r); }
    __device__ static __forceinline__ void mul(const FieldElement* a, const FieldElement* b, FieldElement* r) { field_mul(a, b, r); }
    __device__ static __forceinline__ void sqr(const FieldElement* a, FieldElement* r) { field_sqr(a, r); }
    __device__ static __forceinline__ void scalar_add_u64(const Scalar* a, uint64_t b, Scalar* r) {
        ::secp256k1::cuda::scalar_add_u64(a, b, r);
    }
};

// PORTABLE: HIP-style field add/sub/reduce and scalar add, parameterised on the
// carry policy C. Multiplication is the production hybrid Comba in both forms.
template <class C>
struct GenericOps {
    __device__ static __forceinline__ void add(const FieldElement* a, const FieldElement* b, FieldElement* r) {
        uint64_t carry = 0;
        const uint64_t r0 = C::add(a->limbs[0], b->limbs[0], carry);
        const uint64_t r1 = C::add(a->limbs[1], b->limbs[1], carry);
        const uint64_t r2 = C::add(a->limbs[2], b->limbs[2], carry);
        const uint64_t r3 = C::add(a->limbs[3], b->limbs[3], carry);
        uint64_t borrow = 0;
        const uint64_t t0 = C::sub(r0, MODULUS[0], borrow);
        const uint64_t t1 = C::sub(r1, MODULUS[1], borrow);
        const uint64_t t2 = C::sub(r2, MODULUS[2], borrow);
        const uint64_t t3 = C::sub(r3, MODULUS[3], borrow);
        uint64_t need = (uint64_t)((carry != 0ULL) | (borrow == 0ULL));
        asm volatile("" : "+l"(need));
        const uint64_t msk = (uint64_t)0 - need;
        r->limbs[0] = (t0 & msk) | (r0 & ~msk);
        r->limbs[1] = (t1 & msk) | (r1 & ~msk);
        r->limbs[2] = (t2 & msk) | (r2 & ~msk);
        r->limbs[3] = (t3 & msk) | (r3 & ~msk);
    }

    __device__ static __forceinline__ void sub(const FieldElement* a, const FieldElement* b, FieldElement* r) {
        uint64_t borrow = 0;
        const uint64_t r0 = C::sub(a->limbs[0], b->limbs[0], borrow);
        const uint64_t r1 = C::sub(a->limbs[1], b->limbs[1], borrow);
        const uint64_t r2 = C::sub(a->limbs[2], b->limbs[2], borrow);
        const uint64_t r3 = C::sub(a->limbs[3], b->limbs[3], borrow);
        uint64_t carry = 0;
        const uint64_t a0 = C::add(r0, MODULUS[0], carry);
        const uint64_t a1 = C::add(r1, MODULUS[1], carry);
        const uint64_t a2 = C::add(r2, MODULUS[2], carry);
        const uint64_t a3 = C::add(r3, MODULUS[3], carry);
        uint64_t need = (uint64_t)(borrow != 0ULL);
        asm volatile("" : "+l"(need));
        const uint64_t msk = (uint64_t)0 - need;
        r->limbs[0] = (a0 & msk) | (r0 & ~msk);
        r->limbs[1] = (a1 & msk) | (r1 & ~msk);
        r->limbs[2] = (a2 & msk) | (r2 & ~msk);
        r->limbs[3] = (a3 & msk) | (r3 & ~msk);
    }

    // 512 -> 256 reduction modulo p = 2^256 - K_MOD (same algorithm as the HIP path).
    __device__ static __forceinline__ void reduce(const uint64_t t[8], FieldElement* r) {
        uint64_t t0 = t[0], t1 = t[1], t2 = t[2], t3 = t[3];
        uint64_t carry_p = 0;
        const uint64_t a0 = C::muladd(t[4], K_MOD, 0, carry_p);
        const uint64_t a1 = C::muladd(t[5], K_MOD, 0, carry_p);
        const uint64_t a2 = C::muladd(t[6], K_MOD, 0, carry_p);
        const uint64_t a3 = C::muladd(t[7], K_MOD, 0, carry_p);
        const uint64_t a4 = carry_p;

        uint64_t carry = 0;
        t0 = C::add(t0, a0, carry);
        t1 = C::add(t1, a1, carry);
        t2 = C::add(t2, a2, carry);
        t3 = C::add(t3, a3, carry);

        const uint64_t extra = a4 + carry;
        uint64_t ek_hi = 0;
        const uint64_t ek_lo = C::muladd(extra, K_MOD, 0, ek_hi);
        carry = 0;
        t0 = C::add(t0, ek_lo, carry);
        t1 = C::add(t1, ek_hi, carry);
        t2 = C::add(t2, 0, carry);
        t3 = C::add(t3, 0, carry);
        if (carry) {
            carry = 0;
            t0 = C::add(t0, K_MOD, carry);
            t1 = C::add(t1, 0, carry);
            t2 = C::add(t2, 0, carry);
            t3 = C::add(t3, 0, carry);
        }

        uint64_t borrow = 0;
        const uint64_t s0 = C::sub(t0, MODULUS[0], borrow);
        const uint64_t s1 = C::sub(t1, MODULUS[1], borrow);
        const uint64_t s2 = C::sub(t2, MODULUS[2], borrow);
        const uint64_t s3 = C::sub(t3, MODULUS[3], borrow);
        uint64_t need = (uint64_t)(borrow == 0ULL);
        asm volatile("" : "+l"(need));
        const uint64_t msk = (uint64_t)0 - need;
        r->limbs[0] = (s0 & msk) | (t0 & ~msk);
        r->limbs[1] = (s1 & msk) | (t1 & ~msk);
        r->limbs[2] = (s2 & msk) | (t2 & ~msk);
        r->limbs[3] = (s3 & msk) | (t3 & ~msk);
    }

    __device__ static __forceinline__ void mul(const FieldElement* a, const FieldElement* b, FieldElement* r) {
        uint64_t t[8];
#if SECP256K1_CUDA_USE_HYBRID_MUL
        mul_256_512_hybrid(a, b, t);
#else
        mul_256_512(a, b, t);
#endif
        reduce(t, r);
    }

    __device__ static __forceinline__ void sqr(const FieldElement* a, FieldElement* r) {
        uint64_t t[8];
#if SECP256K1_CUDA_USE_HYBRID_MUL
        uint32_t t32[16];
        sqr_256_comba32(a, t32);
        #pragma unroll
        for (int i = 0; i < 8; i++) t[i] = ((uint64_t)t32[2 * i + 1] << 32) | t32[2 * i];
#else
        sqr_256_512(a, t);
#endif
        reduce(t, r);
    }

    // r = a + b (mod n), b a 64-bit value. Same shape as production scalar_add_u64.
    __device__ static __forceinline__ void scalar_add_u64(const Scalar* a, uint64_t b, Scalar* r) {
        uint64_t carry = 0;
        const uint64_t r0 = C::add(a->limbs[0], b, carry);
        const uint64_t r1 = C::add(a->limbs[1], 0, carry);
        const uint64_t r2 = C::add(a->limbs[2], 0, carry);
        const uint64_t r3 = C::add(a->limbs[3], 0, carry);
        uint64_t borrow = 0;
        const uint64_t t0 = C::sub(r0, ORDER[0], borrow);
        const uint64_t t1 = C::sub(r1, ORDER[1], borrow);
        const uint64_t t2 = C::sub(r2, ORDER[2], borrow);
        const uint64_t t3 = C::sub(r3, ORDER[3], borrow);
        uint64_t need = (uint64_t)((carry != 0ULL) | (borrow == 0ULL));
        asm volatile("" : "+l"(need));
        const uint64_t msk = (uint64_t)0 - need;
        r->limbs[0] = (t0 & msk) | (r0 & ~msk);
        r->limbs[1] = (t1 & msk) | (r1 & ~msk);
        r->limbs[2] = (t2 & msk) | (r2 & ~msk);
        r->limbs[3] = (t3 & msk) | (r3 & ~msk);
    }
};
using PortableOps = GenericOps<PortableCarry>;
using NativeCarryOps = GenericOps<NativeCarry>;   // cross-check only (not timed)

// ---------------------------------------------------------------------------
// Jacobian double-and-add generator multiply, templated on the operation set
// ---------------------------------------------------------------------------
struct Jac { FieldElement x, y, z; };

template <class O>
__device__ __forceinline__ void jac_dbl(Jac* p) {          // a = 0 doubling (dbl-2009-l)
    FieldElement A, B, C, t, u, D, E, F, x3, y3, z3, c2, c4, c8;
    O::sqr(&p->x, &A);
    O::sqr(&p->y, &B);
    O::sqr(&B, &C);
    O::add(&p->x, &B, &t);
    O::sqr(&t, &u);
    O::sub(&u, &A, &t);
    O::sub(&t, &C, &u);
    O::add(&u, &u, &D);
    O::add(&A, &A, &t);
    O::add(&t, &A, &E);
    O::sqr(&E, &F);
    O::sub(&F, &D, &t);
    O::sub(&t, &D, &x3);
    O::sub(&D, &x3, &t);
    O::mul(&E, &t, &u);
    O::add(&C, &C, &c2);
    O::add(&c2, &c2, &c4);
    O::add(&c4, &c4, &c8);
    O::sub(&u, &c8, &y3);
    O::mul(&p->y, &p->z, &t);
    O::add(&t, &t, &z3);
    p->x = x3; p->y = y3; p->z = z3;
}

template <class O>
__device__ __forceinline__ void jac_add_affine(Jac* p, const FieldElement* qx, const FieldElement* qy) {
    FieldElement z1z1, u2, s2, t, u, h, hh, i4, j, rr, v, x3, y3, z3;
    O::sqr(&p->z, &z1z1);
    O::mul(qx, &z1z1, &u2);
    O::mul(&p->z, &z1z1, &t);
    O::mul(qy, &t, &s2);
    O::sub(&u2, &p->x, &h);
    O::sqr(&h, &hh);
    O::add(&hh, &hh, &t);
    O::add(&t, &t, &i4);
    O::mul(&h, &i4, &j);
    O::sub(&s2, &p->y, &t);
    O::add(&t, &t, &rr);
    O::mul(&p->x, &i4, &v);
    O::sqr(&rr, &t);
    O::sub(&t, &j, &u);
    O::sub(&u, &v, &t);
    O::sub(&t, &v, &x3);
    O::sub(&v, &x3, &t);
    O::mul(&rr, &t, &u);
    O::mul(&p->y, &j, &t);
    O::add(&t, &t, &s2);
    O::sub(&u, &s2, &y3);
    O::add(&p->z, &h, &t);
    O::sqr(&t, &u);
    O::sub(&u, &z1z1, &t);
    O::sub(&t, &hh, &z3);
    p->x = x3; p->y = y3; p->z = z3;
}

__device__ __forceinline__ int ab_scalar_bit(const Scalar* k, int i) {
    return (int)((k->limbs[i >> 6] >> (i & 63)) & 1ULL);
}

// r = k*G, variable-time, k in [1, n-1].
template <class O>
__device__ void gen_mul(const Scalar* k, Jac* r) {
    FieldElement gx, gy;
    for (int i = 0; i < 4; i++) { gx.limbs[i] = GENERATOR_X[i]; gy.limbs[i] = GENERATOR_Y[i]; }
    int top = 255;
    while (top > 0 && !ab_scalar_bit(k, top)) --top;
    r->x = gx; r->y = gy; field_set_one(&r->z);
    for (int i = top - 1; i >= 0; --i) {
        jac_dbl<O>(r);
        if (ab_scalar_bit(k, i)) jac_add_affine<O>(r, &gx, &gy);
    }
}

// ---------------------------------------------------------------------------
// Benchmark kernels
// ---------------------------------------------------------------------------
template <class O>
__global__ void k_field_mul(const FieldElement* a, const FieldElement* b, FieldElement* out, int n, int iters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    FieldElement x = a[i], y = b[i], t;
    for (int it = 0; it < iters; ++it) { O::mul(&x, &y, &t); x = y; y = t; }
    out[i] = y;
}

// 3 field add/sub operations per iteration.
template <class O>
__global__ void k_addsub(const FieldElement* a, const FieldElement* b, FieldElement* out, int n, int iters) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    FieldElement x = a[i], y = b[i], t, u;
    for (int it = 0; it < iters; ++it) {
        O::add(&x, &y, &t);
        O::sub(&t, &y, &u);
        O::add(&u, &t, &x);
        y = u;
    }
    out[i] = x;
}

template <class O>
__global__ void k_genmul(const Scalar* ks, Jac* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Scalar k = ks[i];
    Jac r;
    gen_mul<O>(&k, &r);
    out[i] = r;
}

// Vanity-search shape: step a private key by the thread index (scalar carry
// chain across limbs), derive the public point, then fold a few field ops.
template <class O>
__global__ void k_vanity(Scalar base, FieldElement* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Scalar k;
    O::scalar_add_u64(&base, (uint64_t)i, &k);
    Jac r;
    gen_mul<O>(&k, &r);
    FieldElement z2, z3, ax, ay, s;
    O::sqr(&r.z, &z2);
    O::mul(&z2, &r.z, &z3);
    O::mul(&r.x, &z2, &ax);
    O::mul(&r.y, &z3, &ay);
    O::add(&ax, &ay, &s);
    O::sub(&s, &ay, &z2);
    O::add(&z2, &s, &s);
    out[i] = s;
}

// Production reference: scalar_mul_generator_w8 vs harness result, compared
// projectively (X1*Z2^2 == X2*Z1^2 and Y1*Z2^3 == Y2*Z1^3) with production ops.
__global__ void k_check_vs_production(const Scalar* ks, const Jac* got, int* bad, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Scalar k = ks[i];
    JacobianPoint ref;
    scalar_mul_generator_w8(&k, &ref);
    const Jac& g = got[i];
    FieldElement rz2, gz2, rz3, gz3, l, r;
    field_sqr(&ref.z, &rz2);  field_sqr(&g.z, &gz2);
    field_mul(&rz2, &ref.z, &rz3);  field_mul(&gz2, &g.z, &gz3);
    field_mul(&g.x, &rz2, &l);  field_mul(&ref.x, &gz2, &r);
    bool ok = field_eq(&l, &r);
    field_mul(&g.y, &rz3, &l);  field_mul(&ref.y, &gz3, &r);
    ok = ok && field_eq(&l, &r);
    if (!ok) atomicAdd(bad, 1);
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
namespace {

constexpr int kBlock = 128;
constexpr int kNField = 1 << 16;   // threads for field kernels
constexpr int kNPoint = 1 << 15;   // threads for point kernels
constexpr int kIters  = 256;       // loop trip count in field kernels
const uint64_t kPm1[4] = {0xFFFFFFFEFFFFFC2EULL, ~0ULL, ~0ULL, ~0ULL};   // p - 1

inline int grid_for(int n) { return (n + kBlock - 1) / kBlock; }

// Random value below p (top limb masked to 63 bits => < p).
void rand_field(std::mt19937_64& g, FieldElement& f) {
    for (int i = 0; i < 4; i++) f.limbs[i] = g();
    f.limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
}
// Random scalar in [1, n-1]: top limb masked to 63 bits => < n/2.
void rand_scalar(std::mt19937_64& g, Scalar& s) {
    for (int i = 0; i < 4; i++) s.limbs[i] = g();
    s.limbs[3] &= 0x7FFFFFFFFFFFFFFFULL;
    if ((s.limbs[0] | s.limbs[1] | s.limbs[2] | s.limbs[3]) == 0) s.limbs[0] = 1;
}

enum Kernel { K_MUL = 0, K_ADDSUB, K_GENMUL, K_VANITY, K_COUNT };
const char* kKernelName[K_COUNT] = {
    "field_mul", "field add/sub chain", "generator scalar mult", "vanity (key step + pubkey + field ops)"
};

struct Buffers {
    FieldElement *a, *b, *fo;        // field kernels
    Scalar* ks; Jac* jo;             // genmul
    FieldElement* vo;                // vanity
    int* bad;
};

template <class O> struct Results {
    std::vector<FieldElement> mul, addsub, vanity;
    std::vector<Jac> gen;
};

template <class O>
float time_kernel(Kernel which, const Buffers& d, const Scalar& vbase, cudaEvent_t e0, cudaEvent_t e1) {
    CUDA_CHECK(cudaEventRecord(e0));
    switch (which) {
        case K_MUL:    k_field_mul<O><<<grid_for(kNField), kBlock>>>(d.a, d.b, d.fo, kNField, kIters); break;
        case K_ADDSUB: k_addsub<O><<<grid_for(kNField), kBlock>>>(d.a, d.b, d.fo, kNField, kIters); break;
        case K_GENMUL: k_genmul<O><<<grid_for(kNPoint), kBlock>>>(d.ks, d.jo, kNPoint); break;
        case K_VANITY: k_vanity<O><<<grid_for(kNPoint), kBlock>>>(vbase, d.vo, kNPoint); break;
        default: break;
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    return ms;
}

// ops per launch, for ns/op
double ops_per_launch(Kernel k) {
    switch (k) {
        case K_MUL:    return (double)kNField * kIters;
        case K_ADDSUB: return (double)kNField * kIters * 3.0;
        default:       return (double)kNPoint;   // one scalar mult / key candidate per thread
    }
}

template <class O>
void collect(Results<O>& R, const Buffers& d, const Scalar& vbase, cudaEvent_t e0, cudaEvent_t e1) {
    R.mul.resize(kNField); R.addsub.resize(kNField); R.gen.resize(kNPoint); R.vanity.resize(kNPoint);
    time_kernel<O>(K_MUL, d, vbase, e0, e1);
    CUDA_CHECK(cudaMemcpy(R.mul.data(), d.fo, sizeof(FieldElement) * kNField, cudaMemcpyDeviceToHost));
    time_kernel<O>(K_ADDSUB, d, vbase, e0, e1);
    CUDA_CHECK(cudaMemcpy(R.addsub.data(), d.fo, sizeof(FieldElement) * kNField, cudaMemcpyDeviceToHost));
    time_kernel<O>(K_GENMUL, d, vbase, e0, e1);
    CUDA_CHECK(cudaMemcpy(R.gen.data(), d.jo, sizeof(Jac) * kNPoint, cudaMemcpyDeviceToHost));
    time_kernel<O>(K_VANITY, d, vbase, e0, e1);
    CUDA_CHECK(cudaMemcpy(R.vanity.data(), d.vo, sizeof(FieldElement) * kNPoint, cudaMemcpyDeviceToHost));
}

template <class T>
bool same(const std::vector<T>& x, const std::vector<T>& y) {
    return x.size() == y.size() && std::memcmp(x.data(), y.data(), sizeof(T) * x.size()) == 0;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 11;
    if (argc > 1) reps = std::max(1, std::atoi(argv[1]));

    int ndev = 0;
    if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev == 0) {
        std::printf("addcc_ab_bench: no CUDA device available; nothing measured.\n");
        return 0;
    }
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int clock_khz = 0;   // cudaDeviceProp::clockRate is gone in CUDA 13
    cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, dev);
    std::printf("GPU: %s | SM=%d | %d MHz | reps=%d\n", prop.name, prop.multiProcessorCount, clock_khz / 1000, reps);
    std::printf("add_cc form in production build: native (SECP256K1_CUDA_PORTABLE_CARRY=%d)\n\n",
                (int)SECP256K1_CUDA_PORTABLE_CARRY);

    // ---- inputs ----
    std::mt19937_64 rng(0xADDCC0DEULL);
    std::vector<FieldElement> ha(kNField), hb(kNField);
    for (int i = 0; i < kNField; i++) { rand_field(rng, ha[i]); rand_field(rng, hb[i]); }
    // Edge values: p-1, 0, 1 in the first lanes so the reduce/borrow paths are exercised.
    for (int i = 0; i < 4; i++) { ha[0].limbs[i] = kPm1[i]; hb[0].limbs[i] = kPm1[i]; }
    std::vector<Scalar> hk(kNPoint);
    for (int i = 0; i < kNPoint; i++) rand_scalar(rng, hk[i]);
    hk[0] = Scalar{{1, 0, 0, 0}};
    hk[1] = Scalar{{2, 0, 0, 0}};
    Scalar vbase;       // low limbs near 2^128 so stepping carries across limbs 0 and 1
    vbase.limbs[0] = 0xFFFFFFFFFFFFFF00ULL;
    vbase.limbs[1] = 0xFFFFFFFFFFFFFFFFULL;
    vbase.limbs[2] = rng();
    vbase.limbs[3] = rng() & 0x3FFFFFFFFFFFFFFFULL;

    Buffers d{};
    CUDA_CHECK(cudaMalloc(&d.a, sizeof(FieldElement) * kNField));
    CUDA_CHECK(cudaMalloc(&d.b, sizeof(FieldElement) * kNField));
    CUDA_CHECK(cudaMalloc(&d.fo, sizeof(FieldElement) * kNField));
    CUDA_CHECK(cudaMalloc(&d.ks, sizeof(Scalar) * kNPoint));
    CUDA_CHECK(cudaMalloc(&d.jo, sizeof(Jac) * kNPoint));
    CUDA_CHECK(cudaMalloc(&d.vo, sizeof(FieldElement) * kNPoint));
    CUDA_CHECK(cudaMalloc(&d.bad, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d.a, ha.data(), sizeof(FieldElement) * kNField, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.b, hb.data(), sizeof(FieldElement) * kNField, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.ks, hk.data(), sizeof(Scalar) * kNPoint, cudaMemcpyHostToDevice));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    // ---- identical-output checks ----
    Results<NativeOps> rn;
    Results<PortableOps> rp;
    Results<NativeCarryOps> rg;
    collect<NativeOps>(rn, d, vbase, e0, e1);
    // production check on the native genmul result (collect() left d.jo from the vanity-order run;
    // re-run genmul so d.jo holds the native result)
    time_kernel<NativeOps>(K_GENMUL, d, vbase, e0, e1);
    CUDA_CHECK(cudaMemset(d.bad, 0, sizeof(int)));
    k_check_vs_production<<<grid_for(kNPoint), kBlock>>>(d.ks, d.jo, d.bad, kNPoint);
    CUDA_CHECK(cudaGetLastError());
    int bad = 0;
    CUDA_CHECK(cudaMemcpy(&bad, d.bad, sizeof(int), cudaMemcpyDeviceToHost));
    collect<PortableOps>(rp, d, vbase, e0, e1);
    collect<NativeCarryOps>(rg, d, vbase, e0, e1);

    bool ok = true;
    auto report = [&](const char* what, bool good) {
        std::printf("  identical outputs  %-40s %s\n", what, good ? "OK" : "MISMATCH");
        ok = ok && good;
    };
    std::printf("Correctness (native vs portable, bit-for-bit):\n");
    report(kKernelName[K_MUL],    same(rn.mul, rp.mul));
    report(kKernelName[K_ADDSUB], same(rn.addsub, rp.addsub));
    report(kKernelName[K_GENMUL], same(rn.gen, rp.gen));
    report(kKernelName[K_VANITY], same(rn.vanity, rp.vanity));
    report("generic<int128> == native (mul/addsub)", same(rn.mul, rg.mul) && same(rn.addsub, rg.addsub));
    report("generic<int128> == native (genmul/vanity)", same(rn.gen, rg.gen) && same(rn.vanity, rg.vanity));
    std::printf("  genmul vs production scalar_mul_generator_w8: %d of %d points differ\n", bad, kNPoint);
    ok = ok && (bad == 0);
    if (!ok) { std::printf("\nFAILED: forms disagree.\n"); return 1; }

    // ---- timing: alternate which form goes first each rep ----
    std::vector<double> tn[K_COUNT], tp[K_COUNT];
    for (int k = 0; k < K_COUNT; k++) {   // warm-up both forms
        time_kernel<NativeOps>((Kernel)k, d, vbase, e0, e1);
        time_kernel<PortableOps>((Kernel)k, d, vbase, e0, e1);
    }
    for (int rep = 0; rep < reps; rep++) {
        for (int k = 0; k < K_COUNT; k++) {
            double n_ns, p_ns;
            const double ops = ops_per_launch((Kernel)k);
            if (rep & 1) {
                p_ns = time_kernel<PortableOps>((Kernel)k, d, vbase, e0, e1) * 1e6 / ops;
                n_ns = time_kernel<NativeOps>((Kernel)k, d, vbase, e0, e1) * 1e6 / ops;
            } else {
                n_ns = time_kernel<NativeOps>((Kernel)k, d, vbase, e0, e1) * 1e6 / ops;
                p_ns = time_kernel<PortableOps>((Kernel)k, d, vbase, e0, e1) * 1e6 / ops;
            }
            tn[k].push_back(n_ns);
            tp[k].push_back(p_ns);
        }
    }

    std::printf("\nns/op over %d alternating reps (median [min..max]) -- whole-GPU ns per op\n", reps);
    for (int k = 0; k < K_COUNT; k++) {
        auto mm = [](const std::vector<double>& v) {
            return std::make_pair(*std::min_element(v.begin(), v.end()), *std::max_element(v.begin(), v.end()));
        };
        const auto nr = mm(tn[k]);
        const auto pr = mm(tp[k]);
        std::printf("%-42s\n", kKernelName[k]);
        std::printf("    native   %10.4f  [%10.4f .. %10.4f]\n", median(tn[k]), nr.first, nr.second);
        std::printf("    portable %10.4f  [%10.4f .. %10.4f]\n", median(tp[k]), pr.first, pr.second);
    }

    cudaFree(d.a); cudaFree(d.b); cudaFree(d.fo); cudaFree(d.ks); cudaFree(d.jo); cudaFree(d.vo); cudaFree(d.bad);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
