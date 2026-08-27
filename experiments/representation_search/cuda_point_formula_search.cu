// EXP_CUDA_POINT_REPRESENTATION_SEARCH_CLAUDE_715
//
// Isolated CUDA correctness + benchmark experiment comparing the production
// Jacobian point-doubling and mixed-addition formulas (src/cuda/include/secp256k1.cuh)
// against mathematically equivalent alternative formulas.
//
// This file does NOT modify production code. It #includes the production
// header read-only and defines separate `_alt` device functions.
//
// Duplication contract: test_cuda_point_formula_search.cu duplicates the
// `_alt` candidate bodies and the pool-generation kernels below verbatim,
// because each .cu in this experiment must compile as an independent
// translation unit (no shared header is in scope between the benchmark
// binary and the fast correctness-gate binary). This benchmark binary
// already re-validates its own candidates before timing them, so no third
// shared-header file is introduced. Any change to a candidate formula or a
// pool generator here MUST be mirrored in test_cuda_point_formula_search.cu.
//
// Candidate formulas (derivations, both proven by elementary algebra below):
//
//   1. jacobian_double_alt:
//      Production (dbl-2001-b style, 3M+4S) computes S = 4*X*Y^2 directly via
//      one multiplication: S = X*YY, S += S, S += S.
//      Candidate uses the binomial identity (a+b)^2 - a^2 - b^2 = 2ab with
//      a=X, b=YY=Y^2:
//          (X+YY)^2 - X^2 - YY^2 = 2*X*YY  =>  S = 2*[(X+YY)^2 - XX - YYYY]
//      This replaces one multiplication with one extra squaring (X^2 and
//      Y^4 are already needed elsewhere), trading 1M for 1S: net 2M+5S.
//
//   2. jacobian_add_mixed_alt:
//      Production (madd-2007-bl, 7M+4S) computes
//          Z3 = (Z1+H)^2 - Z1Z1 - HH
//      reusing the already-computed Z1Z1 and HH. By the same binomial
//      identity, (Z1+H)^2 - Z1^2 - H^2 == 2*Z1*H exactly (pure algebra, no
//      field-specific trick required). The candidate computes Z3 directly:
//          Z3 = 2*Z1*H
//      trading the extra squaring + 2 subtractions for one multiply + one
//      addition: net 8M+3S (vs 7M+4S).
//
// Both candidates are verified bit-for-bit against the production formulas
// on GPU across identity, equal/opposite-point, deterministic scalar-
// multiple, and randomized-valid-point inputs before any timing is trusted
// (correctness hard-gates timing, per repo policy). Doubling additionally
// covers infinity and y=0 inputs, since jacobian_double explicitly checks
// p->infinity.
//
// jacobian_add_mixed_unchecked intentionally ignores p->infinity (it is the
// "_unchecked" mixed-add variant), so comparing it against
// jacobian_add_mixed_alt on an infinity-flagged P would be vacuous — both
// sides would silently run the ordinary-path formula on whatever field
// values happen to sit in P's coordinates, proving nothing about identity
// handling. Equal, opposite, deterministic-scalar and random cases
// therefore stay on the unchecked formulas (P is always finite there).
// Genuine identity coverage (P = group identity, i.e. an infinity-flagged
// Jacobian point, Q = a valid finite affine point) is instead tested by
// comparing the production *checked* jacobian_add_mixed against
// jacobian_add_mixed_alt_checked, an experiment-only wrapper that mirrors
// production's `if (p->infinity) { r = q; return; }` branch and otherwise
// delegates to jacobian_add_mixed_alt — exercising the real infinity branch
// on both sides instead of the ordinary-path formula on a bogus input.
//
// Benchmark methodology: to prevent GPU clock ramp / thermal drift from
// biasing one formula in a pair, each measured pass launches both formulas
// back-to-back and alternates which one is timed first (production-first on
// even pass indices, alt-first on odd pass indices — "balanced paired
// AB/BA"). Warmup passes alternate the same way. WARMUP (4) and PASSES (8)
// are both even, so this alternation yields an EXACT 50/50 split — 2
// production-first + 2 alt-first warmups, 4 production-first + 4 alt-first
// measured passes — not merely an approximate balance. The exact per-pass
// order is recorded in the JSON report's "pass_order" field for each pair so
// a reader can verify a sub-2% delta isn't an artifact of launch order.
//
// Throughput: not yet measured on this machine — see JSON report emitted by
// this binary for the only trustworthy numbers (per-machine, per-run).
//
// ---------------------------------------------------------------------------
// Build (repo-local, from experiments/representation_search/, artifacts only
// under top-level out/). Validated route: nvcc 12, sm_89 binary with a
// compute_89 PTX fallback for forward compatibility, canonical include root
// pinned explicitly via -I (the .cu also uses a relative #include, so -I is
// belt-and-suspenders documentation of the canonical root, not a functional
// requirement):
//
//   mkdir -p ../../out/experiments
//   nvcc -O3 -std=c++17 \
//        -gencode arch=compute_89,code=sm_89 \
//        -gencode arch=compute_89,code=compute_89 \
//        -I ../../src/cuda/include \
//        -o ../../out/experiments/cuda_point_formula_search \
//        cuda_point_formula_search.cu
//
// Run (writes the JSON report to stdout; redirect to keep an artifact under
// top-level out/):
//
//   ../../out/experiments/cuda_point_formula_search \
//       > ../../out/experiments/cuda_point_formula_search_report.json
//
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#include "../../src/cuda/include/secp256k1.cuh"

using namespace secp256k1::cuda;

#if defined(SECP256K1_CUDA_LIMBS_32)
#error "This experiment assumes the default 64-bit-limb FieldElement layout (SECP256K1_CUDA_LIMBS_32 must stay undefined)."
#endif

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                           \
            std::exit(1);                                                     \
        }                                                                      \
    } while (0)

// ---------------------------------------------------------------------------
// Candidate formulas
// ---------------------------------------------------------------------------

// Candidate A: 2M + 5S (production jacobian_double is 3M + 4S).
__device__ inline void jacobian_double_alt(const JacobianPoint* p, JacobianPoint* r) {
    if (p->infinity) { r->infinity = true; return; }
    if (field_is_zero(&p->y)) { r->infinity = true; return; }

    FieldElement yy, xx, yyyy, t, s, m, x3, y3, z3;

    field_sqr(&p->y, &yy);      // YY = Y^2
    field_sqr(&p->x, &xx);      // XX = X^2
    field_sqr(&yy, &yyyy);      // YYYY = Y^4

    field_add(&p->x, &yy, &t);  // t = X + YY
    field_sqr(&t, &t);          // t = (X + YY)^2
    field_sub(&t, &xx, &t);     // t -= XX
    field_sub(&t, &yyyy, &t);   // t -= YYYY   => t = 2*X*YY  (binomial identity)
    field_add(&t, &t, &s);      // S = 2*t = 4*X*YY

    field_add(&xx, &xx, &m);    // m = 2*XX
    field_add(&m, &xx, &m);     // M = 3*XX  (no multiply, matches production tripling)

    field_sqr(&m, &x3);         // X3 = M^2
    field_add(&s, &s, &t);      // t = 2*S
    field_sub(&x3, &t, &x3);    // X3 -= 2*S

    field_sub(&s, &x3, &t);     // t = S - X3
    field_mul(&m, &t, &y3);     // Y3 = M*(S - X3)
    field_add(&yyyy, &yyyy, &t);// t = 2*YYYY
    field_add(&t, &t, &t);      // t = 4*YYYY
    field_add(&t, &t, &t);      // t = 8*YYYY
    field_sub(&y3, &t, &y3);    // Y3 -= 8*YYYY

    field_mul(&p->y, &p->z, &z3); // Z3 = Y*Z
    field_add(&z3, &z3, &z3);     // Z3 *= 2

    r->x = x3; r->y = y3; r->z = z3; r->infinity = false;
}

// Candidate B: 8M + 3S (production jacobian_add_mixed_unchecked is 7M + 4S).
// Identical to production except Z3, computed via Z3 = 2*Z1*H instead of
// Z3 = (Z1+H)^2 - Z1Z1 - HH (both are algebraically the exact same value).
__device__ inline void jacobian_add_mixed_alt(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r) {
    FieldElement z1z1, u2, s2, h, hh, i, j, rr, v;
    FieldElement x3, y3, z3, t1, t2;

    field_sqr(&p->z, &z1z1);
    field_mul(&q->x, &z1z1, &u2);

    field_mul(&p->z, &z1z1, &t1);
    field_mul(&q->y, &t1, &s2);

    field_sub(&u2, &p->x, &h);

    if (field_is_zero(&h)) {
        field_sub(&s2, &p->y, &t1);
        if (field_is_zero(&t1)) {
            jacobian_double_unchecked(p, r);
            return;
        }
        r->infinity = true;
        return;
    }

    field_sqr(&h, &hh);
    field_add(&hh, &hh, &i);
    field_add(&i, &i, &i);
    field_mul(&h, &i, &j);

    field_sub(&s2, &p->y, &t1);
    field_add(&t1, &t1, &rr);

    field_mul(&p->x, &i, &v);

    field_sqr(&rr, &x3);
    field_sub(&x3, &j, &x3);
    field_add(&v, &v, &t1);
    field_sub(&x3, &t1, &x3);

    field_sub(&v, &x3, &t1);
    field_mul(&rr, &t1, &y3);
    field_mul(&p->y, &j, &t2);
    field_add(&t2, &t2, &t2);
    field_sub(&y3, &t2, &y3);

    // Candidate Z3: (a+b)^2 - a^2 - b^2 == 2ab, a=Z1, b=H (elementary algebra).
    field_mul(&p->z, &h, &z3);
    field_add(&z3, &z3, &z3);

    r->x = x3;
    r->y = y3;
    r->z = z3;
    r->infinity = false;
}

// Experiment-only checked wrapper around jacobian_add_mixed_alt: mirrors
// production jacobian_add_mixed's `if (p->infinity) { r = q; return; }`
// branch verbatim, then delegates to the ordinary-path candidate. Exists
// solely to give the CAT_IDENTITY correctness case a real infinity branch to
// compare against production's checked jacobian_add_mixed.
__device__ inline void jacobian_add_mixed_alt_checked(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r) {
    if (p->infinity) {
        r->x = q->x;
        r->y = q->y;
        field_set_one(&r->z);
        r->infinity = false;
        return;
    }
    jacobian_add_mixed_alt(p, q, r);
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (fixed seed, reproducible across runs)
// ---------------------------------------------------------------------------

static constexpr uint64_t FIXED_SEED = 0xC0FFEE123456789ULL;

__device__ inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// ---------------------------------------------------------------------------
// Test-point pool generation (device, single-thread setup — not timed)
// ---------------------------------------------------------------------------

__global__ void gen_pow2_pool_kernel(JacobianPoint* jac_out, AffinePoint* aff_out, int count) {
    JacobianPoint g;
    g.x.limbs[0] = 0x59F2815B16F81798ULL; g.x.limbs[1] = 0x029BFCDB2DCE28D9ULL;
    g.x.limbs[2] = 0x55A06295CE870B07ULL; g.x.limbs[3] = 0x79BE667EF9DCBBACULL;
    g.y.limbs[0] = 0x9C47D08FFB10D4B8ULL; g.y.limbs[1] = 0xFD17B448A6855419ULL;
    g.y.limbs[2] = 0x5DA4FBFC0E1108A8ULL; g.y.limbs[3] = 0x483ADA7726A3C465ULL;
    field_set_one(&g.z);
    g.infinity = false;

    jac_out[0] = g;
    FieldElement ax, ay;
    jacobian_to_affine(&jac_out[0], &ax, &ay);
    aff_out[0].x = ax; aff_out[0].y = ay;

    for (int idx = 1; idx < count; ++idx) {
        JacobianPoint r;
        jacobian_double(&jac_out[idx - 1], &r); // ground truth: production formula
        jac_out[idx] = r;
        FieldElement bx, by;
        jacobian_to_affine(&r, &bx, &by);
        aff_out[idx].x = bx; aff_out[idx].y = by;
    }
}

__global__ void gen_random_pool_kernel(const JacobianPoint* pow2_jac, const AffinePoint* pow2_aff, int pow2_count,
                                        JacobianPoint* out_jac, AffinePoint* out_aff, int count, uint64_t seed) {
    uint64_t state = seed;
    for (int k = 0; k < count; ++k) {
        int i = (int)(splitmix64_next(state) % (uint64_t)pow2_count);
        int j = (int)(splitmix64_next(state) % (uint64_t)pow2_count);
        JacobianPoint combined;
        jacobian_add_mixed_unchecked(&pow2_jac[i], &pow2_aff[j], &combined); // ground truth
        if (splitmix64_next(state) & 1ULL) {
            JacobianPoint doubled;
            jacobian_double(&combined, &doubled); // ground truth
            combined = doubled;
        }
        out_jac[k] = combined;
        FieldElement ax, ay;
        jacobian_to_affine(&combined, &ax, &ay);
        out_aff[k].x = ax; out_aff[k].y = ay;
    }
}

__global__ void gen_edge_double_kernel(const JacobianPoint* pow2_jac, JacobianPoint* out) {
    JacobianPoint base = pow2_jac[5];

    JacobianPoint inf_pt = base;
    inf_pt.infinity = true;
    out[0] = inf_pt; // CAT_INFINITY

    JacobianPoint yzero_pt = base;
    yzero_pt.infinity = false;
    field_set_zero(&yzero_pt.y);
    out[1] = yzero_pt; // CAT_Y_ZERO
}

__global__ void gen_edge_add_kernel(const JacobianPoint* pow2_jac, const AffinePoint* pow2_aff,
                                     int base_idx,
                                     JacobianPoint* out_p, AffinePoint* out_q) {
    JacobianPoint base = pow2_jac[base_idx];
    AffinePoint base_aff = pow2_aff[base_idx];

    // CAT_IDENTITY is intentionally NOT generated here: it needs a real
    // infinity-flagged P compared via the *checked* formulas (see
    // gen_identity_case_kernel / correctness_identity_kernel below), not a
    // finite P added to a bogus (0,0) affine "identity" run through the
    // unchecked formula. jacobian_add_mixed_unchecked ignores p->infinity by
    // design, so it cannot provide genuine identity coverage.

    out_p[0] = base; out_q[0] = base_aff; // CAT_EQUAL

    AffinePoint neg_q;
    neg_q.x = base_aff.x;
    FieldElement zero_fe;
    field_set_zero(&zero_fe);
    field_sub(&zero_fe, &base_aff.y, &neg_q.y);
    out_p[1] = base; out_q[1] = neg_q; // CAT_OPPOSITE
}

// Genuine CAT_IDENTITY input: P is a real infinity-flagged Jacobian point
// (the group identity), Q is a valid finite affine point. x/y/z of the
// infinity point are zeroed only so the struct is fully defined; production
// jacobian_add_mixed and jacobian_add_mixed_alt_checked both branch on
// p->infinity before ever reading p->x/y/z, so their values are provably
// irrelevant to the result.
__global__ void gen_identity_case_kernel(const AffinePoint* pow2_aff, int base_idx,
                                          JacobianPoint* out_p, AffinePoint* out_q) {
    JacobianPoint inf_p;
    field_set_zero(&inf_p.x);
    field_set_zero(&inf_p.y);
    field_set_zero(&inf_p.z);
    inf_p.infinity = true;

    out_p[0] = inf_p;
    out_q[0] = pow2_aff[base_idx]; // CAT_IDENTITY: valid affine Q
}

__global__ void gen_pairs_kernel(const JacobianPoint* pool_jac, const AffinePoint* pool_aff, int count, int stride,
                                  JacobianPoint* out_p, AffinePoint* out_q) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    out_p[idx] = pool_jac[idx];
    out_q[idx] = pool_aff[(idx + stride) % count];
}

__global__ void tile_pool_kernel(const JacobianPoint* pool_jac, const AffinePoint* pool_aff, int pool_count, int stride,
                                  JacobianPoint* out_jac, AffinePoint* out_aff, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    int src = idx % pool_count;
    out_jac[idx] = pool_jac[src];
    out_aff[idx] = pool_aff[(src + stride) % pool_count];
}

// ---------------------------------------------------------------------------
// Correctness kernels
// ---------------------------------------------------------------------------

struct CheckResult { bool jac_ok; bool aff_ok; };

__global__ void correctness_double_kernel(const JacobianPoint* in, int n, CheckResult* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    JacobianPoint p = in[idx];
    JacobianPoint prod, alt;
    jacobian_double(&p, &prod);
    jacobian_double_alt(&p, &alt);

    bool jac_ok = (prod.infinity == alt.infinity);
    if (jac_ok && !prod.infinity) {
        jac_ok = field_eq(&prod.x, &alt.x) && field_eq(&prod.y, &alt.y) && field_eq(&prod.z, &alt.z);
    }

    FieldElement pax, pay, aax, aay;
    bool p_finite = jacobian_to_affine(&prod, &pax, &pay);
    bool a_finite = jacobian_to_affine(&alt, &aax, &aay);
    bool aff_ok = (p_finite == a_finite);
    if (aff_ok && p_finite) {
        aff_ok = field_eq(&pax, &aax) && field_eq(&pay, &aay);
    }

    out[idx].jac_ok = jac_ok;
    out[idx].aff_ok = aff_ok;
}

__global__ void correctness_add_kernel(const JacobianPoint* in_p, const AffinePoint* in_q, int n, CheckResult* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    JacobianPoint p = in_p[idx];
    AffinePoint q = in_q[idx];
    JacobianPoint prod, alt;
    jacobian_add_mixed_unchecked(&p, &q, &prod);
    jacobian_add_mixed_alt(&p, &q, &alt);

    bool jac_ok = (prod.infinity == alt.infinity);
    if (jac_ok && !prod.infinity) {
        jac_ok = field_eq(&prod.x, &alt.x) && field_eq(&prod.y, &alt.y) && field_eq(&prod.z, &alt.z);
    }

    FieldElement pax, pay, aax, aay;
    bool p_finite = jacobian_to_affine(&prod, &pax, &pay);
    bool a_finite = jacobian_to_affine(&alt, &aax, &aay);
    bool aff_ok = (p_finite == a_finite);
    if (aff_ok && p_finite) {
        aff_ok = field_eq(&pax, &aax) && field_eq(&pay, &aay);
    }

    out[idx].jac_ok = jac_ok;
    out[idx].aff_ok = aff_ok;
}

// CAT_IDENTITY correctness: production checked jacobian_add_mixed (which
// handles p->infinity) vs. the experiment-only checked wrapper
// jacobian_add_mixed_alt_checked, on a genuine infinity-flagged P.
__global__ void correctness_identity_kernel(const JacobianPoint* in_p, const AffinePoint* in_q, int n, CheckResult* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    JacobianPoint p = in_p[idx];
    AffinePoint q = in_q[idx];
    JacobianPoint prod, alt;
    jacobian_add_mixed(&p, &q, &prod);          // production checked
    jacobian_add_mixed_alt_checked(&p, &q, &alt);

    bool jac_ok = (prod.infinity == alt.infinity);
    if (jac_ok && !prod.infinity) {
        jac_ok = field_eq(&prod.x, &alt.x) && field_eq(&prod.y, &alt.y) && field_eq(&prod.z, &alt.z);
    }

    FieldElement pax, pay, aax, aay;
    bool p_finite = jacobian_to_affine(&prod, &pax, &pay);
    bool a_finite = jacobian_to_affine(&alt, &aax, &aay);
    bool aff_ok = (p_finite == a_finite);
    if (aff_ok && p_finite) {
        aff_ok = field_eq(&pax, &aax) && field_eq(&pay, &aay);
    }

    out[idx].jac_ok = jac_ok;
    out[idx].aff_ok = aff_ok;
}

// ---------------------------------------------------------------------------
// Benchmark kernels
// ---------------------------------------------------------------------------

__global__ void bench_double_prod_kernel(const JacobianPoint* in, JacobianPoint* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    JacobianPoint p = in[idx];
    JacobianPoint r;
    jacobian_double(&p, &r);
    out[idx] = r;
}

__global__ void bench_double_alt_kernel(const JacobianPoint* in, JacobianPoint* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    JacobianPoint p = in[idx];
    JacobianPoint r;
    jacobian_double_alt(&p, &r);
    out[idx] = r;
}

__global__ void bench_add_prod_kernel(const JacobianPoint* in_p, const AffinePoint* in_q, JacobianPoint* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    JacobianPoint p = in_p[idx];
    AffinePoint q = in_q[idx];
    JacobianPoint r;
    jacobian_add_mixed_unchecked(&p, &q, &r);
    out[idx] = r;
}

__global__ void bench_add_alt_kernel(const JacobianPoint* in_p, const AffinePoint* in_q, JacobianPoint* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    JacobianPoint p = in_p[idx];
    AffinePoint q = in_q[idx];
    JacobianPoint r;
    jacobian_add_mixed_alt(&p, &q, &r);
    out[idx] = r;
}

__device__ inline unsigned long long checksum_mix64(unsigned long long x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// DCE-resistant checksum: every kernel writes each result to global memory
// (an observable side effect nvcc cannot eliminate). The benchmark input
// pool is RANDOM_COUNT elements tiled to fill BENCH_N (an exact power-of-two
// multiple), so a plain XOR-fold over the buffer cancels to zero: each
// distinct value appears an even number of times and a^a=0. This reduction
// instead mixes each element with its own global index before accumulating
// via addition, so the repeated period cannot cancel — the fingerprint is
// position-sensitive and stays nonzero for real per-run data.
__global__ void checksum_kernel(const JacobianPoint* buf, int n, unsigned long long* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const JacobianPoint& p = buf[idx];
    unsigned long long local =
          p.x.limbs[0] +  3ULL * p.x.limbs[1] +  5ULL * p.x.limbs[2] +  7ULL * p.x.limbs[3]
        + 11ULL * p.y.limbs[0] + 13ULL * p.y.limbs[1] + 17ULL * p.y.limbs[2] + 19ULL * p.y.limbs[3]
        + 23ULL * p.z.limbs[0] + 29ULL * p.z.limbs[1] + 31ULL * p.z.limbs[2] + 37ULL * p.z.limbs[3]
        + (unsigned long long)(p.infinity ? 1u : 0u);
    unsigned long long h = checksum_mix64(local ^ checksum_mix64((unsigned long long)idx));
    atomicAdd(out, h);
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------

enum Category { CAT_INFINITY = 0, CAT_Y_ZERO = 1, CAT_IDENTITY = 2, CAT_EQUAL = 3,
                CAT_OPPOSITE = 4, CAT_DETERMINISTIC = 5, CAT_RANDOM = 6, CAT_COUNT = 7 };

static const char* CATEGORY_NAMES[CAT_COUNT] = {
    "infinity", "y_zero", "identity", "equal_points", "opposite_points",
    "deterministic_scalar_edges", "randomized_valid_points"
};

struct CategoryTally { long total = 0; long jac_pass = 0; long aff_pass = 0; };

static void tally(std::vector<CategoryTally>& tallies, const std::vector<int>& cats,
                   const std::vector<CheckResult>& res) {
    for (size_t i = 0; i < res.size(); ++i) {
        CategoryTally& t = tallies[cats[i]];
        t.total++;
        if (res[i].jac_ok) t.jac_pass++;
        if (res[i].aff_ok) t.aff_pass++;
    }
}

static float median_ms(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 0) return (v[n / 2 - 1] + v[n / 2]) / 2.0f;
    return v[n / 2];
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int driverVersion = 0, runtimeVersion = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtimeVersion));

    const int POW2_COUNT = 40;
    const int RANDOM_COUNT = 512;
    const int EDGE_D = 2;
    const int EDGE_A = 2;       // equal, opposite (unchecked formulas)
    const int IDENTITY_N = 1;   // infinity P, valid Q (checked formulas)
    const int TOTAL_D = EDGE_D + POW2_COUNT + RANDOM_COUNT;
    const int TOTAL_A = EDGE_A + POW2_COUNT + RANDOM_COUNT;
    const int BENCH_N = 1 << 20;
    const int WARMUP = 4;  // even: exactly 2 production-first + 2 alt-first warmups
    const int PASSES = 8;  // even: exactly 4 production-first + 4 alt-first measured passes
    static_assert(WARMUP % 2 == 0, "WARMUP must be even for an exact 50/50 AB/BA split");
    static_assert(PASSES % 2 == 0, "PASSES must be even for an exact 50/50 AB/BA split");
    const int BLOCK = 256;

    // --- generate base pools (ground truth: production formulas only) ---
    JacobianPoint *d_pow2_jac; AffinePoint *d_pow2_aff;
    CUDA_CHECK(cudaMalloc(&d_pow2_jac, POW2_COUNT * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_pow2_aff, POW2_COUNT * sizeof(AffinePoint)));
    gen_pow2_pool_kernel<<<1, 1>>>(d_pow2_jac, d_pow2_aff, POW2_COUNT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    JacobianPoint *d_rand_jac; AffinePoint *d_rand_aff;
    CUDA_CHECK(cudaMalloc(&d_rand_jac, RANDOM_COUNT * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_rand_aff, RANDOM_COUNT * sizeof(AffinePoint)));
    gen_random_pool_kernel<<<1, 1>>>(d_pow2_jac, d_pow2_aff, POW2_COUNT, d_rand_jac, d_rand_aff, RANDOM_COUNT, FIXED_SEED);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- assemble doubling test inputs ---
    JacobianPoint* d_all_d;
    CUDA_CHECK(cudaMalloc(&d_all_d, TOTAL_D * sizeof(JacobianPoint)));
    gen_edge_double_kernel<<<1, 1>>>(d_pow2_jac, d_all_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_all_d + EDGE_D, d_pow2_jac, POW2_COUNT * sizeof(JacobianPoint), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_d + EDGE_D + POW2_COUNT, d_rand_jac, RANDOM_COUNT * sizeof(JacobianPoint), cudaMemcpyDeviceToDevice));

    std::vector<int> h_cat_d(TOTAL_D);
    h_cat_d[0] = CAT_INFINITY; h_cat_d[1] = CAT_Y_ZERO;
    for (int i = 0; i < POW2_COUNT; ++i) h_cat_d[EDGE_D + i] = CAT_DETERMINISTIC;
    for (int i = 0; i < RANDOM_COUNT; ++i) h_cat_d[EDGE_D + POW2_COUNT + i] = CAT_RANDOM;

    CheckResult* d_res_d;
    CUDA_CHECK(cudaMalloc(&d_res_d, TOTAL_D * sizeof(CheckResult)));
    {
        int grid = (TOTAL_D + BLOCK - 1) / BLOCK;
        correctness_double_kernel<<<grid, BLOCK>>>(d_all_d, TOTAL_D, d_res_d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<CheckResult> h_res_d(TOTAL_D);
    CUDA_CHECK(cudaMemcpy(h_res_d.data(), d_res_d, TOTAL_D * sizeof(CheckResult), cudaMemcpyDeviceToHost));

    // --- assemble mixed-add test inputs ---
    JacobianPoint* d_all_a_p; AffinePoint* d_all_a_q;
    CUDA_CHECK(cudaMalloc(&d_all_a_p, TOTAL_A * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_all_a_q, TOTAL_A * sizeof(AffinePoint)));
    gen_edge_add_kernel<<<1, 1>>>(d_pow2_jac, d_pow2_aff, 3, d_all_a_p, d_all_a_q);
    CUDA_CHECK(cudaGetLastError());
    {
        int grid = (POW2_COUNT + BLOCK - 1) / BLOCK;
        gen_pairs_kernel<<<grid, BLOCK>>>(d_pow2_jac, d_pow2_aff, POW2_COUNT, 1, d_all_a_p + EDGE_A, d_all_a_q + EDGE_A);
        CUDA_CHECK(cudaGetLastError());
    }
    {
        int grid = (RANDOM_COUNT + BLOCK - 1) / BLOCK;
        gen_pairs_kernel<<<grid, BLOCK>>>(d_rand_jac, d_rand_aff, RANDOM_COUNT, 3,
                                          d_all_a_p + EDGE_A + POW2_COUNT, d_all_a_q + EDGE_A + POW2_COUNT);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int> h_cat_a(TOTAL_A);
    h_cat_a[0] = CAT_EQUAL; h_cat_a[1] = CAT_OPPOSITE;
    for (int i = 0; i < POW2_COUNT; ++i) h_cat_a[EDGE_A + i] = CAT_DETERMINISTIC;
    for (int i = 0; i < RANDOM_COUNT; ++i) h_cat_a[EDGE_A + POW2_COUNT + i] = CAT_RANDOM;

    CheckResult* d_res_a;
    CUDA_CHECK(cudaMalloc(&d_res_a, TOTAL_A * sizeof(CheckResult)));
    {
        int grid = (TOTAL_A + BLOCK - 1) / BLOCK;
        correctness_add_kernel<<<grid, BLOCK>>>(d_all_a_p, d_all_a_q, TOTAL_A, d_res_a);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<CheckResult> h_res_a(TOTAL_A);
    CUDA_CHECK(cudaMemcpy(h_res_a.data(), d_res_a, TOTAL_A * sizeof(CheckResult), cudaMemcpyDeviceToHost));

    // --- genuine CAT_IDENTITY: infinity P, valid affine Q, checked formulas ---
    JacobianPoint* d_id_p; AffinePoint* d_id_q;
    CUDA_CHECK(cudaMalloc(&d_id_p, IDENTITY_N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_id_q, IDENTITY_N * sizeof(AffinePoint)));
    gen_identity_case_kernel<<<1, 1>>>(d_pow2_aff, 3, d_id_p, d_id_q);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CheckResult* d_res_id;
    CUDA_CHECK(cudaMalloc(&d_res_id, IDENTITY_N * sizeof(CheckResult)));
    correctness_identity_kernel<<<1, IDENTITY_N>>>(d_id_p, d_id_q, IDENTITY_N, d_res_id);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<CheckResult> h_res_id(IDENTITY_N);
    CUDA_CHECK(cudaMemcpy(h_res_id.data(), d_res_id, IDENTITY_N * sizeof(CheckResult), cudaMemcpyDeviceToHost));
    std::vector<int> h_cat_id(IDENTITY_N, CAT_IDENTITY);

    // --- aggregate correctness ---
    std::vector<CategoryTally> tally_d(CAT_COUNT), tally_a(CAT_COUNT);
    tally(tally_d, h_cat_d, h_res_d);
    tally(tally_a, h_cat_a, h_res_a);
    tally(tally_a, h_cat_id, h_res_id);

    bool all_pass = true;
    for (const auto& r : h_res_d) if (!r.jac_ok || !r.aff_ok) all_pass = false;
    for (const auto& r : h_res_a) if (!r.jac_ok || !r.aff_ok) all_pass = false;
    for (const auto& r : h_res_id) if (!r.jac_ok || !r.aff_ok) all_pass = false;

    // --- JSON: header + correctness (always emitted) ---
    std::printf("{\n");
    std::printf("  \"gpu\": \"%s\",\n", prop.name);
    std::printf("  \"driver_runtime_compiler\": {\n");
    std::printf("    \"driver_version\": %d,\n", driverVersion);
    std::printf("    \"runtime_version\": %d,\n", runtimeVersion);
#if defined(__CUDACC_VER_MAJOR__)
    std::printf("    \"nvcc_version\": \"%d.%d.%d\"\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
#else
    std::printf("    \"nvcc_version\": \"unknown\"\n");
#endif
    std::printf("  },\n");

    std::printf("  \"formulas\": [\n");
    std::printf("    {\"name\": \"jacobian_double_production\", \"op_count\": \"3M+4S\"},\n");
    std::printf("    {\"name\": \"jacobian_double_alt_square_identity\", \"op_count\": \"2M+5S\"},\n");
    std::printf("    {\"name\": \"jacobian_add_mixed_production_madd2007bl\", \"op_count\": \"7M+4S\"},\n");
    std::printf("    {\"name\": \"jacobian_add_mixed_alt_z3_identity\", \"op_count\": \"8M+3S\"}\n");
    std::printf("  ],\n");

    std::printf("  \"correctness\": {\n");
    std::printf("    \"overall_pass\": %s,\n", all_pass ? "true" : "false");
    std::printf("    \"fixed_seed\": \"0x%016llx\",\n", (unsigned long long)FIXED_SEED);
    std::printf("    \"double_categories\": {\n");
    for (int c = 0; c < CAT_COUNT; ++c) {
        if (tally_d[c].total == 0) continue;
        std::printf("      \"%s\": {\"total\": %ld, \"jac_pass\": %ld, \"aff_pass\": %ld}%s\n",
                    CATEGORY_NAMES[c], tally_d[c].total, tally_d[c].jac_pass, tally_d[c].aff_pass,
                    (c == CAT_COUNT - 1) ? "" : ",");
    }
    std::printf("    },\n");
    std::printf("    \"add_categories\": {\n");
    for (int c = 0; c < CAT_COUNT; ++c) {
        if (tally_a[c].total == 0) continue;
        std::printf("      \"%s\": {\"total\": %ld, \"jac_pass\": %ld, \"aff_pass\": %ld}%s\n",
                    CATEGORY_NAMES[c], tally_a[c].total, tally_a[c].jac_pass, tally_a[c].aff_pass,
                    (c == CAT_COUNT - 1) ? "" : ",");
    }
    std::printf("    }\n");
    std::printf("  }");

    if (!all_pass) {
        std::printf(",\n  \"benchmark\": null\n}\n");
        std::fprintf(stderr, "CORRECTNESS FAILED: alt formulas diverge from production. Timing suppressed.\n");
        return 1;
    }
    std::printf(",\n");

    // --- benchmark (only reached if correctness fully passed) ---
    JacobianPoint *d_bench_in_jac;
    AffinePoint *d_bench_in_aff;
    JacobianPoint *d_bench_out_prod, *d_bench_out_alt;
    CUDA_CHECK(cudaMalloc(&d_bench_in_jac, BENCH_N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_bench_in_aff, BENCH_N * sizeof(AffinePoint)));
    CUDA_CHECK(cudaMalloc(&d_bench_out_prod, BENCH_N * sizeof(JacobianPoint)));
    CUDA_CHECK(cudaMalloc(&d_bench_out_alt, BENCH_N * sizeof(JacobianPoint)));
    {
        int grid = (BENCH_N + BLOCK - 1) / BLOCK;
        tile_pool_kernel<<<grid, BLOCK>>>(d_rand_jac, d_rand_aff, RANDOM_COUNT, 1, d_bench_in_jac, d_bench_in_aff, BENCH_N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    const int grid = (BENCH_N + BLOCK - 1) / BLOCK;

    // Two distinct output buffers (production/alt) so an alternating AB/BA
    // launch order never lets one formula's write clobber the other's
    // pending checksum evidence.
    auto compute_checksum = [&](const JacobianPoint* buf) -> unsigned long long {
        unsigned long long* d_sum;
        CUDA_CHECK(cudaMalloc(&d_sum, sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(unsigned long long)));
        checksum_kernel<<<grid, BLOCK>>>(buf, BENCH_N, d_sum);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        unsigned long long h_sum = 0;
        CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_sum));
        return h_sum;
    };

    struct PairBenchOut {
        const char* name_prod;
        const char* name_alt;
        std::vector<float> ms_prod;
        std::vector<float> ms_alt;
        std::vector<int> pass_order; // 0 = production launched first this pass, 1 = alt launched first
        unsigned long long checksum_prod;
        unsigned long long checksum_alt;
    };

    // Deterministic balanced paired AB/BA schedule: each measured pass times
    // BOTH formulas of a pair back-to-back, alternating which one goes
    // first (production on even pass indices, alt on odd) so systematic
    // drift (clock ramp, thermal state) cannot land disproportionately on
    // either formula. Warmups alternate the same way. WARMUP and PASSES are
    // both even (enforced where they are declared), so this alternation is
    // an exact 50/50 split, not an approximation. See top-of-file comment
    // for rationale.
    auto run_bench_pair = [&](const char* name_prod, const char* name_alt,
                               auto launch_prod, auto launch_alt,
                               const JacobianPoint* out_prod, const JacobianPoint* out_alt) {
        for (int w = 0; w < WARMUP; ++w) {
            if (w % 2 == 0) { launch_prod(); launch_alt(); }
            else            { launch_alt(); launch_prod(); }
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> ms_prod(PASSES), ms_alt(PASSES);
        std::vector<int> pass_order(PASSES);

        auto timed = [&](auto launch) -> float {
            CUDA_CHECK(cudaEventRecord(start));
            launch();
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventSynchronize(stop));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            return ms;
        };

        for (int i = 0; i < PASSES; ++i) {
            bool prod_first = (i % 2 == 0);
            pass_order[i] = prod_first ? 0 : 1;
            if (prod_first) {
                ms_prod[i] = timed(launch_prod);
                ms_alt[i]  = timed(launch_alt);
            } else {
                ms_alt[i]  = timed(launch_alt);
                ms_prod[i] = timed(launch_prod);
            }
        }

        return PairBenchOut{name_prod, name_alt, ms_prod, ms_alt, pass_order,
                             compute_checksum(out_prod), compute_checksum(out_alt)};
    };

    PairBenchOut double_pair = run_bench_pair(
        "jacobian_double_production", "jacobian_double_alt_square_identity",
        [&]() { bench_double_prod_kernel<<<grid, BLOCK>>>(d_bench_in_jac, d_bench_out_prod, BENCH_N); },
        [&]() { bench_double_alt_kernel<<<grid, BLOCK>>>(d_bench_in_jac, d_bench_out_alt, BENCH_N); },
        d_bench_out_prod, d_bench_out_alt);

    PairBenchOut add_pair = run_bench_pair(
        "jacobian_add_mixed_production_madd2007bl", "jacobian_add_mixed_alt_z3_identity",
        [&]() { bench_add_prod_kernel<<<grid, BLOCK>>>(d_bench_in_jac, d_bench_in_aff, d_bench_out_prod, BENCH_N); },
        [&]() { bench_add_alt_kernel<<<grid, BLOCK>>>(d_bench_in_jac, d_bench_in_aff, d_bench_out_alt, BENCH_N); },
        d_bench_out_prod, d_bench_out_alt);

    // Checksum evidence hard-gate: a real, non-cancelling fingerprint must be
    // nonzero for real data, and since the alt kernels are algebraically
    // proven equal to production and already verified bit-for-bit at
    // correctness-pool scale, their full BENCH_N-scale checksums must match
    // exactly too. Any violation means the checksum (or an alt formula) is
    // not trustworthy evidence, so timing is suppressed just like a
    // correctness-pool failure.
    bool checksum_nonzero = double_pair.checksum_prod != 0 && double_pair.checksum_alt != 0 &&
                             add_pair.checksum_prod != 0 && add_pair.checksum_alt != 0;
    bool checksum_double_match = (double_pair.checksum_prod == double_pair.checksum_alt);
    bool checksum_add_match = (add_pair.checksum_prod == add_pair.checksum_alt);
    if (!checksum_nonzero || !checksum_double_match || !checksum_add_match) {
        std::printf("  \"benchmark\": null\n}\n");
        std::fprintf(stderr,
            "CHECKSUM EVIDENCE FAILED: nonzero=%d double_production_alt_equal=%d add_production_alt_equal=%d\n",
            checksum_nonzero, checksum_double_match, checksum_add_match);
        return 1;
    }

    auto print_pair_json = [&](const PairBenchOut& pb, const char* pair_id, bool last) {
        std::printf("    {\n");
        std::printf("      \"pair_id\": \"%s\",\n", pair_id);
        std::printf("      \"formula_production\": \"%s\",\n", pb.name_prod);
        std::printf("      \"formula_alt\": \"%s\",\n", pb.name_alt);
        std::printf("      \"pass_order\": [");
        for (size_t i = 0; i < pb.pass_order.size(); ++i)
            std::printf("%s\"%s\"", i ? ", " : "", pb.pass_order[i] == 0 ? "production_first" : "alt_first");
        std::printf("],\n");
        std::printf("      \"checksum_evidence\": {\"nonzero\": true, \"production_alt_equal\": true},\n");
        std::printf("      \"results\": [\n");
        std::printf("        {\n");
        std::printf("          \"formula\": \"%s\",\n", pb.name_prod);
        std::printf("          \"raw_timings_ms\": [");
        for (size_t i = 0; i < pb.ms_prod.size(); ++i) std::printf("%s%.6f", i ? ", " : "", pb.ms_prod[i]);
        std::printf("],\n");
        std::printf("          \"median_ms\": %.6f,\n", median_ms(pb.ms_prod));
        std::printf("          \"checksum\": \"0x%016llx\"\n", pb.checksum_prod);
        std::printf("        },\n");
        std::printf("        {\n");
        std::printf("          \"formula\": \"%s\",\n", pb.name_alt);
        std::printf("          \"raw_timings_ms\": [");
        for (size_t i = 0; i < pb.ms_alt.size(); ++i) std::printf("%s%.6f", i ? ", " : "", pb.ms_alt[i]);
        std::printf("],\n");
        std::printf("          \"median_ms\": %.6f,\n", median_ms(pb.ms_alt));
        std::printf("          \"checksum\": \"0x%016llx\"\n", pb.checksum_alt);
        std::printf("        }\n");
        std::printf("      ]\n");
        std::printf("    }%s\n", last ? "" : ",");
    };

    std::printf("  \"benchmark\": {\n");
    std::printf("    \"sample_count\": %d,\n", BENCH_N);
    std::printf("    \"warmup_passes\": %d,\n", WARMUP);
    std::printf("    \"measured_passes\": %d,\n", PASSES);
    std::printf("    \"schedule\": \"balanced_paired_ab_ba: WARMUP and PASSES are both even, so warmups and measured passes alternate the starting formula every iteration (production-first on even index, alt-first on odd) for an exact 50/50 split -- warmup_passes/2 production-first + warmup_passes/2 alt-first warmups, measured_passes/2 production-first + measured_passes/2 alt-first measured passes -- to cancel systematic clock-ramp/thermal-drift bias -- see per-pair pass_order\",\n");
    std::printf("    \"pairs\": [\n");
    print_pair_json(double_pair, "jacobian_double", false);
    print_pair_json(add_pair, "jacobian_add_mixed", true);
    std::printf("    ]\n");
    std::printf("  }\n");
    std::printf("}\n");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return 0;
}
