// EXP_CUDA_POINT_REPRESENTATION_SEARCH_CLAUDE_715 — correctness-only companion test.
//
// Standalone, fast (no benchmark timing) correctness gate for the candidate
// formulas defined in cuda_point_formula_search.cu (duplicated here since
// each .cu in this experiment must compile independently; no shared header
// is in scope). See that file's top comment for the algebraic derivation of
// both candidates.
//
// Duplication contract: this file is an intentional, independent copy of
// the `_alt` candidate bodies and pool-generation kernels from
// cuda_point_formula_search.cu — the benchmark binary already re-validates
// its own candidates before timing, and no third shared-header file is
// permitted in this experiment's two-file scope. Any change to a candidate
// formula or a pool generator in either file MUST be mirrored in the other.
//
// Exits 0 only if every check (y=0 and infinity on doubling; identity,
// equal points, and opposite points on mixed-add; deterministic scalar
// edges; randomized valid points; and affine/projective equivalence)
// matches the production formula bit-for-bit.
//
// jacobian_add_mixed_unchecked intentionally ignores p->infinity (it is the
// "_unchecked" variant), so equal/opposite/deterministic/random cases stay
// on the unchecked formulas (P is always finite there). Genuine
// CAT_IDENTITY coverage (P = an infinity-flagged Jacobian point, Q = a
// valid finite affine point) is tested separately by comparing the
// production *checked* jacobian_add_mixed against
// jacobian_add_mixed_alt_checked, an experiment-only wrapper that mirrors
// production's `if (p->infinity) { r = q; return; }` branch and otherwise
// delegates to jacobian_add_mixed_alt. Exits nonzero on any mismatch or
// CUDA error.
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
//        -o ../../out/experiments/test_cuda_point_formula_search \
//        test_cuda_point_formula_search.cu
//
// Run:
//   ../../out/experiments/test_cuda_point_formula_search
// ---------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

#include "../../src/cuda/include/secp256k1.cuh"

using namespace secp256k1::cuda;

#if defined(SECP256K1_CUDA_LIMBS_32)
#error "This test assumes the default 64-bit-limb FieldElement layout (SECP256K1_CUDA_LIMBS_32 must stay undefined)."
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

// --- candidate formulas (see cuda_point_formula_search.cu for derivation) ---

__device__ inline void jacobian_double_alt(const JacobianPoint* p, JacobianPoint* r) {
    if (p->infinity) { r->infinity = true; return; }
    if (field_is_zero(&p->y)) { r->infinity = true; return; }

    FieldElement yy, xx, yyyy, t, s, m, x3, y3, z3;
    field_sqr(&p->y, &yy);
    field_sqr(&p->x, &xx);
    field_sqr(&yy, &yyyy);

    field_add(&p->x, &yy, &t);
    field_sqr(&t, &t);
    field_sub(&t, &xx, &t);
    field_sub(&t, &yyyy, &t);
    field_add(&t, &t, &s);

    field_add(&xx, &xx, &m);
    field_add(&m, &xx, &m);

    field_sqr(&m, &x3);
    field_add(&s, &s, &t);
    field_sub(&x3, &t, &x3);

    field_sub(&s, &x3, &t);
    field_mul(&m, &t, &y3);
    field_add(&yyyy, &yyyy, &t);
    field_add(&t, &t, &t);
    field_add(&t, &t, &t);
    field_sub(&y3, &t, &y3);

    field_mul(&p->y, &p->z, &z3);
    field_add(&z3, &z3, &z3);

    r->x = x3; r->y = y3; r->z = z3; r->infinity = false;
}

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

// --- fixed-seed PRNG + pool generation (kept small: this is a fast gate) ---

static constexpr uint64_t FIXED_SEED = 0xC0FFEE123456789ULL;

__device__ inline uint64_t splitmix64_next(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

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
        jacobian_double(&jac_out[idx - 1], &r);
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
        jacobian_add_mixed_unchecked(&pow2_jac[i], &pow2_aff[j], &combined);
        if (splitmix64_next(state) & 1ULL) {
            JacobianPoint doubled;
            jacobian_double(&combined, &doubled);
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
    out[0] = inf_pt;

    JacobianPoint yzero_pt = base;
    yzero_pt.infinity = false;
    field_set_zero(&yzero_pt.y);
    out[1] = yzero_pt;
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

// --- correctness kernels ---

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

// --- host driver ---

enum Category { CAT_INFINITY = 0, CAT_Y_ZERO = 1, CAT_IDENTITY = 2, CAT_EQUAL = 3,
                CAT_OPPOSITE = 4, CAT_DETERMINISTIC = 5, CAT_RANDOM = 6, CAT_COUNT = 7 };

static const char* CATEGORY_NAMES[CAT_COUNT] = {
    "infinity", "y_zero", "identity", "equal_points", "opposite_points",
    "deterministic_scalar_edges", "randomized_valid_points"
};

int main() {
    const int POW2_COUNT = 16;
    const int RANDOM_COUNT = 64;
    const int EDGE_D = 2;
    const int EDGE_A = 2;       // equal, opposite (unchecked formulas)
    const int IDENTITY_N = 1;   // infinity P, valid Q (checked formulas)
    const int TOTAL_D = EDGE_D + POW2_COUNT + RANDOM_COUNT;
    const int TOTAL_A = EDGE_A + POW2_COUNT + RANDOM_COUNT;
    const int BLOCK = 128;

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

    JacobianPoint* d_all_d;
    CUDA_CHECK(cudaMalloc(&d_all_d, TOTAL_D * sizeof(JacobianPoint)));
    gen_edge_double_kernel<<<1, 1>>>(d_pow2_jac, d_all_d);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(d_all_d + EDGE_D, d_pow2_jac, POW2_COUNT * sizeof(JacobianPoint), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_d + EDGE_D + POW2_COUNT, d_rand_jac, RANDOM_COUNT * sizeof(JacobianPoint), cudaMemcpyDeviceToDevice));

    std::vector<int> cat_d(TOTAL_D);
    cat_d[0] = CAT_INFINITY; cat_d[1] = CAT_Y_ZERO;
    for (int i = 0; i < POW2_COUNT; ++i) cat_d[EDGE_D + i] = CAT_DETERMINISTIC;
    for (int i = 0; i < RANDOM_COUNT; ++i) cat_d[EDGE_D + POW2_COUNT + i] = CAT_RANDOM;

    CheckResult* d_res_d;
    CUDA_CHECK(cudaMalloc(&d_res_d, TOTAL_D * sizeof(CheckResult)));
    {
        int grid = (TOTAL_D + BLOCK - 1) / BLOCK;
        correctness_double_kernel<<<grid, BLOCK>>>(d_all_d, TOTAL_D, d_res_d);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<CheckResult> res_d(TOTAL_D);
    CUDA_CHECK(cudaMemcpy(res_d.data(), d_res_d, TOTAL_D * sizeof(CheckResult), cudaMemcpyDeviceToHost));

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

    std::vector<int> cat_a(TOTAL_A);
    cat_a[0] = CAT_EQUAL; cat_a[1] = CAT_OPPOSITE;
    for (int i = 0; i < POW2_COUNT; ++i) cat_a[EDGE_A + i] = CAT_DETERMINISTIC;
    for (int i = 0; i < RANDOM_COUNT; ++i) cat_a[EDGE_A + POW2_COUNT + i] = CAT_RANDOM;

    CheckResult* d_res_a;
    CUDA_CHECK(cudaMalloc(&d_res_a, TOTAL_A * sizeof(CheckResult)));
    {
        int grid = (TOTAL_A + BLOCK - 1) / BLOCK;
        correctness_add_kernel<<<grid, BLOCK>>>(d_all_a_p, d_all_a_q, TOTAL_A, d_res_a);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    std::vector<CheckResult> res_a(TOTAL_A);
    CUDA_CHECK(cudaMemcpy(res_a.data(), d_res_a, TOTAL_A * sizeof(CheckResult), cudaMemcpyDeviceToHost));

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
    std::vector<CheckResult> res_id(IDENTITY_N);
    CUDA_CHECK(cudaMemcpy(res_id.data(), d_res_id, IDENTITY_N * sizeof(CheckResult), cudaMemcpyDeviceToHost));
    std::vector<int> cat_id(IDENTITY_N, CAT_IDENTITY);

    long fail_count = 0;
    long per_cat_total[CAT_COUNT] = {0}, per_cat_fail[CAT_COUNT] = {0};

    for (size_t idx = 0; idx < res_d.size(); ++idx) {
        int c = cat_d[idx];
        per_cat_total[c]++;
        if (!res_d[idx].jac_ok || !res_d[idx].aff_ok) { fail_count++; per_cat_fail[c]++; }
    }
    for (size_t idx = 0; idx < res_a.size(); ++idx) {
        int c = cat_a[idx];
        per_cat_total[c]++;
        if (!res_a[idx].jac_ok || !res_a[idx].aff_ok) { fail_count++; per_cat_fail[c]++; }
    }
    for (size_t idx = 0; idx < res_id.size(); ++idx) {
        int c = cat_id[idx];
        per_cat_total[c]++;
        if (!res_id[idx].jac_ok || !res_id[idx].aff_ok) { fail_count++; per_cat_fail[c]++; }
    }

    std::printf("jacobian_double_alt / jacobian_add_mixed_alt correctness gate\n");
    std::printf("fixed_seed=0x%016llx double_cases=%d add_cases=%d\n",
                (unsigned long long)FIXED_SEED, TOTAL_D, TOTAL_A + IDENTITY_N);
    for (int c = 0; c < CAT_COUNT; ++c) {
        if (per_cat_total[c] == 0) continue;
        std::printf("  [%s] total=%ld fail=%ld\n", CATEGORY_NAMES[c], per_cat_total[c], per_cat_fail[c]);
    }

    if (fail_count != 0) {
        std::fprintf(stderr, "FAIL: %ld mismatch(es) between alt and production formulas\n", fail_count);
        return 1;
    }

    std::printf("PASS: all %d correctness checks matched production bit-for-bit\n", TOTAL_D + TOTAL_A + IDENTITY_N);
    return 0;
}
