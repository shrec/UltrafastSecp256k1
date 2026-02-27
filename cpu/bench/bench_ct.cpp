// ============================================================================
// Constant-Time (CT) Layer Benchmarks
// ============================================================================
// Benchmarks for side-channel-resistant operations.
// Compare fast:: vs ct:: to quantify the protection cost.
// ============================================================================

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/selftest.hpp"
#include "secp256k1/ct/field.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/benchmark_harness.hpp"

#include <cstdio>
#include <cstdint>

using namespace secp256k1::fast;
namespace ct = secp256k1::ct;

// Unified harness: 500 warmup, 11 passes, RDTSC on x86, IQR outlier removal
static bench::Harness H(500, 11);

// Convenience: run and return microseconds/iter
template <typename Func>
static double bench_us(Func&& f, int iters) {
    return H.run(iters, std::forward<Func>(f)) / 1000.0;
}

// -- main ---------------------------------------------------------------------

int main() {
    bench::pin_thread_and_elevate();

    printf("================================================================\n");
    printf("  CT Layer Benchmark  (fast:: vs ct::)\n");
    printf("================================================================\n");
    H.print_config();
    printf("\n");

    // Fixed test data -- all 256-bit, representative of real workloads
    auto G  = Point::generator();
    auto k  = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto k2 = Scalar::from_hex(
        "7c076ff316692a3d7eb3c3bb0f8b1488cf72e1afcd929e29307032997a838a3d");
    auto P  = G.scalar_mul(k);

    auto fe_a = FieldElement::from_hex(
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    auto fe_b = FieldElement::from_hex(
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");

    // Pool of random 256-bit scalars to prevent branch-predictor warming
    // and ensure benchmarks reflect real-world workloads with varying inputs.
    constexpr int POOL = 32;
    Scalar scalar_pool[POOL];
    Point  point_pool[POOL];
    {
        auto base = Scalar::from_hex(
            "b5037ebecae0da656179c623f6cb73641db2aa0fabe888ffb78466fa18470379");
        auto step = Scalar::from_hex(
            "9e3779b97f4a7c15f39cc0605cedc8341082276bf3a27251f86c6a11d0c18e95");
        for (int i = 0; i < POOL; ++i) {
            scalar_pool[i] = base;
            point_pool[i]  = G.scalar_mul(base);
            base += step;
        }
    }

    constexpr int N_SCALAR_MUL = 50;
    constexpr int N_POINT_OPS  = 5000;
    constexpr int N_FIELD_OPS  = 50000;
    constexpr int N_SCALAR_OPS = 50000;

    printf("  Pool size: %d random 256-bit scalars\n\n", POOL);

    // -- Field operations -----------------------------------------------------

    printf("--- Field Arithmetic ---\n");

    double const fast_field_mul = bench_us([&]() {
        auto r = fe_a * fe_b;
        bench::DoNotOptimize(r);
    }, N_FIELD_OPS);

    double const ct_field_mul = bench_us([&]() {
        auto r = ct::field_mul(fe_a, fe_b);
        bench::DoNotOptimize(r);
    }, N_FIELD_OPS);

    printf("  field_mul    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_field_mul, ct_field_mul, ct_field_mul / fast_field_mul);

    double const fast_field_sq = bench_us([&]() {
        auto r = fe_a.square();
        bench::DoNotOptimize(r);
    }, N_FIELD_OPS);

    double const ct_field_sq = bench_us([&]() {
        auto r = ct::field_sqr(fe_a);
        bench::DoNotOptimize(r);
    }, N_FIELD_OPS);

    printf("  field_square fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_field_sq, ct_field_sq, ct_field_sq / fast_field_sq);

    double const fast_field_inv = bench_us([&]() {
        auto r = fe_a.inverse();
        bench::DoNotOptimize(r);
    }, N_FIELD_OPS / 10);

    double const ct_field_inv = bench_us([&]() {
        auto r = ct::field_inv(fe_a);
        bench::DoNotOptimize(r);
    }, N_FIELD_OPS / 10);

    printf("  field_inv    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n\n",
           fast_field_inv, ct_field_inv, ct_field_inv / fast_field_inv);

    // -- Scalar operations ----------------------------------------------------

    printf("--- Scalar Arithmetic ---\n");

    double const fast_scalar_add = bench_us([&]() {
        auto r = k + k2;
        bench::DoNotOptimize(r);
    }, N_SCALAR_OPS);

    double const ct_scalar_add = bench_us([&]() {
        auto r = ct::scalar_add(k, k2);
        bench::DoNotOptimize(r);
    }, N_SCALAR_OPS);

    printf("  scalar_add   fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_scalar_add, ct_scalar_add, ct_scalar_add / fast_scalar_add);

    double const fast_scalar_sub = bench_us([&]() {
        auto r = k - k2;
        bench::DoNotOptimize(r);
    }, N_SCALAR_OPS);

    double const ct_scalar_sub = bench_us([&]() {
        auto r = ct::scalar_sub(k, k2);
        bench::DoNotOptimize(r);
    }, N_SCALAR_OPS);

    printf("  scalar_sub   fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n\n",
           fast_scalar_sub, ct_scalar_sub, ct_scalar_sub / fast_scalar_sub);

    // -- Point operations -----------------------------------------------------

    printf("--- Point Operations ---\n");

    auto ct_p = ct::CTJacobianPoint::from_point(P);
    auto ct_g = ct::CTJacobianPoint::from_point(G);

    double const fast_point_add = bench_us([&]() {
        auto r = P.add(G);
        bench::DoNotOptimize(r);
    }, N_POINT_OPS);

    double const ct_point_add = bench_us([&]() {
        auto r = ct::point_add_complete(ct_p, ct_g);
        bench::DoNotOptimize(r);
    }, N_POINT_OPS);

    printf("  point_add    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_point_add, ct_point_add, ct_point_add / fast_point_add);

    // Mixed add (J+A) -- comparable to libsecp's group_add_affine
    auto ct_aff_g = ct::CTAffinePoint::from_point(G);

    double const ct_mixed_add = bench_us([&]() {
        auto r = ct::point_add_mixed_complete(ct_p, ct_aff_g);
        bench::DoNotOptimize(r);
    }, N_POINT_OPS);

    printf("  mixed_add    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n",
           fast_point_add, ct_mixed_add, ct_mixed_add / fast_point_add);

    double const fast_point_dbl = bench_us([&]() {
        auto r = P.dbl();
        bench::DoNotOptimize(r);
    }, N_POINT_OPS);

    double const ct_point_dbl = bench_us([&]() {
        auto r = ct::point_dbl(ct_p);
        bench::DoNotOptimize(r);
    }, N_POINT_OPS);

    printf("  point_dbl    fast: %8.3f us   ct: %8.3f us   ratio: %.2fx\n\n",
           fast_point_dbl, ct_point_dbl, ct_point_dbl / fast_point_dbl);

    // -- Scalar multiplication ------------------------------------------------

    printf("--- Scalar Multiplication (k * P) ---\n");

    int idx_fast_mul = 0;
    double const fast_mul = bench_us([&]() {
        auto r = point_pool[idx_fast_mul % POOL].scalar_mul(scalar_pool[idx_fast_mul % POOL]);
        bench::DoNotOptimize(r);
        ++idx_fast_mul;
    }, N_SCALAR_MUL);

    int idx_ct_mul = 0;
    double const ct_mul = bench_us([&]() {
        auto r = ct::scalar_mul(point_pool[idx_ct_mul % POOL], scalar_pool[idx_ct_mul % POOL]);
        bench::DoNotOptimize(r);
        ++idx_ct_mul;
    }, N_SCALAR_MUL);

    printf("  scalar_mul   fast: %8.1f us   ct: %8.1f us   ratio: %.2fx\n\n",
           fast_mul, ct_mul, ct_mul / fast_mul);

    // -- Generator multiplication ---------------------------------------------

    printf("--- Generator Multiplication (k * G) ---\n");

    int idx_fast_gen = 0;
    double const fast_gen = bench_us([&]() {
        auto r = G.scalar_mul(scalar_pool[idx_fast_gen % POOL]);
        bench::DoNotOptimize(r);
        ++idx_fast_gen;
    }, N_SCALAR_MUL);

    int idx_ct_gen = 0;
    double const ct_gen = bench_us([&]() {
        auto r = ct::generator_mul(scalar_pool[idx_ct_gen % POOL]);
        bench::DoNotOptimize(r);
        ++idx_ct_gen;
    }, N_SCALAR_MUL);

    printf("  generator_mul fast: %7.1f us   ct: %8.1f us   ratio: %.2fx\n\n",
           fast_gen, ct_gen, ct_gen / fast_gen);

    printf("================================================================\n");
    printf("  Lower ratio = smaller CT overhead (1.0x = same speed)\n");
    printf("================================================================\n");

    return 0;
}
