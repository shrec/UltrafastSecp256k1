// ============================================================================
// Test: Affine Batch Addition
// ============================================================================
// Validates batch_add_affine against scalar point arithmetic.
// Ensures precomputed G-tables and batch addition produce correct results.

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>

#include "secp256k1/batch_add_affine.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/field.hpp"

using namespace secp256k1::fast;

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  FAIL: %s\n", name);
    }
}

// -- Test 1: Precompute G-multiples table -------------------------------------

static void test_precompute_g_multiples() {
    std::printf("[BatchAffine] Precompute G-multiples table...\n");

    constexpr std::size_t N = 64;
    auto table = precompute_g_multiples(N);

    check(table.size() == N, "table size == N");

    // Verify: table[0] = 1*G, table[1] = 2*G, ..., table[N-1] = N*G
    Point current = Point::generator();
    for (std::size_t i = 0; i < N; ++i) {
        // Convert Point to affine X, Y for comparison
        FieldElement const expected_x = current.x();
        FieldElement const expected_y = current.y();

        char label_x[64], label_y[64];
        std::snprintf(label_x, sizeof(label_x), "table[%zu].x == %zuG.x", i, i + 1);
        std::snprintf(label_y, sizeof(label_y), "table[%zu].y == %zuG.y", i, i + 1);

        check(table[i].x == expected_x, label_x);
        check(table[i].y == expected_y, label_y);

        current.next_inplace();
    }

    std::printf("  Verified %zu G-multiples\n", N);
}

// -- Test 2: batch_add_affine_x correctness ----------------------------------

static void test_batch_add_x_correctness() {
    std::printf("[BatchAffine] batch_add_affine_x correctness...\n");

    constexpr std::size_t BATCH = 128;
    auto g_table = precompute_g_multiples(BATCH);

    // Base point: P = 42*G
    Scalar const s42 = Scalar::from_uint64(42);
    Point const P = scalar_mul_generator(s42);
    FieldElement const base_x = P.x();
    FieldElement const base_y = P.y();

    // Compute batch: result[i] = P + (i+1)*G = (42 + i + 1)*G
    std::vector<FieldElement> out_x(BATCH);
    std::vector<FieldElement> scratch;
    batch_add_affine_x(base_x, base_y, g_table.data(), out_x.data(), BATCH, scratch);

    // Verify against scalar multiplication
    FieldElement const fe_zero = FieldElement::zero();
    for (std::size_t i = 0; i < BATCH; ++i) {
        // When base == table[i] (i.e. i+1 == 42), dx=0 -> sentinel zero output.
        // This is the degenerate doubling case, handled correctly by sentinel.
        if (i + 1 == 42) {
            char label[80];
            std::snprintf(label, sizeof(label), "P + %zuG: degenerate (dx=0) -> sentinel zero", i + 1);
            check(out_x[i] == fe_zero, label);
            continue;
        }

        Scalar const s = Scalar::from_uint64(42 + i + 1);
        Point const expected = scalar_mul_generator(s);
        FieldElement const expected_x = expected.x();

        char label[64];
        std::snprintf(label, sizeof(label), "P + %zuG == %zuG (x-coord)", i + 1, 42 + i + 1);
        check(out_x[i] == expected_x, label);
    }

    std::printf("  Verified %zu batch additions\n", BATCH);
}

// -- Test 3: batch_add_affine_xy correctness ---------------------------------

static void test_batch_add_xy_correctness() {
    std::printf("[BatchAffine] batch_add_affine_xy correctness...\n");

    constexpr std::size_t BATCH = 64;
    auto g_table = precompute_g_multiples(BATCH);

    // Base: P = 1000*G
    Scalar const s1000 = Scalar::from_uint64(1000);
    Point const P = scalar_mul_generator(s1000);
    FieldElement const base_x = P.x();
    FieldElement const base_y = P.y();

    std::vector<FieldElement> out_x(BATCH), out_y(BATCH);
    std::vector<FieldElement> scratch;
    batch_add_affine_xy(base_x, base_y, g_table.data(),
                        out_x.data(), out_y.data(), BATCH, scratch);

    for (std::size_t i = 0; i < BATCH; ++i) {
        Scalar const s = Scalar::from_uint64(1000 + i + 1);
        Point const expected = scalar_mul_generator(s);

        char label_x[64], label_y[64];
        std::snprintf(label_x, sizeof(label_x), "xy[%zu].x correct", i);
        std::snprintf(label_y, sizeof(label_y), "xy[%zu].y correct", i);
        check(out_x[i] == expected.x(), label_x);
        check(out_y[i] == expected.y(), label_y);
    }

    std::printf("  Verified %zu XY results\n", BATCH);
}

// -- Test 4: Bidirectional batch add ------------------------------------------

static void test_bidirectional() {
    std::printf("[BatchAffine] Bidirectional batch add...\n");

    constexpr std::size_t BATCH = 32;
    auto g_table = precompute_g_multiples(BATCH);
    auto g_table_neg = negate_affine_table(g_table.data(), BATCH);

    // Base: P = 500*G
    Scalar const s500 = Scalar::from_uint64(500);
    Point const P = scalar_mul_generator(s500);
    FieldElement const base_x = P.x();
    FieldElement const base_y = P.y();

    std::vector<FieldElement> out_fwd(BATCH), out_bwd(BATCH);
    std::vector<FieldElement> scratch;
    batch_add_affine_x_bidirectional(
        base_x, base_y,
        g_table.data(), g_table_neg.data(),
        out_fwd.data(), out_bwd.data(), BATCH, scratch);

    for (std::size_t i = 0; i < BATCH; ++i) {
        // Forward: P + (i+1)*G = (500 + i + 1)*G
        Scalar const s_fwd = Scalar::from_uint64(500 + i + 1);
        Point const exp_fwd = scalar_mul_generator(s_fwd);

        // Backward: P - (i+1)*G = (500 - i - 1)*G
        Scalar const s_bwd = Scalar::from_uint64(500 - i - 1);
        Point const exp_bwd = scalar_mul_generator(s_bwd);

        char label_f[64], label_b[64];
        std::snprintf(label_f, sizeof(label_f), "fwd[%zu] == %zuG", i, 500 + i + 1);
        std::snprintf(label_b, sizeof(label_b), "bwd[%zu] == %zuG", i, 500 - i - 1);
        check(out_fwd[i] == exp_fwd.x(), label_f);
        check(out_bwd[i] == exp_bwd.x(), label_b);
    }

    std::printf("  Verified %zu bidirectional pairs\n", BATCH);
}

// -- Test 5: Parity extraction ------------------------------------------------

static void test_parity() {
    std::printf("[BatchAffine] Y-parity extraction...\n");

    constexpr std::size_t BATCH = 32;
    auto g_table = precompute_g_multiples(BATCH);

    Scalar const s100 = Scalar::from_uint64(100);
    Point const P = scalar_mul_generator(s100);
    FieldElement const base_x = P.x();
    FieldElement const base_y = P.y();

    std::vector<FieldElement> out_x(BATCH);
    std::vector<uint8_t> out_parity(BATCH);
    std::vector<FieldElement> scratch;
    batch_add_affine_x_with_parity(
        base_x, base_y, g_table.data(),
        out_x.data(), out_parity.data(), BATCH, scratch);

    for (std::size_t i = 0; i < BATCH; ++i) {
        Scalar const s = Scalar::from_uint64(100 + i + 1);
        Point const expected = scalar_mul_generator(s);
        auto y_bytes = expected.y().to_bytes();
        uint8_t const expected_parity = y_bytes[31] & 1;

        char label[64];
        std::snprintf(label, sizeof(label), "parity[%zu] correct", i);
        check(out_parity[i] == expected_parity, label);
    }

    std::printf("  Verified %zu parity values\n", BATCH);
}

// -- Test 6: Arbitrary point multiples ----------------------------------------

static void test_arbitrary_point_table() {
    std::printf("[BatchAffine] Arbitrary point multiples table...\n");

    // Use 7*G as base
    Scalar const s7 = Scalar::from_uint64(7);
    Point const Q = scalar_mul_generator(s7);
    FieldElement const qx = Q.x();
    FieldElement const qy = Q.y();

    constexpr std::size_t N = 16;
    auto table = precompute_point_multiples(qx, qy, N);

    // table[i] = (i+1) * Q = (i+1) * 7G = ((i+1)*7)*G
    for (std::size_t i = 0; i < N; ++i) {
        Scalar const s = Scalar::from_uint64((i + 1) * 7);
        Point const expected = scalar_mul_generator(s);

        char label[64];
        std::snprintf(label, sizeof(label), "arb_table[%zu] == %zuQ", i, i + 1);
        check(table[i].x == expected.x(), label);
        check(table[i].y == expected.y(), label);
    }

    std::printf("  Verified %zu arbitrary multiples\n", N);
}

// -- Test 7: Large batch (search-scale) ---------------------------------------

static void test_large_batch() {
    std::printf("[BatchAffine] Large batch (1024 points)...\n");

    constexpr std::size_t BATCH = 1024;
    
    auto t0 = std::chrono::high_resolution_clock::now();
    auto g_table = precompute_g_multiples(BATCH);
    auto t1 = std::chrono::high_resolution_clock::now();
    double const precomp_us = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    std::printf("  Precompute %zu G-multiples: %.1f us\n", BATCH, precomp_us);

    // Base: P = 999999*G
    Scalar const sbase = Scalar::from_uint64(999999);
    Point const P = scalar_mul_generator(sbase);
    FieldElement const base_x = P.x();
    FieldElement const base_y = P.y();

    std::vector<FieldElement> out_x(BATCH);
    std::vector<FieldElement> scratch;

    // Warmup
    batch_add_affine_x(base_x, base_y, g_table.data(), out_x.data(), BATCH, scratch);

    // Benchmark
    constexpr int ITERS = 1000;
    t0 = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < ITERS; ++iter) {
        batch_add_affine_x(base_x, base_y, g_table.data(), out_x.data(), BATCH, scratch);
    }
    t1 = std::chrono::high_resolution_clock::now();
    double const total_ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
    double const per_batch_us = total_ns / ITERS / 1000.0;
    double const per_point_ns = total_ns / ITERS / BATCH;

    std::printf("  Batch %zu: %.1f us total, %.1f ns/point\n", BATCH, per_batch_us, per_point_ns);
    std::printf("  Throughput: %.2f Mpoints/s (single thread)\n", 1e9 / per_point_ns / 1e6);

    // Spot-check first and last
    {
        Scalar const s = Scalar::from_uint64(999999 + 1);
        Point const expected = scalar_mul_generator(s);
        check(out_x[0] == expected.x(), "large_batch[0] correct");
    }
    {
        Scalar const s = Scalar::from_uint64(999999 + BATCH);
        Point const expected = scalar_mul_generator(s);
        check(out_x[BATCH - 1] == expected.x(), "large_batch[last] correct");
    }
}

// -- Test 8: Edge case -- empty batch ------------------------------------------

static void test_empty() {
    std::printf("[BatchAffine] Empty batch...\n");
    std::vector<FieldElement> scratch;
    FieldElement const base_x = FieldElement::from_uint64(1);
    FieldElement const base_y = FieldElement::from_uint64(2);
    batch_add_affine_x(base_x, base_y, nullptr, nullptr, 0, scratch);
    check(true, "empty batch: no crash");
}

// -- Test 9: Negate table correctness -----------------------------------------

static void test_negate_table() {
    std::printf("[BatchAffine] Negate table...\n");

    constexpr std::size_t N = 16;
    auto table = precompute_g_multiples(N);
    auto neg = negate_affine_table(table.data(), N);

    for (std::size_t i = 0; i < N; ++i) {
        // neg[i] = -(i+1)*G -> x same, y = -y
        Scalar const s = Scalar::from_uint64(i + 1);
        Point const expected = scalar_mul_generator(s).negate();

        char label[64];
        std::snprintf(label, sizeof(label), "neg_table[%zu] == -%zuG", i, i + 1);
        check(neg[i].x == expected.x(), label);
        check(neg[i].y == expected.y(), label);
    }

    std::printf("  Verified %zu negated points\n", N);
}

// -- Entry point --------------------------------------------------------------

int test_batch_add_affine_run() {
    std::printf("\n=== Affine Batch Addition Tests ===\n");

    test_empty();
    test_precompute_g_multiples();
    test_batch_add_x_correctness();
    test_batch_add_xy_correctness();
    test_bidirectional();
    test_parity();
    test_arbitrary_point_table();
    test_negate_table();
    test_large_batch();

    std::printf("\n  Affine batch add: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() {
    return test_batch_add_affine_run();
}
#endif
