// ============================================================================
// REGRESSION: comb_gen_mul lock-free thread safety (CRIT-01)
// ============================================================================
// Root cause: comb_gen_mul() and comb_gen_mul_ct() held g_comb_mutex on every
// call — even when the precomputed table was already built and read-only. This
// serialised all signing threads to a single lock, killing throughput under
// any concurrency.
//
// Fix: std::call_once for one-time init; read path is entirely lock-free after
// the first call_once completes.
//
// Tests verify:
//   COMB-LF1: Single-threaded correctness — comb_gen_mul(k) == k*G
//   COMB-LF2: comb_gen_mul_ct(k) == k*G
//   COMB-LF3: comb_gen_mul(k1) + comb_gen_mul(k2) == (k1+k2)*G  (linearity)
//   COMB-LF4: comb_gen_mul(0) is point at infinity
//   COMB-LF5: comb_gen_mul(1) == G
//   COMB-LF6: Concurrent init + mul from multiple threads produces correct results
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>

#include "secp256k1/ecmult_gen_comb.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

static int g_pass = 0, g_fail = 0;
static void check(bool cond, const char* msg) {
    if (cond) { ++g_pass; printf("    [OK] %s\n", msg); }
    else       { ++g_fail; printf("  [FAIL] %s\n", msg); }
    fflush(stdout);
}

static bool points_equal(const Point& A, const Point& B) {
    if (A.is_infinity() && B.is_infinity()) return true;
    if (A.is_infinity() || B.is_infinity()) return false;
    return A.to_compressed() == B.to_compressed();
}

static Scalar scalar_from_seed(uint64_t seed) {
    uint8_t buf[32]{};
    for (int i = 0; i < 8; ++i)
        buf[i] = static_cast<uint8_t>((seed >> (i * 8)) & 0xFF);
    buf[31] = 0x01;
    return Scalar::from_bytes(buf);
}

// ─── comb_lockfree_single_thread_correct ─────────────────────────────────────
static void comb_lockfree_single_thread_correct() {
    for (uint64_t seed : {1ULL, 2ULL, 42ULL, 0xDEADBEEFULL}) {
        Scalar k = scalar_from_seed(seed);
        auto ref = Point::generator().scalar_mul(k);
        auto got = secp256k1::fast::comb_gen_mul(k);
        char msg[80];
        std::snprintf(msg, sizeof(msg), "COMB-LF1: comb_gen_mul(seed=%llu) == k*G",
                      static_cast<unsigned long long>(seed));
        check(points_equal(ref, got), msg);
    }
}

// ─── comb_lockfree_ct_correct ────────────────────────────────────────────────
static void comb_lockfree_ct_correct() {
    for (uint64_t seed : {1ULL, 99ULL, 0xCAFEBABEULL}) {
        Scalar k = scalar_from_seed(seed);
        auto ref = Point::generator().scalar_mul(k);
        auto got = secp256k1::fast::comb_gen_mul_ct(k);
        char msg[80];
        std::snprintf(msg, sizeof(msg), "COMB-LF2: comb_gen_mul_ct(seed=%llu) == k*G",
                      static_cast<unsigned long long>(seed));
        check(points_equal(ref, got), msg);
    }
}

// ─── comb_lockfree_linearity ─────────────────────────────────────────────────
static void comb_lockfree_linearity() {
    Scalar k1 = scalar_from_seed(7);
    Scalar k2 = scalar_from_seed(13);
    Scalar k12 = k1 + k2;

    auto P1  = secp256k1::fast::comb_gen_mul(k1);
    auto P2  = secp256k1::fast::comb_gen_mul(k2);
    auto P12 = secp256k1::fast::comb_gen_mul(k12);
    auto P12_add = P1.add(P2);

    check(points_equal(P12, P12_add),
          "COMB-LF3: comb_gen_mul(k1+k2) == comb_gen_mul(k1)+comb_gen_mul(k2)");
}

// ─── comb_lockfree_zero ──────────────────────────────────────────────────────
static void comb_lockfree_zero() {
    auto got = secp256k1::fast::comb_gen_mul(Scalar::zero());
    check(got.is_infinity(), "COMB-LF4: comb_gen_mul(0) == infinity");
}

// ─── comb_lockfree_one ───────────────────────────────────────────────────────
static void comb_lockfree_one() {
    auto got = secp256k1::fast::comb_gen_mul(Scalar::one());
    check(points_equal(got, Point::generator()),
          "COMB-LF5: comb_gen_mul(1) == G");
}

// ─── comb_lockfree_concurrent ────────────────────────────────────────────────
// Launch T threads; each independently calls comb_gen_mul and checks correctness.
// All start at the same barrier — maximises init+read contention.
static void comb_lockfree_concurrent() {
    constexpr int T = 8;
    constexpr int ITERS = 10;

    std::atomic<int> barrier{0};
    std::vector<int> results(T, 0);

    auto worker = [&](int tid) {
        // Spin until all threads ready
        barrier.fetch_add(1, std::memory_order_release);
        while (barrier.load(std::memory_order_acquire) < T) {
            // busy wait
        }
        int ok = 0;
        for (int i = 0; i < ITERS; ++i) {
            Scalar k = scalar_from_seed(static_cast<uint64_t>(tid * 1000 + i + 1));
            auto ref = Point::generator().scalar_mul(k);
            auto got = secp256k1::fast::comb_gen_mul(k);
            if (points_equal(ref, got)) ++ok;
        }
        results[static_cast<std::size_t>(tid)] = ok;
    };

    std::vector<std::thread> threads;
    threads.reserve(T);
    for (int i = 0; i < T; ++i)
        threads.emplace_back(worker, i);
    for (auto& th : threads)
        th.join();

    int total_ok = 0;
    for (int r : results) total_ok += r;
    check(total_ok == T * ITERS,
          "COMB-LF6: concurrent comb_gen_mul correct on all threads");
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
#else
int test_regression_comb_gen_lockfree_run() {
#endif
    printf("====================================================================\n");
    printf("REGRESSION: comb_gen_mul lock-free thread safety (CRIT-01)\n");
    printf("====================================================================\n\n");
    printf("Verifies that comb_gen_mul/comb_gen_mul_ct produce correct results\n");
    printf("after removing the global mutex from the read path.\n\n");

    comb_lockfree_single_thread_correct();
    comb_lockfree_ct_correct();
    comb_lockfree_linearity();
    comb_lockfree_zero();
    comb_lockfree_one();
    comb_lockfree_concurrent();

    printf("\n--- Result: %d passed, %d failed ---\n", g_pass, g_fail);
    return (g_fail == 0) ? 0 : 1;
}
