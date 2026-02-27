#include <iostream>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>
#include <iomanip>
#include <array>

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/selftest.hpp"

using namespace secp256k1::fast;

static std::vector<Scalar> gen_scalars(size_t count, uint64_t seed) {
    std::vector<Scalar> out; out.reserve(count);
    std::mt19937_64 rng(seed);
    for (size_t i = 0; i < count; ++i) {
        std::array<std::uint8_t, 32> b{};
        for (size_t j = 0; j < 4; ++j) {
            std::uint64_t v = rng();
            std::memcpy(&b[j*8], &v, 8);
        }
        out.emplace_back(Scalar::from_bytes(b));
    }
    return out;
}

// Build odd-multiples table [P, 3P, 5P, ..., (2^w-1)P] as Points
static void build_odd_table(const Point& P, int window_bits, std::vector<Point>& table_out) {
    const int max_digit = (1 << window_bits) - 1;  // e.g., 31 for w=5
    const int table_size = (max_digit + 1) / 2;    // number of odd entries
    table_out.resize(static_cast<std::size_t>(table_size));
    if (P.is_infinity()) {
        for (int i = 0; i < table_size; ++i) table_out[static_cast<size_t>(i)] = Point::infinity();
        return;
    }
    table_out[0] = P; // 1*P
    Point const twoP = P.dbl();
    for (int i = 1; i < table_size; ++i) {
        table_out[static_cast<size_t>(i)] = table_out[static_cast<size_t>(i-1)].add(twoP);
    }
}

// Interleaved Shamir using wNAF digits and Point tables for two bases
static Point shamir_interleaved_wnaf(const std::vector<int32_t>& wnaf1, const std::vector<int32_t>& wnaf2,
                                     const std::vector<Point>& tableP,
                                     const std::vector<Point>& tableQ) {
    size_t const max_len = std::max(wnaf1.size(), wnaf2.size());
    Point R = Point::infinity();

    for (int i = static_cast<int>(max_len) - 1; i >= 0; --i) {
        R = R.dbl();
        if (i < static_cast<int>(wnaf1.size())) {
            int32_t const d1 = wnaf1[static_cast<size_t>(i)];
            if (d1 != 0) {
                bool const neg = (d1 < 0);
                int const idx = ((neg ? -d1 : d1) - 1) / 2;
                Point add = tableP[static_cast<size_t>(idx)];
                if (neg) add.negate_inplace();
                R = R.add(add);
            }
        }
        if (i < static_cast<int>(wnaf2.size())) {
            int32_t const d2 = wnaf2[static_cast<size_t>(i)];
            if (d2 != 0) {
                bool const neg = (d2 < 0);
                int const idx = ((neg ? -d2 : d2) - 1) / 2;
                Point add = tableQ[static_cast<size_t>(idx)];
                if (neg) add.negate_inplace();
                R = R.add(add);
            }
        }
    }
    return R;
}

static double bench_separate_prod_like(const std::vector<Scalar>& k1s, const std::vector<Scalar>& k2s,
                                       const Point& Q, size_t iters) {
    Point acc = Point::generator();
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        Point const a = scalar_mul_generator(k1s[i % k1s.size()]); // fast G path
        Point const b = Q.scalar_mul(k2s[i % k2s.size()]);         // variable-base
        acc = a.add(b);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    volatile auto guard = acc.x_raw().limbs()[0]; (void)guard;
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / static_cast<double>(iters);
}

static double bench_separate_variable(const std::vector<Scalar>& k1s, const std::vector<Scalar>& k2s,
                                      const Point& Q, size_t iters, unsigned wbits) {
    Point const P = Point::generator();
    Point acc = P;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        Point const a = scalar_mul_arbitrary(P, k1s[i % k1s.size()], wbits);
        Point const b = scalar_mul_arbitrary(Q, k2s[i % k2s.size()], wbits);
        acc = a.add(b);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    volatile auto guard = acc.x_raw().limbs()[0]; (void)guard;
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / static_cast<double>(iters);
}

static double bench_shamir_variable(const std::vector<Scalar>& k1s, const std::vector<Scalar>& k2s,
                                    const Point& Q, size_t iters, unsigned wbits) {
    Point const P = Point::generator();

    // Precompute tables once for P and Q (variable-base tables are cheap)
    std::vector<Point> tableP; build_odd_table(P, static_cast<int>(wbits), tableP);
    std::vector<Point> tableQ; build_odd_table(Q, static_cast<int>(wbits), tableQ);

    // Prepare wNAF for a batch to amortize compute
    std::vector<std::vector<int32_t>> w1; w1.reserve(k1s.size());
    std::vector<std::vector<int32_t>> w2; w2.reserve(k2s.size());
    for (auto& s : k1s) w1.emplace_back(compute_wnaf(s, wbits));
    for (auto& s : k2s) w2.emplace_back(compute_wnaf(s, wbits));

    Point acc = P;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < iters; ++i) {
        const auto& wa = w1[i % w1.size()];
        const auto& wb = w2[i % w2.size()];
        Point const r = shamir_interleaved_wnaf(wa, wb, tableP, tableQ);
        acc = acc.add(r);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    volatile auto guard = acc.x_raw().limbs()[0]; (void)guard;
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()) / static_cast<double>(iters);
}

static void print_row(const std::string& name, double ns) {
    std::cout << std::left << std::setw(28) << name << ": "
              << std::right << std::setw(9) << std::fixed << std::setprecision(2)
              << ns << " ns (" << (ns / 1000.0) << " \xC2\xB5s)\n";
}

int main() {
    // Validate arithmetic correctness before benchmarking
    std::cout << "Running arithmetic validation...\n";
    secp256k1::fast::Selftest(true);
    std::cout << "\n";
    
    constexpr size_t N = 64;
    constexpr size_t ITERS = 3000;
    constexpr unsigned WBITS = 5;

    // Random Q
    Scalar const sQ = gen_scalars(1, 0xBADC0FFEEULL)[0];
    Point const Q = Point::generator().scalar_mul(sQ);

    // Random scalars for batches
    auto k1s = gen_scalars(N, 0xA5A5A5A5ULL);
    auto k2s = gen_scalars(N, 0x5A5A5A5AULL);

    std::cout << "\nECDSA-style Multi-Scalar (k1*G + k2*Q)\n";
    std::cout << "======================================\n";

    // Warmup
    (void)bench_separate_prod_like(k1s, k2s, Q, 256);

    double const t_prod = bench_separate_prod_like(k1s, k2s, Q, ITERS);
    print_row("Separate (prod-like)", t_prod);

    double const t_var = bench_separate_variable(k1s, k2s, Q, ITERS, WBITS);
    print_row("Separate (variable)", t_var);

    double const t_shamir = bench_shamir_variable(k1s, k2s, Q, ITERS, WBITS);
    print_row("Shamir interleaved", t_shamir);

    std::cout << "\nNotes:\n";
    std::cout << "- 'prod-like' uses generator precompute for k1*G + variable-base for k2*Q.\n";
    std::cout << "- 'variable' computes both with fixed-window variable-base (fair vs Shamir).\n";
    std::cout << "- 'Shamir interleaved' merges streams to reduce doublings.\n";

    return 0;
}
