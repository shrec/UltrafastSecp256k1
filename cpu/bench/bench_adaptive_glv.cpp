// Benchmark adaptive GLV threshold across window sizes
// Measures average nanoseconds per generator multiplication with and without GLV.
// For GLV cases we force enable_glv irrespective of adaptive threshold to get raw numbers.

#include "secp256k1/precompute.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/selftest.hpp"
#include <chrono>
#include <iostream>
#include <vector>
#include <iomanip>
#include <array>

using namespace secp256k1::fast;

static double run_bench(unsigned window_bits, bool glv, unsigned iters) {
    FixedBaseConfig cfg{};
    cfg.window_bits = window_bits;
    cfg.enable_glv = glv;
    cfg.adaptive_glv = false; // force raw behavior for measurement
    cfg.use_jsf = false;      // measure windowed Shamir path for GLV (JSF only valid for limited windows)
    cfg.use_cache = false;    // disable disk cache: measure pure algorithm cost
    configure_fixed_base(cfg);
    ensure_fixed_base_ready();

    // Pre-generate deterministic scalars
    std::vector<Scalar> scalars(iters);
    for (unsigned i = 0; i < iters; ++i) {
        std::array<std::uint8_t, 32> b{};
        for (unsigned j = 0; j < 32; ++j) b[j] = static_cast<std::uint8_t>((i * 1315423911u + j * 2654435761u) >> (j % 13));
        scalars[i] = Scalar::from_bytes(b);
    }

    // Warmup (avoid first-call penalties)
    for (unsigned i = 0; i < std::min(50u, iters); ++i) {
        volatile Point p = scalar_mul_generator(scalars[i]);
        (void)p;
    }

    auto start = std::chrono::high_resolution_clock::now();
    for (unsigned i = 0; i < iters; ++i) {
        volatile Point p = scalar_mul_generator(scalars[i]);
        (void)p;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    return ns / static_cast<double>(iters);
}

int main(int argc, char** argv) {
    // Validate arithmetic correctness before benchmarking
    std::cout << "Running arithmetic validation...\n";
    secp256k1::fast::Selftest(true);
    std::cout << "\n";
    
    unsigned min_w = 8, max_w = 20, iters = 1500;
    if (argc > 1) min_w = static_cast<unsigned>(std::stoi(argv[1]));
    if (argc > 2) max_w = static_cast<unsigned>(std::stoi(argv[2]));
    if (argc > 3) iters = static_cast<unsigned>(std::stoi(argv[3]));

    std::cout << "window_bits,ns_no_glv,ns_glv(jsf),glv_gain_percent" << std::endl;
    for (unsigned w = min_w; w <= max_w; ++w) {
        try {
            double no_glv = run_bench(w, false, iters);
            double glv_jsf = run_bench(w, true, iters);
            double gain = (no_glv - glv_jsf) / no_glv * 100.0; // negative if slower
            std::cout << w << ',' << std::fixed << std::setprecision(2)
                      << no_glv << ',' << glv_jsf << ',' << gain << std::endl;
        } catch (const std::exception& ex) {
            std::cout << w << ",error," << ex.what() << "," << 0.0 << std::endl;
        }
    }
    return 0;
}
