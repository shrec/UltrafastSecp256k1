#include <iostream>
#include <chrono>
#include <cstring>
#include <random>
#include <vector>
#include <iomanip>

#include "secp256k1/precompute.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/selftest.hpp"

using namespace secp256k1::fast;

static std::vector<Scalar> gen_scalars(size_t count) {
    std::vector<Scalar> out; out.reserve(count);
    std::mt19937_64 rng(0xC0FFEEULL);
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

static double time_scalar_mul_generator(const std::vector<Scalar>& scalars, size_t iters) {
    auto start = std::chrono::high_resolution_clock::now();
    Point sink = Point::generator();
    for (size_t i = 0; i < iters; ++i) {
        sink = scalar_mul_generator(scalars[i % scalars.size()]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    volatile auto guard = sink.x_raw().limbs()[0]; (void)guard;
    return static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / static_cast<double>(iters);
}

static void print_row(const char* name, double ns) {
    std::cout << std::left << std::setw(24) << name << ": "
              << std::right << std::setw(8) << std::fixed << std::setprecision(2)
              << ns << " ns (" << (ns / 1000.0) << " \xC2\xB5s)\n";
}

int main() {
    // Validate arithmetic correctness before benchmarking
    std::cout << "Running arithmetic validation...\n";
    secp256k1::fast::Selftest(true);
    std::cout << "\n";
    
    constexpr size_t N_SCALARS = 64;
    constexpr size_t WARMUP_ITERS = 256;
    constexpr size_t ITERS = 4000;

    auto scalars = gen_scalars(N_SCALARS);

    // Base config shared between runs
    FixedBaseConfig base{};
    base.window_bits = 18;     // same defaults as library
    base.enable_glv = true;    // GLV on to exercise shamir path
    base.use_cache = true;

    // Warmup (windowed Shamir)
    {
        auto cfg = base; cfg.use_jsf = false; configure_fixed_base(cfg); ensure_fixed_base_ready();
        (void)time_scalar_mul_generator(scalars, WARMUP_ITERS);
    }

    std::cout << "\nJSF vs Windowed Shamir (GLV path)\n";
    std::cout << "===================================\n";

    // Windowed Shamir
    auto cfg1 = base; cfg1.use_jsf = false; configure_fixed_base(cfg1); ensure_fixed_base_ready();
    double const shamir_ns = time_scalar_mul_generator(scalars, ITERS);
    print_row("Windowed Shamir", shamir_ns);

    // JSF
    auto cfg2 = base; cfg2.use_jsf = true; configure_fixed_base(cfg2); ensure_fixed_base_ready();
    double const jsf_ns = time_scalar_mul_generator(scalars, ITERS);
    print_row("JSF", jsf_ns);

    double const delta = (shamir_ns - jsf_ns) / shamir_ns * 100.0;
    std::cout << "\nDelta: " << std::fixed << std::setprecision(2) << delta << "% (positive = JSF faster)\n";

    return 0;
}
