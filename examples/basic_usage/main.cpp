// ============================================================================
// UltrafastSecp256k1 -- Desktop Quick-Start Example
// ============================================================================
// Demonstrates basic usage: key generation, point operations, serialization.
// Build: cmake --build <build_dir> --target example_basic_usage
// ============================================================================

#include <chrono>
#include <cstdio>
#include <cstdint>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/selftest.hpp"

using namespace secp256k1::fast;

// -- helpers ------------------------------------------------------------------

static void print_hex(const char* label, const uint8_t* data, size_t len) {
    printf("  %s: ", label);
    for (size_t i = 0; i < len; ++i) printf("%02x", data[i]);
    printf("\n");
}

static double now_us() {
    using clk = std::chrono::high_resolution_clock;
    static auto t0 = clk::now();
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

// -- main ---------------------------------------------------------------------

int main() {
    printf("=== UltrafastSecp256k1 -- Basic Usage Example ===\n\n");

    // 1) Self-test
    printf("[1] Running self-test...\n");
    bool const ok = Selftest(false);
    printf("    Result: %s\n\n", ok ? "PASS" : "FAIL");
    if (!ok) return 1;

    // 2) Create a private key (scalar)
    printf("[2] Key generation\n");
    auto priv = Scalar::from_hex(
        "e8f32e723decf4051aefac8e2c93c9c5b214313817cdb01a1494b917c8436b35");
    auto pub = Point::generator().scalar_mul(priv);

    auto compressed = pub.to_compressed();
    auto uncompressed = pub.to_uncompressed();
    printf("  Private key : %s\n", priv.to_hex().c_str());
    print_hex("Public (comp)", compressed.data(), compressed.size());
    print_hex("Public (uncomp)", uncompressed.data(), uncompressed.size());
    printf("\n");

    // 3) Point arithmetic
    printf("[3] Point arithmetic\n");
    auto G = Point::generator();
    auto G2 = G.dbl();
    auto G3 = G2.add(G);
    printf("  G   x = %s\n", G.x().to_hex().c_str());
    printf("  2G  x = %s\n", G2.x().to_hex().c_str());
    printf("  3G  x = %s\n", G3.x().to_hex().c_str());
    printf("\n");

    // 4) Scalar arithmetic
    printf("[4] Scalar arithmetic\n");
    auto a = Scalar::from_uint64(7);
    auto b = Scalar::from_uint64(13);
    auto c = a * b;
    printf("  7 * 13 mod n = %s\n", c.to_hex().c_str());
    printf("  Expected 91  = ...5b (last byte)\n\n");

    // 5) Micro-benchmark: scalar_mul(k, G) with random 256-bit scalars
    printf("[5] Benchmark: k * G  (1000 iterations, random 256-bit scalars)\n");
    constexpr int ITERS = 1000;

    // Pre-generate random-looking 256-bit scalars (deterministic PRNG for reproducibility)
    // Using golden-ratio stepping from a full 256-bit base -- avoids trivially small scalars.
    Scalar scalars[ITERS];
    {
        auto base = Scalar::from_hex(
            "b5037ebecae0da656179c623f6cb73641db2aa0fabe888ffb78466fa18470379");
        auto step = Scalar::from_hex(
            "9e3779b97f4a7c15f39cc0605cedc8341082276bf3a27251f86c6a11d0c18e95");
        for (int i = 0; i < ITERS; ++i) {
            scalars[i] = base;
            base += step;
        }
    }

    volatile uint8_t sink = 0;

    double const t0 = now_us();
    for (int i = 0; i < ITERS; ++i) {
        auto P = G.scalar_mul(scalars[i]);
        sink ^= P.to_compressed()[0];
    }
    double const elapsed = now_us() - t0;
    printf("  Total : %.1f ms\n", elapsed / 1000.0);
    printf("  Per op: %.1f us\n", elapsed / ITERS);
    printf("  (sink = 0x%02x)\n\n", (unsigned)sink);

    printf("=== Done ===\n");
    return 0;
}
