// Quick benchmark: scalar_mul vs scalar_mul_with_plan for k*P performance
// Measures the BIP-352 bottleneck operation: fixed K * variable Q

#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/field.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <array>

using PT = secp256k1::fast::Point;
using SC = secp256k1::fast::Scalar;

static const int WARMUP = 100;
static const int ITERS  = 5000;

int main() {
    // Fixed scalar K (simulates BIP-352 scan_private_key)
    std::array<uint8_t, 32> k_bytes = {
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC,
        0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B, 0x07,
        0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9,
        0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8, 0x17, 0x98
    };
    SC K = SC::from_bytes(k_bytes.data());

    // Precompute KPlan once (simulates startup cost)
    auto plan = secp256k1::fast::KPlan::from_scalar(K, 4);

    // Generate variable Q points (simulates tweak_points)
    std::array<PT, 256> Qs;
    {
        SC s = SC::one();
        for (int i = 0; i < 256; i++) {
            Qs[static_cast<std::size_t>(i)] = PT::generator().scalar_mul(s);
            s = s + SC::one();
        }
    }

    // Warmup
    volatile uint8_t sink = 0;
    for (int i = 0; i < WARMUP; i++) {
        PT r = Qs[static_cast<std::size_t>(i % 256)].scalar_mul(K);
        auto c = r.to_compressed();
        sink ^= c[0];
    }

    // ---- Benchmark scalar_mul (regular path -> scalar_mul_glv52) ----
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            PT r = Qs[static_cast<std::size_t>(i % 256)].scalar_mul(K);
            auto c = r.to_compressed();
            sink ^= c[0];
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(end - start).count() / ITERS;
        std::printf("scalar_mul(K)            : %8.1f ns/op  (%.1f us)\n", ns, ns / 1000.0);
    }

    // ---- Benchmark scalar_mul_with_plan (KPlan path) ----
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            PT r = Qs[static_cast<std::size_t>(i % 256)].scalar_mul_with_plan(plan);
            auto c = r.to_compressed();
            sink ^= c[0];
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(end - start).count() / ITERS;
        std::printf("scalar_mul_with_plan(K)  : %8.1f ns/op  (%.1f us)\n", ns, ns / 1000.0);
    }

    // ---- Benchmark just K*G (generator multiply, for context) ----
    {
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) {
            PT r = PT::generator().scalar_mul(K);
            auto c = r.to_compressed();
            sink ^= c[0];
        }
        auto end = std::chrono::high_resolution_clock::now();
        double ns = std::chrono::duration<double, std::nano>(end - start).count() / ITERS;
        std::printf("K*G (generator mul)      : %8.1f ns/op  (%.1f us)\n", ns, ns / 1000.0);
    }

    (void)sink;
    return 0;
}
