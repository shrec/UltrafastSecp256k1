#include <secp256k1/precompute.hpp>
#include <secp256k1/glv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cstdint>

#if __has_include("../../../rdtsc.h")
#include "../../../rdtsc.h"
#else
int main() {
    std::cerr << "bench_glv_decomp_profile: missing rdtsc.h; skipping.\n";
    return 0;
}
#endif

#if __has_include("../../../rdtsc.h")

using namespace secp256k1::fast;

// Declare profiling counters from library (defined in precompute.cpp when SECP256K1_PROFILE_DECOMP enabled)
namespace secp256k1 { namespace fast {
extern uint64_t g_decomp_scalar_to_limbs_cycles;
extern uint64_t g_decomp_mul_shift_cycles;
extern uint64_t g_decomp_scalar_math_cycles;
extern uint64_t g_decomp_barrett_reduce_cycles;
extern uint64_t g_decomp_normalize_cycles;
} }

// Global cycle counters provided by library (declared in namespace secp256k1::fast)
// Accessible via using directive below.

// Force a volatile sink to prevent dead-code elimination
static volatile uint64_t sink_cycles = 0;

int main(int argc, char** argv) {
    size_t iters = 5000;
    if (argc > 1) {
        iters = static_cast<size_t>(std::stoull(argv[1]));
    }
    std::cout << "GLV Decomposition Profiling (" << iters << " iterations)\n";
    std::cout << "==========================================================\n";
    std::cout << "(Cycles measured with RDTSC; macro SECP256K1_PROFILE_DECOMP=1)\n\n";

    // Reset counters
    g_decomp_scalar_to_limbs_cycles = 0;
    g_decomp_mul_shift_cycles = 0;
    g_decomp_scalar_math_cycles = 0;
    g_decomp_barrett_reduce_cycles = 0;
    g_decomp_normalize_cycles = 0;

    // Deterministic pseudo-random scalars for consistency
    Scalar base = Scalar::from_bytes({
        0x12,0x34,0x56,0x78,0x9A,0xBC,0xDE,0xF0,
        0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x88,
        0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,0x00,
        0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10
    });

    uint64_t t_start = RDTSC();
    for (size_t i = 0; i < iters; ++i) {
        // Slight deterministic variation each iteration
        Scalar s = base + Scalar::from_uint64(i * 0x9E3779B97F4A7C15ULL);
        auto decomp = split_scalar_glv(s);
        // Prevent optimizer removing work
        sink_cycles ^= decomp.k1.limbs()[0];
    }
    uint64_t t_end = RDTSC();
    uint64_t total_cycles = t_end - t_start;

    auto pct = [](uint64_t part, uint64_t total) -> double {
        return total ? (100.0 * double(part) / double(total)) : 0.0;
    };

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Total cycles (wall loop): " << total_cycles << " (" << (double)total_cycles/iters << " per decomp)\n\n";
    uint64_t sum_profiled = g_decomp_scalar_to_limbs_cycles + g_decomp_mul_shift_cycles + g_decomp_scalar_math_cycles + g_decomp_barrett_reduce_cycles + g_decomp_normalize_cycles;
    std::cout << "Profiled component cycles (aggregated): " << sum_profiled << "\n";
    std::cout << "  scalar_to_limbs:      " << g_decomp_scalar_to_limbs_cycles << "  (" << pct(g_decomp_scalar_to_limbs_cycles, sum_profiled) << "%)\n";
    std::cout << "  mul_shift (g1/g2):     " << g_decomp_mul_shift_cycles << "  (" << pct(g_decomp_mul_shift_cycles, sum_profiled) << "%)\n";
    std::cout << "  scalar_math (k2 mix):  " << g_decomp_scalar_math_cycles << "  (" << pct(g_decomp_scalar_math_cycles, sum_profiled) << "%)\n";
    std::cout << "  barrett_reduce:        " << g_decomp_barrett_reduce_cycles << "  (" << pct(g_decomp_barrett_reduce_cycles, sum_profiled) << "%)\n";
    std::cout << "  normalize:             " << g_decomp_normalize_cycles << "  (" << pct(g_decomp_normalize_cycles, sum_profiled) << "%)\n\n";

    std::cout << "Average per component (cycles/decomp):\n";
    std::cout << "  scalar_to_limbs:      " << (double)g_decomp_scalar_to_limbs_cycles/iters << "\n";
    std::cout << "  mul_shift (g1/g2):     " << (double)g_decomp_mul_shift_cycles/iters << "\n";
    std::cout << "  scalar_math (k2 mix):  " << (double)g_decomp_scalar_math_cycles/iters << "\n";
    std::cout << "  barrett_reduce:        " << (double)g_decomp_barrett_reduce_cycles/iters << "\n";
    std::cout << "  normalize:             " << (double)g_decomp_normalize_cycles/iters << "\n";

    std::cout << "Done." << '\n';
    return int(sink_cycles & 0xFF);
}

#endif
