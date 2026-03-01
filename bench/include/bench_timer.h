// bench_timer.h -- High-resolution timing for benchmark measurement
#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#  ifdef _MSC_VER
#    include <intrin.h>
#  else
#    include <x86intrin.h>
#  endif
#  define BENCH_HAS_RDTSC 1
#else
#  define BENCH_HAS_RDTSC 0
#endif

namespace bench {

// ---------------------------------------------------------------------------
// TSC / chrono timer
// ---------------------------------------------------------------------------
inline uint64_t rdtsc_serialize() {
#if BENCH_HAS_RDTSC
#  ifdef _MSC_VER
    unsigned int aux;
    return __rdtscp(&aux);
#  else
    unsigned int lo, hi;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi) :: "rcx");
    return (static_cast<uint64_t>(hi) << 32) | lo;
#  endif
#else
    return 0;
#endif
}

using SteadyClock = std::chrono::steady_clock;
using TimePoint   = SteadyClock::time_point;

inline TimePoint now() { return SteadyClock::now(); }

inline double elapsed_ms(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

inline double elapsed_ns(TimePoint start, TimePoint end) {
    return std::chrono::duration<double, std::nano>(end - start).count();
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------
struct BenchStats {
    double median_ns  = 0.0;
    double p10_ns     = 0.0;
    double p90_ns     = 0.0;
    double ops_per_sec = 0.0;
    size_t total_ops  = 0;
    double total_ms   = 0.0;
};

inline BenchStats compute_stats(std::vector<double>& samples_ns) {
    BenchStats s;
    if (samples_ns.empty()) return s;

    std::sort(samples_ns.begin(), samples_ns.end());
    size_t n = samples_ns.size();

    s.total_ops = n;
    s.total_ms  = std::accumulate(samples_ns.begin(), samples_ns.end(), 0.0) / 1e6;
    s.median_ns = samples_ns[n / 2];
    s.p10_ns    = samples_ns[n / 10];
    s.p90_ns    = samples_ns[n * 9 / 10];
    s.ops_per_sec = (s.median_ns > 0.0) ? 1e9 / s.median_ns : 0.0;

    return s;
}

// ---------------------------------------------------------------------------
// Estimate TSC frequency (MHz)
// ---------------------------------------------------------------------------
inline double estimate_tsc_mhz() {
#if BENCH_HAS_RDTSC
    auto t0 = now();
    uint64_t c0 = rdtsc_serialize();
    // spin ~50ms
    while (elapsed_ms(t0, now()) < 50.0) {}
    uint64_t c1 = rdtsc_serialize();
    auto t1 = now();
    double ms = elapsed_ms(t0, t1);
    if (ms > 0.0) return static_cast<double>(c1 - c0) / (ms * 1000.0);
#endif
    return 0.0;
}

} // namespace bench
