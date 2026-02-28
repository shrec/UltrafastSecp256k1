// ============================================================================
// benchmark_harness.hpp -- Production-Grade Benchmark Infrastructure
// ============================================================================
//
// Standards:
//   * RDTSC cycle counter on x86/x64 (sub-ns precision)
//   * std::chrono::high_resolution_clock fallback (RISC-V, ARM, etc.)
//   * Configurable warm-up iterations
//   * Multi-pass measurement (default: 11)
//   * Median filtering
//   * IQR-based outlier removal
//   * Min/Avg/Median/StdDev tracking
//   * DoNotOptimize / ClobberMemory compiler barriers
//   * Optional thread pinning + priority elevation (Windows)
//
// Usage:
//   #include "secp256k1/benchmark_harness.hpp"
//
//   bench::Harness h;                          // default: 500 warmup, 11 passes
//   double ns = h.run(1000, [&]() { ... });    // 1000 iters per pass
//   h.run_and_print("Field Mul", 100000, [&]() { fe_a * fe_b; });
//
// ============================================================================

#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <vector>

#if defined(_WIN32)
#  define NOMINMAX
#  include <windows.h>
#endif

namespace bench {

// -- DoNotOptimize / ClobberMemory --------------------------------------------
// Prevents the compiler from optimizing away benchmark payloads.

#if defined(__GNUC__) || defined(__clang__)

template <typename T>
inline __attribute__((always_inline)) void DoNotOptimize(T const& value) {
    asm volatile("" : : "r,m"(value) : "memory");
}

template <typename T>
inline __attribute__((always_inline)) void DoNotOptimize(T& value) {
    asm volatile("" : "+r,m"(value) : : "memory");
}

inline __attribute__((always_inline)) void ClobberMemory() {
    asm volatile("" : : : "memory");
}

#elif defined(_MSC_VER)

template <typename T>
__forceinline void DoNotOptimize(T const& value) {
    // Use _ReadWriteBarrier + volatile to prevent optimization
    volatile auto sink = *reinterpret_cast<const volatile char*>(&value);
    (void)sink;
    _ReadWriteBarrier();
}

template <typename T>
__forceinline void DoNotOptimize(T& value) {
    volatile auto sink = *reinterpret_cast<volatile char*>(&value);
    (void)sink;
    _ReadWriteBarrier();
}

__forceinline void ClobberMemory() {
    _ReadWriteBarrier();
}

#else

template <typename T>
inline void DoNotOptimize(T const& value) {
    volatile auto sink = reinterpret_cast<uintptr_t>(&value);
    (void)sink;
}

template <typename T>
inline void DoNotOptimize(T& value) {
    volatile auto sink = reinterpret_cast<uintptr_t>(&value);
    (void)sink;
}

inline void ClobberMemory() {}

#endif

// -- High-Resolution Timer ----------------------------------------------------

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86))
#  define BENCH_HAS_RDTSC 1
#else
#  define BENCH_HAS_RDTSC 0
#endif

#if BENCH_HAS_RDTSC
#  if defined(_MSC_VER)
#    include <intrin.h>
#  endif
#endif

struct Timer {
    // Returns a monotonic tick count.
    // On x86: RDTSC cycles (serialized with RDTSCP or CPUID+RDTSC).
    // Elsewhere: nanoseconds from chrono.
    static inline uint64_t now() noexcept {
#if BENCH_HAS_RDTSC
#  if defined(_MSC_VER)
        unsigned int aux = 0;
        return __rdtscp(&aux);
#  else
        uint32_t lo = 0;
        uint32_t hi = 0;
        // RDTSCP serializes instruction stream and reads TSC
        asm volatile("rdtscp" : "=a"(lo), "=d"(hi) : : "%ecx");
        return (static_cast<uint64_t>(hi) << 32) | lo;
#  endif
#else
        return static_cast<uint64_t>(
            std::chrono::high_resolution_clock::now().time_since_epoch().count());
#endif
    }

    // Ticks -> nanoseconds conversion.
    // On x86 with RDTSC: calibrates TSC frequency at first call.
    // On non-x86: ticks ARE nanoseconds (chrono).
    static double ticks_to_ns(uint64_t ticks) noexcept {
#if BENCH_HAS_RDTSC
        static const double ns_per_tick = calibrate_tsc();
        return static_cast<double>(ticks) * ns_per_tick;
#else
        // chrono ticks -- convert to nanoseconds using the clock's period
        using Period = std::chrono::high_resolution_clock::period;
        constexpr double ratio = static_cast<double>(Period::num) / static_cast<double>(Period::den) * 1e9;
        return static_cast<double>(ticks) * ratio;
#endif
    }

    // Returns the timer type name for display
    static const char* timer_name() noexcept {
#if BENCH_HAS_RDTSC
        return "RDTSCP";
#else
        return "chrono::high_resolution_clock";
#endif
    }

private:
#if BENCH_HAS_RDTSC
    // Calibrate TSC: measure ~10ms with chrono to get ns/tick ratio
    static double calibrate_tsc() noexcept {
        using Clock = std::chrono::high_resolution_clock;
        constexpr int CALIBRATE_MS = 10;

        // Warm up
        volatile uint64_t const warm = now();
        (void)warm;

        auto chrono_start = Clock::now();
        uint64_t const tsc_start = now();

        // Spin for ~10ms
        auto target = chrono_start + std::chrono::milliseconds(CALIBRATE_MS);
        while (Clock::now() < target) {
            // busy-wait
        }

        uint64_t const tsc_end = now();
        auto chrono_end = Clock::now();

        double const ns_elapsed = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(chrono_end - chrono_start).count());
        auto const tsc_elapsed = static_cast<double>(tsc_end - tsc_start);

        return (tsc_elapsed > 0.0) ? (ns_elapsed / tsc_elapsed) : 1.0;
    }
#endif
};

// -- Statistics ---------------------------------------------------------------

struct Stats {
    double min_ns    = 0.0;
    double max_ns    = 0.0;
    double median_ns = 0.0;
    double mean_ns   = 0.0;
    double stddev_ns = 0.0;
    int    samples   = 0;   // after outlier removal
    int    outliers  = 0;   // removed samples
};

// Compute stats with IQR outlier removal
inline Stats compute_stats(std::vector<double>& data) {
    Stats s{};
    if (data.empty()) return s;

    std::sort(data.begin(), data.end());

    const std::size_t n = data.size();

    if (n < 4) {
        // Too few samples for IQR -- use all
        s.min_ns = data.front();
        s.max_ns = data.back();
        s.median_ns = data[n / 2];
        double sum = 0.0;
        for (auto v : data) sum += v;
        s.mean_ns = sum / static_cast<double>(n);
        s.samples = static_cast<int>(n);
        s.outliers = 0;
        return s;
    }

    // IQR-based outlier removal
    double const q1 = data[n / 4];
    double const q3 = data[(3 * n) / 4];
    double const iqr = q3 - q1;
    double const lower = q1 - 1.5 * iqr;
    double const upper = q3 + 1.5 * iqr;

    std::vector<double> filtered;
    filtered.reserve(n);
    for (auto v : data) {
        if (v >= lower && v <= upper) {
            filtered.push_back(v);
        }
    }

    if (filtered.empty()) {
        // All outliers? Fall back to raw data
        filtered = data;
    }

    const std::size_t fn = filtered.size();
    s.outliers = static_cast<int>(n - fn);
    s.samples = static_cast<int>(fn);
    s.min_ns = filtered.front();
    s.max_ns = filtered.back();
    s.median_ns = filtered[fn / 2];

    double sum = 0.0;
    for (auto v : filtered) sum += v;
    s.mean_ns = sum / static_cast<double>(fn);

    double var = 0.0;
    for (auto v : filtered) {
        double const d = v - s.mean_ns;
        var += d * d;
    }
    s.stddev_ns = std::sqrt(var / static_cast<double>(fn));

    return s;
}

// -- Platform Setup -----------------------------------------------------------

inline void pin_thread_and_elevate() {
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), 1ULL);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#elif defined(__linux__) && !defined(__ANDROID__)
    // Pin to CPU 0 on Linux
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(0, &cpuset);
    sched_setaffinity(0, sizeof(cpuset), &cpuset);
#endif
}

// -- Main Harness -------------------------------------------------------------

class Harness {
public:
    int warmup_iters = 500;          // warmup iterations per function call
    std::size_t passes = 11;          // measurement passes (odd number for clean median)

    Harness() = default;
    Harness(int warmup, std::size_t p) : warmup_iters(warmup), passes(p) {}

    // Run benchmark: returns median ns/iter after IQR outlier removal.
    // Func is called `iters` times per pass.
    template <typename Func>
    double run(int iters, Func&& func) const {
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            func();
            ClobberMemory();
        }

        // Measurement passes
        std::vector<double> ns_per_iter;
        ns_per_iter.reserve(passes);

        for (std::size_t p = 0; p < passes; ++p) {
            uint64_t const t0 = Timer::now();
            for (int i = 0; i < iters; ++i) {
                func();
            }
            ClobberMemory();
            uint64_t const t1 = Timer::now();

            double const total_ns = Timer::ticks_to_ns(t1 - t0);
            ns_per_iter.push_back(total_ns / iters);
        }

        Stats const st = compute_stats(ns_per_iter);
        return st.median_ns;
    }

    // Run benchmark and return full statistics.
    template <typename Func>
    Stats run_stats(int iters, Func&& func) const {
        // Warmup
        for (int i = 0; i < warmup_iters; ++i) {
            func();
            ClobberMemory();
        }

        // Measurement passes
        std::vector<double> ns_per_iter;
        ns_per_iter.reserve(passes);

        for (std::size_t p = 0; p < passes; ++p) {
            uint64_t const t0 = Timer::now();
            for (int i = 0; i < iters; ++i) {
                func();
            }
            ClobberMemory();
            uint64_t const t1 = Timer::now();

            double const total_ns = Timer::ticks_to_ns(t1 - t0);
            ns_per_iter.push_back(total_ns / iters);
        }

        return compute_stats(ns_per_iter);
    }

    // Convenience: run + print one-liner result
    template <typename Func>
    double run_and_print(const char* name, int iters, Func&& func) const {
        Stats st = run_stats(iters, func);
        (void)std::printf("  %-28s %9.2f ns  (min=%6.2f  median=%6.2f  stddev=%5.2f  n=%d-%d)\n",
                    name, st.median_ns, st.min_ns, st.median_ns, st.stddev_ns,
                    st.samples, st.outliers);
        return st.median_ns;
    }

    // Print harness configuration info
    void print_config() const {
        (void)std::printf("  Timer:    %s\n", Timer::timer_name());
        (void)std::printf("  Warmup:   %d iterations\n", warmup_iters);
        (void)std::printf("  Passes:   %zu (IQR outlier removal + median)\n", passes);
    }
};

// -- Formatting helpers -------------------------------------------------------

inline const char* format_ns(double ns, char* buf, std::size_t buflen) {
    if (ns < 1000.0) {
        (void)std::snprintf(buf, buflen, "%.2f ns", ns);
    } else if (ns < 1000000.0) {
        (void)std::snprintf(buf, buflen, "%.2f us", ns / 1000.0);
    } else {
        (void)std::snprintf(buf, buflen, "%.2f ms", ns / 1000000.0);
    }
    return buf;
}

// Overload that uses a static buffer (not thread-safe, fine for sequential bench output)
inline const char* format_ns(double ns) {
    static char buf[64];
    return format_ns(ns, buf, sizeof(buf));
}

} // namespace bench
