// ============================================================================
// bench_unified.cpp -- Unified Apple-to-Apple Benchmark
// ============================================================================
//
// Single benchmark binary that runs on ALL platforms (x86, ARM64, RISC-V,
// ESP32) and produces identical output format everywhere.
//
// Measures UltrafastSecp256k1  vs  bitcoin-core libsecp256k1  vs  OpenSSL
// for every operation category:
//
//   1. Field arithmetic    (mul, sqr, inv, add, sub, negate)
//   2. Scalar arithmetic   (mul, inv, add, negate)
//   3. Point arithmetic    (k*G, k*P, a*G+b*P, add, dbl)
//   4. ECDSA               (sign FAST, verify)
//   5. Schnorr / BIP-340   (keypair, sign FAST, verify)
//   6. Constant-time       (CT sign ECDSA, CT sign Schnorr, overhead ratios)
//   7. libsecp256k1        (same ops for direct comparison)
//   7.5 OpenSSL            (ECDSA on secp256k1, system library)
//   8. Apple-to-Apple      (ratio table: Ultra / libsecp256k1 / OpenSSL)
//
// Methodology:
//   - Thread pinned to core 0, priority elevated
//   - 500 warmup iterations per operation (GLV precomputed tables are WARM
//     by the time measurement begins; results reflect steady-state throughput)
//   - 11 measurement passes, IQR outlier removal, median
//   - 64-key pool to prevent caching artifacts
//   - RDTSCP on x86, chrono fallback on ARM/RISC-V/ESP32
//
// Cache-state note:
//   All throughput figures in this benchmark are WARM-CACHE (L1/L2).
//   Cold-start performance (first call after process start, or after
//   ConnectBlock workload evicts pubkey tables) is lower; see the
//   ConnectBlock benchmark in docs/BITCOIN_CORE_BENCH_RESULTS.json
//   for a more realistic end-to-end workload measurement.
//
// CLI:
//   bench_unified [OPTIONS]
//     --json <file>    Write structured JSON report to <file>
//     --suite <name>   Run specific suite: core, extended, all (default: all)
//     --passes <N>     Override number of measurement passes (default: 11)
//     --quick          CI smoke mode: 3 passes, reduced iterations
//     --no-warmup      Skip CPU frequency ramp-up
//     --help           Show usage
//
// Build:
//   Part of CMake: target "bench_unified"
//   Requires libsecp256k1 source at _research_repos/secp256k1/
//
// ============================================================================

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/taproot.hpp"
#include "secp256k1/address.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/bip39.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/zk.hpp"
#include "secp256k1/pedersen.hpp"
#include "secp256k1/adaptor.hpp"
#include "secp256k1/frost.hpp"
#include "secp256k1/musig2.hpp"
#include "secp256k1/ecies.hpp"
#include "secp256k1/multiscalar.hpp"
#include "secp256k1/pippenger.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/sha512.hpp"
#include "secp256k1/bip143.hpp"
#include "secp256k1/bip144.hpp"
#include "secp256k1/segwit.hpp"
#include "secp256k1/coins/message_signing.hpp"
#ifdef SECP256K1_BIP324
#include "secp256k1/bip324.hpp"
#include "secp256k1/chacha20_poly1305.hpp"
#include "secp256k1/ellswift.hpp"
#include "secp256k1/hkdf.hpp"
#endif
#include "secp256k1/selftest.hpp"
#include "secp256k1/init.hpp"
#include "secp256k1/benchmark_harness.hpp"
#include "secp256k1/glv.hpp"
#include "secp256k1/batch_verify.hpp"
#ifdef SECP256K1_BUILD_ETHEREUM
#include "secp256k1/recovery.hpp"
#include "secp256k1/coins/keccak256.hpp"
#include "secp256k1/coins/ethereum.hpp"
#include "secp256k1/coins/eth_signing.hpp"
#endif
#include "secp256k1/coins/coin_address.hpp"
#include "secp256k1/coins/coin_hd.hpp"
#include "secp256k1/coins/coin_params.hpp"
#if defined(__SIZEOF_INT128__) && !defined(__EMSCRIPTEN__)
#include "secp256k1/field_52.hpp"
#endif

// libsecp256k1 public API (linked from libsecp_provider.c)
#include "secp256k1.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"
#ifdef SECP256K1_BUILD_ETHEREUM
#include "secp256k1_recovery.h"
#endif

// Thin wrappers from libsecp_provider.c exposing internal field ops
extern "C" {
    void libsecp_fe_inv_var(unsigned char out32[32], const unsigned char in32[32]);
    void libsecp_fe_inv_var_raw(void *r, const void *a);
    void libsecp_fe_mul(void *r, const void *a, const void *b);
    void libsecp_fe_sqr(void *r, const void *a);
    void libsecp_fe_add(void *r, const void *a);
    void libsecp_fe_negate(void *r, const void *a, int m);
    void libsecp_fe_normalize(void *r);
    void libsecp_fe_set_b32(void *r, const unsigned char *b32);
    void libsecp_scalar_mul(void *r, const void *a, const void *b);
    void libsecp_scalar_inverse(void *r, const void *a);
    void libsecp_scalar_inverse_var(void *r, const void *a);
    void libsecp_scalar_add(void *r, const void *a, const void *b);
    void libsecp_scalar_negate(void *r, const void *a);
    void libsecp_scalar_set_b32(void *r, const unsigned char *b32, int *overflow);
    void libsecp_gej_double_var(void *r, const void *a);
    void libsecp_gej_add_ge_var(void *r, const void *a, const void *b);
    void libsecp_ecmult(void *r, const void *a, const void *na, const void *ng);
    void libsecp_ecmult_gen(const void *ctx_ecmult_gen, void *r, const void *k);
    const void* libsecp_get_ecmult_gen_ctx(const secp256k1_context *ctx);
    void libsecp_gej_set_ge(void *r, const void *a);
    int  libsecp_pubkey_load(const secp256k1_context *ctx, void *ge,
                             const secp256k1_pubkey *pubkey);
}

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

// OpenSSL (optional, system library -- enabled by CMake find_package(OpenSSL))
#ifdef BENCH_HAS_OPENSSL
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/bn.h>
#include <openssl/obj_mac.h>
#include <openssl/opensslv.h>
#include <openssl/crypto.h>
#endif

// ---- JSON Result Collector --------------------------------------------------
// Accumulates all benchmark results for optional JSON export.

struct BenchEntry {
    char section[64];
    char name[64];
    double ns;
    double ratio;     // 0.0 if not a ratio entry
    bool is_ratio;
};

static constexpr int MAX_ENTRIES = 256;

struct BenchReport {
    BenchEntry entries[MAX_ENTRIES];
    int count;
    char cpu_brand[49];
    char compiler[64];
    char arch[32];
    char timer[48];
    double tsc_ghz;
    int passes;
    int warmup;
    int pool_size;
    bool quick_mode;

    void add(const char* section, const char* name, double ns_val) {
        if (count >= MAX_ENTRIES) return;
        auto& e = entries[count++];
        snprintf(e.section, sizeof(e.section), "%s", section);
        snprintf(e.name, sizeof(e.name), "%s", name);
        e.ns = ns_val;
        e.ratio = 0.0;
        e.is_ratio = false;
    }

    void add_ratio(const char* section, const char* name, double ratio_val) {
        if (count >= MAX_ENTRIES) return;
        auto& e = entries[count++];
        snprintf(e.section, sizeof(e.section), "%s", section);
        snprintf(e.name, sizeof(e.name), "%s", name);
        e.ns = 0.0;
        e.ratio = ratio_val;
        e.is_ratio = true;
    }

    bool write_json(const char* path) const {
        FILE* f = fopen(path, "w");
        if (!f) return false;

        fprintf(f, "{\n");
        fprintf(f, "  \"metadata\": {\n");
        fprintf(f, "    \"cpu\": \"%s\",\n", cpu_brand);
        fprintf(f, "    \"compiler\": \"%s\",\n", compiler);
        fprintf(f, "    \"arch\": \"%s\",\n", arch);
        fprintf(f, "    \"timer\": \"%s\",\n", timer);
        fprintf(f, "    \"tsc_ghz\": %.3f,\n", tsc_ghz);
        fprintf(f, "    \"passes\": %d,\n", passes);
        fprintf(f, "    \"warmup\": %d,\n", warmup);
        fprintf(f, "    \"pool_size\": %d,\n", pool_size);
        fprintf(f, "    \"run_mode\": \"%s\"\n", quick_mode ? "quick" : "full");
        fprintf(f, "  },\n");

        fprintf(f, "  \"results\": [\n");
        for (int i = 0; i < count; ++i) {
            const auto& e = entries[i];
            fprintf(f, "    {\"section\": \"%s\", \"name\": \"%s\"", e.section, e.name);
            if (e.is_ratio) {
                fprintf(f, ", \"ratio\": %.4f", e.ratio);
            } else {
                fprintf(f, ", \"ns\": %.2f", e.ns);
            }
            fprintf(f, "}%s\n", (i + 1 < count) ? "," : "");
        }
        fprintf(f, "  ]\n");
        fprintf(f, "}\n");
        fclose(f);
        return true;
    }
};

static BenchReport g_report{};

// ---- CLI Options ------------------------------------------------------------

struct CliOptions {
    const char* json_path;  // NULL if no JSON output
    int passes;             // 0 = use default
    bool quick;             // reduced iterations for CI
    bool no_warmup;         // skip CPU frequency ramp-up
    bool help;
    bool compare;           // --compare: only sign/verify + libsecp ratio, fast

    // Suite filter: 0=all, 1=core (field/scalar/point/ecdsa/schnorr/libsecp/ratio)
    //               2=extended (+ ct, batch, micro-diagnostics)
    int suite;
};

static CliOptions parse_cli(int argc, char** argv) {
    CliOptions opts{};
    opts.json_path = nullptr;
    opts.passes = 0;
    opts.quick = false;
    opts.no_warmup = false;
    opts.help = false;
    opts.compare = false;
    opts.suite = 0;  // all

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--json") == 0 && i + 1 < argc) {
            opts.json_path = argv[++i];
        } else if (strcmp(argv[i], "--passes") == 0 && i + 1 < argc) {
            opts.passes = atoi(argv[++i]);
            if (opts.passes < 3) opts.passes = 3;
        } else if (strcmp(argv[i], "--quick") == 0) {
            opts.quick = true;
        } else if (strcmp(argv[i], "--no-warmup") == 0) {
            opts.no_warmup = true;
        } else if (strcmp(argv[i], "--suite") == 0 && i + 1 < argc) {
            ++i;
            if (strcmp(argv[i], "core") == 0) opts.suite = 1;
            else if (strcmp(argv[i], "extended") == 0) opts.suite = 2;
            else opts.suite = 0;  // all
        } else if (strcmp(argv[i], "--compare") == 0) {
            opts.compare = true;
            opts.no_warmup = true;
            if (opts.passes == 0) opts.passes = 7;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            opts.help = true;
        }
    }
    return opts;
}

static void print_usage() {
    printf("Usage: bench_unified [OPTIONS]\n");
    printf("  --json <file>    Write structured JSON report to <file>\n");
    printf("  --suite <name>   core | extended | all (default: all)\n");
    printf("  --passes <N>     Override measurement passes (default: 11, min: 3)\n");
    printf("  --quick          CI smoke mode (3 passes, reduced iterations)\n");
    printf("  --no-warmup      Skip CPU frequency ramp-up\n");
    printf("  --help           Show this help\n");
}

// ---- CPU identification -----------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  #define BENCH_IS_X86 1
  #if defined(_MSC_VER)
    #include <intrin.h>
  #else
    #include <cpuid.h>
    #include <x86intrin.h>
    static inline void gcc_compat_cpuid(int regs[4], int level) {
        __cpuid(level, regs[0], regs[1], regs[2], regs[3]);
    }
    #undef __cpuid
    // NOLINTNEXTLINE(bugprone-reserved-identifier,cert-dcl37-c,cert-dcl51-cpp)
    #define __cpuid(regs, level) gcc_compat_cpuid(regs, level)
  #endif
#else
  #define BENCH_IS_X86 0
#endif

#define STR_(x) #x
#define STR(x)  STR_(x)

using namespace secp256k1::fast;
using namespace secp256k1;

// ---- CPU brand string -------------------------------------------------------

static void get_cpu_brand(char brand[49]) {
#if BENCH_IS_X86
    int regs[4];
    __cpuid(regs, 0x80000000);
    const auto max_ext = static_cast<unsigned>(regs[0]);
    if (max_ext < 0x80000004u) {
        (void)snprintf(brand, 49, "(unknown x86 CPU)");
        return;
    }
    for (unsigned i = 0; i < 3; ++i) {
        __cpuid(regs, 0x80000002u + i);
        std::memcpy(brand + static_cast<std::size_t>(i) * 16, regs, 16);
    }
    brand[48] = '\0';
    char* p = brand;
    while (*p == ' ') ++p;
    if (p != brand) std::memmove(brand, p, 49 - static_cast<std::size_t>(p - brand));
#elif defined(__aarch64__)
    (void)snprintf(brand, 49, "AArch64");
#elif defined(__riscv)
    (void)snprintf(brand, 49, "RISC-V 64");
#elif defined(__XTENSA__)
    (void)snprintf(brand, 49, "Xtensa (ESP32)");
#else
    (void)snprintf(brand, 49, "(unknown)");
#endif
}

// ---- CPU frequency warmup (defeats powersave governor) -----------------------
// Runs heavy crypto work for `target_ms` to force the CPU frequency up, then
// monitors TSC rate until it stabilises (two consecutive 200ms windows within 1%).

static void cpu_frequency_warmup() {
    using Clock = std::chrono::steady_clock;

    // Phase 1: heavy load for 3 seconds using real crypto work
    constexpr int TARGET_MS = 3000;
    printf("  CPU frequency warmup (%d ms heavy load)...", TARGET_MS);
    fflush(stdout);

    Scalar k = Scalar::from_bytes(std::array<uint8_t,32>{
        0xde,0xad,0xbe,0xef,0x01,0x02,0x03,0x04,
        0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0x0c,
        0x0d,0x0e,0x0f,0x10,0x11,0x12,0x13,0x14,
        0x15,0x16,0x17,0x18,0x19,0x1a,0x1b,0x1c}.data());
    Point G = Point::generator();
    volatile uint64_t anti_opt = 0;

    auto const start = Clock::now();
    int iters = 0;
    while (std::chrono::duration_cast<std::chrono::milliseconds>(
               Clock::now() - start).count() < TARGET_MS) {
        // Real ecmult work — same hot path as verify
        Point R = G.scalar_mul(k);
        auto bytes = R.to_compressed();
        anti_opt += bytes[1];
        k = k + Scalar::one();
        ++iters;
    }
    (void)anti_opt;

    // Phase 2: measure TSC rate stabilisation (two consecutive windows within 1%)
#if BENCH_HAS_RDTSC
    double prev_ghz = 0.0;
    constexpr int WINDOW_MS = 200;
    for (int attempt = 0; attempt < 10; ++attempt) {
        unsigned aux = 0;
        uint64_t const tsc0 = __rdtscp(&aux);
        auto const w0 = Clock::now();
        // Busy spin with light work
        while (std::chrono::duration_cast<std::chrono::milliseconds>(
                   Clock::now() - w0).count() < WINDOW_MS) {
            Point R = G.scalar_mul(k);
            auto bytes = R.to_compressed();
            anti_opt += bytes[1];
            k = k + Scalar::one();
        }
        uint64_t const tsc1 = __rdtscp(&aux);
        auto const w1 = Clock::now();
        double const ns_elapsed = static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(w1 - w0).count());
        double const ghz = static_cast<double>(tsc1 - tsc0) / ns_elapsed;

        if (prev_ghz > 0.1) {
            double const drift = std::abs(ghz - prev_ghz) / prev_ghz;
            if (drift < 0.01) {
                // Stable — done
                printf(" stable at %.3f GHz (%d k*G ops)\n", ghz, iters);
                fflush(stdout);
                return;
            }
        }
        prev_ghz = ghz;
    }
    printf(" done (%d k*G ops, freq may still drift)\n", iters);
#else
    printf(" done (%d k*G ops)\n", iters);
#endif
    fflush(stdout);
}

// ---- TSC frequency calibration (x86 only) -----------------------------------

static double calibrate_tsc_ghz() {
#if BENCH_IS_X86 && (defined(__x86_64__) || defined(_M_X64))
    unsigned aux = 0;
    const uint64_t t0 = __rdtscp(&aux);
    auto wall0 = std::chrono::high_resolution_clock::now();
    volatile uint64_t sink = 0;
    for (int i = 0; i < 5000000; ++i) sink += static_cast<uint64_t>(i);
    const uint64_t t1 = __rdtscp(&aux);
    auto wall1 = std::chrono::high_resolution_clock::now();
    const double ns = std::chrono::duration<double, std::nano>(wall1 - wall0).count();
    const auto cycles = static_cast<double>(t1 - t0);
    (void)sink;
    return cycles / ns;
#else
    return 0.0;
#endif
}

// ---- Harness ----------------------------------------------------------------

static bench::Harness H(500, 11);

template <typename Func>
static double bench_ns(Func&& f, int iters) {
    return H.run(iters, std::forward<Func>(f));
}

// ---- Data helpers -----------------------------------------------------------

static std::array<std::uint8_t, 32> make_hash(uint64_t seed) {
    std::array<std::uint8_t, 32> h{};
    for (int i = 0; i < 4; ++i) {
        uint64_t v = seed ^ (seed << 13) ^ (static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ULL);
        std::memcpy(&h[static_cast<std::size_t>(i) * 8], &v, 8);
    }
    return h;
}

static Scalar make_scalar(uint64_t seed) {
    auto h = make_hash(seed);
    return Scalar::from_bytes(h);
}

// ---- Formatting: single-column output ---------------------------------------

static double g_tsc_ghz = 0.0;

static void print_sep() {
    printf("+----------------------------------------------+------------+\n");
}

static const char* g_current_section = "";

static void print_header(const char* section) {
    print_sep();
    printf("| %-44s | %10s |\n", section, "ns/op");
    print_sep();
    g_current_section = section;
}

static void print_header_ratio(const char* section) {
    print_sep();
    printf("| %-44s | %10s |\n", section, "ratio");
    print_sep();
    g_current_section = section;
}

static void print_row(const char* name, double ns) {
    printf("| %-44s | %10.1f |\n", name, ns);
    g_report.add(g_current_section, name, ns);
}

static void print_ratio(const char* name, double ratio) {
    printf("| %-44s | %9.2fx |\n", name, ratio);
    g_report.add_ratio(g_current_section, name, ratio);
}

static void print_sep_3col() {
    printf("+------------------------------------+----------+----------+-----------+\n");
}

static void print_header_3col(const char* section) {
    print_sep_3col();
    printf("| %-34s | %8s | %8s | %9s |\n", section, "Ultra ns", "libsecp", "ratio");
    print_sep_3col();
    g_current_section = section;
}

static void print_row_3col(const char* name, double ultra, double libsecp) {
    if (libsecp <= 0) {
        printf("| %-34s | %8.1f | %8s | %9s |\n", name, ultra, "---", "---");
    } else {
        double ratio = libsecp / ultra;
        printf("| %-34s | %8.1f | %8.1f | %8.2fx |\n", name, ultra, libsecp, ratio);
    }
    g_report.add(g_current_section, name, ultra);
}

// (libsecp256k1 is benchmarked inline in main() using the SAME Harness)

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv) {
    // ---- CLI ----------------------------------------------------------------
    auto opts = parse_cli(argc, argv);
    if (opts.help) {
        print_usage();
        return 0;
    }

    SECP256K1_INIT();
    bench::pin_thread_and_elevate();

    // ---- Apply CLI overrides ------------------------------------------------
    int effective_passes = 11;
    int effective_warmup = 500;
    double iter_scale = 1.0;

    if (opts.quick) {
        effective_passes = 3;
        effective_warmup = 50;
        iter_scale = 0.2;  // 1/5 iterations
    }
    if (opts.passes > 0) {
        effective_passes = opts.passes;
    }

    H = bench::Harness(effective_warmup, static_cast<std::size_t>(effective_passes));

    // ---- CPU frequency ramp-up (critical for powersave governor) ----
    if (!opts.no_warmup) {
        cpu_frequency_warmup();
    } else {
        printf("  CPU frequency warmup: SKIPPED (--no-warmup)\n");
    }

    char cpu_brand[49] = {};
    get_cpu_brand(cpu_brand);
    g_tsc_ghz = calibrate_tsc_ghz();

    // ---- Populate report metadata -------------------------------------------
    std::memcpy(g_report.cpu_brand, cpu_brand, 49);
    snprintf(g_report.compiler, sizeof(g_report.compiler), "%s",
#if defined(__clang__)
        "Clang " __clang_version__
#elif defined(_MSC_VER)
        "MSVC"
#elif defined(__GNUC__)
        "GCC " __VERSION__
#else
        "Unknown"
#endif
    );
    snprintf(g_report.arch, sizeof(g_report.arch),
#if defined(__x86_64__) || defined(_M_X64)
        "x86-64"
#elif defined(__aarch64__)
        "ARM64"
#elif defined(__riscv)
        "RISC-V 64"
#elif defined(__XTENSA__)
        "Xtensa (ESP32)"
#else
        "Unknown"
#endif
    );
    snprintf(g_report.timer, sizeof(g_report.timer), "%s", bench::Timer::timer_name());
    g_report.tsc_ghz = g_tsc_ghz;
    g_report.passes = effective_passes;
    g_report.warmup = effective_warmup;
    g_report.pool_size = 64;
    g_report.quick_mode = opts.quick;

    // Integrity check
    printf("Running integrity check... ");
    if (!secp256k1::fast::Selftest(false)) {
        printf("FAIL\n");
        return 1;
    }
    printf("OK\n\n");

    // ---- Header -------------------------------------------------------------
    printf("======================================================================\n");
    printf("  UltrafastSecp256k1 -- Unified Apple-to-Apple Benchmark\n");
    printf("======================================================================\n\n");
    printf("  CPU:       %s\n", cpu_brand);
    if (g_tsc_ghz > 0.1)
        printf("  TSC freq:  %.3f GHz\n", g_tsc_ghz);
    printf("  Core:      1 (pinned to core 0, priority elevated)\n");
    printf("  Compiler:  "
#if defined(__clang__)
        "Clang " __clang_version__
#elif defined(_MSC_VER)
        "MSVC " STR(_MSC_VER)
#elif defined(__GNUC__)
        "GCC " __VERSION__
#else
        "Unknown"
#endif
        "\n");
    printf("  Arch:      "
#if defined(__x86_64__) || defined(_M_X64)
        "x86-64"
#elif defined(__aarch64__)
        "ARM64 (AArch64)"
#elif defined(__riscv)
        "RISC-V 64"
#elif defined(__XTENSA__)
        "Xtensa (ESP32)"
#else
        "Unknown"
#endif
        "\n");
    printf("  Ultra:     UltrafastSecp256k1\n");
    printf("  libsecp:   bitcoin-core libsecp256k1 v0.7.x\n");
    printf("  Harness:   3s CPU ramp-up, %d warmup/op, %d passes, IQR outlier removal, median\n",
           effective_warmup, effective_passes);
    printf("  Timer:     %s\n", bench::Timer::timer_name());
    printf("  Pool:      64 independent key/msg/sig sets\n");
    printf("  NOTE:      Both Ultra and libsecp use IDENTICAL harness\n");
    printf("\n");

    // ---- Prepare test data --------------------------------------------------

    constexpr int POOL = 64;

    Scalar privkeys[POOL];
    for (int i = 0; i < POOL; ++i)
        privkeys[i] = make_scalar(0xdeadbeef00ULL + static_cast<uint64_t>(i));

    Point pubkeys[POOL];
    for (int i = 0; i < POOL; ++i) {
        pubkeys[i] = Point::generator().scalar_mul(privkeys[i]);
        pubkeys[i].normalize();  // affine (z_one_=true), same as libsecp internal repr
    }

    // EcdsaPublicKey: pre-parsed with GLV tables (what the C ABI cache stores).
    secp256k1::EcdsaPublicKey epubkeys[POOL];
    {
        uint8_t tmp33[33];
        for (int i = 0; i < POOL; ++i) {
            auto unc = pubkeys[i].to_uncompressed();
            tmp33[0] = 0x02 | ((unc[64] & 1) ? 1 : 0);
            std::memcpy(tmp33 + 1, unc.data() + 1, 32);
            secp256k1::ecdsa_pubkey_parse(epubkeys[i], tmp33, 33);
        }
    }

    std::array<std::uint8_t, 32> msghashes[POOL];
    for (int i = 0; i < POOL; ++i)
        msghashes[i] = make_hash(0xcafebabe00ULL + static_cast<uint64_t>(i));

    std::array<std::uint8_t, 32> aux_rands[POOL];
    for (int i = 0; i < POOL; ++i)
        aux_rands[i] = make_hash(0xfeedface00ULL + static_cast<uint64_t>(i));

    ECDSASignature ecdsa_sigs[POOL];
    for (int i = 0; i < POOL; ++i)
        ecdsa_sigs[i] = ecdsa_sign(msghashes[i], privkeys[i]);

    SchnorrKeypair schnorr_kps[POOL];
    SchnorrSignature schnorr_sigs[POOL];
    std::array<std::uint8_t, 32> schnorr_pubkeys_x[POOL];
    SchnorrXonlyPubkey schnorr_xonly[POOL];
    for (int i = 0; i < POOL; ++i) {
        schnorr_kps[i] = schnorr_keypair_create(privkeys[i]);
        schnorr_sigs[i] = schnorr_sign(schnorr_kps[i], msghashes[i], aux_rands[i]);
        schnorr_pubkeys_x[i] = schnorr_pubkey(privkeys[i]);
        schnorr_xonly[i] = schnorr_xonly_from_keypair(schnorr_kps[i]);
    }

    int idx = 0;

    constexpr int N_SIGN_BASE   = 500;
    constexpr int N_VERIFY_BASE = 500;
    constexpr int N_KEYGEN_BASE = 500;
    constexpr int N_FIELD_BASE  = 50000;
    constexpr int N_POINT_BASE  = 10000;
    constexpr int N_SCALAR_BASE = 500;

    const int N_SIGN   = static_cast<int>(N_SIGN_BASE   * iter_scale);
    const int N_VERIFY = static_cast<int>(N_VERIFY_BASE * iter_scale);
    const int N_KEYGEN = static_cast<int>(N_KEYGEN_BASE * iter_scale);
    const int N_FIELD  = static_cast<int>(N_FIELD_BASE  * iter_scale);
    const int N_POINT  = static_cast<int>(N_POINT_BASE  * iter_scale);
    const int N_SCALAR = static_cast<int>(N_SCALAR_BASE * iter_scale);

    // =====================================================================
    //  SECTION 1: Field Arithmetic
    // =====================================================================

    print_header("FIELD ARITHMETIC (Ultra)");

    auto fe_a = FieldElement::from_hex(
        "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798");
    auto fe_b = FieldElement::from_hex(
        "483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8");

    const double fmul = bench_ns([&]() {
        auto r = fe_a * fe_b; bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_mul", fmul);

    const double fsqr = bench_ns([&]() {
        auto r = fe_a.square(); bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_sqr", fsqr);

    const double finv = bench_ns([&]() {
        auto r = fe_a.inverse(); bench::DoNotOptimize(r);
    }, 200);
    print_row("field_inv", finv);

    const double fadd = bench_ns([&]() {
        auto r = fe_a + fe_b; bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_add", fadd);

    const double fsub = bench_ns([&]() {
        auto r = fe_a - fe_b; bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_sub", fsub);

    const double fneg = bench_ns([&]() {
        auto r = fe_a.negate(); bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_negate", fneg);

    // -- from_bytes: parse 32-byte big-endian --
    auto fe_bytes_a = fe_a.to_bytes();
    const double fe_from_bytes = bench_ns([&]() {
        auto r = FieldElement::from_bytes(fe_bytes_a); bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("field_from_bytes (32B)", fe_from_bytes);
    print_sep();
    printf("\n");

    // =====================================================================
    //  SECTION 2: Scalar Arithmetic
    // =====================================================================

    print_header("SCALAR ARITHMETIC (Ultra)");

    auto sc_a = make_scalar(0xdeadbeef01ULL);
    auto sc_b = make_scalar(0xdeadbeef02ULL);

    const double smul = bench_ns([&]() {
        auto r = sc_a * sc_b; bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("scalar_mul", smul);

    const double sinv = bench_ns([&]() {
        auto r = sc_a.inverse(); bench::DoNotOptimize(r);
    }, 200);
    print_row("scalar_inv", sinv);

    const double sadd = bench_ns([&]() {
        auto r = sc_a + sc_b; bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("scalar_add", sadd);

    const double sneg = bench_ns([&]() {
        auto r = sc_a.negate(); bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("scalar_negate", sneg);

    // -- from_bytes: parse 32-byte to scalar --
    const double sc_from_bytes = bench_ns([&]() {
        auto r = Scalar::from_bytes(msghashes[idx % POOL]);
        bench::DoNotOptimize(r); ++idx;
    }, N_FIELD);
    print_row("scalar_from_bytes (32B)", sc_from_bytes);
    print_sep();
    printf("\n");

    // =====================================================================
    //  SECTION 3: Point Arithmetic
    // =====================================================================

    print_header("POINT ARITHMETIC (Ultra)");

    idx = 0;
    const double keygen = bench_ns([&]() {
        auto pk = Point::generator().scalar_mul(privkeys[idx % POOL]);
        bench::DoNotOptimize(pk); ++idx;
    }, N_KEYGEN);
    print_row("pubkey_create (k*G)", keygen);

    idx = 0;
    const double scalarmul = bench_ns([&]() {
        auto r = pubkeys[idx % POOL].scalar_mul(privkeys[(idx + 1) % POOL]);
        bench::DoNotOptimize(r); ++idx;
    }, N_SCALAR);
    print_row("scalar_mul (k*P)", scalarmul);

    // scalar_mul_with_plan: fixed K * variable Q (BIP-352 bottleneck)
    auto kplan = KPlan::from_scalar(privkeys[0], 4);
    idx = 0;
    const double plan_mul = bench_ns([&]() {
        auto r = pubkeys[idx % POOL].scalar_mul_with_plan(kplan);
        bench::DoNotOptimize(r); ++idx;
    }, N_SCALAR);
    print_row("scalar_mul_with_plan", plan_mul);

    idx = 0;
    const double dualmul = bench_ns([&]() {
        auto r = Point::dual_scalar_mul_gen_point(
            privkeys[idx % POOL], privkeys[(idx + 1) % POOL],
            pubkeys[(idx + 2) % POOL]);
        bench::DoNotOptimize(r); ++idx;
    }, N_SCALAR);
    print_row("dual_mul (a*G + b*P)", dualmul);

    const double ptadd = bench_ns([&]() {
        auto r = pubkeys[0].add(pubkeys[1]);
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("point_add (affine+affine)", ptadd);

    // Mixed add: Jacobian + Affine (8M+3S -- the actual hot path in scalar_mul)
    // Apple-to-apple with libsecp: split I/O, constant inputs, no loop-carried dep.
    // libsecp measures gej_add_ge_var(r, a, b) where a/b are constant per iteration;
    // we use Point::add() which calls jac52_add_mixed_to (same split I/O pattern).
    Point jac_test = pubkeys[0].dbl();  // non-affine (z != 1)
    const double ptadd_mixed = bench_ns([&]() {
        auto r = jac_test.add(pubkeys[1]);
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("point_add (J+A mixed)", ptadd_mixed);

    // Apple-to-apple with libsecp: split I/O, constant input, no loop-carried dep.
    // libsecp measures gej_double_var(r, a) where a is constant per iteration;
    // we use Point::dbl() which returns a new Point (same independence pattern).
    Point dbl_target = pubkeys[0].dbl();  // non-affine start
    const double ptdbl = bench_ns([&]() {
        auto r = dbl_target.dbl();
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("point_dbl", ptdbl);

    // normalize: Jacobian -> affine (1 field inversion + 2 muls)
    {
        Point norm_pts[POOL];
        for (int i = 0; i < POOL; ++i)
            norm_pts[i] = Point::generator().scalar_mul(privkeys[i]);
        idx = 0;
        const double pt_normalize = bench_ns([&]() {
            norm_pts[idx % POOL].normalize();
            bench::DoNotOptimize(norm_pts[idx % POOL]); ++idx;
        }, N_POINT);
        print_row("normalize (J->affine)", pt_normalize);
    }

    // batch_normalize: N points via Montgomery's trick (1 inv + 3(N-1) muls)
    {
        constexpr int BN = 64;
        Point bn_pts[BN];
        for (int i = 0; i < BN; ++i)
            bn_pts[i] = Point::generator().scalar_mul(privkeys[i % POOL]);
        FieldElement bn_out_x[BN], bn_out_y[BN];
        const double pt_batch_norm = bench_ns([&]() {
            Point::batch_normalize(bn_pts, BN, bn_out_x, bn_out_y);
            bench::DoNotOptimize(bn_out_x); bench::DoNotOptimize(bn_out_y);
        }, N_POINT / BN);
        print_row("batch_normalize /pt (N=64)", pt_batch_norm / BN);
    }

    // next_inplace: this += G (search hot-loop operation)
    {
        Point search_pt = pubkeys[0];
        const double pt_next = bench_ns([&]() {
            search_pt.next_inplace();
            bench::DoNotOptimize(search_pt);
        }, N_POINT);
        print_row("next_inplace (+=G)", pt_next);
    }

    // KPlan::from_scalar precomputation cost
    {
        idx = 0;
        const double kplan_cost = bench_ns([&]() {
            auto kp = KPlan::from_scalar(privkeys[idx % POOL], 4);
            bench::DoNotOptimize(kp); ++idx;
        }, N_POINT);
        print_row("KPlan::from_scalar(w=4)", kplan_cost);
    }
    print_sep();
    printf("\n");

    // =====================================================================
    //  SECTION 3.5: Point Serialization (Ultra)
    // =====================================================================
    // Apple-to-apple: both Ultra and libsecp store pubkeys as affine internally
    // (Ultra: z_one_=true after normalize(), libsecp: affine in secp256k1_pubkey).
    // Serialization = byte extraction only, no field inversion.
    // Batch methods shown for reference (amortize Jacobian->affine when needed).

    print_header("POINT SERIALIZATION (Ultra)");

    idx = 0;
    const double u_to_compressed = bench_ns([&]() {
        auto r = pubkeys[idx % POOL].to_compressed();
        bench::DoNotOptimize(r); ++idx;
    }, N_POINT);
    print_row("to_compressed (33B)", u_to_compressed);

    idx = 0;
    const double u_to_uncompressed = bench_ns([&]() {
        auto r = pubkeys[idx % POOL].to_uncompressed();
        bench::DoNotOptimize(r); ++idx;
    }, N_POINT);
    print_row("to_uncompressed (65B)", u_to_uncompressed);

    idx = 0;
    const double u_x_only = bench_ns([&]() {
        auto r = pubkeys[idx % POOL].x_only_bytes();
        bench::DoNotOptimize(r); ++idx;
    }, N_POINT);
    print_row("x_only_bytes (32B)", u_x_only);

    idx = 0;
    const double u_x_parity = bench_ns([&]() {
        auto r = pubkeys[idx % POOL].x_bytes_and_parity();
        bench::DoNotOptimize(r); ++idx;
    }, N_POINT);
    print_row("x_bytes_and_parity", u_x_parity);

    idx = 0;
    const double u_has_even_y = bench_ns([&]() {
        bool r = pubkeys[idx % POOL].has_even_y();
        bench::DoNotOptimize(r); ++idx;
    }, N_POINT);
    print_row("has_even_y", u_has_even_y);

    // Batch serialization (N=64 per batch, amortized cost per point)
    constexpr int BATCH_N = 64;
    {
        Point batch_pts[BATCH_N];
        for (int i = 0; i < BATCH_N; ++i)
            batch_pts[i] = pubkeys[i % POOL];

        std::array<uint8_t, 33> batch_out33[BATCH_N];
        const double u_batch_compressed = bench_ns([&]() {
            Point::batch_to_compressed(batch_pts, BATCH_N, batch_out33);
            bench::DoNotOptimize(batch_out33);
        }, N_POINT / BATCH_N);
        print_row("batch_to_compressed /pt (N=64)", u_batch_compressed / BATCH_N);

        std::array<uint8_t, 32> batch_out32[BATCH_N];
        const double u_batch_xonly = bench_ns([&]() {
            Point::batch_x_only_bytes(batch_pts, BATCH_N, batch_out32);
            bench::DoNotOptimize(batch_out32);
        }, N_POINT / BATCH_N);
        print_row("batch_x_only_bytes /pt (N=64)", u_batch_xonly / BATCH_N);
    }

    print_sep();
    printf("\n");

    // =====================================================================
    //  SECTION 4: ECDSA (Ultra FAST)
    // =====================================================================

    print_header("ECDSA -- Ultra FAST");

    idx = 0;
    const double u_ecdsa_sign = bench_ns([&]() {
        auto sig = ecdsa_sign(msghashes[idx % POOL], privkeys[idx % POOL]);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);
    print_row("ecdsa_sign", u_ecdsa_sign);

    idx = 0;
    const double u_ecdsa_sign_v = bench_ns([&]() {
        auto sig = ecdsa_sign_verified(msghashes[idx % POOL], privkeys[idx % POOL]);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);
    print_row("ecdsa_sign_verified", u_ecdsa_sign_v);

    idx = 0;
    const double u_ecdsa_verify = bench_ns([&]() {
        bool ok = ecdsa_verify(msghashes[idx % POOL], pubkeys[idx % POOL],
                               ecdsa_sigs[idx % POOL]);
        bench::DoNotOptimize(ok); ++idx;
    }, N_VERIFY);
    print_row("ecdsa_verify", u_ecdsa_verify);

    // With precomputed GLV tables (C ABI cache hot path).
    idx = 0;
    const double u_ecdsa_verify_epk = bench_ns([&]() {
        bool ok = secp256k1::ecdsa_verify(msghashes[idx % POOL].data(),
                                          epubkeys[idx % POOL],
                                          ecdsa_sigs[idx % POOL]);
        bench::DoNotOptimize(ok); ++idx;
    }, N_VERIFY);
    print_row("ecdsa_verify (cached EcdsaPublicKey)", u_ecdsa_verify_epk);
    print_sep();
    printf("\n");

    // =====================================================================
    //  SECTION 5: Schnorr / BIP-340 (Ultra FAST)
    // =====================================================================

    print_header("SCHNORR / BIP-340 -- Ultra FAST");

    idx = 0;
    const double u_schnorr_kp = bench_ns([&]() {
        auto kp = schnorr_keypair_create(privkeys[idx % POOL]);
        bench::DoNotOptimize(kp); ++idx;
    }, N_KEYGEN);
    print_row("schnorr_keypair_create", u_schnorr_kp);

    idx = 0;
    const double u_schnorr_sign = bench_ns([&]() {
        auto sig = schnorr_sign(schnorr_kps[idx % POOL], msghashes[idx % POOL],
                                aux_rands[idx % POOL]);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);
    print_row("schnorr_sign", u_schnorr_sign);

    idx = 0;
    const double u_schnorr_sign_v = bench_ns([&]() {
        auto sig = schnorr_sign_verified(schnorr_kps[idx % POOL], msghashes[idx % POOL],
                                          aux_rands[idx % POOL]);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);
    print_row("schnorr_sign_verified", u_schnorr_sign_v);

    idx = 0;
    const double u_schnorr_verify = bench_ns([&]() {
        bool ok = schnorr_verify(schnorr_xonly[idx % POOL],
                                 msghashes[idx % POOL],
                                 schnorr_sigs[idx % POOL]);
        bench::DoNotOptimize(ok); ++idx;
    }, N_VERIFY);
    print_row("schnorr_verify (cached xonly)", u_schnorr_verify);

    // Raw verify: takes 32-byte x-only pubkey bytes (includes lift_x sqrt).
    // This is what libsecp's schnorrsig_verify does internally.
    idx = 0;
    const double u_schnorr_verify_raw = bench_ns([&]() {
        bool ok = schnorr_verify(schnorr_pubkeys_x[idx % POOL].data(),
                                 msghashes[idx % POOL].data(),
                                 schnorr_sigs[idx % POOL]);
        bench::DoNotOptimize(ok); ++idx;
    }, N_VERIFY);
    print_row("schnorr_verify (raw bytes)", u_schnorr_verify_raw);
    print_sep();
    printf("\n");

    // =====================================================================
    //  SECTION 5.5: Micro-Diagnostics (verify sub-operation decomposition)
    // =====================================================================
    // Measures each sub-operation in isolation to identify where time is
    // spent inside verify paths.  Helps find bottlenecks vs libsecp.

    print_header("MICRO-DIAGNOSTICS (sub-ops)");

    // -- Scalar::from_bytes (parse 32-byte msg hash to scalar) --
    idx = 0;
    const double micro_scalar_from_bytes = bench_ns([&]() {
        auto s = Scalar::from_bytes(msghashes[idx % POOL]);
        bench::DoNotOptimize(s); ++idx;
    }, N_FIELD);
    print_row("Scalar::from_bytes (32B->scalar)", micro_scalar_from_bytes);

    // -- Scalar::inverse (safegcd modinv64) --
    const double micro_scalar_inv = bench_ns([&]() {
        auto r = sc_a.inverse(); bench::DoNotOptimize(r);
    }, 200);
    print_row("Scalar::inverse (safegcd)", micro_scalar_inv);

    // -- Scalar multiply (2x in verify: z*w, r*w) --
    const double micro_scalar_mul = bench_ns([&]() {
        auto r = sc_a * sc_b; bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("Scalar::mul", micro_scalar_mul);

    // -- Scalar negate --
    const double micro_scalar_negate = bench_ns([&]() {
        auto r = sc_a.negate(); bench::DoNotOptimize(r);
    }, N_FIELD);
    print_row("Scalar::negate", micro_scalar_negate);

    // -- GLV decomposition (split k -> k1, k2) --
    const double micro_glv = bench_ns([&]() {
        auto d = glv_decompose(privkeys[idx % POOL]);
        bench::DoNotOptimize(d); ++idx;
    }, N_POINT);
    print_row("glv_decompose", micro_glv);

    // -- Point::dbl (wrapper around jac52_double) --
    const double micro_pt_dbl = bench_ns([&]() {
        auto r = pubkeys[0].dbl();
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("Point::dbl (jac52_double)", micro_pt_dbl);

    // -- Point::add: Jacobian + Affine mixed (8M+3S hot-path formula) --
    Point jac_pt = pubkeys[0].dbl();  // non-affine (z != 1)
    const double micro_pt_add = bench_ns([&]() {
        auto r = jac_pt.add(pubkeys[1]);
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("Point::add (J+A mixed)", micro_pt_add);

    // -- dual_scalar_mul_gen_point (verify hot core) --
    idx = 0;
    const double micro_dual_mul = bench_ns([&]() {
        auto r = Point::dual_scalar_mul_gen_point(
            privkeys[idx % POOL], privkeys[(idx + 1) % POOL],
            pubkeys[(idx + 2) % POOL]);
        bench::DoNotOptimize(r); ++idx;
    }, N_SCALAR);
    print_row("dual_scalar_mul_gen_point", micro_dual_mul);

#if defined(SECP256K1_FAST_52BIT)
    // Outer-scope FE52 add/negate/normalize times for ratio table (hot-path repr)
    double micro_fe52_add = 0.0, micro_fe52_neg = 0.0, micro_fe52_norm_val = 0.0;
    // -- FE52::from_4x64_limbs (table lookup conversion cost) --
    {
        using FE52 = fast::FieldElement52;
        alignas(32) std::uint64_t limbs4x64[4] = {
            0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL,
            0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
        };
        const double micro_from_4x64 = bench_ns([&]() {
            auto r = FE52::from_4x64_limbs(limbs4x64);
            bench::DoNotOptimize(r);
        }, N_FIELD);
        print_row("FE52::from_4x64_limbs", micro_from_4x64);
    }

    // -- FE52 mul (52-bit field multiply, chained to prevent optimization) --
    {
        using FE52 = fast::FieldElement52;
        auto fe52_a = FE52::from_fe(fe_a);
        auto fe52_b = FE52::from_fe(fe_b);
        // Dependent chain: each mul feeds into the next to avoid CSE/hoisting
        const double micro_fe52_mul = bench_ns([&]() {
            fe52_a = fe52_a * fe52_b;
            bench::DoNotOptimize(fe52_a);
        }, N_FIELD);
        print_row("FE52::mul (52-bit)", micro_fe52_mul);

        fe52_a = FE52::from_fe(fe_a); // reset for sqr bench
        const double micro_fe52_sqr = bench_ns([&]() {
            fe52_a = fe52_a.square();
            bench::DoNotOptimize(fe52_a);
        }, N_FIELD);
        print_row("FE52::sqr (52-bit)", micro_fe52_sqr);
        fe52_a = FE52::from_fe(fe_a); // restore

        // -- FE52 inverse_safegcd (field inverse used by Schnorr verify) --
        auto fe52_inv_input = FE52::from_fe(fe_a);
        const double micro_fe52_inv = bench_ns([&]() {
            auto r = fe52_inv_input.inverse_safegcd();
            bench::DoNotOptimize(r);
        }, 200);
        print_row("FE52::inverse_safegcd", micro_fe52_inv);

        // -- FE52 inverse (Fermat addchain, 255 sqr + 13 mul) --
        auto fe52_inv_fermat_input = FE52::from_fe(fe_a);
        const double micro_fe52_inv_fermat = bench_ns([&]() {
            auto r = fe52_inv_fermat_input.inverse();
            bench::DoNotOptimize(r);
        }, 200);
        print_row("FE52::inverse (Fermat)", micro_fe52_inv_fermat);
        printf("| %-44s | %8.2fx  |\n",
               "  -> SafeGCD/Fermat speedup",
               micro_fe52_inv_fermat / micro_fe52_inv);

        // -- FE52 add / negate (5x52 lazy, same as libsecp hot path) --
        fe52_a = FE52::from_fe(fe_a);
        fe52_b = FE52::from_fe(fe_b);
        micro_fe52_add = bench_ns([&]() {
            auto r = fe52_a + fe52_b;
            bench::DoNotOptimize(r);
        }, N_FIELD);
        print_row("FE52::add (52-bit)", micro_fe52_add);

        micro_fe52_neg = bench_ns([&]() {
            auto r = fe52_a.negate(1);
            bench::DoNotOptimize(r);
        }, N_FIELD);
        print_row("FE52::negate (52-bit)", micro_fe52_neg);

        // -- FE52 normalize (full reduction to canonical form) --
        fe52_a = FE52::from_fe(fe_a);
        fe52_a = fe52_a + fe52_b; // magnitude > 1 so normalize has work to do
        const double micro_fe52_norm = bench_ns([&]() {
            auto r = fe52_a;
            r.normalize();
            bench::DoNotOptimize(r);
        }, N_FIELD);
        micro_fe52_norm_val = micro_fe52_norm;
        print_row("FE52::normalize", micro_fe52_norm);
    }
#endif

    // -- SHA256 challenge hash (BIP-340 tagged hash with midstate) --
    {
        const double micro_sha256_challenge = bench_ns([&]() {
            SHA256 ctx = detail::g_challenge_midstate;
            ctx.update(schnorr_sigs[idx % POOL].r.data(), 32);
            ctx.update(schnorr_xonly[idx % POOL].x_bytes.data(), 32);
            ctx.update(msghashes[idx % POOL].data(), 32);
            auto h = ctx.finalize();
            bench::DoNotOptimize(h); ++idx;
        }, N_FIELD);
        print_row("SHA256 (BIP0340/challenge)", micro_sha256_challenge);
    }

    // -- tagged_hash vs cached_tagged_hash (fix #1 validation) --
    {
        idx = 0;
        uint8_t th_input[96];
        std::memcpy(th_input, schnorr_sigs[0].r.data(), 32);
        std::memcpy(th_input + 32, schnorr_xonly[0].x_bytes.data(), 32);
        std::memcpy(th_input + 64, msghashes[0].data(), 32);

        const double micro_tagged_hash_slow = bench_ns([&]() {
            auto h = tagged_hash("BIP0340/challenge", th_input, 96);
            bench::DoNotOptimize(h);
        }, N_FIELD);
        print_row("tagged_hash (recompute tag)", micro_tagged_hash_slow);

        const double micro_tagged_hash_fast = bench_ns([&]() {
            auto h = detail::cached_tagged_hash(
                detail::g_challenge_midstate, th_input, 96);
            bench::DoNotOptimize(h);
        }, N_FIELD);
        print_row("cached_tagged_hash (midstate)", micro_tagged_hash_fast);

        printf("| %-44s | %8.2fx  |\n",
               "  -> midstate speedup",
               micro_tagged_hash_slow / micro_tagged_hash_fast);
    }

    // -- lift_x micro-benchmark (fix #2 validation) --
    {
        idx = 0;
        const double micro_lift_x = bench_ns([&]() {
            // Use the Point class lift_x path through schnorr verify's infrastructure
            FieldElement px_fe;
            bool ok = FieldElement::parse_bytes_strict(
                schnorr_xonly[idx % POOL].x_bytes, px_fe);
            if (ok) {
                auto x3 = px_fe.square() * px_fe;
                auto y2 = x3 + FieldElement::from_uint64(7);
                auto y = y2.sqrt();
                bench::DoNotOptimize(y);
            }
            ++idx;
        }, N_POINT);
        print_row("lift_x (4x64 sqrt)", micro_lift_x);

#if defined(SECP256K1_FAST_52BIT)
        {
            using FE52 = fast::FieldElement52;
            idx = 0;
            const double micro_lift_x_52 = bench_ns([&]() {
                FE52 const px52 = FE52::from_bytes(
                    schnorr_xonly[idx % POOL].x_bytes.data());
                FE52 const x3 = px52.square() * px52;
                static const FE52 seven52 = FE52::from_fe(
                    FieldElement::from_uint64(7));
                FE52 const y2 = x3 + seven52;
                FE52 y52 = y2.sqrt();
                bench::DoNotOptimize(y52);
                ++idx;
            }, N_POINT);
            print_row("lift_x (FE52 sqrt)", micro_lift_x_52);

            printf("| %-44s | %8.2fx  |\n",
                   "  -> FE52/4x64 speedup",
                   micro_lift_x / micro_lift_x_52);
        }
#endif
    }

    // -- FieldElement::parse_bytes_strict (BIP-340 range check) --
    {
        idx = 0;
        const double micro_parse_strict = bench_ns([&]() {
            FieldElement out;
            bool ok = FieldElement::parse_bytes_strict(
                schnorr_sigs[idx % POOL].r.data(), out);
            bench::DoNotOptimize(ok); bench::DoNotOptimize(out); ++idx;
        }, N_FIELD);
        print_row("FE::parse_bytes_strict", micro_parse_strict);
    }

    print_sep();

    // -- VERIFY DECOMPOSITION: show where time goes --
    printf("\n");
    printf("  ---- VERIFY COST DECOMPOSITION ----\n");
    printf("  ECDSA verify breakdown (estimated):\n");
    printf("    scalar_inv (1x):           %8.1f ns\n", micro_scalar_inv);
    printf("    scalar_mul (2x):           %8.1f ns\n", 2.0 * micro_scalar_mul);
    printf("    dual_scalar_mul:           %8.1f ns\n", micro_dual_mul);
    double ecdsa_sum = micro_scalar_inv + 2.0 * micro_scalar_mul
                     + micro_scalar_from_bytes + micro_dual_mul;
    printf("    from_bytes + overhead:     %8.1f ns\n", micro_scalar_from_bytes);
    printf("    --------------------------------\n");
    printf("    SUM (sub-ops):             %8.1f ns\n", ecdsa_sum);
    printf("    MEASURED ecdsa_verify:     %8.1f ns\n", u_ecdsa_verify);
    printf("    UNEXPLAINED gap:           %8.1f ns  (%.1f%%)\n",
           u_ecdsa_verify - ecdsa_sum,
           100.0 * (u_ecdsa_verify - ecdsa_sum) / u_ecdsa_verify);
    printf("\n");

    printf("  Schnorr verify breakdown (estimated):\n");
    printf("    SHA256 challenge:          (included in total)\n");
    printf("    scalar_negate:             %8.1f ns\n", micro_scalar_negate);
    printf("    dual_scalar_mul:           %8.1f ns\n", micro_dual_mul);
    printf("    lift_x (sqrt):             (included in total)\n");
    double schnorr_sum = micro_dual_mul + micro_scalar_negate
                       + micro_scalar_from_bytes;
    printf("    from_bytes:                %8.1f ns\n", micro_scalar_from_bytes);
    printf("    --------------------------------\n");
    printf("    SUM (sub-ops, partial):    %8.1f ns\n", schnorr_sum);
    printf("    MEASURED schnorr_verify:   %8.1f ns\n", u_schnorr_verify);
    printf("    UNEXPLAINED gap:           %8.1f ns  (SHA256+lift_x+Z-check)\n",
           u_schnorr_verify - schnorr_sum);
    printf("\n");

    printf("  Verify vs libsecp breakdown:\n");
    printf("    Our dual_mul:              %8.1f ns\n", micro_dual_mul);
    printf("    Our scalar_inv:            %8.1f ns\n", micro_scalar_inv);
    printf("    Our dual+inv:              %8.1f ns\n", micro_dual_mul + micro_scalar_inv);
    printf("    Total ECDSA verify:        %8.1f ns\n", u_ecdsa_verify);
    printf("    Overhead (verify - d+i):   %8.1f ns\n",
           u_ecdsa_verify - micro_dual_mul - micro_scalar_inv);
    printf("\n");

    // -- SIGN DECOMPOSITION: show where time goes --
    printf("  ---- SIGN COST DECOMPOSITION (FAST path) ----\n");
    printf("  ecdsa_sign = RFC6979 + k*G + field_inv + scalar_inv + scalar_muls\n");
    printf("    k*G (generator_mul):       %8.1f ns\n", keygen);
    printf("    field_inv (R.x):           %8.1f ns\n", finv);
    printf("    scalar_inv (k^-1):         %8.1f ns\n", micro_scalar_inv);
    printf("    scalar_mul (2x):           %8.1f ns\n", 2.0 * micro_scalar_mul);
    double sign_core = keygen + finv + micro_scalar_inv + 2.0 * micro_scalar_mul;
    printf("    --------------------------------\n");
    printf("    Core signing (no RFC6979):  %8.1f ns\n", sign_core);
    double rfc6979_cost = u_ecdsa_sign - sign_core;
    printf("    MEASURED ecdsa_sign:        %8.1f ns\n", u_ecdsa_sign);
    printf("    RFC6979 overhead:           %8.1f ns  (%.1f%%)\n",
           rfc6979_cost,
           100.0 * rfc6979_cost / u_ecdsa_sign);
    double verify_overhead = u_ecdsa_sign_v - u_ecdsa_sign;
    printf("    MEASURED ecdsa_sign_verified:%7.1f ns\n", u_ecdsa_sign_v);
    printf("    sign-then-verify overhead:  %8.1f ns  (pubkey + verify)\n",
           verify_overhead);
    printf("\n");

    // =====================================================================
    //  SECTION 5.7: Batch Verification (Schnorr + ECDSA)
    // =====================================================================
    // Measures batch verify for N = {4, 16, 64} signatures.
    // Reports total time, per-signature amortized cost, and speedup
    // vs N individual verify calls.
    //
    // Schnorr batch: single MSM (sum a_i*s_i)*G + sum(-a_i*e_i*P_i) + sum(-a_i*R_i) = O
    // ECDSA batch:   Montgomery batch inversion + per-sig Shamir's trick
    //
    // Acknowledgment: batch verification optimization approach inspired by
    //   Aaron Zhang's "Mastering Taproot" (https://github.com/aaron-recompile/mastering-taproot)
    //   Licensed under CC-BY-SA 4.0 (text) + MIT (code). Supported by OpenSats.
    //   Chapter 5 discusses Schnorr batch verification for block validation.

    print_header("BATCH VERIFICATION (FAST)");

    {
        // Build batch entries from the existing pool
        constexpr int BATCH_SIZES[] = {4, 16, 64, 128, 192};
        constexpr int N_BATCH_SIZES = 5;

        // -- Schnorr Batch Verify --
        for (int bi = 0; bi < N_BATCH_SIZES; ++bi) {
            const int batch_n = BATCH_SIZES[bi];

            // Prepare batch entries (reuse pool cyclically)
            std::vector<SchnorrBatchEntry> schnorr_batch(static_cast<std::size_t>(batch_n));
            std::vector<SchnorrBatchCachedEntry> schnorr_batch_cached(
                static_cast<std::size_t>(batch_n));
            for (int j = 0; j < batch_n; ++j) {
                schnorr_batch[static_cast<std::size_t>(j)].pubkey_x = schnorr_pubkeys_x[j % POOL];
                schnorr_batch[static_cast<std::size_t>(j)].message  = msghashes[j % POOL];
                schnorr_batch[static_cast<std::size_t>(j)].signature = schnorr_sigs[j % POOL];

                schnorr_batch_cached[static_cast<std::size_t>(j)].pubkey =
                    &schnorr_xonly[j % POOL];
                schnorr_batch_cached[static_cast<std::size_t>(j)].message =
                    msghashes[j % POOL];
                schnorr_batch_cached[static_cast<std::size_t>(j)].signature =
                    schnorr_sigs[j % POOL];
            }

            // Correctness sanity check
            bool batch_ok = schnorr_batch_verify(schnorr_batch);
            if (!batch_ok) {
                printf("[!] schnorr_batch_verify(%d) FAILED correctness check\n", batch_n);
            }

            bool cached_batch_ok = schnorr_batch_verify(schnorr_batch_cached);
            if (!cached_batch_ok) {
                printf("[!] schnorr_batch_verify(cached,%d) FAILED correctness check\n",
                       batch_n);
            }

            // Bench: fewer iterations for larger batches
            const int iters = batch_n <= 16 ? 200 :
                              batch_n <= 64 ? 100 :
                              batch_n <= 128 ? 40 : 25;
            const double batch_ns = bench_ns([&]() {
                bool ok = schnorr_batch_verify(schnorr_batch);
                bench::DoNotOptimize(ok);
            }, iters);

            double per_sig = batch_ns / static_cast<double>(batch_n);
            double speedup = u_schnorr_verify / per_sig;

            char label[64];
            snprintf(label, sizeof(label), "schnorr_batch_verify(N=%d)", batch_n);
            print_row(label, batch_ns);

            snprintf(label, sizeof(label), "  -> per-sig amortized (N=%d)", batch_n);
            print_row(label, per_sig);

            printf("| %-44s | %8.2fx  |\n",
                   batch_n <= 9 ? "  -> speedup vs individual" : "  -> speedup vs individual",
                   speedup);

                 const double cached_batch_ns = bench_ns([&]() {
                  bool ok = schnorr_batch_verify(schnorr_batch_cached);
                  bench::DoNotOptimize(ok);
                 }, iters);

                 double cached_per_sig = cached_batch_ns / static_cast<double>(batch_n);
                 double cached_speedup = u_schnorr_verify / cached_per_sig;

                 snprintf(label, sizeof(label), "schnorr_batch_verify(cached,N=%d)", batch_n);
                 print_row(label, cached_batch_ns);

                 snprintf(label, sizeof(label), "  -> per-sig cached (N=%d)", batch_n);
                 print_row(label, cached_per_sig);

                 printf("| %-44s | %8.2fx  |\n",
                     "  -> cached speedup vs individual",
                     cached_speedup);
        }

        printf("|                                              |            |\n");

        // -- ECDSA Batch Verify --
        for (int bi = 0; bi < N_BATCH_SIZES; ++bi) {
            const int batch_n = BATCH_SIZES[bi];

            std::vector<ECDSABatchEntry> ecdsa_batch(static_cast<std::size_t>(batch_n));
            for (int j = 0; j < batch_n; ++j) {
                ecdsa_batch[static_cast<std::size_t>(j)].msg_hash   = msghashes[j % POOL];
                ecdsa_batch[static_cast<std::size_t>(j)].public_key = pubkeys[j % POOL];
                ecdsa_batch[static_cast<std::size_t>(j)].signature  = ecdsa_sigs[j % POOL];
            }

            bool batch_ok = ecdsa_batch_verify(ecdsa_batch);
            if (!batch_ok) {
                printf("[!] ecdsa_batch_verify(%d) FAILED correctness check\n", batch_n);
            }

            const int iters = batch_n <= 16 ? 200 :
                              batch_n <= 64 ? 100 :
                              batch_n <= 128 ? 40 : 25;
            const double batch_ns = bench_ns([&]() {
                bool ok = ecdsa_batch_verify(ecdsa_batch);
                bench::DoNotOptimize(ok);
            }, iters);

            double per_sig = batch_ns / static_cast<double>(batch_n);
            double speedup = u_ecdsa_verify / per_sig;

            char label[64];
            snprintf(label, sizeof(label), "ecdsa_batch_verify(N=%d)", batch_n);
            print_row(label, batch_ns);

            snprintf(label, sizeof(label), "  -> per-sig amortized (N=%d)", batch_n);
            print_row(label, per_sig);

            printf("| %-44s | %8.2fx  |\n",
                   "  -> speedup vs individual", speedup);
        }
    }

    print_sep();
    printf("\n");

    // =====================================================================
    //  SECTION 6: Constant-Time Operations (Ultra CT)
    // =====================================================================


    print_header("CT POINT ARITHMETIC (sub-ops)");

    // -- CT scalar_inverse (SafeGCD on __int128, Fermat fallback) --
    idx = 0;
    const double ct_scalar_inv = bench_ns([&]() {
        auto r = ct::scalar_inverse(privkeys[idx % POOL]);
        bench::DoNotOptimize(r); ++idx;
    }, N_SCALAR);
    print_row("ct::scalar_inverse (SafeGCD)", ct_scalar_inv);

    // -- CT generator_mul (k*G, Hamburg comb + precomputed table) --
    idx = 0;
    const double ct_gen_mul = bench_ns([&]() {
        auto r = ct::generator_mul(privkeys[idx % POOL]);
        bench::DoNotOptimize(r); ++idx;
    }, N_KEYGEN);
    print_row("ct::generator_mul (k*G)", ct_gen_mul);

    // -- CT scalar_mul (k*P, Hamburg comb + GLV) --
    idx = 0;
    const double ct_scalar_mul = bench_ns([&]() {
        auto r = ct::scalar_mul(pubkeys[idx % POOL], privkeys[(idx + 1) % POOL]);
        bench::DoNotOptimize(r); ++idx;
    }, N_SCALAR);
    print_row("ct::scalar_mul (k*P)", ct_scalar_mul);

    // -- CT point_dbl --
    {
        auto ct_p = ct::CTJacobianPoint::from_point(pubkeys[0]);
        const double ct_dbl = bench_ns([&]() {
            auto r = ct::point_dbl(ct_p);
            bench::DoNotOptimize(r);
        }, N_POINT);
        print_row("ct::point_dbl", ct_dbl);
    }

    // -- CT point_add_complete (Jac+Jac, 11M+6S) --
    {
        auto ct_p = ct::CTJacobianPoint::from_point(pubkeys[0]);
        auto ct_q = ct::CTJacobianPoint::from_point(pubkeys[1]);
        const double ct_add_full = bench_ns([&]() {
            auto r = ct::point_add_complete(ct_p, ct_q);
            bench::DoNotOptimize(r);
        }, N_POINT);
        print_row("ct::point_add_complete (11M+6S)", ct_add_full);
    }

    // -- CT point_add_mixed_complete (Jac+Aff, 7M+5S) --
    {
        auto ct_p = ct::CTJacobianPoint::from_point(pubkeys[0]);
        auto ct_q_aff = ct::CTAffinePoint::from_point(pubkeys[1]);
        const double ct_add_mixed = bench_ns([&]() {
            auto r = ct::point_add_mixed_complete(ct_p, ct_q_aff);
            bench::DoNotOptimize(r);
        }, N_POINT);
        print_row("ct::point_add_mixed_complete (7M+5S)", ct_add_mixed);
    }

    // -- CT point_add_mixed_unified (Jac+Aff, 7M+5S, Brier-Joye) --
    {
        auto ct_p = ct::CTJacobianPoint::from_point(pubkeys[0]);
        auto ct_q_aff = ct::CTAffinePoint::from_point(pubkeys[1]);
        const double ct_add_unified = bench_ns([&]() {
            auto r = ct::point_add_mixed_unified(ct_p, ct_q_aff);
            bench::DoNotOptimize(r);
        }, N_POINT);
        print_row("ct::point_add_mixed_unified (7M+5S)", ct_add_unified);
    }

    print_sep();

    // -- CT vs FAST point ops comparison --
    printf("\n");
    printf("  ---- CT vs FAST point ops ----\n");
    printf("  %-36s %8.1f ns\n", "FAST Point::dbl", micro_pt_dbl);
    printf("  %-36s %8.1f ns\n", "FAST Point::add", micro_pt_add);
    printf("  %-36s %8.1f ns\n", "FAST pubkey_create (k*G)", keygen);
    printf("  %-36s %8.1f ns\n", "FAST scalar_mul (k*P)", scalarmul);
    printf("  %-36s %8.1f ns\n", "CT   generator_mul (k*G)", ct_gen_mul);
    printf("  %-36s %8.1f ns\n", "CT   scalar_mul (k*P)", ct_scalar_mul);
    printf("  CT/FAST ratio (k*G):  %.2fx overhead\n", ct_gen_mul / keygen);
    printf("  CT/FAST ratio (k*P):  %.2fx overhead\n", ct_scalar_mul / scalarmul);
    printf("\n");

    // -- CT Signing --
    print_header("CT SIGNING (Ultra CT)");

    idx = 0;
    const double u_ct_ecdsa = bench_ns([&]() {
        auto sig = ct::ecdsa_sign(msghashes[idx % POOL], privkeys[idx % POOL]);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);
    print_row("ct::ecdsa_sign", u_ct_ecdsa);
    print_ratio("  CT overhead (ECDSA)", u_ct_ecdsa / u_ecdsa_sign);

    idx = 0;
    const double u_ct_ecdsa_v = bench_ns([&]() {
        auto sig = ct::ecdsa_sign_verified(msghashes[idx % POOL], privkeys[idx % POOL]);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);
    print_row("ct::ecdsa_sign_verified", u_ct_ecdsa_v);

    idx = 0;
    const double u_ct_schnorr = bench_ns([&]() {
        auto sig = ct::schnorr_sign(schnorr_kps[idx % POOL],
                                     msghashes[idx % POOL],
                                     aux_rands[idx % POOL]);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);
    print_row("ct::schnorr_sign", u_ct_schnorr);
    print_ratio("  CT overhead (Schnorr)", u_ct_schnorr / u_schnorr_sign);

    idx = 0;
    const double u_ct_schnorr_v = bench_ns([&]() {
        auto sig = ct::schnorr_sign_verified(schnorr_kps[idx % POOL],
                                              msghashes[idx % POOL],
                                              aux_rands[idx % POOL]);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);
    print_row("ct::schnorr_sign_verified", u_ct_schnorr_v);

    // -- CT Schnorr Keypair --
    idx = 0;
    const double u_ct_schnorr_kp = bench_ns([&]() {
        auto kp = ct::schnorr_keypair_create(privkeys[idx % POOL]);
        bench::DoNotOptimize(kp); ++idx;
    }, N_KEYGEN);
    print_row("ct::schnorr_keypair_create", u_ct_schnorr_kp);
    print_ratio("  CT overhead (keypair)", u_ct_schnorr_kp / u_schnorr_kp);

    print_sep();

    // -- CT Sign Decomposition --
    printf("\n");
    printf("  ---- CT ECDSA SIGN DECOMPOSITION ----\n");
    printf("    ct::generator_mul (R=k*G): %8.1f ns\n", ct_gen_mul);
    printf("    ct::scalar_inverse (k^-1): %8.1f ns\n", ct_scalar_inv);
    printf("    field_inv (R.x affine):    %8.1f ns\n", finv);
    printf("    scalar_mul (2x):           %8.1f ns\n", 2.0 * micro_scalar_mul);
    double ct_ecdsa_sum = ct_gen_mul + ct_scalar_inv + finv + 2.0 * micro_scalar_mul;
    printf("    --------------------------------\n");
    printf("    SUM (sub-ops):             %8.1f ns\n", ct_ecdsa_sum);
    printf("    MEASURED ct::ecdsa_sign:   %8.1f ns\n", u_ct_ecdsa);
    printf("    UNEXPLAINED gap:           %8.1f ns  (%.1f%%, RFC6979+checks)\n",
           u_ct_ecdsa - ct_ecdsa_sum,
           100.0 * (u_ct_ecdsa - ct_ecdsa_sum) / u_ct_ecdsa);
    printf("\n");

    printf("  ---- CT SCHNORR SIGN DECOMPOSITION ----\n");
    printf("    ct::generator_mul (R=k*G): %8.1f ns\n", ct_gen_mul);
    printf("    SHA256 (tag+nonce+msg):    (included in total)\n");
    printf("    scalar_mul + negate:       %8.1f ns\n", micro_scalar_mul + micro_scalar_negate);
    double ct_schnorr_sum = ct_gen_mul + micro_scalar_mul + micro_scalar_negate;
    printf("    --------------------------------\n");
    printf("    SUM (sub-ops, partial):    %8.1f ns\n", ct_schnorr_sum);
    printf("    MEASURED ct::schnorr_sign: %8.1f ns\n", u_ct_schnorr);
    printf("    UNEXPLAINED gap:           %8.1f ns  (SHA256+aux+serialize)\n",
           u_ct_schnorr - ct_schnorr_sum);
    printf("\n");

    // -- CT vs libsecp comparison (libsecp is always CT) --
    printf("  ---- CT vs libsecp (true apples-to-apples) ----\n");
    printf("  %-36s %8.1f ns\n", "CT   ecdsa_sign", u_ct_ecdsa);
    printf("  %-36s (measured after libsecp section)\n", "lib  ecdsa_sign");
    printf("  %-36s %8.1f ns\n", "CT   schnorr_sign", u_ct_schnorr);
    printf("  %-36s (measured after libsecp section)\n", "lib  schnorr_sign");
    printf("\n");

    // =====================================================================
    //  SECTION 6.5: Ethereum Operations (conditional)
    // =====================================================================

#ifdef SECP256K1_BUILD_ETHEREUM
    double u_keccak_32 = 0, u_eth_addr = 0, u_eip191 = 0;
    double u_eth_sign = 0, u_sign_rec = 0, u_ecrecover = 0;
    double u_personal_sign = 0, u_eip55 = 0;
    {
        using namespace secp256k1::coins;

        print_header("ETHEREUM OPERATIONS");

        // Keccak-256 (32-byte input, typical hash-of-hash)
        idx = 0;
        u_keccak_32 = bench_ns([&]() {
            auto h = keccak256(msghashes[idx % POOL].data(), 32);
            bench::DoNotOptimize(h); ++idx;
        }, N_FIELD);
        print_row("keccak256 (32B)", u_keccak_32);

        // Ethereum address derivation from public key
        idx = 0;
        u_eth_addr = bench_ns([&]() {
            auto addr = ethereum_address_bytes(pubkeys[idx % POOL]);
            bench::DoNotOptimize(addr); ++idx;
        }, N_POINT);
        print_row("ethereum_address", u_eth_addr);

        // EIP-191 personal message hash
        idx = 0;
        const char eth_msg[] = "I agree to the terms of service";
        u_eip191 = bench_ns([&]() {
            auto h = eip191_hash(reinterpret_cast<const uint8_t*>(eth_msg), sizeof(eth_msg) - 1);
            bench::DoNotOptimize(h); ++idx;
        }, N_SIGN);
        print_row("eip191_hash", u_eip191);

        // ECDSA sign with recovery (eth_sign_hash)
        idx = 0;
        u_eth_sign = bench_ns([&]() {
            auto sig = eth_sign_hash(msghashes[idx % POOL], privkeys[idx % POOL], 1);
            bench::DoNotOptimize(sig); ++idx;
        }, N_SIGN);
        print_row("eth_sign_hash", u_eth_sign);

        // ECDSA recoverable sign (raw, no EIP encoding)
        idx = 0;
        u_sign_rec = bench_ns([&]() {
            auto sig = ecdsa_sign_recoverable(msghashes[idx % POOL], privkeys[idx % POOL]);
            bench::DoNotOptimize(sig); ++idx;
        }, N_SIGN);
        print_row("ecdsa_sign_recoverable", u_sign_rec);

        // ecrecover (full pipeline: recover pubkey + derive address)
        // Pre-sign for recovery pool
        EthSignature eth_sigs[POOL];
        for (int i = 0; i < POOL; ++i) {
            eth_sigs[i] = eth_sign_hash(msghashes[i], privkeys[i], 1);
        }

        idx = 0;
        u_ecrecover = bench_ns([&]() {
            auto [addr, ok] = ecrecover(msghashes[idx % POOL], eth_sigs[idx % POOL]);
            bench::DoNotOptimize(addr);
            bench::DoNotOptimize(ok);
            ++idx;
        }, N_VERIFY);
        print_row("ecrecover", u_ecrecover);

        // eth_personal_sign (EIP-191 hash + sign with recovery)
        idx = 0;
        u_personal_sign = bench_ns([&]() {
            auto sig = eth_personal_sign(reinterpret_cast<const uint8_t*>(eth_msg),
                                         sizeof(eth_msg) - 1, privkeys[idx % POOL]);
            bench::DoNotOptimize(sig); ++idx;
        }, N_SIGN);
        print_row("eth_personal_sign", u_personal_sign);

        // EIP-55 checksummed address (string output)
        idx = 0;
        u_eip55 = bench_ns([&]() {
            auto addr = ethereum_address(pubkeys[idx % POOL]);
            bench::DoNotOptimize(addr); ++idx;
        }, N_POINT);
        print_row("ethereum_address_eip55", u_eip55);

        print_sep();
        printf("\n");
    }
#endif // SECP256K1_BUILD_ETHEREUM

    // =====================================================================
    //  SECTION 6.7: Real-World Wallet / Protocol Flows
    // =====================================================================

    double u_ecdh = 0, u_ecdh_raw = 0, u_taproot_out = 0, u_taproot_tweak = 0;
    double u_bip32_master = 0, u_bip32_child = 0, u_coin_addr_btc = 0, u_coin_addr_eth = 0;
    double u_silent_sender = 0, u_silent_scan = 0;
    {
        print_header("REAL-WORLD FLOWS");

        std::array<std::uint8_t, 64> hd_seed{};
        std::memcpy(hd_seed.data(), msghashes[0].data(), 32);
        std::memcpy(hd_seed.data() + 32, msghashes[1].data(), 32);

        auto [master_hd, master_ok] = bip32_master_key(hd_seed.data(), hd_seed.size());
        if (!master_ok) {
            printf("[!] bip32_master_key() setup failed\n");
            return 1;
        }

        std::array<std::uint8_t, 32> empty_merkle{};
        auto internal_key_x = schnorr_pubkeys_x[0];
        std::vector<Scalar> sp_input_sks{privkeys[2], privkeys[3]};
        std::vector<Point> sp_input_pks{pubkeys[2], pubkeys[3]};
        auto sp_addr = silent_payment_address(privkeys[0], privkeys[1]);
        auto [sp_output_pk, _sp_tweak] = silent_payment_create_output(sp_input_sks, sp_addr, 0);
        std::vector<std::array<std::uint8_t, 32>> sp_outputs{sp_output_pk.x().to_bytes()};

        idx = 0;
        u_ecdh = bench_ns([&]() {
            auto s = ecdh_compute(privkeys[idx % POOL], pubkeys[(idx + 1) % POOL]);
            bench::DoNotOptimize(s); ++idx;
        }, N_VERIFY);
        print_row("ecdh_compute (SHA256 shared secret)", u_ecdh);

        idx = 0;
        u_ecdh_raw = bench_ns([&]() {
            auto s = ecdh_compute_raw(privkeys[idx % POOL], pubkeys[(idx + 1) % POOL]);
            bench::DoNotOptimize(s); ++idx;
        }, N_VERIFY);
        print_row("ecdh_compute_raw (x-only shared)", u_ecdh_raw);

        u_taproot_out = bench_ns([&]() {
            auto out = taproot_output_key(internal_key_x, empty_merkle.data(), 0);
            bench::DoNotOptimize(out);
        }, N_SIGN);
        print_row("taproot_output_key (BIP-341 key path)", u_taproot_out);

        u_taproot_tweak = bench_ns([&]() {
            auto s = taproot_tweak_privkey(privkeys[0], empty_merkle.data(), 0);
            bench::DoNotOptimize(s);
        }, N_SIGN);
        print_row("taproot_tweak_privkey (BIP-341)", u_taproot_tweak);

        u_bip32_master = bench_ns([&]() {
            auto mk = bip32_master_key(hd_seed.data(), hd_seed.size());
            bench::DoNotOptimize(mk);
        }, N_SIGN);
        print_row("bip32_master_key (64B seed)", u_bip32_master);

        u_bip32_child = bench_ns([&]() {
            auto child = secp256k1::coins::coin_derive_key(master_hd, secp256k1::coins::Bitcoin, 0, false, 0);
            bench::DoNotOptimize(child);
        }, N_SIGN);
        print_row("bip32_coin_derive_key (BTC m/84'/0'/0'/0/0)", u_bip32_child);

        u_coin_addr_btc = bench_ns([&]() {
            auto addr = secp256k1::coins::coin_address_from_seed(
                hd_seed.data(), hd_seed.size(), secp256k1::coins::Bitcoin, 0, 0);
            bench::DoNotOptimize(addr);
        }, N_SIGN);
        print_row("coin_address_from_seed (BTC end-to-end)", u_coin_addr_btc);

        u_coin_addr_eth = bench_ns([&]() {
            auto addr = secp256k1::coins::coin_address_from_seed(
                hd_seed.data(), hd_seed.size(), secp256k1::coins::Ethereum, 0, 0);
            bench::DoNotOptimize(addr);
        }, N_SIGN);
        print_row("coin_address_from_seed (ETH end-to-end)", u_coin_addr_eth);

        u_silent_sender = bench_ns([&]() {
            auto out = silent_payment_create_output(sp_input_sks, sp_addr, static_cast<std::uint32_t>(idx & 3));
            bench::DoNotOptimize(out); ++idx;
        }, N_SIGN);
        print_row("silent_payment_create_output", u_silent_sender);

        idx = 0;
        u_silent_scan = bench_ns([&]() {
            auto found = silent_payment_scan(privkeys[0], privkeys[1], sp_input_pks, sp_outputs);
            bench::DoNotOptimize(found); ++idx;
        }, N_SIGN);
        print_row("silent_payment_scan (single output set)", u_silent_scan);

        // fast_scan_batch benchmarks — amortised cost per tx for various batch sizes
        {
            // Build ScanTx from the same sp data
            ScanTx sp_scan_tx;
            {
                ScanTxRaw raw;
                raw.input_pubkeys = sp_input_pks;
                raw.smallest_outpoint.fill(0);
                sp_scan_tx = compute_a_eff(raw);
            }
            sp_scan_tx.outputs = sp_outputs;

            for (int bsz : {1, 16, 64, 256, 1024}) {
                std::vector<ScanTx> batch(static_cast<size_t>(bsz), sp_scan_tx);
                double total_ns = bench_ns([&]() {
                    auto found = fast_scan_batch(privkeys[0], privkeys[1], batch);
                    bench::DoNotOptimize(found);
                }, N_SIGN);
                double per_tx_ns = total_ns / static_cast<double>(bsz);
                char row_name[80];
                std::snprintf(row_name, sizeof(row_name),
                              "fast_scan_batch /tx (N=%d)", bsz);
                print_row(row_name, per_tx_ns);
            }
        }

        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 7: libsecp256k1 (bitcoin-core) -- SAME harness, pool, timer
    // =====================================================================

    printf("Running libsecp256k1 benchmark (same harness: RDTSCP, 3s ramp-up, %d warmup, %d passes, IQR)...\n",
           effective_warmup, effective_passes);

    secp256k1_context* ls_ctx = secp256k1_context_create(SECP256K1_CONTEXT_NONE);
    if (!ls_ctx) {
        printf("[FAIL] libsecp256k1 context creation failed\n");
        return 1;
    }

    // Prepare libsecp pool -- same 64 seeds as Ultra
    unsigned char              ls_seckeys[POOL][32];
    secp256k1_pubkey           ls_pubkeys[POOL];
    secp256k1_keypair          ls_keypairs[POOL];
    secp256k1_xonly_pubkey     ls_xonly[POOL];
    unsigned char              ls_msgs[POOL][32];
    unsigned char              ls_aux[POOL][32];
    secp256k1_ecdsa_signature  ls_esigs[POOL];
    unsigned char              ls_schnorr_sigs[POOL][64];

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-result"
    for (int i = 0; i < POOL; ++i) {
        auto const h = make_hash(0xdeadbeef00ULL + static_cast<uint64_t>(i));
        std::memcpy(ls_seckeys[i], h.data(), 32);
        (void)secp256k1_ec_pubkey_create(ls_ctx, &ls_pubkeys[i], ls_seckeys[i]);
        (void)secp256k1_keypair_create(ls_ctx, &ls_keypairs[i], ls_seckeys[i]);
        secp256k1_keypair_xonly_pub(ls_ctx, &ls_xonly[i], NULL, &ls_keypairs[i]);

        auto const mh = make_hash(0xcafebabe00ULL + static_cast<uint64_t>(i));
        std::memcpy(ls_msgs[i], mh.data(), 32);

        auto const ar = make_hash(0xfeedface00ULL + static_cast<uint64_t>(i));
        std::memcpy(ls_aux[i], ar.data(), 32);

        secp256k1_ecdsa_sign(ls_ctx, &ls_esigs[i], ls_msgs[i], ls_seckeys[i], NULL, NULL);
        secp256k1_schnorrsig_sign32(ls_ctx, ls_schnorr_sigs[i], ls_msgs[i],
                                    &ls_keypairs[i], ls_aux[i]);
    }

    // --- libsecp field_inv_var micro-benchmark ---
    // Uses a 32-byte serialisation round-trip wrapper; measures the
    // full inv_var cost (set_b32 + inv_var + normalize + get_b32).
    unsigned char ls_fe_in[32];
    unsigned char ls_fe_out[32];
    std::memcpy(ls_fe_in,
        "\x79\xbe\x66\x7e\xf9\xdc\xbb\xac"
        "\x55\xa0\x62\x95\xce\x87\x0b\x07"
        "\x02\x9b\xfc\xdb\x2d\xce\x28\xd9"
        "\x59\xf2\x81\x5b\x16\xf8\x17\x98", 32);

    const double ls_fe_inv = bench_ns([&]() {
        libsecp_fe_inv_var(ls_fe_out, ls_fe_in);
        bench::DoNotOptimize(ls_fe_out);
        // Feed output back as input to prevent CSE
        std::memcpy(ls_fe_in, ls_fe_out, 32);
    }, 200);

    // Generator * k  (same N_KEYGEN, same bench_ns -> H.run)
    idx = 0;
    const double ls_gen = bench_ns([&]() {
        secp256k1_pubkey pk;
        (void)secp256k1_ec_pubkey_create(ls_ctx, &pk, ls_seckeys[idx % POOL]);
        bench::DoNotOptimize(pk); ++idx;
    }, N_KEYGEN);

    // ECDSA Sign
    idx = 0;
    const double ls_ecdsa_sign = bench_ns([&]() {
        secp256k1_ecdsa_signature sig;
        secp256k1_ecdsa_sign(ls_ctx, &sig, ls_msgs[idx % POOL],
                             ls_seckeys[idx % POOL], NULL, NULL);
        bench::DoNotOptimize(sig); ++idx;
    }, N_SIGN);

    // ECDSA Verify
    idx = 0;
    const double ls_ecdsa_verify = bench_ns([&]() {
        volatile int ok = secp256k1_ecdsa_verify(ls_ctx, &ls_esigs[idx % POOL],
                                                 ls_msgs[idx % POOL],
                                                 &ls_pubkeys[idx % POOL]);
        (void)ok; ++idx;
    }, N_VERIFY);

#ifdef SECP256K1_BUILD_ETHEREUM
    // ECDSA Sign Recoverable (libsecp)
    secp256k1_ecdsa_recoverable_signature ls_rec_sigs[POOL];
    idx = 0;
    const double ls_sign_rec = bench_ns([&]() {
        secp256k1_ecdsa_sign_recoverable(ls_ctx, &ls_rec_sigs[idx % POOL],
                                         ls_msgs[idx % POOL],
                                         ls_seckeys[idx % POOL],
                                         NULL, NULL);
        bench::DoNotOptimize(ls_rec_sigs[idx % POOL]); ++idx;
    }, N_SIGN);

    // Pre-sign all recovery sigs for the recover benchmark
    for (int i = 0; i < POOL; ++i) {
        secp256k1_ecdsa_sign_recoverable(ls_ctx, &ls_rec_sigs[i],
                                         ls_msgs[i], ls_seckeys[i],
                                         NULL, NULL);
    }

    // ECDSA Recover (libsecp)
    idx = 0;
    const double ls_recover = bench_ns([&]() {
        secp256k1_pubkey pk;
        (void)secp256k1_ecdsa_recover(ls_ctx, &pk, &ls_rec_sigs[idx % POOL],
                                      ls_msgs[idx % POOL]);
        bench::DoNotOptimize(pk); ++idx;
    }, N_VERIFY);
#else
    const double ls_sign_rec = 0.0;
    const double ls_recover  = 0.0;
#endif // SECP256K1_BUILD_ETHEREUM

    // Schnorr Keypair Create
    idx = 0;
    const double ls_schnorr_kp = bench_ns([&]() {
        secp256k1_keypair kp;
        (void)secp256k1_keypair_create(ls_ctx, &kp, ls_seckeys[idx % POOL]);
        bench::DoNotOptimize(kp); ++idx;
    }, N_KEYGEN);

    // Schnorr Sign (BIP-340)
    idx = 0;
    const double ls_schnorr_sign = bench_ns([&]() {
        unsigned char sig64[64];
        secp256k1_schnorrsig_sign32(ls_ctx, sig64, ls_msgs[idx % POOL],
                                    &ls_keypairs[idx % POOL],
                                    ls_aux[idx % POOL]);
        bench::DoNotOptimize(sig64); ++idx;
    }, N_SIGN);

    // Schnorr Verify (BIP-340)
    idx = 0;
    const double ls_schnorr_verify = bench_ns([&]() {
        volatile int ok = secp256k1_schnorrsig_verify(
            ls_ctx, ls_schnorr_sigs[idx % POOL],
            ls_msgs[idx % POOL], 32,
            &ls_xonly[idx % POOL]);
        (void)ok; ++idx;
    }, N_VERIFY);

    // k*P (arbitrary-point scalar multiply) -- BIP-352 bottleneck
    idx = 0;
    const double ls_kP = bench_ns([&]() {
        secp256k1_pubkey pk_copy = ls_pubkeys[idx % POOL];
        (void)secp256k1_ec_pubkey_tweak_mul(ls_ctx, &pk_copy,
                                             ls_seckeys[(idx + 1) % POOL]);
        bench::DoNotOptimize(pk_copy); ++idx;
    }, N_SCALAR);

    // Serialization: ec_pubkey_serialize compressed (33 bytes)
    // libsecp stores affine internally -> serialization = byte copy (~15 ns)
    idx = 0;
    const double ls_serialize_comp = bench_ns([&]() {
        unsigned char out33[33];
        size_t outlen = 33;
        secp256k1_ec_pubkey_serialize(ls_ctx, out33, &outlen,
                                     &ls_pubkeys[idx % POOL],
                                     SECP256K1_EC_COMPRESSED);
        bench::DoNotOptimize(out33); ++idx;
    }, N_POINT);

    // Serialization: ec_pubkey_serialize uncompressed (65 bytes)
    idx = 0;
    const double ls_serialize_uncomp = bench_ns([&]() {
        unsigned char out65[65];
        size_t outlen = 65;
        secp256k1_ec_pubkey_serialize(ls_ctx, out65, &outlen,
                                     &ls_pubkeys[idx % POOL],
                                     SECP256K1_EC_UNCOMPRESSED);
        bench::DoNotOptimize(out65); ++idx;
    }, N_POINT);

    // Point addition: ec_pubkey_combine (2 pubkeys)
    idx = 0;
    const double ls_point_add = bench_ns([&]() {
        secp256k1_pubkey result;
        const secp256k1_pubkey* ins[2] = {
            &ls_pubkeys[idx % POOL],
            &ls_pubkeys[(idx + 1) % POOL]
        };
        (void)secp256k1_ec_pubkey_combine(ls_ctx, &result, ins, 2);
        bench::DoNotOptimize(result); ++idx;
    }, N_POINT);
#pragma GCC diagnostic pop

    // -----------------------------------------------------------------
    //  libsecp MICRO-BENCHMARKS: internal field/scalar/point primitives
    // -----------------------------------------------------------------
    // Uses opaque byte buffers sized >=  actual struct sizes.
    // secp256k1_fe = 40B, secp256k1_scalar = 32B,
    // secp256k1_ge = 88B, secp256k1_gej = 128B (on 5x52/4x64 builds)
    // Over-allocate to 256B each for safety/alignment.

    alignas(64) unsigned char ls_raw_fe_a[256], ls_raw_fe_b[256], ls_raw_fe_r[256];
    alignas(64) unsigned char ls_raw_sc_a[256], ls_raw_sc_b[256], ls_raw_sc_r[256];
    alignas(64) unsigned char ls_raw_ge_a[256], ls_raw_ge_b[256];
    alignas(64) unsigned char ls_raw_gej_a[256], ls_raw_gej_r[256];

    // Initialise field elements from known bytes
    {
        static const unsigned char gx[32] = {
            0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,
            0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,0x07,
            0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,
            0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98};
        static const unsigned char gy[32] = {
            0x48,0x3a,0xda,0x77,0x26,0xa3,0xc4,0x65,
            0x5d,0xa4,0xfb,0xfc,0x0e,0x11,0x08,0xa8,
            0xfd,0x17,0xb4,0x48,0xa6,0x85,0x54,0x19,
            0x9c,0x47,0xd0,0x8f,0xfb,0x10,0xd4,0xb8};
        libsecp_fe_set_b32(ls_raw_fe_a, gx);
        libsecp_fe_set_b32(ls_raw_fe_b, gy);
        std::memcpy(ls_raw_fe_r, ls_raw_fe_a, 256);

        int ov = 0;
        libsecp_scalar_set_b32(ls_raw_sc_a, gx, &ov);
        libsecp_scalar_set_b32(ls_raw_sc_b, gy, &ov);
        std::memcpy(ls_raw_sc_r, ls_raw_sc_a, 256);

        // Load a pubkey into affine ge, then convert to Jacobian gej
        libsecp_pubkey_load(ls_ctx, ls_raw_ge_a, &ls_pubkeys[0]);
        libsecp_pubkey_load(ls_ctx, ls_raw_ge_b, &ls_pubkeys[1]);
        libsecp_gej_set_ge(ls_raw_gej_a, ls_raw_ge_a);
    }

    // -- Field arithmetic --
    const double ls_fe_mul = bench_ns([&]() {
        libsecp_fe_mul(ls_raw_fe_r, ls_raw_fe_a, ls_raw_fe_b);
        bench::DoNotOptimize(ls_raw_fe_r);
    }, N_FIELD);

    const double ls_fe_sqr = bench_ns([&]() {
        libsecp_fe_sqr(ls_raw_fe_r, ls_raw_fe_a);
        bench::DoNotOptimize(ls_raw_fe_r);
    }, N_FIELD);

    const double ls_fe_add = bench_ns([&]() {
        // fe_add is in-place: r += a. Copy first to avoid accumulation.
        std::memcpy(ls_raw_fe_r, ls_raw_fe_a, 64);
        libsecp_fe_add(ls_raw_fe_r, ls_raw_fe_b);
        bench::DoNotOptimize(ls_raw_fe_r);
    }, N_FIELD);

    const double ls_fe_neg = bench_ns([&]() {
        libsecp_fe_negate(ls_raw_fe_r, ls_raw_fe_a, 1);
        bench::DoNotOptimize(ls_raw_fe_r);
    }, N_FIELD);

    // -- Field normalize (full reduction to canonical form) --
    // Fair test: normalize a magnitude-2 input (fe_a + fe_b), matching
    // Ultra's micro-diagnostic which also normalizes fe52_a + fe52_b.
    // A magnitude-1 (set_b32) input is trivially easy to normalize since
    // there's no overflow to fold -- that inflates libsecp's score unfairly.
    alignas(64) unsigned char ls_norm_input[256];
    std::memcpy(ls_norm_input, ls_raw_fe_a, 64);
    libsecp_fe_add(ls_norm_input, ls_raw_fe_b);  // magnitude 2
    const double ls_fe_norm = bench_ns([&]() {
        std::memcpy(ls_raw_fe_r, ls_norm_input, 64);
        libsecp_fe_normalize(ls_raw_fe_r);
        bench::DoNotOptimize(ls_raw_fe_r);
    }, N_FIELD);

    // -- Field from bytes (parse 32B -> fe) --
    const double ls_fe_from_bytes = bench_ns([&]() {
        static const unsigned char gx32[32] = {
            0x79,0xbe,0x66,0x7e,0xf9,0xdc,0xbb,0xac,
            0x55,0xa0,0x62,0x95,0xce,0x87,0x0b,0x07,
            0x02,0x9b,0xfc,0xdb,0x2d,0xce,0x28,0xd9,
            0x59,0xf2,0x81,0x5b,0x16,0xf8,0x17,0x98};
        libsecp_fe_set_b32(ls_raw_fe_r, gx32);
        bench::DoNotOptimize(ls_raw_fe_r);
    }, N_FIELD);

    // -- Scalar arithmetic --
    const double ls_sc_mul = bench_ns([&]() {
        libsecp_scalar_mul(ls_raw_sc_r, ls_raw_sc_a, ls_raw_sc_b);
        bench::DoNotOptimize(ls_raw_sc_r);
    }, N_FIELD);

    const double ls_sc_inv = bench_ns([&]() {
        libsecp_scalar_inverse(ls_raw_sc_r, ls_raw_sc_a);
        bench::DoNotOptimize(ls_raw_sc_r);
    }, 200);

    const double ls_sc_inv_var = bench_ns([&]() {
        libsecp_scalar_inverse_var(ls_raw_sc_r, ls_raw_sc_a);
        bench::DoNotOptimize(ls_raw_sc_r);
    }, 200);

    const double ls_sc_add = bench_ns([&]() {
        libsecp_scalar_add(ls_raw_sc_r, ls_raw_sc_a, ls_raw_sc_b);
        bench::DoNotOptimize(ls_raw_sc_r);
    }, N_FIELD);

    const double ls_sc_neg = bench_ns([&]() {
        libsecp_scalar_negate(ls_raw_sc_r, ls_raw_sc_a);
        bench::DoNotOptimize(ls_raw_sc_r);
    }, N_FIELD);

    // -- Scalar from bytes (parse 32B -> scalar) --
    const double ls_sc_from_bytes = bench_ns([&]() {
        int ov = 0;
        libsecp_scalar_set_b32(ls_raw_sc_r, ls_seckeys[0], &ov);
        bench::DoNotOptimize(ls_raw_sc_r);
    }, N_FIELD);

    // -- Point arithmetic --
    // Split I/O: r != a, matching how ecmult internally calls these.
    // Both Ultra and libsecp measure the same way for apple-to-apple.
    const double ls_pt_dbl = bench_ns([&]() {
        libsecp_gej_double_var(ls_raw_gej_r, ls_raw_gej_a);
        bench::DoNotOptimize(ls_raw_gej_r);
    }, N_POINT);

    const double ls_pt_add_ge = bench_ns([&]() {
        libsecp_gej_add_ge_var(ls_raw_gej_r, ls_raw_gej_a, ls_raw_ge_b);
        bench::DoNotOptimize(ls_raw_gej_r);
    }, N_POINT);

    // -- ecmult: a*P + b*G (Strauss dual mul -- verify core) --
    // Pre-parse inputs to measure pure ecmult, not parsing overhead.
    unsigned char ls_ecmult_gej[POOL][256];
    unsigned char ls_ecmult_sca[POOL][256];
    unsigned char ls_ecmult_scb[POOL][256];
    for (int pi = 0; pi < POOL; pi++) {
        unsigned char ge_tmp[256];
        libsecp_pubkey_load(ls_ctx, ge_tmp, &ls_pubkeys[pi]);
        libsecp_gej_set_ge(ls_ecmult_gej[pi], ge_tmp);
        int ov = 0;
        libsecp_scalar_set_b32(ls_ecmult_sca[pi], ls_seckeys[pi], &ov);
        libsecp_scalar_set_b32(ls_ecmult_scb[pi], ls_seckeys[(pi + 1) % POOL], &ov);
    }
    idx = 0;
    const double ls_ecmult = bench_ns([&]() {
        unsigned char gej_out[256];
        libsecp_ecmult(gej_out, ls_ecmult_gej[idx % POOL],
                       ls_ecmult_sca[idx % POOL], ls_ecmult_scb[idx % POOL]);
        bench::DoNotOptimize(gej_out); ++idx;
    }, N_SCALAR);

    // -- ecmult_gen: k*G (comb table generator mul) --
    const void* ls_ecmult_gen_ctx = libsecp_get_ecmult_gen_ctx(ls_ctx);
    idx = 0;
    const double ls_ecmult_gen = bench_ns([&]() {
        unsigned char sc_k[256], gej_out[256];
        int ov = 0;
        libsecp_scalar_set_b32(sc_k, ls_seckeys[idx % POOL], &ov);
        libsecp_ecmult_gen(ls_ecmult_gen_ctx, gej_out, sc_k);
        bench::DoNotOptimize(gej_out); ++idx;
    }, N_KEYGEN);

    secp256k1_context_destroy(ls_ctx);

    print_header("libsecp256k1 (bitcoin-core)");
    print_row("field_mul",                        ls_fe_mul);
    print_row("field_sqr",                        ls_fe_sqr);
    print_row("field_inv_var",                    ls_fe_inv);
    print_row("field_add",                        ls_fe_add);
    print_row("field_negate",                     ls_fe_neg);
    print_row("field_normalize",                   ls_fe_norm);
    print_row("field_from_bytes (set_b32)",        ls_fe_from_bytes);
    print_row("scalar_mul",                       ls_sc_mul);
    print_row("scalar_inverse (CT)",              ls_sc_inv);
    print_row("scalar_inverse_var",               ls_sc_inv_var);
    print_row("scalar_add",                       ls_sc_add);
    print_row("scalar_negate",                    ls_sc_neg);
    print_row("scalar_from_bytes (set_b32)",       ls_sc_from_bytes);
    print_row("point_dbl (gej_double_var)",       ls_pt_dbl);
    print_row("point_add (gej_add_ge_var)",       ls_pt_add_ge);
    print_row("ecmult (a*P + b*G, Strauss)",      ls_ecmult);
    print_row("ecmult_gen (k*G, comb)",           ls_ecmult_gen);
    print_row("generator_mul (ec_pubkey_create)", ls_gen);
    print_row("scalar_mul_P (k*P, tweak_mul)",    ls_kP);
    print_row("serialize_compressed (33B)",       ls_serialize_comp);
    print_row("serialize_uncompressed (65B)",     ls_serialize_uncomp);
    print_row("point_add (pubkey_combine)",       ls_point_add);
    print_row("ecdsa_sign",                       ls_ecdsa_sign);
    print_row("ecdsa_verify",                     ls_ecdsa_verify);
    print_row("schnorr_keypair_create",           ls_schnorr_kp);
    print_row("schnorr_sign (BIP-340)",           ls_schnorr_sign);
    print_row("schnorr_verify (BIP-340)",         ls_schnorr_verify);
    print_sep();
    printf("\n");

    // =====================================================================
    //  SECTION 7.5: OpenSSL (system library) -- SAME harness, pool, timer
    // =====================================================================
    //
    // OpenSSL provides ECDSA on secp256k1 but NOT BIP-340 Schnorr.
    // Uses low-level EC API (ECDSA_do_sign/verify) for raw performance.
    // Pre-allocates scratch (BN_CTX, EC_POINT, BIGNUM) to isolate crypto cost.
    //

    double ossl_gen = 0.0, ossl_ecdsa_sign = 0.0, ossl_ecdsa_verify = 0.0;

#ifdef BENCH_HAS_OPENSSL
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

    printf("Running OpenSSL benchmark (%s, same harness)...\n",
           OPENSSL_VERSION_TEXT);

    BN_CTX *ossl_bn_ctx = BN_CTX_new();
    EC_KEY *ossl_keys[POOL];
    ECDSA_SIG *ossl_esigs[POOL];
    const EC_GROUP *ossl_group = NULL;

    // Set up pool -- same 64 seeds as Ultra and libsecp
    {
        bool ossl_ok = true;
        for (int i = 0; i < POOL; ++i) {
            ossl_keys[i] = EC_KEY_new_by_curve_name(NID_secp256k1);
            if (!ossl_keys[i]) {
                printf("[FAIL] OpenSSL: EC_KEY_new_by_curve_name(NID_secp256k1) "
                       "returned NULL -- secp256k1 not supported?\n");
                ossl_ok = false;
                break;
            }
            if (i == 0) ossl_group = EC_KEY_get0_group(ossl_keys[i]);

            // Private key from same seed as libsecp pool
            BIGNUM *priv = BN_new();
            BN_bin2bn(ls_seckeys[i], 32, priv);
            EC_KEY_set_private_key(ossl_keys[i], priv);

            // Compute public key = priv * G
            EC_POINT *pub = EC_POINT_new(ossl_group);
            EC_POINT_mul(ossl_group, pub, priv, NULL, NULL, ossl_bn_ctx);
            EC_KEY_set_public_key(ossl_keys[i], pub);
            EC_POINT_free(pub);
            BN_free(priv);

            // Pre-sign for verify benchmark (same message bytes as libsecp)
            ossl_esigs[i] = ECDSA_do_sign(ls_msgs[i], 32, ossl_keys[i]);
            if (!ossl_esigs[i]) {
                printf("[FAIL] OpenSSL: ECDSA_do_sign failed for pool[%d]\n", i);
                ossl_ok = false;
                break;
            }
        }

        if (ossl_ok) {
            // Generator * k  (pre-allocate scratch to isolate crypto cost)
            EC_POINT *ossl_R = EC_POINT_new(ossl_group);
            BIGNUM *ossl_bn_k = BN_new();

            idx = 0;
            ossl_gen = bench_ns([&]() {
                BN_bin2bn(ls_seckeys[idx % POOL], 32, ossl_bn_k);
                EC_POINT_mul(ossl_group, ossl_R, ossl_bn_k,
                             NULL, NULL, ossl_bn_ctx);
                bench::DoNotOptimize(ossl_R);
                ++idx;
            }, N_KEYGEN);

            EC_POINT_free(ossl_R);
            BN_free(ossl_bn_k);

            // ECDSA Sign
            idx = 0;
            ossl_ecdsa_sign = bench_ns([&]() {
                ECDSA_SIG *sig = ECDSA_do_sign(ls_msgs[idx % POOL], 32,
                                               ossl_keys[idx % POOL]);
                bench::DoNotOptimize(sig);
                ECDSA_SIG_free(sig);
                ++idx;
            }, N_SIGN);

            // ECDSA Verify
            idx = 0;
            ossl_ecdsa_verify = bench_ns([&]() {
                volatile int ok = ECDSA_do_verify(
                    ls_msgs[idx % POOL], 32,
                    ossl_esigs[idx % POOL],
                    ossl_keys[idx % POOL]);
                (void)ok;
                ++idx;
            }, N_VERIFY);

            print_header("OpenSSL (ECDSA, secp256k1)");
            print_row("generator_mul (EC_POINT_mul k*G)", ossl_gen);
            print_row("ecdsa_sign (ECDSA_do_sign)",       ossl_ecdsa_sign);
            print_row("ecdsa_verify (ECDSA_do_verify)",   ossl_ecdsa_verify);
            print_sep();
            printf("  (OpenSSL has no BIP-340 Schnorr -- ECDSA-only comparison)\n");
            printf("\n");
        }

        // Cleanup
        for (int i = 0; i < POOL; ++i) {
            if (ossl_esigs[i]) ECDSA_SIG_free(ossl_esigs[i]);
            if (ossl_keys[i])  EC_KEY_free(ossl_keys[i]);
        }
        BN_CTX_free(ossl_bn_ctx);
    }

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
#else
    printf("OpenSSL: not linked (rebuild with -DBENCH_HAS_OPENSSL or install libssl-dev)\n\n");
#endif // BENCH_HAS_OPENSSL

    // =====================================================================
    //  SECTION 8: Apple-to-Apple Detailed Head-to-Head
    // =====================================================================

    // Note: libsecp256k1 is ALWAYS constant-time for signing.
    // "FAST" comparison uses Ultra FAST path (unfair for signing/keygen).
    // "CT-vs-CT" shows the true apples-to-apples for signing ops.
    // Verify uses only public data -- no CT needed, same in both paths.

    printf("======================================================================\n");
    printf("  HEAD-TO-HEAD: UltrafastSecp256k1 vs libsecp256k1\n");
    printf("  (ratio > 1.0 = Ultra wins, < 1.0 = libsecp wins)\n");
    printf("======================================================================\n\n");

    // ---- Field Arithmetic ----
    print_header_3col("FIELD ARITHMETIC");
    print_row_3col("mul",               fmul,           ls_fe_mul);
    print_row_3col("sqr",               fsqr,           ls_fe_sqr);
    print_row_3col("inv",               finv,           ls_fe_inv);
    print_row_3col("add",               fadd,           ls_fe_add);
    print_row_3col("sub",               fsub,           0);
    print_row_3col("negate",            fneg,           ls_fe_neg);
#if defined(SECP256K1_FAST_52BIT)
    print_row_3col("normalize (FE52)",  micro_fe52_norm_val, ls_fe_norm);
#endif
    print_row_3col("from_bytes (32B)",  fe_from_bytes,  ls_fe_from_bytes);
#if defined(SECP256K1_FAST_52BIT)
    if (micro_fe52_add > 0) {
        print_row_3col("FE52 add (hot path)",  micro_fe52_add, ls_fe_add);
        print_row_3col("FE52 neg (hot path)",  micro_fe52_neg, ls_fe_neg);
    }
#endif
    print_sep_3col();
    printf("\n");

    // ---- Scalar Arithmetic ----
    print_header_3col("SCALAR ARITHMETIC");
    print_row_3col("mul",               smul,           ls_sc_mul);
    print_row_3col("inv (CT)",          micro_scalar_inv, ls_sc_inv);
    print_row_3col("inv (var-time)",    micro_scalar_inv, ls_sc_inv_var);
    print_row_3col("add",               sadd,           ls_sc_add);
    print_row_3col("negate",            sneg,           ls_sc_neg);
    print_row_3col("from_bytes (32B)",  sc_from_bytes,  ls_sc_from_bytes);
    print_sep_3col();
    printf("\n");

    // ---- Point Arithmetic ----
    print_header_3col("POINT ARITHMETIC");
    print_row_3col("dbl (Jacobian)",     ptdbl,         ls_pt_dbl);
    print_row_3col("add (mixed J+A)",    ptadd_mixed,   ls_pt_add_ge);
    print_row_3col("ecmult (a*P+b*G)",   dualmul,       ls_ecmult);
    print_row_3col("ecmult_gen (k*G raw)",keygen,        ls_ecmult_gen);
    print_row_3col("pubkey_create (API)", keygen,        ls_gen);
    print_row_3col("scalar_mul (k*P)",    scalarmul,     ls_kP);
    print_row_3col("scalar_mul (KPlan)",  plan_mul,      ls_kP);
    print_row_3col("point_add (combine)", ptadd,         ls_point_add);
    print_sep_3col();
    printf("\n");

    // ---- Serialization ----
    print_header_3col("SERIALIZATION");
    print_row_3col("compressed (33B)",    u_to_compressed,   ls_serialize_comp);
    print_row_3col("uncompressed (65B)",  u_to_uncompressed, ls_serialize_uncomp);
    print_sep_3col();
    printf("\n");

    // ---- High-Level Operations (FAST path) — diagnostic only, NOT security-equivalent ----
    print_header_3col("SIGNING (FAST non-CT vs libsecp CT) [diagnostic only — not production-equivalent]");
    print_row_3col("ECDSA Sign",          u_ecdsa_sign,     ls_ecdsa_sign);
    print_row_3col("Schnorr Sign",        u_schnorr_sign,   ls_schnorr_sign);
    print_row_3col("Schnorr Keypair",     u_schnorr_kp,     ls_schnorr_kp);
    print_sep_3col();
    printf("\n");

    print_header_3col("VERIFICATION");
    print_row_3col("ECDSA Verify",              u_ecdsa_verify,       ls_ecdsa_verify);
    print_row_3col("Schnorr Verify (cached)",   u_schnorr_verify,     ls_schnorr_verify);
    print_row_3col("Schnorr Verify (raw)",      u_schnorr_verify_raw, ls_schnorr_verify);
    print_sep_3col();
    printf("\n");

    // CT-vs-CT: libsecp256k1 sign is always CT, so compare with Ultra CT sign.
    // Verify doesn't change (no secret data), so same numbers as FAST.
    print_header_3col("CT-vs-CT (fair signing)");
    print_row_3col("ECDSA Sign",          u_ct_ecdsa,         ls_ecdsa_sign);
    print_row_3col("Schnorr Sign",        u_ct_schnorr,       ls_schnorr_sign);
    print_row_3col("ECDSA Verify",        u_ecdsa_verify,     ls_ecdsa_verify);
    print_row_3col("Schnorr Verify",      u_schnorr_verify_raw, ls_schnorr_verify);
    print_sep_3col();
    printf("\n");

#ifdef SECP256K1_BUILD_ETHEREUM
    // ---- Ethereum / Recovery Operations ----
    print_header_3col("ETHEREUM / RECOVERY");
    print_row_3col("sign_recoverable",    u_sign_rec,         ls_sign_rec);
    print_row_3col("ecrecover",           u_ecrecover,        ls_recover);
    print_row_3col("eth_sign_hash",       u_eth_sign,         ls_sign_rec);
    print_row_3col("eth_personal_sign",   u_personal_sign,    ls_sign_rec);
    print_sep_3col();
    printf("\n");
#endif

    // --- OpenSSL ratios (only if measured) ---
#ifdef BENCH_HAS_OPENSSL
    if (ossl_gen > 0.0) {
        printf("======================================================================\n");
        printf("  APPLE-TO-APPLE: UltrafastSecp256k1 / OpenSSL\n");
        printf("  (ratio > 1.0 = Ultra wins, < 1.0 = OpenSSL wins)\n");
        printf("======================================================================\n\n");

        print_header_ratio("FAST path (Ultra FAST vs OpenSSL)");
        print_ratio("Generator * k",   ossl_gen          / keygen);
        print_ratio("ECDSA Sign",      ossl_ecdsa_sign   / u_ecdsa_sign);
        print_ratio("ECDSA Verify",    ossl_ecdsa_verify / u_ecdsa_verify);
        print_sep();
        printf("\n");

        // OpenSSL ECDSA sign is constant-time (modern versions) --
        // compare with Ultra CT sign for fair assessment.
        print_header_ratio("CT path (Ultra CT vs OpenSSL)");
        print_ratio("ECDSA Sign (CT vs CT)",    ossl_ecdsa_sign   / u_ct_ecdsa);
        print_ratio("ECDSA Verify",             ossl_ecdsa_verify / u_ecdsa_verify);
        print_sep();
        printf("\n");
    }
#endif

    // =====================================================================
    //  SECTION 8.5: ZK Proofs & Commitments
    // =====================================================================
    double u_pedersen = 0, u_kp_prove = 0, u_kp_verify = 0;
    double u_dleq_prove = 0, u_dleq_verify = 0;
    double u_range_prove = 0, u_range_verify = 0;
    {
        using namespace secp256k1::zk;

        auto sk_zk = make_scalar(42);
        auto pk_zk = Point::generator().scalar_mul(sk_zk);
        auto blind = make_scalar(99);
        auto val_scalar = Scalar::from_uint64(12345);
        auto commit = secp256k1::pedersen_commit(val_scalar, blind);
        std::array<uint8_t, 32> aux_zk{};
        aux_zk[0] = 0xAA;

        // Pedersen commit
        u_pedersen = bench_ns([&]{
            auto c = secp256k1::pedersen_commit(val_scalar, blind);
            bench::DoNotOptimize(c);
        }, N_SIGN);

        // Knowledge proof (Schnorr sigma protocol)
        u_kp_prove = bench_ns([&]{
            auto kp = knowledge_prove(sk_zk, pk_zk, {}, aux_zk);
            bench::DoNotOptimize(kp);
        }, N_SIGN);

        auto kp = knowledge_prove(sk_zk, pk_zk, {}, aux_zk);
        u_kp_verify = bench_ns([&]{
            bool ok = knowledge_verify(kp, pk_zk, {});
            bench::DoNotOptimize(ok);
        }, N_SIGN);

        // DLEQ proof
        auto sk2_zk = make_scalar(87654321);
        auto H_zk = Point::generator().scalar_mul(sk2_zk);
        auto P_zk = Point::generator().scalar_mul(sk_zk);
        auto Q_zk = H_zk.scalar_mul(sk_zk);

        u_dleq_prove = bench_ns([&]{
            auto dp = dleq_prove(sk_zk, Point::generator(), H_zk, P_zk, Q_zk, aux_zk);
            bench::DoNotOptimize(dp);
        }, N_SIGN / 2);

        auto dp = dleq_prove(sk_zk, Point::generator(), H_zk, P_zk, Q_zk, aux_zk);
        u_dleq_verify = bench_ns([&]{
            bool ok = dleq_verify(dp, Point::generator(), H_zk, P_zk, Q_zk);
            bench::DoNotOptimize(ok);
        }, N_SIGN / 2);

        // Bulletproof range proof (64-bit)
        std::uint64_t val = 12345;
        auto rp = range_prove(val, blind, commit, aux_zk);
        u_range_prove = bench_ns([&]{
            auto p = range_prove(val, blind, commit, aux_zk);
            bench::DoNotOptimize(p);
        }, std::max(1, N_SIGN / 64));

        u_range_verify = bench_ns([&]{
            bool ok = range_verify(commit, rp);
            bench::DoNotOptimize(ok);
        }, std::max(1, N_SIGN / 32));

        print_header("ZK Proofs & Commitments");
        print_row("Pedersen commit",              u_pedersen);
        print_row("Knowledge prove (sigma)",      u_kp_prove);
        print_row("Knowledge verify",             u_kp_verify);
        print_row("DLEQ prove",                   u_dleq_prove);
        print_row("DLEQ verify",                  u_dleq_verify);
        print_row("Bulletproof range_prove (64b)", u_range_prove);
        print_row("Bulletproof range_verify (64b)",u_range_verify);
        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 8.6: Adaptor Signatures
    // =====================================================================

    double u_schnorr_adaptor_sign = 0, u_schnorr_adaptor_verify = 0;
    double u_schnorr_adaptor_adapt = 0, u_schnorr_adaptor_extract = 0;
    double u_ecdsa_adaptor_sign = 0, u_ecdsa_adaptor_verify = 0;
    {
        // adaptor secret t -> adaptor point T = t*G
        auto adaptor_secret = make_scalar(0xADAD0001ULL);
        auto adaptor_point = Point::generator().scalar_mul(adaptor_secret);

        idx = 0;
        u_schnorr_adaptor_sign = bench_ns([&]{
            auto pre = schnorr_adaptor_sign(privkeys[idx % POOL], msghashes[idx % POOL],
                                            adaptor_point, aux_rands[idx % POOL]);
            bench::DoNotOptimize(pre); ++idx;
        }, N_SIGN);

        auto pre_schnorr = schnorr_adaptor_sign(privkeys[0], msghashes[0],
                                                adaptor_point, aux_rands[0]);
        u_schnorr_adaptor_verify = bench_ns([&]{
            bool ok = schnorr_adaptor_verify(pre_schnorr, schnorr_pubkeys_x[0],
                                             msghashes[0], adaptor_point);
            bench::DoNotOptimize(ok);
        }, N_VERIFY);

        auto final_schnorr = schnorr_adaptor_adapt(pre_schnorr, adaptor_secret);
        u_schnorr_adaptor_adapt = bench_ns([&]{
            auto sig = schnorr_adaptor_adapt(pre_schnorr, adaptor_secret);
            bench::DoNotOptimize(sig);
        }, N_SIGN);

        u_schnorr_adaptor_extract = bench_ns([&]{
            auto [t, ok] = schnorr_adaptor_extract(pre_schnorr, final_schnorr);
            bench::DoNotOptimize(t);
        }, N_SIGN);

        idx = 0;
        u_ecdsa_adaptor_sign = bench_ns([&]{
            auto pre = ecdsa_adaptor_sign(privkeys[idx % POOL], msghashes[idx % POOL],
                                          adaptor_point);
            bench::DoNotOptimize(pre); ++idx;
        }, N_SIGN);

        auto pre_ecdsa = ecdsa_adaptor_sign(privkeys[0], msghashes[0], adaptor_point);
        u_ecdsa_adaptor_verify = bench_ns([&]{
            bool ok = ecdsa_adaptor_verify(pre_ecdsa, pubkeys[0],
                                           msghashes[0], adaptor_point);
            bench::DoNotOptimize(ok);
        }, N_VERIFY);

        print_header("ADAPTOR SIGNATURES");
        print_row("Schnorr adaptor sign",         u_schnorr_adaptor_sign);
        print_row("Schnorr adaptor verify",        u_schnorr_adaptor_verify);
        print_row("Schnorr adaptor adapt",         u_schnorr_adaptor_adapt);
        print_row("Schnorr adaptor extract secret",u_schnorr_adaptor_extract);
        print_row("ECDSA adaptor sign",            u_ecdsa_adaptor_sign);
        print_row("ECDSA adaptor verify",          u_ecdsa_adaptor_verify);
        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 8.7: FROST Threshold Signatures (2-of-3)
    // =====================================================================

    double u_frost_keygen = 0, u_frost_nonce_gen = 0;
    double u_frost_sign_partial = 0, u_frost_verify_partial = 0;
    double u_frost_aggregate = 0;
    {
        // Setup: 2-of-3 DKG
        std::array<std::uint8_t, 32> seeds[3];
        for (int i = 0; i < 3; ++i) seeds[i] = make_hash(0xF2057000ULL + static_cast<uint64_t>(i));

        auto [c1, s1] = frost_keygen_begin(1, 2, 3, seeds[0]);
        auto [c2, s2] = frost_keygen_begin(2, 2, 3, seeds[1]);
        auto [c3, s3] = frost_keygen_begin(3, 2, 3, seeds[2]);

        std::vector<FrostCommitment> all_comms = {c1, c2, c3};

        // Each participant gets shares from others + own
        std::vector<FrostShare> shares_for_1, shares_for_2, shares_for_3;
        for (auto& s : s1) { if (s.id == 1) shares_for_1.push_back(s); else if (s.id == 2) shares_for_2.push_back(s); else shares_for_3.push_back(s); }
        for (auto& s : s2) { if (s.id == 1) shares_for_1.push_back(s); else if (s.id == 2) shares_for_2.push_back(s); else shares_for_3.push_back(s); }
        for (auto& s : s3) { if (s.id == 1) shares_for_1.push_back(s); else if (s.id == 2) shares_for_2.push_back(s); else shares_for_3.push_back(s); }

        auto [kp1, ok1] = frost_keygen_finalize(1, all_comms, shares_for_1, 2, 3);
        auto [kp2, ok2] = frost_keygen_finalize(2, all_comms, shares_for_2, 2, 3);
        (void)ok1; (void)ok2;

        // Nonce gen
        auto nseed1 = make_hash(0xA0ACE01ULL);
        auto nseed2 = make_hash(0xA0ACE02ULL);
        auto [nonce1, ncomm1] = frost_sign_nonce_gen(1, nseed1);
        auto [nonce2, ncomm2] = frost_sign_nonce_gen(2, nseed2);
        std::vector<FrostNonceCommitment> ncomms = {ncomm1, ncomm2};

        auto psig1 = frost_sign(kp1, nonce1, msghashes[0], ncomms);
        auto psig2 = frost_sign(kp2, nonce2, msghashes[0], ncomms);

        u_frost_keygen = bench_ns([&]{
            auto [c, shares] = frost_keygen_begin(1, 2, 3, seeds[0]);
            bench::DoNotOptimize(c);
        }, std::max(1, N_SIGN / 4));

        u_frost_nonce_gen = bench_ns([&]{
            auto [n, nc] = frost_sign_nonce_gen(1, nseed1);
            bench::DoNotOptimize(nc);
        }, N_SIGN);

        // Pre-generate a fresh nonce per iteration so the bench does not
        // reuse a consumed (zeroed) nonce — which would violate H-01 and
        // produce wrong timing due to hitting the zero-check paths.
        {
            const std::size_t frost_bench_calls =
                static_cast<std::size_t>(H.warmup_iters) +
                (H.passes * static_cast<std::size_t>(N_SIGN));
            std::vector<FrostNonce> bench_nonces(frost_bench_calls);
            for (std::size_t i = 0; i < frost_bench_calls; ++i) {
                auto tseed = make_hash(0xBEEF0000ULL + static_cast<std::uint64_t>(i));
                bench_nonces[i] = frost_sign_nonce_gen(1, tseed).first;
            }
            std::size_t bn_i = 0;
            u_frost_sign_partial = bench_ns([&]{
                auto ps = frost_sign(kp1, bench_nonces[bn_i++],
                                     msghashes[0], ncomms);
                bench::DoNotOptimize(ps);
            }, N_SIGN);
        }

        u_frost_verify_partial = bench_ns([&]{
            bool ok = frost_verify_partial(psig1, ncomm1, kp1.verification_share,
                                           msghashes[0], ncomms, kp1.group_public_key);
            bench::DoNotOptimize(ok);
        }, N_VERIFY);

        u_frost_aggregate = bench_ns([&]{
            auto sig = frost_aggregate({psig1, psig2}, ncomms,
                                       kp1.group_public_key, msghashes[0]);
            bench::DoNotOptimize(sig);
        }, N_SIGN);

        print_header("FROST THRESHOLD (2-of-3)");
        print_row("keygen_begin (DKG round 1)",     u_frost_keygen);
        print_row("nonce_gen",                       u_frost_nonce_gen);
        print_row("partial_sign",                    u_frost_sign_partial);
        print_row("partial_verify",                  u_frost_verify_partial);
        print_row("aggregate → Schnorr sig",        u_frost_aggregate);
        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 8.8: MuSig2 Multi-Signatures (2-of-2)
    // =====================================================================

    double u_musig2_key_agg = 0, u_musig2_nonce_gen = 0;
    double u_musig2_partial_sign = 0, u_musig2_partial_verify = 0;
    double u_musig2_sig_agg = 0;
    {
        // Setup: 2-of-2 (both must sign)
        auto pk1_x = schnorr_pubkeys_x[0];
        auto pk2_x = schnorr_pubkeys_x[1];
        // BIP-327 key_agg requires 33-byte compressed keys
        auto pk1_c = pubkeys[0].to_compressed();
        auto pk2_c = pubkeys[1].to_compressed();
        std::vector<std::array<std::uint8_t, 33>> pks = {pk1_c, pk2_c};

        auto key_agg = musig2_key_agg(pks);
        auto [snonce1, pnonce1] = musig2_nonce_gen(privkeys[0], pk1_x, key_agg.Q_x, msghashes[0]);
        auto [snonce2, pnonce2] = musig2_nonce_gen(privkeys[1], pk2_x, key_agg.Q_x, msghashes[0]);
        auto agg_nonce = musig2_nonce_agg({pnonce1, pnonce2});
        auto session = musig2_start_sign_session(agg_nonce, key_agg, msghashes[0]);
        auto ps1 = musig2_partial_sign(snonce1, privkeys[0], key_agg, session, 0);
        auto ps2 = musig2_partial_sign(snonce2, privkeys[1], key_agg, session, 1);

        u_musig2_key_agg = bench_ns([&]{
            auto ka = musig2_key_agg(pks);
            bench::DoNotOptimize(ka);
        }, N_SIGN);

        u_musig2_nonce_gen = bench_ns([&]{
            auto [sn, pn] = musig2_nonce_gen(privkeys[0], pk1_x, key_agg.Q_x, msghashes[0]);
            bench::DoNotOptimize(pn);
        }, N_SIGN);

        // Pre-generate a nonce pool for partial_sign bench (M-03 single-use)
        {
            const std::size_t musig2_bench_calls =
                static_cast<std::size_t>(H.warmup_iters) +
                (H.passes * static_cast<std::size_t>(N_SIGN));
            std::vector<MuSig2SecNonce> m2_nonce_pool(musig2_bench_calls);
            for (std::size_t i = 0; i < musig2_bench_calls; ++i) {
                auto tseed = make_hash(0xBEEFC000ULL + static_cast<std::uint64_t>(i));
                m2_nonce_pool[i] =
                    musig2_nonce_gen(privkeys[0], pk1_x, key_agg.Q_x, msghashes[0],
                                     tseed.data()).first;
            }
            std::size_t m2_i = 0;
            u_musig2_partial_sign = bench_ns([&]{
                auto s = musig2_partial_sign(m2_nonce_pool[m2_i++],
                                             privkeys[0], key_agg, session, 0);
                bench::DoNotOptimize(s);
            }, N_SIGN);
        }

        u_musig2_partial_verify = bench_ns([&]{
            bool ok = musig2_partial_verify(ps1, pnonce1, pk1_x, key_agg, session, 0);
            bench::DoNotOptimize(ok);
        }, N_VERIFY);

        u_musig2_sig_agg = bench_ns([&]{
            auto sig = musig2_partial_sig_agg({ps1, ps2}, session);
            bench::DoNotOptimize(sig);
        }, N_SIGN);

        print_header("MUSIG2 MULTI-SIGNATURES (2-of-2)");
        print_row("key_agg (BIP-327)",              u_musig2_key_agg);
        print_row("nonce_gen",                       u_musig2_nonce_gen);
        print_row("partial_sign",                    u_musig2_partial_sign);
        print_row("partial_verify",                  u_musig2_partial_verify);
        print_row("sig_agg → Schnorr sig",          u_musig2_sig_agg);
        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 8.9: ECIES Encryption
    // =====================================================================

    double u_ecies_encrypt = 0, u_ecies_decrypt = 0;
    {
        auto ecies_msg = std::vector<std::uint8_t>(256, 0x42);
        auto ecies_ct = ecies_encrypt(pubkeys[0], ecies_msg.data(), ecies_msg.size());

        u_ecies_encrypt = bench_ns([&]{
            auto ct = ecies_encrypt(pubkeys[0], ecies_msg.data(), ecies_msg.size());
            bench::DoNotOptimize(ct.data());
        }, N_SIGN);

        u_ecies_decrypt = bench_ns([&]{
            auto pt = ecies_decrypt(privkeys[0], ecies_ct.data(), ecies_ct.size());
            bench::DoNotOptimize(pt.data());
        }, N_SIGN);

        print_header("ECIES ENCRYPTION");
        print_row("ECIES encrypt (256B payload)",    u_ecies_encrypt);
        print_row("ECIES decrypt (256B payload)",    u_ecies_decrypt);
        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 8.10: Message Signing & Hashing
    // =====================================================================

    double u_btc_msg_sign = 0, u_btc_msg_verify = 0;
    double u_sha256_32 = 0, u_sha512_32 = 0;
    double u_msm_4 = 0, u_msm_64 = 0;
    {
        const std::uint8_t msg_text[] = "Hello, Bitcoin!";
        auto btc_sig = secp256k1::coins::bitcoin_sign_message(
            msg_text, sizeof(msg_text) - 1, privkeys[0]);

        idx = 0;
        u_btc_msg_sign = bench_ns([&]{
            auto sig = secp256k1::coins::bitcoin_sign_message(
                msg_text, sizeof(msg_text) - 1, privkeys[idx % POOL]);
            bench::DoNotOptimize(sig); ++idx;
        }, N_SIGN);

        u_btc_msg_verify = bench_ns([&]{
            bool ok = secp256k1::coins::bitcoin_verify_message(
                msg_text, sizeof(msg_text) - 1, pubkeys[0], btc_sig.sig);
            bench::DoNotOptimize(ok);
        }, N_VERIFY);

        // SHA-256 (32-byte input)
        u_sha256_32 = bench_ns([&]{
            auto h = SHA256::hash(msghashes[0].data(), 32);
            bench::DoNotOptimize(h);
        }, N_FIELD);

        // SHA-512 (32-byte input)
        u_sha512_32 = bench_ns([&]{
            auto h = SHA512::hash(msghashes[0].data(), 32);
            bench::DoNotOptimize(h);
        }, N_FIELD);

        // Multi-scalar multiplication (4-point)
        std::vector<Scalar> msm_scalars_4(4);
        std::vector<Point> msm_points_4(4);
        for (int i = 0; i < 4; ++i) {
            msm_scalars_4[static_cast<std::size_t>(i)] = privkeys[i];
            msm_points_4[static_cast<std::size_t>(i)] = pubkeys[i];
        }
        u_msm_4 = bench_ns([&]{
            auto r = multi_scalar_mul(msm_scalars_4, msm_points_4);
            bench::DoNotOptimize(r);
        }, N_POINT);

        // Multi-scalar multiplication (64-point, Pippenger)
        std::vector<Scalar> msm_scalars_64(64);
        std::vector<Point> msm_points_64(64);
        for (int i = 0; i < 64; ++i) {
            msm_scalars_64[static_cast<std::size_t>(i)] = privkeys[i % POOL];
            msm_points_64[static_cast<std::size_t>(i)] = pubkeys[i % POOL];
        }
        u_msm_64 = bench_ns([&]{
            auto r = multi_scalar_mul(msm_scalars_64, msm_points_64);
            bench::DoNotOptimize(r);
        }, std::max(1, N_POINT / 8));

        print_header("MESSAGE SIGNING & HASHING");
        print_row("Bitcoin message sign",            u_btc_msg_sign);
        print_row("Bitcoin message verify",          u_btc_msg_verify);
        print_row("SHA-256 (32B input)",             u_sha256_32);
        print_row("SHA-512 (32B input)",             u_sha512_32);
        print_row("Multi-scalar mul (4 points)",     u_msm_4);
        print_row("Multi-scalar mul (64 points)",    u_msm_64);
        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 8.11: BIP-39 Mnemonic
    // =====================================================================

    double u_bip39_gen12 = 0, u_bip39_gen24 = 0, u_bip39_validate = 0, u_bip39_to_seed = 0;
    {
        using namespace secp256k1;

        // Generate a reference mnemonic for validate/to_seed benchmarks
        auto [mnemonic12, ok12] = bip39_generate(16);  // 128-bit -> 12 words
        if (!ok12) { std::printf("[!] bip39_generate(16) failed\n"); return 1; }

        idx = 0;
        u_bip39_gen12 = bench_ns([&]{
            auto [m, ok] = bip39_generate(16);
            bench::DoNotOptimize(m.data()); ++idx;
        }, N_SIGN);

        u_bip39_gen24 = bench_ns([&]{
            auto [m, ok] = bip39_generate(32);
            bench::DoNotOptimize(m.data()); ++idx;
        }, N_SIGN);

        u_bip39_validate = bench_ns([&]{
            bool v = bip39_validate(mnemonic12);
            bench::DoNotOptimize(v);
        }, N_SIGN);

        u_bip39_to_seed = bench_ns([&]{
            auto [seed, ok] = bip39_mnemonic_to_seed(mnemonic12, "");
            bench::DoNotOptimize(seed.data());
        }, std::max(1, N_SIGN / 4));  // PBKDF2 is expensive

        print_header("BIP-39 MNEMONIC");
        print_row("bip39_generate (12 words)",         u_bip39_gen12);
        print_row("bip39_generate (24 words)",         u_bip39_gen24);
        print_row("bip39_validate (12 words)",         u_bip39_validate);
        print_row("bip39_to_seed (PBKDF2, 12 words)",  u_bip39_to_seed);
        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 8.12: BIP-141/143/144/342 — SegWit & Tapscript
    // =====================================================================

    double u_bip143_sighash = 0, u_bip143_script_code = 0;
    double u_bip144_wtxid = 0, u_bip144_commitment = 0, u_bip144_weight = 0;
    double u_segwit_parse = 0, u_segwit_p2wpkh_spk = 0, u_segwit_p2wsh_spk = 0;
    double u_tapscript_sighash = 0, u_keypath_sighash = 0;
    {
        using namespace secp256k1;

        // --- BIP-143: SegWit v0 sighash ---
        Outpoint op{};
        std::memset(op.txid.data(), 0xAB, 32);
        op.vout = 0;

        TxOutput txout;
        txout.value = 100000;
        txout.script_pubkey.resize(25, 0x76);

        std::array<uint8_t, 20> pkh{};
        std::memset(pkh.data(), 0xCC, 20);
        auto sc = bip143_p2wpkh_script_code(pkh.data()); // 25-byte P2WPKH script_code

        uint32_t seq = 0xFFFFFFFF;
        Bip143Preimage pi = bip143_build_preimage(
            2, &op, 1, &seq, &txout, 1, 0);

        u_bip143_sighash = bench_ns([&]{
            auto h = bip143_sighash(pi, op, sc.data(), sc.size(), 100000, seq,
                                     static_cast<uint32_t>(SighashType::ALL));
            bench::DoNotOptimize(h[0]);
        }, N_SIGN);

        u_bip143_script_code = bench_ns([&]{
            auto s = bip143_p2wpkh_script_code(pkh.data());
            bench::DoNotOptimize(s[0]);
        }, N_SIGN * 4);

        // --- BIP-144: Witness transaction ---
        WitnessTx wtx;
        wtx.version = 2;
        wtx.locktime = 0;
        TxInput tin;
        std::memset(tin.prev_txid.data(), 0xAB, 32);
        tin.prev_vout = 0;
        tin.sequence = 0xFFFFFFFF;
        wtx.inputs.push_back(tin);
        TxOut tout;
        tout.value = 99000;
        tout.script_pubkey.resize(25, 0x76);
        wtx.outputs.push_back(tout);

        // Witness stack: one input with [sig(72), pubkey(33)]
        WitnessStack ws;
        ws.push_back(WitnessItem(72, 0x30));
        ws.push_back(WitnessItem(33, 0x02));
        wtx.witness.push_back(ws);

        u_bip144_wtxid = bench_ns([&]{
            auto id = compute_wtxid(wtx);
            bench::DoNotOptimize(id[0]);
        }, N_SIGN);

        std::array<uint8_t, 32> wr{}, wn{};
        std::memset(wr.data(), 0x11, 32);
        std::memset(wn.data(), 0x00, 32);
        u_bip144_commitment = bench_ns([&]{
            auto c = witness_commitment(wr, wn);
            bench::DoNotOptimize(c[0]);
        }, N_SIGN * 4);

        u_bip144_weight = bench_ns([&]{
            auto w = tx_weight(wtx);
            bench::DoNotOptimize(w);
        }, N_SIGN * 4);

        // --- BIP-141: Witness programs ---
        uint8_t p2wpkh_spk[22] = {0x00, 0x14};
        std::memset(p2wpkh_spk + 2, 0xCC, 20);

        u_segwit_parse = bench_ns([&]{
            auto wp = parse_witness_program(p2wpkh_spk, 22);
            bench::DoNotOptimize(wp.version);
        }, N_SIGN * 4);

        u_segwit_p2wpkh_spk = bench_ns([&]{
            auto spk = segwit_scriptpubkey_p2wpkh(pkh.data());
            bench::DoNotOptimize(spk[0]);
        }, N_SIGN * 4);

        std::array<uint8_t, 32> script_hash{};
        std::memset(script_hash.data(), 0xDD, 32);
        u_segwit_p2wsh_spk = bench_ns([&]{
            auto spk = segwit_scriptpubkey_p2wsh(script_hash.data());
            bench::DoNotOptimize(spk[0]);
        }, N_SIGN * 4);

        // --- BIP-342 / BIP-341: Tapscript & keypath sighash ---
        std::array<uint8_t, 32> prev_txid{};
        std::memset(prev_txid.data(), 0xAB, 32);
        uint32_t prev_vout = 0;
        uint64_t in_amount = 100000;
        uint32_t in_seq = 0xFFFFFFFF;
        std::vector<uint8_t> spk_data(34, 0x51);
        spk_data[0] = 0x51; spk_data[1] = 0x20; // OP_1 PUSH32
        const uint8_t* spk_ptr = spk_data.data();
        size_t spk_len = spk_data.size();
        uint64_t out_val = 99000;

        TapSighashTxData td{};
        td.version = 2;
        td.locktime = 0;
        td.input_count = 1;
        td.prevout_txids = &prev_txid;
        td.prevout_vouts = &prev_vout;
        td.input_amounts = &in_amount;
        td.input_sequences = &in_seq;
        td.input_scriptpubkeys = &spk_ptr;
        td.input_scriptpubkey_lens = &spk_len;
        td.output_count = 1;
        td.output_values = &out_val;
        td.output_scriptpubkeys = &spk_ptr;
        td.output_scriptpubkey_lens = &spk_len;

        u_keypath_sighash = bench_ns([&]{
            auto h = taproot_keypath_sighash(td, 0, 0x00);
            bench::DoNotOptimize(h[0]);
        }, N_SIGN);

        std::array<uint8_t, 32> tapleaf{};
        std::memset(tapleaf.data(), 0xEE, 32);
        u_tapscript_sighash = bench_ns([&]{
            auto h = tapscript_sighash(td, 0, 0x00, tapleaf, 0x00, 0xFFFFFFFF);
            bench::DoNotOptimize(h[0]);
        }, N_SIGN);

        print_header("BIP-141/143/144/342 SEGWIT & TAPSCRIPT");
        print_row("BIP-143 sighash (1-in/1-out)",       u_bip143_sighash);
        print_row("BIP-143 p2wpkh_script_code",         u_bip143_script_code);
        print_row("BIP-144 compute_wtxid",               u_bip144_wtxid);
        print_row("BIP-144 witness_commitment",           u_bip144_commitment);
        print_row("BIP-144 tx_weight",                    u_bip144_weight);
        print_row("BIP-141 parse_witness_program",        u_segwit_parse);
        print_row("BIP-141 p2wpkh_scriptpubkey",          u_segwit_p2wpkh_spk);
        print_row("BIP-141 p2wsh_scriptpubkey",           u_segwit_p2wsh_spk);
        print_row("BIP-341 keypath_sighash",              u_keypath_sighash);
        print_row("BIP-342 tapscript_sighash",            u_tapscript_sighash);
        print_sep();
        printf("\n");
    }

    // =====================================================================
    //  SECTION 9b: BIP-324 Encrypted Transport
    // =====================================================================

    double u_ellswift_create = 0, u_ellswift_xdh = 0;
    double u_aead_encrypt = 0, u_aead_decrypt = 0;
    double u_hkdf_extract = 0, u_hkdf_expand = 0;
    double u_session_handshake = 0, u_session_encrypt_256 = 0, u_session_decrypt_256 = 0;
    double u_session_encrypt_1k = 0, u_session_roundtrip_256 = 0;

#ifdef SECP256K1_BIP324
    {
        using namespace secp256k1;

        static const std::uint8_t BIP324_KEY_A[32] = {
            0xe8,0xf3,0x2e,0x72,0x3d,0xec,0xf4,0x05,
            0x1a,0xef,0xac,0x8e,0x2c,0x93,0xc9,0xc5,
            0xb2,0x14,0x31,0x38,0x17,0xcd,0xb0,0x1a,
            0x14,0x94,0xb9,0x17,0xc8,0x43,0x6b,0x35
        };
        static const std::uint8_t BIP324_KEY_B[32] = {
            0xaa,0xbb,0xcc,0xdd,0x11,0x22,0x33,0x44,
            0x55,0x66,0x77,0x88,0x99,0x00,0xab,0xcd,
            0xef,0x01,0x23,0x45,0x67,0x89,0xab,0xcd,
            0xef,0xfe,0xdc,0xba,0x98,0x76,0x54,0x32
        };

        Scalar sk = Scalar::from_bytes(BIP324_KEY_A);

        // ElligatorSwift create (key encoding)
        u_ellswift_create = bench_ns([&]{
            auto enc = ellswift_create(sk);
            bench::DoNotOptimize(enc.data());
        }, std::max(1, N_SIGN / 4));

        // ElligatorSwift XDH (ECDH key agreement)
        auto enc_a = ellswift_create(sk);
        Scalar sk_b = Scalar::from_bytes(BIP324_KEY_B);
        auto enc_b = ellswift_create(sk_b);
        u_ellswift_xdh = bench_ns([&]{
            auto secret = ellswift_xdh(enc_a.data(), enc_b.data(), sk, true);
            bench::DoNotOptimize(secret.data());
        }, std::max(1, N_SIGN / 4));

        // HKDF-SHA256 extract
        const std::uint8_t salt[32] = {};
        const std::uint8_t ikm[32] = {1,2,3,4,5,6,7,8};
        u_hkdf_extract = bench_ns([&]{
            auto prk = hkdf_sha256_extract(salt, 32, ikm, 32);
            bench::DoNotOptimize(prk[0]);
        }, N_SIGN);

        // HKDF-SHA256 expand
        const std::uint8_t prk[32] = {0x01};
        const std::uint8_t info[] = "bitcoin_v2";
        u_hkdf_expand = bench_ns([&]{
            std::uint8_t okm[32];
            hkdf_sha256_expand(prk, info, sizeof(info) - 1, okm, 32);
            bench::DoNotOptimize(okm[0]);
        }, N_SIGN);

        // ChaCha20-Poly1305 AEAD encrypt (256-byte payload)
        std::uint8_t aead_key[32] = {};
        std::uint8_t nonce[12] = {};
        std::vector<std::uint8_t> pt_256(256, 0x42);
        std::vector<std::uint8_t> ct_buf(256);
        std::uint8_t tag_buf[16];
        u_aead_encrypt = bench_ns([&]{
            aead_chacha20_poly1305_encrypt(
                aead_key, nonce, nullptr, 0,
                pt_256.data(), pt_256.size(),
                ct_buf.data(), tag_buf);
            bench::DoNotOptimize(ct_buf[0]);
        }, N_SIGN);

        // ChaCha20-Poly1305 AEAD decrypt (256-byte payload)
        aead_chacha20_poly1305_encrypt(
            aead_key, nonce, nullptr, 0,
            pt_256.data(), pt_256.size(),
            ct_buf.data(), tag_buf);
        std::vector<std::uint8_t> dec_buf(256);
        u_aead_decrypt = bench_ns([&]{
            bool ok = aead_chacha20_poly1305_decrypt(
                aead_key, nonce, nullptr, 0,
                ct_buf.data(), ct_buf.size(),
                tag_buf, dec_buf.data());
            bench::DoNotOptimize(ok);
        }, N_SIGN);

        // Session handshake (create + complete)
        u_session_handshake = bench_ns([&]{
            Bip324Session init(true, BIP324_KEY_A);
            Bip324Session resp(false, BIP324_KEY_B);
            resp.complete_handshake(init.our_ellswift_encoding().data());
            init.complete_handshake(resp.our_ellswift_encoding().data());
            bench::DoNotOptimize(init.session_id().data());
        }, std::max(1, N_SIGN / 8));

        // Session encrypt 256B
        Bip324Session bench_init(true, BIP324_KEY_A);
        Bip324Session bench_resp(false, BIP324_KEY_B);
        bench_resp.complete_handshake(bench_init.our_ellswift_encoding().data());
        bench_init.complete_handshake(bench_resp.our_ellswift_encoding().data());

        u_session_encrypt_256 = bench_ns([&]{
            auto pkt = bench_init.encrypt(pt_256.data(), pt_256.size());
            bench::DoNotOptimize(pkt.data());
        }, N_SIGN);

        // Session decrypt 256B
        auto sample_pkt = bench_init.encrypt(pt_256.data(), pt_256.size());
        // Need a fresh session for decrypt (counter must match)
        Bip324Session dec_init(true, BIP324_KEY_A);
        Bip324Session dec_resp(false, BIP324_KEY_B);
        dec_resp.complete_handshake(dec_init.our_ellswift_encoding().data());
        dec_init.complete_handshake(dec_resp.our_ellswift_encoding().data());
        u_session_decrypt_256 = bench_ns([&]{
            auto pkt = dec_init.encrypt(pt_256.data(), pt_256.size());
            std::vector<std::uint8_t> plaintext;
            auto ok = dec_resp.decrypt(pkt.data(), pkt.data() + 3, pkt.size() - 3, plaintext);
            bench::DoNotOptimize(ok);
            bench::DoNotOptimize(plaintext.data());
        }, N_SIGN);

        // Session encrypt 1KB
        std::vector<std::uint8_t> pt_1k(1024, 0x55);
        Bip324Session enc1k_init(true, BIP324_KEY_A);
        Bip324Session enc1k_resp(false, BIP324_KEY_B);
        enc1k_resp.complete_handshake(enc1k_init.our_ellswift_encoding().data());
        enc1k_init.complete_handshake(enc1k_resp.our_ellswift_encoding().data());
        u_session_encrypt_1k = bench_ns([&]{
            auto pkt = enc1k_init.encrypt(pt_1k.data(), pt_1k.size());
            bench::DoNotOptimize(pkt.data());
        }, N_SIGN);

        // Full roundtrip 256B (encrypt + decrypt)
        Bip324Session rt_init(true, BIP324_KEY_A);
        Bip324Session rt_resp(false, BIP324_KEY_B);
        rt_resp.complete_handshake(rt_init.our_ellswift_encoding().data());
        rt_init.complete_handshake(rt_resp.our_ellswift_encoding().data());
        u_session_roundtrip_256 = bench_ns([&]{
            auto pkt = rt_init.encrypt(pt_256.data(), pt_256.size());
            std::vector<std::uint8_t> plaintext;
            auto ok = rt_resp.decrypt(pkt.data(), pkt.data() + 3, pkt.size() - 3, plaintext);
            bench::DoNotOptimize(ok);
            bench::DoNotOptimize(plaintext.data());
        }, N_SIGN);

        print_header("BIP-324 ENCRYPTED TRANSPORT");
        print_row("ElligatorSwift create",          u_ellswift_create);
        print_row("ElligatorSwift XDH (ECDH)",      u_ellswift_xdh);
        print_row("HKDF-SHA256 extract",            u_hkdf_extract);
        print_row("HKDF-SHA256 expand",             u_hkdf_expand);
        print_row("AEAD encrypt (256B)",            u_aead_encrypt);
        print_row("AEAD decrypt (256B)",            u_aead_decrypt);
        print_row("Session handshake (full)",       u_session_handshake);
        print_row("Session encrypt (256B)",         u_session_encrypt_256);
        print_row("Session decrypt (256B)",         u_session_decrypt_256);
        print_row("Session encrypt (1KB)",          u_session_encrypt_1k);
        print_row("Session roundtrip (256B)",       u_session_roundtrip_256);
        print_sep();
        printf("\n");
    }
#endif

    // =====================================================================
    //  SECTION 10: Summary
    // =====================================================================

    printf("======================================================================\n");
    printf("  THROUGHPUT SUMMARY (1 core, pinned)\n");
    printf("======================================================================\n\n");

    auto tput = [](const char* name, double ns) {
        const double ops = 1e9 / ns;
        const double us = ns / 1000.0;
        if (ops >= 1e6)
            printf("  %-38s %8.2f us  ->  %8.2f M op/s\n", name, us, ops / 1e6);
        else if (ops >= 1e3)
            printf("  %-38s %8.2f us  ->  %8.1f k op/s\n", name, us, ops / 1e3);
        else
            printf("  %-38s %8.2f us  ->  %8.0f   op/s\n", name, us, ops);
    };

    printf("  --- Ultra FAST ---\n");
    tput("ECDSA sign",                u_ecdsa_sign);
    tput("ECDSA verify",              u_ecdsa_verify);
    tput("Schnorr sign",              u_schnorr_sign);
    tput("Schnorr verify (cached)",   u_schnorr_verify);
    tput("Schnorr verify (raw)",      u_schnorr_verify_raw);
    tput("pubkey_create (k*G)",       keygen);
    tput("ECDH",                      u_ecdh);
    tput("Taproot output key",        u_taproot_out);
    tput("BIP32 derive (BTC)",        u_bip32_child);
    tput("Silent Payment sender",     u_silent_sender);
    tput("Silent Payment scan",       u_silent_scan);
    printf("\n");
    printf("  --- Ultra CT ---\n");
    tput("CT ECDSA sign",             u_ct_ecdsa);
    tput("CT Schnorr sign",           u_ct_schnorr);
    printf("\n");
    printf("  --- Ultra ZK ---\n");
    tput("Pedersen commit",           u_pedersen);
    tput("Knowledge prove",           u_kp_prove);
    tput("Knowledge verify",          u_kp_verify);
    tput("DLEQ prove",                u_dleq_prove);
    tput("DLEQ verify",               u_dleq_verify);
    tput("Bulletproof range_prove",   u_range_prove);
    tput("Bulletproof range_verify",  u_range_verify);
    printf("\n");

    printf("  --- Adaptor / FROST / MuSig2 ---\n");
    tput("Schnorr adaptor sign",      u_schnorr_adaptor_sign);
    tput("Schnorr adaptor verify",    u_schnorr_adaptor_verify);
    tput("ECDSA adaptor sign",        u_ecdsa_adaptor_sign);
    tput("FROST keygen_begin (2/3)",  u_frost_keygen);
    tput("FROST partial_sign",        u_frost_sign_partial);
    tput("FROST aggregate",           u_frost_aggregate);
    tput("MuSig2 key_agg (2-of-2)",   u_musig2_key_agg);
    tput("MuSig2 partial_sign",       u_musig2_partial_sign);
    tput("MuSig2 sig_agg",            u_musig2_sig_agg);
    printf("\n");

    printf("  --- ECIES / Msg-Signing / Hashing ---\n");
    tput("ECIES encrypt (256B)",      u_ecies_encrypt);
    tput("ECIES decrypt (256B)",      u_ecies_decrypt);
    tput("Bitcoin message sign",      u_btc_msg_sign);
    tput("Bitcoin message verify",    u_btc_msg_verify);
    tput("SHA-256 (32B)",             u_sha256_32);
    tput("SHA-512 (32B)",             u_sha512_32);
    tput("MSM (4 points)",            u_msm_4);
    tput("MSM (64 points)",           u_msm_64);
    printf("\n");

    printf("  --- BIP-39 Mnemonic ---\n");
    tput("bip39_generate (12w)",      u_bip39_gen12);
    tput("bip39_generate (24w)",      u_bip39_gen24);
    tput("bip39_validate (12w)",      u_bip39_validate);
    tput("bip39_to_seed (PBKDF2)",    u_bip39_to_seed);
    printf("\n");

    printf("  --- BIP-141/143/144/342 SegWit ---\n");
    tput("BIP-143 sighash",           u_bip143_sighash);
    tput("BIP-144 compute_wtxid",     u_bip144_wtxid);
    tput("BIP-141 parse_witness",     u_segwit_parse);
    tput("BIP-341 keypath_sighash",   u_keypath_sighash);
    tput("BIP-342 tapscript_sighash", u_tapscript_sighash);
    printf("\n");

#ifdef SECP256K1_BIP324
    if (u_ellswift_create > 0.0) {
        printf("  --- BIP-324 Transport ---\n");
        tput("ElligatorSwift create",     u_ellswift_create);
        tput("ElligatorSwift XDH",        u_ellswift_xdh);
        tput("HKDF extract",              u_hkdf_extract);
        tput("HKDF expand",               u_hkdf_expand);
        tput("AEAD encrypt (256B)",       u_aead_encrypt);
        tput("AEAD decrypt (256B)",       u_aead_decrypt);
        tput("Session handshake",         u_session_handshake);
        tput("Session encrypt (256B)",    u_session_encrypt_256);
        tput("Session roundtrip (256B)",  u_session_roundtrip_256);
        printf("\n");
    }
#endif
    printf("  --- libsecp256k1 ---\n");
    tput("field_mul",                 ls_fe_mul);
    tput("field_sqr",                 ls_fe_sqr);
    tput("field_inv_var",             ls_fe_inv);
    tput("scalar_mul",                ls_sc_mul);
    tput("scalar_inverse (CT)",       ls_sc_inv);
    tput("scalar_inverse_var",        ls_sc_inv_var);
    tput("point_dbl",                 ls_pt_dbl);
    tput("point_add (mixed)",         ls_pt_add_ge);
    tput("ecmult (a*P+b*G)",         ls_ecmult);
    tput("ecmult_gen (k*G raw)",      ls_ecmult_gen);
    tput("generator_mul (API)",       ls_gen);
    tput("scalar_mul_P (k*P)",        ls_kP);
    tput("ECDSA sign",                ls_ecdsa_sign);
    tput("ECDSA verify",              ls_ecdsa_verify);
    tput("Schnorr sign",              ls_schnorr_sign);
    tput("Schnorr verify",            ls_schnorr_verify);
    printf("\n");
#ifdef BENCH_HAS_OPENSSL
    if (ossl_gen > 0.0) {
        printf("  --- OpenSSL ---\n");
        tput("ECDSA sign",                ossl_ecdsa_sign);
        tput("ECDSA verify",              ossl_ecdsa_verify);
        tput("generator_mul (k*G)",       ossl_gen);
        printf("\n");
    }
#endif

    // ---- ConnectBlock: 2000-unique-pubkey block validation (MEASURED) -----------
    // Simulates Bitcoin Core ConnectBlock: each tx has a distinct pubkey.
    // Exposes pubkey-cache overhead in any shim middleware:
    //   libsecp: ge stored directly in secp256k1_pubkey → zero per-verify parse.
    //   Ultra shim: a 16-slot cache causes ~0% hit rate with unique pubkeys
    //               → ~1,954 ns rebuild overhead × 2000 txs ≈ 3.9 ms per block.
    // Fix (Variant B): store AffinePointCompact (64B) in secp256k1_pubkey->data
    // at parse time; reconstruct Point in verify without any cache lookup.
    //
    // NOTE: This benchmark measures Ultra's NATIVE C++ verify (no shim) vs
    // libsecp's API. Native verify has no cache overhead. A shim benchmark
    // requires the separate Bitcoin Core integration layer.

#ifndef SECP256K1_PLATFORM_ESP32
    {
        constexpr std::size_t N_BLOCK = 2000;

        // Setup: generate N_BLOCK unique (key, msg, sig) tuples
        std::vector<Point>                  u_blk_pk(N_BLOCK);
        std::vector<EcdsaPublicKey>         u_blk_epk(N_BLOCK);   // pre-parsed (ECDSA PERF-B)
        std::vector<ECDSASignature>         u_blk_esig(N_BLOCK);
        std::vector<std::array<uint8_t,32>> u_blk_spk(N_BLOCK);
        std::vector<SchnorrSignature>       u_blk_ssig(N_BLOCK);
        std::vector<SchnorrXonlyPubkey>     u_blk_xonly(N_BLOCK); // pre-parsed (Variant B)
        std::vector<std::array<uint8_t,32>> blk_msg(N_BLOCK);
        std::vector<secp256k1_pubkey>           ls_blk_pk(N_BLOCK);
        std::vector<secp256k1_ecdsa_signature>  ls_blk_esig(N_BLOCK);
        std::vector<secp256k1_xonly_pubkey>     ls_blk_xonly(N_BLOCK);
        std::vector<std::array<uint8_t,64>>     ls_blk_ssig(N_BLOCK);

        bool blk_setup_ok = true;
        for (std::size_t i = 0; i < N_BLOCK; ++i) {
            blk_msg[i]   = make_hash(0xB10C0000ULL + i);
            auto sk_b    = make_hash(0xB10CFFFFULL + i);
            Scalar sk    = Scalar::from_bytes(sk_b);
            if (SECP256K1_UNLIKELY(sk.is_zero())) sk = Scalar::one();

            // Ultra side
            u_blk_pk[i]   = Point::generator().scalar_mul(sk);
            u_blk_pk[i].normalize();
            u_blk_esig[i] = ecdsa_sign(blk_msg[i], sk);
            auto kp        = schnorr_keypair_create(sk);
            u_blk_spk[i]  = kp.px;
            u_blk_ssig[i] = schnorr_sign(kp, blk_msg[i], aux_rands[i % POOL]);
            // Pre-parse Schnorr pubkey — populates GLV cache (PERF-B).
            schnorr_xonly_pubkey_parse(u_blk_xonly[i], u_blk_spk[i]);
            // Pre-parse ECDSA pubkey — builds GLV P/phi(P) tables once (ECDSA PERF-B).
            // Simulates Bitcoin Core's secp256k1_ec_pubkey_parse path.
            {
                auto unc = u_blk_pk[i].to_uncompressed(); // [0x04 | x:32 | y:32]
                uint8_t comp33[33];
                comp33[0] = 0x02u | (unc[64] & 1u);       // parity from y LSB
                std::memcpy(comp33 + 1, unc.data() + 1, 32);
                ecdsa_pubkey_parse(u_blk_epk[i], comp33, 33);
            }

            // libsecp side (ge stored in secp256k1_pubkey at parse time)
            int rc1 = secp256k1_ec_pubkey_create(ls_ctx, &ls_blk_pk[i], sk_b.data());
            int rc2 = secp256k1_ecdsa_sign(ls_ctx, &ls_blk_esig[i],
                                            blk_msg[i].data(), sk_b.data(), NULL, NULL);
            secp256k1_keypair blk_kp;
            int rc3 = secp256k1_keypair_create(ls_ctx, &blk_kp, sk_b.data());
            int rc4 = secp256k1_schnorrsig_sign32(ls_ctx, ls_blk_ssig[i].data(),
                                                   blk_msg[i].data(), &blk_kp,
                                                   aux_rands[i % POOL].data());
            int rc5 = secp256k1_keypair_xonly_pub(ls_ctx, &ls_blk_xonly[i], NULL, &blk_kp);
            if (SECP256K1_UNLIKELY(!rc1 || !rc2 || !rc3 || !rc4 || !rc5)) blk_setup_ok = false;
        }
        if (SECP256K1_UNLIKELY(!blk_setup_ok)) {
            printf("[ConnectBlock bench] setup failed — skipping\n");
        } else {

        // Use a light-warmup harness (5 warmup × 2000 ops = 10K ops, ~0.5 s)
        bench::Harness blk_H(5, static_cast<std::size_t>(effective_passes));
        auto blk_bench = [&](auto fn) -> double {
            return blk_H.run(1, fn);  // 1 call per pass = one full block
        };

        // AllEcdsa (PERF-B: use pre-parsed EcdsaPublicKey — avoids ~1,954 ns GLV table rebuild)
        double u_cblk_ecdsa = blk_bench([&]() {
            for (std::size_t i = 0; i < N_BLOCK; ++i) {
                bool ok = ecdsa_verify(blk_msg[i].data(), u_blk_epk[i], u_blk_esig[i]);
                bench::DoNotOptimize(ok);
            }
        });
        double ls_cblk_ecdsa = blk_bench([&]() {
            for (std::size_t i = 0; i < N_BLOCK; ++i) {
                int ok = secp256k1_ecdsa_verify(ls_ctx, &ls_blk_esig[i],
                                                blk_msg[i].data(), &ls_blk_pk[i]);
                bench::DoNotOptimize(ok);
            }
        });

        // AllSchnorr (PERF-B: use pre-parsed SchnorrXonlyPubkey — avoids lift_x + GLV rebuild)
        double u_cblk_schnorr = blk_bench([&]() {
            for (std::size_t i = 0; i < N_BLOCK; ++i) {
                bool ok = schnorr_verify(u_blk_xonly[i],
                                         blk_msg[i].data(), u_blk_ssig[i]);
                bench::DoNotOptimize(ok);
            }
        });
        double ls_cblk_schnorr = blk_bench([&]() {
            for (std::size_t i = 0; i < N_BLOCK; ++i) {
                int ok = secp256k1_schnorrsig_verify(ls_ctx, ls_blk_ssig[i].data(),
                                                     blk_msg[i].data(), 32,
                                                     &ls_blk_xonly[i]);
                bench::DoNotOptimize(ok);
            }
        });

        // Mixed: 2000 Schnorr + 1000 ECDSA (Taproot block approximation)
        // PERF-B: pre-parsed SchnorrXonlyPubkey (Schnorr) + pre-parsed EcdsaPublicKey (ECDSA)
        double u_cblk_mixed = blk_bench([&]() {
            for (std::size_t i = 0; i < N_BLOCK + N_BLOCK / 2; ++i) {
                std::size_t const j = i % N_BLOCK;
                bool ok;
                if (i < N_BLOCK)
                    ok = schnorr_verify(u_blk_xonly[j], blk_msg[j].data(), u_blk_ssig[j]);
                else
                    ok = ecdsa_verify(blk_msg[j].data(), u_blk_epk[j], u_blk_esig[j]);
                bench::DoNotOptimize(ok);
            }
        });
        double ls_cblk_mixed = blk_bench([&]() {
            for (std::size_t i = 0; i < N_BLOCK + N_BLOCK / 2; ++i) {
                std::size_t const j = i % N_BLOCK;
                int ok;
                if (i < N_BLOCK) {
                    ok = secp256k1_schnorrsig_verify(ls_ctx, ls_blk_ssig[j].data(),
                                                     blk_msg[j].data(), 32,
                                                     &ls_blk_xonly[j]);
                } else {
                    ok = secp256k1_ecdsa_verify(ls_ctx, &ls_blk_esig[j],
                                                blk_msg[j].data(), &ls_blk_pk[j]);
                }
                bench::DoNotOptimize(ok);
            }
        });

        printf("======================================================================\n");
        printf("  CONNECTBLOCK: 2000-UNIQUE-PUBKEY BLOCK VALIDATION (measured)\n");
        printf("  Ultra = native C++ API (no shim/cache overhead)\n");
        printf("  libsecp = secp256k1_ecdsa_verify API (ge in pubkey struct)\n");
        printf("======================================================================\n");
        printf("\n");
        auto print_blk_row = [](const char* name, double u_ns, double ls_ns) {
            double u_ms  = u_ns  / 1e6;
            double ls_ms = ls_ns / 1e6;
            double ratio = ls_ms / u_ms;
            const char* verdict = (ratio >= 0.99) ? "Ultra ✓" : "libsecp";
            printf("  %-38s  %7.1f ms  %7.1f ms  %6.2fx  %s\n",
                   name, u_ms, ls_ms, ratio, verdict);
        };
        printf("  %-38s  %9s  %9s  %7s\n",
               "Scenario", "Ultra", "libsecp", "ratio");
        printf("  %s\n", std::string(72, '-').c_str());
        print_blk_row("ConnectBlockAllEcdsa (2000 ECDSA)",
                      u_cblk_ecdsa, ls_cblk_ecdsa);
        print_blk_row("ConnectBlockAllSchnorr (2000 Schnorr)",
                      u_cblk_schnorr, ls_cblk_schnorr);
        print_blk_row("ConnectBlockMixedEcdsaSchnorr (2k+1k)",
                      u_cblk_mixed, ls_cblk_mixed);

        // ConnectBlockWithDerParse: both sides start from raw DER bytes + compressed pubkey.
        // Includes: DER parse → normalize → pubkey parse → verify.
        // Ultra side: DER parse via shim → compact → ECDSASignature, then Ultra C++ verify
        //             with pre-computed pubkey (Ultra has no public Point::from_compressed()).
        // libsecp side: full shim DER pipeline (shim routes through Ultra internally).
        std::vector<std::pair<std::array<uint8_t,72>,std::size_t>> blk_der(N_BLOCK);
        std::vector<std::array<uint8_t,33>> blk_comp(N_BLOCK);
        for (std::size_t i = 0; i < N_BLOCK; ++i) {
            blk_der[i]  = u_blk_esig[i].to_der();
            blk_comp[i] = u_blk_pk[i].to_compressed();
        }
        double u_cblk_der = blk_bench([&]() {
            for (std::size_t i = 0; i < N_BLOCK; ++i) {
                // DER parse via shim (Ultra's DER parser under the hood)
                secp256k1_ecdsa_signature usig_raw;
                if (!secp256k1_ecdsa_signature_parse_der(ls_ctx, &usig_raw,
                        blk_der[i].first.data(), blk_der[i].second))
                    continue;
                // Normalize low-S via shim
                (void)secp256k1_ecdsa_signature_normalize(ls_ctx, &usig_raw, &usig_raw);
                // Convert to Ultra C++ ECDSASignature via compact representation
                unsigned char compact64[64];
                (void)secp256k1_ecdsa_signature_serialize_compact(ls_ctx, compact64, &usig_raw);
                ECDSASignature sig = ECDSASignature::from_compact(compact64);
                // Verify using Ultra C++ native path with pre-computed pubkey
                bool v = ecdsa_verify(blk_msg[i].data(), u_blk_pk[i], sig);
                bench::DoNotOptimize(v);
            }
        });
        double ls_cblk_der = blk_bench([&]() {
            for (std::size_t i = 0; i < N_BLOCK; ++i) {
                // Parse DER
                secp256k1_ecdsa_signature lsig;
                (void)secp256k1_ecdsa_signature_parse_der(ls_ctx, &lsig,
                    blk_der[i].first.data(), blk_der[i].second);
                // Normalize to low-S
                (void)secp256k1_ecdsa_signature_normalize(ls_ctx, &lsig, &lsig);
                // Parse pubkey from compressed
                secp256k1_pubkey lpk;
                (void)secp256k1_ec_pubkey_parse(ls_ctx, &lpk, blk_comp[i].data(), 33);
                // Verify
                int v = secp256k1_ecdsa_verify(ls_ctx, &lsig, blk_msg[i].data(), &lpk);
                bench::DoNotOptimize(v);
            }
        });
        print_blk_row("ConnectBlockWithDerParse [DER+parse+norm]",
                      u_cblk_der, ls_cblk_der);
        printf("\n");

        g_report.add("ConnectBlock", "DerParse_ultra_ms",   u_cblk_der  / 1e6);
        g_report.add("ConnectBlock", "DerParse_libsecp_ms", ls_cblk_der / 1e6);

        g_report.add("ConnectBlock", "AllEcdsa_ultra_ms",    u_cblk_ecdsa   / 1e6);
        g_report.add("ConnectBlock", "AllEcdsa_libsecp_ms",  ls_cblk_ecdsa  / 1e6);
        g_report.add("ConnectBlock", "AllSchnorr_ultra_ms",  u_cblk_schnorr / 1e6);
        g_report.add("ConnectBlock", "AllSchnorr_libsecp_ms",ls_cblk_schnorr/ 1e6);
        g_report.add("ConnectBlock", "Mixed_ultra_ms",        u_cblk_mixed   / 1e6);
        g_report.add("ConnectBlock", "Mixed_libsecp_ms",      ls_cblk_mixed  / 1e6);
        // (DerParse rows added above)
        } // closes else (blk_setup_ok)
    }
#endif // SECP256K1_PLATFORM_ESP32

    // ---- Block validation estimates -----------------------------------------

    printf("======================================================================\n");
    printf("  BITCOIN BLOCK VALIDATION ESTIMATES (1 core)\n");
    printf("======================================================================\n\n");

    const double pre_taproot_ms = 3000.0 * u_ecdsa_verify / 1e6;
    const double taproot_ms = (2000.0 * u_schnorr_verify_raw + 1000.0 * u_ecdsa_verify) / 1e6;
    printf("  Pre-Taproot block (~3000 ECDSA verify):\n");
    printf("    Wall time:  %7.1f ms\n", pre_taproot_ms);
    printf("    Blocks/sec: %7.1f\n\n", 1000.0 / pre_taproot_ms);

    printf("  Taproot block (~2000 Schnorr + ~1000 ECDSA):\n");
    printf("    Wall time:  %7.1f ms\n", taproot_ms);
    printf("    Blocks/sec: %7.1f\n\n", 1000.0 / taproot_ms);

    printf("  TX throughput (1 core):\n");
    printf("    ECDSA:    %8.0f tx/sec\n", 1e9 / u_ecdsa_verify);
    printf("    Schnorr:  %8.0f tx/sec\n", 1e9 / u_schnorr_verify_raw);
    printf("\n");

    // ---- Footer -------------------------------------------------------------
    printf("======================================================================\n");
    printf("  %s | 1 core pinned | "
#if defined(__clang__)
        "Clang " __clang_version__
#elif defined(_MSC_VER)
        "MSVC " STR(_MSC_VER)
#elif defined(__GNUC__)
        "GCC " __VERSION__
#else
        "Unknown"
#endif
        "\n", cpu_brand);
    printf("  UltrafastSecp256k1 vs libsecp256k1"
#ifdef BENCH_HAS_OPENSSL
           " vs OpenSSL"
#endif
           " -- Unified Benchmark\n");
    printf("======================================================================\n\n");

    // ---- JSON export --------------------------------------------------------
    if (opts.json_path) {
        if (g_report.write_json(opts.json_path)) {
            printf("  JSON report written to: %s\n", opts.json_path);
        } else {
            printf("  [!] Failed to write JSON report to: %s\n", opts.json_path);
        }
    }

    return 0;
}
