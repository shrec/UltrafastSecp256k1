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
//   - 500 warmup iterations per operation
//   - 11 measurement passes, IQR outlier removal, median
//   - 64-key pool to prevent caching artifacts
//   - RDTSCP on x86, chrono fallback on ARM/RISC-V/ESP32
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
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/selftest.hpp"
#include "secp256k1/init.hpp"
#include "secp256k1/benchmark_harness.hpp"
#include "secp256k1/glv.hpp"
#include "secp256k1/batch_verify.hpp"
#if defined(__SIZEOF_INT128__) && !defined(__EMSCRIPTEN__)
#include "secp256k1/field_52.hpp"
#endif

// libsecp256k1 public API (linked from libsecp_provider.c)
#include "secp256k1.h"
#include "secp256k1_extrakeys.h"
#include "secp256k1_schnorrsig.h"

// Thin wrappers from libsecp_provider.c exposing internal field ops
extern "C" {
    void libsecp_fe_inv_var(unsigned char out32[32], const unsigned char in32[32]);
    void libsecp_fe_inv_var_raw(void *r, const void *a);
}

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>

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
        fprintf(f, "    \"pool_size\": %d\n", pool_size);
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
    for (int i = 0; i < POOL; ++i)
        pubkeys[i] = Point::generator().scalar_mul(privkeys[i]);

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
    print_row("point_add", ptadd);

    const double ptdbl = bench_ns([&]() {
        auto r = pubkeys[0].dbl();
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("point_dbl", ptdbl);
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

    // -- Point::add (wrapper around jac52_add) --
    const double micro_pt_add = bench_ns([&]() {
        auto r = pubkeys[0].add(pubkeys[1]);
        bench::DoNotOptimize(r);
    }, N_POINT);
    print_row("Point::add (jac52_add)", micro_pt_add);

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
        constexpr int BATCH_SIZES[] = {4, 16, 64};
        constexpr int N_BATCH_SIZES = 3;

        // -- Schnorr Batch Verify --
        for (int bi = 0; bi < N_BATCH_SIZES; ++bi) {
            const int batch_n = BATCH_SIZES[bi];

            // Prepare batch entries (reuse pool cyclically)
            std::vector<SchnorrBatchEntry> schnorr_batch(static_cast<std::size_t>(batch_n));
            for (int j = 0; j < batch_n; ++j) {
                schnorr_batch[static_cast<std::size_t>(j)].pubkey_x = schnorr_pubkeys_x[j % POOL];
                schnorr_batch[static_cast<std::size_t>(j)].message  = msghashes[j % POOL];
                schnorr_batch[static_cast<std::size_t>(j)].signature = schnorr_sigs[j % POOL];
            }

            // Correctness sanity check
            bool batch_ok = schnorr_batch_verify(schnorr_batch);
            if (!batch_ok) {
                printf("[!] schnorr_batch_verify(%d) FAILED correctness check\n", batch_n);
            }

            // Bench: fewer iterations for larger batches
            const int iters = batch_n <= 16 ? 200 : 100;
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

            const int iters = batch_n <= 16 ? 200 : 100;
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
    //  SECTION 7: libsecp256k1 (bitcoin-core) -- SAME harness, pool, timer
    // =====================================================================

    printf("Running libsecp256k1 benchmark (same harness: RDTSCP, 3s ramp-up, 500 warmup, 11 passes, IQR)...\n");

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

    for (int i = 0; i < POOL; ++i) {
        auto const h = make_hash(0xdeadbeef00ULL + static_cast<uint64_t>(i));
        std::memcpy(ls_seckeys[i], h.data(), 32);
        secp256k1_ec_pubkey_create(ls_ctx, &ls_pubkeys[i], ls_seckeys[i]);
        secp256k1_keypair_create(ls_ctx, &ls_keypairs[i], ls_seckeys[i]);
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
        secp256k1_ec_pubkey_create(ls_ctx, &pk, ls_seckeys[idx % POOL]);
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

    // Schnorr Keypair Create
    idx = 0;
    const double ls_schnorr_kp = bench_ns([&]() {
        secp256k1_keypair kp;
        secp256k1_keypair_create(ls_ctx, &kp, ls_seckeys[idx % POOL]);
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
        secp256k1_ec_pubkey_tweak_mul(ls_ctx, &pk_copy,
                                      ls_seckeys[(idx + 1) % POOL]);
        bench::DoNotOptimize(pk_copy); ++idx;
    }, N_SCALAR);

    secp256k1_context_destroy(ls_ctx);

    print_header("libsecp256k1 (bitcoin-core)");
    print_row("field_inv_var",                    ls_fe_inv);
    print_row("generator_mul (ec_pubkey_create)", ls_gen);
    print_row("scalar_mul (k*P, tweak_mul)",     ls_kP);
    print_row("ecdsa_sign",                      ls_ecdsa_sign);
    print_row("ecdsa_verify",                    ls_ecdsa_verify);
    print_row("schnorr_keypair_create",          ls_schnorr_kp);
    print_row("schnorr_sign (BIP-340)",          ls_schnorr_sign);
    print_row("schnorr_verify (BIP-340)",        ls_schnorr_verify);
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
    //  SECTION 8: Apple-to-Apple Ratio Table
    // =====================================================================

    // Note: libsecp256k1 is ALWAYS constant-time for signing.
    // "FAST" comparison uses Ultra FAST path (unfair for signing/keygen).
    // "CT-vs-CT" shows the true apples-to-apples for signing ops.
    // Verify uses only public data -- no CT needed, same in both paths.

    printf("======================================================================\n");
    printf("  APPLE-TO-APPLE: UltrafastSecp256k1 / libsecp256k1\n");
    printf("  (ratio > 1.0 = Ultra wins, < 1.0 = libsecp256k1 wins)\n");
    printf("======================================================================\n\n");

    print_header_ratio("FAST path (Ultra FAST vs libsecp)");
    print_ratio("Generator * k",        ls_gen            / keygen);
    print_ratio("Scalar * P (k*P)",     ls_kP             / scalarmul);
    print_ratio("Scalar * P (KPlan)",   ls_kP             / plan_mul);
    print_ratio("ECDSA Sign",           ls_ecdsa_sign     / u_ecdsa_sign);
    print_ratio("ECDSA Verify",         ls_ecdsa_verify   / u_ecdsa_verify);
    print_ratio("Schnorr Keypair",      ls_schnorr_kp     / u_schnorr_kp);
    print_ratio("Schnorr Sign",         ls_schnorr_sign   / u_schnorr_sign);
    print_ratio("Schnorr Verify",       ls_schnorr_verify / u_schnorr_verify);
    print_sep();
    printf("\n");

    // CT-vs-CT: libsecp256k1 sign is always CT, so compare with Ultra CT sign.
    // Verify doesn't change (no secret data), so same numbers as FAST.
    print_header_ratio("CT-vs-CT (Ultra CT vs libsecp CT)");
    print_ratio("ECDSA Sign (CT vs CT)",    ls_ecdsa_sign   / u_ct_ecdsa);
    print_ratio("ECDSA Verify",             ls_ecdsa_verify / u_ecdsa_verify);
    print_ratio("Schnorr Sign (CT vs CT)",  ls_schnorr_sign / u_ct_schnorr);
    print_ratio("Schnorr Verify",           ls_schnorr_verify / u_schnorr_verify);
    print_sep();
    printf("\n");

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
    //  SECTION 9: Summary
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
    tput("Schnorr verify",            u_schnorr_verify);
    tput("pubkey_create (k*G)",       keygen);
    printf("\n");
    printf("  --- Ultra CT ---\n");
    tput("CT ECDSA sign",             u_ct_ecdsa);
    tput("CT Schnorr sign",           u_ct_schnorr);
    printf("\n");
    printf("  --- libsecp256k1 ---\n");
    tput("ECDSA sign",                ls_ecdsa_sign);
    tput("ECDSA verify",              ls_ecdsa_verify);
    tput("Schnorr sign",              ls_schnorr_sign);
    tput("Schnorr verify",            ls_schnorr_verify);
    tput("generator_mul",             ls_gen);
    tput("scalar_mul (k*P)",          ls_kP);
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

    // ---- Block validation estimates -----------------------------------------

    printf("======================================================================\n");
    printf("  BITCOIN BLOCK VALIDATION ESTIMATES (1 core)\n");
    printf("======================================================================\n\n");

    const double pre_taproot_ms = 3000.0 * u_ecdsa_verify / 1e6;
    const double taproot_ms = (2000.0 * u_schnorr_verify + 1000.0 * u_ecdsa_verify) / 1e6;
    printf("  Pre-Taproot block (~3000 ECDSA verify):\n");
    printf("    Wall time:  %7.1f ms\n", pre_taproot_ms);
    printf("    Blocks/sec: %7.1f\n\n", 1000.0 / pre_taproot_ms);

    printf("  Taproot block (~2000 Schnorr + ~1000 ECDSA):\n");
    printf("    Wall time:  %7.1f ms\n", taproot_ms);
    printf("    Blocks/sec: %7.1f\n\n", 1000.0 / taproot_ms);

    printf("  TX throughput (1 core):\n");
    printf("    ECDSA:    %8.0f tx/sec\n", 1e9 / u_ecdsa_verify);
    printf("    Schnorr:  %8.0f tx/sec\n", 1e9 / u_schnorr_verify);
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
