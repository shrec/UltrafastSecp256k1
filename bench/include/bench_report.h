// bench_report.h -- JSON + Markdown table report writer
#pragma once

#include "bench_timer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace bench {

// Per-case result
struct CaseResult {
    const char* case_name;      // e.g. "ecdsa_verify_bytes"
    const char* provider_name;  // "UF" or "libsecp256k1"
    BenchStats  stats;
    bool        correctness_ok;
};

// Environment snapshot
struct EnvInfo {
    char cpu_model[128];
    char os_name[64];
    char compiler[128];
    char uf_commit[48];
    char libsecp_version[32];
    double tsc_mhz;
    int  pinned_core;
};

inline void fill_env_info(EnvInfo& env, int pinned_core) {
    std::memset(&env, 0, sizeof(env));
    env.pinned_core = pinned_core;
    env.tsc_mhz = estimate_tsc_mhz();

#ifdef _WIN32
    std::snprintf(env.os_name, sizeof(env.os_name), "Windows");
#elif defined(__linux__)
    std::snprintf(env.os_name, sizeof(env.os_name), "Linux");
#else
    std::snprintf(env.os_name, sizeof(env.os_name), "Unknown");
#endif

#if defined(__clang__)
    std::snprintf(env.compiler, sizeof(env.compiler),
                  "clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(__GNUC__)
    std::snprintf(env.compiler, sizeof(env.compiler),
                  "gcc %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#elif defined(_MSC_VER)
    std::snprintf(env.compiler, sizeof(env.compiler), "MSVC %d", _MSC_VER);
#else
    std::snprintf(env.compiler, sizeof(env.compiler), "unknown");
#endif

    std::snprintf(env.uf_commit, sizeof(env.uf_commit), "unknown");
    std::snprintf(env.libsecp_version, sizeof(env.libsecp_version), "v0.6.0");

#if defined(__linux__)
    FILE* f = std::fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        while (std::fgets(line, sizeof(line), f)) {
            if (std::strncmp(line, "model name", 10) == 0) {
                const char* p = std::strchr(line, ':');
                if (p) {
                    ++p;
                    while (*p == ' ') ++p;
                    // remove trailing newline
                    char* nl = std::strchr(const_cast<char*>(p), '\n');
                    if (nl) *nl = '\0';
                    std::snprintf(env.cpu_model, sizeof(env.cpu_model), "%s", p);
                }
                break;
            }
        }
        std::fclose(f);
    }
#elif defined(_WIN32)
    std::snprintf(env.cpu_model, sizeof(env.cpu_model), "see WMIC/systeminfo");
#endif
}

// Print markdown table to stdout
inline void print_markdown_table(const CaseResult* results, int count,
                                 const EnvInfo& env) {
    std::printf("\n=== Benchmark Report ===\n\n");
    std::printf("CPU        : %s\n", env.cpu_model);
    std::printf("OS         : %s\n", env.os_name);
    std::printf("Compiler   : %s\n", env.compiler);
    std::printf("TSC freq   : %.1f MHz\n", env.tsc_mhz);
    std::printf("Pinned core: %d\n", env.pinned_core);
    std::printf("UF commit  : %s\n", env.uf_commit);
    std::printf("libsecp256k1: %s\n\n", env.libsecp_version);

    std::printf("| %-30s | %-12s | %10s | %10s | %10s | %12s | %5s |\n",
                "Case", "Provider", "Median(ns)", "P10(ns)", "P90(ns)", "ops/sec", "OK?");
    std::printf("|%s|%s|%s|%s|%s|%s|%s|\n",
                "--------------------------------", "--------------",
                "------------", "------------", "------------",
                "--------------", "-------");

    for (int i = 0; i < count; ++i) {
        const auto& r = results[i];
        std::printf("| %-30s | %-12s | %10.1f | %10.1f | %10.1f | %12.0f | %-5s |\n",
                    r.case_name, r.provider_name,
                    r.stats.median_ns, r.stats.p10_ns, r.stats.p90_ns,
                    r.stats.ops_per_sec,
                    r.correctness_ok ? "OK" : "FAIL");
    }
    std::printf("\n");
}

// Write JSON report to file
inline bool write_json_report(const char* path,
                              const CaseResult* results, int count,
                              const EnvInfo& env) {
    FILE* f = std::fopen(path, "w");
    if (!f) {
        std::fprintf(stderr, "[bench] ERROR: cannot open %s for writing\n", path);
        return false;
    }

    std::fprintf(f, "{\n");
    std::fprintf(f, "  \"environment\": {\n");
    std::fprintf(f, "    \"cpu_model\": \"%s\",\n", env.cpu_model);
    std::fprintf(f, "    \"os\": \"%s\",\n", env.os_name);
    std::fprintf(f, "    \"compiler\": \"%s\",\n", env.compiler);
    std::fprintf(f, "    \"tsc_mhz\": %.1f,\n", env.tsc_mhz);
    std::fprintf(f, "    \"pinned_core\": %d,\n", env.pinned_core);
    std::fprintf(f, "    \"uf_commit\": \"%s\",\n", env.uf_commit);
    std::fprintf(f, "    \"libsecp_version\": \"%s\"\n", env.libsecp_version);
    std::fprintf(f, "  },\n");
    std::fprintf(f, "  \"results\": [\n");

    for (int i = 0; i < count; ++i) {
        const auto& r = results[i];
        std::fprintf(f, "    {\n");
        std::fprintf(f, "      \"case\": \"%s\",\n", r.case_name);
        std::fprintf(f, "      \"provider\": \"%s\",\n", r.provider_name);
        std::fprintf(f, "      \"median_ns\": %.1f,\n", r.stats.median_ns);
        std::fprintf(f, "      \"p10_ns\": %.1f,\n", r.stats.p10_ns);
        std::fprintf(f, "      \"p90_ns\": %.1f,\n", r.stats.p90_ns);
        std::fprintf(f, "      \"ops_per_sec\": %.0f,\n", r.stats.ops_per_sec);
        std::fprintf(f, "      \"correctness\": %s\n", r.correctness_ok ? "true" : "false");
        std::fprintf(f, "    }%s\n", (i + 1 < count) ? "," : "");
    }

    std::fprintf(f, "  ]\n");
    std::fprintf(f, "}\n");
    std::fclose(f);
    return true;
}

} // namespace bench
