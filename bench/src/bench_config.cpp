// bench_config.cpp -- CLI argument parsing implementation
#include "bench_config.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>

namespace bench {

void print_usage(const char* progname) {
    std::printf("Usage: %s [OPTIONS]\n\n", progname);
    std::printf("Apples-to-apples benchmark: UltrafastSecp256k1 vs libsecp256k1\n\n");
    std::printf("Provider selection:\n");
    std::printf("  --uf-only             Run UltrafastSecp256k1 only\n");
    std::printf("  --libsecp-only        Run libsecp256k1 only\n");
    std::printf("\n");
    std::printf("Case selection (default: all enabled):\n");
    std::printf("  --case=ecdsa          ECDSA verify only\n");
    std::printf("  --case=schnorr        Schnorr verify only\n");
    std::printf("  --case=pubkey         pubkey_create only\n");
    std::printf("  --case=ecdh           ECDH only\n");
    std::printf("  --case=all            All cases (default)\n");
    std::printf("\n");
    std::printf("Dataset:\n");
    std::printf("  --n=<count>           Dataset size (default: 100000)\n");
    std::printf("  --seed=<uint64>       PRNG seed (default: 42)\n");
    std::printf("\n");
    std::printf("Timing:\n");
    std::printf("  --warmup=<ms>         Warmup duration (default: 250)\n");
    std::printf("  --measure=<ms>        Measurement duration (default: 2000)\n");
    std::printf("  --pin-core=<id>       Pin to CPU core (-1 = disabled)\n");
    std::printf("\n");
    std::printf("Options:\n");
    std::printf("  --msg=prehashed32     Use 32-byte prehashed messages (default)\n");
    std::printf("  --msg=sha256d         SHA256d over raw messages\n");
    std::printf("  --ecdsa-sig=der       DER-encoded ECDSA signatures (default)\n");
    std::printf("  --ecdsa-sig=compact   64-byte compact signatures\n");
    std::printf("  --libsecp-randomize=0 Disable context randomization\n");
    std::printf("  --libsecp-randomize=1 Enable context randomization (default)\n");
    std::printf("\n");
    std::printf("Output:\n");
    std::printf("  --json=<path>         Write JSON report to file\n");
    std::printf("  --help                Show this help\n");
}

static bool starts_with(const char* str, const char* prefix) {
    return std::strncmp(str, prefix, std::strlen(prefix)) == 0;
}

static const char* after_eq(const char* str) {
    const char* p = std::strchr(str, '=');
    return p ? p + 1 : nullptr;
}

bool parse_args(BenchConfig& cfg, int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
            print_usage(argv[0]);
            return false;
        }
        else if (std::strcmp(arg, "--uf-only") == 0) {
            cfg.run_uf = true;
            cfg.run_libsecp = false;
        }
        else if (std::strcmp(arg, "--libsecp-only") == 0) {
            cfg.run_uf = false;
            cfg.run_libsecp = true;
        }
        else if (starts_with(arg, "--case=")) {
            const char* val = after_eq(arg);
            if (!val) { std::fprintf(stderr, "Bad --case=\n"); return false; }

            if (std::strcmp(val, "all") == 0) {
                cfg.case_ecdsa_verify   = true;
                cfg.case_schnorr_verify = true;
                cfg.case_pubkey_create  = true;
                cfg.case_ecdh           = true;
            } else {
                // Disable all first, enable selected
                cfg.case_ecdsa_verify = cfg.case_schnorr_verify = false;
                cfg.case_pubkey_create = cfg.case_ecdh = false;

                if (std::strcmp(val, "ecdsa") == 0)        cfg.case_ecdsa_verify = true;
                else if (std::strcmp(val, "schnorr") == 0) cfg.case_schnorr_verify = true;
                else if (std::strcmp(val, "pubkey") == 0)   cfg.case_pubkey_create = true;
                else if (std::strcmp(val, "ecdh") == 0)     cfg.case_ecdh = true;
                else {
                    std::fprintf(stderr, "Unknown case: %s\n", val);
                    return false;
                }
            }
        }
        else if (starts_with(arg, "--n=")) {
            cfg.dataset_size = std::strtoull(after_eq(arg), nullptr, 10);
        }
        else if (starts_with(arg, "--seed=")) {
            cfg.seed = std::strtoull(after_eq(arg), nullptr, 10);
        }
        else if (starts_with(arg, "--warmup=")) {
            cfg.warmup_ms = std::atoi(after_eq(arg));
        }
        else if (starts_with(arg, "--measure=")) {
            cfg.measure_ms = std::atoi(after_eq(arg));
        }
        else if (starts_with(arg, "--pin-core=")) {
            cfg.pin_core = std::atoi(after_eq(arg));
        }
        else if (starts_with(arg, "--msg=")) {
            const char* val = after_eq(arg);
            if (std::strcmp(val, "prehashed32") == 0)    cfg.msg_policy = MsgPolicy::PREHASHED32;
            else if (std::strcmp(val, "sha256d") == 0)   cfg.msg_policy = MsgPolicy::SHA256D_RAW;
            else { std::fprintf(stderr, "Unknown --msg=%s\n", val); return false; }
        }
        else if (starts_with(arg, "--ecdsa-sig=")) {
            const char* val = after_eq(arg);
            if (std::strcmp(val, "der") == 0)            cfg.sig_encoding = SigEncoding::DER;
            else if (std::strcmp(val, "compact") == 0)   cfg.sig_encoding = SigEncoding::COMPACT;
            else { std::fprintf(stderr, "Unknown --ecdsa-sig=%s\n", val); return false; }
        }
        else if (starts_with(arg, "--libsecp-randomize=")) {
            cfg.libsecp_randomize = (std::atoi(after_eq(arg)) != 0);
        }
        else if (starts_with(arg, "--json=")) {
            cfg.report_json = after_eq(arg);
        }
        else {
            std::fprintf(stderr, "Unknown option: %s\n", arg);
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

} // namespace bench
