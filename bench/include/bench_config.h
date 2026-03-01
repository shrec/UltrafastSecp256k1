// bench_config.h -- CLI configuration for bench_compare
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace bench {

enum class MsgPolicy { PREHASHED32, SHA256D_RAW };
enum class SigEncoding { DER, COMPACT };

struct BenchConfig {
    // -- provider selection -------------------------------------------------
    bool run_uf      = true;
    bool run_libsecp = true;

    // -- case selection -----------------------------------------------------
    bool case_ecdsa_verify   = true;
    bool case_schnorr_verify = true;
    bool case_pubkey_create  = true;
    bool case_ecdh           = false;

    // -- dataset ------------------------------------------------------------
    size_t   dataset_size = 100000;
    uint64_t seed         = 42;

    // -- timing -------------------------------------------------------------
    int warmup_ms  = 250;
    int measure_ms = 2000;

    // -- CPU affinity -------------------------------------------------------
    int pin_core = -1;  // -1 = disabled

    // -- encoding policy ----------------------------------------------------
    MsgPolicy   msg_policy  = MsgPolicy::PREHASHED32;
    SigEncoding sig_encoding = SigEncoding::DER;

    // -- libsecp options ----------------------------------------------------
    bool libsecp_randomize = true;

    // -- output -------------------------------------------------------------
    std::string report_json;  // empty = no JSON output
    std::string report_table; // empty = no markdown file (still prints to stdout)
};

// Parse CLI arguments into BenchConfig. Returns false on parse error.
bool parse_args(BenchConfig& cfg, int argc, char** argv);

// Print usage/help.
void print_usage(const char* progname);

} // namespace bench
