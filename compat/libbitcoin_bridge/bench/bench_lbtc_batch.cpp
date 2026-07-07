/**
 * bench_lbtc_batch.cpp — throughput benchmark for the libbitcoin batch
 * script-signature verification bridge (ufsecp_lbtc_verify_ecdsa / _schnorr).
 *
 * Measures verifications/second for a large homogeneous batch on whichever
 * backends are available (GPU if built + present, and the CPU reference). This
 * mirrors the IBD use case: a big array of (sig, key, sighash) triples verified
 * in one call. ECDSA records use copied libsecp256k1_ecdsa_signature storage,
 * not compact R||S, so the timed row path matches libbitcoin's ec_signature
 * table shape. Correctness is asserted (all-valid batch + single-corruption
 * detection) before any timing, so reported numbers are for a verified-correct
 * path.
 *
 * Usage: bench_lbtc_batch [batch_size] [iters] [pool] [--json path]
 *   batch_size  rows verified per call      (default 1000000)
 *   iters       timed iterations per backend (default 5)
 *   pool        distinct signatures generated, tiled to batch_size (default 50000)
 *   --json path write a machine-readable benchmark artifact
 *
 * NOTE: numbers are only meaningful as measured on THIS machine. Do not copy
 * them anywhere as estimates for other hardware.
 */
#include "ufsecp_libbitcoin.h"
#include "ufsecp.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>

using clock_t_ = std::chrono::steady_clock;

static double secs_since(clock_t_::time_point t0) {
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

struct BenchResult {
    std::string requested;
    std::string bound;
    std::string device;
    std::string kind;
    size_t batch = 0;
    int iters = 0;
    size_t pool = 0;
    double best_seconds = 0.0;
    double m_sig_per_sec = 0.0;
};

static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch : s) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"':  out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:   out += ch; break;
        }
    }
    return out;
}

static bool write_json_artifact(const std::string& path,
                                const std::vector<BenchResult>& results,
                                size_t batch, int iters, size_t pool) {
    std::ofstream out(path);
    if (!out) return false;
    out << std::setprecision(12);
    out << "{\n";
    out << "  \"schema\": \"ufsecp-lbtc-benchmark-v1\",\n";
    out << "  \"target_context\": \"libbitcoin\",\n";
    out << "  \"claim_scope\": \"local libbitcoin batch-verify bridge throughput\",\n";
    out << "  \"security_gate_dependency\": \"python3 ci/audit_gate.py --libbitcoin-perf-matrix\",\n";
    out << "  \"batch_size\": " << batch << ",\n";
    out << "  \"iters\": " << iters << ",\n";
    out << "  \"pool\": " << pool << ",\n";
    out << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"requested_backend\": \"" << json_escape(r.requested) << "\",\n";
        out << "      \"bound_backend\": \"" << json_escape(r.bound) << "\",\n";
        out << "      \"device\": \"" << json_escape(r.device) << "\",\n";
        out << "      \"kind\": \"" << json_escape(r.kind) << "\",\n";
        out << "      \"batch_size\": " << r.batch << ",\n";
        out << "      \"iters\": " << r.iters << ",\n";
        out << "      \"pool\": " << r.pool << ",\n";
        out << "      \"best_seconds\": " << r.best_seconds << ",\n";
        out << "      \"m_sig_per_sec\": " << r.m_sig_per_sec << "\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.good();
}

int main(int argc, char** argv) {
    std::vector<const char*> positional;
    std::string json_path;
    bool mt_only = false;  /* --mt-only: run ONLY the _mt thread-scaling sweep */
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "--json requires a path\n");
                return 2;
            }
            json_path = argv[i];
        } else if (std::strcmp(argv[i], "--mt-only") == 0) {
            mt_only = true;
        } else {
            positional.push_back(argv[i]);
        }
    }

    const size_t BATCH = positional.size() > 0 ? std::strtoull(positional[0], nullptr, 10) : 1000000ull;
    const int    ITERS = positional.size() > 1 ? std::atoi(positional[1]) : 5;
    const size_t POOL  = positional.size() > 2 ? std::strtoull(positional[2], nullptr, 10) : 50000ull;
    if (BATCH == 0 || ITERS <= 0 || POOL == 0) {
        std::fprintf(stderr, "batch_size, iters, and pool must be positive\n");
        return 2;
    }

    std::printf("== libbitcoin batch sig-verify benchmark ==\n");
    std::printf("batch=%zu  iters=%d  pool=%zu\n\n", BATCH, ITERS, POOL);

    ufsecp_ctx* sctx = nullptr;
    if (ufsecp_ctx_create(&sctx) != UFSECP_OK) { std::printf("ctx fail\n"); return 1; }

    /* --- generate POOL distinct ECDSA + Schnorr records, then tile to BATCH --- */
    std::printf("generating %zu signatures...\n", POOL);
    std::vector<uint8_t> e_pool(POOL * UFSECP_LBTC_ECDSA_RECORD);
    std::vector<uint8_t> s_pool(POOL * UFSECP_LBTC_SCHNORR_RECORD);
    for (size_t i = 0; i < POOL; ++i) {
        uint8_t sk[32] = {0}, msg[32] = {0}, pub[33], aux[32] = {0};
        sk[24] = (uint8_t)(i >> 24); sk[25] = (uint8_t)(i >> 16);
        sk[26] = (uint8_t)(i >> 8);  sk[31] = (uint8_t)(i | 1u);
        for (int b = 0; b < 32; ++b) msg[b] = (uint8_t)((i * 2654435761u) >> (b % 24));
        if (ufsecp_pubkey_create(sctx, sk, pub) != UFSECP_OK) { std::printf("keygen fail\n"); return 1; }
        uint8_t* er = e_pool.data() + i * UFSECP_LBTC_ECDSA_RECORD;
        std::memcpy(er, msg, 32); std::memcpy(er + 32, pub, 33);
        uint8_t compact_sig[64];
        if (ufsecp_ecdsa_sign(sctx, msg, sk, compact_sig) != UFSECP_OK) { std::printf("ecdsa sign fail\n"); return 1; }
        if (ufsecp_ecdsa_sig_compact_to_opaque(sctx, compact_sig, er + 65) != UFSECP_OK) {
            std::printf("ecdsa opaque conversion fail\n");
            return 1;
        }
        uint8_t* sr = s_pool.data() + i * UFSECP_LBTC_SCHNORR_RECORD;
        std::memcpy(sr, msg, 32); std::memcpy(sr + 32, pub + 1, 32); /* msg | xonly */
        if (ufsecp_schnorr_sign(sctx, msg, sk, aux, sr + 64) != UFSECP_OK) { std::printf("schnorr sign fail\n"); return 1; }
    }

    /* Tile pool -> full batch tables. Build both supported bridge layouts:
     * packed rows (horizontal) and independent msg/pub/sig columns (vertical). */
    std::printf("building batch tables (%zu rows each)...\n\n", BATCH);
    std::vector<uint8_t> e_rows(BATCH * UFSECP_LBTC_ECDSA_RECORD);
    std::vector<uint8_t> s_rows(BATCH * UFSECP_LBTC_SCHNORR_RECORD);
    std::vector<uint8_t> e_msg(BATCH * 32), e_pub(BATCH * 33), e_sig(BATCH * 64);
    std::vector<uint8_t> s_msg(BATCH * 32), s_pub(BATCH * 32), s_sig(BATCH * 64);
    for (size_t i = 0; i < BATCH; ++i) {
        size_t p = i % POOL;
        const uint8_t* er = e_pool.data() + p * UFSECP_LBTC_ECDSA_RECORD;
        const uint8_t* sr = s_pool.data() + p * UFSECP_LBTC_SCHNORR_RECORD;
        std::memcpy(e_rows.data() + i * UFSECP_LBTC_ECDSA_RECORD,
                    er, UFSECP_LBTC_ECDSA_RECORD);
        std::memcpy(s_rows.data() + i * UFSECP_LBTC_SCHNORR_RECORD,
                    sr, UFSECP_LBTC_SCHNORR_RECORD);
        std::memcpy(e_msg.data() + i * 32, er, 32);
        std::memcpy(e_pub.data() + i * 33, er + 32, 33);
        std::memcpy(e_sig.data() + i * 64, er + 65, 64);
        std::memcpy(s_msg.data() + i * 32, sr, 32);
        std::memcpy(s_pub.data() + i * 32, sr + 32, 32);
        std::memcpy(s_sig.data() + i * 64, sr + 64, 64);
    }
    ufsecp_ctx_destroy(sctx);

    std::vector<uint8_t> results(BATCH);
    const char* be_name[] = {"CPU", "CUDA", "OpenCL", "Metal"};
    std::vector<BenchResult> bench_results;

    struct Run { ufsecp_lbtc_backend req; const char* label; };
    Run runs[] = { {UFSECP_LBTC_GPU, "GPU"}, {UFSECP_LBTC_CPU, "CPU"} };

    for (auto& r : runs) {
        ufsecp_lbtc_ctrl* ctrl = nullptr;
        if (ufsecp_lbtc_ctrl_create(&ctrl, r.req) != UFSECP_OK || !ctrl) {
            std::printf("[%s] backend unavailable — skipped\n\n", r.label);
            continue;
        }
        const char* bound = be_name[ufsecp_lbtc_ctrl_backend(ctrl)];
        const char* device = ufsecp_lbtc_ctrl_device_name(ctrl);
        std::printf("[%s] bound=%s device=%s\n", r.label, bound,
                    device);

        /* correctness gate before timing — failures come from results[] (the
         * bridge returns void; the caller counts invalids itself). */
        auto count_invalid = [&]() { size_t c = 0; for (auto v : results) if (!v) ++c; return c; };
        ufsecp_lbtc_verify_ecdsa(ctrl, e_rows.data(), BATCH, 0, results.data(), nullptr, 0, nullptr);
        bool ok = (count_invalid() == 0);
        {
            auto saved = e_rows[65]; e_rows[65] ^= 0x01;  // corrupt row 0 sig
            ufsecp_lbtc_verify_ecdsa(ctrl, e_rows.data(), BATCH, 0, results.data(), nullptr, 0, nullptr);
            ok = ok && (count_invalid() >= 1) && (results[0] == 0);
            e_rows[65] = saved;
        }
        /* schnorr correctness gate (mirrors ecdsa; sig byte at record offset 64) */
        {
            ufsecp_lbtc_verify_schnorr(ctrl, s_rows.data(), BATCH, 0, results.data(), nullptr, 0, nullptr);
            bool sok = (count_invalid() == 0);
            auto saved = s_rows[64]; s_rows[64] ^= 0x01;  // corrupt row 0 sig
            ufsecp_lbtc_verify_schnorr(ctrl, s_rows.data(), BATCH, 0, results.data(), nullptr, 0, nullptr);
            sok = sok && (count_invalid() >= 1) && (results[0] == 0);
            s_rows[64] = saved;
            ok = ok && sok;
            std::printf("   schnorr correctness: %s\n", sok ? "PASS" : "FAIL");
        }
        {
            ufsecp_lbtc_verify_ecdsa_columns(
                ctrl, e_msg.data(), e_pub.data(), e_sig.data(), BATCH, results.data());
            bool cok = (count_invalid() == 0);
            auto saved = e_sig[0]; e_sig[0] ^= 0x01;
            ufsecp_lbtc_verify_ecdsa_columns(
                ctrl, e_msg.data(), e_pub.data(), e_sig.data(), BATCH, results.data());
            cok = cok && (count_invalid() >= 1) && (results[0] == 0);
            e_sig[0] = saved;
            ufsecp_lbtc_verify_schnorr_columns(
                ctrl, s_msg.data(), s_pub.data(), s_sig.data(), BATCH, results.data());
            bool scok = (count_invalid() == 0);
            saved = s_sig[0]; s_sig[0] ^= 0x01;
            ufsecp_lbtc_verify_schnorr_columns(
                ctrl, s_msg.data(), s_pub.data(), s_sig.data(), BATCH, results.data());
            scok = scok && (count_invalid() >= 1) && (results[0] == 0);
            s_sig[0] = saved;
            ok = ok && cok && scok;
            std::printf("   column correctness: %s\n", (cok && scok) ? "PASS" : "FAIL");
        }
        std::printf("   correctness: %s\n", ok ? "PASS (all-valid + corruption detected)" : "FAIL");
        if (!ok) { ufsecp_lbtc_ctrl_destroy(ctrl); continue; }

        auto record_result = [&](const char* kind, double best) {
            bench_results.push_back(BenchResult{
                r.label, bound, device, kind, BATCH, ITERS, POOL, best,
                (double)BATCH / best / 1e6
            });
        };
        auto bench = [&](const char* kind, auto verify, const std::vector<uint8_t>& rows) {
            verify(ctrl, rows.data(), BATCH, (size_t)0, results.data(), nullptr, 0, nullptr, nullptr); // warmup
            double best = 1e30;
            for (int it = 0; it < ITERS; ++it) {
                auto t0 = clock_t_::now();
                verify(ctrl, rows.data(), BATCH, (size_t)0, results.data(), nullptr, 0, nullptr, nullptr);
                double dt = secs_since(t0);
                if (dt < best) best = dt;
            }
            double mps = (double)BATCH / best / 1e6;
            std::printf("   %-8s %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                        kind, mps, best, BATCH, ITERS);
            record_result(kind, best);
        };
        auto bench_columns = [&](const char* kind, auto verify,
                                 const std::vector<uint8_t>& msg,
                                 const std::vector<uint8_t>& pub,
                                 const std::vector<uint8_t>& sig) {
            verify(ctrl, msg.data(), pub.data(), sig.data(), BATCH, results.data(), nullptr); // warmup
            double best = 1e30;
            for (int it = 0; it < ITERS; ++it) {
                auto t0 = clock_t_::now();
                verify(ctrl, msg.data(), pub.data(), sig.data(), BATCH, results.data(), nullptr);
                double dt = secs_since(t0);
                if (dt < best) best = dt;
            }
            const double mps = (double)BATCH / best / 1e6;
            std::printf("   %-14s %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                        kind, mps, best, BATCH, ITERS);
            record_result(kind, best);
        };
      if (!mt_only) {
        bench("ECDSA-row", ufsecp_lbtc_verify_ecdsa, e_rows);
        bench("Schnorr-row", ufsecp_lbtc_verify_schnorr, s_rows);
        bench_columns("ECDSA-columns", ufsecp_lbtc_verify_ecdsa_columns,
                      e_msg, e_pub, e_sig);
        bench_columns("Schnorr-columns", ufsecp_lbtc_verify_schnorr_columns,
                      s_msg, s_pub, s_sig);

        /* In-place "collect" path (key_size=4 mutable rows). On the GPU this uses
         * the dedicated on-device collect kernel by default; build with
         * -DUFSECP_LBTC_DISABLE_DEDICATED for the host-collapse control arm — the
         * A/B isolates the dedicated kernel + removed host scatter loop. The
         * collect call re-verifies every row each iteration and the key-cell
         * zeroing is idempotent, so per-iteration work is identical. */
        {
            const size_t KS = 4;
            const size_t es = UFSECP_LBTC_ECDSA_RECORD + KS;
            const size_t ss = UFSECP_LBTC_SCHNORR_RECORD + KS;
            std::vector<uint8_t> ec(BATCH * es), sc(BATCH * ss);
            std::vector<uint8_t> e_key(BATCH * KS), s_key(BATCH * KS);
            for (size_t i = 0; i < BATCH; ++i) {
                std::memcpy(ec.data() + i * es,
                            e_rows.data() + i * UFSECP_LBTC_ECDSA_RECORD,
                            UFSECP_LBTC_ECDSA_RECORD);
                std::memcpy(sc.data() + i * ss,
                            s_rows.data() + i * UFSECP_LBTC_SCHNORR_RECORD,
                            UFSECP_LBTC_SCHNORR_RECORD);
                const uint64_t id = i + 1; /* non-zero id in the key cell */
                for (size_t b = 0; b < KS; ++b) {
                    ec[i * es + UFSECP_LBTC_ECDSA_RECORD + b]   = (uint8_t)(id >> (8 * b));
                    sc[i * ss + UFSECP_LBTC_SCHNORR_RECORD + b] = (uint8_t)(id >> (8 * b));
                    e_key[i * KS + b] = (uint8_t)(id >> (8 * b));
                    s_key[i * KS + b] = (uint8_t)(id >> (8 * b));
                }
            }
            auto bench_collect = [&](const char* kind, auto collect, std::vector<uint8_t>& rows) {
                collect(ctrl, rows.data(), BATCH, KS, nullptr); /* warmup */
                double best = 1e30;
                for (int it = 0; it < ITERS; ++it) {
                    auto t0 = clock_t_::now();
                    collect(ctrl, rows.data(), BATCH, KS, nullptr);
                    double dt = secs_since(t0);
                    if (dt < best) best = dt;
                }
                std::printf("   %-14s %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                            kind, (double)BATCH / best / 1e6, best, BATCH, ITERS);
                record_result(kind, best);
            };
            auto bench_collect_columns = [&](const char* kind, auto collect,
                                             const std::vector<uint8_t>& msg,
                                             const std::vector<uint8_t>& pub,
                                             const std::vector<uint8_t>& sig,
                                             std::vector<uint8_t>& key) {
                collect(ctrl, msg.data(), pub.data(), sig.data(), BATCH, key.data(), KS, nullptr);
                double best = 1e30;
                for (int it = 0; it < ITERS; ++it) {
                    auto t0 = clock_t_::now();
                    collect(ctrl, msg.data(), pub.data(), sig.data(), BATCH, key.data(), KS, nullptr);
                    double dt = secs_since(t0);
                    if (dt < best) best = dt;
                }
                const double mps = (double)BATCH / best / 1e6;
                std::printf("   %-14s %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                            kind, mps, best, BATCH, ITERS);
                record_result(kind, best);
            };
#ifdef UFSECP_LBTC_DISABLE_DEDICATED
            std::printf("   [collect arm: host-collapse control (UFSECP_LBTC_DISABLE_DEDICATED)]\n");
#else
            std::printf("   [collect arm: dedicated on-device kernel]\n");
#endif
            bench_collect("ECDSA-collect",   ufsecp_lbtc_verify_ecdsa_collect,   ec);
            bench_collect("Schnorr-collect", ufsecp_lbtc_verify_schnorr_collect, sc);
            bench_collect_columns("ECDSA-col-collect",
                                  ufsecp_lbtc_verify_ecdsa_columns_collect,
                                  e_msg, e_pub, e_sig, e_key);
            bench_collect_columns("Schnorr-col-collect",
                                  ufsecp_lbtc_verify_schnorr_columns_collect,
                                  s_msg, s_pub, s_sig, s_key);
        }
      } // if (!mt_only)

        /* ── MT scaling sweep — the dev 07a1ece persistent-pool + fused-parse _mt
         * path. The serial benches above call the NON-mt entry points
         * (max_threads == 1). These _mt twins fan the CPU verify across a worker
         * budget on a process-wide pool with fused parse+verify — the actual
         * subject of the perf commit, and the apples-to-apples match for the Linux
         * measurement. Sweep the budget to show parallel speedup over serial on
         * THIS machine. Row API (opaque ECDSA / BIP-340 Schnorr) only — that is
         * what a libbitcoin node calls. */
        {
            unsigned hw = std::thread::hardware_concurrency();
            if (hw == 0) hw = 8;
            size_t threadset[] = {1, 2, 4, 8, (size_t)hw};
            auto bench_mt = [&](const char* kind, auto verify_mt,
                                const std::vector<uint8_t>& rows) {
                for (size_t t : threadset) {
                    verify_mt(ctrl, rows.data(), BATCH, (size_t)0, results.data(),
                              nullptr, (size_t)0, nullptr, t, nullptr); // warmup
                    double best = 1e30;
                    for (int it = 0; it < ITERS; ++it) {
                        auto t0 = clock_t_::now();
                        verify_mt(ctrl, rows.data(), BATCH, (size_t)0, results.data(),
                                  nullptr, (size_t)0, nullptr, t, nullptr);
                        double dt = secs_since(t0);
                        if (dt < best) best = dt;
                    }
                    const double mps = (double)BATCH / best / 1e6;
                    std::printf("   %-15s threads=%-2zu %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                                kind, t, mps, best, BATCH, ITERS);
                    char label[48];
                    std::snprintf(label, sizeof(label), "%s-t%zu", kind, t);
                    record_result(label, best);
                }
            };
            std::printf("   [MT scaling sweep — _mt persistent pool + fused parse]\n");
            bench_mt("ECDSA-row-mt", ufsecp_lbtc_verify_ecdsa_mt, e_rows);
            bench_mt("Schnorr-row-mt", ufsecp_lbtc_verify_schnorr_mt, s_rows);
        }
        std::printf("\n");
        ufsecp_lbtc_ctrl_destroy(ctrl);
    }
    if (!json_path.empty()) {
        if (!write_json_artifact(json_path, bench_results, BATCH, ITERS, POOL)) {
            std::fprintf(stderr, "failed to write JSON benchmark artifact: %s\n", json_path.c_str());
            return 1;
        }
        std::printf("wrote JSON artifact: %s\n", json_path.c_str());
    }
    return 0;
}
