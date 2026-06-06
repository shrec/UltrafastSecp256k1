/**
 * bench_lbtc_batch.cpp — throughput benchmark for the libbitcoin batch
 * script-signature verification bridge (ufsecp_lbtc_verify_ecdsa / _schnorr).
 *
 * Measures verifications/second for a large homogeneous batch on whichever
 * backends are available (GPU if built + present, and the CPU reference). This
 * mirrors the IBD use case: a big array of (sig, key, sighash) triples verified
 * in one call. Correctness is asserted (all-valid batch + single-corruption
 * detection) before any timing, so reported numbers are for a verified-correct
 * path.
 *
 * Usage: bench_lbtc_batch [batch_size] [iters] [pool]
 *   batch_size  rows verified per call      (default 1000000)
 *   iters       timed iterations per backend (default 5)
 *   pool        distinct signatures generated, tiled to batch_size (default 50000)
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
#include <vector>

using clock_t_ = std::chrono::steady_clock;

static double secs_since(clock_t_::time_point t0) {
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

int main(int argc, char** argv) {
    const size_t BATCH = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1000000ull;
    const int    ITERS = argc > 2 ? std::atoi(argv[2]) : 5;
    const size_t POOL  = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 50000ull;

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
        if (ufsecp_ecdsa_sign(sctx, msg, sk, er + 65) != UFSECP_OK) { std::printf("ecdsa sign fail\n"); return 1; }
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

    struct Run { ufsecp_lbtc_backend req; const char* label; };
    Run runs[] = { {UFSECP_LBTC_GPU, "GPU"}, {UFSECP_LBTC_CPU, "CPU"} };

    for (auto& r : runs) {
        ufsecp_lbtc_ctrl* ctrl = nullptr;
        if (ufsecp_lbtc_ctrl_create(&ctrl, r.req) != UFSECP_OK || !ctrl) {
            std::printf("[%s] backend unavailable — skipped\n\n", r.label);
            continue;
        }
        const char* bound = be_name[ufsecp_lbtc_ctrl_backend(ctrl)];
        std::printf("[%s] bound=%s device=%s\n", r.label, bound,
                    ufsecp_lbtc_ctrl_device_name(ctrl));

        /* correctness gate before timing — failures come from results[] (the
         * bridge returns void; the caller counts invalids itself). */
        auto count_invalid = [&]() { size_t c = 0; for (auto v : results) if (!v) ++c; return c; };
        ufsecp_lbtc_verify_ecdsa(ctrl, e_rows.data(), BATCH, 0, results.data());
        bool ok = (count_invalid() == 0);
        {
            auto saved = e_rows[65]; e_rows[65] ^= 0x01;  // corrupt row 0 sig
            ufsecp_lbtc_verify_ecdsa(ctrl, e_rows.data(), BATCH, 0, results.data());
            ok = ok && (count_invalid() >= 1) && (results[0] == 0);
            e_rows[65] = saved;
        }
        /* schnorr correctness gate (mirrors ecdsa; sig byte at record offset 64) */
        {
            ufsecp_lbtc_verify_schnorr(ctrl, s_rows.data(), BATCH, 0, results.data());
            bool sok = (count_invalid() == 0);
            auto saved = s_rows[64]; s_rows[64] ^= 0x01;  // corrupt row 0 sig
            ufsecp_lbtc_verify_schnorr(ctrl, s_rows.data(), BATCH, 0, results.data());
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

        auto bench = [&](const char* kind, auto verify, const std::vector<uint8_t>& rows) {
            verify(ctrl, rows.data(), BATCH, (size_t)0, results.data()); // warmup
            double best = 1e30;
            for (int it = 0; it < ITERS; ++it) {
                auto t0 = clock_t_::now();
                verify(ctrl, rows.data(), BATCH, (size_t)0, results.data());
                double dt = secs_since(t0);
                if (dt < best) best = dt;
            }
            double mps = (double)BATCH / best / 1e6;
            std::printf("   %-8s %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                        kind, mps, best, BATCH, ITERS);
        };
        auto bench_columns = [&](const char* kind, auto verify,
                                 const std::vector<uint8_t>& msg,
                                 const std::vector<uint8_t>& pub,
                                 const std::vector<uint8_t>& sig) {
            verify(ctrl, msg.data(), pub.data(), sig.data(), BATCH, results.data()); // warmup
            double best = 1e30;
            for (int it = 0; it < ITERS; ++it) {
                auto t0 = clock_t_::now();
                verify(ctrl, msg.data(), pub.data(), sig.data(), BATCH, results.data());
                double dt = secs_since(t0);
                if (dt < best) best = dt;
            }
            std::printf("   %-14s %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                        kind, (double)BATCH / best / 1e6, best, BATCH, ITERS);
        };
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
                collect(ctrl, rows.data(), BATCH, KS); /* warmup */
                double best = 1e30;
                for (int it = 0; it < ITERS; ++it) {
                    auto t0 = clock_t_::now();
                    collect(ctrl, rows.data(), BATCH, KS);
                    double dt = secs_since(t0);
                    if (dt < best) best = dt;
                }
                std::printf("   %-14s %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                            kind, (double)BATCH / best / 1e6, best, BATCH, ITERS);
            };
            auto bench_collect_columns = [&](const char* kind, auto collect,
                                             const std::vector<uint8_t>& msg,
                                             const std::vector<uint8_t>& pub,
                                             const std::vector<uint8_t>& sig,
                                             std::vector<uint8_t>& key) {
                collect(ctrl, msg.data(), pub.data(), sig.data(), BATCH, key.data(), KS);
                double best = 1e30;
                for (int it = 0; it < ITERS; ++it) {
                    auto t0 = clock_t_::now();
                    collect(ctrl, msg.data(), pub.data(), sig.data(), BATCH, key.data(), KS);
                    double dt = secs_since(t0);
                    if (dt < best) best = dt;
                }
                std::printf("   %-14s %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
                            kind, (double)BATCH / best / 1e6, best, BATCH, ITERS);
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
        std::printf("\n");
        ufsecp_lbtc_ctrl_destroy(ctrl);
    }
    return 0;
}
