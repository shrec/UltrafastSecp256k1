// Direct C++ libbitcoin batch benchmark.
//
// This is the canonical libbitcoin evidence path: no ufsecp C ABI, no
// libsecp256k1 shim, no ufsecp_lbtc bridge. It links the direct C++ interface
// target and calls ufsecp::lbtc::* inline functions against libbitcoin-shaped
// row and column tables.
//
// Usage: bench_lbtc_direct_batch [batch_size] [iters] [pool] [--json path]
#include "ufsecp/libbitcoin.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <string>
#include <thread>
#include <vector>

namespace {

using clock_t_ = std::chrono::steady_clock;

struct BenchResult {
    std::string surface;
    std::string kind;
    std::size_t batch = 0;
    int iters = 0;
    std::size_t pool = 0;
    std::size_t max_threads = 0;
    double best_seconds = 0.0;
    double m_sig_per_sec = 0.0;
};

std::uint64_t g_xs = 0x9E3779B97F4A7C15ull;

std::uint8_t nb() noexcept
{
    g_xs ^= g_xs << 13;
    g_xs ^= g_xs >> 7;
    g_xs ^= g_xs << 17;
    return static_cast<std::uint8_t>(g_xs);
}

double secs_since(clock_t_::time_point t0)
{
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

std::string json_escape(const std::string& s)
{
    std::string out;
    out.reserve(s.size() + 8);
    for (char ch: s) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default: out += ch; break;
        }
    }
    return out;
}

void rand_hash(std::uint8_t h[32]) noexcept
{
    for (int i = 0; i < 32; ++i)
        h[i] = nb();
}

void rand_sk(std::uint8_t sk[32]) noexcept
{
    do {
        for (int i = 0; i < 32; ++i)
            sk[i] = nb();
    } while (!ufsecp::lbtc::seckey_verify(sk));
}

std::size_t count_invalid(const std::vector<std::uint8_t>& results) noexcept
{
    std::size_t invalid = 0;
    for (const auto v: results)
        if (v == 0)
            ++invalid;
    return invalid;
}

bool write_json_artifact(const std::string& path,
    const std::vector<BenchResult>& results, std::size_t batch, int iters,
    std::size_t pool)
{
    std::ofstream out(path);
    if (!out)
        return false;

    out << std::setprecision(12);
    out << "{\n";
    out << "  \"schema\": \"ufsecp-lbtc-direct-benchmark-v1\",\n";
    out << "  \"target_context\": \"libbitcoin-direct-cpp\",\n";
    out << "  \"claim_scope\": \"local direct C++ libbitcoin batch-verify throughput\",\n";
    out << "  \"c_abi_required\": false,\n";
    out << "  \"shim_required\": false,\n";
    out << "  \"bridge_required\": false,\n";
    out << "  \"batch_size\": " << batch << ",\n";
    out << "  \"iters\": " << iters << ",\n";
    out << "  \"pool\": " << pool << ",\n";
    out << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"surface\": \"" << json_escape(r.surface) << "\",\n";
        out << "      \"kind\": \"" << json_escape(r.kind) << "\",\n";
        out << "      \"batch_size\": " << r.batch << ",\n";
        out << "      \"iters\": " << r.iters << ",\n";
        out << "      \"pool\": " << r.pool << ",\n";
        out << "      \"max_threads\": " << r.max_threads << ",\n";
        out << "      \"best_seconds\": " << r.best_seconds << ",\n";
        out << "      \"m_sig_per_sec\": " << r.m_sig_per_sec << "\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.good();
}

} // namespace

int main(int argc, char** argv)
{
    std::vector<const char*> positional;
    std::string json_path;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "--json requires a path\n");
                return 2;
            }
            json_path = argv[i];
        } else {
            positional.push_back(argv[i]);
        }
    }

    const std::size_t batch = positional.size() > 0 ?
        std::strtoull(positional[0], nullptr, 10) : 1000000ull;
    const int iters = positional.size() > 1 ?
        std::atoi(positional[1]) : 5;
    const std::size_t pool = positional.size() > 2 ?
        std::strtoull(positional[2], nullptr, 10) : 50000ull;

    if (batch == 0 || iters <= 0 || pool == 0) {
        std::fprintf(stderr, "batch_size, iters, and pool must be positive\n");
        return 2;
    }

    std::printf("== libbitcoin direct C++ batch verify benchmark ==\n");
    std::printf("batch=%zu  iters=%d  pool=%zu\n", batch, iters, pool);
    std::printf("surface=ufsecp::lbtc direct header  c_abi=no shim=no bridge=no\n\n");

    constexpr std::size_t ecdsa_stride = 129;
    constexpr std::size_t schnorr_stride = 128;
    std::uint8_t aux[32]{};

    std::printf("generating %zu signatures...\n", pool);
    std::vector<std::uint8_t> e_pool(pool * ecdsa_stride);
    std::vector<std::uint8_t> s_pool(pool * schnorr_stride);
    for (std::size_t i = 0; i < pool; ++i) {
        std::uint8_t sk[32], msg[32], pub33[33], xonly[32], sig64[64];
        rand_sk(sk);
        rand_hash(msg);

        if (!ufsecp::lbtc::pubkey_create(sk, pub33)) {
            std::printf("ecdsa pubkey_create fail\n");
            return 1;
        }
        if (!ufsecp::lbtc::ecdsa_sign(msg, sk, sig64)) {
            std::printf("ecdsa sign fail\n");
            return 1;
        }
        auto* er = e_pool.data() + i * ecdsa_stride;
        std::memcpy(er, msg, 32);
        std::memcpy(er + 32, pub33, 33);
        std::memcpy(er + 65, sig64, 64);

        if (!ufsecp::lbtc::schnorr_keypair_create(sk, xonly)) {
            std::printf("schnorr keypair_create fail\n");
            return 1;
        }
        if (!ufsecp::lbtc::schnorr_sign(xonly, sk, msg, aux, sig64)) {
            std::printf("schnorr sign fail\n");
            return 1;
        }
        auto* sr = s_pool.data() + i * schnorr_stride;
        std::memcpy(sr, msg, 32);
        std::memcpy(sr + 32, xonly, 32);
        std::memcpy(sr + 64, sig64, 64);
    }

    std::printf("building direct row/column tables...\n\n");
    std::vector<std::uint8_t> e_rows(batch * ecdsa_stride);
    std::vector<std::uint8_t> s_rows(batch * schnorr_stride);
    std::vector<std::uint8_t> e_msg(batch * 32), e_pub(batch * 33), e_sig(batch * 64);
    std::vector<std::uint8_t> s_msg(batch * 32), s_pub(batch * 32), s_sig(batch * 64);
    for (std::size_t i = 0; i < batch; ++i) {
        const std::size_t p = i % pool;
        const auto* er = e_pool.data() + p * ecdsa_stride;
        const auto* sr = s_pool.data() + p * schnorr_stride;
        std::memcpy(e_rows.data() + i * ecdsa_stride, er, ecdsa_stride);
        std::memcpy(s_rows.data() + i * schnorr_stride, sr, schnorr_stride);
        std::memcpy(e_msg.data() + i * 32, er, 32);
        std::memcpy(e_pub.data() + i * 33, er + 32, 33);
        std::memcpy(e_sig.data() + i * 64, er + 65, 64);
        std::memcpy(s_msg.data() + i * 32, sr, 32);
        std::memcpy(s_pub.data() + i * 32, sr + 32, 32);
        std::memcpy(s_sig.data() + i * 64, sr + 64, 64);
    }

    std::vector<std::uint8_t> results(batch, 0);
    std::vector<BenchResult> bench_results;
    const auto hw = std::thread::hardware_concurrency();
    const std::size_t auto_threads = hw == 0 ? 0 : static_cast<std::size_t>(hw);

    auto record = [&](const char* kind, std::size_t max_threads, double best) {
        bench_results.push_back(BenchResult{
            "ufsecp::lbtc", kind, batch, iters, pool, max_threads, best,
            static_cast<double>(batch) / best / 1e6
        });
    };

    auto bench_rows = [&](const char* kind, auto verify, const std::vector<std::uint8_t>& rows,
                          std::size_t stride, std::size_t max_threads,
                          std::size_t recorded_threads) {
        if (!verify(rows.data(), stride, batch, results.data(), max_threads)) {
            std::printf("   %-18s correctness warmup failed\n", kind);
            return false;
        }
        double best = 1e30;
        for (int it = 0; it < iters; ++it) {
            const auto t0 = clock_t_::now();
            if (!verify(rows.data(), stride, batch, results.data(), max_threads))
                return false;
            const auto dt = secs_since(t0);
            if (dt < best)
                best = dt;
        }
        std::printf("   %-18s threads=%-2zu %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
            kind, recorded_threads, static_cast<double>(batch) / best / 1e6, best, batch, iters);
        record(kind, recorded_threads, best);
        return true;
    };

    auto bench_columns = [&](const char* kind, auto verify,
                             const std::vector<std::uint8_t>& msg,
                             const std::vector<std::uint8_t>& pub,
                             const std::vector<std::uint8_t>& sig,
                             std::size_t max_threads,
                             std::size_t recorded_threads) {
        if (!verify(msg.data(), pub.data(), sig.data(), batch, results.data(), max_threads)) {
            std::printf("   %-18s correctness warmup failed\n", kind);
            return false;
        }
        double best = 1e30;
        for (int it = 0; it < iters; ++it) {
            const auto t0 = clock_t_::now();
            if (!verify(msg.data(), pub.data(), sig.data(), batch, results.data(), max_threads))
                return false;
            const auto dt = secs_since(t0);
            if (dt < best)
                best = dt;
        }
        std::printf("   %-18s threads=%-2zu %8.2f M sig/s   (%.4f s for %zu, best of %d)\n",
            kind, recorded_threads, static_cast<double>(batch) / best / 1e6, best, batch, iters);
        record(kind, recorded_threads, best);
        return true;
    };

    bool ok = true;
    ok = ok && ufsecp::lbtc::ecdsa_verify_batch(
        e_rows.data(), ecdsa_stride, batch, results.data(), 0);
    ok = ok && count_invalid(results) == 0;
    {
        const auto saved = e_rows[65];
        e_rows[65] ^= 0x01;
        ok = ok && !ufsecp::lbtc::ecdsa_verify_batch(
            e_rows.data(), ecdsa_stride, batch, results.data(), 0);
        ok = ok && count_invalid(results) >= 1 && results[0] == 0;
        e_rows[65] = saved;
    }
    ok = ok && ufsecp::lbtc::schnorr_verify_batch(
        s_rows.data(), schnorr_stride, batch, results.data(), 0);
    ok = ok && count_invalid(results) == 0;
    {
        const auto saved = s_rows[64];
        s_rows[64] ^= 0x01;
        ok = ok && !ufsecp::lbtc::schnorr_verify_batch(
            s_rows.data(), schnorr_stride, batch, results.data(), 0);
        ok = ok && count_invalid(results) >= 1 && results[0] == 0;
        s_rows[64] = saved;
    }
    ok = ok && ufsecp::lbtc::ecdsa_verify_columns(
        e_msg.data(), e_pub.data(), e_sig.data(), batch, results.data(), 0);
    ok = ok && count_invalid(results) == 0;
    ok = ok && ufsecp::lbtc::schnorr_verify_columns(
        s_msg.data(), s_pub.data(), s_sig.data(), batch, results.data(), 0);
    ok = ok && count_invalid(results) == 0;
    std::printf("correctness: %s (all-valid + corruption detected, direct C++)\n",
        ok ? "PASS" : "FAIL");
    if (!ok)
        return 1;

    const std::size_t threadset[] = {1, 0};
    for (const auto max_threads: threadset) {
        const auto recorded_threads = max_threads == 0 ? auto_threads : max_threads;
        ok = ok && bench_rows("ECDSA-row", ufsecp::lbtc::ecdsa_verify_batch,
            e_rows, ecdsa_stride, max_threads, recorded_threads);
        ok = ok && bench_rows("Schnorr-row", ufsecp::lbtc::schnorr_verify_batch,
            s_rows, schnorr_stride, max_threads, recorded_threads);
        ok = ok && bench_columns("ECDSA-columns", ufsecp::lbtc::ecdsa_verify_columns,
            e_msg, e_pub, e_sig, max_threads, recorded_threads);
        ok = ok && bench_columns("Schnorr-columns", ufsecp::lbtc::schnorr_verify_columns,
            s_msg, s_pub, s_sig, max_threads, recorded_threads);
    }
    if (!ok)
        return 1;

    if (!json_path.empty()) {
        if (!write_json_artifact(json_path, bench_results, batch, iters, pool)) {
            std::fprintf(stderr, "failed to write JSON benchmark artifact: %s\n",
                json_path.c_str());
            return 1;
        }
        std::printf("\nwrote JSON artifact: %s\n", json_path.c_str());
    }

    return 0;
}
