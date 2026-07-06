// Direct C++ libbitcoin HASH256-var benchmark.
//
// This measures the canonical bridge-free libbitcoin surface:
// ufsecp::lbtc::hash256_var_batch(inputs, input_lens, stride, count, out32).
// No ufsecp C ABI, no libsecp256k1 shim, no ufsecp_lbtc bridge.
//
// Usage:
//   bench_lbtc_hash256_var [count] [iters] [stride] [min_len] [max_len] [--json path]
#include "ufsecp/libbitcoin.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <vector>

namespace {

using clock_t_ = std::chrono::steady_clock;

struct Args {
    std::size_t count = 65536;
    int iters = 5;
    std::size_t stride = 512;
    std::size_t min_len = 80;
    std::size_t max_len = 512;
    std::string json_path;
};

struct BenchResult {
    std::string kind;
    double best_seconds = 0.0;
    double m_rows_per_sec = 0.0;
    double payload_mib_per_sec = 0.0;
    double stride_mib_per_sec = 0.0;
    double ns_per_row = 0.0;
    double speedup_vs_serial = 0.0;
};

std::uint64_t g_xs = 0xD1B54A32D192ED03ull;

std::uint64_t rng64() noexcept
{
    g_xs ^= g_xs << 13;
    g_xs ^= g_xs >> 7;
    g_xs ^= g_xs << 17;
    return g_xs;
}

std::uint8_t rng8() noexcept
{
    return static_cast<std::uint8_t>(rng64());
}

double secs_since(clock_t_::time_point t0)
{
    return std::chrono::duration<double>(clock_t_::now() - t0).count();
}

bool mul_overflows(std::size_t a, std::size_t b) noexcept
{
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b;
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

bool parse_args(int argc, char** argv, Args& args)
{
    std::vector<const char*> positional;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--json") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "--json requires a path\n");
                return false;
            }
            args.json_path = argv[i];
        } else {
            positional.push_back(argv[i]);
        }
    }

    if (positional.size() > 0)
        args.count = std::strtoull(positional[0], nullptr, 10);
    if (positional.size() > 1)
        args.iters = std::atoi(positional[1]);
    if (positional.size() > 2)
        args.stride = std::strtoull(positional[2], nullptr, 10);
    if (positional.size() > 3)
        args.min_len = std::strtoull(positional[3], nullptr, 10);
    if (positional.size() > 4)
        args.max_len = std::strtoull(positional[4], nullptr, 10);

    if (args.count == 0 || args.iters <= 0 || args.stride == 0 ||
        args.min_len == 0 || args.max_len == 0 || args.min_len > args.max_len ||
        args.max_len > args.stride || args.stride > std::numeric_limits<std::uint32_t>::max() ||
        args.max_len > std::numeric_limits<std::uint32_t>::max() ||
        mul_overflows(args.count, args.stride) ||
        mul_overflows(args.count, std::size_t{32})) {
        std::fprintf(stderr,
            "invalid args: require count>0 iters>0 0<min_len<=max_len<=stride<=UINT32_MAX\n");
        return false;
    }
    return true;
}

void reference_hash256_var(const std::vector<std::uint8_t>& inputs,
                           const std::vector<std::uint32_t>& lens,
                           std::size_t stride, std::vector<std::uint8_t>& out)
{
    for (std::size_t i = 0; i < lens.size(); ++i) {
        const auto d = secp256k1::SHA256::hash256(
            inputs.data() + i * stride, static_cast<std::size_t>(lens[i]));
        std::memcpy(out.data() + i * 32, d.data(), 32);
    }
}

bool direct_hash256_var(const std::vector<std::uint8_t>& inputs,
                        const std::vector<std::uint32_t>& lens,
                        std::size_t stride, std::vector<std::uint8_t>& out)
{
    return ufsecp::lbtc::hash256_var_batch(
        inputs.data(), lens.data(), stride, lens.size(), out.data(), 0);
}

bool write_json_artifact(const std::string& path, const Args& args,
                         bool hook_installed, int hook_sample_status,
                         bool hook_sample_matches, std::uint64_t payload_bytes,
                         const std::vector<BenchResult>& results)
{
    std::ofstream out(path);
    if (!out)
        return false;

    out << std::setprecision(12);
    out << "{\n";
    out << "  \"schema\": \"ufsecp-lbtc-hash256-var-benchmark-v1\",\n";
    out << "  \"target_context\": \"libbitcoin-direct-cpp\",\n";
    out << "  \"surface\": \"ufsecp::lbtc::hash256_var_batch\",\n";
    out << "  \"claim_scope\": \"local bridge-free variable-length HASH256 throughput; production row uses GPU only when the installed hook accepts the batch\",\n";
    out << "  \"c_abi_required\": false,\n";
    out << "  \"shim_required\": false,\n";
    out << "  \"bridge_required\": false,\n";
    out << "  \"count\": " << args.count << ",\n";
    out << "  \"iters\": " << args.iters << ",\n";
    out << "  \"stride\": " << args.stride << ",\n";
    out << "  \"min_len\": " << args.min_len << ",\n";
    out << "  \"max_len\": " << args.max_len << ",\n";
    out << "  \"payload_bytes_per_iter\": " << payload_bytes << ",\n";
    out << "  \"stride_bytes_per_iter\": " << (args.count * args.stride) << ",\n";
    out << "  \"production_hook_installed\": " << (hook_installed ? "true" : "false") << ",\n";
    out << "  \"production_hook_sample_status\": " << hook_sample_status << ",\n";
    out << "  \"production_hook_sample_matched_reference\": "
        << (hook_sample_matches ? "true" : "false") << ",\n";
    out << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"kind\": \"" << json_escape(r.kind) << "\",\n";
        out << "      \"best_seconds\": " << r.best_seconds << ",\n";
        out << "      \"m_rows_per_sec\": " << r.m_rows_per_sec << ",\n";
        out << "      \"payload_mib_per_sec\": " << r.payload_mib_per_sec << ",\n";
        out << "      \"stride_mib_per_sec\": " << r.stride_mib_per_sec << ",\n";
        out << "      \"ns_per_row\": " << r.ns_per_row << ",\n";
        out << "      \"speedup_vs_serial\": " << r.speedup_vs_serial << "\n";
        out << "    }" << (i + 1 == results.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    return out.good();
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    if (!parse_args(argc, argv, args))
        return 2;

    const std::size_t total_input_bytes = args.count * args.stride;
    std::vector<std::uint8_t> inputs(total_input_bytes);
    std::vector<std::uint32_t> lens(args.count);
    std::vector<std::uint8_t> gold(args.count * 32);
    std::vector<std::uint8_t> out(args.count * 32);

    const std::size_t len_span = args.max_len - args.min_len + 1;
    std::uint64_t payload_bytes = 0;
    for (std::size_t i = 0; i < args.count; ++i) {
        const auto len = args.min_len + static_cast<std::size_t>(rng64() % len_span);
        lens[i] = static_cast<std::uint32_t>(len);
        payload_bytes += static_cast<std::uint64_t>(len);
        auto* row = inputs.data() + i * args.stride;
        for (std::size_t j = 0; j < args.stride; ++j)
            row[j] = rng8();
    }

    std::printf("== libbitcoin direct C++ hash256_var benchmark ==\n");
    std::printf("count=%zu  iters=%d  stride=%zu  len=[%zu,%zu]\n",
        args.count, args.iters, args.stride, args.min_len, args.max_len);
    std::printf("surface=ufsecp::lbtc::hash256_var_batch  c_abi=no shim=no bridge=no\n");
    std::printf("payload=%.2f MiB/iter  stride-copy=%.2f MiB/iter\n\n",
        static_cast<double>(payload_bytes) / 1048576.0,
        static_cast<double>(total_input_bytes) / 1048576.0);

    reference_hash256_var(inputs, lens, args.stride, gold);
    if (!direct_hash256_var(inputs, lens, args.stride, out) || out != gold) {
        std::fprintf(stderr, "correctness warmup failed: direct output differs from reference\n");
        return 1;
    }

    {
        std::uint32_t bad_len = 0;
        std::array<std::uint8_t, 32> sentinel;
        sentinel.fill(0xA5);
        auto before = sentinel;
        const bool ok = ufsecp::lbtc::hash256_var_batch(
            inputs.data(), &bad_len, args.stride, 1, sentinel.data(), 0);
        if (ok || sentinel != before) {
            std::fprintf(stderr, "fail-closed check failed: bad len touched out32 or returned true\n");
            return 1;
        }
    }

    auto hook = ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire);
    const bool hook_installed = hook != nullptr;
    int hook_sample_status = -2; // -2 means no hook installed.
    bool hook_sample_matches = false;
    if (hook != nullptr) {
        std::array<std::uint8_t, 32> hook_out{};
        std::array<std::uint8_t, 32> ref0{};
        const auto d = secp256k1::SHA256::hash256(inputs.data(), lens[0]);
        std::memcpy(ref0.data(), d.data(), 32);
        hook_sample_status = hook(inputs.data(), lens.data(), args.stride, 1, hook_out.data());
        hook_sample_matches = hook_sample_status == 0 && hook_out == ref0;
        if (hook_sample_status == 0 && !hook_sample_matches) {
            std::fprintf(stderr, "production hook accepted sample but mismatched reference\n");
            return 1;
        }
    }

    std::printf("correctness: PASS (reference parity + bad-len untouched-output check)\n");
    std::printf("production hash256_var hook: %s",
        hook_installed ? "installed" : "not installed");
    if (hook_installed)
        std::printf("  sample_status=%d  sample_reference=%s",
            hook_sample_status, hook_sample_matches ? "match" : "not-handled");
    std::printf("\n\n");

    std::vector<BenchResult> results;
    double serial_best = 0.0;
    const double payload_mib = static_cast<double>(payload_bytes) / 1048576.0;
    const double stride_mib = static_cast<double>(total_input_bytes) / 1048576.0;

    auto bench = [&](const char* kind, auto&& fn) {
        double best = 1e100;
        for (int it = 0; it < args.iters; ++it) {
            std::fill(out.begin(), out.end(), 0);
            const auto t0 = clock_t_::now();
            if (!fn(out))
                return false;
            const auto dt = secs_since(t0);
            if (out != gold) {
                std::fprintf(stderr, "%s mismatched reference\n", kind);
                return false;
            }
            if (dt < best)
                best = dt;
        }
        if (serial_best == 0.0)
            serial_best = best;
        const double rows_per_sec = static_cast<double>(args.count) / best;
        BenchResult r;
        r.kind = kind;
        r.best_seconds = best;
        r.m_rows_per_sec = rows_per_sec / 1e6;
        r.payload_mib_per_sec = payload_mib / best;
        r.stride_mib_per_sec = stride_mib / best;
        r.ns_per_row = best * 1e9 / static_cast<double>(args.count);
        r.speedup_vs_serial = serial_best / best;
        results.push_back(r);
        std::printf("   %-24s %8.2f M rows/s  %9.2f payload MiB/s  %9.2f stride MiB/s  %8.1f ns/row  %.2fx\n",
            kind, r.m_rows_per_sec, r.payload_mib_per_sec, r.stride_mib_per_sec,
            r.ns_per_row, r.speedup_vs_serial);
        return true;
    };

    if (!bench("serial-reference", [&](std::vector<std::uint8_t>& dst) {
            reference_hash256_var(inputs, lens, args.stride, dst);
            return true;
        })) {
        return 1;
    }

    auto saved_hook = ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook(nullptr);
    const bool forced_cpu_ok = bench("direct-cpu-forced", [&](std::vector<std::uint8_t>& dst) {
        return direct_hash256_var(inputs, lens, args.stride, dst);
    });
    ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook(saved_hook);
    if (!forced_cpu_ok)
        return 1;

    if (!bench("direct-production", [&](std::vector<std::uint8_t>& dst) {
            return direct_hash256_var(inputs, lens, args.stride, dst);
        })) {
        return 1;
    }

    if (!args.json_path.empty()) {
        if (!write_json_artifact(args.json_path, args, hook_installed, hook_sample_status,
                hook_sample_matches, payload_bytes, results)) {
            std::fprintf(stderr, "failed to write JSON benchmark artifact: %s\n",
                args.json_path.c_str());
            return 1;
        }
        std::printf("\nwrote JSON artifact: %s\n", args.json_path.c_str());
    }

    return 0;
}
