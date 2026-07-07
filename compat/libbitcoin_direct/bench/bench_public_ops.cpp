// Direct C++ libbitcoin public-data batch-op benchmark.
//
// Measures the canonical bridge-free surface:
//   ufsecp::lbtc::{xonly_validate,pubkey_validate,taproot_commitment_verify,
//                  tagged_hash,tagged_hash_var,hash256,hash256_var}_batch
//
// No ufsecp C ABI, no libsecp256k1 shim, no ufsecp_lbtc bridge.
//
// Usage:
//   bench_lbtc_public_ops [count] [iters] [fixed_len] [stride] [var_min] [var_max] [--json path]
#include "ufsecp/libbitcoin.hpp"

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
    std::size_t count = 32768;
    int iters = 5;
    std::size_t fixed_len = 80;
    std::size_t stride = 512;
    std::size_t var_min = 80;
    std::size_t var_max = 512;
    std::string json_path;
};

struct BenchResult {
    std::string op;
    std::string mode;
    bool hook_installed = false;
    std::size_t payload_bytes = 0;
    double best_seconds = 0.0;
    double m_rows_per_sec = 0.0;
    double payload_mib_per_sec = 0.0;
    double ns_per_row = 0.0;
};

std::uint64_t g_xs = 0xA0761D6478BD642Full;

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

bool mul_overflows(std::size_t a, std::size_t b) noexcept
{
    return b != 0 && a > std::numeric_limits<std::size_t>::max() / b;
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

    if (positional.size() > 0) args.count = std::strtoull(positional[0], nullptr, 10);
    if (positional.size() > 1) args.iters = std::atoi(positional[1]);
    if (positional.size() > 2) args.fixed_len = std::strtoull(positional[2], nullptr, 10);
    if (positional.size() > 3) args.stride = std::strtoull(positional[3], nullptr, 10);
    if (positional.size() > 4) args.var_min = std::strtoull(positional[4], nullptr, 10);
    if (positional.size() > 5) args.var_max = std::strtoull(positional[5], nullptr, 10);

    if (args.count == 0 || args.iters <= 0 || args.fixed_len == 0 ||
        args.stride == 0 || args.var_min == 0 || args.var_max == 0 ||
        args.var_min > args.var_max || args.var_max > args.stride ||
        args.stride > std::numeric_limits<std::uint32_t>::max() ||
        args.var_max > std::numeric_limits<std::uint32_t>::max() ||
        mul_overflows(args.count, args.fixed_len) ||
        mul_overflows(args.count, args.stride) ||
        mul_overflows(args.count, std::size_t{128})) {
        std::fprintf(stderr,
            "invalid args: require count>0 iters>0 fixed_len>0 0<var_min<=var_max<=stride\n");
        return false;
    }
    return true;
}

void rand_sk(std::uint8_t sk[32])
{
    do {
        for (int i = 0; i < 32; ++i)
            sk[i] = rng8();
    } while (!ufsecp::lbtc::seckey_verify(sk));
}

void rand_hash(std::uint8_t h[32])
{
    for (int i = 0; i < 32; ++i)
        h[i] = rng8();
}

bool make_taproot_row(std::uint8_t ix32[32], std::uint8_t tw32[32],
                      std::uint8_t tx32[32], std::uint8_t* parity)
{
    std::uint8_t sk[32];
    rand_sk(sk);
    if (!ufsecp::lbtc::schnorr_keypair_create(sk, ix32))
        return false;

    secp256k1::SchnorrXonlyPubkey xp;
    if (!secp256k1::schnorr_xonly_pubkey_parse(xp, ix32))
        return false;

    for (int attempt = 0; attempt < 8; ++attempt) {
        rand_hash(tw32);
        const auto t = secp256k1::fast::Scalar::from_bytes(tw32);
        const auto q = secp256k1::fast::Point::dual_scalar_mul_gen_point(
            t, secp256k1::fast::Scalar::one(), xp.point);
        if (q.is_infinity())
            continue;
        const auto comp = q.to_compressed();
        std::memcpy(tx32, comp.data() + 1, 32);
        *parity = comp[0] == 0x03 ? 1 : 0;
        return true;
    }
    return false;
}

template <typename Fn>
bool make_gold(std::vector<std::uint8_t>& out, Fn&& call)
{
    std::fill(out.begin(), out.end(), 0);
    return call(out) && !out.empty();
}

template <typename Fn>
bool bench_output(std::vector<BenchResult>& results, const Args& args,
                  const char* op, const char* mode, bool hook_installed,
                  std::size_t payload_bytes, std::vector<std::uint8_t>& out,
                  const std::vector<std::uint8_t>& gold, Fn&& call)
{
    double best = 1e100;
    for (int it = 0; it < args.iters; ++it) {
        std::fill(out.begin(), out.end(), 0);
        const auto t0 = clock_t_::now();
        if (!call(out)) {
            std::fprintf(stderr, "%s/%s returned false\n", op, mode);
            return false;
        }
        const auto dt = secs_since(t0);
        if (out != gold) {
            std::fprintf(stderr, "%s/%s mismatched forced-CPU gold\n", op, mode);
            return false;
        }
        if (dt < best)
            best = dt;
    }

    BenchResult r;
    r.op = op;
    r.mode = mode;
    r.hook_installed = hook_installed;
    r.payload_bytes = payload_bytes;
    r.best_seconds = best;
    r.m_rows_per_sec = static_cast<double>(args.count) / best / 1e6;
    r.payload_mib_per_sec = (static_cast<double>(payload_bytes) / 1048576.0) / best;
    r.ns_per_row = best * 1e9 / static_cast<double>(args.count);
    results.push_back(r);

    std::printf("   %-24s %-18s hook=%-3s %8.2f M rows/s %9.2f MiB/s %8.1f ns/row\n",
        op, mode, hook_installed ? "yes" : "no", r.m_rows_per_sec,
        r.payload_mib_per_sec, r.ns_per_row);
    return true;
}

bool write_json_artifact(const std::string& path, const Args& args,
                         const std::vector<BenchResult>& results)
{
    std::ofstream out(path);
    if (!out)
        return false;

    out << std::setprecision(12);
    out << "{\n";
    out << "  \"schema\": \"ufsecp-lbtc-public-ops-benchmark-v1\",\n";
    out << "  \"target_context\": \"libbitcoin-direct-cpp\",\n";
    out << "  \"claim_scope\": \"local bridge-free libbitcoin public-data batch-op throughput; production rows use GPU only when the op hook is installed and accepts the batch\",\n";
    out << "  \"c_abi_required\": false,\n";
    out << "  \"shim_required\": false,\n";
    out << "  \"bridge_required\": false,\n";
    out << "  \"count\": " << args.count << ",\n";
    out << "  \"iters\": " << args.iters << ",\n";
    out << "  \"fixed_len\": " << args.fixed_len << ",\n";
    out << "  \"stride\": " << args.stride << ",\n";
    out << "  \"var_min\": " << args.var_min << ",\n";
    out << "  \"var_max\": " << args.var_max << ",\n";
    out << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"op\": \"" << json_escape(r.op) << "\",\n";
        out << "      \"mode\": \"" << json_escape(r.mode) << "\",\n";
        out << "      \"hook_installed\": " << (r.hook_installed ? "true" : "false") << ",\n";
        out << "      \"payload_bytes_per_iter\": " << r.payload_bytes << ",\n";
        out << "      \"best_seconds\": " << r.best_seconds << ",\n";
        out << "      \"m_rows_per_sec\": " << r.m_rows_per_sec << ",\n";
        out << "      \"payload_mib_per_sec\": " << r.payload_mib_per_sec << ",\n";
        out << "      \"ns_per_row\": " << r.ns_per_row << "\n";
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

    std::printf("== libbitcoin direct C++ public-data batch-op benchmark ==\n");
    std::printf("count=%zu  iters=%d  fixed_len=%zu  stride=%zu  var=[%zu,%zu]\n",
        args.count, args.iters, args.fixed_len, args.stride, args.var_min, args.var_max);
    std::printf("surface=ufsecp::lbtc public-data ops  c_abi=no shim=no bridge=no\n\n");

    std::vector<std::uint8_t> xonly(args.count * 32);
    std::vector<std::uint8_t> pub33(args.count * 33);
    std::vector<std::uint8_t> internal_x(args.count * 32);
    std::vector<std::uint8_t> tweak(args.count * 32);
    std::vector<std::uint8_t> tweaked_x(args.count * 32);
    std::vector<std::uint8_t> parity(args.count);
    std::vector<std::uint8_t> fixed_msgs(args.count * args.fixed_len);
    std::vector<std::uint8_t> var_msgs(args.count * args.stride);
    std::vector<std::uint32_t> var_lens(args.count);
    std::uint64_t var_payload = 0;

    const auto tag_hash = secp256k1::SHA256::hash("BIP0340/test", 12);

    std::printf("generating valid public-data rows...\n");
    const std::size_t span = args.var_max - args.var_min + 1;
    for (std::size_t i = 0; i < args.count; ++i) {
        std::uint8_t sk[32];
        rand_sk(sk);
        if (!ufsecp::lbtc::schnorr_keypair_create(sk, xonly.data() + i * 32) ||
            !ufsecp::lbtc::pubkey_create(sk, pub33.data() + i * 33) ||
            !make_taproot_row(internal_x.data() + i * 32, tweak.data() + i * 32,
                              tweaked_x.data() + i * 32, parity.data() + i)) {
            std::fprintf(stderr, "failed to generate row %zu\n", i);
            return 1;
        }
        for (std::size_t j = 0; j < args.fixed_len; ++j)
            fixed_msgs[i * args.fixed_len + j] = rng8();
        for (std::size_t j = 0; j < args.stride; ++j)
            var_msgs[i * args.stride + j] = rng8();
        const auto len = args.var_min + static_cast<std::size_t>(rng64() % span);
        var_lens[i] = static_cast<std::uint32_t>(len);
        var_payload += static_cast<std::uint64_t>(len);
    }

    std::vector<BenchResult> results;
    std::vector<std::uint8_t> gold_validate(args.count);
    std::vector<std::uint8_t> out_validate(args.count);
    std::vector<std::uint8_t> gold_hash(args.count * 32);
    std::vector<std::uint8_t> out_hash(args.count * 32);

    auto bench_validate_op = [&](const char* op, std::size_t payload,
                                 auto install_hook, auto hook_load, auto call) {
        auto saved = install_hook(nullptr);
        const bool gold_ok = make_gold(gold_validate, [&](std::vector<std::uint8_t>& dst) {
            return call(dst);
        });
        if (!gold_ok) {
            std::fprintf(stderr, "%s forced-CPU gold failed\n", op);
            install_hook(saved);
            return false;
        }
        const bool forced_ok = bench_output(results, args, op, "direct-cpu-forced", false,
            payload, out_validate, gold_validate, [&](std::vector<std::uint8_t>& dst) {
                return call(dst);
            });
        install_hook(saved);
        if (!forced_ok)
            return false;
        const bool prod_hook = hook_load() != nullptr;
        return bench_output(results, args, op, "direct-production", prod_hook,
            payload, out_validate, gold_validate, [&](std::vector<std::uint8_t>& dst) {
                return call(dst);
            });
    };

    auto bench_hash_op = [&](const char* op, std::size_t payload,
                             auto install_hook, auto hook_load, auto call) {
        auto saved = install_hook(nullptr);
        const bool gold_ok = make_gold(gold_hash, [&](std::vector<std::uint8_t>& dst) {
            return call(dst);
        });
        if (!gold_ok) {
            std::fprintf(stderr, "%s forced-CPU gold failed\n", op);
            install_hook(saved);
            return false;
        }
        const bool forced_ok = bench_output(results, args, op, "direct-cpu-forced", false,
            payload, out_hash, gold_hash, [&](std::vector<std::uint8_t>& dst) {
                return call(dst);
            });
        install_hook(saved);
        if (!forced_ok)
            return false;
        const bool prod_hook = hook_load() != nullptr;
        return bench_output(results, args, op, "direct-production", prod_hook,
            payload, out_hash, gold_hash, [&](std::vector<std::uint8_t>& dst) {
                return call(dst);
            });
    };

    bool ok = true;
    ok = ok && bench_validate_op("xonly_validate", args.count * 32,
        ufsecp::lbtc::gpu_hook::install_lbtc_xonly_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_xonly_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::xonly_validate_batch(xonly.data(), args.count, dst.data());
        });

    ok = ok && bench_validate_op("pubkey_validate", args.count * 33,
        ufsecp::lbtc::gpu_hook::install_lbtc_pubkey_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_pubkey_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::pubkey_validate_batch(pub33.data(), args.count, dst.data());
        });

    ok = ok && bench_validate_op("commitment_verify", args.count * (32 * 3 + 1),
        ufsecp::lbtc::gpu_hook::install_lbtc_commit_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_commit_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::taproot_commitment_verify_batch(
                internal_x.data(), tweak.data(), tweaked_x.data(), parity.data(),
                args.count, dst.data());
        });

    ok = ok && bench_hash_op("tagged_hash", args.count * args.fixed_len,
        ufsecp::lbtc::gpu_hook::install_lbtc_tagged_hash_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::tagged_hash_batch(
                tag_hash.data(), fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        });

    ok = ok && bench_hash_op("tagged_hash_tag_overload", args.count * args.fixed_len,
        ufsecp::lbtc::gpu_hook::install_lbtc_tagged_hash_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::tagged_hash_batch(
                "BIP0340/test", 12, fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        });

    ok = ok && bench_hash_op("tagged_hash_var", static_cast<std::size_t>(var_payload),
        ufsecp::lbtc::gpu_hook::install_lbtc_tagged_hash_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_var_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::tagged_hash_var_batch(
                tag_hash.data(), var_msgs.data(), var_lens.data(), args.stride,
                args.count, dst.data());
        });

    ok = ok && bench_hash_op("hash256", args.count * args.fixed_len,
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::hash256_batch(
                fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        });

    ok = ok && bench_hash_op("hash256_var", static_cast<std::size_t>(var_payload),
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::hash256_var_batch(
                var_msgs.data(), var_lens.data(), args.stride, args.count, dst.data());
        });

    if (!ok)
        return 1;

    if (!args.json_path.empty()) {
        if (!write_json_artifact(args.json_path, args, results)) {
            std::fprintf(stderr, "failed to write JSON benchmark artifact: %s\n",
                args.json_path.c_str());
            return 1;
        }
        std::printf("\nwrote JSON artifact: %s\n", args.json_path.c_str());
    }

    return 0;
}
