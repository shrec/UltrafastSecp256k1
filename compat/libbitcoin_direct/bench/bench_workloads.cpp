// Direct C++ libbitcoin-shaped workload benchmark harness (schema v2).
//
// Benchmarks 4 libbitcoin block-processing-shaped workloads over the
// bridge-free ufsecp::lbtc direct surface:
//   txid_batch, wtxid_batch, merkle_pair_batch, merkle_root_batch
//
// sighash_batch is intentionally excluded: the descriptor contract has not
// been accepted (no sighash_descriptor_hash_batch exists anywhere in this
// codebase, and workingdocs/libbitcoin_gpu_workloads/
// api_plan_blocker_resolution_deepseek.md B1 explicitly gates any sighash
// kernel work behind an external libbitcoin-developer review that has not
// happened) -- see docs/LIBBITCOIN_PUBLIC_OPS_BENCHMARKS.md.
//
// Evidence honesty (CLAUDE.md benchmark rule): every row in this harness is
// measured with the GPU hook explicitly forced off before the timed call
// (mode="direct-cpu-forced", backend="cpu", evidence_class="api_correctness").
// This harness has no backend/device/driver identification API, so it
// cannot honestly attribute a row to a specific GPU backend/device -- per
// the task contract, rows must stay backend="cpu"/api_correctness unless a
// harness can identify backend/device/driver and measure real
// upload/kernel/download phases. `provider_linked` is reported only as
// informational metadata (whether a GPU hook happens to be installed in
// this binary at all), it never changes `backend` or `evidence_class`.
//
// Independent validation oracles (never call the ufsecp::lbtc batch
// function under test):
//   txid_batch / wtxid_batch : secp256k1::SHA256::hash256(span) per row,
//                               called directly -- bypasses
//                               hash256_var_batch/txid_hash_batch/
//                               wtxid_hash_batch entirely.
//   merkle_pair_batch        : secp256k1::SHA256::hash256(left32||right32)
//                               per row, bypasses merkle_pair_hash_batch.
//   merkle_root_batch        : hand-written level-reduction loop in this
//                               file (independent_merkle_root, below) with
//                               Bitcoin odd-leaf duplication re-derived at
//                               every level. Does NOT call
//                               merkle_pair_hash_batch, merkle_level_reduce_batch,
//                               or merkle_root_from_leaves -- calling any of
//                               those would be a tautological self-check of
//                               the exact combination/duplication logic
//                               under test.
//
// Schema: ufsecp-lbtc-gpu-workload-benchmark-v1
// (see ci/check_lbtc_gpu_workload_evidence.py and
//  workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json).
// That schema's envelope carries a single `workload` + `batch_class` value,
// so this harness writes ONE JSON artifact per workload (not one combined
// file) via --json-dir.
//
// No ufsecp C ABI, no libsecp256k1 shim, no ufsecp_lbtc bridge.
//
// Usage:
//   bench_lbtc_workloads [--batch-class small|medium|block_scale|stress]
//                         [--iters N] [--json-dir DIR]
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
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/utsname.h>
#endif

namespace {

using clock_t_ = std::chrono::steady_clock;

struct Args {
    std::string batch_class = "small";
    int iters = 5;
    std::string json_dir;
};

// Batch-size sizing table. These are workload-shape DESIGN PARAMETERS taken
// from workingdocs/libbitcoin_gpu_workloads/benchmark_plan_claude.md (that
// document is explicit: "every number is a design parameter", not a
// benchmark claim) -- they select how many independent rows this run
// processes, they are not themselves throughput numbers.
struct BatchSizing {
    std::size_t txid_wtxid_count;
    std::size_t merkle_pair_count;
    std::size_t merkle_root_trees;
};

BatchSizing sizing_for(const std::string& batch_class)
{
    if (batch_class == "medium")      return {32768, 65536, 512};
    if (batch_class == "block_scale") return {4096, 8192, 64};
    if (batch_class == "stress")      return {1048576, 2097152, 4096};
    return {64, 128, 8}; // "small" (default / smoke)
}

// merkle_root_batch: leaves per simulated tree. Fixed regardless of
// batch_class (batch_class controls how many trees/blocks are benchmarked,
// not the shape of one tree) -- a synthetic block-shaped leaf count, not
// tied to any specific historical Bitcoin block.
constexpr std::size_t kMerkleRootLeavesPerTree = 2048;

// Representative tx-sized payload ranges for txid_batch/wtxid_batch,
// matching the ranges already used by bench_public_ops.cpp for the same
// underlying hash256_var_batch alias ops.
constexpr std::size_t kTxidMinLen = 190;
constexpr std::size_t kTxidMaxLen = 400;
constexpr std::size_t kTxidStride = 400;
constexpr std::size_t kWtxidMinLen = 220;
constexpr std::size_t kWtxidMaxLen = 600;
constexpr std::size_t kWtxidStride = 600;

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

// Evidence-gate provenance: host/build facts, gathered once per run.
// Linux-only today; non-Linux hosts honestly fall back to "unknown"/false
// rather than fabricating a value.
struct HostContext {
    std::string compiler;
    std::string cpu_model = "unknown";
    bool turbo_disabled = false;
    bool cpu_pinned = false;
    std::string kernel = "unknown";
};

std::string detect_compiler()
{
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("gcc ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

HostContext detect_host_context()
{
    HostContext hc;
    hc.compiler = detect_compiler();
#if defined(__linux__)
    {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.rfind("model name", 0) == 0) {
                const auto pos = line.find(':');
                if (pos != std::string::npos && pos + 2 <= line.size()) {
                    hc.cpu_model = line.substr(pos + 2);
                    break;
                }
            }
        }
    }
    {
        std::ifstream no_turbo("/sys/devices/system/cpu/intel_pstate/no_turbo");
        int v = 0;
        if (no_turbo >> v)
            hc.turbo_disabled = (v != 0);
    }
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        if (sched_getaffinity(0, sizeof(set), &set) == 0)
            hc.cpu_pinned = (CPU_COUNT(&set) == 1);
    }
    {
        struct utsname uts;
        if (uname(&uts) == 0)
            hc.kernel = std::string(uts.sysname) + " " + uts.release;
    }
#endif
    return hc;
}

std::string sha256_hex(const std::vector<std::uint8_t>& data)
{
    const auto digest = secp256k1::SHA256::hash(data.data(), data.size());
    static const char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(digest.size() * 2);
    for (auto b: digest) {
        out += kHex[b >> 4];
        out += kHex[b & 0x0F];
    }
    return out;
}

// Independent Bitcoin merkle-root oracle. Hand-written level-reduction loop
// with odd-leaf duplication re-derived at every level (Bitcoin consensus
// rule). Deliberately does NOT call merkle_pair_hash_batch,
// merkle_level_reduce_batch, or merkle_root_from_leaves -- only the trusted
// scalar SHA256 primitive, so this is a genuine independent check of the
// combination/duplication logic, not a tautological self-check.
void independent_merkle_root(const std::uint8_t* leaves32, std::size_t leaf_count,
                             std::uint8_t out32[32])
{
    std::vector<std::array<std::uint8_t, 32>> level(leaf_count);
    for (std::size_t i = 0; i < leaf_count; ++i)
        std::memcpy(level[i].data(), leaves32 + i * 32, 32);

    while (level.size() > 1) {
        std::vector<std::array<std::uint8_t, 32>> next((level.size() + 1) / 2);
        for (std::size_t i = 0; i < next.size(); ++i) {
            const std::size_t left_idx = 2 * i;
            const std::size_t right_idx = (2 * i + 1 < level.size()) ? (2 * i + 1) : (level.size() - 1);
            std::uint8_t combined[64];
            std::memcpy(combined, level[left_idx].data(), 32);
            std::memcpy(combined + 32, level[right_idx].data(), 32);
            const auto d = secp256k1::SHA256::hash256(combined, 64);
            std::memcpy(next[i].data(), d.data(), 32);
        }
        level = std::move(next);
    }
    std::memcpy(out32, level[0].data(), 32);
}

bool parse_args(int argc, char** argv, Args& args)
{
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--batch-class") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "--batch-class requires a value\n");
                return false;
            }
            args.batch_class = argv[i];
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "--iters requires a value\n");
                return false;
            }
            args.iters = std::atoi(argv[i]);
        } else if (std::strcmp(argv[i], "--json-dir") == 0) {
            if (++i >= argc) {
                std::fprintf(stderr, "--json-dir requires a value\n");
                return false;
            }
            args.json_dir = argv[i];
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return false;
        }
    }
    if (args.batch_class != "small" && args.batch_class != "medium" &&
        args.batch_class != "block_scale" && args.batch_class != "stress") {
        std::fprintf(stderr, "invalid --batch-class (want small|medium|block_scale|stress)\n");
        return false;
    }
    if (args.iters <= 0) {
        std::fprintf(stderr, "--iters must be > 0\n");
        return false;
    }
    return true;
}

// One measured row. Every row in this harness is direct-cpu-forced /
// backend=cpu / evidence_class=api_correctness (see file header).
struct BenchRow {
    std::string workload;
    std::string op;
    std::string mode = "direct-cpu-forced";
    bool hook_installed = false;
    bool provider_linked = false;
    std::string backend = "cpu";
    std::string device = "n/a";
    std::size_t count = 0;
    std::size_t payload_bytes = 0;
    double prep_seconds = 0.0;
    double kernel_seconds = 0.0;
    double best_seconds = 0.0;
    double m_rows_per_sec = 0.0;
    double payload_mib_per_sec = 0.0;
    double ns_per_row = 0.0;
    std::string validation_hash;
    std::string validation_status = "matched_reference";
    std::string evidence_class = "api_correctness";
};

bool write_workload_json(const std::string& path, const std::string& batch_class,
                         int iters, const HostContext& hc, const BenchRow& row)
{
    std::ofstream out(path);
    if (!out)
        return false;

    out << std::setprecision(12);
    out << "{\n";
    out << "  \"schema\": \"ufsecp-lbtc-gpu-workload-benchmark-v1\",\n";
    out << "  \"target_context\": \"libbitcoin\",\n";
    out << "  \"workload\": \"" << json_escape(row.workload) << "\",\n";
    out << "  \"batch_class\": \"" << json_escape(batch_class) << "\",\n";
    out << "  \"claim_scope\": \"local bridge-free libbitcoin-shaped " << json_escape(row.workload)
        << " throughput; GPU hook explicitly forced off for every row (mode=direct-cpu-forced) "
           "because this harness has no backend/device/driver identification API -- every row is "
           "api_correctness evidence, never gpu_acceleration, until real backend-identified "
           "phase-split instrumentation lands\",\n";
    out << "  \"c_abi_required\": false,\n";
    out << "  \"shim_required\": false,\n";
    out << "  \"bridge_required\": false,\n";
    out << "  \"count\": " << row.count << ",\n";
    out << "  \"iters\": " << iters << ",\n";
    out << "  \"payload_bytes_total\": " << row.payload_bytes << ",\n";
    out << "  \"host_context\": {\n";
    out << "    \"compiler\": \"" << json_escape(hc.compiler) << "\",\n";
    out << "    \"cpu_model\": \"" << json_escape(hc.cpu_model) << "\",\n";
    out << "    \"turbo_disabled\": " << (hc.turbo_disabled ? "true" : "false") << ",\n";
    out << "    \"cpu_pinned\": " << (hc.cpu_pinned ? "true" : "false") << ",\n";
    out << "    \"kernel\": \"" << json_escape(hc.kernel) << "\"\n";
    out << "  },\n";
    out << "  \"results\": [\n";
    out << "    {\n";
    out << "      \"op\": \"" << json_escape(row.op) << "\",\n";
    out << "      \"workload\": \"" << json_escape(row.workload) << "\",\n";
    out << "      \"mode\": \"" << json_escape(row.mode) << "\",\n";
    out << "      \"hook_installed\": " << (row.hook_installed ? "true" : "false") << ",\n";
    out << "      \"provider_linked\": " << (row.provider_linked ? "true" : "false") << ",\n";
    out << "      \"backend\": \"" << json_escape(row.backend) << "\",\n";
    out << "      \"device\": \"" << json_escape(row.device) << "\",\n";
    out << "      \"driver_version\": null,\n";
    out << "      \"count\": " << row.count << ",\n";
    out << "      \"payload_bytes_per_iter\": " << row.payload_bytes << ",\n";
    out << "      \"prep_seconds\": " << row.prep_seconds << ",\n";
    out << "      \"upload_seconds\": null,\n";
    out << "      \"kernel_seconds\": " << row.kernel_seconds << ",\n";
    out << "      \"download_seconds\": null,\n";
    out << "      \"best_seconds\": " << row.best_seconds << ",\n";
    out << "      \"m_rows_per_sec\": " << row.m_rows_per_sec << ",\n";
    out << "      \"payload_mib_per_sec\": " << row.payload_mib_per_sec << ",\n";
    out << "      \"ns_per_row\": " << row.ns_per_row << ",\n";
    out << "      \"validation_hash\": \"" << json_escape(row.validation_hash) << "\",\n";
    out << "      \"validation_status\": \"" << json_escape(row.validation_status) << "\",\n";
    out << "      \"evidence_class\": \"" << json_escape(row.evidence_class) << "\"\n";
    out << "    }\n";
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

    const BatchSizing sizing = sizing_for(args.batch_class);

    if (mul_overflows(sizing.txid_wtxid_count, kWtxidStride) ||
        mul_overflows(sizing.merkle_pair_count, std::size_t{32}) ||
        mul_overflows(sizing.merkle_root_trees, kMerkleRootLeavesPerTree) ||
        mul_overflows(sizing.merkle_root_trees * kMerkleRootLeavesPerTree, std::size_t{32})) {
        std::fprintf(stderr, "batch-class sizing overflows buffer bounds\n");
        return 2;
    }

    std::printf("== libbitcoin direct C++ workload benchmark (schema v2) ==\n");
    std::printf("batch_class=%s iters=%d\n", args.batch_class.c_str(), args.iters);
    std::printf("workloads: txid_batch wtxid_batch merkle_pair_batch merkle_root_batch "
                "(sighash_batch excluded -- descriptor contract not accepted)\n");
    std::printf("every row: mode=direct-cpu-forced backend=cpu evidence_class=api_correctness\n\n");

    std::vector<std::pair<std::string, BenchRow>> rows;
    bool ok = true;

    // txid_batch / wtxid_batch share this shape: variable-length serialized
    // records hashed via hash256_var_batch aliases, validated against a
    // per-row direct secp256k1::SHA256::hash256 call (independent of the
    // batch function under test).
    auto run_hash_alias = [&](const char* workload_name, const char* op_name,
                              std::size_t count, std::size_t min_len, std::size_t max_len,
                              std::size_t stride, auto batch_fn, auto install_hook,
                              auto hook_load, BenchRow& row_out) -> bool {
        const auto prep_t0 = clock_t_::now();
        std::vector<std::uint8_t> msgs(count * stride);
        std::vector<std::uint32_t> lens(count);
        std::uint64_t payload = 0;
        const std::size_t span = max_len - min_len + 1;
        for (std::size_t i = 0; i < count; ++i) {
            for (std::size_t j = 0; j < stride; ++j)
                msgs[i * stride + j] = rng8();
            const auto len = min_len + static_cast<std::size_t>(rng64() % span);
            lens[i] = static_cast<std::uint32_t>(len);
            payload += static_cast<std::uint64_t>(len);
        }
        const double prep_seconds = secs_since(prep_t0);

        std::vector<std::uint8_t> expected(count * 32);
        for (std::size_t i = 0; i < count; ++i) {
            const auto d = secp256k1::SHA256::hash256(msgs.data() + i * stride, lens[i]);
            std::memcpy(expected.data() + i * 32, d.data(), 32);
        }

        const bool provider_linked = hook_load() != nullptr;
        auto saved = install_hook(nullptr);

        std::vector<std::uint8_t> out(count * 32);
        double best = 1e100;
        bool call_ok = true;
        for (int it = 0; it < args.iters; ++it) {
            std::fill(out.begin(), out.end(), 0);
            const auto t0 = clock_t_::now();
            if (!batch_fn(msgs.data(), lens.data(), stride, count, out.data(), std::size_t{0})) {
                std::fprintf(stderr, "%s returned false\n", op_name);
                call_ok = false;
                break;
            }
            const auto dt = secs_since(t0);
            if (out != expected) {
                std::fprintf(stderr, "%s mismatched independent HASH256 oracle\n", op_name);
                call_ok = false;
                break;
            }
            if (dt < best)
                best = dt;
        }
        install_hook(saved);
        if (!call_ok)
            return false;

        row_out.workload = workload_name;
        row_out.op = op_name;
        row_out.hook_installed = false;
        row_out.provider_linked = provider_linked;
        row_out.count = count;
        row_out.payload_bytes = static_cast<std::size_t>(payload);
        row_out.prep_seconds = prep_seconds;
        row_out.kernel_seconds = best;
        row_out.best_seconds = best;
        row_out.m_rows_per_sec = static_cast<double>(count) / best / 1e6;
        row_out.payload_mib_per_sec = (static_cast<double>(payload) / 1048576.0) / best;
        row_out.ns_per_row = best * 1e9 / static_cast<double>(count);
        row_out.validation_hash = sha256_hex(out);
        return true;
    };

    BenchRow txid_row;
    ok = ok && run_hash_alias("txid_batch", "txid_hash", sizing.txid_wtxid_count,
        kTxidMinLen, kTxidMaxLen, kTxidStride,
        &ufsecp::lbtc::txid_hash_batch,
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire); },
        txid_row);
    if (ok) rows.emplace_back("txid_batch", txid_row);

    BenchRow wtxid_row;
    ok = ok && run_hash_alias("wtxid_batch", "wtxid_hash", sizing.txid_wtxid_count,
        kWtxidMinLen, kWtxidMaxLen, kWtxidStride,
        &ufsecp::lbtc::wtxid_hash_batch,
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire); },
        wtxid_row);
    if (ok) rows.emplace_back("wtxid_batch", wtxid_row);

    // merkle_pair_batch: fixed 32+32-byte SoA columns, validated against a
    // direct secp256k1::SHA256::hash256(left32||right32) call.
    if (ok) {
        const std::size_t count = sizing.merkle_pair_count;
        const auto prep_t0 = clock_t_::now();
        std::vector<std::uint8_t> left(count * 32), right(count * 32);
        for (std::size_t i = 0; i < count * 32; ++i) {
            left[i] = rng8();
            right[i] = rng8();
        }
        const double prep_seconds = secs_since(prep_t0);

        std::vector<std::uint8_t> expected(count * 32);
        for (std::size_t i = 0; i < count; ++i) {
            std::uint8_t combined[64];
            std::memcpy(combined, left.data() + i * 32, 32);
            std::memcpy(combined + 32, right.data() + i * 32, 32);
            const auto d = secp256k1::SHA256::hash256(combined, 64);
            std::memcpy(expected.data() + i * 32, d.data(), 32);
        }

        const bool provider_linked =
            ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire) != nullptr;
        auto saved = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(nullptr);

        std::vector<std::uint8_t> out(count * 32);
        double best = 1e100;
        bool call_ok = true;
        for (int it = 0; it < args.iters; ++it) {
            std::fill(out.begin(), out.end(), 0);
            const auto t0 = clock_t_::now();
            if (!ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), count, out.data())) {
                std::fprintf(stderr, "merkle_pair_hash returned false\n");
                call_ok = false;
                break;
            }
            const auto dt = secs_since(t0);
            if (out != expected) {
                std::fprintf(stderr, "merkle_pair_hash mismatched independent HASH256 oracle\n");
                call_ok = false;
                break;
            }
            if (dt < best)
                best = dt;
        }
        ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(saved);

        if (!call_ok) {
            ok = false;
        } else {
            BenchRow row;
            row.workload = "merkle_pair_batch";
            row.op = "merkle_pair_hash";
            row.hook_installed = false;
            row.provider_linked = provider_linked;
            row.count = count;
            row.payload_bytes = count * 64;
            row.prep_seconds = prep_seconds;
            row.kernel_seconds = best;
            row.best_seconds = best;
            row.m_rows_per_sec = static_cast<double>(count) / best / 1e6;
            row.payload_mib_per_sec = (static_cast<double>(count * 64) / 1048576.0) / best;
            row.ns_per_row = best * 1e9 / static_cast<double>(count);
            row.validation_hash = sha256_hex(out);
            rows.emplace_back("merkle_pair_batch", row);
        }
    }

    // merkle_root_batch: `trees` simulated blocks, kMerkleRootLeavesPerTree
    // leaves each, validated against independent_merkle_root() (hand-rolled
    // above, does not call any of this library's merkle_* functions).
    if (ok) {
        const std::size_t trees = sizing.merkle_root_trees;
        const std::size_t leaves_per_tree = kMerkleRootLeavesPerTree;
        const auto prep_t0 = clock_t_::now();
        std::vector<std::uint8_t> leaves(trees * leaves_per_tree * 32);
        for (auto& b: leaves)
            b = rng8();
        const double prep_seconds = secs_since(prep_t0);

        std::vector<std::uint8_t> expected(trees * 32);
        for (std::size_t t = 0; t < trees; ++t)
            independent_merkle_root(leaves.data() + t * leaves_per_tree * 32, leaves_per_tree,
                                    expected.data() + t * 32);

        const bool provider_linked =
            ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire) != nullptr;
        auto saved = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(nullptr);

        std::vector<std::uint8_t> scratch(leaves_per_tree * 64);
        std::vector<std::uint8_t> out(trees * 32);
        double best = 1e100;
        bool call_ok = true;
        for (int it = 0; it < args.iters; ++it) {
            std::fill(out.begin(), out.end(), 0);
            const auto t0 = clock_t_::now();
            bool trees_ok = true;
            for (std::size_t t = 0; t < trees; ++t) {
                if (!ufsecp::lbtc::merkle_root_from_leaves(
                        leaves.data() + t * leaves_per_tree * 32, leaves_per_tree,
                        scratch.data(), scratch.size(), out.data() + t * 32)) {
                    trees_ok = false;
                    break;
                }
            }
            const auto dt = secs_since(t0);
            if (!trees_ok) {
                std::fprintf(stderr, "merkle_root_from_leaves returned false\n");
                call_ok = false;
                break;
            }
            if (out != expected) {
                std::fprintf(stderr, "merkle_root_from_leaves mismatched independent merkle oracle\n");
                call_ok = false;
                break;
            }
            if (dt < best)
                best = dt;
        }
        ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(saved);

        if (!call_ok) {
            ok = false;
        } else {
            BenchRow row;
            row.workload = "merkle_root_batch";
            row.op = "merkle_root_from_leaves";
            row.hook_installed = false;
            row.provider_linked = provider_linked;
            row.count = trees;
            row.payload_bytes = trees * leaves_per_tree * 32;
            row.prep_seconds = prep_seconds;
            row.kernel_seconds = best;
            row.best_seconds = best;
            row.m_rows_per_sec = static_cast<double>(trees) / best / 1e6;
            row.payload_mib_per_sec = (static_cast<double>(row.payload_bytes) / 1048576.0) / best;
            row.ns_per_row = best * 1e9 / static_cast<double>(trees);
            row.validation_hash = sha256_hex(out);
            rows.emplace_back("merkle_root_batch", row);
        }
    }

    if (!ok)
        return 1;

    for (const auto& [name, row]: rows) {
        std::printf("   %-18s %8.2f M rows/s %9.2f MiB/s %8.1f ns/row  prep=%.6fs\n",
            name.c_str(), row.m_rows_per_sec, row.payload_mib_per_sec, row.ns_per_row,
            row.prep_seconds);
    }

    if (!args.json_dir.empty()) {
        const HostContext hc = detect_host_context();
        for (const auto& [name, row]: rows) {
            const std::string path = args.json_dir + "/" + name + ".json";
            if (!write_workload_json(path, args.batch_class, args.iters, hc, row)) {
                std::fprintf(stderr, "failed to write JSON artifact: %s\n", path.c_str());
                return 1;
            }
            std::printf("wrote JSON artifact: %s\n", path.c_str());
        }
    }

    return 0;
}
