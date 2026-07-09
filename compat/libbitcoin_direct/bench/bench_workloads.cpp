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
// Evidence honesty (CLAUDE.md benchmark rule): every workload always emits a
// CPU-forced row (GPU hook explicitly forced off before the timed call,
// mode="direct-cpu-forced", backend="cpu", evidence_class="api_correctness").
// When a real GPU backend is linked, initialized, and ready on this host --
// queried through ufsecp::lbtc::gpu_hook::g_lbtc_gpu_telemetry_hook (see
// <ufsecp/lbtc_gpu_ops.hpp> GpuTelemetry, populated only from the
// already-existing GpuBackend::backend_id()/backend_name()/device_info()
// virtuals, no gpu_backend.hpp / backend-file edits) -- each workload ALSO
// runs a second, hook-active ("production") pass and emits a paired row with
// a real backend/device identification (backend != "cpu", device = the
// queried GPU device name) and evidence_class="gpu_acceleration". If no GPU
// provider is linked or ready on this host, only the CPU api_correctness row
// is emitted for that workload -- absence of a GPU is reported honestly,
// never papered over with a fabricated row.
//
// Known, documented limitation: every GpuBackend op used here is a single
// opaque call (e.g. GpuBackend::merkle_pair_hash) with no
// upload/kernel/download phase-split instrumentation, and DeviceInfo carries
// no driver-version field. Adding that instrumentation touches
// gpu_backend.hpp / *_cuda.cu / *_opencl.cpp / *_metal.mm, which are out of
// this change's writable scope. So on every row (CPU or GPU):
// kernel_seconds mirrors best_seconds (the single measured wall-clock span,
// never a fabricated sub-split), upload_seconds/download_seconds stay null,
// and driver_version stays null. None of these are fabricated values -- they
// are honest "not measured" markers. See docs/BENCHMARK_POLICY.md.
//
// Independent validation oracles (never call the ufsecp::lbtc batch function
// under test) -- applied identically to the CPU-forced pass and the
// GPU/production pass, so a GPU row is only ever emitted when its own output
// independently verified correct:
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
// file) via --json-dir; each artifact's `results` array holds 1 row
// (CPU-only host) or 2 paired rows (CPU-forced + GPU/production) on a
// GPU-linked host.
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
#include <optional>
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

// -- GPU telemetry snapshot (benchmark-only; see <ufsecp/lbtc_gpu_ops.hpp>
// GpuTelemetry). Never used on any production/hot code path -- this
// benchmark is the only caller. --------------------------------------------
struct GpuSnapshot {
    bool available = false;
    std::string backend;  // lowercase schema enum: "cuda" | "opencl" | "metal"
    std::string device;
};

GpuSnapshot query_gpu_snapshot()
{
    GpuSnapshot snap;
    const auto fn = ufsecp::lbtc::gpu_hook::g_lbtc_gpu_telemetry_hook.load(std::memory_order_acquire);
    if (fn == nullptr)
        return snap;  // no GPU host TU linked into this binary at all

    ufsecp::lbtc::gpu_hook::GpuTelemetry tel{};
    if (!fn(&tel) || !tel.available)
        return snap;  // GPU host linked, but no backend initialized/ready on this machine

    switch (tel.backend_id) {
    case 1: snap.backend = "cuda"; break;
    case 2: snap.backend = "opencl"; break;
    case 3: snap.backend = "metal"; break;
    default: return GpuSnapshot{};  // unrecognized id -> honestly refuse to label rather than guess
    }
    if (tel.device_name[0] == '\0')
        return GpuSnapshot{};  // no device name available -> cannot satisfy non-"n/a" device requirement
    snap.device = tel.device_name;
    snap.available = true;
    return snap;
}

// Bounded, best-effort decline diagnostic (see <ufsecp/lbtc_gpu_ops.hpp>
// GpuLastError doc comment). Called only after a "GPU hook declined" /
// "did not handle every level" decline has already been detected -- never
// changes backend/evidence_class, purely an extra stderr line explaining
// why. Mirrors the identical helper in bench_public_ops.cpp.
void print_gpu_decline_reason(const char* op_name)
{
    const auto fn = ufsecp::lbtc::gpu_hook::g_lbtc_gpu_last_error_hook.load(std::memory_order_acquire);
    if (fn == nullptr)
        return;
    ufsecp::lbtc::gpu_hook::GpuLastError err{};
    if (fn(&err) && err.available)
        std::fprintf(stderr, "  %s decline reason: gpu_error_code=%d msg=%.220s\n", op_name, err.code, err.message);
}

// Benchmark-only tracking wrapper for merkle_root_from_leaves GPU evidence.
// The direct API intentionally falls back to CPU if the hook declines; for a
// gpu_acceleration row, the benchmark must prove that the hook actually
// handled the level reductions rather than silently falling back.
ufsecp::lbtc::gpu_hook::merkle_pair_hash_fn g_tracking_merkle_pair_hook = nullptr;
bool g_tracking_merkle_pair_called = false;
bool g_tracking_merkle_pair_declined = false;

int tracking_merkle_pair_hook(const std::uint8_t* left32, const std::uint8_t* right32,
                              std::size_t count, std::uint8_t* out32)
{
    g_tracking_merkle_pair_called = true;
    if (g_tracking_merkle_pair_hook == nullptr) {
        g_tracking_merkle_pair_declined = true;
        return -1;
    }
    const int rc = g_tracking_merkle_pair_hook(left32, right32, count, out32);
    if (rc != 0)
        g_tracking_merkle_pair_declined = true;
    return rc;
}

// One measured row. CPU-forced rows are always backend=cpu/api_correctness.
// GPU/production rows (only emitted when query_gpu_snapshot().available) are
// backend=<real>/evidence_class=gpu_acceleration; see file header.
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
    // Only set on a GPU/production row that has a matching CPU-forced row in
    // the SAME artifact -- never a one-sided speedup claim (see
    // ci/check_lbtc_gpu_workload_evidence.py one_sided_speedup rule).
    std::optional<double> speedup_compute_only_ratio;
    std::optional<double> speedup_end_to_end_ratio;
};

// Fills the measured/derived fields shared by every row. kernel_seconds
// mirrors best_seconds intentionally: this harness has no upload/kernel/
// download phase-split instrumentation on either the CPU or the GPU path
// (see file header), so kernel_seconds is never a fabricated sub-split.
void fill_measured_fields(BenchRow& row, std::size_t count, std::size_t payload_bytes,
                          double prep_seconds, double best_seconds,
                          const std::vector<std::uint8_t>& out)
{
    row.count = count;
    row.payload_bytes = payload_bytes;
    row.prep_seconds = prep_seconds;
    row.kernel_seconds = best_seconds;
    row.best_seconds = best_seconds;
    row.m_rows_per_sec = static_cast<double>(count) / best_seconds / 1e6;
    row.payload_mib_per_sec = (static_cast<double>(payload_bytes) / 1048576.0) / best_seconds;
    row.ns_per_row = best_seconds * 1e9 / static_cast<double>(count);
    row.validation_hash = sha256_hex(out);
}

// Paired ratio, computed only from two rows that both already exist in the
// same artifact -- never a one-sided claim.
void fill_paired_speedup(BenchRow& gpu_row, const BenchRow& cpu_row)
{
    if (cpu_row.kernel_seconds > 0.0 && gpu_row.kernel_seconds > 0.0)
        gpu_row.speedup_compute_only_ratio = cpu_row.kernel_seconds / gpu_row.kernel_seconds;
    if (cpu_row.best_seconds > 0.0 && gpu_row.best_seconds > 0.0)
        gpu_row.speedup_end_to_end_ratio = cpu_row.best_seconds / gpu_row.best_seconds;
}

bool write_workload_json(const std::string& path, const std::string& batch_class,
                         int iters, const HostContext& hc,
                         const std::vector<BenchRow>& result_rows)
{
    if (result_rows.empty())
        return false;

    std::ofstream out(path);
    if (!out)
        return false;

    const BenchRow& first = result_rows.front();
    const std::uint64_t payload_total = first.payload_bytes;

    out << std::setprecision(12);
    out << "{\n";
    out << "  \"schema\": \"ufsecp-lbtc-gpu-workload-benchmark-v1\",\n";
    out << "  \"target_context\": \"libbitcoin\",\n";
    out << "  \"workload\": \"" << json_escape(first.workload) << "\",\n";
    out << "  \"batch_class\": \"" << json_escape(batch_class) << "\",\n";
    out << "  \"claim_scope\": \"local bridge-free libbitcoin-shaped " << json_escape(first.workload)
        << " throughput. First row is always mode=direct-cpu-forced (GPU hook explicitly forced "
           "off, backend=cpu, evidence_class=api_correctness). A second mode=direct-production row "
           "is present only when a real GPU backend was linked/initialized/ready on this host at "
           "run time (backend/device identified via GpuTelemetry, evidence_class=gpu_acceleration, "
           "validated against the same independent oracle as the CPU-forced row); its "
           "kernel_seconds mirrors best_seconds because no upload/kernel/download phase-split "
           "instrumentation exists yet, and driver_version is always null because DeviceInfo "
           "carries no driver field -- neither is a fabricated value, both are honest "
           "absence-of-data markers\",\n";
    out << "  \"c_abi_required\": false,\n";
    out << "  \"shim_required\": false,\n";
    out << "  \"bridge_required\": false,\n";
    out << "  \"count\": " << first.count << ",\n";
    out << "  \"iters\": " << iters << ",\n";
    out << "  \"payload_bytes_total\": " << payload_total << ",\n";
    out << "  \"host_context\": {\n";
    out << "    \"compiler\": \"" << json_escape(hc.compiler) << "\",\n";
    out << "    \"cpu_model\": \"" << json_escape(hc.cpu_model) << "\",\n";
    out << "    \"turbo_disabled\": " << (hc.turbo_disabled ? "true" : "false") << ",\n";
    out << "    \"cpu_pinned\": " << (hc.cpu_pinned ? "true" : "false") << ",\n";
    out << "    \"kernel\": \"" << json_escape(hc.kernel) << "\"\n";
    out << "  },\n";
    out << "  \"results\": [\n";
    for (std::size_t i = 0; i < result_rows.size(); ++i) {
        const auto& row = result_rows[i];
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
        const bool has_speedup =
            row.speedup_compute_only_ratio.has_value() && row.speedup_end_to_end_ratio.has_value();
        if (has_speedup) {
            out << "      \"evidence_class\": \"" << json_escape(row.evidence_class) << "\",\n";
            out << "      \"speedup_vs_cpu_forced\": {\n";
            out << "        \"compute_only_ratio\": " << *row.speedup_compute_only_ratio << ",\n";
            out << "        \"end_to_end_ratio\": " << *row.speedup_end_to_end_ratio << "\n";
            out << "      }\n";
        } else {
            out << "      \"evidence_class\": \"" << json_escape(row.evidence_class) << "\"\n";
        }
        out << "    }" << (i + 1 == result_rows.size() ? "\n" : ",\n");
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
    std::printf("every workload: row 0 = mode=direct-cpu-forced backend=cpu evidence_class=api_correctness;\n"
                "  row 1 (only when a real GPU backend is linked+ready) = mode=direct-production "
                "backend=<identified> evidence_class=gpu_acceleration\n\n");

    // (workload_name, rows) -- 1 row (CPU-only host) or 2 paired rows
    // (CPU-forced + GPU/production) per workload.
    std::vector<std::pair<std::string, std::vector<BenchRow>>> workload_rows;
    bool ok = true;

    // txid_batch / wtxid_batch share this shape: variable-length serialized
    // records hashed via hash256_var_batch aliases, validated against a
    // per-row direct secp256k1::SHA256::hash256 call (independent of the
    // batch function under test). Runs a CPU-forced pass, then (only when a
    // real GPU backend is linked+ready) a second hook-active pass.
    auto run_hash_alias = [&](const char* workload_name, const char* op_name,
                              std::size_t count, std::size_t min_len, std::size_t max_len,
                              std::size_t stride, auto batch_fn, auto install_hook,
                              auto hook_load) -> bool {
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
        const std::size_t payload_bytes = static_cast<std::size_t>(payload);

        std::vector<std::uint8_t> expected(count * 32);
        for (std::size_t i = 0; i < count; ++i) {
            const auto d = secp256k1::SHA256::hash256(msgs.data() + i * stride, lens[i]);
            std::memcpy(expected.data() + i * 32, d.data(), 32);
        }

        auto run_pass = [&](std::vector<std::uint8_t>& out, double& best,
                            bool require_hook_handled, bool& hook_declined) -> bool {
            hook_declined = false;
            best = 1e100;
            const auto direct_hook = hook_load();
            if (require_hook_handled && direct_hook == nullptr) {
                std::fprintf(stderr, "%s GPU evidence requested but hook is not installed\n", op_name);
                hook_declined = true;
                return false;
            }
            for (int it = 0; it < args.iters; ++it) {
                std::fill(out.begin(), out.end(), 0);
                const auto t0 = clock_t_::now();
                if (require_hook_handled) {
                    if (direct_hook(msgs.data(), lens.data(), stride, count, out.data()) != 0) {
                        std::printf("%s GPU hook declined; skipping GPU evidence row\n", op_name);
                        print_gpu_decline_reason(op_name);
                        hook_declined = true;
                        return false;
                    }
                } else {
                    if (!batch_fn(msgs.data(), lens.data(), stride, count, out.data(), std::size_t{0})) {
                        std::fprintf(stderr, "%s returned false\n", op_name);
                        return false;
                    }
                }
                const auto dt = secs_since(t0);
                if (out != expected) {
                    std::fprintf(stderr, "%s mismatched independent HASH256 oracle\n", op_name);
                    return false;
                }
                if (dt < best)
                    best = dt;
            }
            return true;
        };

        const bool provider_linked = hook_load() != nullptr;
        std::vector<BenchRow> rows_out;

        {
            auto saved = install_hook(nullptr);
            std::vector<std::uint8_t> out(count * 32);
            double best = 0.0;
            bool hook_declined = false;
            const bool pass_ok = run_pass(out, best, false, hook_declined);
            install_hook(saved);
            if (!pass_ok)
                return false;

            BenchRow row;
            row.workload = workload_name;
            row.op = op_name;
            row.mode = "direct-cpu-forced";
            row.hook_installed = false;
            row.provider_linked = provider_linked;
            row.backend = "cpu";
            row.device = "n/a";
            fill_measured_fields(row, count, payload_bytes, prep_seconds, best, out);
            rows_out.push_back(std::move(row));
        }

        if (provider_linked) {
            const GpuSnapshot snap = query_gpu_snapshot();
            if (snap.available) {
                std::vector<std::uint8_t> out(count * 32);
                double best = 0.0;
                bool hook_declined = false;
                if (!run_pass(out, best, true, hook_declined)) {
                    if (!hook_declined)
                        return false;  // handled-but-wrong GPU output aborts the run
                } else {
                    BenchRow row;
                    row.workload = workload_name;
                    row.op = op_name;
                    row.mode = "direct-production";
                    row.hook_installed = true;
                    row.provider_linked = provider_linked;
                    row.backend = snap.backend;
                    row.device = snap.device;
                    fill_measured_fields(row, count, payload_bytes, prep_seconds, best, out);
                    row.evidence_class = "gpu_acceleration";
                    fill_paired_speedup(row, rows_out.front());
                    rows_out.push_back(std::move(row));
                }
            }
        }

        workload_rows.emplace_back(workload_name, std::move(rows_out));
        return true;
    };

    ok = ok && run_hash_alias("txid_batch", "txid_hash", sizing.txid_wtxid_count,
        kTxidMinLen, kTxidMaxLen, kTxidStride,
        &ufsecp::lbtc::txid_hash_batch,
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire); });

    ok = ok && run_hash_alias("wtxid_batch", "wtxid_hash", sizing.txid_wtxid_count,
        kWtxidMinLen, kWtxidMaxLen, kWtxidStride,
        &ufsecp::lbtc::wtxid_hash_batch,
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire); });

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
        const std::size_t payload_bytes = count * 64;

        std::vector<std::uint8_t> expected(count * 32);
        for (std::size_t i = 0; i < count; ++i) {
            std::uint8_t combined[64];
            std::memcpy(combined, left.data() + i * 32, 32);
            std::memcpy(combined + 32, right.data() + i * 32, 32);
            const auto d = secp256k1::SHA256::hash256(combined, 64);
            std::memcpy(expected.data() + i * 32, d.data(), 32);
        }

        auto run_pass = [&](std::vector<std::uint8_t>& out, double& best,
                            ufsecp::lbtc::gpu_hook::merkle_pair_hash_fn direct_hook,
                            bool& hook_declined) -> bool {
            hook_declined = false;
            best = 1e100;
            if (direct_hook == nullptr && ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire) != nullptr) {
                std::fprintf(stderr, "merkle_pair_hash GPU evidence requested but hook is not installed\n");
                hook_declined = true;
                return false;
            }
            for (int it = 0; it < args.iters; ++it) {
                std::fill(out.begin(), out.end(), 0);
                const auto t0 = clock_t_::now();
                if (direct_hook != nullptr) {
                    if (direct_hook(left.data(), right.data(), count, out.data()) != 0) {
                        std::printf("merkle_pair_hash GPU hook declined; skipping GPU evidence row\n");
                        print_gpu_decline_reason("merkle_pair_hash");
                        hook_declined = true;
                        return false;
                    }
                } else {
                    if (!ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), count, out.data())) {
                        std::fprintf(stderr, "merkle_pair_hash returned false\n");
                        return false;
                    }
                }
                const auto dt = secs_since(t0);
                if (out != expected) {
                    std::fprintf(stderr, "merkle_pair_hash mismatched independent HASH256 oracle\n");
                    return false;
                }
                if (dt < best)
                    best = dt;
            }
            return true;
        };

        const bool provider_linked =
            ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire) != nullptr;
        std::vector<BenchRow> rows_out;

        {
            auto saved = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(nullptr);
            std::vector<std::uint8_t> out(count * 32);
            double best = 0.0;
            bool hook_declined = false;
            const bool pass_ok = run_pass(out, best, nullptr, hook_declined);
            ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(saved);
            if (!pass_ok) {
                ok = false;
            } else {
                BenchRow row;
                row.workload = "merkle_pair_batch";
                row.op = "merkle_pair_hash";
                row.mode = "direct-cpu-forced";
                row.hook_installed = false;
                row.provider_linked = provider_linked;
                row.backend = "cpu";
                row.device = "n/a";
                fill_measured_fields(row, count, payload_bytes, prep_seconds, best, out);
                rows_out.push_back(std::move(row));
            }
        }

        if (ok && provider_linked) {
            const GpuSnapshot snap = query_gpu_snapshot();
            if (snap.available) {
                const auto direct_hook =
                    ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire);
                std::vector<std::uint8_t> out(count * 32);
                double best = 0.0;
                bool hook_declined = false;
                if (!run_pass(out, best, direct_hook, hook_declined)) {
                    if (!hook_declined)
                        ok = false;
                } else {
                    BenchRow row;
                    row.workload = "merkle_pair_batch";
                    row.op = "merkle_pair_hash";
                    row.mode = "direct-production";
                    row.hook_installed = true;
                    row.provider_linked = provider_linked;
                    row.backend = snap.backend;
                    row.device = snap.device;
                    fill_measured_fields(row, count, payload_bytes, prep_seconds, best, out);
                    row.evidence_class = "gpu_acceleration";
                    fill_paired_speedup(row, rows_out.front());
                    rows_out.push_back(std::move(row));
                }
            }
        }

        if (ok)
            workload_rows.emplace_back("merkle_pair_batch", std::move(rows_out));
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
        const std::size_t payload_bytes = trees * leaves_per_tree * 32;

        std::vector<std::uint8_t> expected(trees * 32);
        for (std::size_t t = 0; t < trees; ++t)
            independent_merkle_root(leaves.data() + t * leaves_per_tree * 32, leaves_per_tree,
                                    expected.data() + t * 32);

        std::vector<std::uint8_t> scratch(leaves_per_tree * 64);

        auto run_pass = [&](std::vector<std::uint8_t>& out, double& best) -> bool {
            best = 1e100;
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
                    return false;
                }
                if (out != expected) {
                    std::fprintf(stderr, "merkle_root_from_leaves mismatched independent merkle oracle\n");
                    return false;
                }
                if (dt < best)
                    best = dt;
            }
            return true;
        };

        const bool provider_linked =
            ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire) != nullptr;
        std::vector<BenchRow> rows_out;

        {
            auto saved = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(nullptr);
            std::vector<std::uint8_t> out(trees * 32);
            double best = 0.0;
            const bool pass_ok = run_pass(out, best);
            ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(saved);
            if (!pass_ok) {
                ok = false;
            } else {
                BenchRow row;
                row.workload = "merkle_root_batch";
                row.op = "merkle_root_from_leaves";
                row.mode = "direct-cpu-forced";
                row.hook_installed = false;
                row.provider_linked = provider_linked;
                row.backend = "cpu";
                row.device = "n/a";
                fill_measured_fields(row, trees, payload_bytes, prep_seconds, best, out);
                rows_out.push_back(std::move(row));
            }
        }

        if (ok && provider_linked) {
            const GpuSnapshot snap = query_gpu_snapshot();
            if (snap.available) {
                auto saved = ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire);
                g_tracking_merkle_pair_hook = saved;
                g_tracking_merkle_pair_called = false;
                g_tracking_merkle_pair_declined = false;
                ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(&tracking_merkle_pair_hook);

                std::vector<std::uint8_t> out(trees * 32);
                double best = 0.0;
                const bool pass_ok = run_pass(out, best);
                ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(saved);
                g_tracking_merkle_pair_hook = nullptr;

                if (!pass_ok || !g_tracking_merkle_pair_called || g_tracking_merkle_pair_declined) {
                    if (g_tracking_merkle_pair_declined) {
                        std::printf("merkle_root_from_leaves GPU hook declined; skipping GPU evidence row\n");
                        print_gpu_decline_reason("merkle_root_from_leaves");
                    } else if (pass_ok) {
                        std::fprintf(stderr, "merkle_root_from_leaves GPU hook did not handle every level\n");
                        ok = false;
                    } else {
                        ok = false;
                    }
                } else {
                    BenchRow row;
                    row.workload = "merkle_root_batch";
                    row.op = "merkle_root_from_leaves";
                    row.mode = "direct-production";
                    row.hook_installed = true;
                    row.provider_linked = provider_linked;
                    row.backend = snap.backend;
                    row.device = snap.device;
                    fill_measured_fields(row, trees, payload_bytes, prep_seconds, best, out);
                    row.evidence_class = "gpu_acceleration";
                    fill_paired_speedup(row, rows_out.front());
                    rows_out.push_back(std::move(row));
                }
            }
        }

        if (ok)
            workload_rows.emplace_back("merkle_root_batch", std::move(rows_out));
    }

    if (!ok)
        return 1;

    for (const auto& [name, rows]: workload_rows) {
        for (const auto& row: rows) {
            std::printf("   %-18s %-18s %8.2f M rows/s %9.2f MiB/s %8.1f ns/row  prep=%.6fs "
                        "backend=%-6s device=%s\n",
                name.c_str(), row.mode.c_str(), row.m_rows_per_sec, row.payload_mib_per_sec,
                row.ns_per_row, row.prep_seconds, row.backend.c_str(), row.device.c_str());
        }
    }

    if (!args.json_dir.empty()) {
        const HostContext hc = detect_host_context();
        for (const auto& [name, rows]: workload_rows) {
            const std::string path = args.json_dir + "/" + name + ".json";
            if (!write_workload_json(path, args.batch_class, args.iters, hc, rows)) {
                std::fprintf(stderr, "failed to write JSON artifact: %s\n", path.c_str());
                return 1;
            }
            std::printf("wrote JSON artifact: %s\n", path.c_str());
        }
    }

    return 0;
}
