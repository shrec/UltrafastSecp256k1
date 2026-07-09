// Direct C++ libbitcoin public-data batch-op benchmark.
//
// Measures the canonical bridge-free surface:
//   ufsecp::lbtc::{xonly_validate,pubkey_validate,taproot_commitment_verify,
//                  tagged_hash,tagged_hash_var,hash256,hash256_var,
//                  txid_hash,wtxid_hash,merkle_pair_hash}_batch
//
// txid_hash_batch/wtxid_hash_batch are semantic aliases over hash256_var_batch
// (same hook: g_lbtc_hash256_var_hook) — benchmarked here with realistic
// tx-sized payload ranges instead of the generic var range used above, purely
// for reporting purposes; the underlying computation is identical.
// merkle_pair_hash_batch has its own dedicated hook (g_lbtc_merkle_pair_hook)
// and fixed 32+32-byte column inputs (no length array/stride).
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
#include <optional>
#include <string>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/utsname.h>
#endif

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

// Representative tx-sized payload ranges for txid_hash_batch/wtxid_hash_batch
// (semantic aliases over hash256_var_batch). Not tied to the CLI args above —
// these approximate realistic non-witness / witness-included serialized
// transaction sizes, distinct from the generic 80..stride var range used for
// the plain hash256_var row.
constexpr std::size_t kTxidMinLen = 190;
constexpr std::size_t kTxidMaxLen = 400;
constexpr std::size_t kTxidStride = 400;
constexpr std::size_t kWtxidMinLen = 220;
constexpr std::size_t kWtxidMaxLen = 600;
constexpr std::size_t kWtxidStride = 600;

struct BenchResult {
    std::string op;
    std::string mode;
    bool hook_installed = false;
    // Evidence-gate fields (workingdocs/libbitcoin_gpu_workloads/evidence_matrix_claude.json,
    // schema ufsecp-lbtc-public-ops-benchmark-v1 variant). backend/device are
    // "cpu"/"n/a" by default and are overwritten with a REAL identification
    // (queried via ufsecp::lbtc::gpu_hook::g_lbtc_gpu_telemetry_hook, see
    // query_gpu_snapshot() below) only on a production row where the hook was
    // actually installed AND a GPU backend is linked/initialized/ready.
    // evidence_class stays "api_correctness" UNCONDITIONALLY regardless of
    // backend -- this harness still has no upload/kernel/download phase-split
    // instrumentation, so it structurally cannot produce gpu_acceleration
    // evidence under schema v1 (see ci/check_lbtc_gpu_workload_evidence.py).
    // provider_linked only records whether a GPU hook was self-installed in
    // this binary BEFORE the row forced it off, not which backend it is.
    bool provider_linked = false;
    std::string backend = "cpu";
    std::string device = "n/a";
    std::size_t payload_bytes = 0;
    std::size_t count = 0;
    // Phase timing: this harness measures one wall-clock span per call (see
    // bench_output) with no instrumented upload/kernel/download split, so
    // prep/upload/download stay null and kernel_seconds mirrors best_seconds
    // (the entire measured span) -- never a fabricated sub-split.
    std::optional<double> prep_seconds;
    std::optional<double> upload_seconds;
    std::optional<double> kernel_seconds;
    std::optional<double> download_seconds;
    double best_seconds = 0.0;
    double m_rows_per_sec = 0.0;
    double payload_mib_per_sec = 0.0;
    double ns_per_row = 0.0;
    std::string validation_hash;
    std::string validation_status = "matched_reference";
    std::string evidence_class = "api_correctness";
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

// Evidence-gate provenance: host/build facts, gathered once per run. Linux-only
// today (the dev/CI machines this harness runs on); non-Linux hosts fall back to
// honest "unknown"/false rather than fabricating a value.
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
        // Only honest source available without root/owner tooling: the
        // intel_pstate no_turbo flag. Absent/unreadable -> honestly "unknown
        // whether turbo is disabled", reported as false (cannot claim it IS
        // disabled without evidence).
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

// -- GPU telemetry snapshot (benchmark-only; see <ufsecp/lbtc_gpu_ops.hpp>
// GpuTelemetry). Never used on any production/hot code path -- this
// benchmark is the only caller. Mirrors the identical helper in
// bench_workloads.cpp. -------------------------------------------------
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
                  bool provider_linked, std::size_t payload_bytes,
                  std::vector<std::uint8_t>& out,
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
    r.provider_linked = provider_linked;
    r.payload_bytes = payload_bytes;
    r.count = args.count;
    r.best_seconds = best;
    r.m_rows_per_sec = static_cast<double>(args.count) / best / 1e6;
    r.payload_mib_per_sec = (static_cast<double>(payload_bytes) / 1048576.0) / best;
    r.ns_per_row = best * 1e9 / static_cast<double>(args.count);
    // kernel_seconds mirrors the single measured span (best_seconds): this
    // harness has no separate prep/upload/download instrumentation, so those
    // stay null (see BenchResult comment) rather than inventing a split.
    r.kernel_seconds = best;
    r.validation_hash = sha256_hex(out);  // out == gold, verified above
    results.push_back(r);

    std::printf("   %-24s %-18s hook=%-3s %8.2f M rows/s %9.2f MiB/s %8.1f ns/row\n",
        op, mode, hook_installed ? "yes" : "no", r.m_rows_per_sec,
        r.payload_mib_per_sec, r.ns_per_row);
    return true;
}

void write_optional_seconds(std::ofstream& out, const std::optional<double>& v)
{
    if (v.has_value())
        out << *v;
    else
        out << "null";
}

bool write_json_artifact(const std::string& path, const Args& args,
                         const HostContext& hc,
                         const std::vector<BenchResult>& results)
{
    std::ofstream out(path);
    if (!out)
        return false;

    out << std::setprecision(12);
    out << "{\n";
    out << "  \"schema\": \"ufsecp-lbtc-public-ops-benchmark-v1\",\n";
    out << "  \"target_context\": \"libbitcoin-direct-cpp\",\n";
    out << "  \"claim_scope\": \"local bridge-free libbitcoin public-data batch-op throughput; production rows use GPU only when the op hook is installed and accepts the batch. A production row's backend/device are identified via GpuTelemetry (real, not fabricated) when a GPU backend is linked/initialized/ready on this host; otherwise they stay cpu/n/a. This harness still has no upload/kernel/download phase-split instrumentation, so every row is api_correctness evidence, never gpu_acceleration, regardless of backend -- see ci/check_lbtc_gpu_workload_evidence.py schema v1 rule\",\n";
    out << "  \"c_abi_required\": false,\n";
    out << "  \"shim_required\": false,\n";
    out << "  \"bridge_required\": false,\n";
    out << "  \"count\": " << args.count << ",\n";
    out << "  \"iters\": " << args.iters << ",\n";
    out << "  \"fixed_len\": " << args.fixed_len << ",\n";
    out << "  \"stride\": " << args.stride << ",\n";
    out << "  \"var_min\": " << args.var_min << ",\n";
    out << "  \"var_max\": " << args.var_max << ",\n";
    out << "  \"phase_timing_available\": false,\n";
    out << "  \"host_context\": {\n";
    out << "    \"compiler\": \"" << json_escape(hc.compiler) << "\",\n";
    out << "    \"cpu_model\": \"" << json_escape(hc.cpu_model) << "\",\n";
    out << "    \"turbo_disabled\": " << (hc.turbo_disabled ? "true" : "false") << ",\n";
    out << "    \"cpu_pinned\": " << (hc.cpu_pinned ? "true" : "false") << ",\n";
    out << "    \"kernel\": \"" << json_escape(hc.kernel) << "\"\n";
    out << "  },\n";
    out << "  \"results\": [\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        out << "    {\n";
        out << "      \"op\": \"" << json_escape(r.op) << "\",\n";
        out << "      \"mode\": \"" << json_escape(r.mode) << "\",\n";
        out << "      \"hook_installed\": " << (r.hook_installed ? "true" : "false") << ",\n";
        out << "      \"provider_linked\": " << (r.provider_linked ? "true" : "false") << ",\n";
        out << "      \"backend\": \"" << json_escape(r.backend) << "\",\n";
        out << "      \"device\": \"" << json_escape(r.device) << "\",\n";
        out << "      \"driver_version\": null,\n";
        out << "      \"count\": " << r.count << ",\n";
        out << "      \"payload_bytes_per_iter\": " << r.payload_bytes << ",\n";
        out << "      \"prep_seconds\": "; write_optional_seconds(out, r.prep_seconds); out << ",\n";
        out << "      \"upload_seconds\": "; write_optional_seconds(out, r.upload_seconds); out << ",\n";
        out << "      \"kernel_seconds\": "; write_optional_seconds(out, r.kernel_seconds); out << ",\n";
        out << "      \"download_seconds\": "; write_optional_seconds(out, r.download_seconds); out << ",\n";
        out << "      \"best_seconds\": " << r.best_seconds << ",\n";
        out << "      \"m_rows_per_sec\": " << r.m_rows_per_sec << ",\n";
        out << "      \"payload_mib_per_sec\": " << r.payload_mib_per_sec << ",\n";
        out << "      \"ns_per_row\": " << r.ns_per_row << ",\n";
        out << "      \"validation_hash\": \"" << json_escape(r.validation_hash) << "\",\n";
        out << "      \"validation_status\": \"" << json_escape(r.validation_status) << "\",\n";
        out << "      \"evidence_class\": \"" << json_escape(r.evidence_class) << "\"\n";
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

    if (mul_overflows(args.count, kWtxidStride) || mul_overflows(args.count, std::size_t{64})) {
        std::fprintf(stderr, "count too large for tx-sized / merkle-pair benchmark buffers\n");
        return 2;
    }

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

    // txid_hash_batch / wtxid_hash_batch: realistic tx-sized variable payloads
    // (see kTxid*/kWtxid* above). Same shape as var_msgs/var_lens but a
    // distinct, more representative length range.
    std::vector<std::uint8_t> tx_msgs(args.count * kTxidStride);
    std::vector<std::uint32_t> tx_lens(args.count);
    std::uint64_t tx_payload = 0;

    std::vector<std::uint8_t> wtx_msgs(args.count * kWtxidStride);
    std::vector<std::uint32_t> wtx_lens(args.count);
    std::uint64_t wtx_payload = 0;

    // merkle_pair_hash_batch: fixed 32-byte left/right columns (SoA), no
    // length array — any 32 bytes are a valid Merkle-tree node hash for the
    // purpose of this throughput benchmark.
    std::vector<std::uint8_t> mp_left(args.count * 32);
    std::vector<std::uint8_t> mp_right(args.count * 32);

    const auto tag_hash = secp256k1::SHA256::hash("BIP0340/test", 12);

    std::printf("generating valid public-data rows...\n");
    const std::size_t span = args.var_max - args.var_min + 1;
    const std::size_t tx_span = kTxidMaxLen - kTxidMinLen + 1;
    const std::size_t wtx_span = kWtxidMaxLen - kWtxidMinLen + 1;
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

        for (std::size_t j = 0; j < kTxidStride; ++j)
            tx_msgs[i * kTxidStride + j] = rng8();
        const auto tlen = kTxidMinLen + static_cast<std::size_t>(rng64() % tx_span);
        tx_lens[i] = static_cast<std::uint32_t>(tlen);
        tx_payload += static_cast<std::uint64_t>(tlen);

        for (std::size_t j = 0; j < kWtxidStride; ++j)
            wtx_msgs[i * kWtxidStride + j] = rng8();
        const auto wlen = kWtxidMinLen + static_cast<std::size_t>(rng64() % wtx_span);
        wtx_lens[i] = static_cast<std::uint32_t>(wlen);
        wtx_payload += static_cast<std::uint64_t>(wlen);

        for (std::size_t b = 0; b < 32; ++b) {
            mp_left[i * 32 + b] = rng8();
            mp_right[i * 32 + b] = rng8();
        }
    }

    std::vector<BenchResult> results;
    std::vector<std::uint8_t> gold_validate(args.count);
    std::vector<std::uint8_t> out_validate(args.count);
    std::vector<std::uint8_t> gold_hash(args.count * 32);
    std::vector<std::uint8_t> out_hash(args.count * 32);

    auto bench_validate_op = [&](const char* op, std::size_t payload,
                                 auto install_hook, auto hook_load, auto call,
                                 auto hook_call) {
        // provider_linked: whether a GPU hook was already self-installed in
        // this binary BEFORE this op forces it off for the CPU-forced row --
        // an inspectable runtime fact independent of hook_installed (which
        // tracks whether the hook was active for THIS row).
        const bool provider_linked = hook_load() != nullptr;
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
            provider_linked, payload, out_validate, gold_validate,
            [&](std::vector<std::uint8_t>& dst) {
                return call(dst);
            });
        install_hook(saved);
        if (!forced_ok)
            return false;
        const bool prod_hook = hook_load() != nullptr;
        const bool prod_ok = bench_output(results, args, op, "direct-production", prod_hook,
            provider_linked, payload, out_validate, gold_validate,
            [&](std::vector<std::uint8_t>& dst) {
                return call(dst);
            });
        // Real backend/device identification for this production row, only
        // when the hook actually handled this row AND a GPU backend is linked/
        // initialized/ready -- otherwise backend/device stay the honest
        // cpu/n/a default (see BenchResult / query_gpu_snapshot()).
        if (prod_ok && prod_hook) {
            const GpuSnapshot snap = query_gpu_snapshot();
            if (snap.available) {
                std::vector<std::uint8_t> probe(args.count);
                if (hook_call(probe) == 0 && probe == gold_validate) {
                    results.back().backend = snap.backend;
                    results.back().device = snap.device;
                } else {
                    std::printf("%s GPU hook did not independently handle the batch; keeping backend=cpu\n", op);
                }
            }
        }
        return prod_ok;
    };

    auto bench_hash_op = [&](const char* op, std::size_t payload,
                             auto install_hook, auto hook_load, auto call,
                             auto hook_call) {
        const bool provider_linked = hook_load() != nullptr;
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
            provider_linked, payload, out_hash, gold_hash,
            [&](std::vector<std::uint8_t>& dst) {
                return call(dst);
            });
        install_hook(saved);
        if (!forced_ok)
            return false;
        const bool prod_hook = hook_load() != nullptr;
        const bool prod_ok = bench_output(results, args, op, "direct-production", prod_hook,
            provider_linked, payload, out_hash, gold_hash,
            [&](std::vector<std::uint8_t>& dst) {
                return call(dst);
            });
        // See bench_validate_op above: honest backend/device identification,
        // only when this row's hook actually handled the batch AND a GPU
        // backend is ready.
        if (prod_ok && prod_hook) {
            const GpuSnapshot snap = query_gpu_snapshot();
            if (snap.available) {
                std::vector<std::uint8_t> probe(args.count * 32);
                if (hook_call(probe) == 0 && probe == gold_hash) {
                    results.back().backend = snap.backend;
                    results.back().device = snap.device;
                } else {
                    std::printf("%s GPU hook did not independently handle the batch; keeping backend=cpu\n", op);
                }
            }
        }
        return prod_ok;
    };

    bool ok = true;
    ok = ok && bench_validate_op("xonly_validate", args.count * 32,
        ufsecp::lbtc::gpu_hook::install_lbtc_xonly_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_xonly_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::xonly_validate_batch(xonly.data(), args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_xonly_hook.load(std::memory_order_acquire)(
                xonly.data(), args.count, dst.data());
        });

    ok = ok && bench_validate_op("pubkey_validate", args.count * 33,
        ufsecp::lbtc::gpu_hook::install_lbtc_pubkey_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_pubkey_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::pubkey_validate_batch(pub33.data(), args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_pubkey_hook.load(std::memory_order_acquire)(
                pub33.data(), args.count, dst.data());
        });

    ok = ok && bench_validate_op("commitment_verify", args.count * (32 * 3 + 1),
        ufsecp::lbtc::gpu_hook::install_lbtc_commit_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_commit_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::taproot_commitment_verify_batch(
                internal_x.data(), tweak.data(), tweaked_x.data(), parity.data(),
                args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_commit_hook.load(std::memory_order_acquire)(
                internal_x.data(), tweak.data(), tweaked_x.data(), parity.data(),
                args.count, dst.data());
        });

    ok = ok && bench_hash_op("tagged_hash", args.count * args.fixed_len,
        ufsecp::lbtc::gpu_hook::install_lbtc_tagged_hash_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::tagged_hash_batch(
                tag_hash.data(), fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_hook.load(std::memory_order_acquire)(
                tag_hash.data(), fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        });

    ok = ok && bench_hash_op("tagged_hash_tag_overload", args.count * args.fixed_len,
        ufsecp::lbtc::gpu_hook::install_lbtc_tagged_hash_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::tagged_hash_batch(
                "BIP0340/test", 12, fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_hook.load(std::memory_order_acquire)(
                tag_hash.data(), fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        });

    ok = ok && bench_hash_op("tagged_hash_var", static_cast<std::size_t>(var_payload),
        ufsecp::lbtc::gpu_hook::install_lbtc_tagged_hash_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_var_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::tagged_hash_var_batch(
                tag_hash.data(), var_msgs.data(), var_lens.data(), args.stride,
                args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_tagged_hash_var_hook.load(std::memory_order_acquire)(
                tag_hash.data(), var_msgs.data(), var_lens.data(), args.stride,
                args.count, dst.data());
        });

    ok = ok && bench_hash_op("hash256", args.count * args.fixed_len,
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::hash256_batch(
                fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_hook.load(std::memory_order_acquire)(
                fixed_msgs.data(), args.fixed_len, args.count, dst.data());
        });

    ok = ok && bench_hash_op("hash256_var", static_cast<std::size_t>(var_payload),
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::hash256_var_batch(
                var_msgs.data(), var_lens.data(), args.stride, args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire)(
                var_msgs.data(), var_lens.data(), args.stride, args.count, dst.data());
        });

    // txid_hash_batch / wtxid_hash_batch are semantic aliases over
    // hash256_var_batch — they route through the SAME hook
    // (g_lbtc_hash256_var_hook). Benchmarked separately here only to report
    // throughput at realistic tx-sized payloads instead of the generic
    // 80..stride var range above.
    ok = ok && bench_hash_op("txid_hash", static_cast<std::size_t>(tx_payload),
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::txid_hash_batch(
                tx_msgs.data(), tx_lens.data(), kTxidStride, args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire)(
                tx_msgs.data(), tx_lens.data(), kTxidStride, args.count, dst.data());
        });

    ok = ok && bench_hash_op("wtxid_hash", static_cast<std::size_t>(wtx_payload),
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::wtxid_hash_batch(
                wtx_msgs.data(), wtx_lens.data(), kWtxidStride, args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_hash256_var_hook.load(std::memory_order_acquire)(
                wtx_msgs.data(), wtx_lens.data(), kWtxidStride, args.count, dst.data());
        });

    ok = ok && bench_hash_op("merkle_pair_hash", args.count * std::size_t{64},
        ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook,
        [] { return ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire); },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::merkle_pair_hash_batch(
                mp_left.data(), mp_right.data(), args.count, dst.data());
        },
        [&](std::vector<std::uint8_t>& dst) {
            return ufsecp::lbtc::gpu_hook::g_lbtc_merkle_pair_hook.load(std::memory_order_acquire)(
                mp_left.data(), mp_right.data(), args.count, dst.data());
        });

    if (!ok)
        return 1;

    if (!args.json_path.empty()) {
        const HostContext hc = detect_host_context();
        if (!write_json_artifact(args.json_path, args, hc, results)) {
            std::fprintf(stderr, "failed to write JSON benchmark artifact: %s\n",
                args.json_path.c_str());
            return 1;
        }
        std::printf("\nwrote JSON artifact: %s\n", args.json_path.c_str());
    }

    return 0;
}
