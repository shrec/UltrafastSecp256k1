// ============================================================================
// test_regression_precompute_gcontext_race.cpp
// ============================================================================
// Regression for PRECOMPUTE-GCONTEXT-UAF (2026-06-10): a use-after-free / data
// race in the fixed-base generator multiply.
//
//   scalar_mul_generator() and batch_scalar_mul_generator() (src/cpu/src/
//   precompute.cpp) take g_mutex, bound a reference `PrecomputeContext const&
//   ctx = *g_context`, then UNLOCK and used ctx for the whole scalar-mul loop.
//   A concurrent configure_fixed_base() does `g_context.reset()` under the lock
//   — freeing the PrecomputeContext the first thread still referenced (UAF).
//
//   Fix: g_context is now a std::shared_ptr; the readers take a local shared_ptr
//   snapshot under the lock (`ctx_ptr = g_context`) before unlocking, so a
//   concurrent reset() cannot free the table while a read is in flight.
//
// Two checks:
//   [1] Source-scan (deterministic): g_context is shared_ptr and both readers
//       take a snapshot — the structural guard against regressing to the
//       unlock-then-use-raw-reference pattern.
//   [2] Concurrency smoke (best-effort; the real teeth are under TSan/ASan CI):
//       N reader threads compute scalar_mul_generator / batch against the
//       independent reference Point::generator().scalar_mul(k) while another
//       thread hammers configure_fixed_base() to trigger g_context.reset().
//       All results must stay correct and the process must not crash.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <array>
#include <atomic>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"

#include "audit_check.hpp"

using namespace secp256k1;
using secp256k1::fast::Scalar;
using secp256k1::fast::Point;
using secp256k1::fast::FixedBaseConfig;

static int g_pass = 0, g_fail = 0;

// Repair (issue #335 acceptance repair, round 5): route through the shared,
// UFSECP_SOURCE_ROOT-aware audit_read_source_file() (audit_check.hpp) so
// this resolves identically whether unified_audit_runner is invoked from
// the repo root or from a CWD unrelated to the repo (e.g. /tmp) -- the
// previous bounded CWD-relative walk-up (depth<=6) hard-failed correctly in
// the latter case but did not resolve the source, so real check counts
// diverged between the two invocations.
static std::string read_source_file(const char* rel_path) {
    return audit_read_source_file(rel_path);
}

static std::size_t count_occurrences(const std::string& hay, const std::string& needle) {
    std::size_t n = 0, pos = 0;
    while ((pos = hay.find(needle, pos)) != std::string::npos) { ++n; pos += needle.size(); }
    return n;
}

// ── [1] Source-scan: shared_ptr declaration + snapshot in both readers ────────
static void test_gcontext_shared_ptr_snapshot_source_scan() {
    printf("[1] PRECOMPUTE-GCONTEXT-UAF: precompute.cpp — shared_ptr g_context + snapshot\n");

    std::string src = read_source_file("src/cpu/src/precompute.cpp");
    if (src.empty()) src = read_source_file("precompute.cpp");
    CHECK(!src.empty(), "precompute.cpp must be readable (in-tree source always exists)");
    if (src.empty()) return;

    // g_context must be a shared_ptr (so a snapshot keeps the table alive past unlock).
    bool const decl_shared =
        src.find("std::shared_ptr<PrecomputeContext> g_context") != std::string::npos;
    CHECK(decl_shared, "precompute.cpp: g_context declared as std::shared_ptr (PRECOMPUTE-GCONTEXT-UAF)");

    // Regression guard: the unique_ptr declaration must be gone.
    bool const no_unique_decl =
        src.find("std::unique_ptr<PrecomputeContext> g_context") == std::string::npos;
    CHECK(no_unique_decl, "precompute.cpp: no std::unique_ptr<PrecomputeContext> g_context remains");

    // Both unlock-then-read fast paths (scalar_mul_generator, batch_scalar_mul_generator)
    // must take a local shared_ptr snapshot under the lock before unlocking. (The
    // lock-held helpers — save_precompute_cache_locked, ensure_built_locked,
    // scalar_mul_generator_glv_predecomposed — may read *g_context directly: they never
    // unlock, so a concurrent reset() cannot race them. So we assert the snapshot exists
    // in the two readers, NOT the absence of every raw *g_context deref.)
    std::size_t const snapshots =
        count_occurrences(src, "std::shared_ptr<PrecomputeContext> const ctx_ptr = g_context");
    CHECK(snapshots >= 2,
          "precompute.cpp: scalar_mul_generator AND batch_scalar_mul_generator snapshot g_context");
}

// ── [2] Concurrency smoke: readers vs reference under concurrent reconfigure ──
static void test_gcontext_concurrent_reconfigure() {
    printf("[2] PRECOMPUTE-GCONTEXT-UAF: concurrent scalar_mul_generator + configure_fixed_base\n");

    // Independent reference scalars and their G·k via the general (non-fixed-base) path.
    constexpr int K = 4;
    std::array<Scalar, K> ks{};
    std::array<Point, K> refs{};
    for (int i = 0; i < K; ++i) {
        std::array<std::uint8_t, 32> b{};
        b[31] = static_cast<std::uint8_t>(0x11 * (i + 1));
        b[0]  = static_cast<std::uint8_t>(i + 1);
        Scalar k;
        if (!Scalar::parse_bytes_strict_nonzero(b.data(), k)) { CHECK(false, "ref scalar parse"); return; }
        ks[i] = k;
        refs[i] = Point::generator().scalar_mul(k);  // independent of g_context / precompute table
    }

    // Small, cache-less config so each rebuild after reset() is cheap and touches no disk.
    FixedBaseConfig small_cfg;
    small_cfg.window_bits = 4;
    small_cfg.enable_glv  = false;
    small_cfg.use_cache   = false;
    secp256k1::fast::configure_fixed_base(small_cfg);

    std::atomic<bool> all_ok{true};
    std::atomic<bool> stop{false};

    auto cmp = [](const Point& a, const Point& b) {
        return a.to_compressed() == b.to_compressed();
    };

    auto reader = [&]() {
        for (int iter = 0; iter < 150 && all_ok.load(); ++iter) {
            for (int i = 0; i < K; ++i) {
                Point got = secp256k1::fast::scalar_mul_generator(ks[i]);
                if (!cmp(got, refs[i])) { all_ok.store(false); return; }
            }
            // Exercise the batch reader too.
            std::array<Point, K> out{};
            secp256k1::fast::batch_scalar_mul_generator(ks.data(), out.data(), K);
            for (int i = 0; i < K; ++i) {
                if (!cmp(out[i], refs[i])) { all_ok.store(false); return; }
            }
        }
    };

    // Reconfigure thread: hammer configure_fixed_base() (→ g_context.reset()) concurrently.
    auto reconfigurer = [&]() {
        for (int iter = 0; iter < 60 && !stop.load(); ++iter) {
            secp256k1::fast::configure_fixed_base(small_cfg);
            std::this_thread::yield();
        }
    };

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) threads.emplace_back(reader);
    threads.emplace_back(reconfigurer);
    for (auto& th : threads) th.join();
    stop.store(true);

    CHECK(all_ok.load(),
          "concurrent scalar_mul_generator results stay correct under configure_fixed_base reset");

    // Restore the default fixed-base configuration so other audit modules in the
    // same process are unaffected by the small/cache-less config used above.
    secp256k1::fast::configure_fixed_base(FixedBaseConfig{});
    printf("  OK: restored default fixed-base config\n");
}

// ── entry point ──────────────────────────────────────────────────────────────
int test_regression_precompute_gcontext_race_run() {
    g_pass = 0; g_fail = 0;
    printf("======================================================================\n");
    printf("  Regression: precompute g_context use-after-free race\n");
    printf("  Fix: PRECOMPUTE-GCONTEXT-UAF — shared_ptr snapshot under lock\n");
    printf("======================================================================\n\n");

    test_gcontext_shared_ptr_snapshot_source_scan();
    printf("\n");
    test_gcontext_concurrent_reconfigure();
    printf("\n");

    printf("[regression_precompute_gcontext_race] %d/%d checks passed\n",
           g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_precompute_gcontext_race_run(); }
#endif
