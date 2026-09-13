// ============================================================================
// test_regression_precompute_noop_reconfigure.cpp
// ============================================================================
// Regression: re-applying the SAME fixed-base configuration must not throw away
// the built table and recompute it.
//
// `configure_fixed_base()` unconditionally called `invalidate_context_locked()`,
// so a caller that configured defensively paid a full rebuild every time. The
// library's own `Selftest()` (src/cpu/src/selftest.cpp) does exactly that --
//
//     FixedBaseConfig cfg{};
//     ... optional env overrides ...
//     configure_fixed_base(cfg);
//     ensure_fixed_base_ready();
//
// -- at the top of EVERY invocation, so audit/test_exploit_selftest_api.cpp,
// which runs the selftest eight times to prove idempotency, rebuilt the
// ~250 MB window_bits=18 table eight times.
//
// Why it stayed invisible
//   While `use_cache` defaulted to true, each "rebuild" was a load of
//   cache_w18.bin from disk rather than a recomputation, so the waste showed up
//   as I/O nobody was measuring. Turning that default off -- because the file
//   was being left in the caller's working directory (reported by Eric
//   Voskuil) -- removed the padding and the real cost surfaced at once:
//
//     exploit_selftest_api   6.6 s  ->  120 s ctest timeout (every platform)
//     full ctest suite       496 s  ->  1670 s
//
//   Measured here on one binary with SECP256K1_FIXED_BASE_DISK_CACHE toggled
//   and nothing else changed:
//
//     cache ON, cold   6.0 s
//     cache ON, warm   4.3 s
//     cache OFF       23.8 - 24.3 s
//
//   Neither the cache nor a longer ctest timeout is the fix. A no-op
//   reconfiguration should not cost a quarter-gigabyte of arithmetic.
//
// What this pins
//   PNR-1  a second configure_fixed_base() with an identical config keeps the
//          published context: same identity, same epoch
//   PNR-2  the NEGATIVE CONTROL -- changing window_bits DOES invalidate, so
//          PNR-1 cannot be satisfied by never invalidating at all
//   PNR-3  a config that differs only in a cache field also invalidates: the
//          whole config is snapshotted into ctx->config, so keeping a context
//          across any change would let g_config and ctx->config disagree
//   PNR-4  the table still works after the skipped rebuild (the surviving
//          context is the right one, not a stale or empty table)
//
// Identity and epoch come from the SECP256K1_PRECOMPUTE_TEST_HOOKS diagnostic
// surface added for issue #336; without those hooks this module reports a skip
// rather than passing on nothing.
// ============================================================================

#include <cstdio>
#include <cstdint>

#include "secp256k1/precompute.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;

namespace {

using secp256k1::fast::FixedBaseConfig;

// Small window on purpose: this module is about WHETHER a rebuild happens, not
// how long one takes. window_bits=18 (the shipped default) would make every
// invalidation in here allocate ~250 MB.
constexpr unsigned kWindow = 6;

FixedBaseConfig base_config() {
    FixedBaseConfig cfg;
    cfg.window_bits = kWindow;
    cfg.use_cache = false;      // pin the mode; the disk cache is a separate axis
    return cfg;
}

// Force the table into existence and touch it, so a context that was silently
// dropped cannot go unnoticed.
void use_the_table() {
    secp256k1::fast::ensure_fixed_base_ready();
    volatile bool sink = secp256k1::fast::scalar_mul_generator(
        secp256k1::fast::Scalar::from_uint64(9)).is_infinity();
    (void)sink;
}

} // namespace

int test_regression_precompute_noop_reconfigure_run() {
    g_pass = 0; g_fail = 0;
    std::printf("======================================================================\n");
    std::printf("  Regression: re-applying the same fixed-base config rebuilds nothing\n");
    std::printf("======================================================================\n\n");

#if !defined(SECP256K1_PRECOMPUTE_TEST_HOOKS)
    std::printf("  [skip] built without SECP256K1_PRECOMPUTE_TEST_HOOKS -- context\n");
    std::printf("         identity/epoch are not observable, so there is nothing this\n");
    std::printf("         module can assert. Build the *_standalone target for it.\n");
    std::printf("\n[regression_precompute_noop_reconfigure] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return 0;
#else
    FixedBaseConfig const cfg = base_config();

    secp256k1::fast::configure_fixed_base(cfg);
    use_the_table();

    auto const first = secp256k1::fast::precompute_context_diagnostics();
    CHECK(first.published_identity != 0,
          "PNR-0: a context is published after the first configure + use");
    if (first.published_identity == 0) {
        std::printf("\n[regression_precompute_noop_reconfigure] %d/%d checks passed\n",
                    g_pass, g_pass + g_fail);
        return 1;
    }
    std::printf("  after first build: identity=%#llx epoch=%llu window_bits=%u\n",
                static_cast<unsigned long long>(first.published_identity),
                static_cast<unsigned long long>(first.published_epoch),
                first.published_window_bits);

    // ── PNR-1: identical config, nothing is rebuilt ──────────────────────────
    secp256k1::fast::configure_fixed_base(cfg);
    use_the_table();
    auto const again = secp256k1::fast::precompute_context_diagnostics();

    CHECK(again.published_identity == first.published_identity,
          "PNR-1: re-applying an identical FixedBaseConfig keeps the SAME context "
          "(identity unchanged) -- Selftest() reconfigures on every call, and a "
          "rebuild here is a full recomputation of the fixed-base table");
    CHECK(again.published_epoch == first.published_epoch,
          "PNR-1: re-applying an identical FixedBaseConfig does not bump the "
          "publication epoch");

    // ── PNR-4: the surviving context is the right one ────────────────────────
    CHECK(again.published_window_bits == kWindow,
          "PNR-4: the surviving context still describes the configured window");
    {
        auto const p = secp256k1::fast::scalar_mul_generator(
            secp256k1::fast::Scalar::from_uint64(1));
        auto const g = secp256k1::fast::Point::generator();
        bool const is_g = !p.is_infinity() &&
                          p.to_compressed() == g.to_compressed();
        CHECK(is_g, "PNR-4: 1*G through the surviving table is still G");
    }

    // ── PNR-2: NEGATIVE CONTROL -- a real change must still invalidate ───────
    // Without this, PNR-1 would pass against an implementation that never
    // invalidates anything, which would be a correctness bug rather than a fix.
    FixedBaseConfig changed = cfg;
    changed.window_bits = kWindow + 1;
    secp256k1::fast::configure_fixed_base(changed);
    use_the_table();
    auto const rebuilt = secp256k1::fast::precompute_context_diagnostics();

    CHECK(rebuilt.published_epoch != again.published_epoch,
          "PNR-2: changing window_bits DOES invalidate and republish (proves "
          "PNR-1 is not 'never invalidate')");
    CHECK(rebuilt.published_window_bits == kWindow + 1,
          "PNR-2: the republished context uses the new window_bits");

    // ── PNR-3: a cache-only difference also invalidates ──────────────────────
    // build_context() snapshots the entire config into ctx->config, so a
    // context kept across ANY field change would leave g_config and ctx->config
    // describing different things.
    FixedBaseConfig cache_differs = changed;
    cache_differs.max_windows_to_load = changed.max_windows_to_load + 1;
    secp256k1::fast::configure_fixed_base(cache_differs);
    use_the_table();
    auto const after_cache = secp256k1::fast::precompute_context_diagnostics();

    CHECK(after_cache.published_epoch != rebuilt.published_epoch,
          "PNR-3: a config differing only in a non-table field still invalidates "
          "-- the whole config is snapshotted, so it must not be allowed to drift");

    // Leave the process on the shipped default for whatever runs next.
    secp256k1::fast::configure_fixed_base(FixedBaseConfig{});

    std::printf("\n[regression_precompute_noop_reconfigure] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
#endif
}

#ifdef STANDALONE_TEST
int main() { return test_regression_precompute_noop_reconfigure_run(); }
#endif
