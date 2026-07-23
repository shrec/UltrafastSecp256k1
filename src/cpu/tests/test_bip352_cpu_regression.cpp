// GitHub #336: fixed-base context publication, lifetime and correctness tests.

#include "secp256k1/batch_add_affine.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/tagged_hash.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

using namespace secp256k1::fast;

static int g_pass = 0;
static int g_fail = 0;

static void check(bool condition, const char* label) {
    if (condition) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  FAIL: %s\n", label);
    }
}

static bool equal_points(const Point& lhs, const Point& rhs) {
    return lhs.to_compressed() == rhs.to_compressed();
}

// Deliberately clear Point's generator identity. This must execute the general
// variable-base path, never scalar_mul_generator().
static Point oracle_mul_g(const Scalar& scalar) {
    Point const& generator = Point::generator();
    Point plain_generator =
        Point::from_affine(generator.x(), generator.y());
    return plain_generator.scalar_mul(scalar);
}

static FixedBaseConfig test_config(unsigned window_bits, bool enable_glv) {
    FixedBaseConfig config;
    config.window_bits = window_bits;
    config.enable_glv = enable_glv;
    config.use_cache = false;
    config.thread_count = 1;
    return config;
}

static std::vector<Scalar> test_scalars(std::size_t count) {
    std::vector<Scalar> scalars;
    scalars.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        scalars.push_back(Scalar::from_uint64(
            17U + static_cast<std::uint64_t>(i) * 104729U));
    }
    return scalars;
}

static void test_scalar_batch_correctness() {
    std::printf("[GH336] scalar/batch independent-oracle correctness\n");
    // Odd and above the OpenMP threshold when that optional backend is on.
    std::vector<Scalar> const scalars = test_scalars(67);

    for (bool enable_glv : {false, true}) {
        configure_fixed_base(test_config(6, enable_glv));
        ensure_fixed_base_ready();

        bool scalar_ok = true;
        for (Scalar const& scalar : scalars) {
            scalar_ok = scalar_ok &&
                equal_points(scalar_mul_generator(scalar), oracle_mul_g(scalar));
        }
        check(scalar_ok, enable_glv
            ? "GLV scalar results match independent oracle"
            : "non-GLV scalar results match independent oracle");

        std::vector<Point> results(scalars.size());
        batch_scalar_mul_generator(
            scalars.data(), results.data(), results.size());
        bool batch_ok = true;
        for (std::size_t i = 0; i < scalars.size(); ++i) {
            batch_ok = batch_ok &&
                equal_points(results[i], oracle_mul_g(scalars[i]));
        }
        check(batch_ok, enable_glv
            ? "GLV odd-sized batch matches independent oracle"
            : "non-GLV odd-sized batch matches independent oracle");
    }

    Point sentinel = Point::infinity();
    Scalar one = Scalar::from_uint64(1);
    batch_scalar_mul_generator(&one, &sentinel, 0);
    check(sentinel.is_infinity(), "n=0 leaves output untouched");

    Point single;
    batch_scalar_mul_generator(&one, &single, 1);
    check(equal_points(single, oracle_mul_g(one)),
          "n=1 batch matches independent oracle");
}

#if defined(SECP256K1_PRECOMPUTE_TEST_HOOKS)
static void test_acquisition_cardinality() {
    std::printf("[GH336] acquisition cardinality and atomic capability\n");
    configure_fixed_base(test_config(6, false));
    ensure_fixed_base_ready();

    check(fixed_base_context_identity_is_always_lock_free(),
          "raw context identity passes compile-time lock-free gate");
    check(fixed_base_context_identity_is_lock_free(),
          "raw context identity is lock-free on this runtime host");

    precompute_test_reset_acquisition_count();
    Scalar scalar = Scalar::from_uint64(1234567);
    (void)scalar_mul_generator(scalar);
    PrecomputeContextDiagnostics scalar_diag =
        precompute_context_diagnostics();
    check(scalar_diag.acquisition_calls == 1,
          "one scalar call invokes context acquisition exactly once");
    check(scalar_diag.tls_identity == scalar_diag.published_identity,
          "steady scalar TLS owner matches published identity");

    Point single;
    std::uint64_t const before_single =
        precompute_context_diagnostics().acquisition_calls;
    batch_scalar_mul_generator(&scalar, &single, 1);
    std::uint64_t const after_single =
        precompute_context_diagnostics().acquisition_calls;
    check(after_single - before_single == 1,
          "n=1 batch invokes context acquisition exactly once");

    std::vector<Scalar> scalars = test_scalars(67);
    std::vector<Point> results(scalars.size());
    std::uint64_t const before =
        precompute_context_diagnostics().acquisition_calls;
    batch_scalar_mul_generator(
        scalars.data(), results.data(), results.size());
    std::uint64_t const after =
        precompute_context_diagnostics().acquisition_calls;
    std::printf("  acquisition_count: scalar=%llu batch_delta=%llu\n",
                static_cast<unsigned long long>(scalar_diag.acquisition_calls),
                static_cast<unsigned long long>(after - before));
    check(after - before == 1,
          "n>1 batch invokes context acquisition once, not per element");

    Point untouched = Point::infinity();
    std::uint64_t const before_zero =
        precompute_context_diagnostics().acquisition_calls;
    batch_scalar_mul_generator(scalars.data(), &untouched, 0);
    std::uint64_t const after_zero =
        precompute_context_diagnostics().acquisition_calls;
    check(after_zero == before_zero,
          "n=0 batch performs no context acquisition");
    std::printf("  batch acquisition deltas: n=0:%llu n=1:%llu n=67:%llu\n",
                static_cast<unsigned long long>(after_zero - before_zero),
                static_cast<unsigned long long>(
                    after_single - before_single),
                static_cast<unsigned long long>(after - before));
}

static bool wait_for(const std::atomic<bool>& flag,
                     std::chrono::milliseconds timeout) {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!flag.load(std::memory_order_acquire)) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

static bool wait_for_acquire_pause(std::chrono::milliseconds timeout) {
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!precompute_test_acquire_is_paused()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

static void test_deterministic_old_owner_and_tls_refresh() {
    std::printf("[GH336] deterministic null/new publication lifecycle\n");
    configure_fixed_base(test_config(4, false));
    ensure_fixed_base_ready();

    Scalar const scalar = Scalar::from_uint64(999983);
    Point const oracle = oracle_mul_g(scalar);
    Point old_result;
    Point new_result;
    PrecomputeContextDiagnostics old_diag{};
    PrecomputeContextDiagnostics new_diag{};
    std::atomic<bool> first_done{false};
    std::atomic<bool> allow_refresh{false};
    std::atomic<bool> worker_failed{false};

    precompute_test_set_pause_after_acquire(true);
    std::thread reader([&]() {
        try {
            old_result = scalar_mul_generator(scalar);
            old_diag = precompute_context_diagnostics();
            first_done.store(true, std::memory_order_release);
            while (!allow_refresh.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            new_result = scalar_mul_generator(scalar);
            new_diag = precompute_context_diagnostics();
        } catch (...) {
            worker_failed.store(true, std::memory_order_release);
            first_done.store(true, std::memory_order_release);
        }
    });

    bool const paused =
        wait_for_acquire_pause(std::chrono::milliseconds(5000));
    check(paused, "reader paused after acquiring old TLS owner");

    PrecomputeContextDiagnostics published_new{};
    if (paused) {
        // configure publishes null; ensure then publishes a distinct context.
        configure_fixed_base(test_config(5, false));
        ensure_fixed_base_ready();
        published_new = precompute_context_diagnostics();
    }
    precompute_test_set_pause_after_acquire(false);

    bool const completed_first =
        wait_for(first_done, std::chrono::milliseconds(5000));
    check(completed_first, "already-started old-context operation completed");
    if (completed_first && !worker_failed.load(std::memory_order_acquire)) {
        check(equal_points(old_result, oracle),
              "old TLS owner remains correct across null/new publication");
        check(old_diag.tls_window_bits == 4,
              "old operation retained old configuration");
        check(old_diag.tls_identity != 0 &&
                  old_diag.tls_identity != published_new.published_identity,
              "old TLS identity remains owned and differs from new publication");
        check(old_diag.tls_epoch != published_new.published_epoch,
              "diagnostic epoch distinguishes old and new publication");
    }

    allow_refresh.store(true, std::memory_order_release);
    reader.join();
    check(!worker_failed.load(std::memory_order_acquire),
          "lifecycle worker completed without exception");
    check(equal_points(new_result, oracle),
          "next same-thread acquisition remains oracle-correct");
    check(new_diag.tls_identity == published_new.published_identity,
          "same live thread refreshes TLS to new published identity");
    check(new_diag.tls_epoch == published_new.published_epoch,
          "same live thread refreshes diagnostic epoch");
    check(new_diag.tls_window_bits == 5,
          "same live thread observes new configuration");
}
#endif

static void run_readers_against_all_writer_paths(unsigned reader_count) {
    FixedBaseConfig const config = test_config(4, false);
    configure_fixed_base(config);
    ensure_fixed_base_ready();

    std::string const cache_path = "gh336_precompute_context_cache.bin";
    (void)std::remove(cache_path.c_str());
    check(save_precompute_cache(cache_path),
          "public cache-save wrapper succeeds while holding its lock");

    std::array<Scalar, 7> scalars{};
    std::array<Point, 7> oracles{};
    for (std::size_t i = 0; i < scalars.size(); ++i) {
        scalars[i] = Scalar::from_uint64(1009 + i * 7919);
        oracles[i] = oracle_mul_g(scalars[i]);
    }

    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::atomic<std::uint64_t> completed{0};
    std::vector<std::thread> readers;
    readers.reserve(reader_count);
    for (unsigned thread_index = 0; thread_index < reader_count;
         ++thread_index) {
        readers.emplace_back([&, thread_index]() {
            std::array<Point, 7> batch{};
            while (!stop.load(std::memory_order_acquire)) {
                std::size_t const i = thread_index % scalars.size();
                try {
                    if (!equal_points(
                            scalar_mul_generator(scalars[i]), oracles[i])) {
                        failed.store(true, std::memory_order_release);
                        return;
                    }
                    batch_scalar_mul_generator(
                        scalars.data(), batch.data(), batch.size());
                    for (std::size_t j = 0; j < batch.size(); ++j) {
                        if (!equal_points(batch[j], oracles[j])) {
                            failed.store(true, std::memory_order_release);
                            return;
                        }
                    }
                } catch (...) {
                    failed.store(true, std::memory_order_release);
                    return;
                }
                completed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (unsigned iteration = 0; iteration < 16; ++iteration) {
        configure_fixed_base(config);
        ensure_fixed_base_ready();
        set_cache_directory(".");
        configure_fixed_base(config);
        check(load_precompute_cache(cache_path, 0),
              "public cache-load wrapper publishes validated context");
    }

    stop.store(true, std::memory_order_release);
    for (std::thread& reader : readers) reader.join();
    check(!failed.load(std::memory_order_acquire),
          reader_count == 1
              ? "1 reader stays oracle-correct across all writer paths"
              : "18 readers stay oracle-correct across all writer paths");
    check(completed.load(std::memory_order_relaxed) > 0,
          reader_count == 1
              ? "1 reader made lifecycle progress"
              : "18 readers made lifecycle progress");
    (void)std::remove(cache_path.c_str());
}

static void test_concurrent_writer_paths() {
    std::printf("[GH336] configure/cache-directory/cache-load races\n");
    run_readers_against_all_writer_paths(1);
    run_readers_against_all_writer_paths(18);
}

static void test_auto_config_writer_race() {
    std::printf("[GH336] auto-config vs configure/cache-directory writers\n");
    std::atomic<bool> start{false};
    std::atomic<bool> failed{false};

    auto await_start = [&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
    };

    std::thread auto_configurer([&]() {
        await_start();
        for (unsigned i = 0; i < 128; ++i) {
            if (!configure_fixed_base_auto()) {
                failed.store(true, std::memory_order_release);
            }
        }
    });
    std::thread cache_directory_writer([&]() {
        await_start();
        for (unsigned i = 0; i < 128; ++i) {
            set_cache_directory((i & 1U) == 0U ? "." : "");
        }
    });
    std::thread explicit_configurer([&]() {
        await_start();
        for (unsigned i = 0; i < 128; ++i) {
            configure_fixed_base(test_config(4U + (i & 1U), false));
        }
    });

    start.store(true, std::memory_order_release);
    auto_configurer.join();
    cache_directory_writer.join();
    explicit_configurer.join();

    configure_fixed_base(test_config(4, false));
    ensure_fixed_base_ready();
    Scalar const scalar = Scalar::from_uint64(424242);
    check(!failed.load(std::memory_order_acquire),
          "all concurrent auto-config calls succeeded");
    check(equal_points(scalar_mul_generator(scalar), oracle_mul_g(scalar)),
          "post-race explicit configuration remains oracle-correct");
}

static void test_failed_cache_load_preserves_context() {
    std::printf("[GH336] failed cache loads preserve ready context\n");
    FixedBaseConfig const config = test_config(4, false);
    configure_fixed_base(config);
    ensure_fixed_base_ready();

    Scalar const scalar = Scalar::from_uint64(8675309);
    Point const expected = oracle_mul_g(scalar);
#if defined(SECP256K1_PRECOMPUTE_TEST_HOOKS)
    PrecomputeContextDiagnostics const before =
        precompute_context_diagnostics();
#endif

    std::string const missing_path =
        "gh336_missing_precompute_context_cache.bin";
    std::string const malformed_path =
        "gh336_malformed_precompute_context_cache.bin";
    (void)std::remove(missing_path.c_str());
    (void)std::remove(malformed_path.c_str());

    check(!load_precompute_cache(missing_path, 0),
          "missing cache load fails closed");
    check(fixed_base_ready(),
          "missing cache load preserves ready state");
    check(equal_points(scalar_mul_generator(scalar), expected),
          "missing cache load preserves current context result");

    {
        std::ofstream malformed(malformed_path,
                                std::ios::binary | std::ios::trunc);
        malformed << "not-a-precompute-cache";
    }
    check(!load_precompute_cache(malformed_path, 0),
          "malformed cache load fails closed");
    check(fixed_base_ready(),
          "malformed cache load preserves ready state");
    check(equal_points(scalar_mul_generator(scalar), expected),
          "malformed cache load preserves current context result");

#if defined(SECP256K1_PRECOMPUTE_TEST_HOOKS)
    PrecomputeContextDiagnostics const after =
        precompute_context_diagnostics();
    check(after.published_identity == before.published_identity,
          "failed cache loads do not replace published identity");
    check(after.published_epoch == before.published_epoch,
          "failed cache loads do not advance publication epoch");
#endif
    (void)std::remove(malformed_path.c_str());
}

static void test_end_to_end_valid_invalid_candidates() {
    std::printf("[GH336] end-to-end valid/invalid BIP352 candidates\n");
    configure_fixed_base(test_config(6, false));
    ensure_fixed_base_ready();

    Scalar const scan_key = Scalar::from_uint64(0x424242);
    Point spend = oracle_mul_g(Scalar::from_uint64(0x1357));
    spend.normalize();
    Point const input_sum = oracle_mul_g(Scalar::from_uint64(0x99887766));

    KPlan const plan = KPlan::from_scalar(scan_key);
    Point const shared = input_sum.scalar_mul_with_plan(plan);
    std::array<Point, 1> shared_points{shared};
    std::array<std::array<std::uint8_t, 33>, 1> compressed{};
    Point::batch_to_compressed(
        shared_points.data(), shared_points.size(), compressed.data());

    auto const tag_mid =
        secp256k1::detail::make_tag_midstate("BIP0352/SharedSecret");
    std::uint8_t serialization[37]{};
    std::memcpy(serialization, compressed[0].data(), 33);
    auto const hash = secp256k1::detail::cached_tagged_hash(
        tag_mid, serialization, sizeof(serialization));
    Scalar const tweak = Scalar::from_bytes(hash.data());

    Point tweak_point = scalar_mul_generator(tweak);
    tweak_point.normalize();
    AffinePointCompact tweak_affine{tweak_point.x(), tweak_point.y()};
    std::array<FieldElement, 1> candidate_x{};
    std::vector<FieldElement> scratch;
    batch_add_affine_x(spend.x(), spend.y(), &tweak_affine,
                       candidate_x.data(), candidate_x.size(), scratch);

    Point expected = spend.add(oracle_mul_g(tweak));
    expected.normalize();
    check(candidate_x[0] == expected.x(),
          "valid candidate x matches independent point-add oracle");

    Scalar const wrong_scan_key = Scalar::from_uint64(0x424243);
    Point const wrong_shared =
        input_sum.scalar_mul_with_plan(KPlan::from_scalar(wrong_scan_key));
    std::array<Point, 1> wrong_shared_points{wrong_shared};
    std::array<std::array<std::uint8_t, 33>, 1> wrong_compressed{};
    Point::batch_to_compressed(wrong_shared_points.data(),
                               wrong_shared_points.size(),
                               wrong_compressed.data());
    std::uint8_t wrong_serialization[37]{};
    std::memcpy(wrong_serialization, wrong_compressed[0].data(), 33);
    auto const wrong_hash = secp256k1::detail::cached_tagged_hash(
        tag_mid, wrong_serialization, sizeof(wrong_serialization));
    check(std::memcmp(wrong_hash.data(), hash.data(), hash.size()) != 0,
          "wrong scan key changes shared-secret tweak");
    Scalar const wrong_scan_tweak =
        Scalar::from_bytes(wrong_hash.data());
    Point wrong_scan_point = scalar_mul_generator(wrong_scan_tweak);
    wrong_scan_point.normalize();
    AffinePointCompact wrong_scan_affine{
        wrong_scan_point.x(), wrong_scan_point.y()};
    std::array<FieldElement, 1> wrong_scan_x{};
    std::vector<FieldElement> wrong_scan_scratch;
    batch_add_affine_x(spend.x(), spend.y(), &wrong_scan_affine,
                       wrong_scan_x.data(), wrong_scan_x.size(),
                       wrong_scan_scratch);
    check(!(wrong_scan_x[0] == candidate_x[0]),
          "wrong scan key does not match valid candidate x");

    Point wrong_spend = oracle_mul_g(Scalar::from_uint64(0x1358));
    wrong_spend.normalize();
    std::array<FieldElement, 1> wrong_spend_x{};
    std::vector<FieldElement> wrong_spend_scratch;
    batch_add_affine_x(wrong_spend.x(), wrong_spend.y(), &tweak_affine,
                       wrong_spend_x.data(), wrong_spend_x.size(),
                       wrong_spend_scratch);
    check(!(wrong_spend_x[0] == candidate_x[0]),
          "wrong spend key changes candidate x");

    Scalar const corrupted_tweak =
        tweak + Scalar::from_uint64(1);
    Point corrupted_point = scalar_mul_generator(corrupted_tweak);
    corrupted_point.normalize();
    AffinePointCompact corrupted_affine{
        corrupted_point.x(), corrupted_point.y()};
    std::array<FieldElement, 1> corrupted_x{};
    std::vector<FieldElement> corrupted_scratch;
    batch_add_affine_x(spend.x(), spend.y(), &corrupted_affine,
                       corrupted_x.data(), corrupted_x.size(),
                       corrupted_scratch);
    check(!(corrupted_x[0] == candidate_x[0]),
          "corrupted tweak changes candidate x");
}

int test_bip352_cpu_regression_run() {
    std::printf("\n=== BIP-352 CPU regression tests (GitHub #336) ===\n");
    test_scalar_batch_correctness();
#if defined(SECP256K1_PRECOMPUTE_TEST_HOOKS)
    test_acquisition_cardinality();
    test_deterministic_old_owner_and_tls_refresh();
#else
    check(false, "SECP256K1_PRECOMPUTE_TEST_HOOKS must be enabled");
#endif
    test_concurrent_writer_paths();
    test_auto_config_writer_race();
    test_failed_cache_load_preserves_context();
    test_end_to_end_valid_invalid_candidates();

    configure_fixed_base(FixedBaseConfig{});
    std::printf("  result: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_bip352_cpu_regression_run();
}
#endif
