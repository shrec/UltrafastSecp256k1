// GitHub #336 / PRECOMPUTE-GCONTEXT-UAF structural and concurrency regression.

#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/scalar.hpp"

#include "audit_check.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using secp256k1::fast::FixedBaseConfig;
using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

static int g_pass = 0;
static int g_fail = 0;

static std::size_t count_occurrences(const std::string& text,
                                     const std::string& needle) {
    std::size_t count = 0;
    std::size_t position = 0;
    while ((position = text.find(needle, position)) != std::string::npos) {
        ++count;
        position += needle.size();
    }
    return count;
}

static Point oracle_mul_g(const Scalar& scalar) {
    Point const& generator = Point::generator();
    Point plain_generator =
        Point::from_affine(generator.x(), generator.y());
    return plain_generator.scalar_mul(scalar);
}

static bool equal_points(const Point& lhs, const Point& rhs) {
    return lhs.to_compressed() == rhs.to_compressed();
}

static FixedBaseConfig small_config() {
    FixedBaseConfig config;
    config.window_bits = 4;
    config.enable_glv = false;
    config.use_cache = false;
    config.thread_count = 1;
    return config;
}

static void test_publication_protocol_source_scan() {
    std::printf("[1] raw identity + TLS lifetime-owner protocol\n");
    std::string const source =
        audit_read_source_file("src/cpu/src/precompute.cpp");
    CHECK(!source.empty(), "precompute.cpp is readable through canonical source helper");
    if (source.empty()) return;

    CHECK(source.find(
              "std::shared_ptr<PrecomputeContext> g_context_owner") !=
              std::string::npos,
          "mutex-owned shared_ptr is present");
    CHECK(source.find(
              "std::atomic<PrecomputeContext const*> g_published_context") !=
              std::string::npos,
          "raw published identity atomic is present");
    CHECK(source.find(
              "thread_local std::shared_ptr<PrecomputeContext const> "
              "tl_context_owner") != std::string::npos,
          "TLS shared lifetime owner is present");
    CHECK(source.find("std::atomic<std::shared_ptr") == std::string::npos,
          "atomic<shared_ptr> is forbidden");
    CHECK(source.find("std::atomic_load_explicit(&g_context") ==
              std::string::npos,
          "deprecated free-function shared_ptr atomic load is absent");
    CHECK(source.find("std::atomic_store_explicit(&g_context") ==
              std::string::npos,
          "deprecated free-function shared_ptr atomic store is absent");
    CHECK(source.find("void publish_context_locked(") != std::string::npos &&
              source.find("void invalidate_context_locked(") !=
                  std::string::npos,
          "publication and invalidation are centralized");
    CHECK(source.find("g_published_context.store(nullptr") <
              source.find("g_context_owner.reset()"),
          "invalidation publishes nullptr before releasing global owner");
    CHECK(source.find("scalar_mul_generator_with_context(") !=
              std::string::npos,
          "context-taking fixed-base helper exists");

    std::size_t const acquire_sites =
        count_occurrences(source, "acquire_context_for_current_thread();");
    CHECK(acquire_sites == 3,
          "scalar, predecomposed GLV and batch each have one acquisition site");
}

static void test_concurrent_lifetime() {
    std::printf("[2] readers remain oracle-correct across reset/publication\n");
    FixedBaseConfig const config = small_config();
    secp256k1::fast::configure_fixed_base(config);
    secp256k1::fast::ensure_fixed_base_ready();

    constexpr std::size_t kCount = 5;
    std::array<Scalar, kCount> scalars{};
    std::array<Point, kCount> expected{};
    for (std::size_t i = 0; i < kCount; ++i) {
        scalars[i] = Scalar::from_uint64(73 + i * 65537);
        expected[i] = oracle_mul_g(scalars[i]);
    }

    std::atomic<bool> stop{false};
    std::atomic<bool> failed{false};
    std::vector<std::thread> readers;
    for (unsigned thread_index = 0; thread_index < 18; ++thread_index) {
        readers.emplace_back([&, thread_index]() {
            std::array<Point, kCount> batch{};
            while (!stop.load(std::memory_order_acquire)) {
                std::size_t const i = thread_index % kCount;
                try {
                    if (!equal_points(
                            secp256k1::fast::scalar_mul_generator(scalars[i]),
                            expected[i])) {
                        failed.store(true, std::memory_order_release);
                        return;
                    }
                    secp256k1::fast::batch_scalar_mul_generator(
                        scalars.data(), batch.data(), batch.size());
                    for (std::size_t j = 0; j < kCount; ++j) {
                        if (!equal_points(batch[j], expected[j])) {
                            failed.store(true, std::memory_order_release);
                            return;
                        }
                    }
                } catch (...) {
                    failed.store(true, std::memory_order_release);
                    return;
                }
            }
        });
    }

    for (unsigned iteration = 0; iteration < 40; ++iteration) {
        secp256k1::fast::configure_fixed_base(config);
        secp256k1::fast::ensure_fixed_base_ready();
    }
    stop.store(true, std::memory_order_release);
    for (std::thread& reader : readers) reader.join();

    CHECK(!failed.load(std::memory_order_acquire),
          "18 scalar/batch readers match independent oracle during resets");

#if defined(SECP256K1_PRECOMPUTE_TEST_HOOKS)
    secp256k1::fast::precompute_test_reset_acquisition_count();
    std::array<Point, kCount> batch{};
    secp256k1::fast::batch_scalar_mul_generator(
        scalars.data(), batch.data(), batch.size());
    auto const diagnostics =
        secp256k1::fast::precompute_context_diagnostics();
    std::printf("  acquisition_count for batch=%llu\n",
                static_cast<unsigned long long>(
                    diagnostics.acquisition_calls));
    CHECK(diagnostics.acquisition_calls == 1,
          "batch acquisition cardinality is one per call");
    CHECK(diagnostics.tls_identity == diagnostics.published_identity,
          "TLS lifetime owner matches current publication");
#endif

    secp256k1::fast::configure_fixed_base(FixedBaseConfig{});
}

int test_regression_precompute_gcontext_race_run() {
    g_pass = 0;
    g_fail = 0;
    std::printf("============================================================\n");
    std::printf("  Regression: issue #336 context publication/lifetime\n");
    std::printf("============================================================\n");
    test_publication_protocol_source_scan();
    test_concurrent_lifetime();
    std::printf("[regression_precompute_gcontext_race] %d/%d checks passed\n",
                g_pass, g_pass + g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_precompute_gcontext_race_run();
}
#endif
