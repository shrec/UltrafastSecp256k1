// ============================================================================
// REGRESSION: ECDSA multi-threaded batch verification (ecdsa_batch_verify_mt)
//
// Covers the first-class engine parallelism added so that a large ECDSA batch
// is verified across CPU threads inside the engine (not by the caller/bridge).
// The boolean result MUST be identical to the single-threaded ecdsa_batch_verify
// for ANY thread count, because verification is variable-time over public data.
//
// Asserts:
//   * MT == serial on a fully valid batch, for thread counts {0,1,2,4,8,64}.
//   * A single corrupted signature is detected at every thread count.
//   * Multi-chunk (> internal kChunk=4096): corruption in a LATER chunk still
//     propagates to the overall result (atomic any-invalid flag works across
//     the dynamic work queue), not just in the first chunk.
//   * No arbitrary thread cap: a thread budget above the old 64 cap is accepted
//     and matches serial; batch_verify.cpp uses a dynamic std::vector<std::thread>
//     pool with no hard-coded kMaxThreads.
//   * Edge cases: n==0 -> false (serial contract); n==1 and small-n match serial.
// ============================================================================

#ifndef UNIFIED_AUDIT_RUNNER
#ifndef STANDALONE_TEST
#define STANDALONE_TEST
#endif
#endif

#include "secp256k1/batch_verify.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/sha256.hpp"

#include <cstdio>
#include <cstdint>
#include <string>
#include <fstream>
#include <sstream>
#include <vector>

using namespace secp256k1;
using fast::Scalar;
using fast::Point;

// make_batch() populates inputs via the legacy variable-time ecdsa_sign entry
// point; suppress its deprecation warning so -Werror audit builds succeed.
#if defined(__GNUC__) || defined(__clang__)
#  pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

static int g_pass = 0, g_fail = 0;
#define CHECK(cond, msg) do { \
    if (cond) { ++g_pass; } \
    else { ++g_fail; std::printf("  FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg); } \
} while(0)

static std::vector<ECDSABatchEntry> make_batch(std::size_t N) {
    std::vector<ECDSABatchEntry> e(N);
    auto G = Point::generator();
    for (std::size_t i = 0; i < N; ++i) {
        Scalar const k = Scalar::from_uint64(1000 + i);
        e[i].public_key = G.scalar_mul(k);
        std::uint8_t buf[8];
        for (int b = 0; b < 8; ++b)
            buf[b] = static_cast<std::uint8_t>((i >> (b * 8)) & 0xff);
        e[i].msg_hash = SHA256::hash(buf, 8);
        e[i].signature = ecdsa_sign(e[i].msg_hash, k);
    }
    return e;
}

// MT boolean result must equal serial on a fully valid batch, every thread count.
static void test_mt_matches_serial_valid() {
    std::printf("  [mt-valid] mt(threads) == serial on a valid 200-sig batch\n");
    constexpr std::size_t N = 200;
    auto e = make_batch(N);
    const bool serial = ecdsa_batch_verify(e.data(), N);
    CHECK(serial, "serial valid batch -> true");
    for (std::size_t mt : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                           std::size_t{4}, std::size_t{8}, std::size_t{64}}) {
        CHECK(ecdsa_batch_verify_mt(e.data(), N, mt) == serial,
              "mt(threads=k) == serial on valid batch");
    }
    // vector overload parity
    CHECK(ecdsa_batch_verify_mt(e, 4) == serial, "vector overload == serial");
}

// A single corrupted signature must be detected regardless of thread count.
static void test_mt_detects_corruption() {
    std::printf("  [mt-corrupt] single corrupted sig detected at every thread count\n");
    constexpr std::size_t N = 200;
    auto e = make_batch(N);
    e[123].signature.s = e[123].signature.s + Scalar::one();
    CHECK(!ecdsa_batch_verify(e.data(), N), "serial detects corruption");
    for (std::size_t mt : {std::size_t{0}, std::size_t{1}, std::size_t{2},
                           std::size_t{4}, std::size_t{8}}) {
        CHECK(!ecdsa_batch_verify_mt(e.data(), N, mt),
              "mt detects single corrupted sig");
    }
}

// > kChunk (4096): corruption in a non-first chunk must propagate.
static void test_mt_multichunk() {
    std::printf("  [mt-multichunk] 9000-sig (3 chunks): later-chunk corruption propagates\n");
    constexpr std::size_t N = 9000;  // spans 3 internal 4096-row chunks
    auto e = make_batch(N);
    CHECK(ecdsa_batch_verify_mt(e.data(), N, 4),  "9000 valid (multi-chunk) -> true");
    CHECK(ecdsa_batch_verify_mt(e.data(), N, 4) == ecdsa_batch_verify(e.data(), N),
          "multi-chunk mt == serial (valid)");

    auto e2 = e;  // corrupt 2nd chunk
    e2[5000].signature.s = e2[5000].signature.s + Scalar::one();
    CHECK(!ecdsa_batch_verify_mt(e2.data(), N, 4), "corruption in 2nd chunk detected");

    auto e3 = e;  // corrupt 3rd (last) chunk
    e3[8500].signature.s = e3[8500].signature.s + Scalar::one();
    CHECK(!ecdsa_batch_verify_mt(e3.data(), N, 8), "corruption in 3rd chunk detected");
}

// Read a repo source file by trying the common build-tree-relative paths.
static std::string read_repo_source(const char* rel) {
    static const char* prefixes[] = {
        "", "../", "../../", "../../../", nullptr };
    for (int i = 0; prefixes[i]; ++i) {
        std::ifstream f(std::string(prefixes[i]) + rel);
        if (f) {
            std::ostringstream ss; ss << f.rdbuf();
            return ss.str();
        }
    }
    return "";
}

// The arbitrary 64-thread cap was removed: the worker count is the caller's to
// own, reduced only to hardware_concurrency and the chunk count. Two guards:
//   (1) functional — a thread budget above the old cap (128/256) is accepted and
//       returns the same boolean as serial (engine reduces it to what it can run);
//   (2) source-scan — batch_verify.cpp no longer hard-codes kMaxThreads=64 nor a
//       fixed std::array<std::thread,64> pool; it uses a dynamic std::vector pool.
static void test_mt_no_thread_cap() {
    std::printf("  [mt-no-cap] thread budget above old 64 cap accepted; cap removed in source\n");
    constexpr std::size_t N = 200;
    auto e = make_batch(N);
    const bool serial = ecdsa_batch_verify(e.data(), N);
    for (std::size_t mt : {std::size_t{65}, std::size_t{128}, std::size_t{256},
                           std::size_t{1024}}) {
        CHECK(ecdsa_batch_verify_mt(e.data(), N, mt) == serial,
              "mt(threads>64) accepted and == serial");
    }

    const std::string src = read_repo_source("src/cpu/src/batch_verify.cpp");
    if (src.empty()) {
        std::printf("    [skip] batch_verify.cpp not found from cwd — source scan skipped\n");
        return;
    }
    CHECK(src.find("kMaxThreads") == std::string::npos,
          "batch_verify.cpp: no hard-coded kMaxThreads cap");
    CHECK(src.find("std::array<std::thread") == std::string::npos,
          "batch_verify.cpp: no fixed std::array<std::thread,N> pool");
    // The _mt paths now run on a PERSISTENT worker pool (detail::batch_worker_pool),
    // not a per-call std::thread spawn — this parallelizes block-sized batches AND keeps
    // worker thread_locals warm. (The old design spawned a std::vector<std::thread> per call.)
    CHECK(src.find("batch_worker_pool") != std::string::npos,
          "batch_verify.cpp: uses the persistent worker pool (no per-call thread spawn)");
}

// Edge cases.
static void test_mt_edges() {
    std::printf("  [mt-edges] n==0 / n==1 / small-n parity with serial\n");
    CHECK(ecdsa_batch_verify_mt(nullptr, 0, 0) == false,
          "n==0 -> false (matches serial contract)");
    auto e1 = make_batch(1);
    CHECK(ecdsa_batch_verify_mt(e1.data(), 1, 8) == ecdsa_batch_verify(e1.data(), 1),
          "n==1 mt == serial");
    auto e9 = make_batch(9);  // just over the individual-verify cutoff (8)
    CHECK(ecdsa_batch_verify_mt(e9.data(), 9, 4) == ecdsa_batch_verify(e9.data(), 9),
          "n==9 mt == serial");
}

int test_regression_ecdsa_batch_verify_mt_run() {
    std::printf("[ecdsa-batch-mt] ecdsa_batch_verify_mt parity + threading\n");
    test_mt_matches_serial_valid();
    test_mt_detects_corruption();
    test_mt_multichunk();
    test_mt_no_thread_cap();
    test_mt_edges();
    std::printf("    ecdsa_batch_verify_mt: %d passed, %d failed\n\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() { return test_regression_ecdsa_batch_verify_mt_run(); }
#endif
