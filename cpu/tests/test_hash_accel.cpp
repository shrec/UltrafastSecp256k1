// ============================================================================
// Test: Accelerated Hashing — SHA-256 / RIPEMD-160 / Hash160
// ============================================================================
// Validates correctness against known test vectors (NIST, Bitcoin).
// Benchmarks scalar vs SHA-NI performance.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <array>

#include "secp256k1/hash_accel.hpp"
#include "secp256k1/sha256.hpp"  // reference
#include "secp256k1/point.hpp"

using namespace secp256k1;

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  FAIL: %s\n", name);
    }
}

static void print_hex(const std::uint8_t* data, std::size_t len) {
    for (std::size_t i = 0; i < len; ++i)
        std::printf("%02x", data[i]);
}

// ── Test 1: Feature detection ────────────────────────────────────────────────

static void test_feature_detection() {
    std::printf("[HashAccel] Feature detection...\n");
    auto tier = hash::detect_hash_tier();
    std::printf("  Hash tier: %s\n", hash::hash_tier_name(tier));
    std::printf("  SHA-NI:    %s\n", hash::sha_ni_available() ? "yes" : "no");
    std::printf("  AVX2:      %s\n", hash::avx2_available() ? "yes" : "no");
    std::printf("  AVX-512:   %s\n", hash::avx512_available() ? "yes" : "no");
    check(true, "feature detection completed");
}

// ── Test 2: SHA-256 known vectors ────────────────────────────────────────────

static void test_sha256_vectors() {
    std::printf("[HashAccel] SHA-256 known vectors...\n");

    // NIST vector: SHA256("abc")
    {
        auto h = hash::sha256("abc", 3);
        std::uint8_t expected[32] = {
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea,
            0x41, 0x41, 0x40, 0xde, 0x5d, 0xae, 0x22, 0x23,
            0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c,
            0xb4, 0x10, 0xff, 0x61, 0xf2, 0x00, 0x15, 0xad
        };
        check(std::memcmp(h.data(), expected, 32) == 0, "SHA256(\"abc\") NIST vector");
    }

    // SHA256("") 
    {
        auto h = hash::sha256("", 0);
        std::uint8_t expected[32] = {
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14,
            0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f, 0xb9, 0x24,
            0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c,
            0xa4, 0x95, 0x99, 0x1b, 0x78, 0x52, 0xb8, 0x55
        };
        check(std::memcmp(h.data(), expected, 32) == 0, "SHA256(\"\") empty vector");
    }
}

// ── Test 3: sha256_33 vs reference ───────────────────────────────────────────

static void test_sha256_33_correctness() {
    std::printf("[HashAccel] sha256_33 correctness...\n");

    // Generate a compressed pubkey from 1*G
    auto gen = fast::Point::generator();
    auto compressed = gen.to_compressed();

    // Reference: existing SHA256 class
    auto ref = SHA256::hash(compressed.data(), 33);

    // Accelerated
    std::uint8_t accel[32];
    hash::sha256_33(compressed.data(), accel);

    check(std::memcmp(ref.data(), accel, 32) == 0, "sha256_33(G_compressed) matches reference");

    // Scalar implementation
    std::uint8_t scalar_out[32];
    hash::scalar::sha256_33(compressed.data(), scalar_out);
    check(std::memcmp(ref.data(), scalar_out, 32) == 0, "scalar::sha256_33 matches reference");

    // Test with multiple points
    auto pt = gen;
    for (int i = 0; i < 100; ++i) {
        auto comp = pt.to_compressed();
        auto r = SHA256::hash(comp.data(), 33);
        hash::sha256_33(comp.data(), accel);
        char label[80];
        std::snprintf(label, sizeof(label), "sha256_33(%dG) correct", i + 1);
        check(std::memcmp(r.data(), accel, 32) == 0, label);
        pt.next_inplace();
    }
}

// ── Test 4: sha256_32 correctness ────────────────────────────────────────────

static void test_sha256_32_correctness() {
    std::printf("[HashAccel] sha256_32 correctness...\n");

    std::uint8_t data[32];
    for (int i = 0; i < 32; ++i) data[i] = static_cast<std::uint8_t>(i);

    auto ref = SHA256::hash(data, 32);
    std::uint8_t accel[32];
    hash::sha256_32(data, accel);

    check(std::memcmp(ref.data(), accel, 32) == 0, "sha256_32 matches reference");
}

// ── Test 5: RIPEMD-160 known vectors ─────────────────────────────────────────

static void test_ripemd160_vectors() {
    std::printf("[HashAccel] RIPEMD-160 known vectors...\n");

    // RIPEMD-160("abc") = 8eb208f7e05d987a9b044a8e98c6b087f15a0bfc
    {
        auto h = hash::ripemd160("abc", 3);
        std::uint8_t expected[20] = {
            0x8e, 0xb2, 0x08, 0xf7, 0xe0, 0x5d, 0x98, 0x7a,
            0x9b, 0x04, 0x4a, 0x8e, 0x98, 0xc6, 0xb0, 0x87,
            0xf1, 0x5a, 0x0b, 0xfc
        };
        check(std::memcmp(h.data(), expected, 20) == 0, "RIPEMD160(\"abc\") known vector");
    }

    // RIPEMD-160("") = 9c1185a5c5e9fc54612808977ee8f548b2258d31
    {
        auto h = hash::ripemd160("", 0);
        std::uint8_t expected[20] = {
            0x9c, 0x11, 0x85, 0xa5, 0xc5, 0xe9, 0xfc, 0x54,
            0x61, 0x28, 0x08, 0x97, 0x7e, 0xe8, 0xf5, 0x48,
            0xb2, 0x25, 0x8d, 0x31
        };
        check(std::memcmp(h.data(), expected, 20) == 0, "RIPEMD160(\"\") known vector");
    }

    // RIPEMD-160("message digest") = 5d0689ef49d2fae572b881b123a85ffa21595f36
    {
        const char* msg = "message digest";
        auto h = hash::ripemd160(msg, std::strlen(msg));
        std::uint8_t expected[20] = {
            0x5d, 0x06, 0x89, 0xef, 0x49, 0xd2, 0xfa, 0xe5,
            0x72, 0xb8, 0x81, 0xb1, 0x23, 0xa8, 0x5f, 0xfa,
            0x21, 0x59, 0x5f, 0x36
        };
        check(std::memcmp(h.data(), expected, 20) == 0, "RIPEMD160(\"message digest\") known vector");
    }
}

// ── Test 6: ripemd160_32 correctness ─────────────────────────────────────────

static void test_ripemd160_32_correctness() {
    std::printf("[HashAccel] ripemd160_32 correctness...\n");

    // Hash 32 zero bytes via general API
    std::uint8_t zeros[32] = {};
    auto ref = hash::ripemd160(zeros, 32);

    std::uint8_t fast_out[20];
    hash::ripemd160_32(zeros, fast_out);
    check(std::memcmp(ref.data(), fast_out, 20) == 0, "ripemd160_32(zeros) matches general API");

    // Hash some non-zero data
    std::uint8_t data[32];
    for (int i = 0; i < 32; ++i) data[i] = static_cast<std::uint8_t>(i * 7 + 13);
    auto ref2 = hash::ripemd160(data, 32);

    hash::ripemd160_32(data, fast_out);
    check(std::memcmp(ref2.data(), fast_out, 20) == 0, "ripemd160_32(data) matches general API");
}

// ── Test 7: Hash160 pipeline ─────────────────────────────────────────────────

static void test_hash160_pipeline() {
    std::printf("[HashAccel] Hash160 pipeline correctness...\n");

    // Compute Hash160 of 1*G compressed pubkey via two methods:
    // Method 1: hash::hash160_33
    // Method 2: SHA256 → RIPEMD160 manually

    auto gen = fast::Point::generator();
    auto comp = gen.to_compressed();

    // Manual: SHA256 then RIPEMD160
    auto sha_out = SHA256::hash(comp.data(), 33);
    auto manual_rmd = hash::ripemd160(sha_out.data(), 32);

    // Accelerated hash160_33
    std::uint8_t accel[20];
    hash::hash160_33(comp.data(), accel);

    check(std::memcmp(manual_rmd.data(), accel, 20) == 0, "hash160_33(G) matches manual SHA+RMD");

    // Generic hash160
    auto generic = hash::hash160(comp.data(), 33);
    check(std::memcmp(generic.data(), accel, 20) == 0, "hash160(G) matches hash160_33(G)");

    // Test N points
    auto pt = gen;
    for (int i = 0; i < 50; ++i) {
        auto c = pt.to_compressed();
        auto ref_sha = SHA256::hash(c.data(), 33);
        auto ref_rmd = hash::ripemd160(ref_sha.data(), 32);
        hash::hash160_33(c.data(), accel);
        
        char label[80];
        std::snprintf(label, sizeof(label), "hash160_33(%dG) correct", i + 1);
        check(std::memcmp(ref_rmd.data(), accel, 20) == 0, label);
        pt.next_inplace();
    }
}

// ── Test 8: Batch operations ─────────────────────────────────────────────────

static void test_batch_ops() {
    std::printf("[HashAccel] Batch operations...\n");

    constexpr std::size_t COUNT = 256;

    // Generate pubkeys
    std::array<std::uint8_t, COUNT * 33> pubkeys;
    auto pt = fast::Point::generator();
    for (std::size_t i = 0; i < COUNT; ++i) {
        auto comp = pt.to_compressed();
        std::memcpy(pubkeys.data() + i * 33, comp.data(), 33);
        pt.next_inplace();
    }

    // Batch SHA256
    std::array<std::uint8_t, COUNT * 32> sha_results;
    hash::sha256_33_batch(pubkeys.data(), sha_results.data(), COUNT);

    // Verify each
    for (std::size_t i = 0; i < COUNT; ++i) {
        auto ref = SHA256::hash(pubkeys.data() + i * 33, 33);
        char label[80];
        std::snprintf(label, sizeof(label), "sha256_33_batch[%zu]", i);
        check(std::memcmp(ref.data(), sha_results.data() + i * 32, 32) == 0, label);
    }

    // Batch Hash160
    std::array<std::uint8_t, COUNT * 20> h160_results;
    hash::hash160_33_batch(pubkeys.data(), h160_results.data(), COUNT);

    // Verify each
    pt = fast::Point::generator();
    for (std::size_t i = 0; i < COUNT; ++i) {
        auto comp = pt.to_compressed();
        auto ref_sha = SHA256::hash(comp.data(), 33);
        auto ref_rmd = hash::ripemd160(ref_sha.data(), 32);
        char label[80];
        std::snprintf(label, sizeof(label), "hash160_33_batch[%zu]", i);
        check(std::memcmp(ref_rmd.data(), h160_results.data() + i * 20, 20) == 0, label);
        pt.next_inplace();
    }
}

// ── Test 9: SHA-NI vs Scalar cross-check ─────────────────────────────────────

static void test_shani_vs_scalar() {
    std::printf("[HashAccel] SHA-NI vs Scalar cross-check...\n");

#ifdef SECP256K1_X86_TARGET
    if (!hash::sha_ni_available()) {
        std::printf("  SHA-NI not available, skipping\n");
        check(true, "SHA-NI not available (skip)");
        return;
    }

    auto gen = fast::Point::generator();
    auto pt = gen;

    for (int i = 0; i < 200; ++i) {
        auto comp = pt.to_compressed();

        std::uint8_t scalar_out[32], shani_out[32];
        hash::scalar::sha256_33(comp.data(), scalar_out);
        hash::shani::sha256_33(comp.data(), shani_out);

        char label[80];
        std::snprintf(label, sizeof(label), "SHA-NI == Scalar for %dG", i + 1);
        check(std::memcmp(scalar_out, shani_out, 32) == 0, label);

        pt.next_inplace();
    }
#else
    std::printf("  Not x86, skipping SHA-NI test\n");
    check(true, "Not x86 (skip)");
#endif
}

// ── Test 10: Benchmark ───────────────────────────────────────────────────────

static void test_benchmark() {
    std::printf("[HashAccel] Benchmark...\n");

    // Prepare a typical compressed pubkey
    auto gen = fast::Point::generator();
    auto comp = gen.to_compressed();
    std::uint8_t out32[32], out20[20];

    constexpr int ITERS = 100000;

    // Scalar SHA256_33
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            hash::scalar::sha256_33(comp.data(), out32);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        std::printf("  Scalar SHA256_33:  %.1f ns/call\n", ns / ITERS);
    }

    // Auto-dispatch SHA256_33
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            hash::sha256_33(comp.data(), out32);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        std::printf("  Auto   SHA256_33:  %.1f ns/call (%s)\n", ns / ITERS, hash::hash_tier_name(hash::detect_hash_tier()));
    }

    // Scalar RIPEMD160_32
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            hash::scalar::ripemd160_32(out32, out20);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        std::printf("  Scalar RIPEMD160_32: %.1f ns/call\n", ns / ITERS);
    }

    // Hash160_33 (fused SHA256+RIPEMD160)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            hash::hash160_33(comp.data(), out20);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        std::printf("  Auto   Hash160_33: %.1f ns/call\n", ns / ITERS);
    }

    // Old SHA256 class (reference baseline)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; ++i) {
            auto h = SHA256::hash(comp.data(), 33);
            (void)h;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        std::printf("  Old    SHA256::hash: %.1f ns/call (reference)\n", ns / ITERS);
    }

    // Batch Hash160
    {
        constexpr std::size_t BATCH = 1024;
        std::array<std::uint8_t, BATCH * 33> keys;
        std::array<std::uint8_t, BATCH * 20> hashes;
        auto pt = gen;
        for (std::size_t i = 0; i < BATCH; ++i) {
            auto c = pt.to_compressed();
            std::memcpy(keys.data() + i * 33, c.data(), 33);
            pt.next_inplace();
        }

        // Warmup
        hash::hash160_33_batch(keys.data(), hashes.data(), BATCH);

        constexpr int BATCH_ITERS = 1000;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < BATCH_ITERS; ++i) {
            hash::hash160_33_batch(keys.data(), hashes.data(), BATCH);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_ns = static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        double per_key = total_ns / BATCH_ITERS / BATCH;
        std::printf("  Batch  Hash160_33 (%zu): %.1f ns/key, %.2f Mkeys/s\n",
                    BATCH, per_key, 1e9 / per_key / 1e6);
    }

    check(true, "benchmark complete");
}

// ── Test 11: Double-SHA256 ───────────────────────────────────────────────────

static void test_double_sha256() {
    std::printf("[HashAccel] Double-SHA256...\n");

    std::uint8_t data[] = {0x01, 0x02, 0x03};

    auto ref1 = SHA256::hash(data, 3);
    auto ref2 = SHA256::hash(ref1.data(), 32);

    auto accel = hash::sha256d(data, 3);

    check(std::memcmp(ref2.data(), accel.data(), 32) == 0, "sha256d matches SHA256(SHA256(data))");
}

// ── Entry point ──────────────────────────────────────────────────────────────

int test_hash_accel_run() {
    std::printf("\n=== Accelerated Hashing Tests ===\n");

    test_feature_detection();
    test_sha256_vectors();
    test_sha256_33_correctness();
    test_sha256_32_correctness();
    test_ripemd160_vectors();
    test_ripemd160_32_correctness();
    test_hash160_pipeline();
    test_double_sha256();
    test_batch_ops();
    test_shani_vs_scalar();
    test_benchmark();

    std::printf("\n  Hash accel: %d passed, %d failed\n", g_pass, g_fail);
    return g_fail;
}

#ifdef STANDALONE_TEST
int main() {
    return test_hash_accel_run();
}
#endif
