// ============================================================================
// Test: SIMD Field Operations + Batch Inverse
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <array>
#include <vector>

#include "secp256k1/field_simd.hpp"
#include "secp256k1/field.hpp"

using namespace secp256k1;
using fast::FieldElement;

static int g_pass = 0, g_fail = 0;

static void check(bool cond, const char* name) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  FAIL: %s\n", name);
    }
}

static void test_simd_detection() {
    std::printf("[SIMD] Runtime detection...\n");

    auto tier = simd::detect_simd_tier();
    std::printf("  Detected: %s\n", simd::simd_tier_name(tier));

    // These should not crash regardless of platform
    bool avx2 = simd::avx2_available();
    bool avx512 = simd::avx512_available();
    std::printf("  AVX2: %s, AVX-512: %s\n",
        avx2 ? "yes" : "no", avx512 ? "yes" : "no");

    check(true, "SIMD detection: no crash");
}

static void test_batch_add() {
    std::printf("[SIMD] Batch field add...\n");

    constexpr int N = 16;
    FieldElement a[N], b[N], out[N], expected[N];

    for (int i = 0; i < N; ++i) {
        a[i] = FieldElement::from_uint64(100 + i);
        b[i] = FieldElement::from_uint64(200 + i);
        expected[i] = a[i] + b[i];
    }

    simd::batch_field_add(out, a, b, N);

    bool all_ok = true;
    for (int i = 0; i < N; ++i) {
        if (out[i].to_bytes() != expected[i].to_bytes()) {
            all_ok = false;
            break;
        }
    }
    check(all_ok, "Batch add: matches scalar results");
}

static void test_batch_sub() {
    std::printf("[SIMD] Batch field sub...\n");

    constexpr int N = 16;
    FieldElement a[N], b[N], out[N], expected[N];

    for (int i = 0; i < N; ++i) {
        a[i] = FieldElement::from_uint64(1000 + i);
        b[i] = FieldElement::from_uint64(500 + i);
        expected[i] = a[i] - b[i];
    }

    simd::batch_field_sub(out, a, b, N);

    bool all_ok = true;
    for (int i = 0; i < N; ++i) {
        if (out[i].to_bytes() != expected[i].to_bytes()) {
            all_ok = false;
            break;
        }
    }
    check(all_ok, "Batch sub: matches scalar results");
}

static void test_batch_mul() {
    std::printf("[SIMD] Batch field mul...\n");

    constexpr int N = 8;
    FieldElement a[N], b[N], out[N], expected[N];

    for (int i = 0; i < N; ++i) {
        a[i] = FieldElement::from_uint64(7 + i * 3);
        b[i] = FieldElement::from_uint64(11 + i * 5);
        expected[i] = a[i] * b[i];
    }

    simd::batch_field_mul(out, a, b, N);

    bool all_ok = true;
    for (int i = 0; i < N; ++i) {
        if (out[i].to_bytes() != expected[i].to_bytes()) {
            all_ok = false;
            break;
        }
    }
    check(all_ok, "Batch mul: matches scalar results");
}

static void test_batch_sqr() {
    std::printf("[SIMD] Batch field square...\n");

    constexpr int N = 8;
    FieldElement a[N], out[N], expected[N];

    for (int i = 0; i < N; ++i) {
        a[i] = FieldElement::from_uint64(13 + i * 7);
        expected[i] = a[i].square();
    }

    simd::batch_field_sqr(out, a, N);

    bool all_ok = true;
    for (int i = 0; i < N; ++i) {
        if (out[i].to_bytes() != expected[i].to_bytes()) {
            all_ok = false;
            break;
        }
    }
    check(all_ok, "Batch sqr: matches scalar results");
}

static void test_batch_inv() {
    std::printf("[SIMD] Batch field inverse (Montgomery's trick)...\n");

    constexpr int N = 16;
    FieldElement a[N], out[N];

    for (int i = 0; i < N; ++i) {
        a[i] = FieldElement::from_uint64(3 + i * 2);
    }

    simd::batch_field_inv(out, a, N);

    // Verify: a[i] * out[i] == 1
    bool all_ok = true;
    auto one = FieldElement::one();
    for (int i = 0; i < N; ++i) {
        auto product = a[i] * out[i];
        if (product.to_bytes() != one.to_bytes()) {
            all_ok = false;
            std::printf("  Failed at index %d\n", i);
            break;
        }
    }
    check(all_ok, "Batch inv: a[i] * inv(a[i]) == 1 for all i");
}

static void test_batch_inv_single() {
    std::printf("[SIMD] Batch inverse: single element...\n");

    FieldElement a = FieldElement::from_uint64(42);
    FieldElement out;

    simd::batch_field_inv(&out, &a, 1);

    auto product = a * out;
    check(product.to_bytes() == FieldElement::one().to_bytes(),
          "Batch inv single: a * inv(a) == 1");
}

static void test_batch_inv_with_scratch() {
    std::printf("[SIMD] Batch inverse with explicit scratch...\n");

    constexpr int N = 8;
    FieldElement a[N], out[N], scratch[N];

    for (int i = 0; i < N; ++i) {
        a[i] = FieldElement::from_uint64(17 + i * 11);
    }

    simd::batch_field_inv(out, a, N, scratch);

    bool all_ok = true;
    auto one = FieldElement::one();
    for (int i = 0; i < N; ++i) {
        auto product = a[i] * out[i];
        if (product.to_bytes() != one.to_bytes()) {
            all_ok = false;
            break;
        }
    }
    check(all_ok, "Batch inv with scratch: verified");
}

int main() {
    std::printf("═══════════════════════════════════════════════════════════════\n");
    std::printf("  UltrafastSecp256k1 — SIMD + Batch Field Tests\n");
    std::printf("═══════════════════════════════════════════════════════════════\n\n");

    test_simd_detection();
    test_batch_add();
    test_batch_sub();
    test_batch_mul();
    test_batch_sqr();
    test_batch_inv();
    test_batch_inv_single();
    test_batch_inv_with_scratch();

    std::printf("\n═══════════════════════════════════════════════════════════════\n");
    std::printf("  Results: %d passed, %d failed (total %d)\n",
                g_pass, g_fail, g_pass + g_fail);
    std::printf("═══════════════════════════════════════════════════════════════\n");

    return g_fail > 0 ? 1 : 0;
}
