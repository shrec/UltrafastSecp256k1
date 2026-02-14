/**
 * UltrafastSecp256k1 - ESP32 Integration Test
 *
 * Testing real secp256k1 library on ESP32 using the library's Selftest
 */

#include <stdio.h>
#include "esp_chip_info.h"
#include "esp_system.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// Include real secp256k1 library
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/selftest.hpp"

using namespace secp256k1::fast;

// Helper to get chip model name
static const char* get_chip_model_name(esp_chip_model_t model) {
    switch (model) {
        case CHIP_ESP32:   return "ESP32";
        case CHIP_ESP32S2: return "ESP32-S2";
        case CHIP_ESP32S3: return "ESP32-S3";
        case CHIP_ESP32C3: return "ESP32-C3";
        case CHIP_ESP32C2: return "ESP32-C2";
        case CHIP_ESP32C6: return "ESP32-C6";
        case CHIP_ESP32H2: return "ESP32-H2";
        default:           return "Unknown";
    }
}

extern "C" void app_main() {
    vTaskDelay(pdMS_TO_TICKS(1000));

    printf("\n");
    printf("============================================================\n");
    printf("   UltrafastSecp256k1 - ESP32 Library Test\n");
    printf("============================================================\n");
    printf("\n");

    // Platform information
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    printf("Platform Information:\n");
    printf("  Chip Model:   %s\n", get_chip_model_name(chip_info.model));
    printf("  Cores:        %d\n", chip_info.cores);
    printf("  Revision:     %d.%d\n", chip_info.revision / 100, chip_info.revision % 100);
    printf("  Free Heap:    %lu bytes\n", (unsigned long)esp_get_free_heap_size());
    printf("  Build:        32-bit Portable (no __int128)\n");
    printf("\n");

    // Run the real library self-test
    printf("Running SECP256K1 Library Self-Test...\n");
    printf("(This may take a few seconds on ESP32)\n\n");

    // Quick field diagnostics BEFORE selftest
    printf("\n=== Field Arithmetic Diagnostics ===\n");
    {
        // (p-1)^2 should equal 1
        FieldElement pm1 = FieldElement::from_limbs({
            0xFFFFFFFEFFFFFC2EULL, 0xFFFFFFFFFFFFFFFFULL,
            0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
        });
        FieldElement pm1_mul = pm1 * pm1;
        printf("  (p-1)*(p-1)==1?  %s\n", (pm1_mul == FieldElement::one()) ? "PASS" : "FAIL");
        if (pm1_mul != FieldElement::one()) {
            printf("    Got: %s\n", pm1_mul.to_hex().c_str());
        }
        FieldElement pm1_sq = pm1; pm1_sq.square_inplace();
        printf("  (p-1).sq()==1?   %s\n", (pm1_sq == FieldElement::one()) ? "PASS" : "FAIL");
        if (pm1_sq != FieldElement::one()) {
            printf("    Got: %s\n", pm1_sq.to_hex().c_str());
        }
        printf("  sq==mul?         %s\n", (pm1_sq == pm1_mul) ? "PASS" : "FAIL");

        // Random-ish values
        FieldElement x = FieldElement::from_limbs({
            0xA1B2C3D4E5F60718ULL, 0x1928374655647382ULL,
            0xBBAACCDDEEFF0011ULL, 0x2233445566778899ULL
        });
        FieldElement y = FieldElement::from_limbs({
            0xFEDCBA9876543210ULL, 0x0123456789ABCDEFULL,
            0x1122334455667788ULL, 0x99AABBCCDDEEFF00ULL
        });
        FieldElement z = FieldElement::from_limbs({
            0x1111111122222222ULL, 0x3333333344444444ULL,
            0x5555555566666666ULL, 0x7777777788888888ULL
        });
        printf("  x*y==y*x?        %s\n", (x*y == y*x) ? "PASS" : "FAIL");
        printf("  (x*y)*z==x*(y*z)? %s\n", ((x*y)*z == x*(y*z)) ? "PASS" : "FAIL");
        FieldElement xsq = x; xsq.square_inplace();
        FieldElement xmul = x * x;
        printf("  x.sq==x*x?       %s\n", (xsq == xmul) ? "PASS" : "FAIL");
        if (xsq != xmul) {
            printf("    sq:  %s\n", xsq.to_hex().c_str());
            printf("    mul: %s\n", xmul.to_hex().c_str());
        }
        printf("  x*(y+z)==x*y+x*z? %s\n", (x*(y+z) == x*y + x*z) ? "PASS" : "FAIL");
    }
    printf("=== End Diagnostics ===\n\n");

    bool test_passed = Selftest(true);  // verbose = true

    printf("\n");
    if (test_passed) {
        printf("============================================================\n");
        printf("   SUCCESS: All library tests passed on ESP32!\n");
        printf("============================================================\n");
    } else {
        printf("============================================================\n");
        printf("   FAILURE: Some tests failed. Check output above.\n");
        printf("============================================================\n");
    }

    // Simple performance benchmark
    printf("\n");
    printf("==============================================\n");
    printf("  Basic Performance Benchmark\n");
    printf("==============================================\n");

    const int iterations = 1000;
    FieldElement a = FieldElement::from_limbs({0x12345678, 0xABCDEF01, 0x11223344, 0x55667788});
    FieldElement b = FieldElement::from_limbs({0x87654321, 0xFEDCBA98, 0x99AABBCC, 0xDDEEFF00});

    // Field Multiplication
    {
        int64_t start = esp_timer_get_time();
        FieldElement result = a;
        for (int i = 0; i < iterations; i++) {
            result = result * b;
        }
        int64_t elapsed = esp_timer_get_time() - start;
        printf("  Field Mul:    %5lld ns/op\n", (elapsed * 1000) / iterations);
        // Force use of result to prevent optimization
        if (result == FieldElement::zero()) printf("!");
    }

    // Field Squaring
    {
        int64_t start = esp_timer_get_time();
        FieldElement result = a;
        for (int i = 0; i < iterations; i++) {
            result = result.square();
        }
        int64_t elapsed = esp_timer_get_time() - start;
        printf("  Field Square: %5lld ns/op\n", (elapsed * 1000) / iterations);
        if (result == FieldElement::zero()) printf("!");
    }

    // Field Addition
    {
        int64_t start = esp_timer_get_time();
        FieldElement result = a;
        for (int i = 0; i < iterations; i++) {
            result = result + b;
        }
        int64_t elapsed = esp_timer_get_time() - start;
        printf("  Field Add:    %5lld ns/op\n", (elapsed * 1000) / iterations);
        if (result == FieldElement::zero()) printf("!");
    }

    // Field Inversion
    {
        int64_t start = esp_timer_get_time();
        FieldElement result = a;
        for (int i = 0; i < 100; i++) {
            result = result.inverse();
        }
        int64_t elapsed = esp_timer_get_time() - start;
        printf("  Field Inv:    %5lld us/op\n", elapsed / 100);
        if (result == FieldElement::zero()) printf("!");
    }

    // Scalar Multiplication (full 256-bit scalar)
    if (test_passed) {
        printf("\n  Scalar Mul benchmark (full 256-bit scalar):\n");
        Scalar k = Scalar::from_hex("4727daf2986a9804b1117f8261aba645c34537e4474e19be58700792d501a591");
        Point G = Point::generator();

        // Warmup
        volatile uint64_t sink = 0;
        Point warmup = G.scalar_mul(k);
        sink = warmup.x().limbs()[0];

        int64_t start = esp_timer_get_time();
        Point result = G;
        for (int i = 0; i < 5; i++) {
            result = G.scalar_mul(k);
        }
        int64_t elapsed = esp_timer_get_time() - start;
        sink = result.x().limbs()[0];
        printf("  Scalar*G:     %5lld us/op\n", elapsed / 5);
        (void)sink;
    }

    printf("\n");
    printf("============================================================\n");
    printf("   UltrafastSecp256k1 on ESP32 - Test Complete\n");
    printf("============================================================\n");

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}
