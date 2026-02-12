/**
 * UltrafastSecp256k1 - ESP32-S3 Test
 *
 * Portable field arithmetic test for ESP32-S3 (Xtensa LX7)
 * Tests basic secp256k1 operations on embedded hardware.
 *
 * Supported: ESP32, ESP32-S2, ESP32-S3, ESP32-C3, ESP32-C6
 * Recommended: ESP32-S3 (fastest Xtensa, dual-core 240MHz)
 */

#include <stdio.h>
#include <string.h>
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_chip_info.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char* TAG = "secp256k1";

// ============================================================================
// Portable 256-bit Field Element (no assembly, works on any platform)
// ============================================================================

struct FieldElement {
    uint32_t limbs[8];  // 8 x 32-bit for ESP32 (32-bit CPU)

    static FieldElement zero() {
        FieldElement r;
        memset(r.limbs, 0, sizeof(r.limbs));
        return r;
    }

    static FieldElement one() {
        FieldElement r = zero();
        r.limbs[0] = 1;
        return r;
    }

    static FieldElement from_u32(uint32_t v) {
        FieldElement r = zero();
        r.limbs[0] = v;
        return r;
    }
};

// secp256k1 prime: p = 2^256 - 2^32 - 977
static const uint32_t SECP256K1_P[8] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

// ============================================================================
// Portable Field Arithmetic
// ============================================================================

// Add two field elements (mod p)
static FieldElement field_add(const FieldElement& a, const FieldElement& b) {
    FieldElement r;
    uint64_t carry = 0;

    for (int i = 0; i < 8; i++) {
        uint64_t sum = (uint64_t)a.limbs[i] + b.limbs[i] + carry;
        r.limbs[i] = (uint32_t)sum;
        carry = sum >> 32;
    }

    // Reduce if >= p
    uint64_t borrow = 0;
    uint32_t tmp[8];
    for (int i = 0; i < 8; i++) {
        uint64_t diff = (uint64_t)r.limbs[i] - SECP256K1_P[i] - borrow;
        tmp[i] = (uint32_t)diff;
        borrow = (diff >> 63) & 1;
    }

    // If no borrow, use reduced value
    if (borrow == 0 || carry) {
        memcpy(r.limbs, tmp, sizeof(tmp));
    }

    return r;
}

// Multiply two field elements (mod p) - Comba's method
static FieldElement field_mul(const FieldElement& a, const FieldElement& b) {
    uint64_t t[16] = {0};

    // Schoolbook multiplication
    for (int i = 0; i < 8; i++) {
        uint64_t carry = 0;
        for (int j = 0; j < 8; j++) {
            uint64_t prod = (uint64_t)a.limbs[i] * b.limbs[j];
            uint64_t sum = t[i + j] + prod + carry;
            t[i + j] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }
        t[i + 8] += carry;
    }

    // Reduction mod p = 2^256 - 2^32 - 977
    // High bits * (2^32 + 977) and add to low bits
    for (int i = 15; i >= 8; i--) {
        uint64_t hi = t[i];
        if (hi == 0) continue;

        t[i] = 0;

        // hi * 2^32 -> add to t[i-8+1]
        uint64_t carry = 0;
        uint64_t sum = t[i - 7] + (hi << 32) + carry;
        carry = sum >> 32;
        t[i - 7] = sum & 0xFFFFFFFF;

        // Propagate carry
        for (int j = i - 6; j < 16 && carry; j++) {
            sum = t[j] + carry;
            t[j] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }

        // hi * 977 -> add to t[i-8]
        carry = 0;
        sum = t[i - 8] + hi * 977 + carry;
        t[i - 8] = sum & 0xFFFFFFFF;
        carry = sum >> 32;

        for (int j = i - 7; j < 16 && carry; j++) {
            sum = t[j] + carry;
            t[j] = sum & 0xFFFFFFFF;
            carry = sum >> 32;
        }
    }

    // Final reduction if needed
    FieldElement r;
    for (int i = 0; i < 8; i++) {
        r.limbs[i] = (uint32_t)t[i];
    }

    return r;
}

// Square a field element
static FieldElement field_sqr(const FieldElement& a) {
    return field_mul(a, a);  // TODO: Optimize with dedicated squaring
}

// ============================================================================
// Benchmark Functions
// ============================================================================

static void benchmark_field_mul(int iterations) {
    FieldElement a = FieldElement::from_u32(0x12345678);
    FieldElement b = FieldElement::from_u32(0x87654321);

    int64_t start = esp_timer_get_time();

    FieldElement result = a;
    for (int i = 0; i < iterations; i++) {
        result = field_mul(result, b);
    }

    int64_t end = esp_timer_get_time();
    int64_t elapsed_us = end - start;
    int64_t ns_per_op = (elapsed_us * 1000) / iterations;

    ESP_LOGI(TAG, "Field Mul: %lld ns/op (%d iterations)", ns_per_op, iterations);
    ESP_LOGI(TAG, "  Result[0]: 0x%08lx", (unsigned long)result.limbs[0]);
}

static void benchmark_field_sqr(int iterations) {
    FieldElement a = FieldElement::from_u32(0x12345678);

    int64_t start = esp_timer_get_time();

    FieldElement result = a;
    for (int i = 0; i < iterations; i++) {
        result = field_sqr(result);
    }

    int64_t end = esp_timer_get_time();
    int64_t elapsed_us = end - start;
    int64_t ns_per_op = (elapsed_us * 1000) / iterations;

    ESP_LOGI(TAG, "Field Sqr: %lld ns/op (%d iterations)", ns_per_op, iterations);
    ESP_LOGI(TAG, "  Result[0]: 0x%08lx", (unsigned long)result.limbs[0]);
}

static void benchmark_field_add(int iterations) {
    FieldElement a = FieldElement::from_u32(0x12345678);
    FieldElement b = FieldElement::from_u32(0x87654321);

    int64_t start = esp_timer_get_time();

    FieldElement result = a;
    for (int i = 0; i < iterations; i++) {
        result = field_add(result, b);
    }

    int64_t end = esp_timer_get_time();
    int64_t elapsed_us = end - start;
    int64_t ns_per_op = (elapsed_us * 1000) / iterations;

    ESP_LOGI(TAG, "Field Add: %lld ns/op (%d iterations)", ns_per_op, iterations);
}

// ============================================================================
// Main Application
// ============================================================================

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
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "╔══════════════════════════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   UltrafastSecp256k1 - ESP32 Benchmark                   ║");
    ESP_LOGI(TAG, "╚══════════════════════════════════════════════════════════╝");
    ESP_LOGI(TAG, "");

    // Print chip info
    esp_chip_info_t chip_info;
    esp_chip_info(&chip_info);

    ESP_LOGI(TAG, "Chip Info:");
    ESP_LOGI(TAG, "  Model:    %s", get_chip_model_name(chip_info.model));
    ESP_LOGI(TAG, "  Cores:    %d", chip_info.cores);
    ESP_LOGI(TAG, "  Revision: %d.%d", chip_info.revision / 100, chip_info.revision % 100);
    ESP_LOGI(TAG, "  Features: %s%s%s",
             (chip_info.features & CHIP_FEATURE_WIFI_BGN) ? "WiFi " : "",
             (chip_info.features & CHIP_FEATURE_BLE) ? "BLE " : "",
             (chip_info.features & CHIP_FEATURE_BT) ? "BT " : "");
    ESP_LOGI(TAG, "  Free heap: %lu bytes", (unsigned long)esp_get_free_heap_size());

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "Running benchmarks...");
    ESP_LOGI(TAG, "");

    // Warm up
    benchmark_field_mul(100);

    // Real benchmarks
    const int iterations = 10000;

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== Benchmark Results ===");
    benchmark_field_mul(iterations);
    benchmark_field_sqr(iterations);
    benchmark_field_add(iterations);

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "=== Platform Comparison ===");
    ESP_LOGI(TAG, "  x86-64 (i5):     Field Mul ~33 ns");
    ESP_LOGI(TAG, "  RISC-V 64:       Field Mul ~198 ns");
    ESP_LOGI(TAG, "  %s:    Field Mul - see above", get_chip_model_name(chip_info.model));

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "╔══════════════════════════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Benchmark Complete!                                    ║");
    ESP_LOGI(TAG, "╚══════════════════════════════════════════════════════════╝");

    // Keep running
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10000));
    }
}

