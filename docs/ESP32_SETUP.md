# ESP32 Setup Guide for UltrafastSecp256k1

Guide for testing the library on ESP32 using CLion IDE.

---

## 1. ESP-IDF Installation (Windows)

### Option A: ESP-IDF Installer (Recommended)

1. Download: https://dl.espressif.com/dl/esp-idf/
2. Run `esp-idf-tools-setup-x.x.exe`
3. Choose ESP-IDF version (v5.2+ recommended)
4. Wait for installation (~5-10 minutes)

### Option B: Manual Installation

```powershell
# 1. Clone ESP-IDF
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
git checkout v5.2

# 2. Install tools
.\install.ps1

# 3. Export environment
.\export.ps1
```

---

## 2. CLion Plugin Installation

1. **File → Settings → Plugins**
2. Search: "ESP-IDF"
3. Install: **"ESP-IDF" by JetBrains**
4. Restart CLion

---

## 3. CLion ESP-IDF Configuration

### Settings → Build → ESP-IDF

| Setting | Value |
|---------|-------|
| ESP-IDF Path | `C:\Espressif\frameworks\esp-idf-v5.5.1` |
| Python | `C:\Espressif\python_env\idf5.5_py3.11\Scripts\python.exe` |
| Target | `esp32` / `esp32s3` / `esp32c3` |

---

## 4. Creating a Project for ESP32

Let's create an ESP32 test project:

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.16)

# ESP-IDF project setup
include($ENV{IDF_PATH}/tools/cmake/project.cmake)

project(secp256k1_esp32_test)
```

### main/CMakeLists.txt

```cmake
idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_timer
)

# Add our secp256k1 library
target_include_directories(${COMPONENT_LIB} PUBLIC
    "${CMAKE_SOURCE_DIR}/../cpu/include"
)
```

### main/main.cpp

```cpp
#include <stdio.h>
#include "esp_timer.h"
#include "esp_log.h"

// Simplified field element for ESP32 (no x86 assembly)
#include <cstdint>
#include <array>

static const char* TAG = "secp256k1";

// Portable 256-bit field element
struct FieldElement {
    uint64_t limbs[4];
    
    static FieldElement one() {
        return {{1, 0, 0, 0}};
    }
    
    static FieldElement from_uint64(uint64_t v) {
        return {{v, 0, 0, 0}};
    }
};

// Portable field multiplication (no assembly)
FieldElement field_mul_portable(const FieldElement& a, const FieldElement& b) {
    // Simplified Comba multiplication
    __uint128_t t[8] = {0};
    
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            t[i+j] += (__uint128_t)a.limbs[i] * b.limbs[j];
        }
    }
    
    // Carry propagation
    for (int i = 0; i < 7; i++) {
        t[i+1] += t[i] >> 64;
        t[i] &= 0xFFFFFFFFFFFFFFFFULL;
    }
    
    // Reduction mod p = 2^256 - 2^32 - 977
    // Simplified: just return lower 256 bits (not fully reduced)
    return {{(uint64_t)t[0], (uint64_t)t[1], (uint64_t)t[2], (uint64_t)t[3]}};
}

extern "C" void app_main() {
    ESP_LOGI(TAG, "=== UltrafastSecp256k1 ESP32 Test ===");
    
    // Test field multiplication
    FieldElement a = FieldElement::from_uint64(0x12345678);
    FieldElement b = FieldElement::from_uint64(0x87654321);
    
    int64_t start = esp_timer_get_time();
    
    const int iterations = 10000;
    FieldElement result = a;
    for (int i = 0; i < iterations; i++) {
        result = field_mul_portable(result, b);
    }
    
    int64_t end = esp_timer_get_time();
    int64_t elapsed_us = end - start;
    
    ESP_LOGI(TAG, "Field Mul: %lld ns/op", (elapsed_us * 1000) / iterations);
    ESP_LOGI(TAG, "Result[0]: 0x%llx", result.limbs[0]);
    
    ESP_LOGI(TAG, "=== Test Complete ===");
}
```

---

## 5. Build & Flash

### From CLion:

1. **Run → Edit Configurations**
2. Add: **ESP-IDF → Build**
3. Add: **ESP-IDF → Flash**
4. Select COM port (e.g. `COM3`)

### Command Line:

```bash
# ESP-IDF environment
. $HOME/esp/esp-idf/export.sh  # Linux/Mac
# or Windows: C:\Espressif\frameworks\esp-idf-v5.2\export.bat

# Build
idf.py build

# Flash
idf.py -p COM3 flash

# Monitor
idf.py -p COM3 monitor
```

---

## 6. ESP32 Limitations

### What works:
- ✅ Portable C++ field arithmetic
- ✅ Scalar operations
- ✅ Point operations (slow)
- ✅ Basic tests

### What does not work:
- ❌ x86 assembly (BMI2/ADX)
- ❌ RISC-V assembly (ESP32 is Xtensa, not RISC-V!)
- ❌ AVX2/SIMD
- ❌ 64-bit native (ESP32 is 32-bit)

### ESP32 Variants:

| Chip | Architecture | Bits | Notes |
|------|--------------|------|-------|
| ESP32 | Xtensa LX6 | 32-bit | Original, dual-core |
| ESP32-S2 | Xtensa LX7 | 32-bit | Single-core, USB |
| ESP32-S3 | Xtensa LX7 | 32-bit | AI acceleration |
| **ESP32-C3** | **RISC-V** | **32-bit** | ✅ RISC-V! |
| **ESP32-C6** | **RISC-V** | **32-bit** | ✅ RISC-V + WiFi 6 |
| **ESP32-H2** | **RISC-V** | **32-bit** | ✅ RISC-V + Zigbee |

### Recommendation:
Use **ESP32-C3/C6** — these are RISC-V and some of our RISC-V optimizations may work (32-bit version).

---

## 7. ESP32-C3 (RISC-V) Specific Setup

If you have an ESP32-C3:

```cmake
# CMakeLists.txt
set(IDF_TARGET "esp32c3")
```

Or in CLion: **Settings → ESP-IDF → Target: esp32c3**

---

## 8. Expected Performance

| Chip | Field Mul | Notes |
|------|-----------|-------|
| ESP32 (Xtensa) | ~5-10 μs | 32-bit, no optimization |
| ESP32-C3 (RISC-V) | ~2-5 μs | 32-bit RISC-V |
| ESP32-S3 | ~3-8 μs | Dual-core Xtensa |

For comparison:
- x86-64: 33 ns
- RISC-V 64: 198 ns
- ESP32: ~5000 ns (150× slower)

ESP32 is primarily for IoT/Embedded, not high-performance crypto.

---

## Resources

- [ESP-IDF Documentation](https://docs.espressif.com/projects/esp-idf/)
- [CLion ESP-IDF Plugin](https://plugins.jetbrains.com/plugin/18499-esp-idf)
- [ESP32-C3 RISC-V](https://www.espressif.com/en/products/socs/esp32-c3)


