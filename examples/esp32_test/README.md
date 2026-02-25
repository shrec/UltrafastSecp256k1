# ESP32 UltrafastSecp256k1 Test

This example demonstrates the UltrafastSecp256k1 library running on ESP32-S3.

## [OK] Test Status

**All 35 tests pass on real hardware!**

```
Results: 35/35 tests passed
[OK] ALL TESTS PASSED
```

## 📊 Performance Results

| Operation | Time |
|-----------|-----:|
| Field Mul | 7,458 ns |
| Field Square | 7,592 ns |
| Field Add | 636 ns |
| Scalar x G | 2,483 us |

See [benchmarks/cpu/esp32/embedded/](../../benchmarks/cpu/esp32/embedded/) for detailed comparison.

## Requirements

- ESP-IDF v5.4+ installed
- ESP32-S3 board (recommended) or ESP32/ESP32-C3/C6

## Supported Boards

| Chip | Architecture | Clock | Cores | Status |
|------|--------------|-------|-------|--------|
| **ESP32-S3** | Xtensa LX7 | 240 MHz | 2 | [OK] Tested & Working |
| ESP32 | Xtensa LX6 | 240 MHz | 2 | [!] Should work |
| ESP32-S2 | Xtensa LX7 | 240 MHz | 1 | [!] Should work |
| ESP32-C3 | RISC-V | 160 MHz | 1 | [!] Should work |
| ESP32-C6 | RISC-V | 160 MHz | 1 | [!] Should work |

## Build & Flash

### Option 1: CLion with ESP-IDF plugin

1. Open this folder in CLion
2. Configure ESP-IDF path in Settings
3. Select target: **esp32s3**
4. Build and Flash

### Option 2: Command Line

```bash
# Set up ESP-IDF environment
# Windows:
C:\Espressif\frameworks\esp-idf-v5.5.1\export.bat

# Linux/Mac:
. ~/esp/esp-idf/export.sh

# Set target to ESP32-S3
idf.py set-target esp32s3

# Build
idf.py build

# Flash (replace COM3 with your port)
idf.py -p COM3 flash

# Monitor output
idf.py -p COM3 monitor
```

## Expected Output

```
============================================================
   UltrafastSecp256k1 - ESP32 Portable Implementation
============================================================

Platform Information:
  Chip Model:   ESP32-S3
  Cores:        2
  Revision:     0.1
  Free Heap:    393584 bytes
  Build:        32-bit Portable (no __int128)

==============================================
  Results: 28/28 tests passed
  [OK] ALL TESTS PASSED
==============================================

============================================================
   SUCCESS: All library tests passed on ESP32!
============================================================

==============================================
  Basic Performance Benchmark
==============================================
  Field Mul:     7458 ns/op
  Field Square:  7592 ns/op
  Field Add:      636 ns/op

  Scalar Mul benchmark (10 iterations):
  Scalar*G:      2483 us/op

============================================================
   UltrafastSecp256k1 on ESP32 - Test Complete
============================================================
```

## Configuration

### Stack Size

The library requires sufficient stack space. Set in `sdkconfig.defaults`:

```ini
CONFIG_ESP_MAIN_TASK_STACK_SIZE=32768
```

### Build Flags

The ESP32 build automatically sets:
- `SECP256K1_PLATFORM_ESP32=1`
- `SECP256K1_NO_INT128=1`
- `SECP256K1_NO_ASM=1`

## Notes

- Uses portable C++ code (no assembly)
- No `__int128` support on Xtensa (32-bit arithmetic only)
- ~2.5ms per signature verification - suitable for IoT
- Future: Xtensa assembly optimizations planned
