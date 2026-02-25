# UltrafastSecp256k1 - ESP32 Benchmarks

Performance benchmarks on ESP32-S3 embedded platform.

## 📊 Platform Information

| Property | Value |
|----------|-------|
| **Chip** | ESP32-S3 |
| **Cores** | 2 x Xtensa LX7 |
| **Frequency** | 240 MHz |
| **RAM** | 512 KB SRAM |
| **Build Mode** | Portable C++ (no assembly, no __int128) |
| **ESP-IDF Version** | v5.5.1 |
| **Compiler** | xtensa-esp32s3-elf-g++ 14.2.0 |
| **Optimization** | -O3 |
| **Date** | 2026-02-13 |

## 🧪 Test Results

**All 28 library tests passed successfully!**

Verified operations:
- [OK] Field arithmetic (add, sub, mul, sqr, inverse)
- [OK] Scalar arithmetic
- [OK] Point operations (add, double, multiply)
- [OK] Generator point multiplications
- [OK] Point group identities
- [OK] Test vectors (NIST-style verification)

## 📈 Benchmark Results

### Field Operations

| Operation | Time |
|-----------|-----:|
| Field Mul | 7,458 ns |
| Field Square | 7,592 ns |
| Field Add | 636 ns |

### Point Operations

| Operation | Time |
|-----------|-----:|
| Scalar x G (Generator Mul) | 2,483 us |

## 📊 Comparison with Other Platforms

| Platform | Clock | Field Mul | ScalarxG |
|----------|------:|----------:|---------:|
| **ESP32-S3** | 240 MHz | 7,458 ns | 2,483 us |
| Milk-V Mars (RISC-V) | 1.5 GHz | 197 ns | 40 us |
| x86-64 (i5) | 3.5 GHz | 33 ns | 5 us |

**Notes:**
- ESP32-S3 uses portable 32-bit arithmetic (no `__int128`)
- No assembly optimizations (yet)
- Performance is ~38x slower than x86-64, reasonable for a 240 MHz MCU
- Future: Xtensa assembly optimizations planned

## [TOOL] Build Configuration

```cmake
# ESP32 build flags
-DSECP256K1_PLATFORM_ESP32=1
-DSECP256K1_NO_INT128=1
-DSECP256K1_NO_ASM=1
```

```ini
# sdkconfig.defaults
CONFIG_IDF_TARGET="esp32s3"
CONFIG_COMPILER_OPTIMIZATION_PERF=y
CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ_240=y
CONFIG_ESP_MAIN_TASK_STACK_SIZE=32768
```

## 💡 Use Cases

ESP32-S3 with UltrafastSecp256k1 is suitable for:
- IoT device authentication
- Hardware wallet prototypes
- Bitcoin/Ethereum address generation (low volume)
- Signature verification (with appropriate timeouts)
- Educational/research purposes

**Not recommended for:**
- High-throughput signing operations
- Time-critical cryptographic services
- Batch processing (use GPU instead)

## 🚀 Future Optimizations

Planned improvements:
1. Xtensa-specific assembly for field arithmetic
2. DSP extension utilization (S3-specific)
3. Memory optimization for reduced stack usage
4. Optional precomputed tables in flash

## 📝 Notes

- Stack overflow protection: Requires `CONFIG_ESP_MAIN_TASK_STACK_SIZE=32768` minimum
- All tests pass on real hardware (ESP32-S3 development board)
- Serial output at 115200 baud
