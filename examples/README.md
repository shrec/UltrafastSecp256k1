# UltrafastSecp256k1 Examples

Usage examples for different platforms and use cases.

## Available Examples

### esp32_test -- ESP32-S3 Selftest & Benchmark [STABLE]

**Location:** [`esp32_test/`](esp32_test/)

Complete ESP32-S3 example demonstrating:
- Library self-test execution
- Field arithmetic benchmarks
- Point multiplication performance
- ESP-IDF integration

**Status:** [STABLE] -- 28/28 tests pass, audited on real hardware

**Quick Start:**
```bash
cd esp32_test
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

See [esp32_test/README.md](esp32_test/README.md) for detailed instructions.

---

### esp32_bench_hornet -- ESP32-S3 bench_hornet Benchmark [STABLE]

**Location:** [`esp32_bench_hornet/`](esp32_bench_hornet/)

Full bench_hornet benchmark suite ported for ESP32-S3:
- 6-operation apple-to-apple comparison vs libsecp256k1
- CT and FAST mode benchmarks
- Block validation simulation

**Status:** [STABLE] -- benchmarked on real ESP32-S3 hardware

**Quick Start:**
```bash
cd esp32_bench_hornet
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

---

### basic_usage -- Desktop Usage Example [STABLE]

**Location:** [`basic_usage/`](basic_usage/)

Demonstrates core library usage: key generation, ECDSA signing/verification,
Schnorr signing/verification, field arithmetic.

**Status:** [STABLE]

---

### signing_demo -- ECDSA + Schnorr Signing [STABLE]

**Location:** [`signing_demo/`](signing_demo/)

End-to-end signing and verification demo for ECDSA and BIP-340 Schnorr.

**Status:** [STABLE]

---

### threshold_demo -- Threshold Signatures [STABLE]

**Location:** [`threshold_demo/`](threshold_demo/)

Threshold signature scheme demonstration.

**Status:** [STABLE]

---

### stm32_test -- STM32 Embedded Port [EXPERIMENTAL]

**Location:** [`stm32_test/`](stm32_test/)

STM32F103 (Cortex-M3) port. Runs core field arithmetic and point operations
on extremely constrained hardware (72 MHz, 20 KB SRAM).

**Status:** [EXPERIMENTAL] -- working but not part of audit campaign
