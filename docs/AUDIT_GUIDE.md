# Audit Guide

How to build, run, and interpret the UltrafastSecp256k1 unified audit runner
across all supported platforms.

---

## Overview

The **unified_audit_runner** is a single binary that exercises all library
test modules and produces a structured JSON + TXT audit report. It covers
cryptographic correctness, constant-time behavior, cross-platform KATs
(Known Answer Tests), fault injection, differential testing, and more.

### What It Tests (49 modules, 8 sections)

| Section | Modules | Focus |
|---------|---------|-------|
| 1. Core Arithmetic | 5 | Field mul, square, add, inversion, carry propagation |
| 2. ECC Operations | 5 | Point add, double, scalar mul, generator mul, properties |
| 3. Signing & Verification | 5 | ECDSA, Schnorr, RFC 6979 vectors, BIP-340 vectors, BIP-340 strict |
| 4. Advanced Protocols | 5 | ECDH, key recovery, taproot, MuSig2, BIP-32 |
| 5. Batch & SIMD | 3 | Batch (affine, multi-scalar), SIMD batch |
| 6. Safety & Robustness | 8 | CT equivalence, fault injection, debug invariants, ABI gate, differential |
| 7. Cross-Platform | 4 | Fiat-crypto vectors, cross-platform KAT, exhaustive small-group, comprehensive |
| 8. Side-Channel | 2 | CT mode verification, dudect smoke test |

### Platform Support Matrix

| Platform | Modules Run | Expected Result |
|----------|------------|----------------|
| x86-64 (any OS) | 49 | 48/49 PASS (1 advisory: dudect smoke) |
| RISC-V 64 (real HW) | 49 | 48/49 PASS (1 advisory: dudect smoke) |
| ARM64 (Linux/Android) | 49 | 48/49 PASS (1 advisory: dudect smoke) |
| ESP32-S3 (ESP-IDF) | 41 | 40/40 PASS (8 skipped: platform-incompatible) |

The **dudect smoke** module is always advisory -- it performs a statistical
side-channel timing test that may show variance on real hardware without
indicating a real vulnerability.

---

## Build Instructions

### 1. x86-64 (Native)

```bash
# Configure
cmake -S . -B build-audit -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build the audit runner
cmake --build build-audit --target unified_audit_runner -j

# Run
./build-audit/audit/unified_audit_runner
```

Output files created in the current directory:
- `audit_report.json` -- machine-readable structured result
- `audit_report.txt` -- human-readable summary

To write reports to a specific directory:
```bash
./unified_audit_runner --report-dir /path/to/output/
```

### 2. RISC-V 64 (Cross-compile for Milk-V Mars)

```bash
# Configure (using WSL or Linux host)
cmake -S . -B build-riscv-audit -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=cmake/riscv64-toolchain.cmake

# Build
cmake --build build-riscv-audit --target unified_audit_runner -j

# Deploy
scp build-riscv-audit/audit/unified_audit_runner user@192.168.1.31:/tmp/

# Run on real hardware
ssh user@192.168.1.31 /tmp/unified_audit_runner

# Retrieve reports
scp user@192.168.1.31:audit_report.json ./riscv64-audit.json
scp user@192.168.1.31:audit_report.txt  ./riscv64-audit.txt
```

### 3. ARM64 Android (Cross-compile via NDK)

```bash
cmake -S . -B build-android-audit -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28

cmake --build build-android-audit --target unified_audit_runner -j

adb push build-android-audit/audit/unified_audit_runner /data/local/tmp/
adb shell chmod +x /data/local/tmp/unified_audit_runner
adb shell /data/local/tmp/unified_audit_runner
adb pull /data/local/tmp/audit_report.json .
adb pull /data/local/tmp/audit_report.txt .
```

### 4. ESP32-S3 (ESP-IDF)

The ESP32 version uses a port in `tests/esp32_audit/` (not the native runner)
because of limited memory and incompatible modules.

```bash
cd tests/esp32_audit
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

Results print to serial console. JSON output is embedded in the serial
stream between `JSON_BEGIN` and `JSON_END` markers.

---

## Report Format

### JSON Report Schema

```json
{
  "framework_version": "2.0.0",
  "library_version": "3.16.0",
  "git_hash": "3d6b540...",
  "timestamp": "2026-03-01T12:00:00Z",
  "platform": {
    "arch": "x86_64",
    "os": "Linux",
    "compiler": "Clang 21.1.0",
    "cpu": "Intel Core i7-11700"
  },
  "summary": {
    "total_modules": 49,
    "passed": 48,
    "failed": 0,
    "skipped": 0,
    "advisory": 1,
    "verdict": "AUDIT-READY"
  },
  "sections": [
    {
      "name": "Core Arithmetic",
      "modules": [
        { "name": "field_mul", "result": "PASS", "time_ms": 245 }
      ]
    }
  ]
}
```

### Verdict Logic

| Condition | Verdict |
|-----------|---------|
| All modules PASS (or advisory only) | AUDIT-READY |
| Any module FAIL | AUDIT-FAIL |
| Skip count > 0 but 0 failures | AUDIT-READY (with notes) |

---

## Docker CI Integration

The audit runner is also executed automatically in CI via Docker:

```bash
# Using the local CI script (Windows PowerShell)
docker/local_ci.ps1

# Or directly
docker build -f docker/Dockerfile.ci -t ultra-ci .
docker run --rm ultra-ci
```

The CI container builds and runs the audit runner as part of the test suite.
See `.github/workflows/audit-report.yml` for the GitHub Actions configuration.

---

## Interpreting Results

### Common Advisory: dudect smoke

The `test_ct_sidechannel_smoke` module runs a simplified dudect
(detection of unintended computation time) test. This is a statistical
test that may report variance on platforms with:
- CPU frequency scaling (turbo boost, power saving)
- Background OS activity
- Thermal throttling

An advisory result on dudect does NOT indicate a side-channel
vulnerability -- it means the quick smoke test was inconclusive.
Full dudect analysis requires a dedicated, controlled environment.

### Timing Reference

| Platform | Expected Duration |
|----------|------------------|
| x86-64 (modern) | 30-60 seconds |
| RISC-V (U74) | 200-300 seconds |
| ARM64 (A55) | 60-120 seconds |
| ESP32-S3 | 500-700 seconds |

---

## Files

| File | Purpose |
|------|---------|
| `audit/unified_audit_runner.cpp` | Main audit runner (all platforms) |
| `audit/CMakeLists.txt` | Build configuration for audit targets |
| `audit/platform-reports/` | Generated reports for all platforms |
| `audit/platform-reports/PLATFORM_AUDIT.md` | Cross-platform audit summary |
| `tests/esp32_audit/` | ESP32-S3 port of the audit |
| `docker/Dockerfile.ci` | CI container for automated auditing |
| `.github/workflows/audit-report.yml` | GitHub Actions audit workflow |
