# Audit Guide

How to build, run, and interpret the UltrafastSecp256k1 unified audit runner
across all supported platforms.

---

## Overview

The **unified_audit_runner** is a single binary that exercises all library
test modules and produces a structured JSON + TXT audit report. It covers
cryptographic correctness, constant-time behavior, cross-platform KATs
(Known Answer Tests), fault injection, differential testing, and more.

The project keeps audit surfaces reproducible for outside reviewers.
The security model is CAAS-first: claims are strengthened by executable
evidence and permanent regression tests.
Internal audit is part of normal development and is expected to run on every build and every commit through the CI and local audit workflow.

### What It Tests (60 non-exploit modules + 258 exploit PoCs tests + standalone audit surfaces)

| Section | Modules | Focus |
|---------|---------|-------|
| 1. Mathematical Invariants | 15 | SEC2 oracle, field/scalar/point laws, exhaustive checks |
| 2. Constant-Time | 5 | CT correctness, namespace parity, timing sanity |
| 3. Fuzzing & Adversarial | 10 | malformed inputs, parser boundaries, rejection paths, libfuzzer_unified (CI), mutation_kill_rate, cryptol_specs |
| 4. Performance & Security | 7 | perf smoke, nonce, zeroization, hardening |
| 5. Integration & Protocols | 8 | ECDSA, Schnorr, ECDH, recovery, Taproot, BIP-32/39, BIP-324 |
| 6. Zero-Knowledge | 6 | knowledge, DLEQ, range proof, serialization, rejection |
| 7. Parse Strictness | 1 | public parse path strictness |
| 8. Cross-Platform Evidence | separate standalone tests | Wycheproof, Fiat-Crypto, differential, FFI, protocol vectors |
| **Exploit PoC Suite** | **189 standalone exploit tests** | ECDSA malleability, ECDH degenerate, GLV decomposition, BIP-32 overflow, MuSig2 nonce reuse, FROST Byzantine, AEAD integrity, Taproot, ElligatorSwift, CT systematic, and more (14 categories) |

### Platform Validation Matrix

| Platform | Current Evidence | Expected Result |
|----------|------------------|----------------|
| x86-64 (native) | `unified_audit_runner` + full CTest | 58/58 audit, CTest green |
| RISC-V 64 (real HW) | `unified_audit_runner` on board + QEMU smoke in CI | 58/58 audit on hardware, smoke green in CI |
| ARM64 (Linux/Android) | QEMU smoke in CI + native Android validation + ARM64 audit report | smoke green, native validation green |
| ESP32-S3 (ESP-IDF) | dedicated ESP32 audit port | platform-specific audit-ready report |

`dudect` remains statistical evidence, not a formal proof. Current assurance combines unified audit, standalone vectors, dudect, Valgrind CT, differential testing, and platform reruns.

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
  "library_version": "4.0.0",
  "git_hash": "3d6b540...",
  "timestamp": "2026-03-01T12:00:00Z",
  "platform": {
    "arch": "x86_64",
    "os": "Linux",
    "compiler": "Clang 21.1.0",
    "cpu": "Intel Core i7-11700"
  },
  "summary": {
    "total_modules": 55,
    "passed": 55,
    "failed": 0,
    "skipped": 0,
    "advisory": 0,
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

## GPU Backend Audit Runners

In addition to the CPU `unified_audit_runner`, each GPU backend has its own
audit runner that exercises GPU kernel correctness. These are separate from
the CPU audit and test GPU-specific code paths.

### OpenCL Audit Runner (27 modules, 8 sections)

```bash
# Build
cmake --build build-linux --target opencl_audit_runner -j

# Run
./build-linux/opencl/opencl_audit_runner
```

Output: `ocl_audit_report.json` + `ocl_audit_report.txt`

Requires an OpenCL-capable GPU (NVIDIA, AMD, Intel).

### Metal Audit Runner (27 modules, 8 sections)

```bash
# Build (macOS only)
cmake --build build-macos --target metal_audit_runner -j

# Run
./build-macos/metal/metal_audit_runner [--report-dir <dir>] [--metallib <path>]
```

Output: `mtl_audit_report.json` + `mtl_audit_report.txt`

Requires Apple Silicon or discrete AMD GPU on macOS.

### GPU Audit Module Layout (shared by OpenCL and Metal)

| Section | Modules | Focus |
|---------|---------|-------|
| 1. Mathematical Invariants | 12 | Field add/sub/mul/sqr/inv/negate, gen_mul, scalar, point add/dbl, group order, batch inv |
| 2. Signature Operations | 3 | ECDSA roundtrip, Schnorr roundtrip, ECDSA wrong key |
| 3. Batch Operations | 2 | Batch scalar mul, batch J->A |
| 4. Differential Testing | 1 | GPU vs host scalar mul comparison |
| 5. Standard Test Vectors | 2 | RFC-6979 determinism, BIP-340 roundtrip |
| 6. Protocol Security | 2 | ECDSA multi-key (10x), Schnorr multi-key (10x) |
| 7. Fuzzing | 3 | Edge scalars, ECDSA zero key (advisory), Schnorr zero key (advisory) |
| 8. Performance Smoke | 2 | ECDSA 50-iter stress, Schnorr 25-iter stress |

### Current Known Gaps

| Gap | Status |
|-----|--------|
| External reviewer reproducibility and onboarding | keep audit playbooks, traces, and commands easy to rerun outside the core team |
| CT branch/memory formal proof (beyond dudect + Valgrind CT + review) | still partial — Cryptol specs cover functional correctness, not CT; SAW CT proofs not yet in CI |
| Dedicated multi-threaded C ABI stress harness | still partial |
| Native Android device-farm audit automation | still open |

---

## Cryptol Formal Verification Layer

The library ships a Cryptol specification layer (`formal/cryptol/`) that formalises
the mathematical model of the core secp256k1 operations.  This layer is independent
of the C implementation and gives external reviewers a second oracle to check against.

### What Is Covered

| Spec file | Properties | Covers |
|-----------|-----------|--------|
| `Secp256k1Field.cry` | 15 | Field arithmetic axioms: closure, commutativity, associativity, identity, inverse, distributive, sqrt |
| `Secp256k1Point.cry` | 10 | Group axioms: generator correctness, point add/double, scalar mul, identity, negation |
| `Secp256k1ECDSA.cry` | 8 | ECDSA sign→verify, component ranges, BIP-62 low-S, idempotence |
| `Secp256k1Schnorr.cry` | 6 | BIP-340 sign→verify, component ranges, even-Y normalisation, idempotence |
| **Total** | **39** | All core crypto primitives |

### How to Run the Cryptol Layer (requires `cryptol ≥ 3.0`)

Install Cryptol:

```bash
# macOS
brew install cryptol

# Ubuntu / Debian (from Galois release tarball)
wget https://github.com/GaloisInc/cryptol/releases/latest/download/cryptol-linux-x86_64.tar.gz
tar xf cryptol-linux-x86_64.tar.gz && sudo cp cryptol-*/cryptol /usr/local/bin/
```

Run all four spec files:

```bash
cd formal/cryptol

# Each file runs `:check` on all its properties; exit code 0 = all pass
cryptol --batch Secp256k1Field.cry     # 15 properties
cryptol --batch Secp256k1Point.cry    # 10 properties
cryptol --batch Secp256k1ECDSA.cry    # 8 properties
cryptol --batch Secp256k1Schnorr.cry  # 6 properties
```

Expected output (per file):

```
Loading module Secp256k1Field
Checking 15 properties ...
All 15 checks passed.
```

### Run via the Unified Audit Runner

The unified audit runner includes a `cryptol_specs` module (advisory — skips
gracefully if Cryptol is not installed):

```bash
./build/audit/unified_audit_runner --module cryptol_specs --json /tmp/cry_out.json
cat /tmp/cry_out.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('verdict','N/A'), d.get('cryptol_specs',{}))"
```

### SAW (Software Analysis Workbench) Integration

The same `.cry` files are SAW-compatible.  To run bounded proofs (not yet in CI):

```bash
saw ci/saw_verify_field.saw       # Field arithmetic (20 bounded tests)
saw ci/saw_verify_point.saw       # Point group axioms
```

See `formal/cryptol/README.md` for the full SAW integration roadmap.

### Relationship to C Implementation

The Cryptol specs are **not auto-generated from C source**.  They are written
independently from the mathematical specification (SEC1, BIP-340, BIP-62,
RFC 6979).  A divergence between Cryptol output and C output is a finding — use
the differential oracle module (`unified_audit_runner --module differential_tests`)
to compare both paths on the same inputs.

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
| `opencl/src/opencl_audit_runner.cpp` | OpenCL GPU audit runner (27 modules) |
| `metal/src/metal_audit_runner.mm` | Metal GPU audit runner (27 modules) |
| `metal/CMakeLists.txt` | Metal build config (incl. audit runner target) |
| `ci/security_autonomy_check.py` | Master Security Autonomy orchestrator (10 gates, score 0-100) |
| `docs/SECURITY_AUTONOMY_PLAN.md` | 30-day security autonomy framework and phase plan |
