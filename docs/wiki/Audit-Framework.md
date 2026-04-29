# Audit Framework

> **UltrafastSecp256k1** ships a comprehensive, multi-layer audit framework that runs
> **49+ automated test modules** across **8 verification domains**, targeting
> **4 CPU platforms** and **3 GPU backends**. This page describes what the framework
> covers, how it works, and how to reproduce every result.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Verification Domains (8 Sections)](#verification-domains-8-sections)
- [Test Module Inventory (49+ modules)](#test-module-inventory-49-modules)
- [Full Audit Orchestrator (A-M Categories)](#full-audit-orchestrator-a-m-categories)
- [CI Workflows](#ci-workflows)
- [Platform Support Matrix](#platform-support-matrix)
- [Verdict Logic](#verdict-logic)
- [Constant-Time Verification](#constant-time-verification)
- [Cross-Library Differential Testing](#cross-library-differential-testing)
- [How to Run](#how-to-run)
- [Report Formats](#report-formats)
- [Documentation Index](#documentation-index)

---

## Architecture Overview

```
                          +---------------------------+
                          |  unified_audit_runner      |
                          |  (49+ modules, 8 sections) |
                          +---------------------------+
                                     |
              +-----+-----+-----+---+---+-----+-----+-----+
              |     |     |     |       |     |     |     |
           Sec.1 Sec.2 Sec.3 Sec.4  Sec.5 Sec.6 Sec.7 Sec.8
           Math   CT   Diff  Vectors Fuzz  Proto  ABI  Perf
```

The framework is built around a **single unified binary** (`unified_audit_runner`)
that orchestrates all 49+ test modules, produces structured reports (JSON, TXT, SARIF),
and computes an automated verdict.

On top of this, a **full orchestrator** (`run_full_audit.sh` / `run_full_audit.ps1`,
~1000 lines) runs 13 categories (A-M) covering everything from build integrity to
documentation consistency.

All tests run in **CI** via GitHub Actions on every push, PR, nightly, and weekly
schedules across multiple platforms and compilers.

---

## Verification Domains (8 Sections)

| # | Section ID | Domain | Modules | What It Proves |
|---|-----------|--------|--------:|----------------|
| 1 | `math_invariants` | Mathematical Invariants | 13 | Field Fp and Scalar Zn ring properties, group law correctness, ECC algebraic invariants |
| 2 | `ct_analysis` | Constant-Time & Side-Channel | 6 | No secret-dependent branches/memory accesses, dudect statistical timing, formal CT verification |
| 3 | `differential` | Differential & Cross-Library | 4 | Bit-exact match vs Fiat-Crypto reference, cross-platform KAT determinism |
| 4 | `standard_vectors` | Standard Test Vectors | 8 | BIP-340 (27/27), BIP-32, RFC 6979, Wycheproof ECDSA/ECDH, MuSig2 BIP-327, FROST KAT |
| 5 | `fuzzing` | Fuzzing & Adversarial | 4 | Parser robustness, fault injection resilience, random input survival |
| 6 | `protocol_security` | Protocol Security | 10 | ECDSA/Schnorr sign-verify round-trip, BIP-32 derivation, MuSig2/FROST multi-party protocols |
| 7 | `memory_safety` | ABI & Memory Safety | 4 | Debug invariants, ABI version gate, FFI round-trip, security hardening |
| 8 | `performance` | Performance Validation | 4 | Hash acceleration, SIMD batch, multi-scalar, performance smoke |

---

## Test Module Inventory (49+ modules)

### Section 1: Mathematical Invariants (Fp, Zn, Group Laws)

| Module | Source File | Checks | What It Tests |
|--------|-----------|-------:|---------------|
| `audit_field` | `audit_field.cpp` | 264,484 | Field Fp: identity, commutativity, associativity, distributivity, inverse, boundary, Fermat |
| `audit_scalar` | `audit_scalar.cpp` | 93,847 | Scalar Zn: ring properties, overflow, negate, inverse, boundary-near-order |
| `audit_point` | `audit_point.cpp` | 116,312 | Point ops: on-curve, group law, add, dbl, scalar_mul, compress/decompress, infinity |
| `exhaustive` | `test_exhaustive.cpp` | -- | Exhaustive small-order group verification |
| `comprehensive` | `test_comprehensive.cpp` | -- | Combined field/scalar/point coverage |
| `ecc_properties` | `test_ecc_properties.cpp` | -- | Algebraic ECC properties (associativity, distributivity, double-and-add) |
| `batch_add` | `test_batch_add_affine.cpp` | -- | Batch affine point addition correctness |
| `carry_propagation` | `test_carry_propagation.cpp` | 247 | Carry chain stress: all-ones, limb boundary, near-p, near-n |
| `field_52` | `test_field_52.cpp` | -- | 5x52 field representation (requires `__uint128_t`) |
| `field_26` | `test_field_26.cpp` | -- | 10x26 field representation |
| `mul` | (internal) | -- | Multiplication correctness |
| `arith_correct` | (internal) | -- | Arithmetic correctness meta-tests |
| `scalar_mul` | (internal) | -- | Scalar multiplication correctness |

### Section 2: Constant-Time & Side-Channel Analysis

| Module | Source File | Checks | What It Tests |
|--------|-----------|-------:|---------------|
| `audit_ct` | `audit_ct.cpp` | 120,128 | FAST-vs-CT equivalence, complete formulas, no-branch verification |
| `ct_equivalence` | `test_ct_equivalence.cpp` | -- | CT and FAST layer produce identical results for all operations |
| `ct_sidechannel` | `test_ct_sidechannel.cpp` | -- | dudect timing analysis: Welch t-test for side-channel leakage (smoke mode for CI, full for nightly) |
| `ct_verif_formal` | `test_ct_verif_formal.cpp` | -- | Formal CT verification via Valgrind MSAN taint tracking |
| `diag_scalar_mul` | `diag_scalar_mul.cpp` | -- | Scalar multiplication diagnostic |
| `ct` | (internal) | -- | CT layer meta-test |

### Section 3: Differential & Cross-Library Testing

| Module | Source File | Checks | What It Tests |
|--------|-----------|-------:|---------------|
| `differential` | `differential_test.cpp` | -- | Self-consistency: 5x52 vs 10x26 vs 4x64 cross-impl comparison |
| `fiat_crypto` | `test_fiat_crypto_vectors.cpp` | 647 | Independent reference / Sage golden vectors: mul, sqr, inv, add, sub |
| `fiat_crypto_link` | `test_fiat_crypto_linkage.cpp` | -- | Independent reference linkage cross-check (schoolbook oracle) |
| `cross_platform_kat` | `test_cross_platform_kat.cpp` | 24 | Cross-platform KAT: field, scalar, point, ECDSA, Schnorr |

### Section 4: Standard Test Vectors

| Module | Source File | Checks | What It Tests |
|--------|-----------|-------:|---------------|
| `bip340_vectors` | `test_bip340_vectors.cpp` | 27 | BIP-340 Schnorr official test vectors (27/27) |
| `bip340_strict` | `test_bip340_strict.cpp` | -- | BIP-340 strict encoding enforcement |
| `bip32_vectors` | `test_bip32_vectors.cpp` | -- | BIP-32 HD key derivation test vectors |
| `rfc6979_vectors` | `test_rfc6979_vectors.cpp` | 35 | RFC 6979 deterministic nonce test vectors (35/35) |
| `wycheproof_ecdsa` | `test_wycheproof_ecdsa.cpp` | -- | Google Wycheproof ECDSA secp256k1 vectors |
| `wycheproof_ecdh` | `test_wycheproof_ecdh.cpp` | -- | Google Wycheproof ECDH secp256k1 vectors |
| `frost_kat` | `test_frost_kat.cpp` | -- | FROST reference KAT vectors |
| `musig2_bip327` | `test_musig2_bip327_vectors.cpp` | -- | MuSig2 BIP-327 official reference vectors |

### Section 5: Fuzzing & Adversarial

| Module | Source File | Checks | What It Tests |
|--------|-----------|-------:|---------------|
| `audit_fuzz` | `audit_fuzz.cpp` | 15,423 | Adversarial fuzz: random inputs through all operation paths |
| `fuzz_parsers` | `test_fuzz_parsers.cpp` | 580K ops | Parser fuzz: DER, Schnorr, pubkey serialization |
| `fuzz_addr_bip32` | `test_fuzz_address_bip32_ffi.cpp` | 73,959 ops | Address/BIP32/FFI boundary fuzz |
| `fault_injection` | `test_fault_injection.cpp` | 610 | Fault injection simulation: bit-flips, coord corruption, GLV |

### Section 6: Protocol Security

| Module | Source File | Checks | What It Tests |
|--------|-----------|-------:|---------------|
| `ecdsa_schnorr` | (internal) | -- | ECDSA + Schnorr sign-verify round-trip |
| `bip32` | (internal) | -- | BIP-32 derivation chain correctness |
| `musig2` | (internal) | -- | MuSig2 2/3/5-of-n multi-party signing |
| `ecdh_recovery` | (internal) | -- | ECDH + key recovery protocols |
| `musig2_frost` | `test_musig2_frost.cpp` | 975 | MuSig2 + FROST protocol suite |
| `musig2_frost_adv` | `test_musig2_frost_advanced.cpp` | 316 | MuSig2 + FROST adversarial: rogue-key, malicious participant |
| `audit_integration` | `audit_integration.cpp` | 13,144 | End-to-end protocol flows: sign->verify, derive->use |
| `batch_randomness` | `test_batch_randomness.cpp` | -- | Batch verify weight randomness audit |
| `v4_features` | (internal) | -- | V4 feature set verification |
| `coins` | (internal) | -- | Coin-specific protocol tests |

### Section 7: ABI & Memory Safety

| Module | Source File | Checks | What It Tests |
|--------|-----------|-------:|---------------|
| `audit_security` | `audit_security.cpp` | 17,856 | Nonce uniqueness, invalid input rejection, edge-case handling |
| `debug_invariants` | `test_debug_invariants.cpp` | 372 | Normalize, on_curve, scalar_valid debug assertions |
| `abi_gate` | `test_abi_gate.cpp` | 12 | ABI version macro validation |
| `ffi_round_trip` | `test_ffi_round_trip.cpp` | -- | Cross-ABI/FFI round-trip via ufsecp C API |

### Section 8: Performance Validation

| Module | Source File | Checks | What It Tests |
|--------|-----------|-------:|---------------|
| `hash_accel` | `test_hash_accel.cpp` | -- | SHA-256 hardware acceleration (SHA-NI / ARMv8-A) |
| `simd_batch` | (internal) | -- | SIMD batch operation validation |
| `multiscalar` | (internal) | -- | Multi-scalar multiplication correctness |
| `audit_perf` | `audit_perf.cpp` | -- | Performance smoke: sign/verify timing within expected bounds |

### Conditional Modules (opt-in build flags)

| Module | Build Flag | Source File | What It Tests |
|--------|-----------|-----------|---------------|
| `cross_libsecp256k1` | `SECP256K1_BUILD_CROSS_TESTS=ON` | `test_cross_libsecp256k1.cpp` | ~7,860 checks vs bitcoin-core/libsecp256k1 v0.6.0 |
| `fuzz_parsers` | `SECP256K1_BUILD_FUZZ_TESTS=ON` | `test_fuzz_parsers.cpp` | 580K parser fuzz operations |
| `fuzz_addr_bip32_ffi` | `SECP256K1_BUILD_FUZZ_TESTS=ON` | `test_fuzz_address_bip32_ffi.cpp` | 73,959 FFI boundary fuzz |
| `musig2_frost` | `SECP256K1_BUILD_PROTOCOL_TESTS=ON` | `test_musig2_frost.cpp` | MuSig2 + FROST protocol suite |
| `musig2_frost_advanced` | `SECP256K1_BUILD_PROTOCOL_TESTS=ON` | `test_musig2_frost_advanced.cpp` | MuSig2 + FROST adversarial |
| `frost_kat` | `SECP256K1_BUILD_PROTOCOL_TESTS=ON` | `test_frost_kat.cpp` | FROST reference KAT |
| `musig2_bip327_vectors` | `SECP256K1_BUILD_PROTOCOL_TESTS=ON` | `test_musig2_bip327_vectors.cpp` | MuSig2 BIP-327 vectors |

---

## Full Audit Orchestrator (A-M Categories)

The shell orchestrators (`run_full_audit.sh` / `run_full_audit.ps1`) run **13 categories**
covering the entire software lifecycle from build to documentation:

| Cat | Name | What It Does |
|-----|------|-------------|
| **A** | Environment & Build Integrity | Toolchain fingerprint, CMake build, dependency scan (`ldd`/`otool`/`dumpbin`), SHA256 manifest |
| **B** | Packaging & Supply Chain | SBOM (CycloneDX 1.6), provenance JSON, reproducible build verification |
| **C** | Static Analysis | clang-tidy (30+ checks), cppcheck, dangerous-pattern grep scan |
| **D** | Sanitizers | ASan+UBSan, TSan, Valgrind memcheck |
| **E-I** | Correctness + CT + Fuzz | `unified_audit_runner` (all 49 modules) + full CTest suite |
| **I.extra** | CT Disassembly | `verify_ct_disasm.sh` -- scans CT function machine code for conditional branches |
| **J** | ABI / API Stability | `nm`/`dumpbin` symbol scan for exported `ufsecp_*` symbols |
| **K** | Bindings & FFI Parity | Scans `bindings/` per language, produces parity matrix JSON |
| **L** | Performance Regression | Runs `bench_unified`, checks for >20% regression |
| **M** | Documentation Consistency | Validates README, CHANGELOG, SECURITY, LICENSE, THREAT_MODEL, CONTRIBUTING, VERSION |

**Output:** `audit-output-<timestamp>/audit_report.md` + `artifacts/` tree with all evidence.

---

## CI Workflows

| Workflow | Trigger | Platforms | What It Runs |
|----------|---------|-----------|-------------|
| **[ci.yml]** | Push/PR to main/dev | Linux (GCC-13, Clang-17), Windows (MSVC), macOS (M1), ARM64 cross, Android NDK, WASM, iOS, ROCm/HIP, CUDA | Full CTest suite, sanitizers (ASan/UBSan/TSan), code coverage |
| **[audit-report.yml]** | Weekly + release tags | Linux GCC-13, Linux Clang-17, Windows MSVC | `unified_audit_runner` (all 49 modules), SARIF upload, verdict check |
| **[security-audit.yml]** | Push/PR to main + weekly | Linux (GCC-13, Clang-17) | ASan+UBSan, MSan, TSan, Valgrind memcheck, dudect timing |
| **[ct-verif.yml]** | Push/PR to main/dev | Linux Clang-17 | LLVM IR analysis for CT violations (deterministic, blocking) |
| **[valgrind-ct.yml]** | Push/PR to main/dev + nightly | Linux GCC-13 | Valgrind taint tracking: marks secrets as UNDEFINED, checks for data-dependent branches |
| **[ct-arm64.yml]** | Push/PR + nightly | macOS-14 (Apple M1) | Native ARM64 dudect timing analysis on real hardware |
| **[nightly.yml]** | Daily 03:00 UTC | Linux | Extended differential (M=100, ~1.3M checks), cross-libsecp256k1, full dudect (30 min) |
| **[bench-regression.yml]** | PR to main | Linux | Benchmark regression detection (>20% = warning, >50% = block) |

### Workflow Details

**audit-report.yml** (Weekly Unified Audit):
- Builds with all test flags: `BUILD_TESTING=ON`, `SECP256K1_BUILD_PROTOCOL_TESTS=ON`, `SECP256K1_BUILD_FUZZ_TESTS=ON`
- Runs `unified_audit_runner --report-dir ./audit-output --sarif`
- Uploads SARIF to GitHub Code Scanning (Linux GCC only)
- Verdict job: downloads all platform reports, checks `audit_verdict` in JSON -- any non-PASS fails the workflow
- Publishes artifacts to `gh-pages` branch under `audit/<date>/<run_id>/`
- Artifacts retained for 90 days

**security-audit.yml** (Sanitizer Suite, 6 jobs):
1. **-Werror build** (GCC-13): zero warnings required
2. **ASan + UBSan** (Clang-17): `detect_leaks=1:halt_on_error=1`
3. **MSan** (Clang-17): `halt_on_error=1:print_stacktrace=1`
4. **TSan** (Clang-17): `halt_on_error=1:second_deadlock_stack=1`
5. **Valgrind Memcheck** (GCC-13): `--leak-check=full --error-exitcode=1`
6. **dudect Timing** (GCC-13): 5-min timeout, advisory (non-blocking)

**ct-verif.yml** (CT Compile-Time Verification):
- Compiles `ct_field.cpp`, `ct_scalar.cpp`, `ct_sign.cpp` to LLVM IR
- Runs `ct-verif` LLVM pass (Borrello et al., CCS 2021) if available
- Fallback: scans `.ll` for `switch`, `variable_gep`, indirect calls
- **Deterministic, blocking** -- violations break the build

---

## Platform Support Matrix

### CPU Platforms

| Platform | Hardware | Modules Run | Expected Result | Typical Time |
|----------|----------|:-----------:|:---------------:|:------------:|
| x86-64 (any OS) | Intel/AMD | 49 | 48 PASS + 1 advisory | 30-60s |
| ARM64 (Linux/Android/macOS) | A76, M1/M3 | 49 | 48 PASS + 1 advisory | 60-120s |
| RISC-V 64 (real hardware) | SiFive U74 | 49 | 48 PASS + 1 advisory | 200-300s |
| ESP32-S3 (Xtensa LX7) | 240 MHz | 41 | 40 PASS + 1 advisory | 500-700s |

> The 1 advisory module is `ct_sidechannel` (dudect) -- a statistical test that may
> produce false positives on shared CI runners. It is non-blocking by design.

### GPU Backends

| Backend | Hardware | Tests | Status |
|---------|----------|-------|--------|
| CUDA | RTX 5060 Ti (SM 86/89) | Field, point, scalar, ECDSA, Schnorr, batch | Full test coverage |
| OpenCL | RTX 5060 Ti, AMD | Field, point, scalar, kG | Core coverage |
| Apple Metal | M3 Pro (18 GPU cores) | Field, point, scalar, kG | Core coverage |

### Compiler Matrix (CI)

| Compiler | Version | Platforms | Sanitizers |
|----------|---------|-----------|------------|
| GCC | 13.x | Linux x86-64, ARM64 cross | ASan, UBSan, Valgrind |
| Clang | 17.x / 21.x | Linux x86-64, RISC-V cross | ASan, UBSan, MSan, TSan |
| MSVC | 2022 | Windows x86-64 | -- |
| AppleClang | latest | macOS ARM64 (M1) | -- |
| NDK Clang | r26 (17.0.2) | Android ARM64/ARMv7/x86 | -- |
| Emscripten | latest | WebAssembly | -- |

---

## Verdict Logic

### Unified Audit Runner

```
if (failures == 0)
    verdict = "AUDIT-READY"
else if (only_advisory_failures)
    verdict = "AUDIT-READY"    // advisory modules don't block
else
    verdict = "AUDIT-BLOCKED"
```

The only **advisory** module is `ct_sidechannel` (dudect smoke) -- its failure does not
block the audit verdict since dudect is a statistical test that requires isolated hardware
for reliable results.

### CI Verdict (audit-report.yml)

The `verdict` job downloads JSON reports from all platform builds (Linux GCC, Linux Clang,
Windows MSVC). **Every** platform must independently report `PASS` -- a single platform
failure blocks the entire workflow.

---

## Constant-Time Verification

The CT verification strategy uses **four independent methods**:

| Layer | Method | Tool | Blocking? | What It Catches |
|-------|--------|------|:---------:|-----------------|
| 1 | Compile-time IR analysis | ct-verif LLVM pass | **Yes** | Secret-dependent `switch`, `variable_gep`, indirect calls in LLVM IR |
| 2 | Valgrind taint tracking | `VALGRIND_MAKE_MEM_UNDEFINED` | **Yes** | Data-dependent conditional jumps at binary level |
| 3 | Disassembly scan | `verify_ct_disasm.sh` | **Yes** | Conditional branches in CT function machine code |
| 4 | Statistical timing | dudect (Welch t-test) | No (advisory) | Measurable timing differences (|t| > 4.5 = leak) |

### What Gets CT-Verified

| Function | File | All 4 Layers? |
|----------|------|:---:|
| `ct::field_mul` | `ct_field.cpp` | Yes |
| `ct::field_sqr` | `ct_field.cpp` | Yes |
| `ct::field_inv` (SafeGCD) | `ct_field.cpp` | Yes |
| `ct::scalar_inverse` (SafeGCD) | `ct_scalar.cpp` | Yes |
| `ct::scalar_mul` (k*P) | `ct_sign.cpp` | Yes |
| `ct::generator_mul` (k*G) | `ct_sign.cpp` | Yes |
| `ct::ecdsa_sign` | `ct_sign.cpp` | Yes |
| `ct::schnorr_sign` | `ct_sign.cpp` | Yes |

### Valgrind CT Taint Protocol

```
1. Mark secret key as UNDEFINED:  VALGRIND_MAKE_MEM_UNDEFINED(&secret, 32)
2. Run CT operation:              ct::ecdsa_sign(secret, msg, &sig)
3. Mark output as DEFINED:        VALGRIND_MAKE_MEM_DEFINED(&sig, 64)
4. Check: zero "Conditional jump depends on uninitialised value" = PASS
```

---

## Cross-Library Differential Testing

### vs bitcoin-core/libsecp256k1

- **Version:** v0.6.0 (FetchContent in CMake, `SECP256K1_BUILD_CROSS_TESTS=ON`)
- **Protocol:** 10 test suites, configurable multiplier (M=1 default, M=100 nightly)
- **Checks:** ~7,860 at M=1, ~1.3M at M=100
- **Fields compared:** Field arithmetic, scalar arithmetic, point operations, ECDSA sign/verify, Schnorr sign/verify, batch verification
- **Criterion:** Bit-exact match for all operations

### vs Fiat-Crypto Reference

- **Source:** Fiat-Crypto/Sage-generated reference implementations
- **Checks:** 647 reference vectors
- **Operations:** mul, sqr, inv, add, sub in both field and scalar domains

---

## How to Run

### Quick: CTest (all audit-labeled tests)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
ctest --test-dir build -L audit --output-on-failure
```

### Unified Audit Runner

```bash
# All sections
./build/audit/unified_audit_runner

# JSON + SARIF output
./build/audit/unified_audit_runner --json-only --sarif --report-dir ./reports

# Specific section
./build/audit/unified_audit_runner --section ct_analysis

# List available sections
./build/audit/unified_audit_runner --list-sections
```

### Full Orchestrator (A-M categories)

```bash
# Linux / macOS
./audit/run_full_audit.sh

# Windows (PowerShell)
.\audit\run_full_audit.ps1
```

### With Cross-Library + Protocol Tests

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DSECP256K1_BUILD_CROSS_TESTS=ON \
  -DSECP256K1_BUILD_PROTOCOL_TESTS=ON \
  -DSECP256K1_BUILD_FUZZ_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

### CT Valgrind Verification

```bash
cmake -S . -B build-ct -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DSECP256K1_CT_VALGRIND=1
cmake --build build-ct -j
valgrind --tool=memcheck --track-origins=yes ./build-ct/audit/ct_verif_formal
```

---

## Report Formats

### JSON (`audit_report.json`)

```json
{
  "timestamp": "2026-03-04T12:00:00Z",
  "platform": "x86-64 / Linux / GCC 14.2.0",
  "audit_verdict": "PASS",
  "sections": [
    {
      "id": "math_invariants",
      "title": "Mathematical Invariants",
      "modules": [
        {"name": "audit_field", "status": "PASS", "checks": 264484, "time_ms": 1234},
        {"name": "audit_scalar", "status": "PASS", "checks": 93847, "time_ms": 567}
      ]
    }
  ],
  "summary": {
    "total_modules": 49,
    "passed": 48,
    "failed": 0,
    "advisory": 1,
    "total_checks": 723000
  }
}
```

### SARIF v2.1.0 (`audit_report.sarif`)

Uploaded to GitHub Code Scanning for integration with the Security tab. Shows audit
results as security findings with severity levels.

### Text (`audit_report.txt`)

Human-readable summary with pass/fail status for each module.

---

## Documentation Index

| Document | Description |
|----------|------------|
| [AUDIT_GUIDE.md](../AUDIT_GUIDE.md) | Build/run instructions for all platforms, report format, verdict logic |
| [AUDIT_SCOPE.md](../AUDIT_SCOPE.md) | External audit engagement scope: in/out of scope, objectives, deliverables |
| [AUDIT_TEST_PLAN.md](../../audit/AUDIT_TEST_PLAN.md) | A-M category breakdown, test-to-evidence map, threat model traceability |
| [TEST_MATRIX.md](../TEST_MATRIX.md) | API function-to-test coverage map |
| [CROSS_PLATFORM_TEST_MATRIX.md](../CROSS_PLATFORM_TEST_MATRIX.md) | 22 CTest targets x 9 platforms |
| [CT_VERIFICATION.md](../CT_VERIFICATION.md) | CT architecture, what IS/ISN'T CT, dudect methodology |
| [DIFFERENTIAL_TESTING.md](../DIFFERENTIAL_TESTING.md) | Cross-library protocol, 10 suites, multiplier system |
| [SECURITY_CLAIMS.md](../SECURITY_CLAIMS.md) | FAST/CT semantic equivalence, nonce erasure, strict parsing |
| [THREAT_MODEL.md](../../THREAT_MODEL.md) | Threat model and security assumptions |

---

## Total Check Count

| Domain | Checks |
|--------|-------:|
| Field Fp invariants | 264,484 |
| Scalar Zn invariants | 93,847 |
| Point / group law | 116,312 |
| CT equivalence | 120,128 |
| Security hardening | 17,856 |
| Integration / protocol | 13,144 + 975 + 316 |
| Fuzzing | 15,423 + 580K + 73,959 |
| Fault injection | 610 |
| Carry propagation | 247 |
| Fiat-Crypto reference | 647 |
| Cross-platform KAT | 24 |
| Debug invariants | 372 |
| ABI gate | 12 |
| **Total (structured)** | **>1.2M** |

---

*Last updated: 2026-03-04*
*UltrafastSecp256k1 v3.68.0*
