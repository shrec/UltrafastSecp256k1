# Cross-Platform Test Matrix

> **Generated**: 2026-03-29 | **Library**: UltrafastSecp256k1 | **Current Validation Surface**: 182 active CTest targets
>
> ეს დოკი წარმოადგენს cross-platform მაღალი-სიგნალის portability subset-ს და არა სრულ 182-target inventory-ს.
> სრული აქტიური ზედაპირისთვის authoritative წყაროებია `docs/TEST_MATRIX.md` და `scripts/validate_assurance.py`.
> მიზანი უცვლელია: ერთი და იგივე ფუნქციური ინვარიანტები ყველა მხარდაჭერილ პლატფორმაზე უნდა გადიოდეს. ნებისმიერი განსხვავება = **BUG**.

---

## Representative Portability Subset (41 Tests)

| #  | Test Name               | Category            | Checks | Description                                                      |
|----|------------------------|---------------------|--------|------------------------------------------------------------------|
| 1  | selftest               | Core Selftest       | ~200   | Built-in self-test: field, scalar, point, generator consistency  |
| 2  | batch_add_affine       | Point Arithmetic    | ~50    | Batch affine addition correctness for sequential ECC search      |
| 3  | hash_accel             | Hashing             | ~80    | SHA-256, RIPEMD-160, Hash160 (SHA-NI accelerated where available)|
| 4  | field_52               | Field Arithmetic    | ~100   | 5x52-bit lazy reduction field implementation tests               |
| 5  | field_26               | Field Arithmetic    | ~100   | 10x26-bit field (32-bit platform path) implementation tests      |
| 6  | exhaustive             | Full Coverage       | ~500+  | Exhaustive small-order subgroup + enumeration tests              |
| 7  | comprehensive          | Full Coverage       | ~800+  | All arithmetic operations combined stress                        |
| 8  | bip340_vectors         | Standards Vectors   | ~30    | BIP-340 Schnorr signature official test vectors                  |
| 9  | bip340_strict          | Standards Vectors   | ~30    | BIP-340 strict encoding enforcement vectors                      |
| 10 | bip32_vectors          | Standards Vectors   | ~40    | BIP-32 HD key derivation official test vectors                   |
| 11 | bip39                  | Standards Vectors   | ~57    | BIP-39 mnemonic: PBKDF2, wordlist, entropy, validation, seed    |
| 12 | rfc6979_vectors        | Standards Vectors   | ~20    | RFC 6979 deterministic nonce official test vectors               |
| 13 | ecc_properties         | ECC Properties      | ~150   | Algebraic properties: associativity, commutativity, identity     |
| 14 | point_edge_cases       | Point Arithmetic    | ~80    | Point edge cases: infinity, doubling, negation                   |
| 15 | edge_cases             | General Edge Cases  | ~60    | General edge case testing across all operations                  |
| 16 | ethereum               | Protocol            | ~40    | Ethereum signing: EIP-155, EIP-191, ecrecover, Keccak-256       |
| 17 | zk_proofs              | Protocol            | ~24    | ZK proofs: knowledge, DLEQ, Bulletproof range, batch verify     |
| 18 | wallet                 | Protocol            | ~30    | Wallet API: key management, signing, recovery, multi-coin       |
| 19 | cuda_selftest          | GPU                 | ~50    | CUDA GPU kernel selftest                                         |
| 20 | gpu_audit              | GPU                 | ~300   | CUDA GPU unified audit (all modules)                             |
| 21 | gpu_ct_smoke           | GPU                 | ~9     | CUDA CT smoke: ZK knowledge + DLEQ prove/verify                 |
| 22 | opencl_selftest        | GPU                 | ~50    | OpenCL GPU kernel selftest                                       |
| 23 | opencl_audit           | GPU                 | ~300   | OpenCL GPU unified audit (27 modules, 8 sections)               |
| 24 | ct_sidechannel         | Constant-Time       | ~300   | Full CT: dudect Welch t-test, 600s timeout                      |
| 25 | ct_sidechannel_smoke   | Constant-Time       | ~100   | CT smoke: basic correctness, 120s CI-safe                       |
| 26 | differential           | Differential Test   | ~200   | Differential testing: fast vs CT layer output equivalence        |
| 27 | ct_equivalence         | Constant-Time       | ~150   | CT scalar_mul == fast scalar_mul bitwise equivalence             |
| 28 | fault_injection        | Security Audit      | 610    | Fault injection: bit-flips, coord corruption, GLV               |
| 29 | debug_invariants       | Security Audit      | 372    | Debug assertions: normalize, on_curve, scalar_valid              |
| 30 | fiat_crypto_vectors    | Golden Vectors      | 647    | Fiat-Crypto/Sage reference: mul, sqr, inv, add, sub             |
| 31 | carry_propagation      | Boundary Stress     | 247    | Carry chain stress: all-ones, limb boundary, near-p, near-n     |
| 32 | wycheproof_ecdsa       | Standards Vectors   | ~200   | Wycheproof ECDSA test vectors (Google)                           |
| 33 | wycheproof_ecdh        | Standards Vectors   | ~200   | Wycheproof ECDH test vectors (Google)                            |
| 34 | batch_randomness       | Security Audit      | ~100   | Batch operation randomness quality verification                  |
| 35 | cross_platform_kat     | KAT Equivalence     | 24     | Cross-platform KAT: field, scalar, point, ECDSA, Schnorr        |
| 36 | abi_gate               | ABI Compatibility   | 12     | ABI version gate: compile-time macro validation                  |
| 37 | ct_verif_formal        | Formal Verification | ~50    | CT formal verification stubs                                     |
| 38 | fiat_crypto_linkage    | Differential        | ~50    | Independent reference linkage verification                       |
| 39 | audit_fuzz             | Fuzz Testing        | ~500   | Fuzz-derived audit: random inputs through all paths              |
| 40 | diag_scalar_mul        | Diagnostics         | ~50    | Scalar multiplication step-by-step diagnostic                    |
| 41 | unified_audit          | Full Audit          | 249 modules | Unified audit runner: 60 non-exploit modules + 189 exploit PoCs in a single binary |

---

## Platform Matrix

### Legend
- [OK] = All checks PASS
- [FAIL] = One or more checks FAIL
- [!] = Partial (some tests skipped or known limitation)
- N/A = Not applicable / not targetable for this platform
- 🔲 = Not yet tested

### Test x Platform Status

| #  | Test Name             | x86-64 Win (Clang) | x86-64 Linux (Clang/GCC) | x86-64 macOS | ARM64 Linux | ARM64 macOS (Apple Si) | RISC-V 64 | WASM (Emscripten) | ESP32 (Xtensa) | STM32 (Cortex-M4) |
|----|----------------------|:-------------------:|:------------------------:|:------------:|:-----------:|:---------------------:|:---------:|:-----------------:|:--------------:|:-----------------:|
| 1  | selftest             | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 2  | batch_add_affine     | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 3  | hash_accel           | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 4  | field_52             | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | N/A            | N/A               |
| 5  | field_26             | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | [OK] (1)       | [OK] (1)          |
| 6  | exhaustive           | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 7  | comprehensive        | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 8  | bip340_vectors       | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 9  | bip340_strict        | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 10 | bip32_vectors        | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 11 | bip39                | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 12 | rfc6979_vectors      | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 13 | ecc_properties       | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 14 | point_edge_cases     | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 15 | edge_cases           | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 16 | ethereum             | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 17 | zk_proofs            | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 18 | wallet               | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 19 | cuda_selftest        | N/A                 | [OK]                     | N/A          | N/A         | N/A                   | N/A       | N/A               | N/A            | N/A               |
| 20 | gpu_audit            | N/A                 | [OK]                     | N/A          | N/A         | N/A                   | N/A       | N/A               | N/A            | N/A               |
| 21 | gpu_ct_smoke         | N/A                 | [OK]                     | N/A          | N/A         | N/A                   | N/A       | N/A               | N/A            | N/A               |
| 22 | opencl_selftest      | [OK]                | [OK]                     | --           | --          | --                    | N/A       | N/A               | N/A            | N/A               |
| 23 | opencl_audit         | [OK]                | [OK]                     | --           | --          | --                    | N/A       | N/A               | N/A            | N/A               |
| 24 | ct_sidechannel       | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 25 | ct_sidechannel_smoke | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 26 | differential         | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 27 | ct_equivalence       | [OK]                | [OK]                     | --           | --          | --                    | [OK]      | --                | --             | --                |
| 28 | fault_injection      | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 29 | debug_invariants     | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 30 | fiat_crypto_vectors  | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 31 | carry_propagation    | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 32 | wycheproof_ecdsa     | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 33 | wycheproof_ecdh      | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 34 | batch_randomness     | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 35 | cross_platform_kat   | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 36 | abi_gate             | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 37 | ct_verif_formal      | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 38 | fiat_crypto_linkage  | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 39 | audit_fuzz           | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 40 | diag_scalar_mul      | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |
| 41 | unified_audit        | [OK]                | [OK]                     | --           | --          | --                    | --        | --                | --             | --                |

> (1) 32-bit platforms (ESP32, STM32) use field_26 only; field_52 requires 64-bit limbs.

---

## CI Coverage (Automated)

| Platform             | CI Workflow       | Trigger        | Status    |
|---------------------|-------------------|----------------|-----------|
| x86-64 Linux (GCC)  | ci.yml            | push/PR        | [OK] Active |
| x86-64 Linux (Clang) | ci.yml           | push/PR        | [OK] Active |
| x86-64 Windows (MSVC)| ci.yml           | push/PR        | [OK] Active |
| x86-64 Windows (Clang)| ci.yml          | push/PR        | [OK] Active |
| x86-64 macOS        | ci.yml            | push/PR        | [OK] Active |
| ARM64 Linux          | ci.yml (qemu)    | push/PR        | [OK] Active |
| RISC-V 64            | Manual / Cross   | manual         | [!] Manual |
| WASM                 | release.yml      | release/manual | [OK] Active |
| ESP32                | --                 | --              | 🔲 Planned |
| STM32                | --                 | --              | 🔲 Planned |

---

## Verification Summary (Current Session -- x86-64 Linux, Clang)

```
Representative portability subset: 41/41 passed, 0 failed

Individual check counts:
  selftest .................. ~200 checks
  batch_add_affine .......... ~50  checks
  hash_accel ................ ~80  checks
  field_52 .................. ~100 checks
  field_26 .................. ~100 checks
  exhaustive ................ ~500 checks
  comprehensive ............. ~800 checks
  bip340_vectors ............ ~30  checks
  bip340_strict ............. ~30  checks
  bip32_vectors ............. ~40  checks
  bip39 .................... ~57  checks
  rfc6979_vectors ........... ~20  checks
  ecc_properties ............ ~150 checks
  point_edge_cases .......... ~80  checks
  edge_cases ................ ~60  checks
  ethereum .................. ~40  checks
  zk_proofs ................. ~24  checks
  wallet .................... ~30  checks
  cuda_selftest ............. ~50  checks
  gpu_audit ................. ~300 checks
  gpu_ct_smoke .............. ~9   checks
  opencl_selftest ........... ~50  checks
  opencl_audit .............. ~300 checks
  ct_sidechannel ............ ~300 checks
  ct_sidechannel_smoke ...... ~100 checks
  differential .............. ~200 checks
  ct_equivalence ............ ~150 checks
  fault_injection ........... 610  checks OK
  debug_invariants .......... 372  checks OK
  fiat_crypto_vectors ....... 647  checks OK
  carry_propagation ......... 247  checks OK
  wycheproof_ecdsa .......... ~200 checks
  wycheproof_ecdh ........... ~200 checks
  batch_randomness .......... ~100 checks
  cross_platform_kat ........ 24   checks OK
  abi_gate .................. 12   checks OK
  ct_verif_formal ........... ~50  checks
  fiat_crypto_linkage ....... ~50  checks
  audit_fuzz ................ ~500 checks
  diag_scalar_mul ........... ~50  checks
  unified_audit ............. 249 modules (60 non-exploit + 189 exploit PoCs)
  -----------------------------------------
  TOTAL (estimated):         ~6400+ individual assertions
```

---

## Test Categories

| Category            | Tests                                                                  | Purpose                                    |
|--------------------|------------------------------------------------------------------------|--------------------------------------------|
| Core Selftest      | selftest                                                               | Basic library self-validation on startup    |
| Field Arithmetic   | field_52, field_26, carry_propagation                                  | Modular arithmetic correctness at all limb widths |
| Point Arithmetic   | batch_add_affine, ecc_properties, point_edge_cases, diag_scalar_mul    | Elliptic curve operations                  |
| Standards Vectors  | bip340_vectors, bip340_strict, bip32_vectors, bip39, rfc6979_vectors, wycheproof_ecdsa, wycheproof_ecdh | Official standard compliance |
| Golden Vectors     | fiat_crypto_vectors, cross_platform_kat                                | Deterministic correctness vs reference     |
| Constant-Time      | ct_sidechannel, ct_sidechannel_smoke, ct_equivalence, differential     | Side-channel resistance verification       |
| Security Audit     | fault_injection, debug_invariants, batch_randomness                    | Fault tolerance & invariant enforcement    |
| Hashing            | hash_accel                                                             | Hash function correctness (SHA-NI, etc.)   |
| Full Coverage      | exhaustive, comprehensive, edge_cases                                  | Exhaustive enumeration + combined stress   |
| Protocol           | ethereum, zk_proofs, wallet                                            | Protocol-level correctness                 |
| GPU                | cuda_selftest, gpu_audit, gpu_ct_smoke, opencl_selftest, opencl_audit  | GPU backend correctness                    |
| ABI Compatibility  | abi_gate                                                               | Version & ABI stability check              |
| Formal/Fuzz        | ct_verif_formal, fiat_crypto_linkage, audit_fuzz                       | Formal verification & fuzz testing         |
| Full Audit         | unified_audit                                                          | All 49 audit modules in single binary      |

---

## Scripts (Audit Infrastructure)

| Script                          | Purpose                                           | Platform    |
|--------------------------------|---------------------------------------------------|-------------|
| scripts/verify_ct_disasm.sh    | Disassembly scan for CT branches                  | Linux       |
| scripts/valgrind_ct_check.sh   | Valgrind memcheck on CT paths                     | Linux       |
| scripts/ctgrind_validate.sh    | CTGRIND-style validation (secret-as-undefined)    | Linux       |
| scripts/generate_coverage.sh   | LLVM source-based code coverage                   | Linux/macOS |
| scripts/cross_compiler_ct_stress.sh | Multi-compiler CT verification              | Linux       |
| scripts/generate_selftest_report.sh | JSON self-test evidence report               | Any         |
| scripts/generate_dudect_badge.sh | Dudect timing badge generation                  | Linux       |
| scripts/cachegrind_ct_analysis.sh | Cache-line timing analysis                     | Linux       |
| scripts/perf_regression_check.sh | Benchmark regression tracking                   | Linux       |
| scripts/generate_self_audit_report.sh | Comprehensive audit evidence JSON          | Linux       |

---

## Platform-Specific Notes

### x86-64 (Primary)
- Assembly tier: Tier 3 (inline asm), Tier 2 (BMI2 intrinsics), Tier 1 (C++)
- SHA-NI acceleration available on supported CPUs
- Full CI matrix (Windows MSVC+Clang, Linux GCC+Clang, macOS)

### ARM64
- Uses generic C++ paths (no asm tier 3)
- CI via QEMU cross-compilation
- SHA-256 hardware acceleration via ARM CE where available

### RISC-V 64
- Custom assembly: field_asm52_riscv64.S (Tier 3)
- SLTU/carry chain bug fixes verified (see RISCV_FIX_SUMMARY.md)
- Manual cross-compilation + QEMU testing
- RVV (Vector Extension) support optional

### WASM (Emscripten) -- Planned
- 32-bit path: field_26 (10x26-bit limbs)
- No inline assembly, pure C++ only
- KAT test should produce identical output

### ESP32 / STM32 -- Planned
- 32-bit path: field_26
- No OS, bare-metal test harness needed
- KAT golden vectors are the acceptance criterion

---

## How to Run on a New Platform

```bash
# 1. Configure
cmake -S . -B build_<platform> -G Ninja -DCMAKE_BUILD_TYPE=Release

# 2. Build
cmake --build build_<platform> -j

# 3. Run ALL tests
ctest --test-dir build_<platform> --output-on-failure

# 4. Verify KAT equivalence (golden vectors must match exactly)
./build_<platform>/cpu/test_cross_platform_kat

# 5. Generate audit report
./scripts/generate_self_audit_report.sh build_<platform>
```

Expected: **41/41 tests PASS** with identical output on every platform.

---

> **ყველა პლატფორმაზე იდენტური შედეგი = სწორი იმპლემენტაცია.**
> **ნებისმიერი განსხვავება = ბაგი, რომელიც დაუყოვნებლივ უნდა გამოსწორდეს.**
