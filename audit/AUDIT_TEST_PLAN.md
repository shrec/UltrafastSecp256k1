# Audit Test Plan -- UltrafastSecp256k1 v3.14.0

> **Historical Report -- Snapshot from v3.14.0**
> This plan describes test categories as of v3.14.0. Since then:
> ct-verif and valgrind-ct are active and blocking in CI,
> GPU audit runners (OpenCL, Metal) exist, and additional protocol
> tests (FROST KAT, adversarial protocol) have been added.
> See `docs/TEST_MATRIX.md` for the current test map.

> **Single source of truth** for what the audit tests, how it tests, and where evidence lives.

---

## Quick Start

```bash
# === Linux / macOS ===
bash audit/run_full_audit.sh

# === Windows (PowerShell) ===
pwsh -NoProfile -File audit/run_full_audit.ps1

# === Only unified C++ runner (cross-platform) ===
cmake --build build-audit --target unified_audit_runner
./build-audit/audit/unified_audit_runner
```

Output: `audit-output-<timestamp>/audit_report.md` + `artifacts/`

---

## Category -> Test -> Evidence Map

### A. Environment & Build Integrity

| # | Test | Implementation | Evidence Artifact |
|---|------|---------------|-------------------|
| A.1 | Toolchain fingerprint | `run_full_audit` collects compiler/cmake/ninja/git versions | `artifacts/toolchain_fingerprint.json` |
| A.2 | Reproducible build check | `scripts/verify_reproducible_build.sh` (2 builds, SHA compare) | `artifacts/reproducible_build.log` |
| A.3 | Dependency / zero-deps proof | `ldd`/`dumpbin` scan of binaries | `artifacts/dependency_scan.txt` |
| A.4 | Artifact manifest | SHA256 of all built binaries/libs | `artifacts/SHA256SUMS.txt` |

### B. Packaging & Supply Chain

| # | Test | Implementation | Evidence Artifact |
|---|------|---------------|-------------------|
| B.1 | SBOM generation | `scripts/generate_sbom.sh` (CycloneDX 1.6) | `artifacts/sbom.cdx.json` |
| B.2 | Provenance / SLSA metadata | `run_full_audit` (builder + source + build info) | `artifacts/provenance.json` |
| B.3 | Signature / checksum validation | SHA256SUMS.txt for release assets | `artifacts/SHA256SUMS.txt` |
| B.4 | Install/uninstall tests | CPack (DEB/RPM/ZIP/NuGet), binding package managers | manual / CI |

### C. Static Analysis

| # | Test | Implementation | Evidence Artifact |
|---|------|---------------|-------------------|
| C.1 | clang-tidy | `run_full_audit` + `.clang-tidy` config (30+ checks) | `artifacts/static_analysis/clang_tidy.log` |
| C.2 | cppcheck | `run_full_audit` (secondary signal) | `artifacts/static_analysis/cppcheck.log` |
| C.3 | CodeQL | GitHub Actions CI (`codeql-analysis.yml`) | GitHub Security tab |
| C.4 | SonarCloud | `sonar-project.properties` + CI | SonarCloud dashboard |
| C.5 | Include-what-you-use | Optional, manual | -- |
| C.6 | Dangerous patterns scan | grep-based scan for hot-path violations | `artifacts/static_analysis/dangerous_patterns.log` |

### D. Sanitizers (Memory/UB/Threads)

| # | Test | Implementation | Evidence Artifact |
|---|------|---------------|-------------------|
| D.1 | ASan + UBSan | `run_full_audit` builds with `-fsanitize=address,undefined` | `artifacts/sanitizers/asan_ubsan.log` |
| D.2 | MSan | Linux-only, requires instrumented libc++; `run_full_audit.sh` (future) | `artifacts/sanitizers/msan.log` |
| D.3 | TSan | Separate build; library primarily single-threaded | `artifacts/sanitizers/tsan.log` |
| D.4 | LeakSanitizer | Included with ASan (`detect_leaks=1`) | `artifacts/sanitizers/asan_ubsan.log` |
| D.5 | Valgrind memcheck | `scripts/valgrind_ct_check.sh` / `run_full_audit.sh` | `artifacts/sanitizers/valgrind.log` |

### E. Unit Tests (KAT -- Known Answer Tests)

| # | Test | Implementation (unified runner module) | CTest target |
|---|------|----------------------------------------|-------------|
| E.1a | Field/scalar/point KAT | `audit_field`, `audit_scalar`, `audit_point`, `mul`, `arith_correct` | `debug_invariants`, `carry_propagation` |
| E.1b | ECDSA RFC6979 vectors | `rfc6979_vectors` | `fiat_crypto_vectors` |
| E.1c | Schnorr BIP-340 vectors | `bip340_vectors` | `cross_platform_kat` |
| E.1d | BIP-32 vectors TV1-TV5 | `bip32_vectors` | `cross_platform_kat` |
| E.1e | Address encoding vectors | `coins` | -- |
| E.2 | Serialization roundtrips | `comprehensive`, `ecdsa_schnorr` | -- |
| E.3 | Error-path tests | `audit_fuzz`, `fault_injection`, `fuzz_parsers` | `audit_fuzz`, `fault_injection` |
| E.4 | Boundary tests (0, 1, n-1, p, etc.) | `exhaustive`, `ecc_properties`, `audit_field`, `audit_scalar` | `carry_propagation` |

### F. Property-Based / Algebraic Invariants

| # | Test | Implementation (unified runner module) |
|---|------|----------------------------------------|
| F.1 | Group law: P+O=P, P+(-P)=O, commutativity, associativity | `ecc_properties`, `audit_point`, `exhaustive` |
| F.2 | Scalar/field ring: distributive, inverse | `audit_field`, `audit_scalar`, `arith_correct` |
| F.3 | GLV decomposition correctness | `audit_scalar` (GLV edge cases) |
| F.4 | Batch inversion correctness | `audit_field` (batch inverse sweep) |
| F.5 | Jacobian<->Affine roundtrip | `audit_point`, `batch_add` |
| F.6 | FAST==CT equivalence | `ct_equivalence`, `diag_scalar_mul` |

> **Seed**: All property tests use deterministic seed. Seed is printed in unified runner output and recorded in `audit_report.json`.

### G. Differential Testing

| # | Test | Implementation | CTest target |
|---|------|---------------|-------------|
| G.1 | Internal differential (5x52 vs 10x26 vs 4x64) | `field_52`, `field_26`, `differential` | `differential` |
| G.2 | Cross-library vs bitcoin-core/libsecp256k1 | `test_cross_libsecp256k1.cpp` | `cross_libsecp256k1` (requires `-DSECP256K1_BUILD_CROSS_TESTS=ON`) |
| G.3 | Fiat-Crypto reference vectors | `fiat_crypto` | `fiat_crypto_vectors` |
| G.4 | Cross-platform KAT | `cross_platform_kat` | `cross_platform_kat` |

### H. Fuzzing (Robustness / Parser Safety)

| # | Test | Implementation | CTest target |
|---|------|---------------|-------------|
| H.1a | Pubkey parse fuzz | `fuzz_parsers` | `fuzz_parsers` (requires `-DSECP256K1_BUILD_FUZZ_TESTS=ON`) |
| H.1b | DER parser fuzz | `fuzz_parsers` | `fuzz_parsers` |
| H.1c | Address/BIP32/FFI boundary fuzz | `fuzz_addr_bip32` | `fuzz_address_bip32_ffi` |
| H.1d | ufsecp ABI boundary | `fuzz_addr_bip32` | `fuzz_address_bip32_ffi` |
| H.2 | Adversarial fuzz (malform/edge) | `audit_fuzz` | `audit_fuzz` |
| H.3 | Fault injection simulation | `fault_injection` | `fault_injection` |
| H.4 | Corpus: `audit/corpus/` | seed corpus for deterministic fuzz | -- |

### I. Constant-Time & Side-Channel

| # | Test | Implementation | Evidence Artifact |
|---|------|---------------|-------------------|
| I.1 | CT branch scan (disassembly) | `scripts/verify_ct_disasm.sh` | `artifacts/disasm/disasm_branch_scan.json` |
| I.2a | dudect: scalar_mul | `ct_sidechannel` (smoke: `|t| < 4.5`) | `artifacts/ctest/audit_report.json` |
| I.2b | dudect: field_inv, scalar_inv | `ct_sidechannel` | -- |
| I.2c | dudect: ECDSA sign | `ct_sidechannel` | -- |
| I.2d | dudect: Schnorr sign | `ct_sidechannel` | -- |
| I.2e | dudect: cswap/cmov primitives | `audit_ct` | -- |
| I.3 | Valgrind CT (uninit-as-secret) | `scripts/valgrind_ct_check.sh` | `artifacts/sanitizers/valgrind.log` |
| I.4 | CT contract: `audit_ct` (masks/cmov deep) | `audit_ct`, `ct`, `ct_equivalence` | `audit_report.json` |
| I.5 | FAST==CT equivalence proof | `ct_equivalence`, `diag_scalar_mul` | `audit_report.json` |

### J. ABI / API Stability & Safety

| # | Test | Implementation | CTest target |
|---|------|---------------|-------------|
| J.1 | ABI symbol check | `run_full_audit` (nm/dumpbin scan) | -- |
| J.2 | ABI version gate | `test_abi_gate.cpp` | `abi_gate` |
| J.3 | Calling convention (null/misaligned) | `audit_security` (null/bitflip/nonce) | -- |
| J.4 | Error model compliance | `audit_fuzz`, `fault_injection` | -- |

### K. Bindings & FFI Parity

| # | Test | Implementation | Evidence Artifact |
|---|------|---------------|-------------------|
| K.1 | Parity matrix (all ufsecp.h functions per binding) | `run_full_audit` scans `bindings/` | `artifacts/bindings/parity_matrix.json` |
| K.2 | Binding smoke tests | Per-language test suites in `bindings/<lang>/` | -- |
| K.3 | Memory ownership tests | Binding-specific tests | -- |
| K.4 | Package install tests | `pip`/`npm`/`nuget`/... install -> run sample | manual / CI |

### L. Performance Regression

| # | Test | Implementation |
|---|------|---------------|
| L.1 | Microbench stability | `audit_perf` (sign/verify roundtrip), benchmark targets |
| L.2 | CPU features dispatch | Platform detection in CMakeLists.txt |
| L.3 | GPU kernel sanity | Separate GPU audit (if `SECP256K1_BUILD_CUDA=ON`) |

### M. Documentation & Claims Consistency

| # | Test | Implementation |
|---|------|---------------|
| M.1 | Required docs exist | `run_full_audit` checks README/CHANGELOG/SECURITY/LICENSE/THREAT_MODEL/CONTRIBUTING/VERSION |
| M.2 | Version consistency | VERSION.txt matches CHANGELOG.md |
| M.3 | THREAT_MODEL.md present and current | `run_full_audit` |
| M.4 | AUDIT_GUIDE.md present | `run_full_audit` |

### N. New ABI Surface Edge Cases (v3.22+)

> **Added**: Gap analysis found 26 `ufsecp_*` functions reachable from the public
> C ABI with no dedicated edge-case coverage. Section N closes that gap.
> All tests live in `audit/test_adversarial_protocol.cpp`, functions `test_h1_*`
> through `test_h12_*`, wired into `test_adversarial_protocol_run()`.

**Mandatory edge-case rule** — every ABI function MUST be covered by all four of
the following before an audit release:

1. NULL rejection for every pointer parameter (`UFSECP_ERR_NULL_ARG`)
2. Zero-count / zero-length / zero-key rejection where the contract requires it
3. Invalid-content rejection (bad prefix, off-curve point, truncated ciphertext,
   wrong tag, OOB index, etc.) returning a non-`UFSECP_OK` code
4. A success smoke test demonstrating at least one valid round-trip or output

| # | Test function | ABI functions covered | Checks |
|---|---------------|-----------------------|--------|
| N.1 | `test_h1_ctx_size()` | `ufsecp_ctx_size` | positive result |
| N.2 | `test_h2_aead()` | `ufsecp_aead_chacha20_encrypt`, `ufsecp_aead_chacha20_decrypt` | NULL guards, bad-tag auth failure, wrong-nonce rejection, zero-length roundtrip |
| N.3 | `test_h3_ecies()` | `ufsecp_ecies_encrypt`, `ufsecp_ecies_decrypt` | NULL guards, off-curve pubkey, tampered envelope, length check |
| N.4 | `test_h4_ellswift()` | `ufsecp_ellswift_create`, `ufsecp_ellswift_xdh` | NULL guards, zero privkey, symmetric shared secret; guarded by `SECP256K1_BIP324` |
| N.5 | `test_h5_eth_edge()` | `ufsecp_eth_address_checksummed`, `ufsecp_eth_personal_hash` | NULL guards, undersized buffer; guarded by `SECP256K1_BUILD_ETHEREUM` |
| N.6 | `test_h6_pedersen_switch()` | `ufsecp_pedersen_switch_commit` | NULL guards, valid prefix byte (0x02/0x03) |
| N.7 | `test_h7_schnorr_adaptor_extract()` | `ufsecp_schnorr_adaptor_extract` | NULL guards, zero inputs rejected |
| N.8 | `test_h8_batch_sign()` | `ufsecp_ecdsa_sign_batch`, `ufsecp_schnorr_sign_batch` | NULL ctx/msgs/keys/output, count=0 rejection |
| N.9 | `test_h9_bip143()` | `ufsecp_bip143_sighash`, `ufsecp_bip143_p2wpkh_script_code` | NULL guards, OP_DUP OP_HASH160 PUSH20 format |
| N.10 | `test_h10_bip144()` | `ufsecp_bip144_txid`, `ufsecp_bip144_wtxid`, `ufsecp_bip144_witness_commitment` | NULL guards, determinism |
| N.11 | `test_h11_segwit()` | `ufsecp_is_witness_program`, `ufsecp_parse_witness_program`, `ufsecp_p2wpkh_spk`, `ufsecp_p2wsh_spk`, `ufsecp_p2tr_spk`, `ufsecp_witness_script_hash` | NULL guards, format correctness (OP_0/OP_1 + push), non-witness rejection |
| N.12 | `test_h12_taproot_sighash()` | `ufsecp_taproot_keypath_sighash`, `ufsecp_tapscript_sighash` | NULL guards, count=0, OOB input_index, NULL tapleaf_hash, determinism |

---

## Unified Audit Runner -- 8-Section Internal Mapping

The C++ `unified_audit_runner` binary covers **E, F, G(internal), H(deterministic), I(dudect+CT), J(ABI gate), L(smoke)** in a single executable.

| Section # | unified_audit_runner section | Modules |
|-----------|------------------------------|---------|
| 1 | `math_invariants` | audit_field, audit_scalar, audit_point, mul, arith_correct, scalar_mul, exhaustive, comprehensive, ecc_properties, batch_add, carry_propagation, field_52, field_26 |
| 2 | `ct_analysis` | audit_ct, ct, ct_equivalence, ct_sidechannel (dudect smoke), diag_scalar_mul |
| 3 | `differential` | differential, fiat_crypto, cross_platform_kat |
| 4 | `standard_vectors` | bip340_vectors, bip32_vectors, rfc6979_vectors, frost_kat |
| 5 | `fuzzing` | audit_fuzz, fuzz_parsers, fuzz_addr_bip32, fault_injection |
| 6 | `protocol_security` | ecdsa_schnorr, bip32, musig2, ecdh_recovery, v4_features, coins, musig2_frost, musig2_frost_adv, audit_integration |
| 7 | `memory_safety` | audit_security, debug_invariants, abi_gate |
| 8 | `performance` | hash_accel, simd_batch, multiscalar, audit_perf |

---

## Threat Model -> Test Traceability

| THREAT_MODEL.md Attack | Risk | Tests Covering It | Evidence Location |
|------------------------|------|-------------------|-------------------|
| A1: Timing Side Channels | HIGH | I.1 (disasm), I.2 (dudect), I.4 (audit_ct), I.5 (CT==FAST), F.6 | `artifacts/disasm/`, `audit_report.json` (ct_analysis) |
| A2: Nonce Attacks | CRITICAL | E.1b (RFC6979), E.1c (BIP-340), F.6 (CT equivalence) | `audit_report.json` (standard_vectors) |
| A3: Arithmetic Errors | CRITICAL | E.1a, E.4, F.1-F.5, G.1-G.4 | `audit_report.json` (math_invariants, differential) |
| A4: Memory Safety | CRITICAL | D.1-D.5, H.1-H.4, J.3 | `artifacts/sanitizers/`, `audit_report.json` (fuzzing) |
| A5: Supply Chain | HIGH | A.3, B.1-B.3, A.4 | `artifacts/sbom.cdx.json`, `artifacts/SHA256SUMS.txt` |
| A6: GPU-Specific | HIGH | Separate GPU audit | -- |

### Not Covered by Automated Tests

| Gap | Reason | Mitigation |
|-----|--------|------------|
| Physical power analysis / EM | Requires lab equipment | Code review + CT layer design |
| Deterministic CT verification (`ct-verif`) | Integrated via workflow + blocking CI gate | `ct-verif` LLVM + dudect + disasm scan + Valgrind CT |
| Quantum adversary | secp256k1 is not post-quantum | Document as known limitation |
| OS-level memory disclosure | Caller responsibility | SECURITY.md guidance |

---

## Artifact Tree

```
audit-output-YYYYMMDD-HHMMSS/
+-- audit_report.md                          # სრული აუდიტის რეპორტი
+-- artifacts/
|   +-- SHA256SUMS.txt                       # ყველა ბინარის ჰეშები
|   +-- toolchain_fingerprint.json           # კომპილატორი/CMake/OS ინფო
|   +-- provenance.json                      # SLSA-style build provenance
|   +-- dependency_scan.txt                  # ldd/dumpbin output
|   +-- sbom.cdx.json                        # CycloneDX SBOM
|   +-- static_analysis/
|   |   +-- clang_tidy.log
|   |   +-- cppcheck.log
|   |   +-- dangerous_patterns.log
|   +-- sanitizers/
|   |   +-- asan_ubsan.log
|   |   +-- valgrind.log
|   |   +-- tsan.log
|   +-- ctest/
|   |   +-- unified_runner_output.txt        # Console output
|   |   +-- audit_report.json                # Structured JSON (8 sections)
|   |   +-- audit_report.txt                 # Human-readable text
|   |   +-- results.json                     # CTest summary
|   |   +-- ctest_output.txt
|   +-- disasm/
|   |   +-- disasm_branch_scan.json          # CT function branch scan
|   |   +-- disasm_branch_scan.txt
|   +-- bindings/
|   |   +-- parity_matrix.json
|   +-- benchmark/
|   |   +-- benchmark_output.txt
|   +-- fuzz/
|       +-- summary.json
```

---

## Build Configurations Required

| Configuration | Purpose | CMake Flags |
|---------------|---------|-------------|
| Release (primary) | Main audit run | `-DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_TESTS=ON -DSECP256K1_BUILD_FUZZ_TESTS=ON -DSECP256K1_BUILD_PROTOCOL_TESTS=ON` |
| ASan/UBSan | Memory safety | `-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-fsanitize=address,undefined` |
| Cross-lib | Differential vs libsecp256k1 | `-DSECP256K1_BUILD_CROSS_TESTS=ON` |
| Debug (no ASM) | Valgrind CT | `-DCMAKE_BUILD_TYPE=Debug -DSECP256K1_USE_ASM=OFF` |

---

*UltrafastSecp256k1 v3.14.0 -- Audit Test Plan*
