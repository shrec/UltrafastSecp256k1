# Audit Scope Document

**UltrafastSecp256k1** -- External Security Audit Engagement Scope

Version: 1.0  
Date: 2026-02-24  
Status: Pre-engagement  
Contact: payysoon@gmail.com

---

## 1. Executive Summary

UltrafastSecp256k1 is a high-performance elliptic curve cryptography library implementing secp256k1 operations for ECDSA (RFC 6979), Schnorr (BIP-340), MuSig2, FROST threshold signatures, Pedersen commitments, adaptor signatures, BIP-32 HD derivation, and 27-coin address generation. The library supports 13+ platforms (x86-64, ARM64, RISC-V, WASM, CUDA, OpenCL, Metal, ROCm/HIP, embedded).

This document defines the scope and methodology of the internal CAAS audit system.

---

## 2. Audit Objectives

| # | Objective | Priority |
|---|-----------|----------|
| A1 | Verify correctness of field/scalar/point arithmetic against secp256k1 specification | Critical |
| A2 | Validate ECDSA (RFC 6979) and Schnorr (BIP-340) implementations | Critical |
| A3 | Assess constant-time (`ct::`) layer for timing side-channel resistance | Critical |
| A4 | Review MuSig2 / FROST protocol correctness and rogue-key resistance | High |
| A5 | Validate BIP-32 HD derivation and key generation paths | High |
| A6 | Identify memory safety issues (overflows, use-after-free, uninitialized reads) | High |
| A7 | Review C ABI (`ufsecp`) boundary for NULL handling, error propagation, thread safety | Medium |
| A8 | Assess GPU kernel correctness (CUDA, OpenCL) for public-data operations | Medium |
| A9 | Review build system for reproducibility and supply-chain risks | Low |

---

## 3. In-Scope Components

### 3.1 Core Arithmetic (CRITICAL)

| File | Component | Lines (approx) |
|------|-----------|----------------|
| `src/cpu/src/field.cpp` | Field arithmetic: add, sub, mul, square, inv, sqrt, normalize | ~600 |
| `src/cpu/include/secp256k1/field.hpp` | FieldElement: from_limbs, from_bytes, to_bytes, comparison | ~200 |
| `src/cpu/include/secp256k1/field_branchless.hpp` | Branchless cmov (field_select) | ~50 |
| `src/cpu/src/scalar.cpp` | Scalar arithmetic mod n: add, sub, mul, inv, negate | ~500 |
| `src/cpu/include/secp256k1/scalar.hpp` | Scalar class, from_limbs, from_bytes | ~200 |
| `src/cpu/src/point.cpp` | Point operations: add, double, scalar_mul, to_affine | ~800 |
| `src/cpu/include/secp256k1/point.hpp` | Point types (Affine, Jacobian), generator | ~200 |
| `src/cpu/src/glv.cpp` | GLV endomorphism decomposition | ~300 |

### 3.2 Signature Schemes (CRITICAL)

| File | Component |
|------|-----------|
| `src/cpu/src/ecdsa.cpp` | ECDSA sign/verify (RFC 6979 nonce) |
| `src/cpu/include/secp256k1/ecdsa.hpp` | ECDSA interface |
| `src/cpu/src/schnorr.cpp` | BIP-340 Schnorr sign/verify |
| `src/cpu/include/secp256k1/schnorr.hpp` | Schnorr interface |

### 3.3 Constant-Time Layer (CRITICAL)

| File | Component |
|------|-----------|
| `src/cpu/src/ct/ct_scalar_mul.cpp` | CT scalar multiplication (fixed-window) |
| `src/cpu/include/secp256k1/ct/ct_ops.hpp` | CT operation interfaces |
| `src/cpu/src/ct/ct_field.cpp` | CT field operations |
| `src/cpu/include/secp256k1/ct/ct_sign.hpp` | CT sign wrappers |

### 3.4 Multi-Party Protocols (HIGH)

| File | Component |
|------|-----------|
| `src/cpu/src/musig2.cpp` | MuSig2 key aggregation, nonce generation, partial sign, aggregate |
| `src/cpu/include/secp256k1/musig2.hpp` | MuSig2 types and interfaces |
| `src/cpu/src/frost.cpp` | FROST DKG (Feldman VSS), partial sign, aggregate |
| `src/cpu/include/secp256k1/frost.hpp` | FROST types and interfaces |

### 3.5 Key Derivation & Address Generation (HIGH)

| File | Component |
|------|-----------|
| `src/cpu/src/bip32.cpp` | BIP-32 master key derivation, child derivation, path parsing |
| `src/cpu/src/address.cpp` | 27-coin address generation (Base58Check, Bech32, Bech32m) |
| `src/cpu/src/wif.cpp` | WIF encode/decode |

### 3.6 C ABI Shim (MEDIUM)

| File | Component |
|------|-----------|
| `c_api/ufsecp_impl.cpp` | C ABI bridge (41 exported functions) |
| `c_api/include/ufsecp/ufsecp.h` | Public C header |
| `c_api/include/ufsecp/ufsecp_error.h` | Error codes |

### 3.7 Hash Functions (MEDIUM)

| File | Component |
|------|-----------|
| `src/cpu/src/sha256.cpp` | SHA-256 (used for RFC 6979, BIP-340 tagged hashes, BIP-32) |
| `src/cpu/src/ripemd160.cpp` | RIPEMD-160 (P2PKH addresses) |

---

## 4. Out-of-Scope Components

| Component | Reason |
|-----------|--------|
| GPU kernels (CUDA/OpenCL/Metal/ROCm) | Variable-time; accepts secrets only in trusted single-tenant environments; separate audit recommended |
| Language bindings (Python/Rust/Go/C#/etc.) | Thin FFI wrappers over C ABI; lower risk |
| Build scripts, CI configuration | Infrastructure -- separate DevSecOps review |
| `apps/` directory (GPU search tools) | Application-layer, not library |
| `compat/` (libsecp256k1 shim) | Compatibility wrapper, not primary API |
| Benchmark and example code | Non-production |

---

## 5. Known Assumptions & Limitations

1. **Limited formal verification**: CT layer uses a three-tier verification stack: ct-verif (LLVM IR analysis, CI-enforced), Valgrind CT (memory-undefined tracking, CI-enforced), and dudect (statistical timing, CI-enforced). Cryptol specifications cover field/point/ECDSA/Schnorr algebraic properties. Machine-checked proof frameworks (Vale/Jasmin/Coq/Fiat-Crypto) are not yet applied
2. **Compiler trust**: CT guarantees assume compiler does not introduce secret-dependent branches at `-O2`
3. **MuSig2/FROST C++ API evolving**: C++ API may change before v4.0; C ABI is stable. Protocol correctness validated against IETF RFC 9591 (FROST) and BIP-327 (MuSig2) with exploit PoC coverage
4. **GPU variable-time**: All GPU backends are variable-time; secret-bearing ops (ECDH, BIP-352, BIP-324 AEAD) require trusted single-tenant environment. Documented in THREAT_MODEL.md
5. **No hardware side-channel protection**: Library does not defend against power analysis, EM emanation, or fault injection

---

## 6. Testing Infrastructure Available to Auditors

| Test Suite | Checks | Description |
|------------|--------|-------------|
| `test_field_audit` | 641,194 | Comprehensive field/scalar/point arithmetic |
| `test_bip340_vectors` | 15 | BIP-340 official sign + verify vectors |
| `test_rfc6979_vectors` | 6 | RFC 6979 nonce + sign/verify |
| `test_bip32_vectors` | 90 | BIP-32 TV1-TV5 |
| `test_cross_libsecp256k1` | 7,860 | Differential vs bitcoin-core/libsecp256k1 v0.6.0 |
| `test_ecc_properties` | ~10K | Group law properties (associativity, distributivity) |
| `test_ct_sidechannel` | dudect | Timing leak detection (scalar_mul, sign, inv) |
| `test_musig2_frost` | 975 | MuSig2 + FROST protocol tests |
| `test_musig2_frost_advanced` | 316 | Rogue-key, transcript binding, fault injection |
| `test_fuzz_parsers` | 580K | DER/Schnorr/Pubkey parser fuzz |
| `test_fuzz_address_bip32_ffi` | 73,959 | Address/BIP32/FFI boundary fuzz |
| libFuzzer harnesses | ∞ | Continuous fuzz for field/scalar/point |
| `security_autonomy_check.py` | 10 gates | Security autonomy orchestrator: formal invariants, SLA, supply chain, misuse resistance, evidence governance, incident drills, perf-security co-gate, risk-surface coverage, fuzz campaigns |
| `test_adversarial_protocol` (§H) | 100+ | New ABI surface edge cases: AEAD, ECIES, EllSwift, ETH, Pedersen switch, Schnorr adaptor, batch sign, BIP-143/144, SegWit, Taproot sighash |
| `test_adversarial_protocol` (§I) | 77 | Remaining ABI surface: `ctx_clone`, `last_error_msg`, `pubkey_parse`, `pubkey_create_uncompressed`, `ecdsa_sign_recoverable`, `ecdsa_recover`, `ecdsa_sign_verified`, `schnorr_sign_verified`, deep batch verify |
| `test_gpu_host_api_negative` | 38 | GPU C ABI: NULL ctx, NULL ctx_out, NULL info_out, count=0, count>0 + NULL buffers, invalid backend (0/99/255), invalid device index, error strings for all 7 GPU error codes, backend names |
| `test_gpu_abi_gate` | 28 | GPU ABI gate: backend discovery, device info, context lifecycle, NULL buffer ops, error strings, ops-if-available (1*G smoke, count=0 no-op, NULL-scalar failure) |

### Mandatory Edge-Case Coverage Rule (enforced since v3.22)

Every publicly exported `ufsecp_*` function MUST satisfy all four checks below
before it is considered covered for audit purposes:

1. **NULL rejection** — every pointer parameter returns `UFSECP_ERR_NULL_ARG`
2. **Zero-count / zero-length / zero-key rejection** — where the contract prohibits empty inputs
3. **Invalid-content rejection** — bad prefix, off-curve key, truncated ciphertext, wrong
   authentication tag, OOB index, etc. must return a non-`UFSECP_OK` code
4. **Success smoke** — at least one valid call demonstrates a correct round-trip or output

Evidence for these checks lives in `audit/test_adversarial_protocol.cpp` (§G, §H, and §I)
and the GPU ABI in `audit/test_gpu_host_api_negative.cpp` + `audit/test_gpu_abi_gate.cpp`,
both integrated into the unified audit runner (v3.24+).

### Reproduction Commands

> **One-command full audit**: After building, run `./build_audit/audit/unified_audit_runner`
> to execute all 263 modules across 9 failure classes. Every claim in this document
> maps to a module in that runner. A traditional PDF audit is a snapshot; this runner
> is a living, continuously enforced assurance perimeter.

```bash
# Configure (Linux)
cmake -S . -B build_audit -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CROSS_TESTS=ON \
  -DSECP256K1_BUILD_FUZZ_TESTS=ON \
  -DSECP256K1_BUILD_PROTOCOL_TESTS=ON

# Build
cmake --build build_audit -j

# === FULL AUDIT (263 modules, 9 failure classes, ~10 min) ===
./build_audit/audit/unified_audit_runner

# Run all CTest targets
ctest --test-dir build_audit --output-on-failure

# Run specific suite
ctest --test-dir build_audit -R test_field_audit -V

# dudect constant-time smoke test
ctest --test-dir build_audit -R ct_sidechannel_smoke -V

# Exploit PoC security probes (254 probes)
ctest --test-dir build_audit -R exploit -V

# Machine-readable assurance artifact
python3 ci/export_assurance.py -o assurance_report.json

# Security autonomy gate orchestrator
python3 ci/security_autonomy_check.py --json
```

---

## 7. Independent Review Output Schema

| # | Deliverable | Format |
|---|-------------|--------|
| D1 | Findings report with severity classification (Critical/High/Medium/Low/Info) | Markdown / JSON / PDF |
| D2 | Specific file:line references for each finding | Included in D1 |
| D3 | Reproduction steps for each finding | Included in D1 |
| D4 | Recommendations for remediation | Included in D1 |
| D5 | Assessment of CT layer effectiveness | Dedicated section in D1 |
| D6 | Comparison against libsecp256k1 (bitcoin-core) where applicable | Dedicated section in D1 |
| D7 | Re-test of fixes, if findings land | Separate replay artifact |

---

## 8. Suggested Review Flow

| Phase | Duration | Activities |
|-------|----------|------------|
| Replay | As needed | Re-run CAAS, verify bundle hashes, inspect generated evidence |
| Core review | As needed | Field/scalar/point, signatures, CT layer |
| Protocol review | As needed | MuSig2, FROST, BIP-32, address generation |
| API/boundary review | As needed | C ABI, error handling, thread safety |
| Findings | As needed | Findings compilation, severity assignment |
| Regression pinning | As needed | Convert accepted findings into permanent CAAS tests |

---

## 9. Reference Documents

| Document | Path |
|----------|------|
| Threat Model | `THREAT_MODEL.md` |
| Architecture | `docs/ARCHITECTURE.md` |
| CT Verification | `docs/CT_VERIFICATION.md` |
| Test Matrix | `docs/TEST_MATRIX.md` |
| ABI Versioning | `docs/ABI_VERSIONING.md` |
| Normalization Spec | `docs/NORMALIZATION_SPEC.md` |
| Security Policy | `SECURITY.md` |
| Audit Guide | `AUDIT_GUIDE.md` |
| API Reference | `docs/API_REFERENCE.md` |

---

## 10. Contact & Access

- **Repository**: https://github.com/shrec/UltrafastSecp256k1
- **Primary Contact**: payysoon@gmail.com
- **Vulnerability Reports**: GitHub Security Advisories (preferred)
- **Branch for audit**: `main` (latest stable release tag)
- **Audit artifacts**: Will be published at `docs/audit/` post-engagement
