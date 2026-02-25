# Internal Security Audit -- Full Results

**UltrafastSecp256k1 v3.14.0**  
**Audit Date**: 2026-02-25  
**Branch**: `dev` (HEAD)  
**Methodology**: Automated + manual, deterministic seeds, zero external dependencies  
**Verdict**: **ALL PASSED -- 0 critical / 0 high / 0 medium findings**

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Audit Scope](#2-audit-scope)
3. [Test Infrastructure Overview](#3-test-infrastructure-overview)
4. [Section I -- Core Arithmetic (641K checks)](#4-section-i--core-arithmetic)
5. [Section II -- Constant-Time & Side-Channel](#5-section-ii--constant-time--side-channel)
6. [Section III -- Signature Schemes](#6-section-iii--signature-schemes)
7. [Section IV -- Multi-Party Protocols (MuSig2 + FROST)](#7-section-iv--multi-party-protocols)
8. [Section V -- Cross-Library Differential (vs libsecp256k1)](#8-section-v--cross-library-differential)
9. [Section VI -- Fuzzing & Adversarial](#9-section-vi--fuzzing--adversarial)
10. [Section VII -- Security Hardening](#10-section-vii--security-hardening)
11. [Section VIII -- Integration & Protocol Flows](#11-section-viii--integration--protocol-flows)
12. [Section IX -- Key Derivation & Address Generation](#12-section-ix--key-derivation--address-generation)
13. [Section X -- Performance Baseline](#13-section-x--performance-baseline)
14. [Invariant Catalog Summary](#14-invariant-catalog-summary)
15. [CI/CD Security Measures](#15-cicd-security-measures)
16. [Coverage Gaps & Known Limitations](#16-coverage-gaps--known-limitations)
17. [How to Reproduce](#17-how-to-reproduce)

---

## 1. Executive Summary

This document consolidates **all internal audit results** for UltrafastSecp256k1.
No external audit firm was engaged; all verification is performed by the development
team and automated CI infrastructure.

| Metric | Value |
|--------|-------|
| **Core audit checks** | **641,194** (0 failures) |
| **Extended test checks** (protocols, KAT, fuzz, differential) | **~820,000+** |
| **Audit test suites** | 8 dedicated audit binaries |
| **Extended test suites** | 25+ CTest targets |
| **Fuzz harnesses** | 3 libFuzzer + 2 structured fuzz suites |
| **Side-channel analysis** | dudect (Welch t-test), 1300+ lines |
| **Differential comparison** | vs bitcoin-core/libsecp256k1 v0.6.0 (7,860 checks) |
| **Standard test vectors** | BIP-340 (15), RFC 6979 (6), BIP-32 (90) |
| **Protocol tests** | MuSig2 (975), FROST (316), FROST KAT (76), Advanced (316) |
| **CI workflows** | 14 automated pipelines |
| **Platforms tested** | 13+ (x86-64, ARM64, RISC-V, WASM, ESP32, STM32, CUDA, OpenCL, Metal, ROCm) |

### Risk Assessment

| Component | Maturity | Confidence |
|-----------|----------|------------|
| Field Arithmetic (𝔽ₚ) | Production | **Very High** -- 264K audit checks + fuzz + differential |
| Scalar Arithmetic (ℤ_n) | Production | **Very High** -- 93K audit checks + fuzz + differential |
| Point Operations | Production | **Very High** -- 116K audit checks + fuzz + differential |
| ECDSA (RFC 6979) | Production | **Very High** -- BIP-340 vectors + RFC 6979 vectors + differential vs libsecp256k1 |
| Schnorr (BIP-340) | Production | **Very High** -- All 15 official vectors + differential |
| CT Layer | Production | **High** -- 120K equivalence checks + dudect timing + code review (no formal verification) |
| MuSig2 | Experimental | **High** -- 975 checks + rogue-key + transcript binding + fault injection |
| FROST | Experimental | **High** -- 1,367 checks (DKG + signing + KAT + malicious participant) |
| BIP-32 HD | Experimental | **High** -- TV1-TV5 (90 checks) + fuzz |
| C ABI (ufsecp) | Experimental | **Medium** -- Fuzz + NULL handling (73K checks), no multi-ABI cross-test |
| GPU Backends | Beta | **Medium** -- Functional, NOT constant-time, limited differential vs CPU |

---

## 2. Audit Scope

### In Scope

| Component | Files | Lines (approx) |
|-----------|-------|----------------|
| Field arithmetic | `cpu/src/field.cpp`, `field.hpp`, `field_branchless.hpp` | ~850 |
| Scalar arithmetic | `cpu/src/scalar.cpp`, `scalar.hpp` | ~700 |
| Point operations | `cpu/src/point.cpp`, `point.hpp` | ~1,000 |
| GLV endomorphism | `cpu/src/glv.cpp`, `glv.hpp` | ~300 |
| ECDSA | `cpu/src/ecdsa.cpp`, `ecdsa.hpp` | ~500 |
| Schnorr | `cpu/src/schnorr.cpp`, `schnorr.hpp` | ~400 |
| CT layer | `cpu/src/ct/`, `cpu/include/secp256k1/ct/` | ~1,200 |
| MuSig2 | `cpu/src/musig2.cpp`, `musig2.hpp` | ~600 |
| FROST | `cpu/src/frost.cpp`, `frost.hpp` | ~700 |
| SHA-256 | `cpu/src/sha256.cpp` | ~300 |
| BIP-32 | `cpu/src/bip32.cpp` | ~400 |
| Address gen | `cpu/src/address.cpp`, `wif.cpp` | ~600 |
| C ABI | `c_api/ufsecp_impl.cpp`, `ufsecp.h` | ~800 |
| ASM backends | `field_asm_x64.asm`, `field_asm_arm64.cpp`, `field_asm_riscv64.S` | ~1,500 |

### Out of Scope

| Component | Reason |
|-----------|--------|
| GPU kernels (CUDA/OpenCL/Metal/ROCm) | Public-data only, variable-time by design |
| Language bindings (Python/Rust/Go/C#/Node/etc.) | Thin FFI wrappers over C ABI |
| Build system, CI scripts | Infrastructure, not cryptographic code |
| Example / benchmark code | Non-production |

---

## 3. Test Infrastructure Overview

### Audit Suites (Dedicated, Deterministic)

| Suite | File | Checks | Time | Focus |
|-------|------|-------:|-----:|-------|
| audit_field | `tests/audit_field.cpp` | 264,622 | 0.29s | Field ₚ: add/sub/mul/sqr/inv/sqrt/batch |
| audit_scalar | `tests/audit_scalar.cpp` | 93,215 | 0.32s | Scalar _n: arithmetic, GLV, negate, boundary |
| audit_point | `tests/audit_point.cpp` | 116,124 | 1.71s | Point: add/dbl/mul, ECDSA/Schnorr round-trip |
| audit_ct | `tests/audit_ct.cpp` | 120,652 | 0.93s | CT: FAST==CT equivalence, cmov/cswap, timing |
| audit_fuzz | `tests/audit_fuzz.cpp` | 15,461 | 0.53s | Adversarial: malformed keys, invalid sigs |
| audit_perf | `tests/audit_perf.cpp` | -- | 1.19s | Performance baseline (benchmark) |
| audit_security | `tests/audit_security.cpp` | 17,309 | 17.26s | Bit-flip, RFC 6979, low-S, zeroing |
| audit_integration | `tests/audit_integration.cpp` | 13,811 | 1.62s | ECDH, batch verify, cross-path, mixed ops |
| **Total** | | **641,194** | **~24s** | |

### Extended Test Suites

| Suite | File | Checks | Focus |
|-------|------|-------:|-------|
| test_cross_libsecp256k1 | `tests/test_cross_libsecp256k1.cpp` | 7,860 | Differential vs bitcoin-core/libsecp256k1 v0.6.0 |
| test_musig2_frost | `tests/test_musig2_frost.cpp` | 975 | MuSig2 + FROST protocol simulation |
| test_musig2_frost_advanced | `tests/test_musig2_frost_advanced.cpp` | 316 | Rogue-key, transcript, fault injection |
| test_frost_kat | `tests/test_frost_kat.cpp` | 76 | FROST Known-Answer Tests (pinned vectors) |
| test_fuzz_parsers | `tests/test_fuzz_parsers.cpp` | ~580,000 | DER/Schnorr/Pubkey structured fuzz |
| test_fuzz_address_bip32_ffi | `tests/test_fuzz_address_bip32_ffi.cpp` | 73,959 | Address/BIP32/FFI boundary fuzz |
| test_bip340_vectors | `cpu/tests/test_bip340_vectors.cpp` | 15 | All 15 official BIP-340 test vectors |
| test_rfc6979_vectors | `cpu/tests/test_rfc6979_vectors.cpp` | 6 | RFC 6979 nonce + sign/verify |
| test_bip32_vectors | `cpu/tests/test_bip32_vectors.cpp` | 90 | BIP-32 TV1-TV5 official vectors |
| test_ecc_properties | `cpu/tests/test_ecc_properties.cpp` | ~10,000 | Group law: associativity, distributivity |
| test_ct_sidechannel | `tests/test_ct_sidechannel.cpp` | -- | dudect timing analysis (1300+ lines) |
| test_comprehensive | `cpu/tests/test_comprehensive.cpp` | ~25,000 | 25+ test categories |
| test_ct_equivalence | `cpu/tests/test_ct_equivalence.cpp` | ~5,000 | FAST == CT property-based |

### Fuzz Harnesses (libFuzzer)

| Harness | File | Input Size | Operations |
|---------|------|-----------|------------|
| fuzz_field | `cpu/fuzz/fuzz_field.cpp` | 32 bytes | add/sub round-trip, mul identity, sqr, inv |
| fuzz_scalar | `cpu/fuzz/fuzz_scalar.cpp` | 32 bytes | add/sub, mul identity, distributive law |
| fuzz_point | `cpu/fuzz/fuzz_point.cpp` | 32 bytes | on-curve, negate, compress rt, dbl vs add |

### CI Workflows (Automated Security)

| Workflow | Trigger | What It Checks |
|----------|---------|---------------|
| `ci.yml` | Every push/PR | Full build + test suite (Linux+Windows+macOS) |
| `security-audit.yml` | Weekly + push | ASan, UBSan, TSan, Valgrind memcheck, `-Werror -Wall -Wextra` |
| `codeql.yml` | Every push/PR | C/C++ security-and-quality static analysis |
| `scorecard.yml` | Weekly | OpenSSF Scorecard supply-chain assessment |
| `clang-tidy.yml` | Every push/PR | 30+ static analysis checks |
| `sonarcloud.yml` | Push | Code quality & security hotspots |
| `benchmark.yml` | Every push to dev/main | Performance regression (150% alert threshold) |
| `nightly.yml` | Daily | 1.3M+ differential checks, 30-min dudect full run |
| `release.yml` | Tag push | Reproducible build + cosign signing + SBOM |
| `packaging.yml` | Push | Multi-platform package build verification |
| `bindings.yml` | Push | Language binding CI (Python/Rust/Go/C#/etc.) |
| `docs.yml` | Push | Documentation build + deploy |
| `dependency-review.yml` | PR | Vulnerable dependency scanning |
| `discord-commits.yml` | Push | Notification (non-security) |

---

## 4. Section I -- Core Arithmetic

### 4.1 Field Arithmetic (𝔽ₚ)

**Checks: 264,622** | **File: audit_field.cpp** | **PRNG Seed: 0xA0D17'F1E1D**

| # | Test | Checks | What Was Verified |
|---|------|-------:|-------------------|
| 1 | Addition overflow | 3,101 | `p-1 + 1`, `p-1 + p-1`, `x + 0`, random pairs |
| 2 | Subtraction borrow | 6,102 | `0 - x`, `x - x == 0`, add/sub consistency |
| 3 | Multiplication carry | 11,102 | Mul-by-1, mul-by-0, commutativity, large operands |
| 4 | Square == Mul (10K) | 21,104 | `sqr(x) == mul(x,x)` for 10,000 random elements |
| 5 | Reduction | 22,106 | Above-p values reduce correctly; idempotent |
| 6 | Canonical form (10K) | 42,106 | `from_bytes(to_bytes(x))` round-trip |
| 7 | Limb boundary | 43,109 | Single-limb: 0, 1, UINT64_MAX |
| 8 | Inverse (10K) | 54,110 | `x * inv(x) == 1` for 10,000 non-zero elements |
| 9 | Square root | 64,110 | `sqrt(x^2) == +-x`; 50.72% QR rate (expected ~50%) |
| 10 | Batch inverse | 64,622 | `batch_inv` matches per-element `inv` |
| 11 | Random cross (100K) | 264,622 | 100K mixed ops: add, sub, mul, sqr consistency |

**Key Finding**: Square root QR existence rate was 50.72% -- confirming correct quadratic residue behavior.

### 4.2 Scalar Arithmetic (ℤ_n)

**Checks: 93,215** | **File: audit_scalar.cpp** | **PRNG Seed: 0xA0D17'5CA1A**

| # | Test | Checks | What Was Verified |
|---|------|-------:|-------------------|
| 1 | Mod n reduction | 10,003 | Values above order n reduce correctly |
| 2 | Overflow normalization (10K) | 10,003 | `from_bytes -> to_bytes` canonical |
| 3 | Edge scalars | 10,210 | 0, 1, n-1, n, n+1 |
| 4 | Arithmetic laws (10K) | 60,210 | Commutativity, associativity, distributivity |
| 5 | Scalar inverse (10K) | 71,210 | `s * inv(s) == 1` |
| 6 | GLV split (1K) | 73,210 | `k*G == k1*G + k2*(lambda*G)` algebraic verification |
| 7 | High-bit boundary | 73,214 | Scalars near 2^255 |
| 8 | Negate (10K) | 93,215 | `s + neg(s) == 0` |

**Key Finding**: GLV decomposition verified algebraically through point arithmetic, not just scalar identity.

### 4.3 Point Operations

**Checks: 116,124** | **File: audit_point.cpp** | **PRNG Seed: 0xA0D17'901E7**

| # | Test | Checks | What Was Verified |
|---|------|-------:|-------------------|
| 1 | Infinity identity | 7 | P+O==P, 0*G==O |
| 2 | Jacobian add (1.5K) | 1,508 | P+Q correctness, associativity sampling |
| 3 | Jacobian double | 1,512 | 2P via dbl matches add(P,P) |
| 4 | P+P via add (H=0 case) | 1,612 | Add function handles doubling case |
| 5 | P+(-P) == O (1K) | 3,614 | Additive inverse |
| 6 | Affine conversion (1K) | 7,614 | Jac->Aff round-trip + on-curve check (y^2=x^3+7) |
| 7 | Scalar mul identities (1.5K) | 9,114 | 1*P==P, 0*P==O, (a+b)*P==aP+bP |
| 8 | Known k*G vectors | 9,124 | Test vectors for generator multiplication |
| 9 | ECDSA round-trip (1K) | 14,124 | Sign -> verify for 1,000 random pairs |
| 10 | Schnorr round-trip (1K) | 16,124 | BIP-340 sign -> verify for 1,000 pairs |
| 11 | 100K stress | 116,124 | Mixed add/dbl/mul; zero infinity hits |

**Key Findings**: Zero infinity hits across 100K random operations. 100% sign/verify success rate.

---

## 5. Section II -- Constant-Time & Side-Channel

### 5.1 CT Equivalence (120K checks)

**File: audit_ct.cpp** | **PRNG Seed: 0xA0D17'C71AE**

| # | Test | Checks | What Was Verified |
|---|------|-------:|-------------------|
| 1 | CT mask generation | 12 | `ct_mask_if`, `ct_select` for edge values |
| 2 | CT cmov/cswap (10K) | 30,012 | Conditional move/swap correctness |
| 3 | CT table lookup | 30,028 | Full-scan vs direct access -- identical |
| 4 | CT field differential (10K) | 81,028 | `ct::field_* == fast::field_*` for all ops |
| 5 | CT scalar differential (10K) | 111,028 | `ct::scalar_* == fast::scalar_*` for all ops |
| 6 | CT scalar cmov/cswap (1K) | 113,028 | Scalar conditional correctness |
| 7 | CT field cmov/cswap/select (1K) | 117,028 | Field conditional correctness |
| 8 | CT comparisons | 118,036 | `is_zero`, `eq` on boundary values |
| 9 | CT scalar_mul (1K) | 119,038 | `ct::scalar_mul(k, G) == fast::scalar_mul(k, G)` |
| 10 | CT complete addition (1K) | 120,141 | Unified addition == FAST addition |
| 11 | CT byte utilities | 120,151 | `ct_memzero`, `ct_memeq`, `ct_memcpy_if` |
| 12 | CT generator_mul (500) | 120,651 | `ct::generator_mul == fast::generator_mul` |
| 13 | Timing variance | 120,652 | k=1 vs k=n-1 ratio check |

**FAST == CT Equivalence**: Bit-exact match confirmed for all field, scalar, and point operations across 120K random + edge-case inputs.

### 5.2 dudect Timing Analysis

**File: test_ct_sidechannel.cpp** | **1300+ lines**

| Target | Method | Result | t-statistic |
|--------|--------|--------|-------------|
| `ct::scalar_mul` (k=1 vs k=n-1) | Welch t-test (10K samples) | **PASS** | < 4.5 |
| `ct::ecdsa_sign` (key=low vs key=high) | Welch t-test (10K samples) | **PASS** | < 4.5 |
| `ct::schnorr_sign` (key=low vs key=high) | Welch t-test (10K samples) | **PASS** | < 4.5 |
| `ct::field_inv` (value=1 vs value=p-1) | Welch t-test (10K samples) | **PASS** | < 4.5 |
| `ct::generator_mul` (k=1 vs k=random) | Welch t-test (10K samples) | **PASS** | < 4.5 |

**Methodology**: Binary comparison -- "class A" and "class B" have different secret inputs; execution times are measured and compared via Welch's t-test. A t-statistic below 4.5 (99.999% confidence threshold) means no detectable timing difference.

**CI Integration**:
- **Smoke mode**: Every push/PR (DUDECT_SMOKE, threshold t=25.0)
- **Full mode**: Nightly (30 minutes, threshold t=4.5)

**Limitation**: dudect tests timing on the CI runner's CPU (x86-64). Other microarchitectures (ARM, RISC-V, Apple Silicon) may exhibit different behavior. No formal verification (ct-verif, Vale) has been applied.

### 5.3 CT Timing Ratio

From audit_ct.cpp Section 13:

| Pair | Avg ns (k=1) | Avg ns (k=n-1) | Ratio |
|------|-------------|----------------|-------|
| `ct::scalar_mul` | 363,380 | 351,039 | **1.035** |

Ideal ratio = 1.0. Concern threshold = 1.2. Result is well within acceptable bounds.

---

## 6. Section III -- Signature Schemes

### 6.1 ECDSA (RFC 6979)

| Test Source | Checks | What Was Verified |
|-------------|-------:|-------------------|
| audit_point.cpp (#9) | 1,000 | Random sign -> verify round-trip |
| audit_security.cpp (#3-4) | 3,000 | Bit-flip resilience (sig + msg) |
| audit_security.cpp (#5) | 101 | RFC 6979 determinism |
| audit_security.cpp (#10) | 1,000 | Low-S enforcement (BIP-62) |
| test_rfc6979_vectors.cpp | 6 | RFC 6979 official nonce + sign/verify |
| test_cross_libsecp256k1.cpp | ~2,600 | Differential vs libsecp256k1 |
| test_fuzz_parsers.cpp | ~200K | DER encoding/decoding fuzz |
| **Total** | **~208K** | |

**Key Findings**:
- RFC 6979 deterministic nonce: same (key, msg) always produces identical signature
- Low-S always enforced: 0/1,000 high-S signatures observed
- 100% bit-flip detection rate on both signatures and messages (0/2,000 false positives)
- UF ECDSA output matches libsecp256k1 for 1,000 random key+msg pairs

### 6.2 Schnorr (BIP-340)

| Test Source | Checks | What Was Verified |
|-------------|-------:|-------------------|
| audit_point.cpp (#10) | 1,000 | Random sign -> verify round-trip |
| test_bip340_vectors.cpp | 15 | All 15 official vectors (v0-v3 sign + v4-v14 verify) |
| test_cross_libsecp256k1.cpp | ~2,000 | Differential vs libsecp256k1 schnorrsig |
| test_fuzz_parsers.cpp | ~200K | 64-byte signature fuzz |
| **Total** | **~203K** | |

**Key Findings**:
- All 15 BIP-340 test vectors pass (4 signing + 11 verification, including deliberate failures)
- Tagged hashing per BIP-340 specification confirmed
- UF Schnorr output matches libsecp256k1 for all random test cases

---

## 7. Section IV -- Multi-Party Protocols

### 7.1 MuSig2

| Test Source | Checks | What Was Verified |
|-------------|-------:|-------------------|
| test_musig2_frost.cpp (suites 1-6) | 975 | Key aggregation, nonce gen, partial sign, 2/3/5-party |
| test_musig2_frost_advanced.cpp (suites 1-5) | ~160 | Rogue-key resistance, transcript binding, fault injection |
| **Total** | **~1,135** | |

**Findings**:
- Key aggregation is deterministic for same pubkey set
- Nonce reuse across different messages detected
- Wagner-style rogue-key manipulation detected and rejected
- Invalid partial signature rejected before aggregation
- Aggregated signature verifies as standard BIP-340 Schnorr

### 7.2 FROST Threshold Signatures

| Test Source | Checks | What Was Verified |
|-------------|-------:|-------------------|
| test_musig2_frost.cpp (suites 7-11) | ~500 | DKG (2-of-3, 3-of-5), signing round-trip, share consistency |
| test_musig2_frost_advanced.cpp (suites 6-9) | ~156 | Malicious participant, commitment forgery, below-threshold |
| test_frost_kat.cpp (9 suites) | 76 | Pinned Known-Answer Tests (regression anchors) |
| **Total** | **~732** | |

**FROST KAT Test Suites (test_frost_kat.cpp)**:

| Suite | What Was Verified |
|-------|-------------------|
| Lagrange coefficients | Known mathematical values for lambda_1, lambda_2 |
| DKG share consistency | Shamir secret reconstruction (sum of shares recovers secret) |
| Signing round determinism | Same seeds -> same nonce commitments and partial sigs |
| Aggregate signature validity | BIP-340 schnorr_verify on FROST output |
| Cross-threshold consistency | 2-of-3 vs 3-of-5 group key comparison for same secrets |
| Partial signature verification | frost_verify_partial correctness |
| Multiple signer subsets | Any valid t-subset produces valid signature |
| Nonce commitment binding | Commitment <-> nonce relationship |
| Regression anchors | Pinned hex values for all intermediate outputs |

**Findings**:
- t-of-n DKG produces consistent group public key across all participants
- Aggregated FROST signature verifies as standard BIP-340 Schnorr
- Malicious share in DKG detected by Feldman VSS commitment verification
- Below-threshold subset correctly fails to produce valid signature
- Deterministic seeds produce reproducible outputs (regression-safe)

**Note**: IETF RFC 9591 does not define a secp256k1 ciphersuite, so external cross-check vectors are unavailable. KATs are self-generated with fixed seeds.

---

## 8. Section V -- Cross-Library Differential

**File: test_cross_libsecp256k1.cpp** | **Checks: 7,860** | **Reference: bitcoin-core/libsecp256k1 v0.6.0**

| Suite | Checks | What Was Verified |
|-------|-------:|-------------------|
| Generator multiplication (1K) | 1,000 | `UF(k*G).compressed == libsecp(k*G).compressed` |
| Arbitrary point multiplication (1K) | 1,000 | `UF(k*P) == libsecp(k*P)` |
| ECDSA sign determinism (1K) | 2,000 | Same (key, msg) -> same (r, s) in both libs |
| ECDSA verify cross (1K) | 1,000 | libsecp verifies UF-signed; UF verifies libsecp-signed |
| Schnorr sign cross (1K) | 1,860 | BIP-340 sign + verify cross-checked |
| Scalar arithmetic (500) | 500 | add, mul, inv, negate match |
| Point serialization (500) | 500 | Compressed/uncompressed encoding match |

**Key Finding**: **Zero mismatches** across 7,860 cross-library comparisons. Both libraries implement identical secp256k1 mathematics.

**Nightly run**: `nightly.yml` executes with multiplier=100, producing ~1.3M cross-library checks.

---

## 9. Section VI -- Fuzzing & Adversarial

### 9.1 Audit Fuzz Suite (15K checks)

**File: audit_fuzz.cpp**

| Test | Checks | Result |
|------|-------:|--------|
| Malformed pubkey rejection | 3 | All rejected |
| Invalid ECDSA sigs (r=0, s=0, r=n, s=n) | 7 | All rejected |
| Invalid Schnorr sigs | 11 | All rejected |
| Oversized scalars (> n) | 15 | Correctly reduced |
| Boundary field elements (0, p, p-1, p+1) | 19 | Correctly handled |
| ECDSA recovery edge (1K) | 4,769 | Wrong-ID rejected |
| Random state fuzz (10K) | 6,461 | 0 crashes, 0 UB |
| DER round-trip (1K) | 9,461 | Encode->decode identical |
| Schnorr bytes round-trip (1K) | 11,461 | Serialize->deserialize identical |
| Low-S normalization (1K) | 15,461 | All s in lower half |

### 9.2 Parser Fuzz Suite (~580K checks)

**File: test_fuzz_parsers.cpp**

| Suite | Focus | Checks |
|-------|-------|-------:|
| DER signature: random blobs | No crash on arbitrary input | ~200K |
| DER signature: valid mutations | Bit-flip/truncation detection | ~100K |
| DER round-trip: valid sigs | Encode->decode identity | ~80K |
| Schnorr sig: random blobs | No crash on arbitrary input | ~100K |
| Schnorr round-trip | Serialize->deserialize identity | ~50K |
| Pubkey parse: random blobs | Invalid prefix/point rejection | ~30K |
| Pubkey compressed round-trip | 33-byte encode->decode | ~10K |
| Pubkey uncompressed round-trip | 65-byte encode->decode | ~10K |

### 9.3 Address/BIP32/FFI Fuzz (~74K checks)

**File: test_fuzz_address_bip32_ffi.cpp**

| Suite | Focus | Checks |
|-------|-------|-------:|
| Base58Check encode/decode | Round-trip with random payloads | ~10K |
| Bech32/Bech32m encode/decode | Round-trip with random witness programs | ~10K |
| WIF encode/decode | Private key serialization round-trip | ~5K |
| Address generation (27 coins) | No crash on random pubkeys | ~5K |
| BIP32 path parsing | Valid/invalid path strings | ~10K |
| BIP32 derivation | Random seeds, deep paths | ~10K |
| FFI (ufsecp) boundary | NULL args, invalid lengths, error codes | ~24K |

### 9.4 libFuzzer Harnesses (Continuous)

Three libFuzzer harnesses run continuously in CI and nightly:

```bash
# Field: 32-byte input -> add/sub/mul/sqr/inv operations
# Scalar: 32-byte input -> add/sub/mul/inv operations  
# Point: 32-byte seed -> on-curve, compress, add, dbl operations
```

**No crashes or sanitizer violations detected** in any fuzz campaign.

---

## 10. Section VII -- Security Hardening

**File: audit_security.cpp** | **Checks: 17,309** | **PRNG Seed: 0xA0D17'5EC01**

| # | Test | Checks | Result |
|---|------|-------:|--------|
| 1 | Zero/identity key handling | 5 | `inv(0)` throws; `0*G==O`; zero-key sign fails |
| 2 | Secret zeroization (`ct_memzero`) | 8 | Memory confirmed zero after call |
| 3 | Bit-flip resilience (1K sigs) | 2,008 | 1-bit flip -> verify fails (100% detection) |
| 4 | Message bit-flip (1K) | 3,008 | 1-bit flip -> verify fails (100% detection) |
| 5 | RFC 6979 determinism | 3,109 | Same inputs -> same sig; different msg -> different sig |
| 6 | Serialization round-trip (3K) | 10,109 | Compressed, uncompressed, x-only |
| 7 | Compact recovery (1K) | 12,109 | Compact sig -> recover pubkey -> matches |
| 8 | Double-ops idempotency (2K) | 14,209 | sign-twice==same; verify-twice==same |
| 9 | Cross-algorithm consistency | 14,309 | Same key valid for ECDSA + Schnorr |
| 10 | High-S detection (1K) | 17,309 | Low-S enforced per BIP-62 |

**Key Findings**:
- `inverse(0)` correctly throws -- no silent zero return
- 100% bit-flip detection rate on both signatures and messages
- RFC 6979 determinism confirmed
- Low-S enforcement verified across 1,000 random signatures

---

## 11. Section VIII -- Integration & Protocol Flows

**File: audit_integration.cpp** | **Checks: 13,811** | **PRNG Seed: 0xA0D17'16780**

| # | Test | Checks | Result |
|---|------|-------:|--------|
| 1 | ECDH symmetry (1K) | 4,001 | `ECDH(a, bG) == ECDH(b, aG)` for all 3 variants |
| 2 | Schnorr batch verify | 4,006 | 100 valid sigs; corrupt detection + identify_invalid |
| 3 | ECDSA batch verify | 4,009 | 100 valid sigs; corrupt detection + identify_invalid |
| 4 | ECDSA full round-trip (1K) | 10,009 | sign -> recover -> verify -> DER encode/decode |
| 5 | Schnorr cross-path (500) | 11,010 | Individual verify == batch verify |
| 6 | FAST vs CT integration (500) | 12,510 | `fast::scalar_mul == ct::scalar_mul`; cross-verify |
| 7 | ECDH + ECDSA protocol (100) | 13,010 | Full key-exchange + signing flow |
| 8 | Multi-key consistency (200) | 13,210 | Aggregated P1+P2 and individual verifications |
| 9 | Schnorr/ECDSA key consistency (200) | 13,810 | Same keypair valid for both schemes |
| 10 | Mixed protocol stress (5K) | 13,811 | 5,000 random mixed ops; 100% success rate |

**Key Findings**:
- ECDH symmetry holds for all three variants (hashed, x-only, raw)
- Batch verification correctly identifies individual invalid signatures
- FAST and CT paths produce interoperable signatures/points
- 5,000 mixed random operations completed with zero failures

---

## 12. Section IX -- Key Derivation & Address Generation

### BIP-32 HD Derivation

| Test Source | Checks | What Was Verified |
|-------------|-------:|-------------------|
| test_bip32_vectors.cpp | 90 | TV1-TV5 official vectors (public key decompression fix confirmed) |
| test_fuzz_address_bip32_ffi.cpp | ~10K | Random seed derivation, deep path parsing, edge cases |
| **Total** | **~10,090** | |

**Findings**: All 5 official BIP-32 test vector sets pass (90 individual checks). Invalid paths and out-of-range indices correctly rejected.

### Address Generation (27 Coins)

Verified via test_fuzz_address_bip32_ffi.cpp + test_coins.cpp:
- P2PKH (Base58Check): `1...` prefix for mainnet
- P2WPKH (Bech32): `bc1q...` prefix
- P2TR (Bech32m): `bc1p...` prefix
- WIF round-trip verified
- 27-coin dispatch produces valid addresses for each coin type

---

## 13. Section X -- Performance Baseline

**File: audit_perf.cpp** | **Platform: Linux x86-64, Clang 19, -O3**

| Operation | Iterations | Avg (ns/op) | Throughput |
|-----------|----------:|------------:|-----------:|
| **Field Arithmetic** | | | |
| field_add | 100,000 | 10.4 | 96.3M op/s |
| field_sub | 100,000 | 13.5 | 74.1M op/s |
| field_mul | 100,000 | 43.4 | 23.0M op/s |
| field_sqr | 100,000 | 34.9 | 28.7M op/s |
| field_inv | 10,000 | 736.3 | 1.36M op/s |
| **Scalar Arithmetic** | | | |
| scalar_add | 100,000 | 11.7 | 85.1M op/s |
| scalar_sub | 100,000 | 10.9 | 91.4M op/s |
| scalar_mul | 100,000 | 32.1 | 31.1M op/s |
| scalar_inv | 10,000 | 801.9 | 1.25M op/s |
| **Point Operations** | | | |
| point_add | 10,000 | 200.7 | 4.98M op/s |
| point_dbl | 10,000 | 88.3 | 11.3M op/s |
| scalar_mul (k*P) | 10,000 | 7,096.5 | 140.9K op/s |
| to_compressed | 10,000 | 956.2 | 1.05M op/s |
| **ECDSA** | | | |
| ecdsa_sign | 1,000 | 10,157 | 98.5K op/s |
| ecdsa_verify | 1,000 | 29,493 | 33.9K op/s |
| **Schnorr (BIP-340)** | | | |
| schnorr_sign | 1,000 | 19,710 | 50.7K op/s |
| schnorr_verify | 1,000 | 41,495 | 24.1K op/s |
| **Constant-Time** | | | |
| ct_scalar_mul | 1,000 | 313,350 | 3.19K op/s |
| ct_generator_mul | 1,000 | 316,249 | 3.16K op/s |

**CT overhead**: ~44x for scalar_mul (expected -- fixed iteration count + full table scan).
**Performance regression tracking**: Automated via `benchmark.yml` with 150% alert threshold.

---

## 14. Invariant Catalog Summary

Full invariant catalog: [docs/INVARIANTS.md](INVARIANTS.md)

| Category | Invariants | All Verified |
|----------|----------:|:------------:|
| Field Arithmetic (F1-F17) | 17 | [OK] |
| Scalar Arithmetic (S1-S9) | 9 | [OK] |
| Point / Group (P1-P14) | 14 | [OK] |
| GLV Endomorphism (G1-G4) | 4 | [OK] |
| ECDSA (E1-E8) | 8 | [OK] |
| Schnorr / BIP-340 (B1-B6) | 6 | [OK] |
| MuSig2 (M1-M7) | 7 | [OK] |
| FROST (FR1-FR9) | 9 | [OK] |
| BIP-32 (H1-H7) | 7 | [OK] |
| Address (A1-A6) | 6 | [OK] |
| C ABI (C1-C7) | 7 | [OK] (C7 [!] TSan) |
| Constant-Time (CT1-CT6) | 6 | [OK] (CT5-6 [!] no formal) |
| Batch / Perf (BP1-BP3) | 3 | [OK] |
| Serialization (SP1-SP5) | 5 | [OK] |
| **Total** | **108** | **106 verified, 2 partial** |

**Partial invariants**:
- CT5 (no secret-dependent branches): Verified by code review + dudect, NOT by formal tools
- CT6 (no secret-dependent memory access): Verified by code review + dudect, NOT by formal tools

---

## 15. CI/CD Security Measures

| Measure | Status | Details |
|---------|--------|---------|
| ASan (AddressSanitizer) | [OK] Active | Every push via security-audit.yml |
| UBSan (UndefinedBehaviorSanitizer) | [OK] Active | Every push via security-audit.yml |
| TSan (ThreadSanitizer) | [OK] Active | Every push via security-audit.yml |
| Valgrind Memcheck | [OK] Active | Weekly via security-audit.yml |
| CodeQL (SAST) | [OK] Active | Every push/PR (C/C++ security-and-quality) |
| Clang-Tidy | [OK] Active | Every push/PR (30+ checks) |
| SonarCloud | [OK] Active | Continuous quality + security hotspots |
| OpenSSF Scorecard | [OK] Active | Weekly supply-chain assessment |
| Dependabot | [OK] Active | Automated dependency updates |
| Dependency Review | [OK] Active | PR-level vulnerable dependency scan |
| SLSA Provenance | [OK] Active | Attestation for all release artifacts |
| SHA-256 Checksums | [OK] Active | `SHA256SUMS.txt` in every release |
| Cosign Signing | [OK] Active | Sigstore keyless signing for release binaries |
| SBOM | [OK] Active | CycloneDX 1.6 in every release |
| Reproducible Builds | [OK] Available | `Dockerfile.reproducible` + verification script |
| Docker SHA-pinned | [OK] Active | Digest-pinned base images |
| dudect (smoke) | [OK] Active | Every push/PR (t=25.0 threshold) |
| dudect (full) | [OK] Active | Nightly (30 min, t=4.5 threshold) |
| Nightly differential | [OK] Active | 1.3M+ cross-library checks |
| libFuzzer harnesses | [OK] Available | 3 harnesses for core arithmetic |

---

## 16. Coverage Gaps & Known Limitations

### Acknowledged Gaps

| Gap | Impact | Status |
|-----|--------|--------|
| No formal verification of CT layer | CT properties rely on code review + dudect, not ct-verif/Vale | Planned (long-term) |
| No multi-uarch timing tests | CT may break on specific CPU microarchitectures | Need hardware test farm |
| GPU vs CPU differential | Limited equivalence coverage | PARTIAL (2.6.1-2) |
| CPU vs WASM equivalence | WASM arithmetic may diverge | Not yet tested |
| CPU vs Embedded KAT | ESP32/STM32 runtime tests | Requires physical devices |
| FROST nonce CT | Nonce handling not constant-time audited | Experimental status |
| MuSig2/FROST API stability | API may change before v4.0 | Documented as Experimental |
| Compiler CT trust | Compiler may introduce secret-dependent branches at -O2 | Inherent limitation |

### What We Do NOT Claim

1. **No formal verification** -- CT guarantees are empirical (dudect) and review-based
2. **No hardware side-channel** -- No power analysis, EM emanation, or fault injection testing
3. **No GPU CT** -- All GPU backends are explicitly variable-time
4. **No external audit** -- This is an internal audit only
5. **MuSig2/FROST are experimental** -- Protocol APIs may change

---

## 17. How to Reproduce

### Full Audit Suite (641K checks, ~24s)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CROSS_TESTS=ON \
  -DSECP256K1_BUILD_FUZZ_TESTS=ON \
  -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
cmake --build build -j

# Run all tests
ctest --test-dir build --output-on-failure

# Run audit-only
ctest --test-dir build -L audit --output-on-failure

# Run specific section
ctest --test-dir build -R audit_field -V
ctest --test-dir build -R test_cross_libsecp256k1 -V
ctest --test-dir build -R test_frost_kat -V
```

### Sanitizer Build

```bash
cmake -S . -B build-san -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-san -j
ctest --test-dir build-san --output-on-failure
```

### dudect Side-Channel

```bash
# Smoke (quick, every PR)
ctest --test-dir build -R ct_sidechannel_smoke -V

# Full (30-min statistical run)
./build/tests/test_ct_sidechannel --full
```

### Differential vs libsecp256k1

```bash
# Standard run (~8K checks)
ctest --test-dir build -R test_cross_libsecp256k1 -V

# Extended nightly (~1.3M checks)
SECP256K1_DIFFERENTIAL_MULTIPLIER=100 ./build/tests/test_cross_libsecp256k1
```

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [AUDIT_GUIDE.md](../AUDIT_GUIDE.md) | Auditor navigation guide |
| [AUDIT_REPORT.md](../AUDIT_REPORT.md) | Original v3.9.0 audit report (641K checks) |
| [INVARIANTS.md](INVARIANTS.md) | Complete invariant catalog (108 entries) |
| [TEST_MATRIX.md](TEST_MATRIX.md) | Function -> test coverage map |
| [CT_VERIFICATION.md](CT_VERIFICATION.md) | Constant-time methodology |
| [SECURITY_CLAIMS.md](SECURITY_CLAIMS.md) | FAST vs CT API contract |
| [THREAT_MODEL.md](../THREAT_MODEL.md) | Layer-by-layer risk assessment |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture |
| [BUG_BOUNTY.md](BUG_BOUNTY.md) | Bug bounty scope & rewards |
| [SAFE_DEFAULTS.md](SAFE_DEFAULTS.md) | Recommended production defaults |

---

*UltrafastSecp256k1 v3.14.0 -- Internal Security Audit Report*  
*Generated: 2026-02-25*
