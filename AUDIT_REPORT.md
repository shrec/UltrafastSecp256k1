# UltrafastSecp256k1 -- Cryptographic Audit Report

**Library Version:** 3.9.0  
**Audit Date:** 2026-02-11  
**Commit:** `cc20253` (dev)  
**Platform:** Linux x86_64, clang++-19, C++20, Release (-O3)  
**Total Audit Checks:** 641,194  
**Result:** ALL PASSED (0 failures)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Audit Architecture](#2-audit-architecture)
3. [Section I -- Mathematical Correctness](#3-section-i--mathematical-correctness)
   - [I.1 Field Arithmetic](#31-field-arithmetic)
   - [I.2 Scalar Arithmetic](#32-scalar-arithmetic)
   - [I.3 Point Operations & Signatures](#33-point-operations--signatures)
4. [Section II -- Constant-Time & Side-Channel](#4-section-ii--constant-time--side-channel)
5. [Section III -- Fuzzing & Adversarial Testing](#5-section-iii--fuzzing--adversarial-testing)
6. [Section IV -- Performance Validation](#6-section-iv--performance-validation)
7. [Section V -- Security Hardening](#7-section-v--security-hardening)
8. [Section VI -- Integration Testing](#8-section-vi--integration-testing)
9. [Coverage Matrix](#9-coverage-matrix)
10. [How to Run](#10-how-to-run)
11. [Full CTest Summary](#11-full-ctest-summary)

---

## 1. Executive Summary

This report documents a comprehensive cryptographic audit of the **UltrafastSecp256k1** library,
covering 6 audit sections implemented across **8 dedicated test suites**.
The audit validates mathematical correctness, constant-time behavior, adversarial resilience,
performance characteristics, security hardening, and cross-module integration.

| Metric | Value |
|---|---|
| Total audit checks | **641,194** |
| Failures | **0** |
| Audit test suites | 8 |
| Test sections covered | 73 |
| APIs tested | FieldElement, Scalar, Point, ECDSA, Schnorr, Recovery, ECDH, BatchVerify, CT layer |
| Pre-existing tests | 12 (all passing) |
| Total CTest targets | **20 / 20 passing** |

### Per-Suite Summary

| Suite | Checks | Failures | Time |
|---|---:|---:|---|
| audit_field | 264,622 | 0 | 0.29s |
| audit_scalar | 93,215 | 0 | 0.32s |
| audit_point | 116,124 | 0 | 1.71s |
| audit_ct | 120,652 | 0 | 0.93s |
| audit_fuzz | 15,461 | 0 | 0.53s |
| audit_perf | (benchmark) | -- | 1.19s |
| audit_security | 17,309 | 0 | 17.26s |
| audit_integration | 13,811 | 0 | 1.62s |
| **Total** | **641,194** | **0** | **~24s** |

---

## 2. Audit Architecture

### Test Files

All test sources reside in `libs/UltrafastSecp256k1/tests/`:

| File | Section | Label(s) |
|---|---|---|
| `audit_field.cpp` | I.1 Field Arithmetic | audit, crypto |
| `audit_scalar.cpp` | I.2 Scalar Arithmetic | audit, crypto |
| `audit_point.cpp` | I.3 Point Ops & Signatures | audit, crypto |
| `audit_ct.cpp` | II Constant-Time | audit, ct |
| `audit_fuzz.cpp` | III Fuzzing | audit, fuzz |
| `audit_perf.cpp` | IV Performance | audit, perf |
| `audit_security.cpp` | V Security Hardening | audit, security |
| `audit_integration.cpp` | VI Integration | audit, integration |

### Design Principles

- **Deterministic**: Fixed PRNG seeds (`0xA0D17'xxxxx` family) -- same results every run
- **Self-contained**: Each test is a standalone binary, no external data dependencies
- **Zero heap in hot checks**: Test harness itself may allocate; checked code does not
- **Layered coverage**: Random + boundary + adversarial + known-vector + cross-module

### PRNG Seeding

Each suite uses a distinct deterministic seed for reproducibility:

| Suite | Seed |
|---|---|
| audit_field | `0xA0D17'F1E1D` |
| audit_scalar | `0xA0D17'5CA1A` |
| audit_point | `0xA0D17'901E7` |
| audit_ct | `0xA0D17'C71AE` |
| audit_fuzz | `0xA0D17'F022E` |
| audit_security | `0xA0D17'5EC01` |
| audit_integration | `0xA0D17'16780` |

---

## 3. Section I -- Mathematical Correctness

### 3.1 Field Arithmetic

**File:** `audit_field.cpp`  
**Checks:** 264,622  
**Subsections:** 11

| # | Test | Checks | What it validates |
|---|---|---:|---|
| 1 | Addition mod p -- overflow paths | 3,101 | `p-1 + 1`, `p-1 + p-1`, `x + 0`, random pairs |
| 2 | Subtraction borrow-chain | 6,102 | `0 - x`, `x - x == 0`, cross-subtraction-addition consistency |
| 3 | Multiplication carry propagation | 11,102 | Mul-by-1, mul-by-0, commutativity, large operands |
| 4 | Square vs Mul equivalence (10K) | 21,104 | `sqr(x) == mul(x,x)` for 10,000 random elements |
| 5 | Reduction correctness | 22,106 | Above-p values reduce correctly, re-reduction is idempotent |
| 6 | Canonical representation (10K) | 42,106 | `to_bytes(from_bytes(x))` round-trip canonical check |
| 7 | Limb boundary stress | 43,109 | Single-limb set values (0, 1, UINT64_MAX) |
| 8 | Inverse correctness (10K) | 54,110 | `x * inv(x) == 1` for 10,000 random non-zero elements |
| 9 | Square root | 64,110 | `sqrt(x^2) == +-x`, ~50% existence rate on random inputs |
| 10 | Batch inverse | 64,622 | `batch_inv` matches per-element `inv` |
| 11 | Random cross-check (100K) | 264,622 | 100K mixed operations: add, sub, mul, sqr consistency |

**Key Finding:** Square root existence rate was 50.72% (expected ~50%), confirming correct quadratic residue behavior over GF(p).

---

### 3.2 Scalar Arithmetic

**File:** `audit_scalar.cpp`  
**Checks:** 93,215  
**Subsections:** 8

| # | Test | Checks | What it validates |
|---|---|---:|---|
| 1 | Scalar mod n reduction | 10,003 | Values above group order n reduce correctly |
| 2 | Overflow normalization (10K) | 10,003 | `from_bytes -> to_bytes` round-trip preserves canonical form |
| 3 | Edge scalar handling | 10,210 | Scalars: 0, 1, n-1, n, n+1 -- correct reduction |
| 4 | Arithmetic laws (10K) | 60,210 | Commutativity, associativity, distributivity (add, mul) |
| 5 | Scalar inverse (10K) | 71,210 | `s * inv(s) == 1` for random non-zero scalars |
| 6 | GLV split via point arithmetic (1K) | 73,210 | `k*G == k1*G + k2*(lambda*G)` algebraic split correctness |
| 7 | High-bit boundary (2^255) | 73,214 | Scalars near 2^255 reduce/operate correctly |
| 8 | Negate self-consistency (10K) | 93,215 | `s + neg(s) == 0`, `neg(neg(s)) == s` |

**Key Finding:** GLV decomposition verified algebraically through actual point arithmetic,
not just scalar-level checks -- confirming endomorphism correctness.

---

### 3.3 Point Operations & Signatures

**File:** `audit_point.cpp`  
**Checks:** 116,124  
**Subsections:** 11

| # | Test | Checks | What it validates |
|---|---|---:|---|
| 1 | Point at infinity | 7 | Identity element properties for `+`, `*` |
| 2 | Jacobian add (1K+500) | 1,508 | P+Q correctness, associativity sampling |
| 3 | Jacobian double | 1,512 | 2P via `dbl` matches `add(P,P)` |
| 4 | P+P via add (H=0) | 1,612 | Special case: add function handles doubling case |
| 5 | P+(-P) == O (1K) | 3,614 | Point negation -> additive inverse |
| 6 | Affine conversion (1K) | 7,614 | Jacobian->Affine round-trip + on-curve check (y^2=x^3+7) |
| 7 | Scalar mul identities (1K+500) | 9,114 | `1*P==P`, `0*P==O`, `(a+b)*P==a*P+b*P` |
| 8 | Known K*G vectors | 9,124 | NIST/known test vectors for generator multiplication |
| 9 | ECDSA round-trip (1K) | 14,124 | sign -> verify for 1,000 random (key, message) pairs |
| 10 | Schnorr BIP-340 round-trip (1K) | 16,124 | BIP-340 sign -> verify for 1,000 random pairs |
| 11 | 100K point operation stress | 116,124 | Mixed add/dbl/scalar-mul, zero infinity-hit rate |

**Key Findings:**
- Zero infinity hits across 100K random point operations (expected: 0)
- Both ECDSA and Schnorr roundtrips passed with 100% success rate

---

## 4. Section II -- Constant-Time & Side-Channel

**File:** `audit_ct.cpp`  
**Checks:** 120,652  
**Subsections:** 13

| # | Test | Checks | What it validates |
|---|---|---:|---|
| 1 | CT mask generation | 12 | `ct_mask_if`, `ct_select` for 0/1/edge values |
| 2 | CT cmov/cswap (10K) | 30,012 | Conditional move/swap produce correct results |
| 3 | CT table lookup (256-bit) | 30,028 | Table scan vs direct access -- identical results |
| 4 | CT field ops differential (10K) | 81,028 | `ct::field_add/sub/mul/sqr/inv == fast::` equivalents |
| 5 | CT scalar ops differential (10K) | 111,028 | `ct::scalar_add/sub/mul/inv == fast::` equivalents |
| 6 | CT scalar cmov/cswap (1K) | 113,028 | Scalar conditional operations correctness |
| 7 | CT field cmov/cswap/select (1K) | 117,028 | Field conditional operations correctness |
| 8 | CT comparisons | 118,036 | `is_zero`, `eq` on boundary values |
| 9 | CT scalar_mul vs fast (1K) | 119,038 | `ct::scalar_mul(k, G) == fast::scalar_mul(k, G)` |
| 10 | CT complete addition vs fast (1K) | 120,141 | Complete (unified) addition matches fast path |
| 11 | CT byte utilities | 120,151 | `ct_memzero`, `ct_memeq`, `ct_memcpy_if` |
| 12 | CT generator_mul vs fast (500) | 120,651 | `ct::generator_mul == fast::generator_mul` |
| 13 | Timing variance sanity check | 120,652 | Measures k=1 vs k=n-1 timing ratio |

**Timing Measurement:**
- `k=1` average: 363,380 ns
- `k=n-1` average: 351,039 ns
- **Ratio: 1.035** (ideal ~= 1.0, concern threshold > 1.2)

**Note:** This is a statistical sanity check, not a formal side-channel evaluation.
Proper constant-time verification requires tools like `dudect` or hardware timing analysis.

---

## 5. Section III -- Fuzzing & Adversarial Testing

**File:** `audit_fuzz.cpp`  
**Checks:** 15,461  
**Subsections:** 10

| # | Test | Checks | What it validates |
|---|---|---:|---|
| 1 | Malformed public key rejection | 3 | Off-curve points, wrong prefix bytes |
| 2 | Invalid ECDSA signatures | 7 | r=0, s=0, r=n, s=n -- all rejected |
| 3 | Invalid Schnorr signatures | 11 | Corrupted nonce, wrong tag, zero R |
| 4 | Oversized scalars | 15 | Values > n are reduced, not accepted raw |
| 5 | Boundary field elements | 19 | 0, p, p-1, p+1, all-ones |
| 6 | ECDSA recovery edge cases (1K) | 4,769 | Recovery ID sweep, wrong-ID rejection |
| 7 | Random state fuzzing (10K) | 6,461 | 10K random (key, msg) -> sign, verify, no crash |
| 8 | DER round-trip (1K) | 9,461 | ECDSA signatures: DER encode -> decode -> same |
| 9 | Schnorr bytes round-trip (1K) | 11,461 | 64-byte serialization -> deserialization == original |
| 10 | Signature normalization / low-S (1K) | 15,461 | Verify `s` is in lower half after signing |

**Key Finding:** All malformed/adversarial inputs were correctly rejected.
No crashes or undefined behavior observed across 10K random operations.

---

## 6. Section IV -- Performance Validation

**File:** `audit_perf.cpp`  
**Type:** Benchmark (no pass/fail assertions)

### Performance Results

| Operation | Iterations | Avg ns/op | Throughput |
|---|---:|---:|---|
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
| point_scalar_mul | 10,000 | 7,096.5 | 140.9K op/s |
| point_to_compressed | 10,000 | 956.2 | 1.05M op/s |
| **ECDSA** | | | |
| ecdsa_sign | 1,000 | 10,157.3 | 98.5K op/s |
| ecdsa_verify | 1,000 | 29,493.4 | 33.9K op/s |
| **Schnorr BIP-340** | | | |
| schnorr_sign | 1,000 | 19,709.9 | 50.7K op/s |
| schnorr_verify | 1,000 | 41,495.0 | 24.1K op/s |
| **Constant-Time** | | | |
| ct_scalar_mul | 1,000 | 313,350.1 | 3.19K op/s |
| ct_generator_mul | 1,000 | 316,248.5 | 3.16K op/s |

**Performance Characteristics:**
- Field operations: ~23-96M op/s (well-optimized 64-bit limbs)
- ECDSA signing: ~98K op/s; verification: ~34K op/s
- Schnorr (BIP-340): ~51K sign, ~24K verify
- CT scalar_mul is ~44x slower than fast path -- expected for constant-time guarantees
- Point doubling is ~2.3x faster than point addition (expected: fewer field muls)

---

## 7. Section V -- Security Hardening

**File:** `audit_security.cpp`  
**Checks:** 17,309  
**Subsections:** 10

| # | Test | Checks | What it validates |
|---|---|---:|---|
| 1 | Zero/identity key handling | 5 | `inverse(0)` throws; `0*G == O`; zero-key signing fails |
| 2 | Secret zeroization (ct_memzero) | 8 | Memory is zeroed after `ct_memzero` call |
| 3 | Bit-flip resilience (1K) | 2,008 | Single-bit flip in signature -> verify fails |
| 4 | Message bit-flip detection (1K) | 3,008 | Single-bit flip in message -> verify fails |
| 5 | Nonce determinism (RFC 6979) | 3,109 | Same (key, msg) -> same signature; different msg -> different sig |
| 6 | Serialization round-trip (3K) | 10,109 | Compressed, uncompressed, x-only point serialization |
| 7 | Compact recovery serialization (1K) | 12,109 | Compact ECDSA sig -> recover -> matches original pubkey |
| 8 | Double-ops idempotency (2K) | 14,209 | sign-twice == same; verify-twice == same |
| 9 | Cross-algorithm consistency | 14,309 | Same key works for both ECDSA and Schnorr |
| 10 | High-S detection (1K) | 17,309 | Library enforces low-S normalization per BIP-62 |

**Key Findings:**
- Library correctly throws on `inverse(0)` -- no silent zero return
- 100% bit-flip detection rate on both signatures and messages
- RFC 6979 deterministic nonce generation confirmed
- Low-S enforcement verified across 1,000 random signatures

---

## 8. Section VI -- Integration Testing

**File:** `audit_integration.cpp`  
**Checks:** 13,811  
**Subsections:** 10

| # | Test | Checks | What it validates |
|---|---|---:|---|
| 1 | ECDH key exchange symmetry (1K) | 4,001 | `ECDH(a, b*G) == ECDH(b, a*G)` for hashed, x-only, and raw |
| 2 | Schnorr batch verification | 4,006 | 100 valid sigs batch-verify; corrupt detection + identify_invalid |
| 3 | ECDSA batch verification | 4,009 | 100 valid sigs batch-verify; corrupt detection + identify_invalid |
| 4 | ECDSA full round-trip (1K) | 10,009 | sign -> recover pubkey -> verify -> DER encode/decode |
| 5 | Schnorr cross-path (500) | 11,010 | Individual verify == batch verify results |
| 6 | Fast vs CT integration (500) | 12,510 | `fast::scalar_mul == ct::scalar_mul`, ECDSA verify on fast-signed |
| 7 | Combined ECDH + ECDSA protocol (100) | 13,010 | Full key-exchange + signing protocol flow |
| 8 | Multi-key consistency (200) | 13,210 | Aggregated public keys (P1+P2) and individual verifications |
| 9 | Schnorr/ECDSA key consistency (200) | 13,810 | Same keypair signs valid ECDSA and Schnorr |
| 10 | Stress: mixed protocol ops (5K) | 13,811 | 5,000 mixed random operations, 100% success rate |

**Key Findings:**
- ECDH symmetry holds for all three variants (hashed, x-only, raw)
- Batch verification correctly identifies individual invalid signatures
- Cross-path verification (individual vs batch) produces identical results
- Fast and CT paths produce interoperable signatures/points
- 5,000 mixed random operations completed with zero failures

---

## 9. Coverage Matrix

This matrix maps the audit checklist categories to specific test functions and check counts.

| Audit Area | Suites | Functions Tested | Checks |
|---|---|---|---:|
| **Field add/sub/mul/sqr/inv** | field | `field_add`, `field_sub`, `field_mul`, `field_sqr`, `field_inv`, `field_sqrt`, `batch_inv` | 264,622 |
| **Scalar mod/inv/negate/GLV** | scalar | `scalar_add`, `scalar_sub`, `scalar_mul`, `scalar_inv`, `scalar_negate`, `glv_split` | 93,215 |
| **Point add/dbl/scalar_mul** | point | `jac_add`, `jac_dbl`, `scalar_mul`, `to_affine`, on-curve check | 116,124 |
| **ECDSA sign/verify/recover** | point, fuzz, security, integration | `ecdsa_sign`, `ecdsa_verify`, `ecdsa_recover`, DER encoding | ~30,000 |
| **Schnorr BIP-340** | point, fuzz, integration | `schnorr_sign`, `schnorr_verify`, byte round-trip | ~15,000 |
| **Constant-time layer** | ct, integration | All `ct::` functions, cmov, cswap, table lookup, timing | 120,652 |
| **ECDH** | integration | `ecdh_compute`, `ecdh_compute_xonly`, `ecdh_compute_raw` | 4,001 |
| **Batch verification** | integration | `schnorr_batch_verify`, `ecdsa_batch_verify`, `identify_invalid` | ~4,000 |
| **Serialization** | fuzz, security | Compressed, uncompressed, x-only, DER, compact | ~12,000 |
| **Adversarial inputs** | fuzz | Malformed keys, invalid sigs, boundary values | 15,461 |
| **Security hardening** | security | Zeroization, bit-flip, RFC 6979, low-S, cross-algo | 17,309 |
| **Performance** | perf | All operations benchmarked | (benchmark) |

### API Coverage Summary

| API Module | Covered? | Notes |
|---|---|---|
| `FieldElement` | [OK] Full | add, sub, mul, sqr, inv, sqrt, batch_inv, from_bytes, to_bytes, from_limbs |
| `Scalar` | [OK] Full | add, sub, mul, inv, negate, from_hex, to_bytes, glv_split |
| `Point` | [OK] Full | jac_add, jac_dbl, scalar_mul, to_affine, generator, infinity |
| `ECDSA` | [OK] Full | sign, verify, recover, DER encode/decode, compact format |
| `Schnorr` | [OK] Full | sign, verify, 64-byte serialization |
| `ECDH` | [OK] Full | hashed, x-only, raw variants |
| `BatchVerify` | [OK] Full | schnorr_batch_verify, ecdsa_batch_verify, identify_invalid |
| `CT layer` | [OK] Full | ct_ops, ct_field, ct_scalar, ct_point, ct_utils |
| `Recovery` | [OK] Full | All recovery IDs, wrong-ID rejection |
| `FROST` | [!] Not tested | Threshold signature module -- requires multi-party protocol simulation |

---

## 10. How to Run

### Prerequisites

```bash
# Build (from repository root)
cmake -S . -B build_rel -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build_rel -j
```

### Run All Tests (including pre-existing)

```bash
ctest --test-dir build_rel --output-on-failure -j4
# Expected: 20/20 pass
```

### Run Audit Tests Only

```bash
ctest --test-dir build_rel -L audit --output-on-failure -j4
# Expected: 8/8 pass
```

### Run Specific Audit Section

```bash
# By label
ctest --test-dir build_rel -L crypto --output-on-failure    # Field + Scalar + Point
ctest --test-dir build_rel -L ct --output-on-failure         # Constant-time
ctest --test-dir build_rel -L fuzz --output-on-failure       # Fuzzing
ctest --test-dir build_rel -L perf --output-on-failure       # Performance
ctest --test-dir build_rel -L security --output-on-failure   # Security
ctest --test-dir build_rel -L integration --output-on-failure # Integration
```

### Run Individual Test Binary (full output)

```bash
build_rel/tests/audit_field
build_rel/tests/audit_scalar
build_rel/tests/audit_point
build_rel/tests/audit_ct
build_rel/tests/audit_fuzz
build_rel/tests/audit_perf
build_rel/tests/audit_security
build_rel/tests/audit_integration
```

---

## 11. Full CTest Summary

```
Test project /home/shrek/Secp256K1/Secp256K1fast/build_rel

 1/20 Test #18: audit_perf .......................   Passed    1.19 sec
 2/20 Test #20: audit_integration ................   Passed    1.62 sec
 3/20 Test #15: audit_point ......................   Passed    1.71 sec
 4/20 Test #16: audit_ct .........................   Passed    0.93 sec
 5/20 Test #17: audit_fuzz .......................   Passed    0.53 sec
 6/20 Test #14: audit_scalar .....................   Passed    0.32 sec
 7/20 Test #13: audit_field ......................   Passed    0.29 sec
 8/20 Test  #2: batch_add_affine .................   Passed    0.57 sec
 9/20 Test  #1: selftest .........................   Passed    1.15 sec
10/20 Test  #3: hash_accel .......................   Passed    0.39 sec
11/20 Test  #7: comprehensive ....................   Passed    0.33 sec
12/20 Test #12: test_optimized_precomputed .......   Passed    0.30 sec
13/20 Test  #6: exhaustive .......................   Passed    0.31 sec
14/20 Test #11: test_optimized_ops ...............   Passed    0.37 sec
15/20 Test  #4: field_52 .........................   Passed    0.00 sec
16/20 Test  #5: field_26 .........................   Passed    0.00 sec
17/20 Test #10: test_scalar_arithmetic ...........   Passed    0.00 sec
18/20 Test  #8: cuda_selftest ....................   Passed    0.67 sec
19/20 Test  #9: opencl_selftest ..................   Passed    0.81 sec
20/20 Test #19: audit_security ...................   Passed   17.26 sec

100% tests passed, 0 tests failed out of 20

Label Time Summary:
audit          =  23.91 sec*proc (8 tests)
crypto         =   2.30 sec*proc (3 tests)
ct             =   0.93 sec*proc (1 test)
fuzz           =   0.55 sec*proc (1 test)
integration    =   1.65 sec*proc (1 test)
perf           =   1.22 sec*proc (1 test)
security       =  17.26 sec*proc (1 test)

Total Test time (real) =  17.26 sec
```

---

*Generated by automated audit pipeline. Full raw output: `tests/audit_results.txt`*
