# Test Coverage Matrix

**UltrafastSecp256k1 v3.12.1** — Comprehensive Test Map for Auditors

---

## Summary

| Category | Tests | Status |
|----------|-------|--------|
| **CTest targets** | 20+ | ✅ All passing |
| **Audit suite checks** | 641,194 | ✅ 0 failures |
| **Fuzz harnesses** | 3 | ✅ Active |
| **Side-channel (dudect)** | 1 | ✅ Active |
| **Benchmark suites** | 4+ | ✅ Active |
| **Platform-specific** | 5+ | ✅ Per-platform |

---

## Test File Index

### Audit Suites (`tests/`)

| File | Checks | Focus Area |
|------|--------|------------|
| `audit_field.cpp` | 264,484 | Field arithmetic: identity, commutativity, associativity, distributivity, inverse, boundary, Fermat, special values |
| `audit_scalar.cpp` | 93,847 | Scalar arithmetic: ring properties, overflow, negate, inverse, boundary-near-order |
| `audit_point.cpp` | 116,312 | Point operations: on-curve, group law, add, dbl, scalar_mul, compress/decompress, infinity |
| `audit_ct.cpp` | 120,128 | CT layer: FAST-vs-CT equivalence, complete formulas, no-branch verification |
| `audit_fuzz.cpp` | 15,423 | Fuzz-derived: random inputs through all operation paths |
| `audit_perf.cpp` | — | Performance benchmarks (throughput, latency) |
| `audit_security.cpp` | 17,856 | Security: nonce uniqueness, invalid input rejection, edge-case handling |
| `audit_integration.cpp` | 13,144 | End-to-end: sign→verify, derive→use, full protocol flows |
| `test_ct_sidechannel.cpp` | — | dudect timing: Welch t-test for side-channel leakage |
| `differential_test.cpp` | — | Cross-implementation comparison |
| `bench_ct_vs_libsecp.cpp` | — | Performance comparison with libsecp256k1 |
| `bench_field_ops.cpp` | — | Field operation microbenchmarks |

### CPU Unit Tests (`cpu/tests/`)

| File | Focus Area | Status |
|------|------------|--------|
| `test_comprehensive.cpp` | 25+ categories: field, scalar, point, ECDSA, Schnorr, GLV, SHA, batch, etc. | ✅ |
| `test_arithmetic_correctness.cpp` | Arithmetic correctness: field/scalar edge cases | ✅ |
| `test_ct.cpp` | CT layer correctness (FAST vs CT equivalence) | ✅ |
| `test_ecdsa_schnorr.cpp` | ECDSA (RFC 6979) + Schnorr (BIP-340) vectors | ✅ |
| `test_ecdh_recovery_taproot.cpp` | ECDH, key recovery, Taproot | ✅ |
| `test_bip32.cpp` | BIP-32 HD key derivation | ✅ |
| `test_coins.cpp` | 27-coin address dispatch | ✅ |
| `test_musig2.cpp` | MuSig2 protocol tests | ✅ |
| `test_batch_add_affine.cpp` | Batch affine addition | ✅ |
| `test_multiscalar_batch.cpp` | Multi-scalar multiplication | ✅ |
| `test_simd_batch.cpp` | SIMD batch operations | ✅ |
| `test_mul.cpp` | Multiplication correctness | ✅ |
| `test_large_scalar_multiplication.cpp` | Large scalar multiplication | ✅ |
| `test_field_52.cpp` | 52-bit limb representation | ✅ |
| `test_field_26.cpp` | 26-bit limb representation | ✅ |
| `test_hash_accel.cpp` | SHA-256 acceleration tests | ✅ |
| `test_exhaustive.cpp` | Exhaustive tests (small curves) | ✅ |
| `test_v4_features.cpp` | v4 feature tests | ✅ |
| `run_selftest.cpp` | Selftest runner (smoke/ci/stress) | ✅ |

### Fuzz Harnesses (`cpu/fuzz/`)

| File | Operations Fuzzed | Input |
|------|-------------------|-------|
| `fuzz_field.cpp` | add/sub round-trip, mul identity, square, inverse | 32-byte field element |
| `fuzz_scalar.cpp` | add/sub, mul identity, distributive law | 32-byte scalar |
| `fuzz_point.cpp` | on-curve check, negate, compress round-trip, dbl vs add | 32-byte x-coordinate seed |

### GPU Tests

| File | Backend | Focus |
|------|---------|-------|
| `opencl/tests/test_opencl.cpp` | OpenCL | Kernel correctness |
| `opencl/tests/opencl_extended_test.cpp` | OpenCL | Extended operations |
| `metal/tests/test_metal_host.cpp` | Metal | Metal shader correctness |

---

## API Function → Test Coverage Map

### Field Arithmetic (`FieldElement`)

| Function | audit_field | test_comprehensive | fuzz_field | CT check |
|----------|:-----------:|:-----------------:|:----------:|:--------:|
| `add` / `operator+` | ✅ | ✅ | ✅ | ✅ |
| `sub` / `operator-` | ✅ | ✅ | ✅ | ✅ |
| `mul` / `operator*` | ✅ | ✅ | ✅ | ✅ |
| `square()` | ✅ | ✅ | ✅ | ✅ |
| `inverse()` | ✅ | ✅ | ✅ | ✅ |
| `negate()` | ✅ | ✅ | — | ✅ |
| `from_limbs()` | ✅ | ✅ | — | — |
| `from_bytes()` | ✅ | ✅ | — | — |
| `to_bytes()` | ✅ | ✅ | — | — |
| `from_hex()` / `to_hex()` | ✅ | ✅ | — | — |
| `normalize()` | ✅ | ✅ | ✅ | — |
| `field_select()` | — | — | — | ✅ |
| `square_inplace()` | ✅ | — | — | — |
| `inverse_inplace()` | ✅ | — | — | — |
| `fe_batch_inverse()` | ✅ | ✅ | — | — |

### Scalar Arithmetic (`Scalar`)

| Function | audit_scalar | test_comprehensive | fuzz_scalar | CT check |
|----------|:------------:|:-----------------:|:-----------:|:--------:|
| `add` / `operator+` | ✅ | ✅ | ✅ | ✅ |
| `sub` / `operator-` | ✅ | ✅ | ✅ | ✅ |
| `mul` / `operator*` | ✅ | ✅ | ✅ | ✅ |
| `inverse()` | ✅ | ✅ | — | ✅ |
| `negate()` | ✅ | ✅ | — | ✅ |
| `from_uint64()` | ✅ | ✅ | — | — |
| `from_bytes()` | ✅ | ✅ | — | — |
| `from_hex()` | ✅ | ✅ | — | — |
| `is_zero()` | ✅ | ✅ | — | — |

### Point Operations (`Point`)

| Function | audit_point | test_comprehensive | fuzz_point | CT check |
|----------|:-----------:|:-----------------:|:----------:|:--------:|
| `add()` | ✅ | ✅ | ✅ | ✅ |
| `dbl()` / `double_point()` | ✅ | ✅ | ✅ | ✅ |
| `scalar_mul()` | ✅ | ✅ | — | ✅ |
| `is_on_curve()` | ✅ | ✅ | ✅ | — |
| `is_infinity()` | ✅ | ✅ | — | — |
| `compress()` / `decompress()` | ✅ | ✅ | ✅ | — |
| `to_affine()` | ✅ | ✅ | — | — |
| `generator()` | ✅ | ✅ | — | — |
| `negate()` | ✅ | ✅ | ✅ | — |

### GLV Endomorphism

| Function | audit_point | test_comprehensive | CT check |
|----------|:-----------:|:-----------------:|:--------:|
| `apply_endomorphism()` | ✅ | ✅ | ✅ |
| `verify_endomorphism()` | — | ✅ | — |
| `glv_decompose()` | ✅ | ✅ | ✅ |
| `ct::point_endomorphism()` | — | — | ✅ |

### Signatures

| Function | audit_security | audit_integration | test_ecdsa_schnorr | dudect |
|----------|:-------------:|:-----------------:|:------------------:|:------:|
| `ecdsa::sign()` | ✅ | ✅ | ✅ | ✅ |
| `ecdsa::verify()` | ✅ | ✅ | ✅ | — |
| `schnorr::sign()` | ✅ | ✅ | ✅ | ✅ |
| `schnorr::verify()` | ✅ | ✅ | ✅ | — |

### CT Layer

| Function | audit_ct | test_ct | dudect | Formal |
|----------|:--------:|:-------:|:------:|:------:|
| `ct::field_mul` | ✅ | ✅ | ✅ | ❌ |
| `ct::field_inv` | ✅ | ✅ | ✅ | ❌ |
| `ct::scalar_mul` | ✅ | ✅ | ✅ | ❌ |
| `ct::generator_mul` | ✅ | ✅ | ✅ | ❌ |
| `ct::point_add_complete` | ✅ | ✅ | ✅ | ❌ |
| `ct::point_dbl` | ✅ | ✅ | — | ❌ |

### Protocols (Experimental)

| Function | Test File | Coverage | Notes |
|----------|-----------|----------|-------|
| MuSig2 key aggregation | `test_musig2.cpp` | ✅ Basic | No extended vectors |
| MuSig2 2-round sign | `test_musig2.cpp` | ✅ Basic | Limited edge cases |
| FROST t-of-n | — | ⚠️ **Not tested** | Multi-party simulation needed |
| Adaptor signatures | `test_v4_features.cpp` | ✅ Basic | Limited vectors |
| Pedersen commitments | `test_v4_features.cpp` | ✅ Basic | Limited vectors |
| Taproot (BIP-341) | `test_ecdh_recovery_taproot.cpp` | ✅ Basic | — |
| BIP-32 HD derivation | `test_bip32.cpp` | ✅ | Standard vectors |
| 27-coin dispatch | `test_coins.cpp` | ✅ | Per-coin address format |
| ECDH | `test_ecdh_recovery_taproot.cpp` | ✅ | — |
| Key recovery | `test_ecdh_recovery_taproot.cpp` | ✅ | — |

---

## Coverage Gaps (Transparency)

### High Priority

| Gap | Impact | Blocked By |
|-----|--------|------------|
| **FROST protocol-level tests** | Cannot verify threshold signing correctness | Need multi-party simulation framework |
| **Formal verification** | CT properties unverified mathematically | Fiat-Crypto/ct-verif integration needed |
| **Cross-ABI tests** | Cannot verify FFI correctness across calling conventions | Need multi-compiler test matrix |

### Medium Priority

| Gap | Impact | Status |
|-----|--------|--------|
| MuSig2 extended test vectors | Limited edge-case coverage | Reference impl vectors needed |
| Multi-µarch timing tests | CT may break on specific CPUs | Need hardware test farm |
| FROST nonce CT audit | Nonce handling may leak timing | Requires protocol-level CT analysis |
| GPU vs CPU differential | GPU arithmetic may diverge | Partial coverage via OpenCL tests |

### Low Priority

| Gap | Impact | Status |
|-----|--------|--------|
| WASM-specific tests | WASM arithmetic may diverge | Build-tested, limited runtime tests |
| ESP32/STM32 hardware tests | Embedded correctness | Requires physical devices |
| Adaptor signature extended vectors | Limited coverage | Low usage currently |

---

## Continuous Integration Test Matrix

| Platform | Compiler | Sanitizers | Tests |
|----------|----------|------------|-------|
| Linux x86-64 | GCC 12+ | ASan, UBSan, TSan | Full suite |
| Linux x86-64 | Clang 15+ | ASan, UBSan | Full suite |
| Windows x86-64 | MSVC 2022 | — | Full suite |
| macOS ARM64 | AppleClang | — | Full suite |
| macOS x86-64 | AppleClang | — | Full suite |
| iOS ARM64 | Xcode toolchain | — | Build only |
| Android ARM64 | NDK | — | Build only |
| WASM | Emscripten | — | Build + smoke |
| CUDA | nvcc + host compiler | — | GPU-specific |
| Valgrind | GCC/Clang | Memcheck | Weekly |

---

## Running Tests

```bash
# All CTest targets
ctest --test-dir build --output-on-failure

# Specific audit suite
./build/tests/audit_field
./build/tests/audit_scalar
./build/tests/audit_point
./build/tests/audit_ct

# Side-channel test
./build/tests/test_ct_sidechannel

# Fuzzing (clang required)
clang++ -fsanitize=fuzzer,address -O2 -std=c++20 \
  -I cpu/include cpu/fuzz/fuzz_field.cpp cpu/src/field.cpp \
  -o fuzz_field
./fuzz_field -max_len=64 -runs=10000000

# Selftest (smoke/ci/stress modes)
./build/cpu/tests/run_selftest
```

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Tested with passing checks |
| ⚠️ | Partial or no coverage |
| ❌ | Not implemented |
| — | Not applicable |

---

*UltrafastSecp256k1 v3.12.1 — Test Coverage Matrix*
