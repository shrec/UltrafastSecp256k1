# Test Coverage Matrix

**UltrafastSecp256k1 current repository state** -- Comprehensive Test Map for Auditors

The owner-grade audit bundle and `scripts/validate_assurance.py` are the
authoritative source for live target counts. When a narrative summary below
lags behind the generated validation surfaces, prefer the generated counts.

---

## Summary

| Category | Tests | Status |
|----------|-------|--------|
| **CTest targets** | 162 active targets | [OK] Documented in current assurance validation |
| **Audit suite checks** | 641,194+ | [OK] 0 failures |
| **Exploit PoC test files** | **78 tests, 14 categories** | [OK] 0 failures |
| **Fuzz harnesses** | 3 | [OK] Active |
| **ECIES regression** | 85 | [OK] All passing |
| **Adversarial protocol** | 114 functions, 360+ checks | [OK] Active |
| **Side-channel (dudect)** | 1 | [OK] Active |
| **Benchmark suites** | 4+ | [OK] Active |
| **Platform-specific** | 5+ | [OK] Per-platform |

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
| `audit_perf.cpp` | -- | Performance benchmarks (throughput, latency) |
| `audit_security.cpp` | 17,856 | Security: nonce uniqueness, invalid input rejection, edge-case handling |
| `audit_integration.cpp` | 13,144 | End-to-end: sign->verify, derive->use, full protocol flows |
| `test_ct_sidechannel.cpp` | -- | dudect timing: Welch t-test for side-channel leakage |
| `differential_test.cpp` | -- | Cross-implementation comparison |
| `test_ecies_regression.cpp` | 85 | ECIES hardening: parity tamper, invalid prefix, truncated envelope, tamper matrix, KAT, ABI prefix rejection, pubkey parser consistency, RNG fail-closed |
| `test_adversarial_protocol.cpp` | 114 functions, 360+ checks | Adversarial protocol: MuSig2 (nonce reuse/replay, rogue-key, transcript mutation, signer ordering, malicious aggregator), FROST (below-threshold, malformed commitment, malicious coordinator, duplicate nonce), Silent Payments, ECDSA adaptor (round-trip, transcript mismatch, extraction misuse), Schnorr adaptor, DLEQ (malformed proof, wrong generators), BIP-32, FFI hostile-caller (null args, undersized buffers, overlapping buffers, malformed counts), **New ABI edge cases H.1-H.12** (ctx_size, AEAD, ECIES, EllSwift, ETH address, Pedersen switch, Schnorr adaptor extract, batch sign, BIP-143, BIP-144, SegWit, Taproot sighash), **Remaining ABI surface I.1-I.5** (ctx_clone, last_error_msg, pubkey_parse, pubkey_create_uncompressed, ecdsa_sign_recoverable, ecdsa_recover, ecdsa_sign_verified, schnorr_sign_verified, batch verify deep) |
| `test_fuzz_parsers.cpp` | 10K/suite | Parser fuzz: DER, Schnorr sig, compressed/uncompressed pubkey round-trip |
| `test_fuzz_address_bip32_ffi.cpp` | 10K/suite | Address/BIP-32/FFI fuzz: P2PKH/P2WPKH/P2TR/WIF, BIP-32 paths, BIP-39, coin derivation, FFI boundaries |
| `bench_ct_vs_libsecp.cpp` | -- | Performance comparison with libsecp256k1 |
| `bench_field_ops.cpp` | -- | Field operation microbenchmarks |
| `test_abi_gate.cpp` | -- | ABI compatibility gate: version checks, symbol presence, struct sizes |
| `test_batch_randomness.cpp` | -- | Batch randomness: nonce independence, distribution, uniqueness |
| `test_carry_propagation.cpp` | -- | Carry propagation: field arithmetic edge cases across limb boundaries |
| `test_cross_libsecp256k1.cpp` | -- | Cross-implementation: differential test against bitcoin-core/secp256k1 |
| `test_cross_platform_kat.cpp` | -- | Cross-platform known-answer tests: deterministic outputs across architectures |
| `test_debug_invariants.cpp` | -- | Debug invariants: internal consistency checks under debug mode |
| `test_fiat_crypto_linkage.cpp` | -- | Fiat-Crypto linkage: formal arithmetic verification vectors |
| `test_frost_kat.cpp` | -- | FROST t-of-n threshold signing known-answer tests |
| `test_wycheproof_ecdsa.cpp` | -- | Wycheproof ECDSA: Google Project Wycheproof test vectors |
| `test_wycheproof_ecdh.cpp` | -- | Wycheproof ECDH: Google Project Wycheproof test vectors |
| `unified_audit_runner.cpp` | 70 modules | Unified audit: all 70 audit modules in single binary (includes GPU null-guard paths) |

### CPU Unit Tests (`cpu/tests/`)

| File | Focus Area | Status |
|------|------------|--------|
| `test_comprehensive.cpp` | 25+ categories: field, scalar, point, ECDSA, Schnorr, GLV, SHA, batch, etc. | [OK] |
| `test_arithmetic_correctness.cpp` | Arithmetic correctness: field/scalar edge cases | [OK] |
| `test_ct.cpp` | CT layer correctness (FAST vs CT equivalence) | [OK] |
| `test_ecdsa_schnorr.cpp` | ECDSA (RFC 6979) + Schnorr (BIP-340) vectors | [OK] |
| `test_ecdh_recovery_taproot.cpp` | ECDH, key recovery, Taproot | [OK] |
| `test_bip32.cpp` | BIP-32 HD key derivation | [OK] |
| `test_bip39.cpp` | BIP-39 mnemonic: PBKDF2, wordlist, entropy, validation, seed derivation (57 tests) | [OK] |
| `test_coins.cpp` | 28-coin address dispatch + P2SH/P2SH-P2WPKH/CashAddr | [OK] |
| `test_wallet.cpp` | Wallet API: key management, signing, address formats, recovery | [OK] |
| `test_ethereum.cpp` | Ethereum signing: EIP-155, EIP-191, ecrecover, personal_sign | [OK] |
| `test_musig2.cpp` | MuSig2 protocol tests | [OK] |
| `test_batch_add_affine.cpp` | Batch affine addition | [OK] |
| `test_multiscalar_batch.cpp` | Multi-scalar multiplication | [OK] |
| `test_simd_batch.cpp` | SIMD batch operations | [OK] |
| `test_mul.cpp` | Multiplication correctness | [OK] |
| `test_large_scalar_multiplication.cpp` | Large scalar multiplication | [OK] |
| `test_field_52.cpp` | 52-bit limb representation | [OK] |
| `test_field_26.cpp` | 26-bit limb representation | [OK] |
| `test_zk.cpp` | ZK proofs: knowledge, DLEQ, Bulletproof, batch (24 tests) | [OK] |
| `test_hash_accel.cpp` | SHA-256 acceleration tests | [OK] |
| `test_exhaustive.cpp` | Exhaustive tests (small curves) | [OK] |
| `test_v4_features.cpp` | v4 feature tests | [OK] |
| `run_selftest.cpp` | Selftest runner (smoke/ci/stress) | [OK] |
| `test_ecc_properties.cpp` | ECC algebraic properties: associativity, commutativity, distributivity | [OK] |
| `test_edge_cases.cpp` | Edge cases: scalar zero, infinity arithmetic, BIP-32 IL>=n, cache corruption | [OK] |
| `test_point_edge_cases.cpp` | Point edge cases: infinity, Z=0 guards, roundtrip encoding | [OK] |

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
| `opencl/src/opencl_audit_runner.cpp` | OpenCL | Unified GPU audit (27 modules, 8 sections) |
| `metal/tests/test_metal_host.cpp` | Metal | Metal shader correctness |
| `metal/src/metal_audit_runner.mm` | Metal | `secp256k1_metal_audit`: unified GPU audit (27 modules, 8 sections) |
| `cuda/src/test_ct_smoke.cu` | CUDA | CT smoke tests incl. ZK knowledge + DLEQ prove/verify (9 tests) |
| `cuda/src/test_suite.cu` | CUDA | `cuda_selftest`: kernel correctness, field + scalar + point ops |
| `cuda/src/gpu_audit_runner.cu` | CUDA | `gpu_audit`: unified GPU audit (27 modules, 8 sections) |
| `metal/app/metal_test.mm` | Metal | `secp256k1_metal_test`: shader correctness, compute pipeline |
| `metal/app/bench_metal.mm` | Metal | `secp256k1_metal_bench_full`: comprehensive Metal benchmark |
| `compat/libsecp256k1_shim/tests/shim_test.cpp` | CPU | `secp256k1_shim_test`: libsecp256k1 API compatibility shim |
| `audit/test_gpu_abi_gate.cpp` | GPU (all) | `gpu_abi_gate`: GPU C ABI surface test -- discovery, lifecycle, NULL safety, error strings, generator_mul equivalence |
| `audit/test_gpu_ops_equivalence.cpp` | GPU (all) | `gpu_ops_equivalence`: GPU vs CPU reference for all 6 first-wave ops (skips UNSUPPORTED) |
| `audit/test_gpu_host_api_negative.cpp` | GPU (all) | `gpu_host_api_negative`: NULL ptrs, count=0 no-ops, invalid backend/device, error strings |
| `audit/test_gpu_backend_matrix.cpp` | GPU (all) | `gpu_backend_matrix`: backend enumeration, device info sanity, per-backend op probing |

### Additional CTest Targets

These standalone CTest entries are part of the active validation surface and are tracked explicitly here so the documented matrix matches the real runner surface.

| CTest target | Scope | Notes |
|-------------|-------|-------|
| `audit_invariants` | CPU audit | Algebraic and serialization invariant checks across core arithmetic and point paths |
| `audit_secure_erase` | Security audit | Verifies zeroization / secure erase behavior remains wired and callable |
| `bip141_143_144` | Bitcoin protocol | SegWit serialization and BIP-141/BIP-143/BIP-144 correctness coverage |
| `bip342` | Bitcoin protocol | Taproot / Tapscript BIP-342 correctness coverage |
| `c_abi_negative` | C ABI hostile-caller | Negative-path checks for invalid inputs, malformed pointers, and fail-closed behavior |
| `c_abi_thread_stress` | C ABI concurrency | Threaded stress coverage for public ABI entry points |
| `exploit_ecdsa_der_confusion` | Exploit PoC | Ensures DER parser behavior rejects non-canonical or ambiguous encodings |
| `exploit_frost_commitment_reuse` | Exploit PoC | Adversarial FROST nonce/commitment reuse regression coverage |
| `exploit_pedersen_adversarial` | Exploit PoC | Adversarial Pedersen commitment misuse/regression coverage |
| `ffi_coverage` | FFI surface | Coverage-oriented validation for public foreign-function interface paths |
| `kat_all_operations` | Known-answer tests | Broad deterministic vectors across exposed operations |
| `nonce_uniqueness` | Security audit | Nonce uniqueness and replay-resistance regression coverage |
| `secp256k1_spec` | Spec conformance | Specification-oriented secp256k1 behavior checks |

### Supplemental Active CTest Targets

The following active CTest targets are also part of the documented validation
surface and are named here explicitly so the matrix matches the generated
CTest inventory exactly:

- `[=[metal_host_test]=]`
- `audit_ct_namespace`
- `bip324_transport`
- `bip32_vectors`
- `bip340_strict`
- `bip340_vectors`
- `ct_equivalence`
- `ct_sidechannel_smoke`
- `ct_verif_formal`
- `diag_scalar_mul`
- `exploit_bip324_counter_desync`
- `exploit_bip324_transcript_splice`
- `exploit_ethereum_differential`
- `exploit_musig2_transcript_fork`
- `exploit_zk_adversarial`
- `fault_injection`
- `fiat_crypto_vectors`
- `fuzz_musig2_frost`
- `gpu_ct_smoke`
- `musig2_bip327_vectors`
- `musig2_frost`
- `musig2_frost_advanced`
- `opencl_selftest`
- `parse_strictness`
- `rfc6979_vectors`
- `secp256k1_ecdh_example`
- `secp256k1_ecdsa_example`
- `secp256k1_ellswift_example`
- `secp256k1_musig_example`
- `secp256k1_schnorr_example`

---

## API Function -> Test Coverage Map

### Field Arithmetic (`FieldElement`)

| Function | audit_field | test_comprehensive | fuzz_field | CT check |
|----------|:-----------:|:-----------------:|:----------:|:--------:|
| `add` / `operator+` | [OK] | [OK] | [OK] | [OK] |
| `sub` / `operator-` | [OK] | [OK] | [OK] | [OK] |
| `mul` / `operator*` | [OK] | [OK] | [OK] | [OK] |
| `square()` | [OK] | [OK] | [OK] | [OK] |
| `inverse()` | [OK] | [OK] | [OK] | [OK] |
| `negate()` | [OK] | [OK] | -- | [OK] |
| `from_limbs()` | [OK] | [OK] | -- | -- |
| `from_bytes()` | [OK] | [OK] | -- | -- |
| `to_bytes()` | [OK] | [OK] | -- | -- |
| `from_hex()` / `to_hex()` | [OK] | [OK] | -- | -- |
| `normalize()` | [OK] | [OK] | [OK] | -- |
| `field_select()` | -- | -- | -- | [OK] |
| `square_inplace()` | [OK] | -- | -- | -- |
| `inverse_inplace()` | [OK] | -- | -- | -- |
| `fe_batch_inverse()` | [OK] | [OK] | -- | -- |

### Scalar Arithmetic (`Scalar`)

| Function | audit_scalar | test_comprehensive | fuzz_scalar | CT check |
|----------|:------------:|:-----------------:|:-----------:|:--------:|
| `add` / `operator+` | [OK] | [OK] | [OK] | [OK] |
| `sub` / `operator-` | [OK] | [OK] | [OK] | [OK] |
| `mul` / `operator*` | [OK] | [OK] | [OK] | [OK] |
| `inverse()` | [OK] | [OK] | -- | [OK] |
| `negate()` | [OK] | [OK] | -- | [OK] |
| `from_uint64()` | [OK] | [OK] | -- | -- |
| `from_bytes()` | [OK] | [OK] | -- | -- |
| `from_hex()` | [OK] | [OK] | -- | -- |
| `is_zero()` | [OK] | [OK] | -- | -- |

### Point Operations (`Point`)

| Function | audit_point | test_comprehensive | fuzz_point | CT check |
|----------|:-----------:|:-----------------:|:----------:|:--------:|
| `add()` | [OK] | [OK] | [OK] | [OK] |
| `dbl()` / `double_point()` | [OK] | [OK] | [OK] | [OK] |
| `scalar_mul()` | [OK] | [OK] | -- | [OK] |
| `is_on_curve()` | [OK] | [OK] | [OK] | -- |
| `is_infinity()` | [OK] | [OK] | -- | -- |
| `compress()` / `decompress()` | [OK] | [OK] | [OK] | -- |
| `to_affine()` | [OK] | [OK] | -- | -- |
| `generator()` | [OK] | [OK] | -- | -- |
| `negate()` | [OK] | [OK] | [OK] | -- |

### GLV Endomorphism

| Function | audit_point | test_comprehensive | CT check |
|----------|:-----------:|:-----------------:|:--------:|
| `apply_endomorphism()` | [OK] | [OK] | [OK] |
| `verify_endomorphism()` | -- | [OK] | -- |
| `glv_decompose()` | [OK] | [OK] | [OK] |
| `ct::point_endomorphism()` | -- | -- | [OK] |

### Signatures

| Function | audit_security | audit_integration | test_ecdsa_schnorr | dudect |
|----------|:-------------:|:-----------------:|:------------------:|:------:|
| `ecdsa::sign()` | [OK] | [OK] | [OK] | [OK] |
| `ecdsa::verify()` | [OK] | [OK] | [OK] | -- |
| `schnorr::sign()` | [OK] | [OK] | [OK] | [OK] |
| `schnorr::verify()` | [OK] | [OK] | [OK] | -- |

### CT Layer

CT functions are verified by a layered approach: equivalence tests (`audit_ct`, `test_ct`),
statistical timing tests (`dudect`), and deterministic CT verification (`ct-verif` + `valgrind-ct` in CI).
Machine-checked proofs (Fiat-Crypto/Vale/Jasmin) are not yet applied.

| Function | audit_ct | test_ct | dudect | ct-verif | Machine-Checked Proof |
|----------|:--------:|:-------:|:------:|:--------:|:---------------------:|
| `ct::field_mul` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::field_inv` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::scalar_mul` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::generator_mul` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::point_add_complete` | [OK] | [OK] | [OK] | [OK] | -- |
| `ct::point_dbl` | [OK] | [OK] | -- | [OK] | -- |

### Protocols (Experimental)

| Function | Test File | Coverage | Notes |
|----------|-----------|----------|-------|
| MuSig2 key aggregation | `test_musig2.cpp` | [OK] Basic | No extended vectors |
| MuSig2 2-round sign | `test_musig2.cpp` | [OK] Full | Rogue-key, transcript mutation, signer ordering, malicious aggregator adversarial tests added |
| FROST t-of-n | `test_v4_features.cpp` | [OK] Basic | Keygen, sign, aggregate, verify |
| Adaptor signatures | `test_v4_features.cpp` | [OK] Full | Transcript mismatch, extraction misuse, DLEQ malformed proof, wrong generators adversarial tests added |
| Pedersen commitments | `test_v4_features.cpp` | [OK] Basic | Limited vectors |
| ZK Knowledge proof | `test_zk.cpp` | [OK] | Prove/verify, arbitrary base, serialization |
| ZK DLEQ proof | `test_zk.cpp` | [OK] | Prove/verify, cross-basis equality |
| ZK Bulletproof range | `test_zk.cpp` | [OK] | Prove/verify, boundary values, inner product |
| ZK batch range verify | `test_zk.cpp` | [OK] | Multi-proof batch verification |
| GPU ZK Knowledge proof | `test_ct_smoke.cu` | [OK] | CT prove + fast-path verify on CUDA |
| GPU ZK DLEQ proof | `test_ct_smoke.cu` | [OK] | CT prove + fast-path verify on CUDA |
| Taproot (BIP-341) | `test_ecdh_recovery_taproot.cpp` | [OK] Basic | -- |
| BIP-32 HD derivation | `test_bip32.cpp` | [OK] | Standard vectors |
| 28-coin dispatch | `test_coins.cpp` | [OK] | Per-coin address format (P2PKH, P2WPKH, P2TR, P2SH-P2WPKH, CashAddr, EIP-55, TRON_BASE58) |
| Wallet API | `test_wallet.cpp` | [OK] | Chain-agnostic key mgmt, signing, recovery |
| Ethereum signing | `test_ethereum.cpp` | [OK] | EIP-155/-191, ecrecover, multi-chain |
| ECDH | `test_ecdh_recovery_taproot.cpp` | [OK] | -- |
| Key recovery | `test_ecdh_recovery_taproot.cpp` | [OK] | -- |

---

## New ABI Surface Edge-Case Coverage (v3.22+ §N)

> Gap analysis found 26 `ufsecp_*` functions with no dedicated edge-case tests.
> All gaps are closed by `test_h1_*`–`test_h12_*` in
> `audit/test_adversarial_protocol.cpp`.

| Test ID | ABI functions | NULL | Zero-count/len | Invalid content | Smoke |
|---------|---------------|:----:|:--------------:|:---------------:|:-----:|
| H.1 | `ufsecp_ctx_size` | -- | -- | -- | [OK] |
| H.2 | `ufsecp_aead_chacha20_encrypt`, `ufsecp_aead_chacha20_decrypt` | [OK] | [OK] | [OK] (bad-tag, wrong-nonce) | [OK] |
| H.3 | `ufsecp_ecies_encrypt`, `ufsecp_ecies_decrypt` | [OK] | -- | [OK] (off-curve, tampered) | [OK] |
| H.4 | `ufsecp_ellswift_create`, `ufsecp_ellswift_xdh` | [OK] | -- | [OK] (zero key) | [OK] |
| H.5 | `ufsecp_eth_address_checksummed`, `ufsecp_eth_personal_hash` | [OK] | [OK] | -- | [OK] |
| H.6 | `ufsecp_pedersen_switch_commit` | [OK] | -- | -- | [OK] |
| H.7 | `ufsecp_schnorr_adaptor_extract` | [OK] | -- | [OK] (zero inputs) | -- |
| H.8 | `ufsecp_ecdsa_sign_batch`, `ufsecp_schnorr_sign_batch` | [OK] | [OK] | -- | -- |
| H.9 | `ufsecp_bip143_sighash`, `ufsecp_bip143_p2wpkh_script_code` | [OK] | -- | -- | [OK] |
| H.10 | `ufsecp_bip144_txid`, `ufsecp_bip144_wtxid`, `ufsecp_bip144_witness_commitment` | [OK] | -- | -- | [OK] |
| H.11 | `ufsecp_is_witness_program`, `ufsecp_parse_witness_program`, `ufsecp_p2wpkh_spk`, `ufsecp_p2wsh_spk`, `ufsecp_p2tr_spk`, `ufsecp_witness_script_hash` | [OK] | -- | [OK] (non-witness) | [OK] |
| H.12 | `ufsecp_taproot_keypath_sighash`, `ufsecp_tapscript_sighash` | [OK] | [OK] | [OK] (OOB index) | [OK] |

---

## Coverage Gaps (Transparency)

### High Priority

| Gap | Impact | Blocked By |
|-----|--------|------------|
| **Machine-checked proofs** | CT/math properties not proven in Coq/Jasmin/Vale-style frameworks | Separate proof-bearing core or generated arithmetic path needed |
| **Cross-ABI tests** | Cannot verify FFI correctness across calling conventions | Need multi-compiler test matrix |

### Medium Priority

| Gap | Impact | Status |
|-----|--------|--------|
| MuSig2 extended test vectors | Full adversarial coverage (A.4-A.7) | Reference impl vectors available via BIP-327 |
| Multi-uarch timing tests | CT may break on specific CPUs | Need hardware test farm |
| GPU vs CPU differential | GPU arithmetic may diverge | Covered by gpu_ops_equivalence (6 ops) + OpenCL/CUDA tests |

### Low Priority

| Gap | Impact | Status |
|-----|--------|--------|
| WASM-specific tests | WASM arithmetic may diverge | Build-tested, limited runtime tests |
| ESP32/STM32 hardware tests | Embedded correctness | Requires physical devices |
| Adaptor signature extended vectors | Full adversarial coverage (D.1-D.6, E.1-E.5) | Transcript mismatch and extraction misuse covered |

---

## Continuous Integration Test Matrix

| Platform | Compiler | Sanitizers | Tests |
|----------|----------|------------|-------|
| Linux x86-64 | GCC 12+ | ASan, UBSan, TSan | Full suite |
| Linux x86-64 | Clang 15+ | ASan, UBSan | Full suite |
| Linux ARM64 | aarch64-linux-gnu + QEMU | -- | Cross-build + `run_selftest smoke` + `test_bip324_standalone` + `bench_kP` + `bench_bip324` |
| Linux RISC-V 64 | riscv64-linux-gnu + QEMU | -- | Cross-build + `run_selftest smoke` + `test_bip324_standalone` + `bench_kP` + `bench_bip324` |
| Windows x86-64 | MSVC 2022 | -- | Full suite |
| macOS ARM64 | AppleClang | -- | Full suite |
| macOS x86-64 | AppleClang | -- | Full suite |
| iOS ARM64 | Xcode toolchain | -- | Build only |
| Android ARM64 | NDK | -- | Build only |
| WASM | Emscripten | -- | Build + smoke |
| CUDA | nvcc + host compiler | -- | GPU-specific |
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

# Linux ARM64 smoke under QEMU (cross-compiled)
bash ./scripts/run-qemu-smoke.sh arm64

# Or run the commands manually
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm64/cpu/run_selftest smoke
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm64/cpu/test_bip324_standalone
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm64/cpu/bench_kP
qemu-aarch64 -L /usr/aarch64-linux-gnu ./build-arm64/cpu/bench_bip324

# Linux RISC-V smoke under QEMU (cross-compiled)
bash ./scripts/run-qemu-smoke.sh riscv64

# Or run the commands manually
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build-riscv64/cpu/run_selftest smoke
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build-riscv64/cpu/test_bip324_standalone
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build-riscv64/cpu/bench_kP
qemu-riscv64 -L /usr/riscv64-linux-gnu ./build-riscv64/cpu/bench_bip324
```

---

## Exploit PoC Test Suite (`audit/test_exploit_*.cpp`)

78 standalone exploit-style tests that actively try to break the library.
Each test compiles as a separate binary and verifies that attacks fail, edge cases are handled, and security invariants hold under adversarial inputs.

| Category | File(s) | Attack / Property Verified |
|----------|---------|---------------------------|
| ECDSA / Signature | `test_exploit_ecdsa_malleability` | BIP-62 low-s enforcement, high-s rejection, `normalize()`, strict parser |
| ECDSA / Signature | `test_exploit_ecdsa_edge_cases` | Zero and boundary inputs |
| ECDSA / Signature | `test_exploit_ecdsa_recovery` | Key recovery edge cases |
| ECDSA / Signature | `test_exploit_ecdsa_rfc6979_kat` | RFC 6979 deterministic nonce KAT |
| ECDH | `test_exploit_ecdh` | ECDH correctness |
| ECDH | `test_exploit_ecdh_degenerate` | Degenerate ECDH inputs |
| ECDH | `test_exploit_ecdh_variants` | ECDH variants |
| Schnorr / BIP-340 | `test_exploit_schnorr_edge_cases` | Schnorr edge cases |
| Schnorr / BIP-340 | `test_exploit_schnorr_bip340_kat` | BIP-340 known-answer tests |
| Batch Schnorr | `test_exploit_batch_schnorr` | Basic batch Schnorr verification |
| Batch Schnorr | `test_exploit_batch_schnorr_forge` | Forge detection, `identify_invalid` accuracy |
| Batch Schnorr | `test_exploit_batch_soundness` | Batch soundness properties |
| GLV / Math | `test_exploit_glv_endomorphism` | Endomorphism properties |
| GLV / Math | `test_exploit_glv_kat` | GLV ±k₁±k₂λ≡k, φ(G)=λG, φ²+φ+1=0 decomposition KAT |
| GLV / Math | `test_exploit_field_arithmetic` | Field element arithmetic |
| GLV / Math | `test_exploit_scalar_group_order` | Scalar group-order properties |
| GLV / Math | `test_exploit_scalar_invariants` | Scalar invariants |
| GLV / Math | `test_exploit_scalar_systematic` | Systematic scalar coverage |
| GLV / Math | `test_exploit_point_group_law` | Point group law |
| GLV / Math | `test_exploit_point_serialization` | Point serialization |
| GLV / Math | `test_exploit_multiscalar` | Multi-scalar multiplication |
| GLV / Math | `test_exploit_pippenger_msm` | Pippenger MSM |
| Batch Verify | `test_exploit_batch_verify_correctness` | Batch verify math |
| BIP-32 / HD | `test_exploit_bip32_depth` | Depth overflow |
| BIP-32 / HD | `test_exploit_bip32_derivation` | Derivation correctness |
| BIP-32 / HD | `test_exploit_bip32_path_overflow` | Path overflow attack |
| BIP-32 / HD | `test_exploit_bip32_ckd_hardened` | Hardened isolation, xpub guard, fingerprint |
| BIP-39 | `test_exploit_bip39_entropy` | Entropy edge cases |
| BIP-39 | `test_exploit_bip39_mnemonic` | Mnemonic generation and parsing |
| HD Derivation | `test_exploit_coin_hd_derivation` | HD derivation paths per coin type |
| MuSig2 | `test_exploit_musig2` | MuSig2 protocol |
| MuSig2 | `test_exploit_musig2_key_agg` | Key aggregation |
| MuSig2 | `test_exploit_musig2_nonce_reuse` | Nonce reuse attack |
| MuSig2 | `test_exploit_musig2_ordering` | Key ordering independence |
| FROST | `test_exploit_frost_byzantine` | Byzantine participant |
| FROST | `test_exploit_frost_dkg` | Distributed key generation |
| FROST | `test_exploit_frost_index` | Participant index handling |
| FROST | `test_exploit_frost_lagrange_duplicate` | Duplicate Lagrange coefficients |
| FROST | `test_exploit_frost_participant_zero` | Index-zero participant |
| FROST | `test_exploit_frost_signing` | FROST signing protocol |
| FROST | `test_exploit_frost_threshold_degenerate` | Degenerate threshold |
| Adaptor / ZK | `test_exploit_adaptor_extended` | Extended adaptor attacks |
| Adaptor / ZK | `test_exploit_adaptor_parity` | Adaptor parity handling |
| Adaptor / ZK | `test_exploit_zk_proofs` | ZK proof properties |
| Adaptor / ZK | `test_exploit_pedersen_homomorphism` | Pedersen commitment homomorphism |
| AEAD / ChaCha20 | `test_exploit_aead_integrity` | ChaCha20-Poly1305 MAC bypass, nonce reuse, zeroed output on failure |
| AEAD / ChaCha20 | `test_exploit_chacha20_kat` | ChaCha20 known-answer tests |
| AEAD / ChaCha20 | `test_exploit_chacha20_nonce_reuse` | Nonce reuse hazard |
| AEAD / ChaCha20 | `test_exploit_chacha20_poly1305` | AEAD roundtrip |
| HKDF | `test_exploit_hkdf_kat` | HKDF known-answer tests |
| HKDF | `test_exploit_hkdf_security` | HKDF security properties |
| Hash primitives | `test_exploit_keccak256_kat` | Keccak-256 KAT |
| Hash primitives | `test_exploit_ripemd160_kat` | RIPEMD-160 KAT |
| Hash primitives | `test_exploit_sha256_kat` | SHA-256 KAT |
| Hash primitives | `test_exploit_sha512_kat` | SHA-512 KAT |
| Hash primitives | `test_exploit_sha_kat` | SHA family KAT |
| ECIES | `test_exploit_ecies_auth` | ECIES authentication |
| ECIES | `test_exploit_ecies_encryption` | ECIES encryption |
| ECIES | `test_exploit_ecies_roundtrip` | ECIES roundtrip |
| Protocol BIPs | `test_exploit_bip143_sighash` | BIP-143 sighash |
| Protocol BIPs | `test_exploit_bip144_serialization` | BIP-144 serialization |
| Protocol BIPs | `test_exploit_bip324_session` | BIP-324 encrypted P2P session |
| Protocol BIPs | `test_exploit_segwit_encoding` | SegWit address encoding |
| Protocol BIPs | `test_exploit_taproot_scripts` | Taproot script path |
| Protocol BIPs | `test_exploit_taproot_tweak` | Taproot key tweak |
| Address / Wallet | `test_exploit_address_encoding` | Address encoding |
| Address / Wallet | `test_exploit_address_generation` | Address generation |
| Address / Wallet | `test_exploit_wallet_api` | Wallet API |
| Address / Wallet | `test_exploit_private_key` | Private key handling |
| Address / Wallet | `test_exploit_eth_signing` | Ethereum signing |
| Address / Wallet | `test_exploit_bitcoin_message_signing` | Bitcoin message signing |
| Constant-Time | `test_exploit_ct_recov` | CT key recovery |
| Constant-Time | `test_exploit_ct_systematic` | Systematic CT verification |
| Constant-Time | `test_exploit_backend_divergence` | Backend divergence detection |
| ElligatorSwift | `test_exploit_ellswift` | ElligatorSwift encoding correctness |
| ElligatorSwift | `test_exploit_ellswift_ecdh` | ElligatorSwift ECDH |
| Self-Test / API | `test_exploit_selftest_api` | Self-test API |
| Recovery | `test_exploit_recovery_extended` | Extended recovery edge cases |

Build and run all exploit tests:
```bash
cmake -S . -B build-audit -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_TESTS=ON
cmake --build build-audit -j
ctest --test-dir build-audit -R "exploit" --output-on-failure
```

---

## Remaining ABI Surface Edge-Case Coverage (v3.23+ §I/§O)

| ID  | Functions | NULL args | Invalid inputs | Valid round-trip |
|-----|-----------|-----------|----------------|------------------|
| I.1 | `ufsecp_ctx_clone`, `ufsecp_last_error_msg`, `ufsecp_last_error` | [OK] | [OK] (error state) | [OK] (independent clone) |
| I.2 | `ufsecp_pubkey_parse`, `ufsecp_pubkey_create_uncompressed` | [OK] | [OK] (bad len, bad prefix, zero key) | [OK] (uncompressed→compressed normalisation) |
| I.3 | `ufsecp_ecdsa_sign_recoverable`, `ufsecp_ecdsa_recover` | [OK] | [OK] (zero key, bad recid) | [OK] (recovered pubkey matches original) |
| I.4 | `ufsecp_ecdsa_sign_verified`, `ufsecp_schnorr_sign_verified` | [OK] | [OK] (zero key) | [OK] (outputs verify via _verify counterpart) |
| I.5 | `ufsecp_schnorr_batch_verify`, `ufsecp_ecdsa_batch_verify`, `ufsecp_schnorr_batch_identify_invalid`, `ufsecp_ecdsa_batch_identify_invalid` | [OK] | [OK] (tampered sig) | [OK] (valid entry verifies; identify_invalid returns correct index) |

---

## Legend

| Symbol | Meaning |
|--------|---------|
| [OK] | Tested with passing checks |
| [!] | Partial or no coverage |
| [FAIL] | Not implemented |
| -- | Not applicable |

---

*UltrafastSecp256k1 v3.22.0 -- Test Coverage Matrix*
