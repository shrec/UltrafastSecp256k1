# UltrafastSecp256k1 v3.16.0 Release Notes

> **Release Date:** 2026-03-01
> **Commit:** 28a40d0 (main)
> **Previous Release:** v3.14.0
> **Stats:** 115 commits | 411 files changed | +53,010 / -8,685 lines
> **ABI:** Compatible -- drop-in upgrade from v3.14.0 | SOVERSION unchanged
> **Breaking Changes:** None

---

## Highlights

- **ECDSA Recovery 1.9x speedup** -- `ecdsa_recover()` rewritten with single `dual_scalar_mul_gen_point` using 4-stream GLV Strauss (36us vs 69us)
- **BIP-340 strict parsing** -- new `parse_bytes_strict` APIs reject all malformed inputs
- **OpenSSF Scorecard hardening** -- all GitHub Actions pinned to full SHA, harden-runner on all jobs
- **FROST RFC 9591 invariant tests** -- 7 protocol invariants + exhaustive 3-of-5 subset verification
- **~5,150 code scanning alerts fixed** -- clang-tidy, Cppcheck, CodeQL, SonarCloud
- **CT verification CI** -- Valgrind taint, ct-verif LLVM pass, ARM64 dudect, MuSig2/FROST dudect
- **48/49 audit modules PASS** (1 advisory dudect timing on shared runners)
- **12-language binding support** -- Java, Swift, React Native, Python, Rust, Dart, Go, Node.js, C#, Ruby, PHP, C

---

## 1. Security Hardening

### BIP-340 Strict Parsing (v3.16.0)
- `Scalar::parse_bytes_strict` -- rejects zero and >= group order
- `FieldElement::parse_bytes_strict` -- rejects zero and >= field prime
- `SchnorrSignature::parse_strict` -- validates r < p and s < n
- C ABI functions (`ufsecp_schnorr_verify`, `ufsecp_schnorr_sign`, `ufsecp_xonly_pubkey_parse`) now use strict parsing internally
- 31-test BIP-340 strict suite: reject-zero, reject-overflow, reject-p-plus, accept-valid

### Constant-Time Erasure (v3.16.0)
- `ct::schnorr_sign` and `ct::ecdsa_sign` erase intermediate nonces via volatile function-pointer trick (matches libsecp256k1)
- lift_x deduplication -- single `static lift_x()` replaces duplicated code
- Y-parity fix -- `limbs()[0] & 1` instead of byte-level parity check

### CT Branchless Hardening (v3.15.0)
- Branchless `scalar_window` on RISC-V, branched on x86/ARM
- `value_barrier` after mask derivation in `ct_compare` + WASM KAT target
- `is_zero_mask` -- RISC-V branchless asm + triple barrier + rdcycle fence
- Reverse-scan `ct_compare` with interleaved test data pattern

### GLV Decomposition Fix (v3.13.1 -- Critical)
- `ct_mul_256x_lo128_mod` overflow when GLV's c1/c2 rounded to 2^128
- `lambda*k2` only read 2 lower limbs -- silently dropped `limb[2]=1`
- Replaced with full `ct_scalar_mul_mod_n()`: 4x4 schoolbook -> 8-limb product -> 3-phase `reduce_512`
- `minus_b2` changed from 128-bit to full 256-bit `Scalar(n - b2)`
- CT scalar_mul overhead vs fast path: 1.05x (25.3us vs 24.0us)

---

## 2. Performance

### ECDSA Recovery 1.9x Speedup (v3.15.0)
- Replaced 3 separate scalar multiplications (`s*R`, `z*G`, `r^-1 * result`) with single `dual_scalar_mul_gen_point(u1, u2, R)` using 4-stream GLV Strauss with interleaved wNAF
- Recovery: ~36us (was ~69us)

### Platform Assembly (v3.15.0)
- ARM64: CSEL branchless conditionals, sqr EXTR optimization for field squaring
- RISC-V: preload optimization for field multiply assembly, reduced register pressure
- lift_x() parity: direct `limbs()[0] & 1` instead of `to_bytes()` serialization

### Apple-to-Apple Benchmark (v3.15.0)
- `bench_apple_to_apple` vs libsecp256k1 v0.6.0: 13 operations, same compiler/flags/assembly, IQR outlier removal, median of 11 passes
- Result: 7 FASTER, 5 EQUAL, 0 SLOWER (geometric mean 0.68x = UF 1.47x faster overall)

---

## 3. WASM/Emscripten Support (v3.15.0)

- `SECP256K1_NO_INT128` -- automatic define on Emscripten
- `SECP256K1_FAST_52BIT` -- disabled for Emscripten
- Precompute generator bypass -- avoids timeouts on WASM
- GLV+Shamir fallback -- wNAF w=5 replaced with optimal double-and-add
- KAT test -- SINGLE_FILE=1 + ESM conflict resolution

---

## 4. CI/CD Infrastructure

### OpenSSF Scorecard Hardening (v3.16.0)
- All GitHub Actions pinned to full SHA (codeql-action v4.32.4, upload-artifact v6.0.0)
- `harden-runner` on all jobs (discord-commits, packaging RPM)
- `persist-credentials: false` on all checkout steps with write permissions
- 13 workflow files audited and standardized
- pip deps pinned by hash + cosign signing visibility

### CT Verification CI (v3.16.0)
- `ct-arm64.yml` -- native ARM64 / Apple Silicon dudect (macos-14 M1): smoke per-PR + full nightly
- `ct-verif.yml` -- compile-time CT verification via ct-verif LLVM pass
- `valgrind-ct.yml` -- Valgrind MAKE_MEM_UNDEFINED taint analysis: detects secret-dependent branches
- MuSig2/FROST dudect -- protocol-level timing tests

### Additional CI (v3.15.0)
- ClusterFuzzLite integration + UBSan vptr sanitizer compatibility
- Cppcheck + Mutation testing + SARIF integration
- bench-regression.yml -- per-commit performance regression gate (120% threshold)
- audit-report.yml -- SARIF v2.1.0 upload to GitHub Code Scanning
- packaging.yml -- release workflow race condition fix

### Local CI (Docker) (v3.16.0)
- docker-compose.ci.yml -- single-command orchestration for 14 CI jobs
- pre-push target -- `docker compose run --rm pre-push` in ~5 min
- audit job mirrors audit-report.yml (GCC-13 + Clang-17)
- ccache volume persistence + PowerShell wrapper

---

## 5. Code Quality (~5,150 alerts fixed)

- ~4,600 code scanning alerts (clang-tidy + cppcheck + CodeQL) -- batch 1
- ~550 code scanning alerts -- batch 2
- 136 code scanning alerts -- batch 3 (bench_hornet, glv, audit_integration, point, precompute)
- Duplicate const qualifiers -- GCC-13 build failure fix
- using declarations restored -- clang-tidy removed declarations needed for MSVC/ESP32/WASM
- SonarCloud Quality Gate:
  - SHA-256 S3519 buf_ overflow false positive suppression
  - Coverage: 61.8% -> 85.8%
  - Duplication: 3.3% -> below threshold
  - cpp:S876 CT masking unsigned negation suppression

---

## 6. Audit Framework

### Audit Modules (49 total, 48/49 PASS + 1 advisory)

**Section 1 -- Mathematical Invariants (Fp, Zn, Group Laws): 13/13 PASS**
- Field Fp deep audit (add/mul/inv/sqrt/batch)
- Scalar Zn deep audit (mod/GLV/edge/inv)
- Point ops deep audit (Jac/affine/sigs)
- Field & scalar arithmetic
- Arithmetic correctness
- Scalar multiplication
- Exhaustive algebraic verification
- Comprehensive 500+ suite
- ECC property-based invariants
- Affine batch addition
- Carry chain stress (limb boundary)
- FieldElement52 (5x52) vs 4x64
- FieldElement26 (10x26) vs 4x64

**Section 2 -- Constant-Time & Side-Channel: 4/5 PASS (1 advisory)**
- CT deep audit (masks/cmov/cswap/timing)
- Constant-time layer
- FAST == CT equivalence
- Side-channel dudect (smoke) -- ADVISORY (timing flakes on shared runners)
- CT scalar_mul vs fast (diagnostic)

**Section 3 -- Differential & Cross-Library: 3/3 PASS**
- Differential correctness
- Cross-library (vs bitcoin-core/libsecp256k1)
- Fiat-Crypto field vectors (752 checks)

**Section 4 -- Standard Test Vectors: 6/6 PASS**
- BIP-340 Schnorr
- RFC-6979 ECDSA
- BIP-32 HD key derivation
- BIP-340 strict parsing (31 tests)
- Cross-platform KAT
- Debug invariants

**Section 5 -- Fuzzing & Adversarial: 4/4 PASS**
- Audit fuzz
- Parser fuzz (DER + Schnorr + Pubkey)
- Address/BIP32/FFI fuzz
- Fault injection

**Section 6 -- Protocol Security: 9/9 PASS**
- ECDSA sign/verify round-trip
- Schnorr sign/verify
- MuSig2 basic protocol
- FROST basic protocol
- MuSig2 + FROST advanced/adversarial
- RFC 9591 protocol invariants (7 invariants)
- RFC 9591 3-of-5 exhaustive (C(5,3)=10 subsets)
- BIP-327 MuSig2 reference vectors (35 tests)
- FFI round-trip boundary (103 tests)

**Section 7 -- ABI & Memory Safety: 4/4 PASS**
- ABI version gate (compile-time)
- Zeroization + hardening
- Security integration
- Hash acceleration

**Section 8 -- Performance: 4/4 PASS**
- Performance smoke (sign/verify)
- Batch operations
- Multi-scalar mul
- Shamir's trick

### Audit UX (v3.16.0)
- `audit_check.hpp` -- centralized CHECK macro with 20-char ASCII progress bar
- 22 audit .cpp files migrated to shared CHECK
- Windows unbuffered stdout fix (setvbuf _IONBF)
- SARIF v2.1.0 output for GitHub Code Scanning

---

## 7. Testing Infrastructure

### New Test Modules (v3.16.0)
- `test_musig2_bip327_vectors.cpp` -- 35 BIP-327 MuSig2 reference tests
- `test_ffi_round_trip.cpp` -- 103 FFI round-trip boundary tests
- `test_fiat_crypto_vectors.cpp` -- expanded to 752 cross-checks
- `test_rfc9591_invariants` -- 7 ciphersuite-independent invariants
- `test_rfc9591_3of5` -- exhaustive 3-of-5 FROST over all 10 subsets

### Existing Test Enhancements (v3.15.0)
- MuSig2 + FROST advanced protocol tests
- Parser fuzz (DER, Schnorr, Pubkey)
- Cross-library differential test vs bitcoin-core/libsecp256k1
- Address/BIP32/FFI fuzz tests
- FROST KAT pinned vectors
- Point edge-case tests
- FE52 Jacobian is_on_curve for FAST_52BIT platforms

---

## 8. Documentation

- **COMPATIBILITY.md** -- BIP-340 strict encoding compatibility notes
- **BINDINGS_ERROR_MODEL.md** -- strict semantics for binding authors
- **SECURITY.md** -- memory handling (erasure), planned items, API stability
- **CT_VERIFICATION.md** -- constant-time verification methodology
- **SUPPORTED_GUARANTEES.md** -- what the library guarantees
- **ADOPTERS.md** -- production/development/hobby categories
- **AUDIT_COVERAGE.md** -- full CI infrastructure documentation
- **ABI versioning policy** -- Phase II 2.4.1
- **GitHub Discussion templates** -- Q&A, Show-and-Tell, Ideas, Integration Help

---

## 9. Language Bindings (v3.14.0)

### 12 Languages -- 41-function C API parity
| Language | New Functions | Status |
|----------|--------------|--------|
| Java | 22 JNI functions + 3 helper classes | Full coverage |
| Swift | 20 functions | Full coverage |
| React Native | 15 functions | Full coverage |
| Python | 3 functions (ctx_clone, last_error, last_error_msg) | Full coverage |
| Rust | 2 functions | Full coverage |
| Dart | 1 function (ctx_clone) | Full coverage |
| Go, Node.js, C#, Ruby, PHP | Already complete | Verified |

### Binding Fixes
- Package naming: `libsecp256k1-fast*` -> `libufsecp*`
- RPM spec, Debian control, Arch PKGBUILD corrected
- 3 binding READMEs: removed inaccurate CT-layer claims
- README dead link fix

---

## 10. Build & Platform

- **UFSECP_BITCOIN_STRICT** -- CMake option to enforce strict-only parsing
- **MSVC** -- SECP256K1_NOINLINE macro + s_gen4 race fix
- **GCC -Wpedantic** -- __int128 extension type warning suppression
- **schnorr.cpp** -- from_bytes type mismatch fix (MSVC/wasm/armv7)
- **Pragma balance** -- removed misbalanced push/pop in ct_field.cpp
- **Reproducible builds** -- signed releases, SBOM
- **Signed SHA256SUMS** manifest + verify instructions

---

## Platform Audit Results

### Windows x86_64 (this machine)
<!-- TO BE FILLED after local audit run -->

### Android ARM64 (via ADB)
<!-- TO BE FILLED after Android audit run -->

### RISC-V (Milk-V Mars)
<!-- TO BE FILLED after RISC-V audit run -->

### ESP32
<!-- TO BE FILLED after ESP32 audit run -->

---

## Upgrade Instructions

```bash
# From v3.14.0 or v3.15.x -- no breaking changes
git fetch --tags
git checkout v3.16.0

# Build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Verify
ctest --test-dir build --output-on-failure
./build/audit/unified_audit_runner
```

## SHA256
<!-- TO BE FILLED at release time -->
