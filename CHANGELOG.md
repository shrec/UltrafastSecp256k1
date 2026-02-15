# Changelog

All notable changes to secp256k1-fast will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [3.2.0] - 2025-07-24

### Added — Cryptographic Features
- **ECDH key exchange** — `ecdh_compute` (SHA-256 of compressed point), `ecdh_compute_xonly` (SHA-256 of x-coordinate), `ecdh_compute_raw` (raw x-coordinate) (`cpu/include/ecdh.hpp`, `cpu/src/ecdh.cpp`)
- **ECDSA public key recovery** — `ecdsa_sign_recoverable` (deterministic recid), `ecdsa_recover` (reconstruct pubkey from signature + recid), compact 65-byte serialization (`cpu/include/recovery.hpp`, `cpu/src/recovery.cpp`)
- **Taproot (BIP-341/342)** — Tweak hash, output key computation, private key tweaking, commitment verification, TapLeaf/TapBranch hashing, Merkle root/proof construction (`cpu/include/taproot.hpp`, `cpu/src/taproot.cpp`)
- **Constant-time byte utilities** — `ct_equal`, `ct_is_zero`, `ct_compare`, `ct_memzero` (volatile + asm barrier), `ct_memcpy_if`, `ct_memswap_if`, `ct_select_byte` (`cpu/include/ct_utils.hpp`)

### Added — Performance
- **AVX2/AVX-512 SIMD batch field ops** — Runtime CPUID detection, auto-dispatching `batch_field_add/sub/mul/sqr`, Montgomery batch inverse (1 inversion + 3(n-1) multiplications) (`cpu/include/field_simd.hpp`, `cpu/src/field_simd.cpp`)

### Added — Build & Packaging
- **vcpkg manifest** — `vcpkg.json` with optional features (asm, cuda, lto)
- **Conan 2.x recipe** — `conanfile.py` with CMakeToolchain integration and shared/fPIC/asm/lto options
- **Benchmark dashboard CI** — GitHub Actions workflow (`benchmark.yml`) running benchmarks on Linux + Windows, `parse_benchmark.py` for JSON output, `github-action-benchmark` integration with 120% alert threshold

### Added — Tests (84 new)
- `test_ecdh_recovery_taproot` — 76 tests: ECDH (basic/xonly/raw/zero-key/infinity), Recovery (sign+recover/multiple-keys/compact-roundtrip/wrong-recid/invalid), Taproot (tweak-hash/output-key/privkey-tweak/commitment/leaf-branch/merkle-tree/proof/full-flow), CT Utils (equal/zero/compare/memzero/conditional), Wycheproof vectors (ECDSA edge 8, Schnorr edge 3, Recovery edge 5+)
- `test_simd_batch` — 8 tests: SIMD detection, batch add/sub/mul/sqr, batch inverse (basic/single/scratch)

---

## [3.1.0] - 2025-07-23

### Added — Cryptographic Features
- **Multi-scalar multiplication** — Shamir's trick (2-point) + Strauss interleaved wNAF (n-point) (`cpu/include/multiscalar.hpp`, `cpu/src/multiscalar.cpp`)
- **Batch signature verification** — Schnorr and ECDSA batch verify with random linear combination; `identify_invalid()` to pinpoint bad signatures (`cpu/include/batch_verify.hpp`, `cpu/src/batch_verify.cpp`)
- **BIP-32 HD key derivation** — Master key from seed, hardened/normal child derivation, path parsing (m/0'/1/2h), Base58Check serialization (xprv/xpub), RIPEMD-160 fingerprinting (`cpu/include/bip32.hpp`, `cpu/src/bip32.cpp`)
- **MuSig2 multi-signatures (BIP-327)** — Key aggregation (KeyAgg), deterministic nonce generation, 2-round signing protocol, partial sig verify, Schnorr-compatible aggregate signatures (`cpu/include/musig2.hpp`, `cpu/src/musig2.cpp`)
- **SHA-512** — Header-only implementation for HMAC-SHA512 / BIP-32 (`cpu/include/sha512.hpp`)

### Added — GPU Optimization
- **Occupancy auto-tune utility** — `gpu_occupancy.cuh` with `optimal_launch_1d()` (uses `cudaOccupancyMaxPotentialBlockSize`), `query_occupancy()`, and startup device diagnostics
- **Warp-level reduction primitives** — `warp_reduce_sum()`, `warp_reduce_sum64()`, `warp_reduce_or()`, `warp_broadcast()`, `warp_aggregated_atomic_add()` in reusable header
- **`__launch_bounds__` on library kernels** — `field_mul/add/sub/inv_kernel` (256,4), `scalar_mul_batch/generator_mul_batch_kernel` (128,2), `point_add/dbl_kernel` (256,4), `hash160_pubkey_kernel` (256,4)

### Added — Build & Tooling
- **PGO build scripts** — `build_pgo.sh` (Linux, Clang/GCC auto-detect) and `build_pgo.ps1` (Windows, MSVC/ClangCL)
- **MSVC PGO support** — CMakeLists.txt now handles `/GL` + `/GENPROFILE` / `/USEPROFILE` for MSVC in addition to Clang/GCC

### Added — Tests (63 new)
- `test_multiscalar_batch` — 16 tests: Shamir edge cases, multi-scalar sums, Schnorr & ECDSA batch verify + identify invalid
- `test_bip32` — 28 tests: HMAC-SHA512 vectors, BIP-32 TV1 master/child keys, path derivation, serialization, seed validation
- `test_musig2` — 19 tests: key aggregation, nonce generation, 2-of-2 & 3-of-3 signing with Schnorr cross-verify, single-signer edge case

### Fixed
- **SHA-512 K[23] constant** — Single-bit typo (`0x76f988da831153b6` → `0x76f988da831153b5`) that caused all SHA-512 hashes to be incorrect
- **MuSig2 per-signer Y parity** — `musig2_partial_sign()` now negates the secret key when the signer's public key has odd Y (required for x-only pubkey compatibility)

---

## [3.0.0] - 2025-07-22

### Added — Cryptographic Primitives
- **ECDSA (RFC 6979)** — Deterministic signing & verification (`cpu/include/ecdsa.hpp`)
- **Schnorr BIP-340** — x-only signing & verification (`cpu/include/schnorr.hpp`)
- **SHA-256** — Standalone hash, zero-dependency (`cpu/include/sha256.hpp`)
- **Constant-time benchmarks** — CT layer micro-benchmarks via CTest

### Added — Platform Support
- **iOS** — CMake toolchain, XCFramework build script, SPM (`Package.swift`), CocoaPods (`UltrafastSecp256k1.podspec`), C++ umbrella header
- **WebAssembly (Emscripten)** — C API (11 functions), JS wrapper (`secp256k1.mjs`), TypeScript declarations, npm package `@ultrafastsecp256k1/wasm`
- **ROCm / HIP** — CUDA ↔ HIP portability layer (`gpu_compat.h`), all 24 PTX asm blocks guarded with `#if SECP256K1_USE_PTX` + portable `__int128` alternatives, dual CUDA/HIP CMake build
- **Android NDK** — arm64-v8a CI build with NDK r27c

### Added — Infrastructure
- **CI/CD (GitHub Actions)** — Linux (gcc-13/clang-17 × Release/Debug), Windows (MSVC), macOS (AppleClang), iOS (OS + Simulator + XCFramework), WASM (Emscripten), Android (NDK), ROCm (Docker)
- **Doxygen → GitHub Pages** — Auto-generated API docs on push to main
- **Fuzzing harness** — `tests/fuzz_field.cpp` for libFuzzer field arithmetic testing
- **Version header** — `cmake/version.hpp.in` auto-generates `SECP256K1_VERSION_*` macros
- **`.clang-format` + `.editorconfig`** — Consistent code formatting
- **Desktop example app** — `examples/desktop_example.cpp` with CTest integration
- **CMake install** — `install(TARGETS)` + `install(DIRECTORY)` for system-wide deployment

### Changed
- **Search kernels relocated** — `cuda/include/` → `cuda/app/` (cleaner library vs. app separation)
- **README** — 7 CI badges, comprehensive build instructions for all platforms

### ⚠️ Testers Wanted
> We need community testers for platforms we cannot fully validate in CI:
> - **iOS** — Real device testing (iPhone/iPad with Xcode)
> - **AMD GPU (ROCm/HIP)** — AMD Radeon RX / Instinct hardware
>
> If you have access to these platforms, please run the build and report results!
> Open an issue at https://github.com/shrec/UltrafastSecp256k1/issues

---

## [2.0.0] - 2025-07-10

### Added
- **Shared POD types** (`include/secp256k1/types.hpp`): Canonical data layouts
  (`FieldElementData`, `ScalarData`, `AffinePointData`, `JacobianPointData`,
  `MidFieldElementData`) with `static_assert` layout guarantees across all backends
- **CUDA edge case tests** (10 new): zero scalar, order scalar, point cancellation,
  infinity operand, add/dbl consistency, commutativity, associativity, field inv
  edges, scalar mul cross-check, distributive — now 40/40 total
- **OpenCL edge case tests** (8 new): matching coverage — now 40/40 total
- **Shared test vectors** (`tests/test_vectors.hpp`): canonical K*G vectors,
  edge scalars, large scalar pairs, hex utilities
- **CTest integration for CUDA** (`cuda/CMakeLists.txt`)
- **CPU `data()`/`from_data()`** accessors on FieldElement and Scalar for
  zero-cost cross-backend interop

### Changed
- **CUDA**: `FieldElement`, `Scalar`, `AffinePoint` are now `using` aliases
  to shared POD types (zero overhead, no API change)
- **OpenCL**: Added `static_assert` layout compatibility checks + `to_data()`/
  `from_data()` conversion utilities
- **OpenCL point ops optimized**: 3-temp point doubling (was 12-temp),
  alias-safe mixed addition
- **CUDA point ops optimized**: Local-variable rewrite eliminates pointer aliasing —
  Point Double **2.29× faster** (1.6→0.7 ns), Point Add **1.91× faster** (2.1→1.1 ns),
  kG **2.25× faster** (485→216 ns). CUDA now beats OpenCL on all point ops.
- **PTX inline assembly** for NVIDIA OpenCL: Field ops now at parity with CUDA
- **Benchmarks updated**: Full CUDA + OpenCL numbers on RTX 5060 Ti

### Performance (RTX 5060 Ti, kernel-only)
- CUDA kG: 216.1 ns (4.63 M/s) — **CUDA 1.37× faster than OpenCL**
- OpenCL kG: 295.1 ns (3.39 M/s)
- Point Double: CUDA 0.7 ns (1,352 M/s), OpenCL 0.9 ns — **CUDA 1.29×**
- Point Add: CUDA 1.1 ns (916 M/s), OpenCL 1.6 ns — **CUDA 1.45×**
- Field Mul: 0.2 ns on both (4,139 M/s)

## [1.0.0] - 2026-02-02

### Added
- Complete secp256k1 field arithmetic
- Point addition, doubling, and multiplication
- Scalar arithmetic
- GLV endomorphism optimization
- Assembly optimizations:
  - x86-64 BMI2/ADX (3-5× speedup)
  - RISC-V RV64GC (2-3× speedup)
  - RISC-V Vector Extension (RVV) support
- CUDA batch operations
- Memory-mapped database support
- Comprehensive documentation

### Performance
- x86-64 field multiplication: ~8ns (assembly)
- RISC-V field multiplication: ~75ns (assembly)
- CUDA batch throughput: 8M ops/s (RTX 4090)

---

**Legend:**
- `Added` - New features
- `Changed` - Changes in existing functionality
- `Deprecated` - Soon-to-be removed features
- `Removed` - Removed features
- `Fixed` - Bug fixes
- `Security` - Security fixes
