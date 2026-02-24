# Changelog

All notable changes to UltrafastSecp256k1 are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.13.1] - 2026-02-24

### Fixed
- **Critical: GLV decomposition overflow in `ct::scalar_mul()`** — `ct_mul_256x_lo128_mod` used single-phase reduction (256×128-bit), which overflowed when GLV's `c1`/`c2` rounded to exactly 2^128. Additionally, `lambda*k2` computation only read 2 lower limbs of `k2_abs`, silently dropping `limb[2]=1`. This caused wrong results for ~5/64 random scalar inputs. Replaced with full `ct_scalar_mul_mod_n()`: 4×4 schoolbook → 8-limb product → 3-phase `reduce_512` (512→385→258→256 bits), matching libsecp256k1's algorithm. Both `5×52` (`__int128`) and `4×64` (portable `U128`/`mul64`) paths fixed.
- **GLV constant `minus_b2`** — changed from 128-bit `b2_pos` to full 256-bit `Scalar(n - b2)`, and decomposition formula from `scalar_sub(p1, p2)` to `scalar_add(p1, p2)` since both constants are already negated
- **`-Werror=unused-function`** — added `[[maybe_unused]]` to diagnostic helpers `print_scalar()` and `print_point_xy()` in `diag_scalar_mul.cpp`

### Removed
- Dead code: `ct_mul_lo128_mod()` and `ct_mul_256x_lo128_mod()` (replaced by `ct_scalar_mul_mod_n`)

### Performance
- CT scalar_mul overhead vs fast path: **1.05×** (25.3μs vs 24.0μs) — no regression

---

## [3.13.0] - 2026-02-24

### Added
- **BIP-32 official test vectors TV1–TV5** — 90 comprehensive checks covering master key derivation, hardened/normal child paths, and public-only derivation chains (`test_bip32_vectors.cpp`)
- **Nightly CI workflow** — daily extended verification: differential correctness with 100× multiplier (~1.3M checks) and dudect full-mode statistical analysis (30 min, t=4.5 threshold)
- **Differential test CLI/env multiplier** — `differential_test` accepts `--multiplier=N` or `UFSECP_DIFF_MULTIPLIER` env variable; default 1 preserves existing CI behavior

### Fixed
- **BIP-32 public key decompression** — `public_key()` now correctly decompresses from compressed prefix + x-coordinate via y²=x³+7 square root with parity check; previously treated x-coordinate as scalar, producing wrong public keys for public-only derivation
- **`pub_prefix` field** in `ExtendedKey` — stores y-parity byte (0x02/0x03) across `to_public()`, `derive_child()`, and `serialize()` for correct compressed public key round-trip
- **SonarCloud `ct_sidechannel` exclusion** — changed `-E ct_sidechannel` to exact-match `-E "^ct_sidechannel$"` to prevent accidental exclusion of other tests

---

## [3.12.3] - 2026-02-24

### Fixed
- **Valgrind "still reachable" false positives** — added `valgrind.supp` suppression file for precomputed wNAF/comb table allocations that are intentionally kept for program lifetime
- **CTest memcheck integration** — switched from `enable_testing()` to `include(CTest)` for proper Valgrind memcheck support
- **Security audit CI** — added `--suppressions` flag and exact-match `ct_sidechannel` exclusion in Valgrind step
- **ASan heap-buffer-overflow** in dudect smoke mode — fixed buffer overread in timing analysis
- **aarch64 cross-compilation** — added missing toolchain file for ARM64 CI builds

---

## [3.12.2] - 2026-02-24

### Security
- **Branchless `ct_compare`** — rewritten with bitwise arithmetic and `asm volatile` value barriers; dudect |t| dropped from 22.29 → 2.17, eliminating a timing side-channel leak

### Fixed
- **SonarCloud coverage collection** — use `run_selftest` as primary llvm-cov binary (links full library); coverage report now reflects actual test execution
- **Dead code elimination in `precompute.cpp`** — `RDTSC()` gated behind `SECP256K1_PROFILE_DECOMP`; `multiply_u64`/`mul64x64`/`mul_256` unified to call `_umul128()` instead of duplicating `__int128` inline
- **GCC `#pragma clang diagnostic` warnings** — wrapped in `#ifdef __clang__` guards in 3 test files
- **GCC `-Wstringop-overflow`** — bounds check in `base58check_encode` (address.cpp)
- **All `-Werror` warnings resolved** — 41 files across library, tests, and benchmarks
- **Clang-tidy CI** — filter `.S` assembly from analysis, add `--quiet` and parallel `xargs`
- **Unused variable** — removed `compressed` in `bip32.cpp` `to_public()`

### Changed
- **`const` on hot-path intermediates** — ~60 `FieldElement52` write-once variables in `point.cpp` marked `const`
- **Benchmark exclusion** — `sonar-project.properties` excludes benchmark files from coverage calculation
- **CPD minimum tokens** — set to 100 in `sonar-project.properties`

### Added
- **GOVERNANCE.md** — BDFL governance model with continuity plan (bus factor)
- **ROADMAP.md** — 12-month project roadmap (Mar 2026 – Feb 2027)
- **CONTRIBUTING.md** — Developer Certificate of Origin (DCO) requirement
- **OpenSSF Best Practices badge** — added to README
- **Code scanning fixes** — resolved alerts #281, #282

---

## [3.12.1] - 2026-02-23

### Security
- **bump wheel 0.45.1 → 0.46.2** — fixes CVE-2026-24049 (path traversal in `wheel unpack`)
- **bump setuptools 75.8.0 → 78.1.1** — fixes CVE-2025-47273 (path traversal via vendored wheel)

### Changed
- **VERSION.txt** updated to 3.12.1

---

## [3.12.0] - 2026-02-23

### Security — CI/CD Hardening & Supply-Chain Protection
- **SHA-pinned all GitHub Actions** — every action uses immutable commit SHA instead of mutable tags
- **Harden Runner** — `step-security/harden-runner` v2.14.2 on every CI job (egress audit)
- **CodeQL** — upgraded to v4.32.4, job-level `security-events: write`, custom query filters
- **OpenSSF Scorecard** — daily scorecard workflow with SARIF upload
- **SonarCloud** — CI-based code quality analysis with build-wrapper
- **pip hash pinning** — `--require-hashes` on all pip install steps in release/CI workflows
- **Dependabot** — configured for GitHub Actions, pip, npm, NuGet, Cargo ecosystems
- **Branch protection** — required reviews, dismiss stale, strict status checks on `main`

### Fixed
- **66+ code scanning alerts resolved** — unused variables, permissions, hardcoded credentials, scorecard findings
- **StepSecurity remediation** — merged PR #25 with fixes for GHA best practices

### Changed
- **Dependabot PRs #26–#32 merged** — codeql-action v4.32.4, setup-dotnet v5.1.0, upload-artifact v6.0.0, download-artifact v7.0.0, scorecard-action v2.4.3, attest-build-provenance v3.2.0, sonarqube-scan-action v7.0.0
- **Rust workspace Cargo.toml** — added for Dependabot Cargo ecosystem support

### Added
- **`docs/CODING_STANDARDS.md`** — comprehensive coding standards for OpenSSF CII badge
- **`CONTRIBUTING.md` requirements section** — explicit contribution requirements with links
- **Full AGPL-3.0 LICENSE text** — replaced summary with standard text for GitHub license detection

---

## [3.11.0] - 2026-02-23

### Performance — Effective-Affine & RISC-V Optimization
- **Effective-affine GLV table** — batch-normalize P-multiples to affine in `scalar_mul_glv52`, eliminating Z-coordinate arithmetic from the main loop. Point Add 821→159 ns on x86-64.
- **RISC-V auto-detect CPU** — CMake reads `/proc/cpuinfo` uarch field to set `-mcpu=sifive-u74` automatically. **28–34% speedup** on Milk-V Mars (Scalar Mul 235→154 μs).
- **RISC-V ThinLTO propagation** — ARCH_FLAGS propagated via INTERFACE compile+link options so ThinLTO codegen uses correct CPU scheduling at link time.
- **RISC-V Zba/Zbb fix** — explicit `-march=rv64gc_zba_zbb` alongside `-mcpu` since Clang's sifive-u74 model omits these extensions.
- **ARM64 10×26 field representation** — verified as optimal for Cortex-A76 (74 ns mul vs 100 ns with 5×52).

### Performance — Embedded
- **SafeGCD30 field inverse** — GCD-based modular inverse for non-`__int128` platforms: ESP32 **118 μs** (was 3 ms).
- **SafeGCD30 scalar inverse** — same technique for scalar field; optimized SHA-256/HMAC/RFC-6979 for embedded.
- **ESP32 4-stream GLV Strauss** — parallel endomorphism streams + Z²-verify optimization.
- **CT layer optimizations** — comprehensive CT optimization pass for embedded targets.

### Changed
- **Unified benchmark harness** — all 4 bench binaries share common framework with IQR outlier removal and RDTSCP/chrono auto-selection.
- **CMake 4.x compatibility** — standalone build support with `cmake_minimum_required(3.18)` + project-level CTest.
- **Disable RISC-V FE52 asm** — C++ `__int128` inline is 26–33% faster than hand-written FE52 assembly on RISC-V.
- **Benchmark data refresh** — all platforms re-measured: x86-64 (Clang 21), ARM64 (RK3588), RISC-V (Milk-V Mars).
- **Remove competitor comparison tables** — benchmarks show only UltrafastSecp256k1 results.

### Added
- **Lightning donation** — `shrec@stacker.news` badge in README.
- **ARM64 5×52 MUL/UMULH kernel** — interleaved multiply for exploration (10×26 remains default).
- **ESP32 comprehensive benchmark** — full benchmark matching x86 format.

### Fixed
- **CI Unicode cleanup** — replaced all Unicode characters with ASCII across codebase.
- **CI benchmark parse fix** — reset baseline for Unicode-free benchmark output.
- **Orphaned submodule** — removed stale `cpu/secp256k1` submodule entry.

### Acknowledgments
- Stacker News, Delving Bitcoin, and @0xbitcoiner for community support.

---

## [3.10.0] - 2026-02-21

### Performance — CT Hot-Path Optimization (Phases 5–15)
- **5×52 field representation** — switched point internals from 4×64 to `FieldElement52`, enabling `__int128` lazy reduction across all CT operations
- **Direct asm bypass** — CT `field_mul`/`field_sqr` now call hand-tuned 5×52 multiply/square directly: **70 ns → 33 ns**
- **GLV endomorphism** — CT `scalar_mul` via λ-decomposition + interleaved double-and-add: **304 μs → 20 μs**
- **CT generator_mul precomputed table** — 16-entry precomputed-G table with batch inversion: **310 μs → 9.8 μs (31× speedup)**
- **Batch inversion + Brier-Joye unified add** — Montgomery's trick for multi-point normalization
- **Hamburg signed-digit + batch doubling** — compact signed-digit recoding with merged double passes
- **128-bit split + w=15 for G-stream verify** — Shamir-style dual-stream with wider window: **~14% verify speedup**
- **AVX2 CT table lookup** — `_mm256_cmpeq_epi64` + `_mm256_and_si256` constant-time table scan
- **Effective-affine P table** — batch-normalize P-multiples to skip Z-coordinate arithmetic in main loop
- **Schnorr keypair/pubkey caching + FE52 sqrt** — avoid redundant serialization in sign/verify
- **FE52-native inverse + isomorphic table build + GCD `inv_var`** — SafeGCD field inverse stays in 52-bit form
- **Format conversion elimination** — removed `to_fe()`/`from_fe()` round-trips on every CT hot path
- **Redundant normalize elimination** — `ct_field_mul_impl`/`square_impl` produce already-reduced results
- **Schnorr X-check + Y-parity combined** — single Z-inverse for both x-coordinate check and y-parity in FE52

### Performance — I-Cache Optimization
- **`noinline` on `jac52_add_mixed_inplace`** — prevents inlining of 800+ byte function body into tight loops: **59% I-cache miss reduction**

### Fixed
- **`scalar_mul_glv52` infinity guard** — early return on `base.is_infinity() || scalar.is_zero()` prevents zero-inverse crash in Montgomery batch trick (CI #128–131 regression)
- **CT `complete_add` fallback** — uses affine `x()`/`y()` instead of raw Jacobian `X()`/`Y()`
- **MSVC fallback** — `field_neg` arity, `is_equal_mask`, GLV decompose, `y_bytes` redefinition
- **Cross-platform FE52 guard** — `SECP256K1_FAST_52BIT` gating prevents compilation on 32-bit targets

### Changed
- **Dead code removal** — removed functions superseded by Z-ratio normalization path
- **Barrett → specialized GLV multiplies** — replaced generic Barrett reduction with curve-specific multiply

### CI / Infrastructure
- **npm/nuget publishing fix** — corrected CI workflow for package publishing
- **Comprehensive audit suite** — 8 suites, 641K checks, cryptographic correctness validation
- **CT operations benchmark** — `bench_ct_vs_libsecp` with per-operation ns/op and throughput
- **dudect timing test** — side-channel timing leakage detection for CT operations
- **Doxyfile version auto-injection** — `VERSION.txt` → `Doxyfile` at configure time

---

## [3.6.0] - 2026-02-20

### Added — GPU Signature Operations (CUDA)
- **ECDSA Sign on GPU** — `ecdsa_sign_batch_kernel` with RFC 6979 deterministic nonces, low-S normalization. **204.8 ns / 4.88 M/s** per signature.
- **ECDSA Verify on GPU** — `ecdsa_verify_batch_kernel` with Shamir's trick + GLV endomorphism. **410.1 ns / 2.44 M/s** per verification.
- **ECDSA Sign Recoverable on GPU** — `ecdsa_sign_recoverable_batch_kernel` with recovery ID computation. **311.5 ns / 3.21 M/s**.
- **ECDSA Recover on GPU** — `ecdsa_recover_batch_kernel` for public key recovery from signature + recid.
- **Schnorr Sign (BIP-340) on GPU** — `schnorr_sign_batch_kernel` with tagged hash midstates. **273.4 ns / 3.66 M/s**.
- **Schnorr Verify (BIP-340) on GPU** — `schnorr_verify_batch_kernel` with x-only pubkey verification. **354.6 ns / 2.82 M/s**.
- **6 new batch kernel wrappers** in `secp256k1.cu` — all with `__launch_bounds__(128, 2)` matching scalar_mul kernels.
- **5 GPU signature benchmarks** in `bench_cuda.cu` — ECDSA sign, verify, sign+recid, Schnorr sign, Schnorr verify.
- **`prepare_ecdsa_test_data()`** helper — generates valid signatures on GPU for verify benchmark correctness.

> **No other open-source GPU library provides secp256k1 ECDSA + Schnorr sign/verify.** This is the only production-ready multi-backend (CUDA + OpenCL + Metal) GPU secp256k1 library.

### Changed
- **CUDA benchmark numbers updated** — Scalar Mul improved to 225.8 ns (was 266.5 ns), Field Inv to 10.2 ns (was 12.1 ns) from `__launch_bounds__` thread count fix (128 vs 256 mismatch).
- **README** — Added blockchain coin badges (Bitcoin, Ethereum, +25), GPU signature benchmark tables, 27-coin supported coins section, SEO metadata footer, updated performance headline.
- **BENCHMARKS.md** — Split CUDA section into Core ECC + GPU Signature Operations; updated all comparison tables.

### Fixed
- **CUDA benchmark thread mismatch** — Benchmarks used 256 threads/block but kernels declared `__launch_bounds__(128, 2)`, causing 0.0 ns results. Fixed to use 128 threads.

---

## [3.4.0] - 2026-02-19

### Added — Stable C ABI (`ufsecp`)
- **Complete C ABI library** — `ufsecp.dll` / `libufsecp.so` / `libufsecp.dylib` with 45 exported symbols, opaque `ufsecp_ctx` handle, and structured error model (11 error codes)
- **Headers**: `ufsecp.h` (main API, 37 functions), `ufsecp_version.h` (ABI versioning), `ufsecp_error.h` (error codes)
- **Implementation**: `ufsecp_impl.cpp` wrapping C++ core into C-linkage with zero heap allocations on hot paths
- **Build system**: `include/ufsecp/CMakeLists.txt` — shared + static build, standalone or sub-project mode, pkg-config template (`ufsecp.pc.in`)
- **API coverage**: key generation, ECDSA sign/verify/recover, Schnorr BIP-340 sign/verify, SHA-256, ECDH (compressed/xonly/raw), BIP-32 HD derivation, Bitcoin addresses (P2PKH/P2WPKH/P2TR), WIF encode/decode, DER serialization, public key tweak (add/mul), selftest
- **`SUPPORTED_GUARANTEES.md`** — Tier 1/2/3 stability guarantees documentation
- **`examples/hello_world.c`** — Minimal usage example

### Added — Dual-Layer Constant-Time Architecture
- **Always-on dual layers** — `secp256k1::fast::*` (public operations) and `secp256k1::ct::*` (secret-key operations) are always active simultaneously; no flag-based selection
- **CT layer** — Complete addition formula (12M+2S), fixed-trace scalar multiplication, constant-time table lookup
- **Valgrind/MSAN markers** — `SECP256K1_CLASSIFY()` / `SECP256K1_DECLASSIFY()` for verifiable constant-time guarantees

### Added — SHA-256 Hardware Acceleration
- **SHA-NI hardware dispatch** — Runtime CPUID detection for Intel SHA Extensions; transparent fallback to software implementation
- **Zero-overhead dispatch** — Function pointer set once at init, no branching in hot path

### Added — C# P/Invoke Bindings & Benchmarks
- **`bindings/csharp/UfsepcBenchmark/`** — .NET 8.0 project with complete P/Invoke declarations for all 45 `ufsecp` functions
- **68 correctness tests** — 12 categories covering key ops, ECDSA, Schnorr, SHA-256, ECDH, BIP-32, addresses, DER round-trip, recovery, WIF, tweaks, selftest
- **19 benchmarks** — SHA-256: 137ns, ECDSA Sign: 11.89μs, Verify: 47.95μs, Schnorr Sign: 10.68μs, KeyGen: 1.22μs
- **P/Invoke overhead measured** — ~10–40ns per call (negligible)

### Changed
- `ufsecp_ctx_create()` takes no flags parameter — dual-layer CT architecture is always active

---

## [3.3.0] - 2026-02-16

### Added — Comprehensive Benchmarks
- **Metal GPU benchmark** (`bench_metal.mm`): 9 operations — Field Mul/Add/Sub/Sqr/Inv, Point Add/Double, Scalar Mul (P×k), Generator Mul (G×k). Matches CUDA benchmark format with warmup, kernel-only timing, and throughput tables.
- **3 new Metal GPU kernels**: `field_add_bench`, `field_sub_bench`, `field_inv_bench` in `secp256k1_kernels.metal`
- **WASM benchmark** (`bench_wasm.mjs`): Node.js benchmark for all WASM-exported operations — Pubkey Create (G×k), Point Mul, Point Add, ECDSA Sign/Verify, Schnorr Sign/Verify, SHA-256 (32B/1KB)
- WASM benchmark runs automatically in CI (Node.js 20 setup + execution)

### Added — Security & Maturity
- SECURITY.md v3.2 with vulnerability reporting guidelines
- THREAT_MODEL.md with detailed threat analysis
- API stability guarantees documented
- Fuzz testing documentation and libFuzzer harnesses
- Selftest modes: smoke (fast), ci (full), stress (extended)
- Repro bundle support for deterministic test reproduction
- Sanitizer CI integration (ASan/UBSan/TSan)

### Added — Testing
- Boundary KAT vectors for field limb boundaries
- Batch inverse sweep tests
- Unified test runner (12 test files consolidated into single runner)

### Added — Documentation
- Batch inverse & mixed addition API reference with examples (full point, X-only, CUDA, division, scratch reuse, Montgomery trick)
- CHANGELOG.md (this file), CODE_OF_CONDUCT.md
- Benchmark dashboard link in README

### Changed
- Benchmark alert threshold 120% → 150% (reduces false positive alerts on shared CI runners)
- README: added Apple Silicon/Metal badges, CI status badge, version badge, benchmark dashboard link
- Feature coverage table updated to v3.3.0
- Badge layout reorganized: CI/Bench/Release first, then GPU backends, then platforms

### Fixed
- Metal shader compilation errors (MSL address space mismatches, jacobian_to_affine ordering)
- Metal: skip generator_mul test on non-Apple7+ paravirtual devices (CI fix)
- Keccak `rotl64` undefined behavior (shift by 0)
- macOS build flags for Clang compatibility
- Metal `metal2.4` shader standard for newer Xcode toolchains
- WASM runtime crash: removed `--closure 1`, added `-fno-exceptions`, increased initial memory to 4MB
- Bitcoin CoinFeatures header fix

### Removed
- Unused `.cuh` files and `sorted_ecc_db`
- Database/lookup/bloom references from public documentation
- AI-generated text removed from README

---

## [3.2.0] - 2026-02-16

### Added — Coins Layer
- **Multi-coin infrastructure** — `coins/coin_params.hpp` with constexpr `CoinParams` definitions for 27 secp256k1-based cryptocurrencies: Bitcoin, Litecoin, Dogecoin, Dash, Ethereum, Bitcoin Cash, Bitcoin SV, Zcash, DigiByte, Namecoin, Peercoin, Vertcoin, Viacoin, Groestlcoin, Syscoin, BNB Smart Chain, Polygon, Avalanche, Fantom, Arbitrum, Optimism, Ravencoin, Flux, Qtum, Horizen, Bitcoin Gold, Komodo
- **Unified address generation** — `coin_address()`, `coin_address_p2pkh()`, `coin_address_p2wpkh()`, `coin_address_p2tr()` with automatic encoding dispatch per coin (Base58Check / Bech32 / EIP-55)
- **Per-coin WIF encoding** — `coin_wif_encode()` with coin-specific prefix bytes
- **Full key derivation pipeline** — `coin_derive()` takes private key + CoinParams → public key + address + WIF in one call
- **Coin registry** — `find_by_ticker("BTC")`, `find_by_coin_type(60)`, `ALL_COINS[]` array for iteration

### Added — Ethereum & EVM Support
- **Keccak-256 hash** — Standard Keccak-256 (NOT SHA3-256; Ethereum-compatible 0x01 padding), incremental API (`Keccak256State::update/finalize`), one-shot `keccak256()` (`coins/keccak256.hpp`, `src/keccak256.cpp`)
- **Ethereum addresses (EIP-55)** — `ethereum_address()` with mixed-case checksummed output, `ethereum_address_raw()`, `ethereum_address_bytes()`, `eip55_checksum()`, `eip55_verify()` (`coins/ethereum.hpp`, `src/ethereum.cpp`)
- **EVM chain compatibility** — Same address derivation works for BSC, Polygon, Avalanche, Fantom, Arbitrum, Optimism

### Added — BIP-44 HD Derivation
- **Coin-type derivation** — `coin_derive_key()` with automatic purpose selection: BIP-86 (Taproot) for Bitcoin, BIP-84 (SegWit) for Litecoin, BIP-44 (legacy) for Dogecoin/Ethereum
- **Path construction** — `coin_derive_path()` builds `m/purpose'/coin_type'/account'/change/index`
- **Seed-to-address pipeline** — `coin_address_from_seed()` full pipeline: seed → BIP-32 master → BIP-44 derivation → coin address

### Added — Custom Generator Point & Curve Context
- **CurveContext** — `context.hpp` with custom generator point support, curve order (raw bytes), cofactor, and name (`CurveContext::secp256k1_default()`, `CurveContext::with_generator()`, `CurveContext::custom()`)
- **Context-aware operations** — `derive_public_key(privkey, &ctx)`, `scalar_mul_G(scalar, &ctx)`, `effective_generator(&ctx)` — nullptr = standard secp256k1, custom context = custom G
- **Zero-overhead default** — Standard secp256k1 usage with nullptr context has no extra cost

### Added — Tests
- **test_coins** — 32 tests covering CurveContext, CoinParams registry, Keccak-256 vectors, EIP-55 checksum, Bitcoin/Litecoin/Dogecoin/Dash/Ethereum addresses, WIF encoding, BIP-44 path/derivation, custom generator derivation, full multi-coin pipeline

---

## [3.1.0] - 2026-02-15

### Added — Cryptographic Protocols
- **Pedersen Commitments** — `pedersen_commit(value, blinding)`, `pedersen_verify()`, `pedersen_verify_sum()` (homomorphic balance proofs), `pedersen_blind_sum()`, `pedersen_switch_commit()` (Mimblewimble switch commitments); nothing-up-my-sleeve generators H and J via SHA-256 try-and-increment (`cpu/include/pedersen.hpp`, `cpu/src/pedersen.cpp`)
- **FROST Threshold Signatures** — `frost_keygen_begin()` / `frost_keygen_finalize()` (Feldman VSS distributed key generation), `frost_sign_nonce_gen()` / `frost_sign()` (partial signature rounds), `frost_verify_partial()`, `frost_aggregate()` → standard BIP-340 SchnorrSignature; `frost_lagrange_coefficient()` helper (`cpu/include/frost.hpp`, `cpu/src/frost.cpp`)
- **Adaptor Signatures** — Schnorr adaptor: `schnorr_adaptor_sign()`, `schnorr_adaptor_verify()`, `schnorr_adaptor_adapt()`, `schnorr_adaptor_extract()`; ECDSA adaptor: `ecdsa_adaptor_sign()`, `ecdsa_adaptor_verify()`, `ecdsa_adaptor_adapt()`, `ecdsa_adaptor_extract()` — for atomic swaps and DLCs (`cpu/include/adaptor.hpp`, `cpu/src/adaptor.cpp`)
- **MuSig2 multi-signatures (BIP-327)** — Key aggregation (KeyAgg), deterministic nonce generation, 2-round signing protocol, partial sig verify, Schnorr-compatible aggregate signatures (`cpu/include/musig2.hpp`, `cpu/src/musig2.cpp`)
- **ECDH key exchange** — `ecdh_compute` (SHA-256 of compressed point), `ecdh_compute_xonly` (SHA-256 of x-coordinate), `ecdh_compute_raw` (raw x-coordinate) (`cpu/include/ecdh.hpp`, `cpu/src/ecdh.cpp`)
- **ECDSA public key recovery** — `ecdsa_sign_recoverable` (deterministic recid), `ecdsa_recover` (reconstruct pubkey from signature + recid), compact 65-byte serialization (`cpu/include/recovery.hpp`, `cpu/src/recovery.cpp`)
- **Taproot (BIP-341/342)** — Tweak hash, output key computation, private key tweaking, commitment verification, TapLeaf/TapBranch hashing, Merkle root/proof construction (`cpu/include/taproot.hpp`, `cpu/src/taproot.cpp`)
- **BIP-32 HD key derivation** — Master key from seed, hardened/normal child derivation, path parsing (m/0'/1/2h), Base58Check serialization (xprv/xpub), RIPEMD-160 fingerprinting (`cpu/include/bip32.hpp`, `cpu/src/bip32.cpp`)
- **BIP-352 Silent Payments** — `silent_payment_address()`, `SilentPaymentAddress::encode()`, `silent_payment_create_output()`, `silent_payment_scan()` with ECDH-based stealth addressing and multi-output support (`cpu/include/address.hpp`, `cpu/src/address.cpp`)

### Added — Address & Encoding
- **Bitcoin Address Generation** — `hash160()` (RIPEMD-160 + SHA-256), `base58check_encode()` / `base58check_decode()`, `bech32_encode()` / `bech32_decode()` (BIP-173/BIP-350, Bech32/Bech32m), `address_p2pkh()`, `address_p2wpkh()`, `address_p2tr()`, `wif_encode()` / `wif_decode()` (`cpu/include/address.hpp`, `cpu/src/address.cpp`)

### Added — Core Algorithms
- **Multi-scalar multiplication** — Shamir's trick (2-point) + Strauss interleaved wNAF (n-point) (`cpu/include/multiscalar.hpp`, `cpu/src/multiscalar.cpp`)
- **Batch signature verification** — Schnorr and ECDSA batch verify with random linear combination; `identify_invalid()` to pinpoint bad signatures (`cpu/include/batch_verify.hpp`, `cpu/src/batch_verify.cpp`)
- **SHA-512** — Header-only implementation for HMAC-SHA512 / BIP-32 (`cpu/include/sha512.hpp`)
- **Constant-time byte utilities** — `ct_equal`, `ct_is_zero`, `ct_compare`, `ct_memzero` (volatile + asm barrier), `ct_memcpy_if`, `ct_memswap_if`, `ct_select_byte` (`cpu/include/ct_utils.hpp`)

### Added — Performance
- **AVX2/AVX-512 SIMD batch field ops** — Runtime CPUID detection, auto-dispatching `batch_field_add/sub/mul/sqr`, Montgomery batch inverse (1 inversion + 3(n-1) multiplications) (`cpu/include/field_simd.hpp`, `cpu/src/field_simd.cpp`)

### Added — GPU Optimization
- **Occupancy auto-tune utility** — `gpu_occupancy.cuh` with `optimal_launch_1d()` (uses `cudaOccupancyMaxPotentialBlockSize`), `query_occupancy()`, and startup device diagnostics
- **Warp-level reduction primitives** — `warp_reduce_sum()`, `warp_reduce_sum64()`, `warp_reduce_or()`, `warp_broadcast()`, `warp_aggregated_atomic_add()` in reusable header
- **`__launch_bounds__` on library kernels** — `field_mul/add/sub/inv_kernel` (256,4), `scalar_mul_batch/generator_mul_batch_kernel` (128,2), `point_add/dbl_kernel` (256,4), `hash160_pubkey_kernel` (256,4)

### Added — Build & Packaging
- **PGO build scripts** — `build_pgo.sh` (Linux, Clang/GCC auto-detect) and `build_pgo.ps1` (Windows, MSVC/ClangCL)
- **MSVC PGO support** — CMakeLists.txt now handles `/GL` + `/GENPROFILE` / `/USEPROFILE` for MSVC in addition to Clang/GCC
- **vcpkg manifest** — `vcpkg.json` with optional features (asm, cuda, lto)
- **Conan 2.x recipe** — `conanfile.py` with CMakeToolchain integration and shared/fPIC/asm/lto options
- **Benchmark dashboard CI** — GitHub Actions workflow (`benchmark.yml`) running benchmarks on Linux + Windows, `parse_benchmark.py` for JSON output, `github-action-benchmark` integration with 120% alert threshold

### Added — Tests (237 new)
- `test_v4_features` — 90 tests: Pedersen (basic/homomorphic/balance/switch/serialization/zero-value), FROST (Lagrange/keygen/2-of-3 signing), Adaptor (Schnorr basic/ECDSA basic/identity), Address (Base58Check/Bech32/Bech32m/hash160/P2PKH/P2WPKH/P2TR/WIF/consistency), Silent Payments (address/flow/multi-output)
- `test_ecdh_recovery_taproot` — 76 tests: ECDH, Recovery, Taproot, CT Utils, Wycheproof vectors
- `test_multiscalar_batch` — 16 tests: Shamir edge cases, multi-scalar sums, Schnorr & ECDSA batch verify
- `test_bip32` — 28 tests: HMAC-SHA512 vectors, BIP-32 TV1 master/child keys, path derivation, serialization
- `test_musig2` — 19 tests: key aggregation, nonce generation, 2-of-2 & 3-of-3 signing
- `test_simd_batch` — 8 tests: SIMD detection, batch add/sub/mul/sqr, batch inverse

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
> Open an issue at https://github.com/shrec/Secp256K1fast/issues

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
