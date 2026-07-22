# Changelog

All notable changes to UltrafastSecp256k1 are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Added the ABI-compatible `ufsecp_addr_p2sh_with_ctx` entry point for P2SH
  generation with `ufsecp_last_error()` / `ufsecp_last_error_msg()` diagnostics.
  The existing stateless `ufsecp_addr_p2sh` symbol remains available and
  produces byte-identical addresses.

## [4.5.0] - 2026-07-07

> **Bridge-free libbitcoin integration, public-data GPU parity, and release
> evidence hardening.** A minor, **ABI-compatible** release
> (`UFSECP_ABI_VERSION` stays 4). This release makes the direct libbitcoin
> integration path the primary consumer contract, expands GPU-backed public
> operations without requiring libbitcoin to use the C ABI, and hardens the
> release gates that prove backend parity, package contents, and benchmark
> evidence.

### Added

- **Bridge-free direct libbitcoin C++ integration.** `compat/libbitcoin_direct`
  now provides the direct header/profile, row/column batch verification hooks,
  GPU column hook smoke coverage, and examples for libbitcoin-owned table
  layouts without bridge/shim marshalling.
- **Libbitcoin public-data GPU operation surface.** Direct integration examples
  and benchmarks cover x-only/pubkey validation, commitment verification,
  tagged-hash variants, and hash256/hash256-var style public workloads.
- **Backend parity gate for GPU operations.** `ci/check_gpu_backend_parity.py`
  now fails closed on CUDA/OpenCL/Metal operation drift, missing C ABI exposure,
  undocumented fallback/stub status, and stale public parity claims.
- **Release/package hardening.** Native release archives are checked for
  product-library-only contents before binding packages are assembled.

### Changed

- **Libbitcoin direct profile is canonical.** The integration path favors
  bounded batches, cancellation tokens, reusable row/column staging, and direct
  C++ entry points over C ABI bridges.
- **GPU backend coverage is documented as a strict parity contract.** Public
  docs and assurance matrices now align CUDA, OpenCL, Metal, and C ABI status
  for the exposed GPU operation set.
- **Benchmark evidence is tied to target context.** Libbitcoin direct batch,
  hash256-var, and public-operation benchmarks now carry JSON artifacts and
  perf-matrix validation before claims are trusted.

### Fixed

- **Libbitcoin high-S ECDSA consensus verification.** Direct libbitcoin verify
  paths now accept consensus-valid high-S signatures where Bitcoin consensus
  allows them, while the libsecp256k1 shim keeps libsecp-compatible low-S
  behavior.
- **libsecp256k1 shim high-S parity.** The shim reject path now matches
  libsecp256k1 behavior instead of accepting signatures the shim ABI should
  reject.
- **Standalone C API generated-feature installs.** The generated feature header
  and public header propagation are now covered for standalone consumers.
- **OpenCL/Metal public-operation parity regressions.** Public GPU operation
  drift is now covered by regression tests and the parity gate.
- **Quick performance smoke no longer blocks releases on shared-runner noise.**
  Push-time benchmark JSON validation and alerts remain, while release-grade
  regression decisions stay tied to controlled local benchmark evidence.
- **Static CAAS bundle verification is checkout-resilient.** The PR/push gate now
  verifies the committed external audit bundle via explicit workspace paths and
  falls back to the `HEAD` tree when a runner checkout misses the baseline file,
  while still failing closed if the committed baseline is absent or invalid.
- **Binding package release verification is scoped to native archives.** Python
  and npm package jobs now run the desktop-native package guard only against the
  desktop release archives, so mobile, WASM, and binding archives are not checked
  with the wrong package-layout policy.
- **Windows audit-report CAAS gates use UTF-8 Python output.** Release-tag audit
  snapshots now force UTF-8 Python I/O on Windows so grammar/stateful harness
  progress markers cannot trip `cp1252` console encoding during JSON generation.
- **CI Advisory MSan release checks are bounded.** Release-tag MSan advisory
  runs now build with MemorySanitizer and execute a representative core smoke
  subset instead of the full 400-test CTest suite, while full audit/security
  coverage remains in the dedicated release and security gates.

### Security

- **Release CAAS evidence remains fail-closed.** Audit bundle freshness, module
  counts, advisory wiring, package provenance, and backend parity gates were
  tightened so stale or overbroad release claims block instead of passing.
- **Formal/audit gate regressions fixed.** Cryptol parsing/proof execution and
  cross-platform audit library discovery now fail on real errors instead of
  false-green paths.
- **GPU signing reductions kept constant-time.** OpenCL and Metal field/scalar
  modular reductions on signing paths were aligned with CT requirements.

### Performance

- **BIP-352 scan path improved.** Silent-payment prefix matching and batched
  Montgomery Z-inversion reduce CPU staging work for scan-sized workloads.
- **Direct libbitcoin batch verification gained fused staging.** Persistent
  pools, FE52 decompression, and multithreaded `_mt` verify twins reduce
  repeated parsing/allocation overhead for validation-sized batches.
- **Public-data GPU workload evidence added.** Direct libbitcoin public ops and
  hash256-var benchmarks document where GPU execution is beneficial and where
  fallback/host cost dominates.

## [4.4.0] - 2026-06-19

> **Libbitcoin integration, GPU staging/decompression, CAAS evidence browsing,
> and release/CI hardening.** A minor, **ABI-compatible** release
> (`UFSECP_ABI_VERSION` stays 4). This release keeps the public ABI stable while
> making the libbitcoin table interface easier to integrate, reducing GPU
> staging overhead, tightening CUDA packaging, and turning CAAS evidence into a
> centralized reviewer-facing dashboard.

### Added

- **CAAS Evidence Browser dashboard.** `ci/caas_dashboard.py` now aggregates
  Integration, CT, Fuzz, GPU/hardware, Package Provenance, Libbitcoin
  Performance Matrix, Bastion Requirements, Audit SLA, and External Audit Bundle
  manifests into one searchable/filterable HTML evidence browser.
- **Libbitcoin performance matrix gate.** `bench_lbtc_batch` can emit structured
  JSON artifacts and the new G-21 gate validates libbitcoin benchmark target
  context, claim scope, and backing evidence before performance claims are
  trusted.
- **GPU SEC1 pubkey decompression support across CUDA/OpenCL/Metal.** Public
  compressed keys can stay compact until the backend, reducing host-side staging
  and PCIe traffic for public-data GPU workloads.

### Changed

- **Libbitcoin integration manual aligned with libbitcoin-owned row tables.**
  The documented path keeps libbitcoin's existing `std::span<Batch>` table shape,
  defaults the C++ controller to `UFSECP_LBTC_AUTO`, and documents opaque ECDSA
  normalization semantics for the default path.
- **CUDA runtime links statically by default.** CUDA-enabled engine, GPU host,
  C ABI, benchmarks, and libbitcoin bridge targets now default to static cudart
  linkage while preserving a standard CMake override for dynamic cudart.
- **CUDA/OpenCL ECDSA verify staging reduced.** Compact signatures are uploaded
  and parsed into backend scalar registers in the kernel path, avoiding a
  host-side `N x ECDSASignature` conversion loop.
- **libbitcoin bridge scratch is grow-only.** Hot row/result staging reuses
  thread-local buffers instead of allocating per chunk.
- **Research monitor escalation softened.** Scheduled runs no longer open
  GitHub issues for low-confidence `needs_review` research hits by default,
  while high-confidence findings remain actionable.

### Fixed

- **CUDA signing scalar arithmetic now follows CT primitives.** CUDA
  ECDSA/Schnorr signing paths reuse the CT scalar add/mul primitives instead of
  duplicate local modular arithmetic.
- **OpenCL/Metal compile and backend parity fixes.** Follow-up fixes restored
  OpenCL C++17 compatibility, Metal shader compilation, and backend-specific
  GPU decompression build paths.
- **Bech32 P2WPKH encoding allocation reduction.** The common witness address
  path now uses a stack-backed 5-bit conversion buffer and streaming checksum.
- **XCFramework CI upload hardened.** CI hard-fails on missing/empty
  XCFramework output before upload, while transient GitHub artifact upload
  failures no longer create false-red builds.

### Security

- **CAAS gate surface expanded without hiding residuals.** GPU/hardware
  evidence remains fail-closed for blocking gaps while ROCm real-device testing,
  native SNARK GPU kernels, and physical power/EM/fault evidence stay explicitly
  owner-gated or documented residuals.
- **Research-signal and integration evidence routing remains enforced.** New
  dashboard visibility is backed by existing gates rather than replacing them.

### Performance

- **libbitcoin batch verification path improved for real table workloads.**
  CUDA row scratch reuse and reduced staging overhead target validation-sized
  batches without forcing libbitcoin to reshape tables.
- **GPU public-key decompression trades negligible compute for less transfer and
  host work.** Local RTX 5060 Ti measurement showed compact-key backend
  decompression adds only ~0.8 ns per key while reducing key transfer from 104
  bytes to 33 bytes.

## [4.3.0] - 2026-06-16

> **libsecp256k1-ABI backend under our own `ultrafast_secp256k1` name for Bitcoin
> Knots/Core, libbitcoin integration, and Clang-speed crypto on Windows.** A minor,
> **ABI-compatible** release (`UFSECP_ABI_VERSION` stays 4 — additive only; every
> 4.2.x API and build flag is unchanged). Adds the `SECP256K1_BUILD_KNOTS` minimal
> profile and an opt-in self-contained `ultrafast_secp256k1` shared lib (exports the
> libsecp256k1 ABI, never installs a colliding `libsecp256k1` file), the libbitcoin
> opaque-ECDSA row ABI + GPU batch bridge, the
> `windows-clang-cl` preset, a cold-verify speedup, and a large CAAS "bastion"
> hardening pass (threat-gate coverage matrix reaches 0 gaps) plus several
> use-after-free and secret-erasure fixes.

### Added

- **`SECP256K1_BUILD_KNOTS` — minimal drop-in build profile for Bitcoin Knots.**
  A single flag compiles only the modules Knots needs (ecdsa core + recovery +
  schnorrsig + extrakeys + ellswift) and strips everything else (MuSig2, FROST, ZK,
  ECIES, BIP-352, adaptor, HD wallet, Ethereum, BCH, GPU, tests, benchmarks,
  bindings). The drop-in `.so` shrinks **1.83 MB → ~1.27 MB** (stock libsecp256k1
  0.6.0 is 1.26 MB) with identical verify/sign speed; CT-mandatory signing and
  strict BIP-340 stay on. When not building the shared `.so` the engine compiles as
  an OBJECT library that links straight into the consumer (no separate library).
  See `docs/BITCOIN_KNOTS_INTEGRATION.md`.
- **Self-contained `ultrafast_secp256k1` shared lib exporting the libsecp256k1 C
  ABI — explicit, opt-in Bitcoin Knots/Core backend (never a `libsecp256k1`
  collision).** `SECP256K1_SHIM_BUILD_SHARED=ON` builds and installs a
  self-contained `libultrafast_secp256k1.so` (`.dll`/`.lib`) plus the `secp256k1*.h`
  ABI headers and an `ultrafast_secp256k1.pc` — all under OUR name, so they never
  overwrite or shadow a system `libsecp256k1.so`/`libsecp256k1.pc`; the `secp256k1_*`
  ABI itself is unchanged. To use it as a node backend the integrator aliases
  `secp256k1` explicitly (CMake `add_library(secp256k1 ALIAS secp256k1_shim)` in a
  bundled tree, or their own libsecp256k1 alias/symlink over the installed
  `ultrafast_secp256k1`, at their own risk). Verified against Bitcoin Knots
  `v29.3.knots20260508`.
- **libbitcoin integration — opaque ECDSA row ABI + GPU batch bridge.** New
  opaque/compact ECDSA row entry points matching libbitcoin's row-verify shape,
  with signatures packed to the engine's GPU-native form for compact-column batch
  verify. Ships as a libbitcoin drop-in shim plus an opt-in batch bridge. See
  `docs/LIBBITCOIN_INTEGRATION.md` and `docs/SIGNATURE_FORMS.md`.
- **`windows-clang-cl` CMake preset — the recommended fast build on Windows.** Building
  with `clang-cl` (a drop-in, MSVC-ABI-compatible Clang driver: `link.exe`, MSVC CRT,
  `.pdb`, Visual Studio "LLVM (clang-cl)" toolset) gets Clang's inline MULX/ADCX/ADOX
  field kernels — which MSVC `cl` fundamentally cannot emit (no x64 inline asm, no
  `__int128`). Measured **3.3–3.6× faster** `scalar_mul`/`batch_verify`/`frost` than
  `cl`, i.e. near-clang++ parity, while staying in the MSVC toolchain.
  `run_selftest ci` = 31/31. CMake now auto-locates and links the clang `compiler-rt`
  builtins on Clang/Windows (the native-`__int128` field code emits `__modti3`/
  `__umodti3` that the MSVC CRT lacks), so `clang-cl`/`clang++` builds link out of the
  box. **Note:** mixing `clang-cl` objects with `cl`-compiled **C++** is not ABI-safe
  (by-value field-type returns crash); build a whole target with one compiler, or cross
  the boundary via the `ufsecp` C ABI. See
  `benchmarks/comparison/windows_msvc_vs_clang_20260614.md` §6.

### Changed

- **Windows/MSVC (`cl`) build tuning (for builds that must use cl).** MSVC Release
  builds now add `/Ob3` (most-aggressive inlining) and `/Oi /Gy /Gw`. On the Windows
  MSVC-vs-Clang benchmark sweep this is a consistent **~8–15%** speedup across the
  field/scalar/point kernels (field squaring reaches Clang parity, field add overtakes
  it) with **no** correctness or portability cost — `run_selftest ci` stays 31/31. New
  CMake cache options: `SECP256K1_MSVC_OB3` (default `ON`), `SECP256K1_MSVC_ARCH`
  (`SSE2|AVX|AVX2|AVX512`, default `SSE2`). `/arch:AVX2`/`AVX512` are selectable but
  **default off** — their auto-vectorization regresses the non-vectorizable 64-bit
  big-integer kernels.
- **`SECP256K1_MSVC_WPO` (`/GL` whole-program + `/LTCG` + `/OPT:REF,ICF`) is now
  OFF by default (was ON) — a consumer opt-in, not a forced library policy.** `/GL`
  emits whole-program IL into every object, so a **static** lib compiled with `/GL`
  forces *every* downstream consumer's linker into `/LTCG` for its whole program —
  even consumers who do not want it. Some consumers (e.g. libbitcoin) measure LTCG as
  a net runtime **regression** for their workload, so the library must not impose it.
  Build with `-DSECP256K1_MSVC_WPO=ON` to opt in for a self-contained release/benchmark
  build; when OFF, no `/GL` is emitted and the produced `.lib` never dictates the
  consumer's link-time optimization. The residual ~1.4–5× MSVC-vs-Clang gap on
  compound paths (scalar_mul, batch_verify, frost) is structural (MSVC has no native
  `__int128`/`MULX` inline field multiply) and is left as future work. Full data:
  `benchmarks/comparison/windows_msvc_vs_clang_20260614.md`.
- **win-arm64 (MSVC ARM64) clang-cl portability** groundwork; the Windows MSVC CI
  runner is pinned to VS2022 for reproducible packaging.

### Fixed

- **MuSig2 key-aggregation cache use-after-free.** The keyagg-cache handle could
  outlive its backing storage; the shim now enforces handle lifetime and rejects
  stale handles.
- **`g_context` use-after-free under concurrent context teardown.** A `shared_ptr`
  snapshot plus atomic comb-teeth make the global signing context thread-safe.
- **Secret-material erasure gaps closed.** `frost_sign` and `schnorr_keypair_create`
  now erase secret-derived scalar residues, and MuSig2 erases its `neg_k`/`neg_d`
  residues.
- **libsecp256k1 shim parity.** The opaque ECDSA signature layout now matches
  upstream libsecp256k1 byte-for-byte, and the ECDSA verify cache is keyed by the
  full public key.
- **libbitcoin bridge.** Accepts and normalizes high-S ECDSA batches.
- **BIP-39.** Fail-closed single-source CSPRNG with an entropy gate.

### Security

- **CAAS "bastion" hardening — the threat-gate coverage matrix reaches 0 gaps.**
  A large pass of new audit/evidence gates: soundness-coverage and
  metamorphic-relation gates, a cross-backend CPU↔GPU value-differential gate, a
  dudect binary constant-time probe, fuzz-harness liveness/smoke gates, a
  fault-injection countermeasure gate, secret-parse and secret-erase ABI gates, a
  canonical-encoding / malleability gate, evidence-freshness pre-alerts, and
  incident drills run as real fault injection. External-anchor KAT (NIST + BIP-341)
  defeats common-mode self-anchoring and valid/invalid coverage exercises the live
  ABI reject branches. No engine API or behaviour change — this hardens the
  assurance surface, not the cryptography.

### Performance

- **Cold ECDSA/Schnorr verify.** Removed per-call waste on the verify path (a
  redundant public-key curve-check, an opaque-signature double byte-reverse, and
  over-normalization in the GLV table build): ~120–170 ns faster cold verify,
  byte-identical results, warm verify unchanged.
- **Windows/MSVC.** `/Ob3` + whole-program optimization and faster u128 compat
  arithmetic.

## [4.2.1] - 2026-06-10

> **Security patch.** Fixes a clang-only correctness bug in large-batch ECDSA
> verification caught by the v4.2.0 release CI (the clang build legs). No ABI
> change — patch bump per SemVer (`UFSECP_ABI_VERSION` stays 4; every 4.2.0 API and
> build flag is unchanged). **Supersedes 4.2.0 — use 4.2.1.**

### Security

- **CA-001 — large-batch ECDSA verify accepted an off-curve public key (clang only).**
  `secp256k1_ecdsa_verify_batch` with `n >= 8` (the large-batch MSM path in
  `ecdsa_batch_verify`) accepted an invalid-curve / infinity public key when built
  with **clang** (x86-64 and arm64); **gcc rejected it**, so the gcc build legs and
  `ci_local` (gcc-14) passed. The shim maps an off-curve pubkey to
  `Point::infinity()`; single verify and the small-batch (`n < 8`) fallback reject
  it, but the large-batch path fed the infinity point into
  `dual_scalar_mul_gen_point` + the FE52 Z²-based x-coordinate check, whose behavior
  on an infinity operand is compiler-dependent. **Fix:** `ecdsa_batch_verify` now
  rejects `public_key.is_infinity()` up-front in its pre-validation loop, identical
  to single verify, on every compiler and architecture. Regression coverage:
  `regression_ecdsa_batch_curve_check` BCK-4/5/6 (clang-17 6/6, gcc-14 6/6).

## [4.2.0] - 2026-06-10

> **Security + conformance + libbitcoin/GPU batch release.** Headlined by the
> coordinated fix for **GHSA-c7q2-gv3g-rgxm** (ECDSA adaptor pre-signature
> soundness), this cycle lands a batch of MuSig2 BIP-327 conformance fixes, GPU
> constant-time hardening across CUDA/OpenCL/Metal, official BIP-340/749/749 test
> vectors in CI, and several new libbitcoin-bridge batch-verify endpoints with
> per-item GPU kernels. Minor bump per SemVer — all new APIs are additive — with
> **one breaking change**: the ECDSA adaptor signature serialization grew from
> **130 → 162 bytes** (it now carries the DLEQ proof). `UFSECP_ECDSA_ADAPTOR_SIG_LEN`
> changed; **recompile any consumer of the adaptor API**. The C-ABI function table
> is otherwise unchanged, so `UFSECP_ABI_VERSION` stays **4**.

### Security

- **GHSA-c7q2-gv3g-rgxm — ECDSA adaptor pre-signature soundness (medium, PoC-confirmed).**
  The pre-signature did not cryptographically bind `r` (= x-coord of `k·T`) to the
  adaptor point `T`, so a malicious signer could substitute an arbitrary `r'` and
  pass verification with a pre-signature that does **not** adapt to a valid ECDSA
  signature — breaking the core "verify ⇒ adaptable" guarantee. Fixed by binding the
  pre-signature with a **Chaum-Pedersen DLEQ proof** that `R_hat = k·G` and `R = k·T`
  share the same `k`; verify now checks the DLEQ, `r == R.x`, and the ECDSA relation.
  **Reported by Damir** (responsible disclosure, with PoC) — credited at the
  reporter's request. ⚠️ See the 130→162-byte format note above. As a result,
  `ecdsa_adaptor_verify` is intentionally slower (it now performs real DLEQ
  verification that the old, unsound path skipped).
- **MuSig2 BIP-327 conformance fixes** (cross-implementation interop): corrected the
  binding-factor domain tag (P1), the `gacc`/`tacc` accumulation for tweaked signing
  (P1), and acceptance of the infinity aggregate nonce (`R = G`, BIP-327 §).
- **libsecp256k1 shim:** zeroes the `pubkey_create` output on failure (fail-closed),
  and the dead variable-time secret-key ellswift XDH path was removed (constant-time
  XDH is the only remaining path).
- **GPU constant-time hardening:** branchless scalar `mod n` reduction, field
  reduction, and 64-bit secret arithmetic on CUDA / OpenCL / Metal — eliminates the
  key-dependent warp divergence in GPU ECDSA. Added a white-box Nsight Compute (ncu)
  CT-uniformity gate (the black-box dudect probe masked the divergence).
- **ZK GPU range-proof** Fiat-Shamir domain tags corrected (`Bulletproof/*`).
- **CPU & GPU ABI fail-closed hardening** across the C-ABI and shim surfaces.

### Conformance / spec vectors

- Official **BIP-340** (Schnorr), **BIP-341** (Taproot key-path tweak), and
  **BIP-327** (MuSig2 key aggregation) test vectors are now run in CI, plus a
  systemic tagged-hash tag-conformance gate across all backends.
- The in-process libsecp256k1 differential was extended with ECDH and MuSig2 key
  aggregation, and the in-process engine itself is now gated against libsecp256k1.

### Features — libbitcoin bridge & GPU batch

- **Aggregate (RLC) Schnorr batch-verify** endpoint.
- New batch endpoints: x-only pubkey validation (`lift_x` on-curve check),
  BIP-341 TapLeaf variable-length tagged hash, `HASH256`, and pubkey-validate.
- Three per-item GPU kernels shipped through the engine ABI + bridge
  (backend-agnostic across CUDA / OpenCL / Metal).
- BIP-341 **Taproot commitment batch** (CPU per-row + GPU RLC/Fiat-Shamir
  fast-check), a single-buffer (AoS) mmap-direct columnar commitment batch, and a
  first-class 4-table batch model (multisig + threshold).

### Performance

- `ellswift_create`: the `x3` field sqrt in `ellswift_try_u` is now deferred so it
  is computed only when the `x1`/`x2` branch fails (behavior-preserving).

### CI / audit / tooling

- **New local `-Werror` production-build gate** in `ci_local.sh --full` (gate `[7.5]`,
  `ci/check_werror_build.sh`) mirroring the GitHub "Build with -Werror" Security
  Audit job — closes the gap that let an orphaned static function reach CI.
- Bug-bounty program **marked NOT currently funded** (valid reports receive public
  credit + a CVE; reward tiers are aspirational pending sponsorship).
- Sanitizer zone-isolation (MSan vs Valgrind authority), GPU white-box ncu CT gate
  wired into `ci_local --gpu`, multiple CAAS false-green gate closures, and audit
  module consolidation.
- Dependency bumps (Dependabot): `codecov-action`, `codeql-action`.

## [4.1.1] - 2026-06-05

> **Minor bugfix / hardening release.** Constant-time and fail-closed fixes in the
> BIP-32 and libsecp256k1-shim paths, a batch of audit/CI gate corrections, and
> release/CI reliability fixes. No ABI break — patch bump per SemVer (ABI version
> stays 4; every 4.1.0 API and build flag is unchanged).

### Security / constant-time

- **BIP-32 private derivation now scrubs intermediate secrets** in
  `bip32_derive_path` — intermediate private keys are erased, and the on-failure
  scrub is now unconditional (P2-CT-001).
- **libsecp256k1 shim zeroes signature output on every parse failure** — fail-closed,
  so a failed parse can never leave stale or partial signature bytes in the caller's
  buffer (PASS3-SHIM-001).

### Fixed — audit / CI integrity

- Closed 6 false-green CAAS/CI gates that could report PASS without actually
  validating (PASS6 / P7 / P8 review findings).
- Fixed two tests that passed vacuously (P7-TEST-002, P7-TEST-003).
- Updated the RT-02 source scan to match the parse-DER zeroing fix.

### Fixed — release / CI reliability

- `pack-go` release job now checks out the release tag (`ref: RELEASE_TAG`),
  fixing a checkout race when the `sync-docs` job moves the tag mid-run.
- CI-Advisory Windows job sets `PYTHONUTF8=1` so the Python audit tests no longer
  crash on cp1252 console-encoding of the ✓/✗ status markers.

### Documentation

- Refreshed `API_SECURITY_CONTRACTS`; documented two shim divergences and corrected
  claim/benchmark drift; recorded the BIP-32 secret-lifecycle guarantees.
- Donations: replaced the Stacker News entry with a Bitcoin Silent Payment
  (BIP-352) address.

## [4.1.0] - 2026-06-02

> **Modular build system + multi-coin support + libbitcoin batch acceleration +
> audit / constant-time hardening + a consensus fix.** The cycle opens with a
> feature-module build system — independently toggleable CPU and GPU modules,
> coin-specific CMake presets, a generated feature header, and OBJECT-library
> integration — adds BCH / Litecoin / Dogecoin shim modules and a native MSVC NuGet
> package, lands the GPU/CPU batch script-signature "collect" verify path requested
> by the libbitcoin maintainer (with a dedicated CUDA kernel), closes a set of P1
> security-review findings — including a consensus-relevant ECDSA recovery edge case
> (bbhunt-001) — makes the public constant-time scalar multiply fully branchless on
> secret data (CT-CRYPTO-001), and lands the audit-hardening / benchmark-methodology
> work (MED-3 MuSig2 ABI bypass, canonical bench turbo-lock, Rule 16 post-build
> enforcement). No ABI break — minor bump per SemVer (all new APIs and build flags
> are additive/opt-in; the only behavior change is fail-closed on a previously
> deprecated v1 MuSig2 ABI path; v2 callers are unaffected).

### Build system & packaging
- **Feature-module build system.** CPU feature modules — FROST, ZK, ECIES,
  BIP-352, Adaptor, Wallet, MuSig2, BIP-324, Pippenger — are now independently
  toggleable via `SECP256K1_BUILD_<MODULE>` flags, with a generated
  `secp256k1/secp256k1_features.h` exposing `SECP256K1_HAS_*` macros so consumers
  can compile against exactly the surface they enabled.
- **Independent GPU feature-module flags** (`SECP256K1_GPU_BUILD_FROST`, …),
  separate from the CPU module flags, strip unused kernels from the GPU build
  (e.g. FROST) and enable a full minimal-node GPU build that isolates
  ECDH / MSM / HASH160 / ECRECOVER.
- **Coin-specific CMake presets** (`docs/PRODUCT_PROFILES.md`): `bitcoin-core`,
  `litecoin`, `dogecoin`, `bch-wallet`, `wallet`, `audit`, `cpu-release` — each
  enables only the modules that backend needs (e.g. `bitcoin-core` = MuSig2 +
  BIP-324 + Pippenger, everything else off).
- **`SECP256K1_BUILD_AS_OBJECT`** — embed the engine as a CMake OBJECT library;
  `docs/BUILD_INTEGRATION_GUIDE.md` documents the static / unity / LTO / OBJECT
  integration modes. A unified profile manifest + smoke test is wired into the
  fast gates.
- **Per-module CAAS test isolation by build flag** — audit/exploit tests are gated
  to the modules actually compiled (FROST / ZK / ECIES / BIP-352 / Adaptor /
  Wallet), so a stripped profile still audits cleanly.
- **New coin modules:** **BCH** (`SECP256K1_BUILD_BCH`, off by default) — RPA
  reusable-payment-address scan + CashAddr + hash160, with CUDA and OpenCL grinding
  kernels; **Litecoin** and **Dogecoin** Core shims + coin parameters.
- **`SECP256K1_SHIM_RFC6979_COMPAT`** opt-in build flag for libsecp256k1 RFC6979
  nonce compatibility in the shim.
- **Packaging:** native MSVC static-lib **NuGet** package (vc143/vc145, x64/arm64)
  for libbitcoin; the release ships static + DLL builds of both the `ufsecp` and
  `secp256k1` ABIs.
- Docs: `docs/BUILDING.md`, `docs/BUILD_INTEGRATION_GUIDE.md`,
  `docs/PRODUCT_PROFILES.md`.

### Added — libbitcoin batch acceleration bridge
- **In-place "collect" batch verify** for ECDSA and BIP-340 Schnorr
  (`ufsecp_lbtc_verify_ecdsa_collect` / `_schnorr_collect`; C++20
  `Controller::collect_ecdsa/_schnorr(std::span<Row>)`). The per-row verdict is
  collapsed into the row's trailing key cell (valid → zeroed, invalid → caller id
  survives), so the caller collects every non-zero key cell as the rejected-id set —
  no `results[]` array, no second side table. The existing results-based verify API
  is **byte-for-byte unchanged**; `key_size` must be `> 0`; `n` is unbounded (walked
  in internal chunks); degenerate calls are fail-closed no-ops.
- **Dedicated CUDA collect kernel** (`ufsecp_gpu_ecdsa_verify_collect` /
  `_schnorr_verify_collect`): a verbatim copy of the verify kernel writing the
  verdict on-device. OpenCL/Metal inherit a safe `Unsupported` default and the
  bridge falls back to a consensus-identical host-collapse path. Proven **GPU == CPU
  == libsecp256k1 bit-for-bit** on the rejected set (ECDSA + Schnorr, mixed corpus;
  `tests/test_lbtc_consensus_diff.cpp`). **No throughput improvement is claimed** for
  the collect kernel — it measured within noise of host-collapse on the test GPU and
  ships as a consensus-neutral specialization.

### Security
- **P1-SEC-01 / MED-3 closed** (`src/cpu/src/musig2.cpp`,
  `src/cpu/src/impl/ufsecp_musig2.cpp`). The C++ `musig2_partial_sign` now
  fail-closes (`Scalar::zero()`) when `key_agg_ctx.individual_pubkeys` is
  empty — previously the Rule-13 cross-validation block was skipped,
  allowing an ABI-deserialized caller to sign as ANY claimed signer index.
  `ufsecp_musig2_partial_sign` (v1) now hard-fails at entry with a
  migration message; `ufsecp_musig2_partial_sign_v2` inlines the signing
  path through a new helper `musig2_partial_sign_core` after populating
  `individual_pubkeys` from its `pubkeys` parameter.
- MSI-4 regression test flipped from advisory-skipped (rc=77) to mandatory.
- **bbhunt-001 — ECDSA recovery `r >= (p − n)` (consensus).** `ecdsa_recover`
  now rejects a `recid & 2` recovery when `r >= (p − n)` (where adding the curve
  order overflows the field prime). Applied across the CPU path, the libsecp256k1
  shim, the C ABI, and the CUDA / OpenCL recovery kernels. Regression test
  `test_regression_recover_rplus_n_overflow` (8/8).
- **CT-CRYPTO-001 — branchless `ct::scalar_mul`.** The public variable-base
  constant-time scalar multiply no longer branches on the secret GLV sign bit; it
  uses the branchless masked-select helper already used by the sibling
  `scalar_mul_*` variants. Reached from ECDH, ellswift XDH and secret-scalar
  tweak-mul. Regression `test_regression_ct_glv_make_v_branchless` (256 random
  scalars, both polarities, bit-exact vs the variable-time reference).
- **bbhunt-002 — adaptor entropy erasure.** `aux_rand` (and the secret key on
  early-return paths) are now securely erased in the zero-knowledge adaptor path.
- **PASS-COMPAT-001/002/004 — shim fail-closed output zeroing.** The shim seckey
  operations, `ecdsa_recover`, and `xonly_pubkey_tweak_add` now zero their output
  buffers on failure, matching upstream libsecp256k1.
- **4 P1 PR-readiness blockers** resolved: bindings ABI-version expectation
  corrected to 4; INTEGRATION_PATCH "randomize" no-op claim made truthful; CAAS
  required-checks list matched to real job names; advisory-JSON Rule 16 enforced —
  each with a paired CI gate (`check_abi_version_sync`, `check_randomize_claim_
  consistency`, `check_required_checks_match_jobs`, `check_advisory_json_rule16`).

### Performance / Benchmark
- **P1-BENCH-01 closed.** New canonical artifact
  [`docs/bench_unified_2026-05-23_gcc14_x86-64.json`](docs/bench_unified_2026-05-23_gcc14_x86-64.json)
  was measured under hard turbo lock (`intel_pstate/no_turbo=1`,
  `governor=performance`, `taskset -c 0 nice -20`). Previous canonical had
  `turbo: unknown`. Ratios shifted under controlled conditions:
  - CT ECDSA sign: 1.27× → **1.32× faster** than libsecp256k1
  - CT Schnorr sign: 1.13× → **1.27× faster**
  - ConnectBlock LTO: **+0.9–1.5%**, no-LTO **−0.5–1.0%** (unchanged)
  - SignSchnorrWithMerkleRoot: **+35.1%**

### CI / Evidence
- **P1-CI-01 closed.** `ci/check_advisory_skip_returns.sh` (Rule 16) is now
  invoked POST-BUILD in three independent CI choke points: `preflight.yml`
  after the `unified_audit_runner` build, `gate.yml` `shim-gate` after the
  shim-linked runner build, and `release.yml` `release-caas-gate` after the
  release build. Previously the script ran only PRE-build where rc=77 ("no
  build dir") made it a no-op.
- **Audit surface → 419 modules** (270 exploit PoCs + 149 non-exploit). New GPU
  collect functions covered by the ABI hostile-caller manifest (null/zero/invalid/
  smoke quartet; misuse-resistance 184/184). External audit evidence bundle
  regenerated (12/12) over the new modules, claims, hostile-caller contract and
  assurance ledger.
- `ci/check_security_fix_has_test.py` now recognizes
  `compat/libbitcoin_bridge/tests/` as a valid test location (the bridge
  differential tests are consensus-bearing).
- Scorecard `Pinned-Dependencies` (#10631): the NuGet-native workflow's chocolatey
  `ninja` install is now version-pinned.

### Documentation / Metadata
- **P1-CHG-01 closed.** CHANGELOG `[4.0.0]` performance table replaced
  FAST-path-vs-libsecp-CT diagnostic ratios (+2.76× / +2.57×) with the
  production-equivalent CT-vs-CT ratios from `canonical_numbers.json`.
- **P1-ZEN-01 closed.** `.zenodo.json` description no longer hardcodes the
  stale "252 exploit PoC" count or the unsubstantiated "4.88M GPU
  signs/s" claim; it now references CHANGELOG and BENCHMARKS.md.
- `docs/canonical_data.json` extended with `audit_sections_count: 10`,
  `caas_autonomy_score: "100/100"`, `caas_gates_passed/total: 8/8`,
  `ct_pipeline_ci_enforced/manual: 3/2`, `platform_target_count: 12` and
  a wording string. These previously had no canonical home.
- `ci/sync_canonical_numbers.py` `TARGET_DOCS` extended to scan
  `CHANGELOG.md` and `.zenodo.json` so future canonical updates propagate.
- `ci/sync_module_count.py` `DOC_FILES` extended with `.zenodo.json`.
- `ci/check_protocol_invariants.py` MUSIG2_SIGNER_INDEX rule retargeted at
  `ufsecp_musig2_partial_sign_v2` (the canonical signing path after v1
  hard-fail).

### Compatibility / Migration
- **Backwards-compatible for v2 callers.** External callers of the
  long-deprecated v1 ABI (`ufsecp_musig2_partial_sign`) will now receive
  `UFSECP_ERR_BAD_INPUT` with a migration message at the first call;
  switch to `ufsecp_musig2_partial_sign_v2` (available since the SEC-001
  landing).

## [4.0.0] - 2026-05-16

> **Bitcoin Core secondary backend candidate — CPU path, verification package.**
> This is not a trust request. It is a verification package.

### Performance (GCC 14.2.0 · Release+LTO · Intel i5-14400F · hard turbo lock)

| Workload | Ultra | libsecp256k1 | Delta |
|---|---|---|---|
| CT ECDSA sign | 21.6 µs | 59.7 µs | **+2.76×** [diagnostic] |
| CT Schnorr sign (BIP-340) | 18.1 µs | 46.5 µs | **+2.57×** [diagnostic] |
| Schnorr verify | 84.3 µs | 84.3 µs | equal |
| ConnectBlock ECDSA 2000 sigs (LTO) | 254 ms | 257 ms | **+1.2%** |
| ConnectBlock Schnorr 2000 sigs (LTO) | 253 ms | 255 ms | **+0.9%** |
| SignSchnorrWithMerkleRoot (bench_bitcoin) | 95 µs | 113 µs | **+18%** |

[diagnostic] = measurement from a non-standard comparison environment (different libsecp build flags / older methodology). Standard bench_unified results are in the canonical JSON artifact linked below.

*Canonical: [bench_unified_2026-05-23_gcc14_x86-64.json](docs/bench_unified_2026-05-23_gcc14_x86-64.json) · [BITCOIN_CORE_BENCH_RESULTS.json](docs/BITCOIN_CORE_BENCH_RESULTS.json)*

Without LTO: ConnectBlock ~0.5–1.0% slower (i-cache pressure). LTO required for Ultra to win.

### Security & Audit Evidence

**Headline (curated, from `dbde5711` on main):**
- **270 exploit PoC** security probes, 20+ CVE/attack classes — all pass
- **401 total audit modules** (270 exploit PoC + 131 non-exploit)
- **CAAS autonomy: 100/100** (8/8 gates)
- **Bitcoin Core test suite: 749/749** (GCC 14.2.0, May 2026)
- Constant-time signing: ECDSA, Schnorr, MuSig2, FROST, BIP-324 XDH
- libsecp256k1 shim parity: DER BIP-66, Schnorr BIP-340, ECDH, ElligatorSwift, MuSig2
- Reproducible builds: pinned compiler/toolchain, deterministic LTO

**Detail (dev-history form):**

### Security / Audit
- **Point::scalar_mul exploit PoC** — `test_exploit_scalar_mul.cpp` (59 checks, SM-1..SM-12)
  targeting edge cases across the CPU scalar multiplication path: zero/identity/order-n scalars,
  n-1 negation, GLV consistency (pubkey_create vs tweak_mul), additive homomorphism, Shamir
  trick, multi-scalar-mul, large scalar near-n wrap-around, scalar overflow >n, repeated
  doubling chain. All 12 sub-tests pass; no bugs found — confirms correctness of GLV
  decomposition, wNAF encoding, and endomorphism application paths.
- **Metal `field_reduce_512` truncation fix (issue #226)** — `uint extra = uint(acc[8])` in
  `src/metal/shaders/secp256k1_field.h` truncated a 33-bit accumulator to 32 bits, silently
  producing incorrect field_sqr/field_mul results for ~0.05% of inputs. Root cause: secp256k1
  reduction constant K = 2^32+977 generates carries that exceed 32 bits. Fixed with
  `ulong extra` + `while(acc[8] != 0)` carry-drain loop. Regression PoC:
  `test_exploit_metal_field_reduce.cpp` (14 checks, MFR-1..MFR-7). Cross-backend audit confirmed
  CUDA and OpenCL are unaffected. Commits `3c3aac1f`, `943fb674`.
- **Network selector validation bypass** — all 6 address/WIF C ABI functions
  (`ufsecp_addr_p2pkh`, `p2wpkh`, `p2tr`, `p2sh`, `p2sh_p2wpkh`, `wif_encode`) silently
  accepted arbitrary `int` values as the network parameter, treating unknowns as Mainnet or
  Testnet. An attacker-controlled caller passing `network=99` would receive a valid-looking
  but semantically ambiguous address. Fixed with `valid_network()` guard returning
  `UFSECP_ERR_BAD_INPUT`. Exploit PoC: `test_exploit_network_validation_bypass.cpp`
  (54 checks, NVB-1..NVB-8). Commit `943fb674`.
- **Mutation residue exploit vectors** — `test_exploit_mutation_residue.cpp` (23 checks,
  MR-1..MR-7) targeting wNAF OOB, divsteps mask, scalar inverse, window sign-magnitude,
  field reduction constant, dubious identity shortcut, nonce bit-length. Source integrity
  scanner: `test_mutation_artifact_scan.cpp` (6 checks, MA-1..MA-4). Commit `736e753f`.
- **Python audit script suite** (`ci/`) — 8 dynamic + 1 static analysis scripts, 9 Python
  CTest targets total (`scripts/` is a legacy mirror; canonical path is `ci/`). All run via `--lib path/to/libufsecp.so`. All PASS.
  - `differential_cross_impl.py` (1000+ checks): drives library alongside coincurve + python-ecdsa
    for random (sk, msg) pairs; catches wrong low-S normalization, pubkey parity bugs, ECDH
    mismatches. CTest `py_differential_crossimpl`. Committed `e94523bb` / `ad32e1d1`.
  - `nonce_bias_detector.py` (10,000+ sign ops): chi-squared + Kolmogorov-Smirnov + per-bit sweep;
    catches Minerva/TPM-FAIL-class biases invisible to code review. CTest `py_nonce_bias`.
    Committed `e94523bb` / `ad32e1d1`.
  - `rfc6979_spec_verifier.py` (200+ checks): independent pure-Python RFC 6979 §3.2 HMAC-SHA256
    nonce derivation vs library output + RFC 6979 Appendix A.2.5 KAT. CTest `py_rfc6979_spec`.
    Committed `e94523bb` / `ad32e1d1`.
  - `bip32_cka_demo.py`: live BIP-32 non-hardened parent key recovery demo + hardened immunity
    check. CTest `py_bip32_cka`. Committed `e94523bb` / `ad32e1d1`.
  - `glv_exhaustive_check.py` (5000+ scalars): GLV decomposition verifier — adversarial
    Babai-boundary scalars (near n/2, λ, 2^127, 2^128) vs coincurve reference. CTest
    `py_glv_exhaustive`. Committed `e94523bb` / `ad32e1d1`.
  - `semantic_props.py` (1450+ checks): algebraic property tests (kG+lG==(k+l)G, k(lG)==(kl)G),
    sign/verify roundtrip, determinism, Hypothesis integration. CTest `py_semantic_props`.
    Committed `bdc00c6b`.
  - `invalid_input_grammar.py` (37 checks): structured invalid-input rejection — bad pubkey prefix,
    x≥p, not-on-curve, sk=0/n/overrange, r=0/n, s=0/n, invalid BIP-32 seed/paths. CTest
    `py_invalid_input_grammar`. Committed `bdc00c6b`.
  - `stateful_sequences.py` (401+ checks): API sequence verifier — error-injection recovery,
    multi-level BIP-32 path consistency, dual-context independence, 5000-op endurance. CTest
    `py_stateful_sequences`. Committed `bdc00c6b`.
- **Dev bug scanner** (`scripts/dev_bug_scanner.py`) — 15-category static analyzer for library C++
  source (cpu/src, gpu/src, opencl, metal, bindings, include), 221 files scanned.
  **182 findings (82 HIGH, 100 MEDIUM)**: NULL 51 | CPASTE 45 | SIG 31 | RETVAL 30 | MSET 19 |
  OB1 5 | ZEROIZE 1. Categories: NULL (dereference before check), CPASTE (same lvalue assigned
  twice), SIG (signed/unsigned mismatch), RETVAL (ufsecp_* return value discarded), MSET (wrong
  memset literal size), OB1 (off-by-one ≤ vs <), ZEROIZE (clears fewer bytes than declared),
  SEMI (dangling semicolon), MBREAK (missing break/fallthrough), TRUNC (i64→i32 truncation),
  OVER (32-bit multiply before widening), SPTR (sizeof(pointer)), DBLINIT (double init),
  UNREACH (statement after return), LOGIC (&&/|| confusion). Precision mitigations: balanced-paren
  SEMI, brace-depth UNREACH, preprocessor-reset CPASTE, case-grouping MBREAK. Registered as
  CTest `py_dev_bug_scan`. Committed `79f83220`.
- **Audit test quality scanner** (`scripts/audit_test_quality_scanner.py`) — static analyzer for
  audit C++ test files detecting patterns that cause tests to vacuously pass: A=`CHECK(true,...)`
  always-pass, B=security-rejection gap (unconditional `else{CHECK(true)}`), C=condition/message
  polarity mismatch, D=weak statistical thresholds, E=`ufsecp_*` return value silently discarded,
  F=missing unconditional reject in adversarial test. Manual audit found 17+ instances across
  KR-5/6, FAC-5/7, PSM-2/6, HB-5. Committed `79f83220`.
- **ASan/UBSan clean**: 210/210 C++ tests pass under
  `-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer` after full
  `build-asan/` rebuild. All stale binaries rebuilt and confirmed clean.
- **Schnorr BIP-340 nonce reuse key recovery** (`audit/test_exploit_schnorr_nonce_reuse.cpp`) —
  SNR-1..SNR-16: proves that reusing nonce k in two BIP-340 sigs yields full private key recovery
  via d' = (s1-s2)·(e1-e2)⁻¹ mod n. Covers known-k construction, nonce recovery, RFC6979 safety
  (different messages → different R), even-y negation case, and three-message pairwise agreement.
  Closes Schnorr NRR coverage gap. Committed `c843979c`.
- **BIP-32 non-hardened child key attack** (`audit/test_exploit_bip32_child_key_attack.cpp`) —
  CKA-1..CKA-18: proves that xpub + non-hardened child_sk → parent_sk recovery via
  parent_sk = child_sk - I_L mod n. Covers arbitrary indices, chained upward attacks (grandchild
  → child → master), hardened derivation blockage, to_public() key stripping. Committed `c843979c`.
- **FROST identifiable abort** (`audit/test_exploit_frost_identifiable_abort.cpp`) —
  FIA-1..FIA-14: proves frost_verify_partial() correctly attributes which participant submitted a
  bad partial sig. Covers single/multi-cheater detection, wrong-message cheating, identity swap,
  coordinator scan loop, honest subset completion. Committed `c843979c`.
- **Hash algorithm & format isolation** (`audit/test_exploit_hash_algo_sig_isolation.cpp`) —
  HAS-1..HAS-11: proves cross-hash-algorithm and cross-format confusion is rejected. Covers
  SHA-256 vs alt-hash confusion, raw-bytes vs digest confusion, Schnorr↔ECDSA format confusion
  (compact + DER), double-hash confusion, domain prefix isolation. Committed `c843979c`.
- **ECDSA `ecdsa_verify` large-x fix** (`src/cpu/src/ecdsa.cpp`) — corrected `r_less_than_pmn`
  comparison: wrong PMN constants (`0x402da1732fc9bebf` / `0x14551231950b75fc`) assumed p-n < 2^128;
  actual p-n = `0x14551231950b75fc4402da1722fc9baee` has limb[2]=1. Fixed in both FE52 and 4x64
  paths. Signatures where k·G.x ∈ [n, p-1] (r ≈ 2^128, probability ~2^−128 per signature) were
  erroneously rejected. Equivalent to the Stark Bank CVE-2021-43568..43572 false-negative class.
  Found and confirmed by Wycheproof tcId 346. Committed `ea8cfb3c`.
- **Wycheproof PR #206 ECDSA r-overflow tests** (`audit/test_exploit_ecdsa_r_overflow.cpp`) —
  Track I3-3: 19 checks covering k·G.x ≥ n (tcId 346 accept), r=p-3 strict-parse rejection,
  r=n→zero-reduction reject, r=0 reject, range sanity and sign/verify consistency. Closes
  Wycheproof PR #206 / Stark Bank CVE assurance gap.
- **Wycheproof ECDSA Bitcoin variant vectors** (`audit/test_wycheproof_ecdsa_bitcoin.cpp`) —
  Track I3-4: 53 checks covering BIP-62 low-S enforcement, tcId 346/347/348/351, high-S
  malleability boundary, sign/normalize/compact roundtrip, r=0/s=0 special-value rejection,
  and point-at-infinity rejection during verify.
- **Ethereum differential KAT** (`audit/test_exploit_ethereum_differential.cpp`) — 10 tests, 15 sub-checks against go-ethereum, web3.py, and ethers.js reference vectors: address derivation (go-ethereum testKey KAT), privkey=1 canonical address, ecrecover with go-ethereum test message hash, EIP-191 hash vs web3.py, sign+ecrecover roundtrip, EIP-155 v encoding, eth_personal_sign roundtrip, tamper detection, keccak256("abc") KAT, anti-collision. Closes assurance gap **#4**.
- **MuSig2/FROST/adaptor parser robustness fuzz** (`audit/test_fuzz_musig2_frost.cpp`) — 15 tests, 16 sub-checks: musig2 key_agg/nonce_agg/partial_verify/partial_sig_agg with random inputs (5000/3000/2000 rounds each), FROST keygen_finalize/sign/verify_partial/aggregate random inputs, schnorr+ecdsa adaptor random inputs, boundary test (n_signers=0 → must error). Closes assurance gap **#7**.
- **ClusterFuzzLite expanded to 5 targets**: added `src/cpu/fuzz/fuzz_ecdsa.cpp` (ECDSA sign→verify invariant, wrong-msg false-positive check, parse_compact_strict robustness) and `src/cpu/fuzz/fuzz_schnorr.cpp` (BIP-340 sign→verify, adversarial from_bytes verify, wrong-msg check).
- **LibFuzzer harnesses** (`src/cpu/fuzz/`) — 6 deterministic fuzz harnesses:
  `fuzz_der_parse.cpp` (DER signature parse + round-trip),
  `fuzz_pubkey_parse.cpp` (pubkey parse, tweak_add, encoding),
  `fuzz_schnorr_verify.cpp` (BIP-340 sign→verify + forged-sig rejection),
  `fuzz_ecdsa_verify.cpp` (ECDSA sign→verify round-trip),
  `fuzz_bip32_path.cpp` (BIP-32 path parser — boundary cases + overflow),
  `fuzz_bip324_frame.cpp` (BIP-324 AEAD frame decrypt).
  Real LibFuzzer sessions run with `SECP256K1_BUILD_LIBFUZZER=ON`. Committed `38108b89`.
- **Mutation kill-rate script** (`scripts/mutation_kill_rate.py`) — stochastic mutation
  engine that patches random source bytes and counts test failures; 50 mutations per run,
  threshold 60%. Committed `38108b89`.
- **SLSA provenance verifier** (`scripts/verify_slsa_provenance.py`) — checks `cosign`
  bundle validity, subject digest, and builder identity for release artefacts. Committed `38108b89`.
- **Cryptol formal specs** (`formal/cryptol/`) — 4 machine-checkable Cryptol property files:
  `Secp256k1Field.cry` (10 props: field axioms, Fermat, sqrt), `Secp256k1Point.cry` (7 props:
  commutativity, associativity, scalar distribution), `Secp256k1ECDSA.cry` (6 props: sign→verify,
  wrong-msg reject, sk-uniqueness), `Secp256k1Schnorr.cry` (5 props: BIP-340 round-trip,
  zero-challenge, zero-nonce-reject). Run via `cryptol --batch :check`. Committed `38108b89`.
- **`unified_audit_runner` — 3 new modules in fuzzing section** (commit `00522b57`):
  - `libfuzzer_unified` (**CI-blocking**): deterministic regression across all 6 LibFuzzer
    parser domains — sign/verify round-trip, boundary seeds, pseudo-random sweeps.
    12,097 checks pass in <250 ms. `#ifdef SECP256K1_BIP324` guard for AEAD domain.
  - `mutation_kill_rate` (advisory): popen() bridge to `scripts/mutation_kill_rate.py`;
    --ctest-mode --count 50 --threshold 60; skips gracefully if python3 absent.
  - `cryptol_specs` (advisory): popen() bridge to `cryptol --batch`; 28 formal properties
    across 4 spec files; skips gracefully if cryptol not installed.
  Fuzzing section: 8/10 → **11/11 PASS**. Full audit: **221/221 PASS**.
- **`test_exploit_ellswift_bad_scalar_ecdh` + `test_exploit_ellswift_xdh_overflow` build fix**
  (commit `00522b57`): ellswift API calls wrapped in `#ifdef SECP256K1_BIP324`; both tests
  now compile and pass in builds without BIP-324 enabled.

### Platform Support

Linux x86-64-v3 · Linux ARM64 · Linux RISC-V 64 · macOS ARM64 · Windows x86-64 · Android ARM64

### Reviewer Entry Path

1. [Bitcoin Core backend evidence](docs/BITCOIN_CORE_BACKEND_EVIDENCE.md)
2. [Benchmarks](docs/BENCHMARKS.md)
3. [Audit changelog](docs/AUDIT_CHANGELOG.md)
4. [Shim known divergences](docs/SHIM_KNOWN_DIVERGENCES.md)
5. Replay: `python3 ci/verify_external_audit_bundle.py --allow-commit-mismatch`

### Note on Scope

CPU backend (x86-64, ARM64, RISC-V). GPU backends are diagnostic/experimental — not part of the Bitcoin Core evaluation profile.

## [3.22.0] - 2026-03-10

> **Minor release: v3.21.1 -> v3.22.0** | Modular Ethereum layer | ABI extension (backward compatible)
> New feature: full Ethereum signing/recovery support with conditional build

### Added

- **Modular Ethereum layer** -- EIP-191, EIP-155, ecrecover, personal_sign with conditional
  CMake option `SECP256K1_BUILD_ETHEREUM` (default ON). Bitcoin-only builds exclude all
  Ethereum code via `-DSECP256K1_BUILD_ETHEREUM=OFF`. (#144)
- **eth_signing.hpp/.cpp** -- `eip191_hash()`, `eth_personal_sign()`, `eth_sign_hash()`,
  `ecrecover()`, `eth_personal_verify()` with EIP-155 chain ID encoding (v = 35+2*chainId+recid).
- **C ABI Ethereum functions** -- 6 new `ufsecp_*` functions: `ufsecp_keccak256`,
  `ufsecp_eth_address`, `ufsecp_eth_address_checksummed`, `ufsecp_eth_personal_hash`,
  `ufsecp_eth_sign`, `ufsecp_eth_ecrecover`. All guarded by `#ifdef SECP256K1_BUILD_ETHEREUM`.
- **Ethereum test suite** -- 32 tests across 7 groups (EIP-155 encoding, EIP-191 hash,
  eth_sign_hash, ecrecover, personal_sign+verify, multi-chain, Keccak-256 vectors).
  Registered as standalone target + `run_selftest` module.
- **Ethereum benchmarks** -- 8 benchmarks in `bench_unified` Section 6.5: keccak256, ethereum_address,
  eip191_hash, eth_sign_hash, ecdsa_sign_recoverable, ecrecover, eth_personal_sign, eip55_checksum.
- **libsecp256k1 recovery comparison** -- enabled `ENABLE_MODULE_RECOVERY=1` in libsecp_provider;
  added sign_recoverable + recover benchmarks with 3-column apple-to-apple comparison rows.
- **Ethereum audit module** -- registered in `unified_audit_runner` under `protocol_security` section
  with conditional compilation guard.

### Changed

- **coin_address.cpp** -- EIP-55 dispatch wrapped with `#ifdef SECP256K1_BUILD_ETHEREUM` guard;
  returns empty string when Ethereum module not built.
- **test_coins.cpp** -- All Ethereum-specific tests (Keccak-256, EIP-55, BIP-44 Ethereum paths)
  wrapped with `#ifdef SECP256K1_BUILD_ETHEREUM` for clean Bitcoin-only builds.

## [3.21.1] - 2026-03-09

> **Patch release: v3.21.0 -> v3.21.1** | Bug fixes, CI hardening, Metal audit | ABI compatible
> No breaking changes -- drop-in upgrade from v3.21.0

### Added

- **Metal Unified Audit Runner** -- 27 audit modules across 8 sections for Apple
  GPU (Metal) backend, matching OpenCL audit runner coverage. (#134)

### Fixed

- **CPUID PIC safety** -- replaced raw `cpuid` inline asm with `__get_cpuid_count()`
  from `<cpuid.h>` to avoid EBX register conflicts with clang PIC mode and coverage
  instrumentation. Eliminates SIGILL on clang-17 Release builds. (#135)
- **CI MARCH level** -- downgraded from `x86-64-v3` to `x86-64-v2` in CI/SonarCloud
  workflows to avoid AVX2/FMA instructions on runners without those extensions. (#135)
- **Metal metallib path** -- audit runner now searches for `secp256k1_kernels.metallib`
  (the name CMake copies) in addition to `secp256k1.metallib`. (#135)
- **Benchmark CI output path** -- moved benchmark JSON output to `/tmp/` to prevent
  git checkout conflicts with gh-pages branch. (#135)
- **OpenCL audit 6-bug fix** -- fixed 6 bugs in OpenCL backend, achieving 27/27 audit
  PASS status; dead code removal and Scorecard .sigstore cleanup. (#133)
- **macOS amd64 build** -- fixed build failure on macOS x86-64. (#132)
- **Code scanning alerts** -- resolved 6 clang-tidy findings in `test_edge_cases.cpp`:
  unused `using` declaration, const-correctness on `zero_bytes` / `zero_sig`, unchecked
  `std::remove()` returns, and redundant `one - one` expression. (#136)
- **Delta audit findings** -- addressed audit delta findings with GPU audit runners
  and documentation updates. (#126)

### Dependencies

- `benchmark-action/github-action-benchmark` bumped (#131)
- `actions/dependency-review-action` 4.8.3 -> 4.9.0 (#130)
- `github/codeql-action` 4.32.4 -> 4.32.6 (#129)
- `actions/setup-go` 6.2.0 -> 6.3.0 (#128)
- `ruby/setup-ruby` 1.288.0 -> 1.290.0 (#125)
- `actions/setup-dotnet` 5.1.0 -> 5.2.0 (#124)
- `step-security/harden-runner` 2.15.0 -> 2.15.1 (#123)
- `actions/setup-node` 6.2.0 -> 6.3.0 (#122)
- `actions/cache` 4.2.3 -> 5.0.3 (#121)

## [3.21.0] - 2026-03-08

> **Cumulative release: v3.14.0 -> v3.21.0** | 120+ commits | ABI compatible
> No breaking changes -- drop-in upgrade from v3.14.x

### Added

- **ABI layout guards** -- compile-time `static_assert` checks on struct sizes
  and constant lengths in `ufsecp.h` to catch ABI breaks. (#118)
- **docs/CRYPTO_INVARIANTS.md** -- comprehensive crypto invariant reference for
  integrators and auditors. (#118)

### Fixed

- **Precompute cache validation** -- 65 bytes/point -> 1 byte/point minimum
  bound check; path separator fix on Windows. (#118)
- **Fuzzer edge case** -- `fuzz_point` k=n-1 infinity assertion. (#118)
- **CI stabilization** -- `SECP256K1_MARCH` option to avoid ccache/`-march=native`
  mismatch SIGILL; benchmark regression filter `MIN_REGRESSION_NS=50`. (#118)
- **Packaging workflow** -- handle immutable GitHub releases by recreating with
  preserved metadata when `gh release upload --clobber` fails (HTTP 422). (#119)
- **LTO/ASM build fix** -- restrict `-flto=thin` (Clang) and `-flto` (GCC) to
  C++ sources only via `$<COMPILE_LANGUAGE:CXX>` generator expression, preventing
  the system assembler from seeing unsupported LTO flags on `.S` files. (#120)

## [3.20.0] - 2026-03-07

> **Cumulative release: v3.14.0 -> v3.20.0** | 120+ commits | ABI compatible
> No breaking changes -- drop-in upgrade from v3.14.x
>
> This release consolidates all work from v3.15.0 through v3.19.0, plus
> 19 additional commits (PRs #90--#111) into a single stable release.

### 1. Security & Constant-Time Hardening

- **RISC-V CT timing leak fixes** -- dudect testing on SiFive U74 detected 5 persistent
  timing leaks (`field_sqr`, `scalar_is_zero`, `scalar_sub`, `scalar_window`, `ct_compare`).
  Fixed with register-only `value_barrier()` and corrected `rdcycle` timer (no `fence`).
  (#v3.19.0)
- **CT SafeGCD scalar inverse** -- replaced Fermat chain (294 scalar ops) with constant-time
  Bernstein-Yang divsteps-59 port from bitcoin-core/secp256k1. `ct::scalar_inverse`:
  10,650 ns -> 1,671 ns (**6.4x faster**). CT ECDSA Sign: 26,942 ns -> 15,360 ns
  (**43% faster**). Fermat chain preserved for non-`__int128` platforms (ESP32). (#v3.18.0)
- **Secret zeroization** -- `ecdsa_sign()`, `rfc6979_nonce()`, `musig2_nonce_gen()` now
  guarantee `secure_erase()` of all intermediate secrets (k, k_inv, z, V, K, HMAC state)
  on every code path. (#v3.17.0)
- **Sign-then-verify countermeasure** -- both `ecdsa_sign()` and `ct::schnorr_sign()` verify
  the signature before returning; failure zeroes the result. (#v3.17.0)
- **BIP-340 strict parsing** -- `Scalar::parse_bytes_strict`, `FieldElement::parse_bytes_strict`,
  `SchnorrSignature::parse_strict` reject all malformed inputs. C ABI uses strict parsing
  internally. UFSECP_BITCOIN_STRICT CMake option for compile-time enforcement. (#v3.16.0)
- **CT buffer erasure** -- volatile function-pointer trick in `ct::schnorr_sign` and
  `ct::ecdsa_sign` (same technique as libsecp256k1). (#v3.16.0)
- **Hedged ECDSA** -- `ecdsa_sign_hedged()` + `rfc6979_nonce_hedged()` implementing RFC 6979
  Section 3.6 with 32-byte aux_rand mixed into HMAC-DRBG. (#v3.17.0)
- **PrivateKey strong type** -- `private_key.hpp`: wraps `fast::Scalar`, no implicit
  conversion, `secure_erase` in destructor, `[[nodiscard]]` accessors. CT overloads for
  ECDSA/Schnorr operations. (#v3.17.0)
- **Formal CT verification** -- Valgrind ctgrind (`SECP256K1_CLASSIFY`/`DECLASSIFY` markers),
  independent reference linkage (6085 cross-checks), ct-verif LLVM pass. (#v3.17.0, #v3.16.0)
- **Schnorr parity fix + branchless scalar_window** -- corrected BIP-340 parity bit,
  branchless CT implementation on RISC-V, platform-specific path on x86/ARM. (#v3.15.0)
- **Point on-curve validation** -- audited 18 deserialization paths, fixed 4 CRITICAL +
  1 HIGH + 3 LOW missing validations. (#v3.17.0)

### 2. Performance

- **L1 I-cache optimization** -- `__attribute__((noinline))` on point add/double functions
  reduced verify hot path below L1 I-cache threshold. ECDSA verify ratio vs libsecp:
  0.82x -> 0.92x (+12%). (#v3.19.0)
- **BIP-352 affine add fast path** -- 1.20x speedup for silent payment scanning. (#95)
- **FE52 fast path for scalar_mul_with_plan + GLV MSM** -- optimized multi-scalar
  multiplication with FE52-native operations. (#92)
- **ECDSA recovery 1.9x speedup** -- replaced 3 separate scalar muls with single
  `dual_scalar_mul_gen_point(u1, u2, R)` using 4-stream GLV Strauss. Recovery:
  ~69 us -> ~36 us. (#v3.15.0)
- **Precompute cache atomic write** -- write-then-rename pattern prevents CTest parallel
  test flakes. File size validation on load. (#v3.17.1)

### 3. Testing & Audit

- **Google Wycheproof ECDSA** -- 89 test cases, 10 categories. (#v3.17.0)
- **Google Wycheproof ECDH** -- 36 test cases, 7 categories. (#v3.17.0)
- **FROST RFC 9591 invariants** -- 7 ciphersuite-independent invariants + exhaustive
  3-of-5 signing across all C(5,3) = 10 subsets. (#v3.16.0)
- **MuSig2 BIP-327 vectors** -- 35 reference tests. (#v3.16.0)
- **FFI round-trip tests** -- 103 boundary tests for Schnorr, ECDSA, pubkey, ECDH,
  tweaking, and error paths. (#v3.16.0)
- **Fiat-Crypto cross-checks** -- 752 field arithmetic checks against Coq-extracted
  reference. (#v3.16.0)
- **Cross-platform audit campaign** -- 7 configurations (Windows/Linux/CI x86-64,
  ESP32-S3, RISC-V 64), all AUDIT-READY (40--49 modules each). (#v3.16.1)
- **Cross-platform benchmark campaign** -- 4 platforms (x86-64, ARM64, RISC-V, ESP32-S3)
  with identical apple-to-apple suite vs libsecp256k1 v0.7.2. (#v3.16.1)
- **ASan buffer overread fix** -- `suite_15_ffi_ecdh_edge()` 32-byte buffer corrected
  to 33-byte for compressed pubkey. (#v3.17.1)
- **Batch serialization + bench_unified coverage** -- extended benchmark suite. (#94)
- **Test count**: grew from ~29 to 31 core tests + 49 audit modules.

### 4. CI/CD & Code Quality

- **OpenSSF Scorecard hardening** -- all GitHub Actions pinned to SHA, harden-runner on
  every job, persist-credentials: false, pip hash pinning, Dependabot. (#v3.15.0, #v3.16.0)
- **CT verification CI** -- ct-arm64.yml (native Apple Silicon dudect), ct-verif.yml
  (compile-time LLVM pass), valgrind-ct.yml (taint analysis). (#v3.16.0)
- **ClusterFuzzLite** -- integrated with UBSan vptr compatibility, LTO disabled in fuzz
  builds to prevent link failures. (#v3.15.0, #108)
- **Docker local CI** -- docker-compose.ci.yml, pre-push hook, ~5 min full validation. (#v3.16.0)
- **Performance regression gate** -- per-commit benchmark with 150% threshold. (#v3.16.0)
- **SARIF output** -- `unified_audit_runner --sarif` for GitHub Code Scanning. (#v3.16.0)
- **SonarCloud Quality Gate** -- coverage 61.8% -> 85.8%, duplication below threshold,
  CPD exclusion for CT variants. (#v3.15.0)
- **5,150+ code scanning alerts resolved** -- mass clang-tidy, cppcheck, CodeQL
  remediation across v3.15.0--v3.15.3 + PRs #102, #105, #109, #111.
- **Code deduplication** -- -817 lines net across 8 files: point.cpp (-765), glv.cpp (-64),
  benchmark_harness.hpp, ufsecp_impl.cpp, selftest.cpp, field.cpp, scalar.cpp +
  new shared `detail/arith64.hpp`. (#110)
- **CI dependency bumps** -- actions/attest-build-provenance v4.1.0, sigstore/cosign-installer
  v4.0.0, step-security/harden-runner v2.15.0, actions/upload-artifact v7.0.0. (#80--#84)
- **ClusterFuzzLite + MSan failures** fixed. (#91)
- **6 CI workflow failures** resolved. (#90)

### 5. Platform Support

- **ESP32-S3** -- bench_hornet benchmark data, 40-module audit. (#107, #v3.16.1)
- **WASM / Emscripten** -- `SECP256K1_NO_INT128` auto-defined, `FAST_52BIT` disabled,
  precompute generator bypass, GLV+Shamir fallback. (#v3.15.0)
- **ARM64 Android** -- bench_hornet port with `clock_gettime`, libsecp_bench.c for
  cross-compilation. (#v3.16.1)
- **RISC-V real hardware** -- Milk-V Mars benchmarks, 4/6 ops faster (2.02x--3.08x),
  value_barrier register-only fix. (#v3.16.1, #v3.19.0)
- **Preprocessor branch repair** -- fixed broken conditional compilation in point ops. (#104)

### 6. Build & Packaging

- **Benchmark diagnostics** -- Schnorr verify sub-operation diagnostics (SHA256, FE52_inv,
  parse_strict) added to bench_unified. (#v3.19.0)
- **Build hardening** -- clean `-Werror -Wall -Wextra -Wpedantic` build, fixed
  `-Wsign-conversion` in SafeGCD, `-Wstringop-overflow` in base58. (#v3.19.0, #v3.15.1)
- **MSVC / GCC / Clang compatibility** -- resolved `__int128` pedantic warnings, using
  declaration restoration, duplicate const qualifiers. (#v3.15.0, #v3.15.1)
- **Audit UX** -- centralized CHECK macro with ASCII progress bar, Windows stdout fix. (#v3.16.0)
- **z_one_ member fix** -- removed from constructor initializer lists, restored as member
  with normalize() methods. (#96, #97)
- **SonarCloud fixes** -- `FieldElement52::to_bytes_into()` deduplication, null checks,
  crypto impl exclusion. (#93, #98, #99)

### 7. Documentation

- **BENCHMARKING.md** -- complete guide for all 4 platforms.
- **AUDIT_GUIDE.md** -- 40/48-module audit how-to.
- **FROST_COMPLIANCE.md** -- RFC 9591/BIP-FROST checkpoint matrix.
- **COMPATIBILITY.md** -- BIP-340 strict encoding notes.
- **BINDINGS_ERROR_MODEL.md** -- strict semantics for binding authors.
- **ADOPTERS.md** -- production/development/hobby adopter categories.
- **GitHub Discussion templates** -- Q&A, Show-and-Tell, Ideas, Integration Help.

### Cross-Platform Benchmark Results (vs libsecp256k1 v0.7.2)

#### x86-64 (i5-14400F @ 2.50 GHz, GCC 14.2.0, Ubuntu 24.04)

| Operation | Ultra FAST | Ultra CT | libsecp256k1 | Ratio (fast) | Ratio (CT) |
|-----------|-----------|----------|-------------|-------------|-----------|
| ECDSA sign | 7.45 us | 13.48 us | 17.86 us | **2.40x** | **1.33x** |
| ECDSA verify | 20.39 us | -- | 21.93 us | **1.08x** | -- |
| Schnorr sign | 5.86 us | 10.85 us | 12.58 us | **2.15x** | **1.16x** |
| Schnorr verify | 21.49 us | -- | 22.57 us | **1.05x** | -- |
| k*G | 5.39 us | 9.67 us | 12.78 us | **2.37x** | **1.32x** |

#### x86-64 (i7-11700 @ 2.50 GHz, Clang 21.1.0)

| Operation | Ultra FAST | Ultra CT | libsecp256k1 | Ratio (fast) | Ratio (CT) |
|-----------|-----------|----------|-------------|-------------|-----------|
| ECDSA sign | 8.06 us | 15.74 us | 21.67 us | **2.69x** | **1.38x** |
| ECDSA verify | 29.06 us | -- | 26.62 us | 0.92x | -- |
| Schnorr sign | 6.42 us | 13.59 us | 17.07 us | **2.66x** | **1.26x** |
| Schnorr verify | 28.67 us | -- | 27.72 us | 0.97x | -- |
| k*G | 4.29 us | 11.86 us | 17.59 us | **4.10x** | **1.48x** |

#### ARM64 (Cortex-A55 @ YF_022A, Clang 18.0.1 NDK r27)

| Operation | Ultra FAST | Ultra CT | libsecp256k1 | Ratio (fast) | Ratio (CT) |
|-----------|-----------|----------|-------------|-------------|-----------|
| ECDSA sign | 27.98 us | 71.91 us | 76.35 us | **2.73x** | **1.06x** |
| ECDSA verify | 146.95 us | -- | 148.42 us | **1.01x** | -- |
| Schnorr sign | 20.11 us | 64.00 us | 64.95 us | **3.23x** | **1.02x** |
| Schnorr verify | 147.59 us | -- | 149.06 us | **1.01x** | -- |
| k*G | 17.46 us | -- | 63.20 us | **3.62x** | -- |

#### RISC-V 64 (SiFive U74-MC @ 1.5 GHz, GCC 13.3.0, Milk-V Mars)

| Operation | Ultra FAST | Ultra CT | libsecp256k1 | Ratio (fast) | Ratio (CT) |
|-----------|-----------|----------|-------------|-------------|-----------|
| ECDSA sign | 81.25 us | 159.25 us | 164.12 us | **2.02x** | **1.03x** |
| ECDSA verify | 235.50 us | -- | 221.37 us | 0.94x | -- |
| Schnorr sign | 56.37 us | 133.45 us | 133.01 us | **2.36x** | 1.00x |
| Schnorr verify | 239.44 us | -- | 225.07 us | 0.94x | -- |
| k*G | 40.60 us | -- | 125.05 us | **3.08x** | -- |

#### ESP32-S3 (Xtensa LX7 @ 240 MHz, GCC 14.2.0, ESP-IDF 5.5.1)

| Operation | Ultra FAST | Ultra CT | libsecp256k1 | Ratio (fast) | Ratio (CT) |
|-----------|-----------|----------|-------------|-------------|-----------|
| ECDSA sign | 7,600 us | 7,951 us | 9,538 us | **1.25x** | **1.20x** |
| ECDSA verify | 18,446 us | -- | 29,329 us | **1.59x** | -- |
| Schnorr sign | 6,640 us | 7,051 us | 9,451 us | **1.42x** | **1.34x** |
| Schnorr verify | 19,023 us | -- | 27,203 us | **1.43x** | -- |
| k*G | 6,273 us | -- | 7,214 us | **1.15x** | -- |

#### BIP-352 Silent Payments Pipeline ([bench_bip352](https://github.com/shrec/bench_bip352))

External standalone benchmark isolating the full BIP-352 scanning pipeline:
k\*P -> serialize -> tagged\_SHA256 -> k\*G -> point\_add -> serialize -> prefix match.
Equalized compiler flags (`-O3 -march=native`), 10K ops, 11 passes, median.

##### x86-64 (i5-14400F @ 2.50 GHz, GCC 14.2.0, Ubuntu 24.04)

| Operation | libsecp256k1 | UltrafastSecp256k1 | Ratio |
|---|---:|---:|---:|
| k\*P (scalar mul) | 21,083 ns | 16,689 ns | **1.26x** |
| k\*G (generator mul) | 11,082 ns | 5,149 ns | **2.15x** |
| k\*G (precomputed tables) | -- | 4,396 ns | **2.52x** |
| Point addition | 1,794 ns | 1,336 ns | **1.34x** |
| Tagged SHA-256 | 457 ns | 43 ns | **10.6x** |
| Serialize compressed | 22 ns | 9 ns | **2.4x** |
| **Full pipeline** | **33,642 ns** | **25,079 ns** | **1.34x** |

##### ARM64 (Cortex-A55 @ YF_022A, Clang 18.0.3 NDK r27)

| Operation | libsecp256k1 | UltrafastSecp256k1 | Ratio |
|---|---:|---:|---:|
| k\*P (scalar mul) | 131,694 ns | 130,596 ns | **1.01x** |
| k\*G (generator mul) | 59,199 ns | 16,056 ns | **3.69x** |
| k\*G (precomputed tables) | -- | 12,626 ns | **4.69x** |
| Point addition | 8,161 ns | 3,232 ns | **2.52x** |
| Tagged SHA-256 | 971 ns | 431 ns | **2.25x** |
| Serialize compressed | 44 ns | 12 ns | **3.7x** |
| **Full pipeline** | **200,289 ns** | **153,385 ns** | **1.31x** |

##### RISC-V 64 (SiFive U74-MC @ 1.5 GHz, GCC 13.3.0, Milk-V Mars)

| Operation | libsecp256k1 | UltrafastSecp256k1 | Ratio |
|---|---:|---:|---:|
| k\*P (scalar mul) | 196,364 ns | 201,162 ns | 0.98x |
| k\*G (generator mul) | 133,519 ns | 45,016 ns | **2.97x** |
| k\*G (precomputed tables) | -- | 38,995 ns | **3.42x** |
| Point addition | 14,699 ns | 5,449 ns | **2.70x** |
| Tagged SHA-256 | 5,227 ns | 1,688 ns | **3.10x** |
| Serialize compressed | 274 ns | 131 ns | **2.1x** |
| **Full pipeline** | **354,234 ns** | **257,996 ns** | **1.37x** |

Validation prefix: `0xb63b4601066a6971` (all platforms, both libraries match).

---

*Detailed per-version changelogs follow below for historical reference.*

---

## [3.19.0] - 2026-03-04

> No breaking changes -- drop-in upgrade from v3.18.x | ABI compatible
> Focus: RISC-V constant-time hardening, L1 I-cache optimization, -Werror clean build

### 1. RISC-V Constant-Time Timing Leak Fixes
- **Root cause**: dudect testing on SiFive U74 (in-order core) detected 5 persistent
  timing leaks (|t| > 10): `field_sqr`, `scalar_is_zero`, `scalar_sub`, `scalar_window`,
  `ct_compare`. The compiler (Clang 21) reordered instructions across barriers, and
  store-buffer retirement latency created data-dependent timing differences.
- **Fix (v1)**: Added `"memory"` clobber to RISC-V `value_barrier()` and explicit barriers
  at critical points in `ct_field.cpp` and `ct_scalar.cpp`.
- **Fix (v2)**: Reverted to register-only barrier `asm volatile("" : "+r"(v))` on RISC-V.
  The `"memory"` clobber forced store-to-load-forwarding sequences with data-dependent
  retirement latency on U74's store buffer (zero-coalescing), creating false-positive
  dudect leaks in `field_add`, `field_is_zero`, and `scalar_add`. Register-only is
  sufficient on in-order cores because the pipeline cannot reorder past `asm volatile`.
- **rdtsc fix**: Removed hardware `fence` from RISC-V `rdcycle` timer -- the fence drained
  the store buffer synchronously, capturing its data-dependent retirement latency and
  producing false-positive timing leaks. Matches x86 `rdtscp` and ARM64 `cntvct_el0`
  behavior (neither drains the store buffer).
- Files changed: `src/cpu/include/secp256k1/ct/ops.hpp`, `src/cpu/src/ct_field.cpp`,
  `src/cpu/src/ct_scalar.cpp`, `audit/test_ct_sidechannel.cpp`

### 2. L1 I-Cache Optimization (ECDSA Verify)
- **Root cause**: ECDSA verify performance was 0.82x vs libsecp256k1 due to L1 instruction
  cache thrashing. Aggressive inlining of point arithmetic functions (`jac52_add`,
  `jac52_double`, etc.) caused the hot verify loop to exceed L1 I-cache capacity (32 KB).
- **Fix**: Added `__attribute__((noinline))` to point add/double functions, reducing code
  size in the verify hot path below L1 I-cache threshold.
- **Performance**: ECDSA verify ratio vs libsecp: 0.82x -> 0.92x (+12% improvement)
- Files changed: `src/cpu/src/point.cpp`

### 3. Benchmark Diagnostics
- Added Schnorr verify sub-operation diagnostics (SHA256, FE52_inv, parse_strict) to
  `bench_unified.cpp` for identifying verify bottlenecks.
- Files changed: `src/cpu/bench/bench_unified.cpp`

### 4. Build Hardening
- Fixed `-Wsign-conversion` warnings in `ct_scalar.cpp` SafeGCD `divsteps_59()` function.
  Added explicit `static_cast` for `int64_t` <-> `uint64_t` conversions that were
  previously implicit. Clean `-Werror -Wall -Wextra -Wpedantic` build.
- Files changed: `src/cpu/src/ct_scalar.cpp`

### x86-64 Benchmark Results (i7-11700 @ 2.50 GHz, Clang 21.1.0)

| Operation | Ultra FAST | Ultra CT | libsecp256k1 | Ratio (fast) | Ratio (CT) |
|-----------|-----------|----------|-------------|-------------|-----------|
| ECDSA sign | 8.06 us | 15.74 us | 21.67 us | **2.69x** | **1.38x** |
| ECDSA verify | 29.06 us | -- | 26.62 us | 0.92x | -- |
| Schnorr sign | 6.42 us | 13.59 us | 17.07 us | **2.66x** | **1.26x** |
| Schnorr verify | 28.67 us | -- | 27.72 us | 0.97x | -- |
| k*G | 4.29 us | 11.86 us | 17.59 us | **4.10x** | **1.48x** |

## [3.18.0] - 2026-03-04

> No breaking changes -- drop-in upgrade from v3.17.x | ABI compatible
> Focus: CT scalar inverse -- SafeGCD replaces Fermat chain, 6.4x faster CT signing

### CT Scalar Inverse: SafeGCD (Bernstein-Yang constant-time divsteps)
- **Root cause**: `ct::scalar_inverse()` used Fermat's Little Theorem chain (a^{n-2} mod n)
  with 254 squarings + 40 multiplications = 294 scalar ops, costing ~10,650 ns.
  This single function accounted for ~40% of CT ECDSA sign latency.
- **Fix**: Replaced with constant-time SafeGCD (Bernstein-Yang divsteps-59), port of
  `secp256k1_modinv64` from bitcoin-core/secp256k1. Fixed 10 rounds x 59 branchless
  divsteps = 590 total. All loops fixed-count, all conditionals use bitmasks.
  No `ctz`, no early termination, no secret-dependent branches.
- **Performance**: `ct::scalar_inverse`: 10,650 ns -> 1,671 ns (**6.4x faster**)
- **Impact on CT ECDSA Sign**: 26,942 ns -> 15,360 ns (**43% faster**)
- **CT vs libsecp ECDSA Sign ratio**: 0.80x (lose) -> **1.44x (win)**
- **Fallback**: Fermat chain preserved for platforms without `__int128` (ESP32, etc.)
- Files changed: `src/cpu/src/ct_scalar.cpp`, `src/cpu/include/secp256k1/ct/scalar.hpp`,
  `src/cpu/bench/bench_unified.cpp`
- All 32 tests pass (excluding ct_sidechannel long-run)

## [3.17.1] - 2026-03-05

> No breaking changes -- drop-in upgrade from v3.17.0 | ABI compatible
> Focus: CI reliability fixes -- precompute cache race condition + ASan buffer overread

### 1. Precompute Cache Atomic Write (bip340_vectors CI flake fix)
- **Root cause**: `save_precompute_cache_locked()` wrote directly to `cache_w{N}.bin`. When CTest runs tests in parallel (`-j$(nproc)`), a reader process could see a partially-written cache file from another writer, loading corrupt precompute tables. This caused intermittent `bip340_vectors` failures where `scalar_mul_generator()` produced wrong results for large scalars (higher window tables not yet written).
- **Fix**: Atomic write-then-rename pattern -- cache is written to `cache_w{N}.bin.tmp.{pid}`, then atomically renamed to `cache_w{N}.bin`. Readers always see either the old complete file or the new complete file.
- **Additional hardening**: `load_precompute_cache_locked()` now validates expected file size (computed from header `window_count * digit_count * 65 + sizeof(CacheHeader)`) before reading point data. Truncated or partially-written files are rejected immediately.
- Files changed: `src/cpu/src/precompute.cpp`

### 2. ASan Buffer Overread Fix (fuzz_address_bip32_ffi suite 15a)
- **Root cause**: `suite_15_ffi_ecdh_edge()` passed `uint8_t xpub[32]` (32 bytes) to `ufsecp_ecdh_xonly()` which expects `const uint8_t pubkey33[33]` (33-byte compressed pubkey). ASan detected a 1-byte stack-buffer-overflow in `FieldElement::parse_bytes_strict()`.
- **Fix**: Changed buffer to `uint8_t xpub[33]` with `fill_random(xpub, 33)` to correctly match the API's 33-byte compressed pubkey parameter.
- Files changed: `audit/test_fuzz_address_bip32_ffi.cpp`

## [3.17.0] - 2026-03-04

> No breaking changes -- drop-in upgrade from v3.16.x | ABI compatible
> Focus: Track I Crypto Auditor Gaps -- industrial-grade hardening campaign (16/16 items DONE)

### 1. Secret Zeroization (I1)
- **[I1-1] ECDSA fast-path scalar zeroization** -- `ecdsa_sign()` rewritten with single-cleanup-path structure; `k`, `k_inv`, `z` guaranteed `secure_erase()` on all code paths
- **[I1-2] RFC 6979 HMAC state zeroization** -- `rfc6979_nonce()`: V, K, x_bytes, buf97 all zeroed before every return via `secure_erase()`
- **[I1-3] MuSig2 secret zeroization** -- `sk_bytes`, `aux_hash`, `t` zeroed after XOR block in `musig2_nonce_gen()`

### 2. Fault Attack Countermeasures (I2)
- **[I2-1] ECDSA sign-then-verify** -- `ecdsa_sign()` verifies signature before returning; failure zeroes result. Both fast and CT paths hardened
- **[I2-2] Schnorr sign-then-verify** -- `ct::schnorr_sign()` verifies via `schnorr_verify()` before returning; failure returns empty signature

### 3. Test Vector Coverage (I3)
- **[I3-1] Google Wycheproof ECDSA** -- `audit/test_wycheproof_ecdsa.cpp`: 89 test cases covering 10 categories (valid, invalid r/s, modified, boundary, wrong key/msg, infinity, high-S, known vectors, Schnorr invalid, degenerate). 89/89 passed
- **[I3-2] Google Wycheproof ECDH** -- `audit/test_wycheproof_ecdh.cpp`: 36 test cases covering 7 categories (valid ECDH, infinity, off-curve, zero key, commutativity, point validation, variant consistency). 36/36 passed

### 4. API Misuse Resistance (I4)
- **[I4-1] MuSig2 nonce CT migration** -- `fast::Point::generator().scalar_mul()` replaced with `ct::generator_mul()` in MuSig2 nonce generation
- **[I4-2] Point on-curve validation** -- audited 18 deserialization paths; fixed 4 CRITICAL + 1 HIGH + 3 LOW missing validations
- **[I4-3] PrivateKey strong type** -- new `src/cpu/include/secp256k1/private_key.hpp`: PrivateKey wrapping `fast::Scalar`, no implicit conversion, `secure_erase` in destructor, `[[nodiscard]]` accessors. CT overloads for `ecdsa_sign`, `schnorr_pubkey`, `schnorr_keypair_create`
- **[I4-4] aux_rand entropy contract** -- comprehensive BIP-340 aux_rand documentation on `schnorr_sign()` (both overloads) and CT variant: CSPRNG requirement, XOR nonce hedging, all-zeros safety, reuse warnings

### 5. Formal Verification (I5)
- **[I5-1] Formal CT verification (Valgrind ctgrind)** -- `audit/test_ct_verif_formal.cpp`: marks secrets as undefined via SECP256K1_CLASSIFY(), runs CT operations (ECDSA sign, Schnorr sign, field/scalar/point ops), then declassifies. Any secret-dependent branch triggers Valgrind error. Same technique as libsecp256k1 `valgrind_ctime_test.c` and BoringSSL `constant_time_test`
- **[I5-2] Independent reference linkage** -- `audit/test_fiat_crypto_linkage.cpp`: independent schoolbook secp256k1 reference implementation embedded directly; 6085 cross-checks at function level (mul, sqr, add, sub, neg) with 100% output parity against an independent oracle

### 6. Protocol-Level Hardening (I6)
- **[I6-1] Hedged ECDSA** -- `ecdsa_sign_hedged()` + `rfc6979_nonce_hedged()` implementing RFC 6979 Section 3.6 with 32-byte aux_rand mixed into HMAC-DRBG. Both fast and CT variants. Sign-then-verify + secure_erase included
- **[I6-2] FROST BIP-387 compliance** -- `docs/FROST_COMPLIANCE.md`: RFC 9591/BIP-FROST checkpoint matrix covering DKG, nonce generation, signing, verification, aggregation. 5 deviation notes, 4 recommendations
- **[I6-3] Batch verify randomness audit** -- `audit/test_batch_randomness.cpp`: 1022 checks confirming hash-derived (SHA256) deterministic weights -- not PRNG-dependent. Bellare-Garay immune

### 7. Test Suite Growth
- Test count: 29 -> 31 (added ct_verif_formal, fiat_crypto_linkage)
- All 31 tests pass (31/31)

## [3.16.1] - 2026-03-02

> No breaking changes -- drop-in upgrade from v3.16.0 | ABI compatible
> Focus: cross-platform benchmark campaign and audit on real hardware

### 1. Cross-Platform Benchmark Campaign (bench_hornet)
- Benchmarked **4 platforms** with an identical 6-operation apple-to-apple suite against bitcoin-core/libsecp256k1 v0.7.2:
  - x86-64: Intel i7-11700 @ 2.50 GHz, Clang 21.1.0 (Windows)
  - ARM64: Cortex-A55 (YF_022A), Clang 18.0.1 (Android NDK 27)
  - RISC-V 64: SiFive U74-MC @ 1.5 GHz, GCC 13.3.0 (Milk-V Mars, real hardware)
  - ESP32-S3: Xtensa LX7 @ 240 MHz, GCC 14.2.0 (ESP-IDF 5.5.1, real hardware)
- Added a **CT-vs-CT fair comparison** since libsecp256k1 is always constant-time; the separate CT-vs-CT section shows true relative performance for signing operations
- Generated **13 report files** (JSON + TXT per platform, plus a cross-platform comparison)

### 2. Cross-Platform Audit Campaign (unified_audit_runner)
- **7 platform configurations, all AUDIT-READY** (48/49 or 40/40 modules depending on platform):
  - Windows x86-64 (Clang 21.1.0): 48/49 PASS
  - Linux Docker x86-64 (GCC 13.3.0): 48/49 PASS
  - Linux CI x86-64 (Clang 17.0.6): 46/46 PASS
  - Linux CI x86-64 (GCC 13.3.0): 46/46 PASS
  - Windows CI x86-64 (MSVC 1944): 45/45 PASS
  - ESP32-S3 real hardware (GCC 14.2.0): 40/40 PASS (8 modules skipped as platform-incompatible)
  - RISC-V 64 real hardware (GCC 13.3.0): 48/49 PASS (0 modules skipped, 1 advisory)
- Updated PLATFORM_AUDIT.md with all 7 configurations

### 3. ARM64 Android Benchmark Port
- **bench_hornet_android.cpp** -- complete bench_hornet port for ARM64 Android using `clock_gettime`, median-of-5 timing, and a 32-key pool
- **libsecp_bench.c** -- libsecp256k1 apple-to-apple benchmark for Android NDK cross-compilation
- **android/CMakeLists.txt** -- added C language support, bench_hornet target, and LIBSECP_SRC_DIR

### 4. RISC-V Benchmark on Real Hardware
- Cross-compiled bench_hornet for rv64gc_zba_zbb and deployed to Milk-V Mars via SCP
- Results: UltrafastSecp256k1 wins 4 of 6 operations (2.02x--3.08x faster), loses both Verify variants (0.94x)
- CT-vs-CT comparison: signing operations are essentially tied (1.00x--1.03x); Verify remains at 0.94x

### 5. Build & CI Fixes
- **audit-report.yml** -- scoped `security-events` permission to job level instead of workflow level
- **Dockerfile.ci** -- pinned ubuntu:24.04 to a SHA-256 digest for reproducibility
- **sanitizer_scale.hpp** -- added iteration scaling for sanitizer builds
- **ESP32 audit** -- expanded sdkconfig, CMakeLists, and audit_main to support 40-module auditing
- **.gitignore** -- expanded to exclude local scratch files, build logs, and temporary scripts

### 6. Documentation
- **BENCHMARKING.md** -- complete guide covering how to run bench_hornet on all 4 platforms
- **AUDIT_GUIDE.md** -- complete guide covering how to run the 40/48-module audit on any platform
- Updated examples README with stability markers ([STABLE], [EXPERIMENTAL])

## [3.16.0] - 2026-03-01

> No breaking changes -- drop-in upgrade from v3.15.x | ABI compatible

### 1. Security Hardening
- **BIP-340 strict parsing** -- added `Scalar::parse_bytes_strict`, `FieldElement::parse_bytes_strict`, and `SchnorrSignature::parse_strict`, which reject all malformed inputs (#73)
- **CT buffer erasure** -- `ct::schnorr_sign` and `ct::ecdsa_sign` now erase intermediate nonces via a volatile function-pointer trick (same technique used by libsecp256k1)
- **lift_x deduplication** -- consolidated duplicated code in Schnorr verify/sign into a single `static lift_x()` helper
- **Y-parity fix** -- switched to `limbs()[0] & 1` instead of a byte-level parity check
- **Pragma balance fix** -- removed a misbalanced `#pragma GCC diagnostic push/pop` pair in ct_field.cpp

### 2. Audit Infrastructure
- **Advisory flag** -- marked `ct_sidechannel_smoke` as advisory in unified_audit_runner so that timing flakes on shared CI runners do not fail the overall audit
- **carry_propagation cross-validation** -- the test now verifies the generator-optimized path against the generic GLV path and prints hex diagnostics on ARM64 mismatch
- **BIP-340 strict test suite** -- added 31 tests covering reject-zero, reject-overflow, reject-p-plus, and accept-valid scenarios for all strict parsing APIs

### 3. Local CI (Docker)
- **docker-compose.ci.yml** -- single-command orchestration for all 14 CI jobs
- **pre-push target** -- `docker compose run --rm pre-push` validates warnings, tests, ASan, and the audit in approximately 5 minutes
- **audit job** -- `docker/run_ci.sh audit` mirrors audit-report.yml (GCC-13 + Clang-17)
- **ccache integration** -- Docker volume persistence for fast rebuilds
- **pre-push hook** -- `scripts/hooks/pre-push` blocks pushes on CI failure
- **PowerShell wrapper** -- `scripts/pre-push-ci.ps1` for Windows users

### 4. Documentation
- **COMPATIBILITY.md** -- BIP-340 strict encoding compatibility notes
- **BINDINGS_ERROR_MODEL.md** -- BIP-340 strict semantics for binding authors
- **SECURITY.md** -- updated Memory Handling section (library-side erasure), Planned items checklist, and API Stability references
- **UFSECP_BITCOIN_STRICT** -- new CMake option to enforce strict-only parsing at compile time

### 5. Build & CI
- **packaging.yml** -- fixed a release workflow race condition (`gh release upload` with retry)
- **C ABI** -- `ufsecp_schnorr_verify`, `ufsecp_schnorr_sign`, and `ufsecp_xonly_pubkey_parse` now use strict parsing internally

### 6. CT Verification CI
- **ct-arm64.yml** -- native ARM64 / Apple Silicon dudect on macos-14 M1: smoke tests per-PR, full suite nightly
- **ct-verif.yml** -- compile-time constant-time verification via the ct-verif LLVM pass (deterministic, not statistical)
- **valgrind-ct.yml** -- Valgrind `MAKE_MEM_UNDEFINED` taint analysis that detects secret-dependent branches at the binary level
- **MuSig2/FROST dudect** -- protocol-level timing tests for `musig2_partial_sign`, `frost_sign`, and `frost_lagrange_coefficient`

### 7. Audit Infrastructure (SARIF & Regression)
- **SARIF output** -- `unified_audit_runner --sarif` generates SARIF v2.1.0 reports for GitHub Code Scanning
- **bench-regression.yml** -- per-commit performance regression gate with a 120% threshold (fail-on-alert)
- **audit-report.yml** -- now uploads SARIF results to GitHub Code Scanning (linux-gcc job)

### 8. OpenSSF Scorecard Hardening
- **Pinned actions** -- all GitHub Actions pinned to full SHA (codeql-action v4.32.4, upload-artifact v6.0.0)
- **harden-runner** -- added to discord-commits and packaging RPM jobs
- **persist-credentials: false** -- applied to all checkout steps with write permissions (benchmark, docs, packaging, release, bench-regression)
- **Standardized versions** -- audited and hardened 13 workflow files

### 9. FROST RFC 9591 Protocol Invariant Tests
- **test_rfc9591_invariants** -- verifies 7 ciphersuite-independent invariants: verification_share = signing_share * G, Lagrange interpolation of Y_i, Feldman VSS commitment, partial signature linearity, partial signature verification, wrong-share rejection, and nonce commitment consistency
- **test_rfc9591_3of5** -- exhaustive 3-of-5 FROST signing across all C(5,3) = 10 participant subsets with BIP-340 verification
- **valgrind_ct_check.sh** -- fixed binary path (audit/ instead of cpu/) for `test_ct_sidechannel_standalone`

### 10. Audit UX
- **audit_check.hpp** -- centralized CHECK macro with a 20-character ASCII progress bar (`[####................] N OK`), reporting every 4096 iterations
- **22 audit .cpp files** -- migrated from per-file CHECK macros to the shared `audit_check.hpp`
- **Windows stdout fix** -- applied `setvbuf(stdout, nullptr, _IONBF, 0)` for unbuffered output on Windows (avoids `_IOLBF` crash)

### 11. New Audit Modules
- **test_musig2_bip327_vectors.cpp** -- 35 BIP-327 MuSig2 reference tests covering key aggregation, nonce aggregation, signing, and verification
- **test_ffi_round_trip.cpp** -- 103 FFI round-trip boundary tests covering Schnorr, ECDSA, pubkey, ECDH, tweaking, and error paths
- **test_fiat_crypto_vectors.cpp** -- expanded to 752 cross-checks of field arithmetic against the Fiat-Crypto reference implementation

### 12. Community
- **ADOPTERS.md** -- added production, development, and hobby adopter categories
- **GitHub Discussion templates** -- added Q&A, Show-and-Tell, Ideas, and Integration Help categories

## [3.15.3] - 2026-03-01

### Fixed -- Code Quality (136 code scanning alerts resolved)
- **bench_hornet.cpp** -- 73 fixes covering const-correctness, braces, cert-err33-c, modernize-use-auto, implicit-widening, reserved-identifier, and init-variables
- **glv.cpp** -- 33 fixes for const-correctness in the GLV_MULADD macro and k_arr array
- **audit_integration.cpp** -- 10 fixes for const-correctness, cert-err33-c, and sizes[] arrays
- **point.cpp** -- 5 fixes for const-correctness on Jacobian addition intermediates
- **precompute.cpp** -- 2 fixes: modernize-use-auto and simplify-boolean-expr
- Dismissed 12 containerOutOfBounds alerts (Cppcheck false positives)

## [3.15.1] - 2026-03-01

### Fixed -- Build Compatibility (MSVC / WASM / armv7 / GCC -Wpedantic)
- **schnorr.cpp** -- `FieldElement::from_bytes()` was called with `const uint8_t*` instead of `const std::array<uint8_t,32>&`; added a copy into `std::array` before the call. This had broken MSVC, WASM, and armv7 builds.
- **glv.cpp** -- suppressed the GCC `-Wpedantic` warning for the `__int128` extension type using `#pragma GCC diagnostic push/pop`.
- **glv.cpp** -- removed the unused `mul_shift_384` runtime function in the `__int128` path (only the template `mul_shift_384_const` is used).

## [3.15.0] - 2026-03-01

> 104 commits since v3.14.0 | 368 files changed | +45,388 / -7,639 lines
> No breaking changes -- drop-in upgrade from v3.14.0 | ABI compatible -- SOVERSION unchanged

### 1. Security & Constant-Time Hardening
- **Schnorr parity fix** -- corrected the parity bit computation in BIP-340 signatures (#48)
- **Z=0 guard deduplication** -- added point edge-case tests (#49)
- **CT branchless scalar_window** -- branchless implementation on RISC-V, branched on x86/ARM (#42--#44)
- **value_barrier** -- added after mask derivation in `ct_compare`, plus a WASM KAT target
- **is_zero_mask** -- RISC-V branchless assembly with triple barrier and `rdcycle` fence
- **reverse-scan ct_compare** -- uses an interleaved test data pattern

### 2. WASM / Emscripten Support
- **SECP256K1_NO_INT128** -- automatically defined on Emscripten
- **SECP256K1_FAST_52BIT** -- disabled for Emscripten targets
- **Precompute generator bypass** -- avoids timeouts on WASM
- **GLV+Shamir fallback** -- replaced wNAF w=5 with an optimal double-and-add implementation
- **KAT test** -- resolved `SINGLE_FILE=1` and ESM conflicts

### 3. CI/CD Infrastructure
- **OpenSSF Scorecard** -- pinned all actions to SHA and added harden-runner (#52)
- **pip deps pinned by hash** -- improved supply chain security (#52)
- **ClusterFuzzLite** -- integrated with UBSan `vptr` sanitizer compatibility
- **Cppcheck + Mutation testing + SARIF** -- added new CI workflows
- **Fuzz + Protocol tests** -- enabled in all CI jobs

### 4. Code Quality (~5,150 alerts fixed)
- **~4,600 code scanning alerts** -- mass cleanup (#53)
- **~550 code scanning alerts** -- batch 2 (#56)
- **Duplicate const qualifiers** -- fixed GCC-13 build failure (#54, #55)
- **using declarations** -- restored declarations removed by clang-tidy, required for MSVC/ESP32/WASM (#57)
- **audit_field.cpp** -- fixed unused variable `-Werror` failure (#58)

### 5. SonarCloud Quality Gate
- **SHA-256 SonarCloud blocker** -- suppressed S3519 `buf_` overflow false positive (#50, #60, #61)
- **Coverage** -- raised from 61.8% to 85.8% (exclusions: audit/, include/ufsecp/) (#59)
- **Duplication** -- reduced from 3.3% to below threshold via CT variant CPD exclusion (#61)
- **cpp:S876** -- suppressed CT masking unsigned negation warning (#59)
- **Codecov exclusions** -- corrected configuration (#50)

### 6. Audit Framework
- **A--M audit framework** -- complete audit scripts with a cross-platform test plan
- **audit_ct** -- raised timing sanity threshold for CI from 1.5x to 2.0x
- **AUDIT_COVERAGE.md** -- full CI infrastructure documentation
- **Unified runner + CI workflow** -- added evidence collection scripts

### 7. Testing
- **MuSig2 + FROST** -- advanced protocol tests (Phase II)
- **Parser fuzz** -- DER, Schnorr, and Pubkey fuzzing
- **Cross-library differential test** -- verified against bitcoin-core/libsecp256k1
- **Address/BIP32/FFI fuzz tests** -- added
- **FROST KAT tests** -- added
- **Point edge-case tests** -- added
- **FE52 Jacobian is_on_curve** -- added for `FAST_52BIT` platforms
- **FieldElement::operator== normalize** -- handles non-canonical limb values correctly

### 8. Build & Platform
- **MSVC** -- added `SECP256K1_NOINLINE` macro and fixed `s_gen4` race condition
- **Reproducible builds** -- signed releases with SBOM
- **Fuzz point** -- avoided precomputed-table timeouts under sanitizers

### 9. Performance -- ECDSA Recovery (1.9x speedup)
- **`ecdsa_recover()` rewritten** -- replaced 3 separate scalar multiplications (`s*R`, `z*G`, `r^-1 * result`) with a single call to `dual_scalar_mul_gen_point(u1, u2, R)` using 4-stream GLV Strauss with interleaved wNAF. Recovery now matches libsecp256k1 performance (~36 us vs. the previous ~69 us).
- **`lift_x()` parity optimization** -- replaced `to_bytes()` serialization (32-byte encode) with a direct `limbs()[0] & 1` parity check for y-coordinate odd/even detection.
- **Dudect cache artifact false positives** -- fixed 11 smoke-test false positives in constant-time side-channel tests by tightening thresholds and isolating cache effects.

### 10. Platform Assembly
- **ARM64** -- added CSEL branchless conditionals and EXTR optimization for field squaring.
- **RISC-V** -- applied preload optimization for field multiply assembly and reduced register pressure in `field_asm52_riscv64.S`.
- **Field operations** -- refactored `field.cpp` with improved Montgomery path selection.

### 11. Apple-to-Apple Benchmark
- **`bench_apple_to_apple`** -- definitive head-to-head benchmark against libsecp256k1 v0.6.0 covering 13 operations with the same compiler, flags, and assembly. Uses IQR outlier removal and median-of-11 passes. Result: **7 FASTER, 5 EQUAL, 0 SLOWER** (geometric mean 0.68x = UltrafastSecp256k1 is 1.47x faster on the 13-op suite). **Note:** this geometric mean is not weighted by real-world operation frequency. Workloads dominated by k\*G (signing, key generation) will see the full benefit; workloads dominated by k\*P (ECDH, BIP-352 scanning, key tweaking) may not -- see `bench_unified` ratio table for per-operation breakdown ([#87](https://github.com/shrec/UltrafastSecp256k1/issues/87)).

### 12. Documentation & Bindings (continuation of v3.14.0)
- **Release artifacts** -- signed SHA256SUMS manifest with verification instructions
- **ABI versioning policy** -- documented
- **7 Phase III documents** -- covering audit, invariants, bug bounty, thread safety, and more
- **User guide + FAQ** -- added

---

## [3.14.0] - 2026-02-25

### Added -- Language Bindings (12 languages, 41-function C API parity)
- **Java** -- 22 new JNI functions and 3 helper classes (`RecoverableSignature`, `WifDecoded`, `TaprootOutputKeyResult`): full coverage of ECDSA sign/verify, DER encoding, recovery, ECDH, Schnorr, BIP-32, BIP-39, taproot, WIF, address encoding, and tagged hash
- **Swift** -- 20 new functions: DER encode/decode, recovery sign/recover, ECDH, tagged hash, BIP-32/39, taproot, WIF, and address encoding
- **React Native** -- 15 new functions: DER, recovery, ECDH, Schnorr, BIP-32/39, taproot, WIF, address, and tagged hash
- **Python** -- 3 new functions: `ctx_clone()`, `last_error()`, `last_error_msg()`
- **Rust** -- 2 new functions: `last_error()`, `last_error_msg()`
- **Dart** -- 1 new function: `ctx_clone()`
- **Go, Node.js, C#, Ruby, PHP** -- already complete (verified; no changes needed)
- **9 new binding READMEs** -- for `c_api`, `dart`, `go`, `java`, `php`, `python`, `ruby`, `rust`, and `swift`
- **Selftest report API** -- added `SelftestReport` and `SelftestCase` structs in `selftest.hpp`; `tally()` refactored for programmatic reporting

### Fixed -- Documentation & Packaging
- **Package naming corrected across all documentation** -- renamed `libsecp256k1-fast*` to `libufsecp*` (apt, rpm, arch); CMake target `secp256k1-fast-cpu` to `secp256k1::fast`; linker flag `-lsecp256k1-fast-cpu` to `-lfastsecp256k1`; pkg-config Libs `-lsecp256k1-fast-cpu` to `-lfastsecp256k1`
- **RPM spec renamed** -- from `libsecp256k1-fast.spec` to `libufsecp.spec`
- **Debian control** -- source `libufsecp`, binary packages `libufsecp3`/`libufsecp-dev`
- **Arch PKGBUILD** -- `pkgname=libufsecp`, `provides=('libufsecp')`
- **3 existing binding READMEs fixed** -- Node.js, C#, and React Native: removed inaccurate CT-layer claims (the C API uses the `fast::` path only)
- **README dead link** -- fixed `INDUSTRIAL_ROADMAP_WORKING.md` to point to `ROADMAP.md`

### Fixed -- CI / Build
- **`-Werror=unused-function`** -- added `[[maybe_unused]]` to `get_platform_string()` in `selftest.cpp`
- **Scorecard CI** -- pinned `ubuntu:24.04` by SHA digest in `Dockerfile.local-ci`

---

## [3.13.1] - 2026-02-24

### Fixed
- **Critical: GLV decomposition overflow in `ct::scalar_mul()`** -- `ct_mul_256x_lo128_mod` used a single-phase reduction (256x128-bit), which overflowed when GLV's `c1`/`c2` rounded to exactly 2^128. Additionally, the `lambda*k2` computation only read 2 lower limbs of `k2_abs`, silently dropping `limb[2]=1`. This caused incorrect results for approximately 5 out of 64 random scalar inputs. Replaced with a full `ct_scalar_mul_mod_n()`: 4x4 schoolbook multiply producing an 8-limb product, followed by 3-phase `reduce_512` (512 -> 385 -> 258 -> 256 bits), matching libsecp256k1's algorithm. Both the `5x52` (`__int128`) and `4x64` (portable `U128`/`mul64`) paths were fixed.
- **GLV constant `minus_b2`** -- changed from a 128-bit `b2_pos` to a full 256-bit `Scalar(n - b2)`, and updated the decomposition formula from `scalar_sub(p1, p2)` to `scalar_add(p1, p2)` since both constants are already negated
- **`-Werror=unused-function`** -- added `[[maybe_unused]]` to diagnostic helpers `print_scalar()` and `print_point_xy()` in `diag_scalar_mul.cpp`

### Removed
- Dead code: `ct_mul_lo128_mod()` and `ct_mul_256x_lo128_mod()` (replaced by `ct_scalar_mul_mod_n`)

### Performance
- CT `scalar_mul` overhead vs. the fast path: **1.05x** (25.3 us vs. 24.0 us) -- no regression

---

## [3.13.0] - 2026-02-24

### Added
- **BIP-32 official test vectors TV1--TV5** -- 90 comprehensive checks covering master key derivation, hardened/normal child paths, and public-only derivation chains (`test_bip32_vectors.cpp`)
- **Nightly CI workflow** -- daily extended verification with a differential correctness check using a 100x multiplier (~1.3M checks) and dudect full-mode statistical analysis (30 min, t=4.5 threshold)
- **Differential test CLI/env multiplier** -- `differential_test` accepts `--multiplier=N` or the `UFSECP_DIFF_MULTIPLIER` environment variable; default 1 preserves existing CI behavior

### Fixed
- **BIP-32 public key decompression** -- `public_key()` now correctly decompresses from the compressed prefix + x-coordinate via the y^2 = x^3 + 7 square root with a parity check; previously it treated the x-coordinate as a scalar, producing incorrect public keys for public-only derivation
- **`pub_prefix` field** in `ExtendedKey` -- now stores the y-parity byte (0x02/0x03) across `to_public()`, `derive_child()`, and `serialize()` for correct compressed public key round-trip
- **SonarCloud `ct_sidechannel` exclusion** -- changed `-E ct_sidechannel` to the exact-match `-E "^ct_sidechannel$"` to prevent accidental exclusion of other tests

---

## [3.12.3] - 2026-02-24

### Fixed
- **Valgrind "still reachable" false positives** -- added `valgrind.supp` suppression file for precomputed wNAF/comb table allocations that are intentionally kept for program lifetime
- **CTest memcheck integration** -- switched from `enable_testing()` to `include(CTest)` for proper Valgrind memcheck support
- **Security audit CI** -- added `--suppressions` flag and exact-match `ct_sidechannel` exclusion in Valgrind step
- **ASan heap-buffer-overflow** in dudect smoke mode -- fixed buffer overread in timing analysis
- **aarch64 cross-compilation** -- added missing toolchain file for ARM64 CI builds

---

## [3.12.2] - 2026-02-24

### Security
- **Branchless `ct_compare`** -- rewritten with bitwise arithmetic and `asm volatile` value barriers; dudect |t| dropped from 22.29 -> 2.17, eliminating a timing side-channel leak

### Fixed
- **SonarCloud coverage collection** -- use `run_selftest` as primary llvm-cov binary (links full library); coverage report now reflects actual test execution
- **Dead code elimination in `precompute.cpp`** -- `RDTSC()` gated behind `SECP256K1_PROFILE_DECOMP`; `multiply_u64`/`mul64x64`/`mul_256` unified to call `_umul128()` instead of duplicating `__int128` inline
- **GCC `#pragma clang diagnostic` warnings** -- wrapped in `#ifdef __clang__` guards in 3 test files
- **GCC `-Wstringop-overflow`** -- bounds check in `base58check_encode` (address.cpp)
- **All `-Werror` warnings resolved** -- 41 files across library, tests, and benchmarks
- **Clang-tidy CI** -- filter `.S` assembly from analysis, add `--quiet` and parallel `xargs`
- **Unused variable** -- removed `compressed` in `bip32.cpp` `to_public()`

### Changed
- **`const` on hot-path intermediates** -- ~60 `FieldElement52` write-once variables in `point.cpp` marked `const`
- **Benchmark exclusion** -- `sonar-project.properties` excludes benchmark files from coverage calculation
- **CPD minimum tokens** -- set to 100 in `sonar-project.properties`

### Added
- **GOVERNANCE.md** -- BDFL governance model with continuity plan (bus factor)
- **ROADMAP.md** -- 12-month project roadmap (Mar 2026 - Feb 2027)
- **CONTRIBUTING.md** -- Developer Certificate of Origin (DCO) requirement
- **OpenSSF Best Practices badge** -- added to README
- **Code scanning fixes** -- resolved alerts #281, #282

---

## [3.12.1] - 2026-02-23

### Security
- **bump wheel 0.45.1 -> 0.46.2** -- fixes CVE-2026-24049 (path traversal in `wheel unpack`)
- **bump setuptools 75.8.0 -> 78.1.1** -- fixes CVE-2025-47273 (path traversal via vendored wheel)

### Changed
- **VERSION.txt** updated to 3.12.1

---

## [3.12.0] - 2026-02-23

### Security -- CI/CD Hardening & Supply-Chain Protection
- **SHA-pinned all GitHub Actions** -- every action uses immutable commit SHA instead of mutable tags
- **Harden Runner** -- `step-security/harden-runner` v2.14.2 on every CI job (egress audit)
- **CodeQL** -- upgraded to v4.32.4, job-level `security-events: write`, custom query filters
- **OpenSSF Scorecard** -- daily scorecard workflow with SARIF upload
- **SonarCloud** -- CI-based code quality analysis with build-wrapper
- **pip hash pinning** -- `--require-hashes` on all pip install steps in release/CI workflows
- **Dependabot** -- configured for GitHub Actions, pip, npm, NuGet, Cargo ecosystems
- **Branch protection** -- required reviews, dismiss stale, strict status checks on `main`

### Fixed
- **66+ code scanning alerts resolved** -- unused variables, permissions, hardcoded credentials, scorecard findings
- **StepSecurity remediation** -- merged PR #25 with fixes for GHA best practices

### Changed
- **Dependabot PRs #26-#32 merged** -- codeql-action v4.32.4, setup-dotnet v5.1.0, upload-artifact v6.0.0, download-artifact v7.0.0, scorecard-action v2.4.3, attest-build-provenance v3.2.0, sonarqube-scan-action v7.0.0
- **Rust workspace Cargo.toml** -- added for Dependabot Cargo ecosystem support

### Added
- **`docs/CODING_STANDARDS.md`** -- comprehensive coding standards for OpenSSF CII badge
- **`CONTRIBUTING.md` requirements section** -- explicit contribution requirements with links
- **Full AGPL-3.0 LICENSE text** -- replaced summary with standard text for GitHub license detection

---

## [3.11.0] - 2026-02-23

### Performance -- Effective-Affine & RISC-V Optimization
- **Effective-affine GLV table** -- batch-normalize P-multiples to affine in `scalar_mul_glv52`, eliminating Z-coordinate arithmetic from the main loop. Point Add 821->159 ns on x86-64.
- **RISC-V auto-detect CPU** -- CMake reads `/proc/cpuinfo` uarch field to set `-mcpu=sifive-u74` automatically. **28-34% speedup** on Milk-V Mars (Scalar Mul 235->154 us).
- **RISC-V ThinLTO propagation** -- ARCH_FLAGS propagated via INTERFACE compile+link options so ThinLTO codegen uses correct CPU scheduling at link time.
- **RISC-V Zba/Zbb fix** -- explicit `-march=rv64gc_zba_zbb` alongside `-mcpu` since Clang's sifive-u74 model omits these extensions.
- **ARM64 10x26 field representation** -- verified as optimal for Cortex-A76 (74 ns mul vs 100 ns with 5x52).

### Performance -- Embedded
- **SafeGCD30 field inverse** -- GCD-based modular inverse for non-`__int128` platforms: ESP32 **118 us** (was 3 ms).
- **SafeGCD30 scalar inverse** -- same technique for scalar field; optimized SHA-256/HMAC/RFC-6979 for embedded.
- **ESP32 4-stream GLV Strauss** -- parallel endomorphism streams + Z^2-verify optimization.
- **CT layer optimizations** -- comprehensive CT optimization pass for embedded targets.

### Changed
- **Unified benchmark harness** -- all 4 bench binaries share common framework with IQR outlier removal and RDTSCP/chrono auto-selection.
- **CMake 4.x compatibility** -- standalone build support with `cmake_minimum_required(3.18)` + project-level CTest.
- **Disable RISC-V FE52 asm** -- C++ `__int128` inline is 26-33% faster than hand-written FE52 assembly on RISC-V.
- **Benchmark data refresh** -- all platforms re-measured: x86-64 (Clang 21), ARM64 (RK3588), RISC-V (Milk-V Mars).
- **Remove competitor comparison tables** -- benchmarks show only UltrafastSecp256k1 results.

### Added
- **Lightning donation** -- `shrec@stacker.news` badge in README.
- **ARM64 5x52 MUL/UMULH kernel** -- interleaved multiply for exploration (10x26 remains default).
- **ESP32 comprehensive benchmark** -- full benchmark matching x86 format.

### Fixed
- **CI Unicode cleanup** -- replaced all Unicode characters with ASCII across codebase.
- **CI benchmark parse fix** -- reset baseline for Unicode-free benchmark output.
- **Orphaned submodule** -- removed stale `cpu/secp256k1` submodule entry.

### Acknowledgments
- Stacker News, Delving Bitcoin, and @0xbitcoiner for community support.

---

## [3.10.0] - 2026-02-21

### Performance -- CT Hot-Path Optimization (Phases 5-15)
- **5x52 field representation** -- switched point internals from 4x64 to `FieldElement52`, enabling `__int128` lazy reduction across all CT operations
- **Direct asm bypass** -- CT `field_mul`/`field_sqr` now call hand-tuned 5x52 multiply/square directly: **70 ns -> 33 ns**
- **GLV endomorphism** -- CT `scalar_mul` via lambda-decomposition + interleaved double-and-add: **304 us -> 20 us**
- **CT generator_mul precomputed table** -- 16-entry precomputed-G table with batch inversion: **310 us -> 9.8 us (31x speedup)**
- **Batch inversion + Brier-Joye unified add** -- Montgomery's trick for multi-point normalization
- **Hamburg signed-digit + batch doubling** -- compact signed-digit recoding with merged double passes
- **128-bit split + w=15 for G-stream verify** -- Shamir-style dual-stream with wider window: **~14% verify speedup**
- **AVX2 CT table lookup** -- `_mm256_cmpeq_epi64` + `_mm256_and_si256` constant-time table scan
- **Effective-affine P table** -- batch-normalize P-multiples to skip Z-coordinate arithmetic in main loop
- **Schnorr keypair/pubkey caching + FE52 sqrt** -- avoid redundant serialization in sign/verify
- **FE52-native inverse + isomorphic table build + GCD `inv_var`** -- SafeGCD field inverse stays in 52-bit form
- **Format conversion elimination** -- removed `to_fe()`/`from_fe()` round-trips on every CT hot path
- **Redundant normalize elimination** -- `ct_field_mul_impl`/`square_impl` produce already-reduced results
- **Schnorr X-check + Y-parity combined** -- single Z-inverse for both x-coordinate check and y-parity in FE52

### Performance -- I-Cache Optimization
- **`noinline` on `jac52_add_mixed_inplace`** -- prevents inlining of 800+ byte function body into tight loops: **59% I-cache miss reduction**

### Fixed
- **`scalar_mul_glv52` infinity guard** -- early return on `base.is_infinity() || scalar.is_zero()` prevents zero-inverse crash in Montgomery batch trick (CI #128-131 regression)
- **CT `complete_add` fallback** -- uses affine `x()`/`y()` instead of raw Jacobian `X()`/`Y()`
- **MSVC fallback** -- `field_neg` arity, `is_equal_mask`, GLV decompose, `y_bytes` redefinition
- **Cross-platform FE52 guard** -- `SECP256K1_FAST_52BIT` gating prevents compilation on 32-bit targets

### Changed
- **Dead code removal** -- removed functions superseded by Z-ratio normalization path
- **Barrett -> specialized GLV multiplies** -- replaced generic Barrett reduction with curve-specific multiply

### CI / Infrastructure
- **npm/nuget publishing fix** -- corrected CI workflow for package publishing
- **Comprehensive audit suite** -- 8 suites, 641K checks, cryptographic correctness validation
- **CT operations benchmark** -- `bench_ct_vs_libsecp` with per-operation ns/op and throughput
- **dudect timing test** -- side-channel timing leakage detection for CT operations
- **Doxyfile version auto-injection** -- `VERSION.txt` -> `Doxyfile` at configure time

---

## [3.6.0] - 2026-02-20

### Added -- GPU Signature Operations (CUDA)
- **ECDSA Sign on GPU** -- `ecdsa_sign_batch_kernel` with RFC 6979 deterministic nonces, low-S normalization. **204.8 ns / 4.88 M/s** per signature.
- **ECDSA Verify on GPU** -- `ecdsa_verify_batch_kernel` with Shamir's trick + GLV endomorphism. **410.1 ns / 2.44 M/s** per verification.
- **ECDSA Sign Recoverable on GPU** -- `ecdsa_sign_recoverable_batch_kernel` with recovery ID computation. **311.5 ns / 3.21 M/s**.
- **ECDSA Recover on GPU** -- `ecdsa_recover_batch_kernel` for public key recovery from signature + recid.
- **Schnorr Sign (BIP-340) on GPU** -- `schnorr_sign_batch_kernel` with tagged hash midstates. **273.4 ns / 3.66 M/s**.
- **Schnorr Verify (BIP-340) on GPU** -- `schnorr_verify_batch_kernel` with x-only pubkey verification. **354.6 ns / 2.82 M/s**.
- **6 new batch kernel wrappers** in `secp256k1.cu` -- all with `__launch_bounds__(128, 2)` matching scalar_mul kernels.
- **5 GPU signature benchmarks** in `bench_cuda.cu` -- ECDSA sign, verify, sign+recid, Schnorr sign, Schnorr verify.
- **`prepare_ecdsa_test_data()`** helper -- generates valid signatures on GPU for verify benchmark correctness.

> As of this release, this is the only multi-backend (CUDA + OpenCL + Metal) GPU secp256k1 library with CT-signed batch ECDSA and Schnorr sign/verify that the authors are aware of.

### Changed
- **CUDA benchmark numbers updated** -- Scalar Mul improved to 225.8 ns (was 266.5 ns), Field Inv to 10.2 ns (was 12.1 ns) from `__launch_bounds__` thread count fix (128 vs 256 mismatch).
- **README** -- Added blockchain coin badges (Bitcoin, Ethereum, +25), GPU signature benchmark tables, 27-coin supported coins section, SEO metadata footer, updated performance headline.
- **BENCHMARKS.md** -- Split CUDA section into Core ECC + GPU Signature Operations; updated all comparison tables.

### Fixed
- **CUDA benchmark thread mismatch** -- Benchmarks used 256 threads/block but kernels declared `__launch_bounds__(128, 2)`, causing 0.0 ns results. Fixed to use 128 threads.

---

## [3.4.0] - 2026-02-19

### Added -- Stable C ABI (`ufsecp`)
- **Complete C ABI library** -- `ufsecp.dll` / `libufsecp.so` / `libufsecp.dylib` with 45 exported symbols, opaque `ufsecp_ctx` handle, and structured error model (11 error codes)
- **Headers**: `ufsecp.h` (main API, 37 functions), `ufsecp_version.h` (ABI versioning), `ufsecp_error.h` (error codes)
- **Implementation**: `ufsecp_impl.cpp` wrapping C++ core into C-linkage with zero heap allocations on hot paths
- **Build system**: `include/ufsecp/CMakeLists.txt` -- shared + static build, standalone or sub-project mode, pkg-config template (`ufsecp.pc.in`)
- **API coverage**: key generation, ECDSA sign/verify/recover, Schnorr BIP-340 sign/verify, SHA-256, ECDH (compressed/xonly/raw), BIP-32 HD derivation, Bitcoin addresses (P2PKH/P2WPKH/P2TR), WIF encode/decode, DER serialization, public key tweak (add/mul), selftest
- **`SUPPORTED_GUARANTEES.md`** -- Tier 1/2/3 stability guarantees documentation
- **`examples/hello_world.c`** -- Minimal usage example

### Added -- Dual-Layer Constant-Time Architecture
- **Always-on dual layers** -- `secp256k1::fast::*` (public operations) and `secp256k1::ct::*` (secret-key operations) are always active simultaneously; no flag-based selection
- **CT layer** -- Complete addition formula (12M+2S), fixed-trace scalar multiplication, constant-time table lookup
- **Valgrind/MSAN markers** -- `SECP256K1_CLASSIFY()` / `SECP256K1_DECLASSIFY()` for verifiable constant-time guarantees

### Added -- SHA-256 Hardware Acceleration
- **SHA-NI hardware dispatch** -- Runtime CPUID detection for Intel SHA Extensions; transparent fallback to software implementation
- **Zero-overhead dispatch** -- Function pointer set once at init, no branching in hot path

### Added -- C# P/Invoke Bindings & Benchmarks
- **`bindings/csharp/UfsepcBenchmark/`** -- .NET 8.0 project with complete P/Invoke declarations for all 45 `ufsecp` functions
- **68 correctness tests** -- 12 categories covering key ops, ECDSA, Schnorr, SHA-256, ECDH, BIP-32, addresses, DER round-trip, recovery, WIF, tweaks, selftest
- **19 benchmarks** -- SHA-256: 137ns, ECDSA Sign: 11.89us, Verify: 47.95us, Schnorr Sign: 10.68us, KeyGen: 1.22us
- **P/Invoke overhead measured** -- ~10-40ns per call (negligible)

### Changed
- `ufsecp_ctx_create()` takes no flags parameter -- dual-layer CT architecture is always active

---

## [3.3.0] - 2026-02-16

### Added -- Comprehensive Benchmarks
- **Metal GPU benchmark** (`bench_metal.mm`): 9 operations -- Field Mul/Add/Sub/Sqr/Inv, Point Add/Double, Scalar Mul (Pxk), Generator Mul (Gxk). Matches CUDA benchmark format with warmup, kernel-only timing, and throughput tables.
- **3 new Metal GPU kernels**: `field_add_bench`, `field_sub_bench`, `field_inv_bench` in `secp256k1_kernels.metal`
- **WASM benchmark** (`bench_wasm.mjs`): Node.js benchmark for all WASM-exported operations -- Pubkey Create (Gxk), Point Mul, Point Add, ECDSA Sign/Verify, Schnorr Sign/Verify, SHA-256 (32B/1KB)
- WASM benchmark runs automatically in CI (Node.js 20 setup + execution)

### Added -- Security & Maturity
- SECURITY.md v3.2 with vulnerability reporting guidelines
- THREAT_MODEL.md with detailed threat analysis
- API stability guarantees documented
- Fuzz testing documentation and libFuzzer harnesses
- Selftest modes: smoke (fast), ci (full), stress (extended)
- Repro bundle support for deterministic test reproduction
- Sanitizer CI integration (ASan/UBSan/TSan)

### Added -- Testing
- Boundary KAT vectors for field limb boundaries
- Batch inverse sweep tests
- Unified test runner (12 test files consolidated into single runner)

### Added -- Documentation
- Batch inverse & mixed addition API reference with examples (full point, X-only, CUDA, division, scratch reuse, Montgomery trick)
- CHANGELOG.md (this file), CODE_OF_CONDUCT.md
- Benchmark dashboard link in README

### Changed
- Benchmark alert threshold 120% -> 150% (reduces false positive alerts on shared CI runners)
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

### Added -- Coins Layer
- **Multi-coin infrastructure** -- `coins/coin_params.hpp` with constexpr `CoinParams` definitions for 27 secp256k1-based cryptocurrencies: Bitcoin, Litecoin, Dogecoin, Dash, Ethereum, Bitcoin Cash, Bitcoin SV, Zcash, DigiByte, Namecoin, Peercoin, Vertcoin, Viacoin, Groestlcoin, Syscoin, BNB Smart Chain, Polygon, Avalanche, Fantom, Arbitrum, Optimism, Ravencoin, Flux, Qtum, Horizen, Bitcoin Gold, Komodo
- **Unified address generation** -- `coin_address()`, `coin_address_p2pkh()`, `coin_address_p2wpkh()`, `coin_address_p2tr()` with automatic encoding dispatch per coin (Base58Check / Bech32 / EIP-55)
- **Per-coin WIF encoding** -- `coin_wif_encode()` with coin-specific prefix bytes
- **Full key derivation pipeline** -- `coin_derive()` takes private key + CoinParams -> public key + address + WIF in one call
- **Coin registry** -- `find_by_ticker("BTC")`, `find_by_coin_type(60)`, `ALL_COINS[]` array for iteration

### Added -- Ethereum & EVM Support
- **Keccak-256 hash** -- Standard Keccak-256 (NOT SHA3-256; Ethereum-compatible 0x01 padding), incremental API (`Keccak256State::update/finalize`), one-shot `keccak256()` (`coins/keccak256.hpp`, `src/keccak256.cpp`)
- **Ethereum addresses (EIP-55)** -- `ethereum_address()` with mixed-case checksummed output, `ethereum_address_raw()`, `ethereum_address_bytes()`, `eip55_checksum()`, `eip55_verify()` (`coins/ethereum.hpp`, `src/ethereum.cpp`)
- **EVM chain compatibility** -- Same address derivation works for BSC, Polygon, Avalanche, Fantom, Arbitrum, Optimism

### Added -- BIP-44 HD Derivation
- **Coin-type derivation** -- `coin_derive_key()` with automatic purpose selection: BIP-86 (Taproot) for Bitcoin, BIP-84 (SegWit) for Litecoin, BIP-44 (legacy) for Dogecoin/Ethereum
- **Path construction** -- `coin_derive_path()` builds `m/purpose'/coin_type'/account'/change/index`
- **Seed-to-address pipeline** -- `coin_address_from_seed()` full pipeline: seed -> BIP-32 master -> BIP-44 derivation -> coin address

### Added -- Custom Generator Point & Curve Context
- **CurveContext** -- `context.hpp` with custom generator point support, curve order (raw bytes), cofactor, and name (`CurveContext::secp256k1_default()`, `CurveContext::with_generator()`, `CurveContext::custom()`)
- **Context-aware operations** -- `derive_public_key(privkey, &ctx)`, `scalar_mul_G(scalar, &ctx)`, `effective_generator(&ctx)` -- nullptr = standard secp256k1, custom context = custom G
- **Zero-overhead default** -- Standard secp256k1 usage with nullptr context has no extra cost

### Added -- Tests
- **test_coins** -- 32 tests covering CurveContext, CoinParams registry, Keccak-256 vectors, EIP-55 checksum, Bitcoin/Litecoin/Dogecoin/Dash/Ethereum addresses, WIF encoding, BIP-44 path/derivation, custom generator derivation, full multi-coin pipeline

---

## [3.1.0] - 2026-02-15

### Added -- Cryptographic Protocols
- **Pedersen Commitments** -- `pedersen_commit(value, blinding)`, `pedersen_verify()`, `pedersen_verify_sum()` (homomorphic balance proofs), `pedersen_blind_sum()`, `pedersen_switch_commit()` (Mimblewimble switch commitments); nothing-up-my-sleeve generators H and J via SHA-256 try-and-increment (`src/cpu/include/pedersen.hpp`, `src/cpu/src/pedersen.cpp`)
- **FROST Threshold Signatures** -- `frost_keygen_begin()` / `frost_keygen_finalize()` (Feldman VSS distributed key generation), `frost_sign_nonce_gen()` / `frost_sign()` (partial signature rounds), `frost_verify_partial()`, `frost_aggregate()` -> standard BIP-340 SchnorrSignature; `frost_lagrange_coefficient()` helper (`src/cpu/include/frost.hpp`, `src/cpu/src/frost.cpp`)
- **Adaptor Signatures** -- Schnorr adaptor: `schnorr_adaptor_sign()`, `schnorr_adaptor_verify()`, `schnorr_adaptor_adapt()`, `schnorr_adaptor_extract()`; ECDSA adaptor: `ecdsa_adaptor_sign()`, `ecdsa_adaptor_verify()`, `ecdsa_adaptor_adapt()`, `ecdsa_adaptor_extract()` -- for atomic swaps and DLCs (`src/cpu/include/adaptor.hpp`, `src/cpu/src/adaptor.cpp`)
- **MuSig2 multi-signatures (BIP-327)** -- Key aggregation (KeyAgg), deterministic nonce generation, 2-round signing protocol, partial sig verify, Schnorr-compatible aggregate signatures (`src/cpu/include/musig2.hpp`, `src/cpu/src/musig2.cpp`)
- **ECDH key exchange** -- `ecdh_compute` (SHA-256 of compressed point), `ecdh_compute_xonly` (SHA-256 of x-coordinate), `ecdh_compute_raw` (raw x-coordinate) (`src/cpu/include/ecdh.hpp`, `src/cpu/src/ecdh.cpp`)
- **ECDSA public key recovery** -- `ecdsa_sign_recoverable` (deterministic recid), `ecdsa_recover` (reconstruct pubkey from signature + recid), compact 65-byte serialization (`src/cpu/include/recovery.hpp`, `src/cpu/src/recovery.cpp`)
- **Taproot (BIP-341/342)** -- Tweak hash, output key computation, private key tweaking, commitment verification, TapLeaf/TapBranch hashing, Merkle root/proof construction (`src/cpu/include/taproot.hpp`, `src/cpu/src/taproot.cpp`)
- **BIP-32 HD key derivation** -- Master key from seed, hardened/normal child derivation, path parsing (m/0'/1/2h), Base58Check serialization (xprv/xpub), RIPEMD-160 fingerprinting (`src/cpu/include/bip32.hpp`, `src/cpu/src/bip32.cpp`)
- **BIP-352 Silent Payments** -- `silent_payment_address()`, `SilentPaymentAddress::encode()`, `silent_payment_create_output()`, `silent_payment_scan()` with ECDH-based stealth addressing and multi-output support (`src/cpu/include/address.hpp`, `src/cpu/src/address.cpp`)

### Added -- Address & Encoding
- **Bitcoin Address Generation** -- `hash160()` (RIPEMD-160 + SHA-256), `base58check_encode()` / `base58check_decode()`, `bech32_encode()` / `bech32_decode()` (BIP-173/BIP-350, Bech32/Bech32m), `address_p2pkh()`, `address_p2wpkh()`, `address_p2tr()`, `wif_encode()` / `wif_decode()` (`src/cpu/include/address.hpp`, `src/cpu/src/address.cpp`)

### Added -- Core Algorithms
- **Multi-scalar multiplication** -- Shamir's trick (2-point) + Strauss interleaved wNAF (n-point) (`src/cpu/include/multiscalar.hpp`, `src/cpu/src/multiscalar.cpp`)
- **Batch signature verification** -- Schnorr and ECDSA batch verify with random linear combination; `identify_invalid()` to pinpoint bad signatures (`src/cpu/include/batch_verify.hpp`, `src/cpu/src/batch_verify.cpp`)
- **SHA-512** -- Header-only implementation for HMAC-SHA512 / BIP-32 (`src/cpu/include/sha512.hpp`)
- **Constant-time byte utilities** -- `ct_equal`, `ct_is_zero`, `ct_compare`, `ct_memzero` (volatile + asm barrier), `ct_memcpy_if`, `ct_memswap_if`, `ct_select_byte` (`src/cpu/include/ct_utils.hpp`)

### Added -- Performance
- **AVX2/AVX-512 SIMD batch field ops** -- Runtime CPUID detection, auto-dispatching `batch_field_add/sub/mul/sqr`, Montgomery batch inverse (1 inversion + 3(n-1) multiplications) (`src/cpu/include/field_simd.hpp`, `src/cpu/src/field_simd.cpp`)

### Added -- GPU Optimization
- **Occupancy auto-tune utility** -- `gpu_occupancy.cuh` with `optimal_launch_1d()` (uses `cudaOccupancyMaxPotentialBlockSize`), `query_occupancy()`, and startup device diagnostics
- **Warp-level reduction primitives** -- `warp_reduce_sum()`, `warp_reduce_sum64()`, `warp_reduce_or()`, `warp_broadcast()`, `warp_aggregated_atomic_add()` in reusable header
- **`__launch_bounds__` on library kernels** -- `field_mul/add/sub/inv_kernel` (256,4), `scalar_mul_batch/generator_mul_batch_kernel` (128,2), `point_add/dbl_kernel` (256,4), `hash160_pubkey_kernel` (256,4)

### Added -- Build & Packaging
- **PGO build scripts** -- `build_pgo.sh` (Linux, Clang/GCC auto-detect) and `build_pgo.ps1` (Windows, MSVC/ClangCL)
- **MSVC PGO support** -- CMakeLists.txt now handles `/GL` + `/GENPROFILE` / `/USEPROFILE` for MSVC in addition to Clang/GCC
- **vcpkg manifest** -- `vcpkg.json` with optional features (asm, cuda, lto)
- **Conan 2.x recipe** -- `conanfile.py` with CMakeToolchain integration and shared/fPIC/asm/lto options
- **Benchmark dashboard CI** -- GitHub Actions workflow (`benchmark.yml`) running benchmarks on Linux + Windows, `parse_benchmark.py` for JSON output, `github-action-benchmark` integration with 120% alert threshold

### Added -- Tests (237 new)
- `test_v4_features` -- 90 tests: Pedersen (basic/homomorphic/balance/switch/serialization/zero-value), FROST (Lagrange/keygen/2-of-3 signing), Adaptor (Schnorr basic/ECDSA basic/identity), Address (Base58Check/Bech32/Bech32m/hash160/P2PKH/P2WPKH/P2TR/WIF/consistency), Silent Payments (address/flow/multi-output)
- `test_ecdh_recovery_taproot` -- 76 tests: ECDH, Recovery, Taproot, CT Utils, Wycheproof vectors
- `test_multiscalar_batch` -- 16 tests: Shamir edge cases, multi-scalar sums, Schnorr & ECDSA batch verify
- `test_bip32` -- 28 tests: HMAC-SHA512 vectors, BIP-32 TV1 master/child keys, path derivation, serialization
- `test_musig2` -- 19 tests: key aggregation, nonce generation, 2-of-2 & 3-of-3 signing
- `test_simd_batch` -- 8 tests: SIMD detection, batch add/sub/mul/sqr, batch inverse

### Fixed
- **SHA-512 K[23] constant** -- Single-bit typo (`0x76f988da831153b6` -> `0x76f988da831153b5`) that caused all SHA-512 hashes to be incorrect
- **MuSig2 per-signer Y parity** -- `musig2_partial_sign()` now negates the secret key when the signer's public key has odd Y (required for x-only pubkey compatibility)

---

## [3.0.0] - 2026-02-11

### Added -- Cryptographic Primitives
- **ECDSA (RFC 6979)** -- Deterministic signing & verification (`src/cpu/include/ecdsa.hpp`)
- **Schnorr BIP-340** -- x-only signing & verification (`src/cpu/include/schnorr.hpp`)
- **SHA-256** -- Standalone hash, zero-dependency (`src/cpu/include/sha256.hpp`)
- **Constant-time benchmarks** -- CT layer micro-benchmarks via CTest

### Added -- Platform Support
- **iOS** -- CMake toolchain, XCFramework build script, SPM (`Package.swift`), CocoaPods (`UltrafastSecp256k1.podspec`), C++ umbrella header
- **WebAssembly (Emscripten)** -- C API (11 functions), JS wrapper (`secp256k1.mjs`), TypeScript declarations, npm package `@ultrafastsecp256k1/wasm`
- **ROCm / HIP** -- CUDA <-> HIP portability layer (`gpu_compat.h`), all 24 PTX asm blocks guarded with `#if SECP256K1_USE_PTX` + portable `__int128` alternatives, dual CUDA/HIP CMake build
- **Android NDK** -- arm64-v8a CI build with NDK r27c

### Added -- Infrastructure
- **CI/CD (GitHub Actions)** -- Linux (gcc-13/clang-17 x Release/Debug), Windows (MSVC), macOS (AppleClang), iOS (OS + Simulator + XCFramework), WASM (Emscripten), Android (NDK), ROCm (Docker)
- **Doxygen -> GitHub Pages** -- Auto-generated API docs on push to main
- **Fuzzing harness** -- `tests/fuzz_field.cpp` for libFuzzer field arithmetic testing
- **Version header** -- `cmake/version.hpp.in` auto-generates `SECP256K1_VERSION_*` macros
- **`.clang-format` + `.editorconfig`** -- Consistent code formatting
- **Desktop example app** -- `examples/desktop_example.cpp` with CTest integration
- **CMake install** -- `install(TARGETS)` + `install(DIRECTORY)` for system-wide deployment

### Changed
- **Search kernels relocated** -- `src/cuda/include/` -> `cuda/app/` (cleaner library vs. app separation)
- **README** -- 7 CI badges, comprehensive build instructions for all platforms

### [!] Testers Wanted
> We need community testers for platforms we cannot fully validate in CI:
> - **iOS** -- Real device testing (iPhone/iPad with Xcode)
> - **AMD GPU (ROCm/HIP)** -- AMD Radeon RX / Instinct hardware
>
> If you have access to these platforms, please run the build and report results!
> Open an issue at https://github.com/shrec/Secp256K1fast/issues

---

## [2.0.0] - 2026-02-11

### Added
- **Shared POD types** (`include/secp256k1/types.hpp`): Canonical data layouts
  (`FieldElementData`, `ScalarData`, `AffinePointData`, `JacobianPointData`,
  `MidFieldElementData`) with `static_assert` layout guarantees across all backends
- **CUDA edge case tests** (10 new): zero scalar, order scalar, point cancellation,
  infinity operand, add/dbl consistency, commutativity, associativity, field inv
  edges, scalar mul cross-check, distributive -- now 40/40 total
- **OpenCL edge case tests** (8 new): matching coverage -- now 40/40 total
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
- **CUDA point ops optimized**: Local-variable rewrite eliminates pointer aliasing --
  Point Double **2.29x faster** (1.6->0.7 ns), Point Add **1.91x faster** (2.1->1.1 ns),
  kG **2.25x faster** (485->216 ns). CUDA now beats OpenCL on all point ops.
- **PTX inline assembly** for NVIDIA OpenCL: Field ops now at parity with CUDA
- **Benchmarks updated**: Full CUDA + OpenCL numbers on RTX 5060 Ti

### Performance (RTX 5060 Ti, kernel-only)
- CUDA kG: 216.1 ns (4.63 M/s) -- **CUDA 1.37x faster than OpenCL**
- OpenCL kG: 295.1 ns (3.39 M/s)
- Point Double: CUDA 0.7 ns (1,352 M/s), OpenCL 0.9 ns -- **CUDA 1.29x**
- Point Add: CUDA 1.1 ns (916 M/s), OpenCL 1.6 ns -- **CUDA 1.45x**
- Field Mul: 0.2 ns on both (4,139 M/s)

## [1.0.0] - 2026-02-11

### Added
- Complete secp256k1 field arithmetic
- Point addition, doubling, and multiplication
- Scalar arithmetic
- GLV endomorphism optimization
- Assembly optimizations:
  - x86-64 BMI2/ADX (3-5x speedup)
  - RISC-V RV64GC (2-3x speedup)
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
