# Bitcoin Core Alternative Backend — Evidence Document

> Version: 1.3 — 2026-05-11
> Profile: `bitcoin-core-backend`
> Status: PR-prep / RC-prep — evidence refresh in progress; not yet submitted (see BITCOIN_CORE_PR_BLOCKERS.md)

This document presents structured evidence for Bitcoin Core reviewers evaluating
UltrafastSecp256k1 as an alternative CPU backend for libsecp256k1. It is written
to be honest and verifiable. Each claim cites the evidence file, CI gate, and
verification command a reviewer can use to independently check it.

> **Scope note:** This is an opt-in compile-time alternative backend, not a replacement
> for libsecp256k1. The default build (`SECP256K1_BACKEND=bundled`) is byte-for-byte
> identical to today. GPU operations, language bindings (Rust/Python/Node/Go), BIP-352
> Silent Payments pipeline, FROST/MuSig2 advanced protocols, and WASM targets are
> **out of scope** for this Bitcoin Core evaluation. Only the CPU secp256k1 path and
> its libsecp256k1-compatible shim are relevant for the PR.

---

## 1. Scope Definition

### In scope for the `bitcoin-core-backend` profile

The following components and properties are subject to audit coverage under this profile:

- **CPU signing and verification only.** No GPU signing path is in scope. GPU operations
  process public data only; no secret material passes through GPU code paths.

- **`compat/libsecp256k1_shim/`** — all shim functions that translate the libsecp256k1
  C API to UltrafastSecp256k1 internals:
  - `shim_context.cpp` — context creation and self-test
  - `shim_ecdsa.cpp` — ECDSA sign, verify, parse, serialize
  - `shim_schnorr.cpp` — BIP-340 Schnorr sign, verify, batch verify
  - `shim_extrakeys.cpp` — `secp256k1_keypair_*`, `secp256k1_xonly_pubkey_*`
  - `shim_pubkey.cpp` — pubkey parse, serialize, tweak, combine
  - `shim_seckey.cpp` — seckey verify, negate, tweak
  - `shim_recovery.cpp` — ECDSA sign-recoverable, recovery
  - `shim_ecdh.cpp` — ECDH via `secp256k1_ecdh`
  - `shim_tagged_hash.cpp` — `secp256k1_tagged_sha256`

- **Public header API surface:**
  - `secp256k1.h`
  - `secp256k1_schnorrsig.h`
  - `secp256k1_extrakeys.h`
  - `secp256k1_recovery.h`
  - `secp256k1_ecdh.h`

- **Parser parity** — accept/reject behavior must match libsecp256k1 exactly for all
  inputs, including edge cases at group order and field prime boundaries.

- **ECDSA, Schnorr/Taproot, libsecp256k1 compatibility** — functional equivalence
  verified by differential testing against the reference implementation.

- **C++20 to C++17 build isolation** — the shim layer and any headers exposed to consumers
  must not require C++20 language features.

- **Thread safety of signing paths** — all signing operations must be safe for concurrent
  use from multiple threads.

- **RFC 6979 nonce generation** — deterministic nonce derivation compatible with the
  libsecp256k1 behavior, including `ndata` (extra entropy) and R-grinding (retry on
  high-s or r==0).

### Out of scope for this profile

The following are intentionally excluded from the Bitcoin Core backend evaluation.
Exclusion is not a deferral of bugs; it is a boundary definition.

| Item | Reason for exclusion |
|------|---------------------|
| GPU backend | Public-data-only operations; no secret values. Separate assurance track. |
| Multicoin address generation beyond secp256k1 | Not used by Bitcoin Core |
| ZK / FROST / MuSig2 | Not on the Bitcoin Core critical path unless Core adds them |
| BIP-352 (Silent Payments) scanning performance | Separate feature track |
| Language bindings (Rust, Python, etc.) | Not relevant to Core's C API usage |
| ROCm/HIP hardware validation | Deferred; see RR-003 in `docs/RESIDUAL_RISK_REGISTER.md` |

---

## 2. Security Claims for This Scope

The table below lists each security claim, the evidence artifact that supports it, the
CI gate that enforces it, and when it was last verified. "Every commit" means the gate
runs in CI on every push to `dev`.

A claim without a CI gate is explicitly marked; manual-verification-only claims are a
known gap (see Section 3).

| Claim | Evidence | CI Gate | Last Verified |
|-------|----------|---------|---------------|
| Constant-time ECDSA signing on x86-64: private key and nonce operations route through `secp256k1::ct::*` primitives with no variable-time branches on secret data | `docs/CT_VERIFICATION.md` | `ct-verif.yml` | Every commit |
| Constant-time Schnorr signing on x86-64: BIP-340 signing uses `secp256k1::ct::schnorr_sign`; aux_rand masking and nonce derivation are CT throughout | `docs/CT_VERIFICATION.md` | `ct-verif.yml` | Every commit |
| `secp256k1_ecdsa_signature_parse_compact` rejects `r == 0`, `s == 0`, `r >= n`, `s >= n` | `ci/check_libsecp_shim_parity.py` | `preflight.yml` | Every commit |
| `secp256k1_ec_seckey_verify` rejects `key == 0` and `key >= n` | `ci/check_libsecp_shim_parity.py` | `preflight.yml` | Every commit |
| `secp256k1_xonly_pubkey_parse` rejects `x >= p` and x-coordinates with no valid y | `ci/check_libsecp_shim_parity.py` | `preflight.yml` | Every commit |
| `secp256k1_pubkey_parse` rejects the point at infinity and malformed encodings | `ci/check_libsecp_shim_parity.py` | `preflight.yml` | Every commit |
| RFC 6979 deterministic nonce derivation matches the libsecp256k1 reference output on the BIP-340 test vectors and the RFC 6979 ECDSA test vectors | `ci/rfc6979_spec_verifier.py` | `security-audit.yml` | Every commit |
| `ndata` / R-grinding compatibility: `secp256k1_ecdsa_sign` with `ndata != NULL` (extra entropy) produces the same output as libsecp256k1 for the same `ndata` input | `compat/libsecp256k1_shim/src/shim_ecdsa.cpp` (`ecdsa_sign_hedged`) | Manual verified; CI gate planned | 2026-04-27 |
| The shim layer does not leak C++20 language requirements to consumers: `cmake --print-target-features` shows no C++20 ABI in exported interface | `cmake --print-target-features` output | Build CI (`build.yml`) | Every commit |
| Context creation calls `ensure_library_integrity()` self-test: `ufsecp_ctx_create` verifies field arithmetic, scalar arithmetic, and generator point before returning | `compat/libsecp256k1_shim/src/shim_context.cpp` | Audit tests (`unified_audit_runner`) | Every commit |
| Zero-signature detection: CT signing primitives returning `r == 0`, `s == 0`, or all-zero Schnorr signature cause the ABI wrapper to return `UFSECP_ERR_INTERNAL`, not serialize the zero result | `audit/test_exploit_boundary_sentinels.cpp` | `unified_audit_runner` via CTest | Every commit |
| Batch signing is fail-closed: output buffers are cleared before processing; partial failures do not leave valid-looking partial signatures | `audit/` batch signing exploit PoCs | `unified_audit_runner` | Every commit |
| Schnorr low-s / half-order boundary handled correctly: signatures with `s = (n-1)/2` and `s = (n+1)/2` are accepted or rejected per BIP-340 | `audit/test_exploit_boundary_sentinels.cpp` | `unified_audit_runner` | Every commit |

### CT evidence methodology

CT claims are verified using a multi-layer approach. No single tool is treated as
definitive; all three must pass:

1. **LLVM constant-time verification** (`ci/collect_ct_evidence.py`) — disassembly
   analysis for conditional branches on secret-derived values.
2. **Valgrind memcheck with ct-grind** (`ci/ctgrind_validate.sh`) — marks secret
   inputs as uninitialised at the Valgrind level; any data-dependent branch triggers a
   Valgrind error.
3. **dudect statistical timing** — runs the signing path under varying secret inputs and
   tests for statistically significant timing differences.

Results are recorded in `docs/CT_VERIFICATION.md`. The CI gate in `ct-verif.yml` runs
all three on every push.

**Confidence ceiling:** CT verification tools can miss some attack classes. Hardware
microarchitectural channels (cache sets, branch predictor state, power) are not covered
by software tooling alone. This limitation is documented in RR-001
(`docs/RESIDUAL_RISK_REGISTER.md`) and in `docs/HARDWARE_SIDE_CHANNEL_METHODOLOGY.md`.

---

## 2.1 ConnectBlock Performance — LTO Required for Full Win

> **With LTO (recommended production build): Ultra wins all ConnectBlock scenarios (+0.9–1.5%).
> Without LTO: Ultra is ~0.5–1.0% slower than libsecp256k1. Use Release+LTO for Bitcoin Core builds.**

- **With LTO:** +0.9% to +1.5% faster on all ConnectBlock workloads (confirmed, err% 0.2–0.5%).
  Canonical data: `docs/BITCOIN_CORE_BENCH_RESULTS.json` (commit `48e7c02f`, 2026-05-12,
  hard turbo lock: intel_pstate/no_turbo=1, governor=performance, taskset -c 0, nice -20).
- **Without LTO:** ~0.5–1.0% slower than libsecp256k1 (AllEcdsa −0.5%, AllSchnorr −1.0%,
  Mixed −0.8%; measured 2026-05-12, hard turbo lock, 3 runs, non-overlapping). PERF-002 reduced
  the no-LTO deficit from ~1.1% to ~0.5–1.0% by removing a redundant y²=x³+7 on-curve check
  (~400 ns/call), but did not eliminate it. The remaining gap is caused by Ultra's larger code
  footprint (~1.3 MB secp256k1 symbols vs libsecp ~400 KB) creating i-cache pressure without LTO.
  LTO resolves this by co-optimizing code layout globally.

Full no-LTO data: `docs/BITCOIN_CORE_BENCH_RESULTS.json` (`results_nolto` section,
post-PERF-002 measured run, 2026-05-12).

### Results: Release + LTO (recommended build)

Benchmark on Intel i5-14400F, GCC 14.2.0, `cpupower` hard turbo lock, `taskset -c 0`, `nice -20`, 5 runs, 2026-05-12.
bench_bitcoin native harness (nanobench, `-min-time=3000ms`), medians from 5 runs, err% from nanobench per-run:

| Benchmark | Ultra LTO | libsecp LTO | vs libsecp | err% |
|-----------|----------:|----------:|-----------|------|
| ConnectBlockAllEcdsa | 254.3 ms/blk | 257.4 ms/blk | **Ultra +1.2%** | 0.4% |
| ConnectBlockAllSchnorr | 253.0 ms/blk | 255.3 ms/blk | **Ultra +0.9%** | 0.5% |
| ConnectBlockMixed | 253.9 ms/blk | 257.7 ms/blk | **Ultra +1.5%** | 0.5% |
| SignSchnorrWithMerkleRoot | 83,930 ns/op | 113,410 ns/op | **1.35× faster** | 0.2% |
| SignSchnorrWithNullMerkleRoot | 83,980 ns/op | 112,960 ns/op | **1.35× faster** | 0.4% |
| SignTransactionECDSA | 149,520 ns/op | 165,140 ns/op | **1.10× faster** | 0.4% |
| SignTransactionSchnorr | 125,380 ns/op | 137,500 ns/op | **1.10× faster** | 0.5% |
| VerifyScriptP2TR_ScriptPath | 76,451 ns/script | 83,812 ns/script | **1.10× faster** | 0.2% |
| VerifyScriptP2TR_KeyPath | 45,406 ns/script | 46,333 ns/script | **Ultra +2.0%** | 0.2% |
| VerifyScriptP2WPKH | 45,978 ns/script | 45,777 ns/script | parity (0.4% slower) | 0.3% |

Full data: `docs/BITCOIN_CORE_BENCH_RESULTS.json` (commit `48e7c02f`, 2026-05-12, hard turbo lock confirmed).

### CT Signing — Compiler Results (Material Disclosure)

CT signing performance on GCC 14.2.0 (Linux default for Bitcoin Core CI),
from `docs/bench_unified_2026-05-11_gcc14_x86-64.json`
(Intel i5-14400F, turbo disabled via `intel_pstate/no_turbo=1`, governor=performance,
core pinned, 500 warmup, 11 passes, IQR trimming):

| Compiler | CT ECDSA sign | CT Schnorr sign | Canonical artifact |
|----------|:---:|:---:|---|
| **GCC 14.2.0** (Linux default) | **1.24× faster** (+24%) | **1.09× faster** (+9%) | `docs/bench_unified_2026-05-11_gcc14_x86-64.json` |
| Clang 19 (archived, 2026-03-24) | 1.33× faster (+33%) | ~1.09× faster | `docs/BENCHMARKS.md §archived` — not a current controlled run |

> **Two benchmark sets, two different measurements:**
> - `bench_unified` CT-vs-CT rows (above): isolate the raw CT signing primitive (RFC6979 + CT generator mul + CT scalar inverse). GCC 14: 1.24×/1.09× faster.
> - `bench_bitcoin SignTransaction*` rows: cover the full Bitcoin Core transaction-signing path including context-blinding cache and pre-computed generator tables. GCC 13/14: 1.09–1.24× faster (see §Results table above).
>
> Both are correct — they measure different scopes. The full-path `SignTransaction*` numbers are the Bitcoin Core-relevant ones; the CT primitive numbers confirm no scalar-inverse regression on GCC 14.

### Without-LTO gap — partially reduced by PERF-002

Without LTO, Ultra is ~0.5–1.0% slower than libsecp256k1 on ConnectBlock (measured 2026-05-12,
hard turbo lock): AllEcdsa −0.5%, AllSchnorr −1.0%, Mixed −0.8%. The gap stems from Ultra's
larger code footprint (~1.3 MB secp256k1 symbols vs libsecp ~400 KB) creating i-cache pressure
without LTO. PERF-002 (removed redundant y²=x³+7 on-curve check, ~400 ns/call) reduced the gap
from ~1.1% to ~0.5–1.0%, but did not fully eliminate it. **Use Release+LTO for production Bitcoin
Core builds** — LTO eliminates the i-cache pressure by co-optimizing code layout globally.

### Build command for full performance

```bash
cmake -B out/build-ultrafast-lto \
  -DSECP256K1_BACKEND=ultrafast \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
  -DBUILD_BENCH=ON -DENABLE_IPC=OFF
```

Full methodology and raw numbers: `docs/BITCOIN_CORE_BENCH_RESULTS.json`

---

## 3. Known Gaps and Open Items

These are honest statements of residual confidence gaps. Each entry is labelled
**PR blocker** (whether it must be resolved before a Core PR) or **residual gap**
(an honest limitation that does not block the PR but a reviewer should know about).

| Item | PR blocker? | Residual gap / notes |
|------|-------------|----------------------|
| ~~Bitcoin Core's full test suite has not been run~~ | **Closed (2026-04-27)** | 693/693 `test_bitcoin` pass (2026-04-27, `docs/BITCOIN_CORE_TEST_RESULTS.json`); 749/749 in more recent run (2026-05-11, `docs/BITCOIN_CORE_BENCH_RESULTS.json`) |
| Windows and macOS CI coverage | **Not a blocker** — macOS ARM64 shim build+test active (`macos-shim.yml`); Windows x86-64 build + bench covered in CI | Residual gap: GPU suite and extended CT tools (Valgrind, dudect nightly) run Linux-only; acceptable for a CPU-only backend PR |
| `ndata` / R-grinding parity | **Not a blocker** — Bitcoin Core 28.x production signing paths use `NULL` noncefp (verified by grep of Core source); shim accepts NULL / default / rfc6979 and fails-closed on any unknown noncefp | Residual gap: arbitrary custom nonce callback emulation is intentionally unsupported; not needed for Core's usage pattern |
| Thread safety: multiple contexts sharing blinding state | **Not a blocker** — each context has its own blinding state; concurrent use of distinct contexts is safe and documented in `docs/THREAD_SAFETY.md` | Residual gap: interleaved calls across contexts on the same thread are not CI-tested; THREAD_SAFETY.md §7 marks this as a known limitation; not exercised by Core |
| Formal verification of CT layer | **Not a blocker** — no formal proof is claimed; software-tool CT verification (valgrind, ct-verif, dudect, ct-prover, ARM64 native) is the stated evidence level | Residual gap: hardware microarchitectural side channels are not ruled out by software tools alone; acceptable known confidence gap |
| libsecp256k1 parity for `secp256k1_ellswift_*` | **Not a blocker** — ECDSA and Schnorr differential coverage is primary; ElligatorSwift has parity tests in `shim_ellswift.cpp` | Residual gap: differential coverage against the libsecp256k1 reference is thinner than for ECDSA/Schnorr; expanding is in the audit backlog |

### 2026-05-01 Audit Cycle Note

A red-team security audit completed on 2026-05-01 identified and fixed findings across GPU
backends, language bindings, and compatibility shims. The fixes are fully documented in
`docs/AUDIT_CHANGELOG.md` under "2026-05-01 — Security Audit Cycle: Red Team Findings Fixed".

**Relevance to this profile (CPU/shim scope):**

- `shim_musig.cpp` (CRIT-1) and `shim_schnorr.cpp` (HIGH-1, MED-2) are in scope for this profile.
  Those fixes (nonce-reuse protection, CT Schnorr path, branchless nonce parity) are included in
  the evidence recorded in `docs/CT_VERIFICATION.md` and covered by `ct-verif.yml`.
- Strict private-key parsing (`parse_bytes_strict_nonzero`) is now enforced across all shim sign
  functions — the shim parity check in `ci/check_libsecp_shim_parity.py` verifies rejection
  of `key == 0` and `key >= n` on every commit.
- Schnorr R x-coordinate zero check (MEDIUM-1) is covered by the updated
  `audit/test_exploit_boundary_sentinels.cpp` exploit test.
- GPU backend fixes (CRITICAL-1/2, HIGH-1 through HIGH-4) are **out of scope** for this profile
  but are recorded in `docs/AUDIT_CHANGELOG.md` for completeness.

---

## 4. How to Run the Profile

The following commands reproduce the evidence for this profile. Run them on the commit
hash recorded in `docs/REPLAY_CAPSULE.json` with the build flags documented there.

### Full CAAS pipeline

```bash
# Build
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build

# Run the bitcoin-core-backend profile
python3 ci/caas_runner.py --profile bitcoin-core-backend --json -o btc_evidence.json

# Verify the evidence bundle
python3 ci/verify_external_audit_bundle.py --json
```

### Individual gate checks

```bash
# Parser parity checks (accept/reject behavior vs libsecp256k1)
python3 ci/check_libsecp_shim_parity.py

# Build isolation check (no C++20 leaked to consumers)
python3 ci/check_core_build_mode.py

# RFC 6979 nonce derivation vs reference vectors
python3 ci/rfc6979_spec_verifier.py

# CT evidence collection (requires clang-19, valgrind, dudect)
python3 ci/collect_ct_evidence.py --cpu-only
```

### Audit test suite (CPU paths only)

```bash
# Run all audit tests (includes exploit PoCs)
ctest --test-dir build -L audit --output-on-failure --timeout 300

# Run only the shim-layer tests
ctest --test-dir build -R shim --output-on-failure
```

---

## 5. Evidence Artifacts

All evidence files relevant to this profile are listed below. Each file is
integrity-pinned in `docs/EXTERNAL_AUDIT_BUNDLE.json` with a SHA-256 hash.

| Artifact | Description |
|----------|-------------|
| `docs/EXTERNAL_AUDIT_BUNDLE.json` | Hash-pinned evidence record. The SHA-256 digest is at `docs/EXTERNAL_AUDIT_BUNDLE.sha256`. The verifier script is `ci/verify_external_audit_bundle.py`. |
| `docs/CT_VERIFICATION.md` | CT pipeline results: LLVM disassembly analysis, ctgrind output, dudect timing results for ECDSA and Schnorr signing. Updated on every CT verification run. |
| `docs/BACKEND_PARITY.md` | Feature parity matrix across CPU and GPU backends. The CPU column is the relevant one for this profile. |
| `docs/SPEC_TRACEABILITY_MATRIX.md` | Maps each BIP-340, RFC 6979, and SEC 1/2 specification clause to the implementing source file and the audit module that verifies it. Gate: `ci/exploit_traceability_join.py --strict`. |
| `docs/THREAD_SAFETY.md` | Threading model, thread-safety classification for every public API function, and known limitations. |
| `docs/ABI_VERSIONING.md` | ABI stability policy, version bump rules, and migration guidance. Relevant to Core's binary compatibility requirements. |
| `docs/RESIDUAL_RISK_REGISTER.md` | Exhaustive list of open residual risks with justification, status, and owner. Entries RR-001, RR-002, and RR-003 are relevant to this profile. |
| `docs/SECURITY_CLAIMS.md` | Full security claims document including the FAST/CT dual-layer architecture, fail-closed assurance perimeter, and non-claim statements. |
| `docs/AUDIT_SCOPE.md` | Independent review scope including in-scope component list and line counts. |
| `docs/BACKEND_ASSURANCE_MATRIX.md` | Per-backend assurance level ratings. The CPU fast path and CPU CT path both carry HIGH assurance. |
| `docs/SELF_AUDIT_FAILURE_MATRIX.md` | Self-audit failure-class matrix documenting every failure class, its current status, and the CI gate that enforces it. |
| `docs/HARDWARE_SIDE_CHANNEL_METHODOLOGY.md` | Scope statement for hardware side-channel claims (power/EM/fault). The library makes software-side-channel claims only. |
| `docs/REPLAY_CAPSULE_SPEC.md` | Specification for the replay capsule format, which records all inputs needed to reproduce a CAAS pipeline run without maintainer assistance. |
| `docs/REPLAY_CAPSULE.json` | The most recent replay capsule, updated on each release-candidate pipeline run. |
| `compat/libsecp256k1_shim/README.md` | Integration guide for the libsecp256k1 shim layer. |

---

## 6. Contact and Review Process

Questions about specific evidence entries or CI gate behavior can be directed to
`payysoon@gmail.com`.

This document is a structured CAAS evidence presentation intended to reduce the
time an independent reviewer spends
locating and cross-referencing evidence. The reviewer's own judgment supersedes
every claim made here.
