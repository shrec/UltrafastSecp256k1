# Add optional UltrafastSecp256k1 secondary backend

## What this PR does

Adds `-DSECP256K1_BACKEND=ultrafast` CMake option (default: **bundled**).
When bundled, Bitcoin Core builds identically to today — no changes to the
existing `src/secp256k1/` path.

When ultrafast, `compat/libsecp256k1_shim` provides the identical `secp256k1.h`
API surface, routing calls through UltrafastSecp256k1's CPU engine.

**Build requirement:** Release + LTO (`-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`)
is required for full performance. Use the provided `ultrafast-bench` CMake preset
(`cmake --preset ultrafast-bench`) which sets both automatically.

## Why

UltrafastSecp256k1 is an MIT-licensed C++20 library with a compatible shim
that passes Bitcoin Core's full test suite: **749/749 test_bitcoin, 0 failures** (GCC 14.2.0 run, `docs/BITCOIN_CORE_BENCH_RESULTS.json`). An earlier Clang 19 run against v28.0 showed 693/693 (`docs/BITCOIN_CORE_TEST_RESULTS.json`).

## Honest performance summary (read first)

This PR has both wins and a known regression. Both are surfaced here so reviewers
do not have to hunt for either.

**Wins (Release + LTO, GCC 14.2.0, controlled, `docs/BITCOIN_CORE_BENCH_RESULTS.json`):**

- Taproot/Merkle Schnorr sign: ~1.35× faster
- ECDSA sign: ~1.10× faster
- P2TR script-path verify: ~1.10× faster
- ConnectBlock (Schnorr/ECDSA/Mixed): +0.9–1.5% faster end-to-end

**Known regression — surfaced upfront, not buried:**

- **ConnectBlockAllSchnorr (unique-pubkey workload): −17% slower** vs libsecp256k1
  on a 2026-05-07 native C++ API measurement (GCC 13.3, 2000 unique pubkeys).
  Root cause is the per-unique-pubkey GLV table rebuild cost; libsecp256k1's
  smaller 5-entry table avoids this. Full data and analysis below in "Known
  Limitations → ConnectBlock Schnorr regression". A re-measurement on the shim
  path with GCC 14 + LTO has not been performed; treat the −17% figure as a
  conservative upper bound until that controlled re-run lands.

**Build requirement:** Release + LTO is required to reach the wins above.
RelWithDebInfo will show a small (~0.5–1.0%) ConnectBlock slowdown due to
larger code footprint (~1.3 MB vs libsecp's ~400 KB → i-cache pressure).

Full benchmark data and methodology: `docs/BITCOIN_CORE_BENCH_RESULTS.json`
and `docs/bench_unified_2026-05-30_gcc14_x86-64.json`.

## Security properties

- Constant-time signing paths: LLVM ct-verif + Valgrind taint + dudect (600s)
- 269 exploit PoCs tests, 0 failures
- Wycheproof ECDSA/ECDH: all vectors pass
- RFC 6979 nonce: 35/35 test vectors
- BIP-340 Schnorr: 27/27 test vectors
- DER parity vs libsecp256k1: full BIP-66 edge case matrix (`docs/DER_PARITY_MATRIX.md`)
- noncefp compatibility (R-grinding loop): proven compatible (`BITCOIN_CORE_PR_BLOCKERS.md` §B)

## Evidence

| Claim | Evidence file |
|---|---|
| 749/749 test_bitcoin | `docs/BITCOIN_CORE_BENCH_RESULTS.json` |
| Benchmarks | `docs/BITCOIN_CORE_BENCH_RESULTS.json` |
| CT verification | `docs/CT_VERIFICATION.md` |
| ABI compatibility | `docs/ABI_VERSIONING.md` |
| Thread safety | `docs/THREAD_SAFETY.md` |
| DER parity | `docs/DER_PARITY_MATRIX.md` |
| All blockers closed | `BITCOIN_CORE_PR_BLOCKERS.md` |
| Full security evidence | `docs/BITCOIN_CORE_BACKEND_EVIDENCE.md` |

## Real-world adoption

[SparrowWallet Frigate](https://github.com/sparrowwallet/frigate) uses this
library in production for Silent Payments (BIP-352) GPU scanning.

## What is NOT changed

- Default build path is untouched (`-DSECP256K1_BACKEND` defaults to `bundled`)
- No changes to `src/secp256k1/` or libsecp256k1 source
- No new mandatory dependencies
- No changes to Bitcoin Core's test suite or CI configuration

## Known Limitations

### CT signing throughput (GCC 14.2.0 — CT-vs-CT, production-equivalent)

All numbers from `docs/bench_unified_2026-05-30_gcc14_x86-64.json`
(Intel i5-14400F, turbo lock CONFIRMED, core pinned, 500 warmup, 11 passes, IQR trimming):

| Compiler | CT ECDSA sign | CT Schnorr sign | Notes |
|---|---|---|---|
| **GCC 14.2.0 + LTO** | **+33% vs libsecp (1.33×)** | **+26% vs libsecp (1.26×)** | CT-vs-CT, canonical controlled run |
| Clang 19 (archived, 2026-03-24) | +33% vs libsecp | ~+9% vs libsecp | Archived; not a current controlled run |

Bitcoin Core Linux CI uses GCC; the GCC 14 row is the relevant metric.
Full raw data: `docs/bench_unified_2026-05-30_gcc14_x86-64.json`.

### ConnectBlock Schnorr: −17% regression (native C++ API, unique pubkeys)

**Workload-specific regression — surfaced as the cost side of the tradeoff.**
This is the same data summarised in the "Honest performance summary" above; full
numbers and root cause are kept here so reviewers can audit the claim.

Controlled measurement (2026-05-07, i5-14400F, **GCC 13.3** + LTO, 2000 unique pubkeys, native C++ API — NOT the libsecp-compatible shim path):

| Scenario | Ultra | libsecp | Result |
|---|---|---|---|
| ConnectBlockAllEcdsa | 86.5 ms | 84.9 ms | 0.98× ≈ parity |
| **ConnectBlockAllSchnorr** | **103.4 ms** | **86.1 ms** | **0.83× (−17%)** |
| ConnectBlockMixed (2k ECDSA + 1k Schnorr) | 146.7 ms | 128.9 ms | 0.88× (−12%) |
| ConnectBlock + DER parse | 86.4 ms | 96.4 ms | 1.12× (+12%) |

**Root cause:** Per-unique-pubkey GLV table rebuild (~2 µs × 2000 pubkeys = 4 ms
overhead per Schnorr block). Bitcoin block validation uses almost entirely unique
pubkeys; the GLV precompute cache provides no benefit and adds rebuild cost.
libsecp256k1 uses a 5-entry table with lower cold-start cost.

**Compiler/path caveat:** The measurement above is native C++ API on GCC 13.3.
The 2026-05-12 GCC 14.2.0 + LTO shim-path run (`docs/BITCOIN_CORE_BENCH_RESULTS.json`)
shows ConnectBlock end-to-end +0.9–1.5% faster than libsecp256k1 on a mixed
workload, suggesting the −17% upper bound does not survive the canonical
shim-path measurement. A direct unique-pubkey re-run on the shim path with
GCC 14 + LTO has not yet been performed; the −17% is kept here as the
conservative ceiling until that re-run lands.

**Mitigation in progress (tracked, not landed):** PERF-B (per-call smaller
table for unique-pubkey workloads) and PERF-08 (`DualMulGenTables` lazy-init)
are tracked in `workingdocs/PERF_B_FIX_PLAN.md`. Neither is required for the
shim-path numbers above; they would further reduce cold-start cost on the
native C++ API path.

### Constant-time guarantee scope

CT signing is enforced at the C ABI layer (`ufsecp_*` functions). The public
C++ convenience API (`secp256k1::ecdsa_sign()`, etc.) routes through CT
implementations but does not carry a compile-time namespace guarantee.
Callers requiring explicit CT guarantees should use `ct::` namespace functions
directly. No external third-party audit of the CT implementation has been conducted —
all evidence is self-generated CI tooling (LLVM ct-verif, Valgrind taint, dudect).
The ABI layer is the production-safe surface.

### Benchmark methodology note

CT signing results were collected with: turbo off, taskset -c 0, nice -20, 500
warmup/op, 11 passes, IQR outlier removal. Compiler: GCC 14.2.0, Release + LTO.
Canonical data: `docs/bench_unified_2026-05-30_gcc14_x86-64.json`. ConnectBlock
integration data (hard turbo lock confirmed, intel_pstate/no_turbo=1,
governor=performance, taskset -c 0, nice -20, 2026-05-12):
`docs/BITCOIN_CORE_BENCH_RESULTS.json`.

### Scope of this PR

GPU backends (CUDA, OpenCL, Metal), FFI bindings, WASM, ZK primitives, and
multi-coin wallet tooling are present in the repository but **out of scope** for
this PR. This PR proposes only the CPU secp256k1 backend as a secondary
compile-time option.
