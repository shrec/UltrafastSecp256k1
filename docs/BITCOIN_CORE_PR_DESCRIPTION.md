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

Performance vs libsecp256k1 on Intel Core i5-14400F, GCC 14.2.0, Release + LTO,
taskset -c 0, governor=performance, bench_bitcoin native harness, 2026-05-11
(full data in `docs/BITCOIN_CORE_BENCH_RESULTS.json`):

| Benchmark | libsecp256k1 | UltrafastSecp256k1 | Result |
|---|---|---|---|
| SignSchnorrWithMerkleRoot | 114,479 ns/op | 84,273 ns/op | **1.36× faster** |
| SignSchnorrWithNullMerkleRoot | 112,694 ns/op | 83,742 ns/op | **1.35× faster** |
| SignTransactionECDSA | 168,907 ns/op | 147,262 ns/op | **1.15× faster** |
| SignTransactionSchnorr | 137,388 ns/op | 123,525 ns/op | **1.11× faster** |
| VerifyScriptP2TR_ScriptPath | 83,481 ns/op | 75,549 ns/op | **1.11× faster** |
| VerifyScriptP2TR_KeyPath | 46,223 ns/op | 44,860 ns/op | **1.03× faster** |
| VerifyScriptP2WPKH | 46,062 ns/op | 45,217 ns/op | parity (+1.9%) |
| ConnectBlockAllEcdsa | 257.6 ms/blk | 252.2 ms/blk | +2.1% (see note ¹) |
| ConnectBlockAllSchnorr | 255.2 ms/blk | 251.5 ms/blk | +1.5% (see note ¹) |
| ConnectBlockMixed | 255.7 ms/blk | 253.1 ms/blk | +1.0% (see note ¹) |

¹ ConnectBlock: no hard turbo lock was available during this run — treat as parity until
replicated on fully-pinned hardware. Without LTO (RelWithDebInfo): pre-PERF-002 run shows
~1% slower; a controlled post-PERF-002 re-benchmark is pending.
Full numbers with err% in `docs/BITCOIN_CORE_BENCH_RESULTS.json`.

Full benchmark data and methodology: `docs/BITCOIN_CORE_BENCH_RESULTS.json`

## Security properties

- Constant-time signing paths: LLVM ct-verif + Valgrind taint + dudect (600s)
- 262 exploit PoCs tests, 0 failures
- Wycheproof ECDSA/ECDH: all vectors pass
- RFC 6979 nonce: 35/35 test vectors
- BIP-340 Schnorr: 27/27 test vectors
- DER parity vs libsecp256k1: full BIP-66 edge case matrix (`docs/DER_PARITY_MATRIX.md`)
- noncefp compatibility (R-grinding loop): proven compatible (`BITCOIN_CORE_PR_BLOCKERS.md` §B)

## Evidence

| Claim | Evidence file |
|---|---|
| 749/749 test_bitcoin | `docs/BITCOIN_CORE_TEST_RESULTS.json` |
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

All numbers from `docs/bench_unified_2026-05-11_gcc14_x86-64.json`
(Intel i5-14400F, turbo disabled, core pinned, 500 warmup, 11 passes, IQR trimming):

| Compiler | CT ECDSA sign | CT Schnorr sign | Notes |
|---|---|---|---|
| **GCC 14.2.0 + LTO** | **+24% vs libsecp (1.24×)** | **+9% vs libsecp (1.09×)** | CT-vs-CT, canonical controlled run |
| Clang 19 (archived, 2026-03-24) | +33% vs libsecp | ~+9% vs libsecp | Archived; not a current controlled run |

Bitcoin Core Linux CI uses GCC; the GCC 14 row is the relevant metric.
Full raw data: `docs/bench_unified_2026-05-11_gcc14_x86-64.json`.

### ConnectBlock Schnorr: −17% regression (native C++ API, unique pubkeys)

**Note: measured with GCC 13.3 on native C++ API path (not shim path, not GCC 14).**
GCC 14.2.0 shim-path numbers are expected to be ≈5% better once PERF-B/PERF-08 applied (unconfirmed projection).

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

**Mitigation path (unconfirmed projection):** PERF-B optimization (smaller per-call table for unique-pubkey
workloads) is tracked in workingdocs. Shim-path numbers with LTO are expected to
be ≈5% better than the native C++ API numbers above once the DualMulGenTables
lazy-init fix (PERF-08) is applied. Treat as projection until controlled measurement confirms.

### Constant-time guarantee scope

CT signing is enforced at the C ABI layer (`ufsecp_*` functions). The public
C++ convenience API (`secp256k1::ecdsa_sign()`, etc.) routes through CT
implementations but does not carry a compile-time namespace guarantee.
Callers requiring explicit CT guarantees should use `ct::` namespace functions
directly. No external third-party audit of the CT implementation has been conducted —
all evidence is self-generated CI tooling (LLVM ct-verif, Valgrind taint, dudect).
The ABI layer is the production-safe surface.

### Benchmark methodology note

All controlled results were collected with: turbo disabled (`echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo`, governor=performance), CPU pinned (`taskset -c 0`), `nice -20`, 500 warmup/op, 11 passes, IQR outlier removal. Compiler: GCC 14.2.0, Release + LTO. Canonical data: `docs/bench_unified_2026-05-11_gcc14_x86-64.json`. Bitcoin Core integration data: `docs/BITCOIN_CORE_BENCH_RESULTS.json`.

### Scope of this PR

GPU backends (CUDA, OpenCL, Metal), FFI bindings, WASM, ZK primitives, and
multi-coin wallet tooling are present in the repository but **out of scope** for
this PR. This PR proposes only the CPU secp256k1 backend as a secondary
compile-time option.
