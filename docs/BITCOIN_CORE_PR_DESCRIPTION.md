# Add optional UltrafastSecp256k1 secondary backend

## What this PR does

Adds `-DUSE_ULTRAFAST_SECP256K1=ON` CMake option (default: **OFF**).
When OFF, Bitcoin Core builds identically to today — no changes to the
existing `src/secp256k1/` path.

When ON, `compat/libsecp256k1_shim` provides the identical `secp256k1.h`
API surface, routing calls through UltrafastSecp256k1's CPU engine.

## Why

UltrafastSecp256k1 is an MIT-licensed C++17 library with a compatible shim
that passes Bitcoin Core's full test suite: **693/693 test_bitcoin, 0 failures**.

Performance vs libsecp256k1 on Intel Core i5-14400F, GCC 14, -O2 -flto, taskset -c 0, nice -19
(bench_bitcoin / nanobench, single run; turbo enabled — see methodology note in JSON):

| Benchmark | libsecp256k1 | UltrafastSecp256k1 | Result |
|---|---|---|---|
| SignSchnorrWithMerkleRoot | 109,367 ns/op | 81,835 ns/op | **+23.9%** |
| VerifyScriptP2TR_ScriptPath | 81,054 ns/op | 64,867 ns/op | **+20.0%** |
| SignTransactionSchnorr | 132,729 ns/op | 126,109 ns/op | **+5.0%** |
| SignTransactionECDSA | 156,044 ns/op | 155,714 ns/op | ≈ tie (+0.2%) |
| VerifyScriptP2TR_KeyPath | 44,575 ns/op | 44,429 ns/op | ≈ tie (+0.3%) |
| ConnectBlockAllEcdsa | 249,930,971 ns/blk | 256,447,311 ns/blk | **−2.6%** ¹ |
| ConnectBlockAllSchnorr | 247,907,421 ns/blk | 254,605,768 ns/blk | **−2.7%** ¹ |

¹ See Known Limitations below.

Full benchmark data and methodology: `docs/BITCOIN_CORE_BENCH_RESULTS.json`

## Security properties

- Constant-time signing paths: LLVM ct-verif + Valgrind taint + dudect (600s)
- 254 exploit PoCs tests, 0 failures
- Wycheproof ECDSA/ECDH: all vectors pass
- RFC 6979 nonce: 35/35 test vectors
- BIP-340 Schnorr: 27/27 test vectors
- DER parity vs libsecp256k1: full BIP-66 edge case matrix (`docs/DER_PARITY_MATRIX.md`)
- noncefp compatibility (R-grinding loop): proven compatible (`BITCOIN_CORE_PR_BLOCKERS.md` §B)

## Evidence

| Claim | Evidence file |
|---|---|
| 693/693 test_bitcoin | `docs/BITCOIN_CORE_TEST_RESULTS.json` |
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

- Default build path is untouched (`-DUSE_ULTRAFAST_SECP256K1` defaults to OFF)
- No changes to `src/secp256k1/` or libsecp256k1 source
- No new mandatory dependencies
- No changes to Bitcoin Core's test suite or CI configuration

## Known Limitations

### ConnectBlock regression (−2.5% to −2.7%)

ConnectBlock aggregate benchmarks show a 2.5–2.7% regression vs libsecp256k1
for all-ECDSA, all-Schnorr, and mixed blocks:

| Benchmark | Regression |
|---|---|
| ConnectBlockAllEcdsa | −2.6% |
| ConnectBlockAllSchnorr | −2.7% |
| ConnectBlockMixedEcdsaSchnorr | −2.5% |
| VerifyScriptP2WPKH | −2.5% |

**Root cause:** Two compounding factors:
1. **Shim dispatch overhead** — each verify call crosses the C shim boundary
   (function pointer dispatch + argument marshalling).
2. **GLV precompute cache cold start** — Bitcoin block validation uses almost
   entirely unique pubkeys. The library's GLV table precompute (~2 µs/miss)
   is triggered on every pubkey, yielding a net cost rather than a benefit.
   The cache pays off only when pubkeys repeat within the 256-slot window, which
   is uncommon in P2WPKH/P2TR block validation.

**Remediation path:** Lazy GLV table build (precompute only on second hit for
the same pubkey) is under investigation. This would eliminate the precompute
cost for unique-pubkey workloads while preserving cache benefit for repeated
pubkeys (e.g., P2PK coinbase scripts).

### Constant-time guarantee scope

CT signing is enforced at the C ABI layer (`ufsecp_*` functions). The public
C++ convenience API (`secp256k1::ecdsa_sign()`, etc.) routes through CT
implementations but does not carry a compile-time namespace guarantee.
Callers requiring explicit CT guarantees should use `ct::` namespace functions
directly. The ABI layer is the production-safe surface.

### Benchmark methodology note

All results were collected with turbo boost enabled (no sudo available for
cpupower). CPU frequency scaling (800–2500 MHz) introduces ±2–5% variance.
The ConnectBlock regression is measured at −2.5% to −2.7%; given the variance
envelope, this is a real effect but the exact magnitude should be re-validated
with turbo disabled and frequency pinned before drawing final conclusions.

### Scope of this PR

GPU backends (CUDA, OpenCL, Metal), FFI bindings, WASM, ZK primitives, and
multi-coin wallet tooling are present in the repository but **out of scope** for
this PR. This PR proposes only the CPU secp256k1 backend as a secondary
compile-time option.
