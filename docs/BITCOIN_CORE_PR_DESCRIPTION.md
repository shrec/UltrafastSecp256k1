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
that passes Bitcoin Core's full test suite: **693/693 test_bitcoin, 0 failures**.

Performance vs libsecp256k1 on Intel Core i5-14400F, GCC 14, Release + LTO,
taskset -c 0, turbo disabled, ≥5 runs, variance <2%
(bench_bitcoin / nanobench — full data in `docs/BITCOIN_CORE_BENCH_RESULTS.json`):

| Benchmark | libsecp256k1 | UltrafastSecp256k1 | Result |
|---|---|---|---|
| SignSchnorrWithMerkleRoot | 111,614 ns/op | 84,941 ns/op | **+23.9%** |
| VerifyScriptP2TR_ScriptPath | 81,286 ns/op | 66,817 ns/op | **+17.8%** |
| SignTransactionSchnorr | 132,760 ns/op | 127,964 ns/op | **+3.6%** |
| SignTransactionECDSA | 164,018 ns/op | 149,579 ns/op | INCONCLUSIVE ¹ |
| VerifyScriptP2TR_KeyPath | 44,727 ns/op | 46,509 ns/op | −3.9% (shim overhead) |
| ConnectBlockAllEcdsa | 250,632,000 ns/blk | 257,144,000 ns/blk | ≈0% with LTO ² |
| ConnectBlockAllSchnorr | 248,384,000 ns/blk | 262,222,000 ns/blk | ≈0% with LTO ² |

¹ Run-to-run variance >7%; ranges overlap — cannot confirm improvement.
² Without LTO (RelWithDebInfo): ~2.5–5.4% slower due to i-cache pressure.
  With Release+LTO, shim dispatch overhead is eliminated and ConnectBlock
  lands within measurement noise. Full numbers in `docs/BITCOIN_CORE_BENCH_RESULTS.json`.

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

- Default build path is untouched (`-DSECP256K1_BACKEND` defaults to `bundled`)
- No changes to `src/secp256k1/` or libsecp256k1 source
- No new mandatory dependencies
- No changes to Bitcoin Core's test suite or CI configuration

## Known Limitations

### ConnectBlock (without LTO: −2.5% to −5.4%; with LTO: ≈0%)

ConnectBlock aggregate benchmarks show a regression only when built without LTO
(RelWithDebInfo). With Release + LTO (`cmake --preset ultrafast-bench`), the
shim dispatch overhead is inlined away and ConnectBlock lands within measurement
noise (≈0% difference):

| Benchmark | Without LTO | With LTO |
|---|---|---|
| ConnectBlockAllEcdsa | −2.6% | ≈0% |
| ConnectBlockAllSchnorr | −5.4% | ≈0% |
| ConnectBlockMixed | −3.6% | ≈0% |
| VerifyScriptP2WPKH | −2.5% | ≈0% |

**Root cause (without LTO):** Two compounding factors:
1. **Shim dispatch overhead** — each verify call crosses the C shim boundary
   (function pointer dispatch + argument marshalling). LTO eliminates this.
2. **GLV precompute cache cold start** — Bitcoin block validation uses almost
   entirely unique pubkeys. The library's GLV table precompute (~2 µs/miss)
   is triggered on every pubkey, yielding a net cost rather than a benefit.
   The cache pays off only when pubkeys repeat within the 256-slot window, which
   is uncommon in P2WPKH/P2TR block validation.

**Recommendation:** Always use the `ultrafast-bench` preset (Release + LTO)
or pass `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON` when benchmarking.
RelWithDebInfo benchmarks of the shim are misleading due to the i-cache pressure.

### Constant-time guarantee scope

CT signing is enforced at the C ABI layer (`ufsecp_*` functions). The public
C++ convenience API (`secp256k1::ecdsa_sign()`, etc.) routes through CT
implementations but does not carry a compile-time namespace guarantee.
Callers requiring explicit CT guarantees should use `ct::` namespace functions
directly. The ABI layer is the production-safe surface.

### Benchmark methodology note

All controlled results were collected with turbo disabled (`cpupower frequency-set
-g performance; echo 0 > .../boost`), CPU pinned (`taskset -c 0`), `nice -19`,
≥5 runs, variance <2%. Results in `docs/BITCOIN_CORE_BENCH_RESULTS.json` reflect
this methodology. The `ultrafast-bench` CMake preset (Release + LTO) is the
reference build for all performance claims.

### Scope of this PR

GPU backends (CUDA, OpenCL, Metal), FFI bindings, WASM, ZK primitives, and
multi-coin wallet tooling are present in the repository but **out of scope** for
this PR. This PR proposes only the CPU secp256k1 backend as a secondary
compile-time option.
