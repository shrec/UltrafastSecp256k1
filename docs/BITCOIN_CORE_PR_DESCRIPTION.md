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

Performance improvement over libsecp256k1 on the same hardware:

| Operation | libsecp256k1 | UltrafastSecp256k1 | Improvement |
|---|---|---|---|
| SignTransactionECDSA | 96,006 ns/op | 79,196 ns/op | **+17.5%** |
| SignTransactionSchnorr | 80,368 ns/op | 73,663 ns/op | **+8.3%** |
| VerifyScriptBench | 24,555 ns/op | 23,231 ns/op | **+5.4%** |

Full benchmark data: `docs/BITCOIN_CORE_BENCH_RESULTS.json`

## Security properties

- Constant-time signing paths: LLVM ct-verif + Valgrind taint + dudect (600s)
- 232 exploit PoC tests, 0 failures
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
