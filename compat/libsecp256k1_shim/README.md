# libsecp256k1 Compatibility Shim

A drop-in replacement for [bitcoin-core/secp256k1](https://github.com/bitcoin-core/secp256k1) that routes calls through **UltrafastSecp256k1**'s optimized engine.

Zero source-code changes needed — swap the library at link time.

---

## Quick Start

```cmake
# CMakeLists.txt
add_subdirectory(path/to/UltrafastSecp256k1/compat/libsecp256k1_shim)
target_link_libraries(my_app PRIVATE secp256k1_shim)
```

```c
// Your code — unchanged
#include <secp256k1.h>
#include <secp256k1_schnorrsig.h>

secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN);
secp256k1_context_randomize(ctx, seed32);  // enables blinding
secp256k1_ecdsa_sign(ctx, &sig, msghash, seckey, NULL, NULL);
secp256k1_context_destroy(ctx);
```

---

## Performance

Measured on Intel Core i5-14400F vs bitcoin-core/secp256k1 (same harness):

| Operation | libsecp256k1 | UF shim | Δ |
|---|---|---|---|
| k·G (generator mul) | 9,883 ns | 4,969 ns | **+99%** |
| ECDSA Verify | 22,860 ns | 20,502 ns | **+12%** |
| ECDSA Sign (CT) | 15,917 ns | 17,514 ns | −9% |
| Pubkey create | 11,409 ns | 4,969 ns | **+130%** |

Bitcoin Core `bench_bitcoin` results with shim (same binary, engine swapped):

| Benchmark | libsecp256k1 | UF shim | Δ |
|---|---|---|---|
| `SignTransactionECDSA` | 96,006 ns | 79,196 ns | **+17.5%** |
| `SignTransactionSchnorr` | 80,368 ns | 73,663 ns | **+8.3%** |
| `SignSchnorrWithMerkleRoot` | 66,520 ns | 39,131 ns | **+41%** |
| `VerifyScriptBench` | 24,555 ns | 23,231 ns | **+5.4%** |

vs OpenSSL 3.x (secp256k1): **11–44× faster** depending on operation.  
vs Knuth secp256k1: **3.6–3.8× faster**.

Full results: [`docs/BENCHMARK_CROSS_LIBRARY.md`](../../docs/BENCHMARK_CROSS_LIBRARY.md)

---

## API Coverage

| Module | Functions | Notes |
|---|---|---|
| **Context** | `create`, `clone`, `destroy`, `randomize`, `set_*_callback`, `selftest` | randomize installs thread-local blinding |
| **Public keys** | `pubkey_create/parse/serialize/negate/tweak_add/tweak_mul/combine/cmp` | full |
| **ECDSA** | `ecdsa_sign`, `ecdsa_verify`, `signature_parse/serialize_compact/der`, `signature_normalize` | BIP-66 strict DER |
| **Schnorr (BIP-340)** | `schnorrsig_sign32`, `schnorrsig_sign_custom`, `schnorrsig_verify` | full |
| **Extra keys** | `xonly_pubkey_*`, `keypair_*` | full |
| **Secret keys** | `seckey_verify/negate/tweak_add/tweak_mul` | strict validation |
| **Tagged hash** | `tagged_sha256` | full |
| **ECDH** | `secp256k1_ecdh` (default SHA-256 + custom hashfp) | CT scalar mul |
| **ElligatorSwift (BIP-324)** | `ellswift_encode/decode/create/xdh` | BIP-324 compatible |
| **MuSig2 (BIP-327)** | `musig_pubkey_agg`, `musig_nonce_gen/agg/process`, `musig_partial_sign/verify/sig_agg` | full |

### Headers provided

```
include/
  secp256k1.h
  secp256k1_schnorrsig.h
  secp256k1_extrakeys.h
  secp256k1_ecdh.h
  secp256k1_ellswift.h
  secp256k1_musig.h
  secp256k1_recovery.h    (ECDSA recoverable signatures)
```

---

## Known Divergences from libsecp256k1

### noncefp in secp256k1_ecdsa_sign

Custom `noncefp` callbacks are **not forwarded**. The shim always uses RFC 6979 internally.

- `NULL`, `secp256k1_nonce_function_rfc6979`, `secp256k1_nonce_function_default` → accepted (RFC 6979 used)
- Any other non-NULL pointer → returns 0 (fail-closed, not silently ignored)
- `ndata` IS respected — passed as auxiliary entropy to hedged signing (used by Bitcoin Core's R-grinding loop)

This is compatible with all known Bitcoin Core usage. See `BITCOIN_CORE_PR_BLOCKERS.md §B`.

### Context blinding (secp256k1_context_randomize)

libsecp256k1: blinding is per-context.  
This shim: blinding is per-thread (thread-local state).

**Practical impact:** Bitcoin Core calls `context_randomize` once per context, once per thread, and does not share contexts across threads. This pattern is fully compatible.

If a single thread randomizes two independent contexts, the second call overwrites the first context's blinding state. This is acceptable in Bitcoin Core's single-context-per-thread model.

### MuSig2 global state

`secp256k1_musig_keyagg_cache` uses a process-global map internally (the libsecp2k1 opaque struct is 197 bytes — too small for `std::vector<Scalar> key_coefficients`). Sessions auto-clean up at `partial_sig_agg`. For aborted sessions call `secp256k1_musig_keyagg_cache_clear()` (shim extension).

**Bitcoin Core:** does not use MuSig2 in production signing paths — not on the critical path.

---

## Building from Source

```bash
# Minimal (requires UltrafastSecp256k1 parent added first)
add_subdirectory(path/to/UltrafastSecp256k1)
add_subdirectory(path/to/UltrafastSecp256k1/compat/libsecp256k1_shim)

# Standalone (from UltrafastSecp256k1 root)
cmake -S . -B build -DSECP256K1_BUILD_SHIM=ON
cmake --build build
```

**Requirements:**
- C++20 compiler (GCC 13+, Clang 16+, MSVC 2022+)
- UltrafastSecp256k1 CPU library (`fastsecp256k1` target)
- No external dependencies beyond the standard library

---

## Shim Extensions (non-standard additions)

These functions have no counterpart in upstream libsecp256k1:

```c
/* MuSig2: explicitly release internal state for an abandoned session.
 * Not needed for sessions that complete partial_sig_agg (auto-cleanup).
 * No-op on null or never-initialised cache. */
void secp256k1_musig_keyagg_cache_clear(secp256k1_musig_keyagg_cache *keyagg_cache);
```

---

## Test Suite

```bash
cmake -S . -B build -DSECP256K1_SHIM_BUILD_TESTS=ON
cmake --build build
./build/shim_test
```

The test suite covers all 11 shim modules with parity checks against libsecp256k1
expected behavior, including BUG-1 (BIP-66 DER leading-zero rejection) regression tests.

Parity checker (CI gate):
```bash
python3 scripts/check_libsecp_shim_parity.py
```

---

## Production Use

The shim is used in production by **Sparrow Wallet Frigate** for Silent Payments  
(BIP-352) GPU scanning. See [docs/ADOPTION.md](../../docs/ADOPTION.md).

Bitcoin Core integration: 693/693 `test_bitcoin` pass, 0 failures.  
See [BITCOIN_CORE_PR_BLOCKERS.md](../../BITCOIN_CORE_PR_BLOCKERS.md).
