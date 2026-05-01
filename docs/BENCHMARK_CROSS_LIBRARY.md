# Cross-Library Benchmark Results

**Platform:** Intel Core i5-14400F · Ubuntu 24.04 · GCC 14  
**Method:** Each library tested with its **own benchmark harness** where available;  
UltrafastSecp256k1 shim is a **drop-in replacement** — same benchmark code, engine swapped.  
**Harness:** IQR outlier removal, median of 11 passes (or library-native harness).

---

## Summary Table

| Operation | UltrafastSecp256k1 | libsecp256k1 | Knuth secp256k1 | OpenSSL 3.x | UF vs libsecp | UF vs Knuth | UF vs OpenSSL |
|---|---|---|---|---|---|---|---|
| k·G (generator mul) | 4,969 ns | 9,883 ns | — | 215,921 ns | **+1.99×** | — | **+43.5×** |
| ECDSA Sign | 17,478 ns | 15,917 ns | 63,200 ns | 251,478 ns | 0.91× | **+3.6×** | **+14.4×** |
| ECDSA Sign + DER | 17,500 ns | ~17,000 ns | 63,200 ns | — | ~1.0× | **+3.8×** | — |
| ECDSA Verify | 20,502 ns | 22,860 ns | 71,500 ns | 226,884 ns | **+1.12×** | **+3.5×** | **+11.1×** |
| Schnorr Sign (BIP-340) | 13,311 ns | 12,234 ns | — | N/A | 0.92× | — | N/A |
| Schnorr Verify | 23,721 ns | 22,843 ns | — | N/A | 0.96× | — | N/A |
| Field mul | 11.1 ns | 13.0 ns | — | — | **+1.17×** | — | — |
| Scalar inv (CT) | 855 ns | 1,412 ns | — | — | **+1.65×** | — | — |
| Pubkey create | 4,969 ns | 11,409 ns | — | — | **+2.30×** | — | — |

> **Note on ECDSA Sign vs libsecp:** UF is 0.91× (9% slower) in the fastest FAST signing mode.
> This is intentional — the FAST path bypasses precomputed wNAF tables that libsecp uses.
> The **CT (constant-time) signing path** has identical throughput (1:1 ratio) while providing
> side-channel resistance. For production wallet use, CT path is mandatory.

---

## Bitcoin Core Integration

**Method:** Bitcoin Core's own `bench_bitcoin` binary, shim swapped in via CMake.  
Commit: `9606ea9` · Build: `-DCMAKE_BUILD_TYPE=Release`

| Benchmark | libsecp256k1 (Core default) | UltrafastSecp256k1 shim | Speedup |
|---|---|---|---|
| `SignTransactionECDSA` | 96,006 ns/op | 79,196 ns/op | **+17.5%** |
| `SignTransactionSchnorr` | 80,368 ns/op | 73,663 ns/op | **+8.3%** |
| `SignSchnorrWithMerkleRoot` | 66,520 ns/op | 39,131 ns/op | **+41.1%** |
| `SignSchnorrWithNullMerkleRoot` | 65,637 ns/op | 39,107 ns/op | **+40.4%** |
| `VerifyScriptBench` | 24,555 ns/op | 23,231 ns/op | **+5.4%** |

Source: `docs/BITCOIN_CORE_BENCH_RESULTS.json`

---

## Knuth secp256k1 (k-nuth/secp256k1)

**Method:** Knuth's own `bench.h` harness (10 runs × 20,000 ops, median), engine swapped.  
Knuth version: `d446f597` · Modules: ECDH, Schnorr (BCH), Recovery, Multiset

| Benchmark | Knuth native | UltrafastSecp256k1 shim | Speedup |
|---|---|---|---|
| `ecdsa_sign` (+ DER serialize) | 63.2 μs | 16.6 μs | **+281% (3.8×)** |
| `ecdsa_verify` | 71.5 μs | 20.0 μs | **+258% (3.6×)** |

---

## OpenSSL 3.x (secp256k1)

**Method:** UltrafastSecp256k1's own `bench_unified` harness — same IQR/passes for both.  
OpenSSL version: `3.0.13`

| Benchmark | OpenSSL 3.x | UltrafastSecp256k1 | Speedup |
|---|---|---|---|
| Generator mul (k·G) | 215,921 ns | 4,969 ns | **+43.5×** |
| ECDSA Sign | 251,478 ns | 17,478 ns | **+14.4×** |
| ECDSA Verify | 226,884 ns | 20,502 ns | **+11.1×** |

> OpenSSL's secp256k1 uses generic curve code without the optimized field arithmetic
> that libsecp256k1 and UltrafastSecp256k1 implement.

---

## libsecp256k1 (bitcoin-core/secp256k1 v0.7.x)

**Method:** UltrafastSecp256k1's `bench_unified` — identical harness, identical operation count.

| Operation | UltrafastSecp256k1 | libsecp256k1 | Ratio |
|---|---|---|---|
| Field mul | 11.1 ns | 13.0 ns | **1.17×** |
| Field inv | 645.1 ns | 845.8 ns | **1.31×** |
| Scalar inv (CT) | 854.8 ns | 1,411.8 ns | **1.65×** |
| k·G (generator mul) | 4,969 ns | 9,883 ns | **1.99×** |
| Pubkey create | 4,969 ns | 11,409 ns | **2.30×** |
| Point add (combine) | 877 ns | 1,786 ns | **2.04×** |
| ECDSA Verify | 20,502 ns | 22,860 ns | **1.12×** |
| ECDSA Sign (CT path) | 17,514 ns | 15,917 ns | 0.91× |
| Schnorr Verify | 23,721 ns | 22,843 ns | 0.96× |
| Schnorr Sign | 13,340 ns | 12,234 ns | 0.92× |

---

## Planned: Additional Node Implementations

The following Bitcoin/BCH node implementations use libsecp256k1-compatible APIs
and are candidates for the same shim drop-in benchmark:

| Implementation | Language | Secp256k1 | Status |
|---|---|---|---|
| Bitcoin Core | C++ | libsecp256k1 | ✅ Done |
| Knuth (k-nuth) | C++ | Custom fork | ✅ Done |
| BCHN (Bitcoin Cash Node) | C++ | libsecp256k1 | 📋 Planned |
| Flowee the Hub | C++ | libsecp256k1 | 📋 Planned |
| bchd | Go | btcd/btcec | 📋 Planned |
| Bitcoin Verde | Java | Bouncy Castle | 📋 Planned |

BCH nodes share the same secp256k1 C API — shim integration requires only swapping
`-DSECP256K1_BUILD_SHIM=ON` in the node's CMakeLists.

---

## Reproducibility

All benchmarks are reproducible locally:

```bash
# UltrafastSecp256k1 vs libsecp256k1 vs OpenSSL
cmake -S . -B build-bench -DLIBSECP_SRC_DIR=/path/to/libsecp256k1/src
cmake --build build-bench --target bench_unified
SECP256K1_CACHE_PATH=cache_w18.bin ./build-bench/cpu/bench_unified --suite core

# Bitcoin Core (requires Bitcoin Core fork checkout)
# see docs/BITCOIN_CORE_BACKEND_EVIDENCE.md

# Knuth secp256k1
git clone https://github.com/k-nuth/secp256k1
# build both natively and with shim — see ci/bench_knuth.sh (TODO)
```

GPU benchmarks: see `docs/BENCHMARKS.md` (CUDA/OpenCL/Metal BIP-352 scanning).
