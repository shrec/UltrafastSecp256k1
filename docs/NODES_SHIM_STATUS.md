# Node Shim Status — UltrafastSecp256k1

Drop-in engine replacement for cryptocurrency node implementations.  
**Method:** swap the secp256k1 library at link time; run the node's own benchmark harness.

---

## Shim Compatibility Matrix

| Shim | Covers | API | Status |
|---|---|---|---|
| `compat/libsecp256k1_shim/` | Bitcoin Core, Litecoin Core, Dash Core, Dogecoin | Standard libsecp256k1 | ✅ Complete |
| `compat/libsecp256k1_bchn_shim/` | BCHN, Flowee, Electron Cash (C++) | BCHN fork + BCH Schnorr | ✅ Complete |

---

## Benchmark Results

### Knuth secp256k1 (k-nuth/secp256k1)

**Harness:** Knuth `bench.h` (10 × 20,000 ops, median)  
**Version:** `d446f597`

| Operation | Knuth native | UF shim | Speedup |
|---|---|---|---|
| ECDSA Sign + DER | 63.2 μs | 16.6 μs | **+281% (3.8×)** |
| ECDSA Verify | 71.5 μs | 20.0 μs | **+258% (3.6×)** |

---

### Bitcoin Core (bitcoin/bitcoin)

**Harness:** Bitcoin Core `bench_bitcoin`  
**Version:** v29.x (same as `docs/BITCOIN_CORE_TEST_RESULTS.json`)

| Benchmark | libsecp256k1 | UF shim | Δ |
|---|---|---|---|
| SignTransactionECDSA | 96,006 ns | 79,196 ns | **+17.5%** |
| SignTransactionSchnorr | 80,368 ns | 73,663 ns | **+8.3%** |
| SignSchnorrWithMerkleRoot | 66,520 ns | 39,131 ns | **+41.1%** |
| SignSchnorrWithNullMerkleRoot | 65,637 ns | 39,107 ns | **+40.4%** |
| VerifyScriptBench | 24,555 ns | 23,231 ns | **+5.4%** |

---

### Litecoin Core (litecoin-project/litecoin)

**Harness:** `bench_litecoin` (Bitcoin Core fork, same harness)  
**Version:** 0.21.x  
**Status:** 📋 Planned

| Benchmark | libsecp256k1 | UF shim | Δ |
|---|---|---|---|
| SignTransactionECDSA | — | — | — |
| VerifyScriptBench | — | — | — |

---

### Dash Core (dashpay/dash)

**Harness:** `dash-bench` (Bitcoin Core fork harness)  
**Version:** 21.x  
**Status:** 📋 Planned

| Benchmark | libsecp256k1 | UF shim | Δ |
|---|---|---|---|
| SignTransactionECDSA | — | — | — |
| VerifyScriptBench | — | — | — |

---

### Dogecoin Core (dogecoin/dogecoin)

**Harness:** Dogecoin bench.h harness (10 runs × 20,000 ops)  
**Version:** dogecoin/dogecoin master  
**Note:** Uses newer libsecp256k1 with existing optimizations.

| Operation | Dogecoin native | UF shim | Speedup |
|---|---|---|---|
| ECDSA Sign + DER | 18.7 μs | 16.4 μs | **+14%** |
| ECDSA Verify | 21.8 μs | 20.3 μs | **+7%** |

---

### BCHN (Bitcoin Cash Node)

**Harness:** 10 runs × 20,000 ops per-op timing  
**Version:** master (bitcoin-cash-node/bitcoin-cash-node)  
**Modules:** ECDSA + BCH Schnorr (`secp256k1_schnorr.h` — non-BIP340)

| Operation | BCHN native | UF shim | Speedup |
|---|---|---|---|
| ECDSA Sign + DER | 60.9 μs | 15.6 μs | **+290% (3.9×)** |
| BCH Schnorr Sign | 70.6 μs | 14.6 μs | **+384% (4.8×)** |
| ECDSA Verify | 70.6 μs | 19.4 μs | **+264% (3.6×)** |
| BCH Schnorr Verify | 46.5 μs | 21.3 μs | **+118% (2.2×)** |

---

### Flowee the Hub (floweethehub/thehub)

**Harness:** Custom benchmark  
**Status:** 📋 Planned

---

### bchd (bcoin-org/bchd)

**Language:** Go  
**Crypto:** btcec (Go secp256k1)  
**Shim approach:** Go wrapper via CGo → libsecp256k1_shim  
**Status:** 📋 Research

---

### Bitcoin Verde (bitcoin-verde/bitcoin-verde)

**Language:** Java  
**Crypto:** Bouncy Castle  
**Shim approach:** JNI via existing Java bindings  
**Status:** 📋 Research

---

## Integration Guide

### Standard libsecp256k1 nodes (BTC forks)

All Bitcoin Core forks use identical `secp256k1.h` API. Integration is the same for all:

```cmake
# In node's src/CMakeLists.txt, replace:
#   add_subdirectory(secp256k1)
# with:
add_subdirectory(path/to/UltrafastSecp256k1/compat/libsecp256k1_shim)
# secp256k1_shim provides the same target name: secp256k1
```

Or build with the shim as a separate static library and override link:
```bash
cmake -DSECP256K1_LIBRARY=/path/to/libsecp256k1_shim.a \
      -DSECP256K1_INCLUDE_DIR=/path/to/UltrafastSecp256k1/compat/libsecp256k1_shim/include \
      ...
```

**Applies to:** Bitcoin Core, Litecoin Core, Dash Core, Dogecoin Core, BCHN, Flowee.

### BCH nodes (BCHN secp256k1 fork)

BCHN uses a fork that adds BCH Schnorr (`secp256k1_schnorr.h`). Use `libsecp256k1_bchn_shim`:

```cmake
add_subdirectory(path/to/UltrafastSecp256k1/compat/libsecp256k1_bchn_shim)
```

The BCH Schnorr scheme differs from BIP-340 (uses `SHA256(R.x || P_compressed || msg)` instead of tagged hash). The bchn shim implements the correct BCH variant.

---

## Running Benchmarks

```bash
# Automated benchmark runner (TODO: script per node)
scripts/bench_nodes/run_all.sh

# Manual: Bitcoin Core
cd bitcoin && cmake -B build -DUSE_ULTRAFAST_SECP256K1=ON
cmake --build build --target bench_bitcoin
./build/src/bench/bench_bitcoin -filter="Sign|Verify"

# Manual: Knuth
# (see ci/bench_nodes/bench_knuth.sh)
```

---

## Adding a New Node

1. Clone the node repository
2. Find where it links against secp256k1 (usually `src/CMakeLists.txt`)
3. Swap with `libsecp256k1_shim` or `libsecp256k1_bchn_shim`
4. Build their benchmark binary
5. Run with their native harness
6. Add results to this table

If the node uses libsecp256k1 with no custom extensions: **zero code changes needed** — just swap the CMake target.
