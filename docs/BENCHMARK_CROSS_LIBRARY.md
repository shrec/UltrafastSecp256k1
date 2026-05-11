# Cross-Library Benchmark Results

**Platform:** Intel Core i5-14400F · Ubuntu 24.04 · GCC 14  
**Method:** Each library tested with its **own benchmark harness** where available;  
UltrafastSecp256k1 shim is a **drop-in replacement** — same benchmark code, engine swapped.  
**Harness:** IQR outlier removal, median of 11 passes (or library-native harness).

---

## Summary Table

**[CPU: i5-14400F · 2.496 GHz turbo-disabled · GCC 13.3.0 · warm w=18 cache · **11-pass IQR median** · 2026-05-11]**  
**[hot-cache; CT-vs-CT for signing; Knuth data [archived]]**

| Operation | UltrafastSecp256k1 | libsecp256k1 | Knuth [arch] | OpenSSL 3.x | UF vs libsecp | UF vs Knuth | UF vs OpenSSL |
|---|---|---|---|---|---|---|---|
| k·G (generator mul) | 9,985 ns | 21,224 ns | — | 397,829 ns | **2.13×** | — | **39.8×** |
| CT ECDSA Sign | 21,435 ns | 29,381 ns | 63,200 ns | 422,124 ns | **1.37×** | **2.9×** | **19.7×** |
| ECDSA Verify | 42,615 ns | 41,759 ns | 71,500 ns | 402,147 ns | 0.98× | **1.7×** | **9.4×** |
| CT Schnorr Sign | 18,064 ns | 22,530 ns | — | N/A | **1.25×** | — | N/A |
| Schnorr Verify | 44,947 ns | 42,154 ns | — | N/A | 0.94× | — | N/A |
| Field mul | 22.2 ns | 23.5 ns | — | — | **1.06×** | — | — |
| Scalar inv (CT) | 1,558 ns | 2,601 ns | — | — | **1.67×** | — | — |
| Pubkey create | 9,985 ns | 21,224 ns | — | — | **2.13×** | — | — |

> **CT vs CT signing:** ECDSA sign **1.37×** faster, Schnorr sign **1.25×** faster than libsecp256k1.  
> ECDSA verify 2% slower; Schnorr verify 6% slower (both variable-time, public data — acceptable).  
> `[arch]` = Knuth data from archived run; not re-measured 2026-05-11.

---

## Bitcoin Core Integration

**Method:** Bitcoin Core's own `bench_bitcoin` binary, shim swapped in via CMake.  
**Source:** `docs/BITCOIN_CORE_BENCH_RESULTS.json` · Generated: 2026-05-08  
**Machine:** Intel Core i5-14400F · Ubuntu 24.04 · GCC 14.2.0 · turbo disabled  
**Note:** These numbers are from the `bench_bitcoin` pipeline (full Bitcoin Core transaction/script context), NOT from `bench_unified`. Medians of ≥3 stable runs.

### Release + LTO (`-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`) — recommended production build

| Benchmark | libsecp256k1 | UltrafastSecp256k1 shim | Speedup |
|---|---|---|---|
| `SignTransactionECDSA` | 175,388 ns | 143,788 ns | **+22%** |
| `SignTransactionSchnorr` | 137,561 ns | 124,059 ns | **+11%** |
| `SignSchnorrWithMerkleRoot` | 111,524 ns | 83,248 ns | **+34%** |
| `SignSchnorrWithNullMerkleRoot` | 111,014 ns | 82,387 ns | **+35%** |
| `VerifyScriptP2WPKH` | 45,083 ns | 45,064 ns | **parity** |
| `VerifyScriptP2TR_KeyPath` | 45,667 ns | 44,363 ns | **+3%** |
| `VerifyScriptP2TR_ScriptPath` | 82,961 ns | 64,975 ns | **+28%** |
| `ConnectBlockAllEcdsa` | 255.9 ms | 255.8 ms | **parity** |
| `ConnectBlockAllSchnorr` | 253.6 ms | 254.2 ms | **parity** |
| `ConnectBlockMixed` | 255.5 ms | 256.0 ms | **parity** |

### Release without LTO

| Benchmark | libsecp256k1 | UltrafastSecp256k1 shim | Speedup |
|---|---|---|---|
| `SignTransactionECDSA` | 167,000 ns | 147,000 ns | **+14%** |
| `SignTransactionSchnorr` | 133,000 ns | 124,000 ns | **+7%** |
| `SignSchnorrWithMerkleRoot` | 109,000 ns | 82,300 ns | **+32%** |
| `SignSchnorrWithNullMerkleRoot` | 109,000 ns | 82,500 ns | **+32%** |
| `VerifyScriptP2WPKH` | 45,083 ns | 45,460 ns | ~parity (+1%) |
| `VerifyScriptP2TR_KeyPath` | 45,667 ns | 44,540 ns | **+2%** |
| `VerifyScriptP2TR_ScriptPath` | 82,961 ns | 64,620 ns | **+28%** |
| `ConnectBlockAllEcdsa` | 248.6 ms | 250.8 ms | −1% (libsecp) |
| `ConnectBlockAllSchnorr` | 246.4 ms | 249.2 ms | −1% (libsecp) |
| `ConnectBlockMixed` | 248.7 ms | 251.5 ms | −1% (libsecp) |

> **ConnectBlock without LTO:** Ultra's larger instruction footprint (~900 KB extra secp256k1 code
> vs libsecp's ~400 KB) causes ~1% L2/L3 cache pressure over 2500 iterations. LTO eliminates this
> by co-optimizing code layout globally. The signing speedups (14–35%) are not affected by LTO.
> See `nolto_gap_root_cause` in `BITCOIN_CORE_BENCH_RESULTS.json` for root-cause evidence.

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
OpenSSL version: `3.0.13` · **[i5-14400F · 2.496 GHz turbo-disabled · GCC 13.3.0 · 11-pass IQR · 2026-05-11]**

| Benchmark | OpenSSL 3.x | UltrafastSecp256k1 (CT) | Speedup |
|---|---|---|---|
| Generator mul (k·G) | 397,829 ns | 9,985 ns | **39.8×** |
| CT ECDSA Sign | 422,124 ns | 21,435 ns | **19.7×** |
| ECDSA Verify | 402,147 ns | 42,615 ns | **9.4×** |

> OpenSSL's secp256k1 uses generic curve code without the optimized field arithmetic
> that libsecp256k1 and UltrafastSecp256k1 implement. CT signing path used for Ultra.

---

## libsecp256k1 (bitcoin-core/secp256k1 v0.7.x)

**Method:** UltrafastSecp256k1's `bench_unified` — identical harness, identical operation count.  
**[CPU: i5-14400F · 2.496 GHz (turbo disabled) · GCC 13.3.0 · warm w=18 cache · **11-pass IQR median** · 2026-05-11]**  
**[hot-cache — w=18 precomputed table warm; no LTO]**

| Operation | UltrafastSecp256k1 | libsecp256k1 | Ratio |
|---|---|---|---|
| Field mul | 22.2 ns | 23.5 ns | **1.06×** |
| Field inv (SafeGCD) | 1,271 ns | 1,513 ns | **1.19×** |
| Scalar inv (CT) | 1,558 ns | 2,601 ns | **1.67×** |
| k·G (ecmult_gen raw) | 9,985 ns | 18,451 ns | **1.85×** |
| Pubkey create (API) | 9,985 ns | 21,224 ns | **2.13×** |
| Scalar mul (k·P) | 35,600 ns | 37,588 ns | **1.06×** |
| Point add (combine) | 1,571 ns | 3,306 ns | **2.10×** |
| ECDSA Verify (primitive) | 44,294 ns | 41,484 ns | 0.94× |
| CT ECDSA Sign | 22,292 ns | 29,503 ns | **1.32×** |
| Schnorr Verify (raw) | 45,654 ns | 42,143 ns | 0.93× |
| CT Schnorr Sign | 18,825 ns | 22,501 ns | **1.20×** |

> **CT vs CT (production-equivalent signing):** ECDSA sign **1.32×** faster, Schnorr sign **1.20×** faster.  
> Primitive verify (single call, no pre-parsed pubkey): ECDSA 6% behind, Schnorr 7% behind.  
> **ConnectBlock verify (pre-parsed pubkeys):** both paths Ultra wins — see table below.

### ConnectBlock (2000 unique pubkeys, pre-parsed pubkeys, bench_unified)

> Pubkeys pre-parsed before the loop (matching Bitcoin Core script validation pattern).  
> ECDSA: `EcdsaPublicKey` with prebuilt GLV tables; Schnorr: `SchnorrXonlyPubkey` (PERF-B).

| Scenario | UltrafastSecp256k1 | libsecp256k1 | Ratio |
|---|---|---|---|
| AllECDSA (2000 sigs) | 83.0 ms | 84.9 ms | **1.02×** ✓ |
| AllSchnorr (2000 sigs) | 84.2 ms | 86.3 ms | **1.02×** ✓ |
| Mixed ECDSA+Schnorr (3000) | 125.5 ms | ~129 ms | **~1.03×** ✓ |
| DerParse+Verify+Normalize | 87.1 ms | 96.8 ms | **1.11×** ✓ |

> All ConnectBlock scenarios: Ultra wins. Pre-parsed pubkeys eliminate per-verify GLV table rebuild
> (~1,954 ns × 2000 = 3.9 ms saved), which is the correct pattern for Bitcoin Core block validation.

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
cmake --build b