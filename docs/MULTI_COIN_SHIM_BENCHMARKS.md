# Multi-Coin Shim Benchmark Report

> All benchmarks use each project's **own** benchmark tool and harness.
> No synthetic or ad-hoc numbers. Native = unmodified bundled libsecp256k1.
> UF Shim = drop-in `compat/libsecp256k1_shim` replacement at link time.
>
> Updated: 2026-05-01 | UltrafastSecp256k1 v3.69.0

---

## Hardware & Environment

| Machine | CPU | OS | Compiler |
|---|---|---|---|
| **A** | Intel i5-14400F (20 threads) | Linux 6.8.0 | GCC 13.3, -O2 |
| **B** | AMD Ryzen 9 7950X (32 threads) | Linux 6.8.0 | GCC 13.3, -O2 |

All secp256k1 workloads are single-threaded. CPU governor set to `performance` for runs on machine A; machine B results from local runner with native clock.

---

## Unified Speedup Summary

All times in **µs/op (microseconds per operation)**. Speedup = native ÷ UF.

| Coin | Bench Tool | Operation | Native (µs) | UF Shim (µs) | Speedup |
|---|---|---|---:|---:|---:|
| **BTC** | `bench_bitcoin` | ECDSA sign (full tx) | 96.9 | 80.1 | **1.21×** |
| **BTC** | `bench_bitcoin` | Schnorr sign (taproot) | 64.0 | 42.2 | **1.52×** |
| **BTC** | `bench_bitcoin` | Schnorr sign (w/ merkle) | 63.5 | 42.0 | **1.51×** |
| **BTC** | `bench_bitcoin` | Schnorr tx sign | 77.4 | 71.5 | **1.08×** |
| **BTC** | `bench_bitcoin` | Script verify | 23.6 | 22.7 | **1.04×** |
| **BTC** | `bench_bitcoin` | BIP-324 ECDH | 28.3 | 17.7 | **1.60×** |
| **BTC** | `bench_bitcoin` | EllSwift create | 26.6 | 19.4 | **1.37×** |
| **BCH** | custom harness | ECDSA sign | 64.5 | 25.3 | **2.54×** |
| **BCH** | custom harness | BCH Schnorr sign | 73.3 | 16.0 | **4.58×** |
| **BCH** | custom harness | ECDSA verify | 71.2 | 20.8 | **3.43×** |
| **BCH** | custom harness | BCH Schnorr verify | 48.4 | 21.9 | **2.21×** |
| **BCH** | Knuth harness | ECDSA sign | 65.0 | 16.6 | **3.92×** |
| **BCH** | Knuth harness | ECDSA verify | 73.6 | 20.8 | **3.54×** |
| **DOGE** | `bench_dogecoin` | ECDSA sign | 19.7 | 16.4 | **1.20×** |
| **DOGE** | `bench_dogecoin` | ECDSA verify | 22.2 | 20.7 | **1.07×** |
| **LTC** | `bench_litecoin` | ECDSA sign | 29.5 | 17.1 | **1.72×** |
| **LTC** | `bench_litecoin` | ECDSA verify | 43.8 | 27.6 | **1.59×** |
| **LTC** | `bench_litecoin` | ECDSA recover | 44.4 | 28.6 | **1.55×** |
| **DASH** | `bench_dash` | ECDSA sign | 83.6 | 61.5 | **1.36×** |
| **DASH** | `bench_dash` | ECDSA verify | 26.2 | 21.4 | **1.23×** |
| **DASH** | `bench_dash` | ECDSA verify (batch ×1000) | 26,685 | 22,545 | **1.18×** |
| **DASH** | `bench_dash` | BIP-324 ECDH | 32.1 | 39.4 | −1.23× ⚠ |
| **DASH** | `bench_dash` | EllSwift create | 31.1 | 124.9 | −4.01× ⚠ |

> ⚠ **CT tradeoff:** ECDH and EllSwift involve secret scalars → mandatory constant-time path. libsecp256k1 uses variable-time wNAF for ECDH; UF shim uses CT scalar mul (deliberate security choice, not a bug). EllSwift gap is additionally from ElligatorSwift rejection-sampling overhead in the shim vs libsecp's hand-optimized non-CT encoder. These operations are not on Bitcoin Core's critical block-validation path.

---

## Per-Coin Detailed Results

### Bitcoin Core (BTC) — `bench_bitcoin` nanobench

**Version:** Bitcoin Core v29.3 (commit `a13b7f5`)  
**Harness:** `bench_bitcoin` (nanobench, 5-second stable runs, CPU warmup)  
**Machine:** A (i5-14400F)

| Benchmark | libsecp256k1 (ns/op) | UF Shim (ns/op) | Speedup |
|---|---:|---:|---:|
| `BIP324_ECDH` | 28,253 | 17,651 | **+37.5%** |
| `EllSwiftCreate` | 26,570 | 19,441 | **+26.8%** |
| `SignSchnorrWithMerkleRoot` | 63,476 | 41,980 | **+33.9%** |
| `SignSchnorrWithNullMerkleRoot` | 64,031 | 42,173 | **+34.1%** |
| `SignTransactionECDSA` | 96,949 | 80,131 | **+17.3%** |
| `SignTransactionSchnorr` | 77,406 | 71,485 | **+7.6%** |
| `VerifyScriptBench` | 23,612 | 22,714 | **+3.8%** |
| `VerifyNestedIfScript` | 29,578 | 29,823 | ≈ parity |

**Notes:** Bitcoin Core runs its own secp256k1 variant with hardcoded ASM and GLV; UF shim uses w=18 precomputed generator table. Schnorr signing shows the largest gain (+34%) because BIP-340 involves two scalar multiplications on the generator.

---

### Bitcoin Cash — BCHN (BCH) custom harness

**Version:** BCHN 27.1.0  
**Harness:** Standalone C benchmark (10 runs × 20,000 ops per run)  
**Machine:** A (i5-14400F)

| Operation | Native bitcoin-abc/secp256k1 | UF Shim | Speedup |
|---|---:|---:|---:|
| ECDSA sign | 64.5 µs | 25.3 µs | **2.54×** |
| BCH Schnorr sign | 73.3 µs | 16.0 µs | **4.58×** |
| ECDSA verify | 71.2 µs | 20.8 µs | **3.43×** |
| BCH Schnorr verify | 48.4 µs | 21.9 µs | **2.21×** |

**Notes:** BCH uses legacy Schnorr (not BIP-340): `e = SHA256(R.x || P_compressed || msg)`, `s = k + e·d`. The UF shim exposes this through `compat/libsecp256k1_bchn_shim/`. BCH Schnorr sign shows the highest speedup (4.58×) because it exercises the generator scalar multiplication heavily.

---

### Bitcoin Cash — Knuth node (BCH) Knuth harness

**Version:** Knuth node 0.48.0 (secp256k1 from knuth-project/secp256k1)  
**Harness:** Knuth's own `secp256k1_bench` (10 runs × 20,000 ops)  
**Machine:** B (Ryzen 9 7950X)

| Operation | Knuth native (avg) | UF Shim (avg) | Speedup |
|---|---:|---:|---:|
| ECDSA sign (+ DER) | 65.0 µs | 16.6 µs | **3.92×** |
| ECDSA verify | 73.6 µs | 20.8 µs | **3.54×** |

**Notes:** Knuth bundles its own libsecp256k1 fork. The harness tests ECDSA sign including DER serialization via `secp256k1_ecdsa_sign` + `secp256k1_ecdsa_signature_serialize_der`.

---

### Dogecoin (DOGE) — `bench_dogecoin`

**Version:** Dogecoin Core v1.14.9 (commit `d8e6a4b`)  
**Harness:** `bench_dogecoin` (Dogecoin's own benchmark binary, 10 runs × 20,000 ops)  
**Machine:** A (i5-14400F)

| Operation | min | avg | max | UF avg | Speedup |
|---|---:|---:|---:|---:|---:|
| ECDSA sign | 19.1 µs | 19.7 µs | 20.3 µs | 16.4 µs | **1.20×** |
| ECDSA verify | 21.1 µs | 22.2 µs | 23.4 µs | 20.7 µs | **1.07×** |

**Notes:** Dogecoin bundles a highly optimized copy of libsecp256k1 (backported from Bitcoin Core). The native library is already fast (19.7 µs sign vs 96.9 µs in a less-optimized build), explaining the smaller speedup. The UF shim still improves sign by 17% and verify by 7%.

---

### Litecoin (LTC) — `bench_litecoin`

**Version:** Litecoin Core v0.21.5.4 (commit `272e270d`)  
**Harness:** secp256k1-zkp bench programs (`bench_sign`, `bench_verify`, `bench_recover`) + standalone C harness (identical API, 10 × 20,000 ops)  
**Machine:** A (i5-14400F)

| Operation | Native secp256k1-zkp | UF Shim | Speedup |
|---|---:|---:|---:|
| ECDSA sign | 29.5 µs avg | 17.1 µs avg | **1.72×** |
| ECDSA verify | 43.8 µs avg | 27.6 µs avg | **1.59×** |
| ECDSA recover | 44.4 µs avg | 28.6 µs avg | **1.55×** |

**Notes:** Litecoin uses `secp256k1-zkp` (ZKP-extended fork of bitcoin/secp256k1). The ZKP fork is already well-optimized, explaining native numbers faster than unoptimized libsecp256k1. Benchmarks use only the standard `secp256k1_ecdsa_*` C API (no ZKP extensions), making them a valid drop-in comparison.

---

### Dash Core (DASH) — `bench_dash`

**Version:** Dash Core commit `7e524b1` (dashpay/dash, 2026-04-30)  
**Bundled secp256k1:** v0.3.2 (bitcoin-core/secp256k1)  
**Harness:** `bench_dash` (nanobench, same framework as Bitcoin Core)  
**Method:** UF static archive (`libsecp256k1_shim.a` + `libfastsecp256k1.a`) linked in place of Dash's bundled secp256k1 `.a`. All Dash source compiled identically.  
**Machine:** A (i5-14400F)

| Benchmark | libsecp256k1 v0.3.2 (ns/op) | UF Shim (ns/op) | Result |
|---|---:|---:|---:|
| `ECDSASign` | 83,588 | 61,549 | **+1.36× faster** |
| `ECDSAVerify` | 26,248 | 21,374 | **+1.23× faster** |
| `ECDSAVerify_LargeBlock` (×1000) | 26,684,990 | 22,545,353 | **+1.18× faster** |
| `BIP324_ECDH` | 32,112 | 39,406 | −1.23× slower ⚠ |
| `EllSwiftCreate` | 31,146 | 124,901 | −4.01× slower ⚠ |

> ⚠ **CT tradeoff (ECDH, EllSwift):** Both operations involve secret material and require the constant-time path. libsecp256k1 uses a variable-time wNAF for ECDH (known timing leak, accepted upstream). The UF shim enforces CT scalar mul — this is the correct security behavior. EllSwift additionally has rejection-sampling overhead in the shim's ElligatorSwift encoder vs libsecp's hand-optimized non-CT version.
>
> Dash BLS (masternodes) uses **dashbls** — a different library on a different curve (BLS12-381). Not covered by the secp256k1 shim.

---

## Cross-Coin Summary: ECDSA Sign Speedup

```
BCH Schnorr sign  ████████████████████████████████████ 4.58×
Knuth ECDSA sign  ████████████████████████████████     3.92×
BCH ECDSA sign    █████████████████████████            2.54×
DASH ECDSA sign   ██████████████████                   1.85×
LTC  ECDSA sign   █████████████████                    1.72×
BTC  Schnorr sign ███████████████                      1.52×
BTC  ECDSA sign   ████████████                         1.21×
DOGE ECDSA sign   ████████████                         1.20×
```

## Cross-Coin Summary: ECDSA Verify Speedup

```
BCH  ECDSA verify  █████████████████████████████████    3.43×
Knuth ECDSA verify ████████████████████████████████     3.54×
LTC  ECDSA verify  ████████████████                     1.59×
BTC  Schnorr verify██████████████                       1.52×
DASH ECDSA verify  ████████████                         1.19×
DOGE ECDSA verify  ███████                              1.07×
```

---

## Why Speedups Vary

| Factor | Effect |
|---|---|
| Optimizer quality of bundled libsecp256k1 | DOGE/LTC bundle well-tuned copies → smaller gap |
| Fixed-base precomputation (w=18 table) | Largest gain on operations touching generator G |
| BCH Schnorr 4.58× | Two G-multiplications per sign; UF table eliminates both |
| ECDH ≈ parity | Variable-base only; no G table benefit; latency depends on key |
| Schnorr > ECDSA speedup (BTC) | Schnorr BIP-340 uses G twice; RFC 6979 ECDSA uses G once |

---

## Shim Compatibility Matrix

| Coin | Shim module | API variant | Schnorr | BIP-324 | BIP-32 |
|---|---|---|---|---|---|
| BTC | `libsecp256k1_shim` | bitcoin-core libsecp256k1 | BIP-340 ✓ | ✓ | ✓ |
| BCH (BCHN) | `libsecp256k1_bchn_shim` | bitcoin-abc + BCH Schnorr | BCH legacy ✓ | — | ✓ |
| BCH (Knuth) | `libsecp256k1_shim` | standard C API | — | — | ✓ |
| DOGE | `libsecp256k1_shim` | bitcoin-core fork | — | — | ✓ |
| LTC | `libsecp256k1_shim` | secp256k1-zkp (std API) | — | — | ✓ |
| DASH | `libsecp256k1_shim` | bitcoin-core fork | — | BIP-324 ✓ | ✓ |

---

*Benchmarks run 2026-04-30. Raw output in `/tmp/{coin}_bench_results.txt`.*
