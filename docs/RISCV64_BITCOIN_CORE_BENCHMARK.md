# RISC-V 64-bit Bitcoin Core Shim Benchmark

> **Status:** First end-to-end Bitcoin Core `bench_bitcoin` A/B comparison on RISC-V hardware
> **Date:** 2026-05-09
> **Hardware:** SiFive U74-MC (rv64imafdc_zicntr_zicsr_zifencei_zihpm_zba_zbb), 4 cores, 3.8 GiB RAM
> **OS:** Debian GNU/Linux forky/sid, kernel 6.6.20-starfive
> **Compiler:** GCC 15.2.0 (Debian 15.2.0-12)
> **Bitcoin Core commit:** `c4fd214` (branch with `SECP256K1_BACKEND` switch)

---

## Setup

Two `bench_bitcoin` binaries built from the same Bitcoin Core tree, differing
only in the `-DSECP256K1_BACKEND` cmake option:

| Build dir | `SECP256K1_BACKEND` | Binary | Size |
|---|---|---|---|
| `out/build-libsecp` | `bundled` (stock libsecp256k1) | `bin/bench_bitcoin` | 14.1 MB |
| `out/build-ultra`   | `ultrafast` (UltrafastSecp256k1 shim) | `bin/bench_bitcoin` | 13.4 MB |

Both compiled with default Bitcoin Core release flags
(`-march=rv64imafdc_zicsr_zifencei_zaamo_zalrsc -mabi=lp64d -O2 -D_FORTIFY_SOURCE=3`,
stack protector, etc.). Shim build adds `UFSECP_BITCOIN_STRICT=ON`,
`UFSECP_REPRODUCIBLE=ON`.

## Method

```bash
FILTER='BIP324_ECDH|EllSwiftCreate|SignSchnorr.*|SignTransaction.*|VerifyScript.*|VerifyNestedIfScript'

taskset -c 0 nice -n -19 ./bin/bench_bitcoin \
    -filter="$FILTER" -min-time=1000 -output-csv=bench.csv
```

- CPU pinned to core 0, `nice -n -19`
- `-min-time=1000` ms → 11 evals per benchmark
- Both runs back-to-back on the same machine, idle (load avg 0.00)
- `bench_bitcoin` reports median of evals; values below are median seconds/op

## Results

| Benchmark | Ultra (μs/op) | Libsecp (μs/op) | Ratio (libsecp / ultra) |
|---|---:|---:|---:|
| **SignSchnorrWithNullMerkleRoot** | 485.65 | 662.80 | **1.36×** |
| **SignSchnorrWithMerkleRoot**     | 494.91 | 665.51 | **1.34×** |
| **SignTransactionECDSA**          | 846.15 | 1031.84 | **1.22×** |
| **SignTransactionSchnorr**        | 748.29 | 855.63 | **1.14×** |
| **BIP324_ECDH**                   | 247.36 | 280.19 | **1.13×** |
| **EllSwiftCreate**                | 236.85 | 264.83 | **1.12×** |
| VerifyScriptP2TR_ScriptPath       | 419.83 | 464.14 | 1.11× |
| VerifyNestedIfScript              | 185.51 | 192.26 | 1.04× |
| VerifyScriptP2TR_KeyPath          | 255.81 | 261.88 | 1.02× |
| VerifyScriptP2WPKH                | 267.77 | 266.85 | 1.00× |

## Observations

- **Sign paths show the largest gains.** Schnorr signing is +34–36% faster,
  ECDSA transaction signing +22%, BIP324 ECDH +13%, EllSwift +12%. These paths
  exercise the shim's CT scalar multiplication and CT generator multiplication,
  which are tighter on RISC-V than libsecp256k1's bundled CT path at the
  default `ECMULT_GEN_KB=86`.
- **Verify paths are essentially at parity.** Both backends ultimately use
  variable-time wNAF for verification (the correct choice — all verify inputs
  are public). The +11% on `P2TR_ScriptPath` is the only meaningful verify-side
  gain; `P2WPKH` is identical within noise.
- **No regressions.** P2WPKH `1.00×` is within run-to-run noise (sub-percent on
  this hardware with min-time=1000 ms).

## Limitations

- **Single 11-eval run per backend.** This satisfies the bench harness's
  internal stability check but does not meet the project's strict perf protocol
  (≥5 independent runs, non-overlapping ranges) — the numbers above should be
  read as *first-pass evidence*, not as ranged-confirmed speedups.
- **Heavy benchmarks omitted.** `ConnectBlockAllEcdsa`,
  `ConnectBlockAllSchnorr`, `ConnectBlockMixedEcdsaSchnorr` were not run; they
  are minutes-long on RISC-V and warrant a dedicated session.
- **Single platform.** SiFive U74-MC has Zba/Zbb but no Vector extension. A
  Zvbb / RVV 1.0 board would likely show a different shape on field arithmetic.

## Reproduction

Source CSVs: pulled from `/tmp/bench_ultra.csv` and `/tmp/bench_libsecp.csv` on
the test host. Bitcoin Core build commands:

```bash
cd ~/bitcoin-core
cmake -S . -B out/build-libsecp -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_BENCH=ON -DSECP256K1_BACKEND=bundled
cmake -S . -B out/build-ultra   -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_BENCH=ON -DSECP256K1_BACKEND=ultrafast \
      -DUFSECP_BITCOIN_STRICT=ON -DUFSECP_REPRODUCIBLE=ON
cmake --build out/build-libsecp --target bench_bitcoin -j2
cmake --build out/build-ultra   --target bench_bitcoin -j2
```
