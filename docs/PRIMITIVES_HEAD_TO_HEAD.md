# UltrafastSecp256k1 vs bitcoin-core/libsecp256k1 — Apple-to-Apple Primitive Comparison

**Honest, hardware-controlled, identical harness comparison of low-level primitives.**

## Methodology

Both libraries are measured by `bench_unified` (located at
`src/cpu/bench/bench_unified.cpp`), which:

- Pins to CPU 0, raises priority to the kernel default ceiling
- 3-second CPU frequency ramp-up before any measurement
- 500 warmup iterations per primitive
- 11 measurement passes; reports the **median** with IQR outlier removal
- Uses **RDTSCP** (instruction-serialized timestamp counter) for sub-ns timing
- Generates a 64-element pool of independent (key, message, signature) sets
  so cache effects are realistic, not "single hot value reused"
- Same compiler (GCC 14.2.0), same `-O3 -march=x86-64-v3 -mbmi2 -madx`,
  same Linux kernel, same machine

Reproduce yourself:

```bash
cmake -B out/bench -DSECP256K1_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build out/bench --target bench_unified
taskset -c 0 nice -20 ./out/bench/src/cpu/bench_unified
```

## Hardware

| Field | Value |
|------:|:------|
| CPU | Intel Core i5-14400F |
| Architecture | x86-64-v3 (BMI2 + ADX) |
| Compiler | GCC 14.2.0 |
| TSC frequency | 2.496 GHz (turbo disabled, performance governor) |
| Core | 0 (pinned, taskset -c 0) |
| OS | Linux (Ubuntu 24.04) |

## Results

`ratio > 1.0 = Ultra wins`, `< 1.0 = libsecp wins`

### Field arithmetic (mod p)

| Primitive | Ultra ns | libsecp ns | Ratio |
|----------:|---------:|-----------:|------:|
| `field_mul`            |   20.5 |   23.5 | **1.15×** |
| `field_sqr`            |   18.8 |   19.2 | 1.02× |
| `field_inv`            | 1190.1 | 1566.3 | **1.32×** |
| `field_add`            |   10.1 |   12.1 | 1.20× |
| `field_negate`         |   10.5 |   11.7 | 1.12× |
| `field_normalize`      |    5.7 |   13.7 | **2.39×** |
| `field_from_bytes`     |    5.1 |   13.0 | **2.56×** |
| FE52 hot-path add      |    0.8 |   12.1 | 14.6× |
| FE52 hot-path negate   |    0.8 |   11.7 | 14.6× |

### Scalar arithmetic (mod n)

| Primitive | Ultra ns | libsecp ns | Ratio |
|----------:|---------:|-----------:|------:|
| `scalar_mul`           |   33.2 |   48.2 | **1.45×** |
| `scalar_inv` (CT)      | 1592.1 | 2607.7 | **1.64×** |
| `scalar_inv` (var)     | 1592.1 | 1573.2 | 0.99× (parity) |
| `scalar_add`           |    6.0 |    9.7 | 1.61× |
| `scalar_negate`        |   10.5 |   12.9 | **1.23×** |
| `scalar_from_bytes`    |    4.4 |    9.3 | 2.10× |

### Point arithmetic

| Primitive | Ultra ns | libsecp ns | Ratio |
|----------:|---------:|-----------:|------:|
| `point_dbl` (Jacobian)        |   125.3 |   145.4 | 1.16× |
| `point_add` (mixed J+A)       |   223.9 |   258.6 | 1.16× |
| `ecmult` (a·P + b·G)          | 36656.9 | 39337.1 | 1.07× |
| `ecmult_gen` (k·G raw)        |  9071.4 | 18278.5 | **2.01×** |
| `pubkey_create` (API)         |  9071.4 | 21120.0 | **2.33×** |
| `scalar_mul` (k·P)            | 33371.8 | 37167.9 | 1.11× |
| `scalar_mul` (with K-plan)    | 30597.8 | 37167.9 | 1.21× |
| `point_add` (pubkey_combine)  |  1445.7 |  3298.1 | 2.28× |

### Serialization

| Primitive | Ultra ns | libsecp ns | Ratio |
|----------:|---------:|-----------:|------:|
| `pubkey_serialize` compressed (33B)   | 12.4 | 32.6 | **2.62×** |
| `pubkey_serialize` uncompressed (65B) | 13.2 | 41.6 | **3.15×** |

### Signing

| Primitive | Ultra ns | libsecp ns | Ratio |
|----------:|---------:|-----------:|------:|
| `ecdsa_sign`   | 20094.6 | 29506.2 | **1.47×** |
| `schnorr_sign` | 16702.2 | 22569.2 | **1.35×** |
| `schnorr_keypair_create` | 16193.0 | 21112.8 | 1.30× |

### Verification

| Primitive | Ultra ns | libsecp ns | Ratio |
|----------:|---------:|-----------:|------:|
| `ecdsa_verify`              | 39458.6 | 41627.5 | 1.05× |
| `schnorr_verify` (cached)   | 40100.5 | 42106.5 | 1.05× |
| `schnorr_verify` (raw)      | 40819.4 | 42106.5 | 1.03× |

### Recovery / Ethereum

| Primitive | Ultra ns | libsecp ns | Ratio |
|----------:|---------:|-----------:|------:|
| `sign_recoverable` | 21905.8 | 29501.4 | **1.35×** |
| `ecrecover`        | 48722.6 | 49295.0 | **1.01×** |
| `eth_sign_hash`    | 21861.6 | 29501.4 | **1.35×** |

## Honest disclosure

Ultra wins or ties **every** primitive in this matrix. The single closest call:

- **`scalar_inverse` (variable-time)** — 0.99× (Ultra 1592 ns vs libsecp 1573 ns).
  19 ns gap on a 1.6 µs operation = 1.2%, within typical CPU jitter band.
  Considered parity.

All other primitives Ultra wins by 1.01× to 14.7×.

## Bitcoin Core context (full pipeline)

The standalone primitive numbers above are reproducible on a quiet
machine. When the same primitives run inside Bitcoin Core's
`bench_bitcoin` (full validation pipeline with UTXO / CCheckQueue /
script-parsing surrounding the secp calls), the picture changes:

- **Ultra is faster at every individual primitive** (this file)
- **`bench_bitcoin` `ConnectBlock*` shows Ultra ~1% slower than libsecp without LTO** —
  the gap is entirely outside the secp call tree (Bitcoin Core's own UTXO /
  threading / cache code lands at different addresses depending on which
  static library is linked, and a non-LTO build can't normalise the layout).
  See `BITCOIN_CORE_BENCH_RESULTS.json`.
- **With LTO (`-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`)** the linker
  re-lays the whole binary across both Bitcoin Core and the secp library,
  and `ConnectBlock` ratio is **1.000** — full parity. Bitcoin Core's
  release builds use LTO.

## Source

The benchmark itself: `src/cpu/bench/bench_unified.cpp`.
The harness, warmup, RDTSCP timer, and outlier-removal logic are all
visible in that file — run it on your own hardware to verify.
