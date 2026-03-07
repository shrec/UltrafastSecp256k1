# OPTIMIZATION_ARCHITECTURE.md

## UltrafastSecp256k1 -- Optimization Architecture Reference

> This document is the technical reference for every optimization layer in
> UltrafastSecp256k1.  It is generated from code inspection and is the
> companion to the benchmark framework (bench_unified).

---

## 0. Design Philosophy

```
Algorithm > Architecture > Optimization > Hardware
Memory traffic is more expensive than arithmetic
Correctness is absolute; performance is earned
No abstraction is free at scale
Hot paths must be explicit, boring, and predictable
```

---

## 1. Field Representations

The library maintains multiple field element representations, each optimised
for a specific limb width and platform.  Conversions are explicit and
documented.

| Name               | File                | Layout          | Headroom    | Use Case                                   |
|--------------------|---------------------|-----------------|-------------|-------------------------------------------|
| `FieldElement`     | `field.hpp`         | 4x uint64_t     | 0 (normalized) | I/O, serialization, `from_limbs` (DB loads) |
| `FieldElement52`   | `field_52.hpp`      | 5x uint64_t     | 12 bit/limb | ECC point ops on 64-bit; lazy reduction     |
| `FieldElement26`   | `field_26.hpp`      | 10x uint32_t    | 6 bit/limb  | 32-bit targets (ESP32 Xtensa, Cortex-M)     |
| `MidFieldElement`  | `field.hpp`         | 8x uint32_t     | reinterpret | Zero-cost 32-bit view of 4x64               |

### Lazy Reduction

`FieldElement52` allows ~4096 additions without normalization due to 12-bit
headroom.  `FieldElement26` allows ~64 additions.  This avoids reduction in
inner loops (point addition chains), reducing to full `mod p` only at
serialization or comparison boundaries.

### Hybrid Conversion

`FieldElement52::from_fe()` and `to_fe()` convert between 4x64 and 5x52.
Bit mapping:

```
4x64: [0..63][64..127][128..191][192..255]
5x52: [0..51][52..103][104..155][156..207][208..255]
```

---

## 2. Scalar Arithmetic

256-bit scalar field (mod n) using 4x uint64_t limbs.

- `Scalar::mul` -- schoolbook 4x4 with `__int128` accumulator, followed by Barrett or Montgomery reduction
- `Scalar::inverse` -- SafeGCD (Bernstein-Yang), 590 divsteps, constant-time
- `Scalar::from_bytes` -- big-endian 32-byte input, reduction mod n

---

## 3. GLV Endomorphism

Splits 256-bit scalar `k` into two ~128-bit halves `(k1, k2)` such that
`k*P = k1*P + k2*phi(P)`, where `phi(x,y) = (beta*x, y)`.

### Constants

```
lambda = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
beta   = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
```

### Decomposition (Babai Nearest-Plane)

1. Compute `c1 = round(k * g1 / 2^384)`, `c2 = round(k * g2 / 2^384)`
   via 256x256 Comba multiply (16 `__int128` muls on 64-bit, 64 `uint32_t`
   muls on 32-bit) + right-shift + rounding bit
2. `k2 = c1*(-b1) + c2*(-b2) (mod n)`
3. Pick shorter representation (`k2` vs `-k2`) by bitlength
4. `k1 = k - lambda*k2 (mod n)`, again pick shorter

**Lattice basis vectors (128-bit):**

```
A1  = 0x3086d221a7d46bcde86c90e49284eb15
-B1 = 0xe4437ed6010e88286f547fa90abfe4c3
A2  = 0xe4437ed6010e88286f547fa90abfe4c4
B2  = 0x3086d221a7d46bcde86c90e49284eb15
```

---

## 4. Point Multiplication Strategies

### 4.1 FAST Path (`fast::` namespace)

Variable-time.  Not for secret scalars.

- **k*G (generator):** Precomputed comb (Lim-Lee) or precomputed windows
  - Comb: `teeth`-parameterized, `spacing = ceil(256/teeth)`, table = `2^teeth` affine points
  - Default `comb_width=6`: 64-entry table, ~176 KB, 43 dbl + 43 add = 86 ops
  - Aggressive `teeth=11`: 2048-entry, ~3 MB (L2), 24 dbl + 24 add = 48 ops
- **k*P (arbitrary):** wNAF + GLV decomposition, width-5 (16 precomputed points)
- **a*G + b*P (dual):** Shamir's trick -- interleaved double-scalar multiplication

### 4.2 CT Path (`ct::` namespace)

Constant-time for all secret-dependent operations.

- **ct::generator_mul(k):** Hamburg signed-digit comb encoding
  - 4-bit windows, 8-entry precomputed affine table (odd multiples)
  - 64 mixed-affine additions + 64 signed table lookups, NO doublings
  - Table lookup scans ALL entries (CT: `affine_table_lookup_signed`)
  - ~3x faster than generic CT scalar_mul

- **ct::scalar_mul(P, k):** Hamburg signed-digit + GLV
  - GROUP_SIZE=5, 16 odd multiples per sub-curve
  - 125 dbl + 52 unified_add + 52 signed lookups(16)

### 4.3 CT Primitives

| Function                        | Cost       | Notes                                              |
|---------------------------------|------------|-----------------------------------------------------|
| `point_add_complete`            | 11M + 6S   | Brier-Joye unified, handles all edge cases branchless |
| `point_add_mixed_complete`      | 7M + 5S    | Brier-Joye, Jac+Affine (Z2=1)                       |
| `point_add_mixed_unified`       | 7M + 5S    | Brier-Joye, precondition: b != infinity              |
| `point_add_mixed_unified_into`  | 7M + 5S    | In-place variant, avoids 128-byte copy               |
| `point_dbl`                     | 3M + 4S + h | Libsecp-style, handles identity via cmov             |
| `point_dbl_n_inplace`           | N*(3M+4S+h)| Batch N doublings in-place                           |
| `point_cmov` / `point_select`   | O(n)       | CT conditional move / table scan                     |
| `affine_table_lookup_signed`    | O(n)       | Hamburg signed-digit encoding, scans all entries      |
| `point_endomorphism`            | 1M         | `phi(P) = (beta*X, Y, Z)`                           |

### 4.4 Batch Verification

- **Schnorr batch:** Single MSM `sum(a_i*s_i)*G + sum(-a_i*e_i*P_i) + sum(-a_i*R_i) = O`
- **ECDSA batch:** Montgomery batch inversion + per-sig Shamir's trick

Measured speedup (N=64): ~2.5-3.5x vs individual verification.

---

## 5. Assembly and Intrinsics

| File                          | Target    | Operations                                      |
|-------------------------------|-----------|--------------------------------------------------|
| `field_asm.cpp`               | x64 BMI2  | `mul_4x4_bmi2`, `square_4_bmi2`, `mont_reduce`  |
| `field_asm_x64_gas.S`         | x64 GAS   | `field_mul_full_asm`, `field_sqr_full_asm`       |
| `field_asm_arm64.cpp`         | AArch64   | `field_mul_arm64`, `field_sqr_arm64`, add, sub   |
| `decomposition_optimized.hpp` | x64 BMI2  | 16x `_mulx_u64` for GLV decomposition            |
| `ct_field.cpp`                | portable  | SafeGCD divsteps, 5x52 mul/sqr with `__int128`  |
| `glv.cpp`                     | portable  | 4x4 Comba with `__int128`                        |

### Runtime Detection

BMI2 availability is detected at runtime via CPUID (`field_asm.cpp`).
When BMI2 is available, `_mulx_u64` provides carry-free multiplication;
otherwise, falls back to portable C++ with `__int128`.

---

## 6. Build Feature Gates

| Option                           | Default | Effect                                          |
|----------------------------------|---------|-------------------------------------------------|
| `SECP256K1_USE_ASM`              | ON      | Enable inline assembly (x64/RISC-V, 2-5x)      |
| `SECP256K1_USE_FAST_REDUCTION`   | ON      | Fast mod-p reduction (RISC-V asm, x64 BMI2)     |
| `SECP256K1_USE_RISCV_FE52_ASM`   | OFF     | RISC-V 5x52 asm (slower on in-order U74)        |
| `SECP256K1_RISCV_USE_VECTOR`     | ON      | RISC-V Vector Extension (RVV)                   |
| `SECP256K1_RISCV_USE_PREFETCH`   | ON      | Cache prefetch hints (RISC-V)                   |
| `SECP256K1_USE_LTO`              | OFF     | Link Time Optimization                          |
| `SECP256K1_USE_PGO_GEN`          | OFF     | PGO: generate profile                           |
| `SECP256K1_USE_PGO_USE`          | OFF     | PGO: use profile                                |
| `SECP256K1_SPEED_FIRST`          | ON      | Aggressive unsafe-speed optimizations            |
| `SECP256K1_VERBOSE_DEBUG`        | OFF     | Compile-time gated verbose GPU logging           |

---

## 7. Performance Model

### Cost Model (x86-64 / i5-14400F, Clang 19)

Measured with bench_unified (11 passes, IQR, median, RDTSCP):

| Operation               | ns/op    | Cycles (est.) |
|--------------------------|----------|---------------|
| field_mul (4x64)         | ~18      | ~45           |
| field_sqr (4x64)         | ~14      | ~35           |
| field_inv                | ~730     | ~1840         |
| scalar_mul               | ~18      | ~45           |
| scalar_inv (SafeGCD)     | ~790     | ~1990         |
| pubkey_create (k*G)      | ~5,600   | ~14,100       |
| scalar_mul (k*P)         | ~18,800  | ~47,400       |
| dual_mul (a*G+b*P)       | ~19,900  | ~50,200       |
| ECDSA sign (FAST)        | ~8,500   | ~21,400       |
| ECDSA verify             | ~21,100  | ~53,200       |
| Schnorr sign (FAST)      | ~7,100   | ~17,900       |
| Schnorr verify           | ~21,700  | ~54,700       |
| ct::ecdsa_sign           | ~12,100  | ~30,500       |
| ct::schnorr_sign         | ~10,800  | ~27,200       |

### Apple-to-Apple Ratios (Ultra / libsecp256k1)

| Operation        | FAST path | CT path |
|------------------|-----------|---------|
| ECDSA Sign       | 1.9x      | 1.3x   |
| ECDSA Verify     | 1.0x      | 1.0x   |
| Schnorr Sign     | 1.8x      | 1.2x   |
| Schnorr Verify   | 1.0x      | 1.0x   |
| Generator * k    | 2.1x      | --     |

Notes:
- Verify uses only public data (no CT needed); same path for both
- libsecp256k1 signing is always CT; FAST path comparison is unfair for signing
- CT-vs-CT is the true apples-to-apples for signing operations

---

## 8. Benchmark Framework

### Source of Truth

`bench_unified` is the **single canonical benchmark**.  All other bench_*
targets measure specific subsystems (field representations, CT overhead)
but bench_unified is what gets reported.

### Kept Benchmark Targets

| Target         | Source           | Purpose                              |
|----------------|------------------|--------------------------------------|
| `bench_unified`| `bench_unified.cpp` + `libsecp_provider.c` | Full apple-to-apple vs libsecp256k1 |
| `bench_ct`     | `bench_ct.cpp`   | CT layer overhead measurement         |
| `bench_field_52`| `bench_field_52.cpp` | FE52 vs FE64 regression testing  |
| `bench_field_26`| `bench_field_26.cpp` | FE26 vs FE64 (32-bit targets)    |

### Infrastructure

- **Timer:** RDTSCP on x86 (sub-ns), chrono::high_resolution_clock elsewhere
- **Statistics:** IQR-based outlier removal, median of N passes (default 11)
- **Anti-DCE:** `DoNotOptimize()` / `ClobberMemory()` compiler barriers
- **Pinning:** Thread pinned to core 0, priority elevated (Linux/Windows)
- **Data pool:** 64 independent key/msg/sig sets, cycled to defeat caching

### CLI

```
bench_unified [OPTIONS]
  --json <file>    Write structured JSON report
  --suite <name>   core | extended | all (default: all)
  --passes <N>     Override passes (default: 11, min: 3)
  --quick          CI smoke (3 passes, 1/5 iterations)
  --no-warmup      Skip CPU frequency ramp-up
```

### JSON Schema

```json
{
  "metadata": {
    "cpu": "...", "compiler": "...", "arch": "...",
    "timer": "RDTSCP", "tsc_ghz": 2.524,
    "passes": 11, "warmup": 500, "pool_size": 64
  },
  "results": [
    {"section": "FIELD ARITHMETIC (Ultra)", "name": "field_mul", "ns": 18.07},
    {"section": "FAST path (Ultra FAST vs libsecp)", "name": "ECDSA Sign", "ratio": 1.91}
  ]
}
```

### Cross-Platform Reporting

```bash
# Run on each platform
./bench/scripts/run_bench.sh --out-dir results/

# Merge all reports into comparison table
python3 bench/scripts/merge_reports.py --dir results/ -o comparison.md
```

---

## 9. Platform-Specific Notes

### x86-64

- Primary target.  BMI2 intrinsics (`_mulx_u64`) + GAS assembly
- RDTSCP timer provides sub-nanosecond precision
- CPU frequency warmup (3s heavy crypto load) stabilises turbo before measurement

### AArch64

- Full inline assembly for field ops (mul, sqr, add, sub)
- chrono fallback timer
- No BMI2, uses `__int128` for wide multiplies

### RISC-V 64

- Tested on StarFive VisionFive 2 (SiFive U74, in-order)
- Optional RVV (Vector Extension) support
- 5x52 RISC-V asm available but noted as slower on in-order cores
- Prefetch hints configurable

### ESP32-S3 (Xtensa LX7)

- 32-bit target: uses `FieldElement26` (10x26)
- Separate benchmark binary: `examples/esp32_bench_hornet/`
- Same apple-to-apple ratio table format as bench_unified
- Measured: ECDSA Verify 1.78x, Schnorr Verify 1.65x

---

## 10. Versioning

This document describes the optimization architecture as of the
`dev/optimize-verify-gap` branch.  Generated from code inspection;
update when algorithms or parameters change.
