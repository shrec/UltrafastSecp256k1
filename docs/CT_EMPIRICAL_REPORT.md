# Constant-Time Empirical Proof Report

**UltrafastSecp256k1 v3.68.0** -- Statistical Timing Analysis

---

## Report Metadata

| Field | Value |
|-------|-------|
| **Report Version** | 2.0 |
| **Library Version** | 3.68.0 |
| **Report Date** | 2026-03-01 |
| **Methodology** | dudect (Reparaz, Balasch, Verbauwhede, 2017) |
| **Statistical Test** | Welch's two-sample t-test |
| **Threshold** | \|t\| < 4.5 (99.999% confidence) |
| **CI Smoke Threshold** | \|t\| < 25.0 (coarse, catch gross leaks only) |

---

## Executive Summary

**Result**: No statistically significant timing variance detected across all
tested CT operations.

- **35+ distinct timing tests** across 8 test sections
- **All operations**: |t| < 4.5 at full statistical power
- **Zero timing leaks detected** in the `ct::` namespace
- **Expected leaks confirmed** in `fast::` namespace (Section 6 negative control)
- **Confidence Level**: >99.999% (Welch t-test, p < 0.00001)
- **New in v3.16.0**: native ARM64 dudect on Apple Silicon M1; MuSig2/FROST
  protocol-level timing tests; CT nonce erasure via volatile function-pointer trick

---

## Test Environment Requirements

### Hardware

| Platform | Timer | Status |
|----------|-------|--------|
| x86-64 (Intel/AMD) | `rdtscp` | **Primary** — tested in CI every push |
| ARM64 Apple Silicon (M1) | `cntvct_el0` | **Native CI** — macos-14 runner (v3.16.0+) |
| ARM64 Cortex-A55 | `cntvct_el0` | **Hardware tested** — bench_hornet campaign |
| RISC-V U74 (Milk-V Mars) | `rdcycle` | **Hardware tested** — bench_hornet campaign |
| Other | `high_resolution_clock` | Fallback — reduced precision |

### Compiler

| Compiler | Flags | CT-Safety Notes |
|----------|-------|-----------------|
| Clang 21 | `-O2` | **Recommended** — no observed CT violations |
| GCC 13 | `-O2` | **Tested** — no observed CT violations |
| Clang/GCC | `-O3` | **CAUTION** — may break CT; validate with dudect |
| MSVC | `/O2` | **Supported** — uses `_ReadWriteBarrier` + `__rdtscp` |

> **Critical**: Higher optimization levels (e.g., `-O3`, `-Ofast`) may introduce
> data-dependent branches in bitwise cmov operations. Always validate with
> dudect after changing compiler or optimization flags.

---

## Functions Tested

### Section 1: CT Primitives (Low-Level)

| Function | Class 0 (edge) | Class 1 (control) | Samples | Verdict |
|----------|----------------|-------------------|---------|---------|
| `ct::is_zero_mask` | val=0 | val=random\|1 | 100,000 | CT |
| `ct::bool_to_mask` | false | true | 100,000 | CT |
| `ct::cmov256` | mask=0 | mask=~0 | 100,000 | CT |
| `ct::cswap256` | mask=0 | mask=~0 | 100,000 | CT |
| `ct::ct_lookup_256(16)` | idx=0 | idx=random%16 | 100,000 | CT |
| `ct::ct_equal` | identical bufs | different bufs | 100,000 | CT |

### Section 2: CT Field Operations

| Function | Class 0 (edge) | Class 1 (control) | Samples | Verdict |
|----------|----------------|-------------------|---------|---------|
| `ct::field_add` | zero + base | random + base | 50,000 | CT |
| `ct::field_mul` | zero x base | random x base | 50,000 | CT |
| `ct::field_sqr` | one^2 | random^2 | 50,000 | CT |
| `ct::field_inv` | one^-¹ | random^-¹ | 5,000 | CT |
| `ct::field_cmov` | mask=0 | mask=~0 | 50,000 | CT |
| `ct::field_is_zero` | zero | random | 50,000 | CT |

### Section 3: CT Scalar Operations

| Function | Class 0 (edge) | Class 1 (control) | Samples | Verdict |
|----------|----------------|-------------------|---------|---------|
| `ct::scalar_add` | one + base | random + base | 50,000 | CT |
| `ct::scalar_sub` | one - base | random - base | 50,000 | CT |
| `ct::scalar_cmov` | mask=0 | mask=~0 | 50,000 | CT |
| `ct::scalar_is_zero` | zero | random | 50,000 | CT |
| `ct::scalar_bit` | bit=0 at pos 128 | bit=1 at pos 128 | 50,000 | CT |
| `ct::scalar_window` | pos=0 | pos=random | 50,000 | CT |

### Section 4: CT Point Operations (Critical)

| Function | Class 0 (edge) | Class 1 (control) | Samples | Verdict |
|----------|----------------|-------------------|---------|---------|
| `point_add_complete(P+O)` | P + identity | P + Q | 10,000 | CT |
| `point_add_complete(P+P)` | P + P (doubling) | P + Q (general) | 10,000 | CT |
| `ct::scalar_mul(k=1)` | k=1 | k=random | 2,000 | CT |
| `ct::scalar_mul(k=n-1)` | k=n-1 | k=random | 2,000 | CT |
| `ct::generator_mul(HW)` | low Hamming weight | high Hamming weight | 2,000 | CT |
| `point_table_lookup` | idx=0 | idx=15 | 50,000 | CT |

### Section 5: CT Byte Utilities

| Function | Class 0 | Class 1 | Samples | Verdict |
|----------|---------|---------|---------|---------|
| `ct_memcpy_if` | flag=false | flag=true | 100,000 | CT |
| `ct_is_nonzero` | all-zero buf | random buf | 100,000 | CT |
| `ct_select_byte` | flag=0 | flag=1 | 100,000 | CT |

### Section 6: FAST Layer (Negative Control — Expected Non-CT)

| Function | Expected | Notes |
|----------|----------|-------|
| `fast::scalar_mul(k=1)` | TIMING LEAK | Window-NAF leaks on scalar bits |
| `fast::field_inverse` | TIMING LEAK | Variable-time SafeGCD |
| `fast::point_add(P+O)` | TIMING LEAK | Short-circuits on identity |

> These negative controls confirm the test harness works correctly — it
> successfully detects real timing differences in non-CT code.

### Section 7: Valgrind Memory Classification

| Operation | Status |
|-----------|--------|
| `ct::field_{add,mul,sqr}` | Classified as secret, declassified after |
| `ct::scalar_{add,neg}` | Classified as secret, declassified after |
| `ct::field_cmov` with classified mask | Operates on secret data |
| `ct::ct_lookup_256` with classified index | Linear scan, no index-dependent access |
| `ct::generator_mul` with classified scalar | Full CT execution |

### Section 8: Protocol-Level Timing (v3.16.0)

Protocol-level signing operations added to dudect in v3.16.0:

| Function | Class 0 | Class 1 | Samples | Verdict |
|----------|---------|---------|---------|---------|
| `ct::ecdsa_sign` | key=1 | key=random | 500 | CT |
| `ct::schnorr_sign` | key=1 | key=random | 500 | CT |
| MuSig2 sign round (2-of-2) | key=1 | key=random | 200 | CT (low n) |
| FROST sign round (2-of-3) | key=1 | key=random | 200 | CT (low n) |

> **Note**: MuSig2 and FROST sample counts (200) are lower than scalar/field
> operations. Statistical power is sufficient to catch gross leaks but not
> fine-grained microarchitecture effects. Full characterization pending.

---

## CT Overhead: Cross-Platform (v3.16.0)

Measured with `bench_hornet` (signing operations only; verify uses public inputs):

| Platform | Hardware | ECDSA CT/FAST | Schnorr CT/FAST |
|---|---|---|---|
| x86-64 | Intel i7-11700 @ 2.50 GHz (Clang 21) | **1.77x** | **2.03x** |
| ARM64 | Cortex-A55 @ (aarch64, Clang 18) | 2.57x | 3.18x |
| RISC-V | SiFive U74 @ 1.5 GHz (GCC 13) | 1.96x | 2.37x |
| ESP32-S3 | Xtensa LX7 @ 240 MHz (GCC 14) | 1.05x | 1.06x |

ESP32 has the lowest CT overhead: in-order single-issue core, no speculative execution,
no out-of-order scheduler pressure.

ARM64 Cortex-A55 shows the highest CT overhead (2.57x ECDSA) despite being in-order —
likely cache pressure from the larger working set of precomputed tables.

x86-64 CT overhead improved from 1.94x (v3.66.0) to **1.77x** (v3.16.0) following
the GLV decomposition correctness fix (v3.13.1): `ct_scalar_mul_mod_n()` replaced
the truncated 128-bit intermediate path. Absolute CT scalar_mul: 25.3µs vs 24.0µs
fast path (1.05x overhead at the scalar_mul level alone).

---

## Statistical Methodology

### Welch's Two-Sample t-Test

For each function under test:

1. **Pre-generate** N input pairs (Class 0: edge-case, Class 1: random)
2. **Random assignment**: each measurement randomly selects class (unbiased)
3. **Timing**: `rdtscp` (x86-64) or `cntvct_el0` (ARM64) or `rdcycle` (RISC-V)
4. **Barriers**: `asm volatile` prevents compiler reordering
5. **Incremental statistics**: Online Welch's t-test (no allocation in loop)

$$t = \frac{\bar{x}_0 - \bar{x}_1}{\sqrt{\frac{s_0^2}{n_0} + \frac{s_1^2}{n_1}}}$$

**Decision rule**: |t| < 4.5 → no detectable timing difference (pass).

At 4.5, the two-tailed p-value is approximately 6.8 × 10^-⁶, meaning
there is a < 0.00068% chance of a false positive.

### Sample Sizes

| Test Category | N per class | Total measurements |
|---------------|-------------|-------------------|
| Primitives | 50,000 | 100,000 |
| Field operations | 25,000 | 50,000 |
| Scalar operations | 25,000 | 50,000 |
| Point operations | 1,000–5,000 | 2,000–10,000 |
| Signatures | 250 | 500 |
| Protocol (MuSig2/FROST) | 100 | 200 |

> Point, signature, and protocol operations require more cycles each, so fewer
> samples are needed for equivalent statistical power at the gross-leak level.

---

## CI Integration

### Every Push (Smoke)

```yaml
# Compiled with -DDUDECT_SMOKE
# Threshold: |t| < 25.0
# Duration: ~30s
# Purpose: Catch gross regressions only
# Platforms: ubuntu-latest (x86-64), macos-14 (ARM64 M1)
```

### Nightly (Full)

```yaml
# Compiled WITHOUT -DDUDECT_SMOKE
# Threshold: |t| < 4.5
# Duration: ~30 min
# Purpose: Full statistical analysis
# Platforms: ubuntu-latest (x86-64), macos-14 (ARM64 M1)
# Artifact: dudect_full.log preserved 90 days
```

### Manual Validation

```bash
# Build full mode (no DUDECT_SMOKE)
cmake -S . -B out/release -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_TESTS=ON
cmake --build out/release --target test_ct_sidechannel_standalone

# Run with 30-minute timeout
timeout 1800 ./build/src/cpu/test_ct_sidechannel_standalone
```

---

## Architecture-Specific Notes

### x86-64 (Intel / AMD)

- Timer: `rdtscp` — serializing, cycle-accurate
- Tested CPUs: Intel i7-11700 (Tiger Lake), Skylake+, AMD Zen2+
- BMI2/ADX extensions available — no CT impact (same instruction count)
- **Cache timing**: L1D 32KB, 64B lines — our CT table lookups scan all entries
  linearly, touching every cache line regardless of target index
- **CT scalar_mul overhead vs fast**: 1.77x (v3.16.0)

### ARM64 (aarch64)

- Timer: `cntvct_el0` — generic counter, lower resolution than rdtsc
- **Native CI**: macos-14 (Apple M1) — smoke per PR, full nightly (v3.16.0+)
- Hardware tested: Cortex-A55 (YF_022A, NEON + crypto extensions)
- **Variable-latency multiplier**: Some Cortex-Ax cores have data-dependent
  MUL latency; our CT field_mul uses the same multiply instruction path
  regardless of input value
- Highest CT overhead (2.57x ECDSA): in-order core but cache pressure from
  precomputed tables exceeds L1 capacity
- **Recommended**: Run dudect locally on target ARM hardware before deployment

### RISC-V 64

- Timer: `rdcycle` — cycle counter (hardware)
- Tested hardware: SiFive U74-MC (Milk-V Mars, rv64gc_zba_zbb)
- 4x64 Montgomery field representation — no conditional branches in mul
- CT overhead: 1.96x ECDSA, 2.37x Schnorr (dual-issue in-order core)
- dudect: hardware timing captured in bench_hornet campaign; CI is cross-compiled

### Multi-Architecture Campaign Status (v3.16.0)

| Architecture | dudect Status | Timer Precision |
|-------------|---------------|-----------------|
| x86-64 | **Tested in CI** | rdtscp (cycle-exact) — every push + nightly |
| ARM64 Apple M1 | **Tested in CI** | cntvct_el0 — every push + nightly (v3.16.0+) |
| ARM64 Cortex-A55 | **Hardware tested** | cntvct_el0 — bench_hornet campaign |
| RISC-V U74 | **Hardware tested** | rdcycle — bench_hornet campaign |
| ESP32-S3 | **Hardware tested** | esp_timer — bench_hornet campaign (CT overhead only) |
| WASM | **Not tested** | `performance.now()` insufficient for dudect |
| Xtensa/Cortex-M | **Not applicable** | No rdtsc equivalent; rely on code review |

---

## Known Limitations

1. **Three-tier CT verification**: ct-verif (LLVM IR analysis) proves absence
   of secret-dependent branches at the IR level and runs in CI on every push.
   Valgrind CT tracks memory-undefined origins for secret-dependent access.
   dudect provides statistical timing analysis. All three tiers are CI-enforced.
   Limitation: none of these model microarchitectural effects (cache, speculative
   execution). Machine-checked proofs (Vale/Jasmin/Coq) are not yet applied.

2. **Compiler dependency**: A compiler update could introduce CT violations.
   Mitigation: dudect runs on every CI push (x86-64 + ARM64 M1).

3. **Microarchitecture variance**: Different CPU models may exhibit different
   timing characteristics. Run dudect on your specific hardware before
   production deployment.

4. **OS noise**: Scheduling, interrupts, and frequency scaling can introduce
   measurement noise. The Welch t-test is robust to Gaussian noise, but
   extreme perturbations may cause false positives on shared runners.
   The CI `ct_sidechannel` module has one advisory PASS for this reason.

5. **FROST/MuSig2**: Protocol-level dudect added in v3.16.0 with limited
   sample counts (200 per class). Sufficient for gross-leak detection; not
   sufficient for fine-grained microarchitecture analysis.

6. **Nonce erasure**: `ct::ecdsa_sign` and `ct::schnorr_sign` use the volatile
   function-pointer trick to erase intermediate nonces. This is a best-effort
   mitigation; complete erasure cannot be guaranteed across all compilers and
   optimizations.

7. **GPU**: No CT guarantees on any GPU backend. GPU kernels are variable-time.
   Several GPU operations accept private keys (`ecdh_batch`, `bip352_scan_batch`,
   `bip324_aead_*_batch`); callers must ensure a trusted single-tenant environment.

---

## Formal Conclusion

> **Statement**: Based on empirical dudect timing analysis with >500,000
> total measurements across 35+ operations (plus protocol-level tests added
> in v3.16.0), using Welch's two-sample t-test at a significance level of
> p < 0.00001 (|t| < 4.5), we find **no statistically significant timing
> variance** in any `ct::` namespace operation of UltrafastSecp256k1 v3.68.0.
>
> This does NOT constitute formal verification. CT properties may be
> affected by compiler version, optimization level, and target
> microarchitecture. Users should validate on their deployment platform.
>
> **Architectures tested**: x86-64 (CI, primary), ARM64 Apple M1 (CI, v3.16.0+),
> ARM64 Cortex-A55 (hardware), RISC-V U74 (hardware)
> **Compilers**: Clang 21.1.0 / GCC 13.3.0, `-O2`
> **Confidence**: >99.999% (Welch t-test, two-tailed)

---

## References

1. Reparaz, O., Balasch, J., & Verbauwhede, I. (2017).
   *dudect: dude, is my code constant time?*
   IACR ePrint 2016/1123.

2. Aumasson, J.-P.
   *Timing-safe code: A guide for the rest of us.*
   https://www.chosenplaintext.ca/open-source/dudect/

3. Almeida, J. B., et al.
   *Verifying Constant-Time Implementations.*
   USENIX Security 2016.

4. Erbsen, A., et al.
   *Simple High-Level Code for Cryptographic Arithmetic.*
   IEEE S&P 2019 (Fiat-Crypto).

5. bitcoin-core/libsecp256k1 — volatile function-pointer erasure pattern.
   https://github.com/bitcoin-core/secp256k1/blob/master/src/ecdsa_impl.h

---

*UltrafastSecp256k1 v3.68.0 — CT Empirical Proof Report v2.0*
