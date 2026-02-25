# Benchmark Methodology

How UltrafastSecp256k1 benchmarks are collected, reported, and tracked for
regression detection.

---

## Principles

1. **Reproducibility** -- same code on same hardware produces same results (+-2%)
2. **Isolation** -- benchmarks run with minimal background load
3. **Statistical rigor** -- multiple iterations with median reporting
4. **Cross-platform** -- results collected on multiple architectures
5. **Automated tracking** -- CI catches regressions before merge

---

## Benchmark Framework

### `bench_comprehensive`

The primary benchmark binary:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_BENCH=ON
cmake --build build --target bench_comprehensive
./build/cpu/bench_comprehensive
```

#### Operations Measured

| Category | Operations |
|----------|-----------|
| Field | mul, square, add, sub, negate, inverse, sqrt, batch_inverse |
| Scalar | mul, add, negate, inverse |
| Point | add (Jacobian), double, scalar_mul, generator_mul |
| Signatures | ECDSA sign/verify, Schnorr sign/verify |
| Multi-scalar | Straus (N=2..64), Pippenger (N=64..4096) |
| CT | ct::scalar_mul, ct::ecdsa_sign, ct::schnorr_sign |
| Protocols | MuSig2 (2/3/5 party), FROST (2-of-3, 3-of-5), Adaptor |

#### Timing Method

```cpp
// Warmup phase: 100 iterations (discarded)
for (int i = 0; i < 100; ++i) DoNotOptimize(operation());

// Measurement phase: 1000 iterations
auto t0 = high_resolution_clock::now();
for (int i = 0; i < 1000; ++i) DoNotOptimize(operation());
auto t1 = high_resolution_clock::now();
auto ns = duration_cast<nanoseconds>(t1 - t0).count() / 1000;
```

- **`DoNotOptimize`**: Compiler barrier preventing dead-code elimination
  (equivalent to Google Benchmark's `DoNotOptimize`)
- **Clock**: `std::chrono::high_resolution_clock`
- **Iterations**: 1,000 minimum per operation (adjusted for fast ops)
- **Warmup**: 100 iterations to stabilize cache and branch predictor

### Output Format

```
[field_mul]          17 ns
[field_square]       16 ns
[scalar_mul]         25 us
[ecdsa_sign]         30 us
```

Parsed by `.github/scripts/parse_benchmark.py` into JSON for dashboard:

```json
[
  {"name": "field_mul", "unit": "ns", "value": 17},
  {"name": "scalar_mul", "unit": "ns", "value": 25000}
]
```

---

## Measurement Environment

### Hardware Requirements

For reliable results:

- **CPU frequency**: Fixed (disable turbo boost / dynamic scaling)
- **Thermal throttling**: Monitor -- abort if throttled
- **Background load**: Minimal (no browser, no IDE profiling)
- **Memory**: Sufficient to avoid swapping

### Linux (Recommended)

```bash
# Disable CPU frequency scaling
sudo cpupower frequency-set -g performance

# Disable turbo boost (Intel)
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo

# Pin to single core (reduce scheduling noise)
taskset -c 0 ./build/cpu/bench_comprehensive
```

### Windows

```powershell
# Set high-performance power plan
powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c

# Close background applications
# Use Process Lasso or similar for core affinity
```

---

## Statistical Method

### Single Run

Each operation reports the **median** of 5 measurement blocks (each block = 1000
iterations). This reduces outlier influence from OS scheduling.

### Regression Detection

The CI benchmark workflow (`benchmark.yml`) uses
[github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark):

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Alert threshold | 150% | Warns if >50% slower |
| Tool | `customSmallerIsBetter` | Lower = faster |
| Auto-push | Yes (to `gh-pages`) | Historical tracking |
| Comment-on-alert | Yes | PR notification |
| Fail-on-alert | No | Allows investigation |

### What Counts as a Regression

| Severity | Threshold | Action |
|----------|-----------|--------|
| Warning | >20% slower | Comment on PR |
| Alert | >50% slower | Block merge (investigate) |
| Critical | >100% slower (2x) | Revert immediately |

---

## Platform Matrix

Benchmarks are collected on:

| Platform | Hardware | CI |
|----------|----------|-----|
| x86-64 Linux | Ubuntu 24.04, GH Actions runner | [OK] Every push (dev/main) |
| x86-64 Windows | Windows Latest, GH Actions runner | [OK] Every push (dev/main) |
| ARM64 Linux | Cross-compile + QEMU (estimated) | [!] Nightly |
| RISC-V 64 | SiFive HiFive Unmatched (manual) | [FAIL] Manual |
| ESP32-S3 | 240 MHz LX7 (manual) | [FAIL] Manual |
| Apple Silicon | M3 Pro (manual) | [FAIL] Manual |
| CUDA | RTX 5060 Ti (manual) | [FAIL] Manual |

---

## Reference Numbers

Current baseline (x86-64, Clang 21, AVX2, Release):

| Operation | Time | Notes |
|-----------|------|-------|
| Field mul | 17 ns | ASM (mulx/adcx/adox) |
| Field square | 16 ns | ASM |
| Field inverse | 5 us | Fermat (ASM) |
| Scalar mul | 25 us | GLV + wNAF |
| Generator mul | 5 us | Precomputed table |
| Point add (Jac) | 200 ns | |
| Point double | 150 ns | |
| ECDSA sign | 30 us | CT path: 180 us |
| ECDSA verify | 55 us | |
| Schnorr sign | 28 us | CT path: 170 us |
| Schnorr verify | 50 us | |
| Batch inv (N=1000) | 92 ns/elem | Montgomery trick |

These serve as the alert baseline. Any commit causing >50% regression on a tracked
metric is flagged.

---

## Adding New Benchmarks

1. Add the benchmark to `benchmarks/bench_comprehensive.cpp`
2. Use the standard timing pattern (warmup + DoNotOptimize + median)
3. Output format: `[operation_name]  <value> <unit>`
4. Update `.github/scripts/parse_benchmark.py` if the output format changes
5. The CI dashboard auto-discovers new entries from the JSON output

---

## See Also

- [docs/BENCHMARKS.md](BENCHMARKS.md) -- Full results across all platforms
- [docs/PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) -- Tuning recommendations
- [docs/PERFORMANCE_REGRESSION.md](PERFORMANCE_REGRESSION.md) -- Regression tracking policy
