# Performance Regression Tracking

How UltrafastSecp256k1 detects, manages, and prevents performance regressions.

---

## Overview

Performance regressions in a cryptographic library directly impact users. A 2×
slowdown in `scalar_mul` makes every ECDSA sign/verify operation 2× slower.

We use automated CI benchmarks + manual verification to maintain performance
baselines across commits.

---

## Automated Tracking (CI)

### Benchmark Dashboard Workflow

The `.github/workflows/benchmark.yml` workflow:

1. Runs on **every push** to `dev` and `main`
2. Builds in `Release` mode with optimizations
3. Executes `bench_comprehensive` (all operations)
4. Parses output into JSON via `.github/scripts/parse_benchmark.py`
5. Pushes to [GitHub Pages dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench-v2/)
6. Compares against historical baseline

### Alert Thresholds

| Threshold | Action |
|-----------|--------|
| **150%** (50% slower) | Comment on commit / PR |
| **200%** (2× slower) | Manual investigation required |
| **300%** (3× slower) | Likely a bug — revert candidate |

Alert notifications appear as:
- GitHub commit comments
- PR review comments
- Dashboard red markers

### Tracked Operations

All operations in `bench_comprehensive` are tracked. Key metrics:

| Operation | Baseline (ns) | Max Acceptable |
|-----------|--------------|----------------|
| `field_mul` | 17 | 26 |
| `field_square` | 16 | 24 |
| `field_inverse` | 5,000 | 7,500 |
| `scalar_mul` | 25,000 | 37,500 |
| `generator_mul` | 5,000 | 7,500 |
| `ecdsa_sign` | 30,000 | 45,000 |
| `ecdsa_verify` | 55,000 | 82,500 |
| `schnorr_sign` | 28,000 | 42,000 |
| `batch_inv_1000` | 92 | 138 |

Baselines are from x86-64, Clang 21, AVX2, Ubuntu 24.04 GH Actions runner.

---

## Regression Investigation Process

### Step 1: Identify

Regression detected by CI or manual report:

```
⚠️ Performance Alert: scalar_mul is 165% of baseline (41 μs vs 25 μs baseline)
```

### Step 2: Reproduce Locally

```bash
# Build benchmark
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_BENCH=ON
cmake --build build --target bench_comprehensive

# Run on current commit
./build/cpu/bench_comprehensive | grep scalar_mul

# Compare with parent commit
git stash
cmake --build build --target bench_comprehensive
./build/cpu/bench_comprehensive | grep scalar_mul
git stash pop
```

### Step 3: Root Cause

Common causes:

| Cause | Detection | Fix |
|-------|-----------|-----|
| Accidental copy | Profile shows memcpy | Remove copy, use reference |
| Missed inlining | Assembly shows call instead of inline | Add `inline` / `__attribute__((always_inline))` |
| Cache miss | perf shows high L2/L3 miss rate | Restructure data layout |
| Branch misprediction | perf shows branch-miss > 5% | Use branchless algorithm |
| Compiler regression | Only on specific compiler version | Pin compiler or workaround |
| Algorithm change | Intentional but slower | Document tradeoff, get approval |

### Step 4: Fix & Verify

```bash
# After fix, verify regression is resolved
./build/cpu/bench_comprehensive | grep scalar_mul
# Expected: back to baseline ±5%
```

### Step 5: Document

Every performance-affecting change must include in the commit message:

```
perf: <description>

Before: scalar_mul 41 μs
After:  scalar_mul 25 μs
Cause:  <root cause>
```

---

## Manual Verification

For major releases, manual verification is performed:

1. **Dedicated hardware** — isolated machine, no background processes
2. **CPU pinning** — `taskset -c 0` on Linux
3. **Turbo disabled** — fixed CPU frequency
4. **Multiple runs** — 5× with median
5. **Cross-architecture** — x86-64 + ARM64 minimum

Results are published in `docs/BENCHMARKS.md` with hardware details.

---

## Baseline Management

### Resetting Baseline

The benchmark dashboard stores results in the `gh-pages` branch under
`dev/bench-v2/`. A baseline reset is needed when:

1. **New harness** — `DoNotOptimize` added (v2 reset)
2. **Algorithm change** — intentional performance change
3. **New CI runner hardware** — different GH Actions machine type

Reset process:

```bash
# In benchmark.yml, change the data dir:
benchmark-data-dir-path: 'dev/bench-v3'  # bump version
```

### Adding New Metrics

1. Add operation to `bench_comprehensive.cpp`
2. Follow existing output format: `[name]  value unit`
3. Update `parse_benchmark.py` if format differs
4. CI will auto-discover the new metric

---

## Prevention Strategies

### Code Review Checklist

Before merging performance-sensitive code:

- [ ] No new allocations in hot path
- [ ] No hidden copies (pass by reference)
- [ ] No unnecessary branch additions
- [ ] Benchmark run locally on at least one platform
- [ ] If algorithm changed: documented justification + new baseline

### CI Guardrails

| Guardrail | Location |
|-----------|----------|
| Benchmark on every push | `.github/workflows/benchmark.yml` |
| 150% alert threshold | benchmark.yml `alert-threshold` |
| PR comment on regression | benchmark-action auto-comment |
| Dashboard visualization | GitHub Pages (historical chart) |
| Nightly extended bench | `.github/workflows/nightly.yml` |

---

## See Also

- [docs/BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md) — How benchmarks are run
- [docs/BENCHMARKS.md](BENCHMARKS.md) — Full benchmark results
- [docs/PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) — Tuning for speed
