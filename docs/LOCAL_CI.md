# Local CI (GitHub Parity)

This guide is the practical local gate for Linux developers.

Goal: if local parity passes, GitHub Linux CI should not fail for the same reasons.

## 1. One-command entrypoint

Use the wrapper:

```bash
./scripts/ci-local.sh --list
./scripts/ci-local.sh quick
./scripts/ci-local.sh pre-push
./scripts/ci-local.sh gh-parity
```

`ci-local.sh` auto-detects `docker compose` or `docker-compose`.

## 2. Recommended flow

Fast iteration:

```bash
./scripts/ci-local.sh quick
```

Before push:

```bash
./scripts/ci-local.sh pre-push
```

Before PR/merge (max Linux parity):

```bash
./scripts/ci-local.sh gh-parity
```

## 3. What `gh-parity` runs

`gh-parity` runs Linux jobs mapped to GitHub blockers/advisory checks:

- warnings (`-Werror`)
- linux-gcc (Release + Debug)
- linux-clang (Release + Debug)
- asan
- tsan
- msan (advisory, non-blocking to match workflow behavior)
- valgrind
- clang-tidy
- cppcheck
- ct-verif (deterministic CT LLVM/IR checks)
- bench-regression (local baseline compare)
- audit (unified_audit_runner on GCC + Clang)
- arm64 (cross-compile)
- wasm (build + KAT)

## 4. Benchmark regression baseline

`bench-regression` stores local baseline in:

```text
.ci-baseline/bench_quick_baseline.json
```

Behavior:
- first run creates baseline and passes
- next runs compare current quick bench against baseline
- default threshold is `120` percent (20 percent slower fails)

Override threshold:

```bash
BENCH_ALERT_THRESHOLD=130 ./scripts/ci-local.sh bench-regression
```

## 5. GPU Audit (Local Only)

The GPU audit cannot run on GitHub CI (no GPU runners). It runs **locally only** on any machine with an NVIDIA GPU and CUDA toolkit.

### Prerequisites

- NVIDIA GPU (any compute capability >= 5.0)
- CUDA Toolkit >= 12.0
- CMake >= 3.24, Ninja

### Build

```bash
# From library root:
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="native"   # or e.g. "86;89;90"
ninja -C build-cuda gpu_audit_runner
```

### Run

```bash
./build-cuda/cuda/gpu_audit_runner
```

### Expected Output

The runner executes **43 modules** across **10 sections** and produces:

| Section | Modules | Coverage |
|---------|---------|----------|
| Mathematical Invariants | 12 | Field, scalar, point arithmetic, group order |
| Signature Operations | 3 | ECDSA + Schnorr roundtrip, wrong-key rejection |
| Batch Operations | 4 | Batch inversion, bloom filter, batch ECDSA verify, MSM |
| CPU-GPU Differential | 1 | Generator mul cross-check |
| Device Memory | 2 | Alloc/free stress, CUDA error state |
| Constant-Time Layer | 6 | CT field/scalar/point, CT ECDSA/Schnorr, CT-FAST parity |
| Standard Test Vectors | 3 | BIP-340, RFC-6979, BIP-32 |
| Protocol Security | 6 | Multi-key ECDSA/Schnorr, ECDH, recovery, BIP-32 chain, Hash160 |
| Fuzzing | 4 | Edge scalars, zero-key rejection, serialization roundtrip |
| Performance Smoke | 2 | ECDSA 100-iter stress, Schnorr 50-iter stress |

Verdict: **AUDIT-READY** when all 43/43 pass.

Reports are written to the build directory:
- `gpu_audit_report.json` -- machine-readable
- `gpu_audit_report.txt` -- human-readable summary

### Notes

- First run may take ~5 minutes due to PTX JIT compilation (subsequent runs are faster)
- The `selftest_core` module runs 41+ GPU kernel tests and dominates total runtime
- `CMAKE_CUDA_ARCHITECTURES="native"` auto-detects your GPU; explicit SM values avoid JIT overhead

## 6. What is still not reproducible on Linux local Docker

These GitHub jobs need non-Linux or hosted integrations:

- windows (MSVC)
- macOS/iOS (Apple toolchain and Metal runtime)
- CodeQL / Scorecard / dependency-review / other GitHub-native services

Use local parity for fast prevention, and keep GitHub CI as final cross-platform confirmation.
