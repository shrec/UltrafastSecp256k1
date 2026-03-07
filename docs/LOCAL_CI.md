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

## 5. What is still not reproducible on Linux local Docker

These GitHub jobs need non-Linux or hosted integrations:

- windows (MSVC)
- macOS/iOS (Apple toolchain and Metal runtime)
- CodeQL / Scorecard / dependency-review / other GitHub-native services

Use local parity for fast prevention, and keep GitHub CI as final cross-platform confirmation.
