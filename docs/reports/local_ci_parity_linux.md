# Local Docker CI Parity -- Linux

**Date:** 2026-03-03
**Purpose:** Map GitHub Actions CI jobs to local Docker CI coverage for pre-merge validation.

---

## 1. Two Local CI Systems

The project has two Docker-based local CI setups:

| System | Entry Point | Image | Key Strength |
|--------|-------------|-------|-------------|
| `Dockerfile.local-ci` + `scripts/local-ci.sh` | `bash scripts/local-ci.sh --all` | `ubuntu:24.04` (pinned digest) | Richer (cppcheck, valgrind-ct, dudect, bench, HTML reports) |
| `docker/Dockerfile.ci` + `docker-compose.ci.yml` | `docker compose -f docker-compose.ci.yml run <service>` | `ubuntu:24.04` (pinned digest) | Broader (ARM64, WASM, Emscripten) |

**Recommendation:** Merge into a single system (using `docker/Dockerfile.ci` as the base, adding missing jobs from `local-ci.sh`).

---

## 2. GitHub Actions -> Local Docker Parity Matrix

### Blocking CI Jobs (must pass before merge)

| GH Job | `run_ci.sh` | `local-ci.sh` | Status | Criticality |
|--------|-------------|---------------|--------|-------------|
| linux (gcc-13/Release) | `linux-gcc` | `ci` | COVERED | Blocking |
| linux (gcc-13/Debug) | `linux-debug` | `ci` | COVERED | Blocking |
| linux (clang-17/Release) | `linux-clang` | `ci` | COVERED | Blocking |
| linux (clang-17/Debug) | implicit in `all` | `ci` | COVERED | Blocking |
| ASan+UBSan | `asan` | `asan` | COVERED | Blocking |
| TSan | `tsan` | `tsan` | COVERED | Blocking |
| Build with -Werror | `warnings` | `werror` | COVERED | Blocking |
| Valgrind Memcheck | `valgrind` | `valgrind` | COVERED | Blocking |
| ct-verif LLVM analysis | -- | -- | **GAP** | Blocking |
| Benchmark Regression | -- | (bench) | **PARTIAL** | Blocking |
| CodeQL | -- | -- | N/A (GH only) | Blocking |

### Informational CI Jobs (run but don't block)

| GH Job | `run_ci.sh` | `local-ci.sh` | Status |
|--------|-------------|---------------|--------|
| linux-arm64 (cross) | `arm64` | -- | Partial |
| WASM (Emscripten) | `wasm` | -- | Partial |
| Coverage (LLVM) | `coverage` | `coverage` | COVERED |
| clang-tidy | `clang-tidy` | `clang-tidy` | COVERED |
| cppcheck | -- | `cppcheck` | Partial |
| Audit Report | `audit` | `audit` | COVERED |
| dudect | -- | `dudect` | Partial |
| valgrind-ct | -- | `valgrind-ct` | Partial |

### Platform-Specific (cannot run locally on Linux)

| GH Job | Notes |
|--------|-------|
| windows (MSVC) | Requires native Windows |
| macos (AppleClang + Metal) | Requires macOS + Apple Silicon |
| iOS (OS + Simulator) | Requires Xcode |
| rocm (HIP compile) | Requires ROCm container + AMD GPU |
| android (NDK, 3 ABIs) | Requires Android NDK (could be added to Docker) |

### GitHub-Only Services (no local equivalent possible)

CodeQL, Scorecard, dependency-review, ClusterFuzzLite, mutation, nightly, release, packaging, docs, bindings, Discord webhooks.

---

## 3. Coverage Gaps (Action Required)

### Gap 1: CT-Verif (BLOCKING)
- **GH Job:** `ct-verif.yml` -- LLVM-pass based CT analysis
- **Local:** Neither Docker system covers this
- **Fix:** Add `ct-verif` job to `docker/run_ci.sh`
  - Requires: clang-17, llvm-17, llvm-17-dev (already in image)
  - Replicates: compile CT modules to IR, run ct-verif LLVM pass
  - Fallback: Manual IR analysis for switch/variable_gep patterns

### Gap 2: Benchmark Regression (BLOCKING)
- **GH Job:** `bench-regression.yml` -- compares against stored baseline
- **Local:** `local-ci.sh` runs bench but without baseline comparison
- **Fix:** Add local baseline storage (`~/.cache/ufsecp-bench-baseline.json`) and comparison script

### Gap 3: Android NDK (INFORMATIONAL)
- **GH Job:** `ci.yml` android matrix (3 ABIs)
- **Local:** Neither Docker system covers this
- **Fix:** Could add NDK cross-compilation to `docker/Dockerfile.ci` (low priority)

---

## 4. Job Criticality Tiers (Local Runs)

### Tier 1 -- Must Pass Before Push (pre-push gate)
```
scripts/local-ci.sh --quick
```
Equivalent to: gcc-13/Release + clang-17/Release + basic tests

### Tier 2 -- Must Pass Before PR (pre-merge gate)
```
scripts/local-ci.sh --all
```
Equivalent to: full compiler matrix + ASan + TSan + Valgrind + clang-tidy + audit

### Tier 3 -- Run Before Release
```
docker compose -f docker-compose.ci.yml run all
```
Equivalent to: everything above + ARM64 cross + WASM + coverage

---

## 5. Artifact Expectations

| Job | Expected Output | Location |
|-----|----------------|----------|
| coverage | `lcov.info` + HTML report | `local-ci-output/coverage/` |
| valgrind | `memcheck.*.log` | `local-ci-output/valgrind/` |
| audit | `audit_report.json` + `.txt` | `local-ci-output/audit/` |
| clang-tidy | `clang-tidy-report.txt` | `local-ci-output/clang-tidy/` |
| bench | `benchmark.json` | `local-ci-output/bench/` |

---

## 6. Fail Policy (Local -> Merge Blocking)

| Tier | Failure Blocks Merge? | Rationale |
|------|----------------------|-----------|
| Tier 1 (quick) | YES | Basic correctness |
| Tier 2 (all) | YES | Full security + quality gate |
| Tier 3 (release) | YES for release tags | Full cross-platform coverage |
| dudect (timing) | NO | Statistical, unreliable on shared HW |
| cppcheck | NO | Informational (false positive rate) |
| bench (local, no baseline) | NO | No baseline for comparison |

---

## 7. Local-vs-GH Differences Checklist

| Factor | GH Actions | Local Docker | Risk |
|--------|-----------|-------------|------|
| Compiler versions | Pinned (gcc-13, clang-17) | Pinned (same) | Low |
| OS base | `ubuntu-24.04` | `ubuntu:24.04@sha256:...` | Low |
| CPU | Shared runner (variable) | Developer machine | Medium (perf variance) |
| Memory | 7 GB (standard) | Developer-dependent | Low |
| Parallel jobs | `$(nproc)` | `$(nproc)` | Low |
| tmpfs | No | No (could add) | Low |
| Network | Yes (download deps) | Yes (build time) | Low |
| Build cache | None | ccache (in image) | Low (ccache = faster, same result) |

---

## 8. Recommended Next Steps

1. **Merge two Docker CI systems** into one (`docker/Dockerfile.ci` as base + `local-ci.sh` feature additions)
2. **Add ct-verif** job to local runner
3. **Add local baseline comparison** for benchmark regression
4. **Pin resource profile** (CPU/RAM limits in compose) for reproducible bench results
5. **Document container image update policy** (when to rebuild, digest pinning lifecycle)
