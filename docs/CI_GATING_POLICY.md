# CI Gating Policy — CAAS Block Architecture

## Overview

The CI/CD system has two automatic execution modes:

1. **PR / push gate** — `.github/workflows/gate.yml`
2. **Release gate** — `.github/workflows/release.yml`

Heavy evidence workflows remain available as `workflow_dispatch` tools, but do
not fan out automatically on every push. The only scheduled workflow is
`research-monitor.yml`, which watches for new external signals and opens
actionable reports without burning the full CI matrix every night.

| Mode | Workflow | Trigger | Gate Type |
|------|----------|---------|-----------|
| Local fast loop | `ci/preflight.py` / selected scripts | Manual / hook | Advisory |
| PR / push gate | `gate.yml` | Push + PR to `main` / `dev` | Blocking |
| Release gate | `release.yml` | `v*` tag / manual | Blocking |
| Research intake | `research-monitor.yml` | Scheduled / manual | Advisory |
| Deep assurance tools | Individual workflows | Manual only | Blocking only when invoked by release policy |

## Tier 0 — Local Fast Loop

**Purpose:** Catch obvious mistakes before commit.

```bash
python3 ci/preflight.py --security --abi --drift --autonomy
```

**What runs:**
- Security invariant checking (secure_erase / value_barrier counts)
- ABI surface diff (new/removed `ufsecp_*` functions)
- Narrative drift detection
- Python audit script syntax validation
- Security autonomy gates (formal invariants, SLA, supply chain, misuse resistance, incident drills) — Phase 1: informational only

**What does NOT run:**
- Full compilation
- CTest targets
- Fuzz / dudect / sanitizers
- GPU tests

**Gate behavior:** Advisory only. Findings are displayed but never block `git commit`.

## PR / Push Gate

**Purpose:** Deterministic, impact-based, blocking gate. Every push and PR must
pass the selected blocks.

**Workflow:** `.github/workflows/gate.yml`

**Block order:**

| Block | Purpose |
|-------|---------|
| Block 0 / Detect Impact | `ci/ci_gate_detect.py` maps changed files to product profiles |
| Block 1 / Fast CAAS Gates | repo-map freshness, exploit wiring, canonical docs, audit script quick test, assurance validation |
| Block 2 / Build + Unit Tests | Release build + deterministic CTest subset, skipped for docs-only changes |
| Block 3 / Selected Security/Profile Gates | CAAS, bindings, GPU/WASM routing, compat gates selected by profile |
| Gate / Final Verdict | Single status check summarizing selected blocks |

**What does NOT run:**
- Dudect (statistical, flaky)
- Long fuzz campaigns
- GPU-specific tests (need hardware)
- Cross-library comparison benchmarks
- Cachegrind / Valgrind annotation
- RISC-V / ARM / ESP32 cross-compilation

## Deep Assurance Tools

**Purpose:** Heavy-duty verification including statistical tests, long fuzzing,
cross-platform checks, sanitizers, CT tooling, and performance regression
detection.

**Schedule:** Manual only, except when the release policy invokes the relevant
evidence in `release.yml`. Generic nightly/weekly fan-out is intentionally
disabled.

**What runs:**

| Check | Category | Approx Time |
|-------|----------|-------------|
| `ctest -L dudect` (long runs) | CT verification | 5–20 min |
| `ctest -L fuzz` (extended budget) | Fuzzing | 10–30 min |
| `ctest -L sanitizer` (ASan + UBSan) | Memory safety | 5–10 min |
| `ctest -L benchmark` | Performance | 3–5 min |
| `ci/cross_compiler_ct_stress.sh` | Cross-uarch CT | 10–30 min |
| `ci/cachegrind_ct_analysis.sh` | CT IR analysis | 5–10 min |
| GPU backend tests (CUDA / OpenCL / Metal) | GPU parity | hardware-dependent |
| `ci/artifact_analyzer.py diff` | Regression diff | < 1 min |
| `ci/mutation_kill_rate.py` | Mutation testing | 10–30 min |
| `ci/perf_regression_check.sh` | Performance gate | 3–5 min |
| Cross-library comparison | Benchmarking | 5 min |

**Gate behavior:**
- **Manual:** produces evidence artifacts for focused review.
- **Release:** release CAAS gate is blocking before build/package fan-out.
- **Research:** `research-monitor.yml` is the only scheduled automatic lane.

## Impact-Based Gating

Not all code changes require the same scrutiny. The CI system should
automatically select the appropriate gate severity based on what changed.

### Hard gate

Triggered when changes touch:

| Path pattern | Reason |
|---|---|
| `src/cpu/src/ct_*.cpp` | CT-sensitive code |
| `src/cpu/src/ecdsa.cpp` | Nonce generation |
| `src/cpu/src/schnorr.cpp` | Schnorr signing |
| `src/cpu/include/secp256k1/detail/secure_erase.hpp` | Zeroization |
| `src/cpu/include/secp256k1/ct/*.hpp` | CT primitives |
| `src/cpu/src/field_*.cpp` | Field arithmetic |
| `src/cpu/src/scalar_*.cpp` | Scalar arithmetic |
| `src/opencl/kernels/secp256k1_field.cl` | GPU field reduction |
| `src/cuda/include/secp256k1.cuh` | GPU field reduction |
| `src/metal/shaders/secp256k1_field.h` | GPU field reduction |
| `include/ufsecp/ufsecp.h` | ABI surface |
| `include/ufsecp/ufsecp_impl.cpp` | ABI dispatch |
| `include/ufsecp/ufsecp_gpu.h` | GPU ABI |
| `include/ufsecp/ufsecp_gpu_impl.cpp` | GPU dispatch |

**Additional checks triggered:** selected CAAS/security/profile blocks in
`gate.yml`; deeper tools are manual unless release policy invokes them.

### Light gate

Triggered when changes touch only:

| Path pattern | Reason |
|---|---|
| `docs/**` | Documentation only |
| `bindings/**` | Language bindings |
| `ci/**` | Tooling/automation |
| `benchmarks/**` | Performance measurement |
| `examples/**` | Example code |
| `.github/**` | CI config |
| `schemas/**` | Schema definitions |

## Implementation

### Impact detection script

```bash
python3 ci/ci_gate_detect.py --base origin/dev --github-output "$GITHUB_OUTPUT"
```

### CI workflow integration

The impact detector is integrated into `gate.yml` as the first job and sets
profile outputs consumed by downstream jobs:

```yaml
jobs:
  detect-impact:
    runs-on: ubuntu-latest
    outputs:
      gate: ${{ steps.detect.outputs.gate }}
      profiles: ${{ steps.detect.outputs.profiles }}
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - id: detect
        run: |
          # ... impact detection logic ...
          echo "gate=hard" >> "$GITHUB_OUTPUT"  # or "light"

  fast-gates:
    needs: detect-impact

  build-test:
    needs: [detect-impact, fast-gates]
    if: needs.detect-impact.outputs.docs_only != 'true'

  caas-security:
    needs: [detect-impact, fast-gates]
    if: needs.detect-impact.outputs.run_caas == 'true'

  final-verdict:
    needs: [detect-impact, fast-gates, build-test, caas-security]
    if: always()
```

## Report integration

All tier outputs produce reports in the unified schema v1.0.0 with provenance.
Reports are:
1. Stored as CI artifacts
2. Ingested into `artifact_analyzer.py` database
3. Available for `diff`, `divergence`, and `flakes` analysis
4. SARIF exports uploaded to GitHub Code Scanning

## Workflow consolidation history

The repository historically had two separate workflows that have since been
consolidated; their removal is intentional and the replacements are listed
here so reviewers do not chase deleted files.

The legacy `pipeline.yml` (a `workflow_run`-chained phased CI) was removed
in commit `a009cfaf` and replaced by `gate.yml` (impact-based block gate)
plus `ci.yml` (platform/sanitizer matrix) — both run directly on push/PR
with no `workflow_run` indirection.

The standalone `gcc-analyzer.yml` job was removed in commit `25c8aa8f`
when the static-analysis surface was consolidated. The replacement
coverage is provided by `cppcheck.yml`, `clang-sa.yml`, `clang-tidy.yml`,
`infer.yml`, `codeql.yml`, and `code-quality.yml`. GCC's `-fanalyzer` was
dropped because it duplicated Clang Static Analyzer findings without
adding new signals in our codebase.

The `check_doc_drift.py` gate enforces that no doc references the
removed names outside this exempted tombstone paragraph.

## SKIP handling

When a check cannot run due to platform or feature constraints:

```json
{
  "name": "CUDA GPU Parity",
  "verdict": "SKIP(no CUDA hardware on this runner)",
  "skipped": {
    "platform": "ubuntu-latest",
    "constraint": "no CUDA",
    "detail": "GitHub Actions runners do not have GPU hardware"
  },
  "findings": []
}
```

The overall verdict becomes `PASS with skips` rather than `PASS` or `FAIL`.
