# CI Gating Policy — 3-Tier Architecture

## Overview

| Tier | Name | Trigger | Target Time | Gate Type |
|------|------|---------|-------------|-----------|
| **Tier 0** | Local Fast Loop | `pre-commit` / manual | < 30 seconds | Advisory |
| **Tier 1** | PR Gate | Every push to `dev` | < 5 minutes | **Blocking** |
| **Tier 2** | Deep Assurance | Nightly / weekly / manual | 30–120 minutes | Advisory (escalates to blocking for releases) |

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

## Tier 1 — PR Gate (per-push to dev)

**Purpose:** Deterministic, fast, blocking gate. Every push must pass this.

**Budget:** < 5 minutes total wall time.

**What runs:**

| Check | Blocking? | Notes |
|-------|-----------|-------|
| CMake configure + Ninja build | Yes | Must compile cleanly |
| `ctest -L unit` | Yes | All unit tests |
| `ctest -L audit` (deterministic only) | Yes | Exploit PoC + regression tests |
| `audit_gate.py --json` (P0 + P1 checks) | Yes | ABI completeness, failure-class matrix, hostile-caller |
| `validate_assurance.py --json` | Yes | Doc–code consistency |
| `preflight.py --security --abi` | Yes | Invariant checks |
| SARIF upload (if GitHub) | — | Non-blocking upload |

**What does NOT run:**
- Dudect (statistical, flaky)
- Long fuzz campaigns
- GPU-specific tests (need hardware)
- Cross-library comparison benchmarks
- Cachegrind / Valgrind annotation
- RISC-V / ARM / ESP32 cross-compilation

## Tier 2 — Deep Assurance

**Purpose:** Heavy-duty verification including statistical tests, long fuzzing,
cross-platform, and performance regression detection.

**Schedule:** Nightly on `dev`, weekly full run, or manually triggered.

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
- **Nightly:** Advisory. Report stored, diff computed, alerts on regressions.
- **Pre-release:** Blocking. All Tier 2 checks must pass before any release.

## Impact-Based Gating

Not all code changes require the same scrutiny. The CI system should
automatically select the appropriate gate severity based on what changed.

### Hard gate (Tier 1 + mandatory Tier 2 subset)

Triggered when changes touch:

| Path pattern | Reason |
|---|---|
| `cpu/src/ct_*.cpp` | CT-sensitive code |
| `cpu/src/ecdsa.cpp` | Nonce generation |
| `cpu/src/schnorr.cpp` | Schnorr signing |
| `cpu/include/secp256k1/detail/secure_erase.hpp` | Zeroization |
| `cpu/include/secp256k1/ct/*.hpp` | CT primitives |
| `cpu/src/field_*.cpp` | Field arithmetic |
| `cpu/src/scalar_*.cpp` | Scalar arithmetic |
| `opencl/kernels/secp256k1_field.cl` | GPU field reduction |
| `cuda/include/secp256k1.cuh` | GPU field reduction |
| `metal/shaders/secp256k1_field.h` | GPU field reduction |
| `include/ufsecp/ufsecp.h` | ABI surface |
| `include/ufsecp/ufsecp_impl.cpp` | ABI dispatch |
| `include/ufsecp/ufsecp_gpu.h` | GPU ABI |
| `include/ufsecp/ufsecp_gpu_impl.cpp` | GPU dispatch |

**Additional checks triggered:**
- Full dudect run (not just short)
- Sanitizer sweep
- Cross-platform KAT (if applicable)

### Light gate (Tier 1 only)

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
# Detect which paths changed since the merge base
CHANGED=$(git diff --name-only $(git merge-base HEAD origin/dev) HEAD)

# Check if any hard-gate path is touched
HARD_GATE=false
for pattern in "cpu/src/ct_" "cpu/src/ecdsa" "cpu/src/schnorr" \
               "secure_erase" "cpu/include/secp256k1/ct/" \
               "cpu/src/field_" "cpu/src/scalar_" \
               "opencl/kernels/" "cuda/include/" "metal/shaders/" \
               "include/ufsecp/"; do
    if echo "$CHANGED" | grep -q "$pattern"; then
        HARD_GATE=true
        break
    fi
done

if [ "$HARD_GATE" = true ]; then
    echo "HARD GATE: CT/crypto/ABI change detected"
    # Run Tier 1 + Tier 2 subset
else
    echo "LIGHT GATE: docs/bindings/tooling change only"
    # Run Tier 1 only
fi
```

### CI workflow integration

The impact detection is integrated into the CI workflow via a job that runs
first and sets output variables consumed by downstream jobs:

```yaml
jobs:
  detect-impact:
    runs-on: ubuntu-latest
    outputs:
      gate-level: ${{ steps.detect.outputs.gate }}
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - id: detect
        run: |
          # ... impact detection logic ...
          echo "gate=hard" >> "$GITHUB_OUTPUT"  # or "light"

  tier1:
    needs: detect-impact
    # Always runs

  tier2-crypto:
    needs: [detect-impact, tier1]
    if: needs.detect-impact.outputs.gate-level == 'hard'
    # Runs only when crypto/CT/ABI changes detected
```

## Report integration

All tier outputs produce reports in the unified schema v1.0.0 with provenance.
Reports are:
1. Stored as CI artifacts
2. Ingested into `artifact_analyzer.py` database
3. Available for `diff`, `divergence`, and `flakes` analysis
4. SARIF exports uploaded to GitHub Code Scanning

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
