# Auditor Quickstart

> **3 commands, 3 artifacts — everything a security reviewer needs.**

## Prerequisites

- Linux (Ubuntu 22.04+ recommended) or Docker
- CMake 3.20+, Ninja, GCC 12+ or Clang 15+
- Python 3.10+

## Option A: Docker (recommended)

```bash
bash ci/auditor_kit.sh
```

Output appears in `./audit_output/`:

| Artifact | Format | Purpose |
|----------|--------|---------|
| `audit_results.xml` | JUnit XML | Test results for CI/tooling |
| `audit_assurance.json` | JSON | Machine-readable assurance report |
| `audit_run.log` | Text | Full test log |

## Option B: Host build (3 commands)

```bash
# 1. Build (canonical audit profile — out/audit)
python3 ci/configure_build.py audit   # or: cmake -B out/audit -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C out/audit

# 2. Run audit tests
ctest --test-dir out/audit -L audit --output-on-failure --timeout 300

# 3. Generate reports
python3 ci/audit_gate.py --json -o audit_gate_report.json
python3 ci/export_assurance.py -o assurance_report.json
python3 ci/auditor_mode.py --sarif --json
```

Output:

| Artifact | Format | Purpose |
|----------|--------|---------|
| `audit_gate_report.json` | JSON (schema v1.0.0) | Unified audit gate: provenance + findings + verdicts |
| `assurance_report.json` | JSON | Feature/API coverage + test inventory |
| `build/auditor_mode/auditor_mode_report.sarif` | SARIF 2.1.0 | GitHub Code Scanning / external tooling |

## Report Schema

All JSON reports follow the **unified report schema v1.0.0**:

```json
{
  "schema_version": "1.0.0",
  "run_id": "audit_gate-0a93ff4b12-20260409T...",
  "runner": "audit_gate",
  "generated_at": "2026-04-09T14:30:00+00:00",
  "commit": {
    "value": { "sha": "0a93ff4b...", "dirty": false, "ref": "dev" },
    "status": "available"
  },
  "platform": "Linux",
  "provenance": { "...toolchain, build_flags, submodules..." },
  "verdict": "PASS | PASS with advisory | FAIL",
  "summary": { "blocking": 0, "advisory": 2, "skipped_sections": 0 },
  "sections": [
    {
      "name": "ABI Completeness",
      "verdict": "PASS",
      "findings": [
        {
          "check_id": "P1_abi",
          "severity": "pass",
          "severity_display": "PASS",
          "title": "All 181 ABI functions accounted for"
        }
      ]
    }
  ]
}
```

### Key policies

| Policy | Rule |
|--------|------|
| **No "unknown" strings** | Unavailable data uses `null` + `status` + `reason` |
| **SKIP(reason)** | Checks that can't run (platform/feature) are marked with structured skip |
| **Advisory vs Blocking** | Severity prefix: `blocking:critical`, `advisory:medium`, etc. |
| **Verdicts** | `PASS`, `PASS with advisory`, `PASS with skips`, `FAIL` |

## Deeper analysis

```bash
# Regression diff (before/after)
python3 ci/artifact_analyzer.py diff --before old_report.json --after new_report.json

# Platform divergence
python3 ci/artifact_analyzer.py divergence linux_report.json arm_report.json riscv_report.json

# Bug capsule → regression test
python3 ci/bug_capsule_gen.py --list schemas/bug_capsules/

# Security autonomy gate orchestrator
python3 ci/security_autonomy_check.py --json

# Build and verify an external-audit reproducibility bundle
python3 ci/external_audit_bundle.py
python3 ci/verify_external_audit_bundle.py --json
python3 ci/verify_external_audit_bundle.py --replay-commands --json
```

## Further reading

- [AUDIT_GUIDE.md](AUDIT_GUIDE.md) — Full audit methodology and scope
- [AUDIT_SCOPE.md](AUDIT_SCOPE.md) — Attack surface enumeration
- [AUDIT_MANIFEST.md](AUDIT_MANIFEST.md) — 19 audit principles
- [TEST_MATRIX.md](TEST_MATRIX.md) — Complete test inventory
- [FEATURE_ASSURANCE_LEDGER.md](FEATURE_ASSURANCE_LEDGER.md) — ABI function coverage
- [LAYER_ROUTING_MATRIX.md](LAYER_ROUTING_MATRIX.md) — CT/FAST layer routing rationale
- [SECURITY_AUTONOMY_PLAN.md](SECURITY_AUTONOMY_PLAN.md) — Security autonomy framework & gates
- [EXTERNAL_AUDIT_BUNDLE_SPEC.md](EXTERNAL_AUDIT_BUNDLE_SPEC.md) — Hash-pinned external-audit bundle format and verification
