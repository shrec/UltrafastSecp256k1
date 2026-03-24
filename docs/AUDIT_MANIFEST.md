# Audit Manifest — UltrafastSecp256k1

> **This document defines the mandatory audit principles, invariants, and
> automated gates that every change to UltrafastSecp256k1 must satisfy.**
>
> Version: 1.0 — 2026-03-23

---

## 1. Purpose

This manifest exists so that audit quality is **systematic and reproducible**,
not dependent on any single person remembering to run the right checks.

Every principle below maps to at least one automated check in
`scripts/audit_gate.py`. If a principle cannot be checked automatically, it is
documented with a manual verification procedure.

---

## 2. Core Audit Principles

### P1 — ABI Completeness

> Every `UFSECP_API` function declared in any public header (`ufsecp.h`,
> `ufsecp_gpu.h`, `ufsecp_version.h`) must be present in the project graph DB
> and in `FEATURE_ASSURANCE_LEDGER.md`.

**Automated gate:** `audit_gate.py --abi-completeness`

Checks:
- Header-declared functions = graph `c_abi_functions` (zero diff)
- Header-declared functions ⊆ FEATURE_ASSURANCE_LEDGER (zero missing)
- No stale functions in ledger not in headers (zero extra)

### P2 — Test Coverage Mapping

> Every ABI function must map to at least one test target in
> `function_test_map`. Zero-coverage gaps are a blocking finding.

**Automated gate:** `audit_gate.py --test-coverage`

Checks:
- `v_coverage_gaps` returns empty result set
- Every `c_abi_functions` entry has ≥1 row in `function_test_map`
- Export assurance `test_coverage` field is non-empty for all CPU ABI functions

### P3 — Security Pattern Preservation

> Security-critical patterns (`secure_erase`, `value_barrier`, `CLASSIFY`,
> `DECLASSIFY`) must never decrease in count vs the graph baseline.

**Automated gate:** `audit_gate.py --security-patterns`

Checks:
- For each (file, pattern) in `security_patterns`, actual file count ≥ graph count
- Any decrease is a **FAIL** (patterns were removed)
- Any increase is **INFO** (rebuild graph to update baseline)

### P4 — CT Layer Integrity

> Functions routed through the CT layer must remain constant-time.
> No CT function may be changed without updating CT verification docs.

**Automated gate:** `audit_gate.py --ct-integrity`

Checks:
- All functions in `abi_routing` with `layer='ct'` are listed in
  `CT_VERIFICATION.md`
- Changed CT source files require matching CT doc updates (doc-code pairing)
- CT files retain all `secure_erase`/`value_barrier` calls

### P5 — Narrative Consistency

> Audit documentation must not contain stale claims that contradict the
> actual state of the codebase (e.g., claiming "no CT verification" when
> ct-verif runs in CI).

**Automated gate:** `audit_gate.py --narrative`

Checks:
- Predefined stale-phrase patterns are absent from audit docs
- Historical-exempt files (marked with "superseded by" etc.) are skipped

### P6 — Graph Freshness

> The project graph must be rebuilt whenever source files are modified.
> Stale graphs produce stale audit results.

**Automated gate:** `audit_gate.py --freshness`

Checks:
- No source file has mtime > graph build time
- No source file in graph is deleted from disk
- No new source file exists that the graph doesn't know about

### P7 — GPU Backend Parity

> Every GPU compute operation must exist on all backends (CUDA, OpenCL,
> Metal) and be exposed through the C ABI.

**Automated gate:** `audit_gate.py --gpu-parity`

Checks:
- Every `GpuBackend` virtual method has a `ufsecp_gpu_*` C ABI function
- No `GpuError::Unsupported` return without a `TODO(parity)` comment or
  `PARITY-EXCEPTION` marker
- All GPU ABI functions are in the graph

### P8 — Test Target Documentation

> All CTest targets must appear in `TEST_MATRIX.md`. Undocumented tests
> reduce audit transparency.

**Automated gate:** `audit_gate.py --test-docs`

Checks:
- Every `add_test(NAME ...)` from CMakeLists.txt is referenced in TEST_MATRIX.md
- Missing targets are reported as warnings

### P9 — ABI Routing Consistency

> ABI routing (CT vs fast) in the graph must match the actual implementation
> dispatch. Misrouted functions are a security concern.

**Automated gate:** `audit_gate.py --routing`

Checks:
- Functions declared CT in `abi_routing` call into `ct_*` implementation files
- Functions declared fast do not route through CT layer unnecessarily

### P10 — Doc-Code Pairing

> When a core source file is modified, its paired documentation files must
> also be updated in the same commit.

**Automated gate:** `audit_gate.py --doc-pairing`

Checks:
- Changed files are matched against `DOC_PAIRS` mapping in preflight
- Missing doc updates are reported as warnings

---

## 3. Severity Levels

| Level | Meaning | Gate behavior |
|-------|---------|---------------|
| **FAIL** | Blocking finding — must be fixed before merge | Exit code 1 |
| **WARN** | Non-blocking but should be addressed soon | Exit code 0, reported |
| **INFO** | Informational, no action required | Logged only |

### What blocks a merge:

- Any FAIL from P1–P9
- Security pattern loss (P3)
- ABI surface mismatch (P1)
- CT routing violation (P4)

### What doesn't block but must be tracked:

- Documentation gaps (P5, P8, P10) — tracked in audit report
- Graph freshness warnings (P6) — rebuild resolves
- GPU parity stubs with proper TODO comments (P7)

---

## 4. Running the Audit Gate

```bash
# Full audit gate (all principles)
python3 scripts/audit_gate.py

# Individual checks
python3 scripts/audit_gate.py --abi-completeness
python3 scripts/audit_gate.py --test-coverage
python3 scripts/audit_gate.py --security-patterns
python3 scripts/audit_gate.py --ct-integrity
python3 scripts/audit_gate.py --narrative
python3 scripts/audit_gate.py --freshness
python3 scripts/audit_gate.py --gpu-parity
python3 scripts/audit_gate.py --test-docs
python3 scripts/audit_gate.py --routing
python3 scripts/audit_gate.py --doc-pairing

# JSON output for CI
python3 scripts/audit_gate.py --json

# Generate report file
python3 scripts/audit_gate.py --json -o audit_gate_report.json
```

---

## 5. When to Run

| Trigger | Required checks |
|---------|----------------|
| Before every commit | `audit_gate.py` (full) |
| After adding/removing ABI functions | `--abi-completeness` + rebuild graph |
| After touching CT layer | `--ct-integrity --security-patterns` |
| After GPU backend changes | `--gpu-parity` |
| After adding tests | `--test-coverage --test-docs` |
| Before release | Full gate + `export_assurance.py` + `validate_assurance.py` |

---

## 6. Extending the Manifest

To add a new audit principle:

1. Define the principle (P*N*) with a clear invariant statement
2. Implement the check in `scripts/audit_gate.py`
3. Add the `--flag` to the CLI
4. Define severity (FAIL/WARN/INFO)
5. Add a "When to run" trigger
6. Update this manifest

---

## 7. Relationship to Other Audit Documents

| Document | Purpose |
|----------|---------|
| `AUDIT_MANIFEST.md` (this) | Principles + automation rules |
| `INTERNAL_AUDIT.md` | Detailed audit findings + coverage map |
| `FEATURE_ASSURANCE_LEDGER.md` | Per-function assurance status |
| `TEST_MATRIX.md` | Test target inventory |
| `CT_VERIFICATION.md` | Constant-time verification details |
| `SECURITY_CLAIMS.md` | Security guarantees and non-guarantees |
| `FFI_HOSTILE_CALLER.md` | Hostile-caller resilience analysis |
| `BACKEND_ASSURANCE_MATRIX.md` | GPU backend parity tracking |

---

## 8. Automation History

| Date | Change | Impact |
|------|--------|--------|
| 2026-03-23 | Initial manifest + `audit_gate.py` | All 10 principles automated |
| 2026-03-23 | Fixed `export_assurance.py` test_coverage query | Was using wrong DB table |
| 2026-03-23 | Fixed graph builder missing `ufsecp_gpu.h` | 18 GPU ABI functions were invisible |
| 2026-03-23 | Fixed preflight missing `ufsecp_gpu.h` scan | ABI drift detection was incomplete |
