# Audit Manifest — UltrafastSecp256k1

> **This document defines the mandatory audit principles, invariants, and
> automated gates that every change to UltrafastSecp256k1 must satisfy.**
>
> Version: 2.2 — 2026-04-14

---

## 1. Purpose

This manifest exists so that audit quality is **systematic and reproducible**,
not dependent on any single person remembering to run the right checks.

Every principle below maps to at least one automated check in
`scripts/audit_gate.py`. If a principle cannot be checked automatically, it is
documented with a manual verification procedure.

---

## 2. Core Audit Principles

### P0 — Failure-Class Matrix

> Known failure classes must stay machine-reportable. Covered, partial, and
> deferred states may not live only in narrative prose.

**Automated gate:** `audit_gate.py --failure-matrix`

Checks:
- The self-audit failure-class matrix parses cleanly into executable rows
- Partial and deferred rows retain explicit residual-risk notes
- Owner-grade residual failure classes are surfaced as blocking findings

### P0a — ABI Hostile-Caller Manifest

> Every exported ABI surface must keep the hostile-caller quartet visible:
> success smoke, NULL rejection, zero-edge handling, and invalid-content rejection.

**Automated gate:** `audit_gate.py --abi-negative-tests`

Checks:
- The hostile-caller manifest covers the full exported ABI surface
- Any missing quartet dimension on an exported ABI symbol is a blocking finding

### P0b — Secret-Path Change Gate

> Secret-bearing or CT-evidence surfaces may not change silently. Their paired
> evidence docs must change in the same work unit.

**Automated gate:** `audit_gate.py --secret-paths`

Checks:
- Secret-lifecycle and CT-evidence path changes are detected from the working tree
- Missing paired updates to `SECRET_LIFECYCLE.md` or `SECURITY_CLAIMS.md` block the gate

### P0c — Invalid-Input Grammar

> Structured hostile inputs must be rejected without crashes, silent accepts,
> or parser confusion.

**Automated gate:** `audit_gate.py --invalid-inputs`

Checks:
- Invalid public keys, secret keys, signatures, ECDH peers, and BIP-32 inputs reject deterministically
- The harness writes a machine-readable report and fails closed on unexpected accepts

### P0d — Stateful Sequence Integrity

> The library must remain correct across mixed call sequences, injected errors,
> context recreation, and repeated reuse.

**Automated gate:** `audit_gate.py --stateful-sequences`

Checks:
- Valid operations keep working after adjacent failure paths
- Multi-step BIP-32 derivation remains equivalent to path derivation
- Context reuse and recreate sequences do not corrupt behavior

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
- Direct audit/test caller evidence from the graph is accepted as supplemental mapping
- Export assurance `test_coverage` field is non-empty for all CPU ABI functions

### P2a — Audit Test Quality

> Audit code must not simulate assurance with vacuous checks, polarity bugs,
> ignored return values, or thresholds too weak to catch regressions.

**Automated gate:** `audit_gate.py --audit-test-quality`

Checks:
- Critical and high-severity audit-test-quality findings block
- Medium findings warn
- Low findings remain visible so audit hygiene backlog cannot disappear into prose

### P2b — Mutation Kill Rate

> High-value arithmetic and audit surfaces need a heavier lane that measures
> whether the test suite actually kills wrong behavior rather than only executing code.

**Automated gate:** `audit_gate.py --mutation-kill`

Checks:
- Mutation testing runs in explicit heavy mode against the configured build dir
- Kill rate must meet the configured threshold or the selected run fails

Scope note:

This is intentionally a heavier owner-grade / batch-review lane, not the default
per-commit path inside the full gate.

**Strict execution policy (cost control):**

- Heavy mutation kill-rate runs are **local pre-release requirements**.
- Heavy mutation runs are **not mandatory push/PR GitHub-hosted CI checks**.
- Canonical local command before release:

```bash
python3 scripts/mutation_kill_rate.py \
  --build-dir build_rel \
  --ctest-mode \
  --count 20 \
  --threshold 75 \
  --json -o mutation_kill_report.json
```

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

### P11 — Failure-Class Gap Visibility

> The self-audit failure-class matrix must remain structurally executable.
> Covered, partial, and deferred classes must be machine-reportable rather than
> buried in prose.

**Automated gate:** `audit_gate.py --failure-matrix` and `audit_gap_report.py --strict`

Checks:
- Every failure-class row has deterministic audit surfaces
- Partial/deferred rows keep explicit residual-risk notes
- File-like evidence references resolve inside the repository
- Strict mode exposes the current owner-grade residual set

### P12 — Formal Invariant Completeness

> Critical cryptographic operations must have machine-readable formal invariant
> specs with linked tests. Operations without specs or linked tests are blocking.

**Automated gate:** `check_formal_invariants.py --json`

Checks:
- Every operation in `FORMAL_INVARIANTS_SPEC.json` has preconditions, postconditions, and linked tests
- CT-flagged operations have CT-specific test links
- Missing specs or unlinked tests are FAIL

### P13 — Risk-Surface Coverage

> Seven risk classes (ct_paths, parser_boundary, abi_boundary, gpu_parity,
> secret_lifecycle, determinism, fuzz_corpus) must meet minimum coverage thresholds.

**Automated gate:** `risk_surface_coverage.py --json`

Checks:
- Each risk class meets its configured minimum coverage threshold
- Below-threshold classes are FAIL findings

### P14 — Audit SLA Compliance

> Evidence freshness and audit SLA/SLO targets must be met.

**Automated gate:** `audit_sla_check.py --json`

Checks:
- Evidence staleness within configured limits
- Critical evidence freshness within SLA targets
- Golden reference files exist and are current

### P15 — Supply-Chain Fail-Closed

> Build reproducibility, dependency pinning, provenance, artifact hashing,
> and hardening checks must all pass.

**Automated gate:** `supply_chain_gate.py --json`

Checks:
- Dependency pinning is present
- Build reproducibility script exists and is functional
- SLSA provenance verification passes
- Artifact hash policy is enforced
- Build hardening checks pass

### P16 — Evidence Governance

> An HMAC-verified tamper-resistant evidence chain must validate cleanly.

**Automated gate:** `evidence_governance.py validate --json`

Checks:
- Evidence chain integrity (HMAC verification)
- No tampered or missing records

### P17 — Misuse Resistance

> Every exported ABI function must have ≥3 negative/hostile-caller tests.

**Automated gate:** `check_misuse_resistance.py --json`

Checks:
- Per-ABI-function negative test count meets threshold
- Below-threshold functions are FAIL findings

### P18 — Performance-Security Co-Gating

> Performance changes must not regress security posture. CT evidence, formal
> invariants, secret lifecycle, and GPU parity stubs are checked.

**Automated gate:** `perf_security_cogate.py --json`

Checks:
- CT evidence exists and is current
- Formal invariant spec is complete
- Secret lifecycle doc is present
- No undocumented GPU parity stubs

### P19 — External Auditor Reproducibility Bundle

> External auditors must be able to verify repository claims from immutable,
> hash-pinned evidence rather than maintainer narrative.

**Automated gate:** `external_audit_bundle.py` + `verify_external_audit_bundle.py`

Checks:
- Critical gate outputs are captured with command-output hashes
- Required evidence files are present and SHA-256 pinned in bundle
- Detached bundle digest matches bundle JSON
- Independent verifier can validate the bundle (and optionally replay commands)

### P20 — Continuous Audit as a Service (CAAS)

> Audit gates must run automatically on every push and pull request.
> Any regression must be caught before it reaches the repository, not after.
> Manual audit runs are a supplement — not a substitute — for continuous automation.

**Automated gate:** `caas_runner.py` (local) + `.github/workflows/caas.yml` (CI)

Checks:
- Stage 1: `audit_test_quality_scanner` — 0 findings required (any finding is an immediate CI failure)
- Stage 2: `audit_gate.py` — all P0–P18 principles must pass
- Stage 3: `security_autonomy_check.py` — 100/100 autonomy score required
- Stage 4: `external_audit_bundle.py` — evidence bundle regenerated and pinned
- Stage 5: `verify_external_audit_bundle.py` — integrity check on freshly produced bundle

All five stages are **blocking** — a PR cannot be merged if any stage fails.

Local enforcement: install with `python3 scripts/install_caas_hooks.py` (pre-push hook)

---

### P21 — External-Audit Replacement Completeness

> Every gap that an external audit engagement would normally close is closed
> by an automated CAAS gate or by a published CAAS-pinned document.
> The set of such gaps and gates is enumerated in
> `docs/CAAS_GAP_CLOSURE_ROADMAP.md`.
> External audit becomes a verification of methodology, not a bug-hunt.

**Automated gate:** `audit_gate.py --external-audit-replacement` (included in `ALL_CHECKS`)

Checks:
- All 15 CAAS auditor-replacement documents present (CAAS_PROTOCOL.md, RESIDUAL_RISK_REGISTER.md,
  SECURITY_CLAIMS.md, EXPLOIT_TEST_CATALOG.md, EXTERNAL_AUDIT_BUNDLE.{json,sha256},
  CAAS_REVIEWER_QUICKSTART.md, CAAS_FAQ.md, CAAS_THREAT_MODEL.md,
  NEGATIVE_RESULTS_LEDGER.md, THREAD_SAFETY.md, ABI_VERSIONING.md,
  SECURITY_INCIDENT_TIMELINE.md, AUDIT_PHILOSOPHY.md,
  `scripts/verify_external_audit_bundle.py`)
- Failure: any required document absent blocks the pipeline.

**Status:** All 15 required documents present as of 2026-04-28.
See `docs/CAAS_GAP_CLOSURE_ROADMAP.md` and `docs/CAAS_HARDENING_TODO.md`.

---

## 3. Severity Levels

| Level | Meaning | Gate behavior |
|-------|---------|---------------|
| **FAIL** | Blocking finding — must be fixed before merge | Exit code 1 |
| **WARN** | Non-blocking but should be addressed soon | Exit code 0, reported |
| **INFO** | Informational, no action required | Logged only |

### What blocks a merge:

- Any FAIL from P0, P0a–P0d, P1–P9, P11, P12–P18
- Security pattern loss (P3)
- ABI surface mismatch (P1)
- CT routing violation (P4)
- Critical/high audit-test-quality findings (P2a)
- Formal invariant gaps (P12)
- Risk-surface coverage below threshold (P13)
- Supply-chain integrity failure (P15)
- Misuse-resistance below threshold (P17)

### What doesn't block but must be tracked:

- Documentation gaps (P5, P8, P10) — tracked in audit report
- Medium/low audit-test-quality findings (P2a) — tracked until hardened
- Graph freshness warnings (P6) — rebuild resolves
- GPU parity stubs with proper TODO comments (P7)
- Mutation kill-rate failures block only when the heavy lane is explicitly selected (P2b)
- Audit SLA warnings (P14) — tracked until evidence refreshed
- Evidence governance warnings (P16) — tracked until chain repaired
- Perf-security co-gating warnings (P18) — informational unless regression detected
- External-audit bundle verification failures (P19) — blocking for external audit sign-off

---

## 4. Running the Audit Gate

```bash
# Full audit gate (all principles)
python3 scripts/audit_gate.py

# Individual checks
python3 scripts/audit_gate.py --failure-matrix
python3 scripts/audit_gate.py --abi-negative-tests
python3 scripts/audit_gate.py --invalid-inputs
python3 scripts/audit_gate.py --stateful-sequences
python3 scripts/audit_gate.py --secret-paths
python3 scripts/audit_gate.py --abi-completeness
python3 scripts/audit_gate.py --test-coverage
python3 scripts/audit_gate.py --audit-test-quality
python3 scripts/audit_gate.py --security-patterns
python3 scripts/audit_gate.py --ct-integrity
python3 scripts/audit_gate.py --narrative
python3 scripts/audit_gate.py --freshness
python3 scripts/audit_gate.py --gpu-parity
python3 scripts/audit_gate.py --test-docs
python3 scripts/audit_gate.py --routing
python3 scripts/audit_gate.py --doc-pairing
python3 scripts/audit_gate.py --mutation-kill
python3 scripts/audit_gate.py --mutation-freshness
python3 scripts/audit_gate.py --crash-risks
python3 scripts/audit_gap_report.py
python3 scripts/audit_gap_report.py --strict

# Security Autonomy gates (P12–P18, standalone scripts)
python3 scripts/check_formal_invariants.py --json
python3 scripts/risk_surface_coverage.py --json
python3 scripts/audit_sla_check.py --json
python3 scripts/supply_chain_gate.py --json
python3 scripts/evidence_governance.py validate --json
python3 scripts/check_misuse_resistance.py --json
python3 scripts/perf_security_cogate.py --json
python3 scripts/external_audit_bundle.py
python3 scripts/verify_external_audit_bundle.py --json
python3 scripts/verify_external_audit_bundle.py --replay-commands --json

# Master orchestrator (runs all P12–P18 gates)
python3 scripts/security_autonomy_check.py

# JSON output for CI
python3 scripts/audit_gate.py --json

# Generate report file
python3 scripts/audit_gate.py --json -o audit_gate_report.json
```

---

## 5. When to Run

| Trigger | Required checks |
|---------|----------------|
| Before every commit | `caas_runner.py --skip-bundle` (or pre-push hook) |
| Every push / PR (automated) | Full CAAS pipeline via `.github/workflows/caas.yml` |
| During owner-grade assurance review | `audit_gap_report.py` + `audit_gap_report.py --strict` |
| After parser / ABI hostile-input changes | `--abi-negative-tests --invalid-inputs --audit-test-quality` |
| After protocol / lifecycle changes | `--stateful-sequences --audit-test-quality` |
| After adding/removing ABI functions | `--abi-completeness` + rebuild graph |
| After touching CT layer | `--ct-integrity --security-patterns` |
| After GPU backend changes | `--gpu-parity` |
| After adding tests | `--test-coverage --test-docs` |
| After high-risk arithmetic or audit-harness changes | `--mutation-kill` |
| After changing formal invariants or CT specs | `check_formal_invariants.py --json` |
| After changing risk surfaces or fuzz corpus | `risk_surface_coverage.py --json` |
| After supply-chain or dependency changes | `supply_chain_gate.py --json` |
| After evidence chain changes | `evidence_governance.py validate --json` |
| After adding/removing ABI functions (misuse) | `check_misuse_resistance.py --json` |
| Before handing repository to external auditors | `external_audit_bundle.py` + `verify_external_audit_bundle.py --replay-commands --json` |
| Periodic security autonomy check | `security_autonomy_check.py` |
| Before release | Full gate + `export_assurance.py` + `validate_assurance.py` + `security_autonomy_check.py` |

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
| `SELF_AUDIT_FAILURE_MATRIX.md` | Failure-class coverage map + residual-risk inventory |
| `OWNER_GRADE_AUDIT_TODO.md` | Concrete code-and-tooling backlog for closing owner-grade audit gaps |
| `SECURITY_CLAIMS.md` | Security guarantees and non-guarantees |
| `FFI_HOSTILE_CALLER.md` | Hostile-caller resilience analysis |
| `BACKEND_ASSURANCE_MATRIX.md` | GPU backend parity tracking |
| `SECURITY_AUTONOMY_PLAN.md` | 30-day security autonomy framework and phase plan |
| `FORMAL_INVARIANTS_SPEC.json` | Machine-readable formal invariant specifications |
| `AUDIT_SLA.json` | Measurable audit SLA/SLO definitions |
| `SECURITY_AUTONOMY_KPI.json` | Auto-generated autonomy score and gate results |
| `EXTERNAL_AUDIT_BUNDLE_SPEC.md` | Hash-pinned external auditor evidence format and verification rules |
| `.github/workflows/caas.yml` | Continuous Audit as a Service — five-stage blocking CI pipeline |

---

## 8. Automation History

| Date | Change | Impact |
|------|--------|--------|
| 2026-03-23 | Initial manifest + `audit_gate.py` | All 10 principles automated |
| 2026-03-25 | Added `audit_gap_report.py` | Failure-class matrix became executable in normal/strict modes |
| 2026-04-06 | Added invalid-input grammar, stateful sequence, and audit-test-quality checks; exposed mutation kill as an explicit heavy lane | Runtime hostile-input and audit-hygiene tooling became part of the documented assurance perimeter |
| 2026-03-23 | Fixed `export_assurance.py` test_coverage query | Was using wrong DB table |
| 2026-03-23 | Fixed graph builder missing `ufsecp_gpu.h` | 18 GPU ABI functions were invisible |
| 2026-03-23 | Fixed preflight missing `ufsecp_gpu.h` scan | ABI drift detection was incomplete |
| 2026-03-25 | Added `test_gpu_bip352_scan.cpp` (SW-BIP352-1..13) | BIP-352 Silent Payment GPU scan audit coverage |
| 2026-04-14 | Security Autonomy Program: 10 scripts, 3 spec docs, preflight steps 18-20 | P12-P18 principles added; formal invariants, SLA, supply chain, misuse resistance, evidence governance, incident drills, fuzz campaigns, perf-security co-gating; master orchestrator `security_autonomy_check.py` |
| 2026-04-14 | Added external-audit bundle producer/validator (`external_audit_bundle.py`, `verify_external_audit_bundle.py`) and spec doc | P19 external-auditor reproducibility principle added; external sign-off can be independently hash-verified and replay-validated |
| 2026-04-14 | Added CAAS infrastructure: `caas_runner.py`, `install_caas_hooks.py`, `.github/workflows/caas.yml`; added CAAS stages to `preflight.yml` | P20 added — all five audit stages now run automatically on every push and PR; pre-push hook available for local enforcement |
| 2026-04-28 | Registered P21 External-Audit Replacement Completeness; created `docs/CAAS_HARDENING_TODO.md` (H-1..H-12 all closed); `check_external_audit_replacement` gate in ALL_CHECKS | P21 added — all 15 CAAS auditor-replacement documents verified present; CAAS hardening backlog formally documented and closed |
