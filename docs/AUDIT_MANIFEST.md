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
`ci/audit_gate.py`. If a principle cannot be checked automatically, it is
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
python3 ci/mutation_kill_rate.py \
  --build-dir build_rel \
  --ctest-mode \
  --count 20 \
  --threshold 75 \
  --json -o out/reports/mutation_kill_report.json
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

### P19 — Reproducible Evidence Bundle

> Independent reviewers must be able to verify repository claims from
> immutable, hash-pinned evidence rather than maintainer narrative.

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

Local enforcement: install with `python3 ci/install_caas_hooks.py` (pre-push hook)

---

### P21 — CAAS Completeness

> Every known review gap is closed by an automated CAAS gate or by a published
> CAAS-pinned document.
> The set of such gaps and gates is enumerated in
> `docs/CAAS_GAP_CLOSURE_ROADMAP.md`.
> Independent review becomes methodology replay and novel-hypothesis testing,
> not rediscovery of known bug classes.

**Automated gate:** `audit_gate.py --external-audit-replacement` (included in `ALL_CHECKS`)

**Semantic requirement map (Bastion B1, 2026-06-13).** The gate is no longer a
presence-only file list. It loads `docs/CAAS_BASTION_REQUIREMENTS.json` — a
machine-readable map binding each known review gap (P21-CORE, G-1..G-10, G-9b) to
its `artifact_paths`, enforcing `gate`, `status`, `residual_risk`, and
`last_verified`. A closed gap may no longer point only to prose: it must name a
gate that actually exists.

Checks (per requirement row):
- **Artifacts present** — every path in `artifact_paths` must exist (includes the
  15 CAAS auditor-replacement documents under the `P21-CORE` row).
- **Gate callable** — a `gated` row must name either an `audit_gate.py` sub-check
  registered in `CHECK_MAP`, or a standalone `ci/*.py` gate script that exists.
  A named gate that is not registered blocks.
- **Residual discipline** — a `documented_residual` row must carry a non-empty
  `residual_risk` and `docs/RESIDUAL_RISK_REGISTER.md` must exist; a
  `presence_only` row must explain (via `residual_risk`) why no behavioral gate
  applies.
- **Verification freshness** — `last_verified` must be a valid date within the
  embedded SLA (warn ≥180d, fail ≥540d).
- Failure: a missing/invalid map, a missing artifact, an unregistered gate, a
  stale binding, or an empty documented residual blocks the pipeline.

**Negative proof:** `ci/test_audit_scripts.py::check_p21_semantic_requirement_map`
asserts the gate fails closed on missing artifact, unregistered gate, stale
`last_verified`, empty residual, and missing map, and passes on the real map.

**Status:** 12 requirement rows (6 gated, 5 presence, 1 documented-residual), all
bound, as of 2026-06-13. See `docs/CAAS_BASTION_REQUIREMENTS.json`,
`docs/CAAS_GAP_CLOSURE_ROADMAP.md`, and `docs/CAAS_HARDENING_TODO.md`.

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
- Evidence bundle verification failures (P19) — blocking for release evidence

---

## 4. Running the Audit Gate

```bash
# Full audit gate (all principles)
python3 ci/audit_gate.py

# Individual checks
python3 ci/audit_gate.py --failure-matrix
python3 ci/audit_gate.py --abi-negative-tests
python3 ci/audit_gate.py --invalid-inputs
python3 ci/audit_gate.py --stateful-sequences
python3 ci/audit_gate.py --secret-paths
python3 ci/audit_gate.py --abi-completeness
python3 ci/audit_gate.py --test-coverage
python3 ci/audit_gate.py --audit-test-quality
python3 ci/audit_gate.py --security-patterns
python3 ci/audit_gate.py --ct-integrity
python3 ci/audit_gate.py --narrative
python3 ci/audit_gate.py --freshness
python3 ci/audit_gate.py --gpu-parity
python3 ci/audit_gate.py --test-docs
python3 ci/audit_gate.py --routing
python3 ci/audit_gate.py --doc-pairing
python3 ci/audit_gate.py --mutation-kill
python3 ci/audit_gate.py --mutation-freshness
python3 ci/audit_gate.py --crash-risks
python3 ci/audit_gap_report.py
python3 ci/audit_gap_report.py --strict

# Security Autonomy gates (P12–P18, standalone scripts)
python3 ci/check_formal_invariants.py --json
python3 ci/risk_surface_coverage.py --json
python3 ci/audit_sla_check.py --json
python3 ci/supply_chain_gate.py --json
python3 ci/evidence_governance.py validate --json
python3 ci/check_misuse_resistance.py --json
python3 ci/perf_security_cogate.py --json
python3 ci/external_audit_bundle.py
python3 ci/verify_external_audit_bundle.py --json
python3 ci/verify_external_audit_bundle.py --replay-commands --json

# Master orchestrator (runs all P12–P18 gates)
python3 ci/security_autonomy_check.py

# JSON output for CI
python3 ci/audit_gate.py --json

# Generate report file
python3 ci/audit_gate.py --json -o audit_gate_report.json
```

---

## 5. When to Run

| Trigger | Required checks |
|---------|----------------|
| Before every commit | `caas_runner.py --skip-bundle` (or pre-push hook) |
| Every push / PR (automated) | Block-based CAAS gate via `.github/workflows/gate.yml` |
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
| Before independent evidence review | `external_audit_bundle.py` + `verify_external_audit_bundle.py --replay-commands --json` |
| Periodic security autonomy check | `security_autonomy_check.py` |
| Before release | Full gate + `export_assurance.py` + `validate_assurance.py` + `security_autonomy_check.py` |

---

## 6. Extending the Manifest

To add a new audit principle:

1. Define the principle (P*N*) with a clear invariant statement
2. Implement the check in `ci/audit_gate.py`
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
| `EXTERNAL_AUDIT_BUNDLE_SPEC.md` | Hash-pinned evidence bundle format and verification rules |
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
| 2026-04-14 | Added reproducible evidence bundle producer/validator (`external_audit_bundle.py`, `verify_external_audit_bundle.py`) and spec doc | P19 reproducibility principle added; evidence bundles can be independently hash-verified and replay-validated |
| 2026-04-14 | Added CAAS infrastructure: `caas_runner.py`, `install_caas_hooks.py`, `.github/workflows/caas.yml`; added CAAS stages to `preflight.yml` | P20 added — all five audit stages now run automatically on every push and PR; pre-push hook available for local enforcement |
| 2026-04-28 | Registered P21 External-Audit Replacement Completeness; created `docs/CAAS_HARDENING_TODO.md` (H-1..H-12 all closed); `check_external_audit_replacement` gate in ALL_CHECKS | P21 added — all 15 CAAS auditor-replacement documents verified present; CAAS hardening backlog formally documented and closed |
| 2026-06-13 | Upgraded P21 from presence-only to a **semantic requirement map** (Bastion B1): `docs/CAAS_BASTION_REQUIREMENTS.json` binds each known review gap (P21-CORE, G-1..G-10, G-9b) to its artifact(s), enforcing gate, status, residual risk, and last_verified date. `check_external_audit_replacement` now verifies artifacts exist, every `gated` row names a gate registered in `CHECK_MAP` (or a standalone gate script that exists), every `documented_residual` carries a non-empty residual, and `last_verified` is within SLA. Negative fixture `P21:semantic_requirement_map` proves fail-closed on missing artifact / unregistered gate / stale date / empty residual / missing map | P21 became semantic — a closed gap may no longer point only to prose; it must name a live gate or declare an explicit residual |
| 2026-06-13 | Added **G-13 External Integration Evidence Freshness** (Bastion B13): `check_integration_evidence` gate (`--integration-evidence`, in `ALL_CHECKS`) reads `docs/INTEGRATION_EVIDENCE_STATUS.json` and reports each libbitcoin / Bitcoin Core / libsecp-shim surface as pass/stale/missing/owner_gated with a `days_until_block` runway; blocking surfaces fail on missing/stale evidence, owner_gated full-chain surfaces are explicit and never counted as current. Negative fixture `B13:integration_evidence` | Integration evidence became freshness-gated and machine-tracked rather than a static replay index |
| 2026-06-13 | Added **G-14 CT Evidence Freshness** (Bastion B14): `check_ct_evidence_status` gate (`--ct-evidence-status`, in `ALL_CHECKS`) reads `docs/CT_EVIDENCE_STATUS.json` and reports each CT-sensitive surface as pass/stale/missing/inconclusive/owner_gated. The committed-evidence + freshness dimension gates on every push (cheap); the tool-verdict dimension (ct-verif / valgrind-ct / dudect via `--verdict-dir`) keeps a single PASS + SKIP inconclusive, never a pass, and fails a blocking surface whose required_tools are not all PASS. Negative fixture `B14:ct_evidence_status` | CT claims became freshness-gated, tool-bound artifact evidence rather than table/doc confidence |
| 2026-06-13 | Added **G-15 Fuzz Campaign Evidence** (Bastion B15): `check_fuzz_campaign_status` gate (`--fuzz-campaign-status`, in `ALL_CHECKS`) reads `docs/FUZZ_CAMPAIGN_STATUS.json` and reports each fuzz surface as pass/stale/missing/crash_unconverted/owner_gated. Evidence-status gate only (no campaigns run on push): a blocking surface fails on missing corpus, stale last_verified, or a crash artifact without a matching regression; an unconverted crash blocks regardless of severity; owner_gated heavy/host-only surfaces are explicit and never current. Negative fixture `B15:fuzz_campaign_status` | Fuzz evidence became freshness-gated with enforced crash->regression conversion |
| 2026-06-13 | Added **G-16 GPU/Hardware Evidence** (Bastion B16): `check_gpu_hardware_evidence` gate (`--gpu-hardware-evidence`, in `ALL_CHECKS`) reads `docs/GPU_HARDWARE_EVIDENCE_STATUS.json` and reports each backend/surface as pass/stale/missing/owner_gated/documented_residual. Real-device CUDA/OpenCL/Metal/ROCm evidence is owner_gated (no GitHub GPU runners; never current on push); documented_residual rows must resolve to a RESIDUAL_RISK_REGISTER.md id; a fallback_correctness row must name a fallback_path and is never native-performance evidence (a performance claim naming a fallback_path fails). Negative fixture `B16:gpu_hardware_evidence` | GPU/hardware claims became explicit, honest, freshness-gated (real-device vs host-fallback vs owner-gated residual) |
| 2026-06-13 | Added **G-17 Benchmark Target Context** (Bastion B17, closes RR-BAS-03): `ci/check_bench_target_context.py` (folded into `check_bench_doc_consistency.py --json`, co-gated by `perf_security_cogate.py`) validates each canonical benchmark artifact against `docs/BENCH_TARGET_CONTEXT_SCHEMA.json`: a missing/invalid `target_context`, a timed artifact without `claim_scope`/`security_gate_dependency`, a `gpu_public_data` benchmark presented as native GPU-hardware performance, or a `bitcoin_core`/`libbitcoin` claim lacking an integration-evidence reference all fail. Negative fixture `B17:bench_target_context` | Performance evidence must declare its target context; a microbenchmark can no longer be mistaken for node throughput, nor a GPU/fallback benchmark for native-engine speed |
| 2026-06-13 | Added **G-18 Research Signal Taxonomy** (Bastion B18, closes RR-BAS-04): `ci/check_research_signal_matrix.py` (`audit_gate.py --research-signal-matrix`, in `ALL_CHECKS`) validates `docs/RESEARCH_SIGNAL_MATRIX.json` — every in-scope signal class carries an enum `attack_class` (16-value `attack_class_enum`) and routes to an `expected_evidence` surface + a resolvable `expected_gate`; covered/original_analysis classes need existing evidence and a gate that resolves (audit_gate CHECK_MAP flag or `ci/*.py` script), candidates need a `missing_evidence_action`, out_of_scope need a rationale. `research_monitor.py` now renders attack_class/affected_primitive/affected_surface/expected_gate and puts `missing_evidence_action` first in the patch plan. Mappings adversarially verified (5-agent read-only workflow). Negative fixture `B18:research_signal_matrix` | Research intake became routed audit work: each signal names its attack class and the gate that should catch it, instead of a bare covered/candidate label |
| 2026-06-13 | Added **G-19 Evidence Refresh Coverage** (Bastion B19, RR-BAS-01/02 ready for owner promotion): `ci/check_evidence_refresh_coverage.py` (`audit_gate.py --evidence-refresh-coverage`, in `ALL_CHECKS`) validates the `freshness_artifacts` ledger in `docs/AUDIT_SLA.json` — every freshness artifact tracked by `audit_sla_check.py` must declare a refresh disposition: `auto` (its committed path is cross-checked against the named workflow's actual `git add` list) or `residual` (its id resolves to `docs/RESIDUAL_RISK_REGISTER.md`). A blocking artifact with neither a verifiable producer nor a resolvable residual fails; an `auto` entry whose path the lane does not stage fails; a tracked artifact missing a disposition (or a phantom entry) fails. The `caas-evidence-refresh.yml` lane runs the gate as a fail-fast self-verify step. Dispositions adversarially verified honest (2-panel read-only workflow). Negative fixture `B19:evidence_refresh_coverage` (incl. the RR-BAS-02 stale-drill block self-test) | Freshness evidence is no longer trusted by hand: every blocking freshness SLO is provably backed by an auto-producer or a documented residual, and the incident-drill promotion path is proven safe |
| 2026-06-13 | Added **G-20 Package Provenance Binding** (Bastion B20): `ci/check_package_provenance_binding.py` (`audit_gate.py --package-provenance-binding`, in `ALL_CHECKS`) validates `docs/PACKAGE_PROVENANCE_STATUS.json` — every package surface must declare the full binding contract (artifact / producer_workflow / source_commit / source_branch / artifact_sha256 / caas_bundle_sha256 / audit_gate_verdict / workflow_run_id / status / severity). `template` surfaces hold recognized sentinels with null hash+run_id (no fake current evidence); `bound` surfaces must match HEAD + the committed `EXTERNAL_AUDIT_BUNDLE.sha256` digest + a real 64-hex artifact hash + verdict==pass + a run id; `owner_gated` release artifacts are never current in the dev tree (a real hash/run id = release-marked-current fail). Reuses the existing SLSA / supply-chain infra (`generate_slsa_provenance.py`, `verify_slsa_provenance.py`, `slsa-provenance.yml`, `supply_chain_gate.py`). Surfaces adversarially verified (2-panel read-only workflow corrected `packaging.yml` template→owner_gated once its tag-publish job was confirmed). Negative fixture `B20:package_provenance_binding`. Provenance binding only — NOT release authorization | A built package can no longer call itself "audited" unless it is cryptographically bound to the exact audited commit, the committed CAAS bundle, the audit_gate verdict, and its own hash; release artifacts can never masquerade as current dev evidence |
