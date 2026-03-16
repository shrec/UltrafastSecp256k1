# Audit Infrastructure TODO

> **Status: ALL ITEMS COMPLETED** (2026-03-16)
> All P0, P1, and P2 items have been implemented.
> See commit `audit-infra-completion` branch for the implementation.

This workdoc tracks the remaining programmatic work needed to make the
UltrafastSecp256k1 audit infrastructure fully self-consistent, evidence-rich,
and reviewer-friendly.

Current state:
- Core audit/test coverage is strong.
- `ct-verif`, `valgrind-ct`, `dudect`, `preflight`, project graph, traceability,
  and GPU audit layers already exist.
- The main remaining gaps are evidence packaging, narrative drift control, and
  historical report hygiene.

## P0: Evidence Bundle Completion

### 1. Pull deterministic CT evidence into the audit package
- Update `audit/run_full_audit.sh`.
- Update `audit/run_full_audit.ps1`.
- Update `scripts/generate_audit_package.sh`.
- Update `scripts/generate_audit_package.ps1`.

Required outcome:
- Final audit packages must include or explicitly reference:
  - `ct-verif` results
  - `valgrind-ct` results
  - disassembly CT scan results
  - dudect smoke/full results where available
- `README.txt` inside the audit package must describe CT evidence correctly:
  - statistical timing tests
  - deterministic CT verification
  - machine-checked proofs not present

Acceptance criteria:
- A generated audit package contains deterministic CT evidence files or links.
- No generated summary claims that formal CT verification is missing if
  `ct-verif` / `valgrind-ct` results are available.

### 2. Add explicit artifact slots for CT evidence
- Standardize artifact paths in audit output:
  - `artifacts/ct/ct_verif.log`
  - `artifacts/ct/ct_verif_summary.json`
  - `artifacts/ct/valgrind_ct.log`
  - `artifacts/ct/valgrind_ct_report.json`
  - `artifacts/ct/disasm_branch_scan.json`
  - `artifacts/ct/dudect_smoke.log`
  - `artifacts/ct/dudect_full.log`

Acceptance criteria:
- Shell and PowerShell audit runners use the same artifact layout.
- Audit package generation does not need ad-hoc path guessing.

## P0: Narrative Drift Prevention

### 3. Extend doc drift detection to audit narrative files
- Update `scripts/preflight.py`.
- Update `scripts/validate_assurance.py`.
- Update `scripts/build_project_graph.py` only if needed for doc mapping.

Critical docs to cover:
- `README.md`
- `audit/AUDIT_TEST_PLAN.md`
- `audit/run_full_audit.sh`
- `audit/run_full_audit.ps1`
- `docs/AUDIT_READINESS_REPORT_v1.md`
- `docs/TEST_MATRIX.md`
- `docs/SECURITY_CLAIMS.md`
- `docs/CT_VERIFICATION.md`
- `docs/CI_ENFORCEMENT.md`

Required outcome:
- If CT status, audit count, or enforcement level changes in code/workflows,
  these docs must be checked for stale wording.
- At minimum, preflight should flag stale phrases such as:
  - "planned" for already-implemented CT layers
  - "tool integration not yet done" for integrated workflows
  - "no formal verification applied" when deterministic CT verification exists

Acceptance criteria:
- Preflight fails when these narrative docs drift from current audit reality.
- Drift checks are narrowly targeted and do not create noisy false positives.

### 4. Add terminology normalization for CT claims
- Introduce a single canonical wording model in docs:
  - `Statistical timing leakage testing`
  - `Deterministic CT verification`
  - `Machine-checked proof frameworks`

Required outcome:
- Replace ambiguous uses of "formal verification" where they currently mix:
  - dudect
  - ct-verif
  - ctgrind/valgrind
  - Coq/Jasmin/Vale/Fiat-Crypto-style proofs

Acceptance criteria:
- Same terminology appears consistently in:
  - `README.md`
  - `docs/SECURITY_CLAIMS.md`
  - `docs/CT_VERIFICATION.md`
  - `docs/TEST_MATRIX.md`
  - generated audit summaries

## P1: Historical Report Hygiene

### 5. Mark historical reports as historical
- Review files under `docs/reports/`.
- Review legacy summaries under `audit/` and old readiness reports.

Required outcome:
- Historical reports that intentionally describe old gaps must be clearly marked:
  - `Historical report`
  - `Superseded by current CI enforcement`
  - `Snapshot date`
- Active docs must not contradict current state.

Acceptance criteria:
- Reviewers can distinguish:
  - current guarantees
  - archived historical analysis

### 6. Clean stale local parity reports
- Review `docs/reports/local_ci_parity_linux.md`.

Current issue:
- It still lists `ct-verif` as a local coverage gap.

Decision required:
- Either update it to current truth, or mark it archived/superseded.

Acceptance criteria:
- No active report claims a gap that is already closed.

## P1: Audit Runner Fidelity

### 7. Teach `run_full_audit` to surface blocking/non-blocking CT status correctly
- Ensure generated summaries distinguish:
  - dudect advisory on shared runners
  - ct-verif blocking
  - valgrind-ct blocking

Required outcome:
- Audit summaries should not understate deterministic CT gates.
- Audit summaries should not overstate dudect as a proof layer.

Acceptance criteria:
- Generated reports clearly describe the CT stack:
  - disasm scan
  - ct-verif
  - valgrind-ct
  - dudect

### 8. Surface GPU self-hosted evidence in the same package model
- If `gpu-selfhosted.yml` is active, integrate its artifacts into the audit package
  or document a canonical cross-link.

Required outcome:
- CPU audit package and GPU audit evidence should feel like one system, not two
  parallel stories.

Acceptance criteria:
- Package docs mention where GPU audit artifacts live.
- GPU audit status is easy to correlate with backend matrices.

## P2: Optional Improvements

### 9. Machine-readable audit claim export
- Add a generated JSON/YAML file summarizing:
  - CT evidence layers
  - blocking workflows
  - advisory workflows
  - machine-checked proof status

Potential output:
- `docs/reports/audit_claims.json`
- or `artifacts/audit_claims.json`

### 10. Add a "current truth" CI badge section
- Expose badges or compact status rows for:
  - CI
  - Security Audit
  - ct-verif
  - valgrind-ct
  - GPU self-hosted

This is optional, but useful for reviewers and downstream integrators.

## Suggested Execution Order
1. Evidence bundle completion
2. Narrative drift checks
3. Historical report cleanup
4. Audit runner fidelity cleanup
5. Optional machine-readable claims export

## Done Criteria
- Audit package includes or links all current CT evidence layers.
- No active docs claim CT/audit capabilities are merely planned when implemented.
- Historical reports are clearly labeled or updated.
- Preflight detects future narrative drift in critical audit docs.
- Reviewers can understand the current assurance stack without reading source code.
