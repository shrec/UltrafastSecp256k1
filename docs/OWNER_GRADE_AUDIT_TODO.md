# Owner-Grade Audit TODO

Concrete code-and-tooling backlog for raising the audit class to an
owner-grade standard.

This document is not about public optics. It is the build list for the
infrastructure we would want if the engine had to satisfy the same standard we
would demand before entrusting it with high-value operation.

Use it together with:

- `docs/SELF_AUDIT_FAILURE_MATRIX.md`
- `docs/AUDIT_MANIFEST.md`
- `docs/ASSURANCE_LEDGER.md`

---

## Priority Model

| Priority | Meaning |
|----------|---------|
| P0 | Should exist before claiming owner-grade audit class |
| P1 | Strongly recommended to reduce real residual risk |
| P2 | Nice to have, improves review velocity and evidence quality |

---

## P0 -- Must Write

### 1. Failure-Class Gate Runner

Need:

- `scripts/audit_gap_report.py` [STARTED]
- optional integration into `scripts/audit_gate.py`

What it should do:

1. Read `docs/SELF_AUDIT_FAILURE_MATRIX.md`
2. Verify every row has at least one deterministic surface
3. Verify every `Partial` or `Intentionally deferred` row has a residual-risk note
4. Emit machine-readable JSON showing covered vs partial vs deferred classes

Why it matters:

Right now the failure-class matrix is canonical documentation, but not yet a
hard gate. Owner-grade posture needs the matrix to be executable, not only
descriptive.

Current progress:

1. `scripts/audit_gap_report.py` now exists and parses `docs/SELF_AUDIT_FAILURE_MATRIX.md`.
2. Normal mode validates matrix structure and repository evidence references.
3. `--strict` mode exposes the current owner-grade residual set instead of hiding it behind aggregate coverage language.

### 2. CT Evidence Aggregator

Need:

- `scripts/collect_ct_evidence.py` [STARTED]
- standardized artifact output under `artifacts/ct/`

What it should do:

1. Gather `ct-verif`, `valgrind-ct`, dudect, and disassembly-scan outputs
2. Normalize them into one JSON summary and one text summary
3. Mark which CT claims are deterministic vs statistical vs manual-only

Current progress:

1. `scripts/collect_ct_evidence.py` now discovers `ct-verif`, `valgrind-ct`,
	disassembly scan, and dudect surfaces across repo/build output paths.
2. It normalizes discovered evidence into canonical `artifacts/ct/` slots and
	emits `ct_evidence_summary.json` plus `ct_evidence_summary.txt`.
3. `scripts/generate_audit_package.sh` now delegates CT artifact collection to
	this aggregator instead of maintaining separate ad hoc path guessing.
4. `audit/run_full_audit.sh` now materializes canonical CT artifacts directly:
	disassembly scan output is written under `artifacts/ct/`, legacy
	`artifacts/disasm/` compatibility is preserved, and available dudect smoke /
	full logs are captured into canonical CT slots before normalization.
5. `audit/run_full_audit.ps1` now mirrors the same canonical CT evidence
	contract and invokes the collector when the required local tools exist.
6. Current truthful residuals are no longer blocked on missing deterministic
	CT plumbing: Docker-backed local CI can now materialize both `ct-verif`
	and `valgrind-ct` artifacts, and the collector upgrades them to
	`artifact-present` when those runs are actually performed.
7. Repeated dudect smoke leakage in `ct::scalar_sub`, `ct::scalar_add`, and
	`ct::field_add` was traced to comparison-based carry/borrow extraction in
	CPU CT arithmetic and fixed by switching those hot paths to pure bitwise
	carry/borrow formulas.
8. After rebuild, `test_ct_sidechannel_smoke` passed on repeated runs and the
	strict x86_64 disassembly scan was tightened to exact primitive targets,
	with stack-canary and fixed-trip public-loop artefacts filtered out so the
	report reflects actual suspicious CT control flow instead of known-safe
	compiler epilogues.
9. `audit/run_full_audit.sh` and `audit/run_full_audit.ps1` now best-effort
	capture `ct_verif.log` and `valgrind-ct` outputs into canonical CT slots,
	while the collector keeps passive `ct-verif` runs at `configured-only` so
	local no-backend logs do not get misreported as deterministic owner-grade
	evidence.
10. `docker/run_ci.sh ct-verif` now emits collector-readable
	`build-ci/ct-ir/*_report.txt` files plus `build-ci/ct-ir/ct_verif.log`, so
	Docker local CI produces truthful `ct-verif` evidence instead of only raw
	LLVM IR files.
11. A Docker run of `scripts/valgrind_ct_check.sh` now produces canonical
	`build/valgrind-ct-docker/valgrind_reports/*` artifacts that the collector
	promotes to `artifact-present`; in the validated local run, Valgrind found
	zero uninitialized-branch hits while the standalone dudect timing suite
	still reported advisory leaks in public point-add timing cases.

Why it matters:

The biggest owner-grade residual risk is still CT confidence ceiling. The first
tool we need is one place that says exactly what evidence exists for each CT
claim and what is still manual.

### 3. ABI Negative-Test Generator

Need:

- `scripts/generate_abi_negative_tests.py`

What it should do:

1. Walk public headers and exported ABI functions
2. Generate or update a negative-case manifest for NULL, zero-count, invalid-content, and success-smoke coverage
3. Flag exported functions that do not yet satisfy the 4-part hostile-caller rule

Why it matters:

ABI misuse is one of the most expensive real-world failure modes. Manual audit
tests catch a lot already, but owner-grade posture should make missing edge-case
coverage mechanically visible for every export.

### 4. Secret-Path Change Gate

Need:

- `scripts/check_secret_path_changes.py`
- integration into `preflight.py` or `audit_gate.py`

What it should do:

1. Detect changes in CT files and secret-bearing routing surfaces
2. Require matching updates in `CT_VERIFICATION.md`, `SECURITY_CLAIMS.md`, or paired audit docs
3. Require refreshed CT evidence references when relevant files changed

Why it matters:

Secret-bearing paths should have a stricter change policy than public-data
paths. Owner-grade audit class requires this asymmetry to be enforced, not just
stated.

### 5. Owner-Grade Evidence Bundle Builder

Need:

- `scripts/build_owner_audit_bundle.py`

What it should do:

1. Package current assurance ledger, failure-class matrix, CT evidence, audit summaries, benchmark publishability state, and GPU backend evidence into one reviewable bundle
2. Produce a single JSON + text summary of current readiness and residual risk
3. Fail if mandatory artifacts are missing

Why it matters:

If we cannot assemble one compact owner-grade bundle from repo state, then the
system is still too fragmented for hard internal risk acceptance.

---

## P1 -- Strongly Recommended

### 6. Residual-Risk Registry

Need:

- `docs/RESIDUAL_RISK_REGISTER.md`
- optional `scripts/check_residual_risk.py`

What it should do:

1. Assign stable IDs to explicit residual risks
2. Track owner, mitigation path, blocking status, and last review date
3. Link each residual risk to the relevant failure-class row

Why it matters:

Named residual risk is good. Named residual risk with ownership and aging is
better.

### 7. Protocol Misuse Coverage Expander

Need:

- more exploit-style tests under `audit/test_exploit_*.cpp`
- optional manifest `docs/EXPLOIT_CLASS_MATRIX.md`

What it should do:

1. Track misuse classes per protocol feature
2. Highlight transcript-binding, nonce, replay, and malformed-session gaps
3. Keep protocol expansion tied to exploit-class coverage instead of generic tests

Why it matters:

For large-asset safety, protocol misuse is at least as dangerous as arithmetic
bugs.

### 8. Thread-Safety / Hostile-Caller Stress Harness

Need:

- dedicated stress runner for stateful ABI sequences

What it should do:

1. Exercise clone/destroy/error-string/context lifecycle under contention
2. Produce deterministic summaries plus sanitizer-friendly mode
3. Expand beyond current spot checks for thread-adjacent behavior

Why it matters:

Owner-grade systems fail at boundaries and race conditions, not only in crypto
math.

### 9. Local Supply-Chain Verification Parity

Need:

- scripts to locally re-check as much of dependency-review / workflow integrity / artifact provenance as feasible

What it should do:

1. Verify pinned action refs and workflow invariants locally
2. Verify release artifact hashes and provenance material locally
3. Report which checks still remain GitHub-native only

Why it matters:

Current supply-chain posture is decent, but owner-grade review wants a smaller
GitHub-only trust surface.

---

## P2 -- Nice To Have

### 10. Audit Dashboard Generator

Need:

- `scripts/render_audit_dashboard.py`

What it should do:

1. Render current status of failure classes, residual risks, workflows, and artifacts
2. Show trend deltas rather than raw counts alone
3. Link directly to evidence artifacts

### 11. Reviewer Prompt Templates

Need:

- prompt bundles for auditor / attacker / performance skeptic / documentation skeptic review passes

What it should do:

1. Standardize review prompts by failure class
2. Reduce false positives and duplicated effort
3. Improve repeatability of AI-assisted review

### 12. Audit Drift Changelog

Need:

- `docs/AUDIT_CHANGELOG.md`

What it should do:

1. Summarize what changed in the assurance system itself
2. Record when a risk moved from partial -> covered or covered -> partial
3. Make audit maturity changes easy to review across time

---

## Recommended Execution Order

1. Failure-class gate runner
2. CT evidence aggregator
3. ABI negative-test generator
4. Secret-path change gate
5. Owner-grade evidence bundle builder
8. Protocol misuse coverage expander
9. Local supply-chain verification parity

---

## Done Criteria

We should consider this backlog meaningfully reduced when all of the following
are true:

1. Failure-class coverage can be checked automatically, not only read manually.
2. CT evidence is aggregated into one canonical owner-grade summary.
3. Every exported ABI function has a machine-checkable hostile-caller coverage status.
4. Secret-path changes trigger stricter evidence requirements than public-path changes.
5. Residual risks are named, owned, and reviewable as a living registry.
6. A single owner-grade audit bundle can be produced from repo state without hand assembly.