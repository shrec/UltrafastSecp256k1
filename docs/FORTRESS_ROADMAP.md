# Fortress Roadmap

Roadmap for closing the remaining assurance-system gap in UltrafastSecp256k1.

The objective is not to wait for outside validation before becoming robust.
The objective is to make the repository itself increasingly difficult to break,
increasingly easy to audit, and increasingly strict about public claims.

---

## Goal

Move the assurance model from strong and transparent to fortress-grade.

That means:

1. Claims are machine-linked to evidence.
2. Review blind spots shrink over time.
3. External-style review loops are absorbed into deterministic checks.
4. Documentation drift becomes harder to introduce.
5. Performance and capability governance match public positioning.
6. Known failure classes are covered by explicit self-audit surfaces rather than implied confidence.

Practical interpretation:

The target is not to claim literal proof that "all possible bugs" are gone.
The target is to drive the self-audit system toward near-total coverage of all
known failure classes and to leave any residual risk explicit, scoped, and hard
to hide.

Operating posture:

This roadmap assumes the audit class should be strong enough for owner-grade
responsibility, even if the team is not currently using the engine to manage
its own high-value assets. Because of that, the assurance target is internal
safety and operational trustworthiness first, not external optics or marketing
trust.

---

## Workstream 1: Canonical Assurance Ledger

Status: in progress

Deliverables:

1. `docs/ASSURANCE_LEDGER.md`
2. Machine-readable companion format
3. Claim ID references in public docs
4. Stale-claim review checklist in preflight

Outcome:

Public claims become versioned, scoped, and auditable.

---

## Workstream 2: AI-Assisted Audit Formalization

Status: started

Deliverables:

1. `docs/AI_AUDIT_PROTOCOL.md`
2. Review mode taxonomy
3. Accepted-finding conversion rules
4. Review event logging format

Outcome:

AI review becomes a governed input into the assurance system rather than an informal side channel.

---

## Workstream 3: Graph-Driven Assurance

Status: in progress

Deliverables:

1. Graph coverage checks for new source/doc/test surfaces
2. Stale-claim detection linked to the graph
3. Impact reports for public-surface changes
4. Graph-first audit bundles for reviewers

Outcome:

The source graph becomes part of the assurance engine, not just review tooling.

Current progress:

1. Claim-surface graph coverage is validated through `scripts/validate_assurance.py`.
2. `scripts/preflight.py --claims` exposes graph-driven stale-claim checks locally.
3. Public-surface drift now fails closed when claim evidence resolves on disk but not in the indexed graph.
4. Workspace graph coverage now explicitly includes the library audit scripts surface so orphan runtime harnesses cannot disappear from discovery at the workspace level.

Self-audit completeness criteria for this workstream:

1. Every public claim surface should resolve to concrete repository evidence.
2. Every new subsystem should appear in the graph, in docs, and in at least one deterministic validation path.
3. Missing graph coverage should be treated as an assurance gap, not a documentation detail.

---

## Workstream 4: GPU Governance Hardening

Status: in progress

Deliverables:

1. Stronger GPU-specific benchmark governance
2. Clear distinction between stable host ABI and experimental/internal kernels
3. Optional ROCm/HIP promotion criteria tied to hardware-backed validation
4. Better workflow coverage for GPU public differentiators

Outcome:

The strongest public performance claims become guarded by comparably strong enforcement.

Scope note:

ROCm/HIP hardware-backed validation is a future expansion path for backend
coverage and publishability. It is not a prerequisite for the validity of the
current audit surfaces for CPU, CUDA, OpenCL, or Metal, and it does not weaken
already validated backends while AMD hardware is absent.

Current progress:

1. `docs/GPU_BACKEND_EVIDENCE.json` now defines backend status, publishability, and required artifact classes.
2. `scripts/validate_assurance.py` validates fail-closed ROCm/HIP promotion rules.
3. `scripts/preflight.py --gpu-evidence` exposes the local gate for backend publishability checks.
4. Public narrative surfaces now cross-reference the assurance ledger claim IDs for top-level trust statements.
5. GPU parity documentation has been reconciled around the real 13/13 CUDA/OpenCL/Metal stable C ABI surface.
6. ROCm/HIP promotion is now documented as a hardware-backed checklist rather than a vague future task.

---

## Workstream 5: Drift-Resistant Documentation

Status: in progress

Deliverables:

1. Canonical link graph for README, WHY, SECURITY, AUDIT_GUIDE, and backend docs
2. Reduced metric drift across files
3. Clearer distinction between active, partial, planned, and experimental states
4. Faster stale-count detection

Outcome:

Top-level narrative stays aligned with repository truth as the code evolves.

## Workstream 6: Self-Audit Completeness

Status: in progress

Deliverables:

1. Failure-class inventory for correctness, parser boundaries, CT, protocol misuse, ABI misuse, GPU host misuse, docs drift, and benchmark overclaim risk
2. Deterministic mapping from each failure class to one or more audit surfaces
3. Residual-risk accounting for anything not yet covered by a deterministic check
4. Fail-closed policy for new features that land without an audit-path mapping

Outcome:

The self-audit system becomes harder to bypass because coverage is measured by
problem class, not by raw test counts.

Current progress:

1. `docs/SELF_AUDIT_FAILURE_MATRIX.md` now maps major failure classes to deterministic audit surfaces and named residual-risk notes.
2. `docs/OWNER_GRADE_AUDIT_TODO.md` now translates the remaining assurance gap into concrete code-and-tooling work for an owner-grade audit class.
3. `scripts/audit_gate.py` now absorbs `invalid_input_grammar.py`, `stateful_sequences.py`, and `audit_test_quality_scanner.py` into the default fail-closed perimeter.
4. `scripts/mutation_kill_rate.py` is now exposed through the audit gate as an explicit heavy lane instead of remaining orphan tooling.
5. The library graph builder now maps the BIP352 scan-plan audit surface so `ufsecp_bip352_prepare_scan_plan` is no longer allowed to hide behind a missing test-map edge.

Coverage target:

The project should aim for near-total coverage of known failure classes.
Practically, that means pushing the residual risk surface toward the smallest
possible set of explicitly named gaps instead of treating "lots of tests" as a
proxy for completeness.

---

## Immediate Tasks

1. Drive the ABI hostile-caller quartet blocker count to zero so the matrix can upgrade that row back to fully covered.
2. Burn down the low-severity audit-test-quality backlog so audit code itself stops carrying vacuous or weak assertions.
3. Strengthen mutation-evidence freshness so heavy-lane results cannot go stale invisibly after core arithmetic changes.
4. Expand graph-driven stale-claim checks from core public claims into more subsystem-specific docs.
5. Expand GPU-facing performance governance beyond backend publishability into reproducibility and artifact-retention checks.
6. Push the optional ROCm/HIP promotion checklist into more GPU-facing docs and release procedures without treating it as an audit blocker.
7. Keep shrinking the named residual-risk set with deterministic gates where possible.

---

## Promotion Criteria For "Fortress" Status

The project should only claim fortress-grade self-audit when all of the following are true:

1. Every top-level trust claim maps to ledger evidence.
2. AI-assisted review is governed by a documented protocol.
3. Graph tooling participates directly in assurance checks.
4. GPU public claims have enforcement close to CPU public claims.
5. Public docs have a low stale-drift rate over time.
6. Known failure classes have named coverage paths or explicit residual-risk entries.
7. The default audit gate includes runtime hostile-input and stateful misuse checks, not only static inventories.
8. Heavy mutation evidence exists and is kept current for high-risk arithmetic and audit-harness changes.

---

## What This Roadmap Does Not Assume

This roadmap does not assume trust should depend on a one-time snapshot report.
External review is always welcome; the repository is structured so reviewers can step in and replay all evidence at any time.
The point is to ensure the codebase is already strong, transparent, and rerunnable before anyone arrives.