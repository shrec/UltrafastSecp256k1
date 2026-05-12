# Assurance Ledger

Canonical claim-to-evidence map for UltrafastSecp256k1.

This document exists to reduce assurance drift.
Every high-value public claim should map to concrete code, tests, workflows,
artifacts, and a current status. If a claim cannot be wired to reproducible
evidence, it should be downgraded or removed.

The goal is simple: make the repository itself the source of truth.

## Operating Posture

This ledger is not only for public claims or external review.

The target standard is owner-grade assurance: even if the team is not currently
operating its own large-value assets on top of this engine, the audit class
should be strong enough to answer the harder internal question:

"If we had to depend on this system ourselves for high-value operation, what
evidence would prove that it is safe enough to run, and what residual risks
would still remain?"

That is the standard this ledger is meant to support.

---

## Ledger Schema

Each claim should be tracked with the following fields:

| Field | Meaning |
|-------|---------|
| Claim ID | Stable identifier for cross-reference |
| Area | CPU, GPU, CI, audit, docs, bindings, etc. |
| Claim | The actual statement being made publicly |
| Scope | What is and is not covered |
| Primary Evidence | Code, test, or doc anchor |
| Enforcing Workflow | CI workflow or check that guards the claim |
| Artifact | JSON report, benchmark log, audit output, or generated proof |
| Verification Cadence | Per build, per push, nightly, release-only, manual |
| Current Status | Active, partial, planned, advisory-only |
| Stale Risk | Low, medium, high |
| Owner Surface | File set or subsystem that must stay aligned |

---

## Core Claims

| Claim ID | Area | Claim | Scope | Primary Evidence | Enforcing Workflow | Artifact | Verification Cadence | Current Status | Stale Risk | Owner Surface |
|----------|------|-------|-------|------------------|--------------------|----------|----------------------|----------------|------------|---------------|
| A-001 | CT signing | Secret-bearing production signing routes through the CPU constant-time layer | CPU signing and key-derivation paths; excludes public-data fast paths | `SECURITY.md`, `docs/BACKEND_ASSURANCE_MATRIX.md`, `src/cpu/src/ct_sign.cpp` | `ct-verif.yml`, `ct-arm64.yml`, `valgrind-ct.yml` | CT logs, dudect outputs, Valgrind CT traces | Every push + nightly | Active | Medium | `src/cpu/src/ct_*`, `SECURITY.md`, `docs/BACKEND_ASSURANCE_MATRIX.md` |
| A-002 | GPU ABI | Stable public GPU C ABI exposes 16 backend-neutral batch operations | Public host ABI only; excludes benchmark-only kernels and internal experiments | `include/ufsecp/ufsecp_gpu.h`, `docs/BACKEND_ASSURANCE_MATRIX.md` | `gpu-selfhosted.yml`, backend audit runners, GPU equivalence tests | GPU host API test logs, backend matrix outputs | Per GPU-enabled validation run | Active | Medium | `include/ufsecp/ufsecp_gpu.h`, `src/gpu/src/*`, `docs/BACKEND_ASSURANCE_MATRIX.md` |
| A-003 | GPU parity | CUDA, OpenCL, and Metal implement the stable public GPU surface | Public 16-op GPU ABI; not a statement about every experimental kernel | `include/ufsecp/ufsecp_gpu.h`, `docs/BACKEND_ASSURANCE_MATRIX.md`, backend adapters | GPU backend matrix tests, backend audit runners | Backend audit logs | Per GPU-enabled validation run | Active | Medium | `src/gpu/src/*`, `metal/*`, `opencl/*`, `docs/BACKEND_ASSURANCE_MATRIX.md` |
| A-004 | Bench reproducibility | Published benchmark numbers are tied to documented harnesses, versions, and raw logs | Bench docs and dashboard; not every local experiment is publishable | `docs/BENCHMARKS.md`, `.github/workflows/benchmark.yml` | `benchmark.yml`, `bench-regression.yml` | Parsed benchmark JSON, raw output logs, dashboard history | Push to dev/main + PR gate + manual reruns | Active | Medium | `docs/BENCHMARKS.md`, `.github/workflows/benchmark.yml`, `.github/workflows/bench-regression.yml` |
| A-005 | Exploit audit surface | Exploit-style PoC tests are part of the core assurance story | `audit/test_exploit_*.cpp` suite and related docs | audit tree, README, WHY | `security-audit.yml`, standalone audit builds | Test binaries, CTest logs, unified audit outputs | Every push + manual audit runs | Active | Medium | `audit/`, `README.md`, `WHY_ULTRAFASTSECP256K1.md` |
| A-006 | Source graph review | Repository graph is part of the audit workflow for scoping, impact tracing, and stale-claim detection | Review workflow and tooling; not a proof of correctness by itself | `tools/source_graph_kit/source_graph.py`, `AUDIT_GUIDE.md` | Preflight/review process, graph rebuild discipline | `source_graph.db`, graph manifests, query outputs | Every substantial review or graph rebuild | Active | Medium | `tools/source_graph_kit/`, `AUDIT_GUIDE.md`, repo instructions |
| A-007 | Self-audit transparency | Public assurance claims are backed by rerunnable artifacts rather than trust-only prose | Docs, audit reports, and CI outputs | `AUDIT_GUIDE.md`, `SECURITY.md`, `WHY_ULTRAFASTSECP256K1.md` | CI + document maintenance | Audit reports, workflow artifacts, traceability docs | Continuous | Active | Medium | Docs + CI workflows |
| A-008 | ROCm/HIP status | ROCm/HIP is an early-development compatibility path, not yet a hardware-validated assurance target | Portability layer and planned support only | `README.md`, `docs/BACKEND_ASSURANCE_MATRIX.md` | None yet | None yet | Manual until hardware-backed validation exists | Planned | High | GPU portability layer + public docs |

---

## Promotion Rules

A claim should only be promoted to the README or other top-level public surfaces
when all of the following are true:

1. The scope is stated narrowly enough to be true.
2. The primary evidence is checked into the repository.
3. At least one enforcing workflow or deterministic verification path exists.
4. The artifact or log path is known.
5. The stale-risk owner surface is explicit.

If any one of those is missing, the claim belongs in a roadmap, not in a top-level promise.

For owner-grade assurance, there is one more rule:

6. The claim must be strong enough to guide internal risk acceptance for real asset exposure, not just to sound convincing in documentation.

---

## Stale-Claim Review Checklist

When updating a public claim, verify all of the following:

1. Does the same number or status appear in `README.md`, `WHY_ULTRAFASTSECP256K1.md`, `SECURITY.md`, and backend docs?
2. Is the claim enforced by a workflow, or merely described by one?
3. Is there a current artifact path or benchmark log that another reviewer can rerun?
4. Does the claim describe the stable public surface, or an internal/experimental surface?
5. Has the source graph or audit manifest been updated if the subsystem surface changed?

---

## Immediate Gaps To Close

These are the next assurance-system gaps that should be reduced:

1. Track AI-assisted review passes as first-class assurance events.
2. Add graph-driven stale-claim checks to preflight.
3. Expand performance governance to better cover GPU-facing public claims.
4. Promote ROCm/HIP only after hardware-backed validation enters the matrix.

Machine-readable companion:

- `docs/ASSURANCE_CLAIMS.json`