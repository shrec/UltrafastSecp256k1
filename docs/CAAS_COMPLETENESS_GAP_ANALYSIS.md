# CAAS Completeness Gap Analysis — What's Missing for the Original Vision

**Author:** Audit Agent
**Original date:** 2026-04-27
**Reconciled:** 2026-06-13 — every "Remaining gap" from the original draft was
re-verified against the live gates and current files. All six structural gaps
the original draft listed as missing/partial are now **closed and CI-gated**.
The frontier has moved: see [Bastion Final Mile](#bastion-final-mile-residual-hardening).

> **Reconciliation discipline.** This document used to claim P21, G-7, G-8, G-9b,
> the audit dashboard, and thread-safety were missing or partial. Those claims
> were stale. They are corrected below with concrete file + live-gate-verdict
> evidence so no future reviewer builds new work from outdated prose. **Green
> gates are not the same as a finished Bastion** — the residual hardening work is
> enumerated explicitly rather than hidden behind a "done" status.

---

## Original CAAS Vision

CAAS was created to replace expensive snapshot-PDF review ($40k-$250k) with a continuous, reproducible, open-source pipeline:

> "Every claim made about the project must be expressed as an executable test, every test must run on every commit, every result must be reproducible by an independent third party, and every fixed bug must become a permanent regression test."

**Final goal from the gap-closure roadmap:**

> "when this file is fully checked, CAAS has enough executable evidence that independent reviewers focus on methodology, replay, and novel hypotheses rather than rediscovering known bug classes."

---

## Existing Coverage Through P21

| Component | Status |
|-----------|--------|
| 5-stage pipeline (static analysis -> audit gate -> autonomy -> bundle -> verify) | Closed |
| 12 CAAS pipeline bugs fixed (d0da0c38) — C1-C12 | Closed |
| audit modules + exploit PoCs, ~1M assertions (counts synced by `ci/sync_module_count.py`) | Closed |
| 16+ fuzz harnesses (4066 lines) | Closed |
| CAAS protocol spec (`CAAS_PROTOCOL.md`) | Closed |
| CAAS hardening backlog (`CAAS_HARDENING_TODO.md`) — 12 items, all closed | Closed |
| THREAT_MODEL.md (G-1) | Closed |
| RNG_ENTROPY_ATTESTATION.md (G-2) | Closed |
| HARDWARE_SIDE_CHANNEL_METHODOLOGY.md (G-3) | Closed |
| INTEROP_MATRIX.md (G-4) | Closed |
| COMPLIANCE_STANCE.md (G-6) | Closed |
| MULTI_CI_REPRODUCIBLE_BUILD.md + `ci/multi_ci_repro_check.py` (G-7) | Closed |
| CT_TOOL_INDEPENDENCE.md + `ci/ct_independence_check.py` (G-8) | Closed |
| SPEC_TRACEABILITY_MATRIX.md (G-5) | Closed |
| PROTOCOL_SPEC.md (G-9) | Closed |
| exploit_traceability_join.py script + gate (G-9b) | Closed |
| .well-known/security.txt | Closed |
| CAAS runner (`caas_runner.py`) | Closed |
| CAAS CI workflows (caas.yml, caas-evidence-refresh.yml, preflight.yml) | Closed |
| Evidence governance (HMAC-signed chain) | Closed |
| **P21 — External-Audit Replacement gate** (`audit_gate.py --external-audit-replacement`) | Closed |

---

## Status Reconciliation (2026-06-13)

Each item the original draft flagged as missing/partial, re-verified against the
live repository. "Live verdict" = the result of running the gate today.

### 1. G-10: SECURITY_INCIDENT_TIMELINE.md — **Closed**

`docs/SECURITY_INCIDENT_TIMELINE.md` exists. Disclosure SLA gated by
`audit_gate.py --disclosure-sla`.

### 2. P21 — CAAS Completeness Principle — **Closed (registered + gated)**

- Registered in `docs/AUDIT_MANIFEST.md` (P21).
- Implemented as `check_external_audit_replacement` in `ci/audit_gate.py`
  (`--external-audit-replacement`), wired into `CHECK_MAP` and `ALL_CHECKS`.
- **Live verdict:** `P21: External-Audit Replacement Gate -> PASS`.
- Residual: the gate today verifies *presence* of the required CAAS documents.
  Upgrading it to a **semantic requirement map** (gap → artifact → gate → status
  → residual) is the Bastion Final Mile item **B1** below.

### 3. CAAS_PROTOCOL.md Done Criteria — **Closed**

The two formerly-open items are resolved:

- **H-1 (Evidence age cannot silently fail):** `caas-evidence-refresh.yml` is
  deployed and has operated well past the 30-day observation window. The
  freshness gate (`ci/audit_sla_check.py`) **actively blocks** when evidence
  ages out — confirmed live on 2026-06-13 when `audit/ci-evidence` crossed its
  14-day critical threshold and dropped the autonomy score to 90 (loud failure,
  not silent). The *prevention* upgrade (pre-alert before the block) is Bastion
  Final Mile item **B3**.
- **H-9 (Audit dashboard exists):** `docs/AUDIT_DASHBOARD.md` is generated and
  refreshed by the nightly evidence workflow. (GitHub Pages publication remains
  an optional convenience, not a gate.)

### 4. G-9b: Exploit Traceability Join — **Closed (script + gate)**

- `ci/exploit_traceability_join.py` exists and generates
  `docs/EXPLOIT_TRACEABILITY_JOIN.md`.
- Gated by `audit_gate.py --exploit-traceability` (reported as
  *G-11: Exploit Traceability*). **Live verdict:** PASS.

### 5. G-7: Multi-CI Reproducible Build — **Closed**

- `docs/MULTI_CI_REPRODUCIBLE_BUILD.md`, `.github/workflows/multi-ci-repro.yml`,
  and `ci/multi_ci_repro_check.py` all exist.
- Residual: negative fixtures proving the cross-provider hash-mismatch path
  fails closed are Bastion Final Mile item **B5**.

### 6. G-8: Two-Tool CT Independence — **Closed**

- `docs/CT_TOOL_INDEPENDENCE.md`, `.github/workflows/ct-independence.yml`, and
  `ci/ct_independence_check.py` all exist.
- Gated by `audit_gate.py --ct-tool-agreement`. **Live verdict:** PASS
  (3 tools, 100% agreement).
- Residual: artifact-based CT evidence (consume real per-tool verdict JSONs,
  freshness-tracked) is Bastion Final Mile item **B11**.

### 7. C ABI Thread Safety — **Closed**

`docs/THREAD_SAFETY.md` exists and is one of the P21-required documents.

---

## Gap Summary Table (reconciled)

| # | Item | Original status | 2026-06-13 status | Evidence |
|---|------|-----------------|-------------------|----------|
| P21 | P21 registration (`AUDIT_MANIFEST.md` + `audit_gate.py`) | Not started | **Closed** | `--external-audit-replacement` PASS |
| G-10 | `SECURITY_INCIDENT_TIMELINE.md` | Closed | **Closed** | file present; `--disclosure-sla` |
| G-9b | exploit-traceability gate + CI | Not started | **Closed** | `--exploit-traceability` PASS |
| H-9 | Audit dashboard | Partial | **Closed** | `docs/AUDIT_DASHBOARD.md` nightly-refreshed |
| Threading | Thread-safety doc in C ABI | Not started | **Closed** | `docs/THREAD_SAFETY.md` |
| H-1 | 30-day observation period | Waiting | **Closed** | window elapsed; gate blocks loudly |
| G-7 | Second CI provider repro build | Partial | **Closed** | `multi-ci-repro.yml` + `multi_ci_repro_check.py` |
| G-8 | Second CT tool independence | Partial | **Closed** | `ct-independence.yml` + `ct_independence_check.py` |

---

## Bastion Final Mile (residual hardening)

The original six gaps are closed. The remaining work is **not** new features — it
is turning presence-level green gates into *semantic, replayable, fail-closed*
evidence. Tracked in `workingdocs/CLAUDE_BASTION_WORKPLAN.md` as B0–B11.

| ID | Residual | Why it still matters |
|----|----------|----------------------|
| B1 | P21 presence → **semantic requirement map** (`docs/CAAS_BASTION_REQUIREMENTS.json`) | A green presence check does not prove each gap maps to a *live gate* with a current status and a named residual. |
| B3 | Evidence freshness **pre-alert** + "days-until-block"; nightly refresh covers *all* critical evidence | The gate blocks at the SLA breach; it should warn before, and the nightly refresh must cover every freshness-tracked file (currently misses `audit/ci-evidence`). |
| B4 | External-audit bundle **strict current-run** evidence + negative fixtures | The committed bundle is a historical baseline; reviewers need a clear HEAD-vs-bundle status and tamper/stale/replay negative proofs. |
| B5 | **Negative-fixture suite** for every high-value gate | A green gate without a proof that it fails on bad input is a trust assumption, not evidence. |
| B8 | Performance claims **co-gated** with security + corrupt-artifact rejection | A perf claim is invalid if CT/determinism/integration is red, and corrupt benchmark artifacts must be rejected, not parsed. |
| B9 | Incident drills as **real fault injection** + drill-cadence SLO | A drill that only checks a script exists does not prove detection; cadence must be freshness-tracked. |
| B6/B7/B10/B11 | Research-signal enrichment, integration evidence table, GPU parity-marker hygiene, artifact-based CT evidence | Each turns scattered or narrative evidence into a single replayable artifact. |

**Status wording rule:** until B1–B11 land, CAAS is a *strong, green CI pipeline*
— not yet a *finished Bastion*. The difference is exactly the negative fixtures,
semantic requirement mapping, and pre-alert freshness above. Do not describe CAAS
as "100% Bastion-complete" on the strength of green gates alone.

---

## Final Assessment

After the 12 CAAS pipeline bug fixes (d0da0c38) and the G-1..G-10 + P21 closures,
CAAS is a stable, self-contained, CI-blocking audit pipeline. Every structural gap
the original 2026-04-27 draft enumerated is closed and gated.

The remaining distance to **Bastion** is qualitative, not structural: every gate
must prove it fails closed (B5), every claim must map to a live artifact and gate
(B1), evidence must warn before it blocks (B3), the bundle must prove the current
commit (B4), performance must be co-gated with correctness (B8), and drills must
inject real faults (B9). These are tracked in the Bastion Final Mile table above.

**After B1–B11 land, CAAS will be:** a replayable assurance system that forces
bugs, stale claims, and new research signals into the same evidence chain — not
just a CI pipeline that is currently green.

---

*End of CAAS Completeness Gap Analysis (reconciled 2026-06-13)*
