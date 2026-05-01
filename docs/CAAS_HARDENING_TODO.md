# CAAS Hardening Backlog — H-1 through H-12

**Status:** All 12 hardening items closed as of 2026-04-21.
**Reference:** `docs/AUDIT_CHANGELOG.md` § 2026-04-21 (CAAS hardening)

This document lists the CAAS structural hardening items (H-1..H-12)
identified in the initial CAAS gap analysis. They address fragility,
depth gaps, and visibility gaps in the CAAS pipeline.

---

## P0 — Structural Fragility

### H-1 — Nightly assurance auto-refresh ✅ CLOSED 2026-04-21

**Artifact:** `.github/workflows/caas-evidence-refresh.yml`

Daily cron at 04:30 UTC regenerates `assurance_report.json`,
`EXTERNAL_AUDIT_BUNDLE.{json,sha256}`, and `SECURITY_AUTONOMY_KPI.json`,
commits as `caas-bot` only when content changed. Eliminates the
`audit_sla_check.py max_stale_evidence_days` silent-trip class.

**Status:** Deployed. 30-day continuous observation period started 2026-04-21.

---

### H-2 — HMAC evidence-chain key policy ✅ CLOSED 2026-04-21

**Artifact:** `docs/EVIDENCE_KEY_POLICY.md`

Documents the honest tamper-evident-only scope of the embedded HMAC key,
threat model, and the rotation/escrow procedure for any future move to a
true secret key.

---

### H-3 — CAAS protocol standalone spec ✅ CLOSED 2026-04-21

**Artifact:** `docs/CAAS_PROTOCOL.md`

Stage-by-stage contract, artifact layout, retention policy, drift policy,
and local replay commands. Enables independent third-party verification of
CAAS pipeline results.

---

## P1 — Assurance Depth

### H-4 — Mutation kill-rate weekly gate ✅ CLOSED 2026-04-21

**Artifact:** `.github/workflows/mutation-weekly.yml`

Sunday 03:30 UTC; opens or updates the `mutation-kill-rate-regression`
issue if `mutation_kill_rate.py --threshold 75` fails. Visibility-only,
does not block `dev` pushes (heavy lane: ~1h runtime).

---

### H-5 — GPU schnorr_snark_witness_batch performance gap ✅ CLOSED 2026-04-21

**Artifact:** `docs/RESIDUAL_RISK_REGISTER.md` (RR-005)

`schnorr_snark_witness_batch` performance gap recorded as RR-005. Correctness
parity is closed via the host-side CPU fallback; explicitly tagged as a
performance gap, not a correctness gap.

---

### H-6 — Local supply-chain parity doc ✅ CLOSED 2026-04-21

**Artifact:** `docs/SUPPLY_CHAIN_LOCAL_PARITY.md`

Coverage matrix of which P15 controls run offline vs. need GitHub, with
acceptance criteria for the local-only review pass. Allows a reviewer to
run CAAS without GitHub Actions access.

---

## P2 — Visibility & Hygiene

### H-7 — Review-queue aging SLA ✅ CLOSED 2026-04-21

**Artifact:** `ci/review_queue_age_check.py` + `docs/AUDIT_SLA.json`

`review_queue_max_open_days` SLO (90 days, warning) added. Current run at
close: 23 review-queue rows, 0 over SLA.

---

### H-8 — TODO/FIXME age tracker ✅ CLOSED 2026-04-21

**Artifact:** `ci/todo_age_check.py` + `docs/AUDIT_SLA.json`

`todo_max_open_days` SLO (180 days, warning). Uses `git blame` and honours
`DEFERRED:` / `(tracked: …)` annotations. Current run at close: 61 markers,
0 over SLA.

---

### H-9 — Audit dashboard generator ✅ CLOSED 2026-04-21

**Artifact:** `ci/render_audit_dashboard.py` → `docs/AUDIT_DASHBOARD.md`

Emits a human-readable audit status dashboard. Designed to run inside the
H-1 nightly job so the dashboard is regenerated daily.

**Note:** Dashboard HTML not yet deployed to GitHub Pages (tracked in
`docs/CAAS_COMPLETENESS_GAP_ANALYSIS.md` as a medium-priority gap).

---

### H-10 — Reviewer prompt templates ✅ CLOSED 2026-04-21

**Artifact:** `docs/REVIEWER_PROMPTS/` (auditor.md, attacker.md,
perf_skeptic.md, docs_skeptic.md + usage README)

Four graph-aware reviewer prompts for human code review. Each prompt
assumes access to the source-graph workflow and produces structured
findings against the CAAS evidence chain.

---

### H-11 — ROCm/HIP smoke pipeline scaffold ✅ CLOSED 2026-04-21

**Artifact:** `.github/workflows/rocm-smoke.yml`

Manual / labelled-PR trigger. Scaffold-only — promotion to claim status
requires hardware-backed evidence per RR-003. Ensures the CI wiring is
in place when hardware becomes available.

---

### H-12 — dev_bug_scanner CVE-grounded checkers ✅ CLOSED 2026-04-21

**Artifact:** `ci/dev_bug_scanner.py` (13 new checkers)

Extended dev_bug_scanner with 13 CVE-grounded crypto-specific checkers
covering nonce reuse, missing low-S normalization, unsafe scalar parsing,
ECDH output handling, and similar patterns. Initial run: 3 MEDIUM signals,
zero HIGH false positives.

---

## Summary

| Item | Category | Artifact | Status |
|------|----------|----------|--------|
| H-1 | P0: Structural | `.github/workflows/caas-evidence-refresh.yml` | ✅ 2026-04-21 |
| H-2 | P0: Structural | `docs/EVIDENCE_KEY_POLICY.md` | ✅ 2026-04-21 |
| H-3 | P0: Structural | `docs/CAAS_PROTOCOL.md` | ✅ 2026-04-21 |
| H-4 | P1: Depth | `.github/workflows/mutation-weekly.yml` | ✅ 2026-04-21 |
| H-5 | P1: Depth | `docs/RESIDUAL_RISK_REGISTER.md` (RR-005) | ✅ 2026-04-21 |
| H-6 | P1: Depth | `docs/SUPPLY_CHAIN_LOCAL_PARITY.md` | ✅ 2026-04-21 |
| H-7 | P2: Visibility | `ci/review_queue_age_check.py` | ✅ 2026-04-21 |
| H-8 | P2: Visibility | `ci/todo_age_check.py` | ✅ 2026-04-21 |
| H-9 | P2: Visibility | `ci/render_audit_dashboard.py` | ✅ 2026-04-21 |
| H-10 | P2: Visibility | `docs/REVIEWER_PROMPTS/` | ✅ 2026-04-21 |
| H-11 | P2: Visibility | `.github/workflows/rocm-smoke.yml` | ✅ 2026-04-21 |
| H-12 | P2: Visibility | `ci/dev_bug_scanner.py` (+13 checkers) | ✅ 2026-04-21 |

All 12 items are closed. No open hardening items remain.

---

*CAAS Hardening Backlog — generated 2026-04-28 from `docs/AUDIT_CHANGELOG.md`*
