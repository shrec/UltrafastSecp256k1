# CAAS — Continuous Audit as a Service Protocol

> Version: 1.0 — 2026-04-21
> Authoritative spec for the CAAS pipeline used by `caas_runner.py` and
> `.github/workflows/caas.yml`. Pairs with `AUDIT_MANIFEST.md` P20.

## Origin & Motivation

CAAS started from two questions the author kept asking himself while
building UltrafastSecp256k1:

1. **"If I don't have $100k, does my project never get to see daylight?"**
   Traditional cryptographic audits cost between $40k and $250k per
   engagement (Trail of Bits, NCC Group, Cure53, Quantstamp). For most
   open-source crypto authors that price tag is prohibitive. The result
   is a two-tier ecosystem: well-funded code gets a PDF, everything else
   ships uninspected. That is not a security model — it is a budget
   filter. A real-world, useful, fast crypto engine should not be
   blocked from production use just because its author cannot write a
   six-figure check.

2. **"Would I trust my own library with a billion-dollar asset — and if
   not, what would I have to build to earn that trust without depending
   on someone else's PDF?"**
   The honest answer was no — not because the code was wrong, but
   because no continuous, reproducible evidence existed that it stayed
   correct on every commit, on every platform, against every known
   exploit class. Snapshot audits do not answer that question. They
   describe a moment in time and decay from the day they are signed.
   What was needed was a verification process that runs on every push,
   produces evidence anyone can re-execute, and treats every fixed bug
   as a permanent regression test.

CAAS is the answer to both questions. It replaces the *snapshot-PDF +
reputation-of-the-auditor* model with a *continuous-pipeline +
reproducible-evidence* model:

| Traditional audit | CAAS |
|---|---|
| Snapshot in time (PDF) | Continuous, every commit |
| $40k–$250k per engagement | $0 marginal cost |
| Trust the auditor's reputation | Trust reproducible evidence |
| Outdated within months | Always current |
| Closed methodology | Open-source verifier |
| Cannot be re-run by third parties | Anyone can re-execute the pipeline |
| Single point of failure | Multi-CI cross-validation |

CAAS is not a replacement for human security review — it is the
infrastructure that makes human review *meaningful* by guaranteeing the
artifact under review is the same artifact that production runs, and
that every previously discovered weakness still has a regression test.

## Purpose

CAAS exists so that audit posture is **continuously verified**, not
occasionally inspected. Every push and pull request to `dev` and `main`
runs the full pipeline. Any failing stage blocks merge. Manual runs are a
supplement, never a substitute.

## Pipeline Topology

```
push / pull_request
        │
        ▼
┌──────────────────────────┐
│ Stage 1: Static Analysis │  audit_test_quality_scanner.py
└─────────────┬────────────┘
              │ pass
              ▼
┌──────────────────────────┐
│ Stage 2: Audit Gate      │  audit_gate.py  (P0–P11 + extras)
└─────────────┬────────────┘
              │ pass
              ▼
┌──────────────────────────┐
│ Stage 3: Sec. Autonomy   │  security_autonomy_check.py  (P12–P18)
└─────────────┬────────────┘
              │ pass
              ▼
┌──────────────────────────┐
│ Stage 4: Bundle Produce  │  external_audit_bundle.py    (P19)
└─────────────┬────────────┘
              │ pass
              ▼
┌──────────────────────────┐
│ Stage 5: Bundle Verify   │  verify_external_audit_bundle.py
└─────────────┬────────────┘
              │ pass
              ▼
        merge eligible
```

Failure of any stage prevents the next stage from running locally
(`caas_runner.py` is fail-fast). In CI the next stage's `needs:` clause
short-circuits the same way.

## Stage Contracts

### Stage 1 — Static Analysis

| Field | Value |
|-------|-------|
| Tool | `scripts/audit_test_quality_scanner.py` |
| Pre-step | `scripts/check_exploit_wiring.py` (CAAS Stage 0 wiring gate) |
| Input | full source tree |
| Output | `caas_scanner.json` with `total_findings`, `findings[]` |
| Exit code | 0 iff `total_findings == 0` |
| Blocking? | Yes |
| Local replay | `python3 scripts/audit_test_quality_scanner.py --json -o caas_scanner.json` |
| Artifact retention (CI) | 30 days |

Severity mapping: every scanner finding is treated as blocking. There is no
"low-severity" tier (`ლოუ პრიორიტი ჩვენთან არ არსებობს`).

### Stage 2 — Audit Gate

| Field | Value |
|-------|-------|
| Tool | `scripts/audit_gate.py` |
| Pre-deps | project graph (`scripts/build_project_graph.py --rebuild`), source graph (`tools/source_graph_kit/source_graph.py build -i`), `ufsecp_shared` built |
| Input | both graphs, headers, source, tests, docs |
| Output | `caas_audit_gate.json` with `verdict`, `checks[]` |
| Exit code | 0 iff `verdict in {"PASS", "PASS with advisory"}` |
| Principles enforced | P0, P0a–P0d, P1–P11 |
| Blocking? | Yes |
| Local replay | `python3 scripts/audit_gate.py --json -o caas_audit_gate.json` |
| Artifact retention (CI) | 30 days |

Per-principle checks are also runnable in isolation, e.g.
`python3 scripts/audit_gate.py --gpu-parity`. See
[AUDIT_MANIFEST.md](AUDIT_MANIFEST.md) for the full P0–P11 list.

### Stage 3 — Security Autonomy

| Field | Value |
|-------|-------|
| Tool | `scripts/security_autonomy_check.py` |
| Sub-gates | formal_invariants, risk_surface_coverage, audit_sla, supply_chain, perf_security_cogate, misuse_resistance, evidence_governance, incident_drills |
| Input | both graphs, evidence files, `docs/AUDIT_SLA.json`, `docs/SECURITY_AUTONOMY_KPI.json` |
| Output | `caas_autonomy.json` with `autonomy_score` (0–100), `autonomy_ready` (bool), `gates[]` |
| Exit code | 0 iff `autonomy_score == 100` and `autonomy_ready == true` |
| Principles enforced | P12–P18 |
| Blocking? | Yes |
| Local replay | `python3 scripts/security_autonomy_check.py --json -o caas_autonomy.json` |
| Artifact retention (CI) | 30 days |

Common silent failure: `audit_sla` sub-gate flips to fail when
`assurance_report.json` ages past 30 days. The
[caas-evidence-refresh](.github/workflows/caas-evidence-refresh.yml)
nightly bot is the structural fix (CAAS Hardening TODO H-1).

### Stage 4 — Bundle Produce

| Field | Value |
|-------|-------|
| Tool | `scripts/external_audit_bundle.py` |
| Input | live repo state + previous stage outputs |
| Output | `docs/EXTERNAL_AUDIT_BUNDLE.json`, `docs/EXTERNAL_AUDIT_BUNDLE.sha256` |
| Exit code | 0 on successful bundle write |
| Principles enforced | P19 (External Auditor Reproducibility Bundle) |
| Blocking? | Yes |
| Local replay | `python3 scripts/external_audit_bundle.py` |
| Artifact retention (CI) | 90 days |

The bundle pins:
- git commit + dirty flag
- SHA-256 of every evidence file
- captured stdout of each replayed gate command
- detached bundle digest

### Stage 5 — Bundle Verify

| Field | Value |
|-------|-------|
| Tool | `scripts/verify_external_audit_bundle.py` |
| Input | the bundle produced in Stage 4 |
| Output | `caas_bundle_verify.json` with `overall_pass`, `checks[]` |
| Exit code | 0 iff `overall_pass == true` |
| Blocking? | Yes |
| Local replay | `python3 scripts/verify_external_audit_bundle.py --json` |
| Local deep replay | `python3 scripts/verify_external_audit_bundle.py --replay-commands --json` |
| Artifact retention (CI) | 90 days |

The deep-replay mode re-executes every captured gate command and verifies the
output hashes match. This is the "external auditor" mode and is intentionally
slow.

## Drift Policy

| Drift class | Stage that catches it | Disposition |
|-------------|----------------------|-------------|
| Evidence age past 30d | Stage 3 (`audit_sla`) | Auto-fixed by nightly refresh bot (H-1) |
| Test added but not wired into runner | Stage 1 pre-step (`check_exploit_wiring.py`) | Block PR |
| GPU op added on one backend only | Stage 2 (P7 GPU parity) | Block PR |
| ABI surface changed without doc | Stage 2 (P10 doc-pairing) | Warn, doc-pair update required in same commit per AGENTS.md |
| Bundle SHA mismatch | Stage 5 | Block PR |
| Autonomy score < 100 | Stage 3 | Block PR |

## Adding A New Gate

To add a new gate without breaking the protocol:

1. Implement the gate as a standalone script in `scripts/` with `--json -o`
   output and exit code 0/1.
2. Add a check function to `scripts/audit_gate.py` (for P0–P11 territory) or
   register the gate in `scripts/security_autonomy_check.py` (for P12–P18
   territory) with a `weight` summing to 100 across all gates.
3. Document the principle in `AUDIT_MANIFEST.md`.
4. Add a row in the per-stage table in this document.
5. Verify locally: `python3 scripts/caas_runner.py --json` shows the new
   gate's score.
6. Commit + push to `dev` together with the gate code.

## Local Replay (Developer)

```bash
# Full pipeline (all 5 stages, fail-fast)
python3 scripts/caas_runner.py --json -o caas_local.json

# Individual stages
python3 scripts/audit_test_quality_scanner.py --json -o caas_scanner.json
python3 scripts/audit_gate.py --json -o caas_audit_gate.json
python3 scripts/security_autonomy_check.py --json -o caas_autonomy.json
python3 scripts/external_audit_bundle.py
python3 scripts/verify_external_audit_bundle.py --json -o caas_bundle_verify.json

# Pre-push hook (one-time install)
python3 scripts/install_caas_hooks.py
```

## CI Topology

`.github/workflows/caas.yml` runs the same five stages with each stage as a
separate job, gated by `needs:` so failures short-circuit. The
`caas_report` job at the end aggregates results and writes the GitHub step
summary. The pipeline runs on every push and pull_request to `dev` and `main`,
plus on `workflow_dispatch`.

Companion workflows:

- `.github/workflows/caas-evidence-refresh.yml` — nightly bot keeping
  evidence under the 30-day SLO (H-1).
- `.github/workflows/preflight.yml` — runs Stage 0 (exploit-wiring) plus
  the lighter preflight checks on every push.

## Required Status Checks (Branch Protection)

For `dev` and `main`, the following CAAS jobs are required status checks:

- `CAAS / Static Analysis`
- `CAAS / Audit Gate`
- `CAAS / Security Autonomy`
- `CAAS / Audit Bundle`
- `CAAS / Report`

Bypass is owner-only via repo ruleset disable/enable as documented in
`AGENTS.md`.

## Failure Modes Index

| Symptom | Most likely cause | Fix |
|---------|-------------------|-----|
| Stage 1 fail with finding count > 0 | Vacuous test, polarity bug, ignored return | Fix the audit/test_*.cpp |
| Stage 2 P6 freshness WARN | Source modified after graph build | `python3 scripts/build_project_graph.py --rebuild` |
| Stage 3 score = 90 | `audit_sla` sub-gate failed | `python3 scripts/export_assurance.py -o ../../assurance_report.json` |
| Stage 3 incident_drills WARN | Drill cadence exceeded | `python3 scripts/incident_drills.py --record-all` |
| Stage 5 bundle digest mismatch | Manual edit of bundle file | Re-run Stage 4 |

## Done Criteria For The Protocol Itself

- [x] All 5 stages implemented and CI-enforced.
- [x] Local replay works for every stage.
- [x] Per-stage artifacts retained 30–90 days.
- [x] Adding a new gate has a documented procedure (this file).
- [ ] Evidence age cannot silently fail the pipeline (requires H-1 deployed
      and observed for one full month).
- [ ] Audit dashboard exists (H-9) so non-machine consumers can read status.

See [docs/CAAS_HARDENING_TODO.md](CAAS_HARDENING_TODO.md) for the open
hardening backlog.
