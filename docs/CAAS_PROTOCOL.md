# CAAS — Continuous Audit as a Service Protocol

> Version: 1.0 — 2026-04-21

## Purpose

Every push and pull request to `dev` and `main` runs the block-based `gate.yml` pipeline. Release tags run the full release CAAS gate before build/package fan-out. Manual deep-assurance runs are a supplement, never a substitute for the required gate.

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

## Stage Contracts

### Stage 1 — Static Analysis

| Field | Value |
|-------|-------|
| Tool | `ci/audit_test_quality_scanner.py` |
| Pre-step | `ci/check_exploit_wiring.py` (Stage 0 wiring gate) |
| Output | `caas_scanner.json` — `total_findings`, `findings[]` |
| Pass condition | `total_findings == 0` |
| Local replay | `python3 ci/audit_test_quality_scanner.py --json -o caas_scanner.json` |
| Artifact retention | 30 days |

### Stage 2 — Audit Gate

| Field | Value |
|-------|-------|
| Tool | `ci/audit_gate.py` |
| Pre-deps | project graph, source graph, `ufsecp_shared` built |
| Output | `caas_audit_gate.json` — `verdict`, `checks[]` |
| Pass condition | `verdict in {"PASS", "PASS with advisory"}` |
| Principles | P0, P0a–P0d, P1–P11 |
| Local replay | `python3 ci/audit_gate.py --json -o caas_audit_gate.json` |
| Artifact retention | 30 days |

### Stage 3 — Security Autonomy

| Field | Value |
|-------|-------|
| Tool | `ci/security_autonomy_check.py` |
| Output | `caas_autonomy.json` — `autonomy_score` (0–100), `autonomy_ready` |
| Pass condition | `autonomy_score == 100` and `autonomy_ready == true` |
| Principles | P12–P18 |
| Local replay | `python3 ci/security_autonomy_check.py --json -o caas_autonomy.json` |
| Artifact retention | 30 days |

### Stage 4 — Bundle Produce

| Field | Value |
|-------|-------|
| Tool | `ci/external_audit_bundle.py` |
| Output | `docs/EXTERNAL_AUDIT_BUNDLE.json`, `docs/EXTERNAL_AUDIT_BUNDLE.sha256` |
| Principles | P19 |
| Local replay | `python3 ci/external_audit_bundle.py` |
| Artifact retention | 90 days |

### Stage 5 — Bundle Verify

| Field | Value |
|-------|-------|
| Tool | `ci/verify_external_audit_bundle.py` |
| Output | `caas_bundle_verify.json` — `overall_pass`, `checks[]` |
| Pass condition | `overall_pass == true` |
| Local replay | `python3 ci/verify_external_audit_bundle.py --json` |
| Deep replay | `python3 ci/verify_external_audit_bundle.py --replay-commands --json` |
| Artifact retention | 90 days |

PR/push CAAS also produces `out/current_audit/EXTERNAL_AUDIT_BUNDLE.{json,sha256}`
inside the protected workflow and verifies that current-run bundle without
`--allow-commit-mismatch`. The committed `docs/EXTERNAL_AUDIT_BUNDLE.*` files
are retained as a historical baseline and are hash-checked separately.

## Drift Policy

| Drift class | Stage that catches it | Disposition |
|-------------|----------------------|-------------|
| Evidence age past 30d | Stage 3 (`audit_sla`) | Auto-fixed by nightly refresh bot (H-1) |
| Test added but not wired | Stage 1 pre-step | Block PR |
| GPU op added on one backend only | Stage 2 (P7) | Block PR |
| ABI changed without doc | Stage 2 (P10) | Block PR |
| Current-run bundle SHA mismatch | Stage 5 | Block PR |
| Current-run bundle commit != current `HEAD` | PR-push bundle verify / Stage 5 | Block PR; regenerate current-run bundle |
| Committed bundle SHA mismatch | PR-push static bundle baseline verify | Block PR; committed baseline was tampered or corrupted |
| Source graph DB stale or built for another commit | Source Graph Quality Gate | Block PR; rebuild `tools/source_graph_kit/source_graph.db` |
| Source graph focus no longer routes CAAS terms to CAAS files | Source Graph Quality Gate | Block PR; fix graph config/ranking |
| Autonomy score < 100 | Stage 3 | Block PR |

## Adding a New Gate

1. Implement in `ci/` with `--json -o` output and exit 0/1.
2. Register in `ci/audit_gate.py` (P0–P11) or `ci/security_autonomy_check.py` (P12–P18).
3. Document the principle in `AUDIT_MANIFEST.md`.
4. Verify: `python3 ci/caas_runner.py --json` shows the new gate.
5. Commit gate code + docs together.

## Local Replay

```bash
# Full pipeline
python3 ci/caas_runner.py --json -o caas_local.json

# Individual stages
python3 ci/audit_test_quality_scanner.py --json -o caas_scanner.json
python3 ci/audit_gate.py --json -o caas_audit_gate.json
python3 ci/security_autonomy_check.py --json -o caas_autonomy.json
python3 ci/external_audit_bundle.py
python3 ci/verify_external_audit_bundle.py --json -o caas_bundle_verify.json
```

## Local Reviewer Web Panel

`ci/caas_serve.py` provides an interactive artifact browser for local review sessions.

```bash
python3 ci/caas_serve.py           # binds to 127.0.0.1:8080 (safe default)
python3 ci/caas_serve.py --port 9090
python3 ci/caas_serve.py --lan     # binds to 0.0.0.0 — LAN-accessible
```

> **Security:** `caas_serve.py` is a local reviewer tool. The default bind (`127.0.0.1`)
> is safe. Do not use `--lan` or `--bind 0.0.0.0` on untrusted networks.
> Artifacts may contain local paths, environment metadata, logs, and private evidence.

The panel serves the CAAS dashboard and all files under `docs/` and `out/` artifact roots.
It does not require a build — it reads already-generated artifacts.

## Required Status Checks

- `CAAS / Static Analysis`
- `CAAS / Audit Gate`
- `CAAS / Security Autonomy`
- `CAAS / Audit Bundle`
- `CAAS / Report`

## Failure Modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Stage 1 findings > 0 | Vacuous test, polarity bug, ignored return | Fix the audit/test_*.cpp |
| Stage 2 P6 freshness WARN | Source modified after graph build | `python3 ci/build_project_graph.py --rebuild` |
| Stage 3 score = 90 | `audit_sla` sub-gate failed (stale/missing evidence) | `python3 ci/audit_sla_check.py` to see which artifact + `days_until_block`; refresh per the Evidence Freshness & Refresh Contract (e.g. `ci/export_assurance.py -o ../../assurance_report.json`, or re-snapshot `audit/ci-evidence`) |
| Stage 3 incident_drills WARN | Drill cadence exceeded (stale `docs/INCIDENT_DRILL_LOG.json`) | `python3 ci/incident_drills.py` (each run injects faults, asserts gate detection, and rewrites the drill log) |
| Stage 5 bundle digest mismatch | Manual edit of bundle file | Re-run Stage 4 |

## Evidence Freshness & Refresh Contract

CAAS evidence is fail-closed on age: `ci/audit_sla_check.py` blocks the release
gate when a critical artifact crosses its SLO threshold (`docs/AUDIT_SLA.json`).
To prevent a silent green→blocked jump, the checker (Bastion B3) reports
`days_until_block` for every tracked artifact and emits a non-blocking
**PRE-ALERT** warning while an artifact is within its `pre_alert_buffer_days`
window. Inspect runway any time with:

```bash
python3 ci/audit_sla_check.py          # human table with days_until_block
python3 ci/audit_sla_check.py --json    # min_days_until_block + evidence_status[]
```

| Critical evidence | SLO (block / pre-alert) | Refreshed by |
|-------------------|--------------------------|--------------|
| `assurance_report.json` (suite) | 30d / 25d | `caas-evidence-refresh.yml` (nightly 04:30 UTC) |
| `docs/EXTERNAL_AUDIT_BUNDLE.*` | n/a (baseline) | `caas-evidence-refresh.yml` (nightly) |
| `docs/SECURITY_AUTONOMY_KPI.json` | — | `caas-evidence-refresh.yml` (nightly) |
| `docs/AUDIT_DASHBOARD.md` | — | `caas-evidence-refresh.yml` (nightly, H-9) |
| `docs/DETERMINISM_GOLDEN.json` | 30d / 25d | determinism gate (`ci/check_determinism_gate.py`) when stale |
| `audit/ci-evidence/*` (CT/adversarial/fuzz) | 14d / 10d | **owner/manual chore**: build + run the standalone audit binaries (`adversarial_protocol`, `ecies_regression`, `fuzz_parsers`, `fuzz_address_bip32_ffi`) and commit dated snapshots. *Automation candidate — not yet scheduled.* |
| `docs/API_SECURITY_CONTRACTS.json` | 14d / 10d | re-validated by `ci/check_api_contracts.py`; refreshed when the API changes. *Automation candidate — not yet scheduled.* |

**Refresh discipline.** When `audit_sla_check.py` reports a PRE-ALERT (or a small
`min_days_until_block`), refresh the named artifact **before** it blocks. The two
14-day critical artifacts (`audit/ci-evidence`, `API_SECURITY_CONTRACTS.json`) are
not yet covered by a scheduled workflow; until they are, the pre-alert is the
early-warning signal that the owner/manual refresh is due. Refreshed evidence
must be captured from a real run — never hand-edited to move a timestamp.

## Product Profiles

```bash
python3 ci/caas_runner.py --profile <profile-id> --auditor-mode
```

| Profile ID | Description |
|---|---|
| `bitcoin-core-backend` | CPU + libsecp256k1 shim for Bitcoin Core secondary backend |
| `cpu-signing` | CPU ECDSA/Schnorr/CT layer standalone |
| `ffi-bindings` | Legacy C API + language bindings |
| `wasm` | WebAssembly browser/Node binding |
| `gpu-public-data` | GPU batch verify + BIP-352 scan (public data) |
| `bchn-compat` | Bitcoin Cash Node legacy Schnorr shim |
| `release/full-engine` | All surfaces, release gate |
