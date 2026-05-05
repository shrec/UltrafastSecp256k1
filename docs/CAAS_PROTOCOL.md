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

## Drift Policy

| Drift class | Stage that catches it | Disposition |
|-------------|----------------------|-------------|
| Evidence age past 30d | Stage 3 (`audit_sla`) | Auto-fixed by nightly refresh bot (H-1) |
| Test added but not wired | Stage 1 pre-step | Block PR |
| GPU op added on one backend only | Stage 2 (P7) | Block PR |
| ABI changed without doc | Stage 2 (P10) | Block PR |
| Bundle SHA mismatch | Stage 5 | Block PR |
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
| Stage 3 score = 90 | `audit_sla` sub-gate failed | `python3 ci/export_assurance.py -o ../../assurance_report.json` |
| Stage 3 incident_drills WARN | Drill cadence exceeded | `python3 ci/incident_drills.py --record-all` |
| Stage 5 bundle digest mismatch | Manual edit of bundle file | Re-run Stage 4 |

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
