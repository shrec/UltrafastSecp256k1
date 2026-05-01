# Reproducible Evidence Bundle Spec

Machine-verifiable evidence contract for independent reviewers.

## Goal

Provide an auditor with a deterministic, integrity-checked package that can be
verified without trusting repository maintainers' manual summaries.

## Artifacts

- docs/EXTERNAL_AUDIT_BUNDLE.json
- docs/EXTERNAL_AUDIT_BUNDLE.sha256

## Producer

Run:

```bash
python3 ci/external_audit_bundle.py
```

This executes critical gates, captures command-output hashes, snapshots commit
metadata, and records SHA-256 hashes for required evidence files.

## Verifier

Run:

```bash
python3 ci/verify_external_audit_bundle.py --json
```

Optional strict replay mode:

```bash
python3 ci/verify_external_audit_bundle.py --replay-commands --json
```

Replay mode re-runs bundled gate commands and verifies return codes and output
hashes match the original bundle.

## Required Evidence Set

The bundle currently pins these files (SHA-256):

- docs/AUDIT_MANIFEST.md
- docs/SELF_AUDIT_FAILURE_MATRIX.md
- docs/AUDIT_SLA.json
- docs/FORMAL_INVARIANTS_SPEC.json
- docs/FEATURE_ASSURANCE_LEDGER.md
- docs/TEST_MATRIX.md
- docs/CT_VERIFICATION.md
- docs/FFI_HOSTILE_CALLER.md
- docs/BACKEND_ASSURANCE_MATRIX.md
- docs/SECURITY_AUTONOMY_KPI.json
- mutation_kill_report.json

## Gate Commands Embedded In Bundle

- python3 ci/audit_gate.py --json
- python3 ci/audit_gap_report.py --json --strict
- python3 ci/security_autonomy_check.py --json
- python3 ci/supply_chain_gate.py --json

## Fail-Closed Rules

Bundle generation or verification must fail if any condition is true:

1. Any required gate command fails
2. Any required evidence file is missing
3. Any evidence file hash mismatches
4. Detached bundle digest mismatches
5. Commit mismatch (unless verifier explicitly allows it)

## Auditor Workflow

1. Check detached digest file against bundle JSON
2. Verify bundle with verify_external_audit_bundle.py
3. Optionally replay gate commands for full reproducibility
4. Accept only when verification reports overall_pass = true
