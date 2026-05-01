# External Audit Automation

This document describes how UltrafastSecp256k1 automates as much of the
external-auditor workflow as possible before a human reviewer arrives.

The objective is not to replace auditors.
The objective is to make their first hour productive instead of administrative.

---

## What Is Automated

The repository now has a single auditor-prep entry point:

```bash
bash ci/external_audit_prep.sh
```

That command can:

1. Rebuild the project graph
2. Run hard-fail preflight checks
3. Run assurance-document validation
4. Export machine-readable assurance status
5. Generate traceability artifacts
6. Optionally generate a full audit evidence package
7. Run security autonomy gates (formal invariants, SLA, supply chain, misuse resistance)

---

## Why This Exists

An external auditor should not need to manually reconstruct:

1. Which claims are current
2. Which checks are hard gates vs advisory
3. Which artifacts are reproducible
4. Which docs are canonical
5. Which evidence bundle to start from

This automation reduces that setup cost.

---

## Local Usage

Minimal run:

```bash
bash ci/external_audit_prep.sh
```

Include a full audit evidence package:

```bash
bash ci/external_audit_prep.sh --with-package --build-dir build-audit
```

Use an existing build for traceability:

```bash
bash ci/external_audit_prep.sh --traceability-build build
```

Skip graph rebuild when the graph is already fresh:

```bash
bash ci/external_audit_prep.sh --skip-graph
```

---

## Output Bundle

The script writes an `external-audit-prep-<timestamp>` directory containing:

1. `assurance_report.json`
2. `assurance_claims.json`
3. `validate_assurance.json`
4. `traceability_report.json`
5. `traceability_summary.txt`
6. `logs/`
7. `full_audit_package/` when `--with-package` is used

This gives auditors one directory to start from instead of many disconnected commands.

---

## GitHub Workflow

The repository also provides a manual GitHub Actions entry point:

- `.github/workflows/auditor-prep.yml`

This workflow prepares the same artifact bundle and uploads it as a workflow artifact.

Use it when:

1. You want a clean runner-produced evidence bundle
2. You need to hand an auditor a single downloadable package
3. You want to compare local and CI-generated preparation output

---

## What Remains Human Work

Automation does not remove the need for human review.
It does not answer:

1. Whether the threat model is complete
2. Whether the chosen guarantees are sufficient for a target deployment
3. Whether the code is free of unknown classes of bugs
4. Whether the public narrative should be changed more conservatively

What it does is eliminate avoidable setup and evidence-discovery friction.

---

## Canonical Auditor Start Order

For a reviewer arriving cold, the recommended order is:

1. `AUDIT_GUIDE.md`
2. `docs/ASSURANCE_LEDGER.md`
3. `docs/AUDIT_TRACEABILITY.md`
4. `docs/AI_AUDIT_PROTOCOL.md`
5. `docs/FORTRESS_ROADMAP.md`
6. `docs/SECURITY_AUTONOMY_PLAN.md`
7. `external-audit-prep-<timestamp>/`

---

## Next Automation Targets

The next useful extensions are:

1. Machine-readable claim IDs linked back to the assurance ledger
2. Preflight checks that detect public-claim drift automatically
3. Logged AI-review events tied to resulting tests/docs/gates
4. Stronger GPU-specific auditor bundles and promotion criteria

Note: Items 1–3 are partially addressed by the Security Autonomy Program
(commit 0624390c) — formal invariant specs, evidence governance chain, and
audit SLA enforcement. See `docs/SECURITY_AUTONOMY_PLAN.md`.