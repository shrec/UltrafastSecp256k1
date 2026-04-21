# CAAS Hardening TODO

> Status snapshot (2026-04-21): CAAS pipeline `caas_runner.py` returns
> `overall_pass=True`, all 5 stages green, security autonomy 100/100.
> The structural gaps below are what stops the protocol from being
> "owner-grade complete" — none of them are blocking the current gate,
> but each one is a known fragility, drift surface, or under-automated
> control that should be closed.
>
> Source-of-truth principles: [docs/AUDIT_MANIFEST.md](AUDIT_MANIFEST.md)
> (P0–P20). Source-of-truth state: [docs/AUDIT_CHANGELOG.md](AUDIT_CHANGELOG.md).

## Severity legend

| Tier | Meaning |
|------|---------|
| **P0** | Structural fragility — gate currently passes but breaks silently on time/key/state drift. |
| **P1** | Real assurance depth gap — gate passes but underlying control is weaker than the claim. |
| **P2** | Visibility / hygiene / reviewer-velocity gap — quality of audit, not correctness. |

---

## P0 — Structural Fragility

### H-1. Nightly auto-refresh of `assurance_report.json`

**Problem.** `audit_sla_check.py` enforces `max_stale_evidence_days = 30` against
the suite-root `assurance_report.json` (computed from filesystem mtime). The
file is regenerated only by hand via `scripts/export_assurance.py`. As of
2026-04-21 the file aged past 30 days and silently failed CAAS Stage 3 (audit
SLA gate dropped autonomy from 100 → 90).

**Fix.**

- Add `.github/workflows/assurance-refresh.yml` running on `schedule: cron`
  (daily, ~04:30 UTC) plus `workflow_dispatch`.
- Steps: checkout `dev`, run `export_assurance.py`, commit if changed,
  push back to `dev` with `[bot] CAAS evidence refresh`.
- Optional: also regenerate `EXTERNAL_AUDIT_BUNDLE.json` + `.sha256`,
  `SECURITY_AUTONOMY_KPI.json`.
- Optional hardening: change `audit_sla_check.py` to also accept a
  `assurance_generated_at` field inside the JSON, so freshness can be verified
  even when the file is mounted from CI artifact storage with rewritten mtime.

**Acceptance.** A file aged 25–35 days is automatically refreshed by the bot
before tripping the gate. No manual run required.

### H-2. HMAC evidence-chain key rotation policy

**Problem.** `evidence_governance.py` HMAC-verifies the evidence chain, but
the signing key has no rotation procedure, no escrow doc, no compromise
playbook. A leaked key forges the entire chain silently.

**Fix.**

- Add `docs/EVIDENCE_KEY_POLICY.md`: who holds the key, how it rotates,
  what triggers a rotation, where the escrow lives, how a compromise is
  declared, and how the chain is re-signed during rotation.
- Add `scripts/rotate_evidence_key.py`: generates a new key, re-signs the
  chain in place, atomically swaps the active key, and writes a dated
  rotation record into the chain itself.
- Add an `incident_drills.py` drill for "evidence key compromise" that
  simulates a key change.

**Acceptance.** A rotation can be performed end-to-end by following the
playbook; the drill exits 0 with a recorded rotation event.

### H-3. CAAS protocol standalone specification

**Problem.** CAAS is described in the workflow YAML and briefly in
`AUDIT_MANIFEST.md` P20, but no single doc states the protocol contract,
stage exit codes, artifact layout, retention policy, drift policy, and
owner/reviewer responsibilities.

**Fix.** Create `docs/CAAS_PROTOCOL.md` covering:

- Stage-by-stage contract (input → tool → output JSON schema → exit code).
- Required artifacts per stage and their retention.
- Drift policy: when a stage fails advisory vs blocking.
- How to add a new gate without breaking the protocol.
- Local replay command for each stage.
- Relationship to the bundle (P19) and to autonomy scoring (P12–P18).

**Acceptance.** A new auditor can re-run any single stage locally from the
spec without reading the YAML.

---

## P1 — Assurance Depth

### H-4. Mutation kill-rate weekly CI gate

**Problem.** `P2b` is currently advisory ("heavy lane only, not mandatory
push/PR check"). 75% kill-rate threshold lives only in owner-grade pre-release
runs. Arithmetic regressions can ship for weeks before the next manual run.

**Fix.**

- Add `.github/workflows/mutation-weekly.yml` (`schedule: weekly`,
  self-hosted GPU runner if available, else ubuntu-24.04 with reduced
  scope).
- Run `mutation_kill_rate.py --threshold 75 --json` against `build_rel`.
- On fail: open / update an issue (`mutation-kill-rate-regression`) instead
  of blocking — keeps `dev` push-able while making the regression visible.

**Acceptance.** If kill-rate drops below 75% an automated issue is filed
within 24h.

### H-5. GPU `schnorr_snark_witness_batch` native kernel tracking

**Problem.** Host-side fallback closes the parity gate, but no native CUDA /
OpenCL / Metal kernel exists. Performance claims for batched Schnorr SNARK
witness generation are CPU-bound on every backend.

**Fix.**

- Add a tracked entry in `docs/RESIDUAL_RISK_REGISTER.md` (RR-005) with the
  measured CPU-fallback throughput and a target kernel design note.
- Open an issue in the repository linking to the entry.
- Mark the matrix row in `docs/BACKEND_ASSURANCE_MATRIX.md` with a
  performance footnote in addition to the `Y*` correctness footnote.

**Acceptance.** The fallback is correctly characterised in the residual-risk
register as a performance gap, not a correctness gap.

### H-6. Local supply-chain verification parity

**Problem.** `OWNER_GRADE_AUDIT_TODO` P1#9 is open. Several P15 checks
(SLSA verification, dependency-review, artifact provenance) only run in
GitHub Actions. Owner-grade review wants smaller GitHub-only trust surface.

**Fix.**

- Add `scripts/local_supply_chain_check.py`: runs whatever subset of the
  P15 controls can run without GitHub-hosted services (pinned-action
  hashes from workflow YAML, local SBOM diff, local artifact hash check).
- Document the residual GitHub-only checks in
  `docs/SUPPLY_CHAIN_LOCAL_PARITY.md`.

**Acceptance.** A reviewer with no network access can still run a
meaningful local supply-chain pass.

---

## P2 — Visibility & Hygiene

### H-7. Review-queue aging SLA

**Problem.** Source graph reports 23 review-queue items (audit_gap,
high-gain-low-risk, untested_hotspot). No SLA on how long an item may sit
before being addressed or explicitly deferred.

**Fix.**

- Add `scripts/review_queue_age_check.py` that reads the graph
  `review_queue` table and the git history, computes age per entry, and
  fails if any entry exceeds the configured threshold without an explicit
  deferral note.
- Add an `AUDIT_SLA.json` SLO `review_queue_max_open_days`.

### H-8. TODO/FIXME age tracker

**Problem.** 92 TODO/FIXME comments are tracked by the graph. No SLA on
their age. Many are old PoC bug annotations that look new.

**Fix.**

- Add `scripts/todo_age_check.py` that uses `git blame` to compute the
  age of each TODO/FIXME and fails if any exceeds the SLA without a
  deferral marker.
- Add `AUDIT_SLA.json` SLO `todo_max_open_days`.

### H-9. Audit dashboard generator

**Problem.** `OWNER_GRADE_AUDIT_TODO` P2#10 is open. Trends live only inside
JSON outputs.

**Fix.**

- Add `scripts/render_audit_dashboard.py` that emits a single
  `docs/AUDIT_DASHBOARD.md` summarising failure-class status, residual
  risks, autonomy score, latest CAAS run, and trend deltas vs the previous
  7 / 30 days.
- Refresh dashboard from the same nightly bot job that refreshes
  assurance_report.json (H-1).

### H-10. Reviewer prompt templates

**Problem.** `OWNER_GRADE_AUDIT_TODO` P2#11 is open. AI-assisted review
quality varies wildly without standardised prompts.

**Fix.** Add `docs/REVIEWER_PROMPTS/` directory with:

- `auditor.md` — "find a real, exploitable bug in this diff"
- `attacker.md` — "produce a PoC for the worst-case behaviour"
- `perf_skeptic.md` — "challenge every performance claim with a measurement"
- `docs_skeptic.md` — "find a stale claim that contradicts the code"

Each prompt is graph-aware (assumes the source-graph workflow).

### H-11. ROCm/HIP smoke pipeline (deferred-but-scaffolded)

**Problem.** RR-003 is intentionally deferred, but if AMD performance is
ever quoted publicly we have no smoke evidence.

**Fix.**

- Add `.github/workflows/rocm-smoke.yml` gated behind a label
  (`run-rocm-smoke`) so it only runs when manually requested.
- Build with HIP, run a single deterministic KAT.
- Output goes to `artifacts/gpu/rocm_smoke_<date>.json`.

This is scaffolding only; promotion to claim status still requires
hardware-backed evidence per the existing residual-risk rules.

### H-12. Crypto-grounded bug-pattern enrichment in `dev_bug_scanner.py`

**Problem.** The Python developer-side scanner (`scripts/dev_bug_scanner.py`)
ships 28 generic + crypto-aware checkers. It did **not** cover several famous
real-world cryptographic bug classes (Sony PS3 ECDSA nonce reuse, Apple
goto-fail, Debian OpenSSL RNG, Bleichenbacher / DER laxness, ECDH small-subgroup,
MAC truncation, BIP-340 missing tagged_hash, etc.). A static scanner that
misses these patterns gives false confidence to contributors.

**Fix (landed 2026-04-21).** Added 13 CVE-grounded checkers, each with an
incident anchor in the comment header:

| Category | Anchor incident |
|----------|-----------------|
| `NONCE_REUSE_VAR` | Sony PS3 ECDSA, Bitcoin Android 2013 |
| `MEMCMP_SECRET` | Xbox 360 Hypervisor 2007, Java JCE 2014 |
| `MISSING_LOW_S_CHECK` | BIP-62 / segwit malleability |
| `SCALAR_FROM_RAND` | CVE-2008-0166 Debian OpenSSL |
| `GOTO_FAIL_DUPLICATE` | Apple CVE-2014-1266 |
| `POINT_NO_VALIDATION` | invalid-curve attacks (Brainpool 2015) |
| `ECDH_OUTPUT_NOT_CHECKED` | small-subgroup / twist confinement |
| `HASH_NO_DOMAIN_SEP` | BIP-340 §3.2 cross-protocol reuse |
| `DER_LAX_PARSE` | OpenSSL CVE-2014-8275 |
| `TIMING_BRANCH_ON_KEY` | Bernstein cache-timing class |
| `MAC_TRUNCATION` | GCM-trunc / JOSE alg=none class |
| `SCALAR_NOT_REDUCED` | Stark Bank ECDSA r∈[n,p−1] class |
| `PRINTF_SECRET` | mobile-wallet log-leak class |

Initial run on the codebase produced 3 MEDIUM signals (no HIGH false positives,
none in CT-sensitive paths). All checkers are scoped via `_is_crypto_path()`
or filename guards to keep noise low; each finding includes a `fix_hint`.

**Acceptance.** ✓ Done — checkers landed, registered in `CHECKERS`,
`scripts/dev_bug_scanner.py --json` runs clean against `dev`.

---

## Execution Order

Recommended by impact-per-effort:

1. H-1 (nightly auto-refresh) — kills the recurring SLA failure mode forever.
2. H-3 (CAAS spec doc) — unblocks every other doc-level audit work.
3. H-7 + H-8 (aging SLAs) — make tech-debt visible and bounded.
4. H-2 (HMAC key rotation) — closes the worst silent-forge surface.
5. H-9 (dashboard) — gives the owner a single read for posture.
6. H-10 (reviewer prompts) — improves all future review velocity.
7. H-4 (mutation weekly) — depth gain for arithmetic.
8. H-6 (local supply-chain) — depth gain for build trust.
9. H-5 (GPU witness perf) — performance-honesty hygiene.
10. H-11 (ROCm scaffold) — optional, only if AMD claims are made.

---

## Done Criteria

This file can be closed when all of:

1. CAAS pipeline cannot fail because of evidence age alone (H-1).
2. CAAS protocol is documented as a standalone contract (H-3).
3. Evidence chain has a documented key rotation procedure (H-2).
4. Every audit/test_exploit / TODO / review-queue item has an SLA-bounded
   age (H-7, H-8).
5. Mutation kill-rate is at minimum weekly CI-tracked (H-4).
6. Auditor can render a single dashboard without reading raw JSON (H-9).
