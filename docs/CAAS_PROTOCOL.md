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

## Portability of the Model

CAAS is first an *idea and a methodology*, second an implementation. The
specific pipeline shipped in this repository is tailored to a
cryptographic engine — exploit PoCs, constant-time pipelines, GPU
backends, formal-property tests — but the underlying principle is
domain-agnostic:

> Every claim made about the project must be expressed as an executable
> test, every test must run on every commit, every result must be
> reproducible by an independent third party, and every fixed bug must
> become a permanent regression test.

Any open-source project — wallet, compiler, parser, networking stack,
embedded firmware, web framework — can adopt the same protocol by
keeping the principle constant and re-instantiating the pipeline stages
with tooling appropriate to its domain. A wallet's CAAS pipeline would
swap exploit PoCs for transaction-construction differential tests; a
parser's CAAS pipeline would swap GPU backends for fuzzing harnesses; a
web framework's CAAS pipeline would swap CT verification for HTTP
state-machine conformance tests. The five-stage skeleton (static
analysis → audit gate → autonomy check → bundle produce → bundle
verify) and the two founding questions stay the same.

This document describes the secp256k1 instantiation. The methodology
itself is offered to the wider open-source community as a portable
pattern — adopt it, fork it, adapt it.

## Developer Benefit (Not Just Audit)

CAAS is misread if it is treated only as an audit mechanism. In daily
practice it functions as a **developer safety net** that pays for
itself long before any external auditor looks at the code. The pipeline
catches regressions, platform-specific compiler behaviour, and silent
correctness drift the moment they are introduced — not weeks later when
a user files a bug.

A concrete example from this project: porting the constant-time pipeline
to **RISC-V**. The first build on the new target compiled and passed
functional tests, but the CAAS constant-time stage immediately reported
a **secret-dependent timing leak** — the RISC-V compiler had
re-optimised an arithmetic sequence in a way that re-introduced a data
dependency on a secret. Without CAAS this would have shipped as a
"working RISC-V port" and the leak would have been invisible until
someone ran a side-channel measurement months later. With CAAS the same
commit that introduced the port also surfaced the bug, and the fix
landed in the same session.

This is the practical leverage CAAS gives a small team or a solo
maintainer:

- **Hours, not months, to bring up a new platform.** RISC-V, ARM64,
  Android, iOS, WASM, ESP32, STM32 — each new target inherits the full
  CAAS suite. If anything regresses (functional, CT, GPU parity, ABI),
  the pipeline says so on the first run.
- **No "does it really work on this platform?" anxiety.** The same
  evidence bundle that validates Linux x86_64 validates every other
  target, because every target re-runs the full suite.
- **Compiler surprises caught at commit time.** Compilers vary across
  versions, targets, and optimisation levels. CAAS treats the compiled
  binary as the unit of trust, not the source code, so optimisation
  drift cannot hide.
- **Refactors stay safe.** Rewriting a hot path, swapping an algorithm,
  or restructuring a backend is bounded by the same suite — green
  pipeline means the change is functionally and security-equivalent.

In other words: CAAS is the infrastructure that makes *aggressive,
fearless development on a high-stakes codebase* sustainable for a small
team. The audit story is one consequence of that infrastructure, not
its primary purpose.

## Open-Ended and Self-Evolving

CAAS is a fully open-source pipeline, and that is structurally
significant: it is not a fixed checklist that ages, it is a **living
suite that grows with the threat landscape**. Anyone — the maintainer,
a contributor, an external researcher, a future auditor — can write a
new test, encode a freshly published attack as a PoC, add a new
invariant, or wire in a new differential oracle, and from the next
commit onward that case becomes a **permanent regression gatekeeper**.

The cycle is intentionally simple:

1. A new attack class, exploit technique, or correctness concern is
   identified — by reading a paper, by a CVE disclosure, by a code
   review, by user feedback, or by the maintainer's own intuition.
2. The case is expressed as a test (`test_<name>_run()`), wired into
   `unified_audit_runner.cpp`, registered with a section, and
   documented per the *Exploit / Audit Test Conversion Standard*.
3. From the very next push, that test runs on every commit, on every
   platform, in every CI lane — forever. Any future change that
   re-introduces the issue is rejected automatically.

This is what makes CAAS qualitatively different from a snapshot audit.
A PDF audit ends the day it is signed; CAAS *accumulates*. Every fixed
bug, every researched attack, every newly discovered edge case
strengthens the suite. The system therefore gets stronger with time
rather than weaker — the opposite of the natural decay curve of
traditional audits.

Because the entire pipeline is open-source, this self-evolution is
**not gated by the original author**. A third party who finds a new
attack vector can submit a test as a pull request; once merged it
protects the project for everyone. This turns the broader open-source
community into co-maintainers of the project's security posture, with
no central bottleneck and no proprietary tooling.

## The Repository Is the Infrastructure

A critical property of CAAS is that **the repository itself is the
distribution mechanism**. There is no separate service to install, no
SaaS account to provision, no proprietary backend to license, no
`caas-server.example.com` to depend on. The pipeline lives inside the
repo as ordinary source files, scripts, workflows, and tests.

Concretely, this means:

- **Cloning the repo clones the entire audit infrastructure.**
  `git clone` brings down `audit/`, `caas_runner.py`,
  `.github/workflows/caas.yml`, the unified audit runner, the static
  analyzer, the security-autonomy checker, the bundle producer and
  verifier, every exploit PoC, and every wired regression test — in
  one operation.
- **Forking the repo forks the audit infrastructure.** A fork inherits
  the full pipeline automatically; CI starts running the same gates on
  the fork's commits as soon as Actions are enabled.
- **Vendoring the repo vendors the audit infrastructure.** Downstream
  projects that pull the library in as a submodule, vcpkg port,
  Conan recipe, or vendored copy receive the full evidence chain — not
  a binary blob with a PDF attached.
- **Mirroring the repo mirrors the audit infrastructure.** A mirror on
  GitLab, Codeberg, Forgejo, sourcehut, or any private Git host can
  re-run the entire pipeline locally and reproduce the same evidence
  bundle, byte-for-byte, with no dependency on the upstream
  organisation, the original maintainer, or any external service.

This is the structural difference from the snapshot-PDF model: a PDF
travels separately from the code, gets stale, and may not even be
distributed with downstream copies. CAAS travels *with* the code,
updates *with* the code, and re-executes wherever the code is cloned.
The audit infrastructure has the same supply-chain footprint as the
library itself, which is the only footprint a security-critical
artifact should ever have.

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
