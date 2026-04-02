# CI Enforcement Status

**Last updated**: 2026-03-15

## Enforcement Model

This project does not rely on post-hoc audits.

All security, correctness, and performance checks are enforced in CI.
If a change violates any constraint, it is rejected automatically.

```
Commit → CI → Audit → PASS → Merge
                ↓
              FAIL → Block
```

**Core Rule:** If CI fails — the change does not exist.

This means:
- A failing exploit test blocks the merge
- A performance regression blocks the merge
- A CT verification failure blocks the merge
- A static analysis violation blocks the merge

Security is not reviewed after the fact — it is enforced at every commit.

---

Maps every CI workflow to its enforcement level: merge-blocking, advisory, or nightly-only.

---

## Merge-Blocking Workflows

These workflows MUST pass before code merges to `main` or `dev`.

| Workflow | Trigger | Enforcement Mechanism | Status |
|----------|---------|----------------------|--------|
| `ci.yml` | push/PR main,dev | Required status check (build, ci, test) | ACTIVE |
| `clang-tidy.yml` | push/PR (all paths) | Required status check (clang-tidy) on dev | ACTIVE |
| `cppcheck.yml` | push/PR (all paths) | Required status check (cppcheck) on dev | ACTIVE |
| `ct-verif.yml` | push/PR main,dev | Required status check (ct-verif) on dev | ACTIVE |
| `security-audit.yml` | push/PR, weekly | Required status check (asan+ubsan, werror) on dev | ACTIVE |
| `bench-regression.yml` | push main/dev, PR main | `fail-on-alert: true` (150% threshold) | ACTIVE |
| `codeql.yml` | push/PR, weekly | Required status check (codeql) on dev | ACTIVE |

### Enforcement Details

- **main branch**: 2 required reviewers, signed commits, squash-only. Checks: build, ci, test.
- **dev branch**: 8 required status checks: gcc-13, clang-17, asan+ubsan, codeql, ct-verif, werror, clang-tidy, cppcheck.
- **bench-regression**: Not a required check (path-filtered, may not run on every PR), but `fail-on-alert: true` blocks PR when triggered.

---

## Advisory Workflows

Run automatically but do not block merges. Failures are reviewed manually.

| Workflow | Trigger | What it checks | Fail behavior |
|----------|---------|---------------|---------------|
| `audit-report.yml` | Weekly + manual + release tags | 3-platform unified_audit_runner | Verdict job exits 1 on FAIL, but not a required check |
| `sonarcloud.yml` | push/PR | Code quality, coverage | Informational |
| `scorecard.yml` | push main, weekly | OpenSSF Scorecard | Informational |
| `dependency-review.yml` | PR | Vulnerable dependencies | Warns, does not block |

### audit-report.yml Enforcement

The verdict job is **fail-closed**: it aggregates JSON reports from 3 platforms (Linux GCC, Linux Clang, Windows MSVC) and exits 1 if any verdict != PASS/AUDIT-READY. However, this workflow runs weekly/manually/on-release, not on every push/PR, so it cannot be a required merge check. The audit correctness gate for daily development is provided by the `security-audit.yml` workflow (ASan + UBSan + audit subset).

---

## Nightly/Periodic Workflows

| Workflow | Schedule | Purpose |
|----------|----------|---------|
| `nightly.yml` | Daily 03:00 UTC | Extended differential + 30min dudect |
| `ct-arm64.yml` | Push, PR, daily 04:00 | Native M1 dudect timing |
| `valgrind-ct.yml` | push main/dev | Valgrind taint tracking |
| `cflite.yml` | push, PR, nightly | ClusterFuzzLite |
| `mutation.yml` | Weekly Sun 05:00 | Mutation testing |

---

## Release/Publishing Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `release.yml` | Tags, manual | Multi-platform release artifacts |
| `packaging.yml` | Release tags | .deb/.rpm packaging |
| `benchmark.yml` | push | gh-pages performance tracking |
| `docs.yml` | push | Doxygen to GitHub Pages |
| `bindings.yml` | push/PR | 12-language compile check |
| `auditor-prep.yml` | manual | Auditor-facing evidence bundle artifact |

---

## Local CI Enforcement

Local Docker CI (`docker/run_ci.sh`) mirrors the above tiers:

| Local Mode | Mirrors | When to run |
|------------|---------|-------------|
| `quick` | ci.yml (gcc + clang Release) | Before every commit |
| `pre-push` | ci.yml + ASan + clang-tidy | Before every push |
| `dev-gate` | All 8 dev-protection checks | Before PR to dev |
| `main-gate` | All main-protection checks | Before PR to main |
| `gh-parity` | Full GH Actions matrix | Before release |

### Local Fail-Open Patterns (Documented)

The following `|| true` patterns exist in local scripts. They are mitigated by post-hoc verdict checking but not ideal:

| File | Pattern | Mitigation |
|------|---------|------------|
| `docker/run_ci.sh:602,663` | `unified_audit_runner ... \|\| true` | JSON verdict checked via jq afterward |
| `scripts/local-ci.sh:432` | `unified_audit_runner ... \|\| true` | JSON verdict checked via grep afterward |
| `scripts/local-ci.sh:184` | `ctest -T MemCheck \|\| true` | Valgrind log content parsed afterward |

These are acceptable for local development (the exit code swallowing prevents noisy false-fail on segfault during memcheck), but the verdict/log parsing afterward ensures real failures are caught.

---

## Flaky Test Registry

Tests that may produce non-deterministic results on shared/virtualized CI runners:

| Test | Cause | Policy |
|------|-------|--------|
| `ct_sidechannel` | Statistical timing (dudect, 600s) | Advisory on shared CI, strict on dedicated hardware |
| `ct_sidechannel_smoke` | Statistical timing (120s, CI-safe subset) | Advisory on shared CI |
| Benchmark regression | CPU frequency scaling, neighbor noise | 150% threshold (generous) on shared runners |

---

## Assurance Expansion In Progress

The following assurance-system hardening work is now tracked in-repo:

| Artifact | Purpose |
|----------|---------|
| `docs/ASSURANCE_LEDGER.md` | Canonical claim-to-evidence map |
| `docs/AI_AUDIT_PROTOCOL.md` | Governs AI-assisted adversarial review |
| `docs/AI_REVIEW_EVENTS.json` | Machine-readable record of accepted/rejected AI review findings |
| `docs/GPU_BACKEND_EVIDENCE.json` | Machine-readable GPU backend status, publishability, and artifact requirements |
| `docs/FORTRESS_ROADMAP.md` | Tracks the remaining fortress-grade assurance gap |

All other tests are deterministic and must always pass.

The AI review-event log is enforced locally through:

- `python3 scripts/preflight.py --ai-review`

and schema-validated through:

- `python3 scripts/validate_assurance.py`

GPU backend publishability is enforced locally through:

- `python3 scripts/preflight.py --gpu-evidence`

and schema/policy-validated through:

- `python3 scripts/validate_assurance.py`
