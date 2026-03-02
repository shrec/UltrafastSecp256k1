# P0 Hardening -- Execution Summary

**Date:** 2026-03-03
**Branch:** (to be created)
**Phase:** CPU-first, feature freeze, no behavior changes

---

## Changes Made

### Wave 1 -- Stop the Bleeding (fail-open -> fail-closed)

#### release.yml
- **Cosign signing:** Changed from warning-only to hard-fail. SHA256SUMS signature MUST succeed, and verification runs immediately after signing. Individual artifact signing also hard-fails.
- **Linux ARM64 test:** Removed `|| echo "Some tests may skip"` swallowing pattern. Test failures now propagate.

#### ct-verif.yml
- **Fallback IR analysis:** Changed from `exit 0` (always pass) to `exit 1` on violations. Switch statements and variable GEPs in CT code now cause job failure.
- **ct-verif tool build failure:** Kept non-blocking (tool availability), but fallback analysis is now enforcing.

#### security-audit.yml
- **Valgrind:** Removed `|| true` from `ctest` invocation. CTest failures now propagate to the log-parsing step.
- **dudect:** Documented as intentionally advisory (statistical test, unreliable on shared runners). Deterministic CT checks (ct-verif, valgrind-ct) are the blocking gates.

#### audit-report.yml
- **Linux GCC audit runner:** Removed `|| true` -- audit runner failure now fails the job.
- **Linux Clang audit runner:** Removed `|| true` -- same.
- **Windows MSVC audit runner:** Changed `Write-Warning` to `Write-Error` + `exit $LASTEXITCODE` -- non-zero exit now fails the job.
- **Verdict job:** Added enforcement logic. If any platform verdict is not PASS (or report is missing), the verdict job fails with `exit 1`.

### Wave 2 -- Enforce Gates

#### bench-regression.yml
- **PR path:** Removed `continue-on-error: true`. Performance regression on PR now blocks the merge.
- Comment updated: "alert + block on regression" (was "alert only").

#### parse_benchmark.py
- **Empty parse result:** Changed from creating a dummy entry (which hides regressions) to hard failure with diagnostic output.

### Infrastructure

#### scripts/update_required_checks.sh (NEW)
- Script to update GitHub branch protection required status checks via API.
- Adds: security-audit, ct-verif, Benchmark Regression Check, CodeQL to blocking checks.
- Must be run by admin after branch protection is configured.

### Documentation (NEW)

#### docs/reports/dead_code_inventory.md
- Comprehensive inventory: 16 orphaned source files, 91 build directories, ~235 temp files, ~400 MB artifacts.
- Prioritized cleanup plan with safety rules.

#### docs/reports/local_ci_parity_linux.md
- Maps all 22 GH Actions workflow jobs to local Docker CI coverage.
- Identifies 3 gaps: ct-verif, benchmark regression baseline, Android NDK.
- Defines criticality tiers, artifact expectations, fail policy.

---

## Execution Board (Updated)

| Priority | Item | Status |
|----------|------|--------|
| P0 | `release.yml` fail-open -> fail-closed | **Done** |
| P0 | signing/verification strict blocking | **Done** |
| P0 | `ct-verif.yml` advisory -> blocking | **Done** |
| P0 | `security-audit.yml` fail-open closure | **Done** |
| P0 | `audit-report.yml` fail-open closure | **Done** |
| P0 | `bench-regression.yml` PR gate strict | **Done** |
| P0 | `parse_benchmark.py` dummy entry removal | **Done** |
| P0 | required checks sync script | **Done** (needs admin to run) |
| P0 | Dead/junk inventory draft | **Done** |
| P0 | Local Docker CI parity definition | **Done** |
| P1 | root `CMakeLists.txt` safe default baseline | **Done** (`2a42775`) |
| P1 | `CMakePresets.json` secure canonical preset | **Done** (`2a42775`) |
| P1 | Benchmark naming harmonization | **Done** (`2a42775`) |
| P2 | Dead code cleanup (orphans + artifacts) | **Done** (`3fc0d3a`, `1af0717`) |
| P2 | CODEOWNERS reinforcement | **Done** (`3fc0d3a`) |
| P2 | docs/version sync | **Done** (`2a42775`) |
| CT | musig2_partial_sign timing leak fix | **Done** (`e16247b`) |
| CT | schnorr_pubkey + keypair_create CT fix | **Done** (`3ccf84f`) |
| CT | Batch CT-harden 8 modules (17 sites) | **Done** (`67da5fb`) |

---

## Files Changed

```
.github/workflows/release.yml          -- cosign hard-fail, ARM64 test hard-fail
.github/workflows/ct-verif.yml         -- fallback IR analysis blocks on violations
.github/workflows/security-audit.yml   -- valgrind || true removed, dudect documented
.github/workflows/audit-report.yml     -- || true removed (3 places), verdict enforcing
.github/workflows/bench-regression.yml -- continue-on-error removed on PR path
.github/scripts/parse_benchmark.py     -- dummy entry -> hard failure
scripts/update_required_checks.sh      -- NEW: required checks update script
docs/reports/dead_code_inventory.md    -- NEW: orphaned/dead code inventory
docs/reports/local_ci_parity_linux.md  -- NEW: GH Actions vs Docker CI parity
docs/reports/p0_execution_summary.md   -- NEW: this file
```

---

## Verification

To verify these changes:
1. Push to dev branch and observe CI behavior
2. For signing: tag a test release and verify cosign failure = workflow failure
3. For ct-verif: push code with a switch in CT path and verify job fails
4. For bench-regression: create a PR with intentional regression and verify it blocks
5. For audit-report: if audit runner returns non-zero, verify workflow fails
6. Run `scripts/update_required_checks.sh` with admin access to update branch protection

---

## Remaining Risk

| Risk | Status | Notes |
|------|--------|-------|
| Release signing fail-open | **CLOSED** | Cosign now hard-fails |
| CT-verif fallback advisory | **CLOSED** | Violations now block |
| PR perf gate non-blocking | **CLOSED** | continue-on-error removed |
| Speed-first root build defaults | **CLOSED** (P1) | SPEED_FIRST=OFF default, safe preset canonical |
| Variable-time secret-key scalar_mul | **CLOSED** | 20 call sites migrated to ct:: path |
| Single-owner governance | **OPEN** (P2) | CODEOWNERS has @shrec only |
| Docs version drift | **OPEN** (P2) | SECURITY/THREAT_MODEL/AUDIT_REPORT |
