# Audit Changelog

Focused changelog for changes to the assurance system itself.

This file is not a release changelog. It records audit maturity changes,
evidence upgrades, and changes to what the repository can honestly claim.

---

## 2026-03-27

- Synchronized `docs/TEST_MATRIX.md` summary language with the current
  assurance validator so the top-level target count no longer points at a
  stale 71-target snapshot.
- Added `docs/RESIDUAL_RISK_REGISTER.md` to name the remaining non-blocking
  residual risks and intentional deferrals in one canonical place.
- Marked the originally started owner-grade audit track as closed in
  `docs/OWNER_GRADE_AUDIT_TODO.md`; remaining entries are future hardening,
  not blockers for the completed audit closure.

## 2026-03-26

- Closed the remaining owner-grade audit blockers by finishing ABI
  hostile-caller coverage, CT route documentation, failure-matrix promotion,
  and owner bundle generation.
- `build/owner_audit/owner_audit_bundle.json` reached `overall_status: ready`.
- `scripts/audit_gate.py` passed with no blocking findings for the started
  owner-grade audit track.

## 2026-03-25

- Promoted the failure-class matrix into an executable audit gate surface.
- Added owner-grade visibility for ABI hostile-caller coverage and
  secret-path change control.

## 2026-03-23

- Established the owner-grade audit gate and assurance-manifest workflow.
- Expanded graph and assurance validation so ABI and GPU surfaces were no
  longer invisible to the audit tooling.