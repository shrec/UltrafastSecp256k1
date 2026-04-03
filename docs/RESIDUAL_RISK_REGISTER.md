# Residual Risk Register

Tracked residual risks and intentional deferrals for the current assurance
state.

This register is deliberately narrow: it records named follow-up risks without
inflating them into blocking findings when the current owner-grade bundle does
not classify them as blockers.

Current verified state:

1. `build/owner_audit/owner_audit_bundle.json` reports no blocking residual gaps.
2. The entries below are non-blocking follow-up risks or intentional deferrals.

---

## Active Entries

| ID | Class | Status | Owner | Notes |
|----|-------|--------|-------|-------|
| RR-001 | Constant-time evidence confidence ceiling | Accepted, non-blocking | Audit/tooling | Deterministic CT aggregation exists, but some CT claims still retain formal/manual follow-up nuance. See `docs/SELF_AUDIT_FAILURE_MATRIX.md` constant-time row and `artifacts/ct/` evidence summaries. |
| RR-002 | Local vs GitHub-native workflow parity | Accepted, non-blocking | Infra/audit | Local parity improved materially, but some workflow services remain GitHub-hosted by nature. See `docs/SELF_AUDIT_FAILURE_MATRIX.md` supply-chain / workflow drift row. |
| RR-003 | ROCm/HIP real-device evidence | Intentionally deferred | GPU/backend | AMD hardware-backed evidence is not yet part of the claimed audit surface. This is explicitly deferred rather than hidden. See `docs/GPU_BACKEND_EVIDENCE.json` and `docs/SELF_AUDIT_FAILURE_MATRIX.md` optional backend expansion row. |
| RR-004 | ECDSA large-x r comparison (Stark Bank CVE class) | **CLOSED 2026-04-03** | ECDSA/verify | `ecdsa_verify` `r_less_than_pmn` used wrong PMN constants — signatures with k·G.x ∈ [n, p-1] (~2^−128 per sig) were erroneously rejected. Fixed in `cpu/src/ecdsa.cpp` (FE52 + 4x64 paths). Regressed by Wycheproof tcId 346. Commit `ea8cfb3c`. |

---

## Review Rule

When a residual risk becomes blocking, partially covered, or fully closed:

1. Update the matching entry here.
2. Update `docs/SELF_AUDIT_FAILURE_MATRIX.md` if the failure-class status changes.
3. Record the change in `docs/AUDIT_CHANGELOG.md`.

If a new owner-grade blocker appears, it should be reflected both here and in
the owner audit bundle rather than existing only as narrative commentary.