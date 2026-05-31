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
| RR-004 | ECDSA large-x r comparison (Stark Bank CVE class) | **CLOSED 2026-04-03** | ECDSA/verify | `ecdsa_verify` `r_less_than_pmn` used wrong PMN constants — signatures with k·G.x ∈ [n, p-1] (~2^−128 per sig) were erroneously rejected. Fixed in `src/cpu/src/ecdsa.cpp` (FE52 + 4x64 paths). Regressed by Wycheproof tcId 346. Commit `ea8cfb3c`. |
| RR-005 | GPU `schnorr_snark_witness_batch` performance gap | Accepted, non-blocking (performance only) | GPU/backend | Default `GpuBackend::schnorr_snark_witness_batch` virtual delegates to a host-side CPU loop (`schnorr_snark_witness_batch_cpu_fallback` in `src/gpu/src/gpu_backend_fallback.cpp`). Output is byte-identical to the CPU C ABI on all backends, so the **correctness** parity is closed. Native CUDA / OpenCL / Metal kernels are not yet implemented, so calling the batched API on a GPU backend currently runs at CPU throughput rather than device throughput. Public-data-only operation (no secret values), so there is no security impact. Tracked by `docs/CAAS_HARDENING_TODO.md` H-5; promote to performance claims only after native kernels land. |
| RR-006 | Hardware power/EM/fault side channels | Accepted (operating-environment scope) | Audit/docs | Methodology and scope statement in [HARDWARE_SIDE_CHANNEL_METHODOLOGY.md](HARDWARE_SIDE_CHANNEL_METHODOLOGY.md). No physical-attack lab claim is made; library makes only software-side-channel claims (CT-layer + CT-tooling). |
| RR-007 | Quantum (Shor's algorithm) attack on ECDLP | Accepted (curve-choice consequence) | Project | secp256k1 is a classical-cryptography curve. The library does not claim post-quantum resistance and does not include PQ algorithms. Documented in [THREAT_MODEL.md](THREAT_MODEL.md) §3 AM-10 and §6. |
| RR-008 | Application-layer signature replay | Out-of-scope (caller responsibility) | Caller | The library signs the bytes it is given. Semantic deduplication, nonce tracking, and replay protection are application-layer responsibilities. Documented in [THREAT_MODEL.md](THREAT_MODEL.md) §6. |
| RR-009 | Sybil attack on MuSig2/FROST quorum policy | Out-of-scope (caller responsibility) | Caller | Threshold and participant policy are caller decisions. The library implements the cryptographic protocol correctly given a quorum; it cannot detect a hostile participant set. Documented in [THREAT_MODEL.md](THREAT_MODEL.md) §6. |
| RR-010 | MuSig2 signer-index bypass (MED-3) | **CLOSED 2026-05-31** | MuSig2/ABI | Fully fail-closed across all three layers: (1) the C++ `musig2_partial_sign` now treats the Rule-13 signer-index cross-check as **mandatory** — when `individual_pubkeys` cannot validate the signer (empty or too short) it returns `Scalar::zero()` instead of signing blind (`src/cpu/src/musig2.cpp`); (2) the v1 ABI `ufsecp_musig2_partial_sign` is hard-failed at entry with `UFSECP_ERR_DEPRECATED_API` (it has no pubkeys parameter and cannot validate); (3) the v2 ABI `ufsecp_musig2_partial_sign_v2` validates `privkey ↔ pubkeys[signer_index]` at the boundary and populates `individual_pubkeys` before delegating to the signing core. Regression-guarded by `audit/test_regression_musig2_signer_index_validation.cpp` MSI-4 (now `advisory=false`, asserts the empty-pubkeys context fail-closes) and `test_regression_frost_musig2_degenerate.cpp` FMD-4 (populated context still signs). The earlier v5.0.0 ABI-extension plan is no longer needed: the C++ defense-in-depth + v2 boundary check close the gap without an ABI break. |
| RR-NEW-01 | `ct::scalar_inverse` non-CT on platforms without `__int128` (SEC-001-INCOMPLETE) | Accepted, non-blocking for Bitcoin Core PR | CT/portability | On platforms without `__int128` support (WASM target, MSVC 32-bit), `ct::scalar_inverse` falls back to a `fast::` multiplication chain which is variable-time. Not applicable to Bitcoin Core PR targets (x86-64 and ARM64 have `__int128`). Build system requires `__int128` for CT builds; static assert enforced in the CT path. See SEC-001-INCOMPLETE. |
| RR-NEW-02 | P1 shim regression tests advisory-gated by build configuration | Open, by design. Non-blocking for Bitcoin Core PR | Shim/audit | `regression_shim_security_v8` and `exploit_musig2_infinity_pubnonce` require `SECP256K1_BUILD_COMPAT_SHIM=ON`; without it they return ADVISORY_SKIP_CODE (77). Standalone CTest targets always run. Bitcoin Core PR is out of shim scope. Planned: shim-linked CI matrix job in future milestone. |

---

## RR-NEW-01: ct::scalar_inverse — non-CT on platforms without __int128

- **Severity:** P2
- **Affected function:** `ct::scalar_inverse`
- **Condition:** Platforms without `__int128` support (WASM target, MSVC 32-bit)
- **Behavior:** Falls back to `fast::` multiplication chain which is variable-time
- **Impact for Bitcoin Core PR:** Not applicable — Bitcoin Core targets x86-64 and ARM64 where `__int128` is available. WASM and MSVC 32-bit are not signing targets in this context.
- **Mitigation:** Build system requires __int128 for CT builds. Static assert added in CT path.
- **Status:** Accepted risk for non-supported platforms. No action required for Bitcoin Core backend.
- **Tracking:** SEC-001-INCOMPLETE

---

## RR-NEW-02 — P1 Shim Regression Tests Are Advisory-Gated by Build Configuration

**Type:** Coverage gap — not a vulnerability
**Status:** Open (by design)
**Severity:** Informational (architectural)
**Scope:** Shim layer security regression testing

**Description:**
Two P1-classified security regression tests are marked `advisory=true` in `unified_audit_runner.cpp`
because they require the libsecp256k1 compatibility shim to be linked:
- `regression_shim_security_v8` — covers P1-SEC-NEW-001 (ECDH strict key parse, rejects sk ≥ n)
  and RED-TEAM-008 (ECDSA verify off-curve pubkey rejection).
- `exploit_musig2_infinity_pubnonce` — covers P1-SEC-003 (MuSig2 pubnonce must reject infinity point).

In a build without `SECP256K1_BUILD_COMPAT_SHIM=ON`, these tests return `ADVISORY_SKIP_CODE (77)`
and are reported as `advisory_skipped`, not `advisory_failed`.

**Impact:** A reviewer who runs the unified audit runner without the shim linked will see P1 regressions
silently skipped. The standalone CTest targets for these modules DO exist and always run.

**Mitigation:** Run `cmake -DSECP256K1_BUILD_COMPAT_SHIM=ON ...` and rebuild to get mandatory verdicts
for shim P1 regressions. Alternatively, run the standalone CTest targets directly:
`ctest -R regression_shim_security_v8` and `ctest -R exploit_musig2_infinity_pubnonce`.

**Bitcoin Core scope:** Bitcoin Core uses only the CPU backend (no shim required). The P1 shim
regressions guard the `libsecp256k1_shim` compatibility layer which is out of scope for the Core PR.

**Planned resolution:** Add a shim-linked CI matrix job in a future milestone.

---

## Review Rule

When a residual risk becomes blocking, partially covered, or fully closed:

1. Update the matching entry here.
2. Update `docs/SELF_AUDIT_FAILURE_MATRIX.md` if the failure-class status changes.
3. Record the change in `docs/AUDIT_CHANGELOG.md`.

If a new owner-grade blocker appears, it should be reflected both here and in
the owner audit bundle rather than existing only as narrative commentary.