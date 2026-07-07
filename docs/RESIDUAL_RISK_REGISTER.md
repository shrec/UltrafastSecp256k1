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
| RR-BAS-01 | CAAS evidence auto-refresh (ci-evidence + API contracts) | **Ready for owner promotion (B19)** | Infra/audit | Bastion B19 added the `freshness_artifacts` disposition ledger (`docs/AUDIT_SLA.json`) + coverage gate `ci/check_evidence_refresh_coverage.py` (`audit_gate.py --evidence-refresh-coverage`, G-19): every tracked freshness artifact is now explicitly `auto` (producer cross-checked against the lane commit list) or `residual` (resolves here). `audit/ci-evidence` + `docs/API_SECURITY_CONTRACTS.json` (+ assurance_claims / determinism_golden / risk_coverage_report) remain documented residuals — authored/golden/build/owner-chosen-manual, not honestly nightly-regenerable. Promotion = owner approves a heavy snapshot lane. Not a vulnerability. See RR-BAS-01 below. |
| RR-BAS-02 | Incident-drill freshness SLO promotion to blocking | **Ready for owner promotion (B19)** | Audit/infra | `incident_drill_freshness_days` (`docs/AUDIT_SLA.json`) is advisory (warning). B19 proved the precondition: the coverage gate confirms the drill log is genuinely auto-refreshed by the nightly lane, and the `B19:evidence_refresh_coverage` self-test proves `ci/audit_sla_check.py` BLOCKS a simulated stale log under blocking severity (and only warns under advisory). Promotion = owner flips the severity to `blocking` after one observation window. See RR-BAS-02 below. |
| RR-BAS-03 | Benchmark `target_context` labels | **CLOSED 2026-06-13** | Bench/audit | Closed by Bastion B17: canonical `bench_unified_*.json` (target_context=microbench) and `BITCOIN_CORE_BENCH_RESULTS.json` (target_context=bitcoin_core) now carry explicit `target_context` + `claim_scope` metadata, validated against `docs/BENCH_TARGET_CONTEXT_SCHEMA.json` by `ci/check_bench_target_context.py` (folded into `check_bench_doc_consistency.py`, co-gated by `perf_security_cogate.py`). See RR-BAS-03 below. |
| RR-BAS-04 | Research-monitor attack-class taxonomy | **CLOSED 2026-06-13** | Audit/tooling | Closed by Bastion B18: all 45 signal classes carry `attack_class` (16-value enum) + evidence routing (affected_primitive/surface, expected_gate, missing_evidence_action); validated by `ci/check_research_signal_matrix.py` (`audit_gate.py --research-signal-matrix`, G-18) and rendered by `ci/research_monitor.py`. See RR-BAS-04 below. — was: `docs/RESEARCH_SIGNAL_MATRIX.json` signal classes carry a coverage `status` but no `attack_class` field; Bastion B6 added actionable affected-surface/patch-plan rendering without the taxonomy. See RR-BAS-04 below. |

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

## RR-BAS-01 — CAAS evidence auto-refresh (ci-evidence + API contracts)

**Type:** Process / automation gap — not a vulnerability
**Status:** **Ready for owner promotion** (Bastion B19 — coverage gated; auto-vs-residual disposition formalized)
**Severity:** Informational (operational cadence)
**Scope:** Critical-evidence freshness automation

**Description:**
`audit/ci-evidence/*` (CT / adversarial / fuzz snapshots) and
`docs/API_SECURITY_CONTRACTS.json` are tracked on the 14-day
`critical_evidence_freshness_days` SLO (`docs/AUDIT_SLA.json`), but the nightly
`caas-evidence-refresh.yml` workflow does **not** regenerate them. They are
refreshed by a manual owner chore (build + run the four standalone audit binaries
— `test_adversarial_protocol`, `test_ecies_regression`, `test_fuzz_parsers`,
`test_fuzz_address_bip32_ffi` — and commit dated snapshots). Without automation
they will periodically cross the SLO and drop the autonomy score until refreshed.

**What B19 added (hardening):** Every freshness artifact tracked by
`ci/audit_sla_check.py` now carries an explicit refresh disposition in the
`freshness_artifacts` ledger (`docs/AUDIT_SLA.json`), enforced by
`ci/check_evidence_refresh_coverage.py` (`audit_gate.py --evidence-refresh-coverage`,
G-19). The disposition is honest, not aspirational:

| Artifact | Disposition | Why |
|----------|-------------|-----|
| `assurance_report` | **auto** | derived report regenerated + committed by the lane (`export_assurance.py`) |
| `incident_drill_log` | **auto** | rewritten by the lane's autonomy step (`incident_drills.py`) |
| `ct_evidence` (`audit/ci-evidence`) | **residual** | CT/fuzz output snapshots; owner explicitly chose a manual chore over auto-committing fuzz output |
| `api_contracts` | **residual** | authored security-contract spec; auto-regen would fake freshness; kept current by the fail-closed changed-file policy |
| `assurance_claims` | **residual** | authored claims ledger validated by `validate_assurance.py` |
| `determinism_golden` | **residual** | golden baseline — must stay stable; re-verified, not regenerated |
| `risk_coverage_report` | **residual** | build-only artifact under gitignored `out/` |

The gate fails closed if a **blocking** freshness artifact ever loses *both* its
auto-producer and its documented residual, or if an `auto` disposition claims a
commit path the workflow does not actually stage. So the manual residual can no
longer silently disappear, and a future blocking freshness SLO cannot be added
without a producer-or-residual. The `auto`/`residual` honesty was
adversarially cross-checked (no authored/golden artifact is fake-refreshed).

**Current behavior / mitigation:** The Bastion B3 pre-alert
(`ci/audit_sla_check.py`) warns ~4 days before the block with a per-artifact
`days_until_block`, so the chore is signalled rather than silent. The owner
explicitly chose the manual chore + pre-alert model over an auto-committing
scheduled workflow (a CI-cost / infra decision); B19 respects that choice and
makes the residual explicit + gated rather than overriding it.

**Acceptance criteria (for closure):** `audit/ci-evidence` and
`docs/API_SECURITY_CONTRACTS.json` either (a) are moved to `auto` behind a
scheduled lane that regenerates + commits them on a < 14-day cadence, or
(b) remain documented residuals with the owner accepting the manual-chore model
permanently — and in both cases the G-19 coverage gate stays green.
**Promotion trigger (owner decision):** Owner approves a scheduled refresh lane
(a dedicated `ci-evidence-refresh.yml`, or a snapshot step wired into the heavy
`security-audit.yml` lane) that builds + runs the four audit binaries and commits
dated snapshots on a < 14-day cadence. Flipping the disposition is a one-line
manifest edit (`residual` → `auto` with `committed_paths: ["audit/ci-evidence", ...]`)
that the G-19 gate will then verify against the lane.
**Close condition:** Either the scheduled lane has refreshed the evidence on
cadence for one full window with no unattended SLO breach (path a), or the owner
records permanent acceptance of the manual model (path b); update this entry to
CLOSED.

---

## RR-BAS-02 — Incident-drill freshness SLO promotion to blocking

**Type:** Intentional severity choice — not a vulnerability
**Status:** **Ready for owner promotion** (Bastion B19 — precondition proven + block self-tested)
**Severity:** Informational (cadence)
**Scope:** Incident-drill cadence enforcement

**Description:**
Bastion B9 added a machine-readable drill log (`docs/INCIDENT_DRILL_LOG.json`,
written on every `ci/incident_drills.py` run) and a new
`incident_drill_freshness_days` SLO (`docs/AUDIT_SLA.json`). The SLO is set to
`severity: warning` (advisory) rather than `blocking`, to avoid a self-inflicted
recurring release block before the auto-refresh loop is proven — the same caution
that governed the H-1 30-day observation period.

**Current behavior / mitigation:** The drill log is added to the nightly
`caas-evidence-refresh.yml` commit list (the autonomy step runs `incident_drills.py`,
which rewrites the log), so it is refreshed daily in CI. The advisory SLO + B3
pre-alert surface a stalled-drill condition without blocking.

**What B19 added (promotion readiness):** Two of the three close-condition
elements are now proven, so only the observation window + the owner's one-line
severity flip remain:
- The G-19 coverage gate (`ci/check_evidence_refresh_coverage.py`) confirms
  `incident_drill_log` is genuinely `auto`-refreshed by the lane
  (`incident_drill_autorefresh: true`) — the precondition for promotion.
- The `B19:evidence_refresh_coverage` self-test
  (`ci/test_audit_scripts.py`) already **proves** `ci/audit_sla_check.py` blocks a
  simulated stale drill log under blocking severity *and* only warns under the
  live advisory severity (no false pass). The "confirm the block works" half of
  the close condition is therefore satisfied ahead of the flip.

**Acceptance criteria (for closure):** The nightly auto-commit loop keeps
`docs/INCIDENT_DRILL_LOG.json` within the freshness threshold for a full
observation window with zero false blocks.
**Promotion trigger:** The drill-log auto-commit loop is observed green for one
full window (≈ the H-1 30-day model).
**Close condition:** Flip `incident_drill_freshness_days.severity` from `warning`
to `blocking` in `docs/AUDIT_SLA.json` (the block behaviour is already self-tested),
and update this entry to CLOSED.

---

## RR-BAS-03 — Benchmark `target_context` labels — **CLOSED 2026-06-13 (Bastion B17)**

**Type:** Evidence-metadata gap — not a vulnerability
**Status:** **CLOSED 2026-06-13** (Bastion B17)
**Severity:** Informational (benchmark provenance)
**Scope:** Performance-claim disambiguation

**Description:**
Bastion B8 co-gated benchmark-artifact integrity but the canonical
`bench_unified_*.json` artifacts did not carry an explicit `target_context` label,
so a reviewer could in principle conflate a GPU public-data benchmark with a CPU
microbenchmark, or a microbenchmark with Bitcoin Core node throughput.

**Resolution (B17):**
- Added `docs/BENCH_TARGET_CONTEXT_SCHEMA.json` — the `target_context` enum
  (microbench / batch_verify / bitcoin_core / libbitcoin / gpu_public_data /
  gpu_hardware / wasm / package_integration / unknown_owner_gated) + required
  fields (target_context, operation, claim_scope, evidence_path, reproduce/source
  command, commit, security_gate_dependency).
- Added explicit `target_context` + `claim_scope` + `security_gate_dependency`
  **metadata** (no measurement numbers changed) to the canonical artifacts:
  `docs/bench_unified_*.json` → `microbench`,
  `docs/BITCOIN_CORE_BENCH_RESULTS.json` → `bitcoin_core` (with an
  `integration_evidence` reference).
- Added `ci/check_bench_target_context.py`, folded into
  `ci/check_bench_doc_consistency.py` (now `--json`-capable) and co-gated by
  `ci/perf_security_cogate.py`: a benchmark artifact fails if `target_context` is
  missing/invalid, if a timed artifact lacks `claim_scope`, if `gpu_public_data`
  is presented as native GPU-hardware performance, or if a `bitcoin_core`/`libbitcoin`
  claim lacks an integration-evidence reference. Negative fixture
  `B17:bench_target_context` proves each failure path.

**Acceptance criteria — MET:** every canonical bench artifact carries a valid
`target_context`, and `check_bench_doc_consistency.py` (via the folded context
gate) fails closed when it is missing/invalid/mis-scoped, with a negative fixture.
*(The label is carried at the artifact-`metadata` level — the correct granularity,
since every result within a `bench_unified` run shares the same context — rather
than duplicated per result.)*

---

## RR-BAS-04 — Research-monitor attack-class taxonomy — **CLOSED 2026-06-13 (Bastion B18)**

**Type:** Classification-enrichment gap — not a vulnerability
**Status:** **CLOSED 2026-06-13** (Bastion B18)
**Severity:** Informational (triage ergonomics)
**Scope:** Research-signal classification

**Description:**
Bastion B6 made high-confidence research findings actionable but the signal matrix
carried only a coverage `status` — not a stable attack-class taxonomy nor an
explicit route to the evidence surface/gate that should catch each signal.

**Resolution (B18):**
- Extended `docs/RESEARCH_SIGNAL_MATRIX.json` with an `attack_class_enum` (16 values:
  nonce_bias_or_reuse, signature_malleability, parser_boundary, invalid_curve_or_pubkey,
  scalar_domain, batch_verification, side_channel_ct, gpu_backend_parity,
  protocol_state_machine, threshold_multisig, supply_chain, fuzz_crash,
  integration_consensus, benchmark_claim, hardware_fault_or_em, out_of_scope) and
  added per-class `attack_class`, `affected_primitive`, `affected_surface`,
  `expected_evidence`, `expected_gate`, `missing_evidence_action`, `severity_hint`,
  `owner_route` to **all 45 classes** (mappings adversarially verified by a
  5-agent read-only workflow).
- Added `ci/check_research_signal_matrix.py` (`audit_gate.py --research-signal-matrix`,
  G-18): every in-scope class must have a valid `attack_class`; covered classes must
  have existing `expected_evidence` and a resolvable `expected_gate` (an audit_gate
  CHECK_MAP flag or a ci/*.py script); candidates need a `missing_evidence_action`;
  out_of_scope need a rationale.
- Upgraded `ci/research_monitor.py` to render `attack_class`, `affected_primitive`,
  `affected_surface`, and `expected_gate`, and to put `missing_evidence_action` as
  the **first patch-plan step** (issue-only, no branch/PR — unchanged).

**Acceptance criteria — MET:** all 45 signal classes carry an enum `attack_class`,
the renderer emits it in the finding body, and `B18:research_signal_matrix` asserts
both the validator's failure paths and the rendered routing. *(The matrix holds 45
classes, not the 48 estimated when the residual was filed.)*

---

## RR-GPU-OCL-01 — No white-box CT measurement gate for OpenCL / Metal (source-CT only)

**Type:** CT-measurement tooling gap (the underlying leaks are fixed)
**Status:** **Source fixed for all backends; runtime CT-measurement gate is CUDA-only**
**Severity:** Informational (no known residual leak; verification-depth gap)
**Scope:** OpenCL (NVIDIA + portable `#else`) and Metal signing CT measurement

**Description (updated 2026-06-22):**
All four GPU signing reductions are now branchless: CUDA (already CT — ncu gate 5/5),
NVIDIA OpenCL `reduce_512_to_256_32_ocl` + scalar add/sub (`opencl_test` 44/44), the
portable `#else` `field_reduce` (`if (temp[4]!=0)` / `if (carry)` made unconditional /
masked — CPU equivalence 5,000,000 random+edge inputs, 0 mismatches vs the original),
and Metal `field_reduce_512` (`while (acc[8]!=0)` → fixed 2 iterations —
`test_exploit_metal_field_reduce` 14/14 vs an independent reference, incl. issue #226).
Metal scalar add/sub and the field final-subtract were already branchless.

**Remaining gap:** there is **no ncu-equivalent white-box CT gate for OpenCL or Metal**
(`ci/check_gpu_ct_uniformity.py` is CUDA-only). So OpenCL/Metal CT rests on source-level
branchlessness + correctness equivalence, not a fixed-vs-random branch-uniformity
*measurement*. Runtime CT confirmation on actual AMD/Intel (OpenCL `#else`) and Apple
(Metal) hardware, and a white-box CT gate for those backends, are standing follow-ups.

---

## Review Rule

When a residual risk becomes blocking, partially covered, or fully closed:

1. Update the matching entry here.
2. Update `docs/SELF_AUDIT_FAILURE_MATRIX.md` if the failure-class status changes.
3. Record the change in `docs/AUDIT_CHANGELOG.md`.

If a new owner-grade blocker appears, it should be reflected both here and in
the owner audit bundle rather than existing only as narrative commentary.