# Security Autonomy 30-Day Execution Plan

**Created**: 2026-04-14  
**Target**: Full security autonomy — system self-manages all critical assurance surfaces  
**Measurement**: KPI dashboard at `docs/SECURITY_AUTONOMY_KPI.json`

---

## Gap Analysis Summary

| # | Gap | Current State | Target State | Week |
|---|-----|---------------|--------------|------|
| G1 | Formal invariants (prove-or-block) | Invariants documented, tested dynamically | SMT-checkable spec + dynamic + regression triple | W1-2 |
| G2 | Determinism gate (multi-arch fail-closed) | Single-arch Python gate (`check_determinism_gate.py`) | Cross-arch digest comparison, fail-closed CI gate | W1 |
| G3 | Supply-chain fail-closed | `verify_reproducible_build.sh`, `verify_slsa_provenance.py` exist | Full chain: pin + repro-diff + provenance + artifact hash | W2 |
| G4 | Risk-surface coverage (not code coverage) | Code/module coverage via graph | Risk-class matrix with min thresholds per class | W1 |
| G5 | Stateful adversarial fuzz with replay | 18 fuzz targets, corpus dirs, cryptofuzz CI | Seed replay, crash triage, corpus min, exploit→regression auto | W2-3 |
| G6 | Misuse-resistance gates | `test_c_abi_negative.cpp`, some hostile-caller tests | Systematic hostile-caller simulation per ABI function | W2 |
| G7 | SLA/SLO on audit infra | No measurable control plane | Stale-time, unresolved-window, flake-budget, evidence-freshness | W1 |
| G8 | Evidence governance + tamper-resistance | `export_assurance.py`, `validate_assurance.py` | Provenance-linked, immutable trail, who/what/when/binary chain | W3 |
| G9 | Performance-security co-gating | Separate perf and security gates | Coupled gate: perf cannot pass if CT/determinism/secret regresses | W2 |
| G10 | Incident drill readiness | `INCIDENT_RESPONSE.md` runbook exists | Automated drills: key compromise, CI poison, dep compromise sims | W3-4 |

---

## Week 1 (Days 1-7): Foundation Gates

### Day 1-2: Risk-Surface Coverage Matrix
- **Deliverable**: `ci/risk_surface_coverage.py`
- **What**: Define 7 risk classes (CT, parser-boundary, ABI-boundary, GPU-parity, secret-lifecycle, determinism, fuzz-corpus). Query graph + test registry to compute coverage per class.
- **Gate**: Each critical class must have ≥ 80% symbol coverage. Fail-closed.
- **KPI**: `risk_coverage_percent` per class, `risk_classes_passing` count.

### Day 2-3: Audit SLA/SLO Framework
- **Deliverable**: `ci/audit_sla_check.py`, `docs/AUDIT_SLA.json`
- **What**: Define measurable SLOs:
  - Max stale evidence age: 30 days
  - Max unresolved HIGH finding window: 7 days
  - Flake budget: ≤ 2% of CI runs
  - Evidence freshness: all critical evidence ≤ 14 days old
  - Min gate pass rate: 95% over 30-day window
- **Gate**: If any SLO breached, `release_ready = false`. 
- **KPI**: `sla_violations` count, `release_ready` boolean.

### Day 3-4: Formal Invariant Spec Layer
- **Deliverable**: `docs/FORMAL_INVARIANTS_SPEC.json`, `ci/check_formal_invariants.py`
- **What**: For each critical operation (ecdsa_sign, ecdsa_verify, schnorr_sign, schnorr_verify, ecdh, bip32_derive, bip340_verify), define:
  - Preconditions (input domain constraints)
  - Postconditions (output guarantees)
  - Algebraic identities (prove-or-block)
  - CT requirement flag
  - Linked test IDs
- **Gate**: Every critical operation must have ≥ 1 algebraic identity verified by tests + ≥ 1 boundary condition test. No operation may be untested.
- **KPI**: `invariants_proven` / `invariants_total`, `operations_fully_covered`.

### Day 5-7: Cross-Architecture Determinism Gate Enhancement
- **Deliverable**: Enhanced `ci/check_determinism_gate.py`
- **What**: Extend current gate to:
  - Produce per-vector canonical digests (SHA-256 of outputs)
  - Compare against golden reference file (`docs/DETERMINISM_GOLDEN.json`)
  - Support multi-arch comparison (x86_64, aarch64, riscv64 digest files)
  - Fail-closed: any divergence blocks release
- **KPI**: `determinism_vectors_checked`, `arch_pairs_compared`, `divergences_found`.

---

## Week 2 (Days 8-14): Hardening Gates

### Day 8-9: Supply-Chain Fail-Closed Gate
- **Deliverable**: `ci/supply_chain_gate.py`
- **What**: Unified gate checking:
  - Build-input pinning (CMake version, compiler hash, dependency lockfile)
  - Reproducible-build digest comparison (existing `verify_reproducible_build.sh` output)
  - SLSA provenance validation (existing `verify_slsa_provenance.py` output)
  - Artifact hash policy (all release artifacts must have SHA-256 in manifest)
  - Build hardening flags (existing `verify_build_hardening.py` output)
- **Gate**: All 5 sub-checks must pass. Any failure = release blocked.
- **KPI**: `supply_chain_checks_passing` / 5, `pinned_inputs_verified`.

### Day 10-11: Performance-Security Co-Gating
- **Deliverable**: `ci/perf_security_cogate.py`
- **What**: Before any optimization merges:
  - CT evidence must not regress (dudect, valgrind-ct, cachegrind)
  - Determinism gate must still pass
  - Secret lifecycle paths unchanged or re-verified
  - No new `Unsupported` GPU stubs without tracking
- **Gate**: Perf improvement rejected if any security property regresses.
- **KPI**: `cogate_pass` boolean, `security_regressions_blocked` count.

### Day 12-13: Misuse-Resistance Gate Expansion
- **Deliverable**: `ci/check_misuse_resistance.py`, expand `test_c_abi_negative.cpp`
- **What**: Systematic hostile-caller patterns for every `ufsecp_*` function:
  - NULL pointers for all pointer args
  - Zero-length buffers
  - Oversized length params (SIZE_MAX, SIZE_MAX/2)
  - Double-free / double-destroy
  - Cross-thread concurrent calls
  - Partially initialized context structs
  - Wrong-type context (sign ctx passed to verify)
- **Gate**: Every ABI function must have ≥ 3 negative test cases. No crash, no UB.
- **KPI**: `abi_functions_covered` / `abi_functions_total`, `negative_tests_total`.

### Day 14: Fuzz Infrastructure Upgrade Spec
- **Deliverable**: `docs/FUZZ_INFRASTRUCTURE.md`, `ci/fuzz_campaign_manager.py`
- **What**: Design:
  - Seed replay framework (corpus → CTest registration)
  - Crash triage automation (dedup, severity classify, auto-file)
  - Corpus minimization (cmin integration)
  - Exploit-to-regression pipeline (crash → minimal reproducer → CTest)
- **KPI**: `corpus_seeds_registered`, `crashes_auto_triaged`, `regressions_generated`.

---

## Week 3 (Days 15-21): Evidence & Governance

### Day 15-17: Evidence Governance Framework
- **Deliverable**: `ci/evidence_governance.py`, `docs/EVIDENCE_CHAIN.json`
- **What**: Every critical verdict must record:
  - `who`: script/tool that produced it
  - `what`: exact check performed
  - `when`: ISO 8601 timestamp
  - `commit`: git SHA at time of evidence
  - `binary_hash`: SHA-256 of tested artifact
  - `verdict`: pass/fail/skip with reason
  - `signature`: HMAC of the above fields (tamper detection)
- **Gate**: Evidence chain must be complete and internally consistent. Any gap = fail.
- **KPI**: `evidence_records_total`, `evidence_chain_valid` boolean, `orphaned_verdicts` count.

### Day 18-19: Fuzz Infrastructure Implementation
- **Deliverable**: Implement `ci/fuzz_campaign_manager.py`
- **What**: Working seed replay, corpus minimization wrapper, crash→regression converter.
- **KPI**: `active_corpus_size`, `regressions_from_crashes`, `corpus_coverage_delta`.

### Day 20-21: Integration Testing
- **Deliverable**: All new gates integrated into `ci/preflight.py`
- **What**: Preflight runs all new gates. Any single gate failure = overall fail.
- **KPI**: `preflight_gates_total`, `preflight_gates_passing`.

---

## Week 4 (Days 22-30): Drill & Certification

### Day 22-24: Incident Drill Framework
- **Deliverable**: `ci/incident_drills.py`, `docs/DRILL_RESULTS.json`
- **What**: Automated simulation scenarios:
  - **Key compromise drill**: inject known-weak key, verify detection pipeline catches it
  - **CI poisoning drill**: simulate tampered build output, verify provenance check catches it
  - **Dependency compromise drill**: simulate altered dependency hash, verify supply-chain gate catches it
- **Gate**: All 3 drills must pass. Drill failure = release blocked until remediated.
- **KPI**: `drills_passed` / `drills_total`, `drill_response_time_seconds`.

### Day 25-27: Full System Integration Test
- **Deliverable**: `ci/security_autonomy_check.py` — master orchestrator
- **What**: Single command that runs ALL gates and produces unified verdict:
  ```bash
  python3 ci/security_autonomy_check.py --json -o autonomy_report.json
  ```
- **KPI**: `autonomy_score` (0-100), `gates_passing` / `gates_total`, `autonomy_ready` boolean.

### Day 28-30: KPI Dashboard & Documentation
- **Deliverable**: `docs/SECURITY_AUTONOMY_KPI.json` (auto-generated), updated `docs/PRE_RELEASE_CHECKLIST.md`
- **What**: Final KPI snapshot, documentation of all new gates, integration into release process.

---

## KPI Summary Table

| KPI | Threshold | Measurement |
|-----|-----------|-------------|
| Risk-class coverage (critical) | ≥ 80% per class | `risk_surface_coverage.py` |
| Formal invariant coverage | 100% critical ops | `check_formal_invariants.py` |
| Determinism golden match | 0 divergences | `check_determinism_gate.py` |
| Supply-chain gate pass | 5/5 sub-checks | `supply_chain_gate.py` |
| SLA violations | 0 active | `audit_sla_check.py` |
| Negative test coverage | ≥ 3 per ABI fn | `check_misuse_resistance.py` |
| Evidence chain validity | 100% linked | `evidence_governance.py` |
| Incident drills passed | 3/3 | `incident_drills.py` |
| Perf-security co-gate | No regressions | `perf_security_cogate.py` |
| Fuzz regressions generated | Monotonically ↑ | `fuzz_campaign_manager.py` |
| Overall autonomy score | ≥ 90/100 | `security_autonomy_check.py` |

---

## Success Criteria

The system is "autonomously secure" when:

1. **Triple evidence** on every critical class: formal spec + dynamic test + regression guard
2. **All gates fail-closed**: no-evidence == fail, everywhere
3. **Risk-surface coverage**: 100% on critical classes (CT, parser, ABI, secret, determinism)
4. **Determinism + provenance + reproducibility**: simultaneously green
5. **Exploit → permanent regression**: automatic, no manual step
6. **SLA compliance**: zero violations in 30-day rolling window
7. **Evidence chain**: complete, tamper-resistant, commit-linked
8. **Drills passing**: all 3 incident scenarios pass quarterly

Target date: 2026-05-14 (30 days from plan creation)
