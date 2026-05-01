# Self-Audit Failure-Class Matrix

Canonical mapping from major failure classes to the deterministic audit
surfaces that are expected to catch them.

This document exists to keep the self-audit system grounded in responsibility
rather than presentation. The goal is not to claim that every possible defect
is mathematically impossible; the goal is to ensure every known bug family is
either covered by explicit evidence or listed as residual risk.

---

## Reading Rule

For a failure class to count as covered, it should have all three of the
following:

1. A deterministic repository surface that exercises the class
2. A maintained documentation surface that explains where the evidence lives
3. A non-empty residual-risk note if coverage is still partial

Raw test volume is not enough by itself.

---

## Matrix

| Failure Class | Primary Risk | Deterministic Audit Surfaces | Current Status | Residual Risk / Next Expansion |
|---------------|--------------|------------------------------|----------------|--------------------------------|
| Arithmetic correctness | Wrong field/scalar/point math, invalid reductions, group-law breakage | `AUDIT_COVERAGE.md` sections 1 and 3, `AUDIT_TRACEABILITY.md`, `audit_field.cpp`, `audit_scalar.cpp`, `audit_point.cpp`, exhaustive/property suites | Covered | Keep traceability current when new arithmetic backends or representations land |
| Serialization and parser boundaries | Malformed DER, pubkeys, BIP-32 paths, ciphertext, hostile inputs | `AUDIT_SCOPE.md` test inventory, `audit/test_adversarial_protocol.cpp`, `test_fuzz_parsers.cpp`, `test_fuzz_address_bip32_ffi.cpp`, `ci/invalid_input_grammar.py`, `ci/audit_gate.py --invalid-inputs` | Covered | Expand edge-case catalogs whenever new public parsers or envelope formats are added |
| Constant-time regressions | Secret-dependent branches, memory access, compiler-lifted timing leaks | `CT_VERIFICATION.md`, `test_ct_sidechannel.cpp`, `ct_sidechannel_smoke`, `AUDIT_TRACEABILITY.md` CT rows, `AUDIT_MANIFEST.md` P4 | Covered | Formal/no-secret-branch and no-secret-memory guarantees still retain manual/formal-tool residual risk; keep the CT route table and deterministic CT evidence current when secret-bearing ABI surfaces change |
| Protocol misuse and transcript binding | MuSig2/FROST/adaptor/BIP-324 misuse, replay, forked transcripts, nonce reuse, state corruption after failures | `test_musig2_frost_advanced.cpp`, `audit/test_adversarial_protocol.cpp`, `ci/stateful_sequences.py`, `ci/audit_gate.py --stateful-sequences`, `AUDIT_TRACEABILITY.md` protocol rows, `AI_REVIEW_EVENTS.json` | Covered | New protocol features must add misuse tests before claims are upgraded |
| ABI misuse and hostile-caller behavior | NULL handling, count=0 misuse, error propagation, wrong buffer shapes, thread-stress breakage | `FFI_HOSTILE_CALLER.md`, `audit/test_adversarial_protocol.cpp`, `test_gpu_host_api_negative.cpp`, `test_gpu_abi_gate.cpp`, `ci/audit_gate.py --abi-negative-tests`, `ci/audit_gate.py --invalid-inputs`, `ci/audit_gate.py --stateful-sequences` | Covered | Hostile-caller quartet is currently closed for the exported ABI; keep the manifest, invalid-input grammar, and stateful misuse evidence current when new exports land |
| GPU host/runtime misuse | Invalid backend/device selection, missing buffers, capability drift, unsupported-path confusion | `docs/GPU_VALIDATION_MATRIX.md`, `docs/BACKEND_ASSURANCE_MATRIX.md`, `test_gpu_host_api_negative.cpp`, `test_gpu_backend_matrix.cpp`, `test_gpu_ops_equivalence.cpp` | Covered | Real-device backend diversity still depends on available hardware, especially future ROCm/HIP evidence |
| Documentation drift and public overclaim risk | Docs claim things the repo no longer proves, stale counts, stale capability language | `ci/validate_assurance.py`, `ci/preflight.py --claims`, `ASSURANCE_LEDGER.md`, `ASSURANCE_CLAIMS.json`, `AI_REVIEW_EVENTS.json` | Covered | More subsystem docs still need the same claim-ID and stale-drift discipline as top-level docs |
| Audit harness quality and evidence drift | Vacuous checks, polarity bugs, weak thresholds, ignored returns, or surviving mutants create false assurance | `ci/audit_test_quality_scanner.py`, `ci/audit_gate.py --audit-test-quality`, `ci/mutation_kill_rate.py`, `audit/test_mutation_kill_rate.cpp` | Covered | Mutation kill remains the heavier batch lane; keep its report cadence current when touching core arithmetic or audit harnesses |
| Benchmark methodology and publishability drift | Overstated performance claims, mixed hardware labels, publishability without evidence | `BENCHMARK_POLICY.md`, `PERFORMANCE_REGRESSION.md`, `GPU_BACKEND_EVIDENCE.json`, `ci/preflight.py --gpu-evidence` | Covered | GPU reproducibility and artifact retention should keep expanding alongside backend docs |
| Graph/index blind spots | False confidence caused by missing source/doc/test surfaces in the graph | `AUDIT_MANIFEST.md` P6, `ci/validate_assurance.py`, `tools/source_graph_kit/source_graph.py`, `docs/PROJECT_GRAPH_REASONING.md` | Covered | Any new path class missing from the graph remains an assurance gap until indexed; library audit scripts and harnesses must stay visible in both the library-local graph and the workspace root graph |
| Supply-chain / workflow drift | CI/workflow hardening regression, dependency-review gaps, missing local parity | `docs/CI_ENFORCEMENT.md`, `docs/LOCAL_CI.md`, workflow inventory, `AUDIT_COVERAGE.md` infra summary, `ci/supply_chain_gate.py`, `ci/evidence_governance.py` | Covered | GitHub-native services and some platform-specific integrations still sit outside pure local reproduction, but the repository-maintained local parity and workflow inventory provide deterministic owner-grade coverage for drift in tracked surfaces |
| Optional backend expansion (ROCm/HIP real hardware) | Future AMD-specific publishability claims without real-device evidence | `GPU_BACKEND_EVIDENCE.json`, `GPU_VALIDATION_MATRIX.md`, `BACKEND_ASSURANCE_MATRIX.md`, `BENCHMARK_POLICY.md` | Intentionally deferred | Non-blocking for current audit validity; promote only when AMD hardware-backed evidence exists |
| Formal invariant compliance | Missing or incomplete formal invariant specs for critical operations | `docs/FORMAL_INVARIANTS_SPEC.json`, `ci/check_formal_invariants.py`, `docs/INVARIANTS.md`, `docs/CRYPTO_INVARIANTS.md` | Covered | Automated via `check_formal_invariants.py`; 7 critical operations with preconditions, postconditions, and CT linkage validated |
| Audit SLA / evidence staleness | Evidence drifts past age limits without detection, stale golden references | `docs/AUDIT_SLA.json`, `ci/audit_sla_check.py`, `docs/SECURITY_AUTONOMY_KPI.json`, `docs/DETERMINISM_GOLDEN.json` | Covered | SLA enforcement infra complete; `DETERMINISM_GOLDEN.json` generated and validated; `audit_sla_check.py` returns `release_ready: True`; `check_determinism_gate.py` returns `overall_pass: True` |
| Incident response readiness | Untested triage and disclosure paths | `docs/INCIDENT_RESPONSE.md`, `ci/incident_drills.py` | Covered | Three automated drills (key compromise, CI poisoning, dependency compromise) run as preflight step 20/20 |

---

## Residual-Risk Rules

Residual risk is acceptable only when all three conditions hold:

1. The gap is named explicitly in this matrix or a linked audit doc.
2. The gap does not invalidate already-claimed evidence for covered surfaces.
3. The gap has a plausible expansion path rather than hand-waving.

Examples of currently explicit residual risk:

1. CT branch/memory guarantees still rely partly on manual or formal-tool follow-up.
2. Some workflow and supply-chain checks remain partly GitHub-native rather than purely local.
3. ROCm/HIP real-device evidence is intentionally deferred until AMD hardware is available.

---

## Update Rule

When a new subsystem, protocol, backend, parser, or public claim is added:

1. Add or update the matching failure-class row here.
2. Link the new deterministic evidence surface.
3. Record any residual risk that is still not closed.
4. Update `docs/FORTRESS_ROADMAP.md` if the change materially improves completeness.

If a new feature lands without fitting any row in this matrix, treat that as a
self-audit gap until the mapping is added.