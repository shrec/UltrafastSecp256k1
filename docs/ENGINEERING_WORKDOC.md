# Engineering Work Document

Practical engineering to-do list for CPU and GPU architecture hardening.

This document is not an audit report and not a roadmap for sponsors.
It is a working checklist for the project owner: what is already strong,
what is still missing, and what should be done next to improve engineering
quality, maintainability, and confidence.

---

## Status Legend

- `[OK]` Present and usable
- `[WIP]` In progress / partially covered
- `[TODO]` Missing
- `[LATER]` Intentionally deferred

---

## GPU Work Items

### 1. Backend Validation Unification

- `[OK]` CUDA has `gpu_bench_unified`, `gpu_audit_runner`, selftest, CT smoke, and specialized benches.
- `[OK]` OpenCL has selftest, audit runner, extended host test, and benchmark path.
- `[OK]` Metal has audit runner, extended host test, host integration test, and benchmark path.
- `[TODO]` Add one canonical document mapping all GPU backend entry points from a release engineer perspective.
- `[TODO]` Link [GPU_VALIDATION_MATRIX.md](/home/shrek/Secp256K1/Secp256K1fast/libs/UltrafastSecp256k1/docs/GPU_VALIDATION_MATRIX.md) from README / GPU testing docs / benchmarking docs.
- `[TODO]` Standardize backend naming in reports: `CUDA`, `ROCm/HIP`, `OpenCL`, `Metal`.

### 2. ROCm / AMD Hardware Validation

- `[WIP]` ROCm/HIP source path exists via shared CUDA/HIP code.
- `[TODO]` Run full validation on real AMD hardware.
- `[TODO]` Save one benchmark artifact and one audit artifact from a real AMD device.
- `[TODO]` Record exact ROCm version, compiler, GPU model, and driver.
- `[TODO]` Update docs only after real-device validation succeeds.

### 3. Cross-Backend Equivalence

- `[OK]` Shared type contract exists in `types.hpp`.
- `[OK]` CUDA is effectively the reference GPU backend.
- `[TODO]` Create explicit CPU<->CUDA / CUDA<->OpenCL / CUDA<->Metal equivalence matrix by feature.
- `[TODO]` Add reproducible artifact set for backend parity checks.
- `[TODO]` Keep one canonical list of "must match bit-for-bit" operations.
- `[TODO]` Add a release gate that fails if parity artifacts are missing for active GPU backends.

Suggested parity categories:

- Field add/sub/mul/sqr/inv
- Point add/dbl
- `k*G`
- `k*P`
- ECDSA sign/verify
- Schnorr sign/verify
- ECDH
- Recovery
- Batch inverse
- MSM / batch verify where implemented

### 4. Real-World GPU Flow Coverage

- `[OK]` CUDA has specialized benches such as BIP-352 and ZK.
- `[TODO]` Define one canonical set of "real workload" GPU benches to keep across releases.
- `[TODO]` Decide which flows are mandatory for release:
  - BIP-352
  - batch signatures
  - address/indexing pipeline
  - ZK prove/verify
- `[TODO]` Archive JSON/TXT benchmark artifacts for those flows.

### 5. GPU Documentation Hygiene

- `[OK]` GPU testing guide exists.
- `[OK]` Benchmarking guide mentions `gpu_bench_unified`.
- `[TODO]` Keep docs aligned with actual targets and filenames.
- `[TODO]` Remove any wording that implies ROCm is validated before AMD hardware testing exists.
- `[TODO]` Ensure every GPU backend page states:
  - correctness path
  - audit path
  - benchmark path
  - host integration path

### 6. GPU Release Checklist

- `[TODO]` Before each serious release candidate, require:
  - CUDA unified bench artifact
  - CUDA audit artifact
  - OpenCL audit artifact
  - Metal audit artifact
  - at least one backend-specific host integration pass log
  - toolkit/driver version capture
  - archived JSON + TXT reports

---

## CPU Work Items

### 1. Scope Control Inside the CPU Library

- `[OK]` CPU architecture is layered: field/scalar/point core, CT layer, signatures, protocols, wallet/convenience.
- `[TODO]` Maintain a strict feature assurance table by module.
- `[TODO]` Mark which modules are:
  - core / stable
  - protocol / evolving
  - convenience
  - experimental
- `[TODO]` Keep release notes aligned with that classification.

### 2. Parsing and Validation Unification

- `[WIP]` Strict parsing exists in many public paths.
- `[TODO]` Ensure every public parse path goes through a single strict policy for:
  - compressed pubkeys
  - x-only pubkeys
  - compact signatures
  - DER signatures
  - scalars / private keys
- `[TODO]` Add one regression checklist for "no alternate malformed input path accepted."

### 3. Sensitive Path Hardening

- `[OK]` CT layer exists and is clearly separated.
- `[OK]` CT equivalence, side-channel, fault-injection, Wycheproof, and audit tests exist.
- `[TODO]` Keep secret-bearing APIs on one consistent zeroization policy.
- `[TODO]` Review new features to ensure secret data never bypasses the intended CT path unintentionally.

### 4. Protocol Misuse Coverage

- `[OK]` MuSig2 / FROST / adaptor / Silent Payments already have test presence.
- `[TODO]` Expand adversarial and misuse-driven coverage for:
  - rogue key scenarios
  - nonce reuse / duplicate nonce
  - transcript mutation
  - signer ordering mismatch
  - replay / stale partials
  - malformed commitments
  - malformed wallet/address/seed inputs

### 5. Workflow Benchmarks

- `[OK]` `bench_unified` now includes workflow-level CPU measurements:
  - ECDH
  - Taproot
  - BIP32
  - seed-to-address
  - Silent Payments
- `[TODO]` Re-run and store publishable numbers after ongoing fixes land.
- `[TODO]` Update benchmark docs with fresh artifacts instead of quick-run placeholders.

### 6. ECIES Hardening Track

- `[WIP]` Feature exists in working tree.
- `[TODO]` Complete hardening before treating it as stable:
  - authenticated envelope design review
  - OS CSPRNG path
  - strict point parsing
  - misuse / tamper tests
  - FFI fuzz / buffer edge cases

### 7. CPU Release Checklist

- `[TODO]` For each serious release candidate, require:
  - unified CPU audit pass
  - Wycheproof ECDSA/ECDH pass
  - CT tests pass
  - fault-injection pass
  - FFI round-trip pass
  - benchmark artifact refresh
  - assurance table review

---

## Highest-Value Next Steps

If effort is limited, these are the highest-value items:

1. Finish parser strictness unification.
2. Finish ECIES hardening before broad positioning.
3. Complete ROCm validation on real AMD hardware.
4. Formalize CPU/GPU / backend parity artifacts.
5. Refresh benchmark artifacts after the current fix wave lands.

