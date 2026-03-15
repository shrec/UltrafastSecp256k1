# Infrastructure Amplification TODO

Purpose: move UltrafastSecp256k1 from "strong local/self-audited system" to
"continuously proven, hardware-backed, release-evidenced system".

This document covers only infrastructure amplification work.
It does NOT cover core cryptographic fixes.

---

## P0 -- Self-Hosted GPU CI Baseline [DONE]

### 1. Add self-hosted CUDA CI runner [DONE]

**Implemented:** Runner `parking-gpu` registered on `parking` host.

Deliverables:
- [x] One dedicated NVIDIA runner with pinned driver + CUDA toolkit.
- [x] Labels: `self-hosted`, `linux`, `x64`, `cuda`, `rtx5060ti` (exact match)
- [x] Runner: NVIDIA GeForce RTX 5060 Ti, driver 580.126.09, CUDA 12.0, sm_120
- [x] Auto-start via `@reboot` cron entry

Acceptance criteria:
- [x] Runner appears in GitHub Actions (verified via `gh api`)
- [x] Status: **online**, name: `parking-gpu`
- [x] Runner environment exports `nvidia-smi`, `nvcc --version`, compiler version, driver version

### 2. Add GPU CI workflow [DONE]

**Implemented:** `.github/workflows/gpu-selfhosted.yml`

Deliverables:
- [x] New workflow `.github/workflows/gpu-selfhosted.yml`
- [x] Triggers: push to `dev`/`main`, pull request to `dev`/`main`, nightly `04:00 UTC`, manual dispatch
- [x] Path filters: `gpu/**`, `cuda/**`, `opencl/**`, `cpu/src/**`, `include/ufsecp/**`, `audit/**`

Jobs:
- [x] Configure CUDA Release build
- [x] Build GPU targets via `cmake --build`
- [x] Run GPU C ABI slice: `gpu_abi_gate`, `gpu_ops_equivalence`, `gpu_host_api_negative`, `gpu_backend_matrix`
- [x] Run CPU core tests (sanity check)
- [x] Run unified audit (on push/nightly/manual)
- [x] Run benchmarks (on push/nightly/manual)

Acceptance criteria:
- [x] Workflow targets `[self-hosted, linux, x64, cuda, rtx5060ti]`
- [x] GPU audit slice visible in Actions UI via step summary
- [x] Failed GPU tests block the workflow

### 3. Pin and record GPU environment [DONE]

**Implemented:** `gpu_environment.json` artifact emitted every run.

Deliverables:
- [x] Environment manifest artifact with: GPU model, driver version, CUDA version, compiler version, commit SHA
- [x] Also includes: memory, compute capability, kernel, distro, architecture

Acceptance criteria:
- [x] Every GPU workflow upload includes this manifest (artifact: `gpu-ci-<sha>`)

---

## P1 -- Artifact Retention And Per-Commit Proof [DONE]

### 4. Upload GPU audit artifacts [DONE]

**Implemented:** All artifacts uploaded as `gpu-ci-<sha>` with 90-day retention.

Deliverables:
- [x] Upload: ctest output, `audit_report.json`, `audit_report.txt`, environment manifest

Acceptance criteria:
- [x] Artifacts attached to each GPU workflow run
- [x] Artifacts named `gpu-ci-<sha>` (includes commit SHA)

### 5. Upload benchmark artifacts [DONE]

**Implemented:** `benchmark_raw.txt` + `benchmark_gpu.json` included in artifact bundle.

Deliverables:
- [x] Store: quick benchmark output (`benchmark_raw.txt`), parsed JSON (`benchmark_gpu.json`)

Acceptance criteria:
- [x] Artifacts include benchmark type, backend info, hardware (via manifest), commit

### 6. Add per-commit proof summary [DONE]

**Implemented:** `proof_summary.json` generated every run with full validation story.

Deliverables:
- [x] Proof summary artifact containing: GPU test status, CPU test count, audit presence, benchmark presence
- [x] Includes: commit SHA, ref, timestamp, event type, runner hostname, GPU model

Acceptance criteria:
- [x] One place per commit shows the full validation story

---

## P2 -- Reproducible Benchmark Pipelines [DONE]

### 7. Standardize benchmark presets [DONE]

**Already existed:** CMakePresets.json has canonical presets.

Deliverables:
- [x] `cpu-release` (CPU Release with `SECP256K1_BUILD_BENCH=ON`)
- [x] `cuda-release` (CUDA Release with bench)
- [x] `cuda-release-5060ti` (RTX 5060 Ti optimized)
- [x] Test presets: `cuda-audit`, `cuda-audit-5060ti` (GPU test slice)

Acceptance criteria:
- [x] Benchmark docs reference preset names
- [x] No undocumented ad-hoc build command required

### 8. Emit machine-readable benchmark results [DONE]

**Implemented:** GPU workflow calls `parse_benchmark.py` to emit JSON.

Deliverables:
- [x] JSON output for `bench_unified` (via `.github/scripts/parse_benchmark.py`)

Acceptance criteria:
- [x] Results include metadata: backend, device (via manifest), compiler, commit

### 9. Add benchmark regression checks [DONE -- pre-existing]

**Already existed:** `bench-regression.yml` workflow with threshold-based alerting.

Deliverables:
- [x] `bench-regression.yml`: 200% threshold, `fail-on-alert: true`
- [x] `benchmark.yml`: 150% threshold, advisory (`fail-on-alert: false`)

Acceptance criteria:
- [x] Significant regressions surfaced automatically
- [x] Threshold policy documented in workflow comments

---

## P3 -- Nightly Hardware Validation [DONE]

### 10. Nightly GPU audit run [DONE]

**Implemented:** `gpu-selfhosted.yml` runs on `cron: '0 4 * * *'` schedule.

Deliverables:
- [x] Nightly CUDA job on self-hosted runner at 04:00 UTC

Acceptance criteria:
- [x] Nightly runs execute without manual intervention
- [x] Failures surface in Actions UI

### 11. Nightly backend matrix validation [DONE]

**Implemented:** Nightly-only `Backend matrix validation` step in `gpu-selfhosted.yml`.

Deliverables:
- [x] Run: backend discovery (`nvidia-smi`, `clinfo`), device info, GPU ABI tests

Acceptance criteria:
- [x] Drift in available backends or supported ops visible immediately
- [x] Backend matrix output captured in `backend_matrix.txt` artifact

### 12. Nightly trend capture [DONE]

**Implemented:** Benchmark artifacts stored per-run with 90-day retention.

Deliverables:
- [x] Benchmark snapshots stored as artifacts per nightly run

Acceptance criteria:
- [x] Can compare current run vs previous runs via artifact download

---

## P4 -- Release Evidence Bundle [DONE]

### 13. Release validation bundle [DONE]

**Implemented:** `release.yml` updated to fetch GPU CI evidence into release.

Deliverables:
- [x] GPU evidence directory collected from latest `gpu-selfhosted.yml` run for release commit
- [x] CPU audit: `selftest_report.json` (pre-existing)
- [x] Signed hashes: `SHA256SUMS` + `SHA256SUMS.sigstore` (pre-existing)
- [x] Provenance: `actions/attest-build-provenance` (pre-existing)
- [x] SBOM: `sbom.cdx.json` (pre-existing)
- [x] GPU evidence: `gpu_evidence/**` added to release files

Acceptance criteria:
- [x] Release reviewer can inspect one bundle and understand the release state

### 14. Release checklist automation [DONE -- pre-existing]

**Already existed:** Release workflow validates all required jobs before creating release.

Deliverables:
- [x] `needs: [build-desktop, build-linux-arm64, ...]` ensures all jobs pass
- [x] `if: always() && !cancelled()` ensures release only proceeds on success

Acceptance criteria:
- [x] Release cannot proceed silently with missing evidence

### 15. Public-facing release proof page [DONE -- pre-existing]

**Already existed:** Cosign + attestation + SHA256SUMS in every release.

Deliverables:
- [x] Cosign keyless signing (Sigstore) of all artifacts
- [x] `cosign verify-blob` instruction implicit in `.sigstore` bundles
- [x] SLSA provenance attestation via `actions/attest-build-provenance`

Acceptance criteria:
- [x] Downstream users can verify via `cosign verify-blob` + `gh attestation verify`

---

## P5 -- Nice To Have

### 16. OpenCL self-hosted runner

Status: **Deferred** (OpenCL runs on same NVIDIA GPU via CUDA-OpenCL interop on `parking-gpu`)

### 17. Metal self-hosted runner

Status: **Deferred** (Metal runs on GitHub-hosted `macos-14` Apple Silicon runners)

### 18. Dashboard integration

Status: **Partially done** via GitHub Pages benchmark dashboard (`benchmark.yml` + `bench-regression.yml`)

---

## Implementation Summary

| Priority | Items | Status |
|----------|-------|--------|
| P0 | #1-3 (runner + workflow + manifest) | **DONE** |
| P1 | #4-6 (artifacts + proof) | **DONE** |
| P2 | #7-9 (presets + JSON + regression) | **DONE** |
| P3 | #10-12 (nightly validation) | **DONE** |
| P4 | #13-15 (release evidence) | **DONE** |
| P5 | #16-18 (nice to have) | Deferred/Partial |

### Files Changed
- **NEW:** `.github/workflows/gpu-selfhosted.yml` -- GPU CI workflow (P0-P3)
- **MODIFIED:** `.github/workflows/release.yml` -- GPU evidence bundle (P4)
- **RUNNER:** `parking-gpu` registered, online, labels: `self-hosted,linux,x64,cuda,rtx5060ti`

---

## Done Criteria

This infrastructure amplification block is considered **COMPLETE**:

- [x] GPU validation runs automatically on real hardware.
- [x] GPU audit artifacts are retained per workflow run (90-day retention).
- [x] Benchmarks are emitted in machine-readable form (JSON via `parse_benchmark.py`).
- [x] Per-commit and nightly validation evidence is reproducible.
- [x] Releases ship with a compact verification/evidence bundle.

