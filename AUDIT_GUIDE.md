# Audit Guide

**UltrafastSecp256k1 v3.50.0** -- Independent Auditor Navigation

> The audit is not a document.
> It is a reproducible system: clone, build, run, verify.
>
> If security cannot be reproduced locally, it is only a claim.

## What this guide enables

- Run the full audit locally in ~5 minutes
- Reproduce all ~1,000,000+ security checks
- Validate every public security claim
- Add new exploit tests to extend coverage

Security here is not described — it is executable.

## Reproducible Audit

This project does not ask you to trust a static report.

You can:
- clone the repository
- build locally with standard tools
- run the full audit (56 modules, ~1M+ checks)
- reproduce the CI/CD pipeline via GitHub Actions YAML

## Minimal Setup

UltrafastSecp256k1 uses a standard toolchain only:

- **CMake** + **Ninja**
- **Clang** (or GCC)
- **CUDA** (optional — for GPU audit targets)

No proprietary environment. No hidden pipeline. No special auditor toolchain.

> If you can compile it, you can verify it.

---

> This document is for auditors. Here you will find everything needed
> to evaluate the library's security, correctness, and quality.
>
> The project is open to external audit and keeps the repository audit-ready,
> but it does not wait for a formal third-party engagement before improving assurance.
> Internal audit and verification are expected to run continuously on every build and every commit.
> The intended model is not "trust the report" but "rerun the evidence".

---

## Quick Reference

| What | Where |
|------|-------|
| **This guide** | `AUDIT_GUIDE.md` (you are here) |
| **Threat model** | [THREAT_MODEL.md](THREAT_MODEL.md) |
| **Internal audit report** | [AUDIT_REPORT.md](AUDIT_REPORT.md) |
| **Security policy** | [SECURITY.md](SECURITY.md) |
| **Architecture deep-dive** | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| **CT verification** | [docs/CT_VERIFICATION.md](docs/CT_VERIFICATION.md) |
| **Test coverage matrix** | [docs/TEST_MATRIX.md](docs/TEST_MATRIX.md) |
| **API reference** | [docs/API_REFERENCE.md](docs/API_REFERENCE.md) |
| **Coding standards** | [docs/CODING_STANDARDS.md](docs/CODING_STANDARDS.md) |
| **Build instructions** | [docs/BUILDING.md](docs/BUILDING.md) |
| **Repository graph** | `tools/source_graph_kit/source_graph.py` + `tools/source_graph_kit/source_graph.db` |
| **Assurance ledger** | [docs/ASSURANCE_LEDGER.md](docs/ASSURANCE_LEDGER.md) -- canonical claims `A-001` through `A-008` |
| **AI audit protocol** | [docs/AI_AUDIT_PROTOCOL.md](docs/AI_AUDIT_PROTOCOL.md) |
| **AI review-event log** | [docs/AI_REVIEW_EVENTS.json](docs/AI_REVIEW_EVENTS.json) |
| **Fortress roadmap** | [docs/FORTRESS_ROADMAP.md](docs/FORTRESS_ROADMAP.md) |
| **External audit automation** | [docs/EXTERNAL_AUDIT_AUTOMATION.md](docs/EXTERNAL_AUDIT_AUTOMATION.md) |

---

## 1. Build & Verify (5 minutes)

```bash
# Clone and build
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run all tests (31+ CTest targets plus unified audit and standalone vectors)
ctest --test-dir build --output-on-failure

# Run with sanitizers
cmake -S . -B build-san -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-san -j
ctest --test-dir build-san --output-on-failure

# Valgrind (Linux)
ctest --test-dir build -T memcheck
```

The repository also includes a SQLite-backed source graph used for audit scoping,
impact tracing, and fast reviewer navigation. This is part of the assurance model:
it makes the codebase easier to audit repeatedly, not just easier to browse once.

If you want the repository to prepare the standard auditor bundle for you, run:

```bash
bash scripts/external_audit_prep.sh
```

For a fuller evidence package including unified audit runner output:

```bash
bash scripts/external_audit_prep.sh --with-package --build-dir build-audit
```

```bash
python3 tools/source_graph_kit/source_graph.py summary
python3 tools/source_graph_kit/source_graph.py focus ecdsa_sign 24 --core
python3 tools/source_graph_kit/source_graph.py context cpu/src/ecdsa.cpp
python3 tools/source_graph_kit/source_graph.py impact include/ufsecp/ufsecp_gpu.h
```

The AI-assisted review track is also machine-recorded. Review outcomes that are accepted,
rejected, or left unconfirmed are logged in `docs/AI_REVIEW_EVENTS.json`, and the schema
is enforced by `python3 scripts/preflight.py --ai-review`.

For top-level trust claims, start from the ledger IDs before reading the prose surfaces:
`A-001` CPU CT routing, `A-002` stable GPU ABI, `A-003` GPU parity, `A-004` benchmark reproducibility,
`A-005` exploit-audit surface, `A-006` graph-assisted review, `A-007` self-audit transparency,
and `A-008` ROCm/HIP status discipline.

---

## 2. Source Layout for Auditors

```
UltrafastSecp256k1/
|
+-- cpu/                         ★ PRIMARY AUDIT TARGET
|   +-- include/secp256k1/       -- Public API headers
|   |   +-- field.hpp            -- FieldElement (𝔽ₚ, 4x64-bit limbs)
|   |   +-- scalar.hpp           -- Scalar (ℤ_n, 4x64-bit limbs)
|   |   +-- point.hpp            -- EC Point (Jacobian + Affine)
|   |   +-- ecdsa.hpp            -- ECDSA (RFC 6979)
|   |   +-- schnorr.hpp          -- Schnorr (BIP-340)
|   |   +-- sha256.hpp           -- SHA-256
|   |   +-- glv.hpp              -- GLV endomorphism
|   |   +-- ct/                  -- Constant-time layer
|   |   |   +-- ops.hpp          -- CT arithmetic primitives
|   |   |   +-- field.hpp        -- CT field operations
|   |   |   +-- scalar.hpp       -- CT scalar operations
|   |   |   +-- point.hpp        -- CT point multiplication
|   |   +-- field_branchless.hpp -- Branchless field select/cmov
|   +-- src/                     -- Implementations
|   |   +-- field.cpp            -- Field arithmetic (mul, sqr, inv)
|   |   +-- field_asm_x64.asm    -- x86-64 BMI2/ADX assembly
|   |   +-- field_asm_arm64.cpp  -- ARM64 MUL/UMULH intrinsics
|   |   +-- field_asm_riscv64.S  -- RISC-V RV64GC assembly
|   |   +-- precompute.cpp       -- GLV decomposition, generator table
|   |   +-- ecdsa.cpp            -- ECDSA implementation
|   |   +-- schnorr.cpp          -- Schnorr implementation
|   +-- tests/                   -- Unit tests
|   |   +-- test_comprehensive.cpp   -- 25+ test categories
|   |   +-- test_ct.cpp              -- CT-layer correctness
|   |   +-- ...
|   +-- fuzz/                    -- libFuzzer harnesses
|       +-- fuzz_field.cpp       -- Field arithmetic fuzzing
|       +-- fuzz_scalar.cpp      -- Scalar arithmetic fuzzing
|       +-- fuzz_point.cpp       -- Point operation fuzzing
|
+-- tests/                       ★ AUDIT-SPECIFIC TEST SUITES
|   +-- audit_field.cpp          -- 264,000+ field arithmetic checks
|   +-- audit_scalar.cpp         -- 93,000+ scalar arithmetic checks
|   +-- audit_point.cpp          -- 116,000+ point operation checks
|   +-- audit_ct.cpp             -- 120,000+ constant-time checks
|   +-- audit_fuzz.cpp           -- 15,000+ fuzz-generated checks
|   +-- audit_perf.cpp           -- Performance benchmarks
|   +-- audit_security.cpp       -- 17,000+ security-focused checks
|   +-- audit_integration.cpp    -- 13,000+ integration checks
|   +-- test_ct_sidechannel.cpp  -- dudect-style timing analysis (1300+ lines)
|
+-- cuda/ / opencl/ / metal/     -- GPU backends (NOT constant-time)
+-- wasm/                        -- WebAssembly (Emscripten)
+-- compat/libsecp256k1_shim/    -- libsecp256k1 API compatibility
|
+-- THREAT_MODEL.md              -- Layer-by-layer risk assessment
+-- AUDIT_REPORT.md              -- Internal audit: 641,194 checks
+-- SECURITY.md                  -- Security policy + status
+-- CHANGELOG.md                 -- Version history
+-- CITATION.cff                 -- Academic citation
```

---

## 3. Critical Audit Paths

### Path A: Field Arithmetic Correctness

**Goal**: Verify all field operations mod p = 2^2⁵⁶ - 2^3^2 - 977

| Step | File | What to Check |
|------|------|---------------|
| 1 | `cpu/include/secp256k1/field.hpp` | FieldElement class, 4x64 limb layout |
| 2 | `cpu/src/field.cpp` | `add_impl`, `sub_impl`, `mul_impl`, `square_impl`, `normalize` |
| 3 | `cpu/src/field.cpp` | `from_bytes` (big-endian), `from_limbs` (little-endian) |
| 4 | `cpu/src/field.cpp` | Inversion: SafeGCD (Bernstein-Yang divsteps) |
| 5 | `tests/audit_field.cpp` | 264K checks: identity, commutativity, associativity, distributive |
| 6 | `cpu/fuzz/fuzz_field.cpp` | Fuzz: add/sub round-trip, mul identity, square, inverse |

**Key properties**: p = `0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F`

### Path B: Scalar Arithmetic Correctness

**Goal**: Verify all scalar operations mod n (curve order)

| Step | File | What to Check |
|------|------|---------------|
| 1 | `cpu/include/secp256k1/scalar.hpp` | Scalar class, 4x64 limb layout |
| 2 | `cpu/src/scalar.cpp` | add, sub, mul, inverse, negate |
| 3 | `tests/audit_scalar.cpp` | 93K checks: ring properties, boundary values |
| 4 | `cpu/fuzz/fuzz_scalar.cpp` | Fuzz: add/sub, mul identity, distributive |

**Key properties**: n = `0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141`

### Path C: Point Operations

**Goal**: Verify add, double, scalar multiply on secp256k1

| Step | File | What to Check |
|------|------|---------------|
| 1 | `cpu/include/secp256k1/point.hpp` | Affine + Jacobian representations |
| 2 | `cpu/src/point.cpp` | `add`, `dbl`, `scalar_mul`, `is_on_curve` |
| 3 | `cpu/src/precompute.cpp` | GLV decomposition, endomorphism, generator table |
| 4 | `tests/audit_point.cpp` | 116K checks: curve membership, group law |
| 5 | `cpu/fuzz/fuzz_point.cpp` | Fuzz: on-curve, negate, compress round-trip |

### Path D: Constant-Time Layer

**Goal**: Verify CT operations have no secret-dependent branches/access

| Step | File | What to Check |
|------|------|---------------|
| 1 | `cpu/include/secp256k1/ct/` | CT namespace: ops, field, scalar, point |
| 2 | `cpu/include/secp256k1/field_branchless.hpp` | `field_select` via bitwise cmov |
| 3 | `cpu/include/secp256k1/ct_utils.hpp` | Low-level CT primitives |
| 4 | `tests/audit_ct.cpp` | 120K CT correctness checks |
| 5 | `tests/test_ct_sidechannel.cpp` | dudect timing analysis (Welch t-test) |
| 6 | [docs/CT_VERIFICATION.md](docs/CT_VERIFICATION.md) | CT methodology + known limitations |

### Path E: Signatures (ECDSA + Schnorr)

**Goal**: Verify ECDSA RFC 6979, Schnorr BIP-340

| Step | File | What to Check |
|------|------|---------------|
| 1 | `cpu/include/secp256k1/ecdsa.hpp` | ECDSA API |
| 2 | `cpu/src/ecdsa.cpp` | RFC 6979 nonce, sign, verify |
| 3 | `cpu/include/secp256k1/schnorr.hpp` | Schnorr API |
| 4 | `cpu/src/schnorr.cpp` | BIP-340 tagged hashing, sign, verify |
| 5 | `tests/audit_security.cpp` | 17K security checks |
| 6 | `tests/audit_integration.cpp` | 13K integration checks |

### Path F: GPU Backends

**Goal**: Verify GPU kernel correctness (NOT constant-time)

| Step | File | What to Check |
|------|------|---------------|
| 1 | `cuda/include/secp256k1.cuh` | CUDA device functions |
| 2 | `opencl/kernels/` | OpenCL kernel sources |
| 3 | `metal/shaders/` | Metal shader sources |

**Note**: GPU backends do NOT provide constant-time guarantees. They are for public-data workloads only.

---

## 4. What Exists vs What's Planned

### [OK] Implemented Security Measures

| Measure | Status | Details |
|---------|--------|---------|
| CodeQL static analysis | Active | Every push/PR |
| OpenSSF Scorecard | Active | Weekly assessment |
| Security Audit CI | Active | -Werror, ASan+UBSan, Valgrind |
| Clang-Tidy (30+ checks) | Active | Every push/PR |
| SonarCloud | Active | Continuous quality analysis |
| ASan + UBSan | Active | CI + Debug builds |
| TSan (thread sanitizer) | Active | CI |
| Valgrind Memcheck | Active | Weekly + on push |
| Artifact Attestation | Active | SLSA provenance |
| SHA-256 Checksums | Active | Every release |
| Dependabot | Active | Automated dependency updates |
| Dependency Review | Active | PR-level scanning |
| libFuzzer harnesses | Active | Field/Scalar/Point |
| Docker SHA-pinned | Active | Reproducible builds |
| dudect timing analysis | Active | Welch t-test for CT layer |
| Internal audit suite | Active | 641,194 checks, 8 suites |

### [!] Known Gaps (Transparency)

| Gap | Priority | Notes |
|-----|----------|-------|
| External reviewer reproducibility | Medium | Keep audit playbooks, traces, and commands easy for any outside reviewer to reproduce |
| No formal verification | Medium | Fiat-Crypto / Cryptol planned |
| CT proof for branch/memory independence | Medium | CT5/CT6 remain partial; dudect, Valgrind CT, and review exist, but no full formal proof |
| Dedicated multi-threaded C ABI stress harness | Medium | TSan runs in CI, but traceability item C7 is still partial |
| Native Android device automation | Medium | Manual Android ARM64 reruns exist; no device-farm CI lane yet |
| Hardware timing analysis | Low | More microarchitectures still useful |
| GPU constant-time | N/A | By design: GPU is for public data |

---

## 5. Automated CI Workflows

23 GitHub Actions workflows provide continuous assurance across every push and PR:

| Workflow | File | Trigger | What It Does |
|----------|------|---------|--------------|
| CI | `ci.yml` | push/PR | Linux/Win/macOS/iOS/WASM/Android build + test (17 configs × 7 archs) |
| Preflight | `preflight.yml` | PR | Fast pre-merge smoke gate |
| Security Audit | `security-audit.yml` | push/PR/cron | -Werror + ASan/UBSan + **MSan** + TSan + Valgrind + dudect |
| Unified Audit | `audit-report.yml` | push/manual | Runs `unified_audit_runner` (220 modules: 73 core + 147 exploit-PoC, ~1M checks) |
| CT Verification | `ct-verif.yml` | push/PR | Formal CT verification via ct-verif LLVM pass |
| CT ARM64 | `ct-arm64.yml` | push/PR | Native ARM64 dudect on Apple M1 hardware |
| Valgrind CT | `valgrind-ct.yml` | push/PR | Valgrind CT taint analysis (CLASSIFY/DECLASSIFY) |
| Perf Regression | `bench-regression.yml` | push/PR | Blocks if any op regresses **>50%** vs baseline |
| Benchmark | `benchmark.yml` | push | Full benchmark suite, results to live dashboard |
| Nightly | `nightly.yml` | nightly 03:00 UTC | Extended differential (~1.3M checks) + dudect 30 min |
| Mutation | `mutation.yml` | scheduled | Mutation testing -- verifies tests kill injected faults |
| ClusterFuzz | `cflite.yml` | push | Continuous libFuzzer integration (ClusterFuzz-Lite) |
| Bindings | `bindings.yml` | push/PR | All 12 language bindings compile + FFI test |
| Clang-Tidy | `clang-tidy.yml` | push/PR | 30+ static analysis checks |
| CodeQL | `codeql.yml` | push/PR/cron | GitHub SAST (C++ security + quality queries) |
| CPPCheck | `cppcheck.yml` | push/PR | CPPCheck static analysis |
| Dependency Review | `dependency-review.yml` | PR | Vulnerable dependency scanning |
| Scorecard | `scorecard.yml` | weekly | OpenSSF supply-chain assessment |
| SonarCloud | `sonarcloud.yml` | push/PR | Code quality + security hotspots |
| Docs | `docs.yml` | push | Doxygen -> GitHub Pages |
| Packaging | `packaging.yml` | push/PR | .deb/.rpm/vcpkg/Conan/Swift Package validation |
| Release | `release.yml` | tag | Multi-platform artifacts + SLSA attestation |
| Discord | `discord-commits.yml` | push | Commit notifications |

---

## 6. Test Categories & Check Counts

From [AUDIT_COVERAGE.md](AUDIT_COVERAGE.md) (v3.22.0):

| Suite | Checks | Focus |
|-------|--------|-------|
| `audit_field` | 264,622 | Field arithmetic: identity, commutative, associative, distributive, inverse, boundary |
| `audit_scalar` | 93,215 | Scalar arithmetic: ring properties, overflow, negate, boundary |
| `audit_point` | 116,124 | Point ops: on-curve, group law, scalar mul, compress/decompress |
| `audit_ct` | 120,652 | CT layer: timing-safe ops, no secret-dependent branches |
| `audit_fuzz` | 15,461 | Fuzz-generated: random input correctness |
| `audit_perf` | -- | Performance benchmarks (not a correctness check) |
| `audit_security` | 17,309 | Security: nonce, validation, edge cases |
| `audit_integration` | 13,811 | End-to-end: sign -> verify, derive -> use |
| `audit_zk` | ~1,500 | ZK proofs: knowledge, DLEQ, Bulletproof range, serialization, rejection |
| **Total** | **~1,000,000+** | (includes parser fuzz 530K, Wycheproof, FROST KAT, etc.) |

---

## 7. Reproduction Commands

```bash
# Run specific audit suite
./build/tests/audit_field
./build/tests/audit_scalar
./build/tests/audit_point
./build/tests/audit_ct

# Run dudect side-channel test
./build/tests/test_ct_sidechannel

# Run with Valgrind
valgrind --tool=memcheck --leak-check=full ./build/tests/audit_field

# Run fuzzer (requires clang with libFuzzer)
clang++ -fsanitize=fuzzer,address -O2 -std=c++20 \
  -I cpu/include cpu/fuzz/fuzz_field.cpp cpu/src/field.cpp \
  -o fuzz_field
./fuzz_field -max_len=64 -runs=10000000

# Cross-check with libsecp256k1
# The compat/ layer provides a shim for API comparison
```

---

## 8. Checklist for Auditors

- [ ] **Build succeeds** with `-Werror -Wall -Wextra -Wpedantic`
- [ ] **All 55 audit modules pass** (0 failures expected) -- `./unified_audit_runner`
- [ ] **ASan + UBSan**: no memory errors or undefined behavior
- [ ] **MSan**: no uninitialized reads (`security-audit.yml` MSan job)
- [ ] **TSan**: no data races
- [ ] **Valgrind**: no leaks, no invalid reads/writes
- [ ] **Field arithmetic**: verify reduction mod p is correct in `normalize()`
- [ ] **Scalar arithmetic**: verify reduction mod n is correct
- [ ] **Point addition**: verify complete addition formula handles all edge cases
- [ ] **GLV decomposition**: verify k1 + k2*lambda == k (mod n) for random scalars
- [ ] **ECDSA nonce**: verify RFC 6979 determinism (35 vectors)
- [ ] **Schnorr**: verify BIP-340 tagged hashing (27 vectors)
- [ ] **CT layer**: no secret-dependent branches (manual code review + ct-verif + Valgrind CT)
- [ ] **CT layer**: dudect timing test passes (|t| < 4.5 for all operations)
- [ ] **SafeGCD inverse**: verify Bernstein-Yang divsteps correctness
- [ ] **from_bytes vs from_limbs**: verify endianness handling
- [ ] **ZK proofs**: `audit_zk` passes -- knowledge, DLEQ, range proof, serialization, rejection
- [ ] **Wycheproof**: all 89 ECDSA + 36 ECDH adversarial vectors correctly rejected/accepted
- [ ] **Independent reference linkage**: field arithmetic matches independent schoolbook oracle + Fiat-Crypto golden vectors
- [ ] **BIP-324**: handshake/encrypt/decrypt/tamper tests pass in `test_bip324_standalone` and FFI coverage passes in `test_ffi_coverage`
- [ ] **GPU kernels**: verify arithmetic matches CPU reference (note: no CT guarantee on GPU)
- [ ] **FROST / MuSig2**: verify protocol/adversarial suites and reference vectors remain green
- [ ] **Ethereum layer**: EIP-155 chain ID encoding, ecrecover, personal_sign round-trip

---

## 9. Contact

- **Security issues**: See [SECURITY.md](SECURITY.md) for private reporting
- **Questions**: Open a GitHub Discussion or Issue
- **Email**: [payysoon@gmail.com](mailto:payysoon@gmail.com)

---

*UltrafastSecp256k1 v3.50.0 -- Audit Guide*
