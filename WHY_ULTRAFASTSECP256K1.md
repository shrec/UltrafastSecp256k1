# Why UltrafastSecp256k1?

> A detailed look at what sets this library apart — not just in speed, but in engineering discipline, audit culture, and verified correctness.

---

## 1. Audit-First Engineering Culture

Most high-performance cryptographic libraries ship fast code and trust that it is correct.
UltrafastSecp256k1 ships fast code **and then systematically tries to break it**.

The internal self-audit system is not a layer of unit tests bolted on after the fact —
it was designed in parallel with the cryptographic implementation, as a first-class engineering artifact.

### What the Audit Infrastructure Covers

| Area | What is Tested | Assertion Count |
|------|---------------|-----------------|
| Field arithmetic (𝔽ₚ) | Commutativity, associativity, distributivity, canonical form, carry propagation, batch inverse, sqrt | 264,622 |
| Scalar arithmetic (ℤ_n) | Reduction mod n, overflow, GLV decomposition, negation, edge cases (0, 1, n−1) | 93,215 |
| Point operations | Infinity handling, Jacobian↔Affine round-trip, scalar multiplication, 100K stress | 116,124 |
| Constant-time layer | No secret-dependent branches, no secret-dependent memory access, formal CT verification | 120,652 |
| Exploit PoC tests | 78 dedicated adversarial PoC tests across 14 attack categories (`audit/test_exploit_*.cpp`) | 78 test files, 0 failures |
| Fuzz / adversarial | libFuzzer harnesses + 530K deterministic corpus adversarial checks | ~530,000+ |
| Wycheproof vectors | Google's cryptographic test vectors for ECDSA and ECDH | Hundreds of vectors |
| Fiat-Crypto linkage | Cross-validates field arithmetic against formally-verified Fiat-Crypto reference | Full suite |
| FROST / MuSig2 KAT | Protocol-level Known Answer Tests per BIP-327 and FROST spec | Full suite |
| Fault injection | Tests behaviour under simulated hardware faults (bit flips, counter skips) | Full suite |
| ABI gate | FFI round-trip stability, C ABI regression detection | Full suite |
| Performance regression | Automated micro-benchmark gate — fails CI if throughput regresses | Every push |
| **Nightly differential** | Random round-trip differential tests against reference implementations | **~1,300,000+/night** |
| **Total (audit runner)** | **unified_audit_runner** across 55 modules plus standalone audit surfaces | **~1,000,000+** |
| **Total (exploit PoC tests)** | **78 exploit-style PoC tests** across 14 attack categories, all in `audit/test_exploit_*.cpp` | **78 tests, 0 failures** |

All 55 audit modules across all tested platforms return **AUDIT-READY**. Zero failures.
All 78 exploit PoC tests pass. Zero failures across all 14 attack categories.

### Self-Audit Documents

| Document | Purpose |
|----------|---------|
| [AUDIT_GUIDE.md](AUDIT_GUIDE.md) | Navigation guide for external auditors — build steps, source layout, test commands |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Historical formal audit report (v3.9.0): 641,194 checks, 0 failures |
| [AUDIT_COVERAGE.md](AUDIT_COVERAGE.md) | Current coverage matrix by module and section |
| [THREAT_MODEL.md](THREAT_MODEL.md) | Layer-by-layer risk analysis — what is in scope and out of scope |
| [SECURITY.md](SECURITY.md) | Vulnerability disclosure policy and contact |
| [docs/CT_VERIFICATION.md](docs/CT_VERIFICATION.md) | Constant-time formal verification evidence and methodology |
| [audit/AUDIT_TEST_PLAN.md](audit/AUDIT_TEST_PLAN.md) | Detailed test plan covering all 8 audit sections |
| [audit/platform-reports/](audit/platform-reports/) | Per-platform audit run results and logs |

---

## 2. CI/CD Pipeline — 23 Automated Workflows

The continuous integration pipeline is not a basic build-and-test gate.
It is a multi-layer quality enforcement system with 23 GitHub Actions workflows
covering security, correctness, performance, supply chain, and formal analysis.

### Workflow Index

| Workflow | What It Does | Trigger |
|----------|-------------|---------|
| `ci.yml` | Core build + full test suite across 17 configurations × 7 architectures × 5 OSes | Every push / PR |
| `preflight.yml` | Fast pre-merge smoke check — blocks merge on basic failures | Every PR |
| `nightly.yml` | Nightly stress: 1.3M+ differential checks, extended fuzz, full sanitizer run | Nightly |
| `security-audit.yml` | Runs the full `unified_audit_runner` (46 modules, ~1M assertions) | Every push |
| `audit-report.yml` | Generates and archives structured audit report artifacts | On release / manual |
| `ct-arm64.yml` | Constant-time verification on native ARM64 hardware | Every push |
| `ct-verif.yml` | Formal constant-time verification pass | Every push |
| `valgrind-ct.yml` | Valgrind memcheck + CT analysis on Linux x64 | Every push |
| `bench-regression.yml` | Performance regression gate — CI fails if throughput drops | Every push |
| `benchmark.yml` | Full benchmark suite — results published to live dashboard | On push to main |
| `codeql.yml` | GitHub CodeQL static analysis (C++) | Every push |
| `clang-tidy.yml` | Clang-Tidy lint pass with project-specific rules | Every push |
| `cppcheck.yml` | CPPCheck static analysis | Every push |
| `sonarcloud.yml` | SonarCloud code quality and security rating | Every push |
| `mutation.yml` | Mutation testing — verifies test suite kills injected faults | Scheduled |
| `cflite.yml` | ClusterFuzz-Lite continuous fuzzing integration | Every push |
| `bindings.yml` | Tests all 12 language bindings (Python, Rust, Node, Go, C#, Java, Swift, ...) | Every push |
| `dependency-review.yml` | Scans dependency changes for known vulnerabilities | Every PR |
| `scorecard.yml` | OpenSSF Scorecard supply-chain security scan | Weekly |
| `valgrind-ct.yml` | Valgrind constant-time path analysis | Every push |
| `docs.yml` | Docs build and deployment validation | Every push |
| `packaging.yml` | NuGet, vcpkg, Conan, Swift Package, CocoaPods packaging validation | On release |
| `release.yml` | Full release pipeline: build, sign, attest, publish | On tag |

### Build Matrix Scale

| Dimension | Coverage |
|-----------|---------|
| Configurations | 17 (Release, Debug, ASan+UBSan, TSan, Valgrind, coverage, LTO, PGO, ...) |
| Architectures | 7 (x86-64, ARM64, RISC-V, WASM, Android ARM64, iOS ARM64, ROCm) |
| Operating systems | 5 (Linux, Windows, macOS, Android, iOS) |
| Compilers | GCC 13, Clang 17, Clang 21, MSVC 2022, AppleClang, NDK Clang |

---

## 3. Static Analysis & Sanitizer Stack

Every commit is checked by multiple independent static and dynamic analysis layers:

| Tool | What It Catches |
|------|----------------|
| **CodeQL** | Semantic security vulnerabilities, data-flow bugs |
| **SonarCloud** | Code quality, security hotspots, cognitive complexity |
| **Clang-Tidy** | Style violations, anti-patterns, performance issues |
| **CPPCheck** | Memory errors, null dereferences, buffer overflows |
| **ASan + UBSan** | Memory errors, undefined behaviour in CT paths |
| **TSan** | Data races and threading issues |
| **Valgrind memcheck** | Heap errors, uninitialized reads |
| **Valgrind CT** | Constant-time path analysis via shadow value propagation |
| **libFuzzer** | Corpus-driven bug finding in field, scalar, and point arithmetic |
| **ClusterFuzz-Lite** | Continuous fuzzing integrated into CI |

The `-Werror` flag is enforced — warnings are build failures.

---

## 4. Supply Chain Security

Cryptographic libraries are high-value supply chain targets.
UltrafastSecp256k1 applies the OpenSSF supply-chain hardening model:

- **OpenSSF Scorecard** — automated weekly supply-chain health score
- **OpenSSF Best Practices** badge — verified against the CII/OpenSSF criteria
- **Pinned GitHub Actions** — all third-party actions pinned to commit SHA, not floating tags
- **Dependency Review** — automated PR-level scan for vulnerable dependencies
- **Harden-runner** — runtime monitoring of CI runner behaviour
- **Reproducible builds** — `Dockerfile.reproducible` for bit-for-bit build verification
- **SBOM** — software bill of materials generated on release
- **Artifact attestation** — GitHub Artifact Attestation on release builds

---

## 5. Formal Verification Layers

| Layer | Method | Status |
|-------|--------|--------|
| Field arithmetic correctness | Fiat-Crypto cross-validation (differential testing against formally-verified reference) | Active |
| Constant-time (field/scalar) | `ct-verif` tool + ARM64 hardware CI | Active |
| Constant-time (point ops) | Dedicated `ct-arm64.yml` pipeline + Valgrind shadow analysis | Active |
| Wycheproof ECDSA/ECDH | Google's adversarial test vector suite | Active |
| Fault injection | Simulated hardware faults in signing/verification paths | Active |
| Cross-libsecp256k1 | Differential round-trip against Bitcoin Core's libsecp256k1 | Active |

---

## 6. Performance — Verified, Not Just Claimed

Every benchmark number in this project is:

- Produced by a pinned compiler version with exact flags documented
- Reproducible via a published command in [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- Gated by an automated performance regression check in CI (`bench-regression.yml`)
- Published to a [live dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/) on every push to main

**Sample verified numbers (RTX 5060 Ti, CUDA 12):**

| Operation | Throughput |
|-----------|-----------|
| ECDSA sign | 4.88 M/s |
| ECDSA verify | 2.44 M/s |
| Schnorr sign (BIP-340) | 3.66 M/s |
| Schnorr verify (BIP-340) | 2.82 M/s |

**Sample verified numbers (x86-64, Clang 21.1.0, `-Ofast`):**

| Operation | Latency |
|-----------|---------|
| Generator multiplication (kG) | 8 µs |
| Scalar multiplication (kP) | 25 µs |
| Field multiplication | 20 ns |
| Field squaring | 16 ns |

---

## 7. What "Not Paid-Externally Audited" Actually Means Here

UltrafastSecp256k1 has **not yet undergone a paid third-party professional audit**.
That is a factual status note, not the center of the project's security philosophy.
The project is open to external audit and continuously prepares evidence so outside reviewers can audit it at any time.
At the same time, it does not wait for a third party to begin strengthening correctness and security.

However, "not externally audited" does **not** mean "unverified." The internal quality infrastructure described in this document represents a systematic, multi-layer correctness assurance program that most open-source cryptographic libraries do not have:

- Over **1,000,000 internal audit assertions** executed on every build
- **23 CI/CD workflows** enforcing correctness, security, and performance on every commit
- **Formal constant-time verification** on two independent platforms
- **Supply-chain hardening** at the OpenSSF standard
- **Nightly differential testing** at 1.3M+ additional random checks per night

The honest summary:
> This library does **not** rely on a paid-audit badge as its primary trust story.
> It **does** rely on open self-audit, reproducible evidence, and reviewer-friendly verification so anyone can inspect and challenge the implementation.
> External audit is welcomed, but assurance work already happens continuously through internal audit on every build and every commit.

---

## Summary Table

| Quality Dimension | Evidence |
|------------------|---------|
| Mathematical correctness | 473,961 audit assertions (field + scalar + point) |
| Constant-time guarantees | ct-verif, ARM64 CI, Valgrind CT, 120K CT assertions |
| Adversarial resilience | Wycheproof, fault injection, 530K+ fuzz corpus |
| Protocol correctness | FROST/MuSig2 KAT, cross-libsecp256k1 differential |
| Memory safety | ASan, TSan, Valgrind — every commit |
| Static analysis | CodeQL, SonarCloud, Clang-Tidy, CPPCheck |
| Supply chain | OpenSSF Scorecard, pinned actions, SBOM, artifact attestation |
| Performance regression | Automated gate on every push |
| Build reproducibility | Dockerfile.reproducible + pinned toolchains |
| Self-audit documentation | AUDIT_GUIDE, AUDIT_REPORT, AUDIT_COVERAGE, THREAT_MODEL |

---

*Back to [README.md](README.md)*
