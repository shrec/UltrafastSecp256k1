# Why UltrafastSecp256k1?

> A detailed look at what sets this library apart — not just in speed, but in engineering discipline, audit culture, and verified correctness.

## TL;DR

Traditional audits produce documents. This system produces **continuous evidence**.

| Differentiator | UltrafastSecp256k1 |
|---------------|---------------------|
| Audit model | Continuous — every commit, not one-time |
| Exploit tests | 232 PoC files, 232 registered modules, 0 failures |
| Checks per run | ~1,000,000+ assertions |
| Nightly checks | ~1,300,000+ random differential tests |
| CI workflows | 52 workflows, 16 platform combinations |
| CT verification | 5 independent pipelines (LLVM ct-verif, Valgrind taint, ct-prover, dudect, ARM64 native) |
| GPU performance | 11.00 M BIP-352 scans/s · 4.88 M ECDSA signs/s |
| Philosophy | Don't trust — reproduce |

Every exploit attempt becomes a permanent regression test. Security hardens on every commit, not just on release day.

---

## 1. Audit-First Engineering Culture

Most high-performance cryptographic libraries ship fast code and trust that it is correct.
UltrafastSecp256k1 ships fast code **and then systematically tries to break it**.

The internal self-audit system is not a layer of unit tests bolted on after the fact —
it was designed in parallel with the cryptographic implementation, as a first-class engineering artifact.

The underlying philosophy is Bitcoin-style: **don't trust, verify**. The project does
not center its trust model on a one-time PDF artifact written by someone else at a
fixed moment in the past. Instead, it tries to make assurance **continuously rerunnable**:
every important claim should be tied to code, tests, CI artifacts, benchmark logs, or
traceable documentation that another engineer can reproduce on demand.

This is why the audit framework keeps expanding with the codebase. The repository ships
not only tests, but also reviewer-facing infrastructure: structured audit artifacts,
threat-model docs, adversarial exploit tests, differential checks, and a repo-local
SQLite source graph that makes the codebase searchable as an audit surface rather than
just a pile of files.

These top-level differentiators are claim-keyed in the ledger: exploit-audit surface `A-005`, graph-assisted review `A-006`, self-audit transparency `A-007`, and benchmark reproducibility `A-004` in [docs/ASSURANCE_LEDGER.md](docs/ASSURANCE_LEDGER.md).

### What the Audit Infrastructure Covers

| Area | What is Tested | Assertion Count |
|------|---------------|-----------------|
| Field arithmetic (𝔽ₚ) | Commutativity, associativity, distributivity, canonical form, carry propagation, batch inverse, sqrt | 264,622 |
| Scalar arithmetic (ℤ_n) | Reduction mod n, overflow, GLV decomposition, negation, edge cases (0, 1, n−1) | 93,215 |
| Point operations | Infinity handling, Jacobian↔Affine round-trip, scalar multiplication, 100K stress | 116,124 |
| Constant-time layer | No secret-dependent branches, no secret-dependent memory access, formal CT verification | 120,652 |
| Exploit PoC tests | 232 dedicated adversarial PoC modules across 20+ coverage categories (`audit/test_exploit_*.cpp`) | 232 test files, 0 failures |
| Fuzz / adversarial | libFuzzer harnesses + 530K deterministic corpus adversarial checks | ~530,000+ |
| Wycheproof vectors | Google's cryptographic test vectors for ECDSA and ECDH | Hundreds of vectors |
| Independent reference linkage | Cross-validates field arithmetic against independent schoolbook oracle + golden vectors | Full suite |
| FROST / MuSig2 KAT | Protocol-level Known Answer Tests per BIP-327 and FROST spec | Full suite |
| Fault injection | Tests behaviour under simulated hardware faults (bit flips, counter skips) | Full suite |
| ABI gate | FFI round-trip stability, C ABI regression detection | Full suite |
| Performance regression | Automated micro-benchmark gate — fails CI if throughput regresses | Every push |
| **Nightly differential** | Random round-trip differential tests against reference implementations | **~1,300,000+/night** |
| **Total (audit runner)** | **unified_audit_runner** across 80 non-exploit modules + 232 exploit-PoC modules (312 total) | **~1,000,000+** |
| **Total (exploit PoC tests)** | **232 exploit-style PoC modules** across 20+ coverage categories, all in `audit/test_exploit_*.cpp` | **232 modules, 0 failures** |

All 80 non-exploit audit modules across all tested platforms return **AUDIT-READY**. Zero failures.
All 232 exploit-PoC modules pass. Zero failures across all 20+ coverage categories.

### Self-Audit Documents

| Document | Purpose |
|----------|---------|
| [AUDIT_GUIDE.md](AUDIT_GUIDE.md) | Navigation guide for security reviewers — build steps, source layout, test commands |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Historical formal audit report (v3.9.0): 641,194 checks, 0 failures |
| [AUDIT_COVERAGE.md](AUDIT_COVERAGE.md) | Current coverage matrix by module and section |
| [THREAT_MODEL.md](THREAT_MODEL.md) | Layer-by-layer risk analysis — what is in scope and out of scope |
| [SECURITY.md](SECURITY.md) | Vulnerability disclosure policy and contact |
| [docs/CT_VERIFICATION.md](docs/CT_VERIFICATION.md) | Constant-time formal verification evidence and methodology |
| [audit/AUDIT_TEST_PLAN.md](audit/AUDIT_TEST_PLAN.md) | Detailed test plan covering all 8 audit sections |
| [audit/platform-reports/](audit/platform-reports/) | Per-platform audit run results and logs |
| [tools/source_graph_kit/source_graph.py](tools/source_graph_kit/source_graph.py) | SQLite-backed repository graph for fast impact tracing, audit scoping, and reproducible review |
| [docs/ASSURANCE_LEDGER.md](docs/ASSURANCE_LEDGER.md) | Canonical claim-to-evidence ledger for public trust statements |
| [docs/AI_AUDIT_PROTOCOL.md](docs/AI_AUDIT_PROTOCOL.md) | Formal protocol for AI-assisted auditor/attacker review loops |
| [docs/FORTRESS_ROADMAP.md](docs/FORTRESS_ROADMAP.md) | Gap-closing roadmap for fortress-grade self-audit |

---

## 2. CI/CD Pipeline — 52 Automated Workflows

The continuous integration pipeline is not a basic build-and-test gate.
It is a multi-layer quality enforcement system with 52 GitHub Actions workflows
covering security, correctness, performance, supply chain, and formal analysis.

It is also only one part of the assurance model. The repository is routinely reviewed
through external-style passes as if by auditors, attackers, and bug bounty hunters,
including LLM-assisted review loops that help surface edge cases, exploit ideas, and
documentation gaps. Those passes are not treated as magic or as a replacement for
deterministic tests; they are useful because they feed new cases back into the same
reproducible audit framework.

### Workflow Index (selected)

| Workflow | What It Does | Trigger |
|----------|-------------|---------|
| `ci.yml` | Core build + full test suite across 17 configurations × 7 architectures × 5 OSes | Every push / PR |
| `preflight.yml` | Fast pre-merge smoke check — blocks merge on basic failures | Every PR |
| `nightly.yml` | Nightly stress: 1.3M+ differential checks, extended fuzz, full sanitizer run | Nightly |
| `security-audit.yml` | Runs the full `unified_audit_runner` (80 non-exploit + 232 exploit-PoC modules, ~1M assertions) plus sanitizer and warning gates | Every push |
| `audit-report.yml` | Generates and archives structured audit report artifacts | On release / manual |
| `ct-arm64.yml` | Constant-time verification on native ARM64 hardware | Every push |
| `ct-verif.yml` | Formal constant-time verification pass | Every push |
| `valgrind-ct.yml` | Valgrind memcheck + CT analysis on Linux x64 | Every push |
| `compute-sanitizer.yml` | NVIDIA compute-sanitizer GPU memory and race checks | Every push |
| `bench-regression.yml` | Performance regression gate — CI fails if throughput drops | Every push |
| `benchmark.yml` | Full benchmark suite — results published to live dashboard | On push to dev/main |
| `codeql.yml` | GitHub CodeQL static analysis (C++) | Every push |
| `clang-tidy.yml` | Clang-Tidy lint pass with project-specific rules | Every push |
| `cppcheck.yml` | CPPCheck static analysis | Every push |
| `sonarcloud.yml` | SonarCloud code quality and security rating | Every push |
| `mutation.yml` | Mutation testing — verifies test suite kills injected faults | Scheduled |
| `cflite.yml` | ClusterFuzz-Lite continuous fuzzing integration | Every push |
| `bindings.yml` | Tests all 12 language bindings (Python, Rust, Node, Go, C#, Java, Swift, ...) | Every push |
| `dependency-review.yml` | Scans dependency changes for known vulnerabilities | Every PR |
| `scorecard.yml` | OpenSSF Scorecard supply-chain security scan | Weekly |
| `klee.yml` | KLEE symbolic execution for reachability and path coverage | Scheduled |
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
| Field arithmetic correctness | Independent reference cross-validation (differential testing against schoolbook oracle + golden vectors) | Active |
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
- Published to a [live dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/) on pushes to dev/main

**Sample verified numbers (RTX 5060 Ti, CUDA 12):**

| Operation | Throughput |
|-----------|-----------|
| ECDSA sign | 4.88 M/s |
| ECDSA verify | 4.05 M/s |
| Schnorr sign (BIP-340) | 3.66 M/s |
| Schnorr verify (BIP-340) | 5.38 M/s |
| FROST partial verify | 1.34 M/s |

**Sample verified numbers (x86-64 rerun, i5-14400F, Clang 19):**

| Operation | Latency |
|-----------|---------|
| Generator multiplication (kG) | 5.9 µs |
| Scalar multiplication (kP) | 16.0 µs |
| ECDSA sign | 7.8 µs |
| ECDSA verify | 20.2 µs |

---

## 7. Assurance Model

The internal quality infrastructure described in this document represents a systematic, multi-layer correctness assurance program:

- Over **1,000,000 internal audit assertions** executed on every build
- **37 CI workflows** enforcing correctness, security, and performance on every push/PR plus scheduled assurance runs
- **Formal constant-time verification** on two independent platforms
- **Supply-chain hardening** at the OpenSSF standard
- **Nightly differential testing** at 1.3M+ additional random checks per night

> The project relies on open self-audit, reproducible evidence, graph-assisted review, and reviewer-friendly verification so anyone can inspect and challenge the implementation.
> Assurance work happens continuously through internal audit on every build, every push/PR gate, and every nightly extended run.
> The repository is structured so outside reviewers can step in and replay all evidence at any time.

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
