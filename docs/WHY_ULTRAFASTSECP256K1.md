# Why UltrafastSecp256k1?

> A detailed look at what sets this library apart — not just in speed, but in engineering discipline, audit culture, and verified correctness.

## TL;DR

Traditional audits produce documents. This system produces **continuous evidence**.

| Differentiator | UltrafastSecp256k1 |
|---------------|---------------------|
| Audit model | Continuous — every commit, not one-time |
| Exploit tests | 258 PoC files, 270 registered modules, 0 failures |
| Checks per run | ~1,000,000+ assertions |
| Deep assurance checks | ~1,300,000+ random differential tests on manual/release evidence runs |
| CI/CD model | Block-based PR/push gate + release CAAS gate + manual deep-assurance workflows |
| CT verification | 3 CT verification pipelines available as GitHub Actions workflows (`ct-verif.yml`, `valgrind-ct.yml`, `ct-prover.yml`) — triggered manually or on release tag push, not on every commit push; plus 2 local pipelines (dudect statistical, ARM64 native) |
| Philosophy | Don't trust — reproduce |

Every exploit attempt becomes a permanent regression test. Security hardens on every commit, not just on release day.

> **GPU throughput numbers:** see [GPU Profile — diagnostic, not verified against current build](#gpu-profile--diagnostic-not-verified-against-current-build) below.

---

## GPU Profile — out of scope for canonical claims

GPU throughput is **not yet measured** under the canonical protocol.
`docs/canonical_numbers.json.gpu_throughput` is `null`; there is no canonical
`docs/bench_unified_*_gpu.json` artifact. The GPU backends (CUDA / Metal /
OpenCL) compile and pass functional tests on local hardware, but performance
claims are deferred until a turbo-locked, host-pinned canonical run is
produced and committed.

The CPU CT signing numbers in the TL;DR table above are authoritative and
verified via controlled `bench_unified` runs. Treat GPU as out-of-scope for
the Bitcoin Core PR.

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

These top-level differentiators are claim-keyed in the ledger: exploit-audit surface `A-005`, graph-assisted review `A-006`, self-audit transparency `A-007`, and benchmark reproducibility `A-004` in [docs/ASSURANCE_LEDGER.md](ASSURANCE_LEDGER.md).

### What the Audit Infrastructure Covers

| Area | What is Tested | Assertion Count |
|------|---------------|-----------------|
| Field arithmetic (𝔽ₚ) | Commutativity, associativity, distributivity, canonical form, carry propagation, batch inverse, sqrt | 264,622 |
| Scalar arithmetic (ℤ_n) | Reduction mod n, overflow, GLV decomposition, negation, edge cases (0, 1, n−1) | 93,215 |
| Point operations | Infinity handling, Jacobian↔Affine round-trip, scalar multiplication, 100K stress | 116,124 |
| Constant-time layer | No secret-dependent branches, no secret-dependent memory access, formal CT verification | 120,652 |
| Exploit PoC tests | 270 dedicated adversarial PoC modules across 20+ coverage categories (`audit/test_exploit_*.cpp`) | 270 wired, 0 failures |
| Fuzz / adversarial | libFuzzer harnesses + hundreds of thousands of deterministic corpus adversarial checks (count grows with CI runs; see `audit/test_exploit_kat_corpus.cpp`) | ~hundreds of thousands+ |
| Wycheproof vectors | Google's cryptographic test vectors for ECDSA and ECDH | Hundreds of vectors |
| Independent reference linkage | Cross-validates field arithmetic against independent schoolbook oracle + golden vectors | Full suite |
| FROST / MuSig2 KAT | Protocol-level Known Answer Tests per BIP-327 and FROST spec | Full suite |
| Fault injection | Tests behaviour under simulated hardware faults (bit flips, counter skips) | Full suite |
| ABI gate | FFI round-trip stability, C ABI regression detection | Full suite |
| Performance regression | Micro-benchmark gate available for release/manual deep assurance | Manual / release |
| **Deep differential** | Random round-trip differential tests against reference implementations | **~1,300,000+ per deep run** |
| **Total (audit runner)** | **unified_audit_runner** across 166 non-exploit modules + 270 exploit-PoC modules (436 total) | **~1,000,000+** |
| **Total (exploit PoC tests)** | **270 exploit-PoC modules** (258 source files — 9 modules share a file or use inline shim stubs) across 20+ coverage categories | **270 modules, 0 failures** |

All 166 non-exploit audit modules across all tested platforms return **AUDIT-READY**. Zero failures in the current CI run (not a lifetime claim — see `docs/AUDIT_COVERAGE.md` for current status).
All 270 exploit PoCs modules pass. Zero failures in the current CI run across all 20+ coverage categories.

### Self-Audit Documents

| Document | Purpose |
|----------|---------|
| [AUDIT_GUIDE.md](AUDIT_GUIDE.md) | Navigation guide for security reviewers — build steps, source layout, test commands |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Historical formal audit report (v3.9.0): 641,194 checks, 0 failures |
| [AUDIT_COVERAGE.md](AUDIT_COVERAGE.md) | Current coverage matrix by module and section |
| [THREAT_MODEL.md](THREAT_MODEL.md) | Layer-by-layer risk analysis — what is in scope and out of scope |
| [SECURITY.md](../SECURITY.md) | Vulnerability disclosure policy and contact |
| [docs/CT_VERIFICATION.md](CT_VERIFICATION.md) | Constant-time formal verification evidence and methodology |
| [audit/AUDIT_TEST_PLAN.md](../audit/AUDIT_TEST_PLAN.md) | Detailed test plan covering all 8 audit sections |
| [audit/platform-reports/](../audit/platform-reports/) | Per-platform audit run results and logs |
| [tools/source_graph_kit/source_graph.py](../tools/source_graph_kit/source_graph.py) | SQLite-backed repository graph for fast impact tracing, audit scoping, and reproducible review |
| [docs/ASSURANCE_LEDGER.md](ASSURANCE_LEDGER.md) | Canonical claim-to-evidence ledger for public trust statements |
| [docs/AI_AUDIT_PROTOCOL.md](AI_AUDIT_PROTOCOL.md) | Formal protocol for AI-assisted auditor/attacker review loops |
| [docs/FORTRESS_ROADMAP.md](FORTRESS_ROADMAP.md) | Gap-closing roadmap for fortress-grade self-audit |

---

## 2. CI/CD Pipeline — Block-Based CAAS Flow

The continuous integration pipeline is not a basic build-and-test gate. It is a
block-based quality enforcement system: PR/push uses a small required gate,
release uses a full CAAS preflight before packaging, and heavyweight evidence
tools remain available as manual deep-assurance workflows.

It is also only one part of the assurance model. The repository is routinely reviewed
through external-style passes as if by auditors, attackers, and bug bounty hunters,
including LLM-assisted review loops that help surface edge cases, exploit ideas, and
documentation gaps. Those passes are not treated as magic or as a replacement for
deterministic tests; they are useful because they feed new cases back into the same
reproducible audit framework.

### Workflow Index (selected)

| Workflow | What It Does | Trigger |
|----------|-------------|---------|
| `gate.yml` | Impact detection, fast CAAS checks, selected profile gates, final verdict | Push / PR |
| `release.yml` | Release CAAS gate, docs/version drift check, build/package fan-out, publish | Tag / manual |
| `research-monitor.yml` | Research/CVE/ePrint intake; opens issues only for high-confidence signals | Scheduled / manual |
| Deep assurance workflows | CT-Verif, Valgrind CT, sanitizers, fuzzing, mutation, benchmarks, GPU, CodeQL, Scorecard | Manual / release policy |

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
- Reproducible via a published command in [docs/BENCHMARKS.md](BENCHMARKS.md)
- Gated by an automated performance regression check in CI (`bench-regression.yml`)
- Published to a [live dashboard](https://shrec.github.io/UltrafastSecp256k1/dev/bench/) on pushes to dev/main

**GPU throughput:** not yet measured under canonical protocol — see "GPU Profile" section above. Out of scope for Bitcoin Core evaluation.

**Canonical x86-64 numbers (i5-14400F, GCC 14.2.0, 2026-05-30):**

Source: [`docs/bench_unified_2026-05-30_gcc14_x86-64.json`](bench_unified_2026-05-30_gcc14_x86-64.json)

| Operation | Latency | Note |
|-----------|---------|------|
| CT ECDSA sign | ~22.5 µs | Production-safe CT path (GCC 14, Release+LTO) |
| CT Schnorr sign | ~18.0 µs | Production-safe CT path (GCC 14, Release+LTO) |
| ECDSA verify | ~38.0 µs | Variable-time, warm cache (correct for public data) |
| Schnorr verify (raw) | ~42.6 µs | Variable-time, no GLV cache warmup |
| Schnorr verify (cached) | ~42.2 µs | Variable-time, warm GLV cache (64-key pool) |


---

## 7. Assurance Model

The internal quality infrastructure described in this document represents a systematic, multi-layer correctness assurance program:

- Over **1,000,000 internal audit assertions** available in the CAAS evidence suite
- A **block-based CI/CD flow** enforcing correctness and security on PR/push, with release-only full evidence fan-out
- **Formal constant-time verification** on two independent platforms
- **Supply-chain hardening** at the OpenSSF standard
- **Deep differential testing** at 1.3M+ additional random checks per release/manual evidence run

> The project relies on open self-audit, reproducible evidence, graph-assisted review, and reviewer-friendly verification so anyone can inspect and challenge the implementation.
> Assurance work happens continuously through internal audit, every push/PR gate, release CAAS gates, and manual deep-evidence runs.
> The repository is structured so outside reviewers can step in and replay all evidence at any time.

---

## Summary Table

| Quality Dimension | Evidence |
|------------------|---------|
| Mathematical correctness | 473,961 audit assertions (field + scalar + point) |
| Constant-time guarantees | ct-verif, ARM64 CI, Valgrind CT, 120K CT assertions |
| Adversarial resilience | Wycheproof, fault injection, hundreds of thousands of corpus checks (grows with CI) |
| Protocol correctness | FROST/MuSig2 KAT, cross-libsecp256k1 differential |
| Memory safety | ASan, TSan, Valgrind — manual/release deep-assurance workflows |
| Static analysis | CodeQL, SonarCloud, Clang-Tidy, CPPCheck |
| Supply chain | OpenSSF Scorecard, pinned actions, SBOM, artifact attestation |
| Performance regression | Manual/release benchmark evidence gate |
| Build reproducibility | Dockerfile.reproducible + pinned toolchains |
| Self-audit documentation | AUDIT_GUIDE, AUDIT_REPORT, AUDIT_COVERAGE, THREAT_MODEL |

---

*Back to [README.md](README.md)*
