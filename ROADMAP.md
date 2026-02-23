# UltrafastSecp256k1 — Project Roadmap

> Last updated: 2026-02-24
> Covers: March 2026 – February 2027

This roadmap describes what the project intends to do — and explicitly not do — over the next 12 months. It is organized into three phases.

---

## Phase I: Core Assurance (Q1–Q2 2026)

**Goal**: Strengthen correctness guarantees and testing infrastructure.

### Will Do

- **Differential testing**: In-process harness comparing UltrafastSecp256k1 against libsecp256k1 (FetchContent linking, CI PR runs with ≥10k random cases)
- **Standard test vectors**: Complete BIP-340 (27/27 done), RFC 6979 (35/35 done), BIP-32 vector coverage verification
- **Property-based testing**: Formalized algebraic invariants — associativity, distributivity, identity, inverse, double-and-add, GLV reconstruction (89/89 done)
- **CT leakage testing**: dudect integrated into CI (smoke mode per PR, nightly full statistical runs)
- **Normalization spec**: Document low-S normalization and DER strictness guarantees
- **FAST-mode guardrails**: Compile-time or runtime assert preventing use of non-CT paths for signing

### Won't Do (Phase I)

- Language bindings (deferred to Phase II)
- External security audit engagement (Phase III)
- Bug bounty program (Phase III)

---

## Phase II: Protocol & Production Hardening (Q3–Q4 2026)

**Goal**: Harden advanced protocols, expand fuzzing, prepare for production deployments.

### Will Do

- **MuSig2 (BIP-327)**: Official test vectors, multi-party simulation (2/3/5-of-n), rogue-key resistance tests
- **FROST threshold signatures**: DKG simulation, signing round-trip tests, malicious participant simulation
- **Fuzzing expansion**: DER parsing, Schnorr parsing, pubkey serialize/parse, BIP-32 path parser, address encoders (bech32/base58)
- **ABI stability**: Versioning policy document, symbol visibility controls
- **Language bindings**: Python and Rust bindings with CI (Go and C# stretch goals)
- **Reproducible builds**: Deterministic Docker verification, GPG-signed tags, cosign-signed release binaries
- **SBOM**: CycloneDX or SPDX bill-of-materials generation in CI
- **Multi-backend equivalence**: Formalized CPU vs CUDA, CPU vs OpenCL, CPU vs WASM output equivalence tests

### Won't Do (Phase II)

- External audit (Phase III)
- LTS/deprecation policy (Phase III)
- Embedded-specific certifications

---

## Phase III: Audit & Trust Layer (Q1 2027)

**Goal**: Achieve external validation and long-term sustainability.

### Will Do

- **External security audit**: Scope document, firm selection, public report, all findings fixed with regression tests
- **Bug bounty program**: Scope document, reward tiers, integrated with SECURITY.md disclosure policy
- **Governance maturity**: Deprecation policy, LTS policy, pre-release checklist template
- **Documentation**: Dedicated user guide, performance guide, FAQ, sample applications
- **Operational hardening**: Thread-safety guarantees document, performance regression tracking (automated benchmarks in CI)

### Won't Do (Phase III)

- Formal verification (out of scope for this cycle; may be explored in future)
- Custom hardware acceleration (FPGA/ASIC — out of scope)
- Non-secp256k1 curves (project scope is secp256k1 only)

---

## Explicit Non-Goals (Next 12 Months)

These items are **intentionally out of scope** for the 2026–2027 roadmap:

- **Formal verification** (e.g., Coq/Lean proofs) — prohibitive effort for current team size
- **Non-secp256k1 curves** (ed25519, P-256, etc.) — outside project scope
- **FIPS 140-3 certification** — requires organizational infrastructure beyond current capacity
- **Custom FPGA/ASIC implementations** — hardware projects are out of scope
- **GUI applications** — the project is a library, not an end-user application

---

## Progress Tracking

Progress is tracked via:
- [GitHub Issues](https://github.com/shrec/UltrafastSecp256k1/issues) (task-level tracking)
- [GitHub Actions](https://github.com/shrec/UltrafastSecp256k1/actions) (CI status for all phases)
- [CHANGELOG.md](CHANGELOG.md) (completed work per release)

This roadmap is reviewed and updated quarterly. Changes are committed to the repository and noted in release documentation.
