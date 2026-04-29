# UltrafastSecp256k1 -- Project Roadmap

> Last updated: 2026-04-28
> Covers: March 2026 - February 2027

This roadmap describes what the project intends to do -- and explicitly not do -- over the next 12 months. It is organized into four phases.

---

## Phase I: Core Assurance (Q1-Q2 2026) -- COMPLETE

**Goal**: Strengthen correctness guarantees and testing infrastructure.

### Completed

- [x] **Differential testing**: In-process harness comparing UltrafastSecp256k1 against libsecp256k1 v0.6.0 (FetchContent linking, CI PR runs with >=10k random cases, nightly M=100 ~1.3M checks)
- [x] **Standard test vectors**: BIP-340 (27/27), RFC 6979 (35/35), BIP-32, Wycheproof ECDSA/ECDH, MuSig2 BIP-327, FROST KAT
- [x] **Property-based testing**: Formalized algebraic invariants -- associativity, distributivity, identity, inverse, double-and-add, GLV reconstruction (89/89)
- [x] **CT leakage testing**: dudect integrated into CI (smoke per PR, nightly full 30-min statistical runs, ARM64 native on macOS M1)
- [x] **CT SafeGCD**: Constant-time Bernstein-Yang divsteps for scalar_inverse (1,635 ns, 6.5x faster than Fermat chain)
- [x] **Normalization spec**: Low-S normalization and DER strictness documented (NORMALIZATION_SPEC.md)
- [x] **FAST-mode guardrails**: C ABI auto-routes secret ops to CT layer, no opt-in flag required
- [x] **Unified audit framework**: 49+ modules across 8 sections, unified_audit_runner with JSON/SARIF output
- [x] **4-layer CT verification**: ct-verif LLVM pass + Valgrind taint + disassembly scan + dudect

---

## Phase II: Protocol & Production Hardening (Q2-Q4 2026) — ACTIVE

**Goal**: Harden advanced protocols, expand fuzzing, prepare for production deployments.

### Will Do

- **MuSig2 (BIP-327)**: Official test vectors, multi-party simulation (2/3/5-of-n), rogue-key resistance tests
- **FROST threshold signatures**: DKG simulation, signing round-trip tests, malicious participant simulation
- **Fuzzing expansion**: DER parsing, Schnorr parsing, pubkey serialize/parse, BIP-32 path parser, address encoders (bech32/base58)
- **ABI stability**: Versioning policy document, symbol visibility controls
- **Language bindings**: keep the stable `ufsecp` binding matrix green across C#, Java, Swift, Python, Go, Rust, Node.js, PHP, Ruby, Dart, and the default React Native contract-smoke lane
- **Reproducible builds**: Deterministic Docker verification, GPG-signed tags, cosign-signed release binaries
- **SBOM**: CycloneDX or SPDX bill-of-materials generation in CI
- **Multi-backend equivalence**: Formalized CPU vs CUDA, CPU vs OpenCL, CPU vs WASM output equivalence tests

### Completed During Phase II

- **Bitcoin Core secondary backend readiness**: All 10 original + 3 surface blockers
  closed. 693/693 `test_bitcoin` pass. See `BITCOIN_CORE_PR_BLOCKERS.md`.
- **DER hostile parity matrix**: `docs/DER_PARITY_MATRIX.md` — full BIP-66 edge case
  coverage vs libsecp256k1 reference behavior.
- **Cross-platform CI closure**: Linux x86_64, macOS ARM64, Windows x86_64 all green.
- **Stable binding validation closure**: shared `validate_bindings.sh` smoke matrix now covers C#, Java, Swift, Python, Go, Rust, Node.js, PHP, Ruby, and Dart.
- **React Native validation baseline**: default mock-bridge contract smoke path added, with full native RN smoke retained as an explicit opt-in lane.
- **Binding documentation alignment**: canonical bindings docs, examples, packaging notes, and README surfaces now match the validated package names and current context-based APIs.

### Won't Do (Phase II)

- External audit (Phase III)
- LTS/deprecation policy (Phase III)
- Embedded-specific certifications

---

## Phase III: GPU & Platform Parity (Q4 2026 - Q1 2027)

**Goal**: Bring all GPU/accelerator backends to **full feature parity** with the CPU implementation.

### Will Do

- **CUDA full parity**: ECDSA sign/verify, Schnorr sign/verify, batch verification, BIP-32 derivation -- all operations available on CPU must work identically on CUDA (RTX 5060 Ti currently: 4.88 M ECDSA sign/s)
- **OpenCL full parity**: Extend beyond field/point/kG to include ECDSA sign/verify, Schnorr sign/verify, batch verification on OpenCL backend
- **Apple Metal full parity**: Extend beyond field/point/kG to include signature operations on Metal (M3 Pro: currently field + point only)
- **WebAssembly parity**: Full ECDSA + Schnorr + BIP-32 in WASM build, verified against CPU reference
- **Android NDK parity**: ARM64/ARMv7/x86_64 NDK builds with full test suite, bench_unified on real hardware
- **iOS / XCFramework**: Full signing + verification in iOS builds, App Store-compatible XCFramework
- **ROCm/HIP**: AMD GPU backend with core ECC operations
- **Cross-backend differential testing**: Automated CI job that runs identical inputs through CPU, CUDA, OpenCL, Metal, WASM and verifies bit-exact output
- **Embedded CT layer**: CT signing on ESP32-S3 and STM32, currently partial (ESP32 has CT k*G but not full sign)

### Success Criteria

Every operation in the C ABI (`ufsecp_*`) must produce **identical results** across all backends. Verified by cross-backend differential tests running nightly.

### Won't Do (Phase III)

- Backend-specific optimizations beyond parity (defer to future)
- New hardware targets (FPGA, ASIC)

---

## Phase IV: Bug Bounty & External Audit (Q1-Q2 2027)

**Goal**: Achieve **external validation** and long-term community trust.

### Stage 1: Bug Bounty Program (Q1 2027)

- **Scope document**: Define in-scope targets (CT layer, C ABI, signature schemes, key derivation, parser/serializer)
- **Reward tiers**:
  - Critical (RCE, key extraction, CT bypass): up to $10,000
  - High (signature forgery, nonce bias, memory safety): up to $5,000
  - Medium (denial-of-service, malformed input crash): up to $1,000
  - Low (documentation errors, non-security bugs): recognition
- **Integration**: Link from SECURITY.md disclosure policy, responsible disclosure timeline (90 days)
- **Platform**: GitHub Security Advisories + dedicated bug bounty email
- **Duration**: Minimum 3 months before external audit engagement, to surface issues early

### Stage 2: External Security Audit (Q2 2027)

- **Scope document**: Published as [AUDIT_SCOPE.md](docs/AUDIT_SCOPE.md) (already drafted)
- **Audit firm selection**: Minimum 2 proposals from recognized cryptography audit firms (e.g., NCC Group, Trail of Bits, Cure53, Quarkslab)
- **In-scope**:
  - CT layer (scalar_inverse SafeGCD, field ops, signing)
  - C ABI boundary (ufsecp_* functions, input validation, error handling)
  - ECDSA + Schnorr signing/verification
  - BIP-32 key derivation
  - Parser/serializer (DER, compact, x-only)
  - Memory safety and secret erasure
- **Out-of-scope**: GPU backends (CUDA/OpenCL/Metal), build system, benchmark harness
- **Deliverables**:
  - Public audit report (PDF)
  - All findings fixed with regression tests
  - Re-audit of critical findings
- **Timeline**: 4-6 week engagement, report published within 30 days of completion

### Stage 3: Post-Audit Governance

- **Governance maturity**: Deprecation policy, LTS policy, pre-release checklist
- **Documentation**: User guide, performance guide, FAQ, sample applications (all exist, final polish)
- **Thread-safety guarantees**: Document and test (TSan job already in CI)
- **Performance regression CI**: Automated benchmark regression detection (>20% = warning, >50% = block)

---

## Explicit Non-Goals (Next 12 Months)

These items are **intentionally out of scope** for the 2026-2027 roadmap:

- **Formal verification** (e.g., Coq/Lean proofs) -- prohibitive effort for current team size
- **Non-secp256k1 curves** (ed25519, P-256, etc.) -- outside project scope
- **FIPS 140-3 certification** -- requires organizational infrastructure beyond current capacity
- **Custom FPGA/ASIC implementations** -- hardware projects are out of scope
- **GUI applications** -- the project is a library, not an end-user application

---

## Progress Summary

| Phase | Status | Key Milestone |
|-------|--------|---------------|
| **Phase I** -- Core Assurance | **COMPLETE** | 49+ audit modules, 4-layer CT verification, SafeGCD, 1.2M+ automated checks |
| **Phase II** -- Protocol Hardening | **ACTIVE (Q2 2026)** | CUDA w8 signing complete, infinity flag fixed, stable binding matrix validated; SBOM and remaining protocol hardening still open |
| **Phase III** -- Platform Parity | **ACTIVE** | CUDA parity is ahead; OpenCL/Metal/WASM parity and native-device platform coverage remain open |
| **Phase IV** -- Bug Bounty & Audit | **Planned (Q1-Q2 2027)** | Bug bounty first, then external audit engagement |

---

## Progress Tracking

Progress is tracked via:
- [GitHub Issues](https://github.com/shrec/UltrafastSecp256k1/issues) (task-level tracking)
- [GitHub Actions](https://github.com/shrec/UltrafastSecp256k1/actions) (CI status for all phases)
- [CHANGELOG.md](CHANGELOG.md) (completed work per release)
- [Audit Framework](docs/wiki/Audit-Framework.md) (comprehensive audit documentation)
- [Benchmark Comparison](docs/wiki/Benchmark-Comparison.md) (performance vs libsecp256k1)

This roadmap is reviewed and updated quarterly. Changes are committed to the repository and noted in release documentation.
