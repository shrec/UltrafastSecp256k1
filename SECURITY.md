# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.60.x  | [OK] Active |
| 3.21.x  | [OK] Maintained |
| 3.20.x  | [!] Critical fixes only |
| < 3.20  | [FAIL] Unsupported |

Security fixes apply to the latest release on the `main` branch.

---

## Reporting a Vulnerability

**Do NOT open a public issue for suspected vulnerabilities.**

Report privately via one of:

1. **GitHub Security Advisories** (preferred):
   [Create advisory](https://github.com/shrec/UltrafastSecp256k1/security/advisories/new)
2. **Email**: [payysoon@gmail.com](mailto:payysoon@gmail.com)

We will acknowledge within **72 hours** and provide a fix timeline.

### What to Report

- Incorrect field or scalar arithmetic
- Point operation errors (addition, doubling, scalar multiplication)
- ECDSA / Schnorr signature forgery or invalid verification
- MuSig2, FROST, Adaptor Signature, or Pedersen Commitment correctness failures
- SHA-256 / tagged-hash collisions or incorrect output
- Determinism violations (RFC 6979 nonce generation)
- Constant-time violations (timing side channels in `ct::` namespace)
- Memory safety issues (buffer overflows, use-after-free)
- GPU kernel correctness issues (CUDA, ROCm, OpenCL, Metal)
- BIP-32 / BIP-44 HD derivation errors
- Coin-specific address generation errors (28-coin dispatch)
- Undefined behavior affecting cryptographic correctness

---

## Audit Status

This library has **not undergone a paid external security audit**.
That fact is not the center of the project's assurance model.
The primary security posture is an open, reproducible self-audit program that any outside reviewer can rerun.
We are open to external audit and prepare for it continuously, but we do not wait for a formal third-party process before tightening correctness and security ourselves.
The project philosophy is to strengthen assurance through internal audit on every build and every commit.
It is provided for research, educational, and experimental purposes.

> **Open self-audit and reproducible review**
>
> The project prioritizes transparent audit artifacts, reproducible commands,
> public traceability documents, and CI-backed verification that other engineers
> can independently rerun without waiting for a formal audit engagement.
> The model is deliberately Bitcoin-style: don't trust, verify.
> That includes graph-backed code navigation, continuously expanding adversarial
> tests, and frequent external-style review passes that feed new edge cases back
> into the reproducible audit framework.
> External review is welcome, and the repository is prepared so outside auditors can step in at any time.
> Meanwhile, the internal goal is to keep assurance work active, continuous, and verifiable in the open on every build and every commit.

Claim references for this section: CPU CT signing discipline `A-001`, exploit-audit surface `A-005`, graph-assisted review `A-006`, open self-audit transparency `A-007`, and ROCm/HIP status discipline `A-008` in [docs/ASSURANCE_LEDGER.md](docs/ASSURANCE_LEDGER.md).

### Audit Documentation

For auditors and security researchers, the following documents are available:

| Document | Purpose |
|----------|---------|
| [AUDIT_GUIDE.md](AUDIT_GUIDE.md) | **Start here** -- Auditor navigation, checklist, reproduction commands |
| [AUDIT_REPORT.md](AUDIT_REPORT.md) | Internal audit report (v3.66.0 baseline; test suite significantly restructured since -- see below) |
| [THREAT_MODEL.md](THREAT_MODEL.md) | Layer-by-layer risk + attack surface analysis |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Technical architecture for auditors |
| [docs/CT_VERIFICATION.md](docs/CT_VERIFICATION.md) | Constant-time methodology, dudect, known limitations |
| [docs/TEST_MATRIX.md](docs/TEST_MATRIX.md) | Function -> test coverage map with gap analysis |

### Automated Security Measures

The following automated security measures are in place:

- **CodeQL** -- static analysis on every push/PR (C/C++ security-and-quality queries)
- **OpenSSF Scorecard** -- weekly supply-chain security assessment
- **Security Audit CI** -- `-Werror -Wall -Wextra -Wpedantic -Wconversion -Wshadow` build, ASan+UBSan test suite, Valgrind memcheck (weekly + on push)
- **Clang-Tidy** -- 30+ static analysis checks (bugprone, cert, performance, readability, clang-analyzer) on every push/PR
- **SonarCloud** -- continuous code quality and security hotspot analysis
- **ASan + UBSan** -- address/undefined-behavior sanitizers in CI
- **TSan** -- thread sanitizer in CI
- **Valgrind Memcheck** -- memory error detection in Security Audit workflow
- **Artifact Attestation** -- SLSA provenance for all release artifacts
- **SHA-256 Checksums** -- `SHA256SUMS.txt` ships with every release
- **Dependabot** -- automated dependency updates for all ecosystems
- **Dependency Review** -- PR-level vulnerable dependency scanning
- **libFuzzer harnesses** -- continuous fuzz testing of field/scalar/point layers
- **Docker SHA-pinned images** -- reproducible builds with digest-pinned base images
- **dudect timing analysis** -- Welch t-test side-channel detection (1300+ line test suite)
- **Native ARM64 dudect** -- Apple Silicon (M1) smoke + full statistical analysis on macos-14 runners
- **ct-verif LLVM pass** -- deterministic compile-time constant-time verification of CT modules
- **Internal audit suite** -- 182 active CTest targets in the current validation surface, including fuzz parsers, differential tests, fault injection, CT equivalence, cross-platform KAT, Wycheproof ECDSA/ECDH, independent reference linkage, and a 70-module unified audit runner.
- **Valgrind CT taint analysis** -- MAKE_MEM_UNDEFINED + --track-origins secret-dependent branch detection
- **MuSig2/FROST dudect** -- protocol-level timing analysis (partial_sign, frost_sign, Lagrange)
- **SARIF audit output** -- `--sarif` flag for GitHub Code Scanning integration
- **Perf regression gate** -- per-push/PR benchmark gate, fails on material regressions (>50% slower on the shared-runner threshold used in CI)

### Planned Security Improvements

- [ ] Expand external reproducibility packs for outside reviewers (one-command audit replay, artifact bundles, and reviewer checklists)
- [ ] **Funded bug bounty program** -- seeking sponsors to offer financial rewards for vulnerability reports
- [ ] Formal verification of field/scalar arithmetic (Fiat-Crypto / Cryptol)
- [x] ct-verif LLVM pass integration for compile-time CT verification (`.github/workflows/ct-verif.yml`)
- [x] Native ARM64 / Apple Silicon dudect CI -- macos-14 M1 runner, smoke + full (`.github/workflows/ct-arm64.yml`)
- [x] Multi-uarch dudect campaign -- x86-64 native + RISC-V via QEMU + ARM64 cross-compile
- [x] CT buffer erasure -- volatile function-pointer trick + `explicit_bzero`/`std::atomic_signal_fence` in signing paths
- [x] value_barrier on CT mask derivation
- [x] CT branchless low-S normalization (`ct_normalize_low_s`) -- eliminates timing leak in ECDSA signing
- [x] CT branchless parity handling in Schnorr signing (`scalar_cneg` + `bool_to_mask`)
- [x] Complete secret zeroization in CT Schnorr sign (d_bytes, t_hash, rand_hash, k_prime, k)
- [x] Independent reference linkage test (schoolbook oracle cross-check, 6085 checks) + Fiat-Crypto golden vectors
- [x] Google Wycheproof ECDSA (89 vectors) + ECDH (36 vectors) integration
- [x] Valgrind CT taint CI -- secret-dependent branch detection (`.github/workflows/valgrind-ct.yml`)
- [x] MuSig2/FROST protocol-level dudect -- timing tests for partial_sign, frost_sign, Lagrange
- [x] SARIF output from audit runner -- `--sarif` CLI flag + GitHub Code Scanning upload
- [x] Performance regression gate -- per-commit 120% threshold (`.github/workflows/bench-regression.yml`)
- [x] FROST / MuSig2 reference test vectors from BIP-327/RFC-9591 implementations
- [x] Cross-ABI / FFI hostile-caller and thread-stress validation across the public C ABI

For production cryptographic systems, prefer audited libraries such as
[libsecp256k1](https://github.com/bitcoin-core/secp256k1).

See [THREAT_MODEL.md](THREAT_MODEL.md) for a layer-by-layer risk assessment.

---

## Production Readiness

| Component | Status | Notes |
|-----------|--------|-------|
| Field / Scalar arithmetic | Stable | Extensive KAT + fuzz coverage |
| Point operations (add, dbl, mul) | Stable | Deterministic selftest (smoke/ci/stress) |
| ECDSA (RFC 6979) | Stable | Deterministic nonces, input validation |
| Schnorr (BIP-340) | Stable | Tagged hashing, input validation |
| Constant-time layer (`ct::`) | Stable | No secret-dependent branches; ~5-7x penalty |
| Batch inverse / multi-scalar | Stable | Sweep-tested up to 8192 elements |
| GPU backends (CUDA, OpenCL, Metal; ROCm/HIP build path) | Beta | Functional, not constant-time |
| MuSig2 / FROST / Adaptor | Experimental | API may change |
| Pedersen Commitments | Experimental | API may change |
| Taproot (BIP-341) | Experimental | API may change |
| HD Derivation (BIP-32/44) | Experimental | API may change |
| Multi-coin address dispatch | Experimental | API may change |

---

## Security Design

### Constant-Time Operations

The constant-time layer (`ct::` namespace) provides:

- `ct::field_mul`, `ct::field_inv` -- timing-safe field arithmetic
- `ct::scalar_mul` -- timing-safe scalar multiplication
- `ct::point_add_complete`, `ct::point_dbl` -- complete addition formulas

The CT layer uses no secret-dependent branches or memory access patterns. It carries a ~5-7x performance penalty relative to the optimized (variable-time) path.

**Important**: The default (non-CT) operations prioritize performance and are NOT constant-time. Use the `ct::` variants when processing secret keys or nonces.

### Known Non-CT Exceptions (Q-Series)

The following functions are documented exceptions where a `fast::` code path was historically used in a secret-key context. Each has been assigned a tracking ID; the fix status is noted.

| ID | Function | Issue | Status |
|----|----------|-------|--------|
| Q-07 | `::ecdsa_sign_recoverable()` in `recovery.cpp` | Called by `bitcoin_sign_message()` and the libsecp256k1 shim -- uses `fast::scalar_mul(k)` and `fast::scalar_inverse(k)` on the secret nonce, leaking timing information about k. Affects Sparrow Wallet, ECIES, Ethereum `personal_sign`, and any caller using the recovery-ID signing path. | **Fixed** -- `bitcoin_sign_message()` and `secp256k1_ecdsa_sign_recoverable()` now call `ct::ecdsa_sign_recoverable()` (added in `ct_sign.cpp`), which uses `ct::generator_mul()` for R=k\*G and `ct::scalar_inverse()` for k^{-1}. The variable-time `::ecdsa_sign_recoverable()` remains available for public-data contexts (address search, batch verification) but must not be called with a secret key. |

**Rule**: any function that accepts or derives a private key or secret nonce -- including message-signing wrappers -- must route through `ct::`. Filing a new exception requires an explicit SECURITY.md entry before the code ships.

### ECDSA & Schnorr

- ECDSA: Deterministic nonces via RFC 6979 (no random nonce generation needed)
- Schnorr: BIP-340 compliant with tagged hashing
- Both signature schemes include validation of inputs (point-on-curve, scalar range checks)

### Memory Handling

- No dynamic allocation in hot paths
- **Library-side secret erasure**: `ct::schnorr_sign` and `ct::ecdsa_sign` automatically erase all intermediate nonces, scalar buffers, hash intermediates, and serialized key material via `secure_erase` (volatile function-pointer trick + `explicit_bzero` on glibc/BSD, `std::atomic_signal_fence` compiler barrier). The compiler cannot elide this erasure.
- `value_barrier` applied to CT mask derivations to prevent compiler speculation
- Fixed-size POD types used throughout (no hidden copies)
- Callers should still erase their own copies of private keys after use

---

## Fuzz Testing

libFuzzer harnesses cover the core arithmetic layers:

| Target | File | Operations |
|--------|------|------------|
| Field  | `cpu/fuzz/fuzz_field.cpp` | add/sub round-trip, mul identity, square, inverse |
| Scalar | `cpu/fuzz/fuzz_scalar.cpp` | add/sub, mul identity, distributive law |
| Point  | `cpu/fuzz/fuzz_point.cpp` | on-curve check, negate, compress round-trip, dbl vs add |

```bash
# Example: run field fuzzer
clang++ -fsanitize=fuzzer,address -O2 -std=c++20 \
  -I cpu/include cpu/fuzz/fuzz_field.cpp cpu/src/field.cpp cpu/src/field_asm.cpp \
  -o fuzz_field
./fuzz_field -max_len=64 -runs=10000000
```

---

## Scope

UltrafastSecp256k1 provides:

- Finite field arithmetic (𝔽ₚ for secp256k1 prime)
- Scalar arithmetic (mod n, curve order)
- Elliptic curve point operations (add, double, scalar multiply, multi-scalar)
- Batch inverse (Montgomery trick)
- ECDSA signatures (RFC 6979)
- Schnorr signatures (BIP-340)
- MuSig2 / FROST / Adaptor Signatures / Pedersen Commitments
- Taproot (BIP-341/342)
- HD key derivation (BIP-32/44)
- 27-coin address generation dispatch
- SHA-256 / tagged hashing
- GPU-accelerated batch operations (CUDA, ROCm, OpenCL, Metal)
- Constant-time layer (`ct::` namespace)

**Out of scope**: Key storage, wallet software, network protocols, consensus rules, and application-layer cryptographic protocols. Security responsibility for higher-level integrations remains with the integrating application.

---

## API Stability

The public API is **not yet stable**. Breaking changes may occur in any minor release before v4.0.

Layers marked "Stable" in the Production Readiness table above have mature interfaces that are unlikely to change, but no formal compatibility guarantee exists until v4.0.

For detailed stability classifications, see:
- [docs/adoption/API_STABILITY.md](docs/adoption/API_STABILITY.md) -- Tiered header classification (Stable / Provisional / Experimental / Internal)
- [docs/ABI_VERSIONING.md](docs/ABI_VERSIONING.md) -- MAJOR.MINOR.PATCH + ABI version
- [docs/DEPRECATION_POLICY.md](docs/DEPRECATION_POLICY.md) -- 2 minor release deprecation cycle
- [docs/LTS_POLICY.md](docs/LTS_POLICY.md) -- 12-month LTS, SemVer 2.0.0

---

## Vulnerability Disclosure Policy

We follow a **coordinated disclosure** process with severity-tiered SLAs.
A machine-readable contact record per **RFC 9116** is published at
[`.well-known/security.txt`](.well-known/security.txt).

### Acknowledgement & triage SLA (all reports)

| Phase | Timeline | Action |
|-------|----------|--------|
| Acknowledgment | <= 72 hours | Confirm receipt, assign tracking ID |
| Assessment | <= 7 days | Severity classification (CVSS 3.1) |

### Fix & advisory SLA (severity-tiered)

| Severity | Fix-by | Advisory-by | Notes |
|----------|--------|-------------|-------|
| Critical (CVSS >= 9.0) | <= 7 days | <= 14 days | Private key recovery, signature forgery, parser RCE |
| High (CVSS 7.0-8.9) | <= 30 days | <= 60 days | CT violation in `ct::` namespace, nonce bias, ABI memory safety |
| Medium (CVSS 4.0-6.9) | <= 60 days | <= 90 days | DoS, unexpected abort, secret-bit leakage that does not yield key |
| Low (CVSS 0.1-3.9) | Best effort | Bundled | Non-security correctness, edge-case handling |

These SLAs are **commitments**, not best-effort goals. If we miss an
SLA we file a public note in `docs/AUDIT_CHANGELOG.md` explaining
why; this is verified by the CAAS sub-gate
`scripts/audit_gate.py --disclosure-sla` (planned).

### Severity Guidelines

| CVSS | Example |
|------|---------|
| Critical (9.0+) | Private key recovery, signature forgery |
| High (7.0-8.9) | CT violation in `ct::` namespace, nonce bias |
| Medium (4.0-6.9) | Denial of service, unexpected panic/abort |
| Low (0.1-3.9) | Non-security correctness issues, edge-case handling |

### Credit policy

Reporters are credited in the changelog and the GitHub Security
Advisory (unless they request anonymity). We do not require
embargoes that are longer than 90 days; if 90 days pass without a
fix, the reporter is free to disclose.

### Bug Bounty

For detailed eligibility criteria, scope, and reward guidelines, see
[docs/BUG_BOUNTY.md](docs/BUG_BOUNTY.md).

Summary of scope:
- **In scope**: Field/scalar/point arithmetic, ECDSA/Schnorr/MuSig2/FROST correctness, constant-time violations, memory safety, GPU kernel correctness
- **Out of scope**: Performance issues, documentation errors, features not yet marked "Stable"

---

## Acknowledgments

We appreciate responsible disclosure. Contributors who report valid security issues will be credited in the changelog (unless they prefer anonymity).

---

*UltrafastSecp256k1 v3.66.0 -- Security Policy*
