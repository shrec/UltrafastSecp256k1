# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 3.12.x  | ✅ Active  |
| 3.11.x  | ⚠️ Critical fixes only |
| 3.9.x–3.10.x | ⚠️ Critical fixes only |
| < 3.9   | ❌ Unsupported |

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
- Coin-specific address generation errors (27-coin dispatch)
- Undefined behavior affecting cryptographic correctness

---

## Audit Status

This library has **not undergone an independent security audit**.
It is provided for research, educational, and experimental purposes.

However, the following automated security measures are in place:

- **CodeQL** — static analysis on every push/PR (C/C++ security-and-quality queries)
- **OpenSSF Scorecard** — weekly supply-chain security assessment
- **Security Audit CI** — `-Werror -Wall -Wextra -Wpedantic -Wconversion -Wshadow` build, ASan+UBSan test suite, Valgrind memcheck (weekly + on push)
- **Clang-Tidy** — 30+ static analysis checks (bugprone, cert, performance, readability, clang-analyzer) on every push/PR
- **SonarCloud** — continuous code quality and security hotspot analysis
- **ASan + UBSan** — address/undefined-behavior sanitizers in CI
- **TSan** — thread sanitizer in CI
- **Valgrind Memcheck** — memory error detection in Security Audit workflow
- **Artifact Attestation** — SLSA provenance for all release artifacts
- **SHA-256 Checksums** — `SHA256SUMS.txt` ships with every release
- **Dependabot** — automated dependency updates for all ecosystems
- **Dependency Review** — PR-level vulnerable dependency scanning
- **libFuzzer harnesses** — continuous fuzz testing of field/scalar/point layers
- **Docker SHA-pinned images** — reproducible builds with digest-pinned base images

### Planned Security Improvements

- [ ] Independent third-party cryptographic audit (seeking funding)
- [ ] `dudect` side-channel leakage testing for CT layer
- [ ] Formal verification of field/scalar arithmetic (Fiat-Crypto / Cryptol)
- [ ] Hardware timing analysis on multiple CPU microarchitectures
- [ ] FROST / MuSig2 protocol-level test vectors from reference implementations

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
| Constant-time layer (`ct::`) | Stable | No secret-dependent branches; ~5–7× penalty |
| Batch inverse / multi-scalar | Stable | Sweep-tested up to 8192 elements |
| GPU backends (CUDA, ROCm, OpenCL, Metal) | Beta | Functional, not constant-time |
| MuSig2 / FROST / Adaptor | Experimental | API may change |
| Pedersen Commitments | Experimental | API may change |
| Taproot (BIP-341) | Experimental | API may change |
| HD Derivation (BIP-32/44) | Experimental | API may change |
| 27-Coin Address Dispatch | Experimental | API may change |

---

## Security Design

### Constant-Time Operations

The constant-time layer (`ct::` namespace) provides:

- `ct::field_mul`, `ct::field_inv` — timing-safe field arithmetic
- `ct::scalar_mul` — timing-safe scalar multiplication
- `ct::point_add_complete`, `ct::point_dbl` — complete addition formulas

The CT layer uses no secret-dependent branches or memory access patterns. It carries a ~5–7× performance penalty relative to the optimized (variable-time) path.

**Important**: The default (non-CT) operations prioritize performance and are NOT constant-time. Use the `ct::` variants when processing secret keys or nonces.

### ECDSA & Schnorr

- ECDSA: Deterministic nonces via RFC 6979 (no random nonce generation needed)
- Schnorr: BIP-340 compliant with tagged hashing
- Both signature schemes include validation of inputs (point-on-curve, scalar range checks)

### Memory Handling

- No dynamic allocation in hot paths
- Sensitive data (private keys, nonces) should be zeroed by the caller after use
- Fixed-size POD types used throughout (no hidden copies)

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

---

## Acknowledgments

We appreciate responsible disclosure. Contributors who report valid security issues will be credited in the changelog (unless they prefer anonymity).

---

*UltrafastSecp256k1 v3.12.1 — Security Policy*
