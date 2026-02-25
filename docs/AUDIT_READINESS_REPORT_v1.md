# Verification Transparency Report -- v3.14.0

**Status: NOT externally audited.**  
**Verification artifacts published for independent review.**

---

## Scope

This report covers **UltrafastSecp256k1 v3.14.0** internal verification results.
No external audit firm was engaged. All data below can be independently reproduced
from source using the commands in [How to Reproduce](#how-to-reproduce).

| | |
|---|---|
| Version | 3.14.0 |
| Branch | `dev` |
| Report Date | 2026-02-25 |
| Methodology | Automated deterministic + statistical |
| External Audit | **None** |

---

## 1. Coverage

| Metric | Value |
|--------|-------|
| **Dedicated audit checks** | **641,194** (8 suites, 0 failures) |
| **Extended test checks** | **~820,000+** (protocols, KAT, fuzz, differential) |
| **CTest targets** | 25+ |
| **Standard test vector suites** | BIP-340 (15), RFC 6979 (6), BIP-32 (90) |
| **libFuzzer harnesses** | 3 (field, scalar, point) |
| **Structured fuzz suites** | 2 (parser + address/BIP32/FFI) |
| **Fuzz regression corpus** | 31 pinned inputs (6 categories) |
| **CI workflows** | 14 automated pipelines |

### Platforms Tested

| Platform | Method |
|----------|--------|
| x86-64 Linux (Ubuntu 24.04) | CI (every push) |
| x86-64 Windows | CI (every push) |
| ARM64 macOS | CI (every push) |
| ARM64 Linux | CI (cross-compile + QEMU) |
| RISC-V 64 | Manual cross-compile |
| WASM (Emscripten) | CI (bindings) |
| ESP32-S3 | Manual (ESP-IDF build) |
| STM32 (Cortex-M4) | Manual (CubeMX build) |

### Sanitizers

| Sanitizer | Trigger | Findings |
|-----------|---------|----------|
| AddressSanitizer (ASan) | Every push | **0** |
| UndefinedBehaviorSanitizer (UBSan) | Every push | **0** |
| ThreadSanitizer (TSan) | Every push | **0** |
| Valgrind Memcheck | Weekly | **0** |

---

## 2. Deterministic Validation

### Cross-Library Differential (vs bitcoin-core/libsecp256k1 v0.6.0)

| Operation | Checks | Mismatches |
|-----------|-------:|:----------:|
| Generator multiplication (k*G) | 1,000 | **0** |
| Arbitrary point multiplication (k*P) | 1,000 | **0** |
| ECDSA sign determinism | 2,000 | **0** |
| ECDSA cross-verify | 1,000 | **0** |
| Schnorr BIP-340 cross-verify | 1,860 | **0** |
| Scalar arithmetic | 500 | **0** |
| Point serialization | 500 | **0** |
| **Total** | **7,860** | **0** |

Nightly extended run: **~1.3M checks** (multiplier=100). Zero mismatches.

### Standard Test Vectors

| Standard | Vectors | Pass |
|----------|--------:|:----:|
| BIP-340 (Schnorr sign + verify) | 15 | 15/15 |
| RFC 6979 (ECDSA deterministic nonce) | 6 | 6/6 |
| BIP-32 (HD derivation TV1-TV5) | 90 | 90/90 |
| FROST KAT (pinned intermediate values) | 76 | 76/76 |

### Property Tests

| Property | Checks |
|----------|-------:|
| Group associativity: (P+Q)+R == P+(Q+R) | 10,000 |
| Distributive: k(P+Q) == kP + kQ | 10,000 |
| Jacobian<->Affine round-trip | 10,000 |
| Square == Mul: sqr(x) == mul(x,x) | 10,000 |
| Inverse: x * inv(x) == 1 (field + scalar) | 20,000 |
| GLV: k1*G + k2*(lambda*G) == k*G | 1,000 |
| FAST == CT equivalence (all ops) | 120,652 |

### Roundtrip Serialization

| Format | Verified |
|--------|:--------:|
| DER encode -> decode | OK |
| Compact 64-byte encode -> decode | OK |
| Schnorr 64-byte encode -> decode | OK |
| Compressed pubkey serialize -> parse | OK |
| Uncompressed pubkey serialize -> parse | OK |
| WIF encode -> decode | OK |
| Bech32/Bech32m encode -> decode | OK |
| BIP-32 xpub/xprv serialize -> parse | OK |

---

## 3. Constant-Time Verification

### Method: dudect (Welch t-test)

Measures execution time distributions for two secret-input classes and tests
for statistically significant timing difference.

| Target | Samples | t-statistic | Threshold | Result |
|--------|--------:|:------------|:---------:|:------:|
| `ct::scalar_mul` (k=1 vs k=n-1) | 10,000 | < 4.5 | 4.5 | **PASS** |
| `ct::ecdsa_sign` (key=low vs key=high) | 10,000 | < 4.5 | 4.5 | **PASS** |
| `ct::schnorr_sign` (key=low vs key=high) | 10,000 | < 4.5 | 4.5 | **PASS** |
| `ct::field_inv` (value=1 vs value=p-1) | 10,000 | < 4.5 | 4.5 | **PASS** |
| `ct::generator_mul` (k=1 vs k=random) | 10,000 | < 4.5 | 4.5 | **PASS** |

**CI cadence**: Smoke on every push (t=25.0). Full nightly (30 min, t=4.5).

### CT Timing Ratio

| Operation | Avg ns (secret=low) | Avg ns (secret=high) | Ratio |
|-----------|--------------------:|---------------------:|------:|
| `ct::scalar_mul` | 363,380 | 351,039 | 1.035 |

Ideal: 1.0. Concern threshold: 1.2. Result is within acceptable bounds.

### Limitations

- Architecture tested: x86-64 (CI runner). Other uarch may differ.
- No formal verification (ct-verif, Vale) applied.
- Compiler may introduce secret-dependent branches at optimization levels.
- GPU backends are **NOT constant-time** by design.

---

## 4. Fuzzing

### libFuzzer Harnesses

| Harness | Input | Operations | Crashes |
|---------|-------|-----------|:-------:|
| `fuzz_field` | 32 bytes | add/sub round-trip, mul identity, sqr, inv | **0** |
| `fuzz_scalar` | 32 bytes | add/sub, mul identity, distributive law | **0** |
| `fuzz_point` | 32 bytes | on-curve, negate, compress round-trip, dbl vs add | **0** |

### Structured Fuzz Suites

| Suite | Checks | Focus | Crashes |
|-------|-------:|-------|:-------:|
| DER signature parsing | ~200K | Random blobs, valid mutations, round-trip | **0** |
| Schnorr signature parsing | ~150K | Random blobs, round-trip | **0** |
| Public key parsing | ~50K | Invalid prefix, point rejection, round-trip | **0** |
| Address encoding | ~20K | Base58Check, Bech32/Bech32m, WIF | **0** |
| BIP-32 path parsing | ~20K | Valid/invalid paths, overflow indices | **0** |
| FFI boundary | ~24K | NULL args, invalid lengths, error codes | **0** |

### Regression Corpus

31 pinned inputs across 6 categories (DER, Schnorr, pubkey, address, BIP-32, FFI).
Tracked in `tests/corpus/MANIFEST.txt`. Replayed on every CI run.

---

## 5. Multi-Party Protocols

| Protocol | Checks | What Was Verified |
|----------|-------:|-------------------|
| MuSig2 2-party | 225 | Key aggregation, nonce, partial sign, aggregate, verify |
| MuSig2 3-party | 225 | Same + additional party |
| MuSig2 5-party | 225 | Scalability |
| MuSig2 rogue-key | ~80 | Wagner-style attack detection |
| MuSig2 transcript binding | ~80 | Commitment order dependency |
| MuSig2 fault injection | ~80 | Invalid nonce/partial sig rejection |
| FROST DKG 2-of-3 | ~250 | Share generation, Feldman VSS, key derivation |
| FROST DKG 3-of-5 | ~250 | Same at larger threshold |
| FROST signing round-trip | ~300 | Nonce gen, partial sign, aggregate, BIP-340 verify |
| FROST malicious participant | ~80 | Bad share detection, below-threshold rejection |
| FROST KAT (pinned vectors) | 76 | Lagrange, share consistency, determinism, regression |

**Status**: MuSig2 and FROST are marked **Experimental**. API may change before v4.0.

---

## 6. Static Analysis

| Tool | Trigger | Configuration | Findings |
|------|---------|--------------|:--------:|
| CodeQL | Every push/PR | C/C++ security-and-quality queries | **0 critical** |
| Clang-Tidy | Every push/PR | 30+ checks (bugprone, cert, performance, clang-analyzer) | **0 blocking** |
| SonarCloud | Every push | Quality + security hotspots | **0 critical** |
| OpenSSF Scorecard | Weekly | Supply-chain security | Published |

---

## 7. Supply Chain

| Measure | Status |
|---------|--------|
| SLSA Provenance attestation | OK Every release |
| SHA-256 checksums (`SHA256SUMS.txt`) | OK Every release |
| Cosign keyless signature (.sig + .pem) | OK Every release |
| SBOM (CycloneDX 1.6) | OK Every release |
| Reproducible build (Dockerfile) | OK Available |
| Dependabot | OK Active |
| Dependency review | OK Every PR |
| Docker SHA-pinned images | OK CI + reproducible build |

---

## 8. Machine-Verifiable Artifacts

Every GitHub Release includes:

| Artifact | Format | Verification |
|----------|--------|-------------|
| `SHA256SUMS.txt` | Text | `sha256sum -c SHA256SUMS.txt` |
| `*.tar.gz.sig` / `*.zip.sig` | Cosign signature | `cosign verify-blob --signature <file>.sig --certificate <file>.pem --certificate-identity-regexp '.*' --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' <file>` |
| `sbom.cdx.json` | CycloneDX 1.6 JSON | Any SBOM viewer |
| `selftest_report.json` | SelftestReport JSON | Schema below |

### SelftestReport JSON Schema

```json
{
  "all_passed": true,
  "total": 42,
  "passed": 42,
  "mode": "ci",
  "seed": 0,
  "platform": "x86_64 clang-17 linux",
  "cases": [
    {"name": "field_add_identity", "passed": true, "detail": ""},
    {"name": "scalar_mul_KAT", "passed": true, "detail": ""}
  ]
}
```

Produced by `selftest_report(SelftestMode::ci).to_json()` -- available in C++ API
and all language bindings (Python, Rust, Go, C#, Node.js, etc.).

---

## 9. What We Do NOT Claim

| Claim | Status |
|-------|--------|
| "Fully audited" | **No.** No external audit. |
| "Production ready" | **No.** Experimental protocols (MuSig2, FROST, BIP-32, Taproot). |
| "Provably secure" | **No.** No formal verification. |
| "Constant-time guaranteed" | **Empirically tested** (dudect), not formally verified. |
| "Side-channel free" | **No.** No power analysis, EM, or fault injection testing. |
| "GPU backends are safe for secrets" | **No.** GPU is explicitly variable-time. |

### What We Do Claim

- 641,194 deterministic audit checks with 0 failures
- Bit-exact differential match against bitcoin-core/libsecp256k1 v0.6.0
- All official test vectors pass (BIP-340, RFC 6979, BIP-32)
- 0 sanitizer findings (ASan, UBSan, TSan, Valgrind)
- 0 crashes across ~580K+ fuzz iterations
- dudect timing analysis passes on x86-64 (t < 4.5)
- 14 CI workflows enforcing the above on every commit

---

## 10. Known Gaps

| Gap | Impact | Mitigation |
|-----|--------|-----------|
| No formal CT verification | Compiler may break CT at -O2 | dudect + code review |
| Single uarch timing test | Other CPUs may behave differently | Planned multi-uarch campaign |
| GPU<->CPU limited differential | GPU correctness partially verified | Planned full equivalence |
| FROST no IETF ciphersuite | No external reference vectors for secp256k1 | Self-generated KATs |
| MuSig2/FROST experimental | API may change | Documented, version-gated |

---

## How to Reproduce

Every number in this report can be independently verified:

```bash
# Clone and build
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1
git checkout v3.14.0

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_CROSS_TESTS=ON \
  -DSECP256K1_BUILD_FUZZ_TESTS=ON \
  -DSECP256K1_BUILD_PROTOCOL_TESTS=ON
cmake --build build -j

# Full test suite (~820K+ checks)
ctest --test-dir build --output-on-failure

# Audit-only (641K checks, ~24s)
ctest --test-dir build -L audit --output-on-failure

# Differential vs libsecp256k1 (7,860 checks)
ctest --test-dir build -R test_cross_libsecp256k1 -V

# dudect side-channel (smoke)
ctest --test-dir build -R ct_sidechannel_smoke -V

# Sanitizer build
cmake -S . -B build-san -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-san -j
ctest --test-dir build-san --output-on-failure

# Structured selftest JSON output
./build/examples/example_basic_usage  # or via API: selftest_report().to_json()
```

---

## Related Documents

| Document | Purpose |
|----------|---------|
| [INTERNAL_AUDIT.md](INTERNAL_AUDIT.md) | Full audit results (718 lines, per-check detail) |
| [INVARIANTS.md](INVARIANTS.md) | 108 mathematical invariants catalog |
| [TEST_MATRIX.md](TEST_MATRIX.md) | Function -> test coverage map |
| [CT_VERIFICATION.md](CT_VERIFICATION.md) | Constant-time methodology |
| [THREAT_MODEL.md](../THREAT_MODEL.md) | Layer-by-layer risk assessment |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Technical architecture |
| [SAFE_DEFAULTS.md](SAFE_DEFAULTS.md) | Recommended production configuration |
| [SECURITY.md](../SECURITY.md) | Security policy + disclosure process |
| [BUG_BOUNTY.md](BUG_BOUNTY.md) | Bug bounty scope & rewards |

---

*UltrafastSecp256k1 v3.14.0 -- Verification Transparency Report*  
*Not audited. Verification artifacts published for independent review.*
