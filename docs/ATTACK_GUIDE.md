# Attack Me: External Auditor's Guide to UltrafastSecp256k1

> This document is written *for attackers*. If you want to break this library —
> or prove we missed something — start here. We want you to find real bugs more
> than we want to look clean.

**Current assurance state**: 157 exploit PoC tests, 11 fuzzer harnesses, 58 audit
modules, 39 formal Cryptol properties, dudect + Valgrind CT evidence, full
Wycheproof vector coverage. None of this means the library is bug-free. It means
we tried hard. Now you try.

---

## Table of Contents

1. [Quick-Start: One-Command Verification](#1-quick-start-one-command-verification)
2. [Top 10 Attack Ideas](#2-top-10-attack-ideas)
3. [Hardest Parts of the System](#3-hardest-parts-of-the-system)
4. [Bug Hypothesis List](#4-bug-hypothesis-list)
5. [Known Unknowns](#5-known-unknowns)
6. [Known Limitations (Intentionally Not Fixed)](#6-known-limitations-intentionally-not-fixed)
7. [Minimal Audit Targets (Fast Path)](#7-minimal-audit-targets-fast-path)
8. [Red-Team Mode: How to Run Just the Adversarial Layer](#8-red-team-mode-how-to-run-just-the-adversarial-layer)
9. [Differential Oracle Mode](#9-differential-oracle-mode)
10. [Audit Bounty: What We Want You to Find](#10-audit-bounty-what-we-want-you-to-find)
11. [Formal Threat Model](#11-formal-threat-model)
12. [Contact and Responsible Disclosure](#12-contact-and-responsible-disclosure)

---

## 1. Quick-Start: One-Command Verification

### Prerequisites

```bash
# Core build: cmake ≥ 3.23, clang/gcc, ninja
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -- -j$(nproc)

# Run the full audit suite (categories A-M, ~3 min)
bash audit/run_full_audit.sh

# Or: the unified audit runner with JSON report
./build/audit/unified_audit_runner --json /tmp/audit_out.json
cat /tmp/audit_out.json | python3 -m json.tool | grep -E "FAIL|PASS|ERROR"
```

### Formal Cryptol Layer (requires cryptol ≥ 3.0)

```bash
cd formal/cryptol
cryptol --batch Secp256k1Field.cry    # 15 properties
cryptol --batch Secp256k1Point.cry   # 10 properties
cryptol --batch Secp256k1ECDSA.cry   # 8 properties
cryptol --batch Secp256k1Schnorr.cry # 6 properties
```

All 39 properties should pass `:check` in QuickCheck mode (≤ 10 s total on modern hardware).

---

## 2. Top 10 Attack Ideas

These are ordered by our best guess of risk-vs-effort ratio for an external attacker.

### Attack 1 — Constant-Time Violations in Field Inversion (HIGH RISK)

**Target**: `cpu/src/fe_inv.cpp`, `cpu/src/fe_inv_4x64.cpp`  
**Entry**: `ufsecp_pubkey_create`, `ufsecp_ecdsa_sign`, any path that calls `field_inv`

Field inversion (needed for projective-to-affine conversion) is the most likely
place for a secret-dependent branch or table-lookup to survive compiler
transformations. The current implementation uses constant-time exponentiation
(`x^(p-2) mod p`), but:

- Compiler auto-vectorisation can lift conditional moves back to branches
- The `ASAN + UBSAN` build may differ from the `-O3 -march=native` build
- GLV decomposition of the scalar (in `cpu/src/scalar_decomp.cpp`) produces
  `k1`, `k2` with conditional negation — verify the negate is branchless

**How to probe**: run `valgrind --tool=memcheck` against `test_ct_sidechannel`,
or instrument with `ctgrind`. Compare the generated assembly's conditional move
usage with and without LTO.

---

### Attack 2 — RFC 6979 Nonce Hedging and Re-Use (HIGH RISK)

**Target**: `cpu/src/rfc6979.cpp`  
**Entry**: `ufsecp_ecdsa_sign`

RFC 6979 is deterministic nonce generation. If the implementation:
- reuses nonce `k` across two different messages with the same key → private key recovery
- fails to include the private key `d` as HMAC input → nonce is public
- truncates beyond the standard's specification → nonce bias

**Attack vector**: generate two signatures with the same (sk, msg) and verify the
nonces are identical. Generate signatures with distinct messages and verify k
differs. Extract k from the HMAC chain and verify it matches the standard test
vectors in `cpu/tests/test_rfc6979.cpp`.

Known-good: Wycheproof RFC 6979 vectors at `audit/test_exploit_wycheproof_*.cpp`.

#### Sub-class 2a — Affine Nonce Relation (ePrint 2025/705)

If two nonces satisfy k₂ = a·k₁ + b for known constants a, b, the private key
is algebraically recoverable from the two signatures without brute force.
RFC 6979 prevents this because nonces are deterministic and unrelated.

**Exploit PoC**: `audit/test_exploit_ecdsa_affine_nonce_relation.cpp` (12 sub-tests ANR-1..ANR-12)

#### Sub-class 2b — Half-Half Nonce Construction (ePrint 2023/841)

Real-world attack observed in Bitcoin: nonce constructed as `k = upper_128(hash) ‖ lower_128(private_key)`.
This leaks 128 bits of the private key per signature; two signatures allow full recovery.
RFC 6979 is immune because nonce computation never concatenates key material directly.

**Exploit PoC**: `audit/test_exploit_ecdsa_half_half_nonce.cpp` (10 sub-tests HH-1..HH-10, 13 checks)

#### Sub-class 2c — Modular Reduction Nonce Bias (CVE-2024-31497 / CVE-2024-1544)

Generating nonce as `random_512_bits mod n` instead of rejection sampling creates
a measurable statistical bias (≈2⁻²⁵⁶ per nonce). PuTTY (CVE-2024-31497) and
wolfSSL (CVE-2024-1544) were vulnerable. RFC 6979 is naturally immune because
it uses HMAC_DRBG with proper modular arithmetic.

**Exploit PoC**: `audit/test_exploit_ecdsa_nonce_modular_bias.cpp` (6 sub-tests NMB-1..NMB-6, 19 checks)

#### Sub-class 2d — Differential Fault on Deterministic Nonce (ePrint 2017/975)

If an attacker can inject a bit-flip or additive fault during RFC 6979 nonce
computation for the same message, the difference between the correct and faulted
signatures reveals the private key. Defense: constant-time nonce generation and
physical fault countermeasures. The library's deterministic nonce path is verified
to produce identical results across repeated calls.

**Exploit PoC**: `audit/test_exploit_ecdsa_differential_fault.cpp` (8 sub-tests DF-1..DF-8, 10 checks)

---

### Attack 3 — Scalar Range Boundary Cases: k = 0, k = n, k near n (MEDIUM-HIGH)

**Target**: `cpu/src/scalar.cpp`, `cpu/src/ecdsa.cpp`  
**Entry**: `ufsecp_ecdsa_sign`, `ufsecp_schnorr_sign`

The secp256k1 group order is:

```
n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
```

Vulnerable edge cases:
- k = 0 → s = 0, invalid signature (should be caught)
- k = n → k mod n = 0, same problem
- k = n+1 → k mod n = 1, predictable
- k · G = infinity (only k=0, caught, but check all rejection paths)
- k · G has r = 0 (vanishingly rare but must be re-tried in the loop)
- k · G has r ≥ n (Stark Bank CVE class — fixed as RR-004 but verify no similar paths remain)

**Formal spec**: `ecdsa_sign_r_range` and `ecdsa_sign_s_range` in
`formal/cryptol/Secp256k1ECDSA.cry` cover some of these.

---

### Attack 4 — BIP-340 Schnorr Even-Y Key Normalisation (MEDIUM)

**Target**: `cpu/src/schnorr.cpp`  
**Entry**: `ufsecp_schnorr_sign`, `ufsecp_schnorr_verify`

BIP-340 requires the public key's Y coordinate to be even. If the signer's
private key produces an odd Y, the key must be negated. A violation means:
- Verifier rejects all signatures from "odd" signers silently
- Or verifier accepts both odd and even, allowing forgery classes

**Attack**: Generate a key pair where the Y is odd (roughly half of all keys).
Confirm `schnorr_sign` normalises the key and produces a verifiable signature.
Confirm `schnorr_verify` rejects a signature produced without normalisation.
Formal coverage: `normalised_key_has_even_y`, `normalise_key_idempotent` in
`formal/cryptol/Secp256k1Schnorr.cry`.

---

### Attack 5 — MuSig2 Nonce Reuse and Rogue-Key Attack (MEDIUM)

**Target**: `cpu/src/musig2.cpp`, `cpu/src/musig2_session.cpp`  
**Entry**: `ufsecp_musig2_*`

MuSig2 is the most complex multi-signature protocol in the library.

- **Nonce reuse**: if `musig2_nonce_gen` is called twice with the same inputs
  (or cached nonces are replayed), the entire protocol is broken — private key
  recovery is trivial with two conflicting partial signatures
- **Rogue-key attack**: without MuSig2's key aggregation coefficient, a
  participant can choose a public key that cancels another participant's key out.
  Verify the aggregation hash in `cpu/src/musig2.cpp` matches BIP-327.
- **Round ordering**: session state machine — can a partial signature be injected
  if the session skips a round?

**Differential**: compare every test vector from the BIP-327 reference test suite
against the library's MuSig2 implementation.

---

### Attack 6 — DER Parser: ASN.1 Boundary and Strict-Mode Bypass (MEDIUM)

**Target**: `cpu/src/der.cpp`  
**Entry**: `ufsecp_ecdsa_sign_der`, `ufsecp_ecdsa_verify_der`

Classic DER parser bugs:
- Length field overflow: `len = 0x84 0xFF 0xFF 0xFF 0xFF` on a 4-byte body
- Negative integers without `0x00` prefix byte — Bitcoin consensus rejects these
- Trailing garbage after a valid DER blob accepted silently
- Over-long encoding of short integers (`0x02 0x04 0x00 0x00 0x00 0x2A` vs `0x02 0x01 0x2A`)

Check `audit/test_exploit_der_*.cpp` (157 exploit files total — grep for `der`).
The parser should be strict: reject all of these.

---

### Attack 7 — BIP-352 Silent Payments: Diffie-Hellman Point Clamping (MEDIUM)

**Target**: `cpu/src/bip352.cpp`  
**Entry**: `ufsecp_bip352_*`

Silent payments use a shared secret derived from Diffie-Hellman:
`ecdh = a * B_scan` for the receiver. If the implementation:
- fails to validate that `B_scan` is on the curve → invalid-curve attack
- fails to validate `B_scan` is not the point at infinity → potential key recovery
- does not hash the ECDH output → biased secret

These are the same attacks that took down early ECC implementations.

---

### Attack 8 — BIP-32 CKA_pub (Non-Hardened) Child Key Derivation (MEDIUM)

**Target**: `cpu/src/bip32.cpp`  
**Entry**: `ufsecp_bip32_derive_child`

Non-hardened BIP-32 derivation leaks the **parent private key** if the attacker
knows any child private key plus the parent public key and chain code. This is
not a library bug per se — it is a BIP-32 design property — but the library's
documentation should clearly warn against:
- mixing hardened and non-hardened children under a signing key
- exposing any non-hardened child's private key

**Attack**: given `child_priv[i]` and `parent_pub`, demonstrate parent private
key recovery. Verify the library documentation warns about this in `cpu/src/bip32.cpp`
or the public header.

---

### Attack 9 — GPU Batch Sign: Memory Aliasing and Race Conditions (LOWER RISK, HIGH IMPACT)

**Target**: `gpu/src/`, `opencl/src/`, `metal/src/`  
**Entry**: `ufsecp_gpu_ecdsa_batch_sign`, `ufsecp_gpu_schnorr_batch_sign`

GPU batch operations process thousands of inputs concurrently. Attack surface:
- Output buffer aliased to input buffer → undefined or secret-leaking output
- Race between CPU-side result read and GPU kernel completion (missing sync barrier)
- Batch size that triggers an integer overflow in buffer size calculation
- A batch where one entry has a zero private key — does the GPU kernel abort
  cleanly or produce garbage for all other entries?

**Low-cost probe**: submit a batch with `n=1` using a zero private key. Verify
the error code is correct and all output bytes are zeroed.

---

### Attack 10 — FROST Identifiable Abort: Byzantine Coordinator Injection (LOWER RISK)

**Target**: `cpu/src/frost.cpp`  
**Entry**: `ufsecp_frost_*`

FROST (Flexible Round-Optimized Schnorr Threshold) signatures support
identifiable abort: if one participant cheats, the others can identify who cheated.
Attack scenarios:
- Coordinator sends a forged binding factor — can a participant provide a valid
  partial signature that passes verification but was created with a forged `rho`?
- A participant submits a partial signature for the wrong challenge — is the
  identification step robust?
- Threshold = 1 with n = 1 — does the library degenerate gracefully to single-sig?

---

## 3. Hardest Parts of the System

These are the areas where a subtle bug is most likely to be non-obvious:

| Area | File | Why It's Hard |
|------|------|---------------|
| GLV scalar decomposition | `cpu/src/scalar_decomp.cpp` | Endomorphism splits scalar into `k1`, `k2`; conditional negation must be branchless; off-by-one in the lattice basis breaks all ECDSA |
| Field inversion chain | `cpu/src/fe_inv.cpp` | Exponent is `p-2 = 0xFFFF...FFFD`; wrong chain = wrong inverse = wrong pubkey |
| Low-S normalisation | `cpu/src/ecdsa.cpp` | `normalize_s` must be constant-time — comparison `s > n/2` must not branch on `s` |
| MuSig2 key aggregation | `cpu/src/musig2.cpp` | Aggregation coefficient `a_i = H(agg_pk, P_i)` — wrong hash input domain = rogue-key vulnerability |
| BIP-340 challenge hash | `cpu/src/schnorr.cpp` | Tagged hash with `"BIP0340/challenge"` tag; double-SHA256 prefix; wrong domain = forged sig accepted |
| RFC 6979 HMAC chain | `cpu/src/rfc6979.cpp` | V, K iteration; truncation; retry-if-zero loop — subtle state machine bugs are hard to spot |
| FROST Lagrange coefficients | `cpu/src/frost.cpp` | Lagrange interpolation in the exponent; zero denominator must be caught; wrong participant index = wrong coefficient |
| DER length encoding | `cpu/src/der.cpp` | Strict vs BER parsing; leading-zero rules for integer encoding; consensus-critical in Bitcoin context |

---

## 4. Bug Hypothesis List

These are classes of bugs we believe *could* exist, even given current coverage.
We have tested against all of these — but test coverage is not a proof.

| Hypothesis | Confidence it's absent | Test coverage |
|------------|------------------------|---------------|
| Scalar decomp conditional negate introduces timing leak | ~90% | `test_ct_sidechannel`, dudect, Valgrind CT |
| Field inverse exponent chain wrong for a specific input pattern | ~98% | Wycheproof + 10M random-walk |
| RFC 6979 nonce reuse on key = 1 or key = n-1 | ~99% | `test_rfc6979_edge_cases` |
| MuSig2 aggregation hash domain separator wrong | ~95% | BIP-327 vectors (partial) |
| FROST Lagrange denominator not checked for zero | ~92% | `test_frost_roundtrip`, `test_frost_threshold` |
| DER parser accepts long-form length with leading zero byte | ~97% | `test_exploit_der_lenfield_leading_zero` |
| BIP-352 ECDH does not reject low-order points | ~94% | `test_bip352_*`, no explicit low-order test |
| GPU batch aborts non-atomically on one bad entry | ~85% | `test_gpu_backend_matrix`, limited GPU hardware coverage |
| Schnorr even-Y normalisation skipped on aggregated key path | ~96% | `test_schnorr_*`, BIP-340 vectors |
| BIP-32 non-hardened derivation warning absent from header | ~80% | Doc review only |

Confidence is a subjective estimate, not a statistical measurement.

---

## 5. Known Unknowns

Things we know we have not fully proven or tested:

| Unknown | Category | Why It's Open |
|---------|----------|---------------|
| CT formal proof for all secret-bearing paths | Formal verification | `dudect` + Valgrind is statistical evidence; full CT proof requires SAW or similar |
| GPU kernel CT under GPU-vendor JIT recompilation | Side-channel | Vendor JIT may reintroduce branches; no `ctgrind` equivalent for GPU |
| ROCm/HIP real-device backend parity | Hardware coverage | AMD GPU hardware not yet in CI; stubs exist but untested on real hardware (RR-003) |
| FROST with t=2 and Byzantine participant 2 | Protocol | identified-abort case is partially tested, full Byzantine choreography test pending |
| Post-quantum hybrid mode signature security | Protocol | Not currently scoped; no quantum adversary model |
| Long-term key rotation under MuSig2 aggregated key | Protocol design | BIP-327 does not specify; library follows spec but no explicit test for evolving key sets |
| Side-channel leakage via CPU power/EM channels | Physical | Out of scope for software library; hardware-level mitigations are the platform's responsibility |
| Consensus-critical behaviour vs Bitcoin Core libsecp256k1 | Differential | Core secp256k1 is the gold standard; differential fuzzing is partial, not exhaustive |

---

## 6. Known Limitations (Intentionally Not Fixed)

These are design decisions or environmental constraints that limit the library's
assurance, and are deliberately not treated as bugs:

| Limitation | Reason | Reference |
|-----------|--------|-----------|
| `dudect` CT evidence is statistical, not formal | Full SAW proof is out-of-scope for this release | RR-001, `CT_VERIFICATION.md` |
| ROCm/HIP backend is a stub on real AMD hardware | No AMD hardware in CI, deferred explicitly | RR-003, `RESIDUAL_RISK_REGISTER.md` |
| BIP-32 non-hardened derivation leaks parent key (by design) | BIP-32 protocol property; not a library bug | BIP-32 §4.3 |
| MuSig2 nonce reuse protection is caller's responsibility | Per BIP-327; library provides one-shot nonce gen API | BIP-327 §security |
| `ufsecp_ctx_t` is not thread-safe for concurrent modification | Documented in header; callers must synchronise | `include/ufsecp/ufsecp.h` |
| Targets `-march=native`; binary may not run on older CPUs | Feature detection at runtime for AVX-512 is partial | `CMakeLists.txt` HAVE_AVX512F |

---

## 7. Minimal Audit Targets (Fast Path)

If you have limited time, audit these files first. Each is small, has a formal
spec or exploit test, and a bug here would be critical:

| Priority | File | Lines | Why |
|----------|------|-------|-----|
| 1 | `cpu/src/ecdsa.cpp` | ~600 | Sign + verify core; RR-004 was here |
| 2 | `cpu/src/schnorr.cpp` | ~450 | BIP-340; even-Y normalisation |
| 3 | `cpu/src/rfc6979.cpp` | ~200 | Nonce generation; determinism |
| 4 | `cpu/src/scalar_decomp.cpp` | ~300 | GLV decomposition; CT-critical |
| 5 | `cpu/src/fe_inv.cpp` | ~180 | Field inversion; exponent chain |
| 6 | `cpu/src/der.cpp` | ~350 | DER/ASN.1 parser; consensus-critical |
| 7 | `cpu/src/musig2.cpp` | ~800 | MuSig2 aggregation; rogue-key |
| 8 | `include/ufsecp/ufsecp.h` | ~400 | Public ABI; contract specification |

Run just the exploit tests covering these:

```bash
cd build
ctest -R "exploit" --output-on-failure
```

---

## 8. Red-Team Mode: How to Run Just the Adversarial Layer

### Exploit PoC tests (168 tests)

```bash
cd build
ctest -R "exploit" -j$(nproc) --output-on-failure
```

### Fuzzer harnesses (11 harnesses, LibFuzzer)

```bash
# Run one harness for 60 seconds
./build/cpu/fuzz/fuzz_ecdsa_sign -max_total_time=60 -print_final_stats=1

# All harnesses:
ls cpu/fuzz/fuzz_* build/audit/fuzz_*
```

### Differential oracle (cross-library comparison)

```bash
# Compare against libsecp256k1 reference implementation
./build/audit/differential_tests --oracle libsecp256k1 --count 100000
```

### Mutation testing

```bash
# mutation_kill_rate module (in unified_audit_runner)
./build/audit/unified_audit_runner --module mutation_kill_rate --json /tmp/mut.json
cat /tmp/mut.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('mutation_kill_rate', 'N/A'))"
```

### Formal Cryptol layer

```bash
cd formal/cryptol
cryptol --batch Secp256k1Field.cry
cryptol --batch Secp256k1Point.cry
cryptol --batch Secp256k1ECDSA.cry
cryptol --batch Secp256k1Schnorr.cry
```

### CT leak probing

```bash
# Valgrind CT check (runs test_ct_sidechannel under memcheck-CT mode)
./build/audit/unified_audit_runner --module ct_sidechannel_smoke
# dudect statistical CT evidence
./build/audit/unified_audit_runner --module dudect_ecdsa_sign
```

---

## 9. Differential Oracle Mode

The library includes a differential testing harness that runs both this library
and libsecp256k1 on the same inputs, comparing outputs byte-by-byte.

**Location**: `audit/unified_audit_runner.cpp` — module `differential_tests`  
**Reference library**: libsecp256k1 (Bitcoin Core)  
**Coverage**: ECDSA sign/verify, Schnorr sign/verify, pubkey derivation, DER encode/decode

Run it standalone:

```bash
./build/audit/unified_audit_runner --module differential_tests --json /tmp/diff.json
```

Or fuzz both simultaneously to find divergence:

```bash
./build/cpu/fuzz/fuzz_differential -max_total_time=300
```

A divergence between this library and libsecp256k1 on a valid input is a
critical finding — please report it immediately.

---

## 10. Audit Bounty: What We Want You to Find

We want independent external findings in these categories:

| Category | What We're Looking For |
|----------|------------------------|
| **Critical** | Private key recovery via any code path |
| **Critical** | Signature forgery (ECDSA or Schnorr) |
| **Critical** | Nonce reuse or nonce bias in RFC 6979 / BIP-340 |
| **Critical** | CT timing leak under standard OS-level attacker model |
| **High** | Input that crashes or causes a heap violation |
| **High** | Wrong output vs differential oracle (libsecp256k1) |
| **High** | MuSig2 / FROST rogue-key or partial-sig forgery |
| **Medium** | DER parser accepts input that libsecp256k1 rejects (or vice versa) |
| **Medium** | Incorrect error code returned (success when should fail, or vice versa) |
| **Low** | Documentation errors that could mislead an integrator |

See [SECURITY_CLAIMS.md](SECURITY_CLAIMS.md) for the formal set of claims we
make. A finding that falsifies a claim at tier Critical or High is what we most
want to hear about.

---

## 11. Formal Threat Model

### Assets

| Asset | Sensitivity | Notes |
|-------|-------------|-------|
| ECDSA/Schnorr private key | Credential | Loss = all funds controlled by that key |
| RFC 6979 / BIP-340 nonce `k` | Credential | Reuse or leak → private key recovery |
| BIP-32 master seed / chain code | Credential | Derivation hierarchy collapse |
| Aggregated MuSig2 / FROST key | Credential | Policy-equivalent to single private key |
| Signature validity bit | Integrity | Wrong accept/reject = consensus failure |

### Adversary Model

| Adversary | Capability | In scope? |
|-----------|-----------|-----------|
| Network / API attacker | Sees public keys, signatures, ciphertexts | Yes |
| Local process attacker | Same process memory read | Partial (ct tests only) |
| Cache-timing attacker (Flush+Reload) | OS-level, same physical core | Yes (dudect + Valgrind CT) |
| Physical side-channel (power / EM) | Hardware probing | No — out of scope |
| Quantum adversary | Grover / Shor | No — not scoped for this release |
| Malicious RNG | `getentropy` / `getrandom` returns attacker-controlled bytes | Partial (RFC 6979 is deterministic; BIP-340 uses `aux_rand`)  |

### Trust Boundary

The library trusts:
- The operating system's memory allocator is not compromised
- `getentropy()` returns at least 128 bits of real entropy (used only for non-deterministic modes)
- The caller zeroes output buffers it no longer needs (library zeroizes internal secrets)
- The compiler does not observe-eliminate secret-bearing writes (mitigated via `__volatile__` and `explicit_bzero`)

The library does **not** trust:
- Input length / pointer validity (all NULL-checked at ABI boundary)
- Input scalar is in range (validated before use)
- Input DER blob is well-formed (strict parser rejects malformed)

---

## 12. Contact and Responsible Disclosure

**Preferred**: open a GitHub Security Advisory at  
`https://github.com/shrec/UltrafastSecp256k1/security/advisories/new`

**Alternative**: file a confidential issue in the tracker

**Response time target**: initial acknowledgement within 72 hours.  
**Disclosure timeline**: 90-day coordinated disclosure preferred.

We follow CVE assignment through GitHub Security Advisories.  
All accepted findings will be credited in `CHANGELOG.md` and `AUDIT_CHANGELOG.md`.

---

*Generated: 2026-04-06*  
*Cross-references: [SECURITY_CLAIMS.md](SECURITY_CLAIMS.md), [RESIDUAL_RISK_REGISTER.md](RESIDUAL_RISK_REGISTER.md), [AUDIT_TRACEABILITY.md](AUDIT_TRACEABILITY.md), [SELF_AUDIT_FAILURE_MATRIX.md](SELF_AUDIT_FAILURE_MATRIX.md)*
