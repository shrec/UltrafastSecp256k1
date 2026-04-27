# Security Incident Timeline

**UltrafastSecp256k1 — Chronological Record of Security Findings and Resolutions**

Version: 1.0
Date: 2026-04-27
Status: Active

---

## Purpose

This document is a transparency record. It lists every security finding —
whether a correctness bug, a timing or constant-time failure, an API contract
violation, a parity gap with the reference implementation, or a build system
issue with security implications — in chronological order with:

- What was found and how
- What the actual impact was (not a worst-case characterization)
- How it was fixed
- What permanent regression test or gate was added to prevent recurrence

The goal is an honest account of the library's security history, useful to
auditors, downstream integrators, and contributors evaluating the project.

---

## Findings

---

### 2026-04-03 — RR-004 CLOSED: ECDSA large-x r comparison (Stark Bank CVE class)

**Discovery**: During review of Wycheproof ECDSA test vector coverage, test
case `tcId 346` was found to fail. Investigation traced the failure to the
`r_less_than_pmn` check in `cpu/src/ecdsa.cpp`. The constants used in the
comparison were incorrect: the `p minus n` (PMN) values were wrong in both the
FE52 (52-bit limb) and 4x64 (64-bit limb) field paths.

**Background**: The secp256k1 field prime p and the group order n satisfy
p > n, with p − n ≈ 4.3 × 10^38. A nonce scalar k produces an x-coordinate
k·G.x that lies in the range [0, p−1]. For ECDSA, the signature value r is
this x-coordinate reduced modulo n. If k·G.x is in the range [n, p−1]
(approximately 2^-128 probability per signature), then r = k·G.x − n, and
the verifier must accept both r and r+n as valid values of the x-coordinate.
The `r_less_than_pmn` check is what implements this. With incorrect PMN
constants, signatures in this rare range were erroneously rejected.

**Impact**: False rejection. Signatures that should have been accepted (per
the secp256k1 specification and RFC 6979) were rejected. This is a correctness
failure, not a security downgrade — the function rejected too many signatures
rather than accepting too many. However, the behavior diverged from
libsecp256k1's verification, creating an interoperability hazard. Any
signature produced by libsecp256k1 with a nonce in this range would fail
verification under UltrafastSecp256k1.

Practical probability of a given signature hitting this range: approximately
(p − n) / n ≈ 2^-128, or about one in 3.4 × 10^38 signatures. The Wycheproof
test set includes a deliberately constructed test vector for this case
(tcId 346), which is why the failure was detectable in CI.

**Fix**: Corrected PMN constants in both `fe52` and `4x64` field paths in
`cpu/src/ecdsa.cpp`. Commit `ea8cfb3c`.

**Regression test**: Wycheproof tcId 346 is now passing and is included in
the permanent Wycheproof test suite run in CI. Exploit PoC
`audit/test_exploit_stark_bank_ecdsa_r.cpp` was added as a permanent
regression test, exercising the boundary condition directly with a
constructed nonce that produces k·G.x in the [n, p−1] range.

**Classification**: Correctness / interoperability. Not exploitable for
signature forgery. Analogous to the Stark Bank CVE class, which involved a
different implementation of the same secp256k1 large-x corner case.

---

### 2026-04-27 — Shim parser parity gap identified and closed

**Discovery**: A comprehensive audit of the libsecp256k1 shim
(`include/ufsecp/impl/ufsecp_libsecp_shim.cpp`) was conducted as part of
preparation for the Bitcoin Core alternative backend PR. The audit checked
each shim function against the libsecp256k1 API contract in `secp256k1.h`,
specifically for scalar and field element input validation behavior.

Every scalar-input shim function was found to use `Scalar::from_bytes` to
parse caller-provided scalars. `Scalar::from_bytes` reduces the input
modulo n, silently accepting inputs in the range [n, 2^256−1] as their
modular reduction. The libsecp256k1 contract is to reject such inputs with
a return value of 0/false.

**Affected functions**:
- `secp256k1_ec_seckey_verify` — accepted keys >= n (different key than caller expected)
- `secp256k1_ec_seckey_tweak_add` — accepted tweak scalars >= n
- `secp256k1_ec_seckey_tweak_mul` — accepted tweak scalars >= n
- `secp256k1_ec_pubkey_create` — accepted secret keys >= n
- `secp256k1_ec_pubkey_tweak_add` — accepted tweak scalars >= n
- `secp256k1_ec_pubkey_tweak_mul` — accepted tweak scalars >= n
- `secp256k1_ecdsa_signature_parse_compact` — blind memcpy, no validation at all
- `secp256k1_keypair_create` — accepted secret keys >= n
- `secp256k1_xonly_pubkey_parse` — used `from_bytes` for x-coordinate (no check against p)
- `secp256k1_schnorrsig_sign32` — accepted secret keys >= n

`secp256k1_ecdsa_signature_parse_compact` was the most severe: it used a
raw `memcpy` for both r and s values with no validation. An r value equal to
n would parse successfully (stored as the unreduced value n), then silently
reduce to 0 at verification time, causing a false rejection. An s value equal
to n was similarly affected.

**Criticality**: MEDIUM. The differential behavior means this shim accepted
inputs that libsecp256k1 rejects. For callers who provide keys >= n, the
silently reduced key is a different key than the one the caller believes they
are using. For keys that are exactly n or n+1, the reduced key is 0 or 1
respectively — catastrophically weak keys, though the probability of a random
key landing here is negligible. The parser gap (no validation in
`signature_parse_compact`) could lead to inconsistency between parse and use
if the signature were later serialized and re-parsed.

There is no known exploit that generates a key >= n by accident in normal
operation. The primary risk is in applications that forward externally
supplied key material through the shim without additional validation, where
the difference in acceptance behavior could mask an upstream error.

**Fix**: All boundary functions were updated to use
`Scalar::parse_bytes_strict_nonzero` (rejects inputs == 0 or >= n) or
`Scalar::parse_bytes_strict` (rejects inputs >= n, allows 0 where appropriate)
and `FieldElement::parse_bytes_strict` (rejects inputs >= p) as appropriate
for each function's semantics. Commit `d427c975`.

**Regression test**: `scripts/check_libsecp_shim_parity.py` was added as a
permanent CI gate. It runs the shim and libsecp256k1 in parallel against a
systematic set of boundary inputs (values 0, 1, n-1, n, n+1, p-1, p, p+1,
2^256-1, and random values in each range) for every shim function and fails
if the return values or output buffers differ.

**Classification**: API contract violation / parity gap. Not a timing or
constant-time issue. Potentially relevant to Bitcoin Core compatibility
testing.

---

### 2026-04-27 — CT default cleanup: non-CT signing paths on non-MSVC

**Discovery**: Inspection of `cpu/src/ecdsa.cpp`, `cpu/src/recovery.cpp`,
and `cpu/src/schnorr.cpp` found that the public C++ signing APIs used an
`#ifdef _MSC_VER` guard. On MSVC builds, signing was routed through
`ct::generator_mul_blinded` and `ct::scalar_inverse`. On GCC and Clang
builds — the majority of production deployments — signing used the
variable-time path: `Point::generator().scalar_mul(k)` for the nonce-to-point
step and `Scalar::inverse(k)` for the modular inverse of the nonce in the
ECDSA s computation.

The stated reason for the guard in the original commit message was a
performance regression on non-MSVC compilers when using the CT path — the
compiler was generating suboptimal code for the CT selection primitives
under GCC/Clang `-O2`. The chosen workaround was to route non-MSVC to the
fast path. This was incorrect: the right fix is to address the compiler
interaction, not to remove CT from non-MSVC platforms.

**Affected functions (GCC/Clang, non-CT)**:
- `secp256k1::ecdsa_sign` — nonce scalar multiplication and inverse non-CT
- `secp256k1::ecdsa_sign_hedged` — same
- `secp256k1::ecdsa_sign_recoverable` (in `recovery.cpp`) — same
- `secp256k1::schnorr_pubkey` — secret key multiplication non-CT
- `secp256k1::schnorr_keypair_create` — secret key multiplication non-CT
- `secp256k1::schnorr_sign` — nonce multiplication non-CT
- `secp256k1::schnorr_xonly_from_keypair` — secret key multiplication non-CT

The C ABI wrappers (`ufsecp_*`) were not affected: they dispatched directly
to `ct::*` primitives without going through the public C++ API. Applications
using only the `ufsecp_*` ABI were using CT paths on all platforms. The
vulnerability was specific to code using the public C++ namespace API.

**Impact**: HIGH for public C++ API users on GCC/Clang. The nonce scalar k
was computed via variable-time scalar multiplication, producing timing
variation proportional to the Hamming weight of the nonce. This constitutes
a timing oracle on the nonce, which is sufficient to mount an HNP lattice
attack given enough signatures. The MSVC path was unaffected.

The C ABI (`ufsecp_*`) was unaffected, limiting the exposure to code that
imports `secp256k1/ecdsa.hpp` or `secp256k1/schnorr.hpp` directly.

**Fix**: Removed the `#ifdef _MSC_VER` guard. All platforms now use
`ct::generator_mul_blinded` (when context blinding is active) or
`ct::generator_mul` for nonce-to-point and secret-key-to-pubkey multiplications,
and `ct::scalar_inverse` for the nonce inverse in ECDSA. The compiler
interaction that caused the original performance regression was resolved by
adjusting inlining hints in `cpu/include/secp256k1/ct/select.hpp`. The
resulting performance on GCC/Clang is within acceptable bounds (no measurable
regression versus the prior non-CT path in the common case, because the
blinding branch is unpredictable to the compiler).

Commit date: 2026-04-27. The commit is in the `AUDIT_CHANGELOG.md` entry
for `2026-04-27d`.

**Regression test**: CT pipeline tests (ct-verif analysis, Valgrind
`--tool=memcheck` for uninitialized read on secret paths, dudect timing
uniformity test) now cover all signing paths on both GCC and Clang, with
CI matrix entries for both compilers.

**Classification**: CT correctness failure. High impact for public C++ API
users. The C ABI was not affected.

---

### 2026-04-27 — C++20 requirement leaked to consumers via PUBLIC cxx_std_20

**Discovery**: Review of `cpu/CMakeLists.txt` for Bitcoin Core compatibility
found that the line:

```cmake
target_compile_features(fastsecp256k1 PUBLIC cxx_std_20)
```

uses `PUBLIC` visibility. In CMake, `PUBLIC` means the compile feature is
propagated to any target that links against `fastsecp256k1`. Any CMake-based
consumer that does `target_link_libraries(my_target fastsecp256k1)` will
inherit the `cxx_std_20` requirement and be forced to compile with C++20.

Bitcoin Core targets C++17. A library that forces C++20 on its consumers
cannot be used by Bitcoin Core as a drop-in backend.

Additionally, `secp256k1_shim/CMakeLists.txt` had the same issue:
`target_compile_features(secp256k1_shim PUBLIC cxx_std_20)`.

**Impact**: LOW for current deployments (the library compiles and runs
correctly). High severity as a blocker for Bitcoin Core integration, which
is an explicit project goal. Any project targeting C++17 that links against
the library via CMake would have its C++ standard silently upgraded, possibly
breaking compilation of code that uses C++17-incompatible C++20 constructs
in the library headers.

The `PUBLIC` propagation is also incorrect on principle: the C++20 requirement
is an implementation detail of the library, not an interface requirement. The
public headers are designed to be consumable from C++17. The C++20 requirement
belongs in `PRIVATE`.

**Fix**: Changed to `PRIVATE cxx_std_20` in both `cpu/CMakeLists.txt` and
`secp256k1_shim/CMakeLists.txt`. Commit `613fd2fe`.

No consumer API changes. The library's public headers remain C++17-compatible.
The library's internal implementation continues to use C++20 features, but
these are not exposed through the public interface.

**Regression test**: A CMake consumer test was added that links against
`fastsecp256k1` from a project configured with `cxx_std_17` and verifies that
the link succeeds and the resulting consumer binary does not inherit a C++20
requirement. This runs in CI as part of the compatibility test matrix.

**Classification**: Build system / compatibility. No runtime security impact.
Blocker for Bitcoin Core backend integration.

---

## Timeline Summary

| Date | ID | Severity | Type | Status |
|------|----|----------|------|--------|
| 2026-04-03 | RR-004 ECDSA r-comparison | Medium | Correctness / interoperability | CLOSED |
| 2026-04-27 | Shim parser parity gap | Medium | API contract / parity | CLOSED |
| 2026-04-27 | CT default cleanup (non-CT on GCC/Clang) | High | CT correctness | CLOSED |
| 2026-04-27 | C++20 PUBLIC propagation | Low | Build system / compatibility | CLOSED |

---

## Regression Coverage

All closed findings have permanent regression tests:

| Finding | Regression mechanism |
|---------|---------------------|
| RR-004 ECDSA r-comparison | Wycheproof tcId 346 + `audit/test_exploit_stark_bank_ecdsa_r.cpp` |
| Shim parser parity gap | `scripts/check_libsecp_shim_parity.py` (permanent CI gate) |
| CT default cleanup | CT pipeline (ct-verif, Valgrind, dudect) covering all signing paths, both GCC and Clang |
| C++20 PUBLIC propagation | CMake consumer compatibility test (C++17 consumer link test) |

---

## Future Findings

New findings will be added to this document in chronological order at the
time of disclosure or fix, not deferred to a batch update. The intent is that
this document always reflects the actual security history of the repository
at the time of the latest commit.

If a finding is discovered but not yet fixed, it will appear with status
`OPEN` and a date of discovery. Findings are not withheld pending fix.
