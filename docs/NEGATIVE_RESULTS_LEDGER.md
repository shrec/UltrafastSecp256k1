# Negative Results Ledger

**UltrafastSecp256k1 — Approaches Evaluated and Rejected or Deferred**

Version: 1.0
Date: 2026-04-27
Status: Active

---

## Purpose

This document records design decisions where an approach was considered,
prototyped, or seriously evaluated and then rejected, deferred, or found to
be inapplicable. These outcomes are as important to document as positive
decisions, for two reasons:

1. They prevent the same evaluation from being done twice when a new
   contributor or auditor asks "why didn't you do X?"
2. They surface the reasoning for decisions that may look wrong from the
   outside without context.

Each entry records what was tried, what happened, and why the outcome was
what it was. Where an approach was deferred rather than rejected, the entry
explains what would change the decision.

---

## Index

| ID | Topic | Outcome | Reference |
|----|-------|---------|-----------|
| NR-001 | Side-channel resistant field arithmetic via masking | Rejected | Section below |
| NR-002 | Dedicated GPU constant-time signing path | Deferred | Section below |
| NR-003 | MSVC default signing path (non-CT) | Fixed (was a bug) | Section below |
| NR-004 | Scalar::from_bytes without range check at shim boundary | Fixed (was a bug) | Section below |
| NR-005 | Lattice attack resistance via nonce stretching | Not adopted | Section below |
| NR-006 | Per-call blinding (new blinding scalar per signature) | Not adopted | Section below |
| NR-007 | ROCm/HIP GPU backend for AMD hardware CT | Deferred | Section below |
| NR-008 | MuSig2 key aggregation via GPU | Not adopted for signing | Section below |
| NR-009 | Compile-time-only secp256k1 order check for shim scalars | Not useful | Section below |

---

## NR-001 — Side-Channel Resistant Field Arithmetic via Masking

**What was evaluated**: Randomize field element representations by XOR-masking
limbs with a random value before every field operation, unmasking after.
This is sometimes called "arithmetic masking" or "first-order masking" and is
used in some smartcard implementations to defeat DPA attacks on field
multiplications.

**What happened**: Prototyped and benchmarked. The overhead was approximately
30% on the field multiplication inner loop, which is the dominant cost in all
scalar multiplication paths.

**Why it was rejected**: The threat model for field masking is first-order
power analysis, which requires physical access to measure power consumption.
For software implementations on general-purpose CPUs, the relevant side
channels are timing (via cache behavior and branch prediction) and memory
access patterns. For these threats, the Hamburg scalar blinding transform and
the additive blinding on the nonce scalar (implemented via
`ufsecp_context_randomize`) are stronger and more targeted mitigations.

Arithmetic masking adds overhead without addressing the actual threat class.
It also introduces implementation complexity (the masking/unmasking must be
correct at every operation boundary) that is itself a source of bugs.

The stronger mitigations are: the CT scalar inverse (`ct::scalar_inverse`),
the Hamburg comb for constant-time generator multiplication
(`ct::generator_mul`), and additive blinding (`ct::generator_mul_blinded`).
These address the timing and memory-access channels directly. See
`THREAT_MODEL.md` Section 2 (CT Layer) for the full treatment.

**What would change this**: Physical implementation (embedded hardware signer)
where power analysis is a realistic attack vector. In that context, field
masking would be reconsidered as a complementary control.

---

## NR-002 — Dedicated GPU Constant-Time Signing Path

**What was evaluated**: Implementing a constant-time signing path that runs
on the GPU, to allow GPU-accelerated batch signing without routing secrets
through the CPU.

**What happened**: Design evaluation only; no implementation attempted.

**Why it was deferred**: GPU execution does not provide the isolation
guarantees required for constant-time secret handling. GPU memory is managed
by the driver and the operating system; there is no equivalent of Valgrind
memcheck, AddressSanitizer, or hardware-enforced secret key isolation. GPU
warps execute in lockstep but this is a performance property, not a
constant-time security property — inter-thread timing side channels via warp
divergence are a known GPU microarchitecture issue.

Additionally, GPU kernels run in a shared memory environment. On multi-tenant
systems (cloud GPUs), co-location attacks via GPU memory residue are a known
concern. The isolation assumptions required for a "constant-time" claim cannot
be met on current GPU hardware.

The architecture decision is: GPU handles only public-data operations
(verification, batch verification, BIP-352 scanning, public key derivation).
All secret-key operations run on the CPU CT layer. This is documented
explicitly in `docs/AUDIT_SCOPE.md` as RR-002.

**What would change this**: Hardware-level GPU trusted execution environments
(confidential computing GPUs with key isolation, equivalent to Intel TDX or
AMD SEV for CPU) that provide verified memory isolation. None are currently
available in a form compatible with the library's target platforms.

---

## NR-003 — MSVC Default Signing Path (Non-CT) [BUG, NOW FIXED]

**What was evaluated**: This entry records a bug that was found and fixed,
not an intentional design choice. It appears here because the pattern —
`#ifdef _MSC_VER` routing to a CT path while leaving non-MSVC platforms on
a non-CT path — could plausibly be mistaken for an intentional design.

**What happened**: Initial implementation of the public C++ signing functions
used an `#ifdef _MSC_VER` guard that routed MSVC builds through the CT path
(`ct::generator_mul_blinded`, `ct::scalar_inverse`) while leaving GCC and
Clang builds on the variable-time path (`Point::generator().scalar_mul()`,
`Scalar::inverse()`). The stated reason in the original commit was a
performance regression on non-MSVC compilers for the CT path — the compiler
was not optimizing the constant-time selection as well.

This was incorrect. The correct solution is to fix the compiler interaction
with the CT path, not to leave most production platforms non-CT. The
performance regression was a compiler hint issue, not an inherent cost of
the CT implementation.

**Fix**: Removed the guard. All platforms now use `ct::generator_mul_blinded`
and `ct::scalar_inverse` for all public C++ signing paths. Commit date
2026-04-27. The CT path performance on GCC and Clang was confirmed to be
within acceptable bounds after the compiler hint was corrected.

**Lesson**: `#ifdef` guards that change security properties are always
suspicious, even when the stated reason is performance. They require explicit
documentation and audit attention.

---

## NR-004 — Scalar::from_bytes Without Range Check at Shim Boundary [BUG, NOW FIXED]

**What was evaluated**: Again, this records a bug found and fixed during a
comprehensive shim audit, documented here because it reflects a class of
API design error that may recur.

**What happened**: All scalar-input functions in the libsecp256k1 shim
(`secp256k1_ec_seckey_verify`, `secp256k1_ec_seckey_tweak_add/mul`,
`secp256k1_ec_pubkey_create/tweak_add/tweak_mul`,
`secp256k1_ecdsa_signature_parse_compact`, `secp256k1_keypair_create`,
`secp256k1_xonly_pubkey_parse`, `secp256k1_schnorrsig_sign32`) used
`Scalar::from_bytes` to parse scalar inputs. This function reduces the input
modulo n, silently accepting values in the range [n, 2^256−1] by treating
them as their reduction mod n.

The libsecp256k1 contract is to reject inputs >= n (returning 0/false for
the affected functions). This is the correct behavior because a caller who
passes a value >= n has likely made an error, and silently accepting a
different key than intended is more dangerous than rejecting the input.

For `secp256k1_ecdsa_signature_parse_compact` in particular, the initial
implementation used a blind `memcpy` with no validation at all. An r value
equal to n would be accepted and stored, then later silently reduced to 0
at use time — which would be rejected at sign time but not at parse time,
creating an inconsistency.

**Fix**: All boundary functions switched to `Scalar::parse_bytes_strict_nonzero`
or `Scalar::parse_bytes_strict`, and `FieldElement::parse_bytes_strict`.
These functions return an error if the input >= n (or >= p for field elements)
rather than silently reducing. Commit date 2026-04-27.

**Detection**: `scripts/check_libsecp_shim_parity.py` now runs as a permanent
gate, comparing shim behavior against libsecp256k1 reference output on
boundary inputs.

**Lesson**: Shim functions must match the reference implementation's
rejection behavior, not just its acceptance behavior. A function that accepts
a superset of valid inputs may appear to work correctly on all valid inputs
while silently mishandling invalid ones.

---

## NR-005 — Lattice Attack Resistance via Nonce Stretching

**What was evaluated**: Nonce stretching — adding additional hash rounds or
a hash chain between the RFC 6979 HMAC_DRBG output and the final nonce scalar
— to reduce the per-signature information leakage rate for HNP (Hidden Number
Problem) lattice attacks.

**What happened**: Evaluated against published lattice attack thresholds
(Nguyen-Shparlinski bounds, Brumley-Tuveri, and the 2023 Minerva results).
Nonce stretching does reduce the effective leakage rate by increasing the
number of signatures required for lattice reconstruction, and it does push
attack feasibility below current published thresholds.

**Why it was not adopted**: Two reasons.

First, nonce stretching diverges from RFC 6979. The nonce generation
specification is part of the ECDSA standard. Implementations that deviate
from RFC 6979 cannot claim RFC 6979 compliance, which is a requirement for
some regulatory contexts and is checked by the Wycheproof RFC 6979 test
suite. Adopting nonce stretching would require either forking the standard
or documenting the deviation explicitly, both of which have costs.

Second, the correct mitigation for HNP-based lattice attacks is scalar
blinding on the nonce, not nonce stretching. Scalar blinding (implemented
via `ufsecp_context_randomize`, which installs a random additive blinding
scalar) prevents an attacker from relating the observed signature values to
the nonce scalar, even with perfect timing measurement. Nonce stretching
reduces leakage; blinding eliminates the exploitable correlation. The
stronger mitigation is implemented.

**What would change this**: A regulatory requirement to use RFC 6979 AND
additional leakage reduction, for example in a FIPS 140-3 submission context.
In that case nonce stretching would be reconsidered as an additional layer.

---

## NR-006 — Per-Call Blinding (New Blinding Scalar Per Signature)

**What was evaluated**: Generating a fresh random blinding scalar r for every
individual signing call, rather than installing a thread-local blinding scalar
that persists across multiple calls (the current design via
`ufsecp_context_randomize`).

**What happened**: Prototyped and benchmarked. Per-call blinding requires one
additional scalar generation (SHA-256 call) and one additional point addition
per signature. On the measured hardware, this adds approximately 3 μs per
call, which is approximately 8% overhead on fast signing paths.

**Why it was not adopted**: The threat model for blinding is correlation attacks
across signatures — specifically, DPA and fault injection attacks that exploit
the correlation between the nonce scalar and the observed power trace or fault
response across many signatures from the same key. Per-call blinding eliminates
all inter-call correlation. Thread-local blinding (with periodic refresh via
`ufsecp_context_randomize`) eliminates correlation within the period between
refreshes.

For the target use cases (software signing on general-purpose CPUs, not
embedded hardware signers), the practical threat is timing-based, not
power-based. Timing attacks are addressed by the CT layer, not by blinding.
The residual threat addressed by blinding is: an adversary who can observe
many signatures and measure some side channel correlated with the nonce.
Thread-local blinding with periodic refresh provides adequate protection
against this at approximately zero amortized overhead (the refresh is
infrequent).

Per-call blinding provides marginally stronger protection at meaningful cost.
The tradeoff does not favor per-call blinding for the current deployment
profile.

**What would change this**: Deployment on hardware signers where power traces
are observable (physical attack context). In that case, per-call blinding
would be the correct choice and the 8% overhead would be acceptable given the
threat.

---

## NR-007 — ROCm/HIP GPU Backend for AMD Hardware CT

**What was evaluated**: Implementing a ROCm/HIP backend for AMD GPUs that
provides the same public-data GPU acceleration as the CUDA and OpenCL backends.

**What happened**: Design evaluation only. The OpenCL backend already runs on
AMD hardware (AMD GPUs support OpenCL 3.0). ROCm/HIP would provide a
native-API alternative.

**Why it was deferred**: AMD GPU evidence is explicitly not in scope for the
current audit (see `docs/AUDIT_SCOPE.md` RR-003). Adding a ROCm/HIP backend
before the OpenCL backend's AMD performance is fully characterized and audited
would expand the assurance surface without corresponding audit coverage.

The practical performance of the OpenCL backend on AMD hardware is not well
characterized. A ROCm/HIP native backend might outperform OpenCL on AMD, but
the magnitude of the gain is unknown without measurement.

**What would change this**: A concrete deployment requirement for AMD GPU
acceleration where OpenCL performance is insufficient, combined with willingness
to expand the audit scope to cover the new backend. The existing OpenCL backend
covers the use case adequately in the absence of such a requirement.

---

## NR-008 — MuSig2 Key Aggregation via GPU

**What was evaluated**: Offloading MuSig2 partial signature computation to the
GPU for high-throughput multi-party signing scenarios.

**What happened**: Design evaluation. MuSig2 partial signing involves computing
s_i = k_i + e · x_i (mod n), where x_i is the participant's secret key and
k_i is the participant's nonce secret. Both are secret scalars. The result
is a partial signature that is public, but the computation involves secrets.

**Why it was not adopted for signing**: GPU is public-data-only by
architecture (see NR-002). Partial signing involves secret scalars. This is
not an oversight — it is the same architectural constraint that prevents GPU
CT signing in general. Moving MuSig2 partial signing to the GPU would violate
the GPU security boundary.

GPU can and does accelerate MuSig2 partial signature verification, which is
a public-data operation: it takes the partial signature s_i, the public nonce
R_i, and the public key X_i, and checks that s_i · G == R_i + e · X_i.
This verification is fully public and runs on the GPU verification path.

**What would change this**: Hardware confidential computing GPU with verified
key isolation (see NR-002). Otherwise, this is a permanent architectural
constraint.

---

## NR-009 — Compile-Time-Only Secp256k1 Order Check for Shim Scalars

**What was evaluated**: Adding a compile-time static assertion that verifies
the secp256k1 scalar range at compile time, to avoid the overhead of runtime
range checking on every shim boundary call.

**What happened**: Brief analysis. No prototype was needed.

**Why it was not useful**: The secp256k1 order n is a fixed constant, so
in principle compile-time checking sounds plausible. However, the range check
is not checking a property of the constant n — it is checking whether a
specific runtime input value (the key or scalar passed in by the caller) is
less than n. Runtime input values are not known at compile time. `static_assert`
cannot substitute for `runtime_comparison(input, n)`.

There is no compile-time scalar range check possible for input validation.
The runtime comparison is necessary and not eliminable. `parse_bytes_strict`
uses a constant-time runtime comparison that adds approximately 2 ns per call,
which is not material relative to the cryptographic operations.

**Lesson**: "Compile-time check" is only applicable when the thing being
checked is a compile-time constant. Input validation cannot be compile-time.

---

## Maintenance Note

Entries in this ledger should be updated if:
- New information changes the conclusion (e.g., a new attack demonstrates
  that a rejected mitigation is necessary)
- A deferred item is implemented (move to implemented, add commit reference)
- A fixed bug recurs or is found to have additional instances

The ledger is a living document. Adding an entry does not freeze the decision.
