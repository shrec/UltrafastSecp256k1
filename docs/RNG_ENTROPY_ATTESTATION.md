# RNG_ENTROPY_ATTESTATION.md — UltrafastSecp256k1

> Version: 1.0 — 2026-04-21
> Closes CAAS gap **G-2**.
>
> This document defines how UltrafastSecp256k1 obtains randomness, how
> it attests OS RNG quality, and what guarantees hold when the OS RNG
> is broken.

## 1. Where the library uses randomness

| Use | Source | Failure mode if RNG biased |
|-----|--------|---------------------------|
| `ufsecp_keygen_*` (random secret key) | OS RNG → mapped into [1, n−1] | Predictable / reused secret key |
| `ufsecp_ecdsa_sign*` default | **RFC 6979 deterministic** (no RNG) | None (RNG-independent) |
| `ufsecp_ecdsa_sign*` with `extra_data` | RFC 6979 + extra_data; extra_data may be RNG-derived | Reduces nonce-bias resistance to the RFC-6979 baseline |
| `ufsecp_schnorr_sign*` (BIP-340) | BIP-340 deterministic + optional `aux_rand` | Without aux_rand: deterministic, safe; with bad aux_rand: still safe per BIP-340 |
| `ufsecp_musig2_*` nonce gen | OS RNG (mandatory by MuSig2 spec) | Catastrophic — exposes share |
| `ufsecp_frost_*` round-1 | OS RNG | Catastrophic |
| `ufsecp_bip324_*` ephemeral key | OS RNG | Session compromise |
| `ufsecp_random_*` (caller-facing) | OS RNG | Caller-visible |

**Key invariant:** the default ECDSA / Schnorr signing paths are
**RNG-independent**. A broken OS RNG cannot leak the secret key via
those paths.

## 2. OS RNG sources used

| Platform | Primary | Fallback | Failure handling |
|----------|---------|----------|-----------------|
| Linux | `getrandom(GRND_NONBLOCK)` | `/dev/urandom` | Both fail → return `UFSECP_ERROR_RNG`, never zero-fill |
| macOS | `getentropy()` | `SecRandomCopyBytes` | Same fail-closed |
| Windows | `BCryptGenRandom(BCRYPT_USE_SYSTEM_PREFERRED_RNG)` | none | Same |
| iOS | `SecRandomCopyBytes` | `getentropy` | Same |
| Android | `getrandom` | `/dev/urandom` | Same |
| ESP32 | `esp_fill_random` (HW TRNG) | none | Same |
| FreeBSD/OpenBSD | `getentropy` | `/dev/urandom` | Same |
| RISC-V (Linux) | `getrandom` | `/dev/urandom` | Same |

Implemented in `src/cpu/src/random.cpp`. Tested by:
`audit/test_random_oracle.cpp`, `audit/test_oracle_separation.cpp`,
exploit PoCs `test_exploit_rng_*` series.

## 3. Attestation

At every random draw the library performs:

1. **Source attestation** — record which syscall returned bytes
   (compile-time + runtime).
2. **Length attestation** — caller's requested length matches kernel's
   returned length, byte-for-byte.
3. **Repeat-byte sanity** — reject all-zero / all-FF outputs as a
   structural sanity (not a quality test).
4. **Health-test-on-demand** — `ufsecp_random_self_test()` runs the
   NIST SP 800-90B repetition-count and adaptive-proportion tests on
   a 1 MiB draw on caller request. **This is a self-test, not a
   continuous test** — the library does not silently sample the RNG.

The library deliberately does **not** implement a userland DRBG on top
of the OS RNG. Doing so would (a) reduce auditability (more state to
zeroize), (b) hide kernel reseeding, (c) duplicate attestation at the
wrong layer.

## 4. Failure modes — fail-closed

If the OS RNG syscall fails for any reason, the library **must** return
`UFSECP_ERROR_RNG`. It must never:

- Return zero bytes.
- Return previously-seen bytes.
- Fall back to a userland PRNG (e.g. `rand()`, libc PRNG, `time()`).
- Return success with a partial-fill.

This is enforced by `audit/test_rng_fail_closed.cpp` (currently wired
in `unified_audit_runner.cpp` under `memory_safety`).

## 5. Test coverage

| Property | Test |
|----------|------|
| Linux `getrandom` failure surfaces `UFSECP_ERROR_RNG` | `audit/test_rng_fail_closed.cpp` |
| All-zero RNG output rejected | `audit/test_random_oracle.cpp` |
| RFC 6979 sign path deterministic regardless of RNG | `audit/test_rfc6979_*` |
| MuSig2/FROST refuse to proceed if RNG draw fails | `audit/test_exploit_musig2_*`, `test_exploit_frost_*` |
| Cross-platform RNG syscall present | `src/cpu/tests/test_random_platform.cpp` |
| Self-test SP 800-90B repetition | `audit/test_random_health.cpp` (advisory) |

## 6. What this attestation is NOT

- It is **not** a FIPS 140-3 entropy source claim. See
  [COMPLIANCE_STANCE.md](COMPLIANCE_STANCE.md).
- It is **not** a guarantee that the OS RNG is good. If your kernel is
  backdoored, your ECDH ephemeral keys are too. We can only ensure we
  do not make it worse.
- It is **not** a continuous health test. SP 800-90B continuous tests
  belong in the kernel's RNG, not in userland.

## 7. Change discipline

Adding a new randomness consumer in the library requires:

1. Add a row to §1.
2. Add a test row to §5.
3. If the consumer is in a CT-sensitive path, also document it in
   `docs/AUDIT_MANIFEST.md` under P3 (constant-time).
