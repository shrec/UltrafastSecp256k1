# THREAT_MODEL.md — UltrafastSecp256k1

> Version: 1.0 — 2026-04-21
> Closes CAAS gap **G-1** in `docs/CAAS_GAP_CLOSURE_ROADMAP.md`.
>
> This document defines what UltrafastSecp256k1 defends against, what it
> does not, and where each defence is implemented. It is the
> single-source published threat model. Every public ABI surface must
> appear in §4 with a STRIDE row.

## 1. Assets

| ID | Asset | Sensitivity |
|----|-------|-------------|
| A1 | `secret_key` (32 bytes, scalar in [1, n−1]) | Critical — full key recovery on leak |
| A2 | `nonce_k` (per-signature, transient) | Critical — reveals A1 if leaked |
| A3 | `tweak`/`chain_code` (BIP-32) | High — derivation tree compromise |
| A4 | `frost_share`/`musig2_share` | High — partial signing compromise |
| A5 | `bip324_session_key` | Medium — peer-session traffic compromise |
| A6 | Public key, signature, address | Public — integrity matters, confidentiality does not |
| A7 | Library state (precompute tables, context) | Low — recoverable from public params |

## 2. Trust boundaries

```
   ┌─────────────────────────────────────────────────────────────┐
   │             Untrusted: network / peer / OS RNG quality      │
   ├─────────────────────────────────────────────────────────────┤
   │   Untrusted user input (caller-provided bytes)              │
   │            ▼                                                │
   │   ┌──────────────────────────────────────────┐              │
   │   │ ABI boundary  ufsecp_*                   │  ← Trust 1   │
   │   │ Strict parsers, length checks, NULL guards│              │
   │   └──────────────────────────────────────────┘              │
   │            ▼                                                │
   │   ┌──────────────────────────────────────────┐              │
   │   │ CT layer  src/cpu/src/ct_*                   │  ← Trust 2   │
   │   │ Constant-time + secret-erase guarantees  │              │
   │   └──────────────────────────────────────────┘              │
   │            ▼                                                │
   │   ┌──────────────────────────────────────────┐              │
   │   │ Math primitives (field, scalar, group)   │  ← Trust 3   │
   │   │ Spec-conformant, formally specified      │              │
   │   └──────────────────────────────────────────┘              │
   │            ▼                                                │
   │   ┌──────────────────────────────────────────┐              │
   │   │ GPU offload (CUDA / OpenCL / Metal)      │  ← Trust 4   │
   │   │ Public-data only by design               │              │
   │   └──────────────────────────────────────────┘              │
   └─────────────────────────────────────────────────────────────┘
```

## 3. Attacker models

| Code | Attacker | Capability | In scope |
|------|----------|-----------|----------|
| AM-1 | Hostile caller | Any input bytes to any public ABI; may invoke in any order | Yes |
| AM-2 | Network attacker | Modify/replay/forge wire messages (signatures, BIP-324, MuSig2 nonces, FROST shares) | Yes |
| AM-3 | Hostile peer | Byzantine MuSig2/FROST counterparty; rogue-key, nonce forge, share withholding | Yes |
| AM-4 | OS-side timing observer | Concurrent process measuring cache/branch timing of CT-layer operations | Yes (CT layer) |
| AM-5 | Microarchitectural attacker | Spectre/Meltdown class, port-contention, frequency scaling | Partially — best-effort hardening; see G-3 |
| AM-6 | Hardware side-channel attacker | Power, EM, fault injection on physical device | **Out** — see [HARDWARE_SIDE_CHANNEL_METHODOLOGY.md](HARDWARE_SIDE_CHANNEL_METHODOLOGY.md) |
| AM-7 | Compromised RNG | OS RNG returns biased / replayed / zero output | Partially — RFC 6979 sign path immune; ECDH/Schnorr nonce paths see [RNG_ENTROPY_ATTESTATION.md](RNG_ENTROPY_ATTESTATION.md) |
| AM-8 | Compromised toolchain | Backdoored compiler, malicious dependency | Partially — supply-chain gate (P15), reproducible build, SLSA |
| AM-9 | Memory disclosure | Heartbleed-class read-out-of-bounds | Yes — secret_erase, valgrind, ASan, MSan |
| AM-10 | Quantum attacker (future) | Shor's algorithm against ECDLP | **Out** — secp256k1 is not post-quantum |

## 4. STRIDE per ABI surface

Every exported `ufsecp_*` ABI function must appear in one of the rows
below. CI gate `audit_gate.py --threat-model` enforces this (see G-1
acceptance criterion).

| ABI surface | Spoofing | Tampering | Repudiation | Info-disclosure | DoS | Elevation |
|-------------|----------|-----------|-------------|-----------------|-----|-----------|
| `ufsecp_keygen_*` | n/a — no identity | RFC 6979/random nonce path; bias resistant | n/a | A1/A2 erased on scope exit (CT layer) | Bounded work | n/a |
| `ufsecp_ecdsa_sign*` | n/a | Deterministic per RFC 6979; explicit `extra_data` | n/a | CT layer; secret_erase; nonce never leaves stack | Bounded | n/a |
| `ufsecp_ecdsa_verify*` | Permissive low-S by design (caller may enforce via `is_low_s()`) | Strict DER + raw parsers; reject non-canonical | n/a | Public data | Constant-time enough; bounded | n/a |
| `ufsecp_schnorr_sign*` | BIP-340 conformant | RFC 6979 nonce + extra_data | n/a | CT layer | Bounded | n/a |
| `ufsecp_schnorr_verify*` | BIP-340 batch + single | Strict 64-byte parser | n/a | Public | Batch rejects on first fail | n/a |
| `ufsecp_ecdh*` | n/a | Strict pubkey parse + on-curve check | n/a | CT layer; output hashed (no raw point bytes) | Bounded | n/a |
| `ufsecp_ec_recover` | Reveals signer pubkey by design | Strict r/s parse + recovery_id range | n/a | Public | Bounded | n/a |
| `ufsecp_musig2_*` | Rogue-key resistant via key-agg coefficient | Round-state machine; rejects out-of-order | n/a | CT layer for nonces/partials | Bounded | n/a |
| `ufsecp_frost_*` | DKG identifiable abort; Byzantine-aware | Round-state machine; commitment check | Identifiable abort surfaces participant ID | CT layer | Bounded | n/a |
| `ufsecp_bip32_*` | Hardened path enforcement | Strict 4-byte version magic; depth limit | n/a | CT layer for chaincode/key | Path length capped | n/a |
| `ufsecp_bip324_*` | AEAD-authenticated | ChaCha20-Poly1305 over framed messages | n/a | Session keys CT-erased; rekey at counter rollover | Counter rollover blocks | n/a |
| `ufsecp_bip352_*` (silent payments) | Scan key cannot forge spend | Strict point validation | n/a | Scan key CT-erased | Bounded | n/a |
| `ufsecp_taproot_*` | Tapleaf commitment binding | Strict tree depth + version | n/a | Public | Depth capped | n/a |
| `ufsecp_address_*` | Network byte enforced | Strict bech32m / base58check | n/a | Public | Bounded | n/a |
| `ufsecp_pubkey_parse/serialize` | n/a | On-curve + non-identity check | n/a | Public | Bounded | n/a |
| `ufsecp_random_*` | n/a | OS RNG attestation per [RNG_ENTROPY_ATTESTATION.md](RNG_ENTROPY_ATTESTATION.md) | n/a | n/a — produces public output | Syscall failure FAILs closed | n/a |
| `ufsecp_gpu_*` | Public data only by design | Host-validated inputs; no secrets crossed | n/a | Not used for secrets | Device-failure surfaced | n/a |
| `ufsecp_context_*` | n/a | Lifecycle state machine | n/a | Erased on `ufsecp_context_destroy` | Re-init bounded | n/a |

Coverage of every `UFSECP_API` export is verified by
`ci/audit_gate.py --threat-model`.

## 5. Residual risks

Each residual is also tracked in `docs/RESIDUAL_RISK_REGISTER.md` (RR-NNN
IDs) so the audit pipeline knows about them and tests / docs cannot
silently regress them.

| ID | Residual | Why accepted | Mitigation in place |
|----|----------|-------------|---------------------|
| RR-001 | Spectre v2 BTI on extreme adversary co-location | Mitigation belongs to OS / CPU vendor | `compute-sanitizer.yml`; CT layer avoids secret-dependent branches |
| RR-002 | Microarchitectural frequency scaling (Hertzbleed) | Power-state observable is OS/firmware territory | CT layer constant-time at instruction granularity; documented in `RESIDUAL_RISK_REGISTER.md` |
| RR-003 | AMD ROCm/HIP backend | No hardware-backed evidence; deferred | Smoke workflow scaffolded (H-11); no public claim made |
| RR-004 | Stark Bank ECDSA `r∈[n,p−1]` class | Closed 2026-04-03 (`ea8cfb3c`) | Wycheproof tcId 346 regression test; permanent |
| RR-005 | GPU `schnorr_snark_witness_batch` performance | Native kernel pending; correctness is fine | Host-side CPU fallback; perf-only residual; tracked |
| RR-006 | Hardware side-channel (power/EM/fault) | Hardware operating-environment territory | Documented in `HARDWARE_SIDE_CHANNEL_METHODOLOGY.md` |
| RR-007 | Quantum (Shor) | secp256k1 is classical-only | Documented; library does not claim PQ |

## 6. What this model does NOT defend against

- Stolen private keys outside the library boundary (HSM compromise,
  filesystem leak of seed, social engineering of recovery phrase).
- Wrong-curve confusion at the application layer (caller using a
  non-secp256k1 key with a secp256k1 ABI). The ABI rejects malformed
  bytes; it cannot detect a semantically wrong key.
- Application-layer replay of valid signed messages. The library signs
  what it is told to sign; semantic deduplication is the caller's job.
- Sybil attacks against MuSig2/FROST quorum policies. Threshold
  selection is the application's responsibility.
- RNG quality below OS attestation. If the OS RNG is broken, every
  random-nonce path inherits the breakage; deterministic RFC 6979
  signing remains safe (see G-2).

## 7. Verification

| Property | How verified | Where |
|----------|-------------|-------|
| Every ABI export has a STRIDE row | `audit_gate.py --threat-model` | CI Stage 2 |
| Every residual has an `RR-NNN` entry | `audit_gate.py --residual-risk-register` | CI Stage 2 |
| Each STRIDE control points to a real test | `ci/exploit_traceability_join.py` | G-9b |
| AM-* attacker models exercised by PoCs | `EXPLOIT_TEST_CATALOG.md` (189 PoCs) | static |

## 8. Change discipline

Adding or modifying an ABI surface requires:

1. New row in §4 STRIDE table in the same commit.
2. Linked test (audit module or exploit PoC) in
   `EXPLOIT_TEST_CATALOG.md`.
3. If a new attacker capability becomes relevant, add an `AM-N` row
   in §3.
4. If a new residual is created, add an `RR-NNN` row in §5 and in
   `docs/RESIDUAL_RISK_REGISTER.md`.

This is enforced by the doc-pairing check (P10) and the threat-model
gate.
