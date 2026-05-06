# COMPLIANCE_STANCE.md — UltrafastSecp256k1

> Version: 1.0 — 2026-04-21
> Closes CAAS gap **G-6**.
>
> This document states what compliance frameworks UltrafastSecp256k1
> claims and does not claim, with verifiable rationale. It exists so
> independent reviewers do not have to reverse-engineer our position.

## 1. Position summary

UltrafastSecp256k1 is **not** a certified cryptographic module under
any government scheme.

Specifically the project does **not** claim:

| Framework | Claim status |
|-----------|--------------|
| FIPS 140-3 (cryptographic module validation) | **No claim** |
| FIPS 186-5 (digital signatures) | **No claim** (we are not in the validated module list) |
| NIST SP 800-90A/B/C (DRBG / entropy) | **No claim** (we use OS RNG, not a userland DRBG) |
| Common Criteria (any EAL) | **No claim** |
| ISO/IEC 19790 | **No claim** |
| BSI AIS-31 entropy | **No claim** |
| ANSSI CSPN | **No claim** |
| CNSA 1.0 / 2.0 algorithm policy | **N/A** — secp256k1 is not on CNSA |
| FIPS 140-3 OE: GPU offload | **No claim** — by design, GPU is public-data only |

We deliberately avoid these claims because:

1. None of them have been pursued through a NIST-recognised CSTL.
2. Making the claim without certification is non-compliant *with the
   compliance framework itself*.
3. The library's value proposition is auditability, not certification.

## 2. What we DO claim — and how to verify each

| Claim | Evidence |
|-------|----------|
| BIP-340 conformant Schnorr | `audit/test_bip340_vectors.cpp` + Wycheproof `wycheproof_secp256k1_*.json` |
| RFC 6979 deterministic ECDSA | `audit/test_rfc6979_*` + Cryptol property `RFC6979.cry` |
| BIP-32 HD derivation matches reference | `audit/test_bip32_*` + reference vectors |
| BIP-324 wire conformance | `audit/test_bip324_*` + interop vectors |
| BIP-352 silent payments conformance | `audit/test_bip352_*` |
| MuSig2 (BIP-327) conformance | `audit/test_musig2_*` + spec vectors |
| FROST (RFC 9591) conformance | `audit/test_frost_*` |
| Constant-time on tested platforms | `gpu_ct_leakage_report.json`, dudect, Valgrind CT, ct-verif (3-tool) |
| Strict parser behaviour for ECDSA, DER, BIP-340, point encodings | 253 exploit PoCs tests; full Wycheproof |
| Reproducible build | SLSA Level 3 provenance + `ci/check_reproducibility.sh` |
| 1.3M+ nightly differential checks vs libsecp256k1 + libtomcrypt | `differential` audit section |

## 3. Algorithm scope

| Algorithm | Status |
|-----------|--------|
| ECDSA over secp256k1 (sign / verify / recover) | Implemented + audited |
| Schnorr (BIP-340) | Implemented + audited |
| ECDH (X9.63 + libsecp256k1 hashed) | Implemented + audited |
| MuSig2 (BIP-327) | Implemented + audited |
| FROST (RFC 9591) | Implemented + audited |
| BIP-32 HD | Implemented + audited |
| BIP-324 v2 transport | Implemented + audited |
| BIP-352 silent payments | Implemented + audited |
| Taproot tweaks (BIP-341) | Implemented + audited |
| Other curves (P-256, ed25519, etc.) | **Not in scope** |
| Symmetric crypto (AES, ChaCha) | Only ChaCha20-Poly1305 inside BIP-324; no standalone API |
| Hash standalone API | SHA-256 / SHA-512 / RIPEMD-160 used internally; not a general hash library |
| Post-quantum | **Not in scope** — secp256k1 is classical only |

## 4. Boundary statement (would be the FIPS module boundary if we were certified)

For the avoidance of any future ambiguity, *if* we ever pursued
certification, the conceptual cryptographic boundary is:

```
include/ufsecp/*.h            (public ABI)
src/cpu/src/ct_*                  (constant-time secret-bearing core)
src/cpu/src/{field,scalar,group}* (math primitives)
src/cpu/src/random.cpp            (OS RNG bridge)
```

GPU code is **outside** this boundary by construction.

## 5. SBOM and supply chain stance

Independent of any compliance scheme, UltrafastSecp256k1 publishes:

| Artefact | Where |
|----------|-------|
| CycloneDX SBOM | CI artefact `sbom-cyclonedx.json` per release |
| SPDX SBOM | CI artefact `sbom-spdx.json` per release |
| SLSA Level 3 provenance | GitHub attestations API |
| Cosign-signed release artifacts | `.sig` next to each release file |
| Reproducible build attestation | `ci/check_reproducibility.sh` log |

These satisfy NIST SSDF / EO-14028-style supply-chain requirements
even though we make no FIPS 140-3 claim.

## 6. Change discipline

If the compliance posture ever changes (e.g. a CMVP submission is
filed), this document and `docs/AUDIT_MANIFEST.md` must be updated in
the same commit, and a sub-gate in `ci/audit_gate.py` must
verify the claim against an evidence file.
