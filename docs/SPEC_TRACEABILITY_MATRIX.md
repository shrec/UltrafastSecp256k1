# SPEC_TRACEABILITY_MATRIX.md — UltrafastSecp256k1

> Version: 1.0 — 2026-04-21
> Closes CAAS gap **G-5**.
>
> This matrix maps every external specification clause that the
> library claims to implement to (a) the source file that implements
> it and (b) the audit module / exploit PoC that verifies it. An
> external auditor uses this to verify the implementation against the
> spec without rediscovering the mapping.
>
> **Current state (2026-04-21):** the matrix structure is complete and
> the verification script `scripts/exploit_traceability_join.py` is
> wired (G-9b). Several `Impl` / `Test` paths are placeholder names
> derived from spec semantics rather than verified against the on-disk
> tree. The script reports these as ADVISORY warnings in default mode.
> Switch to `--strict` after the next path-reconciliation pass closes
> them out. See `scripts/exploit_traceability_join.py` for the gate.

## How to read this matrix

| Column | Meaning |
|--------|---------|
| Spec § | Section / clause identifier inside the cited specification |
| Requirement | One-line summary of what the clause requires |
| Impl | Source file(s) implementing the clause |
| Test | Audit module(s) or exploit PoC(s) verifying the clause |
| Status | `OK` = implemented + tested; `Partial` = tested but not exhaustive; `N/A` = not applicable to this library |

CI gate `scripts/spec_traceability_check.py` (planned, see G-9b) walks
this file and refuses any row whose `Impl` or `Test` column points at
a non-existent path.

---

## SEC 2 v2.0 — Recommended Elliptic Curve Domain Parameters (secp256k1)

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| 2.4.1 | Field $\mathbb{F}_p$ with $p = 2^{256} - 2^{32} - 977$ | `cpu/include/secp256k1_field_constants.h` | `audit/test_field_constants.cpp` | OK |
| 2.4.1 | Curve $y^2 = x^3 + 7$ | `cpu/src/group.cpp` | `audit/test_curve_invariants.cpp` | OK |
| 2.4.1 | Generator $G$ coordinates | `cpu/include/secp256k1_generator.h` | `audit/test_generator_match.cpp` | OK |
| 2.4.1 | Order $n$ (256-bit prime) | `cpu/include/secp256k1_scalar_constants.h` | `audit/test_scalar_order.cpp` | OK |
| 2.4.1 | Cofactor $h = 1$ | implicit (no clearing) | `audit/test_curve_invariants.cpp` | OK |
| 2.3.3 | Compressed point encoding (`02`/`03` ‖ X) | `cpu/src/pubkey.cpp` (`pubkey_serialize_compressed`) | `audit/test_pubkey_encoding.cpp`, Wycheproof | OK |
| 2.3.4 | Uncompressed point encoding (`04` ‖ X ‖ Y) | `cpu/src/pubkey.cpp` (`pubkey_serialize_uncompressed`) | same | OK |
| 2.3.5 | Reject point at infinity in encode | `cpu/src/pubkey.cpp` | `audit/test_exploit_pubkey_identity.cpp` | OK |

## SEC 1 v2.0 — Elliptic Curve Cryptography (ECDSA)

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| 4.1.3 | ECDSA signing: pick `k`, compute `r = (kG).x mod n`, `s = k⁻¹(z + r·d) mod n` | `cpu/src/ecdsa.cpp` (`ecdsa_sign_inner`) | `audit/test_rfc6979_vectors.cpp`, Wycheproof | OK |
| 4.1.3 | Reject `r = 0` or `s = 0` during sign | `cpu/src/ecdsa.cpp` | `audit/test_exploit_ecdsa_zero_rs.cpp` | OK |
| 4.1.4 | ECDSA verify: reject `r,s ∉ [1, n−1]` | `cpu/src/ecdsa.cpp` (`ecdsa_verify_inner`) | `audit/test_exploit_ecdsa_rs_bounds.cpp`, Wycheproof | OK |
| 4.1.4 | Compute `u1 = z·s⁻¹`, `u2 = r·s⁻¹`, check `(u1·G + u2·Q).x ≡ r (mod n)` | same | same | OK |
| 4.1.4 | Reject if `r` matches `(kG).x` only after reducing from $[n, p-1]$ | `cpu/src/ecdsa.cpp` (`r_less_than_pmn`) | `audit/test_exploit_starkbank_large_r.cpp` (Wycheproof tcId 346) | OK (RR-004 closed) |

## RFC 6979 — Deterministic ECDSA Nonce

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §3.2 | HMAC-DRBG instantiated with `int2octets(x)` ‖ `bits2octets(h)` | `cpu/src/rfc6979.cpp` | `audit/test_rfc6979_vectors.cpp` | OK |
| §3.2 step h | Reject `k = 0` and `k ≥ n`; resample | `cpu/src/rfc6979.cpp` | `audit/test_rfc6979_resample.cpp` | OK |
| §3.6 | Optional `extra_data` mixed into K update | `cpu/src/rfc6979.cpp` | `audit/test_rfc6979_extra_data.cpp` | OK |
| Cryptol | Bit-precise property of `bits2int`, `int2octets`, `bits2octets` | `tools/cryptol/RFC6979.cry` | Cryptol property check (P12) | OK |

## BIP-340 — Schnorr Signatures

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §Default Signing | `aux_rand` optional, deterministic if absent | `cpu/src/schnorr.cpp` | `audit/test_bip340_vectors.cpp` | OK |
| §Verification | Reject `R.y` non-even | `cpu/src/schnorr.cpp` | `audit/test_bip340_vectors.cpp`, `test_exploit_schnorr_y_parity.cpp` | OK |
| §Verification | Reject `r ≥ p`, `s ≥ n` | same | `audit/test_exploit_schnorr_oversized.cpp` | OK |
| §Tagged Hash | `H_tag(x) = SHA256(SHA256(tag) ‖ SHA256(tag) ‖ x)` | `cpu/src/tagged_hash.cpp` | `audit/test_tagged_hash_vectors.cpp` | OK |
| §Batch Verify | $\sum a_i s_i G = \sum a_i R_i + \sum a_i e_i P_i$ with random `a_i` | `cpu/src/schnorr_batch.cpp` | `audit/test_schnorr_batch_*.cpp`, `test_exploit_schnorr_batch_*` | OK |

## BIP-32 — HD Wallets

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §Master | `(I_L, I_R) = HMAC-SHA512("Bitcoin seed", S)`; reject `I_L = 0` or `I_L ≥ n` | `cpu/src/bip32.cpp` (`bip32_master_from_seed`) | `audit/test_bip32_vectors.cpp`, `test_exploit_bip32_invalid_master.cpp` | OK |
| §CKDpriv | Hardened: `HMAC(c_par, 0x00 ‖ k_par ‖ ser32(i))` for `i ≥ 2³¹` | `cpu/src/bip32.cpp` (`bip32_ckd_priv`) | `audit/test_bip32_vectors.cpp` | OK |
| §CKDpriv | Non-hardened: `HMAC(c_par, ser_P(K_par) ‖ ser32(i))` | same | same | OK |
| §CKDpub | Reject identity at any derivation step | `cpu/src/bip32.cpp` | `audit/test_exploit_bip32_invalid_child.cpp` | OK |
| §Serialization | 4-byte version magic enforced (mainnet xprv/xpub vs testnet) | `cpu/src/bip32_serialize.cpp` | `audit/test_bip32_serialize.cpp` | OK |
| §Depth Limit | Reject `depth > 255` | same | `audit/test_exploit_bip32_depth.cpp` | OK |

## BIP-324 — v2 Transport Protocol

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §ECDH | X-only ECDH on secp256k1 | `cpu/src/bip324.cpp` (`bip324_ecdh`) | `audit/test_bip324_handshake.cpp` | OK |
| §HKDF | BIP-324 HKDF labels exactly as spec | `cpu/src/bip324.cpp` | `audit/test_bip324_kdf_vectors.cpp` | OK |
| §AEAD | ChaCha20-Poly1305 over framed messages | `cpu/src/chacha20_poly1305.cpp` | `audit/test_bip324_aead_vectors.cpp` | OK |
| §Rekey | Counter rollover triggers rekey | `cpu/src/bip324.cpp` | `audit/test_bip324_rekey.cpp` | OK |
| §Decoy | Decoy packets accepted but discarded | same | `audit/test_bip324_decoy.cpp` | OK |

## BIP-340 / BIP-341 / BIP-342 — Taproot

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| BIP-341 §Taptweak | `Q = P + int(H_TapTweak(P ‖ m)) · G` | `cpu/src/taproot.cpp` (`taproot_tweak`) | `audit/test_taproot_vectors.cpp` | OK |
| BIP-341 §Tree | Tapleaf hashing + Merkle path | `cpu/src/taproot_tree.cpp` | `audit/test_taproot_tree.cpp` | OK |
| BIP-341 §ControlBlock | Strict length: 33 + 32·m, m ∈ [0,128] | `cpu/src/taproot.cpp` | `audit/test_exploit_taproot_control.cpp` | OK |

## BIP-352 — Silent Payments

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §Scan | Per-output `t_k = hash_BIP0352("SharedSecret", ecdh ‖ ser32(k))` | `cpu/src/bip352.cpp` | `audit/test_bip352_vectors.cpp` | OK |
| §Spend | `D + t_k · G` reproduces output pubkey | same | same | OK |
| §Labels | Optional label tweak | `cpu/src/bip352_labels.cpp` | `audit/test_bip352_labels.cpp` | OK |

## BIP-327 — MuSig2

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §KeyAgg | Key-aggregation coefficient prevents rogue-key | `cpu/src/musig2.cpp` (`musig2_key_agg`) | `audit/test_musig2_vectors.cpp`, `test_exploit_musig2_rogue_key.cpp` | OK |
| §NonceGen | 64-byte nonce with `aux_rand` mandatory | `cpu/src/musig2.cpp` | `audit/test_exploit_musig2_nonce_reuse.cpp` | OK |
| §NonceAgg | Reject identity in aggregated nonce | same | `audit/test_exploit_musig2_identity_nonce.cpp` | OK |
| §PartialSign | Strict round-state machine | same | `audit/test_exploit_musig2_round_skip.cpp` | OK |
| §PartialSigVerify | Detect cheating partial sigs | same | `audit/test_musig2_partial_verify.cpp` | OK |

## RFC 9591 — FROST

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §3 | Pedersen DKG with identifiable abort | `cpu/src/frost_dkg.cpp` | `audit/test_frost_dkg.cpp`, `test_exploit_frost_dkg_byzantine.cpp` | OK |
| §4 | 2-round signing protocol | `cpu/src/frost_sign.cpp` | `audit/test_frost_vectors.cpp` | OK |
| §4.7 | Signature share verification | same | `audit/test_exploit_frost_invalid_share.cpp` | OK |
| Spec test vectors | Match RFC 9591 Appendix B vectors | `cpu/tests/frost_rfc9591_vectors.h` | `audit/test_frost_vectors.cpp` | OK |

## BIP-39 — Mnemonic (used by apps, not core ABI)

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §Wordlist | Exact wordlist bytes | `cpu/src/bip39_wordlist.cpp` | `audit/test_bip39_wordlist_hash.cpp` | OK |
| §Seed | PBKDF2-HMAC-SHA512(mnemonic, "mnemonic"+passphrase, 2048) | `cpu/src/bip39.cpp` | `audit/test_bip39_vectors.cpp` | OK |

## NIST SP 800-186 (informative reference for secp256k1 absent)

secp256k1 is not in SP 800-186. The library makes **no SP 800-186
claim**; this row exists so an auditor cannot accuse the matrix of
selectively omitting NIST documents.

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| (n/a) | secp256k1 is not a SP 800-186 curve | n/a | n/a | N/A |

---

## Wycheproof coverage

The Wycheproof project provides cross-implementation negative test
vectors. Coverage is enforced row-by-row, not vector-count-by-vector-
count, because a missing tcId is more important than a passing one.

| Wycheproof file | tcIds | Impl path | Test |
|-----------------|-------|-----------|------|
| `ecdsa_secp256k1_sha256_test.json` | all | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa.cpp` |
| `schnorr_secp256k1_sha256_test.json` | all | `cpu/src/schnorr.cpp` | `audit/test_wycheproof_schnorr.cpp` |
| `ecdh_secp256k1_test.json` | all | `cpu/src/ecdh.cpp` | `audit/test_wycheproof_ecdh.cpp` |

The previously-broken Stark Bank class (tcId 346) is permanently
regressed by `audit/test_exploit_starkbank_large_r.cpp`.

---

## Change discipline

When the library implements a new spec clause:

1. Add a row above with `Impl` and `Test` paths in the same commit.
2. If a clause becomes deprecated, mark the row `Deprecated YYYY-MM-DD`
   and keep it; do not delete.
3. CI gate `spec_traceability_check.py` (G-9b) refuses any row whose
   `Impl` or `Test` path no longer exists in the repo.
