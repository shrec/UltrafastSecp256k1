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
> **Current state (2026-04-21):** matrix paths reconciled against the
> on-disk tree; every `Impl` / `Test` cell points at a real file. The
> verification script `scripts/exploit_traceability_join.py --strict`
> is wired in CAAS Stage 2 and rejects any row whose path no longer
> exists. See G-9b in `docs/AUDIT_CHANGELOG.md` for the gate.

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
| 2.4.1 | Field $\mathbb{F}_p$ with $p = 2^{256} - 2^{32} - 977$ | `cpu/include/secp256k1/field.hpp`, `cpu/src/field.cpp` | `audit/audit_field.cpp`, `audit/test_exploit_field_arithmetic.cpp` | OK |
| 2.4.1 | Curve $y^2 = x^3 + 7$ | `cpu/src/point.cpp` | `audit/audit_invariants.cpp` | OK |
| 2.4.1 | Generator $G$ coordinates | `cpu/src/precompute.cpp` | `audit/audit_invariants.cpp` | OK |
| 2.4.1 | Order $n$ (256-bit prime) | `cpu/include/secp256k1/scalar.hpp`, `cpu/src/scalar.cpp` | `audit/audit_scalar.cpp`, `audit/test_exploit_scalar_group_order.cpp` | OK |
| 2.4.1 | Cofactor $h = 1$ | implicit (no clearing) | `audit/audit_invariants.cpp` | OK |
| 2.3.3 | Compressed point encoding (`02`/`03` ‖ X) | `cpu/src/point.cpp` | `audit/test_wycheproof_ecdsa.cpp`, `audit/fuzz_pubkey_parse.cpp` | OK |
| 2.3.4 | Uncompressed point encoding (`04` ‖ X ‖ Y) | `cpu/src/point.cpp` | `audit/test_wycheproof_ecdsa.cpp`, `audit/fuzz_pubkey_parse.cpp` | OK |
| 2.3.5 | Reject point at infinity in encode | `cpu/src/point.cpp` | `audit/test_exploit_pubkey_arith.cpp`, `audit/test_infinity_edge_cases.cpp` | OK |

## SEC 1 v2.0 — Elliptic Curve Cryptography (ECDSA)

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| 4.1.3 | ECDSA signing: pick `k`, compute `r = (kG).x mod n`, `s = k⁻¹(z + r·d) mod n` | `cpu/src/ecdsa.cpp`, `cpu/src/ct_sign.cpp` | `audit/test_exploit_ecdsa_rfc6979_kat.cpp`, `audit/test_wycheproof_ecdsa.cpp` | OK |
| 4.1.3 | Reject `r = 0` or `s = 0` during sign | `cpu/src/ecdsa.cpp` | `audit/test_exploit_ecdsa_der_confusion.cpp`, `audit/test_wycheproof_ecdsa.cpp` | OK |
| 4.1.4 | ECDSA verify: reject `r,s ∉ [1, n−1]` | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa.cpp`, `audit/test_wycheproof_ecdsa_extended.cpp` | OK |
| 4.1.4 | Compute `u1 = z·s⁻¹`, `u2 = r·s⁻¹`, check `(u1·G + u2·Q).x ≡ r (mod n)` | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa.cpp`, `audit/test_kat_all_operations.cpp` | OK |
| 4.1.4 | Reject if `r` matches `(kG).x` only after reducing from $[n, p-1]$ | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa_bitcoin.cpp` (Wycheproof tcId 346) | OK (RR-004 closed) |

## RFC 6979 — Deterministic ECDSA Nonce

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §3.2 | HMAC-DRBG instantiated with `int2octets(x)` ‖ `bits2octets(h)` | `cpu/src/ecdsa.cpp` (RFC6979 nonce) | `audit/test_exploit_ecdsa_rfc6979_kat.cpp` | OK |
| §3.2 step h | Reject `k = 0` and `k ≥ n`; resample | `cpu/src/ecdsa.cpp` | `audit/test_exploit_rfc6979_truncation_bias.cpp`, `audit/test_exploit_rfc6979_minerva_amplified.cpp` | OK |
| §3.6 | Optional `extra_data` mixed into K update | `cpu/src/ecdsa.cpp` | `audit/test_exploit_ecdsa_rfc6979_kat.cpp` | OK |
| Cryptol | Bit-precise property of ECDSA over RFC6979 nonce | `formal/cryptol/Secp256k1ECDSA.cry` | `audit/test_cryptol_specs.cpp` (P12) | OK |

## BIP-340 — Schnorr Signatures

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §Default Signing | `aux_rand` optional, deterministic if absent | `cpu/src/schnorr.cpp` | `audit/test_exploit_schnorr_bip340_kat.cpp` | OK |
| §Verification | Reject `R.y` non-even | `cpu/src/schnorr.cpp` | `audit/test_exploit_schnorr_bip340_kat.cpp`, `audit/test_exploit_schnorr_hash_order.cpp` | OK |
| §Verification | Reject `r ≥ p`, `s ≥ n` | `cpu/src/schnorr.cpp` | `audit/test_exploit_schnorr_bip340_kat.cpp`, `audit/test_exploit_schnorr_hash_order.cpp` | OK |
| §Tagged Hash | `H_tag(x) = SHA256(SHA256(tag) ‖ SHA256(tag) ‖ x)` | `cpu/src/schnorr.cpp` | `audit/test_exploit_schnorr_bip340_kat.cpp` | OK |
| §Batch Verify | $\sum a_i s_i G = \sum a_i R_i + \sum a_i e_i P_i$ with random `a_i` | `cpu/src/batch_verify.cpp` | `audit/test_exploit_schnorr_batch_inflation.cpp`, `audit/test_batch_randomness.cpp` | OK |

## BIP-32 — HD Wallets

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §Master | `(I_L, I_R) = HMAC-SHA512("Bitcoin seed", S)`; reject `I_L = 0` or `I_L ≥ n` | `cpu/src/bip32.cpp` | `audit/test_exploit_bip32_derivation.cpp`, `audit/test_exploit_bip32_child_key_attack.cpp` | OK |
| §CKDpriv | Hardened: `HMAC(c_par, 0x00 ‖ k_par ‖ ser32(i))` for `i ≥ 2³¹` | `cpu/src/bip32.cpp` | `audit/test_exploit_bip32_ckd_hardened.cpp` | OK |
| §CKDpriv | Non-hardened: `HMAC(c_par, ser_P(K_par) ‖ ser32(i))` | `cpu/src/bip32.cpp` | `audit/test_exploit_bip32_derivation.cpp` | OK |
| §CKDpub | Reject identity at any derivation step | `cpu/src/bip32.cpp` | `audit/test_exploit_bip32_parent_fingerprint_confusion.cpp` | OK |
| §Serialization | 4-byte version magic enforced (mainnet xprv/xpub vs testnet) | `cpu/src/bip32.cpp` | `audit/fuzz_bip32_path.cpp`, `audit/test_exploit_bip32_path_overflow.cpp` | OK |
| §Depth Limit | Reject `depth > 255` | `cpu/src/bip32.cpp` | `audit/test_exploit_bip32_depth.cpp` | OK |

## BIP-324 — v2 Transport Protocol

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §ECDH | X-only ECDH on secp256k1 | `cpu/src/bip324.cpp`, `cpu/src/ecdh.cpp` | `audit/test_exploit_bip324_session.cpp`, `audit/test_wycheproof_ecdh.cpp` | OK |
| §HKDF | BIP-324 HKDF labels exactly as spec | `cpu/src/bip324.cpp`, `cpu/src/hkdf.cpp` | `audit/test_exploit_hkdf_kat.cpp`, `audit/test_wycheproof_hkdf_sha256.cpp` | OK |
| §AEAD | ChaCha20-Poly1305 over framed messages | `cpu/src/chacha20_poly1305.cpp` | `audit/test_exploit_bip324_aead_forgery.cpp`, `audit/fuzz_bip324_frame.cpp` | OK |
| §Rekey | Counter rollover triggers rekey | `cpu/src/bip324.cpp` | `audit/test_exploit_bip324_counter_desync.cpp` | OK |
| §Decoy | Decoy packets accepted but discarded | `cpu/src/bip324.cpp` | `audit/test_exploit_bip324_transcript_splice.cpp` | OK |

## BIP-340 / BIP-341 / BIP-342 — Taproot

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| BIP-341 §Taptweak | `Q = P + int(H_TapTweak(P ‖ m)) · G` | `cpu/src/taproot.cpp` | `audit/test_exploit_taproot_tweak.cpp` | OK |
| BIP-341 §Tree | Tapleaf hashing + Merkle path | `cpu/src/taproot.cpp` | `audit/test_exploit_taproot_merkle_path_alias.cpp` | OK |
| BIP-341 §ControlBlock | Strict length: 33 + 32·m, m ∈ [0,128] | `cpu/src/taproot.cpp` | `audit/test_exploit_taproot_scripts.cpp`, `audit/test_exploit_taproot_commitment_adversarial.cpp` | OK |

## BIP-352 — Silent Payments

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §Scan | Per-output `t_k = hash_BIP0352("SharedSecret", ecdh ‖ ser32(k))` | `opencl/kernels/secp256k1_bip352.cl`, `cuda/src/bench_bip352.cu` | `audit/test_bip352_kat.cpp`, `audit/test_gpu_bip352_scan.cpp` | OK |
| §Spend | `D + t_k · G` reproduces output pubkey | `opencl/kernels/secp256k1_bip352.cl` | `audit/test_bip352_kat.cpp`, `audit/test_exploit_bip352_parity_confusion.cpp` | OK |
| §Labels | Optional label tweak | `opencl/kernels/secp256k1_bip352.cl` | `audit/test_bip352_kat.cpp` | OK |

## BIP-327 — MuSig2

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §KeyAgg | Key-aggregation coefficient prevents rogue-key | `cpu/src/musig2.cpp` | `audit/test_musig2_bip327_vectors.cpp`, `audit/test_exploit_musig2_key_agg.cpp` | OK |
| §NonceGen | 64-byte nonce with `aux_rand` mandatory | `cpu/src/musig2.cpp` | `audit/test_exploit_musig2_nonce_reuse.cpp` | OK |
| §NonceAgg | Reject identity in aggregated nonce | `cpu/src/musig2.cpp` | `audit/test_exploit_musig2.cpp` | OK |
| §PartialSign | Strict round-state machine | `cpu/src/musig2.cpp` | `audit/test_exploit_musig2_ordering.cpp`, `audit/test_exploit_musig2_partial_forgery.cpp` | OK |
| §PartialSigVerify | Detect cheating partial sigs | `cpu/src/musig2.cpp` | `audit/test_exploit_musig2_partial_forgery.cpp`, `audit/test_exploit_musig2_byzantine_multiparty.cpp` | OK |

## RFC 9591 — FROST

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §3 | Pedersen DKG with identifiable abort | `cpu/src/frost.cpp` | `audit/test_exploit_frost_dkg.cpp`, `audit/test_exploit_frost_byzantine.cpp` | OK |
| §4 | 2-round signing protocol | `cpu/src/frost.cpp` | `audit/test_frost_kat.cpp`, `audit/test_exploit_frost_signing.cpp` | OK |
| §4.7 | Signature share verification | `cpu/src/frost.cpp` | `audit/test_exploit_frost_identifiable_abort.cpp`, `audit/test_exploit_frost_lagrange_duplicate.cpp` | OK |
| Spec test vectors | Match RFC 9591 Appendix B vectors | `cpu/src/frost.cpp` | `audit/test_frost_kat.cpp` | OK |

## BIP-39 — Mnemonic (used by apps, not core ABI)

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| §Wordlist | Exact wordlist bytes | `cpu/include/secp256k1/bip39_wordlist.hpp` | `audit/test_kat_all_operations.cpp` | OK |
| §Seed | PBKDF2-HMAC-SHA512(mnemonic, "mnemonic"+passphrase, 2048) | `cpu/src/bip39.cpp` | `audit/test_kat_all_operations.cpp` | OK |

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
| `ecdsa_secp256k1_sha256_test.json` | all | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa_secp256k1_sha256.cpp` |
| `ecdsa_secp256k1_sha256_p1363_test.json` | all | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa_secp256k1_sha256_p1363.cpp` |
| `ecdsa_secp256k1_sha512_test.json` | all | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa_secp256k1_sha512.cpp` |
| `ecdsa_secp256k1_sha512_p1363_test.json` | all | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa_secp256k1_sha512_p1363.cpp` |
| `ecdsa_bitcoin_test.json` | all | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa_bitcoin.cpp` |
| `ecdh_secp256k1_test.json` | all | `cpu/src/ecdh.cpp` | `audit/test_wycheproof_ecdh.cpp` |
| `hkdf_sha256_test.json` | all | `cpu/src/hkdf.cpp` | `audit/test_wycheproof_hkdf_sha256.cpp` |
| `hmac_sha256_test.json` | all | `cpu/src/hash_accel.cpp` | `audit/test_wycheproof_hmac_sha256.cpp` |

The previously-broken Stark Bank class (tcId 346) is permanently
regressed by `audit/test_wycheproof_ecdsa_bitcoin.cpp` (RR-004 closed).

---

## Change discipline

When the library implements a new spec clause:

1. Add a row above with `Impl` and `Test` paths in the same commit.
2. If a clause becomes deprecated, mark the row `Deprecated YYYY-MM-DD`
   and keep it; do not delete.
3. CI gate `spec_traceability_check.py` (G-9b) refuses any row whose
   `Impl` or `Test` path no longer exists in the repo.

## DER / BIP-66 — Strict DER Encoding

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| BIP-66 §2 | SEQUENCE tag = 0x30, correct length, no trailing bytes | `cpu/src/ecdsa.cpp` (`ufsecp_ecdsa_sig_from_der`) | `audit/test_fuzz_parsers.cpp`, `audit/test_exploit_der_parsing_differential.cpp` | OK |
| BIP-66 §2 | INTEGER tags = 0x02 for R and S; reject wrong tags | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa.cpp` (500+ vectors) | OK |
| BIP-66 §2 | R and S ≤ 32 scalar bytes; reject oversized | `cpu/src/ecdsa.cpp` | `audit/test_fuzz_parsers.cpp` | OK |
| BIP-66 §2 | Required 0x00 pad when high bit set; reject missing pad | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa.cpp` | OK |
| BIP-66 §2 | Reject unnecessary leading 0x00 pads | `cpu/src/ecdsa.cpp` | `audit/test_wycheproof_ecdsa.cpp` | OK |
| BIP-62 §Low-S | s ≤ n/2 for signing output | `cpu/src/ecdsa.cpp` | `audit/test_exploit_batch_verify_low_s.cpp`, `audit/test_exploit_der_parsing_differential.cpp` (test 12) | OK |
| Round-trip | `sig_to_der(sig_from_der(x)) == x` for all valid DER | `cpu/src/ecdsa.cpp` | `audit/test_fuzz_parsers.cpp` (580K fuzz inputs) | OK |
| DER max size | Output ≤ 72 bytes (`UFSECP_SIG_DER_MAX_LEN`) | `cpu/src/ecdsa.cpp` | `audit/test_fuzz_parsers.cpp`, `docs/DER_PARITY_MATRIX.md` | OK |

## libsecp256k1 — Compatibility Shim (compat layer)

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| secp256k1.h API | `secp256k1_context_*`, `secp256k1_ec_*`, `secp256k1_ecdsa_*` — identical signatures | `compat/libsecp256k1_shim/src/` | `compat/libsecp256k1_shim/tests/shim_test.cpp` | OK |
| secp256k1_extrakeys.h | `keypair_*`, `xonly_pubkey_*` — BIP-340/341 key handling | `compat/libsecp256k1_shim/src/shim_extrakeys.cpp` | `compat/libsecp256k1_shim/tests/shim_test.cpp` | OK |
| secp256k1_schnorrsig.h | `secp256k1_schnorrsig_sign32`, `_verify` | `compat/libsecp256k1_shim/src/shim_schnorr.cpp` | `compat/libsecp256k1_shim/tests/shim_test.cpp` | OK |
| secp256k1_ecdh.h | `secp256k1_ecdh` — shared secret | `compat/libsecp256k1_shim/src/shim_ecdh.cpp` | `compat/libsecp256k1_shim/tests/shim_test.cpp` | OK |
| secp256k1_recovery.h | Recoverable ECDSA sign / recover | `compat/libsecp256k1_shim/src/shim_recovery.cpp` | `compat/libsecp256k1_shim/tests/shim_test.cpp` | OK |
| secp256k1_ellswift.h | ElligatorSwift encode/decode (BIP-324) | `compat/libsecp256k1_shim/src/shim_ellswift.cpp` | `compat/libsecp256k1_shim/tests/shim_test.cpp` | OK |
| secp256k1_musig.h | MuSig2 BIP-327: all 14 functions | `compat/libsecp256k1_shim/src/shim_musig2.cpp` | `compat/libsecp256k1_shim/tests/shim_test.cpp` | OK |
| Parity | Full differential parity vs libsecp256k1 reference | `compat/libsecp256k1_shim/` | `audit/test_exploit_differential_libsecp.cpp` | OK |

## x-only / xonly Pubkeys — BIP-340 / BIP-341

| Spec § | Requirement | Impl | Test | Status |
|--------|-------------|------|------|--------|
| BIP-340 §Encoding | x-only pubkey = 32-byte x coordinate, even y assumed | `cpu/src/schnorr.cpp`, `compat/libsecp256k1_shim/src/shim_extrakeys.cpp` | `audit/test_exploit_schnorr_bip340_kat.cpp` | OK |
| BIP-340 §Verification | Lift x-only to even-y point; reject if not on curve | `cpu/src/schnorr.cpp` | `audit/test_wycheproof_ecdsa.cpp`, `audit/test_exploit_ectester_point_validation.cpp` | OK |
| BIP-341 §key_path | Internal key + tweak → x-only output key | `cpu/src/impl/ufsecp_taproot.cpp` | `audit/test_exploit_taproot_merkle_collision.cpp`, `audit/test_exploit_taproot_script_path.cpp` | OK |
| BIP-341 §annex | Taproot annex hash does not affect x-only key path | `cpu/src/impl/ufsecp_taproot.cpp` | `audit/test_exploit_taproot_annex.cpp` | OK |
| secp256k1_extrakeys.h | `secp256k1_xonly_pubkey_*` shim covers serialize/parse/tweak | `compat/libsecp256k1_shim/src/shim_extrakeys.cpp` | `compat/libsecp256k1_shim/tests/shim_test.cpp` | OK |
