# Audit Changelog

Focused changelog for changes to the assurance system itself.

This file is not a release changelog. It records audit maturity changes,
evidence upgrades, and changes to what the repository can honestly claim.

---

## 2026-04-04 (Wycheproof extended KAT wave — commits `40a0f218`, `a3a9289a`)

- **Added** `audit/test_wycheproof_ecdsa_secp256k1_sha256.cpp` (27 checks): secp256k1
  ECDSA-SHA256 DER (ASN.1 SEQUENCE) vectors — valid: Group 1 tcId 1/3, high-S malleability
  (tcId 5), large-x (tcId 350), r=n-1/s=n-2 boundary (tcId 352), s==1 (tcId 373); invalid:
  r/s==0 (tcId 168/169/176/374), r>=n (tcId 192/193), r==p (tcId 216), BER long-form, empty,
  tag-only, truncated. Committed `40a0f218`.

- **Added** `audit/test_wycheproof_ecdsa_secp256k1_sha256_p1363.cpp` (32 checks): secp256k1
  ECDSA-SHA256 P1363 (raw r‖s, 64 bytes) vectors — valid: Group 1 tcId 1, large-x (tcId 115),
  small r/s (tcId 120/122), s==1 (tcId 148); invalid: r==0 variants (tcId 11–14), s==0
  (tcId 18/32/149), r>=n (tcId 25/26/39/46), r=p-3>n (tcId 116), wrong size (tcId 121),
  infinity (tcId 165), comparison-with-infinity (tcId 204). Committed `40a0f218`.

- **Added** `audit/test_wycheproof_hmac_sha256.cpp` (174 checks): full Wycheproof HMAC-SHA256
  test set. Committed `a3a9289a`.

- **Added** `audit/test_wycheproof_hkdf_sha256.cpp` (86 checks): full Wycheproof HKDF-SHA256
  test set. Committed `a3a9289a`.

- **Added** `audit/test_wycheproof_chacha20_poly1305.cpp` (316 vectors / 1084 encrypt+decrypt
  ops): full Wycheproof ChaCha20-Poly1305 AEAD test set. Committed `a3a9289a`.

- **Added** `audit/test_wycheproof_ecdsa_secp256k1_sha512.cpp` (544 checks): secp256k1
  ECDSA-SHA512 DER vectors. Includes **DER parser fix**: 33-byte integer with non-zero leading
  byte (e.g. tcId 145, 158) now correctly rejected; previously accepted due to missing
  `leading_zero_required` guard. Committed `a3a9289a`.

- **Added** `audit/test_wycheproof_ecdsa_secp256k1_sha512_p1363.cpp` (312 checks): secp256k1
  ECDSA-SHA512 P1363 vectors. Committed `a3a9289a`.

- **Extended** `audit/test_exploit_hkdf_kat.cpp`: added RFC 5869 TC2 OKM (L=42, all 42 bytes
  verified including bytes 32–41 that were previously missing) and TC3 full test (long salt +
  info, L=82, PRK bytes 0–31 and OKM bytes 0–81 cross-checked). Also corrected wrong expected
  values in TC2 and TC3 that were hallucinated in a prior session. Committed `a3a9289a`.

- **Extended** `audit/test_exploit_ecdsa_rfc6979_kat.cpp`: added Test 13 — secp256k1
  cross-validated sign/verify KAT using private key
  `C9AFA9D845BA75166B5C215767B1D6934E50C3DB36E89B127B8A622B120F6721`.
  Public key G·sk (X=`2C8C31FC9F990C6729E8B6AD839D593B9CB584DA6A37BA78E3A4ABBA3A099B2A`,
  Y=`79BE667EF9DCBBAC...`) verified. SHA-256/"sample" and SHA-256/"test" r,s validated by both
  python-ecdsa and the library. SHA-512/"sample" and SHA-512/"test" vectors confirm determinism
  and sign/verify roundtrip. Committed `a3a9289a`.

**Running total after this wave: 187 ctest audit targets, 2287 new checks across both commits,
100% passing. All 191 audit test source files build and pass.**

---

## 2026-04-04

- **Added** `audit/test_exploit_schnorr_nonce_reuse.cpp` (SNR-1..SNR-16, 16 checks): BIP-340
  Schnorr nonce reuse key recovery PoC.  Proves d' = (s1-s2)·(e1-e2)⁻¹ mod n recovers the
  private key from two signatures sharing nonce k.  Covers known-k construction, nonce
  recovery k = s1-e1·d', RFC6979 safety (different messages → different R), even-y d negation
  case, k vs n-k normalization, and three-message pairwise recovery agreement.
  Committed `c843979c`.

- **Added** `audit/test_exploit_bip32_child_key_attack.cpp` (CKA-1..CKA-18, 18 checks): BIP-32
  non-hardened parent private key recovery PoC.  Proves parent_sk = (child_sk - I_L) mod n
  when the attacker has xpub (chain_code + compressed pubkey).  Covers arbitrary normal indices
  (0, 1, 100, 0x7FFFFFFF), chained upward attack (grandchild → child → master), hardened
  derivation blockage, and to_public() key stripping correctness.  Committed `c843979c`.

- **Added** `audit/test_exploit_frost_identifiable_abort.cpp` (FIA-1..FIA-14, 14 checks): FROST
  identifiable abort / per-participant attribution.  Proves frost_verify_partial() correctly names
  the cheating participant rather than merely detecting protocol failure.  Covers single cheater,
  multi-cheater, honest-not-blamed, coordinator scan loop, wrong-message detection, identity swap
  (P1's z_i submitted as P2), honest-subset protocol completion.  Committed `c843979c`.

- **Added** `audit/test_exploit_hash_algo_sig_isolation.cpp` (HAS-1..HAS-11, 11 checks): hash
  algorithm and format isolation.  Proves: SHA-256 sig rejects under alt-digest; sig rejects
  under raw message bytes; Schnorr sig bytes rejected by ECDSA verify (compact parse);
  ECDSA compact bytes rejected by schnorr_verify; DER ECDSA bytes rejected by schnorr_verify;
  double-hash confusion (H(msg) ≠ H(H(msg))); domain prefix isolation (domain-A sig ≠ domain-B
  sig).  Committed `c843979c`.

**Running total after this wave: 135 exploit PoC files, 59 new checks.**

---

## 2026-04-03

- **Security fix**: discovered and corrected `ecdsa_verify` `r_less_than_pmn` comparison bug in
  `cpu/src/ecdsa.cpp`. The PMN constants were numerically wrong — the code used
  `PMN_1 = 0x14551231950b75fc` (the full high 64-bit word of p-n treated as if p-n < 2^128),
  whereas actual p-n = `0x14551231950b75fc4402da1722fc9baee` has limb[2]=1, not 0.
  The old guard `if (rl[2] != 0) r_less_than_pmn = false` mistakenly declared any r with
  limb[2]=1 as out-of-range for the "try r+n" rare-case path; valid signatures where
  k·G.x ∈ [n, p-1] (~2^−128 probability per sig) were erroneously rejected.
  Fixed both the FE52 (`#if defined(SECP256K1_FAST_52BIT)`) and 4x64 paths with correct
  constants (`PMN_0 = 0x402da1722fc9baee`, `PMN_1 = 0x4551231950b75fc4`) and a 3-way
  comparison (`rl[2]==0` → true, `rl[2]==1` → compare limbs, `rl[2]>1` → false).
  Equivalent vulnerability class: Stark Bank CVE-2021-43568..43572 (false-negative on valid
  large-x signatures). Discovered via Wycheproof tcId 346. Committed `ea8cfb3c`.

- **Added** `audit/test_exploit_ecdsa_r_overflow.cpp` (Track I3-3, 19 checks): r <=> x-coordinate
  comparison edge cases per Wycheproof PR #206 — tcId 346 large-x accept, tcId 347 strict-parse
  rejection (and reduction identity `from_bytes(p-3) mod n` = tcId 346's r confirmed), r=n → zero
  reduction reject, r=0 reject, range sanity and sign/verify consistency.

- **Added** `audit/test_wycheproof_ecdsa_bitcoin.cpp` (Track I3-4, 53 checks): Wycheproof ECDSA
  Bitcoin-variant vectors with BIP-62 low-S enforcement — tcId 346 accept (large-x, low-S),
  tcId 347/348/351, high-S malleability boundary (`is_low_s()` at n/2 and n/2+1),
  sign/normalize/compact roundtrip, r=0/s=0 rejection, point-at-infinity rejection.

- **Corrected** three embedded Wycheproof pubkey hex vectors in `test_wycheproof_ecdsa_bitcoin.cpp`
  that had wrong lengths (63 or 65 hex chars instead of 64) due to JSON leading-zero stripping.
  Fixed from local canonical JSON at
  `_deps/libsecp256k1_ref-src/src/wycheproof/ecdsa_secp256k1_sha256_bitcoin_test.json`:
  tcId 348 (wy missing leading `7`), tcId 351 wx/wy, tcId 386 (entirely different pubkey).

- CMakeLists.txt in `audit/` wired with labels `audit;exploit;ecdsa;wycheproof` and
  `audit;kat;wycheproof;bitcoin`. CTest `ctest -R "wycheproof_ecdsa|exploit_ecdsa_r_overflow"` → 3/3 PASSED.

## 2026-03-27

- Synchronized `docs/TEST_MATRIX.md` summary language with the current
  assurance validator so the top-level target count no longer points at a
  stale 71-target snapshot.
- Added `docs/RESIDUAL_RISK_REGISTER.md` to name the remaining non-blocking
  residual risks and intentional deferrals in one canonical place.
- Marked the originally started owner-grade audit track as closed in
  `docs/OWNER_GRADE_AUDIT_TODO.md`; remaining entries are future hardening,
  not blockers for the completed audit closure.

## 2026-03-26

- Closed the remaining owner-grade audit blockers by finishing ABI
  hostile-caller coverage, CT route documentation, failure-matrix promotion,
  and owner bundle generation.
- `build/owner_audit/owner_audit_bundle.json` reached `overall_status: ready`.
- `scripts/audit_gate.py` passed with no blocking findings for the started
  owner-grade audit track.

## 2026-03-25

- Promoted the failure-class matrix into an executable audit gate surface.
- Added owner-grade visibility for ABI hostile-caller coverage and
  secret-path change control.

## 2026-03-23

- Established the owner-grade audit gate and assurance-manifest workflow.
- Expanded graph and assurance validation so ABI and GPU surfaces were no
  longer invisible to the audit tooling.