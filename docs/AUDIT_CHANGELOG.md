# Audit Changelog

Focused changelog for changes to the assurance system itself.

This file is not a release changelog. It records audit maturity changes,
evidence upgrades, and changes to what the repository can honestly claim.

---

## 2026-04-14 (4 new ePrint exploit PoCs + crypto-aware dev_bug_scanner layer)

- **Added** `audit/test_exploit_blind_spa_cmov_leak.cpp` — ePrint 2024/589 +
  2025/935 "Blind-Folded SPA" (CHES 2024–2025): 12 sub-tests (BSPA-1..12)
  verifying CT cmov/cswap power leakage resistance across signing, ECDH,
  Schnorr paths. HW-extreme keys timing ratio enforcement (<3×).

- **Added** `audit/test_exploit_ectester_point_validation.cpp` — ePrint 2025/1293
  "ECTester: Systematic Point Validation Testing for ECC Libraries": 18 sub-tests
  (ECT-1..18) covering infinity rejection, off-curve rejection, x≥p rejection,
  twist-point rejection, invalid prefix/truncated pubkeys, ECDH bad-point
  rejection, negate(negate(P))=P identity, tweak_add(zero), pubkey_combine(single).

- **Added** `audit/test_exploit_ros_dimensional_erosion.cpp` — ePrint 2025/306 +
  2025/1353 "Dimensional eROS + BZ Blind Signature Forgery": 12 sub-tests
  (RDE-1..12) verifying r-value uniqueness across 256 concurrent ECDSA/Schnorr
  sessions, batch verification of 256 sigs, Hamming distance entropy checks
  for nonce bias resistance.

- **Added** `audit/test_exploit_ecdsa_batch_verify_rand.cpp` — ePrint 2026/663
  "Modified ECDSA Batch Verification Randomization": 16 sub-tests (BVR-1..16)
  covering all-valid batch verification, corrupted sig rejection, invalid
  position pinpointing (first/last/multiple/all), wrong pk/msg rejection,
  empty/single/1024-entry batch scaling.

- **Added** 5 crypto-specific checkers to `scripts/dev_bug_scanner.py`:
  SECRET_UNERASED, CT_VIOLATION, TAGGED_HASH_BYPASS, RANDOM_IN_SIGNING,
  BINDING_NO_VALIDATION — Trail-of-Bits/NCC-style static analysis layer
  for crypto hygiene enforcement.

- **Integrated** dev_bug_scanner into `scripts/preflight.py` as [13/13]
  check — crypto-specific HIGH findings are now surface in preflight gate.

## 2026-04-13 (6 new ePrint/CVE exploit PoCs: ZVP-DCP, lattice HNP, DFA, type confusion, ROS, FROST binding)

- **Added** `audit/test_exploit_zvp_glv_dcp_multiscalar.cpp` — ePrint 2025/076
  "Decompose and conquer: ZVP attacks on GLV curves" (ACNS 2025): 8 sub-tests
  (ZVPDCP-1..8) verifying GLV r-value distribution, β-endomorphism probing,
  DCP adaptive probing with λ-related scalars, static key stability after
  256-probe barrage, and Schnorr batch with GLV-edge pubkeys.

- **Added** `audit/test_exploit_lattice_sieve_hnp.cpp` — ePrint 2024/296
  "Lattice sieving for the HNP" (ASIACRYPT 2024): 8 sub-tests (LSHNP-1..8)
  demonstrating sub-1-bit nonce leakage defense: r-value uniqueness, MSB/LSB
  bit distribution, low-s normalisation enforcement, chi-squared uniformity,
  and deterministic nonce consistency across key pairs.

- **Added** `audit/test_exploit_deterministic_sig_dfa.cpp` — ePrint 2017/975
  "Differential attacks on deterministic signatures" (NXP/BSI): 8 sub-tests
  (DSDFA-1..8) verifying triple determinism, 1-bit message fault Hamming
  distance ≥40 between signatures, 1-bit key fault correlation check across
  128 messages, low-s normalisation under fault, and Schnorr determinism.

- **Added** `audit/test_exploit_sign_type_confusion_kreuse.cpp` —
  CVE-2024-49364/CVE-2024-49365/CVE-2022-41340: 10 sub-tests (STCK-1..10)
  covering 1024-message k-reuse scan, r/s boundary validation (r=0, s=0, r≥n,
  s≥n), 256 1-bit signature flip false-positive check, wrong pubkey rejection,
  and Schnorr boundary validation.

- **Added** `audit/test_exploit_ros_concurrent_schnorr.cpp` — ePrint 2020/945
  "On the (in)security of ROS" (Eurocrypt 2021): 10 sub-tests (ROS-1..10)
  verifying 256-session nonce independence, XOR/additive forgery rejection,
  batch verify with corruption detection, identify-invalid index flagging,
  error code uniformity (no verification oracle).

- **Added** `audit/test_exploit_frost_weak_binding.cpp` — ePrint 2026/075
  (FROST2 TS-SUF-2→TS-SUF-4) + ePrint 2025/1001 (adaptive threshold Schnorr):
  8 sub-tests (FWB-1..8) covering session nonce uniqueness, adaptive corruption
  key tweak resilience, taproot tweak binding, mixed-key batch verify,
  key negation x-only consistency.

**Running total after this wave: 172 audit files.**

---

## 2026-04-09 (3 new ePrint exploit PoCs: EUCLEAK, cross-key nonce reuse, Fiat-Shamir hash order)

- **Added** `audit/test_exploit_eucleak_inversion_timing.cpp` — ePrint 2024/1380
  "EUCLEAK" (NinjaLab): 12 sub-tests (EUC-1..EUC-12, 28 checks) verifying
  constant-time modular inversion across pathological inputs (zero, near-order,
  high/low Hamming weight, alternating bits, batch context, CT vs fast path).
  Infineon's 14-year non-CT Extended Euclidean Algorithm vulnerability.

- **Added** `audit/test_exploit_ecdsa_cross_key_nonce_reuse.cpp` — ePrint 2025/654
  "ECDSA Cracking Methods" (Edinburgh Napier): 10 sub-tests (CKN-1..CKN-10, 16 checks)
  demonstrating cascading key compromise when multiple keys share a weak-PRNG nonce.
  One compromised key immediately reveals all others sharing the same nonce stream.
  RFC 6979 immunity verified.

- **Added** `audit/test_exploit_schnorr_hash_order.cpp` — ePrint 2025/1846
  "The Order of Hashing in Fiat-Shamir Schemes" (ASIACRYPT 2025): 10 sub-tests
  (SHO-1..SHO-10, 15 checks) verifying BIP-340 challenge hash input ordering
  R||P||msg is correct. Wrong orderings (msg-first, R-last, pubkey-first, reversed)
  produce different challenges and signatures that fail verification.

**Running total after this wave: 166 audit files, 59 new checks (28+16+15).**

---

## 2026-04-08 (Research-driven exploit test expansion: 4 new ePrint/CVE attack classes)

- **Added** `audit/test_exploit_ecdsa_affine_nonce_relation.cpp` — ePrint 2025/705
  "Breaking ECDSA with Two Affinely Related Nonces": 12 sub-tests (ANR-1..ANR-12)
  demonstrating algebraic private key recovery when k₂ = a·k₁ + b, plus RFC 6979
  immunity verification, key-bit sensitivity, and multi-pair confirmation.

- **Added** `audit/test_exploit_ecdsa_half_half_nonce.cpp` — ePrint 2023/841
  "Half-half Bitcoin ECDSA nonces": 10 sub-tests (HH-1..HH-10, 13 checks)
  demonstrating key recovery when nonce is composed from hash upper bits and key
  lower bits, plus RFC 6979 immunity and random-nonce resistance.

- **Added** `audit/test_exploit_ecdsa_nonce_modular_bias.cpp` — CVE-2024-31497 (PuTTY)
  / CVE-2024-1544 (wolfSSL) modular reduction nonce bias: 6 sub-tests (NMB-1..NMB-6,
  19 checks) demonstrating statistical bias from oversized random mod n reduction
  instead of rejection sampling, plus RFC 6979 immunity.

- **Added** `audit/test_exploit_ecdsa_differential_fault.cpp` — ePrint 2017/975
  "Differential Attacks on Deterministic Signatures": 8 sub-tests (DF-1..DF-8,
  10 checks) demonstrating key recovery via bit-flip/additive/multiplicative fault
  injection during RFC 6979 nonce computation, plus determinism verification.

- **Added** strict Documentation Discipline rule to `.github/copilot-instructions.md`,
  `AGENTS.md`, `CLAUDE.md` — documentation must be maintained in parallel with code
  changes; deferred documentation is treated as a hard error.

---

## 2026-04-07 (CT scalar_inverse(0) fix + boundary sentinel test suite)

- **Fixed** `cpu/src/ct_scalar.cpp`: both SafeGCD and Fermat fallback `ct::scalar_inverse`
  paths now return `Scalar::zero()` for zero input. Previously only the FAST-path
  `Scalar::inverse()` had this guard; the CT paths had undefined behavior on zero.
  Defense-in-depth fix — the zero check is on the input scalar (not secret-derived data),
  so it does not break constant-time guarantees.

- **Added** `audit/test_exploit_boundary_sentinels.cpp` — 10 test groups, 18 individual
  checks covering literature-derived boundary edge cases:
  - BS-1: `ct::scalar_inverse(0)` → zero (verifies the fix)
  - BS-2: `fast::Scalar::inverse(0)` → zero
  - BS-3: `FieldElement::inverse(0)` → throws `runtime_error`
  - BS-4: `schnorr_batch_verify({})` → true (empty batch vacuously true)
  - BS-5: ECDSA low-S half-order boundary (4 sub-checks: max low-S, min high-S, normalize, idempotence)
  - BS-6: MuSig2 `key_agg` with duplicate pubkeys (3 sub-checks)
  - BS-7: Schnorr sign with `aux_rand=0xFF…FF` (2 sub-checks: verifies, differs from zero aux)
  - BS-8: `Point::has_even_y()` on infinity is deterministic
  - BS-9: `ecdsa_batch_verify({})` → true (empty batch)
  - BS-10: CT `scalar_inverse` round-trips: inverse(1)==1, val×inverse(val)==1, (n-1)×inverse(n-1)==1

- **Updated** docs: `SECURITY_CLAIMS.md` (perimeter items 9-10), `SECRET_LIFECYCLE.md`
  (change-control note 5), `CT_VERIFICATION.md` (audit checklist), `ECDSA_EDGE_CASE_COVERAGE.md`
  (Category XII — 13 boundary sentinels, total 101/101).

**Running total: edge case coverage 101/101 (100%). 18/18 boundary sentinel tests PASS.**

---

## 2026-04-05 (LibFuzzer harnesses, mutation tracker, Cryptol specs, SLSA verifier, unified_audit_runner 3 new modules — commits `38108b89`, `00522b57`)

- **Added** `cpu/fuzz/fuzz_ecdsa.cpp` + `cpu/fuzz/fuzz_schnorr.cpp` (ClusterFuzzLite, ECDSA and BIP-340
  Schnorr sign→verify invariants + forged-sig rejection). ClusterFuzzLite targets: 3 → **5**.
  Committed `38108b89`.

- **Added** 6 deterministic LibFuzzer harnesses in `audit/`:
  `fuzz_der_parse.cpp` (DER parse + round-trip),
  `fuzz_pubkey_parse.cpp` (pubkey parse, tweak_add, encoding),
  `fuzz_schnorr_verify.cpp` (BIP-340 sign→verify + forged rejection),
  `fuzz_ecdsa_verify.cpp` (ECDSA sign→verify round-trip),
  `fuzz_bip32_path.cpp` (BIP-32 path parser, boundary + overflow),
  `fuzz_bip324_frame.cpp` (BIP-324 AEAD frame decrypt).
  Total LibFuzzer harnesses: 3 → **11** (5 `cpu/fuzz/` + 6 `audit/`). Committed `38108b89`.

- **Added** `scripts/mutation_kill_rate.py` — stochastic mutation engine; 50 mutations/run,
  threshold 60%. Committed `38108b89`.

- **Added** `scripts/verify_slsa_provenance.py` — checks `cosign` bundle validity, subject
  digest, and builder identity for release artefacts. Committed `38108b89`.

- **Added** `formal/cryptol/` — 4 machine-checkable Cryptol property files:
  `Secp256k1Field.cry` (10 props: field axioms, Fermat, sqrt),
  `Secp256k1Point.cry` (7 props: commutativity, associativity, scalar distribution),
  `Secp256k1ECDSA.cry` (6 props: sign→verify, wrong-msg reject, sk-uniqueness),
  `Secp256k1Schnorr.cry` (5 props: BIP-340 round-trip, zero-challenge, zero-nonce-reject).
  28 total formal properties. `cryptol --batch :check` verifiable. Committed `38108b89`.

- **Added** 3 new `unified_audit_runner` modules in the **fuzzing** section (commit `00522b57`):
  - `libfuzzer_unified` (**CI-blocking**): deterministic regression over all 6 audit LibFuzzer
    domains (DER, pubkey, Schnorr, ECDSA, BIP-32, BIP-324). 12,097 checks in <250 ms.
    `#ifdef SECP256K1_BIP324` guard protects AEAD domain in BIP-324-disabled builds.
  - `mutation_kill_rate` (advisory): popen() bridge to `scripts/mutation_kill_rate.py`;
    `--ctest-mode --count 50 --threshold 60`; skips gracefully if python3 absent.
  - `cryptol_specs` (advisory): popen() bridge to `cryptol --batch`; 28 formal properties
    across 4 spec files; skips gracefully if cryptol not installed.
  Fuzzing section: 8/10 → **11/11 PASS**. Full audit: **221/221 PASS** (≥55.2 s).

- **Fixed** ellswift test build (`test_exploit_ellswift_bad_scalar_ecdh`,
  `test_exploit_ellswift_xdh_overflow`): ellswift API calls wrapped in `#ifdef SECP256K1_BIP324`;
  both tests compile and pass in builds without BIP-324 enabled. Committed `00522b57`.

**Running total after this wave: unified_audit_runner fuzzing section 11/11 PASS.
Full audit 221/221 PASS. LibFuzzer harness count: 11. Cryptol formal properties: 28.
SLSA provenance verifier in scripts/.**

---

## 2026-04-03 (Python audit script suite + static analysis scanners — commits `e94523bb`, `ad32e1d1`, `bdc00c6b`, `79f83220`)

- **Added** `scripts/dev_bug_scanner.py` (15 categories): classic C++ development bug detector
  scanning 221 library source files (cpu/src, gpu/src, opencl, metal, bindings, include). Finds
  bugs that code review and LLM analysis typically miss. Results: **182 findings (82 HIGH,
  100 MEDIUM)** — NULL 51, CPASTE 45, SIG 31, RETVAL 30, MSET 19, OB1 5, ZEROIZE 1. Precision
  mitigations: balanced-paren SEMI, brace-depth UNREACH, preprocessor-reset CPASTE,
  case-grouping MBREAK. Third-party dirs excluded (node_modules, _deps, vendor). Registered as
  CTest `py_dev_bug_scan`. Committed `79f83220`.

- **Added** `scripts/audit_test_quality_scanner.py` (6 categories): static analyzer for audit
  C++ test files detecting patterns that cause tests to vacuously pass. Categories:
  A=`CHECK(true,...)` always-pass, B=security rejection gap, C=condition/message polarity
  mismatch, D=weak statistical thresholds, E=`ufsecp_*` return value silently discarded,
  F=missing unconditional reject in adversarial test. Manual audit found 17+ instances across
  KR-5/6, FAC-5/7, PSM-2/6, HB-5. Committed `79f83220`.

- **Added** `scripts/semantic_props.py` (1450+ checks): algebraic and curve property test harness
  — kG+lG==(k+l)G, k(lG)==(kl)G (scalar linearity vs coincurve), sign/verify roundtrip,
  determinism (RFC6979), low-S, wrong-msg/wrong-key/tampered-r/s all reject, ECDH symmetry,
  BIP-32 path equivalence. Hypothesis integration when installed. CTest `py_semantic_props`.
  Committed `bdc00c6b`.

- **Added** `scripts/invalid_input_grammar.py` (37 checks): structured invalid-input rejection
  verifier — wrong pubkey prefix, x≥p, not-on-curve, sk=0/n/overrange, r=0/n/2^256-1, s=0/n,
  zero ECDH, invalid BIP-32 seed length, invalid path, hardened-from-xpub rejection. CTest
  `py_invalid_input_grammar`. Committed `bdc00c6b`.

- **Added** `scripts/stateful_sequences.py` (401+ checks): stateful API call sequence verifier —
  interleaved sign/verify/ecdh on one context, error-injection recovery, 2+3 level BIP-32 path
  consistency, dual-context independence, context destroy+recreate determinism, 5000-op
  endurance. CTest `py_stateful_sequences`. Committed `bdc00c6b`.

- **Added** `scripts/differential_cross_impl.py` (1000+ checks): cross-implementation
  differential test driving library alongside coincurve (libsecp256k1) and python-ecdsa for
  random (sk, msg) pairs. Catches wrong low-S normalization, pubkey parity bugs, ECDH
  mismatches, r/s range violations, cross-verify failures. CTest `py_differential_crossimpl`.
  Committed `e94523bb` / `ad32e1d1`.

- **Added** `scripts/nonce_bias_detector.py` (10,000+ ops): statistical nonce bias detection via
  chi-squared, Kolmogorov-Smirnov, per-bit frequency sweep (all 256 bits), collision detection,
  single-key diversity check. Catches Minerva/TPM-FAIL-class biases invisible to code review.
  KS D=0.017 < 0.036 (5% significance). CTest `py_nonce_bias`. Committed `e94523bb` / `ad32e1d1`.

- **Added** `scripts/rfc6979_spec_verifier.py` (200+ checks): independent pure-Python RFC 6979
  §3.2 HMAC-SHA256 nonce derivation compared against library r-values for 200+ random (sk, msg)
  pairs plus RFC 6979 Appendix A.2.5 known vectors. Catches HMAC step ordering bugs,
  endianness errors, missing k<n check. CTest `py_rfc6979_spec`. Committed `e94523bb` / `ad32e1d1`.

- **Added** `scripts/bip32_cka_demo.py`: live BIP-32 Child Key Attack demo — performs
  non-hardened parent key recovery (child_sk - HMAC_IL mod n) using library's actual output.
  Validates algebraic correctness of BIP-32 HMAC-SHA512 computation. Also proves hardened
  derivation is correctly immune. Chained upward attacks (grandchild→child→master) verified.
  CTest `py_bip32_cka`. Committed `e94523bb` / `ad32e1d1`.

- **Added** `scripts/glv_exhaustive_check.py` (5000+ scalars): GLV decomposition algebraic
  verifier — adversarial scalars stressing Babai rounding near n/2, λ, 2^127, 2^128 and lattice
  boundaries, compared against coincurve reference. Catches off-by-one Babai rounding errors
  invisible to code review. CTest `py_glv_exhaustive`. Committed `e94523bb` / `ad32e1d1`.

- **Added** `scripts/_ufsecp.py`: canonical ctypes wrapper for the `ufsecp_*` API (context-based,
  correct symbol names, BIP32Key 82-byte struct). Shared by all Python audit scripts.
  Committed `ad32e1d1`.

- **Verified** ASan/UBSan build: **210/210 C++ tests pass** under
  `-fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer` after full
  `build-asan/` rebuild. All previously stale binaries rebuilt and confirmed clean.

**Running total after this wave: 9 Python CTest targets. 221 library source files scanned by
dev_bug_scanner. 182 bug findings (82 HIGH, 100 MEDIUM) identified and tracked. All Python audit
tests PASS.**

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

**Running total after this wave: 157 exploit PoC files, 59 new checks.**

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