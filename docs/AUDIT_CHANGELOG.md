# Audit Changelog

Focused changelog for changes to the assurance system itself.

This file is not a release changelog. It records audit maturity changes,
evidence upgrades, and changes to what the repository can honestly claim.

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