# Bitcoin Core PR Readiness — Current Status Matrix

**Commit:** `c1df659e` (v3.68.0-87-g717976a5)
**Date:** 2026-04-27
**Overall PR Readiness:** **~95%** (9 of 10 original blockers closed)

---

## Executive Summary

All four MUST-FIX blockers are closed. All SHOULD-FIX items are closed except
cross-platform CI (#9, partial). The Bitcoin Core test suite runs **693/693 pass,
0 failures** with UltrafastSecp256k1 as the backend.

Remaining surface work: cross-platform CI runners, explicit noncefp compatibility
proof, and DER hostile parity evidence matrix.

---

## ✅ MUST-FIX — All 4 Closed

| # | Item | Status | How Resolved |
|---|------|--------|--------------|
| 1 | C++20 → C++17 build (PUBLIC propagation) | ✅ **DONE** | `PRIVATE cxx_std_20` — not propagated to consumers |
| 2 | Bitcoin Core `make check` integration test | ✅ **DONE** | 693/693 pass, 0 failures — `docs/BITCOIN_CORE_TEST_RESULTS.json` |
| 3 | Thread safety doc reference in public header | ✅ **DONE** | `ufsecp.h:18` — "Full policy: docs/THREAD_SAFETY.md" |
| 4 | ABI versioning policy documented | ✅ **DONE** | `docs/ABI_VERSIONING.md` — referenced from `ufsecp.h:14` |

---

## ✅ SHOULD-FIX — 4 of 5 Closed

| # | Item | Status | How Resolved |
|---|------|--------|--------------|
| 5 | `.cpp` implementation files in `include/` | ✅ **DONE** | Moved to `cpu/src/` — 0 `.cpp` files remain in `include/ufsecp/impl/` |
| 6 | README exploit test count stale (189→N) | ✅ **DONE** | README reflects current 207-test count |
| 7 | `VERSION.txt` stale | ✅ **DONE** | `VERSION.txt` = `3.68.0` (matches release tag) |
| 8 | Wycheproof CI not visible | ✅ **DONE** | `.github/workflows/wycheproof.yml` — 11 test targets, artifact upload, weekly schedule |
| 9 | Cross-platform build CI | ⚠️ **PARTIAL** | Linux x86_64 confirmed in CI; macOS ARM64 + Windows + Linux ARM64 runners not yet added |

---

## ✅ NICE-TO-HAVE — Closed

| # | Item | Status | Evidence |
|---|------|--------|---------|
| 10 | Bitcoin Core workload benchmarks | ✅ **DONE** | `docs/BITCOIN_CORE_BENCH_RESULTS.json` — SignTransactionECDSA +17.5%, Schnorr +8.3%, VerifyScript +5.4% |

---

## Remaining Open Items

These were not part of the original 10 blockers but are genuine review-surface gaps:

### A. Cross-platform CI (from #9)

**Status:** PARTIAL  
Linux x86_64 is the only CI platform confirmed. Bitcoin Core ships on:
- macOS ARM64 (Apple Silicon) — GitHub Actions `macos-14`
- Windows x86_64 — MinGW cross-compile or `windows-latest`
- Linux ARM64 — QEMU or ARM runner

**Required:** Add CI matrix rows for at least `ubuntu-22.04-arm` + `macos-14`.
The compat shim must compile and `ctest -R shim` must pass on all platforms.

---

### B. Noncefp Compatibility Proof

**Context:** Bitcoin Core's `secp256k1_ecdsa_sign()` accepts a `noncefp` callback
for custom nonce functions. The shim signature accepts this parameter but the
routing logic has not been independently audited against the libsecp256k1 contract.

**Required:** An explicit compatibility proof or audit test demonstrating that
`secp256k1_ecdsa_sign(ctx, sig, msg, sk, noncefp, ndata)` with non-NULL `noncefp`
produces correct output — or a documented statement that Bitcoin Core never passes
a non-NULL `noncefp` in production paths (true as of Bitcoin Core 28.x).

**Evidence location:** `audit/test_exploit_libsecp_eckey_api.cpp` — ECKEY-17
covers sign+verify correctness but not the noncefp parameter routing specifically.

---

### C. DER Hostile Parity Matrix

**Context:** Bitcoin Core's script validation rejects malleable DER-encoded
signatures. The shim passes all 693 test_bitcoin cases, but an explicit hostile
parity matrix (paired encoding, sign-flip, R length injection, leading-zero
stripping) against libsecp256k1's canonical output is not yet documented.

**Required:** `docs/DER_PARITY_MATRIX.md` — a table showing shim DER output
matches libsecp256k1 for canonical, normalized, and edge-case encodings.
Alternatively, `audit/test_exploit_bitcoin_core_rgrinding.cpp` (BC-01) can be
extended with a DER comparison section.

---

## Bitcoin Core Integration Test History

| Date | Commit | Passed | Failed | Note |
|------|--------|--------|--------|------|
| 2026-04-27 | a0d89669 | 0 | 661 | Before hybrid pubkey fix |
| 2026-04-27 | a0d89669 + local patch | 623 | 70 | After hybrid pubkey fix |
| 2026-04-27 | **c1df659e** | **693** | **0** | After musig2 Y-parity fix — **current** |

Two fixes were required:
1. **hybrid pubkey prefix**: shim rejected `0x06`/`0x07` prefixes in 65-byte parse path (`shim_pubkey.cpp`)
2. **musig2 Y-parity**: `musig2_key_agg` forced even-Y on all input keys in violation of BIP-327 (`cpu/src/musig2.cpp`)

---

## PR Readiness by Category

| Category | Status | Evidence |
|----------|--------|---------|
| **Code Quality** | ✅ A | CAAS 207 exploit PoCs, 0 drift |
| **CT Security** | ✅ A | ct-verif, dudect, Valgrind — all sign paths CT |
| **C++17 Compatibility** | ✅ A | `PRIVATE cxx_std_20`, verified no public C++20 symbols |
| **ABI Surface** | ✅ A | 693/693 test_bitcoin pass; `docs/BITCOIN_CORE_TEST_RESULTS.json` |
| **Build Compatibility** | ✅ A | No PUBLIC C++ standard leak; shim builds standalone |
| **Documentation** | ✅ A- | Thread safety, ABI versioning, integration guide, API reference all current |
| **Evidence Package** | ✅ A | Wycheproof CI, bench results, CAAS gates, differential tests |
| **Cross-platform CI** | ⚠️ B | Linux x86_64 only; macOS/Windows/ARM64 pending |
| **Noncefp compat proof** | ⚠️ B+ | Implicit via test pass; no explicit audit test |
| **DER hostile parity** | ⚠️ B+ | Passes all 693; no standalone comparison matrix |

---

## CAAS Evidence Chain

| Gate | Status | What it checks |
|------|--------|---------------|
| Stage 0 — exploit wiring | ✅ 207/207 | Every `test_exploit_*.cpp` has `_run()` in runner |
| Stage 1 — CT analysis | ✅ PASS | Constant-time verification on signing paths |
| Stage 2a — core build mode | ✅ PASS | CMake build config correctness |
| Stage 2b — ABI stability | ✅ PASS | `static_assert` struct layout guards |
| Stage 2c — differential | ✅ PASS | Cross-validation against libsecp256k1 |
| Stage 2d — Wycheproof | ✅ 11/11 | All Wycheproof test suites pass |
| Stage 2e — Bitcoin Core tests | ✅ 693/693 | `docs/BITCOIN_CORE_TEST_RESULTS.json` |

---

*Last updated: 2026-04-27 — reflects commit c1df659e*
