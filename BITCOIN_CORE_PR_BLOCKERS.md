# 🚧 Bitcoin Core PR Blockers — Comprehensive Analysis

**Commit:** `c85dc200` (v3.68.0-70-gc85dc200)  
**Date:** 2026-04-27  
**Overall PR Readiness:** **~70%** (code is 96% ready, integration process is at ~70%)

---

## Executive Summary

UltrafastSecp256k1's code quality is **A-**, and the library itself is functionally
ready for Bitcoin Core. However, **opening a PR on bitcoin-core/secp256k1** involves
more than code quality — it requires **integration compatibility**, **build system
alignment**, and **demonstrated drop-in correctness**.

Below are the **10 identified blockers**, categorized by priority.

---

## 🔴 MUST-FIX (4 items — PR will not be accepted without these)

### 1. C++20 → C++17 Build Requirement

**File:** `cpu/CMakeLists.txt:406`

```cmake
target_compile_features(${SECP256K1_LIB_NAME} PUBLIC cxx_std_20)
```

**Problem:** Bitcoin Core builds with **C++17** (`-std=c++17`). The `PUBLIC`
keyword propagates the C++20 requirement to **all consumers** of the library,
including Bitcoin Core's own build. This would cause a compile error when
Bitcoin Core tries to link against UltrafastSecp256k1.

**Fix:**
```cmake
# Option A: Change to cxx_std_17 (if library code actually compiles as C++17)
target_compile_features(${SECP256K1_LIB_NAME} PUBLIC cxx_std_17)

# Option B: Keep C++20 internally but don't propagate
target_compile_features(${SECP256K1_LIB_NAME} PRIVATE cxx_std_20)
```

**Risk:** If the library uses any C++20 features (e.g., `std::span`, concepts,
designated initializers, `<=>`), Option A won't work. Need to verify.

**Effort:** ~1 hour (change + build test + C++17 compatibility scan)

**Verification check:**
```bash
# Scan for C++20 features in source files
grep -rn "concept\|requires\|std::span\|co_await\|co_yield\|co_return\|<=>\|constexpr std::\|constinit\|consteval\|char8_t\|import\|module" \
    libs/UltrafastSecp256k1/cpu/src/ libs/UltrafastSecp256k1/include/ \
    --include="*.cpp" --include="*.h" --include="*.hpp" 2>/dev/null
```

---

### 2. No Bitcoin Core Integration Test (THE Blocker)

**Problem:** The compat shim (`compat/libsecp256k1_shim/`) exists and looks
comprehensive (11 source files), but **no one has run Bitcoin Core's `make check`
with UltrafastSecp256k1 as a drop-in replacement**.

Until this happens, we cannot claim:
- All ABI surfaces match perfectly
- All edge cases (e.g., invalid DER, malleable signatures) are handled identically
- No linker conflicts or ODR violations
- Behavior matches for all 1000+ Bitcoin Core tests

**Required steps:**
1. Build Bitcoin Core from source with `libsecp256k1` replaced by the compat shim
2. Run `make check` (all ~1000 tests)
3. Run `test/functional/` tests (Python integration tests)
4. Document any failures and fix them
5. Add a CI job that runs this periodically

**Effort:** ~2-3 days (initial run + fix + CI setup)

**Current compatibility level:** The shim supports all major API surfaces
(context, keys, ECDSA, Schnorr, extrakeys, ECDH, ElligatorSwift). The recovery
module (`shim_recovery.cpp`) is critical for legacy P2PKH support.

---

### 3. Thread Safety Documentation Not Referenced from Public Header

**Files:** 
- `docs/THREAD_SAFETY.md` — ✅ Exists, 240 lines, comprehensive
- `include/ufsecp/ufsecp.h:14` — ✅ Mentions "each ctx is single-thread"

**Problem:** The public header mentions thread safety once (line 14:
"each ctx is single-thread; create one per thread or protect externally")
but does not **reference** the detailed thread safety documentation.

Bitcoin Core maintainers **will ask**: "Where is the thread safety model
documented?" The answer is "in `docs/THREAD_SAFETY.md`", but there's no
pointer from the header.

**Fix:** Add a comment in `ufsecp.h` design principles section:
```c
 *   5. Thread safety: each ctx is single-thread; create one per thread or
 *      protect externally. See docs/THREAD_SAFETY.md for full model.
```

**Effort:** 5 minutes

---

### 4. ABI Stability — No Documented Versioning Policy

**Problem:** The library has:
- ✅ ABI layout guards (`static_assert` on struct sizes: lines 1723-1740)
- ✅ `ufsecp_abi_version()` link-time check
- ✅ `UFSECP_ABI_VERSION` constant
- ❌ **No documented versioning policy** — what constitutes ABI break vs.
   compatible extension? How is the ABI version number managed?

Bitcoin Core maintainers need to know:
- What changes bump the ABI version?
- Can they safely update the library without recompiling consumers?
- Is there a backward-compatibility guarantee?

**Fix:** Add `docs/ABI_VERSIONING_POLICY.md` with:
- Semver-like rules for ABI version
- What constitutes a breaking change (struct layout change, function removal,
  parameter change, enum value change)
- What is safe (new functions, new constants, performance improvements)
- How ABI version is bumped (automatically? manually? tied to VERSION.txt?)

**Effort:** ~2 hours

---

## 🟡 SHOULD-FIX (5 items — PR will likely be accepted without these, but they'll be review comments)

### 5. `.cpp` Files in `include/` Directory

**Files:** `include/ufsecp/impl/*.cpp` (8 files, 5350 lines total)

**Problem:** Having `.cpp` implementation files in the `include/` directory
is highly unconventional. Bitcoin Core maintainers will flag this as a code
organization issue.

These are unity-build source files that get `#include`d into
`include/ufsecp/ufsecp_impl.cpp`. The pattern works, but the placement is
misleading — developers expect headers (`.h`) in `include/`, not sources.

**Fix:** Move `include/ufsecp/impl/*.cpp` to `cpu/src/impl/` and update the
`#include` paths in `ufsecp_impl.cpp`:
```cpp
// Before:
#include "impl/ufsecp_core.cpp"
// After:
#include "../../cpu/src/impl/ufsecp_core.cpp"
```

**Effort:** ~2 hours (move files + update paths + fix CMake include dirs + test)

---

### 6. README Exploit Test Count Stale

**File:** `README.md:19,20`

```
- 🧠 Continuous audit system — ... 189 exploit PoCs
- 🧪 Adversarially tested — 189 exploit PoC tests
```

**Reality:** `ls audit/test_exploit_*.cpp | wc -l` = **205**

**Problem:** Stale numbers undermine confidence. If the README claims 189 but
there are 205, reviewers will wonder what else is stale.

**Fix:** Update both instances on lines 19 and 20:
```
189 → 205
```

Also update any other stale numbers:
- Line 17 mentions "60 non-exploit modules" — this may also be stale (actual: 247 - 205 = 42 non-exploit? need to verify)

**Effort:** 1 minute

---

### 7. VERSION.txt Stale

**File:** `VERSION.txt`

```
3.66.0
```

**Reality:** `git describe --tags` = `v3.68.0-70-gc85dc200` (70 commits past v3.68.0)

**Effect:** CMake reads VERSION.txt as single source of truth:
```cmake
# CMakeLists.txt reads VERSION.txt
→ UFSECP_VERSION = 3.66.0
→ Binary reports version 3.66.0
→ `ufsecp_abi_version()` returns 3.66.0
```

This is misleading — the actual release is v3.68.0 with 70 additional commits.

**Fix:** Update `VERSION.txt`:
```
3.68.0
```

**Effort:** 1 minute

**Note:** If the version is tightly coupled to release tagging, this may need
a policy decision about whether to bump before or after the PR. Given that
the tag chain shows v3.68.0, bumping to 3.69.0 or 3.68.1 would be reasonable.

---

### 8. Wycheproof Test Results Not Visible in CI

**Files:** `audit/test_wycheproof_*.cpp` (11 files)

**Problem:** The 11 Wycheproof test vectors are present and wired into the
audit runner, but the CI output doesn't explicitly show:

1. Which Wycheproof files are tested
2. Test results (pass/fail per test vector set)
3. Coverage against known invalid curves/keys

Bitcoin Core uses Wycheproof for cross-validation. Having visible results
would strengthen the PR.

**Fix:** Either:
- Add a CI step that runs Wycheproof tests specifically and reports results
- Or add a summary in `docs/WYCHEPROOF_COVERAGE.md`

**Effort:** ~1 day (CI step + reporting)

---

### 9. Cross-Platform Build CI

**Problem:** The library has ambitious cross-platform support claims (ARM64,
RISC-V, WASM, ESP32, STM32) and GPU backends (CUDA, OpenCL, Metal), but
the visible CI (`ci.yml`, `gpu-selfhosted.yml`, `codeql.yml`, `ct-verif.yml`)
appears to focus on x86_64 Linux.

Bitcoin Core is built on:
- **Linux x86_64** (CI primary)
- **macOS ARM64** (Apple Silicon)
- **Windows x86_64** (MinGW/MSVC)
- **Linux ARM64** (aarch64, e.g., Raspberry Pi)

**Fix:** Add CI runners for:
- macOS ARM64 (GitHub Actions macOS runner)
- Windows x86_64 (Windows runner or cross-compile)
- Linux ARM64 (QEMU or ARM runner)

At minimum, show that the compat shim compiles and tests pass on all three
Bitcoin Core target platforms.

**Effort:** ~1 week (CI setup + cross-compilation fixes)

---

## 🟢 NICE-TO-HAVE (1 item — would strengthen PR but not required)

### 10. Bitcoin Core Workload Benchmarks

**Problem:** The existing benchmarks measure microbenchmarks (ECDSA sign/verify,
Schnorr sign/verify, k*G). But Bitcoin Core cares about **real workloads**:

| Workload | What Matters | Current Evidence |
|----------|-------------|-----------------|
| Block validation (~4000 sigs/block) | ECDSA verify batch throughput | ✅ Individual verify: 1.26x |
| Mempool acceptance | ECDSA sign + Schnorr verify | ✅ Individual: 2.45x sign |
| BIP-324 (v2 transport) | ElligatorSwift + AEAD throughput | ❌ No evidence |
| Silent Payments scanning | BIP-352 batch scan | ✅ 11M scans/s (GPU) |
| Descriptor wallet | BIP-32 derivation | ❌ No evidence |

Having a Bitcoin Core-specific benchmark (e.g., "time to validate mainnet
block 850,000") would be very compelling for the PR. The library already has
all the components — just needs to measure the composite workload.

**Effort:** ~1-2 days (build benchmark + run + document)

---

## 📋 PR Readiness Checklist

### 🔴 MUST-FIX (Blocking)

| # | Item | Status | Effort | Owner |
|---|------|--------|--------|-------|
| 1 | C++20 → C++17 build (PUBLIC cxx_std_20) | ❌ Unfixed | ~1h | Library |
| 2 | Bitcoin Core `make check` integration test | ❌ Unfixed | ~2-3d | Library + Core |
| 3 | Thread safety doc reference in public header | ❌ Unfixed | 5min | Library |
| 4 | ABI versioning policy documented | ❌ Unfixed | ~2h | Library |

### 🟡 SHOULD-FIX (Strongly Recommended)

| # | Item | Status | Effort | Owner |
|---|------|--------|--------|-------|
| 5 | `.cpp` files in `include/` → `src/` | ❌ Unfixed | ~2h | Library |
| 6 | README exploit test count (189→205) | ❌ Unfixed | 1min | Library |
| 7 | VERSION.txt (3.66.0→3.68.0) | ❌ Unfixed | 1min | Library |
| 8 | Wycheproof CI visibility | ❌ Unfixed | ~1d | Library |
| 9 | Cross-platform build CI (macOS/Windows/ARM64) | ❌ Unfixed | ~1w | Library |

### 🟢 NICE-TO-HAVE (PR Strengthener)

| # | Item | Status | Effort | Owner |
|---|------|--------|--------|-------|
| 10 | Bitcoin Core workload benchmarks | ❌ Unfixed | ~1-2d | Library |

---

## 🎯 Suggested Action Plan

### Phase 1 — Quick Wins (1 hour)
| Item | Time | Details |
|------|------|---------|
| 3 | 5 min | Add `See docs/THREAD_SAFETY.md` to ufsecp.h |
| 6 | 1 min | Fix README exploit count 189→205 |
| 7 | 1 min | Fix VERSION.txt 3.66.0→3.68.0 |
| 1 | ~1h | Fix CMake C++ standard, verify C++17 compatibility |

### Phase 2 — Documentation (3 hours)
| Item | Time | Details |
|------|------|---------|
| 4 | ~2h | Write `docs/ABI_VERSIONING_POLICY.md` |
| 5 | ~2h | Move .cpp files out of include/ |

### Phase 3 — Integration (3-5 days)
| Item | Time | Details |
|------|------|---------|
| 2 | ~2-3d | Bitcoin Core `make check` integration test |
| 8 | ~1d | Wycheproof CI visibility |
| 9 | ~1w | Cross-platform build CI |

### Phase 4 — Polish (2 days)
| Item | Time | Details |
|------|------|---------|
| 10 | ~1-2d | Bitcoin Core workload benchmarks |

---

## 📊 PR Readiness by Category

| Category | Current | Target | Gap |
|----------|---------|--------|-----|
| **Code Quality** | A- (96%) | A | Negligible |
| **CT Security** | A (verified) | A | ✅ None |
| **GPU Parity** | A | N/A (Bitcoin Core doesn't use GPU) | ✅ Irrelevant |
| **Build Compatibility** | C (C++20 barrier) | A (C++17 clean) | ⚠️ #1, #5 |
| **Integration Testing** | D (untested) | A (CI-gated) | 🔴 #2, #9 |
| **Documentation for Reviewers** | B (good docs, no ABI policy) | A | 🟡 #4, #3 |
| **Evidence Package** | B (benchmarks exist, no real workload) | A | 🟡 #8, #10 |

**Final estimate:** PR readiness **~70%** — code is excellent, but the
integration bridge needs substantial work before Bitcoin Core maintainers
would accept it.

---

*End of Bitcoin Core PR Blockers Analysis*
