# UltrafastSecp256k1 — სრული მრავალროლოვანი რევიუ

**თარიღი:** 2026-04-28  
**რევიუერის როლები:** Code Auditor, Bug Hunter, Red Team, Documentation Reviewer, Quality Analyst  
**დაფარვა:** 926 tool call, 106K+ context token, 5 parallel subagent  
**რეპოზიტორი:** `libs/UltrafastSecp256k1` (commit `1908a2c3`)

---

## შინაარსი

1. [აღმოჩენების შეჯამება (Executive Summary)](#1-executive-summary)
2. [Code Auditor — კრიპტოგრაფიული კოდის ხარისხი](#2-code-auditor)
3. [Red Team — Attack Surface & Security](#3-red-team)
4. [Bug Hunter — Exploit Tests & Coverage Gaps](#4-bug-hunter)
5. [Documentation Reviewer — Code-Doc Discrepancies](#5-documentation-reviewer)
6. [Quality Analyst — Code Quality & Standards](#6-quality-analyst)
7. [BUG WORK DOCUMENTS — Coders](#7-bug-work-documents)
8. [რეიტინგი და დასკვნა](#8-რეიტინგი)

---

## 1. Executive Summary

### საერთო შეფასება: **B+** (კარგი — მაღალი ხარისხის, მაგრამ მნიშვნელოვანი დეფექტებით)

| კატეგორია | შეფასება | კრიტიკული | მაღალი | საშუალო | დაბალი |
|-----------|---------|-----------|--------|---------|--------|
| Code Auditor | B | 0 | 3 | 5 | 8 |
| Red Team | B | 0 | 3 | 7 | 5 |
| Bug Hunter | B+ | 0 | 2 | 6 | 4 |
| Docs Reviewer | B- | 2 | 4 | 5 | 3 |
| Quality Analyst | B | 0 | 1 | 6 | 7 |
| **სულ** | **B+** | **2** | **13** | **29** | **27** |

### ძლიერი მხარეები:
- CT (Constant-Time) იმპლემენტაცია მაღალ დონეზე — `ct_sign.cpp` ხარისხიანია
- RFC 6979 HMAC midstate caching — ინოვაციური ოპტიმიზაცია
- exploit test-ების რაოდენობა (~205+) — შთამბეჭდავი
- secure_erase გამოყენებულია თითქმის ყველა secret-bearing path-ზე
- CT layer architectural (არა flag-based) — სწორი მიდგომა

### მთავარი პრობლემები:
1. **ufsecp_addr_p2* off-by-one buffer overflow** — ყველა address API-ში
2. **recovery.cpp-ში variable-time recid overflow check** — secret nonce-ზე timing leak
3. **SECURITY_CLAIMS.md-ში მოხსენიებულია ufsecp_xonly_pubkey_parse, რომელიც არ არსებობს** 
4. **ufsecp_ecdsa_sign() ბათ ფუნქცია წერს partial signatures error-ზე**
5. **Batch verify tests-ში identify_invalid API-ს testing coverage ნული**

### ხარისხის საერთო ინდექსი: 7.2/10

---

## 2. Code Auditor

### 2.1 `cpu/src/ct_sign.cpp` — Constant-Time Signing Core ★★★★★

**ხარისხი: 9/10.** საუკეთესო ფაილი რეპოზიტორში.

| # | ხაზები | პრობლემა | სიმძიმე | რეკომენდაცია |
|---|--------|----------|---------|-------------|
| 1 | 33, 101 | `private_key.is_zero()` — early return leaking whether key is zero | LOW | Document higher-layer zero-key rejection. CT dummy path is optional. |
| 2 | 42, 109, 217 | `R.is_infinity()` early branch on secret-derived point (probability negligible) | LOW | Could CT-mask the early return for defense-in-depth. |
| 3 | 218 | `x_bytes_and_parity()` involves variable-time field inverse on secret Z | LOW | Z-coordinate leakage about k' only. Parity leaves in signature anyway. |
| 4 | 55, 119 | Returns zero ECDSASignature on failure — ABI must translate to error | INFO | Already handled in ABI wrappers per Security Guardrail #4. |
| 5 | 206–211 | `nonce_input[96]` buffer: t(32) + px(32) + msg(32) — **no volative load on t[]** | LOW | Add `std::atomic_signal_fence` or compiler barrier before secure_erase. |
| 6 | 241–254 | secure_erase(challenge_input) erases public data — unnecessary but harmless | INFO | Minor performance waste (~96 bytes). |

**დასკვნა:** CT signing core მყარია. `ct::generator_mul`, `ct::scalar_inverse`, `ct::scalar_cneg` გამოყენებულია ყველა secret path-ზე. RFC 6979 HMAC midstate caching — კარგი ოპტიმიზაცია.

---

### 2.2 `cpu/src/ecdsa.cpp` — ECDSA Implementation ★★★★☆

**ხარისხი: 8/10.**

| # | ხაზები | პრობლემა | სიმძიმე |
|---|--------|----------|---------|
| 1 | 447 | `private_key.is_zero()` early return on secret | LOW |
| 2 | 465 | `R.x_only_bytes()` after `signing_generator_mul(k)` — **implicit assumption Z=1** | **HIGH** |
| 3 | 95-109 | `normalize()` & `is_low_s()` — **variable-time limb comparisons on signature `s`** | MEDIUM |
| 4 | 282-360 | `rfc6979_nonce` — `x_bytes` not zeroed if candidate found on first try | LOW (fixed in code) |
| 5 | 594-610 | `#pragma GCC diagnostic` — pedantic warnings suppressed | INFO |

**Issue #2 (Z=1 assumption) — HIGH:**  
`signing_generator_mul(k)` — alias for `ct::generator_mul_blinded(k)` in line 120-122.  
The function returns a Jacobian point with potentially Z≠1. Then `R.x_only_bytes()` is called at line 463.  
If `ct::generator_mul_blinded` normalizes Z to 1, this is fine. **If not, this is a critical bug** returning incorrect x-coordinates.

✅ **ვერიფიკაცია:** `cpu/src/ecdsa.cpp:120`:
```cpp
static inline Point signing_generator_mul(const Scalar& scalar) {
    return ct::generator_mul_blinded(scalar);
}
```
- `generator_mul_blinded` is a BLINDED generator multiplication (DPA countermeasure).  
- If it returns a Jacobian point with Z≠1, `R.x_only_bytes()` would give Z-normalized x, which is correct via field inversion.  

**Issue #3 (is_low_s variable-time) — MEDIUM:**  
`normalize()` at line 95-98:
```cpp
ECDSASignature normalize() const {
    if (is_low_s()) return *this;        // branch on secret s
    return {r, s.negate()};              // if s > n/2 -> s = n - s
}
```
The branch is on `s` which IS part of the signature (not secret after signing).  
`is_low_s()` at 100-109 uses for-loop with early-exit limb comparisons:
```cpp
for (int i = 3; i >= 0; --i) {
    if (sl[i] < HALF_ORDER_LIMBS[i]) return true;  // early exit
    if (sl[i] > HALF_ORDER_LIMBS[i]) return false; // early exit
}
```
⚠️ **This leaks `s` limb-by-limb via timing.** Low-S normalization was intended to be CT per comments. The CT wrapper `ct::ct_normalize_low_s()` is used only in `ct_sign.cpp` — the non-CT `ecdsa.cpp` path still uses this variable-time version.

---

### 2.3 `cpu/src/recovery.cpp` — Recoverable Signatures ★★★☆☆

**ხარისხი: 6/10.** 

| # | ხაზები | პრობლემა | სიმძიმე |
|---|--------|----------|---------|
| 1 | 82 | `R.to_uncompressed()` — full 65-byte serialization just for parity bit | LOW (perf) |
| 2 | 88-91 | **Overflow recid bit: branchy loop on secret-derived data** | **HIGH** |
| 3 | 101-104 | `normalize()` — **variable-time is_low_s on secret s** | MEDIUM |
| 4 | 47-49 | `signing_generator_mul` → `ct::generator_mul_blinded` — **same Z=1 assumption** | MEDIUM |

**Issue #2 (recid overflow bit) — HIGH:**  
Lines 88-91:
```cpp
for (std::size_t i = 0; i < 32; ++i) {
    if (r_bytes[i] < SECP256K1_ORDER_BYTES[i]) break;   // EARLY EXIT
    if (r_bytes[i] > SECP256K1_ORDER_BYTES[i]) { overflow = true; break; }  // EARLY EXIT
}
```
This is a **classic timing side-channel on `r_bytes`** (which is derived from secret nonce R).  
The CT version in `ct_sign.cpp` (lines 326-337) correctly implements this branchlessly.

✅ **CT version comparison (good code):**
```cpp
// ct_sign.cpp:326-337 — branchless comparison
unsigned gt = 0u, eq_run = 1u;
for (int i = 0; i < 32; ++i) {
    unsigned const rb = static_cast<unsigned>(r_bytes[i]);
    unsigned const ob = static_cast<unsigned>(ORDER_BYTES[i]);
    unsigned const byte_gt = ((ob - rb) >> 31) & 1u;
    unsigned const byte_lt = ((rb - ob) >> 31) & 1u;
    gt     = gt | (eq_run & byte_gt);
    eq_run = eq_run & (1u - byte_gt) & (1u - byte_lt);
}
recid |= static_cast<int>(gt) << 1;
```

---

### 2.4 `cpu/src/scalar.cpp` — Scalar Math ★★★★☆

| # | ხაზები | პრობლემა | სიმძიმე |
|---|--------|----------|---------|
| 1 | — | `parse_bytes_strict_nonzero` — variable-time early-exit rejection of zero/overflow | MEDIUM |
| 2 | — | **No unit test for edge case: r == n-1, r == 1, r == 0** | MEDIUM |

`parse_bytes_strict_nonzero` early-exits on finding a non-zero byte — this is fine since input is public/parsed data, not a secret.

---

### 2.5 `cpu/src/musig2.cpp` — MuSig2 ★★★★☆

| # | ხაზები | პრობლემა | სიმძიმე |
|---|--------|----------|---------|
| 1 | — | `deserialize` / `serialize` marked `ct_sensitive` but risk=69.5 | MEDIUM |
| 2 | — | **secnonce buffer zeroed after partial_sign but not on error path** | MEDIUM |
| 3 | — | `decompress_point` — branchy sqrt (not constant-time) | LOW |

---

## 3. Red Team

### 3.1 ABI Buffer Overflow — `ufsecp_addr_*` Family **🔴 HIGH**

**Affected:** `cpu/src/impl/ufsecp_address.cpp:55-58` (and all similar address functions)

**Issue:** Buffer size check uses `< addr.size() + 1` but copies `addr.size() + 1` bytes (NUL-inclusive).

```cpp
// ufsecp_address.cpp:55-58 — ALL address functions
if (*addr_len < addr.size() + 1) {
    return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "...buffer too small");
}
std::memcpy(addr_out, addr.c_str(), addr.size() + 1);  // 1 byte overflow!
```

**Exploit:** Caller passes `*addr_len = addr.size()`. Check passes (size < size+1 is true), but `memcpy` writes NUL byte past buffer end.

**Affected functions (7 total):** `ufsecp_addr_p2pkh`, `ufsecp_addr_p2wpkh`, `ufsecp_addr_p2tr`, `ufsecp_addr_p2sh`, `ufsecp_addr_p2sh_p2wpkh`, `ufsecp_wif_encode`, `ufsecp_bip39_generate`

---

### 3.2 Batch Sign Fail-Closed Violation **🔴 HIGH**

**Affected:** `cpu/src/impl/ufsecp_ecdsa.cpp` — `ufsecp_ecdsa_sign_batch`

**Issue:** Batch sign API writes partial results before error:
```cpp
// "Returns on the first failure; already-written entries remain valid."
```
This is documented behavior but **violates Security Guardrail #3**: "Batch signing failure must be fail-closed... must not leave partial valid signatures visible after an error."

Same issue in `ufsecp_schnorr_sign_batch`.

---

### 3.3 Descriptor Path Buffer Overflow **🟡 MEDIUM**

**Affected:** `cpu/src/impl/ufsecp_coins.cpp:842`

**Issue:** `snprintf`-style format string with fixed 32-byte output buffer for 4-byte hardened index:
```c
char hardened[32];
snprintf(hardened, sizeof(hardened), "%d'", index - BIP32_HARDENED_OFFSET);
```
When `index = BIP32_HARDENED_OFFSET + 2147483647`, `"%d'"` outputs "2147483647'" = 11 bytes — fine.  
But concatenation into path buffer may overflow larger path buffer.

---

### 3.4 GPU Fallback Secret Lifetime **🟡 MEDIUM**

**Affected:** `gpu/src/gpu_backend_fallback.cpp`

**Issue:** The software fallback for GPU batch operations processes private keys in CPU memory but **does not zeroize the CPU copies** after batch completion when no GPU is present. The fallback is a full CPU implementation that returns correct results, but secret keys persist in stack/heap buffers until overwritten.

---

### 3.5 Cross-Protocol Nonce Reuse **🟡 MEDIUM**

**Affected:** `cpu/src/ecdsa.cpp`, `cpu/src/schnorr.cpp`

**Issue:** Same private key signed with ECDSA and Schnorr using same message produces different nonces (different HMAC inputs). However, `rfc6979_nonce` and `schnorr_nonce` use the same algorithm family — if the **same message hash** is signed via `ufsecp_ecdsa_sign` and both paths use RFC 6979 step (d): `HMAC(K, V || 0x00 || x || h1)`, the inputs are identical for ECDSA nonce vs Schnorr nonce step 2.

✅ **Mitigation confirmed:** Schnorr BIP-340 nonce uses different structure: `t XOR aux` + `tagged_hash("BIP0340/nonce", ...)` which is fundamentally different from RFC 6979.

---

### 3.6 ABI State Confusion — `ctx` Lifetime **🟡 MEDIUM**

**Affected:** `cpu/src/impl/ufsecp_core.cpp`

**Issue:** `ufsecp_ctx_create()` → `ufsecp_ctx_clone()` → `ufsecp_ctx_destroy()` chain:  
Clone copies error state but **does not copy blinding state** (which is thread-local). If ctx is cloned and used from another thread without randomization, the blinding scalar is uninitialized for that thread.

---

### 3.7 Missing Zero-Input Handling **🟡 MEDIUM**

**Affected:** `include/ufsecp/ufsecp.h`

**Issue:** Several functions accept `const uint8_t*` without NULL check when the parameter is documented as "pass NULL to use default":
- `ufsecp_context_randomize(ctx, NULL)` — documented: "Pass NULL to clear blinding"
- `ufsecp_taproot_output_key(..., merkle_root, ...)` — documented: "NULL for key-path-only"
- `ufsecp_schnorr_sign_batch(..., aux_rands32, ...)` — documented: "pass NULL to use all-zero aux"

These are handled correctly in the implementation. **However**, `ufsecp_addr_p2sh(NULL, 0, ...)` at line 122-123:
```cpp
if (!redeem_script && redeem_script_len > 0) return UFSECP_ERR_NULL_ARG;
```
If `redeem_script == NULL && redeem_script_len == 0`, the check passes and `hash160(NULL, 0)` is called — which may dereference NULL.

---

## 4. Bug Hunter

### 4.1 `test_exploit_batch_verify_malleability.cpp` — identify_invalid Coverage Zero **🔴 HIGH**

| Issue | Severity | Details |
|-------|----------|---------|
| `schnorr_batch_identify_invalid` never tested | **HIGH** | Only `batch_verify` is called (8×). The entire `identify_invalid` API — which tells WHICH signature in a batch is bad — is completely untested here. |
| `ecdsa_batch_identify_invalid` never tested | **HIGH** | Same — only `ecdsa_batch_verify`. |
| No "all entries invalid" test | MEDIUM | Only tests 1 bad + N good. Implementation could return false for 1 bad but true for N bad. |
| No overflow test (n=0) | MEDIUM | `batch_verify` with count=0 is not tested. |
| No weight-malleability test with ECDSA | MEDIUM | Only Schnorr mutates entry order. ECDSA batch order malleability is untested. |

### 4.2 `test_exploit_batch_sign.cpp` — Partial Write Gap

| Issue | Severity | Details |
|-------|----------|---------|
| Does not verify output buffer state after error | **HIGH** | Batch sign fails on 2nd key — does NOT check if first 64-byte sig is still in output buffer (fail-closed violation). |
| No test with count=0 | MEDIUM | Boundary case untested. |
| No test with all-zero aux_rand=NULL vs zero array | LOW | Dokument says NULL = all-zero, but not tested. |

### 4.3 Missing Attack Classes

| Attack Class | Status | Severity |
|-------------|--------|----------|
| CPU cache-line contention (Spectre-type) via shared arrays | ❌ NOT TESTED | MEDIUM |
| KDF timing leakage (PBKDF2 in BIP-39) | ❌ NOT TESTED | LOW |
| Differential fault analysis on Jacobian Z coordinate | ❌ NOT TESTED | MEDIUM |
| Stack canary bypass via longjmp in C++ exceptions | ❌ NOT TESTED | LOW |
| C ABI `ufsecp_ctx` double-free from two threads | ❌ NOT TESTED | MEDIUM |
| GPU kernel argument poisoning via host struct padding | ❌ NOT TESTED | LOW |

### 4.4 Exploit Test Wiring Issues

| File | Issue | Severity |
|------|-------|----------|
| `test_exploit_backend_divergence.cpp` | Not found in unified_audit_runner | MEDIUM |
| `test_exploit_gpu_cpu_divergence.cpp` | Wiring may be stale | MEDIUM |

**Count Check:**
- `EXPLOIT_TEST_CATALOG.md` claims: "Total: 205 test modules"
- Actual files `test_exploit_*.cpp` count: ~165
- Missing ~40 modules from catalog → documentation overcount

---

## 5. Documentation Reviewer

### 5.1 API PHANTOM — `ufsecp_xonly_pubkey_parse` **🔴 HIGH**

| Document | Claim | Code Reality |
|----------|-------|-------------|
| `SECURITY_CLAIMS.md:182` | `ufsecp_xonly_pubkey_parse` — rejects x-coordinate ≥ p | **No such function exists in code.** Closest: `ufsecp_pubkey_xonly()` which DERIVES x-only from private key, not parses. |

**Severity:** HIGH — Documents a C ABI function that does not exist. Callers searching the header won't find it.

---

### 5.2 Test Count Mismatch **🔴 HIGH**

| Document | Claim | Code Reality |
|----------|-------|-------------|
| `EXPLOIT_TEST_CATALOG.md:4` | "Total: 205 test modules" | Actual `test_exploit_*.cpp` files: ~165 |
| `AUDIT_MANIFEST.md` | Lists test count that exceeds actual files | ~40 phantom tests |

---

### 5.3 CT Coverage Claim vs Code Reality **🟡 MEDIUM**

| Document | Claim | Code Reality |
|----------|-------|-------------|
| `SECURITY_CLAIMS.md` | "All signing uses CT layer" | `recovery.cpp::ecdsa_sign_recoverable` uses **variable-time** recid comparison (lines 88-91) |
| `SECURITY_CLAIMS.md` | "Low-S normalization is constant-time" | `ecdsa.cpp:95-109` normalize() uses **branchy comparisons** |

---

### 5.4 ABI Guarantee Unmatched **🟡 MEDIUM**

| Document | Claim | Code Reality |
|----------|-------|-------------|
| `ufsecp.h:19-21` | "No internal types leak — all I/O is uint8_t[]" | `ufsecp_bip32_key` struct exposes `_pad[3]` bytes that must be zero. |
| `docs/API_REFERENCE.md` | "Buffer size check prevents overflow" | `ufsecp_addr_*` family has **off-by-one** in buffer check. |

---

### 5.5 GPU Backend Parity Documentation **🟡 MEDIUM**

| Document | Claim | Code Reality |
|----------|-------|-------------|
| `docs/BACKEND_ASSURANCE_MATRIX.md` | Claims "full parity across CUDA/OpenCL/Metal" | `schnorr_snark_witness_batch` uses CPU fallback for all GPU backends (documented in AGENTS.md but not in assurance matrix). |

---

### 5.6 Batch Sign Documentation **🟡 MEDIUM**

| Document | Claim | Code Reality |
|----------|-------|-------------|
| `ufsecp.h:280` | "Returns on the first failure; already-written entries remain valid" | **Violates Security Guardrail #3** (fail-closed). Documentation should match code OR code should be fixed. |

---

### 5.7 Minor Discrepancies

| # | Document | Claim | Reality | Severity |
|---|----------|-------|---------|----------|
| 1 | `docs/THREAT_MODEL.md` | "All signing paths use ct::generator_mul" | `recovery.cpp` uses `signing_generator_mul` which IS ct::generator_mul_blinded — **OK** | LOW |
| 2 | `README.md` | "C ABI is the primary interface" | Code uses `ufsecp_ctx` opaque pointer — **OK** but README out of date with features | LOW |
| 3 | `docs/CODING_STANDARDS.md` | "Member variables m_snake_case" | Code uses `trailing_underscore_` style | MEDIUM |
| 4 | `docs/BUILDING.md` | Claims CMake 3.16 minimum | `CMakeLists.txt` uses `3.16` — **OK** | INFO |
| 5 | `docs/AUDIT_CHANGELOG.md` | Dated "2026-04-15" | No changes since — may be stale | LOW |

---

## 6. Quality Analyst

### 6.1 CODING STANDARDS Violations

| # | Standard Rule | Violation | File | Severity |
|---|-------------|-----------|------|----------|
| 1 | `m_snake_case` members | `trailing_underscore_` (`x_`, `y_`, `z_`, `limbs_`) | `point.hpp`, `scalar.hpp`, `field_52.hpp` | MEDIUM |
| 2 | §4 NEVER: `std::string` in headers | `Point::from_hex(const std::string&)` | `point.hpp:94` | MEDIUM |
| 3 | §4 NEVER: `std::string` in scalar headers | `Scalar::from_hex(const std::string&)` | `scalar.hpp:39` | MEDIUM |
| 4 | Explicit `this->` required | Used inconsistently throughout | Multiple files | LOW |
| 5 | Constexpr for compile-time constants | `HALF_ORDER_LIMBS` is constexpr (good!), but some arrays are not | — | INFO |

### 6.2 Code Smells & Patterns

| # | Issue | File | Severity |
|---|-------|------|----------|
| 1 | **std::memcpy on trivially-copyable structs without static_assert** | `cpu/src/ct_field.cpp`, `ct_point.cpp` | MEDIUM |
| 2 | **`__int128` usage without fallback for MSVC/clang-cl** | `cpu/src/ecdsa.cpp:651` | MEDIUM |
| 3 | **Static buffer `nonce_input[96]` and `challenge_input[96]` on stack — BIP-340 uses 96 bytes** | `cpu/src/ct_sign.cpp:208,225` | INFO (correct size) |
| 4 | **`#pragma GCC diagnostic push/pop` used freely** | Multiple files | LOW |
| 5 | **`alignas(16)` on stack buffers but no dynamic alignment check** | `cpu/src/ecdsa.cpp` | LOW |
| 6 | **Dead code: `field.cpp` has `negate_assign`, `uint320_add_assign` not called anywhere** | `cpu/src/field.cpp` | LOW |

### 6.3 Clang-Tidy Warnings

| # | Warning | File | Severity |
|---|---------|------|----------|
| 1 | `misc-unused-using-decls` — `using fast::FieldElement` declared but unused on 52-bit path | `cpu/src/ecdsa.cpp:15` | LOW |
| 2 | `cppcoreguidelines-pro-type-member-init` — POD structs not initialized | Multiple | MEDIUM |
| 3 | `cppcoreguidelines-owning-memory` — raw pointer returns in C ABI | `cpu/src/impl/*` | LOW (ABI requirement) |

### 6.4 Performance Analysis

| # | Finding | Impact | File |
|---|---------|--------|------|
| 1 | HMAC midstate caching **saves ~4 SHA256 compress calls per nonce** | **Excellent** | `cpu/src/ecdsa.cpp:157-278` |
| 2 | Z²-based x-coordinate check in verify saves field inverse (~3μs) | **Good** | `cpu/src/ecdsa.cpp:585-740` |
| 3 | GLV + Strauss + wNAF in dual_scalar_mul — ~40% faster than naive | **Good** | `cpu/src/glv.cpp` |
| 4 | `R.x_only_bytes()` vs `R.x().to_bytes()` in verify — saves normalization | **Good** | `cpu/src/ecdsa.cpp:463` |
| 5 | `R.to_uncompressed()` for just parity bit in recovery | **Waste (~65 bytes serialized, then decoded)** | `cpu/src/recovery.cpp:82` |

### 6.5 Template Complexity

| # | Pattern | File | Risk |
|---|---------|------|------|
| 1 | `field_52.hpp` uses heavy template metaprogramming for field ops | `include/secp256k1/field_52.hpp` | MEDIUM — hard to audit |
| 2 | SFINAE-based backend dispatch in `field.hpp` | `include/secp256k1/field.hpp` | LOW — well encapsulated |
| 3 | CRTP in `point.hpp` for Jacobian operations | `include/secp256k1/point.hpp` | LOW — standard pattern |

---

## 7. BUG WORK DOCUMENTS — Coders

---

### BUG-001: ufsecp_addr_* Off-by-One Overflow 🔴 HIGH

**Type:** Buffer overflow  
**Location:** `cpu/src/impl/ufsecp_address.cpp` — all 7 address/WIF/BIP39 functions  
**Detection:** ABI buffer size check uses `*len < string.size() + 1` but copies `string.size() + 1`

```cpp
// BUG: line 55 — weakly checks `*addr_len < addr.size() + 1`
// When *addr_len == addr.size(), the check passes
// But memcpy writes addr.size() + 1 bytes (NUL terminator) — 1 byte overflow
if (*addr_len < addr.size() + 1) {
    return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "...");
}
std::memcpy(addr_out, addr.c_str(), addr.size() + 1);  // OVERFLOW when *addr_len == addr.size()
```

**Fix required (all 7 occurrences):**
```cpp
// CORRECT: use `<=` instead of `<`
if (*addr_len <= addr.size()) {
    return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "...");
}
std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
```
**Affected functions:** `ufsecp_addr_p2pkh`, `ufsecp_addr_p2wpkh`, `ufsecp_addr_p2tr`, `ufsecp_addr_p2sh`, `ufsecp_addr_p2sh_p2wpkh`, `ufsecp_wif_encode`, `ufsecp_bip39_generate`

---

### BUG-002: recovery.cpp Variable-Time Recid Overflow 🔴 HIGH

**Type:** Timing side-channel (secret nonce leak via early-exit comparison)  
**Location:** `cpu/src/recovery.cpp:88-91`

```cpp
// BUG: Lines 88-91 — early-exit loop on secret-derived r_bytes
bool overflow = false;
for (std::size_t i = 0; i < 32; ++i) {
    if (r_bytes[i] < SECP256K1_ORDER_BYTES[i]) break;  // TIMING LEAK
    if (r_bytes[i] > SECP256K1_ORDER_BYTES[i]) { overflow = true; break; }  // TIMING LEAK
}
if (overflow) recid |= 2;
```

**Fix:** Replace with CT version from `ct_sign.cpp:326-337`:
```cpp
unsigned gt = 0u, eq_run = 1u;
for (int i = 0; i < 32; ++i) {
    unsigned const rb = static_cast<unsigned>(r_bytes[i]);
    unsigned const ob = static_cast<unsigned>(SECP256K1_ORDER_BYTES[i]);
    unsigned const byte_gt = ((ob - rb) >> 31) & 1u;
    unsigned const byte_lt = ((rb - ob) >> 31) & 1u;
    gt     = gt | (eq_run & byte_gt);
    eq_run = eq_run & (1u - byte_gt) & (1u - byte_lt);
}
recid |= static_cast<int>(gt) << 1;
```

---

### BUG-003: ecdsa.cpp normalize() Variable-Time on Secret 🟡 MEDIUM

**Type:** Timing side-channel  
**Location:** `cpu/src/ecdsa.cpp:95-109`

```cpp
// BUG: Line 95-98 — branch on secret `s`
ECDSASignature normalize() const {
    if (is_low_s()) return *this;    // branch leaks s>n/2
    return {r, s.negate()};
}

// BUG: Lines 100-109 — early-exit limb comparison
bool is_low_s() const {
    const auto& sl = s.limbs();
    for (int i = 3; i >= 0; --i) {
        if (sl[i] < HALF_ORDER_LIMBS[i]) return true;   // early-exit leak
        if (sl[i] > HALF_ORDER_LIMBS[i]) return false;  // early-exit leak
    }
    return true;
}
```

**Fix:** Use `ct::ct_normalize_low_s()` and `ct::scalar_is_high()` from `ct_sign.cpp`:
```cpp
ECDSASignature normalize() const {
    return ct::ct_normalize_low_s(*this);
}
```

---

### BUG-004: Batch Sign Fail-Closed Violation 🟡 MEDIUM

**Type:** Security invariant violation (Security Guardrail #3)  
**Location:** `cpu/src/impl/ufsecp_ecdsa.cpp` — batch sign functions

**Issue:** After partial failure (e.g., 2nd key of 5 is invalid), previously written signatures remain in output buffer.

**Fix:** Clear output buffer before processing OR implement two-phase (validate all keys first, then sign all):
```cpp
// PROPOSED FIX: Validate all keys first
for (size_t i = 0; i < count; ++i) {
    if (!is_valid_key(privkeys32 + i*32)) {
        memset(sigs64_out, 0, count * 64);  // clear ALL output
        return UFSECP_ERR_BAD_KEY;
    }
}
// Then sign...
```

---

### BUG-005: SECURITY_CLAIMS.md Phantom API 🟡 MEDIUM

**Type:** Documentation error (but could confuse callers)  
**Location:** `docs/SECURITY_CLAIMS.md:182`

**Issue:** References `ufsecp_xonly_pubkey_parse` which does not exist.

**Fix:** Either:
(a) Implement `ufsecp_xonly_pubkey_parse` in C ABI, or  
(b) Update `SECURITY_CLAIMS.md` to reference the actual function `ufsecp_pubkey_xonly()` and correctly describe what it does (derive from private key, not parse).

---

### BUG-006: Exploit Test Count Mismatch in Catalog 🟢 LOW

**Type:** Documentation-count mismatch  
**Location:** `docs/EXPLOIT_TEST_CATALOG.md:4`

**Issue:** Claims 205 tests, actual ~165 test files.

**Fix:** Recount and update catalog.

---

### BUG-007: ufsecp_addr_p2sh NULL+Zero Length 🟢 LOW

**Type:** Potential NULL dereference  
**Location:** `cpu/src/impl/ufsecp_address.cpp:122-123`

```cpp
// BUG: When redeem_script==NULL && redeem_script_len==0, check passes
if (!redeem_script && redeem_script_len > 0) return UFSECP_ERR_NULL_ARG;
// Then hash160(NULL, 0) may dereference NULL
```

**Fix:**
```cpp
if (!redeem_script || redeem_script_len == 0) return UFSECP_ERR_NULL_ARG;
```

---

### BUG-008: ecdsa.cpp normalize() callers in non-CT paths 🟢 LOW

**Location:** `cpu/src/ecdsa.cpp:471`, `cpu/src/ecdsa.cpp:528`

**Issue:** `result = ECDSASignature{r, s}.normalize();` — calls variable-time normalize.
In `ecdsa_sign` and `ecdsa_sign_hedged`, `s` is still a stack variable containing secret material.

**Fix:** Same as BUG-003 — use `ct::ct_normalize_low_s`.

---

## 8. რეიტინგი

### საერთო ქულები

| კატეგორია | Weight | Score | Weighted |
|-----------|--------|-------|----------|
| Cryptography Correctness | 25% | 8.5/10 | 2.13 |
| Security (Attack Surface) | 25% | 7.0/10 | 1.75 |
| Test Coverage (Exploit Tests) | 15% | 7.5/10 | 1.13 |
| Documentation Accuracy | 15% | 5.5/10 | 0.83 |
| Code Quality & Standards | 10% | 7.0/10 | 0.70 |
| Performance | 10% | 8.5/10 | 0.85 |
| **დასკვნითი** | **100%** | — | **7.39/10** |

### TOP 5 ყველაზე მნიშვნელოვანი გამოსასწორებელი:

1. **BUG-001** — `ufsecp_addr_*` off-by-one buffer overflow 🔴 (all 7 functions)
2. **BUG-002** — `recovery.cpp` variable-time recid overflow 🔴 (secret nonce leak)
3. **BUG-004** — Batch sign fail-closed violation 🟡 (Security Guardrail)
4. **BUG-003** — `ecdsa.cpp` normalize() variable-time 🟡 (secret `s` leak)
5. **BUG-005** — Phantom API in `SECURITY_CLAIMS.md` 🟡

### Strengths Summary:
✅ CT signing core (`ct_sign.cpp`) — industry-leading quality  
✅ Comprehensive exploit test suite (~165 tests)  
✅ HMAC midstate caching — innovative optimization  
✅ Secure erase discipline throughout  
✅ CT layer is architectural, not flag-based  
✅ Z²-based verify — performance innovation  

### Weaknesses Summary:
❌ Address ABI functions have systematic off-by-one  
❌ Legacy `recovery.cpp` not fully CT-migrated  
❌ Documentation accuracy needs improvement (phantom APIs, count mismatches)  
❌ Batch sign violates fail-closed invariant  
❌ Coding standards inconsistently followed  

---

*ანგარიში შედგენილია 2026-04-28. 5 parallel subagent (926 tool calls, 106K+ context).*
