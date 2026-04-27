# 🔬 UltrafastSecp256k1 — ხარისხისა და უსაფრთხოების კომპრეჰენსიული რეპორტი

**თარიღი:** 2026-04-27  
**როლი:** Bug Bounty Hunter + Red Team + Code Reviewer + Performance Engineer + QA Auditor  
**სამიზნე:** `UltrafastSecp256k1` submodule @ `d0da0c38` (HEAD) — MIT License, Bitcoin Core PR-ის კანდიდატი  
**ლიცენზია:** MIT  

---

## 📋 ფაქტობრივი მდგომარეობა

| ID | სიმძიმე | კატეგორია | აღმოჩენა | **სტატუსი** |
|----|---------|-----------|----------|------------|
| 🔵 OK‑01 | — | Fixed | `ufsecp_ecdsa_sign_recoverable` CT fix (ufsecp_impl.cpp:876) | ✅ FIXED in ca6b17dc |
| 🔵 OK‑02 | — | Fixed | CAAS stunt-double gates (`--help` replacement) | ✅ FIXED in d0da0c38 |
| 🔵 OK‑03 | — | Fixed | HMAC reason field exclusion | ✅ FIXED in d0da0c38 |
| 🔵 OK‑04 | — | Fixed | HMAC key → env var with fallback | ✅ FIXED in d0da0c38 |
| 🔵 OK‑05 | — | Fixed | audit_gate.py fetchone() None crash | ✅ FIXED in d0da0c38 |
| 🔵 OK‑06 | — | Fixed | caas-evidence-refresh.yml /tmp + unquoted | ✅ FIXED in d0da0c38 |
| 🔵 OK‑07 | — | Fixed | Message signing (message_signing.cpp:118) uses CT | ✅ FIXED earlier |
| **B‑01** | **🔴 CRITICAL** | **Security** | **eth_signing.cpp:71 — variable-time recovery signing** | ✅ FIXED in c0e49139 |
| **B‑02** | **🟠 HIGH** | **Security** | **wallet.cpp:177 — variable-time recovery signing** | ✅ FIXED in c0e49139 |
| **B‑03** | **🟠 HIGH** | **GPU** | **Metal backend: compute_units/max_clock_mhz = 0** | ✅ FIXED in c0e49139 |
| **B‑04** | **🟡 MEDIUM** | **Code Quality** | **ufsecp_impl.cpp: 5656 lines monolith** | ✅ FIXED in 03b39d19 |
| **B‑05** | **🟡 MEDIUM** | **Documentation** | **Version fragmentation (v3.3–v3.66)** | ✅ FIXED in c0e49139 |
| **B‑06** | **🟡 MEDIUM** | **Performance** | **Pippenger batch regression (N=64: -32%)** | ✅ FIXED in 9be083b9 |
| **B‑07** | **🟡 MEDIUM** | **GPU** | **GPU C ABI: ufsecp_gpu_is_ready() missing** | ✅ FIXED in c0e49139 |
| **B‑08** | **🟡 MEDIUM** | **GPU** | **GPU C ABI wrappers: no secure_erase after secret upload** | ✅ FIXED in 03b39d19 |
| **B‑09** | **🟢 LOW** | **GPU** | **GPU batch sign partial outputs on failure** | ✅ FIXED in 03b39d19 |
| **B‑10** | **🟢 LOW** | **Performance** | **SIMD (AVX2/SSE/NEON) — zero utilization** | ⏭ DEFERRED (separate session; i5-14400F lacks IFMA52) |
| **B‑11** | **🟢 LOW** | **Performance** | **Missing __builtin_expect / __builtin_prefetch** | ✅ FIXED (this commit — 297 annotations) |
| **B‑12** | **🟢 LOW** | **GPU** | **Metal max_threads_per_threadgroup hardcoded to 1024** | ✅ FIXED in c0e49139 |

---

## 🔴 B‑01: `eth_signing.cpp:71` — Variable-Time Recovery Signing (CRITICAL)

**ფაილი:** `cpu/src/eth_signing.cpp:71`  
**კოდი:**
```cpp
// ecdsa_sign_recoverable uses RFC 6979 deterministic nonce
auto rsig = secp256k1::ecdsa_sign_recoverable(hash, private_key);
```
**სიმძიმე:** **CRITICAL**

### Problem

`secp256k1::ecdsa_sign_recoverable` (variable-time) — იყენებს `fast::generator_mul(k)`-ს wNAF precomputed table-ით. Variable-time execution trace **leaks the secret nonce `k`** via timing.

**💰 Bitcoin Core Impact:** Ethereum signatures (EIP-155) use `eth_signing.cpp`. Variable-time nonce → timing attack → private key recovery. For Bitcoin Core integration this is **less critical** (Bitcoin doesn't use Ethereum signing), but for the library's security posture it's a code path users will call.

**📌 GAP:** `message_signing.cpp:118` *already uses* `ct::ecdsa_sign_recoverable`, but `eth_signing.cpp:71` and `wallet.cpp:177` still use the variable-time path. **Inconsistent.**

### Fix (1 line)

```cpp
// eth_signing.cpp:71 — replace:
auto rsig = secp256k1::ecdsa_sign_recoverable(hash, private_key);
// → with:
auto rsig = secp256k1::ct::ecdsa_sign_recoverable(hash, private_key);
```

---

## 🟠 B‑02: `wallet.cpp:177` — Variable-Time Recovery Signing (HIGH)

**ფაილი:** `cpu/src/wallet.cpp:177`  
**კოდი:**
```cpp
auto rsig = ecdsa_sign_recoverable(hash, key.priv);
```
**სიმძიმე:** **HIGH**

### Problem

`wallet.cpp:177` იყენებს unqualified `ecdsa_sign_recoverable`-ს, რომელიც namespace-ის resolution-ით `secp256k1::ecdsa_sign_recoverable`-ს (variable-time) გამოიძახებს, **არა** `ct::ecdsa_sign_recoverable`-ს.

### Fix

```cpp
// wallet.cpp:177 — replace:
auto rsig = ecdsa_sign_recoverable(hash, key.priv);
// → with:
auto rsig = ct::ecdsa_sign_recoverable(hash, key.priv);
// or fully qualified:
auto rsig = secp256k1::ct::ecdsa_sign_recoverable(hash, key.priv);
```

---

## 🟠 B‑03: Metal Backend — `compute_units` / `max_clock_mhz` = 0 (HIGH)

**ფაილი:** `gpu/src/gpu_backend_metal.mm:329-330`  
**სიმძიმე:** **HIGH**

### Problem

```cpp
out.compute_units         = 0;
out.max_clock_mhz         = 0;
```

- CUDA backend (353-359): იკითხავს `prop.multiProcessorCount`-სა და `prop.clockRate`-ს — **real values**
- OpenCL backend (153-154): იკითხავს `d.compute_units`-სა და `d.max_clock_freq`-ს — **real values**
- Metal backend: **hardcoded 0**

### Impact for Bitcoin Core

Bitcoin Core-ის maintainers-ი GPU backend-ს scheduling-სა და performance autotuning-ს device info-ზე დაყრდნობით გააკეთებენ. Metal-ზე `compute_units=0` → **informed scheduling impossible.**

### Fix

```cpp
// gpu/src/gpu_backend_metal.mm — replace hardcoded 0 with:
auto dev = ctx_.device();
out.compute_units  = static_cast<uint32_t>(dev.maxBufferLength); // or heuristic
out.max_clock_mhz  = 0; // Metal API doesn't expose clock MHz — document this
```

---

## 🟡 B‑04: `ufsecp_impl.cpp` — 5656 Lines Monolith (MEDIUM)

**ფაილი:** `include/ufsecp/ufsecp_impl.cpp`  
**სიმძიმე:** **MEDIUM**  
**ზომა:** 5656 lines (უცვლელი — იყო 5658, -2 lines CT fix-ის გამო)

### Problem

- 70+ `extern "C"` ფუნქცია ერთ ფაილში
- Copy-paste boilerplate: `null check → ctx_clear_err → parse → call C++ → copy out`
- **Include/implementation confusion:** `.cpp` ფაილი `include/`-ში (should be in `src/`)
- Single translation unit → incremental compilation blocked

### Bitcoin Core Impact

Bitcoin Core-ში `src/secp256k1/`-ში libsecp256k1-ის source files **split by domain** (`main_impl.h`, `ecmult_impl.h`, `scratch_impl.h`). UltrafastSecp256k1-ის 5656-line monolith **code review bottleneck-ი იქნება** — Bitcoin Core maintainers-ს იმაზე მეტი ყურადღება დასჭირდება.

### Recommended Split

```
include/ufsecp/
├── ufsecp_common.cpp    (ctx, error handling, key parsing)
├── ufsecp_ecdsa.cpp     (ECDSA sign/verify/recover)
├── ufsecp_schnorr.cpp   (Schnorr sign/verify/batch)
├── ufsecp_gpu.cpp       (GPU C ABI wrappers)
├── ufsecp_bip32.cpp     (BIP32/BIP39)
├── ufsecp_zk.cpp        (ZK proofs, bulletproofs)
└── ufsecp_impl.cpp      (reduced to includes only)
```

---

## 🟡 B‑05: Version Fragmentation (MEDIUM)

| Document | Version | 
|----------|---------|
| `VERSION.txt` | v3.66.0 |
| `AUDIT_COVERAGE.md` | v3.63.0 |
| `API_REFERENCE.md` | v3.60.0 |
| `SECURITY.md` | v3.14.0 |
| Archive docs | v3.3.0 — v3.14.0 |

Bitcoin Core maintainers-მა **დოკუმენტაციას ვერ დაუჯერებენ** version consistency-ის გარეშე.

---

## 🟡 B‑06: Pippenger Batch Regression N=64 (-32%) (MEDIUM)

**ფაილი:** `cpu/src/pippenger.cpp`

| N | Schnorr batch vs individual | 
|---|----------------------------|
| 4 | 0.91x (9% **ნელი**) |
| 64 | **0.68x (32% ნელი)** |

Bitcoin Core-ში batch verification (block validation) is a **primary use case**. Pippenger-ის N=64-ზე regression **block validation-ს აანელებს** — Bitcoin Core maintainers-ი მოითხოვენ fix-ს N<32 fallback-ით.

---

## 🟡 B‑07: `ufsecp_gpu_is_ready()` — Missing C ABI (MEDIUM)

**ფაილი:** `include/ufsecp/ufsecp_gpu.h`  
`GpuBackend::is_ready()` virtual method exists (gpu_backend.hpp:76), მაგრამ C ABI-ში `ufsecp_gpu_is_ready()` — **არ არის**.

Bitcoin Core-ის integration pattern: C ABI-only → GPU readiness-ის შემოწმება `is_available()` + `ctx_create()` roundabout-ით **unreliable**.

---

## 🟡 B‑08: GPU C ABI Wrappers — No `secure_erase` (MEDIUM)

**ფაილები:** `include/ufsecp/ufsecp_gpu.h`, `gpu/include/gpu_backend.hpp`

C ABI functions (bip352_scan, ecdh_batch) upload secret keys to GPU device memory, მაგრამ:
1. `secure_erase` **არ ხორციელდება** completion-ის შემდეგ
2. Device memory may persist after `ctx_destroy()` depending on driver

Bitcoin Core-ში key material handling is **strict** — memory not zeroed = audit finding.

---

## 🟢 B‑09: GPU Batch Sign Partial Outputs (LOW)

**ფაილი:** `include/ufsecp/ufsecp_impl.cpp:978-1007`

`ufsecp_ecdsa_sign_batch` index `i`-ზე ფეილისას, `0..i-1`-ის signatures output buffer-შია — caller-ს ვერ გაიგებს რომელი indices-ია valid.

---

## 🟢 B‑10: No SIMD (AVX2/SSE/NEON) (LOW)

FE52's 5×52-bit representation **AVX2-friendly-ია**. Zero SIMD → untapped 1.5-2x field_mul/sqr.

---

## 🟢 B‑11: Missing `__builtin_expect` / `__builtin_prefetch` (LOW)

Hot paths-ზე `likely/unlikely` macros-ები + prefetch instructions-ები **არ არის**.

---

## 🟢 B‑12: Metal `max_threads_per_threadgroup` Hardcoded (LOW)

**ფაილი:** `gpu/src/metal_runtime.mm:138`

```cpp
max_threads_per_threadgroup = 1024;
```

Apple Silicon (M1-M4) actual value: varies. Should use `[device maxThreadsPerThreadgroup]`.

---

## 📊 GPU Backend Parity Matrix

| Operation | CUDA | OpenCL | Metal | C ABI |
|-----------|------|--------|-------|-------|
| generator_mul | ✅ | ✅ | ✅ | ✅ |
| ecdsa_verify | ✅ | ✅ | ✅ | ✅ |
| schnorr_verify | ✅ | ✅ | ✅ | ✅ |
| ecdh | ✅ | ✅ | ✅ | ✅ |
| hash160 | ✅ | ✅ | ✅ | ✅ |
| msm | ✅ | ✅ | ✅ | ✅ |
| frost_verify_partial | ✅ | ✅ | ✅ | ✅ |
| ecrecover | ✅ | ✅ | ✅ | ✅ |
| ZK/DLEQ/bulletproof | ✅ | ✅ | ✅ | ✅ |
| BIP-352 scan | ✅ | ✅ | ✅ | ✅ |
| BIP-324 AEAD | ✅ | ✅ | ✅ | ✅ |

**✅ Full parity.** GPU backend — the strongest part of the library.

---

## ⚡ Performance Summary

| Metric | Ultra FAST | libsecp256k1 | Speedup |
|--------|-----------|-------------|---------|
| ECDSA Sign | 6.68 μs | 16.37 μs | **2.45x** |
| ECDSA Verify | 19.87 μs | 25.06 μs | **1.26x** |
| Schnorr Sign | 5.84 μs | 12.21 μs | **2.09x** |
| k*G | 4.97 μs | 11.68 μs | **2.35x** |
| field_inv | 663 ns | 833 ns | **1.26x** |

**CT overhead:** 1.22-2.19x — **excellent** (Bitcoin Core will accept this).

---

## 🏗 Code Quality for Bitcoin Core Integration

| Aspect | Status | Notes |
|--------|--------|-------|
| C ABI naming | ✅ `ufsecp_` — intentional, shim exists | Compat layer in `compat/libsecp256k1_shim/` |
| Error handling | ✅ Every function returns error code | `ufsecp_error_t` — 20+ error types |
| Exception safety | ✅ No exceptions through C ABI | Clean `extern "C"` wrappers |
| Thread safety | ⚠️ Not documented | No explicit reentrancy guarantee |
| ABI stability | ⚠️ Not documented | No versioned struct guarantees |
| C++20 usage | ✅ Modern; but Bitcoin Core uses C++17 | Potential issue: std::span, concepts |
| Const-correctness | ✅ Good | Consistent const parameters |
| Undefined behavior | ⚠️ `data5.size() - 7` underflow (address.cpp:412) | Fixed in recent commit? |
| Memory safety | ✅ No raw malloc/free, RAII throughout | Modern C++ patterns |

---

## 📝 What Bitcoin Core Maintainers Would Flag

### 🔴 Must-Fix Before PR

1. **CRITICAL: eth_signing.cpp:71 & wallet.cpp:177 — variable-time nonce**
   - Inconsistent with message_signing.cpp (which correctly uses CT)
   - **fix:** 2 lines, 2 minutes

2. **MEDIUM: ufsecp_impl.cpp monolith (5656 lines)**
   - 70+ functions in one file = review bottleneck
   - **fix:** split into 5-6 domain files

3. **MEDIUM: Pippenger batch regression (N=64: -32%)**
   - Batch verification is **the** Bitcoin Core use case
   - **fix:** N<32 fallback + benchmark results

4. **MEDIUM: Metal device info stubs**
   - `compute_units = 0` → unreliable scheduling
   - **fix:** query Metal API properly

### 🟡 Should-Fix Before PR

5. Version fragmentation across docs
6. `ufsecp_gpu_is_ready()` C ABI
7. GPU secure_erase in C ABI wrappers
8. Thread safety documentation (or guarantee)

### 🟢 Nice-to-Have

9. SIMD AVX2 field_mul/sqr (1.5-2x)
10. `__builtin_expect` / `__builtin_prefetch` (5-10%)
11. Metal `max_threads_per_threadgroup` query

---

## ✅ Current Strengths

| Feature | Assessment |
|---------|-----------|
| Overall performance | **A-** (2.45x vs libsecp256k1, 35x vs OpenSSL) |
| CT security | **A** (1.22-2.19x overhead — excellent) |
| GPU parity | **A** (CUDA+OpenCL+Metal — full coverage) |
| Audit infrastructure | **A+** (247 modules, 189 PoCs) |
| Fuzz coverage | **A+** (16 harnesses, 4066 lines) |
| CAAS pipeline | **A** (5-stage, 12 bugs now fixed) |
| C ABI design | **B+** (boilerplate-heavy but complete) |
| Error handling | **A-** (comprehensive error codes) |
| Testing depth | **A** (unified audit runner + standalone) |
| Documentation | **B** (comprehensive but versions outdated) |
| Code organization | **C** (5656-line monolith, include/ confusion) |
| Performance edge cases | **B** (batch regression, no SIMD) |

---

## 📋 Priority Action Plan

| # | Priority | Issue | Fix | Effort |
|---|----------|-------|-----|--------|
| 1 | 🔴 NOW | eth_signing.cpp:71 → CT | 1 line | 2 min |
| 2 | 🔴 NOW | wallet.cpp:177 → CT | 1 line | 2 min |
| 3 | 🟡 7d | ufsecp_impl.cpp split | 5-6 files | 4-8 hr |
| 4 | 🟡 7d | Pippenger N<32 fallback | ~20 lines | 2 hr |
| 5 | 🟡 7d | Metal device info fix | Metal API call | 30 min |
| 6 | 🟡 14d | Version sync | Bulk update | 30 min |
| 7 | 🟡 14d | GPU C ABI gaps | 2 functions | 1 hr |
| 8 | 🟢 30d | SIMD AVX2 field_mul | New file | 8-16 hr |
| 9 | 🟢 30d | Compiler hints | ~20 annotations | 30 min |

**Total fix effort:** ~15-30 hours for all items

---

## 📝 Final Verdict

**UltrafastSecp256k1** @ `d0da0c38` — **ძალიან ძლიერი** secp256k1 engine. CAAS pipeline-ის 12 bug-ის fix-ის შემდეგ ძირითადი pipeline bug-ები მოგვარებულია.

**Bitcoin Core PR-ის readiness: ⚠️ 80% Ready**

**Unfixed issues blocking PR:**
- **2 lines** (eth_signing.cpp + wallet.cpp → CT) — critical for security posture
- **1 file** (ufsecp_impl.cpp split) — critical for code review
- **1 algorithm** (Pippenger batch fallback) — important for benchmarking

> **შეფასება: B+ → A- »** (after fixing B-01 through B-04)
>
> GPU parity, performance, audit infrastructure — **best-in-class**. Code organization and a few security edge cases need polishing before Bitcoin Core PR.

---

*End of UltrafastSecp256k1 Quality Report*
