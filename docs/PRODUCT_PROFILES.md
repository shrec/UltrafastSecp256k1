# UltrafastSecp256k1 — Product Profiles and Surface Taxonomy

**Version:** 1.0 · **Last Updated:** 2026-05-01

## Profiles

| Profile ID | Description | CAAS Runner |
|---|---|---|
| `bitcoin-core-backend` | CPU+shim for Bitcoin Core secondary backend | ✅ implemented |
| `cpu-signing` | CPU ECDSA/Schnorr/CT layer standalone | ✅ implemented |
| `ffi-bindings` | Legacy C API + language bindings | ✅ implemented |
| `wasm` | WebAssembly browser/Node binding | ✅ implemented |
| `gpu-public-data` | GPU batch verify + BIP-352 scan (public data) | ✅ implemented |
| `bchn-compat` | Bitcoin Cash Node legacy Schnorr shim | ✅ implemented |
| `release/full-engine` | All surfaces, release gate | ✅ implemented |

## Tiers

| Tier | Meaning |
|---|---|
| `production` | Full CT, CAAS-hard-gated, production-safe |
| `beta` | Feature-complete, CAAS partial, promotion depends on evidence closure |
| `experimental` | CAAS evidence incomplete; not for production secrets |
| `compat-only` | Backward-compat only; do not use in new code |
| `deprecated` | Scheduled for removal |

## Surface Inventory

### 1. Canonical CPU C ABI (`ufsecp_*`)
- **Profile:** `bitcoin-core-backend`, `cpu-signing`, `release/full-engine`
- **Tier:** `production`
- **Files:** `src/cpu/src/impl/ufsecp_*.cpp`
- **CT:** All secret-bearing paths → `secp256k1::ct::*`
- **CAAS:** audit_gate + security_autonomy + bundle_verify (all hard)

### 2. libsecp256k1 Compatibility Shim
- **Profile:** `bitcoin-core-backend`
- **Tier:** `production`
- **Files:** `compat/libsecp256k1_shim/src/`
- **CT:** Routes through `secp256k1::ct::*` as of 2026-04-28/2026-05-01
- **Context flags:** Enforced (SIGN/VERIFY) as of 2026-05-01

### 3. Public C++ API (`secp256k1::ecdsa_sign`, `schnorr_sign`)
- **Profile:** `cpu-signing`
- **Tier:** `production`
- **Files:** `src/cpu/include/secp256k1/ecdsa.hpp`, `schnorr.hpp`, `recovery.hpp`
- **CT:** `signing_generator_mul()` aliases `ct::generator_mul_blinded()` — CT internally
- **Note:** For new code, prefer the canonical `ufsecp_*` ABI or `secp256k1::ct::*` directly

### 4. Legacy C API
- **Profile:** `ffi-bindings`
- **Tier:** `beta`
- **Files:** `bindings/c_api/ultrafast_secp256k1.cpp`
- **CT:** Routes through `secp256k1::ct::*` as of 2026-05-01. Strict key parsing.

### 5. Language Bindings (Node.js, Python, Ruby, Go, Swift, Dart)
- **Profile:** `ffi-bindings`
- **Tier:** `beta`
- **Files:** `bindings/nodejs/`, `bindings/python/`, etc.
- **CT:** Inherit security from legacy C API

### 6. WASM Binding
- **Profile:** `wasm`
- **Tier:** `experimental`
- **Files:** `bindings/wasm/`
- **CT:** Prebuilt artifact — WASM-specific CT evidence is not yet sufficient for production-CT claims
- **Restriction:** Do not claim WASM production-CT without CI rebuild + timing analysis

### 7. GPU Backend — Public Data
- **Profile:** `gpu-public-data`
- **Tier:** `beta` (verify/scan), `experimental` (GPU signing)
- **Files:** `src/gpu/`, `src/cuda/`, `src/opencl/`, `src/metal/`
- **CT:** GPU signing CT paths added 2026-05-01; key erasure added 2026-05-01
- **Restriction:** GPU signing tier is `experimental` pending timing analysis

### 8. BCHN Compatibility Shim
- **Profile:** `bchn-compat`
- **Tier:** `compat-only`
- **Files:** `compat/libsecp256k1_bchn_shim/`
- **CT:** CT generator mul added 2026-05-01; strict key parsing added 2026-05-01
- **Note:** NOT Bitcoin Core profile. NOT BIP-340.

### 9. MuSig2
- **Profile:** `cpu-signing`, `ffi-bindings`
- **Tier:** `beta`
- **CT:** `partial_sign` uses `ct::scalar_mul` as of 2026-05-01

### 10. FROST
- **Profile:** `cpu-signing` (via `release`)
- **Tier:** `beta`
- **CT:** Covered by shared CAAS CT gates; feature-specific protocol evidence is still maturing

## Claims Wording Policy

### Correct
- "The canonical `ufsecp_*` ABI and libsecp256k1 shim route all secret-bearing signing paths through `secp256k1::ct::*`."
- "The `bitcoin-core-backend` profile covers CPU signing and libsecp256k1 shim only. GPU, FFI, WASM, and BCHN are separate profiles."
- "GPU batch verification operates on public data only — no CT requirement."
- "The legacy C API uses CT signing as of 2026-05-01."

### Prohibited
- "Everything in this repository is constant-time safe."
- "The full repository is production-ready."
- "All bindings are as secure as the CPU layer."
- "GPU signing is production-safe." (use `experimental` tier)
- "Benchmarks prove performance." (unless CI-artifact and fresh)
