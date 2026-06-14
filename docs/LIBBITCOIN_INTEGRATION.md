# libbitcoin ⇄ UltrafastSecp256k1 — Integration Guide

How to use UltrafastSecp256k1 as the secp256k1 backend in libbitcoin (or any node):
what to do, how, and what is required. See also [SIGNATURE_FORMS.md](SIGNATURE_FORMS.md)
(forms + entry points) and `compat/libbitcoin_bridge/README.md` (bridge API).

---

## 0. Assessment of the two integration assumptions

> **(1) "Drop-in use of ufsecp works directly over the secp interface."** ✅ **Correct.**
> The libsecp256k1 compatibility shim (`secp256k1_*`) is a 1:1 drop-in replacement. Parse,
> normalize, sign, verify, serialize all behave like stock libsecp256k1 — no extra shim, no
> code changes. (The one historical break — the opaque `secp256k1_ecdsa_signature` byte layout
> — is fixed; see §6.)
>
> **(2) "Produce 64-byte compact low-S via `secp256k1_ecdsa_signature_serialize_compact` and
> pass that to the batch interface."** ⚠️ **You do not need to, and it costs more, not less.**
> The default batch row verifier `ufsecp_lbtc_verify_ecdsa` (== `_opaque`) consumes the **opaque**
> `ec_signature` bytes directly: the opaque LE form *is* the engine's native scalar-limb layout, so
> each `r`/`s` is read with a direct little-endian limb load — **no byte-reversal and no
> per-signature copy** (`serialize_compact` would *add* a bswap + a copy). Feeding compact bytes to
> the opaque verifier mismatches the form (it reads them as LE limbs) and fails; if you genuinely
> store big-endian compact, call **`ufsecp_lbtc_verify_ecdsa_compact`** (same 8-arg shape). **Best:**
> pass `ec_signature` **unchanged** to `ufsecp_lbtc_verify_ecdsa` — that is the true no-copy path,
> and the engine's in-register low-S normalize (a conditional negate, consensus-required — see §4.1)
> is free.

**Net:** the integration needs **no code change** on your side. `ufsecp_lbtc_verify_ecdsa` /
`_schnorr` keep the exact 8-argument `ufsecp_error_t(... , results, invalid_idx, invalid_cap,
invalid_count)` shape your `batch_verify` already calls — the engine API now matches your
structure. Pick `_opaque` (default) or `_compact` for ECDSA per how you store signatures.

---

## 1. Architecture — two layers

| Layer | API | Role | Opt-in |
|-------|-----|------|--------|
| **1. Drop-in shim** | `secp256k1_*` (libsecp256k1 ABI) | 1:1 replacement for libsecp256k1 | always |
| **2. Batch/GPU bridge** | `ufsecp_lbtc_*` | additive accelerator for batch script-sig verify (ECDSA/Schnorr) on CPU threads or GPU | `WITH_ULTRAFAST` |

Layer 2 is **additive**: with `WITH_ULTRAFAST` off, `ecdsa::batch_verify` / `schnorr::batch_verify`
fall back to parallel single `secp256k1_ecdsa_verify` (layer 1). The GPU result is consensus-anchored
to the CPU/libsecp reference (`test_lbtc_consensus_diff`).

---

## 2. Build wiring (CMake) — already correct in libbitcoin

```cmake
option(with-ultrafast "Use UltrafastSecp256k1 ..." OFF)
if(with-ultrafast)
  # deps/UltrafastSecp256k1 (submodule) or -DUltrafastSecp256k1_DIR=<path>
  set(SECP256K1_BUILD_LIBBITCOIN ON CACHE BOOL "" FORCE)
  add_subdirectory("${UltrafastSecp256k1_DIR}" ultrafast_secp256k1 EXCLUDE_FROM_ALL)
  add_compile_definitions(WITH_ULTRAFAST)
  include_directories("${UltrafastSecp256k1_DIR}/compat/libbitcoin_bridge/include")
endif()
```
`SECP256K1_BUILD_LIBBITCOIN=ON` is the minimal node profile (shim + bridge + BIP-352, extras off).

---

## 3. Layer 1 — drop-in secp public API (single verify)

No changes. The opaque `ec_signature` is `secp256k1_ecdsa_signature.data`; parse DER into it and
verify:
```cpp
secp256k1_ecdsa_signature sig;
ecdsa_signature_parse_der_lax(ctx, &sig, der, derlen);    // your parser (uses parse_compact)
secp256k1_ecdsa_signature_normalize(ctx, &sig, &sig);     // Bitcoin accepts high-S; normalize
secp256k1_ecdsa_verify(ctx, &sig, hash, &pubkey);         // (ufsecp shim accepts high-S anyway)
```
The opaque `ec_signature` bytes are **identical** to stock libsecp256k1 (little-endian internal
scalar limbs) — you may copy them across the boundary unchanged.

---

## 4. Layer 2 — batch bridge (`ufsecp_lbtc_*`)

### 4.1 Signature form — pass the OPAQUE `ec_signature`, zero-copy (DO THIS)

The ECDSA row/columns batch verifiers take the **opaque** form (your `ec_signature` bytes). Your
packed `triple` span *is* the bridge row buffer:

```
ECDSA row   : digest(32) | pubkey(33) | ec_signature(64, opaque) | token(key_size)
Schnorr row : digest(32) | x-only(32) | signature(64, BIP-340)   | token(key_size)
```
- **No `serialize_compact`. No `normalize`. No per-signature copy.** The opaque `ec_signature`
  (libsecp's internal little-endian scalar limbs) **is the engine's native scalar layout**: the
  ECDSA opaque verifier parses each `s`/`r` with a direct little-endian limb load
  (`opaque_scalar_parse_strict_nonzero` → `Scalar::from_limbs`) — **not** a byte-reversal. This is
  the unavoidable bytes→scalar parse every secp256k1 backend does; the opaque (LE) form is if
  anything *cheaper* than the public big-endian compact form, which is the one that needs a bswap.
- The high-S → low-S normalization is an **in-register conditional scalar negate** (`s ← n−s` iff
  `s > n/2`), fused into that parse — not a copy of the signature. It is **consensus-required**, not
  a "shim hack": Bitcoin consensus accepts high-S (historical blocks), and libbitcoin's own single
  `verify_signature` already calls `secp256k1_ecdsa_signature_normalize` for exactly this reason;
  the batch path normalizes so it agrees with that single path. Doing it in-register here is the
  copy-free choice — normalizing on the libbitcoin side would mean mutating your `const` zero-copy
  rows (a copy).
- Pass `ec_signature` exactly as `parse_der_lax` produced it. The trailing `token` (`key_size`
  bytes) is your opaque per-row id; the engine never interprets it.

### 4.2 The API signature — matches your `batch_verify`, no change required

```c
/* The engine API matches libbitcoin's existing call exactly. Returns UFSECP_OK for
 * any well-formed batch (per-row failures are verdicts, not errors); a non-OK return
 * is an unrecoverable fault. results / invalid_idx / invalid_count are all optional. */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa(           /* == _ecdsa_opaque (default form) */
    ufsecp_lbtc_ctrl* ctrl, const uint8_t* rows, size_t n, size_t key_size,
    uint8_t* results, size_t* invalid_idx, size_t invalid_cap, size_t* invalid_count);
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_opaque (...same args...);  /* opaque ec_signature */
ufsecp_error_t ufsecp_lbtc_verify_ecdsa_compact(...same args...);  /* big-endian compact r||s */
ufsecp_error_t ufsecp_lbtc_verify_schnorr(...same args...);        /* BIP-340 (single form) */
```
```cpp
// Your existing call compiles + runs unchanged:
static thread_local ufsecp::lbtc::Controller control{ UFSECP_LBTC_AUTO };
if (!control.ok()) std::abort();
const auto status = ufsecp_lbtc_verify_ecdsa(control.get(),
    reinterpret_cast<const uint8_t*>(rows.data()), count,
    sizeof(triple::token), results.data(), nullptr, 0, nullptr);   // or &invalid_count
if (status != UFSECP_OK) std::abort();                             // unrecoverable fault only
```
- `results` (optional): `results[i]` = 1 valid / 0 invalid.
- `invalid_idx` + `invalid_cap` (optional): indices of failing rows, up to `invalid_cap`.
- `invalid_count` (optional): total failing rows (all valid iff `*invalid_count == 0`; may exceed
  `invalid_cap`, so you can detect truncation — handy to skip the per-row scan when zero).

> The engine was aligned to this exact 8-argument shape so your `ecdsa::batch_verify` /
> `schnorr::batch_verify` need no edits. Choose `_opaque` (default) or `_compact` for ECDSA.

Variants (same shape): `ufsecp_lbtc_verify_ecdsa_collect` / `_columns` / `_columns_collect` (and the
Schnorr equivalents). `*_collect` writes the verdict into each row's key cell instead of `results[]`.

### 4.3 If you prefer compact rows or columns

- **Packed rows, compact:** `ufsecp_lbtc_verify_ecdsa_compact(ctrl, rows, n, key_size, results,
  invalid_idx, invalid_cap, invalid_count)` — same row layout, but the 64-byte signature field holds
  **public big-endian compact r||s**. Use `secp256k1_ecdsa_signature_serialize_compact` to fill it.
- **Columns, compact:** `ufsecp_lbtc_verify_ecdsa_columns_compact(ctrl, msg32, pub33, sigs64_compact,
  n, results)` (separate msg/pub/sig arrays).
Both are correctness-equivalent to the opaque path; opaque is the zero-copy default.

---

## 5. What libbitcoin must do — checklist

1. **Bump `deps/UltrafastSecp256k1`** to current `dev` (opaque-layout fix, bridge high-S
   normalization, the 8-arg `_opaque`/`_compact` row verifiers, `_columns_compact`, `ecdsa_sig_pack`,
   and these docs).
2. **Batch call sites:** **no change** — `ufsecp_lbtc_verify_ecdsa` / `_schnorr` already match your
   8-argument `ufsecp_error_t` call (§4.2). Optionally switch to `_opaque`/`_compact` for explicitness.
3. **Keep passing the opaque `ec_signature`** in rows, zero-copy (the default `ufsecp_lbtc_verify_ecdsa`
   == `_opaque`); the bridge normalizes high-S. Use `_compact` only if you store compact sigs.
4. **Single verify / sign / parse:** unchanged (layer 1 drop-in).

That is the entire integration. No code change, no additional shim, no extra copies.

---

## 6. Why the earlier validation failures happened (closed)

Both were engine bugs, now fixed — not libbitcoin wiring:
- **`df724f4d`** — the shim stored `secp256k1_ecdsa_signature.data` as big-endian compact instead of
  libsecp's little-endian internal limbs, so libbitcoin saw `ec_signature`/DER byte-reversed. (This is
  the "compatibility break between secp and ufsecp" evoskuil identified — a real drop-in violation,
  now fixed and pinned by `shim_test.cpp` + the libbitcoin scenario-3 test.)
- **`acb7fefe` / `91ae9baf` / `836ae25c`** — the bridge batch path did not low-S normalize high-S
  signatures, so high-S sigs passed single-verify (which normalizes) but failed batch-verify (the 3
  batch cases). The bridge now normalizes high-S on CPU and GPU.

---

## 7. Engine API status

The API now fully matches libbitcoin's structure — nothing is missing for the integration:

- **8-arg `ufsecp_error_t` row verifiers** (`ufsecp_lbtc_verify_ecdsa` / `_schnorr`, with
  `results` + `invalid_idx`/`invalid_cap`/`invalid_count`) — matches your `batch_verify` call. ✅
- **Explicit `_opaque` / `_compact` ECDSA row verifiers** — pick the form by name. ✅
- **Packed-row compact verify** (`_compact`) and **columns compact** (`_columns_compact`). ✅
- Verified CPU + CUDA, with the GPU↔CPU consensus differential (`test_lbtc_consensus_diff`) green,
  including high-S rows and the `invalid_idx`/`invalid_count` reporting.
