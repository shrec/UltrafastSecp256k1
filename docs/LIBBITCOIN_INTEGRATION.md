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
> pass that to the batch interface, with no shim and no copies."** ❌ **Incorrect — it would
> re-break verification.** The batch *row* interface (`ufsecp_lbtc_verify_ecdsa`) consumes the
> **opaque** `secp256k1_ecdsa_signature` (`ec_signature`) bytes, **not** public compact. Passing
> `serialize_compact` output there makes the bridge byte-reverse already-big-endian bytes →
> verification fails. **Pass `ec_signature` unchanged** (truly zero-copy); the bridge low-S
> normalizes internally. `serialize_compact` is the right conversion **only** for the separate
> `*_columns_compact` API (§4.3), which you do not need.

**Net:** the integration needs exactly **one** change — update the batch call sites to the
current API signature (§4.2). Everything else (your row layout, the opaque zero-copy form, the
single-verify path) is already correct.

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
- **No `serialize_compact`. No `normalize`.** The bridge low-S normalizes high-S internally on CPU
  **and** GPU. Pass `ec_signature` exactly as `parse_der_lax`/`normalize` produced it.
- The trailing `token` (`key_size` bytes) is your opaque per-row id; the engine never interprets it.

### 4.2 The current API signature — UPDATE YOUR CALL SITES

```c
/* current shipped shape (since 2026-06-01, the shape evoskuil endorsed) */
void ufsecp_lbtc_verify_ecdsa  (ufsecp_lbtc_ctrl* ctrl, const uint8_t* rows,
                                size_t n, size_t key_size, uint8_t* results);
void ufsecp_lbtc_verify_schnorr(ufsecp_lbtc_ctrl* ctrl, const uint8_t* rows,
                                size_t n, size_t key_size, uint8_t* results);
```
```cpp
// C++ wrapper (unchanged — matches libbitcoin's usage):
static thread_local ufsecp::lbtc::Controller control{ UFSECP_LBTC_AUTO };
if (!control.ok()) std::abort();                       // controller-init is the only recoverable error
ufsecp_lbtc_verify_ecdsa(control.get(),
    reinterpret_cast<const uint8_t*>(rows.data()), count,
    sizeof(triple::token), results.data());            // results[i] = 1 valid / 0 invalid
```
> **Migration:** the `batch-verify-triple` / `ultrafast-batch-verify` branches call an older
> **8-argument, `ufsecp_error_t`-returning** form
> (`..., results, nullptr, 0, &invalid_count)` + `std::abort()` on non-OK). That predates the
> 2026-06-01 API and **will not compile** against current ufsecp. Drop to the **5-argument `void`**
> form above: per-row verdicts already arrive in `results[]`, and the typed `std::span<const triple>`
> makes a malformed stride impossible, so there is no recoverable error to check.

Variants (same shape): `ufsecp_lbtc_verify_ecdsa_collect` / `_columns` / `_columns_collect` (and the
Schnorr equivalents). `*_collect` writes the verdict into each row's key cell instead of `results[]`.

### 4.3 If you ever want compact instead of opaque (not needed for your row layout)

`ufsecp_lbtc_verify_ecdsa_columns_compact(ctrl, msg32, pub33, sigs64_compact, n, results)` takes
**public big-endian compact** sigs (column layout). Only *here* is
`secp256k1_ecdsa_signature_serialize_compact` the correct conversion. It is a separate columnar API,
not your packed-row form, and offers no advantage over the opaque zero-copy path.

---

## 5. What libbitcoin must do — checklist

1. **Bump `deps/UltrafastSecp256k1`** to current `dev` (has the opaque-layout fix, the bridge high-S
   normalization, `_columns_compact`, `ecdsa_sig_pack`, and these docs).
2. **Update the batch call sites** (`ecdsa::batch_verify`, `schnorr::batch_verify`) to the 5-arg
   `void` signature (§4.2).
3. **Keep passing the opaque `ec_signature`** in rows, zero-copy — **do not** `serialize_compact`
   for the batch (§4.1).
4. **Single verify / sign / parse:** unchanged (layer 1 drop-in).

That is the entire integration. No additional shim, no extra copies.

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

## 7. Open items on the engine side ("what's missing")

The recommended path (opaque rows + 5-arg `void`) needs **nothing new**. These are optional and only
relevant if libbitcoin prefers not to touch its branch:

- **(Optional) 8-arg `ufsecp_error_t` source-compat overload** of `ufsecp_lbtc_verify_{ecdsa,schnorr}`,
  forwarding to the `void` core, so the pre-2026-06-01 branches build unchanged. evoskuil endorsed the
  `void` shape, so this is a courtesy, not a requirement.
- **(Optional) row-form compact batch verify.** Compact batch is currently **columns-only**
  (`*_columns_compact`); there is no packed-*row* compact verify. Not needed for libbitcoin (it uses
  opaque rows), but would round out the matrix if a future integrator wants packed-row compact.

Neither is required to close the libbitcoin integration.
