# Signature Forms & Verify Entry Points (ECDSA + Schnorr)

This library deliberately supports **several signature byte forms and several verify
entry points** across three API layers (the libsecp256k1 compatibility shim, the
stable `ufsecp_*` C ABI, and the libbitcoin batch/GPU bridge). They all interoperate
and verify the same curve math — **pick whichever fits your integration**. Nothing
forces a single form; the variety is intentional, so a consumer can use the path that
costs it the least conversion.

> **Schnorr is simple:** there is exactly one form (BIP-340, 64 bytes) and no low-S /
> malleability concept. Everything below about *forms* and *high-S* is **ECDSA-only**.
> See [§6](#6-schnorr-bip-340).

---

## TL;DR — pick your path

| You have / want | Recommended entry point |
|-----------------|-------------------------|
| DER from a Bitcoin script (BIP-66 / lax) | `secp256k1_ecdsa_signature_parse_der_lax` → (optional `secp256k1_ecdsa_signature_normalize`) → `secp256k1_ecdsa_verify` (shim). Or parse to compact and use the `ufsecp_*` C ABI. |
| libbitcoin `ec_signature` (opaque object) | Pass it **unchanged** to `ufsecp_lbtc_verify_ecdsa` / `_columns` (batch, zero-copy, normalizes internally) or `ufsecp_ecdsa_verify_opaque` (single). |
| Standard compact `r‖s` (big-endian) | `ufsecp_ecdsa_verify` (single, strict low-S) or `ufsecp_lbtc_verify_ecdsa_columns_compact` (batch). |
| Maximum GPU batch throughput | Bridge **opaque rows** `ufsecp_lbtc_verify_ecdsa` — the GPU kernel parses + low-S normalizes on-device (no host staging). |
| To pre-normalize once and pass directly | `ufsecp_lbtc_ecdsa_sig_pack` (build the column) → `ufsecp_lbtc_verify_ecdsa_columns_compact`. |
| The exact Bitcoin Core flow | `parse_der_lax` → `secp256k1_ecdsa_signature_normalize` → `secp256k1_ecdsa_verify` (all in the shim). |
| Schnorr (BIP-340) | `ufsecp_schnorr_verify` (single) / `ufsecp_lbtc_verify_schnorr[_columns]` (batch) / `secp256k1_schnorrsig_verify` (shim). |

---

## 1. The byte forms

| Form | Bytes | Layout | Notes |
|------|-------|--------|-------|
| **ECDSA opaque** — the `secp256k1_ecdsa_signature` object, == libbitcoin `ec_signature` | 64 | `r`(32, **little-endian**) ‖ `s`(32, **little-endian**) | The in-memory libsecp256k1 object. Output of `secp256k1_ecdsa_signature_parse_compact` / `_parse_der[_lax]`. |
| **ECDSA compact** — the standard serialized form | 64 | `r`(32, **big-endian**) ‖ `s`(32, **big-endian**) | Output of `ufsecp_ecdsa_sign` and `secp256k1_ecdsa_signature_serialize_compact`. The "normal" wire form. |
| **ECDSA DER** (BIP-66 / lax) | variable | `SEQ{ INT r, INT s }` | On-wire Bitcoin script signatures. |
| **Schnorr** (BIP-340) | 64 | `r`(32, BE) ‖ `s`(32, BE) | The only Schnorr form. No opaque variant, no low-S. |

### Portability of the opaque form (important)

The opaque little-endian layout is **not** a custom invention of this library. Real
libsecp256k1 stores the raw little-endian scalar limbs in `secp256k1_ecdsa_signature.data`
(the `sizeof(secp256k1_scalar)==32` `memcpy` fast path in `secp256k1_ecdsa_signature_save`),
so on little-endian hosts the opaque object **is** the byte-reverse of compact. This
library's shim mirrors that exactly. Therefore libbitcoin's `ec_signature` bytes are
**identical** whether the backend is stock libsecp256k1 or UltrafastSecp256k1 — you can
copy the opaque object across the boundary unchanged.

---

## 2. Verify entry points by layer

### libsecp256k1 compatibility shim (`secp256k1_*`, drop-in libsecp256k1 ABI)

| Function | Input form | Single/Batch | high-S |
|----------|-----------|--------------|--------|
| `secp256k1_ecdsa_verify` | opaque | single | **accepts** high-S (consensus convenience — diverges from stock libsecp, which rejects; see [§3](#3-high-s-policy)) |
| `secp256k1_schnorrsig_verify` / `_verify_batch` | BIP-340 | single / batch | n/a |

### Stable C ABI (`ufsecp_*`)

| Function | Input form | Single/Batch | high-S |
|----------|-----------|--------------|--------|
| `ufsecp_ecdsa_verify` | compact | single | **rejects** high-S (BIP-62, strict) |
| `ufsecp_ecdsa_verify_opaque` | opaque | single | **normalizes** (accepts) |
| `ufsecp_ecdsa_verify_opaque_batch` | opaque (columns) | batch | **normalizes** |
| `ufsecp_ecdsa_verify_opaque_rows` | opaque (packed rows) | batch | **normalizes** |
| `ufsecp_ecdsa_batch_verify` | compact (packed records) | batch | **rejects** high-S (strict) |
| `ufsecp_schnorr_verify` | BIP-340 | single | n/a |

### libbitcoin bridge (`ufsecp_lbtc_*`, batch + GPU accelerator)

| Function | Input form | Single/Batch | high-S |
|----------|-----------|--------------|--------|
| `ufsecp_lbtc_verify_ecdsa` / `_collect` | opaque (rows) | batch | **normalizes** (CPU + on-device GPU) |
| `ufsecp_lbtc_verify_ecdsa_columns` / `_collect` | opaque (columns) | batch | **normalizes** |
| `ufsecp_lbtc_verify_ecdsa_columns_compact` / `_collect` | compact (columns) | batch | **normalizes** |
| `ufsecp_lbtc_verify_schnorr[...]` | BIP-340 | batch | n/a |

The bridge's GPU result is required to match the CPU/libsecp256k1 reference bit-for-bit
(enforced by `test_lbtc_consensus_diff`). The CPU fallback is always available.

---

## 3. High-S policy

secp256k1 admits two valid `s` for every ECDSA signature (`s` and `n−s`). **Bitcoin
consensus accepts both** (historical blocks contain high-S); BIP-62 / relay policy
prefers low-S. The entry points therefore come in two flavours, by design:

- **Strict** (`ufsecp_ecdsa_verify`, `ufsecp_ecdsa_batch_verify`): reject high-S.
  Use these when you have already normalized, or when you want BIP-62 semantics.
- **Normalizing / consensus** (the `*_opaque*` C-ABI calls, the shim
  `secp256k1_ecdsa_verify`, and all bridge ECDSA calls): accept high-S by low-S
  normalizing during parse. Use these for raw consensus verification of arbitrary
  on-chain signatures.

Verification is mathematically unaffected by `s` vs `n−s`, so normalizing then verifying
yields the correct consensus verdict. "Normalize" here means `s ← min(s, n−s)`.

---

## 4. Conversion & normalization functions

You rarely need these — the verify entry points above already convert/normalize
internally for their declared form. They exist for callers who want to convert or
pre-normalize explicitly (e.g. to build a signature column once).

**libsecp256k1 shim (standard ABI):**
- `secp256k1_ecdsa_signature_parse_compact` / `_serialize_compact` — compact ↔ opaque
- `secp256k1_ecdsa_signature_parse_der` / `_serialize_der`
- `secp256k1_ecdsa_signature_normalize(ctx, &out, &in)` — **the standard low-S normalizer** (opaque in/out)

**Stable C ABI:**
- `ufsecp_ecdsa_sig_opaque_to_compact` / `ufsecp_ecdsa_sig_compact_to_opaque`
- `ufsecp_ecdsa_sig_normalize_opaque(ctx, in, out, &changed)` — low-S normalize (opaque)
- `ufsecp_ecdsa_sign` — produces compact, **already low-S**

**libbitcoin bridge:**
- `ufsecp_lbtc_ecdsa_sig_pack(in, input_is_opaque, out64)` / `ufsecp_lbtc_ecdsa_sigs_pack(...)`
  — opaque **or** compact → GPU-native compact, low-S normalized (build the table once)

To normalize the **compact** form directly today, round-trip through the shim object:
`parse_compact → secp256k1_ecdsa_signature_normalize → serialize_compact`, or use
`ufsecp_lbtc_ecdsa_sig_pack(in, /*input_is_opaque=*/0, out)`.

---

## 5. Scenario recipes

**A. You store libbitcoin `ec_signature` (opaque) and want batch consensus verify**
(fastest on GPU — zero conversion, on-device parse):
```c
ufsecp_lbtc_verify_ecdsa(ctrl, rows /*32 msg|33 pub|64 ec_signature*/, n, key_size, results);
// or columns:
ufsecp_lbtc_verify_ecdsa_columns(ctrl, msg32, pub33, ec_signatures64, n, results);
```
No conversion, no normalize call — the engine does both internally.

**B. You want the standard libsecp256k1 single-verify flow (DER on the wire):**
```c
secp256k1_ecdsa_signature sig;
ecdsa_signature_parse_der_lax(ctx, &sig, der, derlen);   // your existing parser
secp256k1_ecdsa_signature_normalize(ctx, &sig, &sig);    // optional: force low-S
secp256k1_ecdsa_verify(ctx, &sig, msg32, &pubkey);       // shim accepts high-S anyway
```

**C. You keep standard compact `r‖s` and want batch verify:**
```c
ufsecp_lbtc_verify_ecdsa_columns_compact(ctrl, msg32, pub33, sigs64_compact, n, results);
```

**D. You want to pre-normalize once and pass the column directly (no per-call work):**
```c
ufsecp_lbtc_ecdsa_sigs_pack(ec_signatures64, n, /*input_is_opaque=*/1, packed64);  // build once
ufsecp_lbtc_verify_ecdsa_columns_compact(ctrl, msg32, pub33, packed64, n, results); // verify
```

**E. Single-signature strict (BIP-62) verify of a normalized compact sig:**
```c
ufsecp_ecdsa_verify(ctx, msg32, sig64_compact /*low-S*/, pubkey33);
```

### ⚠️ Common pitfall — `serialize_compact` + the opaque batch API

`secp256k1_ecdsa_signature_serialize_compact` produces **big-endian compact**. The bridge's
**row / `_columns` ECDSA verify** (`ufsecp_lbtc_verify_ecdsa`, `ufsecp_lbtc_verify_ecdsa_columns`)
expects the **opaque** form (the raw `secp256k1_ecdsa_signature` / `ec_signature` bytes,
little-endian). **Do NOT `serialize_compact` and feed the result to those calls** — they would
byte-reverse the already-big-endian bytes and verification would fail.

- **Bridge batch (`ufsecp_lbtc_verify_ecdsa[_columns]`):** pass the `ec_signature` **unchanged**.
  No `serialize_compact`, and no `normalize` needed either — the bridge low-S normalizes
  internally on CPU *and* GPU.
- **Only if you want the compact form:** use `ufsecp_lbtc_verify_ecdsa_columns_compact` — *that*
  one expects big-endian compact, so `serialize_compact` (optionally after `normalize`) is the
  correct conversion for it.

> Note on single vs batch (the source of the original integration failure): libbitcoin's
> single `verify_signature` calls `secp256k1_ecdsa_signature_normalize` explicitly before
> `secp256k1_ecdsa_verify`, so high-S always worked there. The zero-copy `batch_verify` passes
> the raw opaque sigs to the bridge with **no** normalize call, relying on the bridge to do it.
> High-S signatures therefore passed single-verify but failed batch-verify until the bridge was
> fixed to normalize high-S internally (now done, CPU + GPU). Pass opaque, unchanged; the bridge
> handles low-S.

### The conversion algorithm (if you want to do it yourself, no dependency)

opaque `ec_signature` → standard compact, low-S normalized:
1. Byte-reverse each 32-byte half: `out[i] = in[31 - i]` (little-endian → big-endian).
2. Low-S: read `s` (the second 32 bytes) as a big-endian integer; if `s > n/2`
   (`HALF_ORDER`), replace `s` with `n − s`.

---

## 6. Schnorr (BIP-340)

Schnorr has a single canonical 64-byte form (`r‖s`, big-endian) and **no** low-S /
malleability concept, so none of the ECDSA form/high-S complexity applies. Use:
- `ufsecp_schnorr_verify` (single) / `ufsecp_lbtc_verify_schnorr[_columns]` (batch),
- `secp256k1_schnorrsig_verify` / `_verify_batch` (shim).

Pass the BIP-340 signature bytes directly in all cases.

---

## 7. Notes

- **Consensus:** for block validation use a normalizing/consensus entry point (it accepts
  the high-S signatures present in historical blocks). The GPU path is consensus-anchored
  to the CPU/libsecp reference (`test_lbtc_consensus_diff`).
- **Performance:** for libbitcoin the opaque path is the fastest (zero-copy; the GPU
  kernel parses + normalizes on-device with no host staging). The compact/pack path is for
  integrators that prefer the standard serialized form or already store compact sigs; it is
  correctness-equivalent, not a throughput win.
- **Divergences:** the shim's accept-high-S `secp256k1_ecdsa_verify` is a deliberate
  divergence from stock libsecp256k1 (which rejects high-S); see
  `docs/SHIM_KNOWN_DIVERGENCES.md`.
