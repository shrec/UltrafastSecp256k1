# libbitcoin integration (bridge-free, CPU-first)

This guide shows how libbitcoin-system can use UltrafastSecp256k1 for batch verification
**through the standard `libsecp256k1` shim**, without the bespoke `libbitcoin_bridge`
(`ufsecp_lbtc_*`). It implements **Model 1** from [`INTEGRATION_MODELS.md`](INTEGRATION_MODELS.md).

> This is a guide only — it does **not** edit the libbitcoin checkout. Apply the mapping in your
> libbitcoin fork's `WITH_ULTRAFAST` path if/when you want to drop the bridge dependency.

## Background

- libbitcoin's production verify path is the per-signature `secp256k1_ecdsa_verify` /
  `secp256k1_schnorrsig_verify` in the script interpreter. The shim already covers this fully
  (**Model 0** — no changes needed).
- The bridge is used in exactly one place: `ecdsa::batch_verify` / `schnorr::batch_verify`
  → `ufsecp_lbtc_verify_ecdsa` / `ufsecp_lbtc_verify_schnorr` (under `WITH_ULTRAFAST`). This is
  the only thing to remap, and it uses just 2 of the bridge's ~28 symbols.

## What changes

Replace the packed-row bridge call (`hash32|pubkey33|sig64|opaque-key` rows + controller +
backend selection) with the shim's per-row batch results API. Threads, scratch memory, and the
fixed-base cache are all engine-managed; there is no controller to create and no config file.

### Before (bridge)

```cpp
// rows: contiguous hash32|pubkey33|sig64|opaque-key, n rows; results: 1 byte/row.
ufsecp_lbtc_ctrl* ctrl = nullptr;
ufsecp_lbtc_ctrl_create(&ctrl, UFSECP_LBTC_BACKEND_AUTO);
ufsecp_lbtc_verify_ecdsa(ctrl, rows, n, key_size, results);
ufsecp_lbtc_ctrl_destroy(ctrl);
```

### After (shim, Model 1)

```cpp
#include <secp256k1.h>
#include <secp256k1_batch.h>   // shim batch extension

// ctx: a verify-capable secp256k1_context (libbitcoin already keeps one).
// Build the array-of-pointers the standard API expects from your row data:
std::vector<const secp256k1_ecdsa_signature*> sigs(n);
std::vector<const unsigned char*>             msgs32(n);
std::vector<const secp256k1_pubkey*>          pubkeys(n);
// ... fill sigs[i]/msgs32[i]/pubkeys[i] from your parsed inputs ...

std::vector<int> results(n);
// max_threads: 0 = auto for one big batch; 1 = serial if you call this from your OWN pool.
int all_ok = secp256k1_ecdsa_verify_batch_results(
    ctx, sigs.data(), msgs32.data(), pubkeys.data(), n, /*max_threads=*/0, results.data());

if (!all_ok) {
    for (size_t i = 0; i < n; ++i)
        if (results[i] == 0) { /* row i failed (invalid or malformed) */ }
}
```

Schnorr is identical with `secp256k1_schnorrsig_verify_batch_results(ctx, sigs64, msgs, 32,
xonly_pubkeys, n, max_threads, results)`.

## Mapping `ecdsa::batch_verify` / `schnorr::batch_verify`

| libbitcoin concept | shim equivalent |
|---|---|
| controller + `UFSECP_LBTC_BACKEND_AUTO` | none (engine-managed) |
| packed `hash32\|pubkey33\|sig64\|key` rows | array-of-pointers (`sigs`, `msgs32`, `pubkeys`) |
| 1-byte/row verdict buffer | `int results[n]` (`1`=valid, `0`=invalid) |
| return = all rows valid | function returns `1` iff all rows valid |
| GPU selection / fallback | not in this path (use Model 2 if you need GPU) |

## Choosing `max_threads`

- **libbitcoin verifies one large batch and owns the CPU:** pass `0` (auto). The engine splits
  the batch into 4096-row chunks across up to 64 threads.
- **libbitcoin already parallelizes (its own thread pool calls batch_verify per worker):** pass
  `max_threads = 1` so the engine does **not** spawn nested threads (avoids oversubscription).
- **Bounded CPU budget:** pass an explicit `N`.

The result is identical for any thread count — only throughput changes.

## Notes

- **High-S ECDSA** signatures are accepted (matches single `secp256k1_ecdsa_verify`); normalize
  before storage if your application requires low-S.
- **No config file** is written; if you don't want the fixed-base `.bin` cache in the CWD, call
  `ufsecp_set_cache_dir("<your dir>")` once at startup (or set `SECP256K1_CACHE_DIR`).
- **No-failure:** the batch calls never throw across the ABI; on internal thread-spawn failure
  they fall back to serial verification with an identical result.
- **GPU / zero-copy** stays in Model 2 (`libbitcoin_bridge` / `ufsecp_gpu_*`) and is opt-in.
