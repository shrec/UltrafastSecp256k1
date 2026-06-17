# Review: Bridge-free integration standard (CPU-first)

**Status:** implemented, all fast CI gates green, ready for review.
**Scope:** CPU batch verification through the standard `libsecp256k1` shim — no bespoke bridge required. GPU stays opt-in.
**Audience:** integrators (libbitcoin / Bitcoin Core / Litecoin / Knots / Dogecoin) and reviewers.

> **How to give feedback:** please add inline comments or fill in the [Reviewer notes](#reviewer-notes) section at the bottom and send it back. Each open question is numbered (Q1, Q2, …) for easy reference.

---

## 1. Why this change

The trigger was a batch-verification run of ~429M ECDSA signatures that appeared to hang for 8+ hours. Root cause: the call ran the **entire batch on a single core** with large intermediate allocations. From that, three requirements emerged:

1. **The library should own CPU parallelism and device selection** — the caller should not have to manage threads or pick CPU/GPU.
2. **No oversubscription** when the caller already runs its own thread pool (evoskuil's concern).
3. **No failure modes / no surprise files** — the API should degrade gracefully and must not silently write a `config.ini` into the integrator's working directory.

This work makes the **standard `libsecp256k1` shim** the single integration surface and pushes "smartness" (multi-threading, per-row results) *inside* it. The bespoke `libbitcoin_bridge` is repositioned as an optional advanced tier (GPU / zero-copy), not a required layer.

---

## 2. Integration models (the standard we publish)

| | Model 0 | Model 1 | Model 2 |
|---|---|---|---|
| **Surface** | shim per-signature | shim **batch extension** | `libbitcoin_bridge` / `ufsecp_gpu_*` |
| **Effort** | none (drop-in) | small (call batch API) | larger (bridge ABI) |
| **Threads** | none | engine-managed (`max_threads`) | engine / GPU |
| **Per-row results** | n/a | yes (`_results`) | yes |
| **GPU / zero-copy** | no | no | yes |
| **Required?** | default | recommended for batch | opt-in only |

- **Model 0 — Drop-in (already works):** link the shim instead of upstream `libsecp256k1`; per-signature `secp256k1_ecdsa_verify` / `secp256k1_schnorrsig_verify` are identical to upstream. This already covers script/interpreter validation.
- **Model 1 — Batch throughput (this change):** the shim batch extension, with engine-managed multi-threading, explicit caller thread control, and optional per-row results. **No bridge, no config file.**
- **Model 2 — Advanced (kept):** GPU + zero-copy packed-row/columnar/collect via the existing bridge. Unchanged; opt-in.

Full detail: [`INTEGRATION_MODELS.md`](INTEGRATION_MODELS.md). libbitcoin-specific mapping: [`LIBBITCOIN_INTEGRATION.md`](LIBBITCOIN_INTEGRATION.md).

---

## 3. The Model 1 API (what's new)

Header: `compat/libsecp256k1_shim/include/secp256k1_batch.h` (additive — existing `secp256k1_*` symbols are unchanged).

```c
/* All-or-nothing, auto threads (back-compat wrappers; unchanged signatures). */
int secp256k1_ecdsa_verify_batch     (ctx, sigs, msgs32, pubkeys, n);
int secp256k1_schnorrsig_verify_batch(ctx, sigs64, msgs, msglen, pubkeys, n);

/* NEW: explicit thread control. */
int secp256k1_ecdsa_verify_batch_mt     (ctx, sigs, msgs32, pubkeys, n, max_threads);
int secp256k1_schnorrsig_verify_batch_mt(ctx, sigs64, msgs, msglen, pubkeys, n, max_threads);

/* NEW: per-row verdicts. results[i] = 1 (valid) / 0 (invalid or malformed). */
int secp256k1_ecdsa_verify_batch_results     (ctx, sigs, msgs32, pubkeys, n, max_threads, results);
int secp256k1_schnorrsig_verify_batch_results(ctx, sigs64, msgs, msglen, pubkeys, n, max_threads, results);
```

### `max_threads` contract

| value | meaning | when to use |
|-------|---------|-------------|
| `0` | auto: `hardware_concurrency()`, capped at 64 | one big batch, library owns the CPU |
| `1` | serial (no worker threads spawned) | **you already run your own pool** — avoids oversubscription |
| `N` | cap at `N` worker threads | bounded CPU budget |

The engine splits a batch into 4096-row chunks pulled from an atomic work queue; per-thread scratch stays `O(chunk)`, never `O(n)` — this also removes the large-intermediate-allocation problem from the original 8-hour run.

---

## 4. Guarantees and properties

- **Result is thread-count-independent.** The boolean ("all valid") result is identical to the single-threaded path for any `max_threads`. Threading is a pure throughput change.
- **Constant-time:** batch verify is variable-time over **PUBLIC** data only (pubkey / message / signature). There is no secret material and threading adds no secret-dependent branch — **zero CT impact** (same class as the existing `ecdsa_batch_verify_mt`).
- **No-failure contract:** the batch functions never throw across the C ABI. If internal thread creation fails, they fall back to serial verification; the result is deterministic and identical to the serial path. The `_results` variants return `0` on a NULL `results` pointer; NULL `ctx` fires the illegal callback as everywhere else.
- **Per-row diagnostics in one call:** `_results` reports exactly which rows failed (invalid signature *or* malformed input) without re-verifying individually.
- **No config files:** the engine self-manages its fixed-base `.bin` cache. Point it at a directory with `ufsecp_set_cache_dir()` / `SECP256K1_CACHE_DIR`, otherwise it uses the CWD. **No `config.ini` is created or read** (delivered earlier; confirmed unchanged by this batch path).
- **ECDSA high-S** signatures are accepted in batch, matching single `secp256k1_ecdsa_verify`. Normalize before storage if your application requires low-S.
- **Schnorr `msglen != 32`** is served via per-signature verify (MSM needs fixed 32-byte slots), matching upstream BIP-340 arbitrary-length semantics.

---

## 5. libbitcoin mapping (no checkout edits)

The bridge is used in exactly one place today: `ecdsa::batch_verify` / `schnorr::batch_verify` → `ufsecp_lbtc_verify_ecdsa/schnorr` (`WITH_ULTRAFAST`), and `batch_verify` is currently test-only (future IBD). That maps directly onto the shim:

```cpp
// Before (bridge): controller + packed hash32|pubkey33|sig64|key rows + 1-byte/row verdicts
ufsecp_lbtc_ctrl_create(&ctrl, UFSECP_LBTC_BACKEND_AUTO);
ufsecp_lbtc_verify_ecdsa(ctrl, rows, n, key_size, results);
ufsecp_lbtc_ctrl_destroy(ctrl);

// After (shim, Model 1): array-of-pointers + int results[n], no controller, no config file
std::vector<int> results(n);
int all_ok = secp256k1_ecdsa_verify_batch_results(
    ctx, sigs.data(), msgs32.data(), pubkeys.data(), n, /*max_threads=*/0, results.data());
// pass max_threads = 1 if libbitcoin calls this from its OWN worker pool
```

Guidance, decision table, and the Schnorr twin are in [`LIBBITCOIN_INTEGRATION.md`](LIBBITCOIN_INTEGRATION.md). **No libbitcoin source was modified** — this is a guide only.

---

## 6. Changes in this commit

**Engine** (`src/cpu/`)
- New `secp256k1::schnorr_batch_verify_mt(entries, n, max_threads)` — the Schnorr twin of `ecdsa_batch_verify_mt` (same chunked atomic-queue design, cap 64, `0`=auto / `1`=serial).

**Shim** (`compat/libsecp256k1_shim/`)
- `shim_batch_verify.cpp` rewritten around shared cores: large-batch ECDSA → `ecdsa_batch_verify_mt`, Schnorr → `schnorr_batch_verify_mt`.
- Four additive symbols (`*_verify_batch_mt`, `*_verify_batch_results`); existing two symbols kept as thin auto-threaded wrappers.
- No-failure fallback (serial on thread-spawn failure).

**Bug fix (latent):** the pre-existing `secp256k1_ecdsa_verify_batch` parsed the opaque `secp256k1_ecdsa_signature.data` as **big-endian** compact `r||s`, but the shim stores it in the engine's **native little-endian** limb form (see `shim_ecdsa.cpp::ecdsa_sig_from_data`). All ECDSA batch parse sites now use `Scalar::parse_bytes_strict_le`, matching single `secp256k1_ecdsa_verify`. There was no prior ECDSA-batch test; Schnorr was unaffected. **Reviewers: please sanity-check this against your `ec_signature` fixtures.** (Q1)

**Tests:** `compat/libsecp256k1_shim/tests/test_shim_batch_mt.cpp` (`shim_batch_mt`) — MT == single across thread counts `{0,1,2,8,64}` for ECDSA and Schnorr (n > one chunk so threads actually spawn), per-row `results` pinpoint injected-invalid rows, `n==0` vacuous, small-`n` parity, `max_threads==1` (caller-pool) parity. All pass; existing shim regressions unaffected.

**Docs:** `INTEGRATION_MODELS.md` (new), `LIBBITCOIN_INTEGRATION.md` (new), plus updates to `secp256k1_batch.h`, `API_REFERENCE.md`, `SHIM_KNOWN_DIVERGENCES.md` (SHIM-BATCH-EXT), `AUDIT_CHANGELOG.md`, `TEST_MATRIX.md`.

**Ledgers / CI:** shim API function count 80 → 84 (canonical sync); `schnorr_batch_verify_mt` added to the soundness allowlist with justification (public-data verifier, differential-covered). `ufsecp_*` C ABI count is **unchanged** (the new symbols are shim-side, not `ufsecp_*`). `ci/run_fast_gates.sh` passes; source graph rebuilt.

---

## 7. Out of scope (future, additive)

- GPU dispatch **inside** the shim/engine AUTO path (promoting Model 2 into the standard surface). Kept in the bridge for now.
- Zero-copy packed-row / columnar / collect layouts in the standard surface.
- Editing the libbitcoin-system source (guide/snippet only).

---

## 8. Open questions for reviewers

- **Q1 — ECDSA opaque layout:** does `parse_bytes_strict_le` for the opaque `secp256k1_ecdsa_signature.data` match your `ec_signature` fixtures / expectations on your target platforms (LE hosts)?
- **Q2 — Default threading of the legacy symbols:** we made the existing `secp256k1_ecdsa_verify_batch` / `secp256k1_schnorrsig_verify_batch` **auto-threaded** (was single-threaded). Result is identical; only throughput changes. Acceptable, or would you prefer the legacy symbols to stay serial and only `_mt` thread?
- **Q3 — `max_threads` semantics:** is `0=auto / 1=serial / N=cap` the right contract for your call sites, and is the cap of 64 sensible for your hardware?
- **Q4 — Per-row `results` type:** we use `int results[n]` (`1`/`0`). Do you need a richer per-row status (e.g. distinguish "malformed input" from "valid-form but bad signature"), or is pass/fail enough?
- **Q5 — IBD batch shape:** for your future IBD path, do you expect one large batch (use `max_threads=0`) or many batches from your own pool (use `max_threads=1`)? This affects where we recommend threading lives.
- **Q6 — Anything missing** from Model 1 that would still force you onto the bridge for a CPU-only deployment?

---

## <a id="reviewer-notes"></a>9. Reviewer notes (please fill in and return)

| # | Reviewer | Section / file | Note / concern | Suggested change |
|---|----------|----------------|----------------|------------------|
|   |          |                |                |                  |
|   |          |                |                |                  |
|   |          |                |                |                  |

**Overall verdict:** ☐ approve   ☐ approve with changes   ☐ needs discussion

---

### Appendix — pointers

- Standard surface header: `compat/libsecp256k1_shim/include/secp256k1_batch.h`
- Implementation: `compat/libsecp256k1_shim/src/shim_batch_verify.cpp`
- Engine MT: `src/cpu/src/batch_verify.cpp` (`ecdsa_batch_verify_mt`, `schnorr_batch_verify_mt`)
- Test: `compat/libsecp256k1_shim/tests/test_shim_batch_mt.cpp`
- Deep docs: `docs/INTEGRATION_MODELS.md`, `docs/LIBBITCOIN_INTEGRATION.md`, `docs/SHIM_KNOWN_DIVERGENCES.md`, `docs/AUDIT_CHANGELOG.md`
