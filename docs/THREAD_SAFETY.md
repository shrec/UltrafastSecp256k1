# Thread Safety

**UltrafastSecp256k1 v4.4.0**

---

## CPU Backend Thread Safety

The CPU backend is stateless. All functions are re-entrant and thread-safe
provided each thread uses its own private keys and ephemeral scalars.

The comb generator table (`ecmult_gen_comb`) is built once via `std::call_once`
and is thereafter read-only — no lock is taken on any read path.

The windowed fixed-base table (`g_context` in `precompute.cpp`) may be (re)built at
runtime via `configure_fixed_base()` / autotune. It is a `std::shared_ptr`:
`scalar_mul_generator` and `batch_scalar_mul_generator` take a local `shared_ptr`
snapshot under `g_mutex` *before* releasing the lock to read the table, so a
concurrent `configure_fixed_base()` (which does `g_context.reset()`) cannot free the
table mid-read (PRECOMPUTE-GCONTEXT-UAF). Configuring the fixed base once before
spawning worker threads remains the recommended pattern; the snapshot makes the
reconfigure-during-use case safe (defined behavior) rather than a use-after-free.

---

## libsecp256k1 Shim — Known Thread-Safety Deviations

### Per-Context Blinding (tracked deviation)

**Upstream libsecp256k1 behavior:** `secp256k1_context_randomize(ctx, seed)`
installs blinding state into `ctx`. Each context has independent blinding.

**This shim's behavior:** Blinding is thread-local. `context_randomize` stores
the seed in `ctx->blind` but activates CT blinding on the *calling thread*,
not scoped to the context. Two contexts on the same thread overwrite each
other's blinding state.

**Practical impact:** Bitcoin Core creates one context per thread and calls
`context_randomize` once — this pattern is unaffected.

**References:** `compat/libsecp256k1_shim/src/shim_context.cpp` lines 104–117.

---

## GPU Backends

One `ufsecp_gpu_ctx*` must not be shared across threads without external
synchronization. Create one GPU context per thread.
