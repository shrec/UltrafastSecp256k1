# Libsecp256k1 Shim — Known Behavioral Divergences

This document lists every intentional behavioral difference between the
UltrafastSecp256k1 libsecp256k1 shim and upstream libsecp256k1.

**An unlisted divergence is a bug.** If you discover shim behavior that differs
from libsecp256k1 and it is not documented here, it must either be fixed or added
to this file before any PR.

For the complete compatibility test matrix see `compat/libsecp256k1_shim/tests/`.

---

> **AGENT RULE — ABSOLUTE**
>
> The sections **"Security improvements"** and **"Performance improvements"** below are
> **owner-approved intentional divergences from upstream**. These represent places where
> the shim is **strictly better** than libsecp256k1 — either safer or faster — while
> remaining correct. They are explicitly protected per `CLAUDE.md` ("სადაც უკეთესები ვართ
> არ ვეხებით" — where we are better, we do not touch).
>
> **DO NOT revert, "fix", align, or remove any entry in those two sections.**
> They are not bugs. They are not divergences to be eliminated. Any agent that
> attempts to align these entries with upstream behavior is making a security or
> performance regression. Entries in those sections must remain exactly as documented.
>
> The **"Capability gaps"** section may be fixed or removed as gaps are closed.

---

## Security improvements — LOCKED: DO NOT REVERT (shim is stricter than upstream)

<!-- OWNER-LOCKED: all entries below are intentional security hardening.
     Reverting any of these to match upstream behavior is a security regression.
     See agent rule at top of file. -->

These divergences are intentional hardening beyond upstream libsecp256k1. They
are stricter but never less safe. Callers using well-formed inputs are unaffected.

---

### secp256k1_ecdh — private key >= curve order rejected

- **Upstream behavior:** `secp256k1_ecdh` with a private key value `>= n` (curve order)
  reduces the key silently mod n and proceeds. A key equal to `n` reduces to 0 and is
  rejected (returns 0); `n+1` reduces to 1 and succeeds.
- **Shim behavior:** Any private key value `>= n` returns 0 immediately.
  `parse_bytes_strict_nonzero` is used instead of `from_bytes` (CLAUDE.md Rule 11).
- **Reason:** Rule 11 requires strict private key parsing for all functions accepting a
  secret key byte array. Silent mod-n reduction can mask caller errors (e.g., passing
  `n+1` believing it is invalid). Strict rejection is the safer default.
- **Impact:** Any caller passing a private key value `>= n` to `secp256k1_ecdh`.
  In practice, well-formed private keys from `secp256k1_ec_seckey_verify` or
  `secp256k1_keypair_create` are always `< n` and are unaffected.
- **Test:** `test_ecdh_privkey_out_of_range` in
  `audit/test_regression_shim_security_v8.cpp` (checks ORDER, ORDER+1, 0xff..ff).

---

### secp256k1_ecdh — off-curve pubkey rejected

- **Upstream behavior:** libsecp256k1 trusts the `secp256k1_pubkey` opaque struct to
  contain a valid on-curve point (invariant: always populated via `ec_pubkey_parse`).
  No runtime on-curve check is performed in `secp256k1_ecdh` itself.
- **Shim behavior:** The shim performs a `y²=x³+7` check before the scalar
  multiplication. If the point is off-curve, returns 0.
- **Reason:** An invalid-curve attack using a small-order subgroup point can recover
  private key bits modulo the subgroup order. While normal use routes through
  `ec_pubkey_parse` (which validates), the C ABI struct can be written directly by
  hostile callers. The check cost is negligible vs the ECDH scalar multiplication.
- **Impact:** None for normal callers. Only affects callers who bypass `ec_pubkey_parse`.
- **Test:** `test_ecdh_pubkey_off_curve` in `audit/test_regression_shim_security_v8.cpp`.

---

### secp256k1_context_randomize — seed >= n

- **Upstream behavior:** Seeds are treated as opaque bytes for blinding; no range check.
  Upstream libsecp256k1 applies the seed directly as a blinding scalar (with mod-n reduction
  internally), so seeds >= n are accepted and result in a reduced blinding value.
- **Shim behavior:** Uses `Scalar::parse_bytes_strict_nonzero` on the seed. Seeds >= n or == 0
  disable blinding (the blinding scalar is left at its current value) rather than silently
  reducing mod-n. No conditional subtraction of n is applied to the seed.
- **Reason:** Rule 11 (CLAUDE.md) requires `parse_bytes_strict_nonzero` for any private-key
  or secret-scalar input. Seeds >= n are astronomically rare with fresh OS randomness;
  disabling blinding on such seeds is safe (the call is advisory in libsecp256k1 too — a
  failed randomization leaves the context operational, just without blinding).
- **Impact:** A caller passing a seed value in [n, 2^256) will not get a blinding scalar
  derived from that seed — blinding is left disabled. Callers using the recommended 32 bytes
  of fresh randomness from the OS are effectively never affected (probability ~2^-128).
- **Test:** `audit/test_regression_p2_ct_shim_fixes.cpp` covers the `parse_bytes_strict_nonzero`
  path in `secp256k1_context_randomize` (CT-003).

---

### secp256k1_ecdsa_signature_parse_der — rejects r=0/s=0; requires exact SEQUENCE consumption

- **Upstream behavior:** `secp256k1_ecdsa_sig_parse` accepts r=0 and s=0 at parse time (they
  are syntactically valid DER INTEGERs); rejection is deferred to `secp256k1_ecdsa_sig_verify`.
  It also consumes the SEQUENCE body exactly with r and s.
- **Shim behavior:**
  - **r=0 / s=0 are rejected at parse** (returns 0). The only canonical DER encoding of zero
    is `02 01 00`, which the parser's minimal-encoding rule rejects (a single leading `0x00`
    value byte fails the `len < 2` check), so no DER encoding of zero ever reaches the
    in-range scalar test. Implemented inline via `in_range_scalar` + `parse_int` — this
    replaced the earlier `parse_bytes_strict_nonzero` mechanism but preserves the same
    reject-at-parse behavior.
  - **Exact SEQUENCE consumption (RT-02, 2026-05-28):** after parsing r and s the parser
    requires `p == end`. An inflated-length SEQUENCE that fills the input buffer but leaves
    trailing bytes *inside* the SEQUENCE after s (e.g. `30 08 02 01 0F 02 01 01 7F 7F`) is
    now rejected — matching upstream strict parsing and the native C ABI parser
    (`src/cpu/src/impl/ufsecp_ecdsa.cpp`).
- **Reason:** Defense-in-depth. r=0/s=0 are astronomically rare and invalid; rejecting at
  parse is never less safe than deferring to verify (verify rejects them too). The
  exact-SEQUENCE check rejects malformed encodings upstream also rejects.
- **Impact:** None for well-formed signatures. Note the asymmetry: `parse_compact` uses
  `parse_bytes_strict` and ACCEPTS r=0/s=0 (matching libsecp), while `parse_der` rejects them
  — see KB `DER-COMPACT-ASYMMETRY`.
- **Test:** `compat/libsecp256k1_shim/tests/test_shim_der_zero_r.cpp` (r=0, s=0, trailing-bytes,
  valid-sig) and source-scan `audit/test_regression_shim_seckey_erase.cpp`.

---

### secp256k1_schnorr_sign (BCHN legacy shim) — noncefp/ndata ignored, always RFC 6979

- **Scope:** Bitcoin Cash (BCHN) legacy-Schnorr compatibility shim
  (`compat/libsecp256k1_bchn_shim/src/shim_schnorr_bch.cpp`) — **not part of the Bitcoin Core
  CPU-backend evaluation.**
- **Upstream behavior:** The BCHN legacy `secp256k1_schnorr_sign` API takes a `noncefp` /
  `ndata` pair. In practice BCHN callers pass `NULL` (deterministic RFC 6979).
- **Shim behavior:** Accepts the `noncefp` / `ndata` parameters but **silently ignores both**
  and always uses internal RFC 6979 nonce derivation. It does **not** fire the illegal
  callback for a non-NULL `noncefp` (unlike the BIP-340 `secp256k1_schnorrsig_sign_custom`
  shim, which under PASS3-001 rejects unsupported custom nonce functions).
- **Reason:** The BCHN legacy API only ever supported deterministic RFC 6979 nonces in
  practice; a fixed RFC 6979 nonce is the correct and only behavior. The silent-ignore is
  acceptable for this legacy compatibility surface and is documented here for completeness
  rather than changed (BCHN callers historically pass NULL). Documented per CLAUDE.md Rule 3.
- **Impact:** A BCHN caller passing a non-NULL `noncefp` expecting a custom nonce would get
  RFC 6979 instead, without an error. No Bitcoin Core caller is affected (out of scope).
- **Test:** Behavioral parity covered by the BCHN shim's own Schnorr round-trip tests; no
  dedicated negative test (documentation-only divergence).

---

### secp256k1_musig_partial_sig_agg — all-zero check uses CT accumulator (SHIM-MUSIG-CT)

- **Upstream behavior:** No all-zero check — returns the aggregated value unconditionally.
- **Shim behavior:** Checks whether the 64-byte aggregated signature is all-zero (degenerate
  aggregation result) and returns 0 if so. The check uses a branchless OR-accumulator
  (`uint32_t nonzero = 0; for(i) nonzero |= sig[i]`) running all 64 bytes unconditionally.
- **Reason:** The early-exit pattern would leak information about the signature bytes via
  branch-predictor and cache timing — the loop would terminate faster when a non-zero byte
  appeared early, revealing the index of the first non-zero byte. The OR-accumulator
  eliminates this side-channel. The all-zero check itself is an additional fail-closed guard
  not present in upstream.
- **Impact:** No behavioral change for callers — the return value is the same. Timing
  behavior is now data-independent for the all-zero check.
- **Test:** `test_regression_shim_security_v7_run` (musig2 aggregate round-trips).

---

### secp256k1_ecdsa_sign / secp256k1_ecdsa_sign_recoverable / secp256k1_schnorrsig_sign_custom — custom nonce function rejected

- **Upstream behavior:** Any `noncefp` / `extraparams->noncefp` is dispatched (called with the
  message, key, and counter to produce a nonce).
- **Shim behavior:** Only the standard nonce functions are accepted (NULL, `rfc6979`, `default`
  for ECDSA; NULL, `bip340` for Schnorr). Any other non-NULL `noncefp` fires the illegal callback
  with a descriptive message before returning 0. Previously returned 0 silently (fixed 2026-05-21,
  PASS3-001).
- **Reason:** The shim uses RFC 6979 / BIP-340 nonce generation internally and cannot forward
  an arbitrary nonce function. Fail-closed so callers relying on a specific nonce function see
  an error rather than silently receiving RFC 6979 output.
- **Impact:** Callers with custom nonce functions. Bitcoin Core uses NULL (RFC 6979 default) —
  unaffected. Callers with callbacks installed now receive the callback notification.
- **Test:** `compat/libsecp256k1_shim/tests/test_shim_recovery_and_noncefp.cpp` NFP-1..3.

---

### secp256k1_nonce_function_rfc6979 / secp256k1_nonce_function_default

- **Upstream behavior:** These function pointers generate RFC 6979 nonce bytes when
  called directly.
- **Shim behavior:** Both pointers are stubs that return 0 (failure) and write nothing.
  They are exported for ABI symbol compatibility only. The shim never calls them.
- **Reason:** The shim's signing path uses its own RFC 6979 implementation. Returning 0
  from a stub that writes nothing is correct: a caller that invokes these pointers
  directly gets explicit failure rather than empty output with a success code (SC-08).
- **Impact:** Any caller using these pointers as standalone hash primitives (rare).

---

### secp256k1_nonce_function_bip340

- **Upstream behavior:** Generates BIP-340 aux-entropy-mixed nonce bytes when called.
- **Shim behavior:** Returns 0 (failure) and writes nothing. Exported for ABI compatibility.
- **Reason:** Same as rfc6979 stubs above. The shim handles BIP-340 nonce internally
  via the `aux_rand32` parameter of `secp256k1_schnorrsig_sign32`.
- **Impact:** Any caller using this pointer as a standalone hash primitive.

---

## Performance improvements — LOCKED: DO NOT REVERT (shim is faster; correctness identical to upstream)

<!-- OWNER-LOCKED: all entries below are intentional performance optimizations.
     Reverting any of these to match upstream behavior is a performance regression.
     Correctness is identical to upstream for all of these.
     See agent rule at top of file. -->

These divergences are intentional optimizations. The observable results (pubkeys,
signatures, verification outcomes) are identical to upstream; only latency differs.

---

### secp256k1_context_preallocated_* — placement-new semantics

- **Upstream behavior:** `secp256k1_context_preallocated_size(flags)` returns a
  flags-dependent byte count. The preallocated buffer holds ALL context state
  inline; the context is entirely self-contained in that buffer.
- **Shim behavior:** `secp256k1_context_preallocated_size` always returns
  `sizeof(secp256k1_context)` regardless of flags (our context struct size is
  flags-independent). `secp256k1_context_preallocated_create` places a
  `secp256k1_context` object into the caller's buffer via placement-new.
  The struct contains non-trivially-destructible members (Point, Scalar),
  so `secp256k1_context_preallocated_destroy` calls the destructor explicitly
  but does NOT free the buffer — caller owns it. This matches upstream semantics.
- **Reason:** Our internal context state fits in a fixed-size struct. Placement-new
  is used so member initializers (Point default ctor, Scalar default ctor) run
  correctly in the caller's buffer without requiring heap allocation. Precomputed
  tables are globally shared (not per-context), so the size is flags-independent.
- **Impact:** None for correct callers. The size returned by `preallocated_size` may
  differ from upstream (upstream may include precomputed table space in certain
  configurations). Callers must use the shim's own `preallocated_size` return value,
  not a size hard-coded from upstream libsecp256k1.
- **Test:** `audit/test_regression_shim_preallocated_ctx.cpp` PAC-1..6.

---

### secp256k1_schnorrsig_verify — thread-local xonly-pubkey cache (NEW-SHIM-004)

- **Upstream behavior:** No caching. Every `secp256k1_schnorrsig_verify` call
  re-runs `lift_x` (sqrt) and rebuilds GLV precomputation tables for the supplied
  x-only pubkey.
- **Shim behavior:** A 256-slot, thread-local, FNV-1a-fingerprinted cache stores
  the lifted point + precomputed tables. Warm cache hits skip `lift_x` and the
  GLV rebuild (~1,954 ns saved per hit on the ConnectBlock workload).
- **Reason:** ConnectBlock-style hot paths (Bitcoin Core block validation) re-verify
  the same set of x-only pubkeys many times within a small window. Caching the
  lifted point is a strict perf optimisation; correctness is identical to upstream.
- **Impact / divergence shape:**
  1. **Memory:** ~256 × ~1.5 KB ≈ 384 KB of additional thread-local memory per
     thread that ever calls `secp256k1_schnorrsig_verify` in the shim.
  2. **Hash collisions are silent:** the cache uses a 32-bit FNV-1a fingerprint
     to index into 256 slots; on collision (~2^-32 per slot ≈ 0.1% per million
     unique pubkeys) the previously cached entry is overwritten without warning.
     The next verify of the evicted key re-runs `lift_x` + table build (~2 µs penalty).
     This is a perf footprint, not a correctness issue — verification is still sound.
  3. **Thread isolation:** cache is `thread_local`, so multi-threaded callers
     each pay the cold-cache cost on the first verify per thread.
  4. **msglen != 32:** for variable-length messages, the function bypasses the cache
     and calls the varlen verify overload directly — identical latency to upstream.
     The cache speedup applies to msglen == 32 only (standard BIP-340 / ConnectBlock
     use case).
- **Note on the `verify_precomp` API:** callers that need deterministic warm-cache
  behaviour (no eviction) should use `secp256k1_xonly_pubkey_parse_precomp` +
  `secp256k1_schnorrsig_verify_precomp` from `secp256k1_schnorrsig.h`, which
  exposes the prebuilt object explicitly and bypasses the global cache.
- **Test:** Planned — `audit/test_regression_shim_schnorr_cache_collision.cpp`
  will exercise the eviction-and-rebuild path: verify key A, fill the cache by
  verifying ≥256 distinct keys (forcing at least one slot collision), re-verify
  key A and assert it still returns 1. Current coverage is indirect via the
  unique-pubkey ConnectBlock workload exercised in `bench_unified` and the
  high-diversity verify path in `audit/test_exploit_schnorr_verify_*`.

---

## Capability gaps and structural divergences

These divergences reflect architectural constraints or unimplemented features.
They are not security issues; callers are affected only if they use the specific
unsupported API.

---

### secp256k1_ecdsa_sign — ndata/extra_entropy nonce divergence (SHIM-P3-006)

- **Upstream behavior:** When `ndata` is non-NULL, `secp256k1_ecdsa_sign` passes it
  to `secp256k1_nonce_function_rfc6979` as the `extra_entropy` argument. Upstream
  mixes it using the `secp256k1_rfc6979_hmac_sha256` keydata-based structure:
  `keydata = key32 || msg32 || algo16("ECDSA") || extra32` (112 bytes), which is
  hashed into the HMAC-DRBG state as a key material block.
- **Shim behavior:** When `ndata` is non-NULL, the shim calls
  `ct::ecdsa_sign_hedged(msg, key, ndata)`. Our hedged nonce uses RFC6979 Section 3.2
  with extra data in the HMAC **message** (not the key material):
  `K = HMAC(K0, V || 0x00 || x || h1 || extra)` (129 bytes). This produces valid
  signatures but different nonce values than upstream libsecp256k1 for the same inputs.
- **Reason:** The hedged signing path was designed for forward-secrecy/DPA resistance
  and uses a cryptographically equivalent (but not byte-identical) structure.
- **Impact:** Bitcoin Core's R-grinding loop (`CKey::Sign()`) calls `secp256k1_ecdsa_sign`
  with increasing `extra_entropy` counter bytes. Our shim produces **valid** signatures
  on each iteration (verify passes), but the specific `(r, s)` values differ from
  upstream. The final (low-S) signature accepted by the loop is cryptographically correct;
  only the byte representation differs. No consensus impact: script validation accepts
  any valid (r,s) that satisfies `r,s ∈ [1,n-1]` and DER encoding.
- **Fix available:** Build the shim with `-DSECP256K1_SHIM_RFC6979_COMPAT=ON` to enable
  `rfc6979_nonce_libsecp_compat`, which appends the 16-byte `ECDSA` algo16 tag and passes
  `ndata` directly — producing byte-identical nonces to upstream libsecp256k1. Trade-off:
  the hedged nonce's OS-CSPRNG fault-attack resistance is not available in compat mode.
- **Tracking:** SHIM-P3-006. Functional test:
  `audit/test_regression_shim_rgrind_functional.cpp` RGF-1..4 (valid sig across 32 iterations).
  `audit/test_regression_shim_rfc6979_compat.cpp` — rfc6979_nonce_libsecp_compat determinism
  and signing correctness.

---

### secp256k1_musig_pubkey_agg — session map cap

- **Upstream behavior:** Unlimited concurrent MuSig2 sessions; all state stored inline in opaque structs.
- **Shim behavior:** Hard cap at 1024 concurrent sessions (`kMaxKaEntries`). The 1025th
  call returns 0. (Fixed 2026-05-11: previously returned 1 even when cap was hit.)
- **Reason:** The shim uses a global `unordered_map` to associate opaque `secp256k1_musig_keyagg_cache`
  pointers with internal session state (because the opaque struct is too small for variable-length
  data). Unbounded growth is a DoS risk.
- **Impact:** Applications that open >1024 simultaneous unfinished MuSig2 sessions.
  Sessions that complete normally free their slot.
- **Leak risk:** If a caller errors out before completing a session, the map entry persists
  until the cache address is reused. Callers must ensure sessions are completed or abandoned
  cleanly. A future `secp256k1_musig_keyagg_cache_destroy` API could address this.
- **Test:** `audit/test_exploit_shim_musig_ka_cap.cpp` KAC-1..4.
  KAC-4 fills kMaxKaEntries+1 sessions and asserts at least one pubkey_agg returns 0.

---

### secp256k1_musig_session — token-based session reference (process-local-only)

- **Upstream behavior:** `secp256k1_musig_session` is a 133-byte self-contained opaque
  struct. All data is encoded in those bytes; the struct can be copied or checkpointed
  safely within a process.
- **Shim behavior:** The shim's `secp256k1_musig_session` (133 bytes) stores an 8-byte
  `uint64_t` token at byte offset 98. The token is a key into the global session map
  (`g_ka`); `secp256k1_musig_nonce_process` writes it, `secp256k1_musig_partial_sig_agg`
  reads it for O(1) map lookup and removal. The token is opaque and not a heap address.
- **Reason:** The shim uses a global token-keyed map (previously pointer-keyed, fixed 2026-05)
  to associate opaque `secp256k1_musig_keyagg_cache` addresses with internal session state.
  Storing a token rather than a raw pointer eliminates the ASLR-bypass concern present in
  the prior design.
- **Impact:** **The session struct is process-local-only.**
  - Safe: `memcpy` within the same process (token remains valid while the map entry exists).
  - Unsafe: serializing the session to disk/network and deserializing in a new process
    — the token does not resolve to a map entry in the new process. Do not persist sessions.
- **Test:** Any attempt to serialize + deserialize a session across process restart will
  produce a map-miss (returns 0). No differential test is feasible; the divergence is structural.

---

### secp256k1_context_randomize — blinding is per-thread, not per-context (SHIM-THREAD-BLIND)

- **Upstream behavior:** `secp256k1_context_randomize` stores the blinding scalar inside the
  `secp256k1_context` struct. The blinding is strictly per-context: two contexts on the same
  thread each maintain independent blinding scalars. Signing with context A always uses A's
  blinding; signing with context B always uses B's blinding. Multiple threads signing
  concurrently with different contexts do not interfere with each other's blinding state
  (each context's blinding is local to that struct).
- **Shim behavior:** The shim stores the blinding scalar in a `static thread_local BlindingState
  g_blinding` variable (`src/cpu/src/ct_point.cpp:3417`). The per-context blinding seed IS
  stored in `secp256k1_context::blind[]` and `cached_r` / `cached_r_G`, but the _active_
  blinding applied during signing is a thread-local singleton — not the context's own stored
  seed. `ContextBlindingScope` (entered at sign time via `shim_context.cpp:230`) loads the
  context's cached seed into `g_blinding`, uses it for the signing operation, then clears it
  via `clear_blinding()` on scope exit.
  **Consequence:** Two contexts on the same thread used in an interleaved pattern will
  overwrite each other's active blinding. This cannot happen via the shim's synchronous API
  but could happen via callbacks or coroutines. The source contains a comment
  `// DEVIATION FROM LIBSECP CONTRACT` acknowledging this limitation.
- **Reason:** The CT blinding API (`ct::set_blinding` / `ct::clear_blinding` /
  `ct::generator_mul_blinded`) is a global singleton by design — it operates on a thread-local
  scalar pair that applies to all CT operations on that thread. Plumbing per-context blinding
  through the CT layer would require a significant internal API refactor (passing the blinding
  state through every CT call as an explicit parameter). The current design is safe for the
  dominant use pattern: one context per thread, or multiple contexts used sequentially
  (not interleaved within a single call frame).
- **Impact:**
  1. **Single-context-per-thread callers:** No divergence — identical to upstream.
  2. **Multiple-contexts-same-thread, sequential callers:** No divergence. As long as signing
     operations from different contexts do not interleave, each `ContextBlindingScope` loads
     its own context's seed and clears it correctly.
  3. **Interleaved contexts (callbacks, coroutines, C++ co_await):** Potential mismatch.
     If context A's signing path is suspended after entering its `ContextBlindingScope` and
     context B's signing is invoked on the same thread, B will overwrite `g_blinding`; when
     A resumes, it will operate under B's (or no) blinding. This scenario does not arise in
     Bitcoin Core's synchronous signing paths.
- **Tracking:** SHIM-THREAD-BLIND. The source acknowledgement `// DEVIATION FROM LIBSECP
  CONTRACT` is at `src/cpu/src/ct_point.cpp` near the `thread_local BlindingState` definition.
  This divergence is not a security vulnerability for the targeted Bitcoin Core use case
  (one-context-per-thread, synchronous signing), but is a correctness gap for advanced
  multi-context patterns.
- **Planned fix:** Refactor CT blinding to accept an explicit `BlindingState*` parameter,
  thread-local as a fallback only. Deferred — requires changes throughout the CT layer.
- **Test:** `audit/test_regression_shim_thread_blinding.cpp` (to be added) — TBL-1: sign with
  ctx_a then sign with ctx_b on same thread, both must verify. TBL-2: randomize then sign 1000
  times with alternating contexts, all must verify.

---

### secp256k1_musig_nonce_gen — NULL pubkey accepted (SHIM-NEW-005)

- **Upstream behavior:** `secp256k1_musig_nonce_gen` requires the `pubkey` argument to be
  non-NULL when `seckey` is non-NULL (BIP-327 §5 recommends including the pubkey in the
  nonce derivation to prevent cross-key attacks). Upstream libsecp256k1 returns 0 if
  `pubkey` is NULL when `seckey` is non-NULL.
- **Shim behavior:** `pubkey` may be NULL. When NULL, `pub_x` is zeroed and the nonce is
  derived from `session_id32 + seckey + msg` with a zero pubkey contribution. Signing
  still produces a valid signature; the nonce is cryptographically sound but does not
  incorporate the public key.
- **Reason:** Accepting NULL `pubkey` allows callers that do not have the `secp256k1_pubkey`
  struct available at nonce generation time (e.g., those who only have the raw private key
  bytes) to call the function without serializing a pubkey. The risk is marginal: the session
  ID uniqueness requirement is the primary nonce-uniqueness guarantee; pubkey binding is
  defense-in-depth.
- **Impact:** Callers omitting `pubkey` receive a valid but suboptimally-derived nonce.
  Cross-key binding is absent. For Bitcoin Core use (single-signer pubkey always available),
  this is a non-issue.
- **Fix scope:** Document-only. Enforcing non-NULL `pubkey` when `seckey` is present is a
  possible future hardening (low priority).

---

### secp256k1_musig_nonce_gen — keyagg_cache parameter ignored (SHIM-NEW-006)

- **Upstream behavior:** When `keyagg_cache` is non-NULL, `secp256k1_musig_nonce_gen`
  extracts the aggregated public key `Q` from the cache and includes it in the nonce
  derivation, binding the nonce to the specific key-aggregation context. This provides
  an additional domain-separation guarantee against nonce reuse across different MuSig2
  key aggregations.
- **Shim behavior:** The `keyagg_cache` parameter is silently ignored (declared as
  `/*keyagg_cache*/`). The aggregated public key `agg_x` is left zeroed in the nonce
  derivation. The nonce is still derived from `session_id32`, `seckey`, `pubkey`, and
  `msg` — which is sufficient for nonce uniqueness when `session_id32` is fresh random.
- **Reason:** The shim's `musig2_nonce_gen` internal API does not accept the aggregated
  key as a parameter; adding it would require a C++ API change. The missing binding only
  matters if the same `(seckey, session_id32, pubkey, msg)` tuple is reused across
  different key-aggregation contexts — an extremely rare accident in practice, and
  prevented by the session ID freshness requirement.
- **Fix scope:** Document-only. Adding `agg_x` extraction from the cache token is a
  medium-priority improvement but not a security regression for the primary use case.
- **Test:** No differential test currently exists for this path.
