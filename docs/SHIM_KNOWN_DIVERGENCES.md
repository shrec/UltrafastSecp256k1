# Libsecp256k1 Shim — Known Behavioral Divergences

This document lists every intentional behavioral difference between the
UltrafastSecp256k1 libsecp256k1 shim and upstream libsecp256k1.

**An unlisted divergence is a bug.** If you discover shim behavior that differs
from libsecp256k1 and it is not documented here, it must either be fixed or added
to this file before any PR.

For the complete compatibility test matrix see `compat/libsecp256k1_shim/tests/`.

---

## secp256k1_context_preallocated_* — placement-new semantics

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
  correctly in the caller's buffer without requiring heap allocation.
- **Impact:** None for correct callers. The size returned by `preallocated_size` may
  differ from upstream (upstream may include precomputed table space in certain
  configurations; ours does not — precomputed tables are globally shared). Callers
  must use the shim's own `preallocated_size` return value, not a size hard-coded
  from upstream libsecp256k1.
- **Test:** `audit/test_regression_shim_preallocated_ctx.cpp` PAC-1..6.

---

## secp256k1_schnorrsig_verify with msglen != 32 — ShimSchnorrCache bypassed (SHIM-007)

- **Upstream behavior:** libsecp256k1 `secp256k1_schnorrsig_verify` supports variable-length
  messages with full lift_x computation on every call (no lift_x caching in the reference impl).
- **Shim behavior:** For `msglen == 32`, the shim uses `ShimSchnorrCache` to amortize the
  lift_x and GLV table build across repeated verifications of the same pubkey. For
  `msglen != 32`, the cache is bypassed and lift_x runs on every call — matching upstream
  latency but not achieving the steady-state speedup that 32-byte message paths can achieve.
- **Reason:** The ShimSchnorrCache keys on the full 32-byte pubkey x-coordinate to identify
  cache entries. The `msglen != 32` code path uses the existing direct-verify overload which
  does not consult the cache. This is a performance inconsistency, not a correctness or
  security issue.
- **Impact:** Callers using variable-length Schnorr messages (e.g., non-BIP-340 protocols)
  will not benefit from lift_x amortization. Production Bitcoin validation only uses 32-byte
  messages, so this does not affect the ConnectBlock benchmark.
- **Test:** No dedicated differential test — behavior is equivalent to upstream on this path.

---

## secp256k1_musig_pubnonce_serialize / pubnonce_parse / nonce_agg / nonce_process — `ctx` ignored

- **Upstream behavior:** These four MuSig2 functions accept a `secp256k1_context*`
  argument but do not require it to be non-NULL in upstream libsecp256k1-zkp at
  the version targeted by this shim. The argument is reserved for forward
  compatibility but is currently unused.
- **Shim behavior:** The `ctx` parameter is declared `/*ctx*/` (unused) in all four.
  A NULL `ctx` will NOT fire the illegal-callback path on these functions; only
  the explicit argument checks apply.
- **Reason:** Matches upstream. These calls are pure data-transformation operations
  (serialize/parse/aggregate nonces, compute session state) that do not consume
  context flags, signing permission, or blinding.
- **Impact:** Callers relying on a NULL-ctx illegal-callback firing for these
  specific functions will not see it. The illegal-callback contract holds for
  every other shim function that uses the context.
- **Previously listed functions now FIXED (2026-05-25):** `secp256k1_musig_pubkey_get`,
  `secp256k1_musig_partial_sig_verify`, `secp256k1_musig_partial_sig_serialize`,
  and `secp256k1_musig_partial_sig_parse` previously shared this behavior — all four
  now use `SHIM_REQUIRE_CTX(ctx)` and fire the illegal callback on NULL ctx.
  `secp256k1_musig_pubkey_ec_tweak_add` and `secp256k1_musig_pubkey_xonly_tweak_add`
  are also fixed — see SHIM-MUSIG-CTX-001 below.
- **Test:** Covered indirectly by `audit/test_regression_shim_security_v7.cpp`
  (musig2 partial sign + verify round-trip) and `audit/test_exploit_shim_musig_*`
  family. A targeted differential test against upstream libsecp256k1-zkp can be
  added if a divergence ever appears.

---

## secp256k1_ecdh — private key >= curve order rejected

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

## secp256k1_ecdh — off-curve pubkey rejected

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

## secp256k1_ecdsa_sign — ndata/extra_entropy nonce divergence (SHIM-P3-006)

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
  Compat mode also works correctly when `ndata == nullptr` (no extra entropy case).
- **Tracking:** SHIM-P3-006. Functional test:
  `audit/test_regression_shim_rgrind_functional.cpp` RGF-1..4 (valid sig across 32 iterations).
  `audit/test_regression_shim_rfc6979_compat.cpp` — rfc6979_nonce_libsecp_compat determinism
  and signing correctness.

---

## secp256k1_ecdsa_sign — custom nonce function

- **Upstream behavior:** Any non-NULL `noncefp` is called unconditionally; the
  caller-supplied function generates the `k` value.
- **Shim behavior:** If `noncefp` is non-NULL and is not `secp256k1_nonce_function_rfc6979`
  or `secp256k1_nonce_function_default`, the function returns 0 immediately.
- **Reason:** The shim uses its own RFC 6979 nonce derivation internally (CT path)
  and cannot safely delegate to an arbitrary callback. Fail-closed is correct: a caller
  that depends on a specific nonce function (hardware wallet, libwally custom nonce) is
  told explicitly that the operation failed, rather than silently receiving RFC 6979 output.
- **Impact:** Callers with custom nonce functions. The common case (NULL, rfc6979,
  default) is unaffected. Bitcoin Core passes NULL noncefp — unaffected.
- **Test:** `test_shim_ecdsa_custom_nonce_rejected` in `compat/libsecp256k1_shim/tests/`.

---

## secp256k1_nonce_function_rfc6979 / secp256k1_nonce_function_default

- **Upstream behavior:** These function pointers generate RFC 6979 nonce bytes when
  called directly.
- **Shim behavior:** Both pointers are stubs that return 0 (failure) and write nothing.
  They are exported for ABI symbol compatibility only. The shim never calls them.
- **Reason:** The shim's signing path uses its own RFC 6979 implementation. Returning 0
  from a stub that writes nothing is correct: a caller that invokes these pointers
  directly gets explicit failure rather than empty output with a success code (SC-08).
- **Impact:** Any caller using these pointers as standalone hash primitives (rare).

---

## secp256k1_nonce_function_bip340

- **Upstream behavior:** Generates BIP-340 aux-entropy-mixed nonce bytes when called.
- **Shim behavior:** Returns 0 (failure) and writes nothing. Exported for ABI compatibility.
- **Reason:** Same as rfc6979 stubs above. The shim handles BIP-340 nonce internally
  via the `aux_rand32` parameter of `secp256k1_schnorrsig_sign32`.
- **Impact:** Any caller using this pointer as a standalone hash primitive.

---

## secp256k1_schnorrsig_verify — variable-length message

- **Upstream behavior (v0.4+):** Accepts any `msglen`. Computes BIP-340 challenge as
  `H_BIP0340/challenge(R.x || P.x || msg[:msglen])`. Empty message (`msglen == 0`)
  and NULL msg with `msglen == 0` are accepted.
- **Shim behavior (current):** Accepts any `msglen` — matches upstream. For `msglen == 32`
  uses the optimized fixed-length path; other lengths use the generic tagged-hash path.
  `msg == NULL` with `msglen > 0` fires illegal callback and returns 0.
- **Reason:** Fixed (was: rejected msglen != 32). The core library now exposes
  `schnorr_verify(pubkey_x32, msg, msglen, sig)` using `schnorr_challenge_scalar_varlen`.
- **Impact:** No divergence. Callers using variable-length Schnorr (FROST, Lightning,
  sign_custom with msglen != 32) work correctly through the shim.

---

## secp256k1_xonly_pubkey_tweak_add / tweak_add_check / keypair_xonly_tweak_add

- **Upstream behavior:** NULL ctx triggers illegal callback (default: abort).
- **Shim behavior:** NULL ctx triggers illegal callback via `SHIM_REQUIRE_CTX` (abort).
  **Fixed 2026-05-11.** Previously `(void)ctx;` — NULL ctx would silently succeed.
- **Impact:** None for correct callers. Defense against accidental NULL ctx in Taproot paths.

---

## secp256k1_keypair_create — context flag check

- **Upstream behavior (≤v0.5):** Requires SIGN context; VERIFY-only returns 0.
  **Upstream behavior (v0.6+):** CONTEXT_NONE accepted for all operations.
- **Shim behavior:** Uses `ctx_can_sign()` which accepts SIGN and CONTEXT_NONE
  (libsecp v0.6+ compat), rejects VERIFY-only, and aborts on NULL. **Fixed 2026-05-11.**
- **Impact:** Callers using VERIFY-only context to call keypair_create. Correct behavior
  per libsecp v0.6+ (Bitcoin Core v26+ uses CONTEXT_NONE or SIGN).

---

## secp256k1_context_randomize — seed >= n

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

## secp256k1_schnorrsig_sign_custom — NULL context

- **Upstream behavior:** Fires illegal callback (default: abort) when ctx is NULL.
- **Shim behavior:** Fires illegal callback via explicit NULL check before `schnorr_ctx_can_sign`.
  **Fixed 2026-05-11** (PASS3-008). Previously `schnorr_ctx_can_sign(NULL)` returned false
  without firing the callback — a silent return 0 instead of abort.
- **Impact:** None for correct callers (NULL ctx is always a bug).
- **Test:** Differential test: `secp256k1_schnorrsig_sign_custom(NULL, ...)` — verify callback fires.

---

## secp256k1_ec_pubkey_serialize — invalid flags

- **Upstream behavior:** Fires illegal callback for any flags value other than
  `SECP256K1_EC_COMPRESSED` or `SECP256K1_EC_UNCOMPRESSED`.
- **Shim behavior:** Fires illegal callback and returns 0 for invalid flags.
  **Fixed 2026-05-11** (PASS3-011). Previously garbage flags (e.g. `0xDEAD`) were
  treated as `SECP256K1_EC_UNCOMPRESSED` — silent wrong output.
- **Impact:** None for correct callers (invalid flags are always a bug).
- **Test:** Differential test: `secp256k1_ec_pubkey_serialize(ctx, ..., 0xDEAD)` — verify return 0 + callback fires.

---

## secp256k1_ecdsa_recoverable_signature_parse_compact — r/s zero check

- **Upstream behavior:** Accepts r=0 or s=0 at parse time; rejects at recover time.
- **Shim behavior:** **Fixed 2026-05-21 (PASS3-002).** Now uses `parse_bytes_strict` —
  accepts r=0 or s=0 at parse time, matching upstream libsecp256k1. Rejection of degenerate
  r/s values happens at `secp256k1_ecdsa_recover` time.
- **Previous shim behavior:** Used `parse_bytes_strict_nonzero` — rejected r=0 or s=0 at
  parse time (stricter than upstream, divergence).
- **Reason for fix:** Behavioral parity with upstream libsecp256k1; the previous divergence
  had no security benefit (zero r/s signatures are always invalid; rejecting earlier vs later
  makes no difference in practice).
- **Impact:** No impact for callers using valid signatures. Callers that parse r=0 or s=0
  compact bytes now get parse return 1 (like libsecp) instead of 0.
- **Test:** `compat/libsecp256k1_shim/tests/test_shim_recovery_and_noncefp.cpp` REC-1..4.

---

## secp256k1_musig_pubkey_agg — session map cap

- **Upstream behavior:** Unlimited concurrent MuSig2 sessions; all state stored inline in opaque structs.
- **Shim behavior:** Hard cap at 1024 concurrent sessions (`kMaxKaEntries`). The 1025th
  call returns 0. **Fixed 2026-05-11:** previously returned 1 even when cap was hit.
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

## secp256k1_ecdsa_sign / secp256k1_ecdsa_sign_recoverable / secp256k1_schnorrsig_sign_custom — custom nonce function

- **Upstream behavior:** Any `noncefp` / `extraparams->noncefp` is dispatched (called with the
  message, key, and counter to produce a nonce).
- **Shim behavior:** Only the standard nonce functions are accepted (NULL, `rfc6979`, `default`
  for ECDSA; NULL, `bip340` for Schnorr). **Updated 2026-05-21 (PASS3-001):** any other non-NULL
  `noncefp` now fires the illegal callback with a descriptive message before returning 0.
  Previously returned 0 silently (divergence from upstream callback behavior).
- **Reason:** The shim uses RFC 6979 / BIP-340 nonce generation internally and cannot forward
  an arbitrary nonce function. Fail-closed so callers relying on a specific nonce function see
  an error rather than silently receiving RFC 6979 output.
- **Impact:** Callers with custom nonce functions. Bitcoin Core uses NULL (RFC 6979 default) —
  unaffected. Callers with callbacks installed now receive the callback notification.
- **Test:** `compat/libsecp256k1_shim/tests/test_shim_recovery_and_noncefp.cpp` NFP-1..3.

---

## secp256k1_musig_session — internal raw pointer (process-local-only)

- **Upstream behavior:** `secp256k1_musig_session` is a 133-byte self-contained opaque
  struct. All data needed to reconstruct the session is encoded in those bytes. The struct
  can be copied, memcpy'd, or checkpointed safely within a process.
- **Shim behavior:** The shim's `secp256k1_musig_session` (133 bytes) stores an 8-byte
  raw pointer to the associated `secp256k1_musig_keyagg_cache` at byte offset 98.
  This pointer is written by `secp256k1_musig_session_init` and read by `partial_sig_agg`.
- **Reason:** The shim uses a global token-keyed map to associate opaque cache pointers
  with internal state. The raw pointer allows O(1) map removal in `partial_sig_agg`.
- **Impact:** **The session struct is process-local-only.**
  - Safe: `memcpy` within the same process (pointer remains valid).
  - Unsafe: serializing the session to disk/network and deserializing in a new process
    — the pointer becomes dangling. Do not checkpoint or persist sessions.
  - The pointer at offset 98–105 also leaks a heap address (ASLR bypass) in any
    inter-process context (hypothetical only — MuSig2 sessions should never cross
    process boundaries).
- **Planned fix:** Store a token/index instead of a raw pointer (v2 scope, MED-3).
- **Test:** Any attempt to serialize + deserialize a session across process restart
  will produce a dangling-pointer UB. No differential test is feasible; the divergence
  is structural.

---

## secp256k1_context_clone — blinding seed copied verbatim

- **Upstream behavior:** `secp256k1_context_clone` creates a deep copy of the context,
  including the blinding seed set by `secp256k1_context_randomize`. The clone shares
  the same blinding scalar as the original until re-randomized.
- **Shim behavior:** Identical to upstream — `secp256k1_context_clone` copies all
  context fields including the stored blinding seed. The clone starts with the same
  blinding state as the original. This is intentional and matches libsecp256k1 exactly.
- **Reason:** There is no security benefit to generating a new blinding seed in the clone;
  the original and clone are equally trusted contexts. Re-randomizing each clone separately
  is the caller's responsibility, as it is in libsecp256k1.
- **Impact:** None for correct callers. Callers who want independent blinding in the clone
  must call `secp256k1_context_randomize` on the cloned context with fresh randomness.
- **Test:** N/A — behavior is identical to upstream; no differential test needed.

---

## secp256k1_ecdsa_signature_parse_der — rejects r=0 or s=0

- **Upstream behavior:** Accepts r=0 or s=0 in DER signatures at parse time; rejects
  them at verify time (r=0 makes the group element undefined; s=0 makes the equation
  degenerate).
- **Shim behavior:** Uses `parse_bytes_strict_nonzero` in the DER parser — rejects
  r=0 or s=0 at parse time.
- **Asymmetry:** The compact signature parser (`signature_parse_compact`) uses
  `parse_bytes_strict` and does NOT reject r=0/s=0 at parse time, matching upstream.
  DER parse is stricter than compact parse — this asymmetry is intentional (defense-in-depth
  for the DER path, where zero-value fields indicate malformed input).
- **Reason for the divergence (per CLAUDE.md Canonical Rule 3):** the failure modes are
  observationally equivalent for any cryptographically valid caller — a signature with
  r=0 or s=0 fails verify on both upstream and the shim. The only place the difference
  surfaces is in code that distinguishes between parse failure and verify failure on
  the same input. For Bitcoin Core's consensus path that means: parse failure returns
  `SCRIPT_ERR_SIG_DER` while verify failure (after parse OK) returns
  `SCRIPT_ERR_SIG_VERIFY`. Bitcoin Core script validation treats both as a hard reject,
  so the divergence is not consensus-observable in practice. Any caller that DOES want
  to know whether a signature is structurally well-formed before verification should
  use the compact parser (which matches upstream) or accept the DER-strictness contract.
- **Impact:** Callers parsing DER-encoded ECDSA signatures with r=0 or s=0 (astronomically
  rare in practice; all such signatures are cryptographically invalid and any caller
  that gets one is going to fail verify anyway). Probability per random 256-bit r/s:
  ≤ 2^-128 each. No mainnet/testnet signature with r=0 or s=0 has ever been recorded.
- **Test:** `test_shim_der_zero_r.cpp` in `compat/libsecp256k1_shim/tests/` —
  calls `secp256k1_ecdsa_signature_parse_der` with a hardcoded DER blob encoding r=0
  and verifies return value is 0. Upstream libsecp returns 1 (passes to verify which
  then rejects).
- **Tracking:** Review finding `P1-SEC-003` / `RED-003` — kept as a documented
  divergence rather than a fix because the no-observable-consensus-change argument
  above stands. If a future PR needs full parse-time symmetry with upstream, the fix is
  one-line: switch the DER parser at `compat/libsecp256k1_shim/src/shim_ecdsa.cpp:248`
  from `Scalar::parse_bytes_strict_nonzero` to `Scalar::parse_bytes_strict`, and let
  the verify path reject the zero scalars.

---

## secp256k1_ecdsa_recover — NULL pubkey/sig/msghash fires illegal callback (SHIM-NEW-001)

- **Upstream behavior:** NULL required arguments (`pubkey`, `sig`, `msghash32`) trigger
  the illegal callback (default: abort).
- **Previous shim behavior:** Returned 0 silently without firing the callback.
- **Current shim behavior:** Fires `secp256k1_shim_call_illegal_cb` before returning 0.
  **Fixed 2026-05-12** (SHIM-NEW-001).
- **Reason:** Matches libsecp256k1 contract. Silent return 0 on NULL args is a divergence
  that could mask caller bugs.
- **Impact:** None for correct callers. A caller that passes NULL args now gets the same
  abort behavior as with libsecp256k1.
- **Test:** `test_shim_recover_null_pubkey_fires_callback` — pass NULL pubkey, verify
  callback flag is set.

---

## secp256k1_schnorrsig_sign_custom — NULL sig64/keypair fires illegal callback (SHIM-NEW-003)

- **Upstream behavior:** NULL `sig64` or `keypair` triggers the illegal callback (default: abort).
- **Previous shim behavior:** Returned 0 silently without firing the callback.
- **Current shim behavior:** Fires `secp256k1_shim_call_illegal_cb` before returning 0.
  **Fixed 2026-05-12** (SHIM-NEW-003).
- **Reason:** Matches libsecp256k1 contract.
- **Impact:** None for correct callers.
- **Test:** `test_shim_schnorrsig_null_sig_fires_callback` — pass NULL sig64, verify callback fires.

---

## secp256k1_ecdh — uses Scalar::from_bytes (silent mod-n reduction) (SHIM-NEW-002) [HISTORICAL/RESOLVED]

> **This entry is HISTORICAL and describes behavior that no longer exists.**
> The shim was corrected to use `Scalar::parse_bytes_strict_nonzero()` (rejects zero and >= n),
> matching upstream `secp256k1_scalar_set_b32_seckey` semantics. See SHIM-A06 below for the
> current accurate description. Entry retained for historical reference only. Corrected 2026-05-20.

---

## secp256k1_ecdsa_verify -- high-S signature acceptance (SEC-007 — NOT A DIVERGENCE)

- **Upstream behavior:** libsecp256k1 `secp256k1_ecdsa_verify` does NOT internally normalize
  before verifying. Both low-S and high-S signatures pass `secp256k1_ecdsa_verify` (the
  mathematical ECDSA equation is valid for either s value). BIP-62 low-S enforcement in
  Bitcoin Core is done separately at the script validation layer, not inside libsecp.
- **Shim behavior:** Identical — the shim passes signatures directly to
  `secp256k1::ecdsa_verify` without normalizing. Both low-S and high-S signatures verify.
- **Verdict:** No behavioral divergence. A prior version of this entry incorrectly claimed
  that upstream "internally normalizes" before verify. That claim was factually wrong.
  SEC-007 is closed: the behavior is identical between shim and upstream.
- **Impact:** None — behavior matches upstream exactly.
- **Test:** `audit/test_regression_shim_high_s_verify.cpp` confirms shim behavior matches
  upstream: high-S signatures verify successfully in both implementations.

---

## All sign functions -- wrong-flag context (VERIFY-only passed to sign) (SHIM-001)

- **Upstream behavior:** libsecp256k1 fires the illegal callback (default: abort) when a
  VERIFY-only context is passed to a sign function, then returns 0.
- **Shim behavior:** **Fixed 2026-05-21 (SHIM-001).** `ctx_can_sign()` now fires
  `secp256k1_shim_call_illegal_cb(ctx, "sign: context not initialized for signing")` before
  returning false when a VERIFY-only context is detected. The calling sign function then returns 0.
  This matches libsecp256k1 behaviour exactly: callback fires, then returns error.
- **Previous shim behavior:** Returned 0 silently without firing the illegal callback.
- **Reason:** Matches upstream libsecp256k1 contract. Fuzzing harnesses and callers relying
  on callback invocation for error detection now receive the callback signal correctly.
- **Impact:** None for correct callers. Callers using a VERIFY-only context with sign functions
  now trigger the callback (default: abort), matching upstream behaviour.
- **Test:** Differential test -- pass VERIFY-only ctx to secp256k1_ecdsa_sign, verify callback fires.

---

## secp256k1_ecdsa_verify_batch — large-batch curve membership check (CA-001, RESOLVED 2026-05-13)

- **Previous shim behavior (PERF-004, now removed):** The large-batch ECDSA path (n >= 8)
  omitted the `y²=x³+7` curve membership check that the small-batch path (n < 8) performed.
  Comment stated: "from_affine + Point arithmetic rejects infinity points" and "incorrect verify
  result, caught below." This was a P1 security inconsistency — an invalid-curve point could
  pass through `from_affine` (which does not validate the curve equation) and enter the batch MSM.
- **Current behavior:** Both small-batch and large-batch paths now perform the `y²=x³+7` check
  before calling `Point::from_affine`. Invalid-curve points are rejected at input time, not
  downstream. **Fixed 2026-05-13 (CA-001).**
- **Upstream behavior:** libsecp256k1 does not expose a batch ECDSA verify API; this is an
  additive shim-only function with no upstream divergence to track.
- **Impact:** None for callers with valid public keys (all keys from `secp256k1_ec_pubkey_parse`
  are on-curve). Only affects callers who bypass `ec_pubkey_parse` and write the opaque struct
  directly with an off-curve point.
- **Test:** Covered by `audit/test_regression_shim_security_v8.cpp` `test_ecdh_pubkey_off_curve`
  pattern; a targeted batch-ECDSA off-curve test should be added in a follow-up.

---

## secp256k1_schnorrsig_verify -- thread-local xonly-pubkey cache (NEW-SHIM-004)

- **Upstream behavior:** No caching.  Every `secp256k1_schnorrsig_verify` call
  re-runs `lift_x` (sqrt) and rebuilds GLV precomputation tables for the supplied
  x-only pubkey.
- **Shim behavior:** A 256-slot, thread-local, FNV-1a-fingerprinted cache stores
  the lifted point + precomputed tables.  Warm cache hits skip `lift_x` and the
  GLV rebuild (~1,954 ns saved per hit on the ConnectBlock workload).
- **Reason:** ConnectBlock-style hot paths (Bitcoin Core block validation) re-verify
  the same set of x-only pubkeys many times within a small window.  Caching the
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
- **Note on the `verify_precomp` API:** callers that need deterministic warm-cache
  behaviour (no eviction) should use `secp256k1_xonly_pubkey_parse_precomp` +
  `secp256k1_schnorrsig_verify_precomp` from `secp256k1_schnorrsig.h`, which
  exposes the prebuilt object explicitly and bypasses the global cache.
- **Test:** Planned — `audit/test_regression_shim_schnorr_cache_collision.cpp`
  will exercise the eviction-and-rebuild path: verify key A, fill the cache by
  verifying ≥256 distinct keys (forcing at least one slot collision), re-verify
  key A and assert it still returns 1.  Correctness must hold after eviction;
  only the per-call latency changes.  Current coverage is indirect via the
  unique-pubkey ConnectBlock workload exercised in `bench_unified` and the
  high-diversity verify path in `audit/test_exploit_schnorr_verify_*`.

---

## secp256k1_schnorrsig_verify -- null msg without firing illegal callback (SHIM-003)

- **Status:** FIXED 2026-05-21. The shim now matches upstream behavior for `msg=NULL` with `msglen==0`
  (upstream allows NULL msg when msglen==0; the shim previously fired the illegal callback in all
  NULL-msg cases). NULL msg with msglen>0 still fires the illegal callback (correct).
- **Upstream behavior:** `secp256k1_schnorrsig_verify` allows `msg=NULL` when `msglen==0`
  (zero-length message is a valid BIP-340 construct). For `msg=NULL` with `msglen>0`, fires
  the illegal callback (default: abort) then returns 0.
- **Shim behavior (post-fix):** Matches upstream. NULL msg is permitted when msglen==0. NULL msg
  with msglen>0 fires the illegal callback and returns 0.
- **Residual divergence:** None for the msglen==0 case. For msglen==32 with msg=NULL, both upstream
  and shim fire the illegal callback.
- **Test:** `tests/test_shim_security_edge_cases.cpp` SHIM-003 test: NULL msg + msglen=0 must NOT
  fire callback and must return 0 (valid vacuous-verify result).

---

## secp256k1_ecdh -- parse_bytes_strict_nonzero (SHIM-A06)

- **Upstream behavior:** `secp256k1_ecdh` parses the scalar with `secp256k1_scalar_set_b32_seckey`
  which rejects zero and scalars >= n (returning 0).
- **Shim behavior:** Uses `Scalar::parse_bytes_strict_nonzero()` — identical semantics (rejects
  zero and >= n). **This entry in SHIM_KNOWN_DIVERGENCES.md was previously incorrect** (it claimed
  the shim used `from_bytes` which silently reduces mod n). The code uses `parse_bytes_strict_nonzero`.
- **Impact:** No divergence — behavior matches upstream. Entry corrected 2026-05-20.
- **Test:** `audit/test_exploit_shim_musig_secnonce.cpp` — tests strict nonzero parse rejection.

---

## secp256k1_musig_nonce_process -- raw pointer ASLR information leak (SHIM-A07)

- **Upstream behavior:** libsecp256k1 stores a pointer to `secp256k1_musig_keyagg_cache` inside
  the session struct. Raw pointer value leaks ASLR offset if the session struct is serialized
  or exposed across trust boundaries.
- **Shim behavior:** Same — `sess_stash_cache_ptr` stores the raw `secp256k1_musig_keyagg_cache*`
  at a fixed offset in the session opaque bytes.
- **Reason:** This matches libsecp256k1 behavior exactly. The pointer is only valid within the
  same process and must not be serialized.
- **Impact:** If callers serialize `secp256k1_musig_session` across process boundaries (which they
  must not do per the API contract), the raw pointer leaks ASLR. Callers that use the session
  within a single process are unaffected.
- **Test:** `audit/test_exploit_shim_musig_secnonce.cpp` — validates session lifecycle within
  a single process. Cross-process serialization is out of scope.

---

## pubkey_data_to_point (internal helper) — no curve membership check (SEC-004)

- **Status:** RESOLVED (2026-05-21). The curve membership check `y²=x³+7 mod p` has been
  added to `pubkey_data_to_point` via `shim_pubkey_helpers.hpp`. All callers that use
  `pubkey_data_to_point` now get the check. Off-curve input causes the function to return
  `Point::infinity()`, which all callers treat as failure via `is_infinity()` checks.
- **Previous risk:** If this helper were called without a prior curve check, an off-curve
  point would be accepted silently, potentially enabling invalid-point attacks where
  a small-order subgroup point leaks private key bits modulo the subgroup order.
- **Tracking:** SEC-004
- **Impact:** None for callers with valid on-curve public keys. Off-curve inputs are now
  detected and rejected at the helper boundary rather than at individual call sites.

---

## secp256k1_ecdsa_sign — ndata nonce derivation differs from libsecp256k1 (SHIM-010)

- **Upstream behavior:** Uses RFC6979(msg, seckey, alg="", extra_entropy=ndata) to derive nonce.
- **Shim behavior:** Uses hedged nonce derivation (HMAC-DRBG with ndata as auxiliary input), not byte-identical RFC6979+extra_data. Produces a different but cryptographically valid signature for the same inputs when ndata != NULL.
- **Reason:** Hedged nonce provides defense against bad RNG without requiring exact RFC6979 parity.
- **Impact:** Signatures differ byte-for-byte from libsecp256k1 when ndata is used. R-grinding terminates correctly and produces a valid low-S/low-R signature. Bitcoin Core production path uses noncefp=NULL (unaffected).
- **Test:** Verify R-grinding terminates within bounded iterations; confirm produced signature is valid (not necessarily identical to libsecp output).

---

## secp256k1_ec_pubkey_cmp — NULL ctx now fires illegal callback (SHIM-005-FIX)

- **Upstream behavior:** NULL ctx fires the illegal callback (default: abort) before any other
  argument check. The pubkey comparison result is never computed if ctx is NULL.
- **Previous shim behavior:** NULL ctx was passed directly to `secp256k1_shim_call_illegal_cb`
  only when pubkeys were null — not when ctx itself was null. This meant a NULL ctx could
  silently reach `secp256k1_ec_pubkey_serialize` with a NULL context, a separate bug.
- **Current shim behavior (post-fix):** Explicit `!ctx` guard fires the illegal callback
  and returns 0 before any pubkey or serialize logic. **Fixed 2026-05-21 (SHIM-005-FIX).**
- **Reason:** Matches upstream libsecp256k1 contract — NULL ctx is always a programming
  error and must fire the callback.
- **Impact:** None for correct callers. Callers that accidentally pass NULL ctx now get the
  same abort behavior as with upstream libsecp256k1.
- **Test:** `test_shim_security_edge_cases_run` — calls `secp256k1_ec_pubkey_cmp` with NULL
  ctx and asserts the illegal callback fires.

---

## secp256k1_musig_partial_sig_agg — all-zero check uses CT accumulator (SHIM-MUSIG-CT)

- **Upstream behavior:** No all-zero check — returns the aggregated value unconditionally.
- **Shim behavior:** Checks whether the 64-byte aggregated signature is all-zero (degenerate
  aggregation result) and returns 0 if so. **Updated 2026-05-21:** the check now uses a
  branchless OR-accumulator (`uint32_t nonzero = 0; for(i) nonzero |= sig[i]`) instead of
  the previous early-exit `break` loop.
- **Reason:** The early-exit loop leaked information about the signature bytes via
  branch-predictor and cache timing — the loop terminated faster when a non-zero byte
  appeared early, revealing the index of the first non-zero byte. The OR-accumulator
  runs all 64 bytes unconditionally, eliminating this side-channel.
- **Impact:** No behavioral change for callers — the return value is the same. Timing
  behavior is now data-independent for the all-zero check.
- **Test:** `test_regression_shim_security_v7_run` (musig2 aggregate round-trips).

---

## secp256k1_musig_nonce_gen — extra_input32 silently ignored (SHIM-NONCEGEN-001)

- **Upstream behavior:** `secp256k1_musig_nonce_gen` accepts an `extra_input32` parameter
  that is mixed into the nonce derivation as additional entropy (defense-in-depth).
- **Shim behavior:** The `extra_input32` parameter is accepted in the function signature but
  not forwarded to the internal `frost_sign_nonce_gen` / `musig2_nonce_gen` primitives.
  The parameter is silently ignored.
- **Reason:** The shim's internal nonce generation API does not expose an `extra_input32`
  parameter. Adding support requires a non-trivial API change to the nonce derivation path.
- **Impact:** Callers relying on `extra_input32` for additional entropy get correct nonces
  (RFC 6979 / BIP-340 hedged) but without the extra input mixed in. For production Bitcoin
  Core usage (extra_input32 = NULL or ignored), there is no difference.
- **Test:** `audit/test_regression_musig_noncegen_extra_input.cpp` — behavioral freeze test:
  verifies that `extra_input32` is silently ignored (pubnonces are identical with NULL vs non-NULL
  extra_input32, and identical for two distinct non-NULL extra_input32 values). Sub-tests NCI-1..3
  also scan `shim_musig.cpp` for the `SHIM-NONCEGEN-001` marker. The test is `advisory=true` in
  the unified runner (requires shim) and is designed to **fail** when SHIM-NONCEGEN-001 is fixed
  (diverging pubnonces = correct signal to remove the advisory flag and promote to mandatory).

---

## secp256k1_schnorrsig_verify_batch — msglen != 32 silently rejects (SHIM-BATCH-001)

- **Upstream behavior:** libsecp256k1 does not expose `secp256k1_schnorrsig_verify_batch`
  in the standard API; this is a shim-only extension.
- **Shim behavior:** The batch verify shim (`secp256k1_schnorrsig_verify_batch`) requires
  all messages to have `msglen == 32`. If `msglen != 32`, the illegal callback IS fired
  (`secp256k1_shim_call_illegal_cb`) and the function returns 0.
  (Note: an earlier version of this document incorrectly stated the callback was not fired.
  The code has always fired it — SHIM-BATCH-001 doc corrected 2026-05-21.)
- **Reason:** The batch code path uses fixed 32-byte message processing for performance.
  Variable-length batch verify requires a different internal API path not yet exposed.
- **Impact:** Callers using variable-length messages in batch verify will get a silent
  rejection (return 0) rather than an error callback. Single-message verify (`secp256k1_schnorrsig_verify`)
  supports any msglen correctly.
- **Test:** TODO — `secp256k1_schnorrsig_verify_batch` with msglen=0 and msglen=64,
  assert returns 0.

---

## secp256k1_tagged_sha256 — msg=NULL with msglen=0 now allowed (SHIM-A03 fixed 2026-05-21)

- **Upstream behavior:** libsecp256k1 `secp256k1_tagged_sha256` allows `msg=NULL` when
  `msglen=0` — a zero-length message is valid input and produces a well-defined tagged hash
  of the empty string.
- **Previous shim behavior (divergence, now fixed):** The shim previously rejected `msg=NULL`
  unconditionally (`if (!hash32 || !tag || !msg) return 0`), causing false rejection of
  zero-length messages.
- **Current shim behavior (fixed):** `msg=NULL` with `msglen=0` is now accepted and produces
  the correct tagged hash of the empty message. `msg=NULL` with `msglen>0` fires the illegal
  callback and returns 0, matching libsecp256k1 behavior.
- **Reason:** Zero-length messages are valid per the SHA256 spec and libsecp256k1 does not
  restrict them. The previous guard was overly strict.
- **Impact:** Callers hashing empty messages now work correctly. No behavioral change for
  callers passing non-NULL msg or msglen=0 with non-NULL msg.
- **Test:** Covered by shim null-arg tests. A dedicated zero-length message round-trip test
  should be added to `compat/libsecp256k1_shim/tests/`.

---

## secp256k1_musig_pubkey_ec_tweak_add / secp256k1_musig_pubkey_xonly_tweak_add — NULL ctx now fires illegal callback (SHIM-MUSIG-CTX-001, FIXED 2026-05-25)

- **Upstream behavior:** NULL ctx fires the illegal callback (default: abort). Non-NULL ctx is
  validated for context flags before proceeding.
- **Previous shim behavior:** Both functions declared ctx as `/*ctx*/` — completely discarded.
  NULL ctx silently proceeded to `!keyagg_cache || !tweak32` check without firing the callback.
- **Current shim behavior:** Both functions now call `SHIM_REQUIRE_CTX(ctx)` as their first
  statement. NULL ctx fires the illegal callback (default: abort) and returns 0 — matching
  upstream libsecp256k1 behavior. **Fixed 2026-05-25 (SHIM-MUSIG-CTX-001).**
- **Reason:** Matches upstream libsecp256k1 contract. NULL ctx is always a programming error
  and should fire the callback for all public API functions, not silently return 0.
- **Impact:** None for correct callers. Callers that accidentally pass NULL ctx now get the
  same abort behavior as with upstream libsecp256k1.
- **Test:** Differential test: call `secp256k1_musig_pubkey_ec_tweak_add(NULL, ...)` with a
  valid keyagg_cache and tweak32 — verify callback fires and function returns 0.

---

## secp256k1_ec_pubkey_tweak_mul vs secp256k1_ec_pubkey_tweak_add (zero tweak)

- **Upstream behavior:** `tweak_mul` rejects zero tweak (returns 0); `tweak_add` accepts zero tweak (point unchanged, returns 1)
- **Shim behavior:** Matches upstream — `tweak_mul` uses `parse_bytes_strict_nonzero` (rejects zero); `tweak_add` uses `parse_bytes_strict` (accepts zero)
- **Reason:** Intentional asymmetry matching libsecp256k1 v0.5+ semantics; multiplying by zero always produces infinity (invalid pubkey), while adding zero is a no-op
- **Impact:** Callers must not pass a zero tweak to `tweak_mul`; zero tweak to `tweak_add` is well-defined
- **Test:** `test_differential_tweak_zero` (to be added)

---

## secp256k1_keypair_sec (BIP-340 normalization)

- **Upstream behavior:** libsecp256k1 stores the BIP-340-normalized private key in the keypair (negates if P.y is odd); `keypair_sec` returns the normalized key
- **Shim behavior:** The shim stores the raw private key; negation is applied at sign time. `keypair_sec` returns the raw (possibly non-BIP-340-normalized) key
- **Reason:** Normalization-at-store vs normalization-at-sign is semantically equivalent for signing but differs in what `keypair_sec` returns for odd-Y keys
- **Impact:** If a caller extracts the private key via `keypair_sec` and compares it to the original, they may observe a different value for keys where P.y is odd
- **Test:** `test_differential_keypair_sec` (to be added)

---

## secp256k1_context_set_illegal_callback — NULL ctx silently returns (SHIM-ILLCB-001)

- **Upstream behavior:** When `ctx=NULL`, libsecp256k1 fires the static default illegal
  callback (`secp256k1_default_illegal_callback_fn`) before returning. The callback fires
  with message `"ctx != NULL"`. Default behavior: `abort()`.
- **Shim behavior:** `if (!ctx) return;` — silently returns without invoking any callback.
  No `abort()` occurs. A caller that passes NULL ctx to `secp256k1_context_set_illegal_callback`
  receives no feedback that their argument is invalid.
- **Reason:** The shim has no static default callback object equivalent to
  `secp256k1_default_illegal_callback_fn`. Recreating the upstream behavior requires
  storing a global fallback callback pointer, which adds complexity for a function
  that is almost never called with NULL ctx in practice.
- **Impact:** Callers that pass NULL ctx to this function will not trigger an abort.
  The most likely effect is that a mis-initialized ctx pointer silently passes through,
  leaving the callback unset. Any subsequent operation with that context will use
  the default callback (defined at context creation time). No security impact — this
  is a programming-error path, not a signing or verification path.
- **Tracking:** SHIM-ILLCB-001. Low priority — NULL ctx here is a caller bug, not an
  attacker-controlled input. Bitcoin Core never calls `set_illegal_callback(NULL)`.
- **Test:** None planned — this is a programming-error case with no security impact.

---

## secp256k1_ec_pubkey_parse — NULL pubkey/input silently returns 0 (SHIM-ILLCB-002)

- **Upstream behavior:** NULL `pubkey` or NULL `input` fires the illegal callback
  (default: `abort()`) via `ARG_CHECK`. After the callback, returns 0.
- **Shim behavior:** `if (!pubkey || !input) return 0;` — returns 0 silently without
  invoking the illegal callback. The callback is NOT fired.
- **Reason:** The NULL checks were written as early-exit guards, not as illegal-callback
  dispatches. Matching upstream would require replacing these with
  `secp256k1_shim_call_illegal_cb(ctx, "pubkey != NULL"); return 0;` style guards.
  The T-09/10 fix applied the same treatment to keypair_create/sec/pub/xonly_pub and
  ecdsa_signature_parse_compact/der, but did not reach ec_pubkey_parse.
- **Impact:** Callers that accidentally pass NULL pubkey or NULL input will receive
  return 0 (failure) without the debugging signal of the illegal callback. No security
  impact — the call still fails. This is a programming-error path, not an attacker path.
- **Tracking:** SHIM-ILLCB-002. Can be fixed in the same style as the T-09/10 NULL-arg
  callback fixes. Low priority — callers that pass NULL `pubkey` or `input` to
  `secp256k1_ec_pubkey_parse` have a programming error that return-0 already surfaces.
- **Test:** None planned — future T-09/10 follow-up can add coverage when fixed.

---

## secp256k1_context_randomize — blinding is per-thread, not per-context (SHIM-THREAD-BLIND)

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
  overwrite each other's active blinding. For example, if context A's `ContextBlindingScope`
  is entered, then context B's `ContextBlindingScope` is entered before A's exits (which
  cannot happen via the shim's current synchronous API but could happen via callbacks or
  coroutines), context A would clear B's blinding on exit. The source contains a comment
  `// DEVIATION FROM LIBSECP CONTRACT` acknowledging this limitation.
- **Reason:** The CT blinding API (`ct::set_blinding` / `ct::clear_blinding` /
  `ct::generator_mul_blinded`) is a global singleton by design — it operates on a thread-local
  scalar pair that applies to all CT operations on that thread. Plumbing per-context blinding
  through the CT layer would require a significant internal API refactor (passing the blinding
  state through every CT call as an explicit parameter). The current design is safe for the
  dominant use pattern: one context per thread, or multiple contexts used sequentially
  (not interleaved within a single call frame).
- **Impact:**
  1. **Single-context-per-thread callers:** No divergence. The behavior is identical to
     upstream — `secp256k1_context_randomize(ctx, seed)` sets the blinding, and all signing
     calls using `ctx` on that thread apply the blinding correctly.
  2. **Multiple-contexts-same-thread, sequential callers:** No divergence. As long as signing
     operations from different contexts do not interleave (i.e., one complete sign call
     finishes before another starts), each `ContextBlindingScope` loads its own context's seed
     and clears it correctly.
  3. **Interleaved contexts (callbacks, coroutines, C++ co_await):** Potential mismatch.
     If context A's signing path is suspended (co_await) after entering its
     `ContextBlindingScope` and context B's signing is invoked on the same thread, B will
     overwrite `g_blinding`; when A resumes, it will operate under B's (or no) blinding.
     This scenario does not arise in Bitcoin Core's synchronous signing paths.
  4. **Two randomized contexts same thread:** `secp256k1_context_randomize(ctx_a, seed_a)`
     then `secp256k1_context_randomize(ctx_b, seed_b)` — both stores succeed. But if both
     contexts are used to sign on the same thread within the same call stack (nested),
     only the innermost `ContextBlindingScope` is active at any given moment. This is
     architecturally safe for synchronous stacks.
- **Tracking:** SHIM-THREAD-BLIND. The source acknowledgement `// DEVIATION FROM LIBSECP
  CONTRACT` is at `src/cpu/src/ct_point.cpp` near the `thread_local BlindingState` definition.
  This divergence is not a security vulnerability for the targeted Bitcoin Core use case
  (one-context-per-thread, synchronous signing), but is a correctness gap for advanced
  multi-context patterns.
- **Planned fix:** Refactor CT blinding to accept an explicit `BlindingState*` parameter,
  thread-local as a fallback only. Deferred — requires changes throughout the CT layer.
- **Differential test:** Instantiate two contexts A and B, randomize both with distinct seeds,
  call a sign function with A and verify the result, then call with B and verify. Assert both
  produce valid signatures. This proves sequential multi-context operation is safe. Testing
  interleaved/nested is not feasible via the synchronous C ABI.
- **Test:** `audit/test_regression_shim_thread_blinding.cpp` (to be added) — TBL-1: sign with
  ctx_a then sign with ctx_b on same thread, both must verify. TBL-2: randomize then sign 1000
  times with alternating contexts, all must verify.

---

## secp256k1_ec_pubkey_parse — hybrid pubkeys (0x06/0x07 prefix) now accepted (SHIM-HYBRID-001)

- **Upstream behavior:** libsecp256k1 accepts 0x04 (uncompressed), 0x02/0x03 (compressed),
  and 0x06/0x07 (hybrid) prefixes in `secp256k1_ec_pubkey_parse`. Hybrid-format public keys
  (which encode both the full X and Y coordinates plus a parity byte) are a valid but rarely
  used SEC encoding. All three formats parse to the same on-curve point.
- **Previous shim behavior (divergence — FIXED 2026-05-21):** The shim's 65-byte parse path
  in `shim_pubkey.cpp` only accepted 0x04 (uncompressed). 0x06 and 0x07 (hybrid prefix)
  caused the parse to return 0 (failure) without firing the illegal callback. This caused
  70 out of 693 Bitcoin Core integration test cases to fail (hybrid-pubkey test vectors).
- **Current shim behavior:** `secp256k1_ec_pubkey_parse` now accepts 0x04, 0x06, and 0x07
  prefixes for 65-byte inputs, parsing the Y coordinate directly and ignoring the parity byte
  (Y is reconstructed from the SEC encoding, and the encoded Y is used as-is). This matches
  libsecp256k1 behavior. 0x02/0x03 (33-byte compressed) continue to be handled separately.
- **Reason:** BIP-340, Bitcoin Core test vectors, and SEC-encoded public keys distributed
  on-chain can carry the 0x06/0x07 prefix. Rejection broke the Bitcoin Core integration test
  suite. Fixed to match upstream.
- **Impact:** None for callers sending standard compressed (0x02/0x03) or uncompressed (0x04)
  public keys. Callers sending hybrid-encoded keys (0x06/0x07) now succeed where they
  previously failed.
- **Test:** Bitcoin Core integration test suite — 693/693 pass after fix (2026-04-27,
  commit c1df659e). Dedicated shim unit test in `compat/libsecp256k1_shim/tests/` for
  hybrid-prefix round-trip (to be added).

---

## secp256k1_context_* — SECP256K1_CONTEXT_NONE accepted for sign and verify (SHIM-CTX-NONE)

- **Upstream behavior (libsecp256k1 v0.6+):** Starting with v0.6, libsecp256k1 deprecated
  the `SECP256K1_CONTEXT_SIGN` / `SECP256K1_CONTEXT_VERIFY` flags. All operations now accept
  any non-NULL context regardless of the creation flags. `SECP256K1_CONTEXT_NONE` is valid for
  both signing and verification in v0.6+. The flags are kept for source compatibility but are
  no longer meaningful — `secp256k1_context_create(SECP256K1_CONTEXT_NONE)` is the recommended
  idiomatic form.
- **Shim behavior:** `ctx_can_sign()` and `ctx_can_verify()` return `true` when the context
  flags are `SECP256K1_CONTEXT_NONE` (0x0), in addition to the explicit SIGN and VERIFY flags.
  This matches the libsecp256k1 v0.6+ behavior and ensures callers that pass
  `secp256k1_context_create(SECP256K1_CONTEXT_NONE)` to sign functions do not receive a
  spurious illegal callback.
- **Reason:** Bitcoin Core 28.x uses `SECP256K1_CONTEXT_NONE` as the recommended context type.
  Rejecting it in the shim would cause all Bitcoin Core signing and verification calls to fail
  or fire illegal callbacks, breaking the integration.
- **Impact:** Callers using the older `SECP256K1_CONTEXT_SIGN` / `SECP256K1_CONTEXT_VERIFY`
  flags continue to work. Callers using `SECP256K1_CONTEXT_NONE` (v0.6+ idiom) now work
  correctly. This is intentional v0.6+ compat.
- **Note:** If a strictly wrong-flag context is passed (e.g., VERIFY-only context to a sign
  function in a build that enforces the old semantics), the shim fires the illegal callback
  per the `ctx_can_sign()` path — see the "All sign functions -- wrong-flag context (SHIM-001)"
  entry above. CONTEXT_NONE is explicitly exempt from that path.
- **Test:** `audit/test_regression_shim_context_none.cpp` — creates a CONTEXT_NONE context,
  calls `secp256k1_ecdsa_sign` and `secp256k1_schnorrsig_sign` with it, and asserts both
  return 1 without firing the illegal callback. Also calls `secp256k1_ecdsa_verify` and
  `secp256k1_schnorrsig_verify` with CONTEXT_NONE and asserts both return correct results.

---

## secp256k1_schnorrsig_verify_batch — msglen != 32 returns 0 silently (SHIM-006)

- **Upstream behavior:** `secp256k1_schnorrsig_verify_batch` in libsecp256k1 supports
  variable-length messages (`msglen` need not be 32). The BIP-340 challenge hash accepts
  any message length; batch verify is defined for arbitrary `msglen`.
- **Shim behavior:** The shim's batch verify does not implement the varlen code path — it
  only supports `msglen == 32`. When `msglen != 32`, the function returns `0` (fail-closed)
  without firing the illegal callback. Callers needing varlen must use the singular
  `secp256k1_schnorrsig_verify` which handles any `msglen` correctly.
- **Reason:** Varlen batch verify requires a generalized tagged-hash path through the MSM
  accumulator. The shim's current batch MSM uses 32-byte message slots. This is a shim
  capability limitation, not an illegal API call — firing `abort()` via the illegal callback
  for an unsupported-but-valid input was a divergence from upstream (SHIM-006 correction,
  2026-05-26). Previously the shim fired the illegal callback on msglen != 32, which would
  abort callers that legitimately use varlen batch verify against upstream libsecp256k1.
- **Impact:** Callers using `msglen == 32` (standard BIP-340 use) are unaffected. Callers
  using varlen batch verify receive `0` (verification failed) instead of aborting. They
  should fall back to singular verify for varlen messages.
- **Test:** `test_shim006_verify_batch_nonstandard_msglen_returns_zero()` in
  `compat/libsecp256k1_shim/tests/test_shim_security_edge_cases.cpp` — verifies that
  `secp256k1_schnorrsig_verify_batch` with `msglen=64` returns 0 without firing the
  illegal callback.
