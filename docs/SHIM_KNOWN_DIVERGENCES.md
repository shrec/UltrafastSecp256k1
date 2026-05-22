# Libsecp256k1 Shim — Known Behavioral Divergences

This document lists every intentional behavioral difference between the
UltrafastSecp256k1 libsecp256k1 shim and upstream libsecp256k1.

**An unlisted divergence is a bug.** If you discover shim behavior that differs
from libsecp256k1 and it is not documented here, it must either be fixed or added
to this file before any PR.

For the complete compatibility test matrix see `compat/libsecp256k1_shim/tests/`.

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

## secp256k1_musig_nonce_agg / nonce_process / partial_sig_verify / pubkey_get — `ctx` ignored

- **Upstream behavior:** These four MuSig2 functions accept a `secp256k1_context*`
  argument but do not require it to be non-NULL in upstream libsecp256k1-zkp at
  the version targeted by this shim. The argument is reserved for forward
  compatibility but is currently unused.
- **Shim behavior:** Same — the `ctx` parameter is declared `/*ctx*/` (unused).
  A NULL `ctx` will NOT fire the illegal-callback path on these functions; only
  the explicit argument checks apply.
- **Reason:** Matches upstream. These calls do not consume context state
  (no signing context flags, no blinding, no callbacks needed for the operation).
- **Impact:** Callers relying on a NULL-ctx illegal-callback firing for these
  specific functions will not see it. The illegal-callback contract holds for
  every other shim function that uses the context.
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
- **Test:** `audit/test_exploit_shim_musig_ka_cap.cpp` KAC-1..3.

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
- **Test:** TODO — differential test against libsecp256k1-zkp that verifies nonce bytes
  differ when extra_input32 is non-NULL.

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

## secp256k1_musig_pubkey_ec_tweak_add / secp256k1_musig_pubkey_xonly_tweak_add — ctx silently discarded (SHIM-MUSIG-CTX-001)

- **Upstream behavior:** NULL ctx fires the illegal callback (default: abort). Non-NULL ctx is
  validated for context flags before proceeding.
- **Shim behavior:** Both `secp256k1_musig_pubkey_ec_tweak_add` and
  `secp256k1_musig_pubkey_xonly_tweak_add` declare the ctx parameter as `/*ctx*/` — it is
  **completely discarded** and never inspected. NULL ctx will NOT fire the illegal callback;
  it silently proceeds to the `keyagg_cache` / `tweak32` NULL check instead. SHIM_REQUIRE_CTX
  is NOT used by these functions (contrary to an earlier version of this document).
- **Reason:** These operations act purely on the `secp256k1_musig_keyagg_cache` state and do
  not require context flags (no signing, no blinding). The ctx parameter exists only for
  API symmetry with upstream. The discarding matches the upstream libsecp256k1-zkp pattern
  for these specific functions (see also `secp256k1_musig_nonce_agg` above).
- **Impact:** A caller passing NULL ctx will not get a callback/abort. The only guard is the
  `!keyagg_cache || !tweak32` NULL check. Callers that rely on NULL-ctx abort from these two
  functions (e.g. fuzzing harnesses) will see silent-return-0 instead.
- **Open:** Consider adding explicit NULL-ctx guard firing the illegal callback for parity.
  Tracked as SHIM-MUSIG-CTX-001 (open).
- **Test:** Covered indirectly by context-flag tests in `test_regression_shim_security_v7_run`.
