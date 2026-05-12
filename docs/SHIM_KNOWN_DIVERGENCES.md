# Libsecp256k1 Shim — Known Behavioral Divergences

This document lists every intentional behavioral difference between the
UltrafastSecp256k1 libsecp256k1 shim and upstream libsecp256k1.

**An unlisted divergence is a bug.** If you discover shim behavior that differs
from libsecp256k1 and it is not documented here, it must either be fixed or added
to this file before any PR.

For the complete compatibility test matrix see `compat/libsecp256k1_shim/tests/`.

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

- **Upstream behavior (v0.4+):** `msg == NULL` with `msglen == 0` is accepted (empty
  message is a valid BIP-340 sign_custom input).
- **Shim behavior:** Requires exactly `msglen == 32`. Returns 0 for any other length.
- **Reason:** The internal Schnorr verifier is optimized for 32-byte messages (the only
  length used by Bitcoin consensus: `SIGHASH`, Taproot). Variable-length support requires
  a different code path not yet implemented in the shim.
- **Impact:** Callers using variable-length Schnorr (FROST, silent payments with tweaked
  hashes, Lightning custom sig types with msglen != 32).
- **Note:** Documented in `secp256k1_schnorrsig.h` divergence comment.

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
- **Shim behavior:** Uses `Scalar::from_bytes` (mod-n reduction) on the seed. Seeds with
  value >= n are silently reduced, so the cached blinding scalar does not correspond to
  the raw seed bytes.
- **Reason:** The shim's context blinding uses the seed as a scalar for `r*G` computation;
  this requires it to be in [0, n-1]. Silent reduction is safe here because the blinding
  scalar is never derived from or compared against the seed.
- **Impact:** None for callers using the recommended 32 bytes of fresh randomness from OS.
- **Test:** Not testable as a divergence — upstream behavior on seeds >= n is unspecified.

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
- **Shim behavior:** Uses `parse_bytes_strict_nonzero` — rejects r=0 or s=0 at parse time.
  Stricter than upstream.
- **Reason:** Consistent with the internal invariant that parsed secret-adjacent scalars
  are non-zero. The extra strictness has no practical impact (r=0 or s=0 signatures are
  astronomically rare and always invalid).
- **Impact:** Callers parsing r=0 or s=0 recoverable compact signatures (astronomically rare).

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

## secp256k1_schnorrsig_sign_custom — non-BIP-340 nonce function

- **Upstream behavior:** Any `extraparams->noncefp` is dispatched.
- **Shim behavior:** Only `secp256k1_nonce_function_bip340` (or NULL) is accepted.
  Any other non-NULL `noncefp` returns 0.
- **Reason:** Same as ECDSA sign custom nonce rejection.
- **Impact:** Callers with custom BIP-340 nonce functions.

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
- **Impact:** Callers parsing DER-encoded ECDSA signatures with r=0 or s=0 (astronomically
  rare; all such signatures are cryptographically invalid).
- **Test:** `test_shim_der_zero_r.cpp` in `compat/libsecp256k1_shim/tests/` —
  calls `secp256k1_ecdsa_signature_parse_der` with a hardcoded DER blob encoding r=0
  and verifies return value is 0. Upstream libsecp returns 1 (passes to verify which
  then rejects).

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

## secp256k1_ecdh — uses Scalar::from_bytes (silent mod-n reduction) (SHIM-NEW-002)

- **Upstream behavior:** libsecp256k1 `secp256k1_ecdh` also uses silent mod-n reduction
  on the private key (it does not call `secp256k1_ec_seckey_verify` internally).
  A key equal to n reduces to 0, which then causes `secp256k1_ecdh` to return 0.
- **Shim behavior:** Uses `Scalar::from_bytes` (mod-n reduction), matching upstream.
  This is intentional: `secp256k1_ecdh` is consistent with upstream behavior.
- **Contrast:** All other shim signing functions use `parse_bytes_strict_nonzero` (which
  rejects keys >= n). `secp256k1_ecdh` intentionally does NOT reject keys >= n at parse
  time, matching upstream semantics.
- **Reason:** Consistency with libsecp256k1 contract for `secp256k1_ecdh`.
- **Impact:** Callers passing private keys >= n to `secp256k1_ecdh` will get 0 (ECDH fails
  on reduced-to-zero key) rather than a strict rejection error. Same behavior as upstream.
- **Test:** `test_shim_ecdh_from_bytes_behavior` — pass key = n, verify return is 0
  (consistent with libsecp256k1: key reduces to 0, ECDH on 0 fails).

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
- **Shim behavior:** Returns 0 silently without firing the illegal callback. `ctx_can_sign()`
  returns false and the function returns 0 immediately.
- **Reason:** Ultra's context model does not replicate libsecp's callback architecture exactly.
  Fail-closed (returns error) is maintained; callback notification is absent.
- **Impact:** Fuzzing harnesses that count illegal-callback invocations will see count=0 from
  the Ultra shim when context-misuse is tested. Callers relying on callback for error detection
  will miss the signal. Callers that check only the return value are unaffected.
- **Test:** Differential test -- pass VERIFY-only ctx to secp256k1_ecdsa_sign, check callback.

---

## secp256k1_schnorrsig_verify -- null msg without firing illegal callback (SHIM-003)

- **Upstream behavior:** `secp256k1_schnorrsig_verify` with `msg=NULL` and `msglen=32` fires
  the illegal callback (default: abort) then returns 0.
- **Shim behavior:** The `(!msg || msglen != 32)` short-circuit guard returns 0 immediately
  without firing the illegal callback.
- **Reason:** Same as SHIM-001: the callback architecture is not replicated. Fail-closed
  behavior (return 0) is preserved; callback notification is absent.
- **Impact:** Same as SHIM-001 -- fuzzing harnesses counting illegal-callback invocations will
  see count=0. Callers that check only return value are unaffected.
- **Test:** Differential test -- pass null msg with msglen=32, check callback count.
