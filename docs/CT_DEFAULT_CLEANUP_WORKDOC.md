# CT Default Cleanup Work Document

> Scope: public C++ signing APIs only. This document does not claim the `ufsecp`
> C ABI is currently routing through non-CT signing; the ABI wrappers already call
> `secp256k1::ct::*` for the main signing paths. The remaining cleanup is about
> making the normal-looking public C++ API safe by default.

## Goal

Secret-bearing signing defaults must route through the constant-time layer:

- ECDSA signing and hedged signing
- ECDSA recoverable signing
- Schnorr keypair creation, x-only pubkey derivation, signing, and sign-verify
- Any wallet, message-signing, Ethereum, Bitcoin, ECIES, or shim path that uses
  these C++ APIs internally

Variable-time signing may remain only as an explicit opt-in API with a name that
cannot be mistaken for production-safe default behavior, for example:

- `ecdsa_sign_fast_unsafe`
- `schnorr_sign_fast_unsafe`
- `secp256k1::fast_unsafe::*`
- or a build-gated path requiring `SECP256K1_ALLOW_FAST_SIGN`

## Current Non-CT Default Surfaces

### 1. `cpu/src/ecdsa.cpp` — ECDSA default signer

`signing_generator_mul()` uses the CT path only on MSVC. On other compilers,
secret nonce `k` is multiplied through the fast generator path.

```cpp
static inline Point signing_generator_mul(const Scalar& scalar) {
#if defined(_MSC_VER)
    // MSVC Release has shown runtime fixed-base generator divergence for some
    // scalars after hostile reconfiguration. Keep signing on the proven CT path.
    return ct::generator_mul(scalar);
#else
    return Point::generator().scalar_mul(scalar);
#endif
}
```

Affected public functions:

- `secp256k1::ecdsa_sign`
- `secp256k1::ecdsa_sign_verified`
- `secp256k1::ecdsa_sign_hedged`
- `secp256k1::ecdsa_sign_hedged_verified`

The default signer also uses `k.inverse()` on the secret nonce:

```cpp
auto k_inv = k.inverse();
auto s = k_inv * (z + r * private_key);
```

Required cleanup:

- Make public default `secp256k1::ecdsa_sign*` wrappers call
  `secp256k1::ct::ecdsa_sign*`.
- Move the existing fast implementation behind an explicit unsafe/fast name.
- Keep RFC 6979 output compatibility covered by differential tests.

### 2. `cpu/src/recovery.cpp` — ECDSA recoverable default signer

This file already documents the problem:

```cpp
// WARNING: Variable-time path -- uses fast::scalar_mul(k) and fast::inverse(k)
// on the secret nonce. For side-channel-resistant signing, use ct::ecdsa_sign()
// (which does not produce recovery IDs). This function is suitable only for
// environments where timing attacks are not a concern.
```

Affected public function:

- `secp256k1::ecdsa_sign_recoverable`

Current risky operations:

- `Point::generator().scalar_mul(k)` through `signing_generator_mul(k)`
- branchy recovery ID overflow comparison
- `k.inverse()`
- low-S normalization branch

Required cleanup:

- Make public default `secp256k1::ecdsa_sign_recoverable` call
  `secp256k1::ct::ecdsa_sign_recoverable`.
- Rename the existing variable-time implementation to an explicit unsafe name
  if it is still needed for benchmarking or controlled experiments.
- Update docs so the public recovery signing API no longer points users at a
  variable-time default.

### 3. `cpu/src/schnorr.cpp` — Schnorr default signing and key derivation

Public Schnorr helpers currently use fast generator multiplication on secret
inputs.

Affected public functions:

- `secp256k1::schnorr_pubkey`
- `secp256k1::schnorr_keypair_create`
- `secp256k1::schnorr_sign`
- `secp256k1::schnorr_sign_verified`
- `secp256k1::schnorr_xonly_from_keypair` if used on secret-derived keypairs

Current examples:

```cpp
std::array<uint8_t, 32> schnorr_pubkey(const Scalar& private_key) {
    SECP_ASSERT_SCALAR_VALID(private_key);
    auto P = Point::generator().scalar_mul(private_key);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    (void)p_y_odd;
    return px;
}
```

```cpp
auto P = Point::generator().scalar_mul(d_prime);
auto [px, p_y_odd] = P.x_bytes_and_parity();

kp.d = p_y_odd ? d_prime.negate() : d_prime;
kp.px = px;
```

```cpp
// Step 3: R = k' * G (single gen_mul -- the only expensive point op)
auto R = Point::generator().scalar_mul(k_prime);
auto [rx, r_y_odd] = R.x_bytes_and_parity();

// Step 4: k = k' if has_even_y(R), else n - k'
auto k = r_y_odd ? k_prime.negate() : k_prime;
```

Required cleanup:

- Make public default Schnorr helpers call `secp256k1::ct::schnorr_pubkey`,
  `secp256k1::ct::schnorr_keypair_create`, `secp256k1::ct::schnorr_sign`, and
  `secp256k1::ct::schnorr_sign_verified`.
- Keep variable-time Schnorr signing only under explicit unsafe naming or
  build-time opt-in.
- Ensure parity-based key negation and nonce negation use CT conditional
  operations in the default path.

### 4. Public headers expose normal-looking non-CT APIs

The public headers make the non-CT APIs look like ordinary production signing
entry points.

Affected headers:

- `cpu/include/secp256k1/ecdsa.hpp`
- `cpu/include/secp256k1/schnorr.hpp`
- `cpu/include/secp256k1/recovery.hpp`

Required cleanup:

- Update comments to state that default signing is CT.
- If unsafe fast signing remains, expose it with explicit warning language.
- Keep `ct/sign.hpp` as the implementation/source-of-truth for secret-bearing
  production signing.

## What Is Already Good

The C ABI signing paths already route through the CT layer for the main signing
functions:

- `ufsecp_ecdsa_sign`
- `ufsecp_ecdsa_sign_verified`
- `ufsecp_ecdsa_sign_recoverable`
- `ufsecp_schnorr_sign`
- `ufsecp_schnorr_sign_verified`

Follow-up still needed:

- Source graph metadata currently shows ABI wrapper audit metadata as `ct=0`
  in some places even when the wrapper dispatches to CT implementation. That is
  an evidence-mapping issue, not necessarily an implementation issue.

## Remediation Plan

1. Introduce explicit unsafe/fast function names for the current variable-time
   C++ signing implementations.
2. Change public default `secp256k1::ecdsa_sign*` wrappers to dispatch to
   `secp256k1::ct::ecdsa_sign*`.
3. Change public default `secp256k1::ecdsa_sign_recoverable` to dispatch to
   `secp256k1::ct::ecdsa_sign_recoverable`.
4. Change public default Schnorr helpers to dispatch to `secp256k1::ct::*`.
5. Add compile-time guardrails:
   - `SECP256K1_REQUIRE_CT=ON` by default, or equivalent behavior.
   - variable-time sign APIs require `SECP256K1_ALLOW_FAST_SIGN`.
6. Update source graph tags:
   - public default signing symbols: `ct=1`
   - unsafe fast symbols: `ct=0`, `unsafe_fast_signing`
   - ABI wrappers: inherit CT evidence from the CT implementation they call
7. Update docs:
   - `docs/API_REFERENCE.md`
   - `docs/CRYPTO_INVARIANTS.md`
   - `docs/SAFE_DEFAULTS.md`
   - `docs/CT_VERIFICATION.md`
   - `docs/AUDIT_CHANGELOG.md`
   - `include/ufsecp/SUPPORTED_GUARANTEES.md`
8. Update tests and gates:
   - public C++ default signing equals CT signing
   - public C++ default signing does not call unsafe fast symbols
   - unsafe fast signing is unavailable unless explicitly enabled
   - source graph coverage reports public default signing as CT

## Suggested Test Plan

Run or add coverage for:

- `test_ct_equivalence`
- `test_ct_sidechannel`
- `test_cross_libsecp256k1`
- `test_bip340_vectors`
- `test_rfc6979_vectors`
- `test_batch_randomness`
- `test_c_abi_negative`
- source graph symbol/coverage queries for:
  - `ecdsa_sign`
  - `ecdsa_sign_recoverable`
  - `schnorr_sign`
  - `schnorr_keypair_create`
  - `ufsecp_ecdsa_sign`
  - `ufsecp_schnorr_sign`

## Definition of Done

- `secp256k1::ecdsa_sign`, `ecdsa_sign_hedged`,
  `ecdsa_sign_verified`, and `ecdsa_sign_recoverable` default to CT.
- `secp256k1::schnorr_sign`, `schnorr_sign_verified`,
  `schnorr_keypair_create`, and `schnorr_pubkey` default to CT.
- Non-CT sign functions no longer appear as normal production APIs.
- Any remaining variable-time signing is explicitly marked unsafe/fast and
  opt-in only.
- C ABI signing remains CT and source graph evidence reflects that.
- Documentation and audit changelog are updated in the same change.
- CAAS, CT, differential, BIP-340, and RFC 6979 checks pass.

## Priority

High for Bitcoin Core readiness and public review posture.

Reason: even if the C ABI/shim is already CT, reviewers will inspect the public
C++ API and notice normal-looking signing functions that use fast variable-time
internals. Closing this removes a narrative and security-review weakness before
opening a Bitcoin Core PR.
