# FROST Compliance Statement

## Implementation Reference

This implementation follows **FROST (Flexible Round-Optimized Schnorr Threshold
Signatures)** as described in:
- **RFC 9591** (FROST: Flexible Round-Optimized Schnorr Threshold Signatures)
- **BIP-340** (Schnorr Signatures for secp256k1) -- for final signature format
- **Draft BIP-FROST** (Bitcoin threshold signing) -- partial alignment

Source files:
- `src/cpu/include/secp256k1/frost.hpp` -- public API
- `src/cpu/src/frost.cpp` -- implementation

## Protocol Checkpoint Matrix

### DKG (Distributed Key Generation)

| Checkpoint | RFC 9591 | Implementation | Status |
|---|---|---|---|
| Feldman VSS polynomial generation | Required | `frost_keygen_begin()` generates random polynomial of degree t-1 | Compliant |
| Polynomial commitment broadcast | Required | `FrostCommitment` (vector of `A_{i,j} = a_{i,j}*G`) | Compliant |
| Share evaluation `f_i(j)` | Required | `poly_eval` Horner's method, shares to all n participants | Compliant |
| Share verification against commitment | Required | `frost_keygen_finalize()` verifies `share*G == Sum(A_j * x^j)` | Compliant |
| Signing share aggregation `s_i = Sum(f_j(i))` | Required | Computed in `frost_keygen_finalize()` | Compliant |
| Verification share `Y_i = s_i * G` | Required | Computed via `ct::generator_mul` (constant-time) | Compliant |
| Group public key `Y = Sum(A_{j,0})` | Required | Sum of constant coefficients from all commitments | Compliant |
| DKG uses CT path for secret ops | Best practice | `ct::generator_mul` for commitment + verification share | Compliant |

### Nonce Generation

| Checkpoint | RFC 9591 | Implementation | Status |
|---|---|---|---|
| Two-nonce scheme (hiding + binding) | Required | `FrostNonce` has `hiding_nonce` (d_i) + `binding_nonce` (e_i) | Compliant |
| Nonce commitment `D_i = d_i*G, E_i = e_i*G` | Required | `FrostNonceCommitment` struct | Compliant |
| Nonce freshness | Required | Derived from `nonce_seed` via SHA256 | See Note 1 |
| Single-use nonce enforcement | Required | Caller responsibility (no built-in state) | Partial |

### Signing

| Checkpoint | RFC 9591 | Implementation | Status |
|---|---|---|---|
| Binding factor `rho_i = H(group_key, i, commitments, msg)` | Required | `compute_binding_factor()` SHA256 tagged hash | Compliant |
| Group commitment `R = Sum(D_i + rho_i*E_i)` | Required | `compute_group_commitment()` | Compliant |
| BIP-340 even-Y normalization | BIP-340 compat | R/group_key negated for even Y | Compliant |
| Challenge `e = H("BIP0340/challenge", R.x, P.x, m)` | BIP-340 compat | `compute_challenge()` uses BIP-340 tagged hash | Compliant |
| Lagrange coefficient `lambda_i` | Required | `frost_lagrange_coefficient()` | Compliant |
| Partial sig `z_i = d_i + rho_i*e_i + lambda_i*s_i*e` | Required | Computed with proper negate handling | Compliant |

### Partial Signature Verification

| Checkpoint | RFC 9591 | Implementation | Status |
|---|---|---|---|
| Verify `z_i*G == R_i + lambda_i*e*Y_i` | Required | `frost_verify_partial()` | Compliant |
| Robustness (identify malicious signers) | Optional | Supported via per-signer verification | Compliant |

### Aggregation

| Checkpoint | RFC 9591 | Implementation | Status |
|---|---|---|---|
| Aggregate `s = Sum(z_i)` | Required | `frost_aggregate()` | Compliant |
| Output standard BIP-340 signature | BIP-340 compat | Returns `SchnorrSignature{R.x, s}` | Compliant |
| Even-Y normalization on R | BIP-340 compat | R negated if odd Y | Compliant |
| Final sig verifiable with `schnorr_verify` | BIP-340 compat | Standard Schnorr verification applies | Compliant |

## Known Deviations and Notes

### Note 1: Nonce Derivation Method
RFC 9591 specifies nonce generation using `random_bytes(32)` (true CSPRNG).
The implementation uses deterministic derivation via `SHA256(seed || context || id)`.
This is safe when `nonce_seed` is 32 bytes of fresh CSPRNG output, but the API
does not enforce this. Callers MUST provide cryptographically random seeds.

### Note 2: Single-Use Nonce State
RFC 9591 requires that nonces are never reused. The implementation does not
maintain internal state to prevent reuse -- this is the caller's responsibility.
Nonce reuse with different messages under the same key leaks the signing share.

### Note 3: Nonce Commitment Sorting
RFC 9591 requires deterministic ordering of nonce commitments. The implementation
processes them in the order provided. Callers MUST ensure consistent ordering
across all signers (e.g., sorted by participant ID).

### Note 4: BIP-FROST (Draft) Status
BIP-FROST (threshold signing for Bitcoin) is still a draft BIP. The implementation
aligns with the current draft where compatible with RFC 9591. As the BIP evolves,
the following areas may need updates:
- Serialization format for share exchange messages
- Specific tagged hash context strings (currently uses "FROST_binding", "FROST_keygen_poly", etc.)
- Compatibility with ROAST (Robust Asynchronous Schnorr Threshold) wrapper protocol

### Note 5: Secret Zeroization
`frost_keygen_begin()` generates polynomial coefficients as local `std::vector<Scalar>`.
These are not explicitly zeroed on return. For production deployment, consider
adding `secure_erase` to the polynomial coefficient vector before returning.

## Test Coverage

FROST functionality is tested via:
- Unit tests: `test_frost.cpp` (DKG round-trip, signing, aggregation, verification)
- The aggregate signature is verified against standard `schnorr_verify`

## Recommendations

1. **Nonce state management**: Consider adding a `FrostSignerState` struct that
   tracks used nonces and prevents reuse.
2. **Secret zeroization**: Add `secure_erase` for polynomial coefficients in DKG
   and for `FrostNonce` secret scalars after signing.
3. **Commitment sorting**: Add internal sorting by participant ID in signing
   functions to prevent ordering-dependent bugs.
4. **Tagged hash alignment**: When BIP-FROST is finalized, update context strings
   to match the standardized tag values.
