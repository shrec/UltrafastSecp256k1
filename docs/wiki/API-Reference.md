# API Reference

Complete API documentation for UltrafastSecp256k1.

For the full detailed reference, see [docs/API_REFERENCE.md](https://github.com/shrec/UltrafastSecp256k1/blob/main/docs/API_REFERENCE.md).

---

## CPU API

**Namespace:** `secp256k1::fast`

**Headers:**
```cpp
#include <secp256k1/field.hpp>    // FieldElement
#include <secp256k1/scalar.hpp>   // Scalar
#include <secp256k1/point.hpp>    // Point, KPlan, Selftest
#include <secp256k1/ecdsa.hpp>    // ECDSA sign/verify
#include <secp256k1/schnorr.hpp>  // Schnorr BIP-340 sign/verify
#include <secp256k1/ecdh.hpp>     // ECDH key exchange
#include <secp256k1/bip32.hpp>    // BIP-32 HD derivation
#include <secp256k1/address.hpp>  // Address generation (P2PKH, P2WPKH, P2TR)
#include <secp256k1/wif.hpp>      // WIF encode/decode
#include <secp256k1/sha256.hpp>   // SHA-256 (SHA-NI accelerated)
```

---

## FieldElement

256-bit field element for secp256k1 (mod p where p = 2^256 - 2^32 - 977).

### Construction

| Method | Description |
|--------|-------------|
| `FieldElement::zero()` | Zero element |
| `FieldElement::one()` | One element |
| `FieldElement::from_uint64(val)` | From 64-bit integer |
| `FieldElement::from_limbs(arr)` | From 4x64-bit array (little-endian) |
| `FieldElement::from_bytes(arr)` | From 32 bytes (big-endian) |
| `FieldElement::from_hex(str)` | From hex string (64 chars) |

### Arithmetic

| Operator/Method | Description |
|-----------------|-------------|
| `a + b` | Addition |
| `a - b` | Subtraction |
| `a * b` | Multiplication |
| `a.square()` | Squaring (a^2) |
| `a.inverse()` | Modular inverse (a^-1) |
| `a += b` | In-place addition |
| `a -= b` | In-place subtraction |
| `a *= b` | In-place multiplication |
| `a.square_inplace()` | In-place squaring |
| `a.inverse_inplace()` | In-place inverse |

### Serialization

| Method | Description |
|--------|-------------|
| `a.to_bytes()` | To 32 bytes (big-endian) |
| `a.to_bytes_into(ptr)` | To buffer (no allocation) |
| `a.to_hex()` | To hex string |
| `a.limbs()` | Raw limb access |

### Comparison

| Operator | Description |
|----------|-------------|
| `a == b` | Equality |
| `a != b` | Inequality |

---

## Scalar

256-bit scalar for secp256k1 (mod n, the group order).

### Construction

| Method | Description |
|--------|-------------|
| `Scalar::zero()` | Zero |
| `Scalar::one()` | One |
| `Scalar::from_uint64(val)` | From 64-bit integer |
| `Scalar::from_limbs(arr)` | From 4x64-bit array |
| `Scalar::from_bytes(arr)` | From 32 bytes (big-endian) |
| `Scalar::from_hex(str)` | From hex string |

### Arithmetic

| Operator | Description |
|----------|-------------|
| `a + b` | Addition (mod n) |
| `a - b` | Subtraction (mod n) |
| `a * b` | Multiplication (mod n) |

### Utility

| Method | Description |
|--------|-------------|
| `s.is_zero()` | Check if zero |
| `s.bit(i)` | Get i-th bit |
| `s.to_naf()` | NAF encoding |
| `s.to_wnaf(w)` | wNAF encoding |

---

## Point

Elliptic curve point on secp256k1 (Jacobian coordinates internally).

### Construction

| Method | Description |
|--------|-------------|
| `Point::generator()` | Generator point G |
| `Point::infinity()` | Point at infinity |
| `Point::from_affine(x, y)` | From affine coordinates |
| `Point::from_hex(x_hex, y_hex)` | From hex strings |

### Point Operations

| Method | Description |
|--------|-------------|
| `p.add(q)` | Point addition (p + q) |
| `p.dbl()` | Point doubling (2p) |
| `p.scalar_mul(k)` | Scalar multiplication (k*p) |
| `p.negate()` | Negation (-p) |

### Optimized Operations

| Method | Description |
|--------|-------------|
| `p.scalar_mul_with_plan(plan)` | Fixed-K multiplication (fastest) |
| `p.next()` | p + G |
| `p.prev()` | p - G |
| `p.add_mixed_inplace(x, y)` | Mixed Jacobian+Affine addition (7M + 4S) |

### In-Place Operations

| Method | Description |
|--------|-------------|
| `p.add_inplace(q)` | p += q |
| `p.sub_inplace(q)` | p -= q |
| `p.dbl_inplace()` | p = 2p |
| `p.negate_inplace()` | p = -p |
| `p.next_inplace()` | p += G |
| `p.prev_inplace()` | p -= G |

### Serialization

| Method | Description |
|--------|-------------|
| `p.x()` | Affine x-coordinate |
| `p.y()` | Affine y-coordinate |
| `p.to_compressed()` | 33-byte compressed format |
| `p.to_uncompressed()` | 65-byte uncompressed format |
| `p.x_first_half()` | First 16 bytes of x |
| `p.x_second_half()` | Last 16 bytes of x |

### Properties

| Method | Description |
|--------|-------------|
| `p.is_infinity()` | Check if point at infinity |
| `p.X()`, `p.Y()`, `p.z()` | Raw Jacobian coordinates |

---

## KPlan

Pre-computed plan for fixed-K scalar multiplication.

```cpp
// Create plan once
Scalar K = Scalar::from_hex("...");
KPlan plan = KPlan::from_scalar(K);

// Use for multiple Q values (reuses cached GLV decomposition + wNAF)
Point R1 = Q1.scalar_mul_with_plan(plan);
Point R2 = Q2.scalar_mul_with_plan(plan);
```

---

## Selftest

```cpp
#include "secp256k1/selftest.hpp"
using namespace secp256k1::fast;

// Quick startup check
Selftest(true, SelftestMode::smoke);

// Full CI suite
Selftest(true, SelftestMode::ci);

// Nightly stress test with custom seed
Selftest(true, SelftestMode::stress, 0xDEADBEEF);
```

---

## ECDSA (RFC 6979)

**Header:** `#include <secp256k1/ecdsa.hpp>`

### Functions

| Function | Description |
|----------|-------------|
| `ecdsa_sign(msg_hash, seckey)` | Sign with RFC 6979 deterministic nonces, low-S normalization |
| `ecdsa_verify(msg_hash, pubkey, sig)` | Verify ECDSA signature |
| `ecdsa_sign_recoverable(msg_hash, seckey)` | Sign with recovery ID (for EIP-155) |
| `ecdsa_recover(msg_hash, sig, recid)` | Recover public key from signature |

### Example

```cpp
#include <secp256k1/ecdsa.hpp>
using namespace secp256k1::fast;

// Sign
uint8_t msg_hash[32] = { /* SHA-256 of message */ };
Scalar seckey = Scalar::from_hex("...");
auto [sig_r, sig_s] = ecdsa_sign(msg_hash, seckey);

// Verify
Point pubkey = Point::generator().scalar_mul(seckey);
bool valid = ecdsa_verify(msg_hash, pubkey, sig_r, sig_s);
```

### Performance (x86-64)

| Operation | Time |
|-----------|------|
| ECDSA Sign | 8.5 us |
| ECDSA Verify | 23.6 us |

---

## Schnorr (BIP-340)

**Header:** `#include <secp256k1/schnorr.hpp>`

### Functions

| Function | Description |
|----------|-------------|
| `schnorr_sign(msg, seckey, aux_rand)` | BIP-340 Schnorr sign with tagged hashing |
| `schnorr_verify(msg, pubkey_x, sig)` | BIP-340 Schnorr verify with x-only pubkey |

### Example

```cpp
#include <secp256k1/schnorr.hpp>
using namespace secp256k1::fast;

uint8_t msg[32] = { /* message hash */ };
uint8_t aux[32] = { /* auxiliary randomness */ };
Scalar seckey = Scalar::from_hex("...");
auto sig = schnorr_sign(msg, seckey, aux);

// Verify with x-only pubkey (32 bytes)
auto pubkey_x = Point::generator().scalar_mul(seckey).x().to_bytes();
bool valid = schnorr_verify(msg, pubkey_x, sig);
```

### Performance (x86-64)

| Operation | Time |
|-----------|------|
| Schnorr Sign | 6.8 us |
| Schnorr Verify | 24.0 us |

---

## ECDH

**Header:** `#include <secp256k1/ecdh.hpp>`

### Functions

| Function | Description |
|----------|-------------|
| `ecdh_raw(seckey, pubkey)` | Raw ECDH shared point |
| `ecdh_xonly(seckey, pubkey)` | x-coordinate only (32 bytes) |
| `ecdh_compressed(seckey, pubkey)` | SHA-256(compressed shared point) |

### Example

```cpp
#include <secp256k1/ecdh.hpp>
using namespace secp256k1::fast;
namespace ct = secp256k1::ct;

Scalar alice_sk = Scalar::from_hex("...");
Point bob_pk = Point::from_hex("...", "...");

// Use CT for secret-dependent operations!
Point shared = ct::scalar_mul(bob_pk, alice_sk);
auto secret = shared.x().to_bytes();  // 32-byte shared secret
```

### Performance (x86-64)

| Operation | Time |
|-----------|------|
| ECDH | 23.9 us |

---

## BIP-32 HD Derivation

**Header:** `#include <secp256k1/bip32.hpp>`

### Functions

| Function | Description |
|----------|-------------|
| `bip32_from_seed(seed, seed_len)` | Master key from seed |
| `bip32_derive_child(parent, index)` | Derive child key (hardened if index >= 0x80000000) |
| `bip32_serialize(key, version)` | Serialize to xprv/xpub |

### Derivation Paths

Supports all standard BIP-44 paths for 27 blockchains:
- Bitcoin: `m/86'/0'/0'` (P2TR), `m/84'/0'/0'` (P2WPKH)
- Ethereum: `m/44'/60'/0'`
- See the [27 Supported Coins table](https://github.com/shrec/UltrafastSecp256k1#secp256k1-supported-coins-27-blockchains)

---

## Address Generation

**Header:** `#include <secp256k1/address.hpp>`

### Functions

| Function | Description |
|----------|-------------|
| `address_p2pkh(pubkey, version)` | Pay-to-Public-Key-Hash (Base58Check) |
| `address_p2wpkh(pubkey, hrp)` | Pay-to-Witness-Public-Key-Hash (Bech32) |
| `address_p2tr(pubkey_x, hrp)` | Pay-to-Taproot (Bech32m) |
| `address_eip55(pubkey)` | Ethereum EIP-55 checksummed address |

---

## SHA-256

**Header:** `#include <secp256k1/sha256.hpp>`

Hardware-accelerated when SHA-NI is available.

| Function | Description |
|----------|-------------|
| `sha256(data, len)` | Single-shot SHA-256 |
| `sha256_hmac(key, data)` | HMAC-SHA-256 |

---

## Constant-Time (CT) API

**Namespace:** `secp256k1::ct`

**Headers:**
```cpp
#include <secp256k1/ct/ops.hpp>    // Low-level CT primitives
#include <secp256k1/ct/field.hpp>  // CT field arithmetic
#include <secp256k1/ct/scalar.hpp> // CT scalar arithmetic
#include <secp256k1/ct/point.hpp>  // CT point operations
```

All CT functions operate on the **same types** as `fast::` (`FieldElement`, `Scalar`, `Point`). Both namespaces are always compiled -- no flags or `#ifdef` needed.

---

### CT Primitives (`ct/ops.hpp`)

Low-level building blocks. Every function has a data-independent execution trace.

| Function | Description |
|----------|-------------|
| `value_barrier(v)` | Compiler optimization barrier |
| `is_zero_mask(v)` | Returns `0xFFF...F` if `v == 0`, else `0` |
| `is_nonzero_mask(v)` | Returns `0xFFF...F` if `v != 0`, else `0` |
| `eq_mask(a, b)` | Returns `0xFFF...F` if `a == b`, else `0` |
| `bool_to_mask(flag)` | Converts bool to all-ones/all-zeros mask |
| `lt_mask(a, b)` | Returns all-ones if `a < b` (unsigned) |
| `cmov64(dst, src, mask)` | CT conditional move (64-bit) |
| `cmov256(dst, src, mask)` | CT conditional move (4x64-bit) |
| `cswap256(a, b, mask)` | CT conditional swap (4x64-bit) |
| `ct_select(a, b, mask)` | Returns `a` if mask=all-ones, else `b` |
| `ct_lookup(table, count, stride, index, out)` | CT table lookup (scans all entries) |
| `ct_lookup_256(table, count, index, out)` | CT lookup for 256-bit entries |

---

### CT Field Operations (`ct/field.hpp`)

| Function | Description |
|----------|-------------|
| `field_add(a, b)` | CT addition mod p |
| `field_sub(a, b)` | CT subtraction mod p |
| `field_mul(a, b)` | CT multiplication mod p |
| `field_sqr(a)` | CT squaring mod p |
| `field_neg(a)` | CT negation mod p |
| `field_inv(a)` | CT inverse (addition-chain Fermat, fixed 255 sqr + 14 mul) |
| `field_cmov(r, a, mask)` | CT conditional move |
| `field_cswap(a, b, mask)` | CT conditional swap |
| `field_select(a, b, mask)` | CT select: `a` if mask=1s, else `b` |
| `field_cneg(a, mask)` | CT conditional negate |
| `field_is_zero(a)` | CT zero check -> mask |
| `field_eq(a, b)` | CT equality -> mask |
| `field_normalize(a)` | CT reduce to canonical form |

---

### CT Scalar Operations (`ct/scalar.hpp`)

| Function | Description |
|----------|-------------|
| `scalar_add(a, b)` | CT addition mod n |
| `scalar_sub(a, b)` | CT subtraction mod n |
| `scalar_neg(a)` | CT negation mod n |
| `scalar_cmov(r, a, mask)` | CT conditional move |
| `scalar_cswap(a, b, mask)` | CT conditional swap |
| `scalar_select(a, b, mask)` | CT select |
| `scalar_cneg(a, mask)` | CT conditional negate |
| `scalar_is_zero(a)` | CT zero check -> mask |
| `scalar_eq(a, b)` | CT equality -> mask |
| `scalar_bit(a, index)` | CT bit access (reads all limbs) |
| `scalar_window(a, pos, width)` | CT w-bit window extraction |

---

### CT Point Operations (`ct/point.hpp`)

| Function | Description |
|----------|-------------|
| `point_add_complete(p, q)` | Complete addition (handles all cases branchlessly) |
| `point_dbl(p)` | CT doubling |
| `point_neg(p)` | CT negation |
| `point_cmov(r, a, mask)` | CT conditional move |
| `point_select(a, b, mask)` | CT select |
| `point_table_lookup(table, size, index)` | CT table lookup (scans all entries) |
| `scalar_mul(p, k)` | **CT scalar multiplication** (GLV + effective-affine) |
| `generator_mul(k)` | CT generator multiplication (k*G, precomputed table) |
| `point_is_on_curve(p)` | CT curve membership check -> mask |
| `point_eq(a, b)` | CT point equality -> mask |

### CTJacobianPoint

Internal representation with `uint64_t infinity` flag (for branchless operations):

```cpp
struct CTJacobianPoint {
    FieldElement x, y, z;
    std::uint64_t infinity;  // 0 = normal, 0xFFF...F = infinity

    static CTJacobianPoint from_point(const Point& p) noexcept;
    Point to_point() const noexcept;
    static CTJacobianPoint make_infinity() noexcept;
};
```

---

## Stable C ABI (`ufsecp`)

**Header:** `#include "ufsecp.h"`

Starting with **v3.4.0**, UltrafastSecp256k1 ships a stable C ABI designed for FFI bindings (C#, Python, Rust, Go, Java, Node.js, etc.). 45 exported functions.

### Context Management

```c
ufsecp_ctx* ctx = NULL;
ufsecp_ctx_create(&ctx);
// ... use ctx ...
ufsecp_ctx_destroy(ctx);
```

### API Categories

| Category | Functions |
|----------|-----------|
| **Context** | `ctx_create`, `ctx_destroy`, `selftest`, `last_error` |
| **Keys** | `keygen`, `seckey_verify`, `pubkey_create`, `pubkey_parse`, `pubkey_serialize` |
| **ECDSA** | `ecdsa_sign`, `ecdsa_verify`, `ecdsa_sign_der`, `ecdsa_verify_der`, `ecdsa_recover` |
| **Schnorr** | `schnorr_sign`, `schnorr_verify` |
| **SHA-256** | `sha256` (SHA-NI accelerated) |
| **ECDH** | `ecdh_compressed`, `ecdh_xonly`, `ecdh_raw` |
| **BIP-32** | `bip32_from_seed`, `bip32_derive_child`, `bip32_serialize` |
| **Address** | `address_p2pkh`, `address_p2wpkh`, `address_p2tr` |
| **WIF** | `wif_encode`, `wif_decode` |
| **Tweak** | `pubkey_tweak_add`, `pubkey_tweak_mul` |
| **Version** | `version`, `abi_version`, `version_string` |

### Quick Example (C)

```c
#include "ufsecp.h"

ufsecp_ctx* ctx = NULL;
ufsecp_ctx_create(&ctx);

unsigned char seckey[32], pubkey[33];
ufsecp_keygen(ctx, seckey, pubkey);

unsigned char msg[32] = { /* SHA-256 hash */ };
unsigned char sig[64];
ufsecp_ecdsa_sign(ctx, seckey, msg, sig);

int valid = 0;
ufsecp_ecdsa_verify(ctx, pubkey, 33, msg, sig, &valid);

ufsecp_ctx_destroy(ctx);
```

---

## CUDA API

**Namespace:** `secp256k1::cuda`

**Header:** `#include <secp256k1.cuh>`

### Data Structures

```cpp
struct FieldElement { uint64_t limbs[4]; };
struct Scalar { uint64_t limbs[4]; };
struct JacobianPoint { FieldElement x, y, z; bool infinity; };
struct AffinePoint { FieldElement x, y; };
```

### Device Functions

| Function | Description |
|----------|-------------|
| `field_add(a, b, r)` | r = a + b |
| `field_sub(a, b, r)` | r = a - b |
| `field_mul(a, b, r)` | r = a * b |
| `field_sqr(a, r)` | r = a^2 |
| `field_inv(a, r)` | r = a^-1 |
| `jacobian_add(p, q, r)` | r = p + q |
| `jacobian_double(p, r)` | r = 2p |
| `scalar_mul(p, k, r)` | r = k*p |
| `jacobian_to_affine(p, r)` | Convert to affine |

### GPU Signature Operations (v3.6.0+)

```cpp
#include <ecdsa.cuh>
#include <schnorr.cuh>
#include <recovery.cuh>

// ECDSA (RFC 6979, low-S, Shamir + GLV)
__device__ bool ecdsa_sign(msg_hash, privkey, sig);
__device__ bool ecdsa_verify(msg_hash, pubkey, sig);
__device__ bool ecdsa_sign_recoverable(msg_hash, privkey, sig);
__device__ bool ecdsa_recover(msg_hash, sig, pubkey);

// Schnorr BIP-340 (tagged hash midstates)
__device__ bool schnorr_sign(privkey, msg, aux, sig);
__device__ bool schnorr_verify(pubkey_x, msg, sig);
```

### Hash Functions

```cpp
#include <hash160.cuh>

__device__ void hash160_compressed(const uint8_t pubkey[33], uint8_t hash[20]);
__device__ void hash160_uncompressed(const uint8_t pubkey[65], uint8_t hash[20]);
```

---

## See Also

- [[Getting Started]] - Build and installation
- [[Examples]] - Code examples
- [[Benchmarks]] - Performance data
- [[CPU Guide]] - CPU implementation details
- [[CUDA Guide]] - GPU implementation details
