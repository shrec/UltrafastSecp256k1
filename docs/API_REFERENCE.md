# UltrafastSecp256k1 API Reference

Complete API documentation for CPU, CUDA, and WASM implementations.

---

## Table of Contents

1. [CPU API](#cpu-api)
   - [FieldElement](#fieldelement)
   - [Scalar](#scalar)
   - [Point](#point)
   - [ECDSA (RFC 6979)](#ecdsa-rfc-6979)
   - [Schnorr (BIP-340)](#schnorr-bip-340)
   - [SHA-256](#sha-256)
   - [Constant-Time Layer](#constant-time-layer)
   - [Utility Functions](#utility-functions)
2. [CUDA API](#cuda-api)
   - [Data Structures](#cuda-data-structures)
   - [Field Operations](#cuda-field-operations)
   - [Point Operations](#cuda-point-operations)
   - [Batch Operations](#cuda-batch-operations)
   - [Signature Operations](#cuda-signature-operations)
3. [WASM API](#wasm-api)
4. [Performance Tips](#performance-tips)
5. [Examples](#examples)

---

## CPU API

**Namespace:** `secp256k1::fast`

**Headers:**
```cpp
#include <secp256k1/field.hpp>
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
```

---

### FieldElement

256-bit field element for secp256k1 curve (mod p where p = 2^256 - 2^32 - 977).

#### Construction

```cpp
// Zero element
FieldElement a = FieldElement::zero();

// One element
FieldElement b = FieldElement::one();

// From 64-bit integer
FieldElement c = FieldElement::from_uint64(12345);

// From 4 x 64-bit limbs (little-endian, RECOMMENDED for binary I/O)
std::array<uint64_t, 4> limbs = {0x123, 0x456, 0x789, 0xABC};
FieldElement d = FieldElement::from_limbs(limbs);

// From 32 bytes (big-endian, for hex/test vectors only)
std::array<uint8_t, 32> bytes = {...};
FieldElement e = FieldElement::from_bytes(bytes);

// From hex string (developer-friendly)
FieldElement f = FieldElement::from_hex(
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798"
);
```

#### Arithmetic Operations

```cpp
FieldElement a, b;

// Basic arithmetic (immutable, returns new object)
FieldElement sum = a + b;
FieldElement diff = a - b;
FieldElement prod = a * b;
FieldElement sq = a.square();
FieldElement inv = a.inverse();

// In-place arithmetic (mutable, ~10-15% faster)
a += b;
a -= b;
a *= b;
a.square_inplace();    // a = a²
a.inverse_inplace();   // a = a⁻¹
```

#### Serialization

```cpp
FieldElement a;

// To bytes (big-endian)
std::array<uint8_t, 32> bytes = a.to_bytes();

// To bytes into existing buffer (no allocation)
uint8_t buffer[32];
a.to_bytes_into(buffer);

// To hex string
std::string hex = a.to_hex();

// Access raw limbs (little-endian)
const auto& limbs = a.limbs();  // std::array<uint64_t, 4>
```

#### Comparison

```cpp
FieldElement a, b;
if (a == b) { ... }
if (a != b) { ... }
```

---

### Scalar

256-bit scalar for secp256k1 curve (mod n where n is the group order).

#### Construction

```cpp
// Zero
Scalar a = Scalar::zero();

// One
Scalar b = Scalar::one();

// From 64-bit integer
Scalar c = Scalar::from_uint64(12345);

// From limbs (little-endian)
std::array<uint64_t, 4> limbs = {...};
Scalar d = Scalar::from_limbs(limbs);

// From bytes (big-endian)
std::array<uint8_t, 32> bytes = {...};
Scalar e = Scalar::from_bytes(bytes);

// From hex string
Scalar f = Scalar::from_hex(
    "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
);
```

#### Arithmetic Operations

```cpp
Scalar a, b;

// Basic arithmetic
Scalar sum = a + b;
Scalar diff = a - b;
Scalar prod = a * b;

// In-place
a += b;
a -= b;
a *= b;
```

#### Utility Methods

```cpp
Scalar s;

// Check if zero
bool isZero = s.is_zero();

// Get specific bit
uint8_t bit = s.bit(index);  // 0 or 1

// NAF encoding (Non-Adjacent Form)
std::vector<int8_t> naf = s.to_naf();

// wNAF encoding (width-w NAF)
std::vector<int8_t> wnaf = s.to_wnaf(4);  // width = 4
```

---

### Point

Elliptic curve point on secp256k1 (internally Jacobian coordinates).

#### Construction

```cpp
// Generator point G
Point G = Point::generator();

// Point at infinity (identity)
Point inf = Point::infinity();

// From affine coordinates
FieldElement x = FieldElement::from_hex("...");
FieldElement y = FieldElement::from_hex("...");
Point p = Point::from_affine(x, y);

// From hex strings (developer-friendly)
Point p2 = Point::from_hex(
    "79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798",
    "483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8"
);
```

#### Point Operations

```cpp
Point p, q;
Scalar k;

// Addition and doubling
Point sum = p.add(q);       // p + q
Point doubled = p.dbl();    // 2p

// Scalar multiplication
Point result = p.scalar_mul(k);  // k * p

// Negation
Point neg = p.negate();  // -p
```

#### Optimized Scalar Multiplication

```cpp
// For fixed K × variable Q pattern (same K, different Q points):
Scalar K = Scalar::from_hex("...");
KPlan plan = KPlan::from_scalar(K);  // Precompute once

// Then for each Q:
Point Q1 = Point::from_hex("...", "...");
Point Q2 = Point::from_hex("...", "...");

Point R1 = Q1.scalar_mul_with_plan(plan);  // Fastest!
Point R2 = Q2.scalar_mul_with_plan(plan);
```

#### In-Place Operations (Fastest)

```cpp
Point p;

// Increment/decrement by generator
p.next_inplace();    // p += G
p.prev_inplace();    // p -= G

// In-place arithmetic
p.add_inplace(q);    // p += q
p.sub_inplace(q);    // p -= q
p.dbl_inplace();     // p = 2p
p.negate_inplace();  // p = -p

// Mixed addition (when q is affine, z=1)
FieldElement qx, qy;
p.add_mixed_inplace(qx, qy);  // Branchless, ~12% faster
```

#### Serialization

```cpp
Point p;

// Get affine coordinates
FieldElement x = p.x();
FieldElement y = p.y();

// Compressed format (33 bytes: 0x02/0x03 + x)
std::array<uint8_t, 33> compressed = p.to_compressed();

// Uncompressed format (65 bytes: 0x04 + x + y)
std::array<uint8_t, 65> uncompressed = p.to_uncompressed();

// Split x-coordinate for database lookups
std::array<uint8_t, 16> first_half = p.x_first_half();
std::array<uint8_t, 16> second_half = p.x_second_half();
```

#### Properties

```cpp
Point p;

// Check if point at infinity
bool isInf = p.is_infinity();

// Direct Jacobian coordinate access
const FieldElement& X = p.X();  // Jacobian X
const FieldElement& Y = p.Y();  // Jacobian Y
const FieldElement& Z = p.z();  // Jacobian Z
```

---

### Utility Functions

#### Self-Test

```cpp
#include <secp256k1/point.hpp>

// Run correctness tests
bool passed = secp256k1::fast::Selftest(true);  // verbose=true
if (!passed) {
    std::cerr << "Self-test failed!" << std::endl;
}
```

---

### ECDSA (RFC 6979)

**Namespace:** `secp256k1`

**Header:**
```cpp
#include <secp256k1/ecdsa.hpp>
```

#### ECDSASignature

```cpp
struct ECDSASignature {
    fast::Scalar r;
    fast::Scalar s;

    // DER encoding (variable length, max 72 bytes)
    std::pair<std::array<uint8_t, 72>, std::size_t> to_der() const;

    // Compact 64-byte encoding: r (32 bytes) || s (32 bytes)
    std::array<uint8_t, 64> to_compact() const;

    // Decode from compact
    static ECDSASignature from_compact(const std::array<uint8_t, 64>& data);

    // Normalize to low-S (BIP-62)
    ECDSASignature normalize() const;

    // Check low-S
    bool is_low_s() const;
};
```

#### Signing

```cpp
// Sign a 32-byte message hash with RFC 6979 deterministic nonce.
// Returns normalized (low-S) signature. Returns {0,0} on failure.
ECDSASignature ecdsa_sign(
    const std::array<uint8_t, 32>& msg_hash,
    const fast::Scalar& private_key
);
```

#### Verification

```cpp
// Verify ECDSA signature. Accepts both low-S and high-S.
bool ecdsa_verify(
    const std::array<uint8_t, 32>& msg_hash,
    const fast::Point& public_key,
    const ECDSASignature& sig
);
```

#### RFC 6979 Nonce

```cpp
// Deterministic nonce generation per RFC 6979.
fast::Scalar rfc6979_nonce(
    const fast::Scalar& private_key,
    const std::array<uint8_t, 32>& msg_hash
);
```

#### Example

```cpp
#include <secp256k1/ecdsa.hpp>
#include <secp256k1/sha256.hpp>
#include <secp256k1/point.hpp>

using namespace secp256k1;

auto msg_hash = SHA256::hash("Hello ECDSA", 11);
fast::Scalar sk = fast::Scalar::from_hex("...");
fast::Point pk = fast::Point::generator().scalar_mul(sk);

// Sign
auto sig = ecdsa_sign(msg_hash, sk);

// Verify
bool ok = ecdsa_verify(msg_hash, pk, sig);

// Compact encoding (64 bytes)
auto compact = sig.to_compact();
auto recovered = ECDSASignature::from_compact(compact);
```

---

### Schnorr (BIP-340)

**Namespace:** `secp256k1`

**Header:**
```cpp
#include <secp256k1/schnorr.hpp>
```

#### SchnorrSignature

```cpp
struct SchnorrSignature {
    std::array<uint8_t, 32> r;  // R.x (nonce point x-coordinate)
    fast::Scalar s;              // scalar s

    // 64-byte encoding: r (32) || s (32)
    std::array<uint8_t, 64> to_bytes() const;
    static SchnorrSignature from_bytes(const std::array<uint8_t, 64>& data);
};
```

#### Signing

```cpp
// BIP-340 Schnorr sign.
// aux_rand: 32 bytes of auxiliary randomness (use zeros for deterministic).
SchnorrSignature schnorr_sign(
    const fast::Scalar& private_key,
    const std::array<uint8_t, 32>& msg,
    const std::array<uint8_t, 32>& aux_rand
);
```

#### Verification

```cpp
// BIP-340 Schnorr verify with x-only public key.
bool schnorr_verify(
    const std::array<uint8_t, 32>& pubkey_x,
    const std::array<uint8_t, 32>& msg,
    const SchnorrSignature& sig
);
```

#### Utilities

```cpp
// X-only public key (BIP-340: negate if Y is odd)
std::array<uint8_t, 32> schnorr_pubkey(const fast::Scalar& private_key);

// Tagged hash: H_tag(msg) = SHA256(SHA256(tag) || SHA256(tag) || msg)
std::array<uint8_t, 32> tagged_hash(
    const char* tag, const void* data, std::size_t len
);
```

#### Example

```cpp
#include <secp256k1/schnorr.hpp>

using namespace secp256k1;

fast::Scalar sk = fast::Scalar::from_hex("...");
auto pk_x = schnorr_pubkey(sk);

std::array<uint8_t, 32> msg = { /* message hash */ };
std::array<uint8_t, 32> aux = {}; // zeros for deterministic

auto sig = schnorr_sign(sk, msg, aux);
bool ok = schnorr_verify(pk_x, msg, sig);
```

---

### SHA-256

**Namespace:** `secp256k1`

**Header:**
```cpp
#include <secp256k1/sha256.hpp>
```

#### One-shot Hashing

```cpp
// SHA-256
SHA256::digest_type SHA256::hash(const void* data, std::size_t len);

// Double-SHA256: SHA256(SHA256(data))
SHA256::digest_type SHA256::hash256(const void* data, std::size_t len);
```

#### Streaming API

```cpp
secp256k1::SHA256 ctx;
ctx.update("part1", 5);
ctx.update("part2", 5);
auto digest = ctx.finalize();

// Reuse
ctx.reset();
ctx.update("new data", 8);
auto digest2 = ctx.finalize();
```

#### Example

```cpp
#include <secp256k1/sha256.hpp>

auto hash = secp256k1::SHA256::hash("Hello, world!", 13);
// hash is std::array<uint8_t, 32>

// Double-SHA256 (Bitcoin's hash)
auto hash256 = secp256k1::SHA256::hash256("tx_data", 7);
```

---

### Constant-Time Layer

**Namespace:** `secp256k1::fast::ct`

**Headers:**
```cpp
#include <secp256k1/ct/field.hpp>
#include <secp256k1/ct/scalar.hpp>
#include <secp256k1/ct/point.hpp>
#include <secp256k1/ct/ops.hpp>
```

The CT layer provides side-channel resistant variants of critical operations:

```cpp
namespace secp256k1::fast::ct {

// Constant-time field operations
void field_mul(const FieldElement& a, const FieldElement& b, FieldElement& out);
void field_inv(const FieldElement& a, FieldElement& out);

// Constant-time scalar multiplication (branchless double-and-add)
void scalar_mul(const Point& base, const Scalar& k, Point& out);

// Complete addition formula (handles all edge cases without branching)
void point_add_complete(const Point& p, const Point& q, Point& out);

// Constant-time point doubling
void point_dbl(const Point& p, Point& out);

} // namespace secp256k1::fast::ct
```

> ⚠️ CT operations are ~5-7× slower than the fast variants. Use only for private key operations (signing, ECDH).

---

## CUDA API

**Namespace:** `secp256k1::cuda`

**Header:**
```cpp
#include <secp256k1.cuh>
```

---

### CUDA Data Structures

```cpp
// Field element (4 × 64-bit limbs, little-endian)
struct FieldElement {
    uint64_t limbs[4];
};

// Scalar (4 × 64-bit limbs)
struct Scalar {
    uint64_t limbs[4];
};

// Jacobian point (X, Y, Z)
struct JacobianPoint {
    FieldElement x;
    FieldElement y;
    FieldElement z;
    bool infinity;
};

// Affine point (x, y)
struct AffinePoint {
    FieldElement x;
    FieldElement y;
};

// 32-bit view for optimized operations (zero-cost conversion)
struct MidFieldElement {
    uint32_t limbs[8];
};
```

---

### CUDA Field Operations

All functions are `__device__` and can only be called from GPU kernels.

```cpp
// Initialization
__device__ void field_set_zero(FieldElement* r);
__device__ void field_set_one(FieldElement* r);

// Comparison
__device__ bool field_is_zero(const FieldElement* a);
__device__ bool field_eq(const FieldElement* a, const FieldElement* b);

// Arithmetic
__device__ void field_add(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_sub(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_mul(const FieldElement* a, const FieldElement* b, FieldElement* r);
__device__ void field_sqr(const FieldElement* a, FieldElement* r);
__device__ void field_inv(const FieldElement* a, FieldElement* r);
__device__ void field_neg(const FieldElement* a, FieldElement* r);

// Domain conversion (Montgomery mode only)
__device__ void field_to_mont(const FieldElement* a, FieldElement* r);
__device__ void field_from_mont(const FieldElement* a, FieldElement* r);
```

---

### CUDA Point Operations

```cpp
// Initialization
__device__ void jacobian_set_infinity(JacobianPoint* p);
__device__ void jacobian_set_generator(JacobianPoint* p);
__device__ bool jacobian_is_infinity(const JacobianPoint* p);

// Point arithmetic
__device__ void jacobian_double(const JacobianPoint* p, JacobianPoint* r);
__device__ void jacobian_add(const JacobianPoint* p, const JacobianPoint* q, JacobianPoint* r);
__device__ void jacobian_add_mixed(const JacobianPoint* p, const AffinePoint* q, JacobianPoint* r);

// Scalar multiplication
__device__ void scalar_mul(const JacobianPoint* p, const Scalar* k, JacobianPoint* r);
__device__ void scalar_mul_generator(const Scalar* k, JacobianPoint* r);

// Conversion
__device__ void jacobian_to_affine(const JacobianPoint* p, AffinePoint* r);
```

---

### CUDA Batch Operations

```cpp
#include <batch_inversion.cuh>

// Batch field inversion (Montgomery's trick)
// Inverts n field elements using only 1 modular inversion + 3(n-1) multiplications
__device__ void batch_invert(FieldElement* elements, int n, FieldElement* scratch);
```

---

### CUDA Hash Operations

```cpp
#include <hash160.cuh>

// Compute HASH160 = RIPEMD160(SHA256(pubkey))
__device__ void hash160_compressed(const uint8_t pubkey[33], uint8_t hash[20]);
__device__ void hash160_uncompressed(const uint8_t pubkey[65], uint8_t hash[20]);
```

---

### CUDA Signature Operations

> **World-first:** No other open-source GPU library provides secp256k1 ECDSA + Schnorr sign/verify.

#### Data Structures

```cpp
#include <ecdsa.cuh>
#include <schnorr.cuh>
#include <recovery.cuh>

// ECDSA signature (r, s as Scalars)
struct ECDSASignatureGPU {
    Scalar r;
    Scalar s;
};

// Schnorr BIP-340 signature (32-byte R x-coordinate + Scalar s)
struct SchnorrSignatureGPU {
    uint8_t r[32];  // x-coordinate of R point
    Scalar s;
};

// Recoverable ECDSA signature
struct RecoverableSignatureGPU {
    ECDSASignatureGPU sig;
    int recid;  // Recovery ID (0-3)
};
```

#### Device Functions

```cpp
// ECDSA Sign (RFC 6979 deterministic nonces, low-S normalization)
// Returns true on success
__device__ bool ecdsa_sign(
    const uint8_t msg_hash[32],   // 32-byte message hash
    const Scalar* privkey,         // Private key
    ECDSASignatureGPU* sig         // Output signature
);

// ECDSA Verify (Shamir's trick + GLV endomorphism)
// Returns true if signature is valid
__device__ bool ecdsa_verify(
    const uint8_t msg_hash[32],   // 32-byte message hash
    const JacobianPoint* pubkey,   // Public key (Jacobian)
    const ECDSASignatureGPU* sig   // Signature to verify
);

// ECDSA Sign with Recovery ID
__device__ bool ecdsa_sign_recoverable(
    const uint8_t msg_hash[32],
    const Scalar* privkey,
    RecoverableSignatureGPU* sig   // Output: signature + recid
);

// ECDSA Recover public key from signature
__device__ bool ecdsa_recover(
    const uint8_t msg_hash[32],
    const RecoverableSignatureGPU* sig,
    JacobianPoint* pubkey          // Output: recovered public key
);

// Schnorr Sign (BIP-340, tagged hash midstates for performance)
__device__ bool schnorr_sign(
    const Scalar* privkey,
    const uint8_t msg[32],
    const uint8_t aux_rand[32],    // Auxiliary randomness
    SchnorrSignatureGPU* sig       // Output signature
);

// Schnorr Verify (BIP-340, x-only pubkey)
__device__ bool schnorr_verify(
    const uint8_t pubkey_x[32],    // X-only public key (32 bytes)
    const uint8_t msg[32],
    const SchnorrSignatureGPU* sig
);
```

#### Batch Kernel Wrappers

Host-callable kernel wrappers for batch processing:

```cpp
// Launch batch ECDSA sign (128 threads/block, 2 blocks/SM)
void ecdsa_sign_batch_kernel<<<blocks, 128>>>(
    const uint8_t* msg_hashes,     // N × 32 bytes
    const Scalar* privkeys,         // N scalars
    ECDSASignatureGPU* sigs,        // N output signatures
    int count
);

// Launch batch ECDSA verify
void ecdsa_verify_batch_kernel<<<blocks, 128>>>(
    const uint8_t* msg_hashes,
    const JacobianPoint* pubkeys,
    const ECDSASignatureGPU* sigs,
    bool* results,                  // N output booleans
    int count
);

// Launch batch Schnorr sign
void schnorr_sign_batch_kernel<<<blocks, 128>>>(
    const Scalar* privkeys,
    const uint8_t* msgs,
    const uint8_t* aux_rands,
    SchnorrSignatureGPU* sigs,
    int count
);

// Launch batch Schnorr verify
void schnorr_verify_batch_kernel<<<blocks, 128>>>(
    const uint8_t* pubkey_xs,
    const uint8_t* msgs,
    const SchnorrSignatureGPU* sigs,
    bool* results,
    int count
);
```

#### Performance

| Operation | Time/Op | Throughput |
|-----------|---------|------------|
| ECDSA Sign | 204.8 ns | 4.88 M/s |
| ECDSA Verify | 410.1 ns | 2.44 M/s |
| ECDSA Sign + Recid | 311.5 ns | 3.21 M/s |
| Schnorr Sign | 273.4 ns | 3.66 M/s |
| Schnorr Verify | 354.6 ns | 2.82 M/s |

*RTX 5060 Ti, kernel-only timing, batch 16K*

---

## WASM API

**Module:** `@ultrafastsecp256k1/wasm`

**Usage:**
```javascript
import { Secp256k1 } from './secp256k1.mjs';
const lib = await Secp256k1.create();
```

### Functions

| Function | Parameters | Returns | Description |
|----------|-----------|---------|-------------|
| `selftest()` | — | `boolean` | Run built-in self-test |
| `version()` | — | `string` | Library version (`"3.0.0"`) |
| `pubkeyCreate(seckey)` | `Uint8Array(32)` | `{x, y}` | Public key from private key |
| `pointMul(px, py, scalar)` | `Uint8Array(32)` × 3 | `{x, y}` | Scalar × Point |
| `pointAdd(px, py, qx, qy)` | `Uint8Array(32)` × 4 | `{x, y}` | Point addition |
| `ecdsaSign(msgHash, seckey)` | `Uint8Array(32)` × 2 | `Uint8Array(64)` | ECDSA sign (r‖s) |
| `ecdsaVerify(msgHash, pubX, pubY, sig)` | `Uint8Array(32)` × 3 + `Uint8Array(64)` | `boolean` | ECDSA verify |
| `schnorrSign(seckey, msg, aux?)` | `Uint8Array(32)` × 2-3 | `Uint8Array(64)` | Schnorr BIP-340 sign |
| `schnorrVerify(pubkeyX, msg, sig)` | `Uint8Array(32)` × 2 + `Uint8Array(64)` | `boolean` | Schnorr verify |
| `schnorrPubkey(seckey)` | `Uint8Array(32)` | `Uint8Array(32)` | X-only public key |
| `sha256(data)` | `Uint8Array` | `Uint8Array(32)` | SHA-256 hash |

### C API

For direct C/C++ or custom WASM bindings, see [secp256k1_wasm.h](../wasm/secp256k1_wasm.h).

### Example

```javascript
const lib = await Secp256k1.create();
console.log('v' + lib.version(), lib.selftest() ? '✓' : '✗');

// ECDSA workflow
const privkey = new Uint8Array(32);
privkey[31] = 1;
const { x, y } = lib.pubkeyCreate(privkey);
const msgHash = lib.sha256(new TextEncoder().encode('Hello'));
const sig = lib.ecdsaSign(msgHash, privkey);
const valid = lib.ecdsaVerify(msgHash, x, y, sig);
```

See [wasm/README.md](../wasm/README.md) for detailed build and usage instructions.

---

## Performance Tips

### CPU

1. **Use in-place operations** when possible:
   ```cpp
   // Slower: creates temporary
   point = point.add(other);
   
   // Faster: no allocation
   point.add_inplace(other);
   ```

2. **Use KPlan for fixed-K multiplication**:
   ```cpp
   // If K is constant and Q varies, precompute K once
   KPlan plan = KPlan::from_scalar(K);
   for (auto& Q : points) {
       result = Q.scalar_mul_with_plan(plan);
   }
   ```

3. **Use `from_limbs` for binary I/O** (not `from_bytes`):
   ```cpp
   // Database/binary files: use from_limbs (native little-endian)
   FieldElement::from_limbs(limbs);
   
   // Hex strings/test vectors: use from_bytes (big-endian)
   FieldElement::from_bytes(bytes);
   ```

### CUDA

1. **Batch operations**: Process thousands of points in parallel
2. **Avoid divergence**: Use branchless algorithms where possible
3. **Memory coalescing**: Align data structures to 32/64 bytes
4. **Use hybrid 32-bit multiplication**: Enabled by default (`SECP256K1_CUDA_USE_HYBRID_MUL=1`)

---

## Examples

### Generate Bitcoin Address (CPU)

```cpp
#include <secp256k1/point.hpp>
#include <secp256k1/scalar.hpp>

using namespace secp256k1::fast;

int main() {
    // Private key (256-bit)
    Scalar private_key = Scalar::from_hex(
        "E9873D79C6D87DC0FB6A5778633389F4453213303DA61F20BD67FC233AA33262"
    );
    
    // Public key = private_key × G
    Point G = Point::generator();
    Point public_key = G.scalar_mul(private_key);
    
    // Get compressed public key (33 bytes)
    auto compressed = public_key.to_compressed();
    
    // Print as hex
    for (auto byte : compressed) {
        printf("%02x", byte);
    }
    printf("\n");
    
    return 0;
}
```

### Batch Point Generation (CUDA)

```cpp
#include <secp256k1.cuh>

using namespace secp256k1::cuda;

__global__ void generate_points_kernel(
    const Scalar* private_keys,
    AffinePoint* public_keys,
    int count
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    JacobianPoint result;
    scalar_mul_generator(&private_keys[idx], &result);
    jacobian_to_affine(&result, &public_keys[idx]);
}

void generate_points(
    const Scalar* d_private_keys,
    AffinePoint* d_public_keys,
    int count
) {
    int threads = 256;
    int blocks = (count + threads - 1) / threads;
    generate_points_kernel<<<blocks, threads>>>(
        d_private_keys, d_public_keys, count
    );
}
```

### Verify Self-Test (CPU)

```cpp
#include <secp256k1/point.hpp>
#include <iostream>

int main() {
    bool ok = secp256k1::fast::Selftest(true);
    if (ok) {
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } else {
        std::cerr << "TESTS FAILED!" << std::endl;
        return 1;
    }
}
```

---

## Build Configuration Macros

### CPU

| Macro | Default | Description |
|-------|---------|-------------|
| `SECP256K1_USE_ASM` | ON | Enable x64/RISC-V assembly |
| `SECP256K1_RISCV_FAST_REDUCTION` | ON | Fast modular reduction (RISC-V) |
| `SECP256K1_RISCV_USE_VECTOR` | ON | RVV vector extension |

### CUDA

| Macro | Default | Description |
|-------|---------|-------------|
| `SECP256K1_CUDA_USE_HYBRID_MUL` | 1 | 32-bit hybrid multiplication (~10% faster) |
| `SECP256K1_CUDA_USE_MONTGOMERY` | 0 | Montgomery domain arithmetic |
| `SECP256K1_CUDA_LIMBS_32` | 0 | Use 8×32-bit limbs (experimental) |

---

## Platform Support

| Platform | Assembly | SIMD | Status |
|----------|----------|------|--------|
| x86-64 Linux/Windows/macOS | BMI2/ADX | AVX2 | ✅ Production |
| RISC-V 64 | RV64GC | RVV 1.0 | ✅ Production |
| ARM64 (Android/iOS/macOS) | MUL/UMULH | NEON | ✅ Production |
| CUDA (sm_75+) | PTX | — | ✅ Production |
| ROCm/HIP (AMD) | Portable | — | ✅ CI |
| OpenCL 3.0 | PTX | — | ✅ Production |
| WebAssembly | Portable | — | ✅ Production |
| ESP32-S3 / ESP32 | Portable | — | ✅ Tested |
| STM32F103 (Cortex-M3) | UMULL | — | ✅ Tested |

---

## Version

UltrafastSecp256k1 v3.6.0

For more information, see the [README](../README.md) or [GitHub repository](https://github.com/shrec/UltrafastSecp256k1).

