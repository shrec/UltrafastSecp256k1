# API Reference

Complete API documentation for UltrafastSecp256k1.

---

## CPU API

**Namespace:** `secp256k1::fast`

**Headers:**
```cpp
#include <secp256k1/field.hpp>   // FieldElement
#include <secp256k1/scalar.hpp>  // Scalar
#include <secp256k1/point.hpp>   // Point, KPlan, Selftest
```

---

## FieldElement

256-bit field element for secp256k1 (mod p where p = 2²⁵⁶ - 2³² - 977).

### Construction

| Method | Description |
|--------|-------------|
| `FieldElement::zero()` | Zero element |
| `FieldElement::one()` | One element |
| `FieldElement::from_uint64(val)` | From 64-bit integer |
| `FieldElement::from_limbs(arr)` | From 4×64-bit array (little-endian) |
| `FieldElement::from_bytes(arr)` | From 32 bytes (big-endian) |
| `FieldElement::from_hex(str)` | From hex string (64 chars) |

### Arithmetic

| Operator/Method | Description |
|-----------------|-------------|
| `a + b` | Addition |
| `a - b` | Subtraction |
| `a * b` | Multiplication |
| `a.square()` | Squaring (a²) |
| `a.inverse()` | Modular inverse (a⁻¹) |
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
| `Scalar::from_limbs(arr)` | From 4×64-bit array |
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
| `p.scalar_mul(k)` | Scalar multiplication (k×p) |
| `p.negate()` | Negation (-p) |

### Optimized Operations

| Method | Description |
|--------|-------------|
| `p.scalar_mul_with_plan(plan)` | Fixed-K multiplication (fastest) |
| `p.next()` | p + G |
| `p.prev()` | p - G |

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

// Use for multiple Q values
Point R1 = Q1.scalar_mul_with_plan(plan);
Point R2 = Q2.scalar_mul_with_plan(plan);
```

---

## Selftest

```cpp
// Run correctness tests
bool ok = secp256k1::fast::Selftest(true);  // verbose=true
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
| `field_mul(a, b, r)` | r = a × b |
| `field_sqr(a, r)` | r = a² |
| `field_inv(a, r)` | r = a⁻¹ |
| `jacobian_add(p, q, r)` | r = p + q |
| `jacobian_double(p, r)` | r = 2p |
| `scalar_mul(p, k, r)` | r = k×p |
| `jacobian_to_affine(p, r)` | Convert to affine |

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

