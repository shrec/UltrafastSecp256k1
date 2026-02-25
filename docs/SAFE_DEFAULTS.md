# Safe Defaults

UltrafastSecp256k1 ships with safe defaults that prioritize correctness over
performance. This document lists every default and explains when (and how) to
override it.

---

## Design Principle

> **If the user does nothing special, the library should be correct and safe.**
>
> Performance-critical overrides exist but require explicit opt-in.

---

## Build Defaults

| Setting | Default | Safe? | Override |
|---------|---------|-------|----------|
| `CMAKE_BUILD_TYPE` | None (Debug) | [OK] | Set `Release` for production |
| `SECP256K1_USE_ASM` | `ON` | [OK] | `OFF` for portable builds |
| `SECP256K1_BUILD_SHARED` | `OFF` | [OK] | `ON` for shared libraries |
| `SECP256K1_BUILD_TESTS` | `ON` | [OK] | `OFF` for production |
| `SECP256K1_BUILD_BENCH` | `ON` | [OK] | `OFF` for production |
| `SECP256K1_SPEED_FIRST` | `OFF` | [OK] | `ON` enables unsafe fast-math; **never for crypto** |
| `SECP256K1_REQUIRE_CT` | `0` | [!] | Set `1` to compile-error on non-CT signing |
| `SECP256K1_VERBOSE_DEBUG` | `OFF` | [OK] | Only for development |

### Recommended Production Build

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_USE_ASM=ON \
  -DSECP256K1_BUILD_TESTS=OFF \
  -DSECP256K1_BUILD_BENCH=OFF \
  -DSECP256K1_BUILD_EXAMPLES=OFF \
  -DSECP256K1_REQUIRE_CT=1
```

---

## Runtime Defaults

### Signature Verification

| Behavior | Default | Notes |
|----------|---------|-------|
| ECDSA low-S check | Enforced on sign | Signatures always low-S |
| ECDSA low-S verify | Accepts both | BIP-62 optional; set strict mode if needed |
| DER encoding | Strict | Rejects non-canonical DER |
| Schnorr BIP-340 | Even-Y nonce | Per specification |

### Key Validation

| Behavior | Default | Notes |
|----------|---------|-------|
| Private key range check | Always | Rejects 0 and >= n |
| Public key on-curve check | Always | Rejects invalid points |
| Point-at-infinity check | Always | Rejects infinity pubkeys |
| BIP-32 key validation | Always | Checks chain code + key bytes |

### Nonce Generation

| Behavior | Default | Notes |
|----------|---------|-------|
| ECDSA | RFC 6979 deterministic | No RNG needed |
| Schnorr | BIP-340 auxiliary rand | Caller provides aux_rand |
| MuSig2 | Seed-based CSPRNG | Caller provides seed |
| FROST | Seed-based CSPRNG | Caller provides seed |

**No operation generates random numbers internally.** All randomness must come
from the caller. This is a security design choice -- the library never silently
uses a potentially weak system RNG.

---

## Memory Defaults

| Behavior | Default | Notes |
|----------|---------|-------|
| Hot-path allocation | Zero | No malloc/new in arithmetic |
| Secret zeroing | Caller responsibility | Library does not auto-zero |
| Stack buffer size | Platform-appropriate | 4 MB on Windows test builds |

### Secret Zeroing

```cpp
// The library does NOT auto-zero. Always zero secrets explicitly:
Scalar private_key = Scalar::from_bytes(key_bytes);
// ... use private_key ...
std::memset(&private_key, 0, sizeof(private_key));
// Or use a secure_zero function from your platform
```

**Why not auto-zero?** Automatic zeroing in destructors can be optimized away by
compilers. Explicit zeroing with volatile or platform-specific APIs is the only
reliable approach, and is the caller's responsibility.

---

## CT (Constant-Time) Defaults

| Behavior | Default | Notes |
|----------|---------|-------|
| Default namespace | `secp256k1::fast` | Variable-time (fast) |
| CT namespace | `secp256k1::ct` | Constant-time (slow) |
| Signing default | FAST | Use `ct::ecdsa_sign` for CT |
| `SECP256K1_REQUIRE_CT` | `0` | Set `1` to enforce CT signing |

### Safe CT Configuration

For high-security applications:

```bash
# Compile-time: error if non-CT sign is used
-DSECP256K1_REQUIRE_CT=1
```

```cpp
// Runtime: always use ct:: namespace for secret-dependent ops
#include "secp256k1/ct/sign.hpp"
auto sig = secp256k1::ct::ecdsa_sign(private_key, message_hash);
```

---

## GPU Defaults

| Behavior | Default | Notes |
|----------|---------|-------|
| Device selection | `device_id: 0` | First GPU |
| threads_per_batch | `131072` | Adjust to GPU SM count |
| Constant-time | **NOT CT** | Never for secrets |

**Security warning**: GPU operations are variable-time by design. Never process
private keys on GPU. GPU is for public-key batch operations only (key search,
batch verification, etc.).

---

## Protocol Defaults

### MuSig2

| Behavior | Default | Notes |
|----------|---------|-------|
| Key format | x-only (32 bytes) | Diverges from BIP-327's 33-byte |
| Nonce reuse prevention | Caller responsibility | Library does not track nonces |

### FROST

| Behavior | Default | Notes |
|----------|---------|-------|
| Key generation | Pedersen DKG | Fully distributed |
| Share verification | Feldman VSS | Automatic during finalize |
| Nonce reuse prevention | Caller responsibility | **Fatal: reuse leaks secret** |

---

## What to Change for Production

| Change | Why | How |
|--------|-----|-----|
| Set `REQUIRE_CT=1` | Prevent accidental non-CT signing | Build flag |
| Build `Release` | Enable optimizations | `-DCMAKE_BUILD_TYPE=Release` |
| Disable tests/bench | Smaller binary, faster build | Build flags |
| Zero secrets after use | Prevent memory disclosure | `memset_s` or platform API |
| Pin compiler version | Reproducible builds | Lock in CI |
| Use batch operations | 10-50x faster for bulk work | API choice |

---

## See Also

- [docs/CT_VERIFICATION.md](CT_VERIFICATION.md) -- Constant-time verification details
- [docs/THREAD_SAFETY.md](THREAD_SAFETY.md) -- Concurrency guarantees
- [docs/PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) -- Tuning for speed
- [SECURITY.md](../SECURITY.md) -- Security policy
