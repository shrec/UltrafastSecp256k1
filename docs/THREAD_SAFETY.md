# Thread-Safety Guarantees

**UltrafastSecp256k1** -- Concurrency Model & Thread-Safety Documentation

---

## 1. Overview

UltrafastSecp256k1 is designed for high-throughput concurrent use. This document details the thread-safety guarantees for each layer. The library follows a **shared-nothing** model: separate contexts and data structures can be used from different threads without synchronization.

---

## 2. Thread-Safety Classification

| Classification | Meaning |
|---------------|---------|
| **Thread-safe** | Can be called from multiple threads concurrently without synchronization |
| **Thread-compatible** | Safe if each thread uses its own instance (no sharing without synchronization) |
| **Not thread-safe** | Requires external synchronization for concurrent use |

---

## 3. Core Arithmetic Layer

### 3.1 Pure Functions (Thread-Safe)

All pure computation functions in `secp256k1::fast::` and `secp256k1::ct::` namespaces are **thread-safe**:

| Function Category | Thread Safety | Notes |
|-------------------|--------------|-------|
| `FieldElement` arithmetic (add, sub, mul, square, inv, sqrt) | [OK] Thread-safe | Pure functions, no global state |
| `Scalar` arithmetic (add, sub, mul, inv, negate) | [OK] Thread-safe | Pure functions, no global state |
| `Point` operations (add, double, scalar_mul, to_affine) | [OK] Thread-safe | Pure functions, no global state |
| GLV decomposition | [OK] Thread-safe | Uses only stack-local computation |
| Hamburg comb (generator mul) | [OK] Thread-safe | Reads precomputed table (const after init) |
| Batch inversion | [OK] Thread-safe | Caller provides scratch buffer |

**Guarantee**: Any function that takes `const&` inputs and returns by value or writes to caller-provided output buffers is thread-safe. No global mutable state is accessed.

### 3.2 Read-Only Globals

| Global | Access Pattern | Thread Safety |
|--------|---------------|---------------|
| Generator point `G` | Read-only after static init | [OK] Thread-safe |
| Precomputed comb table | Read-only after static init | [OK] Thread-safe |
| Field prime `p` | Compile-time constant | [OK] Thread-safe |
| Group order `n` | Compile-time constant | [OK] Thread-safe |
| Endomorphism constants `lambda`, `beta` | Compile-time constant | [OK] Thread-safe |

---

## 4. Signature Operations

| Function | Thread Safety | Notes |
|----------|--------------|-------|
| `ecdsa_sign(msg, sk)` | [OK] Thread-safe | RFC 6979: deterministic, pure function |
| `ecdsa_verify(msg, sig, pk)` | [OK] Thread-safe | Pure computation |
| `schnorr_sign(msg, sk, aux)` | [OK] Thread-safe | BIP-340: deterministic with aux randomness |
| `schnorr_verify(msg, sig, pk)` | [OK] Thread-safe | Pure computation |
| `ct::ecdsa_sign(msg, sk)` | [OK] Thread-safe | CT variant, same guarantees |
| `ct::schnorr_sign(msg, sk, aux)` | [OK] Thread-safe | CT variant, same guarantees |

**Note**: RFC 6979 nonce generation uses only message + key inputs (no RNG state), ensuring determinism and thread safety.

---

## 5. Multi-Party Protocols

### 5.1 MuSig2

| Function | Thread Safety | Notes |
|----------|--------------|-------|
| `musig2_key_agg(pubkeys)` | [OK] Thread-safe | Pure computation |
| `musig2_nonce_gen(sk, pk, msg)` | [!] Thread-compatible | Reads from system RNG if aux randomness not provided; use separate RNG per thread |
| `musig2_partial_sign(...)` | [OK] Thread-safe | Given pre-generated nonces |
| `musig2_partial_verify(...)` | [OK] Thread-safe | Pure computation |
| `musig2_aggregate(...)` | [OK] Thread-safe | Pure computation |

### 5.2 FROST

| Function | Thread Safety | Notes |
|----------|--------------|-------|
| `frost_keygen_begin(id, t, n, seed)` | [OK] Thread-safe | Deterministic from seed |
| `frost_keygen_finalize(...)` | [OK] Thread-safe | Pure verification + computation |
| `frost_sign_nonce_gen(id, seed)` | [OK] Thread-safe | Deterministic from seed |
| `frost_sign(key_pkg, nonce, msg, ...)` | [OK] Thread-safe | Pure computation |
| `frost_verify_partial(...)` | [OK] Thread-safe | Pure computation |
| `frost_aggregate(...)` | [OK] Thread-safe | Pure computation |
| `frost_lagrange_coefficient(i, ids)` | [OK] Thread-safe | Pure computation |

**Protocol note**: FROST DKG requires coordinated communication between participants. The library functions themselves are thread-safe, but the protocol coordination (message passing) must be handled by the caller.

---

## 6. BIP-32 HD Derivation

| Function | Thread Safety | Notes |
|----------|--------------|-------|
| `bip32_master_from_seed(seed)` | [OK] Thread-safe | Pure HMAC-SHA512 computation |
| `bip32_derive_child(parent, index)` | [OK] Thread-safe | Pure computation |
| `bip32_derive_path(master, path)` | [OK] Thread-safe | Sequential derivation, no shared state |
| `bip32_parse_path(path_string)` | [OK] Thread-safe | Pure string parsing |

---

## 7. Address Generation

| Function | Thread Safety | Notes |
|----------|--------------|-------|
| `address_p2pkh(pubkey, network)` | [OK] Thread-safe | Pure computation (SHA-256 + RIPEMD-160 + Base58Check) |
| `address_p2wpkh(pubkey, network)` | [OK] Thread-safe | Pure computation (SHA-256 + RIPEMD-160 + Bech32) |
| `address_p2tr(pubkey, network)` | [OK] Thread-safe | Pure computation (Bech32m) |
| `wif_encode(privkey)` | [OK] Thread-safe | Pure computation |
| `wif_decode(wif_string)` | [OK] Thread-safe | Pure computation |

---

## 8. C ABI (ufsecp)

### 8.1 Context

| Function | Thread Safety | Notes |
|----------|--------------|-------|
| `ufsecp_context_create()` | [OK] Thread-safe | Returns new independent context |
| `ufsecp_context_destroy(ctx)` | [!] Thread-compatible | Do not destroy from two threads simultaneously |
| `ufsecp_context_clone(ctx)` | [!] Thread-compatible | Source must not be modified during clone |

### 8.2 Context Usage Rules

**The `ufsecp_context` object is NOT thread-safe.** Each thread must use its own context:

```c
// [OK] CORRECT: One context per thread
void worker_thread(void) {
    ufsecp_context* ctx = ufsecp_context_create();
    // ... use ctx ...
    ufsecp_context_destroy(ctx);
}

// [FAIL] WRONG: Sharing context across threads
ufsecp_context* shared_ctx;  // NOT safe!
void thread_a(void) { ufsecp_ecdsa_sign(shared_ctx, ...); }
void thread_b(void) { ufsecp_ecdsa_verify(shared_ctx, ...); }
```

### 8.3 Operation Functions

When called with separate contexts, all C ABI functions are thread-safe:

| Function | Thread Safety (separate contexts) |
|----------|-----------------------------------|
| `ufsecp_pubkey_create` | [OK] Thread-safe |
| `ufsecp_ecdsa_sign` | [OK] Thread-safe |
| `ufsecp_ecdsa_verify` | [OK] Thread-safe |
| `ufsecp_schnorr_sign` | [OK] Thread-safe |
| `ufsecp_schnorr_verify` | [OK] Thread-safe |
| `ufsecp_ecdh` | [OK] Thread-safe |
| `ufsecp_seckey_tweak_add/mul` | [OK] Thread-safe |
| All address functions | [OK] Thread-safe |
| All BIP-32 functions | [OK] Thread-safe |

### 8.4 Error State

`ufsecp_last_error()` returns the last error for a given context. Since contexts are per-thread, error state is also per-thread.

---

## 9. GPU Backends

| Backend | Thread Safety | Notes |
|---------|--------------|-------|
| CUDA | [!] Thread-compatible | One CUDA context per host thread (CUDA runtime default) |
| OpenCL | [!] Thread-compatible | Command queues are per-thread; shared `cl_context` requires synchronization |
| Metal | [!] Thread-compatible | Metal command buffers can be created from any thread |
| ROCm/HIP | [!] Thread-compatible | Similar model to CUDA |

**Rule**: Each host thread should manage its own GPU resources. Do not share GPU buffers across threads without explicit synchronization.

---

## 10. Hash Functions

| Function | Thread Safety | Notes |
|----------|--------------|-------|
| `sha256(data, len)` | [OK] Thread-safe | Pure function |
| `sha256_tagged(tag, data)` | [OK] Thread-safe | Pure function |
| `ripemd160(data, len)` | [OK] Thread-safe | Pure function |
| `hmac_sha512(key, data)` | [OK] Thread-safe | Pure function; used by BIP-32 |

---

## 11. Summary Table

| Component | Safety Level | Recommendation |
|-----------|-------------|----------------|
| Field/Scalar/Point math | Thread-safe | Use freely from any thread |
| ECDSA/Schnorr sign/verify | Thread-safe | Use freely from any thread |
| CT layer | Thread-safe | Use freely from any thread |
| MuSig2 (with pre-gen nonces) | Thread-safe | Pre-generate nonces outside hot loop |
| FROST (all functions) | Thread-safe | Coordinate protocol messages externally |
| BIP-32 / Addresses | Thread-safe | Use freely from any thread |
| Hash functions | Thread-safe | Use freely from any thread |
| `ufsecp_context` | Thread-compatible | One context per thread |
| GPU backends | Thread-compatible | One device context per thread |

---

## 12. Verified By

- **TSan (ThreadSanitizer)**: Enabled in CI (`security-audit.yml`)
- **Code inspection**: No global mutable state in core arithmetic
- **Architecture**: Shared-nothing design, caller-owned buffers
- **dudect**: CT operations verified timing-independent (no thread interference)
