# UltrafastSecp256k1 — User Guide

> Getting started with the fastest open-source secp256k1 library.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Building from Source](#2-building-from-source)
3. [C API Usage](#3-c-api-usage)
4. [C++ API Usage](#4-c-api-usage-1)
5. [Key Operations](#5-key-operations)
6. [ECDSA Signatures](#6-ecdsa-signatures)
7. [Schnorr / BIP-340 Signatures](#7-schnorr--bip-340-signatures)
8. [ECDH Key Agreement](#8-ecdh-key-agreement)
9. [Address Generation](#9-address-generation)
10. [HD Key Derivation (BIP-32)](#10-hd-key-derivation-bip-32)
11. [MuSig2 Multi-Signatures](#11-musig2-multi-signatures)
12. [FROST Threshold Signatures](#12-frost-threshold-signatures)
13. [Taproot](#13-taproot)
14. [Thread Safety](#14-thread-safety)
15. [Error Handling](#15-error-handling)
16. [Platform-Specific Notes](#16-platform-specific-notes)
17. [Troubleshooting](#17-troubleshooting)

---

## 1. Quick Start

### Install via Package Manager

```bash
# vcpkg
vcpkg install ultrafastsecp256k1

# Conan
conan install ultrafastsecp256k1/3.14.0@

# Cargo (Rust binding)
cargo add ultrafastsecp256k1

# npm (WASM)
npm install @ultrafastsecp256k1/wasm

# pip (Python)
pip install ultrafastsecp256k1
```

### Minimal C Example

```c
#include <ufsecp/ufsecp.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    ufsecp_ctx* ctx;
    ufsecp_ctx_create(&ctx);

    // Generate a public key from a private key
    uint8_t privkey[32] = {0};
    privkey[31] = 1;  // key = 1 (for demonstration only!)

    uint8_t pubkey[33];
    ufsecp_pubkey_create(ctx, privkey, pubkey);

    // Sign a message
    uint8_t msg[32] = {0};
    msg[0] = 0x42;
    uint8_t sig[64];
    ufsecp_ecdsa_sign(ctx, msg, privkey, sig);

    // Verify the signature
    ufsecp_error_t err = ufsecp_ecdsa_verify(ctx, msg, sig, pubkey);
    printf("Verify: %s\n", err == UFSECP_OK ? "OK" : "FAIL");

    ufsecp_ctx_destroy(ctx);
    return 0;
}
```

### Build & Link

```bash
# CMake FetchContent (recommended)
cmake_minimum_required(VERSION 3.20)
project(myapp)

include(FetchContent)
FetchContent_Declare(ufsecp
    GIT_REPOSITORY https://github.com/shrec/UltrafastSecp256k1.git
    GIT_TAG        v3.14.0)
FetchContent_MakeAvailable(ufsecp)

add_executable(myapp main.c)
target_link_libraries(myapp PRIVATE ufsecp_static)
```

---

## 2. Building from Source

### Prerequisites

| Tool | Minimum Version |
|------|----------------|
| CMake | 3.20 |
| C++ compiler | C++20 (Clang 15+, GCC 13+, MSVC 2022) |
| Ninja (optional) | 1.10 |

### Build Commands

```bash
# Clone
git clone https://github.com/shrec/UltrafastSecp256k1.git
cd UltrafastSecp256k1

# Configure
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j

# Test
ctest --test-dir build --output-on-failure
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `SECP256K1_BUILD_CPU` | ON | Build CPU library |
| `SECP256K1_BUILD_CUDA` | OFF | Build CUDA GPU library |
| `SECP256K1_BUILD_CABI` | ON | Build C ABI shim (`ufsecp`) |
| `SECP256K1_BUILD_TESTS` | ON | Build unit tests |
| `SECP256K1_BUILD_BENCH` | OFF | Build benchmarks |
| `SECP256K1_BUILD_EXAMPLES` | OFF | Build example programs |
| `SECP256K1_SPEED_FIRST` | OFF | Aggressive speed optimizations |
| `SECP256K1_USE_LTO` | OFF | Link-time optimization |

---

## 3. C API Usage

The C API is the **recommended** interface for all applications and language bindings.

### Design Principles

1. **Opaque context** — all state in `ufsecp_ctx*`
2. **Every function returns `ufsecp_error_t`** (0 = success)
3. **No leaking internal types** — all I/O is `uint8_t[]` with fixed sizes
4. **Caller owns all buffers** — library never allocates on behalf of caller

### Context Lifecycle

```c
ufsecp_ctx* ctx = NULL;
ufsecp_error_t err = ufsecp_ctx_create(&ctx);
if (err != UFSECP_OK) { /* handle error */ }

// ... use ctx for all operations ...

ufsecp_ctx_destroy(ctx);  // NULL-safe
```

### Linking

| Library | File | Use Case |
|---------|------|----------|
| Static | `ufsecp_s.lib` / `libufsecp_s.a` | Embed in your binary |
| Shared | `ufsecp.dll` / `libufsecp.so` | Dynamic linking |

**Static linking on Windows**: define `UFSECP_API=` (empty) to suppress `__declspec(dllimport)`:

```cmake
target_compile_definitions(myapp PRIVATE "UFSECP_API=")
target_link_libraries(myapp PRIVATE ufsecp_static)
```

---

## 4. C++ API Usage

The C++ API provides higher-level types (`Scalar`, `Point`, etc.) with operator overloads.

```cpp
#include <secp256k1/scalar.hpp>
#include <secp256k1/point.hpp>
#include <secp256k1/ecdsa.hpp>

using namespace secp256k1;

// Key generation
fast::Scalar privkey = fast::Scalar::random();
fast::Point pubkey = fast::Point::generator().scalar_mul(privkey);

// ECDSA sign + verify
std::array<uint8_t, 32> msg = {/* hash */};
auto sig = ecdsa_sign(msg, privkey);
bool ok = ecdsa_verify(msg, sig, pubkey);
```

---

## 5. Key Operations

### Private Key Validation

```c
uint8_t privkey[32] = { /* your key bytes */ };
ufsecp_error_t err = ufsecp_seckey_verify(ctx, privkey);
// Returns UFSECP_OK if 0 < key < secp256k1 order
```

### Public Key Derivation

```c
uint8_t pubkey33[33];   // Compressed (02/03 prefix)
ufsecp_pubkey_create(ctx, privkey, pubkey33);

uint8_t pubkey65[65];   // Uncompressed (04 prefix)
ufsecp_pubkey_create_uncompressed(ctx, privkey, pubkey65);

uint8_t xonly[32];      // X-only (BIP-340)
ufsecp_pubkey_xonly(ctx, privkey, xonly);
```

### Key Tweaking

```c
// Add tweak: privkey = (privkey + tweak) mod n
ufsecp_seckey_tweak_add(ctx, privkey, tweak32);

// Multiply tweak: privkey = (privkey × tweak) mod n
ufsecp_seckey_tweak_mul(ctx, privkey, tweak32);

// Negate: privkey = -privkey mod n
ufsecp_seckey_negate(ctx, privkey);
```

---

## 6. ECDSA Signatures

### Sign

```c
uint8_t msg32[32] = { /* SHA-256 of your message */ };
uint8_t sig64[64];  // Compact R||S

ufsecp_error_t err = ufsecp_ecdsa_sign(ctx, msg32, privkey, sig64);
```

### Verify

```c
ufsecp_error_t err = ufsecp_ecdsa_verify(ctx, msg32, sig64, pubkey33);
if (err == UFSECP_OK) { /* signature valid */ }
```

### DER Encoding

```c
uint8_t der[72];
size_t der_len = sizeof(der);
ufsecp_ecdsa_sig_to_der(ctx, sig64, der, &der_len);

// Decode back
uint8_t sig64_back[64];
ufsecp_ecdsa_sig_from_der(ctx, der, der_len, sig64_back);
```

### Recoverable Signatures

```c
uint8_t sig64[64];
int recid;
ufsecp_ecdsa_sign_recoverable(ctx, msg32, privkey, sig64, &recid);

// Recover public key
uint8_t recovered_pub[33];
ufsecp_ecdsa_recover(ctx, msg32, sig64, recid, recovered_pub);
```

---

## 7. Schnorr / BIP-340 Signatures

### Sign

```c
uint8_t msg32[32], sig64[64];
uint8_t aux_rand[32] = {0};  // All-zeros for deterministic signing

ufsecp_schnorr_sign(ctx, msg32, privkey, aux_rand, sig64);
```

### Verify

```c
uint8_t xonly_pubkey[32];
ufsecp_pubkey_xonly(ctx, privkey, xonly_pubkey);

ufsecp_error_t err = ufsecp_schnorr_verify(ctx, msg32, sig64, xonly_pubkey);
```

---

## 8. ECDH Key Agreement

```c
// Standard: SHA256(compressed shared point)
uint8_t secret[32];
ufsecp_ecdh(ctx, my_privkey, their_pubkey33, secret);

// X-only: SHA256(x-coordinate only)
ufsecp_ecdh_xonly(ctx, my_privkey, their_pubkey33, secret);

// Raw: x-coordinate without hashing (advanced)
ufsecp_ecdh_raw(ctx, my_privkey, their_pubkey33, secret);
```

---

## 9. Address Generation

### P2PKH (Legacy, Base58Check)

```c
char addr[64];
size_t addr_len = sizeof(addr);
ufsecp_addr_p2pkh(ctx, pubkey33, UFSECP_NET_MAINNET, addr, &addr_len);
// Result: "1BvBMSEYstWetqT..."
```

### P2WPKH (SegWit, Bech32)

```c
char addr[128];
size_t addr_len = sizeof(addr);
ufsecp_addr_p2wpkh(ctx, pubkey33, UFSECP_NET_MAINNET, addr, &addr_len);
// Result: "bc1q..."
```

### P2TR (Taproot, Bech32m)

```c
char addr[128];
size_t addr_len = sizeof(addr);
ufsecp_addr_p2tr(ctx, xonly_pubkey, UFSECP_NET_MAINNET, addr, &addr_len);
// Result: "bc1p..."
```

### Network Constants

| Constant | Value | Prefixes |
|----------|-------|----------|
| `UFSECP_NET_MAINNET` | 0 | `1`, `bc1q`, `bc1p` |
| `UFSECP_NET_TESTNET` | 1 | `m`/`n`, `tb1q`, `tb1p` |

---

## 10. HD Key Derivation (BIP-32)

### Master Key from Seed

```c
uint8_t seed[64] = { /* BIP-39 derived seed */ };
ufsecp_bip32_key master;
ufsecp_bip32_master(ctx, seed, 64, &master);
```

### Path Derivation

```c
ufsecp_bip32_key child;
// Standard BIP-44 Bitcoin path
ufsecp_bip32_derive_path(ctx, &master, "m/44'/0'/0'/0/0", &child);
```

### Extract Keys

```c
uint8_t privkey[32];
ufsecp_bip32_privkey(ctx, &child, privkey);

uint8_t pubkey[33];
ufsecp_bip32_pubkey(ctx, &child, pubkey);
```

### Common Derivation Paths

| BIP | Path | Use |
|-----|------|-----|
| BIP-44 | `m/44'/0'/0'/0/i` | Legacy P2PKH |
| BIP-49 | `m/49'/0'/0'/0/i` | Wrapped SegWit |
| BIP-84 | `m/84'/0'/0'/0/i` | Native SegWit (P2WPKH) |
| BIP-86 | `m/86'/0'/0'/0/i` | Taproot (P2TR) |

> Replace `i` with the address index (0, 1, 2, ...).

---

## 11. MuSig2 Multi-Signatures

BIP-327 compatible multi-party Schnorr signatures (C++ API).

```cpp
#include <secp256k1/musig2.hpp>
using namespace secp256k1;

// 1. Key aggregation
auto [agg_key, key_coeffs] = musig2_key_agg({pubkey_a, pubkey_b});

// 2. Nonce generation (each signer)
auto [secnonce_a, pubnonce_a] = musig2_nonce_gen(seed_a, privkey_a, agg_key, msg);

// 3. Nonce aggregation
auto aggnonce = musig2_nonce_agg({pubnonce_a, pubnonce_b});

// 4. Start signing session
auto session = musig2_start_sign_session(aggnonce, agg_key, msg);

// 5. Partial sign (each signer)
auto psig_a = musig2_partial_sign(session, secnonce_a, privkey_a, key_coeffs[0]);

// 6. Verify + aggregate
bool ok = musig2_partial_verify(session, psig_a, pubnonce_a, key_coeffs[0], pubkey_a);
auto final_sig = musig2_partial_sig_agg(session, {psig_a, psig_b});

// Result: standard BIP-340 Schnorr signature
bool valid = schnorr_verify(msg, final_sig, agg_key);
```

---

## 12. FROST Threshold Signatures

t-of-n threshold Schnorr signatures with Feldman VSS DKG (C++ API).

```cpp
#include <secp256k1/frost.hpp>
using namespace secp256k1;

// 1. DKG: each participant generates commitments + shares
auto [commit_1, shares_1] = frost_keygen_begin(1, threshold, n, seed_1);
// ... exchange commitments and shares ...

// 2. Finalize: compute signing key + group public key
auto [key_pkg_1, ok] = frost_keygen_finalize(1, all_commitments,
                                              received_shares, t, n);

// 3. Signing: generate nonces
auto [nonce_1, nonce_commit_1] = frost_sign_nonce_gen(1, nonce_seed_1);

// 4. Partial sign
auto psig_1 = frost_sign(key_pkg_1, nonce_1, msg, all_nonce_commitments);

// 5. Verify partial sig (optional)
bool partial_ok = frost_verify_partial(psig_1, nonce_commit_1,
    key_pkg_1.verification_share, msg, all_nonce_commitments,
    key_pkg_1.group_public_key);

// 6. Aggregate: any t partial sigs → BIP-340 signature
auto final_sig = frost_aggregate(partial_sigs, nonce_commitments,
                                  key_pkg_1.group_public_key, msg);

// Result: standard BIP-340 Schnorr signature
bool valid = schnorr_verify(msg, final_sig, key_pkg_1.group_public_key);
```

---

## 13. Taproot

### Output Key (BIP-341)

```c
uint8_t output_x[32];
int parity;

// Key-path only (no scripts)
ufsecp_taproot_output_key(ctx, internal_x, NULL, output_x, &parity);

// With script tree
uint8_t merkle_root[32] = { /* Merkle root of tapscript tree */ };
ufsecp_taproot_output_key(ctx, internal_x, merkle_root, output_x, &parity);
```

### Key-Path Spending

```c
uint8_t tweaked_privkey[32];
ufsecp_taproot_tweak_privkey(ctx, privkey, NULL, tweaked_privkey);
// Sign with tweaked_privkey using ufsecp_schnorr_sign()
```

---

## 14. Thread Safety

**Rule: one context per thread, or external synchronization.**

```c
// Option A: one context per thread (recommended)
void worker_thread(void) {
    ufsecp_ctx* ctx;
    ufsecp_ctx_create(&ctx);
    // ... all operations use this ctx ...
    ufsecp_ctx_destroy(ctx);
}

// Option B: shared context with mutex (not recommended for performance)
pthread_mutex_lock(&ctx_mutex);
ufsecp_ecdsa_sign(shared_ctx, msg, key, sig);
pthread_mutex_unlock(&ctx_mutex);
```

**Thread-safe operations** (no context needed):
- `ufsecp_error_str()` — pure function, always safe
- `ufsecp_sha256()` / `ufsecp_hash160()` — stateless hash functions
- `ufsecp_abi_version()` — returns a constant

---

## 15. Error Handling

### Error Codes

| Code | Constant | Meaning |
|------|----------|---------|
| 0 | `UFSECP_OK` | Success |
| 1 | `UFSECP_ERR_NULL_ARG` | Required pointer was NULL |
| 2 | `UFSECP_ERR_BAD_KEY` | Invalid private key |
| 3 | `UFSECP_ERR_BAD_PUBKEY` | Invalid public key |
| 4 | `UFSECP_ERR_BAD_SIG` | Malformed signature |
| 5 | `UFSECP_ERR_BAD_INPUT` | Wrong length or format |
| 6 | `UFSECP_ERR_VERIFY_FAIL` | Signature verification failed |
| 7 | `UFSECP_ERR_ARITH` | Arithmetic overflow |
| 8 | `UFSECP_ERR_SELFTEST` | Library self-test failed |
| 9 | `UFSECP_ERR_INTERNAL` | Unexpected internal error |
| 10 | `UFSECP_ERR_BUF_TOO_SMALL` | Output buffer too small |

### Diagnostic

```c
ufsecp_error_t err = ufsecp_ecdsa_sign(ctx, msg, key, sig);
if (err != UFSECP_OK) {
    printf("Error %d: %s\n", err, ufsecp_error_str(err));

    // Per-context detail
    ufsecp_error_t last = ufsecp_last_error(ctx);
    const char* detail = ufsecp_last_error_msg(ctx);
    printf("Detail: %s\n", detail);
}
```

---

## 16. Platform-Specific Notes

### Windows (MSVC / Clang-cl)

- Use `ufsecp_s.lib` (static) or `ufsecp.dll` + `ufsecp.lib` (import lib)
- Static: define `UFSECP_API=` to suppress dllimport
- Stack size: some tests need `/STACK:8388608`

### Linux

- Link: `-lufsecp_s` (static) or `-lufsecp` (shared)
- RPATH is set by CMake for shared builds

### macOS

- Universal builds (arm64 + x86_64) supported via `CMAKE_OSX_ARCHITECTURES`
- Framework packaging available for Xcode integration

### WASM (Emscripten)

- Use the Emscripten toolchain file: `-DCMAKE_TOOLCHAIN_FILE=<emsdk>/upstream/emscripten/cmake/Modules/Platform/Emscripten.cmake`
- See `wasm/` for pre-built bindings

### Embedded (ESP32, STM32)

- Use CMake cross-compilation toolchain files in `cmake/`
- Requires C++20-capable cross compiler
- See `PORTING.md` for hardware-specific notes

---

## 17. Troubleshooting

### "Undefined symbol: ufsecp_*"

**Static linking on Windows**: Add `target_compile_definitions(myapp PRIVATE "UFSECP_API=")`.

**Static linking on Linux**: Ensure you link `ufsecp_static` (not `ufsecp`), and the library is listed **after** your object files.

### "Self-test failed" (UFSECP_ERR_SELFTEST)

The library runs a self-test on first `ufsecp_ctx_create()`. This should never fail unless:
- Memory corruption
- Incompatible binary (wrong architecture)
- Compiler bug (report to us)

### "Buffer too small" (UFSECP_ERR_BUF_TOO_SMALL)

Address generation functions need sufficient output buffers:
- P2PKH: at least 35 bytes
- P2WPKH: at least 63 bytes (Bech32 "bc1q...")
- P2TR: at least 63 bytes (Bech32m "bc1p...")
- WIF: at least 52 bytes + null terminator

### Build fails with "no member named '__int128'"

Your compiler doesn't support `__int128`. The library will fall back to a portable implementation automatically, but if it doesn't, try:

```bash
cmake -DSECP256K1_USE_INT128=OFF ...
```

### CUDA errors

- Ensure `CMAKE_CUDA_ARCHITECTURES` matches your GPU (e.g., `86` for RTX 3000, `89` for RTX 4000)
- Check that the CUDA toolkit version matches your driver
- See GPU-specific docs in `cuda/README.md`
