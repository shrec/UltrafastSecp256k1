# FAQ & Common Pitfalls

**UltrafastSecp256k1** -- Frequently Asked Questions

---

## General

### Q: What is UltrafastSecp256k1?

A high-performance secp256k1 elliptic curve cryptography library supporting ECDSA (RFC 6979), Schnorr (BIP-340), MuSig2, FROST threshold signatures, BIP-32 HD derivation, and 27-coin address generation. Targets x86-64, ARM64, RISC-V, WASM, CUDA, OpenCL, Metal, and embedded platforms.

### Q: How does it compare to bitcoin-core/libsecp256k1?

UltrafastSecp256k1 focuses on breadth (multi-protocol, multi-platform, GPU backends) while libsecp256k1 focuses on depth (formally reviewed, battle-tested ECDSA/Schnorr for Bitcoin Core). Our differential test suite (`test_cross_libsecp256k1`) verifies matching results for 7,860+ operations.

### Q: Is it production-ready?

The library has not yet undergone an independent external security audit. Core arithmetic is verified with 641K+ audit checks, but cryptographic libraries should be audited before production use with real funds. See [AUDIT_SCOPE.md](AUDIT_SCOPE.md).

### Q: What license is it under?

MIT License. See `LICENSE` in the repository root.

---

## Building

### Q: What compilers are supported?

| Compiler | Minimum Version | Recommended |
|----------|----------------|-------------|
| GCC | 10+ | 13+ |
| Clang | 14+ | 18+ |
| MSVC | 19.29+ (VS 2019) | 19.38+ (VS 2022) |
| Apple Clang | 14+ | 15+ |
| Emscripten | 3.1+ | 3.1.50+ |

The CMake toolchain auto-detects Clang 19+ and prefers it when available.

### Q: How do I build from source?

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

See [BUILDING.md](BUILDING.md) for full details.

### Q: I get "ninja not found" on Windows

Install Ninja via `pip install ninja` or download from https://ninja-build.org/. Ensure it's on your PATH.

### Q: Build fails with CUDA errors

- Set `CMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability (e.g., `-DCMAKE_CUDA_ARCHITECTURES=86`)
- Do not add global `-flto` flags -- CUDA device-link breaks with host LTO
- CUDA >= 11.0 required

### Q: How do I build without GPU support?

GPU support is auto-detected. To explicitly disable:
```bash
cmake -S . -B build -DSECP256K1_BUILD_CUDA=OFF -DSECP256K1_BUILD_OPENCL=OFF
```

---

## API Usage

### Q: FAST vs CT -- which should I use?

| Operation | Namespace | When to Use |
|-----------|-----------|-------------|
| Public key generation (from known pubkey) | `fast::` | When input is public |
| Signature verification | `fast::` | Always public-data operation |
| Signing (ECDSA/Schnorr) | `ct::` | **Always** for signing with secret keys |
| Scalar multiplication with secret | `ct::` | **Always** when scalar is sensitive |
| Batch operations | `fast::` | When all inputs are public |

**Rule**: Use `ct::` whenever _any_ input is a secret key or secret nonce. Use `fast::` only when all inputs are public.

### Q: What happens if I use FAST for signing?

Variable-time code may leak information about the private key through timing side channels. On shared hardware (cloud VMs, multitenant systems), an attacker may recover the key after observing enough signing operations.

Compile with `SECP256K1_REQUIRE_CT=1` to get warnings when using FAST for signing.

### Q: How do I use the C API?

```c
#include <ufsecp/ufsecp.h>

ufsecp_context* ctx = ufsecp_context_create();

uint8_t privkey[32] = { /* your key */ };
uint8_t pubkey[33];
ufsecp_pubkey_create(ctx, privkey, pubkey);

uint8_t msg[32] = { /* message hash */ };
uint8_t sig[64];
ufsecp_ecdsa_sign(ctx, msg, privkey, sig);

int valid = ufsecp_ecdsa_verify(ctx, msg, sig, pubkey);

ufsecp_context_destroy(ctx);
```

See [USER_GUIDE.md](USER_GUIDE.md) for complete examples.

### Q: Is `ufsecp_context` thread-safe?

**No.** Each thread must create its own context. See [THREAD_SAFETY.md](THREAD_SAFETY.md).

---

## Common Pitfalls

### Pitfall 1: Sharing ufsecp_context across threads

```c
// [FAIL] WRONG -- context is not thread-safe
ufsecp_context* ctx = ufsecp_context_create();
// Thread A: ufsecp_ecdsa_sign(ctx, ...);
// Thread B: ufsecp_ecdsa_verify(ctx, ...);  // DATA RACE

// [OK] CORRECT -- one context per thread
void worker(void) {
    ufsecp_context* ctx = ufsecp_context_create();
    ufsecp_ecdsa_sign(ctx, ...);
    ufsecp_context_destroy(ctx);
}
```

### Pitfall 2: Using from_bytes for binary database I/O

```cpp
// [FAIL] WRONG -- from_bytes is big-endian (for hex/test vectors)
auto fe = FieldElement::from_bytes(db_record);

// [OK] CORRECT -- from_limbs is little-endian (native x86/64)
auto fe = FieldElement::from_limbs(reinterpret_cast<const uint64_t*>(db_record));
```

`from_limbs` is the **primary** function for internal I/O. `from_bytes` is only for standard crypto vectors and hex strings.

### Pitfall 3: Forgetting low-S normalization

ECDSA signatures must have `s <= n/2` (BIP-62 / BIP-66). UltrafastSecp256k1 enforces this automatically in `ecdsa_sign()`, but if you construct signatures manually, you must check and normalize.

### Pitfall 4: Using GPU for secret key operations

```cpp
// [FAIL] WRONG -- GPU is variable-time, leaks timing information
cuda_scalar_mul(secret_key, G);

// [OK] CORRECT -- use CT layer on CPU for secret operations
auto pubkey = ct::scalar_mul(secret_key, G);
```

GPU backends are for **public-data operations only** (verification, public key batch generation, search).

### Pitfall 5: Not zeroing secret keys after use

The library does not manage key lifetimes. After use, explicitly zero secret material:

```cpp
// [OK] Zero sensitive buffers
std::memset(privkey, 0, 32);
std::memset(&signing_share, 0, sizeof(signing_share));
```

Consider using `SecureZeroMemory` (Windows) or `explicit_bzero` (Linux) to prevent compiler optimization from removing the zeroing.

### Pitfall 6: Confusing x-only and compressed pubkeys

| Format | Size | Functions |
|--------|------|-----------|
| Compressed | 33 bytes (`02/03 ∥ x`) | `ufsecp_pubkey_create`, `ufsecp_ecdsa_verify` |
| X-only | 32 bytes (`x` only) | `ufsecp_schnorr_verify`, `ufsecp_pubkey_xonly` |

Schnorr (BIP-340) uses **x-only** (32 bytes). ECDSA uses **compressed** (33 bytes). Don't mix them.

### Pitfall 7: FROST nonce reuse

```cpp
// [FAIL] WRONG -- reusing nonce for different messages
auto [nonce, commit] = frost_sign_nonce_gen(my_id, seed);
auto sig1 = frost_sign(key_pkg, nonce, msg1, commits);
auto sig2 = frost_sign(key_pkg, nonce, msg2, commits);  // KEY LEAK!

// [OK] CORRECT -- fresh nonce for each signing session
auto [nonce1, commit1] = frost_sign_nonce_gen(my_id, seed1);
auto sig1 = frost_sign(key_pkg, nonce1, msg1, commits1);
auto [nonce2, commit2] = frost_sign_nonce_gen(my_id, seed2);
auto sig2 = frost_sign(key_pkg, nonce2, msg2, commits2);
```

### Pitfall 8: Assuming BIP-32 paths are always valid

```cpp
// [FAIL] WRONG -- no error checking
auto keys = bip32_derive_path(master, user_input);

// [OK] CORRECT -- validate path first
auto parsed = bip32_parse_path(user_input);
if (!parsed.has_value()) {
    // Handle invalid path
    return error;
}
auto keys = bip32_derive_path(master, *parsed);
```

### Pitfall 9: Static linking without UFSECP_API define (Windows)

On Windows, static linking against `ufsecp_static` requires defining `UFSECP_API=` to prevent `__declspec(dllimport)`:

```cmake
target_compile_definitions(my_app PRIVATE UFSECP_API=)
target_link_libraries(my_app PRIVATE ufsecp_static fastsecp256k1)
```

### Pitfall 10: MuSig2 with attacker-controlled public keys

Always use the MuSig2 key aggregation function which includes key-prefixed hashing (KeyAgg coefficient). Never manually sum public keys -- this enables rogue-key attacks.

```cpp
// [FAIL] WRONG -- naive key aggregation
auto agg_pk = pk1 + pk2;  // Rogue-key attack!

// [OK] CORRECT -- use MuSig2 key aggregation
auto agg = musig2_key_agg({pk1, pk2});
```

---

## Performance

### Q: What's the throughput for ECDSA verification?

Platform-dependent. Typical on modern x86-64 (single-core):
- ECDSA verify: ~15,000-25,000 ops/sec
- Schnorr verify: ~20,000-30,000 ops/sec
- Key generation: ~30,000-50,000 ops/sec

See `docs/BENCHMARKS.md` for detailed numbers.

### Q: How fast is the GPU batch mode?

CUDA on RTX 3090: millions of point operations per second. Performance depends on batch size, memory bandwidth, and compute capability. Set `threads_per_batch` in `config.json` for optimal throughput.

### Q: How do I run benchmarks?

```bash
cmake -S . -B build -DSECP256K1_BUILD_BENCH=ON
cmake --build build -j --target bench_secp256k1
./build/bench_secp256k1
```

---

## Protocols

### Q: Is the MuSig2 implementation compatible with BIP-327?

The protocol structure matches BIP-327, but key format differs: we use **x-only 32-byte** pubkeys throughout, while BIP-327 specifies **33-byte compressed** for hash inputs. This means signatures are not bitwise-identical to other BIP-327 implementations when producing the aggregate. Final BIP-340 verification remains compatible.

### Q: Can I use FROST for Bitcoin multisig?

FROST produces standard BIP-340 Schnorr signatures, so the final signature is indistinguishable from a single-signer signature. However, FROST is experimental and should not be used for production Bitcoin transactions without additional review.

### Q: What threshold configurations does FROST support?

Any `t-of-n` where `2 <= t <= n`. Tested with:
- 2-of-3
- 3-of-5
- Arbitrary `t` and `n` via API

---

## Troubleshooting

### Q: Tests fail with "stack overflow"

Some tests use deep recursion or large stack allocations. On Windows, link with `/STACK:8388608` (8MB). CMake does this automatically for test targets.

### Q: "Unresolved external symbol" when linking ufsecp

Ensure you link both libraries and define `UFSECP_API=`:
```cmake
target_compile_definitions(my_app PRIVATE UFSECP_API=)
target_link_libraries(my_app PRIVATE ufsecp_static fastsecp256k1)
```

### Q: dudect test fails intermittently

dudect is statistical. A single borderline pass/fail is normal. The CI uses conservative thresholds (t=25 for smoke, t=4.5 for nightly). If it fails consistently, there may be a real timing leak -- investigate with the full nightly run.

### Q: How do I report a bug?

- Non-security bugs: Open a GitHub issue
- Security vulnerabilities: See [BUG_BOUNTY.md](BUG_BOUNTY.md) and [SECURITY.md](../SECURITY.md)

---

*Last updated: 2026-02-24*
