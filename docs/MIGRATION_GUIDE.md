# Migration Guide: libsecp256k1 → UltrafastSecp256k1

This guide covers migrating existing code from bitcoin-core/libsecp256k1 to
UltrafastSecp256k1. A libsecp256k1 compatibility shim is also available for
drop-in replacement without code changes.

---

## Option A: Drop-in Compatibility Shim (zero code changes)

UltrafastSecp256k1 ships a compatibility shim under `compat/libsecp256k1_shim/`
that re-exports the `secp256k1_*` API surface from libsecp256k1.

```cmake
# CMakeLists.txt
find_package(UltrafastSecp256k1 REQUIRED)
target_link_libraries(myapp PRIVATE UltrafastSecp256k1::secp256k1_shim)
```

Your existing `#include <secp256k1.h>` and `secp256k1_*` calls continue to
work without modification. The shim routes all calls through the constant-time
UltrafastSecp256k1 backends.

**Known shim gaps:** `secp256k1_scratch_space_*` (internal allocation API),
`secp256k1_context_set_error_callback`, `secp256k1_context_set_illegal_callback`.
If your code uses these, use Option B.

---

## Option B: Native `ufsecp_*` API

The native API offers more functionality (Schnorr, MuSig2, BIP-352, ECIES,
Taproot, batch verification, CT-guaranteed signing) and is the recommended
path for new code.

### Context lifecycle

```c
// libsecp256k1
secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
secp256k1_context_destroy(ctx);

// UltrafastSecp256k1
ufsecp_ctx* ctx = NULL;
if (ufsecp_ctx_create(&ctx) != UFSECP_OK) { /* handle error */ }
ufsecp_ctx_destroy(ctx);
```

### Key generation

```c
// libsecp256k1
secp256k1_pubkey pubkey;
secp256k1_ec_pubkey_create(ctx, &pubkey, privkey);
uint8_t buf[33]; size_t len = 33;
secp256k1_ec_pubkey_serialize(ctx, buf, &len, &pubkey, SECP256K1_EC_COMPRESSED);

// UltrafastSecp256k1 (one call, always compressed)
uint8_t pubkey33[33];
ufsecp_pubkey_create(ctx, privkey, pubkey33);
```

### ECDSA sign

```c
// libsecp256k1
secp256k1_ecdsa_signature sig;
secp256k1_ecdsa_sign(ctx, &sig, msg32, privkey, NULL, NULL);
uint8_t buf[64];
secp256k1_ecdsa_signature_serialize_compact(ctx, buf, &sig);

// UltrafastSecp256k1 (always compact, always constant-time)
uint8_t sig64[64];
ufsecp_ecdsa_sign(ctx, msg32, privkey, sig64);
```

### ECDSA verify

```c
// libsecp256k1
secp256k1_ecdsa_signature sig;
secp256k1_ecdsa_signature_parse_compact(ctx, &sig, sig64);
secp256k1_pubkey pubkey;
secp256k1_ec_pubkey_parse(ctx, &pubkey, pubkey33, 33);
int ok = secp256k1_ecdsa_verify(ctx, &sig, msg32, &pubkey);

// UltrafastSecp256k1
int ok = (ufsecp_ecdsa_verify(ctx, msg32, sig64, pubkey33) == UFSECP_OK);
```

### Schnorr (BIP-340)

```c
// libsecp256k1 (with secp256k1_schnorrsig module)
secp256k1_keypair kp;
secp256k1_keypair_create(ctx, &kp, privkey);
secp256k1_schnorrsig_sign32(ctx, sig64, msg32, &kp, aux_rand32);

// UltrafastSecp256k1
ufsecp_schnorr_sign(ctx, msg32, privkey, aux_rand32, sig64);
```

### Schnorr verify

```c
// libsecp256k1
secp256k1_xonly_pubkey xpub;
secp256k1_xonly_pubkey_parse(ctx, &xpub, xonly32);
int ok = secp256k1_schnorrsig_verify(ctx, sig64, msg32, 32, &xpub);

// UltrafastSecp256k1
int ok = (ufsecp_schnorr_verify(ctx, msg32, sig64, xonly32) == UFSECP_OK);
```

### Batch verify (ECDSA)

```c
// UltrafastSecp256k1 packed format: [msg32 | pubkey33 | sig64] = 129 bytes/entry
std::vector<uint8_t> packed(n * 129);
for (size_t i = 0; i < n; ++i) {
    memcpy(packed.data() + i*129,      msgs[i],    32);
    memcpy(packed.data() + i*129 + 32, pubkeys[i], 33);
    memcpy(packed.data() + i*129 + 65, sigs[i],    64);
}
int ok = (ufsecp_ecdsa_batch_verify(ctx, packed.data(), n) == UFSECP_OK);
```

### Schnorr batch verify

```c
// UltrafastSecp256k1 packed format: [xonly32 | msg32 | sig64] = 128 bytes/entry
std::vector<uint8_t> packed(n * 128);
for (size_t i = 0; i < n; ++i) {
    memcpy(packed.data() + i*128,      xonly_keys[i], 32);
    memcpy(packed.data() + i*128 + 32, msgs[i],       32);
    memcpy(packed.data() + i*128 + 64, sigs[i],       64);
}
int ok = (ufsecp_schnorr_batch_verify(ctx, packed.data(), n) == UFSECP_OK);
```

### Error handling

```c
// libsecp256k1: functions return int (1=success, 0=failure)
// UltrafastSecp256k1: functions return ufsecp_error_t (UFSECP_OK=success)

// Mapping:
//   secp256k1_* return 1  →  ufsecp_* return UFSECP_OK
//   secp256k1_* return 0  →  ufsecp_* return non-UFSECP_OK error code
```

---

## Feature comparison

| Feature | libsecp256k1 | UltrafastSecp256k1 |
|---------|-------------|-------------------|
| ECDSA sign/verify | ✅ | ✅ |
| Schnorr BIP-340 | ✅ (module) | ✅ |
| ECDSA batch verify | ✅ (module) | ✅ |
| Schnorr batch verify | ✅ (module) | ✅ |
| MuSig2 BIP-327 | ✅ (experimental) | ✅ |
| Taproot BIP-341 | ✅ (module) | ✅ |
| BIP-352 Silent Payments | ❌ | ✅ |
| ECIES | ❌ | ✅ |
| Ethereum signing | ❌ | ✅ |
| Bitcoin wallet (multi-coin) | ❌ | ✅ |
| GPU acceleration | ❌ | ✅ (CUDA/OpenCL) |
| Constant-time guarantee | ✅ | ✅ (audited) |
| C ABI | ✅ | ✅ |
| C++ API | ❌ | ✅ |

---

## Security notes

- All signing paths in UltrafastSecp256k1 default to `ct::` (constant-time)
  primitives. Unlike libsecp256k1, there is no variable-time fast path reachable
  from the public API without an explicit opt-in.
- Private keys passed to `ufsecp_*` functions are erased from stack memory
  immediately after use.
- See `docs/SECURITY_ARCHITECTURE.md` and `docs/ATTACK_GUIDE.md` for the
  full threat model.

---

## Getting help

- GitHub Issues: https://github.com/shrec/UltrafastSecp256k1/issues
- Security reports: see `SECURITY.md`
- API reference: `docs/API_REFERENCE.md`
