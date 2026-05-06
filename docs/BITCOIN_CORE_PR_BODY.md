# Bitcoin Core PR — Draft Body

> Internal use: paste this as the PR description when submitting to bitcoin/secp256k1 or bitcoin/bitcoin.
> Keep to 1-2 screens. Reviewers should be able to verify every claim in ≤30 min.

---

## cmake: add optional secp256k1 backend evaluation path

This PR introduces an **opt-in, compile-time alternative backend** for libsecp256k1.
No existing code paths change. Bitcoin Core behavior is identical when using the default backend.

### What this adds

A thin shim layer (`compat/libsecp256k1_shim/`) that maps the existing `secp256k1.h`
C API surface to UltrafastSecp256k1 internals. To enable it:

```cmake
# In CMakeLists.txt (or as -D flag):
set(SECP256K1_BACKEND ultrafastsecp256k1)
```

No other changes are required. All `secp256k1_*` call sites in Bitcoin Core continue
to work without modification.

### Evidence summary

| Claim | Verification command | Evidence |
|-------|----------------------|----------|
| 693/693 `make check` tests pass | `python3 ci/check_bitcoin_core_test_results.py` | `docs/BITCOIN_CORE_TEST_RESULTS.json` |
| All signing paths constant-time | `python3 ci/audit_gate.py --ct-integrity` | `docs/CT_SIGNING_PATHS.md` |
| Differential parity with libsecp256k1 | CTest `differential_*` targets | `docs/BITCOIN_CORE_BACKEND_EVIDENCE.md §2` |
| 253 exploit PoCs tests, 0 failures | `python3 ci/check_exploit_wiring.py` | `audit/unified_audit_runner.cpp` |
| Reproducible evidence bundle | `python3 ci/verify_external_audit_bundle.py` | `docs/EXTERNAL_AUDIT_BUNDLE.json` |

### Constant-time guarantee

All signing paths (ECDSA, Schnorr, recovery) route through `secp256k1::ct::*`
primitives — branchless scalar inversion, constant-time scalar multiplication,
and blinding. Verified by three independent tools:
- **ct-verif** (LLVM-based): [CT_INDEPENDENCE.md](CT_INDEPENDENCE.md)
- **Valgrind CT mode**: run `ctest -R ct_verif`
- **dudect** (timing-based): run `ctest -R cycle_ct`

### Functional parity

Differential testing against bitcoin-core/secp256k1 reference:
- ECDSA sign/verify: 500+ Wycheproof vectors + 50K random round-trips
- Schnorr: all BIP-340 test vectors + batch verify (N=1/64/128/192)
- ECDH: 200+ Wycheproof vectors
- Recoverable signing: recid ∈ [0,3], round-trip recovery
- DER parse/serialize: 1M+ random malformed inputs

### Build requirements

- C++20 compiler for the core library (GCC 12+, Clang 14+)
- The shim layer and all headers exposed to Bitcoin Core are C++17-compatible
- No new runtime dependencies; no GPU drivers required

### Known gaps and honest statements

- macOS ARM64 CI covers shim build + test only; full GPU suite remains Linux x86-64
- Formal verification is not claimed; software-tool CT verification only
- Thread safety: each context is independent; concurrent use of distinct contexts is safe

### How to verify independently

```bash
# Clone and check out the evidence commit
git clone https://github.com/shrec/UltrafastSecp256k1
cd UltrafastSecp256k1
git checkout <commit>

# Reproduce the evidence bundle
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
python3 ci/caas_runner.py --profile bitcoin-core-backend --json

# Verify bundle integrity
python3 ci/verify_external_audit_bundle.py
```

Full evidence document: [`docs/BITCOIN_CORE_BACKEND_EVIDENCE.md`](BITCOIN_CORE_BACKEND_EVIDENCE.md)
