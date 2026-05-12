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
| 749/749 `make check` tests pass | `python3 ci/check_bitcoin_core_test_results.py` | `docs/BITCOIN_CORE_BENCH_RESULTS.json` |
| All signing paths constant-time | `python3 ci/audit_gate.py --ct-integrity` | `docs/CT_VERIFICATION.md` |
| Differential parity with libsecp256k1 | CTest `differential_*` targets | `docs/BITCOIN_CORE_BACKEND_EVIDENCE.md §2` |
| 258 exploit PoCs tests, 0 failures | `python3 ci/check_exploit_wiring.py` | `audit/unified_audit_runner.cpp` |
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

### Performance (bench_bitcoin, Release+LTO, GCC 14.2.0, i5-14400F, hard turbo lock (intel_pstate/no_turbo=1), taskset -c 0, nice -20, 2026-05-12)

| Operation | libsecp256k1 | Ultra | vs libsecp |
|-----------|-------------|-------|-----------|
| Schnorr sign (Taproot) | 114,479 ns | 84,273 ns | **1.36×** |
| ECDSA sign | 168,907 ns | 147,262 ns | **1.15×** |
| P2TR ScriptPath verify | 83,481 ns | 75,549 ns | **1.11×** |
| ConnectBlockAllSchnorr | 255.3 ms/blk | 253.0 ms/blk | **+0.9%** |
| ConnectBlockAllEcdsa | 257.4 ms/blk | 254.3 ms/blk | **+1.2%** |
| ConnectBlockMixed | 257.7 ms/blk | 253.9 ms/blk | **+1.5%** |
| P2WPKH verify | 46,062 ns | 45,217 ns | parity |

Full data with err% in `docs/BITCOIN_CORE_BENCH_RESULTS.json`. Note: all CT signing paths are
unchanged from this session; these signing numbers do not include non-CT paths.
No external third-party audit has been conducted — all CT verification is self-generated CI tooling.

### Known gaps and honest statements

- macOS ARM64 CI covers shim build + test only; full GPU suite remains Linux x86-64
- Formal verification is not claimed; software-tool CT verification only (LLVM ct-verif, Valgrind, dudect)
- No external third-party security audit has been conducted
- Thread safety: each context is independent; concurrent use of distinct contexts is safe
- Without LTO: ConnectBlock ~0.5–1.0% slower due to instruction-cache pressure from larger code footprint (~1.3 MB vs libsecp ~400 KB); gap closes with LTO
- ConnectBlock benchmark uses governor=performance, taskset -c 0, hard turbo lock (intel_pstate/no_turbo=1, sudo pinned, 2026-05-12).

### How to verify independently

```bash
# Clone and check out the evidence commit
git clone https://github.com/shrec/UltrafastSecp256k1
cd UltrafastSecp256k1
git checkout 48e7c02fff02d823d2396f7eb05e425dfb3689e4

# Reproduce the evidence bundle
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
ninja -C build
python3 ci/caas_runner.py --profile bitcoin-core-backend --json

# Verify bundle integrity
python3 ci/verify_external_audit_bundle.py
```

Full evidence document: [`docs/BITCOIN_CORE_BACKEND_EVIDENCE.md`](BITCOIN_CORE_BACKEND_EVIDENCE.md)
