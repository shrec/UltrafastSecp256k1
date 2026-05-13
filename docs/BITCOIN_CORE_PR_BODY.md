# Bitcoin Core PR — Draft Body

> Internal use: paste this as the PR description when submitting to bitcoin/secp256k1 or bitcoin/bitcoin.
> Keep to 1-2 screens. Reviewers should be able to verify every claim in ≤30 min.

---

## cmake: add optional secp256k1 backend evaluation path

This PR introduces **UltrafastSecp256k1 as an optional secondary backend**, selectable at
compile time. **This is not a replacement for libsecp256k1**, which remains the default and
the ecosystem reference implementation. No existing code paths change. Bitcoin Core behavior
is identical when using the default backend.

The motivation is a compile-time opt-in for evaluation on hardware where the measurably
faster CT signing paths (+10–35% on Taproot/ECDSA sign workloads with Release+LTO,
controlled benchmarks with raw evidence below; ConnectBlock aggregate +0.9–1.5% faster
with LTO but ~0.5–1.0% slower without LTO) may benefit validation performance — while
maintaining full libsecp256k1 C ABI compatibility.

### What this adds

A thin shim layer (`compat/libsecp256k1_shim/`) that maps the existing `secp256k1.h`
C API surface to UltrafastSecp256k1 internals. To enable it:

```cmake
# In CMakeLists.txt (or as -D flag) — canonical flag name:
option(SECP256K1_USE_ULTRAFAST "Use UltrafastSecp256k1 instead of bundled secp256k1" OFF)
# Or equivalently: -DSECP256K1_USE_ULTRAFAST=ON
```

No other changes are required. All `secp256k1_*` call sites in Bitcoin Core continue
to work without modification.

### Evidence summary

| Claim | Verification command | Evidence |
|-------|----------------------|----------|
| 749/749 `make check` tests pass | `python3 ci/check_bitcoin_core_test_results.py` | `docs/BITCOIN_CORE_BENCH_RESULTS.json` |
| All signing paths constant-time | `python3 ci/audit_gate.py --ct-integrity` | `docs/CT_VERIFICATION.md` |
| Differential parity with libsecp256k1 | CTest `differential_*` targets | `docs/BITCOIN_CORE_BACKEND_EVIDENCE.md §2` |
| 261 exploit PoCs tests, 0 failures | `python3 ci/check_exploit_wiring.py` | `audit/unified_audit_runner.cpp` |
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
| Schnorr sign (Taproot/Merkle) | 113,410 ns | 83,930 ns | **1.35×** |
| ECDSA sign | 165,140 ns | 149,520 ns | **1.10×** |
| P2TR ScriptPath verify | 83,812 ns | 76,451 ns | **1.10×** |
| ConnectBlockAllSchnorr | 255.3 ms/blk | 253.0 ms/blk | **+0.9%** |
| ConnectBlockAllEcdsa | 257.4 ms/blk | 254.3 ms/blk | **+1.2%** |
| ConnectBlockMixed | 257.7 ms/blk | 253.9 ms/blk | **+1.5%** |
| P2WPKH verify | 45,777 ns | 45,978 ns | ≈parity (0.4% slower, within noise margin) |

Full data with err% in `docs/BITCOIN_CORE_BENCH_RESULTS.json` (commit `48e7c02f`, 2026-05-12,
hard turbo lock, GCC 14.2.0). All CT signing paths use `generator_mul_blinded` for nonce
multiplication (DPA defense active when `secp256k1_context_randomize` is called).
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
