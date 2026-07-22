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
option(SECP256K1_BACKEND "secp256k1 backend: bundled (default) or ultrafast" bundled)
# Or equivalently: -DSECP256K1_BACKEND=ultrafast
```

No other changes are required. All `secp256k1_*` call sites in Bitcoin Core continue
to work without modification.

### Evidence summary

| Claim | Verification command | Evidence |
|-------|----------------------|----------|
| 749/749 `make check` tests pass | `python3 ci/check_bitcoin_core_test_results.py` | `docs/BITCOIN_CORE_BENCH_RESULTS.json` |
| All signing paths constant-time | `python3 ci/audit_gate.py --ct-integrity` | `docs/CT_VERIFICATION.md` |
| Differential parity with libsecp256k1 | CTest `differential_*` targets | `docs/BITCOIN_CORE_BACKEND_EVIDENCE.md §2` |
| 275 exploit PoCs tests, 0 failures | `python3 ci/check_exploit_wiring.py` | `audit/unified_audit_runner.cpp` |
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
| ConnectBlockAllSchnorr | 255.3 ms/blk | 253.0 ms/blk | **+0.9%** ±0.5% err |
| ConnectBlockAllEcdsa | 257.4 ms/blk | 254.3 ms/blk | **+1.2%** ±0.2% err |
| ConnectBlockMixed | 257.7 ms/blk | 253.9 ms/blk | **+1.5%** ±0.3% err |
| P2WPKH verify | 45,777 ns | 45,978 ns | ≈parity (0.4% slower, within noise margin) |

Full data in `docs/BITCOIN_CORE_BENCH_RESULTS.json` (benchmark run 2026-05-12, hard turbo lock, GCC 14.2.0; P1 security fixes applied through 2026-05-23). All CT signing paths use `generator_mul_blinded` for nonce multiplication (DPA defense active when `secp256k1_context_randomize` is called). CT verification is via CAAS — the project's automated multi-layer audit framework (LLVM ct-verif, Valgrind taint, dudect, 418-module unified runner).

**Honest disclosure:** a 2026-05-07 native-C++-API run (GCC 13.3, 2000 unique pubkeys) showed
ConnectBlockAllSchnorr at 0.83× (−17%) vs libsecp256k1 due to per-pubkey GLV table rebuild
cost. The 2026-05-12 shim-path measurement above shows +0.9–1.5% faster end-to-end; the
−17% is kept as a conservative upper bound until a direct unique-pubkey re-run on the shim
path with GCC 14 + LTO lands. Full disclosure in `docs/BITCOIN_CORE_PR_DESCRIPTION.md`.

### Known gaps and honest statements

- **Without LTO (development builds): ConnectBlock is ~0.5–1.0% slower than libsecp256k1.** The positive results above require `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON`. Development builds (`RelWithDebInfo`) will show a small regression due to larger code footprint (measured 2,310 KB Ultra `.text` vs libsecp256k1 1,261 KB = 1.83×, bitcoin-core profile, no-LTO, 2026-05-22, causing i-cache pressure; see `docs/SHIM_FOOTPRINT_COMPARISON.md`). Release builds with LTO recover and surpass libsecp.
- macOS ARM64 CI covers shim build + test only; full GPU suite remains Linux x86-64
- Formal verification is not claimed; software-tool CT verification only (LLVM ct-verif, Valgrind, dudect)
- Thread safety: each context is independent; concurrent use of distinct contexts is safe
- ConnectBlock benchmark uses governor=performance, taskset -c 0, hard turbo lock (intel_pstate/no_turbo=1, sudo pinned, 2026-05-12).

### How to verify independently

```bash
# Clone and check out the evidence commit
git clone https://github.com/shrec/UltrafastSecp256k1
cd UltrafastSecp256k1
git checkout f6b920354495e3b369862ef90f21006b69f9e31a

# Reproduce the evidence bundle
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON
ninja -C build
python3 ci/caas_runner.py --profile bitcoin-core-backend --json

# Verify bundle integrity
python3 ci/verify_external_audit_bundle.py
```

Full evidence document: [`docs/BITCOIN_CORE_BACKEND_EVIDENCE.md`](BITCOIN_CORE_BACKEND_EVIDENCE.md)
