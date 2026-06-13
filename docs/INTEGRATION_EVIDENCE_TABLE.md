# Integration Evidence Table

**Purpose (Bastion B7).** A single, replayable index of the external-integration
evidence that was previously scattered across the shim-parity gate, the audit
regression suite, the Bitcoin Core result JSON, and the libbitcoin bridge tests.
Every row names the evidence file and the command an independent reviewer runs to
reproduce it. Correctness evidence is kept separate from performance findings
(the latter live in `docs/BITCOIN_CORE_BENCH_RESULTS.json` and
`docs/bench_unified_*.json`).

> **Rule:** every number here traces to a committed artifact or a runnable
> command. Pass counts that require a build (libbitcoin bridge) name the target
> to run rather than asserting an unverified figure.

## Correctness evidence

| Integration surface | Evidence (committed) | What it verifies | Reproduce |
|---------------------|----------------------|------------------|-----------|
| libsecp256k1 shim parity | `ci/check_libsecp_shim_parity.py` | All shim entry points use strict parse boundaries; no regression to permissive `from_bytes`. **17/17 checks pass.** | `python3 ci/check_libsecp_shim_parity.py --json` |
| Normal ECDSA verify | `audit/test_cross_libsecp256k1.cpp` (CI `cross-libsecp256k1`) | Bit-exact verdict vs `bitcoin-core/libsecp256k1` v0.6.0 (hard gate) | `ctest --test-dir build -R '^cross_libsecp256k1$'` |
| DER parse + normalize + verify | `audit/test_regression_ecdsa_verify_cache_consistency.cpp` | block 704,789 tuple: DER parse → normalize → repeat-verify after cache hits | `cmake --build build-audit -t test_regression_ecdsa_verify_cache_consistency_standalone && ctest -R ecdsa_verify_cache_consistency` |
| ECDSA cache identity (same-X / opposite-Y) | `audit/test_regression_ecdsa_verify_cache_consistency.cpp` | `02‖X` vs `03‖X` compressed keys must not share a cached `EcdsaPublicKey` (fix `684141e7`); opposite-parity cache-poisoning regression | same as above |
| Schnorr verify (BIP-340) | `ci/check_libsecp_shim_parity.py` + shim differential | x-only key + variable-length message handling matches upstream semantics | `python3 ci/check_libsecp_shim_parity.py` |
| Batch ECDSA / Schnorr verify | `compat/libbitcoin_bridge/tests/test_lbtc_bridge.cpp` | Batch verify path (CPU + optional GPU) vs shim reference | `cmake --build out -t test_lbtc_bridge && ctest -R lbtc_bridge` |
| libbitcoin consensus differential | `compat/libbitcoin_bridge/tests/test_lbtc_consensus_diff.cpp` | Script-sig batch verify verdicts match libbitcoin reference | `cmake --build out -t test_lbtc_consensus_diff && ctest -R lbtc_consensus_diff` |
| libbitcoin collect / commitment / multisig | `compat/libbitcoin_bridge/tests/test_lbtc_collect.cpp`, `test_lbtc_commitment.cpp`, `test_lbtc_multisig_threshold.cpp` | In-place collect verify, BIP-341 commitment, threshold paths | `cmake --build out -t run_lbtc_tests` |
| libbitcoin no-libsecp build mode | `ci/check_core_build_mode.py` (+ bridge build) | Bridge builds and validates with libsecp256k1 absent | `python3 ci/check_core_build_mode.py` |
| Bitcoin Core full suite | `docs/BITCOIN_CORE_TEST_RESULTS.json` | UltrafastSecp256k1 as the secondary backend: **749/749 pass (GCC 14.2.0)**; earlier 693/693 (Clang 19.1.7, v28.0) preserved in history | see the `reproduction` field in `docs/BITCOIN_CORE_TEST_RESULTS.json` |

## Opaque-layout note (internal, not a portable signature expectation)

The libsecp256k1 shim stores `secp256k1_ecdsa_signature.data` / `secp256k1_pubkey.data`
in an **internal opaque layout** (little-endian scalar storage; full `X‖Y` pubkey
bytes — fix `684141e7`). libbitcoin raw-signature fixtures depend on this layout.
This is an *internal representation*, **not** a portable serialized-signature
expectation: callers must round-trip through the shim's serialize/parse entry
points rather than assuming a byte layout. Divergences from upstream
libsecp256k1 are tracked in `docs/SHIM_KNOWN_DIVERGENCES.md`.

## Performance vs correctness separation

ConnectBlock / batch-verify throughput and microbenchmarks are **not** correctness
evidence and are kept in `docs/BITCOIN_CORE_BENCH_RESULTS.json` and
`docs/bench_unified_*.json`, gated separately by
`ci/check_bench_doc_consistency.py` and the perf/security co-gate
(`ci/perf_security_cogate.py`). A performance claim is invalid while any
correctness row above is red (Bastion B8 co-gating).

## Scheduled heavy validation (tracked enhancement)

The libbitcoin bridge correctness tests run on demand and via the differential
lanes. A dedicated scheduled libbitcoin-bridge conformance lane (weekly) that
runs `run_lbtc_tests` with a GPU auto-detect fallback and archives a result JSON
is a recommended future addition; it is **owner-gated CI infrastructure** and is
not created here.

---

*Generated for Bastion B7. Update this index whenever an integration surface,
regression vector, or result artifact changes.*
