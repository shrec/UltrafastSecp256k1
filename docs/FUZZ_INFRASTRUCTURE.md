# Fuzz Infrastructure

How UltrafastSecp256k1 fuzzes its parsers, signing/verify paths, and stateful
protocols — and how that fuzzing becomes Bastion-grade, freshness-gated evidence.

## Harnesses

LibFuzzer / deterministic-standalone harnesses live under `audit/` and
`src/cpu/fuzz/`:

| Surface | Harness |
|---------|---------|
| DER / signature parser | `audit/fuzz_der_parse.cpp`, `audit/test_fuzz_parsers.cpp` |
| ECDSA verify/sign | `audit/fuzz_ecdsa_verify.cpp`, `src/cpu/fuzz/fuzz_ecdsa.cpp` |
| Schnorr verify/sign | `audit/fuzz_schnorr_verify.cpp`, `src/cpu/fuzz/fuzz_schnorr.cpp` |
| Pubkey parse | `audit/fuzz_pubkey_parse.cpp` |
| BIP-32 path / BIP-324 frame | `audit/fuzz_bip32_path.cpp`, `audit/fuzz_bip324_frame.cpp` |
| Address + BIP32 + FFI boundary | `audit/test_fuzz_address_bip32_ffi.cpp` |
| MuSig2 / FROST stateful | `audit/test_fuzz_musig2_frost.cpp` |
| Field / point / scalar primitives | `src/cpu/fuzz/fuzz_{field,point,scalar}.cpp` |
| ABI hostile-caller / invalid-input grammar | `ci/invalid_input_grammar.py`, `audit/test_adversarial_protocol.cpp` |

Standalone deterministic mode (no LibFuzzer runtime) is built with
`-DSECP256K1_BUILD_LIBFUZZER_STANDALONE=ON`; LibFuzzer mode with
`-DSECP256K1_BUILD_LIBFUZZER=ON`. The dated `audit/ci-evidence/fuzz_*.txt`
snapshots record deterministic-mode run counts (e.g. `fuzz_parsers` = 580,019
cases) and are refreshed under the audit ci-evidence freshness SLO.

## Corpus

Seed corpora are committed under:

- `audit/corpus/{address,bip32,ffi}` (+ `MANIFEST.txt`, `README.md`)
- `src/cpu/fuzz/corpus/{fuzz_field,fuzz_point,fuzz_scalar}`

## Crash → regression discipline

**Every crash a fuzzer finds must become a permanent regression test.** Crash
artifacts are *not* left in the tree as raw reproducers — they are converted into
named `audit/test_*` regressions and the crash directory is emptied. The
crash→regression conversion is enforced by the freshness gate below: a crash
artifact present without a matching regression is a blocking failure
(`crash_unconverted`).

## Freshness gate (Bastion B15)

The fuzzing evidence is bound to a machine-readable manifest
[`docs/FUZZ_CAMPAIGN_STATUS.json`](FUZZ_CAMPAIGN_STATUS.json) and gated by
[`ci/check_fuzz_campaign_status.py`](../ci/check_fuzz_campaign_status.py) (also
`audit_gate.py --fuzz-campaign-status`, principle **G-15**). Each row binds a fuzz
surface to its `corpus_path`, `crash_path`, `regression_path`, a `replay_command`,
a `freshness_days` SLO, and a `severity`.

This is an **evidence-status gate** — it does **not** run long fuzz campaigns on
push (those run in `cflite.yml` / `cryptofuzz.yml` / `klee.yml` / the audit fuzz
suite). On every push it cheaply checks:

- a `blocking` row fails if its `corpus_path` is missing;
- a `blocking` row fails if `last_verified` is stale/malformed;
- **any** non-`owner_gated` row fails if `crash_path` holds a crash artifact
  (`crash-*`, `leak-*`, `timeout-*`, `oom-*`, `poc-*`, `*.crash`) without a
  matching `regression_path` (an unconverted crash is a correctness gap regardless
  of severity);
- `owner_gated` heavy/differential or host-only (`--gpu`) surfaces are surfaced
  explicitly and are **never** counted as current evidence;
- a per-row `days_until_block` runway + pre-alert is reported.

Run: `python3 ci/check_fuzz_campaign_status.py --json`.
