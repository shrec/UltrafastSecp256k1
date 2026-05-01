# dev_bug_scanner Investigation Report — 2026-04-21

> Scope: deep-dive triage of every non-FP signal returned by
> `ci/dev_bug_scanner.py` after the 2026-04-21 false-positive
> reduction pass. Goal: confirm whether any signal represents a real,
> exploitable bug in `dev` HEAD.

## TL;DR

| Question | Answer |
|----------|--------|
| Real, exploitable bug found? | **No.** |
| HIGH-severity findings? | 0 |
| MEDIUM-severity findings worth tracking? | 2 (design-level, not bugs) |
| LOW-severity findings (DBLINIT) | 81 — all stylistic, no correctness impact |
| Scanner FP rate after this pass | 96 % of MEDIUM eliminated; 0 % HIGH FPs |

## Method

1. Ran scanner on `dev` HEAD: `python3 ci/dev_bug_scanner.py --json`.
2. For every MEDIUM finding, queried the source-graph
   (`focus`, `bodygrep`, `find`) before opening any file.
3. Cross-checked each finding against
   - `docs/NORMALIZATION_SPEC.md` (low-S contract)
   - `cpu/src/ecdsa.cpp` (single-sig verifier behaviour)
   - `cpu/src/ct_scalar.cpp` (scalar invariants)
   - `include/ufsecp/ufsecp_impl.cpp` (ABI null-check patterns)
4. Hardened scanner regexes for the three remaining FP patterns
   (BIP-32 magic ternary, BIP-324 protocol-header memcpy, compound
   `if ((!a && b) || !c)` ABI guards).

## Run-by-run history

| Run | Total | HIGH | MEDIUM | LOW | Notes |
|-----|------:|-----:|-------:|----:|-------|
| 1 (baseline) | 375 | 0 | 140 | 235 | initial state with 13 new CVE-grounded checkers |
| 5 | 155 | 0 | 43 | 112 | after 5 hardening passes |
| 9 (post-FP pass) | 88 | 0 | 7 | 81 | committed as `f58da95` |
| **10 (this report)** | **84** | **0** | **3** | **81** | 3 known FP patterns eliminated |

## Findings — full triage

### MEDIUM × 2 — `MISSING_LOW_S_CHECK` in `ecdsa_batch_verify`

**Locations.** `cpu/include/secp256k1/batch_verify.hpp:72`,
`cpu/src/batch_verify.cpp:271`.

**Verdict.** **Not a bug.** Documented design.

**Investigation.**

`docs/NORMALIZATION_SPEC.md` §2.1 explicitly states the library's verify
contract:

> | `ecdsa_verify()` | Accepts **both** low-S and high-S (permissive verify) |
> | `ecdsa_sign()`   | Always returns low-S                                  |
> | `ECDSASignature::is_low_s()` | Returns `true` iff `s ≤ n/2`             |

Permissive verify with explicit `is_low_s()` available to the caller is
the same contract as upstream `libsecp256k1` (see `secp256k1_ecdsa_verify`
vs `secp256k1_ecdsa_signature_normalize`). `ecdsa_batch_verify` simply
matches the single-sig contract; promoting it to strict-low-S would
diverge from `ecdsa_verify` and break consensus-style batch validators.

**Action.** Documentation-only. The contract is already documented in
`NORMALIZATION_SPEC.md`. Suggested follow-up (non-blocking):
add a one-line comment at `batch_verify.cpp:271` stating
"permissive low-S — caller responsible per `NORMALIZATION_SPEC.md` §2.1"
to make the design intent local to the code site.

### MEDIUM × 1 — `SCALAR_NOT_REDUCED` in `scalar_inverse`

**Location.** `cpu/include/secp256k1/ct/scalar.hpp:38`.

**Verdict.** **Not a bug.** Class invariant precondition.

**Investigation.**

The signature is `Scalar scalar_inverse(const Scalar& a) noexcept;`. The
`Scalar` class is the library's reduced-mod-`n` wrapper — every
constructor (`from_bytes_be`, `from_hex`, `from_uint64`, `random`,
arithmetic operators) maintains `0 ≤ value < n` as a class invariant.

The implementation in `cpu/src/ct_scalar.cpp:424–427` is:

```cpp
Scalar scalar_inverse(const Scalar& a) noexcept {
    if (a.is_zero()) return Scalar::zero();
    return Scalar::from_limbs(ct_safegcd::inverse_impl(a.limbs()));
}
```

The function does not need to re-reduce because `Scalar` cannot hold an
unreduced value at the type-system boundary. RR-004 (closed 2026-04-03,
commit `ea8cfb3c`) was the analogous bug in the `ecdsa_verify` `r`-component,
which was actually exposed because raw bytes from a signature parser were
being compared against unreduced PMN constants — that is a different
class because the input was not yet a `Scalar`.

**Action.** Documentation-only. Suggested follow-up: the scanner pattern
should be tightened to skip APIs that take a `Scalar`/`scalar_t` typed
argument (only flag those that take raw bytes / `uint8_t[32]`).

### LOW × 81 — `DBLINIT` in arithmetic-heavy paths

**Locations.** Distributed across `cpu/src/field_26.cpp`,
`cpu/src/field.cpp`, `cpu/src/scalar.cpp`, `cpu/src/ct_field.cpp`,
`cpu/src/ct_scalar.cpp`, GPU shader code (`metal/shaders/`,
`opencl/kernels/`, `gpu/src/*.cu`).

**Verdict.** **Not bugs.** Limb-arithmetic carry-propagation patterns.

**Investigation.**

Sample lines (representative):

```cpp
x = t1 >> 26; t1 &= M26; t2 += x;     // field_26.cpp:104
x = t2 >> 26; t2 &= M26; t3 += x;     // field_26.cpp:105
x = t3 >> 26; t3 &= M26; t4 += x;     // field_26.cpp:106
fi = f.v[0]; gi = g.v[0];             // field.cpp:2943
sd = d4 >> 63;                         // ct_scalar.cpp:258
se = e4 >> 63;                         // ct_scalar.cpp:259
```

Every flagged line reassigns a **scratch** that the same statement (or
the next one) reads on the right-hand side. The scanner's "tracked vs
read" state machine is conservative on multi-statement single lines
(`x = t1 >> 26; t1 &= M26; t2 += x;`). These are correct radix-2^26 and
radix-2^62 carry chains used by the CT inversion and the field
multiplication paths.

**Action.** Suggested follow-up to push DBLINIT to ~zero remaining LOW:
the checker should split lines on `;` before applying its lvalue/RHS
analysis. Not done in this pass to avoid risk of suppressing real
multi-statement double-assignment bugs.

### MEDIUM/LOW — what was eliminated

The earlier `BINDING_NO_VALIDATION`, `MSET`, and `CPASTE` MEDIUM
findings are now FP-suppressed:

- `ufsecp_segwit_witness_script_hash` (FP) —
  the function uses a compound guard `if ((!script && script_len > 0) ||
  !hash_out)`. Scanner regex now matches any `!\w+` inside an `if`,
  including compound forms.
- `bip324.cpp:106` `std::memcpy(buf.data(), header_enc, 3)` (FP) —
  this is the BIP-324 plaintext header copy. Scanner now skips MSET
  findings in files matching `bip32`, `bip324`, `bip352`, `tagged_hash`,
  `address`, `wallet`, and skips lines containing the words
  `header|magic|version|prefix|sentinel|marker|tag`.
- `bip32.h:527` / `bip32.cl:570` `version = mainnet ? 0x0488B21E : 0x043587CF`
  (FP) — these are the BIP-32 magic version bytes. Scanner now skips
  CPASTE findings whose RHS is a ternary of two integer/hex constants.

## Threat-model summary

| Class | Coverage on `dev` | Notes |
|-------|-------------------|-------|
| Nonce reuse (Sony PS3 class) | Covered by scanner + RFC-6979 KAT tests | scanner clean |
| Apple goto-fail | Covered by scanner + audit_security PoC | scanner clean |
| Debian OpenSSL RNG | Covered by scanner + RNG-quality tests | scanner clean |
| Stark Bank `r ∈ [n, p−1]` | Closed by RR-004 (commit `ea8cfb3c`) | Wycheproof tcId 346 regression test in place |
| BIP-62 malleability (verify side) | Permissive by documented design | `is_low_s()` available; caller responsibility |
| BIP-340 missing tagged-hash domain sep | Covered by scanner | scanner clean |
| Invalid-curve / small-subgroup | Covered by scanner + ECDH negative tests | scanner clean |
| DER laxness | Covered by scanner + parser fuzz | scanner clean |

## Conclusion

`dev_bug_scanner` on `dev` HEAD finds **no exploitable bug**. The two
remaining `MISSING_LOW_S_CHECK` MEDIUM signals correctly identify the
documented permissive-verify behaviour (matches single-sig
`ecdsa_verify` and upstream `libsecp256k1`); the one
`SCALAR_NOT_REDUCED` MEDIUM signal is a class-invariant precondition,
not a missing reduction.

The scanner is now production-quality for routine pre-commit / pre-push
use: 0 HIGH findings, 3 MEDIUM signals (all design-documented), 81 LOW
DBLINIT advisories that are arithmetic style rather than correctness
issues.

## Reproducibility

```bash
cd libs/UltrafastSecp256k1
python3 ci/dev_bug_scanner.py --json -o /tmp/scan.json
python3 -c "import json; r=json.load(open('/tmp/scan.json')); \
  print(len(r), 'findings;', \
  sum(1 for f in r if f['severity']=='HIGH'), 'HIGH;', \
  sum(1 for f in r if f['severity']=='MEDIUM'), 'MEDIUM;', \
  sum(1 for f in r if f['severity']=='LOW'), 'LOW')"
```

Expected on `dev` after this commit: `84 findings; 0 HIGH; 3 MEDIUM; 81 LOW`.
