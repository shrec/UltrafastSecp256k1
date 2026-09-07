# P3: bounded deferred-normalization field sums — 2026-09-05

Status: two native C++ series measured and reviewed; retain experimental
candidates, with the single-cell duration-floor limitation below. No production
optimization was integrated. Field research is not complete.

## What changed

For `(x0 + sum(N canonical RHS)) mod p`, when only the final canonical result
is observable, deferred FE52 normalization substantially improves the actual
corrected FE64 serial sum in this build. **N counts RHS; x0 is added once.**

- At every measured N >= 16, all six candidates beat FE64 in all same-round
  comparisons: 480/480 across both series. Full1's gain is only about 1.09–1.24x;
  deferred schedules are materially faster than that per-step-normalized control.
- At N=1, every candidate's paired median loses in both series. Four isolated
  candidate wins occur among the 96 pairs; no universal small-input winner.
- E2E Full256 gives 5.644–5.809x at N=16 and 11.661–11.923x at N=256.
  E2E Full4094 gives 12.502–12.564x at N=4096 and 12.556–12.579x at N=65536.
- At N=1048576, E2E Full16 gives 6.968–7.380x and is the strongest median
  E2E candidate in both series. The largest safe chunk is not always fastest.
- Resident Weak4095 reaches 29.474/30.129x at N=256, but its input preconversion
  is excluded. It is a DIFFERENT setup contract, not a 30x drop-in field add.
  At N=1048576 its gain falls to 5.808/5.631x.

These are final-only field-p vector sums on one machine/profile, not scalar-n
F2 results, individual field-add latencies, a whole-engine gain, a CT approval,
a novelty claim or a world-speed record. The illustrative retained schedules
above were selected after examining the preregistered full matrix; they do not
establish unmeasured crossover thresholds or a production dispatcher.

## All paired baseline comparisons

Entries: median of eight SAME-ROUND FE64/candidate ratios, S1 / S2. Above one
favors the candidate. All measured records are retained, including the short
N=1 Full16 S2 records. A median of ratios is not a ratio of separate medians.
All columns except the last accept FE64 input and pack RHS inside the sum.

| N RHS | Full1 | Full16 | Full256 | Full4094 | Weak4095 | Resident Weak4095 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.907 / 0.903 | 0.904 / 0.887 | 0.880 / 0.887 | 0.870 / 0.880 | 0.651 / 0.665 | 0.802 / 0.782 |
| 16 | 1.109 / 1.090 | 5.567 / 5.602 | 5.809 / 5.644 | 5.485 / 5.587 | 4.478 / 4.478 | 7.692 / 7.724 |
| 256 | 1.177 / 1.209 | 9.065 / 9.379 | 11.923 / 11.661 | 11.351 / 11.751 | 11.519 / 11.043 | 29.474 / 30.129 |
| 4096 | 1.224 / 1.224 | 8.960 / 9.316 | 12.387 / 12.252 | 12.502 / 12.564 | 12.294 / 12.149 | 24.602 / 25.228 |
| 65536 | 1.215 / 1.237 | 9.271 / 9.112 | 12.021 / 11.734 | 12.579 / 12.556 | 12.663 / 12.507 | 13.806 / 13.315 |
| 1048576 | 1.226 / 1.211 | 7.380 / 6.968 | 6.177 / 6.139 | 6.369 / 6.082 | 6.379 / 6.071 | 5.808 / 5.631 |

Resident seed packing, final canonical decoding and output materialization are
timed; only resident RHS preparation is excluded. Allocation/corpus creation
are outside repeated jobs for every route. A one-time resident setup observation
is recorded descriptively, not as a controlled setup-inclusive crossover test.

## Absolute normalized region means

Median ns/RHS, S1 / S2. Whole-sum ns is this value multiplied by N; raw JSON
also records ns/sum directly. This is not a standalone-add latency, and includes
amortized job call, seed, loads, conversion, reduction and full-byte output.

| N RHS | FE64 | Full1 | Full16 | Full256 | Full4094 | Weak4095 | Resident Weak4095 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 11.023 / 11.531 | 12.195 / 12.605 | 12.302 / 12.976 | 12.495 / 12.934 | 12.708 / 13.062 | 16.923 / 17.053 | 13.888 / 14.601 |
| 16 | 8.912 / 9.355 | 8.222 / 8.591 | 1.615 / 1.669 | 1.522 / 1.675 | 1.623 / 1.689 | 1.997 / 2.072 | 1.142 / 1.218 |
| 256 | 9.704 / 10.537 | 8.284 / 8.711 | 1.083 / 1.133 | 0.812 / 0.882 | 0.870 / 0.887 | 0.852 / 0.934 | 0.333 / 0.349 |
| 4096 | 9.982 / 10.591 | 8.093 / 8.681 | 1.099 / 1.147 | 0.815 / 0.866 | 0.793 / 0.844 | 0.816 / 0.867 | 0.407 / 0.424 |
| 65536 | 10.130 / 10.606 | 8.191 / 8.633 | 1.099 / 1.154 | 0.850 / 0.874 | 0.801 / 0.848 | 0.806 / 0.847 | 0.742 / 0.794 |
| 1048576 | 9.909 / 10.575 | 8.457 / 8.755 | 1.384 / 1.472 | 1.599 / 1.729 | 1.540 / 1.739 | 1.562 / 1.743 | 1.746 / 1.862 |

For scale, the N=1048576 FE64 sum medians are 10.391/11.089 ms, versus
1.451/1.543 ms for E2E Full16. These separate medians illustrate elapsed cost;
the speedup table still uses paired ratios.

## Schedule and representation findings

Exploratory posthoc SAME-ROUND internal comparisons, not additional preregistered
performance claims or causal component ablations:

- At N=1048576, Full16 beats Full4094 in 15/16 pairs and Weak4095 in 15/16.
  More frequent normalization can coexist with lower total time.
- At that same N, E2E Weak4095 beats resident Weak4095 in 15/16 pairs. It loads
  32-byte FE64 inputs and packs on use; resident loads 40-byte FE52 inputs.
  Their same normalization schedule makes this a useful representation/control
  observation, but does not isolate memory bandwidth, cache behavior or packing
  cost. No traffic counters were available.
- Machine-code inspection shows per-RHS scalar carry work for FE64/Full1,
  an unrolled/vectorized 16-input chunk with explicit stack slots for Full16,
  four FE64 inputs per larger E2E vector iteration, and two FE52 inputs per
  resident vector iteration. The latter use packed integer adds. This supports
  the structural observation that deferring canonicalization permits different
  instruction scheduling; it does not quantify each component's contribution.

The [codegen report](P3_FIELD_SUM_CODEGEN.md) distinguishes inner-loop accesses,
surrounding work and individual symbol sizes. Its 12 retained disassemblies and
12 symbol sizes were replayed against the exact binary by the manager. Repeated
jobs remain between clock calls and every job stores all 32 output bytes.
No dynamic instruction count, spill cost, critical-path cycle count, cache or
bandwidth attribution follows from static code alone.

## Correctness, headroom and input contracts

The shared experimental kernels admit an arbitrary canonical prefix seed plus
at most 4094 RHS before direct full normalization, or 4095 RHS before weak then
full. Seed conversion happens once globally; the running canonical prefix
counts once toward each local budget. Final `to_fe()` supplies exactly one
full normalization. N=0/null returns x0; positive N requires valid canonical
input ranges. Same-type seed/RHS aliasing and input preservation are supported.

The independent [integer-bound argument](P3_FIELD_SUM_CONTRACT.md) is specific
to sums of canonical terms, not subtraction, arbitrary magnitude or products.
P2's unsafe unsplit 4096-term direct-normalization witness remains unchanged;
the admitted P3 schedules split at the required boundaries. No production fix
to that separate normalization-headroom contract was made.

The same finite C++ corpus passed three manager-run configurations:

1. GCC14 native ASM linked to the frozen corrected P1 library.
2. Independently compiled Clang18 field sources with the retained GAS object.
3. GCC NO_ASM, ASan+UBSan with halt-on-error and leak checking.

Each: 147 fixtures, 283 seeded/alias cases, all five schedules in both layouts,
3,703,557 assertions, zero mismatches, output checksum `6c81b078909211cb`.
Checks include canonical raw limbs, all 32 bytes, 136 aliases, 88 null identities,
43 comparator negative controls and complete input-byte preservation. Counts
overlap; repetitions across compilers are not extra distinct fixtures.
Checkpoint replay covers N <= 8193 and checks 461744 exact raw accumulations,
922808 canonical checkpoints and 208 weak-output bounds. The three large fixtures
check complete calls and preserved inputs, not every checkpoint. Replay invokes
the actual exposed phase helpers separately, not instrumented timed kernels.

Three forbidden template schedules fail compilation at the intended assertions:
Full4095, Weak4096 and chunk zero. See [validation](data/p3_validation_20260905.json).
Finite tests and the written bound argument are not machine-checked proof or
side-channel certification.

## Raw evidence and one duration-floor deviation

Seven routes x six sizes, eight measured rounds and two warmups per series.
Actual N-element prefixes from one deterministic million-element corpus;
source-bound SplitMix64 recipe, x0, complete-byte hashes and prefix hashes are
recorded. The input arrays themselves are reconstructed, not archived as a
million hexadecimal values. FE64 checksum `a228d8a510693ab4`; resident raw
checksum `7efcc456d19c1fbd`. All inputs are regenerated and checked after
untimed validation and after the regions.

- S1: 569 records = 149 calibration + 84 warmup + 336 measurement; 65 short
  calibration records only. Measured durations 225.145734–378.393837 ms.
- S2: 573 records = 153 calibration + 84 warmup + 336 measurement; 67 short
  calibration records, ONE short warmup and EIGHT short measurement records.
  Overall measured durations 173.934008–463.107908 ms.
- The later short records are ONLY N=1 E2E Full16. Its same-count calibration
  passed at 279.320339 and 283.242527 ms, but measured time later became
  173.934008–195.343627 ms. All N>=16 warmup/measurement regions meet 200 ms.
  Process completion and calibrated qualification therefore must not be read as
  universal later-phase duration qualification. No records were discarded or
  selectively rerun. [Exact deviation](data/p3_duration_floor_deviation_20260905.json).

All full-series CPU endpoints are CPU4. Powersave governor and enabled turbo
were unchanged; endpoint frequency is not an average. No owned build,
correctness test or competing benchmark overlapped either full series. External
services, indexing, thermal conditions and SMT sibling activity remain
uncontrolled. PMU access was denied; [obstacles](data/p3_environment_obstacles_20260905.json)
also records transient Source Graph catch-up and unavailable worker-role tools.
Manager file receipts ultimately match the frozen source hashes.

Raw [S1](data/p3_run1_20260905.json) / [S2](data/p3_run2_20260905.json);
independent [numerical audit](data/p3_numeric_audit_20260905.json);
[build/source manifest](data/p3_build_20260905.json). Full series use GCC14.2,
native ISA and LTO OFF; Clang correctness/smoke does not establish Clang speed.
The corrected library hash is unchanged from P1/P2.
The independent audit recomputed 16,215 numeric/identity checks with zero errors,
including exact timestamp differences and the disclosed duration-floor deviation.

## Collected frontier, not production integration

The [25-lens outcome map](P3_FIELD_SUM_LENS_RESULTS.md) records what is measured,
what is derived/static, and what remains unresolved. Native 4x64 accumulation
controls, setup-inclusive crossover, additional compiler/platform profiles,
later-phase duration qualification and security remain open. Product/reducer
scheduling and inverse/batch-inverse field contracts are subsequent field-only
work; canonical-sum bounds must not be imported into those domains.

Keep P2's single-operation/region findings, this P3 vector primitive, and scalar
F2 separate in the [central ledger](RESEARCH_FRONTIER.md). No production,
scalar/upper-layer, default build/CI, commit, push or tag change was made here.
Owner Task MCP suspension remains in force; this is local manager review,
not canonical task acceptance. Earlier P1/P2/F2 reports and code are preserved.
