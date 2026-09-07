# P2 field-only representation results — 2026-09-05

Status: canonical representation comparison measured; retain scenario-specific
candidates and negative results. No performance optimization was integrated.
The field investigation is not finished: bounded lazy regions, product/reducer
scheduling and inverse candidates remain next. Scalar F2 remains retained for
later combined integration, not superseded by this field experiment.

## What the experiment establishes

Four actual execution routes, five field operations, dependent chain1 and four
interleaved independent states (ILP4) on one core: 40 cells, two full series.

- Canonical resident FE52 wins add/sub in all 64 same-round pairs across both
  operations, modes and series. This includes full normalization every step.
- Resident FE52 multiply has about 1.08x paired-median advantage in chain1,
  but wins only 13/16 pairs. Retain a modest, noisy candidate signal.
- Resident square wins chain1 by about 1.06x, in all 16 pairs.
- Existing FE64 wins independent-state multiply and square in all 32 pairs.
  A representation that helps one dependent stream is not a universal winner.
- The canonical FE52 per-operation bridge loses all 128 add/sub/mul/square
  pairs. Its conversion, normalization and exact wrapper costs are included.
- FE52 inverse has no stable advantage: chain1 loses; ILP4 is close/mixed.
  Existing CT inversion is a separate security baseline, not interchangeable
  with a faster variable-time inverse.

These are measurements of existing representations and this exact harness,
not a novel field algorithm, whole-engine gain or world-speed record.

## Paired comparison

Entries are medians of eight SAME-ROUND FE64/candidate ratios, series1 / series2.
Above 1 favors the candidate; below 1 favors FE64. These are not ratios of
separately computed medians, which can differ. No samples were removed.

| Operation | Mode | FE52 bridge | Canonical resident FE52 |
|---|---|---:|---:|
| Add | chain1 | 0.691 / 0.713 | 1.172 / 1.190 |
| Add | ILP4 | 0.753 / 0.760 | 1.377 / 1.399 |
| Subtract | chain1 | 0.830 / 0.838 | 1.427 / 1.449 |
| Subtract | ILP4 | 0.728 / 0.730 | 1.323 / 1.344 |
| Multiply | chain1 | 0.800 / 0.791 | 1.082 / 1.083 |
| Multiply | ILP4 | 0.548 / 0.537 | 0.722 / 0.716 |
| Square | chain1 | 0.764 / 0.778 | 1.060 / 1.063 |
| Square | ILP4 | 0.680 / 0.673 | 0.896 / 0.874 |
| Inverse + lookup | chain1 | 0.975 / 0.951 | 0.960 / 0.954 |
| Inverse + lookup | ILP4 | 1.015 / 0.991 | 0.995 / 0.989 |

Resident RHS values are preconverted outside repeated timed jobs. Seed
conversion and final canonical decoding/32-byte output are inside each job.
This is an amortized region contract, not a drop-in FE64 API replacement.

## Absolute measurements

Median normalized region means, ns/op, series1 / series2. They include the
declared API, loop, index/load and amortized per-job materialization costs.

| Operation | Mode | FE64 API | FE52 bridge | Canonical resident FE52 | Existing CT API |
|---|---|---:|---:|---:|---:|
| Add | chain1 | 9.75 / 9.81 | 14.20 / 13.82 | 8.26 / 8.30 | 9.91 / 9.87 |
| Add | ILP4 | 5.76 / 5.82 | 7.61 / 7.57 | 4.16 / 4.13 | 6.25 / 6.40 |
| Subtract | chain1 | 12.06 / 12.05 | 14.40 / 14.46 | 8.49 / 8.23 | 11.80 / 11.98 |
| Subtract | ILP4 | 6.04 / 6.00 | 8.25 / 8.19 | 4.52 / 4.44 | 6.44 / 6.55 |
| Multiply | chain1 | 23.48 / 23.46 | 28.92 / 29.00 | 21.45 / 21.22 | 29.63 / 29.70 |
| Multiply | ILP4 | 12.24 / 12.03 | 22.51 / 22.32 | 17.16 / 16.68 | 22.79 / 23.08 |
| Square | chain1 | 20.69 / 21.06 | 27.36 / 27.28 | 19.55 / 19.59 | 27.57 / 27.15 |
| Square | ILP4 | 11.45 / 11.16 | 16.68 / 16.57 | 12.75 / 12.75 | 16.29 / 16.25 |
| Inverse + lookup | chain1 | 892.55 / 883.27 | 934.36 / 926.59 | 934.38 / 923.21 | 1412.89 / 1432.74 |
| Inverse + lookup | ILP4 | 932.20 / 916.79 | 924.54 / 914.06 | 925.09 / 919.40 | 1368.98 / 1428.35 |

Inversion selects the next nonzero input through a result-dependent lookup.
It avoids a two-value inverse recurrence but includes address/load cost; the
lookup-driven benchmark is not itself constant-time. Full zero-domain contracts
also differ: FE64 throws, while FE52 SafeGCD and the tested CT API return zero.

## Correctness and the new boundary failure

The final C++ Boost-integer/Euclidean oracle corpus passed:

1. GCC14, linked to the actual corrected P1 CMake library.
2. Clang18, independently compiling field.cpp, field_52.cpp, ct_field.cpp and
   field_asm.cpp, linked with the retained corrected GAS object.
3. GCC NO_ASM, ASan+UBSan, halt-on-error and leak checking enabled.

Each repeats the SAME corpus: 16,200 binary cases, 3,030 nonzero inverses per
route, 256 chains of 32 steps, P1's two explicit regression KATs, input
preservation, supported aliases and 75 comparator negative controls. Each
has 2,769,460 assertions, 824,624 FE64 output checks, 384,174 canonical FE52
output checks and 76,022 raw bridge-residue checks. Checks overlap by design;
these counts are not distinct random inputs. All three report zero mismatches
and checksum `ef652f4c3eada148`.

Canonical resident storage is checked BEFORE conversion can normalize a copy.
Raw bridge intermediates instead use the decode relation, then exactly one
canonical `to_fe()` boundary. Every output checks all 32 BE bytes.
Finite testing is not a universal proof or a side-channel certification.

A separate optional lazy-sum diagnostic reproduces a new normalization-boundary
failure in both GCC and Clang: 4096 canonical copies of
`{2^52-1,0,0,0,2^48-1}` fit in raw limb accumulators, but direct full
normalization loses exactly `2^64`. Weak normalization followed by full
normalization is correct. At 4097 the raw addition already overflows.

For sums of arbitrary canonical terms, a sufficient universal cap is 4095
TOTAL terms before direct full normalization, or 4096 before weak-then-full.
An arbitrary canonical initial accumulator counts toward that total. This is
not a general noncanonical-magnitude bound. It is retained as NeedFix for the
documented headroom/normalization contract; production callers were not shown
to reach the failing case and no production fix was made in this wave.

Evidence: [validation](data/p2_validation_20260905.json),
[lazy boundary and bound argument](data/p2_lazy_needfix_20260905.json).

## What 25 lenses changed in our decision

- Decode/observability: raw FE52 equality is not field equality; raw residue,
  canonical resident state and canonical FE64 return require different checks.
- Range/invariants: counting spare limb bits alone missed the normalization
  fold's own headroom. The 4096 witness and carry-first alternative make that
  constraint explicit before any lazy-region performance experiment.
- Conversion/dependency: resident versus bridge and chain versus ILP4 expose
  different winners. We will select by public workload contract, not insist
  on one representation for every operation.
- Resource/code shape: FE52 RHS storage is 10,240 bytes versus 8,192 for FE64.
  Linked multiply bridge is 1345 bytes; resident multiply jobs are 1653/2519
  bytes (chain1/ILP4). FE64 jobs are 198/365 bytes plus their callees, including
  the 610-byte native full-multiply kernel. These are symbol sizes, not total
  hot-path sizes or cache/spill evidence. [All symbols](data/p2_symbol_sizes_20260905.txt).
- Security/reachability: CT mul/square already use FE52 bridges with compiler
  barriers; CT inverse uses fixed divsteps. FE52 SafeGCD directly converts to
  signed62. Stale API comments must not define our benchmark categories.

The complete [25-lens map](P2_FIELD_25_LENSES.md) keeps unmeasured dependency
depth, spills, cache traffic, setup break-even and other targets explicitly
`PENDING IMPORT`. No claim of reduced memory bandwidth follows from these runs.

## Measurement and review provenance

Source HEAD/branch and corrected library are the frozen P1 baseline. GCC14.2,
C++20, native ISA, assembly/fast reduction enabled, LTO explicitly OFF. Inline
FE52 kernels are enabled. CPU4 is pinned; powersave governor and turbo enabled
were unchanged. Endpoint frequency is not an average, and external services,
thermal conditions and background indexing remain uncontrolled. No owned builds
or correctness runs overlap either full performance series.

Each job has 1024 operations per lane. Runtime corpus: 256 nonzero RHS values
plus four seeds, identical across routes/series; checksum `e706509ec61dda1d`.
Eight paired measured rounds and two warmups per cell, with reversed second
series ordering. Calibration requires two consecutive >=200ms regions at the
same job count. All raw records are retained: 520 in S1, 522 in S2, comprising
242 calibration, 160 warmup and 640 measurement regions. The 81 short regions
are calibration only; no warmup or measurement is short. Measured durations:
S1 247.760097–406.131241ms; S2 255.395411–403.868409ms.

Every job materializes its active full outputs; only the last identical job
in each region is byte-checked outside timing. Separate untimed replay checks
102,400 intermediate values and the full-job schedule per invocation.

Manager ran mechanical gates before reading the complete driver and oracle.
A separate read-only review found no blocker. The initial double-normalizing
bridge was corrected before full runs, and tests aligned to the single
normalization boundary. Pre-review binaries/smokes are retained but excluded.
Unary bridge's unused second argument can retain avoidable address/ABI work;
the result describes this exact wrapper, not the best possible unary bridge.

Raw series: [S1](data/p2_run1_20260905.json), [S2](data/p2_run2_20260905.json).
Build/source hashes: [manifest](data/p2_build_20260905.json).
Independent numerical replay: [audit](data/p2_numeric_audit_20260905.json).
It independently recomputed 13,831 checks with zero errors, including the
extra S2 short calibration and its correctly restarted qualification streak.

## Next field-only gate

Keep canonical resident add/sub as qualified experimental region candidates;
keep chain square and the noisier multiply signal separate from ILP4 losses.
Next, price bounded deferred-normalization regions with the derived sum bounds
and explicit decode/observation boundaries. Then investigate product/reduction
scheduling and inverse/batch-inverse contracts. No upper-layer work or combined
performance integration resumes until the owner's collection phase is complete.

P1/F2 files and existing production corrections are unchanged. This wave adds
experimental C++ tests/driver and evidence/docs only: no commit, push, default
build/CI change or production dispatcher change. Task MCP remains suspended by
the owner; verified manager Source Graph and exact worker scopes were used.
