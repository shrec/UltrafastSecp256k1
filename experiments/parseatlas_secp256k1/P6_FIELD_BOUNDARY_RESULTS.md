# P6 field boundary controls — measured results

Status: both full series complete; retained experimental observations, not production integration.
The [preregistered protocol](P6_FIELD_BOUNDARY_PROTOCOL.md), [value/ABI contract](P6_FIELD_BOUNDARY_CONTRACT.md)
and [25-lens plan](P6_FIELD_BOUNDARY_LENSES.md) were fixed before timing.
See the [measured lens map](P6_FIELD_BOUNDARY_LENS_RESULTS.md) for all25 identities.

## Outcome

The unchanged existing ASM leaf is a useful optimization target **at the calling boundary**.
It beats the actual FE64 API in64/64 registered paired observations:
multiply chain1 1.1173/1.1228x, multiply ILP4 1.2392/1.2364x,
square chain1 1.0645/1.0702x and square ILP4 1.2990/1.2888x (S1/S2).
This reuses existing arithmetic; it is not a new multiplication algorithm or whole-engine gain.

The P5 inline-row chain candidate remains conditional. Against actual API,
multiply wins15/16 pairs at1.1291/1.1341x and square16/16 at1.1302/1.1645x.
Against **direct ASM**, row-inline multiply has only1.01058/1.01099x paired medians,
12/16 wins and observed losses: treat this as a near tie, not a robust new method.
Row-inline square beats direct ASM16/16 at1.07447/1.08815x in these two GCC series.

The same row source **does not retain its chain advantage behind an opaque function boundary**.
Outlined row loses to API64/64 across the four groups, and to direct ASM64/64.
All row ILP4 candidates lose to API64/64 as well.
Outlining is not universally harmful: outlined row multiply beats inline row multiply
in ILP4 at1.08968/1.09173x,16/16 pairs, but both still lose to the existing ASM path.
These counts overlap comparisons and must not be added as independent experiments.

## Exact observation boundary

Four families: actual corrected FE64 API, direct unchanged ASM, unchanged P5 row
product/serial reducer inline, and the same row composition in a separate-TU C pointer-ABI leaf.
Canonical inputs and raw canonical outputs throughout; p=2^256-2^32-977.
a/b may alias read-only; output is a disjoint four-word range. Caller BMI2/ADX
admission precedes all direct operations; the native direct wrapper adds no per-call guard.
No arithmetic proof, intermediate-width bound or carry fold changed.

There are16cells: multiply/square x chain1/ILP4 x four families.
Each job initializes its active seeds, executes1024steps/lane and emits128bytes:
one BE32 plus96zero bytes for chain1, four BE32 results for ILP4.
Multiply uses the shared canonical RHS at step&255; square uses x->x² without RHS.
ILP4 is four independent scalar states on one core, not threads or a vector-multiply claim.
Generation/validation are untimed; reset, recurrence and complete output are timed.
No per-step barrier. Each region checks its last identical job outside timing.

The exact P5 corpus FNV is7c2e57392893ac80:4seeds,256canonical RHS and256raw512 entries.
The raw512 array is retained only for identity/preservation and is unused by P6 arithmetic.
Every series checks40960 trace states, raw canonical limbs and all bytes; no zero/one state
occurred. Complete evaluator/job outputs and cross-mode lane0 agreement pass.
Both pre/post checks regenerate and compare every input word, not only its hash.
The independent Boost correctness corpus below is a different fixed fixture recipe.

## Measurements

Only two full GCC14.2.0 -O3 -march=native -fno-lto series on this i5-14400F profile.
CPU4, powersave, turbo enabled; sibling5/frequency/thermals/background services uncontrolled.
The second series reverses the complete preregistered order.
Source/object/binary/build identities are in the [premeasurement freeze](data/p6_performance_freeze_20260905.json).
A retained-object relink reproduced the timed executable byte-for-byte.

| Series | All regions | Calibration (short) | Warmup / measured | Min later duration | Later extensions | All probes |
|---|---:|---:|---:|---:|---:|---:|
| S1 | 214 | 54 (19) | 32 /128 | 200.387601ms | 6 (1warm,5measured) | 220 |
| S2 reversed | 208 | 48 (16) | 32 /128 | 200.716604ms | 3 (1warm,2measured) | 211 |

All320 later regions qualify at>=200ms, all422 region outputs verify, endpoint CPU samples
are4. There are35 retained short calibration records and9 continuous extensions.
Two consecutive same-count qualifying calibrations are required per cell; a short
later prefix is extended under the original clock, never restarted/discarded.
All actual jobs, operation counts, endpoints, cumulative probes, outputs and ratios remain in
[raw S1](data/p6_run1_20260905.json) and [raw S2](data/p6_run2_20260905.json).
Raw source_recipe_sha256 placeholders are externally bound to the frozen driver hash;
the raw files are not rewritten.

Normalized costs below are medians of eight **region-mean** ns/operation values;
they are not individual-operation latency percentiles.
Each registered ratio is instead the median of eight same-round A/B ratios.
Do not derive paired ratios by dividing the following independently computed medians.

| Cell / route | Mode | S1 median ns/op | S2 median ns/op |
|---|---|---:|---:|
| 0: mul API | chain1 | 24.322 | 25.211 |
| 1: mul direct ASM | chain1 | 21.310 | 22.459 |
| 2: mul row inline | chain1 | 20.592 | 22.217 |
| 3: mul row outlined | chain1 | 30.681 | 31.131 |
| 4: mul API | ILP4 | 12.162 | 12.927 |
| 5: mul direct ASM | ILP4 | 9.947 | 10.447 |
| 6: mul row inline | ILP4 | 17.351 | 18.454 |
| 7: mul row outlined | ILP4 | 15.738 | 16.757 |
| 8: square API | chain1 | 22.827 | 23.885 |
| 9: square direct ASM | chain1 | 21.151 | 22.285 |
| 10: square row inline | chain1 | 19.885 | 20.500 |
| 11: square row outlined | chain1 | 28.029 | 29.444 |
| 12: square API | ILP4 | 11.944 | 11.954 |
| 13: square direct ASM | ILP4 | 9.238 | 9.339 |
| 14: square row inline | ILP4 | 16.192 | 16.424 |
| 15: square row outlined | ILP4 | 16.456 | 16.866 |

### Primary: API / each candidate

A/B>1 favors B. Each combined win count has16 observed pairs, eight per series.
The raw files retain every ratio and min/median/max; no P5, smoke or synthetic data is pooled.

| Numerator / denominator | S1 paired median | S2 paired median | Denominator wins / losses / ties (16) |
|---|---:|---:|---:|
| mul API chain1 / mul direct ASM chain1 | 1.117278 | 1.122834 | 16 / 0 / 0 |
| mul API chain1 / mul row inline chain1 | 1.129094 | 1.134142 | 15 / 1 / 0 |
| mul API chain1 / mul row outlined chain1 | 0.810039 | 0.811185 | 0 / 16 / 0 |
| mul API ILP4 / mul direct ASM ILP4 | 1.239191 | 1.236437 | 16 / 0 / 0 |
| mul API ILP4 / mul row inline ILP4 | 0.692477 | 0.702033 | 0 / 16 / 0 |
| mul API ILP4 / mul row outlined ILP4 | 0.759619 | 0.773509 | 0 / 16 / 0 |
| square API chain1 / square direct ASM chain1 | 1.064527 | 1.070183 | 16 / 0 / 0 |
| square API chain1 / square row inline chain1 | 1.130220 | 1.164467 | 16 / 0 / 0 |
| square API chain1 / square row outlined chain1 | 0.805813 | 0.810953 | 0 / 16 / 0 |
| square API ILP4 / square direct ASM ILP4 | 1.299023 | 1.288793 | 16 / 0 / 0 |
| square API ILP4 / square row inline ILP4 | 0.742597 | 0.727815 | 0 / 16 / 0 |
| square API ILP4 / square row outlined ILP4 | 0.715018 | 0.709334 | 0 / 16 / 0 |

### Secondary: ASM / row and inline / outlined

| Numerator / denominator | S1 paired median | S2 paired median | Denominator wins / losses / ties (16) |
|---|---:|---:|---:|
| mul direct ASM chain1 / mul row inline chain1 | 1.010576 | 1.010992 | 12 / 4 / 0 |
| mul direct ASM chain1 / mul row outlined chain1 | 0.724057 | 0.722205 | 0 / 16 / 0 |
| mul row inline chain1 / mul row outlined chain1 | 0.714361 | 0.715606 | 0 / 16 / 0 |
| mul direct ASM ILP4 / mul row inline ILP4 | 0.570013 | 0.570815 | 0 / 16 / 0 |
| mul direct ASM ILP4 / mul row outlined ILP4 | 0.608213 | 0.623587 | 0 / 16 / 0 |
| mul row inline ILP4 / mul row outlined ILP4 | 1.089681 | 1.091729 | 16 / 0 / 0 |
| square direct ASM chain1 / square row inline chain1 | 1.074473 | 1.088145 | 16 / 0 / 0 |
| square direct ASM chain1 / square row outlined chain1 | 0.759907 | 0.755753 | 0 / 16 / 0 |
| square row inline chain1 / square row outlined chain1 | 0.698850 | 0.695687 | 0 / 16 / 0 |
| square direct ASM ILP4 / square row inline ILP4 | 0.567236 | 0.564543 | 0 / 16 / 0 |
| square direct ASM ILP4 / square row outlined ILP4 | 0.551646 | 0.556519 | 0 / 16 / 0 |
| square row inline ILP4 / square row outlined ILP4 | 0.970226 | 0.979069 | 3 / 13 / 0 |

### Mode: chain1 / ILP4 per primitive

| Numerator / denominator | S1 paired median | S2 paired median | Denominator wins / losses / ties (16) |
|---|---:|---:|---:|
| mul API chain1 / mul API ILP4 | 1.939415 | 1.950025 | 16 / 0 / 0 |
| mul direct ASM chain1 / mul direct ASM ILP4 | 2.181870 | 2.152216 | 16 / 0 / 0 |
| mul row inline chain1 / mul row inline ILP4 | 1.224294 | 1.197380 | 16 / 0 / 0 |
| mul row outlined chain1 / mul row outlined ILP4 | 1.847814 | 1.855780 | 16 / 0 / 0 |
| square API chain1 / square API ILP4 | 1.914694 | 1.999932 | 16 / 0 / 0 |
| square direct ASM chain1 / square direct ASM ILP4 | 2.354439 | 2.390825 | 16 / 0 / 0 |
| square row inline chain1 / square row inline ILP4 | 1.247637 | 1.238545 | 15 / 1 / 0 |
| square row outlined chain1 / square row outlined ILP4 | 1.712574 | 1.751068 | 16 / 0 / 0 |

The32summaries per series share cells;64summaries/512ratios are not512 independent experiments.
The [independent numeric audit](data/p6_numeric_audit_20260905.json) passes42411checks
with zero errors and zero published floating-point discrepancy. Manager
[full-object replay](data/p6_numeric_replay_20260905.json) reproduces it exactly. A separate
recomputation in the [build/evidence manifest](data/p6_build_20260905.json) binds these summaries
to actual elapsed time and work. Lossless integer checks are required for timestamps.

## Correctness and duration gates

Root ran mechanical qualification before reading all new implementation/test files and contracts.
[Correctness evidence](data/p6_validation_20260905.json):

| Configuration | Assertions | Mismatches | Direct ASM |
|---|---:|---:|---|
| GCC native + corrected frozen library | 406239 | 0 | 47498 checked outputs |
| Clang with independently compiled field sources + retained ASM object | 406239 | 0 | 47498 checked outputs |
| GCC NO_ASM ASan+UBSan | 358744 | 0 | 47498 explicitly skipped;2 disabled-wrapper rejections |

Shared checksum a9076830a8ab714b; native ASM checksum4993beaa5fe90e19.
Each configuration uses13102 pair entries/13074 distinct pairs (derived chain states excluded),
eight1024-step chains, both exact P1 carry KATs plus the older large-square KAT,
raw canonical output/all32bytes, same-input behavior and input preservation.
47498 guarded four-word output calls qualify the **new row C ABI**;
this does not claim canaries inside the existing ASM kernel.
45 negative observer controls check byte/limb/canary corruption.
The finite fixture is shared across configurations, not three independent corpora.
No ASM sanitizer instrumentation or unsupported-native-host emulation is claimed.

[Duration diagnostics](data/p6_duration_validation_20260905.json): both compilers pass
96synthetic cases/5282assertions, including continuous extensions, invalid clocks,
initial-cap rejection, malformed lanes, preservation and actual-main failure retention.
Each failed-main diagnostic exits2 and preserves33regions with13probes in the failed warmup.
Within-region cap exhaustion/capped tail remain statically reviewed only.
Both16cell real-clock smokes pass; their0.1ms timing is validation-only and may overlap other checks.
Exclusive-output rejection preserves the existing smoke file exactly.

## Emitted code and attribution

The [codegen report](P6_FIELD_BOUNDARY_CODEGEN.md) and [raw static evidence](data/p6_codegen_20260905.json)
retain66complete symbol slices,32GCC/Clang jobs and64call/output windows.
Manager [independent replay](data/p6_codegen_replay_20260905.json) passes10521checks,
including complete contiguous bytes/nm extents, all128B stores and eight matched caller pairs.

For both compilers, direct-ASM and outlined-row caller instructions match after only
normalizing local addresses and their named arithmetic-leaf destinations. This strengthens
the matched-boundary control; the arithmetic leaves still differ. Direct and API paths
reach the same existing ASM bytes. Actual API retains its cached dispatcher/returning API.
Inline row changes optimizer visibility, materialization, liveness and generated instructions;
neither comparison gives a constant pure call-instruction cost.

GCC already recognizes ten symmetric square products. It emits six reducer multiply sites
in the row paths, versus five in the existing ASM reducer. Clang outlines some source-inline
evaluators and shared reduction, inlines other ILP4 reductions, and has different static
instruction counts. Source operation counts alone therefore do not establish performance.

Important negative: Clang's shared reducer contains a result-dependent correction branch
at0x18cc7. No row path is accepted here as a constant-time production replacement.
Static stack operands, symbol sizes and call counts are not dynamic spills, bandwidth,
cycles, energy, cache residency or security certificates. PMU access remains unavailable.

## Environment and review limits

No owned compiler, arithmetic test or competing benchmark runs during full timing.
A final lightweight codegen-artifact jq/sha256sum/xxd check has no exact retained UTC timestamp:
brief overlap cannot be excluded. This [explicit supplemental note](data/p6_measurement_environment_note_20260905.json)
qualifies any stronger quiet-window interpretation; samples were not discarded or rerun.
This further discourages turning the1percent chain-multiply difference into a robust claim.
Frequency/SMT/thermal/background noise and this single-host/single-profile scope remain limits.

The [obstacle ledger](data/p6_environment_obstacles_20260905.json) retains the Clang ADX-builtin
admission fix, resolved new-source index lag, refresh-status inconsistency, unavailable worker
tool surfaces and PMU restriction. Source Graph hashes bind every frozen new code file.
Task MCP remains owner-suspended; workers use exact manager handoffs and stop for manager review.
A final prose-only codegen chronology correction distinguishes pre-timing raw-JSON replay
from post-series report review; no kernel, harness, binary or raw measurement changed.

## Disposition and next bounded field wave

Retain existing-ASM boundary optimization as a candidate; keep inline square as a
conditional chain candidate. Do not replace the current arithmetic universally.
The opaque row variant is a negative drop-in replacement result; inline multiply versus
direct ASM is near-tied. Preserve the ILP4 outlined-row-multiply improvement as a scheduling
observation, not a reason to choose it over faster ASM.

Next field experiments can separately test the final overflow*K selection, explicitly
unrolled Comba and fused product/reduction, each with written bounds, canonical byte-exact
oracles, fixed observation contracts and matched-boundary C++ timings.
They are not implemented or measured in P6. Then qualify inverse/batch inverse separately
under VT/CT, zero, alias, scratch and setup contracts. Platform/security gates remain.
Follow the [collected frontier](RESEARCH_FRONTIER.md): field first, scalar next, then combined
integration and actual upper-layer transfer. The earlier F2 scalar-sum and P3/P4 field-sum
gains remain intact; none is a standalone-add or whole-engine multiplier.
