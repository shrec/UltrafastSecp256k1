# P5 field product/reducer results

Status: reviewed experimental evidence; arithmetic and numeric gates pass. Field-only
collection continues. No production, scalar or upper-layer integration.

## Outcome

Keep the inline row-product + serial-reduction implementation as a **conditional
chain candidate**. Against the corrected actual FE64 API, same-round median
speedups are 1.119451 / 1.107246 for multiplication (15/16 wins), and
1.078533 / 1.182849 for square (16/16 wins). These are two series on one GCC/native
profile, not confidence bounds, two compilers or a universal primitive gain.

Every new ILP4 candidate loses to the actual API: 96/96 paired comparisons.
Every Comba/symmetric-Comba candidate also loses to the API across both modes:
128/128 pairs. These counts overlap; they must not be added as disjoint samples.
Row+serial beats the corresponding Comba+serial route in all64secondary pairs.
Keep those negative results. There is no universal replacement selected.

The serial/precomputed reducer source alternatives mostly become the same
machine instructions. Their small timing differences do not establish a new
reducer algorithm or a causal scheduling gain. The winning row chain comparison
also changes compiler visibility, API boundaries and materialization: it is not
an isolated multiplication-count or call-overhead ablation.

## Exact measured contracts

[Protocol](P5_FIELD_PRODUCT_PROTOCOL.md), [independent bounds](P5_FIELD_PRODUCT_CONTRACT.md),
[preregistered lenses](P5_FIELD_PRODUCT_LENSES.md), and
[measured25-lens synthesis](P5_FIELD_PRODUCT_LENS_RESULTS.md).

Twenty cells: four multiplication routes, four square routes and two raw512
reducer routes, each chain1/ILP4. Each job initializes active states, performs
1024canonical steps per active lane, then materializes128bytes. Chain1 has one
active BE32 result plus96zero bytes; ILP4 returns all four BE32 results. These
output/reset costs are timed. Generation, allocation and prevalidation are not.
No per-step benchmark barriers; whole-job output remains observable.

Multiplication cycles256canonical nonzero RHS values, shared at step&255 by all
active lanes. Square evolves x->x² with no RHS mixing. Raw reduction uses low=x
XOR corpus-low and corpus-high as the upper half, an arbitrary512input rather
than necessarily a product. It includes XOR construction and must not be priced
as a field multiplication. All field/reducer steps remain canonical; no deferred
sum headroom from P3/P4 is imported. Lane0 agrees between dependency modes.

The deterministic seed20260905 corpus hash is7c2e57392893ac80. Each series checks
51,200 helper-trace states against the corrected FE64 reference, then checks
complete compiled evaluators/jobs against final trace outputs. No trace state
is zero or one. This is separate from the Boost test and not instrumentation
inside each timed job. Every timed region checks its final identical job's full
128bytes outside the clock. All4seeds/256RHS/256raw512inputs are regenerated and
compared after validation and after measurement, including raw words and hashes.

## Full observations, including losses

S1/S2 are two complete GCC series; S2 reverses execution order. Times are medians
of eight region-mean ns/operation values. Speedups are medians of eight SAME-ROUND
API/candidate ratios, not ratios of the displayed median times. Greater than one
favors the candidate. Raw rows have no actual-API baseline in their timing group.

| Operation / mode | Route | ns/op S1 / S2 | API/candidate S1 / S2 | Candidate wins /16 |
|---|---|---:|---:|---:|
| Multiply / chain1 | mul_api | 24.347 / 23.291 | — | — |
| Multiply / chain1 | mul_row_serial | 21.704 / 20.893 | 1.119451 / 1.107246 | 15 |
| Multiply / chain1 | mul_comba_serial | 34.510 / 32.715 | 0.709505 / 0.700328 | 0 |
| Multiply / chain1 | mul_comba_parallel | 36.000 / 34.981 | 0.675270 / 0.672205 | 0 |
| Multiply / ILP4 | mul_api | 12.233 / 11.946 | — | — |
| Multiply / ILP4 | mul_row_serial | 18.121 / 17.851 | 0.689388 / 0.656365 | 0 |
| Multiply / ILP4 | mul_comba_serial | 27.047 / 26.304 | 0.460148 / 0.449047 | 0 |
| Multiply / ILP4 | mul_comba_parallel | 27.229 / 26.143 | 0.450878 / 0.451050 | 0 |
| Square / chain1 | square_api | 21.531 / 23.354 | — | — |
| Square / chain1 | square_row_serial | 19.654 / 19.404 | 1.078533 / 1.182849 | 16 |
| Square / chain1 | square_comba_serial | 29.459 / 28.525 | 0.744741 / 0.807209 | 0 |
| Square / chain1 | square_comba_parallel | 29.020 / 27.945 | 0.749898 / 0.808567 | 0 |
| Square / ILP4 | square_api | 11.719 / 11.509 | — | — |
| Square / ILP4 | square_row_serial | 15.675 / 15.664 | 0.750572 / 0.722759 | 0 |
| Square / ILP4 | square_comba_serial | 28.687 / 28.951 | 0.403640 / 0.393675 | 0 |
| Square / ILP4 | square_comba_parallel | 29.451 / 30.596 | 0.396582 / 0.382207 | 0 |
| Raw reduce / chain1 | reduce_serial | 12.706 / 13.351 | — | — |
| Raw reduce / chain1 | reduce_parallel | 12.873 / 12.492 | — | — |
| Raw reduce / ILP4 | reduce_serial | 8.963 / 8.403 | — | — |
| Raw reduce / ILP4 | reduce_parallel | 8.803 / 8.775 | — | — |


All raw samples and min/max distributions remain in
[series1](data/p5_run1_20260905.json) and [series2](data/p5_run2_20260905.json).
[Independent numeric audit](data/p5_numeric_audit_20260905.json) replays integer
work/timestamps, durations, ordering, outputs, all64comparison summaries and
512same-round ratios:45,534checks, zero errors. Manager also reproduced65actual
disassembly slices/symbol extents, six normalized pairs and14phase windows
([424-check replay](data/p5_codegen_replay_20260905.json)).
No historical P2/P3/P4/F2 timing is pooled here.

### Reducer and dependency observations

Raw serial/precomputed ratios are1.002708 /1.040950 in chain1 (12/16precomputed
wins), but1.020062 /0.956821 in ILP4 (8/16). This is no robust distinct-reducer
winner. Five of six serial/precomputed complete-job pairs are normalized
instruction-identical in the frozen GCC binary; the raw chain pair differs only
in two XOR input-staging instructions. The reducer sequence itself matches.
See [static code evidence](P5_FIELD_PRODUCT_CODEGEN.md), including outlined
callees and the exact normalization/replay rules.

The actual API's chain/ILP normalized ratios are1.973046 /1.979415 for multiply
and1.841663 /2.005417 for square. Row+serial reaches only1.185583 /1.173694 and
1.231291 /1.267204 respectively. These are whole-job amortized comparisons, not
hardware thread, port-throughput or instruction-latency measurements. One-core
ILP4 is deliberately different from multicore execution.

The source row self-product expresses16multiplications, but this compiler reuses
symmetric products. Therefore the hand-written ten-product symmetric Comba does
not automatically remove six emitted products relative to this row-square
control. Public-column loop control and carry scheduling remain different.
No spill-cost, bandwidth or critical-path causality is inferred from timing.

## Qualification and retained deviations

[Arithmetic receipt](data/p5_validation_20260905.json): native GCC, independently
compiled Clang field sources with retained GAS object, and NO_ASM ASan+UBSan each
pass714,283assertions, zero mismatches, checksum75711b730275b15a. The same finite
corpus is reused:14,132raw-pair entries/14,100distinct,13,102canonical-pair entries/
13,073distinct,12,071direct-wide entries/12,058distinct. Those sets can overlap;
assertions and helper checks are not distinct fixtures. There are1,024derived
mul/square chain steps and110negative observer controls.

All512product bits and canonical raw residues/all32BEbytes are checked against
Boost ordinary integers; corrected FE64 is an additional reference on its valid
canonical domain. Actual phase checks include487q=K cases,504qK-above-u64 cases,
103second-fold-overflow cases and8final-p-correction cases. These counts overlap.
All-max products stress130-bit Comba columns and129-bit doubled cross products.
The written proof retains65-bit qK, proves third-fold high zero and one final
canonical correction; it is not machine-checked universal implementation proof.

[Separate synthetic duration tests](data/p5_duration_validation_20260905.json)
pass120cases/6,550assertions on GCC and Clang. Fake clocks never enter performance
binaries. An actual renamed-main failure returns2, retains41regions including
all13failure probes, and never reports complete. Invalid clocks, initial work
cap and malformed serializer lanes are tested. Within-region capped-final-batch
and work-cap termination remain static-reviewed, not dynamically exercised.

| Real duration evidence | S1 | S2 |
|---|---:|---:|
| Raw regions | 262 | 261 |
| Calibration records (short retained) | 62 (21) | 61 (21) |
| Warmup / measured records | 40 /160 | 40 /160 |
| Continuously extended later regions | 20 | 5 |
| Cumulative probes | 282 | 266 |
| Minimum later-region duration, ms | 206.721981 | 200.452804 |
| Later regions below200ms | 0 | 0 |

All400real later regions reach200ms;25short prefixes were extended within the
same clock, never restarted or discarded. All42short calibration attempts remain.
CPU endpoints are4throughout; this does not prove constant frequency or absence
of intervening interference. No owned builds/tests/other benchmarks ran during
full series. Powersave/turbo policy stayed unchanged; background services,
thermals and SMT sibling activity were uncontrolled. PMU remains unavailable.

Prefreeze driver review fixed five misleading-indentation warnings and an
unbounded serializer-lane warning by explicit admissible-lane checks and bounded
iteration. Four malformed-lane assertions were added. All affected builds and
smokes were rerun before any full measurement; no arithmetic kernel changed.
One manager smoke-command omission and one read-only Node quoting error were
invocation errors, not algorithm failures. New-file indexing lag and transient
refresh identity-slot failures were recorded; final five code hashes matched
indexed evidence before measurement. [Environment/obstacles](data/p5_environment_obstacles_20260905.json).

## Disposition and next field-only experiment

Retain row+serial for dependent multiplication/square; keep actual FE64 API for
the measured independent-state mode. Do not implement a production dispatcher
or claim an unmeasured crossover. Retain Comba/symmetric losses and compiler-
equivalent reducer controls as evidence that source operation counts and source
parallelism are insufficient selection criteria.

Next isolate the actual API/raw-ASM boundary with equal observation costs, then
consider explicitly unrolled/fused product-reduction schedules and compiler
replication. This will test whether the row-chain advantage transfers to a
usable primitive boundary; current data does not assign its cause. Inversion
and separately contracted batch inversion remain subsequent field work.
Scalar/point/signature investigation and combined integration remain deferred.
No CT approval, novel-mathematics/algorithm classification or world-record claim.

Reproduction identities and exact commands: [premeasurement freeze](data/p5_performance_freeze_20260905.json)
and [build/evidence manifest](data/p5_build_20260905.json). Review outcome:
[manager review](data/p5_final_review_20260905.json). P4's live ledger bytes were
archived before P5 in [the P4 frontier snapshot](data/p4_frontier_snapshot_20260905.md),
SHAdd09f228bf6b9cf9c9af6b57862eb4f288e1a40398ad0170ed3a00a925253522.
Original earlier reports/manifests remain unchanged. Owner Task MCP suspension
remains active; this is reviewed local research, not canonical task acceptance.
