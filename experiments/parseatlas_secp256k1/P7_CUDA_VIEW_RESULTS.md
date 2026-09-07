# P7 CUDA view side probe: measured outcome

Disposition: retain the view-only negative result and the existing hybrid
algorithm control. No production implementation is replaced. This side probe
does not reproduce the owner's private two-pointer chain and does not replace
the [primary field research plan](FIELD_REPRESENTATION_SEARCH_PLAN.md).

## Result

Changing only the Comba input loader from explicit shifts to legal memcpy
views did **not** yield a useful speedup on this measured profile. The paired
ratios are essentially one despite different intermediate PTX. Existing hybrid
field multiplication remains substantially faster than the existing all64
product-plus-reducer contrast; that is existing code, not a new discovery.

| Comparison (A/B; >1 favors B) | Threads | Series 1 | Reversed series 2 |
|---|---:|---:|---:|
| Clone shifts / clone memcpy | 32 | 1.00000135 | 1.00000018 |
| Clone shifts / clone memcpy | 32768 | 0.99999564 | 0.99999941 |
| Existing all64 / current hybrid | 32 | 1.15901938 | 1.15898060 |
| Existing all64 / current hybrid | 32768 | 1.52347068 | 1.51431902 |

All64/current hybrid favors the current hybrid in 32/32 same-round pairs.
Input-view controls are not a new winner. Current hybrid and its shift clone
have equal normalized PTX yet their large-grid paired ratios differ by up to
about 0.45% at series-median level; tiny timing differences are not causal proof.
Ratios compare eight same-round elapsed/work ratios, not ratios of medians.

The fixed [protocol](P7_CUDA_VIEW_PROTOCOL.md) distinguishes all four routes.
Route 3 changes both product and reduction, whereas routes 1/2 share identical
64-product Comba order and differ only in the loader. Field inputs/outputs are
canonical; each thread repeats x=x*b 128 times with fixed b. Counts 32/32768
have one active warp / a larger independent grid. Event-amortized ns/op is
neither an individual instruction latency nor CPU ILP4 performance.

## Qualification and reproducibility

Manager-built native C++ oracle: **78,521 assertions, zero mismatches**;
25,928 canonical outputs and 24,688 full 512-bit outputs; 10,545 CPU-reference
checks; 111 negative observer controls; 19 invalid eval/raw calls rejected.
Finite corpora contain 3,088 field entries (3,081 distinct) and 3,086 raw entries
(3,079 distinct), with 2,048 random pairs/domain and edge/carry cases.
17 chain lanes use nine selected prefixes through 256 steps, eight roundtrip
transitions, and a==b with fixed-original-RHS semantics. Same-input alias,
host guards and input preservation pass; host guards are not GPU instrumentation.

[Validation](data/p7_validation_20260906.json) retains commands, limitations and
the failed sanitizer attempts. Compute Sanitizer 2022.4.1 initially missed its
injection library; using the package-discovered path and then all-process mode
still terminated before the first instrumented API. **No GPU memory-sanitizer
pass or CT certification.** Benchmark invalid-argument and duration-failure
paths have static review only, not a new synthetic failure suite.

Both complete series passed without reruns: each has 104 raw event intervals:
24 calibration (8 short), 16 warmups and 64 measurements. All 160 later intervals
reach 200 ms; minima are 226.884476 / 243.060089 ms. No duration retries occurred.
All 208 intervals' returned output buffers matched the CPU reference. The 56-interval
[smoke](data/p7_smoke_20260906.json) is separate and performance-ineligible.

Raw records: [series 1](data/p7_run1_20260906.json),
[series 2](data/p7_run2_20260906.json);
[2,044 numerical checks](data/p7_numeric_audit_20260906.json), zero errors;
[independent disk replay](data/p7_numeric_replay_20260906.json) passed 761 checks.
Actual work is count * 128 * launches; checksum/ordering/duration/summary gates
passed. Fixed corpus identity: A c2f96be1403a6a21, B 6f56c26cf29ee800.
Read-only input reconstruction confirms all 65,536 inputs nonzero and not one.
The generic driver rejects zero but not one; this finite fixed-seed admission
does not assert a general no-one generator contract.

[13 frozen input identities](data/p7_performance_freeze_20260906.json) matched
[after timing](data/p7_postfreeze_20260906.json). Production CUDA/CPU arithmetic
and the corrected reference library remain unchanged. Workers stopped before
full timing. Known owned competing tests/builds were absent; external services,
display activity, clocks, thermals and background activity were not controlled.
GPU idle endpoint snapshots were P8/37C before and P8/39C after; these do not
describe clock behavior inside the intervals.

## Emitted code

[Codegen evidence](data/p7_codegen_20260906.json) extracts PTX from the exact
manager CUDA object and embeds a read-only replay program, hashes, all nine
entry summaries and complete field-view bodies. Normalization preserves
instructions, constants and non-kernel symbols while renaming registers/labels.
Current hybrid and the shift clone match for both field and raw-product entries.

Field shifts/memcpy have 366/346 static PTX instruction sites; raw-product
shifts/memcpy have 294/270. The memcpy view uses paired 32-bit loads and
mov.b64 splitting where the explicit version uses 64-bit loads, shifts and
conversions. These are static sites, not retired instructions or measured
physical traffic. Runtime field attributes are 44 registers, zero local bytes
for all four routes; raw attributes are 38/38/40/40 registers, zero local bytes.
No SASS equivalence, dynamic spill cost or bandwidth explanation is established.
The device endian sanity kernel folds to constant 1 in PTX; the full byte oracle
qualifies the observed compiled pipeline, not that kernel as a hardware probe.

Profile: RTX5060Ti, cc12.0; driver580.178.04 (driver API13000),
NVCC12.0.140 with GCC12 host and explicit unsupported-compiler admission,
compute89 PTX JIT to binary120; GCC14 C++20 benchmark/oracle host.
This is not a native Blackwell-toolkit or second-device/compiler replication.
NVIDIA describes this [PTX compatibility mechanism](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).
No clock/toolkit/driver installation or system setting was changed.

## The 25 lenses together

| Lens | Observed outcome / limit |
|---|---|
| V01 Reachability | Four intended product/reducer routes reached; current/clone normalized PTX agree |
| V02 Corpus | Frozen hashes, actual nonzero/non-one input admission; qualification corpus separate |
| V03 Equivalence | Independent Boost and corrected CPU bytes pass for all four routes |
| V04 Definedness | Real uint32_t arrays plus memcpy; no incompatible integer-pointer dereference |
| V05 Range / overflow | 64 terms and 16 output columns preserved; 67-bit column fits 96-bit accumulator |
| V06 Aliasing | Read-only a==b and fixed RHS tested; host output overlap statically reviewed only |
| V07 Observable output | All final field/raw bytes checked; each field step canonical by existing reducer |
| V08 Invariants / carries | Bit/limb boundaries, P1 witnesses and selected chain prefixes pass |
| V09 Constant time | Unqualified; all64 reducer contains a carry-dependent branch |
| V10 Resources | Runtime field registers44/local0 for every route; not physical traffic measurement |
| V11 Property visibility | A source representation change changes PTX but yields no useful timing gain here |
| V12 Dependency depth | Comba pair order unchanged; no measured critical-path decomposition |
| V13 Live state | Different virtual PTX registers; same reported physical field register count |
| V14 Operations | Same 64 Comba terms; all64 contrast also changes reducer and loop lowering |
| V15 Conversion | Explicit shifts/conversions versus paired loads/register splitting in PTX |
| V16 Latency | Small-grid event average; not isolated field-call or instruction latency |
| V17 Throughput | Existing hybrid beats all64 about1.51–1.52x for larger grid |
| V18 Instructions / PMU | Static PTX only; no runtime SASS or retired-counter measurement |
| V19 Cache / bandwidth | Logical data is unchanged; no measured bandwidth reduction |
| V20 Code size | Fewer PTX sites do not establish smaller hot runtime SASS footprint |
| V21 Portability | One old-toolkit/JIT/GPU profile; CPU transfer remains unmeasured |
| V22 Compiler | Loader changes survive to PTX; where later differences disappear is unverified |
| V23 Policy / ties | Two balanced series; all160later>=200ms; view result practically tied |
| V24 Transfer | Keep GPU side probe bounded; resume arithmetic-structure CPU field work |
| V25 Prior art / novelty | Existing Comba/hybrid identity; no novel mathematics/world-record claim |

## Next primary work

Follow the [representation-search plan](FIELD_REPRESENTATION_SEARCH_PLAN.md):
compare fused product/pseudo-Mersenne folding against the frozen row and direct
ASM controls, with full carry bounds. Keep final-correction scheduling a
separate controlled wave. Continue collecting positive and negative field
results before scalar-n and upper-layer integration.

NeedFix workflow evidence: worker-role MCP exposure remains absent under the
owner-suspended Task flow. Source Graph refresh intermittently reported
identity_slot_owned, while final per-file indexed hashes were fresh and matched.
No manager-role impersonation, canonical task mutations or synthetic HMAC receipt.
