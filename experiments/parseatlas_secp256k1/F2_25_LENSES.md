# F2: 25-lens delta ledger

Status: the manager reports passing correctness and numerical-replay gates.
Native C++ timing is complete across two series, 12 invocations, and 384
measured samples; independent audit and final report review are complete.
This is a ledger of one F2 transformation family viewed through 25 lenses,
not 25 implemented precedence policies, 625 completed policy/lens cells, or a
production acceptance decision. Measured comparisons are recorded in
[F2_REDUCTION_RESULTS.md](F2_REDUCTION_RESULTS.md), independently reviewed;
numerical timings are intentionally not duplicated here.

## Frozen contract and representation

Source: [f2_reduce.hpp](probes/f2_reduce.hpp), SHA-256
`57ede13d3f0df8b9c12f8578c60e533b416c23da670671d19ec3d731b860c651`.
The unchanged modular prerequisite is [modn_add_control.hpp](probes/modn_add_control.hpp),
SHA-256 `0858f280f415d247e569e652e49a6635c999e80c3f7d89494a715c3585daa04f`.

For canonical scalar-order inputs and public `0 <= N <= 1,000,000,000`, F2
represents the exact integer `T = x0 + sum(rhs[0..N))`, then returns the canonical
32-byte value of `T mod n`. Here `n` is the group order, not coordinate prime `p`.
`x0` occurs once. Empty input returns `x0`; input arrays are unchanged.

- `Wide5` decodes as `sum(wide[i] * 2^(64*i), i=0..4)`. It propagates low-word
  carries per input, retains the high word, and delays modular reduction.
- `Columns` decodes as `sum(column[i] * 2^(64*i), i=0..3)`. Its four 128-bit
  columns are components of **one integer**, not four unrelated recurrences.
  Cross-column carries are propagated once, then the same `reduce320` runs.

Transformation labels are **limb layout**, **lazy reduction**, and
**reassociation**. Column accumulation groups same-significance integer
contributions and delays cross-column carry normalization; this explanation
is not an additional transformation label. This is not generically a carry-save
representation or a claim of discovering new mathematics. Neither form promises
all recurrence prefixes or eligibility for secret-dependent constant-time
operations.

## Lens-by-lens observations

| Lens | F2 evidence / boundary | Unresolved gate |
| --- | --- | --- |
| V01 reachability | Final-only sums are the declared workload. No production caller or dispatch was changed. | Actual caller eligibility and workload frequency: PENDING IMPORT. |
| V02 corpus | Manager reports five native C++ configurations passing the same corpus: 101,445 reducer cases, 10,153 column cases, and 2,276 arrays, with zero mismatches. | These are not five independent corpora or multiplied unique fixtures; final artifact binding belongs to the results report. |
| V03 equivalence | Intermediate decoding checks exact unreduced `T`; final comparisons use full canonical bytes, an independent Boost oracle, and unchanged public `Scalar`. | Finite tests are not an exhaustive proof over all arrays. |
| V04 definedness | Count/addressability and positive-count null checks precede reads. Oversized columns are rejected before normalization. Canonical values and valid buffer extents remain unchecked preconditions. | No claim of validating arbitrary pointer lifetimes or invalid canonical inputs. |
| V05 rangeoverflow | With `K=N+1`, `T <= K*(2^256-1) < 2^286`; each column is below `2^94`. `reduce320` separately accepts every 320-bit input and retains the first fold carry. | Source bounds plus finite gates, not a machine-checked proof. |
| V06 alias | All inputs are const and results are returned by value. Reading `x0` from the RHS allocation is compatible with these nonmutating APIs. | No mutable overlapping-output or in-place production API was introduced. |
| V07 observable | Only the final canonical scalar and exposed exact intermediate decodes are required; array bytes remain unchanged. | Prefix outputs, side-channel behavior, and complete engine observables are outside this contract. |
| V08 invariants | `x0` is included once; accumulation is exact; column normalization preserves its represented integer; final output lies in `[0,n)`. | Full-job handoff must preserve these invariants, not merely a checksum. |
| V09 CT | F2 uses conditional reduction and inherits frozen carry-helper configuration choices; byte equality does not establish constant time. | CT eligibility: PENDING IMPORT; no secret-operation replacement is authorized by these gates alone. |
| V10 resources | F2 uses one caller thread and no heap allocation. Wide5 and Columns payloads are 40 and 64 bytes respectively. | Stack usage, registers, spills, CPU time, energy, and system resource effects are distinct measurements. |
| V11 propertyvisibility | Exact Wide5 and column APIs expose carry retention and delayed normalization separately from final modular equality. | Passing only final-byte checks would not make hidden intermediate errors acceptable. |
| V12 depth | Wide5 retains a per-input four-limb carry chain. Columns separate four logical component chains and defer cross-column carries until the boundary. | Numeric critical-path depth and machine scheduling: PENDING IMPORT. |
| V13 livestate | The representations have five 64-bit limbs or four 128-bit columns, plus boundary temporaries. These type payloads are not register counts. | Peak live registers, lifetime overlap, and total spills: PENDING IMPORT. |
| V14 operations | Wide5 performs four frozen `add64` calls plus one high-word accumulation per RHS. Columns performs four 128-bit additions per RHS, then four normalization steps. Both call `reduce320` once for nonempty input. | These are source-level operations, not instruction, micro-op, cycle, or executed-subtraction counts. |
| V15 conversion | Initial state conversion, accumulation, column bound checks/normalization, fold reduction, and final canonicalization belong inside each complete job. | No free reusable preprocessing or excluded boundary conversion is assumed. |
| V16 latency | Two native C++ series measure normalized region means per complete job; per-size comparisons are in the results report. These are not individually timed job latencies. | Individual-job latency distributions: PENDING IMPORT. |
| V17 throughput | Complete-job and per-RHS throughput comparisons are measured under the same final-only contract, with repetition and one-core/multicore resource conditions disclosed in the results report. | Consult the independently reviewed measured report; throughput does not establish individual-job latency or universal superiority. |
| V18 instructions | Manager-observed GCC binary SHA-256 prefix `b06a6d46...`: Columns loop `0xb400-0xb426` has four `add/adc` pairs and no stack-memory accesses; Wide5 loop `0xc330-0xc3a2` has stack writes. | This exact-binary static observation is not PMU evidence or a demonstrated cause of speedup. Full binary binding belongs to the results report. |
| V19 cache | Smaller or differently placed live state may affect locality, but type sizes and a stack write do not quantify cache traffic. | Cache misses, bandwidth, residency, and causal cache attribution: PENDING IMPORT. |
| V20 codesize | Root `nm` counts for the measured GCC hot full-job wrappers: Serial 514 bytes (`0x202`), Wide5 848 (`0x350`), Columns 919 (`0x397`), ColdMulti 112 (`0x70`); its separate cold function is 3,883 bytes (`0xf2b`). | These exclude cold clones, other helpers, rodata, and shared code; they are not total instruction-cache cost. Cache effects: PENDING IMPORT. |
| V21 portability | F2 requires GCC/Clang-compatible unsigned 128-bit integers. `SECP256K1_NO_INT128` changes frozen carry helpers, including Wide5 and the optional reducer fold; it does not remove new columns/products. | Other architectures/toolchains are not established by this extension-based implementation. |
| V22 compiler | Native configuration passes check correctness on the shared corpus. The quoted assembly belongs to one exact GCC binary only. | Cross-compiler performance and code-generation stability: PENDING IMPORT. |
| V23 policyties | Candidate choice must be scenario-specific: public `N`, available resources, required outputs, and CT requirements constrain eligibility before timing comparisons. | No production dispatch thresholds, universal winner, or secret-value-dependent policy selection was set. |
| V24 transfer | Final-only scalar sums may admit F2; prefix-producing APIs and coordinate-field arithmetic require different contracts. | End-to-end engine benefit, real caller validation, and production integration: PENDING IMPORT. |
| V25 priorart | Current labels describe the representation and measured obligations, not historical novelty. | Prior-art comparison and any novelty assessment: PENDING IMPORT. |

## Portfolio rule and review boundary

Retain the serial recurrence, delayed-normalization representations, and any
multicore control as scenario candidates until comparable complete-job results
are reviewed. A tiny job may favor a different method than a large stream;
using more cores also changes the resource budget. Eligibility and any future
dispatch must depend only on public workload information, available resources,
and security requirements—not secret scalar values. No production threshold
or automatic replacement has been installed.

This document stops at Codex review. Measured native C++ comparisons are in
the independently reviewed results report. Individual-job latency, cross-compiler
performance, PMU/cache attribution, peak live state, end-to-end transfer, and
historical novelty remain **PENDING IMPORT**, not inferred from correctness,
arithmetic counts, normalized region means, or static assembly alone.
