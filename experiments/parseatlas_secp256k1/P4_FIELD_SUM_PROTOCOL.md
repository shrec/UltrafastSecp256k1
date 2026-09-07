# P4 field sums — preregistered native 4x64 controls

Status: design before P4 performance measurement. Continue field-only collection;
no production integration. P3 code, protocol and raw evidence remain unchanged.

## Observation and question

Return canonical FE64 and all 32 big-endian bytes of
(x0 + sum(N canonical RHS)) mod p. N counts RHS and x0 occurs once globally.
N=0/null is the identity; positive N requires a valid canonical input range.
Inputs are read-only; same-type x0/RHS alias is allowed. Only the final output
is observable, not every prefix. No new output-buffer alias contract.

P3 established a strong final-only FE52 vector-sum candidate against the actual
corrected FE64 returning-add API. P4 adds native 64-bit-radix controls to separate
implementation-boundary, normalization schedule, accumulator-bank and input
representation choices. These are controlled comparisons of complete generated
programs, not perfect causal ablations of one machine cost.

## Fixed matrix

| Route ID | Route | RHS storage | RHS per chunk | Independent RHS banks |
|---:|---|---|---:|---:|
| 0 | Existing FE64 API serial sum | FE64 | 1 | 1 |
| 1 | Inline eager native 4x64 | FE64 | 1 | 1 |
| 2 | Native wide16 lane1 | FE64 | 16 | 1 |
| 3 | Native wide4094 lane1 | FE64 | 4094 | 1 |
| 4 | Native wide16 lane4 | FE64 | 16 | 4 |
| 5 | Native wide4094 lane4 | FE64 | 4094 | 4 |
| 6 | P3 FE52 E2E Full16 | FE64 packed at use | 16 | Compiler-selected |
| 7 | P3 FE52 E2E Full4094 | FE64 packed at use | 4094 | Compiler-selected |

Native wide accumulators have four unsigned 128-bit columns per bank, with
radix 2^64 input limbs. Banks accumulate RHS independently, then merge with
the canonical prefix seed exactly once per chunk. Inter-column carries and
p-reduction happen at boundaries. No input representation conversion is needed.
Allowed native template chunk bounds are 1..4095 RHS, with banks exactly 1 or 4.
Each chunk budgets its canonical prefix as one additional term, hence M<=4096.
These are deliberately conservative admitted bounds, not maximal u128 capacity.

Inline eager keeps native raw limbs and canonicalizes each RHS addition, then
constructs the returned FE. It avoids the library per-add API boundary. Changes
to calling convention, optimizer visibility and accumulator materialization
remain part of this comparison; it is not an isolated call-overhead measurement.

FE52 controls use the unchanged P3 header and include all on-use packing.
All eight routes read the same FE64 array (32N bytes). No resident FE52 input
array is needed in P4. Full 32-byte final output is materialized per job.
Allocation, corpus generation and untimed validation are outside measured jobs.

## Workloads and execution

- N = 1,16,256,4096,65536,1048576: 8 routes x 6 sizes = 48 cells.
- Preserve P3 canonical FE64 generator, seed 20260905 and actual shared prefixes.
  Preserve x0 and the complete FE64 input checksum a228d8a510693ab4.
  Bind source recipe by exact source SHA in the build manifest; input arrays
  are reconstructible from the source and seed, not individually archived.
- GCC14, C++20, -O3 -march=native, LTO OFF, corrected frozen P1 library.
  Native ASM/fast reduction macros match P3. CPU4, existing power/turbo policy.
- Two complete series; reverse the second series order. Seeded size/route
  permutations and round rotations/reversals. Two warmups and eight measured
  rounds per cell. Retain every calibration/warmup/measurement record.
- One core is intentional: this wave measures carry dependencies and compiler
  instruction scheduling without inter-thread dispatch costs. It is not a
  measurement of the multicore optimum; public independent partitions remain
  a later design opportunity.
- No owned builds, correctness tests or other benchmarks during full series.
  Record power/CPU endpoints without changing settings. Background services,
  thermals and SMT sibling activity are not controlled.
- No pooling of P3/P2/P1/F2 timing records into P4 ratios.

## Duration qualification, including later phases

P3 retained a cell whose two qualifying calibration regions were followed by
shorter measured regions. P4 must qualify each complete later-phase region.

Calibrate the initial batch size using two consecutive qualifying same-count
regions, retaining all attempts. For each warmup/measured record, start one
continuous wall-clock region, run its initial batch, and inspect elapsed time.
If it is below the requested 200 ms, append another batch of jobs INSIDE THAT
SAME continuous region and repeat until the duration is reached. Never discard
a short prefix or restart its clock. Record every cumulative elapsed/jobs probe,
the actual total jobs, and whether an extension occurred. Clock-check overhead
inside the continuous region is timed and disclosed. A public work cap and
finite extension limit must fail the series explicitly rather than report a
short region as duration-qualified; preserve evidence on failure.

Use actual jobs for ns/sum and ns/RHS, not the initial calibrated count.
Smoke runs use a shorter requested floor and are validation-only. No selective
rerun, sample deletion, or best-attempt selection. Job noinline/noipa and full
output observability must prevent hoisting/erasure, without per-RHS barriers.
Final-region result checks and input preservation checks are outside the clock.

## Preregistered comparisons

Primary: all seven candidates against route0, using median eight SAME-ROUND
baseline/candidate ratios for each N and each series. Also report full ns/sum
and ns/RHS distributions and wins, including negative results.

Secondary same-round comparisons, fixed before timing:
- route0 / route1: library API versus inline eager implementation.
- route1 / routes2..5: eager versus deferred native schedules.
- route2 / route4 and route3 / route5: one versus four native banks.
- route2 / route3 and route4 / route5: chunk16 versus chunk4094.
- route2 / route6 and route3 / route7: same-chunk native lane1 versus FE52.
- route4 / route6 and route5 / route7: same-chunk native lane4 versus FE52.

For secondary A/B ratios above one favors B; name both operands explicitly.
Do not infer memory bandwidth, port pressure, spill cost or critical-path cycles
from speed alone. Inspect emitted code for actual scalar/vector loops, packing,
calls, stack accesses and output stores. Existing denied PMU access is not
permission to change security settings. No CT certification or novelty claim.

## Correctness before performance review

Independent C++ Boost ordinary-integer modulo-p oracle plus corrected FE64
reference. Compare canonical raw limbs and all32bytes, preserve every input
byte, and test nonzero seeds, aliases, null identities, zero residues, limb
carry patterns, p-1, cancellation, P2 witness, random inputs and chunk/lane tails.
Test chunk1 and maximum4095 in addition to timed16/4094, lanes1/4. Exercise
boundaries around4,16,4094,4095,4096 and multiple chunks plus65536/1048576.

Replay actual narrow phase helpers separately on a bounded corpus to check
u128 columns, bank merge, propagated highword, two folds and final canonical
state against ordinary integers. Identify any uninstrumented large cases and
avoid counting repeated compiler runs as extra distinct fixtures. Reject
chunk0, chunk4096 and lanes2 at compile time.

Manager runs native GCC, independently compiled Clang field sources, and
NO_ASM ASan+UBSan halt-on-error. Workers author code in disjoint new files;
manager runs mechanical gates before complete code/contract review. Keep all
failures and corrections in evidence. No code modifications after performance
source freeze without invalidating and redoing the affected validation.

## Deliverables and boundary

P4 kernel, independent C++ oracle, benchmark driver, independent bound argument,
25-lens map, raw series, compiler/source manifest and explicit manager review.
Earlier P3 manifest's mutable frontier document must be archived byte-for-byte
before this wave updates the live collection ledger.

No production/scalar/upper-layer/default build or CI changes, commits, pushes
or tags. Owner Task MCP suspension remains active; this is bounded local
manager/worker research, not canonical task acceptance.
