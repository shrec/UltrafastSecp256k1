# P5 field products — preregistered C++ scheduling controls

Status: fixed before P5 performance measurement, 2026-09-05. Field-only
collection continues; no production integration. P1–P4 evidence is unchanged.

## Question and contract

Can a changed product, square or sparse-prime reduction schedule improve the
complete canonical primitive on this machine? The corrected FE64 API, including
its BMI2/ADX implementation, is the primary reference. A faster isolated reducer
does not establish a faster field multiply. Historical P2 timings are context,
never samples in P5 comparisons.

Let B=2^256, K=2^32+977, p=B-K. Experimental full-product functions accept any
two 256-bit unsigned values and return all 512 product bits. Experimental
reducers accept any 512-bit unsigned value and return canonical [0,p) limbs.
Complete field multiplication/squaring uses canonical operands. Every step is
canonical; the final-only sum headroom from P3/P4 is not imported.

See the separately authored [bound contract](P5_FIELD_PRODUCT_CONTRACT.md) and
[25-lens verification map](P5_FIELD_PRODUCT_LENSES.md). The map distinguishes
measured, derived, static and unperformed observations, not 25 discoveries.

## Fixed matrix

| Operation | Four or two routes | Modes |
|---|---|---|
| Multiply | actual FE64 API; row product + serial reducer; Comba product + serial reducer; Comba product + precomputed reducer | chain1, ILP4 |
| Square | actual FE64 API; row self-product + serial reducer; symmetric Comba + serial reducer; symmetric Comba + precomputed reducer | chain1, ILP4 |
| Raw 512-bit reduction | serial reducer; four precomputed high-word products reducer | chain1, ILP4 |

Total: 20 cells. Row multiplication uses safe per-cell u128 accumulation;
Comba uses three 64-bit accumulator words, not a potentially overflowing sum
of four products in u128. Symmetric square computes ten distinct products and
adds each cross-product twice; a doubled product may require 129 bits.
Both reducers retain q*K in u128 and the subsequent overflow bit.

Each job starts from four deterministic canonical nonzero seeds and performs
1024 steps per active lane. Multiplication cycles a shared 256-value canonical
nonzero RHS corpus. Square evolves x -> x^2 without an RHS. Raw reduction
cycles a 256-value arbitrary-512 corpus: input low words are x XOR corpus-low,
high words are corpus-high. This creates actual dependence while preserving
the arbitrary-512 domain. ILP4 runs four independent states on one core.

Only one core is used deliberately to compare dependency schedules without
thread dispatch; this is not a multicore-optimum experiment. All active output
lanes remain observable. Allocation, corpus generation and prevalidation are
outside timing; seed initialization, all 1024 steps/lane and full final output
materialization are inside each job. No per-step benchmark barriers. Job
noinline/noipa plus a whole-output barrier must prevent repeated-job erasure.

## Execution and duration policy

GCC14 C++20, -O3 -march=native -fno-lto -DNDEBUG, native ASM/fast-reduction
macros, corrected frozen P1 library. CPU4, existing powersave/turbo policy
unchanged. Record compiler, binary and source hashes before full series.

Two complete series, second order reversed; seeded permutations and round
rotation/reversal, two warmup and eight measured rounds per cell. Each later
region must last at least 200 ms. Calibration requires two consecutive
qualifying regions at the same job count. Keep every attempt.

Use P4's continuous-region method: append calibrated-size batches within the
same clock if a later region is short, recording each cumulative elapsed/jobs
probe. Never discard a short prefix or restart the clock. Actual total work
determines ns/operation. A finite extension/work limit explicitly fails the
series and retains failed raw records. Smoke and synthetic-clock tests are
validation-only; they are not performance observations. Output files must not
overwrite existing data.

No owned builds, tests or other benchmarks during full series. Background
services, SMT sibling activity, thermals and frequency remain uncontrolled.
CPU endpoints are recorded but do not prove no intervening migration. Existing
denied PMU access is a limitation; no security or power-policy changes.

## Preregistered comparisons and interpretation

Primary: for multiply and square, compare each of three candidates with the
actual API in each mode and each series. Report median of eight same-round
API/candidate ratios, every win/loss and per-operation time distributions.
Raw reduction compares serial/precomputed. Ratios above one favor denominator.

Secondary: row+serial / Comba+serial (product schedule); Comba+serial /
Comba+precomputed (reducer schedule), separately for each operation and mode.
Compare chain/ILP ns/op as whole-job amortized costs, not pure port throughput
or instruction latency. These are comparisons of complete compiler-generated
programs, not perfect causal ablations of a single instruction or memory cost.

Inspect emitted job/kernel symbols for actual multiplies, carry chains, calls,
stack operands and full output stores. Source-level operation counts alone do
not establish compiler instruction counts, cycles, bandwidth, spill cost,
constant time, novelty or a whole-engine speedup. Retain negative results.

## Independent qualification before performance

C++ Boost ordinary integers check exact 512-bit products, arbitrary-512 modulo
p and composed canonical results. Compare corrected FE64 mul/square raw limbs
before serialization and all 32 big-endian bytes. Include zero/one/p-1, all-max
limbs, carry patterns, p and p^2 boundaries, high/low-only wide values, both P1
regression KATs, deterministic random inputs, same-object operands and input
preservation. Trace bounded dependency chains at every step independently.

Exercise q=K, q*K beyond 64 bits, second-fold overflow, final correction, the
130-bit Comba column and 129-bit doubled cross-product. Count fixtures and
assertions separately. Run native GCC, independently compiled Clang field
sources, and NO_ASM ASan+UBSan halt-on-error. Separately validate duration
extension/failure behavior using synthetic clocks, never as performance data.

Workers author disjoint new files. Manager runs mechanical gates, then reads
the complete code and contract; any correction invalidates affected gates.
Freeze performance sources before timing. Record failures and rework.

## Authority and handoff

Verified manager repo_id repo_666797171f0141c58bf05f579b2ee16e and session
01a06be6-2904-7c62-9e7d-1245c34a5312. Bootstrap and Task health agree. Owner's
Task MCP suspension remains; no canonical task acceptance/launch is implied.
Worker-role Source Graph tools remain unexposed (recorded NeedFix); manager
supplies exact verified targets, performs continuing graph queries and final
freshness review. Newly declared files may await indexing; record that lag.
No worker manager-role impersonation or context-database writes.

No production, scalar, point, signature, default build/CI edits, commits,
pushes or tags. Preserve earlier dirty correctness repairs. The exact pre-P5
frontier is archived in [P4 snapshot](data/p4_frontier_snapshot_20260905.md);
historical manifests keep their original hashes. Stop at manager review.
