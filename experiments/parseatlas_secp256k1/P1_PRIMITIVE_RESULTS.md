# P1 primitive baseline and correctness repairs — 2026-09-05

Status: two existing field arithmetic defects corrected and independently
reviewed; first native fast FE64/Scalar baseline complete. No new optimization
candidate or upper-layer speedup is accepted.

## What changed

The independent Boost integer oracle found failures before any performance run.
The original implementation remains useful as a differential reference, but is
not a sufficient correctness oracle by itself.

1. GAS final reduction used `K & carry`, with carry equal to 0 or 1. That adds
   0 or 1, not 0 or K, for K=2^32+977. Three sites were affected:
   `reduce_4_asm`, `field_mul_full_asm`, `field_sqr_full_asm`.
   Each now expands the bit to an all-bits mask using NEG.
2. Portable first-fold reduction could wrap limb 3 without carrying into limb 4.
   Both native-int128 and manual-word branches now propagate through all
   remaining limbs. The existing second-fold logic was not rewritten.

The original failing cases are saved, including exact binary/source hashes, in
[data/p1_correctness_needfix_20260905.json](data/p1_correctness_needfix_20260905.json).

| Failure | Input | Correct result | Observed old result |
|---|---|---|---|
| Native square | a=2^256−2^33−1 | low word `fffff860000e8900`, other words zero | low word `fffff85f000e8530`, other words zero |
| Portable multiply | a=2^255+1, b=p−1 | p−a | p−a−K |

The native error is exactly K−1. The portable error is exactly K.
Both observed C++ results match the independent source-level diagnosis.
Downstream protocol impact was not exhaustively characterized.

Bound argument: let B=2^256 and T=L+BH<B^2. Folding gives L+KH<(K+1)B,
so five limbs retain the first folded integer. Its high coefficient is at most
K; the next fold is below B+K^2 and has at most one overflow bit. If that bit is
one, its retained low part is below K^2, and adding K cannot overflow B again.
A carry lost before this fold cannot be repaired by later normalization.

## Correctness and independent review

- GCC actual CMake Release assembly library: PASS.
- Independently compiled Clang assembly path: PASS.
- GCC portable NO_ASM with ASan+UBSan, halt-on-error: PASS.
- GCC NO_ASM + NO_INT128 manual-word/fallback path: PASS.
- New audit regressions against the frozen original native library: 25/45 pass,
  20 fail. Against the corrected library: 45/45 pass.
- Manager ran mechanical gates first, then reviewed the driver, test oracle and
  entire production/regression diff. A second bounded read-only review checked
  timing logic and independently replayed the persisted numeric evidence.

Each of the four configurations repeats the SAME deterministic finite corpus,
not four distinct corpora. Per domain: 16,198 binary pairs, 3,030 nonzero inverse
inputs, all 256 bit positions and neighbours, modulus/carry/borrow boundaries,
aliases, input preservation and 37 corrupted-observation controls. FE64 also
checks square/inverse in-place methods. Full raw canonical limbs and all 32 BE
bytes are checked; field equality alone would normalize away some mistakes.

Each configuration checks 258,125 field outputs and 238,897 scalar outputs:
497,022 total, with 1,471,940 assertions. Checksums are identical across all four:
field `b1e698d8ff7668ff`, scalar `f6e30294a30c1fa3`.
This is finite correctness evidence, not a universal proof or CT certification.
Clang emitted two pre-existing unused-function warnings in field/scalar sources.

Full records: [validation](data/p1_validation_20260905.json).

## Native primitive measurements

Values are medians of eight normalized region means, ns/op; each entry is
series 1 / series 2. Smaller means less time per arithmetic operation in the
specified workload. ILP4 means four interleaved independent states on ONE core,
not four threads. It is a different workload, not a replacement algorithm.

| Domain | Operation | Dependent chain1, S1 / S2 | Independent ILP4, S1 / S2 |
|---|---|---:|---:|
| Fp | add | 9.54 / 9.47 | 5.77 / 5.82 |
| Fp | sub | 13.40 / 13.74 | 5.92 / 5.82 |
| Fp | mul | 23.15 / 22.80 | 11.76 / 11.82 |
| Fp | square | 20.77 / 20.75 | 11.06 / 11.27 |
| Fp | inverse + lookup | 932.92 / 918.16 | 896.65 / 900.46 |
| Scalar n | add | 11.13 / 11.08 | 7.81 / 8.09 |
| Scalar n | sub | 9.77 / 10.30 | 5.83 / 5.88 |
| Scalar n | mul | 31.55 / 31.93 | 24.77 / 24.73 |
| Scalar n | square | 32.34 / 32.04 | 23.84 / 23.84 |
| Scalar n | inverse + lookup | 926.93 / 928.95 | 912.68 / 924.23 |

The corrected baseline uses the actual CPU CMake library, GCC14.2.0 Release,
native x86 assembly/fast reduction enabled, explicitly LTO OFF. It includes
existing public returning-API ABI and runtime dispatch costs. This is not a
claim about the default LTO-ON build, resident FE52, CT primitives, or another CPU.

Timing includes arithmetic, assignments, RHS/index loads and loop control,
plus amortized per-job reset, function call and final 32-byte serialization per
lane. Each job has 1,024 steps/lane. Each region repeats identical jobs.
Every job materializes full outputs; only the final identical job in a region
is byte-compared outside timing. Separate lane-major replay checks harness
scheduling; the separate Boost test provides the arithmetic oracle.

Inverse uses result-dependent lookup in a fixed 256-value nonzero corpus.
It avoids the two-value inverse(inverse(x)) recurrence, but includes address/
load cost and is NOT a pure inverse-call latency measurement. ILP4 also has
additional input trajectories; ratios between modes are descriptive workload
comparisons, not same-input causal isolation.

Two complete series use the same 520 canonical input values and reversed
calibration/round ordering. Each series contains 60 calibration, 40 warmup and
160 measured regions. Qualification requires two consecutive >=200ms regions
at the same job count. All 520 raw regions are retained, including the 40 short
initial calibration regions. No warmup or measured region was below 200ms.
Measured durations: S1 206.897479–345.234326ms; S2 204.794030–343.413639ms.

All region CPU endpoints were CPU4. i5-14400F, 16 logical / 10 physical cores;
powersave governor, turbo enabled. Endpoint frequencies ranged from roughly
1.12 to 4.60GHz and are NOT average measurement frequency. No system power
setting was changed. External services, temperature and scheduler interference
remain uncontrolled. This profile differs from F2; absolute times are not
compared across them. Largest median drift between the two series was scalar
sub chain1 (+5.38%); scalar add ILP4 was +3.50%; others <=2.56% in magnitude.

Raw data: [series 1](data/p1_run1_20260905.json),
[series 2](data/p1_run2_20260905.json).
The [smoke run](data/p1_smoke_20260905.json) is validation only, not pooled
into the published numbers.

## What the 25 lenses show now

| Lens | Current evidence / limit |
|---|---|
| V01 Reachability | Actual CMake assembly route and returning FE64/Scalar APIs; no FE52/CT substitution |
| V02 Corpus | Boundary + random correctness corpus; exact 520 timing inputs retained |
| V03 Equivalence | Independent oracle exposed two old failures; corrected four-configuration results agree |
| V04 Definedness | Field zero inverse throws; scalar zero inverse returns zero; portable sanitizers pass |
| V05 Range | First-fold five-limb bound and final one-bit correction established above |
| V06 Aliasing | Compound/self-alias cases and direct native ASM aliases tested |
| V07 Observable output | All raw limbs canonical, all 32 BE bytes checked |
| V08 Invariants | B≡K mod p explains both failures and their exact deltas |
| V09 CT | Fast-path scope only; no new branch in GAS correction; no new CT assurance |
| V10 Resources | Existing representations retained; peak allocation/stack telemetry PENDING IMPORT |
| V11 Property visibility | A carry bit and a mask are distinct representations; both must preserve decoded meaning |
| V12 Dependency depth | Chain/ILP timings separated; exact dependency-graph depth PENDING IMPORT |
| V13 Live state | Register pressure/spill attribution PENDING IMPORT |
| V14 Operations | Exact arithmetic counts replayed; symbolic multiply counts do not determine speed |
| V15 Conversion | End-of-job canonical byte conversion amortized; FE52 bridge cost not yet measured |
| V16 Latency | Chain1 normalized region means, with inverse lookup caveat |
| V17 Throughput | ILP4 measured on the same core; no multicore/upper-layer gain asserted |
| V18 Instructions | Corrected object disassembly shows NEG/AND/ADD at all three sites; no PMU claim |
| V19 Cache | Fixed small input corpus; cache misses/bandwidth effects PENDING IMPORT |
| V20 Code size | GAS reducer256B, fullmul610B, fullsquare520B, each +3B for NEG; not total linked hot-path size |
| V21 Portability | GCC, Clang, NO_ASM sanitizer and NO_INT128 host configurations; no other-architecture claim |
| V22 Compiler | Exact original/corrected build commands, source/object/library hashes retained |
| V23 Policy / ties | All short samples retained; future dispatch may depend on public workload, not secrets |
| V24 Transfer | Upper-layer work remains paused; no proportional speedup assumption |
| V25 Prior art | These are correctness repairs and baseline observations, not new mathematics or a world-speed record |

Disassembly: [native object](data/p1_asm_disassembly_20260905.txt).
The three size deltas follow from the three 3-byte NEG additions, not a
measured improvement in instruction-cache behavior.

## Next gate

Stay within primitives. First establish separately labeled FE52 and CT
baselines; then compare symmetric scalar square and full canonical FE52
multiply/square bridges against this corrected reference in paired same-build
experiments. Keep the full reduction/normalization cost, byte equality and
security contract. The scalar API currently computes square with general a*a;
10 unique products versus16 is a candidate rationale, NOT an observed gain.
Do not repeat the unchanged previously losing modular-n add candidates.

Roadmap and lens requirements:
[P1_PRIMITIVE_ROADMAP.md](P1_PRIMITIVE_ROADMAP.md).

## Provenance and boundaries

Source HEAD `fef231d4e4173bd016fb2a3a1eff67087396a203`,
`experiment/representation-search`, plus only the recorded field.cpp,
GAS and existing audit regression edits. Existing .gitignore and earlier
research artifacts were preserved. No commits, pushes, release tags, default
build/CI edits or upper-layer implementation changes were made.

[Original build](data/p1_build_20260905.json),
[corrected build, complete diff and binary hashes](data/p1_corrected_build_20260905.json).
Original binaries/library remain in /tmp/parseatlas-p1-primitives.1hlZNv;
temporary paths may be removed by the host later, so commands, diffs, inputs
and hashes are also durably recorded in the experiment directory.

Task MCP remains suspended at the owner's request; no canonical task was
claimed/launched/accepted. Initial session01a06be6-2904-7c62-9e7d-1245c34a5312,
then re-verified route01a072e3-1855-7412-9491-5b3e09eafc61 in the same
window_598fe5564b71e8ad51cdfa48/repository. Source Graph refresh failure and
bounded unindexed CMake/GAS fallback are recorded; subsequent refresh
4bd2938949d0460bb072e8a6d331111f succeeded and corrected source bodies were
fresh before final review. No plugin repair was attempted.
