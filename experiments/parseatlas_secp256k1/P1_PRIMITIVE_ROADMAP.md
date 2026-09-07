# P1: field and scalar primitives first

Owner decision, 2026-09-05: accelerate arithmetic modulo p and modulo n before
returning to point, signature, aggregation or other upper-layer work.
This supersedes the proposed post-F2 upper-layer follow-up.

## Contract and order

1. Freeze and measure the actual fast FE64 and Scalar implementations: add,
   subtract, multiply, square, inverse. Native C++, actual configured library,
   independent correctness oracle, dependent and independent-state workloads.
2. Establish separately labeled resident FE52 and constant-time primitive
   baselines. Never compare lazy output with canonical output as equal work,
   or a variable-time operation with a secret-safe operation as interchangeable.
3. Compare candidate primitives against the frozen baseline in the SAME process,
   build profile, input corpus, timing boundary and alternating measurement
   rounds. Keep every negative result and conversion cost.
4. Integrate only qualified winners, including their direct primitive call sites,
   tests, aliases, platform guards and fallback paths. Production acceptance
   requires measured improvement, correctness and preserved security contracts.
5. Only then re-open upper-layer experiments to measure actual transfer.

This is a research sequence, not a requirement to find one universal algorithm
or to improve every operation. A rejected candidate is a valid result.

## Current implementation inventory

| Domain | Add / subtract | Multiply / square | Inverse |
|---|---|---|---|
| Fast FE64, p | Canonical 4x64 carry/borrow and sparse-prime correction | Configured x86 GAS BMI2+ADX full modular kernels with runtime dispatch | Signed 5x62 SafeGCD on this host; zero throws |
| Fast Scalar, n | 4x64, data-dependent correction branches | Native C++ 16-product multiply and complement folding; public square is `a*a` | Signed-62 SafeGCD here; zero returns zero |

Authority: `src/cpu/src/field.cpp` add423, sub404, mul1137, square1180,
inverse3541; `src/cpu/src/scalar.cpp` add81, sub105, multiply342,
inverse893; public headers `field.hpp` and `scalar.hpp`.
Scalar's header Fermat comment does not describe its current implementation.
Field equality normalizes its operands, so equality alone is not a canonical
output test: compare raw limb range and all 32 serialized bytes as well.

## Candidate queue, after the baseline gate

| Candidate | Different observation | Required equal-work gate |
|---|---|---|
| Scalar dedicated symmetric square | 10 unique products rather than 16 | Retain the full modular reducer; prove all doubled cross-product carries, compare canonical bytes; fewer multiplies is not a speed result |
| Field FE52 multiply/square bridge | Unsaturated limbs expose independent work | Include FE64-to-FE52 packing, normalization and canonical FE64 return; separately label resident FE52 |
| Field add/sub carry scheduling | Critical path and live state can dominate operation count | Preserve full canonical correction, aliases and appropriate security contract; inspect actual generated code |
| Scalar bound-aware reduction | High folded coefficients have tighter bounds than generic words | Prove bounds for the full 512-bit product, not F2's 320-bit accumulator; measure selects versus multiplies |
| Field/scalar inverse scheduling | Divstep blocks and normalization dependencies | Keep current SafeGCD as baseline, zero behavior and security class; no unchanged retry of historically rejected guards |

Existing Blocked2 and GPrecomputed modular-n add candidates already lost;
the earlier raw-u256 carry advantage is not a full modular primitive advantage.
F1/F2 measured final aggregate sums, not the latency of one scalar addition.
Their results remain frozen and are not new P1 evidence.

## Twenty-five lenses: evidence required for each candidate

| Lens | Concrete question / evidence |
|---|---|
| V01 Reachability | Which exact primitive API and compiled branch execute? |
| V02 Corpus | What inputs, edge cases, runtime seed and distributions were used? |
| V03 Equivalence | Independent integer oracle and all 32 reference bytes agree? |
| V04 Definedness | Zero behavior, valid inputs, UB and exceptions preserved? |
| V05 Range / overflow | Limb and accumulator bounds proved for every intermediate? |
| V06 Aliasing | Self-alias and in-place contracts tested? |
| V07 Observable output | Canonical limbs, endianness and error behavior unchanged? |
| V08 Invariants | Which mathematical invariant explains the implementation? |
| V09 Constant time | Public/secret contract, branches and memory addresses audited separately? |
| V10 Resources | Exact scratch space, allocation and stack requirements? |
| V11 Property visibility | Which representation exposes the useful property? |
| V12 Dependency depth | Generated carry, multiplication and reduction dependencies? |
| V13 Live state | Registers, spills and simultaneously live values? |
| V14 Operations | Exact work performed, not just symbolic operation count? |
| V15 Conversion | Packing, normalization, return and materialization costs included? |
| V16 Latency | Dependent workload timing; harness costs and input cycles disclosed? |
| V17 Throughput | Independent-state timing; one core versus multicore distinguished? |
| V18 Instructions | Disassembly and, if available, hardware counters? |
| V19 Cache | Working-set sizes and measurements, not inferred bandwidth claims? |
| V20 Code size | Symbol/text sizes and call-boundary differences? |
| V21 Portability | GCC/Clang, assembly/portable paths and fallback behavior? |
| V22 Compiler | Exact compiler, macros, flags, LTO and binary hash? |
| V23 Policy / ties | Scenario-specific selection, no hidden reruns or secret-dependent routing? |
| V24 Transfer | Upper-layer transfer remains unmeasured until the primitive gate passes |
| V25 Prior art | Distinguish known representations/algorithms from a genuinely new finding |

These are observation lenses, not 25 asserted wins or a claim that Atlas's
operator-precedence spaces transfer literally to machine arithmetic.
Unknown quantitative evidence remains `PENDING IMPORT`, never an invented zero.
Finite testing is not a universal proof or a side-channel certification.

## P1 first-wave measurement boundary

The first correctness gate found two existing field reduction defects, before
any performance run: three GAS carry bits were used as masks without expansion,
and both portable Step-2 carry tails could drop a carry into limb 4. Minimal
corrections and explicit regression KATs passed independent review. See
`data/p1_correctness_needfix_20260905.json` for original failing bytes and
`data/p1_validation_20260905.json` for four corrected configurations. Performance
results identify the corrected baseline, not the mathematically incorrect
unmodified library. Original binaries and build hashes remain retained.

First wave: fast FE64/Scalar only, ten operations, one dependent state and four
independent states on one pinned core. No CT claim and no upper-layer benchmark.
Returning public arithmetic APIs include their existing ABI/dispatch costs.
End-of-job byte materialization is amortized, not a per-operation byte API.
Inverse uses result-dependent lookup in nonzero runtime inputs, avoiding the
two-value `inverse(inverse(x))` cycle; its timing also includes index/load costs.
Full numerical claims must identify this workload rather than claim pure
instruction latency.

Build: source HEAD `fef231d4e4173bd016fb2a3a1eff67087396a203` plus the recorded
field.cpp / GAS / regression corrections, experimental
branch `experiment/representation-search`; actual standalone CPU CMake Release
library, default assembly and fast reduction, explicitly LTO OFF for visible
and reproducible call boundaries. This is not the default LTO-ON performance.
Original build: `/tmp/parseatlas-p1-primitives.1hlZNv/build`.
Corrected build: `/tmp/parseatlas-p1-primitives.1hlZNv/build-fixed`.
Record compiler-generated commands and content hashes in the evidence manifest.

Current machine is i5-14400F, 16 logical CPUs / 10 physical cores. Build workers
are min(8, available CPUs minus 2), observed as 8; timing runs alone on CPU4.
CPU4's sibling is CPU5. Preflight observed powersave governor and turbo enabled,
unlike F2's earlier recorded configuration. Do not compare absolute timings
across those runs. No global power policy is changed. External services,
temperature and scheduler noise are not fully controlled.

Correctness, compiler/sanitizer validation, raw timings and review findings are
reported in `P1_PRIMITIVE_RESULTS.md` when available. No speedup or production
replacement is accepted by this roadmap alone.

## Workflow provenance

Owner's temporary Task MCP suspension continues; no task was claimed, launched
or accepted. Root uses live manager Source Graph and verifies exact source
hashes. Worker-scoped tools are not exposed: workers use manager-provided exact
targets, not manager impersonation. Manager reviews code it did not write.
Source Graph manual refresh again reported `identity_slot_owned`; bounded
unindexed CMake/assembly inputs and the failure are recorded in
`data/p1_source_graph_needfix_20260905.json`. No plugin repair is in scope.
