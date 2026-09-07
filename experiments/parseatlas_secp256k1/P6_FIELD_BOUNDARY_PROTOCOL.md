# P6 field boundary controls — preregistered protocol

Status: fixed before P6 full timing. This wave isolates implementation boundaries;
new carry-fold, unrolled Comba and fused-product algorithms are later experiments.
All prior P1–P5 code/evidence remains unchanged. No production integration.

## Question and fixed matrix

P5 row+serial won dependent chains against the actual FE64 API but lost ILP4.
Does that advantage persist with a separate, opaque pointer-ABI arithmetic call?
Compare the actual API with its unchanged ASM leaf, then compare inline and
outlined forms of the unchanged P5 row arithmetic. These controls change
visibility, materialization and liveness as well as calls; timings are not pure
call-instruction costs or guaranteed causal decomposition.

| Route IDs multiply / square | Implementation |
|---|---|
| 0 /4 | Corrected actual FE64 returning APIs, P5 control |
| 1 /5 | Direct existing field_mul_full_asm / field_sqr_full_asm |
| 2 /6 | Inline unchanged P5 row product + serial reduction |
| 3 /7 | Same row arithmetic in separate-TU extern-C pointer/output functions |

Four groups: multiply chain1, multiply ILP4, square chain1, square ILP4;
4routes/group =16cells. Group g uses cell IDs4g..4g+3. Row outlined kernels
accept a/b/output pointers for multiply, a/output for square, exactly the native
ASM pointer ABI. Definitions are absent from the caller translation unit;
-fno-lto and noinline/noipa (GCC), noinline (Clang) preserve the boundary.
Inspect emitted hot loops to verify real external calls, not just source intent.

Canonical [0,p) inputs return canonical four limbs/all32BEbytes, p=2^256-2^32-977.
Input operands are read-only and a/b may be the same object. Output arrays must
be valid, hold four uint64 words, and be disjoint from inputs. No output/input
alias extension, restrict promise or production security contract is introduced.
Use the unchanged [P5 integer proof](P5_FIELD_PRODUCT_CONTRACT.md). P6 changes
no arithmetic identity, fold count or intermediate width bound.

Direct native wrappers call the existing ASM symbol immediately, with no new
per-call runtime dispatch. **The caller must first admit native execution** using
native_asm_available(): this profile requires Linux x86-64, GCC/Clang, ASM build,
BMI2 and ADX. Benchmark/test admission precedes every possible direct job/trace.
An unsupported native host must not execute the direct helper. Compile-disabled
wrappers fail closed without unresolved ASM references. This precondition belongs
to experiment-local helpers, not a weakened production API. NO_ASM sanitizer
qualification explicitly skips ASM arithmetic and tests fail-closed wrappers;
it must not be reported as sanitizer instrumentation of ASM.

## Work, observation and corpus

Preserve the exact P5 seed20260905 source recipe and complete corpus hash
7c2e57392893ac80:4canonical nonzero seeds,256canonical nonzero RHS and256raw512
entries. The last array is retained solely for corpus-identity controls and is
not used by P6 arithmetic; its allocation/generation/hash checks are untimed.
Each active lane performs1024steps: multiply x=mul(x,rhs[step&255]); square x=x².
ILP4 has four independent scalar states on one core, not four threads. One core
is intentional to study dependencies without thread dispatch, not a multicore optimum.

Each job includes active seed initialization, all steps and final128-byte output:
chain1 one BE32 result plus96zero inactive bytes; ILP4 four BE32 outputs. All
128bytes remain observable. No per-step benchmark barrier. Use whole-job
noinline/noipa and a complete-output barrier, and verify emitted repeated calls.
Allocation, corpus generation and prevalidation are outside timing.

Each primitive step stays canonical. Untimed traces compare raw canonical limbs
and every byte against the corrected FE64 reference; full compiled evaluator/job
outputs must match the final trace, and lane0 must agree between modes. All
inputs are regenerated and compared after validation and after all regions.
Every recorded region checks its last identical job's full output outside timing.
This is not instrumenting every timed step or every repeated job with a byte test.

## Timing policy and comparisons

Native C++20, GCC14 -O3 -march=native -fno-lto -DNDEBUG and frozen P1 library;
build the new outlined .cpp as a separate translation unit. CPU4, existing
powersave/turbo policy unchanged. No owned builds/tests/competing benchmarks
while full timing runs. Record hashes and exact commands before measurement.
Background services, thermals, frequency and SMT sibling activity remain uncontrolled.

Two complete series; reverse the second series. Two warmups/eight measured rounds
per cell, at least200ms per later region. Preserve P5 continuous-region method:
two qualifying same-count calibration regions; short later prefixes are extended
inside the same clock, never restarted or discarded. Retain cumulative probes,
actual work, all failures and short calibrations. Fail closed on finite extension
or work caps. O_EXCL prevents output overwrite; no selective reruns.

Primary: API /each of three candidates in each group =12paired summaries/series.
Secondary: directASM/rowINLINE, directASM/rowOUTLINED, rowINLINE/rowOUTLINED,
each in all four groups =12summaries. Mode comparisons: chain/ILP4 for each of
eight operation/routes =8summaries. Each summary has eight same-round ratios;
use elapsed/actual_operations, actual_operations=jobs*1024*lanes. Greater than one
favors denominator. Report paired medians and all wins/losses/ties; never replace
with a ratio of separately computed medians or pool P5/historical/smoke timing.

A matched pointer ABI does not imply matching machine instructions/register use.
API/directASM removes wrappers/dispatch around the same ASM kernel, but generated
caller state handling can also differ. Inline/outlined row changes optimizer
visibility. Preserve these limits even if one route wins every pair.

## Qualification and deliverables

Independent C++ Boost mod-p oracle plus corrected FE64 reference, canonical raw
words and all32bytes, zero/one/p-1, both exact P1 carry KATs and older large-square
KAT, carry-boundary neighbors, deterministic random canonical pairs and bounded
step-by-step chains. Check read-only input and same-object a/b behavior; guarded
output buffers must detect writes outside four words. No inadmissible output alias.

Run GCC and independently compiled Clang field sources with all four routes;
NO_ASM ASan+UBSan tests three non-ASM routes and explicitly reports skipped direct
ASM. Run separate synthetic-duration diagnostics on16cells, including extensions,
invalid clocks, initial work cap, malformed lane counts and retained failed-main
JSON. Synthetic clocks are never performance data. Within-region capped-tail/work
exhaustion may remain static-reviewed only if not dynamically tested; disclose it.

Workers own disjoint new feature/test files; manager runs mechanical gates before
complete code/contract review. Frozen-source changes invalidate affected gates.
Use [contract](P6_FIELD_BOUNDARY_CONTRACT.md) and [25lens plan](P6_FIELD_BOUNDARY_LENSES.md);
all observations, negative results and unavailable quantities must be retained.
No constant-time certification, novel-mathematics/world-record or whole-engine claim.

Manager repo_id repo_666797171f0141c58bf05f579b2ee16e, session
01a06be6-2904-7c62-9e7d-1245c34a5312 verified by bootstrap/Task health. Owner's
Task suspension remains; no canonical card actions. Worker-role Source Graph
unexposed NeedFix remains; exact manager handoffs plus continuous manager graph
queries, recorded new-file fallbacks and final indexed hash gates. No impersonation
or context-database writes. No production/scalar/point/signature/default-CI changes,
commits, pushes or tags. P5 frontier bytes archived before edits in
[data/p5_frontier_snapshot_20260905.md](data/p5_frontier_snapshot_20260905.md), SHA
25370020b8c7fcada0675f86418667d63ae6673cfe3accd1ab625c73a4bd84be.
