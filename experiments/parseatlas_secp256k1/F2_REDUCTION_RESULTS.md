# F2 results: deferred scalar-order reduction and a scenario portfolio

Research disposition: **retain the new methods as validated experimental candidates; no production code or dispatcher changed.**

Two separately retained native C++ series show that a size-dependent portfolio is useful.
For every tested N >= 16, Columns beats both eager Serial and Wide5 in all 80/80 same-round pairs.
At N=65,536, one-core Columns is 7.312x / 7.382x faster than eager Serial by the registered paired metric.
At N=1, Serial has the better median in both series.
At N=1,048,576, one-core Columns and the eight-worker cold method have similar wall time; their median ranking reverses between series. There is no stable speed winner between those two at this size.

This is one final scalar-order-n sum, not a whole-engine speedup, a field-p result, or a claim of mathematical novelty.
This narrative report is not a completed or admitted ParseAtlas case-study record. Unknown measurements remain PENDING IMPORT.

## Matched contract and transformation

All variants return only the canonical 32-byte result of the SAME computation:

~~~text
before:
  x = x0
  for a in rhs[0..N):
      x = (x + a) mod n
  return x

Wide5:
  T = [x0[0], x0[1], x0[2], x0[3], 0]
  for a in rhs[0..N):
      add a exactly to the five-word T, retaining the low256 carry
  return reduce320(T)

Columns:
  C[i] = uint128(x0[i]), i=0..3
  for a in rhs[0..N):
      C[0] += a[0]; C[1] += a[1]
      C[2] += a[2]; C[3] += a[3]
  T = normalize_columns(C)
  return reduce320(T)
~~~

Inputs are canonical in [0,n), all RHS values are runtime inputs independent of intermediate outputs, x0 is included once, and only the final modular answer is observable.
The four columns are parts of ONE integer, not four unrelated output streams.
The exact decode obligations are:

~~~text
decode(Wide5)   = sum(T[i] * 2^(64*i), i=0..4)
decode(Columns) = sum(C[i] * 2^(64*i), i=0..3)
decode(Wide5) = decode(Columns) = x0 + sum(rhs)
encode32(reduce320(decode)) = encode32(frozen Scalar recurrence)
~~~

With B=2^256, W=2^64 and K=N+1, the N<=1,000,000,000 contract gives
T<=K(B-1)<2^286. Each column is <=K(W-1)<2^94.
Normalization validates ORIGINAL columns before adding carries; incoming carry<=K-1 implies the adjusted column<=KW-1, and final high word<=N.
Buffer extents/lifetimes and canonical input values remain preconditions; count/addressability/null and column-cap errors are checked.

For the general reducer, n is the secp256k1 group order and

~~~text
D = B-n = 0x14551231950b75fc4402da1732fc9bebf
T = L + H*B, 0<=L<B, 0<=H<W
T mod n = (L + H*D) mod n
~~~

H*D can occupy 193 bits: the implementation retains four product words.
S=L+H*D<B+2^193<2B. If its first high carry is one,
R=S-B<=H*D-1, so R+D<=(H+1)D-1<=WD-1<2^193<B.
Thus that second fold cannot overflow B. Otherwise S was already below B.
The folded result is in [0,B); since B<2n, one conditional subtraction canonicalizes it.
The general reducer accepts every 320-bit integer, beyond the stream's tighter bound.
This is an algebraic argument plus finite implementation tests, not an exhaustive machine-checked proof.

Transformation labels: **limb layout, lazy reduction, reassociation**.
Reduction count at the source modular-map boundary changes from N eager additions with per-step correction to one final reduce320 for nonempty F2 jobs.
Wide5 still propagates cross-limb carries each input; Columns postpones them until the boundary.
This is not a transfer of 25 notation-precedence conventions to machine arithmetic.

## Two-series results

Primary statistic: median of eight same-round ratios
(serial elapsed/jobs) / (candidate elapsed/jobs). Above 1 favors the candidate.
These are ratios of normalized REGION MEANS with separately calibrated job counts, not equal-work trials or individual-job latency.
Each entry below is series 1 / series 2; no best-series selection or pooling.

| RHS count N | Serial ÷ Wide5 | Serial ÷ Columns | Serial ÷ cold multicore |
| --- | --- | --- | --- |
| 1 | 0.708 / 0.707 | 0.677 / 0.652 | 0.000311 / 0.000291 |
| 16 | 1.486 / 1.478 | 2.543 / 2.561 | 0.000328 / 0.000321 |
| 256 | 1.697 / 1.679 | 3.654 / 3.652 | 0.005017 / 0.004951 |
| 4096 | 1.751 / 1.762 | 3.975 / 3.971 | 0.077 / 0.080 |
| 65536 | 3.502 / 3.513 | 7.312 / 7.382 | 1.924 / 2.041 |
| 1048576 | 3.023 / 3.002 | 4.072 / 4.051 | 4.174 / 3.978 |

For transparency, absolute normalized times follow: **ns/full job, median [inclusive IQR]**.
Dispersion is across eight region means in each cell. Full min/max/MAD, all raw pairs, calibration jobs and per-RHS values are in the numeric audit and raw artifacts.

| N / series | Serial | Wide5 | Columns | Cold multicore |
| --- | --- | --- | --- | --- |
| 1 / 1 | 8.869 [0.039] | 12.558 [0.067] | 13.124 [0.116] | 28506.639 [1614.554] |
| 1 / 2 | 9.548 [2.025] | 12.734 [2.484] | 14.169 [1.791] | 34257.286 [5051.796] |
| 16 / 1 | 69.595 [0.399] | 46.867 [0.251] | 27.328 [0.252] | 212120.023 [28782.203] |
| 16 / 2 | 69.842 [0.285] | 47.156 [2.623] | 27.315 [0.791] | 218208.517 [8733.129] |
| 256 / 1 | 1053.958 [4.486] | 621.536 [3.922] | 287.928 [0.777] | 210669.598 [9521.762] |
| 256 / 2 | 1050.817 [1.199] | 625.999 [4.347] | 287.726 [0.639] | 213060.664 [29370.998] |
| 4096 / 1 | 16829.859 [59.657] | 9613.294 [44.753] | 4233.862 [7.818] | 219791.189 [35261.984] |
| 4096 / 2 | 16841.448 [36.022] | 9564.506 [94.210] | 4240.062 [11.003] | 210333.485 [21183.403] |
| 65536 / 1 | 555713.474 [7891.298] | 157792.618 [975.294] | 75546.124 [930.843] | 288435.070 [58824.829] |
| 65536 / 2 | 554329.156 [6525.358] | 157746.106 [1021.221] | 75116.880 [503.827] | 273539.646 [43489.760] |
| 1048576 / 1 | 9444207.547 [46263.820] | 3124726.806 [81773.964] | 2322898.488 [31623.766] | 2261336.174 [126124.525] |
| 1048576 / 2 | 9414854.953 [23960.305] | 3140145.884 [41488.811] | 2325478.335 [14815.727] | 2363853.618 [122744.567] |

Scenario conclusions:

- **N=1:** retain eager Serial as the experimental median winner. Both delayed methods lose by median in both series. The second series is noisy: each delayed method has one favorable pair, so this is not an all-pairs tiny-case dominance claim.
- **N=16,256,4096,65536:** Columns wins over Serial and Wide5 in every registered pair; it also beats the cold method. At N=65,536, direct cold/Columns paired medians are 3.817x / 3.634x. One core is sufficient for this measured winner.
- **N=1,048,576:** Columns takes about 2.323 / 2.325 ms per full job; cold takes about 2.261 / 2.364 ms. Direct cold/Columns paired medians are 0.974343 / 1.020863; Columns wins 3/8 then 5/8 pairs. Do not declare a stable speed ranking from this crossover. Columns uses one coordinator core; cold uses eight heterogeneous worker cores plus its coordinator. CPU time, energy, and equivalence within a statistical margin are not measured.
- Wide5 beats Serial in all 80/80 N>=16 pairs but loses to Columns in those same pairs. It remains an informative control for separating reduction deferral from cross-limb carry deferral.

Across all six sizes, favorable counts are Wide5 81/96, Columns 81/96 and cold 32/96.
These counts are descriptive and are not a pooled performance estimate.
Untested sizes 2..15 and realistic caller-size distributions do not have measured dispatch thresholds.

## Timing custody and limitations

The [prospective protocol](F2_REDUCTION_PROTOCOL.md) and
[run plan](data/f2_run_plan_20260905.json) were frozen before sustained timing;
plan SHA256 70b739ce35f63e3709cb7edfc8c985275b5960c19d4a653c26972ae77ca84f62,
registered 2026-09-05 17:15:25 UTC.
All manager mechanical gates and independent source/reachable-binary reviews completed before measurement.
Series 1 uses N=[256,1,65536,16,1048576,4096]; series 2 reverses that list.
The same seed 20260905 is reused; separate process invocations do not make independent input corpora.

Retained: **384 measured regions, 96 warmup regions, 192 untimed preconditioning jobs, 178 timed calibration attempts**.
The diagnostic unqualified run and smoke/sanitizer records are separate and excluded from performance statistics.
The manager independently replayed exact timestamp differences with BigInt, calibration states/caps, work accounting, balanced order and every statistical summary from raw native output.
Unix-nanosecond JSON lexemes are preserved without floating-point reserialization.

Exactly **6/384** measured regions are below the 200 ms target, all cold multicore at N=65,536, series 2:
172.682597, 171.098046, 187.207996, 174.955514, 184.920545 and 182.169660 ms.
That variant had qualified at 671 jobs with consecutive 312.730955 and 263.167142 ms trials.
The stronger prospective calibration therefore does NOT guarantee subsequent durations.
No short or slow sample was removed, recalibrated or replaced.
The measured-region range is 171.098046–563.971241 ms.
The same cell's Serial/Columns regions all meet the target, so the short-duration limitation affects the cold comparison, not the 7.38x Serial/Columns ratio.

Every full job includes initialization, structural checks, accumulation, final normalization/folds/canonicalization and materialized output.
The cold method includes allocation, creation, explicit affinity, join, merge and destruction.
Input generation and construction/replay of the unchanged public Scalar reference are uniformly outside timing.
GCC noinline/noipa boundaries and empty compiler memory clobbers occur per FULL job, never per addition.
All full-job calls and 32-byte stores survive in each compiled timed repetition loop.

Repeated jobs reuse the same immutable x0/RHS. Every output is materialized, but only the last identical-job output in each timed region is compared byte-for-byte with Scalar; each untimed conditioning job is checked.
Checksums in artifacts are evidence tokens, not the equality oracle.
Caches, allocator resources and thread infrastructure can remain warm; “cold” refers to per-job thread lifecycle, not flushed caches.
Per-variant job volumes differ. External load, sibling activity and thermals are uncontrolled. No PMU-backed bandwidth/cache causality, continuous frequency, energy, constant-time or end-to-end engine claim follows.

## Native correctness and compiled mechanism

Five manager-executed correctness configurations pass with zero mismatches:
GCC14 native -O3/-DNDEBUG, GCC14 with SECP256K1_NO_INT128, Clang18 native,
UBSan, and ASan+UBSan.
Each uses the SAME seed/corpus, not five multiplied unique data sets:

- 101,445 general 320-bit reducer cases: 100,000 random, 1,061 boundary-family cases and 384 deliberate fold-boundary cases. 345 cases exercise first-fold carry.
- 10,153 column cases, including 10,000 random and 32 synthetic repeated-array witnesses. Four cap witnesses model N=1,000,000,000 without allocating or executing a billion-element array.
- 2,276 actual array fixtures: 228 boundary-family and 2,048 randomized arrays; 646,214 actual public Scalar replay additions.
- A 2,227-case reducer subset replays all 320 bits through Scalar using 980,934 additions. No truncating conversion hides the fifth limb.
- All 40 intermediate bytes, all 128 bits of every column, and all 32 final bytes are compared. Public big-endian serialization checks bind the corresponding Scalar result.
- Input preservation, 20 checked errors, empty identities, 320/256/512 single-bit comparator corruptions and explicit lost-high/carry controls pass. Fixture categories overlap.

Correctness checksum is 6dfb20c5e053ec85 in all five configurations.
Driver self-tests, six native size smokes, a UBSan driver smoke, parser/error gates and the real capped-unqualified exit pass.
Only the preexisting scalar.cpp:47 unused ge warning remains.
SECP256K1_NO_INT128 changes frozen carry helpers, not the new UInt128 columns/products: this is NOT a fully portable F2 implementation.
Default build, production call sites and upstream suite wiring were not changed; production-suite integration status is PENDING IMPORT.

The measured binary contains a Columns accumulation loop at 0xb400–0xb426 with four independent add/adc pairs and no stack-memory accesses in that loop.
Wide5's loop at 0xc330–0xc3a2 retains its cross-limb chain and has stack writes.
Normalization and reduction are outside these loops but inside the timed full jobs.
This is a static instruction observation, not an isolated causal attribution of the measured gain.
Source accumulator payloads 40/64 bytes are not register or total-spill counts.
[Disassembly evidence](data/f2_disassembly_20260905.txt) retains the job/repetition/worker/lifecycle ranges.

## Identity, 25 lenses and next gate

Engine branch: experiment/representation-search; HEAD fef231d4e4173bd016fb2a3a1eff67087396a203.
Machine: Intel i5-14400F, 16 logical / 10 observed physical cores, Linux 6.8.0-138-generic.
Timing compiler: GCC 14.2.0, C++17, -O3 -march=native -DNDEBUG -Wall -Wextra -pthread, no LTO.
Coordinator CPU4; selected worker representatives [0,2,6,8,10,12,13,14], reserving the coordinator core and an additional core.
N=1 uses ONE worker; other registered sizes use eight.
Governor endpoints report performance and no_turbo=1; snapshots are not fixed in-region frequency measurements.
No global settings or sudo were used.
All 36 inherited F1/control/caller/artifact hashes matched before timing.

Measured binary SHA256: b06a6d46c0896c4c579b465e4c5f49dbfc249b3da0604b49671fb50f8da89a76.
Kernel SHA256: 57ede13d3f0df8b9c12f8578c60e533b416c23da670671d19ec3d731b860c651.
Driver SHA256: aa04c645707a15ba126d3d5606789a8d879543f7c186a13e7c00350b465a1c3e.
Other identities, exact commands and evidence paths are bound in the [manifest](data/f2_manifest_20260905.json).
See [correctness gates](data/f2_validation_20260905.json),
[driver gates](data/f2_driver_validation_20260905.json),
[premeasurement review](data/f2_review_gate_20260905.json), and
[numerical audit](data/f2_numeric_audit_20260905.json).

The [25-lens delta ledger](F2_25_LENSES.md) records this transformation family from all 25 viewpoints, including unresolved gates; it does not claim 625 completed searches.
Local ParseAtlas guidance is pinned to clone f428fb53948bc15c4668c303366231817a947708;
modular_arithmetic_guidance.md SHA256 03d70a369e4df8094091e235fce1bad2635121cd0976795dc9ff22639fb07ca1.
Its decode obligation and separate depth/live-state/cost criteria organize this experiment. ParseAtlas itself did not measure this engine.

Next gate: measure small real caller-sized batches and construct/review a production-eligible constant-time adapter before replacing ct::scalar_add call sites.
Previously inspected MuSig2/FROST aggregation has CT and validation requirements not met by these variable-time probes.
Any eventual portfolio may depend only on public N/resources/output contract/security requirements, not secret scalar values.
No production threshold, external release, commit or push is part of this result.
Task MCP remains owner-suspended; this is direct research review, not acceptance of canonical task PA_SECP_M1_MODN_NATIVE_015.
