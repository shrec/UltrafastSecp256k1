# Field research: change the representation and priorities, not only instruction count

Owner clarification (2026-09-05): the CUDA 32/64-bit example illustrates how
unconventional choices can change cost. It is not a request to hunt for a private
file or limit the search to pointer tricks. Field-p remains first, scalar-n next,
then upper-layer integration after collecting qualified candidates.

## Search families, ordered by existing evidence

| Family | Change to investigate | Control / reason | Next gate |
|---|---|---|---|
| Representation inside one operation | Choose width separately for product, carry, fold and storage | P7 is a small GPU input-view control; CPU FE64/FE52 residency and conversion already differ in P2 | Distinguish representation identity from an actual emitted instruction change; include setup |
| Dependency structure | Split independent products/folds; schedule carry only when mathematically needed | P5 precomputed/serial variants often compile identically; P6 chain and ILP winners differ | Explicit carry bounds, real dependency/codegen change, chain and independent-state C++ measurements |
| Materialization boundary | Fuse product with pseudo-Mersenne folding instead of materializing all 512 bits | Retained P5 row product supplies an exact reference; p = 2^256-(2^32+977) supplies the residue identity | Prove every high term/carry is preserved; raw-reducer tests plus complete canonical field outputs |
| Normalization placement | Delay only within a proven range or final-only contract | P3/P4 large sum wins and P2 weak-before-full constraint | Exact headroom/overflow proof, final-only vs every-step observers, setup/size crossover |
| Memory and liveness | Resident representation, bounded blocks, streaming partial state, public workload selection | P4 bank losses and size crossover prohibit assuming more independent accumulators always win | Same work/data, codegen and working-set sweep; physical bandwidth needs counters, not speculation |
| Final correction | Alternative add/sub/mask/carry schedules at the canonicalization boundary | P6 Clang borrow-dependent branch is an explicit security limit | C++ proof/oracle first; emitted-code CT review separate from runtime gain |
| Inversion organization | VT/CT single inverse and public batch inverse as distinct primitives | Field collection remains incomplete | Zero policy, scratch/alias/setup contracts; no secret-dependent scheduling substitution |

These are candidate families and remaining work, not implemented discoveries.
Run bounded waves with non-overlapping worker writes, then independent manager
mechanical gates and source review. Start each wave from a falsifiable question.
A larger operation count may be accepted when measured latency/throughput improves;
it does not itself prove memory traffic or critical-path reduction.

## What the 25 lenses must jointly produce

For every candidate retain one record combining: exact equivalence/invariant,
range and alias contract, visible outputs, actual reachable code, carry and
dependency structure, live state, representation transitions, latency and
throughput workload, logical memory layout, measured versus unmeasured resources,
compiler/platform/security boundary, positive and negative results, and where
the candidate may transfer. The fixed V01–V25 map remains in each wave's protocol
and measured result; do not turn the lenses into 25 independent success claims.

Useful outcomes include a conditional faster candidate, a correctness witness,
a compiler-equivalent control, a size-dependent crossover, and a failed
hypothesis that narrows the next search. Reusing Comba, bit representations or
pseudo-Mersenne identities does not by itself establish novel mathematics.

## Current boundary

[P6](P6_FIELD_BOUNDARY_RESULTS.md) remains accepted experimental evidence.
[P7](P7_CUDA_VIEW_PROTOCOL.md) is only the already-started small GPU side probe.
The next primary CPU arithmetic wave should separate final-correction scheduling
from fused product/folding; do not change both and call the result a view-only
gain. No new canonical task cards, production integration or upper-layer rollout
are authorized by this planning document.
