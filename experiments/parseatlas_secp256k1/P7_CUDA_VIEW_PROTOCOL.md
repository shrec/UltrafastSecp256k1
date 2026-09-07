# P7: field multiplication views — bounded GPU side probe

Status: preregistered before full timing. Owner clarification: the dual-pointer
CUDA example motivates unconventional representations generally; locating or
reproducing the private version is not the research objective. This small probe
does not replace the CPU field-first frontier or prove the reported >10% gain.

## Question and controls

For the same canonical modular-p multiplication, does exposing 4x64 input words
as legal 8x32 values change generated code or execution cost? Keep a changed
algorithm as a separately labelled contrast; never attribute its difference to
pointer syntax.

| Route | Product | Reduction | Attribution |
|---|---|---|---|
| 0 | Existing CUDA field_mul / Comba32 | Existing hybrid reducer | Current production control |
| 1 | Exact cloned Comba32; explicit split shifts | Same hybrid reducer | Clone/control check |
| 2 | Same cloned Comba32; memcpy into live uint32_t arrays | Same hybrid reducer | Input-view-only comparison against 1 |
| 3 | Existing 4x64 product | Existing 64-bit reducer | Algorithm-plus-reducer contrast, not view-only |

No incompatible uint32_t lvalue dereference of a uint64_t object, no restrict,
inactive-union punning, changed production default or unsupported CT claim.
Memcpy here is an experimental representation control, not a claim to reproduce
the owner's two persistent pointers or to cause physical memory traffic.
Current C++ type accessibility is documented in
[the working draft](https://eel.is/c++draft/basic.lval#11); object representation
and trivially-copyable rules are in
[basic.types](https://eel.is/c++draft/basic.types).
The host/device little-endian premise must be checked or qualified explicitly.

## Invariant and finite qualification

Field p = 2^256 - 2^32 - 977; canonical inputs and canonical 32-byte output.
Each independent thread has x=a[i], fixed rhs=b[i], and repeats x=x*rhs mod p.
Do not use 0/1 inputs in performance corpora. Raw product tests admit arbitrary
256-bit words and compare the complete 512-bit result separately.
Comba keeps a three-word, radix-2^32 accumulator: at most eight 64-bit products
per column plus the preceding carry need at most 67 significant bits, within
96-bit state. Preserve every original term and carry boundary.

Independent native C++ Boost integer oracle plus the frozen corrected CPU
FieldElement reference; random/boundary/all-ones/bit-carry cases, serialized
bytes, raw high words, derived chains, same-object inputs, input preservation,
host output guards and intentionally corrupted observer controls.
Finite passing tests do not prove CT, universal arithmetic correctness or
cross-type alias legality. A failing route is blocked before performance,
with exact inputs retained; no silent fallback.

## Timing and emitted-code gates

Native CUDA kernels, native C++ host driver; no Python arithmetic or timing.
Four routes x count 32 / 32768, steps 128. These are low-occupancy and larger-grid
dependent-thread workloads, not CPU ILP4 or single-instruction latency.
One device/stream. Allocation, transfers and JIT warmup outside event interval;
input loads, arithmetic, output stores and inter-launch gaps inside the timed
event span. Repeated launches reset seeds; all final outputs checked outside it.
Two warmups and eight balanced-order measured rounds, two full series with
reversed ordering, every accepted warmup/measurement interval >=200 ms.
Retain short calibration/attempt intervals and bounded duration retries; compare
same-round ratios from actual work, not ratios of medians. Report duration or
isolation deviations rather than removing them. Full timing belongs to manager
after independent correctness gates, code review and source/binary hash freeze.

Inspect PTX and runtime kernel attributes. Equal normalized PTX is meaningful
negative evidence for a source-level view change, but is not verified equal
runtime SASS. Installed toolkit observed 12.0.140 and GPU RTX5060Ti cc12.0:
any old-target PTX JIT must be named, not advertised as a native Blackwell build.
NVIDIA documents the
[PTX forward-compatibility route](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/).
No driver/toolkit installation, lock-clock change, production integration or
performance claim about all GPUs/CPU/the whole engine.

## Preregistered 25 lenses

These are observation dimensions, not 25 experiments or discoveries.

| Lens | Observation / discriminating gate |
|---|---|
| V01 Reachability | Each named route reaches the intended product and reducer |
| V02 Corpus | Frozen nontrivial input identity, exact count/steps |
| V03 Equivalence | Full raw 512-bit and canonical byte outputs vs independent oracle |
| V04 Definedness | Live typed objects, memcpy, supported compiler/runtime admission |
| V05 Range / overflow | 67-bit Comba column within 96-bit state; full raw high words |
| V06 Aliasing | Same read-only input accepted; no cross-type dereference; host guards |
| V07 Observable output | Every field step canonical; all final outputs materialized |
| V08 Invariants / carries | Carry-neighbor, maximal limb and repeated-chain cases |
| V09 Constant time | Separate security gate; no CT inferred from source or speed |
| V10 Resources | Runtime registers/local/shared attributes; not measured spill cost |
| V11 Property visibility | Does input presentation expose an actually different schedule? |
| V12 Dependency depth | Same Comba order for view control; low-occupancy chain behavior |
| V13 Live state | Observe allocation changes; no causal inference from register count alone |
| V14 Operations | Same 64 product terms for routes 0/1/2; route 3 distinct |
| V15 Conversion | Are source split/copy operations eliminated or materialized in PTX? |
| V16 Latency | Small-grid per-work event average, not isolated instruction latency |
| V17 Throughput | Larger-grid event average; do not conflate with CPU parallelism |
| V18 Instructions / PMU | PTX static evidence; no invented SASS/counter measurements |
| V19 Cache / bandwidth | Known logical loads/stores; physical traffic unmeasured |
| V20 Code size | PTX bodies/entries separately from runtime machine code footprint |
| V21 Portability | One GPU, old-toolkit PTX JIT; CPU transfer remains unmeasured |
| V22 Compiler | Clone and view codegen normalization; host compiler qualification |
| V23 Policy / ties | Balanced paired rounds, all raw records, conservative near-tie decision |
| V24 Transfer | Local side probe only; preserve CPU field-first remaining sequence |
| V25 Prior art / novelty | Existing Comba/representation identities; no new-math claim |

## Workflow evidence

Manager bootstrap and health bound repo_id
repo_666797171f0141c58bf05f579b2ee16e, session
01a06be6-2904-7c62-9e7d-1245c34a5312. Session P6 handoff recovered;
task-specific Memory/KB/Context Graph searches returned zero additional context.
Source Graph located exact CUDA sources and was re-queried across boundaries.
NeedFix: worker-role MCP APIs remain unexposed; owner-suspended Task MCP local
workflow and exact manager handoffs retained, no role impersonation.
No injected project-context acknowledgement/HMAC tool was exposed; none invented.
New files require refresh and indexed review before accepting performance.
