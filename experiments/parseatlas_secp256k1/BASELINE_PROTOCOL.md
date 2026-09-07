# L0 native baseline protocol

This is a preregistered measurement protocol, not a performance result or a
ranking of eligible worlds. The owner specifically requests measured baselines
before comparing methods: additional arithmetic may be worthwhile when it
improves locality or reduces memory traffic. Static operation count, dependency
depth, elapsed time and logical traffic therefore remain separate observables.

## Fixed scope and correctness gate

The only arithmetic reference is the actual `detail::add64` header at engine
commit `fef231d4e4173bd016fb2a3a1eff67087396a203`, SHA-256
`cc95a42137df602eaee3a560993b8c59c091e017364a0345be486ba012de205d`.
The collector compares current header bytes with both this hash and the exact
frozen git blob before building and after collection. It records the active
engine commit but does not silently establish a newer baseline. The harness and
runner source hashes, compiler version/target/arguments and compiled binary hash
are recorded separately. Redirected source paths and compiler/runtime include or
loader environment overrides are rejected.

The native baseline compiles with GCC `g++`, `-std=c++17 -O3 -march=native
-DNDEBUG -Wall -Wextra`, without LTO. It is one- or four-limb unsigned addition
with carry-in and carry-out, not Fp/Fn reduction or a complete engine benchmark.
Four-limb addition is a harness composition of the original primitive, not a
claim that a production modular-arithmetic call site uses this exact loop.

Before sustained timing, strict C++ check input covers both incoming carry bits,
cross-products of boundary values (including every 64-bit limb boundary), plus
512 seeded random triples per width by default. Python's independent unbounded
integer sum is split into fixed-width little-endian bytes and a separate carry;
both must match the actual C++ result. Invalid, missing or extra output fields
fail collection. Exact ASCII input corpus bytes and decoded output bytes plus
carry have SHA-256 identities. Agreement is finite **OBSERVED** evidence, not a
full-domain proof, cryptographic constant-time evidence or production acceptance.

The timed workload generator is independently checked on small fixed cases for
both kernels against whole-integer arithmetic and the registered SplitMix64,
conditioning and checksum contract. Its seeded initialization is separate from
the explicit correctness corpus. This does not claim that all outputs of the
large timed streaming dataset received an independent bigint comparison.

## Workloads and units

Six baselines are collected: one and four 64-bit limbs, each in three regimes.

| Regime | Work and capacity selection | Interpretation |
|---|---|---|
| Loop-carried | 4096 dependent steps per pass, calibrated passes | Conditioned recurrence, not isolated add instruction latency |
| Hot bulk | Complete input/output records occupy at most half the observed target CPU's L1 data cache | Same bulk kernel as streaming, small working set |
| LLC-exceeding bulk | Complete records occupy at least four times observed target CPU LLC capacity | Large working-set experiment, not proof of a DRAM-bound kernel |

The bulk input record contains `a[L]`, `b[L]`, and a 64-bit carry slot; output is
`low[L]` and a 64-bit carry slot. Complete record strides sum to 40 bytes for one
limb and 112 bytes for four. These are **logical** bytes. A compiler may load only
one byte of a carry slot, and cache-line fetches, write allocation, eviction,
prefetching, register allocation and spills differ from record accounting.

The latency recurrence feeds the output limbs and carry into the next step and
updates each `b[j]` to `rotl64(b[j],13) XOR new_a[j]`. This conditioning prevents a
trivial loop-invariant workload but adds real overhead. Logical streamed bytes
are zero because there is no explicit bulk array stream; this does **not** mean
zero machine memory traffic. Four-limb compiled latency code can spill to the
stack. Reported logical state size is not measured stack footprint.

The denominator is one complete logical addition, with `L` original `add64`
calls per addition. Report `ns/logical_addition`, retaining the call count. Bulk
logical GB/s is logical bytes per addition divided by ns per addition; decimal
GB/s follows from bytes/ns. It is neither measured DRAM bandwidth nor a memory
controller counter. Do not turn it into an actual bandwidth claim.

## Calibration, locality and environmental controls

The selected CPU must be in the caller's allowed affinity. Each child benchmark
process is pinned before allocating/initializing data, so its default first-touch
placement happens on that CPU. No affinity of the owner shell, governor, turbo,
sysctl, security setting or global machine state is changed. Linux sysfs cache
topology and sharing, allowed CPUs, observed CPU count and available memory are
recorded. Working memory is capped at the smaller of 256 MiB and one sixteenth of
observed available memory; failure to fit four times LLC is an explicit resource
failure, not a silently relabeled smaller dataset.

Timing runs **serially** even though workloads are independent: concurrent
benchmarks would alter the cache/frequency/bandwidth conditions being measured.
The measurement worker count is one with this reason recorded, rather than a
claim to test multicore throughput. Other processes, sibling activity, turbo and
thermal state are not controlled. CPU0 on the currently observed hybrid host is
a performance core, with logical sibling1; no portability to efficiency cores is
implied. Per-sample and overall load/frequency/governor snapshots accompany raw
measurements, but they do not establish a quiet or fixed-frequency machine.

The initial order is seeded once, before reading any elapsed results. Subsequent
rounds rotate it; every workload occupies every order position in the first six
rounds. Calibration starts at one pass and targets at least 50 ms, with a maximum
of eight calibration trials, one million passes and one billion operations.
At least two full external warmups and seven measured repetitions follow, with
the same final pass count and seed per workload. The C++ probe also has an
untimed internal warmup on each fresh process. Calibration, warmups and every
measured sample are retained separately; no outlier trimming or best-run choice.
Samples that later fall below the calibration target are retained and counted.

Allocation, input initialization, internal warmup, full output checksum and
process startup are outside the C++ elapsed interval. Bulk pass barriers require
repeated stores to remain observable to the optimizer. The final checksum is
checked for repeatability; independent assembly inspection is still required to
validate the intended compiled loop. A checksum is not a substitute for the
pre-timing independent correctness gate.

Summaries include median, minimum, maximum, median absolute deviation and
inclusive interquartile range of per-sample ns/addition. These describe spread,
not a confidence interval or cross-machine uncertainty estimate. No automatic
speed threshold accepts or rejects a method.

PMU counters are `NOT_MEASURED` by this collector. Separate owner-authorized
whole-process `perf` diagnostics must remain separately labeled: they include
initialization/checksum/process work and are not inner-loop counters. No sudo or
credential handling is implemented here.

## Reproduction and next comparison

Run from the repository root. Collection prints one JSON document to stdout;
durable result files are persisted by the manager after review. The collector
retains only compiler-generated binaries in a unique `/tmp/parseatlas-l0-baseline-*`
directory and reports their exact paths for disassembly or separate diagnostics.

```sh
.venv/bin/python -P -m pytest -q experiments/parseatlas_secp256k1/tests/test_baseline_runner.py
.venv/bin/python -P experiments/parseatlas_secp256k1/baseline_runner.py --check --cpu 0
.venv/bin/python -P experiments/parseatlas_secp256k1/baseline_runner.py --measure --cpu 0
```

Use the sustained command only in the declared quiet measurement window.
Future serial/prefix/carry-save or layout comparisons must preserve the exact
boundary contract, corpus, compiler settings, selected CPU and working-set
regime, while adding their own source/build identity, correctness and optimizer
checks. Representation conversions and complete region costs require their own
measured boundaries. Extra operations may then be justified by measured locality
or time, but causal cache/bandwidth claims require the relevant counter evidence.
