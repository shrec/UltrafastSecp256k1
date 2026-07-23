# GPU batch-path policy advisory API

Status: implementation specification for `GPU_BATCH_POLICY_API_012`.

## Purpose and boundaries

The library needs one cheap, caller-facing query that answers two different
questions without conflating them:

1. Which execution path would this process select now?
2. Is batching on that path predicted to improve the supplied workload?

The answer is advisory. It must expose unavailable or uncalibrated states
instead of treating backend presence as proof of selection or profitability.
It must not hard-code a batch-size crossover from one development machine.

This policy applies to deferrable signature verification. Per-block identity
hashing remains on the block-processing path. Signature-hash work is performed
only for conditionally traversed scripts and is not generally a cross-block
batch candidate. A caller may therefore accumulate signature-verification
items across blocks when its validation contract permits, but must not infer
that block identity or all signature-hash traversal can also be deferred.

## Smallest stable public surface

Expose one versioned value query; names below are illustrative C/C++ API names
and may be adapted to existing project naming conventions:

```c
#define LB_GPU_BATCH_POLICY_ABI_V1 1u

typedef enum {
    LB_BATCH_OP_ECDSA_VERIFY = 1,
    LB_BATCH_OP_SCHNORR_VERIFY = 2,
    LB_BATCH_OP_GENERIC_MSM = 3
} lb_batch_operation;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;
    lb_batch_operation operation;
    uint64_t item_count;
    uint32_t concurrency;
} lb_gpu_batch_policy_request_v1;

typedef enum {
    LB_BACKEND_NONE = 0,
    LB_BACKEND_CPU = 1,
    LB_BACKEND_CUDA = 2,
    LB_BACKEND_OPENCL = 3,
    LB_BACKEND_METAL = 4,
    LB_BACKEND_OTHER_GPU = 5
} lb_compute_backend;

typedef enum {
    LB_PATH_UNAVAILABLE = 0,
    LB_PATH_CPU_INLINE = 1,
    LB_PATH_CPU_BATCH = 2,
    LB_PATH_GPU_BATCH = 3
} lb_batch_path;

typedef enum {
    LB_ADVICE_UNKNOWN = 0,
    LB_ADVICE_NOT_BENEFICIAL = 1,
    LB_ADVICE_BENEFICIAL = 2
} lb_batch_advice;

typedef enum {
    LB_CALIBRATION_NONE = 0,
    LB_CALIBRATION_MEASURED_CURRENT = 1,
    LB_CALIBRATION_MEASURED_CACHED = 2
} lb_calibration_state;

typedef struct {
    uint32_t struct_size;
    uint32_t abi_version;

    uint64_t available_backends;       /* capability bits, not selection */
    lb_compute_backend selected_backend;
    lb_batch_path selected_path;

    lb_calibration_state calibration;
    lb_batch_advice predicted_benefit;
    uint8_t confidence_percent;         /* 0 when unknown */
    uint8_t reserved[7];

    uint64_t predicted_inline_ns;
    uint64_t predicted_selected_ns;
    uint64_t calibration_age_ms;
    uint64_t calibration_generation;
} lb_gpu_batch_policy_result_v1;

bool lb_gpu_batch_policy_query_v1(
    const lb_gpu_batch_policy_request_v1* request,
    lb_gpu_batch_policy_result_v1* result);
```

The required inputs are operation, item count, and expected simultaneous
callers/streams (`concurrency`). Hardware and memory topology are process
observations owned by the library, not caller-supplied claims. This keeps the
hot query small and prevents mismatches between the requested topology and the
device actually selected. A future API may add workload-specific constraints
through a larger `struct_size`.

The output deliberately separates:

- `available_backends`: what initialized successfully;
- `selected_backend` and `selected_path`: what would actually execute this
  request under current policy;
- `calibration` and age/generation: whether the estimate is measured and
  current;
- `predicted_benefit`, predicted times, and confidence: the profitability
  estimate, which is not a capability statement.

The query must be allocation-free, must not initialize a device or benchmark,
and should be a snapshot read of immutable/atomically published calibration
data. Startup calibration owns expensive discovery and measurement.

### Versioning and compatibility

Both structures begin with `struct_size` and `abi_version`. Version 1 accepts
only `LB_GPU_BATCH_POLICY_ABI_V1`. Implementations read no bytes beyond the
smaller of their known size and the caller's size, zero all output bytes made
available by the caller before filling fields, and reject a missing required
prefix. New fields append to structures; enum numeric values never change.
Semantic changes that reinterpret existing fields require a new ABI function
suffix/version.

### Fail-closed behavior

The function returns `false` for invalid pointers, unsupported ABI, undersized
required prefixes, unknown operations, zero item count, zero concurrency,
overflow, or an inconsistent calibration record. On `false`, a writable result
is zeroed as far as safely described by its validated size.

Missing, stale, or nonmatching calibration is not guessed. The query returns
`true` with `calibration = NONE`, `predicted_benefit = UNKNOWN`, confidence
zero, and a safe `CPU_INLINE` path when CPU inline is available; otherwise it
selects `UNAVAILABLE`. Unknown advice must never enable batching. A GPU being
available must never by itself select `GPU_BATCH`.

## Calibration and crossover cache

Calibration runs after backend/device initialization and before advisory
queries are expected, preferably once at startup with a bounded budget. It
publishes a new generation atomically. Failure preserves safe inline behavior,
not a partial curve.

Measure end-to-end wall time for each supported operation over representative
batch-size buckets and concurrency levels:

- CPU inline: verification in the capture/validation thread;
- CPU batch: capture/enqueue, synchronization, and batch execution;
- GPU batch: capture/enqueue, host preparation, transfers or shared-memory
  synchronization, kernel execution, and completion synchronization.

Use warm-up runs, multiple timed samples, robust statistics (for example,
median plus dispersion), and held-out confirmation around each crossover.
Store measured curves or piecewise interpolation inputs rather than a universal
threshold. A crossover is usable only when the candidate beats its baseline by
a configured safety margin larger than observed noise; that margin is policy
configuration/calibration metadata, not part of the public ABI.

Topology-specific treatment is mandatory:

- **Discrete PCIe GPU:** include host allocation/pinning, host-to-device and
  device-to-host traffic, launch, queueing, and synchronization. Record the
  link identity/capability and transfer regime used.
- **Unified-memory GPU:** do not charge a synthetic PCIe copy, but include
  mapping, cache/coherency, page migration where applicable, queueing, and
  synchronization. Record unified-memory mode and relevant memory properties.
- **CPU fallback:** independently compare CPU batch against CPU inline,
  including additive capture/queue overhead. CPU batch is eligible only above
  its measured crossover; it may be slower at every measured size, in which
  case it is never advised.

The cache contains per operation/backend/path/topology/concurrency bucket:
sample sizes, sample count, central latency and dispersion, confirmed crossover
interval (or “none observed”), safety margin, calibration timestamp,
generation, and calibration schema/algorithm version. Predictions outside the
measured range return unknown rather than extrapolate.

Invalidate or segregate cached data on every key that can materially change the
curve:

- calibration schema and benchmark algorithm version;
- library build/version and relevant compile features;
- operation and cryptographic implementation/algorithm revision;
- selected backend, backend runtime/driver version, and kernel/binary identity;
- device vendor, stable device/architecture ID, compute capability, device
  count, and clock/power profile when observable;
- CPU vendor/model/ISA feature set, logical/physical core topology, and thread
  pool configuration;
- memory topology (discrete versus unified), memory mode, relevant capacity or
  bandwidth class, PCIe/link generation and width for discrete devices;
- OS/kernel and relevant runtime versions;
- concurrency bucket, affinity policy, and configured safety margin.

If a key cannot be obtained reliably, omit disk-cache reuse for that device
rather than weakening the match. Runtime health events such as device loss,
backend errors, power/thermal regime changes, or repeated prediction misses
invalidate the in-memory generation and force unknown/inline advice until
recalibration. Normal calibration age may lower confidence; exceeding the
configured maximum age invalidates profitability advice.

## Selection and profitability semantics

Selection evaluates eligible calibrated candidates and returns the lowest
predicted end-to-end time that clears its measured safety margin. It does not
select a batching path merely because that path exists.

For CPU batching, the comparison is:

```text
T_cpu_batch_total(item_count, concurrency)
    < T_cpu_inline(item_count, concurrency) - safety_margin
```

Capture overhead is additive and included in `T_cpu_batch_total`. If the
inequality is not supported by current measurements, CPU inline is selected.

GPU verification is treated as offload onto otherwise idle capacity, but it is
useful to the capture pipeline only when completion stays ahead of incoming
work. Calibration therefore records end-to-end GPU completion time and the
runtime policy compares it with the observed or configured interval until the
next captured batch. GPU advice is beneficial only when both conditions hold:

```text
T_gpu_end_to_end < T_replaced_or_overlapped_cpu - safety_margin
T_gpu_end_to_end < next_batch_capture_interval - headroom
```

The version-1 request does not carry a caller-guessed capture interval. The
runtime maintains a conservative rolling interval per operation/concurrency
bucket. Until enough observations exist, or when input rate is unstable,
GPU profitability is unknown and batching stays disabled. Confidence derives
from sample count, dispersion, distance from crossover, age, prediction error,
and capture-interval stability; it is never a synonym for backend availability.

## Cryptographic applicability

No performance result is asserted by this specification; every profitability
decision must come from measurements on the matched system.

Standard ECDSA verification does **not** use Pippenger in the current batching
design. Its batch acceleration is batch inversion plus per-signature
GLV/Strauss work. Pippenger applies to Schnorr verification and generic
multi-scalar multiplication workloads where an MSM formulation is valid.
Calibration and cache records must therefore be operation-specific and must
not transfer a Schnorr/MSM result to ECDSA.

## Required benchmarks and tests

### Calibration benchmarks

1. For every supported operation, sweep batch sizes below, around, and above
   observed crossovers at concurrency 1 and representative higher concurrency.
   Report raw sample metadata and dispersion; do not publish an unverified
   throughput claim.
2. On CPU-only/fallback configurations, benchmark inline and CPU batch
   end-to-end, including capture. Verify a “no crossover observed” cache entry
   when batching never wins.
3. On discrete GPUs, benchmark with actual transfer and synchronization costs.
   On unified-memory GPUs, benchmark the shared-memory/coherency path separately
   and verify it does not reuse a discrete-GPU cache entry.
4. Run a sustained producer/consumer benchmark using realistic capture
   intervals. Assert GPU completion latency remains below the next-batch
   interval with configured headroom over the full steady-state window; also
   test an overload rate where advice becomes unknown/not beneficial or the
   path falls back before backlog grows without bound.

### Automated policy tests

- With CPU batch slower than inline in all calibrated buckets, assert every
  query selects `CPU_INLINE` and never reports beneficial CPU batching.
- With a measured crossover, test buckets on both sides and within the noisy
  confidence band; the band must fail closed to inline/unknown.
- Make GPU available but uncalibrated and assert availability is reported while
  selection remains inline and benefit unknown.
- Supply a GPU curve faster than CPU but slower than the capture interval and
  assert GPU batching is not enabled; supply a curve with confirmed headroom
  and assert it may be selected.
- Exercise capture-rate jitter, concurrency changes, stale calibration, device
  loss, and prediction misses; each must reduce confidence or invalidate advice
  as specified.
- Verify cache hit/miss behavior for every invalidation-key class, especially
  discrete versus unified memory, driver/kernel identity, CPU ISA, operation,
  concurrency, and calibration schema.
- Verify invalid inputs, structure truncation/extension, reserved-byte zeroing,
  enum stability, overflow, and concurrent snapshot reads during generation
  replacement.
- Verify ECDSA, Schnorr, and generic MSM records cannot alias, and metadata
  identifies ECDSA batch inversion plus GLV/Strauss versus Pippenger-eligible
  Schnorr/generic MSM.

## Incremental implementation plan (no code changes in this task)

1. **Public types and query contract:** add the versioned request/result enums
   and query declaration in the project's existing public API headers; add
   API reference text and a caller example in the existing public docs.
2. **Topology and backend snapshot:** extend the existing backend/device
   discovery area to publish available backends, actual selected backend/path,
   discrete/unified memory topology, and stable invalidation identities.
3. **Calibration engine:** add startup CPU-inline, CPU-batch, and GPU
   end-to-end measurements beside the existing batch dispatch/benchmark
   facilities. Produce immutable per-operation/concurrency curves and
   uncertainty metadata.
4. **Cache:** add a versioned crossover-cache serializer/loader beside existing
   runtime configuration/cache support. Validate the complete key before
   publishing a cached generation; use atomic replacement and fail closed on
   parse, key, or integrity errors.
5. **Runtime advisory:** implement the allocation-free query beside current
   backend selection/dispatch policy. Add rolling capture-interval and
   prediction-error observations at existing batch capture/completion points;
   those observations update policy state, never the public ABI.
6. **Unit tests:** place ABI, fail-closed, curve selection, invalidation,
   operation-separation, and concurrency tests in the existing API/policy test
   suites.
7. **Integration/performance tests:** extend existing CPU/GPU batch benchmarks
   with inline-versus-batch capture cost, discrete/unified-memory cases, and a
   sustained capture-ahead test. Gate enablement on measured inequalities, not
   a fixed throughput number.
8. **Documentation:** update backend selection, batching, configuration/cache,
   and benchmarking documentation with availability-versus-selection,
   calibration provenance, confidence meaning, and the ECDSA/Pippenger
   distinction.

Before implementation, Source Graph should resolve the repository's exact
header, backend-dispatch, benchmark, cache/configuration, and test filenames.
The present research task's bounded Source Graph query returned no matches, so
this report intentionally names exact likely API/docs/test **areas** rather
than inventing unverified file paths.
