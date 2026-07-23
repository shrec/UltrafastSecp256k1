# Issue #336: M5 Max profiling request

## Evidence and limits

Craig Raw reported the following measurements on 2026-07-23, on a physical
Apple M5 Max, for dev `4556fa05`:

- `v3.68 per-row real 12.8s/user 194s/sys 1.0s`
- `v4.5 per-row 15.9s/221s/8.8s`
- `dev per-row 15.6-15.7s/214-216s/8.6-9.7s`
- `dev batch API 17.8-18.6s/218-229s/43s`
- batch calls contain about `300K rows per call`, with `18 concurrent callers`

Those are reporter-supplied data points, not results reproduced locally. The
only inference used to prioritize the first request is that the much larger
reported system time in the dev batch case (43s versus 8.6-9.7s for dev
per-row) makes kernel/VM/scheduling evidence the most discriminating first
capture. It does not establish a cause. No Apple hardware result is claimed
here.

## Controlled experiment matrix

Use the existing, unmodified invocation that produced each roughly 15-second
measurement. Keep the full 10,356,829-row workload, the same input, 18
concurrent callers, build type/options, machine, power mode, and other workload
and thread settings identical. Do not add a start barrier, pause the process, or
add per-call logging/output. Run one unprofiled confirmation immediately before
the profiled run.

| Case | Revision/API | First request | Evidence-driven follow-up |
|---|---|---:|---:|
| A | v3.68 steady-state old/per-row API | no | yes |
| B | current dev `4556fa05`, old/per-row API | yes | — |
| C | current dev `4556fa05`, batch API (about 300K rows/call) | yes | — |

The first comparison is B versus C because it holds the current code revision
fixed. Run B then C, and if affordable repeat in C then B order to expose order
or thermal effects. Record the command's existing real/user/sys output in
`GH336-M5-B-old-time.txt` and `GH336-M5-C-batch-time.txt`; the profiler run
itself need not reproduce the exact unprofiled duration.

## Minimal first capture: B and C

For each case, start the existing invocation normally. While it is running,
obtain its PID (`pgrep -n -x <actual-executable-name>` is sufficient when the
name is unambiguous), then run:

```sh
sample <PID> 15 -file GH336-M5-B-old-sample.txt
xctrace record --template 'System Trace' --attach <PID> --time-limit 15s \
  --output GH336-M5-B-old-system-trace.trace
```

Use the corresponding C names:

```sh
sample <PID> 15 -file GH336-M5-C-batch-sample.txt
xctrace record --template 'System Trace' --attach <PID> --time-limit 15s \
  --output GH336-M5-C-batch-system-trace.trace
```

`sample` and System Trace are separate runs so their overhead does not overlap.
If PID lookup is awkward, open Instruments, choose **System Trace**, select the
already-running process as the attach target, record until it exits (or about
15 seconds), and save with the exact `.trace` name above. Attaching after a
short-lived process starts can miss startup; that limitation should be noted,
not worked around by changing the invocation.

In each System Trace, inspect the 18 worker threads over the common active
interval and report:

- running versus runnable versus blocked/waiting time, plus the dominant
  blocked/wait stack and CPU stack;
- syscall duration/count by syscall, especially `mmap`, `munmap`, and related
  VM calls;
- lock/condition-wait duration and representative stack;
- thread-creation events/count and representative creation stack;
- whether the expensive activity is clustered near process/call startup or
  persists throughout the capture.

This first pair localizes CPU execution, scheduler/blocked states, syscalls,
VM mapping, synchronization, and thread creation. The requested shared
artifacts are only the two `*-sample.txt` files and a short summary or
screenshots of the bullets above. The `.trace` packages may contain local
paths or symbols and need not be uploaded.

## Follow-up only if the first capture warrants it

Run A with identical workload/thread settings and produce
`GH336-M5-A-v368-time.txt`, `GH336-M5-A-v368-sample.txt`, and
`GH336-M5-A-v368-system-trace.trace` only when B/C suggests CPU arithmetic
regression or when a v3.68 baseline is needed to distinguish old-API revision
effects. Use the same `sample` and System Trace commands with A's names.

Run Instruments **Allocations** as a separate B/C pair only when sample/System
Trace shows allocator or VM activity. Attach to the normally launched process,
record its remaining run, save
`GH336-M5-B-old-allocations.trace` and
`GH336-M5-C-batch-allocations.trace`, and report allocation count/bytes,
transient versus persistent allocations, and the top allocation stacks.
Together with the System Trace `mmap`/`munmap` events, this tests allocator
pressure without assuming per-call instrumentation exists.

Per-call warmup/scratch behavior cannot be measured directly without changing
the program. Infer it only when existing symbols/stacks permit: compare whether
setup, allocation, mapping, thread creation, or waits are bursty near the
beginning versus steady over the capture, and label that conclusion as an
inference. Do not request new per-call output from Craig.

## Decision thresholds for the next engineering task

Use B versus C over the common active interval. Assign the next task only when
one category explains at least 50% of the extra sampled CPU or extra
blocked/syscall time in C, and C is at least 25% worse than B in that category:

- **Per-call setup:** setup/warmup/scratch stacks are bursty and meet the
  50%/25% rule; target reuse/hoisting of call-local state.
- **Allocator pressure:** allocation/free or `mmap`/`munmap` stacks meet the
  rule; confirm with the Allocations follow-up, then target scratch reuse or
  allocation reduction.
- **Scheduling/locking contention:** runnable-but-not-running time or
  lock/condition waits meet the rule; target the identified lock, wakeup, or
  worker scheduling path.
- **Arithmetic regression:** useful running CPU stacks in the row computation
  meet the rule while allocation/VM/wait evidence does not; then collect A and
  compare A versus B under the identical matrix before targeting the arithmetic
  path.

If no category meets both thresholds, do not pick a cause from these data;
request a second ordered B/C pair or a narrower capture around the dominant
observed stack.

## Concise GitHub-ready ask

> Craig, could you run the same unmodified ~15-second invocation twice on
> current dev `4556fa05`: (B) old/per-row API and (C) batch API, keeping the
> full 10,356,829 rows, 18 callers, input, and build/settings identical? For
> each, please attach `sample` for 15s and make a separate 15s Instruments
> System Trace. The commands and exact names are:
> `sample <PID> 15 -file GH336-M5-{B-old|C-batch}-sample.txt` and
> `xctrace record --template 'System Trace' --attach <PID> --time-limit 15s
> --output GH336-M5-{B-old|C-batch}-system-trace.trace`.
> Please share the two sample text files, real/user/sys output, and a short
> summary or screenshots of thread states, top CPU/wait stacks, syscall time
> (including mmap/munmap), locks/condition waits, and thread creation. No input
> data or `.trace` package is needed. We can request v3.68 or Allocations only
> if this first comparison points there.
