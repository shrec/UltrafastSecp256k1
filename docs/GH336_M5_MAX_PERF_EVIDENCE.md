# Issue 336: Apple M5 Max performance evidence

## Status and scope

Issue 336 remains **open**. This runner is not reporter-equivalent hardware, so no
target-hardware performance measurement was run or inferred. All M5 Max results
below are explicitly pending an owner run.

The comparison is pinned to these immutable revisions:

- baseline `v3.68.0`: `a671ea2e3d355a26596d67d583ecf01252afd9d7`
- candidate `dev`: `3fbf1cf47fbc590c1c4570744f6195b6477d0377`

`10,360,000` is the fixed total tweak workload for each benchmark run. It is not
a tweaks-per-second target. The acceptance ceiling is the project's documented
normalized current-vs-v3.68 metric: candidate/baseline must be `<= 0.36x`.
Raw throughput is recorded separately and is not substituted for that metric.

## Immutable runner evidence and blocker

Captured on 2026-07-22 UTC:

```text
$ uname -a
Linux parking 6.8.0-134-generic #134-Ubuntu SMP PREEMPT_DYNAMIC Fri Jun 26 18:43:11 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
$ uname -s
Linux
$ uname -m
x86_64
$ uname -r
6.8.0-134-generic
$ getconf _NPROCESSORS_ONLN
16
$ git rev-parse HEAD
3fbf1cf47fbc590c1c4570744f6195b6477d0377
```

Required host: reporter-equivalent Apple M5 Max, ARM64, with at least 18 usable
threads. Actual host: Linux x86_64 with 16 logical processors. The OS,
architecture, processor count, and lack of an 18-thread capacity are all
disqualifying. Running the matrix here would not answer issue 336.

## Required matrix and evidence state

Every row uses a clean, non-LTO build and exactly 10,360,000 total tweaks.

| Revision | Threads | Normalized current/v3.68 | Raw throughput | wall | user | sys | peak RSS | profile |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `a671ea2e3d355a26596d67d583ecf01252afd9d7` | 1 | pending M5 Max owner run | pending | pending | pending | pending | pending | pending |
| `3fbf1cf47fbc590c1c4570744f6195b6477d0377` | 1 | pending M5 Max owner run | pending | pending | pending | pending | pending | pending |
| `a671ea2e3d355a26596d67d583ecf01252afd9d7` | 18 | pending M5 Max owner run | pending | pending | pending | pending | pending | pending |
| `3fbf1cf47fbc590c1c4570744f6195b6477d0377` | 18 | pending M5 Max owner run | pending | pending | pending | pending | pending | pending |

The normalized value is applicable to each candidate row after pairing it with
the baseline row at the same thread count. Baseline rows supply the denominator;
their normalized field remains pending until the paired calculation is recorded.

## M5 Max owner-run checklist

For reproducibility, preserve the literal commands and complete output alongside
the filled table; do not report a number without its command/output evidence.

- [ ] Record UTC time, `sw_vers`, `uname -a`, `uname -m`, `sysctl -n
  machdep.cpu.brand_string`, `sysctl -n hw.physicalcpu`, and `sysctl -n
  hw.logicalcpu`; confirm Apple M5 Max, Darwin ARM64, and 18 usable threads.
- [ ] Record compiler identity/version and the build tool versions. Disable
  Turbo/low-power variability where practicable and record power mode, thermal
  state, and other active workloads.
- [ ] Resolve `v3.68.0^{commit}` and `dev^{commit}` and verify they equal the two
  pinned hashes above. Use detached clean worktrees (or equivalent clean source
  trees), one per revision.
- [ ] Configure a release build for each revision with link-time optimization
  explicitly disabled. Save configure and build commands, configuration output,
  compiler/linker flags, and proof that no LTO flag or LTO artifact is present.
- [ ] For each revision, run the project's issue-336 benchmark with exactly
  `10,360,000` total tweaks at `1` thread, then at `18` threads. Do not interpret
  the workload as a rate. Save the exact benchmark command, stdout, and stderr.
- [ ] Run enough warm-ups and measured repetitions under the project's documented
  benchmark protocol. Preserve every sample and use its documented aggregation;
  do not select the best result.
- [ ] Wrap every measured run with `/usr/bin/time -l` and record raw throughput,
  wall time, user time, system time, and maximum resident set size. Retain units.
- [ ] Attribute system time: profile representative 1-thread and 18-thread runs
  with the project's macOS profiling procedure (for example, Instruments or
  `sample` when that is the documented procedure). Save the profile artifact and
  report syscall/kernel hotspots and their shares; do not fold `sys` into user or
  wall time.
- [ ] Compute the project's documented normalized current-vs-v3.68 metric for
  each thread count from the corresponding pinned-revision results. Record the
  formula, source samples, aggregation, units, and calculated value. Pass only
  when each required value is `<= 0.36x`.
- [ ] Attach raw logs, build/configuration records, timing output, RSS evidence,
  and profiles to issue 336, fill every pending cell above, and keep the issue
  open until the target-host evidence demonstrates resolution.

## Focused follow-on defect if the ceiling is missed

If either normalized candidate/baseline value is `> 0.36x`, open a focused defect
linked to issue 336 with this specification:

- title: `M5 Max non-LTO tweak benchmark exceeds 0.36x normalized ceiling`
- environment: immutable host/OS/architecture evidence, power/thermal state,
  compiler and build-tool versions, and both pinned revision hashes
- reproduction: exact clean non-LTO build commands and exact 10,360,000-total-
  tweak commands for both 1 and 18 threads
- evidence: every repetition, raw throughput, wall/user/sys, peak RSS, normalized
  formula and result, plus profiles that attribute system time
- expected: documented current-vs-v3.68 normalized metric `<= 0.36x` at both 1
  and 18 threads
- actual: the failing value(s), absolute and percentage distance above `0.36x`,
  and whether the regression is CPU, system-time, memory, or contention dominated
- done: reproduce, identify the dominant hotspot, land a fix, and rerun the full
  pinned matrix on the same M5 Max under the same protocol

Do not close issue 336 merely because the follow-on defect exists.
