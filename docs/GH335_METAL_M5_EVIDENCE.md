# GitHub issue 335 — Apple Metal M5 evidence

## Disposition

**KEEP ISSUE 335 OPEN.** This runner is not Apple Silicon and cannot provide Metal hardware evidence. No GPU acceptance test was run and no GPU result is inferred. The immutable blocker is the host identity captured below: Linux on `x86_64`, with an Intel CPU, and no Apple or Metal developer tools.

Evidence was captured at `2026-07-22T23:28:06Z` (UTC) on host `parking`.

## Host evidence

Commands were executed from the task worktree; their relevant, verbatim output follows.

```text
$ uname -a
Linux parking 6.8.0-134-generic #134-Ubuntu SMP PREEMPT_DYNAMIC Fri Jun 26 18:43:11 UTC 2026 x86_64 x86_64 x86_64 GNU/Linux
$ uname -s
Linux
$ uname -m
x86_64
$ uname -r
6.8.0-134-generic
$ getconf LONG_BIT
64
$ head -n 20 /etc/os-release
PRETTY_NAME="Ubuntu 24.04.4 LTS"
NAME="Ubuntu"
VERSION_ID="24.04"
VERSION="24.04.4 LTS (Noble Numbat)"
VERSION_CODENAME=noble
ID=ubuntu
ID_LIKE=debian
$ lscpu
Architecture:                            x86_64
CPU(s):                                  16
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Core(TM) i5-14400F
Thread(s) per core:                      2
Core(s) per socket:                      10
Socket(s):                               1
$ command -v system_profiler xcrun sw_vers metal  # checked individually
system_profiler: NOT FOUND
xcrun: NOT FOUND
sw_vers: NOT FOUND
metal: NOT FOUND
```

This is positive evidence of an Ubuntu/x86-64/Intel host and negative evidence for an executable Apple Metal toolchain on this runner. It is not evidence about behavior on an M5 Max.

## Issue-335 acceptance matrix on this runner

| Hardware criterion | Result | Evidence / blocker |
|---|---|---|
| Runner is real Apple Silicon (M5 Max) with Metal available | **FAIL — evidenced** | `uname` reports Linux/x86_64; `lscpu` reports Intel i5-14400F; Apple and Metal tools are absent. |
| Metal shader loading when launched from an unrelated current working directory | **NOT RUN** | Requires the real Apple Silicon Metal backend and its compiled shader/library artifacts. |
| CPU/GPU differential comparison on issue-335 vectors | **NOT RUN** | No Metal GPU backend can execute here. |
| Fail-closed behavior for missing, invalid, or unloadable Metal shader/library | **NOT RUN** | Metal initialization and error paths cannot be exercised on this host. |
| Reuse of one initialized Metal backend across repeated operations | **NOT RUN** | No Metal backend can be initialized here. |
| Metal/backend test suite on M5 Max | **NOT RUN** | Apple Silicon hardware, macOS, Xcode command-line tools, and Metal are unavailable. |

None of the five behavioral criteria passed on this runner. The only completed criterion is host qualification, which failed. Therefore there is no basis to close issue 335.

## M5 Max owner-run checklist

Run the following on a physical M5 Max macOS machine. Preserve the command lines, complete stdout/stderr, exit status, UTC timestamp, repository revision, and test artifacts for every item. Use the repository's documented build/test commands and record their exact resolved form; this report does not invent target names that were unavailable in the supplied task context.

- [ ] **Prove the host.** Record `date -u +%Y-%m-%dT%H:%M:%SZ`, `uname -a`, `uname -m`, `sw_vers`, `system_profiler SPHardwareDataType`, `system_profiler SPDisplaysDataType`, and `xcrun metal --version`. Confirm the hardware report identifies Apple M5 Max, the architecture is `arm64`, Metal support is present, and the run is not an emulated/non-Apple runner.
- [ ] **Freeze inputs.** Record the repository commit (`git rev-parse HEAD`), submodule state if applicable, compiler/Xcode version (`xcodebuild -version` and `xcrun --show-sdk-version`), build configuration, environment overrides, and the exact issue-335 vectors/seeds. Start from a clean build using the project's documented Metal-enabled configuration.
- [ ] **Baseline backend tests.** From the repository root, run the full documented test suite and the dedicated Metal/backend tests. Record exact commands, exit codes, pass/skip/fail counts, and confirm that Metal tests executed rather than skipped or silently falling back to CPU.
- [ ] **Unrelated-CWD shader loading.** Build the Metal-enabled target, create a temporary directory outside the repository/build tree, `cd` into it, and launch the built executable by absolute path with a known GPU vector. Record the unrelated `pwd`, executable path, shader/library resolution diagnostics, exit code, and result. It passes only if the intended packaged shader/library loads without relying on the launch CWD and the Metal backend actually executes.
- [ ] **CPU/GPU differential.** Execute identical issue-335 inputs through an explicitly selected CPU path and an explicitly selected Metal GPU path. Include boundary values, known vectors, deterministic randomized coverage, and repeated runs. Compare full outputs byte-for-byte (or by the project's documented tolerance), record any mismatch with its input/seed, and prove from logs/counters that each requested backend was used.
- [ ] **Fail closed.** In separate recoverable test copies, test a missing shader/library, a corrupt or invalid artifact, and an incompatible/unloadable artifact. For each, record stderr and nonzero exit/error status. It passes only if initialization/operation rejects the artifact clearly and deterministically, produces no purported GPU result, and does not silently fall back to CPU or stale embedded state.
- [ ] **Backend reuse.** Initialize one Metal backend/context, execute multiple sequential issue-335 operations through that same instance, and include at least one different input between repeated inputs. Capture initialization/destruction counters or equivalent diagnostics proving exactly one initialization and no per-operation recreation. Compare every result with CPU, then cleanly release the backend and run the project's leak/lifetime checks if available.
- [ ] **Repeat backend tests from unrelated CWD.** From the external temporary directory, run the dedicated backend test binary by absolute path, then the full documented suite if supported. Record pass/skip/fail counts and ensure all expected Metal tests execute.
- [ ] **Assess every row.** Mark each matrix row PASS or FAIL with a pointer to its raw evidence. Any skip, fallback, mismatch, unexpected success in a fail-closed case, missing reuse proof, or absent raw log is a failure for closure purposes.
- [ ] **Disposition.** Close issue 335 only if host qualification and every behavioral row pass on real M5 Max hardware. Otherwise keep it open and attach the failing command, input/seed, complete diagnostics, environment, and artifact hashes.

## Evidence policy

This document deliberately reports only observations made on the current runner. The owner-run items are instructions, not results. A checklist mark must not be changed to PASS without retaining the corresponding raw M5 Max evidence.
