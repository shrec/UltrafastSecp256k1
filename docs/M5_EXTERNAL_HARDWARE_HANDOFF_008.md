# Apple M5 external-hardware handoff for issues 335 and 336

## Status and ownership

Issues **335 and 336 remain open** and are classified as
**external-hardware validation**. This environment has no Apple M5/Metal
hardware: the recorded runner is Ubuntu Linux on `x86_64`, with an Intel
processor, no Apple/Metal developer tools, and only 16 logical processors.
Consequently, no Apple M5 Max, Metal, or reporter-equivalent performance test
was run here. This handoff makes no local pass claim and provides no basis for
closing either issue.

A qualified external owner needs physical Apple M5 Max hardware running macOS.
Issue 335 additionally requires Xcode command-line tools and a functioning Metal
toolchain/backend. Issue 336 requires at least 18 usable threads and a controlled
environment suitable for repeatable performance measurement.

## Existing acceptance checklists

The external owner must execute and complete the existing checklists rather than
treating this handoff as test evidence:

- Issue 335: `docs/GH335_METAL_M5_EVIDENCE.md`, section **M5 Max owner-run
  checklist**, with its **Issue-335 acceptance matrix** and **Evidence policy**.
- Issue 336: `docs/GH336_M5_MAX_PERF_EVIDENCE.md`, sections **M5 Max owner-run
  checklist**, **Required matrix and evidence state**, and **Focused follow-on
  defect if the ceiling is missed**.

## Issue 335 — Metal validation

### Required commands and execution

Preserve the literal commands, complete stdout/stderr, exit status, UTC
timestamp, repository revision, and artifacts for each checklist item.

1. Prove the physical host and Metal toolchain with:
   `date -u +%Y-%m-%dT%H:%M:%SZ`, `uname -a`, `uname -m`, `sw_vers`,
   `system_profiler SPHardwareDataType`,
   `system_profiler SPDisplaysDataType`, `xcrun metal --version`,
   `git rev-parse HEAD`, `xcodebuild -version`, and
   `xcrun --show-sdk-version`. Record submodule state if applicable.
2. Use the repository's documented clean, Metal-enabled build and test commands
   and record their exact resolved forms, configuration, environment overrides,
   exit codes, and pass/skip/fail counts. Confirm Metal tests execute without CPU
   fallback.
3. Build the Metal target; create a temporary directory outside the repository
   and build tree; `cd` there; record `pwd`; then run the built executable and
   dedicated backend test binary by absolute path. Run the full documented suite
   there too if supported.
4. Run identical issue-335 vectors through explicitly selected CPU and Metal
   paths, including boundary values, known vectors, deterministic randomized
   seeds, and repeated runs. Compare full outputs byte-for-byte or with the
   project's documented tolerance.
5. In recoverable test copies, separately exercise missing, corrupt/invalid, and
   incompatible/unloadable shader or library artifacts.
6. Reuse one initialized Metal backend/context for multiple sequential
   operations, including changed input, then cleanly release it and run available
   leak/lifetime checks.

### Required evidence and closure

Evidence must prove Apple M5 Max, `arm64`, and Metal availability; packaged
shader/library loading independent of launch CWD; actual CPU and GPU backend
selection; matching results; deterministic fail-closed behavior with a nonzero
status and no CPU fallback or purported GPU result; exactly one backend
initialization across repeated operations; clean release; and all expected Metal
tests executed. Include raw logs, inputs/seeds, diagnostics or counters, artifact
hashes, and a PASS/FAIL evidence pointer for every acceptance-matrix row.

Close issue 335 only when host qualification and every behavioral row pass on a
physical M5 Max. A skip, fallback, mismatch, unexpected fail-closed success,
missing reuse proof, or missing raw log is a closure failure; retain the issue
open with the failing command and complete evidence.

## Issue 336 — M5 Max performance validation

### Required commands and execution

Preserve literal commands and complete output. The immutable comparison is:

- baseline `v3.68.0`: `a671ea2e3d355a26596d67d583ecf01252afd9d7`
- candidate `dev`: `3fbf1cf47fbc590c1c4570744f6195b6477d0377`

The workload is exactly `10,360,000` total tweaks per run; it is not a
tweaks-per-second target.

1. Prove the host with `date -u +%Y-%m-%dT%H:%M:%SZ`, `sw_vers`, `uname -a`,
   `uname -m`, `sysctl -n machdep.cpu.brand_string`,
   `sysctl -n hw.physicalcpu`, and `sysctl -n hw.logicalcpu`. Confirm physical
   Apple M5 Max, Darwin ARM64, and at least 18 usable threads.
2. Resolve `v3.68.0^{commit}` and `dev^{commit}` and verify the exact hashes
   above. Use separate detached clean worktrees or equivalent clean source trees.
3. Run the repository's documented release configuration/build command at each
   revision with LTO explicitly disabled. Retain commands, compiler/linker flags,
   configuration output, tool versions, and proof that no LTO flag or artifact
   is present.
4. Run the documented issue-336 benchmark for both revisions with exactly
   `10,360,000` total tweaks at 1 thread and 18 threads. Use the documented
   warm-up, repetition, and aggregation protocol and retain every sample.
5. Wrap every measured run with `/usr/bin/time -l`. Profile representative
   1-thread and 18-thread runs with the project's documented macOS procedure,
   such as Instruments or `sample` when applicable.

### Required evidence and closure

Record power mode, thermal state, other workloads, compiler/build-tool identity,
exact build and benchmark commands, stdout/stderr, exit status, every repetition,
raw throughput, wall/user/system time, peak RSS with units, and profile artifacts
attributing syscall/kernel hotspots and shares. Fill every cell in the existing
four-row matrix. For each thread count, record the documented normalized
current-vs-v3.68 formula, source samples, aggregation, units, and candidate-to-
baseline result.

Close issue 336 only if the complete reporter-equivalent M5 Max evidence shows
the normalized candidate/baseline metric is `<= 0.36x` at both 1 and 18 threads.
If either result is `> 0.36x`, keep issue 336 open and open the precisely
specified focused follow-on defect from the existing checklist; creating that
defect is not grounds to close issue 336. Missing cells, artifacts, or protocol
evidence also prevent closure.

## Exact post-ready GitHub issue updates

Coordinator review text for issue 335:

```text
External-hardware validation handoff is ready in `docs/M5_EXTERNAL_HARDWARE_HANDOFF_008.md`.

Issue 335 remains open and is classified as external-hardware validation. The current environment is Ubuntu/x86_64 on Intel hardware and has no Apple M5 Max or Metal toolchain, so no Apple/Metal test was run and no pass or closure is claimed.

Qualified external owner requested: run the complete `docs/GH335_METAL_M5_EVIDENCE.md` M5 Max owner-run checklist on a physical Apple M5 Max with macOS, Xcode command-line tools, and Metal. Attach the exact commands, complete stdout/stderr, exit statuses, host/toolchain evidence, inputs and seeds, artifact hashes, backend-selection and reuse diagnostics, and a PASS/FAIL evidence pointer for every acceptance-matrix row. Close only if host qualification and every behavioral row pass; otherwise keep this issue open with the failing evidence.
```

Coordinator review text for issue 336:

```text
External-hardware validation handoff is ready in `docs/M5_EXTERNAL_HARDWARE_HANDOFF_008.md`.

Issue 336 remains open and is classified as external-hardware validation. The current environment is Ubuntu/x86_64 with 16 logical processors, not reporter-equivalent Apple M5 Max hardware with at least 18 usable threads, so no target-hardware performance run was made and no pass or closure is claimed.

Qualified external owner requested: run the complete `docs/GH336_M5_MAX_PERF_EVIDENCE.md` M5 Max owner-run checklist on a physical Apple M5 Max. Compare baseline `a671ea2e3d355a26596d67d583ecf01252afd9d7` with candidate `3fbf1cf47fbc590c1c4570744f6195b6477d0377` in clean non-LTO builds, using exactly 10,360,000 total tweaks at 1 and 18 threads. Attach literal commands and full outputs, every sample, wall/user/sys, peak RSS, profiles, the documented normalized formula, and all four completed matrix rows. Close only if the complete evidence shows candidate/baseline <= 0.36x at both thread counts; otherwise keep this issue open and follow the checklist's focused-defect procedure.
```
