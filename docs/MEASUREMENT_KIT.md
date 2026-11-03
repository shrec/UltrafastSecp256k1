# CPU variant measurement kit

`tools/measure_cpu_variants.py` measures, on the machine in front of you, how
`bench_unified` behaves with each CPU dispatch variant of the library, and writes
one JSON file you can send back and commit. It works on Linux and Windows, uses
only the Python standard library, and needs one command.

This page covers what it measures, what to install, how to keep the machine
quiet, the command, what it writes, and what to send back. It also records the
rule that decides when a variant may become a default: only a measurement like
this one, on real hardware, can do that.

The kit does not change any setting of your machine, and it never writes to a
file that already exists.

## What it measures

| Variant | Environment | Meaning |
|---|---|---|
| `default` | none | the library as it ships: no mask, automatic lane selection. The baseline for every verdict. |
| `no_bmi2_adx` | `UFSECP_CPU_DISABLE=bmi2,adx` | BMI2 and ADX masked off, so the dispatchers take their non-BMI2/ADX path |
| `lanes_off` | `UFSECP_LANES=off` | lane backends disabled |
| `lanes_<backend>` | `UFSECP_LANES=<backend>` | one variant per lane backend that `cpu_hwinfo` lists as available (`avx2`, `avx_ifma`, `ifma256`, `ifma512`) |

The kit builds `cpu_hwinfo` and `bench_unified`, runs `cpu_hwinfo --json`, and
plans the variants from that:

- a tree without lane backends gets `default` and `no_bmi2_adx` only; the lane
  variants appear as soon as `cpu_hwinfo` lists backends;
- a CPU that reports neither BMI2 nor ADX gets no `no_bmi2_adx` variant (the mask
  would change nothing);
- the JSON lists every variant that was left out, and why.

Your own `UFSECP_CPU_DISABLE` and `UFSECP_LANES` are ignored: each variant sets
its own, and `default` runs with neither.

## Prerequisites

Linux:

- CMake 3.25 or newer (`CMakePresets.json` uses preset schema version 6), Ninja,
  and a C++ compiler that builds this repository;
- Python 3.8 or newer;
- `taskset` and `nice` (util-linux and coreutils, present on mainstream
  distributions);
- git, optional: it only records the commit in the JSON.

Windows (the default preset is `windows-clang-cl`):

- Visual Studio 2022 or its Build Tools with "Desktop development with C++" and
  the component "C++ Clang tools for Windows" (it provides `clang-cl`), plus a
  Windows SDK;
- CMake 3.25 or newer and Ninja (both ship with Visual Studio's CMake component,
  or install them separately);
- Python 3.8 or newer with the `py` launcher, and Git for Windows;
- run everything from an "x64 Native Tools Command Prompt for VS 2022" or a
  "Developer PowerShell for VS 2022", so that `clang-cl`, `link.exe` and the
  Windows SDK are on the path. The kit warns when the Visual Studio environment
  looks missing.

## Quiet-machine rules

A measurement is only as good as the machine it ran on. Before you start:

- **Power:** plug in the AC adapter and leave battery saver off. The kit warns
  and records it when it can see that the machine is on battery.
- **Power plan:** use the high-performance plan. Windows: Settings, System,
  Power, Power mode "Best performance" (or `powercfg /setactive SCHEME_MIN`).
  Linux: `sudo cpupower frequency-set -g performance`. The kit records the plan
  or governor it finds.
- **Nothing else running:** close browsers, IDEs, chat and sync clients; pause
  antivirus scans, Windows Update, indexing and backups; do not use the machine
  while the kit runs. Keep a laptop on a hard surface with the lid open and the
  vents free, so that thermal throttling does not move during the run.
- **Turbo:** leave it as it is. The kit does not change boost settings, but it
  records what it can read (the turbo state, and the turbo line that
  `bench_unified` itself writes). Do not toggle it in the middle of a session; if
  you changed it on purpose, say so when you send the file back.
- **Bare metal:** measure on the machine itself, not in a virtual machine or a
  container. If you cannot, say so when you send the file back.

The kit rotates the variant order every round, so a slow drift (the CPU warming
up or cooling down, background load) is spread over the variants instead of
favouring one. Round `r` runs the variants rotated left by `r` places. After as
many rounds as there are variants, every variant has run once at every position.
With the default 5 rounds that is exactly balanced for 5 variants; with more
variants, pick `--rounds` as a multiple of the variant count.

## Run it

Linux, from anywhere inside the repository:

```bash
python3 tools/measure_cpu_variants.py
```

Windows, from a Visual Studio developer prompt:

```bat
py -3 tools\measure_cpu_variants.py
```

The kit then:

1. configures and builds with the preset (`cpu-release` on Linux,
   `windows-clang-cl` on Windows), targets `bench_unified` and `cpu_hwinfo`;
2. reads the CPU features and lane backends from `cpu_hwinfo --json`;
3. runs `bench_unified --json` for every variant, 5 rounds, in rotating order,
   pinned to one core at raised priority;
4. writes the JSON and prints a table of verdict counts.

One run executes `bench_unified` once per variant per round, so the time grows
with both numbers; the kit prints progress and an estimate as it goes. Stop it
with Ctrl+C and it keeps the runs finished so far and writes them as an
incomplete result.

Pinning and priority:

- Linux: `taskset -c <core>` and `nice -n -20`. A negative nice value needs
  root, and the kit does not run the build as root. For the raised priority,
  build once as yourself, then measure with
  `sudo python3 tools/measure_cpu_variants.py --skip-build`. Without root the
  kit still pins the core, says so, and records it.
- Windows: the benchmark runs in the high priority class with a one-core affinity
  mask, the effect of `start /affinity 1 /high`. The kit sets it through the
  Windows API instead of `cmd`, so that exit codes, captured output and Ctrl+C
  keep working.
- `bench_unified` also pins its own measuring thread to CPU 0 and raises its own
  priority, so the default core is 0 and agrees with it.

Common options (`--help` lists all of them):

| Option | Meaning |
|---|---|
| `--preset NAME` | another CMake configure preset |
| `--rounds N` | rounds per variant (default 5; fewer than 5 gives no verdicts) |
| `--suite core\|extended\|all`, `--passes N` | passed to `bench_unified` |
| `--core N` | core to pin to (default 0) |
| `--skip-build` | measure the binaries already built by the preset; `--build-dir DIR` points at another build and needs `--skip-build` |
| `--variants a,b` | run only these variants (`default` is always kept) |
| `--lanes auto\|none\|a,b` | where the lane backends come from: `cpu_hwinfo` (default), nowhere, or a list you name |
| `--out-dir DIR` | write the JSON elsewhere than `docs/` |
| `--print-lanes FILE` | print the lane backends the kit finds in a saved `cpu_hwinfo --json` file |
| `--self-test` | check the ordering, verdict and schema logic on synthetic data, with no build and no CPU work |

## What it writes

One file, `docs/bench_variants_<YYYY-MM-DD>_<cpu-slug>.json`. The date is UTC and
`<cpu-slug>` is the CPU brand string reduced to lower-case words joined by hyphens,
without the trademark marks and the clock text (or `<vendor>-<family>-<model>-<stepping>`
when there is no brand string). An existing file is never overwritten: a second run
on the same day and CPU is written next to it as `..._2.json`, then `..._3.json`.

It records:

- **who and where:** `generated_by`, UTC time, the git commit and whether the tree
  was modified, the command line;
- **the machine:** CPU brand, vendor, family, model, stepping, microcode (from
  `/proc/cpuinfo` on Linux, or the `Update Revision` registry value on Windows;
  `unknown` when unreadable), the OS, and power, governor or plan and turbo state
  as far as they can be read;
- **the build:** preset, generator, build type, the compiler id and version from
  the build directory, the cache entries that shape the code (such as the march
  flag), and the sha256 of `bench_unified` and `cpu_hwinfo`;
- **the CPU as the library sees it:** the full `cpu_hwinfo --json`, and for each
  masked or lane variant the same view under that variant's environment, so you
  can check that the mask or backend took effect;
- **the measurement:** the order of each round, the pinning that was applied, the
  `bench_unified` metadata, and for every variant and op the per-round values,
  `median`, `min`, `max`, `spread_pct`;
- **the verdict** of each variant against `default`, per op.

### How a verdict is decided

Times are lower-is-better. With `base_*` the range of `default` over its rounds
and `new_*` the range of the variant, the repository rule is:

| Verdict | Condition |
|---|---|
| `improvement` | only if `new_max < base_min` |
| `regression` | only if `new_min > base_max` |
| `inconclusive` | otherwise: the ranges overlap or touch |

Two more values exist so that nothing is overclaimed. `insufficient_rounds`: one
side has fewer than 5 rounds, which the performance protocol does not accept for
a claim. `not_applicable`: ratio rows, which are derived values with no fixed
direction.

An `inconclusive` result is not a green light: it means the difference is too
small to measure on this machine, or the machine was too noisy. The per-op
`baseline_noisy` flag and the `baseline_noisy_ops` count in the summary mark the
ops where `default` itself spread by more than 3% between rounds. When that
count is not zero, quiet the machine and repeat before concluding anything.

### Partial runs

If a variant fails (for example a backend that crashes), the failure is recorded
under `failures` with the tail of its output, the sample is skipped, and a
variant that fails twice is dropped from the later rounds. The result is then
`status: "partial"` and `complete: false`, and the kit exits with a non-zero code.
If `default` itself fails, nothing can be compared, so the run stops. Rows with
zero, negative or non-finite timings are never used; they are listed under
`invalid_rows`.

The kit validates its own JSON against the schema before writing it, and refuses
to write a document that does not pass.

## What to send back

- the JSON file, unedited (a hand-edited file is not evidence);
- the console output as well, when the run ended with `INCOMPLETE`, a warning or
  an error;
- anything unusual about the session: turbo changed, on battery, a virtual
  machine, other programs running.

## When a variant becomes a default

A variant becomes a default only from a measurement like this one, on real
hardware of the CPU class in question. Not from reading the code, not from a
different CPU, not from an emulator or a virtual machine, and not from "the
algorithm is the same". A different microarchitecture can rank the variants
differently, so each CPU class needs its own file.

A change that makes a variant the default must cite the JSON file it rests on.
That file must come from a full run (`complete: true`, at least 5 rounds), on a
quiet machine (no noisy baseline ops for the ops that matter), with the verdict
`improvement` for the ops the change is meant to speed up and no `regression` on
the others. An `inconclusive` result changes nothing.

## The SDE workflow

`.github/workflows/sde-dispatch.yml` runs on every push to `dev` that touches
`src/cpu/**` or the CMake files, and on demand. It builds `cpu_hwinfo`,
`test_cpu_features_standalone` and, when the tree has it, `test_field_lanes`, then
runs them natively and under Intel SDE as Tiger Lake (`-tgl`), Sapphire Rapids
(`-spr`) and Sierra Forest (`-srf`, AVX-IFMA). Under `-spr` it requires the
`ifma512` and `ifma256` lane backends, and under `-srf` the `avx_ifma` backend
when it is built.

SDE is an emulator. That workflow proves the dispatch picks a path that runs and
agrees with the reference; it says nothing about speed. Speed comes only from this
kit, on real hardware.

## Troubleshooting

- `not found on PATH: clang-cl` (or `cmake`, `ninja`): open a Visual Studio
  developer prompt, or install the missing component from the prerequisites.
- `bench_unified not found` or `cpu_hwinfo not found`: the preset must enable
  `SECP256K1_BUILD_TESTS` (for `cpu_hwinfo`) and `SECP256K1_BUILD_BENCH` (for
  `bench_unified`); `cpu-release` and `windows-clang-cl` do.
- `could not set the one-core affinity mask` (Windows): the run was not pinned
  from outside. `bench_unified` still pins its own measuring thread, but record
  the warning and send the console output.
- Lane variants are missing although `cpu_hwinfo` prints lanes in a form the kit
  does not recognise: run `--print-lanes` on a saved `cpu_hwinfo --json` file to
  see what the kit understood, and name the backends yourself with
  `--lanes avx2,ifma256`.
- You only want to measure again: `--skip-build` reuses the binaries.

## Self-test

```bash
python3 tools/measure_cpu_variants.py --self-test
```

checks the rotation order, the verdict rule, the lane discovery, the variant
planning, the never-overwrite writer and the JSON schema on synthetic bench
output. It builds nothing and does no CPU work, and the SDE workflow runs it too.
