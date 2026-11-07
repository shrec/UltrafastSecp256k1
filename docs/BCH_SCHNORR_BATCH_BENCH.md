# BCH 2019 Schnorr verification benchmark: reproduction guide (issue #466)

This guide is for testers who want to measure UltrafastSecp256k1's Bitcoin Cash
2019 Schnorr verifier against the libsecp256k1 fork that Knuth vendors (Knuth
PRs #825 / #826, which added `secp256k1_schnorr_verify_batch`) on their own
machine, and post the result to
[issue #466](https://github.com/shrec/UltrafastSecp256k1/issues/466).

This page contains no results. Every number comes from the JSON the harness
writes on your machine.

## What the harness measures

`bench_bch_schnorr` ([`benchmarks/bch/bench_bch_schnorr.cpp`](../benchmarks/bch/bench_bch_schnorr.cpp))
runs these rows for batch sizes N = 1, 2, 8, 32, 128, 512, 2048 (`--sizes` to change):

| Row | What one timed call does |
|---|---|
| `ufsecp single` | N x (`bch_schnorr_pubkey_parse` + `bch_schnorr_verify`), one signature at a time |
| `ufsecp batch` | N x `bch_schnorr_pubkey_parse`, then one `bch_schnorr_batch_verify` |
| `ufsecp C ABI batch` | one `ufsecp_bch_schnorr_batch_verify` on the 129-byte rows (it parses the keys itself) |
| `ufsecp lanes <backend>` | N x `bch_schnorr_pubkey_parse`, then one `lanes::bch_schnorr_verify_lanes` with that lane backend (`avx2`, `avx_ifma`, `ifma256`, `ifma512`); one row per available backend, see [The `ufsecp lanes` rows](#the-ufsecp-lanes-rows) |
| `ref single` | N x (`secp256k1_ec_pubkey_parse` + `secp256k1_schnorr_verify`) |
| `ref batch` | N x `secp256k1_ec_pubkey_parse`, then one `secp256k1_schnorr_verify_batch` |

Each row reports ns per signature (call time / N), signatures per second, and
the median, min and max over the timed passes.

- **Same inputs everywhere.** N distinct keys and messages, signed by the
  harness, laid out once before timing. The batch rows (`ufsecp batch`,
  `ref batch`, `ufsecp lanes <backend>`) all start with N key parses, then
  make one call; the lanes call parses the keys a second time (see below).
- **Cross-library check first.** Before anything is timed, every
  implementation must accept every set, and must reject the same set with one
  signature corrupted. Any disagreement stops the run with exit code 1 and no
  numbers. Every warm-up and timed call is checked as well.
- **Method.** This is `bench_unified`'s method, using its harness
  (`src/cpu/include/secp256k1/benchmark_harness.hpp`). A CPU frequency warm-up
  runs first. Then each row gets warm-up calls and 11 passes (`--passes`). Each
  pass is the mean of a calibrated number of calls (`--pass-ms` sets the target
  pass length). Outliers are removed by IQR, then the median is taken.
  Everything runs on one thread.
- **Reference rows are optional.** Without the reference library the harness
  runs the `ufsecp` rows only (the lane rows included) and says so.
- The BCHN compatibility shim in this repository
  (`compat/libsecp256k1_bchn_shim`) is never linked into the harness. It
  exports the same `secp256k1_schnorr_*` symbols as the reference.

### The `ufsecp lanes` rows

`ufsecp lanes <backend>` verifies several signatures per SIMD instruction
stream, one per lane, with
`secp256k1::lanes::bch_schnorr_verify_lanes`
([`src/cpu/include/secp256k1/lanes.hpp`](../src/cpu/include/secp256k1/lanes.hpp))
and that backend passed explicitly. Its verdict is, entry by entry, the one
`bch_schnorr_verify` returns; an entry the lanes cannot decide goes to the
scalar verifier. Inside the lanes, R = s*G - e*P is a GLV-split dual
multiplication, and both checks stay projective: x(R) == r as X == r*Z^2, and
y(R) a quadratic residue as Euler's criterion on Y*Z. A timed call is
N x `bch_schnorr_pubkey_parse` (the caller-side step of the other batch rows)
and one lanes call on the wire-form entries. The lanes call decompresses every
key again itself, so this row does the key decompression twice.

The harness adds one row per lane backend that is **available**: compiled
into this build and supported by your CPU, as `UFSECP_CPU_DISABLE` leaves it.
`cpu_hwinfo` lists them (`dispatch.lanes.available`):

| Backend | Lanes | Instructions | CPUs |
|---|---|---|---|
| `ifma512` | 8 | AVX-512F + AVX-512VL + AVX-512 IFMA, zmm | AVX-512 IFMA CPUs: AMD Zen 4 and Zen 5; Intel Ice Lake, Tiger Lake, Sapphire Rapids and later Xeons |
| `ifma256` | 4 | AVX-512VL + AVX-512 IFMA, ymm | the same CPUs as `ifma512` |
| `avx_ifma` | 4 | AVX-IFMA (VEX encoded), ymm | AVX-IFMA without AVX-512, for example Intel Sierra Forest; built only when the compiler accepts `-mavxifma` |
| `avx2` | 4 | AVX2 | any x86-64 CPU with AVX2 |

So on an AVX-512 IFMA machine (Zen 4/5, Ice Lake and later, Sapphire Rapids)
every N gets `ufsecp lanes ifma512`, `ufsecp lanes ifma256` and
`ufsecp lanes avx2` rows next to the other rows. The stdout header prints a
`lane rows:` line, and the JSON lists them in `lanes.bench_rows` and puts
`lane_backend` on every row.

Before anything is timed, each lane row passes the cross-check like every
other row. In addition, the lane kernel itself must decide every entry of
every valid set (`LaneVerifyStats.lane_verified == N`); a row that would time
the scalar fallback stops the run with exit code 1.

To force or limit the lane rows:

- `UFSECP_LANES=<backend>` (`avx2`, `avx_ifma`, `ifma256`, `ifma512`) keeps
  only that backend's row when it is available; `UFSECP_LANES=off` drops the
  lane rows. A value that names nothing available is reported on a `WARNING`
  line and ignored (every available backend keeps its row). It can never
  select an instruction set the CPU lacks.
- `UFSECP_CPU_DISABLE=avx512` (or `ifma`, `avx_ifma`, `avx2`) masks CPU
  features, which removes the backends that need them.

## The pinned reference

| | |
|---|---|
| Repository | <https://github.com/k-nuth/kth> |
| Commit | `d1b9a8ac50cce15843c6925374709e2391e7aba4` |
| Directory | `src/secp256k1` (Knuth's vendored libsecp256k1 v0.8.0) |
| Build | static, `-DSECP256K1_ENABLE_MODULE_SCHNORR_BCH=ON`, tests and benchmarks off |

The harness finds the reference through two CMake cache variables:
`UFSECP_BCH_REF_SECP256K1_INCLUDE_DIR` (the reference's `include/` directory)
and `UFSECP_BCH_REF_SECP256K1_LIBRARY` (its static library). Set both, or
neither. The JSON records the reference commit and whether it matches the pin.

Every platform below uses the same layout: one work directory holding
`UltrafastSecp256k1/`, `kth/` (the reference sources) and `kth-build/`.

---

## Linux x86-64 (GCC or Clang)

### 1. Prerequisites and clone

```bash
# Debian/Ubuntu; use your distribution's equivalents elsewhere
sudo apt install git cmake ninja-build build-essential      # add clang for a Clang build

# Pick ONE compiler and use it for both libraries:
export CC=gcc   CXX=g++          # or:  export CC=clang CXX=clang++

mkdir -p ~/bch-bench && cd ~/bch-bench
git clone --branch dev https://github.com/shrec/UltrafastSecp256k1.git
```

### 2. Fetch and build the reference

```bash
cd ~/bch-bench
KTH_COMMIT=d1b9a8ac50cce15843c6925374709e2391e7aba4
git init -q kth
git -C kth remote add origin https://github.com/k-nuth/kth.git
git -C kth config core.sparseCheckout true
printf 'src/secp256k1/\n' > kth/.git/info/sparse-checkout
git -C kth fetch --depth 1 --filter=blob:none origin "$KTH_COMMIT"
git -C kth checkout --detach FETCH_HEAD
git -C kth rev-parse HEAD        # must print d1b9a8ac50cce15843c6925374709e2391e7aba4

cmake -S kth/src/secp256k1 -B kth-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
  -DSECP256K1_ENABLE_MODULE_SCHNORR_BCH=ON \
  -DSECP256K1_BUILD_TESTS=OFF \
  -DSECP256K1_BUILD_EXHAUSTIVE_TESTS=OFF \
  -DSECP256K1_BUILD_BENCHMARK=OFF \
  -DSECP256K1_BUILD_CTIME_TESTS=OFF \
  -DSECP256K1_BUILD_EXAMPLES=OFF
cmake --build kth-build
REF_LIB="$(find "$PWD/kth-build" -type f -name libsecp256k1.a | head -n 1)"
echo "$REF_LIB"                  # must not be empty
```

### 3. Configure UltrafastSecp256k1 with the two cache variables, build the harness

```bash
cd ~/bch-bench
cmake -S UltrafastSecp256k1 -B UltrafastSecp256k1/out/bch-bench -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSECP256K1_BUILD_TESTS=OFF \
  -DSECP256K1_BUILD_EXAMPLES=OFF \
  -DSECP256K1_BUILD_JAVA=OFF \
  -DUFSECP_BCH_REF_SECP256K1_INCLUDE_DIR="$PWD/kth/src/secp256k1/include" \
  -DUFSECP_BCH_REF_SECP256K1_LIBRARY="$REF_LIB"
cmake --build UltrafastSecp256k1/out/bch-bench --target bench_bch_schnorr
```

The configure output must contain `bench_bch_schnorr: reference rows ON`.
The binary is `UltrafastSecp256k1/out/bch-bench/benchmarks/bch/bench_bch_schnorr`.

### 4. Fix the clock and pick one core

Read and note the current values first (`cat` the files below) so you can
restore them afterwards.

```bash
# performance governor on every CPU
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# turbo / boost off: use whichever file exists on your machine
echo 1 | sudo tee /sys/devices/system/cpu/intel_pstate/no_turbo     # Intel, intel_pstate driver
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost             # acpi-cpufreq, amd-pstate (Zen)
# amd-pstate kernels without the global file expose it per policy:
for f in /sys/devices/system/cpu/cpufreq/policy*/boost; do echo 0 | sudo tee "$f"; done
```

Pick one core for the run.

- Avoid CPU 0, which services most interrupts.
- Leave the core's SMT sibling idle too. `lscpu -e` shows which CPUs share a core.
- On hybrid Intel parts, pick a P-core.

The examples below use CPU 2. Close everything else that uses CPU.

### 5. Run it five times

```bash
cd ~/bch-bench
for i in 1 2 3 4 5; do
  taskset -c 2 ./UltrafastSecp256k1/out/bch-bench/benchmarks/bch/bench_bch_schnorr \
    --note "run $i: <CPU model>, turbo off, governor performance, taskset -c 2" \
    --json "bch466_linux_run$i.json"
done
```

Use `taskset`, not `--pin`. `--pin` would move the thread to CPU 0.

The JSON records the affinity it ran with, the turbo state it read from sysfs,
and the governor of that CPU. Afterwards, write back the values you noted in
step 4.

---

## macOS arm64 (Apple Silicon, AppleClang)

### 1. Prerequisites and clone

```bash
xcode-select --install            # Command Line Tools: AppleClang, git
brew install cmake ninja          # Homebrew

mkdir -p ~/bch-bench && cd ~/bch-bench
git clone --branch dev https://github.com/shrec/UltrafastSecp256k1.git
```

### 2. Fetch and build the reference

Run exactly the commands of Linux step 2: the same `git` and `cmake` lines,
and the same `find` for `libsecp256k1.a`.

### 3. Configure and build

Run exactly the commands of Linux step 3. AppleClang is the default compiler;
leave `CC`/`CXX` unset.

### 4. Run it five times

macOS has no way to pin a thread to a core, and no switch for frequency
boosting. The OS decides where the single benchmark thread runs. Reduce the
noise you can:

- plug in the power adapter;
- turn Low Power Mode off (System Settings > Battery, or `sudo pmset -a lowpowermode 0`);
- quit other applications and let the machine sit idle for a minute.

```bash
cd ~/bch-bench
for i in 1 2 3 4 5; do
  ./UltrafastSecp256k1/out/bch-bench/benchmarks/bch/bench_bch_schnorr \
    --note "run $i: <Mac model / chip>, AC power, Low Power Mode off" \
    --json "bch466_macos_run$i.json"
done
```

The JSON records `cpu_affinity: "not available on this OS"`. That is expected.

---

## Windows x64 (MSVC)

### 1. Prerequisites and clone

Install Visual Studio 2022 with the "Desktop development with C++" workload,
which includes MSVC and CMake. Also install Git for Windows. Run every command
below in "Developer PowerShell for VS 2022".

```powershell
mkdir $HOME\bch-bench; cd $HOME\bch-bench
git clone --branch dev https://github.com/shrec/UltrafastSecp256k1.git
```

### 2. Fetch and build the reference

```powershell
cd $HOME\bch-bench
$KTH_COMMIT = 'd1b9a8ac50cce15843c6925374709e2391e7aba4'
git init -q kth
git -C kth remote add origin https://github.com/k-nuth/kth.git
git -C kth config core.sparseCheckout true
[System.IO.File]::WriteAllText("$PWD\kth\.git\info\sparse-checkout", "src/secp256k1/`n")
git -C kth fetch --depth 1 --filter=blob:none origin $KTH_COMMIT
git -C kth checkout --detach FETCH_HEAD
git -C kth rev-parse HEAD        # must print d1b9a8ac50cce15843c6925374709e2391e7aba4

cmake -S kth\src\secp256k1 -B kth-build `
  -DBUILD_SHARED_LIBS=OFF `
  -DSECP256K1_ENABLE_MODULE_SCHNORR_BCH=ON `
  -DSECP256K1_BUILD_TESTS=OFF `
  -DSECP256K1_BUILD_EXHAUSTIVE_TESTS=OFF `
  -DSECP256K1_BUILD_BENCHMARK=OFF `
  -DSECP256K1_BUILD_CTIME_TESTS=OFF `
  -DSECP256K1_BUILD_EXAMPLES=OFF
cmake --build kth-build --config Release
$REF_LIB = (Get-ChildItem kth-build -Recurse -Include libsecp256k1.lib, secp256k1.lib |
            Select-Object -First 1).FullName
$REF_LIB                         # must not be empty
```

### 3. Configure UltrafastSecp256k1 with the two cache variables, build the harness

```powershell
cd $HOME\bch-bench
cmake -S UltrafastSecp256k1 -B UltrafastSecp256k1\out\bch-bench `
  -DSECP256K1_BUILD_TESTS=OFF -DSECP256K1_BUILD_EXAMPLES=OFF -DSECP256K1_BUILD_JAVA=OFF `
  "-DUFSECP_BCH_REF_SECP256K1_INCLUDE_DIR=$PWD\kth\src\secp256k1\include" `
  "-DUFSECP_BCH_REF_SECP256K1_LIBRARY=$REF_LIB"
cmake --build UltrafastSecp256k1\out\bch-bench --config Release --target bench_bch_schnorr
```

The configure output must contain `bench_bch_schnorr: reference rows ON`.
The binary is `UltrafastSecp256k1\out\bch-bench\benchmarks\bch\Release\bench_bch_schnorr.exe`.

### 4. Boost off, then run it five times

Switch processor boost off for the active power plan. Note the current AC
value first, so you can restore it afterwards.

```powershell
powercfg /query scheme_current sub_processor perfboostmode      # note "Current AC Power Setting Index"
powercfg /setacvalueindex scheme_current sub_processor perfboostmode 0
powercfg /setactive scheme_current
```

`--pin` pins the benchmark thread to logical CPU 0 and raises the process to
HIGH priority (the harness's `bench::pin_thread_and_elevate`):

```powershell
cd $HOME\bch-bench
1..5 | ForEach-Object {
  .\UltrafastSecp256k1\out\bch-bench\benchmarks\bch\Release\bench_bch_schnorr.exe --pin `
    --note "run $($_): <CPU model>, boost off (perfboostmode 0), --pin" `
    --json "bch466_windows_run$($_).json"
}
```

To use a core other than CPU 0, drop `--pin` and start the harness from
`cmd.exe` with an affinity mask. Mask `4` is CPU 2:
`start "" /wait /b /high /affinity 4 UltrafastSecp256k1\out\bch-bench\benchmarks\bch\Release\bench_bch_schnorr.exe --json bch466_windows_run1.json`

Afterwards, restore boost: rerun the `setacvalueindex` line with the value you
noted, then `powercfg /setactive scheme_current`.

---

## What to post to issue #466

1. All five JSON files from your platform, attached to the comment or in a
   gist. Post the JSON, not only the stdout table: the JSON carries the CPU,
   OS, compiler, flags, affinity, turbo state and lane selection.
2. One line of context the JSON cannot know: cooling, power mode, how turbo
   was switched off (or that it could not be), and which core you used.
3. Anything you changed from this guide: another compiler, other flags, or
   another reference commit. The JSON records this project's build flags, but
   not the reference's; those are whatever its configure line said.
4. If a run printed a `WARNING` line, say so.
5. On a CPU with AVX-512 IFMA (AMD Zen 4 / Zen 5, Intel Ice Lake and later,
   Sapphire Rapids), the table should have `ufsecp lanes ifma512` and
   `ufsecp lanes ifma256` rows. If they are missing, say so and attach
   `cpu_hwinfo --json`.
6. If the harness exits 1 with `CROSS-CHECK FAILED` (or with `the lane kernel
   decided ...`), post its full output instead of timings. That is a
   correctness report and matters more than any speed comparison.

---

## Optional variants

Post every variant as its own set of runs, and say in `--note` which one it is.

### One lane backend per run (x86-64)

The `ufsecp lanes` rows (see [above](#the-ufsecp-lanes-rows)) run by default,
one per available backend. To post one run per backend instead, for example
`ifma512` and `ifma256` separately on a Zen 4/5 or Ice Lake+ machine, list the
backends your CPU and build provide:

```bash
cmake --build UltrafastSecp256k1/out/bch-bench --target cpu_hwinfo
./UltrafastSecp256k1/out/bch-bench/src/cpu/cpu_hwinfo     # see dispatch.lanes.available
```

Then run once per backend with `UFSECP_LANES` set to it (`avx2`, `avx_ifma`,
`ifma256`, `ifma512`), which limits the lane rows to that backend:

```bash
UFSECP_LANES=ifma512 taskset -c 2 ./UltrafastSecp256k1/out/bch-bench/benchmarks/bch/bench_bch_schnorr \
  --note "UFSECP_LANES=ifma512, <CPU model>, turbo off, taskset -c 2" --json bch466_linux_lanes_ifma512.json
```

`UFSECP_LANES` changes only which lane rows run: `bch_schnorr_verify` and
`bch_schnorr_batch_verify`, and so the other `ufsecp` rows, never use a lane
backend. The JSON's `lanes.verify_backend` is the backend the ECDSA and BIP-340
batch verifiers would get; it reads `off` unless `UFSECP_LANES` forces one.
`lanes.selected` is only the field-kernel default pick, which `cpu_hwinfo` also
prints.

On Windows the tool is
`UltrafastSecp256k1\out\bch-bench\src\cpu\Release\cpu_hwinfo.exe`. Set the
variable with `$env:UFSECP_LANES = 'ifma512'` before the run and clear it with
`Remove-Item Env:UFSECP_LANES` afterwards.

### clang-cl on Windows

MSVC's `cl` cannot compile this library's inline-assembly field kernels; the
top-level `CMakeLists.txt` recommends clang-cl for speed on Windows. If Visual
Studio has the "C++ Clang tools for Windows" component, repeat steps 2 and 3:

- add `-T ClangCL` to both configure commands (the reference's and this project's);
- use separate build directories, for example `kth-build-clangcl` and `out\bch-bench-clangcl`;
- label the runs.

### Same ISA flags for both libraries

On x86-64 with GCC or Clang this project compiles with `-march=native` unless
`SECP256K1_MARCH` is set (top-level `CMakeLists.txt`). The reference keeps its
own default flags. For a same-ISA comparison, do one of:

- build this project for a portable level, with `-DSECP256K1_MARCH=x86-64-v3`
  in step 3;
- add `-DCMAKE_C_FLAGS=-march=native` to the reference's configure in step 2,
  in a fresh `kth-build`.

---

## GPU rows (CUDA, OpenCL)

The GPU verifier is `ufsecp_gpu_bch_schnorr_verify_batch`. Two harnesses time
it, and both check every result against the CPU verifier:

- `bench_gpu_abi_batch`
  ([`benchmarks/gpu/bench_gpu_abi_batch.cpp`](../benchmarks/gpu/bench_gpu_abi_batch.cpp))
  times whole C ABI calls: host-to-device copy, kernel and copy back. Its
  `ns/call` at small N is the end-to-end latency, and its `ns/item` at large N
  is the throughput.
- `gpu_bench_unified` (CUDA only) has a kernel-only `BCH Schnorr verify` row at
  batch 65536. The difference from the C ABI row at the same N is the transfer
  and launch overhead.

Linux, CUDA. Use your GPU's compute capability; for example, 89 is Ada and 120
is Blackwell:

```bash
cd ~/bch-bench
cmake -S UltrafastSecp256k1 -B UltrafastSecp256k1/out/bch-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DSECP256K1_BUILD_CUDA=ON -DSECP256K1_BUILD_BENCH=ON \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DSECP256K1_BUILD_TESTS=OFF
cmake --build UltrafastSecp256k1/out/bch-cuda --target bench_gpu_abi_batch gpu_bench_unified
./UltrafastSecp256k1/out/bch-cuda/benchmarks/gpu/bench_gpu_abi_batch --backend cuda \
  --ops bch_schnorr_verify --sizes 2,8,32,128,512,2048,65536 --json bch466_gpu_cuda_abi.json
./UltrafastSecp256k1/out/bch-cuda/src/cuda/gpu_bench_unified --json bch466_gpu_cuda_kernel.json
```

Linux, OpenCL: use `-DSECP256K1_BUILD_OPENCL=ON` instead of the CUDA options,
a separate build directory, and `--backend opencl`. On NVIDIA drivers, the first
OpenCL run compiles the kernels and can take several minutes; time a second
run.

Post both JSON files, the GPU model and the driver version. GPU timings do not
need the CPU pinning above, but close other GPU work first.

---

## The JSON, field by field

| Field | Content |
|---|---|
| `generated_by`, `schema` | `bench_bch_schnorr --json`, `ufsecp.bch_schnorr_bench.v1` |
| `date_utc`, `argv`, `note` | when, the exact command line, your `--note` |
| `ufsecp` | version, C ABI version string, `git describe` and commit (taken at CMake configure time) |
| `machine` | CPU model, OS, architecture, CPU affinity, `--pin`, turbo state and its source, governor, threads in the process after the run |
| `build` | compiler and version (from the compiler itself and from CMake), configuration, flags, `NDEBUG`, optimization |
| `lanes` | `verify_backend` (what the ECDSA / BIP-340 batch verifiers get; `off` unless forced), the field-kernel compiled / available / selected / requested backends and why, `bench_rows` (the backends that got a `ufsecp lanes` row, or `none`) and `bench_rows_rule` (every available backend, or only the one `UFSECP_LANES` names), plus `UFSECP_LANES` and `UFSECP_CPU_DISABLE` as set |
| `reference` | present or not, include dir, header, library, its git commit and whether that matches the pin |
| `method` | timer, passes, pass target, warm-up, statistic |
| `cross_check` | sizes and implementations that passed the pre-timing check; `lanes_rule`, the extra lane-kernel check of the `ufsecp lanes` rows |
| `rows[]` | per implementation and N: `lane_backend` (the backend of a `ufsecp lanes` row, `null` on every other row), calls per pass, warm-up calls, passes kept/trimmed, `ns_per_call` and `ns_per_sig` (median, min, max), `sigs_per_sec_median` |

`min` and `max` are the fastest and slowest kept pass after IQR trimming.

## Continuous integration

`.github/workflows/benchmark.yml` runs the same harness on GitHub's hosted
Linux and Windows runners. It fetches and builds the pinned reference, then
runs the short mode `--passes 3 --sizes 1,8,128,2048`, and uploads the
`bench-bch-schnorr-*` JSON as an artifact.

Those runners are uncontrolled: no pinning and no turbo lock. Their numbers are
**non-canonical**, like the job's `bench_unified` numbers. They show that
everything builds and that the implementations agree; they are not
measurements to cite.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `bench_bch_schnorr: reference rows OFF` | the two cache variables were not passed; rerun step 3 with both |
| `no secp256k1_*.h ... declares secp256k1_schnorr_verify_batch` | wrong checkout or include dir; check `git -C kth rev-parse HEAD` |
| undefined `secp256k1_schnorr_verify_batch` at link time | the reference was built without `-DSECP256K1_ENABLE_MODULE_SCHNORR_BCH=ON`; rebuild `kth-build` from an empty directory |
| Windows `LNK2019` on `__imp_secp256k1_*` | the reference was built as a DLL; rebuild it with `-DBUILD_SHARED_LIBS=OFF` |
| `WARNING: built without optimization` or `NDEBUG is not defined` | not a Release build; reconfigure with `-DCMAKE_BUILD_TYPE=Release` (or `--config Release`) |
| exit code 1, `CROSS-CHECK FAILED` | the implementations disagree on a signature set; post the full output (see above) |
| exit code 1, `the lane kernel decided ... valid entries` | a `ufsecp lanes` row would have timed the scalar fallback instead of the lane kernel; post the full output |
| no `ufsecp lanes` rows | no lane backend is available (not x86-64, a CPU without AVX2, `UFSECP_CPU_DISABLE`), or `UFSECP_LANES=off`; the `lane rows:` header line and `cpu_hwinfo` say which |
| `ufsecp lanes ifma512` missing on a Zen 4/5 or Ice Lake+ CPU | the OS does not enable AVX-512 state, the feature is masked by `UFSECP_CPU_DISABLE`, or `UFSECP_LANES` names another backend; check `cpu_hwinfo` (`dispatch.lanes.available`) |
