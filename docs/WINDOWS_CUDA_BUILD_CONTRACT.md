# Windows / MSVC CUDA Build Contract (PR #353)

Source: PR #353 (Eric Voskuil / evoskuil), upstream branch `fix/windows-msvc-cuda`
(head `bd0ae6b4c6d4330d3db60423e2cacada1fcafc29`, forked from an old common
ancestor of `main`/`dev`, base ref `main`). Reproduced by hand into this
repo's `dev` — **not** merged or cherry-picked, per this project's branch
policy (the PR's own history is main-based and predates most of `dev`).

Paired with the CI gate `ci/check_windows_cuda_contract.py` (wired into
`ci/preflight.py`) — if this document and that gate ever disagree, the gate
is normative; update both together.

## 1. `field_mul_small` reserved parameter name (`src/cuda/include/secp256k1.cuh`)

**Problem.** Windows' `<rpcndr.h>` (pulled in transitively by `<windows.h>`
via `<rpc.h>`, whenever a CUDA/MSVC translation unit touches the Win32/COM
API before this header) defines `#define small char` (and, for the same
historical MIDL-stub-type reason, `#define hyper __int64`). Before this fix,
`field_mul_small()`'s second parameter was literally named `small`. The
preprocessor macro-substitutes any bare `small` token — including a
parameter *name*, which is not part of the type system in C++ — turning the
declaration into the syntactically invalid `uint32_t char` and the internal
use into `static_cast<uint64_t>(char)`. Both are hard nvcc errors.

**Fix.** Renamed the parameter (and its one internal use) from `small` to
`factor`. No signature/ABI/arithmetic change — parameter names are invisible
to callers. `field_mul_small` has zero callers anywhere in this codebase
today (confirmed via source_graph `api`/`context` over the full CUDA
include/source tree), so the blast radius of this rename is zero beyond the
declaration itself.

**Fail-before / pass-after** (measured locally, this session, nvcc release
12.0, `nvcc -c --std=c++17 -I src/cuda/include -I include <fixture>.cu -arch=sm_75`):

```
$ nvcc -c --std=c++17 -I src/cuda/include -I include \
    ci/fixtures/pr353_windows_small_macro_smoke.cu -o /tmp/smoke.o -arch=sm_75
src/cuda/include/secp256k1.cuh(1234): error: invalid combination of type specifiers
src/cuda/include/secp256k1.cuh(1241): error: type name is not allowed
2 errors detected in the compilation of "ci/fixtures/pr353_windows_small_macro_smoke.cu".
$ echo $?
2
```
against the pre-fix header (parameter named `small`); exit `0` (one
pre-existing, unrelated warning at line 894 about an unused variable `d` in
`mul_shift_384`, untouched by this fix) against the fixed header. Verified
identical error class on nvcc 12.0 and 13.2 on this machine — exact wording
is nvcc-version-dependent, so the gate matches on error class, not string.

**Regression fixture:** `ci/fixtures/pr353_windows_small_macro_smoke.cu` —
portably reproduces the `<rpcndr.h>` collision on any host (no Windows
machine, no GPU — `nvcc -c` is compile-only; device codegen via ptxas/cicc
does not require a physical GPU).

**CI enforcement:**
- Structural: `ci/check_windows_cuda_contract.py` parses the *actual
  parameter list* of `field_mul_small`'s definition (comments stripped first
  — a comment cannot cause a false PASS or a false REJECT).
- Compile-time: the `cuda-compile-check` job in `.github/workflows/ci.yml`
  compiles this fixture with `nvcc` on every push/PR to `main`/`dev`
  (ordinary `ubuntu-24.04` runner, no GPU hardware).

## 2. Platform-specific GPU-hook link retention

**Problem.** The self-installing `GpuColumnsVerifyHook` provider
(`gpu_engine_hook.cpp`'s `EngineGpuColumnsInstaller`) has no symbol any
libbitcoin-direct executable references directly. A normal static link
discards the whole TU. Before this fix,
`compat/libbitcoin_direct/CMakeLists.txt` forced retention unconditionally
with `LINKER:--undefined=secp256k1_gpu_columns_provider_anchor` — a
**GNU-linker-only** flag. MSVC's `link.exe` does not understand it (and
silently ignores it rather than failing the build), so on an MSVC build the
anchor was never retained.

**Fix.**
```cmake
if(MSVC)
    set(_lbtc_gpu_retain "LINKER:/INCLUDE:secp256k1_gpu_columns_provider_anchor")
else()
    set(_lbtc_gpu_retain "LINKER:--undefined=secp256k1_gpu_columns_provider_anchor")
endif()
```

| toolchain | linker | retention option |
|-----------|--------|-------------------|
| GCC, Clang (Linux, macOS, MinGW-w64) | GNU `ld` / LLVM `lld` / Apple `ld` | `LINKER:--undefined=secp256k1_gpu_columns_provider_anchor` |
| MSVC (Visual Studio 17 2022+) | `link.exe` | `LINKER:/INCLUDE:secp256k1_gpu_columns_provider_anchor` |

`${_lbtc_gpu_retain}` is applied identically (via `target_link_options`) to
all 8 targets that link `secp256k1_gpu_host` in this file:
`test_lbtc_direct_verify`, `test_lbtc_direct_operations`,
`test_lbtc_direct_gpu_columns_hook`, `bench_lbtc_direct_batch`,
`bench_lbtc_hash256_var`, `bench_lbtc_public_ops`, `bench_lbtc_workloads`,
`example_lbtc_public_ops`.

**Fallback behavior.** If this regresses (wrong option, swapped branches,
anchor-name typo), the build still succeeds and every CPU-path test still
passes — the regression is a **silent CPU fallback**, not a crash or a
wrong-answer bug: `SECP256K1_BUILD_LIBBITCOIN_GPU=ON` on MSVC would silently
downgrade to CPU-only column verify, with no build error and no test
failure. Classified as a build-contract/performance regression, not a
security exploit (no attacker-controlled data or cryptographic property is
involved).

**Regression fixture:** `ci/fixtures/pr353_msvc_link_retention/` — standalone
CMake project, CUDA-free, GPU-free: a static archive (`anchor.cpp`, mirrors
the production anchor shape — an exported symbol plus a static initializer
with no other externally-referenced symbol) linked into two executables:
- `pr353_link_retention_baseline` — no retention option (negative control;
  proves the positive result below is not vacuous).
- `pr353_link_retention_retained` — the SAME `if(MSVC)/else()` pattern as
  production, with a fixture-local anchor name
  (`pr353_link_retention_anchor`, deliberately distinct from the production
  anchor — this fixture tests the generic CMake/linker mechanism; cross-file
  anchor-name *consistency* for the real production anchor is checked
  separately and structurally by `ci/check_windows_cuda_contract.py`).

Two independent proofs per executable: (1) runtime behavior — does the
static initializer's `PR353_ANCHOR_LINKED` marker print; (2) symbol-table
evidence via `nm`/`llvm-nm`/`dumpbin`.

**CI enforcement:**
- Structural: `ci/check_windows_cuda_contract.py` parses
  `CMakeLists.txt`'s `if`/`elseif`/`else`/`endif` nesting (not proximity) to
  confirm `/INCLUDE:` is scoped inside `if(MSVC)` and `--undefined=` inside
  that construct's `else()`, and cross-checks both anchor names against
  `gpu_engine_hook.cpp`'s `extern "C" int ... = 1;` definition.
- Local, GNU-like (validated this session): `ctest` inside the fixture's own
  build directory runs both the runtime-behavior and (via `find_program`)
  the `nm`/`llvm-nm` symbol-table CTest cases — all 4 pass on Linux/GCC.
- Machine-verified on real MSVC: the `windows` job (`windows-2022`, VS 17
  2022, runs on every push/PR) builds the fixture via `ctest -R
  pr353_msvc_link_retention`, then runs an **explicit** `dumpbin /symbols`
  assertion (locating `dumpbin.exe` via `vswhere`, independent of whatever
  the plain `cmake --build` PATH exposes) confirming the anchor symbol is
  present in `pr353_link_retention_retained.exe` and absent from
  `pr353_link_retention_baseline.exe`.
- Integration-level (unchanged): `lbtc_direct_gpu_columns_hook` remains the
  real production hook proof when the CUDA/libbitcoin GPU profile is
  available.

## Test coverage summary

| Layer | Fixture / test | Proves |
|-------|----------------|--------|
| Structural (no build) | `ci/check_windows_cuda_contract.py` + `ci/test_check_windows_cuda_contract.py` | No reserved `small` param; retention present/scoped/not-swapped; anchor names consistent; both fixtures reachable from mandatory CI |
| Compile-only, no GPU | `pr353_windows_small_macro_smoke.cu` (`cuda-compile-check` job) | `field_mul_small` compiles under the `<rpcndr.h>` macro collision |
| Link-only, MSVC, no GPU/CUDA | `pr353_msvc_link_retention/` (`windows` job) | `LINKER:/INCLUDE:<anchor>` genex is correctly translated by `link.exe`; anchor retained (`dumpbin /symbols`) |
| Integration, real GPU hardware | `lbtc_direct_gpu_columns_hook` (local/self-hosted CUDA+libbitcoin-GPU only) | The real production hook self-installs |

## Windows toolchain expectations

- CI: `windows-2022` GitHub-hosted runner, Visual Studio 17 2022, CMake
  `"Visual Studio 17 2022"` generator, `-A x64` — same configuration the
  existing `windows` job already uses for the main build.
- The MSVC link-retention fixture needs no CUDA Toolkit — pure C++.
- `dumpbin.exe` is located via `vswhere.exe`, not assumed to be on `PATH`
  (the CMake VS generator does not run inside a Developer Command Prompt).

## Unresolved limitations

1. No GitHub-hosted or self-hosted Windows+CUDA runner exists in this repo.
   A real `SECP256K1_BUILD_CUDA=ON` Windows build (nvcc + MSBuild CUDA
   integration) is not exercised by any CI leg today. These two fixtures
   prove the two specific PR #353 fixes in isolation; they do not prove a
   full Windows+CUDA project build.
2. `gpu-selfhosted.yml`'s `on:` trigger is `workflow_dispatch` only, despite
   its own header comment claiming push/PR/nightly triggers — a
   pre-existing, unrelated discrepancy, intentionally not touched by this
   narrow-scope task. Because of this, `cuda-compile-check` (this task's new
   mandatory, no-GPU-hardware CUDA compile job) was added to `ci.yml`
   instead, which genuinely triggers on push/PR to `main`/`dev`.
3. The fixture's anchor symbol (`pr353_link_retention_anchor`) is
   intentionally distinct from the production anchor
   (`secp256k1_gpu_columns_provider_anchor`) — it tests the CMake/linker
   *mechanism* generically. Cross-file anchor-name *consistency* for the
   real production anchor is instead checked structurally by
   `ci/check_windows_cuda_contract.py`.
