# CUDA Architecture Policy

This document is the implementation record for the explicit CUDA architecture
policy applied in `CMakeLists.txt` (outer Secp256K1fast suite),
`libs/UltrafastSecp256k1/CMakeLists.txt` (standalone library), and
`libs/UltrafastSecp256k1/cmake/CudaArchPolicy.cmake` (shared policy module).

It replaces the previous behavior, where both entry points silently defaulted
`CMAKE_CUDA_ARCHITECTURES` to `"89"` (Ada Lovelace) whenever the caller did not
supply a value, and `src/cuda/CMakeLists.txt` carried a second, redundant
silent fallback to the same value. Neither fallback is architecturally honest:
no single bounded architecture list is universal, and a silent default's
failure mode is a runtime `cudaErrorNoKernelImageForDevice` (or worse, a
silently different code path) on hardware outside the guessed list — not a
clear configure-time signal.

## Precedence (never violated)

1. **Explicit `-DCMAKE_CUDA_ARCHITECTURES=...`** on the command line (or an
   equivalent CMakePresets.json `cacheVariables` entry). Always wins.
2. **`CUDAARCHS` environment variable.** Honored manually (not relying solely
   on CMake's own `>=3.20` auto-seeding) so behavior is identical on the
   `cmake_minimum_required(VERSION 3.18)` floor this project declares and on
   modern CMake.
3. **`-DSECP256K1_CUDA_ARCH_PROFILE=<name>`** — an explicit, opt-in, named
   repository profile. Only consulted when both 1 and 2 are absent. Never a
   parallel override mechanism: it only ever populates
   `CMAKE_CUDA_ARCHITECTURES`, at this precedence tier, exactly like the
   single hardcoded value it replaces.
4. **Fail-fast.** With `SECP256K1_BUILD_CUDA=ON` and none of 1–3 supplied,
   configuration halts with `FATAL_ERROR` before any CUDA target is created,
   listing the full profile menu. CMake's own compiler-detection step
   (`check_language(CUDA)` / `enable_language(CUDA)`) still runs and still
   invokes `nvcc` — that is unavoidable and is not what this policy prevents.
   What it prevents is *choosing an architecture list* without being told to.

The policy lives in exactly one place — `CudaArchPolicy.cmake`'s
`ufsecp_apply_cuda_arch_profile()` — and is `include()`d by both the
standalone library entry point and (indirectly, by delegation) the outer
suite. The outer suite's own `CMakeLists.txt` no longer sets any
`CMAKE_CUDA_ARCHITECTURES` value of its own; it only detects which precedence
tier applies and logs it, then lets the submodule decide.

## Named profiles

| Profile | Value | Guard | Deterministic (real/SASS) coverage | Notes |
|---|---|---|---|---|
| `local-native` | `native` | CMake ≥ 3.24 | Exactly the detected local device | Compile-time nvcc diagnostic if the toolkit can't target the detected device; no configure-time device probe is attempted. |
| `ci-bounded-recent` | `89-real;89-virtual;120-real;120-virtual` | nvcc ≥ 13.0 | sm_89, sm_120 | sm_75/80/86/87 have **zero** path (89 > those numbers). sm_90/100/103/110/121 are PTX-JIT-only, driver-conditional. |
| `legacy-compat` | `52-virtual;89-real;89-virtual` | nvcc >= 12.0 AND < 13.0 (both bounds REJECTED — nvcc 11.x or older AND nvcc 13.x are both refused before any target is built) | sm_89 | PTX-JIT floor at compute_52. nvcc 13 dropped compute_50–61 virtual-architecture support, so this profile refuses to silently keep claiming compute_52 support under nvcc 13 — it hard-fails instead. nvcc 11.x and older are never an accepted toolchain for this repository — the lower bound exists so this profile can never be satisfied by an unvetted, older compiler either. |
| `redistributable` | `75-virtual;86-real;89-real;120-real;120-virtual` | nvcc ≥ 13.0 | sm_86, sm_89, sm_120 | PTX-JIT floor at compute_75 for everything else, driver-conditional. Broader than `ci-bounded-recent` but still not universal; a packager shipping this fatbin must publish its own per-architecture coverage matrix. |

None of these profiles is the default. `SECP256K1_BUILD_CUDA=ON` with no
architecture guidance at all always fails configuration (see Precedence #4).

## PTX reachability relation (why "driver-conditional" is not "works")

A virtual (PTX) target `V` embedded in a fatbin may be JIT-compiled by the
CUDA driver to run on a physical device `D` if and only if **both**:

1. `D`'s compute capability >= `V`'s compute capability number (forward-only —
   a virtual target can never JIT to a device with a *lower* compute
   capability than itself), **and**
2. the deployed driver's bundled PTX-JIT compiler can parse the PTX ISA
   version emitted by the toolkit that produced that PTX (a toolkit-wide
   property — empirically ISA 8.0 for nvcc 12.0 and ISA 9.2 for nvcc 13.2 on
   this development machine, independent of which target arch number was
   requested).

A real (SASS) cubin entry is treated as an **exact-target match only**: a
cubin compiled with `code=sm_89` executes on sm_89 devices and nowhere
else — it does not extend upward even to a numerically adjacent device
(e.g. a `120-real` entry does not cover `sm_121`).

## Provenance of the coverage tables and measured costs

The per-architecture reachability tables, fatbin object sizes, and compile
wall-times referenced above trace to
`workingdocs/secondary_invariant_constants/cuda_arch_fatbin_policy_claude_v7.json`
(and, transitively, v5/v6 of the same artifact), all measured on this same
development machine (RTX 5060 Ti, driver 580.173.02) with real `nvcc` /
`ptxas` / `cuobjdump` / `cmake` / CTest invocations against the production
`gpu_backend_cuda.cu` translation unit. Per this repository's benchmark
policy, no number in this document is an estimate or an extrapolation from
different hardware — every number is either directly measured on this
machine or explicitly re-cited (with that label) from a prior real
measurement on this same machine. This document does not restate the full
per-architecture tables from the v7 JSON; consult that artifact directly for
`sm75`…`sm121` cell-by-cell detail.

## Implementation-session verification (this task)

This section records what was actually configured/built during the
implementation of this policy (`cuda-architecture-policy-implementation-claude-v1`),
as distinct from the architecture/coverage research recorded above.

- CPU-only configure (`SECP256K1_BUILD_CUDA` left at its default `OFF`):
  verified to succeed unaffected by this policy change.
- Explicit `-DCMAKE_CUDA_ARCHITECTURES=120-real` (precedence tier #1) against
  the local RTX 5060 Ti with nvcc 13.2.78 (`-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc`):
  verified to configure and build the `secp256k1_cuda_lib` target
  successfully — `cuobjdump -lelf` on the resulting object confirms a real
  `sm_120` ELF cubin (`secp256k1.cu.1.sm_120.cubin`), and the linked static
  library (`libsecp256k1_cuda_lib.a`) is 3,761,100 bytes. This is a
  deterministic real-SASS build for this machine's own GPU, not a claim
  about any other device.
- The default CUDA toolkit on this machine's `PATH` is nvcc 12.0.140 (ceiling
  compute_90); nvcc 13.2.78 is available at `/usr/local/cuda-13.2/bin/nvcc`
  and must be selected explicitly (`-DCMAKE_CUDA_COMPILER=...`) for any
  profile or explicit value that targets `compute_100` and above (including
  `120-real`, `ci-bounded-recent`, `redistributable`).
- `libs/UltrafastSecp256k1/ci/check_cuda_arch_policy.py` exercises, against
  fresh (never-reused) `-B` build directories at both the root suite and
  standalone library entry points: explicit `-D` precedence, `CUDAARCHS`
  precedence, missing-input fail-fast, invalid-profile fail-fast, each named
  profile's happy path, and each profile's version-guard failure path.

## Named preset compiler pin (RTX 5060 Ti)

This machine's default `nvcc` on `PATH` is 12.0.140 (`/usr/bin/nvcc`, ceiling
compute_90). `SECP256K1_CUDA_ARCH_PROFILE=ci-bounded-recent` (used by both
named RTX 5060 Ti presets — `linux-cuda-5060ti` in the outer suite's
`CMakePresets.json` and `cuda-release-5060ti` in
`libs/UltrafastSecp256k1/CMakePresets.json`) requires nvcc >= 13.0. Without an
explicit compiler pin, a fresh invocation of either preset would resolve the
`PATH` default (nvcc 12.0) and `FATAL_ERROR` on the `ci-bounded-recent`
version guard instead of building `sm_120`.

Both named 5060 Ti presets therefore pin `CMAKE_CUDA_COMPILER` to
`/usr/local/cuda-13.2/bin/nvcc` — the real, installed nvcc 13.2.78 toolkit on
this documented host. This is deliberate and host-specific: these are named
**local-hardware** presets (the precedence-#3 named-profile tier is opt-in
by design, and pinning a compiler path alongside it is consistent with that
same "you are choosing this specific hardware/toolchain" contract). Generic,
portable presets (`cuda-release`, `cuda-debug`, `windows-cuda`, `bch-gpu`)
are NOT given a hardcoded compiler path — they use `CMAKE_CUDA_ARCHITECTURES`
values (`native` or an explicit numeric list) that do not require a specific
toolkit version, so they stay portable across machines.

`libs/UltrafastSecp256k1/ci/check_cuda_arch_policy.py` includes static
(no-cmake-invocation) checks that assert both named 5060 Ti presets pin an
explicit `CMAKE_CUDA_COMPILER` whenever they select an nvcc>=13-requiring
profile, that the pinned path exists on this machine, and that the pinned
compiler really does report major version >= 13 — so a future edit that
silently drops the pin (reintroducing the exact defect this document
describes) fails this gate immediately.

## Self-hosted CI wiring (gpu-selfhosted.yml)

`libs/UltrafastSecp256k1/.github/workflows/gpu-selfhosted.yml` runs on a
self-hosted RTX 5060 Ti runner. Before the environment manifest and both
CUDA configure steps, a "Select CUDA 13 toolchain (preflight)" step:

1. Resolves the CUDA 13 nvcc path (default `/usr/local/cuda-13.2/bin/nvcc`,
   overridable via `UFSECP_CUDA13_NVCC` if the toolkit ever moves).
2. Verifies the binary exists and is executable — fails loudly (not a silent
   fallback to the wrong `PATH` default) if not.
3. Verifies its reported major version is >= 13 — fails loudly if not.
4. Exports exactly one `CUDACXX` value (`$GITHUB_ENV`) that every later step
   in the job uses — the environment manifest step reports this exact value
   (never a re-derived `nvcc --version` off `PATH`), and both CUDA configure
   steps pass `-DCMAKE_CUDA_COMPILER="${CUDACXX}"` explicitly (not relying on
   `CUDACXX` env auto-detection alone).

Immediately after toolchain selection, a "CUDA architecture policy gate" step
runs `ci/check_cuda_arch_policy.py` for real — this is what makes the gate an
actual CI check instead of a standalone script nothing invokes. The script's
outer-suite scenarios are skipped (loudly, via an explicit `[SKIP]` message,
not silently) on this runner because it checks out only the standalone
`UltrafastSecp256k1` repository — there is no nested Secp256K1fast outer
suite on that runner's disk; the standalone-entry scenarios and the static
coherence checks already cover the policy module in full there.
