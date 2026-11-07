# Validation record

Initial local validation on 2026-09-27, Windows x86-64, RTX 5070 Ti Laptop GPU
(SM 12.0), driver 616.92. Rust NVPTX: nightly-2026-04-02, LLVM 22; native GPU:
CUDA 13.4, upstream `5536321b3af2322cd2b8e90221a5bb9e4587fb7d`.

- Host: 1,539 field vectors and 1,539 scalar vectors against `num-bigint`, with
  full-width point samples against `libsecp256k1`; release tests passed.
- Actual GPU: 773 boundary/random cases of field/scalar operations, including
  generic Barrett multiplication, scalar inversion, and full-width generator
  multiplication, matched the independently checked host path.
- Downstream Pickaxe: 13,920 reconstructed mining candidates passed across
  eight transaction ages, three synthetic keys, partial batches, index
  boundaries, target boundaries, and capped winner readback.
- BCH 2026 VM: 103 accepted and 113 rejected transactions matched both standard
  policy and consensus checks.

Initial complete downstream pipeline trials used 565,248 candidates per batch,
32 points per walk lane, 16 inversions per batch lane, 45 seconds warmup, one
second settling, and eight 12-second ABBAABBA trials. One process held the GPU
exclusively; no compilation ran during timing. Rates aggregate total candidates
over total elapsed time. All trials are retained, including low-clock samples.

| Matched session | Native baseline (million candidates/s) | Rust port | Change |
|---|---:|---:|---:|
| Original Pickaxe | 114.5414 | 122.0914 | +6.59% |
| Current native upstream adapter | 114.7347 | 119.7729 | +4.39% |

These are **mining pipeline** measurements, not standalone library arithmetic
benchmarks. Pickaxe retains its own Rust SHA-256, transaction construction,
fixed-key Montgomery multiplication, and scheduling. The native upstream
adapter uses the same fixed-key specialization. The improvement must not be
attributed entirely to the arithmetic port or extrapolated to CPU signing,
other GPUs, other algorithms, or all UltrafastSecp256k1 APIs.

This is an additive experimental backend. It does not replace native defaults.
Portable CPU operations have functional coverage, not a claim to match the
native assembly/GLV engine's performance. Constant-time auditing, AMD/Intel GPU
backends, and physical testing beyond this NVIDIA device are outside this port.

The pinned package at `e346efe3` was rebuilt and retested. Final matched sessions:

| Matched session | Native baseline (million candidates/s) | Rust port | Change |
|---|---:|---:|---:|
| Original Pickaxe | 114.2391 | 123.1706 | +7.82% |
| Current native upstream adapter | 115.2049 | 120.4858 | +4.58% |

Every trial, compiler/revision metadata, and PTX SHA-256 values are retained in
[`mining-results.json`](mining-results.json). The second native session also had
clock fluctuations; no samples were removed. The candidate was faster in all
four session aggregates, but individual-trial and hardware variability remain.

Final functional checks repeated the 773-vector GPU probe, 13,920 mining
candidates and 216 VM cases. The miner's normal suite passed 266 tests, with
21 explicit ignores and one physical wgpu test filtered. Its existing CUDA
full-pipeline and RFC6979-path tests ran on the physical GPU. Only SM 12.0 has
physical coverage; SM 7.5 compilation passed locally and in CI.

Downstream's subsequent Rust 1.98 Clippy cleanup rebuilt its PTX as
`8e0df066f08e4d0cc5d1f2c1f6343fe268cae01dff05cccc13a951472e38d1c1`.
Raw-byte comparison against the timed `b2081dc6...` artifact found only its
RFC6979 helper changed. All incremental mining kernels and their arithmetic/hash
functions were byte-identical. The physical GPU probe, mining oracle, VM cases
and normal miner suite passed again, including RFC6979. Timings were not
repeated for this helper-only change; both hashes are recorded separately.

The [Rust core workflow](https://github.com/CyberAshven/UltrafastSecp256k1/actions/runs/36317306518)
passed Windows/Linux host checks and both GPU architecture compilations.

Native regression validation used unchanged C++ sources. The first Windows
build exceeded MSVC output-path limits; the shorter `D:/Qubes/build/uf-native-port`
build succeeded. CTest passed 413/438 cases initially. All 24 source-lookup and
console-encoding failures passed when their exact commands were rerun from the
repository root with `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1`. Thus 437 test
processes passed. The remaining OpenSSL cross-check returns its existing
advisory code 77 because optional OpenSSL headers are absent; this is not a
claim that the unmodified CTest invocation was green. Optional Python
coincurve/noble comparisons were also unavailable and are not counted as
independent coverage. No native tests or skip policies were changed.
