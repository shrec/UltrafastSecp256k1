# Windows / MSVC — libbitcoin batch sig-verify (`ufsecp_lbtc_*`) measured artifact

**Date:** 2026-06-23
**Host:** Intel Core i7-11700 (Rocket Lake, 8C/16T)
**OS:** Windows 11 Pro (10.0.26200)
**Compiler:** MSVC 19.44.35227 (`cl`, Visual Studio 2022 Enterprise), x64
**Build:** CMake + Ninja, `Release`, libbitcoin profile, CPU only
(`-DSECP256K1_BUILD_LIBBITCOIN=ON -DSECP256K1_BUILD_CABI=ON
-DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON -DSECP256K1_USE_LTO=OFF`).
Default MSVC tuning (`SSE2 + /Ob3`, `WPO` opt-in OFF).

**Engine revision:** `dev` @ `07a1eceee` —
*perf(cpu): batch-verify persistent pool + fused parse + FE52 decompress*.
**Benchmark:** `bench_lbtc_batch` (`std::chrono::steady_clock`, best-of-N, correctness
gate before timing). No GPU backend present → CPU reference path only.

> This is the **Windows / MSVC counterpart of the Linux `_mt` measurement** in the
> `07a1eceee` commit. It attaches the per-hardware JSON artifact that
> `docs/LIBBITCOIN_PERF_MATRIX_STATUS.json` (surface `lbtc_msvc_windows_profile`)
> and `docs/LIBBITCOIN_INTEGRATION.md` previously listed as "owner-side until a
> `bench_lbtc_batch --json` artifact from the same hardware is attached".
> All numbers are this machine only — do **not** copy them as estimates for other
> hardware. They are a *scaling/relative* artifact for the MSVC profile, not a
> library-vs-library or cross-platform absolute claim.

---

## What the commit changed (and what is measured here)

`07a1eceee` makes the libbitcoin bridge `_mt` batch-verify path actually parallelize
**block-sized** batches. Before it, the CPU `_mt` path silently collapsed to **one
thread** for any batch below the fixed 4096-row work-steal chunk, so a node calling
the bridge as a single `_mt` batch saw ~1 active core. Four changes fix it:

1. `ecdsa_batch_verify_mt` / `schnorr_batch_verify_mt` — worker count decoupled from
   the 4096 chunk; bounded by hardware/request and `n/kMinRowsPerThread`.
2. A **process-wide persistent worker pool** (created once, reused; intentionally
   leaked to avoid a Windows DLL-unload loader-lock at static destruction).
3. `ufsecp_ecdsa_verify_opaque_rows_mt` — **fused** parse+verify inside each worker
   chunk (was: serial parse of all rows, an Amdahl ceiling).
4. `pubkey33_to_point` does the field sqrt in `FieldElement52` (the rep verify uses
   internally) and drops the wasted Schnorr-table build.

The serial entry points (`ufsecp_lbtc_verify_ecdsa` etc.) run `max_threads == 1` and
are byte-for-byte identical to `_mt` at `t=1`. So the honest measurement of the
commit is the **`_mt` thread sweep** below (the serial table is the `t=1` baseline).

**Correctness:** every run passes the bench's pre-timing gate —
`all-valid + corruption detected`, `schnorr correctness: PASS`,
`column correctness: PASS`.

---

## 1. `_mt` thread-scaling sweep — the headline (matches the Linux `_mt` measurement)

`bench_lbtc_batch 100000 3 10000 --mt-only`, opaque ECDSA rows / BIP-340 Schnorr rows.

| path | t=1 | t=2 | t=4 | t=8 | t=16 |
|---|--:|--:|--:|--:|--:|
| **ECDSA-row `_mt`** (sig/s)   | 8,628 | 17,077 | 31,275 | 51,705 | **69,995** |
| speedup vs t=1                | 1.00× | 1.98×  | 3.62×  | 5.99×  | **8.11×**  |
| **Schnorr-row `_mt`** (sig/s) | 44,975 | 60,769 | 112,835 | 166,838 | **235,808** |
| speedup vs t=1                | 1.00×  | 1.35×  | 2.51×   | 3.71×   | **5.24×**   |

- ECDSA scales **8.1×** at 16 threads on an 8C/16T part — near-linear to the 8
  physical cores plus a solid SMT gain. This is exactly the commit's win: a
  block-sized batch now spreads across cores instead of running on one.
- Schnorr starts higher per-thread (it batches through a Pippenger MSM, so each
  thread does chunkier work) and scales **5.2×**.
- `t=1` reproduces the serial baseline in §2 (ECDSA 8,628 ≈ 8,784 sig/s; Schnorr
  44,975 ≈ 44,869 sig/s), confirming the `_mt`-at-`t=1` == serial guarantee.

Raw artifact: [`lbtc_mt_msvc_20260623.json`](lbtc_mt_msvc_20260623.json).

## 2. Serial baseline (default non-`_mt` entry points)

`bench_lbtc_batch 200000 5 20000`. These are the functions a node calls when it
already shards across its own pool (one core per controller call).

| kind | best (s / 200k) | throughput (sig/s) | note |
|---|--:|--:|---|
| ECDSA-row            | 22.77 | 8,784  | opaque per-sig verify |
| Schnorr-row          |  4.46 | 44,869 | BIP-340 batch MSM (sublinear) |
| ECDSA-columns        | 23.61 | 8,470  | per-sig verify |
| Schnorr-columns      | 22.07 | 9,062  | per-row fallback (no MSM on the column path) |
| ECDSA-collect        | 22.32 | 8,962  | in-place key-cell verdict |
| Schnorr-collect      |  4.22 | 47,379 | batch MSM |
| ECDSA-col-collect    | 22.16 | 9,026  | |
| Schnorr-col-collect  | 22.13 | 9,039  | per-row fallback |

The Schnorr **row** path (MSM, 44.9k/s) is ~5× the Schnorr **column** path
(per-row, 9.1k/s); ECDSA is per-sig on every layout. This asymmetry is a property
of the bridge dispatch, not of MSVC.

Raw artifact: [`lbtc_batch_msvc_20260623.json`](lbtc_batch_msvc_20260623.json).

## 3. Notes / caveats

- **Absolute throughput is MSVC-bound, not the point.** The per-core verify is
  limited by MSVC's structural `__int128` gap (no inline `MULX/ADCX/ADOX`; field mul
  goes through a non-inlined MASM `call`) — see
  [`windows_msvc_vs_clang_20260614.md`](windows_msvc_vs_clang_20260614.md). The
  *scaling* (8.1× / 5.2×) is what `07a1eceee` delivers, and it reproduces on MSVC.
- **For higher absolute Windows numbers, build with `clang-cl`** (the
  `windows-clang-cl` preset): its codegen uses the inline MULX field kernels and is
  ~3.3–3.6× faster per core than `cl` on the compound verify path, while staying in
  the MSVC toolchain. The `_mt` scaling shown here is compiler-independent.
- Cross-platform: this is Windows x86_64. Windows ARM64 (NEON baseline) is a
  separate benchmark surface.

## 4. Reproduce

From a **Developer Command Prompt for VS 2022** (so `cl` + `ninja` are on `PATH`),
in the library root:

```powershell
cmake -S . -B out/libbitcoin-msvc -G Ninja `
  -DCMAKE_BUILD_TYPE=Release `
  -DSECP256K1_BUILD_LIBBITCOIN=ON `
  -DSECP256K1_BUILD_CABI=ON `
  -DSECP256K1_BUILD_LIBBITCOIN_BENCH=ON `
  -DSECP256K1_USE_LTO=OFF
cmake --build out/libbitcoin-msvc --target bench_lbtc_batch

$exe = "out/libbitcoin-msvc/include/ufsecp/bench_lbtc_batch.exe"
# serial baseline + JSON artifact
& $exe 200000 5 20000 --json out/libbitcoin-msvc/lbtc_batch_msvc.json
# _mt thread-scaling sweep (1,2,4,8,hw) + JSON artifact
& $exe 100000 3 10000 --mt-only --json out/libbitcoin-msvc/lbtc_mt_msvc.json
```

The helper script `bench_tools/build_lbtc_bench.ps1` (parent workspace) wraps the
vcvars import + configure + build + run for both `msvc` and `clangcl`.
