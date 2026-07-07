# Windows MSVC `cl` — FE52-compute ECDSA/Schnorr verify (2026-06-23)

Machine: i7-11700, Windows 11, MSVC `cl` (VS 2022), no clang-cl, no GPU.
Comparison built from the same `.bench_ext/secp256k1` (libsecp256k1) tree.

## The problem

On plain MSVC `cl` the secp256k1 ECDSA-verify hot path (`u1·G + u2·Q`) trailed
libsecp. Session start: **0.42×** of libsecp. Root cause was NOT the field in
isolation (Ultra's 5×52 field micro-bench already *beats* libsecp on `cl`) — it was
the field codegen **in the point-op context**.

## Root cause (workflow segment-diff vs libsecp `int128_struct`)

`u128_compat` implemented its 128-bit accumulator with **value-returning operator
overloads** (`operator*`, `operator+`). A Comba column sum `d = p0+p1+p2+p3`
materialises a fresh 16-byte `{lo,hi}` temporary per term. In isolation MSVC keeps
them in registers, so the single-mul micro-bench *wins*. But inside a point op
(7+ live accumulators) those temporaries **spill** — and the verify lost.

libsecp avoids this with **pointer-accumulation** (`int128_struct_impl.h:58-72`):
`secp256k1_u128_accum_mul(&d, a, b)` mutates ONE named accumulator in place.

## The fix

* **Step A** — added `__forceinline` free helpers `u128_mul` / `u128_accum_mul` /
  `u128_accum_u64` to `u128_compat.hpp` (MSVC `_umul128` + libsecp carry
  `r->hi += hi + (r->lo < lo)`; native `__int128` = the operator forms, neutral) and
  rewrote the generic `#else` of `fe52_mul_inner` / `fe52_sqr_inner` (both CT and
  `_var`) to pointer-accumulation. Arithmetic byte-identical; carry data-independent
  (CT-safe).
* **FE52-compute** — `SECP256K1_FE52_COMPUTE` (config.hpp) decouples FE52 *compute*
  from FE52 *storage*. On `cl` the Point STORAGE stays 4×64; the ECDSA/Schnorr
  verify dual-mul (w=15 G-tables) runs in 5×52, bridging once with
  `to_jac52`/`from_jac52`. Linux is unchanged (native `__int128`, FE52 storage).

## Measured — the in-context PoC that proved it (`bench_tools/poc/poc_fe_main.cpp`)

Jacobian-double-shaped chain (5 sqr + 2 mul + adds, A..F live), ns/op:

| field idiom            | isolated single-mul | **in-context double** |
|------------------------|--------------------:|----------------------:|
| operator-chain (pre-A) |            16.1 ns ⚡ | 219 ns (1.37× *slower* than 4×64) |
| pointer-accum (Step A) |            26.6 ns  | **142.7 ns (0.88× — beats 4×64)** |

The isolated micro-bench is misleading (no register pressure); only the in-context
measurement predicts the verify.

## Measured — verify vs libsecp (`bench_vs_libsecp`, best-of-21, ratio = stable metric)

| op                          | result          |
|-----------------------------|-----------------|
| ECDSA verify (single)       | **1.02× Ultra** (was 0.42× at session start; 0.87× with 4×64) |
| ECDSA verify (precomp warm) | **1.03× Ultra** |
| Schnorr verify (raw 64 keys)| **1.23× Ultra** |
| Schnorr verify (xonly+GLV)  | **1.11× Ultra** |
| Schnorr verify (both parsed)| **1.02× Ultra** |
| Schnorr verify (precomp)    | **1.17× Ultra** |

`run_selftest`: **31/31 modules PASS** (KAT + CT-equivalence).

## Measured — libbitcoin batch `_mt`, apples-to-apples @100k rows, best-of-5

| path        | ECDSA t=16 | Schnorr t=16 |
|-------------|-----------:|-------------:|
| 4×64 re-gate| 0.839 s    | 0.542 s      |
| **FE52**    | **0.815 s**| **0.494 s**  |
| Δ           | +2.9%      | **+9.0%**    |

No regression (the batch row is parse-dominated, ~18 µs, which dilutes the ~17%
single-verify gain). Schnorr batch confirmed unaffected by the storage decoupling
(Point STORAGE stays 4×64, so the Pippenger MSM cache footprint is unchanged).

## Takeaway

`cl` ECDSA-verify parity with libsecp is reachable **natively** — no clang-cl, no
GPU. The lever was replicating libsecp's documented `cl` accumulation idiom at the
exact segment where it won, and trusting the *in-context* measurement over the
isolated micro-bench.
