# Known CI Limitations — Pre-existing Failures Not Caused by Recent Commits

This document catalogs CI workflow failures that pre-date the 2026-05-14
CI cleanup cycle and require dedicated investigation beyond a single
commit. Each entry includes the root cause (or best hypothesis), the
affected jobs, and the path to a real fix.

The goal of this document is to keep the green CI surface honest: we
should not silently mark these as "passing" by adjusting tests; nor
should we pretend the fixes are simple one-liners.

---

## 1. FE52 generic (no-asm) Comba multiplication has wrong outputs for certain input patterns

**Affected CI jobs:**
- `CI / Sanitizers (TSan)`
- `CI / Sanitizers (MSan)`
- `CI / Sanitizers (ASan+UBSan)`
- `Security Audit / TSan`
- `Security Audit / ASan + UBSan`
- `Security Audit / MSan`
- `CI / coverage` (gcov build, Debug, no-asm)
- `CI / linux (gcc-13, Debug)` (sometimes, depending on flags)
- `CI / linux (clang-17, Debug)`

**Symptom:**
- `field_52` test: `FAIL: mul(mid, large)`, `mul(large, large)`, etc.
  (long list of specific input pair failures)
- `selftest`: `Boundary Scalar KAT (2^128 - 1) * G` fails with wrong X
- `Point::add` triggers `SECP_ASSERT_ON_CURVE FAILED: point not on
  secp256k1` in Debug builds (because `is_on_curve` uses FE52 mul
  internally and gets a wrong y² check)

**Root cause (hypothesis):**
The C++ Comba/Barrett 5x52 multiplication kernel in
`src/cpu/include/secp256k1/field_52_impl.hpp` (line 670+ generic
fallback when `SECP256K1_HAS_ASM` is false) produces incorrect results
for specific large input patterns. The hand-written x86_64 ADCX/ADOX
asm path produces correct results — the bug is in the portable C++
fallback.

This is unrelated to the wasm32 `__int128` emulation bug (that one is
fixed via `u128_compat`). The TSan/MSan/ASan/coverage builds DO have
native `__int128` available; they just have `-DSECP256K1_USE_ASM=OFF`
(necessary because Clang sanitizers can't instrument hand-written
asm).

**Reproducer locally:**
```bash
cmake -S . -B out/test-noasm -DCMAKE_BUILD_TYPE=Debug \
    -DSECP256K1_USE_ASM=OFF -DSECP256K1_BUILD_TESTS=ON
cmake --build out/test-noasm --target run_selftest
./out/test-noasm/src/cpu/run_selftest smoke
#   → 17/25 tests passed; "Boundary Scalar KAT" mismatch
```

**Path to fix:**
A) Audit `field_52_impl.hpp` lines 670–746 (the generic mul fallback).
   Compare the carry chain limb-by-limb against the working x86_64 asm
   kernel at line 488. The bug is likely a missing carry propagation or
   an over-flow in one of the column sums.
B) Alternatively, replace the generic mul with a call to the same
   `mulx64`/`adcx64` portable helpers used by `u128_compat`. The helpers
   are already proven correct (240k-op parity test in `test_u128_compat_parity_run`).

**Pre-existence verified:** failure observed at commits `100bd19d`,
`a0b35c8c`, `47aba907`, and earlier (>= 6 consecutive `Sanitizers`
runs on the same code path).

---

## 2. macOS Metal GPU batch verify tests return UNSUPPORTED on macos-latest runners

**Affected CI jobs:**
- `CI / macos (Release)`

**Symptom:**
After the link-step fix in `863fc8a8` (which solved the
`schnorr_r_zero_ct_standalone` undefined symbol error), the macOS
runner now reaches the test phase and fails on:
- `FAIL: generator_mul_batch unexpected error: 102 (GPU kernel launch failed)`
- `FAIL: ecdsa_verify_batch invalid pubkey: UNSUPPORTED ...`
- `FAIL: schnorr_verify_batch invalid signature: UNSUPPORTED ...`
- 6 more `*_batch invalid input` patterns

**Root cause (hypothesis):**
The Metal batch verify functions return `UFSECP_ERR_UNSUPPORTED` (or
similar) when handed adversarial inputs on the GitHub macOS runner's
software-Metal device. The tests' assertion was written assuming the
backend returns `ERR_BAD_INPUT` or marks the per-slot result invalid.

This is a test-vs-runtime mismatch on a GitHub-hosted macOS image
(software Metal device, no GPU acceleration). Real Apple Silicon
hardware returns the expected `ERR_BAD_INPUT`.

**Path to fix:**
Loosen the test assertion to accept `UNSUPPORTED` as a valid response
when running on a software Metal device — or wrap the test in an
`if (!is_real_metal_device()) advisory-skip`.

**Pre-existence verified:** Failures present at `100bd19d`, `a0b35c8c`,
`47aba907` — independent of recent shim/wasm/u128 work.

---

## 3. wasm KAT Point operations produce wrong outputs

**Affected CI jobs:**
- `CI / wasm`

**Symptom:**
```
[3] Point operation KAT
  [FAIL] 2G mismatch (line 68)
  [FAIL] 3G mismatch
  [FAIL] s2G mismatch
  ...
[4] ECDSA KAT — verify fails
[5] Schnorr KAT — verify fails
```

**Root cause:**
Emscripten's Clang defines `__SIZEOF_INT128__=16` on wasm32 but
emulates `__int128` via compiler-rt `__multi3`, which produces wrong
results for the 5x52 Comba multiply path.

**Status:** Partial fix in place. Commit `a0b35c8c` introduced
`u128_compat` (portable struct) and switched FE52 to it. Stress test
(`test_u128_compat_parity_run`) verifies 240,000 ops match native
`__int128` byte-for-byte.

The wasm KAT may still fail because:
- `src/cpu/src/scalar.cpp` lines 559+ and 1122+ still have unguarded
  `__int128` paths that activate on wasm32.
- The wasm build may need a fresh emcc invocation to pick up the new
  `u128_compat.hpp` header.

**Path to fix:**
A) Gate `scalar.cpp:559` and `scalar.cpp:1122` with `&& !defined(SECP256K1_NO_INT128)`
   and replace `__int128` use with `u128_compat`.
B) Run the wasm KAT locally with an Emscripten SDK installed
   (`emsdk install latest && emsdk activate latest`) to verify before
   pushing.

**Pre-existence verified:** Failure present at all commits going back
several weeks.

---

## 4. linux-arm64 / linux-riscv64 QEMU smoke

**Affected CI jobs:**
- `CI / linux-arm64`
- `CI / linux-riscv64`

**Symptom:**
QEMU user-mode emulation runs the cross-compiled binaries. RISC-V
reports "Module 17: hash accel" FAILED despite all internal assertions
passing (`678 passed, 0 failed`). The module return code is non-zero
even though counters say pass.

**Root cause (hypothesis):**
Either:
- A failure inside the test harness's exit-code assembly (e.g., the
  test sets `errno` after writing `g_pass` and the exit code path
  picks up the wrong value), or
- A QEMU user-mode bug in a specific intrinsic (likely `__builtin_clz`
  or a SIMD-aware sha256/ripemd path)

**Path to fix:**
Run the failing module under QEMU directly:
`qemu-riscv64 -L /usr/riscv64-linux-gnu \
    ./build/riscv64/audit/test_hash_accel_standalone`
Look for the actual aborting step.

**Pre-existence verified:** Present going back >= 4 weeks.

---

## 5. Windows (Release) — fast fail

**Affected CI jobs:**
- `CI / windows (Release)`
- `Benchmark Dashboard / benchmark-windows`

**Symptom:**
Build fails in ~3 minutes. No detailed log captured yet (need to dump
the windows job log specifically).

**Path to fix:**
`gh api ... /actions/runs/<id>/jobs --jq '.jobs[] | select(.name=="windows (Release)") | .id'`,
then `gh api ... /actions/jobs/<id>/logs` to see the actual error.

---

## 6. rocm

**Affected CI jobs:**
- `CI / rocm`

**Symptom:**
HIP/ROCm compilation step fails. May be a HIP toolchain mismatch on
the runner or a missing AMD GPU feature.

**Path to fix:**
Dump `rocm` job log and identify the failing build step.

---

## What IS fixed in the 2026-05-14 CI cleanup cycle

The following CI jobs went from RED → GREEN during this work:

| Job | Before | After | Commit |
|-----|--------|-------|--------|
| `Gate / PR-Push` | failure (30+ runs) | success | `a90f70bf` |
| `Block 3 / Shim Security Gate` | failure (6+ runs) | success | `a0b35c8c` |
| `Block 3 / CAAS Security Gates` | failure (TEST_MATRIX drift) | success | `a90f70bf` |
| `Block 3 / Security Gate (aggregator)` | failure | success | `a90f70bf` |
| `Gate / Final Verdict` | failure | success | `a90f70bf` |
| `CI / android (armeabi-v7a)` | failure (`__int128 not supported`) | success | `ec2464e9` |
| `CI / linux (gcc-13, Debug)` 1-module FAIL | metal advisory regression | reverted | `7f12b2cd` |

Build/link errors resolved (no longer aborting the CI step):

- macOS undefined symbols (`_secp256k1_context_create` etc. — linker
  error in `schnorr_r_zero_ct_standalone` and `musig_ka_cap_standalone`)
- TSan undefined symbols (same shim guard)
- API drift in `test_regression_hash_three_block_bounds.cpp`
- Subprocess aborts in `exploit_encoding_memory_corruption` and
  `exploit_shim_der_bip66` (NULL ctx → illegal_callback → abort)
- Debug `SECP_ASSERT_SCALAR_VALID FAILED: scalar is zero` in
  `schnorr_keypair_create` (redundant assertion contradicting the
  graceful zero-scalar handling already in the function)
