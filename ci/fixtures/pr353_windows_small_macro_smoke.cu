// ci/fixtures/pr353_windows_small_macro_smoke.cu
//
// Adversarial CUDA build-input regression fixture -- PR #353 Windows/MSVC
// CUDA hardening (Eric Voskuil / evoskuil), reproduced by hand in this
// repo's secp256k1::cuda::field_mul_small parameter rename.
//
// WHY THIS EXISTS
// ----------------
// Windows' RPC/COM header <rpcndr.h> (pulled in transitively through
// <windows.h> -> <rpc.h> whenever a CUDA host translation unit on MSVC
// touches the Win32/COM API before this header, directly or via a project
// dependency) contains the legacy MIDL-compatibility line:
//
//     #ifndef small
//     #define small char
//     #endif
//
// (rpcndr.h also defines `#define hyper __int64` for the same reason.) Any
// bare identifier named `small` downstream of that include -- including a
// function *parameter name*, which is not part of the type system in C++ --
// is textually substituted by the preprocessor before the compiler's parser
// ever runs. field_mul_small's multiplier parameter used to be named
// `small`; under the poisoned macro `uint32_t small` becomes the ill-formed
// `uint32_t char`.
//
// This fixture reproduces the collision deterministically, on any host
// OS/toolchain (no real Windows SDK required): it defines the same poison
// macro by hand before including secp256k1.cuh.
//
// FAIL-BEFORE / PASS-AFTER CONTRACT
// ----------------------------------
//   fail-before (parameter named `small`): nvcc rejects this translation
//   unit while parsing field_mul_small's declaration -- error class
//   "invalid combination of type specifiers" / "type name is not allowed"
//   (exact wording is nvcc-version-dependent; match on class, not string --
//   verified identical error class on nvcc 12.0 and 13.2 on this machine).
//   pass-after (parameter renamed to `factor`): this file compiles cleanly
//   (exit 0) because `factor` is not a reserved rpcndr.h macro.
//
// Measured locally (this session), nvcc 12.0:
//   $ nvcc -c --std=c++17 -I src/cuda/include -I include \
//       ci/fixtures/pr353_windows_small_macro_smoke.cu -o /tmp/smoke.o -arch=sm_75
//   src/cuda/include/secp256k1.cuh(1234): error: invalid combination of type specifiers
//   src/cuda/include/secp256k1.cuh(1241): error: type name is not allowed
//   2 errors detected ... / exit 2   (against the pre-fix header)
//   exit 0, one pre-existing unrelated warning at line 894 (mul_shift_384,
//   untouched by this fix)   (against the fixed header)
//
// Never rename `small` back here to "fix" a compile failure -- the whole
// point is to poison `small` unconditionally, exactly like a stray Windows
// SDK header would. If field_mul_small's parameter is ever renamed back to
// `small`, this fixture stops compiling and the mandatory CUDA compile-check
// CI job (.github/workflows/ci.yml, job `cuda-compile-check`) fails closed --
// no Windows host or GPU hardware required (compile-only, nvcc -c).
//
// See docs/WINDOWS_CUDA_BUILD_CONTRACT.md for the full contract and
// ci/check_windows_cuda_contract.py for the structural gate that also
// rejects a reserved `small` parameter directly in secp256k1.cuh and
// verifies this fixture is reachable from a CUDA-capable, non-self-hosted,
// push/pull_request-triggered CI job.

// Simulate the exact Windows rpcndr.h collision without requiring the real
// Windows SDK (portable across Linux/macOS/Windows CI runners).
#define small char

#include "secp256k1.cuh"

#undef small

using namespace secp256k1::cuda;

// Actually calls field_mul_small (not just parses its declaration) with 3
// real arguments, matching real kernel usage. Not launched by this fixture:
// nvcc -c (device codegen via ptxas/cicc) is sufficient and requires no GPU
// hardware; the self-hosted GPU CI runs the full runtime test suite
// separately (gpu-selfhosted.yml).
__global__ void pr353_field_mul_small_smoke_kernel(const FieldElement* a,
                                                     FieldElement* r) {
    field_mul_small(a, 7u, r);   // 7 == secp256k1 curve constant b
}
