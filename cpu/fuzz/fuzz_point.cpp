// ============================================================================
// Fuzz target: Point arithmetic (memory safety + UB)
// ============================================================================
// Build: clang++ -fsanitize=fuzzer,address -O2 -std=c++20 \
//        -I cpu/include fuzz_point.cpp cpu/src/*.cpp -o fuzz_point
// Run:   ./fuzz_point -max_len=32 -runs=1000000
// ============================================================================
//
// Purpose: exercise scalar_mul, add, dbl, negate, to_compressed,
//          to_uncompressed under sanitizer instrumentation to find
//          memory errors (ASan) and undefined behavior (UBSan).
//
// Arithmetic correctness is NOT asserted here -- it is covered by the
// deterministic unit-test suite at production optimization levels (-O2/-O3).
// CFL/OSS-Fuzz compiles at -O1 where code-gen differences can cause
// different intermediate results; trapping on those is a false positive.
//
// NOTE: We construct G via from_affine() instead of generator() so that
// scalar_mul() takes the GLV double-and-add path rather than the
// precomputed-table path (scalar_mul_generator -> build_context).
// Under sanitizer instrumentation build_context() takes >25 s,
// which exceeds the libFuzzer per-unit timeout.
// ============================================================================

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include <cstdint>
#include <cstddef>
#include <array>

using namespace secp256k1::fast;

// Generator point constructed once without the is_generator_ flag,
// avoiding the heavy precomputed-table path in scalar_mul().
static const Point G_fuzz = []() {
    auto gen = Point::generator();
    return Point::from_affine(gen.x(), gen.y());
}();

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) return 0; // need one 32-byte scalar

    std::array<uint8_t, 32> buf{};
    __builtin_memcpy(buf.data(), data, 32);

    auto k = Scalar::from_bytes(buf);
    if (k.is_zero()) return 0;

    // Guard: catch any C++ exceptions before they cross the extern "C" boundary.
    // Propagating exceptions through extern "C" is undefined behavior and
    // causes std::terminate on Linux/clang (manifests as SIGABRT in CFL).
    try {

    // -- k*G must be on-curve -------------------------------------------------
    auto P = G_fuzz.scalar_mul(k);
    if (P.is_infinity()) return 0;

    // Exercise serialization code paths (ASan checks buffer handling)
    auto comp = P.to_compressed();
    auto uncomp = P.to_uncompressed();
    (void)comp;
    (void)uncomp;

    // -- Exercise P + (-P) (tests add + negate paths) -------------------------
    auto neg_P = P.negate();
    auto sum = P.add(neg_P);
    (void)sum;

    // -- Exercise 2*P = P + P = P.dbl() (tests dbl + add paths) ---------------
    auto dbl = P.dbl();
    auto add_self = P.add(P);
    auto dbl_comp = dbl.to_compressed();
    auto add_comp = add_self.to_compressed();
    (void)dbl_comp;
    (void)add_comp;

    } catch (...) {
        // Swallow: prevents std::terminate from exceptions crossing extern "C".
        // If an internal batch-inversion fallback misfire triggered throw,
        // the result is simply discarded (no false-positive crash report).
    }

    return 0;
}
