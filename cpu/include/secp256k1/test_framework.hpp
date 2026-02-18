#ifndef SECP256K1_TEST_FRAMEWORK_HPP
#define SECP256K1_TEST_FRAMEWORK_HPP
#pragma once

// ============================================================================
// UltrafastSecp256k1 — Built-in Test Framework
// ============================================================================
//
// Self-contained test infrastructure that is an organic part of the library.
// No external test frameworks needed. Portable to ANY platform.
//
// Architecture:
//   TestCategory (enum) → groups related tests (field, scalar, point, ...)
//   TestResult          → per-check pass/fail with message
//   TestSuite           → collects results, runs categories, reports
//
// Usage:
//   secp256k1::test::TestSuite suite;
//   suite.run(TestCategory::All);           // run everything
//   suite.run(TestCategory::FieldArith);    // single category
//   suite.run({TestCategory::Point, TestCategory::CT});  // multiple
//   suite.summary();                        // print results
//   return suite.failed() ? 1 : 0;
//
// On embedded/RTOS, individual categories can run standalone without
// any I/O dependency — results are stored in a POD counter struct.
// ============================================================================

#include <cstdint>
#include <cstddef>

namespace secp256k1::test {

// ── Test Categories ──────────────────────────────────────────────────────────
// Each enum maps to a logical group of tests. Port authors can select
// which categories to run based on platform capabilities.

enum class TestCategory : uint32_t {
    // ── Core Arithmetic ──
    FieldArith          = 0x0001,  // FieldElement +, -, *, sqr, inv, normalize
    FieldConversions    = 0x0002,  // from_hex, to_hex, from_bytes, to_bytes, from_limbs
    FieldEdgeCases      = 0x0004,  // zero, one, p-1, max limbs, double overflow
    FieldInverse        = 0x0008,  // standalone inverse, batch inverse, all algos
    FieldRepresentations= 0x0010,  // 4×64 vs 5×52 vs 10×26 cross-check
    FieldBranchless     = 0x0020,  // cmov, cmovznz, select, is_zero, eq
    FieldOptimal        = 0x0040,  // to_optimal / from_optimal roundtrip
    
    ScalarArith         = 0x0100,  // Scalar +, -, *, inv, neg, is_zero
    ScalarConversions   = 0x0200,  // from_hex, to_hex, from_bytes, to_bytes, bit()
    ScalarEdgeCases     = 0x0400,  // 0, 1, n-1, n (wraps to 0), n+1 (wraps to 1)
    ScalarEncoding      = 0x0800,  // NAF, wNAF encoding correctness

    // ── Point Operations ──
    PointBasic          = 0x1000,  // add, dbl, negate, infinity, generator
    PointScalarMul      = 0x2000,  // scalar_mul, known vectors, iterated
    PointInplace        = 0x4000,  // next/prev/dbl/negate_inplace, add_inplace
    PointPrecomputed    = 0x8000,  // KPlan, predecomposed, precomputed_wnaf
    PointSerialization  = 0x00010000,  // to_compressed, to_uncompressed, from_affine
    PointEdgeCases      = 0x00020000,  // P+O, O+P, P+(-P), O+O, 2·O, n·G=O

    // ── Constant-Time Layer ──
    CTOps               = 0x00040000,  // cmov, cswap, select, masks
    CTField             = 0x00080000,  // ct field arithmetic
    CTScalar            = 0x00100000,  // ct scalar arithmetic
    CTPoint             = 0x00200000,  // complete add, ct scalar_mul

    // ── Advanced Algorithms ──
    GLV                 = 0x00400000,  // GLV decomposition, endomorphism
    MSM                 = 0x00800000,  // Strauss, Pippenger, unified MSM
    CombGen             = 0x01000000,  // Comb generator, CT comb, cache I/O
    BatchInverse        = 0x02000000,  // Montgomery batch inverse, SIMD batch

    // ── Protocols ──
    ECDSA               = 0x04000000,  // sign, verify, deterministic nonce
    Schnorr             = 0x08000000,  // BIP-340 sign/verify
    ECDH                = 0x10000000,  // shared secret computation
    Recovery            = 0x20000000,  // recoverable sig, key recovery
    
    // ── Aggregate categories ──
    // Use these helper values programmatically:
    AllField            = 0x007F,      // all field tests
    AllScalar           = 0x0F00,      // all scalar tests
    AllPoint            = 0x003F0000 & 0xFFFF0000, // all point tests — computed below
    AllCT               = 0x003C0000,  // all CT tests
    AllCore             = 0x0FFFFFFF,  // field + scalar + point + CT + algorithms
    
    All                 = 0xFFFFFFFF,  // everything
};

// Bitwise operators for combining categories
inline constexpr TestCategory operator|(TestCategory a, TestCategory b) {
    return static_cast<TestCategory>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}
inline constexpr TestCategory operator&(TestCategory a, TestCategory b) {
    return static_cast<TestCategory>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}
inline constexpr bool has_flag(TestCategory set, TestCategory flag) {
    return (static_cast<uint32_t>(set) & static_cast<uint32_t>(flag)) != 0;
}

// ── Test Counters (POD, embeddable) ──────────────────────────────────────────
struct TestCounters {
    uint32_t passed   = 0;
    uint32_t failed   = 0;
    uint32_t skipped  = 0;
    
    uint32_t total()     const { return passed + failed + skipped; }
    bool     all_pass()  const { return failed == 0; }
    
    void merge(const TestCounters& other) {
        passed  += other.passed;
        failed  += other.failed;
        skipped += other.skipped;
    }
};

// ── Category name mapping ────────────────────────────────────────────────────
inline const char* category_name(TestCategory cat) {
    switch (cat) {
        case TestCategory::FieldArith:          return "Field Arithmetic";
        case TestCategory::FieldConversions:    return "Field Conversions";
        case TestCategory::FieldEdgeCases:      return "Field Edge Cases";
        case TestCategory::FieldInverse:        return "Field Inverse";
        case TestCategory::FieldRepresentations:return "Field Representations";
        case TestCategory::FieldBranchless:     return "Field Branchless";
        case TestCategory::FieldOptimal:        return "Field Optimal Dispatch";
        case TestCategory::ScalarArith:         return "Scalar Arithmetic";
        case TestCategory::ScalarConversions:   return "Scalar Conversions";
        case TestCategory::ScalarEdgeCases:     return "Scalar Edge Cases";
        case TestCategory::ScalarEncoding:      return "Scalar NAF/wNAF";
        case TestCategory::PointBasic:          return "Point Basic";
        case TestCategory::PointScalarMul:      return "Point Scalar Mul";
        case TestCategory::PointInplace:        return "Point In-Place";
        case TestCategory::PointPrecomputed:    return "Point Precomputed";
        case TestCategory::PointSerialization:  return "Point Serialization";
        case TestCategory::PointEdgeCases:      return "Point Edge Cases";
        case TestCategory::CTOps:               return "CT Primitives";
        case TestCategory::CTField:             return "CT Field";
        case TestCategory::CTScalar:            return "CT Scalar";
        case TestCategory::CTPoint:             return "CT Point";
        case TestCategory::GLV:                 return "GLV Endomorphism";
        case TestCategory::MSM:                 return "Multi-Scalar Mul";
        case TestCategory::CombGen:             return "Comb Generator";
        case TestCategory::BatchInverse:        return "Batch Inverse";
        case TestCategory::ECDSA:               return "ECDSA";
        case TestCategory::Schnorr:             return "Schnorr";
        case TestCategory::ECDH:                return "ECDH";
        case TestCategory::Recovery:            return "Key Recovery";
        default:                                return "Unknown";
    }
}

// ── Complete list of individual categories for iteration ─────────────────────
inline constexpr TestCategory ALL_CATEGORIES[] = {
    TestCategory::FieldArith,
    TestCategory::FieldConversions,
    TestCategory::FieldEdgeCases,
    TestCategory::FieldInverse,
    TestCategory::FieldRepresentations,
    TestCategory::FieldBranchless,
    TestCategory::FieldOptimal,
    TestCategory::ScalarArith,
    TestCategory::ScalarConversions,
    TestCategory::ScalarEdgeCases,
    TestCategory::ScalarEncoding,
    TestCategory::PointBasic,
    TestCategory::PointScalarMul,
    TestCategory::PointInplace,
    TestCategory::PointPrecomputed,
    TestCategory::PointSerialization,
    TestCategory::PointEdgeCases,
    TestCategory::CTOps,
    TestCategory::CTField,
    TestCategory::CTScalar,
    TestCategory::CTPoint,
    TestCategory::GLV,
    TestCategory::MSM,
    TestCategory::CombGen,
    TestCategory::BatchInverse,
    TestCategory::ECDSA,
    TestCategory::Schnorr,
    TestCategory::ECDH,
    TestCategory::Recovery,
};
inline constexpr int NUM_CATEGORIES = sizeof(ALL_CATEGORIES) / sizeof(ALL_CATEGORIES[0]);

} // namespace secp256k1::test

#endif // SECP256K1_TEST_FRAMEWORK_HPP
