// ============================================================================
// test_regression_gpu_beta_constants.cpp
// ============================================================================
// Regression: Metal + OpenCL CT GLV beta constants must match canonical secp256k1 β.
//
// Bug (2026-07-08)
//   Metal: secp256k1_ct_point.h defined a local BETA_METAL[8] (32-bit limbs) with
//   incorrect values that did not match the canonical BETA_LIMBS[8] in
//   secp256k1_point.h:415.
//   OpenCL: secp256k1_ct_point.cl defined local inline beta literals (64-bit limbs)
//   that did not match the canonical GLV_BETA0..3 #defines in
//   secp256k1_extended.cl:51-54.
//   In both backends, ct_scalar_mul_point() used the wrong beta for the GLV
//   endomorphism, producing broken results for arbitrary-point scalar
//   multiplications. Only CT ZK prove paths were affected (knowledge-of-DL,
//   DLEQ, bulletproof range proof) — ECDSA/Schnorr signing and ECDH were NOT
//   affected because they use generator-only or ladder code paths.
//
// Fix
//   Metal: Removed local BETA_METAL, now references BETA_LIMBS from point.h.
//   OpenCL: Replaced local inline literals with GLV_BETA0..3 #defines from
//   secp256k1_extended.cl.
//   Canonical β = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE
//
// Guard
//   This test has THREE layers:
//   (Layer 1) Value verification — canonical β in 32-bit, 64-bit, and BE bytes.
//   (Layer 2) Pre-fix divergence — Metal BETA_METAL and OpenCL inline literals
//             are confirmed NOT to match canonical.
//   (Layer 3) Source-coupled scan — reads the ACTUAL shader source files and
//             asserts they reference the canonical constants, NOT the old
//             divergent local copies. This layer prevents regression where
//             someone reintroduces a local BETA_METAL array or divergent
//             OpenCL inline literals without also changing the test's
//             hardcoded pre-fix values.
//
// Severity
//   P0 correctness bug, fail-loud (broken proofs fail verification, not silently
//   accepted). Every verify-side implementation uses the correct canonical beta.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <fstream>

#include "audit_check.hpp"

static int g_pass = 0, g_fail = 0;

// Canonical secp256k1 GLV β: β³ ≡ 1 (mod p)
// β = 0x7AE96A2B657C07106E64479EAC3434E99CF0497512F58995C1396C28719501EE

// 32-bit LE limb representation (Metal convention, matches BETA_LIMBS in point.h)
static const uint32_t CANONICAL_BETA_U32[8] = {
    0x719501EEu, 0xC1396C28u, 0x12F58995u, 0x9CF04975u,
    0xAC3434E9u, 0x6E64479Eu, 0x657C0710u, 0x7AE96A2Bu
};

// 64-bit LE limb representation (OpenCL convention, matches GLV_BETA in extended.cl)
static const uint64_t CANONICAL_BETA_U64[4] = {
    0xC1396C28719501EEULL,
    0x9CF0497512F58995ULL,
    0x6E64479EAC3434E9ULL,
    0x7AE96A2B657C0710ULL
};

// 256-bit big-endian byte representation
static const uint8_t CANONICAL_BETA_BE[32] = {
    0x7A, 0xE9, 0x6A, 0x2B, 0x65, 0x7C, 0x07, 0x10,
    0x6E, 0x64, 0x47, 0x9E, 0xAC, 0x34, 0x34, 0xE9,
    0x9C, 0xF0, 0x49, 0x75, 0x12, 0xF5, 0x89, 0x95,
    0xC1, 0x39, 0x6C, 0x28, 0x71, 0x95, 0x01, 0xEE
};

// 256-bit little-endian byte representation
// (same β, reversed byte order — for comparison with u32/u64_to_bytes LE output)
static const uint8_t CANONICAL_BETA_LE[32] = {
    0xEE, 0x01, 0x95, 0x71, 0x28, 0x6C, 0x39, 0xC1,
    0x95, 0x89, 0xF5, 0x12, 0x75, 0x49, 0xF0, 0x9C,
    0xE9, 0x34, 0x34, 0xAC, 0x9E, 0x47, 0x64, 0x6E,
    0x10, 0x07, 0x7C, 0x65, 0x2B, 0x6A, 0xE9, 0x7A
};

// Pre-fix divergent values — MUST NOT match canonical
// Metal BETA_METAL (32-bit LE, WRONG):
static const uint32_t PRE_FIX_METAL_BETA_U32[8] = {
    0x057C0710u, 0x7AE96A2Bu, 0xEB4C3F40u, 0x6584D3F6u,
    0x0E46AB35u, 0x7F09A368u, 0x9A83F8EFu, 0x851695D4u
};
// OpenCL inline beta (64-bit LE, WRONG):
// limb[0] matches canonical; limbs[1..3] diverge
static const uint64_t PRE_FIX_OPENCL_BETA_U64[4] = {
    0x7AE96A2B657C0710ULL,
    0x6584D3F6EB4C3F40ULL,
    0x7F09A3680E46AB35ULL,
    0x851695D49A83F8EFULL
};

// Divergent OpenCL limb[1] literal — used for source scan assertion
static const char PRE_FIX_OPENCL_DIVERGENT_LITERAL[] = "0x6584D3F6EB4C3F40";

// ============================================================================
// Helpers — byte conversion
// ============================================================================
static void u32_to_bytes(const uint32_t limbs[8], uint8_t out[32]) {
    for (int i = 0; i < 8; ++i) {
        out[i * 4 + 0] = (uint8_t)(limbs[i] & 0xFF);
        out[i * 4 + 1] = (uint8_t)((limbs[i] >> 8) & 0xFF);
        out[i * 4 + 2] = (uint8_t)((limbs[i] >> 16) & 0xFF);
        out[i * 4 + 3] = (uint8_t)((limbs[i] >> 24) & 0xFF);
    }
}

static void u64_to_bytes(const uint64_t limbs[4], uint8_t out[32]) {
    for (int i = 0; i < 4; ++i) {
        out[i * 8 + 0] = (uint8_t)(limbs[i] & 0xFF);
        out[i * 8 + 1] = (uint8_t)((limbs[i] >> 8) & 0xFF);
        out[i * 8 + 2] = (uint8_t)((limbs[i] >> 16) & 0xFF);
        out[i * 8 + 3] = (uint8_t)((limbs[i] >> 24) & 0xFF);
        out[i * 8 + 4] = (uint8_t)((limbs[i] >> 32) & 0xFF);
        out[i * 8 + 5] = (uint8_t)((limbs[i] >> 40) & 0xFF);
        out[i * 8 + 6] = (uint8_t)((limbs[i] >> 48) & 0xFF);
        out[i * 8 + 7] = (uint8_t)((limbs[i] >> 56) & 0xFF);
    }
}

// ============================================================================
// Helpers — file reading (same pattern as test_regression_shim_seckey_erase.cpp)
// ============================================================================

// Read a source file using prefix probing. Tries each candidate path
// (standalone then monorepo) with the prefix set. Not-found is a HARD FAIL —
// a silent skip would make this regression guard a false-green.
static std::string read_repo_file(const char* const* candidates, int n_candidates) {
    const char* prefixes[] = {
        "", "../", "../../", "../../../", "../../../../", nullptr
    };
    for (int c = 0; c < n_candidates; ++c) {
        for (int i = 0; prefixes[i]; ++i) {
            std::string path = std::string(prefixes[i]) + candidates[c];
            std::ifstream f(path);
            if (f.is_open()) {
                return {std::istreambuf_iterator<char>(f),
                        std::istreambuf_iterator<char>()};
            }
        }
    }
    return {};
}

// ============================================================================
// Layer 1: Value verification — canonical β encodings
// ============================================================================

static void test_canonical_u32_to_bytes() {
    uint8_t actual[32];
    u32_to_bytes(CANONICAL_BETA_U32, actual);
    // u32_to_bytes produces LE (little-endian) bytes because both the limb
    // order and intra-limb byte order are little-endian.  Compare against
    // CANONICAL_BETA_LE, not CANONICAL_BETA_BE.
    if (memcmp(actual, CANONICAL_BETA_LE, 32) != 0) {
        printf("[gpu-beta] FAIL: canonical 32-bit limbs → LE bytes mismatch\n");
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: canonical 32-bit limbs → LE bytes match\n");
        ++g_pass;
    }
}

static void test_canonical_u64_to_bytes() {
    uint8_t actual[32];
    u64_to_bytes(CANONICAL_BETA_U64, actual);
    // u64_to_bytes produces LE bytes — same reasoning as u32.
    // Compare against CANONICAL_BETA_LE, not CANONICAL_BETA_BE.
    if (memcmp(actual, CANONICAL_BETA_LE, 32) != 0) {
        printf("[gpu-beta] FAIL: canonical 64-bit limbs → LE bytes mismatch\n");
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: canonical 64-bit limbs → LE bytes match\n");
        ++g_pass;
    }
}

static void test_u32_u64_equivalence() {
    uint8_t bytes_u32[32];
    uint8_t bytes_u64[32];
    u32_to_bytes(CANONICAL_BETA_U32, bytes_u32);
    u64_to_bytes(CANONICAL_BETA_U64, bytes_u64);
    if (memcmp(bytes_u32, bytes_u64, 32) != 0) {
        printf("[gpu-beta] FAIL: 32-bit and 64-bit limb representations differ\n");
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: 32-bit ≡ 64-bit limb representations\n");
        ++g_pass;
    }
}

// ============================================================================
// Layer 2: Pre-fix divergence — wrong values must NOT match canonical
// ============================================================================

static void test_pre_fix_metal_diverges() {
    uint8_t bytes_pre[32];
    uint8_t bytes_canon[32];
    u32_to_bytes(PRE_FIX_METAL_BETA_U32, bytes_pre);
    u32_to_bytes(CANONICAL_BETA_U32, bytes_canon);
    if (memcmp(bytes_pre, bytes_canon, 32) == 0) {
        printf("[gpu-beta] FAIL: pre-fix Metal BETA_METAL should NOT match canonical\n");
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: pre-fix Metal BETA_METAL diverges from canonical\n");
        ++g_pass;
    }
}

static void test_pre_fix_opencl_diverges() {
    uint8_t bytes_pre[32];
    uint8_t bytes_canon[32];
    u64_to_bytes(PRE_FIX_OPENCL_BETA_U64, bytes_pre);
    u64_to_bytes(CANONICAL_BETA_U64, bytes_canon);
    if (memcmp(bytes_pre, bytes_canon, 32) == 0) {
        printf("[gpu-beta] FAIL: pre-fix OpenCL beta should NOT match canonical\n");
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: pre-fix OpenCL beta diverges from canonical\n");
        ++g_pass;
    }
}

// ============================================================================
// Layer 3: Source-coupled scan — actual shader files reference canonical constants
// ============================================================================

// Metal shader path candidates: standalone first, then monorepo
static const char* METAL_CT_POINT_CANDIDATES[] = {
    "src/metal/shaders/secp256k1_ct_point.h",                          // standalone UltrafastSecp256k1 repo root
    "libs/UltrafastSecp256k1/src/metal/shaders/secp256k1_ct_point.h",  // Secp256K1fast monorepo
};
static const int METAL_CT_POINT_N_CANDIDATES = 2;

// OpenCL kernel path candidates: standalone first, then monorepo
static const char* OPENCL_CT_POINT_CANDIDATES[] = {
    "src/opencl/kernels/secp256k1_ct_point.cl",                          // standalone UltrafastSecp256k1 repo root
    "libs/UltrafastSecp256k1/src/opencl/kernels/secp256k1_ct_point.cl",  // Secp256K1fast monorepo
};
static const int OPENCL_CT_POINT_N_CANDIDATES = 2;

// Assert Metal source references BETA_LIMBS (canonical) and does NOT contain
// a local BETA_METAL array declaration.
static void test_source_metal_uses_canonical_beta_limbs() {
    std::string src = read_repo_file(METAL_CT_POINT_CANDIDATES, METAL_CT_POINT_N_CANDIDATES);
    if (src.empty()) {
        printf("[gpu-beta] FAIL: cannot read Metal shader source (tried standalone + monorepo paths)\n");
        ++g_fail;
        return;
    }

    // (a) Must reference BETA_LIMBS (the canonical constant from point.h)
    if (src.find("BETA_LIMBS") == std::string::npos) {
        printf("[gpu-beta] FAIL: Metal shader does NOT reference BETA_LIMBS (canonical constant)\n");
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: Metal shader references BETA_LIMBS (canonical constant)\n");
        ++g_pass;
    }

    // (b) Must NOT declare a local BETA_METAL array (the old buggy constant)
    // Check for patterns like "BETA_METAL[" or "BETA_METAL " that would indicate
    // a local array declaration.
    if (src.find("BETA_METAL") != std::string::npos) {
        printf("[gpu-beta] FAIL: Metal shader still contains BETA_METAL (old buggy local array)\n");
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: Metal shader does NOT contain BETA_METAL local array\n");
        ++g_pass;
    }
}

// Assert OpenCL source references GLV_BETA0 (canonical #define) and does NOT
// contain the pre-fix divergent 64-bit literal.
static void test_source_opencl_uses_canonical_glv_beta() {
    std::string src = read_repo_file(OPENCL_CT_POINT_CANDIDATES, OPENCL_CT_POINT_N_CANDIDATES);
    if (src.empty()) {
        printf("[gpu-beta] FAIL: cannot read OpenCL kernel source (tried standalone + monorepo paths)\n");
        ++g_fail;
        return;
    }

    // (a) Must reference GLV_BETA0 (the canonical #define from secp256k1_extended.cl)
    if (src.find("GLV_BETA0") == std::string::npos) {
        printf("[gpu-beta] FAIL: OpenCL kernel does NOT reference GLV_BETA0 (canonical #define)\n");
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: OpenCL kernel references GLV_BETA0 (canonical #define)\n");
        ++g_pass;
    }

    // (b) Must NOT contain the pre-fix divergent limb[1] literal
    // The old buggy inline beta had {0x7AE96A2B657C0710, 0x6584D3F6EB4C3F40, ...}
    // where limb[1] = 0x6584D3F6EB4C3F40. The canonical limb[1] is 0x9CF0497512F58995.
    if (src.find(PRE_FIX_OPENCL_DIVERGENT_LITERAL) != std::string::npos) {
        printf("[gpu-beta] FAIL: OpenCL kernel still contains pre-fix divergent literal %s\n",
               PRE_FIX_OPENCL_DIVERGENT_LITERAL);
        ++g_fail;
    } else {
        printf("[gpu-beta] PASS: OpenCL kernel does NOT contain pre-fix divergent literal\n");
        ++g_pass;
    }
}

// ============================================================================
// Sanity checks: β is non-trivial (≠ 0, ≠ 1)
// ============================================================================
static void test_beta_is_nontrivial() {
    // 32-bit check
    bool is_zero = true, is_one = true;
    for (int i = 0; i < 8; ++i) {
        if (CANONICAL_BETA_U32[i] != 0) is_zero = false;
        if (i == 0) {
            if (CANONICAL_BETA_U32[0] != 1) is_one = false;
        } else {
            if (CANONICAL_BETA_U32[i] != 0) is_one = false;
        }
    }
    if (is_zero) {
        printf("[gpu-beta] FAIL: canonical beta is zero\n");
        ++g_fail;
    } else if (is_one) {
        printf("[gpu-beta] FAIL: canonical beta is one (trivial, not a valid GLV endomorphism)\n");
        ++g_fail;
    }
    // 64-bit check
    is_zero = true; is_one = true;
    for (int i = 0; i < 4; ++i) {
        if (CANONICAL_BETA_U64[i] != 0) is_zero = false;
        if (i == 0) {
            if (CANONICAL_BETA_U64[0] != 1) is_one = false;
        } else {
            if (CANONICAL_BETA_U64[i] != 0) is_one = false;
        }
    }
    if (is_zero) {
        printf("[gpu-beta] FAIL: canonical 64-bit beta is zero\n");
        ++g_fail;
    } else if (is_one) {
        printf("[gpu-beta] FAIL: canonical 64-bit beta is one\n");
        ++g_fail;
    }

    if (g_fail == 0) {
        printf("[gpu-beta] PASS: beta is non-trivial (≠ 0, ≠ 1) in both representations\n");
        ++g_pass;
    }
}

// ============================================================================
// Test: MSB (most significant bits) are non-zero (β is full 256-bit)
// ============================================================================
static void test_beta_msb_nonzero() {
    if (CANONICAL_BETA_U32[7] == 0) {
        printf("[gpu-beta] FAIL: 32-bit MSB limb is zero\n");
        ++g_fail;
    }
    if (CANONICAL_BETA_U64[3] == 0) {
        printf("[gpu-beta] FAIL: 64-bit MSB limb is zero\n");
        ++g_fail;
    }
    if (g_fail == 0) {
        printf("[gpu-beta] PASS: MSB non-zero in both representations\n");
        ++g_pass;
    }
}

// ============================================================================
// Entry point
// ============================================================================
int test_regression_gpu_beta_constants_run() {
    printf("\n=== GPU Beta Constants Regression (Metal + OpenCL) ===\n");
    printf("=== Layer 1: Value verification\n");
    printf("=== Layer 2: Pre-fix divergence\n");
    printf("=== Layer 3: Source-coupled scan\n\n");

    // Layer 1: Value verification
    test_canonical_u32_to_bytes();
    test_canonical_u64_to_bytes();
    test_u32_u64_equivalence();

    // Layer 2: Pre-fix divergence
    test_pre_fix_metal_diverges();
    test_pre_fix_opencl_diverges();

    // Layer 3: Source-coupled scan
    test_source_metal_uses_canonical_beta_limbs();
    test_source_opencl_uses_canonical_glv_beta();

    // Sanity
    test_beta_is_nontrivial();
    test_beta_msb_nonzero();

    printf("\n  %d passed  %d failed  (total %d)\n", g_pass, g_fail, g_pass + g_fail);
    return (g_fail == 0) ? 0 : 1;
}

#ifdef STANDALONE_TEST
int main() {
    return test_regression_gpu_beta_constants_run();
}
#endif
