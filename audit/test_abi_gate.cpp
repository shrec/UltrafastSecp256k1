// ============================================================================
// ABI Version Gate Test
// Validates ABI compatibility between runtime library and compile-time header
// ============================================================================
// This test ensures:
//   1. Compile-time ABI macros are defined and sane
//   2. Version string format is correct
//   3. Packed version matches individual components
//   4. Demonstrates what a binding MUST check at load time
//
// NOTE: This test validates compile-time macros only (no runtime lib needed).
//       The runtime functions (ufsecp_abi_version() etc.) are tested in
//       test_fuzz_address_bip32_ffi.cpp which links against ufsecp_impl.
// ============================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>

// Include version header directly
#include "ufsecp/ufsecp_version.h"

static int g_pass = 0, g_fail = 0;

#define CHECK(cond, msg) do { \
    if (!(cond)) { \
        printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
        ++g_fail; \
    } else { \
        ++g_pass; \
    } \
} while(0)

// Exportable run function (for unified audit runner)
int test_abi_gate_run() {
    g_pass = g_fail = 0;
    // Just run the checks inline:
    CHECK(UFSECP_ABI_VERSION > 0, "ABI version is positive");
    CHECK(UFSECP_ABI_VERSION == 1, "ABI version is 1 (current)");
    unsigned int const packed = UFSECP_VERSION_PACKED;
    unsigned int const major = UFSECP_VERSION_MAJOR;
    unsigned int const minor = UFSECP_VERSION_MINOR;
    unsigned int const patch = UFSECP_VERSION_PATCH;
    unsigned int const expected_packed = (major << 16) | (minor << 8) | patch;
    CHECK(packed == expected_packed, "Packed version matches components");
    CHECK(((packed >> 16) & 0xFF) == major, "Packed major matches");
    CHECK(((packed >> 8) & 0xFF) == minor, "Packed minor matches");
    CHECK((packed & 0xFF) == patch, "Packed patch matches");
    const char* ver_str = UFSECP_VERSION_STRING;
    // cppcheck-suppress nullPointerRedundantCheck
    CHECK(ver_str != nullptr, "Version string is non-null");
    // cppcheck-suppress nullPointerRedundantCheck
    CHECK(strlen(ver_str) > 0, "Version string is non-empty");
    bool has_dot = false, has_digit = false;
    // cppcheck-suppress nullPointerRedundantCheck
    for (const char* p = ver_str; *p; ++p) {
        if (*p == '.') has_dot = true;
        if (*p >= '0' && *p <= '9') has_digit = true;
    }
    CHECK(has_digit, "Version string contains digits");
    CHECK(has_dot, "Version string contains dot separator");
    CHECK(UFSECP_ABI_VERSION == 1, "ABI gate: library matches binding");
    CHECK(UFSECP_ABI_VERSION == 1, "ABI gate: exact version match");
    printf("  [abi_gate] %d passed, %d failed\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}

#ifndef UNIFIED_AUDIT_RUNNER
int main() {
    printf("============================================================\n");
    printf("  ABI Version Gate Test (compile-time)\n");
    printf("============================================================\n\n");

    // 1. ABI version macro must be defined and positive
    printf("  UFSECP_ABI_VERSION:       %u\n", (unsigned)UFSECP_ABI_VERSION);
    CHECK(UFSECP_ABI_VERSION > 0, "ABI version is positive");
    CHECK(UFSECP_ABI_VERSION == 1, "ABI version is 1 (current)");

    // 2. Packed version
    unsigned int const packed = UFSECP_VERSION_PACKED;
    printf("  UFSECP_VERSION_PACKED:    0x%06X\n", packed);

    unsigned int const major = UFSECP_VERSION_MAJOR;
    unsigned int const minor = UFSECP_VERSION_MINOR;
    unsigned int const patch = UFSECP_VERSION_PATCH;
    printf("  Version: %u.%u.%u\n", major, minor, patch);

    // 3. Packed encoding consistency
    unsigned int const expected_packed = (major << 16) | (minor << 8) | patch;
    CHECK(packed == expected_packed, "Packed version matches components");

    // 4. Components extracted from packed match originals
    CHECK(((packed >> 16) & 0xFF) == major, "Packed major matches");
    CHECK(((packed >> 8) & 0xFF) == minor, "Packed minor matches");
    CHECK((packed & 0xFF) == patch, "Packed patch matches");

    // 5. Version string
    const char* ver_str = UFSECP_VERSION_STRING;
    printf("  UFSECP_VERSION_STRING:    \"%s\"\n", ver_str);
    // cppcheck-suppress nullPointerRedundantCheck
    CHECK(ver_str != nullptr, "Version string is non-null");
    // cppcheck-suppress nullPointerRedundantCheck
    CHECK(strlen(ver_str) > 0, "Version string is non-empty");

    // 6. Validate that version string contains digits and dots
    {
        bool has_dot = false;
        bool has_digit = false;
        // cppcheck-suppress nullPointerRedundantCheck
        for (const char* p = ver_str; *p; ++p) {
            if (*p == '.') has_dot = true;
            if (*p >= '0' && *p <= '9') has_digit = true;
        }
        CHECK(has_digit, "Version string contains digits");
        CHECK(has_dot, "Version string contains dot separator");
    }

    // 7. ABI gate simulation: what a binding SHOULD check
    printf("\n  [Binding Gate Simulation]\n");
    unsigned int const binding_expected_abi = 1;  // hardcoded in binding
    if (UFSECP_ABI_VERSION != binding_expected_abi) {
        printf("  *** ABI MISMATCH: library=%u, binding expects=%u ***\n",
               (unsigned)UFSECP_ABI_VERSION, binding_expected_abi);
        printf("  A real binding MUST refuse to load on mismatch.\n");
        CHECK(false, "ABI gate: version mismatch (would refuse load)");
    } else {
        printf("  ABI version matches binding expectation (%u). LOAD OK.\n",
               binding_expected_abi);
        CHECK(true, "ABI gate: version matches");
    }

    // 8. Backward compatibility: packed encoding fits in valid 24-bit range
    // Use volatile load to prevent CodeQL cpp/unsigned-comparison-zero
    // when all version components are zero at compile time.
    volatile unsigned int packed_rt = packed;
    CHECK(packed_rt <= 0x00FFFFFFu, "Packed version within valid 24-bit range");

    printf("\n============================================================\n");
    printf("  Summary: %d passed, %d failed\n", g_pass, g_fail);
    printf("============================================================\n");

    return g_fail > 0 ? 1 : 0;
}
#endif // UNIFIED_AUDIT_RUNNER
