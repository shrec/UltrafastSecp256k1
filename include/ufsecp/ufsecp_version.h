/* ============================================================================
 * UltrafastSecp256k1 -- Version & ABI Compatibility
 * ============================================================================
 * NOTE: When building with CMake the real header is generated from
 *       ufsecp_version.h.in and placed in the build tree (takes priority).
 *       This source-tree copy contains dev placeholders for non-CMake use.
 * ============================================================================
 * RULES:
 *   UFSECP_VERSION_MAJOR bump  ->  ABI breaking (struct layout, removed funcs)
 *   UFSECP_VERSION_MINOR bump  ->  ABI compatible (new funcs only)
 *   UFSECP_VERSION_PATCH bump  ->  ABI compatible (bugfixes only)
 *   UFSECP_ABI_VERSION   bump  ->  only on ABI-incompatible changes
 *
 * Clients should check:  ufsecp_abi_version() == expected_abi
 * ============================================================================ */

#ifndef UFSECP_VERSION_H
#define UFSECP_VERSION_H

#ifdef __cplusplus
extern "C" {
#endif

/* -- Compile-time version.
 * NOTE: when building with CMake the generated copy in ${CMAKE_BINARY_DIR}/include
 * SHOULD override this, but because ufsecp.h includes "ufsecp_version.h" with a
 * relative path, the compiler finds this co-located copy first.  Both files must
 * stay in sync. Keep this file updated to match ufsecp_version.h.in. */

#define UFSECP_VERSION_MAJOR   4
#define UFSECP_VERSION_MINOR   0
#define UFSECP_VERSION_PATCH   0

/** Packed: (major << 16) | (minor << 8) | patch.  Compare with >= for compat.
 *  Casts to unsigned to avoid signed-integer-overflow UB (C11 6.5p5). */
#define UFSECP_VERSION_PACKED \
    (((unsigned int)UFSECP_VERSION_MAJOR << 16) | \
     ((unsigned int)UFSECP_VERSION_MINOR <<  8) | \
      (unsigned int)UFSECP_VERSION_PATCH)

#define UFSECP_VERSION_STRING  "4.0.0"

/* -- ABI version (incremented ONLY on binary-incompatible changes) ---------- */

#define UFSECP_ABI_VERSION     1

/* -- Runtime queries -------------------------------------------------------- */

#ifndef UFSECP_API
  #if defined(_WIN32) || defined(__CYGWIN__)
    #ifdef UFSECP_BUILDING
      #define UFSECP_API __declspec(dllexport)
    #elif defined(UFSECP_STATIC_LIB)
      #define UFSECP_API
    #else
      #define UFSECP_API __declspec(dllimport)
    #endif
  #elif __GNUC__ >= 4
    #define UFSECP_API __attribute__((visibility("default")))
  #else
    #define UFSECP_API
  #endif
#endif

/** Return packed version at runtime (same as UFSECP_VERSION_PACKED). */
UFSECP_API unsigned int ufsecp_version(void);

/** Return ABI version at runtime (same as UFSECP_ABI_VERSION). */
UFSECP_API unsigned int ufsecp_abi_version(void);

/** Return human-readable version string, e.g. "4.0.0". */
UFSECP_API const char* ufsecp_version_string(void);

#ifdef __cplusplus
}
#endif

#endif /* UFSECP_VERSION_H */
