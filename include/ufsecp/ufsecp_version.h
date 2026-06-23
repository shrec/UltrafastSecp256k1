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
#define UFSECP_VERSION_MINOR   4
#define UFSECP_VERSION_PATCH   0

/** Packed: (major << 16) | (minor << 8) | patch.  Compare with >= for compat.
 *  Casts to unsigned to avoid signed-integer-overflow UB (C11 6.5p5). */
#define UFSECP_VERSION_PACKED \
    (((unsigned int)UFSECP_VERSION_MAJOR << 16) | \
     ((unsigned int)UFSECP_VERSION_MINOR <<  8) | \
      (unsigned int)UFSECP_VERSION_PATCH)

#define UFSECP_VERSION_STRING  "4.4.0"

/* -- ABI version (incremented ONLY on binary-incompatible changes) ---------- */

#define UFSECP_ABI_VERSION     4

/* -- Runtime queries -------------------------------------------------------- */

#ifndef UFSECP_API
  #if defined(_WIN32) || defined(__CYGWIN__)
    /* A static-lib / single-TU build defines BOTH UFSECP_STATIC_LIB and
     * UFSECP_BUILDING (the latter to compile the engine sources). STATIC_LIB must
     * win so the static build emits NO dllexport: a static archive needs none, and
     * clang-cl rejects a `static thread_local` inside a dllexport'd function
     * (MSVC cl tolerates it), which broke the libbitcoin/static + bench builds. */
    #ifdef UFSECP_STATIC_LIB
      #define UFSECP_API
    #elif defined(UFSECP_BUILDING)
      #define UFSECP_API __declspec(dllexport)
    #else
      #define UFSECP_API __declspec(dllimport)
    #endif
  #elif __GNUC__ >= 4
    #define UFSECP_API __attribute__((visibility("default")))
  #else
    #define UFSECP_API
  #endif
#endif

/* -- Deprecation annotation ------------------------------------------------- */
/* UFSECP_DEPRECATED("msg") marks a function as deprecated with a diagnostic
 * message.  Use it on the declaration in ufsecp.h to guide callers to the
 * replacement API without breaking the ABI.
 *
 * Example (pseudo-code, not a real declaration):
 *   UFSECP_DEPRECATED("Use <new_fn>()") UFSECP_API ufsecp_error_t <old_fn>(...);
 */
#ifndef UFSECP_DEPRECATED
  #if defined(__cplusplus) && (__cplusplus >= 201402L)
    /* C++14 and later: standard [[deprecated("msg")]] attribute */
    #define UFSECP_DEPRECATED(msg) [[deprecated(msg)]]
  #elif defined(__GNUC__) || defined(__clang__)
    /* GCC / Clang C: __attribute__((deprecated("msg"))) */
    #define UFSECP_DEPRECATED(msg) __attribute__((deprecated(msg)))
  #elif defined(_MSC_VER)
    /* MSVC: __declspec(deprecated("msg")) */
    #define UFSECP_DEPRECATED(msg) __declspec(deprecated(msg))
  #else
    #define UFSECP_DEPRECATED(msg)
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
