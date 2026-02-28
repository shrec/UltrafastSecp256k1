#ifndef C52FED12_BFF1_451A_9296_FAE694E6A33E
#define C52FED12_BFF1_451A_9296_FAE694E6A33E
#ifndef SECP256K1_CONFIG_HPP_INCLUDED
#define SECP256K1_CONFIG_HPP_INCLUDED

// ============================================================================
// Platform Detection
// ============================================================================

// ESP32 platform detection
#if defined(ESP_PLATFORM) || defined(SECP256K1_PLATFORM_ESP32)
    #define SECP256K1_ESP32 1
    #define SECP256K1_32BIT 1
    #define SECP256K1_NO_INT128 1
    #ifndef SECP256K1_NO_ASM
        #define SECP256K1_NO_ASM 1  // For now, disable x86 asm on ESP32
    #endif
#endif

// 32-bit platform detection
#if defined(__i386__) || defined(_M_IX86) || defined(__arm__) || defined(__xtensa__) || defined(SECP256K1_32BIT)
    #ifndef SECP256K1_32BIT
        #define SECP256K1_32BIT 1
    #endif
#endif

// Disable __int128 on 32-bit platforms or when explicitly requested
#if defined(SECP256K1_32BIT) || defined(SECP256K1_NO_INT128)
    #ifndef SECP256K1_NO_INT128
        #define SECP256K1_NO_INT128 1
    #endif
#endif

// Performance optimization macros for hot path functions

// Force inline for critical performance paths
// MSVC: Using 'inline' instead of '__forceinline' to avoid linker issues with static libraries
// GCC/Clang: Can safely use __attribute__((always_inline)) which works with exported symbols
#if defined(__GNUC__) || defined(__clang__)
    #define SECP256K1_INLINE __attribute__((always_inline)) inline
#else
    #define SECP256K1_INLINE inline
#endif

// Branch prediction hints (for unlikely conditions like infinity checks)
// NOTE: MSVC [[likely]]/[[unlikely]] syntax incompatible with __builtin_expect wrapper
// MSVC requires: if (x) [[unlikely]] { } vs GCC: if (UNLIKELY(x)) { }
// For cross-platform compatibility, we only use builtin_expect for GCC/Clang
#if defined(__GNUC__) || defined(__clang__)
    #define SECP256K1_LIKELY(x)   __builtin_expect(!!(x), 1)
    #define SECP256K1_UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
    // MSVC: No branch hints (attributes have different syntax)
    #define SECP256K1_LIKELY(x)   (x)
    #define SECP256K1_UNLIKELY(x) (x)
#endif

// Prevent inlining: critical for large-stack-frame functions where MSVC's
// GS-cookie (stack canary) can be corrupted when the compiler inlines ~5KB+
// of local arrays into a caller's frame. GCC/Clang use __attribute__; MSVC
// requires __declspec(noinline) (it silently ignores __attribute__).
#if defined(_MSC_VER)
    #define SECP256K1_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
    #define SECP256K1_NOINLINE __attribute__((noinline))
#else
    #define SECP256K1_NOINLINE
#endif

// Restrict pointer aliasing optimization
#if defined(_MSC_VER)
    #define SECP256K1_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define SECP256K1_RESTRICT __restrict__
#else
    #define SECP256K1_RESTRICT
#endif

// GCC-specific aggressive optimization attributes for hot functions
// These force maximum compiler optimization even beyond -O3
#if defined(__GNUC__) && !defined(__clang__)
    // GCC-specific: Target specific CPU features and optimizations
    #define SECP256K1_HOT_FUNCTION \
        __attribute__((hot)) \
        __attribute__((optimize("O3,unroll-loops,inline-functions,tree-vectorize")))
    
    // For critical arithmetic functions (field operations, point operations)
    #define SECP256K1_CRITICAL_FUNCTION \
        __attribute__((hot)) \
        __attribute__((optimize("O3,unroll-loops,inline-functions,tree-vectorize,ipa-pta"))) \
        __attribute__((flatten))
    
    // Pure functions (no side effects, result depends only on args)
    #define SECP256K1_PURE __attribute__((pure))
    #define SECP256K1_CONST __attribute__((const))
    
#elif defined(__clang__)
    // Clang doesn't support function-level optimization attributes
    #define SECP256K1_HOT_FUNCTION __attribute__((hot))
    #define SECP256K1_CRITICAL_FUNCTION __attribute__((hot)) __attribute__((flatten))
    #define SECP256K1_PURE __attribute__((pure))
    #define SECP256K1_CONST __attribute__((const))
#else
    #define SECP256K1_HOT_FUNCTION
    #define SECP256K1_CRITICAL_FUNCTION
    #define SECP256K1_PURE
    #define SECP256K1_CONST
#endif

// Disable stack protector (GS-cookie / stack canary) on a per-function basis.
// Required for large-stack-frame hot functions (~5 KB+ of local arrays)
// where try/catch was previously used as a workaround to force SEH frame
// emission.  This attribute is the proper fix: no try/catch overhead,
// no SEH unwind tables, no GS-cookie check on each return.
#if defined(_MSC_VER)
    #define SECP256K1_NO_STACK_PROTECTOR __declspec(safebuffers)
#elif defined(__clang__) || defined(__GNUC__)
    #define SECP256K1_NO_STACK_PROTECTOR __attribute__((no_stack_protector))
#else
    #define SECP256K1_NO_STACK_PROTECTOR
#endif

// Prefetch macros for cache optimization (Phase 4)
// Hints to CPU to load data into cache before it's needed
#if defined(_MSC_VER)
    #include <xmmintrin.h>  // For _mm_prefetch
    // _MM_HINT_T0: Prefetch to all cache levels (L1, L2, L3)
    // _MM_HINT_T1: Prefetch to L2 and L3 cache (not L1)
    // _MM_HINT_T2: Prefetch to L3 cache only
    // _MM_HINT_NTA: Non-temporal prefetch (bypass cache)
    #define SECP256K1_PREFETCH_READ(addr)  _mm_prefetch((const char*)(addr), _MM_HINT_T0)
    #define SECP256K1_PREFETCH_WRITE(addr) _mm_prefetch((const char*)(addr), _MM_HINT_T0)
#elif defined(__GNUC__) || defined(__clang__)
    // __builtin_prefetch(addr, rw, locality)
    // rw: 0 = read, 1 = write
    // locality: 0 = no temporal locality, 3 = high temporal locality
    #define SECP256K1_PREFETCH_READ(addr)  __builtin_prefetch((addr), 0, 3)
    #define SECP256K1_PREFETCH_WRITE(addr) __builtin_prefetch((addr), 1, 3)
#else
    // No prefetch support
    #define SECP256K1_PREFETCH_READ(addr)  ((void)0)
    #define SECP256K1_PREFETCH_WRITE(addr) ((void)0)
#endif

#endif // SECP256K1_CONFIG_HPP_INCLUDED


#endif /* C52FED12_BFF1_451A_9296_FAE694E6A33E */
