// sanitizer_scale.hpp -- reduce test iteration counts under sanitizers
// Sanitizers (ASan, TSan, UBSan) add 3-15x runtime overhead.
// This header provides a compile-time flag and helper to scale down
// iteration counts so tests finish within CI timeouts.
//
// Usage:
//   #include "secp256k1/sanitizer_scale.hpp"
//   const int N = SCALED(10000, 200);  // 10000 normal, 200 under sanitizer

#ifndef SECP256K1_SANITIZER_SCALE_HPP
#define SECP256K1_SANITIZER_SCALE_HPP

// Detect sanitizer builds (Clang, GCC, MSVC)
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#  define SECP256K1_SANITIZER_BUILD 1
#elif defined(__has_feature)
#  if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer) || \
      __has_feature(memory_sanitizer)  || __has_feature(undefined_behavior_sanitizer)
#    define SECP256K1_SANITIZER_BUILD 1
#  endif
#endif

#ifndef SECP256K1_SANITIZER_BUILD
#  define SECP256K1_SANITIZER_BUILD 0
#endif

// SCALED(normal, reduced) -- pick count based on build type
#define SCALED(normal, reduced) (SECP256K1_SANITIZER_BUILD ? (reduced) : (normal))

#endif // SECP256K1_SANITIZER_SCALE_HPP
