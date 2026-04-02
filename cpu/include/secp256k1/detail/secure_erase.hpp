#ifndef SECP256K1_DETAIL_SECURE_ERASE_HPP
#define SECP256K1_DETAIL_SECURE_ERASE_HPP

// -- Secure buffer erasure (not optimized away by the compiler) ---------------
// Defense-in-depth strategy:
//   1. Platform-specific guaranteed-not-elided erase when available
//      (memset_s / explicit_bzero / SecureZeroMemory)
//   2. Volatile function-pointer trick (libsecp256k1 pattern) as fallback
//   3. std::atomic_signal_fence as compiler barrier to prevent LTO/IPO
//      from seeing through the volatile trick

#include <cstddef>
#include <cstring>
#include <atomic>
#if defined(_MSC_VER)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>  // SecureZeroMemory
#  ifdef min
#    undef min
#  endif
#  ifdef max
#    undef max
#  endif
#endif

namespace secp256k1::detail {

inline void secure_erase(void* ptr, std::size_t len) noexcept {
    if (len == 0) return;

#if defined(_MSC_VER)
    // Windows/MSVC provides an explicit secure zero primitive.
    // It is documented as non-elidable and avoids Annex-K availability issues.
    SecureZeroMemory(ptr, len);
#elif defined(__STDC_LIB_EXT1__)
    // C11 Annex K secure erase when available.
    (void)memset_s(ptr, len, 0, len);
#elif defined(__GLIBC__) && (__GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 25))
    // glibc 2.25+: explicit_bzero is guaranteed not to be optimized away.
    explicit_bzero(ptr, len);
#elif defined(__OpenBSD__) || defined(__FreeBSD__)
    // BSD: explicit_bzero available.
    explicit_bzero(ptr, len);
#else
    // Fallback: volatile function-pointer trick (from libsecp256k1).
    // The compiler cannot prove the callee is memset, so cannot elide the call.
    void *(*volatile const volatile_memset)(void *, int, std::size_t) = std::memset;
    volatile_memset(ptr, 0, len);
#endif

    // Defense-in-depth: compiler fence prevents LTO/IPO from reasoning
    // across the barrier and optimizing away the erase above.
    // std::atomic_signal_fence has no runtime cost but prevents reordering.
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

} // namespace secp256k1::detail

#endif // SECP256K1_DETAIL_SECURE_ERASE_HPP
