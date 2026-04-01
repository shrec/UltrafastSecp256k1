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
#  include <wchar.h>  // pulls in memset_s via MSVC CRT
#endif

namespace secp256k1::detail {

inline void secure_erase(void* ptr, std::size_t len) noexcept {
    if (len == 0) return;

#if defined(_MSC_VER)
    // MSVC /GL + LTCG can eliminate a volatile write loop because LTCG reasons
    // across TUs. memset_s (C11 §K.3.7.4.1) is explicitly forbidden from being
    // optimized away by conforming implementations; MSVC honours this.
    // A volatile loop fallback guards against a hypothetical non-conforming build.
    if (memset_s(ptr, len, 0, len) != 0) {
        volatile unsigned char* vp = static_cast<volatile unsigned char*>(ptr);
        for (std::size_t i = 0; i < len; ++i) { vp[i] = 0; }
    }
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
