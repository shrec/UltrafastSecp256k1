#ifndef SECP256K1_DETAIL_SECURE_ERASE_HPP
#define SECP256K1_DETAIL_SECURE_ERASE_HPP

// -- Secure buffer erasure (not optimized away by the compiler) ---------------
// Uses the volatile function-pointer trick from libsecp256k1: the compiler
// cannot prove the callee is memset, so it is not allowed to elide the call.

#include <cstddef>
#include <cstring>

namespace secp256k1::detail {

inline void secure_erase(void* ptr, std::size_t len) noexcept {
    void *(*volatile const volatile_memset)(void *, int, std::size_t) = std::memset;
    volatile_memset(ptr, 0, len);
}

} // namespace secp256k1::detail

#endif // SECP256K1_DETAIL_SECURE_ERASE_HPP
