#pragma once

#include <cstdint>

namespace secp256k1::detail {

#if defined(_MSC_VER) && !defined(__clang__)

inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    unsigned __int64 out;
    carry = _addcarry_u64(carry, a, b, &out);
    return out;
}

inline std::uint64_t sub64(std::uint64_t a, std::uint64_t b, unsigned char& borrow) {
    unsigned __int64 out;
    borrow = _subborrow_u64(borrow, a, b, &out);
    return out;
}

#else

#ifdef SECP256K1_NO_INT128

inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    std::uint64_t result = a + b;
    unsigned char new_carry = (result < a) ? 1 : 0;
    if (carry) {
        std::uint64_t temp = result + 1;
        new_carry |= (temp < result) ? 1 : 0;
        result = temp;
    }
    carry = new_carry;
    return result;
}

inline std::uint64_t sub64(std::uint64_t a, std::uint64_t b, unsigned char& borrow) {
    std::uint64_t temp = a - borrow;
    unsigned char const borrow1 = (a < borrow);
    std::uint64_t result = temp - b;
    unsigned char const borrow2 = (temp < b);
    borrow = borrow1 | borrow2;
    return result;
}

#else

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

inline std::uint64_t add64(std::uint64_t a, std::uint64_t b, unsigned char& carry) {
    unsigned __int128 const sum = static_cast<unsigned __int128>(a) + b + carry;
    carry = static_cast<unsigned char>(sum >> 64);
    return static_cast<std::uint64_t>(sum);
}

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

inline std::uint64_t sub64(std::uint64_t a, std::uint64_t b, unsigned char& borrow) {
    std::uint64_t const temp = a - borrow;
    unsigned char const borrow1 = (a < borrow);
    std::uint64_t const result = temp - b;
    unsigned char const borrow2 = (temp < b);
    borrow = borrow1 | borrow2;
    return result;
}

#endif // SECP256K1_NO_INT128

#endif // _MSC_VER

} // namespace secp256k1::detail
