// ============================================================================
// Fuzz target: scalar ring invariants
// Input: 64 bytes -> two 32-byte scalars (a, b)
// ============================================================================
#include <cstdint>
#include <cstring>
#include <array>
#include "secp256k1/scalar.hpp"

using secp256k1::fast::Scalar;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 64) return 0;

    std::array<uint8_t, 32> buf_a{}, buf_b{};
    std::memcpy(buf_a.data(), data, 32);
    std::memcpy(buf_b.data(), data + 32, 32);

    auto a = Scalar::from_bytes(buf_a);
    auto b = Scalar::from_bytes(buf_b);
    auto zero = Scalar::zero();

    // -- Commutativity: a + b == b + a
    auto sum1 = a + b;
    auto sum2 = b + a;
    if (!(sum1 == sum2)) __builtin_trap();

    // -- Commutativity: a * b == b * a
    auto prod1 = a * b;
    auto prod2 = b * a;
    if (!(prod1 == prod2)) __builtin_trap();

    // -- Associativity: (a + b) - b == a
    auto diff = sum1 - b;
    if (!(diff == a)) __builtin_trap();

    // -- Negation: a + (-a) == 0
    auto neg = a.negate();
    auto should_be_zero = a + neg;
    if (!(should_be_zero == zero)) __builtin_trap();

    // -- Inverse: a * a^-1 == 1 (if a != 0)
    if (!(a == zero)) {
        auto inv = a.inverse();
        auto should_be_one = a * inv;
        auto one = Scalar::one();
        if (!(should_be_one == one)) __builtin_trap();
    }

    // -- Serialization round-trip
    auto bytes_out = a.to_bytes();
    auto a_back = Scalar::from_bytes(bytes_out);
    if (!(a_back == a)) __builtin_trap();

    return 0;
}
