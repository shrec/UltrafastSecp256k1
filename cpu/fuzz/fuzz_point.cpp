// ============================================================================
// Fuzz target: curve point operations
// Input: 32 bytes -> one scalar k, compute k*G and verify on-curve
// ============================================================================
#include <cstdint>
#include <cstring>
#include <array>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) return 0;

    std::array<uint8_t, 32> buf{};
    std::memcpy(buf.data(), data, 32);

    auto k = Scalar::from_bytes(buf);
    auto zero = Scalar::zero();
    auto G = Point::generator();

    // k*G should be a valid point (or infinity if k == 0)
    auto P = G.scalar_mul(k);

    if (k == zero) {
        // 0*G == infinity
        if (!P.is_infinity()) __builtin_trap();
    } else {
        // k*G should NOT be infinity for nonzero k
        if (P.is_infinity()) __builtin_trap();

        // Point should be on curve: y^2 == x^3 + 7
        // Verify via serialize/deserialize round-trip (parse validates on-curve)
        auto compressed = P.serialize_compressed();
        auto P2 = Point::parse_compressed(compressed.data());
        if (!P2.has_value()) __builtin_trap();

        // -- Distributivity: (k+1)*G == k*G + G
        auto k1 = k + Scalar::one();
        auto P_k1 = G.scalar_mul(k1);
        auto P_plus_G = P.add(G);
        if (!(P_k1 == P_plus_G)) __builtin_trap();

        // -- 2*k*G == k*G + k*G (doubling)
        auto two = Scalar::one() + Scalar::one();
        auto k2 = k * two;
        auto P_2k = G.scalar_mul(k2);
        auto P_double = P.add(P);
        if (!(P_2k == P_double)) __builtin_trap();
    }

    return 0;
}
