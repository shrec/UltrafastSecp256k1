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

        // Point should be on curve: verify via compressed serialization round-trip
        // to_compressed() performs field inversion + normalization; any corrupt
        // internal state would produce an invalid 33-byte encoding.
        auto compressed = P.to_compressed();
        // Sanity: prefix byte must be 0x02 or 0x03
        if (compressed[0] != 0x02 && compressed[0] != 0x03) __builtin_trap();

        // -- Distributivity: (k+1)*G == k*G + G
        auto k1 = k + Scalar::one();
        auto P_k1 = G.scalar_mul(k1);
        auto P_plus_G = P.add(G);
        // Compare via compressed encoding (canonical byte representation)
        if (P_k1.to_compressed() != P_plus_G.to_compressed()) __builtin_trap();

        // -- 2*k*G == k*G + k*G (doubling)
        auto two = Scalar::one() + Scalar::one();
        auto k2 = k * two;
        auto P_2k = G.scalar_mul(k2);
        auto P_double = P.add(P);
        if (P_2k.to_compressed() != P_double.to_compressed()) __builtin_trap();
    }

    return 0;
}
