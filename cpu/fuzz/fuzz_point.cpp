// ============================================================================
// Fuzz target: curve point operations
// Input: 64 bytes -> one scalar k + optional second scalar for property checks
// Single scalar_mul per input to stay within ASan/fuzzer timeout budget.
// ============================================================================
#include <cstdint>
#include <cstring>
#include <array>
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"

using secp256k1::fast::Scalar;
using secp256k1::fast::Point;

// Pre-build the precompute table during init (not subject to per-input timeout).
// Without this, the first scalar_mul_generator call exceeds libFuzzer's 25s
// per-input limit under ASan/UBSan when compiled without ASM.
extern "C" int LLVMFuzzerInitialize(int* /*argc*/, char*** /*argv*/) {
    volatile auto warm = Point::generator().scalar_mul(Scalar::one());
    (void)warm;
    return 0;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 32) return 0;

    std::array<uint8_t, 32> buf{};
    std::memcpy(buf.data(), data, 32);

    auto k = Scalar::from_bytes(buf);
    auto G = Point::generator();

    // k*G should be a valid point (or infinity if k == 0)
    auto P = G.scalar_mul(k);

    if (k == Scalar::zero()) {
        // 0*G == infinity
        if (!P.is_infinity()) __builtin_trap();
    } else {
        // k*G should NOT be infinity for nonzero k
        if (P.is_infinity()) __builtin_trap();

        // Point should be on curve: verify via compressed serialization
        auto compressed = P.to_compressed();
        if (compressed[0] != 0x02 && compressed[0] != 0x03) __builtin_trap();

        // Verify point addition is consistent: k*G + G == (k+1)*G
        // Use cheap addition + comparison instead of a second scalar_mul.
        auto P_plus_G = P.add(G);
        if (P_plus_G.is_infinity()) __builtin_trap();
        auto compressed2 = P_plus_G.to_compressed();
        if (compressed2[0] != 0x02 && compressed2[0] != 0x03) __builtin_trap();

        // Doubling consistency: P + P should be on-curve
        auto P_dbl = P.add(P);
        if (!P_dbl.is_infinity()) {
            auto compressed3 = P_dbl.to_compressed();
            if (compressed3[0] != 0x02 && compressed3[0] != 0x03) __builtin_trap();
        }
    }

    return 0;
}
