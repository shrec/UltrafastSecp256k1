// GLV endomorphism optimization for secp256k1
// phi(x,y) = (beta*x, y) where beta^3 == 1 (mod p)
// lambda*P = phi(P) where lambda^2 + lambda + 1 == 0 (mod n)

#pragma once

#include "scalar.hpp"
#include "point.hpp"
#include <array>
#include <cstdint>

namespace secp256k1::fast {

// GLV constants for secp256k1
namespace glv_constants {
    // lambda -- endomorphism eigenvalue: lambda * G equals phi(G)
    // lambda = 0x5363ad4cc05c30e0a5261c028812645a122e22ea20816678df02967c1b23bd72
    constexpr std::array<uint8_t, 32> LAMBDA = {
        0x53, 0x63, 0xad, 0x4c, 0xc0, 0x5c, 0x30, 0xe0,
        0xa5, 0x26, 0x1c, 0x02, 0x88, 0x12, 0x64, 0x5a,
        0x12, 0x2e, 0x22, 0xea, 0x20, 0x81, 0x66, 0x78,
        0xdf, 0x02, 0x96, 0x7c, 0x1b, 0x23, 0xbd, 0x72
    };
    
    // beta (beta) - cube root of unity mod p
    // beta = 0x7ae96a2b657c07106e64479eac3434e99cf0497512f58995c1396c28719501ee
    constexpr std::array<uint8_t, 32> BETA = {
        0x7a, 0xe9, 0x6a, 0x2b, 0x65, 0x7c, 0x07, 0x10,
        0x6e, 0x64, 0x47, 0x9e, 0xac, 0x34, 0x34, 0xe9,
        0x9c, 0xf0, 0x49, 0x75, 0x12, 0xf5, 0x89, 0x95,
        0xc1, 0x39, 0x6c, 0x28, 0x71, 0x95, 0x01, 0xee
    };
    
    // Precomputed values for GLV decomposition
    // These are used to split k -> (k1, k2) such that k = k1 + k2*lambda (mod n)
    // Using the lattice basis vectors from secp256k1 optimization
    
    // a1 = 0x3086d221a7d46bcde86c90e49284eb15
    constexpr std::array<uint8_t, 16> A1 = {
        0x30, 0x86, 0xd2, 0x21, 0xa7, 0xd4, 0x6b, 0xcd,
        0xe8, 0x6c, 0x90, 0xe4, 0x92, 0x84, 0xeb, 0x15
    };
    
    // -b1 = 0xe4437ed6010e88286f547fa90abfe4c3
    constexpr std::array<uint8_t, 16> MINUS_B1 = {
        0xe4, 0x43, 0x7e, 0xd6, 0x01, 0x0e, 0x88, 0x28,
        0x6f, 0x54, 0x7f, 0xa9, 0x0a, 0xbf, 0xe4, 0xc3
    };
    
    // a2 = 0xe4437ed6010e88286f547fa90abfe4c4
    constexpr std::array<uint8_t, 16> A2 = {
        0xe4, 0x43, 0x7e, 0xd6, 0x01, 0x0e, 0x88, 0x28,
        0x6f, 0x54, 0x7f, 0xa9, 0x0a, 0xbf, 0xe4, 0xc4
    };
    
    // b2 = 0x3086d221a7d46bcde86c90e49284eb15
    constexpr std::array<uint8_t, 16> B2 = {
        0x30, 0x86, 0xd2, 0x21, 0xa7, 0xd4, 0x6b, 0xcd,
        0xe8, 0x6c, 0x90, 0xe4, 0x92, 0x84, 0xeb, 0x15
    };
}

// GLV decomposition result
struct GLVDecomposition {
    Scalar k1;
    Scalar k2;
    bool k1_neg;  // true if k1 should be negated
    bool k2_neg;  // true if k2 should be negated
};

// Decompose scalar k into k1, k2 such that k = k1 + k2*lambda (mod n)
// The resulting k1, k2 are roughly half the bit length of k (~128 bits each)
GLVDecomposition glv_decompose(const Scalar& k);

// Apply endomorphism to point: phi(x,y) = (beta*x, y)
// This is very cheap - just one field multiplication
Point apply_endomorphism(const Point& P);

// Verify endomorphism properties (for testing)
// Should verify: phi(phi(P)) + P = O (point at infinity)
bool verify_endomorphism(const Point& P);

} // namespace secp256k1::fast
