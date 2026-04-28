// ============================================================================
// shim_ecdh.cpp -- ECDH key exchange (secp256k1_ecdh)
// ============================================================================
// Implements the libsecp256k1 ECDH module API using the fast engine.
// ECDH is a public-side operation: the private key scalar multiplication
// uses Point::scalar_mul (GLV, ~17 µs) rather than CT (~40 µs).
// ============================================================================

#include "secp256k1_ecdh.h"
#include "secp256k1.h"

#include <cstring>
#include <array>

#include "secp256k1/ecdh.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/sha256.hpp"

using namespace secp256k1::fast;

// -- Default hash: SHA-256(compressed_point) -----------------------------------
// Matches libsecp256k1's secp256k1_ecdh_hashfp_sha256 behaviour exactly.

static int default_hashfp(unsigned char* output,
    const unsigned char* x32, const unsigned char* y32, void* /*data*/)
{
    // prefix: 02 for even y, 03 for odd y
    uint8_t prefix = ((y32[31] & 1) ? 0x03 : 0x02);
    uint8_t compressed[33];
    compressed[0] = prefix;
    std::memcpy(compressed + 1, x32, 32);
    auto h = secp256k1::SHA256::hash(compressed, 33);
    std::memcpy(output, h.data(), 32);
    return 1;
}

extern "C" {

const secp256k1_ecdh_hashfp secp256k1_ecdh_hashfp_sha256 = default_hashfp;

int secp256k1_ecdh(
    const secp256k1_context* ctx,
    unsigned char* output,
    const secp256k1_pubkey* pubkey,
    const unsigned char* seckey,
    secp256k1_ecdh_hashfp hashfp,
    void* data)
{
    (void)ctx;
    if (!output || !pubkey || !seckey) return 0;

    if (!hashfp) hashfp = default_hashfp;

    // Deserialize private key
    std::array<uint8_t, 32> kb{};
    std::memcpy(kb.data(), seckey, 32);
    auto sk = Scalar::from_bytes(kb);
    if (sk.is_zero()) return 0;

    // Deserialize public key (shim layout: X || Y, 64 bytes)
    std::array<uint8_t, 32> xb{}, yb{};
    std::memcpy(xb.data(), pubkey->data,      32);
    std::memcpy(yb.data(), pubkey->data + 32, 32);
    auto x = FieldElement::from_bytes(xb);
    auto y = FieldElement::from_bytes(yb);
    auto pk = Point::from_affine(x, y);
    if (pk.is_infinity()) return 0;

    // ECDH: result = sk * PK  (fast GLV)
    auto result = pk.scalar_mul(sk);
    if (result.is_infinity()) return 0;

    // Extract affine X, Y
    auto unc = result.to_uncompressed(); // 04 || X[32] || Y[32]
    return hashfp(output, unc.data() + 1, unc.data() + 33, data);
}

} // extern "C"
