// ============================================================================
// shim_seckey.cpp -- Secret key verification and tweaking
// ============================================================================
#include "secp256k1.h"
#include "shim_internal.hpp"

#include <cstring>
#include <array>

#include "secp256k1/scalar.hpp"
#include "secp256k1/ct/scalar.hpp"

using namespace secp256k1::fast;

extern "C" {

int secp256k1_ec_seckey_verify(
    const secp256k1_context *ctx, const unsigned char *seckey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!seckey) return 0;
    Scalar k;
    return Scalar::parse_bytes_strict_nonzero(seckey, k) ? 1 : 0;
}

int secp256k1_ec_seckey_negate(
    const secp256k1_context *ctx, unsigned char *seckey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!seckey) return 0;
    Scalar k;
    if (!Scalar::parse_bytes_strict_nonzero(seckey, k)) return 0;
    auto neg = k.negate();
    auto out = neg.to_bytes();
    std::memcpy(seckey, out.data(), 32);
    return 1;
}

int secp256k1_ec_seckey_tweak_add(
    const secp256k1_context *ctx, unsigned char *seckey,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!seckey || !tweak32) return 0;
    Scalar k, t;
    if (!Scalar::parse_bytes_strict_nonzero(seckey, k)) return 0;
    // tweak in [0, n-1]; 0 is valid (result == seckey)
    if (!Scalar::parse_bytes_strict(tweak32, t)) return 0;
    auto result = secp256k1::ct::scalar_add(k, t);  // CT-001: k is secret
    if (result.is_zero_ct()) return 0;
    auto out = result.to_bytes();
    std::memcpy(seckey, out.data(), 32);
    return 1;
}

int secp256k1_ec_seckey_tweak_mul(
    const secp256k1_context *ctx, unsigned char *seckey,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!seckey || !tweak32) return 0;
    Scalar k, t;
    if (!Scalar::parse_bytes_strict_nonzero(seckey, k)) return 0;
    if (!Scalar::parse_bytes_strict_nonzero(tweak32, t)) return 0;
    auto result = secp256k1::ct::scalar_mul(k, t);  // CT-001: k is secret
    if (result.is_zero_ct()) return 0;
    auto out = result.to_bytes();
    std::memcpy(seckey, out.data(), 32);
    return 1;
}

} // extern "C"
