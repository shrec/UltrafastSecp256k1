// ============================================================================
// shim_ellswift.cpp -- BIP-324 ElligatorSwift (secp256k1_ellswift_* API)
// ============================================================================
#include "secp256k1_ellswift.h"
#include "secp256k1.h"
#include "shim_internal.hpp"

#include <cstring>
#include <array>

#include "secp256k1/ellswift.hpp"
#include "secp256k1/bip324.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"

using namespace secp256k1::fast;

// -- BIP-324 tagged hash (matches secp256k1_ellswift_xdh_hash_function_bip324) -

static int bip324_hash_impl(unsigned char* output,
    const unsigned char* x32,
    const unsigned char* ell_a64,
    const unsigned char* ell_b64,
    void* /*data*/)
{
    constexpr char tag[] = "bip324_ellswift_xonly_ecdh";
    auto tag_hash = secp256k1::SHA256::hash(tag, sizeof(tag) - 1);

    secp256k1::SHA256 h;
    h.update(tag_hash.data(), 32);
    h.update(tag_hash.data(), 32);
    h.update(ell_a64, 64);
    h.update(ell_b64, 64);
    h.update(x32, 32);
    auto result = h.finalize();
    std::memcpy(output, result.data(), 32);
    return 1;
}

static int prefix_hash_impl(unsigned char* output,
    const unsigned char* x32,
    const unsigned char* ell_a64,
    const unsigned char* ell_b64,
    void* data)
{
    // SHA256(prefix64 || ell_a64 || ell_b64 || x32)
    secp256k1::SHA256 h;
    if (data) h.update(static_cast<const unsigned char*>(data), 64);
    h.update(ell_a64, 64);
    h.update(ell_b64, 64);
    h.update(x32, 32);
    auto result = h.finalize();
    std::memcpy(output, result.data(), 32);
    return 1;
}

extern "C" {

const secp256k1_ellswift_xdh_hash_function secp256k1_ellswift_xdh_hash_function_bip324 = bip324_hash_impl;
const secp256k1_ellswift_xdh_hash_function secp256k1_ellswift_xdh_hash_function_prefix  = prefix_hash_impl;

// -- Encode -------------------------------------------------------------------

int secp256k1_ellswift_encode(
    const secp256k1_context* ctx,
    unsigned char* ell64,
    const secp256k1_pubkey* pubkey,
    const unsigned char* rnd32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!ell64)  { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_encode: NULL ell64");  return 0; }
    if (!pubkey) { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_encode: NULL pubkey"); return 0; }
    if (!rnd32)  { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_encode: NULL rnd32");  return 0; }

    std::array<uint8_t, 32> xb{};
    std::memcpy(xb.data(), pubkey->data, 32);
    auto x = FieldElement::from_bytes(xb);
    auto enc = secp256k1::ellswift_encode_x(x, rnd32);
    std::memcpy(ell64, enc.data(), 64);
    return 1;
}

// -- Decode -------------------------------------------------------------------

int secp256k1_ellswift_decode(
    const secp256k1_context* ctx,
    secp256k1_pubkey* pubkey,
    const unsigned char* ell64)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey || !ell64) return 0;

    auto x = secp256k1::ellswift_decode(ell64);

    // Lift x to a point: y² = x³ + 7
    auto y2 = x * x * x + FieldElement::from_uint64(7);
    auto y = y2.sqrt();
    if (!(y.square() == y2)) return 0;

    // BIP-324 XSwiftEC: y parity must match t parity (t = ell64[32..63]).
    // If t == 0 (mod p) it is replaced by 1 (odd); edge case is included.
    std::array<uint8_t, 32> t_bytes{};
    std::memcpy(t_bytes.data(), ell64 + 32, 32);
    auto t = FieldElement::from_bytes(t_bytes);
    bool t_odd = (t == FieldElement::zero()) ? true : ((t.to_bytes()[31] & 1) != 0);
    bool y_odd = (y.to_bytes()[31] & 1) != 0;
    if (y_odd != t_odd) y.negate_assign();

    auto xb = x.to_bytes();
    auto yb = y.to_bytes();
    std::memcpy(pubkey->data,      xb.data(), 32);
    std::memcpy(pubkey->data + 32, yb.data(), 32);
    return 1;
}

// -- Create -------------------------------------------------------------------

int secp256k1_ellswift_create(
    const secp256k1_context* ctx,
    unsigned char* ell64,
    const unsigned char* seckey32,
    const unsigned char* auxrnd32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!ell64)    { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_create: NULL ell64");    return 0; }
    if (!seckey32) { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_create: NULL seckey32"); return 0; }

    std::array<uint8_t, 32> kb{};
    std::memcpy(kb.data(), seckey32, 32);
    // Strict: reject sk >= n or sk == 0 (Rule 11: no silent mod-n reduction)
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(kb, sk)) {
        secp256k1::detail::secure_erase(&sk, sizeof(sk));   // SHIM-01
        secp256k1::detail::secure_erase(kb.data(), 32);
        return 0;
    }

    // CT path: secret key requires constant-time generator multiplication.
    // ellswift_create() uses ct::generator_mul() internally.
    auto enc = secp256k1::ellswift_create(sk, auxrnd32);
    std::memcpy(ell64, enc.data(), 64);
    // SHIM-01: erase the secret scalar and its byte copy (BIP-324 handshake key).
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(kb.data(), 32);
    return 1;
}

// -- XDH (x-only ECDH) --------------------------------------------------------

int secp256k1_ellswift_xdh(
    const secp256k1_context* ctx,
    unsigned char* output,
    const unsigned char* ell_a64,
    const unsigned char* ell_b64,
    const unsigned char* seckey32,
    int party,
    secp256k1_ellswift_xdh_hash_function hashfp,
    void* data)
{
    SHIM_REQUIRE_CTX(ctx);
    // SHIM-008 fix: NULL hashfp is a programming error — fire illegal callback so
    // callers that registered a no-op fuzz callback can intercept it instead of
    // getting a silent failure (matches upstream libsecp256k1 argument validation).
    if (!hashfp) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_xdh: NULL hashfp");
        return 0;
    }
    if (!output)   { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_xdh: NULL output");   return 0; }
    if (!ell_a64)  { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_xdh: NULL ell_a64");  return 0; }
    if (!ell_b64)  { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_xdh: NULL ell_b64");  return 0; }
    if (!seckey32) { secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ellswift_xdh: NULL seckey32"); return 0; }

    std::array<uint8_t, 32> kb{};
    std::memcpy(kb.data(), seckey32, 32);
    // Strict: reject sk >= n or sk == 0 (Rule 11: no silent mod-n reduction)
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(kb, sk)) {
        secp256k1::detail::secure_erase(&sk, sizeof(sk));   // SHIM-02
        secp256k1::detail::secure_erase(kb.data(), 32);
        return 0;
    }

    bool initiating = (party == 0);

    // BIP-324 path: dedicated backend with transparent peer GLV cache.
    // Bitcoin Core sees identical behavior; internally saves ~1,954 ns on reconnect.
    if (hashfp == secp256k1_ellswift_xdh_hash_function_bip324) {
        auto secret = secp256k1::bip324::xdh(ell_a64, ell_b64, sk, initiating);
        std::memcpy(output, secret.data(), 32);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        secp256k1::detail::secure_erase(kb.data(), 32);
        return 1;
    }

    // General path: decode → ECDH → custom hashfp.
    // SHIM-02: every error return below must erase the secret scalar `sk` and its
    // byte copy `kb` (previously only the success path at the end erased them).
    auto erase_secrets = [&]() {
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        secp256k1::detail::secure_erase(kb.data(), 32);
    };
    const unsigned char* their_ell = initiating ? ell_b64 : ell_a64;

    auto their_x = secp256k1::ellswift_decode(their_ell);
    auto y2 = their_x * their_x * their_x + FieldElement::from_uint64(7);
    auto y = y2.sqrt();
    if (!(y.square() == y2)) { erase_secrets(); return 0; }
    auto yb = y.to_bytes();
    if (yb[31] & 1) y.negate_assign();

    auto their_point = Point::from_affine(their_x, y);
    if (their_point.is_infinity()) { erase_secrets(); return 0; }

    auto ecdh_point = secp256k1::ct::scalar_mul(their_point, sk);
    if (ecdh_point.is_infinity()) { erase_secrets(); return 0; }

    auto x32_arr = ecdh_point.x().to_bytes();
    int rc = hashfp(output, x32_arr.data(), ell_a64, ell_b64, data);
    secp256k1::detail::secure_erase(x32_arr.data(), 32);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(kb.data(), 32);
    return rc;
}

} // extern "C"
