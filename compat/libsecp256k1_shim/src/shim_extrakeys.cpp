// ============================================================================
// shim_extrakeys.cpp -- x-only pubkeys and keypairs
// ============================================================================
#include "secp256k1_extrakeys.h"
#include "secp256k1.h"

#include <cstring>
#include <array>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"

using namespace secp256k1::fast;

extern "C" {

// -- X-only public key ----------------------------------------------------
// Opaque layout: data[0..31] = X big-endian, data[32..63] = Y big-endian (even parity)
// Caching Y avoids re-lifting (sqrt) on every Schnorr verify — saves ~1.5 us/verify.

int secp256k1_xonly_pubkey_parse(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    const unsigned char *input32)
{
    (void)ctx;
    if (!pubkey || !input32) return 0;

    try {
        std::array<uint8_t, 32> xb{};
        std::memcpy(xb.data(), input32, 32);
        auto x = FieldElement::from_bytes(xb);
        auto y2 = x * x * x + FieldElement::from_uint64(7);
        auto y = y2.sqrt();
        // Ensure even Y (x-only canonical form)
        auto yb = y.to_bytes();
        if (yb[31] & 1) y = y.negate();
        yb = y.to_bytes();
        // Store X || Y (even)
        std::memcpy(pubkey->data,      input32, 32);
        std::memcpy(pubkey->data + 32, yb.data(), 32);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_xonly_pubkey_serialize(
    const secp256k1_context *ctx, unsigned char *output32,
    const secp256k1_xonly_pubkey *pubkey)
{
    (void)ctx;
    if (!output32 || !pubkey) return 0;
    std::memcpy(output32, pubkey->data, 32);
    return 1;
}

int secp256k1_xonly_pubkey_cmp(
    const secp256k1_context *ctx,
    const secp256k1_xonly_pubkey *pk1,
    const secp256k1_xonly_pubkey *pk2)
{
    (void)ctx;
    return std::memcmp(pk1->data, pk2->data, 32);
}

int secp256k1_xonly_pubkey_from_pubkey(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *xonly_pubkey,
    int *pk_parity, const secp256k1_pubkey *pubkey)
{
    (void)ctx;
    if (!xonly_pubkey || !pubkey) return 0;

    // pubkey layout: data[0..31] = X, data[32..63] = Y (both big-endian)
    int y_is_odd = (pubkey->data[63] & 1) ? 1 : 0;

    // X-only key always uses even Y: if pubkey Y is odd, negate Y
    std::memcpy(xonly_pubkey->data, pubkey->data, 32); // copy X
    if (!y_is_odd) {
        std::memcpy(xonly_pubkey->data + 32, pubkey->data + 32, 32); // even Y, use as-is
    } else {
        // Negate Y: -Y mod p
        std::array<uint8_t, 32> yb{};
        std::memcpy(yb.data(), pubkey->data + 32, 32);
        auto y = FieldElement::from_bytes(yb);
        y = y.negate();
        auto yn = y.to_bytes();
        std::memcpy(xonly_pubkey->data + 32, yn.data(), 32);
    }

    if (pk_parity) *pk_parity = y_is_odd;
    return 1;
}

// -- Keypair --------------------------------------------------------------
// Layout: data[0..31] = secret key, data[32..95] = pubkey opaque (X || Y)

int secp256k1_keypair_create(
    const secp256k1_context *ctx, secp256k1_keypair *keypair,
    const unsigned char *seckey)
{
    (void)ctx;
    if (!keypair || !seckey) return 0;

    try {
        std::array<uint8_t, 32> kb{};
        std::memcpy(kb.data(), seckey, 32);
        auto k = Scalar::from_bytes(kb);
        if (k.is_zero()) return 0;

        auto P = scalar_mul_generator(k);
        if (P.is_infinity()) return 0;

        // Store seckey
        std::memcpy(keypair->data, seckey, 32);

        // Store pubkey (X || Y)
        auto unc = P.to_uncompressed();
        std::memcpy(keypair->data + 32, unc.data() + 1, 64);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_keypair_sec(
    const secp256k1_context *ctx, unsigned char *seckey,
    const secp256k1_keypair *keypair)
{
    (void)ctx;
    if (!seckey || !keypair) return 0;
    std::memcpy(seckey, keypair->data, 32);
    return 1;
}

int secp256k1_keypair_pub(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const secp256k1_keypair *keypair)
{
    (void)ctx;
    if (!pubkey || !keypair) return 0;
    std::memcpy(pubkey->data, keypair->data + 32, 64);
    return 1;
}

int secp256k1_keypair_xonly_pub(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    int *pk_parity, const secp256k1_keypair *keypair)
{
    (void)ctx;
    if (!pubkey || !keypair) return 0;

    // keypair layout: data[0..31]=sk, data[32..63]=X, data[64..95]=Y (big-endian)
    int y_is_odd = (keypair->data[95] & 1) ? 1 : 0;

    std::memcpy(pubkey->data, keypair->data + 32, 32); // X

    // x-only canonical form: use even Y
    if (!y_is_odd) {
        std::memcpy(pubkey->data + 32, keypair->data + 64, 32);
    } else {
        std::array<uint8_t, 32> yb{};
        std::memcpy(yb.data(), keypair->data + 64, 32);
        auto y = FieldElement::from_bytes(yb);
        y = y.negate();
        auto yn = y.to_bytes();
        std::memcpy(pubkey->data + 32, yn.data(), 32);
    }

    if (pk_parity) *pk_parity = y_is_odd;
    return 1;
}

// -- Taproot tweak operations -----------------------------------------------

// Helper: reconstruct a Point from an xonly_pubkey using cached X||Y — no sqrt.
static Point xonly_to_point(const secp256k1_xonly_pubkey *xp)
{
    std::array<uint8_t, 32> xb{}, yb{};
    std::memcpy(xb.data(), xp->data,      32);
    std::memcpy(yb.data(), xp->data + 32, 32);
    auto x = FieldElement::from_bytes(xb);
    auto y = FieldElement::from_bytes(yb);
    return Point::from_affine(x, y);
}

int secp256k1_xonly_pubkey_tweak_add(
    const secp256k1_context *ctx,
    secp256k1_pubkey *output_pubkey,
    const secp256k1_xonly_pubkey *internal_pubkey,
    const unsigned char *tweak32)
{
    (void)ctx;
    if (!output_pubkey || !internal_pubkey || !tweak32) return 0;

    try {
        auto P = xonly_to_point(internal_pubkey);
        if (P.is_infinity()) return 0;

        std::array<uint8_t, 32> tb{};
        std::memcpy(tb.data(), tweak32, 32);
        auto t = Scalar::from_bytes(tb);
        // t == 0 is technically valid per libsecp (returns P itself), but
        // treat as error to match libsecp behavior (returns 0 for zero tweak)
        if (t.is_zero()) return 0;

        auto tG = scalar_mul_generator(t);
        auto Q  = P.add(tG);
        if (Q.is_infinity()) return 0;

        auto unc = Q.to_uncompressed();
        std::memcpy(output_pubkey->data, unc.data() + 1, 64);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_xonly_pubkey_tweak_add_check(
    const secp256k1_context *ctx,
    const unsigned char *tweaked_pubkey32,
    int tweaked_pk_parity,
    const secp256k1_xonly_pubkey *internal_pubkey,
    const unsigned char *tweak32)
{
    (void)ctx;
    if (!tweaked_pubkey32 || !internal_pubkey || !tweak32) return 0;

    try {
        auto P = xonly_to_point(internal_pubkey);
        if (P.is_infinity()) return 0;

        std::array<uint8_t, 32> tb{};
        std::memcpy(tb.data(), tweak32, 32);
        auto t = Scalar::from_bytes(tb);
        if (t.is_zero()) return 0;

        auto tG = scalar_mul_generator(t);
        auto Q  = P.add(tG);
        if (Q.is_infinity()) return 0;

        auto unc = Q.to_uncompressed(); // 65 bytes: 04 || X || Y
        int q_parity = (unc[64] & 1) ? 1 : 0;

        if (q_parity != tweaked_pk_parity) return 0;
        return std::memcmp(unc.data() + 1, tweaked_pubkey32, 32) == 0 ? 1 : 0;
    } catch (...) { return 0; }
}

int secp256k1_keypair_xonly_tweak_add(
    const secp256k1_context *ctx,
    secp256k1_keypair *keypair,
    const unsigned char *tweak32)
{
    (void)ctx;
    if (!keypair || !tweak32) return 0;

    try {
        std::array<uint8_t, 32> skb{};
        std::memcpy(skb.data(), keypair->data, 32);
        auto sk = Scalar::from_bytes(skb);
        if (sk.is_zero()) return 0;

        // If Y is odd, negate the secret key (so the x-only key has even Y)
        int y_parity = (keypair->data[95] & 1) ? 1 : 0;
        if (y_parity) sk = sk.negate();

        std::array<uint8_t, 32> tb{};
        std::memcpy(tb.data(), tweak32, 32);
        auto t = Scalar::from_bytes(tb);
        if (t.is_zero()) return 0;

        auto new_sk = sk + t;
        if (new_sk.is_zero()) return 0;

        auto P = scalar_mul_generator(new_sk);
        if (P.is_infinity()) return 0;

        auto new_skb = new_sk.to_bytes();
        std::memcpy(keypair->data, new_skb.data(), 32);
        auto unc = P.to_uncompressed();
        std::memcpy(keypair->data + 32, unc.data() + 1, 64);
        return 1;
    } catch (...) { return 0; }
}

} // extern "C"
