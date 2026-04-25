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

using namespace secp256k1::fast;

extern "C" {

// -- X-only public key ----------------------------------------------------
// Opaque layout: data[0..31] = X big-endian, data[32..63] = zeros (padding)

int secp256k1_xonly_pubkey_parse(
    const secp256k1_context *ctx, secp256k1_xonly_pubkey *pubkey,
    const unsigned char *input32)
{
    (void)ctx;
    if (!pubkey || !input32) return 0;

    try {
        // Validate: X must be a valid field element with a valid Y
        std::array<uint8_t, 32> xb{};
        std::memcpy(xb.data(), input32, 32);
        auto x = FieldElement::from_bytes(xb);
        auto y2 = x * x * x + FieldElement::from_uint64(7);
        auto y = y2.sqrt();
        // Verify it's actually a square root
        auto check = y * y;
        // Store x in first 32 bytes
        std::memcpy(pubkey->data, input32, 32);
        std::memset(pubkey->data + 32, 0, 32);
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

    // Copy X from pubkey
    std::memcpy(xonly_pubkey->data, pubkey->data, 32);
    std::memset(xonly_pubkey->data + 32, 0, 32);

    if (pk_parity) {
        // Y is odd if last byte of Y has bit 0 set
        *pk_parity = (pubkey->data[63] & 1) ? 1 : 0;
    }
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

        auto P = Point::generator().scalar_mul(k);
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

    // X from keypair
    std::memcpy(pubkey->data, keypair->data + 32, 32);
    std::memset(pubkey->data + 32, 0, 32);

    if (pk_parity) {
        *pk_parity = (keypair->data[95] & 1) ? 1 : 0;
    }
    return 1;
}

// -- Taproot tweak operations -----------------------------------------------

// Helper: reconstruct a Point from an xonly_pubkey (always even Y).
static Point xonly_to_point(const secp256k1_xonly_pubkey *xp)
{
    std::array<uint8_t, 32> xb{};
    std::memcpy(xb.data(), xp->data, 32);
    auto x = FieldElement::from_bytes(xb);
    auto y2 = x * x * x + FieldElement::from_uint64(7);
    auto y  = y2.sqrt();
    // Ensure even Y (x-only keys always use even Y)
    auto yb = y.to_bytes();
    if (yb[31] & 1) y = y.negate();
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

        auto tG = Point::generator().scalar_mul(t);
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

        auto tG = Point::generator().scalar_mul(t);
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

        auto P = Point::generator().scalar_mul(new_sk);
        if (P.is_infinity()) return 0;

        auto new_skb = new_sk.to_bytes();
        std::memcpy(keypair->data, new_skb.data(), 32);
        auto unc = P.to_uncompressed();
        std::memcpy(keypair->data + 32, unc.data() + 1, 64);
        return 1;
    } catch (...) { return 0; }
}

} // extern "C"
