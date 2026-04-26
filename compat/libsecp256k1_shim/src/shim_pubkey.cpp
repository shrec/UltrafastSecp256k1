// ============================================================================
// shim_pubkey.cpp -- Public key parse/serialize/create/tweak
// ============================================================================
#include "secp256k1.h"

#include <cstring>
#include <array>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"

using namespace secp256k1::fast;

// -- Internal helpers ---------------------------------------------------------
// The opaque 64-byte pubkey stores the 65-byte uncompressed form minus prefix.
// Layout: data[0..31] = X big-endian, data[32..63] = Y big-endian.

static void point_to_pubkey_data(const Point& pt, unsigned char data[64]) {
    auto unc = pt.to_uncompressed(); // 65 bytes: 04 || X || Y
    std::memcpy(data, unc.data() + 1, 64);
}

static Point pubkey_data_to_point(const unsigned char data[64]) {
    std::array<uint8_t, 32> xb{}, yb{};
    std::memcpy(xb.data(), data, 32);
    std::memcpy(yb.data(), data + 32, 32);
    auto x = FieldElement::from_bytes(xb);
    auto y = FieldElement::from_bytes(yb);
    return Point::from_affine(x, y);
}

extern "C" {

int secp256k1_ec_pubkey_parse(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *input, size_t inputlen)
{
    (void)ctx;
    if (!pubkey || !input) return 0;

    try {
        if (inputlen == 33) {
            // Compressed: 02/03 || X
            uint8_t prefix = input[0];
            if (prefix != 0x02 && prefix != 0x03) return 0;

            std::array<uint8_t, 32> xb{};
            std::memcpy(xb.data(), input + 1, 32);
            auto x = FieldElement::from_bytes(xb);

            // y^2 = x^3 + 7; reject if x is not a valid curve x-coordinate
            auto y2 = x * x * x + FieldElement::from_uint64(7);
            auto y = y2.sqrt();
            if (!(y.square() == y2)) return 0;

            // Select y with correct parity
            auto yb = y.to_bytes();
            bool y_is_odd = (yb[31] & 1) != 0;
            bool want_odd = (prefix == 0x03);
            if (y_is_odd != want_odd) y = y.negate();

            auto pt = Point::from_affine(x, y);
            point_to_pubkey_data(pt, pubkey->data);
            return 1;

        } else if (inputlen == 65) {
            // Uncompressed: 04 || X || Y
            if (input[0] != 0x04) return 0;

            std::array<uint8_t, 32> xb{}, yb_arr{};
            std::memcpy(xb.data(), input + 1, 32);
            std::memcpy(yb_arr.data(), input + 33, 32);
            auto x = FieldElement::from_bytes(xb);
            auto y = FieldElement::from_bytes(yb_arr);
            // Reject if not on curve: y^2 != x^3 + 7
            auto rhs = x * x * x + FieldElement::from_uint64(7);
            if (!(y.square() == rhs)) return 0;
            auto pt = Point::from_affine(x, y);
            point_to_pubkey_data(pt, pubkey->data);
            return 1;
        }
    } catch (...) {}

    return 0;
}

int secp256k1_ec_pubkey_serialize(
    const secp256k1_context *ctx, unsigned char *output, size_t *outputlen,
    const secp256k1_pubkey *pubkey, unsigned int flags)
{
    (void)ctx;
    if (!output || !outputlen || !pubkey) return 0;

    if (flags & SECP256K1_FLAGS_BIT_COMPRESSION) {
        // Compressed
        if (*outputlen < 33) return 0;
        bool y_is_odd = (pubkey->data[63] & 1) != 0;
        output[0] = y_is_odd ? 0x03 : 0x02;
        std::memcpy(output + 1, pubkey->data, 32);
        *outputlen = 33;
    } else {
        // Uncompressed
        if (*outputlen < 65) return 0;
        output[0] = 0x04;
        std::memcpy(output + 1, pubkey->data, 64);
        *outputlen = 65;
    }
    return 1;
}

int secp256k1_ec_pubkey_cmp(
    const secp256k1_context *ctx,
    const secp256k1_pubkey *pubkey1, const secp256k1_pubkey *pubkey2)
{
    (void)ctx;
    // Compare compressed serializations lexicographically
    unsigned char c1[33], c2[33];
    size_t len1 = 33, len2 = 33;
    secp256k1_ec_pubkey_serialize(ctx, c1, &len1, pubkey1, SECP256K1_EC_COMPRESSED);
    secp256k1_ec_pubkey_serialize(ctx, c2, &len2, pubkey2, SECP256K1_EC_COMPRESSED);
    return std::memcmp(c1, c2, 33);
}

int secp256k1_ec_pubkey_create(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *seckey)
{
    (void)ctx;
    if (!pubkey || !seckey) return 0;

    try {
        std::array<uint8_t, 32> kb{};
        std::memcpy(kb.data(), seckey, 32);
        auto k = Scalar::from_bytes(kb);
        if (k.is_zero()) return 0;
        auto P = scalar_mul_generator(k);
        if (P.is_infinity()) return 0;
        point_to_pubkey_data(P, pubkey->data);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_ec_pubkey_negate(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey)
{
    (void)ctx;
    if (!pubkey) return 0;
    try {
        auto P = pubkey_data_to_point(pubkey->data);
        auto neg = P.negate();
        point_to_pubkey_data(neg, pubkey->data);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_ec_pubkey_tweak_add(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *tweak32)
{
    (void)ctx;
    if (!pubkey || !tweak32) return 0;
    try {
        auto P = pubkey_data_to_point(pubkey->data);
        std::array<uint8_t, 32> tb{};
        std::memcpy(tb.data(), tweak32, 32);
        auto t = Scalar::from_bytes(tb);
        auto T = scalar_mul_generator(t);
        auto result = P.add(T);
        if (result.is_infinity()) return 0;
        point_to_pubkey_data(result, pubkey->data);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_ec_pubkey_tweak_mul(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *tweak32)
{
    (void)ctx;
    if (!pubkey || !tweak32) return 0;
    try {
        auto P = pubkey_data_to_point(pubkey->data);
        std::array<uint8_t, 32> tb{};
        std::memcpy(tb.data(), tweak32, 32);
        auto t = Scalar::from_bytes(tb);
        if (t.is_zero()) return 0;
        auto result = P.scalar_mul(t);
        if (result.is_infinity()) return 0;
        point_to_pubkey_data(result, pubkey->data);
        return 1;
    } catch (...) { return 0; }
}

int secp256k1_ec_pubkey_combine(
    const secp256k1_context *ctx, secp256k1_pubkey *out,
    const secp256k1_pubkey * const *ins, size_t n)
{
    (void)ctx;
    if (!out || !ins || n == 0) return 0;
    try {
        auto acc = pubkey_data_to_point(ins[0]->data);
        for (size_t i = 1; i < n; ++i) {
            auto P = pubkey_data_to_point(ins[i]->data);
            acc = acc.add(P);
        }
        if (acc.is_infinity()) return 0;
        point_to_pubkey_data(acc, out->data);
        return 1;
    } catch (...) { return 0; }
}

} // extern "C"
