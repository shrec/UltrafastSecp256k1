// ============================================================================
// shim_pubkey.cpp -- Public key parse/serialize/create/tweak
// ============================================================================
#include "secp256k1.h"
#include "shim_internal.hpp"

#include <cstring>
#include <algorithm>
#include <array>
#include <vector>
#include <numeric>

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/ct/point.hpp"

using namespace secp256k1::fast;

// -- Internal helpers ---------------------------------------------------------
// The opaque 64-byte pubkey stores the 65-byte uncompressed form minus prefix.
// Layout: data[0..31] = X big-endian, data[32..63] = Y big-endian.

static void point_to_pubkey_data(const Point& pt, unsigned char data[64]) {
    if (pt.is_normalized()) {
        // Z=1 (affine, z_one_=true): X and Y are the actual affine coordinates.
        // Serialize directly — no field inversion needed (saves ~1,300 ns vs to_uncompressed()).
        // from_affine() always sets z_one_=true, so parse → store paths always hit this fast path.
        pt.x_raw().to_bytes_into(reinterpret_cast<uint8_t*>(data));
        pt.y_raw().to_bytes_into(reinterpret_cast<uint8_t*>(data) + 32);
    } else {
        // Jacobian (Jacobian result from scalar_mul etc.): full inversion via to_uncompressed().
        auto unc = pt.to_uncompressed();
        std::memcpy(data, unc.data() + 1, 64);
    }
}

// `[[maybe_unused]]` because some shim TUs include this file but don't use
// this helper directly, which trips -Werror=unused-function on GCC.
[[maybe_unused]]
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
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey || !input) return 0;

    if (inputlen == 33) {
        // Compressed: 02/03 || X
        uint8_t prefix = input[0];
        if (prefix != 0x02 && prefix != 0x03) return 0;

        // Reject x >= p (libsecp uses secp256k1_fe_set_b32_limit)
        FieldElement x;
        if (!FieldElement::parse_bytes_strict(input + 1, x)) return 0;

        // y^2 = x^3 + 7; reject if x is not a valid curve x-coordinate
        auto y2 = x * x * x + FieldElement::from_uint64(7);
        auto y = y2.sqrt();
        if (!(y.square() == y2)) return 0;

        // Select y with correct parity
        bool y_is_odd = (y.limbs()[0] & 1u) != 0u;
        bool want_odd = (prefix == 0x03);
        if (y_is_odd != want_odd) y = y.negate();

        auto pt = Point::from_affine(x, y);
        point_to_pubkey_data(pt, pubkey->data);
        return 1;

    } else if (inputlen == 65) {
        // Uncompressed (04) or hybrid (06/07): all carry explicit X || Y.
        // libsecp accepts 06/07 without STRICTENC (hybrid pubkey compatibility).
        uint8_t pfx = input[0];
        if (pfx != 0x04 && pfx != 0x06 && pfx != 0x07) return 0;

        // Reject x >= p or y >= p (libsecp strict boundary)
        FieldElement x, y;
        if (!FieldElement::parse_bytes_strict(input + 1,  x)) return 0;
        if (!FieldElement::parse_bytes_strict(input + 33, y)) return 0;
        // Reject if not on curve: y^2 != x^3 + 7
        auto rhs = x * x * x + FieldElement::from_uint64(7);
        if (!(y.square() == rhs)) return 0;
        // For hybrid prefix: validate Y parity matches (06 = even Y, 07 = odd Y)
        if (pfx == 0x06 || pfx == 0x07) {
            bool y_is_odd = (y.limbs()[0] & 1u) != 0u;
            if (y_is_odd != (pfx == 0x07)) return 0;
        }
        auto pt = Point::from_affine(x, y);
        point_to_pubkey_data(pt, pubkey->data);
        return 1;
    }

    return 0;
}

int secp256k1_ec_pubkey_serialize(
    const secp256k1_context *ctx, unsigned char *output, size_t *outputlen,
    const secp256k1_pubkey *pubkey, unsigned int flags)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!output || !outputlen || !pubkey) return 0;

    // Validate flags: only SECP256K1_EC_COMPRESSED or SECP256K1_EC_UNCOMPRESSED
    // are valid. Garbage flags (e.g. 0xDEAD) must fire the illegal callback
    // (PASS3-011 divergence fix — matches upstream libsecp256k1 behavior).
    const unsigned int valid_flags = SECP256K1_EC_COMPRESSED | SECP256K1_EC_UNCOMPRESSED;
    if ((flags & ~valid_flags) != 0 ||
        (flags != SECP256K1_EC_COMPRESSED && flags != SECP256K1_EC_UNCOMPRESSED)) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_serialize: invalid flags");
        return 0;
    }

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
    if (!pubkey1 || !pubkey2) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_ec_pubkey_cmp: invalid pubkey argument");
        return 0;
    }
    // Compare compressed serializations lexicographically.
    // Zero-initialize so unwritten bytes don't produce UB in memcmp.
    unsigned char c1[33]{}, c2[33]{};
    size_t len1 = 33, len2 = 33;
    secp256k1_ec_pubkey_serialize(ctx, c1, &len1, pubkey1, SECP256K1_EC_COMPRESSED);
    secp256k1_ec_pubkey_serialize(ctx, c2, &len2, pubkey2, SECP256K1_EC_COMPRESSED);
    return std::memcmp(c1, c2, 33);
}

int secp256k1_ec_pubkey_create(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *seckey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey || !seckey) return 0;

    Scalar k;
    if (!Scalar::parse_bytes_strict_nonzero(seckey, k)) return 0;
    auto P = secp256k1::ct::generator_mul(k);   // CT: Rule 12 — sk is a private key
    if (P.is_infinity()) return 0;
    point_to_pubkey_data(P, pubkey->data);
    return 1;
}

int secp256k1_ec_pubkey_negate(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey) return 0;
    auto P = pubkey_data_to_point(pubkey->data);
    auto neg = P.negate();
    point_to_pubkey_data(neg, pubkey->data);
    return 1;
}

int secp256k1_ec_pubkey_tweak_add(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey || !tweak32) return 0;
    auto P = pubkey_data_to_point(pubkey->data);
    // tweak in [0, n-1]; 0 is valid (result == pubkey)
    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) return 0;
    auto T = scalar_mul_generator(t);
    auto result = P.add(T);
    if (result.is_infinity()) return 0;
    point_to_pubkey_data(result, pubkey->data);
    return 1;
}

int secp256k1_ec_pubkey_tweak_mul(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey || !tweak32) return 0;
    auto P = pubkey_data_to_point(pubkey->data);
    Scalar t;
    if (!Scalar::parse_bytes_strict_nonzero(tweak32, t)) return 0;
    auto result = P.scalar_mul(t);
    if (result.is_infinity()) return 0;
    point_to_pubkey_data(result, pubkey->data);
    return 1;
}

int secp256k1_ec_pubkey_combine(
    const secp256k1_context *ctx, secp256k1_pubkey *out,
    const secp256k1_pubkey * const *ins, size_t n)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!out || !ins || n == 0) return 0;
    for (size_t i = 0; i < n; ++i) { if (!ins[i]) return 0; }
    auto acc = pubkey_data_to_point(ins[0]->data);
    for (size_t i = 1; i < n; ++i) {
        auto P = pubkey_data_to_point(ins[i]->data);
        acc = acc.add(P);
    }
    if (acc.is_infinity()) return 0;
    point_to_pubkey_data(acc, out->data);
    return 1;
}

void secp256k1_ec_pubkey_sort(
    const secp256k1_context *ctx,
    const secp256k1_pubkey **pubkeys,
    size_t n_pubkeys)
{
    if (!pubkeys) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_ec_pubkey_sort: NULL pubkeys array");
        return;
    }
    for (size_t i = 0; i < n_pubkeys; ++i) {
        if (!pubkeys[i]) {
            secp256k1_shim_call_illegal_cb(ctx,
                "secp256k1_ec_pubkey_sort: NULL pubkey element");
            return;
        }
    }

    if (n_pubkeys == 0) return;

    // PERF-004: pre-compute all compressed serializations before sorting to avoid
    // O(N log N) calls to secp256k1_ec_pubkey_serialize inside the comparator.
    // Each comparator call previously serialized both keys — O(N log N) × 2 ops.
    // Now: O(N) serializations up-front, then sort indices on pre-computed buffers.
    std::vector<std::array<unsigned char, 33>> bufs(n_pubkeys);
    for (size_t i = 0; i < n_pubkeys; ++i) {
        bufs[i] = {};  // zero-initialize: well-defined memcmp even on serialize failure
        size_t len = 33;
        secp256k1_ec_pubkey_serialize(secp256k1_context_static, bufs[i].data(), &len,
                                      pubkeys[i], SECP256K1_EC_COMPRESSED);
    }

    // Sort an index array, then apply the permutation to pubkeys in-place.
    std::vector<size_t> idx(n_pubkeys);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
        [&bufs](size_t a, size_t b) {
            return std::memcmp(bufs[a].data(), bufs[b].data(), 33) < 0;
        });

    // Apply permutation: build sorted pointer array, then copy back.
    std::vector<const secp256k1_pubkey*> sorted(n_pubkeys);
    for (size_t i = 0; i < n_pubkeys; ++i) sorted[i] = pubkeys[idx[i]];
    for (size_t i = 0; i < n_pubkeys; ++i) pubkeys[i] = sorted[i];
}

} // extern "C"
