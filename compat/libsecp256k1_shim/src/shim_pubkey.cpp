// ============================================================================
// shim_pubkey.cpp -- Public key parse/serialize/create/tweak
// ============================================================================
#include "secp256k1.h"
#include "shim_internal.hpp"
#include "shim_pubkey_helpers.hpp"

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
#include "secp256k1/detail/secure_erase.hpp"   // secret-residue sweep (CT-04/RT-05 class)

using namespace secp256k1::fast;

// point_to_pubkey_data and pubkey_data_to_point from shim_pubkey_helpers.hpp
using secp256k1_shim_internal::point_to_pubkey_data;
using secp256k1_shim_internal::pubkey_data_to_point;
// Manipulation / serialization paths use the curve-checked loader (fail-closed on
// an off-curve / malformed opaque pubkey); the hot verify paths use the unchecked one.
using secp256k1_shim_internal::pubkey_data_to_point_checked;

// -- Thread-local pubkey-parse (decompression) cache --------------------------
// PERF: parsing a COMPRESSED pubkey computes y = sqrt(x^3 + 7) — a full field
// square root that dominates the parse cost (~5% of a P2WPKH ECDSA verify, perf
// 2026-05-30). Bitcoin Core re-parses the same pubkey on every CPubKey::Verify,
// so reused-key workloads (wallets, ConnectBlock with few distinct keys) pay the
// sqrt repeatedly. This cache stores the decompressed 64-byte pubkey->data keyed
// on the raw input bytes; a repeat parse returns the bytes and skips the sqrt.
//
// Public data only — pubkeys are public, no secret material, no constant-time
// concern (mirrors the verify-side ShimSchnorrCache in shim_schnorr.cpp). Only
// SUCCESSFUL parses are cached; invalid inputs fall through and re-validate.
// Unique-pubkey workloads pay only a ~64-byte slot write per miss (negligible
// vs the sqrt saved on hits) — never a regression. The Schnorr x-only path
// already caches its lift_x; this brings the ECDSA path to the same footing.
namespace {
struct ShimPubkeyParseCache {
    static constexpr std::size_t SLOTS = 256;  // power of two for mask indexing
    struct Slot {
        std::uint64_t fp{0};
        std::uint8_t  in[65]{};        // raw input bytes (33 compressed or 65 uncompressed)
        std::uint8_t  inlen{0};
        std::uint8_t  data[64]{};      // decompressed pubkey->data
        bool          valid{false};
    };
    Slot slots[SLOTS]{};

    // FNV-1a over (input bytes, length). Public data → no per-thread salt needed:
    // a slot collision only forces a cache miss (re-validate), never a leak.
    static std::uint64_t hash(const unsigned char* in, std::size_t n, std::size_t& idx) noexcept {
        std::uint64_t h = 14695981039346656037ULL;
        for (std::size_t i = 0; i < n; ++i) h = (h ^ in[i]) * 1099511628211ULL;
        h = (h ^ n) * 1099511628211ULL;
        idx = static_cast<std::size_t>(h & (SLOTS - 1));
        return h | 1ULL;  // nonzero fingerprint
    }
    const std::uint8_t* get(const unsigned char* in, std::size_t n,
                            std::size_t& idx, std::uint64_t& fp) const noexcept {
        fp = hash(in, n, idx);
        const Slot& s = slots[idx];
        if (s.valid && s.fp == fp && s.inlen == n && std::memcmp(s.in, in, n) == 0)
            return s.data;
        return nullptr;
    }
    void put(const unsigned char* in, std::size_t n, std::size_t idx,
             std::uint64_t fp, const std::uint8_t data[64]) noexcept {
        Slot& s = slots[idx];
        s.fp = fp;
        s.inlen = static_cast<std::uint8_t>(n);
        std::memcpy(s.in, in, n);
        std::memcpy(s.data, data, 64);
        s.valid = true;
    }
};
thread_local ShimPubkeyParseCache s_pubkey_parse_cache;
} // namespace

extern "C" {

int secp256k1_ec_pubkey_parse(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *input, size_t inputlen)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_parse: pubkey is NULL");
        return 0;
    }
    if (!input) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_parse: input is NULL");
        return 0;
    }
    // Decompression cache: on a repeated pubkey, return the stored 64-byte data
    // and skip the parse (compressed path's sqrt is the hot cost). Computes the
    // slot index / fingerprint once; reused by put() on a miss.
    std::size_t cidx = 0;
    std::uint64_t cfp = 0;
    if (inputlen == 33 || inputlen == 65) {
        if (const std::uint8_t* hit =
                s_pubkey_parse_cache.get(input, inputlen, cidx, cfp)) {
            std::memcpy(pubkey->data, hit, 64);
            return 1;
        }
    }

    std::memset(pubkey->data, 0, sizeof(pubkey->data));

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
        s_pubkey_parse_cache.put(input, inputlen, cidx, cfp, pubkey->data);
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
        s_pubkey_parse_cache.put(input, inputlen, cidx, cfp, pubkey->data);
        return 1;
    }

    return 0;
}

int secp256k1_ec_pubkey_serialize(
    const secp256k1_context *ctx, unsigned char *output, size_t *outputlen,
    const secp256k1_pubkey *pubkey, unsigned int flags)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!output || !outputlen || !pubkey) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_ec_pubkey_serialize: NULL argument");
        return 0;
    }

    // Validate flags: only SECP256K1_EC_COMPRESSED or SECP256K1_EC_UNCOMPRESSED
    // are valid. Garbage flags (e.g. 0xDEAD) must fire the illegal callback
    // (PASS3-011 divergence fix — matches upstream libsecp256k1 behavior).
    const unsigned int valid_flags = SECP256K1_EC_COMPRESSED | SECP256K1_EC_UNCOMPRESSED;
    if ((flags & ~valid_flags) != 0 ||
        (flags != SECP256K1_EC_COMPRESSED && flags != SECP256K1_EC_UNCOMPRESSED)) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_serialize: invalid flags");
        return 0;
    }
    auto P = pubkey_data_to_point_checked(pubkey->data);
    if (P.is_infinity()) {
        *outputlen = 0;
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
    // SHIM-005: NULL ctx must fire the illegal callback before any other check.
    // The previous code only fired the callback when pubkeys were null, passing
    // NULL ctx to secp256k1_shim_call_illegal_cb which is itself unsafe. This
    // guard fires first so the callback receives a valid (non-NULL) ctx, and
    // matches upstream libsecp256k1 behavior (NULL ctx → callback → abort).
    if (SECP256K1_UNLIKELY(!ctx)) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_cmp: NULL ctx");
        return 0;
    }
    if (!pubkey1 || !pubkey2) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_ec_pubkey_cmp: invalid pubkey argument");
        return 0;
    }
    // Compare compressed representations directly from the internal layout.
    // secp256k1_pubkey.data[0..31] = X coordinate (big-endian).
    // secp256k1_pubkey.data[63] & 1 = Y parity bit (0=even → 0x02, 1=odd → 0x03).
    // This avoids two full serialize calls (flag validation, outputlen check, Y-parity
    // extraction) and produces an identical lexicographic result.
    unsigned char p1 = (pubkey1->data[63] & 1u) ? 0x03u : 0x02u;
    unsigned char p2 = (pubkey2->data[63] & 1u) ? 0x03u : 0x02u;
    if (p1 != p2) return static_cast<int>(p1) - static_cast<int>(p2);
    return std::memcmp(pubkey1->data, pubkey2->data, 32);
}

int secp256k1_ec_pubkey_create(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *seckey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey || !seckey) {
        secp256k1_shim_call_illegal_cb(ctx,
            "secp256k1_ec_pubkey_create: NULL argument");
        return 0;
    }

    Scalar k;
    if (!Scalar::parse_bytes_strict_nonzero(seckey, k)) {
        secp256k1::detail::secure_erase(&k, sizeof(k));   // secret-residue sweep
        // PASS3-SHIM-CREATE-ZERO: zero the output on failure to match upstream
        // secp256k1_ec_pubkey_create's memczero(pubkey, !ret) and the shim's own
        // convention (keypair_create, ec_pubkey_negate). Was the only seckey-consuming
        // creation API that left the caller's output buffer untouched on failure.
        std::memset(pubkey->data, 0, sizeof(pubkey->data));
        return 0;
    }
    auto P = secp256k1::ct::generator_mul(k);   // CT: Rule 12 — sk is a private key
    secp256k1::detail::secure_erase(&k, sizeof(k));   // erase parsed private key after last use
    if (P.is_infinity()) {
        std::memset(pubkey->data, 0, sizeof(pubkey->data));  // PASS3-SHIM-CREATE-ZERO
        return 0;
    }
    point_to_pubkey_data(P, pubkey->data);
    return 1;
}

int secp256k1_ec_pubkey_negate(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_negate: pubkey is NULL");
        return 0;
    }
    auto P = pubkey_data_to_point_checked(pubkey->data);
    if (P.is_infinity()) {
        std::memset(pubkey->data, 0, sizeof(pubkey->data));
        return 0;
    }  // SHIM-002: off-curve input rejected
    auto neg = P.negate();
    point_to_pubkey_data(neg, pubkey->data);
    return 1;
}

int secp256k1_ec_pubkey_tweak_add(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_tweak_add: pubkey is NULL");
        return 0;
    }
    if (!tweak32) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_tweak_add: tweak32 is NULL");
        return 0;
    }
    auto P = pubkey_data_to_point_checked(pubkey->data);
    if (P.is_infinity()) { std::memset(pubkey->data, 0, sizeof(pubkey->data)); return 0; }
    // tweak in [0, n-1]; 0 is valid (result == pubkey)
    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) { std::memset(pubkey->data, 0, sizeof(pubkey->data)); return 0; }
    auto T = scalar_mul_generator(t);
    auto result = P.add(T);
    if (result.is_infinity()) { std::memset(pubkey->data, 0, sizeof(pubkey->data)); return 0; }
    point_to_pubkey_data(result, pubkey->data);
    return 1;
}

int secp256k1_ec_pubkey_tweak_mul(
    const secp256k1_context *ctx, secp256k1_pubkey *pubkey,
    const unsigned char *tweak32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!pubkey) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_tweak_mul: pubkey is NULL");
        return 0;
    }
    if (!tweak32) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_tweak_mul: tweak32 is NULL");
        return 0;
    }
    auto P = pubkey_data_to_point_checked(pubkey->data);
    Scalar t;
    if (P.is_infinity()) { std::memset(pubkey->data, 0, sizeof(pubkey->data)); return 0; }
    if (!Scalar::parse_bytes_strict_nonzero(tweak32, t)) { std::memset(pubkey->data, 0, sizeof(pubkey->data)); return 0; }
    auto result = P.scalar_mul(t);
    if (result.is_infinity()) { std::memset(pubkey->data, 0, sizeof(pubkey->data)); return 0; }
    point_to_pubkey_data(result, pubkey->data);
    return 1;
}

int secp256k1_ec_pubkey_combine(
    const secp256k1_context *ctx, secp256k1_pubkey *out,
    const secp256k1_pubkey * const *ins, size_t n)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!out) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_combine: out is NULL");
        return 0;
    }
    if (!ins) {
        secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_combine: ins is NULL");
        return 0;
    }
    if (n == 0) { std::memset(out->data, 0, sizeof(out->data)); return 0; }
    for (size_t i = 0; i < n; ++i) {
        if (!ins[i]) {
            secp256k1_shim_call_illegal_cb(ctx, "secp256k1_ec_pubkey_combine: ins[i] is NULL");
            return 0;
        }
    }
    auto acc = pubkey_data_to_point_checked(ins[0]->data);
    if (acc.is_infinity()) { std::memset(out->data, 0, sizeof(out->data)); return 0; }
    for (size_t i = 1; i < n; ++i) {
        auto P = pubkey_data_to_point_checked(ins[i]->data);
        if (P.is_infinity()) { std::memset(out->data, 0, sizeof(out->data)); return 0; }
        acc = acc.add(P);
    }
    if (acc.is_infinity()) { std::memset(out->data, 0, sizeof(out->data)); return 0; }
    point_to_pubkey_data(acc, out->data);
    return 1;
}

void secp256k1_ec_pubkey_sort(
    const secp256k1_context *ctx,
    const secp256k1_pubkey **pubkeys,
    size_t n_pubkeys)
{
    if (!ctx) {
        secp256k1_shim_call_illegal_cb(nullptr, "secp256k1_ec_pubkey_sort: ctx is NULL");
        return;
    }
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
    //
    // PERF-B3: SBO for N≤16 — stack-allocate bufs, idx, sorted to avoid three
    // heap allocations for the common MuSig2 case (N=2) and other small sorts.
    static constexpr size_t kSBOLimit = 16;

    if (n_pubkeys <= kSBOLimit) {
        // Stack path: no heap allocation for N≤16.
        std::array<std::array<unsigned char, 33>, kSBOLimit> bufs{};
        std::array<size_t, kSBOLimit> idx{};
        std::array<const secp256k1_pubkey*, kSBOLimit> sorted{};

        for (size_t i = 0; i < n_pubkeys; ++i) {
            bufs[i] = {};
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp256k1_context_static, bufs[i].data(), &len,
                                          pubkeys[i], SECP256K1_EC_COMPRESSED);
            idx[i] = i;
        }
        std::stable_sort(idx.begin(), idx.begin() + n_pubkeys,
            [&bufs](size_t a, size_t b) {
                return std::memcmp(bufs[a].data(), bufs[b].data(), 33) < 0;
            });
        for (size_t i = 0; i < n_pubkeys; ++i) sorted[i] = pubkeys[idx[i]];
        for (size_t i = 0; i < n_pubkeys; ++i) pubkeys[i] = sorted[i];
    } else {
        // Heap path for N>16.
        std::vector<std::array<unsigned char, 33>> bufs(n_pubkeys);
        for (size_t i = 0; i < n_pubkeys; ++i) {
            bufs[i] = {};  // zero-initialize: well-defined memcmp even on serialize failure
            size_t len = 33;
            secp256k1_ec_pubkey_serialize(secp256k1_context_static, bufs[i].data(), &len,
                                          pubkeys[i], SECP256K1_EC_COMPRESSED);
        }

        std::vector<size_t> idx(n_pubkeys);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(),
            [&bufs](size_t a, size_t b) {
                return std::memcmp(bufs[a].data(), bufs[b].data(), 33) < 0;
            });

        std::vector<const secp256k1_pubkey*> sorted(n_pubkeys);
        for (size_t i = 0; i < n_pubkeys; ++i) sorted[i] = pubkeys[idx[i]];
        for (size_t i = 0; i < n_pubkeys; ++i) pubkeys[i] = sorted[i];
    }
}

} // extern "C"
