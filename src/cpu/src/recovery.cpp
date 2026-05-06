#include "secp256k1/recovery.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// -- Lift x-coordinate to curve point -----------------------------------------
// Given x, compute y such that y^2 = x^3 + 7 (mod p).
// parity selects which square root: 0 = even y, 1 = odd y.
// Returns {Point, bool} where bool = true if point is valid.
static std::pair<Point, bool> lift_x(const FieldElement& x_fe, int parity) {
    // y^2 = x^3 + 7
    auto x3 = x_fe.square() * x_fe;
    auto y2 = x3 + FieldElement::from_uint64(7);

    // Optimized sqrt via addition chain
    auto y = y2.sqrt();

    // Verify: y^2 == y2
    if (y.square() != y2) return {Point::infinity(), false};

    // Adjust parity -- check LSB directly from normalized limbs (avoids
    // expensive to_bytes() serialization just for one parity bit).
    // FE64 limbs are always fully reduced, so limbs()[0] & 1 == value mod 2.
    bool const y_is_odd = (y.limbs()[0] & 1) != 0;
    if ((parity != 0) != y_is_odd) {
        y = FieldElement::zero() - y;
    }

    return {Point::from_affine(x_fe, y), true};
}

// -- secp256k1 order n --------------------------------------------------------
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
static const std::array<uint8_t, 32> SECP256K1_ORDER_BYTES = {
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
};

static inline Point signing_generator_mul(const Scalar& scalar) {
    return ct::generator_mul_blinded(scalar);
}

// -- Sign with Recovery ID ----------------------------------------------------
// CT path: uses ct::generator_mul_blinded(k), ct::scalar_inverse(k),
// ct::scalar_mul, and ct::scalar_add for all secret-bearing arithmetic.
// Recovery ID computation branches only on public data (r, r_bytes).

RecoverableSignature ecdsa_sign_recoverable(
    const std::array<uint8_t, 32>& msg_hash,
    const Scalar& private_key) {

    if (private_key.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    auto z = Scalar::from_bytes(msg_hash);
    auto k = rfc6979_nonce(private_key, msg_hash);
    if (k.is_zero_ct()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // R = k * G  (k is non-zero by check above; kG is never infinity)
    auto R = signing_generator_mul(k);

    // r = R.x mod n
    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    if (r.is_zero()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // Determine recovery ID
    int recid = 0;

    // bit 0: parity of R.y — R is already normalized by R.x() above.
    // Avoid to_uncompressed() (65-byte array + 64 field muls): read parity
    // directly from the affine y limbs. limbs()[0] is the least-significant
    // 64-bit word; its LSB is the parity of the integer value mod p.
    recid |= static_cast<int>(R.y().limbs()[0] & 1u);

    // bit 1: whether R.x >= n (r overflowed past n); branchless comparison on r_bytes (public data)
    unsigned gt = 0u, eq_run = 1u;
    for (int oi = 0; oi < 32; ++oi) {
        unsigned const rb = static_cast<unsigned>(r_bytes[static_cast<unsigned>(oi)]);
        unsigned const ob = static_cast<unsigned>(SECP256K1_ORDER_BYTES[static_cast<unsigned>(oi)]);
        unsigned const byte_gt = ((ob - rb) >> 31) & 1u;
        unsigned const byte_lt = ((rb - ob) >> 31) & 1u;
        gt     = gt | (eq_run & byte_gt);
        eq_run = eq_run & (1u - byte_gt) & (1u - byte_lt);
    }
    recid |= (int)(gt << 1);

    // s = k^-1 * (z + r * d) mod n
    // All three multiplications and the addition use CT primitives — fast::Scalar
    // operator* has secret-dependent branches (V7-01 audit finding).
    auto k_inv      = ct::scalar_inverse(k);
    auto r_times_d  = ct::scalar_mul(r, private_key);
    auto z_plus_rd  = ct::scalar_add(z, r_times_d);
    auto s          = ct::scalar_mul(k_inv, z_plus_rd);
    if (s.is_zero()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // Normalize to low-S (BIP-62): CT path — no branch on secret s
    ECDSASignature sig{r, s};
    std::uint64_t const s_was_high = ct::scalar_is_high(s);
    sig = ct::ct_normalize_low_s(sig);
    recid ^= static_cast<int>(s_was_high & 1u); // flip y parity if s was negated

    return {sig, recid};
}

// -- Public Key Recovery ------------------------------------------------------

std::pair<Point, bool> ecdsa_recover(
    const std::array<uint8_t, 32>& msg_hash,
    const ECDSASignature& sig,
    int recid) {

    if (recid < 0 || recid > 3) return {Point::infinity(), false};
    if (sig.r.is_zero() || sig.s.is_zero()) return {Point::infinity(), false};

    // Step 1: Reconstruct R.x
    // if recid bit 1 is set, R.x = r + n (the x-coordinate overflowed)
    auto r_bytes = sig.r.to_bytes();
    FieldElement rx_fe;

    if (recid & 2) {
        // R.x = r + n -- need to add order to r as field element
        // This is extremely rare (~2^-128 probability)
        auto n_fe = FieldElement::from_bytes(SECP256K1_ORDER_BYTES);
        auto r_fe_val = FieldElement::from_bytes(r_bytes);
        rx_fe = r_fe_val + n_fe;
    } else {
        rx_fe = FieldElement::from_bytes(r_bytes);
    }

    // Step 2: Lift x to curve point R with correct y parity
    int const y_parity = recid & 1;
    auto [R, valid] = lift_x(rx_fe, y_parity);
    if (!valid) return {Point::infinity(), false};

    // Step 3: Recover public key
    // Q = r^-1 * (s*R - z*G)
    //   = (s * r^-1) * R  +  (-z * r^-1) * G
    //   = u2 * R  +  u1 * G
    // This is exactly dual_scalar_mul_gen_point(u1, u2, R) which uses
    // 4-stream GLV Strauss with interleaved wNAF -- a single combined
    // multi-scalar multiplication instead of 3 separate scalar muls.
    auto z = Scalar::from_bytes(msg_hash);
    auto r_inv = sig.r.inverse();
    auto u1 = z.negate() * r_inv;    // -z * r^-1 mod n  (G coefficient)
    auto u2 = sig.s * r_inv;         //  s * r^-1 mod n  (R coefficient)

    auto Q = Point::dual_scalar_mul_gen_point(u1, u2, R);

    if (Q.is_infinity()) return {Point::infinity(), false};

    return {Q, true};
}

// -- Compact Serialization ----------------------------------------------------

std::array<uint8_t, 65> recoverable_to_compact(
    const RecoverableSignature& rsig,
    bool compressed) {

    std::array<uint8_t, 65> out{};
    out[0] = static_cast<uint8_t>(27 + rsig.recid + (compressed ? 4 : 0));

    auto r_bytes = rsig.sig.r.to_bytes();
    auto s_bytes = rsig.sig.s.to_bytes();
    std::memcpy(out.data() + 1, r_bytes.data(), 32);
    std::memcpy(out.data() + 33, s_bytes.data(), 32);

    return out;
}

std::pair<RecoverableSignature, bool> recoverable_from_compact(
    const std::array<uint8_t, 65>& data) {

    uint8_t const header = data[0];
    if (header < 27 || header > 34) return {{}, false};

    int const recid = (header - 27) & 3;

    std::array<uint8_t, 32> r_bytes{}, s_bytes{};
    std::memcpy(r_bytes.data(), data.data() + 1, 32);
    std::memcpy(s_bytes.data(), data.data() + 33, 32);

    Scalar r, s;
    if (!Scalar::parse_bytes_strict_nonzero(r_bytes, r)) return {{}, false};
    if (!Scalar::parse_bytes_strict_nonzero(s_bytes, s)) return {{}, false};

    return {{ECDSASignature{r, s}, recid}, true};
}

} // namespace secp256k1
