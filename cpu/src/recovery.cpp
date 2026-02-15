#include "secp256k1/recovery.hpp"
#include "secp256k1/sha256.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// ── Lift x-coordinate to curve point ─────────────────────────────────────────
// Given x, compute y such that y² = x³ + 7 (mod p).
// parity selects which square root: 0 = even y, 1 = odd y.
// Returns {Point, bool} where bool = true if point is valid.
static std::pair<Point, bool> lift_x(const FieldElement& x_fe, int parity) {
    // y² = x³ + 7
    auto x3 = x_fe.square() * x_fe;
    auto seven = FieldElement::from_uint64(7);
    auto y2 = x3 + seven;

    // sqrt(y2) = y2^((p+1)/4) because p ≡ 3 (mod 4)
    // (p+1)/4 = 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c
    auto exp = FieldElement::from_hex(
        "3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c");

    auto y = FieldElement::one();
    auto base = y2;
    auto exp_bytes = exp.to_bytes();

    for (int i = 0; i < 256; ++i) {
        y = y.square();
        int byte_idx = i / 8;
        int bit_idx = 7 - (i % 8);
        if ((exp_bytes[byte_idx] >> bit_idx) & 1) {
            y = y * base;
        }
    }

    // Verify: y² == y2
    auto y_check = y.square();
    if (y_check != y2) return {Point::infinity(), false};

    // Adjust parity
    auto y_bytes = y.to_bytes();
    bool y_is_odd = (y_bytes[31] & 1) != 0;
    if ((parity != 0) != y_is_odd) {
        y = FieldElement::zero() - y;
    }

    return {Point::from_affine(x_fe, y), true};
}

// ── secp256k1 order n ────────────────────────────────────────────────────────
// n = FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
static const std::array<uint8_t, 32> SECP256K1_ORDER_BYTES = {
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFF,
    0xFF,0xFF,0xFF,0xFF, 0xFF,0xFF,0xFF,0xFE,
    0xBA,0xAE,0xDC,0xE6, 0xAF,0x48,0xA0,0x3B,
    0xBF,0xD2,0x5E,0x8C, 0xD0,0x36,0x41,0x41
};

// ── Sign with Recovery ID ────────────────────────────────────────────────────

RecoverableSignature ecdsa_sign_recoverable(
    const std::array<uint8_t, 32>& msg_hash,
    const Scalar& private_key) {

    if (private_key.is_zero()) return {{Scalar::zero(), Scalar::zero()}, 0};

    auto z = Scalar::from_bytes(msg_hash);
    auto k = rfc6979_nonce(private_key, msg_hash);
    if (k.is_zero()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // R = k * G
    auto R = Point::generator().scalar_mul(k);
    if (R.is_infinity()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // r = R.x mod n
    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    if (r.is_zero()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // Determine recovery ID
    int recid = 0;

    // bit 0: parity of R.y
    auto R_uncomp = R.to_uncompressed();
    if ((R_uncomp[64] & 1) != 0) recid |= 1;

    // bit 1: whether R.x >= n (overflow)
    // Compare r_bytes (big-endian x-coordinate) with order
    // If x >= n then r = x - n, and we need recid bit 1
    bool overflow = false;
    for (int i = 0; i < 32; ++i) {
        if (r_bytes[i] < SECP256K1_ORDER_BYTES[i]) break;
        if (r_bytes[i] > SECP256K1_ORDER_BYTES[i]) { overflow = true; break; }
    }
    if (overflow) recid |= 2;

    // s = k⁻¹ * (z + r * d) mod n
    auto k_inv = k.inverse();
    auto s = k_inv * (z + r * private_key);
    if (s.is_zero()) return {{Scalar::zero(), Scalar::zero()}, 0};

    // Normalize to low-S (BIP-62)
    ECDSASignature sig{r, s};
    if (!sig.is_low_s()) {
        sig = sig.normalize();
        recid ^= 1; // Negating s flips y parity
    }

    return {sig, recid};
}

// ── Public Key Recovery ──────────────────────────────────────────────────────

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
        // R.x = r + n — need to add order to r as field element
        // This is extremely rare (~2^-128 probability)
        auto n_fe = FieldElement::from_bytes(SECP256K1_ORDER_BYTES);
        auto r_fe_val = FieldElement::from_bytes(r_bytes);
        rx_fe = r_fe_val + n_fe;
    } else {
        rx_fe = FieldElement::from_bytes(r_bytes);
    }

    // Step 2: Lift x to curve point R with correct y parity
    int y_parity = recid & 1;
    auto [R, valid] = lift_x(rx_fe, y_parity);
    if (!valid) return {Point::infinity(), false};

    // Step 3: Recover public key
    // Q = r⁻¹ * (s*R - z*G)
    auto z = Scalar::from_bytes(msg_hash);
    auto r_inv = sig.r.inverse();

    auto sR = R.scalar_mul(sig.s);
    auto zG = Point::generator().scalar_mul(z);
    auto neg_zG = zG.negate();
    auto sR_minus_zG = sR.add(neg_zG);

    auto Q = sR_minus_zG.scalar_mul(r_inv);

    if (Q.is_infinity()) return {Point::infinity(), false};

    return {Q, true};
}

// ── Compact Serialization ────────────────────────────────────────────────────

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

    uint8_t header = data[0];
    if (header < 27 || header > 34) return {{}, false};

    int recid = (header - 27) & 3;

    std::array<uint8_t, 32> r_bytes{}, s_bytes{};
    std::memcpy(r_bytes.data(), data.data() + 1, 32);
    std::memcpy(s_bytes.data(), data.data() + 33, 32);

    auto r = Scalar::from_bytes(r_bytes);
    auto s = Scalar::from_bytes(s_bytes);

    if (r.is_zero() || s.is_zero()) return {{}, false};

    return {{ECDSASignature{r, s}, recid}, true};
}

} // namespace secp256k1
