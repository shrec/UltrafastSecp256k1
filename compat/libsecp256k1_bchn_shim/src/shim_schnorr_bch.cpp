// ============================================================================
// shim_schnorr_bch.cpp -- Bitcoin Cash Node legacy Schnorr (secp256k1_schnorr_*)
// ============================================================================
// Implements the BCHN secp256k1 Schnorr API. This is NOT BIP-340:
//
//   nonce k  = RFC6979(seckey, msg32)
//   R        = k * G                          (CT blinded generator mul)
//   P        = seckey * G                     (CT blinded generator mul)
//   e_bytes  = SHA256(R.x[32] || P_comp[33] || msg32[32])
//   e        = Scalar::from_bytes(e_bytes)
//   s        = k + e * seckey  (mod n)
//   sig[64]  = R.x[32] || s.to_bytes()[32]
//
// Verify:
//   R_check  = s*G - e*P                      (fast GLV for variable-base)
//   valid    iff R_check.x == sig.r
// ============================================================================

#include "secp256k1_schnorr.h"
#include "secp256k1.h"

#include <cstring>
#include <array>

#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/detail/secure_erase.hpp"

using namespace secp256k1::fast;

// -- Internal helpers ---------------------------------------------------------

// Trust contract (matches libsecp256k1): secp256k1_pubkey is populated only
// by secp256k1_ec_pubkey_parse / secp256k1_ec_pubkey_create, both of which
// validate y²=x³+7. We do not re-check curve membership here.
static Point pubkey_data_to_point(const unsigned char data[64]) noexcept {
    const auto& xb = *reinterpret_cast<const std::array<uint8_t, 32>*>(data);
    const auto& yb = *reinterpret_cast<const std::array<uint8_t, 32>*>(data + 32);
    FieldElement x = FieldElement::from_bytes(xb);
    FieldElement y = FieldElement::from_bytes(yb);
    return Point::from_affine(x, y);
}

// BCH Schnorr hash: SHA256(R.x[32] || P_compressed[33] || msg[32])
// px32: P's x-coordinate (32 bytes). y_is_odd: Y-parity bit (avoids to_uncompressed).
static Scalar bch_schnorr_e(const std::array<uint8_t,32>& rx,
                             const uint8_t* px32,
                             bool y_is_odd,
                             const std::array<uint8_t,32>& msg)
{
    uint8_t pcomp[33];
    pcomp[0] = y_is_odd ? 0x03 : 0x02;
    std::memcpy(pcomp + 1, px32, 32);

    secp256k1::SHA256 h;
    h.update(rx.data(), 32);
    h.update(pcomp, 33);
    h.update(msg.data(), 32);
    auto digest = h.finalize();

    return Scalar::from_bytes(digest);
}

extern "C" {

int secp256k1_schnorr_sign(
    const secp256k1_context* ctx,
    unsigned char* sig64,
    const unsigned char* msg32,
    const unsigned char* seckey,
    secp256k1_nonce_function /*noncefp*/,
    const void* /*ndata*/)
{
    if (!ctx) return 0;  // fail-closed: NULL context is invalid
    if (!sig64 || !msg32 || !seckey) return 0;
    std::memset(sig64, 0, 64);

    try {
        std::array<uint8_t, 32> kb{}, msg{};
        std::memcpy(kb.data(),  seckey, 32);
        std::memcpy(msg.data(), msg32,  32);

        // Fix 2a: strict parse — rejects >= n and zero, unlike from_bytes().
        Scalar d;
        if (!Scalar::parse_bytes_strict_nonzero(kb.data(), d)) {
            secp256k1::detail::secure_erase(kb.data(), 32);
            return 0;
        }

        // Nonce via RFC6979 (same path as ECDSA)
        auto k = secp256k1::rfc6979_nonce(d, msg);
        if (k.is_zero_ct()) {
            secp256k1::detail::secure_erase(kb.data(), 32);
            secp256k1::detail::secure_erase(&d, sizeof(d));
            return 0;
        }

        // Fix 2b: CT generator mul for R = k*G (nonce point — nonce is secret).
        auto R = secp256k1::ct::generator_mul_blinded(k);
        if (R.is_infinity()) {
            secp256k1::detail::secure_erase(kb.data(), 32);
            secp256k1::detail::secure_erase(&d, sizeof(d));
            secp256k1::detail::secure_erase(&k, sizeof(k));
            return 0;
        }
        auto rx = R.x().to_bytes();

        // Fix 2b: CT generator mul for P = d*G (private key is secret).
        auto P = secp256k1::ct::generator_mul_blinded(d);
        // Read Y-parity from affine limbs[0] LSB — avoids to_uncompressed() (~940ns).
        bool const p_y_odd = (P.y().limbs()[0] & 1u) != 0;
        auto px = P.x().to_bytes();

        // e = SHA256(R.x || P_compressed || msg)
        auto e = bch_schnorr_e(rx, px.data(), p_y_odd, msg);

        // CT arithmetic for s = k + e*d (k and d are secrets — use ct:: primitives).
        auto s = secp256k1::ct::scalar_add(k, secp256k1::ct::scalar_mul(e, d));
        // CT: s is nonce+privkey-derived — use is_zero_ct() not is_zero() (SEC-001 fix).
        if (s.is_zero_ct()) {
            secp256k1::detail::secure_erase(kb.data(), 32);
            secp256k1::detail::secure_erase(&d, sizeof(d));
            secp256k1::detail::secure_erase(&k, sizeof(k));
            return 0;
        }

        std::memcpy(sig64,      rx.data(),          32);
        auto sb = s.to_bytes();
        std::memcpy(sig64 + 32, sb.data(), 32);

        // Fix 2d: erase secret material from stack before return.
        secp256k1::detail::secure_erase(kb.data(), 32);
        secp256k1::detail::secure_erase(&d, sizeof(d));
        secp256k1::detail::secure_erase(&k, sizeof(k));

        return 1;
    } catch (const std::exception&) {
        std::memset(sig64, 0, 64);
        return 0;
    }
      catch (...) { std::terminate(); }
}

int secp256k1_schnorr_verify(
    const secp256k1_context* ctx,
    const unsigned char* sig64,
    const unsigned char* msg32,
    const secp256k1_pubkey* pubkey)
{
    if (!ctx) return 0;  // fail-closed: NULL context is invalid
    if (!sig64 || !msg32 || !pubkey) return 0;

    try {
        std::array<uint8_t, 32> rx{}, sb{}, msg{};
        std::memcpy(rx.data(),  sig64,      32);
        std::memcpy(sb.data(),  sig64 + 32, 32);
        std::memcpy(msg.data(), msg32,      32);

        // COMPAT-010: strict parse rejects s >= n (from_bytes silently reduces).
        // s is a public signature component — VT is_zero() is correct for verify.
        Scalar s;
        if (!Scalar::parse_bytes_strict(sb.data(), s) || s.is_zero()) return 0;

        auto P = pubkey_data_to_point(pubkey->data);
        if (P.is_infinity()) return 0;

        // Y-parity from pubkey->data[63] (last byte of Y, layout: [x:32][y:32]).
        // Avoids to_uncompressed() Jacobian→affine conversion (~940ns) for one bit.
        bool const p_y_odd = (pubkey->data[63] & 1u) != 0;

        // e = SHA256(R.x || P_compressed || msg)
        auto e = bch_schnorr_e(rx, pubkey->data, p_y_odd, msg);

        // R_check = s*G + (-e)*P  via Shamir's wNAF trick (SHIM-004 fix).
        // dual_scalar_mul_gen_point(a, b, Q) = a*G + b*Q using interleaved
        // wNAF — ~1.5× faster than two separate scalar_mul calls.
        auto neg_e   = e.negate();
        auto R_check = Point::dual_scalar_mul_gen_point(s, neg_e, P);
        if (R_check.is_infinity()) return 0;

        // Valid iff R_check.x == rx
        auto rx_check = R_check.x().to_bytes();
        return (std::memcmp(rx_check.data(), rx.data(), 32) == 0) ? 1 : 0;
    } catch (const std::exception&) { return 0; }
      catch (...) { std::terminate(); }
}

} // extern "C"
