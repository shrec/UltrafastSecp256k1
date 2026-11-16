// ============================================================================
// shim_schnorr_bch.cpp -- Bitcoin Cash Node legacy Schnorr (secp256k1_schnorr_*)
// ============================================================================
// Implements the BCHN secp256k1 Schnorr API. This is NOT BIP-340. The
// authoritative construction is the 2019-05-15 BCH upgrade specification,
// https://upgradespecs.bitcoincashnode.org/2019-05-15-schnorr/ :
//
//   nonce k  = RFC6979(seckey, msg32, algo16 = "Schnorr+SHA256  ")
//   R        = k * G                          (CT blinded generator mul)
//   if Jacobi(R.y) != 1: k = -k               (R.x is unchanged by this)
//   P        = seckey * G                     (CT blinded generator mul)
//   e_bytes  = SHA256(R.x[32] || P_comp[33] || msg32[32])
//   e        = Scalar::from_bytes(e_bytes)
//   s        = k + e * seckey  (mod n)
//   sig[64]  = R.x[32] || s.to_bytes()[32]
//
// Verify:
//   R_check  = s*G - e*P                      (fast GLV for variable-base)
//   fail     if R_check is infinity
//   fail     if Jacobi(R_check.y) != 1        (spec verification step 10)
//   valid    iff R_check.x == sig.r
//
// The Jacobi condition and the algo16 tag are both load-bearing and both were
// missing until 2026-09-14. Without the signer-side negation half of every
// signature this shim produced had a non-residue R.y, which BCHN and Libauth
// reject; without the verifier-side check this shim accepted both R and -R for
// the same r, which they also reject. Without the tag the nonces -- and so the
// signature bytes -- differ from BCHN and Libauth for the same key and message,
// which is valid but not interoperable with their deterministic output.
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
#include "secp256k1/bch_schnorr.hpp"

#include <cstdint>
#include <vector>

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

// The point plus the 33 bytes the BCH challenge hashes: prefix from the Y
// parity in data[63] (layout [x:32][y:32], big-endian), then the x bytes as
// stored -- exactly the bytes bch_schnorr_e() hashes for the signer.
static Point pubkey_data_to_compressed(const unsigned char data[64],
                                       std::uint8_t out33[33]) noexcept {
    out33[0] = (data[63] & 1u) ? 0x03 : 0x02;
    std::memcpy(out33 + 1, data, 32);
    return pubkey_data_to_point(data);
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

        // Nonce via RFC6979 with the BCH Schnorr domain separator. The spec
        // requires the 16-byte ASCII tag "Schnorr+SHA256  " (two trailing
        // 0x20 spaces) in the HMAC-DRBG keydata; that is what makes a BCH
        // Schnorr nonce a different stream from the ECDSA nonce for the same
        // key and message, and what makes our output byte-identical to BCHN
        // and to Libauth rather than merely valid.
        static constexpr uint8_t kBchSchnorrAlgo16[16] = {
            'S','c','h','n','o','r','r','+','S','H','A','2','5','6',' ',' '
        };
        auto k = secp256k1::rfc6979_nonce_libsecp_compat(d, msg, nullptr,
                                                          kBchSchnorrAlgo16);
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

        // BCH 2019 requires Jacobi(R.y) == 1. The signer satisfies it by
        // negating the nonce when R.y is not a quadratic residue: R = kG and
        // -R = (-k)G share an X coordinate, so `rx` above is already final and
        // only k moves.
        //
        // p == 3 (mod 4), so sqrt(y)^2 == y exactly when y is a residue -- the
        // Jacobi test, using the square root the field already provides.
        //
        // R is public: R.x IS the signature's r, and any observer can lift_x(r)
        // and recompute this same bit. The negation is nevertheless constant
        // time, because k is secret and this sits in the middle of its live
        // range; the mask is arithmetic rather than a branch for the same
        // reason.
        {
            FieldElement const ry = R.y();
            FieldElement const rt = ry.sqrt();
            bool const y_is_qr = (rt.square() == ry);
            // y_is_qr -> 0 (keep k); !y_is_qr -> ~0 (negate k)
            std::uint64_t const neg_mask =
                static_cast<std::uint64_t>(y_is_qr) - 1ULL;
            k = secp256k1::ct::scalar_cneg(k, neg_mask);
        }

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

    // Delegates to the engine's BCH verifier (src/cpu/src/bch_schnorr.cpp):
    // the same rule this function used to inline -- strict s (s >= n and
    // s == 0 rejected), P != O, e = SHA256(r || P_compressed || msg),
    // R = s*G - e*P != O, Jacobi(R.y) == 1, x(R) == r -- evaluated without an
    // inversion or a square root. Verdicts are identical; pinned by
    // audit/test_bch_schnorr_differential.cpp against a copy of the old body.
    // Public-data path, variable-time by design (CLAUDE.md CT-VERIFY).
    try {
        std::uint8_t pub33[33];
        Point const P = pubkey_data_to_compressed(pubkey->data, pub33);
        return secp256k1::bch_schnorr_verify(sig64, msg32, P, pub33) ? 1 : 0;
    } catch (const std::exception&) { return 0; }
      catch (...) { std::terminate(); }
}

int secp256k1_schnorr_verify_batch(
    const secp256k1_context* ctx,
    const unsigned char* const* sig64s,
    const unsigned char* const* msg32s,
    const secp256k1_pubkey* const* pubkeys,
    size_t n,
    const unsigned char* seed32)
{
    if (!ctx) return 0;  // fail-closed: NULL context is invalid
    if (n == 0) return 1;  // empty batch: vacuously valid
    if (!sig64s || !msg32s || !pubkeys) return 0;

    try {
        std::vector<secp256k1::BchSchnorrBatchPointEntry> batch(n);
        for (size_t i = 0; i < n; ++i) {
            if (!sig64s[i] || !msg32s[i] || !pubkeys[i]) return 0;
            batch[i].point = pubkey_data_to_compressed(pubkeys[i]->data,
                                                       batch[i].pubkey33.data());
            std::memcpy(batch[i].message.data(), msg32s[i], 32);
            std::memcpy(batch[i].signature.data(), sig64s[i], 64);
        }
        // Verdict == AND of secp256k1_schnorr_verify over the entries; seed32
        // is extra entropy only, mixed in on top of the engine's CSPRNG bytes.
        return secp256k1::bch_schnorr_batch_verify(batch.data(), n, seed32) ? 1 : 0;
    } catch (const std::exception&) { return 0; }
      catch (...) { std::terminate(); }
}

} // extern "C"
