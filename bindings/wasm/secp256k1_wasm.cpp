// ============================================================================
// UltrafastSecp256k1 -- WebAssembly C API Implementation
// ============================================================================

#include "secp256k1_wasm.h"

#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/selftest.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/sha256.hpp"

#include <array>
#include <cstring>

// -- Helpers ------------------------------------------------------------------

namespace {

using Bytes32 = std::array<std::uint8_t, 32>;

inline Bytes32 to_array(const uint8_t* p) {
    Bytes32 a;
    std::memcpy(a.data(), p, 32);
    return a;
}

inline void write_point(const secp256k1::fast::Point& pt,
                         uint8_t* out_x32, uint8_t* out_y32) {
    auto xb = pt.x().to_bytes();
    auto yb = pt.y().to_bytes();
    std::memcpy(out_x32, xb.data(), 32);
    std::memcpy(out_y32, yb.data(), 32);
}

} // namespace

// -- Implementation -----------------------------------------------------------

extern "C" {

int secp256k1_wasm_selftest(void) {
    return secp256k1::fast::Selftest(false) ? 1 : 0;
}

const char* secp256k1_wasm_version(void) {
    // Matches PROJECT_VERSION in CMakeLists.txt
    return "3.0.0";
}

int secp256k1_wasm_pubkey_create(const uint8_t* seckey32,
                                  uint8_t* pubkey_x32,
                                  uint8_t* pubkey_y32) {
    auto sk = secp256k1::fast::Scalar::from_bytes(to_array(seckey32));
    if (sk.is_zero()) return 0;

    auto G  = secp256k1::fast::Point::generator();
    auto P  = G.scalar_mul(sk);
    if (P.is_infinity()) return 0;

    write_point(P, pubkey_x32, pubkey_y32);
    return 1;
}

int secp256k1_wasm_point_mul(const uint8_t* point_x32,
                              const uint8_t* point_y32,
                              const uint8_t* scalar32,
                              uint8_t* out_x32,
                              uint8_t* out_y32) {
    auto px = secp256k1::fast::FieldElement::from_bytes(to_array(point_x32));
    auto py = secp256k1::fast::FieldElement::from_bytes(to_array(point_y32));
    auto s  = secp256k1::fast::Scalar::from_bytes(to_array(scalar32));

    if (s.is_zero()) return 0;

    auto P = secp256k1::fast::Point::from_affine(px, py);
    auto R = P.scalar_mul(s);
    if (R.is_infinity()) return 0;

    write_point(R, out_x32, out_y32);
    return 1;
}

int secp256k1_wasm_point_add(const uint8_t* p_x32, const uint8_t* p_y32,
                              const uint8_t* q_x32, const uint8_t* q_y32,
                              uint8_t* out_x32, uint8_t* out_y32) {
    auto px = secp256k1::fast::FieldElement::from_bytes(to_array(p_x32));
    auto py = secp256k1::fast::FieldElement::from_bytes(to_array(p_y32));
    auto qx = secp256k1::fast::FieldElement::from_bytes(to_array(q_x32));
    auto qy = secp256k1::fast::FieldElement::from_bytes(to_array(q_y32));

    auto P = secp256k1::fast::Point::from_affine(px, py);
    auto Q = secp256k1::fast::Point::from_affine(qx, qy);
    auto R = P.add(Q);

    if (R.is_infinity()) return 0;

    write_point(R, out_x32, out_y32);
    return 1;
}

int secp256k1_wasm_ecdsa_sign(const uint8_t* msg32,
                               const uint8_t* seckey32,
                               uint8_t* sig64) {
    auto msg  = to_array(msg32);
    auto sk   = secp256k1::fast::Scalar::from_bytes(to_array(seckey32));
    if (sk.is_zero()) return 0;

    auto signature = secp256k1::ecdsa_sign(msg, sk);
    if (signature.r.is_zero()) return 0;

    auto compact = signature.to_compact();
    std::memcpy(sig64, compact.data(), 64);
    return 1;
}

int secp256k1_wasm_ecdsa_verify(const uint8_t* msg32,
                                 const uint8_t* pubkey_x32,
                                 const uint8_t* pubkey_y32,
                                 const uint8_t* sig64) {
    auto msg = to_array(msg32);
    auto px  = secp256k1::fast::FieldElement::from_bytes(to_array(pubkey_x32));
    auto py  = secp256k1::fast::FieldElement::from_bytes(to_array(pubkey_y32));
    auto P   = secp256k1::fast::Point::from_affine(px, py);

    std::array<std::uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);
    auto sig = secp256k1::ECDSASignature::from_compact(compact);

    return secp256k1::ecdsa_verify(msg, P, sig) ? 1 : 0;
}

int secp256k1_wasm_schnorr_sign(const uint8_t* seckey32,
                                 const uint8_t* msg32,
                                 const uint8_t* aux32,
                                 uint8_t* sig64) {
    auto sk  = secp256k1::fast::Scalar::from_bytes(to_array(seckey32));
    auto msg = to_array(msg32);
    auto aux = to_array(aux32);

    if (sk.is_zero()) return 0;

    auto signature = secp256k1::schnorr_sign(sk, msg, aux);
    auto bytes = signature.to_bytes();
    std::memcpy(sig64, bytes.data(), 64);
    return 1;
}

int secp256k1_wasm_schnorr_verify(const uint8_t* pubkey_x32,
                                   const uint8_t* msg32,
                                   const uint8_t* sig64) {
    auto pk  = to_array(pubkey_x32);
    auto msg = to_array(msg32);

    std::array<std::uint8_t, 64> sig_bytes;
    std::memcpy(sig_bytes.data(), sig64, 64);
    auto sig = secp256k1::SchnorrSignature::from_bytes(sig_bytes);

    return secp256k1::schnorr_verify(pk, msg, sig) ? 1 : 0;
}

int secp256k1_wasm_schnorr_pubkey(const uint8_t* seckey32,
                                   uint8_t* pubkey_x32) {
    auto sk = secp256k1::fast::Scalar::from_bytes(to_array(seckey32));
    if (sk.is_zero()) return 0;

    auto xonly = secp256k1::schnorr_pubkey(sk);
    std::memcpy(pubkey_x32, xonly.data(), 32);
    return 1;
}

void secp256k1_wasm_sha256(const uint8_t* data, size_t len, uint8_t* out32) {
    secp256k1::SHA256 sha;
    sha.update(data, len);
    auto digest = sha.finalize();
    std::memcpy(out32, digest.data(), 32);
}

} // extern "C"
