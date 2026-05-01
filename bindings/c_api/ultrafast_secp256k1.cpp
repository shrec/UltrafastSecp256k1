/* ============================================================================
 * UltrafastSecp256k1 -- C API Implementation
 * ============================================================================
 * Wraps the C++ UltrafastSecp256k1 library into a stable C ABI.
 * All functions convert between opaque byte arrays and internal C++ types.
 * ============================================================================ */

#ifndef ULTRAFAST_SECP256K1_BUILDING
#define ULTRAFAST_SECP256K1_BUILDING
#endif
#include "ultrafast_secp256k1.h"

#include <cstring>
#include <array>
#include <cstdint>
#include <string>

// UltrafastSecp256k1 C++ headers
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/ecdh.hpp"
#include "secp256k1/recovery.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/address.hpp"
#include "secp256k1/bip32.hpp"
#include "secp256k1/taproot.hpp"
#include "secp256k1/init.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"

using Scalar = secp256k1::fast::Scalar;
using Point  = secp256k1::fast::Point;

/* -- Helpers --------------------------------------------------------------- */

static inline Scalar scalar_from_bytes(const uint8_t b[32]) {
    std::array<uint8_t, 32> arr;
    std::memcpy(arr.data(), b, 32);
    return Scalar::from_bytes(arr);
}

static inline void scalar_to_bytes(const Scalar& s, uint8_t out[32]) {
    auto arr = s.to_bytes();
    std::memcpy(out, arr.data(), 32);
}

static inline Point point_from_compressed(const uint8_t pub[33]) {
    // Parse compressed: 0x02/0x03 prefix + 32-byte x
    // Recover y from x using curve equation y^2 = x^3 + 7
    // Strict: reject x >= p (no reduction)
    secp256k1::fast::FieldElement x;
    if (!secp256k1::fast::FieldElement::parse_bytes_strict(pub + 1, x))
        return Point::infinity();

    // y^2 = x^3 + 7
    auto x2 = x * x;
    auto x3 = x2 * x;
    auto seven = secp256k1::fast::FieldElement::from_uint64(7);
    auto y2 = x3 + seven;

    // sqrt via Fermat: y = y2^((p+1)/4)
    // secp256k1 p == 3 (mod 4), so (p+1)/4 = 0x3fffffffffffffffffffffffffffffffffffffffffffffffffffffffbfffff0c
    // We compute this using the addition chain: repeated squarings then multiplies.
    // pow((p+1)/4) = pow(2^254 - 2^30 - 2^4 - ... ) -- use simple binary approach:
    //   (p+1)/4 in binary is 254 bits; we use the standard secp256k1 square chain.
    //
    // Simple method: y = y2^((p+1)/4) using the identity:
    //   y2^((p-1)/2) = +/-1 (Euler criterion)
    //   y = y2^((p+1)/4)
    //
    // Compute via: e2 = y2^(2^2-1), e4 = y2^(2^4-1), etc. reusing inverse's chain.
    auto t = y2;
    // x2 = t^(2^1) * t  ->  t^3 = t^(2^2-1)
    auto a = t.square() * t;
    // x3 = a^(2^1) * t  ->  t^(2^3-1)
    auto b = a.square() * t;
    // x6 = b^(2^3) * b  ->  t^(2^6-1)
    auto c = b.square().square().square() * b;
    // x9 = c^(2^3) * b  ->  t^(2^9-1)
    auto d = c.square().square().square() * b;
    // x11 = d^(2^2) * a  ->  t^(2^11-1)
    auto e = d.square().square() * a;
    // x22 = e^(2^11) * e  ->  t^(2^22-1)
    auto f = e;
    for (int i = 0; i < 11; ++i) f = f.square();
    f = f * e;
    // x44 = f^(2^22) * f  ->  t^(2^44-1)
    auto g = f;
    for (int i = 0; i < 22; ++i) g = g.square();
    g = g * f;
    // x88 = g^(2^44) * g  ->  t^(2^88-1)
    auto h = g;
    for (int i = 0; i < 44; ++i) h = h.square();
    h = h * g;
    // x176 = h^(2^88) * h  ->  t^(2^176-1)
    auto j = h;
    for (int i = 0; i < 88; ++i) j = j.square();
    j = j * h;
    // x220 = j^(2^44) * g  ->  t^(2^220-1)
    auto k = j;
    for (int i = 0; i < 44; ++i) k = k.square();
    k = k * g;
    // x223 = k^(2^3) * b  ->  t^(2^223-1)
    auto m = k.square().square().square() * b;
    // result = m^(2^23) * e * t^(2^6) ...
    // Following secp256k1 (p+1)/4 addition chain:
    //   (p+1)/4 = 2^254 - 2^30 - 244  (approximately)
    //   = (2^223 - 1) * 2^31 + (2^22 + ... )
    // Exact exponent: shift m by 23 bits, multiply by specific window.
    auto y = m;
    for (int i = 0; i < 23; ++i) y = y.square();
    y = y * f; // f = t^(2^22-1)
    for (int i = 0; i < 6; ++i) y = y.square();
    y = y * a; // *t^(2^2-1) = *t^3
    y = y.square();
    y = y.square(); // two more squares to reach bit 0

    // Verify sqrt: y^2 must equal y2 (reject if x has no valid y on curve)
    if (y * y != y2) return Point::infinity();

    // Check parity -- 0x02 = even, 0x03 = odd
    auto y_bytes = y.to_bytes();
    bool y_is_odd = (y_bytes[31] & 1) != 0;
    bool want_odd = (pub[0] == 0x03);
    if (y_is_odd != want_odd) {
        y = secp256k1::fast::FieldElement::from_uint64(0) - y;
    }

    return Point::from_affine(x, y);
}

static inline void point_to_compressed(const Point& p, uint8_t out[33]) {
    auto comp = p.to_compressed();
    std::memcpy(out, comp.data(), 33);
}

/* -- Version --------------------------------------------------------------- */

const char* secp256k1_version(void) {
    return "1.0.0";
}

/* -- Library Lifecycle ----------------------------------------------------- */

int secp256k1_init(void) {
    return secp256k1::fast::ensure_library_integrity(false) ? 0 : 1;
}

/* -- Key Operations -------------------------------------------------------- */

int secp256k1_ec_pubkey_create(const uint8_t privkey[32], uint8_t pubkey_out[33]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(pubkey_out, 0, 33);
        return 1;
    }
    auto pk = secp256k1::ct::generator_mul(sk);
    if (pk.is_infinity()) {
        std::memset(pubkey_out, 0, 33);
        return 1;
    }
    point_to_compressed(pk, pubkey_out);
    return 0;
}

int secp256k1_ec_pubkey_create_uncompressed(const uint8_t privkey[32], uint8_t pubkey_out[65]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(pubkey_out, 0, 65);
        return 1;
    }
    auto pk = secp256k1::ct::generator_mul(sk);
    if (pk.is_infinity()) {
        std::memset(pubkey_out, 0, 65);
        return 1;
    }
    auto uncomp = pk.to_uncompressed();
    std::memcpy(pubkey_out, uncomp.data(), 65);
    return 0;
}

int secp256k1_ec_pubkey_parse(const uint8_t* input, size_t input_len, uint8_t pubkey_out[33]) {
    if (input_len == 33 && (input[0] == 0x02 || input[0] == 0x03)) {
        auto p = point_from_compressed(input);
        if (p.is_infinity()) return 1;
        point_to_compressed(p, pubkey_out);
        return 0;
    }
    if (input_len == 65 && input[0] == 0x04) {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), input + 1, 32);
        std::memcpy(y_bytes.data(), input + 33, 32);
        // Strict: reject x >= p or y >= p
        secp256k1::fast::FieldElement x, y;
        if (!secp256k1::fast::FieldElement::parse_bytes_strict(x_bytes, x) ||
            !secp256k1::fast::FieldElement::parse_bytes_strict(y_bytes, y)) return 1;
        // On-curve check: y^2 == x^3 + 7
        auto lhs = y * y;
        auto rhs = x * x * x + secp256k1::fast::FieldElement::from_uint64(7);
        if (lhs != rhs) return 1;
        auto p = Point::from_affine(x, y);
        if (p.is_infinity()) return 1;
        point_to_compressed(p, pubkey_out);
        return 0;
    }
    return 1;
}

int secp256k1_ec_seckey_verify(const uint8_t privkey[32]) {
    Scalar sk;
    return Scalar::parse_bytes_strict_nonzero(privkey, sk) ? 1 : 0;
}

int secp256k1_ec_privkey_negate(uint8_t privkey[32]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) return 0;
    auto neg = sk.negate();
    scalar_to_bytes(neg, privkey);
    return 1;  // success
}

int secp256k1_ec_privkey_tweak_add(uint8_t privkey[32], const uint8_t tweak[32]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) return 1;
    auto tw = scalar_from_bytes(tweak);
    auto result = sk + tw;
    if (result.is_zero()) return 1;
    scalar_to_bytes(result, privkey);
    return 0;
}

int secp256k1_ec_privkey_tweak_mul(uint8_t privkey[32], const uint8_t tweak[32]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) return 1;
    auto tw = scalar_from_bytes(tweak);
    auto result = sk * tw;
    if (result.is_zero()) return 1;
    scalar_to_bytes(result, privkey);
    return 0;
}

/* -- ECDSA ----------------------------------------------------------------- */

int secp256k1_ecdsa_sign(const uint8_t msg_hash[32], const uint8_t privkey[32],
                         uint8_t sig_out[64]) {
    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg_hash, 32);
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(sig_out, 0, 64);
        return 1;
    }

    auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
    if (!sig.is_valid()) {
        std::memset(sig_out, 0, 64);
        return 1;
    }
    sig.normalize();
    auto compact = sig.to_compact();
    std::memcpy(sig_out, compact.data(), 64);
    return 0;
}

int secp256k1_ecdsa_verify(const uint8_t msg_hash[32], const uint8_t sig[64],
                           const uint8_t pubkey[33]) {
    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg_hash, 32);

    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig, 64);
    auto ecdsasig = secp256k1::ECDSASignature::from_compact(compact);

    auto pk = point_from_compressed(pubkey);
    return secp256k1::ecdsa_verify(msg, pk, ecdsasig) ? 1 : 0;
}

int secp256k1_ecdsa_signature_serialize_der(const uint8_t sig[64],
                                            uint8_t* der_out, size_t* der_len) {
    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig, 64);
    auto ecdsasig = secp256k1::ECDSASignature::from_compact(compact);

    auto [der, actual_len] = ecdsasig.to_der();
    if (*der_len < actual_len) return 1;
    std::memcpy(der_out, der.data(), actual_len);
    *der_len = actual_len;
    return 0;
}

/* -- ECDSA Recovery -------------------------------------------------------- */

int secp256k1_ecdsa_sign_recoverable(const uint8_t msg_hash[32],
                                     const uint8_t privkey[32],
                                     uint8_t sig_out[64], int* recid_out) {
    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg_hash, 32);
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(sig_out, 0, 64);
        return 1;
    }

    auto rsig = secp256k1::ct::ecdsa_sign_recoverable(msg, sk);
    if (!rsig.sig.is_valid()) {
        std::memset(sig_out, 0, 64);
        return 1;
    }
    rsig.sig.normalize();
    auto compact = rsig.sig.to_compact();
    std::memcpy(sig_out, compact.data(), 64);
    *recid_out = rsig.recid;
    return 0;
}

int secp256k1_ecdsa_recover(const uint8_t msg_hash[32], const uint8_t sig[64],
                            int recid, uint8_t pubkey_out[33]) {
    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg_hash, 32);

    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig, 64);
    auto ecdsasig = secp256k1::ECDSASignature::from_compact(compact);

    auto [point, ok] = secp256k1::ecdsa_recover(msg, ecdsasig, recid);
    if (!ok) return 1;
    point_to_compressed(point, pubkey_out);
    return 0;
}

/* -- Schnorr (BIP-340) ---------------------------------------------------- */

int secp256k1_schnorr_sign(const uint8_t msg[32], const uint8_t privkey[32],
                           const uint8_t aux_rand[32], uint8_t sig_out[64]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(sig_out, 0, 64);
        return 1;
    }

    std::array<uint8_t, 32> msg_arr, aux_arr;
    std::memcpy(msg_arr.data(), msg, 32);
    std::memcpy(aux_arr.data(), aux_rand, 32);

    auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    auto sig = secp256k1::ct::schnorr_sign(kp, msg_arr, aux_arr);
    auto bytes = sig.to_bytes();

    // Check for degenerate Schnorr output: r and s must both be non-zero
    bool r_zero = true, s_zero = true;
    for (int i = 0; i < 32; i++) { if (bytes[i]    != 0) r_zero = false; }
    for (int i = 0; i < 32; i++) { if (bytes[i+32] != 0) s_zero = false; }
    if (r_zero || s_zero) {
        std::memset(sig_out, 0, 64);
        return 1;
    }
    std::memcpy(sig_out, bytes.data(), 64);
    return 0;
}

int secp256k1_schnorr_verify(const uint8_t msg[32], const uint8_t sig[64],
                             const uint8_t pubkey_x[32]) {
    std::array<uint8_t, 32> pk_arr, msg_arr;
    std::memcpy(pk_arr.data(), pubkey_x, 32);
    std::memcpy(msg_arr.data(), msg, 32);

    std::array<uint8_t, 64> sig_arr;
    std::memcpy(sig_arr.data(), sig, 64);
    auto schnorr_sig = secp256k1::SchnorrSignature::from_bytes(sig_arr);

    return secp256k1::schnorr_verify(pk_arr, msg_arr, schnorr_sig) ? 1 : 0;
}

int secp256k1_schnorr_pubkey(const uint8_t privkey[32], uint8_t pubkey_x_out[32]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(pubkey_x_out, 0, 32);
        return 1;
    }

    auto xonly = secp256k1::ct::schnorr_pubkey(sk);
    std::memcpy(pubkey_x_out, xonly.data(), 32);
    return 0;
}

/* -- ECDH ------------------------------------------------------------------ */

int secp256k1_ecdh(const uint8_t privkey[32], const uint8_t pubkey[33],
                   uint8_t secret_out[32]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(secret_out, 0, 32);
        return 1;
    }
    auto pk = point_from_compressed(pubkey);
    if (pk.is_infinity()) {
        std::memset(secret_out, 0, 32);
        return 1;
    }
    auto secret = secp256k1::ecdh_compute(sk, pk);
    std::memcpy(secret_out, secret.data(), 32);
    return 0;
}

int secp256k1_ecdh_xonly(const uint8_t privkey[32], const uint8_t pubkey[33],
                         uint8_t secret_out[32]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(secret_out, 0, 32);
        return 1;
    }
    auto pk = point_from_compressed(pubkey);
    if (pk.is_infinity()) {
        std::memset(secret_out, 0, 32);
        return 1;
    }
    auto secret = secp256k1::ecdh_compute_xonly(sk, pk);
    std::memcpy(secret_out, secret.data(), 32);
    return 0;
}

int secp256k1_ecdh_raw(const uint8_t privkey[32], const uint8_t pubkey[33],
                       uint8_t secret_out[32]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(secret_out, 0, 32);
        return 1;
    }
    auto pk = point_from_compressed(pubkey);
    if (pk.is_infinity()) {
        std::memset(secret_out, 0, 32);
        return 1;
    }
    auto secret = secp256k1::ecdh_compute_raw(sk, pk);
    std::memcpy(secret_out, secret.data(), 32);
    return 0;
}

/* -- Hashing --------------------------------------------------------------- */

void secp256k1_sha256(const uint8_t* data, size_t data_len, uint8_t digest_out[32]) {
    secp256k1::SHA256 hasher;
    hasher.update(data, data_len);
    auto digest = hasher.finalize();
    std::memcpy(digest_out, digest.data(), 32);
}

void secp256k1_hash160(const uint8_t* data, size_t data_len, uint8_t digest_out[20]) {
    auto h = secp256k1::hash160(data, data_len);
    std::memcpy(digest_out, h.data(), 20);
}

void secp256k1_tagged_hash(const char* tag, const uint8_t* data, size_t data_len,
                           uint8_t digest_out[32]) {
    auto h = secp256k1::tagged_hash(tag, data, data_len);
    std::memcpy(digest_out, h.data(), 32);
}

/* -- Bitcoin Addresses ----------------------------------------------------- */

static secp256k1::Network to_network(int n) {
    return n == SECP256K1_NETWORK_TESTNET ? secp256k1::Network::Testnet
                                          : secp256k1::Network::Mainnet;
}

int secp256k1_address_p2pkh(const uint8_t pubkey[33], int network,
                            char* addr_out, size_t* addr_len) {
    auto pk = point_from_compressed(pubkey);
    auto addr = secp256k1::address_p2pkh(pk, to_network(network));
    if (addr.empty()) return 1;
    if (*addr_len < addr.size() + 1) return 1;
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return 0;
}

int secp256k1_address_p2wpkh(const uint8_t pubkey[33], int network,
                             char* addr_out, size_t* addr_len) {
    auto pk = point_from_compressed(pubkey);
    auto addr = secp256k1::address_p2wpkh(pk, to_network(network));
    if (addr.empty()) return 1;
    if (*addr_len < addr.size() + 1) return 1;
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return 0;
}

int secp256k1_address_p2tr(const uint8_t internal_key_x[32], int network,
                           char* addr_out, size_t* addr_len) {
    std::array<uint8_t, 32> key_x;
    std::memcpy(key_x.data(), internal_key_x, 32);
    auto addr = secp256k1::address_p2tr_raw(key_x, to_network(network));
    if (addr.empty()) return 1;
    if (*addr_len < addr.size() + 1) return 1;
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return 0;
}

/* -- WIF ------------------------------------------------------------------- */

int secp256k1_wif_encode(const uint8_t privkey[32], int compressed, int network,
                         char* wif_out, size_t* wif_len) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) return 1;
    auto wif = secp256k1::wif_encode(sk, compressed != 0, to_network(network));
    if (wif.empty()) return 1;
    if (*wif_len < wif.size() + 1) return 1;
    std::memcpy(wif_out, wif.c_str(), wif.size() + 1);
    *wif_len = wif.size();
    return 0;
}

int secp256k1_wif_decode(const char* wif, uint8_t privkey_out[32],
                         int* compressed_out, int* network_out) {
    auto result = secp256k1::wif_decode(std::string(wif));
    if (!result.valid) return 1;
    scalar_to_bytes(result.key, privkey_out);
    *compressed_out = result.compressed ? 1 : 0;
    *network_out = result.network == secp256k1::Network::Testnet
                       ? SECP256K1_NETWORK_TESTNET
                       : SECP256K1_NETWORK_MAINNET;
    return 0;
}

/* -- BIP-32 ---------------------------------------------------------------- */

static void extkey_to_c(const secp256k1::ExtendedKey& ek, secp256k1_bip32_key* out) {
    auto serialized = ek.serialize();
    std::memcpy(out->data, serialized.data(), 78);
    out->is_private = ek.is_private ? 1 : 0;
}

static secp256k1::ExtendedKey extkey_from_c(const secp256k1_bip32_key* k) {
    // Reconstruct ExtendedKey from serialized 78-byte form
    // The BIP-32 serialized format:
    //   4 bytes version | 1 byte depth | 4 bytes fingerprint |
    //   4 bytes child_number | 32 bytes chain_code | 33 bytes key
    secp256k1::ExtendedKey ek{};
    // Parse from serialized data
    ek.depth = k->data[4];
    std::memcpy(ek.parent_fingerprint.data(), k->data + 5, 4);
    ek.child_number = (uint32_t(k->data[9]) << 24) | (uint32_t(k->data[10]) << 16) |
                      (uint32_t(k->data[11]) << 8) | uint32_t(k->data[12]);
    std::memcpy(ek.chain_code.data(), k->data + 13, 32);
    if (k->is_private) {
        // Private: 0x00 || 32-byte key
        std::memcpy(ek.key.data(), k->data + 46, 32);
        ek.is_private = true;
    } else {
        // Public: 33-byte compressed pubkey -- store x-coordinate
        std::memcpy(ek.key.data(), k->data + 46, 32);
        ek.is_private = false;
    }
    return ek;
}

int secp256k1_bip32_master_key(const uint8_t* seed, size_t seed_len,
                                secp256k1_bip32_key* key_out) {
    auto [ek, ok] = secp256k1::bip32_master_key(seed, seed_len);
    if (!ok) return 1;
    extkey_to_c(ek, key_out);
    return 0;
}

int secp256k1_bip32_derive_child(const secp256k1_bip32_key* parent, uint32_t index,
                                  secp256k1_bip32_key* child_out) {
    auto ek = extkey_from_c(parent);
    auto [child, ok] = ek.derive_child(index);
    if (!ok) return 1;
    extkey_to_c(child, child_out);
    return 0;
}

int secp256k1_bip32_derive_path(const secp256k1_bip32_key* master, const char* path,
                                 secp256k1_bip32_key* key_out) {
    auto ek = extkey_from_c(master);
    auto [derived, ok] = secp256k1::bip32_derive_path(ek, std::string(path));
    if (!ok) return 1;
    extkey_to_c(derived, key_out);
    return 0;
}

int secp256k1_bip32_get_privkey(const secp256k1_bip32_key* key, uint8_t privkey_out[32]) {
    if (!key->is_private) return 1;
    auto ek = extkey_from_c(key);
    auto sk = ek.private_key();
    scalar_to_bytes(sk, privkey_out);
    return 0;
}

int secp256k1_bip32_get_pubkey(const secp256k1_bip32_key* key, uint8_t pubkey_out[33]) {
    auto ek = extkey_from_c(key);
    auto pk = ek.public_key();
    point_to_compressed(pk, pubkey_out);
    return 0;
}

/* -- Taproot --------------------------------------------------------------- */

int secp256k1_taproot_output_key(const uint8_t internal_key_x[32],
                                 const uint8_t* merkle_root,
                                 uint8_t output_key_x_out[32], int* parity_out) {
    std::array<uint8_t, 32> ik;
    std::memcpy(ik.data(), internal_key_x, 32);

    size_t mr_len = merkle_root ? 32 : 0;
    auto [ok_x, parity] = secp256k1::taproot_output_key(ik, merkle_root, mr_len);
    std::memcpy(output_key_x_out, ok_x.data(), 32);
    *parity_out = parity;
    return 0;
}

int secp256k1_taproot_tweak_privkey(const uint8_t privkey[32],
                                    const uint8_t* merkle_root,
                                    uint8_t tweaked_privkey_out[32]) {
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        std::memset(tweaked_privkey_out, 0, 32);
        return 1;
    }
    size_t mr_len = merkle_root ? 32 : 0;
    auto tweaked = secp256k1::taproot_tweak_privkey(sk, merkle_root, mr_len);
    if (tweaked.is_zero()) {
        std::memset(tweaked_privkey_out, 0, 32);
        return 1;
    }
    scalar_to_bytes(tweaked, tweaked_privkey_out);
    return 0;
}

int secp256k1_taproot_verify_commitment(const uint8_t output_key_x[32],
                                        int output_key_parity,
                                        const uint8_t internal_key_x[32],
                                        const uint8_t* merkle_root,
                                        size_t merkle_root_len) {
    std::array<uint8_t, 32> ok_x, ik_x;
    std::memcpy(ok_x.data(), output_key_x, 32);
    std::memcpy(ik_x.data(), internal_key_x, 32);
    return secp256k1::taproot_verify_commitment(ok_x, output_key_parity, ik_x,
                                                merkle_root, merkle_root_len)
               ? 1
               : 0;
}
