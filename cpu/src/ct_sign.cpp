// ============================================================================
// ct_sign.cpp — Constant-Time Signing Functions
// ============================================================================
// Drop-in CT replacements for ecdsa_sign() and schnorr_sign().
// Uses ct::generator_mul() (data-independent execution trace) for all
// point multiplications involving secret nonces or private keys.
// ============================================================================

#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/config.hpp"
#include <cstring>

namespace secp256k1::ct {

using fast::Scalar;
using fast::Point;
using fast::FieldElement;

// ============================================================================
// CT ECDSA Sign
// ============================================================================

ECDSASignature ecdsa_sign(const std::array<uint8_t, 32>& msg_hash,
                          const Scalar& private_key) {
    if (private_key.is_zero()) return {Scalar::zero(), Scalar::zero()};

    auto z = Scalar::from_bytes(msg_hash);

    // Deterministic nonce (RFC 6979)
    auto k = rfc6979_nonce(private_key, msg_hash);
    if (k.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // R = k * G  — CT path
    auto R = ct::generator_mul(k);
    if (R.is_infinity()) return {Scalar::zero(), Scalar::zero()};

    // r = R.x mod n
    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    if (r.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // s = k^{-1} * (z + r * d) mod n
    auto k_inv = k.inverse();
    auto s = k_inv * (z + r * private_key);
    if (s.is_zero()) return {Scalar::zero(), Scalar::zero()};

    ECDSASignature sig{r, s};
    return sig.normalize();
}

// ============================================================================
// CT Schnorr helpers
// ============================================================================

// Cached BIP-340 tagged-hash midstates (identical to schnorr.cpp)
static SHA256 make_tag_midstate(const char* tag) {
    auto tag_hash = SHA256::hash(tag, std::strlen(tag));
    SHA256 ctx;
    ctx.update(tag_hash.data(), 32);
    ctx.update(tag_hash.data(), 32);
    return ctx;
}

static const SHA256 g_aux_midstate       = make_tag_midstate("BIP0340/aux");
static const SHA256 g_nonce_midstate     = make_tag_midstate("BIP0340/nonce");
static const SHA256 g_challenge_midstate = make_tag_midstate("BIP0340/challenge");

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"
#endif
static std::array<uint8_t, 32> cached_tagged_hash(const SHA256& midstate,
                                                    const void* data, std::size_t len) {
    SHA256 ctx = midstate;
    ctx.update(data, len);
    return ctx.finalize();
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

// ============================================================================
// CT Schnorr Pubkey
// ============================================================================

std::array<uint8_t, 32> schnorr_pubkey(const Scalar& private_key) {
    auto P = ct::generator_mul(private_key);
    auto [px, p_y_odd] = P.x_bytes_and_parity();
    (void)p_y_odd;
    return px;
}

// ============================================================================
// CT Schnorr Keypair Create
// ============================================================================

SchnorrKeypair schnorr_keypair_create(const Scalar& private_key) {
    SchnorrKeypair kp{};
    auto d_prime = private_key;
    if (d_prime.is_zero()) return kp;

    auto P = ct::generator_mul(d_prime);
    auto [px, p_y_odd] = P.x_bytes_and_parity();

    kp.d = p_y_odd ? d_prime.negate() : d_prime;
    kp.px = px;
    return kp;
}

// ============================================================================
// CT Schnorr Sign (BIP-340, keypair variant)
// ============================================================================

SchnorrSignature schnorr_sign(const SchnorrKeypair& kp,
                              const std::array<uint8_t, 32>& msg,
                              const std::array<uint8_t, 32>& aux_rand) {
    if (kp.d.is_zero()) return SchnorrSignature{};

    // Step 1: t = d XOR tagged_hash("BIP0340/aux", aux_rand)
    auto t_hash = cached_tagged_hash(g_aux_midstate, aux_rand.data(), 32);
    auto d_bytes = kp.d.to_bytes();
    uint8_t t[32];
    for (std::size_t i = 0; i < 32; ++i) t[i] = d_bytes[i] ^ t_hash[i];

    // Step 2: k' = tagged_hash("BIP0340/nonce", t || pubkey_x || msg)
    uint8_t nonce_input[96];
    std::memcpy(nonce_input, t, 32);
    std::memcpy(nonce_input + 32, kp.px.data(), 32);
    std::memcpy(nonce_input + 64, msg.data(), 32);
    auto rand_hash = cached_tagged_hash(g_nonce_midstate, nonce_input, 96);
    auto k_prime = Scalar::from_bytes(rand_hash);
    if (k_prime.is_zero()) return SchnorrSignature{};

    // Step 3: R = k' * G — CT path
    auto R = ct::generator_mul(k_prime);
    auto [rx, r_y_odd] = R.x_bytes_and_parity();

    // Step 4: k = k' if even_y(R), else n - k'
    auto k = r_y_odd ? k_prime.negate() : k_prime;

    // Step 5: e = tagged_hash("BIP0340/challenge", R.x || pubkey_x || msg)
    uint8_t challenge_input[96];
    std::memcpy(challenge_input, rx.data(), 32);
    std::memcpy(challenge_input + 32, kp.px.data(), 32);
    std::memcpy(challenge_input + 64, msg.data(), 32);
    auto e_hash = cached_tagged_hash(g_challenge_midstate, challenge_input, 96);
    auto e = Scalar::from_bytes(e_hash);

    // Step 6: sig = (R.x, k + e * d)
    SchnorrSignature sig{};
    sig.r = rx;
    sig.s = k + e * kp.d;
    return sig;
}

} // namespace secp256k1::ct
