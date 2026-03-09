// ============================================================================
// ct_sign.cpp -- Constant-Time Signing Functions
// ============================================================================
// Drop-in CT replacements for ecdsa_sign() and schnorr_sign().
// Uses ct::generator_mul() (data-independent execution trace) for all
// point multiplications involving secret nonces or private keys.
// ============================================================================

#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/scalar.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/tagged_hash.hpp"
#include "secp256k1/config.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

namespace {
using secp256k1::detail::secure_erase;
} // anonymous namespace

namespace secp256k1::ct {

// ============================================================================
// CT ECDSA Sign
// ============================================================================
// Pure CT sign: no sign-then-verify countermeasure.
// Use ct::ecdsa_sign_verified() if fault attack resistance is needed.

ECDSASignature ecdsa_sign(const std::array<uint8_t, 32>& msg_hash,
                          const Scalar& private_key) {
    if (private_key.is_zero()) return {Scalar::zero(), Scalar::zero()};

    auto z = Scalar::from_bytes(msg_hash);

    // Deterministic nonce (RFC 6979)
    auto k = rfc6979_nonce(private_key, msg_hash);
    if (k.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // R = k * G  -- CT path
    auto R = ct::generator_mul(k);
    if (R.is_infinity()) return {Scalar::zero(), Scalar::zero()};

    // r = R.x mod n
    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    if (r.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // s = k^{-1} * (z + r * d) mod n
    // CT inverse: SafeGCD Bernstein-Yang divsteps-59, constant-time.
    auto k_inv = ct::scalar_inverse(k);
    auto s = k_inv * (z + r * private_key);
    if (s.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // CT low-S normalization: branchless comparison with n/2 + conditional negate.
    ECDSASignature const sig = ct::ct_normalize_low_s(ECDSASignature{r, s});

    return sig;
}

// ============================================================================
// CT ECDSA Sign + Verify (fault attack countermeasure)
// ============================================================================
// Signs and then verifies (FIPS 186-4 fault countermeasure).
// Verify uses fast path -- public key and signature are not secret.

ECDSASignature ecdsa_sign_verified(const std::array<uint8_t, 32>& msg_hash,
                                   const Scalar& private_key) {
    auto sig = ecdsa_sign(msg_hash, private_key);

    if (!sig.r.is_zero()) {
        auto pk = ct::generator_mul(private_key);
        if (!ecdsa_verify(msg_hash.data(), pk, sig)) {
            return {Scalar::zero(), Scalar::zero()};
        }
    }

    return sig;
}

// ============================================================================
// CT ECDSA Sign (hedged, with extra entropy)
// ============================================================================
// RFC 6979 Section 3.6: aux_rand mixed into HMAC-DRBG for defense-in-depth.
// Uses ct::generator_mul() for R = k*G (constant-time).

// Forward declaration (defined in ecdsa.cpp, declared in ecdsa.hpp)
// rfc6979_nonce_hedged is in namespace secp256k1 (already accessible via ecdsa.hpp include)

ECDSASignature ecdsa_sign_hedged(const std::array<uint8_t, 32>& msg_hash,
                                  const Scalar& private_key,
                                  const std::array<uint8_t, 32>& aux_rand) {
    if (private_key.is_zero()) return {Scalar::zero(), Scalar::zero()};

    auto z = Scalar::from_bytes(msg_hash);
    auto k = secp256k1::rfc6979_nonce_hedged(private_key, msg_hash, aux_rand);
    if (k.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // R = k * G  -- CT path
    auto R = ct::generator_mul(k);
    if (R.is_infinity()) return {Scalar::zero(), Scalar::zero()};

    auto r_fe = R.x();
    auto r_bytes = r_fe.to_bytes();
    auto r = Scalar::from_bytes(r_bytes);
    if (r.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // CT inverse: SafeGCD Bernstein-Yang divsteps-59, same as ecdsa_sign above.
    auto k_inv = ct::scalar_inverse(k);
    auto s = k_inv * (z + r * private_key);
    if (s.is_zero()) return {Scalar::zero(), Scalar::zero()};

    // CT low-S normalization (branchless)
    ECDSASignature const sig = ct::ct_normalize_low_s(ECDSASignature{r, s});

    return sig;
}

// ============================================================================
// CT ECDSA Sign Hedged + Verify (fault attack countermeasure)
// ============================================================================

ECDSASignature ecdsa_sign_hedged_verified(const std::array<uint8_t, 32>& msg_hash,
                                          const Scalar& private_key,
                                          const std::array<uint8_t, 32>& aux_rand) {
    auto sig = ecdsa_sign_hedged(msg_hash, private_key, aux_rand);

    if (!sig.r.is_zero()) {
        auto pk = ct::generator_mul(private_key);
        if (!ecdsa_verify(msg_hash.data(), pk, sig)) {
            return {Scalar::zero(), Scalar::zero()};
        }
    }

    return sig;
}

// ============================================================================
// CT Schnorr helpers
// ============================================================================

// -- Shared BIP-340 tagged-hash midstates (from tagged_hash.hpp) ---------------
using detail::g_aux_midstate;
using detail::g_nonce_midstate;
using detail::g_challenge_midstate;
using detail::cached_tagged_hash;

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

    // CT: conditional negate based on parity. p_y_odd is derived from the
    // secret key, so the ternary branch would leak via timing.
    kp.d = ct::scalar_cneg(d_prime, ct::bool_to_mask(p_y_odd));
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

    // Step 3: R = k' * G -- CT path
    auto R = ct::generator_mul(k_prime);
    auto [rx, r_y_odd] = R.x_bytes_and_parity();

    // Step 4: k = k' if even_y(R), else n - k'
    // CT: branchless conditional negate. r_y_odd is secret-derived (from k').
    auto k = ct::scalar_cneg(k_prime, ct::bool_to_mask(r_y_odd));

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

    // Erase ALL stack buffers that held secret-derived material:
    //   d_bytes[32]          -- private key serialized
    //   t_hash[32]           -- tagged_hash output (XOR'd with d_bytes)
    //   t[32]                -- d XOR t_hash (derived from private key)
    //   nonce_input[96]      -- t || pubkey_x || msg (contains t)
    //   rand_hash[32]        -- nonce hash output (determines k')
    //   challenge_input[96]  -- R.x || pubkey_x || msg (public but erase for hygiene)
    //   k_prime, k           -- secret nonce scalars
    secure_erase(d_bytes.data(), d_bytes.size());
    secure_erase(t_hash.data(), t_hash.size());
    secure_erase(t, sizeof(t));
    secure_erase(nonce_input, sizeof(nonce_input));
    secure_erase(rand_hash.data(), rand_hash.size());
    secure_erase(challenge_input, sizeof(challenge_input));
    // Erase secret nonce scalars. Scalar is a POD-like 32-byte struct (4x uint64_t).
    secure_erase(&k_prime, sizeof(k_prime));
    secure_erase(&k, sizeof(k));

    return sig;
}

// ============================================================================
// CT Schnorr Sign + Verify (fault attack countermeasure)
// ============================================================================
// Signs and then verifies (FIPS 186-4 fault countermeasure).
// Public key and signature are not secret -- fast verify is safe.

SchnorrSignature schnorr_sign_verified(const SchnorrKeypair& kp,
                                       const std::array<uint8_t, 32>& msg,
                                       const std::array<uint8_t, 32>& aux_rand) {
    auto sig = ct::schnorr_sign(kp, msg, aux_rand);

    if (sig.s.is_zero()) return SchnorrSignature{};

    if (!schnorr_verify(kp.px, msg, sig)) {
        return SchnorrSignature{};
    }

    return sig;
}

} // namespace secp256k1::ct
