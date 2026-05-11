// ============================================================================
// BIP-324: Version 2 P2P Encrypted Transport Protocol
// ============================================================================
// Session management, key derivation, and packet encrypt/decrypt.
//
// Key derivation (from BIP-324 spec):
//   1. ECDH: shared_secret = tagged_hash("bip324_ellswift_xonly_ecdh",
//            ell_a || ell_b || x(a * B))
//   2. Keys: PRK = HKDF-Extract(salt="bitcoin_v2_shared_secret", IKM=shared_secret)
//   3. send_key = HKDF-Expand(PRK, info="initiator_L" or "responder_L", 32)
//   4. recv_key = HKDF-Expand(PRK, info="responder_L" or "initiator_L", 32)
//   5. session_id = HKDF-Expand(PRK, info="session_id", 32)
//
// Packet format:
//   [3B encrypted length] [N bytes encrypted payload] [16B Poly1305 tag]
//   Length is encrypted with ChaCha20 (counter 0, first 3 bytes of keystream)
//   Payload+tag use AEAD (ChaCha20-Poly1305) with counter 0 and AAD = enc_length
// ============================================================================

#include "secp256k1/bip324.hpp"
#include "secp256k1/chacha20_poly1305.hpp"
#include "secp256k1/hkdf.hpp"
#include "secp256k1/ellswift.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

#include "secp256k1/detail/csprng.hpp"

namespace secp256k1 {

namespace {

using secp256k1::detail::csprng_fill;

} // anonymous namespace

// ============================================================================
// Bip324Cipher
// ============================================================================

void Bip324Cipher::init(const std::uint8_t* key) noexcept {
    std::memcpy(key_, key, 32);
    packet_counter_ = 0;
}

void Bip324Cipher::build_nonce(std::uint8_t* nonce) const noexcept {
    // Nonce = 4 zero bytes || 8-byte little-endian packet counter
    std::memset(nonce, 0, 4);
    for (int i = 0; i < 8; ++i) {
        nonce[4 + i] = static_cast<std::uint8_t>(packet_counter_ >> (i * 8));
    }
}

std::vector<std::uint8_t> Bip324Cipher::encrypt(
    const std::uint8_t* aad, std::size_t aad_len,
    const std::uint8_t* plaintext, std::size_t plaintext_len) noexcept {

    // BIP-324 length field is 3 bytes: reject plaintext > 0xFFFFFF
    if (plaintext_len > 0xFFFFFF) return {};

    // Overflow guard: 3 + plaintext_len + 16 must not wrap
    if (plaintext_len > SIZE_MAX - 19) return {};

    // Output: [3-byte encrypted length] [encrypted payload] [16-byte tag]
    std::size_t const ct_len = 3 + plaintext_len;
    std::vector<std::uint8_t> output(ct_len + 16);

    // Build combined plaintext [length(3)][payload(N)] directly in output
    output[0] = static_cast<std::uint8_t>(plaintext_len & 0xFF);
    output[1] = static_cast<std::uint8_t>((plaintext_len >> 8) & 0xFF);
    output[2] = static_cast<std::uint8_t>((plaintext_len >> 16) & 0xFF);
    if (plaintext_len > 0) {
        std::memcpy(output.data() + 3, plaintext, plaintext_len);
    }

    // Encrypt in place (AEAD supports aliased in/out)
    std::uint8_t nonce[12];
    build_nonce(nonce);

    aead_chacha20_poly1305_encrypt(
        key_, nonce,
        aad, aad_len,
        output.data(), ct_len,
        output.data(),
        output.data() + ct_len);

    packet_counter_++;
    return output;
}

bool Bip324Cipher::decrypt(
    const std::uint8_t* aad, std::size_t aad_len,
    const std::uint8_t* header_enc,
    const std::uint8_t* contents, std::size_t contents_len,
    std::vector<std::uint8_t>& plaintext_out) noexcept {

    plaintext_out.clear();

    if (contents_len < 16) return false;

    std::size_t const ct_len = 3 + (contents_len - 16);

    // Single allocation: reconstruct ciphertext and decrypt in place
    std::vector<std::uint8_t> buf(ct_len);
    std::memcpy(buf.data(), header_enc, 3);
    if (contents_len > 16) {
        std::memcpy(buf.data() + 3, contents, contents_len - 16);
    }

    const std::uint8_t* tag = contents + (contents_len - 16);

    std::uint8_t nonce[12];
    build_nonce(nonce);

    // Decrypt in place (AEAD supports aliased in/out)
    bool ok = aead_chacha20_poly1305_decrypt(
        key_, nonce,
        aad, aad_len,
        buf.data(), ct_len,
        tag,
        buf.data());

    if (!ok) return false;

    std::uint32_t const payload_len = static_cast<std::uint32_t>(buf[0])
                                    | (static_cast<std::uint32_t>(buf[1]) << 8)
                                    | (static_cast<std::uint32_t>(buf[2]) << 16);

    if (payload_len > ct_len - 3) return false;

    plaintext_out.assign(buf.begin() + 3, buf.begin() + 3 + payload_len);
    packet_counter_++;
    return true;
}

// ============================================================================
// Bip324Session
// ============================================================================

Bip324Session::Bip324Session(bool initiator) noexcept
    : initiator_(initiator) {
    // Generate ephemeral private key — retry until valid (≈ 2^-128 chance of retry)
    fast::Scalar sk;
    csprng_fill(privkey_.data(), 32);
    while (!fast::Scalar::parse_bytes_strict_nonzero(privkey_.data(), sk)) {
        csprng_fill(privkey_.data(), 32);
    }
    our_encoding_ = ellswift_create(sk);
    detail::secure_erase(&sk, sizeof(sk));
}

Bip324Session::Bip324Session(bool initiator, const std::uint8_t* privkey) noexcept
    : initiator_(initiator) {
    std::memcpy(privkey_.data(), privkey, 32);
    fast::Scalar sk;
    if (!fast::Scalar::parse_bytes_strict_nonzero(privkey_.data(), sk)) {
        // Invalid caller-supplied key: zero out and produce unusable session
        std::memset(privkey_.data(), 0, 32);
        our_encoding_ = {};
        return;
    }
    our_encoding_ = ellswift_create(sk);
    detail::secure_erase(&sk, sizeof(sk));
}

bool Bip324Session::complete_handshake(const std::uint8_t* peer_encoding) noexcept {
    if (established_) return false;

    std::memcpy(peer_encoding_.data(), peer_encoding, 64);

    fast::Scalar sk;
    if (!fast::Scalar::parse_bytes_strict_nonzero(privkey_.data(), sk)) {
        return false;  // invalid stored key (should not happen if constructors are used)
    }

    // Determine ell_a and ell_b (initiator = a, responder = b)
    const std::uint8_t* ell_a = initiator_ ? our_encoding_.data() : peer_encoding_.data();
    const std::uint8_t* ell_b = initiator_ ? peer_encoding_.data() : our_encoding_.data();

    // 1. ECDH via ElligatorSwift
    auto shared_secret = ellswift_xdh(ell_a, ell_b, sk, initiator_);

    // Check for failure (all zeros) — constant-time accumulator, no early exit.
    // An early-exit loop leaks timing about the number of leading zero bytes in
    // the shared secret (P1-009 fix). The probability of all-zero is < 2^-256.
    std::uint8_t acc = 0;
    for (auto b : shared_secret) acc |= b;
    if (acc == 0) {
        // BUG-FIX: zeroize sk before early return so the ephemeral private key
        // scalar does not remain on the stack after an attacker-induced ECDH failure.
        detail::secure_erase(&sk, sizeof(sk));
        return false;
    }

    // 2. Derive PRK via HKDF-Extract
    constexpr char salt[] = "bitcoin_v2_shared_secret";
    auto prk = hkdf_sha256_extract(
        reinterpret_cast<const std::uint8_t*>(salt), sizeof(salt) - 1,
        shared_secret.data(), shared_secret.size());

    // 3. Derive directional keys via HKDF-Expand
    std::uint8_t initiator_key[32], responder_key[32];

    constexpr char init_info[] = "initiator_L";
    constexpr char resp_info[] = "responder_L";

    hkdf_sha256_expand(prk.data(),
                        reinterpret_cast<const std::uint8_t*>(init_info), sizeof(init_info) - 1,
                        initiator_key, 32);
    hkdf_sha256_expand(prk.data(),
                        reinterpret_cast<const std::uint8_t*>(resp_info), sizeof(resp_info) - 1,
                        responder_key, 32);

    // 4. Derive session ID
    constexpr char sid_info[] = "session_id";
    hkdf_sha256_expand(prk.data(),
                        reinterpret_cast<const std::uint8_t*>(sid_info), sizeof(sid_info) - 1,
                        session_id_.data(), 32);

    // 5. Assign send/recv keys based on role
    if (initiator_) {
        send_cipher_.init(initiator_key);
        recv_cipher_.init(responder_key);
    } else {
        send_cipher_.init(responder_key);
        recv_cipher_.init(initiator_key);
    }

    // Secure erase intermediates
    detail::secure_erase(shared_secret.data(), shared_secret.size());
    detail::secure_erase(prk.data(), prk.size());
    detail::secure_erase(initiator_key, sizeof(initiator_key));
    detail::secure_erase(responder_key, sizeof(responder_key));
    detail::secure_erase(&sk, sizeof(sk));
    // Proactively erase privkey_ now that it has served its purpose.
    // The destructor will also clear it, but zeroing here reduces the window.
    detail::secure_erase(privkey_.data(), privkey_.size());

    established_ = true;
    return true;
}

std::vector<std::uint8_t> Bip324Session::encrypt(
    const std::uint8_t* plaintext, std::size_t plaintext_len) noexcept {
    if (!established_) return {};
    return send_cipher_.encrypt(nullptr, 0, plaintext, plaintext_len);
}

bool Bip324Session::decrypt(
    const std::uint8_t* header,
    const std::uint8_t* payload_and_tag, std::size_t len,
    std::vector<std::uint8_t>& plaintext_out) noexcept {
    if (!established_) {
        plaintext_out.clear();
        return false;
    }
    return recv_cipher_.decrypt(nullptr, 0, header, payload_and_tag, len, plaintext_out);
}

// ============================================================================
// BIP-324 optimized XDH backend — sqrt-free x-only path
// ============================================================================
//
// Uses ct::ecmult_const_xonly (port of libsecp256k1 secp256k1_ecmult_const_xonly)
// to compute x(sk * P) directly from P's x-coordinate, without sqrt.
//
// Prior approach: decode ELL64 → x → sqrt(y) → build GLV tables → scalar_mul.
// New approach:   decode ELL64 → x → ecmult_const_xonly(x, 1, sk).
//   Eliminates: ~3.8 µs sqrt, ~1.95 µs table build, 16-slot peer cache overhead.
//   Trade-off:  one extra field inverse (~1.6 µs) vs. sqrt (~3.8 µs) → net ~2.2 µs saved.
//   Bonus:      removes thread-local state entirely.

namespace bip324 {

namespace {

static std::array<std::uint8_t,32> bip324_xdh_hash(
    const std::uint8_t x32[32],
    const std::uint8_t ell_a[64],
    const std::uint8_t ell_b[64]) noexcept
{
    static const auto kTag = SHA256::hash(
        reinterpret_cast<const std::uint8_t*>("bip324_ellswift_xonly_ecdh"), 26);
    SHA256 h;
    h.update(kTag.data(), 32); h.update(kTag.data(), 32);
    h.update(ell_a, 64); h.update(ell_b, 64); h.update(x32, 32);
    return h.finalize();
}

} // anonymous

std::array<std::uint8_t,32> xdh(
    const std::uint8_t ell_a64[64],   // initiator's ELL (party 0)
    const std::uint8_t ell_b64[64],   // responder's ELL (party 1)
    const fast::Scalar& sk,
    bool initiating) noexcept
{
    // peer_ell: the key we ECDH against (always the OTHER party's key)
    //   initiating → their key is ell_b64
    //   responding → their key is ell_a64
    const std::uint8_t* peer_ell = initiating ? ell_b64 : ell_a64;

    // Decode peer ELL as fraction (xn:xd) — no field inverse, no sqrt.
    // Port of libsecp's secp256k1_ellswift_xswiftec_frac_var.
    // ~3.8 µs (1 sqrt for QR check) vs ~7 µs for ellswift_decode (2 inverses + 1 sqrt).
    auto [xn, xd] = ellswift_decode_frac(peer_ell);

    // CT scalar multiply via x-only path: no sqrt, one combined inverse.
    // Passes fraction (xn:xd) directly — avoids the inverse in xd→x conversion.
    auto px = ct::ecmult_const_xonly(xn, xd, sk);

    if (px == fast::FieldElement::zero()) return {};

    auto x32 = px.to_bytes();
    // Hash: (x, ell_a64, ell_b64) — same order for both parties
    return bip324_xdh_hash(x32.data(), ell_a64, ell_b64);
}

} // namespace bip324

} // namespace secp256k1
