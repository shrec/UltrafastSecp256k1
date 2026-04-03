#ifndef SECP256K1_BIP324_HPP
#define SECP256K1_BIP324_HPP
#pragma once

// ============================================================================
// BIP-324: Version 2 P2P Encrypted Transport Protocol
// ============================================================================
// Implements the BIP-324 v2 encrypted transport, consisting of:
//   1. Key exchange via ElligatorSwift-encoded ECDH
//   2. Key derivation via HKDF-SHA256
//   3. Packet encryption via ChaCha20-Poly1305 AEAD
//
// Session lifecycle:
//   1. Both peers generate ephemeral keys and ElligatorSwift encodings
//   2. Exchange 64-byte encodings
//   3. Derive symmetric keys via ECDH + HKDF
//   4. Encrypt/decrypt packets with ChaCha20-Poly1305
//
// Packet format:
//   [3 bytes encrypted length] [N bytes encrypted payload] [16 bytes Poly1305 tag]
//
// Reference: https://github.com/bitcoin/bips/blob/master/bip-0324.mediawiki
// ============================================================================

#include <array>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <vector>
#include "secp256k1/scalar.hpp"

namespace secp256k1 {

// -- BIP-324 Cipher Suite (per-direction) -------------------------------------

class Bip324Cipher {
public:
    Bip324Cipher() noexcept = default;
    ~Bip324Cipher() noexcept {
        // Volatile write to prevent dead-store elimination
        volatile std::uint8_t* p = key_;
        for (std::size_t i = 0; i < 32; ++i) p[i] = 0;
    }

    // Initialize with a 32-byte key for this direction
    void init(const std::uint8_t* key) noexcept;

    // Encrypt a packet (plaintext -> header_enc[3] + payload_enc[len] + tag[16])
    // aad is optional associated data.
    // Returns the encrypted output vector: [3-byte enc length][payload][16-byte tag]
    std::vector<std::uint8_t> encrypt(
        const std::uint8_t* aad, std::size_t aad_len,
        const std::uint8_t* plaintext, std::size_t plaintext_len) noexcept;

    // Decrypt a packet. header_enc is the 3-byte encrypted length prefix.
    // contents is the encrypted payload + 16-byte tag.
    // Returns true on success and writes the decrypted payload to plaintext_out.
    // Returns false on auth failure or malformed decrypted packet framing.
    bool decrypt(
        const std::uint8_t* aad, std::size_t aad_len,
        const std::uint8_t* header_enc,
        const std::uint8_t* contents, std::size_t contents_len,
        std::vector<std::uint8_t>& plaintext_out) noexcept;

    // Get the current packet counter (nonce)
    std::uint64_t packet_counter() const noexcept { return packet_counter_; }

private:
    std::uint8_t key_[32]{};
    std::uint64_t packet_counter_ = 0;

    // Build a 12-byte nonce from the packet counter
    void build_nonce(std::uint8_t* nonce) const noexcept;
};

// -- BIP-324 Session ----------------------------------------------------------

class Bip324Session {
public:
    // Initialize session: generate ephemeral key and ElligatorSwift encoding.
    // After construction, call our_ellswift_encoding() to get our 64-byte message.
    explicit Bip324Session(bool initiator) noexcept;

    // Initialize with a specific private key (for testing / deterministic use)
    Bip324Session(bool initiator, const std::uint8_t* privkey) noexcept;

    ~Bip324Session() noexcept {
        // Erase private key material on destruction
        volatile std::uint8_t* p = privkey_.data();
        for (std::size_t i = 0; i < 32; ++i) p[i] = 0;
    }

    // Get our 64-byte ElligatorSwift-encoded public key to send to peer
    const std::array<std::uint8_t, 64>& our_ellswift_encoding() const noexcept {
        return our_encoding_;
    }

    // Complete the handshake by providing the peer's 64-byte ElligatorSwift encoding.
    // This derives the shared secret and symmetric keys.
    // Returns true on success.
    bool complete_handshake(const std::uint8_t* peer_encoding) noexcept;

    // Encrypt a message for sending to the peer.
    // Returns: [3-byte encrypted length][payload][16-byte tag]
    std::vector<std::uint8_t> encrypt(
        const std::uint8_t* plaintext, std::size_t plaintext_len) noexcept;

    // Decrypt a received message.
    // header is the 3-byte encrypted length prefix.
    // payload_and_tag is the encrypted payload followed by the 16-byte tag.
    // Returns true on success and writes the decrypted payload to plaintext_out.
    // Returns false on auth failure or malformed decrypted packet framing.
    bool decrypt(
        const std::uint8_t* header,
        const std::uint8_t* payload_and_tag, std::size_t len,
        std::vector<std::uint8_t>& plaintext_out) noexcept;

    // Check if handshake is complete
    bool is_established() const noexcept { return established_; }

    // Get session ID (32 bytes, derived during key setup)
    const std::array<std::uint8_t, 32>& session_id() const noexcept {
        return session_id_;
    }

private:
    bool initiator_;
    bool established_ = false;

    // Our ephemeral key
    std::array<std::uint8_t, 32> privkey_{};
    std::array<std::uint8_t, 64> our_encoding_{};

    // Peer's encoding
    std::array<std::uint8_t, 64> peer_encoding_{};

    // Derived keys
    std::array<std::uint8_t, 32> session_id_{};
    Bip324Cipher send_cipher_;
    Bip324Cipher recv_cipher_;
};

} // namespace secp256k1

#endif // SECP256K1_BIP324_HPP
