// ============================================================================
// Unified Wallet API -- Implementation
// ============================================================================
// Dispatches to chain-specific implementations based on CoinParams.
// ============================================================================

#include "secp256k1/coins/wallet.hpp"
#include "secp256k1/coins/coin_address.hpp"
#include "secp256k1/coins/message_signing.hpp"
#include "secp256k1/context.hpp"
#include "secp256k1/recovery.hpp"
#include <cstring>

#if defined(SECP256K1_BUILD_ETHEREUM)
#include "secp256k1/coins/eth_signing.hpp"
#include "secp256k1/coins/ethereum.hpp"
#endif

namespace secp256k1::coins::wallet {

// -- MessageSignature ---------------------------------------------------------

std::array<std::uint8_t, 65> MessageSignature::to_rsv() const {
    std::array<std::uint8_t, 65> out;
    std::memcpy(out.data(), r.data(), 32);
    std::memcpy(out.data() + 32, s.data(), 32);
    out[64] = static_cast<std::uint8_t>(v & 0xFF);
    return out;
}

// -- Hex helpers (no heap for small buffers) -----------------------------------

static char hex_digit(int n) {
    return (n < 10) ? static_cast<char>('0' + n) : static_cast<char>('a' + n - 10);
}

static std::string to_hex(const std::uint8_t* data, std::size_t len,
                          bool prefix_0x = false) {
    std::string result;
    result.reserve(len * 2 + (prefix_0x ? 2 : 0));
    if (prefix_0x) { result += '0'; result += 'x'; }
    for (std::size_t i = 0; i < len; ++i) {
        result += hex_digit(data[i] >> 4);
        result += hex_digit(data[i] & 0x0F);
    }
    return result;
}

// -- Key Management -----------------------------------------------------------

std::pair<WalletKey, bool> from_private_key(const std::uint8_t* priv32) {
    WalletKey key{};

    // Parse and validate private key
    auto scalar = fast::Scalar::from_bytes(priv32);
    if (scalar.is_zero()) return {key, false};

    key.priv = scalar;
    key.pub = derive_public_key(scalar);
    return {key, true};
}

std::string get_address(const CoinParams& coin, const WalletKey& key,
                        bool testnet) {
    return coin_address(key.pub, coin, testnet);
}

std::string get_address_p2pkh(const CoinParams& coin, const WalletKey& key,
                              bool testnet) {
    return coin_address_p2pkh(key.pub, coin, testnet);
}

std::string get_address_p2wpkh(const CoinParams& coin, const WalletKey& key,
                               bool testnet) {
    return coin_address_p2wpkh(key.pub, coin, testnet);
}

std::string get_address_p2sh_p2wpkh(const CoinParams& coin, const WalletKey& key,
                                    bool testnet) {
    return coin_address_p2sh_p2wpkh(key.pub, coin, testnet);
}

std::string get_address_p2tr(const CoinParams& coin, const WalletKey& key,
                             bool testnet) {
    return coin_address_p2tr(key.pub, coin, testnet);
}

std::string get_address_cashaddr(const CoinParams& coin, const WalletKey& key,
                                 bool testnet) {
    return coin_address_cashaddr(key.pub, coin, testnet);
}

std::string export_private_key(const CoinParams& coin, const WalletKey& key,
                               bool testnet) {
    if (coin.features.uses_evm) {
        // EVM: 0x-prefixed hex
        auto bytes = key.priv.to_bytes();
        return to_hex(bytes.data(), 32, true);
    }
    if (coin.default_encoding == AddressEncoding::TRON_BASE58) {
        // Tron: raw hex (no 0x prefix)
        auto bytes = key.priv.to_bytes();
        return to_hex(bytes.data(), 32, false);
    }
    // Bitcoin-family: WIF
    return coin_wif_encode(key.priv, coin, true, testnet);
}

std::string export_public_key_hex(const CoinParams& coin, const WalletKey& key) {
    if (coin.features.uses_evm || coin.default_encoding == AddressEncoding::TRON_BASE58) {
        // EVM/Tron: uncompressed (65 bytes)
        auto uncompressed = key.pub.to_uncompressed();
        return to_hex(uncompressed.data(), 65);
    }
    // Bitcoin-family: compressed (33 bytes)
    auto compressed = key.pub.to_compressed();
    return to_hex(compressed.data(), 33);
}

// -- Signing ------------------------------------------------------------------

// Helper: build MessageSignature from RecoverableSignature
static MessageSignature from_recoverable(const RecoverableSignature& rsig,
                                         std::uint64_t v_value) {
    MessageSignature result{};
    auto compact = rsig.sig.to_compact();
    std::memcpy(result.r.data(), compact.data(), 32);
    std::memcpy(result.s.data(), compact.data() + 32, 32);
    result.recid = rsig.recid;
    result.v = v_value;
    return result;
}

MessageSignature sign_message(const CoinParams& coin, const WalletKey& key,
                              const std::uint8_t* msg, std::size_t msg_len) {
#if defined(SECP256K1_BUILD_ETHEREUM)
    if (coin.features.uses_evm || coin.default_encoding == AddressEncoding::TRON_BASE58) {
        // EVM/Tron: EIP-191 personal_sign
        auto eth_sig = eth_personal_sign(msg, msg_len, key.priv);
        MessageSignature result{};
        result.r = eth_sig.r;
        result.s = eth_sig.s;
        result.v = eth_sig.v;
        result.recid = eip155_recid(eth_sig.v);
        return result;
    }
#endif
    // Bitcoin-family: Bitcoin signed message format
    auto rsig = bitcoin_sign_message(msg, msg_len, key.priv);
    return from_recoverable(rsig, static_cast<std::uint64_t>(27 + rsig.recid));
}

MessageSignature sign_hash(const CoinParams& coin, const WalletKey& key,
                           const std::uint8_t* hash32) {
    std::array<std::uint8_t, 32> hash;
    std::memcpy(hash.data(), hash32, 32);

#if defined(SECP256K1_BUILD_ETHEREUM)
    if (coin.features.uses_evm || coin.default_encoding == AddressEncoding::TRON_BASE58) {
        auto eth_sig = eth_sign_hash(hash, key.priv, coin.chain_id);
        MessageSignature result{};
        result.r = eth_sig.r;
        result.s = eth_sig.s;
        result.v = eth_sig.v;
        result.recid = eip155_recid(eth_sig.v);
        return result;
    }
#endif
    auto rsig = ecdsa_sign_recoverable(hash, key.priv);
    return from_recoverable(rsig, static_cast<std::uint64_t>(27 + rsig.recid));
}

// -- Verification -------------------------------------------------------------

bool verify_message(const CoinParams& coin, const fast::Point& pubkey,
                    const std::uint8_t* msg, std::size_t msg_len,
                    const MessageSignature& sig) {
    // Reconstruct ECDSASignature from r, s
    std::array<std::uint8_t, 64> compact{};
    std::memcpy(compact.data(), sig.r.data(), 32);
    std::memcpy(compact.data() + 32, sig.s.data(), 32);
    auto ecdsa_sig = ECDSASignature::from_compact(compact);

#if defined(SECP256K1_BUILD_ETHEREUM)
    if (coin.features.uses_evm || coin.default_encoding == AddressEncoding::TRON_BASE58) {
        auto hash = eip191_hash(msg, msg_len);
        return ecdsa_verify(hash.data(), pubkey, ecdsa_sig);
    }
#endif
    return bitcoin_verify_message(msg, msg_len, pubkey, ecdsa_sig);
}

// -- Recovery -----------------------------------------------------------------

std::pair<fast::Point, bool>
recover_signer(const CoinParams& coin,
               const std::uint8_t* msg, std::size_t msg_len,
               const MessageSignature& sig) {
    std::array<std::uint8_t, 64> compact{};
    std::memcpy(compact.data(), sig.r.data(), 32);
    std::memcpy(compact.data() + 32, sig.s.data(), 32);
    auto ecdsa_sig = ECDSASignature::from_compact(compact);

#if defined(SECP256K1_BUILD_ETHEREUM)
    if (coin.features.uses_evm || coin.default_encoding == AddressEncoding::TRON_BASE58) {
        auto hash = eip191_hash(msg, msg_len);
        return ecdsa_recover(hash, ecdsa_sig, sig.recid);
    }
#endif
    return bitcoin_recover_message(msg, msg_len, ecdsa_sig, sig.recid);
}

std::pair<std::string, bool>
recover_address(const CoinParams& coin,
                const std::uint8_t* msg, std::size_t msg_len,
                const MessageSignature& sig) {
    auto [pubkey, ok] = recover_signer(coin, msg, msg_len, sig);
    if (!ok) return {"", false};
    return {coin_address(pubkey, coin), true};
}

} // namespace secp256k1::coins::wallet
