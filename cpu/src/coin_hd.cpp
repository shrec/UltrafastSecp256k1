// ============================================================================
// Coin HD -- Implementation
// ============================================================================
// BIP-44 coin-type derivation using existing BIP-32 infrastructure.
// ============================================================================

#include "secp256k1/coins/coin_hd.hpp"
#include "secp256k1/coins/coin_address.hpp"

namespace secp256k1::coins {

// -- Purpose Selection --------------------------------------------------------

DerivationPurpose best_purpose(const CoinParams& coin) {
    if (coin.features.supports_taproot) return DerivationPurpose::BIP86;
    if (coin.features.supports_segwit)  return DerivationPurpose::BIP84;
    return DerivationPurpose::BIP44;
}

// -- Path Construction --------------------------------------------------------

std::string coin_derive_path(const CoinParams& coin,
                             std::uint32_t account,
                             bool change,
                             std::uint32_t address_index,
                             DerivationPurpose purpose) {
    // Build path: m / purpose' / coin_type' / account' / change / index
    std::string path = "m/";
    path += std::to_string(static_cast<std::uint32_t>(purpose));
    path += "'/";
    path += std::to_string(coin.coin_type);
    path += "'/";
    path += std::to_string(account);
    path += "'/";
    path += std::to_string(change ? 1u : 0u);
    path += "/";
    path += std::to_string(address_index);
    return path;
}

// -- Key Derivation -----------------------------------------------------------

std::pair<ExtendedKey, bool>
coin_derive_key(const ExtendedKey& master,
                const CoinParams& coin,
                std::uint32_t account,
                bool change,
                std::uint32_t address_index) {
    DerivationPurpose const purpose = best_purpose(coin);
    return coin_derive_key_with_purpose(master, coin, purpose,
                                         account, change, address_index);
}

std::pair<ExtendedKey, bool>
coin_derive_key_with_purpose(const ExtendedKey& master,
                             const CoinParams& coin,
                             DerivationPurpose purpose,
                             std::uint32_t account,
                             bool change,
                             std::uint32_t address_index) {
    std::string const path = coin_derive_path(coin, account, change,
                                         address_index, purpose);
    return bip32_derive_path(master, path);
}

// -- Seed -> Address -----------------------------------------------------------

std::pair<std::string, bool>
coin_address_from_seed(const std::uint8_t* seed, std::size_t seed_len,
                       const CoinParams& coin,
                       std::uint32_t account,
                       std::uint32_t address_index) {
    // Step 1: Master key from seed
    auto [master, master_ok] = bip32_master_key(seed, seed_len);
    if (!master_ok) return {{}, false};
    
    // Step 2: Derive coin-specific child
    auto [child, child_ok] = coin_derive_key(master, coin, account, false, address_index);
    if (!child_ok) return {{}, false};
    
    // Step 3: Generate address
    auto pubkey = child.public_key();
    std::string const addr = coin_address(pubkey, coin);
    
    return {addr, true};
}

} // namespace secp256k1::coins
