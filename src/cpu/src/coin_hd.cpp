// ============================================================================
// Coin HD -- Implementation
// ============================================================================
// BIP-44 coin-type derivation using existing BIP-32 infrastructure.
// ============================================================================

#include "secp256k1/coins/coin_hd.hpp"
#include "secp256k1/coins/coin_address.hpp"
#include "secp256k1/detail/secure_erase.hpp"

namespace secp256k1::coins {

namespace {

constexpr std::uint32_t BIP32_INDEX_MAX = 0x7FFFFFFFu;

void secure_erase_extended_key(ExtendedKey& key) noexcept {
    detail::secure_erase(key.key.data(), key.key.size());
    detail::secure_erase(key.chain_code.data(), key.chain_code.size());
    key.parent_fingerprint.fill(0);
    key.depth = 0;
    key.child_number = 0;
    key.is_private = false;
    key.pub_prefix = 0;
}

class ExtendedKeyEraseGuard {
public:
    explicit ExtendedKeyEraseGuard(ExtendedKey& key) noexcept : key_(key) {}
    ~ExtendedKeyEraseGuard() { secure_erase_extended_key(key_); }

private:
    ExtendedKey& key_;
};

bool is_valid_bip32_index(std::uint32_t index) noexcept {
    return index <= BIP32_INDEX_MAX;
}

} // namespace

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
    auto const purpose_index = static_cast<std::uint32_t>(purpose);
    std::uint32_t const coin_index = coin.coin_type;
    std::uint32_t const change_index = change ? 1u : 0u;

    if (!is_valid_bip32_index(purpose_index) ||
        !is_valid_bip32_index(coin_index) ||
        !is_valid_bip32_index(account) ||
        !is_valid_bip32_index(address_index)) {
        return {ExtendedKey{}, false};
    }

    auto [purpose_key, purpose_ok] = master.derive_hardened(purpose_index);
    const ExtendedKeyEraseGuard purpose_guard(purpose_key);
    if (!purpose_ok) return {ExtendedKey{}, false};

    auto [coin_key, coin_ok] = purpose_key.derive_hardened(coin_index);
    const ExtendedKeyEraseGuard coin_guard(coin_key);
    if (!coin_ok) return {ExtendedKey{}, false};

    auto [account_key, account_ok] = coin_key.derive_hardened(account);
    const ExtendedKeyEraseGuard account_guard(account_key);
    if (!account_ok) return {ExtendedKey{}, false};

    auto [change_key, change_ok] = account_key.derive_normal(change_index);
    const ExtendedKeyEraseGuard change_guard(change_key);
    if (!change_ok) return {ExtendedKey{}, false};

    return change_key.derive_normal(address_index);
}

// -- Seed -> Address -----------------------------------------------------------

std::pair<std::string, bool>
coin_address_from_seed(const std::uint8_t* seed, std::size_t seed_len,
                       const CoinParams& coin,
                       std::uint32_t account,
                       std::uint32_t address_index) {
    // Step 1: Master key from seed
    auto [master, master_ok] = bip32_master_key(seed, seed_len);
    const ExtendedKeyEraseGuard master_guard(master);
    if (!master_ok) return {{}, false};
    
    // Step 2: Derive coin-specific child
    auto [child, child_ok] = coin_derive_key(master, coin, account, false, address_index);
    const ExtendedKeyEraseGuard child_guard(child);
    if (!child_ok) return {{}, false};
    
    // Step 3: Generate address
    auto pubkey = child.public_key();
    std::string const addr = coin_address(pubkey, coin);
    
    return {addr, true};
}

} // namespace secp256k1::coins
