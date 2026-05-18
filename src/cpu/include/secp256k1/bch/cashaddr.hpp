#pragma once
#include "bch_types.hpp"
#include <string>
#include <optional>

namespace secp256k1::bch {

constexpr std::string_view CASHADDR_PREFIX_MAINNET = "bitcoincash";
constexpr std::string_view CASHADDR_PREFIX_TESTNET = "bchtest";
constexpr std::string_view CASHADDR_PREFIX_CHIPNET = "chipnet";
constexpr std::string_view CASHADDR_PREFIX_REGTEST = "bchreg";

[[nodiscard]] std::string cashaddr_encode(const uint8_t* hash, size_t hash_len,
    AddrType type, Network network = Network::Mainnet) noexcept;

template<size_t N>
[[nodiscard]] std::string cashaddr_encode(const std::array<uint8_t,N>& hash,
    AddrType type, Network network = Network::Mainnet) noexcept {
    return cashaddr_encode(hash.data(), N, type, network);
}

[[nodiscard]] std::optional<CashAddr> cashaddr_decode(std::string_view addr) noexcept;
[[nodiscard]] std::string cashaddr_from_pubkey(const uint8_t* pubkey33,
    Network network = Network::Mainnet) noexcept;
[[nodiscard]] std::string cashaddr_from_script_hash(const uint8_t* hash20,
    Network network = Network::Mainnet) noexcept;
[[nodiscard]] bool cashaddr_is_valid(std::string_view addr) noexcept;

} // namespace secp256k1::bch
