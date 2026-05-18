#pragma once
#include <cstdint>
#include <array>
#include <string>
#include <string_view>
#include <optional>

namespace secp256k1::bch {

enum class Network : uint8_t { Mainnet = 0, Testnet = 1, Regtest = 2, Chipnet = 3 };
enum class AddrType : uint8_t { P2PKH = 0, P2SH = 1, P2SH32 = 2 };

struct CashAddr {
    Network  network;
    AddrType type;
    std::array<uint8_t, 32> hash;
    uint8_t  hash_len;
};

struct RpaPaycode {
    uint8_t  version;
    uint8_t  prefix_bits;
    std::array<uint8_t, 33> scan_pubkey;
    std::array<uint8_t, 33> spend_pubkey;
    uint32_t expiry;
    Network network() const noexcept {
        return (version >= 5) ? Network::Testnet : Network::Mainnet;
    }
    AddrType addr_type() const noexcept {
        return (version == 3 || version == 4 || version == 7 || version == 8)
               ? AddrType::P2SH : AddrType::P2PKH;
    }
};

struct RpaSharedSecret { std::array<uint8_t, 32> value; };

struct GrindResult {
    bool     found = false;
    uint32_t nonce = 0;
    std::array<uint8_t, 64> signature{};
    std::array<uint8_t, 32> input_hash{};
};

} // namespace secp256k1::bch
