// ============================================================================
// Ethereum Address -- Implementation
// ============================================================================
// EIP-55 mixed-case checksum addresses using Keccak-256.
// ============================================================================

#include "secp256k1/coins/ethereum.hpp"
#include "secp256k1/coins/keccak256.hpp"
#include <cstring>

namespace secp256k1::coins {

// -- Hex helpers (no heap, no formatting) -------------------------------------

static constexpr char HEX_LOWER[] = "0123456789abcdef";

static void bytes_to_hex_lower(const std::uint8_t* bytes, std::size_t len,
                                char* out) {
    for (std::size_t i = 0; i < len; ++i) {
        out[i * 2]     = HEX_LOWER[bytes[i] >> 4];
        out[i * 2 + 1] = HEX_LOWER[bytes[i] & 0x0F];
    }
}

// -- Raw Address Bytes --------------------------------------------------------

std::array<std::uint8_t, 20> ethereum_address_bytes(const fast::Point& pubkey) {
    // Get uncompressed public key (65 bytes: 0x04 + x[32] + y[32])
    auto uncompressed = pubkey.to_uncompressed();
    
    // Keccak-256 of the 64-byte public key (skip 0x04 prefix)
    auto hash = keccak256(uncompressed.data() + 1, 64);
    
    // Take last 20 bytes
    std::array<std::uint8_t, 20> addr;
    std::memcpy(addr.data(), hash.data() + 12, 20);
    return addr;
}

// -- Raw Hex Address ----------------------------------------------------------

std::string ethereum_address_raw(const fast::Point& pubkey) {
    auto addr_bytes = ethereum_address_bytes(pubkey);
    char hex[40];
    bytes_to_hex_lower(addr_bytes.data(), 20, hex);
    return std::string(hex, 40);
}

// -- EIP-55 Checksum ----------------------------------------------------------

std::string eip55_checksum(const std::string& hex_addr) {
    // Keccak-256 of the lowercase hex address string
    auto hash = keccak256(reinterpret_cast<const std::uint8_t*>(hex_addr.data()),
                          hex_addr.size());
    
    std::string result(40, '\0');
    for (std::size_t i = 0; i < 40; ++i) {
        char const c = hex_addr[i];
        if (c >= 'a' && c <= 'f') {
            // Get the corresponding nibble from the hash
            std::uint8_t const hash_nibble = (hash[i / 2] >> ((1 - (i % 2)) * 4)) & 0x0F;
            result[i] = (hash_nibble >= 8) ? static_cast<char>(c - 32) : c; // uppercase if hash nibble >= 8
        } else {
            result[i] = c; // digits 0-9 stay as-is
        }
    }
    return result;
}

bool eip55_verify(const std::string& addr) {
    // Strip "0x" prefix if present
    const char* hex = addr.c_str();
    if (addr.size() >= 2 && hex[0] == '0' && (hex[1] == 'x' || hex[1] == 'X')) {
        hex += 2;
    }
    
    if (std::strlen(hex) != 40) return false;
    
    // Get lowercase version
    char lower[40];
    for (int i = 0; i < 40; ++i) {
        char const c = hex[i];
        lower[i] = (c >= 'A' && c <= 'F') ? static_cast<char>(c + 32) : c;
    }
    
    // Re-apply checksum and compare
    std::string const lower_str(lower, 40);
    std::string checksummed = eip55_checksum(lower_str);
    
    return std::memcmp(hex, checksummed.data(), 40) == 0;
}

// -- EIP-55 Address -----------------------------------------------------------

std::string ethereum_address(const fast::Point& pubkey) {
    std::string const raw = ethereum_address_raw(pubkey);
    std::string const checksummed = eip55_checksum(raw);
    return "0x" + checksummed;
}

} // namespace secp256k1::coins
