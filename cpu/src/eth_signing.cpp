// ============================================================================
// Ethereum Signing Primitives -- Implementation
// ============================================================================
// EIP-191 personal_sign, EIP-155 chain ID, ecrecover
// ============================================================================

#include "secp256k1/coins/eth_signing.hpp"
#include "secp256k1/coins/keccak256.hpp"
#include "secp256k1/coins/ethereum.hpp"
#include "secp256k1/recovery.hpp"
#include <cstring>

namespace secp256k1::coins {

using fast::Scalar;

// -- Decimal length encoding (no heap) ----------------------------------------

// Write decimal representation of len into buf. Returns number of chars written.
static std::size_t write_decimal(std::size_t val, char* buf, std::size_t buf_size) {
    if (val == 0) {
        buf[0] = '0';
        return 1;
    }
    // Write digits in reverse
    char tmp[20];  // max uint64 is 20 digits
    std::size_t n = 0;
    while (val > 0 && n < sizeof(tmp)) {
        tmp[n++] = static_cast<char>('0' + static_cast<char>(val % 10));
        val /= 10;
    }
    if (n > buf_size) n = buf_size;
    // Reverse into buf
    for (std::size_t i = 0; i < n; ++i) {
        buf[i] = tmp[n - 1 - i];
    }
    return n;
}

// -- EIP-191: Personal Message Hash -------------------------------------------

std::array<std::uint8_t, 32> eip191_hash(const std::uint8_t* msg, std::size_t msg_len) {
    // Prefix: "\x19Ethereum Signed Message:\n"
    static constexpr char PREFIX[] = "\x19" "Ethereum Signed Message:\n";
    static constexpr std::size_t PREFIX_LEN = 26;  // strlen of above

    // Decimal string of msg_len
    char len_str[20];
    std::size_t const len_str_n = write_decimal(msg_len, len_str, sizeof(len_str));

    // Keccak256(prefix + decimal_len + msg)
    Keccak256State hasher;
    hasher.update(reinterpret_cast<const std::uint8_t*>(PREFIX), PREFIX_LEN);
    hasher.update(reinterpret_cast<const std::uint8_t*>(len_str), len_str_n);
    hasher.update(msg, msg_len);
    return hasher.finalize();
}

// -- Ethereum Sign ------------------------------------------------------------

EthSignature eth_personal_sign(const std::uint8_t* msg, std::size_t msg_len,
                               const Scalar& private_key) {
    auto hash = eip191_hash(msg, msg_len);
    return eth_sign_hash(hash, private_key, 0);
}

EthSignature eth_sign_hash(const std::array<std::uint8_t, 32>& hash,
                           const Scalar& private_key,
                           std::uint64_t chain_id) {
    // ecdsa_sign_recoverable uses RFC 6979 deterministic nonce
    auto rsig = secp256k1::ecdsa_sign_recoverable(hash, private_key);

    EthSignature result;
    auto r_bytes = rsig.sig.r.to_bytes();
    auto s_bytes = rsig.sig.s.to_bytes();
    std::memcpy(result.r.data(), r_bytes.data(), 32);
    std::memcpy(result.s.data(), s_bytes.data(), 32);

    if (chain_id == 0) {
        result.v = 27 + static_cast<std::uint64_t>(rsig.recid);
    } else {
        result.v = eip155_v(rsig.recid, chain_id);
    }
    return result;
}

// -- ecrecover ----------------------------------------------------------------

std::pair<std::array<std::uint8_t, 20>, bool>
ecrecover(const std::array<std::uint8_t, 32>& msg_hash,
          const std::array<std::uint8_t, 32>& r,
          const std::array<std::uint8_t, 32>& s,
          std::uint64_t v) {
    // Parse r, s
    const Scalar r_scalar = Scalar::from_bytes(r);
    const Scalar s_scalar = Scalar::from_bytes(s);

    if (r_scalar.is_zero() || s_scalar.is_zero()) {
        return {{}, false};
    }

    // Build ECDSA signature
    secp256k1::ECDSASignature sig;
    sig.r = r_scalar;
    sig.s = s_scalar;

    // Extract recovery ID from v
    int const recid = eip155_recid(v);
    if (recid < 0 || recid > 3) {
        return {{}, false};
    }

    // Recover public key (fast path -- public data, no secret)
    auto [pubkey, ok] = secp256k1::ecdsa_recover(msg_hash, sig, recid);
    if (!ok || pubkey.is_infinity()) {
        return {{}, false};
    }

    // Derive Ethereum address from recovered public key
    auto addr = ethereum_address_bytes(pubkey);
    return {addr, true};
}

std::pair<std::array<std::uint8_t, 20>, bool>
ecrecover(const std::array<std::uint8_t, 32>& msg_hash,
          const EthSignature& sig) {
    return ecrecover(msg_hash, sig.r, sig.s, sig.v);
}

// -- Verify -------------------------------------------------------------------

bool eth_personal_verify(const std::uint8_t* msg, std::size_t msg_len,
                         const EthSignature& sig,
                         const std::array<std::uint8_t, 20>& expected_addr) {
    auto hash = eip191_hash(msg, msg_len);
    auto [recovered_addr, ok] = ecrecover(hash, sig);
    if (!ok) return false;
    return std::memcmp(recovered_addr.data(), expected_addr.data(), 20) == 0;
}

} // namespace secp256k1::coins
