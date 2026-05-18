#pragma once
#include "bch_types.hpp"
#include "../scalar.hpp"
#include "../point.hpp"
#include <functional>
#include <optional>

namespace secp256k1::bch {

[[nodiscard]] std::optional<RpaPaycode> rpa_parse_paycode(std::string_view) noexcept;
[[nodiscard]] std::string rpa_encode_paycode(const RpaPaycode&) noexcept;

[[nodiscard]] RpaSharedSecret rpa_sender_shared_secret(
    const fast::Scalar& input_privkey, const uint8_t* scan_pubkey33,
    const uint8_t* outpoint_bytes, size_t outpoint_len) noexcept;

[[nodiscard]] RpaSharedSecret rpa_receiver_shared_secret(
    const fast::Scalar& scan_privkey, const uint8_t* input_pubkey33,
    const uint8_t* outpoint_bytes, size_t outpoint_len) noexcept;

[[nodiscard]] std::array<uint8_t, 33> rpa_derive_payment_pubkey(
    const uint8_t* spend_pubkey33, const RpaSharedSecret& secret,
    uint32_t index = 0) noexcept;

using GrindProgressFn = std::function<void(uint32_t)>;

[[nodiscard]] GrindResult rpa_grind_cpu(
    const fast::Scalar& input_privkey, const uint8_t* msg32,
    uint8_t prefix_bits, const uint8_t* prefix_data,
    uint32_t max_tries = 0, GrindProgressFn on_progress = nullptr) noexcept;

[[nodiscard]] bool rpa_prefix_matches(const uint8_t* sig64,
    uint8_t prefix_bits, const uint8_t* prefix_data) noexcept;
[[nodiscard]] std::array<uint8_t, 32> rpa_sig_hash(const uint8_t* sig64) noexcept;

} // namespace secp256k1::bch
