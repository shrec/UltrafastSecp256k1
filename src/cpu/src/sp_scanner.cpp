// sp_scanner.cpp — SilentPaymentScanner implementation (isolated TU for LTO)
#include "secp256k1/address.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

namespace secp256k1 {

using fast::Scalar;
using fast::Point;

// ── SilentPaymentScanner ─────────────────────────────────────────────────────

SilentPaymentScanner::SilentPaymentScanner(const fast::Scalar& scan_sk,
                                            const fast::Scalar& spend_sk)
    : scan_privkey_(scan_sk)
    , spend_privkey_(spend_sk)
    , spend_pubkey_(ct::generator_mul(spend_sk))  // precompute B_spend once
{}

std::vector<std::pair<std::uint32_t, fast::Scalar>>
SilentPaymentScanner::scan_tx(const std::vector<fast::Point>& input_pubkeys,
                               const std::vector<std::array<std::uint8_t, 32>>& output_pubkeys) const {
    std::vector<std::pair<std::uint32_t, Scalar>> results;

    Point A_sum = Point::infinity();
    for (const auto& A : input_pubkeys) A_sum = A_sum.add(A);
    if (A_sum.is_infinity()) return results;

    // S = b_scan × A_sum  — variable-time GLV: A_sum is public on-chain,
    // timing leaks nothing about scan_privkey to an attacker without oracle.
    Point S = A_sum.scalar_mul(scan_privkey_);
    auto S_comp = S.to_compressed();

    // Static tag hash — computed once (same pattern as LtcSpScanner)
    static const auto s_tag =
        SHA256::hash(reinterpret_cast<const std::uint8_t*>("BIP0352/SharedSecret"), 20);
    // Pre-build midstate: tag||tag||S_comp — only 4-byte k appended per output
    SHA256 h_base;
    h_base.update(s_tag.data(), 32);
    h_base.update(s_tag.data(), 32);
    h_base.update(S_comp.data(), 33);

    for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(output_pubkeys.size()); ++k) {
        SHA256 h = h_base;
        std::uint8_t k_be[4] = {
            std::uint8_t(k >> 24), std::uint8_t(k >> 16),
            std::uint8_t(k >> 8),  std::uint8_t(k)
        };
        h.update(k_be, 4);
        auto t_hash = h.finalize();
        Scalar t_k = Scalar::from_bytes(t_hash);

        // t_k is SHA256 output — uniform random, no CT needed here
        Point expected = spend_pubkey_.add(Point::generator().scalar_mul(t_k));
        if (!expected.is_infinity()) {
            auto expected_x = expected.x().to_bytes();
            if (expected_x == output_pubkeys[k])
                results.emplace_back(k, spend_privkey_ + t_k);
        }
        detail::secure_erase(t_hash.data(), t_hash.size());
    }

    detail::secure_erase(S_comp.data(), S_comp.size());
    return results;
}


} // namespace secp256k1
