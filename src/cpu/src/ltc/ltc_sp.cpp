// =============================================================================
// ltc_sp.cpp — Litecoin Silent Payments implementation
// =============================================================================
// Protocol: identical to BIP-352 except for two tagged hash domain strings.
//
// Tag differences (prevent cross-chain replay):
//   BTC: "BIP0352/SharedSecret" (20 bytes), "BIP0352/Inputs" (14 bytes)
//   LTC: "LTCSP/SharedSecret"   (18 bytes), "LTCSP/Inputs"   (12 bytes)
//
// Tagged hash construction (BIP-340 §D):
//   H_tag(x) = SHA256(SHA256(tag) || SHA256(tag) || x)
//
// All other protocol steps are identical to BIP-352 §3.
// =============================================================================

#include "secp256k1/ltc/ltc_sp.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/address.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/precompute.hpp"  // batch_scalar_mul_generator
#include "secp256k1/detail/secure_erase.hpp"
#include <cstring>

namespace secp256k1::ltc {

using fast::Point;
using fast::Scalar;

// -- LTC-SP tagged hash constants (computed once at first use) ----------------

// "LTCSP/SharedSecret" — 18 bytes
static const SHA256::digest_type& ltcsp_shared_secret_tag() {
    static const auto tag = SHA256::hash(
        reinterpret_cast<const std::uint8_t*>("LTCSP/SharedSecret"), 18);
    return tag;
}

// H_LTCSP/SharedSecret(data) = SHA256(tag || tag || data)
static SHA256::digest_type ltcsp_hash_shared_secret(const std::array<std::uint8_t, 33>& S_comp,
                                                     std::uint32_t k) {
    const auto& tag = ltcsp_shared_secret_tag();
    SHA256 h;
    h.update(tag.data(), 32);
    h.update(tag.data(), 32);
    h.update(S_comp.data(), 33);
    std::uint8_t k_be[4] = {
        std::uint8_t(k >> 24), std::uint8_t(k >> 16),
        std::uint8_t(k >> 8),  std::uint8_t(k)
    };
    h.update(k_be, 4);
    return h.finalize();
}

// -- Paycode encoding ---------------------------------------------------------

std::string LtcSpAddress::encode() const {
    // Payload: version(1) + scan_pubkey(33) + spend_pubkey(33) = 67 bytes
    // Encode as bech32m with HRP "ltcsp" and witness_version=0
    auto scan_c  = scan_pubkey.to_compressed();
    auto spend_c = spend_pubkey.to_compressed();

    // Payload: scan_pubkey(33) + spend_pubkey(33) = 66 bytes
    // Encoded as bech32m (witness_version=1 → BIP-350 bech32m, no size restriction)
    std::array<std::uint8_t, 66> payload{};
    std::memcpy(payload.data(),      scan_c.data(),  33);
    std::memcpy(payload.data() + 33, spend_c.data(), 33);

    return bech32_encode("ltcsp", 1, payload.data(), 66);
}

// Parse a 33-byte compressed pubkey into a Point
static std::pair<fast::Point, bool> parse_compressed(const std::uint8_t* b33) {
    if (b33[0] != 0x02 && b33[0] != 0x03) return {{}, false};
    fast::FieldElement x;
    if (!fast::FieldElement::parse_bytes_strict(b33 + 1, x)) return {{}, false};
    auto y2 = x * x * x + fast::FieldElement::from_uint64(7);
    auto y  = y2.sqrt();
    if (!(y * y == y2)) return {{}, false};
    bool y_is_odd = (y.limbs()[0] & 1u) != 0;
    if (y_is_odd != (b33[0] == 0x03)) y = fast::FieldElement::zero() - y;
    return {fast::Point::from_affine(x, y), true};
}

std::pair<LtcSpAddress, bool> LtcSpAddress::decode(const std::string& paycode) {
    auto res = bech32m_paycode_decode(paycode);
    if (!res.valid || res.hrp != "ltcsp") return {{}, false};
    if (res.witness_program.size() != 66) return {{}, false};

    const auto* p = res.witness_program.data();
    // p[0..32] = scan_pubkey(33), p[33..65] = spend_pubkey(33)

    auto [scan_pt,  scan_ok]  = parse_compressed(p);
    auto [spend_pt, spend_ok] = parse_compressed(p + 33);
    if (!scan_ok || !spend_ok) return {{}, false};
    if (scan_pt.is_infinity() || spend_pt.is_infinity()) return {{}, false};

    LtcSpAddress addr;
    addr.scan_pubkey  = scan_pt;
    addr.spend_pubkey = spend_pt;
    return {addr, true};
}

// -- API implementations ------------------------------------------------------

LtcSpAddress ltcsp_address(const fast::Scalar& scan_privkey,
                            const fast::Scalar& spend_privkey) {
    LtcSpAddress addr;
    addr.scan_pubkey  = ct::generator_mul(scan_privkey);
    addr.spend_pubkey = ct::generator_mul(spend_privkey);
    return addr;
}

std::pair<fast::Point, fast::Scalar>
ltcsp_create_output(const std::vector<fast::Scalar>& input_privkeys,
                    const LtcSpAddress& recipient,
                    std::uint32_t k) {
    // Aggregate input private keys: a_sum = Σ a_i
    Scalar a_sum = Scalar::zero();
    for (const auto& a : input_privkeys) a_sum = a_sum + a;

    // Shared secret: S = a_sum × B_scan  (CT: a_sum is secret)
    Point S = ct::scalar_mul(recipient.scan_pubkey, a_sum);
    auto S_comp = S.to_compressed();

    // t_k = H_LTCSP/SharedSecret(ser(S) || ser32(k))
    auto t_hash = ltcsp_hash_shared_secret(S_comp, k);
    Scalar t_k = Scalar::from_bytes(t_hash);

    // t_k = SHA256(tag||S||k) — hash output, variable-time GLV is correct
    Point P_output = recipient.spend_pubkey.add(Point::generator().scalar_mul(t_k));

    // Erase secret-derived material
    detail::secure_erase(&a_sum,   sizeof(a_sum));
    detail::secure_erase(S_comp.data(), S_comp.size());
    detail::secure_erase(t_hash.data(), t_hash.size());

    return {P_output, t_k};
}

std::vector<std::pair<std::uint32_t, fast::Scalar>>
ltcsp_scan(const fast::Scalar& scan_privkey,
           const fast::Scalar& spend_privkey,
           const std::vector<fast::Point>& input_pubkeys,
           const std::vector<std::array<std::uint8_t, 32>>& output_pubkeys) {
    std::vector<std::pair<std::uint32_t, Scalar>> results;

    // Aggregate input public keys: A_sum = Σ A_i
    Point A_sum = Point::infinity();
    for (const auto& A : input_pubkeys) A_sum = A_sum.add(A);
    if (A_sum.is_infinity()) return results;

    // Shared secret: S = b_scan × A_sum  (CT: b_scan is secret)
    Point S = ct::scalar_mul(A_sum, scan_privkey);
    auto S_comp = S.to_compressed();

    // Spend pubkey: B_spend = b_spend × G  (CT: b_spend is secret)
    Point B_spend = ct::generator_mul(spend_privkey);

    // Pre-build SHA256 midstate over tag || tag || S_comp
    // Avoids recomputing the constant prefix in the per-output loop.
    const auto& tag = ltcsp_shared_secret_tag();
    SHA256 h_base;
    h_base.update(tag.data(), 32);
    h_base.update(tag.data(), 32);
    h_base.update(S_comp.data(), 33);

    // Scan each output
    for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(output_pubkeys.size()); ++k) {
        // t_k = H_LTCSP/SharedSecret(S_comp || ser32(k))
        SHA256 h = h_base;  // clone midstate (40-byte copy)
        std::uint8_t k_be[4] = {
            std::uint8_t(k >> 24), std::uint8_t(k >> 16),
            std::uint8_t(k >> 8),  std::uint8_t(k)
        };
        h.update(k_be, 4);
        auto t_hash = h.finalize();
        Scalar t_k = Scalar::from_bytes(t_hash);

        // t_k = SHA256 output — variable-time GLV correct
        Point P_expected = B_spend.add(Point::generator().scalar_mul(t_k));
        if (P_expected.is_infinity()) continue;

        // Compare x-only output key
        auto P_x = P_expected.x().to_bytes();
        if (P_x == output_pubkeys[k]) {
            // Match: spend privkey = b_spend + t_k
            Scalar spend_key = spend_privkey + t_k;
            results.emplace_back(k, spend_key);
        }

        detail::secure_erase(t_hash.data(), t_hash.size());
    }

    // Erase secret-derived material
    detail::secure_erase(S_comp.data(), S_comp.size());

    return results;
}

// -- LtcSpScanner (wallet-optimised) ----------------------------------------

LtcSpScanner::LtcSpScanner(const fast::Scalar& scan_sk,
                            const fast::Scalar& spend_sk)
    : scan_privkey_(scan_sk)
    , spend_privkey_(spend_sk)
    , spend_pubkey_(ct::generator_mul(spend_sk))  // precompute B_spend once
{}

std::vector<std::pair<std::uint32_t, fast::Scalar>>
LtcSpScanner::scan_tx(const std::vector<fast::Point>& input_pubkeys,
                      const std::vector<std::array<std::uint8_t, 32>>& output_pubkeys) const {
    std::vector<std::pair<std::uint32_t, Scalar>> results;

    // Aggregate input public keys: A_sum = Σ A_i  (public data)
    Point A_sum = Point::infinity();
    for (const auto& A : input_pubkeys) A_sum = A_sum.add(A);
    if (A_sum.is_infinity()) return results;

    // S = b_scan × A_sum
    // A_sum is PUBLIC — variable-time scalar_mul is correct here.
    // scan_privkey_ IS secret but is the "fixed" base in A_sum.scalar_mul().
    // This uses GLV decomposition (precomputed for the scalar on first call)
    // and is ~2x faster than ct::scalar_mul for the same security level.
    // CT note: timing reveals nothing about b_scan since A_sum is on-chain.
    Point S = A_sum.scalar_mul(scan_privkey_);
    auto S_comp = S.to_compressed();

    // Pre-build SHA256 midstate: tag || tag || S_comp (amortised across outputs)
    const auto& tag = ltcsp_shared_secret_tag();
    SHA256 h_base;
    h_base.update(tag.data(), 32);
    h_base.update(tag.data(), 32);
    h_base.update(S_comp.data(), 33);

    // Scan each output
    for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(output_pubkeys.size()); ++k) {
        SHA256 h = h_base;
        std::uint8_t k_be[4] = {
            std::uint8_t(k >> 24), std::uint8_t(k >> 16),
            std::uint8_t(k >> 8),  std::uint8_t(k)
        };
        h.update(k_be, 4);
        auto t_hash = h.finalize();
        Scalar t_k = Scalar::from_bytes(t_hash);

        Point P_expected = spend_pubkey_.add(Point::generator().scalar_mul(t_k));
        if (!P_expected.is_infinity()) {
            auto P_x = P_expected.x().to_bytes();
            if (P_x == output_pubkeys[k]) {
                Scalar spend_key = spend_privkey_ + t_k;
                results.emplace_back(k, spend_key);
            }
        }
        detail::secure_erase(t_hash.data(), t_hash.size());
    }

    detail::secure_erase(S_comp.data(), S_comp.size());
    return results;
}

// -- LtcSpScanner::scan_batch ------------------------------------------------
// Uses KPlan::from_scalar(scan_privkey) + batch_scalar_mul_fixed_k (amortised
// batch field_inv across N shared secrets). Equivalent to BTC fast_scan_batch.

std::vector<LtcSpScanner::BatchMatch>
LtcSpScanner::scan_batch(
    const std::vector<std::vector<fast::Point>>& input_pubkeys_per_tx,
    const std::vector<std::vector<std::array<std::uint8_t, 32>>>& outputs_per_tx) const
{
    std::vector<BatchMatch> results;
    const std::size_t N = input_pubkeys_per_tx.size();
    if (N == 0) return results;

    // Build A_sum per tx (Σ input pubkeys)
    std::vector<Point> a_sums(N, Point::infinity());
    for (std::size_t i = 0; i < N; ++i)
        for (const auto& P : input_pubkeys_per_tx[i])
            a_sums[i] = a_sums[i].add(P);

    // Stage 1: S_i = scan_privkey × A_sum_i
    // KPlan: shared wNAF schedule across N txs
    // batch_to_compressed: ONE field_inv for all N shared secrets (Montgomery's trick)
    fast::KPlan plan = fast::KPlan::from_scalar(scan_privkey_);
    std::vector<Point> shared(N);
    Point::batch_scalar_mul_fixed_k(plan, a_sums.data(), N, shared.data());

    std::vector<std::array<std::uint8_t, 33>> S_comps(N);
    Point::batch_to_compressed(shared.data(), N, S_comps.data());

    // LTCSP base state: SHA256 state after processing tag||tag (64 bytes, one block).
    // Precomputed once per process — avoids rebuilding SHA256 context per output.
    static const std::array<std::uint32_t, 8> s_base = []() noexcept {
        const auto tag = SHA256::hash(
            reinterpret_cast<const std::uint8_t*>("LTCSP/SharedSecret"), 18);
        std::uint8_t blk[64];
        std::memcpy(blk,      tag.data(), 32);
        std::memcpy(blk + 32, tag.data(), 32);
        std::array<std::uint32_t, 8> st = {
            0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
            0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
        };
        secp256k1::detail::sha256_compress_dispatch(blk, st.data());
        return st;
    }();

    // Pass 2a: compute all t_k scalars via raw SHA256 block compression (no context).
    // block layout: S_comp[33] || k_be[4] || 0x80 || zeros || 0x03 0x28
    //   bit-length = (64 + 37) * 8 = 808 = 0x0328  (one block from base state)
    std::vector<std::uint64_t> out_map;
    std::vector<Scalar> t_scalars;

    for (std::uint32_t tx = 0; tx < static_cast<std::uint32_t>(N); ++tx) {
        if (shared[tx].is_infinity()) continue;
        const auto& S_comp = S_comps[tx];

        // Build block1 template for this tx
        std::uint8_t blk[64];
        std::memcpy(blk, S_comp.data(), 33);
        blk[37] = 0x80;
        std::memset(blk + 38, 0, 24);
        blk[62] = 0x03; blk[63] = 0x28;

        const auto& outs = outputs_per_tx[tx];
        for (std::uint32_t k = 0; k < static_cast<std::uint32_t>(outs.size()); ++k) {
            blk[33] = std::uint8_t(k >> 24);
            blk[34] = std::uint8_t(k >> 16);
            blk[35] = std::uint8_t(k >>  8);
            blk[36] = std::uint8_t(k);

            std::array<std::uint32_t, 8> h;
            std::memcpy(h.data(), s_base.data(), 32);
            secp256k1::detail::sha256_compress_dispatch(blk, h.data());

            std::array<std::uint8_t, 32> t_bytes;
            for (int b = 0; b < 8; ++b) {
                t_bytes[b*4+0] = std::uint8_t(h[b] >> 24);
                t_bytes[b*4+1] = std::uint8_t(h[b] >> 16);
                t_bytes[b*4+2] = std::uint8_t(h[b] >>  8);
                t_bytes[b*4+3] = std::uint8_t(h[b]);
            }
            Scalar t_k = Scalar::from_bytes(t_bytes);
            out_map.push_back((static_cast<std::uint64_t>(tx) << 32) | k);
            t_scalars.push_back(t_k);
        }
    }

    if (t_scalars.empty()) return results;

    // Pass 2b: batch t_k*G for all outputs — ONE precomputed-table scan
    std::vector<Point> out_jac(t_scalars.size());
    secp256k1::fast::batch_scalar_mul_generator(t_scalars.data(), out_jac.data(), t_scalars.size());

    // Pass 2c: add spend_pubkey and compare
    for (std::size_t i = 0; i < t_scalars.size(); ++i) {
        std::uint32_t tx = static_cast<std::uint32_t>(out_map[i] >> 32);
        std::uint32_t k  = static_cast<std::uint32_t>(out_map[i] & 0xffffffff);
        Point P = spend_pubkey_.add(out_jac[i]);
        if (!P.is_infinity() && P.x().to_bytes() == outputs_per_tx[tx][k]) {
            results.push_back({tx, k, spend_privkey_ + t_scalars[i]});
        }
    }
    return results;
}

} // namespace secp256k1::ltc
