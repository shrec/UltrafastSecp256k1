// Minimal bridge-free libbitcoin direct public-data batch-op example.
//
// Build via:
//   -DSECP256K1_BUILD_LIBBITCOIN=ON
//   -DSECP256K1_BUILD_LIBBITCOIN_EXAMPLES=ON
//
// This links only secp256k1::fastsecp256k1_libbitcoin + the engine. No C ABI,
// no libsecp256k1 shim, no ufsecp_lbtc bridge.
#include "ufsecp/libbitcoin.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

void die_if(bool bad, const char* what)
{
    if (bad) {
        std::printf("FAIL: %s\n", what);
        std::exit(1);
    }
}

void make_sk(std::uint8_t sk[32], std::uint8_t v)
{
    std::memset(sk, 0, 32);
    sk[31] = v;
    die_if(!ufsecp::lbtc::seckey_verify(sk), "fixture secret key");
}

void hex4(const std::uint8_t h[32])
{
    std::printf("%02x%02x%02x%02x", h[0], h[1], h[2], h[3]);
}

} // namespace

int main()
{
    constexpr std::size_t n = 4;
    std::vector<std::uint8_t> xonly(n * 32);
    std::vector<std::uint8_t> pub33(n * 33);
    std::vector<std::uint8_t> internal_x(n * 32);
    std::vector<std::uint8_t> tweak(n * 32);
    std::vector<std::uint8_t> tweaked_x(n * 32);
    std::vector<std::uint8_t> parity(n);
    std::vector<std::uint8_t> results(n);

    for (std::size_t i = 0; i < n; ++i) {
        std::uint8_t sk[32];
        make_sk(sk, static_cast<std::uint8_t>(i + 1));
        die_if(!ufsecp::lbtc::schnorr_keypair_create(sk, xonly.data() + i * 32),
               "schnorr_keypair_create");
        die_if(!ufsecp::lbtc::pubkey_create(sk, pub33.data() + i * 33),
               "pubkey_create");
        std::memcpy(internal_x.data() + i * 32, xonly.data() + i * 32, 32);

        std::memset(tweak.data() + i * 32, 0, 32);
        tweak[i * 32 + 31] = static_cast<std::uint8_t>(10 + i);

        secp256k1::SchnorrXonlyPubkey xp;
        die_if(!secp256k1::schnorr_xonly_pubkey_parse(xp, internal_x.data() + i * 32),
               "lift_x internal key");
        const auto t = secp256k1::fast::Scalar::from_bytes(tweak.data() + i * 32);
        const auto q = secp256k1::fast::Point::dual_scalar_mul_gen_point(
            t, secp256k1::fast::Scalar::one(), xp.point);
        die_if(q.is_infinity(), "taproot tweak produced infinity");
        const auto comp = q.to_compressed();
        std::memcpy(tweaked_x.data() + i * 32, comp.data() + 1, 32);
        parity[i] = comp[0] == 0x03 ? 1 : 0;
    }

    die_if(!ufsecp::lbtc::xonly_validate_batch(xonly.data(), n, results.data()),
           "xonly_validate_batch");
    die_if(!ufsecp::lbtc::pubkey_validate_batch(pub33.data(), n, results.data()),
           "pubkey_validate_batch");
    die_if(!ufsecp::lbtc::taproot_commitment_verify_batch(
               internal_x.data(), tweak.data(), tweaked_x.data(), parity.data(),
               n, results.data()),
           "taproot_commitment_verify_batch");

    constexpr std::size_t fixed_len = 80;
    std::vector<std::uint8_t> fixed_msgs(n * fixed_len);
    for (std::size_t i = 0; i < fixed_msgs.size(); ++i)
        fixed_msgs[i] = static_cast<std::uint8_t>(i * 17 + 3);

    std::vector<std::uint8_t> out(n * 32);
    const auto tag_hash = secp256k1::SHA256::hash("BIP0340/test", 12);
    die_if(!ufsecp::lbtc::tagged_hash_batch(
               tag_hash.data(), fixed_msgs.data(), fixed_len, n, out.data()),
           "tagged_hash_batch");
    std::printf("tagged_hash[0]="); hex4(out.data()); std::printf("...\n");

    die_if(!ufsecp::lbtc::tagged_hash_batch(
               "BIP0340/test", 12, fixed_msgs.data(), fixed_len, n, out.data()),
           "tagged_hash_batch tag overload");

    constexpr std::size_t stride = 128;
    std::vector<std::uint8_t> var_msgs(n * stride);
    std::vector<std::uint32_t> lens(n);
    for (std::size_t i = 0; i < n; ++i) {
        lens[i] = static_cast<std::uint32_t>(20 + i * 11);
        for (std::size_t j = 0; j < stride; ++j)
            var_msgs[i * stride + j] = static_cast<std::uint8_t>(i * 29 + j);
    }
    die_if(!ufsecp::lbtc::tagged_hash_var_batch(
               tag_hash.data(), var_msgs.data(), lens.data(), stride, n, out.data()),
           "tagged_hash_var_batch");

    die_if(!ufsecp::lbtc::hash256_batch(
               fixed_msgs.data(), fixed_len, n, out.data()),
           "hash256_batch");
    std::printf("hash256[0]="); hex4(out.data()); std::printf("...\n");

    die_if(!ufsecp::lbtc::hash256_var_batch(
               var_msgs.data(), lens.data(), stride, n, out.data()),
           "hash256_var_batch");
    std::printf("hash256_var[0]="); hex4(out.data()); std::printf("...\n");

    std::printf("example_lbtc_public_ops: PASS (direct C++, no C ABI/shim/bridge)\n");
    return 0;
}
