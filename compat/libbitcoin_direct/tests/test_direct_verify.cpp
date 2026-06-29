// Standalone correctness test for the direct libbitcoin integration header
// (ufsecp/libbitcoin.hpp): ECDSA + Schnorr verify, single + batch, fail-closed.
// Build: g++ -O2 -std=c++20 -I<compat/libbitcoin_direct/include> -I<src/cpu/include>
//        test_direct_verify.cpp <engine libs> -pthread
// Returns 0 on success, 1 on any failure.
//
// NOTE: Test-data generation uses CT-backed ufsecp::lbtc::* entrypoints so the
// build emits no deprecated non-CT signing/keypair warnings from this file.
#include "ufsecp/libbitcoin.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/schnorr.hpp"
#include "secp256k1/batch_verify.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {
std::uint64_t g_xs = 0x9E3779B97F4A7C15ull;
std::uint8_t nb() { g_xs ^= g_xs << 13; g_xs ^= g_xs >> 7; g_xs ^= g_xs << 17; return static_cast<std::uint8_t>(g_xs); }
int fails = 0;
void check(bool cond, const char* what) { if (!cond) { std::printf("FAIL: %s\n", what); ++fails; } }

// Generate a valid random secret key (CT-backed via ufsecp::lbtc::seckey_verify).
void rand_sk(std::uint8_t sk[32]) {
    do {
        for (int i = 0; i < 32; ++i) sk[i] = nb();
    } while (!ufsecp::lbtc::seckey_verify(sk));
}

// Generate a valid random message hash.
void rand_hash(std::uint8_t h[32]) {
    for (int i = 0; i < 32; ++i) h[i] = nb();
}
} // namespace

int main() {
    constexpr int M = 20000;
    constexpr int SERIAL_N = 5000;  // Crosses the 4096 bounded-chunk threshold.

    // ---- ECDSA: rows [hash32 | pub33 | sig64(opaque LE)] ----
    // Test-data generation uses CT-backed ufsecp::lbtc entrypoints.
    constexpr int ES = 129;
    std::vector<std::uint8_t> erows(M * ES);
    for (int i = 0; i < M; ++i) {
        std::uint8_t sk[32], msg[32], sig64[64], pub33[33];
        rand_sk(sk);
        rand_hash(msg);
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "ecdsa test-data pubkey_create");  // CT-backed
        check(ufsecp::lbtc::ecdsa_sign(msg, sk, sig64), "ecdsa test-data ecdsa_sign");  // CT-backed
        std::uint8_t* r = erows.data() + i * ES;
        std::memcpy(r, msg, 32);
        std::memcpy(r + 32, pub33, 33);
        std::memcpy(r + 65, sig64, 64);
    }
    long eok = 0;
    for (int i = 0; i < M; ++i) { const std::uint8_t* r = erows.data() + i * ES; eok += ufsecp::lbtc::ecdsa_verify(r + 32, r, r + 65); }
    check(eok == M, "ecdsa single verify all-valid");
    std::vector<std::uint8_t> eres(M, 0);
    check(ufsecp::lbtc::ecdsa_verify_batch(erows.data(), ES, M, eres.data(), 0), "ecdsa batch all-valid");
    { long ok = 0; for (auto v : eres) ok += v; check(ok == M, "ecdsa batch per-row results all-valid"); }
    std::vector<std::uint8_t> eres_engine(SERIAL_N, 0);
    check(secp256k1::ecdsa_batch_verify_opaque_rows(erows.data(), ES, SERIAL_N, eres_engine.data(), 1),
          "engine ecdsa opaque rows serial all-valid");
    { long ok = 0; for (auto v : eres_engine) ok += v; check(ok == SERIAL_N, "engine ecdsa rows serial per-row all-valid"); }
    // ECDSA columns (Structure-of-Arrays) view of the same valid data.
    {
        std::vector<std::uint8_t> cd(M * 32), cp(M * 33), cs(M * 64);
        for (int i = 0; i < M; ++i) {
            const std::uint8_t* r = erows.data() + i * ES;
            std::memcpy(cd.data() + i * 32, r, 32);
            std::memcpy(cp.data() + i * 33, r + 32, 33);
            std::memcpy(cs.data() + i * 64, r + 65, 64);
        }
        std::vector<std::uint8_t> cr(M, 0);
        check(ufsecp::lbtc::ecdsa_verify_columns(cd.data(), cp.data(), cs.data(), M, cr.data(), 0),
              "ecdsa columns all-valid");
        long ok = 0; for (auto v : cr) ok += v; check(ok == M, "ecdsa columns per-row all-valid");
        std::vector<std::uint8_t> cr_engine(SERIAL_N, 0);
        check(secp256k1::ecdsa_batch_verify_opaque_columns(cd.data(), cp.data(), cs.data(), SERIAL_N, cr_engine.data(), 1),
              "engine ecdsa opaque columns serial all-valid");
        long ok_engine = 0; for (auto v : cr_engine) ok_engine += v;
        check(ok_engine == SERIAL_N, "engine ecdsa columns serial per-row all-valid");
        cs[5 * 64] ^= 1;  // tamper row 5
        std::vector<std::uint8_t> cr2(M, 0);
        check(!ufsecp::lbtc::ecdsa_verify_columns(cd.data(), cp.data(), cs.data(), M, cr2.data(), 0),
              "ecdsa columns fail-closed on tamper");
        check(cr2[5] == 0, "ecdsa columns tampered row marked invalid");
        std::vector<std::uint8_t> cr3(SERIAL_N, 0);
        check(!secp256k1::ecdsa_batch_verify_opaque_columns(cd.data(), cp.data(), cs.data(), SERIAL_N, cr3.data(), 1),
              "engine ecdsa columns serial fail-closed on tamper");
        check(cr3[5] == 0, "engine ecdsa columns tampered row marked invalid");
    }
    erows[ES * 7 + 65] ^= 1;  // tamper row 7
    std::vector<std::uint8_t> eres2(M, 0);
    check(!ufsecp::lbtc::ecdsa_verify_batch(erows.data(), ES, M, eres2.data(), 0), "ecdsa batch fail-closed on tamper");
    check(eres2[7] == 0, "ecdsa tampered row marked invalid");
    std::vector<std::uint8_t> eres3(SERIAL_N, 0);
    check(!secp256k1::ecdsa_batch_verify_opaque_rows(erows.data(), ES, SERIAL_N, eres3.data(), 1),
          "engine ecdsa rows serial fail-closed on tamper");
    check(eres3[7] == 0, "engine ecdsa rows tampered row marked invalid");

    // ---- Schnorr: rows [msg32 | xonly32 | sig64(BIP-340)] ----
    // Test-data generation uses CT-backed ufsecp::lbtc entrypoints.
    constexpr int SS = 128;
    std::vector<std::uint8_t> srows(M * SS);
    std::uint8_t aux[32]{};
    for (int i = 0; i < M; ++i) {
        std::uint8_t sk[32], msg[32], xonly[32], sig64[64];
        rand_sk(sk);
        rand_hash(msg);
        check(ufsecp::lbtc::schnorr_keypair_create(sk, xonly), "schnorr test-data keypair_create");  // CT-backed
        check(ufsecp::lbtc::schnorr_sign(xonly, sk, msg, aux, sig64), "schnorr test-data sign");  // CT-backed
        std::uint8_t* r = srows.data() + i * SS;
        std::memcpy(r, msg, 32);
        std::memcpy(r + 32, xonly, 32);
        std::memcpy(r + 64, sig64, 64);
    }
    long sok = 0;
    for (int i = 0; i < M; ++i) { const std::uint8_t* r = srows.data() + i * SS; sok += ufsecp::lbtc::schnorr_verify(r + 32, r, r + 64); }
    check(sok == M, "schnorr single verify all-valid");
    std::vector<std::uint8_t> sres(M, 0);
    check(ufsecp::lbtc::schnorr_verify_batch(srows.data(), SS, M, sres.data(), 0), "schnorr batch all-valid");
    { long ok = 0; for (auto v : sres) ok += v; check(ok == M, "schnorr batch per-row results all-valid"); }
    std::vector<std::uint8_t> sres_engine(SERIAL_N, 0);
    check(secp256k1::schnorr_batch_verify_bip340_rows(srows.data(), SS, SERIAL_N, sres_engine.data(), 1),
          "engine schnorr bip340 rows serial all-valid");
    { long ok = 0; for (auto v : sres_engine) ok += v; check(ok == SERIAL_N, "engine schnorr rows serial per-row all-valid"); }
    // Schnorr columns (Structure-of-Arrays) view of the same valid data.
    {
        std::vector<std::uint8_t> cd(M * 32), cx(M * 32), cs(M * 64);
        for (int i = 0; i < M; ++i) {
            const std::uint8_t* r = srows.data() + i * SS;
            std::memcpy(cd.data() + i * 32, r, 32);
            std::memcpy(cx.data() + i * 32, r + 32, 32);
            std::memcpy(cs.data() + i * 64, r + 64, 64);
        }
        std::vector<std::uint8_t> cr(M, 0);
        check(ufsecp::lbtc::schnorr_verify_columns(cd.data(), cx.data(), cs.data(), M, cr.data(), 0),
              "schnorr columns all-valid");
        long ok = 0; for (auto v : cr) ok += v; check(ok == M, "schnorr columns per-row all-valid");
        std::vector<std::uint8_t> cr_engine(SERIAL_N, 0);
        check(secp256k1::schnorr_batch_verify_bip340_columns(cd.data(), cx.data(), cs.data(), SERIAL_N, cr_engine.data(), 1),
              "engine schnorr bip340 columns serial all-valid");
        long ok_engine = 0; for (auto v : cr_engine) ok_engine += v;
        check(ok_engine == SERIAL_N, "engine schnorr columns serial per-row all-valid");
        cs[9 * 64] ^= 1;  // tamper row 9 in the column view only
        std::vector<std::uint8_t> cr2(M, 0);
        check(!ufsecp::lbtc::schnorr_verify_columns(cd.data(), cx.data(), cs.data(), M, cr2.data(), 0),
              "schnorr columns fail-closed on tamper");
        check(cr2[9] == 0, "schnorr columns tampered row marked invalid");
        std::vector<std::uint8_t> cr3(SERIAL_N, 0);
        check(!secp256k1::schnorr_batch_verify_bip340_columns(cd.data(), cx.data(), cs.data(), SERIAL_N, cr3.data(), 1),
              "engine schnorr columns serial fail-closed on tamper");
        check(cr3[9] == 0, "engine schnorr columns tampered row marked invalid");
    }
    srows[SS * 3 + 64] ^= 1;  // tamper row 3
    std::vector<std::uint8_t> sres2(M, 0);
    check(!ufsecp::lbtc::schnorr_verify_batch(srows.data(), SS, M, sres2.data(), 0), "schnorr batch fail-closed on tamper");
    check(sres2[3] == 0, "schnorr tampered row marked invalid");
    std::vector<std::uint8_t> sres3(SERIAL_N, 0);
    check(!secp256k1::schnorr_batch_verify_bip340_rows(srows.data(), SS, SERIAL_N, sres3.data(), 1),
          "engine schnorr rows serial fail-closed on tamper");
    check(sres3[3] == 0, "engine schnorr rows tampered row marked invalid");

    std::vector<std::uint8_t> bad_results(4, 1);
    check(!secp256k1::ecdsa_batch_verify_opaque_rows(nullptr, ES, bad_results.size(), bad_results.data(), 1),
          "engine ecdsa rows null input fails closed");
    { long ok = 0; for (auto v : bad_results) ok += v; check(ok == 0, "engine ecdsa null input zeroes results"); }
    std::fill(bad_results.begin(), bad_results.end(), 1);
    check(!secp256k1::schnorr_batch_verify_bip340_rows(srows.data(), 64, bad_results.size(), bad_results.data(), 1),
          "engine schnorr rows short stride fails closed");
    { long ok = 0; for (auto v : bad_results) ok += v; check(ok == 0, "engine schnorr short stride zeroes results"); }

    if (fails == 0) std::printf("test_direct_verify: ALL PASS (ecdsa+schnorr single+batch+fail-closed)\n");
    return fails == 0 ? 0 : 1;
}
