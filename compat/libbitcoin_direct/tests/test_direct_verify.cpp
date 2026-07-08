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
#include "secp256k1/sha256.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
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

// Independent BIP-340 tagged-hash reference (recomputed with raw SHA256, so it
// cross-checks the batch surface rather than reusing its helper).
void ref_tagged_hash(const std::uint8_t th[32], const std::uint8_t* msg,
                     std::size_t len, std::uint8_t out[32]) {
    secp256k1::SHA256 ctx;
    ctx.update(th, 32);
    ctx.update(th, 32);
    if (len != 0) ctx.update(msg, len);
    const auto d = ctx.finalize();
    std::memcpy(out, d.data(), 32);
}

// Independent Bitcoin HASH256 reference.
void ref_hash256(const std::uint8_t* in, std::size_t len, std::uint8_t out[32]) {
    const auto d = secp256k1::SHA256::hash256(in, len);
    std::memcpy(out, d.data(), 32);
}

static constexpr std::uint8_t kScalarOrder[32] = {
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xfe,
    0xba, 0xae, 0xdc, 0xe6, 0xaf, 0x48, 0xa0, 0x3b,
    0xbf, 0xd2, 0x5e, 0x8c, 0xd0, 0x36, 0x41, 0x41
};

static constexpr std::uint8_t kScalarHalfOrder[32] = {
    0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
    0x5d, 0x57, 0x6e, 0x73, 0x57, 0xa4, 0x50, 0x1d,
    0xdf, 0xe9, 0x2f, 0x46, 0x68, 0x1b, 0x20, 0xa0
};

int cmp32_be(const std::uint8_t* a, const std::uint8_t* b) {
    for (std::size_t i = 0; i < 32; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

void scalar_order_minus(std::uint8_t out[32], const std::uint8_t x[32]) {
    std::uint16_t borrow = 0;
    for (int i = 31; i >= 0; --i) {
        const std::uint16_t lhs = kScalarOrder[static_cast<std::size_t>(i)];
        const std::uint16_t rhs = static_cast<std::uint16_t>(x[i]) + borrow;
        if (lhs < rhs) {
            out[i] = static_cast<std::uint8_t>(lhs + 256u - rhs);
            borrow = 1;
        } else {
            out[i] = static_cast<std::uint8_t>(lhs - rhs);
            borrow = 0;
        }
    }
}

bool ecdsa_s_is_high(const std::uint8_t* sig64_be) {
    return cmp32_be(sig64_be + 32, kScalarOrder) < 0 &&
           cmp32_be(sig64_be + 32, kScalarHalfOrder) > 0;
}

void scalar_be_to_opaque(std::uint8_t out[32], const std::uint8_t in[32]) {
    for (std::size_t i = 0; i < 32; ++i) out[i] = in[31 - i];
}

void scalar_opaque_to_be(std::uint8_t out[32], const std::uint8_t in[32]) {
    for (std::size_t i = 0; i < 32; ++i) out[i] = in[31 - i];
}

void lbtc_opaque_to_compact(const std::uint8_t sig64[64],
                            std::uint8_t out[64]) {
    scalar_opaque_to_be(out, sig64);
    scalar_opaque_to_be(out + 32, sig64 + 32);
}

void compact_to_lbtc_opaque(std::uint8_t sig64[64]) {
    std::uint8_t tmp[64];
    scalar_be_to_opaque(tmp, sig64);
    scalar_be_to_opaque(tmp + 32, sig64 + 32);
    std::memcpy(sig64, tmp, 64);
}

bool lbtc_ecdsa_s_is_high(const std::uint8_t sig64[64]) {
    std::uint8_t compact[64];
    lbtc_opaque_to_compact(sig64, compact);
    return ecdsa_s_is_high(compact);
}

void make_high_s(std::uint8_t sig64_be[64]) {
    std::uint8_t high_s[32];
    scalar_order_minus(high_s, sig64_be + 32);
    std::memcpy(sig64_be + 32, high_s, 32);
}

void make_lbtc_high_s(std::uint8_t sig64[64]) {
    std::uint8_t compact[64];
    lbtc_opaque_to_compact(sig64, compact);
    make_high_s(compact);
    compact_to_lbtc_opaque(compact);
    std::memcpy(sig64, compact, 64);
}

// Fake GPU hook for hash256_var_batch: writes a fixed sentinel pattern and
// reports "handled" (0). Used only to prove the wrapper actually consults
// the installed hook instead of silently falling through to the CPU path.
int fake_hash256_var_hook(const std::uint8_t*, const std::uint32_t*, std::size_t,
                          std::size_t count, std::uint8_t* out32) {
    std::memset(out32, 0xEE, count * 32);
    return 0;
}

// Fake GPU hook for merkle_pair_hash_batch: writes a fixed sentinel pattern
// and reports "handled" (0). Used only to prove the wrapper actually
// consults the installed hook instead of silently falling through to CPU.
int fake_merkle_pair_hook(const std::uint8_t*, const std::uint8_t*, std::size_t count,
                          std::uint8_t* out32) {
    std::memset(out32, 0xDD, count * 32);
    return 0;
}

// Fake GPU hook for merkle_pair_hash_batch that always declines (-1),
// forcing the CPU fallback loop to run regardless of what a real production
// hook (if any is installed by a direct-GPU build) would otherwise do.
int decline_merkle_pair_hook(const std::uint8_t*, const std::uint8_t*, std::size_t,
                             std::uint8_t*) {
    return -1;
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
    // High-S ECDSA signatures are consensus-valid in Bitcoin. The direct
    // libbitcoin surface must not treat low-S as a strict validity rule; low-S
    // normalization/standardness remains a separate policy layer.
    {
        constexpr int HS = 11;
        constexpr int BAD = 5;
        std::vector<std::uint8_t> high_rows = erows;
        std::uint8_t* hs_row = high_rows.data() + HS * ES;
        make_lbtc_high_s(hs_row + 65);
        check(lbtc_ecdsa_s_is_high(hs_row + 65), "ecdsa high-S fixture constructed");
        check(ufsecp::lbtc::ecdsa_verify(hs_row + 32, hs_row, hs_row + 65),
              "ecdsa single verify accepts consensus-valid high-S");

        std::vector<std::uint8_t> hr(M, 0);
        check(ufsecp::lbtc::ecdsa_verify_batch(high_rows.data(), ES, M, hr.data(), 0),
              "ecdsa batch accepts consensus-valid high-S");
        check(hr[HS] == 1, "ecdsa batch high-S row marked valid");
        std::vector<std::uint8_t> hr_engine(SERIAL_N, 0);
        check(secp256k1::ecdsa_batch_verify_opaque_rows(
                  high_rows.data(), ES, SERIAL_N, hr_engine.data(), 1),
              "engine ecdsa opaque rows accept consensus-valid high-S");
        check(hr_engine[HS] == 1, "engine ecdsa rows high-S row marked valid");

        std::vector<std::uint8_t> cd(M * 32), cp(M * 33), cs(M * 64);
        for (int i = 0; i < M; ++i) {
            const std::uint8_t* r = high_rows.data() + i * ES;
            std::memcpy(cd.data() + i * 32, r, 32);
            std::memcpy(cp.data() + i * 33, r + 32, 33);
            std::memcpy(cs.data() + i * 64, r + 65, 64);
        }
        std::vector<std::uint8_t> hc(M, 0);
        check(ufsecp::lbtc::ecdsa_verify_columns(cd.data(), cp.data(), cs.data(), M, hc.data(), 0),
              "ecdsa columns accept consensus-valid high-S");
        check(hc[HS] == 1, "ecdsa columns high-S row marked valid");
        std::vector<std::uint8_t> hc_engine(SERIAL_N, 0);
        check(secp256k1::ecdsa_batch_verify_opaque_columns(
                  cd.data(), cp.data(), cs.data(), SERIAL_N, hc_engine.data(), 1),
              "engine ecdsa opaque columns accept consensus-valid high-S");
        check(hc_engine[HS] == 1, "engine ecdsa columns high-S row marked valid");

        cs[BAD * 64] ^= 1;
        std::vector<std::uint8_t> hm(M, 0);
        check(!ufsecp::lbtc::ecdsa_verify_columns(cd.data(), cp.data(), cs.data(), M, hm.data(), 0),
              "ecdsa columns high-S remains valid beside tamper");
        check(hm[BAD] == 0, "ecdsa columns tampered row invalid with high-S present");
        check(hm[HS] == 1, "ecdsa columns high-S row still valid beside tamper");
        std::vector<std::uint8_t> hm_engine(SERIAL_N, 0);
        check(!secp256k1::ecdsa_batch_verify_opaque_columns(
                  cd.data(), cp.data(), cs.data(), SERIAL_N, hm_engine.data(), 1),
              "engine ecdsa columns high-S remains valid beside tamper");
        check(hm_engine[BAD] == 0, "engine ecdsa columns tampered row invalid");
        check(hm_engine[HS] == 1, "engine ecdsa columns high-S row still valid");
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

    // ════════════════════════════════════════════════════════════════════════
    // libbitcoin public-data batch ops (one surface: internal GPU accel when the
    // GPU host is linked, else deterministic CPU fallback). All PUBLIC data.
    // ════════════════════════════════════════════════════════════════════════
    constexpr std::size_t BN = 300;                 // crosses a 256-wide device tile
    const std::size_t OVF32 = (SIZE_MAX / 32) + 1;  // count*32 overflows size_t

    // ---- xonly_validate_batch ----
    {
        std::vector<std::uint8_t> keys(BN * 32);
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t sk[32], x[32];
            rand_sk(sk);
            check(ufsecp::lbtc::schnorr_keypair_create(sk, x), "xonly test-data keypair");
            std::memcpy(keys.data() + i * 32, x, 32);
        }
        std::vector<std::uint8_t> res(BN, 0);
        check(ufsecp::lbtc::xonly_validate_batch(keys.data(), BN, res.data()),
              "xonly_validate_batch all-valid true");
        { long ok = 0; for (auto v : res) ok += v; check(ok == (long)BN, "xonly per-row all 1"); }
        // GPU-vs-serial parity: max_threads has no CPU/GPU split, results are stable.
        std::vector<std::uint8_t> res_ref(BN, 0);
        for (std::size_t i = 0; i < BN; ++i)
            res_ref[i] = ufsecp::lbtc::schnorr_xonly_pubkey_parse(keys.data() + i * 32) ? 1 : 0;
        check(std::memcmp(res.data(), res_ref.data(), BN) == 0, "xonly matches serial reference");
        // Malformed row k: x >= p (all-0xFF) is not a valid x-coordinate.
        std::vector<std::uint8_t> keys2 = keys;
        std::memset(keys2.data() + 11 * 32, 0xFF, 32);
        std::vector<std::uint8_t> res2(BN, 0);
        check(!ufsecp::lbtc::xonly_validate_batch(keys2.data(), BN, res2.data()),
              "xonly fail-closed on malformed row");
        check(res2[11] == 0, "xonly malformed row marked invalid");
        check(res2[10] == 1 && res2[12] == 1, "xonly other rows still valid");
        // Null keys32 -> false, out zeroed.
        std::vector<std::uint8_t> res3(BN, 1);
        check(!ufsecp::lbtc::xonly_validate_batch(nullptr, BN, res3.data()),
              "xonly null keys fails closed");
        { long ok = 0; for (auto v : res3) ok += v; check(ok == 0, "xonly null keys zeroes out"); }
        // Null out_results -> false.
        check(!ufsecp::lbtc::xonly_validate_batch(keys.data(), BN, nullptr),
              "xonly null out fails closed");
        // count==0 -> true, out untouched.
        std::vector<std::uint8_t> res4(BN, 0xAB);
        check(ufsecp::lbtc::xonly_validate_batch(keys.data(), 0, res4.data()),
              "xonly count==0 vacuous true");
        check(res4[0] == 0xAB, "xonly count==0 leaves out untouched");
        // Hostile huge count -> overflow rejected (null out keeps it safe).
        check(!ufsecp::lbtc::xonly_validate_batch(keys.data(), OVF32, nullptr),
              "xonly huge count overflow rejected");
    }

    // ---- pubkey_validate_batch ----
    {
        std::vector<std::uint8_t> pks(BN * 33);
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t sk[32], p[33];
            rand_sk(sk);
            check(ufsecp::lbtc::pubkey_create(sk, p), "pubkey test-data create");
            std::memcpy(pks.data() + i * 33, p, 33);
        }
        std::vector<std::uint8_t> res(BN, 0);
        check(ufsecp::lbtc::pubkey_validate_batch(pks.data(), BN, res.data()),
              "pubkey_validate_batch all-valid true");
        { long ok = 0; for (auto v : res) ok += v; check(ok == (long)BN, "pubkey per-row all 1"); }
        // Wrong prefix at row k (0x00/0x04/0x05).
        std::vector<std::uint8_t> pks2 = pks;
        pks2[7 * 33] = 0x04;
        std::vector<std::uint8_t> res2(BN, 0);
        check(!ufsecp::lbtc::pubkey_validate_batch(pks2.data(), BN, res2.data()),
              "pubkey wrong-prefix fail-closed");
        check(res2[7] == 0, "pubkey wrong-prefix row invalid");
        // Off-curve x (valid prefix, x with no y root): 0x02 || all-0x03 has
        // x^3+7 a quadratic non-residue mod p, so it is off the curve. (Note:
        // all-0x01 is ON the curve, hence a poor off-curve fixture.)
        std::vector<std::uint8_t> pks3 = pks;
        pks3[13 * 33] = 0x02;
        std::memset(pks3.data() + 13 * 33 + 1, 0x03, 32);
        std::vector<std::uint8_t> res3(BN, 0);
        check(!ufsecp::lbtc::pubkey_validate_batch(pks3.data(), BN, res3.data()),
              "pubkey off-curve fail-closed");
        check(res3[13] == 0, "pubkey off-curve row invalid");
        // Null / count==0 / overflow.
        std::vector<std::uint8_t> res4(BN, 1);
        check(!ufsecp::lbtc::pubkey_validate_batch(nullptr, BN, res4.data()),
              "pubkey null in fails closed");
        { long ok = 0; for (auto v : res4) ok += v; check(ok == 0, "pubkey null zeroes out"); }
        check(!ufsecp::lbtc::pubkey_validate_batch(pks.data(), BN, nullptr),
              "pubkey null out fails closed");
        std::vector<std::uint8_t> res5(BN, 0xAB);
        check(ufsecp::lbtc::pubkey_validate_batch(pks.data(), 0, res5.data()),
              "pubkey count==0 vacuous true");
        check(res5[0] == 0xAB, "pubkey count==0 out untouched");
        check(!ufsecp::lbtc::pubkey_validate_batch(pks.data(), (SIZE_MAX / 33) + 1, nullptr),
              "pubkey huge count overflow rejected");
    }

    // ---- taproot_commitment_verify_batch (RAW tweak: Q = P + tweak*G) ----
    {
        std::vector<std::uint8_t> ix(BN * 32), tw(BN * 32), tx(BN * 32), par(BN);
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t sk[32], x[32], t[32];
            rand_sk(sk);
            check(ufsecp::lbtc::schnorr_keypair_create(sk, x), "taproot test-data internal key");
            std::memcpy(ix.data() + i * 32, x, 32);
            rand_hash(t);  // raw 32-byte tweak scalar (reduced mod n on parse)
            std::memcpy(tw.data() + i * 32, t, 32);
            // Expected Q = lift_x_even(internal) + t*G via the engine (parity cross-check).
            secp256k1::SchnorrXonlyPubkey xp;
            check(secp256k1::schnorr_xonly_pubkey_parse(xp, x), "taproot lift internal");
            const secp256k1::fast::Scalar ts = secp256k1::fast::Scalar::from_bytes(t);
            const auto Q = secp256k1::fast::Point::dual_scalar_mul_gen_point(
                ts, secp256k1::fast::Scalar::one(), xp.point);
            check(!Q.is_infinity(), "taproot Q finite");
            const auto comp = Q.to_compressed();
            std::memcpy(tx.data() + i * 32, comp.data() + 1, 32);
            par[i] = (comp[0] == 0x03) ? 1 : 0;
        }
        std::vector<std::uint8_t> res(BN, 0);
        check(ufsecp::lbtc::taproot_commitment_verify_batch(
                  ix.data(), tw.data(), tx.data(), par.data(), BN, res.data()),
              "taproot_commitment all-valid true");
        { long ok = 0; for (auto v : res) ok += v; check(ok == (long)BN, "taproot per-row all 1"); }
        // Wrong tweaked_x at row k.
        std::vector<std::uint8_t> tx2 = tx; tx2[5 * 32] ^= 1;
        std::vector<std::uint8_t> res2(BN, 0);
        check(!ufsecp::lbtc::taproot_commitment_verify_batch(
                  ix.data(), tw.data(), tx2.data(), par.data(), BN, res2.data()),
              "taproot wrong tweaked_x fail-closed");
        check(res2[5] == 0, "taproot wrong tweaked_x row invalid");
        // Correct x but wrong parity byte.
        std::vector<std::uint8_t> par2(par.begin(), par.end()); par2[9] ^= 1;
        std::vector<std::uint8_t> res3(BN, 0);
        check(!ufsecp::lbtc::taproot_commitment_verify_batch(
                  ix.data(), tw.data(), tx.data(), par2.data(), BN, res3.data()),
              "taproot wrong parity fail-closed");
        check(res3[9] == 0, "taproot wrong parity row invalid");
        // internal_x not liftable (off-curve x >= p).
        std::vector<std::uint8_t> ix2 = ix; std::memset(ix2.data() + 3 * 32, 0xFF, 32);
        std::vector<std::uint8_t> res4(BN, 0);
        check(!ufsecp::lbtc::taproot_commitment_verify_batch(
                  ix2.data(), tw.data(), tx.data(), par.data(), BN, res4.data()),
              "taproot unliftable internal fail-closed");
        check(res4[3] == 0, "taproot unliftable row invalid");
        // Null ptr / count==0 / overflow.
        std::vector<std::uint8_t> res5(BN, 1);
        check(!ufsecp::lbtc::taproot_commitment_verify_batch(
                  nullptr, tw.data(), tx.data(), par.data(), BN, res5.data()),
              "taproot null internal fails closed");
        { long ok = 0; for (auto v : res5) ok += v; check(ok == 0, "taproot null zeroes out"); }
        check(!ufsecp::lbtc::taproot_commitment_verify_batch(
                  ix.data(), tw.data(), tx.data(), par.data(), BN, nullptr),
              "taproot null out fails closed");
        std::vector<std::uint8_t> res6(BN, 0xAB);
        check(ufsecp::lbtc::taproot_commitment_verify_batch(
                  ix.data(), tw.data(), tx.data(), par.data(), 0, res6.data()),
              "taproot count==0 vacuous true");
        check(res6[0] == 0xAB, "taproot count==0 out untouched");
        check(!ufsecp::lbtc::taproot_commitment_verify_batch(
                  ix.data(), tw.data(), tx.data(), par.data(), OVF32, nullptr),
              "taproot huge count overflow rejected");
        // Distinctness: RAW semantics — a mismatched tweak (recomputed, not the
        // supplied raw one) must NOT validate. Rebuild expected from tweak+1.
        std::vector<std::uint8_t> tw3 = tw;
        for (std::size_t i = 0; i < BN; ++i) tw3[i * 32 + 31] ^= 0x01;  // different scalar
        std::vector<std::uint8_t> res7(BN, 0);
        check(!ufsecp::lbtc::taproot_commitment_verify_batch(
                  ix.data(), tw3.data(), tx.data(), par.data(), BN, res7.data()),
              "taproot uses supplied raw tweak (not a recomputed one)");
    }

    // ---- tagged_hash_batch (fixed-length, HASH op: never all-zero) ----
    {
        const auto thd = secp256k1::SHA256::hash("BIP0340/test", 12);
        std::uint8_t th[32]; std::memcpy(th, thd.data(), 32);
        constexpr std::size_t ML = 37;
        std::vector<std::uint8_t> msgs(BN * ML);
        for (auto& b : msgs) b = nb();
        std::vector<std::uint8_t> out(BN * 32, 0);
        check(ufsecp::lbtc::tagged_hash_batch(th, msgs.data(), ML, BN, out.data()),
              "tagged_hash_batch computes true");
        int mism = 0, nonzero_seen = 0;
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t ref[32];
            ref_tagged_hash(th, msgs.data() + i * ML, ML, ref);
            if (std::memcmp(out.data() + i * 32, ref, 32) != 0) ++mism;
            for (int j = 0; j < 32; ++j) if (out[i * 32 + j] != 0) { ++nonzero_seen; break; }
        }
        check(mism == 0, "tagged_hash bit-exact vs serial reference");
        check(nonzero_seen == (int)BN, "tagged_hash rows non-zero (not pre-zeroed)");
        // Tag-string overload equals precomputed path.
        std::vector<std::uint8_t> out2(BN * 32, 0);
        check(ufsecp::lbtc::tagged_hash_batch("BIP0340/test", 12, msgs.data(), ML, BN, out2.data()),
              "tagged_hash tag-string overload true");
        check(std::memcmp(out.data(), out2.data(), BN * 32) == 0,
              "tagged_hash tag-string == precomputed");
        // Null / msg_len==0 must NOT touch out (HASH op).
        std::vector<std::uint8_t> outs(BN * 32, 0xCD);
        check(!ufsecp::lbtc::tagged_hash_batch(nullptr, msgs.data(), ML, BN, outs.data()),
              "tagged_hash null tag fails");
        check(outs[0] == 0xCD, "tagged_hash null tag leaves out untouched");
        check(!ufsecp::lbtc::tagged_hash_batch(th, nullptr, ML, BN, outs.data()),
              "tagged_hash null msgs fails");
        check(outs[0] == 0xCD, "tagged_hash null msgs leaves out untouched");
        check(!ufsecp::lbtc::tagged_hash_batch(th, msgs.data(), 0, BN, outs.data()),
              "tagged_hash msg_len==0 fails");
        check(outs[0] == 0xCD, "tagged_hash msg_len==0 leaves out untouched");
        check(ufsecp::lbtc::tagged_hash_batch(th, msgs.data(), ML, 0, outs.data()),
              "tagged_hash count==0 vacuous true");
        check(outs[0] == 0xCD, "tagged_hash count==0 out untouched");
        // Overflow safe with real buffer (HASH op never writes on bad input).
        check(!ufsecp::lbtc::tagged_hash_batch(th, msgs.data(), 32, OVF32, out.data()),
              "tagged_hash huge count overflow rejected");
    }

    // ---- tagged_hash_var_batch (variable-length; CPU must not cap at 256) ----
    {
        const auto thd = secp256k1::SHA256::hash("BIP0340/test", 12);
        std::uint8_t th[32]; std::memcpy(th, thd.data(), 32);
        const std::size_t STRIDE = 512;  // > 256 to force GPU decline -> CPU covers
        std::vector<std::uint8_t> msgs(BN * STRIDE);
        for (auto& b : msgs) b = nb();
        std::vector<std::uint32_t> lens(BN);
        for (std::size_t i = 0; i < BN; ++i)
            lens[i] = (i % 3 == 0) ? (std::uint32_t)(300 + (i % 50))   // > 256 (no cap!)
                                   : (std::uint32_t)(1 + (i % 40));
        std::vector<std::uint8_t> out(BN * 32, 0);
        check(ufsecp::lbtc::tagged_hash_var_batch(th, msgs.data(), lens.data(), STRIDE, BN, out.data()),
              "tagged_hash_var_batch computes true");
        int mism = 0;
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t ref[32];
            ref_tagged_hash(th, msgs.data() + i * STRIDE, lens[i], ref);
            if (std::memcmp(out.data() + i * 32, ref, 32) != 0) ++mism;
        }
        check(mism == 0, "tagged_hash_var bit-exact vs serial (no 256-byte cap)");
        // stride < msg_lens[i] -> false, out untouched.
        std::vector<std::uint8_t> outs(BN * 32, 0xCD);
        std::vector<std::uint32_t> lens_bad = lens; lens_bad[0] = (std::uint32_t)STRIDE + 1;
        check(!ufsecp::lbtc::tagged_hash_var_batch(th, msgs.data(), lens_bad.data(), STRIDE, BN, outs.data()),
              "tagged_hash_var stride<len fails");
        check(outs[0] == 0xCD, "tagged_hash_var stride<len leaves out untouched");
        // Null cases.
        check(!ufsecp::lbtc::tagged_hash_var_batch(nullptr, msgs.data(), lens.data(), STRIDE, BN, outs.data()),
              "tagged_hash_var null tag fails");
        check(!ufsecp::lbtc::tagged_hash_var_batch(th, nullptr, lens.data(), STRIDE, BN, outs.data()),
              "tagged_hash_var null msgs fails");
        check(!ufsecp::lbtc::tagged_hash_var_batch(th, msgs.data(), nullptr, STRIDE, BN, outs.data()),
              "tagged_hash_var null lens fails");
        check(outs[0] == 0xCD, "tagged_hash_var null leaves out untouched");
        check(ufsecp::lbtc::tagged_hash_var_batch(th, msgs.data(), lens.data(), STRIDE, 0, outs.data()),
              "tagged_hash_var count==0 vacuous true");
        check(!ufsecp::lbtc::tagged_hash_var_batch(th, msgs.data(), lens.data(), STRIDE, OVF32, out.data()),
              "tagged_hash_var huge count overflow rejected");
    }

    // ---- hash256_batch (double-SHA256, HASH op) ----
    {
        constexpr std::size_t IL = 80;
        std::vector<std::uint8_t> in(BN * IL);
        for (auto& b : in) b = nb();
        std::vector<std::uint8_t> out(BN * 32, 0);
        check(ufsecp::lbtc::hash256_batch(in.data(), IL, BN, out.data()),
              "hash256_batch computes true");
        int mism = 0, nonzero_seen = 0;
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t ref[32];
            ref_hash256(in.data() + i * IL, IL, ref);
            if (std::memcmp(out.data() + i * 32, ref, 32) != 0) ++mism;
            for (int j = 0; j < 32; ++j) if (out[i * 32 + j] != 0) { ++nonzero_seen; break; }
        }
        check(mism == 0, "hash256 bit-exact vs serial double-SHA reference");
        check(nonzero_seen == (int)BN, "hash256 rows non-zero (not pre-zeroed)");
        // Null / input_len==0 / count==0 / overflow (HASH op never writes on bad input).
        std::vector<std::uint8_t> outs(BN * 32, 0xCD);
        check(!ufsecp::lbtc::hash256_batch(nullptr, IL, BN, outs.data()),
              "hash256 null in fails");
        check(outs[0] == 0xCD, "hash256 null in leaves out untouched");
        check(!ufsecp::lbtc::hash256_batch(in.data(), IL, BN, nullptr),
              "hash256 null out fails");
        check(!ufsecp::lbtc::hash256_batch(in.data(), 0, BN, outs.data()),
              "hash256 input_len==0 fails");
        check(outs[0] == 0xCD, "hash256 input_len==0 leaves out untouched");
        check(ufsecp::lbtc::hash256_batch(in.data(), IL, 0, outs.data()),
              "hash256 count==0 vacuous true");
        check(!ufsecp::lbtc::hash256_batch(in.data(), IL, OVF32, out.data()),
              "hash256 huge count overflow rejected");
    }

    // ---- hash256_var_batch (variable-length double-SHA256, HASH op) ----
    {
        const std::size_t STRIDE = 200;
        std::vector<std::uint8_t> in(BN * STRIDE);
        for (auto& b : in) b = nb();
        std::vector<std::uint32_t> lens(BN);
        for (std::size_t i = 0; i < BN; ++i) lens[i] = (std::uint32_t)(1 + (i % STRIDE));

        // STARTUP CAPTURE (non-destructive): whatever is installed right now is
        // either nullptr (CPU-only build) or the REAL production hook
        // self-installed at process start by gpu_engine_hook.cpp's
        // EngineLbtcOpsInstaller (direct-GPU profile, SECP256K1_LBTC_GPU_OPS).
        // Swap-and-restore reads it without disturbing whatever is genuinely live.
        const auto prod_hook = ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook(nullptr);
        ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook(prod_hook);
        check(prod_hook != fake_hash256_var_hook,
              "hash256_var production hook (if any) is not this test's fake");

        // Correctness through whatever is actually installed at startup: the
        // production engine hook in a direct-GPU build (whether it computes on
        // a real device or internally declines to the header's CPU loop), or
        // the CPU-only header path otherwise. No test double is installed for
        // this call, so it proves hash256_var_batch reaches a real, non-test
        // hook end-to-end -- a fake sentinel alone cannot prove that.
        std::vector<std::uint8_t> out(BN * 32, 0);
        check(ufsecp::lbtc::hash256_var_batch(in.data(), lens.data(), STRIDE, BN, out.data()),
              "hash256_var_batch computes true");
        int mism = 0, nonzero_seen = 0;
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t ref[32];
            ref_hash256(in.data() + i * STRIDE, lens[i], ref);
            if (std::memcmp(out.data() + i * 32, ref, 32) != 0) ++mism;
            for (int j = 0; j < 32; ++j) if (out[i * 32 + j] != 0) { ++nonzero_seen; break; }
        }
        check(mism == 0, prod_hook != nullptr
              ? "hash256_var bit-exact vs serial double-SHA reference (production hook path)"
              : "hash256_var bit-exact vs serial double-SHA reference (CPU fallback)");
        check(nonzero_seen == (int)BN, "hash256_var rows non-zero (not pre-zeroed)");

        // GPU-hook path: install a fake hook, confirm the wrapper actually uses
        // its output (sentinel 0xEE) instead of silently falling through to CPU.
        // The hook displaced here must be exactly what startup capture read
        // above (nullptr, or the real production hook) -- not an assumption
        // that no hook can ever already be installed.
        check(ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook(fake_hash256_var_hook) == prod_hook,
              "hash256_var pre-existing hook (production or none) matches startup capture");
        std::vector<std::uint8_t> out_hook(BN * 32, 0);
        check(ufsecp::lbtc::hash256_var_batch(in.data(), lens.data(), STRIDE, BN, out_hook.data()),
              "hash256_var_batch with hook computes true");
        bool all_sentinel = true;
        for (auto b : out_hook) if (b != 0xEE) { all_sentinel = false; break; }
        check(all_sentinel, "hash256_var_batch used the installed hook's output");
        check(ufsecp::lbtc::gpu_hook::install_lbtc_hash256_var_hook(prod_hook) == fake_hash256_var_hook,
              "hash256_var fake hook uninstalled, previous fn returned; production hook restored");

        // Hook restored to whatever was genuinely live at startup: recomputes
        // and must match the earlier reference-checked output again
        // (deterministic, whether served by the production hook or CPU).
        std::vector<std::uint8_t> out2(BN * 32, 0);
        check(ufsecp::lbtc::hash256_var_batch(in.data(), lens.data(), STRIDE, BN, out2.data()),
              "hash256_var_batch after hook removal computes true");
        check(std::memcmp(out.data(), out2.data(), BN * 32) == 0,
              "hash256_var output stable after hook removal");

        // Null / stride==0 / count==0 / overflow / hostile length (HASH op never writes on bad input).
        std::vector<std::uint8_t> outs(BN * 32, 0xCD);
        check(!ufsecp::lbtc::hash256_var_batch(nullptr, lens.data(), STRIDE, BN, outs.data()),
              "hash256_var null inputs fails");
        check(outs[0] == 0xCD, "hash256_var null inputs leaves out untouched");
        check(!ufsecp::lbtc::hash256_var_batch(in.data(), nullptr, STRIDE, BN, outs.data()),
              "hash256_var null lens fails");
        check(outs[0] == 0xCD, "hash256_var null lens leaves out untouched");
        check(!ufsecp::lbtc::hash256_var_batch(in.data(), lens.data(), STRIDE, BN, nullptr),
              "hash256_var null out fails");
        check(!ufsecp::lbtc::hash256_var_batch(in.data(), lens.data(), 0, BN, outs.data()),
              "hash256_var stride==0 fails");
        check(outs[0] == 0xCD, "hash256_var stride==0 leaves out untouched");
        check(ufsecp::lbtc::hash256_var_batch(in.data(), lens.data(), STRIDE, 0, outs.data()),
              "hash256_var count==0 vacuous true");
        check(outs[0] == 0xCD, "hash256_var count==0 out untouched");
        check(!ufsecp::lbtc::hash256_var_batch(in.data(), lens.data(), STRIDE, OVF32, out.data()),
              "hash256_var huge count overflow rejected");
        // Hostile: one row's length exceeds stride -> reject, out untouched.
        std::vector<std::uint32_t> lens_bad = lens; lens_bad[3] = (std::uint32_t)STRIDE + 1;
        check(!ufsecp::lbtc::hash256_var_batch(in.data(), lens_bad.data(), STRIDE, BN, outs.data()),
              "hash256_var length>stride fails closed");
        check(outs[0] == 0xCD, "hash256_var length>stride leaves out untouched");
        // Hostile: a zero-length row is also rejected (every row must hash >=1 byte).
        std::vector<std::uint32_t> lens_zero = lens; lens_zero[4] = 0;
        check(!ufsecp::lbtc::hash256_var_batch(in.data(), lens_zero.data(), STRIDE, BN, outs.data()),
              "hash256_var zero-length row fails closed");
        check(outs[0] == 0xCD, "hash256_var zero-length row leaves out untouched");
    }

    // ---- txid_hash_batch / wtxid_hash_batch (semantic aliases over hash256_var_batch) ----
    {
        const std::size_t STRIDE = 250;
        std::vector<std::uint8_t> txs(BN * STRIDE);
        for (auto& b : txs) b = nb();
        std::vector<std::uint32_t> lens(BN);
        for (std::size_t i = 0; i < BN; ++i) lens[i] = (std::uint32_t)(10 + (i % (STRIDE - 10)));

        std::vector<std::uint8_t> out_txid(BN * 32, 0);
        check(ufsecp::lbtc::txid_hash_batch(txs.data(), lens.data(), STRIDE, BN, out_txid.data()),
              "txid_hash_batch computes true");
        std::vector<std::uint8_t> out_wtxid(BN * 32, 0);
        check(ufsecp::lbtc::wtxid_hash_batch(txs.data(), lens.data(), STRIDE, BN, out_wtxid.data()),
              "wtxid_hash_batch computes true");
        // Both are thin aliases of hash256_var_batch: byte-identical output
        // for the same input, computed through the exact same call path
        // (production hook or CPU fallback, whichever is currently live).
        std::vector<std::uint8_t> out_ref(BN * 32, 0);
        check(ufsecp::lbtc::hash256_var_batch(txs.data(), lens.data(), STRIDE, BN, out_ref.data()),
              "hash256_var_batch reference computes true");
        check(std::memcmp(out_txid.data(), out_ref.data(), BN * 32) == 0,
              "txid_hash_batch byte-identical to hash256_var_batch");
        check(std::memcmp(out_wtxid.data(), out_ref.data(), BN * 32) == 0,
              "wtxid_hash_batch byte-identical to hash256_var_batch");
        // Independent double-SHA256 (HASH256) oracle cross-check.
        int mism_txid = 0, mism_wtxid = 0;
        for (std::size_t i = 0; i < BN; ++i) {
            std::uint8_t ref[32];
            ref_hash256(txs.data() + i * STRIDE, lens[i], ref);
            if (std::memcmp(out_txid.data() + i * 32, ref, 32) != 0) ++mism_txid;
            if (std::memcmp(out_wtxid.data() + i * 32, ref, 32) != 0) ++mism_wtxid;
        }
        check(mism_txid == 0, "txid_hash_batch bit-exact vs independent double-SHA256 oracle");
        check(mism_wtxid == 0, "wtxid_hash_batch bit-exact vs independent double-SHA256 oracle");
    }

    // ---- merkle_pair_hash_batch (SoA double-SHA256 over left32||right32 pairs) ----
    {
        // (a) count==0 -> true, out32 completely untouched.
        {
            std::uint8_t left32[32], right32[32], out32[32];
            std::memset(left32, 0x11, 32);
            std::memset(right32, 0x22, 32);
            std::memset(out32, 0xAB, 32);
            check(ufsecp::lbtc::merkle_pair_hash_batch(left32, right32, 0, out32),
                  "merkle_pair count==0 vacuous true");
            bool untouched = true;
            for (int i = 0; i < 32; ++i) if (out32[i] != 0xAB) { untouched = false; break; }
            check(untouched, "merkle_pair count==0 leaves out untouched");
        }

        std::vector<std::uint8_t> left(BN * 32), right(BN * 32);
        for (auto& b : left) b = nb();
        for (auto& b : right) b = nb();

        // (b) null left32/right32/out32, each individually -> false, out untouched.
        {
            std::vector<std::uint8_t> out(BN * 32, 0xCD);
            check(!ufsecp::lbtc::merkle_pair_hash_batch(nullptr, right.data(), BN, out.data()),
                  "merkle_pair null left fails closed");
            check(out[0] == 0xCD, "merkle_pair null left leaves out untouched");
            check(!ufsecp::lbtc::merkle_pair_hash_batch(left.data(), nullptr, BN, out.data()),
                  "merkle_pair null right fails closed");
            check(out[0] == 0xCD, "merkle_pair null right leaves out untouched");
            check(!ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), BN, nullptr),
                  "merkle_pair null out fails closed");
        }

        // (c) overflow rejection: count*32 overflows size_t (mirrors the
        // hash256/hash256_var OVF32 fixture and column_layout_overflows(count,32)).
        {
            std::vector<std::uint8_t> out(BN * 32, 0xCD);
            check(!ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), OVF32, out.data()),
                  "merkle_pair huge count overflow rejected");
            check(out[0] == 0xCD, "merkle_pair overflow leaves out untouched");
        }

        // Startup capture (non-destructive), mirroring the hash256_var_batch
        // pattern above: whatever is installed right now is either nullptr
        // (CPU-only build) or the real production hook self-installed at
        // process start (direct-GPU profile). Swap-and-restore reads it
        // without disturbing whatever is genuinely live.
        const auto prod_merkle_hook = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(nullptr);
        ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prod_merkle_hook);
        check(prod_merkle_hook != fake_merkle_pair_hook,
              "merkle_pair production hook (if any) is not this test's fake");

        // (d) CPU fallback KAT: force decline (-1) so this exercises the CPU
        // loop regardless of what a real production hook would otherwise do,
        // then check against an independently computed
        // double-SHA256(left||right) oracle, including an all-zeros pair and
        // a couple of arbitrary non-trivial byte patterns.
        {
            const auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(decline_merkle_pair_hook);

            constexpr std::size_t KATN = 3;
            std::uint8_t kl[KATN * 32], kr[KATN * 32], kout[KATN * 32];
            std::memset(kl + 0 * 32, 0x00, 32);                                          // all-zeros pair
            std::memset(kr + 0 * 32, 0x00, 32);
            for (int i = 0; i < 32; ++i) kl[1 * 32 + i] = static_cast<std::uint8_t>(i);          // 0x00..0x1F
            for (int i = 0; i < 32; ++i) kr[1 * 32 + i] = static_cast<std::uint8_t>(0xFF - i);    // 0xFF..0xE0
            for (int i = 0; i < 32; ++i) kl[2 * 32 + i] = nb();                                  // arbitrary
            for (int i = 0; i < 32; ++i) kr[2 * 32 + i] = nb();                                  // arbitrary

            check(ufsecp::lbtc::merkle_pair_hash_batch(kl, kr, KATN, kout),
                  "merkle_pair CPU-fallback KAT computes true (hook forced decline)");
            int mism = 0;
            for (std::size_t i = 0; i < KATN; ++i) {
                std::uint8_t combined[64], ref[32];
                std::memcpy(combined, kl + i * 32, 32);
                std::memcpy(combined + 32, kr + i * 32, 32);
                ref_hash256(combined, 64, ref);
                if (std::memcmp(kout + i * 32, ref, 32) != 0) ++mism;
            }
            check(mism == 0, "merkle_pair CPU fallback bit-exact vs independent double-SHA256 oracle");

            ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prev);
        }

        // (e) hook-decline fallback over the larger BN batch: install a hook
        // that always returns -1; verify the CPU path still produces the
        // correct real hash (not just "doesn't crash").
        {
            const auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(decline_merkle_pair_hook);
            std::vector<std::uint8_t> out(BN * 32, 0);
            check(ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), BN, out.data()),
                  "merkle_pair hook-decline (-1) still computes true");
            int mism = 0;
            for (std::size_t i = 0; i < BN; ++i) {
                std::uint8_t combined[64], ref[32];
                std::memcpy(combined, left.data() + i * 32, 32);
                std::memcpy(combined + 32, right.data() + i * 32, 32);
                ref_hash256(combined, 64, ref);
                if (std::memcmp(out.data() + i * 32, ref, 32) != 0) ++mism;
            }
            check(mism == 0, "merkle_pair hook-decline path bit-exact vs independent oracle");
            ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prev);
        }

        // (f) hook-success path: fake hook writes a recognizable sentinel and
        // returns 0 (handled). Verify out32 == sentinel exactly, proving the
        // hook path (not a silent CPU override) is what ran.
        {
            check(ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(fake_merkle_pair_hook) == prod_merkle_hook,
                  "merkle_pair pre-existing hook (production or none) matches startup capture");
            std::vector<std::uint8_t> out(BN * 32, 0);
            check(ufsecp::lbtc::merkle_pair_hash_batch(left.data(), right.data(), BN, out.data()),
                  "merkle_pair with fake hook computes true");
            bool all_sentinel = true;
            for (auto b : out) if (b != 0xDD) { all_sentinel = false; break; }
            check(all_sentinel, "merkle_pair_hash_batch used the installed hook's output");
            check(ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prod_merkle_hook) == fake_merkle_pair_hook,
                  "merkle_pair fake hook uninstalled, previous fn returned; production hook restored");
        }

        // (g) left/right SoA byte order: for a pair where left != right,
        // verify swap(left,right) changes the output (not commutative), and
        // that the un-swapped call equals doubleSHA256(left||right)
        // specifically -- not right||left.
        {
            const auto prev = ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(decline_merkle_pair_hook);
            std::uint8_t l[32], r[32], out_lr[32], out_rl[32];
            for (int i = 0; i < 32; ++i) { l[i] = static_cast<std::uint8_t>(i); r[i] = static_cast<std::uint8_t>(200 + i); }
            check(std::memcmp(l, r, 32) != 0, "merkle_pair byte-order fixture left != right");
            check(ufsecp::lbtc::merkle_pair_hash_batch(l, r, 1, out_lr),
                  "merkle_pair byte-order left||right computes true");
            check(ufsecp::lbtc::merkle_pair_hash_batch(r, l, 1, out_rl),
                  "merkle_pair byte-order right||left (swapped) computes true");
            check(std::memcmp(out_lr, out_rl, 32) != 0,
                  "merkle_pair swap(left,right) changes output (not commutative)");
            std::uint8_t combined[64], ref_lr[32];
            std::memcpy(combined, l, 32);
            std::memcpy(combined + 32, r, 32);
            ref_hash256(combined, 64, ref_lr);
            check(std::memcmp(out_lr, ref_lr, 32) == 0,
                  "merkle_pair un-swapped output == doubleSHA256(left||right), not right||left");
            ufsecp::lbtc::gpu_hook::install_lbtc_merkle_pair_hook(prev);
        }
    }


    // ════════════════════════════════════════════════════════════════════════

    // ════════════════════════════════════════════════════════════════════════
    // Direct operations coverage (moved from test_direct_operations.cpp per
    // Codex review — all in one allowed file).
    // ════════════════════════════════════════════════════════════════════════

    // ─── pubkey_create_uncompressed (CT) ──────────────────────────────
    {
        std::uint8_t sk[32], pub65[65];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create_uncompressed(sk, pub65), "pubkey_create_uncompressed valid");
        check(pub65[0] == 0x04, "pubkey_create_uncompressed header 0x04");
        // Verify against compressed version: compress(uncompressed) == pubkey_create(compressed)
        std::uint8_t pub33[33], pub33_from65[33];
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "pubkey_create for cross-check");
        check(ufsecp::lbtc::pubkey_compress(pub65, pub33_from65), "pubkey_compress cross-check");
        check(std::memcmp(pub33, pub33_from65, 33) == 0, "uncompressed->compressed matches direct compressed");
        // Invalid key must fail
        std::uint8_t zero_sk[32] = {0};
        check(!ufsecp::lbtc::pubkey_create_uncompressed(zero_sk, pub65), "pubkey_create_uncompressed zero sk fails");
        // Verify output zeroed
        for (int i = 0; i < 65; ++i) check(pub65[i] == 0, "pubkey_create_uncompressed zero sk zeroes output");
    }

    // ─── pubkey_parse_uncompressed ───────────────────────────────────
    {
        std::uint8_t sk[32], pub65[65];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create_uncompressed(sk, pub65), "pubkey_parse_uncompressed create");
        check(ufsecp::lbtc::pubkey_parse_uncompressed(pub65), "pubkey_parse_uncompressed valid");
        // Invalid header
        std::uint8_t bad65[65];
        std::memset(bad65, 0xFF, 65);
        bad65[0] = 0x05;
        check(!ufsecp::lbtc::pubkey_parse_uncompressed(bad65), "pubkey_parse_uncompressed invalid header");
        // Off-curve point (x = all-0x03, which is off-curve)
        std::uint8_t off65[65];
        off65[0] = 0x04;
        std::memset(off65 + 1, 0x03, 32);  // x = 0x0303...
        std::memset(off65 + 33, 0x00, 32); // y = 0 (not on curve)
        check(!ufsecp::lbtc::pubkey_parse_uncompressed(off65), "pubkey_parse_uncompressed off-curve rejected");
    }

    // ─── pubkey_compress / pubkey_decompress roundtrip ───────────────
    {
        std::uint8_t sk[32], pub33[33], pub65[65], pub33_rt[33];
        rand_sk(sk);
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "compress-rt create compressed");
        check(ufsecp::lbtc::pubkey_decompress(pub33, pub65), "pubkey_decompress");
        check(pub65[0] == 0x04, "pubkey_decompress header check");
        check(ufsecp::lbtc::pubkey_compress(pub65, pub33_rt), "pubkey_compress back");
        check(std::memcmp(pub33, pub33_rt, 33) == 0, "compress(decompress(pub33)) == pub33 roundtrip");
        // Invalid inputs
        std::uint8_t bad33[33] = {0xFF};
        check(!ufsecp::lbtc::pubkey_decompress(bad33, pub65), "pubkey_decompress invalid fails");
        std::uint8_t bad65[65] = {0xFF};
        check(!ufsecp::lbtc::pubkey_compress(bad65, pub33_rt), "pubkey_compress invalid fails");
    }

    // ─── ECDSA sign recoverable hedged + recover (CT) ────────────────
    {
        std::uint8_t sk[32], hash[32], aux32[32], sig65[65], pub33[33], recovered[33];
        rand_sk(sk);
        rand_hash(hash);
        for (int i = 0; i < 32; ++i) aux32[i] = nb();
        check(ufsecp::lbtc::ecdsa_sign_hedged_recoverable(hash, sk, aux32, sig65),
              "ecdsa_sign_hedged_recoverable");
        // Parse back
        std::uint8_t sig64[64];
        int recid = 0;
        check(ufsecp::lbtc::recoverable_from_compact(sig65, sig64, recid),
              "hedged recoverable_from_compact");
        check(recid >= 0 && recid <= 3, "hedged recid in range");
        check(ufsecp::lbtc::ecdsa_recover(hash, sig64, recid, recovered),
              "hedged ecdsa_recover");
        check(ufsecp::lbtc::pubkey_create(sk, pub33), "hedged pubkey_create");
        check(std::memcmp(pub33, recovered, 33) == 0, "hedged recovered pubkey matches created");
        // Verify the signature
        check(ufsecp::lbtc::ecdsa_verify(pub33, hash, sig64), "hedged ecdsa_verify after recover");
        // Invalid sk must fail
        std::uint8_t zero_sk[32] = {0};
        check(!ufsecp::lbtc::ecdsa_sign_hedged_recoverable(hash, zero_sk, aux32, sig65),
              "ecdsa_sign_hedged_recoverable zero sk fails");
        for (int i = 0; i < 65; ++i) check(sig65[i] == 0, "hedged recoverable zero sk zeroes output");
        // Hedged signing uses aux randomness for defense-in-depth;
        // signatures may differ across calls even with identical inputs
        // (the aux seeds internal HMAC-DRBG state). Verify each sig independently.
        std::uint8_t sig65_b[65];
        check(ufsecp::lbtc::ecdsa_sign_hedged_recoverable(hash, sk, aux32, sig65_b),
              "hedged recoverable second call");
        // Recover and verify the second signature independently
        std::uint8_t sig64_b[64], recovered_b[33];
        int recid_b = 0;
        check(ufsecp::lbtc::recoverable_from_compact(sig65_b, sig64_b, recid_b),
              "hedged second recoverable_from_compact");
        check(ufsecp::lbtc::ecdsa_recover(hash, sig64_b, recid_b, recovered_b),
              "hedged second ecdsa_recover");
        check(std::memcmp(pub33, recovered_b, 33) == 0,
              "hedged second recovered pubkey matches created");
    }

    // ─── pubkey_combine negative test ────────────────────────────────
    {
        std::uint8_t out[33];
        check(!ufsecp::lbtc::pubkey_combine((const std::uint8_t* const*)nullptr, 0, out),
              "pubkey_combine zero count fails");
        // Single invalid key
        std::uint8_t bad33[33] = {0xFF};
        const std::uint8_t* ptrs[1] = {bad33};
        check(!ufsecp::lbtc::pubkey_combine(ptrs, 1, out), "pubkey_combine invalid key fails");
    }

    // ─── taproot_tweak_add_check with merkle root ───────────────────
    {
        std::uint8_t sk[32], internal_xonly[32];
        rand_sk(sk);
        check(ufsecp::lbtc::schnorr_keypair_create(sk, internal_xonly), "taproot mr internal key");
        // Build a fake merkle root (32 bytes)
        std::uint8_t merkle_root[32];
        rand_hash(merkle_root);
        // Compute expected output via tagged hash
        // Concatenate internal_xonly || merkle_root for BIP-341 tagged hash
        std::uint8_t tap_tweak_msg[64];
        std::memcpy(tap_tweak_msg, internal_xonly, 32);
        std::memcpy(tap_tweak_msg + 32, merkle_root, 32);
        auto tweak_hash = secp256k1::tagged_hash("TapTweak", tap_tweak_msg, 64);
        secp256k1::fast::Scalar tweak;
        check(secp256k1::fast::Scalar::parse_bytes_strict(tweak_hash.data(), tweak), "taproot mr tweak parse");
        std::uint8_t internal_compressed[33] = {0x02};
        std::memcpy(internal_compressed + 1, internal_xonly, 32);
        std::uint8_t tweak32[32];
        std::memcpy(tweak32, tweak_hash.data(), 32);
        check(ufsecp::lbtc::pubkey_tweak_add(internal_compressed, tweak32),
              "taproot mr output key");
        std::uint8_t output_xonly[32];
        std::memcpy(output_xonly, internal_compressed + 1, 32);
        int parity = (internal_compressed[0] == 0x03) ? 1 : 0;
        check(ufsecp::lbtc::taproot_tweak_add_check(output_xonly, parity, internal_xonly,
              merkle_root, 32), "taproot_tweak_add_check with merkle root");
    }

    // ─── Fail-closed: zero pubkey to verify ─────────────────────────
    {
        std::uint8_t sk[32], hash[32], sig64[64];
        rand_sk(sk);
        rand_hash(hash);
        check(ufsecp::lbtc::ecdsa_sign(hash, sk, sig64), "fail-closed sign");
        std::uint8_t zero_pub[33] = {0};
        check(!ufsecp::lbtc::ecdsa_verify(zero_pub, hash, sig64), "ecdsa_verify zero pubkey fails");
    }


    if (fails == 0) std::printf("test_direct_verify: ALL PASS (ecdsa+schnorr single+batch+fail-closed+lbtc batch ops)\n");
    return fails == 0 ? 0 : 1;
}
