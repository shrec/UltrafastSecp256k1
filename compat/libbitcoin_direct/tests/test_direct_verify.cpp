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

    if (fails == 0) std::printf("test_direct_verify: ALL PASS (ecdsa+schnorr single+batch+fail-closed+lbtc batch ops)\n");
    return fails == 0 ? 0 : 1;
}
