// ============================================================================
// test_bch_schnorr_differential.cpp
// ============================================================================
// Differential test for the Bitcoin Cash 2019 Schnorr verifier
// (src/cpu/src/bch_schnorr.cpp). NOT BIP-340; spec:
// https://upgradespecs.bitcoincashnode.org/2019-05-15-schnorr/
//
// The optimised verifier replaced the body of the BCHN shim's
// secp256k1_schnorr_verify. That body is the reference: old_shim_verify()
// below is a verbatim copy of it as it stood before the delegation (one dual
// multiplication, an affine conversion, a sqrt-based Jacobi test, a byte
// compare of x). Every verifier surface must return the old verdict on every
// input, valid or not:
//
//   bch_schnorr_verify(sig, msg, pubkey33)         engine, wire key
//   bch_schnorr_verify(sig, msg, P, pubkey33)      engine, parsed key
//   secp256k1_schnorr_verify                       BCHN shim (now delegating)
//   ufsecp_bch_schnorr_verify                      C ABI
//
// plus the eight regression_bch_schnorr_spec vectors (independent oracle) and
// their -R twins, and the batch surfaces against the AND of their singles:
// secp256k1_schnorr_verify_batch (Knuth PR #826 contract),
// ufsecp_bch_schnorr_batch_verify and ufsecp_bch_schnorr_batch_identify.
// ============================================================================

#include "secp256k1.h"
#include "secp256k1_schnorr.h"
#include "ufsecp.h"

#include "secp256k1/bch_schnorr.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/sha256.hpp"
#include "secp256k1/ct/point.hpp"

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using secp256k1::fast::FieldElement;
using secp256k1::fast::Point;
using secp256k1::fast::Scalar;

int g_pass = 0;
int g_fail = 0;

void check(bool cond, const char* msg) {
    if (cond) {
        ++g_pass;
    } else {
        ++g_fail;
        std::printf("  [FAIL] %s\n", msg);
    }
}

// The BCH shim only null-checks its context argument (see
// test_regression_bch_schnorr_spec.cpp).
const char kCtxStorage = 0;
const secp256k1_context* bch_ctx() {
    return reinterpret_cast<const secp256k1_context*>(&kCtxStorage);
}

// ---- verbatim copy of the pre-delegation shim verifier ---------------------
Point old_pubkey_data_to_point(const unsigned char data[64]) noexcept {
    std::array<std::uint8_t, 32> xb{}, yb{};
    std::memcpy(xb.data(), data, 32);
    std::memcpy(yb.data(), data + 32, 32);
    FieldElement x = FieldElement::from_bytes(xb);
    FieldElement y = FieldElement::from_bytes(yb);
    return Point::from_affine(x, y);
}

Scalar old_bch_schnorr_e(const std::array<std::uint8_t, 32>& rx, const std::uint8_t* px32,
                         bool y_is_odd, const std::array<std::uint8_t, 32>& msg) {
    std::uint8_t pcomp[33];
    pcomp[0] = y_is_odd ? 0x03 : 0x02;
    std::memcpy(pcomp + 1, px32, 32);
    secp256k1::SHA256 h;
    h.update(rx.data(), 32);
    h.update(pcomp, 33);
    h.update(msg.data(), 32);
    auto digest = h.finalize();
    return Scalar::from_bytes(digest);
}

int old_shim_verify(const unsigned char* sig64, const unsigned char* msg32,
                    const unsigned char pubkey_data[64]) {
    std::array<std::uint8_t, 32> rx{}, sb{}, msg{};
    std::memcpy(rx.data(), sig64, 32);
    std::memcpy(sb.data(), sig64 + 32, 32);
    std::memcpy(msg.data(), msg32, 32);

    Scalar s;
    if (!Scalar::parse_bytes_strict(sb.data(), s) || s.is_zero()) return 0;

    auto P = old_pubkey_data_to_point(pubkey_data);
    if (P.is_infinity()) return 0;

    bool const p_y_odd = (pubkey_data[63] & 1u) != 0;
    auto e = old_bch_schnorr_e(rx, pubkey_data, p_y_odd, msg);

    auto neg_e = e.negate();
    auto R_check = Point::dual_scalar_mul_gen_point(s, neg_e, P);
    if (R_check.is_infinity()) return 0;
    {
        FieldElement const ry = R_check.y();
        if (ry == FieldElement::zero()) return 0;
        FieldElement const rt = ry.sqrt();
        if (!(rt.square() == ry)) return 0;
    }
    auto rx_check = R_check.x().to_bytes();
    return (std::memcmp(rx_check.data(), rx.data(), 32) == 0) ? 1 : 0;
}
// ---- end of verbatim copy ---------------------------------------------------

std::array<std::uint8_t, 32> stream32(const char* label, std::uint32_t ctr) {
    secp256k1::SHA256 h;
    h.update(label, std::strlen(label));
    std::uint8_t c[4] = {static_cast<std::uint8_t>(ctr), static_cast<std::uint8_t>(ctr >> 8),
                         static_cast<std::uint8_t>(ctr >> 16), static_cast<std::uint8_t>(ctr >> 24)};
    h.update(c, 4);
    return h.finalize();
}

bool from_hex(const char* hex, std::uint8_t* out, std::size_t n) {
    if (std::strlen(hex) != n * 2) return false;
    for (std::size_t i = 0; i < n; ++i) {
        unsigned v = 0;
        if (std::sscanf(hex + 2 * i, "%2x", &v) != 1) return false;
        out[i] = static_cast<std::uint8_t>(v);
    }
    return true;
}

struct Key {
    std::array<std::uint8_t, 32> sk{};
    Scalar d;
    Point P;
    secp256k1_pubkey pk{};                 // [x:32][y:32] big-endian
    std::array<std::uint8_t, 33> pub33{};
};

Key key_from_scalar(const Scalar& d) {
    Key k;
    k.d = d;
    k.sk = d.to_bytes();
    k.P = secp256k1::ct::generator_mul(d);
    auto const x = k.P.x().to_bytes();
    auto const y = k.P.y().to_bytes();
    std::memcpy(k.pk.data, x.data(), 32);
    std::memcpy(k.pk.data + 32, y.data(), 32);
    k.pub33 = k.P.to_compressed();
    return k;
}

Key make_key(std::uint32_t i) {
    Scalar d;
    std::uint32_t ctr = i * 1000u;
    do {
        d = Scalar::from_bytes(stream32("bch-diff-key", ctr++));
    } while (d.is_zero());
    return key_from_scalar(d);
}

Scalar challenge(const std::uint8_t* r32, const std::uint8_t* pub33, const std::uint8_t* msg32) {
    secp256k1::SHA256 h;
    h.update(r32, 32);
    h.update(pub33, 33);
    h.update(msg32, 32);
    return Scalar::from_bytes(h.finalize());
}

void set_s(std::uint8_t* sig64, const Scalar& s) {
    auto const b = s.to_bytes();
    std::memcpy(sig64 + 32, b.data(), 32);
}

struct Case {
    std::array<std::uint8_t, 64> sig{};
    std::array<std::uint8_t, 32> msg{};
    std::size_t key = 0;
};

struct Surfaces {
    ufsecp_ctx* uctx = nullptr;
    int disagreements = 0;
    int valid = 0;
    int invalid = 0;
};

// Compare every single-verifier surface against the old shim verdict.
void compare_one(Surfaces& sf, const Case& c, const Key& k) {
    int const want = old_shim_verify(c.sig.data(), c.msg.data(), k.pk.data);
    int const core = secp256k1::bch_schnorr_verify(c.sig.data(), c.msg.data(), k.pub33.data()) ? 1 : 0;
    int const core_pt =
        secp256k1::bch_schnorr_verify(c.sig.data(), c.msg.data(), k.P, k.pub33.data()) ? 1 : 0;
    int const shim = secp256k1_schnorr_verify(bch_ctx(), c.sig.data(), c.msg.data(), &k.pk);
    int const abi = (ufsecp_bch_schnorr_verify(sf.uctx, c.msg.data(), c.sig.data(), k.pub33.data())
                     == UFSECP_OK) ? 1 : 0;
    if (core != want || core_pt != want || shim != want || abi != want) {
        ++sf.disagreements;
        std::printf("  [DIFF] key %zu: old=%d core=%d core_pt=%d shim=%d abi=%d\n",
                    c.key, want, core, core_pt, shim, abi);
    }
    if (want) ++sf.valid; else ++sf.invalid;
}

// The eight regression_bch_schnorr_spec vectors (independent pure-Python oracle).
struct BchVector { const char *seckey, *msg, *pub_x, *pub_y, *sig; };
const BchVector kVectors[] = {
    {"8174099687a26621f4e2cdd7cc03b3dacedb3fb962255b1aafd033cabe831530",
     "289e5175e02c788c2d442cfe81d6be0533d8c13e253ef763fda45d37accfe4d4",
     "f771877964fa2ce401d87bc2558a0df1e6921acef99389f059712b32cfda35fd",
     "90294bf51aedc048671c77d21792f2c7320e02234afb8ee77e1fce48a9390947",
     "7e32606ecf3439a34cf2b2b0b264c8bfbe78854b8c5a5a8f75669b091712cfe5d26de69291c11509e006a60d05e033d427636bd0adfde51b36eaa0fa69e06840"},
    {"b10253764c8b233fb37542e23401c7b450e5a6f9751f3b5a014f6f67e8bc999d",
     "a78521e49048b6e0d368d3fba417fc20c7546272dafa78a8a173fcca6c81233b",
     "f039fdcdb728efbbddf4ee452419a988497debb7bd1b42644c5fa66e9af8c8b6",
     "08f516435d1217f63e786a5a66c2122c343d07c583239329cc7c2256d71d1d27",
     "2e38ea24145c2e139c84a3dd7cbe1d02c7a4a6573262317bc585c397ed4844abe57420edac4e1b2230087ebd2b50f1e38a7bcc235331349bf57f2c22186d080d"},
    {"f576104eebeab09651d83acffc77c8b8c6eaa4b767aeab24d7da80f83f51d865",
     "89da2bd31a5d008c84323c9693f12f09e62a75a688a55f2a6fd24660afba5660",
     "6da0e4d65a933e828c9de388005281dfa7e4948895d10c7c3ef617b5e40d97fd",
     "f424db33400fd7fab1d6e84f4551babd04662a8b1699fbcbf96aba3a3eaedd86",
     "8d66933e3593f104f72e27815bfcaae2c702b7f9ca6ebb3ba99a8d38b6aec1df5e3cbfcf4178b4fe144ebc541623b5cf293cd8a5cf56d09e883ef0c7b2eda6b7"},
    {"a4b3504c2769fce9547f6dda310dd8b094d630a044d65f5324d4b37310aab714",
     "51b5df22eaeaf7a6101b57cfb45084cb98864b1502c6ed1a692da604366a13a4",
     "b15867cb022210f38bf0ae40f4deadc680779338d2bf2569bb99507f95280a2e",
     "68af1237eb189b5f5e0d948135a1fc274cace4171fe65b0d5220a09528717f91",
     "67451e09febdd0a1ece301dd22f0b071e298c1815a3588308b184c330201e820024cb8a6f717ac7d40223740e9160963ce82f954959c3be78979bbaf3468d4ac"},
    {"07e7394e0702340d9fd1d777fbbad2804a5188d1dd07ff580f473bd7645ff205",
     "92253243f3471651d425293dfe382cb9017fe15fc46b1deb79e561f5a38f7242",
     "70e3043d088ec7ebe970529cd9ae6f90c9e04ce4c71f883b1f73b3e6cb2202b3",
     "4c7c4f90f6afef5fedce6ee64bc99e11485d49548fcf0b5fbb37b4d0611522f9",
     "51734be9d97e86521cb3f9d9d57edc01aa06c28e6359b6f917e40776445979e736e1efe840b5dbc3deed3afdd93cfd43b3debdfe3c0efc256dc3a22154bd032d"},
    {"8dfad052fee5c62957d3ebe1752219a02f45634b2c32a6ac408b26ffcedfb7da",
     "cd682266d91f196c09590f35c003fa8aaf01321dcb8cb675c6bc1d70c5b6822b",
     "4141a1ef664b76572adc99eb6a57c927aef4b84e37441338b0eb0f698aa74242",
     "f3a0b00ca1df74816c6252d6b54cbf4524d9063268431881e10e80d06026e423",
     "0c1b442a8bca3ce72affffea110e62bafdf2aabacdb1bf36a106de5a175b4c563d1cb666e93aa9e65ae26a760a5d3c569df8f745bd6c7f8566c25df3024a0f75"},
    {"1e3f92d0f678eb83b0bf93855d90699d8ae5dfb4ae023bdf352bd8d93f2060b1",
     "f6bca2ea669a35112ab2a8292f5a88c675cc265e0b7af2e0ae1b9fe080be108f",
     "732beac3f258cf9d0f3a372e1dd7fe507707b137600953cfd53211b69a08f969",
     "c4da0b301d1baf48f537b75ddac333079944dbaef63fee21d4ff83154dad4efe",
     "f655371c9fb5d011fc5404a76b283415cc1938c23319f47cb90cb9b584b898cc766e2f165f327278d91bd77f9583b24e599b355617ba7600ebda14b9da28c6db"},
    {"5bda7a9c0fb9e5102411342f9bedf63f37d8835ee6e903e95dd3050fd15af81c",
     "231cf49d59380f36c61d1a7b53e95b2cc337bb1c13a935a1e81951e22d248c43",
     "c8a20fe0d0ffa89561717ba19863df4a6cd37ab7f284fac30b4d5677becb5927",
     "e58dff892acab580bc92cda66b44345a9fc45aae370d15eb684dc0c8df6bb79d",
     "7f9e5751fece09c949462a80951a76deb181556815153c657cbccdd5af2015a23825c46eff40c838e05a63662976ce1d173714edc12d105d3d0c2685f9caf3f1"},
};

const std::uint8_t kP[32] = {
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe,0xff,0xff,0xfc,0x2f};
const std::uint8_t kN[32] = {
    0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xfe,
    0xba,0xae,0xdc,0xe6,0xaf,0x48,0xa0,0x3b,0xbf,0xd2,0x5e,0x8c,0xd0,0x36,0x41,0x41};

void test_spec_vectors(Surfaces& sf) {
    std::printf("\n--- regression_bch_schnorr_spec vectors and their -R twins ---\n");
    int valid_all = 0, twin_rejected = 0, loaded = 0;
    for (const BchVector& v : kVectors) {
        std::uint8_t sk[32], pxb[32], pyb[32];
        Case c;
        if (!from_hex(v.seckey, sk, 32) || !from_hex(v.msg, c.msg.data(), 32) ||
            !from_hex(v.pub_x, pxb, 32) || !from_hex(v.pub_y, pyb, 32) ||
            !from_hex(v.sig, c.sig.data(), 64)) {
            check(false, "spec vector hex decodes");
            continue;
        }
        Scalar d;
        if (!Scalar::parse_bytes_strict_nonzero(sk, d)) { check(false, "spec seckey parses"); continue; }
        Key const k = key_from_scalar(d);
        check(std::memcmp(k.pk.data, pxb, 32) == 0 && std::memcmp(k.pk.data + 32, pyb, 32) == 0,
              "spec vector pubkey matches its seckey");
        ++loaded;
        compare_one(sf, c, k);
        if (secp256k1::bch_schnorr_verify(c.sig.data(), c.msg.data(), k.pub33.data())) ++valid_all;

        // -R twin: same r, s' = 2ed - s.
        Case twin = c;
        Scalar const e = challenge(c.sig.data(), k.pub33.data(), c.msg.data());
        Scalar const ed = e * d;
        set_s(twin.sig.data(), (ed + ed) - Scalar::from_bytes(c.sig.data() + 32));
        compare_one(sf, twin, k);
        if (!secp256k1::bch_schnorr_verify(twin.sig.data(), twin.msg.data(), k.pub33.data())) ++twin_rejected;
    }
    check(loaded == 8, "all eight spec vectors loaded");
    check(valid_all == loaded, "every spec vector verifies on the new verifier");
    check(twin_rejected == loaded, "every -R twin is rejected by the new verifier");
}

void test_random(Surfaces& sf, const std::vector<Key>& keys) {
    std::printf("\n--- random valid and invalid inputs: every surface == old shim ---\n");
    std::uint32_t ctr = 0;
    for (std::size_t ki = 0; ki < keys.size(); ++ki) {
        const Key& k = keys[ki];
        for (std::uint32_t m = 0; m < 3; ++m) {
            Case c;
            c.key = ki;
            c.msg = stream32("bch-diff-msg", ctr++);
            if (secp256k1_schnorr_sign(bch_ctx(), c.sig.data(), c.msg.data(), k.sk.data(),
                                       nullptr, nullptr) != 1) {
                check(false, "shim signer succeeds");
                continue;
            }
            compare_one(sf, c, k);                         // valid

            Scalar const s = Scalar::from_bytes(c.sig.data() + 32);
            Scalar const e = challenge(c.sig.data(), k.pub33.data(), c.msg.data());
            Scalar const ed = e * k.d;

            Case v = c; set_s(v.sig.data(), s + Scalar::one());          compare_one(sf, v, k);
            v = c; set_s(v.sig.data(), (ed + ed) - s);                     compare_one(sf, v, k);  // -R twin
            v = c; std::memset(v.sig.data() + 32, 0, 32);                  compare_one(sf, v, k);  // s == 0
            v = c; std::memcpy(v.sig.data() + 32, kN, 32);                 compare_one(sf, v, k);  // s == n
            v = c; std::memset(v.sig.data() + 32, 0xff, 32);               compare_one(sf, v, k);  // s > n
            v = c; std::memcpy(v.sig.data(), kP, 32);                      compare_one(sf, v, k);  // r == p
            v = c; std::memset(v.sig.data(), 0xff, 32);                    compare_one(sf, v, k);  // r > p
            v = c; v.sig[static_cast<std::size_t>(ctr % 32)] ^= 0x04;      compare_one(sf, v, k);  // r moves
            v = c; v.msg[3] ^= 0x01;                                       compare_one(sf, v, k);  // wrong msg
            v = c; v.key = (ki + 1) % keys.size();                         compare_one(sf, v, keys[v.key]);
            v = c; v.sig = {};                                             compare_one(sf, v, k);  // all zero
            v = c;
            auto const rs = stream32("bch-diff-randsig", ctr);
            auto const ss = stream32("bch-diff-randsig-s", ctr);
            std::memcpy(v.sig.data(), rs.data(), 32);
            std::memcpy(v.sig.data() + 32, ss.data(), 32);
            compare_one(sf, v, k);                                                                 // random
        }
    }
}

void test_shim_batch(const std::vector<Key>& keys) {
    std::printf("\n--- secp256k1_schnorr_verify_batch (Knuth PR #826 contract) ---\n");
    std::size_t const n_max = 40;
    std::vector<std::array<std::uint8_t, 64>> sigs(n_max);
    std::vector<std::array<std::uint8_t, 32>> msgs(n_max);
    for (std::size_t i = 0; i < n_max; ++i) {
        msgs[i] = stream32("bch-diff-batch-msg", static_cast<std::uint32_t>(i));
        (void)secp256k1_schnorr_sign(bch_ctx(), sigs[i].data(), msgs[i].data(),
                                     keys[i % keys.size()].sk.data(), nullptr, nullptr);
    }
    auto run = [&](std::size_t n, const unsigned char* seed) {
        std::vector<const unsigned char*> sp(n), mp(n);
        std::vector<const secp256k1_pubkey*> pp(n);
        for (std::size_t i = 0; i < n; ++i) {
            sp[i] = sigs[i].data();
            mp[i] = msgs[i].data();
            pp[i] = &keys[i % keys.size()].pk;
        }
        return secp256k1_schnorr_verify_batch(bch_ctx(), sp.data(), mp.data(), pp.data(), n, seed);
    };
    auto and_singles = [&](std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            if (secp256k1_schnorr_verify(bch_ctx(), sigs[i].data(), msgs[i].data(),
                                         &keys[i % keys.size()].pk) != 1) return 0;
        }
        return 1;
    };

    std::uint8_t seed[32];
    std::memset(seed, 0x3c, sizeof(seed));
    check(run(0, nullptr) == 1, "n == 0 returns 1");
    check(secp256k1_schnorr_verify_batch(bch_ctx(), nullptr, nullptr, nullptr, 0, nullptr) == 1,
          "n == 0 with NULL arrays returns 1");
    check(run(n_max, nullptr) == 1 && run(n_max, seed) == 1, "honest batch returns 1 (with and without seed)");
    check(run(1, nullptr) == and_singles(1), "n == 1 equals the single verdict");
    const unsigned char* one_sig = sigs[0].data();
    const unsigned char* one_msg = msgs[0].data();
    const secp256k1_pubkey* one_pk = &keys[0].pk;
    check(secp256k1_schnorr_verify_batch(bch_ctx(), nullptr, &one_msg, &one_pk, 1, nullptr) == 0,
          "NULL sig array with n > 0 returns 0");
    check(secp256k1_schnorr_verify_batch(bch_ctx(), &one_sig, nullptr, &one_pk, 1, nullptr) == 0,
          "NULL msg array with n > 0 returns 0");
    check(secp256k1_schnorr_verify_batch(bch_ctx(), &one_sig, &one_msg, nullptr, 1, nullptr) == 0,
          "NULL pubkey array with n > 0 returns 0");
    const unsigned char* null_sig = nullptr;
    check(secp256k1_schnorr_verify_batch(bch_ctx(), &null_sig, &one_msg, &one_pk, 1, nullptr) == 0,
          "NULL element with n > 0 returns 0");
    check(secp256k1_schnorr_verify_batch(nullptr, &one_sig, &one_msg, &one_pk, 1, nullptr) == 0,
          "NULL context returns 0");

    int agree = 0, total = 0;
    for (std::uint32_t t = 0; t < 24; ++t) {
        auto const r = stream32("bch-diff-batch-mix", t);
        std::size_t const n = 1 + (r[0] % n_max);
        std::size_t const bad = r[1] % n;
        auto const saved = sigs[bad];
        if (t % 2 == 1) sigs[bad][40] ^= 0x01;           // odd trials carry one bad entry
        ++total;
        if (run(n, (t % 3 == 0) ? seed : nullptr) == and_singles(n)) ++agree;
        sigs[bad] = saved;
    }
    check(agree == total, "shim batch verdict equals the AND of shim single verdicts");
}

void test_abi_batch(Surfaces& sf, const std::vector<Key>& keys) {
    std::printf("\n--- ufsecp_bch_schnorr_batch_verify / _identify ---\n");
    std::size_t const n = 36;
    std::vector<std::uint8_t> rows(n * 129);
    for (std::size_t i = 0; i < n; ++i) {
        std::uint8_t* row = rows.data() + i * 129;
        const Key& k = keys[i % keys.size()];
        auto const msg = stream32("bch-diff-abi-msg", static_cast<std::uint32_t>(i));
        std::memcpy(row, k.pub33.data(), 33);
        std::memcpy(row + 33, msg.data(), 32);
        (void)secp256k1_schnorr_sign(bch_ctx(), row + 65, msg.data(), k.sk.data(), nullptr, nullptr);
    }
    std::vector<std::uint8_t> res(n, 0xAA);
    check(ufsecp_bch_schnorr_batch_verify(sf.uctx, rows.data(), n) == UFSECP_OK, "ABI honest batch OK");
    check(ufsecp_bch_schnorr_batch_verify(sf.uctx, nullptr, 0) == UFSECP_OK, "ABI empty batch (n=0) OK");
    check(ufsecp_bch_schnorr_batch_verify(sf.uctx, nullptr, 2) == UFSECP_ERR_NULL_ARG, "ABI NULL entries");
    check(ufsecp_bch_schnorr_batch_verify(nullptr, rows.data(), n) == UFSECP_ERR_NULL_ARG, "ABI NULL ctx");

    rows[7 * 129 + 65 + 50] ^= 0x01;                       // entry 7: s moves
    check(ufsecp_bch_schnorr_batch_verify(sf.uctx, rows.data(), n) == UFSECP_ERR_VERIFY_FAIL,
          "ABI batch with one bad entry -> VERIFY_FAIL");
    check(ufsecp_bch_schnorr_batch_identify(sf.uctx, rows.data(), n, res.data()) == UFSECP_OK,
          "ABI identify completes");
    bool exact = true;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint8_t* row = rows.data() + i * 129;
        int const single = (ufsecp_bch_schnorr_verify(sf.uctx, row + 33, row + 65, row) == UFSECP_OK) ? 1 : 0;
        exact = exact && (res[i] == static_cast<std::uint8_t>(single)) && (single == (i == 7 ? 0 : 1));
    }
    check(exact, "ABI identify results equal the single verdicts (only entry 7 invalid)");

    std::vector<std::uint8_t> bad_pk(rows);
    bad_pk[3 * 129] = 0x04;
    check(ufsecp_bch_schnorr_batch_verify(sf.uctx, bad_pk.data(), n) == UFSECP_ERR_BAD_PUBKEY,
          "ABI batch: bad pubkey prefix -> BAD_PUBKEY");
    std::memset(res.data(), 0xAA, n);
    check(ufsecp_bch_schnorr_batch_identify(sf.uctx, bad_pk.data(), n, res.data()) == UFSECP_ERR_BAD_PUBKEY,
          "ABI identify: bad pubkey -> BAD_PUBKEY");
    bool zeroed = true;
    for (std::uint8_t b : res) zeroed = zeroed && (b == 0);
    check(zeroed, "ABI identify zeroes every result on error (fail closed)");

    std::vector<std::uint8_t> bad_s(rows);
    std::memcpy(bad_s.data() + 5 * 129 + 97, kN, 32);
    check(ufsecp_bch_schnorr_batch_verify(sf.uctx, bad_s.data(), n) == UFSECP_ERR_BAD_SIG,
          "ABI batch: s == n -> BAD_SIG");
    std::vector<std::uint8_t> bad_r(rows);
    std::memcpy(bad_r.data() + 5 * 129 + 65, kP, 32);
    check(ufsecp_bch_schnorr_batch_verify(sf.uctx, bad_r.data(), n) == UFSECP_ERR_BAD_SIG,
          "ABI batch: r == p -> BAD_SIG");
    const std::uint8_t* row0 = rows.data();
    check(ufsecp_bch_schnorr_verify(sf.uctx, row0 + 33, bad_s.data() + 5 * 129 + 65, bad_s.data() + 5 * 129)
              == UFSECP_ERR_BAD_SIG, "ABI single: s == n -> BAD_SIG");
    check(ufsecp_bch_schnorr_verify(sf.uctx, row0 + 33, row0 + 65, bad_pk.data() + 3 * 129)
              == UFSECP_ERR_BAD_PUBKEY, "ABI single: bad pubkey -> BAD_PUBKEY");
}

} // namespace

int test_bch_schnorr_differential_run() {
    std::printf("======================================================================\n");
    std::printf("  Differential: BCH 2019 Schnorr verifier vs the pre-delegation shim\n");
    std::printf("  (engine x2, BCHN shim, C ABI; spec vectors; batch == AND of singles)\n");
    std::printf("======================================================================\n");

    Surfaces sf;
    if (ufsecp_ctx_create(&sf.uctx) != UFSECP_OK || sf.uctx == nullptr) {
        check(false, "ufsecp_ctx_create");
        std::printf("\n[bch_schnorr_differential] %d/%d checks passed\n", g_pass, g_pass + g_fail);
        return 1;
    }

    std::vector<Key> keys;
    for (std::uint32_t i = 0; i < 24; ++i) keys.push_back(make_key(i));

    test_spec_vectors(sf);
    test_random(sf, keys);
    check(sf.disagreements == 0, "every verifier surface agrees with the old shim verdict");
    // 8 spec vectors + 24 keys x 3 messages valid; 13 mutations of each invalid.
    check(sf.valid >= 72 && sf.invalid >= 800, "the corpus holds both valid and invalid inputs");
    std::printf("  %d valid, %d invalid inputs, %d disagreements\n",
                sf.valid, sf.invalid, sf.disagreements);

    test_shim_batch(keys);
    test_abi_batch(sf, keys);

    ufsecp_ctx_destroy(sf.uctx);
    std::printf("\n[bch_schnorr_differential] %d/%d checks passed\n", g_pass, g_pass + g_fail);
    return (g_fail > 0) ? 1 : 0;
}

#ifdef STANDALONE_TEST
int main() { return test_bch_schnorr_differential_run(); }
#endif
