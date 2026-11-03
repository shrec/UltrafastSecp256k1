// ============================================================================
// test_lanes_verify_differential.cpp -- lane-parallel ECDSA / BIP-340 verify
//                                       against the scalar verifiers
// ============================================================================
//
// For every lane backend available on this CPU (forced through the explicit
// LaneBackend argument), ecdsa_verify_lanes / schnorr_verify_lanes must return,
// entry by entry, exactly what ecdsa_verify / schnorr_verify return:
//
//   * random valid signatures (and the lane kernel, not a fallback, decides
//     them);
//   * single-field corruptions of r, s, message and public key;
//   * high-S ECDSA (accepted by the scalar verifier's policy), r == 0, s == 0,
//     the infinity key, z == 0 (u1 == 0); BIP-340 s == 0, r >= p, x >= p,
//     x not on the curve;
//   * mixed valid/invalid batches of every size 1 .. 2W+3 (partial groups);
//   * constructed exceptional lanes: R = u1*G + u2*Q = O, and tiny u1/u2 with
//     Q a small multiple of G, which force equal/opposite operands inside the
//     lane additions -- every R = O lane must be flagged and re-verified;
//   * backend None (scalar path) and null arguments.
//
// Batch integration (secp256k1/batch_verify.hpp, lane path opt-in): with each
// available backend forced through lanes::set_lane_verify_backend, every batch
// API (ECDSA / BIP-340 batch verify, _mt, identify-invalid, opaque / BIP-340
// rows and columns) returns the same verdict, invalid-index set and per-row
// results as with the lanes forced off -- on valid batches, 1-3 invalid entries
// at random positions, n not a multiple of the lane width and n around the
// ECDSA (8) and BIP-340 (27) cutoffs -- and the lanes-off result matches the
// per-entry scalar verifiers. With no backend chosen, no batch API takes the
// lane path (lanes::detail::batch_lane_dispatch_count does not move), and
// unavailable / out-of-range backends are refused.
//
// The ECDSA projective x-compare (X == r*Z^2, or X == (r+n)*Z^2 when
// r + n < p) is also unit-tested on constructed field values through
// lanes::detail::ecdsa_xcheck_lanes_for_test, including both sides of the
// r + n < p boundary.
//
// No lane backend available: the standalone binary exits 77 (CTest skip); the
// unified runner module returns 0 with a SKIP line, because a non-advisory
// module that returns 77 counts as a failure there. --require=<name>[,<name>]
// (names as UFSECP_LANES: avx2, avx_ifma, ifma256, ifma512) turns absence of a
// named backend into a failure, for runs under Intel SDE.
//
// Public data only: verification. Test keys come from ct::generator_mul and
// test signatures from ct::ecdsa_sign / ct::schnorr_sign.
// ============================================================================

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "secp256k1/batch_verify.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/ct/sign.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/lanes.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/schnorr.hpp"

namespace {

using secp256k1::ECDSABatchEntry;
using secp256k1::ECDSASignature;
using secp256k1::SchnorrBatchEntry;
using secp256k1::SchnorrSignature;
using secp256k1::fast::FieldElement;
using secp256k1::fast::Point;
using secp256k1::fast::Scalar;
namespace lanes = secp256k1::lanes;
using lanes::LaneBackend;
using lanes::LaneVerifyStats;

constexpr int kSkipCode = 77;

int g_checks = 0;
int g_failures = 0;

void check(bool cond, const char* what, const char* backend, std::size_t index) {
    ++g_checks;
    if (!cond) {
        ++g_failures;
        if (g_failures <= 40)
            std::printf("    FAIL [%s] %s (index %zu)\n", backend, what, index);
    }
}

using Bytes32 = std::array<std::uint8_t, 32>;

struct Rng {
    std::mt19937_64 gen;
    explicit Rng(std::uint64_t seed) : gen(seed) {}
    std::uint64_t u64() { return gen(); }
    Bytes32 bytes32() {
        Bytes32 b{};
        for (std::size_t i = 0; i < 32; i += 8) {
            const std::uint64_t v = gen();
            for (std::size_t j = 0; j < 8; ++j) b[i + j] = static_cast<std::uint8_t>(v >> (8 * j));
        }
        return b;
    }
    Scalar scalar() {
        for (;;) {
            const Bytes32 b = bytes32();
            Scalar s;
            if (Scalar::parse_bytes_strict_nonzero(b.data(), s)) return s;
        }
    }
};

Point gen_mul(const Scalar& k) { return secp256k1::ct::generator_mul(k); }

Scalar scalar_u64(std::uint64_t v) { return Scalar::from_uint64(v); }

Bytes32 x_bytes(const Point& p) { return p.x().to_bytes(); }

// ---- ECDSA cases --------------------------------------------------------------

ECDSABatchEntry ecdsa_valid(Rng& rng) {
    const Scalar d = rng.scalar();
    ECDSABatchEntry e{};
    e.msg_hash = rng.bytes32();
    e.public_key = gen_mul(d);
    e.signature = secp256k1::ct::ecdsa_sign(e.msg_hash, d);
    return e;
}

// R = u1*G + u2*Q = O: Q = d*G, z = -r*d, any r, s.
ECDSABatchEntry ecdsa_infinity(Rng& rng) {
    const Scalar d = rng.scalar();
    const Scalar r = rng.scalar();
    const Scalar s = rng.scalar();
    ECDSABatchEntry e{};
    e.public_key = gen_mul(d);
    e.signature = ECDSASignature{r, s};
    e.msg_hash = (r * d).negate_var().to_bytes();
    return e;
}

// u1 = k, u2 = 1 (s = r, z = k*r), Q = j*G (j may be "negative": n - |j|):
// R = (k + j)*G, valid when r = x(R) mod n. Small k and j make the Strauss
// chain meet equal and opposite operands.
ECDSABatchEntry ecdsa_small(std::uint64_t k, const Scalar& j) {
    ECDSABatchEntry e{};
    e.public_key = gen_mul(j);
    const Scalar sum = scalar_u64(k) + j;
    Scalar r = scalar_u64(1);
    if (!sum.is_zero()) {
        const Scalar rx = Scalar::from_bytes(x_bytes(gen_mul(sum)));
        if (!rx.is_zero()) r = rx;
    }
    e.signature = ECDSASignature{r, r};
    e.msg_hash = (scalar_u64(k) * r).to_bytes();
    return e;
}

std::vector<ECDSABatchEntry> ecdsa_corruptions(const ECDSABatchEntry& base) {
    std::vector<ECDSABatchEntry> out;
    ECDSABatchEntry e = base;
    e.signature.r = base.signature.r + scalar_u64(1);
    out.push_back(e);
    e = base;
    e.signature.s = base.signature.s + scalar_u64(1);
    out.push_back(e);
    e = base;
    e.msg_hash[7] ^= 0x10;
    out.push_back(e);
    e = base;
    e.public_key = base.public_key.add(Point::generator());
    out.push_back(e);
    e = base;
    e.signature.s = base.signature.s.negate_var();   // high-S: accepted by policy
    out.push_back(e);
    e = base;
    e.signature.r = Scalar::zero();
    out.push_back(e);
    e = base;
    e.signature.s = Scalar::zero();
    out.push_back(e);
    e = base;
    e.public_key = Point::infinity();
    out.push_back(e);
    e = base;
    e.msg_hash = Bytes32{};   // z == 0 -> u1 == 0
    out.push_back(e);
    return out;
}

// ---- Schnorr cases -------------------------------------------------------------

SchnorrBatchEntry schnorr_valid(Rng& rng) {
    const auto kp = secp256k1::ct::schnorr_keypair_create(rng.scalar());
    SchnorrBatchEntry e{};
    e.pubkey_x = kp.px;
    e.message = rng.bytes32();
    e.signature = secp256k1::ct::schnorr_sign(kp, e.message, rng.bytes32());
    return e;
}

// R = s*G - e*P = O: s = e*d for the even-Y key d, r = any valid x.
SchnorrBatchEntry schnorr_infinity(Rng& rng) {
    const auto kp = secp256k1::ct::schnorr_keypair_create(rng.scalar());
    SchnorrBatchEntry e{};
    e.pubkey_x = kp.px;
    e.message = rng.bytes32();
    e.signature.r = x_bytes(gen_mul(rng.scalar()));
    std::uint8_t buf[96];
    std::memcpy(buf, e.signature.r.data(), 32);
    std::memcpy(buf + 32, e.pubkey_x.data(), 32);
    std::memcpy(buf + 64, e.message.data(), 32);
    const Scalar ev = Scalar::from_bytes(secp256k1::tagged_hash("BIP0340/challenge", buf, sizeof(buf)));
    e.signature.s = ev * kp.d;
    return e;
}

// An x below p with x^3 + 7 not a square (no curve point).
Bytes32 off_curve_x() {
    for (std::uint64_t v = 1;; ++v) {
        const FieldElement x = FieldElement::from_uint64(v);
        const FieldElement y2 = x.square() * x + FieldElement::from_uint64(7);
        const FieldElement y = y2.sqrt();
        if (!(y.square() == y2)) return x.to_bytes();
    }
}

std::vector<SchnorrBatchEntry> schnorr_corruptions(const SchnorrBatchEntry& base,
                                                   const SchnorrBatchEntry& other) {
    std::vector<SchnorrBatchEntry> out;
    SchnorrBatchEntry e = base;
    e.signature.r[31] ^= 0x01;
    out.push_back(e);
    e = base;
    e.signature.s = base.signature.s + scalar_u64(1);
    out.push_back(e);
    e = base;
    e.message[3] ^= 0x80;
    out.push_back(e);
    e = base;
    e.pubkey_x = other.pubkey_x;
    out.push_back(e);
    e = base;
    e.signature.s = Scalar::zero();
    out.push_back(e);
    e = base;
    e.signature.r.fill(0xFF);   // r >= p
    out.push_back(e);
    e = base;
    e.pubkey_x.fill(0xFF);      // x >= p
    out.push_back(e);
    e = base;
    e.pubkey_x = off_curve_x();
    out.push_back(e);
    return out;
}

// ---- Comparison driver -----------------------------------------------------------

template <class Entry, class LaneFn, class ScalarFn>
void compare_batch(const char* label, const std::vector<Entry>& v, LaneBackend b, LaneFn lane_fn,
                   ScalarFn scalar_fn, LaneVerifyStats* stats_out) {
    if (v.empty()) return;
    std::unique_ptr<bool[]> got_buf(new bool[v.size()]());
    bool* got = got_buf.get();
    LaneVerifyStats st;
    const bool all = lane_fn(v.data(), v.size(), got, b, &st);
    bool expect_all = true;
    for (std::size_t i = 0; i < v.size(); ++i) {
        const bool want = scalar_fn(v[i]);
        expect_all = expect_all && want;
        check(got[i] == want, label, lanes::lane_backend_name(b), i);
    }
    check(all == expect_all, "return value == AND of verdicts", lanes::lane_backend_name(b), v.size());
    check(st.lane_verified + st.prep_fallback + st.exceptional_fallback + st.scalar_only == v.size(),
          "stats account for every entry", lanes::lane_backend_name(b), v.size());
    if (stats_out) *stats_out = st;
}

bool ecdsa_ref(const ECDSABatchEntry& e) {
    return secp256k1::ecdsa_verify(e.msg_hash, e.public_key, e.signature);
}

bool schnorr_ref(const SchnorrBatchEntry& e) {
    return secp256k1::schnorr_verify(e.pubkey_x.data(), e.message.data(), e.signature);
}

bool ecdsa_lane(const ECDSABatchEntry* p, std::size_t n, bool* r, LaneBackend b, LaneVerifyStats* s) {
    return lanes::ecdsa_verify_lanes(p, n, r, b, s);
}

bool schnorr_lane(const SchnorrBatchEntry* p, std::size_t n, bool* r, LaneBackend b, LaneVerifyStats* s) {
    return lanes::schnorr_verify_lanes(p, n, r, b, s);
}

void test_ecdsa(LaneBackend b, Rng& rng) {
    const char* name = lanes::lane_backend_name(b);
    const unsigned W = lanes::lane_backend_width(b);
    LaneVerifyStats st;

    // Random valid, partial final group: every verdict from the lane kernel.
    std::vector<ECDSABatchEntry> valid;
    for (unsigned i = 0; i < 4 * W + 3; ++i) valid.push_back(ecdsa_valid(rng));
    compare_batch("ecdsa random valid", valid, b, ecdsa_lane, ecdsa_ref, &st);
    check(st.lane_verified == valid.size(), "ecdsa valid decided by lanes", name, st.lane_verified);

    // Single-field corruptions, high-S, zero r/s, infinity key, z == 0.
    std::vector<ECDSABatchEntry> bad;
    for (unsigned i = 0; i < W + 1; ++i) {
        const auto c = ecdsa_corruptions(valid[i]);
        bad.insert(bad.end(), c.begin(), c.end());
    }
    compare_batch("ecdsa corruptions", bad, b, ecdsa_lane, ecdsa_ref, &st);
    check(st.lane_verified > 0 && st.prep_fallback > 0, "ecdsa corruptions use both paths", name, 0);

    // Constructed R = O: every lane must be flagged and re-verified (false).
    std::vector<ECDSABatchEntry> inf;
    for (unsigned i = 0; i < 2 * W + 1; ++i) inf.push_back(ecdsa_infinity(rng));
    compare_batch("ecdsa R = O", inf, b, ecdsa_lane, ecdsa_ref, &st);
    check(st.exceptional_fallback == inf.size(), "every R = O lane flagged", name, st.exceptional_fallback);

    // Tiny scalars with Q = +-j*G: equal / opposite operands inside the lanes.
    std::vector<ECDSABatchEntry> small;
    const std::uint64_t ks[] = {1, 2, 3, 5, 7, 31, 32, 33};
    const std::int64_t js[] = {1, 2, 3, -1, -2, -3, -5, -31, -32, -33, 17};
    for (std::uint64_t k : ks)
        for (std::int64_t j : js) {
            const Scalar jj = (j > 0) ? scalar_u64(static_cast<std::uint64_t>(j))
                                      : scalar_u64(static_cast<std::uint64_t>(-j)).negate_var();
            small.push_back(ecdsa_small(k, jj));
        }
    compare_batch("ecdsa tiny scalars / Q = +-jG", small, b, ecdsa_lane, ecdsa_ref, &st);
    check(st.exceptional_fallback > 0, "tiny-scalar set hits exceptional lanes", name, st.exceptional_fallback);
    std::printf("    [%s] ecdsa tiny-scalar set: %zu lane, %zu exceptional, %zu prep\n", name,
                st.lane_verified, st.exceptional_fallback, st.prep_fallback);

    // Mixed batches of every size 1 .. 2W+3 from a shuffled pool.
    std::vector<ECDSABatchEntry> pool = valid;
    pool.insert(pool.end(), bad.begin(), bad.end());
    pool.insert(pool.end(), inf.begin(), inf.end());
    pool.insert(pool.end(), small.begin(), small.end());
    for (unsigned n = 1; n <= 2 * W + 3; ++n) {
        std::vector<ECDSABatchEntry> mix;
        for (unsigned i = 0; i < n; ++i) mix.push_back(pool[rng.u64() % pool.size()]);
        compare_batch("ecdsa mixed batch", mix, b, ecdsa_lane, ecdsa_ref, nullptr);
    }
}

void test_schnorr(LaneBackend b, Rng& rng) {
    const char* name = lanes::lane_backend_name(b);
    const unsigned W = lanes::lane_backend_width(b);
    LaneVerifyStats st;

    std::vector<SchnorrBatchEntry> valid;
    for (unsigned i = 0; i < 4 * W + 3; ++i) valid.push_back(schnorr_valid(rng));
    compare_batch("schnorr random valid", valid, b, schnorr_lane, schnorr_ref, &st);
    check(st.lane_verified == valid.size(), "schnorr valid decided by lanes", name, st.lane_verified);

    std::vector<SchnorrBatchEntry> bad;
    for (unsigned i = 0; i < W + 1; ++i) {
        const auto c = schnorr_corruptions(valid[i], valid[i + 1]);
        bad.insert(bad.end(), c.begin(), c.end());
    }
    compare_batch("schnorr corruptions", bad, b, schnorr_lane, schnorr_ref, &st);
    check(st.lane_verified > 0 && st.prep_fallback > 0, "schnorr corruptions use both paths", name, 0);

    std::vector<SchnorrBatchEntry> inf;
    for (unsigned i = 0; i < 2 * W + 1; ++i) inf.push_back(schnorr_infinity(rng));
    compare_batch("schnorr R = O", inf, b, schnorr_lane, schnorr_ref, &st);
    check(st.exceptional_fallback == inf.size(), "every schnorr R = O lane flagged", name,
          st.exceptional_fallback);

    std::vector<SchnorrBatchEntry> pool = valid;
    pool.insert(pool.end(), bad.begin(), bad.end());
    pool.insert(pool.end(), inf.begin(), inf.end());
    for (unsigned n = 1; n <= 2 * W + 3; ++n) {
        std::vector<SchnorrBatchEntry> mix;
        for (unsigned i = 0; i < n; ++i) mix.push_back(pool[rng.u64() % pool.size()]);
        compare_batch("schnorr mixed batch", mix, b, schnorr_lane, schnorr_ref, nullptr);
    }
}

// ---- Projective x-compare ------------------------------------------------------------

using Limbs = std::array<std::uint64_t, 4>;

constexpr Limbs kN = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL, 0xFFFFFFFFFFFFFFFEULL,
                      0xFFFFFFFFFFFFFFFFULL};
constexpr Limbs kPMinusN = {0x402DA1722FC9BAEEULL, 0x4551231950B75FC4ULL, 1u, 0u};

bool lt(const Limbs& a, const Limbs& b) {
    for (int i = 3; i >= 0; --i)
        if (a[i] != b[i]) return a[i] < b[i];
    return false;
}

// a + b mod 2^256 (portable: no __int128, so MSVC builds the test too).
Limbs add_limbs(const Limbs& a, const Limbs& b) {
    Limbs r{};
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        const std::uint64_t t = a[i] + b[i];
        const std::uint64_t c1 = (t < a[i]) ? 1u : 0u;
        r[i] = t + carry;
        carry = c1 | ((r[i] < t) ? 1u : 0u);
    }
    return r;
}

FieldElement fe(const Limbs& l) { return FieldElement::from_limbs(l); }

// Independent reference of the compare (FieldElement arithmetic).
bool xcheck_ref(const Limbs& X, const Limbs& Z, const Limbs& r) {
    const FieldElement z2 = fe(Z).square();
    if (fe(X) == fe(r) * z2) return true;
    if (!lt(r, kPMinusN)) return false;
    return fe(X) == fe(add_limbs(r, kN)) * z2;
}

void test_xcheck(LaneBackend b, Rng& rng) {
    const char* name = lanes::lane_backend_name(b);
    std::vector<Limbs> X, Z, R;
    auto add_case = [&](const Limbs& x, const Limbs& z, const Limbs& r) {
        X.push_back(x);
        Z.push_back(z);
        R.push_back(r);
    };
    auto rand_fe = [&]() { return fe(Limbs{rng.u64(), rng.u64(), rng.u64(), rng.u64()}).limbs(); };
    auto times_z2 = [&](const Limbs& v, const Limbs& z) { return (fe(v) * fe(z).square()).limbs(); };

    for (int i = 0; i < 24; ++i) {
        const Limbs z = rand_fe();
        const Limbs r_big = rng.scalar().limbs();                       // r >= p - n (overwhelmingly)
        const Limbs r_small = {rng.u64(), rng.u64(), 0u, 0u};           // r <  p - n
        add_case(times_z2(r_big, z), z, r_big);                         // X = r Z^2       -> 1
        add_case(times_z2(r_small, z), z, r_small);                     // X = r Z^2       -> 1
        add_case(times_z2(add_limbs(r_small, kN), z), z, r_small);      // X = (r+n) Z^2   -> 1
        add_case(rand_fe(), z, r_big);                                  // unrelated X     -> 0
        // r >= p - n: X = (r + n mod p) Z^2 must NOT match (branch disabled).
        add_case(times_z2(add_limbs(r_big, kN), z), z, r_big);
    }
    // The r + n < p boundary itself.
    const Limbs z = rand_fe();
    Limbs edge = kPMinusN;
    edge[0] -= 1;                                                       // p - n - 1: branch on
    add_case(times_z2(add_limbs(edge, kN), z), z, edge);
    add_case(times_z2(add_limbs(kPMinusN, kN), z), z, kPMinusN);        // p - n: branch off
    Limbs past = kPMinusN;
    past[0] += 5;
    add_case(times_z2(add_limbs(past, kN), z), z, past);
    add_case(Limbs{0, 0, 0, 0}, Limbs{0, 0, 0, 0}, Limbs{1, 0, 0, 0});   // Z = 0 (R = O) compare alone

    const std::size_t n = X.size();
    std::vector<std::uint64_t> fx(4 * n), fz(4 * n), fr(4 * n);
    for (std::size_t i = 0; i < n; ++i)
        for (int j = 0; j < 4; ++j) {
            fx[4 * i + j] = X[i][j];
            fz[4 * i + j] = Z[i][j];
            fr[4 * i + j] = R[i][j];
        }
    std::vector<std::uint8_t> out(n, 0xEE);
    const bool ran = lanes::detail::ecdsa_xcheck_lanes_for_test(b, fx.data(), fz.data(), fr.data(),
                                                                 out.data(), n);
    check(ran, "xcheck hook available", name, 0);
    int ones = 0;
    for (std::size_t i = 0; i < n; ++i) {
        const bool want = xcheck_ref(X[i], Z[i], R[i]);
        ones += want ? 1 : 0;
        check(out[i] == (want ? 1 : 0), "projective x-compare", name, i);
    }
    // The constructed cases must exercise both verdicts and the rn branch.
    check(ones > 24 * 2 && ones < static_cast<int>(n), "xcheck covers both verdicts", name, 0);
    check(out[n - 4] == 1 && out[n - 3] == 0 && out[n - 2] == 0, "r + n < p boundary", name, n);
}

void test_scalar_backend_and_args(Rng& rng) {
    std::vector<ECDSABatchEntry> ev;
    std::vector<SchnorrBatchEntry> sv;
    for (int i = 0; i < 5; ++i) {
        ev.push_back(ecdsa_valid(rng));
        sv.push_back(schnorr_valid(rng));
    }
    ev[2].signature.s = ev[2].signature.s + scalar_u64(1);
    sv[3].message[0] ^= 1;
    LaneVerifyStats st;
    compare_batch("ecdsa backend None", ev, LaneBackend::None, ecdsa_lane, ecdsa_ref, &st);
    check(st.scalar_only == ev.size(), "None -> scalar path", "off", st.scalar_only);
    compare_batch("schnorr backend None", sv, LaneBackend::None, schnorr_lane, schnorr_ref, &st);
    check(st.scalar_only == sv.size(), "None -> scalar path", "off", st.scalar_only);

    bool res[2] = {true, true};
    check(lanes::ecdsa_verify_lanes(nullptr, 0, nullptr, LaneBackend::None), "n == 0 -> true", "off", 0);
    check(!lanes::ecdsa_verify_lanes(nullptr, 2, res, LaneBackend::None) && !res[0] && !res[1],
          "null entries -> false, results cleared", "off", 0);
    check(!lanes::schnorr_verify_lanes(sv.data(), 2, nullptr, LaneBackend::None), "null results -> false",
          "off", 0);
}

constexpr LaneBackend kBackends[] = {LaneBackend::Avx2x4, LaneBackend::AvxIfma256x4, LaneBackend::Ifma256x4,
                                     LaneBackend::Ifma512x8};

// ---- Batch API integration ---------------------------------------------------------
//
// The batch APIs of secp256k1/batch_verify.hpp take the lane path only when a
// backend was chosen explicitly (lanes::set_lane_verify_backend, or UFSECP_LANES
// naming an available backend). For every available backend, forced through
// set_lane_verify_backend, each batch API must give the same verdict, the same
// invalid-index set and the same out_results as with the lanes forced off
// (set_lane_verify_backend(None)), on valid batches, batches with 1-3 invalid
// entries at random positions, n not a multiple of the lane width, and n around
// the ECDSA (8) and BIP-340 (27) individual cutoffs. The lanes-off result is
// itself checked against the per-entry scalar verifiers. Whether the lane path
// ran is read from lanes::detail::batch_lane_dispatch_count().

using secp256k1::lanes::detail::batch_lane_dispatch_count;

constexpr std::size_t kSchnorrIndividualCutoff = 27;   // batch_verify.cpp
constexpr std::size_t kEcdsaRowBytes = 32 + 33 + 64;
constexpr std::size_t kSchnorrRowBytes = 32 + 32 + 64;

struct BatchOutcome {
    bool verdict = false;                 // *_batch_verify
    bool verdict_mt = false;              // *_batch_verify_mt (2 threads)
    std::vector<std::size_t> invalid;     // *_batch_identify_invalid
    bool cols_ok = false;                 // opaque / BIP-340 columns
    std::vector<std::uint8_t> cols;
    bool rows_ok = false;                 // opaque / BIP-340 rows
    std::vector<std::uint8_t> rows;
};

void put_le_scalar(std::vector<std::uint8_t>& out, const Scalar& s) {
    for (const std::uint64_t limb : s.limbs())
        for (int j = 0; j < 8; ++j) out.push_back(static_cast<std::uint8_t>(limb >> (8 * j)));
}

BatchOutcome ecdsa_outcome(const std::vector<ECDSABatchEntry>& v) {
    BatchOutcome o;
    const std::size_t n = v.size();
    o.verdict = secp256k1::ecdsa_batch_verify(v.data(), n);
    o.verdict_mt = secp256k1::ecdsa_batch_verify_mt(v.data(), n, 2);
    o.invalid = secp256k1::ecdsa_batch_identify_invalid(v.data(), n);
    std::vector<std::uint8_t> d, p, s, rows;
    for (const ECDSABatchEntry& e : v) {
        const auto pk = e.public_key.to_compressed();
        d.insert(d.end(), e.msg_hash.begin(), e.msg_hash.end());
        p.insert(p.end(), pk.begin(), pk.end());
        put_le_scalar(s, e.signature.r);
        put_le_scalar(s, e.signature.s);
        rows.insert(rows.end(), e.msg_hash.begin(), e.msg_hash.end());
        rows.insert(rows.end(), pk.begin(), pk.end());
        put_le_scalar(rows, e.signature.r);
        put_le_scalar(rows, e.signature.s);
    }
    o.cols.assign(n, 0xEE);
    o.cols_ok = secp256k1::ecdsa_batch_verify_opaque_columns(d.data(), p.data(), s.data(), n, o.cols.data(), 1);
    o.rows.assign(n, 0xEE);
    o.rows_ok = secp256k1::ecdsa_batch_verify_opaque_rows(rows.data(), kEcdsaRowBytes, n, o.rows.data(), 1);
    return o;
}

BatchOutcome schnorr_outcome(const std::vector<SchnorrBatchEntry>& v) {
    BatchOutcome o;
    const std::size_t n = v.size();
    o.verdict = secp256k1::schnorr_batch_verify(v.data(), n);
    o.verdict_mt = secp256k1::schnorr_batch_verify_mt(v.data(), n, 2);
    o.invalid = secp256k1::schnorr_batch_identify_invalid(v.data(), n);
    std::vector<std::uint8_t> d, x, s, rows;
    for (const SchnorrBatchEntry& e : v) {
        const auto sig = e.signature.to_bytes();
        d.insert(d.end(), e.message.begin(), e.message.end());
        x.insert(x.end(), e.pubkey_x.begin(), e.pubkey_x.end());
        s.insert(s.end(), sig.begin(), sig.end());
        rows.insert(rows.end(), e.message.begin(), e.message.end());
        rows.insert(rows.end(), e.pubkey_x.begin(), e.pubkey_x.end());
        rows.insert(rows.end(), sig.begin(), sig.end());
    }
    o.cols.assign(n, 0xEE);
    o.cols_ok = secp256k1::schnorr_batch_verify_bip340_columns(d.data(), x.data(), s.data(), n, o.cols.data(), 1);
    o.rows.assign(n, 0xEE);
    o.rows_ok = secp256k1::schnorr_batch_verify_bip340_rows(rows.data(), kSchnorrRowBytes, n, o.rows.data(), 1);
    return o;
}

// lanes on vs lanes off, then lanes off vs the per-entry scalar reference.
//   ref_invalid:  indices whose scalar verifier rejects (ascending)
//   strict_ok:    expected *_batch_verify verdict (ECDSA: also requires low-S)
//   opaque_ok:    expected rows/columns verdict (every row valid)
void compare_outcomes(const char* what, const BatchOutcome& off, const BatchOutcome& on,
                      const std::vector<std::size_t>& ref_invalid, bool strict_ok, bool opaque_ok,
                      const char* name, std::size_t n) {
    char label[160];
    auto lab = [&](const char* field) {
        std::snprintf(label, sizeof(label), "%s n=%zu: %s", what, n, field);
        return label;
    };
    check(on.verdict == off.verdict, lab("batch verdict lanes == scalar"), name, n);
    check(on.verdict_mt == off.verdict_mt, lab("_mt verdict lanes == scalar"), name, n);
    check(on.invalid == off.invalid, lab("identify-invalid set lanes == scalar"), name, n);
    check(on.cols_ok == off.cols_ok && on.cols == off.cols, lab("columns lanes == scalar"), name, n);
    check(on.rows_ok == off.rows_ok && on.rows == off.rows, lab("rows lanes == scalar"), name, n);

    check(off.verdict == strict_ok && off.verdict_mt == strict_ok, lab("scalar batch verdict"), name, n);
    check(off.invalid == ref_invalid, lab("scalar identify-invalid == reference"), name, n);
    check(off.cols_ok == opaque_ok && off.rows_ok == opaque_ok, lab("scalar opaque verdict"), name, n);
    std::size_t next = 0;
    bool rows_match = true;
    for (std::size_t i = 0; i < n; ++i) {
        const bool bad = next < ref_invalid.size() && ref_invalid[next] == i;
        if (bad) ++next;
        const std::uint8_t want = bad ? 0u : 1u;
        rows_match = rows_match && off.cols[i] == want && off.rows[i] == want;
    }
    check(rows_match, lab("scalar out_results == reference"), name, n);
}

// n entries cycled from `pool` at a random start, then `k` distinct random
// positions corrupted with `corrupt`.
template <class Entry, class Corrupt>
std::vector<Entry> make_batch(const std::vector<Entry>& pool, std::size_t n, unsigned k, Rng& rng,
                              Corrupt corrupt, std::vector<std::size_t>& bad_pos) {
    std::vector<Entry> v;
    const std::size_t start = static_cast<std::size_t>(rng.u64() % pool.size());
    for (std::size_t i = 0; i < n; ++i) v.push_back(pool[(start + i) % pool.size()]);
    bad_pos.clear();
    while (bad_pos.size() < k) {
        const std::size_t pos = static_cast<std::size_t>(rng.u64() % n);
        bool seen = false;
        for (std::size_t q : bad_pos) seen = seen || q == pos;
        if (!seen) bad_pos.push_back(pos);
    }
    for (std::size_t pos : bad_pos) corrupt(v[pos], rng);
    std::sort(bad_pos.begin(), bad_pos.end());
    return v;
}

void ecdsa_corrupt(ECDSABatchEntry& e, Rng& rng) {
    switch (rng.u64() % 3) {
    case 0: e.signature.s = e.signature.s + scalar_u64(1); break;
    case 1: e.signature.r = e.signature.r + scalar_u64(1); break;
    default: e.msg_hash[5] ^= 0x40; break;
    }
}

void schnorr_corrupt(SchnorrBatchEntry& e, Rng& rng) {
    switch (rng.u64() % 3) {
    case 0: e.signature.s = e.signature.s + scalar_u64(1); break;
    case 1: e.signature.r[31] ^= 0x01; break;
    default: e.message[9] ^= 0x02; break;
    }
}

template <class Entry, class Ref>
std::vector<std::size_t> reference_invalid(const std::vector<Entry>& v, Ref ref) {
    std::vector<std::size_t> out;
    for (std::size_t i = 0; i < v.size(); ++i)
        if (!ref(v[i])) out.push_back(i);
    return out;
}

bool all_low_s(const std::vector<ECDSABatchEntry>& v) {
    for (const ECDSABatchEntry& e : v)
        if (!e.signature.is_low_s()) return false;
    return true;
}

// Distinct sizes >= 1, in order of first appearance.
std::vector<std::size_t> sizes_of(std::initializer_list<std::size_t> list) {
    std::vector<std::size_t> out;
    for (std::size_t n : list) {
        bool seen = n == 0;
        for (std::size_t m : out) seen = seen || m == n;
        if (!seen) out.push_back(n);
    }
    return out;
}

void test_batch_integration(LaneBackend b, Rng& rng) {
    const char* name = lanes::lane_backend_name(b);
    const std::size_t W = lanes::lane_backend_width(b);

    std::vector<ECDSABatchEntry> epool;
    std::vector<SchnorrBatchEntry> spool;
    for (int i = 0; i < 72; ++i) {
        epool.push_back(ecdsa_valid(rng));
        spool.push_back(schnorr_valid(rng));
    }
    std::vector<std::size_t> bad;

    // ECDSA: around the lane width, the individual cutoff (8) and a 64-block.
    for (std::size_t n : sizes_of({1, 2, 3, W - 1, W, W + 1, 8, 9, 2 * W + 3, 64, 67})) {
        for (unsigned k = 0; k <= 3 && k <= n; ++k) {
            const auto v = make_batch(epool, n, k, rng, ecdsa_corrupt, bad);
            const auto ref = reference_invalid(v, ecdsa_ref);
            check(ref == bad, "ecdsa corrupted positions are exactly the invalid ones", name, n);
            check(lanes::set_lane_verify_backend(LaneBackend::None), "force lanes off", name, n);
            const std::uint64_t c0 = batch_lane_dispatch_count();
            const BatchOutcome off = ecdsa_outcome(v);
            check(batch_lane_dispatch_count() == c0, "ecdsa lanes off: lane path not taken", name, n);
            check(lanes::set_lane_verify_backend(b), "force backend", name, n);
            const std::uint64_t c1 = batch_lane_dispatch_count();
            const BatchOutcome on = ecdsa_outcome(v);
            check(batch_lane_dispatch_count() > c1, "ecdsa lanes forced: lane path taken", name, n);
            compare_outcomes("ecdsa", off, on, ref, ref.empty() && all_low_s(v), ref.empty(), name, n);
        }
    }

    // ECDSA high-S (valid): strict batch rejects, opaque accepts, identify-invalid
    // does not list it -- with lanes on and off alike.
    {
        const std::size_t n = W + 1;
        auto v = make_batch(epool, n, 0, rng, ecdsa_corrupt, bad);
        v[n / 2].signature.s = v[n / 2].signature.s.negate_var();
        const auto ref = reference_invalid(v, ecdsa_ref);
        check(ref.empty(), "high-S entry is valid for ecdsa_verify", name, n);
        check(lanes::set_lane_verify_backend(LaneBackend::None), "force lanes off", name, n);
        const BatchOutcome off = ecdsa_outcome(v);
        check(lanes::set_lane_verify_backend(b), "force backend", name, n);
        const BatchOutcome on = ecdsa_outcome(v);
        compare_outcomes("ecdsa high-S", off, on, ref, false, true, name, n);
    }

    // BIP-340: around the lane width and the individual cutoff (27); n >= 28 keeps
    // the randomized MSM for the verdict, the identify-invalid and per-row
    // out_results still go through the lanes.
    const std::size_t C = kSchnorrIndividualCutoff;
    for (std::size_t n : sizes_of({1, 2, W + 1, 2 * W + 3, C - 1, C, C + 1, C + 2, 64})) {
        for (unsigned k = 0; k <= 3 && k <= n; ++k) {
            const auto v = make_batch(spool, n, k, rng, schnorr_corrupt, bad);
            const auto ref = reference_invalid(v, schnorr_ref);
            check(ref == bad, "schnorr corrupted positions are exactly the invalid ones", name, n);
            check(lanes::set_lane_verify_backend(LaneBackend::None), "force lanes off", name, n);
            const std::uint64_t c0 = batch_lane_dispatch_count();
            const BatchOutcome off = schnorr_outcome(v);
            check(batch_lane_dispatch_count() == c0, "schnorr lanes off: lane path not taken", name, n);
            check(lanes::set_lane_verify_backend(b), "force backend", name, n);
            const std::uint64_t c1 = batch_lane_dispatch_count();
            const BatchOutcome on = schnorr_outcome(v);
            check(batch_lane_dispatch_count() > c1, "schnorr lanes forced: lane path taken", name, n);
            compare_outcomes("schnorr", off, on, ref, ref.empty(), ref.empty(), name, n);
        }
    }

    // Which calls dispatch, with the backend forced, on valid batches:
    //   ECDSA n == 1 stays scalar, n >= 2 uses the lanes;
    //   BIP-340 n <= 27 uses the lanes, n >= 28 keeps the MSM.
    {
        check(lanes::set_lane_verify_backend(b), "force backend", name, 0);
        const auto e1 = make_batch(epool, 1, 0, rng, ecdsa_corrupt, bad);
        const auto e2 = make_batch(epool, 2, 0, rng, ecdsa_corrupt, bad);
        const auto sc = make_batch(spool, C, 0, rng, schnorr_corrupt, bad);
        const auto sm = make_batch(spool, C + 1, 0, rng, schnorr_corrupt, bad);
        std::uint64_t c = batch_lane_dispatch_count();
        check(secp256k1::ecdsa_batch_verify(e1.data(), 1) && batch_lane_dispatch_count() == c,
              "ecdsa n == 1 stays scalar", name, 1);
        c = batch_lane_dispatch_count();
        check(secp256k1::ecdsa_batch_verify(e2.data(), 2) && batch_lane_dispatch_count() == c + 1,
              "ecdsa n == 2 uses lanes", name, 2);
        c = batch_lane_dispatch_count();
        check(secp256k1::schnorr_batch_verify(sc.data(), C) && batch_lane_dispatch_count() == c + 1,
              "schnorr n == cutoff uses lanes", name, C);
        c = batch_lane_dispatch_count();
        check(secp256k1::schnorr_batch_verify(sm.data(), C + 1) && batch_lane_dispatch_count() == c,
              "schnorr n > cutoff keeps the MSM", name, C + 1);
    }
    lanes::reset_lane_verify_backend();
}

// Opt-in rule, refusals, and: with no backend forced, no batch API takes the
// lane path. Runs whether or not a lane backend is available.
void test_batch_default_off(Rng& rng) {
    lanes::reset_lane_verify_backend();
    const lanes::LaneSelection& sel = lanes::lane_selection();
    const bool env_forced = sel.reason == lanes::LaneSelectionReason::Forced;
    const LaneBackend expect = env_forced ? sel.backend : LaneBackend::None;
    check(lanes::lane_verify_backend() == expect,
          "lane_verify_backend(): UFSECP_LANES only when forced, never the default pick", "default", 0);

    // Refusals leave the override untouched.
    check(lanes::set_lane_verify_backend(LaneBackend::None) &&
              lanes::lane_verify_backend() == LaneBackend::None,
          "set None -> scalar path", "default", 0);
    check(!lanes::set_lane_verify_backend(static_cast<LaneBackend>(lanes::kLaneBackendCount)) &&
              lanes::lane_verify_backend() == LaneBackend::None,
          "out-of-range backend refused", "default", 0);
    const unsigned avail = lanes::available_lane_backends();
    for (LaneBackend b : kBackends) {
        if ((avail & lanes::lane_backend_bit(b)) != 0) continue;
        check(!lanes::set_lane_verify_backend(b) && lanes::lane_verify_backend() == LaneBackend::None,
              "unavailable backend refused", lanes::lane_backend_name(b), 0);
    }
    lanes::reset_lane_verify_backend();
    check(lanes::lane_verify_backend() == expect, "reset -> back to the UFSECP_LANES rule", "default", 0);

    if (env_forced && sel.backend != LaneBackend::None) {
        std::printf("    note: UFSECP_LANES forces %s, so the no-opt-in check is skipped\n",
                    lanes::lane_backend_name(sel.backend));
        return;
    }
    // Every batch API, valid and mixed, with no backend chosen: counter unchanged.
    std::vector<ECDSABatchEntry> ev;
    std::vector<SchnorrBatchEntry> sv;
    for (int i = 0; i < 9; ++i) {
        ev.push_back(ecdsa_valid(rng));
        sv.push_back(schnorr_valid(rng));
    }
    const std::uint64_t c0 = batch_lane_dispatch_count();
    (void)ecdsa_outcome(ev);
    (void)schnorr_outcome(sv);
    ev[4].signature.s = ev[4].signature.s + scalar_u64(1);
    sv[6].message[0] ^= 1;
    const BatchOutcome eo = ecdsa_outcome(ev);
    const BatchOutcome so = schnorr_outcome(sv);
    check(batch_lane_dispatch_count() == c0, "no backend chosen: lane path never taken", "default", 0);
    check(!eo.verdict && eo.invalid == std::vector<std::size_t>{4}, "default ecdsa batch still correct",
          "default", 0);
    check(!so.verdict && so.invalid == std::vector<std::size_t>{6}, "default schnorr batch still correct",
          "default", 0);
}

// --require=<name>[,<name>...] -> mask of required backends; -1 on a bad name.
long parse_require(const char* arg) {
    long mask = 0;
    std::string s(arg);
    std::size_t pos = 0;
    while (pos <= s.size()) {
        const std::size_t comma = s.find(',', pos);
        const std::string tok = s.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        bool found = false;
        for (LaneBackend b : kBackends)
            if (tok == lanes::lane_backend_name(b)) {
                mask |= static_cast<long>(lanes::lane_backend_bit(b));
                found = true;
            }
        if (!found) return -1;
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return mask;
}

// 0 = pass, 1 = fail, kSkipCode = no lane backend (and none required).
int run_lanes_verify_differential(unsigned required_mask) {
    std::printf("  lanes verify differential (ECDSA + BIP-340 vs scalar verifiers)\n");
    const unsigned avail = lanes::available_lane_backends();
    Rng rng(0x5ECB256C0FFEE001ULL);
    test_scalar_backend_and_args(rng);
    test_batch_default_off(rng);

    int tested = 0;
    for (LaneBackend b : kBackends) {
        const bool have = (avail & lanes::lane_backend_bit(b)) != 0;
        if ((required_mask & lanes::lane_backend_bit(b)) != 0 && !have) {
            std::printf("    FAIL: required backend %s is not available\n", lanes::lane_backend_name(b));
            ++g_failures;
        }
        if (!have) continue;
        const int before = g_failures;
        test_xcheck(b, rng);
        test_ecdsa(b, rng);
        test_schnorr(b, rng);
        test_batch_integration(b, rng);
        std::printf("    [%s] width %u: %s\n", lanes::lane_backend_name(b), lanes::lane_backend_width(b),
                    g_failures == before ? "PASS" : "FAIL");
        ++tested;
    }
    std::printf("  %d checks, %d failures, %d backend(s) tested\n", g_checks, g_failures, tested);
    if (g_failures != 0) return 1;
    if (tested == 0) {
        std::printf("  SKIP: no lane backend available on this CPU/build\n");
        return kSkipCode;
    }
    return 0;
}

} // namespace

// Unified-runner entry: a skip is not a failure there (non-advisory module).
int test_lanes_verify_differential_run() {
    g_checks = 0;
    g_failures = 0;
    const int rc = run_lanes_verify_differential(0);
    return rc == kSkipCode ? 0 : rc;
}

#ifdef STANDALONE_TEST
int main(int argc, char** argv) {
    unsigned required = 0;
    for (int i = 1; i < argc; ++i) {
        const char* a = argv[i];
        if (std::strncmp(a, "--require=", 10) == 0) {
            const long m = parse_require(a + 10);
            if (m < 0) {
                std::fprintf(stderr, "unknown backend in %s (expected avx2|avx_ifma|ifma256|ifma512)\n", a);
                return 2;
            }
            required |= static_cast<unsigned>(m);
        } else {
            std::fprintf(stderr, "usage: %s [--require=<backend>[,<backend>...]]\n", argv[0]);
            return 2;
        }
    }
    return run_lanes_verify_differential(required);
}
#endif
