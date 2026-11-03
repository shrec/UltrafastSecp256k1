// ============================================================================
// lanes/lanes_verify.cpp -- ECDSA / BIP-340 verification, several signatures
//                           per SIMD instruction stream (scalar side)
// ============================================================================
//
// Generic code: compiled with the library's normal flags, never with an ISA
// flag (src/cpu/CMakeLists.txt gives it only the SECP256K1_LANES_HAVE_* macros).
// It does everything that is not lane arithmetic -- parse and policy exactly as
// ecdsa_verify / schnorr_verify, s^-1 by one batch inversion per group, the
// affine pubkey by one batch normalisation per group, BIP-340 lift_x and
// challenge -- and hands plain arrays to the per-ISA kernel
// (lanes_verify_kernels.hpp) through the table lanes_tables.hpp declares. Any
// entry the kernel cannot decide (prep reject, exceptional lane) gets the
// scalar verifier's verdict, so the result always equals the scalar one.
//
// PUBLIC DATA ONLY (CT-VERIFY): every input is a signature, key or message.
// Variable time is correct here; nothing secret passes through.
// ============================================================================

#include "secp256k1/lanes.hpp"

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "secp256k1/batch_verify.hpp"
#include "secp256k1/ecdsa.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/schnorr.hpp"
#include "lanes_tables.hpp"

namespace secp256k1::lanes {

namespace {

using fast::FieldElement;
using fast::Point;
using fast::Scalar;
using detail::kLaneVerifyMaxLanes;
using detail::LaneVerifyGroup;
using detail::LaneVerifyKernels;
using detail::LaneVerifyResult;

// p, n and p - n, 4x64 little endian.
constexpr std::uint64_t kP[4] = {0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
                                 0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL};
constexpr std::uint64_t kN[4] = {0xBFD25E8CD0364141ULL, 0xBAAEDCE6AF48A03BULL,
                                 0xFFFFFFFFFFFFFFFEULL, 0xFFFFFFFFFFFFFFFFULL};
constexpr std::uint64_t kPMinusN[4] = {0x402DA1722FC9BAEEULL, 0x4551231950B75FC4ULL, 1u, 0u};

bool lt256(const std::uint64_t* a, const std::uint64_t* b) noexcept {
    for (int i = 3; i >= 0; --i)
        if (a[i] != b[i]) return a[i] < b[i];
    return false;
}

// rn = r + n when r + n < p (the second x-candidate of ecdsa_check_xcoord),
// returning true; otherwise rn = r and false. r < n.
bool r_plus_n(const std::uint64_t* r, std::uint64_t* rn) noexcept {
    if (!lt256(r, kPMinusN)) {
        std::memcpy(rn, r, 32);
        return false;
    }
    std::uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        const std::uint64_t t = r[i] + kN[i];
        const std::uint64_t c1 = (t < r[i]) ? 1u : 0u;
        rn[i] = t + carry;
        carry = c1 | ((rn[i] < t) ? 1u : 0u);
    }
    return true;   // r + n < p < 2^256: the final carry is 0
}

void put4(std::uint64_t* dst, const std::array<std::uint64_t, 4>& limbs) noexcept {
    for (int i = 0; i < 4; ++i) dst[i] = limbs[i];
}

// 32 big-endian bytes -> 4x64 little endian.
void be32_to_limbs(const std::uint8_t* b, std::uint64_t* out) noexcept {
    for (int i = 0; i < 4; ++i) {
        std::uint64_t v = 0;
        for (int j = 0; j < 8; ++j) v = (v << 8) | b[(3 - i) * 8 + j];
        out[i] = v;
    }
}

const LaneVerifyKernels* verify_kernels(LaneBackend b) noexcept {
    // Availability (compiled AND supported by the masked CPU view) is the
    // field dispatcher's decision; a verify table exists for exactly the
    // backends whose field table does.
    if (lane_field_kernels(b) == nullptr) return nullptr;
    switch (b) {
#if defined(SECP256K1_LANES_HAVE_AVX2)
    case LaneBackend::Avx2x4: return &detail::kLaneVerifyKernelsAvx2;
#endif
#if defined(SECP256K1_LANES_HAVE_AVX_IFMA)
    case LaneBackend::AvxIfma256x4: return &detail::kLaneVerifyKernelsAvxIfma;
#endif
#if defined(SECP256K1_LANES_HAVE_IFMA256)
    case LaneBackend::Ifma256x4: return &detail::kLaneVerifyKernelsIfma256;
#endif
#if defined(SECP256K1_LANES_HAVE_IFMA512)
    case LaneBackend::Ifma512x8: return &detail::kLaneVerifyKernelsIfma512;
#endif
    default: return nullptr;
    }
}

// Unused lanes of a partial group repeat lane 0 (valid inputs, result ignored).
void pad_group(LaneVerifyGroup& g, unsigned used, unsigned width) noexcept {
    for (unsigned l = used; l < width; ++l) {
        std::memcpy(g.u1 + 4 * l, g.u1, 32);
        std::memcpy(g.u2 + 4 * l, g.u2, 32);
        std::memcpy(g.qx + 4 * l, g.qx, 32);
        std::memcpy(g.qy + 4 * l, g.qy, 32);
        std::memcpy(g.r + 4 * l, g.r, 32);
        std::memcpy(g.rn + 4 * l, g.rn, 32);
        g.rn_ok |= (g.rn_ok & 1u) << l;
    }
}

bool on_curve(const FieldElement& x, const FieldElement& y) {
    return y.square() == x.square() * x + FieldElement::from_uint64(7);
}

bool all_true(const bool* r, std::size_t n) noexcept {
    for (std::size_t i = 0; i < n; ++i)
        if (!r[i]) return false;
    return true;
}

bool ecdsa_scalar(const ECDSABatchEntry& e) noexcept {
    return ecdsa_verify(e.msg_hash, e.public_key, e.signature);
}

bool schnorr_scalar(const SchnorrBatchEntry& e) noexcept {
    return schnorr_verify(e.pubkey_x.data(), e.message.data(), e.signature);
}

bool bad_args(const void* entries, std::size_t n, bool* results, LaneVerifyStats* stats) noexcept {
    if (stats) *stats = LaneVerifyStats{};
    if (entries != nullptr && results != nullptr) return false;
    if (results != nullptr)
        for (std::size_t i = 0; i < n; ++i) results[i] = false;
    return true;
}

} // namespace

// low-S / high-S policy: identical to ecdsa_verify(), which accepts both by
// design (see ecdsa.cpp). BIP-62 enforcement (sig.is_low_s()) stays with the
// caller, e.g. ecdsa_batch_verify()'s pre-validation, which runs before this path.
bool ecdsa_verify_lanes(const ECDSABatchEntry* entries, std::size_t n, bool* results,
                        LaneBackend backend, LaneVerifyStats* stats) noexcept {
    if (n == 0) {
        if (stats) *stats = LaneVerifyStats{};
        return true;
    }
    if (bad_args(entries, n, results, stats)) return false;

    LaneVerifyStats st;
    const LaneVerifyKernels* k = verify_kernels(backend);
    if (k == nullptr) {
        for (std::size_t i = 0; i < n; ++i) results[i] = ecdsa_scalar(entries[i]);
        st.scalar_only = n;
        if (stats) *stats = st;
        return all_true(results, n);
    }

    const unsigned width = k->lanes;
    for (std::size_t off = 0; off < n; off += width) {
        const std::size_t m = (n - off < width) ? (n - off) : std::size_t{width};

        // 1. The scalar verifier's own early rejects decide these entries.
        std::size_t cand[kLaneVerifyMaxLanes];
        unsigned nc = 0;
        for (std::size_t j = 0; j < m; ++j) {
            const std::size_t i = off + j;
            const ECDSABatchEntry& e = entries[i];
            if (e.public_key.is_infinity() || e.signature.r.is_zero() || e.signature.s.is_zero()) {
                results[i] = ecdsa_scalar(e);
                ++st.prep_fallback;
            } else {
                cand[nc++] = i;
            }
        }
        if (nc == 0) continue;

        // 2. Affine Q (one inversion) and w = s^-1 (one inversion) per group.
        Point pts[kLaneVerifyMaxLanes];
        FieldElement qx[kLaneVerifyMaxLanes];
        FieldElement qy[kLaneVerifyMaxLanes];
        Scalar prefix[kLaneVerifyMaxLanes];
        Scalar w[kLaneVerifyMaxLanes];
        for (unsigned c = 0; c < nc; ++c) pts[c] = entries[cand[c]].public_key;
        Point::batch_normalize(pts, nc, qx, qy);
        prefix[0] = entries[cand[0]].signature.s;
        for (unsigned c = 1; c < nc; ++c) prefix[c] = prefix[c - 1] * entries[cand[c]].signature.s;
        Scalar inv = prefix[nc - 1].inverse();
        for (unsigned c = nc - 1; c > 0; --c) {
            w[c] = inv * prefix[c - 1];
            inv = inv * entries[cand[c]].signature.s;
        }
        w[0] = inv;

        // 3. Lane inputs: u1 = z*w, u2 = r*w, Q affine, r and r + n.
        LaneVerifyGroup g{};
        std::size_t lane_entry[kLaneVerifyMaxLanes];
        unsigned used = 0;
        for (unsigned c = 0; c < nc; ++c) {
            const std::size_t i = cand[c];
            const ECDSABatchEntry& e = entries[i];
            if (!on_curve(qx[c], qy[c])) {   // the lane formulas assume a curve point
                results[i] = ecdsa_scalar(e);
                ++st.prep_fallback;
                continue;
            }
            const Scalar z = Scalar::from_bytes(e.msg_hash);
            const Scalar u1 = z * w[c];
            const Scalar u2 = e.signature.r * w[c];
            if (u1.is_zero() || u2.is_zero()) {   // recoding needs [1, n-1]
                results[i] = ecdsa_scalar(e);
                ++st.prep_fallback;
                continue;
            }
            put4(g.u1 + 4 * used, u1.limbs());
            put4(g.u2 + 4 * used, u2.limbs());
            put4(g.qx + 4 * used, qx[c].limbs());
            put4(g.qy + 4 * used, qy[c].limbs());
            put4(g.r + 4 * used, e.signature.r.limbs());
            if (r_plus_n(g.r + 4 * used, g.rn + 4 * used)) g.rn_ok |= 1u << used;
            lane_entry[used++] = i;
        }
        if (used == 0) continue;
        pad_group(g, used, width);

        // 4. Lane dual multiplication + projective x-compare.
        LaneVerifyResult res{};
        k->ecdsa_group(g, res);
        for (unsigned l = 0; l < used; ++l) {
            const std::size_t i = lane_entry[l];
            if ((res.flagged >> l) & 1u) {
                results[i] = ecdsa_scalar(entries[i]);
                ++st.exceptional_fallback;
            } else {
                results[i] = ((res.ok >> l) & 1u) != 0;
                ++st.lane_verified;
            }
        }
    }
    if (stats) *stats = st;
    return all_true(results, n);
}

bool schnorr_verify_lanes(const SchnorrBatchEntry* entries, std::size_t n, bool* results,
                          LaneBackend backend, LaneVerifyStats* stats) noexcept {
    if (n == 0) {
        if (stats) *stats = LaneVerifyStats{};
        return true;
    }
    if (bad_args(entries, n, results, stats)) return false;

    LaneVerifyStats st;
    const LaneVerifyKernels* k = verify_kernels(backend);
    if (k == nullptr) {
        for (std::size_t i = 0; i < n; ++i) results[i] = schnorr_scalar(entries[i]);
        st.scalar_only = n;
        if (stats) *stats = st;
        return all_true(results, n);
    }

    const unsigned width = k->lanes;
    for (std::size_t off = 0; off < n; off += width) {
        const std::size_t m = (n - off < width) ? (n - off) : std::size_t{width};
        LaneVerifyGroup g{};
        std::size_t lane_entry[kLaneVerifyMaxLanes];
        unsigned used = 0;
        for (std::size_t j = 0; j < m; ++j) {
            const std::size_t i = off + j;
            const SchnorrBatchEntry& e = entries[i];
            // schnorr_verify's order: s == 0, r >= p, lift_x(pubkey_x), e.
            std::uint64_t rl[4];
            be32_to_limbs(e.signature.r.data(), rl);
            FieldElement px;
            bool prep_ok = !e.signature.s.is_zero() && lt256(rl, kP);
            if (prep_ok) prep_ok = FieldElement::parse_bytes_strict(e.pubkey_x.data(), px);
            FieldElement py;
            if (prep_ok) {
                // lift_x: y = sqrt(x^3 + 7), rejected when x^3 + 7 is not a
                // square; the even root (BIP-340).
                const FieldElement y2 = px.square() * px + FieldElement::from_uint64(7);
                py = y2.sqrt();
                prep_ok = (py.square() == y2);
                if (prep_ok && (py.limbs()[0] & 1u) != 0) py = py.negate();
            }
            Scalar neg_e;
            if (prep_ok) {
                std::uint8_t buf[96];
                std::memcpy(buf, e.signature.r.data(), 32);
                std::memcpy(buf + 32, e.pubkey_x.data(), 32);
                std::memcpy(buf + 64, e.message.data(), 32);
                const Scalar ev = Scalar::from_bytes(tagged_hash("BIP0340/challenge", buf, sizeof(buf)));
                neg_e = ev.negate_var();   // e is public
                prep_ok = !neg_e.is_zero();   // recoding needs [1, n-1]
            }
            if (!prep_ok) {
                results[i] = schnorr_scalar(e);
                ++st.prep_fallback;
                continue;
            }
            put4(g.u1 + 4 * used, e.signature.s.limbs());
            put4(g.u2 + 4 * used, neg_e.limbs());
            put4(g.qx + 4 * used, px.limbs());
            put4(g.qy + 4 * used, py.limbs());
            std::memcpy(g.r + 4 * used, rl, 32);
            std::memcpy(g.rn + 4 * used, rl, 32);
            lane_entry[used++] = i;
        }
        if (used == 0) continue;
        pad_group(g, used, width);

        LaneVerifyResult res{};
        k->schnorr_group(g, res);
        for (unsigned l = 0; l < used; ++l) {
            const std::size_t i = lane_entry[l];
            if ((res.flagged >> l) & 1u) {
                results[i] = schnorr_scalar(entries[i]);
                ++st.exceptional_fallback;
            } else {
                results[i] = ((res.ok >> l) & 1u) != 0;
                ++st.lane_verified;
            }
        }
    }
    if (stats) *stats = st;
    return all_true(results, n);
}

// ---- Batch-verify opt-in (see secp256k1/lanes.hpp) ---------------------------
//
// g_verify_override holds a LaneBackend value, or kNoOverride. It is a plain
// enum with no data published alongside it, so acquire/release is sufficient.
namespace {
constexpr unsigned kNoOverride = 0xFFu;
std::atomic<unsigned> g_verify_override{kNoOverride};
std::atomic<std::uint64_t> g_batch_lane_dispatches{0};
} // namespace

bool set_lane_verify_backend(LaneBackend b) noexcept {
    const unsigned v = static_cast<unsigned>(b);
    if (v >= kLaneBackendCount) return false;
    if (b != LaneBackend::None && lane_field_kernels(b) == nullptr) return false;   // never SIGILL
    g_verify_override.store(v, std::memory_order_release);
    return true;
}

void reset_lane_verify_backend() noexcept {
    g_verify_override.store(kNoOverride, std::memory_order_release);
}

LaneBackend lane_verify_backend() noexcept {
    const unsigned v = g_verify_override.load(std::memory_order_acquire);
    if (v != kNoOverride) return static_cast<LaneBackend>(v);
    // Opt-in only: the cached UFSECP_LANES resolution counts when it was an
    // explicit, available request. Its Default pick is deliberately ignored
    // until a measurement makes a lane backend the default for batch verify.
    const LaneSelection& sel = lane_selection();
    return (sel.reason == LaneSelectionReason::Forced) ? sel.backend : LaneBackend::None;
}

namespace detail {

std::uint64_t batch_lane_dispatch_count() noexcept {
    return g_batch_lane_dispatches.load(std::memory_order_relaxed);
}

void note_batch_lane_dispatch() noexcept {
    g_batch_lane_dispatches.fetch_add(1, std::memory_order_relaxed);
}

} // namespace detail

namespace detail {

bool ecdsa_xcheck_lanes_for_test(LaneBackend b, const std::uint64_t* X, const std::uint64_t* Z,
                                 const std::uint64_t* r, std::uint8_t* out, std::size_t n) noexcept {
    const LaneVerifyKernels* k = verify_kernels(b);
    if (k == nullptr) return false;
    const unsigned width = k->lanes;
    for (std::size_t off = 0; off < n; off += width) {
        const std::size_t m = (n - off < width) ? (n - off) : std::size_t{width};
        alignas(64) std::uint64_t bx[4 * kLaneVerifyMaxLanes];
        alignas(64) std::uint64_t bz[4 * kLaneVerifyMaxLanes];
        alignas(64) std::uint64_t br[4 * kLaneVerifyMaxLanes];
        alignas(64) std::uint64_t brn[4 * kLaneVerifyMaxLanes];
        unsigned rn_ok = 0;
        for (unsigned l = 0; l < width; ++l) {
            const std::size_t src = off + ((l < m) ? l : 0u);
            std::memcpy(bx + 4 * l, X + 4 * src, 32);
            std::memcpy(bz + 4 * l, Z + 4 * src, 32);
            std::memcpy(br + 4 * l, r + 4 * src, 32);
            if (r_plus_n(br + 4 * l, brn + 4 * l)) rn_ok |= 1u << l;
        }
        const unsigned bits = k->ecdsa_xcheck(bx, bz, br, brn, rn_ok);
        for (std::size_t l = 0; l < m; ++l) out[off + l] = static_cast<std::uint8_t>((bits >> l) & 1u);
    }
    return true;
}

} // namespace detail

} // namespace secp256k1::lanes
