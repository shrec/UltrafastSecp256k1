// RED-TEAM-009 fix: ka_put returns nullptr at DoS cap → pubkey_agg returns 0.
// Regression test: audit/test_exploit_shim_musig_ka_cap.cpp
#include "secp256k1_musig.h"
#include "secp256k1_extrakeys.h"
#include "shim_internal.hpp"
#include <atomic>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "secp256k1/musig2.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"
#include "secp256k1/ct/point.hpp"
#include "secp256k1/detail/secure_erase.hpp"

using secp256k1::fast::Point;
using secp256k1::fast::Scalar;
using secp256k1::fast::FieldElement;

namespace {

// ---------------------------------------------------------------------------
// Decompress a 33-byte compressed point (matches musig2.cpp internal helper)
// ---------------------------------------------------------------------------
static Point decompress(const unsigned char compressed[33]) {
    if (compressed[0] != 0x02 && compressed[0] != 0x03) return Point::infinity();
    std::array<uint8_t, 32> xb;
    std::memcpy(xb.data(), compressed + 1, 32);
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(xb.data(), x)) return Point::infinity();
    auto y2 = x * x * x + FieldElement::from_uint64(7);
    auto y = y2.sqrt();
    if (!(y.square() == y2)) return Point::infinity();
    bool y_odd = (y.limbs()[0] & 1) != 0;
    if (y_odd != (compressed[0] == 0x03)) y = y.negate();
    return Point::from_affine(x, y);
}

// Decompress and return (x, y) as byte arrays. Returns false if invalid.
// Used by pubnonce_parse to cache affine coords, avoiding re-decompression later.
static bool decompress_to_xy(const unsigned char compressed[33],
                              uint8_t out_x32[32], uint8_t out_y32[32]) {
    if (compressed[0] != 0x02 && compressed[0] != 0x03) return false;
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(compressed + 1, x)) return false;
    auto y2 = x * x * x + FieldElement::from_uint64(7);
    auto y  = y2.sqrt();
    if (!(y.square() == y2)) return false;
    bool y_odd = (y.limbs()[0] & 1) != 0;
    if (y_odd != (compressed[0] == 0x03)) y = y.negate();
    auto xb = x.to_bytes(); std::memcpy(out_x32, xb.data(), 32);
    auto yb = y.to_bytes(); std::memcpy(out_y32, yb.data(), 32);
    return true;
}

static void compress(const Point& pt, unsigned char out[33]) {
    auto arr = pt.to_compressed();
    std::memcpy(out, arr.data(), 33);
}

// ---------------------------------------------------------------------------
// keyagg_cache side-channel registry
//
// WHY A GLOBAL MAP:
//   MuSig2KeyAggCtx contains key_coefficients (std::vector<Scalar>) — one
//   scalar per signer.  The libsecp256k1-compatible opaque struct
//   secp256k1_musig_keyagg_cache is 197 bytes, which cannot hold a
//   variable-length vector.  We therefore store the context in a process-
//   global map keyed on the caller's struct address.
//
// THREAD SAFETY:
//   g_mu guards all accesses to g_ka.  Concurrent signing sessions with
//   distinct secp256k1_musig_keyagg_cache objects are fully independent.
//
// LIFETIME / KNOWN LIMITATION:
//   Entries are inserted in secp256k1_musig_pubkey_agg and automatically
//   removed in secp256k1_musig_partial_sig_agg (the final protocol step).
//   If the caller abandons a session before aggregation (e.g. on error),
//   the entry persists until the keyagg_cache address is reused by another
//   pubkey_agg call, which overwrites the slot, or until process exit.
//   For most signing sessions (which reach partial_sig_agg), there is no
//   persistent memory growth.
//
// ADDRESS-REUSE HAZARD:
//   If a secp256k1_musig_keyagg_cache is freed and a new struct is
//   stack/heap-allocated at the same address before pubkey_agg is called
//   again, the old entry will be silently overwritten on the next pubkey_agg.
//   Callers must treat each keyagg_cache as a single-session opaque handle
//   and must not move, copy, or reuse the struct across sessions.
//
// NOTE FOR REVIEWERS:
//   This shim-private registry stores MuSig2KeyAggCtx state that cannot fit
//   inline into the opaque secp256k1_musig_keyagg_cache struct (197 bytes,
//   but key_coefficients is a variable-length vector — count depends on
//   signers). No per-process global state exists in UF's CPU or CT layers;
//   this map is an unavoidable consequence of the fixed-size opaque ABI.
//
//   Bitcoin Core usage: Bitcoin Core does not use MuSig2 in its current
//   production signing paths. This registry is therefore not on the critical
//   path for the bitcoin-core-backend profile.
//   See docs/BITCOIN_CORE_BACKEND_EVIDENCE.md §MuSig2.
// ---------------------------------------------------------------------------
struct KAEntry {
    secp256k1::MuSig2KeyAggCtx ctx;
    std::array<unsigned char, 33> agg_pk_comp;             // compressed aggregated pubkey
    std::vector<std::array<unsigned char, 33>> compressed;  // 33-byte compressed keys in aggregation order

    // Return {signer_index, true} if keypair_data matches an aggregated key; {0, false} otherwise.
    // keypair_create (BIP-340) always normalizes to even-Y, so the compressed pubkey is
    // 0x02 || data[32..63] — avoids the ~10-15µs ct::generator_mul_blinded call per sign.
    std::pair<std::size_t, bool> find_index(const unsigned char keypair_data[96]) const {
        std::array<unsigned char, 33> pk33;
        pk33[0] = 0x02;  // even-Y guaranteed by secp256k1_keypair_create BIP-340 normalization
        std::memcpy(pk33.data() + 1, keypair_data + 32, 32);  // X at data[32..63]
        for (std::size_t i = 0; i < compressed.size(); ++i) {
            if (compressed[i] == pk33) return {i, true};
        }
        return {0, false}; // key not in aggregation — caller must treat as error
    }
};

// ---------------------------------------------------------------------------
// SHIM-010 fix: token-keyed session map
//
// Previously keyed on struct address (const void*), which caused a dangling-
// context hazard: if a secp256k1_musig_keyagg_cache is stack-allocated and
// then freed, a new struct at the same address would retrieve the old session.
//
// Fix: write a monotonically-incrementing 64-bit token into data[0..7] of the
// opaque struct at pubkey_agg time. The map is now keyed on the token value,
// not the pointer. A new struct at the same address gets a new token → no
// collision, no dangling context.
//
// secp256k1_musig_keyagg_cache is 197 bytes of caller-opaque storage; using
// bytes 0..7 as our token is safe as long as we write a fresh token on every
// pubkey_agg call (which we do).
// ---------------------------------------------------------------------------
static std::atomic<std::uint64_t> g_token_counter{1};

static std::uint64_t read_token(const secp256k1_musig_keyagg_cache* p) {
    std::uint64_t tok = 0;
    std::memcpy(&tok, p->data, sizeof(tok));
    return tok;
}

static void write_token(secp256k1_musig_keyagg_cache* p, std::uint64_t tok) {
    std::memcpy(p->data, &tok, sizeof(tok));
}

std::mutex g_mu;
// PRECOMPUTE-GCONTEXT-UAF class (shim variant): the map holds shared_ptr, and the
// accessors return a shared_ptr SNAPSHOT (not a raw `it->second.get()`). A caller that
// holds the snapshot keeps the KAEntry alive for the whole operation even if a concurrent
// ka_remove / partial_sig_agg erases the map entry — eliminating the unlock-then-use-raw
// use-after-free of the secret-adjacent KAEntry::ctx. Mirrors the g_context shared_ptr fix.
std::unordered_map<std::uint64_t, std::shared_ptr<KAEntry>> g_ka;

// Hard cap prevents DoS from abandoned sessions on error paths.
static constexpr std::size_t kMaxKaEntries = 1024;

static std::shared_ptr<KAEntry> ka_get(const secp256k1_musig_keyagg_cache* p) {
    std::uint64_t tok = read_token(p);
    if (tok == 0) return nullptr;  // never initialized
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_ka.find(tok);
    return it != g_ka.end() ? it->second : nullptr;  // shared_ptr snapshot, not a raw .get()
}

// Look up the keyagg entry directly by token (used by partial_sig_agg to recover the
// BIP-327 tweak accumulators — the session blob has no room to carry tweak_s).
static std::shared_ptr<KAEntry> ka_get_by_token(std::uint64_t tok) {
    if (tok == 0) return nullptr;
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_ka.find(tok);
    return it != g_ka.end() ? it->second : nullptr;  // shared_ptr snapshot, not a raw .get()
}

static std::shared_ptr<KAEntry> ka_put(secp256k1_musig_keyagg_cache* p, std::unique_ptr<KAEntry> v) {
    std::lock_guard<std::mutex> lk(g_mu);
    if (g_ka.size() >= kMaxKaEntries) return nullptr;  // DoS cap
    // Assign a fresh monotonic token — eliminates address-reuse hazard.
    std::uint64_t tok = g_token_counter.fetch_add(1, std::memory_order_relaxed);
    write_token(p, tok);
    auto& slot = g_ka[tok];
    slot = std::move(v);  // unique_ptr -> shared_ptr
    return slot;          // shared_ptr snapshot
}

// Called after the final protocol step — releases the session entry.
static void ka_remove(const secp256k1_musig_keyagg_cache* p) {
    std::uint64_t tok = read_token(p);
    if (tok == 0) return;
    std::lock_guard<std::mutex> lk(g_mu);
    g_ka.erase(tok);
}

// ---------------------------------------------------------------------------
// SecNonce: k1[32] | k2[32] packed into data[132]
// ---------------------------------------------------------------------------
static void sn_pack(secp256k1_musig_secnonce* out, const Scalar& k1, const Scalar& k2) {
    auto b1 = k1.to_bytes(); std::memcpy(out->data,      b1.data(), 32);
    auto b2 = k2.to_bytes(); std::memcpy(out->data + 32, b2.data(), 32);
}

static bool sn_unpack(const secp256k1_musig_secnonce* in, Scalar& k1, Scalar& k2) {
    // SECURITY: must use nonzero variant. A zeroed secnonce (after single-use
    // consumption) would pass parse_bytes_strict with k1=k2=0, causing
    // musig2_partial_sign to return e*a_i*d — leaking the effective signing key.
    return Scalar::parse_bytes_strict_nonzero(in->data,      k1) &&
           Scalar::parse_bytes_strict_nonzero(in->data + 32, k2);
}

// PubNonce internal layout (SHIM-007: caches affine coords to avoid re-decompression)
// data[132]:
//   [0..31]   R1.x bytes (32)
//   [32..63]  R1.y bytes (32)
//   [64..95]  R2.x bytes (32)
//   [96..127] R2.y bytes (32)
//   [128]     flags: bit7=valid, bit0=R1_y_odd, bit1=R2_y_odd
//   [129..131] reserved
// Wire format (66 bytes): R1_compressed[33] | R2_compressed[33]  (unchanged externally)

static void pn_pack_affine(secp256k1_musig_pubnonce* out,
                            const uint8_t rx1[32], const uint8_t ry1[32],
                            const uint8_t rx2[32], const uint8_t ry2[32]) {
    std::memset(out->data, 0, sizeof(out->data));
    std::memcpy(out->data,       rx1, 32);
    std::memcpy(out->data + 32,  ry1, 32);
    std::memcpy(out->data + 64,  rx2, 32);
    std::memcpy(out->data + 96,  ry2, 32);
    uint8_t flags = 0x80;  // valid
    if (ry1[31] & 1) flags |= 0x01;  // R1 y odd
    if (ry2[31] & 1) flags |= 0x02;  // R2 y odd
    out->data[128] = flags;
}

// Build a pair of Points from cached affine coords — O(1), no sqrt.
// Returns {infinity, infinity} if any field element fails strict parsing
// (indicates corrupted secp256k1_musig_pubnonce struct).
static std::pair<Point, Point> pn_unpack_points(const secp256k1_musig_pubnonce* in) {
    FieldElement x1, y1, x2, y2;
    if (!FieldElement::parse_bytes_strict(in->data,      x1) ||
        !FieldElement::parse_bytes_strict(in->data + 32, y1) ||
        !FieldElement::parse_bytes_strict(in->data + 64, x2) ||
        !FieldElement::parse_bytes_strict(in->data + 96, y2))
        return { Point::infinity(), Point::infinity() };
    return { Point::from_affine(x1, y1), Point::from_affine(x2, y2) };
}

// Legacy: reconstruct compressed MuSig2PubNonce from cached affine coords.
// Only used by paths that still need the compressed representation.
static secp256k1::MuSig2PubNonce pn_unpack(const secp256k1_musig_pubnonce* in) {
    secp256k1::MuSig2PubNonce pn;
    uint8_t flags = in->data[128];
    // R1: parity from flags bit0, x from data[0..31]
    pn.R1[0] = (flags & 0x01) ? 0x03 : 0x02;
    std::memcpy(pn.R1.data() + 1, in->data,      32);
    // R2: parity from flags bit1, x from data[64..95]
    pn.R2[0] = (flags & 0x02) ? 0x03 : 0x02;
    std::memcpy(pn.R2.data() + 1, in->data + 64, 32);
    return pn;
}

// Legacy pn_pack (for nonce_gen path which still uses MuSig2PubNonce internally).
static void pn_pack(secp256k1_musig_pubnonce* out, const secp256k1::MuSig2PubNonce& pn) {
    // Decompress both points to fill the affine layout.
    uint8_t x1[32], y1[32], x2[32], y2[32];
    if (!decompress_to_xy(pn.R1.data(), x1, y1) ||
        !decompress_to_xy(pn.R2.data(), x2, y2)) {
        std::memset(out->data, 0, sizeof(out->data));
        return;
    }
    pn_pack_affine(out, x1, y1, x2, y2);
}

// Session layout in data[133]:
//   [0..32]   R         (33 bytes, compressed point)
//   [33..64]  b         (32 bytes, scalar)
//   [65..96]  e         (32 bytes, scalar)
//   [97]      R_negated (1 byte)
//   [98..105] keyagg_cache ptr (8 bytes, same-process use only)
//   [106..132] reserved (26 bytes)
static void sess_pack(secp256k1_musig_session* out, const secp256k1::MuSig2Session& s) {
    compress(s.R, out->data);
    auto b = s.b.to_bytes(); std::memcpy(out->data + 33, b.data(), 32);
    auto e = s.e.to_bytes(); std::memcpy(out->data + 65, e.data(), 32);
    out->data[97] = s.R_negated ? 1 : 0;
}

static secp256k1::MuSig2Session sess_unpack(const secp256k1_musig_session* in) {
    secp256k1::MuSig2Session s;
    s.R = decompress(in->data);
    Scalar::parse_bytes_strict(in->data + 33, s.b);
    Scalar::parse_bytes_strict(in->data + 65, s.e);
    s.R_negated = (in->data[97] != 0);
    return s;
}

// AUDIT-004 fix: store the uint64_t token (from g_ka key) instead of the raw
// heap pointer. session->data[98..105] now contains an opaque counter value,
// not a real address — eliminates the ASLR information leak.
static void sess_stash_cache_token(secp256k1_musig_session* s,
                                    const secp256k1_musig_keyagg_cache* p) {
    std::uint64_t tok = read_token(p);
    std::memcpy(s->data + 98, &tok, sizeof(tok));
}

static std::uint64_t
sess_load_cache_token(const secp256k1_musig_session* s) {
    std::uint64_t tok = 0;
    std::memcpy(&tok, s->data + 98, sizeof(tok));
    return tok;
}

static void ka_remove_by_token(std::uint64_t tok) {
    if (tok == 0) return;
    std::lock_guard<std::mutex> lk(g_mu);
    g_ka.erase(tok);
}

} // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Key aggregation
// ---------------------------------------------------------------------------

int secp256k1_musig_pubkey_agg(
    const secp256k1_context* ctx,
    secp256k1_xonly_pubkey* agg_pk,
    secp256k1_musig_keyagg_cache* keyagg_cache,
    const secp256k1_pubkey* const* pubkeys,
    size_t n_pubkeys)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!keyagg_cache || !pubkeys || n_pubkeys == 0) return 0;

    std::vector<std::array<unsigned char, 33>> comp33(n_pubkeys);
    for (size_t i = 0; i < n_pubkeys; ++i) {
        if (!pubkeys[i]) return 0;
        // Direct extraction: opaque struct stores X at data[0..31], Y parity at data[63]&1.
        // Avoids the full secp256k1_ec_pubkey_serialize call (flag validation, branch overhead).
        comp33[i][0] = (pubkeys[i]->data[63] & 1) ? 0x03 : 0x02;
        std::memcpy(comp33[i].data() + 1, pubkeys[i]->data, 32);
    }

    auto e = std::make_unique<KAEntry>();
    e->ctx = secp256k1::musig2_key_agg(comp33);
    if (e->ctx.Q.is_infinity()) return 0;

    // musig2_key_agg normalizes Q to even-Y for signing, but secp256k1_musig_pubkey_get
    // must return the plain (possibly odd-Y) aggregate key. Restore original Y if negated.
    Point orig_Q = e->ctx.Q_negated ? e->ctx.Q.negate() : e->ctx.Q;

    // PERF-B2: compute to_uncompressed() ONCE to get both X and Y in a single
    // field inversion. Previously: compress() (1 inversion) + lift_x sqrt (~3.8 µs).
    // Now: to_uncompressed() (1 inversion) → fill agg_pk_comp AND agg_pk->data
    // directly, eliminating the sqrt entirely.
    auto unc = orig_Q.to_uncompressed();  // [04][X:32][Y:32]
    bool y_is_odd = (unc[64] & 1) != 0;
    e->agg_pk_comp[0] = y_is_odd ? 0x03 : 0x02;
    std::memcpy(e->agg_pk_comp.data() + 1, unc.data() + 1, 32);
    e->compressed = comp33;

    if (agg_pk) {
        // X coordinate directly from uncompressed form.
        std::memcpy(agg_pk->data, unc.data() + 1, 32);
        // P1-PERF-001: store even-Y in data[32..63] so secp256k1_schnorrsig_verify
        // can reconstruct the point directly without lift_x sqrt.
        // Y is already available from to_uncompressed() — no sqrt needed.
        if (y_is_odd) {
            // Negate Y: for byte representation, Y_even = p - Y_odd.
            // Use FieldElement to compute the negation cleanly.
            std::array<uint8_t, 32> yb{};
            std::memcpy(yb.data(), unc.data() + 33, 32);
            FieldElement y_fe = FieldElement::from_bytes(yb);
            auto y_even = y_fe.negate();
            y_even.to_bytes_into(reinterpret_cast<uint8_t*>(agg_pk->data) + 32);
        } else {
            std::memcpy(agg_pk->data + 32, unc.data() + 33, 32);
        }
    }

    // RED-TEAM-009: check ka_put return value — returns nullptr when DoS cap (1024)
    // is hit. In that case the token was never written, so subsequent MuSig
    // operations would fail silently. Fail-closed here instead of returning 1.
    if (!ka_put(keyagg_cache, std::move(e))) return 0;
    return 1;
}

int secp256k1_musig_pubkey_get(
    const secp256k1_context* ctx,
    secp256k1_pubkey* agg_pk,
    const secp256k1_musig_keyagg_cache* keyagg_cache)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!agg_pk || !keyagg_cache) return 0;
    auto e = ka_get(keyagg_cache);  // shared_ptr snapshot — keeps KAEntry alive for this op
    if (!e) return 0;
    secp256k1_ec_pubkey_parse(secp256k1_context_static, agg_pk, e->agg_pk_comp.data(), 33);
    return 1;
}

int secp256k1_musig_pubkey_ec_tweak_add(
    const secp256k1_context* ctx,
    secp256k1_pubkey* output_pubkey,
    secp256k1_musig_keyagg_cache* keyagg_cache,
    const unsigned char* tweak32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!keyagg_cache || !tweak32) return 0;
    auto e = ka_get(keyagg_cache);  // shared_ptr snapshot — keeps KAEntry alive for this op
    if (!e) return 0;
    // SHIM-003: parse_bytes_strict (not nonzero) — tweak=0 valid per libsecp
    // (result is Q unchanged). parse_bytes_strict_nonzero incorrectly rejected
    // zero tweak, diverging from upstream behavior.
    // SHIM-003: parse_bytes_strict (not nonzero) — tweak=0 valid per libsecp.
    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) return 0;
    Point tG = secp256k1::fast::scalar_mul_generator(t);
    // BIP-327 ordinary tweak (g=1): operate on the ACTUAL aggregate. e->ctx.Q is stored
    // even-Y normalized, so recover the actual point via Q_negated before adding t*G.
    Point A  = e->ctx.Q_negated ? e->ctx.Q.negate() : e->ctx.Q;
    Point A2 = A.add(tG);
    if (A2.is_infinity()) return 0;
    e->ctx.tacc = e->ctx.tacc + t;                       // tacc += t  (gacc unchanged for EC)
    // re-store the cache in its even-Y + Q_negated convention (signer relies on it)
    bool a_odd = !A2.has_even_y();
    e->ctx.Q = a_odd ? A2.negate() : A2;
    e->ctx.Q_negated = a_odd;
    {
        // output + agg_pk_comp = the ACTUAL tweaked aggregate (real parity); Q_x = x (parity-free)
        auto unc_q = A2.to_uncompressed();
        std::memcpy(e->ctx.Q_x.data(), unc_q.data() + 1, 32);
        bool q_odd = (unc_q[64] & 1) != 0;
        e->agg_pk_comp[0] = q_odd ? 0x03 : 0x02;
        std::memcpy(e->agg_pk_comp.data() + 1, unc_q.data() + 1, 32);
        if (output_pubkey) {
            std::memcpy(output_pubkey->data,      unc_q.data() + 1,  32);
            std::memcpy(output_pubkey->data + 32, unc_q.data() + 33, 32);
        }
    }
    return 1;
}

int secp256k1_musig_pubkey_xonly_tweak_add(
    const secp256k1_context* ctx,
    secp256k1_pubkey* output_pubkey,
    secp256k1_musig_keyagg_cache* keyagg_cache,
    const unsigned char* tweak32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!keyagg_cache || !tweak32) return 0;
    auto e = ka_get(keyagg_cache);  // shared_ptr snapshot — keeps KAEntry alive for this op
    if (!e) return 0;
    // SHIM-001: parse_bytes_strict (not nonzero) — tweak=0 valid per libsecp256k1.
    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) return 0;
    Point tG = secp256k1::fast::scalar_mul_generator(t);
    // BIP-327 x-only tweak: g = -1 if the ACTUAL aggregate has odd Y, else 1. e->ctx.Q is
    // even-Y normalized so the actual aggregate is odd iff Q_negated, and even-Y(actual)
    // IS e->ctx.Q. So A2 = g*A + t*G = e->ctx.Q + t*G.
    bool const g_neg = e->ctx.Q_negated;
    Point A2 = e->ctx.Q.add(tG);
    if (A2.is_infinity()) return 0;
    if (g_neg) e->ctx.gacc = e->ctx.gacc.negate();                  // gacc = g*gacc
    e->ctx.tacc = (g_neg ? e->ctx.tacc.negate() : e->ctx.tacc) + t; // tacc = t + g*tacc
    bool a_odd = !A2.has_even_y();
    e->ctx.Q = a_odd ? A2.negate() : A2;
    e->ctx.Q_negated = a_odd;
    {
        auto unc_q = A2.to_uncompressed();
        std::memcpy(e->ctx.Q_x.data(), unc_q.data() + 1, 32);
        bool q_odd = (unc_q[64] & 1) != 0;
        e->agg_pk_comp[0] = q_odd ? 0x03 : 0x02;
        std::memcpy(e->agg_pk_comp.data() + 1, unc_q.data() + 1, 32);
        if (output_pubkey) {
            std::memcpy(output_pubkey->data,      unc_q.data() + 1,  32);
            std::memcpy(output_pubkey->data + 32, unc_q.data() + 33, 32);
        }
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Nonce generation
// ---------------------------------------------------------------------------

int secp256k1_musig_nonce_gen(
    const secp256k1_context* ctx,
    secp256k1_musig_secnonce* secnonce,
    secp256k1_musig_pubnonce* pubnonce,
    const unsigned char* session_id32,
    const unsigned char* seckey,
    const secp256k1_pubkey* pubkey,
    const unsigned char* msg32,
    const secp256k1_musig_keyagg_cache* /*keyagg_cache*/,
    const unsigned char* extra_input32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!secnonce || !pubnonce || !session_id32) return 0;

    // P2-SEC-002: when seckey is NULL, do not use session_id32 as a private key
    // substitute. session_id32 is a caller-controlled identifier (may be a counter,
    // timestamp, or short string) — it is not suitable HMAC entropy. Per BIP-327 §5,
    // session_id32 MUST be a fresh 32-byte random value; when seckey is absent the
    // nonce is derived from session_id32+pubkey+msg alone (zero sk), which is safe
    // only when session_id32 is uniformly random. Callers SHOULD always supply seckey.
    Scalar sk;
    if (seckey) {
        if (!Scalar::parse_bytes_strict_nonzero(seckey, sk)) return 0;
    } else {
        // seckey absent: use Scalar::zero() — nonce is derived from session_id32+pubkey+msg.
        // Safe when session_id32 is truly random (BIP-327 requirement).
        sk = Scalar::zero();
    }

    std::array<unsigned char, 32> pub_x = {};
    if (pubkey) {
        std::memcpy(pub_x.data(), pubkey->data, 32);
    }

    std::array<unsigned char, 32> msg = {};
    if (msg32) std::memcpy(msg.data(), msg32, 32);

    std::array<unsigned char, 32> agg_x = {};

    auto [sn, pn] = secp256k1::musig2_nonce_gen(sk, pub_x, agg_x, msg, session_id32, extra_input32);
    // P2-SEC-003: erase sk from stack before returning — sk was derived from the
    // caller's private key and must not persist after musig2_nonce_gen consumed it.
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    sn_pack(secnonce, sn.k1, sn.k2);
    pn_pack(pubnonce, pn);
    return 1;
}

int secp256k1_musig_pubnonce_serialize(
    const secp256k1_context* ctx,
    unsigned char* out66,
    const secp256k1_musig_pubnonce* nonce)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!out66 || !nonce) return 0;
    // Reconstruct compressed form from cached affine coords (no sqrt needed).
    uint8_t flags = nonce->data[128];
    out66[0] = (flags & 0x01) ? 0x03 : 0x02;
    std::memcpy(out66 + 1,  nonce->data,      32);  // R1.x
    out66[33] = (flags & 0x02) ? 0x03 : 0x02;
    std::memcpy(out66 + 34, nonce->data + 64, 32);  // R2.x
    return 1;
}

int secp256k1_musig_pubnonce_parse(
    const secp256k1_context* ctx,
    secp256k1_musig_pubnonce* nonce,
    const unsigned char* in66)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!nonce || !in66) return 0;
    // SHIM-007: decompress once, cache affine coords. Later users call
    // pn_unpack_points() to get Points in O(1) without re-decompressing.
    // SEC-003: decompress_to_xy already validates curve membership (y²=x³+7
    // check) and rejects non-02/03 prefixes, so infinity cannot be stored.
    // We add an explicit is_infinity() guard for defense-in-depth and to
    // match upstream libsecp256k1 behaviour which rejects infinity pubnonces.
    uint8_t x1[32], y1[32], x2[32], y2[32];
    if (!decompress_to_xy(in66,      x1, y1)) return 0;
    if (!decompress_to_xy(in66 + 33, x2, y2)) return 0;
    // Reconstruct points temporarily to verify neither is infinity.
    {
        using secp256k1::fast::FieldElement;
        using secp256k1::fast::Point;
        FieldElement fx1, fy1, fx2, fy2;
        if (!FieldElement::parse_bytes_strict(x1, fx1)) return 0;
        if (!FieldElement::parse_bytes_strict(y1, fy1)) return 0;
        if (!FieldElement::parse_bytes_strict(x2, fx2)) return 0;
        if (!FieldElement::parse_bytes_strict(y2, fy2)) return 0;
        Point const R1 = Point::from_affine(fx1, fy1);
        Point const R2 = Point::from_affine(fx2, fy2);
        if (R1.is_infinity() || R2.is_infinity()) return 0;
    }
    pn_pack_affine(nonce, x1, y1, x2, y2);
    return 1;
}

int secp256k1_musig_nonce_agg(
    const secp256k1_context* ctx,
    secp256k1_musig_aggnonce* aggnonce,
    const secp256k1_musig_pubnonce* const* pubnonces,
    size_t n_pubnonces)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!aggnonce || !pubnonces || n_pubnonces == 0) return 0;
    // SHIM-007: use pre-cached affine coords — no decompress (no sqrt) here.
    // P2-PERF-004: SBO — for N<=16 (common 2-of-2 case) accumulate directly on
    // the stack, avoiding a heap allocation for the intermediate pts vector.
    constexpr size_t kSBOLimit = 16;
    secp256k1::MuSig2AggNonce agg{};
    agg.R1 = Point::infinity();
    agg.R2 = Point::infinity();
    if (n_pubnonces <= kSBOLimit) {
        for (size_t i = 0; i < n_pubnonces; ++i) {
            if (!pubnonces[i]) return 0;
            auto [r1, r2] = pn_unpack_points(pubnonces[i]);
            agg.R1 = agg.R1.add(r1);
            agg.R2 = agg.R2.add(r2);
        }
    } else {
        std::vector<std::pair<Point, Point>> pts;
        pts.reserve(n_pubnonces);
        for (size_t i = 0; i < n_pubnonces; ++i) {
            if (!pubnonces[i]) return 0;
            pts.push_back(pn_unpack_points(pubnonces[i]));
        }
        agg = secp256k1::musig2_nonce_agg_points(pts);
    }
    std::memset(aggnonce->data, 0, sizeof(aggnonce->data));  // zero all 132 bytes (B-01)
    compress(agg.R1, aggnonce->data);
    compress(agg.R2, aggnonce->data + 33);
    return 1;
}

// ---------------------------------------------------------------------------
// Session signing
// ---------------------------------------------------------------------------

int secp256k1_musig_nonce_process(
    const secp256k1_context* ctx,
    secp256k1_musig_session* session,
    const secp256k1_musig_aggnonce* aggnonce,
    const unsigned char* msg32,
    const secp256k1_musig_keyagg_cache* keyagg_cache)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!session || !aggnonce || !msg32 || !keyagg_cache) return 0;
    auto e = ka_get(keyagg_cache);  // shared_ptr snapshot — keeps KAEntry alive for this op
    if (!e) return 0;
    secp256k1::MuSig2AggNonce an;
    // BIP-327 cpoint_ext: a 33-zero aggnonce half is the point at infinity and is VALID
    // (participant nonces cancelled). Reject only a half that is non-zero yet fails to
    // decompress (an invalid contribution). The combined-nonce-infinity case is handled
    // by musig2_start_sign_session (R = G), matching libsecp256k1.
    static const unsigned char kZero33[33] = {0};
    const bool r1_is_zero = std::memcmp(aggnonce->data, kZero33, 33) == 0;
    an.R1 = decompress(aggnonce->data);
    if (an.R1.is_infinity() && !r1_is_zero) return 0;
    const bool r2_is_zero = std::memcmp(aggnonce->data + 33, kZero33, 33) == 0;
    an.R2 = decompress(aggnonce->data + 33);
    if (an.R2.is_infinity() && !r2_is_zero) return 0;
    std::array<unsigned char, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    auto s = secp256k1::musig2_start_sign_session(an, e->ctx, msg);
    if (s.e.is_zero()) return 0;  // defensive: degenerate session (challenge e == 0)
    sess_pack(session, s);
    sess_stash_cache_token(session, keyagg_cache);
    return 1;
}

int secp256k1_musig_partial_sign(
    const secp256k1_context* ctx,
    secp256k1_musig_partial_sig* partial_sig,
    secp256k1_musig_secnonce* secnonce,
    const secp256k1_keypair* keypair,
    const secp256k1_musig_keyagg_cache* keyagg_cache,
    const secp256k1_musig_session* session)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!partial_sig || !secnonce || !keypair || !keyagg_cache || !session) return 0;
    // T-01: Apply DPA blinding for the duration of this signing call (matches ECDSA/Schnorr shim).
    secp256k1_shim_internal::ContextBlindingScope _blind(ctx);
    auto e = ka_get(keyagg_cache);  // shared_ptr snapshot — keeps KAEntry alive for this op
    if (!e) return 0;

    Scalar k1, k2;
    if (!sn_unpack(secnonce, k1, k2)) {
        secp256k1::detail::secure_erase(&k1, sizeof(k1));   // secret-residue sweep
        secp256k1::detail::secure_erase(&k2, sizeof(k2));
        return 0;
    }
    std::memset(secnonce->data, 0, sizeof(secnonce->data)); // zeroize (single-use)

    Scalar sk;
    // MuSig2 signing-share residue: erase the nonce scalars (k1,k2) and the parsed
    // signing key (sk) on every return path after this point. Erasure runs AFTER
    // each value's last use, so the signature output is unchanged.
    auto erase_secrets = [&]() {
        secp256k1::detail::secure_erase(&k1, sizeof(k1));
        secp256k1::detail::secure_erase(&k2, sizeof(k2));
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
    };
    if (!Scalar::parse_bytes_strict_nonzero(keypair->data, sk)) { erase_secrets(); return 0; }

    secp256k1::MuSig2SecNonce sn{ k1, k2 };
    auto s = sess_unpack(session);
    auto [idx, idx_found] = e->find_index(keypair->data);
    if (!idx_found) {
        // Unknown key: clear output and fail-closed to prevent signing as signer #0.
        std::memset(partial_sig->data, 0, sizeof(partial_sig->data));
        secp256k1::detail::secure_erase(&sn, sizeof(sn));
        erase_secrets();
        return 0;
    }
    auto psig = secp256k1::musig2_partial_sign(sn, sk, e->ctx, s, idx);
    secp256k1::detail::secure_erase(&sn, sizeof(sn));   // nonce-pair copy consumed
    // Fail-closed: zero partial sig means degenerate nonce (k=0).
    // Returning success with zeros would silently break the aggregation.
    if (psig.is_zero()) { erase_secrets(); return 0; }
    auto b = psig.to_bytes();
    std::memcpy(partial_sig->data, b.data(), 32);
    std::memset(partial_sig->data + 32, 0, 4);
    erase_secrets();
    return 1;
}

int secp256k1_musig_partial_sig_verify(
    const secp256k1_context* ctx,
    const secp256k1_musig_partial_sig* partial_sig,
    const secp256k1_musig_pubnonce* pubnonce,
    const secp256k1_pubkey* pubkey,
    const secp256k1_musig_keyagg_cache* keyagg_cache,
    const secp256k1_musig_session* session)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!partial_sig || !pubnonce || !pubkey || !keyagg_cache || !session) return 0;
    auto e = ka_get(keyagg_cache);  // shared_ptr snapshot — keeps KAEntry alive for this op
    if (!e) return 0;

    Scalar psig;
    if (!Scalar::parse_bytes_strict(partial_sig->data, psig)) return 0;

    auto pn = pn_unpack(pubnonce);
    auto s  = sess_unpack(session);

    // P2-PERF-003: reconstruct compressed form directly from secp256k1_pubkey
    // internal layout (X at data[0..31], Y parity at data[63]&1) instead of
    // calling secp256k1_ec_pubkey_serialize() — avoids the full serialize path.
    std::array<unsigned char, 33> pk33;
    pk33[0] = (pubkey->data[63] & 1u) ? 0x03u : 0x02u;
    std::memcpy(pk33.data() + 1, pubkey->data, 32);

    // Fail-closed signer lookup (Rule 13): returning 0 for "not found" is banned
    // because 0 is a valid signer index. Use explicit found flag.
    bool found = false;
    std::size_t idx = 0;
    for (std::size_t i = 0; i < e->compressed.size(); ++i) {
        if (e->compressed[i] == pk33) { idx = i; found = true; break; }
    }
    if (!found) return 0;  // unrecognized signer — fail, do not sign as signer 0

    std::array<unsigned char, 32> pk_x;
    std::memcpy(pk_x.data(), pk33.data() + 1, 32);
    return secp256k1::musig2_partial_verify(psig, pn, pk_x, e->ctx, s, idx) ? 1 : 0;
}

int secp256k1_musig_partial_sig_agg(
    const secp256k1_context* ctx,
    unsigned char* sig64,
    const secp256k1_musig_session* session,
    const secp256k1_musig_partial_sig* const* partial_sigs,
    size_t n_sigs)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!sig64 || !session || !partial_sigs || n_sigs == 0) return 0;

    // P2-PERF-005: SBO — for N<=16 (common 2-of-2 case) parse into a stack
    // array first; construct the vector from the stack data in one shot,
    // avoiding default-init + element-by-element assignment on the heap path.
    constexpr size_t kSBOLimit = 16;
    auto s = sess_unpack(session);
    // BIP-327 additive-tweak correction: tweak_s = e*g*tacc is not carried in the session
    // blob (no spare bytes), so recompute it from the still-live keyagg cache (released
    // only at the end of this function). tacc defaults to 0 for untweaked sessions.
    if (auto ke = ka_get_by_token(sess_load_cache_token(session))) {
        Scalar const gtacc = ke->ctx.Q_negated ? ke->ctx.tacc.negate() : ke->ctx.tacc;
        s.tweak_s = s.e * gtacc;
    }
    std::array<uint8_t, 64> final_sig{};
    if (n_sigs <= kSBOLimit) {
        Scalar sbo_buf[kSBOLimit];
        for (size_t i = 0; i < n_sigs; ++i) {
            if (!partial_sigs[i]) return 0;
            if (!Scalar::parse_bytes_strict(partial_sigs[i]->data, sbo_buf[i])) return 0;
        }
        std::vector<Scalar> psigs(sbo_buf, sbo_buf + n_sigs);
        final_sig = secp256k1::musig2_partial_sig_agg(psigs, s);
    } else {
        std::vector<Scalar> psigs(n_sigs);
        for (size_t i = 0; i < n_sigs; ++i) {
            if (!partial_sigs[i]) return 0;
            if (!Scalar::parse_bytes_strict(partial_sigs[i]->data, psigs[i])) return 0;
        }
        final_sig = secp256k1::musig2_partial_sig_agg(psigs, s);
    }

    // SHIM-004: fail-closed on all-zero signature — degenerate aggregation result.
    // A 64-byte all-zero Schnorr signature is always invalid; returning it as success
    // would allow a caller to serialize and broadcast a trivially invalid signature.
    // SHIM-MUSIG-CT: use a CT accumulator (no early-exit branch) so the all-zero
    // check does not leak information about the aggregated signature value via
    // cache or branch-predictor timing. The loop runs all 64 bytes unconditionally.
    uint32_t nonzero = 0;
    for (int i = 0; i < 64; ++i) nonzero |= static_cast<uint32_t>(final_sig[i]);
    const bool all_zero = (nonzero == 0);
    if (all_zero) {
        ka_remove_by_token(sess_load_cache_token(session));
        return 0;
    }
    std::memcpy(sig64, final_sig.data(), 64);
    // Protocol complete — release the side-channel map entry so the
    // keyagg_cache address can be safely reused by a future session.
    ka_remove_by_token(sess_load_cache_token(session));
    return 1;
}

int secp256k1_musig_partial_sig_serialize(
    const secp256k1_context* ctx,
    unsigned char* out32,
    const secp256k1_musig_partial_sig* partial_sig)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!out32 || !partial_sig) return 0;
    std::memcpy(out32, partial_sig->data, 32);
    return 1;
}

int secp256k1_musig_partial_sig_parse(
    const secp256k1_context* ctx,
    secp256k1_musig_partial_sig* partial_sig,
    const unsigned char* in32)
{
    SHIM_REQUIRE_CTX(ctx);
    if (!partial_sig || !in32) return 0;
    Scalar s;
    if (!Scalar::parse_bytes_strict(in32, s)) return 0;
    auto b = s.to_bytes();
    std::memcpy(partial_sig->data, b.data(), 32);
    std::memset(partial_sig->data + 32, 0, 4);
    return 1;
}

void secp256k1_musig_keyagg_cache_clear(secp256k1_musig_keyagg_cache* keyagg_cache) {
    if (!keyagg_cache) return;
    ka_remove(keyagg_cache);
}

} // extern "C"
