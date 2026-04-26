#include "secp256k1_musig.h"
#include "secp256k1_extrakeys.h"
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "secp256k1/musig2.hpp"
#include "secp256k1/field.hpp"
#include "secp256k1/scalar.hpp"
#include "secp256k1/point.hpp"
#include "secp256k1/precompute.hpp"

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
    if (y.square() != y2) return Point::infinity();
    bool y_odd = (y.limbs()[0] & 1) != 0;
    if (y_odd != (compressed[0] == 0x03)) y = y.negate();
    return Point::from_affine(x, y);
}

static void compress(const Point& pt, unsigned char out[33]) {
    auto arr = pt.to_compressed();
    std::memcpy(out, arr.data(), 33);
}

// ---------------------------------------------------------------------------
// keyagg_cache registry: MuSig2KeyAggCtx has a std::vector so cannot be
// stored inline.  Registry keyed on the caller's struct address.
// ---------------------------------------------------------------------------
struct KAEntry {
    secp256k1::MuSig2KeyAggCtx ctx;
    std::array<unsigned char, 33> agg_pk_comp;             // compressed aggregated pubkey
    std::vector<std::array<unsigned char, 33>> compressed;  // 33-byte compressed keys in aggregation order

    // Return signer index for a given secret key; 0 if not found.
    std::size_t find_index(const Scalar& sk) const {
        auto pk_comp = secp256k1::fast::scalar_mul_generator(sk).to_compressed();
        std::array<unsigned char, 33> pk33;
        std::memcpy(pk33.data(), pk_comp.data(), 33);
        for (std::size_t i = 0; i < compressed.size(); ++i) {
            if (compressed[i] == pk33) return i;
        }
        return 0; // fallback: single-signer or unrecognised key
    }
};

std::mutex g_mu;
std::unordered_map<const void*, std::unique_ptr<KAEntry>> g_ka;

static KAEntry* ka_get(const secp256k1_musig_keyagg_cache* p) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto it = g_ka.find(p);
    return it != g_ka.end() ? it->second.get() : nullptr;
}

static KAEntry* ka_put(const secp256k1_musig_keyagg_cache* p, std::unique_ptr<KAEntry> v) {
    std::lock_guard<std::mutex> lk(g_mu);
    auto& slot = g_ka[p];
    slot = std::move(v);
    return slot.get();
}

// ---------------------------------------------------------------------------
// SecNonce: k1[32] | k2[32] packed into data[132]
// ---------------------------------------------------------------------------
static void sn_pack(secp256k1_musig_secnonce* out, const Scalar& k1, const Scalar& k2) {
    auto b1 = k1.to_bytes(); std::memcpy(out->data,      b1.data(), 32);
    auto b2 = k2.to_bytes(); std::memcpy(out->data + 32, b2.data(), 32);
}

static bool sn_unpack(const secp256k1_musig_secnonce* in, Scalar& k1, Scalar& k2) {
    return Scalar::parse_bytes_strict(in->data,      k1) &&
           Scalar::parse_bytes_strict(in->data + 32, k2);
}

// PubNonce: R1[33] | R2[33] = 66 bytes (inline, exact fit)
static void pn_pack(secp256k1_musig_pubnonce* out, const secp256k1::MuSig2PubNonce& pn) {
    std::memcpy(out->data,      pn.R1.data(), 33);
    std::memcpy(out->data + 33, pn.R2.data(), 33);
}

static secp256k1::MuSig2PubNonce pn_unpack(const secp256k1_musig_pubnonce* in) {
    secp256k1::MuSig2PubNonce pn;
    std::memcpy(pn.R1.data(), in->data,      33);
    std::memcpy(pn.R2.data(), in->data + 33, 33);
    return pn;
}

// Session: R[33] | b[32] | e[32] | R_negated[1] = 98 bytes in data[133]
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

} // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Key aggregation
// ---------------------------------------------------------------------------

int secp256k1_musig_pubkey_agg(
    const secp256k1_context* /*ctx*/,
    secp256k1_xonly_pubkey* agg_pk,
    secp256k1_musig_keyagg_cache* keyagg_cache,
    const secp256k1_pubkey* const* pubkeys,
    size_t n_pubkeys)
{
    if (!keyagg_cache || !pubkeys || n_pubkeys == 0) return 0;

    std::vector<std::array<unsigned char, 33>> comp33(n_pubkeys);
    for (size_t i = 0; i < n_pubkeys; ++i) {
        if (!pubkeys[i]) return 0;
        unsigned char buf[33]; size_t len = 33;
        secp256k1_ec_pubkey_serialize(nullptr, buf, &len, pubkeys[i], SECP256K1_EC_COMPRESSED);
        std::memcpy(comp33[i].data(), buf, 33);
    }

    auto e = std::make_unique<KAEntry>();
    e->ctx = secp256k1::musig2_key_agg(comp33);
    if (e->ctx.Q.is_infinity()) return 0;

    // musig2_key_agg normalizes Q to even-Y for signing, but secp256k1_musig_pubkey_get
    // must return the plain (possibly odd-Y) aggregate key. Restore original Y if negated.
    Point orig_Q = e->ctx.Q_negated ? e->ctx.Q.negate() : e->ctx.Q;
    compress(orig_Q, e->agg_pk_comp.data());
    e->compressed = comp33;

    if (agg_pk) std::memcpy(agg_pk->data, e->agg_pk_comp.data() + 1, 32);

    ka_put(keyagg_cache, std::move(e));
    return 1;
}

int secp256k1_musig_pubkey_get(
    const secp256k1_context* /*ctx*/,
    secp256k1_pubkey* agg_pk,
    const secp256k1_musig_keyagg_cache* keyagg_cache)
{
    if (!agg_pk || !keyagg_cache) return 0;
    KAEntry* e = ka_get(keyagg_cache);
    if (!e) return 0;
    secp256k1_ec_pubkey_parse(nullptr, agg_pk, e->agg_pk_comp.data(), 33);
    return 1;
}

int secp256k1_musig_pubkey_ec_tweak_add(
    const secp256k1_context* /*ctx*/,
    secp256k1_pubkey* output_pubkey,
    secp256k1_musig_keyagg_cache* keyagg_cache,
    const unsigned char* tweak32)
{
    if (!keyagg_cache || !tweak32) return 0;
    KAEntry* e = ka_get(keyagg_cache);
    if (!e) return 0;
    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) return 0;
    Point tG = secp256k1::fast::scalar_mul_generator(t);
    e->ctx.Q = e->ctx.Q.add(tG);
    if (e->ctx.Q.is_infinity()) return 0;
    compress(e->ctx.Q, e->agg_pk_comp.data());
    if (output_pubkey) secp256k1_ec_pubkey_parse(nullptr, output_pubkey, e->agg_pk_comp.data(), 33);
    return 1;
}

int secp256k1_musig_pubkey_xonly_tweak_add(
    const secp256k1_context* /*ctx*/,
    secp256k1_pubkey* output_pubkey,
    secp256k1_musig_keyagg_cache* keyagg_cache,
    const unsigned char* tweak32)
{
    if (!keyagg_cache || !tweak32) return 0;
    KAEntry* e = ka_get(keyagg_cache);
    if (!e) return 0;
    if (!e->ctx.Q.has_even_y()) {
        e->ctx.Q = e->ctx.Q.negate();
        e->ctx.Q_negated = !e->ctx.Q_negated;
    }
    Scalar t;
    if (!Scalar::parse_bytes_strict(tweak32, t)) return 0;
    Point tG = secp256k1::fast::scalar_mul_generator(t);
    e->ctx.Q = e->ctx.Q.add(tG);
    if (e->ctx.Q.is_infinity()) return 0;
    compress(e->ctx.Q, e->agg_pk_comp.data());
    if (output_pubkey) secp256k1_ec_pubkey_parse(nullptr, output_pubkey, e->agg_pk_comp.data(), 33);
    return 1;
}

// ---------------------------------------------------------------------------
// Nonce generation
// ---------------------------------------------------------------------------

int secp256k1_musig_nonce_gen(
    const secp256k1_context* /*ctx*/,
    secp256k1_musig_secnonce* secnonce,
    secp256k1_musig_pubnonce* pubnonce,
    const unsigned char* session_id32,
    const unsigned char* seckey,
    const secp256k1_pubkey* pubkey,
    const unsigned char* msg32,
    const secp256k1_musig_keyagg_cache* /*keyagg_cache*/,
    const unsigned char* /*extra_input32*/)
{
    if (!secnonce || !pubnonce || !session_id32) return 0;

    Scalar sk;
    if (seckey) {
        if (!Scalar::parse_bytes_strict(seckey, sk)) return 0;
    } else {
        if (!Scalar::parse_bytes_strict(session_id32, sk)) return 0;
    }

    std::array<unsigned char, 32> pub_x = {};
    if (pubkey) {
        unsigned char buf[33]; size_t len = 33;
        secp256k1_ec_pubkey_serialize(nullptr, buf, &len, pubkey, SECP256K1_EC_COMPRESSED);
        std::memcpy(pub_x.data(), buf + 1, 32);
    }

    std::array<unsigned char, 32> msg = {};
    if (msg32) std::memcpy(msg.data(), msg32, 32);

    std::array<unsigned char, 32> agg_x = {};

    auto [sn, pn] = secp256k1::musig2_nonce_gen(sk, pub_x, agg_x, msg, session_id32);
    sn_pack(secnonce, sn.k1, sn.k2);
    pn_pack(pubnonce, pn);
    return 1;
}

int secp256k1_musig_pubnonce_serialize(
    const secp256k1_context* /*ctx*/,
    unsigned char* out66,
    const secp256k1_musig_pubnonce* nonce)
{
    if (!out66 || !nonce) return 0;
    std::memcpy(out66, nonce->data, 66);
    return 1;
}

int secp256k1_musig_pubnonce_parse(
    const secp256k1_context* /*ctx*/,
    secp256k1_musig_pubnonce* nonce,
    const unsigned char* in66)
{
    if (!nonce || !in66) return 0;
    if (decompress(in66).is_infinity()) return 0;
    if (decompress(in66 + 33).is_infinity()) return 0;
    std::memcpy(nonce->data, in66, 66);
    return 1;
}

int secp256k1_musig_nonce_agg(
    const secp256k1_context* /*ctx*/,
    secp256k1_musig_aggnonce* aggnonce,
    const secp256k1_musig_pubnonce* const* pubnonces,
    size_t n_pubnonces)
{
    if (!aggnonce || !pubnonces || n_pubnonces == 0) return 0;
    std::vector<secp256k1::MuSig2PubNonce> pnv;
    pnv.reserve(n_pubnonces);
    for (size_t i = 0; i < n_pubnonces; ++i) {
        if (!pubnonces[i]) return 0;
        pnv.push_back(pn_unpack(pubnonces[i]));
    }
    auto agg = secp256k1::musig2_nonce_agg(pnv);
    compress(agg.R1, aggnonce->data);
    compress(agg.R2, aggnonce->data + 33);
    return 1;
}

// ---------------------------------------------------------------------------
// Session signing
// ---------------------------------------------------------------------------

int secp256k1_musig_nonce_process(
    const secp256k1_context* /*ctx*/,
    secp256k1_musig_session* session,
    const secp256k1_musig_aggnonce* aggnonce,
    const unsigned char* msg32,
    const secp256k1_musig_keyagg_cache* keyagg_cache)
{
    if (!session || !aggnonce || !msg32 || !keyagg_cache) return 0;
    KAEntry* e = ka_get(keyagg_cache);
    if (!e) return 0;
    secp256k1::MuSig2AggNonce an;
    an.R1 = decompress(aggnonce->data);
    an.R2 = decompress(aggnonce->data + 33);
    std::array<unsigned char, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    auto s = secp256k1::musig2_start_sign_session(an, e->ctx, msg);
    sess_pack(session, s);
    return 1;
}

int secp256k1_musig_partial_sign(
    const secp256k1_context* /*ctx*/,
    secp256k1_musig_partial_sig* partial_sig,
    secp256k1_musig_secnonce* secnonce,
    const secp256k1_keypair* keypair,
    const secp256k1_musig_keyagg_cache* keyagg_cache,
    const secp256k1_musig_session* session)
{
    if (!partial_sig || !secnonce || !keypair || !keyagg_cache || !session) return 0;
    KAEntry* e = ka_get(keyagg_cache);
    if (!e) return 0;

    Scalar k1, k2;
    if (!sn_unpack(secnonce, k1, k2)) return 0;
    std::memset(secnonce->data, 0, sizeof(secnonce->data)); // zeroize (single-use)

    Scalar sk;
    if (!Scalar::parse_bytes_strict(keypair->data, sk)) return 0;

    secp256k1::MuSig2SecNonce sn{ k1, k2 };
    auto s = sess_unpack(session);
    std::size_t idx = e->find_index(sk);
    auto psig = secp256k1::musig2_partial_sign(sn, sk, e->ctx, s, idx);
    auto b = psig.to_bytes();
    std::memcpy(partial_sig->data, b.data(), 32);
    std::memset(partial_sig->data + 32, 0, 4);
    return 1;
}

int secp256k1_musig_partial_sig_verify(
    const secp256k1_context* /*ctx*/,
    const secp256k1_musig_partial_sig* partial_sig,
    const secp256k1_musig_pubnonce* pubnonce,
    const secp256k1_pubkey* pubkey,
    const secp256k1_musig_keyagg_cache* keyagg_cache,
    const secp256k1_musig_session* session)
{
    if (!partial_sig || !pubnonce || !pubkey || !keyagg_cache || !session) return 0;
    KAEntry* e = ka_get(keyagg_cache);
    if (!e) return 0;

    Scalar psig;
    if (!Scalar::parse_bytes_strict(partial_sig->data, psig)) return 0;

    auto pn = pn_unpack(pubnonce);
    auto s  = sess_unpack(session);

    unsigned char buf[33]; size_t len = 33;
    secp256k1_ec_pubkey_serialize(nullptr, buf, &len, pubkey, SECP256K1_EC_COMPRESSED);
    std::array<unsigned char, 33> pk33;
    std::memcpy(pk33.data(), buf, 33);

    std::size_t idx = 0;
    for (std::size_t i = 0; i < e->compressed.size(); ++i) {
        if (e->compressed[i] == pk33) { idx = i; break; }
    }

    std::array<unsigned char, 32> pk_x;
    std::memcpy(pk_x.data(), buf + 1, 32);
    return secp256k1::musig2_partial_verify(psig, pn, pk_x, e->ctx, s, idx) ? 1 : 0;
}

int secp256k1_musig_partial_sig_agg(
    const secp256k1_context* /*ctx*/,
    unsigned char* sig64,
    const secp256k1_musig_session* session,
    const secp256k1_musig_partial_sig* const* partial_sigs,
    size_t n_sigs)
{
    if (!sig64 || !session || !partial_sigs || n_sigs == 0) return 0;
    std::vector<Scalar> psigs(n_sigs);
    for (size_t i = 0; i < n_sigs; ++i) {
        if (!partial_sigs[i]) return 0;
        if (!Scalar::parse_bytes_strict(partial_sigs[i]->data, psigs[i])) return 0;
    }
    auto s = sess_unpack(session);
    auto final_sig = secp256k1::musig2_partial_sig_agg(psigs, s);
    std::memcpy(sig64, final_sig.data(), 64);
    return 1;
}

int secp256k1_musig_partial_sig_serialize(
    const secp256k1_context* /*ctx*/,
    unsigned char* out32,
    const secp256k1_musig_partial_sig* partial_sig)
{
    if (!out32 || !partial_sig) return 0;
    std::memcpy(out32, partial_sig->data, 32);
    return 1;
}

int secp256k1_musig_partial_sig_parse(
    const secp256k1_context* /*ctx*/,
    secp256k1_musig_partial_sig* partial_sig,
    const unsigned char* in32)
{
    if (!partial_sig || !in32) return 0;
    Scalar s;
    if (!Scalar::parse_bytes_strict(in32, s)) return 0;
    auto b = s.to_bytes();
    std::memcpy(partial_sig->data, b.data(), 32);
    std::memset(partial_sig->data + 32, 0, 4);
    return 1;
}

} // extern "C"
