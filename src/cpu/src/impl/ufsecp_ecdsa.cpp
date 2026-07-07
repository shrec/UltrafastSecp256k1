/* ============================================================================
 * ECDSA sign/verify/recover, Schnorr sign/verify, ECDH, batch signing
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

// -- Thread-local pubkey caches (PERF-01/02) ----------------------------------
// Caching EcdsaPublicKey / SchnorrXonlyPubkey with precomputed GLV tables
// eliminates sqrt decompression (~2.8µs) + GLV table rebuild (~1.9µs) on
// every verify call for repeated pubkeys (ConnectBlock hot path).
namespace {

struct UfsecpEcdsaPkCache {
    static constexpr std::size_t SLOTS = 256;
    struct Slot {
        std::uint8_t raw33[33]{};
        secp256k1::EcdsaPublicKey epk{};
        bool valid = false;
    };
    Slot slots[SLOTS]{};

    static std::size_t slot_of(const uint8_t* k) noexcept {
        std::uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < 8; ++i) h = (h ^ k[i]) * 1099511628211ULL;
        return h & (SLOTS - 1);
    }
    const secp256k1::EcdsaPublicKey* get(const uint8_t* k) const noexcept {
        const Slot& s = slots[slot_of(k)];
        if (s.valid && std::memcmp(s.raw33, k, 33) == 0) return &s.epk;
        return nullptr;
    }
    const secp256k1::EcdsaPublicKey* put(const uint8_t* k) noexcept {
        Slot& s = slots[slot_of(k)];
        s.valid = secp256k1::ecdsa_pubkey_parse(s.epk, k, 33);
        if (s.valid) std::memcpy(s.raw33, k, 33);
        return s.valid ? &s.epk : nullptr;
    }
};
static thread_local UfsecpEcdsaPkCache s_ecdsa_pk_cache;

struct UfsecpSchnorrPkCache {
    static constexpr std::size_t SLOTS = 256;
    struct Slot {
        std::uint8_t rawx[32]{};
        secp256k1::SchnorrXonlyPubkey epk{};
        bool valid = false;
    };
    Slot slots[SLOTS]{};

    static std::size_t slot_of(const uint8_t* k) noexcept {
        std::uint64_t h = 14695981039346656037ULL;
        for (int i = 0; i < 8; ++i) h = (h ^ k[i]) * 1099511628211ULL;
        return h & (SLOTS - 1);
    }
    const secp256k1::SchnorrXonlyPubkey* get(const uint8_t* k) const noexcept {
        const Slot& s = slots[slot_of(k)];
        if (s.valid && std::memcmp(s.rawx, k, 32) == 0) return &s.epk;
        return nullptr;
    }
    const secp256k1::SchnorrXonlyPubkey* put(const uint8_t* k) noexcept {
        Slot& s = slots[slot_of(k)];
        s.valid = secp256k1::schnorr_xonly_pubkey_parse(s.epk, k);
        if (s.valid) std::memcpy(s.rawx, k, 32);
        return s.valid ? &s.epk : nullptr;
    }
};
static thread_local UfsecpSchnorrPkCache s_schnorr_pk_cache;

using ScalarLimbs = Scalar::limbs_type;

static constexpr ScalarLimbs kScalarOrderLimbs{
    0xBFD25E8CD0364141ULL,
    0xBAAEDCE6AF48A03BULL,
    0xFFFFFFFFFFFFFFFEULL,
    0xFFFFFFFFFFFFFFFFULL
};

static inline uint64_t opaque_read_le64(const uint8_t* p) noexcept {
    uint64_t v = 0;
    for (unsigned i = 0; i < 8; ++i)
        v |= static_cast<uint64_t>(p[i]) << (i * 8);
    return v;
}

static inline void opaque_write_le64(uint8_t* p, uint64_t v) noexcept {
    for (unsigned i = 0; i < 8; ++i)
        p[i] = static_cast<uint8_t>(v >> (i * 8));
}

static inline bool scalar_limbs_zero(const ScalarLimbs& a) noexcept {
    return (a[0] | a[1] | a[2] | a[3]) == 0;
}

static inline bool scalar_limbs_ge(const ScalarLimbs& a,
                                   const ScalarLimbs& b) noexcept {
    for (int i = 3; i >= 0; --i) {
        const auto idx = static_cast<std::size_t>(i);
        if (a[idx] > b[idx]) return true;
        if (a[idx] < b[idx]) return false;
    }
    return true;
}

static inline bool opaque_scalar_parse_strict_nonzero(const uint8_t* opaque,
                                                      Scalar& out) noexcept {
    const ScalarLimbs limbs{
        opaque_read_le64(opaque),
        opaque_read_le64(opaque + 8),
        opaque_read_le64(opaque + 16),
        opaque_read_le64(opaque + 24)
    };
    if (scalar_limbs_zero(limbs) || scalar_limbs_ge(limbs, kScalarOrderLimbs))
        return false;
    out = Scalar::from_limbs(limbs);
    return true;
}

static inline void opaque_scalar_write(const Scalar& s,
                                       uint8_t opaque[32]) noexcept {
    const auto& limbs = s.limbs();
    opaque_write_le64(opaque, limbs[0]);
    opaque_write_le64(opaque + 8, limbs[1]);
    opaque_write_le64(opaque + 16, limbs[2]);
    opaque_write_le64(opaque + 24, limbs[3]);
}

static inline bool ecdsa_sig_parse_opaque(
    const uint8_t opaque64[64],
    secp256k1::ECDSASignature& out,
    bool normalize) noexcept {
    Scalar r;
    Scalar s;
    if (!opaque_scalar_parse_strict_nonzero(opaque64, r) ||
        !opaque_scalar_parse_strict_nonzero(opaque64 + 32, s))
        return false;
    out = secp256k1::ECDSASignature{r, s};
    if (normalize) out = out.normalize();
    return true;
}

static inline void ecdsa_sig_write_opaque(
    const secp256k1::ECDSASignature& sig,
    uint8_t opaque64[64]) noexcept {
    opaque_scalar_write(sig.r, opaque64);
    opaque_scalar_write(sig.s, opaque64 + 32);
}

// Point-only decompress: recover the curve point from a 33-byte compressed pubkey
// WITHOUT building the cached GLV verify tables. The batch path uses only the point
// (dual_scalar_mul_gen_point builds its own GLV decomposition), so ecdsa_pubkey_parse's
// build_schnorr_verify_tables (~1.95us/key) was pure waste here and dominated the
// per-row parse (~2.6x libsecp's decompress). Mirrors the len==33 branch of
// ecdsa_pubkey_parse (src/cpu/src/ecdsa.cpp) without the table build.
static inline bool pubkey33_to_point(const uint8_t pubkey33[33],
                                     secp256k1::fast::Point& out) noexcept {
    using secp256k1::fast::FieldElement;
    using secp256k1::fast::FieldElement52;
    using secp256k1::fast::Point;
    if (pubkey33[0] != 0x02 && pubkey33[0] != 0x03) return false;
    FieldElement x;
    if (!FieldElement::parse_bytes_strict(pubkey33 + 1, x)) return false;
    // The dominant cost is the field sqrt. Do it in FE52 (5x52): its mul/sqr are ~3x
    // faster than the 4x64 fast::FieldElement, and FE52 is the representation the verify
    // (dual_scalar_mul_gen_point) uses internally anyway. y = sqrt(x^3 + 7); the QR
    // self-check (y^2 == x^3+7) rejects x values that are not on the curve.
    const FieldElement52 x52 = FieldElement52::from_fe(x);
    static const std::uint64_t k7[4] = {7u, 0u, 0u, 0u};
    const FieldElement52 y2  = x52.square() * x52 + FieldElement52::from_4x64_limbs(k7);
    const FieldElement52 y52 = y2.sqrt();
    if (!(y52.square() == y2)) return false;   // not on curve
    FieldElement y = y52.to_fe();              // FE52 -> 4x64 (normalizes)
    const bool y_is_odd = (y.limbs()[0] & 1u) != 0;
    const bool want_odd = (pubkey33[0] == 0x03);
    if (y_is_odd != want_odd) y = FieldElement::zero() - y;
    out = Point::from_affine(x, y);
    return !out.is_infinity();
}

static inline bool ecdsa_entry_from_opaque(
    const uint8_t msg32[32],
    const uint8_t pubkey33[33],
    const uint8_t opaque64[64],
    secp256k1::ECDSABatchEntry& out) noexcept {
    std::memcpy(out.msg_hash.data(), msg32, 32);
    // Point-only decompress (no wasted GLV table build — batch builds its own GLV).
    if (!pubkey33_to_point(pubkey33, out.public_key)) return false;
    return ecdsa_sig_parse_opaque(opaque64, out.signature, true);
}

} // namespace

ufsecp_error_t ufsecp_ecdsa_sign(ufsecp_ctx* ctx,
                                 const uint8_t msg32[32],
                                 const uint8_t privkey[32],
                                 uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !privkey || !sig64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    // Guardrail #4: CT path returns {0,0} on degenerate nonce (≈2^-256).
    if (SECP256K1_UNLIKELY(!sig.is_valid())) {
        std::memset(sig64_out, 0, 64);
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "signing produced degenerate output");
    }
    auto compact = sig.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sign_verified(ufsecp_ctx* ctx,
                                          const uint8_t msg32[32],
                                          const uint8_t privkey[32],
                                          uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !privkey || !sig64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto sig = secp256k1::ct::ecdsa_sign_verified(msg, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (SECP256K1_UNLIKELY(!sig.is_valid())) {
        std::memset(sig64_out, 0, 64);
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "signing produced degenerate output");
    }
    auto compact = sig.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_verify(ufsecp_ctx* ctx,
                                   const uint8_t msg32[32],
                                   const uint8_t sig64[64],
                                   const uint8_t pubkey33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !sig64 || !pubkey33)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);

    secp256k1::ECDSASignature ecdsasig;
    if (SECP256K1_UNLIKELY(!secp256k1::ECDSASignature::parse_compact_strict(compact, ecdsasig))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "non-canonical compact sig");
    }
    // BIP-62 low-S enforcement: reject high-S signatures (s > n/2)
    if (!ecdsasig.is_low_s()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "high-S signature rejected (BIP-62)");
    }
    const secp256k1::EcdsaPublicKey* epk = s_ecdsa_pk_cache.get(pubkey33);
    if (!epk) epk = s_ecdsa_pk_cache.put(pubkey33);
    if (SECP256K1_UNLIKELY(!epk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid public key");
    }

    if (SECP256K1_UNLIKELY(!secp256k1::ecdsa_verify(msg, *epk, ecdsasig))) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "ECDSA verify failed");
    }

    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sig_compact_to_opaque(
    ufsecp_ctx* ctx,
    const uint8_t sig64[64],
    uint8_t opaque64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !sig64 || !opaque64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);

    secp256k1::ECDSASignature sig;
    if (SECP256K1_UNLIKELY(!secp256k1::ECDSASignature::parse_compact_strict(compact, sig))) {
        std::memset(opaque64_out, 0, 64);
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "non-canonical compact sig");
    }

    ecdsa_sig_write_opaque(sig, opaque64_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sig_opaque_to_compact(
    ufsecp_ctx* ctx,
    const uint8_t opaque64[64],
    uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !opaque64 || !sig64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::ECDSASignature sig;
    if (SECP256K1_UNLIKELY(!ecdsa_sig_parse_opaque(opaque64, sig, false))) {
        std::memset(sig64_out, 0, 64);
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid opaque ECDSA sig");
    }

    const auto compact = sig.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sig_normalize_opaque(
    ufsecp_ctx* ctx,
    const uint8_t opaque64[64],
    uint8_t opaque64_out[64],
    int* changed_out) {
    if (SECP256K1_UNLIKELY(!ctx || !opaque64 || !opaque64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::ECDSASignature sig;
    if (SECP256K1_UNLIKELY(!ecdsa_sig_parse_opaque(opaque64, sig, false))) {
        std::memset(opaque64_out, 0, 64);
        if (changed_out) *changed_out = 0;
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid opaque ECDSA sig");
    }

    const auto normalized = sig.normalize();
    uint8_t tmp[64];
    ecdsa_sig_write_opaque(normalized, tmp);
    const int changed = std::memcmp(opaque64, tmp, 64) != 0 ? 1 : 0;
    std::memcpy(opaque64_out, tmp, 64);
    if (changed_out) *changed_out = changed;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_verify_opaque(
    ufsecp_ctx* ctx,
    const uint8_t msg32[32],
    const uint8_t opaque64[64],
    const uint8_t pubkey33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !opaque64 || !pubkey33)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);

    secp256k1::ECDSASignature sig;
    if (SECP256K1_UNLIKELY(!ecdsa_sig_parse_opaque(opaque64, sig, true))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid opaque ECDSA sig");
    }

    const secp256k1::EcdsaPublicKey* epk = s_ecdsa_pk_cache.get(pubkey33);
    if (!epk) epk = s_ecdsa_pk_cache.put(pubkey33);
    if (SECP256K1_UNLIKELY(!epk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid public key");
    }

    if (SECP256K1_UNLIKELY(!secp256k1::ecdsa_verify(msg, *epk, sig))) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "ECDSA verify failed");
    }

    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_verify_opaque_batch(
    ufsecp_ctx* ctx,
    const uint8_t* msg_hashes32,
    const uint8_t* pubkeys33,
    const uint8_t* opaque_sigs64,
    size_t count,
    uint8_t* out_results) {
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (SECP256K1_UNLIKELY(!msg_hashes32 || !pubkeys33 || !opaque_sigs64 || !out_results))
        return UFSECP_ERR_NULL_ARG;
    if (count > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    std::memset(out_results, 0, count);

    try {
        std::vector<secp256k1::ECDSABatchEntry> batch(count);
        bool parsed = true;
        for (size_t i = 0; i < count; ++i) {
            parsed = ecdsa_entry_from_opaque(msg_hashes32 + i * 32,
                                             pubkeys33 + i * 33,
                                             opaque_sigs64 + i * 64,
                                             batch[i]);
            if (!parsed) break;
        }

        if (parsed && secp256k1::ecdsa_batch_verify(batch.data(), count)) {
            std::memset(out_results, 1, count);
            return UFSECP_OK;
        }

        for (size_t i = 0; i < count; ++i) {
            secp256k1::ECDSABatchEntry one{};
            const bool ok = ecdsa_entry_from_opaque(msg_hashes32 + i * 32,
                                                    pubkeys33 + i * 33,
                                                    opaque_sigs64 + i * 64,
                                                    one) &&
                            secp256k1::ecdsa_batch_verify(&one, 1);
            out_results[i] = ok ? 1u : 0u;
        }
        return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_ecdsa_verify_opaque_rows(
    ufsecp_ctx* ctx,
    const uint8_t* rows,
    size_t stride,
    size_t count,
    uint8_t* out_results) {
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (SECP256K1_UNLIKELY(!rows || !out_results)) return UFSECP_ERR_NULL_ARG;
    if (stride < 129u) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "opaque row stride < 129");
    if (count > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    std::memset(out_results, 0, count);

    try {
        std::vector<secp256k1::ECDSABatchEntry> batch(count);
        size_t parsed_count = 0;
        for (; parsed_count < count; ++parsed_count) {
            const size_t i = parsed_count;
            const uint8_t* row = rows + i * stride;
            if (!ecdsa_entry_from_opaque(row, row + 32, row + 65, batch[i])) {
                break;
            }
        }

        if (parsed_count == count && secp256k1::ecdsa_batch_verify(batch.data(), count)) {
            std::memset(out_results, 1, count);
            return UFSECP_OK;
        }

        for (size_t i = 0; i < count; ++i) {
            const uint8_t* row = rows + i * stride;
            secp256k1::ECDSABatchEntry one{};
            const bool ok =
                ecdsa_entry_from_opaque(row, row + 32, row + 65, one) &&
                secp256k1::ecdsa_batch_verify(&one, 1);
            out_results[i] = ok ? 1u : 0u;
        }
        return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

// Multi-threaded twin of ufsecp_ecdsa_verify_opaque_rows: identical marshalling
// and per-row semantics; only the all-valid fast check fans across the engine's
// multi-threaded path. The per-row locate fallback (run only after a failure)
// stays serial. max_threads: 0 => auto, 1 => serial.
ufsecp_error_t ufsecp_ecdsa_verify_opaque_rows_mt(
    ufsecp_ctx* ctx,
    const uint8_t* rows,
    size_t stride,
    size_t count,
    uint8_t* out_results,
    size_t max_threads) {
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (count == 0) return UFSECP_OK;
    if (SECP256K1_UNLIKELY(!rows || !out_results)) return UFSECP_ERR_NULL_ARG;
    if (stride < 129u) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "opaque row stride < 129");
    if (count > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    std::memset(out_results, 0, count);

    try {
        // Fused parallel parse + verify: each worker parses AND batch-verifies its own
        // chunk of raw opaque rows in one parallel region. The previous version parsed
        // ALL rows serially (per-row pubkey curve-check) before the parallel verify,
        // making the serial parse an Amdahl ceiling (~28% of runtime at 50k rows) that
        // kept the bridge slower than libsecp regardless of core count. libbitcoin runs
        // the libsecp path with std::for_each(par) — parse+verify parallel together —
        // so this matches that model. Thread/steal sizing mirrors ecdsa_batch_verify_mt.
        static constexpr std::size_t kMinRowsPerThread = 128;
        static constexpr std::size_t kStealFloor = 64;   // >= batch-inversion cutoff (8)
        auto& wpool = secp256k1::detail::batch_worker_pool();
        const unsigned hw = wpool.size();
        const unsigned want = (max_threads == 0)
            ? hw
            : static_cast<unsigned>(std::min<std::size_t>(max_threads, hw));
        const std::size_t by_work = std::max<std::size_t>(1, count / kMinRowsPerThread);
        const unsigned n_threads = static_cast<unsigned>(std::min<std::size_t>(
            static_cast<std::size_t>(want), by_work));
        const std::size_t steal = (n_threads <= 1)
            ? count
            : std::clamp<std::size_t>(count / (static_cast<std::size_t>(n_threads) * 4),
                                      kStealFloor, std::size_t{4096});

        // Fused parse+verify per chunk on the PERSISTENT pool: no per-call thread spawn,
        // and the thread_local scratch below stays warm across calls (one block per call
        // during IBD). Returns false as soon as any chunk has a bad parse or verify.
        const bool all_valid = wpool.run(count, steal, n_threads,
            [&](std::size_t start, std::size_t end) -> bool {
                static thread_local std::vector<secp256k1::ECDSABatchEntry> local;
                local.clear();
                local.resize(end - start);
                for (std::size_t i = start; i < end; ++i) {
                    const uint8_t* row = rows + i * stride;
                    if (!ecdsa_entry_from_opaque(row, row + 32, row + 65, local[i - start]))
                        return false;
                }
                return secp256k1::ecdsa_batch_verify(local.data(), end - start);
            });

        if (all_valid) {
            std::memset(out_results, 1, count);
            return UFSECP_OK;
        }

        // Mixed/invalid batch: serial per-row locate so the caller learns exactly which
        // rows failed (fail-closed). Only runs after the fast path reported a failure.
        for (size_t i = 0; i < count; ++i) {
            const uint8_t* row = rows + i * stride;
            secp256k1::ECDSABatchEntry one{};
            const bool ok =
                ecdsa_entry_from_opaque(row, row + 32, row + 65, one) &&
                secp256k1::ecdsa_batch_verify(&one, 1);
            out_results[i] = ok ? 1u : 0u;
        }
        return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_ecdsa_sig_to_der(ufsecp_ctx* ctx,
                                        const uint8_t sig64[64],
                                        uint8_t* der_out, size_t* der_len) {
    if (SECP256K1_UNLIKELY(!ctx || !sig64 || !der_out || !der_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);

    secp256k1::ECDSASignature ecdsasig;
    if (SECP256K1_UNLIKELY(!secp256k1::ECDSASignature::parse_compact_strict(compact, ecdsasig))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "non-canonical compact sig");
    }

    auto [der, actual_len] = ecdsasig.to_der();
    if (*der_len < actual_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "DER buffer too small");
}

    std::memcpy(der_out, der.data(), actual_len);
    *der_len = actual_len;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sig_from_der(ufsecp_ctx* ctx,
                                         const uint8_t* der, size_t der_len,
                                         uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !der || !sig64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    /* Strict DER parser for ECDSA secp256k1 signatures.
     * Format: 0x30 <total_len> 0x02 <r_len> <r_bytes...> 0x02 <s_len> <s_bytes...>
     *
     * Enforces:
     * - Single-byte length encoding only (no long form)
     * - No negative integers (high bit of first data byte must be 0)
     * - No unnecessary leading zero padding
     * - Exact total length (no trailing bytes)
     * - r, s must be in [1, n-1] (canonical, nonzero)
     * - Max total DER length: 72 bytes */

    /* Max DER ECDSA sig: 2 + 2 + 33 + 2 + 33 = 72 */
    if (der_len < 8 || der_len > 72 || der[0] != 0x30) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: missing/oversized SEQUENCE");
    }

    /* Reject long-form length encoding (bit 7 set = multi-byte length) */
    if (der[1] & 0x80) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: long-form length");
    }

    size_t const seq_len = der[1];
    if (seq_len + 2 != der_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: length mismatch");
    }

    size_t pos = 2;

    /* --- Helper lambda: parse one INTEGER component strictly --- */
    auto parse_int = [&](const char* name, const uint8_t*& out_ptr,
                         size_t& out_len) -> ufsecp_error_t {
        if (pos >= der_len || der[pos] != 0x02) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: missing INTEGER");
        }
        pos++;
        if (pos >= der_len) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: truncated");
        }
        /* Reject long-form length for component */
        if (der[pos] & 0x80) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: long-form int length");
        }
        size_t const int_len = der[pos++];
        if (int_len == 0 || pos + int_len > der_len) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: int length out of bounds");
        }
        /* Reject negative: high bit set on first data byte means negative in DER */
        if (der[pos] & 0x80) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: negative integer");
        }
        /* Reject unnecessary leading zero: 0x00 prefix only valid when next byte
         * has high bit set (positive number needs padding to stay positive).
         * If next byte has high bit clear, the 0x00 is superfluous padding.  */
        if (int_len > 1 && der[pos] == 0x00 && !(der[pos + 1] & 0x80)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: unnecessary leading zero");
        }

        out_ptr = der + pos;
        out_len = int_len;
        /* Strip valid leading zero pad (high bit of next byte is set) */
        if (out_len > 0 && out_ptr[0] == 0x00) { out_ptr++; out_len--; }
        pos += int_len;
        (void)name;
        return UFSECP_OK;
    };

    /* Read R */
    const uint8_t* r_ptr = nullptr;
    size_t r_data_len = 0;
    {
        auto rc = parse_int("R", r_ptr, r_data_len);
        if (rc != UFSECP_OK) return rc;
    }

    /* Read S */
    const uint8_t* s_ptr = nullptr;
    size_t s_data_len = 0;
    {
        auto rc = parse_int("S", s_ptr, s_data_len);
        if (rc != UFSECP_OK) return rc;
    }

    /* Reject trailing bytes after S (must consume entire SEQUENCE) */
    if (pos != der_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: trailing bytes");
    }

    if (r_data_len > 32 || s_data_len > 32) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: component > 32 bytes");
    }

    /* Build compact sig64 (big-endian, right-aligned in 32-byte slots) */
    std::memset(sig64_out, 0, 64);
    /* Explicit null checks for static analyzer (r_ptr/s_ptr guaranteed non-null
     * when *_data_len > 0 by parse_int() success, but SonarCloud can't track it) */
    if (r_data_len > 0 && r_ptr) {
        std::memcpy(sig64_out + (32 - r_data_len), r_ptr, r_data_len);
    }
    if (s_data_len > 0 && s_ptr) {
        std::memcpy(sig64_out + 32 + (32 - s_data_len), s_ptr, s_data_len);
    }

    /* Range check: r and s must be in [1, n-1] (strict nonzero, no reduce) */
    Scalar r_sc, s_sc;
    if (!Scalar::parse_bytes_strict_nonzero(sig64_out, r_sc) ||
        !Scalar::parse_bytes_strict_nonzero(sig64_out + 32, s_sc)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "bad DER: r or s out of range [1,n-1]");
    }

    return UFSECP_OK;
}

/* -- ECDSA Recovery -------------------------------------------------------- */

ufsecp_error_t ufsecp_ecdsa_sign_recoverable(ufsecp_ctx* ctx,
                                             const uint8_t msg32[32],
                                             const uint8_t privkey[32],
                                             uint8_t sig64_out[64],
                                             int* recid_out) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !privkey || !sig64_out || !recid_out)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    // CT path: ct::ecdsa_sign_recoverable uses ct::generator_mul(k) for R=k*G,
    // ct::scalar_inverse(k) via SafeGCD divsteps-59, branchless recovery ID bits,
    // and branchless low-S normalization. All secret stack buffers are securely
    // erased inside ct::ecdsa_sign_recoverable before return.
    auto rsig = secp256k1::ct::ecdsa_sign_recoverable(msg, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (SECP256K1_UNLIKELY(!rsig.sig.is_valid())) {
        std::memset(sig64_out, 0, 64);
        *recid_out = 0;
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "signing produced degenerate output");
    }
    // ct::ecdsa_sign_recoverable already applied CT low-S normalization internally
    auto compact = rsig.sig.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    *recid_out = rsig.recid;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_recover(ufsecp_ctx* ctx,
                                    const uint8_t msg32[32],
                                    const uint8_t sig64[64],
                                    int recid,
                                    uint8_t pubkey33_out[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !sig64 || !pubkey33_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (recid < 0 || recid > 3) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "recid must be 0..3");
    }

    std::array<uint8_t, 32> msg;
    std::memcpy(msg.data(), msg32, 32);
    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);

    secp256k1::ECDSASignature ecdsasig;
    if (SECP256K1_UNLIKELY(!secp256k1::ECDSASignature::parse_compact_strict(compact, ecdsasig))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "non-canonical compact sig");
    }

    auto [point, ok] = secp256k1::ecdsa_recover(msg, ecdsasig, recid);
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "recovery failed");
    }

    point_to_compressed(point, pubkey33_out);
    return UFSECP_OK;
}

/* ===========================================================================
 * Schnorr (BIP-340)
 * =========================================================================== */

ufsecp_error_t ufsecp_schnorr_sign(ufsecp_ctx* ctx,
                                   const uint8_t msg32[32],
                                   const uint8_t privkey[32],
                                   const uint8_t aux_rand[32],
                                   uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !privkey || !aux_rand || !sig64_out)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    std::array<uint8_t, 32> msg_arr, aux_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::memcpy(aux_arr.data(), aux_rand, 32);

    auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    auto sig = secp256k1::ct::schnorr_sign(kp, msg_arr, aux_arr);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
    if (SECP256K1_UNLIKELY(sig.s.is_zero())) {
        std::memset(sig64_out, 0, 64);
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "signing produced degenerate output");
    }
    {
        bool r_all_zero = std::all_of(sig.r.begin(), sig.r.end(),
                                      [](uint8_t b) { return b == 0; });
        if (SECP256K1_UNLIKELY(r_all_zero)) {
            std::memset(sig64_out, 0, 64);
            return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "signing produced degenerate output");
        }
    }
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_sign_verified(ufsecp_ctx* ctx,
                                            const uint8_t msg32[32],
                                            const uint8_t privkey[32],
                                            const uint8_t aux_rand[32],
                                            uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !privkey || !aux_rand || !sig64_out)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    std::array<uint8_t, 32> msg_arr, aux_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::memcpy(aux_arr.data(), aux_rand, 32);

    auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    auto sig = secp256k1::ct::schnorr_sign_verified(kp, msg_arr, aux_arr);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
    if (SECP256K1_UNLIKELY(sig.s.is_zero())) {
        std::memset(sig64_out, 0, 64);
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "signing produced degenerate output");
    }
    {
        bool r_all_zero = std::all_of(sig.r.begin(), sig.r.end(),
                                      [](uint8_t b) { return b == 0; });
        if (SECP256K1_UNLIKELY(r_all_zero)) {
            std::memset(sig64_out, 0, 64);
            return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "signing produced degenerate output");
        }
    }
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
    return UFSECP_OK;
}

// ---------------------------------------------------------------------------
// batch_parallel: dispatch sign_one(i) over [0, count) using N CPU threads.
// Each slot i writes to sigs64_out[i*64..(i+1)*64-1] — non-overlapping so
// no synchronisation is needed on the output buffer.  On any slot error the
// first error code is captured atomically; remaining slots in every thread
// are zeroed and skipped.  After joining, the whole output is re-zeroed
// fail-closed and the captured error is returned.
// Serial fallback: when hardware_concurrency() <= 1 or count == 1.
// ---------------------------------------------------------------------------
template<typename Fn>
static ufsecp_error_t batch_parallel(size_t count, uint8_t* sigs64_out, Fn sign_one)
{
    unsigned hw = std::thread::hardware_concurrency();
    unsigned n_threads = (hw > 1) ? static_cast<unsigned>(
        std::min<size_t>(hw, count)) : 1u;

    if (n_threads <= 1) {
        for (size_t i = 0; i < count; ++i) {
            ufsecp_error_t e = sign_one(i);
            if (SECP256K1_UNLIKELY(e != UFSECP_OK)) return e;
        }
        return UFSECP_OK;
    }

    // Stack-allocated thread pool — no heap allocation. kMaxParallelThreads
    // caps at 64 (no real CPU batch-signing use-case needs more).
    // std::thread default constructor is noexcept and creates an empty thread
    // (joinable()==false), so the array init is safe and cheap.
    static constexpr unsigned kMaxParallelThreads = 64u;
    std::array<std::thread, kMaxParallelThreads> pool{};
    n_threads = std::min(n_threads, kMaxParallelThreads);

    std::atomic<int> first_err{static_cast<int>(UFSECP_OK)};
    const size_t chunk = (count + n_threads - 1) / n_threads;
    unsigned n_active = 0;
    for (unsigned t = 0; t < n_threads; ++t) {
        const size_t start = t * chunk;
        const size_t end   = std::min(start + chunk, count);
        if (start >= end) break;
        pool[t] = std::thread([&, start, end]() {
            for (size_t i = start; i < end; ++i) {
                if (first_err.load(std::memory_order_relaxed) != static_cast<int>(UFSECP_OK)) {
                    std::memset(sigs64_out + i * 64, 0, 64);
                    continue;
                }
                const ufsecp_error_t e = sign_one(i);
                if (SECP256K1_UNLIKELY(e != UFSECP_OK)) {
                    int expected = static_cast<int>(UFSECP_OK);
                    first_err.compare_exchange_strong(
                        expected, static_cast<int>(e),
                        std::memory_order_acq_rel, std::memory_order_relaxed);
                    // zero remaining slots in this chunk
                    for (size_t j = i + 1; j < end; ++j)
                        std::memset(sigs64_out + j * 64, 0, 64);
                    return;
                }
            }
        });
        ++n_active;
    }
    for (unsigned t = 0; t < n_active; ++t) pool[t].join();

    const int err = first_err.load(std::memory_order_acquire);
    if (err != static_cast<int>(UFSECP_OK)) {
        std::memset(sigs64_out, 0, count * 64);
        return static_cast<ufsecp_error_t>(err);
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_sign_batch(
    ufsecp_ctx* ctx,
    size_t count,
    const uint8_t* msgs32,
    const uint8_t* privkeys32,
    uint8_t* sigs64_out)
{
    if (SECP256K1_UNLIKELY(!ctx || !msgs32 || !privkeys32 || !sigs64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (count == 0) return UFSECP_ERR_BAD_INPUT;
    if (count > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    std::size_t total_msg_bytes, total_sig_bytes;
    if (!checked_mul_size(count, std::size_t{32}, total_msg_bytes)
        || !checked_mul_size(count, std::size_t{64}, total_sig_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch size overflow");
    // SEC-003/SEC-008/BSG-12 Fail-closed: pre-zero output so any partial-write
    // before an error is not visible.  batch_parallel re-zeros on error too.
    std::memset(sigs64_out, 0, count * 64);

    const ufsecp_error_t err = batch_parallel(count, sigs64_out,
        [&](size_t i) -> ufsecp_error_t {
            std::array<uint8_t, 32> msg;
            std::memcpy(msg.data(), msgs32 + i * 32, 32);
            Scalar sk;
            if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkeys32 + i * 32, sk))) {
                secp256k1::detail::secure_erase(&sk, sizeof(sk));
                return UFSECP_ERR_BAD_KEY;
            }
            auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            if (SECP256K1_UNLIKELY(!sig.is_valid())) return UFSECP_ERR_INTERNAL;
            const auto compact = sig.to_compact();
            std::memcpy(sigs64_out + i * 64, compact.data(), 64);
            return UFSECP_OK;
        });

    if (SECP256K1_UNLIKELY(err != UFSECP_OK)) {
        std::memset(sigs64_out, 0, count * 64);
        return ctx_set_err(ctx, err,
            err == UFSECP_ERR_BAD_KEY ? "privkey[i] is zero or >= n"
                                       : "signing produced degenerate output");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_sign_batch(
    ufsecp_ctx* ctx,
    size_t count,
    const uint8_t* msgs32,
    const uint8_t* privkeys32,
    const uint8_t* aux_rands32,
    uint8_t* sigs64_out)
{
    // SEC-006: aux_rands32 is required — null pointer means no hedging, reject explicitly.
    if (SECP256K1_UNLIKELY(!ctx || !msgs32 || !privkeys32 || !sigs64_out || !aux_rands32)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (count == 0) return UFSECP_ERR_BAD_INPUT;
    if (count > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    std::size_t total_msg_bytes, total_sig_bytes;
    if (!checked_mul_size(count, std::size_t{32}, total_msg_bytes)
        || !checked_mul_size(count, std::size_t{64}, total_sig_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch size overflow");
    // SEC-008/BSG-12 Fail-closed: pre-zero output.  batch_parallel re-zeros on error.
    std::memset(sigs64_out, 0, count * 64);

    const ufsecp_error_t err = batch_parallel(count, sigs64_out,
        [&](size_t i) -> ufsecp_error_t {
            Scalar sk;
            if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkeys32 + i * 32, sk))) {
                secp256k1::detail::secure_erase(&sk, sizeof(sk));
                return UFSECP_ERR_BAD_KEY;
            }
            std::array<uint8_t, 32> msg_arr, aux_arr;
            std::memcpy(msg_arr.data(), msgs32 + i * 32, 32);
            std::memcpy(aux_arr.data(), aux_rands32 + i * 32, 32);
            auto kp  = secp256k1::ct::schnorr_keypair_create(sk);
            auto sig = secp256k1::ct::schnorr_sign(kp, msg_arr, aux_arr);
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
            if (SECP256K1_UNLIKELY(sig.s.is_zero())) return UFSECP_ERR_INTERNAL;
            const bool r_all_zero = std::all_of(sig.r.begin(), sig.r.end(),
                                                [](uint8_t b) { return b == 0; });
            if (SECP256K1_UNLIKELY(r_all_zero)) return UFSECP_ERR_INTERNAL;
            const auto sig_bytes = sig.to_bytes();
            std::memcpy(sigs64_out + i * 64, sig_bytes.data(), 64);
            return UFSECP_OK;
        });

    if (SECP256K1_UNLIKELY(err != UFSECP_OK)) {
        std::memset(sigs64_out, 0, count * 64);
        return ctx_set_err(ctx, err,
            err == UFSECP_ERR_BAD_KEY ? "privkey[i] is zero or >= n"
                                       : "signing produced degenerate output");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_verify(ufsecp_ctx* ctx,
                                     const uint8_t msg32[32],
                                     const uint8_t sig64[64],
                                     const uint8_t pubkey_x[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !sig64 || !pubkey_x)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    // BIP-340 strict parse: reject non-canonical r >= p, s >= n, or s == 0
    secp256k1::SchnorrSignature schnorr_sig;
    if (!secp256k1::SchnorrSignature::parse_strict(sig64, schnorr_sig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "Non-canonical Schnorr sig (r>=p or s>=n)");
    }

    const secp256k1::SchnorrXonlyPubkey* epk = s_schnorr_pk_cache.get(pubkey_x);
    if (!epk) epk = s_schnorr_pk_cache.put(pubkey_x);
    if (SECP256K1_UNLIKELY(!epk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "Non-canonical pubkey (x>=p)");
    }

    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);

    if (!secp256k1::schnorr_verify(*epk, msg_arr, schnorr_sig)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "Schnorr verify failed");
    }

    return UFSECP_OK;
}

/* ===========================================================================
 * ECDH
 * =========================================================================== */

static ufsecp_error_t ecdh_parse_args(ufsecp_ctx* ctx,
                                      const uint8_t privkey[32],
                                      const uint8_t pubkey33[33],
                                      Scalar& sk, Point& pk) {
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid or infinity pubkey");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdh(ufsecp_ctx* ctx,
                           const uint8_t privkey[32],
                           const uint8_t pubkey33[33],
                           uint8_t secret32_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !pubkey33 || !secret32_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk; Point pk;
    const ufsecp_error_t err = ecdh_parse_args(ctx, privkey, pubkey33, sk, pk);
    if (err != UFSECP_OK) return err;
    auto secret = secp256k1::ecdh_compute(sk, pk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    std::memcpy(secret32_out, secret.data(), 32);
    secp256k1::detail::secure_erase(secret.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdh_xonly(ufsecp_ctx* ctx,
                                 const uint8_t privkey[32],
                                 const uint8_t pubkey33[33],
                                 uint8_t secret32_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !pubkey33 || !secret32_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk; Point pk;
    const ufsecp_error_t err = ecdh_parse_args(ctx, privkey, pubkey33, sk, pk);
    if (err != UFSECP_OK) return err;
    auto secret = secp256k1::ecdh_compute_xonly(sk, pk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    std::memcpy(secret32_out, secret.data(), 32);
    secp256k1::detail::secure_erase(secret.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdh_raw(ufsecp_ctx* ctx,
                               const uint8_t privkey[32],
                               const uint8_t pubkey33[33],
                               uint8_t secret32_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !pubkey33 || !secret32_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk; Point pk;
    const ufsecp_error_t err = ecdh_parse_args(ctx, privkey, pubkey33, sk, pk);
    if (err != UFSECP_OK) return err;
    auto secret = secp256k1::ecdh_compute_raw(sk, pk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    std::memcpy(secret32_out, secret.data(), 32);
    secp256k1::detail::secure_erase(secret.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * Hashing (stateless -- no ctx required, but returns error_t for consistency)
 * =========================================================================== */
