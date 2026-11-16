/* ============================================================================
 * ufsecp C ABI -- Bitcoin Cash 2019 Schnorr verification
 * ============================================================================
 * NOT BIP-340: https://upgradespecs.bitcoincashnode.org/2019-05-15-schnorr/
 * Engine: src/cpu/src/bch_schnorr.cpp (secp256k1/bch_schnorr.hpp).
 *
 * Like every file in impl/, this is a fragment of ufsecp_impl.cpp: it is
 * compiled by `#include "impl/ufsecp_bch_schnorr.cpp"` there and uses that
 * translation unit's ctx_set_err / ctx_clear_err / UFSECP_CATCH_RETURN /
 * kMaxBatchN / checked_mul_size. It is not a standalone translation unit.
 *
 * Verification only; every input is public, so the engine is variable-time
 * by design (CT-VERIFY).
 *
 * Error semantics mirror the BIP-340 functions (ufsecp_schnorr_verify,
 * ufsecp_schnorr_batch_verify, ufsecp_schnorr_batch_identify_invalid):
 *   NULL argument                          -> UFSECP_ERR_NULL_ARG
 *   pubkey: prefix not 02/03, x >= p,
 *           or x not on the curve          -> UFSECP_ERR_BAD_PUBKEY
 *   sig: r >= p, s >= n, or s == 0         -> UFSECP_ERR_BAD_SIG
 *   well-formed but not valid              -> UFSECP_ERR_VERIFY_FAIL
 *   batch: n == 0 -> OK; n > kMaxBatchN -> UFSECP_ERR_BAD_INPUT
 * ============================================================================ */

#include "secp256k1/bch_schnorr.hpp"

namespace {

constexpr std::size_t kBchBatchRowBytes = 33 + 32 + 64;  // pubkey33 | msg32 | sig64

// Structural signature gate shared by the three entry points: r < p, 0 < s < n.
bool bch_sig_wellformed(const uint8_t* sig64) noexcept {
    FE r_fe;
    if (!FE::parse_bytes_strict(sig64, r_fe)) return false;
    Scalar s;
    return Scalar::parse_bytes_strict(sig64 + 32, s) && !s.is_zero();
}

// Parses wire row i into out; on a structural error returns the ufsecp error
// code (and sets the context message), else UFSECP_OK.
ufsecp_error_t bch_parse_row(ufsecp_ctx* ctx, const uint8_t* row,
                             secp256k1::BchSchnorrBatchPointEntry& out) {
    if (!secp256k1::bch_schnorr_pubkey_parse(row, out.point)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY,
                           "invalid compressed pubkey in BCH Schnorr batch");
    }
    std::memcpy(out.pubkey33.data(), row, 33);
    std::memcpy(out.message.data(), row + 33, 32);
    if (!bch_sig_wellformed(row + 65)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG,
                           "non-canonical BCH Schnorr sig (r>=p, s>=n or s==0) in batch");
    }
    std::memcpy(out.signature.data(), row + 65, 64);
    return UFSECP_OK;
}

} // namespace

ufsecp_error_t ufsecp_bch_schnorr_verify(ufsecp_ctx* ctx,
                                         const uint8_t msg32[32],
                                         const uint8_t sig64[64],
                                         const uint8_t pubkey33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !sig64 || !pubkey33)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    try {
    Point P;
    if (!secp256k1::bch_schnorr_pubkey_parse(pubkey33, P)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid compressed pubkey");
    }
    if (!bch_sig_wellformed(sig64)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG,
                           "non-canonical BCH Schnorr sig (r>=p, s>=n or s==0)");
    }
    if (!secp256k1::bch_schnorr_verify(sig64, msg32, P, pubkey33)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "BCH Schnorr verify failed");
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_bch_schnorr_batch_verify(ufsecp_ctx* ctx,
                                               const uint8_t* entries, size_t n) {
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;  /* empty batch is vacuously valid; entries ptr irrelevant */
    if (SECP256K1_UNLIKELY(!entries)) return UFSECP_ERR_NULL_ARG;
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    std::size_t total_bytes = 0;
    if (!checked_mul_size(n, kBchBatchRowBytes, total_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch size overflow");
    try {
    std::vector<secp256k1::BchSchnorrBatchPointEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        ufsecp_error_t const rc = bch_parse_row(ctx, entries + i * kBchBatchRowBytes, batch[i]);
        if (rc != UFSECP_OK) return rc;
    }
    if (!secp256k1::bch_schnorr_batch_verify(batch.data(), n)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "BCH Schnorr batch verify failed");
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_bch_schnorr_batch_identify(ufsecp_ctx* ctx,
                                                 const uint8_t* entries, size_t n,
                                                 uint8_t* results_out) {
    if (SECP256K1_UNLIKELY(!ctx)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;  /* nothing to report; pointers irrelevant */
    if (SECP256K1_UNLIKELY(!entries || !results_out)) return UFSECP_ERR_NULL_ARG;
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    std::size_t total_bytes = 0;
    if (!checked_mul_size(n, kBchBatchRowBytes, total_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch size overflow");
    /* Fail closed: no row reads as valid unless the verifier said so. */
    std::memset(results_out, 0, n);
    try {
    std::vector<secp256k1::BchSchnorrBatchPointEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        ufsecp_error_t const rc = bch_parse_row(ctx, entries + i * kBchBatchRowBytes, batch[i]);
        if (rc != UFSECP_OK) return rc;
    }
    (void)secp256k1::bch_schnorr_batch_verify_identify(batch.data(), n, results_out);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}
