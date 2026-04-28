/* ============================================================================
 * ECDSA sign/verify/recover, Schnorr sign/verify, ECDH, batch signing
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

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
    // CT path returns already-normalized (low-S) signature
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
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid public key");
    }

    if (SECP256K1_UNLIKELY(!secp256k1::ecdsa_verify(msg, pk, ecdsasig))) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "ECDSA verify failed");
    }

    return UFSECP_OK;
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
    auto normalized = rsig.sig.normalize();
    auto compact = normalized.to_compact();
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
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
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
    if (count > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    std::size_t total_msg_bytes, total_sig_bytes;
    if (!checked_mul_size(count, std::size_t{32}, total_msg_bytes)
        || !checked_mul_size(count, std::size_t{64}, total_sig_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch size overflow");
    // Clear output buffer upfront: on early failure caller cannot tell which
    // indices are valid, so a zeroed buffer is the safest partial-failure state.
    std::memset(sigs64_out, 0, count * 64);
    for (size_t i = 0; i < count; ++i) {
        std::array<uint8_t, 32> msg;
        std::memcpy(msg.data(), msgs32 + i * 32, 32);
        Scalar sk;
        if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkeys32 + i * 32, sk))) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            std::memset(sigs64_out, 0, count * 64); // re-clear partial output (fail-closed)
            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY,
                               "privkey[i] is zero or >= n");
        }
        auto sig = secp256k1::ct::ecdsa_sign(msg, sk);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        auto compact = sig.to_compact();
        std::memcpy(sigs64_out + i * 64, compact.data(), 64);
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
    if (SECP256K1_UNLIKELY(!ctx || !msgs32 || !privkeys32 || !sigs64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (count > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    std::size_t total_msg_bytes, total_sig_bytes;
    if (!checked_mul_size(count, std::size_t{32}, total_msg_bytes)
        || !checked_mul_size(count, std::size_t{64}, total_sig_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch size overflow");

    static constexpr uint8_t kZeroAux[32] = {};
    // Fail-closed: clear output before signing so partial failure leaves no valid sigs visible
    std::memset(sigs64_out, 0, count * 64);

    for (size_t i = 0; i < count; ++i) {
        Scalar sk;
        if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkeys32 + i * 32, sk))) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            std::memset(sigs64_out, 0, count * 64); // re-clear partial output (fail-closed)
            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY,
                               "privkey[i] is zero or >= n");
        }

        std::array<uint8_t, 32> msg_arr, aux_arr;
        std::memcpy(msg_arr.data(), msgs32 + i * 32, 32);
        const uint8_t* aux_src = aux_rands32 ? aux_rands32 + i * 32 : kZeroAux;
        std::memcpy(aux_arr.data(), aux_src, 32);

        auto kp  = secp256k1::ct::schnorr_keypair_create(sk);
        auto sig = secp256k1::ct::schnorr_sign(kp, msg_arr, aux_arr);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));

        auto sig_bytes = sig.to_bytes();
        std::memcpy(sigs64_out + i * 64, sig_bytes.data(), 64);
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

    // BIP-340 strict: reject pubkey x >= p
    FE pk_fe;
    if (!FE::parse_bytes_strict(pubkey_x, pk_fe)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "Non-canonical pubkey (x>=p)");
    }

    std::array<uint8_t, 32> pk_arr, msg_arr;
    std::memcpy(pk_arr.data(), pubkey_x, 32);
    std::memcpy(msg_arr.data(), msg32, 32);

    if (!secp256k1::schnorr_verify(pk_arr, msg_arr, schnorr_sig)) {
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

