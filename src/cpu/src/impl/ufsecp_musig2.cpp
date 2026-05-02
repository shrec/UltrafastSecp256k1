/* ============================================================================
 * MuSig2 (BIP-327) and FROST threshold signatures
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

ufsecp_error_t ufsecp_musig2_key_agg(ufsecp_ctx* ctx,
                                     const uint8_t* pubkeys, size_t n,
                                     uint8_t keyagg_out[UFSECP_MUSIG2_KEYAGG_LEN],
                                     uint8_t agg_pubkey32_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkeys || !keyagg_out || !agg_pubkey32_out)) return UFSECP_ERR_NULL_ARG;
    std::memset(keyagg_out, 0, UFSECP_MUSIG2_KEYAGG_LEN);
    std::memset(agg_pubkey32_out, 0, 32);
    if (n < 2) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "need >= 2 pubkeys");
    if (n > kMuSig2MaxKeyAggParticipants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "too many pubkeys for keyagg blob");
    }
    ctx_clear_err(ctx);
    try {
    // BIP-327: pubkeys are 33-byte compressed (02/03 prefix preserves Y parity)
    std::vector<std::array<uint8_t, 33>> pks(n);
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(pks[i].data(), pubkeys + i * 33, 33);
    }
    auto kagg = secp256k1::musig2_key_agg(pks);
    std::memcpy(agg_pubkey32_out, kagg.Q_x.data(), 32);
    /* Serialize key agg ctx: n(4) | Q_negated(1) | Q_compressed(33) | coefficients(n*32) */
    std::memset(keyagg_out, 0, UFSECP_MUSIG2_KEYAGG_LEN);
    const auto nk = static_cast<uint32_t>(kagg.key_coefficients.size());
    std::memcpy(keyagg_out, &nk, 4);
    keyagg_out[4] = kagg.Q_negated ? 1 : 0;
    point_to_compressed(kagg.Q, keyagg_out + 5);
    for (uint32_t i = 0; i < nk && (38u + (i+1)*32u <= UFSECP_MUSIG2_KEYAGG_LEN); ++i) {
        scalar_to_bytes(kagg.key_coefficients[i], keyagg_out + 38 + static_cast<size_t>(i) * 32);
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_musig2_nonce_gen(ufsecp_ctx* ctx,
                                       const uint8_t privkey[32],
                                       const uint8_t pubkey32[32],
                                       const uint8_t agg_pubkey32[32],
                                       const uint8_t msg32[32],
                                       const uint8_t extra_in[32],
                                       uint8_t secnonce_out[UFSECP_MUSIG2_SECNONCE_LEN],
                                       uint8_t pubnonce_out[UFSECP_MUSIG2_PUBNONCE_LEN]) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !pubkey32 || !agg_pubkey32 || !msg32 ||
        !secnonce_out || !pubnonce_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    std::array<uint8_t, 32> pk_arr, agg_arr, msg_arr;
    std::memcpy(pk_arr.data(), pubkey32, 32);
    std::memcpy(agg_arr.data(), agg_pubkey32, 32);
    std::memcpy(msg_arr.data(), msg32, 32);
    auto [sec, pub] = secp256k1::musig2_nonce_gen(sk, pk_arr, agg_arr, msg_arr, extra_in);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    /* Secret nonce: k1 || k2 */
    auto k1_bytes = sec.k1.to_bytes();
    auto k2_bytes = sec.k2.to_bytes();
    std::memcpy(secnonce_out, k1_bytes.data(), 32);
    std::memcpy(secnonce_out + 32, k2_bytes.data(), 32);
    secp256k1::detail::secure_erase(k1_bytes.data(), k1_bytes.size());
    secp256k1::detail::secure_erase(k2_bytes.data(), k2_bytes.size());
    /* Public nonce: R1(33) || R2(33) */
    auto pn = pub.serialize();
    std::memcpy(pubnonce_out, pn.data(), 66);
    secp256k1::detail::secure_erase(&sec, sizeof(sec));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_nonce_agg(ufsecp_ctx* ctx,
                                       const uint8_t* pubnonces, size_t n,
                                       uint8_t aggnonce_out[UFSECP_MUSIG2_AGGNONCE_LEN]) {
    if (SECP256K1_UNLIKELY(!ctx || !pubnonces || !aggnonce_out)) return UFSECP_ERR_NULL_ARG;
    std::memset(aggnonce_out, 0, UFSECP_MUSIG2_AGGNONCE_LEN);
    if (n < 2) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "need >= 2 nonces");
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "nonce count too large");
    ctx_clear_err(ctx);
    std::size_t total_bytes = 0;
    if (!checked_mul_size(n, std::size_t{66}, total_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "nonce array size overflow");
    try {
    std::vector<secp256k1::MuSig2PubNonce> pns(n);
    for (size_t i = 0; i < n; ++i) {
        if (point_from_compressed(pubnonces + i * 66).is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid pubnonce R1");
        }
        if (point_from_compressed(pubnonces + i * 66 + 33).is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid pubnonce R2");
        }
        std::array<uint8_t, 66> buf;
        std::memcpy(buf.data(), pubnonces + i * 66, 66);
        pns[i] = secp256k1::MuSig2PubNonce::deserialize(buf);
    }
    auto agg = secp256k1::musig2_nonce_agg(pns);
    /* Serialize: R1(33) || R2(33) */
    auto r1 = agg.R1.to_compressed();
    auto r2 = agg.R2.to_compressed();
    std::memcpy(aggnonce_out, r1.data(), 33);
    std::memcpy(aggnonce_out + 33, r2.data(), 33);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_musig2_start_sign_session(
    ufsecp_ctx* ctx,
    const uint8_t aggnonce[UFSECP_MUSIG2_AGGNONCE_LEN],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t msg32[32],
    uint8_t session_out[UFSECP_MUSIG2_SESSION_LEN]) {
    if (SECP256K1_UNLIKELY(!ctx || !aggnonce || !keyagg || !msg32 || !session_out)) return UFSECP_ERR_NULL_ARG;
    std::memset(session_out, 0, UFSECP_MUSIG2_SESSION_LEN);
    ctx_clear_err(ctx);
    /* Deserialize agg nonce */
    secp256k1::MuSig2AggNonce an;
    an.R1 = point_from_compressed(aggnonce);
    if (an.R1.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid agg nonce R1");
    }
    an.R2 = point_from_compressed(aggnonce + 33);
    if (an.R2.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid agg nonce R2");
    }
    /* Deserialize key agg context */
    secp256k1::MuSig2KeyAggCtx kagg;
    {
        const ufsecp_error_t rc = parse_musig2_keyagg(ctx, keyagg, kagg);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    auto sess = secp256k1::musig2_start_sign_session(an, kagg, msg_arr);
    /* Serialize session: R(33) | b(32) | e(32) | R_negated(1) = 98 bytes */
    std::memset(session_out, 0, UFSECP_MUSIG2_SESSION_LEN);
    point_to_compressed(sess.R, session_out);
    scalar_to_bytes(sess.b, session_out + 33);
    scalar_to_bytes(sess.e, session_out + 65);
    session_out[97] = sess.R_negated ? 1 : 0;
    const uint32_t participant_count = static_cast<uint32_t>(kagg.key_coefficients.size());
    std::memcpy(session_out + kMuSig2SessionCountOffset, &participant_count, sizeof(participant_count));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_partial_sign(
    ufsecp_ctx* ctx,
    uint8_t secnonce[UFSECP_MUSIG2_SECNONCE_LEN],
    const uint8_t privkey[32],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    size_t signer_index,
    uint8_t partial_sig32_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !secnonce || !privkey || !keyagg || !session || !partial_sig32_out)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    secp256k1::MuSig2SecNonce sn;
    Scalar k1, k2;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(secnonce, k1))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid secnonce k1");
    }
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(secnonce + 32, k2))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid secnonce k2");
    }
    sn.k1 = k1;
    sn.k2 = k2;
    secp256k1::MuSig2KeyAggCtx kagg;
    {
        const ufsecp_error_t rc = parse_musig2_keyagg(ctx, keyagg, kagg);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (signer_index >= kagg.key_coefficients.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "signer_index out of range");
    }
    // SECURITY NOTE (MED-3): signer_index is caller-supplied and cannot be cross-validated
    // against privkey — the keyagg blob stores only aggregation coefficients, not individual
    // public keys. A mismatched signer_index produces a partial signature that fails at
    // aggregation (DoS only, not key extraction). Full fix requires bumping
    // UFSECP_MUSIG2_KEYAGG_LEN to include per-signer compressed pubkeys (ABI-breaking,
    // scheduled for v2 — tracked in CHANGELOG as MED-3).
    secp256k1::MuSig2Session sess;
    uint32_t session_participant_count = 0;
    {
        const ufsecp_error_t rc = parse_musig2_session(ctx, session, sess, session_participant_count);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (session_participant_count != kagg.key_coefficients.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "session participant count does not match keyagg");
    }
    auto psig = secp256k1::musig2_partial_sign(sn, sk, kagg, sess, signer_index);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&sn, sizeof(sn));
    // Consume caller's secnonce to prevent catastrophic nonce reuse
    secp256k1::detail::secure_erase(secnonce, UFSECP_MUSIG2_SECNONCE_LEN);
    scalar_to_bytes(psig, partial_sig32_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_partial_verify(
    ufsecp_ctx* ctx,
    const uint8_t partial_sig32[32],
    const uint8_t pubnonce[UFSECP_MUSIG2_PUBNONCE_LEN],
    const uint8_t pubkey32[32],
    const uint8_t keyagg[UFSECP_MUSIG2_KEYAGG_LEN],
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    size_t signer_index) {
    if (SECP256K1_UNLIKELY(!ctx || !partial_sig32 || !pubnonce || !pubkey32 || !keyagg || !session)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar psig;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict(partial_sig32, psig))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "partial sig >= n");
    }
    std::array<uint8_t, 66> pn_buf;
    std::memcpy(pn_buf.data(), pubnonce, 66);
    auto pn = secp256k1::MuSig2PubNonce::deserialize(pn_buf);
    std::array<uint8_t, 32> pk_arr;
    std::memcpy(pk_arr.data(), pubkey32, 32);
    secp256k1::MuSig2KeyAggCtx kagg;
    {
        const ufsecp_error_t rc = parse_musig2_keyagg(ctx, keyagg, kagg);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (signer_index >= kagg.key_coefficients.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "signer_index out of range");
    }
    secp256k1::MuSig2Session sess;
    uint32_t session_participant_count = 0;
    {
        const ufsecp_error_t rc = parse_musig2_session(ctx, session, sess, session_participant_count);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (session_participant_count != kagg.key_coefficients.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "session participant count does not match keyagg");
    }
    if (!secp256k1::musig2_partial_verify(psig, pn, pk_arr, kagg, sess, signer_index)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "partial sig verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_musig2_partial_sig_agg(
    ufsecp_ctx* ctx,
    const uint8_t* partial_sigs, size_t n,
    const uint8_t session[UFSECP_MUSIG2_SESSION_LEN],
    uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !partial_sigs || !session || !sig64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (n == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial_sigs must be non-empty");
    }
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial sig count too large");
    std::size_t total_bytes = 0;
    if (!checked_mul_size(n, std::size_t{32}, total_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial sig size overflow");
    try {
    std::vector<Scalar> psigs(n);
    for (size_t i = 0; i < n; ++i) {
        if (SECP256K1_UNLIKELY(!scalar_parse_strict(partial_sigs + i * 32, psigs[i]))) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "partial sig >= n");
        }
    }
    secp256k1::MuSig2Session sess;
    uint32_t session_participant_count = 0;
    {
        const ufsecp_error_t rc = parse_musig2_session(ctx, session, sess, session_participant_count);
        if (rc != UFSECP_OK) {
            return rc;
        }
    }
    if (n != session_participant_count) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial_sigs count does not match session participant count");
    }
    auto final_sig = secp256k1::musig2_partial_sig_agg(psigs, sess);
    std::memcpy(sig64_out, final_sig.data(), 64);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * FROST (threshold signatures)
 * =========================================================================== */

ufsecp_error_t ufsecp_frost_keygen_begin(
    ufsecp_ctx* ctx,
    uint32_t participant_id, uint32_t threshold, uint32_t num_participants,
    const uint8_t seed[32],
    uint8_t* commits_out, size_t* commits_len,
    uint8_t* shares_out, size_t* shares_len) {
    if (SECP256K1_UNLIKELY(!ctx || !seed || !commits_out || !commits_len || !shares_out || !shares_len)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (threshold < 2 || threshold > num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid threshold");
    }
    if (participant_id == 0 || participant_id > num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid participant_id");
    }
    if (num_participants > kMaxBatchN) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "num_participants too large");
    }
    try {
    std::size_t required_commit_coeff_bytes = 0;
    std::size_t required_commits = 0;
    std::size_t required_shares = 0;
    if (!checked_mul_size(static_cast<std::size_t>(threshold), static_cast<std::size_t>(33), required_commit_coeff_bytes)
        || !checked_add_size(static_cast<std::size_t>(8), required_commit_coeff_bytes, required_commits)
        || !checked_mul_size(static_cast<std::size_t>(num_participants), static_cast<std::size_t>(UFSECP_FROST_SHARE_LEN), required_shares)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "FROST cardinality too large");
    }
    if (*commits_len < required_commits) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "commits buffer too small");
    }
    if (*shares_len < required_shares) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "shares buffer too small");
    }
    std::array<uint8_t, 32> seed_arr;
    std::memcpy(seed_arr.data(), seed, 32);
    auto [commit, shares] = secp256k1::frost_keygen_begin(
        participant_id, threshold, num_participants, seed_arr);
    secp256k1::detail::secure_erase(seed_arr.data(), 32);
    auto erase_shares = [&]() {
        for (auto& share : shares) {
            secp256k1::detail::secure_erase(&share.value, sizeof(share.value));
        }
    };
    /* Serialize commitment: coeff count(4) + from(4) + coeffs(33 each) */
    const size_t coeff_count = commit.coeffs.size();
    const size_t needed_commits = 8 + coeff_count * 33;
    if (*commits_len < needed_commits) {
        erase_shares();
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "commits buffer too small");
    }
    const auto cc32 = static_cast<uint32_t>(coeff_count);
    std::memcpy(commits_out, &cc32, 4);
    std::memcpy(commits_out + 4, &commit.from, 4);
    for (size_t i = 0; i < coeff_count; ++i) {
        point_to_compressed(commit.coeffs[i], commits_out + 8 + i * 33);

    }
    *commits_len = 8 + coeff_count * 33;
    /* Serialize shares */
    const size_t needed_shares = shares.size() * UFSECP_FROST_SHARE_LEN;
    if (*shares_len < needed_shares) {
        erase_shares();
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "shares buffer too small");
    }
    for (size_t i = 0; i < shares.size(); ++i) {
        uint8_t* s = shares_out + i * UFSECP_FROST_SHARE_LEN;
        std::memcpy(s, &shares[i].from, 4);
        scalar_to_bytes(shares[i].value, s + 4);
    }
    *shares_len = needed_shares;
    erase_shares();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_frost_keygen_finalize(
    ufsecp_ctx* ctx,
    uint32_t participant_id,
    const uint8_t* all_commits, size_t commits_len,
    const uint8_t* received_shares, size_t shares_len,
    uint32_t threshold, uint32_t num_participants,
    uint8_t keypkg_out[UFSECP_FROST_KEYPKG_LEN]) {
    if (SECP256K1_UNLIKELY(!ctx || !all_commits || !received_shares || !keypkg_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (threshold < 2 || threshold > num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid threshold");
    }
    if (participant_id == 0 || participant_id > num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid participant_id");
    }
    if (num_participants > kMaxBatchN) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "num_participants too large");
    }
    try {
    std::size_t expected_commit_coeff_bytes = 0;
    std::size_t expected_commit_record_len = 0;
    std::size_t expected_commits_len = 0;
    std::size_t expected_shares_len = 0;
    if (!checked_mul_size(static_cast<std::size_t>(threshold), static_cast<std::size_t>(33), expected_commit_coeff_bytes)
        || !checked_add_size(static_cast<std::size_t>(8), expected_commit_coeff_bytes, expected_commit_record_len)
        || !checked_mul_size(static_cast<std::size_t>(num_participants), expected_commit_record_len, expected_commits_len)
        || !checked_mul_size(static_cast<std::size_t>(num_participants), static_cast<std::size_t>(UFSECP_FROST_SHARE_LEN), expected_shares_len)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "FROST cardinality too large");
    }
    if (commits_len != expected_commits_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "all_commits length does not match threshold and num_participants");
    }
    if (shares_len != expected_shares_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "received_shares length does not match num_participants");
    }
    /* Deserialize commitments */
    std::vector<secp256k1::FrostCommitment> commits;
    std::vector<uint8_t> seen_commit_from(static_cast<size_t>(num_participants) + 1, 0);
    size_t pos = 0;
    while (pos < commits_len) {
        secp256k1::FrostCommitment fc;
        uint32_t cc = 0;
        if (pos + 8 > commits_len) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "truncated commit header");
        }
        std::memcpy(&cc, all_commits + pos, 4); pos += 4;
        std::memcpy(&fc.from, all_commits + pos, 4); pos += 4;
        if (cc != threshold) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment coefficient count");
        }
        if (fc.from == 0 || fc.from > num_participants) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment sender");
        }
        if (seen_commit_from[fc.from] != 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate commitment sender");
        }
        seen_commit_from[fc.from] = 1;
        if (pos + static_cast<size_t>(cc) * 33 > commits_len) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "truncated commit coefficients");
        }
        for (uint32_t j = 0; j < cc; ++j) {
            auto pt = point_from_compressed(all_commits + pos);
            if (pt.is_infinity()) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment coefficient");
            }
            fc.coeffs.push_back(pt);
            pos += 33;
        }
        commits.push_back(std::move(fc));
    }
    if (commits.size() != num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment count");
    }
    /* Deserialize shares */
    if (shares_len == 0 || (shares_len % UFSECP_FROST_SHARE_LEN) != 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid share blob length");
    }
    const size_t n_shares = shares_len / UFSECP_FROST_SHARE_LEN;
    if (n_shares != num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid share count");
    }
    std::vector<secp256k1::FrostShare> shares(n_shares);
    auto erase_shares = [&]() {
        for (auto& share : shares) {
            secp256k1::detail::secure_erase(&share.value, sizeof(share.value));
        }
    };
    std::vector<uint8_t> seen_share_from(static_cast<size_t>(num_participants) + 1, 0);
    for (size_t i = 0; i < n_shares; ++i) {
        const uint8_t* s = received_shares + i * UFSECP_FROST_SHARE_LEN;
        std::memcpy(&shares[i].from, s, 4);
        if (shares[i].from == 0 || shares[i].from > num_participants) {
            erase_shares();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid share sender");
        }
        if (seen_share_from[shares[i].from] != 0) {
            erase_shares();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate share sender");
        }
        seen_share_from[shares[i].from] = 1;
        shares[i].id = participant_id;
        Scalar v;
        if (SECP256K1_UNLIKELY(!scalar_parse_strict(s + 4, v))) {
            erase_shares();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid share scalar");
        }
        shares[i].value = v;
    }
    auto [kp, ok] = secp256k1::frost_keygen_finalize(
        participant_id, commits, shares, threshold, num_participants);
    if (SECP256K1_UNLIKELY(!ok)) {
        erase_shares();
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "FROST keygen finalize failed");
    }
    erase_shares();
    /* Serialize FrostKeyPackage: id(4) | threshold(4) | num_participants(4) |
       signing_share(32) | verification_share(33) | group_public_key(33) = 110 bytes */
    std::memset(keypkg_out, 0, UFSECP_FROST_KEYPKG_LEN);
    std::memcpy(keypkg_out, &kp.id, 4);
    std::memcpy(keypkg_out + 4, &kp.threshold, 4);
    std::memcpy(keypkg_out + 8, &kp.num_participants, 4);
    scalar_to_bytes(kp.signing_share, keypkg_out + 12);
    point_to_compressed(kp.verification_share, keypkg_out + 44);
    point_to_compressed(kp.group_public_key, keypkg_out + 77);
    secp256k1::detail::secure_erase(&kp.signing_share, sizeof(kp.signing_share));
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_frost_sign_nonce_gen(
    ufsecp_ctx* ctx,
    uint32_t participant_id,
    const uint8_t nonce_seed[32],
    uint8_t nonce_out[UFSECP_FROST_NONCE_LEN],
    uint8_t nonce_commit_out[UFSECP_FROST_NONCE_COMMIT_LEN]) {
    if (SECP256K1_UNLIKELY(!ctx || !nonce_seed || !nonce_out || !nonce_commit_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (participant_id == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid participant_id");
    }
    std::array<uint8_t, 32> seed_arr;
    std::memcpy(seed_arr.data(), nonce_seed, 32);
    auto [nonce, commit] = secp256k1::frost_sign_nonce_gen(participant_id, seed_arr);
    auto h_bytes = nonce.hiding_nonce.to_bytes();
    auto b_bytes = nonce.binding_nonce.to_bytes();
    std::memcpy(nonce_out, h_bytes.data(), 32);
    std::memcpy(nonce_out + 32, b_bytes.data(), 32);
    secp256k1::detail::secure_erase(seed_arr.data(), 32);
    secp256k1::detail::secure_erase(&nonce.hiding_nonce, sizeof(nonce.hiding_nonce));
    secp256k1::detail::secure_erase(&nonce.binding_nonce, sizeof(nonce.binding_nonce));
    secp256k1::detail::secure_erase(h_bytes.data(), 32);
    secp256k1::detail::secure_erase(b_bytes.data(), 32);
    std::memcpy(nonce_commit_out, &commit.id, 4);
    auto hp = commit.hiding_point.to_compressed();
    auto bp = commit.binding_point.to_compressed();
    std::memcpy(nonce_commit_out + 4, hp.data(), 33);
    std::memcpy(nonce_commit_out + 37, bp.data(), 33);
    return UFSECP_OK;
}

/// @brief Sign a FROST round-2 partial signature.
///
/// Bridges the stable C ABI to the internal FROST signing protocol:
///   - Validates signer-count and key-package invariants.
///   - Decodes the serialised nonce-commit list into typed structures.
///   - Returns the participant-id-prefixed partial signature.
///
/// @param ctx          Library context (must not be null).
/// @param keypkg       Serialised FROST key package (UFSECP_FROST_KEYPKG_LEN bytes).
/// @param nonce        Signing nonce generated in round 1 (UFSECP_FROST_NONCE_LEN bytes).
/// @param msg32        32-byte message hash to sign.
/// @param nonce_commits Array of n_signers serialised nonce commitments.
/// @param n_signers    Number of participants in this signing round.
/// @param partial_sig_out Output buffer for the 36-byte partial signature.
/// @return UFSECP_OK on success, an error code otherwise.
ufsecp_error_t ufsecp_frost_sign(
    ufsecp_ctx* ctx,
    const uint8_t keypkg[UFSECP_FROST_KEYPKG_LEN],
    uint8_t nonce[UFSECP_FROST_NONCE_LEN],
    const uint8_t msg32[32],
    const uint8_t* nonce_commits, size_t n_signers,
    uint8_t partial_sig_out[36]) {
    if (SECP256K1_UNLIKELY(!ctx || !keypkg || !nonce || !msg32 || !nonce_commits || !partial_sig_out)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (n_signers == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n_signers must be non-zero");
    }
    if (n_signers > kMaxBatchN) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n_signers too large");
    }
    std::size_t nc_total = 0;
    if (!checked_mul_size(n_signers, std::size_t{UFSECP_FROST_NONCE_COMMIT_LEN}, nc_total))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "nonce commit size overflow");
    try {
    secp256k1::FrostKeyPackage kp;
    std::memcpy(&kp.id, keypkg, 4);
    std::memcpy(&kp.threshold, keypkg + 4, 4);
    std::memcpy(&kp.num_participants, keypkg + 8, 4);
    if (kp.num_participants == 0 || kp.id == 0 || kp.id > kp.num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid key package participant metadata");
    }
    if (kp.threshold < 2 || kp.threshold > kp.num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid key package threshold");
    }
    if (n_signers > kp.num_participants) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid signer count");
    }
    if (SECP256K1_UNLIKELY(!scalar_parse_strict(keypkg + 12, kp.signing_share))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid signing share in keypkg");
    }
    kp.verification_share = point_from_compressed(keypkg + 44);
    if (kp.verification_share.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid verification share");
    }
    kp.group_public_key = point_from_compressed(keypkg + 77);
    if (kp.group_public_key.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid group public key");
    }
    secp256k1::FrostNonce fn;
    Scalar h, b;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(nonce, h))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hiding nonce");
    }
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(nonce + 32, b))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid binding nonce");
    }
    fn.hiding_nonce = h;
    fn.binding_nonce = b;
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::vector<secp256k1::FrostNonceCommitment> ncs(n_signers);
    size_t self_commitment_count = 0;
    ufsecp_error_t nc_err = UFSECP_OK;
    for (size_t i = 0; i < n_signers; ++i) {
        const uint8_t* nc = nonce_commits + i * UFSECP_FROST_NONCE_COMMIT_LEN;
        std::memcpy(&ncs[i].id, nc, 4);
        if (ncs[i].id == 0 || ncs[i].id > kp.num_participants) {
            nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid nonce commitment signer");
            break;
        }
        for (size_t j = 0; j < i; ++j) {
            if (ncs[j].id == ncs[i].id) {
                nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate nonce commitment signer");
                break;
            }
        }
        if (nc_err != UFSECP_OK) break;
        if (ncs[i].id == kp.id) {
            ++self_commitment_count;
        }
        ncs[i].hiding_point = point_from_compressed(nc + 4);
        if (ncs[i].hiding_point.is_infinity()) {
            nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hiding nonce point");
            break;
        }
        ncs[i].binding_point = point_from_compressed(nc + 37);
        if (ncs[i].binding_point.is_infinity()) {
            nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid binding nonce point");
            break;
        }
    }
    if (nc_err == UFSECP_OK && self_commitment_count != 1) {
        nc_err = ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "missing signer nonce commitment");
    }
    if (nc_err != UFSECP_OK) {
        secp256k1::detail::secure_erase(&kp.signing_share, sizeof(kp.signing_share));
        secp256k1::detail::secure_erase(&fn.hiding_nonce, sizeof(fn.hiding_nonce));
        secp256k1::detail::secure_erase(&fn.binding_nonce, sizeof(fn.binding_nonce));
        secp256k1::detail::secure_erase(&h, sizeof(h));
        secp256k1::detail::secure_erase(&b, sizeof(b));
        return nc_err;
    }
    auto psig = secp256k1::frost_sign(kp, fn, msg_arr, ncs);
    secp256k1::detail::secure_erase(&kp.signing_share, sizeof(kp.signing_share));
    secp256k1::detail::secure_erase(&fn.hiding_nonce, sizeof(fn.hiding_nonce));
    secp256k1::detail::secure_erase(&fn.binding_nonce, sizeof(fn.binding_nonce));
    secp256k1::detail::secure_erase(&h, sizeof(h));
    secp256k1::detail::secure_erase(&b, sizeof(b));
    // Consume caller's nonce to prevent catastrophic nonce reuse (mirrors MuSig2 secnonce erasure)
    secp256k1::detail::secure_erase(nonce, UFSECP_FROST_NONCE_LEN);
    std::memcpy(partial_sig_out, &psig.id, 4);
    scalar_to_bytes(psig.z_i, partial_sig_out + 4);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_frost_verify_partial(
    ufsecp_ctx* ctx,
    const uint8_t partial_sig[36],
    const uint8_t verification_share33[33],
    const uint8_t* nonce_commits, size_t n_signers,
    const uint8_t msg32[32],
    const uint8_t group_pubkey33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !partial_sig || !verification_share33 || !nonce_commits || !msg32 || !group_pubkey33)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (n_signers == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n_signers must be non-zero");
    }
    if (n_signers > kMaxBatchN) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n_signers too large");
    }
    std::size_t nc_total = 0;
    if (!checked_mul_size(n_signers, std::size_t{UFSECP_FROST_NONCE_COMMIT_LEN}, nc_total))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "nonce commit size overflow");
    try {
    secp256k1::FrostPartialSig psig;
    std::memcpy(&psig.id, partial_sig, 4);
    if (psig.id == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial_sig.id must be non-zero");
    }
    Scalar z;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict(partial_sig + 4, z))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid partial sig scalar");
    }
    psig.z_i = z;
    auto vs = point_from_compressed(verification_share33);
    if (vs.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid verification share");
    }
    std::vector<secp256k1::FrostNonceCommitment> ncs(n_signers);
    secp256k1::FrostNonceCommitment signer_commit{};
    size_t signer_matches = 0;
    for (size_t i = 0; i < n_signers; ++i) {
        const uint8_t* nc = nonce_commits + i * UFSECP_FROST_NONCE_COMMIT_LEN;
        std::memcpy(&ncs[i].id, nc, 4);
        if (ncs[i].id == 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "nonce commitment signer IDs must be non-zero");
        }
        for (size_t j = 0; j < i; ++j) {
            if (ncs[j].id == ncs[i].id) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate nonce commitment signer IDs");
            }
        }
        ncs[i].hiding_point = point_from_compressed(nc + 4);
        if (ncs[i].hiding_point.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hiding nonce point");
        }
        ncs[i].binding_point = point_from_compressed(nc + 37);
        if (ncs[i].binding_point.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid binding nonce point");
        }
        if (ncs[i].id == psig.id) {
            signer_commit = ncs[i];
            ++signer_matches;
        }
    }
    if (signer_matches != 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT,
            signer_matches == 0 ? "partial_sig.id not found in nonce_commits"
                                : "partial_sig.id must appear exactly once in nonce_commits");
    }
    auto gp = point_from_compressed(group_pubkey33);
    if (gp.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid group public key");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    const bool ok = secp256k1::frost_verify_partial(psig, signer_commit, vs, msg_arr, ncs, gp);
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "FROST partial signature verification failed");
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_frost_aggregate(
    ufsecp_ctx* ctx,
    const uint8_t* partial_sigs, size_t n,
    const uint8_t* nonce_commits, size_t n_signers,
    const uint8_t group_pubkey33[33],
    const uint8_t msg32[32],
    uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !partial_sigs || !nonce_commits || !group_pubkey33 || !msg32 || !sig64_out)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (n == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial_sigs must be non-empty");
    }
    if (n_signers == 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n_signers must be non-zero");
    }
    if (n != n_signers) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial/nonces signer count mismatch");
    }
    if (n > kMaxBatchN) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "signer count too large");
    }
    std::size_t psig_total = 0, nc_total = 0;
    if (!checked_mul_size(n, std::size_t{36}, psig_total)
        || !checked_mul_size(n_signers, std::size_t{UFSECP_FROST_NONCE_COMMIT_LEN}, nc_total))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "frost aggregate size overflow");
    try {
    std::vector<secp256k1::FrostPartialSig> psigs(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* ps = partial_sigs + i * 36;
        std::memcpy(&psigs[i].id, ps, 4);
        if (psigs[i].id == 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid partial sig signer");
        }
        for (size_t j = 0; j < i; ++j) {
            if (psigs[j].id == psigs[i].id) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate partial sig signer");
            }
        }
        Scalar z;
        if (SECP256K1_UNLIKELY(!scalar_parse_strict(ps + 4, z))) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid partial sig scalar");
        }
        psigs[i].z_i = z;
    }
    std::vector<secp256k1::FrostNonceCommitment> ncs(n_signers);
    for (size_t i = 0; i < n_signers; ++i) {
        const uint8_t* nc = nonce_commits + i * UFSECP_FROST_NONCE_COMMIT_LEN;
        std::memcpy(&ncs[i].id, nc, 4);
        if (ncs[i].id == 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid nonce commitment signer");
        }
        for (size_t j = 0; j < i; ++j) {
            if (ncs[j].id == ncs[i].id) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "duplicate nonce commitment signer");
            }
        }
        ncs[i].hiding_point = point_from_compressed(nc + 4);
        if (ncs[i].hiding_point.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hiding nonce point");
        }
        ncs[i].binding_point = point_from_compressed(nc + 37);
        if (ncs[i].binding_point.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid binding nonce point");
        }
    }
    for (const auto& psig : psigs) {
        bool found = false;
        for (const auto& nc : ncs) {
            if (nc.id == psig.id) {
                found = true;
                break;
            }
        }
        if (SECP256K1_UNLIKELY(!found)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "partial sig signer missing from nonce commitments");
        }
    }
    auto gp = point_from_compressed(group_pubkey33);
    if (gp.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid group public key");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    auto sig = secp256k1::frost_aggregate(psigs, ncs, gp, msg_arr);
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * Adaptor signatures
 * =========================================================================== */

