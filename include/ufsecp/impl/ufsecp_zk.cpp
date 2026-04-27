/* ============================================================================
 * Schnorr/ECDSA adaptors, Pedersen commitments, ZK knowledge/DLEQ/range proofs
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

ufsecp_error_t ufsecp_schnorr_adaptor_sign(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33],
    const uint8_t aux_rand[32],
    uint8_t pre_sig_out[UFSECP_SCHNORR_ADAPTOR_SIG_LEN]) {
    if (!ctx || !privkey || !msg32 || !adaptor_point33 || !aux_rand || !pre_sig_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    std::array<uint8_t, 32> msg_arr, aux_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::memcpy(aux_arr.data(), aux_rand, 32);
    auto ap = point_from_compressed(adaptor_point33);
    if (ap.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid adaptor point");
    }
    auto pre = secp256k1::schnorr_adaptor_sign(sk, msg_arr, ap, aux_arr);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto rhat = pre.R_hat.to_compressed();
    auto shat = pre.s_hat.to_bytes();
    std::memcpy(pre_sig_out, rhat.data(), 33);
    std::memcpy(pre_sig_out + 33, shat.data(), 32);
    /* Serialize needs_negation as a 32-byte flag for completeness */
    std::memset(pre_sig_out + 65, 0, 32);
    pre_sig_out[65] = pre.needs_negation ? 1 : 0;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_adaptor_verify(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t pubkey_x[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33]) {
    if (!ctx || !pre_sig || !pubkey_x || !msg32 || !adaptor_point33) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    secp256k1::SchnorrAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    as.needs_negation = (pre_sig[65] != 0);
    // Strict: reject x-only pubkey >= p at ABI gate
    FE pk_fe;
    if (!FE::parse_bytes_strict(pubkey_x, pk_fe)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "non-canonical pubkey (x>=p)");
    }
    std::array<uint8_t, 32> pk_arr, msg_arr;
    std::memcpy(pk_arr.data(), pubkey_x, 32);
    std::memcpy(msg_arr.data(), msg32, 32);
    auto ap = point_from_compressed(adaptor_point33);
    if (ap.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid adaptor point");
    }
    if (!secp256k1::schnorr_adaptor_verify(as, pk_arr, msg_arr, ap)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "adaptor verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_adaptor_adapt(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t adaptor_secret[32],
    uint8_t sig64_out[64]) {
    if (!ctx || !pre_sig || !adaptor_secret || !sig64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    secp256k1::SchnorrAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    as.needs_negation = (pre_sig[65] != 0);
    Scalar secret;
    if (!scalar_parse_strict_nonzero(adaptor_secret, secret)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "adaptor secret is zero or >= n");
    }
    auto sig = secp256k1::schnorr_adaptor_adapt(as, secret);
    secp256k1::detail::secure_erase(&secret, sizeof(secret));
    auto bytes = sig.to_bytes();
    std::memcpy(sig64_out, bytes.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_schnorr_adaptor_extract(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_SCHNORR_ADAPTOR_SIG_LEN],
    const uint8_t sig64[64],
    uint8_t secret32_out[32]) {
    if (!ctx || !pre_sig || !sig64 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    secp256k1::SchnorrAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    as.needs_negation = (pre_sig[65] != 0);
    secp256k1::SchnorrSignature sig;
    if (!secp256k1::SchnorrSignature::parse_strict(sig64, sig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid schnorr signature");
    }
    auto [secret, ok] = secp256k1::schnorr_adaptor_extract(as, sig);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "adaptor extract failed");
    }
    scalar_to_bytes(secret, secret32_out);
    secp256k1::detail::secure_erase(&secret, sizeof(secret));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_adaptor_sign(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33],
    uint8_t pre_sig_out[UFSECP_ECDSA_ADAPTOR_SIG_LEN]) {
    if (!ctx || !privkey || !msg32 || !adaptor_point33 || !pre_sig_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    auto ap = point_from_compressed(adaptor_point33);
    if (ap.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid adaptor point");
    }
    auto pre = secp256k1::ecdsa_adaptor_sign(sk, msg_arr, ap);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto rhat = pre.R_hat.to_compressed();
    auto shat = pre.s_hat.to_bytes();
    auto r_bytes = pre.r.to_bytes();
    std::memcpy(pre_sig_out, rhat.data(), 33);
    std::memcpy(pre_sig_out + 33, shat.data(), 32);
    std::memcpy(pre_sig_out + 65, r_bytes.data(), 32);
    /* zero-pad remainder */
    std::memset(pre_sig_out + 97, 0, UFSECP_ECDSA_ADAPTOR_SIG_LEN - 97);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_adaptor_verify(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32],
    const uint8_t adaptor_point33[33]) {
    if (!ctx || !pre_sig || !pubkey33 || !msg32 || !adaptor_point33) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    secp256k1::ECDSAAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    if (!scalar_parse_strict(pre_sig + 65, as.r)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig r");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    auto ap = point_from_compressed(adaptor_point33);
    if (ap.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid adaptor point");
    }
    if (!secp256k1::ecdsa_adaptor_verify(as, pk, msg_arr, ap)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "ECDSA adaptor verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_adaptor_adapt(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t adaptor_secret[32],
    uint8_t sig64_out[64]) {
    if (!ctx || !pre_sig || !adaptor_secret || !sig64_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    secp256k1::ECDSAAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    if (!scalar_parse_strict(pre_sig + 65, as.r)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig r");
    }
    Scalar secret;
    if (!scalar_parse_strict_nonzero(adaptor_secret, secret)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "adaptor secret is zero or >= n");
    }
    auto sig = secp256k1::ecdsa_adaptor_adapt(as, secret);
    secp256k1::detail::secure_erase(&secret, sizeof(secret));
    auto compact = sig.to_compact();
    std::memcpy(sig64_out, compact.data(), 64);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ecdsa_adaptor_extract(
    ufsecp_ctx* ctx,
    const uint8_t pre_sig[UFSECP_ECDSA_ADAPTOR_SIG_LEN],
    const uint8_t sig64[64],
    uint8_t secret32_out[32]) {
    if (!ctx || !pre_sig || !sig64 || !secret32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    secp256k1::ECDSAAdaptorSig as;
    as.R_hat = point_from_compressed(pre_sig);
    if (as.R_hat.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor R_hat");
    }
    Scalar shat;
    if (!scalar_parse_strict(pre_sig + 33, shat)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig scalar");
    }
    as.s_hat = shat;
    if (!scalar_parse_strict(pre_sig + 65, as.r)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid adaptor sig r");
    }
    std::array<uint8_t, 64> compact;
    std::memcpy(compact.data(), sig64, 64);
    secp256k1::ECDSASignature ecdsasig;
    if (!secp256k1::ECDSASignature::parse_compact_strict(compact, ecdsasig)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid ECDSA sig");
    }
    auto [secret, ok] = secp256k1::ecdsa_adaptor_extract(as, ecdsasig);
    if (!ok) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "ECDSA adaptor extract failed");
    }
    scalar_to_bytes(secret, secret32_out);
    secp256k1::detail::secure_erase(&secret, sizeof(secret));
    return UFSECP_OK;
}

/* ===========================================================================
 * Pedersen commitments
 * =========================================================================== */

ufsecp_error_t ufsecp_pedersen_commit(ufsecp_ctx* ctx,
                                      const uint8_t value[32],
                                      const uint8_t blinding[32],
                                      uint8_t commitment33_out[33]) {
    if (!ctx || !value || !blinding || !commitment33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar v, b;
    if (!scalar_parse_strict(value, v)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "value >= n");
    }
    if (!scalar_parse_strict(blinding, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blinding >= n");
    }
    auto c = secp256k1::pedersen_commit(v, b);
    auto comp = c.point.to_compressed();
    std::memcpy(commitment33_out, comp.data(), 33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pedersen_verify(ufsecp_ctx* ctx,
                                      const uint8_t commitment33[33],
                                      const uint8_t value[32],
                                      const uint8_t blinding[32]) {
    if (!ctx || !commitment33 || !value || !blinding) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar v, b;
    if (!scalar_parse_strict(value, v)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "value >= n");
    }
    if (!scalar_parse_strict(blinding, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blinding >= n");
    }
    auto commit_pt = point_from_compressed(commitment33);
    if (commit_pt.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment point");
    }
    if (!secp256k1::pedersen_verify(secp256k1::PedersenCommitment{commit_pt}, v, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "Pedersen verify failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pedersen_verify_sum(ufsecp_ctx* ctx,
                                          const uint8_t* pos, size_t n_pos,
                                          const uint8_t* neg, size_t n_neg) {
    if (!ctx || (!pos && n_pos > 0) || (!neg && n_neg > 0)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (n_pos > kMaxBatchN || n_neg > kMaxBatchN)
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "commitment count too large");
    std::size_t pos_bytes = 0, neg_bytes = 0;
    if (!checked_mul_size(n_pos, std::size_t{33}, pos_bytes)
        || !checked_mul_size(n_neg, std::size_t{33}, neg_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "commitment array size overflow");
    try {
    std::vector<secp256k1::PedersenCommitment> pcs(n_pos), ncs(n_neg);
    for (size_t i = 0; i < n_pos; ++i) {
        auto p = point_from_compressed(pos + i * 33);
        if (p.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid positive commitment");
        }
        pcs[i] = secp256k1::PedersenCommitment{p};
    }
    for (size_t i = 0; i < n_neg; ++i) {
        auto p = point_from_compressed(neg + i * 33);
        if (p.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid negative commitment");
        }
        ncs[i] = secp256k1::PedersenCommitment{p};
    }
    if (!secp256k1::pedersen_verify_sum(pcs.data(), n_pos, ncs.data(), n_neg)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "Pedersen sum verify failed");
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_pedersen_blind_sum(ufsecp_ctx* ctx,
                                         const uint8_t* blinds_in, size_t n_in,
                                         const uint8_t* blinds_out, size_t n_out,
                                         uint8_t sum32_out[32]) {
    if (!ctx || (!blinds_in && n_in > 0) || (!blinds_out && n_out > 0) || !sum32_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (n_in > kMaxBatchN || n_out > kMaxBatchN)
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blind count too large");
    std::size_t in_bytes = 0, out_bytes = 0;
    if (!checked_mul_size(n_in, std::size_t{32}, in_bytes)
        || !checked_mul_size(n_out, std::size_t{32}, out_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blind array size overflow");
    try {
    std::vector<Scalar> ins(n_in), outs(n_out);
    for (size_t i = 0; i < n_in; ++i) {
        if (!scalar_parse_strict(blinds_in + i * 32, ins[i])) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid input blind");
        }
    }
    for (size_t i = 0; i < n_out; ++i) {
        if (!scalar_parse_strict(blinds_out + i * 32, outs[i])) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid output blind");
        }
    }
    auto sum = secp256k1::pedersen_blind_sum(ins.data(), n_in, outs.data(), n_out);
    scalar_to_bytes(sum, sum32_out);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_pedersen_switch_commit(ufsecp_ctx* ctx,
                                             const uint8_t value[32],
                                             const uint8_t blinding[32],
                                             const uint8_t switch_blind[32],
                                             uint8_t commitment33_out[33]) {
    if (!ctx || !value || !blinding || !switch_blind || !commitment33_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar v, b, sb;
    if (!scalar_parse_strict(value, v)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "value >= n");
    }
    if (!scalar_parse_strict(blinding, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blinding >= n");
    }
    if (!scalar_parse_strict(switch_blind, sb)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "switch_blind >= n");
    }
    auto c = secp256k1::pedersen_switch_commit(v, b, sb);
    auto comp = c.point.to_compressed();
    std::memcpy(commitment33_out, comp.data(), 33);
    return UFSECP_OK;
}

/* ===========================================================================
 * Zero-knowledge proofs
 * =========================================================================== */

ufsecp_error_t ufsecp_zk_knowledge_prove(
    ufsecp_ctx* ctx,
    const uint8_t secret[32],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32],
    const uint8_t aux_rand[32],
    uint8_t proof_out[UFSECP_ZK_KNOWLEDGE_PROOF_LEN]) {
    if (!ctx || !secret || !pubkey33 || !msg32 || !aux_rand || !proof_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar s;
    if (!scalar_parse_strict_nonzero(secret, s)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "secret is zero or >= n");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    std::array<uint8_t, 32> msg_arr, aux_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    std::memcpy(aux_arr.data(), aux_rand, 32);
    auto proof = secp256k1::zk::knowledge_prove(s, pk, msg_arr, aux_arr);
    secp256k1::detail::secure_erase(&s, sizeof(s));
    auto ser = proof.serialize();
    std::memcpy(proof_out, ser.data(), UFSECP_ZK_KNOWLEDGE_PROOF_LEN);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_knowledge_verify(
    ufsecp_ctx* ctx,
    const uint8_t proof[UFSECP_ZK_KNOWLEDGE_PROOF_LEN],
    const uint8_t pubkey33[33],
    const uint8_t msg32[32]) {
    if (!ctx || !proof || !pubkey33 || !msg32) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    secp256k1::zk::KnowledgeProof kp;
    if (!secp256k1::zk::KnowledgeProof::deserialize(proof, kp)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid knowledge proof");
    }
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), msg32, 32);
    if (!secp256k1::zk::knowledge_verify(kp, pk, msg_arr)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "knowledge proof failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_dleq_prove(
    ufsecp_ctx* ctx,
    const uint8_t secret[32],
    const uint8_t G33[33], const uint8_t H33[33],
    const uint8_t P33[33], const uint8_t Q33[33],
    const uint8_t aux_rand[32],
    uint8_t proof_out[UFSECP_ZK_DLEQ_PROOF_LEN]) {
    if (!ctx || !secret || !G33 || !H33 || !P33 || !Q33 || !aux_rand || !proof_out) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar s;
    if (!scalar_parse_strict_nonzero(secret, s)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "secret is zero or >= n");
    }
    auto G = point_from_compressed(G33);
    auto H = point_from_compressed(H33);
    auto P = point_from_compressed(P33);
    auto Q = point_from_compressed(Q33);
    if (G.is_infinity() || H.is_infinity() || P.is_infinity() || Q.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid DLEQ point");
    }
    std::array<uint8_t, 32> aux_arr;
    std::memcpy(aux_arr.data(), aux_rand, 32);
    auto proof = secp256k1::zk::dleq_prove(s, G, H, P, Q, aux_arr);
    secp256k1::detail::secure_erase(&s, sizeof(s));
    auto ser = proof.serialize();
    std::memcpy(proof_out, ser.data(), UFSECP_ZK_DLEQ_PROOF_LEN);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_dleq_verify(
    ufsecp_ctx* ctx,
    const uint8_t proof[UFSECP_ZK_DLEQ_PROOF_LEN],
    const uint8_t G33[33], const uint8_t H33[33],
    const uint8_t P33[33], const uint8_t Q33[33]) {
    if (!ctx || !proof || !G33 || !H33 || !P33 || !Q33) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto G = point_from_compressed(G33);
    auto H = point_from_compressed(H33);
    auto P = point_from_compressed(P33);
    auto Q = point_from_compressed(Q33);
    if (G.is_infinity() || H.is_infinity() || P.is_infinity() || Q.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid DLEQ point");
    }
    secp256k1::zk::DLEQProof dp;
    if (!secp256k1::zk::DLEQProof::deserialize(proof, dp)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid DLEQ proof");
    }
    if (!secp256k1::zk::dleq_verify(dp, G, H, P, Q)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "DLEQ proof failed");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_range_prove(
    ufsecp_ctx* ctx,
    uint64_t value,
    const uint8_t blinding[32],
    const uint8_t commitment33[33],
    const uint8_t aux_rand[32],
    uint8_t* proof_out, size_t* proof_len) {
    if (!ctx || !blinding || !commitment33 || !aux_rand || !proof_out || !proof_len) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    Scalar b;
    if (!scalar_parse_strict(blinding, b)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "blinding >= n");
    }
    auto commit_pt = point_from_compressed(commitment33);
    if (commit_pt.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment point");
    }
    auto commit = secp256k1::PedersenCommitment{commit_pt};
    std::array<uint8_t, 32> aux_arr;
    std::memcpy(aux_arr.data(), aux_rand, 32);
    auto rp = secp256k1::zk::range_prove(value, b, commit, aux_arr);
    /* Serialize range proof: A(33)+S(33)+T1(33)+T2(33)+tau_x(32)+mu(32)+t_hat(32)+L[6]*33+R[6]*33+a(32)+b(32) */
    const size_t needed = 33*4 + 32*3 + 6*33 + 6*33 + 32*2;
    if (*proof_len < needed) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "range proof buffer too small");
    }
    size_t off = 0;
    auto write_point = [&](const Point& p) {
        auto c = p.to_compressed();
        std::memcpy(proof_out + off, c.data(), 33);
        off += 33;
    };
    auto write_scalar = [&](const Scalar& s) {
        scalar_to_bytes(s, proof_out + off);
        off += 32;
    };
    write_point(rp.A); write_point(rp.S);
    write_point(rp.T1); write_point(rp.T2);
    write_scalar(rp.tau_x); write_scalar(rp.mu); write_scalar(rp.t_hat);
    for (int i = 0; i < 6; ++i) write_point(rp.L[i]);
    for (int i = 0; i < 6; ++i) write_point(rp.R[i]);
    write_scalar(rp.a); write_scalar(rp.b);
    *proof_len = off;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_zk_range_verify(
    ufsecp_ctx* ctx,
    const uint8_t commitment33[33],
    const uint8_t* proof, size_t proof_len) {
    if (!ctx || !commitment33 || !proof) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    /* Deserialize range proof */
    const size_t expected = 33*4 + 32*3 + 6*33 + 6*33 + 32*2;
    if (proof_len < expected) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "range proof too short");
    }
    if (proof_len != expected) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "range proof length mismatch");
    }
    secp256k1::zk::RangeProof rp;
    size_t off = 0;
    bool point_ok = true;
    auto read_point = [&]() -> Point {
        auto p = point_from_compressed(proof + off);
        if (p.is_infinity()) point_ok = false;
        off += 33;
        return p;
    };
    bool scalar_ok = true;
    auto read_scalar = [&]() -> Scalar {
        Scalar s;
        if (!scalar_parse_strict(proof + off, s)) {
            scalar_ok = false;
        }
        off += 32;
        return s;
    };
    rp.A = read_point(); rp.S = read_point();
    rp.T1 = read_point(); rp.T2 = read_point();
    rp.tau_x = read_scalar(); rp.mu = read_scalar(); rp.t_hat = read_scalar();
    for (int i = 0; i < 6; ++i) rp.L[i] = read_point();
    for (int i = 0; i < 6; ++i) rp.R[i] = read_point();
    rp.a = read_scalar(); rp.b = read_scalar();
    if (!point_ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid point in range proof");
    }
    if (!scalar_ok) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid scalar in range proof");
    }
    auto commit_pt = point_from_compressed(commitment33);
    if (commit_pt.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid commitment point");
    }
    auto commit = secp256k1::PedersenCommitment{commit_pt};
    if (!secp256k1::zk::range_verify(commit, rp)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "range proof failed");
    }
    return UFSECP_OK;
}

/* ---------------------------------------------------------------------------
 * ECDSA foreign-field SNARK witness (eprint 2025/695)
 * ------------------------------------------------------------------------- */

ufsecp_error_t ufsecp_zk_ecdsa_snark_witness(
    ufsecp_ctx* ctx,
    const uint8_t msg_hash32[32],
    const uint8_t pubkey33[33],
    const uint8_t sig64[64],
    ufsecp_ecdsa_snark_witness_t* out)
{
    if (!ctx || !msg_hash32 || !pubkey33 || !sig64 || !out)
        return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    /* Parse public key from compressed 33-byte encoding */
    auto pubkey = point_from_compressed(pubkey33);
    if (pubkey.is_infinity())
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid pubkey");

    /* Parse r and s from compact 64-byte signature */
    std::array<uint8_t, 32> r_bytes{};
    std::array<uint8_t, 32> s_bytes{};
    std::memcpy(r_bytes.data(), sig64,      32);
    std::memcpy(s_bytes.data(), sig64 + 32, 32);

    Scalar sig_r, sig_s;
    if (!scalar_parse_strict(r_bytes.data(), sig_r) || sig_r.is_zero())
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid sig r");
    if (!scalar_parse_strict(s_bytes.data(), sig_s) || sig_s.is_zero())
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid sig s");

    /* Compute witness via C++ layer */
    std::array<uint8_t, 32> msg_arr{};
    std::memcpy(msg_arr.data(), msg_hash32, 32);
    auto w = secp256k1::zk::ecdsa_snark_witness(msg_arr, pubkey, sig_r, sig_s);

    /* ── public inputs ──────────────────────────────────────────────── */
    std::memcpy(out->msg,    msg_hash32, 32);
    std::memcpy(out->sig_r,  r_bytes.data(), 32);
    std::memcpy(out->sig_s,  s_bytes.data(), 32);

    auto px_bytes = pubkey.x().to_bytes();
    auto py_bytes = pubkey.y().to_bytes();
    std::memcpy(out->pub_x, px_bytes.data(), 32);
    std::memcpy(out->pub_y, py_bytes.data(), 32);

    /* ── private witness bytes ──────────────────────────────────────── */
    std::memcpy(out->s_inv,          w.bytes_s_inv.data(),          32);
    std::memcpy(out->u1,             w.bytes_u1.data(),             32);
    std::memcpy(out->u2,             w.bytes_u2.data(),             32);
    std::memcpy(out->result_x,       w.bytes_result_x.data(),       32);
    std::memcpy(out->result_y,       w.bytes_result_y.data(),       32);
    std::memcpy(out->result_x_mod_n, w.bytes_result_x_mod_n.data(), 32);

    /* ── 5×52-bit limb decompositions ───────────────────────────────── */
    static_assert(sizeof(ufsecp_ff_limbs_t) == sizeof(secp256k1::zk::ForeignFieldLimbs),
                  "limb struct size mismatch");
    std::memcpy(&out->lmb_sig_r,          &w.sig_r,          sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_sig_s,          &w.sig_s,          sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_pub_x,          &w.pub_x,          sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_pub_y,          &w.pub_y,          sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_s_inv,          &w.s_inv,          sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_u1,             &w.u1,             sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_u2,             &w.u2,             sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_result_x,       &w.result_x,       sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_result_y,       &w.result_y,       sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_result_x_mod_n, &w.result_x_mod_n, sizeof(ufsecp_ff_limbs_t));

    out->valid = w.valid ? 1 : 0;
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP340 Schnorr-in-SNARK Foreign-Field Witness
 * =========================================================================== */

ufsecp_error_t ufsecp_zk_schnorr_snark_witness(
    ufsecp_ctx* ctx,
    const uint8_t msg32[32],
    const uint8_t pubkey_x32[32],
    const uint8_t sig64[64],
    ufsecp_schnorr_snark_witness_t* out)
{
    if (!ctx || !msg32 || !pubkey_x32 || !sig64 || !out)
        return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    /* Parse s from signature bytes [32..64) */
    std::array<uint8_t, 32> s_bytes{};
    std::memcpy(s_bytes.data(), sig64 + 32, 32);
    Scalar sig_s;
    if (!scalar_parse_strict(s_bytes.data(), sig_s) || sig_s.is_zero())
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid sig s");

    /* Build input arrays */
    std::array<uint8_t, 32> msg_arr{};
    std::array<uint8_t, 32> pubkey_x_arr{};
    std::array<uint8_t, 32> sig_r_arr{};
    std::memcpy(msg_arr.data(),      msg32,       32);
    std::memcpy(pubkey_x_arr.data(), pubkey_x32,  32);
    std::memcpy(sig_r_arr.data(),    sig64,        32);

    /* Compute witness via C++ layer */
    auto w = secp256k1::zk::schnorr_snark_witness(msg_arr, pubkey_x_arr,
                                                    sig_r_arr, sig_s);

    /* ── public inputs ──────────────────────────────────────────────── */
    std::memcpy(out->msg,   msg32,       32);
    std::memcpy(out->sig_r, sig64,       32);
    std::memcpy(out->sig_s, s_bytes.data(), 32);
    std::memcpy(out->pub_x, pubkey_x32,  32);

    /* ── private witness bytes ──────────────────────────────────────── */
    std::memcpy(out->r_y,   w.bytes_r_y.data(),   32);
    std::memcpy(out->pub_y, w.bytes_pub_y.data(),  32);
    std::memcpy(out->e,     w.bytes_e.data(),      32);

    /* ── 5×52-bit limb decompositions ───────────────────────────────── */
    static_assert(sizeof(ufsecp_ff_limbs_t) == sizeof(secp256k1::zk::ForeignFieldLimbs),
                  "limb struct size mismatch");
    std::memcpy(&out->lmb_sig_r, &w.sig_r, sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_sig_s, &w.sig_s, sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_pub_x, &w.pub_x, sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_r_y,   &w.r_y,   sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_pub_y, &w.pub_y, sizeof(ufsecp_ff_limbs_t));
    std::memcpy(&out->lmb_e,     &w.e,     sizeof(ufsecp_ff_limbs_t));

    out->valid = w.valid ? 1 : 0;
    return UFSECP_OK;
}

/* ===========================================================================
 * Multi-coin wallet infrastructure
 * =========================================================================== */

static const secp256k1::coins::CoinParams* find_coin(uint32_t coin_type) {
    return secp256k1::coins::find_by_coin_type(coin_type);
}

