/* ============================================================================
 * Coin-specific APIs: coin address/WIF, BTC message, Silent Payments, ECIES, BIP-324, EllSwift, Ethereum, BIP-85, Schnorr message sign
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

ufsecp_error_t ufsecp_coin_address(ufsecp_ctx* ctx,
                                   const uint8_t pubkey33[33],
                                   uint32_t coin_type, int testnet,
                                   char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !addr_out || !addr_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto coin = find_coin(coin_type);
    if (SECP256K1_UNLIKELY(!coin)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unknown coin type");
    }
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    try {
    auto addr = secp256k1::coins::coin_address(pk, *coin, testnet != 0);
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "address generation failed");
    }
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "address buffer too small");
    }
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_coin_derive_from_seed(
    ufsecp_ctx* ctx,
    const uint8_t* seed, size_t seed_len,
    uint32_t coin_type, uint32_t account, int change, uint32_t index,
    int testnet,
    uint8_t* privkey32_out,
    uint8_t* pubkey33_out,
    char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !seed)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if ((addr_out == nullptr) != (addr_len == nullptr)) {
        return ctx_set_err(ctx, UFSECP_ERR_NULL_ARG,
            "addr_out and addr_len must both be null or both be non-null");
    }
    if (seed_len < 16 || seed_len > 64) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "seed must be 16-64 bytes");
    }
    auto coin = find_coin(coin_type);
    if (SECP256K1_UNLIKELY(!coin)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unknown coin type");
    }
    try {
    /* BIP-32 master */
    auto bip32_result = secp256k1::bip32_master_key(seed, seed_len);
    if (SECP256K1_UNLIKELY(!bip32_result.second)) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-32 master key failed");
    }
    auto master = bip32_result.first;
    const auto cleanup_keys = [&]() {
        secp256k1::detail::secure_erase(master.key.data(), master.key.size());
        secp256k1::detail::secure_erase(master.chain_code.data(), master.chain_code.size());
    };
    /* Derive coin key */
    auto derived = secp256k1::coins::coin_derive_key(
        master, *coin, account, change != 0, index);
    auto key = derived.first;
    bool const d_ok = derived.second;
    if (SECP256K1_UNLIKELY(!d_ok)) {
        cleanup_keys();
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "coin key derivation failed");
    }
    const auto cleanup_derived_key = [&]() {
        secp256k1::detail::secure_erase(key.key.data(), key.key.size());
        secp256k1::detail::secure_erase(key.chain_code.data(), key.chain_code.size());
    };
    if (privkey32_out) {
        auto sk = key.private_key();
        scalar_to_bytes(sk, privkey32_out);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
    }
    auto pk = key.public_key();
    cleanup_keys();
    cleanup_derived_key();
    if (pubkey33_out) {
        point_to_compressed(pk, pubkey33_out);
    }
    if (addr_out && addr_len) {
        auto addr = secp256k1::coins::coin_address(pk, *coin, testnet != 0);
        if (*addr_len < addr.size() + 1) {
            return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "address buffer too small");
        }
        std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
        *addr_len = addr.size();
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_coin_wif_encode(ufsecp_ctx* ctx,
                                      const uint8_t privkey[32],
                                      uint32_t coin_type, int testnet,
                                      char* wif_out, size_t* wif_len) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !wif_out || !wif_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto coin = find_coin(coin_type);
    if (SECP256K1_UNLIKELY(!coin)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unknown coin type");
    }
    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    ScopeSecureErase<Scalar> sk_erase{&sk, sizeof(sk)}; // erases sk on all exit paths
    try {
    auto wif = secp256k1::coins::coin_wif_encode(sk, *coin, true, testnet != 0);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (wif.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "WIF encode failed");
    }
    if (*wif_len < wif.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "WIF buffer too small");
    }
    std::memcpy(wif_out, wif.c_str(), wif.size() + 1);
    *wif_len = wif.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_btc_message_sign(ufsecp_ctx* ctx,
                                       const uint8_t* msg, size_t msg_len,
                                       const uint8_t privkey[32],
                                       char* base64_out, size_t* base64_len) {
    if (SECP256K1_UNLIKELY(!ctx || !msg || !privkey || !base64_out || !base64_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    ScopeSecureErase<Scalar> sk_erase{&sk, sizeof(sk)}; // erases sk on all exit paths
    try {
    auto rsig = secp256k1::coins::bitcoin_sign_message(msg, msg_len, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto b64 = secp256k1::coins::bitcoin_sig_to_base64(rsig);
    if (*base64_len < b64.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "base64 buffer too small");
    }
    std::memcpy(base64_out, b64.c_str(), b64.size() + 1);
    *base64_len = b64.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_btc_message_verify(ufsecp_ctx* ctx,
                                         const uint8_t* msg, size_t msg_len,
                                         const uint8_t pubkey33[33],
                                         const char* base64_sig) {
    if (SECP256K1_UNLIKELY(!ctx || !msg || !pubkey33 || !base64_sig)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    try {
    auto dec = secp256k1::coins::bitcoin_sig_from_base64(std::string(base64_sig));
    if (SECP256K1_UNLIKELY(!dec.valid)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid base64 signature");
    }
    if (!secp256k1::coins::bitcoin_verify_message(msg, msg_len, pk, dec.sig)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "BTC message verify failed");
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_btc_message_hash(const uint8_t* msg, size_t msg_len,
                                       uint8_t digest32_out[32]) {
    if (SECP256K1_UNLIKELY(!msg || !digest32_out)) return UFSECP_ERR_NULL_ARG;
    auto h = secp256k1::coins::bitcoin_message_hash(msg, msg_len);
    std::memcpy(digest32_out, h.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-352 Silent Payments
 * =========================================================================== */

ufsecp_error_t ufsecp_silent_payment_address(
    ufsecp_ctx* ctx,
    const uint8_t scan_privkey[32],
    const uint8_t spend_privkey[32],
    uint8_t scan_pubkey33_out[33],
    uint8_t spend_pubkey33_out[33],
    char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !scan_privkey || !spend_privkey || !scan_pubkey33_out ||
        !spend_pubkey33_out || !addr_out || !addr_len)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    Scalar scan_sk, spend_sk;
    ScopeSecureErase<Scalar> scan_sk_erase{&scan_sk, sizeof(scan_sk)}; // erases on all exit paths
    ScopeSecureErase<Scalar> spend_sk_erase{&spend_sk, sizeof(spend_sk)};
    auto cleanup = [&]() {
        secp256k1::detail::secure_erase(&scan_sk, sizeof(scan_sk));
        secp256k1::detail::secure_erase(&spend_sk, sizeof(spend_sk));
    };
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(scan_privkey, scan_sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "scan privkey is zero or >= n");
    }
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(spend_privkey, spend_sk))) {
        cleanup();
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "spend privkey is zero or >= n");
    }

    auto spa = secp256k1::silent_payment_address(scan_sk, spend_sk);
    auto scan_comp  = spa.scan_pubkey.to_compressed();
    auto spend_comp = spa.spend_pubkey.to_compressed();
    std::memcpy(scan_pubkey33_out, scan_comp.data(), 33);
    std::memcpy(spend_pubkey33_out, spend_comp.data(), 33);

    auto addr_str = spa.encode();
    if (addr_str.size() >= *addr_len) {
        cleanup();
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "address buffer too small");
    }
    std::memcpy(addr_out, addr_str.c_str(), addr_str.size() + 1);
    *addr_len = addr_str.size();

    cleanup();
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_silent_payment_create_output(
    ufsecp_ctx* ctx,
    const uint8_t* input_privkeys, size_t n_inputs,
    const uint8_t scan_pubkey33[33],
    const uint8_t spend_pubkey33[33],
    uint32_t k,
    uint8_t output_pubkey33_out[33],
    uint8_t* tweak32_out) {
    if (SECP256K1_UNLIKELY(!ctx || !input_privkeys || n_inputs == 0 || !scan_pubkey33 ||
        !spend_pubkey33 || !output_pubkey33_out)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);
    if (n_inputs > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "input count too large");
    std::size_t total = 0;
    if (!checked_mul_size(n_inputs, std::size_t{32}, total))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "input privkey array size overflow");
    try {

    // Parse input private keys
    std::vector<Scalar> privkeys;
    auto cleanup_privkeys = [&]() {
        for (auto& sk : privkeys) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
        }
    };
    ScopeExit privkeys_erase{cleanup_privkeys}; // erases privkeys on all exit paths (including exception)
    privkeys.reserve(n_inputs);
    for (size_t i = 0; i < n_inputs; ++i) {
        Scalar sk;
        if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(input_privkeys + i * 32, sk))) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            cleanup_privkeys();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "input privkey is zero or >= n");
        }
        privkeys.push_back(sk);
    }

    // Parse recipient address
    secp256k1::SilentPaymentAddress recipient;
    recipient.scan_pubkey = point_from_compressed(scan_pubkey33);
    recipient.spend_pubkey = point_from_compressed(spend_pubkey33);
    if (recipient.scan_pubkey.is_infinity() || recipient.spend_pubkey.is_infinity()) {
        cleanup_privkeys();
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid recipient pubkey");
    }

    auto [output_point, tweak] = secp256k1::silent_payment_create_output(privkeys, recipient, k);
    if (output_point.is_infinity()) {
        cleanup_privkeys();
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "output point is infinity");
    }

    auto out_comp = output_point.to_compressed();
    std::memcpy(output_pubkey33_out, out_comp.data(), 33);

    if (tweak32_out) {
        auto tweak_bytes = tweak.to_bytes();
        std::memcpy(tweak32_out, tweak_bytes.data(), 32);
    }

    cleanup_privkeys();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_silent_payment_scan(
    ufsecp_ctx* ctx,
    const uint8_t scan_privkey[32],
    const uint8_t spend_privkey[32],
    const uint8_t* input_pubkeys33, size_t n_input_pubkeys,
    const uint8_t* output_xonly32, size_t n_outputs,
    uint32_t* found_indices_out,
    uint8_t* found_privkeys_out,
    size_t* n_found) {
    if (SECP256K1_UNLIKELY(!ctx || !scan_privkey || !spend_privkey || !input_pubkeys33 ||
        !output_xonly32 || !n_found)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (n_input_pubkeys == 0 || n_outputs == 0) {
        return UFSECP_ERR_BAD_INPUT;
    }
    if (n_input_pubkeys > kMaxBatchN || n_outputs > kMaxBatchN)
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "input/output count too large");
    std::size_t pk_bytes = 0, out_bytes = 0;
    if (!checked_mul_size(n_input_pubkeys, std::size_t{33}, pk_bytes)
        || !checked_mul_size(n_outputs, std::size_t{32}, out_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "silent payment array size overflow");
    ctx_clear_err(ctx);
    try {

    Scalar scan_sk, spend_sk;
    ScopeSecureErase<Scalar> scan_sk_erase{&scan_sk, sizeof(scan_sk)}; // erases on all exit paths
    ScopeSecureErase<Scalar> spend_sk_erase{&spend_sk, sizeof(spend_sk)};
    auto cleanup = [&]() {
        secp256k1::detail::secure_erase(&scan_sk, sizeof(scan_sk));
        secp256k1::detail::secure_erase(&spend_sk, sizeof(spend_sk));
    };
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(scan_privkey, scan_sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "scan privkey is zero or >= n");
    }
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(spend_privkey, spend_sk))) {
        cleanup();
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "spend privkey is zero or >= n");
    }

    // Parse input pubkeys
    std::vector<Point> input_pks;
    input_pks.reserve(n_input_pubkeys);
    for (size_t i = 0; i < n_input_pubkeys; ++i) {
        auto pk = point_from_compressed(input_pubkeys33 + i * 33);
        if (pk.is_infinity()) {
            cleanup();
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid input pubkey");
        }
        input_pks.push_back(pk);
    }

    // Parse output x-only pubkeys
    std::vector<std::array<uint8_t, 32>> outputs;
    outputs.reserve(n_outputs);
    for (size_t i = 0; i < n_outputs; ++i) {
        std::array<uint8_t, 32> x;
        std::memcpy(x.data(), output_xonly32 + i * 32, 32);
        outputs.push_back(x);
    }

    auto results = secp256k1::silent_payment_scan(scan_sk, spend_sk, input_pks, outputs);

    size_t const capacity = *n_found;
    size_t const count = results.size() < capacity ? results.size() : capacity;
    *n_found = results.size();

    for (size_t i = 0; i < count; ++i) {
        if (found_indices_out) found_indices_out[i] = results[i].first;
        if (found_privkeys_out) {
            auto key_bytes = results[i].second.to_bytes();
            std::memcpy(found_privkeys_out + i * 32, key_bytes.data(), 32);
            secp256k1::detail::secure_erase(key_bytes.data(), key_bytes.size());
        }
    }

    // Erase result private keys from heap before vector destruction
    for (auto& r : results) {
        secp256k1::detail::secure_erase(&r.second, sizeof(r.second));
    }

    cleanup();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * ECIES (Elliptic Curve Integrated Encryption Scheme)
 * =========================================================================== */

ufsecp_error_t ufsecp_ecies_encrypt(
    ufsecp_ctx* ctx,
    const uint8_t recipient_pubkey33[33],
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* envelope_out, size_t* envelope_len) {
    if (SECP256K1_UNLIKELY(!ctx || !recipient_pubkey33 || !plaintext || !envelope_out || !envelope_len)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (plaintext_len == 0) {
        return UFSECP_ERR_BAD_INPUT;
    }
    ctx_clear_err(ctx);

    if (plaintext_len > SIZE_MAX - UFSECP_ECIES_OVERHEAD) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "plaintext_len too large");
    }
    size_t const needed = plaintext_len + UFSECP_ECIES_OVERHEAD;
    if (*envelope_len < needed) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "envelope buffer too small");
    }

    auto pk = point_from_compressed(recipient_pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid recipient pubkey");
    }
    try {
    auto envelope = secp256k1::ecies_encrypt(pk, plaintext, plaintext_len);
    if (envelope.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "ECIES encryption failed");
    }

    std::memcpy(envelope_out, envelope.data(), envelope.size());
    *envelope_len = envelope.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_ecies_decrypt(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t* envelope, size_t envelope_len,
    uint8_t* plaintext_out, size_t* plaintext_len) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !envelope || !plaintext_out || !plaintext_len)) {
        return UFSECP_ERR_NULL_ARG;
    }
    if (envelope_len < 82) { // min: 33 + 16 + 1 + 32
        return UFSECP_ERR_BAD_INPUT;
    }
    ctx_clear_err(ctx);

    size_t const expected_pt_len = envelope_len - UFSECP_ECIES_OVERHEAD;
    if (*plaintext_len < expected_pt_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "plaintext buffer too small");
    }

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    ScopeSecureErase<Scalar> sk_erase{&sk, sizeof(sk)}; // erases sk on all exit paths
    try {
    auto pt = secp256k1::ecies_decrypt(sk, envelope, envelope_len);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));

    if (pt.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "ECIES decryption failed (bad key or tampered)");
    }

    std::memcpy(plaintext_out, pt.data(), pt.size());
    *plaintext_len = pt.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * BIP-324: Version 2 P2P Encrypted Transport (conditional: SECP256K1_BIP324)
 * =========================================================================== */

#if defined(SECP256K1_BIP324)

struct ufsecp_bip324_session {
    secp256k1::Bip324Session* cpp_session;
};

ufsecp_error_t ufsecp_bip324_create(
    ufsecp_ctx* ctx,
    int initiator,
    ufsecp_bip324_session** session_out,
    uint8_t ellswift64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !session_out || !ellswift64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    *session_out = nullptr;
    if (initiator != 0 && initiator != 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "initiator must be 0 or 1");
    }

    auto* sess = new (std::nothrow) ufsecp_bip324_session;
    if (!sess) return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "allocation failed");
    sess->cpp_session = nullptr;

    sess->cpp_session = new (std::nothrow) secp256k1::Bip324Session(initiator == 1);
    if (SECP256K1_UNLIKELY(!sess->cpp_session)) {
        delete sess;
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "allocation failed");
    }

    auto& enc = sess->cpp_session->our_ellswift_encoding();
    std::memcpy(ellswift64_out, enc.data(), 64);

    *session_out = sess;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip324_handshake(
    ufsecp_bip324_session* session,
    const uint8_t peer_ellswift64[64],
    uint8_t session_id32_out[32]) {
    if (SECP256K1_UNLIKELY(!session || !session->cpp_session || !peer_ellswift64)) return UFSECP_ERR_NULL_ARG;

    if (!session->cpp_session->complete_handshake(peer_ellswift64)) {
        return UFSECP_ERR_INTERNAL;
    }

    if (session_id32_out) {
        auto& sid = session->cpp_session->session_id();
        std::memcpy(session_id32_out, sid.data(), 32);
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip324_encrypt(
    ufsecp_bip324_session* session,
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* out, size_t* out_len) {
    if (SECP256K1_UNLIKELY(!session || !session->cpp_session || !out || !out_len)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!plaintext && plaintext_len > 0)) return UFSECP_ERR_NULL_ARG;
    if (plaintext_len > SIZE_MAX - 19) return UFSECP_ERR_BAD_INPUT;

    size_t const needed = plaintext_len + 19; // 3 (length) + payload + 16 (tag)
    if (*out_len < needed) return UFSECP_ERR_BUF_TOO_SMALL;
    try {
    auto enc = session->cpp_session->encrypt(plaintext, plaintext_len);
    if (enc.empty()) return UFSECP_ERR_INTERNAL;

    std::memcpy(out, enc.data(), enc.size());
    *out_len = enc.size();
    return UFSECP_OK;
    } catch (...) { return UFSECP_ERR_INTERNAL; }
}

ufsecp_error_t ufsecp_bip324_decrypt(
    ufsecp_bip324_session* session,
    const uint8_t* encrypted, size_t encrypted_len,
    uint8_t* plaintext_out, size_t* plaintext_len) {
    if (!session || !session->cpp_session || !encrypted || !plaintext_out || !plaintext_len)
        return UFSECP_ERR_NULL_ARG;

    // encrypted = [3B header][payload][16B tag], minimum length 19
    if (encrypted_len < 19) return UFSECP_ERR_BUF_TOO_SMALL;

    const uint8_t* header = encrypted;
    const uint8_t* payload_tag = encrypted + 3;
    const size_t payload_tag_len = encrypted_len - 3;
    try {
    std::vector<uint8_t> dec;
    if (!session->cpp_session->decrypt(header, payload_tag, payload_tag_len, dec)) {
        return UFSECP_ERR_VERIFY_FAIL;
    }

    if (*plaintext_len < dec.size()) return UFSECP_ERR_BUF_TOO_SMALL;
    if (dec.size() > 0) std::memcpy(plaintext_out, dec.data(), dec.size());
    *plaintext_len = dec.size();
    return UFSECP_OK;
    } catch (...) { return UFSECP_ERR_INTERNAL; }
}

void ufsecp_bip324_destroy(ufsecp_bip324_session* session) {
    if (session) {
        if (session->cpp_session) {
            delete session->cpp_session;
        }
        delete session;
    }
}

ufsecp_error_t ufsecp_aead_chacha20_poly1305_encrypt(
    const uint8_t key[32], const uint8_t nonce[12],
    const uint8_t* aad, size_t aad_len,
    const uint8_t* plaintext, size_t plaintext_len,
    uint8_t* out, uint8_t tag[16]) {
    if (SECP256K1_UNLIKELY(!key || !nonce || !tag)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!out && plaintext_len > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!plaintext && plaintext_len > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!aad && aad_len > 0)) return UFSECP_ERR_NULL_ARG;

    secp256k1::aead_chacha20_poly1305_encrypt(
        key, nonce, aad, aad_len, plaintext, plaintext_len, out, tag);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_aead_chacha20_poly1305_decrypt(
    const uint8_t key[32], const uint8_t nonce[12],
    const uint8_t* aad, size_t aad_len,
    const uint8_t* ciphertext, size_t ciphertext_len,
    const uint8_t tag[16], uint8_t* out) {
    if (SECP256K1_UNLIKELY(!key || !nonce || !tag)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!out && ciphertext_len > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!ciphertext && ciphertext_len > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!aad && aad_len > 0)) return UFSECP_ERR_NULL_ARG;

    bool ok = secp256k1::aead_chacha20_poly1305_decrypt(
        key, nonce, aad, aad_len, ciphertext, ciphertext_len, tag, out);
    return ok ? UFSECP_OK : UFSECP_ERR_VERIFY_FAIL;
}

ufsecp_error_t ufsecp_ellswift_create(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    uint8_t encoding64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !encoding64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto enc = secp256k1::ellswift_create(sk);
    std::memcpy(encoding64_out, enc.data(), 64);

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ellswift_xdh(
    ufsecp_ctx* ctx,
    const uint8_t ell_a64[64],
    const uint8_t ell_b64[64],
    const uint8_t our_privkey[32],
    int initiating,
    uint8_t secret32_out[32]) {
    if (!ctx || !ell_a64 || !ell_b64 || !our_privkey || !secret32_out)
        return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(our_privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto secret = secp256k1::ellswift_xdh(ell_a64, ell_b64, sk, initiating != 0);
    std::memcpy(secret32_out, secret.data(), 32);

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    return UFSECP_OK;
}

#endif /* SECP256K1_BIP324 */

/* ===========================================================================
 * Ethereum (conditional: SECP256K1_BUILD_ETHEREUM)
 * =========================================================================== */

#if defined(SECP256K1_BUILD_ETHEREUM)

ufsecp_error_t ufsecp_keccak256(const uint8_t* data, size_t len,
                                uint8_t digest32_out[32]) {
    if (SECP256K1_UNLIKELY(!data && len > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!digest32_out)) return UFSECP_ERR_NULL_ARG;

    auto hash = secp256k1::coins::keccak256(data, len);
    std::memcpy(digest32_out, hash.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_address(ufsecp_ctx* ctx,
                                  const uint8_t pubkey33[33],
                                  uint8_t addr20_out[20]) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !addr20_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    const Point pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid compressed pubkey");
    }

    auto addr = secp256k1::coins::ethereum_address_bytes(pk);
    std::memcpy(addr20_out, addr.data(), 20);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_address_checksummed(ufsecp_ctx* ctx,
                                              const uint8_t pubkey33[33],
                                              char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !addr_out || !addr_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (*addr_len < 43) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "need >= 43 bytes for ETH address");
    }

    const Point pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid compressed pubkey");
    }

    try {
    const std::string addr_str = secp256k1::coins::ethereum_address(pk);
    std::memcpy(addr_out, addr_str.c_str(), addr_str.size());
    addr_out[addr_str.size()] = '\0';
    *addr_len = addr_str.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_eth_personal_hash(const uint8_t* msg, size_t msg_len,
                                        uint8_t digest32_out[32]) {
    if (SECP256K1_UNLIKELY(!msg && msg_len > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!digest32_out)) return UFSECP_ERR_NULL_ARG;

    auto hash = secp256k1::coins::eip191_hash(msg, msg_len);
    std::memcpy(digest32_out, hash.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_sign(ufsecp_ctx* ctx,
                               const uint8_t msg32[32],
                               const uint8_t privkey[32],
                               uint8_t r_out[32],
                               uint8_t s_out[32],
                               uint64_t* v_out,
                               uint64_t chain_id) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !privkey || !r_out || !s_out || !v_out)) {
        return UFSECP_ERR_NULL_ARG;
    }
    ctx_clear_err(ctx);

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    std::array<uint8_t, 32> hash;
    std::memcpy(hash.data(), msg32, 32);

    auto esig = secp256k1::coins::eth_sign_hash(hash, sk, chain_id);
    std::memcpy(r_out, esig.r.data(), 32);
    std::memcpy(s_out, esig.s.data(), 32);
    *v_out = esig.v;

    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_eth_ecrecover(ufsecp_ctx* ctx,
                                    const uint8_t msg32[32],
                                    const uint8_t r[32],
                                    const uint8_t s[32],
                                    uint64_t v,
                                    uint8_t addr20_out[20]) {
    if (SECP256K1_UNLIKELY(!ctx || !msg32 || !r || !s || !addr20_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> hash, r_arr, s_arr;
    std::memcpy(hash.data(), msg32, 32);
    std::memcpy(r_arr.data(), r, 32);
    std::memcpy(s_arr.data(), s, 32);

    auto [addr, ok] = secp256k1::coins::ecrecover(hash, r_arr, s_arr, v);
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "ecrecover failed");
    }

    std::memcpy(addr20_out, addr.data(), 20);
    return UFSECP_OK;
}

#endif /* SECP256K1_BUILD_ETHEREUM */

/* ===========================================================================
 * BIP-85 — Deterministic Entropy from BIP-32 Keychains
 * =========================================================================== */

ufsecp_error_t ufsecp_bip85_entropy(
    ufsecp_ctx* ctx,
    const ufsecp_bip32_key* master_xprv,
    const char* path,
    uint8_t* entropy_out, size_t entropy_len) {
    if (SECP256K1_UNLIKELY(!ctx || !master_xprv || !path || !entropy_out)) return UFSECP_ERR_NULL_ARG;
    if (entropy_len == 0 || entropy_len > 32) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "entropy_len must be 1-32");
    }
    ctx_clear_err(ctx);

    if (SECP256K1_UNLIKELY(!master_xprv->is_private)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "BIP-85 requires xprv (private key)");
    }

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, master_xprv, ek);
    if (parse_rc != UFSECP_OK) return parse_rc;
    ScopeExit ek_erase{[&ek]() noexcept {
        secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
        secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    }};

    try {
    // Derive at the given path (all components must be hardened per BIP-85)
    auto [derived, ok] = secp256k1::bip32_derive_path(ek, std::string(path));
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "BIP-85 path derivation failed");
    }

    // Get the derived private key
    auto child_privkey = derived.private_key().to_bytes();
    secp256k1::detail::secure_erase(derived.key.data(), derived.key.size());
    secp256k1::detail::secure_erase(derived.chain_code.data(), derived.chain_code.size());

    // HMAC-SHA512(key="bip-85", data=child_privkey)
    static const uint8_t BIP85_KEY[] = {'b','i','p','-','8','5'};
    auto hmac = secp256k1::hmac_sha512(BIP85_KEY, 6, child_privkey.data(), 32);
    secp256k1::detail::secure_erase(child_privkey.data(), child_privkey.size());

    std::memcpy(entropy_out, hmac.data(), entropy_len);
    secp256k1::detail::secure_erase(hmac.data(), hmac.size());
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_bip85_bip39(
    ufsecp_ctx* ctx,
    const ufsecp_bip32_key* master_xprv,
    uint32_t words, uint32_t language_index, uint32_t index,
    char* mnemonic_out, size_t* mnemonic_len) {
    if (SECP256K1_UNLIKELY(!ctx || !master_xprv || !mnemonic_out || !mnemonic_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (words != 12 && words != 18 && words != 24) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "words must be 12, 18, or 24");
    }
    if (language_index != 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "only language_index=0 (English) supported");
    }

    // BIP-85 path for BIP-39: m/83696968'/39'/<language>'/<words>'/<index>'
    char path[128];
    std::snprintf(path, sizeof(path),
        "m/83696968'/39'/%u'/%u'/%u'",
        language_index, words, index);

    // entropy_len = (words / 3) * 4
    size_t const entropy_len = (static_cast<size_t>(words) / 3) * 4;
    uint8_t entropy[32]{};
    ScopeSecureErase<uint8_t> entropy_erase{entropy, sizeof(entropy)}; // erases entropy on all exit paths

    ufsecp_error_t rc = ufsecp_bip85_entropy(ctx, master_xprv, path, entropy, entropy_len);
    if (rc != UFSECP_OK) {
        secp256k1::detail::secure_erase(entropy, sizeof(entropy));
        return rc;
    }

    try {
    auto [mnemonic, ok] = secp256k1::bip39_generate(entropy_len, entropy);
    secp256k1::detail::secure_erase(entropy, sizeof(entropy));
    if (!ok || mnemonic.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-85 BIP-39 mnemonic generation failed");
    }
    if (*mnemonic_len < mnemonic.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "mnemonic buffer too small");
    }
    std::memcpy(mnemonic_out, mnemonic.c_str(), mnemonic.size() + 1);
    *mnemonic_len = mnemonic.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * BIP-340 Variable-Length Schnorr
 * =========================================================================== */

ufsecp_error_t ufsecp_schnorr_sign_msg(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    const uint8_t* msg, size_t msg_len,
    const uint8_t* aux_rand32,
    uint8_t sig64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !sig64_out)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!msg && msg_len > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!aux_rand32)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    // Hash the message with BIP-340 tagged hash "BIP0340/msg"
    static const uint8_t tag_data[] = "BIP0340/msg";
    uint8_t msg_hash[32];
    {
        // tagged_hash("BIP0340/msg", msg) = SHA256(SHA256(tag)||SHA256(tag)||msg)
        auto tag_hash = secp256k1::SHA256::hash(tag_data, sizeof(tag_data) - 1);
        secp256k1::SHA256 h;
        h.update(tag_hash.data(), 32);
        h.update(tag_hash.data(), 32);
        if (msg && msg_len > 0) h.update(msg, msg_len);
        auto digest = h.finalize();
        std::memcpy(msg_hash, digest.data(), 32);
    }

    // Use the fixed 32-byte sign API
    uint8_t aux_arr[32];
    std::memcpy(aux_arr, aux_rand32, 32);
    return ufsecp_schnorr_sign(ctx, msg_hash, privkey, aux_arr, sig64_out);
}

ufsecp_error_t ufsecp_schnorr_verify_msg(
    ufsecp_ctx* ctx,
    const uint8_t pubkey_x[32],
    const uint8_t* msg, size_t msg_len,
    const uint8_t sig64[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey_x || !sig64)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!msg && msg_len > 0)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    static const uint8_t tag_data[] = "BIP0340/msg";
    uint8_t msg_hash[32];
    {
        auto tag_hash = secp256k1::SHA256::hash(tag_data, sizeof(tag_data) - 1);
        secp256k1::SHA256 h;
        h.update(tag_hash.data(), 32);
        h.update(tag_hash.data(), 32);
        if (msg && msg_len > 0) h.update(msg, msg_len);
        auto digest = h.finalize();
        std::memcpy(msg_hash, digest.data(), 32);
    }

    return ufsecp_schnorr_verify(ctx, msg_hash, sig64, pubkey_x);
}

/* ===========================================================================
 * BIP-322 — Generic Message Signing
 * =========================================================================== */

