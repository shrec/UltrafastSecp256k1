/* ============================================================================
 * Hash functions, address encoding (P2PKH/P2WPKH/P2TR/P2SH), WIF, BIP-32
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

ufsecp_error_t ufsecp_sha256(const uint8_t* data, size_t len,
                             uint8_t digest32_out[32]) {
    if (SECP256K1_UNLIKELY(!data || !digest32_out)) return UFSECP_ERR_NULL_ARG;
    secp256k1::SHA256 hasher;
    hasher.update(data, len);
    auto digest = hasher.finalize();
    std::memcpy(digest32_out, digest.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_hash160(const uint8_t* data, size_t len,
                              uint8_t digest20_out[20]) {
    if (SECP256K1_UNLIKELY(!data || !digest20_out)) return UFSECP_ERR_NULL_ARG;
    auto h = secp256k1::hash160(data, len);
    std::memcpy(digest20_out, h.data(), 20);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_tagged_hash(const char* tag,
                                  const uint8_t* data, size_t len,
                                  uint8_t digest32_out[32]) {
    if (SECP256K1_UNLIKELY(!tag || !data || !digest32_out)) return UFSECP_ERR_NULL_ARG;
    auto h = secp256k1::tagged_hash(tag, data, len);
    std::memcpy(digest32_out, h.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * Bitcoin addresses
 * =========================================================================== */

ufsecp_error_t ufsecp_addr_p2pkh(ufsecp_ctx* ctx,
                                 const uint8_t pubkey33[33], int network,
                                 char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !addr_out || !addr_len)) return UFSECP_ERR_NULL_ARG;
    if (!valid_network(network)) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid network");
    ctx_clear_err(ctx);

    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    try {
    auto addr = secp256k1::address_p2pkh(pk, to_network(network));
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "P2PKH generation failed");
}
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "P2PKH buffer too small");
}
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_addr_p2wpkh(ufsecp_ctx* ctx,
                                  const uint8_t pubkey33[33], int network,
                                  char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !addr_out || !addr_len)) return UFSECP_ERR_NULL_ARG;
    if (!valid_network(network)) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid network");
    ctx_clear_err(ctx);

    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    try {
    auto addr = secp256k1::address_p2wpkh(pk, to_network(network));
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "P2WPKH generation failed");
}
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "P2WPKH buffer too small");
}
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_addr_p2tr(ufsecp_ctx* ctx,
                                const uint8_t internal_key_x[32], int network,
                                char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !internal_key_x || !addr_out || !addr_len)) return UFSECP_ERR_NULL_ARG;
    if (!valid_network(network)) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid network");
    ctx_clear_err(ctx);

    // Reject all-zero x-only key (not a valid curve point)
    {
        uint8_t acc = 0;
        for (int i = 0; i < 32; ++i) acc |= internal_key_x[i];
        if (acc == 0) return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "zero x-only key");
    }

    std::array<uint8_t, 32> key_x;
    std::memcpy(key_x.data(), internal_key_x, 32);
    try {
    auto addr = secp256k1::address_p2tr_raw(key_x, to_network(network));
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "P2TR generation failed");
}
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "P2TR buffer too small");
}
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_addr_p2sh(
    const uint8_t* redeem_script, size_t redeem_script_len,
    int network,
    char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!redeem_script && redeem_script_len > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!addr_out || !addr_len)) return UFSECP_ERR_NULL_ARG;
    if (!valid_network(network)) return UFSECP_ERR_BAD_INPUT;

    try {
    // hash160 of redeem_script
    auto script_hash = secp256k1::hash160(redeem_script, redeem_script_len);
    auto addr = secp256k1::address_p2sh(script_hash, to_network(network));
    if (addr.empty()) return UFSECP_ERR_INTERNAL;
    if (*addr_len < addr.size() + 1) return UFSECP_ERR_BUF_TOO_SMALL;
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
    } catch (...) { return UFSECP_ERR_INTERNAL; }
}

ufsecp_error_t ufsecp_addr_p2sh_p2wpkh(
    ufsecp_ctx* ctx,
    const uint8_t pubkey33[33],
    int network,
    char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !addr_out || !addr_len)) return UFSECP_ERR_NULL_ARG;
    if (!valid_network(network)) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid network");
    ctx_clear_err(ctx);

    auto pk = point_from_compressed(pubkey33);
    if (pk.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    try {
    auto addr = secp256k1::address_p2sh_p2wpkh(pk, to_network(network));
    if (addr.empty()) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "P2SH-P2WPKH generation failed");
    }
    if (*addr_len < addr.size() + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "P2SH-P2WPKH buffer too small");
    }
    std::memcpy(addr_out, addr.c_str(), addr.size() + 1);
    *addr_len = addr.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * WIF
 * =========================================================================== */

ufsecp_error_t ufsecp_wif_encode(ufsecp_ctx* ctx,
                                 const uint8_t privkey[32],
                                 int compressed, int network,
                                 char* wif_out, size_t* wif_len) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !wif_out || !wif_len)) return UFSECP_ERR_NULL_ARG;
    if (!valid_network(network)) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid network");
    ctx_clear_err(ctx);

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    ScopeSecureErase<Scalar> sk_erase{&sk, sizeof(sk)}; // erases sk on all exit paths
    try {
    auto wif = secp256k1::wif_encode(sk, compressed != 0, to_network(network));
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

ufsecp_error_t ufsecp_wif_decode(ufsecp_ctx* ctx,
                                 const char* wif,
                                 uint8_t privkey32_out[32],
                                 int* compressed_out,
                                 int* network_out) {
    if (SECP256K1_UNLIKELY(!ctx || !wif || !privkey32_out || !compressed_out || !network_out)) {
        return UFSECP_ERR_NULL_ARG;
}
    ctx_clear_err(ctx);

    try {
    auto result = secp256k1::wif_decode(std::string(wif));
    if (SECP256K1_UNLIKELY(!result.valid)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid WIF string");
}

    scalar_to_bytes(result.key, privkey32_out);
    secp256k1::detail::secure_erase(&result.key, sizeof(result.key));
    *compressed_out = result.compressed ? 1 : 0;
    *network_out = result.network == secp256k1::Network::Testnet
                       ? UFSECP_NET_TESTNET : UFSECP_NET_MAINNET;
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * BIP-32
 * =========================================================================== */

static void extkey_to_uf(const secp256k1::ExtendedKey& ek, ufsecp_bip32_key* out) {
    auto serialized = ek.serialize();
    std::memcpy(out->data, serialized.data(), 78);
    out->is_private = ek.is_private ? 1 : 0;
    std::memset(out->_pad, 0, sizeof(out->_pad));
}

static secp256k1::ExtendedKey extkey_from_uf(const ufsecp_bip32_key* k) {
    secp256k1::ExtendedKey ek{};
    ek.depth = k->data[4];
    std::memcpy(ek.parent_fingerprint.data(), k->data + 5, 4);
    ek.child_number = (uint32_t(k->data[9]) << 24)  | (uint32_t(k->data[10]) << 16) |
                      (uint32_t(k->data[11]) << 8)   | uint32_t(k->data[12]);
    std::memcpy(ek.chain_code.data(), k->data + 13, 32);
    std::memcpy(ek.key.data(), k->data + 46, 32);
    if (k->is_private) {
        ek.is_private = true;
    } else {
        ek.is_private = false;
        ek.pub_prefix = k->data[45];
    }
    return ek;
}

static ufsecp_error_t parse_bip32_key(ufsecp_ctx* ctx,
                                      const ufsecp_bip32_key* key,
                                      secp256k1::ExtendedKey& out) {
    if (key->is_private > 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 key kind");
    }
    if (key->_pad[0] != 0 || key->_pad[1] != 0 || key->_pad[2] != 0) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 reserved bytes");
    }

    const uint32_t version = (uint32_t(key->data[0]) << 24) |
                             (uint32_t(key->data[1]) << 16) |
                             (uint32_t(key->data[2]) << 8) |
                             uint32_t(key->data[3]);
    const uint32_t expected_version = key->is_private ? 0x0488ADE4u : 0x0488B21Eu;
    if (version != expected_version) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 version");
    }

    if (key->is_private != 0) {
        if (key->data[45] != 0x00) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 private marker");
        }
        Scalar sk;
        if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(key->data + 46, sk))) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "invalid BIP-32 private key");
        }
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
    } else {
        if (key->data[45] != 0x02 && key->data[45] != 0x03) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid BIP-32 public key prefix");
        }
        auto pk = point_from_compressed(key->data + 45);
        if (pk.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid BIP-32 public key");
        }
    }

    out = extkey_from_uf(key);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_master(ufsecp_ctx* ctx,
                                   const uint8_t* seed, size_t seed_len,
                                   ufsecp_bip32_key* key_out) {
    if (SECP256K1_UNLIKELY(!ctx || !seed || !key_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (seed_len < 16 || seed_len > 64) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "seed must be 16-64 bytes");
}

    auto [ek, ok] = secp256k1::bip32_master_key(seed, seed_len);
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-32 master key failed");
}

    extkey_to_uf(ek, key_out);
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_derive(ufsecp_ctx* ctx,
                                   const ufsecp_bip32_key* parent,
                                   uint32_t index,
                                   ufsecp_bip32_key* child_out) {
    if (SECP256K1_UNLIKELY(!ctx || !parent || !child_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, parent, ek);
    if (parse_rc != UFSECP_OK) {
        return parse_rc;
    }
    auto [child, ok] = ek.derive_child(index);
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-32 derivation failed");
}

    extkey_to_uf(child, child_out);
    secp256k1::detail::secure_erase(child.key.data(), child.key.size());
    secp256k1::detail::secure_erase(child.chain_code.data(), child.chain_code.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_derive_path(ufsecp_ctx* ctx,
                                        const ufsecp_bip32_key* master,
                                        const char* path,
                                        ufsecp_bip32_key* key_out) {
    if (SECP256K1_UNLIKELY(!ctx || !master || !path || !key_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, master, ek);
    if (parse_rc != UFSECP_OK) {
        return parse_rc;
    }
    ScopeExit ek_erase{[&ek]() noexcept {
        secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
        secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    }};
    try {
    auto [derived, ok] = secp256k1::bip32_derive_path(ek, std::string(path));
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid BIP-32 path");
}

    extkey_to_uf(derived, key_out);
    secp256k1::detail::secure_erase(derived.key.data(), derived.key.size());
    secp256k1::detail::secure_erase(derived.chain_code.data(), derived.chain_code.size());
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_bip32_privkey(ufsecp_ctx* ctx,
                                    const ufsecp_bip32_key* key,
                                    uint8_t privkey32_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !key || !privkey32_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (SECP256K1_UNLIKELY(!key->is_private)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "key is public, not private");
}

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, key, ek);
    if (parse_rc != UFSECP_OK) {
        return parse_rc;
    }
    auto sk = ek.private_key();
    scalar_to_bytes(sk, privkey32_out);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip32_pubkey(ufsecp_ctx* ctx,
                                   const ufsecp_bip32_key* key,
                                   uint8_t pubkey33_out[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !key || !pubkey33_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::ExtendedKey ek{};
    ufsecp_error_t const parse_rc = parse_bip32_key(ctx, key, ek);
    if (parse_rc != UFSECP_OK) {
        return parse_rc;
    }
    auto pk = ek.public_key();
    if (pk.is_infinity()) {
        secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
        secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid BIP-32 public key");
    }
    point_to_compressed(pk, pubkey33_out);
    secp256k1::detail::secure_erase(ek.key.data(), ek.key.size());
    secp256k1::detail::secure_erase(ek.chain_code.data(), ek.chain_code.size());
    return UFSECP_OK;
}

/* ===========================================================================
 * Taproot (BIP-341)
 * =========================================================================== */

