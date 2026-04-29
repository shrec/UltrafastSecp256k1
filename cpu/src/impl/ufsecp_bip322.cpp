/* ============================================================================
 * BIP-322 generic message signing, GCS filters, PSBT signing, descriptors
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

ufsecp_error_t ufsecp_bip322_sign(
    ufsecp_ctx* ctx,
    const uint8_t privkey[32],
    ufsecp_bip322_addr_type addr_type,
    const uint8_t* msg, size_t msg_len,
    uint8_t* sig_out, size_t* sig_len) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !sig_out || !sig_len)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!msg && msg_len > 0)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    // BIP-322 message hash: tagged_hash("BIP0322-signed-message", msg)
    static const uint8_t bip322_tag[] = "BIP0322-signed-message";
    uint8_t msg_hash[32];
    {
        auto tag_hash = secp256k1::SHA256::hash(bip322_tag, sizeof(bip322_tag) - 1);
        secp256k1::SHA256 h;
        h.update(tag_hash.data(), 32);
        h.update(tag_hash.data(), 32);
        if (msg && msg_len > 0) h.update(msg, msg_len);
        auto digest = h.finalize();
        std::memcpy(msg_hash, digest.data(), 32);
    }

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    ScopeSecureErase<Scalar> sk_erase{&sk, sizeof(sk)}; // erases sk on all exit paths

    try {
    if (addr_type == UFSECP_BIP322_ADDR_P2TR) {
        // P2TR: Schnorr sign
        if (*sig_len < 64) {
            return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "P2TR sig buffer too small (need 64)");
        }
        std::array<uint8_t, 32> msg_arr{}, aux_arr{};
        std::memcpy(msg_arr.data(), msg_hash, 32);
        auto kp = secp256k1::ct::schnorr_keypair_create(sk);
        ScopeSecureErase<decltype(kp.d)> kp_d_erase{&kp.d, sizeof(kp.d)}; // erases kp.d on all exit paths
        auto sig = secp256k1::ct::schnorr_sign(kp, msg_arr, aux_arr);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));
        auto bytes = sig.to_bytes();
        std::memcpy(sig_out, bytes.data(), 64);
        *sig_len = 64;
    } else {
        // P2PKH, P2WPKH, P2SH-P2WPKH: ECDSA sign
        if (*sig_len < 65) {
            secp256k1::detail::secure_erase(&sk, sizeof(sk));
            return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "ECDSA sig buffer too small (need 65)");
        }
        std::array<uint8_t, 32> msg_arr;
        std::memcpy(msg_arr.data(), msg_hash, 32);
        auto sig = secp256k1::ct::ecdsa_sign(msg_arr, sk);
        secp256k1::detail::secure_erase(&sk, sizeof(sk));
        auto compact = sig.to_compact();
        std::memcpy(sig_out, compact.data(), 32);
        std::memcpy(sig_out + 32, compact.data() + 32, 32);
        sig_out[64] = 0x01; // SIGHASH_ALL type byte
        *sig_len = 65;
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_bip322_verify(
    ufsecp_ctx* ctx,
    const uint8_t* pubkey, size_t pubkey_len,
    ufsecp_bip322_addr_type addr_type,
    const uint8_t* msg, size_t msg_len,
    const uint8_t* sig, size_t sig_len) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey || !sig)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!msg && msg_len > 0)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    static const uint8_t bip322_tag[] = "BIP0322-signed-message";
    uint8_t msg_hash[32];
    {
        auto tag_hash = secp256k1::SHA256::hash(bip322_tag, sizeof(bip322_tag) - 1);
        secp256k1::SHA256 h;
        h.update(tag_hash.data(), 32);
        h.update(tag_hash.data(), 32);
        if (msg && msg_len > 0) h.update(msg, msg_len);
        auto digest = h.finalize();
        std::memcpy(msg_hash, digest.data(), 32);
    }

    try {
    if (addr_type == UFSECP_BIP322_ADDR_P2TR) {
        if (pubkey_len != 32) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "P2TR requires 32-byte x-only pubkey");
        }
        if (sig_len < 64) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "P2TR sig must be at least 64 bytes");
        }
        return ufsecp_schnorr_verify(ctx, msg_hash, sig, pubkey);
    } else {
        if (pubkey_len != 33) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "requires 33-byte compressed pubkey");
        }
        if (sig_len < 64) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "sig must be at least 64 bytes");
        }
        auto pk = point_from_compressed(pubkey);
        if (pk.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
        }
        std::array<uint8_t, 32> msg_arr;
        std::memcpy(msg_arr.data(), msg_hash, 32);
        // Build compact sig from first 64 bytes
        std::array<uint8_t, 64> compact64;
        std::memcpy(compact64.data(), sig, 64);
        auto esig = secp256k1::ECDSASignature::from_compact(compact64);
        if (!esig.is_low_s()) return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "BIP-322 ECDSA high-S (non-BIP-62)");
        bool ok = secp256k1::ecdsa_verify(msg_arr, pk, esig);
        if (!ok) return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "BIP-322 ECDSA verify failed");
        return UFSECP_OK;
    }
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * BIP-157/158 — Compact Block Filters (Golomb-Coded Set)
 * =========================================================================== */

// SipHash-2-4 implementation (for GCS)
namespace {

/* Portable 64×64→high-64 multiply.
 * Provided by secp256k1::detail::mulhi64 (arith64.hpp).
 * Alias it into the anonymous namespace used by the GCS code below. */
using secp256k1::detail::mulhi64;

static inline uint64_t siphash_rotl64(uint64_t x, int b) {
    return (x << b) | (x >> (64 - b));
}

static uint64_t siphash24(const uint8_t key[16], const uint8_t* data, size_t len) {
    uint64_t k0, k1;
    std::memcpy(&k0, key, 8);
    std::memcpy(&k1, key + 8, 8);

    uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = k1 ^ 0x7465646279746573ULL;

    auto sipround = [&]() {
        v0 += v1; v1 = siphash_rotl64(v1, 13); v1 ^= v0; v0 = siphash_rotl64(v0, 32);
        v2 += v3; v3 = siphash_rotl64(v3, 16); v3 ^= v2;
        v0 += v3; v3 = siphash_rotl64(v3, 21); v3 ^= v0;
        v2 += v1; v1 = siphash_rotl64(v1, 17); v1 ^= v2; v2 = siphash_rotl64(v2, 32);
    };

    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t m;
        std::memcpy(&m, data + i, 8);
        v3 ^= m;
        sipround(); sipround();
        v0 ^= m;
    }
    // Remaining bytes + length byte
    uint64_t last = static_cast<uint64_t>(len & 0xff) << 56;
    size_t rem = len - i;
    for (size_t j = 0; j < rem; ++j) last |= static_cast<uint64_t>(data[i + j]) << (j * 8);
    v3 ^= last;
    sipround(); sipround();
    v0 ^= last;
    v2 ^= 0xff;
    sipround(); sipround(); sipround(); sipround();
    return v0 ^ v1 ^ v2 ^ v3;
}

} // anonymous namespace

static constexpr uint64_t GCS_P = 19;
static constexpr uint64_t GCS_M = 784931ULL;

// Encode a GCS filter: sort, delta-encode, Golomb-Rice encode with P=19
static bool gcs_encode(const std::vector<uint64_t>& values,
                        uint8_t* out, size_t* out_len) {
    std::vector<uint64_t> sorted = values;
    std::sort(sorted.begin(), sorted.end());

    // Golomb-Rice encode: quotient (unary) + P-bit remainder
    // Pack bits into bytes
    std::vector<uint8_t> bits;
    bits.reserve(sorted.size() * 4);

    uint64_t prev = 0;
    for (uint64_t v : sorted) {
        uint64_t delta = v - prev;
        prev = v;
        uint64_t q = delta >> GCS_P;
        uint64_t r = delta & ((1ULL << GCS_P) - 1);
        // Write q ones then a zero (unary)
        for (uint64_t i = 0; i < q; ++i) bits.push_back(1);
        bits.push_back(0);
        // Write P-bit remainder, MSB first
        for (int i = static_cast<int>(GCS_P) - 1; i >= 0; --i) {
            bits.push_back(static_cast<uint8_t>((r >> i) & 1));
        }
    }

    // Pack bits into bytes
    size_t nbytes = (bits.size() + 7) / 8;
    if (*out_len < nbytes) return false;
    std::memset(out, 0, nbytes);
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i]) out[i / 8] |= static_cast<uint8_t>(0x80u >> (i % 8));
    }
    *out_len = nbytes;
    return true;
}

static bool gcs_decode(const uint8_t* filter, size_t filter_len, size_t n_items,
                        std::vector<uint64_t>& out) {
    out.clear();
    out.reserve(n_items);

    // Unpack bits
    size_t bit_pos = 0;
    auto read_bit = [&]() -> int {
        if (bit_pos / 8 >= filter_len) return -1;
        return (filter[bit_pos / 8] >> (7 - (bit_pos % 8))) & 1;
    };

    uint64_t prev = 0;
    for (size_t i = 0; i < n_items; ++i) {
        // Read unary quotient
        uint64_t q = 0;
        while (true) {
            int b = read_bit();
            if (b < 0) return false;
            ++bit_pos;
            if (b == 0) break;
            ++q;
        }
        // Read P-bit remainder
        uint64_t r = 0;
        for (size_t j = 0; j < GCS_P; ++j) {
            int b = read_bit();
            if (b < 0) return false;
            ++bit_pos;
            r = (r << 1) | static_cast<uint64_t>(b);
        }
        uint64_t delta = (q << GCS_P) | r;
        prev += delta;
        out.push_back(prev);
    }
    return true;
}

ufsecp_error_t ufsecp_gcs_build(
    const uint8_t key[16],
    const uint8_t** data, const size_t* data_sizes, size_t count,
    uint8_t* filter_out, size_t* filter_len) {
    if (SECP256K1_UNLIKELY(!key || !filter_out || !filter_len)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!data && count > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!data_sizes && count > 0)) return UFSECP_ERR_NULL_ARG;

    try {
    uint64_t const modulus = static_cast<uint64_t>(count) * GCS_M;
    std::vector<uint64_t> hashed;
    hashed.reserve(count);

    for (size_t i = 0; i < count; ++i) {
        uint64_t h = siphash24(key, data[i], data_sizes[i]);
        // Reduce modulo N*M using multiplication technique to avoid bias
        hashed.push_back(mulhi64(h, modulus));
    }

    if (!gcs_encode(hashed, filter_out, filter_len)) {
        return UFSECP_ERR_BUF_TOO_SMALL;
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(nullptr)
}

ufsecp_error_t ufsecp_gcs_match(
    const uint8_t key[16],
    const uint8_t* filter, size_t filter_len,
    size_t n_items,
    const uint8_t* item, size_t item_len) {
    if (SECP256K1_UNLIKELY(!key || !filter || !item)) return UFSECP_ERR_NULL_ARG;

    try {
    std::vector<uint64_t> decoded;
    if (!gcs_decode(filter, filter_len, n_items, decoded)) {
        return UFSECP_ERR_BAD_INPUT;
    }

    uint64_t const modulus = static_cast<uint64_t>(n_items) * GCS_M;
    uint64_t h = siphash24(key, item, item_len);
    uint64_t target = mulhi64(h, modulus);

    for (uint64_t v : decoded) {
        if (v == target) return UFSECP_OK;
        if (v > target) break;
    }
    return UFSECP_ERR_NOT_FOUND;
    } UFSECP_CATCH_RETURN(nullptr)
}

ufsecp_error_t ufsecp_gcs_match_any(
    const uint8_t key[16],
    const uint8_t* filter, size_t filter_len,
    size_t n_items,
    const uint8_t** query, const size_t* query_sizes, size_t query_count) {
    if (SECP256K1_UNLIKELY(!key || !filter)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!query && query_count > 0)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!query_sizes && query_count > 0)) return UFSECP_ERR_NULL_ARG;

    try {
    std::vector<uint64_t> decoded;
    if (!gcs_decode(filter, filter_len, n_items, decoded)) {
        return UFSECP_ERR_BAD_INPUT;
    }

    uint64_t const modulus = static_cast<uint64_t>(n_items) * GCS_M;
    for (size_t qi = 0; qi < query_count; ++qi) {
        /* cppcheck-suppress nullPointer ; query elements guaranteed non-null per API contract */
        uint64_t h = siphash24(key, query[qi], query_sizes[qi]);
        uint64_t target = mulhi64(h, modulus);
        for (uint64_t v : decoded) {
            if (v == target) return UFSECP_OK;
            if (v > target) break;
        }
    }
    return UFSECP_ERR_NOT_FOUND;
    } UFSECP_CATCH_RETURN(nullptr)
}

/* ===========================================================================
 * BIP-174/370 — PSBT Signing Helpers
 * =========================================================================== */

ufsecp_error_t ufsecp_psbt_sign_legacy(
    ufsecp_ctx* ctx,
    const uint8_t sighash32[32],
    const uint8_t privkey[32],
    uint8_t sighash_type,
    uint8_t* sig_out, size_t* sig_len) {
    if (SECP256K1_UNLIKELY(!ctx || !sighash32 || !privkey || !sig_out || !sig_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    ScopeSecureErase<Scalar> sk_erase{&sk, sizeof(sk)}; // erases sk on all exit paths

    if (*sig_len < 73) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "legacy sig buffer too small (need 73)");
    }

    try {
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), sighash32, 32);
    auto sig = secp256k1::ct::ecdsa_sign(msg_arr, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));

    // DER encode: extract r and s from compact encoding
    auto compact = sig.to_compact();
    std::array<uint8_t, 32> r_arr, s_arr;
    std::memcpy(r_arr.data(), compact.data(), 32);
    std::memcpy(s_arr.data(), compact.data() + 32, 32);
    // Build DER manually
    uint8_t der[72];
    size_t der_len = 0;
    // r: add 0x00 prefix if high bit set
    uint8_t r_data[33], s_data[33];
    size_t r_len = 0, s_len = 0;
    // Strip leading zeros from r
    size_t r_start = 0;
    while (r_start < 31 && r_arr[r_start] == 0) ++r_start;
    if (r_arr[r_start] & 0x80) { r_data[0] = 0x00; ++r_len; }
    for (size_t i = r_start; i < 32; ++i) r_data[r_len++] = r_arr[i];
    // Strip leading zeros from s
    size_t s_start = 0;
    while (s_start < 31 && s_arr[s_start] == 0) ++s_start;
    if (s_arr[s_start] & 0x80) { s_data[0] = 0x00; ++s_len; }
    for (size_t i = s_start; i < 32; ++i) s_data[s_len++] = s_arr[i];

    size_t total = 2 + r_len + 2 + s_len;
    der[der_len++] = 0x30;
    der[der_len++] = static_cast<uint8_t>(total);
    der[der_len++] = 0x02;
    der[der_len++] = static_cast<uint8_t>(r_len);
    std::memcpy(der + der_len, r_data, r_len); der_len += r_len;
    der[der_len++] = 0x02;
    der[der_len++] = static_cast<uint8_t>(s_len);
    std::memcpy(der + der_len, s_data, s_len); der_len += s_len;

    if (*sig_len < der_len + 1) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "legacy sig buffer too small");
    }
    std::memcpy(sig_out, der, der_len);
    sig_out[der_len] = sighash_type;
    *sig_len = der_len + 1;
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_psbt_sign_segwit(
    ufsecp_ctx* ctx,
    const uint8_t sighash32[32],
    const uint8_t privkey[32],
    uint8_t sighash_type,
    uint8_t* sig_out, size_t* sig_len) {
    if (SECP256K1_UNLIKELY(!ctx || !sighash32 || !privkey || !sig_out || !sig_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (*sig_len < 65) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "segwit sig buffer too small (need 65)");
    }

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    ScopeSecureErase<Scalar> sk_erase{&sk, sizeof(sk)}; // erases sk on all exit paths

    try {
    std::array<uint8_t, 32> msg_arr;
    std::memcpy(msg_arr.data(), sighash32, 32);
    auto sig = secp256k1::ct::ecdsa_sign(msg_arr, sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    auto compact = sig.to_compact();
    std::memcpy(sig_out, compact.data(), 32);
    std::memcpy(sig_out + 32, compact.data() + 32, 32);
    sig_out[64] = sighash_type;
    *sig_len = 65;
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_psbt_sign_taproot(
    ufsecp_ctx* ctx,
    const uint8_t sighash32[32],
    const uint8_t privkey[32],
    uint8_t sighash_type,
    const uint8_t* aux_rand32,
    uint8_t* sig_out, size_t* sig_len) {
    if (SECP256K1_UNLIKELY(!ctx || !sighash32 || !privkey || !sig_out || !sig_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    size_t const expected_len = (sighash_type == UFSECP_SIGHASH_DEFAULT) ? 64 : 65;
    if (*sig_len < expected_len) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "taproot sig buffer too small");
    }

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    ScopeSecureErase<Scalar> sk_erase{&sk, sizeof(sk)}; // erases sk on all exit paths

    try {
    std::array<uint8_t, 32> msg_arr;
    std::array<uint8_t, 32> aux_arr{};
    std::memcpy(msg_arr.data(), sighash32, 32);
    if (aux_rand32) std::memcpy(aux_arr.data(), aux_rand32, 32);

    auto kp = secp256k1::ct::schnorr_keypair_create(sk);
    ScopeSecureErase<decltype(kp.d)> kp_d_erase{&kp.d, sizeof(kp.d)}; // erases kp.d on all exit paths
    auto sig = secp256k1::ct::schnorr_sign(kp, msg_arr, aux_arr);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&kp.d, sizeof(kp.d));

    auto bytes = sig.to_bytes();
    std::memcpy(sig_out, bytes.data(), 64);
    if (sighash_type != UFSECP_SIGHASH_DEFAULT) {
        sig_out[64] = sighash_type;
        *sig_len = 65;
    } else {
        *sig_len = 64;
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_psbt_derive_key(
    ufsecp_ctx* ctx,
    const ufsecp_bip32_key* master_xprv,
    const char* key_path,
    uint8_t privkey_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !master_xprv || !key_path || !privkey_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    ufsecp_bip32_key derived{};
    ufsecp_error_t rc = ufsecp_bip32_derive_path(ctx, master_xprv, key_path, &derived);
    if (rc != UFSECP_OK) return rc;

    return ufsecp_bip32_privkey(ctx, &derived, privkey_out);
}

/* ===========================================================================
 * BIP-380..386 — Output Descriptors (key expression parser)
 * =========================================================================== */

ufsecp_error_t ufsecp_descriptor_parse(
    ufsecp_ctx* ctx,
    const char* descriptor,
    uint32_t index,
    ufsecp_desc_key* key_out,
    char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !descriptor || !key_out)) return UFSECP_ERR_NULL_ARG;
    if (addr_out && !addr_len) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::string desc(descriptor);

    try {
    // Determine descriptor type by prefix
    ufsecp_desc_type dtype = UFSECP_DESC_WPKH;
    std::string inner;
    std::string outer_func;

    auto strip_func = [](const std::string& s, const std::string& fname, std::string& inner_out) -> bool {
        if (s.size() <= fname.size() + 2) return false;
        if (s.substr(0, fname.size() + 1) != fname + "(") return false;
        if (s.back() != ')') return false;
        inner_out = s.substr(fname.size() + 1, s.size() - fname.size() - 2);
        return true;
    };

    if (strip_func(desc, "wpkh", inner)) {
        dtype = UFSECP_DESC_WPKH;
    } else if (desc.substr(0, 8) == "sh(wpkh(" && desc.back() == ')' &&
               desc[desc.size()-2] == ')') {
        dtype = UFSECP_DESC_SH_WPKH;
        inner = desc.substr(8, desc.size() - 10); // strip sh(wpkh( and ))
    } else if (strip_func(desc, "tr", inner)) {
        dtype = UFSECP_DESC_TR;
    } else if (strip_func(desc, "pkh", inner)) {
        dtype = UFSECP_DESC_PKH;
    } else if (strip_func(desc, "pk", inner)) {
        dtype = UFSECP_DESC_PK;
    } else {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unsupported descriptor type");
    }

    // Validate no extra ')' in inner
    // Find path suffix: inner may be "xpub.../path" or "xpub.../path/*"
    // or just a raw hex pubkey
    key_out->type = dtype;
    key_out->network = UFSECP_NET_MAINNET;
    key_out->path[0] = '\0';

    // Find the key part: if it starts with xpub/xprv, parse BIP-32
    std::string key_str, path_suffix;
    auto slash_pos = inner.find('/');
    if (slash_pos != std::string::npos) {
        key_str = inner.substr(0, slash_pos);
        path_suffix = inner.substr(slash_pos);
    } else {
        key_str = inner;
    }

    // Validate path_suffix - reject non-numeric characters besides / and '*
    for (char c : path_suffix) {
        if (c != '/' && c != '*' && c != '\'' && !std::isdigit((unsigned char)c) && c != ';' && c != '<' && c != '>') {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid path characters");
        }
    }

    uint8_t pubkey33[33]{};
    int net = UFSECP_NET_MAINNET;

    if (key_str.size() == 111 || key_str.size() == 112 || // xpub
        (key_str.substr(0,4) == "xpub" || key_str.substr(0,4) == "xprv" ||
         key_str.substr(0,4) == "tpub" || key_str.substr(0,4) == "tprv")) {
        // BIP-32 extended key
        if (key_str[0] == 't') net = UFSECP_NET_TESTNET;

        // Decode the xpub/xprv using base58check
        auto [data, valid] = secp256k1::base58check_decode(key_str);
        if (!valid || data.size() < 78) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid xpub/xprv in descriptor");
        }

        // Build ufsecp_bip32_key
        ufsecp_bip32_key bip32k{};
        std::memcpy(bip32k.data, data.data(), 78);
        // Determine if private
        uint32_t ver = (uint32_t(data[0]) << 24) | (uint32_t(data[1]) << 16) |
                       (uint32_t(data[2]) << 8) | uint32_t(data[3]);
        bip32k.is_private = (ver == 0x0488ADE4u || ver == 0x04358394u) ? 1 : 0;

        // Derive path + index
        std::string derive_path = "m";
        if (!path_suffix.empty()) {
            // Replace '*' with index, handle <a;b> range as index selection
            std::string ps = path_suffix;
            // Handle <0;1> style
            auto ab = ps.find('<');
            if (ab != std::string::npos) {
                auto ae = ps.find('>', ab);
                if (ae != std::string::npos) ps.replace(ab, ae - ab + 1, "0");
            }
            // Replace * with index
            auto star = ps.find('*');
            if (star != std::string::npos) {
                ps.replace(star, 1, std::to_string(index));
            }
            derive_path += ps;
        }

        ufsecp_bip32_key derived{};
        if (derive_path != "m") {
            ufsecp_ctx* dummy = ctx;
            ufsecp_error_t rc = ufsecp_bip32_derive_path(dummy, &bip32k, derive_path.c_str(), &derived);
            if (rc != UFSECP_OK) return rc;
        } else {
            derived = bip32k;
        }

        // Extract pubkey
        ufsecp_error_t rc = ufsecp_bip32_pubkey(ctx, &derived, pubkey33);
        if (rc != UFSECP_OK) return rc;
    } else if (key_str.size() == 64 || key_str.size() == 66) {
        // Raw hex pubkey
        if (key_str.size() % 2 != 0) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "odd hex length in descriptor");
        }
        size_t expected_bytes = key_str.size() / 2;
        if (expected_bytes != 33 && expected_bytes != 32) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "pubkey must be 33 or 32 bytes");
        }
        for (size_t i = 0; i < key_str.size(); i += 2) {
            unsigned int byte_val = 0;
            // Manual hex parse
            auto hexchar = [](char c) -> int {
                if (c >= '0' && c <= '9') return c - '0';
                if (c >= 'a' && c <= 'f') return c - 'a' + 10;
                if (c >= 'A' && c <= 'F') return c - 'A' + 10;
                return -1;
            };
            int hi = hexchar(key_str[i]), lo = hexchar(key_str[i+1]);
            if (hi < 0 || lo < 0) {
                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid hex in descriptor key");
            }
            byte_val = static_cast<unsigned int>(hi * 16 + lo);
            pubkey33[i / 2] = static_cast<uint8_t>(byte_val);
        }
    } else {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unrecognized key format in descriptor");
    }

    // Fill key_out
    if (dtype == UFSECP_DESC_TR) {
        // x-only pubkey (skip prefix byte if 33 bytes)
        if (pubkey33[0] == 0x02 || pubkey33[0] == 0x03) {
            std::memcpy(key_out->pubkey, pubkey33 + 1, 32);
            key_out->pubkey_len = 32;
        } else {
            std::memcpy(key_out->pubkey, pubkey33, 32);
            key_out->pubkey_len = 32;
        }
    } else {
        std::memcpy(key_out->pubkey, pubkey33, 33);
        key_out->pubkey_len = 33;
    }
    key_out->network = net;
    if (path_suffix.size() < sizeof(key_out->path)) {
        std::memcpy(key_out->path, path_suffix.c_str(), path_suffix.size() + 1);
    } else {
        key_out->path[0] = '\0';
    }

    // Generate address if requested
    if (addr_out && addr_len) {
        switch (dtype) {
        case UFSECP_DESC_WPKH:
            return ufsecp_addr_p2wpkh(ctx, pubkey33, net, addr_out, addr_len);
        case UFSECP_DESC_PKH:
        case UFSECP_DESC_PK:
            return ufsecp_addr_p2pkh(ctx, pubkey33, net, addr_out, addr_len);
        case UFSECP_DESC_TR: {
            uint8_t xonly[32];
            if (pubkey33[0] == 0x02 || pubkey33[0] == 0x03) {
                std::memcpy(xonly, pubkey33 + 1, 32);
            } else {
                std::memcpy(xonly, pubkey33, 32);
            }
            return ufsecp_addr_p2tr(ctx, xonly, net, addr_out, addr_len);
        }
        case UFSECP_DESC_SH_WPKH:
            return ufsecp_addr_p2sh_p2wpkh(ctx, pubkey33, net, addr_out, addr_len);
        default:
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "unknown descriptor type for address");
        }
    }

    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_descriptor_address(
    ufsecp_ctx* ctx,
    const char* descriptor,
    uint32_t index,
    char* addr_out, size_t* addr_len) {
    if (SECP256K1_UNLIKELY(!ctx || !descriptor || !addr_out || !addr_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    ufsecp_desc_key key_out{};
    return ufsecp_descriptor_parse(ctx, descriptor, index, &key_out, addr_out, addr_len);
}
