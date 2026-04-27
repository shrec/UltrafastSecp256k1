/* ============================================================================
 * Context, version, error, seckey and pubkey operations
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

unsigned int ufsecp_version(void) {
    return UFSECP_VERSION_PACKED;
}

unsigned int ufsecp_abi_version(void) {
    return UFSECP_ABI_VERSION;
}

const char* ufsecp_version_string(void) {
    return UFSECP_VERSION_STRING;
}

const char* ufsecp_error_str(ufsecp_error_t err) {
    switch (err) {
    case UFSECP_OK:                return "OK";
    case UFSECP_ERR_NULL_ARG:      return "NULL argument";
    case UFSECP_ERR_BAD_KEY:       return "invalid private key";
    case UFSECP_ERR_BAD_PUBKEY:    return "invalid public key";
    case UFSECP_ERR_BAD_SIG:       return "invalid signature";
    case UFSECP_ERR_BAD_INPUT:     return "malformed input";
    case UFSECP_ERR_VERIFY_FAIL:   return "verification failed";
    case UFSECP_ERR_ARITH:         return "arithmetic error";
    case UFSECP_ERR_SELFTEST:      return "self-test failed";
    case UFSECP_ERR_INTERNAL:      return "internal error";
    case UFSECP_ERR_BUF_TOO_SMALL: return "buffer too small";
    default:                       return "unknown error";
    }
}

/* ===========================================================================
 * Context lifecycle
 * =========================================================================== */

ufsecp_error_t ufsecp_ctx_create(ufsecp_ctx** ctx_out) {
    if (!ctx_out) return UFSECP_ERR_NULL_ARG;
    *ctx_out = nullptr;

    auto* ctx = static_cast<ufsecp_ctx*>(std::calloc(1, sizeof(ufsecp_ctx)));
    if (!ctx) return UFSECP_ERR_INTERNAL;

    ctx->last_err   = UFSECP_OK;
    ctx->last_msg[0] = '\0';

    /* Run selftest once (cached globally by ensure_library_integrity) */
    ctx->selftest_ok = secp256k1::fast::ensure_library_integrity(false);
    if (!ctx->selftest_ok) {
        std::free(ctx);
        return UFSECP_ERR_SELFTEST;
    }

    *ctx_out = ctx;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_ctx_clone(const ufsecp_ctx* src, ufsecp_ctx** ctx_out) {
    if (!src || !ctx_out) return UFSECP_ERR_NULL_ARG;
    *ctx_out = nullptr;

    auto* dst = static_cast<ufsecp_ctx*>(std::malloc(sizeof(ufsecp_ctx)));
    if (!dst) return UFSECP_ERR_INTERNAL;

    std::memcpy(dst, src, sizeof(ufsecp_ctx));
    ctx_clear_err(dst);

    *ctx_out = dst;
    return UFSECP_OK;
}

void ufsecp_ctx_destroy(ufsecp_ctx* ctx) {
    std::free(ctx);  // free(NULL) is a no-op per C standard
}

ufsecp_error_t ufsecp_last_error(const ufsecp_ctx* ctx) {
    return ctx ? ctx->last_err : UFSECP_ERR_NULL_ARG;
}

const char* ufsecp_last_error_msg(const ufsecp_ctx* ctx) {
    if (!ctx) return "NULL context";
    return ctx->last_msg[0] ? ctx->last_msg : ufsecp_error_str(ctx->last_err);
}

size_t ufsecp_ctx_size(void) {
    return sizeof(ufsecp_ctx);
}

/* ===========================================================================
 * Private key utilities
 * =========================================================================== */

ufsecp_error_t ufsecp_seckey_verify(const ufsecp_ctx* ctx,
                                    const uint8_t privkey[32]) {
    if (!ctx || !privkey) return UFSECP_ERR_NULL_ARG;
    // BIP-340 strict: reject if privkey == 0 or privkey >= n (no reduction)
    Scalar sk;
    if (!Scalar::parse_bytes_strict_nonzero(privkey, sk)) {
        return UFSECP_ERR_BAD_KEY;
    }
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_seckey_negate(ufsecp_ctx* ctx, uint8_t privkey[32]) {
    if (!ctx || !privkey) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    auto neg = sk.negate();
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    // negate of valid nonzero scalar is always nonzero
    scalar_to_bytes(neg, privkey);
    secp256k1::detail::secure_erase(&neg, sizeof(neg));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_seckey_tweak_add(ufsecp_ctx* ctx, uint8_t privkey[32],
                                       const uint8_t tweak[32]) {
    if (!ctx || !privkey || !tweak) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    Scalar tw;
    if (!scalar_parse_strict(tweak, tw)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "tweak >= n");
    }
    auto result = sk + tw;
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&tw, sizeof(tw));
    if (result.is_zero()) {
        secp256k1::detail::secure_erase(&result, sizeof(result));
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_add resulted in zero");
    }
    scalar_to_bytes(result, privkey);
    secp256k1::detail::secure_erase(&result, sizeof(result));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_seckey_tweak_mul(ufsecp_ctx* ctx, uint8_t privkey[32],
                                       const uint8_t tweak[32]) {
    if (!ctx || !privkey || !tweak) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    Scalar tw;
    // tweak_mul: reject tweak==0 (result would be zero) and tweak >= n
    if (!scalar_parse_strict_nonzero(tweak, tw)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "tweak is zero or >= n");
    }
    auto result = sk * tw;
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    secp256k1::detail::secure_erase(&tw, sizeof(tw));
    if (result.is_zero()) {
        secp256k1::detail::secure_erase(&result, sizeof(result));
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_mul resulted in zero");
    }
    scalar_to_bytes(result, privkey);
    secp256k1::detail::secure_erase(&result, sizeof(result));
    return UFSECP_OK;
}

/* ===========================================================================
 * Public key
 * =========================================================================== */

static ufsecp_error_t pubkey_create_core(ufsecp_ctx* ctx,
                                         const uint8_t privkey[32],
                                         Point& pk_out) {
    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    pk_out = secp256k1::ct::generator_mul(sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (pk_out.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "pubkey at infinity");
    }
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_create(ufsecp_ctx* ctx,
                                    const uint8_t privkey[32],
                                    uint8_t pubkey33_out[33]) {
    if (!ctx || !privkey || !pubkey33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Point pk;
    const ufsecp_error_t err = pubkey_create_core(ctx, privkey, pk);
    if (err != UFSECP_OK) return err;
    point_to_compressed(pk, pubkey33_out);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_create_uncompressed(ufsecp_ctx* ctx,
                                                 const uint8_t privkey[32],
                                                 uint8_t pubkey65_out[65]) {
    if (!ctx || !privkey || !pubkey65_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Point pk;
    const ufsecp_error_t err = pubkey_create_core(ctx, privkey, pk);
    if (err != UFSECP_OK) return err;
    auto uncomp = pk.to_uncompressed();
    std::memcpy(pubkey65_out, uncomp.data(), 65);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_parse(ufsecp_ctx* ctx,
                                   const uint8_t* input, size_t input_len,
                                   uint8_t pubkey33_out[33]) {
    if (!ctx || !input || !pubkey33_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    if (input_len == 33 && (input[0] == 0x02 || input[0] == 0x03)) {
        auto p = point_from_compressed(input);
        if (p.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "decompression failed");
}
        point_to_compressed(p, pubkey33_out);
        return UFSECP_OK;
    }
    if (input_len == 65 && input[0] == 0x04) {
        std::array<uint8_t, 32> x_bytes, y_bytes;
        std::memcpy(x_bytes.data(), input + 1, 32);
        std::memcpy(y_bytes.data(), input + 33, 32);
        // Strict: reject x >= p or y >= p
        FE x, y;
        if (!FE::parse_bytes_strict(x_bytes, x) ||
            !FE::parse_bytes_strict(y_bytes, y)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "coordinate >= p");
        }
        // On-curve check: y^2 == x^3 + 7
        auto lhs = y * y;
        auto rhs = x * x * x + FE::from_uint64(7);
        if (lhs != rhs) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "point not on curve");
        }
        auto p = Point::from_affine(x, y);
        if (p.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "point at infinity");
}
        point_to_compressed(p, pubkey33_out);
        return UFSECP_OK;
    }
    return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "expected 33 or 65 byte pubkey");
}

ufsecp_error_t ufsecp_pubkey_xonly(ufsecp_ctx* ctx,
                                   const uint8_t privkey[32],
                                   uint8_t xonly32_out[32]) {
    if (!ctx || !privkey || !xonly32_out) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (!scalar_parse_strict_nonzero(privkey, sk)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }

    auto xonly = secp256k1::schnorr_pubkey(sk);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    std::memcpy(xonly32_out, xonly.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * ECDSA
 * =========================================================================== */

