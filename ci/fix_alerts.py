#!/usr/bin/env python3
"""Apply all readability-braces and misc-const-correctness fixes to ufsecp_impl.cpp"""

import sys

PATH = "include/ufsecp/ufsecp_impl.cpp"

# Each entry: (old_string, new_string)
REPLACEMENTS = [
    # L1380: if (!ok)
    (
        "    if (!ok)\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid mnemonic\");",
        "    if (!ok) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid mnemonic\");\n"
        "    }",
    ),
    # L1382: if (*entropy_len < ent.length)
    (
        "    if (*entropy_len < ent.length)\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, \"entropy buffer too small\");",
        "    if (*entropy_len < ent.length) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, \"entropy buffer too small\");\n"
        "    }",
    ),
    # L1404: if (!FE::parse_bytes_strict(e, pk_fe)) in ufsecp_schnorr_batch_verify
    (
        "    for (size_t i = 0; i < n; ++i) {\n"
        "        const uint8_t* e = entries + i * 128;\n"
        "        // Strict: reject x-only pubkey >= p at ABI gate\n"
        "        FE pk_fe;\n"
        "        if (!FE::parse_bytes_strict(e, pk_fe))\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"non-canonical pubkey (x>=p) in batch\");\n"
        "        std::memcpy(batch[i].pubkey_x.data(), e, 32);\n"
        "        std::memcpy(batch[i].message.data(), e + 32, 32);\n"
        "        if (!secp256k1::SchnorrSignature::parse_strict(e + 64, batch[i].signature))\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid Schnorr sig in batch\");",
        "    for (size_t i = 0; i < n; ++i) {\n"
        "        const uint8_t* e = entries + i * 128;\n"
        "        // Strict: reject x-only pubkey >= p at ABI gate\n"
        "        FE pk_fe;\n"
        "        if (!FE::parse_bytes_strict(e, pk_fe)) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"non-canonical pubkey (x>=p) in batch\");\n"
        "        }\n"
        "        std::memcpy(batch[i].pubkey_x.data(), e, 32);\n"
        "        std::memcpy(batch[i].message.data(), e + 32, 32);\n"
        "        if (!secp256k1::SchnorrSignature::parse_strict(e + 64, batch[i].signature)) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid Schnorr sig in batch\");\n"
        "        }",
    ),
    # L1448: if (!FE::parse_bytes_strict(e, pk_fe)) in ufsecp_schnorr_batch_identify_invalid
    (
        "    for (size_t i = 0; i < n; ++i) {\n"
        "        const uint8_t* e = entries + i * 128;\n"
        "        FE pk_fe;\n"
        "        if (!FE::parse_bytes_strict(e, pk_fe))\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"non-canonical pubkey (x>=p) in batch\");\n"
        "        std::memcpy(batch[i].pubkey_x.data(), e, 32);\n"
        "        std::memcpy(batch[i].message.data(), e + 32, 32);\n"
        "        if (!secp256k1::SchnorrSignature::parse_strict(e + 64, batch[i].signature))\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid Schnorr sig in batch\");",
        "    for (size_t i = 0; i < n; ++i) {\n"
        "        const uint8_t* e = entries + i * 128;\n"
        "        FE pk_fe;\n"
        "        if (!FE::parse_bytes_strict(e, pk_fe)) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"non-canonical pubkey (x>=p) in batch\");\n"
        "        }\n"
        "        std::memcpy(batch[i].pubkey_x.data(), e, 32);\n"
        "        std::memcpy(batch[i].message.data(), e + 32, 32);\n"
        "        if (!secp256k1::SchnorrSignature::parse_strict(e + 64, batch[i].signature)) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid Schnorr sig in batch\");\n"
        "        }",
    ),
    # L1661: widening + braces — in ufsecp_musig2_start_sign_session
    (
        "    for (uint32_t i = 0; i < nk && (38u + (i+1)*32u <= UFSECP_MUSIG2_KEYAGG_LEN); ++i) {\n"
        "        Scalar s;\n"
        "        if (!scalar_parse_strict(keyagg + 38 + i * 32, s))\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid key coefficient in keyagg\");\n"
        "        kagg.key_coefficients.push_back(s);\n"
        "    }",
        "    for (uint32_t i = 0; i < nk && (38u + (i+1)*32u <= UFSECP_MUSIG2_KEYAGG_LEN); ++i) {\n"
        "        Scalar s;\n"
        "        if (!scalar_parse_strict(keyagg + 38 + static_cast<size_t>(i) * 32, s)) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid key coefficient in keyagg\");\n"
        "        }\n"
        "        kagg.key_coefficients.push_back(s);\n"
        "    }",
    ),
    # L1707: widening + braces — inside { } block in ufsecp_musig2_partial_sign
    (
        "        auto qc = kagg.Q.to_compressed(); std::memcpy(kagg.Q_x.data(), qc.data() + 1, 32);\n"
        "        for (uint32_t i = 0; i < nk && (38u + (i+1)*32u <= UFSECP_MUSIG2_KEYAGG_LEN); ++i) {\n"
        "            Scalar s; if (!scalar_parse_strict(keyagg + 38 + i * 32, s))\n"
        "                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid key coefficient\");\n"
        "            kagg.key_coefficients.push_back(s); } }\n"
        "      secp256k1::MuSig2Session sess;",
        "        auto qc = kagg.Q.to_compressed(); std::memcpy(kagg.Q_x.data(), qc.data() + 1, 32);\n"
        "        for (uint32_t i = 0; i < nk && (38u + (i+1)*32u <= UFSECP_MUSIG2_KEYAGG_LEN); ++i) {\n"
        "            Scalar s;\n"
        "            if (!scalar_parse_strict(keyagg + 38 + static_cast<size_t>(i) * 32, s)) {\n"
        "                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid key coefficient\");\n"
        "            }\n"
        "            kagg.key_coefficients.push_back(s);\n"
        "        }\n"
        "    }\n"
        "      secp256k1::MuSig2Session sess;",
    ),
    # L1715: if (!scalar_parse_strict(session + 33, sess.b)) — in ufsecp_musig2_partial_sign
    (
        "      if (!scalar_parse_strict(session + 33, sess.b))\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar b\");\n"
        "      if (!scalar_parse_strict(session + 65, sess.e))\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar e\");\n"
        "      sess.R_negated = (session[97] != 0);\n"
        "      auto psig = secp256k1::musig2_partial_sign",
        "      if (!scalar_parse_strict(session + 33, sess.b)) {\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar b\");\n"
        "      }\n"
        "      if (!scalar_parse_strict(session + 65, sess.e)) {\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar e\");\n"
        "      }\n"
        "      sess.R_negated = (session[97] != 0);\n"
        "      auto psig = secp256k1::musig2_partial_sign",
    ),
    # L1756: widening + braces — inside { } block in ufsecp_musig2_partial_verify
    (
        "        auto qc = kagg.Q.to_compressed(); std::memcpy(kagg.Q_x.data(), qc.data() + 1, 32);\n"
        "        for (uint32_t i = 0; i < nk && (38u + (i+1)*32u <= UFSECP_MUSIG2_KEYAGG_LEN); ++i) {\n"
        "            Scalar s; if (!scalar_parse_strict(keyagg + 38 + i * 32, s))\n"
        "                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid key coefficient\");\n"
        "            kagg.key_coefficients.push_back(s); } }\n"
        "      secp256k1::MuSig2Session sess;\n"
        "      sess.R = point_from_compressed(session);\n"
        "      if (sess.R.is_infinity()) {\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session R point\");\n"
        "      }\n"
        "      if (!scalar_parse_strict(session + 33, sess.b))\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar b\");\n"
        "      if (!scalar_parse_strict(session + 65, sess.e))\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar e\");\n"
        "      sess.R_negated = (session[97] != 0);\n"
        "      if (!secp256k1::musig2_partial_verify",
        "        auto qc = kagg.Q.to_compressed(); std::memcpy(kagg.Q_x.data(), qc.data() + 1, 32);\n"
        "        for (uint32_t i = 0; i < nk && (38u + (i+1)*32u <= UFSECP_MUSIG2_KEYAGG_LEN); ++i) {\n"
        "            Scalar s;\n"
        "            if (!scalar_parse_strict(keyagg + 38 + static_cast<size_t>(i) * 32, s)) {\n"
        "                return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid key coefficient\");\n"
        "            }\n"
        "            kagg.key_coefficients.push_back(s);\n"
        "        }\n"
        "    }\n"
        "      secp256k1::MuSig2Session sess;\n"
        "      sess.R = point_from_compressed(session);\n"
        "      if (sess.R.is_infinity()) {\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session R point\");\n"
        "      }\n"
        "      if (!scalar_parse_strict(session + 33, sess.b)) {\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar b\");\n"
        "      }\n"
        "      if (!scalar_parse_strict(session + 65, sess.e)) {\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar e\");\n"
        "      }\n"
        "      sess.R_negated = (session[97] != 0);\n"
        "      if (!secp256k1::musig2_partial_verify",
    ),
    # L1791+L1793: in ufsecp_musig2_partial_sig_agg
    (
        "      if (!scalar_parse_strict(session + 33, sess.b))\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar b\");\n"
        "      if (!scalar_parse_strict(session + 65, sess.e))\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar e\");\n"
        "      sess.R_negated = (session[97] != 0);\n"
        "      auto final_sig",
        "      if (!scalar_parse_strict(session + 33, sess.b)) {\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar b\");\n"
        "      }\n"
        "      if (!scalar_parse_strict(session + 65, sess.e)) {\n"
        "          return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid session scalar e\");\n"
        "      }\n"
        "      sess.R_negated = (session[97] != 0);\n"
        "      auto final_sig",
    ),
    # L1822+L1823: const for coeff_count and needed_commits
    (
        "    size_t coeff_count = commit.coeffs.size();\n"
        "    size_t needed_commits = 8 + coeff_count * 33;",
        "    const size_t coeff_count = commit.coeffs.size();\n"
        "    const size_t needed_commits = 8 + coeff_count * 33;",
    ),
    # L1845: for (auto& s : shares) — erase in ufsecp_frost_keygen_begin
    (
        "    // Erase secret shares from memory\n"
        "    for (auto& s : shares)\n"
        "        secp256k1::detail::secure_erase(&s.value, sizeof(s.value));\n"
        "    return UFSECP_OK;\n"
        "}\n"
        "\n"
        "ufsecp_error_t ufsecp_frost_keygen_finalize(",
        "    // Erase secret shares from memory\n"
        "    for (auto& s : shares) {\n"
        "        secp256k1::detail::secure_erase(&s.value, sizeof(s.value));\n"
        "    }\n"
        "    return UFSECP_OK;\n"
        "}\n"
        "\n"
        "ufsecp_error_t ufsecp_frost_keygen_finalize(",
    ),
    # L1864: uint32_t cc; — init-variables
    (
        "        secp256k1::FrostCommitment fc;\n"
        "        uint32_t cc;\n"
        "        if (pos + 8 > commits_len)\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"truncated commit header\");\n"
        "        std::memcpy(&cc, all_commits + pos, 4); pos += 4;\n"
        "        std::memcpy(&fc.from, all_commits + pos, 4); pos += 4;\n"
        "        if (pos + static_cast<size_t>(cc) * 33 > commits_len)\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"truncated commit coefficients\");\n"
        "        for (uint32_t j = 0; j < cc; ++j) {\n"
        "            auto pt = point_from_compressed(all_commits + pos);",
        "        secp256k1::FrostCommitment fc;\n"
        "        uint32_t cc = 0;\n"
        "        if (pos + 8 > commits_len) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"truncated commit header\");\n"
        "        }\n"
        "        std::memcpy(&cc, all_commits + pos, 4); pos += 4;\n"
        "        std::memcpy(&fc.from, all_commits + pos, 4); pos += 4;\n"
        "        if (pos + static_cast<size_t>(cc) * 33 > commits_len) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"truncated commit coefficients\");\n"
        "        }\n"
        "        for (uint32_t j = 0; j < cc; ++j) {\n"
        "            auto pt = point_from_compressed(all_commits + pos);",
    ),
    # L1889: if (!scalar_parse_strict(s + 4, v)) in ufsecp_frost_keygen_finalize
    (
        "        if (!scalar_parse_strict(s + 4, v))\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid share scalar\");",
        "        if (!scalar_parse_strict(s + 4, v)) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid share scalar\");\n"
        "        }",
    ),
    # L1895+L1898: if (!ok) + for (auto& s : shares) — erase in ufsecp_frost_keygen_finalize
    (
        "    if (!ok)\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, \"FROST keygen finalize failed\");\n"
        "    // Erase secret shares\n"
        "    for (auto& s : shares)\n"
        "        secp256k1::detail::secure_erase(&s.value, sizeof(s.value));",
        "    if (!ok) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, \"FROST keygen finalize failed\");\n"
        "    }\n"
        "    // Erase secret shares\n"
        "    for (auto& s : shares) {\n"
        "        secp256k1::detail::secure_erase(&s.value, sizeof(s.value));\n"
        "    }",
    ),
    # L1955: if (!scalar_parse_strict(keypkg + 12, kp.signing_share))
    (
        "    if (!scalar_parse_strict(keypkg + 12, kp.signing_share))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"invalid signing share in keypkg\");",
        "    if (!scalar_parse_strict(keypkg + 12, kp.signing_share)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"invalid signing share in keypkg\");\n"
        "    }",
    ),
    # L1967+L1969: if (!scalar_parse_strict(nonce, h)) + if (!scalar_parse_strict(nonce + 32, b))
    (
        "    if (!scalar_parse_strict(nonce, h))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid hiding nonce\");\n"
        "    if (!scalar_parse_strict(nonce + 32, b))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid binding nonce\");",
        "    if (!scalar_parse_strict(nonce, h)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid hiding nonce\");\n"
        "    }\n"
        "    if (!scalar_parse_strict(nonce + 32, b)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"invalid binding nonce\");\n"
        "    }",
    ),
    # L2006+L2012: multi-line if null check + scalar parse in ufsecp_frost_verify_partial
    (
        "    if (!ctx || !partial_sig || !verification_share33 || !nonce_commits || !msg32 || !group_pubkey33)\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    ctx_clear_err(ctx);\n"
        "    secp256k1::FrostPartialSig psig;\n"
        "    std::memcpy(&psig.id, partial_sig, 4);\n"
        "    Scalar z;\n"
        "    if (!scalar_parse_strict(partial_sig + 4, z))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid partial sig scalar\");",
        "    if (!ctx || !partial_sig || !verification_share33 || !nonce_commits || !msg32 || !group_pubkey33) {\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    }\n"
        "    ctx_clear_err(ctx);\n"
        "    secp256k1::FrostPartialSig psig;\n"
        "    std::memcpy(&psig.id, partial_sig, 4);\n"
        "    Scalar z;\n"
        "    if (!scalar_parse_strict(partial_sig + 4, z)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid partial sig scalar\");\n"
        "    }",
    ),
    # L2057+L2065: multi-line if null check + scalar parse in ufsecp_frost_aggregate
    (
        "    if (!ctx || !partial_sigs || !nonce_commits || !group_pubkey33 || !msg32 || !sig64_out)\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    ctx_clear_err(ctx);\n"
        "    std::vector<secp256k1::FrostPartialSig> psigs(n);\n"
        "    for (size_t i = 0; i < n; ++i) {\n"
        "        const uint8_t* ps = partial_sigs + i * 36;\n"
        "        std::memcpy(&psigs[i].id, ps, 4);\n"
        "        Scalar z;\n"
        "        if (!scalar_parse_strict(ps + 4, z))\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid partial sig scalar\");",
        "    if (!ctx || !partial_sigs || !nonce_commits || !group_pubkey33 || !msg32 || !sig64_out) {\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    }\n"
        "    ctx_clear_err(ctx);\n"
        "    std::vector<secp256k1::FrostPartialSig> psigs(n);\n"
        "    for (size_t i = 0; i < n; ++i) {\n"
        "        const uint8_t* ps = partial_sigs + i * 36;\n"
        "        std::memcpy(&psigs[i].id, ps, 4);\n"
        "        Scalar z;\n"
        "        if (!scalar_parse_strict(ps + 4, z)) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid partial sig scalar\");\n"
        "        }",
    ),
    # L2144+L2150: in ufsecp_schnorr_adaptor_verify
    (
        "    if (!scalar_parse_strict(pre_sig + 33, shat))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid adaptor sig scalar\");\n"
        "    as.s_hat = shat;\n"
        "    as.needs_negation = (pre_sig[65] != 0);\n"
        "    // Strict: reject x-only pubkey >= p at ABI gate\n"
        "    FE pk_fe;\n"
        "    if (!FE::parse_bytes_strict(pubkey_x, pk_fe))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"non-canonical pubkey (x>=p)\");",
        "    if (!scalar_parse_strict(pre_sig + 33, shat)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid adaptor sig scalar\");\n"
        "    }\n"
        "    as.s_hat = shat;\n"
        "    as.needs_negation = (pre_sig[65] != 0);\n"
        "    // Strict: reject x-only pubkey >= p at ABI gate\n"
        "    FE pk_fe;\n"
        "    if (!FE::parse_bytes_strict(pubkey_x, pk_fe)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"non-canonical pubkey (x>=p)\");\n"
        "    }",
    ),
    # L2176: in ufsecp_schnorr_adaptor_adapt
    (
        "    if (!scalar_parse_strict(pre_sig + 33, shat))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid adaptor sig scalar\");\n"
        "    as.s_hat = shat;\n"
        "    as.needs_negation = (pre_sig[65] != 0);\n"
        "    Scalar secret;\n"
        "    if (!scalar_parse_strict_nonzero(adaptor_secret, secret))",
        "    if (!scalar_parse_strict(pre_sig + 33, shat)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid adaptor sig scalar\");\n"
        "    }\n"
        "    as.s_hat = shat;\n"
        "    as.needs_negation = (pre_sig[65] != 0);\n"
        "    Scalar secret;\n"
        "    if (!scalar_parse_strict_nonzero(adaptor_secret, secret))",
    ),
    # L2203: in ufsecp_schnorr_adaptor_extract
    (
        "    if (!scalar_parse_strict(pre_sig + 33, shat))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid adaptor sig scalar\");\n"
        "    as.s_hat = shat;\n"
        "    as.needs_negation = (pre_sig[65] != 0);\n"
        "    secp256k1::SchnorrSignature sig;",
        "    if (!scalar_parse_strict(pre_sig + 33, shat)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, \"invalid adaptor sig scalar\");\n"
        "    }\n"
        "    as.s_hat = shat;\n"
        "    as.needs_negation = (pre_sig[65] != 0);\n"
        "    secp256k1::SchnorrSignature sig;",
    ),
    # L2810+L2815+L2817: in ufsecp_silent_payment_address_create
    (
        "    if (!ctx || !scan_privkey || !spend_privkey || !scan_pubkey33_out ||\n"
        "        !spend_pubkey33_out || !addr_out || !addr_len)\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    Scalar scan_sk, spend_sk;\n"
        "    if (!scalar_parse_strict_nonzero(scan_privkey, scan_sk))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"scan privkey is zero or >= n\");\n"
        "    if (!scalar_parse_strict_nonzero(spend_privkey, spend_sk))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"spend privkey is zero or >= n\");",
        "    if (!ctx || !scan_privkey || !spend_privkey || !scan_pubkey33_out ||\n"
        "        !spend_pubkey33_out || !addr_out || !addr_len) {\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    }\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    Scalar scan_sk, spend_sk;\n"
        "    if (!scalar_parse_strict_nonzero(scan_privkey, scan_sk)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"scan privkey is zero or >= n\");\n"
        "    }\n"
        "    if (!scalar_parse_strict_nonzero(spend_privkey, spend_sk)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"spend privkey is zero or >= n\");\n"
        "    }",
    ),
    # L2827: if (addr_str.size() >= *addr_len)
    (
        "    if (addr_str.size() >= *addr_len)\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, \"address buffer too small\");",
        "    if (addr_str.size() >= *addr_len) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, \"address buffer too small\");\n"
        "    }",
    ),
    # L2846+L2855+L2864+L2868: in ufsecp_silent_payment_create_output
    (
        "    if (!ctx || !input_privkeys || n_inputs == 0 || !scan_pubkey33 ||\n"
        "        !spend_pubkey33 || !output_pubkey33_out)\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    // Parse input private keys\n"
        "    std::vector<Scalar> privkeys;\n"
        "    privkeys.reserve(n_inputs);\n"
        "    for (size_t i = 0; i < n_inputs; ++i) {\n"
        "        Scalar sk;\n"
        "        if (!scalar_parse_strict_nonzero(input_privkeys + i * 32, sk))\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"input privkey is zero or >= n\");\n"
        "        privkeys.push_back(sk);\n"
        "    }\n"
        "\n"
        "    // Parse recipient address\n"
        "    secp256k1::SilentPaymentAddress recipient;\n"
        "    recipient.scan_pubkey = point_from_compressed(scan_pubkey33);\n"
        "    recipient.spend_pubkey = point_from_compressed(spend_pubkey33);\n"
        "    if (recipient.scan_pubkey.is_infinity() || recipient.spend_pubkey.is_infinity())\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"invalid recipient pubkey\");\n"
        "\n"
        "    auto [output_point, tweak] = secp256k1::silent_payment_create_output(privkeys, recipient, k);\n"
        "    if (output_point.is_infinity())\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_ARITH, \"output point is infinity\");",
        "    if (!ctx || !input_privkeys || n_inputs == 0 || !scan_pubkey33 ||\n"
        "        !spend_pubkey33 || !output_pubkey33_out) {\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    }\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    // Parse input private keys\n"
        "    std::vector<Scalar> privkeys;\n"
        "    privkeys.reserve(n_inputs);\n"
        "    for (size_t i = 0; i < n_inputs; ++i) {\n"
        "        Scalar sk;\n"
        "        if (!scalar_parse_strict_nonzero(input_privkeys + i * 32, sk)) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"input privkey is zero or >= n\");\n"
        "        }\n"
        "        privkeys.push_back(sk);\n"
        "    }\n"
        "\n"
        "    // Parse recipient address\n"
        "    secp256k1::SilentPaymentAddress recipient;\n"
        "    recipient.scan_pubkey = point_from_compressed(scan_pubkey33);\n"
        "    recipient.spend_pubkey = point_from_compressed(spend_pubkey33);\n"
        "    if (recipient.scan_pubkey.is_infinity() || recipient.spend_pubkey.is_infinity()) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"invalid recipient pubkey\");\n"
        "    }\n"
        "\n"
        "    auto [output_point, tweak] = secp256k1::silent_payment_create_output(privkeys, recipient, k);\n"
        "    if (output_point.is_infinity()) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_ARITH, \"output point is infinity\");\n"
        "    }",
    ),
    # L2879: for (auto& sk : privkeys) — erase in ufsecp_silent_payment_create_output
    (
        "    for (auto& sk : privkeys)\n"
        "        secp256k1::detail::secure_erase(&sk, sizeof(sk));\n"
        "    return UFSECP_OK;\n"
        "}\n"
        "\n"
        "ufsecp_error_t ufsecp_silent_payment_scan(",
        "    for (auto& sk : privkeys) {\n"
        "        secp256k1::detail::secure_erase(&sk, sizeof(sk));\n"
        "    }\n"
        "    return UFSECP_OK;\n"
        "}\n"
        "\n"
        "ufsecp_error_t ufsecp_silent_payment_scan(",
    ),
    # L2894+L2896+L2901+L2903+L2911: in ufsecp_silent_payment_scan
    (
        "    if (!ctx || !scan_privkey || !spend_privkey || !input_pubkeys33 ||\n"
        "        !output_xonly32 || !n_found)\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    if (n_input_pubkeys == 0 || n_outputs == 0)\n"
        "        return UFSECP_ERR_BAD_INPUT;\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    Scalar scan_sk, spend_sk;\n"
        "    if (!scalar_parse_strict_nonzero(scan_privkey, scan_sk))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"scan privkey is zero or >= n\");\n"
        "    if (!scalar_parse_strict_nonzero(spend_privkey, spend_sk))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"spend privkey is zero or >= n\");\n"
        "\n"
        "    // Parse input pubkeys\n"
        "    std::vector<Point> input_pks;\n"
        "    input_pks.reserve(n_input_pubkeys);\n"
        "    for (size_t i = 0; i < n_input_pubkeys; ++i) {\n"
        "        auto pk = point_from_compressed(input_pubkeys33 + i * 33);\n"
        "        if (pk.is_infinity())\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"invalid input pubkey\");",
        "    if (!ctx || !scan_privkey || !spend_privkey || !input_pubkeys33 ||\n"
        "        !output_xonly32 || !n_found) {\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    }\n"
        "    if (n_input_pubkeys == 0 || n_outputs == 0) {\n"
        "        return UFSECP_ERR_BAD_INPUT;\n"
        "    }\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    Scalar scan_sk, spend_sk;\n"
        "    if (!scalar_parse_strict_nonzero(scan_privkey, scan_sk)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"scan privkey is zero or >= n\");\n"
        "    }\n"
        "    if (!scalar_parse_strict_nonzero(spend_privkey, spend_sk)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"spend privkey is zero or >= n\");\n"
        "    }\n"
        "\n"
        "    // Parse input pubkeys\n"
        "    std::vector<Point> input_pks;\n"
        "    input_pks.reserve(n_input_pubkeys);\n"
        "    for (size_t i = 0; i < n_input_pubkeys; ++i) {\n"
        "        auto pk = point_from_compressed(input_pubkeys33 + i * 33);\n"
        "        if (pk.is_infinity()) {\n"
        "            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"invalid input pubkey\");\n"
        "        }",
    ),
    # L2953+L2964+L2968+L2972: in ufsecp_ecies_encrypt
    (
        "    if (!ctx || !recipient_pubkey33 || !plaintext || !envelope_out || !envelope_len)\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    if (plaintext_len == 0) {\n"
        "        return UFSECP_ERR_BAD_INPUT;\n"
        "    }\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    if (plaintext_len > SIZE_MAX - UFSECP_ECIES_OVERHEAD) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"plaintext_len too large\");\n"
        "    }\n"
        "    size_t const needed = plaintext_len + UFSECP_ECIES_OVERHEAD;\n"
        "    if (*envelope_len < needed)\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, \"envelope buffer too small\");\n"
        "\n"
        "    auto pk = point_from_compressed(recipient_pubkey33);\n"
        "    if (pk.is_infinity())\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"invalid recipient pubkey\");\n"
        "\n"
        "    auto envelope = secp256k1::ecies_encrypt(pk, plaintext, plaintext_len);\n"
        "    if (envelope.empty())\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, \"ECIES encryption failed\");",
        "    if (!ctx || !recipient_pubkey33 || !plaintext || !envelope_out || !envelope_len) {\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    }\n"
        "    if (plaintext_len == 0) {\n"
        "        return UFSECP_ERR_BAD_INPUT;\n"
        "    }\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    if (plaintext_len > SIZE_MAX - UFSECP_ECIES_OVERHEAD) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, \"plaintext_len too large\");\n"
        "    }\n"
        "    size_t const needed = plaintext_len + UFSECP_ECIES_OVERHEAD;\n"
        "    if (*envelope_len < needed) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, \"envelope buffer too small\");\n"
        "    }\n"
        "\n"
        "    auto pk = point_from_compressed(recipient_pubkey33);\n"
        "    if (pk.is_infinity()) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, \"invalid recipient pubkey\");\n"
        "    }\n"
        "\n"
        "    auto envelope = secp256k1::ecies_encrypt(pk, plaintext, plaintext_len);\n"
        "    if (envelope.empty()) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, \"ECIES encryption failed\");\n"
        "    }",
    ),
    # L2985+L2987+L2992+L2996+L3002: in ufsecp_ecies_decrypt
    (
        "    if (!ctx || !privkey || !envelope || !plaintext_out || !plaintext_len)\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    if (envelope_len < 82) // min: 33 + 16 + 1 + 32\n"
        "        return UFSECP_ERR_BAD_INPUT;\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    size_t const expected_pt_len = envelope_len - UFSECP_ECIES_OVERHEAD;\n"
        "    if (*plaintext_len < expected_pt_len)\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, \"plaintext buffer too small\");\n"
        "\n"
        "    Scalar sk;\n"
        "    if (!scalar_parse_strict_nonzero(privkey, sk))\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"privkey is zero or >= n\");\n"
        "\n"
        "    auto pt = secp256k1::ecies_decrypt(sk, envelope, envelope_len);\n"
        "    secp256k1::detail::secure_erase(&sk, sizeof(sk));\n"
        "\n"
        "    if (pt.empty())\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, \"ECIES decryption failed (bad key or tampered)\");",
        "    if (!ctx || !privkey || !envelope || !plaintext_out || !plaintext_len) {\n"
        "        return UFSECP_ERR_NULL_ARG;\n"
        "    }\n"
        "    if (envelope_len < 82) { // min: 33 + 16 + 1 + 32\n"
        "        return UFSECP_ERR_BAD_INPUT;\n"
        "    }\n"
        "    ctx_clear_err(ctx);\n"
        "\n"
        "    size_t const expected_pt_len = envelope_len - UFSECP_ECIES_OVERHEAD;\n"
        "    if (*plaintext_len < expected_pt_len) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, \"plaintext buffer too small\");\n"
        "    }\n"
        "\n"
        "    Scalar sk;\n"
        "    if (!scalar_parse_strict_nonzero(privkey, sk)) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, \"privkey is zero or >= n\");\n"
        "    }\n"
        "\n"
        "    auto pt = secp256k1::ecies_decrypt(sk, envelope, envelope_len);\n"
        "    secp256k1::detail::secure_erase(&sk, sizeof(sk));\n"
        "\n"
        "    if (pt.empty()) {\n"
        "        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, \"ECIES decryption failed (bad key or tampered)\");\n"
        "    }",
    ),
]


def main():
    with open(PATH, "r") as f:
        content = f.read()

    for i, (old, new) in enumerate(REPLACEMENTS):
        count = content.count(old)
        if count == 0:
            print(f"[FAIL] Replacement {i+1}: NOT FOUND")
            print(f"  Looking for: {repr(old[:80])}")
            sys.exit(1)
        if count > 1:
            print(f"[WARN] Replacement {i+1}: found {count} occurrences, replacing first")
        content = content.replace(old, new, 1)
        print(f"[OK] Replacement {i+1} applied")

    with open(PATH, "w") as f:
        f.write(content)
    print(f"\nAll {len(REPLACEMENTS)} replacements applied to {PATH}")


if __name__ == "__main__":
    main()
