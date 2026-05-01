/* ============================================================================
 * Taproot, BIP-143/144, SegWit, pubkey arithmetic, BIP-39, MSM, batch identify
 * ============================================================================
 * Included by ufsecp_impl.cpp (unity build). Not a standalone compilation unit.
 * All includes, type aliases and helpers are provided by ufsecp_impl.cpp.
 * ============================================================================ */

ufsecp_error_t ufsecp_taproot_output_key(ufsecp_ctx* ctx,
                                         const uint8_t internal_x[32],
                                         const uint8_t* merkle_root,
                                         uint8_t output_x_out[32],
                                         int* parity_out) {
    if (SECP256K1_UNLIKELY(!ctx || !internal_x || !output_x_out || !parity_out)) {
        return UFSECP_ERR_NULL_ARG;
}
    ctx_clear_err(ctx);

    // Reject all-zero x-only key (not a valid curve point)
    {
        uint8_t acc = 0;
        for (int i = 0; i < 32; ++i) acc |= internal_x[i];
        if (acc == 0) return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "zero x-only key");
    }

    std::array<uint8_t, 32> ik;
    std::memcpy(ik.data(), internal_x, 32);
    size_t const mr_len = merkle_root ? 32 : 0;

    auto [ok_x, parity] = secp256k1::taproot_output_key(ik, merkle_root, mr_len);
    std::memcpy(output_x_out, ok_x.data(), 32);
    *parity_out = parity;
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_taproot_tweak_seckey(ufsecp_ctx* ctx,
                                           const uint8_t privkey[32],
                                           const uint8_t* merkle_root,
                                           uint8_t tweaked32_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !privkey || !tweaked32_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    Scalar sk;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(privkey, sk))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_KEY, "privkey is zero or >= n");
    }
    size_t const mr_len = merkle_root ? 32 : 0;

    auto tweaked = secp256k1::taproot_tweak_privkey(sk, merkle_root, mr_len);
    secp256k1::detail::secure_erase(&sk, sizeof(sk));
    if (tweaked.is_zero()) {
        secp256k1::detail::secure_erase(&tweaked, sizeof(tweaked));
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "taproot tweak resulted in zero");
}

    scalar_to_bytes(tweaked, tweaked32_out);
    secp256k1::detail::secure_erase(&tweaked, sizeof(tweaked));
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_taproot_verify(ufsecp_ctx* ctx,
                                     const uint8_t output_x[32], int output_parity,
                                     const uint8_t internal_x[32],
                                     const uint8_t* merkle_root, size_t merkle_root_len) {
    if (SECP256K1_UNLIKELY(!ctx || !output_x || !internal_x)) return UFSECP_ERR_NULL_ARG;
    if (SECP256K1_UNLIKELY(!merkle_root && merkle_root_len > 0)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    std::array<uint8_t, 32> ok_x, ik_x;
    std::memcpy(ok_x.data(), output_x, 32);
    std::memcpy(ik_x.data(), internal_x, 32);

    if (!secp256k1::taproot_verify_commitment(ok_x, output_parity, ik_x,
                                              merkle_root, merkle_root_len)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "taproot commitment invalid");
}

    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-143: SegWit v0 Sighash
 * =========================================================================== */

ufsecp_error_t ufsecp_bip143_sighash(
    ufsecp_ctx* ctx,
    uint32_t version,
    const uint8_t hash_prevouts[32],
    const uint8_t hash_sequence[32],
    const uint8_t outpoint_txid[32], uint32_t outpoint_vout,
    const uint8_t* script_code, size_t script_code_len,
    uint64_t value,
    uint32_t sequence,
    const uint8_t hash_outputs[32],
    uint32_t locktime,
    uint32_t sighash_type,
    uint8_t sighash_out[32]) {
    if (!ctx || !hash_prevouts || !hash_sequence || !outpoint_txid ||
        !script_code || !hash_outputs || !sighash_out)
        return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);

    secp256k1::Bip143Preimage pre{};
    pre.version = version;
    std::memcpy(pre.hash_prevouts.data(), hash_prevouts, 32);
    std::memcpy(pre.hash_sequence.data(), hash_sequence, 32);
    std::memcpy(pre.hash_outputs.data(),  hash_outputs,  32);
    pre.locktime = locktime;

    secp256k1::Outpoint op{};
    std::memcpy(op.txid.data(), outpoint_txid, 32);
    op.vout = outpoint_vout;

    auto h = secp256k1::bip143_sighash(pre, op, script_code, script_code_len,
                                        value, sequence, sighash_type);
    std::memcpy(sighash_out, h.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip143_p2wpkh_script_code(
    const uint8_t pubkey_hash[20],
    uint8_t script_code_out[25]) {
    if (SECP256K1_UNLIKELY(!pubkey_hash || !script_code_out)) return UFSECP_ERR_NULL_ARG;

    auto sc = secp256k1::bip143_p2wpkh_script_code(pubkey_hash);
    std::memcpy(script_code_out, sc.data(), 25);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-144: Witness Transaction Serialization
 * =========================================================================== */

// Helper: read Bitcoin CompactSize from buffer; returns 0 on overflow
static size_t read_compact_size(const uint8_t* buf, size_t len,
                                size_t& offset, uint64_t& val) {
    if (offset >= len) return 0;
    uint8_t first = buf[offset++];
    if (first < 0xFD) { val = first; return 1; }
    if (first == 0xFD) {
        if (offset + 2 > len) return 0;
        val = uint64_t(buf[offset]) | (uint64_t(buf[offset+1]) << 8);
        offset += 2; return 3;
    }
    if (first == 0xFE) {
        if (offset + 4 > len) return 0;
        val = uint64_t(buf[offset]) | (uint64_t(buf[offset+1]) << 8) |
              (uint64_t(buf[offset+2]) << 16) | (uint64_t(buf[offset+3]) << 24);
        offset += 4; return 5;
    }
    // 0xFF
    if (offset + 8 > len) return 0;
    val = 0;
    for (int i = 0; i < 8; ++i) val |= uint64_t(buf[offset+i]) << (8*i);
    offset += 8; return 9;
}

// Helper: skip CompactSize-prefixed blob (e.g. scriptSig or scriptPubKey)
static bool skip_compact_bytes(const uint8_t* buf, size_t len, size_t& offset) {
    uint64_t sz = 0;
    if (!read_compact_size(buf, len, offset, sz)) return false;
    if (offset + sz > len) return false;
    offset += static_cast<size_t>(sz);
    return true;
}

ufsecp_error_t ufsecp_bip144_txid(
    ufsecp_ctx* ctx,
    const uint8_t* raw_tx, size_t raw_tx_len,
    uint8_t txid_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !raw_tx || !txid_out)) return UFSECP_ERR_NULL_ARG;
    if (raw_tx_len < 10) return UFSECP_ERR_BAD_INPUT;

    // Detect witness flag: version(4) + marker(0x00) + flag(0x01)
    bool has_witness = (raw_tx_len > 6 && raw_tx[4] == 0x00 && raw_tx[5] == 0x01);

    if (SECP256K1_UNLIKELY(!has_witness)) {
        // Legacy tx: txid = double-SHA256 of the entire raw bytes
        auto h = secp256k1::SHA256::hash256(raw_tx, raw_tx_len);
        std::memcpy(txid_out, h.data(), 32);
        return UFSECP_OK;
    }

    // Witness tx: strip marker+flag and witness data
    // Legacy = version(4) | inputs | outputs | locktime(4)
    secp256k1::SHA256 h1;
    // version
    h1.update(raw_tx, 4);

    // Skip marker+flag, parse inputs+outputs from offset 6
    size_t off = 6;
    uint64_t n_in = 0;
    size_t cs_start = off;
    if (!read_compact_size(raw_tx, raw_tx_len, off, n_in)) return UFSECP_ERR_BAD_INPUT;

    // Record start of vin count for hashing
    size_t io_start = cs_start;

    // Skip all inputs (each: txid(32) + vout(4) + scriptSig + sequence(4))
    for (uint64_t i = 0; i < n_in; ++i) {
        if (off + 36 > raw_tx_len) return UFSECP_ERR_BAD_INPUT;
        off += 36; // txid + vout
        if (!skip_compact_bytes(raw_tx, raw_tx_len, off)) return UFSECP_ERR_BAD_INPUT;
        if (off + 4 > raw_tx_len) return UFSECP_ERR_BAD_INPUT;
        off += 4; // sequence
    }

    // Parse outputs count
    uint64_t n_out = 0;
    if (!read_compact_size(raw_tx, raw_tx_len, off, n_out)) return UFSECP_ERR_BAD_INPUT;

    // Skip all outputs (each: value(8) + scriptPubKey)
    for (uint64_t i = 0; i < n_out; ++i) {
        if (off + 8 > raw_tx_len) return UFSECP_ERR_BAD_INPUT;
        off += 8; // value
        if (!skip_compact_bytes(raw_tx, raw_tx_len, off)) return UFSECP_ERR_BAD_INPUT;
    }
    size_t io_end = off;

    // Hash inputs+outputs section (cs_start..io_end)
    h1.update(raw_tx + io_start, io_end - io_start);

    // locktime = last 4 bytes
    if (raw_tx_len < 4) return UFSECP_ERR_BAD_INPUT;
    h1.update(raw_tx + raw_tx_len - 4, 4);

    auto first = h1.finalize();
    secp256k1::SHA256 h2;
    h2.update(first.data(), 32);
    auto txid = h2.finalize();
    std::memcpy(txid_out, txid.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip144_wtxid(
    ufsecp_ctx* ctx,
    const uint8_t* raw_tx, size_t raw_tx_len,
    uint8_t wtxid_out[32]) {
    if (SECP256K1_UNLIKELY(!ctx || !raw_tx || !wtxid_out)) return UFSECP_ERR_NULL_ARG;
    if (raw_tx_len < 10) return UFSECP_ERR_BAD_INPUT;

    // wtxid = double-SHA256 of the full witness-serialized tx
    auto h = secp256k1::SHA256::hash256(raw_tx, raw_tx_len);
    std::memcpy(wtxid_out, h.data(), 32);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_bip144_witness_commitment(
    const uint8_t witness_root[32],
    const uint8_t witness_nonce[32],
    uint8_t commitment_out[32]) {
    if (!witness_root || !witness_nonce || !commitment_out)
        return UFSECP_ERR_NULL_ARG;

    std::array<uint8_t, 32> wr, wn;
    std::memcpy(wr.data(), witness_root, 32);
    std::memcpy(wn.data(), witness_nonce, 32);

    auto c = secp256k1::witness_commitment(wr, wn);
    std::memcpy(commitment_out, c.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-141: Segregated Witness — Witness Programs
 * =========================================================================== */

int ufsecp_segwit_is_witness_program(
    const uint8_t* script, size_t script_len) {
    if (!script) return 0;
    return secp256k1::is_witness_program(script, script_len) ? 1 : 0;
}

ufsecp_error_t ufsecp_segwit_parse_program(
    const uint8_t* script, size_t script_len,
    int* version_out,
    uint8_t* program_out, size_t* program_len_out) {
    if (!script || !version_out || !program_out || !program_len_out)
        return UFSECP_ERR_NULL_ARG;

    auto wp = secp256k1::parse_witness_program(script, script_len);
    if (wp.version < 0) {
        *version_out = -1;
        *program_len_out = 0;
        return UFSECP_ERR_BAD_INPUT;
    }
    if (wp.program.size() > 40) return UFSECP_ERR_INTERNAL;   /* BIP-141 cap */
    *version_out = wp.version;
    *program_len_out = wp.program.size();
    std::memcpy(program_out, wp.program.data(), wp.program.size());
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_segwit_p2wpkh_spk(
    const uint8_t pubkey_hash[20],
    uint8_t spk_out[22]) {
    if (SECP256K1_UNLIKELY(!pubkey_hash || !spk_out)) return UFSECP_ERR_NULL_ARG;

    auto spk = secp256k1::segwit_scriptpubkey_p2wpkh(pubkey_hash);
    std::memcpy(spk_out, spk.data(), 22);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_segwit_p2wsh_spk(
    const uint8_t script_hash[32],
    uint8_t spk_out[34]) {
    if (SECP256K1_UNLIKELY(!script_hash || !spk_out)) return UFSECP_ERR_NULL_ARG;

    auto spk = secp256k1::segwit_scriptpubkey_p2wsh(script_hash);
    std::memcpy(spk_out, spk.data(), 34);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_segwit_p2tr_spk(
    const uint8_t output_key[32],
    uint8_t spk_out[34]) {
    if (SECP256K1_UNLIKELY(!output_key || !spk_out)) return UFSECP_ERR_NULL_ARG;

    auto spk = secp256k1::segwit_scriptpubkey_p2tr(output_key);
    std::memcpy(spk_out, spk.data(), 34);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_segwit_witness_script_hash(
    const uint8_t* script, size_t script_len,
    uint8_t hash_out[32]) {
    // Allow (nullptr, 0) as a valid empty-script input; only reject null when len > 0
    if ((!script && script_len > 0) || !hash_out) return UFSECP_ERR_NULL_ARG;

    auto h = secp256k1::witness_script_hash(script, script_len);
    std::memcpy(hash_out, h.data(), 32);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-342: Tapscript Sighash
 * =========================================================================== */

// Helper: build TapSighashTxData from flat arrays,
// converting flattened prevout_txids to array-of-array.
static secp256k1::TapSighashTxData build_tap_tx_data(
    uint32_t version, uint32_t locktime,
    size_t input_count,
    const uint8_t* prevout_txids_flat,
    const uint32_t* prevout_vouts,
    const uint64_t* input_amounts,
    const uint32_t* input_sequences,
    const uint8_t* const* input_spks,
    const size_t* input_spk_lens,
    size_t output_count,
    const uint64_t* output_values,
    const uint8_t* const* output_spks,
    const size_t* output_spk_lens,
    std::vector<std::array<uint8_t, 32>>& txid_storage) {

    // Convert flat txid array to array-of-array
    txid_storage.resize(input_count);
    for (size_t i = 0; i < input_count; ++i) {
        std::memcpy(txid_storage[i].data(), prevout_txids_flat + i * 32, 32);
    }

    secp256k1::TapSighashTxData td{};
    td.version = version;
    td.locktime = locktime;
    td.input_count = input_count;
    td.prevout_txids = txid_storage.data();
    td.prevout_vouts = prevout_vouts;
    td.input_amounts = input_amounts;
    td.input_sequences = input_sequences;
    td.input_scriptpubkeys = input_spks;
    td.input_scriptpubkey_lens = input_spk_lens;
    td.output_count = output_count;
    td.output_values = output_values;
    td.output_scriptpubkeys = output_spks;
    td.output_scriptpubkey_lens = output_spk_lens;
    return td;
}

ufsecp_error_t ufsecp_taproot_keypath_sighash(
    ufsecp_ctx* ctx,
    uint32_t version, uint32_t locktime,
    size_t input_count,
    const uint8_t* prevout_txids,
    const uint32_t* prevout_vouts,
    const uint64_t* input_amounts,
    const uint32_t* input_sequences,
    const uint8_t* const* input_spks,
    const size_t* input_spk_lens,
    size_t output_count,
    const uint64_t* output_values,
    const uint8_t* const* output_spks,
    const size_t* output_spk_lens,
    size_t input_index,
    uint8_t hash_type,
    const uint8_t* annex, size_t annex_len,
    uint8_t sighash_out[32]) {
    if (!ctx || !prevout_txids || !prevout_vouts || !input_amounts ||
        !input_sequences || !input_spks || !input_spk_lens ||
        !output_values || !output_spks || !output_spk_lens || !sighash_out)
        return UFSECP_ERR_NULL_ARG;
    if (input_index >= input_count)
        return UFSECP_ERR_BAD_INPUT;
    ctx_clear_err(ctx);

    try {
    std::vector<std::array<uint8_t, 32>> txid_storage;
    auto td = build_tap_tx_data(version, locktime, input_count,
        prevout_txids, prevout_vouts, input_amounts, input_sequences,
        input_spks, input_spk_lens, output_count, output_values,
        output_spks, output_spk_lens, txid_storage);

    auto h = secp256k1::taproot_keypath_sighash(td, input_index, hash_type,
                                                 annex, annex_len);
    std::memcpy(sighash_out, h.data(), 32);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_tapscript_sighash(
    ufsecp_ctx* ctx,
    uint32_t version, uint32_t locktime,
    size_t input_count,
    const uint8_t* prevout_txids,
    const uint32_t* prevout_vouts,
    const uint64_t* input_amounts,
    const uint32_t* input_sequences,
    const uint8_t* const* input_spks,
    const size_t* input_spk_lens,
    size_t output_count,
    const uint64_t* output_values,
    const uint8_t* const* output_spks,
    const size_t* output_spk_lens,
    size_t input_index,
    uint8_t hash_type,
    const uint8_t tapleaf_hash[32],
    uint8_t key_version,
    uint32_t code_separator_pos,
    const uint8_t* annex, size_t annex_len,
    uint8_t sighash_out[32]) {
    if (!ctx || !prevout_txids || !prevout_vouts || !input_amounts ||
        !input_sequences || !input_spks || !input_spk_lens ||
        !output_values || !output_spks || !output_spk_lens ||
        !tapleaf_hash || !sighash_out)
        return UFSECP_ERR_NULL_ARG;
    if (input_index >= input_count)
        return UFSECP_ERR_BAD_INPUT;
    ctx_clear_err(ctx);

    try {
    std::vector<std::array<uint8_t, 32>> txid_storage;
    auto td = build_tap_tx_data(version, locktime, input_count,
        prevout_txids, prevout_vouts, input_amounts, input_sequences,
        input_spks, input_spk_lens, output_count, output_values,
        output_spks, output_spk_lens, txid_storage);

    std::array<uint8_t, 32> tlh;
    std::memcpy(tlh.data(), tapleaf_hash, 32);

    auto h = secp256k1::tapscript_sighash(td, input_index, hash_type,
                                           tlh, key_version, code_separator_pos,
                                           annex, annex_len);
    std::memcpy(sighash_out, h.data(), 32);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * Public key arithmetic
 * =========================================================================== */

ufsecp_error_t ufsecp_pubkey_add(ufsecp_ctx* ctx,
                                 const uint8_t a33[33],
                                 const uint8_t b33[33],
                                 uint8_t out33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !a33 || !b33 || !out33)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto pa = point_from_compressed(a33);
    if (pa.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey a");
    }
    auto pb = point_from_compressed(b33);
    if (pb.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey b");
    }
    auto sum = pa.add(pb);
    if (sum.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "sum is point at infinity");
    }
    point_to_compressed(sum, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_negate(ufsecp_ctx* ctx,
                                    const uint8_t pubkey33[33],
                                    uint8_t out33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !out33)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto p = point_from_compressed(pubkey33);
    if (p.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    auto neg = p.negate();
    point_to_compressed(neg, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_tweak_add(ufsecp_ctx* ctx,
                                       const uint8_t pubkey33[33],
                                       const uint8_t tweak[32],
                                       uint8_t out33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !tweak || !out33)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto p = point_from_compressed(pubkey33);
    if (p.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    Scalar tw;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict(tweak, tw))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "tweak >= n");
    }
    auto tG = Point::generator().scalar_mul(tw);
    auto result = p.add(tG);
    if (result.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_add resulted in infinity");
    }
    point_to_compressed(result, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_tweak_mul(ufsecp_ctx* ctx,
                                       const uint8_t pubkey33[33],
                                       const uint8_t tweak[32],
                                       uint8_t out33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkey33 || !tweak || !out33)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    auto p = point_from_compressed(pubkey33);
    if (p.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey");
    }
    Scalar tw;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict_nonzero(tweak, tw))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "tweak is zero or >= n");
    }
    auto result = p.scalar_mul(tw);
    if (result.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "tweak_mul resulted in infinity");
    }
    point_to_compressed(result, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_pubkey_combine(ufsecp_ctx* ctx,
                                     const uint8_t* pubkeys,
                                     size_t n,
                                     uint8_t out33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !pubkeys || !out33)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "need >= 1 pubkey");
    ctx_clear_err(ctx);
    std::size_t total_pubkey_bytes = 0;
    if (!checked_mul_size(n, static_cast<std::size_t>(33), total_pubkey_bytes)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "pubkey array length too large");
    }
    auto acc = point_from_compressed(pubkeys);
    if (acc.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey[0]");
    }
    for (size_t i = 1; i < n; ++i) {
        auto pi = point_from_compressed(pubkeys + i * 33);
        if (pi.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey in array");
        }
        acc = acc.add(pi);
    }
    if (acc.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "combined pubkey is infinity");
    }
    point_to_compressed(acc, out33);
    return UFSECP_OK;
}

/* ===========================================================================
 * BIP-39
 * =========================================================================== */

ufsecp_error_t ufsecp_bip39_generate(ufsecp_ctx* ctx,
                                     size_t entropy_bytes,
                                     const uint8_t* entropy_in,
                                     char* mnemonic_out,
                                     size_t* mnemonic_len) {
    if (SECP256K1_UNLIKELY(!ctx || !mnemonic_out || !mnemonic_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    if (entropy_bytes != 16 && entropy_bytes != 20 && entropy_bytes != 24 &&
        entropy_bytes != 28 && entropy_bytes != 32) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "entropy must be 16/20/24/28/32");
    }
    try {
    auto [mnemonic, ok] = secp256k1::bip39_generate(entropy_bytes, entropy_in);
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_INTERNAL, "BIP-39 generation failed");
    }
    if (*mnemonic_len <= mnemonic.size()) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "mnemonic buffer too small");
    }
    std::memcpy(mnemonic_out, mnemonic.c_str(), mnemonic.size() + 1);
    *mnemonic_len = mnemonic.size();
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_bip39_validate(const ufsecp_ctx* ctx,
                                     const char* mnemonic) {
    if (SECP256K1_UNLIKELY(!ctx || !mnemonic)) return UFSECP_ERR_NULL_ARG;
    try {
    if (!secp256k1::bip39_validate(std::string(mnemonic))) {
        return UFSECP_ERR_BAD_INPUT;
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(nullptr)
}

ufsecp_error_t ufsecp_bip39_to_seed(ufsecp_ctx* ctx,
                                    const char* mnemonic,
                                    const char* passphrase,
                                    uint8_t seed64_out[64]) {
    if (SECP256K1_UNLIKELY(!ctx || !mnemonic || !seed64_out)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    try {
    const std::string pass = passphrase ? passphrase : "";
    auto [seed, ok] = secp256k1::bip39_mnemonic_to_seed(std::string(mnemonic), pass);
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid mnemonic");
    }
    std::memcpy(seed64_out, seed.data(), 64);
    secp256k1::detail::secure_erase(seed.data(), seed.size());
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_bip39_to_entropy(ufsecp_ctx* ctx,
                                       const char* mnemonic,
                                       uint8_t* entropy_out,
                                       size_t* entropy_len) {
    if (SECP256K1_UNLIKELY(!ctx || !mnemonic || !entropy_out || !entropy_len)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    try {
    auto [ent, ok] = secp256k1::bip39_mnemonic_to_entropy(std::string(mnemonic));
    if (SECP256K1_UNLIKELY(!ok)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "invalid mnemonic");
    }
    if (*entropy_len < ent.length) {
        return ctx_set_err(ctx, UFSECP_ERR_BUF_TOO_SMALL, "entropy buffer too small");
    }
    std::memcpy(entropy_out, ent.data.data(), ent.length);
    *entropy_len = ent.length;
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * Batch verification
 * =========================================================================== */

ufsecp_error_t ufsecp_schnorr_batch_verify(ufsecp_ctx* ctx,
                                           const uint8_t* entries, size_t n) {
    if (SECP256K1_UNLIKELY(!ctx || !entries)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    std::size_t total_bytes = 0;
    if (!checked_mul_size(n, std::size_t{128}, total_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch size overflow");
    try {
    /* Each entry: 32-byte xonly pubkey | 32-byte msg | 64-byte sig = 128 bytes */
    std::vector<secp256k1::SchnorrBatchEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = entries + i * 128;
        // Strict: reject x-only pubkey >= p at ABI gate
        FE pk_fe;
        if (!FE::parse_bytes_strict(e, pk_fe)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "non-canonical pubkey (x>=p) in batch");
        }
        std::memcpy(batch[i].pubkey_x.data(), e, 32);
        std::memcpy(batch[i].message.data(), e + 32, 32);
        if (!secp256k1::SchnorrSignature::parse_strict(e + 64, batch[i].signature)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid Schnorr sig in batch");
        }
    }
    if (!secp256k1::schnorr_batch_verify(batch)) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "batch verify failed");
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_ecdsa_batch_verify(ufsecp_ctx* ctx,
                                         const uint8_t* entries, size_t n) {
    if (SECP256K1_UNLIKELY(!ctx || !entries)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return UFSECP_OK;
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    std::size_t total_bytes = 0;
    if (!checked_mul_size(n, std::size_t{129}, total_bytes))
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch size overflow");
    try {
    /* Each entry: 32-byte msg | 33-byte pubkey | 64-byte sig = 129 bytes */
    std::vector<secp256k1::ECDSABatchEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = entries + i * 129;
        std::memcpy(batch[i].msg_hash.data(), e, 32);
        batch[i].public_key = point_from_compressed(e + 32);
        if (batch[i].public_key.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey in batch");
        }
        std::array<uint8_t, 64> compact;
        std::memcpy(compact.data(), e + 65, 64);
        if (SECP256K1_UNLIKELY(!secp256k1::ECDSASignature::parse_compact_strict(compact, batch[i].signature))) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid ECDSA sig in batch");
        }
    }
    if (SECP256K1_UNLIKELY(!secp256k1::ecdsa_batch_verify(batch))) {
        return ctx_set_err(ctx, UFSECP_ERR_VERIFY_FAIL, "batch verify failed");
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_schnorr_batch_identify_invalid(
    ufsecp_ctx* ctx, const uint8_t* entries, size_t n,
    size_t* invalid_out, size_t* invalid_count) {
    if (SECP256K1_UNLIKELY(!ctx || !entries || !invalid_out || !invalid_count)) return UFSECP_ERR_NULL_ARG;
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    try {
    std::vector<secp256k1::SchnorrBatchEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = entries + i * 128;
        FE pk_fe;
        if (!FE::parse_bytes_strict(e, pk_fe)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "non-canonical pubkey (x>=p) in batch");
        }
        std::memcpy(batch[i].pubkey_x.data(), e, 32);
        std::memcpy(batch[i].message.data(), e + 32, 32);
        if (!secp256k1::SchnorrSignature::parse_strict(e + 64, batch[i].signature)) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid Schnorr sig in batch");
        }
    }
    auto invalids = secp256k1::schnorr_batch_identify_invalid(batch.data(), n);
    size_t const capacity = *invalid_count;
    size_t const count = invalids.size() < capacity ? invalids.size() : capacity;
    *invalid_count = invalids.size();
    for (size_t i = 0; i < count; ++i) {
        invalid_out[i] = invalids[i];
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

ufsecp_error_t ufsecp_ecdsa_batch_identify_invalid(
    ufsecp_ctx* ctx, const uint8_t* entries, size_t n,
    size_t* invalid_out, size_t* invalid_count) {
    if (SECP256K1_UNLIKELY(!ctx || !entries || !invalid_out || !invalid_count)) return UFSECP_ERR_NULL_ARG;
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "batch count too large");
    ctx_clear_err(ctx);
    try {
    std::vector<secp256k1::ECDSABatchEntry> batch(n);
    for (size_t i = 0; i < n; ++i) {
        const uint8_t* e = entries + i * 129;
        std::memcpy(batch[i].msg_hash.data(), e, 32);
        batch[i].public_key = point_from_compressed(e + 32);
        if (batch[i].public_key.is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid pubkey in batch");
        }
        std::array<uint8_t, 64> compact;
        std::memcpy(compact.data(), e + 65, 64);
        if (SECP256K1_UNLIKELY(!secp256k1::ECDSASignature::parse_compact_strict(compact, batch[i].signature))) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_SIG, "invalid ECDSA sig in batch");
        }
    }
    auto invalids = secp256k1::ecdsa_batch_identify_invalid(batch.data(), n);
    size_t const capacity = *invalid_count;
    size_t const count = invalids.size() < capacity ? invalids.size() : capacity;
    *invalid_count = invalids.size();
    for (size_t i = 0; i < count; ++i) {
        invalid_out[i] = invalids[i];
    }
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * SHA-512
 * =========================================================================== */

ufsecp_error_t ufsecp_sha512(const uint8_t* data, size_t len,
                             uint8_t digest64_out[64]) {
    if (SECP256K1_UNLIKELY(!data || !digest64_out)) return UFSECP_ERR_NULL_ARG;
    auto hash = secp256k1::SHA512::hash(data, len);
    std::memcpy(digest64_out, hash.data(), 64);
    return UFSECP_OK;
}

/* ===========================================================================
 * Multi-scalar multiplication
 * =========================================================================== */

ufsecp_error_t ufsecp_shamir_trick(ufsecp_ctx* ctx,
                                   const uint8_t a[32], const uint8_t P33[33],
                                   const uint8_t b[32], const uint8_t Q33[33],
                                   uint8_t out33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !a || !P33 || !b || !Q33 || !out33)) return UFSECP_ERR_NULL_ARG;
    ctx_clear_err(ctx);
    Scalar sa, sb;
    if (SECP256K1_UNLIKELY(!scalar_parse_strict(a, sa))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "scalar a >= n");
    }
    if (SECP256K1_UNLIKELY(!scalar_parse_strict(b, sb))) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "scalar b >= n");
    }
    auto P = point_from_compressed(P33);
    if (P.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid point P");
    }
    auto Q = point_from_compressed(Q33);
    if (Q.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid point Q");
    }
    auto result = secp256k1::shamir_trick(sa, P, sb, Q);
    if (result.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "result is infinity");
    }
    point_to_compressed(result, out33);
    return UFSECP_OK;
}

ufsecp_error_t ufsecp_multi_scalar_mul(ufsecp_ctx* ctx,
                                       const uint8_t* scalars,
                                       const uint8_t* points,
                                       size_t n,
                                       uint8_t out33[33]) {
    if (SECP256K1_UNLIKELY(!ctx || !scalars || !points || !out33)) return UFSECP_ERR_NULL_ARG;
    if (n == 0) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n must be >= 1");
    if (n > kMaxBatchN) return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "n too large");
    ctx_clear_err(ctx);
    std::size_t total_scalar_bytes = 0;
    std::size_t total_point_bytes = 0;
    if (!checked_mul_size(n, static_cast<std::size_t>(32), total_scalar_bytes)
        || !checked_mul_size(n, static_cast<std::size_t>(33), total_point_bytes)) {
        return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "scalar/point array length too large");
    }
    try {
    std::vector<Scalar> svec(n);
    std::vector<Point> pvec(n);
    for (size_t i = 0; i < n; ++i) {
        if (SECP256K1_UNLIKELY(!scalar_parse_strict(scalars + i * 32, svec[i]))) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_INPUT, "scalar >= n");
        }
        pvec[i] = point_from_compressed(points + i * 33);
        if (pvec[i].is_infinity()) {
            return ctx_set_err(ctx, UFSECP_ERR_BAD_PUBKEY, "invalid point in array");
        }
    }
    auto result = secp256k1::multi_scalar_mul(svec, pvec);
    if (result.is_infinity()) {
        return ctx_set_err(ctx, UFSECP_ERR_ARITH, "MSM result is infinity");
    }
    point_to_compressed(result, out33);
    return UFSECP_OK;
    } UFSECP_CATCH_RETURN(ctx)
}

/* ===========================================================================
 * MuSig2 (BIP-327)
 * =========================================================================== */

