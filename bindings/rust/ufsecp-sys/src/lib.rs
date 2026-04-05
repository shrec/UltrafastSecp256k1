//! UltrafastSecp256k1 — Rust FFI binding (ufsecp stable C ABI v1).
//!
//! Raw `extern "C"` declarations for the ufsecp shared library.
//! This is the `-sys` crate; the safe wrapper is in `ufsecp` crate.

#![allow(non_camel_case_types)]

use std::os::raw::{c_char, c_int, c_void};

/// Opaque context type.
pub type ufsecp_ctx = c_void;

extern "C" {
    // ── Context ────────────────────────────────────────────────────────
    pub fn ufsecp_ctx_create(ctx_out: *mut *mut ufsecp_ctx) -> c_int;
    pub fn ufsecp_ctx_clone(src: *const ufsecp_ctx, ctx_out: *mut *mut ufsecp_ctx) -> c_int;
    pub fn ufsecp_ctx_destroy(ctx: *mut ufsecp_ctx);
    pub fn ufsecp_ctx_size() -> usize;

    // ── Version ────────────────────────────────────────────────────────
    pub fn ufsecp_version() -> u32;
    pub fn ufsecp_abi_version() -> u32;
    pub fn ufsecp_version_string() -> *const c_char;
    pub fn ufsecp_error_str(err: c_int) -> *const c_char;
    pub fn ufsecp_last_error(ctx: *const ufsecp_ctx) -> c_int;
    pub fn ufsecp_last_error_msg(ctx: *const ufsecp_ctx) -> *const c_char;

    // ── Key ops ────────────────────────────────────────────────────────
    pub fn ufsecp_seckey_verify(ctx: *const ufsecp_ctx, privkey: *const u8) -> c_int;
    pub fn ufsecp_seckey_negate(ctx: *mut ufsecp_ctx, privkey: *mut u8) -> c_int;
    pub fn ufsecp_seckey_tweak_add(ctx: *mut ufsecp_ctx, privkey: *mut u8, tweak: *const u8) -> c_int;
    pub fn ufsecp_seckey_tweak_mul(ctx: *mut ufsecp_ctx, privkey: *mut u8, tweak: *const u8) -> c_int;
    pub fn ufsecp_pubkey_create(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey33: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_create_uncompressed(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey65: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_parse(ctx: *mut ufsecp_ctx, input: *const u8, input_len: usize, pubkey33: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_xonly(ctx: *mut ufsecp_ctx, privkey: *const u8, xonly32: *mut u8) -> c_int;

    // ── Public key arithmetic ──────────────────────────────────────────
    pub fn ufsecp_pubkey_add(ctx: *mut ufsecp_ctx, a33: *const u8, b33: *const u8, out33: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_negate(ctx: *mut ufsecp_ctx, pubkey33: *const u8, out33: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_tweak_add(ctx: *mut ufsecp_ctx, pubkey33: *const u8, tweak: *const u8, out33: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_tweak_mul(ctx: *mut ufsecp_ctx, pubkey33: *const u8, tweak: *const u8, out33: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_combine(ctx: *mut ufsecp_ctx, pubkeys: *const u8, n: usize, out33: *mut u8) -> c_int;

    // ── ECDSA ──────────────────────────────────────────────────────────
    pub fn ufsecp_ecdsa_sign(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, sig64: *mut u8) -> c_int;
    pub fn ufsecp_ecdsa_sign_verified(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, sig64: *mut u8) -> c_int;
    pub fn ufsecp_ecdsa_verify(ctx: *mut ufsecp_ctx, msg32: *const u8, sig64: *const u8, pubkey33: *const u8) -> c_int;
    pub fn ufsecp_ecdsa_sig_to_der(ctx: *mut ufsecp_ctx, sig64: *const u8, der: *mut u8, der_len: *mut usize) -> c_int;
    pub fn ufsecp_ecdsa_sig_from_der(ctx: *mut ufsecp_ctx, der: *const u8, der_len: usize, sig64: *mut u8) -> c_int;
    pub fn ufsecp_ecdsa_sign_batch(ctx: *mut ufsecp_ctx, count: usize, msgs32: *const u8, privkeys32: *const u8, sigs64_out: *mut u8) -> c_int;
    pub fn ufsecp_ecdsa_batch_verify(ctx: *mut ufsecp_ctx, entries: *const u8, n: usize) -> c_int;
    pub fn ufsecp_ecdsa_batch_identify_invalid(ctx: *mut ufsecp_ctx, entries: *const u8, n: usize, invalid_out: *mut usize, invalid_count: *mut usize) -> c_int;

    // ── Recovery ───────────────────────────────────────────────────────
    pub fn ufsecp_ecdsa_sign_recoverable(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, sig64: *mut u8, recid: *mut c_int) -> c_int;
    pub fn ufsecp_ecdsa_recover(ctx: *mut ufsecp_ctx, msg32: *const u8, sig64: *const u8, recid: c_int, pubkey33: *mut u8) -> c_int;

    // ── Schnorr ────────────────────────────────────────────────────────
    pub fn ufsecp_schnorr_sign(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, aux_rand: *const u8, sig64: *mut u8) -> c_int;
    pub fn ufsecp_schnorr_sign_verified(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, aux_rand: *const u8, sig64: *mut u8) -> c_int;
    pub fn ufsecp_schnorr_verify(ctx: *mut ufsecp_ctx, msg32: *const u8, sig64: *const u8, pubkey_x: *const u8) -> c_int;
    pub fn ufsecp_schnorr_sign_batch(ctx: *mut ufsecp_ctx, count: usize, msgs32: *const u8, privkeys32: *const u8, aux_rands32: *const u8, sigs64_out: *mut u8) -> c_int;
    pub fn ufsecp_schnorr_batch_verify(ctx: *mut ufsecp_ctx, entries: *const u8, n: usize) -> c_int;
    pub fn ufsecp_schnorr_batch_identify_invalid(ctx: *mut ufsecp_ctx, entries: *const u8, n: usize, invalid_out: *mut usize, invalid_count: *mut usize) -> c_int;
    pub fn ufsecp_schnorr_sign_msg(ctx: *mut ufsecp_ctx, privkey: *const u8, msg: *const u8, msg_len: usize, aux_rand32: *const u8, sig64_out: *mut u8) -> c_int;
    pub fn ufsecp_schnorr_verify_msg(ctx: *mut ufsecp_ctx, pubkey_x: *const u8, msg: *const u8, msg_len: usize, sig64: *const u8) -> c_int;

    // ── ECDH ───────────────────────────────────────────────────────────
    pub fn ufsecp_ecdh(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey33: *const u8, secret32: *mut u8) -> c_int;
    pub fn ufsecp_ecdh_xonly(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey33: *const u8, secret32: *mut u8) -> c_int;
    pub fn ufsecp_ecdh_raw(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey33: *const u8, secret32: *mut u8) -> c_int;

    // ── Hashing ────────────────────────────────────────────────────────
    pub fn ufsecp_sha256(data: *const u8, len: usize, digest32: *mut u8) -> c_int;
    pub fn ufsecp_sha512(data: *const u8, len: usize, digest64: *mut u8) -> c_int;
    pub fn ufsecp_hash160(data: *const u8, len: usize, digest20: *mut u8) -> c_int;
    pub fn ufsecp_tagged_hash(tag: *const c_char, data: *const u8, len: usize, digest32: *mut u8) -> c_int;

    // ── Addresses ──────────────────────────────────────────────────────
    pub fn ufsecp_addr_p2pkh(ctx: *mut ufsecp_ctx, pubkey33: *const u8, network: c_int, addr: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_addr_p2wpkh(ctx: *mut ufsecp_ctx, pubkey33: *const u8, network: c_int, addr: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_addr_p2tr(ctx: *mut ufsecp_ctx, xonly32: *const u8, network: c_int, addr: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_addr_p2sh(redeem_script: *const u8, redeem_script_len: usize, network: c_int, addr_out: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_addr_p2sh_p2wpkh(ctx: *mut ufsecp_ctx, pubkey33: *const u8, network: c_int, addr_out: *mut c_char, addr_len: *mut usize) -> c_int;

    // ── WIF ────────────────────────────────────────────────────────────
    pub fn ufsecp_wif_encode(ctx: *mut ufsecp_ctx, privkey: *const u8, compressed: c_int, network: c_int, wif: *mut c_char, wif_len: *mut usize) -> c_int;
    pub fn ufsecp_wif_decode(ctx: *mut ufsecp_ctx, wif: *const c_char, privkey32: *mut u8, compressed: *mut c_int, network: *mut c_int) -> c_int;

    // ── BIP-32 ─────────────────────────────────────────────────────────
    pub fn ufsecp_bip32_master(ctx: *mut ufsecp_ctx, seed: *const u8, seed_len: usize, key82: *mut u8) -> c_int;
    pub fn ufsecp_bip32_derive(ctx: *mut ufsecp_ctx, parent82: *const u8, index: u32, child82: *mut u8) -> c_int;
    pub fn ufsecp_bip32_derive_path(ctx: *mut ufsecp_ctx, master82: *const u8, path: *const c_char, key82: *mut u8) -> c_int;
    pub fn ufsecp_bip32_privkey(ctx: *mut ufsecp_ctx, key82: *const u8, privkey32: *mut u8) -> c_int;
    pub fn ufsecp_bip32_pubkey(ctx: *mut ufsecp_ctx, key82: *const u8, pubkey33: *mut u8) -> c_int;

    // ── BIP-39 ─────────────────────────────────────────────────────────
    pub fn ufsecp_bip39_generate(ctx: *mut ufsecp_ctx, entropy_bytes: usize, entropy_in: *const u8, mnemonic_out: *mut c_char, mnemonic_len: *mut usize) -> c_int;
    pub fn ufsecp_bip39_validate(ctx: *const ufsecp_ctx, mnemonic: *const c_char) -> c_int;
    pub fn ufsecp_bip39_to_seed(ctx: *mut ufsecp_ctx, mnemonic: *const c_char, passphrase: *const c_char, seed64_out: *mut u8) -> c_int;
    pub fn ufsecp_bip39_to_entropy(ctx: *mut ufsecp_ctx, mnemonic: *const c_char, entropy_out: *mut u8, entropy_len: *mut usize) -> c_int;

    // ── BIP-85 ─────────────────────────────────────────────────────────
    pub fn ufsecp_bip85_entropy(ctx: *mut ufsecp_ctx, master_xprv: *const u8, path: *const c_char, entropy_out: *mut u8, entropy_len: usize) -> c_int;
    pub fn ufsecp_bip85_bip39(ctx: *mut ufsecp_ctx, master_xprv: *const u8, words: u32, language_index: u32, index: u32, mnemonic_out: *mut c_char, mnemonic_len: *mut usize) -> c_int;

    // ── Taproot ────────────────────────────────────────────────────────
    pub fn ufsecp_taproot_output_key(ctx: *mut ufsecp_ctx, internal_x: *const u8, merkle_root: *const u8, output_x: *mut u8, parity: *mut c_int) -> c_int;
    pub fn ufsecp_taproot_tweak_seckey(ctx: *mut ufsecp_ctx, privkey: *const u8, merkle_root: *const u8, tweaked32: *mut u8) -> c_int;
    pub fn ufsecp_taproot_verify(ctx: *mut ufsecp_ctx, output_x: *const u8, parity: c_int, internal_x: *const u8, merkle_root: *const u8, mr_len: usize) -> c_int;
    pub fn ufsecp_taproot_keypath_sighash(ctx: *mut ufsecp_ctx, version: u32, locktime: u32, input_count: usize, prevout_txids: *const u8, prevout_vouts: *const u32, input_amounts: *const u64, input_sequences: *const u32, input_spks: *const *const u8, input_spk_lens: *const usize, output_count: usize, output_values: *const u64, output_spks: *const *const u8, output_spk_lens: *const usize, input_index: usize, hash_type: u8, annex: *const u8, annex_len: usize, sighash_out: *mut u8) -> c_int;
    pub fn ufsecp_tapscript_sighash(ctx: *mut ufsecp_ctx, version: u32, locktime: u32, input_count: usize, prevout_txids: *const u8, prevout_vouts: *const u32, input_amounts: *const u64, input_sequences: *const u32, input_spks: *const *const u8, input_spk_lens: *const usize, output_count: usize, output_values: *const u64, output_spks: *const *const u8, output_spk_lens: *const usize, input_index: usize, hash_type: u8, tapleaf_hash: *const u8, key_version: u8, code_separator_pos: u32, annex: *const u8, annex_len: usize, sighash_out: *mut u8) -> c_int;

    // ── BIP-143 sighash ────────────────────────────────────────────────
    pub fn ufsecp_bip143_sighash(ctx: *mut ufsecp_ctx, version: u32, hash_prevouts: *const u8, hash_sequence: *const u8, outpoint_txid: *const u8, outpoint_vout: u32, script_code: *const u8, script_code_len: usize, value: u64, sequence: u32, hash_outputs: *const u8, locktime: u32, sighash_type: u32, sighash_out: *mut u8) -> c_int;
    pub fn ufsecp_bip143_p2wpkh_script_code(pubkey_hash: *const u8, script_code_out: *mut u8) -> c_int;

    // ── BIP-144 ─────────────────────────────────────────────────────────
    pub fn ufsecp_bip144_txid(ctx: *mut ufsecp_ctx, raw_tx: *const u8, raw_tx_len: usize, txid_out: *mut u8) -> c_int;
    pub fn ufsecp_bip144_wtxid(ctx: *mut ufsecp_ctx, raw_tx: *const u8, raw_tx_len: usize, wtxid_out: *mut u8) -> c_int;
    pub fn ufsecp_bip144_witness_commitment(witness_root: *const u8, witness_nonce: *const u8, commitment_out: *mut u8) -> c_int;

    // ── SegWit utils ────────────────────────────────────────────────────
    pub fn ufsecp_segwit_is_witness_program(script: *const u8, script_len: usize) -> c_int;
    pub fn ufsecp_segwit_parse_program(script: *const u8, script_len: usize, version_out: *mut c_int, program_out: *mut u8, program_len_out: *mut usize) -> c_int;
    pub fn ufsecp_segwit_p2wpkh_spk(pubkey_hash: *const u8, spk_out: *mut u8) -> c_int;
    pub fn ufsecp_segwit_p2wsh_spk(script_hash: *const u8, spk_out: *mut u8) -> c_int;
    pub fn ufsecp_segwit_p2tr_spk(output_key: *const u8, spk_out: *mut u8) -> c_int;
    pub fn ufsecp_segwit_witness_script_hash(script: *const u8, script_len: usize, hash_out: *mut u8) -> c_int;

    // ── MuSig2 ─────────────────────────────────────────────────────────
    pub fn ufsecp_musig2_key_agg(ctx: *mut ufsecp_ctx, pubkeys: *const u8, n: usize, keyagg_out: *mut u8, agg_pubkey32_out: *mut u8) -> c_int;
    pub fn ufsecp_musig2_nonce_gen(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey32: *const u8, agg_pubkey32: *const u8, msg32: *const u8, extra_in: *const u8, secnonce_out: *mut u8, pubnonce_out: *mut u8) -> c_int;
    pub fn ufsecp_musig2_nonce_agg(ctx: *mut ufsecp_ctx, pubnonces: *const u8, n: usize, aggnonce_out: *mut u8) -> c_int;
    pub fn ufsecp_musig2_start_sign_session(ctx: *mut ufsecp_ctx, aggnonce: *const u8, keyagg: *const u8, msg32: *const u8, session_out: *mut u8) -> c_int;
    pub fn ufsecp_musig2_partial_sign(ctx: *mut ufsecp_ctx, secnonce: *mut u8, privkey: *const u8, keyagg: *const u8, session: *const u8, signer_index: usize, partial_sig32_out: *mut u8) -> c_int;
    pub fn ufsecp_musig2_partial_verify(ctx: *mut ufsecp_ctx, partial_sig32: *const u8, pubnonce: *const u8, pubkey32: *const u8, keyagg: *const u8, session: *const u8, signer_index: usize) -> c_int;
    pub fn ufsecp_musig2_partial_sig_agg(ctx: *mut ufsecp_ctx, partial_sigs: *const u8, n: usize, session: *const u8, sig64_out: *mut u8) -> c_int;

    // ── FROST ──────────────────────────────────────────────────────────
    pub fn ufsecp_frost_keygen_begin(ctx: *mut ufsecp_ctx, participant_id: u32, threshold: u32, num_participants: u32, seed: *const u8, commits_out: *mut u8, commits_len: *mut usize, shares_out: *mut u8, shares_len: *mut usize) -> c_int;
    pub fn ufsecp_frost_keygen_finalize(ctx: *mut ufsecp_ctx, participant_id: u32, all_commits: *const u8, commits_len: usize, received_shares: *const u8, shares_len: usize, threshold: u32, num_participants: u32, keypkg_out: *mut u8) -> c_int;
    pub fn ufsecp_frost_sign_nonce_gen(ctx: *mut ufsecp_ctx, participant_id: u32, nonce_seed: *const u8, nonce_out: *mut u8, nonce_commit_out: *mut u8) -> c_int;
    pub fn ufsecp_frost_sign(ctx: *mut ufsecp_ctx, keypkg: *const u8, nonce: *const u8, msg32: *const u8, nonce_commits: *const u8, n_signers: usize, partial_sig_out: *mut u8) -> c_int;
    pub fn ufsecp_frost_verify_partial(ctx: *mut ufsecp_ctx, partial_sig: *const u8, verification_share33: *const u8, nonce_commits: *const u8, n_signers: usize, msg32: *const u8, group_pubkey33: *const u8) -> c_int;
    pub fn ufsecp_frost_aggregate(ctx: *mut ufsecp_ctx, partial_sigs: *const u8, n: usize, nonce_commits: *const u8, n_signers: usize, group_pubkey33: *const u8, msg32: *const u8, sig64_out: *mut u8) -> c_int;

    // ── Adaptor signatures ─────────────────────────────────────────────
    pub fn ufsecp_schnorr_adaptor_sign(ctx: *mut ufsecp_ctx, privkey: *const u8, msg32: *const u8, adaptor_point33: *const u8, aux_rand: *const u8, pre_sig_out: *mut u8) -> c_int;
    pub fn ufsecp_schnorr_adaptor_verify(ctx: *mut ufsecp_ctx, pre_sig: *const u8, pubkey_x: *const u8, msg32: *const u8, adaptor_point33: *const u8) -> c_int;
    pub fn ufsecp_schnorr_adaptor_adapt(ctx: *mut ufsecp_ctx, pre_sig: *const u8, adaptor_secret: *const u8, sig64_out: *mut u8) -> c_int;
    pub fn ufsecp_schnorr_adaptor_extract(ctx: *mut ufsecp_ctx, pre_sig: *const u8, sig64: *const u8, secret32_out: *mut u8) -> c_int;
    pub fn ufsecp_ecdsa_adaptor_sign(ctx: *mut ufsecp_ctx, privkey: *const u8, msg32: *const u8, adaptor_point33: *const u8, pre_sig_out: *mut u8) -> c_int;
    pub fn ufsecp_ecdsa_adaptor_verify(ctx: *mut ufsecp_ctx, pre_sig: *const u8, pubkey33: *const u8, msg32: *const u8, adaptor_point33: *const u8) -> c_int;
    pub fn ufsecp_ecdsa_adaptor_adapt(ctx: *mut ufsecp_ctx, pre_sig: *const u8, adaptor_secret: *const u8, sig64_out: *mut u8) -> c_int;
    pub fn ufsecp_ecdsa_adaptor_extract(ctx: *mut ufsecp_ctx, pre_sig: *const u8, sig64: *const u8, secret32_out: *mut u8) -> c_int;

    // ── Pedersen ───────────────────────────────────────────────────────
    pub fn ufsecp_pedersen_commit(ctx: *mut ufsecp_ctx, value: *const u8, blinding: *const u8, commitment33: *mut u8) -> c_int;
    pub fn ufsecp_pedersen_verify(ctx: *mut ufsecp_ctx, commitment33: *const u8, value: *const u8, blinding: *const u8) -> c_int;
    pub fn ufsecp_pedersen_verify_sum(ctx: *mut ufsecp_ctx, pos: *const u8, n_pos: usize, neg: *const u8, n_neg: usize) -> c_int;
    pub fn ufsecp_pedersen_blind_sum(ctx: *mut ufsecp_ctx, blinds_in: *const u8, n_in: usize, blinds_out: *const u8, n_out: usize, sum32_out: *mut u8) -> c_int;
    pub fn ufsecp_pedersen_switch_commit(ctx: *mut ufsecp_ctx, value: *const u8, blinding: *const u8, switch_blind: *const u8, commitment33_out: *mut u8) -> c_int;

    // ── ZK proofs ─────────────────────────────────────────────────────
    pub fn ufsecp_zk_knowledge_prove(ctx: *mut ufsecp_ctx, secret: *const u8, pubkey33: *const u8, msg32: *const u8, aux_rand: *const u8, proof_out: *mut u8) -> c_int;
    pub fn ufsecp_zk_knowledge_verify(ctx: *mut ufsecp_ctx, proof: *const u8, pubkey33: *const u8, msg32: *const u8) -> c_int;
    pub fn ufsecp_zk_dleq_prove(ctx: *mut ufsecp_ctx, secret: *const u8, g33: *const u8, h33: *const u8, p33: *const u8, q33: *const u8, aux_rand: *const u8, proof_out: *mut u8) -> c_int;
    pub fn ufsecp_zk_dleq_verify(ctx: *mut ufsecp_ctx, proof: *const u8, g33: *const u8, h33: *const u8, p33: *const u8, q33: *const u8) -> c_int;
    pub fn ufsecp_zk_range_prove(ctx: *mut ufsecp_ctx, value: u64, blinding: *const u8, commitment33: *const u8, aux_rand: *const u8, proof_out: *mut u8, proof_len: *mut usize) -> c_int;
    pub fn ufsecp_zk_range_verify(ctx: *mut ufsecp_ctx, commitment33: *const u8, proof: *const u8, proof_len: usize) -> c_int;
    pub fn ufsecp_zk_ecdsa_snark_witness(ctx: *mut ufsecp_ctx, msg_hash32: *const u8, pubkey33: *const u8, sig64: *const u8, out: *mut c_void) -> c_int;

    // ── Multi-scalar mul ───────────────────────────────────────────────
    pub fn ufsecp_shamir_trick(ctx: *mut ufsecp_ctx, a: *const u8, p33: *const u8, b: *const u8, q33: *const u8, out33: *mut u8) -> c_int;
    pub fn ufsecp_multi_scalar_mul(ctx: *mut ufsecp_ctx, scalars: *const u8, points: *const u8, n: usize, out33: *mut u8) -> c_int;

    // ── Multi-coin wallet ──────────────────────────────────────────────
    pub fn ufsecp_coin_address(ctx: *mut ufsecp_ctx, pubkey33: *const u8, coin_type: u32, testnet: c_int, addr_out: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_coin_derive_from_seed(ctx: *mut ufsecp_ctx, seed: *const u8, seed_len: usize, coin_type: u32, account: u32, change: c_int, index: u32, testnet: c_int, privkey32_out: *mut u8, pubkey33_out: *mut u8, addr_out: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_coin_wif_encode(ctx: *mut ufsecp_ctx, privkey: *const u8, coin_type: u32, testnet: c_int, wif_out: *mut c_char, wif_len: *mut usize) -> c_int;

    // ── Bitcoin message signing ────────────────────────────────────────
    pub fn ufsecp_btc_message_sign(ctx: *mut ufsecp_ctx, msg: *const u8, msg_len: usize, privkey: *const u8, base64_out: *mut c_char, base64_len: *mut usize) -> c_int;
    pub fn ufsecp_btc_message_verify(ctx: *mut ufsecp_ctx, msg: *const u8, msg_len: usize, pubkey33: *const u8, base64_sig: *const c_char) -> c_int;
    pub fn ufsecp_btc_message_hash(msg: *const u8, msg_len: usize, digest32_out: *mut u8) -> c_int;

    // ── BIP-322 ────────────────────────────────────────────────────────
    pub fn ufsecp_bip322_sign(ctx: *mut ufsecp_ctx, privkey: *const u8, addr_type: c_int, msg: *const u8, msg_len: usize, sig_out: *mut u8, sig_len: *mut usize) -> c_int;
    pub fn ufsecp_bip322_verify(ctx: *mut ufsecp_ctx, pubkey: *const u8, pubkey_len: usize, addr_type: c_int, msg: *const u8, msg_len: usize, sig: *const u8, sig_len: usize) -> c_int;

    // ── BIP-352 Silent Payments ────────────────────────────────────────
    pub fn ufsecp_silent_payment_address(ctx: *mut ufsecp_ctx, scan_privkey: *const u8, spend_privkey: *const u8, scan_pubkey33_out: *mut u8, spend_pubkey33_out: *mut u8, addr_out: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_silent_payment_create_output(ctx: *mut ufsecp_ctx, input_privkeys: *const u8, n_inputs: usize, scan_pubkey33: *const u8, spend_pubkey33: *const u8, k: u32, output_pubkey33_out: *mut u8, tweak32_out: *mut u8) -> c_int;
    pub fn ufsecp_silent_payment_scan(ctx: *mut ufsecp_ctx, scan_privkey: *const u8, spend_privkey: *const u8, input_pubkeys33: *const u8, n_input_pubkeys: usize, output_xonly32: *const u8, n_outputs: usize, found_indices_out: *mut u32, found_privkeys_out: *mut u8, n_found: *mut usize) -> c_int;

    // ── ECIES ──────────────────────────────────────────────────────────
    pub fn ufsecp_ecies_encrypt(ctx: *mut ufsecp_ctx, recipient_pubkey33: *const u8, plaintext: *const u8, plaintext_len: usize, envelope_out: *mut u8, envelope_len: *mut usize) -> c_int;
    pub fn ufsecp_ecies_decrypt(ctx: *mut ufsecp_ctx, privkey: *const u8, envelope: *const u8, envelope_len: usize, plaintext_out: *mut u8, plaintext_len: *mut usize) -> c_int;

    // ── BIP-324 (conditional: SECP256K1_BIP324) ───────────────────────
    pub fn ufsecp_bip324_create(ctx: *mut ufsecp_ctx, initiator: c_int, session_out: *mut *mut c_void, ellswift64_out: *mut u8) -> c_int;
    pub fn ufsecp_bip324_handshake(session: *mut c_void, peer_ellswift64: *const u8, session_id32_out: *mut u8) -> c_int;
    pub fn ufsecp_bip324_encrypt(session: *mut c_void, plaintext: *const u8, plaintext_len: usize, out: *mut u8, out_len: *mut usize) -> c_int;
    pub fn ufsecp_bip324_decrypt(session: *mut c_void, encrypted: *const u8, encrypted_len: usize, plaintext_out: *mut u8, plaintext_len: *mut usize) -> c_int;
    pub fn ufsecp_bip324_destroy(session: *mut c_void);
    pub fn ufsecp_aead_chacha20_poly1305_encrypt(key: *const u8, nonce: *const u8, aad: *const u8, aad_len: usize, plaintext: *const u8, plaintext_len: usize, out: *mut u8, tag: *mut u8) -> c_int;
    pub fn ufsecp_aead_chacha20_poly1305_decrypt(key: *const u8, nonce: *const u8, aad: *const u8, aad_len: usize, ciphertext: *const u8, ciphertext_len: usize, tag: *const u8, out: *mut u8) -> c_int;
    pub fn ufsecp_ellswift_create(ctx: *mut ufsecp_ctx, privkey: *const u8, encoding64_out: *mut u8) -> c_int;
    pub fn ufsecp_ellswift_xdh(ctx: *mut ufsecp_ctx, ell_a64: *const u8, ell_b64: *const u8, our_privkey: *const u8, initiating: c_int, secret32_out: *mut u8) -> c_int;

    // ── Ethereum (conditional: SECP256K1_BUILD_ETHEREUM) ──────────────
    pub fn ufsecp_keccak256(data: *const u8, len: usize, digest32_out: *mut u8) -> c_int;
    pub fn ufsecp_eth_address(ctx: *mut ufsecp_ctx, pubkey33: *const u8, addr20_out: *mut u8) -> c_int;
    pub fn ufsecp_eth_address_checksummed(ctx: *mut ufsecp_ctx, pubkey33: *const u8, addr_out: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_eth_personal_hash(msg: *const u8, msg_len: usize, digest32_out: *mut u8) -> c_int;
    pub fn ufsecp_eth_sign(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, r_out: *mut u8, s_out: *mut u8, v_out: *mut u64, chain_id: u64) -> c_int;
    pub fn ufsecp_eth_ecrecover(ctx: *mut ufsecp_ctx, msg32: *const u8, r: *const u8, s: *const u8, v: u64, addr20_out: *mut u8) -> c_int;

    // ── Descriptors ───────────────────────────────────────────────────
    pub fn ufsecp_descriptor_parse(ctx: *mut ufsecp_ctx, descriptor: *const c_char, index: u32, key_out: *mut u8, addr_out: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_descriptor_address(ctx: *mut ufsecp_ctx, descriptor: *const c_char, index: u32, addr_out: *mut c_char, addr_len: *mut usize) -> c_int;

    // ── PSBT helpers ──────────────────────────────────────────────────
    pub fn ufsecp_psbt_sign_legacy(ctx: *mut ufsecp_ctx, sighash32: *const u8, privkey: *const u8, sighash_type: u8, sig_out: *mut u8, sig_len: *mut usize) -> c_int;
    pub fn ufsecp_psbt_sign_segwit(ctx: *mut ufsecp_ctx, sighash32: *const u8, privkey: *const u8, sighash_type: u8, sig_out: *mut u8, sig_len: *mut usize) -> c_int;
    pub fn ufsecp_psbt_sign_taproot(ctx: *mut ufsecp_ctx, sighash32: *const u8, privkey: *const u8, sighash_type: u8, aux_rand32: *const u8, sig_out: *mut u8, sig_len: *mut usize) -> c_int;
    pub fn ufsecp_psbt_derive_key(ctx: *mut ufsecp_ctx, master_xprv: *const u8, key_path: *const c_char, privkey_out: *mut u8) -> c_int;

    // ── GCS / Compact Block Filters ───────────────────────────────────
    pub fn ufsecp_gcs_build(key: *const u8, data: *const *const u8, data_sizes: *const usize, count: usize, filter_out: *mut u8, filter_len: *mut usize) -> c_int;
    pub fn ufsecp_gcs_match(key: *const u8, filter: *const u8, filter_len: usize, n_items: usize, item: *const u8, item_len: usize) -> c_int;
    pub fn ufsecp_gcs_match_any(key: *const u8, filter: *const u8, filter_len: usize, n_items: usize, query: *const *const u8, query_sizes: *const usize, query_count: usize) -> c_int;
}

// ── GPU ────────────────────────────────────────────────────────────────

/// Opaque GPU context type.
pub type ufsecp_gpu_ctx = c_void;

/// Size in bytes of one flat ECDSA SNARK witness record (eprint 2025/695).
pub const UFSECP_ECDSA_SNARK_WITNESS_BYTES: usize = 760;

/// Size in bytes of the BIP-352 GPU scan key plan.
pub const UFSECP_BIP352_SCAN_PLAN_BYTES: usize = 264;

extern "C" {
    // Discovery
    pub fn ufsecp_gpu_backend_count(backend_ids: *mut u32, max_ids: u32) -> u32;
    pub fn ufsecp_gpu_backend_name(backend_id: u32) -> *const c_char;
    pub fn ufsecp_gpu_is_available(backend_id: u32) -> c_int;
    pub fn ufsecp_gpu_device_count(backend_id: u32) -> u32;
    pub fn ufsecp_gpu_device_info(backend_id: u32, device_index: u32, info_out: *mut u8) -> c_int;

    // Lifecycle
    pub fn ufsecp_gpu_ctx_create(ctx_out: *mut *mut ufsecp_gpu_ctx, backend_id: u32, device_index: u32) -> c_int;
    pub fn ufsecp_gpu_ctx_destroy(ctx: *mut ufsecp_gpu_ctx);
    pub fn ufsecp_gpu_last_error(ctx: *const ufsecp_gpu_ctx) -> c_int;
    pub fn ufsecp_gpu_last_error_msg(ctx: *const ufsecp_gpu_ctx) -> *const c_char;

    // Batch ops
    pub fn ufsecp_gpu_generator_mul_batch(ctx: *mut ufsecp_gpu_ctx, scalars32: *const u8, count: usize, out_pubkeys33: *mut u8) -> c_int;
    pub fn ufsecp_gpu_ecdsa_verify_batch(ctx: *mut ufsecp_gpu_ctx, msg32: *const u8, pk33: *const u8, sig64: *const u8, count: usize, results: *mut u8) -> c_int;
    pub fn ufsecp_gpu_schnorr_verify_batch(ctx: *mut ufsecp_gpu_ctx, msg32: *const u8, pkx32: *const u8, sig64: *const u8, count: usize, results: *mut u8) -> c_int;
    pub fn ufsecp_gpu_ecdh_batch(ctx: *mut ufsecp_gpu_ctx, sk32: *const u8, pk33: *const u8, count: usize, secrets32: *mut u8) -> c_int;
    pub fn ufsecp_gpu_hash160_pubkey_batch(ctx: *mut ufsecp_gpu_ctx, pk33: *const u8, count: usize, h20: *mut u8) -> c_int;
    pub fn ufsecp_gpu_msm(ctx: *mut ufsecp_gpu_ctx, s32: *const u8, p33: *const u8, count: usize, out33: *mut u8) -> c_int;
    pub fn ufsecp_gpu_frost_verify_partial_batch(ctx: *mut ufsecp_gpu_ctx, z_i32: *const u8, d_i33: *const u8, e_i33: *const u8, y_i33: *const u8, rho_i32: *const u8, lambda_ie32: *const u8, negate_r: *const u8, negate_key: *const u8, count: usize, out_results: *mut u8) -> c_int;
    pub fn ufsecp_gpu_ecrecover_batch(ctx: *mut ufsecp_gpu_ctx, msg_hashes32: *const u8, sigs64: *const u8, recids: *const c_int, count: usize, out_pubkeys33: *mut u8, out_valid: *mut u8) -> c_int;

    // ZK batch
    pub fn ufsecp_gpu_zk_knowledge_verify_batch(ctx: *mut ufsecp_gpu_ctx, proofs64: *const u8, pubkeys65: *const u8, messages32: *const u8, count: usize, out_results: *mut u8) -> c_int;
    pub fn ufsecp_gpu_zk_dleq_verify_batch(ctx: *mut ufsecp_gpu_ctx, proofs64: *const u8, g_pts65: *const u8, h_pts65: *const u8, p_pts65: *const u8, q_pts65: *const u8, count: usize, out_results: *mut u8) -> c_int;
    pub fn ufsecp_gpu_bulletproof_verify_batch(ctx: *mut ufsecp_gpu_ctx, proofs324: *const u8, commitments65: *const u8, h_generator65: *const u8, count: usize, out_results: *mut u8) -> c_int;

    // BIP-324 AEAD batch
    pub fn ufsecp_gpu_bip324_aead_encrypt_batch(ctx: *mut ufsecp_gpu_ctx, keys32: *const u8, nonces12: *const u8, plaintexts: *const u8, sizes: *const u32, max_payload: u32, count: usize, wire_out: *mut u8) -> c_int;
    pub fn ufsecp_gpu_bip324_aead_decrypt_batch(ctx: *mut ufsecp_gpu_ctx, keys32: *const u8, nonces12: *const u8, wire_in: *const u8, sizes: *const u32, max_payload: u32, count: usize, plaintext_out: *mut u8, out_valid: *mut u8) -> c_int;

    // ZK: ECDSA SNARK witness batch (eprint 2025/695)
    pub fn ufsecp_gpu_zk_ecdsa_snark_witness_batch(
        ctx:           *mut ufsecp_gpu_ctx,
        msg_hashes32:  *const u8,
        pubkeys33:     *const u8,
        sigs64:        *const u8,
        count:         usize,
        out_witnesses: *mut u8,
    ) -> c_int;

    // BIP-352: Silent Payment (CPU plan utility + GPU batch)
    pub fn ufsecp_bip352_prepare_scan_plan(
        scan_privkey32: *const u8,
        plan264_out:    *mut u8,
    ) -> c_int;
    pub fn ufsecp_gpu_bip352_scan_batch(
        ctx:             *mut ufsecp_gpu_ctx,
        scan_privkey32:  *const u8,
        spend_pubkey33:  *const u8,
        tweak_pubkeys33: *const u8,
        n_tweaks:        usize,
        prefix64_out:    *mut u64,
    ) -> c_int;

    // Error
    pub fn ufsecp_gpu_error_str(err: c_int) -> *const c_char;
}


    // ── Version ────────────────────────────────────────────────────────
    pub fn ufsecp_version() -> u32;
    pub fn ufsecp_abi_version() -> u32;
    pub fn ufsecp_version_string() -> *const c_char;
    pub fn ufsecp_error_str(err: c_int) -> *const c_char;
    pub fn ufsecp_last_error(ctx: *const ufsecp_ctx) -> c_int;
    pub fn ufsecp_last_error_msg(ctx: *const ufsecp_ctx) -> *const c_char;

    // ── Key ops ────────────────────────────────────────────────────────
    pub fn ufsecp_seckey_verify(ctx: *const ufsecp_ctx, privkey: *const u8) -> c_int;
    pub fn ufsecp_seckey_negate(ctx: *mut ufsecp_ctx, privkey: *mut u8) -> c_int;
    pub fn ufsecp_seckey_tweak_add(ctx: *mut ufsecp_ctx, privkey: *mut u8, tweak: *const u8) -> c_int;
    pub fn ufsecp_seckey_tweak_mul(ctx: *mut ufsecp_ctx, privkey: *mut u8, tweak: *const u8) -> c_int;
    pub fn ufsecp_pubkey_create(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey33: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_create_uncompressed(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey65: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_parse(ctx: *mut ufsecp_ctx, input: *const u8, input_len: usize, pubkey33: *mut u8) -> c_int;
    pub fn ufsecp_pubkey_xonly(ctx: *mut ufsecp_ctx, privkey: *const u8, xonly32: *mut u8) -> c_int;

    // ── ECDSA ──────────────────────────────────────────────────────────
    pub fn ufsecp_ecdsa_sign(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, sig64: *mut u8) -> c_int;
    pub fn ufsecp_ecdsa_verify(ctx: *mut ufsecp_ctx, msg32: *const u8, sig64: *const u8, pubkey33: *const u8) -> c_int;
    pub fn ufsecp_ecdsa_sig_to_der(ctx: *mut ufsecp_ctx, sig64: *const u8, der: *mut u8, der_len: *mut usize) -> c_int;
    pub fn ufsecp_ecdsa_sig_from_der(ctx: *mut ufsecp_ctx, der: *const u8, der_len: usize, sig64: *mut u8) -> c_int;

    // ── Recovery ───────────────────────────────────────────────────────
    pub fn ufsecp_ecdsa_sign_recoverable(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, sig64: *mut u8, recid: *mut c_int) -> c_int;
    pub fn ufsecp_ecdsa_recover(ctx: *mut ufsecp_ctx, msg32: *const u8, sig64: *const u8, recid: c_int, pubkey33: *mut u8) -> c_int;

    // ── Schnorr ────────────────────────────────────────────────────────
    pub fn ufsecp_schnorr_sign(ctx: *mut ufsecp_ctx, msg32: *const u8, privkey: *const u8, aux_rand: *const u8, sig64: *mut u8) -> c_int;
    pub fn ufsecp_schnorr_verify(ctx: *mut ufsecp_ctx, msg32: *const u8, sig64: *const u8, pubkey_x: *const u8) -> c_int;

    // ── ECDH ───────────────────────────────────────────────────────────
    pub fn ufsecp_ecdh(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey33: *const u8, secret32: *mut u8) -> c_int;
    pub fn ufsecp_ecdh_xonly(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey33: *const u8, secret32: *mut u8) -> c_int;
    pub fn ufsecp_ecdh_raw(ctx: *mut ufsecp_ctx, privkey: *const u8, pubkey33: *const u8, secret32: *mut u8) -> c_int;

    // ── Hashing ────────────────────────────────────────────────────────
    pub fn ufsecp_sha256(data: *const u8, len: usize, digest32: *mut u8) -> c_int;
    pub fn ufsecp_hash160(data: *const u8, len: usize, digest20: *mut u8) -> c_int;
    pub fn ufsecp_tagged_hash(tag: *const c_char, data: *const u8, len: usize, digest32: *mut u8) -> c_int;

    // ── Addresses ──────────────────────────────────────────────────────
    pub fn ufsecp_addr_p2pkh(ctx: *mut ufsecp_ctx, pubkey33: *const u8, network: c_int, addr: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_addr_p2wpkh(ctx: *mut ufsecp_ctx, pubkey33: *const u8, network: c_int, addr: *mut c_char, addr_len: *mut usize) -> c_int;
    pub fn ufsecp_addr_p2tr(ctx: *mut ufsecp_ctx, xonly32: *const u8, network: c_int, addr: *mut c_char, addr_len: *mut usize) -> c_int;

    // ── WIF ────────────────────────────────────────────────────────────
    pub fn ufsecp_wif_encode(ctx: *mut ufsecp_ctx, privkey: *const u8, compressed: c_int, network: c_int, wif: *mut c_char, wif_len: *mut usize) -> c_int;
    pub fn ufsecp_wif_decode(ctx: *mut ufsecp_ctx, wif: *const c_char, privkey32: *mut u8, compressed: *mut c_int, network: *mut c_int) -> c_int;

    // ── BIP-32 ─────────────────────────────────────────────────────────
    pub fn ufsecp_bip32_master(ctx: *mut ufsecp_ctx, seed: *const u8, seed_len: usize, key82: *mut u8) -> c_int;
    pub fn ufsecp_bip32_derive(ctx: *mut ufsecp_ctx, parent82: *const u8, index: u32, child82: *mut u8) -> c_int;
    pub fn ufsecp_bip32_derive_path(ctx: *mut ufsecp_ctx, master82: *const u8, path: *const c_char, key82: *mut u8) -> c_int;
    pub fn ufsecp_bip32_privkey(ctx: *mut ufsecp_ctx, key82: *const u8, privkey32: *mut u8) -> c_int;
    pub fn ufsecp_bip32_pubkey(ctx: *mut ufsecp_ctx, key82: *const u8, pubkey33: *mut u8) -> c_int;

    // ── Taproot ────────────────────────────────────────────────────────
    pub fn ufsecp_taproot_output_key(ctx: *mut ufsecp_ctx, internal_x: *const u8, merkle_root: *const u8, output_x: *mut u8, parity: *mut c_int) -> c_int;
    pub fn ufsecp_taproot_tweak_seckey(ctx: *mut ufsecp_ctx, privkey: *const u8, merkle_root: *const u8, tweaked32: *mut u8) -> c_int;
    pub fn ufsecp_taproot_verify(ctx: *mut ufsecp_ctx, output_x: *const u8, parity: c_int, internal_x: *const u8, merkle_root: *const u8, mr_len: usize) -> c_int;

    // ── Pedersen ───────────────────────────────────────────────────────
    pub fn ufsecp_pedersen_commit(ctx: *mut ufsecp_ctx, value: *const u8, blinding: *const u8, commitment33: *mut u8) -> c_int;
    pub fn ufsecp_pedersen_verify(ctx: *mut ufsecp_ctx, commitment33: *const u8, value: *const u8, blinding: *const u8) -> c_int;
}

// ── GPU ────────────────────────────────────────────────────────────────

/// Opaque GPU context type.
pub type ufsecp_gpu_ctx = c_void;

/// Size in bytes of one flat ECDSA SNARK witness record (eprint 2025/695).
pub const UFSECP_ECDSA_SNARK_WITNESS_BYTES: usize = 760;

extern "C" {
    // Discovery
    pub fn ufsecp_gpu_backend_count(backend_ids: *mut u32, max_ids: u32) -> u32;
    pub fn ufsecp_gpu_backend_name(backend_id: u32) -> *const c_char;
    pub fn ufsecp_gpu_is_available(backend_id: u32) -> c_int;
    pub fn ufsecp_gpu_device_count(backend_id: u32) -> u32;

    // Lifecycle
    pub fn ufsecp_gpu_ctx_create(ctx_out: *mut *mut ufsecp_gpu_ctx, backend_id: u32, device_index: u32) -> c_int;
    pub fn ufsecp_gpu_ctx_destroy(ctx: *mut ufsecp_gpu_ctx);

    // Batch ops
    pub fn ufsecp_gpu_generator_mul_batch(ctx: *mut ufsecp_gpu_ctx, scalars32: *const u8, count: usize, out_pubkeys33: *mut u8) -> c_int;
    pub fn ufsecp_gpu_ecdsa_verify_batch(ctx: *mut ufsecp_gpu_ctx, msg32: *const u8, pk33: *const u8, sig64: *const u8, count: usize, results: *mut u8) -> c_int;
    pub fn ufsecp_gpu_schnorr_verify_batch(ctx: *mut ufsecp_gpu_ctx, msg32: *const u8, pkx32: *const u8, sig64: *const u8, count: usize, results: *mut u8) -> c_int;
    pub fn ufsecp_gpu_ecdh_batch(ctx: *mut ufsecp_gpu_ctx, sk32: *const u8, pk33: *const u8, count: usize, secrets32: *mut u8) -> c_int;
    pub fn ufsecp_gpu_hash160_pubkey_batch(ctx: *mut ufsecp_gpu_ctx, pk33: *const u8, count: usize, h20: *mut u8) -> c_int;
    pub fn ufsecp_gpu_msm(ctx: *mut ufsecp_gpu_ctx, s32: *const u8, p33: *const u8, count: usize, out33: *mut u8) -> c_int;

    // ZK: ECDSA SNARK witness batch (eprint 2025/695)
    pub fn ufsecp_gpu_zk_ecdsa_snark_witness_batch(
        ctx:           *mut ufsecp_gpu_ctx,
        msg_hashes32:  *const u8,
        pubkeys33:     *const u8,
        sigs64:        *const u8,
        count:         usize,
        out_witnesses: *mut u8,
    ) -> c_int;

    // BIP-352: Silent Payment GPU batch
    pub fn ufsecp_gpu_bip352_scan_batch(
        ctx:             *mut ufsecp_gpu_ctx,
        scan_privkey32:  *const u8,
        spend_pubkey33:  *const u8,
        tweak_pubkeys33: *const u8,
        n_tweaks:        usize,
        prefix64_out:    *mut u64,
    ) -> c_int;

    // Error
    pub fn ufsecp_gpu_error_str(err: c_int) -> *const c_char;
}
