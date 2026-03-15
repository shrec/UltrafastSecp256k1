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

    // Error
    pub fn ufsecp_gpu_error_str(err: c_int) -> *const c_char;
}
