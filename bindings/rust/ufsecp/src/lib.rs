//! UltrafastSecp256k1 — safe Rust wrapper (ufsecp stable C ABI v1).
//!
//! ```no_run
//! use ufsecp::Context;
//! let ctx = Context::new().unwrap();
//! let pk = [0u8; 31].iter().copied().chain(std::iter::once(1u8)).collect::<Vec<u8>>();
//! let pubkey = ctx.pubkey_create(&pk).unwrap();
//! ```

use std::ffi::{CStr, CString};
use std::fmt;
use std::os::raw::{c_char, c_int};

pub use ufsecp_sys;

// ── Error ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    NullArg,
    BadKey,
    BadPubkey,
    BadSig,
    BadInput,
    VerifyFail,
    Arith,
    Selftest,
    Internal,
    BufSmall,
    Unknown(i32),
}

impl ErrorCode {
    fn from_raw(rc: c_int) -> Self {
        match rc {
            1 => Self::NullArg,
            2 => Self::BadKey,
            3 => Self::BadPubkey,
            4 => Self::BadSig,
            5 => Self::BadInput,
            6 => Self::VerifyFail,
            7 => Self::Arith,
            8 => Self::Selftest,
            9 => Self::Internal,
            10 => Self::BufSmall,
            x => Self::Unknown(x),
        }
    }
}

#[derive(Debug)]
pub struct Error {
    pub op: &'static str,
    pub code: ErrorCode,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ufsecp {} failed: {:?}", self.op, self.code)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

fn chk(rc: c_int, op: &'static str) -> Result<()> {
    if rc == 0 { Ok(()) } else { Err(Error { op, code: ErrorCode::from_raw(rc) }) }
}

// ── Network ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Network {
    Mainnet = 0,
    Testnet = 1,
}

// ── Recovery Signature ─────────────────────────────────────────────────

pub struct RecoverableSignature {
    pub signature: [u8; 64],
    pub recovery_id: i32,
}

pub struct TaprootOutputKey {
    pub output_key_x: [u8; 32],
    pub parity: i32,
}

pub struct WifDecoded {
    pub privkey: [u8; 32],
    pub compressed: bool,
    pub network: Network,
}

// ── Context ────────────────────────────────────────────────────────────

pub struct Context {
    ptr: *mut ufsecp_sys::ufsecp_ctx,
}

// SAFETY: The ufsecp context is internally thread-safe (dual-layer CT, no mutable shared state).
unsafe impl Send for Context {}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ufsecp_sys::ufsecp_ctx_destroy(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

impl Context {
    pub fn new() -> Result<Self> {
        let mut ptr: *mut ufsecp_sys::ufsecp_ctx = std::ptr::null_mut();
        let rc = unsafe { ufsecp_sys::ufsecp_ctx_create(&mut ptr) };
        chk(rc, "ctx_create")?;
        Ok(Context { ptr })
    }

    pub fn clone_ctx(&self) -> Result<Self> {
        let mut ptr: *mut ufsecp_sys::ufsecp_ctx = std::ptr::null_mut();
        let rc = unsafe { ufsecp_sys::ufsecp_ctx_clone(self.ptr, &mut ptr) };
        chk(rc, "ctx_clone")?;
        Ok(Context { ptr })
    }

    /// Return the last error code stored in this context.
    pub fn last_error(&self) -> i32 {
        unsafe { ufsecp_sys::ufsecp_last_error(self.ptr) }
    }

    /// Return the last error message stored in this context.
    pub fn last_error_msg(&self) -> &str {
        unsafe {
            let p = ufsecp_sys::ufsecp_last_error_msg(self.ptr);
            if p.is_null() { return ""; }
            CStr::from_ptr(p).to_str().unwrap_or("")
        }
    }

    // ── Version ────────────────────────────────────────────────────────

    pub fn version() -> u32 { unsafe { ufsecp_sys::ufsecp_version() } }
    pub fn abi_version() -> u32 { unsafe { ufsecp_sys::ufsecp_abi_version() } }
    pub fn version_string() -> &'static str {
        unsafe { CStr::from_ptr(ufsecp_sys::ufsecp_version_string()) }.to_str().unwrap_or("?")
    }

    // ── Key Operations ─────────────────────────────────────────────────

    pub fn pubkey_create(&self, privkey: &[u8]) -> Result<[u8; 33]> {
        assert_eq!(privkey.len(), 32);
        let mut out = [0u8; 33];
        let rc = unsafe { ufsecp_sys::ufsecp_pubkey_create(self.ptr, privkey.as_ptr(), out.as_mut_ptr()) };
        chk(rc, "pubkey_create")?;
        Ok(out)
    }

    pub fn pubkey_create_uncompressed(&self, privkey: &[u8]) -> Result<[u8; 65]> {
        assert_eq!(privkey.len(), 32);
        let mut out = [0u8; 65];
        let rc = unsafe { ufsecp_sys::ufsecp_pubkey_create_uncompressed(self.ptr, privkey.as_ptr(), out.as_mut_ptr()) };
        chk(rc, "pubkey_create_uncompressed")?;
        Ok(out)
    }

    pub fn pubkey_parse(&self, pubkey: &[u8]) -> Result<[u8; 33]> {
        let mut out = [0u8; 33];
        let rc = unsafe { ufsecp_sys::ufsecp_pubkey_parse(self.ptr, pubkey.as_ptr(), pubkey.len(), out.as_mut_ptr()) };
        chk(rc, "pubkey_parse")?;
        Ok(out)
    }

    pub fn pubkey_xonly(&self, privkey: &[u8]) -> Result<[u8; 32]> {
        assert_eq!(privkey.len(), 32);
        let mut out = [0u8; 32];
        let rc = unsafe { ufsecp_sys::ufsecp_pubkey_xonly(self.ptr, privkey.as_ptr(), out.as_mut_ptr()) };
        chk(rc, "pubkey_xonly")?;
        Ok(out)
    }

    pub fn seckey_verify(&self, privkey: &[u8]) -> bool {
        assert_eq!(privkey.len(), 32);
        unsafe { ufsecp_sys::ufsecp_seckey_verify(self.ptr, privkey.as_ptr()) == 0 }
    }

    pub fn seckey_negate(&self, privkey: &[u8]) -> Result<[u8; 32]> {
        assert_eq!(privkey.len(), 32);
        let mut buf = [0u8; 32];
        buf.copy_from_slice(privkey);
        let rc = unsafe { ufsecp_sys::ufsecp_seckey_negate(self.ptr, buf.as_mut_ptr()) };
        chk(rc, "seckey_negate")?;
        Ok(buf)
    }

    pub fn seckey_tweak_add(&self, privkey: &[u8], tweak: &[u8]) -> Result<[u8; 32]> {
        assert_eq!(privkey.len(), 32);
        assert_eq!(tweak.len(), 32);
        let mut buf = [0u8; 32];
        buf.copy_from_slice(privkey);
        let rc = unsafe { ufsecp_sys::ufsecp_seckey_tweak_add(self.ptr, buf.as_mut_ptr(), tweak.as_ptr()) };
        chk(rc, "seckey_tweak_add")?;
        Ok(buf)
    }

    pub fn seckey_tweak_mul(&self, privkey: &[u8], tweak: &[u8]) -> Result<[u8; 32]> {
        assert_eq!(privkey.len(), 32);
        assert_eq!(tweak.len(), 32);
        let mut buf = [0u8; 32];
        buf.copy_from_slice(privkey);
        let rc = unsafe { ufsecp_sys::ufsecp_seckey_tweak_mul(self.ptr, buf.as_mut_ptr(), tweak.as_ptr()) };
        chk(rc, "seckey_tweak_mul")?;
        Ok(buf)
    }

    // ── ECDSA ──────────────────────────────────────────────────────────

    pub fn ecdsa_sign(&self, msg_hash: &[u8], privkey: &[u8]) -> Result<[u8; 64]> {
        assert_eq!(msg_hash.len(), 32);
        assert_eq!(privkey.len(), 32);
        let mut sig = [0u8; 64];
        let rc = unsafe { ufsecp_sys::ufsecp_ecdsa_sign(self.ptr, msg_hash.as_ptr(), privkey.as_ptr(), sig.as_mut_ptr()) };
        chk(rc, "ecdsa_sign")?;
        Ok(sig)
    }

    pub fn ecdsa_verify(&self, msg_hash: &[u8], sig: &[u8], pubkey: &[u8]) -> bool {
        assert_eq!(msg_hash.len(), 32);
        assert_eq!(sig.len(), 64);
        assert_eq!(pubkey.len(), 33);
        unsafe { ufsecp_sys::ufsecp_ecdsa_verify(self.ptr, msg_hash.as_ptr(), sig.as_ptr(), pubkey.as_ptr()) == 0 }
    }

    pub fn ecdsa_sig_to_der(&self, sig: &[u8]) -> Result<Vec<u8>> {
        assert_eq!(sig.len(), 64);
        let mut der = [0u8; 72];
        let mut len: usize = 72;
        let rc = unsafe { ufsecp_sys::ufsecp_ecdsa_sig_to_der(self.ptr, sig.as_ptr(), der.as_mut_ptr(), &mut len) };
        chk(rc, "ecdsa_sig_to_der")?;
        Ok(der[..len].to_vec())
    }

    pub fn ecdsa_sig_from_der(&self, der: &[u8]) -> Result<[u8; 64]> {
        let mut sig = [0u8; 64];
        let rc = unsafe { ufsecp_sys::ufsecp_ecdsa_sig_from_der(self.ptr, der.as_ptr(), der.len(), sig.as_mut_ptr()) };
        chk(rc, "ecdsa_sig_from_der")?;
        Ok(sig)
    }

    // ── Recovery ───────────────────────────────────────────────────────

    pub fn ecdsa_sign_recoverable(&self, msg_hash: &[u8], privkey: &[u8]) -> Result<RecoverableSignature> {
        assert_eq!(msg_hash.len(), 32);
        assert_eq!(privkey.len(), 32);
        let mut sig = [0u8; 64];
        let mut recid: c_int = 0;
        let rc = unsafe {
            ufsecp_sys::ufsecp_ecdsa_sign_recoverable(self.ptr, msg_hash.as_ptr(), privkey.as_ptr(), sig.as_mut_ptr(), &mut recid)
        };
        chk(rc, "ecdsa_sign_recoverable")?;
        Ok(RecoverableSignature { signature: sig, recovery_id: recid })
    }

    pub fn ecdsa_recover(&self, msg_hash: &[u8], sig: &[u8], recid: i32) -> Result<[u8; 33]> {
        assert_eq!(msg_hash.len(), 32);
        assert_eq!(sig.len(), 64);
        let mut pubkey = [0u8; 33];
        let rc = unsafe {
            ufsecp_sys::ufsecp_ecdsa_recover(self.ptr, msg_hash.as_ptr(), sig.as_ptr(), recid, pubkey.as_mut_ptr())
        };
        chk(rc, "ecdsa_recover")?;
        Ok(pubkey)
    }

    // ── Schnorr ────────────────────────────────────────────────────────

    pub fn schnorr_sign(&self, msg: &[u8], privkey: &[u8], aux_rand: &[u8]) -> Result<[u8; 64]> {
        assert_eq!(msg.len(), 32);
        assert_eq!(privkey.len(), 32);
        assert_eq!(aux_rand.len(), 32);
        let mut sig = [0u8; 64];
        let rc = unsafe {
            ufsecp_sys::ufsecp_schnorr_sign(self.ptr, msg.as_ptr(), privkey.as_ptr(), aux_rand.as_ptr(), sig.as_mut_ptr())
        };
        chk(rc, "schnorr_sign")?;
        Ok(sig)
    }

    pub fn schnorr_verify(&self, msg: &[u8], sig: &[u8], pubkey_x: &[u8]) -> bool {
        assert_eq!(msg.len(), 32);
        assert_eq!(sig.len(), 64);
        assert_eq!(pubkey_x.len(), 32);
        unsafe { ufsecp_sys::ufsecp_schnorr_verify(self.ptr, msg.as_ptr(), sig.as_ptr(), pubkey_x.as_ptr()) == 0 }
    }

    // ── ECDH ───────────────────────────────────────────────────────────

    pub fn ecdh(&self, privkey: &[u8], pubkey: &[u8]) -> Result<[u8; 32]> {
        assert_eq!(privkey.len(), 32);
        assert_eq!(pubkey.len(), 33);
        let mut out = [0u8; 32];
        let rc = unsafe { ufsecp_sys::ufsecp_ecdh(self.ptr, privkey.as_ptr(), pubkey.as_ptr(), out.as_mut_ptr()) };
        chk(rc, "ecdh")?;
        Ok(out)
    }

    pub fn ecdh_xonly(&self, privkey: &[u8], pubkey: &[u8]) -> Result<[u8; 32]> {
        assert_eq!(privkey.len(), 32);
        assert_eq!(pubkey.len(), 33);
        let mut out = [0u8; 32];
        let rc = unsafe { ufsecp_sys::ufsecp_ecdh_xonly(self.ptr, privkey.as_ptr(), pubkey.as_ptr(), out.as_mut_ptr()) };
        chk(rc, "ecdh_xonly")?;
        Ok(out)
    }

    pub fn ecdh_raw(&self, privkey: &[u8], pubkey: &[u8]) -> Result<[u8; 32]> {
        assert_eq!(privkey.len(), 32);
        assert_eq!(pubkey.len(), 33);
        let mut out = [0u8; 32];
        let rc = unsafe { ufsecp_sys::ufsecp_ecdh_raw(self.ptr, privkey.as_ptr(), pubkey.as_ptr(), out.as_mut_ptr()) };
        chk(rc, "ecdh_raw")?;
        Ok(out)
    }

    // ── Hashing (context-free) ─────────────────────────────────────────

    pub fn sha256(data: &[u8]) -> Result<[u8; 32]> {
        let mut out = [0u8; 32];
        let rc = unsafe { ufsecp_sys::ufsecp_sha256(data.as_ptr(), data.len(), out.as_mut_ptr()) };
        chk(rc, "sha256")?;
        Ok(out)
    }

    pub fn hash160(data: &[u8]) -> Result<[u8; 20]> {
        let mut out = [0u8; 20];
        let rc = unsafe { ufsecp_sys::ufsecp_hash160(data.as_ptr(), data.len(), out.as_mut_ptr()) };
        chk(rc, "hash160")?;
        Ok(out)
    }

    pub fn tagged_hash(tag: &str, data: &[u8]) -> Result<[u8; 32]> {
        let mut out = [0u8; 32];
        let ctag = CString::new(tag).map_err(|_| Error { op: "tagged_hash", code: ErrorCode::BadInput })?;
        let rc = unsafe { ufsecp_sys::ufsecp_tagged_hash(ctag.as_ptr(), data.as_ptr(), data.len(), out.as_mut_ptr()) };
        chk(rc, "tagged_hash")?;
        Ok(out)
    }

    // ── Addresses ──────────────────────────────────────────────────────

    pub fn addr_p2pkh(&self, pubkey: &[u8], network: Network) -> Result<String> {
        assert_eq!(pubkey.len(), 33);
        self.get_addr(|ctx, pk, n, buf, len| unsafe {
            ufsecp_sys::ufsecp_addr_p2pkh(ctx, pk, n, buf, len)
        }, pubkey, network, "addr_p2pkh")
    }

    pub fn addr_p2wpkh(&self, pubkey: &[u8], network: Network) -> Result<String> {
        assert_eq!(pubkey.len(), 33);
        self.get_addr(|ctx, pk, n, buf, len| unsafe {
            ufsecp_sys::ufsecp_addr_p2wpkh(ctx, pk, n, buf, len)
        }, pubkey, network, "addr_p2wpkh")
    }

    pub fn addr_p2tr(&self, xonly: &[u8], network: Network) -> Result<String> {
        assert_eq!(xonly.len(), 32);
        self.get_addr(|ctx, pk, n, buf, len| unsafe {
            ufsecp_sys::ufsecp_addr_p2tr(ctx, pk, n, buf, len)
        }, xonly, network, "addr_p2tr")
    }

    fn get_addr(
        &self,
        f: impl Fn(*mut ufsecp_sys::ufsecp_ctx, *const u8, c_int, *mut c_char, *mut usize) -> c_int,
        key: &[u8],
        net: Network,
        op: &'static str,
    ) -> Result<String> {
        let mut buf = [0 as c_char; 128];
        let mut len: usize = 128;
        let rc = f(self.ptr, key.as_ptr(), net as c_int, buf.as_mut_ptr(), &mut len);
        chk(rc, op)?;
        let bytes: Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    // ── WIF ────────────────────────────────────────────────────────────

    pub fn wif_encode(&self, privkey: &[u8], compressed: bool, network: Network) -> Result<String> {
        assert_eq!(privkey.len(), 32);
        let mut buf = [0 as c_char; 128];
        let mut len: usize = 128;
        let rc = unsafe {
            ufsecp_sys::ufsecp_wif_encode(
                self.ptr, privkey.as_ptr(),
                if compressed { 1 } else { 0 },
                network as c_int,
                buf.as_mut_ptr(), &mut len,
            )
        };
        chk(rc, "wif_encode")?;
        let bytes: Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
        Ok(String::from_utf8_lossy(&bytes).to_string())
    }

    pub fn wif_decode(&self, wif: &str) -> Result<WifDecoded> {
        let cwif = CString::new(wif).map_err(|_| Error { op: "wif_decode", code: ErrorCode::BadInput })?;
        let mut privkey = [0u8; 32];
        let mut comp: c_int = 0;
        let mut net: c_int = 0;
        let rc = unsafe {
            ufsecp_sys::ufsecp_wif_decode(self.ptr, cwif.as_ptr(), privkey.as_mut_ptr(), &mut comp, &mut net)
        };
        chk(rc, "wif_decode")?;
        Ok(WifDecoded {
            privkey,
            compressed: comp == 1,
            network: if net == 0 { Network::Mainnet } else { Network::Testnet },
        })
    }

    // ── BIP-32 ─────────────────────────────────────────────────────────

    pub fn bip32_master(&self, seed: &[u8]) -> Result<[u8; 82]> {
        assert!((16..=64).contains(&seed.len()));
        let mut key = [0u8; 82];
        let rc = unsafe { ufsecp_sys::ufsecp_bip32_master(self.ptr, seed.as_ptr(), seed.len(), key.as_mut_ptr()) };
        chk(rc, "bip32_master")?;
        Ok(key)
    }

    pub fn bip32_derive(&self, parent: &[u8], index: u32) -> Result<[u8; 82]> {
        assert_eq!(parent.len(), 82);
        let mut child = [0u8; 82];
        let rc = unsafe { ufsecp_sys::ufsecp_bip32_derive(self.ptr, parent.as_ptr(), index, child.as_mut_ptr()) };
        chk(rc, "bip32_derive")?;
        Ok(child)
    }

    pub fn bip32_derive_path(&self, master: &[u8], path: &str) -> Result<[u8; 82]> {
        assert_eq!(master.len(), 82);
        let cpath = CString::new(path).map_err(|_| Error { op: "bip32_derive_path", code: ErrorCode::BadInput })?;
        let mut key = [0u8; 82];
        let rc = unsafe { ufsecp_sys::ufsecp_bip32_derive_path(self.ptr, master.as_ptr(), cpath.as_ptr(), key.as_mut_ptr()) };
        chk(rc, "bip32_derive_path")?;
        Ok(key)
    }

    pub fn bip32_privkey(&self, key: &[u8]) -> Result<[u8; 32]> {
        assert_eq!(key.len(), 82);
        let mut priv_key = [0u8; 32];
        let rc = unsafe { ufsecp_sys::ufsecp_bip32_privkey(self.ptr, key.as_ptr(), priv_key.as_mut_ptr()) };
        chk(rc, "bip32_privkey")?;
        Ok(priv_key)
    }

    pub fn bip32_pubkey(&self, key: &[u8]) -> Result<[u8; 33]> {
        assert_eq!(key.len(), 82);
        let mut pub_key = [0u8; 33];
        let rc = unsafe { ufsecp_sys::ufsecp_bip32_pubkey(self.ptr, key.as_ptr(), pub_key.as_mut_ptr()) };
        chk(rc, "bip32_pubkey")?;
        Ok(pub_key)
    }

    // ── Taproot ────────────────────────────────────────────────────────

    pub fn taproot_output_key(&self, internal_x: &[u8], merkle_root: Option<&[u8]>) -> Result<TaprootOutputKey> {
        assert_eq!(internal_x.len(), 32);
        let mut out_x = [0u8; 32];
        let mut parity: c_int = 0;
        let mr_ptr = merkle_root.map_or(std::ptr::null(), |m| m.as_ptr());
        let rc = unsafe {
            ufsecp_sys::ufsecp_taproot_output_key(self.ptr, internal_x.as_ptr(), mr_ptr, out_x.as_mut_ptr(), &mut parity)
        };
        chk(rc, "taproot_output_key")?;
        Ok(TaprootOutputKey { output_key_x: out_x, parity })
    }

    pub fn taproot_tweak_seckey(&self, privkey: &[u8], merkle_root: Option<&[u8]>) -> Result<[u8; 32]> {
        assert_eq!(privkey.len(), 32);
        let mut out = [0u8; 32];
        let mr_ptr = merkle_root.map_or(std::ptr::null(), |m| m.as_ptr());
        let rc = unsafe {
            ufsecp_sys::ufsecp_taproot_tweak_seckey(self.ptr, privkey.as_ptr(), mr_ptr, out.as_mut_ptr())
        };
        chk(rc, "taproot_tweak_seckey")?;
        Ok(out)
    }

    pub fn taproot_verify(&self, output_x: &[u8], parity: i32, internal_x: &[u8], merkle_root: Option<&[u8]>) -> bool {
        assert_eq!(output_x.len(), 32);
        assert_eq!(internal_x.len(), 32);
        let (mr_ptr, mr_len) = merkle_root.map_or((std::ptr::null(), 0), |m| (m.as_ptr(), m.len()));
        unsafe {
            ufsecp_sys::ufsecp_taproot_verify(self.ptr, output_x.as_ptr(), parity, internal_x.as_ptr(), mr_ptr, mr_len) == 0
        }
    }
}
