"""
UltrafastSecp256k1 -- Python ctypes binding (ufsecp stable C ABI v1).

High-performance secp256k1 elliptic curve cryptography with dual-layer
constant-time architecture. Context-based API.

Usage::

    from ufsecp import Ufsecp

    ctx = Ufsecp()
    pubkey = ctx.pubkey_create(bytes(31) + b'\\x01')
    sig = ctx.ecdsa_sign(bytes(32), bytes(31) + b'\\x01')
    ok = ctx.ecdsa_verify(bytes(32), sig, pubkey)
    ctx.close()      # or use as context manager
"""

from __future__ import annotations

import ctypes
import os
import platform
import sys
from ctypes import (
    POINTER, Structure, byref, c_char, c_char_p, c_int, c_size_t,
    c_uint8, c_uint32, c_void_p, create_string_buffer,
)
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

__all__ = [
    "Ufsecp",
    "UfsecpError",
    "NET_MAINNET",
    "NET_TESTNET",
]

# -- Constants ------------------------------------------------------------

NET_MAINNET = 0
NET_TESTNET = 1

# Error codes
_OK              = 0
_ERR_NULL_ARG    = 1
_ERR_BAD_KEY     = 2
_ERR_BAD_PUBKEY  = 3
_ERR_BAD_SIG     = 4
_ERR_BAD_INPUT   = 5
_ERR_VERIFY_FAIL = 6
_ERR_ARITH       = 7
_ERR_SELFTEST    = 8
_ERR_INTERNAL    = 9
_ERR_BUF_SMALL   = 10

_ERROR_NAMES = {
    _ERR_NULL_ARG:    "null argument",
    _ERR_BAD_KEY:     "invalid private key",
    _ERR_BAD_PUBKEY:  "invalid public key",
    _ERR_BAD_SIG:     "invalid signature",
    _ERR_BAD_INPUT:   "bad input",
    _ERR_VERIFY_FAIL: "verification failed",
    _ERR_ARITH:       "arithmetic error",
    _ERR_SELFTEST:    "selftest failed",
    _ERR_INTERNAL:    "internal error",
    _ERR_BUF_SMALL:   "buffer too small",
}


class UfsecpError(Exception):
    """Raised when a ufsecp C function returns non-zero."""

    def __init__(self, operation: str, code: int):
        self.operation = operation
        self.code = code
        msg = _ERROR_NAMES.get(code, f"unknown ({code})")
        super().__init__(f"ufsecp {operation} failed: {msg}")


# -- Result types ---------------------------------------------------------

class RecoverableSignature(NamedTuple):
    signature: bytes  # 64 bytes
    recovery_id: int

class WifDecoded(NamedTuple):
    privkey: bytes    # 32 bytes
    compressed: bool
    network: int      # NET_MAINNET / NET_TESTNET

class TaprootOutputKey(NamedTuple):
    output_key_x: bytes  # 32 bytes
    parity: int


# -- Library loader -------------------------------------------------------

def _find_library() -> str:
    """Locate the ufsecp shared library."""
    system = platform.system()
    if system == "Windows":
        names = ["ufsecp.dll"]
    elif system == "Darwin":
        names = ["libufsecp.dylib"]
    else:
        names = ["libufsecp.so"]

    # 1. UFSECP_LIB env var
    env = os.environ.get("UFSECP_LIB")
    if env and os.path.isfile(env):
        return env

    # 2. Next to this file
    here = Path(__file__).resolve().parent
    for n in names:
        p = here / n
        if p.is_file():
            return str(p)

    # 3. Package native/ directory
    for n in names:
        p = here / "native" / n
        if p.is_file():
            return str(p)

    # 4. Common build dirs
    root = here.parent
    for d in ("build_rel", "build-linux", "build"):
        for n in names:
            p = root / d / n
            if p.is_file():
                return str(p)

    # 5. System default
    return names[0]


# -- Main class -----------------------------------------------------------

class Ufsecp:
    """Context-based wrapper around the ufsecp C ABI.

    Use as a context manager::

        with Ufsecp() as ctx:
            pub = ctx.pubkey_create(privkey)
    """

    def __init__(self, lib_path: Optional[str] = None):
        self._lib = ctypes.CDLL(lib_path or _find_library())
        self._bind()
        ctx = c_void_p()
        rc = self._lib.ufsecp_ctx_create(byref(ctx))
        if rc != _OK:
            raise UfsecpError("ctx_create", rc)
        self._ctx = ctx

    def close(self) -> None:
        """Destroy the context. Safe to call multiple times."""
        if hasattr(self, "_ctx") and self._ctx:
            self._lib.ufsecp_ctx_destroy(self._ctx)
            self._ctx = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -- Version ----------------------------------------------------------

    def version(self) -> int:
        return self._lib.ufsecp_version()

    def abi_version(self) -> int:
        return self._lib.ufsecp_abi_version()

    def version_string(self) -> str:
        self._lib.ufsecp_version_string.restype = c_char_p
        return self._lib.ufsecp_version_string().decode()

    # -- Context extras ---------------------------------------------------

    def clone(self) -> 'Ufsecp':
        """Clone this context (deep copy)."""
        clone_ptr = c_void_p()
        self._throw(self._lib.ufsecp_ctx_clone(self._ctx, byref(clone_ptr)), "ctx_clone")
        obj = object.__new__(Ufsecp)
        obj._lib = self._lib
        obj._ctx = clone_ptr
        return obj

    def last_error(self) -> int:
        """Return last error code."""
        return self._lib.ufsecp_last_error(self._ctx)

    def last_error_msg(self) -> str:
        """Return last error message."""
        return self._lib.ufsecp_last_error_msg(self._ctx).decode()

    # -- Key operations ---------------------------------------------------

    def pubkey_create(self, privkey: bytes) -> bytes:
        """Compressed public key (33 bytes) from 32-byte private key."""
        _chk(privkey, 32, "privkey")
        out = (c_uint8 * 33)()
        self._throw(self._lib.ufsecp_pubkey_create(self._ctx, privkey, out), "pubkey_create")
        return bytes(out)

    def pubkey_create_uncompressed(self, privkey: bytes) -> bytes:
        """Uncompressed public key (65 bytes)."""
        _chk(privkey, 32, "privkey")
        out = (c_uint8 * 65)()
        self._throw(self._lib.ufsecp_pubkey_create_uncompressed(self._ctx, privkey, out), "pubkey_create_uncompressed")
        return bytes(out)

    def pubkey_parse(self, pubkey: bytes) -> bytes:
        """Parse compressed (33) or uncompressed (65) -> compressed 33 bytes."""
        out = (c_uint8 * 33)()
        self._throw(self._lib.ufsecp_pubkey_parse(self._ctx, pubkey, len(pubkey), out), "pubkey_parse")
        return bytes(out)

    def pubkey_xonly(self, privkey: bytes) -> bytes:
        """X-only public key (32 bytes, BIP-340)."""
        _chk(privkey, 32, "privkey")
        out = (c_uint8 * 32)()
        self._throw(self._lib.ufsecp_pubkey_xonly(self._ctx, privkey, out), "pubkey_xonly")
        return bytes(out)

    def seckey_verify(self, privkey: bytes) -> bool:
        _chk(privkey, 32, "privkey")
        return self._lib.ufsecp_seckey_verify(self._ctx, privkey) == _OK

    def seckey_negate(self, privkey: bytes) -> bytes:
        _chk(privkey, 32, "privkey")
        buf = (c_uint8 * 32)(*privkey)
        self._throw(self._lib.ufsecp_seckey_negate(self._ctx, buf), "seckey_negate")
        return bytes(buf)

    def seckey_tweak_add(self, privkey: bytes, tweak: bytes) -> bytes:
        _chk(privkey, 32, "privkey"); _chk(tweak, 32, "tweak")
        buf = (c_uint8 * 32)(*privkey)
        self._throw(self._lib.ufsecp_seckey_tweak_add(self._ctx, buf, tweak), "seckey_tweak_add")
        return bytes(buf)

    def seckey_tweak_mul(self, privkey: bytes, tweak: bytes) -> bytes:
        _chk(privkey, 32, "privkey"); _chk(tweak, 32, "tweak")
        buf = (c_uint8 * 32)(*privkey)
        self._throw(self._lib.ufsecp_seckey_tweak_mul(self._ctx, buf, tweak), "seckey_tweak_mul")
        return bytes(buf)

    # -- ECDSA ------------------------------------------------------------

    def ecdsa_sign(self, msg_hash: bytes, privkey: bytes) -> bytes:
        """ECDSA sign (RFC 6979). Returns 64-byte compact signature."""
        _chk(msg_hash, 32, "msg_hash"); _chk(privkey, 32, "privkey")
        sig = (c_uint8 * 64)()
        self._throw(self._lib.ufsecp_ecdsa_sign(self._ctx, msg_hash, privkey, sig), "ecdsa_sign")
        return bytes(sig)

    def ecdsa_verify(self, msg_hash: bytes, sig: bytes, pubkey: bytes) -> bool:
        _chk(msg_hash, 32, "msg_hash"); _chk(sig, 64, "sig"); _chk(pubkey, 33, "pubkey")
        return self._lib.ufsecp_ecdsa_verify(self._ctx, msg_hash, sig, pubkey) == _OK

    def ecdsa_sig_to_der(self, sig: bytes) -> bytes:
        _chk(sig, 64, "sig")
        der = (c_uint8 * 72)()
        length = c_size_t(72)
        self._throw(self._lib.ufsecp_ecdsa_sig_to_der(self._ctx, sig, der, byref(length)), "ecdsa_sig_to_der")
        return bytes(der[:length.value])

    def ecdsa_sig_from_der(self, der: bytes) -> bytes:
        sig = (c_uint8 * 64)()
        self._throw(self._lib.ufsecp_ecdsa_sig_from_der(self._ctx, der, len(der), sig), "ecdsa_sig_from_der")
        return bytes(sig)

    # -- Recovery ---------------------------------------------------------

    def ecdsa_sign_recoverable(self, msg_hash: bytes, privkey: bytes) -> RecoverableSignature:
        _chk(msg_hash, 32, "msg_hash"); _chk(privkey, 32, "privkey")
        sig = (c_uint8 * 64)()
        recid = c_int()
        self._throw(self._lib.ufsecp_ecdsa_sign_recoverable(self._ctx, msg_hash, privkey, sig, byref(recid)), "ecdsa_sign_recoverable")
        return RecoverableSignature(bytes(sig), recid.value)

    def ecdsa_recover(self, msg_hash: bytes, sig: bytes, recid: int) -> bytes:
        _chk(msg_hash, 32, "msg_hash"); _chk(sig, 64, "sig")
        pub = (c_uint8 * 33)()
        self._throw(self._lib.ufsecp_ecdsa_recover(self._ctx, msg_hash, sig, recid, pub), "ecdsa_recover")
        return bytes(pub)

    # -- Schnorr ----------------------------------------------------------

    def schnorr_sign(self, msg: bytes, privkey: bytes, aux_rand: bytes) -> bytes:
        _chk(msg, 32, "msg"); _chk(privkey, 32, "privkey"); _chk(aux_rand, 32, "aux_rand")
        sig = (c_uint8 * 64)()
        self._throw(self._lib.ufsecp_schnorr_sign(self._ctx, msg, privkey, aux_rand, sig), "schnorr_sign")
        return bytes(sig)

    def schnorr_verify(self, msg: bytes, sig: bytes, pubkey_x: bytes) -> bool:
        _chk(msg, 32, "msg"); _chk(sig, 64, "sig"); _chk(pubkey_x, 32, "pubkey_x")
        return self._lib.ufsecp_schnorr_verify(self._ctx, msg, sig, pubkey_x) == _OK

    # -- ECDH -------------------------------------------------------------

    def ecdh(self, privkey: bytes, pubkey: bytes) -> bytes:
        _chk(privkey, 32, "privkey"); _chk(pubkey, 33, "pubkey")
        out = (c_uint8 * 32)()
        self._throw(self._lib.ufsecp_ecdh(self._ctx, privkey, pubkey, out), "ecdh")
        return bytes(out)

    def ecdh_xonly(self, privkey: bytes, pubkey: bytes) -> bytes:
        _chk(privkey, 32, "privkey"); _chk(pubkey, 33, "pubkey")
        out = (c_uint8 * 32)()
        self._throw(self._lib.ufsecp_ecdh_xonly(self._ctx, privkey, pubkey, out), "ecdh_xonly")
        return bytes(out)

    def ecdh_raw(self, privkey: bytes, pubkey: bytes) -> bytes:
        _chk(privkey, 32, "privkey"); _chk(pubkey, 33, "pubkey")
        out = (c_uint8 * 32)()
        self._throw(self._lib.ufsecp_ecdh_raw(self._ctx, privkey, pubkey, out), "ecdh_raw")
        return bytes(out)

    # -- Hashing (context-free) -------------------------------------------

    def sha256(self, data: bytes) -> bytes:
        out = (c_uint8 * 32)()
        self._throw(self._lib.ufsecp_sha256(data, len(data), out), "sha256")
        return bytes(out)

    def hash160(self, data: bytes) -> bytes:
        out = (c_uint8 * 20)()
        self._throw(self._lib.ufsecp_hash160(data, len(data), out), "hash160")
        return bytes(out)

    def tagged_hash(self, tag: str, data: bytes) -> bytes:
        out = (c_uint8 * 32)()
        self._throw(self._lib.ufsecp_tagged_hash(tag.encode(), data, len(data), out), "tagged_hash")
        return bytes(out)

    # -- Addresses --------------------------------------------------------

    def addr_p2pkh(self, pubkey: bytes, network: int = NET_MAINNET) -> str:
        _chk(pubkey, 33, "pubkey")
        return self._get_addr(self._lib.ufsecp_addr_p2pkh, pubkey, network)

    def addr_p2wpkh(self, pubkey: bytes, network: int = NET_MAINNET) -> str:
        _chk(pubkey, 33, "pubkey")
        return self._get_addr(self._lib.ufsecp_addr_p2wpkh, pubkey, network)

    def addr_p2tr(self, xonly_key: bytes, network: int = NET_MAINNET) -> str:
        _chk(xonly_key, 32, "xonly_key")
        return self._get_addr(self._lib.ufsecp_addr_p2tr, xonly_key, network)

    # -- WIF --------------------------------------------------------------

    def wif_encode(self, privkey: bytes, compressed: bool = True, network: int = NET_MAINNET) -> str:
        _chk(privkey, 32, "privkey")
        buf = create_string_buffer(128)
        length = c_size_t(128)
        self._throw(self._lib.ufsecp_wif_encode(
            self._ctx, privkey, 1 if compressed else 0, network,
            ctypes.cast(buf, POINTER(c_uint8)), byref(length)
        ), "wif_encode")
        return buf.raw[:length.value].decode()

    def wif_decode(self, wif: str) -> WifDecoded:
        key = (c_uint8 * 32)()
        comp = c_int()
        net = c_int()
        self._throw(self._lib.ufsecp_wif_decode(
            self._ctx, wif.encode(), key, byref(comp), byref(net)
        ), "wif_decode")
        return WifDecoded(bytes(key), comp.value == 1, net.value)

    # -- BIP-32 -----------------------------------------------------------

    def bip32_master(self, seed: bytes) -> bytes:
        if not (16 <= len(seed) <= 64):
            raise ValueError(f"Seed must be 16-64 bytes, got {len(seed)}")
        key = (c_uint8 * 82)()
        self._throw(self._lib.ufsecp_bip32_master(self._ctx, seed, len(seed), key), "bip32_master")
        return bytes(key)

    def bip32_derive(self, parent: bytes, index: int) -> bytes:
        _chk(parent, 82, "parent")
        child = (c_uint8 * 82)()
        self._throw(self._lib.ufsecp_bip32_derive(self._ctx, parent, c_uint32(index), child), "bip32_derive")
        return bytes(child)

    def bip32_derive_path(self, master: bytes, path: str) -> bytes:
        _chk(master, 82, "master")
        key = (c_uint8 * 82)()
        self._throw(self._lib.ufsecp_bip32_derive_path(self._ctx, master, path.encode(), key), "bip32_derive_path")
        return bytes(key)

    def bip32_privkey(self, key: bytes) -> bytes:
        _chk(key, 82, "key")
        priv = (c_uint8 * 32)()
        self._throw(self._lib.ufsecp_bip32_privkey(self._ctx, key, priv), "bip32_privkey")
        return bytes(priv)

    def bip32_pubkey(self, key: bytes) -> bytes:
        _chk(key, 82, "key")
        pub = (c_uint8 * 33)()
        self._throw(self._lib.ufsecp_bip32_pubkey(self._ctx, key, pub), "bip32_pubkey")
        return bytes(pub)

    # -- Taproot ----------------------------------------------------------

    def taproot_output_key(self, internal_x: bytes, merkle_root: Optional[bytes] = None) -> TaprootOutputKey:
        _chk(internal_x, 32, "internal_x")
        out = (c_uint8 * 32)()
        parity = c_int()
        mr = merkle_root if merkle_root else None
        self._throw(self._lib.ufsecp_taproot_output_key(self._ctx, internal_x, mr, out, byref(parity)), "taproot_output_key")
        return TaprootOutputKey(bytes(out), parity.value)

    def taproot_tweak_seckey(self, privkey: bytes, merkle_root: Optional[bytes] = None) -> bytes:
        _chk(privkey, 32, "privkey")
        out = (c_uint8 * 32)()
        mr = merkle_root if merkle_root else None
        self._throw(self._lib.ufsecp_taproot_tweak_seckey(self._ctx, privkey, mr, out), "taproot_tweak_seckey")
        return bytes(out)

    def taproot_verify(self, output_x: bytes, parity: int, internal_x: bytes, merkle_root: Optional[bytes] = None) -> bool:
        _chk(output_x, 32, "output_x"); _chk(internal_x, 32, "internal_x")
        mr = merkle_root if merkle_root else None
        mr_len = len(merkle_root) if merkle_root else 0
        return self._lib.ufsecp_taproot_verify(self._ctx, output_x, parity, internal_x, mr, mr_len) == _OK

    # -- Internals --------------------------------------------------------

    def _throw(self, rc: int, op: str) -> None:
        if rc != _OK:
            raise UfsecpError(op, rc)

    def _get_addr(self, fn, key: bytes, network: int) -> str:
        buf = create_string_buffer(128)
        length = c_size_t(128)
        self._throw(fn(self._ctx, key, network, ctypes.cast(buf, POINTER(c_uint8)), byref(length)), "address")
        return buf.raw[:length.value].decode()

    def _bind(self) -> None:
        """Set up argtypes / restypes for type safety."""
        L = self._lib
        vp = c_void_p
        p8 = POINTER(c_uint8)
        pvp = POINTER(c_void_p)

        L.ufsecp_ctx_create.argtypes = [pvp]
        L.ufsecp_ctx_create.restype = c_int
        L.ufsecp_ctx_destroy.argtypes = [vp]
        L.ufsecp_ctx_destroy.restype = None

        L.ufsecp_ctx_clone.argtypes = [vp, pvp]
        L.ufsecp_ctx_clone.restype = c_int

        L.ufsecp_last_error.argtypes = [vp]
        L.ufsecp_last_error.restype = c_int
        L.ufsecp_last_error_msg.argtypes = [vp]
        L.ufsecp_last_error_msg.restype = c_char_p

        L.ufsecp_version.restype = c_uint32
        L.ufsecp_abi_version.restype = c_uint32

        for name in (
            "ufsecp_pubkey_create", "ufsecp_pubkey_create_uncompressed",
            "ufsecp_pubkey_xonly",
        ):
            getattr(L, name).argtypes = [vp, p8, p8]
            getattr(L, name).restype = c_int

        L.ufsecp_pubkey_parse.argtypes = [vp, p8, c_size_t, p8]
        L.ufsecp_pubkey_parse.restype = c_int

        L.ufsecp_seckey_verify.argtypes = [vp, p8]
        L.ufsecp_seckey_verify.restype = c_int
        L.ufsecp_seckey_negate.argtypes = [vp, p8]
        L.ufsecp_seckey_negate.restype = c_int

        for name in ("ufsecp_seckey_tweak_add", "ufsecp_seckey_tweak_mul"):
            getattr(L, name).argtypes = [vp, p8, p8]
            getattr(L, name).restype = c_int

        for name in ("ufsecp_ecdsa_sign", "ufsecp_ecdsa_verify"):
            getattr(L, name).argtypes = [vp, p8, p8, p8]
            getattr(L, name).restype = c_int

        L.ufsecp_ecdsa_sig_to_der.argtypes = [vp, p8, p8, POINTER(c_size_t)]
        L.ufsecp_ecdsa_sig_to_der.restype = c_int
        L.ufsecp_ecdsa_sig_from_der.argtypes = [vp, p8, c_size_t, p8]
        L.ufsecp_ecdsa_sig_from_der.restype = c_int

        L.ufsecp_ecdsa_sign_recoverable.argtypes = [vp, p8, p8, p8, POINTER(c_int)]
        L.ufsecp_ecdsa_sign_recoverable.restype = c_int
        L.ufsecp_ecdsa_recover.argtypes = [vp, p8, p8, c_int, p8]
        L.ufsecp_ecdsa_recover.restype = c_int

        L.ufsecp_schnorr_sign.argtypes = [vp, p8, p8, p8, p8]
        L.ufsecp_schnorr_sign.restype = c_int
        L.ufsecp_schnorr_verify.argtypes = [vp, p8, p8, p8]
        L.ufsecp_schnorr_verify.restype = c_int

        for name in ("ufsecp_ecdh", "ufsecp_ecdh_xonly", "ufsecp_ecdh_raw"):
            getattr(L, name).argtypes = [vp, p8, p8, p8]
            getattr(L, name).restype = c_int

        L.ufsecp_sha256.argtypes = [p8, c_size_t, p8]
        L.ufsecp_sha256.restype = c_int
        L.ufsecp_hash160.argtypes = [p8, c_size_t, p8]
        L.ufsecp_hash160.restype = c_int
        L.ufsecp_tagged_hash.argtypes = [c_char_p, p8, c_size_t, p8]
        L.ufsecp_tagged_hash.restype = c_int

        for name in ("ufsecp_addr_p2pkh", "ufsecp_addr_p2wpkh", "ufsecp_addr_p2tr"):
            getattr(L, name).argtypes = [vp, p8, c_int, p8, POINTER(c_size_t)]
            getattr(L, name).restype = c_int

        L.ufsecp_wif_encode.argtypes = [vp, p8, c_int, c_int, p8, POINTER(c_size_t)]
        L.ufsecp_wif_encode.restype = c_int
        L.ufsecp_wif_decode.argtypes = [vp, c_char_p, p8, POINTER(c_int), POINTER(c_int)]
        L.ufsecp_wif_decode.restype = c_int

        L.ufsecp_bip32_master.argtypes = [vp, p8, c_size_t, p8]
        L.ufsecp_bip32_master.restype = c_int
        L.ufsecp_bip32_derive.argtypes = [vp, p8, c_uint32, p8]
        L.ufsecp_bip32_derive.restype = c_int
        L.ufsecp_bip32_derive_path.argtypes = [vp, p8, c_char_p, p8]
        L.ufsecp_bip32_derive_path.restype = c_int
        L.ufsecp_bip32_privkey.argtypes = [vp, p8, p8]
        L.ufsecp_bip32_privkey.restype = c_int
        L.ufsecp_bip32_pubkey.argtypes = [vp, p8, p8]
        L.ufsecp_bip32_pubkey.restype = c_int

        L.ufsecp_taproot_output_key.argtypes = [vp, p8, p8, p8, POINTER(c_int)]
        L.ufsecp_taproot_output_key.restype = c_int
        L.ufsecp_taproot_tweak_seckey.argtypes = [vp, p8, p8, p8]
        L.ufsecp_taproot_tweak_seckey.restype = c_int
        L.ufsecp_taproot_verify.argtypes = [vp, p8, c_int, p8, p8, c_size_t]
        L.ufsecp_taproot_verify.restype = c_int


def _chk(data: bytes, expected: int, name: str) -> None:
    if len(data) != expected:
        raise ValueError(f"{name} must be {expected} bytes, got {len(data)}")
