"""
_ufsecp.py -- shared ctypes wrapper for libufsecp
==================================================

All audit Python scripts import from here. This is the single canonical place
where the ufsecp C ABI is mapped to Python.

Key facts about the ABI:
  - Every function takes ``ufsecp_ctx*`` as first argument
  - All functions return ``ufsecp_error_t`` (int, 0 = UFSECP_OK)
  - Context is created with ufsecp_ctx_create() and destroyed with ufsecp_ctx_destroy()
  - BIP-32 key storage: ufsecp_bip32_key struct, 82 bytes total:
      data[78]      -- serialized BIP-32 key (version+depth+fpr+index+chaincode+key)
      is_private[1] -- 1 = xprv, 0 = xpub
      _pad[3]       -- reserved
    Within data[78] (standard BIP-32 layout):
      data[0:4]   version
      data[4]     depth
      data[5:9]   parent fingerprint
      data[9:13]  child number (big-endian uint32)
      data[13:45] chaincode (32 bytes)
      data[45:78] key material (33 bytes: 0x00+privkey or compressed pubkey)
"""

from __future__ import annotations

import ctypes
import struct
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# BIP-32 struct layout
# ---------------------------------------------------------------------------

UFSECP_BIP32_SERIALIZED_LEN = 78  # data[] field size


class BIP32Key(ctypes.Structure):
    """Mirrors: typedef struct { uint8_t data[78]; uint8_t is_private; uint8_t _pad[3]; } ufsecp_bip32_key;"""
    _fields_ = [
        ("data",       ctypes.c_uint8 * UFSECP_BIP32_SERIALIZED_LEN),
        ("is_private", ctypes.c_uint8),
        ("_pad",       ctypes.c_uint8 * 3),
    ]


def bip32_chaincode(key: BIP32Key) -> bytes:
    return bytes(key.data[13:45])


def bip32_child_number(key: BIP32Key) -> int:
    return struct.unpack(">I", bytes(key.data[9:13]))[0]


def bip32_key_material(key: BIP32Key) -> bytes:
    """33 bytes: 0x00+privkey (xprv) or compressed pubkey (xpub)."""
    return bytes(key.data[45:78])


def bip32_depth(key: BIP32Key) -> int:
    return key.data[4]


# ---------------------------------------------------------------------------
# Library path discovery
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_LIB_ROOT   = _SCRIPT_DIR.parent


def find_lib(hint: Optional[str] = None) -> str:
    """Locate libufsecp.so.* or libultrafast_secp256k1.so on the filesystem."""
    candidates = []
    if hint:
        candidates.append(Path(hint))
    root = _LIB_ROOT
    # Prefer the versioned .so.3 (avoids dlopen confusion)
    suite = root.parent.parent
    for bd in ["build_opencl", "build_rel", "build-cuda"]:
        candidates += [
            suite / bd / "include" / "ufsecp" / "libufsecp.so.3",
            suite / bd / "include" / "ufsecp" / "libufsecp.so",
        ]
    candidates += [
        root / "build_opencl" / "include" / "ufsecp" / "libufsecp.so.3",
        root / "build_opencl" / "include" / "ufsecp" / "libufsecp.so",
        root / "build-audit" / "include" / "ufsecp" / "libufsecp.so.3",
        root / "build-audit" / "include" / "ufsecp" / "libufsecp.so",
        root / "build" / "include" / "ufsecp" / "libufsecp.so.3",
        root / "build" / "include" / "ufsecp" / "libufsecp.so",
        root / "build-packaging-repro" / "include" / "ufsecp" / "libufsecp.so.3",
        root / "bindings" / "c_api" / "build" / "libultrafast_secp256k1.so",
    ]
    for c in candidates:
        if Path(c).exists():
            return str(c)
    raise FileNotFoundError(
        "Cannot locate libufsecp.so / libultrafast_secp256k1.so.\n"
        "Pass --lib /path/to/libufsecp.so.3 explicitly."
    )


# ---------------------------------------------------------------------------
# UfSecp — main wrapper class
# ---------------------------------------------------------------------------

class UfSecp:
    """
    Thread-unsafe ctypes wrapper. Create one instance per thread.

    Usage::
        lib = UfSecp("/path/to/libufsecp.so.3")
        pk  = lib.pubkey(sk32)
        sig = lib.sign(msg32, sk32)
        ok  = lib.verify(msg32, sig, pk)
    """

    def __init__(self, lib_path: str):
        self._raw = ctypes.CDLL(lib_path)
        self._ctx = self._create_ctx()
        self._bind()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _create_ctx(self) -> ctypes.c_void_p:
        fn = self._raw.ufsecp_ctx_create
        fn.restype  = ctypes.c_int
        fn.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        ctx = ctypes.c_void_p(0)
        rc = fn(ctypes.byref(ctx))
        if rc != 0:
            raise RuntimeError(f"ufsecp_ctx_create failed rc={rc}")
        return ctx

    def _bind(self):
        lib = self._raw
        u8p = ctypes.c_char_p
        VPP = ctypes.POINTER(ctypes.c_void_p)
        ctx_t = ctypes.c_void_p
        b32p  = ctypes.POINTER(BIP32Key)

        def _fn(name, res, args):
            f = getattr(lib, name)
            f.restype  = res
            f.argtypes = args
            return f

        _fn("ufsecp_ctx_destroy",         None,           [ctx_t])
        _fn("ufsecp_pubkey_create",        ctypes.c_int,   [ctx_t, u8p, u8p])
        _fn("ufsecp_ecdsa_sign",           ctypes.c_int,   [ctx_t, u8p, u8p, u8p])
        _fn("ufsecp_ecdsa_verify",         ctypes.c_int,   [ctx_t, u8p, u8p, u8p])
        _fn("ufsecp_ecdsa_sign_recoverable", ctypes.c_int,
            [ctx_t, u8p, u8p, u8p, ctypes.POINTER(ctypes.c_int)])
        _fn("ufsecp_ecdh",                 ctypes.c_int,   [ctx_t, u8p, u8p, u8p])
        _fn("ufsecp_bip32_master",         ctypes.c_int,   [ctx_t, u8p, ctypes.c_size_t, b32p])
        _fn("ufsecp_bip32_derive",         ctypes.c_int,   [ctx_t, b32p, ctypes.c_uint32, b32p])
        _fn("ufsecp_bip32_derive_path",    ctypes.c_int,   [ctx_t, b32p, u8p, b32p])
        _fn("ufsecp_bip32_privkey",        ctypes.c_int,   [ctx_t, b32p, u8p])
        _fn("ufsecp_bip32_pubkey",         ctypes.c_int,   [ctx_t, b32p, u8p])

    def __del__(self):
        if hasattr(self, "_raw") and hasattr(self, "_ctx") and self._ctx:
            try:
                self._raw.ufsecp_ctx_destroy(self._ctx)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # EC primitives
    # ------------------------------------------------------------------

    def pubkey(self, sk32: bytes) -> bytes:
        """Compressed 33-byte public key from 32-byte private key."""
        assert len(sk32) == 32
        out = ctypes.create_string_buffer(33)
        rc  = self._raw.ufsecp_pubkey_create(self._ctx, sk32, out)
        if rc != 0:
            raise ValueError(f"ufsecp_pubkey_create rc={rc}")
        return out.raw

    def sign(self, msg32: bytes, sk32: bytes) -> bytes:
        """64-byte compact R||S ECDSA signature (RFC 6979, low-S normalised)."""
        assert len(msg32) == 32 and len(sk32) == 32
        out = ctypes.create_string_buffer(64)
        rc  = self._raw.ufsecp_ecdsa_sign(self._ctx, msg32, sk32, out)
        if rc != 0:
            raise ValueError(f"ufsecp_ecdsa_sign rc={rc}")
        return out.raw

    def verify(self, msg32: bytes, sig64: bytes, pk33: bytes) -> bool:
        """Return True if signature is valid."""
        assert len(msg32) == 32 and len(sig64) == 64 and len(pk33) == 33
        rc = self._raw.ufsecp_ecdsa_verify(self._ctx, msg32, sig64, pk33)
        return rc == 0

    def sign_recoverable(self, msg32: bytes, sk32: bytes) -> tuple[bytes, int]:
        """Returns (sig64, recid). recid ∈ {0,1,2,3}."""
        assert len(msg32) == 32 and len(sk32) == 32
        out   = ctypes.create_string_buffer(64)
        recid = ctypes.c_int(0)
        rc    = self._raw.ufsecp_ecdsa_sign_recoverable(self._ctx, msg32, sk32, out, ctypes.byref(recid))
        if rc != 0:
            raise ValueError(f"ufsecp_ecdsa_sign_recoverable rc={rc}")
        return out.raw, recid.value

    def ecdh(self, sk32: bytes, pk33: bytes) -> bytes:
        """32-byte ECDH shared secret (SHA-256 of compressed shared point)."""
        assert len(sk32) == 32 and len(pk33) == 33
        out = ctypes.create_string_buffer(32)
        rc  = self._raw.ufsecp_ecdh(self._ctx, sk32, pk33, out)
        if rc != 0:
            raise ValueError(f"ufsecp_ecdh rc={rc}")
        return out.raw

    # ------------------------------------------------------------------
    # BIP-32 key derivation
    # ------------------------------------------------------------------

    def bip32_master(self, seed: bytes) -> BIP32Key:
        """Derive master extended key from seed (16-64 bytes)."""
        key = BIP32Key()
        rc  = self._raw.ufsecp_bip32_master(self._ctx, seed, len(seed), ctypes.byref(key))
        if rc != 0:
            raise ValueError(f"ufsecp_bip32_master rc={rc}")
        return key

    def bip32_derive(self, parent: BIP32Key, index: int) -> BIP32Key:
        """Derive single child by index (index >= 0x80000000 = hardened)."""
        child = BIP32Key()
        rc    = self._raw.ufsecp_bip32_derive(
            self._ctx, ctypes.byref(parent), ctypes.c_uint32(index), ctypes.byref(child)
        )
        if rc != 0:
            raise ValueError(f"ufsecp_bip32_derive(index={index}) rc={rc}")
        return child

    def bip32_derive_path(self, parent: BIP32Key, path: str) -> BIP32Key:
        """Derive by path string, e.g. \"m/44'/0'/0'/0/0\"."""
        child = BIP32Key()
        rc    = self._raw.ufsecp_bip32_derive_path(
            self._ctx, ctypes.byref(parent), path.encode("ascii"), ctypes.byref(child)
        )
        if rc != 0:
            raise ValueError(f"ufsecp_bip32_derive_path({path!r}) rc={rc}")
        return child

    def bip32_privkey(self, key: BIP32Key) -> bytes:
        """Extract 32-byte private key from an xprv key blob."""
        out = ctypes.create_string_buffer(32)
        rc  = self._raw.ufsecp_bip32_privkey(self._ctx, ctypes.byref(key), out)
        if rc != 0:
            raise ValueError(f"ufsecp_bip32_privkey rc={rc}")
        return out.raw

    def bip32_pubkey(self, key: BIP32Key) -> bytes:
        """Extract 33-byte compressed public key from an extended key blob."""
        out = ctypes.create_string_buffer(33)
        rc  = self._raw.ufsecp_bip32_pubkey(self._ctx, ctypes.byref(key), out)
        if rc != 0:
            raise ValueError(f"ufsecp_bip32_pubkey rc={rc}")
        return out.raw
