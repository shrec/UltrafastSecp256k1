"""
UltrafastSecp256k1 -- ctypes FFI wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Loads the native shared library and exposes all C API functions
as Python methods with proper type conversion.
"""

import ctypes
import ctypes.util
import os
import platform
import warnings
from pathlib import Path
from typing import Optional, Tuple

#: Size in bytes of one flat ECDSA SNARK witness record (eprint 2025/695).
ECDSA_SNARK_WITNESS_BYTES = 760


def _find_library() -> str:
    """Locate the ultrafast_secp256k1 shared library."""
    system = platform.system()

    if system == "Windows":
        lib_names = ["ultrafast_secp256k1.dll", "libultrafast_secp256k1.dll"]
    elif system == "Darwin":
        lib_names = ["libultrafast_secp256k1.dylib", "libultrafast_secp256k1.1.dylib"]
    else:
        lib_names = ["libultrafast_secp256k1.so", "libultrafast_secp256k1.so.1"]

    # 1. Check next to this file
    base = Path(__file__).parent
    for name in lib_names:
        p = base / name
        if p.exists():
            return str(p)

    # 2. Check environment variable
    env_path = os.environ.get("ULTRAFAST_SECP256K1_LIB")
    if env_path and os.path.exists(env_path):
        return env_path

    # 3. Check common build directories (relative to bindings/python)
    root = base.parent.parent  # libs/UltrafastSecp256k1
    build_dirs = [
        root / "bindings" / "c_api" / "build",
        root / "bindings" / "c_api" / "build" / "Release",
        root / "bindings" / "c_api" / "build" / "Debug",
        root / "build_rel",
        root / "build-linux",
    ]
    for bd in build_dirs:
        for name in lib_names:
            p = bd / name
            if p.exists():
                return str(p)

    # 4. System search
    found = ctypes.util.find_library("ultrafast_secp256k1")
    if found:
        return found

    raise OSError(
        "Cannot find ultrafast_secp256k1 shared library. "
        "Set ULTRAFAST_SECP256K1_LIB environment variable to the library path, "
        "or place the library next to this Python module."
    )


class Secp256k1:
    """Python wrapper around the UltrafastSecp256k1 C API."""

    # -- Constants ---------------------------------------------------------
    NETWORK_MAINNET = 0
    NETWORK_TESTNET = 1

    def __init__(self, lib_path: Optional[str] = None):
        """Initialize the library.

        Args:
            lib_path: Path to the shared library. Auto-detected if None.
        """
        warnings.warn(
            "ultrafast_secp256k1._ffi is a legacy stateless wrapper over the variable-time c_api surface. "
            "Use bindings/python/ufsecp for the standardized context-based security model.",
            DeprecationWarning,
            stacklevel=2,
        )
        path = lib_path or _find_library()
        self._lib = ctypes.CDLL(path)
        self._setup_prototypes()

        rc = self._lib.secp256k1_init()
        if rc != 0:
            raise RuntimeError("secp256k1_init() failed: library selftest failure")

    # -- Key Operations ----------------------------------------------------

    def ec_pubkey_create(self, privkey: bytes) -> bytes:
        """Compute compressed public key (33 bytes) from private key (32 bytes)."""
        self._check_bytes(privkey, 32, "privkey")
        out = ctypes.create_string_buffer(33)
        rc = self._lib.secp256k1_ec_pubkey_create(privkey, out)
        if rc != 0:
            raise ValueError("Invalid private key")
        return out.raw

    def ec_pubkey_create_uncompressed(self, privkey: bytes) -> bytes:
        """Compute uncompressed public key (65 bytes) from private key (32 bytes)."""
        self._check_bytes(privkey, 32, "privkey")
        out = ctypes.create_string_buffer(65)
        rc = self._lib.secp256k1_ec_pubkey_create_uncompressed(privkey, out)
        if rc != 0:
            raise ValueError("Invalid private key")
        return out.raw

    def ec_pubkey_parse(self, pubkey: bytes) -> bytes:
        """Parse compressed (33) or uncompressed (65) public key. Returns compressed."""
        if len(pubkey) not in (33, 65):
            raise ValueError(f"pubkey must be 33 or 65 bytes, got {len(pubkey)}")
        out = ctypes.create_string_buffer(33)
        rc = self._lib.secp256k1_ec_pubkey_parse(pubkey, len(pubkey), out)
        if rc != 0:
            raise ValueError("Invalid public key")
        return out.raw

    def ec_seckey_verify(self, privkey: bytes) -> bool:
        """Check if a private key is valid."""
        self._check_bytes(privkey, 32, "privkey")
        return self._lib.secp256k1_ec_seckey_verify(privkey) == 1

    def ec_privkey_negate(self, privkey: bytes) -> bytes:
        """Negate a private key (mod n). Returns new key."""
        self._check_bytes(privkey, 32, "privkey")
        buf = ctypes.create_string_buffer(privkey)
        rc = self._lib.secp256k1_ec_privkey_negate(buf)
        if rc != 0:
            raise ValueError("ec_privkey_negate failed: invalid (zero) key")
        return buf.raw

    def ec_privkey_tweak_add(self, privkey: bytes, tweak: bytes) -> bytes:
        """Add tweak to private key: (key + tweak) mod n."""
        self._check_bytes(privkey, 32, "privkey")
        self._check_bytes(tweak, 32, "tweak")
        buf = ctypes.create_string_buffer(privkey)
        rc = self._lib.secp256k1_ec_privkey_tweak_add(buf, tweak)
        if rc != 0:
            raise ValueError("Tweak add resulted in invalid key")
        return buf.raw

    def ec_privkey_tweak_mul(self, privkey: bytes, tweak: bytes) -> bytes:
        """Multiply private key by tweak: (key * tweak) mod n."""
        self._check_bytes(privkey, 32, "privkey")
        self._check_bytes(tweak, 32, "tweak")
        buf = ctypes.create_string_buffer(privkey)
        rc = self._lib.secp256k1_ec_privkey_tweak_mul(buf, tweak)
        if rc != 0:
            raise ValueError("Tweak mul resulted in invalid key")
        return buf.raw

    # -- ECDSA -------------------------------------------------------------

    def ecdsa_sign(self, msg_hash: bytes, privkey: bytes) -> bytes:
        """Sign a 32-byte message hash. Returns 64-byte compact signature."""
        self._check_bytes(msg_hash, 32, "msg_hash")
        self._check_bytes(privkey, 32, "privkey")
        sig = ctypes.create_string_buffer(64)
        rc = self._lib.secp256k1_ecdsa_sign(msg_hash, privkey, sig)
        if rc != 0:
            raise ValueError("Signing failed")
        return sig.raw

    def ecdsa_verify(self, msg_hash: bytes, sig: bytes, pubkey: bytes) -> bool:
        """Verify an ECDSA signature. Returns True if valid."""
        self._check_bytes(msg_hash, 32, "msg_hash")
        self._check_bytes(sig, 64, "sig")
        self._check_bytes(pubkey, 33, "pubkey")
        return self._lib.secp256k1_ecdsa_verify(msg_hash, sig, pubkey) == 1

    def ecdsa_signature_serialize_der(self, sig: bytes) -> bytes:
        """Encode compact signature to DER format."""
        self._check_bytes(sig, 64, "sig")
        der = ctypes.create_string_buffer(72)
        der_len = ctypes.c_size_t(72)
        rc = self._lib.secp256k1_ecdsa_signature_serialize_der(
            sig, der, ctypes.byref(der_len)
        )
        if rc != 0:
            raise ValueError("DER serialization failed")
        return der.raw[: der_len.value]

    # -- ECDSA Recovery ----------------------------------------------------

    def ecdsa_sign_recoverable(self, msg_hash: bytes, privkey: bytes) -> Tuple[bytes, int]:
        """Sign with recovery id. Returns (64-byte signature, recid)."""
        self._check_bytes(msg_hash, 32, "msg_hash")
        self._check_bytes(privkey, 32, "privkey")
        sig = ctypes.create_string_buffer(64)
        recid = ctypes.c_int(0)
        rc = self._lib.secp256k1_ecdsa_sign_recoverable(
            msg_hash, privkey, sig, ctypes.byref(recid)
        )
        if rc != 0:
            raise ValueError("Recoverable signing failed")
        return sig.raw, recid.value

    def ecdsa_recover(self, msg_hash: bytes, sig: bytes, recid: int) -> bytes:
        """Recover compressed public key from recoverable signature."""
        self._check_bytes(msg_hash, 32, "msg_hash")
        self._check_bytes(sig, 64, "sig")
        pubkey = ctypes.create_string_buffer(33)
        rc = self._lib.secp256k1_ecdsa_recover(msg_hash, sig, recid, pubkey)
        if rc != 0:
            raise ValueError("Recovery failed")
        return pubkey.raw

    # -- Schnorr (BIP-340) -------------------------------------------------

    def schnorr_sign(self, msg: bytes, privkey: bytes, aux_rand: bytes) -> bytes:
        """Create Schnorr signature. Returns 64-byte signature."""
        self._check_bytes(msg, 32, "msg")
        self._check_bytes(privkey, 32, "privkey")
        self._check_bytes(aux_rand, 32, "aux_rand")
        sig = ctypes.create_string_buffer(64)
        rc = self._lib.secp256k1_schnorr_sign(msg, privkey, aux_rand, sig)
        if rc != 0:
            raise ValueError("Schnorr signing failed")
        return sig.raw

    def schnorr_verify(self, msg: bytes, sig: bytes, pubkey_x: bytes) -> bool:
        """Verify Schnorr signature. Returns True if valid."""
        self._check_bytes(msg, 32, "msg")
        self._check_bytes(sig, 64, "sig")
        self._check_bytes(pubkey_x, 32, "pubkey_x")
        return self._lib.secp256k1_schnorr_verify(msg, sig, pubkey_x) == 1

    def schnorr_pubkey(self, privkey: bytes) -> bytes:
        """Get x-only public key (32 bytes) for Schnorr."""
        self._check_bytes(privkey, 32, "privkey")
        out = ctypes.create_string_buffer(32)
        rc = self._lib.secp256k1_schnorr_pubkey(privkey, out)
        if rc != 0:
            raise ValueError("Invalid private key")
        return out.raw

    # -- ECDH --------------------------------------------------------------

    def ecdh(self, privkey: bytes, pubkey: bytes) -> bytes:
        """Compute ECDH shared secret: SHA256(compressed_point)."""
        self._check_bytes(privkey, 32, "privkey")
        self._check_bytes(pubkey, 33, "pubkey")
        out = ctypes.create_string_buffer(32)
        rc = self._lib.secp256k1_ecdh(privkey, pubkey, out)
        if rc != 0:
            raise ValueError("ECDH failed")
        return out.raw

    def ecdh_xonly(self, privkey: bytes, pubkey: bytes) -> bytes:
        """ECDH x-only: SHA256(x-coordinate)."""
        self._check_bytes(privkey, 32, "privkey")
        self._check_bytes(pubkey, 33, "pubkey")
        out = ctypes.create_string_buffer(32)
        rc = self._lib.secp256k1_ecdh_xonly(privkey, pubkey, out)
        if rc != 0:
            raise ValueError("ECDH xonly failed")
        return out.raw

    def ecdh_raw(self, privkey: bytes, pubkey: bytes) -> bytes:
        """ECDH raw: raw x-coordinate of shared point."""
        self._check_bytes(privkey, 32, "privkey")
        self._check_bytes(pubkey, 33, "pubkey")
        out = ctypes.create_string_buffer(32)
        rc = self._lib.secp256k1_ecdh_raw(privkey, pubkey, out)
        if rc != 0:
            raise ValueError("ECDH raw failed")
        return out.raw

    # -- Hashing -----------------------------------------------------------

    def sha256(self, data: bytes) -> bytes:
        """SHA-256 hash. Returns 32 bytes."""
        out = ctypes.create_string_buffer(32)
        self._lib.secp256k1_sha256(data, len(data), out)
        return out.raw

    def hash160(self, data: bytes) -> bytes:
        """HASH160: RIPEMD160(SHA256(data)). Returns 20 bytes."""
        out = ctypes.create_string_buffer(20)
        self._lib.secp256k1_hash160(data, len(data), out)
        return out.raw

    def tagged_hash(self, tag: str, data: bytes) -> bytes:
        """BIP-340 tagged hash. Returns 32 bytes."""
        tag_bytes = tag.encode("utf-8")
        out = ctypes.create_string_buffer(32)
        self._lib.secp256k1_tagged_hash(tag_bytes, data, len(data), out)
        return out.raw

    # -- Bitcoin Addresses -------------------------------------------------

    def address_p2pkh(self, pubkey: bytes, network: int = NETWORK_MAINNET) -> str:
        """Generate P2PKH address from compressed public key."""
        self._check_bytes(pubkey, 33, "pubkey")
        buf = ctypes.create_string_buffer(128)
        buf_len = ctypes.c_size_t(128)
        rc = self._lib.secp256k1_address_p2pkh(pubkey, network, buf, ctypes.byref(buf_len))
        if rc != 0:
            raise ValueError("P2PKH address generation failed")
        return buf.value.decode("ascii")

    def address_p2wpkh(self, pubkey: bytes, network: int = NETWORK_MAINNET) -> str:
        """Generate P2WPKH (SegWit v0) address from compressed public key."""
        self._check_bytes(pubkey, 33, "pubkey")
        buf = ctypes.create_string_buffer(128)
        buf_len = ctypes.c_size_t(128)
        rc = self._lib.secp256k1_address_p2wpkh(pubkey, network, buf, ctypes.byref(buf_len))
        if rc != 0:
            raise ValueError("P2WPKH address generation failed")
        return buf.value.decode("ascii")

    def address_p2tr(self, internal_key_x: bytes, network: int = NETWORK_MAINNET) -> str:
        """Generate P2TR (Taproot) address from x-only public key."""
        self._check_bytes(internal_key_x, 32, "internal_key_x")
        buf = ctypes.create_string_buffer(128)
        buf_len = ctypes.c_size_t(128)
        rc = self._lib.secp256k1_address_p2tr(internal_key_x, network, buf, ctypes.byref(buf_len))
        if rc != 0:
            raise ValueError("P2TR address generation failed")
        return buf.value.decode("ascii")

    # -- WIF ---------------------------------------------------------------

    def wif_encode(self, privkey: bytes, compressed: bool = True,
                   network: int = NETWORK_MAINNET) -> str:
        """Encode private key as WIF string."""
        self._check_bytes(privkey, 32, "privkey")
        buf = ctypes.create_string_buffer(128)
        buf_len = ctypes.c_size_t(128)
        rc = self._lib.secp256k1_wif_encode(
            privkey, 1 if compressed else 0, network, buf, ctypes.byref(buf_len)
        )
        if rc != 0:
            raise ValueError("WIF encoding failed")
        return buf.value.decode("ascii")

    def wif_decode(self, wif: str) -> Tuple[bytes, bool, int]:
        """Decode WIF string. Returns (privkey, compressed, network)."""
        wif_bytes = wif.encode("ascii")
        privkey = ctypes.create_string_buffer(32)
        compressed = ctypes.c_int(0)
        network = ctypes.c_int(0)
        rc = self._lib.secp256k1_wif_decode(
            wif_bytes, privkey, ctypes.byref(compressed), ctypes.byref(network)
        )
        if rc != 0:
            raise ValueError("WIF decoding failed")
        return privkey.raw, compressed.value == 1, network.value

    # -- BIP-32 ------------------------------------------------------------

    def bip32_master_key(self, seed: bytes) -> bytes:
        """Create master key from seed. Returns opaque 79-byte key."""
        if not (16 <= len(seed) <= 64):
            raise ValueError("Seed must be 16-64 bytes")
        key = ctypes.create_string_buffer(79)  # 78 + 1 (is_private)
        rc = self._lib.secp256k1_bip32_master_key(seed, len(seed), key)
        if rc != 0:
            raise ValueError("Master key generation failed")
        return key.raw

    def bip32_derive_path(self, master_key: bytes, path: str) -> bytes:
        """Derive key from path string. Returns opaque 79-byte key."""
        self._check_bytes(master_key, 79, "master_key")
        child = ctypes.create_string_buffer(79)
        rc = self._lib.secp256k1_bip32_derive_path(master_key, path.encode("ascii"), child)
        if rc != 0:
            raise ValueError(f"Path derivation failed: {path}")
        return child.raw

    def bip32_get_privkey(self, key: bytes) -> bytes:
        """Get private key bytes from extended key."""
        self._check_bytes(key, 79, "key")
        privkey = ctypes.create_string_buffer(32)
        rc = self._lib.secp256k1_bip32_get_privkey(key, privkey)
        if rc != 0:
            raise ValueError("Key is not a private key")
        return privkey.raw

    def bip32_get_pubkey(self, key: bytes) -> bytes:
        """Get compressed public key from extended key."""
        self._check_bytes(key, 79, "key")
        pubkey = ctypes.create_string_buffer(33)
        rc = self._lib.secp256k1_bip32_get_pubkey(key, pubkey)
        if rc != 0:
            raise ValueError("Public key extraction failed")
        return pubkey.raw

    # -- Taproot -----------------------------------------------------------

    def taproot_output_key(self, internal_key_x: bytes,
                           merkle_root: Optional[bytes] = None) -> Tuple[bytes, int]:
        """Derive Taproot output key. Returns (output_key_x, parity)."""
        self._check_bytes(internal_key_x, 32, "internal_key_x")
        out = ctypes.create_string_buffer(32)
        parity = ctypes.c_int(0)
        mr = merkle_root if merkle_root else None
        rc = self._lib.secp256k1_taproot_output_key(
            internal_key_x, mr, out, ctypes.byref(parity)
        )
        if rc != 0:
            raise ValueError("Taproot output key derivation failed")
        return out.raw, parity.value

    def taproot_tweak_privkey(self, privkey: bytes,
                              merkle_root: Optional[bytes] = None) -> bytes:
        """Tweak private key for Taproot key-path spending."""
        self._check_bytes(privkey, 32, "privkey")
        out = ctypes.create_string_buffer(32)
        mr = merkle_root if merkle_root else None
        rc = self._lib.secp256k1_taproot_tweak_privkey(privkey, mr, out)
        if rc != 0:
            raise ValueError("Taproot privkey tweaking failed")
        return out.raw

    # -- Prototypes setup --------------------------------------------------

    def _setup_prototypes(self):
        """Declare C API function signatures for type safety."""
        lib = self._lib
        u8p = ctypes.c_char_p  # uint8_t*
        sz = ctypes.c_size_t
        i = ctypes.c_int

        # version
        lib.secp256k1_version.restype = ctypes.c_char_p
        lib.secp256k1_version.argtypes = []

        # init
        lib.secp256k1_init.restype = i
        lib.secp256k1_init.argtypes = []

        # key ops
        lib.secp256k1_ec_pubkey_create.restype = i
        lib.secp256k1_ec_pubkey_create.argtypes = [u8p, u8p]

        lib.secp256k1_ec_pubkey_create_uncompressed.restype = i
        lib.secp256k1_ec_pubkey_create_uncompressed.argtypes = [u8p, u8p]

        lib.secp256k1_ec_pubkey_parse.restype = i
        lib.secp256k1_ec_pubkey_parse.argtypes = [u8p, sz, u8p]

        lib.secp256k1_ec_seckey_verify.restype = i
        lib.secp256k1_ec_seckey_verify.argtypes = [u8p]

        lib.secp256k1_ec_privkey_negate.restype = i
        lib.secp256k1_ec_privkey_negate.argtypes = [u8p]

        lib.secp256k1_ec_privkey_tweak_add.restype = i
        lib.secp256k1_ec_privkey_tweak_add.argtypes = [u8p, u8p]

        lib.secp256k1_ec_privkey_tweak_mul.restype = i
        lib.secp256k1_ec_privkey_tweak_mul.argtypes = [u8p, u8p]

        # ECDSA
        lib.secp256k1_ecdsa_sign.restype = i
        lib.secp256k1_ecdsa_sign.argtypes = [u8p, u8p, u8p]

        lib.secp256k1_ecdsa_verify.restype = i
        lib.secp256k1_ecdsa_verify.argtypes = [u8p, u8p, u8p]

        lib.secp256k1_ecdsa_signature_serialize_der.restype = i
        lib.secp256k1_ecdsa_signature_serialize_der.argtypes = [
            u8p, u8p, ctypes.POINTER(sz)
        ]

        # Recovery
        lib.secp256k1_ecdsa_sign_recoverable.restype = i
        lib.secp256k1_ecdsa_sign_recoverable.argtypes = [
            u8p, u8p, u8p, ctypes.POINTER(i)
        ]

        lib.secp256k1_ecdsa_recover.restype = i
        lib.secp256k1_ecdsa_recover.argtypes = [u8p, u8p, i, u8p]

        # Schnorr
        lib.secp256k1_schnorr_sign.restype = i
        lib.secp256k1_schnorr_sign.argtypes = [u8p, u8p, u8p, u8p]

        lib.secp256k1_schnorr_verify.restype = i
        lib.secp256k1_schnorr_verify.argtypes = [u8p, u8p, u8p]

        lib.secp256k1_schnorr_pubkey.restype = i
        lib.secp256k1_schnorr_pubkey.argtypes = [u8p, u8p]

        # ECDH
        lib.secp256k1_ecdh.restype = i
        lib.secp256k1_ecdh.argtypes = [u8p, u8p, u8p]

        lib.secp256k1_ecdh_xonly.restype = i
        lib.secp256k1_ecdh_xonly.argtypes = [u8p, u8p, u8p]

        lib.secp256k1_ecdh_raw.restype = i
        lib.secp256k1_ecdh_raw.argtypes = [u8p, u8p, u8p]

        # Hashing
        lib.secp256k1_sha256.restype = None
        lib.secp256k1_sha256.argtypes = [u8p, sz, u8p]

        lib.secp256k1_hash160.restype = None
        lib.secp256k1_hash160.argtypes = [u8p, sz, u8p]

        lib.secp256k1_tagged_hash.restype = None
        lib.secp256k1_tagged_hash.argtypes = [ctypes.c_char_p, u8p, sz, u8p]

        # Addresses
        lib.secp256k1_address_p2pkh.restype = i
        lib.secp256k1_address_p2pkh.argtypes = [u8p, i, ctypes.c_char_p, ctypes.POINTER(sz)]

        lib.secp256k1_address_p2wpkh.restype = i
        lib.secp256k1_address_p2wpkh.argtypes = [u8p, i, ctypes.c_char_p, ctypes.POINTER(sz)]

        lib.secp256k1_address_p2tr.restype = i
        lib.secp256k1_address_p2tr.argtypes = [u8p, i, ctypes.c_char_p, ctypes.POINTER(sz)]

        # WIF
        lib.secp256k1_wif_encode.restype = i
        lib.secp256k1_wif_encode.argtypes = [u8p, i, i, ctypes.c_char_p, ctypes.POINTER(sz)]

        lib.secp256k1_wif_decode.restype = i
        lib.secp256k1_wif_decode.argtypes = [
            ctypes.c_char_p, u8p, ctypes.POINTER(i), ctypes.POINTER(i)
        ]

        # BIP-32
        lib.secp256k1_bip32_master_key.restype = i
        lib.secp256k1_bip32_master_key.argtypes = [u8p, sz, u8p]

        lib.secp256k1_bip32_derive_path.restype = i
        lib.secp256k1_bip32_derive_path.argtypes = [u8p, ctypes.c_char_p, u8p]

        lib.secp256k1_bip32_get_privkey.restype = i
        lib.secp256k1_bip32_get_privkey.argtypes = [u8p, u8p]

        lib.secp256k1_bip32_get_pubkey.restype = i
        lib.secp256k1_bip32_get_pubkey.argtypes = [u8p, u8p]

        # Taproot
        lib.secp256k1_taproot_output_key.restype = i
        lib.secp256k1_taproot_output_key.argtypes = [u8p, u8p, u8p, ctypes.POINTER(i)]

        lib.secp256k1_taproot_tweak_privkey.restype = i
        lib.secp256k1_taproot_tweak_privkey.argtypes = [u8p, u8p, u8p]

        lib.secp256k1_taproot_verify_commitment.restype = i
        lib.secp256k1_taproot_verify_commitment.argtypes = [u8p, i, u8p, u8p, sz]

    @staticmethod
    def _check_bytes(data: bytes, expected_len: int, name: str):
        if not isinstance(data, (bytes, bytearray)):
            raise TypeError(f"{name} must be bytes, got {type(data).__name__}")
        if len(data) != expected_len:
            raise ValueError(f"{name} must be {expected_len} bytes, got {len(data)}")
