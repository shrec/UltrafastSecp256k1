#!/usr/bin/env python3 -u
"""
UltrafastSecp256k1 — Node Activator Server
===========================================
Runs on Linux.  Serves the challenge API for activate.html (open on Windows
with MetaMask).

Usage:
    python3 node_server.py [--port 8765] [--lib path/to/libufsecp.so]

Endpoints:
    GET  /challenge          → {"nonce": "...", "message": "..."}
    POST /activate           → {"ok": true, "address": "0x...", "recovered": "0x..."}

Requirements:  Python 3.8+   (no pip packages — stdlib only)
"""

import argparse
import ctypes
import ctypes.util
import hashlib
import http.server
import json
import os
import secrets
import struct
import sys
import threading
from pathlib import Path

# ── locate libufsecp.so ──────────────────────────────────────────────────────

def find_lib(explicit: str | None) -> str:
    if explicit:
        return explicit

    # Common build-tree locations relative to this script
    script_dir = Path(__file__).resolve().parent
    candidates = [
        # build_opencl (default build)
        script_dir.parents[3] / "build_opencl" / "include" / "ufsecp" / "libufsecp.so",
        script_dir.parents[2] / "build_opencl" / "include" / "ufsecp" / "libufsecp.so",
        # same-level build dirs
        script_dir.parents[1] / "build_opencl" / "include" / "ufsecp" / "libufsecp.so",
        # system install
        Path("/usr/local/lib/libufsecp.so"),
        Path("/usr/lib/libufsecp.so"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Fall back to ldconfig search
    name = ctypes.util.find_library("ufsecp")
    if name:
        return name

    raise FileNotFoundError(
        "libufsecp.so not found.  Pass --lib /path/to/libufsecp.so explicitly.\n"
        f"Searched:\n" + "\n".join(f"  {c}" for c in candidates)
    )


# ── ctypes bindings ──────────────────────────────────────────────────────────

UFSECP_OK = 0

class UfSecp:
    def __init__(self, lib_path: str):
        self._lib = ctypes.CDLL(lib_path)
        L = self._lib

        # ctx
        L.ufsecp_ctx_create.restype  = ctypes.c_int
        L.ufsecp_ctx_create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        L.ufsecp_ctx_destroy.restype  = None
        L.ufsecp_ctx_destroy.argtypes = [ctypes.c_void_p]

        # keccak / hash
        L.ufsecp_keccak256.restype  = ctypes.c_int
        L.ufsecp_keccak256.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                       ctypes.c_char_p]

        # eth_personal_hash
        L.ufsecp_eth_personal_hash.restype  = ctypes.c_int
        L.ufsecp_eth_personal_hash.argtypes = [ctypes.c_char_p, ctypes.c_size_t,
                                               ctypes.c_char_p]

        # eth_ecrecover
        L.ufsecp_eth_ecrecover.restype  = ctypes.c_int
        L.ufsecp_eth_ecrecover.argtypes = [
            ctypes.c_void_p,          # ctx
            ctypes.c_char_p,          # hash32
            ctypes.c_char_p,          # r32
            ctypes.c_char_p,          # s32
            ctypes.c_uint64,          # v (EIP-155 or 27/28)
            ctypes.c_char_p,          # addr20_out
        ]

        # create context
        ctx_ptr = ctypes.c_void_p()
        rc = L.ufsecp_ctx_create(ctypes.byref(ctx_ptr))
        if rc != UFSECP_OK or not ctx_ptr:
            raise RuntimeError(f"ufsecp_ctx_create failed: {rc}")
        self._ctx = ctx_ptr

    def __del__(self):
        if hasattr(self, '_ctx') and self._ctx:
            self._lib.ufsecp_ctx_destroy(self._ctx)

    def eth_personal_hash(self, message: bytes) -> bytes:
        out = ctypes.create_string_buffer(32)
        rc = self._lib.ufsecp_eth_personal_hash(message, len(message), out)
        if rc != UFSECP_OK:
            raise RuntimeError(f"eth_personal_hash error: {rc}")
        return bytes(out)

    def eth_ecrecover(self, hash32: bytes, r: bytes, s: bytes, v: int) -> bytes:
        """Returns 20-byte Ethereum address."""
        addr = ctypes.create_string_buffer(20)
        rc = self._lib.ufsecp_eth_ecrecover(
            self._ctx,
            ctypes.create_string_buffer(hash32, 32),
            ctypes.create_string_buffer(r, 32),
            ctypes.create_string_buffer(s, 32),
            ctypes.c_uint64(v),
            addr,
        )
        if rc != UFSECP_OK:
            raise RuntimeError(f"eth_ecrecover error: {rc}")
        return bytes(addr)


# ── nonce store (in-memory, single-use) ─────────────────────────────────────

_nonces: dict[str, str] = {}   # nonce → message
_nonces_lock = threading.Lock()

def issue_nonce(node_name: str) -> tuple[str, str]:
    nonce = secrets.token_hex(16)
    message = (
        f"Connect to UltrafastSecp256k1 ZK-Layer Node\n"
        f"Node: {node_name}\n"
        f"Nonce: {nonce}"
    )
    with _nonces_lock:
        _nonces[nonce] = message
    return nonce, message

def consume_nonce(nonce: str) -> str | None:
    """Pops and returns the message for a nonce, or None if not found."""
    with _nonces_lock:
        return _nonces.pop(nonce, None)


# ── HTTP request handler ─────────────────────────────────────────────────────

class Handler(http.server.BaseHTTPRequestHandler):
    ufsecp: UfSecp = None      # set by main()
    node_name: str = "node-01" # set by main()

    def log_message(self, fmt, *args):  # override to add colour
        print(f"\033[36m[{self.address_string()}]\033[0m  " + fmt % args)

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _json(self, code: int, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    # GET /challenge
    def _handle_challenge(self):
        nonce, message = issue_nonce(self.node_name)
        self._json(200, {"nonce": nonce, "message": message})

    # POST /activate
    def _handle_activate(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._json(400, {"ok": False, "error": "Invalid JSON"})
            return

        address = (body.get("address") or "").lower().strip()
        nonce   = body.get("nonce", "")
        r_hex   = body.get("r", "")
        s_hex   = body.get("s", "")
        v_raw   = body.get("v", 0)

        if not all([address, nonce, r_hex, s_hex]):
            self._json(400, {"ok": False, "error": "Missing fields"})
            return

        message = consume_nonce(nonce)
        if message is None:
            self._json(400, {"ok": False, "error": "Unknown or expired nonce"})
            return

        try:
            r_bytes = bytes.fromhex(r_hex)
            s_bytes = bytes.fromhex(s_hex)
            if len(r_bytes) != 32 or len(s_bytes) != 32:
                raise ValueError("r/s must be 32 bytes")

            # MetaMask returns v=27 or v=28; ufsecp_eth_ecrecover accepts both
            v = int(v_raw)

            msg_bytes = message.encode("utf-8")
            personal_hash = self.ufsecp.eth_personal_hash(msg_bytes)
            recovered_raw = self.ufsecp.eth_ecrecover(personal_hash, r_bytes, s_bytes, v)
            recovered = "0x" + recovered_raw.hex()

            # Normalise claimed address (strip 0x, lower)
            claimed = address.replace("0x", "").lower()
            recovered_cmp = recovered_raw.hex().lower()

            ok = (claimed == recovered_cmp)
            colour = "\033[32m" if ok else "\033[31m"
            print(f"{colour}  ecrecover: claimed={address}  recovered={recovered}  "
                  f"{'PASS ✓' if ok else 'FAIL ✗'}\033[0m")

            self._json(200, {
                "ok": ok,
                "address": address,
                "recovered": recovered,
                "error": None if ok else "Address mismatch — signature invalid",
            })
        except Exception as e:
            print(f"\033[31m  Error during ecrecover: {e}\033[0m")
            self._json(500, {"ok": False, "error": str(e)})

    def do_GET(self):
        if self.path == "/challenge":
            self._handle_challenge()
        elif self.path == "/":
            self._json(200, {"service": "UltrafastSecp256k1 Node Activator", "status": "up"})
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path == "/activate":
            self._handle_activate()
        else:
            self._json(404, {"error": "Not found"})


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="UltrafastSecp256k1 Node Activator Server")
    ap.add_argument("--port", type=int, default=8765, help="TCP port (default: 8765)")
    ap.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    ap.add_argument("--lib",  default=None, help="Path to libufsecp.so")
    ap.add_argument("--name", default="node-01", help="Node name shown in challenge")
    args = ap.parse_args()

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(" UltrafastSecp256k1 — Node Activator Server")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        lib_path = find_lib(args.lib)
        print(f" Library : {lib_path}")
    except FileNotFoundError as e:
        print(f"\033[31m\n{e}\n\033[0m")
        sys.exit(1)

    try:
        ufsecp = UfSecp(lib_path)
        print(f" ABI     : loaded OK")
    except Exception as e:
        print(f"\033[31m Failed to load library: {e}\033[0m")
        sys.exit(1)

    Handler.ufsecp    = ufsecp
    Handler.node_name = args.name

    # LAN IP hint
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        lan_ip = s.getsockname()[0]
        s.close()
    except Exception:
        lan_ip = "127.0.0.1"

    print(f" Node    : {args.name}")
    print(f" Listening on  http://{args.host}:{args.port}")
    print(f"\n Open activate.html on Windows and set backend URL to:")
    print(f"\033[93m   http://{lan_ip}:{args.port}\033[0m")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n Stopped.")


if __name__ == "__main__":
    main()
