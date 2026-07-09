#!/usr/bin/env python3
"""GPU backend native-parity gate — enforces the owner rule that any
`GpuBackend` operation implemented on one shipping backend must be native
on CUDA, OpenCL, AND Metal, plus have required C ABI exposure.

Unlike `ci/audit_gate.py`'s `check_gpu_parity` (which only checks that a
`return ...Unsupported` site carries a nearby `TODO(parity)`/`PARITY-EXCEPTION`
comment — trivially satisfied by adding a comment, and blind to overrides that
are present but merely delegate to a CPU loop instead of dispatching a GPU
kernel), this gate independently determines, per backend, whether each
`GpuBackend` virtual operation actually launches device work. It never trusts
an inline source comment as proof of parity. A gap is only accepted if it is
recorded in docs/BACKEND_ASSURANCE_MATRIX.md's "Permanent Architecture
Exceptions" table — a reviewed doc, not a code comment any single commit can
add unilaterally.

Checks performed:
  1. Enumerate every `GpuBackend` virtual operation directly from
     src/gpu/include/gpu_backend.hpp (dynamic parse — a new operation added
     later is picked up automatically, no hardcoded op list to go stale).
  2. For each operation, inspect the CudaBackend/OpenCLBackend/MetalBackend
     override in gpu_backend_cuda.cu / gpu_backend_opencl.cpp /
     gpu_backend_metal.mm and classify it as native / fallback_only /
     stub_unsupported / missing_no_override / unrecognized, based on real
     dispatch evidence in the override body (CUDA `<<<...>>>` kernel launch,
     OpenCL `clEnqueueNDRangeKernel`/context-wrapper dispatch, Metal
     `dispatch_sync`) rather than trusting comments.
  3. Any non-native status fails UNLESS the (operation, backend) pair is
     listed in docs/BACKEND_ASSURANCE_MATRIX.md's "Permanent Architecture
     Exceptions" table.
  4. Cross-check that operations requiring public C ABI exposure have a
     matching `ufsecp_gpu_*` symbol declared in include/ufsecp/ufsecp_gpu.h.
  5. Cross-check docs/BACKEND_ASSURANCE_MATRIX.md's "Public GPU ABI
     operations" Feature Matrix table: a `Y` cell for a backend that the
     code shows as non-native is a doc-overclaim failure.

Exit codes:
  0 -- every enumerated operation has genuine native CUDA/OpenCL/Metal
       coverage (or a documented, owner-approved exception), required ABI
       symbols exist, and the doc Feature Matrix does not overclaim parity.
  1 -- one or more violations found, or a declared source file is missing
       (fail-closed: an unreadable input is a failure, not a silent pass).

Usage:
  python3 ci/check_gpu_backend_parity.py            # text report, exit 0/1
  python3 ci/check_gpu_backend_parity.py --json      # machine-readable report
  python3 ci/check_gpu_backend_parity.py --list      # list every op + per-backend status, no exit-1 evaluation
  python3 ci/check_gpu_backend_parity.py -o out.json # also write JSON report to file
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LIB_ROOT = SCRIPT_DIR.parent

GPU_BACKEND_HPP = LIB_ROOT / "src/gpu/include/gpu_backend.hpp"
ABI_HEADER = LIB_ROOT / "include/ufsecp/ufsecp_gpu.h"
DOC_MATRIX = LIB_ROOT / "docs/BACKEND_ASSURANCE_MATRIX.md"

# Backend source files and the class each defines, plus the regex evidence
# that proves a real device dispatch happens in an override's body (NOT
# merely that a `PARITY-EXCEPTION`/`TODO(parity)` comment exists nearby).
BACKENDS = {
    "cuda": {
        "file": LIB_ROOT / "src/gpu/src/gpu_backend_cuda.cu",
        "class_name": "CudaBackend",
        "native_markers": [re.compile(r"<<<.*?>>>", re.DOTALL)],
    },
    "opencl": {
        "file": LIB_ROOT / "src/gpu/src/gpu_backend_opencl.cpp",
        "class_name": "OpenCLBackend",
        "native_markers": [
            re.compile(r"clEnqueueNDRangeKernel\s*\("),
            re.compile(r"\bctx_->\w+\s*\("),
        ],
    },
    "metal": {
        "file": LIB_ROOT / "src/gpu/src/gpu_backend_metal.mm",
        "class_name": "MetalBackend",
        "native_markers": [re.compile(r"\bdispatch_sync\s*\(")],
    },
}

FALLBACK_MARKERS = [
    re.compile(r"_cpu_fallback\s*\("),
    re.compile(r"\bfor\s*\("),
    re.compile(r"\bwhile\s*\("),
]

# Structural/administrative operations (gpu_backend.hpp's own "Backend
# identity" / "Device enumeration" / "Context init / teardown" / "Error
# tracking" sections, all preceding the "First-wave batch ops" comment).
# These are not batch compute operations -- there is no "GPU kernel launch"
# concept for e.g. "return backend_id()" or device enumeration via a host
# API call, so the native/fallback/stub dispatch-marker classification below
# does not apply to them. Each is pure-virtual, so the compiler already
# guarantees SOME override exists per backend; this gate only confirms the
# parser actually found one (missing would indicate a parser bug or a
# genuinely broken backend file, not a compute-parity gap).
LIFECYCLE_OPS = {
    "backend_id", "backend_name",
    "device_count", "device_info",
    "init", "shutdown", "is_ready",
    "last_error", "last_error_msg",
}

# Operations that legitimately have no direct public C ABI wrapper. This is a
# distinct concern from backend-native parity: it documents *interface
# design intent* (verified manually against src/cpu/src/ufsecp_gpu_impl.cpp),
# not a hidden gap. Any operation NOT in this set and NOT in
# ABI_SYMBOL_FOR_OP fails loudly rather than being silently skipped, so a
# newly added GpuBackend operation cannot slip through unmapped.
ABI_NOT_REQUIRED = {
    "backend_id": "never invoked via ABI; the caller supplies the backend id "
                   "directly to ufsecp_gpu_ctx_create(bid, ...) instead of "
                   "querying it back from the instance",
    "backend_name": "ufsecp_gpu_backend_name(uint32_t bid) exists but is an "
                     "independent hardcoded id->name switch, not a passthrough "
                     "of this virtual — no ABI wrapper calls this method directly",
    "shutdown": "invoked implicitly via the backend unique_ptr's destructor "
                "chain on ufsecp_gpu_ctx_destroy(); no direct ABI wrapper by design",
}

# Explicit operation -> C ABI symbol mapping (derived from
# src/cpu/src/ufsecp_gpu_impl.cpp; ABI names diverge from method names via
# "zk_" prefixes and an "opaque_rows"/"lbtc_rows" alias, so a mechanical
# name transform is not reliable). A value may be a list when more than one
# symbol name is an accepted match (e.g. a compatibility alias). If a new
# GpuBackend operation is added without a corresponding entry here, the ABI
# check fails closed with an explicit "unmapped operation" violation instead
# of silently skipping it.
ABI_SYMBOL_FOR_OP = {
    "device_count": "ufsecp_gpu_device_count",
    "device_info": "ufsecp_gpu_device_info",
    "init": "ufsecp_gpu_ctx_create",
    "is_ready": "ufsecp_gpu_is_ready",
    "last_error": "ufsecp_gpu_last_error",
    "last_error_msg": "ufsecp_gpu_last_error_msg",
    "generator_mul_batch": "ufsecp_gpu_generator_mul_batch",
    "ecdsa_verify_batch": "ufsecp_gpu_ecdsa_verify_batch",
    "ecdsa_verify_lbtc_rows": ["ufsecp_gpu_ecdsa_verify_opaque_rows", "ufsecp_gpu_ecdsa_verify_lbtc_rows"],
    "schnorr_verify_batch": "ufsecp_gpu_schnorr_verify_batch",
    "ecdsa_verify_lbtc_columns": "ufsecp_gpu_ecdsa_verify_lbtc_columns",
    "schnorr_verify_lbtc_columns": "ufsecp_gpu_schnorr_verify_lbtc_columns",
    "ecdsa_verify_collect": "ufsecp_gpu_ecdsa_verify_collect",
    "schnorr_verify_collect": "ufsecp_gpu_schnorr_verify_collect",
    "ecdh_batch": "ufsecp_gpu_ecdh_batch",
    "hash160_pubkey_batch": "ufsecp_gpu_hash160_pubkey_batch",
    "msm": "ufsecp_gpu_msm",
    "xonly_validate": "ufsecp_gpu_xonly_validate",
    "commitment_verify": "ufsecp_gpu_commitment_verify",
    "tagged_hash": "ufsecp_gpu_tagged_hash",
    "pubkey_validate": "ufsecp_gpu_pubkey_validate",
    "tagged_hash_var": "ufsecp_gpu_tagged_hash_var",
    "hash256": "ufsecp_gpu_hash256",
    "hash256_var": "ufsecp_gpu_hash256_var",
    "frost_verify_partial_batch": "ufsecp_gpu_frost_verify_partial_batch",
    "ecrecover_batch": "ufsecp_gpu_ecrecover_batch",
    "zk_knowledge_verify_batch": "ufsecp_gpu_zk_knowledge_verify_batch",
    "zk_dleq_verify_batch": "ufsecp_gpu_zk_dleq_verify_batch",
    "bulletproof_verify_batch": "ufsecp_gpu_bulletproof_verify_batch",
    "snark_witness_batch": "ufsecp_gpu_zk_ecdsa_snark_witness_batch",
    "schnorr_snark_witness_batch": "ufsecp_gpu_zk_schnorr_snark_witness_batch",
    "bip324_aead_encrypt_batch": "ufsecp_gpu_bip324_aead_encrypt_batch",
    "bip324_aead_decrypt_batch": "ufsecp_gpu_bip324_aead_decrypt_batch",
    "bip352_scan_batch": "ufsecp_gpu_bip352_scan_batch",
    "merkle_pair_hash": "ufsecp_gpu_merkle_pair_hash",
}

VIRTUAL_DECL_RE = re.compile(
    r"virtual\s+(?P<ret>(?:const\s+)?[\w:]+(?:\s*[*&])?)\s+(?P<name>[A-Za-z_]\w*)\s*\("
)


# ---------------------------------------------------------------------------
# Low-level brace/paren matching helpers (same pragmatic approach as
# ci/check_backend_parity.py's _function_block: plain char-by-char depth
# counting, no full C++ tokenizer -- this codebase's GPU backend files do not
# use string literals or char constants containing unbalanced braces/parens
# in the spans we scan).
# ---------------------------------------------------------------------------

def _brace_match(text: str, open_pos: int) -> int:
    depth = 0
    for pos in range(open_pos, len(text)):
        ch = text[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return pos
    return -1


def _paren_match(text: str, open_pos: int) -> int:
    depth = 0
    for pos in range(open_pos, len(text)):
        ch = text[pos]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return pos
    return -1


def _find_class_span(text: str, class_name: str):
    """Return (brace_start, brace_end) spanning `class <class_name> { ... }`."""
    m = re.search(r"\bclass\s+" + re.escape(class_name) + r"\b", text)
    if not m:
        return None
    brace_pos = text.find("{", m.end())
    if brace_pos < 0:
        return None
    end_pos = _brace_match(text, brace_pos)
    if end_pos < 0:
        return None
    return (brace_pos, end_pos)


def _strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


# ---------------------------------------------------------------------------
# 1. Enumerate GpuBackend operations from the header (dynamic, drift-proof)
# ---------------------------------------------------------------------------

def parse_gpu_backend_operations(hpp_text: str) -> list[dict]:
    span = _find_class_span(hpp_text, "GpuBackend")
    if span is None:
        raise RuntimeError("could not locate 'class GpuBackend { ... }' in gpu_backend.hpp")
    start, end = span
    body = hpp_text[start:end + 1]

    ops = []
    seen = set()
    for m in VIRTUAL_DECL_RE.finditer(body):
        name = m.group("name")
        ret = m.group("ret").strip()
        if name in seen or name == "GpuBackend":
            continue
        paren_open = m.end() - 1
        paren_close = _paren_match(body, paren_open)
        if paren_close < 0:
            continue

        pos = paren_close + 1
        const_m = re.match(r"\s*const\b", body[pos:pos + 20])
        if const_m:
            pos += const_m.end()

        is_pure = bool(re.match(r"\s*=\s*0\s*;", body[pos:pos + 20]))
        default_body = None
        if not is_pure:
            body_open_m = re.match(r"\s*\{", body[pos:pos + 20])
            if body_open_m:
                brace_pos = pos + body_open_m.end() - 1
                brace_end = _brace_match(body, brace_pos)
                if brace_end > 0:
                    default_body = body[brace_pos:brace_end + 1]

        seen.add(name)
        ops.append({
            "name": name,
            "return_type": ret,
            "pure_virtual": is_pure,
            "has_default_body": default_body is not None,
        })
    return ops


# ---------------------------------------------------------------------------
# 2 & 3. Per-backend override extraction + native/fallback/stub classification
# ---------------------------------------------------------------------------

def _find_override_body(class_body: str, op_name: str) -> str | None:
    for m in re.finditer(r"(?<![.\w>])\b" + re.escape(op_name) + r"\s*\(", class_body):
        start = m.start()
        prefix = class_body[max(0, start - 3):start]
        if prefix.endswith("->") or prefix.endswith("."):
            continue  # a call site (obj->op_name(...) / obj.op_name(...)), not a definition

        paren_open = m.end() - 1
        paren_close = _paren_match(class_body, paren_open)
        if paren_close < 0:
            continue

        pos = paren_close + 1
        qualifiers_m = re.match(r"\s*(?:const\s+)?(?:override\s+)?", class_body[pos:pos + 40])
        pos2 = pos + qualifiers_m.end() if qualifiers_m else pos
        body_open_m = re.match(r"\s*\{", class_body[pos2:pos2 + 20])
        if body_open_m:
            brace_pos = pos2 + body_open_m.end() - 1
            brace_end = _brace_match(class_body, brace_pos)
            if brace_end > 0:
                return class_body[brace_pos:brace_end + 1]
    return None


def classify_override(body: str | None, backend_key: str, is_lifecycle: bool = False) -> str:
    if body is None:
        return "missing_no_override"
    if is_lifecycle:
        return "lifecycle_present"

    stripped = _strip_comments(body)
    for pat in BACKENDS[backend_key]["native_markers"]:
        if pat.search(stripped):
            return "native"

    core = re.sub(r"\(void\)[^;]*;", "", stripped).strip()
    core = core.lstrip("{").rstrip("}").strip()
    if re.fullmatch(
        r"(?:return\s+(?:set_error\s*\([^;]*?)?[\w:]*Unsupported[^;]*;\s*)+",
        core,
        re.DOTALL,
    ):
        return "stub_unsupported"

    for pat in FALLBACK_MARKERS:
        if pat.search(stripped):
            return "fallback_only"

    return "unrecognized"


def load_backend_status(ops: list[dict], errors: list[str]) -> dict:
    """Return {(op_name, backend_key): status} plus populate `errors` for
    missing/unreadable declared source files (fail-closed, not advisory)."""
    status = {}
    for backend_key, info in BACKENDS.items():
        path = info["file"]
        if not path.exists():
            errors.append(f"declared backend source not found: {path.relative_to(LIB_ROOT)}")
            for op in ops:
                status[(op["name"], backend_key)] = "missing_no_override"
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        span = _find_class_span(text, info["class_name"])
        if span is None:
            errors.append(
                f"could not locate 'class {info['class_name']} {{ ... }}' in "
                f"{path.relative_to(LIB_ROOT)}"
            )
            for op in ops:
                status[(op["name"], backend_key)] = "missing_no_override"
            continue
        start, end = span
        class_body = text[start:end + 1]
        for op in ops:
            override_body = _find_override_body(class_body, op["name"])
            status[(op["name"], backend_key)] = classify_override(
                override_body, backend_key, is_lifecycle=op["name"] in LIFECYCLE_OPS
            )
    return status


# ---------------------------------------------------------------------------
# 4. Permanent Architecture Exceptions ledger (docs/BACKEND_ASSURANCE_MATRIX.md)
# ---------------------------------------------------------------------------

def parse_permanent_exceptions(doc_text: str) -> dict:
    m = re.search(
        r"##\s*Permanent Architecture Exceptions\s*\n(.*?)\n(?:---|##\s)",
        doc_text,
        re.DOTALL,
    )
    if not m:
        return {}

    exceptions = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        op_cell, backend_cell, reason_cell = cells[0], cells[1], cells[2]
        if op_cell.lower() == "operation" or set(op_cell) <= {"-", ":", " "}:
            continue  # header / separator row

        op_names = [re.sub(r"[`*]", "", o).strip() for o in op_cell.split("/")]
        backend_names = [b.strip().lower() for b in backend_cell.split("/")]
        for op_name in op_names:
            for backend_name in backend_names:
                if backend_name in BACKENDS:
                    exceptions[(op_name, backend_name)] = reason_cell
    return exceptions


# ---------------------------------------------------------------------------
# 5. C ABI exposure cross-check
# ---------------------------------------------------------------------------

def check_abi_exposure(abi_header_text: str, ops: list[dict]) -> list[dict]:
    declared = {
        m.group(1)
        for m in re.finditer(r"\b(ufsecp_gpu_\w+)\s*\(", abi_header_text)
    }

    violations = []
    for op in ops:
        name = op["name"]
        if name in ABI_NOT_REQUIRED:
            continue
        candidates = ABI_SYMBOL_FOR_OP.get(name)
        if candidates is None:
            violations.append({
                "kind": "abi_mapping_missing",
                "op": name,
                "backend": None,
                "message": (
                    f"'{name}' is a GpuBackend operation with no entry in "
                    f"ABI_SYMBOL_FOR_OP (check_gpu_backend_parity.py) -- a new "
                    f"operation was added without updating the ABI cross-check "
                    f"mapping, or it needs to be added to ABI_NOT_REQUIRED with "
                    f"a documented reason."
                ),
            })
            continue
        if isinstance(candidates, str):
            candidates = [candidates]
        if not any(c in declared for c in candidates):
            violations.append({
                "kind": "abi_missing",
                "op": name,
                "backend": None,
                "message": (
                    f"'{name}' has no matching C ABI symbol declared in "
                    f"include/ufsecp/ufsecp_gpu.h (expected one of {candidates})."
                ),
            })
    return violations


# ---------------------------------------------------------------------------
# 6. Docs Feature Matrix cross-check (docs cannot claim parity code lacks)
# ---------------------------------------------------------------------------

def parse_doc_feature_matrix(doc_text: str) -> dict:
    m = re.search(
        r"###\s*Public GPU ABI operations.*?\n\|.*?\n\|[-\s|]+\n(.*?)\n\n",
        doc_text,
        re.DOTALL,
    )
    if not m:
        return {}

    rows = {}
    for line in m.group(1).splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 6:
            continue
        func_cell = cells[0]
        cuda_cell, opencl_cell, metal_cell = cells[-3], cells[-2], cells[-1]
        for sym in re.findall(r"`(ufsecp_gpu_\w+)`", func_cell):
            rows[sym] = {"cuda": cuda_cell, "opencl": opencl_cell, "metal": metal_cell}
    return rows


def _op_for_abi_symbol_map() -> dict:
    out = {}
    for op_name, syms in ABI_SYMBOL_FOR_OP.items():
        for s in ([syms] if isinstance(syms, str) else syms):
            out[s] = op_name
    return out


def cross_check_docs(doc_rows: dict, backend_status: dict) -> list[dict]:
    op_for_symbol = _op_for_abi_symbol_map()
    violations = []
    for sym, cells in doc_rows.items():
        op_name = op_for_symbol.get(sym)
        if op_name is None:
            continue
        for backend_key in BACKENDS:
            claim = cells.get(backend_key, "").strip().upper()
            if claim != "Y":
                continue
            actual = backend_status.get((op_name, backend_key))
            if actual != "native":
                violations.append({
                    "kind": "doc_overclaim",
                    "op": op_name,
                    "backend": backend_key,
                    "message": (
                        f"docs/BACKEND_ASSURANCE_MATRIX.md 'Public GPU ABI operations' "
                        f"table claims '{sym}' is native on {backend_key} (Y) but code "
                        f"shows status={actual}. Docs cannot claim parity that code "
                        f"does not provide."
                    ),
                })
    return violations


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate() -> dict:
    errors = []

    if not GPU_BACKEND_HPP.exists():
        return {
            "overall_pass": False,
            "fatal_errors": [f"declared header not found: {GPU_BACKEND_HPP.relative_to(LIB_ROOT)}"],
            "violations": [],
            "operation_count": 0,
        }
    hpp_text = GPU_BACKEND_HPP.read_text(encoding="utf-8", errors="replace")

    try:
        ops = parse_gpu_backend_operations(hpp_text)
    except RuntimeError as exc:
        return {
            "overall_pass": False,
            "fatal_errors": [str(exc)],
            "violations": [],
            "operation_count": 0,
        }

    if not ops:
        return {
            "overall_pass": False,
            "fatal_errors": ["parsed zero GpuBackend operations -- header parser is broken or the interface changed shape"],
            "violations": [],
            "operation_count": 0,
        }

    backend_status = load_backend_status(ops, errors)

    doc_text = ""
    if DOC_MATRIX.exists():
        doc_text = DOC_MATRIX.read_text(encoding="utf-8", errors="replace")
    else:
        errors.append(f"declared doc not found: {DOC_MATRIX.relative_to(LIB_ROOT)}")
    exceptions = parse_permanent_exceptions(doc_text) if doc_text else {}

    abi_text = ""
    if ABI_HEADER.exists():
        abi_text = ABI_HEADER.read_text(encoding="utf-8", errors="replace")
    else:
        errors.append(f"declared ABI header not found: {ABI_HEADER.relative_to(LIB_ROOT)}")

    violations = []

    # Backend native-coverage check
    for op in ops:
        for backend_key in BACKENDS:
            status = backend_status.get((op["name"], backend_key), "missing_no_override")
            if status in ("native", "lifecycle_present"):
                continue
            reason = exceptions.get((op["name"], backend_key))
            if reason:
                continue
            violations.append({
                "kind": "backend_native_coverage",
                "op": op["name"],
                "backend": backend_key,
                "status": status,
                "message": (
                    f"'{op['name']}' on {backend_key} is {status} (no genuine native "
                    f"GPU dispatch evidence found in its override body), and no "
                    f"owner-approved exception is recorded in docs/BACKEND_ASSURANCE_MATRIX.md "
                    f"'Permanent Architecture Exceptions'."
                ),
            })

    # ABI exposure check
    if abi_text:
        violations.extend(check_abi_exposure(abi_text, ops))

    # Docs cross-check
    if doc_text:
        doc_rows = parse_doc_feature_matrix(doc_text)
        violations.extend(cross_check_docs(doc_rows, backend_status))

    overall_pass = len(violations) == 0 and len(errors) == 0

    return {
        "overall_pass": overall_pass,
        "fatal_errors": errors,
        "violations": violations,
        "violation_count": len(violations),
        "operation_count": len(ops),
        "operations": [op["name"] for op in ops],
        "backend_status": {f"{op}/{b}": s for (op, b), s in backend_status.items()},
        "permanent_exceptions": {f"{op}/{b}": r for (op, b), r in exceptions.items()},
    }


def run(json_mode: bool, list_mode: bool, out_file: str | None) -> int:
    report = evaluate()

    if list_mode:
        print(f"GpuBackend operations: {report.get('operation_count', 0)}")
        for op_backend, status in sorted(report.get("backend_status", {}).items()):
            print(f"  {op_backend:60s} {status}")
        return 0

    rendered = json.dumps(report, indent=2)
    if out_file:
        Path(out_file).write_text(rendered, encoding="utf-8")

    if json_mode:
        print(rendered)
    else:
        if report["fatal_errors"]:
            print("  FATAL:")
            for err in report["fatal_errors"]:
                print(f"    {err}")

        violations = report.get("violations", [])
        if violations:
            print(f"  GPU backend parity violations ({len(violations)} total):")
            for v in violations:
                loc = f"{v['op']}/{v['backend']}" if v.get("backend") else v["op"]
                print(f"    [{v['kind']}] {loc}")
                print(f"         {v['message']}")
        else:
            print(
                f"  Checked {report.get('operation_count', 0)} GpuBackend operation(s) "
                f"across CUDA/OpenCL/Metal -- no violations"
            )

        if report["overall_pass"]:
            print(f"PASS gpu-backend-parity gate ({report.get('operation_count', 0)} operations)")
        else:
            print(
                f"FAIL gpu-backend-parity gate -- {len(violations)} violation(s), "
                f"{len(report['fatal_errors'])} fatal error(s)"
            )

    return 0 if report["overall_pass"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--list", action="store_true", help="list every operation + per-backend status, no pass/fail evaluation")
    parser.add_argument("-o", dest="out_file", help="write JSON report to file")
    args = parser.parse_args()
    return run(args.json, args.list, args.out_file)


if __name__ == "__main__":
    raise SystemExit(main())
