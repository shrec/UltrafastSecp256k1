"""Offline, network-free validator for the GH335 bounded public-authority capture.

Independently re-derives every semantic and cryptographic relation between
``authority_snapshot.json``, ``capture_receipt.json`` and ``READY.md`` produced
by the accepted task-439 capture tool, without importing or executing that
tool. Also provides a privilege-independent, no-replace atomic file publish
primitive used by the validator's own ``--self-check`` fixture path.

No network access is performed anywhere in this module.
"""

import argparse
import base64
import errno
import hashlib
import json
import os
import re
import secrets
import stat
import sys
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical public identity constants (accepted task-439 authority epoch).
# These describe *public* GitHub issue/comment identity and repo coordinates;
# none are secret. They pin the validator to the accepted authority capture.
# ---------------------------------------------------------------------------

HOST = "api.github.com"
REPO = "shrec/UltrafastSecp256k1"
ISSUE_PATH = f"/repos/{REPO}/issues/335"
COMMENTS_PATH = f"/repos/{REPO}/issues/335/comments"
COMMENT_PATH = f"/repos/{REPO}/issues/comments/5066918142"
ISSUE_URL = f"https://{HOST}{ISSUE_PATH}"
COMMENTS_URL = f"https://{HOST}{COMMENTS_PATH}?per_page=100&page=1"
COMMENT_URL = f"https://{HOST}{COMMENT_PATH}"
ISSUE_336_API = f"https://{HOST}/repos/{REPO}/issues/336"
COMMENT_PERMALINK = f"https://github.com/{REPO}/issues/336#issuecomment-5066918142"

TASK_ID = "GH335_BOUNDED_AUTHORITY_CAPTURE_CODEX_RESEARCH_439"
TOPIC = "gh335_bounded_authority_capture_research_v1"
RUNNER = "codex_gh335_bounded_capture_research_v1"
HEAD = "aed0ebcc5822b577d7ababa31a28172c1c38a2c4"
TREE = "2adb9ce493129f30d6d3f2db1884887fa609c27f"
CONTEXT_SHA = "13c8efea37f393bda49cedbeaa268a2c4fb0b57da31c551dc9b8bd81239cb22f"
DELIVERY_PROMPT = "Execute the exact injected GH335 bounded public-authority research capture and stop at Codex review."

RETAINED_SNAPSHOT_SHA = "61c52521f4e5ac9d5ba96535145d205e27548045da4b5a7e51055bcdb3cd6996"
RETAINED_RAW_SHA = [
    "cf64644e60f8bd236727f6c745d468f9afa54dbd3343a88f86a469ccd0b69a9c",
    "f12801132c3aa0cf703fd7b96ad452726d4c2e08c8163c794ddf713ff4fa97f2",
    "332aec43ad416bd6602bc3df622976cf34f441db29f464bc1e954390e4eb34ef",
]
_ACCEPTED_SCRIPT_SHA_PATTERN = re.compile(r"\A[0-9a-f]{64}\Z")


def _validated_script_shas(raw_shas):
    """Fail closed if any allowlisted script digest is not exactly 64
    lowercase hexadecimal characters. In a provenance allowlist, silently
    truncating (e.g. via a stray ``[:64]`` slice) an over-length, uppercase,
    or non-hex pasted value would normalize it into acceptance instead of
    rejecting the malformed entry outright."""
    for value in raw_shas:
        if not (isinstance(value, str) and _ACCEPTED_SCRIPT_SHA_PATTERN.match(value)):
            raise ValueError(
                f"ACCEPTED_SCRIPT_SHAS entry is not exactly 64 lowercase hex characters: {value!r}"
            )
    return raw_shas


ACCEPTED_SCRIPT_SHAS = _validated_script_shas({
    "aef4703b1e8b0f1bf1c62d7022a1ac8d854b1ef9309262ff4f92ba344eaf26a3",
    "7f82e09043b3307e9426e2e13ffafe1c39685aa566067da5216d41010f2e1583",
    "cd354550034af8f04ae15c34e7c46e3521d891ff7232b11ac29d9721db333e41",
    "315fbab5c95aee47ffe642e0543b1e1f81c91321a29faf9a484318f680f59fb0",
    "29a7e45b0f7ff040cb906d6dc4f32f2ab4a4a45fea9ff4b462bb722168648db5",
    "395f6de06cd8d4fe7cf0c2e465eee5f9aabef29142bf9c44ce85d73370f7ffa2",
    "993fb495a2070cb330364ed2801007e23fe951e472568e7d4cd9651e9d2c93e4",
    "dac31a5872cef71cdf7c5654abc7e563d721b65977d47f25627d000795f693e5",
    "133888c343893d39e48b70299b4ecf396a30811cf4e067b4fc60ef82a6479463",
})
# The lexicographically smallest allowlisted digest, used wherever a single
# representative member of ACCEPTED_SCRIPT_SHAS is needed (e.g. the
# self-check fixture receipt). str comparison, unlike set iteration order,
# does not depend on PYTHONHASHSEED, so this is stable across processes and
# hash seeds -- picking via next(iter(...)) is not.
CANONICAL_ACCEPTED_SCRIPT_SHA = min(ACCEPTED_SCRIPT_SHAS)
CONTEXT_ACKNOWLEDGEMENT = {
    "injected_task_contract": True,
    "project_context_acknowledged": True,
    "bundle_sha256": CONTEXT_SHA,
}
CLAIMS_EXPECTED = {
    "issue_closed": False,
    "m5_inferred": False,
    "note": "Public authority capture only; no closure or release inference.",
}
SAFE_HEADERS = {"content-type", "etag", "last-modified", "link", "x-github-api-version"}
BODY_LIMIT = 2 * 1024 * 1024
TOTAL_LIMIT = 8 * 1024 * 1024
MAX_PAGES = 20

# Bounds for reading task-root leaves off disk, enforced while streaming from
# the identity-pinned O_NOFOLLOW dirfd/fd -- independently of, and strictly
# before, TOTAL_LIMIT's own aggregate-raw-content accounting inside the
# parsed snapshot (that check only runs *after* a leaf has already been read
# fully into memory and JSON-decoded). TASK_ROOT_ARTIFACT_LIMIT is sized with
# generous margin over the largest legitimate authority_snapshot.json implied
# by TOTAL_LIMIT plus base64 (4/3) and JSON indent/quoting overhead;
# capture_receipt.json and READY.md are always far smaller but share the same
# cap for simplicity. TASK_ROOT_AGGREGATE_LIMIT bounds the sum of all three
# leaf reads in one validate_task_root call.
TASK_ROOT_ARTIFACT_LIMIT = 16 * 1024 * 1024
TASK_ROOT_AGGREGATE_LIMIT = 24 * 1024 * 1024


# ---------------------------------------------------------------------------
# Reject taxonomy
# ---------------------------------------------------------------------------

class Reject(str, Enum):
    SNAPSHOT_TYPE = "SNAPSHOT_TYPE"
    SNAPSHOT_SCHEMA = "SNAPSHOT_SCHEMA"
    SNAPSHOT_TASK_ID = "SNAPSHOT_TASK_ID"
    SNAPSHOT_CAPTURED_UTC = "SNAPSHOT_CAPTURED_UTC"
    SNAPSHOT_REQUEST_COUNT = "SNAPSHOT_REQUEST_COUNT"
    REQUEST_TYPE = "REQUEST_TYPE"
    REQUEST_URL = "REQUEST_URL"
    REQUEST_EFFECTIVE_URL = "REQUEST_EFFECTIVE_URL"
    REQUEST_METHOD = "REQUEST_METHOD"
    REQUEST_STATUS = "REQUEST_STATUS"
    REQUEST_HEADERS = "REQUEST_HEADERS"
    REQUEST_CONTENT_TYPE = "REQUEST_CONTENT_TYPE"
    REQUEST_ETAG = "REQUEST_ETAG"
    REQUEST_FETCH_UTC = "REQUEST_FETCH_UTC"
    REQUEST_BASE64 = "REQUEST_BASE64"
    REQUEST_BYTE_LENGTH = "REQUEST_BYTE_LENGTH"
    REQUEST_RAW_HASH = "REQUEST_RAW_HASH"
    REQUEST_DECODED_HASH = "REQUEST_DECODED_HASH"
    REQUEST_PAGE = "REQUEST_PAGE"
    PAGE_LINK_MALFORMED = "PAGE_LINK_MALFORMED"
    PAGE_SEQUENCE_INCOMPLETE = "PAGE_SEQUENCE_INCOMPLETE"
    AGGREGATE_BYTES = "AGGREGATE_BYTES"
    PAGE_SEQUENCE = "PAGE_SEQUENCE"
    ISSUE_IDENTITY = "ISSUE_IDENTITY"
    COMMENT_IDENTITY = "COMMENT_IDENTITY"
    DUPLICATE_COMMENT_ID = "DUPLICATE_COMMENT_ID"
    COMMENTS_COUNT_MISMATCH = "COMMENTS_COUNT_MISMATCH"
    SPECIAL_COMMENT_IDENTITY = "SPECIAL_COMMENT_IDENTITY"
    SEMANTIC_PROJECTION = "SEMANTIC_PROJECTION"
    REVIEWER_DRIFT = "REVIEWER_DRIFT"
    CLAIMS = "CLAIMS"
    AUTHORITY_SNAPSHOT_EPOCH = "AUTHORITY_SNAPSHOT_EPOCH"
    AUTHORITY_RAW_EPOCH = "AUTHORITY_RAW_EPOCH"
    RECEIPT_TYPE = "RECEIPT_TYPE"
    RECEIPT_SCHEMA = "RECEIPT_SCHEMA"
    RECEIPT_TASK_ID = "RECEIPT_TASK_ID"
    RECEIPT_TOPIC = "RECEIPT_TOPIC"
    RECEIPT_RUNNER = "RECEIPT_RUNNER"
    RECEIPT_CANONICAL_HEAD = "RECEIPT_CANONICAL_HEAD"
    RECEIPT_CANONICAL_TREE = "RECEIPT_CANONICAL_TREE"
    RECEIPT_CREATED_UTC = "RECEIPT_CREATED_UTC"
    RECEIPT_SNAPSHOT_BINDING = "RECEIPT_SNAPSHOT_BINDING"
    RECEIPT_RAW_BINDING = "RECEIPT_RAW_BINDING"
    RECEIPT_PAGE_BINDING = "RECEIPT_PAGE_BINDING"
    RECEIPT_CONTEXT = "RECEIPT_CONTEXT"
    RECEIPT_DELIVERY_PROMPT = "RECEIPT_DELIVERY_PROMPT"
    RECEIPT_SCRIPT_BINDING = "RECEIPT_SCRIPT_BINDING"
    READY_MISSING = "READY_MISSING"
    READY_CONTENT_MISMATCH = "READY_CONTENT_MISMATCH"
    TASK_ROOT_IDENTITY = "TASK_ROOT_IDENTITY"
    TASK_ROOT_ARTIFACT_TOO_LARGE = "TASK_ROOT_ARTIFACT_TOO_LARGE"


class ValidationReject(RuntimeError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = Reject(code)
        self.message = message


def reject(code, message):
    raise ValidationReject(code, message)


class Verdict:
    __slots__ = ("accepted", "code", "detail")

    def __init__(self, accepted, code, detail):
        self.accepted = accepted
        self.code = code
        self.detail = detail

    def as_dict(self):
        return {"accepted": self.accepted, "code": self.code, "detail": self.detail}

    def __eq__(self, other):
        return isinstance(other, Verdict) and self.as_dict() == other.as_dict()

    def __repr__(self):
        return f"Verdict({self.accepted!r}, {self.code!r}, {self.detail!r})"


# ---------------------------------------------------------------------------
# Small helpers (independently reimplemented; no import of the capture tool)
# ---------------------------------------------------------------------------

def sha256_hex(data):
    return hashlib.sha256(data).hexdigest()


def parse_exact_utc(value, code, message):
    if type(value) is not str:
        reject(code, message)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        reject(code, message)
        return None
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        reject(code, message)
    return parsed


def require(condition, code, message):
    if not condition:
        reject(code, message)


def require_equal(actual, expected, code, message):
    if actual != expected:
        reject(code, message)


def decode_json_strict(raw, content_type, code_bad_type, code_bad_utf8, code_bad_json):
    media = (content_type or "").split(";", 1)[0].strip().lower()
    require(media == "application/json", code_bad_type, "response is not application/json")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError:
        reject(code_bad_utf8, "invalid UTF-8")
        return None, None
    try:
        return text, json.loads(text)
    except json.JSONDecodeError:
        reject(code_bad_json, "invalid JSON")
        return None, None


def parse_link_next(link_value, prior_page):
    if not link_value:
        return None
    next_urls = []
    for part in link_value.split(","):
        match = re.fullmatch(r'\s*<([^>]+)>\s*;\s*rel="([^"]+)"\s*', part)
        if not match:
            reject(Reject.PAGE_LINK_MALFORMED, "malformed Link header")
        if "next" in match.group(2).split():
            next_urls.append(match.group(1))
    require(len(next_urls) <= 1, Reject.PAGE_LINK_MALFORMED, "duplicate Link next")
    if not next_urls:
        return None
    expected = f"https://{HOST}{COMMENTS_PATH}?per_page=100&page={prior_page + 1}"
    require_equal(next_urls[0], expected, Reject.PAGE_LINK_MALFORMED, "unexpected Link next target")
    return next_urls[0]


# ---------------------------------------------------------------------------
# Per-record and per-section verification
# ---------------------------------------------------------------------------

def _require_string(obj, key, code, message):
    value = obj.get(key)
    require(isinstance(value, str), code, message)
    return value


def _require_string_or_null(obj, key, code, message):
    value = obj.get(key)
    require(value is None or isinstance(value, str), code, message)
    return value


def _require_int(obj, key, code, message):
    value = obj.get(key)
    require(type(value) is int, code, message)
    return value


def verify_request_record(record, index, total, prior_page):
    """Bind every record -- including interior pagination pages, not just the
    fixed issue/special-comment endpoints -- to its exact canonical
    requested/effective URL. The expected page number (and therefore the
    expected ``?per_page=100&page=N`` URL) is derived solely from the prior
    page in the chain, never from the record's own claimed ``page`` field, so
    a forged requested_url/effective_url on any interior page is rejected
    against the true canonical sequence rather than against attacker-supplied
    data."""
    require(isinstance(record, dict), Reject.REQUEST_TYPE, f"request[{index}] is not an object")

    if index == 0:
        expected_url, is_page, expected_page = ISSUE_URL, False, None
    elif index == total - 1:
        expected_url, is_page, expected_page = COMMENT_URL, False, None
    else:
        is_page = True
        expected_page = 1 if prior_page is None else prior_page + 1
        expected_url = f"https://{HOST}{COMMENTS_PATH}?per_page=100&page={expected_page}"

    require_equal(record.get("requested_url"), expected_url, Reject.REQUEST_URL, f"request[{index}] requested_url mismatch")
    require_equal(record.get("effective_url"), expected_url, Reject.REQUEST_EFFECTIVE_URL, f"request[{index}] effective_url mismatch")

    require_equal(record.get("method"), "GET", Reject.REQUEST_METHOD, f"request[{index}] method mismatch")
    require_equal(record.get("status"), 200, Reject.REQUEST_STATUS, f"request[{index}] status mismatch")

    headers = record.get("safe_headers")
    require(isinstance(headers, dict), Reject.REQUEST_HEADERS, f"request[{index}] safe_headers not an object")
    for key, value in headers.items():
        require(key in SAFE_HEADERS and isinstance(value, str), Reject.REQUEST_HEADERS, f"request[{index}] safe_headers key/value rejected")
    require_equal(record.get("etag"), headers.get("etag"), Reject.REQUEST_ETAG, f"request[{index}] etag rejected")

    parse_exact_utc(record.get("fetch_utc"), Reject.REQUEST_FETCH_UTC, f"request[{index}] fetch_utc rejected")

    raw_b64 = _require_string(record, "raw_base64", Reject.REQUEST_BASE64, f"request[{index}] raw_base64 missing")
    try:
        raw = base64.b64decode(raw_b64, validate=True)
    except (ValueError, base64.binascii.Error):
        reject(Reject.REQUEST_BASE64, f"request[{index}] raw_base64 invalid")
        raw = b""
    require(len(raw) <= BODY_LIMIT, Reject.REQUEST_BYTE_LENGTH, f"request[{index}] body exceeds bound")
    require_equal(record.get("byte_length"), len(raw), Reject.REQUEST_BYTE_LENGTH, f"request[{index}] byte_length mismatch")
    require_equal(record.get("raw_sha256"), sha256_hex(raw), Reject.REQUEST_RAW_HASH, f"request[{index}] raw_sha256 mismatch")

    text, obj = decode_json_strict(
        raw, headers.get("content-type", ""),
        Reject.REQUEST_CONTENT_TYPE, Reject.REQUEST_DECODED_HASH, Reject.REQUEST_DECODED_HASH,
    )
    require_equal(record.get("decoded_utf8_body_sha256"), sha256_hex(text.encode("utf-8")), Reject.REQUEST_DECODED_HASH, f"request[{index}] decoded hash mismatch")

    page = None
    if is_page:
        page = record.get("page")
        require(type(page) is int and page == expected_page, Reject.REQUEST_PAGE, f"request[{index}] page number rejected")
    else:
        require("page" not in record, Reject.REQUEST_PAGE, f"request[{index}] unexpected page marker")

    return raw, obj, headers, page


def semantic_issue(obj):
    require(
        isinstance(obj, dict) and obj.get("number") == 335 and obj.get("repository_url") == f"https://{HOST}/repos/{REPO}",
        Reject.ISSUE_IDENTITY, "wrong issue repository/number identity",
    )
    user = obj.get("user")
    require(isinstance(user, dict), Reject.ISSUE_IDENTITY, "malformed issue user")
    login = _require_string(user, "login", Reject.ISSUE_IDENTITY, "malformed issue user login")
    _require_int(obj, "id", Reject.ISSUE_IDENTITY, "malformed issue id")
    _require_int(obj, "comments", Reject.ISSUE_IDENTITY, "malformed issue comments")
    _require_string(obj, "state", Reject.ISSUE_IDENTITY, "malformed issue state")
    _require_string(obj, "title", Reject.ISSUE_IDENTITY, "malformed issue title")
    _require_string(obj, "html_url", Reject.ISSUE_IDENTITY, "malformed issue html_url")
    # body is consumed by .encode() in reviewer_hint_drift's issue_body_sha256
    # hint -- it must be typed before that use or a truthy non-string body
    # (int/list/dict) crashes with AttributeError instead of rejecting.
    _require_string_or_null(obj, "body", Reject.ISSUE_IDENTITY, "malformed issue body")
    return {k: obj.get(k) for k in ("id", "number", "state", "title", "body", "html_url", "repository_url", "comments")} | {"user": login}


def semantic_comment(obj, issue_url):
    require(isinstance(obj, dict), Reject.COMMENT_IDENTITY, "malformed comment")
    require_equal(obj.get("issue_url"), issue_url, Reject.COMMENT_IDENTITY, "wrong comment issue identity")
    user = obj.get("user")
    require(isinstance(user, dict), Reject.COMMENT_IDENTITY, "malformed comment user")
    login = _require_string(user, "login", Reject.COMMENT_IDENTITY, "malformed comment user login")
    _require_int(obj, "id", Reject.COMMENT_IDENTITY, "malformed comment id")
    _require_string(obj, "html_url", Reject.COMMENT_IDENTITY, "malformed comment html_url")
    # body is consumed by .encode() in reviewer_hint_drift's special_body_sha256
    # hint for the #336 comment -- same crash risk as semantic_issue's body.
    _require_string_or_null(obj, "body", Reject.COMMENT_IDENTITY, "malformed comment body")
    return {k: obj.get(k) for k in ("id", "issue_url", "html_url", "body", "created_at", "updated_at")} | {"user": login}


def validate_special_comment(sem):
    require(
        sem["id"] == 5066918142 and sem["html_url"] == COMMENT_PERMALINK and sem["user"] == "craigraw",
        Reject.SPECIAL_COMMENT_IDENTITY, "wrong #336 comment/permalink/author identity",
    )


def reviewer_hint_drift(issue_sem, comments_sem, special_sem):
    hints = {
        "issue_state": ["open", issue_sem["state"]],
        "issue_id": [4892938505, issue_sem["id"]],
        "issue_user": ["craigraw", issue_sem["user"]],
        "comment_count": [4, len(comments_sem)],
        "issue_body_sha256": ["03a2831860d633f640d4feb1336247aea161147877d6ffa942f49a8f2639001d", sha256_hex((issue_sem["body"] or "").encode())],
        "special_body_sha256": ["917d4f5e35e5273f46824bde49a7e84ab63a3f98ce511cc8605b20ff1094e18e", sha256_hex((special_sem["body"] or "").encode())],
    }
    return {k: {"reviewer_hint": v[0], "observed": v[1], "matches": v[0] == v[1]} for k, v in hints.items()}


# ---------------------------------------------------------------------------
# Snapshot / receipt / READY verification
# ---------------------------------------------------------------------------

def verify_snapshot(snapshot):
    """Recompute every semantic/hash relation inside the snapshot. Returns
    (raw_hashes, page_hashes) reconstructed independently from raw bytes."""
    require(isinstance(snapshot, dict), Reject.SNAPSHOT_TYPE, "snapshot is not an object")
    require_equal(snapshot.get("schema"), "gh335.bounded-public-authority.v1", Reject.SNAPSHOT_SCHEMA, "snapshot schema rejected")
    require_equal(snapshot.get("task_id"), TASK_ID, Reject.SNAPSHOT_TASK_ID, "snapshot task identity rejected")
    parse_exact_utc(snapshot.get("captured_utc"), Reject.SNAPSHOT_CAPTURED_UTC, "snapshot captured_utc rejected")

    records = snapshot.get("ordered_requests")
    require(isinstance(records, list) and 3 <= len(records) <= MAX_PAGES + 2, Reject.SNAPSHOT_REQUEST_COUNT, "snapshot request count rejected")

    decoded, raw_hashes, page_hashes, pages, aggregate = [], [], [], [], 0
    prior_page = None
    for index, record in enumerate(records):
        raw, obj, headers, page = verify_request_record(record, index, len(records), prior_page)
        aggregate += len(raw)
        require(aggregate <= TOTAL_LIMIT, Reject.AGGREGATE_BYTES, "aggregate byte bound exceeded")
        raw_hashes.append(record["raw_sha256"])
        if page is not None:
            page_hashes.append(record["raw_sha256"])
            pages.append(page)
            prior_page = page
            next_url = parse_link_next(headers.get("link"), page)
            is_last_page = index == len(records) - 2
            if is_last_page:
                require(next_url is None, Reject.PAGE_SEQUENCE_INCOMPLETE, "pagination completeness rejected")
            else:
                require(next_url is not None, Reject.PAGE_SEQUENCE_INCOMPLETE, "pagination completeness rejected")
        decoded.append(obj)

    require_equal(snapshot.get("aggregate_bytes"), aggregate, Reject.AGGREGATE_BYTES, "aggregate_bytes rejected")
    require_equal(snapshot.get("page_sequence"), pages, Reject.PAGE_SEQUENCE, "page_sequence rejected")

    issue_sem = semantic_issue(decoded[0])
    comments_sem, seen_ids = [], set()
    for page_data in decoded[1:-1]:
        require(isinstance(page_data, list), Reject.SNAPSHOT_TYPE, "comments page is not an array")
        for item in page_data:
            sem = semantic_comment(item, ISSUE_URL)
            require(type(sem["id"]) is int and sem["id"] > 0 and sem["id"] not in seen_ids, Reject.DUPLICATE_COMMENT_ID, "duplicate/malformed comment id")
            seen_ids.add(sem["id"])
            comments_sem.append(sem)
    require(
        type(issue_sem["comments"]) is int and issue_sem["comments"] >= 0 and len(comments_sem) == issue_sem["comments"],
        Reject.COMMENTS_COUNT_MISMATCH, "comments truncated or count mismatch",
    )
    special_sem = semantic_comment(decoded[-1], ISSUE_336_API)
    validate_special_comment(special_sem)

    rebuilt_semantic = {"issue_335": issue_sem, "comments_335_ordered": comments_sem, "issue_336_comment": special_sem}
    require_equal(snapshot.get("semantic"), rebuilt_semantic, Reject.SEMANTIC_PROJECTION, "semantic projection rejected")

    drift = reviewer_hint_drift(issue_sem, comments_sem, special_sem)
    require_equal(snapshot.get("reviewer_hint_drift"), drift, Reject.REVIEWER_DRIFT, "reviewer hint drift rejected")
    require_equal(snapshot.get("claims"), CLAIMS_EXPECTED, Reject.CLAIMS, "claims rejected")

    return raw_hashes, page_hashes


def verify_authority_epoch(snapshot_bytes, raw_hashes):
    require_equal(sha256_hex(snapshot_bytes), RETAINED_SNAPSHOT_SHA, Reject.AUTHORITY_SNAPSHOT_EPOCH, "authority snapshot epoch rejected")
    require_equal(raw_hashes, RETAINED_RAW_SHA, Reject.AUTHORITY_RAW_EPOCH, "authority raw epoch rejected")


def verify_receipt(receipt, snapshot_bytes, raw_hashes, page_hashes, accepted_script_hashes, snapshot_captured_utc):
    require(isinstance(receipt, dict), Reject.RECEIPT_TYPE, "receipt is not an object")
    fixed = {
        Reject.RECEIPT_SCHEMA: ("schema", "gh335.capture-receipt.v1"),
        Reject.RECEIPT_TASK_ID: ("task_id", TASK_ID),
        Reject.RECEIPT_TOPIC: ("topic", TOPIC),
        Reject.RECEIPT_RUNNER: ("runner", RUNNER),
        Reject.RECEIPT_CANONICAL_HEAD: ("canonical_head", HEAD),
        Reject.RECEIPT_CANONICAL_TREE: ("canonical_tree", TREE),
    }
    for code, (key, expected) in fixed.items():
        require_equal(receipt.get(key), expected, code, f"receipt {key} rejected")

    parse_exact_utc(receipt.get("created_utc"), Reject.RECEIPT_CREATED_UTC, "receipt created_utc rejected")
    require_equal(
        receipt.get("created_utc"), snapshot_captured_utc,
        Reject.RECEIPT_CREATED_UTC, "receipt created_utc does not match snapshot captured_utc",
    )
    require_equal(receipt.get("snapshot_sha256"), sha256_hex(snapshot_bytes), Reject.RECEIPT_SNAPSHOT_BINDING, "receipt snapshot binding rejected")
    require_equal(receipt.get("ordered_raw_sha256"), raw_hashes, Reject.RECEIPT_RAW_BINDING, "receipt raw binding rejected")
    require_equal(receipt.get("ordered_page_sha256"), page_hashes, Reject.RECEIPT_PAGE_BINDING, "receipt page binding rejected")
    require_equal(receipt.get("request_context_acknowledgement"), CONTEXT_ACKNOWLEDGEMENT, Reject.RECEIPT_CONTEXT, "receipt context rejected")
    require_equal(receipt.get("delivery_prompt"), DELIVERY_PROMPT, Reject.RECEIPT_DELIVERY_PROMPT, "receipt delivery prompt rejected")
    require(receipt.get("script_sha256") in accepted_script_hashes, Reject.RECEIPT_SCRIPT_BINDING, "receipt script binding rejected")


def expected_ready_text(receipt, snapshot):
    tests = receipt.get("self_tests")
    requests = snapshot.get("ordered_requests")
    tests_len = len(tests) if isinstance(tests, list) else "?"
    requests_len = len(requests) if isinstance(requests, list) else "?"
    return (
        f"# READY for Codex review\n\nTask: `{TASK_ID}`\n\n"
        f"Capture command succeeded: `python3 out/.tasks/{TASK_ID}/capture_public_authority.py "
        f"--self-test --task-root out/.tasks/{TASK_ID}`\n\n"
        f"Offline exact-code fixtures passed: {tests_len}. Authority requests verified: {requests_len}. "
        "No GitHub/product mutation, issue closure, or M5 inference was performed.\n"
    )


def verify_ready(ready_text, receipt, snapshot):
    require_equal(ready_text, expected_ready_text(receipt, snapshot), Reject.READY_CONTENT_MISMATCH, "READY.md content rejected")


# ---------------------------------------------------------------------------
# No-symlink-ancestry guard (shared by task-root reads and atomic publish)
# ---------------------------------------------------------------------------

class SymlinkAncestryError(RuntimeError):
    pass


class ArtifactTooLargeError(RuntimeError):
    pass


def require_no_symlink_ancestry(path):
    """Reject if ``path`` (or any ancestor directory) is a symlink.

    Uses ``Path.absolute()``, never ``Path.resolve()``: resolve() follows
    symlinks to compute the real path, which would silently defeat this
    check for relative inputs by canonicalizing away the very symlinks it
    must detect.
    """
    path = Path(path)
    if path.is_symlink():
        raise SymlinkAncestryError(f"path is itself a symlink: {path}")
    node = path.absolute().parent
    seen = set()
    for _ in range(256):
        if node in seen:
            break
        seen.add(node)
        if node.is_symlink():
            raise SymlinkAncestryError(f"symlink found in ancestry: {node}")
        parent = node.parent
        if parent == node:
            break
        node = parent


# ---------------------------------------------------------------------------
# Task-root validation entry point
# ---------------------------------------------------------------------------

def _read_task_root_artifact(dirfd, name, max_bytes):
    """Open ``name`` inside the pinned ``dirfd`` and read up to ``max_bytes``
    of it.

    ``O_NOFOLLOW`` on this single open call is the *only* symlink check for
    the leaf — there is no separate lstat()-then-later-pathname-open step
    for an attacker to race in between. Every ancestor up to and including
    the task-root directory is already pinned by the caller's ``dirfd``
    (obtained via ``_open_dir_chain_nofollow``, itself O_NOFOLLOW at every
    component), so this closes the leaf half of the same no-follow
    guarantee the ancestor chain already provides.

    Bytes are accumulated one bounded chunk at a time and the running total
    is checked against ``max_bytes`` after every chunk, so an oversized leaf
    is rejected while still bounded in memory (the excess never grows past
    one extra chunk) rather than only after the entire file has already been
    buffered.
    """
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        fd = os.open(name, flags, dir_fd=dirfd)
    except OSError as exc:
        if exc.errno in _NOT_A_DIRECTORY_ERRNOS:
            raise SymlinkAncestryError(f"symlink found at leaf: {name}") from exc
        raise
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise SymlinkAncestryError(f"not a regular file: {name}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ArtifactTooLargeError(f"{name} exceeds {max_bytes}-byte bound")
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        _close_ambiguous(fd)


def validate_task_root(task_root):
    """Validate the three task-root artifacts using identity-pinned,
    no-follow fd/dirfd reads throughout: the directory chain from the
    filesystem root down to (and including) ``task_root`` is opened via
    ``_open_dir_chain_nofollow`` (O_NOFOLLOW at every ancestor component,
    the same primitive the atomic publish path already uses), and each leaf
    artifact is then opened by name inside that pinned ``dirfd`` with its
    own O_NOFOLLOW via ``_read_task_root_artifact``. There is no
    precondition check (lstat/``is_file``) followed later by a separate
    pathname-based open — the open call itself is the only place a symlink
    can be observed, at both the ancestor and leaf level, eliminating the
    check-to-open TOCTOU window a prior lstat()-then-``Path.read_bytes()``/
    ``read_text()`` sequence would leave open to artifact substitution.
    """
    task_root = Path(task_root)
    dirfd = None
    remaining_aggregate = TASK_ROOT_AGGREGATE_LIMIT

    def read_leaf(name):
        nonlocal remaining_aggregate
        budget = min(TASK_ROOT_ARTIFACT_LIMIT, remaining_aggregate)
        data = _read_task_root_artifact(dirfd, name, budget)
        remaining_aggregate -= len(data)
        return data

    try:
        dirfd = _open_dir_chain_nofollow(task_root, create=False)

        try:
            snapshot_bytes = read_leaf("authority_snapshot.json")
            receipt_bytes = read_leaf("capture_receipt.json")
        except FileNotFoundError:
            reject(Reject.TASK_ROOT_IDENTITY, "required artifacts missing under task root")

        try:
            ready_bytes = read_leaf("READY.md")
        except FileNotFoundError:
            reject(Reject.READY_MISSING, "READY.md missing")

        snapshot = json.loads(snapshot_bytes.decode("utf-8", "strict"))
        raw_hashes, page_hashes = verify_snapshot(snapshot)
        verify_authority_epoch(snapshot_bytes, raw_hashes)

        # Script provenance is validated only against the fixed, immutable
        # ACCEPTED_SCRIPT_SHAS allowlist below — never by hashing bytes found
        # inside the task root being validated. Self-authorizing from the
        # bundle under test would let a forged local script legitimize its
        # own receipt.
        receipt = json.loads(receipt_bytes.decode("utf-8", "strict"))
        verify_receipt(receipt, snapshot_bytes, raw_hashes, page_hashes, ACCEPTED_SCRIPT_SHAS, snapshot.get("captured_utc"))

        ready_text = ready_bytes.decode("utf-8", "strict")
        verify_ready(ready_text, receipt, snapshot)

        return Verdict(True, "ACCEPT", "all authority/receipt/READY relations verified")
    except SymlinkAncestryError as exc:
        return Verdict(False, Reject.TASK_ROOT_IDENTITY.value, str(exc))
    except ArtifactTooLargeError as exc:
        return Verdict(False, Reject.TASK_ROOT_ARTIFACT_TOO_LARGE.value, str(exc))
    except ValidationReject as exc:
        return Verdict(False, exc.code.value, exc.message)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return Verdict(False, Reject.TASK_ROOT_IDENTITY.value, f"task root read/parse failed: {exc}")
    finally:
        if dirfd is not None:
            _close_ambiguous(dirfd)


# ---------------------------------------------------------------------------
# Privilege-independent, no-replace, identity-bound atomic publish
# ---------------------------------------------------------------------------

class PublishError(RuntimeError):
    def __init__(self, code, message, leftover_tmp_name=None):
        super().__init__(message)
        self.code = code
        self.leftover_tmp_name = leftover_tmp_name


_UNSUPPORTED_LINK_ERRNOS = {
    errno.EXDEV,
    errno.EPERM,
    errno.EMLINK,
    errno.ENOENT,  # /proc/self/fd unreachable (e.g. procfs not mounted, or the
                    # source dentry was removed out from under an open fd)
    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP if hasattr(errno, "EOPNOTSUPP") else errno.EPERM),
}
_UNSUPPORTED_TMPFILE_ERRNOS = {
    errno.EISDIR,
    errno.EINVAL,
    getattr(errno, "EOPNOTSUPP", errno.EINVAL),
    getattr(errno, "ENOTSUP", errno.EINVAL),
}
_MAX_ZERO_WRITE_RETRIES = 64


def _write_all(fd, data):
    view = memoryview(data)
    total = 0
    zero_streak = 0
    while total < len(view):
        try:
            n = os.write(fd, view[total:])
        except InterruptedError:
            continue
        except OSError as exc:
            raise PublishError("WRITE_FAILED", f"os.write failed after {total} bytes: {exc}") from exc
        if n == 0:
            zero_streak += 1
            if zero_streak > _MAX_ZERO_WRITE_RETRIES:
                raise PublishError("ZERO_WRITE", "os.write returned 0 repeatedly (no progress)")
            continue
        zero_streak = 0
        total += n


def _fsync_or_raise(fd, path):
    while True:
        try:
            os.fsync(fd)
            return
        except InterruptedError:
            continue
        except OSError as exc:
            raise PublishError("FSYNC_FAILED", f"fsync failed for {path}: {exc}") from exc


def _close_ambiguous(fd):
    """POSIX close(2) leaves fd state unspecified on failure; never retry."""
    try:
        os.close(fd)
    except OSError:
        pass


_MAX_DIR_CREATE_RETRIES = 16

# open(O_DIRECTORY | O_NOFOLLOW) on a path whose trailing component is a
# symlink is documented to fail with ELOOP, but on some kernels the
# O_DIRECTORY check is evaluated first against the symlink inode itself
# (which is never a directory) and reports ENOTDIR instead. Both errnos mean
# the same thing here -- "this component is not safely usable as a
# directory, quite possibly because it is a symlink" -- so both must fail
# closed identically rather than only recognizing ELOOP.
_NOT_A_DIRECTORY_ERRNOS = {errno.ELOOP, errno.ENOTDIR}


def _open_or_create_dir_component(dirfd, name):
    """Open ``name`` inside ``dirfd`` with O_NOFOLLOW, creating it first via
    dirfd-relative ``mkdir(2)`` if it does not yet exist.

    Every step — the probe open, the creation, and the follow-up open — is
    dirfd-relative, never a bare pathname. This closes the window a
    pathname-based ``Path.mkdir(parents=True)`` would leave open: such a call
    resolves each intermediate component by name and *follows* any symlink
    an attacker plants there, silently creating directories underneath a
    foreign target. Here, if an attacker wins the race between our mkdir and
    our follow-up open by replacing the freshly created directory with a
    symlink, the follow-up open's O_NOFOLLOW observes that symlink directly
    and raises ELOOP/ENOTDIR -> SymlinkAncestryError; it is never treated as
    a successfully opened directory, so no further operation ever descends
    into it.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    for _ in range(_MAX_DIR_CREATE_RETRIES):
        try:
            return os.open(name, flags, dir_fd=dirfd)
        except FileNotFoundError:
            pass
        except OSError as exc:
            if exc.errno in _NOT_A_DIRECTORY_ERRNOS:
                raise SymlinkAncestryError(f"symlink found in ancestry: {name}") from exc
            raise
        try:
            os.mkdir(name, 0o700, dir_fd=dirfd)
        except FileExistsError:
            pass
    raise PublishError("DIR_CREATE_FAILED", f"could not create/open directory component {name!r}")


def _open_dir_chain_nofollow(path, create=False):
    """Open every ancestor component of ``path`` from the filesystem root via
    openat(2) with O_NOFOLLOW, chaining each directory fd from the previous.

    openat(O_NOFOLLOW) is atomic against the exact named entry at each level —
    there is no separate lstat-then-open race window at any point in the
    ancestry. The returned fd is a *pinned* capability for the target
    directory's identity: every subsequent name-based operation inside it
    (create, link, unlink, fsync) uses this fd, not a pathname, so it stays
    bound to the originally validated directory even if that directory's
    pathname is later renamed, replaced, or symlinked over.

    When ``create`` is set, missing intermediate components are created via
    dirfd-relative ``mkdir(2)`` — see ``_open_or_create_dir_component`` for
    why a pathname-based mkdir is unsafe here.
    """
    path = Path(path).absolute()
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    parts = path.parts
    fd = os.open(parts[0], flags)
    for part in parts[1:]:
        try:
            next_fd = _open_or_create_dir_component(fd, part) if create else os.open(part, flags, dir_fd=fd)
        except SymlinkAncestryError:
            _close_ambiguous(fd)
            raise
        except OSError as exc:
            _close_ambiguous(fd)
            if exc.errno in _NOT_A_DIRECTORY_ERRNOS:
                raise SymlinkAncestryError(f"symlink found in ancestry: {part}") from exc
            raise
        _close_ambiguous(fd)
        fd = next_fd
    return fd


def _open_tmpfile(dirfd):
    """Attempt to create an unnamed (O_TMPFILE) temp file inside the
    directory identified by ``dirfd``. Returns the fd, or ``None`` if
    O_TMPFILE is unsupported on this platform or filesystem — never raises
    for an unsupported-primitive condition, but propagates any other OSError
    raw (e.g. the directory itself having disappeared), matching
    ``_mkstemp_in_dir``'s existing convention below.

    An O_TMPFILE file has no directory entry at any point in its lifetime —
    unless and until it is deliberately linked into place by
    ``_publish_from_fd``. This eliminates the entire class of "cleanup
    victim race" bug by construction for the common case: there is never a
    named, substitutable temp entry for an attacker to race against, and
    therefore nothing this module ever needs to (mis-)clean up.
    """
    if not hasattr(os, "O_TMPFILE"):
        return None
    flags = os.O_TMPFILE | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    try:
        return os.open(".", flags, 0o600, dir_fd=dirfd)
    except OSError as exc:
        if exc.errno in _UNSUPPORTED_TMPFILE_ERRNOS:
            return None
        raise


def _mkstemp_in_dir(dirfd, name, max_attempts=64):
    """Create a uniquely-named, exclusively-owned temp file inside the
    directory identified by ``dirfd`` — never by pathname. Returns (fd, name).
    """
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0)
    for _ in range(max_attempts):
        candidate = f".{name}.{secrets.token_hex(12)}"
        try:
            fd = os.open(candidate, flags, 0o600, dir_fd=dirfd)
        except FileExistsError:
            continue
        return fd, candidate
    raise PublishError(
        "TMP_CREATE_FAILED", f"could not create unique temp name for {name!r} after {max_attempts} attempts"
    )


def _fallback_exclusive_publish(name):
    """Used only when hardlink-by-fd is unsupported on this filesystem
    (EXDEV/EPERM/EMLINK/ENOTSUP/unreachable procfs). There is no
    privilege-independent, source-identity-bound primitive available in that
    case: a name-based renameat2(RENAME_NOREPLACE) would rename whatever
    currently occupies the temp file's *name* in the directory, not the
    exact fsynced inode this process wrote and verified — and a
    stat-then-rename identity check cannot close that race, since the name
    can be substituted again between the check and the rename. Fails closed
    instead of ever publishing unauthenticated bytes under the destination
    name.
    """
    raise PublishError(
        "ATOMIC_PRIMITIVE_UNAVAILABLE",
        f"no source-identity-bound atomic publish primitive available on this "
        f"filesystem for {name!r} (hardlink-by-fd unsupported here)",
    )


def _publish_from_fd(dirfd, fd, name):
    """Publish the already-written-and-fsynced file referenced by ``fd`` into
    ``name`` inside ``dirfd``. Links from ``/proc/self/fd/{fd}`` — the exact
    inode this process wrote and fsynced — rather than from any pathname, so
    the publish is immune to substitution of a temp file's *name* between
    fsync and link (source-substitution immunity), and needs no privileged
    primitive (no ``linkat(AT_EMPTY_PATH)``, which generally requires
    ``CAP_DAC_READ_SEARCH``): following a process's own open-fd entry under
    /proc and hardlinking the target is ordinary, unprivileged linkat(2)
    behavior available for any file the process itself has open, whether
    that file currently has a directory entry (a named temp file) or not (an
    O_TMPFILE file). If that fd-bound link is unavailable, there is no safe,
    source-identity-bound way to publish instead — a bare name can have been
    substituted by an attacker in the meantime — so the publish fails closed
    rather than ever renaming an unauthenticated name into place.
    """
    proc_path = f"/proc/self/fd/{fd}"
    try:
        os.link(proc_path, name, dst_dir_fd=dirfd, follow_symlinks=True)
        return
    except FileExistsError as exc:
        raise PublishError("NO_REPLACE_VIOLATION", f"destination already exists: {name}") from exc
    except OSError as exc:
        if exc.errno not in _UNSUPPORTED_LINK_ERRNOS:
            raise PublishError("LINK_FAILED", f"os.link failed: {exc}") from exc
    _fallback_exclusive_publish(name)


def _publish_via_named_temp(dirfd, name, data):
    """Fallback used only when an unnamed (O_TMPFILE) temp file is
    unsupported for this destination directory. Creates a uniquely-named
    temp file, writes and fsyncs it, and links it into ``name`` exactly as
    the O_TMPFILE path does (via ``/proc/self/fd``, immune to source-name
    substitution).

    The temp directory entry is deliberately never removed by this
    function, whether the publish succeeds or fails: no POSIX primitive can
    unlink an entry only-if its identity still matches what this process
    created — fstat-then-unlink is not atomic, and the gap between the
    check and the unlink is exactly the window in which an attacker can
    substitute a foreign entry at that name. Rather than ever risk deleting
    a foreign entry it cannot prove ownership of, this function always
    leaves the entry in place — a harmless, uniquely-named, resumable
    leftover — and reports its name so the caller can act on the true
    outcome instead of a false assumption of a clean removal.
    """
    tmp_fd, tmp_name = _mkstemp_in_dir(dirfd, name)
    try:
        _write_all(tmp_fd, data)
        _fsync_or_raise(tmp_fd, tmp_name)
        _publish_from_fd(dirfd, tmp_fd, name)
    except PublishError as exc:
        exc.leftover_tmp_name = tmp_name
        raise
    finally:
        _close_ambiguous(tmp_fd)
    _fsync_or_raise(dirfd, name)
    return tmp_name


def _publish_within_pinned_dir(dirfd, name, data):
    try:
        st = os.stat(name, dir_fd=dirfd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        if stat.S_ISLNK(st.st_mode):
            raise PublishError("SYMLINK_ANCESTRY", f"destination is a symlink: {name}")

    tmp_fd = _open_tmpfile(dirfd)
    if tmp_fd is not None:
        tmpfile_unsupported = False
        try:
            _write_all(tmp_fd, data)
            _fsync_or_raise(tmp_fd, f"<O_TMPFILE fd {tmp_fd}>")
            _publish_from_fd(dirfd, tmp_fd, name)
        except PublishError as exc:
            if exc.code != "ATOMIC_PRIMITIVE_UNAVAILABLE":
                raise
            # Relinking this anonymous inode is unsupported for this
            # destination directory — fall through to the named-temp
            # fallback below. The O_TMPFILE inode itself needs no cleanup:
            # it was never linked into any directory, so closing its sole
            # fd (in `finally`, below) atomically discards it.
            tmpfile_unsupported = True
        finally:
            _close_ambiguous(tmp_fd)
        if not tmpfile_unsupported:
            _fsync_or_raise(dirfd, name)
            return None

    return _publish_via_named_temp(dirfd, name, data)


def publish_no_replace(dest, data):
    """Atomically publish ``data`` to ``dest``.

    Never overwrites an existing destination (no-replace), never follows a
    symlinked destination or symlinked ancestor directory (victim safety /
    no-symlink-ancestry), is immune to parent-directory and source-name
    substitution races (identity-bound: every operation after the initial
    check uses a pinned directory fd and the temp file's own fd, never a
    pathname), and requires no privilege beyond ordinary file creation in the
    destination directory.

    The temp file is preferentially created unnamed (O_TMPFILE): it never
    has a directory entry, so there is nothing for an attacker to substitute
    and nothing this module ever needs to (mis-)clean up. Only when
    O_TMPFILE is unsupported for the destination directory does this fall
    back to a uniquely-named temp file — and in that fallback, the temp
    entry is deliberately never unlinked (no atomic identity-conditional
    unlink exists in POSIX to do so safely); it is left in place as a
    truthful, resumable leftover.

    Returns ``{"leftover_tmp_name": name_or_None}`` describing the actual
    cleanup outcome — ``None`` when nothing was left behind (the common,
    O_TMPFILE case), or the leftover temp file's name when the named-temp
    fallback was used. On failure, raises ``PublishError`` carrying the same
    ``leftover_tmp_name`` attribute. If no atomic no-replace primitive is
    available on the destination filesystem at all, fails closed rather than
    exposing partial content or unlinking a name it cannot prove ownership
    of.
    """
    dest = Path(dest)
    try:
        require_no_symlink_ancestry(dest)
    except SymlinkAncestryError as exc:
        raise PublishError("SYMLINK_ANCESTRY", str(exc)) from exc

    try:
        dirfd = _open_dir_chain_nofollow(dest.parent, create=True)
    except SymlinkAncestryError as exc:
        raise PublishError("SYMLINK_ANCESTRY", str(exc)) from exc

    try:
        leftover_tmp_name = _publish_within_pinned_dir(dirfd, dest.name, data)
    finally:
        _close_ambiguous(dirfd)

    return {"leftover_tmp_name": leftover_tmp_name}


# ---------------------------------------------------------------------------
# Deterministic fixture used by --self-check (no real network capture)
# ---------------------------------------------------------------------------

def _fixture_snapshot():
    issue_body = "fixture issue body"
    special_body = "fixture special body"
    fetch_utc = "2026-08-21T01:13:48.926822Z"
    page_headers = {"content-type": "application/json; charset=utf-8"}
    issue_headers = {"content-type": "application/json; charset=utf-8"}
    comment_headers = {"content-type": "application/json; charset=utf-8"}

    issue_obj = {
        "id": 999000001, "number": 335, "state": "open", "title": "fixture",
        "body": issue_body, "html_url": "https://github.com/x", "repository_url": f"https://{HOST}/repos/{REPO}",
        "comments": 1, "user": {"login": "u"},
    }
    comments_page = [{
        "id": 1, "issue_url": ISSUE_URL, "html_url": "https://github.com/x/1",
        "body": "comment", "user": {"login": "u"},
    }]
    special_obj = {
        "id": 5066918142, "issue_url": ISSUE_336_API, "html_url": COMMENT_PERMALINK,
        "body": special_body, "user": {"login": "craigraw"},
    }

    def record(url, obj, headers, page=None):
        raw = json.dumps(obj).encode("utf-8")
        r = {
            "requested_url": url, "effective_url": url, "method": "GET", "status": 200,
            "safe_headers": dict(headers), "etag": headers.get("etag"), "fetch_utc": fetch_utc,
            "raw_base64": base64.b64encode(raw).decode("ascii"), "raw_sha256": sha256_hex(raw),
            "decoded_utf8_body_sha256": sha256_hex(raw), "byte_length": len(raw),
        }
        if page is not None:
            r["page"] = page
        return r

    records = [
        record(ISSUE_URL, issue_obj, issue_headers),
        record(COMMENTS_URL, comments_page, page_headers, page=1),
        record(COMMENT_URL, special_obj, comment_headers),
    ]
    aggregate = sum(r["byte_length"] for r in records)

    issue_sem = semantic_issue(issue_obj)
    comments_sem = [semantic_comment(comments_page[0], ISSUE_URL)]
    special_sem = semantic_comment(special_obj, ISSUE_336_API)

    snapshot = {
        "schema": "gh335.bounded-public-authority.v1", "task_id": TASK_ID, "captured_utc": fetch_utc,
        "ordered_requests": records, "aggregate_bytes": aggregate, "page_sequence": [1],
        "semantic": {"issue_335": issue_sem, "comments_335_ordered": comments_sem, "issue_336_comment": special_sem},
        "reviewer_hint_drift": reviewer_hint_drift(issue_sem, comments_sem, special_sem),
        "claims": dict(CLAIMS_EXPECTED),
    }
    return snapshot


def self_check(out_root):
    """Run one full, deterministic validation computation and publish its
    verdict into ``out_root`` (a repo-local, gitignored directory) via the
    no-replace atomic publish primitive. Returns a canonical result dict
    suitable for cross-run identity comparison."""
    out_root = Path(out_root)
    snapshot = _fixture_snapshot()
    snapshot_bytes = (json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    reloaded = json.loads(snapshot_bytes.decode("utf-8"))
    raw_hashes, page_hashes = verify_snapshot(reloaded)

    receipt = {
        "schema": "gh335.capture-receipt.v1", "task_id": TASK_ID, "topic": TOPIC, "runner": RUNNER,
        "canonical_head": HEAD, "canonical_tree": TREE, "created_utc": snapshot["captured_utc"],
        "script_sha256": CANONICAL_ACCEPTED_SCRIPT_SHA,
        "snapshot_sha256": sha256_hex(snapshot_bytes),
        "ordered_raw_sha256": raw_hashes, "ordered_page_sha256": page_hashes,
        "request_context_acknowledgement": dict(CONTEXT_ACKNOWLEDGEMENT),
        "delivery_prompt": DELIVERY_PROMPT, "self_tests": ["fixture sentinel"],
    }
    verify_receipt(receipt, snapshot_bytes, raw_hashes, page_hashes, ACCEPTED_SCRIPT_SHAS, reloaded["captured_utc"])

    ready_text = expected_ready_text(receipt, reloaded)
    verify_ready(ready_text, receipt, reloaded)

    result = {
        "schema": "gh335.validator-selfcheck.v1",
        "snapshot_sha256": sha256_hex(snapshot_bytes),
        "raw_hashes": raw_hashes,
        "page_hashes": page_hashes,
        "receipt_snapshot_binding": receipt["snapshot_sha256"],
        "verdict": "ACCEPT",
    }
    result_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    dest = out_root / "selfcheck_result.json"
    # Never unlink a pre-existing destination: publish_no_replace already
    # fails closed with NO_REPLACE_VIOLATION rather than overwrite it, and
    # that failure must be allowed to propagate so an existing victim's
    # content/mode/link-count is preserved. Callers that want a fresh
    # computation must pass a fresh, unique out_root.
    publish_outcome = publish_no_replace(dest, result_bytes)
    result["publish_cleanup"] = "no_leftover" if publish_outcome["leftover_tmp_name"] is None else "leftover_left_in_place"
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _identity_relevant(result):
    return {k: v for k, v in result.items() if k != "schema"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", help="directory containing authority_snapshot.json/capture_receipt.json/READY.md")
    parser.add_argument("--self-check", action="store_true", help="run two independent fixture computations in disjoint repo-local out/ dirs and require identity")
    parser.add_argument("--out-root", default="out/gh335_validator_selfcheck", help="repo-local root for --self-check disjoint run directories")
    args = parser.parse_args(argv)

    exit_code = 0

    if args.task_root:
        verdict = validate_task_root(args.task_root)
        print(json.dumps(verdict.as_dict(), sort_keys=True))
        if not verdict.accepted:
            exit_code = 1

    if args.self_check:
        out_root = Path(args.out_root)
        result_a = self_check(out_root / "run_a")
        result_b = self_check(out_root / "run_b")
        identical = _identity_relevant(result_a) == _identity_relevant(result_b)
        print(json.dumps({"self_check_identical": identical, "run_a": result_a, "run_b": result_b}, sort_keys=True))
        if not identical:
            exit_code = 1

    if not args.task_root and not args.self_check:
        parser.error("at least one of --task-root or --self-check is required")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
