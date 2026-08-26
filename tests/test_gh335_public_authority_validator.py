"""Regression suite for the offline GH335 public-authority validator.

Covers: strict canonical/raw/body/hash/pagination/issue/comment/provenance/UTC
validation with an isolated one-leaf mutant per reachable Reject code plus
separate multi-leaf mutants; two independent full self-check computations in
disjoint repo-local ``out/`` directories required to be identical; and the
privilege-independent, no-symlink-ancestry, identity-bound, no-replace atomic
publish primitive (EINTR, short/zero write, fsync failure, close ambiguity,
unsupported link primitive, substitution, victim safety).
"""

import base64
import copy
import errno
import importlib.util
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MODULE_PATH = _REPO_ROOT / "audit" / "gh335_public_authority_validator.py"
_ACCEPTED_TASK_ROOT = _REPO_ROOT / "out" / ".tasks" / "GH335_BOUNDED_AUTHORITY_CAPTURE_CODEX_RESEARCH_439"


def _load_module():
    spec = importlib.util.spec_from_file_location("gh335_public_authority_validator", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gh335 = _load_module()


# ---------------------------------------------------------------------------
# Shared fixture builders (independent of gh335.self_check's own wiring)
# ---------------------------------------------------------------------------

def _pristine_snapshot():
    return copy.deepcopy(gh335._fixture_snapshot())


def _replace_record_json(snapshot, index, mutate):
    """Rewrite one record's embedded JSON body, keeping raw_base64/raw_sha256/
    decoded hash/byte_length/aggregate_bytes internally self-consistent, so the
    mutation targets only the semantic identity check it is meant to trip."""
    record = snapshot["ordered_requests"][index]
    raw = base64.b64decode(record["raw_base64"])
    obj = json.loads(raw.decode("utf-8"))
    mutate(obj)
    new_raw = json.dumps(obj).encode("utf-8")
    record["raw_base64"] = base64.b64encode(new_raw).decode("ascii")
    record["raw_sha256"] = gh335.sha256_hex(new_raw)
    record["decoded_utf8_body_sha256"] = gh335.sha256_hex(new_raw)
    record["byte_length"] = len(new_raw)
    snapshot["aggregate_bytes"] = sum(r["byte_length"] for r in snapshot["ordered_requests"])


def _pristine_receipt_bundle():
    snapshot = _pristine_snapshot()
    snapshot_bytes = (json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    reloaded = json.loads(snapshot_bytes.decode("utf-8"))
    raw_hashes, page_hashes = gh335.verify_snapshot(reloaded)
    receipt = {
        "schema": "gh335.capture-receipt.v1", "task_id": gh335.TASK_ID, "topic": gh335.TOPIC,
        "runner": gh335.RUNNER, "canonical_head": gh335.HEAD, "canonical_tree": gh335.TREE,
        "created_utc": snapshot["captured_utc"], "script_sha256": gh335.CANONICAL_ACCEPTED_SCRIPT_SHA,
        "snapshot_sha256": gh335.sha256_hex(snapshot_bytes),
        "ordered_raw_sha256": raw_hashes, "ordered_page_sha256": page_hashes,
        "request_context_acknowledgement": dict(gh335.CONTEXT_ACKNOWLEDGEMENT),
        "delivery_prompt": gh335.DELIVERY_PROMPT, "self_tests": ["fixture sentinel"],
    }
    return receipt, snapshot_bytes, raw_hashes, page_hashes, reloaded


def _write_task_root(root, snapshot, receipt, ready_text):
    root.mkdir(parents=True, exist_ok=True)
    snapshot_bytes = (json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    (root / "authority_snapshot.json").write_bytes(snapshot_bytes)
    (root / "capture_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "READY.md").write_text(ready_text, encoding="utf-8")
    return snapshot_bytes


def _self_contained_task_root(root, monkeypatch):
    """Build a fully self-contained, deterministic ``validate_task_root``
    fixture from in-suite synthetic bytes only -- no dependency on the
    gitignored ``out/.tasks/<task-id>`` real accepted capture, so this works
    unmodified in a clean checkout.

    ``validate_task_root`` binds the snapshot to the module's fixed
    ``RETAINED_SNAPSHOT_SHA``/``RETAINED_RAW_SHA`` epoch pins, which are
    hardcoded to the one real accepted capture and therefore cannot match a
    synthetic snapshot's bytes. Rather than skip the epoch-pinning logic
    entirely, this monkeypatches those two module constants to the
    synthetic snapshot's own hashes for the duration of the calling test --
    the epoch-pinning *mechanism* is still exercised end-to-end via
    ``verify_authority_epoch``, only the specific pinned values are swapped
    for reproducible, offline ones.
    """
    receipt, snapshot_bytes, raw_hashes, page_hashes, reloaded = _pristine_receipt_bundle()
    monkeypatch.setattr(gh335, "RETAINED_SNAPSHOT_SHA", gh335.sha256_hex(snapshot_bytes))
    monkeypatch.setattr(gh335, "RETAINED_RAW_SHA", raw_hashes)
    ready_text = gh335.expected_ready_text(receipt, reloaded)
    _write_task_root(root, _pristine_snapshot(), receipt, ready_text)
    return receipt, snapshot_bytes, raw_hashes, page_hashes, reloaded


def _expect_reject(code, fn):
    with pytest.raises(gh335.ValidationReject) as excinfo:
        fn()
    assert excinfo.value.code is gh335.Reject(code)
    return excinfo.value


# ---------------------------------------------------------------------------
# verify_snapshot: one-leaf mutants (isolated single-field corruption each)
# ---------------------------------------------------------------------------

def _mut_snapshot_schema(s):
    s["schema"] = "forged"


def _mut_snapshot_task_id(s):
    s["task_id"] = "forged"


def _mut_snapshot_captured_utc(s):
    s["captured_utc"] = "not-a-date"


def _mut_snapshot_request_count(s):
    s["ordered_requests"] = s["ordered_requests"][:2]


def _mut_request_url(s):
    s["ordered_requests"][0]["requested_url"] = gh335.COMMENT_URL


def _mut_request_effective_url(s):
    s["ordered_requests"][0]["effective_url"] = gh335.COMMENT_URL


def _mut_request_method(s):
    s["ordered_requests"][0]["method"] = "POST"


def _mut_request_status(s):
    s["ordered_requests"][0]["status"] = 201


def _mut_request_headers(s):
    s["ordered_requests"][0]["safe_headers"] = {"authorization": "x"}


def _mut_request_content_type(s):
    s["ordered_requests"][0]["safe_headers"]["content-type"] = "text/plain"


def _mut_request_etag(s):
    s["ordered_requests"][0]["etag"] = "forged-etag"


def _mut_request_fetch_utc(s):
    s["ordered_requests"][0]["fetch_utc"] = "not-a-date"


def _mut_request_base64(s):
    s["ordered_requests"][0]["raw_base64"] = "!!!not-base64!!!"


def _mut_request_byte_length(s):
    s["ordered_requests"][0]["byte_length"] = 0


def _mut_request_raw_hash(s):
    s["ordered_requests"][0]["raw_sha256"] = "0" * 64


def _mut_request_decoded_hash(s):
    s["ordered_requests"][0]["decoded_utf8_body_sha256"] = "0" * 64


def _mut_request_page(s):
    s["ordered_requests"][1]["page"] = 5


def _mut_page_link_malformed(s):
    s["ordered_requests"][1]["safe_headers"]["link"] = "garbage-not-a-link-header"


def _mut_page_sequence_incomplete(s):
    s["ordered_requests"][1]["safe_headers"]["link"] = (
        f'<https://{gh335.HOST}{gh335.COMMENTS_PATH}?per_page=100&page=2>; rel="next"'
    )


def _mut_aggregate_bytes(s):
    s["aggregate_bytes"] = 0


def _mut_page_sequence(s):
    s["page_sequence"] = [5]


def _mut_issue_identity(s):
    _replace_record_json(s, 0, lambda obj: obj.__setitem__("number", 336))


def _mut_comment_identity(s):
    _replace_record_json(s, 1, lambda obj: obj[0].__setitem__("issue_url", gh335.COMMENT_URL))


def _mut_duplicate_comment_id(s):
    _replace_record_json(s, 1, lambda obj: obj.append(dict(obj[0])))
    _replace_record_json(s, 0, lambda obj: obj.__setitem__("comments", 2))


def _mut_comments_count_mismatch(s):
    _replace_record_json(s, 0, lambda obj: obj.__setitem__("comments", 99))


def _mut_special_comment_identity(s):
    _replace_record_json(s, 2, lambda obj: obj["user"].__setitem__("login", "mallory"))


def _mut_issue_body_malformed(s):
    """A truthy non-string issue body (int) must reject cleanly via
    ISSUE_IDENTITY, not crash reviewer_hint_drift's (body or "").encode()
    with an uncaught AttributeError."""
    _replace_record_json(s, 0, lambda obj: obj.__setitem__("body", 12345))


def _mut_special_comment_body_malformed(s):
    """Same crash risk as issue body, but for the #336 special comment
    (last ordered_requests record) consumed by special_body_sha256."""
    _replace_record_json(s, 2, lambda obj: obj.__setitem__("body", {"nested": "object"}))


def _mut_request_raw_invalid_utf8_valid_hash_length(s):
    """valid raw_sha256/byte_length, but the raw bytes themselves are not
    valid UTF-8 -- exercises decode_json_strict's UnicodeDecodeError branch
    independently of its JSONDecodeError branch."""
    record = s["ordered_requests"][0]
    raw = b"\xff\xfe\x00\x01invalid-utf8-bytes-follow"
    record["raw_base64"] = base64.b64encode(raw).decode("ascii")
    record["raw_sha256"] = gh335.sha256_hex(raw)
    record["byte_length"] = len(raw)
    s["aggregate_bytes"] = sum(r["byte_length"] for r in s["ordered_requests"])


def _mut_request_raw_invalid_json_valid_utf8(s):
    """valid raw_sha256/byte_length and valid UTF-8, but the decoded text is
    not valid JSON -- exercises decode_json_strict's JSONDecodeError branch
    independently of its UnicodeDecodeError branch."""
    record = s["ordered_requests"][0]
    raw = b'{"not": "valid json"'
    record["raw_base64"] = base64.b64encode(raw).decode("ascii")
    record["raw_sha256"] = gh335.sha256_hex(raw)
    record["byte_length"] = len(raw)
    s["aggregate_bytes"] = sum(r["byte_length"] for r in s["ordered_requests"])


def _mut_semantic_projection(s):
    s["semantic"]["issue_335"]["title"] = "forged"


def _mut_reviewer_drift(s):
    s["reviewer_hint_drift"]["issue_id"]["observed"] = 0


def _mut_claims(s):
    s["claims"]["issue_closed"] = True


SNAPSHOT_MUTATIONS = [
    ("schema", gh335.Reject.SNAPSHOT_SCHEMA, _mut_snapshot_schema),
    ("task_id", gh335.Reject.SNAPSHOT_TASK_ID, _mut_snapshot_task_id),
    ("captured_utc", gh335.Reject.SNAPSHOT_CAPTURED_UTC, _mut_snapshot_captured_utc),
    ("request_count", gh335.Reject.SNAPSHOT_REQUEST_COUNT, _mut_snapshot_request_count),
    ("request_url", gh335.Reject.REQUEST_URL, _mut_request_url),
    ("request_effective_url", gh335.Reject.REQUEST_EFFECTIVE_URL, _mut_request_effective_url),
    ("request_method", gh335.Reject.REQUEST_METHOD, _mut_request_method),
    ("request_status", gh335.Reject.REQUEST_STATUS, _mut_request_status),
    ("request_headers", gh335.Reject.REQUEST_HEADERS, _mut_request_headers),
    ("request_content_type", gh335.Reject.REQUEST_CONTENT_TYPE, _mut_request_content_type),
    ("request_etag", gh335.Reject.REQUEST_ETAG, _mut_request_etag),
    ("request_fetch_utc", gh335.Reject.REQUEST_FETCH_UTC, _mut_request_fetch_utc),
    ("request_base64", gh335.Reject.REQUEST_BASE64, _mut_request_base64),
    ("request_byte_length", gh335.Reject.REQUEST_BYTE_LENGTH, _mut_request_byte_length),
    ("request_raw_hash", gh335.Reject.REQUEST_RAW_HASH, _mut_request_raw_hash),
    ("request_decoded_hash", gh335.Reject.REQUEST_DECODED_HASH, _mut_request_decoded_hash),
    ("request_page", gh335.Reject.REQUEST_PAGE, _mut_request_page),
    ("page_link_malformed", gh335.Reject.PAGE_LINK_MALFORMED, _mut_page_link_malformed),
    ("page_sequence_incomplete", gh335.Reject.PAGE_SEQUENCE_INCOMPLETE, _mut_page_sequence_incomplete),
    ("aggregate_bytes", gh335.Reject.AGGREGATE_BYTES, _mut_aggregate_bytes),
    ("page_sequence", gh335.Reject.PAGE_SEQUENCE, _mut_page_sequence),
    ("issue_identity", gh335.Reject.ISSUE_IDENTITY, _mut_issue_identity),
    ("comment_identity", gh335.Reject.COMMENT_IDENTITY, _mut_comment_identity),
    ("duplicate_comment_id", gh335.Reject.DUPLICATE_COMMENT_ID, _mut_duplicate_comment_id),
    ("comments_count_mismatch", gh335.Reject.COMMENTS_COUNT_MISMATCH, _mut_comments_count_mismatch),
    ("special_comment_identity", gh335.Reject.SPECIAL_COMMENT_IDENTITY, _mut_special_comment_identity),
    ("semantic_projection", gh335.Reject.SEMANTIC_PROJECTION, _mut_semantic_projection),
    ("reviewer_drift", gh335.Reject.REVIEWER_DRIFT, _mut_reviewer_drift),
    ("claims", gh335.Reject.CLAIMS, _mut_claims),
    ("issue_body_malformed", gh335.Reject.ISSUE_IDENTITY, _mut_issue_body_malformed),
    ("special_comment_body_malformed", gh335.Reject.COMMENT_IDENTITY, _mut_special_comment_body_malformed),
    ("request_raw_invalid_utf8_valid_hash_length", gh335.Reject.REQUEST_DECODED_HASH, _mut_request_raw_invalid_utf8_valid_hash_length),
    ("request_raw_invalid_json_valid_utf8", gh335.Reject.REQUEST_DECODED_HASH, _mut_request_raw_invalid_json_valid_utf8),
]


@pytest.mark.parametrize("label,code,mutate", SNAPSHOT_MUTATIONS, ids=[m[0] for m in SNAPSHOT_MUTATIONS])
def test_verify_snapshot_one_leaf_mutant(label, code, mutate):
    snapshot = _pristine_snapshot()
    mutate(snapshot)
    _expect_reject(code, lambda: gh335.verify_snapshot(snapshot))


def test_verify_snapshot_pristine_accepts():
    gh335.verify_snapshot(_pristine_snapshot())


def test_verify_snapshot_type_rejected():
    _expect_reject(gh335.Reject.SNAPSHOT_TYPE, lambda: gh335.verify_snapshot("not-a-dict"))


def test_verify_snapshot_request_type_rejected():
    snapshot = _pristine_snapshot()
    snapshot["ordered_requests"][0] = "not-a-dict"
    _expect_reject(gh335.Reject.REQUEST_TYPE, lambda: gh335.verify_snapshot(snapshot))


def test_verify_snapshot_multi_leaf_mutant_first_check_wins():
    """Two simultaneous corruptions in the same record: byte_length is
    checked before raw_sha256, so REQUEST_BYTE_LENGTH must fire even though
    raw_sha256 is independently wrong too."""
    snapshot = _pristine_snapshot()
    snapshot["ordered_requests"][0]["byte_length"] = 0
    snapshot["ordered_requests"][0]["raw_sha256"] = "1" * 64
    _expect_reject(gh335.Reject.REQUEST_BYTE_LENGTH, lambda: gh335.verify_snapshot(snapshot))


def test_verify_snapshot_multi_leaf_mutant_across_sections():
    """Corrupting both a request field and the top-level claims block at the
    same time must still reject on the earlier (request-level) check."""
    snapshot = _pristine_snapshot()
    snapshot["ordered_requests"][0]["status"] = 500
    snapshot["claims"]["issue_closed"] = True
    _expect_reject(gh335.Reject.REQUEST_STATUS, lambda: gh335.verify_snapshot(snapshot))


# ---------------------------------------------------------------------------
# verify_request_record: interior pagination page requested_url/effective_url
# provenance. gh335._fixture_snapshot() only ever has a single comments page
# (page=1), which cannot distinguish an *interior* page (not the first, not
# the last record) from the terminal one, so this uses an independent
# two-page in-suite fixture built entirely from pinned public facts/bytes.
# ---------------------------------------------------------------------------

def _paginated_record(url, obj, headers, fetch_utc, page=None):
    raw = json.dumps(obj).encode("utf-8")
    record = {
        "requested_url": url, "effective_url": url, "method": "GET", "status": 200,
        "safe_headers": dict(headers), "etag": headers.get("etag"), "fetch_utc": fetch_utc,
        "raw_base64": base64.b64encode(raw).decode("ascii"), "raw_sha256": gh335.sha256_hex(raw),
        "decoded_utf8_body_sha256": gh335.sha256_hex(raw), "byte_length": len(raw),
    }
    if page is not None:
        record["page"] = page
    return record


def _multi_page_snapshot():
    fetch_utc = "2026-08-21T01:13:48.926822Z"
    headers = {"content-type": "application/json; charset=utf-8"}
    page1_headers = dict(headers, link=f'<https://{gh335.HOST}{gh335.COMMENTS_PATH}?per_page=100&page=2>; rel="next"')
    page2_headers = dict(headers)

    issue_obj = {
        "id": 999000001, "number": 335, "state": "open", "title": "fixture",
        "body": "fixture issue body", "html_url": "https://github.com/x",
        "repository_url": f"https://{gh335.HOST}/repos/{gh335.REPO}", "comments": 2, "user": {"login": "u"},
    }
    page1_comments = [{"id": 1, "issue_url": gh335.ISSUE_URL, "html_url": "https://github.com/x/1", "body": "c1", "user": {"login": "u"}}]
    page2_comments = [{"id": 2, "issue_url": gh335.ISSUE_URL, "html_url": "https://github.com/x/2", "body": "c2", "user": {"login": "u"}}]
    special_obj = {
        "id": 5066918142, "issue_url": gh335.ISSUE_336_API, "html_url": gh335.COMMENT_PERMALINK,
        "body": "fixture special body", "user": {"login": "craigraw"},
    }

    page1_url = gh335.COMMENTS_URL
    page2_url = f"https://{gh335.HOST}{gh335.COMMENTS_PATH}?per_page=100&page=2"

    records = [
        _paginated_record(gh335.ISSUE_URL, issue_obj, headers, fetch_utc),
        _paginated_record(page1_url, page1_comments, page1_headers, fetch_utc, page=1),
        _paginated_record(page2_url, page2_comments, page2_headers, fetch_utc, page=2),
        _paginated_record(gh335.COMMENT_URL, special_obj, headers, fetch_utc),
    ]
    aggregate = sum(r["byte_length"] for r in records)

    issue_sem = gh335.semantic_issue(issue_obj)
    comments_sem = [
        gh335.semantic_comment(page1_comments[0], gh335.ISSUE_URL),
        gh335.semantic_comment(page2_comments[0], gh335.ISSUE_URL),
    ]
    special_sem = gh335.semantic_comment(special_obj, gh335.ISSUE_336_API)

    return {
        "schema": "gh335.bounded-public-authority.v1", "task_id": gh335.TASK_ID, "captured_utc": fetch_utc,
        "ordered_requests": records, "aggregate_bytes": aggregate, "page_sequence": [1, 2],
        "semantic": {"issue_335": issue_sem, "comments_335_ordered": comments_sem, "issue_336_comment": special_sem},
        "reviewer_hint_drift": gh335.reviewer_hint_drift(issue_sem, comments_sem, special_sem),
        "claims": dict(gh335.CLAIMS_EXPECTED),
    }


def test_verify_snapshot_multi_page_pristine_accepts():
    gh335.verify_snapshot(_multi_page_snapshot())


def _mut_interior_page_requested_url(s):
    s["ordered_requests"][1]["requested_url"] = "https://api.github.com/repos/other/repo/issues/335/comments?per_page=100&page=1"


def _mut_interior_page_effective_url(s):
    s["ordered_requests"][1]["effective_url"] = "https://api.github.com/repos/other/repo/issues/335/comments?per_page=100&page=1"


def _mut_interior_page_requested_url_missing(s):
    del s["ordered_requests"][1]["requested_url"]


def _mut_interior_page_requested_url_wrong_page_number(s):
    """requested_url points at a *plausible but wrong* page (page=2 for what
    is actually the page=1 interior record) -- must reject against the true
    canonical sequence, not merely against an obviously-foreign host/path."""
    s["ordered_requests"][1]["requested_url"] = f"https://{gh335.HOST}{gh335.COMMENTS_PATH}?per_page=100&page=2"


INTERIOR_PAGE_URL_MUTATIONS = [
    ("interior_page_requested_url", gh335.Reject.REQUEST_URL, _mut_interior_page_requested_url),
    ("interior_page_effective_url", gh335.Reject.REQUEST_EFFECTIVE_URL, _mut_interior_page_effective_url),
    ("interior_page_requested_url_missing", gh335.Reject.REQUEST_URL, _mut_interior_page_requested_url_missing),
    ("interior_page_requested_url_wrong_page_number", gh335.Reject.REQUEST_URL, _mut_interior_page_requested_url_wrong_page_number),
]


@pytest.mark.parametrize("label,code,mutate", INTERIOR_PAGE_URL_MUTATIONS, ids=[m[0] for m in INTERIOR_PAGE_URL_MUTATIONS])
def test_verify_snapshot_interior_page_url_mutant(label, code, mutate):
    snapshot = _multi_page_snapshot()
    mutate(snapshot)
    _expect_reject(code, lambda: gh335.verify_snapshot(snapshot))


# ---------------------------------------------------------------------------
# verify_receipt: one-leaf mutants
# ---------------------------------------------------------------------------

def _mut_receipt_field(key, value):
    def mutate(r):
        r[key] = value
    return mutate


RECEIPT_MUTATIONS = [
    ("schema", gh335.Reject.RECEIPT_SCHEMA, _mut_receipt_field("schema", "forged")),
    ("task_id", gh335.Reject.RECEIPT_TASK_ID, _mut_receipt_field("task_id", "forged")),
    ("topic", gh335.Reject.RECEIPT_TOPIC, _mut_receipt_field("topic", "forged")),
    ("runner", gh335.Reject.RECEIPT_RUNNER, _mut_receipt_field("runner", "forged")),
    ("canonical_head", gh335.Reject.RECEIPT_CANONICAL_HEAD, _mut_receipt_field("canonical_head", "f" * 40)),
    ("canonical_tree", gh335.Reject.RECEIPT_CANONICAL_TREE, _mut_receipt_field("canonical_tree", "f" * 40)),
    ("created_utc", gh335.Reject.RECEIPT_CREATED_UTC, _mut_receipt_field("created_utc", "not-a-date")),
    ("snapshot_binding", gh335.Reject.RECEIPT_SNAPSHOT_BINDING, _mut_receipt_field("snapshot_sha256", "0" * 64)),
    ("raw_binding", gh335.Reject.RECEIPT_RAW_BINDING, _mut_receipt_field("ordered_raw_sha256", [])),
    ("page_binding", gh335.Reject.RECEIPT_PAGE_BINDING, _mut_receipt_field("ordered_page_sha256", ["0" * 64])),
    ("context", gh335.Reject.RECEIPT_CONTEXT, _mut_receipt_field("request_context_acknowledgement", {})),
    ("delivery_prompt", gh335.Reject.RECEIPT_DELIVERY_PROMPT, _mut_receipt_field("delivery_prompt", "")),
    ("script_binding", gh335.Reject.RECEIPT_SCRIPT_BINDING, _mut_receipt_field("script_sha256", "0" * 64)),
]


@pytest.mark.parametrize("label,code,mutate", RECEIPT_MUTATIONS, ids=[m[0] for m in RECEIPT_MUTATIONS])
def test_verify_receipt_one_leaf_mutant(label, code, mutate):
    receipt, snapshot_bytes, raw_hashes, page_hashes, reloaded = _pristine_receipt_bundle()
    mutated = copy.deepcopy(receipt)
    mutate(mutated)
    _expect_reject(
        code,
        lambda: gh335.verify_receipt(
            mutated, snapshot_bytes, raw_hashes, page_hashes, gh335.ACCEPTED_SCRIPT_SHAS, reloaded["captured_utc"],
        ),
    )


def test_verify_receipt_created_utc_binding_valid_but_different_utc_rejected():
    """created_utc must be exactly bound to the snapshot's captured_utc, not
    merely well-formed: a valid ISO-8601 UTC timestamp that simply differs
    from the snapshot's captured_utc must still be rejected."""
    receipt, snapshot_bytes, raw_hashes, page_hashes, reloaded = _pristine_receipt_bundle()
    mutated = copy.deepcopy(receipt)
    assert mutated["created_utc"] != "2099-01-01T00:00:00.000000Z"
    mutated["created_utc"] = "2099-01-01T00:00:00.000000Z"
    _expect_reject(
        gh335.Reject.RECEIPT_CREATED_UTC,
        lambda: gh335.verify_receipt(
            mutated, snapshot_bytes, raw_hashes, page_hashes, gh335.ACCEPTED_SCRIPT_SHAS, reloaded["captured_utc"],
        ),
    )


def test_verify_receipt_pristine_accepts():
    receipt, snapshot_bytes, raw_hashes, page_hashes, reloaded = _pristine_receipt_bundle()
    gh335.verify_receipt(receipt, snapshot_bytes, raw_hashes, page_hashes, gh335.ACCEPTED_SCRIPT_SHAS, reloaded["captured_utc"])


def test_verify_receipt_type_rejected():
    _receipt, snapshot_bytes, raw_hashes, page_hashes, reloaded = _pristine_receipt_bundle()
    _expect_reject(
        gh335.Reject.RECEIPT_TYPE,
        lambda: gh335.verify_receipt(
            "not-a-dict", snapshot_bytes, raw_hashes, page_hashes, gh335.ACCEPTED_SCRIPT_SHAS, reloaded["captured_utc"],
        ),
    )


def test_verify_receipt_multi_leaf_mutant_first_check_wins():
    """schema is checked before runner in the fixed-field iteration order."""
    receipt, snapshot_bytes, raw_hashes, page_hashes, reloaded = _pristine_receipt_bundle()
    mutated = copy.deepcopy(receipt)
    mutated["schema"] = "forged"
    mutated["runner"] = "forged"
    _expect_reject(
        gh335.Reject.RECEIPT_SCHEMA,
        lambda: gh335.verify_receipt(
            mutated, snapshot_bytes, raw_hashes, page_hashes, gh335.ACCEPTED_SCRIPT_SHAS, reloaded["captured_utc"],
        ),
    )


# ---------------------------------------------------------------------------
# verify_ready
# ---------------------------------------------------------------------------

def test_verify_ready_pristine_accepts():
    receipt, _, _, _, snapshot = _pristine_receipt_bundle()
    ready_text = gh335.expected_ready_text(receipt, snapshot)
    gh335.verify_ready(ready_text, receipt, snapshot)


def test_verify_ready_content_mismatch_rejected():
    receipt, _, _, _, snapshot = _pristine_receipt_bundle()
    ready_text = gh335.expected_ready_text(receipt, snapshot).replace("READY", "NOT READY")
    _expect_reject(gh335.Reject.READY_CONTENT_MISMATCH, lambda: gh335.verify_ready(ready_text, receipt, snapshot))


# ---------------------------------------------------------------------------
# Authority epoch pinning: the *mechanism* is exercised with an in-suite
# synthetic snapshot and its own matching hashes (via monkeypatched
# RETAINED_SNAPSHOT_SHA/RETAINED_RAW_SHA), so this is fully self-contained
# and does not require the real accepted task-439 capture to be present.
# ---------------------------------------------------------------------------

def test_authority_epoch_pristine_accepts(monkeypatch):
    _, snapshot_bytes, raw_hashes, _, _ = _pristine_receipt_bundle()
    monkeypatch.setattr(gh335, "RETAINED_SNAPSHOT_SHA", gh335.sha256_hex(snapshot_bytes))
    monkeypatch.setattr(gh335, "RETAINED_RAW_SHA", raw_hashes)
    gh335.verify_authority_epoch(snapshot_bytes, raw_hashes)


def test_authority_snapshot_epoch_rejected(monkeypatch):
    _, snapshot_bytes, raw_hashes, _, _ = _pristine_receipt_bundle()
    monkeypatch.setattr(gh335, "RETAINED_SNAPSHOT_SHA", gh335.sha256_hex(snapshot_bytes))
    monkeypatch.setattr(gh335, "RETAINED_RAW_SHA", raw_hashes)
    _expect_reject(
        gh335.Reject.AUTHORITY_SNAPSHOT_EPOCH,
        lambda: gh335.verify_authority_epoch(snapshot_bytes + b"x", raw_hashes),
    )


def test_authority_raw_epoch_rejected(monkeypatch):
    _, snapshot_bytes, raw_hashes, _, _ = _pristine_receipt_bundle()
    monkeypatch.setattr(gh335, "RETAINED_SNAPSHOT_SHA", gh335.sha256_hex(snapshot_bytes))
    monkeypatch.setattr(gh335, "RETAINED_RAW_SHA", raw_hashes)
    _expect_reject(
        gh335.Reject.AUTHORITY_RAW_EPOCH,
        lambda: gh335.verify_authority_epoch(snapshot_bytes, ["0" * 64]),
    )


# ---------------------------------------------------------------------------
# validate_task_root: end-to-end, including task-root-identity/READY-missing
# and full-pipeline multi-leaf mutants, all against a self-contained in-suite
# fixture (see _self_contained_task_root) so this section runs unmodified in
# a clean checkout without out/.tasks/<task-id> present. A separate,
# explicitly gated optional real-artifact integration section appears near
# the end of this file.
# ---------------------------------------------------------------------------

def test_validate_task_root_accepts_self_contained_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "accepted"
    _self_contained_task_root(root, monkeypatch)
    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is True
    assert verdict.code == "ACCEPT"


def test_validate_task_root_missing_files_rejected(tmp_path):
    root = tmp_path / "empty_root"
    root.mkdir()
    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.TASK_ROOT_IDENTITY.value


def test_validate_task_root_ready_missing_rejected(tmp_path):
    root = tmp_path / "no_ready"
    root.mkdir()
    (root / "authority_snapshot.json").write_text("{}", encoding="utf-8")
    (root / "capture_receipt.json").write_text("{}", encoding="utf-8")
    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.READY_MISSING.value


def test_validate_task_root_symlinked_ancestor_rejected(tmp_path, monkeypatch):
    victim = tmp_path / "victim"
    _self_contained_task_root(victim, monkeypatch)
    evil_link = tmp_path / "evil_link"
    evil_link.symlink_to(victim, target_is_directory=True)
    verdict = gh335.validate_task_root(evil_link)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.TASK_ROOT_IDENTITY.value


@pytest.mark.parametrize(
    "leaf_name", ["authority_snapshot.json", "capture_receipt.json", "READY.md"],
)
def test_validate_task_root_leaf_substituted_with_symlink_mid_open_fails_closed(tmp_path, monkeypatch, leaf_name):
    """Simulate an attacker racing to replace a task-root artifact with a
    symlink to a secret victim file at the exact moment validate_task_root
    is about to read it -- the strongest possible timing for a
    check-to-open race. Because each leaf is opened with O_NOFOLLOW as a
    single atomic step (no prior lstat-then-later-open gap), the open call
    itself must observe the symlink and fail closed: the victim's content
    must never be read into the process or leaked into the verdict."""
    root = tmp_path / "task_root"
    _self_contained_task_root(root, monkeypatch)
    victim = tmp_path / "victim_secret.txt"
    victim.write_bytes(b"SECRET-VICTIM-CONTENT")

    real_open = os.open
    state = {"swapped": False}

    def hijack_open(name, flags, mode=0o777, *, dir_fd=None):
        is_leaf_read = (flags & os.O_NOFOLLOW) and not (flags & os.O_DIRECTORY)
        if name == leaf_name and is_leaf_read and not state["swapped"]:
            state["swapped"] = True
            os.unlink(name, dir_fd=dir_fd)
            os.symlink(str(victim), name, dir_fd=dir_fd)
        return real_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", hijack_open)

    verdict = gh335.validate_task_root(root)

    assert state["swapped"] is True
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.TASK_ROOT_IDENTITY.value
    assert "SECRET-VICTIM-CONTENT" not in verdict.detail
    assert victim.read_bytes() == b"SECRET-VICTIM-CONTENT"


def test_validate_task_root_multi_leaf_mutant(tmp_path, monkeypatch):
    """Corrupt two independent fields (a request field and a receipt field)
    against a self-contained task root; the request-level defect must be
    caught first because verify_snapshot runs before verify_receipt."""
    root = tmp_path / "multi_leaf"
    _self_contained_task_root(root, monkeypatch)
    snapshot_path = root / "authority_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["ordered_requests"][0]["status"] = 500
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = root / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["runner"] = "forged-runner"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.REQUEST_STATUS.value


def _mutate_task_root_snapshot_record(root, index, mutate):
    snapshot_path = root / "authority_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    _replace_record_json(snapshot, index, mutate)
    snapshot_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def test_validate_task_root_malformed_issue_body_fails_closed(tmp_path, monkeypatch):
    """A truthy non-string issue body against a self-contained task root
    must fail closed via a Verdict, not propagate an uncaught
    AttributeError out of validate_task_root."""
    root = tmp_path / "malformed_issue_body"
    _self_contained_task_root(root, monkeypatch)
    _mutate_task_root_snapshot_record(root, 0, lambda obj: obj.__setitem__("body", 12345))

    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.ISSUE_IDENTITY.value


def test_validate_task_root_malformed_special_comment_body_fails_closed(tmp_path, monkeypatch):
    """Same crash risk as the issue body, for the #336 special comment
    (the last ordered_requests record) against a self-contained task root."""
    root = tmp_path / "malformed_special_comment_body"
    _self_contained_task_root(root, monkeypatch)
    snapshot = json.loads((root / "authority_snapshot.json").read_text(encoding="utf-8"))
    last_index = len(snapshot["ordered_requests"]) - 1
    _mutate_task_root_snapshot_record(root, last_index, lambda obj: obj.__setitem__("body", {"nested": "object"}))

    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.COMMENT_IDENTITY.value


def test_validate_task_root_forged_local_script_self_authorization_rejected(tmp_path, monkeypatch):
    """A forged capture_public_authority.py placed in the task root, with the
    receipt's script_sha256 rewritten to match that forged script's own
    hash, must never self-authorize: script provenance is validated only
    against the fixed ACCEPTED_SCRIPT_SHAS allowlist, never by hashing bytes
    found inside the very task root being validated."""
    root = tmp_path / "forged_script"
    _self_contained_task_root(root, monkeypatch)
    forged_script = b"# attacker-controlled capture script\n"
    (root / "capture_public_authority.py").write_bytes(forged_script)
    forged_hash = gh335.sha256_hex(forged_script)
    assert forged_hash not in gh335.ACCEPTED_SCRIPT_SHAS

    receipt_path = root / "capture_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["script_sha256"] = forged_hash
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.RECEIPT_SCRIPT_BINDING.value


# ---------------------------------------------------------------------------
# _read_task_root_artifact / validate_task_root: per-artifact and aggregate
# byte ceilings, enforced while reading from the pinned O_NOFOLLOW dirfd/fd.
# Exact-bound and over-bound coverage for all three task-root leaves.
# ---------------------------------------------------------------------------

_TASK_ROOT_LEAF_NAMES = ["authority_snapshot.json", "capture_receipt.json", "READY.md"]


@pytest.mark.parametrize("leaf_name", _TASK_ROOT_LEAF_NAMES)
def test_read_task_root_artifact_exact_bound_accepted(tmp_path, leaf_name):
    root = tmp_path / "exact_bound"
    root.mkdir()
    payload = b"x" * 4096
    (root / leaf_name).write_bytes(payload)
    dirfd = gh335._open_dir_chain_nofollow(root, create=False)
    try:
        data = gh335._read_task_root_artifact(dirfd, leaf_name, len(payload))
        assert data == payload
    finally:
        gh335._close_ambiguous(dirfd)


@pytest.mark.parametrize("leaf_name", _TASK_ROOT_LEAF_NAMES)
def test_read_task_root_artifact_over_bound_rejected(tmp_path, leaf_name):
    root = tmp_path / "over_bound"
    root.mkdir()
    payload = b"x" * 4097
    (root / leaf_name).write_bytes(payload)
    dirfd = gh335._open_dir_chain_nofollow(root, create=False)
    try:
        with pytest.raises(gh335.ArtifactTooLargeError):
            gh335._read_task_root_artifact(dirfd, leaf_name, 4096)
    finally:
        gh335._close_ambiguous(dirfd)


def test_validate_task_root_artifact_over_per_leaf_limit_rejected(tmp_path, monkeypatch):
    root = tmp_path / "oversized_snapshot"
    _self_contained_task_root(root, monkeypatch)
    snapshot_size = (root / "authority_snapshot.json").stat().st_size
    monkeypatch.setattr(gh335, "TASK_ROOT_ARTIFACT_LIMIT", snapshot_size - 1)

    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.TASK_ROOT_ARTIFACT_TOO_LARGE.value


def test_validate_task_root_artifact_at_exact_per_leaf_limit_not_rejected_for_size(tmp_path, monkeypatch):
    """At exactly the per-leaf limit, the read itself must succeed (no
    ArtifactTooLargeError) -- the boundary is inclusive."""
    root = tmp_path / "exact_snapshot"
    _self_contained_task_root(root, monkeypatch)
    snapshot_size = (root / "authority_snapshot.json").stat().st_size
    receipt_size = (root / "capture_receipt.json").stat().st_size
    ready_size = (root / "READY.md").stat().st_size
    monkeypatch.setattr(gh335, "TASK_ROOT_ARTIFACT_LIMIT", max(snapshot_size, receipt_size, ready_size))
    monkeypatch.setattr(gh335, "TASK_ROOT_AGGREGATE_LIMIT", snapshot_size + receipt_size + ready_size)

    verdict = gh335.validate_task_root(root)
    assert verdict.code != gh335.Reject.TASK_ROOT_ARTIFACT_TOO_LARGE.value
    assert verdict.accepted is True


def test_validate_task_root_aggregate_limit_rejected_on_second_leaf(tmp_path, monkeypatch):
    """Each leaf individually fits under the per-leaf cap, but the aggregate
    cap is exhausted by the first leaf -- the second leaf's read must be
    rejected before its bytes are ever fully buffered, proving the aggregate
    ceiling is enforced cumulatively across leaves, not just per file."""
    root = tmp_path / "aggregate_over"
    _self_contained_task_root(root, monkeypatch)
    snapshot_size = (root / "authority_snapshot.json").stat().st_size
    receipt_size = (root / "capture_receipt.json").stat().st_size
    ready_size = (root / "READY.md").stat().st_size
    monkeypatch.setattr(gh335, "TASK_ROOT_ARTIFACT_LIMIT", max(snapshot_size, receipt_size, ready_size))
    monkeypatch.setattr(gh335, "TASK_ROOT_AGGREGATE_LIMIT", snapshot_size)

    verdict = gh335.validate_task_root(root)
    assert verdict.accepted is False
    assert verdict.code == gh335.Reject.TASK_ROOT_ARTIFACT_TOO_LARGE.value


# ---------------------------------------------------------------------------
# ACCEPTED_SCRIPT_SHAS: fail-closed invariant that every allowlisted script
# digest is exactly 64 lowercase hexadecimal characters -- a provenance
# allowlist must reject a malformed pasted value outright, never silently
# normalize (e.g. truncate) it into acceptance.
# ---------------------------------------------------------------------------

_VALID_SCRIPT_SHA = "a" * 64


def test_accepted_script_shas_all_exactly_64_lowercase_hex():
    for value in gh335.ACCEPTED_SCRIPT_SHAS:
        assert isinstance(value, str)
        assert len(value) == 64
        assert value == value.lower()
        assert all(c in "0123456789abcdef" for c in value)


def test_validated_script_shas_accepts_well_formed_set():
    shas = {_VALID_SCRIPT_SHA, "b" * 64}
    assert gh335._validated_script_shas(shas) is shas


def test_validated_script_shas_rejects_over_length_value():
    with pytest.raises(ValueError):
        gh335._validated_script_shas({_VALID_SCRIPT_SHA + "0"})


def test_validated_script_shas_rejects_under_length_value():
    with pytest.raises(ValueError):
        gh335._validated_script_shas({_VALID_SCRIPT_SHA[:-1]})


def test_validated_script_shas_rejects_uppercase_value():
    with pytest.raises(ValueError):
        gh335._validated_script_shas({_VALID_SCRIPT_SHA.upper()})


def test_validated_script_shas_rejects_non_hex_value():
    with pytest.raises(ValueError):
        gh335._validated_script_shas({"g" * 64})


def test_validated_script_shas_rejects_non_string_value():
    with pytest.raises(ValueError):
        gh335._validated_script_shas({123})


# ---------------------------------------------------------------------------
# CANONICAL_ACCEPTED_SCRIPT_SHA: the self-check fixture (and this suite's own
# _pristine_receipt_bundle) must pick one fixed member of ACCEPTED_SCRIPT_SHAS
# deterministically. `next(iter(a_set))` is unsafe for this: CPython's set
# iteration order for strings depends on hash(str), which is randomized per
# process by PYTHONHASHSEED unless disabled -- so a prior version of this
# fixture could select a *different* accepted digest on every fresh
# interpreter invocation. `min()` orders by string value, never by hash, so
# it is identical across processes and hash seeds by construction. This is
# proven directly below by re-deriving the value in fresh subprocesses under
# several distinct PYTHONHASHSEED settings and requiring one identical
# answer -- the exact reproduction of the manager's PYTHONHASHSEED=1,2,3
# divergence check that flagged the prior next(iter(...)) fixture.
# ---------------------------------------------------------------------------

def test_canonical_accepted_script_sha_is_min_and_a_member():
    assert gh335.CANONICAL_ACCEPTED_SCRIPT_SHA == min(gh335.ACCEPTED_SCRIPT_SHAS)
    assert gh335.CANONICAL_ACCEPTED_SCRIPT_SHA in gh335.ACCEPTED_SCRIPT_SHAS


def test_canonical_accepted_script_sha_stable_across_hash_seeds():
    probe = (
        "import importlib.util\n"
        f"spec = importlib.util.spec_from_file_location('gh335_probe', {str(_MODULE_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "print(m.CANONICAL_ACCEPTED_SCRIPT_SHA)\n"
    )
    observed = set()
    for seed in ("0", "1", "2", "3"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        result = subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True,
        )
        observed.add(result.stdout.strip())
    assert observed == {min(gh335.ACCEPTED_SCRIPT_SHAS)}, (
        f"CANONICAL_ACCEPTED_SCRIPT_SHA must not vary with PYTHONHASHSEED, got {observed}"
    )


# ---------------------------------------------------------------------------
# Completeness: every reachable Reject code has a one-leaf mutant above.
# ---------------------------------------------------------------------------

_STANDALONE_ONE_LEAF_CODES = {
    gh335.Reject.SNAPSHOT_TYPE,
    gh335.Reject.REQUEST_TYPE,
    gh335.Reject.RECEIPT_TYPE,
    gh335.Reject.READY_CONTENT_MISMATCH,
    gh335.Reject.READY_MISSING,
    gh335.Reject.TASK_ROOT_IDENTITY,
    gh335.Reject.AUTHORITY_SNAPSHOT_EPOCH,
    gh335.Reject.AUTHORITY_RAW_EPOCH,
    gh335.Reject.TASK_ROOT_ARTIFACT_TOO_LARGE,
}


def test_every_reachable_reject_code_has_a_one_leaf_mutant():
    covered = {m[1] for m in SNAPSHOT_MUTATIONS} | {m[1] for m in RECEIPT_MUTATIONS} | _STANDALONE_ONE_LEAF_CODES
    all_codes = set(gh335.Reject)
    missing = all_codes - covered
    assert not missing, f"Reject codes without an isolated one-leaf mutant: {sorted(c.value for c in missing)}"


# ---------------------------------------------------------------------------
# Self-check: two independent full computations in disjoint repo-local out/
# directories must be identical.
# ---------------------------------------------------------------------------

def test_self_check_two_disjoint_runs_are_identical():
    base = _REPO_ROOT / "out" / "gh335_validator_test_selfcheck"
    if base.exists():
        shutil.rmtree(base)
    try:
        result_a = gh335.self_check(base / "run_a")
        result_b = gh335.self_check(base / "run_b")
        assert gh335._identity_relevant(result_a) == gh335._identity_relevant(result_b)
        assert result_a["verdict"] == "ACCEPT"
        assert (base / "run_a" / "selfcheck_result.json").is_file()
        assert (base / "run_b" / "selfcheck_result.json").is_file()
    finally:
        if base.exists():
            shutil.rmtree(base)


def test_self_check_preexisting_destination_fails_closed_and_preserves_victim(tmp_path):
    """self_check must never unlink a pre-existing selfcheck_result.json to
    make room for its own publish: publish_no_replace's own no-replace
    contract must be allowed to fail closed, and the victim's content, mode,
    and link-count must survive untouched."""
    out_root = tmp_path / "victim_run"
    out_root.mkdir()
    victim_path = out_root / "selfcheck_result.json"
    victim_path.write_bytes(b"victim-content")
    victim_path.chmod(0o640)
    victim_hardlink = out_root / "selfcheck_result.json.hardlink"
    os.link(victim_path, victim_hardlink)
    before = victim_path.stat()

    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.self_check(out_root)

    assert excinfo.value.code == "NO_REPLACE_VIOLATION"
    after = victim_path.stat()
    assert victim_path.read_bytes() == b"victim-content"
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode) == 0o640
    assert after.st_nlink == before.st_nlink == 2
    assert victim_hardlink.read_bytes() == b"victim-content"


# ---------------------------------------------------------------------------
# Atomic publish: privilege-independent, no-symlink-ancestry, identity-bound,
# no-replace, substitution-immune, EINTR/short-write/zero-write/fsync-
# failure/close-ambiguity/unsupported-primitive/victim-safety coverage.
# ---------------------------------------------------------------------------

def test_publish_success_unprivileged(tmp_path):
    dest = tmp_path / "out.txt"
    result = gh335.publish_no_replace(dest, b"hello world")
    assert dest.read_bytes() == b"hello world"
    assert not dest.is_symlink()
    assert result == {"leftover_tmp_name": None}


def test_publish_primary_path_uses_o_tmpfile_and_leaves_no_named_temp_entry(tmp_path):
    """On a filesystem that supports it (the common case), the temp file is
    unnamed (O_TMPFILE): there is never a directory entry for it to
    substitute or to (mis-)clean up. This is the structural fix for the
    cleanup victim race — proven directly by showing the directory contains
    nothing but the published destination."""
    dest = tmp_path / "out.txt"
    result = gh335.publish_no_replace(dest, b"hello world")
    assert result["leftover_tmp_name"] is None
    assert sorted(os.listdir(tmp_path)) == ["out.txt"]


def test_publish_no_replace_rejects_existing_destination_and_preserves_victim(tmp_path):
    dest = tmp_path / "out.txt"
    dest.write_bytes(b"victim-content")
    dest.chmod(0o640)
    dest_hardlink = tmp_path / "out.txt.hardlink"
    os.link(dest, dest_hardlink)
    before = dest.stat()

    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"attacker-content")
    assert excinfo.value.code == "NO_REPLACE_VIOLATION"

    after = dest.stat()
    assert dest.read_bytes() == b"victim-content"
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode) == 0o640
    assert after.st_nlink == before.st_nlink == 2
    assert dest_hardlink.read_bytes() == b"victim-content"


def test_publish_rejects_symlinked_destination_and_preserves_victim(tmp_path):
    victim = tmp_path / "victim.txt"
    victim.write_bytes(b"victim-content")
    link = tmp_path / "link.txt"
    link.symlink_to(victim)
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(link, b"attacker-content")
    assert excinfo.value.code == "SYMLINK_ANCESTRY"
    assert victim.read_bytes() == b"victim-content"


def test_publish_creates_missing_nested_parent_directories(tmp_path):
    dest = tmp_path / "a" / "b" / "c" / "out.txt"
    result = gh335.publish_no_replace(dest, b"hello world")
    assert dest.read_bytes() == b"hello world"
    assert result == {"leftover_tmp_name": None}
    assert stat.S_ISDIR(os.stat(tmp_path / "a").st_mode)
    assert stat.S_ISDIR(os.stat(tmp_path / "a" / "b").st_mode)
    assert stat.S_ISDIR(os.stat(tmp_path / "a" / "b" / "c").st_mode)


def test_publish_creates_missing_parent_via_dirfd_never_pathname_mkdir_race_safe(tmp_path, monkeypatch):
    """The missing ancestor directory `raced` is created only through a
    dirfd-relative mkdir(2) immediately followed by an O_NOFOLLOW open on
    the same dirfd — never a pathname-based mkdir that could resolve through
    a symlink planted in the race window. Simulate an attacker winning
    exactly that window: right after our mkdir creates `raced`, before our
    follow-up open observes it, replace it with a symlink to a foreign
    decoy directory. The follow-up O_NOFOLLOW open must observe the symlink
    directly and fail closed; nothing may ever land under the decoy."""
    decoy = tmp_path / "decoy"
    decoy.mkdir()
    real_mkdir = os.mkdir
    raced = {"done": False}

    def hijack_mkdir(name, mode=0o777, *, dir_fd=None):
        real_mkdir(name, mode, dir_fd=dir_fd)
        if name == "raced" and not raced["done"]:
            raced["done"] = True
            os.rmdir(name, dir_fd=dir_fd)
            os.symlink(str(decoy), name, target_is_directory=True, dir_fd=dir_fd)

    monkeypatch.setattr(os, "mkdir", hijack_mkdir)
    dest = tmp_path / "raced" / "out.txt"

    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"hello world")

    assert excinfo.value.code == "SYMLINK_ANCESTRY"
    assert raced["done"] is True
    assert not (decoy / "out.txt").exists()
    assert not dest.exists()


def test_publish_rejects_symlinked_ancestor_directory_no_write_lands(tmp_path):
    real_target = tmp_path / "real_target"
    real_target.mkdir()
    evil_link = tmp_path / "evil_link"
    evil_link.symlink_to(real_target, target_is_directory=True)
    dest = evil_link / "out.txt"
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"attacker-content")
    assert excinfo.value.code == "SYMLINK_ANCESTRY"
    assert not (real_target / "out.txt").exists()


def _patch_tmp_fd_capture(monkeypatch):
    """Capture the real fd handed out by gh335._open_tmpfile (the O_TMPFILE
    fd used by the primary path of publish_no_replace on this platform) so
    write/fsync/close faults can be targeted at exactly that fd."""
    real_open_tmpfile = gh335._open_tmpfile
    state = {"fd": None}

    def fake_open_tmpfile(dirfd):
        fd = real_open_tmpfile(dirfd)
        state["fd"] = fd
        return fd

    monkeypatch.setattr(gh335, "_open_tmpfile", fake_open_tmpfile)
    return state


def _force_named_temp_fallback(monkeypatch):
    """Force publish_no_replace onto its named-temp-file fallback path, as
    if O_TMPFILE were unsupported on this filesystem, and capture the
    dir_fd/tmp_name of the fallback's temp file for substitution tests."""
    monkeypatch.setattr(gh335, "_open_tmpfile", lambda dirfd: None)
    real_mkstemp_in_dir = gh335._mkstemp_in_dir
    state = {"dirfd": None, "tmp_name": None}

    def fake_mkstemp_in_dir(dirfd, name, max_attempts=64):
        fd, tmp_name = real_mkstemp_in_dir(dirfd, name, max_attempts)
        state["dirfd"] = dirfd
        state["tmp_name"] = tmp_name
        return fd, tmp_name

    monkeypatch.setattr(gh335, "_mkstemp_in_dir", fake_mkstemp_in_dir)
    return state


def test_open_tmpfile_returns_none_for_unsupported_errno(tmp_path, monkeypatch):
    dirfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        def fake_open(path, flags, mode=0o777, dir_fd=None):
            raise OSError(errno.EOPNOTSUPP, "simulated O_TMPFILE unsupported")

        monkeypatch.setattr(os, "open", fake_open)
        assert gh335._open_tmpfile(dirfd) is None
    finally:
        os.close(dirfd)


def test_open_tmpfile_propagates_unexpected_errno(tmp_path, monkeypatch):
    dirfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        def fake_open(*args, **kwargs):
            raise OSError(errno.EIO, "simulated unexpected I/O error")

        monkeypatch.setattr(os, "open", fake_open)
        with pytest.raises(OSError) as excinfo:
            gh335._open_tmpfile(dirfd)
        assert excinfo.value.errno == errno.EIO
    finally:
        os.close(dirfd)


def test_open_tmpfile_returns_none_when_platform_lacks_flag(tmp_path, monkeypatch):
    dirfd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        monkeypatch.delattr(os, "O_TMPFILE", raising=False)
        assert gh335._open_tmpfile(dirfd) is None
    finally:
        os.close(dirfd)


def test_publish_named_temp_fallback_succeeds_and_reports_leftover_as_hardlink(tmp_path, monkeypatch):
    """When O_TMPFILE is unsupported, publish still succeeds via the named-
    temp fallback, and honestly reports the leftover: it is a second
    hardlink to the exact same published inode (harmless, not foreign
    content), never silently deleted."""
    _force_named_temp_fallback(monkeypatch)
    dest = tmp_path / "out.txt"
    result = gh335.publish_no_replace(dest, b"hello world")

    assert dest.read_bytes() == b"hello world"
    assert result["leftover_tmp_name"] is not None
    leftover = tmp_path / result["leftover_tmp_name"]
    assert leftover.is_file()
    assert leftover.read_bytes() == b"hello world"
    assert leftover.stat().st_ino == dest.stat().st_ino
    assert leftover.stat().st_nlink == 2


def test_publish_eintr_on_write_is_retried(tmp_path, monkeypatch):
    state = _patch_tmp_fd_capture(monkeypatch)
    real_write = os.write
    calls = {"n": 0}

    def fake_write(fd, data):
        if fd == state["fd"] and calls["n"] == 0:
            calls["n"] += 1
            raise InterruptedError()
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", fake_write)
    dest = tmp_path / "out.txt"
    gh335.publish_no_replace(dest, b"hello world")
    assert dest.read_bytes() == b"hello world"
    assert calls["n"] == 1


def test_publish_short_writes_are_completed(tmp_path, monkeypatch):
    state = _patch_tmp_fd_capture(monkeypatch)
    real_write = os.write

    def fake_write(fd, data):
        if fd == state["fd"]:
            return real_write(fd, data[:1]) if len(data) > 1 else real_write(fd, data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", fake_write)
    dest = tmp_path / "out.txt"
    payload = b"hello world, this is a longer payload to force many short writes"
    gh335.publish_no_replace(dest, payload)
    assert dest.read_bytes() == payload


def test_publish_zero_length_writes_are_bounded_and_rejected(tmp_path, monkeypatch):
    state = _patch_tmp_fd_capture(monkeypatch)

    def fake_write(fd, data):
        if fd == state["fd"]:
            return 0
        return os.write(fd, data)

    monkeypatch.setattr(os, "write", fake_write)
    dest = tmp_path / "out.txt"
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"hello world")
    assert excinfo.value.code == "ZERO_WRITE"
    assert not dest.exists()


def test_publish_fsync_failure_cleans_up_and_leaves_no_destination(tmp_path, monkeypatch):
    state = _patch_tmp_fd_capture(monkeypatch)
    real_fsync = os.fsync

    def fake_fsync(fd):
        if fd == state["fd"]:
            raise OSError(errno.EIO, "simulated fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fake_fsync)
    dest = tmp_path / "out.txt"
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"hello world")
    assert excinfo.value.code == "FSYNC_FAILED"
    assert not dest.exists()
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".out.txt.")]
    assert leftovers == []


def test_publish_close_ambiguity_still_publishes_successfully(tmp_path, monkeypatch):
    state = _patch_tmp_fd_capture(monkeypatch)
    real_close = os.close

    def fake_close(fd):
        if fd == state["fd"]:
            raise OSError(errno.EIO, "simulated close ambiguity")
        return real_close(fd)

    monkeypatch.setattr(os, "close", fake_close)
    dest = tmp_path / "out.txt"
    gh335.publish_no_replace(dest, b"hello world")
    assert dest.read_bytes() == b"hello world"


def test_publish_unsupported_link_primitive_fails_closed(tmp_path, monkeypatch):
    """When hardlink-by-fd is unsupported (e.g. EXDEV) for both the O_TMPFILE
    attempt and the named-temp fallback, there is no privilege-independent,
    source-identity-bound alternative primitive: publish_no_replace must
    fail closed rather than renaming an unauthenticated name into the
    destination. Since no atomic identity-conditional unlink exists, the
    named-temp fallback's entry is truthfully left in place (never
    name-unlinked) and reported via leftover_tmp_name, rather than silently
    discarded or silently claimed absent."""
    def fake_link(src, dst, *args, **kwargs):
        raise OSError(errno.EXDEV, "simulated cross-device link")

    monkeypatch.setattr(os, "link", fake_link)
    dest = tmp_path / "out.txt"
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"hello world")
    assert excinfo.value.code == "ATOMIC_PRIMITIVE_UNAVAILABLE"
    assert not dest.exists()
    assert excinfo.value.leftover_tmp_name is not None
    leftover = tmp_path / excinfo.value.leftover_tmp_name
    assert leftover.is_file()
    assert leftover.read_bytes() == b"hello world"


def test_publish_unsupported_link_primitive_fails_closed_preserves_existing_destination(tmp_path, monkeypatch):
    """The same fail-closed behavior applies even when a legitimate
    destination already exists: the failure must never fall through to a
    name-based rename that could replace or expose it."""
    def fake_link(src, dst, *args, **kwargs):
        raise OSError(errno.EXDEV, "simulated cross-device link")

    monkeypatch.setattr(os, "link", fake_link)
    dest = tmp_path / "out.txt"
    dest.write_bytes(b"victim-content")
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"attacker-content")
    assert excinfo.value.code == "ATOMIC_PRIMITIVE_UNAVAILABLE"
    assert dest.read_bytes() == b"victim-content"
    assert excinfo.value.leftover_tmp_name is not None


def test_fallback_exclusive_publish_always_fails_closed():
    """Direct unit coverage: _fallback_exclusive_publish never attempts a
    name-based rename — it always fails closed, since a check-then-rename
    on a bare (substitutable) name cannot be made source-identity-bound."""
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335._fallback_exclusive_publish("anything")
    assert excinfo.value.code == "ATOMIC_PRIMITIVE_UNAVAILABLE"


def test_publish_link_failure_other_errno_propagates(tmp_path, monkeypatch):
    """An unrelated link failure (not an unsupported-primitive errno) must
    propagate immediately without attempting the named-temp fallback — a
    retry would not help and would only create an unnecessary leftover."""
    def fake_link(src, dst, *args, **kwargs):
        raise OSError(errno.EIO, "simulated unrelated link failure")

    monkeypatch.setattr(os, "link", fake_link)
    dest = tmp_path / "out.txt"
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"hello world")
    assert excinfo.value.code == "LINK_FAILED"
    assert not dest.exists()
    assert excinfo.value.leftover_tmp_name is None


def test_publish_does_not_require_privileged_primitives(tmp_path):
    """No CAP_DAC_READ_SEARCH / root is available in this test process; a
    successful publish here demonstrates privilege-independence directly."""
    assert os.geteuid() != 0
    dest = tmp_path / "unprivileged.txt"
    gh335.publish_no_replace(dest, b"ok")
    assert dest.read_bytes() == b"ok"


# ---------------------------------------------------------------------------
# Substitution-window and victim-safety coverage: the parent directory, the
# temp file's own name, and the destination name can each be substituted by
# an attacker mid-flight. publish_no_replace must never expose attacker
# content or write into a decoy for any of the three: it succeeds only when
# it can prove source/destination identity via a pinned dir_fd and the temp
# file's own fd (never a re-resolved pathname), and fails closed otherwise —
# a name-based fallback is never used to paper over an identity it cannot
# prove.
# ---------------------------------------------------------------------------

def test_publish_survives_parent_directory_pathname_substitution(tmp_path, monkeypatch):
    """After the parent directory fd is pinned, an attacker destroys the
    original directory entirely and replaces its *pathname* with a symlink
    to an unrelated decoy directory. The pinned dir_fd cannot be redirected
    to the decoy — it stays bound to the original (now-removed) directory —
    and Linux refuses to create new entries via an fd whose directory has
    been rmdir'd, so publish must fail closed. The correct contract here is
    fail-closed, proving no write ever reaches the decoy (or any other
    foreign location), not a silent 'success' that actually wrote into an
    orphaned, unreachable directory. The exact errno for "operate on a
    removed directory via a pinned fd" is kernel/primitive-dependent (ENOENT
    for a named create, but some kernels report EPERM specifically for
    O_TMPFILE against a removed directory) — either is an acceptable
    fail-closed outcome; what matters is that it is not silently absorbed."""
    real_dir = tmp_path / "real_dir"
    real_dir.mkdir()
    decoy_dir = tmp_path / "decoy_dir"
    decoy_dir.mkdir()
    dest = real_dir / "out.txt"

    real_open_chain = gh335._open_dir_chain_nofollow

    def hijack(path, create=False):
        assert create is True, "publish_no_replace must request parent-dir auto-create"
        dirfd = real_open_chain(path, create=create)
        shutil.rmtree(real_dir)
        real_dir.symlink_to(decoy_dir, target_is_directory=True)
        return dirfd

    monkeypatch.setattr(gh335, "_open_dir_chain_nofollow", hijack)

    with pytest.raises((FileNotFoundError, PermissionError)):
        gh335.publish_no_replace(dest, b"hello world")

    assert not (decoy_dir / "out.txt").exists()
    assert real_dir.is_symlink()


def _assert_foreign_entry_preserved(path, before_stat, expected_content, expected_mode):
    after_stat = path.stat()
    assert path.read_bytes() == expected_content
    assert stat.S_IMODE(after_stat.st_mode) == stat.S_IMODE(before_stat.st_mode) == expected_mode
    assert after_stat.st_nlink == before_stat.st_nlink == 2
    assert after_stat.st_ino == before_stat.st_ino


def test_publish_temp_name_substitution_before_link_fails_closed_and_preserves_foreign_entry(tmp_path, monkeypatch):
    """An attacker unlinks the named-temp-fallback's temp file *name* and
    recreates it with attacker-controlled content (plus a second hardlink,
    proving link-count matters) immediately before the publish primitive
    links it in. Our own link source is always ``/proc/self/fd/{tmp_fd}``
    (the exact fd this process wrote and fsynced), never ``tmp_name`` — but
    on this platform, relinking a since-renamed-away fd via /proc/self/fd is
    itself unsupported (observed: ENOENT), which this module already
    classifies as an unsupported-primitive condition and fails closed for.
    Critically, failing closed must never fall back to deleting the
    attacker's substituted entry to "clean up" — no atomic
    identity-conditional unlink exists to do that safely, so the entry must
    survive untouched (name, content, mode, link-count) regardless of our
    own outcome."""
    state = _force_named_temp_fallback(monkeypatch)
    dest = tmp_path / "out.txt"
    foreign = {}

    real_write_all = gh335._write_all

    def write_then_substitute(fd, data):
        real_write_all(fd, data)
        tmp_name, dirfd = state["tmp_name"], state["dirfd"]
        os.unlink(tmp_name, dir_fd=dirfd)
        foreign_fd = os.open(tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640, dir_fd=dirfd)
        try:
            os.write(foreign_fd, b"attacker-content")
        finally:
            os.close(foreign_fd)
        os.link(tmp_path / tmp_name, tmp_path / (tmp_name + ".hardlink"))
        foreign["stat"] = (tmp_path / tmp_name).stat()

    monkeypatch.setattr(gh335, "_write_all", write_then_substitute)

    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"hello world")

    assert excinfo.value.code == "ATOMIC_PRIMITIVE_UNAVAILABLE"
    assert not dest.exists()
    assert excinfo.value.leftover_tmp_name == state["tmp_name"]
    _assert_foreign_entry_preserved(
        tmp_path / state["tmp_name"], foreign["stat"], b"attacker-content", 0o640
    )


def test_publish_success_then_late_temp_name_substitution_is_never_deleted(tmp_path, monkeypatch):
    """Success window: an attacker substitutes the named-temp fallback's
    temp *name* (with a second hardlink, proving link-count matters) exactly
    after our own publish resolves successfully. Since our destination link
    used the fd (identity-bound), the correct destination content is
    published regardless. Because no atomic identity-conditional unlink
    exists, the module must never attempt a name-based cleanup afterward —
    the attacker's substituted entry must survive untouched."""
    state = _force_named_temp_fallback(monkeypatch)
    dest = tmp_path / "out.txt"
    foreign = {}
    real_publish_from_fd = gh335._publish_from_fd

    def substitute_after(dirfd, fd, name):
        result = real_publish_from_fd(dirfd, fd, name)
        tmp_name = state["tmp_name"]
        os.unlink(tmp_name, dir_fd=dirfd)
        foreign_fd = os.open(tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640, dir_fd=dirfd)
        try:
            os.write(foreign_fd, b"foreign-content-success-window")
        finally:
            os.close(foreign_fd)
        os.link(tmp_path / tmp_name, tmp_path / (tmp_name + ".hardlink"))
        foreign["stat"] = (tmp_path / tmp_name).stat()
        return result

    monkeypatch.setattr(gh335, "_publish_from_fd", substitute_after)

    result = gh335.publish_no_replace(dest, b"hello world")

    assert dest.read_bytes() == b"hello world"
    assert result["leftover_tmp_name"] == state["tmp_name"]
    _assert_foreign_entry_preserved(
        tmp_path / state["tmp_name"], foreign["stat"], b"foreign-content-success-window", 0o640
    )


def test_publish_failure_then_late_temp_name_substitution_is_never_deleted(tmp_path, monkeypatch):
    """Failure window: an attacker substitutes the named-temp fallback's
    temp *name* exactly after our own publish attempt fails (destination
    already exists). The failure path must never attempt a name-based
    cleanup either — the attacker's substituted entry must survive
    untouched, and the pre-existing destination victim must survive too."""
    state = _force_named_temp_fallback(monkeypatch)
    dest = tmp_path / "out.txt"
    dest.write_bytes(b"victim-content")
    foreign = {}
    real_publish_from_fd = gh335._publish_from_fd

    def substitute_after(dirfd, fd, name):
        try:
            return real_publish_from_fd(dirfd, fd, name)
        finally:
            tmp_name = state["tmp_name"]
            os.unlink(tmp_name, dir_fd=dirfd)
            foreign_fd = os.open(tmp_name, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640, dir_fd=dirfd)
            try:
                os.write(foreign_fd, b"foreign-content-failure-window")
            finally:
                os.close(foreign_fd)
            os.link(tmp_path / tmp_name, tmp_path / (tmp_name + ".hardlink"))
            foreign["stat"] = (tmp_path / tmp_name).stat()

    monkeypatch.setattr(gh335, "_publish_from_fd", substitute_after)

    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"attacker-content")

    assert excinfo.value.code == "NO_REPLACE_VIOLATION"
    assert dest.read_bytes() == b"victim-content"
    assert excinfo.value.leftover_tmp_name == state["tmp_name"]
    _assert_foreign_entry_preserved(
        tmp_path / state["tmp_name"], foreign["stat"], b"foreign-content-failure-window", 0o640
    )


def test_publish_destination_substituted_after_check_still_rejected(tmp_path, monkeypatch):
    """An attacker creates the destination *after* the pre-check but exactly
    at the moment of publish. The final link must still observe the
    substitution atomically and refuse to replace it, preserving the
    attacker's own file (proving no-replace, not proving trust in the
    attacker — the point is our write never lands), including its mode and
    link-count (a second hardlink proves link-count matters). The victim
    must be created through the real ``dst_dir_fd`` used by the publish call
    (the destination name is dir_fd-relative, not resolvable via plain
    ``Path(dst)`` against the process cwd), or this never exercises the
    actual destination the code observes."""
    dest = tmp_path / "out.txt"
    real_link = os.link
    victim = {}

    def hijack_link(src, dst, *args, **kwargs):
        dst_dir_fd = kwargs.get("dst_dir_fd")
        victim_fd = os.open(dst, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o640, dir_fd=dst_dir_fd)
        try:
            os.write(victim_fd, b"victim-content-created-midflight")
        finally:
            os.close(victim_fd)
        real_link(dest, tmp_path / "out.txt.hardlink")
        victim["stat"] = dest.stat()
        return real_link(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "link", hijack_link)
    with pytest.raises(gh335.PublishError) as excinfo:
        gh335.publish_no_replace(dest, b"attacker-content")
    assert excinfo.value.code == "NO_REPLACE_VIOLATION"
    _assert_foreign_entry_preserved(dest, victim["stat"], b"victim-content-created-midflight", 0o640)


def test_main_task_root_accepts_self_contained_artifacts_exit_zero(tmp_path, monkeypatch, capsys):
    root = tmp_path / "cli_accepted"
    _self_contained_task_root(root, monkeypatch)
    exit_code = gh335.main(["--task-root", str(root)])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["accepted"] is True
    assert out["code"] == "ACCEPT"


def test_main_task_root_rejects_missing_artifacts_exit_one(tmp_path, capsys):
    root = tmp_path / "empty_root"
    root.mkdir()
    exit_code = gh335.main(["--task-root", str(root)])
    assert exit_code == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert out["accepted"] is False
    assert out["code"] == gh335.Reject.TASK_ROOT_IDENTITY.value


def test_main_self_check_succeeds_exit_zero(tmp_path, capsys):
    out_root = tmp_path / "cli_selfcheck"
    exit_code = gh335.main(["--self-check", "--out-root", str(out_root)])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["self_check_identical"] is True


def test_main_self_check_divergent_results_exit_one(tmp_path, monkeypatch, capsys):
    """If the two self-check runs were ever to diverge (e.g. a future
    regression leaking per-run entropy into the result), main() must
    surface exit code 1 and self_check_identical: false rather than
    silently reporting success."""
    real_self_check = gh335.self_check
    calls = {"n": 0}

    def fake_self_check(out_root):
        calls["n"] += 1
        result = real_self_check(out_root)
        if calls["n"] == 2:
            result = dict(result)
            result["verdict"] = "DIVERGED"
        return result

    monkeypatch.setattr(gh335, "self_check", fake_self_check)
    out_root = tmp_path / "cli_selfcheck_divergent"
    exit_code = gh335.main(["--self-check", "--out-root", str(out_root)])
    assert exit_code == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert out["self_check_identical"] is False


def test_main_missing_mode_raises_parser_error():
    with pytest.raises(SystemExit) as excinfo:
        gh335.main([])
    assert excinfo.value.code == 2


def test_publish_leaves_unrelated_sibling_content_mode_and_linkcount_untouched(tmp_path):
    """A foreign victim file (with a second hardlink, proving link-count
    matters) sitting next to the destination must be byte-for-byte, mode-for-
    mode, and link-count-for-link-count unchanged after an unrelated
    publish_no_replace call in the same directory."""
    victim = tmp_path / "sibling.txt"
    victim.write_bytes(b"sibling-content")
    victim.chmod(0o640)
    victim_hardlink = tmp_path / "sibling_hardlink.txt"
    os.link(victim, victim_hardlink)
    before = victim.stat()

    dest = tmp_path / "out.txt"
    gh335.publish_no_replace(dest, b"hello world")

    after = victim.stat()
    assert victim.read_bytes() == b"sibling-content"
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode)
    assert after.st_nlink == before.st_nlink == 2
    assert victim_hardlink.read_bytes() == b"sibling-content"


# ---------------------------------------------------------------------------
# Optional real-artifact integration (explicitly gated, never required): all
# security/correctness regression coverage above is fully self-contained and
# requires nothing outside this repository's tracked files. When the
# gitignored out/.tasks/<task-id> real accepted capture also happens to be
# present in the local workspace, these additional tests confirm that the
# validator's actual hardcoded epoch pins (RETAINED_SNAPSHOT_SHA/
# RETAINED_RAW_SHA) genuinely match that one real accepted capture -- a fact
# that is, by construction, only checkable when those real bytes exist. A
# clean checkout without out/.tasks/... skips this section outright; it is
# never treated as a failure and never substitutes for the required
# self-contained coverage above.
# ---------------------------------------------------------------------------

_real_artifacts_available = _ACCEPTED_TASK_ROOT.is_dir()
_skip_without_real_artifacts = pytest.mark.skipif(
    not _real_artifacts_available,
    reason=f"optional real-artifact integration fixture not present in this checkout: {_ACCEPTED_TASK_ROOT}",
)


@_skip_without_real_artifacts
def test_authority_epoch_accepts_real_accepted_artifacts_optional():
    snapshot_bytes = (_ACCEPTED_TASK_ROOT / "authority_snapshot.json").read_bytes()
    snapshot = json.loads(snapshot_bytes.decode("utf-8"))
    raw_hashes, _ = gh335.verify_snapshot(snapshot)
    gh335.verify_authority_epoch(snapshot_bytes, raw_hashes)


@_skip_without_real_artifacts
def test_validate_task_root_accepts_real_accepted_artifacts_optional():
    verdict = gh335.validate_task_root(_ACCEPTED_TASK_ROOT)
    assert verdict.accepted is True
    assert verdict.code == "ACCEPT"


@_skip_without_real_artifacts
def test_main_task_root_accepts_real_artifacts_exit_zero_optional(capsys):
    exit_code = gh335.main(["--task-root", str(_ACCEPTED_TASK_ROOT)])
    assert exit_code == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["accepted"] is True
    assert out["code"] == "ACCEPT"
