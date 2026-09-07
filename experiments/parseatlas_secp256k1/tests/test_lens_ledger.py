"""Ledger integrity tests; no timed benchmarks, candidate ranking or proof claims."""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
MODULE = ROOT / "experiments/parseatlas_secp256k1/lens_ledger.py"
SPEC = importlib.util.spec_from_file_location("parseatlas_lens_ledger", MODULE)
ledger = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger)


@pytest.fixture
def record():
    return ledger.parse_json((ROOT / ledger.DEFAULT_LEDGER).read_bytes())


def test_real_record_is_addressed_not_completed(record):
    result = ledger.validate(record, ROOT)
    assert result["valid"]
    assert result["addressed"] == 25
    assert result["assessment_counts"] == {"evidenced": 4, "partial": 16, "pending": 5}
    assert result["artifacts"] == 15
    assert result["eligibility"] == ledger.GATES
    assert [x["id"] for x in record["lenses"]] == [f"V{i:02d}" for i in range(1, 26)]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "reordered", "renamed", "method"])
def test_canonical_lens_identity_rejections(record, mutation):
    if mutation == "missing":
        record["lenses"].pop()
    elif mutation == "duplicate":
        record["lenses"][24] = deepcopy(record["lenses"][0])
    elif mutation == "unknown":
        record["lenses"][0]["id"] = "V26"
    elif mutation == "reordered":
        record["lenses"].reverse()
    elif mutation == "renamed":
        record["lenses"][0]["observable"] = "my favorite view"
    else:
        record["lenses"][0]["method"] = "trust prose"
    with pytest.raises(ledger.LedgerError):
        ledger.validate(record, ROOT)


@pytest.mark.parametrize("mutation", ["field_p", "ct", "proof", "rank", "covered", "extra", "verified"])
def test_diagnostic_cannot_forge_domain_or_eligibility(record, mutation):
    if mutation == "field_p":
        record["object"]["kind"] = "field_p"
    elif mutation == "ct":
        record["eligibility"]["constant_time_eligibility"] = True
    elif mutation == "proof":
        record["eligibility"]["full_domain_boundary_equivalence"] = "PROVEN"
    elif mutation == "rank":
        record["eligibility"]["ranking"] = "prefix wins"
    elif mutation == "covered":
        record["coverage"]["meaning"] = "completed"
    elif mutation == "extra":
        record["accepted"] = True
    else:
        record["lenses"][2]["claim_class"] = "VERIFIED"
    with pytest.raises(ledger.LedgerError):
        ledger.validate(record, ROOT)


@pytest.mark.parametrize("index", [2, 7, 8, 10, 19, 22, 23])
def test_pending_full_domain_and_ct_cannot_upgrade(record, index):
    record["lenses"][index]["assessment"] = "evidenced"
    with pytest.raises(ledger.LedgerError):
        ledger.validate(record, ROOT)


@pytest.mark.parametrize("replacement", ["N/A", "unknown", None, 0, False])
def test_pending_metric_cannot_become_fake_not_applicable(record, replacement):
    record["lenses"][8]["metrics"][0]["value"] = replacement
    with pytest.raises(ledger.LedgerError):
        ledger.validate(record, ROOT)


def test_assessment_is_not_claim_class(record):
    record["lenses"][0]["assessment"] = "OBSERVED"
    with pytest.raises(ledger.LedgerError):
        ledger.validate(record, ROOT)


@pytest.mark.parametrize("mutation", ["hash", "dangling", "duplicate", "missing", "source_id"])
def test_artifact_binding_rejections(record, mutation):
    if mutation == "hash":
        record["artifacts"][0]["sha256"] = "0" * 64
    elif mutation == "dangling":
        record["lenses"][0]["artifact_refs"].append("absent")
    elif mutation == "duplicate":
        record["artifacts"].append(deepcopy(record["artifacts"][0]))
    elif mutation == "missing":
        record["artifacts"].pop(0)
        record["lenses"][8]["artifact_refs"].append("methodology")
    else:
        record["lenses"][15]["metrics"][0]["artifact"] = "region_doc"
    with pytest.raises(ledger.LedgerError):
        ledger.validate(record, ROOT)


@pytest.mark.parametrize("relative", ["/etc/passwd", "../outside", "a/../../outside",
                                      "a//b", "./a", "a/./b", "a\\b"])
def test_paths_cannot_escape_or_be_ambiguous(tmp_path, relative):
    with pytest.raises(ledger.LedgerError):
        ledger.safe_path(tmp_path, relative)


def test_symlink_file_or_directory_rejected(tmp_path):
    actual = tmp_path / "actual"
    actual.mkdir()
    (actual / "evidence.json").write_text("{}")
    (tmp_path / "file_link.json").symlink_to(actual / "evidence.json")
    (tmp_path / "dir_link").symlink_to(actual, target_is_directory=True)
    for path in ("file_link.json", "dir_link/evidence.json"):
        with pytest.raises(ledger.LedgerError, match="symlink"):
            ledger.safe_path(tmp_path, path)


@pytest.mark.parametrize("mutation", ["value", "pointer", "kind", "scope", "missing", "prose_only"])
def test_measured_assertion_needs_actual_provenance(record, mutation):
    view = record["lenses"][15]
    item = view["metrics"][0]
    if mutation == "value":
        item["value"] += 1.0
    elif mutation == "pointer":
        item["pointer"] = "/measurement/workloads/900/summary/median"
    elif mutation == "kind":
        item["provenance_kind"] = "trust_me"
    elif mutation == "scope":
        view["evidence_scopes"] = ["python_model"]
    elif mutation == "missing":
        del item["provenance_kind"]
    else:
        view["metrics"] = [{"name": "latency", "unit": "ns", "value": ledger.PENDING,
                             "reason": "unmeasured"}]
    with pytest.raises(ledger.LedgerError):
        ledger.validate(record, ROOT)


def copy_evidence(record, root):
    for artifact in record["artifacts"]:
        target = root / artifact["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / artifact["path"], target)


@pytest.mark.parametrize("mutation", ["warmups", "version", "uncertainty", "pmu_scope", "pmu_binary"])
def test_rehashed_source_cannot_drop_measurement_context(record, tmp_path, mutation):
    copy_evidence(record, tmp_path)
    aid = "pmu" if mutation.startswith("pmu") else "cpu0"
    artifact = next(a for a in record["artifacts"] if a["id"] == aid)
    path = tmp_path / artifact["path"]
    data = json.loads(path.read_text())
    if mutation == "warmups":
        data["config"]["warmups"] = 0
    elif mutation == "version":
        del data["build"]["compiler_version"]
    elif mutation == "uncertainty":
        del data["measurement"]["workloads"][0]["summary"]["uncertainty_kind"]
    elif mutation == "pmu_scope":
        data["scope"] = "isolated arithmetic counters"
    else:
        data["binary_sha256"] = "0" * 64
    raw = json.dumps(data).encode()
    path.write_bytes(raw)
    artifact["sha256"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ledger.LedgerError):
        ledger.validate(record, tmp_path)


@pytest.mark.parametrize("raw", ['{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'])
def test_invalid_json_constants_and_duplicate_keys_rejected(raw):
    with pytest.raises(ledger.LedgerError):
        ledger.parse_json(raw)


@pytest.mark.parametrize("location", ["", "x", "/x/01", "/x/-1", "/x/9", "/z", "/bad~2"])
def test_pointer_rejections(location):
    with pytest.raises(ledger.LedgerError):
        ledger.pointer({"x": [5], "bad~2": 3}, location)


def test_pointer_escaped_keys_and_values():
    assert ledger.pointer({"a/b": {"~": [5]}}, "/a~1b/~0/0") == 5


def test_cli_readonly_json_result():
    result = subprocess.run([sys.executable, "-P", str(MODULE), "validate"],
                            cwd=ROOT, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["addressed"] == 25
    assert result.stderr == ""


def test_cli_rejects_outside_ledger():
    result = subprocess.run([sys.executable, "-P", str(MODULE), "validate", "--ledger", "../outside"],
                            cwd=ROOT, capture_output=True, text=True, timeout=15, check=False)
    assert result.returncode == 2
    assert json.loads(result.stdout)["valid"] is False
