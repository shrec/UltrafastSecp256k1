"""Deterministic runner gates, including negative provenance/result fixtures.

No test uses a timing speed threshold or starts a sustained benchmark.
"""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "baseline_runner.py"
SPEC = importlib.util.spec_from_file_location("parseatlas_baseline_runner", MODULE_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def raw_sample(workload, passes=3, seed=20260905, elapsed_ns=50_000_000):
    input_bytes, output_bytes = (2 * workload.limbs + 1) * 8, (workload.limbs + 1) * 8
    logical = input_bytes + output_bytes if workload.mode == "bulk" else 0
    return {"protocol": runner.PROTOCOL, "scope": "original_add64_composition",
            "mode": workload.mode, "limbs": workload.limbs, "count": workload.count,
            "passes": passes, "seed": seed, "operation_count": workload.count * passes,
            "add64_calls": workload.count * passes * workload.limbs, "elapsed_ns": elapsed_ns,
            "input_record_bytes": input_bytes, "output_record_bytes": output_bytes,
            "working_set_bytes": workload.count * logical if logical else input_bytes,
            "logical_bytes_per_operation": logical,
            "logical_bytes_processed": workload.count * passes * logical,
            "checksum": "0123456789abcdef",
            "warmup_operations": workload.count if logical else min(workload.count, 4096),
            "backend": "gcc_uint128", "no_int128_defined": False, "int128_macro_defined": True}


@pytest.mark.parametrize("change", [
    {"seed": -1}, {"seed": True}, {"seed": 1 << 64}, {"random_cases": 0},
    {"random_cases": 100001}, {"warmups": 1}, {"warmups": True}, {"warmups": 21},
    {"repetitions": 6}, {"repetitions": 102}, {"target_ns": 49_999_999},
    {"target_ns": 1_000_000_001}, {"cpu": -1}, {"cpu": True},
])
def test_config_rejects_invalid_or_undersampled_contract(change):
    with pytest.raises(runner.BaselineError):
        runner.Config(**change)


@pytest.mark.parametrize("limbs", [1, 4])
def test_corpus_deterministic_boundaries_carry_and_seed_sensitivity(limbs):
    first = runner.corpus(limbs, 10, 64)
    assert first == runner.corpus(limbs, 10, 64)
    assert first != runner.corpus(limbs, 11, 64)
    limit = 1 << (64 * limbs)
    for carry in (0, 1):
        assert (0, 0, carry) in first
        assert (limit - 1, limit - 1, carry) in first
    for shift in range(64, 64 * limbs, 64):
        assert ((1 << shift) - 1, 1, 0) in first
    payload = runner.encode_cases(limbs, first)
    assert payload.endswith(b"\n")
    assert runner.sha256(payload) != runner.sha256(runner.encode_cases(limbs, runner.corpus(limbs, 11, 64)))


def test_oracle_checks_little_endian_bytes_and_high_carry():
    cases = ((runner.MASK64, 1, 0), (0x0102030405060708, 0, 1))
    output = "0000000000000000 1\n0102030405060709 0\n"
    result = runner.validate_oracle_output(1, cases, output)
    expected = bytes(8) + b"\1" + bytes.fromhex("0907060504030201") + b"\0"
    assert result["output_bytes_and_carry_sha256"] == runner.sha256(expected)
    assert result["claim_class"] == "OBSERVED"
    assert result["mismatches"] == 0


@pytest.mark.parametrize("output", [
    "0000000000000000 0\n", "0000000000000001 1\n", "0000000000000000 2\n",
    "0000000000000000 True\n", "0 1\n", "000000000000000A 1\n",
    "0000000000000000  1\n", "0000000000000000 1 extra\n", "", "\n",
])
def test_oracle_rejects_wrong_bytes_carry_and_malformed_output(output):
    with pytest.raises(runner.BaselineError):
        runner.validate_oracle_output(1, ((runner.MASK64, 1, 0),), output)


@pytest.mark.parametrize("cases", [(), ((-1, 0, 0),), ((1 << 64, 0, 0),),
                                    ((0, 0, 2),), ((0, 0, True),)])
def test_encoder_rejects_invalid_state(cases):
    with pytest.raises(runner.BaselineError):
        runner.encode_cases(1, cases)


@pytest.mark.parametrize("text,expected", [("48K", 49152), ("20M", 20971520),
                                            ("1G", 1073741824), ("64", 64)])
def test_cache_size_parser(text, expected):
    assert runner.parse_cache_size(text) == expected


@pytest.mark.parametrize("text", ["", "0K", "-1K", "48KiB", "48k", "1.5M"])
def test_cache_size_parser_rejects_ambiguous_units(text):
    with pytest.raises(runner.BaselineError):
        runner.parse_cache_size(text)


def test_working_sets_derive_from_observed_caches_and_leave_headroom():
    l1, llc = 48 * 1024, 20 * 1024 * 1024
    workloads = runner.make_workloads(l1, llc, 18 * 1024 ** 3)
    assert len(workloads) == 6
    for workload in workloads:
        size = workload.count * (3 * workload.limbs + 2) * 8
        if workload.name.endswith("hot"):
            assert size <= l1 // 2
        elif workload.name.endswith("stream"):
            assert size >= 4 * llc
            assert size - 4 * llc < (3 * workload.limbs + 2) * 8
    with pytest.raises(runner.BaselineError, match="headroom"):
        runner.make_workloads(l1, llc, llc * 32)


@pytest.mark.parametrize("change", [{"limbs": True}, {"limbs": 2}, {"mode": "hot"},
                                     {"count": 0}, {"count": 1 << 25}, {"name": ""}])
def test_workload_rejects_invalid_configuration(change):
    with pytest.raises(runner.BaselineError):
        replace(runner.Workload("test", 1, "bulk", 7, "smoke"), **change)


def test_calibration_scales_without_unbounded_work():
    assert runner.next_passes(1000, 2, 60_000_000, 50_000_000) == 2
    assert runner.next_passes(1000, 2, 10_000_000, 50_000_000) == 11
    assert runner.next_passes(1 << 24, 1, 1, 50_000_000) == 59
    with pytest.raises(runner.BaselineError, match="unreachable"):
        runner.next_passes(1 << 24, 59, 1, 50_000_000)
    with pytest.raises(runner.BaselineError):
        runner.next_passes(1000, 1, 0, 50_000_000)


def test_order_is_seeded_rotating_and_complete():
    names = [f"work{i}" for i in range(6)]
    orders = runner.round_orders(names, 20260905, 7)
    assert orders == runner.round_orders(names, 20260905, 7)
    assert names == [f"work{i}" for i in range(6)]
    assert all(sorted(order) == names for order in orders)
    assert orders[1] == orders[0][1:] + orders[0][:1]
    assert orders[6] == orders[0]
    for position in range(6):
        assert len({order[position] for order in orders[:6]}) == 6


@pytest.mark.parametrize("names", [[], ["same", "same"]])
def test_order_rejects_empty_or_duplicate_names(names):
    with pytest.raises(runner.BaselineError):
        runner.round_orders(names, 0, 7)


def test_summary_preserves_outlier_and_reports_spread_not_ci():
    samples = [{"elapsed_ns": value * 10, "operation_count": 10} for value in [1, 2, 3, 4, 5, 6, 100]]
    summary = runner.summarize(samples)
    assert summary["median"] == 4
    assert summary["min"] == 1
    assert summary["max"] == 100
    assert summary["mad"] == 2
    assert summary["iqr"] == 3
    assert summary["uncertainty_kind"] == "descriptive_spread_not_confidence_interval"
    with pytest.raises(runner.BaselineError):
        runner.summarize(samples[:6])


@pytest.mark.parametrize("limbs,mode", [(1, "latency"), (4, "latency"), (1, "bulk"), (4, "bulk")])
def test_raw_sample_contract_positive(limbs, mode):
    workload = runner.Workload("test", limbs, mode, 7, "smoke")
    raw = raw_sample(workload)
    assert runner.validate_sample(raw, workload, 3, 20260905) is raw


@pytest.mark.parametrize("field,value", [
    ("protocol", "forged"), ("scope", "full_secp256k1"), ("mode", "latency"), ("limbs", True),
    ("count", 8), ("passes", 4), ("seed", 0), ("operation_count", 1), ("add64_calls", 1),
    ("elapsed_ns", 0), ("elapsed_ns", True), ("elapsed_ns", float("nan")),
    ("input_record_bytes", 1), ("output_record_bytes", 1), ("working_set_bytes", 1),
    ("logical_bytes_per_operation", 0), ("logical_bytes_processed", 0),
    ("checksum", "0123456789abcdeF"), ("checksum", 123), ("warmup_operations", 0),
    ("backend", "portable_carry"), ("no_int128_defined", 0), ("int128_macro_defined", False),
])
def test_raw_sample_rejects_wrong_contract_or_typed_count(field, value):
    workload = runner.Workload("test", 4, "bulk", 7, "smoke")
    raw = raw_sample(workload)
    raw[field] = value
    with pytest.raises(runner.BaselineError):
        runner.validate_sample(raw, workload, 3, 20260905)


def test_raw_sample_rejects_missing_extra_duplicate_json_fields():
    workload = runner.Workload("test", 1, "bulk", 7, "smoke")
    raw = raw_sample(workload)
    for mutation in ({**raw, "actual_DRAM_GBs": 100}, {key: value for key, value in raw.items() if key != "checksum"}):
        with pytest.raises(runner.BaselineError):
            runner.validate_sample(mutation, workload, 3, 20260905)
    with pytest.raises(runner.BaselineError, match="duplicate"):
        json.loads('{"elapsed_ns":1,"elapsed_ns":2}', object_pairs_hook=runner.reject_duplicate_keys)


@pytest.mark.parametrize("limbs,mode", [(1, "latency"), (4, "latency"), (1, "bulk"), (4, "bulk")])
def test_small_checksum_determinism_and_pass_semantics(limbs, mode):
    first = runner.expected_small_checksum(limbs, mode, 7, 3, 20260905)
    assert first == runner.expected_small_checksum(limbs, mode, 7, 3, 20260905)
    assert first != runner.expected_small_checksum(limbs, mode, 7, 3, 20260906)
    another_pass = runner.expected_small_checksum(limbs, mode, 7, 4, 20260905)
    assert (first == another_pass) is (mode == "bulk")


def test_measurement_keeps_calibration_warmups_all_raw_samples_and_order(monkeypatch):
    monkeypatch.setattr(runner, "context_snapshot", lambda cpu: {"cpu": cpu})
    workload = runner.Workload("test", 1, "bulk", 7, "smoke")
    calls = []

    def invoke(binary, item, passes, config):
        calls.append((item.name, passes))
        return raw_sample(item, passes, config.seed)

    result = runner.measure("unused", (workload,), runner.Config(cpu=0), invoke)
    record = result["workloads"][0]
    assert len(calls) == 10
    assert len(record["calibration"]) == 1
    assert len(record["warmups"]) == 2
    assert len(record["samples"]) == 7
    assert record["samples_below_calibration_target"] == 0
    assert record["summary"]["logical_GB_per_s_at_median"] == 40 / (50_000_000 / 7)


def test_measurement_rejects_repeated_workload_checksum_drift(monkeypatch):
    monkeypatch.setattr(runner, "context_snapshot", lambda cpu: {})
    workload = runner.Workload("test", 1, "bulk", 7, "smoke")
    calls = 0

    def invoke(binary, item, passes, config):
        nonlocal calls
        calls += 1
        raw = raw_sample(item, passes, config.seed)
        raw["checksum"] = f"{calls:016x}"
        return raw

    with pytest.raises(runner.BaselineError, match="checksum changed"):
        runner.measure("unused", (workload,), runner.Config(cpu=0), invoke)


def test_provenance_rejects_active_source_drift_even_with_real_git_blob(monkeypatch):
    original = Path.read_bytes
    header = MODULE_PATH.parents[2] / runner.HEADER

    def read_bytes(path):
        data = original(path)
        return data + b"\n" if path == header else data

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(runner.BaselineError, match="provenance mismatch"):
        runner.provenance(MODULE_PATH.parents[2])


def test_provenance_rejects_forged_frozen_git_blob(monkeypatch):
    monkeypatch.setattr(runner.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout=b"forged", stderr=b""))
    with pytest.raises(runner.BaselineError, match="provenance mismatch"):
        runner.provenance(MODULE_PATH.parents[2])


def test_canonical_json_rejects_nonfinite_measurement():
    with pytest.raises(ValueError):
        runner.canonical_json({"measurement": float("inf")})
