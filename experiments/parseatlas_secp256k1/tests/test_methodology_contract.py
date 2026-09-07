import copy
import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator


ROOT = Path(__file__).parents[1]
SCHEMA_PATH = ROOT / "schemas" / "research_contract.schema.json"
DOCS = (ROOT / "METHODOLOGY.en.md", ROOT / "METHODOLOGY.ka.md")
WORLD_IDS = [f"W{i:02d}" for i in range(1, 26)]
VIEW_IDS = [f"V{i:02d}" for i in range(1, 26)]
PARITY_RE = re.compile(r"`CONTRACT_VERSION=.*?MUTANTS_19_OF_22`")


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def errors(schema, instance):
    return list(Draft202012Validator(schema).iter_errors(instance))


def validate_metadata(contract):
    assert [item["id"] for item in contract["worlds"]] == WORLD_IDS
    assert [item["id"] for item in contract["views"]] == VIEW_IDS


def valid_record():
    return {
        "claim_id": "field-p-w02-v03-0001",
        "world_id": "W02",
        "view_id": "V03",
        "primitive": "field_p",
        "applicability": "APPLICABLE",
        "claim_class": "PROVEN",
        "claim": "The decoded candidate equals the baseline over the domain.",
        "provenance": "artifact:proof-0001",
        "limits": ["Only the declared field operation region"],
        "artifacts": ["proof-0001", "boundary-test-0001"],
        "boundary_witness": {
            "encode_in": "canonical residue to candidate state",
            "candidate": "candidate field operation",
            "decode_out": "candidate state to canonical residue",
            "input_domain": "canonical field inputs modulo p",
            "output_domain": "canonical field output modulo p",
            "ranges": "all intermediates within proved bounds",
            "preconditions": "inputs canonical",
            "carried_state": "residue and magnitude",
            "caller_visible_behavior": "same return and error behavior",
            "overflow": "proved absent under bounds",
            "aliasing": "all supported overlap cases preserved",
            "artifact": "boundary-test-0001",
        },
        "equivalence_evidence": {
            "mode": "FULL_DOMAIN_PROOF",
            "artifact": "proof-0001",
            "assumptions": ["secp256k1 field modulus p"],
        },
        "invariant_evidence": [{"invariant": "congruence", "artifact": "proof-0001"}],
        "gate_results": {
            "search_reachable": True,
            "boundary_equivalent": True,
            "invariants_eligible": True,
            "resource_feasible": True,
            "property_manifest": "N/A",
        },
    }


def test_schema_and_canonical_instance(schema):
    Draft202012Validator.check_schema(schema)
    assert errors(schema, valid_record()) == []


def test_exact_unique_worlds_and_views(schema):
    contract = schema["x-research-contract"]
    worlds, views = contract["worlds"], contract["views"]
    assert [x["id"] for x in worlds] == WORLD_IDS
    assert [x["id"] for x in views] == VIEW_IDS
    raw = SCHEMA_PATH.read_text(encoding="utf-8")
    for identifier in WORLD_IDS + VIEW_IDS:
        assert raw.count(f'"{identifier}"') == 1
    for world in worlds:
        assert all(world.get(k) for k in ("representation_policy", "derivations", "state", "normal_forms", "encode_decode", "invariants", "evaluation"))
    for view in views:
        assert all(view.get(k) for k in ("observable", "method", "applicability"))


@pytest.mark.parametrize("collection", ["worlds", "views"])
def test_mutation_rejects_missing_and_duplicate_ids(schema, collection):
    contract = copy.deepcopy(schema["x-research-contract"])
    contract[collection].pop()
    contract[collection].append(copy.deepcopy(contract[collection][0]))
    with pytest.raises(AssertionError):
        validate_metadata(contract)


def test_ranking_is_reachability_first(schema):
    assert schema["x-research-contract"]["matrix"]["ranking_order"] == [
        "search_reachable", "boundary_equivalent", "invariants_eligible",
        "resource_feasible", "cost_pareto_within_feasible_slice",
    ]
    bad = valid_record()
    bad["claim_class"] = "MEASURED"
    bad["measurement"] = measurement()
    bad["cost_vector"] = {"region_latency": 1.0, "provenance_type": "MEASURED_RUNTIME"}
    bad["gate_results"]["search_reachable"] = False
    assert errors(schema, bad)


def test_sampled_observation_cannot_pass_equivalence_gate(schema):
    bad = valid_record()
    bad["claim_class"] = "OBSERVED"
    bad["equivalence_evidence"]["mode"] = "SAMPLED_AGREEMENT"
    assert errors(schema, bad)


def test_measured_cost_can_reference_independent_proof_gates(schema):
    record = valid_record()
    record["claim_class"] = "MEASURED"
    record["measurement"] = measurement()
    record["cost_vector"] = {"region_latency": 12.0, "provenance_type": "MEASURED_RUNTIME"}
    assert errors(schema, record) == []


def test_hypothesis_cannot_pass_equivalence_gate(schema):
    bad = valid_record()
    bad["claim_class"] = "HYPOTHESIS"
    assert errors(schema, bad)


@pytest.mark.parametrize("missing", [
    "encode_in", "decode_out", "input_domain", "output_domain", "ranges",
    "preconditions", "carried_state", "caller_visible_behavior", "overflow", "aliasing",
])
def test_boundary_obligations_fail_closed(schema, missing):
    bad = valid_record()
    del bad["boundary_witness"][missing]
    assert errors(schema, bad)


@pytest.mark.parametrize("missing", [
    "environment", "software_versions", "corpus", "warm_up", "repetitions",
    "statistic", "uncertainty", "unit", "value",
])
def test_measured_metadata_is_mandatory(schema, missing):
    record = valid_record()
    record["claim_class"] = "MEASURED"
    record["gate_results"]["boundary_equivalent"] = False
    record["measurement"] = measurement()
    del record["measurement"][missing]
    assert errors(schema, record)


@pytest.mark.parametrize("missing", ["supersedes_claim_id", "reason", "author", "timestamp"])
def test_correction_metadata_is_mandatory(schema, missing):
    record = valid_record()
    record["claim_class"] = "CORRECTION"
    record["gate_results"]["boundary_equivalent"] = False
    record["correction"] = {"supersedes_claim_id": "old-claim", "reason": "wrong bound", "author": "reviewer", "timestamp": "2026-09-04T00:00:00Z"}
    del record["correction"][missing]
    assert errors(schema, record)


def test_na_requires_reason_and_forbids_cost(schema):
    record = valid_record()
    record["applicability"] = "N/A"
    record["cost_vector"] = {"dependency_depth": 1, "provenance_type": "STATIC_MODEL"}
    del record["boundary_witness"]
    assert errors(schema, record)


def test_negative_result_retains_identity(schema):
    record = valid_record()
    record["negative_result"] = {"kind": "UNREACHABLE", "detail": "no derivation", "retained_identity": False}
    assert errors(schema, record)


def measurement():
    return {"environment": "cpu/os", "software_versions": "compiler 1", "corpus": "c1", "warm_up": 10, "repetitions": 20, "statistic": "median", "uncertainty": "IQR", "unit": "cycles/region", "value": 12.0}


@pytest.mark.parametrize("claim_class,mode,valid", [
    ("PROVEN", "FULL_DOMAIN_PROOF", True),
    ("PROVEN", "MECHANICAL_FULL_DOMAIN", False),
    ("PROVEN", "SAMPLED_AGREEMENT", False),
    ("VERIFIED", "MECHANICAL_FULL_DOMAIN", True),
    ("VERIFIED", "FULL_DOMAIN_PROOF", False),
    ("VERIFIED", "SAMPLED_AGREEMENT", False),
])
def test_proof_class_requires_its_evidence_mode_independent_of_gate(schema, claim_class, mode, valid):
    record = valid_record()
    record["claim_class"] = claim_class
    record["equivalence_evidence"]["mode"] = mode
    record["gate_results"]["boundary_equivalent"] = False
    assert (errors(schema, record) == []) is valid


def test_observed_sampled_evidence_is_valid_but_ineligible(schema):
    record = valid_record()
    record["claim_class"] = "OBSERVED"
    record["equivalence_evidence"]["mode"] = "SAMPLED_AGREEMENT"
    record["gate_results"]["boundary_equivalent"] = False
    assert errors(schema, record) == []


def test_na_requires_all_gates_false_and_no_boundary_evidence(schema):
    record = valid_record()
    record["applicability"] = "N/A"
    record["applicability_reason"] = "primitive has no such boundary"
    del record["boundary_witness"]
    del record["equivalence_evidence"]
    del record["invariant_evidence"]
    record["claim_class"] = "HYPOTHESIS"
    for gate in ("search_reachable", "boundary_equivalent", "invariants_eligible", "resource_feasible"):
        record["gate_results"][gate] = False
    assert errors(schema, record) == []
    record["gate_results"]["resource_feasible"] = True
    assert errors(schema, record)


def test_static_model_cost_is_allowed_but_runtime_coordinate_is_not(schema):
    record = valid_record()
    record["cost_vector"] = {"dependency_depth": 7, "provenance_type": "STATIC_MODEL"}
    assert errors(schema, record) == []
    record["cost_vector"]["region_latency"] = 1.0
    assert errors(schema, record)


def test_static_conversion_counts_are_typed_and_valid(schema):
    record = valid_record()
    record["cost_vector"] = {
        "conversion_operation_counts": {"multiply": 2, "reduce": 1},
        "provenance_type": "STATIC_MODEL",
    }
    assert errors(schema, record) == []
    record["cost_vector"] = {"conversion_runtime": 3, "provenance_type": "STATIC_MODEL"}
    assert errors(schema, record)


@pytest.mark.parametrize("cost_vector", [
    {"provenance_type": "MEASURED_RUNTIME"},
    {"dependency_depth": 7, "provenance_type": "MEASURED_RUNTIME"},
    {"conversion_runtime": 3, "provenance_type": "MEASURED_RUNTIME"},
])
def test_runtime_provenance_requires_coordinate_measurement_artifacts_and_class(schema, cost_vector):
    record = valid_record()
    record["cost_vector"] = cost_vector
    assert errors(schema, record)
    record["claim_class"] = "MEASURED"
    record["measurement"] = measurement()
    if len(cost_vector) > 1 and "dependency_depth" not in cost_vector:
        assert errors(schema, record) == []
    else:
        assert errors(schema, record)


def test_runtime_provenance_requires_artifacts_independently(schema):
    record = valid_record()
    record["claim_class"] = "MEASURED"
    record["measurement"] = measurement()
    record["cost_vector"] = {"conversion_runtime": 3, "provenance_type": "MEASURED_RUNTIME"}
    del record["artifacts"]
    assert errors(schema, record)


def test_runtime_cost_requires_measurement_metadata(schema):
    record = valid_record()
    record["claim_class"] = "MEASURED"
    record["cost_vector"] = {"region_throughput": 1.0, "provenance_type": "MEASURED_RUNTIME"}
    assert errors(schema, record)
    record["measurement"] = measurement()
    assert errors(schema, record) == []


@pytest.mark.parametrize("missing", ["seed", "artifacts"])
def test_negative_result_requires_reproduction_inputs(schema, missing):
    record = valid_record()
    record["seed"] = "DETERMINISTIC-NOT-APPLICABLE"
    record["negative_result"] = {"kind": "UNREACHABLE", "detail": "no derivation", "retained_identity": True}
    del record[missing]
    assert errors(schema, record)


def test_bilingual_numeric_and_rule_parity():
    texts = [path.read_text(encoding="utf-8") for path in DOCS]
    blocks = [PARITY_RE.search(text).group(0) for text in texts]
    assert blocks[0] == blocks[1]
    anchors = ["W01–W25", "V01–V25", "PENDING IMPORT", "decode_out(candidate(encode_in(x))) = baseline(x)", "52/1450", "19/22", "2.333", "E0039=46:1_MODEL_DEFECT+8_OPERATOR_LANGUAGE+18_TOKEN_WORD+19_RULE_LIMIT", "fef231d4e4173bd016fb2a3a1eff67087396a203", "FieldElement::to_bytes", "ORACLE_011"]
    for anchor in anchors:
        assert all(anchor in text for text in texts), anchor
    def catalog_rows(text, prefix):
        rows = []
        for line in text.splitlines():
            if re.match(rf"^\|{prefix}\d{{2}}\|", line):
                cells = line.strip("|").split("|")
                assert len(cells) == 4 and all(cell.strip() for cell in cells)
                rows.append(cells)
        return rows

    world_rows = [catalog_rows(text, "W") for text in texts]
    view_rows = [catalog_rows(text, "V") for text in texts]
    assert [[row[0] for row in rows] for rows in world_rows] == [WORLD_IDS, WORLD_IDS]
    assert [[row[0] for row in rows] for rows in view_rows] == [VIEW_IDS, VIEW_IDS]
    assert all(en[1:] != ka[1:] for en, ka in zip(*world_rows))


def test_pinned_inputs_and_counts(schema):
    contract = schema["x-research-contract"]
    assert contract["matrix"]["cells"] == 625
    assert contract["immutable_inputs"] == {
        "historical_baseline": "d71b406c95141d81749671431a4c0f4605e0c4e4",
        "guidance_revision": "13fac0a300cf07eac9424434ec1e2c6bd5f75564",
        "parallel_worlds_revision": "f428fb53948bc15c4668c303366231817a947708",
        "direction": "B",
        "experiments": ["0037", "0038", "0039", "0040", "0041"],
    }
