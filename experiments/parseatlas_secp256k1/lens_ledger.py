"""Validate the bounded L0 diagnostic ledger; never rank or grant eligibility.

This is not a research_contract.schema.json candidate. That contract currently
has no L0 primitive. Its unchanged V01--V25 catalog is reused only as lenses.
Source hashes bind the evidence snapshot, not the truth of arbitrary prose;
the latter still needs independent review. No imports execute evidence code.
"""

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys


PREFIX = "experiments/parseatlas_secp256k1/"
DEFAULT_LEDGER = PREFIX + "data/l0_25_lens_ledger.json"
SCHEMA = PREFIX + "schemas/research_contract.schema.json"
PENDING = "PENDING IMPORT"
FORMAT = "parseatlas_l0_diagnostic_ledger_v1"
SCOPES = {"methodology", "python_model", "native_cpp", "whole_process_pmu"}
OBJECT = {
    "kind": "L0_unsigned_addition_with_carry",
    "native_word_bits": 64,
    "native_limb_counts": [1, 4],
    "equation": "a + b + carry_in = low + 2**(word_bits*limbs) * carry_out",
    "boundary": "fixed-width little-endian low bytes and separate carry bit",
    "modulus": "none; not field_p or scalar_n",
    "reference_commit": "fef231d4e4173bd016fb2a3a1eff67087396a203",
}
GATES = {
    "full_domain_boundary_equivalence": PENDING,
    "constant_time_eligibility": PENDING,
    "production_acceptance": "NOT ESTABLISHED",
    "ranking": "NOT PERFORMED",
    "novelty": "NOT CLAIMED",
}
# This first snapshot lacks these artifacts. A later audit needs a new version
# and reviewed validator changes, not a textual status upgrade in this ledger.
PENDING_LENSES = {"V09", "V11", "V20", "V23", "V24"}


class LedgerError(ValueError):
    """The diagnostic record or one of its bound artifacts is inconsistent."""


def require(condition, message):
    if not condition:
        raise LedgerError(message)


def fields(value, expected, label):
    require(type(value) is dict and set(value) == set(expected), f"{label}: fields")


def nonempty(value, label):
    require(type(value) is str and bool(value.strip()), f"{label}: nonempty text")


def texts(value, label):
    require(type(value) is list and bool(value), f"{label}: nonempty list")
    for item in value:
        nonempty(item, label)


def safe_path(root, relative):
    """Exact relative regular files only; reject symlinks even inside the root."""
    nonempty(relative, "artifact path")
    path = PurePosixPath(relative)
    require(not path.is_absolute() and ".." not in path.parts
            and str(path) == relative and "\\" not in relative, "unsafe artifact path")
    resolved_root = Path(root).resolve(strict=True)
    current = resolved_root
    for part in path.parts:
        current /= part
        require(not current.is_symlink(), "symlink artifact path")
    resolved = current.resolve(strict=True)
    require(resolved.is_relative_to(resolved_root) and resolved.is_file(),
            "artifact must be a repository file")
    return resolved


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw):
    def reject_constant(value):
        raise LedgerError(f"non-finite JSON constant: {value}")
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=reject_constant)


def pointer(document, location):
    """A strict read-only JSON pointer; no expression evaluation or wildcards."""
    require(type(location) is str and location.startswith("/"), "JSON pointer required")
    current = document
    for raw in location[1:].split("/"):
        require(re.search(r"~(?![01])", raw) is None, "invalid JSON pointer escape")
        token = raw.replace("~1", "/").replace("~0", "~")
        if type(current) is list:
            require(re.fullmatch(r"0|[1-9][0-9]*", token) is not None,
                    "invalid JSON pointer index")
            require(int(token) < len(current), "dangling JSON pointer")
            current = current[int(token)]
        else:
            require(type(current) is dict and token in current, "dangling JSON pointer")
            current = current[token]
    return current


def measurement_provenance(source, kind, documents):
    """Require context at the source, not decorative provenance in the ledger."""
    if kind == "collection":
        require(source.get("protocol") == "parseatlas_l0_collection_v1", "collection protocol")
        for key in ("machine", "build", "correctness", "context_start", "context_end", "provenance"):
            require(bool(source.get(key)), f"measurement provenance missing {key}")
        config = source.get("config", {})
        for key in ("warmups", "repetitions"):
            require(type(config.get(key)) is int and config[key] >= 2,
                    f"measurement provenance missing {key}")
        require(type(config.get("seed")) is int, "measurement corpus seed")
        require(bool(source["build"].get("compiler_version"))
                and bool(source["build"].get("flags")), "measurement software versions")
        workloads = source.get("measurement", {}).get("workloads", [])
        require(bool(workloads), "measurement workloads missing")
        for workload in workloads:
            require(len(workload.get("samples", [])) == config["repetitions"]
                    and len(workload.get("warmups", [])) == config["warmups"],
                    "measurement repetitions mismatch")
            summary = workload.get("summary", {})
            for key in ("median", "min", "max", "mad", "iqr", "unit", "uncertainty_kind"):
                require(key in summary, f"measurement statistic missing {key}")
    elif kind == "whole_process_pmu":
        require(source.get("protocol") == "parseatlas_l0_process_counters_v1", "PMU protocol")
        require(source.get("scope") ==
                "Whole benchmark process user-space PMU events, not timed-region-only counters",
                "PMU scope cannot be upgraded to inner-loop")
        reference = documents.get("cpu4", {})
        measurement_provenance(reference, "collection", documents)
        require(source.get("binary_sha256") == reference["build"]["binary_sha256"]
                and source.get("probe_sha256") == reference["provenance"]["probe_sha256"],
                "PMU reference identity mismatch")
        require(bool(source.get("recorded_at_utc")) and bool(source.get("limits")), "PMU provenance")
        require(bool(source.get("workloads")), "PMU workloads missing")
        for workload in source["workloads"]:
            repeat = workload.get("repeat")
            require(type(repeat) is int and repeat >= 2, "PMU repetitions")
            require(len(workload.get("probe_outputs", [])) == repeat, "PMU outputs missing")
            rows = workload.get("aggregate_perf_rows", [])
            require(bool(rows) and all("variation_as_reported" in row for row in rows),
                    "PMU uncertainty missing")
    else:
        raise LedgerError("unknown measured provenance kind")


def validate(ledger, root):
    fields(ledger, ("format", "object", "coverage", "eligibility", "artifacts", "lenses"), "ledger")
    require(ledger["format"] == FORMAT, "diagnostic format only")
    require(ledger["object"] == OBJECT, "L0 object contract changed")
    require(ledger["coverage"] == {"meaning": "addressed_not_completed_or_eligible", "addressed": 25},
            "coverage cannot mean completed or eligible")
    require(ledger["eligibility"] == GATES, "unestablished eligibility or novelty upgrade")
    require(type(ledger["artifacts"]) is list and bool(ledger["artifacts"]), "artifacts required")
    artifacts, documents = {}, {}
    for artifact in ledger["artifacts"]:
        fields(artifact, ("id", "path", "sha256"), "artifact")
        aid = artifact["id"]
        nonempty(aid, "artifact id")
        require(aid not in artifacts, "duplicate artifact id")
        require(type(artifact["sha256"]) is str and
                re.fullmatch("[0-9a-f]{64}", artifact["sha256"]) is not None, "artifact hash syntax")
        path = safe_path(root, artifact["path"])
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == artifact["sha256"], f"stale artifact: {aid}")
        artifacts[aid] = artifact
        if path.suffix == ".json":
            documents[aid] = parse_json(raw)
    require("schema" in artifacts and artifacts["schema"]["path"] == SCHEMA, "canonical schema missing")
    for aid in ("cpu0", "cpu4"):
        require(aid in documents, "baseline collection missing")
        provenance = documents[aid].get("provenance", {})
        require(provenance.get("frozen_engine_commit") == OBJECT["reference_commit"],
                "baseline frozen reference mismatch")
        for source_id, hash_key in (("probe", "probe_sha256"), ("runner", "runner_sha256")):
            require(source_id in artifacts and
                    provenance.get(hash_key) == artifacts[source_id]["sha256"],
                    "baseline source identity mismatch")
    catalog = documents["schema"]["x-research-contract"]["views"]
    require(len(catalog) == 25 and [v["id"] for v in catalog] ==
            [f"V{i:02d}" for i in range(1, 26)], "canonical 25-view catalog")
    lenses = ledger["lenses"]
    require(type(lenses) is list and len(lenses) == 25, "exactly 25 lenses required")
    require([v.get("id") for v in lenses if type(v) is dict] ==
            [v["id"] for v in catalog], "missing, duplicate, reordered or unknown lens ID")
    for lens, canonical in zip(lenses, catalog):
        fields(lens, ("id", "observable", "method", "assessment", "claim_class", "evidence_scopes",
                      "artifact_refs", "observation", "limitations", "next_evidence", "metrics"), "lens")
        for key in ("id", "observable", "method"):
            require(lens[key] == canonical[key], f"canonical lens {key} changed")
        require(lens["assessment"] in {"evidenced", "partial", "pending"}, "invalid assessment; unknown is not N/A")
        require(lens["claim_class"] in {"OBSERVED", "MEASURED", "HYPOTHESIS"},
                "this diagnostic snapshot does not admit proof/verification upgrades")
        if lens["id"] in PENDING_LENSES:
            require(lens["assessment"] == "pending", "missing artifact cannot upgrade pending lens")
        if lens["id"] in {"V03", "V08"}:
            require(lens["assessment"] != "evidenced", "full-domain/invariant eligibility unestablished")
        texts(lens["evidence_scopes"], "evidence scopes")
        require(set(lens["evidence_scopes"]) <= SCOPES, "unknown evidence scope")
        texts(lens["artifact_refs"], "artifact refs")
        require(len(set(lens["artifact_refs"])) == len(lens["artifact_refs"])
                and set(lens["artifact_refs"]) <= artifacts.keys(), "dangling or duplicate evidence")
        nonempty(lens["observation"], "factual observation")
        texts(lens["limitations"], "limitations")
        texts(lens["next_evidence"], "missing next evidence")
        require(type(lens["metrics"]) is list and bool(lens["metrics"]), "metrics required")
        measured = 0
        names = set()
        for metric in lens["metrics"]:
            require(type(metric) is dict, "metric object required")
            nonempty(metric.get("name"), "metric name")
            require(metric["name"] not in names, "duplicate metric name")
            names.add(metric["name"])
            nonempty(metric.get("unit"), "metric unit")
            value = metric.get("value")
            if value == PENDING:
                fields(metric, ("name", "unit", "value", "reason"), "pending metric")
                nonempty(metric["reason"], "pending reason")
                continue
            fields(metric, ("name", "unit", "value", "artifact", "pointer", "provenance_kind"), "imported metric")
            require(type(value) in (int, float) and math.isfinite(value),
                    "unknown metric must be exactly PENDING IMPORT")
            aid = metric["artifact"]
            require(aid in lens["artifact_refs"] and aid in documents, "dangling metric evidence")
            imported = pointer(documents[aid], metric["pointer"])
            require(type(imported) is type(value) and imported == value, "metric differs from bound source")
            if metric["provenance_kind"] != "observed_corpus":
                kind = metric["provenance_kind"]
                if kind == "collection":
                    match = re.fullmatch(r"/measurement/workloads/(0|[1-9][0-9]*)/summary/median",
                                         metric["pointer"])
                    require(match is not None and lens["id"] in {"V16", "V17"},
                            "timing metric scope")
                    workload = documents[aid]["measurement"]["workloads"][int(match[1])]
                    expected_mode = "latency" if lens["id"] == "V16" else "bulk"
                    require(workload["workload"]["mode"] == expected_mode
                            and metric["unit"] == workload["summary"]["unit"],
                            "timing unit or regime mismatch")
                elif kind == "whole_process_pmu":
                    match = re.fullmatch(r"/workloads/(0|[1-9][0-9]*)/aggregate_perf_rows/([123])/count_as_reported",
                                         metric["pointer"])
                    require(match is not None, "PMU metric scope")
                    expected = {"1": ("V18", "perf aggregate user-space instructions as reported"),
                                "2": ("V19", "perf aggregate cache references as reported"),
                                "3": ("V19", "perf aggregate cache misses as reported")}[match[2]]
                    require((lens["id"], metric["unit"]) == expected, "PMU unit or lens mismatch")
                measurement_provenance(documents[aid], kind, documents)
                expected_scope = "whole_process_pmu" if metric["provenance_kind"] == "whole_process_pmu" else "native_cpp"
                require(expected_scope in lens["evidence_scopes"], "measured evidence scope mismatch")
                measured += 1
            else:
                require(lens["id"] == "V02" and lens["claim_class"] == "OBSERVED"
                        and re.fullmatch(r"/correctness/[01]/(cases|mismatches)",
                                         metric["pointer"]) is not None, "corpus evidence scope")
                require(metric["unit"] == ("input triples" if metric["pointer"].endswith("/cases")
                                           else "mismatches"), "corpus metric unit")
        if lens["claim_class"] == "MEASURED":
            require(measured > 0, "measured assertion missing measurement provenance")
    return {"valid": True, "format": FORMAT, "addressed": 25,
            "assessment_counts": dict(sorted(Counter(v["assessment"] for v in lenses).items())),
            "artifacts": len(artifacts), "eligibility": GATES.copy(),
            "limits": "Structural/provenance validation only; prose still requires independent review."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate"])
    parser.add_argument("--ledger", default=DEFAULT_LEDGER, help="exact repository-relative JSON path")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    try:
        result = validate(parse_json(safe_path(root, args.ledger).read_bytes()), root)
    except (LedgerError, OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
