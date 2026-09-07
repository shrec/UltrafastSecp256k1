"""Collect bounded native L0 baselines; stdout is the sole result artifact.

This Linux collector intentionally measures serially: concurrently timing independent
workloads would change cache, frequency and bandwidth contention, not just speed.
It does not rank worlds, modify production or claim actual DRAM traffic.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile


FROZEN_COMMIT = "fef231d4e4173bd016fb2a3a1eff67087396a203"
HEADER = "src/cpu/include/secp256k1/detail/arith64.hpp"
HEADER_SHA256 = "cc95a42137df602eaee3a560993b8c59c091e017364a0345be486ba012de205d"
PROBE = "experiments/parseatlas_secp256k1/probes/l0_baseline.cpp"
FLAGS = ("-std=c++17", "-O3", "-march=native", "-DNDEBUG", "-Wall", "-Wextra")
MASK64 = (1 << 64) - 1
MAX_OPERATIONS = 1_000_000_000
MAX_PASSES = 1_000_000
MAX_WORKING_SET = 256 * 1024 * 1024
PROTOCOL = "parseatlas_l0_baseline_v1"
HEX16 = re.compile(r"[0-9a-f]{16}\Z")


class BaselineError(ValueError):
    """Invalid experiment contract, provenance, oracle or measurement."""


def exact_int(value, minimum, maximum, label):
    if type(value) is not int or not minimum <= value <= maximum:
        raise BaselineError(f"invalid {label}")
    return value


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Config:
    seed: int = 20260905
    random_cases: int = 512
    warmups: int = 2
    repetitions: int = 7
    target_ns: int = 50_000_000
    cpu: int | None = None

    def __post_init__(self):
        exact_int(self.seed, 0, MASK64, "seed")
        exact_int(self.random_cases, 1, 100_000, "random cases")
        exact_int(self.warmups, 2, 20, "warmups")
        exact_int(self.repetitions, 7, 101, "repetitions")
        exact_int(self.target_ns, 50_000_000, 1_000_000_000, "target duration")
        if self.cpu is not None:
            exact_int(self.cpu, 0, 1_000_000, "cpu")


@dataclass(frozen=True)
class Workload:
    name: str
    limbs: int
    mode: str
    count: int
    regime: str

    def __post_init__(self):
        if type(self.name) is not str or not self.name or type(self.regime) is not str or not self.regime:
            raise BaselineError("workload names/regimes must be nonempty text")
        if type(self.limbs) is not int or self.limbs not in (1, 4):
            raise BaselineError("unsupported workload limbs")
        if self.mode not in ("latency", "bulk"):
            raise BaselineError("unsupported workload mode")
        exact_int(self.count, 1, 1 << 24, "workload count")


def corpus(limbs, seed, random_cases):
    if limbs not in (1, 4) or type(limbs) is not int:
        raise BaselineError("unsupported limb count")
    exact_int(seed, 0, MASK64, "seed")
    exact_int(random_cases, 1, 100_000, "random cases")
    limit = 1 << (64 * limbs)
    boundaries = {0, 1, 2, limit - 1, limit - 2, limit // 2 - 1, limit // 2}
    for shift in range(64, 64 * limbs, 64):
        boundaries.update({(1 << shift) - 1, 1 << shift, (1 << shift) + 1})
    cases = [(a, b, carry) for a in sorted(boundaries)
             for b in sorted(boundaries) for carry in (0, 1)]
    rng = random.Random(seed ^ (limbs << 32))
    cases.extend((rng.getrandbits(64 * limbs), rng.getrandbits(64 * limbs),
                  rng.randrange(2)) for _ in range(random_cases))
    return tuple(cases)


def encode_cases(limbs, cases):
    limit = 1 << (64 * limbs)
    lines = []
    for a, b, carry in cases:
        exact_int(a, 0, limit - 1, "a")
        exact_int(b, 0, limit - 1, "b")
        exact_int(carry, 0, 1, "carry")
        words = [f"{(value >> (64 * i)) & MASK64:016x}"
                 for value in (a, b) for i in range(limbs)]
        lines.append(" ".join([str(carry), *words]))
    if not lines:
        raise BaselineError("empty correctness corpus")
    return ("\n".join(lines) + "\n").encode("ascii")


def validate_oracle_output(limbs, cases, output):
    lines = output.splitlines()
    if len(lines) != len(cases):
        raise BaselineError("oracle output count mismatch")
    limit = 1 << (64 * limbs)
    decoded = bytearray()
    for index, ((a, b, carry), line) in enumerate(zip(cases, lines)):
        fields = line.split(" ")
        if (len(fields) != limbs + 1 or fields[-1] not in ("0", "1")
                or any(HEX16.fullmatch(word) is None for word in fields[:-1])):
            raise BaselineError(f"invalid oracle output at case {index}")
        low_bytes = b"".join(int(word, 16).to_bytes(8, "little") for word in fields[:-1])
        actual_carry = int(fields[-1])
        expected_carry, expected_low = divmod(a + b + carry, limit)
        if (low_bytes, actual_carry) != (expected_low.to_bytes(8 * limbs, "little"), expected_carry):
            raise BaselineError(f"byte/carry disagreement at case {index}")
        decoded.extend(low_bytes)
        decoded.append(actual_carry)
    return {"claim_class": "OBSERVED", "cases": len(cases), "mismatches": 0,
            "output_bytes_and_carry_sha256": sha256(decoded)}


def parse_cache_size(text):
    match = re.fullmatch(r"([1-9][0-9]*)([KMG]?)", text.strip())
    if not match:
        raise BaselineError("unrecognized cache size")
    return int(match[1]) * (1024 ** {"": 0, "K": 1, "M": 2, "G": 3}[match[2]])


def make_workloads(l1_data_bytes, llc_bytes, available_bytes):
    for value, label in ((l1_data_bytes, "L1"), (llc_bytes, "LLC"),
                         (available_bytes, "available memory")):
        exact_int(value, 1, 1 << 60, label)
    cap = min(MAX_WORKING_SET, available_bytes // 16)
    target = 4 * llc_bytes
    if target > cap:
        raise BaselineError("insufficient memory headroom for preregistered 4x LLC working set")
    result = []
    for limbs in (1, 4):
        record_bytes = (3 * limbs + 2) * 8
        hot_count = max(1, (l1_data_bytes // 2) // record_bytes)
        stream_count = math.ceil(target / record_bytes)
        if stream_count * record_bytes > cap or stream_count > (1 << 24):
            raise BaselineError("working set exceeds probe resource ceiling")
        result.extend((Workload(f"u{64 * limbs}_latency", limbs, "latency", 4096,
                                "loop_carried_conditioned"),
                       Workload(f"u{64 * limbs}_hot", limbs, "bulk", hot_count,
                                "working_set_below_half_observed_L1D"),
                       Workload(f"u{64 * limbs}_stream", limbs, "bulk", stream_count,
                                "working_set_at_least_four_times_observed_LLC")))
    return tuple(result)


def next_passes(count, passes, elapsed_ns, target_ns):
    exact_int(count, 1, 1 << 24, "count")
    exact_int(passes, 1, MAX_PASSES, "passes")
    exact_int(elapsed_ns, 1, 1 << 63, "elapsed duration")
    exact_int(target_ns, 1, 1 << 63, "target duration")
    ceiling = min(MAX_PASSES, MAX_OPERATIONS // count)
    if elapsed_ns >= target_ns:
        return passes
    candidate = min(ceiling, max(passes + 1, math.ceil(passes * target_ns / elapsed_ns * 1.1)))
    if candidate <= passes:
        raise BaselineError("calibration target unreachable within operation ceiling")
    return candidate


def round_orders(names, seed, repetitions):
    if len(names) != len(set(names)) or not names:
        raise BaselineError("workload names must be unique and nonempty")
    order = list(names)
    random.Random(seed ^ 0x504152534541544C).shuffle(order)
    return [order[index % len(order):] + order[:index % len(order)]
            for index in range(repetitions)]


def summarize(samples):
    if len(samples) < 7:
        raise BaselineError("at least seven measured repetitions required")
    values = []
    for sample in samples:
        elapsed = exact_int(sample["elapsed_ns"], 1, 1 << 63, "elapsed duration")
        count = exact_int(sample["operation_count"], 1, MAX_OPERATIONS, "operation count")
        values.append(elapsed / count)
    median = statistics.median(values)
    quartiles = statistics.quantiles(values, n=4, method="inclusive")
    return {"unit": "ns/logical_addition", "median": median, "min": min(values),
            "max": max(values), "mad": statistics.median(abs(v - median) for v in values),
            "iqr": quartiles[2] - quartiles[0], "iqr_method": "inclusive",
            "repetitions": len(values), "uncertainty_kind": "descriptive_spread_not_confidence_interval"}


RAW_FIELDS = frozenset({"protocol", "scope", "mode", "limbs", "count", "passes", "seed",
    "operation_count", "add64_calls", "elapsed_ns", "input_record_bytes", "output_record_bytes",
    "working_set_bytes", "logical_bytes_per_operation", "logical_bytes_processed", "checksum",
    "warmup_operations", "backend", "no_int128_defined", "int128_macro_defined"})


def validate_sample(raw, workload, passes, seed):
    exact_int(passes, 1, MAX_PASSES, "passes")
    exact_int(seed, 0, MASK64, "seed")
    if type(raw) is not dict or set(raw) != RAW_FIELDS:
        raise BaselineError("raw sample fields mismatch")
    input_bytes, output_bytes = (2 * workload.limbs + 1) * 8, (workload.limbs + 1) * 8
    logical = input_bytes + output_bytes if workload.mode == "bulk" else 0
    expected = {"protocol": PROTOCOL, "scope": "original_add64_composition",
        "mode": workload.mode, "limbs": workload.limbs, "count": workload.count,
        "passes": passes, "seed": seed, "operation_count": workload.count * passes,
        "add64_calls": workload.count * passes * workload.limbs,
        "input_record_bytes": input_bytes, "output_record_bytes": output_bytes,
        "working_set_bytes": workload.count * logical if logical else input_bytes,
        "logical_bytes_per_operation": logical,
        "logical_bytes_processed": workload.count * passes * logical,
        "warmup_operations": workload.count if logical else min(workload.count, 4096),
        "backend": "gcc_uint128", "no_int128_defined": False, "int128_macro_defined": True}
    for key, value in expected.items():
        if type(raw[key]) is not type(value) or raw[key] != value:
            raise BaselineError(f"raw sample {key} mismatch")
    exact_int(raw["elapsed_ns"], 1, 1 << 63, "elapsed duration")
    if type(raw["checksum"]) is not str or HEX16.fullmatch(raw["checksum"]) is None:
        raise BaselineError("invalid checksum")
    if raw["operation_count"] > MAX_OPERATIONS or raw["working_set_bytes"] > MAX_WORKING_SET:
        raise BaselineError("raw sample resource ceiling exceeded")
    return raw


def expected_small_checksum(limbs, mode, count, passes, seed):
    """Independent bigint replay of bounded benchmark inputs and loop outputs.

    This checks the two timed kernels before sustained timing, not every output
    of the later large streaming workloads.
    """
    if type(limbs) is not int or limbs not in (1, 4) or mode not in ("latency", "bulk"):
        raise BaselineError("unsupported smoke workload")
    exact_int(count, 1, 64, "smoke count")
    exact_int(passes, 1, 16, "smoke passes")
    exact_int(seed, 0, MASK64, "seed")
    state = seed

    def random_word():
        nonlocal state
        state = (state + 0x9E3779B97F4A7C15) & MASK64
        value = state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
        return value ^ (value >> 31)

    def record():
        return ([random_word() for _ in range(limbs)],
                [random_word() for _ in range(limbs)], random_word() & 1)

    def add(a, b, carry):
        whole_a = sum(word << (64 * index) for index, word in enumerate(a))
        whole_b = sum(word << (64 * index) for index, word in enumerate(b))
        outgoing, low = divmod(whole_a + whole_b + carry, 1 << (64 * limbs))
        return [(low >> (64 * index)) & MASK64 for index in range(limbs)], outgoing

    values = []
    if mode == "bulk":
        for _ in range(count):
            low, carry = add(*record())
            values.extend([*low, carry])
    else:
        a, b, carry = record()
        for _ in range(min(count, 4096) + count * passes):
            a, carry = add(a, b, carry)
            b = [(((word << 13) | (word >> 51)) & MASK64) ^ low
                 for word, low in zip(b, a)]
        values = [*a, *b, carry]
    checksum = 0xCBF29CE484222325
    for value in values:
        checksum = ((checksum ^ value) * 0x100000001B3) & MASK64
    return f"{checksum:016x}"


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BaselineError("duplicate raw JSON key")
        result[key] = value
    return result


def run_command(argv, *, cwd=None, input_bytes=None, cpu=None, timeout=120, env=None):
    preexec = None
    if cpu is not None:
        # Affinity changes only this child, never the owner shell or machine policy.
        preexec = lambda: os.sched_setaffinity(0, {cpu})
    result = subprocess.run([str(arg) for arg in argv], cwd=cwd, input=input_bytes,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            timeout=timeout, check=False, preexec_fn=preexec, env=env)
    if result.returncode != 0:
        raise BaselineError(f"command failed ({result.returncode}): {argv!r}: "
                            + result.stderr.decode("utf-8", errors="replace")[:2000])
    return result.stdout.decode("utf-8")


def provenance(repo):
    header = repo / HEADER
    probe = repo / PROBE
    for path in (header, probe):
        if path.is_symlink() or path.resolve() != path:
            raise BaselineError("source path is redirected by symlink")
    active = header.read_bytes()
    frozen = subprocess.run(["git", "show", f"{FROZEN_COMMIT}:{HEADER}"], cwd=repo,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if frozen.returncode != 0 or sha256(frozen.stdout) != HEADER_SHA256 or active != frozen.stdout:
        raise BaselineError("frozen/active arithmetic header provenance mismatch")
    return {"frozen_engine_commit": FROZEN_COMMIT, "header_path": HEADER,
            "header_sha256": sha256(active), "header_matches_frozen_git_blob": True,
            "active_engine_commit": run_command(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
            "probe_path": PROBE, "probe_sha256": sha256(probe.read_bytes()),
            "runner_sha256": sha256(Path(__file__).read_bytes())}


def optional_text(path):
    try:
        return Path(path).read_text().strip()
    except (OSError, UnicodeError):
        return None


def observe_machine(cpu):
    caches = []
    for entry in sorted(Path(f"/sys/devices/system/cpu/cpu{cpu}/cache").glob("index*")):
        kind, level, size = (optional_text(entry / field) for field in ("type", "level", "size"))
        if kind in ("Data", "Unified") and level is not None and size is not None:
            caches.append({"source": str(entry), "type": kind, "level": int(level),
                           "size_bytes": parse_cache_size(size),
                           "shared_cpu_list": optional_text(entry / "shared_cpu_list"),
                           "coherency_line_size": optional_text(entry / "coherency_line_size")})
    l1 = [cache["size_bytes"] for cache in caches if cache["level"] == 1]
    if not l1 or not caches:
        raise BaselineError("observed target CPU cache topology unavailable")
    top_level = max(cache["level"] for cache in caches)
    llc = max(cache["size_bytes"] for cache in caches if cache["level"] == top_level)
    meminfo = optional_text("/proc/meminfo") or ""
    available = re.search(r"^MemAvailable:\s+([0-9]+) kB$", meminfo, re.MULTILINE)
    if available is None:
        raise BaselineError("observed available memory unavailable")
    cpuinfo = optional_text("/proc/cpuinfo") or ""
    model = re.search(r"^model name\s*:\s*(.+)$", cpuinfo, re.MULTILINE)
    return {"platform": sys.platform, "uname": list(os.uname()),
            "python": sys.version, "cpu_model_first_proc_entry": model.group(1) if model else None,
            "selected_cpu": cpu, "allowed_cpus": sorted(os.sched_getaffinity(0)),
            "observed_logical_cpus": os.cpu_count(), "measurement_workers": 1,
            "parallelism_reason": "serial timing avoids cross-workload cache/frequency/bandwidth interference",
            "caches": caches, "l1_data_bytes": min(l1), "llc_bytes": llc,
            "mem_available_bytes": int(available.group(1)) * 1024,
            "numa_policy": "default; benchmark allocation/first-touch occurs on pinned child CPU"}


def context_snapshot(cpu):
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq")
    return {"utc": datetime.now(timezone.utc).isoformat(),
            "loadavg": optional_text("/proc/loadavg"),
            "scaling_cur_freq_khz": optional_text(base / "scaling_cur_freq"),
            "scaling_governor": optional_text(base / "scaling_governor"),
            "scaling_driver": optional_text(base / "scaling_driver"),
            "cpuinfo_max_freq_khz": optional_text(base / "cpuinfo_max_freq"),
            "intel_pstate_no_turbo": optional_text("/sys/devices/system/cpu/intel_pstate/no_turbo")}


def build(repo, directory):
    compiler = shutil.which("g++")
    if compiler is None:
        raise BaselineError("g++ unavailable")
    env = os.environ.copy()
    influence = ("CPATH", "CPLUS_INCLUDE_PATH", "C_INCLUDE_PATH", "LIBRARY_PATH",
                 "GCC_EXEC_PREFIX", "COMPILER_PATH", "LD_PRELOAD", "LD_LIBRARY_PATH")
    present = {key: env[key] for key in influence if env.get(key)}
    if present:
        raise BaselineError("compiler/runtime environment overrides present: " + ", ".join(sorted(present)))
    binary = Path(directory) / "l0_baseline"
    argv = [compiler, *FLAGS, f"-I{repo / 'src/cpu/include'}", str(repo / PROBE), "-o", str(binary)]
    run_command(argv, cwd=repo, env=env)
    compiler_version = run_command([compiler, "--version"]).strip()
    if "g++" not in compiler_version.splitlines()[0] or "clang" in compiler_version.lower():
        raise BaselineError("collector requires recorded native GCC g++ baseline")
    metadata = {"compiler": compiler, "compiler_version": compiler_version,
                "target": run_command([compiler, "-dumpmachine"]).strip(),
                "argv": argv, "flags": list(FLAGS), "lto": False,
                "binary_sha256": sha256(binary.read_bytes()),
                "compiler_environment_overrides": present}
    return binary, metadata


def check_reference(binary, config):
    evidence = []
    for limbs in (1, 4):
        cases = corpus(limbs, config.seed, config.random_cases)
        payload = encode_cases(limbs, cases)
        output = run_command([binary, "--check", limbs], input_bytes=payload, cpu=config.cpu)
        result = validate_oracle_output(limbs, cases, output)
        evidence.append({"limbs": limbs, "word_bits": 64, "seed": config.seed,
                         "corpus_encoding": "strict check stdin ASCII incl final newline",
                         "corpus_sha256": sha256(payload), **result})
    return evidence


def invoke_sample(binary, workload, passes, config):
    text = run_command([binary, "--bench", workload.limbs, workload.mode, workload.count,
                        passes, config.seed], cpu=config.cpu, timeout=120)
    try:
        raw = json.loads(text, object_pairs_hook=reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise BaselineError("probe returned malformed JSON") from exc
    return validate_sample(raw, workload, passes, config.seed)


def measure(binary, workloads, config, invoke=invoke_sample):
    calibration = {}
    passes_by_name = {}
    # Fix the complete order before seeing any timing. Retain all calibration and
    # warm-up samples separately; do not select or trim measured repetitions.
    orders = round_orders([workload.name for workload in workloads], config.seed, config.repetitions)
    by_name = {workload.name: workload for workload in workloads}
    for name in orders[0]:
        workload = by_name[name]
        passes, trials = 1, []
        for _ in range(8):
            sample = invoke(binary, workload, passes, config)
            trials.append(sample)
            updated = next_passes(workload.count, passes, sample["elapsed_ns"], config.target_ns)
            if updated == passes:
                break
            passes = updated
        else:
            raise BaselineError("calibration failed to converge within eight trials")
        calibration[name], passes_by_name[name] = trials, passes
    warmups = {workload.name: [] for workload in workloads}
    samples = {workload.name: [] for workload in workloads}
    checksums = {}
    for phase, count, destination in (("warmup", config.warmups, warmups),
                                       ("measurement", config.repetitions, samples)):
        phase_orders = round_orders(list(by_name), config.seed, count)
        for round_index, order in enumerate(phase_orders):
            for name in order:
                snapshot_before = context_snapshot(config.cpu)
                raw = invoke(binary, by_name[name], passes_by_name[name], config)
                if name in checksums and checksums[name] != raw["checksum"]:
                    raise BaselineError("same workload/pass/seed checksum changed between repetitions")
                checksums[name] = raw["checksum"]
                destination[name].append({**raw, "round": round_index, "phase": phase,
                                          "context_before": snapshot_before,
                                          "context_after": context_snapshot(config.cpu)})
    result = []
    for workload in workloads:
        name = workload.name
        summary = summarize(samples[name])
        logical = samples[name][0]["logical_bytes_per_operation"]
        summary["logical_GB_per_s_at_median"] = logical / summary["median"] if logical else None
        result.append({"workload": asdict(workload), "passes": passes_by_name[name],
                       "calibration": calibration[name], "warmups": warmups[name],
                       "samples": samples[name], "summary": summary,
                       "samples_below_calibration_target": sum(sample["elapsed_ns"] < config.target_ns
                                                                for sample in samples[name])})
    return {"preregistered_order": orders, "workloads": result}


def collect(repo, config, do_measure):
    if sys.platform != "linux" or not hasattr(os, "sched_getaffinity"):
        raise BaselineError("collector requires Linux child CPU affinity")
    allowed = sorted(os.sched_getaffinity(0))
    cpu = config.cpu if config.cpu is not None else allowed[0]
    if cpu not in allowed:
        raise BaselineError("requested CPU outside current allowed affinity")
    config = Config(**{**asdict(config), "cpu": cpu})
    before_provenance = provenance(repo)
    machine = observe_machine(cpu)
    workloads = make_workloads(machine["l1_data_bytes"], machine["llc_bytes"], machine["mem_available_bytes"])
    result = {"protocol": "parseatlas_l0_collection_v1", "claim_class": "MEASURED" if do_measure else "OBSERVED",
              "scope": "native original add64; one or four limb harness composition; no Fp/Fn reduction",
              "config": asdict(config), "provenance": before_provenance, "machine": machine,
              "context_start": context_snapshot(cpu),
              "hardware_counters": {"status": "NOT_MEASURED",
                                    "reason": "collector does not collect PMU counters; no instruction/cache/DRAM traffic claim"},
              "limits": ["logical GB/s counts requested input/output record bytes, not measured memory traffic",
                         "latency includes dependency conditioning, loop and arithmetic overhead",
                         "cache-sized labels use observed capacity, not proof of cache residency or DRAM saturation",
                         "finite differential correctness is OBSERVED, not full-domain equivalence or CT proof",
                         "native compiler and one selected CPU only; no cross-platform or novelty claim",
                         "explicit correctness corpus is separate from seeded benchmark initialization"]}
    # Retain compiler-generated artifacts for independent objdump/perf review.
    # The collector never writes source, documentation or result files.
    directory = tempfile.mkdtemp(prefix="parseatlas-l0-baseline-")
    binary, result["build"] = build(repo, directory)
    result["build"]["retained_binary_path"] = str(binary)
    result["correctness"] = check_reference(binary, config)
    result["benchmark_kernel_smoke"] = []
    for limbs in (1, 4):
        for mode in ("latency", "bulk"):
            workload = Workload(f"smoke_{limbs}_{mode}", limbs, mode, 7, "untimed_claim_smoke")
            raw = invoke_sample(binary, workload, 3, config)
            expected = expected_small_checksum(limbs, mode, 7, 3, config.seed)
            if raw["checksum"] != expected:
                raise BaselineError("benchmark kernel bigint checksum disagreement")
            result["benchmark_kernel_smoke"].append({"raw": raw, "expected_checksum": expected,
                                                     "claim_class": "OBSERVED"})
    if do_measure:
        result["measurement"] = measure(binary, workloads, config)
    else:
        result["planned_workloads"] = [asdict(workload) for workload in workloads]
    if provenance(repo) != before_provenance:
        raise BaselineError("source provenance changed during collection")
    result["context_end"] = context_snapshot(cpu)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="compile and verify only; no sustained benchmark")
    mode.add_argument("--measure", action="store_true", help="run calibrated serial benchmark in a quiet window")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--random-cases", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--target-ms", type=int, default=50)
    args = parser.parse_args(argv)
    config = Config(args.seed, args.random_cases, args.warmups, args.repetitions,
                    args.target_ms * 1_000_000, args.cpu)
    repo = Path(__file__).resolve().parents[2]
    result = collect(repo, config, args.measure)
    print(canonical_json(result))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (BaselineError, OSError, subprocess.SubprocessError) as error:
        print(f"baseline collection failed: {error}", file=sys.stderr)
        raise SystemExit(2)
