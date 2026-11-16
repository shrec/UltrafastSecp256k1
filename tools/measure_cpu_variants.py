#!/usr/bin/env python3
r"""Measure the CPU dispatch variants of bench_unified on THIS machine.

One command, Linux or Windows, from anywhere inside the repository:

    python3 tools/measure_cpu_variants.py            (Linux)
    py -3 tools\measure_cpu_variants.py              (Windows, VS developer prompt)

The kit
  1. configures and builds with a CMake preset (cpu-release on Linux,
     windows-clang-cl on Windows; --preset overrides) - targets bench_unified
     and cpu_hwinfo;
  2. runs `cpu_hwinfo --json` to learn the raw CPU features and which lane
     backends this build offers (a tree without lanes is handled);
  3. plans the variants: default; no_bmi2_adx (UFSECP_CPU_DISABLE=bmi2,adx);
     lanes_off (UFSECP_LANES=off) and one lanes_<backend> per available lane
     backend;
  4. runs `bench_unified --json` for every variant, --rounds rounds, rotating
     the variant order each round so drift does not favour one variant, pinned
     to one core at raised priority (taskset + nice on Linux; a high-priority
     process that inherits a one-core affinity mask on Windows);
  5. writes docs/bench_variants_<YYYY-MM-DD>_<cpu-slug>.json - never over an
     existing file - with the CPU, microcode, OS, compiler, binary sha256, the
     cpu_hwinfo JSON, every per-round value, median/min/max and a per-op
     verdict against `default` by the repository rule (lower is better):
        improvement   only if  new_max < base_min
        regression    only if  new_min > base_max
        inconclusive  otherwise (an inconclusive result is not a green light).

--self-test checks the ordering, verdict and JSON-schema logic on synthetic
bench output. It builds nothing and does no CPU work.

A variant becomes a default only from such a measurement on real hardware; see
docs/MEASUREMENT_KIT.md. Python 3.8+, standard library only.
"""

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "ufsecp.bench_variants/1"
GENERATED_BY = "tools/measure_cpu_variants.py"

# Repository performance protocol (CLAUDE.md): at least 5 runs before any claim,
# and a baseline whose run-to-run spread is under 3 percent.
MIN_ROUNDS_FOR_VERDICT = 5
BASELINE_SPREAD_LIMIT_PCT = 3.0

MASK_ENV = "UFSECP_CPU_DISABLE"
LANES_ENV = "UFSECP_LANES"
SCRUBBED_ENV = (MASK_ENV, LANES_ENV)

TARGETS = ("bench_unified", "cpu_hwinfo")
VERDICTS = ("improvement", "regression", "inconclusive", "insufficient_rounds", "not_applicable")
KNOWN_LANES = ("avx2", "avx_ifma", "ifma256", "ifma512")
NOT_A_BACKEND = frozenset(
    ("off", "none", "auto", "default", "scalar", "portable", "disabled", "true", "false", "null")
)
MAX_VARIANT_FAILURES = 2  # a non-default variant that fails this often is dropped
WIN_HIGH_PRIORITY_CLASS = 0x00000080  # Win32 value; subprocess exports it only on Windows

RULE = {
    "base": "default",
    "direction": "lower is better (ns per op; ms for the *_ms rows)",
    "improvement": "new_max < base_min",
    "regression": "new_min > base_max",
    "otherwise": "inconclusive",
    "min_rounds": MIN_ROUNDS_FOR_VERDICT,
    "baseline_spread_limit_pct": BASELINE_SPREAD_LIMIT_PCT,
    "spread_pct": "(max - min) / median * 100 over the rounds of one variant",
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def log(message):
    print(message, file=sys.stderr, flush=True)


def fmt_duration(seconds):
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return "%dh%02dm%02ds" % (hours, minutes, secs)
    return "%dm%02ds" % (minutes, secs)


def tail(text, limit=600):
    return (text or "").strip()[-limit:]


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def repo_root():
    return Path(__file__).resolve().parent.parent


def display_path(path, root):
    """Repo-relative POSIX path when the file is inside the repo, else the path as given."""
    try:
        return Path(path).resolve().relative_to(Path(root).resolve()).as_posix()
    except (ValueError, OSError):
        return str(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scrubbed_env(environ):
    """The environment with the two dispatch overrides removed: each variant sets its own."""
    drop = set(SCRUBBED_ENV)
    return {key: value for key, value in environ.items() if key.upper() not in drop}


# ----------------------------------------------------------------------------
# CMake presets, build, binaries
# ----------------------------------------------------------------------------


def default_preset(system):
    return "windows-clang-cl" if system == "Windows" else "cpu-release"


def read_presets(root):
    """Configure presets by name from CMakePresets.json and CMakeUserPresets.json."""
    found = {}
    for filename in ("CMakePresets.json", "CMakeUserPresets.json"):
        path = Path(root) / filename
        if not path.is_file():
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for preset in doc.get("configurePresets", []):
            if isinstance(preset, dict) and "name" in preset:
                found[preset["name"]] = preset
    return found


def effective_preset(presets, name, _path=()):
    """A preset with `inherits` applied: earlier parents win over later ones, own fields win over all."""
    if name in _path:
        raise ValueError("preset inheritance cycle at %r" % name)
    preset = presets.get(name)
    if preset is None:
        raise KeyError(name)
    parents = preset.get("inherits", [])
    if isinstance(parents, str):
        parents = [parents]
    merged = {}
    cache = {}
    for parent in reversed(parents):
        inherited = effective_preset(presets, parent, _path + (name,))
        cache.update(inherited.get("cacheVariables", {}))
        merged.update({k: v for k, v in inherited.items() if k != "cacheVariables"})
    cache.update(preset.get("cacheVariables", {}))
    merged.update({k: v for k, v in preset.items() if k != "cacheVariables"})
    merged["cacheVariables"] = cache
    return merged


def cache_value(entry):
    if isinstance(entry, dict):
        entry = entry.get("value", "")
    if isinstance(entry, bool):
        return "ON" if entry else "OFF"
    return str(entry)


def preset_binary_dir(eff, name, root):
    raw = eff.get("binaryDir")
    if not raw:
        return Path(root) / "out" / name
    raw = str(raw).replace("${sourceDir}", Path(root).as_posix()).replace("${presetName}", name)
    if "${" in raw:  # a macro this script does not know: fall back to the repo convention
        return Path(root) / "out" / name
    path = Path(raw)
    return path if path.is_absolute() else Path(root) / path


def missing_tools(eff, which=shutil.which):
    """Tools the preset needs on PATH to configure and build, and that are not there."""
    need = ["cmake"]
    if str(eff.get("generator", "")).lower().startswith("ninja"):
        need.append("ninja")
    cache = eff.get("cacheVariables", {})
    for var in ("CMAKE_CXX_COMPILER", "CMAKE_C_COMPILER"):
        compiler = cache_value(cache.get(var, ""))
        if compiler and not os.path.isabs(compiler):
            need.append(compiler)
    return [tool for tool in dict.fromkeys(need) if which(tool) is None]


def run_checked(cmd, cwd):
    log("+ " + " ".join(cmd))
    code = subprocess.call(cmd, cwd=str(cwd))
    if code != 0:
        sys.exit("error: command failed (exit %d): %s" % (code, " ".join(cmd)))


def find_binary(build_dir, stem, system):
    """Newest executable called `stem` anywhere under the build directory."""
    name = stem + ".exe" if system == "Windows" else stem
    hits = [
        path
        for path in Path(build_dir).rglob(name)
        if path.is_file() and (system == "Windows" or os.access(str(path), os.X_OK))
    ]
    if not hits:
        sys.exit(
            "error: %s not found under %s (build it; the preset must enable "
            "SECP256K1_BUILD_TESTS and SECP256K1_BUILD_BENCH)" % (name, build_dir)
        )
    hits.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return hits[0]


# ----------------------------------------------------------------------------
# cpu_hwinfo: features and lane backends
# ----------------------------------------------------------------------------


def run_hwinfo(binary, env):
    try:
        proc = subprocess.run(
            [str(binary), "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            timeout=120,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("cannot run %s: %s" % (binary, exc))
    if proc.returncode != 0:
        raise RuntimeError("cpu_hwinfo --json exited %d: %s" % (proc.returncode, tail(proc.stderr, 300)))
    try:
        doc = json.loads(proc.stdout)
    except ValueError as exc:
        raise RuntimeError("cpu_hwinfo --json printed no JSON: %s" % exc)
    if not isinstance(doc, dict):
        raise RuntimeError("cpu_hwinfo --json did not print a JSON object")
    return doc


def feature_usable(hwinfo, name):
    """True when cpu_hwinfo reports the feature as usable (after the mask)."""
    entry = hwinfo.get("features", {}).get(name)
    if isinstance(entry, dict):
        return bool(entry.get("masked", entry.get("raw", False)))
    return bool(entry)


def _norm(name):
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


_LIST_KEYS = ("available", "backends", "supported", "usable", "compiled", "built")
_META_KEYS = frozenset(
    ("selected", "active", "chosen", "default", "requested", "current", "env", "reason", "count", "width", "note", "source")
)


def _entry_available(entry):
    for key in ("available", "usable", "supported", "enabled", "raw", "compiled", "built"):
        if key in entry:
            return bool(entry[key])
    return True


def _lane_names(value):
    """Names marked available inside one `lanes` value, whatever its shape.

    Accepts a comma/space separated string, a list of names, a list of
    {"name": .., "available": ..} objects, a {name: bool} map, a {name: {..}}
    map, or an object holding such a list/map under available/backends/...
    """
    names = []
    if isinstance(value, str):
        names.extend(_norm(part) for part in re.split(r"[,;|\s]+", value) if part)
    elif isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                label = item.get("name", item.get("backend", item.get("id")))
                if label is not None and _entry_available(item):
                    names.append(_norm(label))
            else:
                names.extend(_lane_names(item))
    elif isinstance(value, dict):
        listed = [key for key in value if _norm(key) in _LIST_KEYS]
        if listed:
            for key in listed:
                names.extend(_lane_names(value[key]))
        else:
            for key, entry in value.items():
                if _norm(key) in _META_KEYS:
                    continue
                if isinstance(entry, bool):
                    if entry:
                        names.append(_norm(key))
                elif isinstance(entry, dict) and _entry_available(entry):
                    names.append(_norm(key))
    return [name for name in names if name]


def lane_backends(hwinfo):
    """Lane backends this build offers on this CPU, read from any `lane*` key of cpu_hwinfo --json.

    An empty list means the tree has no lane backends (or cpu_hwinfo does not
    list them): the kit then plans no lane variants. --lanes overrides this.
    """
    found = []

    def walk(node, depth):
        if depth > 4 or not isinstance(node, dict):
            return
        for key, value in node.items():
            if "lane" in str(key).lower():
                found.extend(_lane_names(value))
            elif isinstance(value, dict):
                walk(value, depth + 1)

    walk(hwinfo, 0)
    unique = []
    for name in found:
        if name in NOT_A_BACKEND or name in unique or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            continue
        unique.append(name)
    rank = {name: index for index, name in enumerate(KNOWN_LANES)}
    return sorted(unique, key=lambda name: (rank.get(name, len(rank)), name))


def plan_variants(hwinfo, lanes, only=None):
    """The variant list for this CPU and build, plus the variants that were left out and why."""
    variants = [
        {
            "name": "default",
            "env": {},
            "description": "library defaults: no mask, automatic lane selection",
        }
    ]
    skipped = []
    if feature_usable(hwinfo, "bmi2") or feature_usable(hwinfo, "adx"):
        variants.append(
            {
                "name": "no_bmi2_adx",
                "env": {MASK_ENV: "bmi2,adx"},
                "description": "BMI2 and ADX masked off: dispatchers take their non-BMI2/ADX path",
            }
        )
    else:
        skipped.append(
            {"name": "no_bmi2_adx", "reason": "cpu_hwinfo reports neither BMI2 nor ADX, so the mask changes nothing"}
        )
    if lanes:
        variants.append(
            {"name": "lanes_off", "env": {LANES_ENV: "off"}, "description": "lane backends disabled"}
        )
        for backend in lanes:
            variants.append(
                {
                    "name": "lanes_" + backend,
                    "env": {LANES_ENV: backend},
                    "description": "lane backend forced to " + backend,
                }
            )
    else:
        skipped.append(
            {
                "name": "lanes_*",
                "reason": "cpu_hwinfo lists no lane backends (none in this build, or none available on this CPU)",
            }
        )
    if only:
        known = set(variant["name"] for variant in variants)
        unknown = [name for name in only if name not in known]
        if unknown:
            sys.exit(
                "error: unknown variant(s) %s; this machine can run: %s"
                % (", ".join(unknown), ", ".join(sorted(known)))
            )
        variants = [variant for variant in variants if variant["name"] == "default" or variant["name"] in only]
    return variants, skipped


# ----------------------------------------------------------------------------
# Ordering, launching, measuring
# ----------------------------------------------------------------------------


def rotation_order(names, round_index):
    """Variant order for one round: the list rotated left by the round number.

    Over n consecutive rounds every variant takes every position once, so a
    slow drift (thermal, frequency, background load) cannot favour one variant.
    """
    if not names:
        return []
    shift = round_index % len(names)
    return list(names[shift:]) + list(names[:shift])


def resolve_core(requested, system):
    if requested is not None:
        return requested
    if system == "Linux" and hasattr(os, "sched_getaffinity"):
        allowed = sorted(os.sched_getaffinity(0))
        if allowed and 0 not in allowed:
            return allowed[0]
    return 0


def launch_plan(system, core, nice, euid=None, which=shutil.which, have_sched_affinity=None):
    """How one bench_unified process is pinned and prioritised on this OS (pure; unit-tested).

    Linux:   taskset -c <core> [nice -n <nice>]; nice below zero needs root.
    Windows: HIGH_PRIORITY_CLASS plus a one-core affinity mask that this process
             holds while it starts the child (the child inherits it).
    """
    if have_sched_affinity is None:
        have_sched_affinity = hasattr(os, "sched_setaffinity")
    plan = {
        "prefix": [],
        "creationflags": 0,
        "inherit_affinity_core": None,
        "os_affinity_core": None,
        "pinned": False,
        "priority": "default",
        "notes": [],
    }
    if system == "Linux":
        if which("taskset"):
            plan["prefix"] += ["taskset", "-c", str(core)]
            plan["pinned"] = True
        elif have_sched_affinity:
            plan["os_affinity_core"] = core
            plan["pinned"] = True
            plan["notes"].append("taskset not found: pinning with os.sched_setaffinity")
        else:
            plan["notes"].append("no way to pin to a core on this system")
        if nice and which("nice"):
            if nice < 0 and euid != 0:
                plan["notes"].append(
                    "not root: nice %d needs root, so the priority is not raised (run with sudo to raise it)" % nice
                )
            else:
                plan["prefix"] += ["nice", "-n", str(nice)]
                plan["priority"] = "nice %d" % nice
    elif system == "Windows":
        plan["creationflags"] = WIN_HIGH_PRIORITY_CLASS
        plan["inherit_affinity_core"] = core
        plan["pinned"] = True
        plan["priority"] = "HIGH_PRIORITY_CLASS"
    else:
        plan["notes"].append("no core pinning or priority raise is implemented for %s" % system)
    return plan


def win_set_self_affinity(mask):
    """Set this process's affinity mask (children inherit it); returns the previous mask, or None."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetProcessAffinityMask.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.GetProcessAffinityMask.restype = wintypes.BOOL
        kernel32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        process_mask, system_mask = ctypes.c_size_t(), ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(handle, ctypes.byref(process_mask), ctypes.byref(system_mask)):
            return None
        if not kernel32.SetProcessAffinityMask(handle, mask):
            return None
        return process_mask.value
    except (OSError, AttributeError, ValueError):
        return None


class BenchOutputError(ValueError):
    """bench_unified wrote something that is not a usable benchmark artifact."""


def parse_bench_json(text):
    """Parse one `bench_unified --json` file into {metadata, ops, invalid}.

    Rows with zero, negative or non-finite timings are dropped and listed under
    `invalid` (a corrupt benchmark row is not evidence); a file with no usable
    timing row at all is rejected.
    """
    try:
        doc = json.loads(text)
    except ValueError as exc:
        raise BenchOutputError("not JSON: %s" % exc)
    if not isinstance(doc, dict) or not isinstance(doc.get("metadata"), dict) or not isinstance(doc.get("results"), list):
        raise BenchOutputError("missing the 'metadata' object or the 'results' list")
    ops = {}
    invalid = []
    seen = {}
    for row in doc["results"]:
        if not isinstance(row, dict) or "section" not in row or "name" not in row:
            raise BenchOutputError("malformed result row: %r" % (row,))
        key = "%s/%s" % (row["section"], row["name"])
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            key = "%s#%d" % (key, seen[key])
        if "ns" in row:
            kind, raw, unit = "time", row["ns"], str(row.get("unit", "ns"))
        elif "ratio" in row:
            kind, raw, unit = "ratio", row["ratio"], "ratio"
        else:
            raise BenchOutputError("row %s has neither 'ns' nor 'ratio'" % key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            invalid.append({"op": key, "value": repr(raw), "reason": "not a number"})
            continue
        if not math.isfinite(value) or (kind == "time" and value <= 0.0) or (kind == "ratio" and value < 0.0):
            invalid.append({"op": key, "value": repr(raw), "reason": "non-finite or non-positive"})
            continue
        ops[key] = {
            "section": str(row["section"]),
            "name": str(row["name"]),
            "kind": kind,
            "unit": unit,
            "value": value,
        }
    if not any(op["kind"] == "time" for op in ops.values()):
        raise BenchOutputError("no usable timing row (%d rejected)" % len(invalid))
    return {"metadata": doc["metadata"], "ops": ops, "invalid": invalid}


def run_one(bench, variant, round_index, bench_args, base_env, plan, timeout, workdir):
    """One bench_unified run for one variant; returns what happened, never raises for a bad run."""
    out_json = Path(workdir) / ("run_%02d_%s.json" % (round_index + 1, variant["name"]))
    env = dict(base_env)
    env.update(variant["env"])
    cmd = list(plan["prefix"]) + [str(bench), "--json", str(out_json)] + list(bench_args)
    kwargs = {
        "env": env,
        "cwd": str(workdir),
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "encoding": "utf-8",
        "errors": "replace",
    }
    if plan["creationflags"]:
        kwargs["creationflags"] = plan["creationflags"]
    core = plan["os_affinity_core"]
    if core is not None:
        kwargs["preexec_fn"] = lambda: os.sched_setaffinity(0, {core})
    result = {
        "error": None,
        "timed_out": False,
        "returncode": None,
        "stdout": "",
        "stderr": "",
        "json_text": None,
        "seconds": 0.0,
        "pin_failed": False,
    }
    started = time.monotonic()
    previous_mask = None
    if plan["inherit_affinity_core"] is not None:
        previous_mask = win_set_self_affinity(1 << plan["inherit_affinity_core"])
        result["pin_failed"] = previous_mask is None
    try:
        proc = subprocess.Popen(cmd, **kwargs)
    except OSError as exc:
        result["error"] = "cannot start %s: %s" % (bench, exc)
        return result
    finally:
        if previous_mask is not None:
            win_set_self_affinity(previous_mask)
    try:
        result["stdout"], result["stderr"] = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        result["stdout"], result["stderr"] = proc.communicate()
        result["timed_out"] = True
    result["returncode"] = proc.returncode
    result["seconds"] = time.monotonic() - started
    if out_json.is_file():
        result["json_text"] = out_json.read_text(encoding="utf-8", errors="replace")
    return result


class _Abort(Exception):
    pass


def measure(bench, variants, rounds, bench_args, base_env, plan, timeout, workdir):
    """Run every variant for `rounds` rounds in rotating order; returns runs, failures, order, status."""
    runs = dict((variant["name"], []) for variant in variants)
    by_name = dict((variant["name"], variant) for variant in variants)
    active = [variant["name"] for variant in variants]
    failures = []
    failure_count = {}
    round_order = []
    status = "complete"
    total = rounds * len(variants)
    done = 0
    started = time.monotonic()
    pin_warned = False
    try:
        for round_index in range(rounds):
            order = rotation_order(active, round_index)
            round_order.append(order)
            log("round %d/%d: %s" % (round_index + 1, rounds, " > ".join(order)))
            for name in order:
                result = run_one(bench, by_name[name], round_index, bench_args, base_env, plan, timeout, workdir)
                done += 1
                if result["pin_failed"] and not pin_warned:
                    pin_warned = True
                    log("warning: could not set the one-core affinity mask; the run is not pinned")
                parsed = None
                problem = None
                if result["error"]:
                    problem = result["error"]
                elif result["timed_out"]:
                    problem = "timed out after %d s" % timeout
                elif result["returncode"] != 0:
                    problem = "exit code %s" % result["returncode"]
                elif result["json_text"] is None:
                    problem = "bench_unified wrote no JSON file"
                else:
                    try:
                        parsed = parse_bench_json(result["json_text"])
                    except BenchOutputError as exc:
                        problem = "unusable bench output: %s" % exc
                elapsed = time.monotonic() - started
                if problem is None:
                    runs[name].append(parsed)
                    eta = (total - done) * elapsed / done
                    log(
                        "  %-16s ok  %6.1fs   %d/%d   elapsed %s   eta %s"
                        % (name, result["seconds"], done, total, fmt_duration(elapsed), fmt_duration(eta))
                    )
                    continue
                failures.append(
                    {
                        "round": round_index + 1,
                        "variant": name,
                        "problem": problem,
                        "stderr_tail": tail(result["stderr"]),
                        "stdout_tail": tail(result["stdout"]),
                    }
                )
                log("  %-16s FAILED round %d: %s" % (name, round_index + 1, problem))
                if name == "default":
                    status = "aborted"
                    raise _Abort()
                failure_count[name] = failure_count.get(name, 0) + 1
                if failure_count[name] >= MAX_VARIANT_FAILURES:
                    active.remove(name)
                    failures.append(
                        {
                            "round": round_index + 1,
                            "variant": name,
                            "problem": "dropped from the later rounds after %d failures" % failure_count[name],
                            "stderr_tail": "",
                            "stdout_tail": "",
                        }
                    )
                    log("  %-16s dropped from the later rounds" % name)
    except KeyboardInterrupt:
        status = "interrupted"
        log("interrupted: keeping the runs finished so far")
    except _Abort:
        log("the default variant failed: nothing to compare against, stopping")
    return {"runs": runs, "failures": failures, "round_order": round_order, "status": status}


# ----------------------------------------------------------------------------
# Statistics, verdicts, the report
# ----------------------------------------------------------------------------


def summarize(values):
    median = statistics.median(values)
    low, high = min(values), max(values)
    spread = round((high - low) / median * 100.0, 4) if median else None
    return {
        "rounds": [float(value) for value in values],
        "n": len(values),
        "median": float(median),
        "min": float(low),
        "max": float(high),
        "spread_pct": spread,
    }


def judge(base, new, min_rounds=MIN_ROUNDS_FOR_VERDICT):
    """Repository rule, lower is better: improvement only if new_max < base_min,
    regression only if new_min > base_max, otherwise inconclusive."""
    if base["n"] < min_rounds or new["n"] < min_rounds:
        return "insufficient_rounds"
    if new["max"] < base["min"]:
        return "improvement"
    if new["min"] > base["max"]:
        return "regression"
    return "inconclusive"


def build_ops_table(runs, order):
    """Per op and variant: the rounds, median/min/max and (against `default`) the verdict."""
    identity = {}
    keys = []
    for name in order:
        for run in runs.get(name, []):
            for key, op in run["ops"].items():
                if key not in identity:
                    identity[key] = {
                        "section": op["section"],
                        "name": op["name"],
                        "kind": op["kind"],
                        "unit": op["unit"],
                    }
                    keys.append(key)
    table = {}
    for key in keys:
        entry = dict(identity[key])
        entry["variants"] = {}
        for name in order:
            values = [run["ops"][key]["value"] for run in runs.get(name, []) if key in run["ops"]]
            if values:
                entry["variants"][name] = summarize(values)
        base = entry["variants"].get("default")
        is_time = entry["kind"] == "time"
        for name, stat in entry["variants"].items():
            if name == "default":
                if is_time:
                    spread = stat["spread_pct"]
                    stat["baseline_noisy"] = spread is not None and spread > BASELINE_SPREAD_LIMIT_PCT
                continue
            if not is_time:
                stat["verdict"] = "not_applicable"
            elif base is None:
                stat["verdict"] = "insufficient_rounds"
            else:
                stat["verdict"] = judge(base, stat)
                if base["median"]:
                    stat["delta_median_pct"] = round((stat["median"] - base["median"]) / base["median"] * 100.0, 4)
        table[key] = entry
    return table


def verdict_counts(table, order):
    counts = dict((name, dict((verdict, 0) for verdict in VERDICTS)) for name in order if name != "default")
    for entry in table.values():
        for name, stat in entry["variants"].items():
            if name != "default" and stat.get("verdict") in VERDICTS:
                counts[name][stat["verdict"]] += 1
    return counts


def build_report(host, variants, skipped, runs, round_order, failures, rounds_requested, status):
    """The canonical JSON document: host description plus the measured per-op tables."""
    order = [variant["name"] for variant in variants]
    table = build_ops_table(runs, order)
    default_runs = runs.get("default", [])
    first_meta = default_runs[0]["metadata"] if default_runs else {}
    varying = sorted(
        set(
            key
            for name in order
            for run in runs.get(name, [])
            for key, value in run["metadata"].items()
            if key != "date" and first_meta.get(key) != value
        )
    )
    invalid_rows = []
    for name in order:
        for index, run in enumerate(runs.get(name, [])):
            for row in run["invalid"]:
                invalid_rows.append(dict(row, variant=name, run=index + 1))
    variant_entries = []
    for variant in variants:
        entry = {
            "name": variant["name"],
            "description": variant["description"],
            "env": dict(variant["env"]),
            "rounds_completed": len(runs.get(variant["name"], [])),
        }
        if "cpu_hwinfo" in variant:
            entry["cpu_hwinfo"] = variant["cpu_hwinfo"]
        variant_entries.append(entry)
    time_ops_with_baseline = [
        entry for entry in table.values() if entry["kind"] == "time" and "default" in entry["variants"]
    ]
    noisy = [entry for entry in time_ops_with_baseline if entry["variants"]["default"].get("baseline_noisy")]
    if status == "complete" and not all(len(runs.get(name, [])) == rounds_requested for name in order):
        status = "partial"  # every round ran, but a variant failed or was dropped in some of them
    complete = status == "complete"
    doc = {"schema": SCHEMA, "generated_by": GENERATED_BY}
    doc.update(host)
    doc.update(
        {
            "status": status,
            "complete": complete,
            "rounds_requested": rounds_requested,
            "variants": variant_entries,
            "skipped_variants": skipped,
            "round_order": round_order,
            "rule": dict(RULE),
            "bench_metadata": first_meta,
            "bench_metadata_varies": varying,
            "invalid_rows": invalid_rows,
            "failures": failures,
            "ops": table,
            "summary": {
                "verdicts": verdict_counts(table, order),
                "baseline_time_ops": len(time_ops_with_baseline),
                "baseline_noisy_ops": len(noisy),
            },
        }
    )
    return doc


_REQUIRED = (
    ("generated_at_utc", str),
    ("status", str),
    ("complete", bool),
    ("rounds_requested", int),
    ("invocation", dict),
    ("cpu", dict),
    ("os", dict),
    ("power", dict),
    ("git", dict),
    ("build", dict),
    ("binaries", dict),
    ("cpu_hwinfo", dict),
    ("lane_backends", dict),
    ("pinning", dict),
    ("variants", list),
    ("skipped_variants", list),
    ("round_order", list),
    ("rule", dict),
    ("bench_metadata", dict),
    ("bench_metadata_varies", list),
    ("invalid_rows", list),
    ("failures", list),
    ("ops", dict),
    ("summary", dict),
)


def validate_report(doc):
    """Schema check of the canonical document; returns a list of problems (empty = valid)."""
    problems = []

    def need(condition, message):
        if not condition:
            problems.append(message)

    if not isinstance(doc, dict):
        return ["the report is not a JSON object"]
    need(doc.get("schema") == SCHEMA, "schema must be %r" % SCHEMA)
    need(doc.get("generated_by") == GENERATED_BY, "generated_by must be %r" % GENERATED_BY)
    need(
        doc.get("status") in ("complete", "partial", "interrupted", "aborted"),
        "status must be complete, partial, interrupted or aborted",
    )
    need((doc.get("status") == "complete") == (doc.get("complete") is True), "complete must be true exactly when status is complete")
    for key, kind in _REQUIRED:
        need(isinstance(doc.get(key), kind), "%s is missing or is not a %s" % (key, kind.__name__))
    if problems:
        return problems
    for key in ("brand", "vendor", "family", "model", "stepping", "microcode", "slug"):
        need(key in doc["cpu"], "cpu.%s is missing" % key)
    for key in ("system", "release", "machine"):
        need(key in doc["os"], "os.%s is missing" % key)
    need(isinstance(doc["build"].get("compiler"), dict), "build.compiler is missing")
    for binary in ("bench_unified", "cpu_hwinfo"):
        info = doc["binaries"].get(binary)
        digest_ok = isinstance(info, dict) and re.fullmatch(r"[0-9a-f]{64}", str(info.get("sha256", ""))) is not None
        need(digest_ok, "binaries.%s.sha256 must be 64 hex digits" % binary)
    names = [variant.get("name") for variant in doc["variants"] if isinstance(variant, dict)]
    need(len(names) == len(doc["variants"]) and len(set(names)) == len(names), "variant names must be unique")
    need(bool(names) and names[0] == "default", "the first variant must be 'default'")
    for variant in doc["variants"]:
        for key in ("name", "description", "env", "rounds_completed"):
            need(isinstance(variant, dict) and key in variant, "variant entry lacks %s" % key)
    for order in doc["round_order"]:
        need(isinstance(order, list) and set(order) <= set(names), "round_order names an unknown variant")
    for key, entry in doc["ops"].items():
        need(isinstance(entry, dict), "op %s is not an object" % key)
        if not isinstance(entry, dict):
            continue
        need(entry.get("kind") in ("time", "ratio"), "op %s: kind must be time or ratio" % key)
        for field in ("section", "name", "unit"):
            need(field in entry, "op %s lacks %s" % (key, field))
        stats = entry.get("variants")
        need(isinstance(stats, dict) and bool(stats), "op %s has no variant data" % key)
        if not isinstance(stats, dict):
            continue
        for name, stat in stats.items():
            where = "op %s / %s" % (key, name)
            need(name in names, "%s: unknown variant" % where)
            rounds = stat.get("rounds") if isinstance(stat, dict) else None
            rounds_ok = isinstance(rounds, list) and bool(rounds) and all(
                isinstance(value, (int, float)) and math.isfinite(value) for value in rounds
            )
            if not rounds_ok:
                problems.append("%s: rounds must be a non-empty list of finite numbers" % where)
                continue
            low, mid, high = stat.get("min"), stat.get("median"), stat.get("max")
            numeric = all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in (low, mid, high))
            need(stat.get("n") == len(rounds), "%s: n must equal the number of rounds" % where)
            need(numeric and low == min(rounds) and high == max(rounds), "%s: min/max do not match the rounds" % where)
            need(numeric and low <= mid <= high, "%s: min <= median <= max violated" % where)
            if name == "default":
                need("verdict" not in stat, "%s: the baseline carries no verdict" % where)
            elif entry.get("kind") == "time" and "default" in stats:
                need(stat.get("verdict") in VERDICTS, "%s: verdict must be one of %s" % (where, ", ".join(VERDICTS)))
            elif entry.get("kind") == "ratio":
                need(stat.get("verdict") == "not_applicable", "%s: ratio rows get not_applicable" % where)
    verdicts = doc["summary"].get("verdicts")
    need(isinstance(verdicts, dict), "summary.verdicts is missing")
    if isinstance(verdicts, dict):
        for name, counts in verdicts.items():
            need(name in names and name != "default", "summary.verdicts names %s" % name)
            need(isinstance(counts, dict) and set(counts) == set(VERDICTS), "summary.verdicts.%s needs every verdict" % name)
    return problems


def write_new_json(out_dir, stem, doc):
    """Write the document to <stem>.json, or <stem>_2.json, ... - an existing file is never touched."""
    text = json.dumps(doc, indent=2, ensure_ascii=True) + "\n"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 1000):
        path = out_dir / ("%s.json" % stem if attempt == 1 else "%s_%d.json" % (stem, attempt))
        try:
            with open(str(path), "x", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
        except FileExistsError:
            continue
        return path
    raise RuntimeError("no free file name for %s in %s" % (stem, out_dir))


# ----------------------------------------------------------------------------
# What the machine and the build are
# ----------------------------------------------------------------------------


def cpu_slug(brand, vendor="", family=0, model=0, stepping=0):
    text = str(brand or "")
    text = re.sub(r"\((?:r|tm|c)\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"@\s*[\d.]+\s*[gm]hz", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwith\s+radeon\b.*$", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\b(?:cpu|processor)\b", " ", text, flags=re.IGNORECASE)
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    if not slug:
        fallback = "%s %s %s %s" % (vendor or "cpu", family, model, stepping)
        slug = re.sub(r"[^a-z0-9]+", "-", fallback.lower()).strip("-")
    return slug[:64].strip("-") or "cpu"


def parse_proc_cpuinfo_microcode(text):
    for line in text.splitlines():
        if line.lower().startswith("microcode"):
            value = line.partition(":")[2].strip()
            if value:
                return value
    return None


def decode_update_revision(raw):
    """Windows 'Update Revision': two little-endian DWORDs; Intel keeps the revision in the high one, AMD in the low one."""
    if len(raw) < 4:
        return None
    low = int.from_bytes(raw[0:4], "little")
    high = int.from_bytes(raw[4:8], "little") if len(raw) >= 8 else 0
    value = high or low
    return "0x%x" % value if value else None


def read_microcode(system):
    """(value, source); the value is 'unknown' when it cannot be read."""
    try:
        if system == "Linux":
            value = parse_proc_cpuinfo_microcode(read_text("/proc/cpuinfo") or "")
            if value:
                return value, "/proc/cpuinfo"
        elif system == "Windows":
            import winreg

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                raw, _ = winreg.QueryValueEx(key, "Update Revision")
            raw = bytes(raw)
            value = decode_update_revision(raw)
            if value:
                return value, "windows registry CentralProcessor\\0 'Update Revision' (raw %s)" % raw.hex()
    except (OSError, ImportError, ValueError, TypeError):
        pass
    return "unknown", "unreadable"


def os_info(system):
    info = {
        "system": system,
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }
    if system == "Linux":
        for line in (read_text("/etc/os-release") or "").splitlines():
            if line.startswith("PRETTY_NAME="):
                info["distribution"] = line.split("=", 1)[1].strip().strip('"')
    elif system == "Windows":
        try:
            info["edition"] = platform.win32_edition()
        except AttributeError:
            pass
    return info


def power_info(system, core):
    """Best effort: AC power, governor/power plan and turbo state. Never raises."""
    info = {"ac_power": "unknown", "governor": "unknown", "power_plan": "unknown", "turbo": "unknown"}
    try:
        if system == "Linux":
            governor = read_text("/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor" % core)
            info["governor"] = governor or "unknown"
            no_turbo = read_text("/sys/devices/system/cpu/intel_pstate/no_turbo")
            boost = read_text("/sys/devices/system/cpu/cpufreq/boost")
            if no_turbo in ("0", "1"):
                info["turbo"] = "off" if no_turbo == "1" else "on"
            elif boost in ("0", "1"):
                info["turbo"] = "on" if boost == "1" else "off"
            mains = []
            supply = Path("/sys/class/power_supply")
            if supply.is_dir():
                for entry in sorted(supply.iterdir()):
                    if read_text(entry / "type") == "Mains":
                        mains.append(read_text(entry / "online") == "1")
            if mains:
                info["ac_power"] = any(mains)
        elif system == "Windows":
            import ctypes

            class _PowerStatus(ctypes.Structure):
                _fields_ = [
                    ("ACLineStatus", ctypes.c_ubyte),
                    ("BatteryFlag", ctypes.c_ubyte),
                    ("BatteryLifePercent", ctypes.c_ubyte),
                    ("SystemStatusFlag", ctypes.c_ubyte),
                    ("BatteryLifeTime", ctypes.c_ulong),
                    ("BatteryFullLifeTime", ctypes.c_ulong),
                ]

            status = _PowerStatus()
            if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
                if status.ACLineStatus in (0, 1):
                    info["ac_power"] = status.ACLineStatus == 1
            proc = subprocess.run(
                ["powercfg", "/getactivescheme"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=15,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode == 0 and proc.stdout.strip():
                info["power_plan"] = proc.stdout.strip()
    except Exception:  # best effort only: a failed probe must never stop a measurement
        pass
    return info


def git_info(root):
    try:
        head = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        dirty = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            encoding="utf-8",
            errors="replace",
        )
        if head.returncode == 0 and dirty.returncode == 0:
            return {"commit": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        pass
    return {"commit": "unknown", "dirty": None}


def read_cmake_cache(path):
    values = {}
    for line in (read_text(path) or "").splitlines():
        if not line or line.startswith(("//", "#")):
            continue
        match = re.match(r"([^:=]+):[A-Za-z]+=(.*)$", line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


def parse_compiler_module(text):
    """id/version/frontend/path of the C++ compiler from CMakeFiles/<ver>/CMakeCXXCompiler.cmake."""

    def grab(variable):
        match = re.search(r'set\(%s\s+"([^"]*)"\)' % re.escape(variable), text)
        return match.group(1) if match else None

    return {
        "id": grab("CMAKE_CXX_COMPILER_ID"),
        "version": grab("CMAKE_CXX_COMPILER_VERSION"),
        "frontend": grab("CMAKE_CXX_COMPILER_FRONTEND_VARIANT"),
        "path": grab("CMAKE_CXX_COMPILER"),
    }


_CACHE_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_GENERATOR",
    "CMAKE_CXX_COMPILER",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_FLAGS",
    "CMAKE_CXX_FLAGS_RELEASE",
    "SECP256K1_MARCH",
    "SECP256K1_SPEED_FIRST",
    "SECP256K1_USE_LTO",
    "SECP256K1_USE_ASM",
    "SECP256K1_ENABLE_OPENMP",
)


def build_info(build_dir, preset):
    """Compiler id/version (CMakeFiles/<ver>/CMakeCXXCompiler.cmake) and the cache entries that shape the code."""
    cache = read_cmake_cache(Path(build_dir) / "CMakeCache.txt")
    modules = sorted(
        Path(build_dir).glob("CMakeFiles/*/CMakeCXXCompiler.cmake"), key=lambda path: path.stat().st_mtime
    )
    found = parse_compiler_module(read_text(modules[-1]) or "") if modules else {}
    compiler = {
        "id": found.get("id") or "unknown",
        "version": found.get("version") or "unknown",
        "frontend": found.get("frontend") or "",
        "path": found.get("path") or cache.get("CMAKE_CXX_COMPILER") or "unknown",
    }
    return {
        "preset": preset,
        "generator": cache.get("CMAKE_GENERATOR", "unknown"),
        "build_type": cache.get("CMAKE_BUILD_TYPE", "unknown"),
        "compiler": compiler,
        "cache": dict((key, cache[key]) for key in _CACHE_KEYS if key in cache),
    }


# ----------------------------------------------------------------------------
# The command
# ----------------------------------------------------------------------------


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Measure the CPU dispatch variants of bench_unified on this machine "
        "and write one canonical JSON (see docs/MEASUREMENT_KIT.md).",
    )
    parser.add_argument("--preset", help="CMake configure preset (default: cpu-release; windows-clang-cl on Windows)")
    parser.add_argument("--rounds", type=int, default=5, help="rounds per variant (default 5; fewer than 5 gives no verdicts)")
    parser.add_argument("--suite", choices=("core", "extended", "all"), help="passed to bench_unified --suite")
    parser.add_argument("--passes", type=int, help="passed to bench_unified --passes")
    parser.add_argument("--core", type=int, help="core to pin to (default 0; bench_unified also pins itself to core 0)")
    parser.add_argument("--nice", type=int, default=-20, help="Linux nice value; below 0 needs root (default -20)")
    parser.add_argument("--timeout", type=int, default=3600, help="seconds allowed for one bench_unified run (default 3600)")
    parser.add_argument("--jobs", type=int, help="parallel build jobs (default: the generator's choice)")
    parser.add_argument("--skip-build", action="store_true", help="measure the binaries already in the preset's build directory")
    parser.add_argument("--build-dir", help="build directory to measure; requires --skip-build")
    parser.add_argument("--variants", help="comma list restricting the variants (default is always kept)")
    parser.add_argument("--lanes", default="auto", help="auto (ask cpu_hwinfo), none, or a comma list of lane backends")
    parser.add_argument("--out-dir", help="directory for the JSON (default: docs/ of this repository)")
    parser.add_argument("--print-lanes", metavar="HWINFO_JSON", help="print the lane backends listed in a saved 'cpu_hwinfo --json' file and exit")
    parser.add_argument("--self-test", action="store_true", help="check the ordering, verdict and schema logic on synthetic data and exit")
    return parser.parse_args(argv)


def run_kit(args):
    system = platform.system()
    root = repo_root()
    if args.rounds < 1:
        sys.exit("error: --rounds must be at least 1")
    if args.passes is not None and args.passes < 3:
        sys.exit("error: --passes must be at least 3 (bench_unified raises lower values to 3)")
    if args.build_dir and not args.skip_build:
        sys.exit("error: --build-dir needs --skip-build (the preset decides where a build goes)")
    if args.rounds < MIN_ROUNDS_FOR_VERDICT:
        log(
            "warning: %d round(s) is below the %d the repository protocol requires: the JSON will carry "
            "no improvement/regression verdicts" % (args.rounds, MIN_ROUNDS_FOR_VERDICT)
        )

    preset = args.preset or default_preset(system)
    presets = read_presets(root)
    if preset not in presets:
        sys.exit("error: unknown preset %r; CMakePresets.json has: %s" % (preset, ", ".join(sorted(presets)) or "(none)"))
    try:
        eff = effective_preset(presets, preset)
    except ValueError as exc:
        sys.exit("error: %s" % exc)
    cache_vars = dict((key, cache_value(value)) for key, value in eff.get("cacheVariables", {}).items())
    for option in ("SECP256K1_BUILD_TESTS", "SECP256K1_BUILD_BENCH"):
        if cache_vars.get(option, "ON").upper() in ("OFF", "0", "NO", "FALSE"):
            sys.exit(
                "error: preset %s sets %s=OFF; the kit needs cpu_hwinfo (tests) and bench_unified (bench)"
                % (preset, option)
            )

    core = resolve_core(args.core, system)
    if system == "Linux" and hasattr(os, "sched_getaffinity") and core not in os.sched_getaffinity(0):
        sys.exit("error: core %d is not among the CPUs this process may use; pass --core" % core)

    build_dir = Path(args.build_dir).resolve() if args.build_dir else preset_binary_dir(eff, preset, root)
    if not args.skip_build:
        missing = missing_tools(eff)
        if missing:
            sys.exit(
                "error: not found on PATH: %s (see the prerequisites in docs/MEASUREMENT_KIT.md; on Windows "
                "use a Visual Studio developer prompt)" % ", ".join(missing)
            )
        if system == "Windows" and not os.environ.get("INCLUDE"):
            log("warning: INCLUDE is not set - this does not look like a Visual Studio developer prompt")
        run_checked(["cmake", "--preset", preset], root)
        build_cmd = ["cmake", "--build", str(build_dir), "--target"] + list(TARGETS)
        if args.jobs:
            build_cmd += ["--parallel", str(args.jobs)]
        run_checked(build_cmd, root)
    if not (build_dir / "CMakeCache.txt").is_file():
        sys.exit("error: %s has no CMakeCache.txt; build with the preset first (drop --skip-build)" % build_dir)

    bench = find_binary(build_dir, "bench_unified", system)
    hwinfo_bin = find_binary(build_dir, "cpu_hwinfo", system)

    base_env = scrubbed_env(os.environ)
    for var in SCRUBBED_ENV:
        if any(key.upper() == var for key in os.environ):
            log("note: %s is set in your environment; it is ignored (each variant sets its own)" % var)
    try:
        hwinfo = run_hwinfo(hwinfo_bin, base_env)
    except RuntimeError as exc:
        sys.exit("error: %s" % exc)

    lanes_option = (args.lanes or "auto").strip().lower()
    if lanes_option == "auto":
        lanes = lane_backends(hwinfo)
        lane_source = "cpu_hwinfo" if lanes else "none"
    elif lanes_option == "none":
        lanes, lane_source = [], "none"
    else:
        lanes = [n for n in (_norm(item) for item in lanes_option.split(",")) if n and n not in NOT_A_BACKEND]
        lane_source = "override"
    only = [item.strip() for item in args.variants.split(",") if item.strip()] if args.variants else None
    variants, skipped = plan_variants(hwinfo, lanes, only)
    for variant in variants:
        if variant["env"]:
            env = dict(base_env)
            env.update(variant["env"])
            try:
                variant["cpu_hwinfo"] = run_hwinfo(hwinfo_bin, env)
            except RuntimeError as exc:
                variant["cpu_hwinfo"] = {"error": str(exc)}

    log("variants (%d), %d round(s) each, rotating order:" % (len(variants), args.rounds))
    for variant in variants:
        env_text = " ".join("%s=%s" % item for item in sorted(variant["env"].items())) or "(no overrides)"
        log("  %-16s %s" % (variant["name"], env_text))
    for item in skipped:
        log("  skipped %s: %s" % (item["name"], item["reason"]))

    plan = launch_plan(system, core, args.nice, euid=os.geteuid() if hasattr(os, "geteuid") else None)
    for note in plan["notes"]:
        log("note: " + note)
    power = power_info(system, core)
    if power["ac_power"] is False:
        log("warning: the machine is on battery; plug it in and repeat for a usable measurement")

    bench_args = []
    if args.suite:
        bench_args += ["--suite", args.suite]
    if args.passes:
        bench_args += ["--passes", str(args.passes)]

    with tempfile.TemporaryDirectory(prefix="ufsecp_variants_") as workdir:
        outcome = measure(bench, variants, args.rounds, bench_args, base_env, plan, args.timeout, workdir)
    if not outcome["runs"].get("default"):
        sys.exit("error: no usable run of the default variant; nothing was written")

    brand = str(hwinfo.get("brand") or platform.processor() or "")
    microcode, microcode_source = read_microcode(system)
    now = datetime.datetime.now(datetime.timezone.utc)
    slug = cpu_slug(
        brand, hwinfo.get("vendor", ""), hwinfo.get("family", 0), hwinfo.get("model", 0), hwinfo.get("stepping", 0)
    )
    binaries = {}
    for label, path in (("bench_unified", bench), ("cpu_hwinfo", hwinfo_bin)):
        binaries[label] = {"path": display_path(path, root), "sha256": sha256_file(path), "size": path.stat().st_size}
    build = build_info(build_dir, preset)
    build["build_dir"] = display_path(build_dir, root)
    host = {
        "generated_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date": now.strftime("%Y-%m-%d"),
        "invocation": {
            "argv": sys.argv[1:],
            "preset": preset,
            "rounds": args.rounds,
            "suite": args.suite or "bench_unified default",
            "passes": args.passes if args.passes else "bench_unified default",
            "core": core,
            "skip_build": bool(args.skip_build),
            "timeout_s": args.timeout,
        },
        "cpu": {
            "brand": brand,
            "vendor": hwinfo.get("vendor", "unknown"),
            "family": hwinfo.get("family", 0),
            "model": hwinfo.get("model", 0),
            "stepping": hwinfo.get("stepping", 0),
            "microcode": microcode,
            "microcode_source": microcode_source,
            "slug": slug,
        },
        "os": os_info(system),
        "power": power,
        "git": git_info(root),
        "build": build,
        "binaries": binaries,
        "cpu_hwinfo": hwinfo,
        "lane_backends": {"discovered": lanes, "source": lane_source, "option": lanes_option},
        "pinning": {
            "core": core,
            "pinned": plan["pinned"],
            "priority": plan["priority"],
            "notes": plan["notes"],
            "bench_unified": "also pins its measuring thread to CPU 0 and raises its own priority "
            "(benchmark_harness.hpp pin_thread_and_elevate)",
        },
    }
    report = build_report(
        host, variants, skipped, outcome["runs"], outcome["round_order"], outcome["failures"], args.rounds, outcome["status"]
    )
    problems = validate_report(report)
    if problems:
        sys.exit("error: the report failed its own schema check, nothing was written:\n  " + "\n  ".join(problems))
    out_dir = Path(args.out_dir) if args.out_dir else root / "docs"
    path = write_new_json(out_dir, "bench_variants_%s_%s" % (host["date"], slug), report)

    print("wrote %s" % display_path(path, root))
    incomplete = " (INCOMPLETE: not every variant finished every round; see 'failures' in the JSON)"
    print("status: %s%s" % (report["status"], "" if report["complete"] else incomplete))
    counts = report["summary"]["verdicts"]
    if counts:
        print("%-16s %s" % ("variant", "  ".join(VERDICTS)))
        for name, row in counts.items():
            print("%-16s %s" % (name, "  ".join("%*d" % (len(verdict), row[verdict]) for verdict in VERDICTS)))
    noisy, total = report["summary"]["baseline_noisy_ops"], report["summary"]["baseline_time_ops"]
    if noisy:
        print(
            "warning: %d of %d baseline ops spread more than %.0f%% between rounds: the machine was not quiet "
            "enough; repeat before claiming anything" % (noisy, total, BASELINE_SPREAD_LIMIT_PCT)
        )
    print("send back the JSON file named above, unedited%s" % ("" if report["complete"] else ", with this console output"))
    return 0 if report["complete"] else 1


# ----------------------------------------------------------------------------
# Self-test: synthetic data only, no build, no CPU work
# ----------------------------------------------------------------------------


def _fake_bench_text(values, ratio=None):
    results = [{"section": "field", "name": name, "ns": value, "unit": "ns"} for name, value in values.items()]
    if ratio is not None:
        results.append({"section": "ratio", "name": "ultra/ref", "ratio": ratio})
    metadata = {
        "generated_by": "bench_unified --json",
        "date": "2000-01-01",
        "cpu": "Synthetic CPU",
        "compiler": "synthetic 1.0",
        "arch": "x86_64",
        "timer": "synthetic",
        "tsc_ghz": 1.0,
        "passes": 11,
        "warmup": 500,
        "pool_size": 64,
        "turbo": "unknown",
        "run_mode": "full",
    }
    return json.dumps({"metadata": metadata, "results": results})


def _fake_host(lanes):
    sha = "0123456789abcdef" * 4
    return {
        "generated_at_utc": "2000-01-01T00:00:00Z",
        "date": "2000-01-01",
        "invocation": {"argv": [], "preset": "cpu-release", "rounds": 5},
        "cpu": {
            "brand": "Synthetic CPU",
            "vendor": "Intel",
            "family": 6,
            "model": 1,
            "stepping": 1,
            "microcode": "0x1",
            "microcode_source": "synthetic",
            "slug": "synthetic-cpu",
        },
        "os": {"system": "Linux", "release": "0", "version": "0", "machine": "x86_64", "python": "3"},
        "power": {"ac_power": "unknown", "governor": "unknown", "power_plan": "unknown", "turbo": "unknown"},
        "git": {"commit": "0" * 40, "dirty": False},
        "build": {
            "preset": "cpu-release",
            "generator": "Ninja",
            "build_type": "Release",
            "compiler": {"id": "GNU", "version": "0.0.0", "frontend": "", "path": "/usr/bin/c++"},
            "cache": {},
        },
        "binaries": {
            "bench_unified": {"path": "out/x/bench_unified", "sha256": sha, "size": 1},
            "cpu_hwinfo": {"path": "out/x/cpu_hwinfo", "sha256": sha, "size": 1},
        },
        "cpu_hwinfo": {"vendor": "Intel", "features": {"bmi2": {"raw": True, "masked": True}}},
        "lane_backends": {"discovered": lanes, "source": "cpu_hwinfo", "option": "auto"},
        "pinning": {"core": 0, "pinned": True, "priority": "default", "notes": []},
    }


def self_test():
    failures = []
    passed = [0]

    def check(condition, message):
        if condition:
            passed[0] += 1
        else:
            failures.append(message)

    # -- verdict rule (lower is better); every number below is a synthetic fixture, not a measurement
    base = summarize([100, 101, 99, 100, 102])
    check(judge(base, summarize([80, 81, 79, 80, 82])) == "improvement", "disjoint lower range must be an improvement")
    check(judge(base, summarize([150, 151, 149, 150, 152])) == "regression", "disjoint higher range must be a regression")
    check(judge(base, summarize([99, 100, 101, 100, 98])) == "inconclusive", "overlapping ranges must be inconclusive")
    low = summarize([10, 11, 12, 13, 14])
    check(judge(low, summarize([5, 6, 7, 8, 10])) == "inconclusive", "new_max == base_min is not an improvement")
    check(judge(low, summarize([14, 15, 16, 17, 18])) == "inconclusive", "new_min == base_max is not a regression")
    check(judge(base, summarize([80, 81, 79])) == "insufficient_rounds", "fewer than 5 rounds gives no verdict")
    check(judge(summarize([100, 101, 99]), base) == "insufficient_rounds", "a short baseline gives no verdict")
    stat = summarize([3, 1, 2, 5, 4])
    check(stat["median"] == 3.0 and stat["min"] == 1.0 and stat["max"] == 5.0 and stat["n"] == 5, "median/min/max/n")
    check(stat["rounds"] == [3.0, 1.0, 2.0, 5.0, 4.0], "per-round values keep their order")
    check(abs(stat["spread_pct"] - 133.3333) < 0.001, "spread is (max-min)/median in percent")

    # -- rotation
    names = ["default", "no_bmi2_adx", "lanes_off", "lanes_avx2", "lanes_ifma256"]
    orders = [rotation_order(names, index) for index in range(5)]
    check(orders[0] == names, "round 1 starts with the planned order")
    check(all(sorted(order) == sorted(names) for order in orders), "every round runs every variant once")
    check(
        all(sorted(order[pos] for order in orders) == sorted(names) for pos in range(5)),
        "over n rounds every variant takes every position once",
    )
    check(all(orders[i] != orders[i + 1] for i in range(4)), "consecutive rounds differ")
    check(rotation_order(names, 5) == names, "the rotation wraps after n rounds")
    check(rotation_order(names[:2], 1) == ["no_bmi2_adx", "default"], "two variants alternate")
    check(rotation_order(["default"], 3) == ["default"] and rotation_order([], 2) == [], "degenerate lists")

    # -- bench output parsing
    good = parse_bench_json(_fake_bench_text({"field_mul": 100.0}, ratio=1.5))
    check(set(good["ops"]) == {"field/field_mul", "ratio/ultra/ref"}, "section/name keys")
    check(good["ops"]["field/field_mul"]["kind"] == "time" and good["ops"]["ratio/ultra/ref"]["kind"] == "ratio", "kinds")
    dropped = parse_bench_json(_fake_bench_text({"a": 10.0, "zero": 0.0, "nan": float("nan")}))
    check(set(dropped["ops"]) == {"field/a"} and len(dropped["invalid"]) == 2, "zero and NaN timings are dropped, not used")
    twin = parse_bench_json(
        json.dumps({"metadata": {}, "results": [{"section": "s", "name": "n", "ns": 1.0}, {"section": "s", "name": "n", "ns": 2.0}]})
    )
    check(set(twin["ops"]) == {"s/n", "s/n#2"}, "duplicate section/name rows stay distinct")
    for bad in (
        "not json",
        "{}",
        json.dumps({"metadata": {}, "results": []}),
        json.dumps({"metadata": {}, "results": [{"section": "s", "name": "n"}]}),
        json.dumps({"metadata": {}, "results": [{"section": "s", "name": "n", "ns": 0.0}]}),
    ):
        try:
            parse_bench_json(bad)
            check(False, "must reject %r" % bad)
        except BenchOutputError:
            check(True, "")

    # -- lane discovery in every shape cpu_hwinfo might print, and its absence
    shapes = [
        ({"lanes": ["avx2", "ifma256"]}, ["avx2", "ifma256"]),
        ({"lanes": {"available": ["ifma512", "avx2"], "selected": "ifma512"}}, ["avx2", "ifma512"]),
        ({"lanes": {"avx2": True, "ifma512": False, "avx_ifma": True}}, ["avx2", "avx_ifma"]),
        ({"dispatch": {"lanes": "avx_ifma, ifma256"}}, ["avx_ifma", "ifma256"]),
        ({"lane_backends": [{"name": "avx2", "available": True}, {"name": "ifma512", "available": False}]}, ["avx2"]),
        ({"lanes": {"backends": {"ifma256": {"available": True}, "avx2": {"available": False}}}}, ["ifma256"]),
        ({"lanes": {"selected": "off", "count": 4}}, []),
        ({"lanes": "off"}, []),
        ({"features": {"avx2": {"raw": True, "masked": True}}, "dispatch": {"field_mul": "portable"}}, []),
    ]
    for doc, expected in shapes:
        found = lane_backends(doc)
        check(found == expected, "lane_backends(%r) -> %r, got %r" % (doc, expected, found))

    # -- variant planning with and without lanes and BMI2/ADX
    full = {"features": {"bmi2": {"raw": True, "masked": True}, "adx": {"raw": True, "masked": True}}}
    plain, left_out = plan_variants(full, [])
    check([v["name"] for v in plain] == ["default", "no_bmi2_adx"], "no lanes: default and no_bmi2_adx only")
    check(plain[1]["env"] == {MASK_ENV: "bmi2,adx"} and plain[0]["env"] == {}, "mask variant sets UFSECP_CPU_DISABLE=bmi2,adx")
    check([item["name"] for item in left_out] == ["lanes_*"], "the missing lanes are reported")
    laned, _ = plan_variants(full, ["avx2", "ifma512"])
    check(
        [v["name"] for v in laned] == ["default", "no_bmi2_adx", "lanes_off", "lanes_avx2", "lanes_ifma512"],
        "lanes add lanes_off and one variant per backend",
    )
    check(laned[2]["env"] == {LANES_ENV: "off"} and laned[4]["env"] == {LANES_ENV: "ifma512"}, "lane variant environments")
    old_cpu, left_out = plan_variants({"features": {"bmi2": {"raw": False, "masked": False}}}, [])
    check(
        [v["name"] for v in old_cpu] == ["default"] and left_out[0]["name"] == "no_bmi2_adx",
        "no BMI2/ADX: the mask variant is skipped",
    )
    picked, _ = plan_variants(full, ["avx2"], only=["lanes_avx2"])
    check([v["name"] for v in picked] == ["default", "lanes_avx2"], "--variants keeps default and the named variants")

    # -- launching: pinning and priority per OS
    which_all = lambda tool: "/usr/bin/" + tool  # noqa: E731
    root_plan = launch_plan("Linux", 0, -20, euid=0, which=which_all)
    check(root_plan["prefix"] == ["taskset", "-c", "0", "nice", "-n", "-20"], "Linux root: taskset + nice -20")
    user_plan = launch_plan("Linux", 2, -20, euid=1000, which=which_all)
    check(
        user_plan["prefix"] == ["taskset", "-c", "2"] and any("root" in note for note in user_plan["notes"]),
        "Linux user: pinned, nice skipped with a note",
    )
    bare_plan = launch_plan("Linux", 1, 0, euid=1000, which=lambda tool: None, have_sched_affinity=True)
    check(bare_plan["prefix"] == [] and bare_plan["os_affinity_core"] == 1, "no taskset: os.sched_setaffinity fallback")
    win_plan = launch_plan("Windows", 3, -20, which=lambda tool: None)
    check(
        win_plan["creationflags"] == WIN_HIGH_PRIORITY_CLASS
        and win_plan["inherit_affinity_core"] == 3
        and win_plan["prefix"] == [],
        "Windows: high priority class + inherited one-core mask",
    )
    check(launch_plan("Darwin", 0, -20)["pinned"] is False, "an unsupported OS is reported as not pinned")

    # -- environment hygiene
    check(
        scrubbed_env({"PATH": "/bin", "UFSECP_LANES": "x", "ufsecp_cpu_disable": "y"}) == {"PATH": "/bin"},
        "the two overrides are scrubbed",
    )

    # -- presets
    presets = {
        "base": {
            "name": "base",
            "hidden": True,
            "binaryDir": "${sourceDir}/out/${presetName}",
            "generator": "Ninja",
            "cacheVariables": {"A": "1"},
        },
        "left": {"name": "left", "inherits": "base", "cacheVariables": {"A": "left", "B": "b"}},
        "right": {"name": "right", "cacheVariables": {"A": "right", "C": {"type": "STRING", "value": "c"}}},
        "leaf": {"name": "leaf", "inherits": ["left", "right"], "cacheVariables": {"D": "d"}},
        "loop": {"name": "loop", "inherits": "loop"},
    }
    eff = effective_preset(presets, "leaf")
    check(
        eff["cacheVariables"]["A"] == "left" and cache_value(eff["cacheVariables"]["C"]) == "c" and eff["cacheVariables"]["D"] == "d",
        "earlier parent wins, own fields are added",
    )
    check(eff["generator"] == "Ninja", "fields are inherited through the chain")
    check(preset_binary_dir(eff, "leaf", Path("/r")).as_posix() == "/r/out/leaf", "binaryDir macros are expanded")
    check(preset_binary_dir({}, "x", Path("/r")).as_posix() == "/r/out/x", "default build directory is out/<preset>")
    try:
        effective_preset(presets, "loop")
        check(False, "an inheritance cycle must be rejected")
    except ValueError:
        check(True, "")
    check(
        missing_tools({"generator": "Ninja", "cacheVariables": {"CMAKE_CXX_COMPILER": "clang-cl"}}, which=lambda tool: None)
        == ["cmake", "ninja", "clang-cl"],
        "the tools a preset needs are listed",
    )
    check(default_preset("Windows") == "windows-clang-cl" and default_preset("Linux") == "cpu-release", "default presets")

    # -- command line
    defaults = parse_args([])
    check(defaults.rounds == 5 and defaults.nice == -20 and defaults.lanes == "auto" and not defaults.skip_build, "CLI defaults")
    check(parse_args(["--rounds", "3", "--suite", "core", "--skip-build"]).rounds == 3, "CLI options")

    # -- host description helpers
    check(cpu_slug("11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz") == "11th-gen-intel-core-i7-1165g7", "slug of an Intel brand string")
    check(cpu_slug("AMD Ryzen 9 7950X 16-Core Processor") == "amd-ryzen-9-7950x-16-core", "slug of an AMD brand string")
    check(cpu_slug("", "Intel", 6, 140, 1) == "intel-6-140-1", "slug without a brand string")
    check(parse_proc_cpuinfo_microcode("model\t: 140\nmicrocode\t: 0xb4\n") == "0xb4", "Linux microcode from /proc/cpuinfo")
    check(parse_proc_cpuinfo_microcode("model\t: 140\n") is None, "no microcode line")
    check(decode_update_revision(bytes([0, 0, 0, 0, 0xB4, 0, 0, 0])) == "0xb4", "Windows Update Revision, Intel layout")
    check(decode_update_revision(bytes([0x0D, 0, 0x50, 0x0A, 0, 0, 0, 0])) == "0xa50000d", "Windows Update Revision, AMD layout")
    check(decode_update_revision(b"") is None, "empty Update Revision")
    module = (
        'set(CMAKE_CXX_COMPILER "/usr/bin/c++")\nset(CMAKE_CXX_COMPILER_ID "GNU")\n'
        'set(CMAKE_CXX_COMPILER_VERSION "13.2.0")\nset(CMAKE_CXX_COMPILER_FRONTEND_VARIANT "")\n'
    )
    compiler = parse_compiler_module(module)
    check(
        compiler["id"] == "GNU" and compiler["version"] == "13.2.0" and compiler["path"] == "/usr/bin/c++",
        "compiler id/version from CMakeCXXCompiler.cmake",
    )

    with tempfile.TemporaryDirectory(prefix="ufsecp_selftest_") as scratch:
        cache_file = Path(scratch) / "CMakeCache.txt"
        cache_file.write_text(
            "// comment\n# comment\nCMAKE_BUILD_TYPE:STRING=Release\nSECP256K1_MARCH:STRING=x86-64-v3\n", encoding="utf-8"
        )
        parsed = read_cmake_cache(cache_file)
        check(
            parsed.get("CMAKE_BUILD_TYPE") == "Release" and parsed.get("SECP256K1_MARCH") == "x86-64-v3",
            "CMakeCache.txt parsing",
        )
        module_dir = Path(scratch) / "CMakeFiles" / "3.99.0"
        module_dir.mkdir(parents=True)
        (module_dir / "CMakeCXXCompiler.cmake").write_text(module, encoding="utf-8")
        info = build_info(scratch, "cpu-release")
        check(
            info["compiler"]["id"] == "GNU" and info["compiler"]["version"] == "13.2.0" and info["build_type"] == "Release",
            "build_info reads the compiler and the cache",
        )

        # -- a full report from synthetic runs: tables, verdicts, schema, writer
        noisy_series = [100, 110, 90, 105, 95]
        fixtures = {
            "default": {
                "field_mul": [100, 101, 99, 100, 101],
                "field_sqr": [50, 50.5, 49.5, 50, 50.2],
                "ec_op": [900, 905, 895, 900, 902],
                "noisy_op": noisy_series,
            },
            "no_bmi2_adx": {
                "field_mul": [150, 151, 149, 150, 152],
                "field_sqr": [50, 50.4, 49.6, 50.1, 50.3],
                "ec_op": [900, 905, 895, 900, 902],
                "noisy_op": noisy_series,
            },
            "lanes_off": {
                "field_mul": [99, 100, 101, 100, 98],
                "field_sqr": [50, 50.2, 49.8, 50, 50.1],
                "ec_op": [900, 905, 895, 900, 902],
                "noisy_op": noisy_series,
            },
            "lanes_avx2": {
                "field_mul": [80, 81, 79, 80, 82],
                "field_sqr": [50, 50.3, 49.7, 50, 50.1],
                "ec_op": [900, 905, 895, 900, 902],
                "noisy_op": noisy_series,
            },
        }
        test_variants, test_skipped = plan_variants(full, ["avx2"])
        check([v["name"] for v in test_variants] == list(fixtures), "the fixture variants match the plan")
        runs = dict((name, []) for name in fixtures)
        for round_index in range(5):
            for name, series in fixtures.items():
                values = dict((op, values_by_round[round_index]) for op, values_by_round in series.items())
                if name == "lanes_avx2" and round_index == 4:
                    values.pop("ec_op")  # one round without this op: too few rounds for a verdict
                runs[name].append(parse_bench_json(_fake_bench_text(values, ratio=1.25)))
        order = [v["name"] for v in test_variants]
        round_order = [rotation_order(order, index) for index in range(5)]
        report = build_report(_fake_host(["avx2"]), test_variants, test_skipped, runs, round_order, [], 5, "complete")
        problems = validate_report(report)
        check(problems == [], "a well-formed report validates: %s" % problems)
        check(report["complete"] is True and report["status"] == "complete", "a finished run is flagged complete")
        check(json.loads(json.dumps(report)) == report, "the report survives a JSON round trip")
        mul = report["ops"]["field/field_mul"]["variants"]
        check(
            mul["no_bmi2_adx"]["verdict"] == "regression"
            and mul["lanes_avx2"]["verdict"] == "improvement"
            and mul["lanes_off"]["verdict"] == "inconclusive",
            "verdicts per variant",
        )
        check("verdict" not in mul["default"] and mul["default"]["baseline_noisy"] is False, "the baseline has no verdict and is quiet")
        check(report["ops"]["field/noisy_op"]["variants"]["default"]["baseline_noisy"] is True, "a baseline spread above the limit is flagged")
        check(report["summary"]["baseline_noisy_ops"] == 1, "the noisy baseline ops are counted")
        check(report["ops"]["field/ec_op"]["variants"]["lanes_avx2"]["verdict"] == "insufficient_rounds", "an op missing in a round gets no verdict")
        check(report["ops"]["ratio/ultra/ref"]["variants"]["lanes_avx2"]["verdict"] == "not_applicable", "ratio rows are not judged")
        verdicts = report["summary"]["verdicts"]
        check(verdicts["no_bmi2_adx"]["regression"] == 1 and verdicts["lanes_avx2"]["improvement"] == 1, "verdict counts")
        check(mul["no_bmi2_adx"]["rounds"] == [150.0, 151.0, 149.0, 150.0, 152.0], "per-round values are kept in the report")
        check(all(len(report["ops"][op]["variants"]["default"]["rounds"]) == 5 for op in report["ops"]), "five rounds recorded for the baseline")
        partial = build_report(_fake_host([]), test_variants, [], runs, round_order, [], 7, "complete")
        check(partial["complete"] is False and partial["status"] == "partial", "fewer rounds than requested is partial, not complete")
        check(validate_report(partial) == [], "a partial report is still a valid document")
        inconsistent = json.loads(json.dumps(report))
        inconsistent["complete"] = False
        check(validate_report(inconsistent) != [], "the validator catches complete disagreeing with status")

        broken = json.loads(json.dumps(report))
        broken["ops"]["field/field_mul"]["variants"]["lanes_off"]["median"] = 1e9
        check(validate_report(broken) != [], "the validator catches a median outside min/max")
        broken = json.loads(json.dumps(report))
        broken["ops"]["field/field_mul"]["variants"]["lanes_off"]["verdict"] = "great"
        check(validate_report(broken) != [], "the validator catches an unknown verdict")
        broken = json.loads(json.dumps(report))
        del broken["binaries"]
        check(validate_report(broken) != [], "the validator catches a missing section")
        broken = json.loads(json.dumps(report))
        broken["binaries"]["bench_unified"]["sha256"] = "xyz"
        check(validate_report(broken) != [], "the validator catches a bad sha256")
        broken = json.loads(json.dumps(report))
        broken["variants"][0]["name"] = "other"
        check(validate_report(broken) != [], "the validator catches a missing default variant")
        check(validate_report([]) != [] and validate_report({}) != [], "the validator rejects non-reports")

        # -- the writer never overwrites
        stem = "bench_variants_2000-01-01_synthetic-cpu"
        first = write_new_json(scratch, stem, {"n": 1})
        second = write_new_json(scratch, stem, {"n": 2})
        third = write_new_json(scratch, stem, {"n": 3})
        check(
            [first.name, second.name, third.name] == [stem + ".json", stem + "_2.json", stem + "_3.json"],
            "an existing file gets a numbered sibling",
        )
        check(json.loads(first.read_text(encoding="utf-8")) == {"n": 1}, "the first file is untouched")
        check(display_path(first, scratch) == first.name, "display_path is repo-relative inside the repo")

        # -- the --print-lanes entry point the SDE workflow uses
        hwinfo_file = Path(scratch) / "hwinfo.json"
        hwinfo_file.write_text(json.dumps({"lanes": {"available": ["avx_ifma", "avx2"]}}), encoding="utf-8")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            code = main(["--print-lanes", str(hwinfo_file)])
        check(code == 0 and captured.getvalue().split() == ["avx2", "avx_ifma"], "--print-lanes prints one backend per line")

    if failures:
        print("self-test FAILED (%d failed, %d passed):" % (len(failures), passed[0]))
        for message in failures:
            print("  - " + message)
        return 1
    print("self-test: %d checks passed" % passed[0])
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    if args.print_lanes:
        text = read_text(args.print_lanes)
        try:
            doc = json.loads(text) if text else None
        except ValueError:
            doc = None
        if not isinstance(doc, dict):
            sys.exit("error: %s is not a cpu_hwinfo --json file" % args.print_lanes)
        for name in lane_backends(doc):
            print(name)
        return 0
    return run_kit(args)


if __name__ == "__main__":
    sys.exit(main())
