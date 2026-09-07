"""Finite original-header observations and harness checks, not speed assertions."""

import json
from pathlib import Path
import random
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / "experiments/parseatlas_secp256k1/probes/l0_baseline.cpp"
MASK = (1 << 64) - 1


@pytest.fixture(scope="module", params=[False, True], ids=["native", "portable_carry"])
def binary(request, tmp_path_factory):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.fail("g++ is required; original-header correctness may not silently skip")
    target = tmp_path_factory.mktemp("l0_cpp") / "baseline"
    flags = ["-std=c++17", "-O3", "-DNDEBUG", "-march=native", "-Wall", "-Wextra", "-Werror"]
    if request.param:
        flags.append("-DSECP256K1_NO_INT128")
    result = subprocess.run(
        [compiler, *flags, "-I", str(ROOT / "src/cpu/include"), str(PROBE), "-o", str(target)],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    return target, request.param


def invoke(binary, *args, data=None):
    return subprocess.run(
        [str(binary[0]), *(str(arg) for arg in args)], input=data,
        capture_output=True, text=True, timeout=15, check=False,
    )


def limbs_of(value, limbs):
    return [(value >> (64 * i)) & MASK for i in range(limbs)]


def encode_case(limbs, a, b, carry):
    return " ".join([str(carry), *(f"{w:016x}" for w in limbs_of(a, limbs) + limbs_of(b, limbs))])


@pytest.mark.parametrize("limbs", [1, 4])
def test_original_header_matches_independent_bigint(binary, limbs):
    modulus = 1 << (64 * limbs)
    values = {0, 1, 2, modulus - 1, modulus - 2, modulus // 2, modulus // 2 - 1}
    for bit in range(64, 64 * limbs, 64):
        values.update({(1 << bit) - 1, 1 << bit, (1 << bit) + 1})
    cases = [(a, b, c) for a in sorted(values) for b in sorted(values) for c in (0, 1)]
    rng = random.Random(0xA71A5 + limbs)
    cases.extend((rng.randrange(modulus), rng.randrange(modulus), rng.randrange(2)) for _ in range(300))
    result = invoke(binary, "--check", limbs,
                    data="\n".join(encode_case(limbs, *case) for case in cases) + "\n")
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    lines = result.stdout.splitlines()
    assert len(lines) == len(cases)
    for line, (a, b, carry) in zip(lines, cases):
        tokens = line.split(" ")
        assert len(tokens) == limbs + 1
        assert all(len(token) == 16 and set(token) <= set("0123456789abcdef") for token in tokens[:-1])
        assert tokens[-1] in {"0", "1"}
        actual_low = sum(int(token, 16) << (64 * i) for i, token in enumerate(tokens[:-1]))
        expected_carry, expected_low = divmod(a + b + carry, modulus)
        assert actual_low.to_bytes(8 * limbs, "little") == expected_low.to_bytes(8 * limbs, "little")
        assert int(tokens[-1]) == expected_carry


@pytest.mark.parametrize("data", [
    "", "\n", "   \n", "0 0 0\n", "2 0000000000000000 0000000000000000\n",
    "00 0000000000000000 0000000000000000\n", "-1 0000000000000000 0000000000000000\n",
    "0 ffffffffffffffff FFFFFFFFFFFFFFFF\n", "0 10000000000000000 0000000000000000\n",
    "0 -000000000000001 0000000000000000\n", "0 000000000000000g 0000000000000000\n",
    "0 0000000000000000 0000000000000000 extra\n",
])
def test_invalid_correctness_records_rejected(binary, data):
    result = invoke(binary, "--check", 1, data=data)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("l0_baseline:")


@pytest.mark.parametrize("args", [
    [], ["--check", "2"], ["--check", "01"], ["--check", "1", "extra"],
    ["--bench", "2", "bulk", "1", "1", "0"],
    ["--bench", "1", "hot", "1", "1", "0"],
    ["--bench", "1", "bulk", "0", "1", "0"],
    ["--bench", "1", "bulk", "1", "0", "0"],
    ["--bench", "1", "bulk", "1", "1000001", "0"],
    ["--bench", "1", "bulk", "16777217", "1", "0"],
    ["--bench", "4", "bulk", "2400000", "1", "0"],
    ["--bench", "1", "latency", "16777216", "60", "0"],
    ["--bench", "1", "bulk", "-1", "1", "0"],
    ["--bench", "1", "bulk", "+1", "1", "0"],
    ["--bench", "1", "bulk", "1x", "1", "0"],
    ["--bench", "1", "bulk", "1", "1", "18446744073709551616"],
])
def test_invalid_cli_and_resource_requests_rejected(binary, args):
    result = invoke(binary, *args)
    assert result.returncode == 2
    assert result.stdout == ""
    assert result.stderr.startswith("l0_baseline:")


def input_records(seed, limbs, count):
    def draw():
        nonlocal seed
        seed = (seed + 0x9E3779B97F4A7C15) & MASK
        z = ((seed ^ (seed >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    for _ in range(count):
        yield [draw() for _ in range(limbs)], [draw() for _ in range(limbs)], draw() & 1


def bigint_add(a, b, carry):
    # Deliberately one whole-integer addition, not a reimplementation of add64.
    left = sum(word << (64 * i) for i, word in enumerate(a))
    right = sum(word << (64 * i) for i, word in enumerate(b))
    carry_out, low = divmod(left + right + carry, 1 << (64 * len(a)))
    return limbs_of(low, len(a)), carry_out


def expected_checksum(limbs, mode, count, passes, seed):
    values = []
    if mode == "bulk":
        for a, b, carry in input_records(seed, limbs, count):
            low, carry = bigint_add(a, b, carry)
            values.extend([*low, carry])
    else:
        a, b, carry = next(input_records(seed, limbs, 1))
        for _ in range(min(count, 4096) + count * passes):
            a, carry = bigint_add(a, b, carry)
            b = [(((word << 13) | (word >> 51)) & MASK) ^ low for word, low in zip(b, a)]
        values.extend([*a, *b, carry])
    checksum = 0xCBF29CE484222325
    for value in values:
        checksum = ((checksum ^ value) * 0x100000001B3) & MASK
    return f"{checksum:016x}"


@pytest.mark.parametrize("limbs", [1, 4])
@pytest.mark.parametrize("mode", ["latency", "bulk"])
@pytest.mark.parametrize("seed", [0, MASK])
def test_benchmark_smoke_and_bigint_checksum(binary, limbs, mode, seed):
    # Nanosecond timing values are not compared: this is a tiny harness check,
    # not a usable performance sample or a concurrent sustained benchmark.
    count, passes = 7, 3
    result = invoke(binary, "--bench", limbs, mode, count, passes, seed)
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    record = json.loads(result.stdout)
    expected_fields = {
        "protocol", "scope", "mode", "limbs", "count", "passes", "seed",
        "operation_count", "add64_calls", "elapsed_ns", "input_record_bytes",
        "output_record_bytes", "working_set_bytes", "logical_bytes_per_operation",
        "logical_bytes_processed", "checksum", "warmup_operations", "backend",
        "no_int128_defined", "int128_macro_defined",
    }
    assert set(record) == expected_fields
    assert record["protocol"] == "parseatlas_l0_baseline_v1"
    assert record["scope"] == "original_add64_composition"
    assert (record["limbs"], record["mode"], record["count"], record["passes"], record["seed"]) == (
        limbs, mode, count, passes, seed)
    assert record["operation_count"] == count * passes
    assert record["add64_calls"] == limbs * count * passes
    assert type(record["elapsed_ns"]) is int and record["elapsed_ns"] > 0
    input_bytes, output_bytes = (2 * limbs + 1) * 8, (limbs + 1) * 8
    assert record["input_record_bytes"] == input_bytes
    assert record["output_record_bytes"] == output_bytes
    expected_working = count * (input_bytes + output_bytes) if mode == "bulk" else input_bytes
    logical_bytes = input_bytes + output_bytes if mode == "bulk" else 0
    assert record["working_set_bytes"] == expected_working
    assert record["logical_bytes_per_operation"] == logical_bytes
    assert record["logical_bytes_processed"] == logical_bytes * count * passes
    assert record["warmup_operations"] == (count if mode == "bulk" else min(count, 4096))
    assert record["checksum"] == expected_checksum(limbs, mode, count, passes, seed)
    assert record["backend"] == ("portable_carry" if binary[1] else "gcc_uint128")
    assert record["no_int128_defined"] is binary[1]
    assert record["int128_macro_defined"] is True
    again = invoke(binary, "--bench", limbs, mode, count, passes, seed)
    assert again.returncode == 0
    assert json.loads(again.stdout)["checksum"] == record["checksum"]
