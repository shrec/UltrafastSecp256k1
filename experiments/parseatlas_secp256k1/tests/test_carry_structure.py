"""Independent arithmetic oracles and explicit carry/state-loss controls."""

import hashlib
import importlib.util
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "carry_structure.py"
SPEC = importlib.util.spec_from_file_location("parseatlas_carry_structure_under_test", SOURCE)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


@pytest.mark.parametrize("gp,truth", [((0, 0), (0, 0)), ((0, 1), (0, 1)),
                                    ((1, 0), (1, 1)), ((1, 1), (1, 1))])
def test_four_encodings_three_functions(gp, truth):
    assert m.from_gp(*gp).outputs == truth


@pytest.mark.parametrize("bits", [1, 2, 3, 4])
def test_discovered_from_actual_digit_addition(bits):
    radix = 1 << bits
    seen = set()
    for a, b in product(range(radix), repeat=2):
        raw, summary = m.digit_summary(a, b, bits)
        truth = tuple((a + b + c) // radix for c in (0, 1))
        assert summary.outputs == truth
        assert raw == (a + b) % radix
        seen.add(truth)
    assert seen == {(0, 0), (0, 1), (1, 1)}


def test_composition_direction_not_just_associativity():
    assert m.compose(m.K, m.G) == m.K
    assert m.compose(m.G, m.K) == m.G
    assert m.compose_gp((0, 0), (1, 0)) == (0, 0)
    assert m.compose_gp((1, 0), (0, 0)) == (1, 0)
    # Low digit generates; high digit kills. Reversing significance gives G.
    assert m.block_summary((1, 0), (1, 0), 1) == m.K
    out = m.add_reblocked((1, 0), (1, 0), 0, 1, (1, 1))
    assert (out.low_limbs, out.carry_out, out.total) == ((0, 1), 0, 2)


def test_all_gp_encodings_compose_including_redundant_generate():
    encodings = tuple(product((0, 1), repeat=2))
    for high, low in product(encodings, repeat=2):
        encoded = m.from_gp(*m.compose_gp(high, low))
        for cin in (0, 1):
            assert encoded.apply(cin) == m.from_gp(*high).apply(m.from_gp(*low).apply(cin))


def test_associativity_identity_and_minimality_scope():
    for h, mid, low in product(m.TRANSFERS, repeat=3):
        assert m.compose(h, m.compose(mid, low)) == m.compose(m.compose(h, mid), low)
    for t in m.TRANSFERS:
        assert m.compose(t, m.P) == m.compose(m.P, t) == t
    assert m.K.apply(1) != m.P.apply(1)
    assert m.K.apply(0) != m.G.apply(0)
    assert m.P.apply(0) != m.G.apply(0)
    # If cin is fixed to 0, K and P are indistinguishable; minimality needs both.
    assert m.K.apply(0) == m.P.apply(0)


def test_same_transfer_different_low_outputs():
    raw0, transfer0 = m.digit_summary(0, 0, 2)
    raw1, transfer1 = m.digit_summary(0, 1, 2)
    assert transfer0 == transfer1 == m.K
    assert raw0 != raw1
    out0 = m.add_reblocked((0,), (0,), 0, 2, (1,))
    out1 = m.add_reblocked((0,), (1,), 0, 2, (1,))
    assert out0.carry_out == out1.carry_out
    assert out0.low_bytes != out1.low_bytes


@pytest.mark.parametrize("limbs", [1, 2, 3, 4])
def test_partitions_are_all_contiguous_cuts(limbs):
    result = m.partitions(limbs)
    assert len(result) == len(set(result)) == 2 ** (limbs - 1)
    assert all(sum(partition) == limbs and all(n > 0 for n in partition) for partition in result)
    assert (limbs,) in result and (1,) * limbs in result


@pytest.mark.parametrize("bits,limbs", [(1, 1), (1, 4), (2, 2), (3, 1)])
def test_exhaustive_reblocking_full_low_carry_bytes_and_total(bits, limbs):
    radix, limit = 1 << bits, 1 << (bits * limbs)
    for a, b, cin in product(range(limit), range(limit), (0, 1)):
        aa = tuple(a // radix ** i % radix for i in range(limbs))
        bb = tuple(b // radix ** i % radix for i in range(limbs))
        expected = a + b + cin
        high, low = divmod(expected, limit)
        for blocks in m.partitions(limbs):
            actual = m.add_reblocked(aa, bb, cin, bits, blocks)
            assert actual.low_limbs == tuple(low // radix ** i % radix for i in range(limbs))
            assert actual.carry_out == high
            assert actual.low == low
            assert actual.low_bytes == low.to_bytes((bits * limbs + 7) // 8, "little")
            assert actual.total == expected


@pytest.mark.parametrize("limbs", [1, 2, 3, 4])
def test_full_word_boundary_fixtures_not_full_domain_proof(limbs):
    bits = 64
    radix, limit = 1 << bits, 1 << (bits * limbs)
    values = (0, 1, radix - 1, limit // 2, limit - 2, limit - 1)
    for a, b, cin in product(values, values, (0, 1)):
        aa = tuple(a // radix ** i % radix for i in range(limbs))
        bb = tuple(b // radix ** i % radix for i in range(limbs))
        for blocks in m.partitions(limbs):
            out = m.add_reblocked(aa, bb, cin, bits, blocks)
            expected = a + b + cin
            assert out.total == expected
            assert out.low_bytes == (expected % limit).to_bytes(8 * limbs, "little")
            assert out.carry_out == expected // limit


def test_lossy_tuple_combine_is_not_associative_but_modulo_low_is():
    combine = m.naive_low_only_combine
    left = combine(combine((1, 0), (1, 0), 2), (3, 0), 2)
    right = combine((1, 0), combine((1, 0), (3, 0), 2), 2)
    assert left == (1, 1)
    assert right == (1, 0)
    assert left[0] == right[0] == (1 + 1 + 3) % 4


def test_three_max_operands_need_two_high_units_not_nonassociativity():
    combine = m.naive_low_only_combine
    left = combine(combine((3, 0), (3, 0), 2), (3, 0), 2)
    right = combine((3, 0), combine((3, 0), (3, 0), 2), 2)
    assert left == right == (1, 1)
    assert divmod(3 + 3 + 3, 4) == (2, 1)
    assert left[0] + 4 * left[1] != 9
    with pytest.raises(m.ContractError):
        m.AdditionOutput((1,), 2, 2)


@pytest.mark.parametrize("truth", [(1, 0), (0, 2), (True, 1), [0, 1], (0,), "01", None])
def test_reject_bad_transfers(truth):
    with pytest.raises(m.ContractError):
        m.Transfer(truth)


@pytest.mark.parametrize("bit", [-1, 2, True, 0.0, "0", None])
def test_reject_bad_carry_or_gp_bit(bit):
    with pytest.raises(m.ContractError):
        m.P.apply(bit)
    with pytest.raises(m.ContractError):
        m.from_gp(bit, 0)
    with pytest.raises(m.ContractError):
        m.from_gp(0, bit)
    with pytest.raises(m.ContractError):
        m.compose_gp((0, bit), (0, 1))
    with pytest.raises(m.ContractError):
        m.add_reblocked((0,), (0,), bit, 1, (1,))


@pytest.mark.parametrize("bits", [0, 65, -1, True, 1.0, "2", None])
def test_reject_bad_word_width(bits):
    with pytest.raises(m.ContractError):
        m.digit_summary(0, 0, bits)
    with pytest.raises(m.ContractError):
        m.add_reblocked((0,), (0,), 0, bits, (1,))


@pytest.mark.parametrize("a,b", [([], []), ((0,), [0]), ((), ()), ((0,), (0, 0)),
                               ((0,) * 5, (0,) * 5), ((2,), (0,)), ((-1,), (0,)),
                               ((True,), (0,)), ((0.0,), (0,))])
def test_reject_malformed_digit_vectors(a, b):
    with pytest.raises(m.ContractError):
        m.block_summary(a, b, 1)
    with pytest.raises(m.ContractError):
        m.add_reblocked(a, b, 0, 1, (1,))


@pytest.mark.parametrize("partition", [(), [], (0, 2), (-1, 3), (1,), (3,),
                                      (1, 1, 1), (True, 1), (1.0, 1), ("1", 1)])
def test_reject_malformed_reblocking(partition):
    with pytest.raises(m.ContractError):
        m.add_reblocked((0, 0), (0, 0), 0, 1, partition)


@pytest.mark.parametrize("limbs", [0, 5, True, "2", 2.0])
def test_reject_unbounded_partitions(limbs):
    with pytest.raises(m.ContractError):
        m.partitions(limbs)


@pytest.mark.parametrize("args", [(0, 6, 4), (5, 6, 4), (4, 0, 4), (4, 7, 4),
                                  (4, 6, 0), (4, 6, 5), (True, 6, 4), (4, 6.0, 4)])
def test_reject_exhaustive_resource_bounds(args):
    with pytest.raises(m.ContractError):
        m.exhaustive_report(*args)


def test_reject_malformed_gp_and_lossy_state():
    for value in (None, [0, 1], (0,), (0, 1, 0)):
        with pytest.raises(m.ContractError):
            m.compose_gp(value, (0, 1))
        with pytest.raises(m.ContractError):
            m.naive_low_only_combine(value, (0, 0), 2)
    with pytest.raises(m.ContractError):
        m.compose((0, 0), m.P)
    with pytest.raises(m.ContractError):
        m.naive_low_only_combine((0, 2), (0, 0), 2)


def test_runtime_invariant_not_silently_ignored(monkeypatch):
    monkeypatch.setattr(m, "block_summary", lambda *args: m.G)
    with pytest.raises(m.DiagnosticMismatch, match="block carry"):
        m.add_reblocked((0,), (0,), 0, 2, (1,))


def test_report_is_deterministic_bounded_and_complete():
    first = m.exhaustive_report(2, 3, 3)
    assert first == m.exhaustive_report(2, 3, 3)
    assert first["source_sha256"] == hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    assert first["input_triples_across_layouts"] == 200
    assert first["full_output_evaluations"] == 616
    assert sum(r["ordered_digit_pairs"] for r in first["digit_enumerations"]) == 20
    assert first["algebra"]["gp_composition_cases"] == 16
    assert first["algebra"]["associativity_cases"] == 27
    assert len(first["algebra"]["pairwise_distinguishability"]) == 3
    assert all(row["mismatches"] == 0 for row in first["layouts"])
    assert first["runtime"] == "PENDING IMPORT"
    assert first["claim_class"] == "OBSERVED"
    negatives = first["negative_controls"]
    assert negatives["triple_cases"] == 64
    assert negatives["exact_sum_requires_high_two_cases"] == 4
    assert negatives["lossy_tuple_nonassociativity_cases"] > 0


def test_report_checks_low_outputs_not_just_final_carry(monkeypatch):
    original = m.add_reblocked

    def corrupt_low(*args):
        out = original(*args)
        return m.AdditionOutput((out.low_limbs[0] ^ 1,) + out.low_limbs[1:],
                                out.carry_out, out.word_bits)

    monkeypatch.setattr(m, "add_reblocked", corrupt_low)
    with pytest.raises(m.DiagnosticMismatch, match="full-boundary"):
        m.exhaustive_report(1, 1, 1)


def test_report_checks_final_carry_not_just_low_outputs(monkeypatch):
    original = m.add_reblocked

    def corrupt_carry(*args):
        out = original(*args)
        return m.AdditionOutput(out.low_limbs, out.carry_out ^ 1, out.word_bits)

    monkeypatch.setattr(m, "add_reblocked", corrupt_carry)
    with pytest.raises(m.DiagnosticMismatch, match="full-boundary"):
        m.exhaustive_report(1, 1, 1)


def test_cli_stdout_json_and_stable_reproduction():
    result = subprocess.run([sys.executable, "-P", str(SOURCE), "--report", "--max-word-bits", "1",
                             "--max-total-bits", "1", "--max-limbs", "1"],
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0 and result.stderr == ""
    actual = json.loads(result.stdout)
    assert actual == json.loads(json.dumps(m.exhaustive_report(1, 1, 1)))


@pytest.mark.parametrize("args", [[], ["--report", "--max-word-bits", "5"],
                                  ["--report", "--max-total-bits", "7"],
                                  ["--report", "--max-limbs", "0"],
                                  ["--report", "--max-total-bits", "bad"]])
def test_cli_rejects_invalid_bounds_without_json_success(args):
    result = subprocess.run([sys.executable, "-P", str(SOURCE), *args], text=True,
                            capture_output=True, check=False)
    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr
