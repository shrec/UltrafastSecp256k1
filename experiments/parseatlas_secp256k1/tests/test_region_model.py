"""Executable L0 controls and rejection branches; no production equivalence claim."""

from dataclasses import FrozenInstanceError, replace
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "region_model.py"
SPEC = importlib.util.spec_from_file_location("parseatlas_region_model", MODULE_PATH)
model = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = model
SPEC.loader.exec_module(model)

BoundaryState = model.BoundaryState
ContractError = model.ContractError
L0Domain = model.L0Domain
Node = model.Node
Op = model.Op
Region = model.Region


@pytest.mark.parametrize("word_bits,limbs", [(1, 1), (2, 1), (2, 2), (2, 3), (1, 4)])
def test_tiny_exhaustive_whole_domain(word_bits, limbs):
    domain = L0Domain(word_bits, limbs)
    serial = model.serial_region(domain)
    prefix, witness = model.derive_prefix(serial)
    model.replay_derivation(serial, prefix, witness)
    for a, b, cin in itertools.product(range(domain.limit), range(domain.limit), (0, 1)):
        expected = model.bigint_baseline(domain, a, b, cin)
        for region in (serial, prefix):
            actual = model.decode_output(model.evaluate(region, model.encode_inputs(domain, a, b, cin)))
            assert actual == expected
            assert actual.total == a + b + cin


@pytest.mark.parametrize("limbs", [1, 2, 3, 4])
def test_64_bit_limb_boundaries(limbs):
    domain = L0Domain(64, limbs)
    values = {0, 1, domain.limit - 1, domain.limit - 2}
    for i in range(1, limbs):
        values.update({(1 << (64 * i)) - 1, 1 << (64 * i), (1 << (64 * i)) + 1})
    for region in (model.serial_region(domain), model.prefix_region(domain)):
        for a, b, cin in itertools.product(sorted(values), sorted(values), (0, 1)):
            observation = model.observe(region, a, b, cin)
            assert observation.agrees
            assert observation.claim_class == "OBSERVED"
            assert len(observation.actual.low_bytes) == 8 * limbs
            assert observation.actual.total == a + b + cin


def test_gp_directional_fixture_and_final_carry():
    region = model.prefix_region(L0Domain(2, 2))
    out = model.observe(region, 3, 1, 0).actual
    assert (out.low, out.carry_out, out.low_bytes) == (4, 0, b"\x04")
    out = model.observe(region, 15, 15, 1).actual
    assert (out.low, out.carry_out) == (15, 1)


@pytest.mark.parametrize("kwargs", [
    {"word_bits": 0, "limbs": 1}, {"word_bits": 65, "limbs": 1},
    {"word_bits": True, "limbs": 1}, {"word_bits": 2, "limbs": False},
    {"word_bits": 2, "limbs": 0}, {"word_bits": 2, "limbs": 5},
    {"word_bits": 2, "limbs": 1, "kind": "field_p"},
    {"word_bits": 2, "limbs": 1, "kind": "scalar_n"},
    {"word_bits": 2, "limbs": 1, "modulus": 17},
    {"word_bits": 2, "limbs": 1, "modulus": False},
])
def test_invalid_domain(kwargs):
    with pytest.raises(ContractError):
        L0Domain(**kwargs)


@pytest.mark.parametrize("args", [(-1, 0, 0), (16, 0, 0), (0, -1, 0),
                                       (0, 16, 0), (0, 0, -1), (0, 0, 2),
                                       (True, 0, 0), (0, False, 0), (0, 0, True)])
def test_invalid_input_ranges(args):
    for encoder in (model.encode_inputs, model.bigint_baseline):
        with pytest.raises(ContractError):
            encoder(L0Domain(2, 2), *args)


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate", "reorder", "carry2", "overflow", "bool", "list"])
def test_invalid_state(mutation):
    state = model.encode_inputs(L0Domain(2, 2), 0, 0)
    entries = state.entries
    if mutation == "missing":
        entries = entries[:-1]
    elif mutation == "extra":
        entries += (("hidden", 0),)
    elif mutation == "duplicate":
        entries = entries[:-1] + (entries[0],)
    elif mutation == "reorder":
        entries = tuple(reversed(entries))
    elif mutation == "carry2":
        entries = entries[:-1] + (("carry_in", 2),)
    elif mutation == "overflow":
        entries = (("a0", 4),) + entries[1:]
    elif mutation == "bool":
        entries = (("a0", False),) + entries[1:]
    elif mutation == "list":
        entries = list(entries)
    with pytest.raises(ContractError):
        replace(state, entries=entries)


def test_state_domain_role_representation_and_immutability():
    state = model.encode_inputs(L0Domain(2, 2), 0, 0)
    region = model.serial_region(state.domain)
    with pytest.raises(ContractError, match="domain mismatch"):
        model.evaluate(region, model.encode_inputs(L0Domain(4, 1), 0, 0))
    with pytest.raises(ContractError):
        model.decode_output(state)
    with pytest.raises(ContractError):
        replace(state, representation="lazy_normalized")
    with pytest.raises(ContractError):
        replace(state, role="anything")
    with pytest.raises(FrozenInstanceError):
        state.entries = ()
    with pytest.raises(FrozenInstanceError):
        state.domain.word_bits = 64
    assert model.evaluate(region, state) == model.evaluate(region, state)
    assert dict(state.entries)["carry_in"] == 0


@pytest.mark.parametrize("mutation", ["arity", "type", "future", "cycle", "duplicate", "unknown_output", "output_type", "missing_output", "output_order", "duplicate_input", "input_order", "mutable_nodes", "empty"])
def test_invalid_region(mutation):
    region = model.serial_region(L0Domain(2, 2))
    kwargs = {}
    if mutation == "arity":
        kwargs["nodes"] = (replace(region.nodes[0], args=("a0", "b0")),) + region.nodes[1:]
    elif mutation == "type":
        kwargs["nodes"] = (replace(region.nodes[0], args=("a0", "b0", "a1")),) + region.nodes[1:]
    elif mutation in ("future", "cycle"):
        reference = "serial1.carry" if mutation == "future" else "serial0.carry"
        kwargs["nodes"] = (replace(region.nodes[0], args=("a0", "b0", reference)),) + region.nodes[1:]
    elif mutation == "duplicate":
        kwargs["nodes"] = region.nodes + (region.nodes[0],)
    elif mutation == "unknown_output":
        kwargs["outputs"] = region.outputs[:-1] + (("carry_out", "absent.carry"),)
    elif mutation == "output_type":
        kwargs["outputs"] = region.outputs[:-1] + (("carry_out", "serial1.low"),)
    elif mutation == "missing_output":
        kwargs["outputs"] = region.outputs[:-1]
    elif mutation == "output_order":
        kwargs["outputs"] = tuple(reversed(region.outputs))
    elif mutation == "duplicate_input":
        kwargs["inputs"] = region.inputs[:-1] + (("carry_in", "a0"),)
    elif mutation == "input_order":
        kwargs["inputs"] = tuple(reversed(region.inputs))
    elif mutation == "mutable_nodes":
        kwargs["nodes"] = list(region.nodes)
    elif mutation == "empty":
        kwargs["nodes"] = ()
    with pytest.raises(ContractError):
        replace(region, **kwargs)


@pytest.mark.parametrize("op", ["adc", "multiply", lambda x: x, None])
def test_unregistered_operations(op):
    with pytest.raises(ContractError, match="unregistered"):
        Node("rogue", op, ())


def _rename(region):
    aliases = {ref: f"input{i}" for i, (_, ref) in enumerate(region.inputs)}
    for i, node in enumerate(region.nodes):
        aliases.update((ref, f"renamed{i}.{ref.split('.')[1]}") for ref, _ in node.ports)
    inputs = tuple((key, aliases[ref]) for key, ref in region.inputs)
    nodes = tuple(Node(f"renamed{i}", node.op, tuple(aliases[arg] for arg in node.args))
                  for i, node in enumerate(region.nodes))
    outputs = tuple((key, aliases[ref]) for key, ref in region.outputs)
    return Region(region.domain, inputs, nodes, outputs)


def test_identifier_neutral_null_control():
    for region in (model.serial_region(L0Domain(2, 4)), model.prefix_region(L0Domain(2, 4))):
        renamed = _rename(region)
        for args in ((0, 0, 0), (255, 1, 0), (7, 201, 1)):
            assert model.observe(region, *args).actual == model.observe(renamed, *args).actual
        before, after = model.static_diagnostics(region), model.static_diagnostics(renamed)
        assert replace(before, schedule=after.schedule) == after
        assert model.region_digest(region) != model.region_digest(renamed)
    with pytest.raises(ContractError, match="registered serial"):
        model.derive_prefix(_rename(model.serial_region(L0Domain(2, 2))))


def test_witness_json_roundtrip_and_replay():
    source = model.serial_region(L0Domain(64, 4))
    target, witness = model.derive_prefix(source)
    parsed = model.DerivationWitness.from_record(json.loads(json.dumps(witness.to_record())))
    assert parsed == witness
    model.replay_derivation(source, target, parsed)
    with pytest.raises(ContractError, match="replay mismatch"):
        model.replay_derivation(source, target, replace(witness, target_digest="0" * 64))
    with pytest.raises(ContractError, match="domain mismatch"):
        model.replay_derivation(source, target, replace(witness, domain=L0Domain(32, 4)))


@pytest.mark.parametrize("mutation", ["extra", "missing", "domain_extra", "modulus", "rule", "bool_version", "bad_digest"])
def test_witness_metadata_rejections(mutation):
    _, witness = model.derive_prefix(model.serial_region(L0Domain(2, 2)))
    record = witness.to_record()
    if mutation == "extra":
        record["equivalent"] = True
    elif mutation == "missing":
        del record["source_digest"]
    elif mutation == "domain_extra":
        record["domain"]["passed"] = True
    elif mutation == "modulus":
        record["domain"]["modulus"] = 13
    elif mutation == "rule":
        record["rule"] = "trust_me"
    elif mutation == "bool_version":
        record["version"] = True
    elif mutation == "bad_digest":
        record["target_digest"] = "verified"
    with pytest.raises(ContractError):
        model.DerivationWitness.from_record(record)


@pytest.mark.parametrize("mutation", ["swapped_low", "carry_dropped", "cin_dropped", "reverse_gp", "same_type_op"])
def test_forged_recomputed_witness_cannot_certify_wrong_graph(mutation):
    source = model.serial_region(L0Domain(2, 4))
    target, witness = model.derive_prefix(source)
    if mutation == "swapped_low":
        outputs = list(target.outputs)
        outputs[0], outputs[1] = ("low0", outputs[1][1]), ("low1", outputs[0][1])
        forged = replace(target, outputs=tuple(outputs))
    elif mutation == "carry_dropped":
        forged = replace(target, outputs=target.outputs[:-1] + (("carry_out", "carry_in"),))
    else:
        nodes = list(target.nodes)
        if mutation == "cin_dropped":
            index = next(i for i, node in enumerate(nodes) if node.id == "finish0")
            nodes[index] = replace(nodes[index], args=("split0.low", "split0.g", "split0.g"))
        elif mutation == "reverse_gp":
            index = next(i for i, node in enumerate(nodes) if node.op is Op.COMPOSE_GP)
            nodes[index] = replace(nodes[index], args=nodes[index].args[2:] + nodes[index].args[:2])
        else:
            # APPLY_GP's three bit args retain their types but lose their roles.
            index = next(i for i, node in enumerate(nodes) if node.op is Op.APPLY_GP)
            g, p, cin = nodes[index].args
            nodes[index] = replace(nodes[index], args=(p, g, cin))
        forged = replace(target, nodes=tuple(nodes))
    forged_witness = replace(witness, target_digest=model.region_digest(forged))
    with pytest.raises(ContractError, match="registered prefix template"):
        model.replay_derivation(source, forged, forged_witness)
    # Independently retain an actual disagreement, not just a shape rejection.
    assert any(not model.observe(forged, a, b, cin).agrees
               for a, b, cin in itertools.product((0, 1, 3, 4, 15, 16, 63, 255), repeat=3)
               if cin in (0, 1))


def test_serial_source_with_forged_recomputed_hash_is_rejected():
    source = model.serial_region(L0Domain(2, 2))
    target, witness = model.derive_prefix(source)
    forged_source = replace(source, outputs=source.outputs[:-1] + (("carry_out", "carry_in"),))
    forged_witness = replace(witness, source_digest=model.region_digest(forged_source))
    with pytest.raises(ContractError, match="registered serial template"):
        model.replay_derivation(forged_source, target, forged_witness)


def test_static_costs_are_separate_and_no_favorable_prefix_claim():
    domain = L0Domain(64, 4)
    serial = model.static_diagnostics(model.serial_region(domain))
    prefix = model.static_diagnostics(model.prefix_region(domain))
    assert serial.operation_counts == (("adc", 4),)
    assert serial.region_depth == serial.output_depth == 4
    assert prefix.operation_counts == (("apply_gp", 3), ("compose_gp", 3), ("finish", 4), ("split_gp", 4))
    assert prefix.region_depth == prefix.output_depth == 5
    assert serial.encode_counts == prefix.encode_counts == (("limb_divmod", 8),)
    assert serial.decode_counts == prefix.decode_counts == (("limb_shift", 4), ("limb_sum_term", 4), ("fixed_width_bytes", 1))
    assert serial.provenance == prefix.provenance == "STATIC_MODEL"
    assert not hasattr(serial, "latency")
    assert not hasattr(prefix, "boundary_equivalent")


def test_liveness_atomic_result_allocation_and_schedule():
    one = model.static_diagnostics(model.serial_region(L0Domain(2, 1)))
    assert one.peak_live_values == 5  # two words+cin, then low+cout before freeing
    assert one.peak_live_bits == 8
    region = model.prefix_region(L0Domain(2, 4))
    changed = tuple(node.id for node in region.nodes)
    changed = (changed[1], changed[0]) + changed[2:]
    cost = model.static_diagnostics(region, changed)
    assert cost.region_depth == model.static_diagnostics(region).region_depth
    with pytest.raises(ContractError, match="topological"):
        model.static_diagnostics(region, tuple(reversed(changed)))
    for bad in (changed[:-1], changed + (changed[0],), list(changed)):
        with pytest.raises(ContractError):
            model.static_diagnostics(region, bad)


def test_low_bytes_cannot_hide_a_bad_carry():
    source = model.serial_region(L0Domain(64, 4))
    bad = replace(source, outputs=source.outputs[:-1] + (("carry_out", "carry_in"),))
    observation = model.observe(bad, source.domain.limit - 1, 1, 0)
    assert observation.actual.low_bytes == observation.expected.low_bytes
    assert observation.actual.carry_out != observation.expected.carry_out
    assert not observation.agrees


def test_finish_rejects_arithmetic_carry_two_in_arbitrary_typed_graph():
    domain = L0Domain(2, 1)
    base = model.serial_region(domain)
    graph = Region(domain, base.inputs,
                   (Node("bad", Op.FINISH, ("a0", "carry_in", "carry_in")),),
                   (("low0", "bad.low"), ("carry_out", "bad.carry")))
    # 1*radix + (radix-1) + 1 = 2*radix needs carry2, outside this boundary.
    with pytest.raises(ContractError, match="bad.carry: integer outside"):
        model.evaluate(graph, model.encode_inputs(domain, 3, 0, 1))


@pytest.mark.parametrize("entries", [(("low0", 4), ("carry_out", 0)),
                                    (("low0", 0), ("carry_out", 2)),
                                    (("low0", 0), ("carry_out", True)),
                                    (("low0", -1), ("carry_out", 0))])
def test_invalid_output_state(entries):
    with pytest.raises(ContractError):
        BoundaryState(L0Domain(2, 1), "output", entries)
