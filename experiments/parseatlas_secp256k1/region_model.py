"""Bounded L0 addition research kernel; not production arithmetic or a proof engine.

The closed operation vocabulary and exact derivation templates are trusted model
code. Structural validity, template reachability and finite boundary agreement are
different checks. Static diagnostic costs confer no research-contract eligibility.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import re
from typing import Mapping


class ContractError(ValueError):
    """A domain, state, graph, schedule or derivation contract was violated."""


def _integer(value: object, minimum: int, maximum: int, label: str) -> None:
    # bool is deliberately excluded: a machine carry is exactly integer 0 or 1.
    if type(value) is not int or not minimum <= value <= maximum:
        raise ContractError(f"{label}: integer outside [{minimum}, {maximum}]")


def _tuple(value: object, label: str) -> None:
    if type(value) is not tuple:
        raise ContractError(f"{label}: immutable tuple required")


def _pairs(value: object, label: str) -> None:
    _tuple(value, label)
    if any(type(pair) is not tuple or len(pair) != 2 for pair in value):
        raise ContractError(f"{label}: immutable key/value pairs required")


def _identifier(value: object) -> None:
    if type(value) is not str or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value):
        raise ContractError("invalid identifier")


@dataclass(frozen=True)
class L0Domain:
    word_bits: int
    limbs: int
    kind: str = "L0_UNSIGNED"
    modulus: None = None

    def __post_init__(self) -> None:
        _integer(self.word_bits, 1, 64, "word_bits")
        _integer(self.limbs, 1, 4, "limbs")
        if self.kind != "L0_UNSIGNED" or type(self.kind) is not str:
            raise ContractError("unsupported domain: only L0_UNSIGNED is implemented")
        if self.modulus is not None:
            raise ContractError("L0 has no modulus; Fp/Fn require different contracts")

    @property
    def width(self) -> int:
        return self.word_bits * self.limbs

    @property
    def radix(self) -> int:
        return 1 << self.word_bits

    @property
    def limit(self) -> int:
        return 1 << self.width


def _domain(domain: object) -> None:
    if type(domain) is not L0Domain:
        raise ContractError("exact L0Domain required")


def _input_keys(domain: L0Domain) -> tuple[str, ...]:
    return tuple(f"{side}{i}" for side in ("a", "b") for i in range(domain.limbs)) + ("carry_in",)


def _output_keys(domain: L0Domain) -> tuple[str, ...]:
    return tuple(f"low{i}" for i in range(domain.limbs)) + ("carry_out",)


@dataclass(frozen=True)
class BoundaryState:
    domain: L0Domain
    role: str
    entries: tuple[tuple[str, int], ...]
    representation: str = "canonical_little_endian_limbs"

    def __post_init__(self) -> None:
        _domain(self.domain)
        if self.role not in ("input", "output") or type(self.role) is not str:
            raise ContractError("invalid boundary state role")
        if self.representation != "canonical_little_endian_limbs" or type(self.representation) is not str:
            raise ContractError("unsupported representation; no implicit normalization")
        _pairs(self.entries, "state entries")
        keys = _input_keys(self.domain) if self.role == "input" else _output_keys(self.domain)
        if tuple(k for k, _ in self.entries) != keys:
            raise ContractError("missing, extra, duplicate or noncanonical state keys")
        for key, value in self.entries:
            maximum = 1 if key.startswith("carry_") else self.domain.radix - 1
            _integer(value, 0, maximum, key)


def encode_inputs(domain: L0Domain, a: int, b: int, carry_in: int = 0) -> BoundaryState:
    _domain(domain)
    _integer(a, 0, domain.limit - 1, "a")
    _integer(b, 0, domain.limit - 1, "b")
    _integer(carry_in, 0, 1, "carry_in")
    entries = []
    for side, value in (("a", a), ("b", b)):
        for i in range(domain.limbs):
            value, limb = divmod(value, domain.radix)
            entries.append((f"{side}{i}", limb))
        if value:
            raise ContractError("encoder discarded high state")
    return BoundaryState(domain, "input", tuple(entries) + (("carry_in", carry_in),))


@dataclass(frozen=True)
class DecodedOutput:
    low: int
    carry_out: int
    low_bytes: bytes
    width: int

    @property
    def total(self) -> int:
        return self.low + (self.carry_out << self.width)


def decode_output(state: BoundaryState) -> DecodedOutput:
    if type(state) is not BoundaryState or state.role != "output":
        raise ContractError("output BoundaryState required")
    # State validation occurs at construction; immutable tuples prevent drift.
    low = sum(value << (i * state.domain.word_bits)
              for i, (_, value) in enumerate(state.entries[:-1]))
    carry = state.entries[-1][1]
    return DecodedOutput(low, carry, low.to_bytes((state.domain.width + 7) // 8, "little"), state.domain.width)


def bigint_baseline(domain: L0Domain, a: int, b: int, carry_in: int = 0) -> DecodedOutput:
    """Whole-integer specification, independent of limb encoding and DAG ops."""
    _domain(domain)
    _integer(a, 0, domain.limit - 1, "a")
    _integer(b, 0, domain.limit - 1, "b")
    _integer(carry_in, 0, 1, "carry_in")
    carry, low = divmod(a + b + carry_in, domain.limit)
    return DecodedOutput(low, carry, low.to_bytes((domain.width + 7) // 8, "little"), domain.width)


class Kind(Enum):
    WORD = "word"
    BIT = "bit"


class Op(Enum):
    ADC = "adc"
    SPLIT_GP = "split_gp"
    COMPOSE_GP = "compose_gp"
    APPLY_GP = "apply_gp"
    FINISH = "finish"


def _signature(op: Op) -> tuple[tuple[Kind, ...], tuple[tuple[str, Kind], ...]]:
    w, b = Kind.WORD, Kind.BIT
    if op is Op.ADC:
        return (w, w, b), (("low", w), ("carry", b))
    if op is Op.SPLIT_GP:
        return (w, w), (("low", w), ("g", b), ("p", b))
    if op is Op.COMPOSE_GP:
        return (b, b, b, b), (("g", b), ("p", b))
    if op is Op.APPLY_GP:
        return (b, b, b), (("carry", b),)
    if op is Op.FINISH:
        return (w, b, b), (("low", w), ("carry", b))
    raise ContractError("unregistered operation")


@dataclass(frozen=True)
class Node:
    id: str
    op: Op
    args: tuple[str, ...]

    def __post_init__(self) -> None:
        _identifier(self.id)
        if type(self.op) is not Op:
            raise ContractError("unregistered operation")
        _tuple(self.args, "node args")
        if any(type(arg) is not str for arg in self.args):
            raise ContractError("node args must be symbol references")

    @property
    def ports(self) -> tuple[tuple[str, Kind], ...]:
        return tuple((f"{self.id}.{suffix}", kind) for suffix, kind in _signature(self.op)[1])


@dataclass(frozen=True)
class Region:
    domain: L0Domain
    inputs: tuple[tuple[str, str], ...]
    nodes: tuple[Node, ...]
    outputs: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        validate_region(self)


def validate_region(region: Region) -> dict[str, Kind]:
    if type(region) is not Region:
        raise ContractError("exact Region required")
    _domain(region.domain)
    _pairs(region.inputs, "input bindings")
    _pairs(region.outputs, "output bindings")
    _tuple(region.nodes, "nodes")
    if not 1 <= len(region.nodes) <= 128:
        raise ContractError("region requires 1..128 nodes")
    if tuple(k for k, _ in region.inputs) != _input_keys(region.domain):
        raise ContractError("noncanonical input boundary binding")
    if tuple(k for k, _ in region.outputs) != _output_keys(region.domain):
        raise ContractError("noncanonical output boundary binding")
    symbols: dict[str, Kind] = {}
    for key, symbol in region.inputs:
        _identifier(symbol)
        if symbol in symbols:
            raise ContractError("duplicate input symbol")
        symbols[symbol] = Kind.BIT if key == "carry_in" else Kind.WORD
    ids = set(symbols)
    for node in region.nodes:
        if type(node) is not Node:
            raise ContractError("exact Node required")
        if node.id in ids:
            raise ContractError("duplicate node identifier")
        ids.add(node.id)
        expected, _ = _signature(node.op)
        if len(node.args) != len(expected):
            raise ContractError("operation arity mismatch")
        for arg, kind in zip(node.args, expected):
            if arg not in symbols:
                raise ContractError("undefined or forward reference; topological DAG required")
            if symbols[arg] is not kind:
                raise ContractError("operation type mismatch")
        symbols.update(node.ports)
    for key, ref in region.outputs:
        kind = Kind.BIT if key == "carry_out" else Kind.WORD
        if type(ref) is not str or ref not in symbols or symbols[ref] is not kind:
            raise ContractError("output reference missing or wrong type")
    return symbols


def _operation(op: Op, values: tuple[int, ...], radix: int) -> tuple[int, ...]:
    if op is Op.ADC:
        carry, low = divmod(values[0] + values[1] + values[2], radix)
        return low, carry
    if op is Op.SPLIT_GP:
        g, low = divmod(values[0] + values[1], radix)
        return low, g, int(low == radix - 1)
    if op is Op.COMPOSE_GP:
        # Ordered composition: high segment after low segment, never reversed.
        gh, ph, gl, pl = values
        return gh | (ph & gl), ph & pl
    if op is Op.APPLY_GP:
        g, p, cin = values
        return (g | (p & cin),)
    if op is Op.FINISH:
        raw_low, cin, raw_generate = values
        increment_carry, low = divmod(raw_low + cin, radix)
        # Keep arithmetic carries additive. An arbitrary typed graph can feed an
        # incoherent raw state; it must fail the BIT range check, not lose carry2.
        return low, raw_generate + increment_carry
    raise ContractError("unregistered operation")


def evaluate(region: Region, state: BoundaryState) -> BoundaryState:
    validate_region(region)
    if type(state) is not BoundaryState or state.role != "input" or state.domain != region.domain:
        raise ContractError("input state/domain mismatch")
    state_values = dict(state.entries)
    env = {symbol: state_values[key] for key, symbol in region.inputs}
    for node in region.nodes:
        values = _operation(node.op, tuple(env[arg] for arg in node.args), region.domain.radix)
        for (ref, kind), value in zip(node.ports, values):
            _integer(value, 0, 1 if kind is Kind.BIT else region.domain.radix - 1, ref)
            env[ref] = value
    return BoundaryState(region.domain, "output", tuple((key, env[ref]) for key, ref in region.outputs))


def serial_region(domain: L0Domain) -> Region:
    _domain(domain)
    nodes = []
    carry = "carry_in"
    outputs = []
    for i in range(domain.limbs):
        node = Node(f"serial{i}", Op.ADC, (f"a{i}", f"b{i}", carry))
        nodes.append(node)
        outputs.append((f"low{i}", f"{node.id}.low"))
        carry = f"{node.id}.carry"
    return Region(domain, tuple((k, k) for k in _input_keys(domain)), tuple(nodes), tuple(outputs) + (("carry_out", carry),))


def prefix_region(domain: L0Domain) -> Region:
    """Known-method control: deterministic inclusive scan, not blind synthesis."""
    _domain(domain)
    nodes = [Node(f"split{i}", Op.SPLIT_GP, (f"a{i}", f"b{i}")) for i in range(domain.limbs)]
    # The highest limb's prefix is unnecessary: FINISH exposes its final carry.
    pairs = [(f"split{i}.g", f"split{i}.p") for i in range(domain.limbs - 1)]
    distance = 1
    while distance < len(pairs):
        previous = tuple(pairs)
        for i in range(distance, len(pairs)):
            name = f"scan{distance}_{i}"
            nodes.append(Node(name, Op.COMPOSE_GP, previous[i] + previous[i - distance]))
            pairs[i] = (f"{name}.g", f"{name}.p")
        distance *= 2
    outputs = []
    for i in range(domain.limbs):
        carry = "carry_in"
        if i:
            name = f"incoming{i}"
            nodes.append(Node(name, Op.APPLY_GP, pairs[i - 1] + ("carry_in",)))
            carry = f"{name}.carry"
        name = f"finish{i}"
        nodes.append(Node(name, Op.FINISH, (f"split{i}.low", carry, f"split{i}.g")))
        outputs.append((f"low{i}", f"{name}.low"))
    return Region(domain, tuple((k, k) for k in _input_keys(domain)), tuple(nodes), tuple(outputs) + (("carry_out", f"finish{domain.limbs - 1}.carry"),))


def _domain_record(domain: L0Domain) -> dict[str, object]:
    return {"word_bits": domain.word_bits, "limbs": domain.limbs, "kind": domain.kind, "modulus": domain.modulus}


def region_record(region: Region) -> dict[str, object]:
    validate_region(region)
    return {"format": 1, "domain": _domain_record(region.domain),
            "inputs": region.inputs, "outputs": region.outputs,
            "nodes": [{"id": node.id, "op": node.op.value, "args": node.args} for node in region.nodes]}


def region_digest(region: Region) -> str:
    raw = json.dumps(region_record(region), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


RULE = "serial_adc_to_inclusive_gp_v1"


@dataclass(frozen=True)
class DerivationWitness:
    domain: L0Domain
    source_digest: str
    target_digest: str
    rule: str = RULE
    version: int = 1

    def __post_init__(self) -> None:
        _domain(self.domain)
        _integer(self.version, 1, 1, "witness version")
        if type(self.rule) is not str or self.rule != RULE:
            raise ContractError("unregistered derivation rule")
        for digest in (self.source_digest, self.target_digest):
            if type(digest) is not str or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ContractError("invalid witness digest")

    def to_record(self) -> dict[str, object]:
        return {"version": self.version, "rule": self.rule, "domain": _domain_record(self.domain),
                "source_digest": self.source_digest, "target_digest": self.target_digest}

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> DerivationWitness:
        if type(record) is not dict or set(record) != {"version", "rule", "domain", "source_digest", "target_digest"}:
            raise ContractError("missing or extra witness fields")
        domain = record["domain"]
        if type(domain) is not dict or set(domain) != {"word_bits", "limbs", "kind", "modulus"}:
            raise ContractError("missing or extra witness domain fields")
        return cls(L0Domain(**domain), record["source_digest"], record["target_digest"], record["rule"], record["version"])


def derive_prefix(source: Region) -> tuple[Region, DerivationWitness]:
    validate_region(source)
    if source != serial_region(source.domain):
        raise ContractError("derivation source is not the registered serial template")
    target = prefix_region(source.domain)
    return target, DerivationWitness(source.domain, region_digest(source), region_digest(target))


def replay_derivation(source: Region, target: Region, witness: DerivationWitness) -> None:
    """Check exact registered reachability; hashes alone never certify a rewrite."""
    validate_region(source)
    validate_region(target)
    if type(witness) is not DerivationWitness or source.domain != witness.domain or target.domain != witness.domain:
        raise ContractError("witness domain mismatch")
    expected, expected_witness = derive_prefix(source)
    if target != expected:
        raise ContractError("derivation target is not the registered prefix template")
    if witness != expected_witness:
        raise ContractError("witness replay mismatch")


@dataclass(frozen=True)
class Observation:
    inputs: tuple[int, int, int]
    actual: DecodedOutput
    expected: DecodedOutput

    @property
    def agrees(self) -> bool:
        return self.actual == self.expected

    @property
    def claim_class(self) -> str:
        return "OBSERVED"


def observe(region: Region, a: int, b: int, carry_in: int = 0) -> Observation:
    actual = decode_output(evaluate(region, encode_inputs(region.domain, a, b, carry_in)))
    expected = bigint_baseline(region.domain, a, b, carry_in)
    return Observation((a, b, carry_in), actual, expected)


@dataclass(frozen=True)
class StaticDiagnostics:
    operation_counts: tuple[tuple[str, int], ...]
    region_depth: int
    output_depth: int
    peak_live_values: int
    peak_live_bits: int
    schedule: tuple[str, ...]
    encode_counts: tuple[tuple[str, int], ...]
    decode_counts: tuple[tuple[str, int], ...]

    @property
    def provenance(self) -> str:
        return "STATIC_MODEL"


def static_diagnostics(region: Region, schedule: tuple[str, ...] | None = None) -> StaticDiagnostics:
    """Unit per registered multi-output op; atomic results-before-free liveness.

    Every input is initially resident. Boundary output values remain resident at
    the end. Validation, host allocation and actual machine costs are excluded.
    """
    symbols = validate_region(region)
    by_id = {node.id: node for node in region.nodes}
    if schedule is None:
        schedule = tuple(by_id)
    _tuple(schedule, "schedule")
    if any(type(name) is not str for name in schedule) or len(schedule) != len(by_id) or set(schedule) != set(by_id):
        raise ContractError("schedule must be an exact node permutation")
    depths = {ref: 0 for _, ref in region.inputs}
    for node in region.nodes:
        depth = 1 + max(depths[arg] for arg in node.args)
        depths.update((ref, depth) for ref, _ in node.ports)
    uses = Counter(arg for node in region.nodes for arg in node.args)
    uses.update(ref for _, ref in region.outputs)
    live = {ref for _, ref in region.inputs}
    available = set(live)

    def live_bits() -> int:
        return sum(1 if symbols[ref] is Kind.BIT else region.domain.word_bits for ref in live)

    peak_values, peak_bits = len(live), live_bits()
    live.intersection_update(ref for ref in live if uses[ref])
    for name in schedule:
        node = by_id[name]
        if any(arg not in available for arg in node.args):
            raise ContractError("schedule violates topological dependencies")
        produced = tuple(ref for ref, _ in node.ports)
        live.update(produced)
        available.update(produced)
        peak_values, peak_bits = max(peak_values, len(live)), max(peak_bits, live_bits())
        for arg in node.args:
            uses[arg] -= 1
            if not uses[arg]:
                live.discard(arg)
        for ref in produced:
            if not uses[ref]:
                live.discard(ref)
    n = region.domain.limbs
    return StaticDiagnostics(tuple(sorted(Counter(node.op.value for node in region.nodes).items())),
                             max(depths.values()), max(depths[ref] for _, ref in region.outputs),
                             peak_values, peak_bits, schedule,
                             (("limb_divmod", 2 * n),),
                             (("limb_shift", n), ("limb_sum_term", n), ("fixed_width_bytes", 1)))
