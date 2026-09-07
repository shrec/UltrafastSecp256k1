"""Independent, bounded carry-structure diagnostic; no performance candidate.

Truth-table enumeration and finite reblocking checks are OBSERVED evidence, not
a full-width proof, production equivalence, or a constant-time implementation.
See CARRY_STRUCTURE.md for the separate mathematical derivation and its scope.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
from itertools import combinations, product
import json
from pathlib import Path


class ContractError(ValueError):
    """Malformed domain, digit state, transfer, or requested exhaustive bound."""


class DiagnosticMismatch(AssertionError):
    """The finite diagnostic found a violation; never silently emit success."""


def _integer(value: object, low: int, high: int, label: str) -> None:
    if type(value) is not int or not low <= value <= high:
        raise ContractError(f"{label}: exact integer in [{low}, {high}] required")


def _word_bits(word_bits: object) -> None:
    _integer(word_bits, 1, 64, "word_bits")


def _pair(value: object, label: str) -> None:
    if type(value) is not tuple or len(value) != 2:
        raise ContractError(f"{label}: immutable pair required")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DiagnosticMismatch(message)


@dataclass(frozen=True)
class Transfer:
    """Outputs for BOTH possible inputs, in (C(0), C(1)) order."""

    outputs: tuple[int, int]

    def __post_init__(self) -> None:
        _pair(self.outputs, "truth table")
        for value in self.outputs:
            _integer(value, 0, 1, "truth table entry")
        if self.outputs not in ((0, 0), (0, 1), (1, 1)):
            raise ContractError("nonmonotone carry transfer is not attainable")

    @property
    def name(self) -> str:
        return {(0, 0): "K", (0, 1): "P", (1, 1): "G"}[self.outputs]

    def apply(self, carry_in: int) -> int:
        _integer(carry_in, 0, 1, "carry_in")
        return self.outputs[carry_in]


K = Transfer((0, 0))
P = Transfer((0, 1))
G = Transfer((1, 1))
TRANSFERS = (K, P, G)


def _transfer(value: object) -> None:
    if type(value) is not Transfer:
        raise ContractError("exact Transfer required")


def from_gp(generate: int, propagate: int) -> Transfer:
    """Four encodings are permitted here; (1, 0) and (1, 1) both encode G."""
    _integer(generate, 0, 1, "generate")
    _integer(propagate, 0, 1, "propagate")
    return Transfer(tuple(generate | (propagate & c) for c in (0, 1)))


def compose(high: Transfer, low: Transfer) -> Transfer:
    """High segment AFTER low segment, not the reverse significance order."""
    _transfer(high)
    _transfer(low)
    return Transfer(tuple(high.apply(low.apply(c)) for c in (0, 1)))


def compose_gp(high: tuple[int, int], low: tuple[int, int]) -> tuple[int, int]:
    """Boolean formula checked separately against direct truth composition."""
    _pair(high, "high g/p")
    _pair(low, "low g/p")
    for value in high + low:
        _integer(value, 0, 1, "g/p bit")
    gh, ph = high
    gl, pl = low
    return gh | (ph & gl), ph & pl


def digit_summary(a: int, b: int, word_bits: int) -> tuple[int, Transfer]:
    """Return raw low AND transfer; the transfer alone cannot recover low."""
    _word_bits(word_bits)
    radix = 1 << word_bits
    _integer(a, 0, radix - 1, "a digit")
    _integer(b, 0, radix - 1, "b digit")
    generate, raw_low = divmod(a + b, radix)
    propagate = int(raw_low == radix - 1)
    return raw_low, from_gp(generate, propagate)


def _digits(a: object, b: object, word_bits: int) -> None:
    _word_bits(word_bits)
    if type(a) is not tuple or type(b) is not tuple:
        raise ContractError("immutable little-endian digit tuples required")
    if not 1 <= len(a) <= 4 or len(a) != len(b):
        raise ContractError("equal digit counts in [1, 4] required")
    for digit in a + b:
        _integer(digit, 0, (1 << word_bits) - 1, "digit")


def block_summary(a: tuple[int, ...], b: tuple[int, ...], word_bits: int) -> Transfer:
    """Only a block's boundary carry response, not its complete result state."""
    _digits(a, b, word_bits)
    summary = P
    for ai, bi in zip(a, b):
        _, transfer = digit_summary(ai, bi, word_bits)
        summary = compose(transfer, summary)
    return summary


def partitions(limbs: int) -> tuple[tuple[int, ...], ...]:
    """All ordered contiguous reblockings; no digit permutation is permitted."""
    _integer(limbs, 1, 4, "limbs")
    result = []
    for cuts in range(1 << (limbs - 1)):
        lengths = []
        length = 1
        for boundary in range(limbs - 1):
            if cuts & (1 << boundary):
                lengths.append(length)
                length = 1
            else:
                length += 1
        lengths.append(length)
        result.append(tuple(lengths))
    return tuple(result)


@dataclass(frozen=True)
class AdditionOutput:
    low_limbs: tuple[int, ...]
    carry_out: int
    word_bits: int

    def __post_init__(self) -> None:
        _digits(self.low_limbs, self.low_limbs, self.word_bits)
        _integer(self.carry_out, 0, 1, "carry_out")

    @property
    def low(self) -> int:
        return sum(digit << (i * self.word_bits) for i, digit in enumerate(self.low_limbs))

    @property
    def low_bytes(self) -> bytes:
        return self.low.to_bytes((self.word_bits * len(self.low_limbs) + 7) // 8, "little")

    @property
    def total(self) -> int:
        return self.low + (self.carry_out << (self.word_bits * len(self.low_limbs)))


def add_reblocked(a: tuple[int, ...], b: tuple[int, ...], carry_in: int,
                  word_bits: int, block_lengths: tuple[int, ...]) -> AdditionOutput:
    """Reconstruct every low digit while composing ordered block transfers.

    Raw residuals remain available inside each block. This Python diagnostic is
    deliberately sequential: it measures no speed, and these tiny checks would
    gain no evidence from concurrent execution. It is not a parallel scan kernel.
    """
    _digits(a, b, word_bits)
    _integer(carry_in, 0, 1, "carry_in")
    if type(block_lengths) is not tuple or not 1 <= len(block_lengths) <= len(a):
        raise ContractError("immutable, nonempty block-length tuple required")
    for length in block_lengths:
        _integer(length, 1, len(a), "block length")
    if sum(block_lengths) != len(a):
        raise ContractError("block lengths must cover each digit exactly once")
    radix = 1 << word_bits
    raw = tuple(digit_summary(ai, bi, word_bits) for ai, bi in zip(a, b))
    prefix = P
    output = []
    start = 0
    for length in block_lengths:
        stop = start + length
        summary = block_summary(a[start:stop], b[start:stop], word_bits)
        local_carry = prefix.apply(carry_in)
        for residual, transfer in raw[start:stop]:
            output.append((residual + local_carry) % radix)
            local_carry = transfer.apply(local_carry)
        prefix = compose(summary, prefix)
        _require(local_carry == prefix.apply(carry_in), "block carry mismatch")
        start = stop
    return AdditionOutput(tuple(output), prefix.apply(carry_in), word_bits)


def naive_low_only_combine(left: tuple[int, int], right: tuple[int, int],
                           word_bits: int) -> tuple[int, int]:
    """INTENTIONALLY LOSSY negative control: ignores both stored high parts."""
    _word_bits(word_bits)
    radix = 1 << word_bits
    for state in (left, right):
        _pair(state, "lossy (low, carry) state")
        _integer(state[0], 0, radix - 1, "low")
        _integer(state[1], 0, 1, "carry")
    carry, low = divmod(left[0] + right[0], radix)
    return low, carry


def _groupings(a: int, b: int, c: int, word_bits: int) -> tuple[tuple[int, int], tuple[int, int]]:
    left = naive_low_only_combine(naive_low_only_combine((a, 0), (b, 0), word_bits),
                                  (c, 0), word_bits)
    right = naive_low_only_combine((a, 0),
                                   naive_low_only_combine((b, 0), (c, 0), word_bits), word_bits)
    return left, right


def negative_controls() -> dict:
    bits, radix = 2, 4
    nonassociative = 0
    needs_high_two = 0
    for a, b, c in product(range(radix), repeat=3):
        left, right = _groupings(a, b, c, bits)
        _require(left[0] == right[0] == (a + b + c) % radix,
                 "even low-only modulo associativity failed")
        nonassociative += left != right
        needs_high_two += (a + b + c) // radix == 2
    left, right = _groupings(1, 1, 3, bits)
    _require(left != right, "nonassociativity negative control disappeared")
    all_max = _groupings(3, 3, 3, bits)
    _require(all_max == ((1, 1), (1, 1)), "high-width negative control changed")
    raw0, summary0 = digit_summary(0, 0, bits)
    raw1, summary1 = digit_summary(0, 1, bits)
    _require(summary0 == summary1 and raw0 != raw1, "summary-only loss control disappeared")
    return {
        "claim_class": "OBSERVED",
        "radix": radix,
        "triple_cases": radix ** 3,
        "low_modulo_associativity_cases": radix ** 3,
        "lossy_tuple_nonassociativity_cases": nonassociative,
        "exact_sum_requires_high_two_cases": needs_high_two,
        "nonassociativity_witness": {"operands": [1, 1, 3], "left": left, "right": right},
        "insufficient_high_witness": {
            "operands": [3, 3, 3], "exact_low_high": [1, 2],
            "naive_left": all_max[0], "naive_right": all_max[1],
            "note": "A state-width failure, not a nonassociativity witness.",
        },
        "summary_only_loses_low": {
            "pairs": [[0, 0], [0, 1]], "carry_in": 0,
            "same_summary": summary0.name, "different_low": [raw0, raw1],
        },
    }


def _algebra_report() -> dict:
    encodings = tuple(product((0, 1), repeat=2))
    for high, low in product(encodings, repeat=2):
        _require(from_gp(*compose_gp(high, low)) == compose(from_gp(*high), from_gp(*low)),
                 "Boolean formula differs from high-after-low truth composition")
    for high, middle, low in product(TRANSFERS, repeat=3):
        _require(compose(high, compose(middle, low)) == compose(compose(high, middle), low),
                 "carry-transfer associativity failed")
    for transfer in TRANSFERS:
        _require(compose(P, transfer) == compose(transfer, P) == transfer, "identity failed")
    _require(compose(K, G) != compose(G, K), "noncommutativity control disappeared")
    witnesses = []
    for first, second in combinations(TRANSFERS, 2):
        cin = next(c for c in (0, 1) if first.apply(c) != second.apply(c))
        witnesses.append({"states": [first.name, second.name], "carry_in": cin,
                          "outputs": [first.apply(cin), second.apply(cin)]})
    return {
        "claim_class": "OBSERVED",
        "gp_encodings": [{"gp": gp, "truth_table": from_gp(*gp).outputs,
                          "state": from_gp(*gp).name} for gp in encodings],
        "composition_direction": "high_after_low",
        "composition_table": [{"high": high.name, "low": low.name,
                               "result": compose(high, low).name}
                              for high, low in product(TRANSFERS, repeat=2)],
        "gp_composition_cases": 16,
        "associativity_cases": 27,
        "two_sided_identity_cases": 3,
        "identity": "P",
        "noncommutativity_witness": {"K_after_G": compose(K, G).name,
                                     "G_after_K": compose(G, K).name},
        "pairwise_distinguishability": witnesses,
        "minimality_scope": "Only carry response for BOTH possible incoming bits; not low digits, "
                            "full arithmetic state, or known-carry serial state.",
    }


def exhaustive_report(max_word_bits: int = 4, max_total_bits: int = 6,
                      max_limbs: int = 4) -> dict:
    """Exhaust tiny domains, capped at six total bits per input operand."""
    _integer(max_word_bits, 1, 4, "max_word_bits")
    _integer(max_total_bits, 1, 6, "max_total_bits")
    _integer(max_limbs, 1, 4, "max_limbs")
    digit_reports = []
    for bits in range(1, max_word_bits + 1):
        radix = 1 << bits
        counts: Counter = Counter()
        for a, b in product(range(radix), repeat=2):
            # Discover from integer addition first, independently of the GP form.
            truth = tuple((a + b + cin) // radix for cin in (0, 1))
            actual = Transfer(truth)
            raw_low, encoded = digit_summary(a, b, bits)
            _require(actual == encoded, "digit GP encoding mismatch")
            for cin in (0, 1):
                _require((raw_low + cin) % radix + radix * encoded.apply(cin) == a + b + cin,
                         "digit reconstruction mismatch")
            counts[actual.name] += 1
        _require(set(counts) == {"K", "P", "G"}, "not exactly three observed transfer states")
        _require(counts == {"K": radix * (radix - 1) // 2, "P": radix,
                            "G": radix * (radix - 1) // 2}, "digit-state count mismatch")
        digit_reports.append({"word_bits": bits, "ordered_digit_pairs": radix * radix,
                              "incoming_carry_evaluations": 2 * radix * radix,
                              "state_counts": dict(sorted(counts.items()))})
    layouts = []
    output_digest = hashlib.sha256()
    for bits in range(1, max_word_bits + 1):
        for limbs in range(1, max_limbs + 1):
            width = bits * limbs
            if width > max_total_bits:
                continue
            limit = 1 << width
            mask = (1 << bits) - 1
            blocks = partitions(limbs)
            cases = 0
            evaluations = 0
            for a, b in product(range(limit), repeat=2):
                aa = tuple((a >> (i * bits)) & mask for i in range(limbs))
                bb = tuple((b >> (i * bits)) & mask for i in range(limbs))
                for cin in (0, 1):
                    expected_carry, expected_low = divmod(a + b + cin, limit)
                    expected_limbs = tuple((expected_low >> (i * bits)) & mask for i in range(limbs))
                    expected_bytes = expected_low.to_bytes((width + 7) // 8, "little")
                    for partition in blocks:
                        actual = add_reblocked(aa, bb, cin, bits, partition)
                        _require((actual.low_limbs, actual.carry_out, actual.low, actual.low_bytes,
                                  actual.total) == (expected_limbs, expected_carry, expected_low,
                                                    expected_bytes, a + b + cin),
                                 f"full-boundary mismatch: {(bits, limbs, a, b, cin, partition)}")
                        record = [bits, limbs, a, b, cin, partition, actual.low_limbs,
                                  actual.carry_out, actual.low_bytes.hex()]
                        output_digest.update((json.dumps(record, separators=(",", ":")) + "\n").encode())
                        evaluations += 1
                    cases += 1
            layouts.append({"word_bits": bits, "limbs": limbs, "total_bits": width,
                            "input_triples": cases, "contiguous_partitions": blocks,
                            "full_output_evaluations": evaluations, "mismatches": 0})
    return {
        "format": "parseatlas_carry_structure_v1",
        "claim_class": "OBSERVED",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bounds": {"max_word_bits": max_word_bits, "max_total_bits": max_total_bits,
                   "max_limbs": max_limbs, "exhaustive_seed": None},
        "scope": "Unsigned L0 two-operand addition plus incoming carry; no Fp/Fn modulus.",
        "digit_enumerations": digit_reports,
        "algebra": _algebra_report(),
        "layouts": layouts,
        "input_triples_across_layouts": sum(row["input_triples"] for row in layouts),
        "full_output_evaluations": sum(row["full_output_evaluations"] for row in layouts),
        "output_records_sha256": output_digest.hexdigest(),
        "negative_controls": negative_controls(),
        "runtime": "PENDING IMPORT",
        "limits": ["Different layouts repeat some mathematical input triples; counts are not unique inputs.",
                   "No 64/256-bit full-domain observation, formal proof engine, or production differential check.",
                   "No CPU timing, instruction/resource measurement, CT eligibility, novelty, or method ranking.",
                   "Transfer summaries preserve carry behavior only; reblocking also retains raw residual digits.",
                   "Known-carry serial addition needs one current carry bit, not a three-state transfer summary."],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", required=True)
    parser.add_argument("--max-word-bits", type=int, default=4)
    parser.add_argument("--max-total-bits", type=int, default=6)
    parser.add_argument("--max-limbs", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        result = exhaustive_report(args.max_word_bits, args.max_total_bits, args.max_limbs)
    except ContractError as exc:
        parser.error(str(exc))
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
