"""Offline inventory validation. This does NOT execute/prove C++ arithmetic."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

from jsonschema import Draft202012Validator, ValidationError
import pytest


ROOT = Path(__file__).resolve().parents[1]
# ROOT is repo/experiments/parseatlas_secp256k1, not repo/experiments.
REPO_ROOT = ROOT.parents[1]
REVISION = "fef231d4e4173bd016fb2a3a1eff67087396a203"
REQUIRED_NODES = {
    "word64": (0, "word"),
    "fp_4x64": (1, "Fp"),
    "fp_5x52": (1, "Fp"),
    "fn_4x64": (1, "Fn"),
    "point": (2, "group"),
    "scalar_mul": (3, "group"),
    "dual_scalar_mul": (3, "group"),
    "msm": (3, "group"),
    "ecdsa_verify": (4, "protocol"),
    "schnorr_batch_verify": (4, "protocol"),
}
REQUIRED_EDGES = {
    ("word64", "fp_4x64", "consumed_by"),
    ("word64", "fn_4x64", "consumed_by"),
    ("fp_4x64", "fp_5x52", "converts_to"),
    ("fp_5x52", "fp_4x64", "converts_to"),
    ("fp_4x64", "point", "consumed_by"),
    ("fp_5x52", "point", "consumed_by"),
    ("fn_4x64", "scalar_mul", "consumed_by"),
    ("point", "scalar_mul", "consumed_by"),
    ("point", "dual_scalar_mul", "consumed_by"),
    ("point", "msm", "consumed_by"),
    ("fn_4x64", "msm", "consumed_by"),
    ("fn_4x64", "ecdsa_verify", "consumed_by"),
    ("dual_scalar_mul", "ecdsa_verify", "consumed_by"),
    ("msm", "schnorr_batch_verify", "consumed_by"),
}

# Frozen role bindings are deliberate inventory fixtures, not learned from the
# JSON being validated. A new signature/span or role requires a reviewed update.
EXPECTED_WITNESSES = {
  "word_carry": [
    "src/cpu/include/secp256k1/detail/arith64.hpp",
    "add64",
    53,
    71,
    [
      "unsigned __int128 const sum",
      "sum >> 64",
      "sub64",
      "borrow = borrow1 | borrow2"
    ]
  ],
  "word_product": [
    "src/cpu/src/field.cpp",
    "mul64",
    31,
    72,
    [
      "unsigned __int128 const product",
      "product >> 64"
    ]
  ],
  "fp_modulus": [
    "src/cpu/src/field.cpp",
    "PRIME",
    76,
    86,
    [
      "0xFFFFFFFEFFFFFC2FULL",
      "MOD_ADJUST"
    ]
  ],
  "fp_add": [
    "src/cpu/src/field.cpp",
    "add_impl",
    423,
    448,
    [
      "s[0] = add64(a[0], b[0], c1);",
      "c1 | c2",
      "MOD_ADJUST"
    ]
  ],
  "fp_mul_dispatch": [
    "src/cpu/src/field.cpp",
    "mul_impl",
    1136,
    1178,
    [
      "has_bmi2_support() && has_adx_support()",
      "return reduce(mul_wide(a, b));",
      "field_mul_full_asm(a.data(), b.data(), out.data());"
    ]
  ],
  "fp_inverse": [
    "src/cpu/src/field.cpp",
    "FieldElement::inverse",
    3541,
    3554,
    [
      "Inverse of zero not defined",
      "return zero();",
      "fe_inverse_safegcd_impl(*this)",
      "field_safegcd30::inverse_impl(*this)"
    ]
  ],
  "fp_bytes": [
    "src/cpu/src/field.cpp",
    "FieldElement::to_bytes",
    2517,
    2526,
    [
      "limbs_[3 - i]",
      "limb >> (56 - 8 * j)"
    ]
  ],
  "fp_raw_boundary": [
    "src/cpu/include/secp256k1/field.hpp",
    "from_limbs_raw",
    62,
    70,
    [
      "fe.limbs_ = limbs;",
      "return fe;"
    ]
  ],
  "fp52_mul": [
    "src/cpu/include/secp256k1/field_52_impl.hpp",
    "FieldElement52::operator*",
    2438,
    2443,
    [
      "fe52_mul_inner(r.n, n, rhs.n);"
    ]
  ],
  "fp52_lazy": [
    "src/cpu/include/secp256k1/field_52_impl.hpp",
    "FieldElement52::operator+",
    2494,
    2527,
    [
      "r.n[4] = n[4] + rhs.n[4];",
      "FieldElement52::negate",
      "r.n[0] = m1 * P0 - n[0];"
    ]
  ],
  "fp52_weak": [
    "src/cpu/include/secp256k1/field_52_impl.hpp",
    "fe52_normalize_weak",
    2410,
    2430,
    [
      "t0 += x * 0x1000003D1ULL;",
      "t4 += (t3 >> 52);"
    ]
  ],
  "fp52_full": [
    "src/cpu/include/secp256k1/field_52_impl.hpp",
    "fe52_normalize_inline",
    2636,
    2668,
    [
      "t0 >= 0xFFFFEFFFFFC2FULL",
      "t4 &= M48;"
    ]
  ],
  "fp52_from": [
    "src/cpu/include/secp256k1/field_52_impl.hpp",
    "FieldElement52::from_fe",
    2744,
    2754,
    [
      "const auto& L = fe.limbs();",
      "r.n[4] =  L[3] >> 16;"
    ]
  ],
  "fp52_to": [
    "src/cpu/include/secp256k1/field_52_impl.hpp",
    "FieldElement52::to_fe",
    2758,
    2769,
    [
      "fe52_normalize_inline(tmp.n);",
      "return FieldElement::from_limbs_raw(L);"
    ]
  ],
  "fp52_inverse_boundary": [
    "src/cpu/src/field_52.cpp",
    "FieldElement52::inverse",
    111,
    126,
    [
      "SECP256K1_HYBRID_4X64_ACTIVE",
      "fe52_normalize_and_pack_4x64(n, a);",
      "field_sqr_full_asm(a, x2);"
    ]
  ],
  "fp52_layout": [
    "src/cpu/include/secp256k1/field_52.hpp",
    "FieldElement52",
    54,
    100,
    [
      "std::uint64_t n[5];",
      "void normalize_weak() noexcept;",
      "void normalize() noexcept;"
    ]
  ],
  "fp4_inline_kernel": [
    "src/cpu/include/secp256k1/field_4x64_inline.hpp",
    "void mul",
    22,
    76,
    [
      "defined(__ADX__) && defined(__BMI2__)",
      "__restrict__ r",
      "__asm__ __volatile__"
    ]
  ],
  "fn_modulus": [
    "src/cpu/src/scalar.cpp",
    "ORDER",
    19,
    27,
    [
      "0xBFD25E8CD0364141ULL",
      "0xBAAEDCE6AF48A03BULL"
    ]
  ],
  "fn_add": [
    "src/cpu/src/scalar.cpp",
    "add_impl",
    83,
    103,
    [
      "sum[i] = add64(a[i], b[i], carry);",
      "reduced[i] = sub64(sum[i], ORDER[i], borrow);"
    ]
  ],
  "fn_mul": [
    "src/cpu/src/scalar.cpp",
    "Scalar::operator*",
    342,
    445,
    [
      "#ifndef SECP256K1_NO_INT128",
      "0x402DA1732FC9BEBFULL",
      "const unsigned __int128 p",
      "extract_to(l6);"
    ]
  ],
  "fn_inverse": [
    "src/cpu/src/scalar.cpp",
    "Scalar::inverse",
    893,
    897,
    [
      "if (is_zero()) return Scalar::zero();",
      "scalar_safegcd::inverse_impl(limbs_)"
    ]
  ],
  "fn_bytes": [
    "src/cpu/src/scalar.cpp",
    "Scalar::to_bytes",
    278,
    289,
    [
      "write_bytes(out.data());",
      "store_be64(out32 + 24, limbs_[0]);"
    ]
  ],
  "fn_layout": [
    "src/cpu/include/secp256k1/scalar.hpp",
    "class Scalar",
    14,
    41,
    [
      "std::array<std::uint64_t, 4>",
      "parse_bytes_strict_nonzero"
    ]
  ],
  "point_layout": [
    "src/cpu/include/secp256k1/point.hpp",
    "FieldElement52",
    14,
    32,
    [
      "SECP256K1_FE52_COMPUTE",
      "SECP256K1_FAST_52BIT"
    ]
  ],
  "point_add": [
    "src/cpu/src/point.cpp",
    "Point::add",
    1869,
    1947,
    [
      "dx52.to_fe();",
      "dx.inverse();",
      "FieldElement const dx = other.x_ - x_;",
      "jac52_add(p52, q52);",
      "jacobian_add(p, q);"
    ]
  ],
  "point_double": [
    "src/cpu/src/point.cpp",
    "Point::dbl",
    1949,
    1974,
    [
      "jac52_double_z1_to",
      "jac52_double_to",
      "jacobian_double(p)"
    ]
  ],
  "single_mul": [
    "src/cpu/src/point.cpp",
    "Point::scalar_mul",
    2623,
    2791,
    [
      "glv_decompose(scalar)",
      "result.dbl_inplace();",
      "result.add_inplace(base);",
      "scalar_mul_glv52(*this, scalar)",
      "r.normalize();"
    ]
  ],
  "dual_fe52": [
    "src/cpu/src/point.cpp",
    "Point::dual_scalar_mul_gen_point",
    4379,
    4408,
    [
      "SECP256K1_FE52_COMPUTE",
      "a.limbs()",
      "glv_decompose(b)",
      "aG.add_inplace(P.scalar_mul(b));"
    ]
  ],
  "dual_embedded": [
    "src/cpu/src/point.cpp",
    "Point::dual_scalar_mul_gen_point",
    4726,
    4757,
    [
      "SECP256K1_PLATFORM_ESP32",
      "glv_decompose(a)",
      "compute_wnaf_into(decomp_b.k2"
    ]
  ],
  "dual_fallback": [
    "src/cpu/src/point.cpp",
    "Point::dual_scalar_mul_gen_point",
    4925,
    4933,
    [
      "Point::generator().scalar_mul(a);",
      "aG.add_inplace(P.scalar_mul(b));"
    ]
  ],
  "msm_digits": [
    "src/cpu/src/pippenger.cpp",
    "extract_digit",
    80,
    100,
    [
      "auto const& limbs = s.limbs();",
      "limbs[limb_idx] >> bit_idx"
    ]
  ],
  "msm_entry": [
    "src/cpu/src/pippenger.cpp",
    "pippenger_msm",
    479,
    496,
    [
      "n == 0",
      "points[0].scalar_mul(scalars[0])",
      "n < 48",
      "pippenger_core"
    ]
  ],
  "msm_dispatch": [
    "src/cpu/src/pippenger.cpp",
    "Point msm",
    652,
    669,
    [
      "kStraussCrossover = 48",
      "kStraussCrossover = 60",
      "multi_scalar_mul(scalars, points, n)",
      "pippenger_msm_glv(scalars, points, n)"
    ]
  ],
  "ecdsa_verify_call": [
    "src/cpu/src/ecdsa.cpp",
    "ecdsa_verify",
    890,
    913,
    [
      "Scalar::from_bytes(msg_hash32)",
      "sig.s.inverse()",
      "auto u1 = z * w;",
      "Point::dual_scalar_mul_gen_point(u1, u2, public_key)"
    ]
  ],
  "ecdsa_coordinate_check": [
    "src/cpu/src/ecdsa.cpp",
    "ecdsa_check_xcoord",
    798,
    855,
    [
      "FE52::from_4x64_limbs(sig.r.limbs().data())",
      "R_prime.Z52().square()",
      "FieldElement::from_limbs(sig.r.limbs())"
    ]
  ],
  "ecdsa_ct_handoff": [
    "src/cpu/src/ecdsa.cpp",
    "ecdsa_sign_verified",
    724,
    738,
    [
      "ct::ecdsa_sign(msg_hash, private_key)",
      "ct::generator_mul(private_key)"
    ]
  ],
  "batch_consumer": [
    "src/cpu/src/batch_verify.cpp",
    "schnorr_batch_verify_impl",
    361,
    463,
    [
      "n <= kSchnorrBatchIndividualCutoff",
      "secp256k1::detail::csprng_fill",
      "std::size_t const msm_n = 2 * n;",
      "auto G_term = Point::generator().scalar_mul(g_coeff);",
      "auto rest = msm(scalars, points, msm_n);",
      "auto result = G_term.add(rest);"
    ]
  ],
  "test_fp": [
    "src/cpu/tests/test_comprehensive.cpp",
    "test_field_arith",
    99,
    211,
    [
      "a*(b+c)==a*b+a*c",
      "a^2==a*a"
    ]
  ],
  "test_fn": [
    "src/cpu/tests/test_comprehensive.cpp",
    "test_scalar_arith",
    675,
    768,
    [
      "s: a*(b+c)==a*b+a*c",
      "s: a*a^-^1==1"
    ]
  ],
  "test_point": [
    "src/cpu/tests/test_comprehensive.cpp",
    "test_point_basic",
    968,
    1060,
    [
      "G on curve",
      "dbl(G)==G+G"
    ]
  ],
  "test_msm": [
    "src/cpu/tests/test_comprehensive.cpp",
    "test_msm",
    1825,
    1910,
    [
      "secp256k1::pippenger_msm(scalars, points)",
      "Pippenger(n=64)",
      "Pippenger(n=256)"
    ]
  ],
  "test_ecdsa": [
    "src/cpu/tests/test_comprehensive.cpp",
    "test_ecdsa",
    2050,
    2117,
    [
      "ECDSA sign+verify",
      "ECDSA wrong msg fails"
    ]
  ],
  "history_fp_costs": [
    "experiments/representation_search/README.md",
    "FE52 operation costs",
    216,
    251,
    [
      "21.52",
      "18.81",
      "ILP saturates at two field multiplies"
    ]
  ],
  "history_point_negative": [
    "experiments/representation_search/README.md",
    "What the measurements establish",
    287,
    311,
    [
      "No representation beats the production formulas",
      "inside the noise",
      "not a speedup"
    ]
  ],
  "history_wins": [
    "experiments/representation_search/README.md",
    "Confirmed wins",
    319,
    332,
    [
      "co-Z coordinate change",
      "CT SafeGCD inverse",
      "macro-guarded"
    ]
  ],
  "history_refutations": [
    "experiments/representation_search/README.md",
    "Refuted or below the noise floor",
    342,
    361,
    [
      "Limb schedule, more columns",
      "Scalar reduction mod n (int128)",
      "AVX2 vectorisation",
      "Strauss vs Pippenger"
    ]
  ],
  "history_open_table": [
    "experiments/representation_search/README.md",
    "Still open on x86",
    363,
    370,
    [
      "ConnectBlock on a quiet frequency-locked machine",
      "1280 KB"
    ]
  ],
  "history_doc_conflict": [
    "experiments/representation_search/README.md",
    "Status",
    403,
    420,
    [
      "nothing here is a measurement",
      "No number in this document came from running code on hardware"
    ]
  ],
  "history_slice": [
    "experiments/representation_search/README.md",
    "Slices are",
    440,
    484,
    [
      "off the slice (Z1 free): agreed  0/64",
      "on the slice  (Z1 = 1) : agreed 64/64",
      "Extra outputs are reported"
    ]
  ],
  "history_model_fix": [
    "experiments/representation_search/README.md",
    "Defect 1:",
    493,
    524,
    [
      "sub` now costs `neg + add",
      "reverses both"
    ]
  ],
  "history_harness_fix": [
    "experiments/representation_search/README.md",
    "Defect 2:",
    526,
    551,
    [
      "doubling family had never actually been compared",
      "raw gate         34 agreeing pairs",
      "projective gate  74 agreeing pairs",
      "Z == 1"
    ]
  ]
}
EXPECTED_NODE_WITNESSES = {
  "word64": [
    "word_carry",
    "word_product"
  ],
  "fp_4x64": [
    "fp_modulus",
    "fp_add",
    "fp_mul_dispatch",
    "fp_inverse",
    "fp_bytes",
    "fp_raw_boundary",
    "fp4_inline_kernel"
  ],
  "fp_5x52": [
    "fp52_layout",
    "fp52_mul",
    "fp52_lazy",
    "fp52_weak",
    "fp52_full",
    "fp52_from",
    "fp52_to",
    "fp52_inverse_boundary"
  ],
  "fn_4x64": [
    "fn_layout",
    "fn_modulus",
    "fn_add",
    "fn_mul",
    "fn_inverse",
    "fn_bytes"
  ],
  "point": [
    "point_layout",
    "point_add",
    "point_double"
  ],
  "scalar_mul": [
    "single_mul"
  ],
  "dual_scalar_mul": [
    "dual_fe52",
    "dual_embedded",
    "dual_fallback"
  ],
  "msm": [
    "msm_digits",
    "msm_entry",
    "msm_dispatch"
  ],
  "ecdsa_verify": [
    "ecdsa_verify_call",
    "ecdsa_coordinate_check",
    "ecdsa_ct_handoff"
  ],
  "schnorr_batch_verify": [
    "batch_consumer"
  ]
}
EXPECTED_EDGE_WITNESSES = {
    ("word64", "fp_4x64", "consumed_by"): ["fp_add","word_carry"],
    ("word64", "fn_4x64", "consumed_by"): ["fn_add","word_carry"],
    ("fp_4x64", "fp_5x52", "converts_to"): ["fp52_from"],
    ("fp_5x52", "fp_4x64", "converts_to"): ["fp52_to"],
    ("fp_4x64", "point", "consumed_by"): ["point_add"],
    ("fp_5x52", "point", "consumed_by"): ["point_add","point_double"],
    ("fn_4x64", "scalar_mul", "consumed_by"): ["single_mul"],
    ("point", "scalar_mul", "consumed_by"): ["single_mul"],
    ("fn_4x64", "dual_scalar_mul", "consumed_by"): ["dual_fe52","dual_embedded"],
    ("point", "dual_scalar_mul", "consumed_by"): ["dual_fe52","dual_fallback"],
    ("point", "msm", "consumed_by"): ["msm_entry"],
    ("fn_4x64", "msm", "consumed_by"): ["msm_digits","msm_dispatch"],
    ("fn_4x64", "ecdsa_verify", "consumed_by"): ["ecdsa_verify_call"],
    ("dual_scalar_mul", "ecdsa_verify", "consumed_by"): ["ecdsa_verify_call"],
    ("fp_5x52", "ecdsa_verify", "consumed_by"): ["ecdsa_coordinate_check"],
    ("scalar_mul", "schnorr_batch_verify", "consumed_by"): ["batch_consumer"],
    ("msm", "schnorr_batch_verify", "consumed_by"): ["batch_consumer"],
    ("fn_4x64", "schnorr_batch_verify", "consumed_by"): ["batch_consumer"],
}

EXPECTED_TEST_ANCHORS = {
  "field_checks": [
    [
      "fp_4x64"
    ],
    [
      "test_fp"
    ],
    "NOT_RUN_BY_INVENTORY"
  ],
  "scalar_checks": [
    [
      "fn_4x64"
    ],
    [
      "test_fn"
    ],
    "NOT_RUN_BY_INVENTORY"
  ],
  "point_checks": [
    [
      "point"
    ],
    [
      "test_point"
    ],
    "NOT_RUN_BY_INVENTORY"
  ],
  "msm_checks": [
    [
      "msm"
    ],
    [
      "test_msm"
    ],
    "NOT_RUN_BY_INVENTORY"
  ],
  "ecdsa_checks": [
    [
      "ecdsa_verify"
    ],
    [
      "test_ecdsa"
    ],
    "NOT_RUN_BY_INVENTORY"
  ]
}
EXPECTED_HISTORY = {
  "fp_cost_calibration": [
    "experiments/representation_search/README.md",
    [
      "fp_5x52"
    ],
    [
      "history_fp_costs"
    ],
    "REPORTED_MEASUREMENT"
  ],
  "point_formula_refutations": [
    "experiments/representation_search/README.md",
    [
      "point"
    ],
    [
      "history_point_negative"
    ],
    "REPORTED_MEASUREMENT"
  ],
  "coz_prior_win": [
    "experiments/representation_search/README.md",
    [
      "dual_scalar_mul",
      "ecdsa_verify",
      "point"
    ],
    [
      "history_wins"
    ],
    "REPORTED_MEASUREMENT"
  ],
  "inverse_prior_win": [
    "experiments/representation_search/README.md",
    [
      "fp_4x64",
      "fp_5x52"
    ],
    [
      "history_wins"
    ],
    "REPORTED_MEASUREMENT"
  ],
  "word_schedule_refutation": [
    "experiments/representation_search/README.md",
    [
      "fp_4x64",
      "fp_5x52",
      "word64"
    ],
    [
      "history_refutations"
    ],
    "REPORTED_MEASUREMENT"
  ],
  "fn_reduction_prior_limit": [
    "experiments/representation_search/README.md",
    [
      "fn_4x64"
    ],
    [
      "history_refutations"
    ],
    "REPORTED_MODEL_OR_LIMIT"
  ],
  "msm_prior_limit": [
    "experiments/representation_search/README.md",
    [
      "msm"
    ],
    [
      "history_refutations"
    ],
    "REPORTED_MODEL_OR_LIMIT"
  ],
  "dual_table_open_question": [
    "experiments/representation_search/README.md",
    [
      "dual_scalar_mul",
      "ecdsa_verify"
    ],
    [
      "history_open_table"
    ],
    "REPORTED_MODEL_OR_LIMIT"
  ],
  "documentation_conflict": [
    "experiments/representation_search/README.md",
    [
      "fn_4x64",
      "fp_4x64",
      "fp_5x52",
      "msm",
      "point",
      "word64"
    ],
    [
      "history_doc_conflict"
    ],
    "REPORTED_MODEL_OR_LIMIT"
  ],
  "cost_model_correction": [
    "experiments/representation_search/README.md",
    [
      "fp_5x52",
      "point"
    ],
    [
      "history_model_fix"
    ],
    "REPORTED_MODEL_OR_LIMIT"
  ],
  "slice_and_harness_controls": [
    "experiments/representation_search/README.md",
    [
      "dual_scalar_mul",
      "point"
    ],
    [
      "history_harness_fix",
      "history_slice"
    ],
    "REPORTED_MODEL_OR_LIMIT"
  ]
}


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)


def _index_unique(items, key):
    result = {}
    for item in items:
        identity = item[key]
        if identity in result:
            raise ValueError(f"duplicate {key}: {identity}")
        result[identity] = item
    return result


def _read_pinned_source(repo_root, source):
    relative = PurePosixPath(source["path"])
    if relative.is_absolute() or ".." in relative.parts or str(relative) != source["path"]:
        raise ValueError("noncanonical source path")
    target = repo_root / relative
    # A symlink must not smuggle an unrelated tree in as active source evidence.
    if target.resolve() != target or not target.is_file():
        raise ValueError(f"missing or redirected source: {relative}")
    active = target.read_bytes()
    if hashlib.sha256(active).hexdigest() != source["sha256"]:
        raise ValueError(f"active source hash mismatch: {relative}")
    try:
        pinned = subprocess.run(
            ["git", "show", f"{REVISION}:{relative}"], cwd=repo_root,
            capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError as error:
        raise ValueError(f"frozen reference unavailable: {relative}") from error
    if pinned != active:
        raise ValueError(f"frozen source mismatch: {relative}")
    return source["path"], active.decode("utf-8").splitlines()


def _check_consumption_dag(node_ids, edges):
    """Return direct consumers; reject consumption cycles independently of bindings."""
    adjacency = {identity: set() for identity in node_ids}
    for edge in edges:
        if edge["kind"] == "consumed_by":
            if edge["from"] not in adjacency or edge["to"] not in adjacency:
                raise ValueError("dangling consumption endpoint")
            adjacency[edge["from"]].add(edge["to"])
    visiting, visited = set(), set()

    def visit(identity):
        if identity in visiting:
            raise ValueError("cycle in consumption DAG")
        if identity in visited:
            return
        visiting.add(identity)
        for consumer in sorted(adjacency[identity]):
            visit(consumer)
        visiting.remove(identity)
        visited.add(identity)

    for identity in adjacency:
        visit(identity)
    return adjacency


def validate_graph(graph, repo_root=REPO_ROOT):
    """Fail closed on missing evidence, drift, malformed schema or graph links."""
    schema = _read_json(ROOT / "schemas/primitive_graph.schema.json")
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(graph)
    sources = _index_unique(graph["sources"], "path")
    witnesses = _index_unique(graph["witnesses"], "id")
    nodes = _index_unique(graph["nodes"], "id")
    anchors = _index_unique(graph["test_anchors"], "id")
    history = _index_unique(graph["historical_evidence"], "id")
    if not REQUIRED_NODES.keys() <= nodes.keys():
        raise ValueError("missing mandatory active primitive")
    if nodes.keys() != EXPECTED_NODE_WITNESSES.keys():
        raise ValueError("unregistered node binding")
    if anchors.keys() != EXPECTED_TEST_ANCHORS.keys():
        raise ValueError("test anchor vocabulary mismatch")
    if history.keys() != EXPECTED_HISTORY.keys():
        raise ValueError("historical vocabulary mismatch")
    if witnesses.keys() != EXPECTED_WITNESSES.keys():
        raise ValueError("witness vocabulary mismatch")
    for identity, (layer, domain) in REQUIRED_NODES.items():
        if (nodes[identity]["layer"], nodes[identity]["domain"]) != (layer, domain):
            raise ValueError(f"wrong layer/domain: {identity}")

    # Independent exact-file reads run concurrently; preserve input order and
    # leave one observed CPU available to interactive tooling.
    workers = min(len(sources), max(1, (os.cpu_count() or 1) - 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        lines = dict(pool.map(lambda item: _read_pinned_source(repo_root, item), sources.values()))
    if not EXPECTED_WITNESSES.keys() <= witnesses.keys():
        raise ValueError("missing mandatory witness binding")
    for witness in witnesses.values():
        path = witness["path"]
        if path not in lines:
            raise ValueError(f"witness source unpinned: {path}")
        start, end = witness["start_line"], witness["end_line"]
        if not 1 <= start <= end <= len(lines[path]):
            raise ValueError(f"invalid witness span: {witness['id']}")
        if end - start > 320:
            raise ValueError("witness span too broad")
        body = "\n".join(lines[path][start - 1:end])
        if witness["symbol"] not in body or any(token not in body for token in witness["tokens"]):
            raise ValueError(f"witness token drift: {witness['id']}; symbol={witness['symbol']}; "
                             f"missing={[token for token in witness['tokens'] if token not in body]}")
        expected = EXPECTED_WITNESSES[witness["id"]]
        identity = [path, witness["symbol"], start, end, witness["tokens"]]
        if identity != expected:
            raise ValueError("witness role binding mismatch")

    def require_witnesses(item):
        if not set(item["witness_ids"]) <= witnesses.keys():
            raise ValueError("dangling witness reference")

    for node in nodes.values():
        require_witnesses(node)
        required = EXPECTED_NODE_WITNESSES[node["id"]]
        if set(required) != set(node["witness_ids"]):
            raise ValueError("node witness role mismatch")
    observed_edges = set()
    for edge in graph["edges"]:
        require_witnesses(edge)
        if edge["from"] not in nodes or edge["to"] not in nodes:
            raise ValueError("dangling edge endpoint")
        identity = (edge["from"], edge["to"], edge["kind"])
        if identity in observed_edges:
            raise ValueError("duplicate graph edge")
        observed_edges.add(identity)
        if identity not in EXPECTED_EDGE_WITNESSES:
            raise ValueError("unregistered edge binding")
        expected = EXPECTED_EDGE_WITNESSES[identity]
        if set(expected) != set(edge["witness_ids"]):
            raise ValueError("edge witness role mismatch")
        if edge["kind"] == "converts_to":
            if nodes[edge["from"]]["domain"] != nodes[edge["to"]]["domain"]:
                raise ValueError("conversion does not preserve mathematical domain")
        elif nodes[edge["from"]]["layer"] > nodes[edge["to"]]["layer"]:
            raise ValueError("dependency edge reverses layer order")
    if observed_edges != EXPECTED_EDGE_WITNESSES.keys():
        raise ValueError("missing mandatory dependency/boundary")
    if any(edge["from"] == "msm" and edge["to"] == "ecdsa_verify" for edge in graph["edges"]):
        raise ValueError("single ECDSA uses dual_scalar_mul_gen_point, not msm")
    adjacency = _check_consumption_dag(nodes, graph["edges"])
    for identity, node in nodes.items():
        if set(node["consumers"]) != adjacency[identity]:
            raise ValueError("consumer/edge disagreement")
        test_ids = {anchor_id for anchor_id, anchor in anchors.items()
                    if identity in anchor["node_ids"]}
        if set(node["test_evidence"]["anchor_ids"]) != test_ids:
            raise ValueError("test evidence/anchor disagreement")
        expected_status = "SOURCE_ANCHORS_ONLY" if test_ids else "NO_ANCHOR_IN_SCOPE"
        if node["test_evidence"]["status"] != expected_status:
            raise ValueError("test evidence status mismatch")
        history_ids = {history_id for history_id, item in history.items()
                       if identity in item["node_ids"]}
        if set(node["benchmark_evidence"]["historical_ids"]) != history_ids:
            raise ValueError("benchmark evidence/history disagreement")
        expected_status = "SECONDARY_NOT_IMPORTED" if history_ids else "NOT_LOCATED"
        if node["benchmark_evidence"]["status"] != expected_status:
            raise ValueError("benchmark evidence status mismatch")
    for anchor in graph["test_anchors"]:
        require_witnesses(anchor)
        actual = [sorted(anchor["node_ids"]), sorted(anchor["witness_ids"]), anchor["execution_status"]]
        if actual != EXPECTED_TEST_ANCHORS[anchor["id"]]:
            raise ValueError("test anchor role binding mismatch")
        if not set(anchor["node_ids"]) <= nodes.keys():
            raise ValueError("dangling test node reference")
        if any(not witnesses[identity]["path"].startswith("src/cpu/tests/")
               for identity in anchor["witness_ids"]):
            raise ValueError("test anchor does not cite a test source")
    referenced_test_paths = {
        witnesses[identity]["path"] for anchor in anchors.values()
        for identity in anchor["witness_ids"]
    }
    declared_test_paths = {path for path in sources if path.startswith("src/cpu/tests/")}
    if declared_test_paths != referenced_test_paths:
        raise ValueError("declared test source has no anchor")
    for item in graph["historical_evidence"]:
        require_witnesses(item)
        actual = [item["path"], sorted(item["node_ids"]), sorted(item["witness_ids"]),
                  item["reported_evidence_kind"]]
        if actual != EXPECTED_HISTORY[item["id"]]:
            raise ValueError("historical role binding mismatch")
        if item["path"] not in sources:
            raise ValueError("secondary evidence not pinned")
        if not set(item["node_ids"]) <= nodes.keys():
            raise ValueError("historical node reference missing")
        if any(witnesses[identity]["path"] != item["path"] for identity in item["witness_ids"]):
            raise ValueError("historical span belongs to different source")
    return graph


@pytest.fixture(scope="module")
def graph():
    return _read_json(ROOT / "data/primitive_graph.json")


def test_schema_and_exact_active_frozen_sources(graph):
    validate_graph(graph)


def test_source_root_is_repository_not_parent():
    assert REPO_ROOT / "experiments/parseatlas_secp256k1" == ROOT
    assert (REPO_ROOT / "src/cpu/src/field.cpp").is_file()


@pytest.mark.parametrize("node_id", sorted(REQUIRED_NODES))
def test_missing_mandatory_node_is_rejected(graph, node_id):
    broken = deepcopy(graph)
    broken["nodes"] = [item for item in broken["nodes"] if item["id"] != node_id]
    with pytest.raises((ValueError, ValidationError)):
        validate_graph(broken)


@pytest.mark.parametrize("edge_id", sorted(REQUIRED_EDGES))
def test_missing_mandatory_edge_is_rejected(graph, edge_id):
    broken = deepcopy(graph)
    broken["edges"] = [item for item in broken["edges"]
                       if (item["from"], item["to"], item["kind"]) != edge_id]
    with pytest.raises(ValueError, match="missing mandatory"):
        validate_graph(broken)


@pytest.mark.parametrize("mutation", [
    "unknown_property", "wrong_revision", "pending_node", "empty_evidence",
    "bad_hash", "duplicate_node", "duplicate_source", "duplicate_witness",
    "reversed_span", "out_of_file_span", "invented_token", "unlisted_source",
    "bad_domain", "dangling_edge", "duplicate_edge", "fake_msm_ecdsa",
    "fake_test_status", "history_promoted", "path_escape", "dangling_witness",
])
def test_malformed_or_unsupported_inventory_is_rejected(graph, mutation):
    broken = deepcopy(graph)
    if mutation == "unknown_property":
        broken["performance_gain"] = "fastest"
    elif mutation == "wrong_revision":
        broken["reference_revision"] = "0" * 40
    elif mutation == "pending_node":
        broken["nodes"][0]["status"] = "pending"
    elif mutation == "empty_evidence":
        broken["nodes"][0]["witness_ids"] = []
    elif mutation == "bad_hash":
        broken["sources"][0]["sha256"] = "0" * 64
    elif mutation == "duplicate_node":
        broken["nodes"].append(deepcopy(broken["nodes"][0]))
    elif mutation == "duplicate_source":
        broken["sources"].append(deepcopy(broken["sources"][0]))
    elif mutation == "duplicate_witness":
        broken["witnesses"].append(deepcopy(broken["witnesses"][0]))
    elif mutation == "reversed_span":
        broken["witnesses"][0]["end_line"] = 1
    elif mutation == "out_of_file_span":
        broken["witnesses"][0]["end_line"] = 1000000
    elif mutation == "invented_token":
        broken["witnesses"][0]["tokens"].append("NOT_A_REAL_PRIMITIVE_123456")
    elif mutation == "unlisted_source":
        broken["witnesses"][0]["path"] = "src/cpu/absent.cpp"
    elif mutation == "bad_domain":
        next(item for item in broken["nodes"] if item["id"] == "fn_4x64")["domain"] = "Fp"
    elif mutation == "dangling_edge":
        broken["edges"][0]["to"] = "invented_node"
    elif mutation == "duplicate_edge":
        broken["edges"].append(deepcopy(broken["edges"][0]))
    elif mutation == "fake_msm_ecdsa":
        edge = deepcopy(broken["edges"][0])
        edge.update({"from": "msm", "to": "ecdsa_verify"})
        broken["edges"].append(edge)
    elif mutation == "fake_test_status":
        broken["test_anchors"][0]["execution_status"] = "PASSED"
    elif mutation == "history_promoted":
        broken["historical_evidence"][0]["status"] = "PROVEN"
    elif mutation == "path_escape":
        broken["sources"][0]["path"] = "src/cpu/../../../outside.cpp"
    elif mutation == "dangling_witness":
        broken["nodes"][0]["witness_ids"] = ["missing_witness"]
    with pytest.raises((ValueError, ValidationError)):
        validate_graph(broken)


def test_absent_repository_is_not_pending_success(graph, tmp_path):
    with pytest.raises(ValueError, match="missing or redirected source"):
        validate_graph(graph, repo_root=tmp_path)


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="duplicate JSON key"):
        json.loads('{"x": 1, "x": 2}', object_pairs_hook=_unique_object)


def test_changed_active_source_cannot_repin_itself(graph, tmp_path, monkeypatch):
    source = deepcopy(graph["sources"][0])
    original = (REPO_ROOT / source["path"]).read_bytes()
    changed = original + b"\n// forged evidence\n"
    target = tmp_path / source["path"]
    target.parent.mkdir(parents=True)
    target.write_bytes(changed)
    source["sha256"] = hashlib.sha256(changed).hexdigest()
    monkeypatch.setattr(
        subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, stdout=original),
    )
    with pytest.raises(ValueError, match="frozen source mismatch"):
        _read_pinned_source(tmp_path, source)


def test_joint_hash_span_symbol_tampering_is_rejected(graph):
    broken = deepcopy(graph)
    broken["sources"][0]["sha256"] = "0" * 64
    broken["witnesses"][0].update(
        {"start_line": 1, "end_line": 1, "symbol": "forged", "tokens": ["forged"]}
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_graph(broken)


def test_valid_other_function_cannot_replace_required_witness(graph):
    broken = deepcopy(graph)
    replacement = deepcopy(next(w for w in graph["witnesses"] if w["id"] == "fn_inverse"))
    replacement["id"] = "word_carry"
    broken["witnesses"][0] = replacement
    with pytest.raises(ValueError, match="witness role binding mismatch"):
        validate_graph(broken)


@pytest.mark.parametrize("mutation, reason", [
    ("wrong_consumer", "consumer/edge disagreement"),
    ("wrong_node_role", "node witness role mismatch"),
    ("wrong_edge_role", "edge witness role mismatch"),
    ("unresolved_test", "test evidence/anchor disagreement"),
    ("unresolved_history", "benchmark evidence/history disagreement"),
    ("wrong_history_span", "historical role binding mismatch"),
])
def test_graph_semantic_contracts(graph, mutation, reason):
    broken = deepcopy(graph)
    if mutation == "wrong_consumer":
        broken["nodes"][0]["consumers"] = []
    elif mutation == "wrong_node_role":
        broken["nodes"][0]["witness_ids"] = ["fn_inverse"]
    elif mutation == "wrong_edge_role":
        broken["edges"][0]["witness_ids"] = ["fn_inverse"]
    elif mutation == "unresolved_test":
        broken["nodes"][0]["test_evidence"]["anchor_ids"] = ["missing"]
    elif mutation == "unresolved_history":
        broken["nodes"][0]["benchmark_evidence"]["historical_ids"] = ["missing"]
    elif mutation == "wrong_history_span":
        broken["historical_evidence"][0]["witness_ids"] = ["fp_bytes"]
    with pytest.raises(ValueError, match=reason):
        validate_graph(broken)


@pytest.mark.parametrize("mutation, reason", [
    ("invented_edge", "unregistered edge binding"),
    ("invented_node", "unregistered node binding"),
    ("wrong_test_role", "test anchor role binding mismatch"),
    ("wrong_history_role", "historical role binding mismatch"),
    ("promoted_history_kind", "historical role binding mismatch"),
])
def test_independent_review_counterexamples(graph, mutation, reason):
    """Exact mutants accepted before the independent manager review."""
    broken = deepcopy(graph)
    if mutation == "invented_edge":
        broken["edges"].append({
            "from": "word64", "to": "schnorr_batch_verify", "kind": "consumed_by",
            "condition": "always",
            "claim": "The batch verifier directly computes its result using Scalar::inverse.",
            "witness_ids": ["fn_inverse"],
        })
        next(n for n in broken["nodes"] if n["id"] == "word64")["consumers"].append(
            "schnorr_batch_verify"
        )
    elif mutation == "invented_node":
        node = deepcopy(broken["nodes"][0])
        node.update({
            "id": "forged_protocol", "layer": 4, "domain": "protocol",
            "witness_ids": ["fn_inverse"], "consumers": [],
            "test_evidence": {"status": "NO_ANCHOR_IN_SCOPE", "anchor_ids": [],
                              "reason": "No test imported"},
            "benchmark_evidence": {"status": "NOT_LOCATED", "historical_ids": [],
                                   "reason": "No benchmark imported"},
        })
        broken["nodes"].append(node)
    elif mutation == "wrong_test_role":
        next(a for a in broken["test_anchors"] if a["id"] == "field_checks")[
            "witness_ids"
        ] = ["test_fn"]
    elif mutation == "wrong_history_role":
        next(h for h in broken["historical_evidence"] if h["id"] == "fp_cost_calibration")[
            "witness_ids"
        ] = ["history_doc_conflict"]
    elif mutation == "promoted_history_kind":
        next(h for h in broken["historical_evidence"] if h["id"] == "fn_reduction_prior_limit")[
            "reported_evidence_kind"
        ] = "REPORTED_MEASUREMENT"
    with pytest.raises(ValueError, match=reason):
        validate_graph(broken)


@pytest.mark.parametrize("section, reason", [
    ("witnesses", "witness vocabulary mismatch"),
    ("test_anchors", "test anchor vocabulary mismatch"),
    ("historical_evidence", "historical vocabulary mismatch"),
])
def test_extra_binding_requires_reviewed_fixture_extension(graph, section, reason):
    broken = deepcopy(graph)
    extra = deepcopy(broken[section][0])
    extra["id"] = "unreviewed_binding"
    broken[section].append(extra)
    with pytest.raises(ValueError, match=reason):
        validate_graph(broken)


@pytest.mark.parametrize("pairs", [
    [("a", "a")],
    [("a", "b"), ("b", "a")],
    [("a", "b"), ("b", "c"), ("c", "a")],
])
def test_consumption_cycle_guard_independent_of_closed_vocabulary(pairs):
    edges = [{"from": start, "to": end, "kind": "consumed_by"} for start, end in pairs]
    nodes = {node for pair in pairs for node in pair}
    with pytest.raises(ValueError, match="^cycle in consumption DAG$"):
        _check_consumption_dag(nodes, edges)


def test_bidirectional_conversion_is_not_a_consumption_cycle():
    edges = [
        {"from": "a", "to": "b", "kind": "converts_to"},
        {"from": "b", "to": "a", "kind": "converts_to"},
        {"from": "a", "to": "c", "kind": "consumed_by"},
    ]
    assert _check_consumption_dag({"a", "b", "c"}, edges) == {"a": {"c"}, "b": set(), "c": set()}
