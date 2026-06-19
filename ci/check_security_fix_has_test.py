#!/usr/bin/env python3
"""
check_security_fix_has_test.py — CI gate: every security-relevant commit must include a test.

Exit 0 = all recent security-touching commits have a corresponding test change.
Exit 1 = one or more security commits lack a test (prints offending commits).
Exit 77 = no recent commits to check (advisory skip).

What counts as "security-relevant":
  - Changes to src/cpu/src/*.cpp (signing, CT, field arithmetic, protocols)
  - Changes to compat/libsecp256k1_shim/src/*.cpp (shim behavior)
  - Changes to src/cpu/include/secp256k1/ct/*.hpp (CT primitives)

What counts as a "test":
  - Any new or modified audit/test_*.cpp file in the SAME commit
  - Any new or modified tests/*.cpp file in the SAME commit
  - Any new or modified compat/libsecp256k1_shim/tests/*.cpp

Usage:
    python3 ci/check_security_fix_has_test.py [--since <ref>] [--commits N]

Options:
    --since REF    Check commits since REF (default: last 10 commits on dev)
    --commits N    Maximum commits to check (default: 10)
"""
from __future__ import annotations
import re
import sys
import subprocess
import argparse
from pathlib import Path

# Files in these directories are security-relevant and require a test.
SECURITY_DIRS = (
    "src/cpu/src/",
    "compat/libsecp256k1_shim/src/",
    "src/cpu/include/secp256k1/ct/",
    # GPU backends: signing kernels, ECDH, CT signing paths on GPU
    "src/gpu_backend",        # src/gpu_backend_{cuda,opencl,metal,fallback}.*
    "src/secp256k1.cu",       # main CUDA kernel (CT sign, ECDH, verify)
    "shaders/",               # Metal GPU shader files (.metal)
)

# CI files that are themselves security-critical — changes to these must include
# a test even though they live in ci/ or .github/. Without this guard the gate
# could be weakened in a commit that the gate itself classifies as docs-only.
# Scope: only the gate scripts and the GitHub gate workflow. Other CI workflows
# (security-audit.yml, benchmark.yml, etc.) may be updated without a test since
# they configure runner behavior, not security logic.
SECURITY_CI_FILES = frozenset({
    "ci/check_security_fix_has_test.py",
    "ci/run_fast_gates.sh",
    ".github/workflows/gate.yml",
    # Meta-gate: loosening the advisory-skip ceiling silently grows CI coverage gaps,
    # so a change here must come with a test (its paired ci/test_check_advisory_skip_ceiling.py).
    "ci/check_advisory_skip_ceiling.py",
})

# If a commit touches only these directories/files, no test is required.
DOCS_ONLY_PATTERNS = (
    r"^docs/",
    r"^workingdocs/",
    r"^\.github/",
    r"^CHANGELOG",
    r"^README",
    r"\.md$",
    r"\.json$",
    r"\.yml$",
    r"\.yaml$",
    r"\.sh$",
    r"^ci/",
    r"^tools/",
    r"CLAUDE\.md$",
)

# Test file patterns — if any of these appear in the commit, it counts.
TEST_PATTERNS = (
    r"^audit/test_",
    r"^tests/",
    r"^src/cpu/tests/",
    r"^compat/libsecp256k1_shim/tests/",
    r"^compat/libsecp256k1_bchn_shim/tests/",
    # libbitcoin bridge differential/correctness tests (consensus-bearing:
    # test_lbtc_consensus_diff proves GPU==CPU==libsecp, test_lbtc_collect /
    # test_lbtc_bridge prove the bridge contracts).
    r"^compat/libbitcoin_bridge/tests/",
    # A change to a CI security tool (a SECURITY_CI_FILE — a Python gate / scanner)
    # is correctly covered by its paired Python unit test, not a C++ audit test.
    r"^ci/test_.*\.py$",
)


# Commits that modified security files but had their tests added in a LATER
# commit (retroactive coverage). The gate accepts these provided the named
# test file exists on disk. Format: sha_prefix → [test_file, reason].
RETROACTIVELY_COVERED: dict[str, tuple[list[str], str]] = {
    "ce2906b86c": (
        ["compat/libsecp256k1_shim/tests/shim_test.cpp", "ci/run_libsecp_shim_api_test.sh"],
        "CI security gate wiring commit: enabled the already-existing shim API/layout "
        "parity test in protected workflows. The executable test body lives in "
        "compat/libsecp256k1_shim/tests/shim_test.cpp and the runner is "
        "ci/run_libsecp_shim_api_test.sh.",
    ),
    "719a9de7db": (
        ["ci/test_check_test_assertions.py"],
        "CI-tooling commit: wired the new non-asserting-probe scanner "
        "(ci/check_test_assertions.py) into ci/run_fast_gates.sh and added CAAS-001/002 "
        "evidence-integrity hardening to caas.yml. No engine/shim/CT source changed. The "
        "scanner is validated by ci/test_check_test_assertions.py (added in the immediately-"
        "following commit), which asserts it flags a non-asserting probe and passes a clean "
        "asserting test. The GitHub Gate (gate.yml/caas.yml) does not run this check; it is a "
        "local pre-push gate, and the commit was green on GitHub.",
    ),
    "beb7385ee5": (
        ["audit/test_exploit_batch_verify_correctness.cpp",
         "audit/test_exploit_batch_verify_poison.cpp"],
        "Comment-only change to src/cpu/src/batch_verify.cpp — added "
        "NOSONAR/rationale block on the line that seeds the hash-flooding "
        "guard's per-call RNG. No executable code touched. Existing audit "
        "tests (batch_verify_correctness / poison) already validate the "
        "function's behavior including the dedup path.",
    ),
    "c389c98443": (
        ["audit/test_regression_ct_sanitizer_detection.cpp"],
        "Clang sanitizer detection fix in ct_field.cpp. The regression test "
        "was added in the immediately-following commit (test paired with the "
        "same end-to-end fix scope). It verifies ct::field_add/sub/mul/sqr "
        "matches the fast equivalent across many values — the same check "
        "loop that previously failed under Clang TSan in test_comprehensive::"
        "test_ct_field. Wired to unified_audit_runner as "
        "regression_ct_sanitizer_detection in section ct_analysis.",
    ),
    "d9c21c9bcc": (
        ["audit/test_regression_ct_sanitizer_detection.cpp"],
        "Build-only cleanup of unused add256/add_carry_u64 helpers + revert "
        "of __int128 sub256 (Werror=unused-function / pedantic / MSVC). No "
        "semantics change. Covered by the same parity regression test that "
        "the delegation refactor 92c9c719 retroactively cites.",
    ),
    "92c9c719f9": (
        ["audit/test_regression_ct_sanitizer_detection.cpp"],
        "ThinLTO miscompile follow-up to c389c984. Delegated ct::field_add/"
        "sub/neg to fast::operator+/-/unary minus. The existing regression "
        "test (test_regression_ct_sanitizer_detection.cpp wired at "
        "c389c98443) is the regression test for this fix too — it asserts "
        "ct::field_add/sub/mul/sqr parity with fast across 64 values + "
        "boundaries; the same check that previously failed under Clang "
        "ThinLTO/Sanitizers now passes (Clang Debug local repro: "
        "test_comprehensive 12023/0/10).",
    ),
    # field.cpp reduce() carry-propagation fix — paired test in same scope.
    # The fix and its regression test (test_regression_field_reduce_carry.cpp)
    # land in the same commit but git's diff parser treats the .cpp under
    # audit/ as a *new* file rather than a "modified" entry; some
    # check_security_fix_has_test.py heuristics miss the pairing. List the
    # test explicitly to remove any doubt.
    "FIELD_REDUCE_CARRY_2026_05_14": (
        ["audit/test_regression_field_reduce_carry.cpp"],
        "FE64 reduce() carry propagation in field.cpp. The regression test "
        "asserts (2^255-1)^2 mod p matches Python ground truth byte-for-byte, "
        "which is the exact case that previously broke test_field_52::mul(large, large) "
        "and downstream selftest Boundary Scalar KAT.",
    ),
    "9572d8adfd": (
        ["src/cpu/tests/test_arithmetic_correctness.cpp",
         "src/cpu/tests/test_large_scalar_multiplication.cpp"],
        "WASM/MSVC follow-up to 76054b26: gated scalar.cpp's residual "
        "__int128 sites with !SECP256K1_NO_INT128 so wasm32 takes the "
        "portable safegcd30 + manual-carry negate path, plus added a "
        "SECP256K1_RESTRICT macro shim so the MSVC field_52_impl.hpp "
        "build no longer fails on __restrict__ syntax. The existing "
        "scalar arithmetic test suite (test_arithmetic_correctness + "
        "test_large_scalar_multiplication) covers inverse + negate on "
        "the safegcd / negate paths — every USE_ASM=OFF build re-runs "
        "them and they catch any divergence between the int128 and "
        "portable scalar code paths.",
    ),
    "c67edc1ca5": (
        ["audit/test_exploit_ecdsa_sign_sentinels.cpp",
         "audit/test_regression_ellswift_ct_path.cpp"],
        "PERF-002: redundant y²=x³+7 curve check removed from 4 verify-path shim "
        "locations (pubkey_data_to_point, secp256k1_ecdsa_verify first-encounter, "
        "secp256k1_ecdsa_verify_batch small+large, shim_schnorr_bch.cpp). The "
        "libsecp256k1 trust contract validates curve membership exactly once at "
        "ec_pubkey_parse/ec_pubkey_create; subsequent callers trust the opaque struct. "
        "No security semantics changed — only redundant re-validation removed from the "
        "hot path. Retroactive coverage: test_exploit_ecdsa_sign_sentinels exercises "
        "ecdsa_verify paths using pubkey_data_to_point; test_regression_ellswift_ct_path "
        "ECP-6 exercises the shim XDH general path using the same helper. Both test files "
        "were updated in the immediately-following security fix commit.",
    ),
    "76054b2683": (
        ["src/cpu/tests/test_field_52.cpp"],
        "FE64 mul_wide column-3 u128 overflow fix. The existing test_field_52 "
        "FE52-vs-FE64 cross-check is the regression test: before the fix, "
        "mul(large, *), mul(p-1, *), mul(Gy, *) etc. produced wrong outputs "
        "from FE64 mul_wide, which the cross-check detected. After the fix, "
        "all cross-check pairs match on every USE_ASM=OFF build (sanitizers, "
        "coverage, no-asm cross-compiles).",
    ),
    "a0b35c8c6f": (
        ["audit/test_u128_compat_parity.cpp"],
        "WASM portable u128 fix in field_52.cpp / field_52_impl.hpp / "
        "u128_compat.hpp. The parity test was added in the follow-up "
        "commit and stress-tests 240,000 ops comparing the portable "
        "struct to native __int128.",
    ),
    "a3f56aed99": (
        [
            "audit/test_exploit_shim_musig_ka_cap.cpp",     # RED-TEAM-009
            "audit/test_exploit_shim_recovery_null_arg.cpp", # SHIM-005
        ],
        "Tests added retroactively in 7f85877f (ka_cap) and subsequent commit "
        "(shim_recovery_null_arg). PERF-006 (shim_ecdsa.cpp remove-redundant-copy) "
        "is a non-security refactor covered by existing ecdsa differential tests.",
    ),
    "ecf47967ae": (
        ["audit/test_regression_shim_static_ctx.cpp"],
        "Regression test added in the follow-up commit. Fix: g_static_ctx aggregate initializer was missing cached_r_G/cached_r_G_valid fields added by PERF-005, causing a compile error on shim builds.",
    ),
    "535ece4aa7": (
        ["audit/test_regression_pippenger_stale_used.cpp"],
        "test_regression_pippenger_stale_used.cpp already existed and contained PIP-R9 (n=1000, c=8) which was the failing test. The commit fixed pippenger.cpp to make PIP-R9 pass — the test file is the pairing test.",
    ),
    "97e00a0fe3": (
        ["audit/test_exploit_schnorr_edge_cases.cpp",
         "audit/test_regression_schnorr_ct_arithmetic.cpp"],
        "PERF-001 (shim_schnorr.cpp stack buf), SEC-005 (adaptor.cpp remove const_cast), SEC-008 (schnorr.cpp comment). Covered by existing schnorr edge-case and CT regression tests already in the suite.",
    ),
    "ddd3897cda": (
        ["compat/libsecp256k1_shim/tests/test_shim_null_ctx.cpp",
         "compat/libsecp256k1_shim/tests/test_shim_der_zero_r.cpp"],
        "SHIM-001 (ecdh null ctx), SHIM-002 (ellswift null ctx), SHIM-003 (musig null ctx), "
        "SHIM-004 (musig partial_sig_agg zero-output guard). Tests added retroactively in "
        "follow-up commit. PERF-002 (remove y²=x³+7 check) and other PERF fixes are "
        "non-security refactors covered by existing shim differential tests.",
    ),
    "b8d6bd830c": (
        ["audit/test_regression_ecdsa_batch_curve_check.cpp"],
        "CA-001: curve membership check (y²=x³+7) restored in large-batch ECDSA verify path "
        "(shim_batch_verify.cpp). Small-batch (n<8) had the check; large-batch (n≥8) was missing it (PERF-004). "
        "CA-002: comment in shim_schnorr.cpp clarifies from_bytes is correct per BIP-340 §Signing step 3 "
        "(mod-n reduction; parse_bytes_strict_nonzero would incorrectly fail at prob 2^-128). "
        "test_regression_ecdsa_batch_curve_check verifies both small-batch and large-batch paths "
        "reject invalid-curve pubkeys consistently (BCK-1..6).",
    ),
    "b2e561355f": (
        ["audit/test_exploit_shim_musig_secnonce.cpp",
         "audit/test_exploit_shim_musig_ka_cap.cpp"],
        "PERF-007: inline pubkey compression in musig_pubkey_agg (shim_musig.cpp). Pure performance change — inlines secp256k1_ec_pubkey_serialize call. Covered by existing MuSig2 shim tests (secnonce + ka_cap).",
    ),
    "1805224a7d": (
        ["audit/test_exploit_shim_musig_secnonce.cpp",
         "audit/test_exploit_musig2.cpp"],
        "SHIM-007: cache affine coords in pubnonce — musig2.cpp + shim_musig.cpp performance optimization. No new signing surface; covered by existing MuSig2 exploit and shim secnonce tests.",
    ),
    "e0ba350d40": (
        ["audit/test_exploit_schnorr_edge_cases.cpp",
         "audit/test_regression_schnorr_ct_arithmetic.cpp"],
        "P1-PERF-001: Schnorr verify lift_x elimination in schnorr.cpp + shim_schnorr.cpp. Pure verification optimization — eliminates redundant lift_x call. Covered by existing schnorr edge-case and CT regression tests.",
    ),
    "d9ecdc1883": (
        ["audit/test_exploit_schnorr_edge_cases.cpp",
         "audit/test_regression_schnorr_ct_arithmetic.cpp"],
        "P1-PERF-001 follow-up: fix FE52 API in schnorr_verify(Point) overload (y_aff.n[0] not .limbs()). Same scope as e0ba350d40 — pure verify path optimization. Covered by existing schnorr edge-case and CT regression tests.",
    ),
    "52e37d45bb": (
        ["audit/test_exploit_schnorr_edge_cases.cpp",
         "audit/test_exploit_musig2.cpp"],
        "Revert P1-PERF-001 Y-shortcut in secp256k1_schnorrsig_verify: musig_pubkey_agg stores non-Y data in pubkey->data[32..63], causing wrong-point bug. Covered by existing Schnorr edge-case and MuSig2 exploit tests.",
    ),
    "3a4b8a7841": (
        ["compat/libsecp256k1_shim/tests/test_shim_null_ctx.cpp",
         "compat/libsecp256k1_shim/tests/test_shim_der_zero_r.cpp"],
        "P2-SHIM-001: add SHIM_REQUIRE_CTX(ctx) to 5 parse/serialize/normalize functions in shim_ecdsa.cpp. NULL ctx now fires illegal callback and returns 0, matching libsecp256k1 ARG_CHECK(ctx!=NULL). Covered by existing NULL ctx and DER edge-case shim tests.",
    ),
    "71bba05ce0": (
        ["audit/test_regression_musig2_abi_signer_index.cpp",
         "audit/test_regression_musig2_signer_index_validation.cpp"],
        "SEC-001: ufsecp_musig2_partial_sign_v2 signer-index cross-validation added to "
        "src/cpu/src/impl/ufsecp_musig2.cpp. Covered by test_regression_musig2_abi_signer_index "
        "(new SEC-001 regression test) and test_regression_musig2_signer_index_validation "
        "(existing signer-index boundary test).",
    ),
    "0327d09b33": (
        ["audit/test_exploit_shim_recovery_null_arg.cpp",
         "audit/test_exploit_schnorr_edge_cases.cpp"],
        "T-01–T-10 security review findings: T-05 adds NULL arg checks to secp256k1_ecdsa_recover "
        "(shim_recovery.cpp) and secp256k1_schnorrsig_sign_custom (shim_schnorr.cpp). T-06 changes "
        "shim_batch_verify.cpp from per-call std::vector to thread_local scratch (performance). "
        "Covered by existing NULL arg test (shim_recovery_null_arg) and schnorr edge-case tests.",
    ),
    "be7ec0ad06": (
        ["audit/test_exploit_shim_musig_ka_cap.cpp",
         "audit/test_exploit_shim_musig_secnonce.cpp"],
        "P1-PERF-001 restore: shim_musig.cpp stores even-Y in musig_pubkey_agg (lift_x sqrt); "
        "shim_schnorr.cpp restores Y-shortcut using the correctly populated even-Y. "
        "Fixes MuSig2 verify regression caused by missing Y in agg_pk->data[32..63]. "
        "Covered by existing MuSig2 ka_cap and secnonce shim tests which exercise the full "
        "pubkey_agg→schnorrsig_verify path that depends on the even-Y layout.",
    ),
    "cdc55fb189": (
        ["audit/test_regression_hash_three_block_bounds.cpp",
         "audit/test_regression_frost_threshold_zero.cpp",
         "audit/test_regression_shim_high_s_verify.cpp",
         "audit/test_regression_shim_perf_correctness.cpp"],
        "SEC-004 (ecdsa.cpp compute_three_block bounds guard) and SEC-010 (frost.cpp "
        "threshold==0 quorum bypass guard): tests added in the follow-up commit (dcbcb6a6). "
        "PERF-001 (shim_recovery.cpp is_normalized fast path) and PERF-005 "
        "(shim_schnorr.cpp raw-pointer parse): correctness tests in test_regression_shim_perf_correctness. "
        "OpenCL scan key boundary fix (gpu_backend_opencl.cpp): covered by "
        "test_regression_opencl_bip352_scan_key_boundary.",
    ),
    "d5b1565632": (
        ["audit/test_regression_shim_security_v8.cpp"],
        "NEW-PERF-002 (shim_extrakeys.cpp: secp256k1_keypair_xonly_tweak_add — replace "
        "to_uncompressed() with point_to_pubkey_data) and NEW-PERF-001/004 (shim_schnorr.cpp "
        "ShimSchnorrCache::put — remove redundant double schnorr_xonly_pubkey_parse call). "
        "Pure performance cleanups, no new signing surface. test_keypair_xonly_tweak_add_roundtrip "
        "added retroactively in 22f840c9 to test_regression_shim_security_v8.cpp verifies "
        "NEW-PERF-002 sign+verify round-trip. The cache cleanup is covered by existing "
        "Schnorr verify regression tests that exercise the warm-cache path via repeated calls.",
    ),
    "a3b77fde44": (
        ["audit/test_exploit_schnorr_edge_cases.cpp",
         "audit/test_regression_schnorr_ct_arithmetic.cpp"],
        "feat(schnorr+shim): variable-length msg verify — removes msglen==32 restriction "
        "from secp256k1_schnorrsig_verify (shim_schnorr.cpp) to match libsecp256k1 API. "
        "Adds schnorr_verify(pubkey_x32, msg, msglen, sig) overload in schnorr.cpp using "
        "schnorr_challenge_scalar_varlen for arbitrary-length BIP-340 challenge hashing. "
        "Covered by test_exploit_schnorr_edge_cases (edge-case Schnorr sign/verify paths "
        "including NULL-arg, degenerate-output, and parity checks) and "
        "test_regression_schnorr_ct_arithmetic (CT arithmetic parity on all Schnorr ops).",
    ),
    "ac63c0ad38": (
        ["audit/test_regression_shim_security_v8.cpp",
         "audit/test_regression_signing_ct_scalar_correctness.cpp"],
        "NEW-003 (frost.cpp): doc-only comment change strengthening the caller-erasure "
        "contract on frost_keygen_finalize. No semantic change. FROST CT correctness "
        "covered by test_regression_signing_ct_scalar_correctness and FROST exploit tests.",
    ),
    "0e25ff9e42": (
        ["audit/test_exploit_adaptor_parity.cpp",
         "audit/test_ffi_round_trip.cpp",
         "audit/test_exploit_adaptor_extended.cpp"],
        "schnorr_adaptor_verify: remove redundant-and-incorrect integrity check that "
        "compared (pre_sig.R_hat + T).y_odd to needs_negation. The check rejected valid "
        "pre-sigs with needs_negation=true (~50% of cases). Covered by the existing "
        "test_exploit_adaptor_parity (which had been failing as [105] in unified runner), "
        "test_ffi_round_trip [28] Adaptor signatures, and test_exploit_adaptor_extended.",
    ),
    "acfc1654c8": (
        ["src/cpu/tests/test_v4_features.cpp"],
        "SilentPaymentScanner::scan_batch (sp_scanner.cpp): pure throughput optimization "
        "(KPlan + batch_scalar_mul_fixed_k + batch_to_compressed + batch_scalar_mul_generator + "
        "batch_x_only_bytes). No new signing surface; batch path produces identical outputs to "
        "the per-tx scan_tx path (same BIP0352/SharedSecret tag, same hash pipeline). "
        "Covered by test_v4_features.cpp which exercises the full silent payment address "
        "encode/decode + create_output + scan pipeline using the same address.hpp primitives.",
    ),
    "fd27b21bd4": (
        ["src/cpu/tests/test_v4_features.cpp"],
        "Thread-local scratch + BCH RpaScanner::scan_batch (ltc_sp.cpp, bch_scan.cpp): "
        "pure performance optimization — no new signing surface, no behavioral change. "
        "scan_batch produces the same outputs as N individual scan calls. "
        "Covered by test_v4_features.cpp (exercises address/SP primitives that both LTC-SP "
        "and BCH-RPA build on).",
    ),
    "65cc69bddf": (
        ["src/cpu/tests/test_v4_features.cpp",
         "src/cpu/tests/test_comprehensive.cpp"],
        "precompute.cpp: enable load_precompute_cache (was stub) + w=18 benchmark. "
        "No new signing surface; the precompute cache is a read-only optimization for "
        "generator table lookups. Functional correctness covered by test_v4_features.cpp "
        "(exercises SP/LTC-SP pipelines that use the precomputed generator tables) and "
        "test_comprehensive.cpp (exercises point arithmetic paths that rely on the same tables).",
    ),
    "f8f17b5104": (
        ["src/cpu/tests/test_v4_features.cpp"],
        "LtcSpScanner::scan_batch Pass 2d — batch_x_only_bytes (1 field_inv for N candidates). "
        "Pure performance optimization; functionally equivalent to N per-point field_inv calls. "
        "Covered by test_v4_features.cpp (silent payment scan pipeline includes the "
        "batch_x_only_bytes codepath).",
    ),
    "a771d15c67": (
        ["src/cpu/tests/test_v4_features.cpp",
         "audit/test_adversarial_protocol.cpp"],
        "P1-002: frost.cpp Lagrange coefficient fast::operator* → ct::scalar_mul for "
        "defensive CT consistency. Inputs (participant IDs) are public, but CT prevents "
        "regression if a secret is ever introduced. Covered by test_v4_features (SP/FROST "
        "pipeline) and test_adversarial_protocol (full FROST adversarial suite). "
        "P1-005..009 are docs/CI-only changes with no new security surface.",
    ),
    "f13b59df57": (
        ["audit/test_exploit_shim_musig_secnonce.cpp",
         "compat/libsecp256k1_shim/tests/test_shim_null_ctx.cpp"],
        "SHIM-004: shim_schnorr.cpp schnorrsig_verify now forwards msglen to the "
        "varlen overload for msglen != 32. Covered by existing shim tests which exercise "
        "the verify path. A dedicated varlen round-trip test "
        "(sign_custom(msg64, 64) + verify(msg64, 64)) is planned for the next commit. "
        "RT-003: context.hpp derive_public_key uses ct::generator_mul for nullptr ctx. "
        "REL-007/PR-013/019: CI/doc fixes only.",
    ),
    "8af91b7a02": (
        ["audit/test_regression_ct_ops.cpp"],
        "fix(werror): ECDSASignature::from_compact(array) overload inlined to avoid "
        "deprecated→deprecated call chain. Both overloads remain [[deprecated]] per SEC-003. "
        "No behavioral or security change — pure Werror fix. Covered by test_regression_ct_ops "
        "which exercises the ECDSA signing and result-handling code paths that use these "
        "overloads. The [[deprecated]] attribute only affects the compiler diagnostic, not "
        "the runtime signature of from_compact.",
    ),
    "9177df7399": (
        ["audit/test_parse_strictness.cpp",
         "audit/test_exploit_encoding_memory_corruption.cpp"],
        "fix(ci+build): MSVC portability for shim_ecdsa.cpp. Replaced "
        "__builtin_memcpy/__builtin_memset/__builtin_bswap64 (GCC/Clang-only) "
        "with the UFSECP_SHIM_MEMCPY/_MEMSET/_BSWAP64 compile-time abstraction "
        "(std::memcpy + _byteswap_uint64 on _MSC_VER, original __builtin_* "
        "elsewhere). Six call sites updated — fingerprint helper (load 8 X "
        "bytes), DER int parser (zero-prefix + payload memcpy), and load64be "
        "lambda inside the fast valid_scalar branchless check. Generated code "
        "is identical on GCC/Clang; the fix only unblocks MSVC. No behavior or "
        "security change. The two listed tests exercise the affected paths: "
        "test_parse_strictness covers the DER scalar validator (load64be + the "
        "zero-prefix memcpy in parse_int), and "
        "test_exploit_encoding_memory_corruption hits the cache fingerprint "
        "via repeated verify() calls through the shim.",
    ),
    "4bb75f2e50": (
        ["audit/test_exploit_bitcoin_message_signing.cpp",
         "audit/test_regression_ct_ops.cpp"],
        "fix(werror): migrate all from_compact callers to parse_compact_strict in "
        "message_signing.cpp, wallet.cpp, ufsecp_bip322.cpp, ultrafast_secp256k1.cpp, "
        "secp256k1_wasm.cpp. For valid inputs, parse_compact_strict produces identical "
        "results to from_compact (same Scalar::from_bytes path). The migration is purely "
        "a Werror fix with no behavioral change for valid signatures. "
        "Covered by test_exploit_bitcoin_message_signing (exercises message_signing.cpp "
        "verify_message/recover_signer paths) and test_regression_ct_ops (ECDSA operations).",
    ),
    "13eeef7e83": (
        ["audit/test_parse_strictness.cpp",
         "audit/test_regression_shim_security_v8.cpp"],
        "perf(cache): two-phase ShimSchnorrCache + ShimEcdsaCache — reverted one-phase "
        "to avoid L2 thrashing on ConnectBlock unique-pubkey workloads. Touches shim_ecdsa.cpp "
        "and shim_schnorr.cpp (cache policy change). Covered by PS-EC-02/03 cache correctness "
        "tests in test_parse_strictness.cpp (verify same pubkey on miss and hit paths) and "
        "shim_security_v8 (ShimEcdsaCache roundtrip). Pure performance policy change — no "
        "new signing surface, no behavioral change to verify results.",
    ),
    "4a67f9c088": (
        ["audit/test_parse_strictness.cpp"],
        "perf(schnorr): eliminate dead FieldElement in SchnorrSignature::parse_strict "
        "(src/cpu/src/schnorr.cpp). Replaced FieldElement::parse_bytes_strict(r_fe) with "
        "parse_and_check_lt_p helper — same r < p validation, avoids constructing a "
        "FieldElement that is immediately discarded. Pure optimization — parse semantics "
        "unchanged. Covered by test_parse_strictness PS-30c..f which verify r=0/n/n-1 "
        "edge cases through secp256k1_ecdsa_signature_parse_der → SchnorrSignature::parse_strict.",
    ),
    "9a9eccb18e": (
        ["audit/test_regression_shim_null_callback.cpp"],
        "refactor: consolidate duplicate static helpers (pubkey_data_to_point, "
        "point_to_pubkey_data) from shim_ecdsa.cpp, shim_pubkey.cpp, shim_extrakeys.cpp, "
        "shim_recovery.cpp into shim_pubkey_helpers.hpp. Pure refactor — no behavioral "
        "change, no new API surface. All four functions are exercised by "
        "test_regression_shim_null_callback (SNC-4: pubkey_negate calls pubkey_data_to_point) "
        "and by the existing shim differential tests.",
    ),
    "626a372430": (
        ["audit/test_exploit_shim_musig_ka_cap.cpp",
         "audit/test_exploit_shim_musig_secnonce.cpp"],
        "P2-PERF-003/004/005: shim_musig.cpp performance-only changes — "
        "musig_partial_sig_verify direct layout prefix reconstruction (eliminates one "
        "serialize call), musig_nonce_agg SBO N≤16 (stack accumulation, no heap), "
        "musig_partial_sig_agg SBO N≤16 (Scalar[16] stack array). No new signing surface, "
        "no behavioral change. Covered by existing MuSig2 shim tests "
        "(test_exploit_shim_musig_ka_cap and test_exploit_shim_musig_secnonce) which "
        "exercise the full partial_sig_verify, nonce_agg, and partial_sig_agg paths.",
    ),
    "68c2aefcfa": (
        ["audit/test_regression_shim_null_callback.cpp"],
        "SHIM-A01/A02/A03/A07/A08: shim_ecdsa.cpp, shim_pubkey.cpp, shim_tagged_hash.cpp — "
        "NULL argument handling now fires secp256k1_shim_call_illegal_cb instead of silently "
        "returning 0, matching libsecp256k1 upstream ARG_CHECK behaviour. "
        "test_regression_shim_null_callback.cpp (SNC-1..5) added in the follow-up commit "
        "(c9c92a4a→next) and wired to unified_audit_runner as section shim_regression "
        "(advisory=true since it requires the shim to be linked).",
    ),
    "f6b9203544": (
        ["audit/test_regression_shim_security_v9.cpp"],
        "SHIM-NEW-012b: secp256k1_ecdsa_signature_serialize_der NULL output/outputlen/sig now "
        "fires secp256k1_shim_call_illegal_cb matching libsecp256k1 ARG_CHECK behaviour. "
        "serialize_compact was fixed in commit 93010b66. This commit fixes serialize_der "
        "which was missed in that batch. test_regression_shim_security_v9.cpp (wired to "
        "unified_audit_runner as regression_shim_security_v9, advisory=true) covers both "
        "serialize_compact and serialize_der NULL arg callback tests (v9-012/015).",
    ),
    "efd113c125": (
        ["audit/test_regression_dedup_refactors_2026-05-24.cpp"],
        "DEDUP-2026-05-24 part 1: SonarCloud-driven deduplication of bip39.cpp's "
        "decode_bip39_words helper (was inlined twice across bip39_validate / "
        "bip39_mnemonic_to_entropy) + sp_scanner.cpp ↔ ltc/ltc_sp.cpp scan_batch "
        "(extracted to sp_scan_batch_impl.hpp templated helper). Pure code-shape "
        "refactor — both call sites still secure_erase entropy / random scratch on "
        "every exit path, both scanners still tag-separate via SHA256 domain string. "
        "The retroactive regression test (wired to unified_audit_runner as "
        "regression_dedup_refactors_2026_05_24, advisory=false, behavioral_freeze) "
        "pins agreement of bip39_validate vs bip39_mnemonic_to_entropy on the "
        "12/24-word zero-entropy vector + bad-checksum negative case (#1), and the "
        "shared SP/LTC scan_batch_impl early-exit semantics (#2). Landed in the "
        "immediately-following commit b7f64228 alongside the ellswift refactor "
        "extension of the same dedup work.",
    ),
    "1fd72fcdeb": (
        ["audit/test_regression_dedup_refactors_2026-05-24.cpp"],
        "DEDUP-2026-05-24 part 3 (ellswift cleanup): remove leftover FE_ZERO/"
        "FE_THREE/FE_FOUR local shadow declarations in ellswift_try_u + "
        "xswiftec_inv_var (src/cpu/src/ellswift.cpp). Pure dead-code removal — "
        "the named-constant shadows were unused after the retry-loop helper "
        "refactor in b7f64228; this commit completes that cleanup. No semantic "
        "change. ellswift CT correctness covered by regression_dedup_refactors_"
        "2026_05_24 (same wave's regression test) and existing ellswift "
        "differential tests (ellswift round-trip).",
    ),
    "9d31ea0bc8": (
        ["audit/test_regression_dedup_refactors_2026-05-24.cpp"],
        "DEDUP-2026-05-24 part 2: collapsed four {fe,sc}_to_data static_cast bodies "
        "(in both public include/secp256k1/types.hpp and the internal "
        "src/cpu/include/secp256k1/types.hpp mirror) into one detail::to_data_cast<T>() "
        "template. types.hpp defines struct layouts + zero-cost cast helpers only — "
        "no secret material is touched, and -O3 inlines the wrapper to the original "
        "cast. The regression test (#3 in regression_dedup_refactors_2026_05_24) "
        "checks fe_to_data / sc_to_data return the original address across all four "
        "void* / const void* overloads, pinning the no-op nature of the refactor.",
    ),
    "6faecd4dc5": (
        ["audit/test_regression_musig_noncegen_extra_input.cpp"],
        "SHIM-NONCEGEN-001: musig2_nonce_gen gains nonce_extra parameter; "
        "secp256k1_musig_nonce_gen now forwards extra_input32. "
        "The self-healing regression test (test_regression_musig_noncegen_extra_input.cpp) "
        "is the pairing test — it automatically switches from bug-open mode (asserts "
        "identical nonces) to bug-fixed mode (asserts distinct nonces) when the "
        "SHIM-NONCEGEN-001 marker is removed from shim_musig.cpp. No code change to "
        "the test file was needed because of the self-healing design; the test was "
        "already wired as advisory=true and is promoted to advisory=false in this commit.",
    ),
    "82b9c074dd": (
        [
            "audit/test_regression_bip324_session.cpp",
            "audit/test_exploit_bip324_session.cpp",
        ],
        "Dead-code removal only: erased orphaned reference to privkey_ (renamed to "
        "privkey_scalar_ by SEC-006 in 546b893c). No behavioral change — the code was "
        "unreachable since privkey_ no longer existed as a member. Covered by "
        "test_regression_bip324_session (full BIP-324 handshake + AEAD round-trip) and "
        "test_exploit_bip324_session (adversarial session-setup paths that exercise "
        "complete_handshake(), the function where the dead erase lived).",
    ),
    "546b893cc4": (
        [
            "audit/test_regression_bip324_session.cpp",
            "audit/test_regression_musig2_session_token.cpp",
            "audit/test_regression_schnorr_varlen_ct_fixes.cpp",
            "audit/test_regression_ecdsa_batch_curve_check.cpp",
        ],
        "SEC-006 (P1): bip324.cpp/bip324.hpp — ephemeral key stored as Scalar member "
        "(not raw bytes), eliminating the fork-window privkey exposure. Covered by "
        "test_regression_bip324_session (BIP-324 session CT path). "
        "AUDIT-004 (P2): shim_musig.cpp — MuSig2 session token replaces raw heap pointer "
        "stash in session->data, eliminating ASLR leak. Covered by "
        "test_regression_musig2_session_token (session token isolation checks). "
        "AUDIT-003 (P2): shim_schnorr.cpp — sign_custom rejects msglen!=32 with "
        "illegal_callback. Covered by test_regression_schnorr_varlen_ct_fixes (varlen "
        "sign/verify parity). PERF-004 (P2): shim_batch_verify.cpp — remove "
        "shrink_to_fit(). Covered by test_regression_ecdsa_batch_curve_check (batch "
        "verify correctness including large-batch path). PERF-001 (P2): shim_schnorr.cpp "
        "ShimSchnorrCache 1-phase warm-up — pure performance, covered by shim parity tests.",
    ),
    "abbba7b3": (
        [
            "audit/test_regression_ecdh_off_curve.cpp",
        ],
        "SEC-005 (P2): ecdh.cpp — ecdh_compute/ecdh_compute_xonly/ecdh_compute_raw now reject "
        "off-curve pubkeys (y²≠x³+7) and the point-at-infinity before invoking ct::scalar_mul, "
        "closing the invalid-curve twist-injection attack vector (ePrint 2015/1233). "
        "SEC-006 (P1): frost.cpp — compute_challenge() erases e_hash after signing. "
        "CT-008/SEC-004 (P1): ecdsa.cpp — is_zero_ct() on sig.r in hedged+verified paths. "
        "COMPAT-010/SEC-001 (P1): BCH shim strict parse + is_zero_ct on sk. "
        "COMPAT-001 (P2): shim_context.cpp rejects unknown flag bits above type field. "
        "COMPAT-004 (P2): shim_schnorr.cpp sign_custom returns 0 for msglen!=32. "
        "COMPAT-006 (P2): shim_extrakeys.cpp NULL args fire illegal callback. "
        "CT-007 (P2): CUDA recovery overflow comparison is now branchless. "
        "VER-006 (P2): CocoaPods podspec removes SECP256K1_FAST_NO_SECURITY_CHECKS. "
        "Covered by test_regression_ecdh_off_curve (OCK-1..5: off-curve + infinity rejection "
        "in all three ecdh_compute* variants, plus positive guard for valid key).",
    ),
    "1b73e45abc": (
        ["audit/test_parse_strictness.cpp",
         "audit/test_regression_shim_security_v8.cpp"],
        "fix(bench): add cold-start Schnorr benchmark + fix stale ShimSchnorrCache — "
        "shim_schnorr.cpp ShimSchnorrCache stale-entry fix (replace iteration-count "
        "guard with explicit is_valid() flag). Pure performance/correctness fix to the "
        "cache; no new signing surface. Covered by test_parse_strictness (PS-EC-02/03 "
        "cache correctness: same-pubkey miss+hit paths) and test_regression_shim_security_v8 "
        "(ShimEcdsaCache roundtrip exercising the same cache lifecycle). Retroactive entry "
        "added in the same session as TRNC-1..4 shim null-arg fixes "
        "(test_regression_shim_tweak_recover_null_cb).",
    ),
    "9d4a8b30": (
        ["audit/test_regression_shim_xonly_parse.cpp"],
        "PASS4-003: shim_extrakeys.cpp secp256k1_xonly_pubkey_parse replaces manual "
        "FieldElement::sqrt() with schnorr_xonly_pubkey_parse (FE52 Jacobi QR pre-rejection "
        "+ lift_x/GLV cache). Pure performance optimization — parse semantics unchanged. "
        "Covered by test_regression_shim_xonly_parse (SXP-1..5: parse roundtrip, "
        "not-on-curve rejection, x>=p strict boundary, from_pubkey consistency, parity flag). "
        "Test added in the follow-up commit (batch parallel + build-dir fixes).",
    ),
    "cc88b0cf56": (
        ["audit/test_regression_s_scalar_erasure.cpp"],
        "fix(security): erase nonce-derived r in ecdsa_sign_hedged (src/cpu/src/ecdsa.cpp, "
        "one-line secure_erase(&r) defense-in-depth, companion to the secure_erase(&s) it "
        "sits beside). Covered by test_regression_s_scalar_erasure.cpp, whose header and "
        "scan explicitly cover BOTH secure_erase(&s) and secure_erase(&r) in ecdsa.cpp "
        "(added immediately after secure_erase(&k_inv)). The commit also updated "
        "SECRET_LIFECYCLE.md + SECURITY_CLAIMS.md (the secret-path doc pairing). No new "
        "signing surface; pure stack-residue hardening.",
    ),
    "7d094c7c09": (
        ["ci/test_caas_integrity.py"],
        "fix(ci): increase shim_audit_report artifact retention 90→365 days in "
        ".github/workflows/gate.yml. Operational config-only change (retention-days), NO "
        "security logic touched — flagged solely because gate.yml ∈ SECURITY_CI_FILES. "
        "There is no executable security surface to test; gate.yml workflow structure / CAAS "
        "gate integrity is exercised by ci/test_caas_integrity.py. Retroactively whitelisted "
        "as a benign CI-retention change.",
    ),
    "f0ea17663a": (
        ["ci/test_caas_integrity.py"],
        "fix(ci): revert the CAAS-FG-01 gate.yml hard-gate ctest step. The gate's shim "
        "job builds with SECP256K1_BUILD_TESTS=OFF, so NO audit standalone CTest registers "
        "there — the added `ctest --no-tests=error` step matched zero tests and correctly "
        "failed, so it was reverted to the prior working step. Pure CI-workflow config revert "
        "(.github/workflows/gate.yml) plus a docs/REVIEW_VALIDATED_FINDINGS.md note — NO "
        "security logic touched, flagged solely because gate.yml ∈ SECURITY_CI_FILES. The "
        "gate.yml workflow structure / CAAS gate integrity is exercised by "
        "ci/test_caas_integrity.py. Same class as the 7d094c7c09 retention-change entry.",
    ),
    "02602cf420": (
        ["ci/test_check_tag_conformance.py"],
        "ci(caas): systemic tagged-hash tag-conformance gate. Flagged SOLELY because it "
        "wires ci/check_tag_conformance.py into ci/run_fast_gates.sh (∈ SECURITY_CI_FILES). "
        "The gate is a pure CI scanner (no executable crypto surface); its behavior is "
        "exercised by the paired ci/test_check_tag_conformance.py unit test (added in the "
        "immediately-following commit). Same class as the gate.yml SECURITY_CI_FILES "
        "entries (7d094c7c09 / f0ea17663a) and the 8f5915c5b6 ZK-tag-gate entry.",
    ),
    "8f5915c5b6": (
        ["ci/check_zk_tag_conformance.py"],
        "fix(zk): GPU CT range-prove Fiat-Shamir tags corrected to Bulletproof/* (Metal "
        "src/metal/shaders/secp256k1_ct_zk.h + OpenCL src/opencl/kernels/secp256k1_ct_zk.cl) "
        "plus wiring the new gate into ci/run_fast_gates.sh. Flagged SOLELY because "
        "ci/run_fast_gates.sh ∈ SECURITY_CI_FILES. The Metal/OpenCL shader tag fix is not "
        "runnable on the CUDA-only dev box, so its regression guard IS the new "
        "ci/check_zk_tag_conformance.py gate (added in the SAME commit), which bans the "
        "abbreviated BP/<chal> tag across all backends. Same class as the gate.yml "
        "SECURITY_CI_FILES entries (7d094c7c09 / f0ea17663a).",
    ),
    "1f34150867": (
        ["audit/test_regression_ecdsa_batch_verify_mt.cpp",
         "compat/libsecp256k1_shim/tests/test_shim_batch_mt.cpp"],
        "Remove the arbitrary 64-thread cap from ecdsa/schnorr_batch_verify_mt "
        "(src/cpu/src/batch_verify.cpp): replaced the fixed std::array<std::thread,64> "
        "pool with a dynamic std::vector<std::thread> sized to n_threads; the worker "
        "count is reduced only to hardware_concurrency and the chunk count, no arbitrary "
        "cap. The shim change (compat/libsecp256k1_shim/src/shim_batch_verify.cpp) was a "
        "doc-comment-only threading-model update. Pure throughput change — verify is "
        "variable-time over public data, boolean result unchanged for any thread count. "
        "Covered by test_regression_ecdsa_batch_verify_mt (extended in the follow-up "
        "commit with the >64-thread parity check {65,128,256,1024} and a source-scan "
        "asserting no kMaxThreads cap remains + dynamic std::vector pool) and "
        "test_shim_batch_mt (shim ECDSA/Schnorr batch-mt parity across thread counts).",
    ),
    "0d2edda60b": (
        ["ci/test_gen_build_options.py"],
        "docs(build): wired the new ci/gen_build_options.py BUILD_OPTIONS.md drift gate "
        "into ci/run_fast_gates.sh (∈ SECURITY_CI_FILES) + added docs/BUILD_OPTIONS.md. "
        "Flagged SOLELY because run_fast_gates.sh ∈ SECURITY_CI_FILES; the gate is a "
        "pure doc-drift scanner (no executable crypto surface). Its behaviour is "
        "exercised by the paired ci/test_gen_build_options.py unit test (parser + "
        "deterministic render + live --check), added in the immediately-following commit. "
        "Same class as the gate-wiring entries 02602cf420 / 8f5915c5b6 / 7d094c7c09.",
    ),
    "9b30f7b0f8": (
        ["src/cpu/tests/test_v4_features.cpp"],
        "Address encoder allocation optimization in src/cpu/src/address.cpp: "
        "bech32_encode now uses stack-backed 5-bit conversion for normal witness "
        "programs and heap fallback for oversized payloads. No secret-bearing "
        "or consensus parser semantics changed. Retroactive coverage in "
        "test_v4_features.cpp checks the exact BIP173 P2WPKH vector and an "
        "oversized Bech32m paycode-style roundtrip, covering both paths.",
    ),
}

# Frozen count guard (CAAS-006): prevents silent whitelist growth.
# When adding a new entry above, increment this constant too.
# Unauthorized bypass (adding an entry without incrementing) → import-time assertion failure.
RETROACTIVELY_COVERED_FROZEN_COUNT: int = 63
assert len(RETROACTIVELY_COVERED) == RETROACTIVELY_COVERED_FROZEN_COUNT, (
    f"RETROACTIVELY_COVERED has {len(RETROACTIVELY_COVERED)} entries but "
    f"RETROACTIVELY_COVERED_FROZEN_COUNT={RETROACTIVELY_COVERED_FROZEN_COUNT}. "
    "Increment RETROACTIVELY_COVERED_FROZEN_COUNT when adding a new retroactive entry."
)

# Bot commits that auto-update evidence — skip.
BOT_MSG_PREFIXES = (
    "[bot]",
    "chore: bump submodule",
    "chore: update submodule",
)


def run(cmd: list[str], cwd: str | None = None) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, timeout=30)
    return r.stdout.strip()


def get_commits(since: str | None, n: int) -> list[str]:
    if since:
        out = run(["git", "log", f"{since}..HEAD", "--format=%H"])
    else:
        out = run(["git", "log", f"-{n}", "--format=%H"])
    return [h for h in out.splitlines() if h]


def commit_files(sha: str) -> list[str]:
    out = run(["git", "diff-tree", "--no-commit-id", "-r", "--name-only", sha])
    return out.splitlines()


def commit_message(sha: str) -> str:
    return run(["git", "log", "-1", "--format=%s", sha])


def is_security_file(path: str) -> bool:
    if path in SECURITY_CI_FILES:
        return True
    return any(path.startswith(d) for d in SECURITY_DIRS)


def is_docs_only(path: str) -> bool:
    if path in SECURITY_CI_FILES:
        return False  # security-critical CI files are never classified as docs-only
    return any(re.search(p, path) for p in DOCS_ONLY_PATTERNS)


def is_test_file(path: str) -> bool:
    return any(re.search(p, path) for p in TEST_PATTERNS)


def check_commit(sha: str) -> tuple[bool, str]:
    """Return (ok, reason). ok=True means pass."""
    msg = commit_message(sha)

    # Skip bot commits
    if any(msg.startswith(pfx) for pfx in BOT_MSG_PREFIXES):
        return True, "bot commit — skip"

    # Retroactively covered: tests added in a later commit
    for prefix, (test_files, note) in RETROACTIVELY_COVERED.items():
        if sha.startswith(prefix) or sha == prefix:
            missing = [f for f in test_files if not Path(f).exists()]
            if not missing:
                return True, f"retroactively covered: {', '.join(test_files)} — {note}"
            return False, f"retroactive coverage incomplete: missing {missing}"

    files = commit_files(sha)
    if not files:
        return True, "no files"

    # Find security files changed
    security_files = [f for f in files if is_security_file(f) and not is_docs_only(f)]
    if not security_files:
        return True, "no security files"

    # Check if all changed files are docs-only (no real code change)
    non_docs = [f for f in files if not is_docs_only(f)]
    if not non_docs:
        return True, "docs-only commit"

    # Security files changed — require at least one test file in the commit
    test_files = [f for f in files if is_test_file(f)]
    if test_files:
        return True, f"has tests: {', '.join(test_files[:3])}"

    reason = (
        f"security files changed ({', '.join(security_files[:3])}) "
        f"but no test file found in commit"
    )
    return False, reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", default=None,
                        help="Check commits since this ref (e.g. HEAD~5, a tag)")
    parser.add_argument("--commits", type=int, default=10,
                        help="Max commits to check (default: 10)")
    args = parser.parse_args()

    commits = get_commits(args.since, args.commits)
    if not commits:
        print("::notice::check_security_fix_has_test: no commits to check")
        return 77

    failures: list[tuple[str, str, str]] = []
    for sha in commits:
        ok, reason = check_commit(sha)
        msg = commit_message(sha)
        status = "OK  " if ok else "FAIL"
        print(f"  {status} {sha[:10]}  {msg[:60]}")
        if not ok:
            failures.append((sha, msg, reason))

    if failures:
        print(f"\n  {len(failures)} commit(s) have security changes WITHOUT tests:\n")
        for sha, msg, reason in failures:
            print(f"    {sha[:10]}  {msg[:70]}")
            print(f"             Reason: {reason}")
        print()
        print("  Rule: every commit touching src/cpu/src/, compat/libsecp256k1_shim/src/,")
        print("  or src/cpu/include/secp256k1/ct/ MUST include a test file in audit/test_*")
        print("  or tests/*.cpp in the SAME commit.")
        print("  See CLAUDE.md: 'Exploit / Audit Test Conversion Standard'")
        return 1

    print(f"\n  All {len(commits)} checked commits have tests or are docs-only. OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
