window.BENCHMARK_DATA = {
  "lastUpdate": 1772625675291,
  "repoUrl": "https://github.com/shrec/UltrafastSecp256k1",
  "entries": {
    "Perf Regression Gate": [
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "28a40d0a3767cfc97385bd73e7648672512e4927",
          "message": "feat: v3.16.0 -- BIP-340 strict, OpenSSF hardening, FROST RFC 9591, audit infrastructure (#77)\n\n* feat: v3.16.0 -- BIP-340 strict parsing, CT erasure, local Docker CI\n\nSecurity:\n- BIP-340 strict parsing: Scalar::parse_bytes_strict, FieldElement::parse_bytes_strict, SchnorrSignature::parse_strict\n- CT buffer erasure via volatile function-pointer trick in schnorr_sign/ecdsa_sign\n- lift_x deduplication, Y-parity fix (limbs()[0] & 1), pragma balance fix\n- C ABI functions now use strict parsing internally\n\nAudit:\n- ct_sidechannel_smoke marked advisory (timing flakes on shared CI runners)\n- carry_propagation test: cross-validation (generator vs generic path) + hex diagnostics for ARM64\n- 31-test BIP-340 strict suite (test_bip340_strict.cpp)\n\nLocal CI (Docker):\n- docker-compose.ci.yml: single-command orchestration for 14 CI jobs\n- pre-push target: warnings + tests + ASan + audit in ~5 min\n- audit job mirrors audit-report.yml (GCC-13 + Clang-17)\n- ccache volume for fast rebuilds\n- scripts/hooks/pre-push + scripts/pre-push-ci.ps1\n\nDocs:\n- COMPATIBILITY.md, BINDINGS_ERROR_MODEL.md updates\n- SECURITY.md: library-side erasure, planned items checklist, API stability refs\n- UFSECP_BITCOIN_STRICT CMake option\n- packaging.yml release workflow race fix\n\nTests: 26/26 pass locally (0 failures)\n\n* feat: ARM64 native dudect CI + ct-verif LLVM pass CI, docs update\n\nCI:\n- ct-arm64.yml: native Apple Silicon (M1) dudect -- smoke per-PR, full nightly\n- ct-verif.yml: compile-time CT verification via LLVM pass (deterministic)\n\nDocs:\n- SECURITY.md: mark ARM64 dudect + ct-verif as done, update version table\n- CT_VERIFICATION.md: update known limitations, planned improvements, v3.16.0\n- CHANGELOG.md: add CT Verification CI section\n- README.md: add CT ARM64 + CT-Verif badges\n\n* audit: MuSig2/FROST dudect, Valgrind CT CI, SARIF output, perf regression gate\n\n- test_ct_sidechannel.cpp: add group [9] MuSig2/FROST protocol timing\n  tests (musig2_partial_sign, frost_sign, frost_lagrange_coefficient)\n- unified_audit_runner.cpp: add write_sarif_report() + --sarif CLI flag\n  for GitHub Code Scanning integration (SARIF v2.1.0)\n- valgrind-ct.yml: new CI workflow wrapping scripts/valgrind_ct_check.sh\n  (nightly + on push to main/dev)\n- bench-regression.yml: per-commit benchmark regression gate (120% threshold,\n  fail-on-alert: true)\n- audit-report.yml: add --sarif flag + SARIF upload step for linux-gcc job,\n  security-events:write permission\n- SECURITY.md: check off Valgrind CT, MuSig2/FROST dudect, SARIF, perf gate\n- CHANGELOG.md: document all new items under v3.16.0\n- README.md: add Valgrind CT + Perf Gate workflow badges\n- CT_VERIFICATION.md: check off dudect expansion + Valgrind CT taint\n\n* v3.16.1: OpenSSF Scorecard hardening, FROST RFC 9591 tests, audit progress bar, community files\n\nOpenSSF Scorecard (7.3 -> 9+ target):\n- Pin all GitHub Actions to full SHA (codeql-action v4.32.4, upload-artifact v6.0.0)\n- Add harden-runner to discord-commits, packaging RPM jobs\n- Add persist-credentials: false to all checkout steps with write permissions\n- Standardize action versions across 13 workflow files\n\nFROST RFC 9591 Protocol Invariant Tests:\n- test_rfc9591_invariants: 7 invariants (verification share, Lagrange interpolation,\n  Feldman VSS, partial sig linearity, partial sig verification, wrong share rejection,\n  nonce commitment consistency)\n- test_rfc9591_3of5: exhaustive 3-of-5 signing over all C(5,3)=10 subsets\n\nAudit Sub-test Progress Visibility:\n- New audit_check.hpp: centralized CHECK macro with 20-char ASCII progress bar\n- Migrated all 22 audit .cpp files to use shared CHECK macro\n- Windows-safe unbuffered stdout (setvbuf _IONBF)\n\nNew Audit Modules:\n- test_musig2_bip327_vectors.cpp: 35 BIP-327 reference tests\n- test_ffi_round_trip.cpp: 103 FFI boundary tests\n- test_fiat_crypto_vectors.cpp: expanded to 752 checks\n\nCommunity Files:\n- ADOPTERS.md with production/development/hobby categories\n- 4 GitHub Discussion templates (Q&A, Show-and-Tell, Ideas, Integration Help)\n\nBuild: 24/26 CTest pass (2 ct_sidechannel = known Windows timing noise)\nAudit: 48/49 AUDIT-READY (1 advisory dudect smoke)\n\n* fix: valgrind_ct_check.sh binary path (audit/ not cpu/), update CHANGELOG for v3.16.0\n\n* fix: valgrind_ct_check.sh grep -c double-zero bug (0\\\\n0 integer parse failure)\n\ngrep -c prints '0' on no match but exits 1. The || echo '0' fallback\nappended a second '0', producing '0\\n0' which broke bash [[ -eq 0 ]]\ncomparisons. Changed to || true with  default.",
          "timestamp": "2026-03-01T17:09:31+04:00",
          "tree_id": "68ea4692e39817d711a3da784395fdeb6972add0",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/28a40d0a3767cfc97385bd73e7648672512e4927"
        },
        "date": 1772370667769,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 22,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 280,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 36000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 131,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7668767aacbd9d01a4feedade1c71694f3dac6ec",
          "message": "v3.16.1: cross-platform bench/audit campaign + docs (#78)\n\n* docs: add cross-platform audit reports (5 configs, all AUDIT-READY)\n\n- audit/platform-reports/: 5 platform audit reports (txt + json)\n  - Windows x86-64 Clang 21.1.0 (local, v3.16.0): 48/49 PASS\n  - Linux x86-64 GCC 13.3.0 (Docker CI, v3.16.0): 48/49 PASS\n  - Linux x86-64 Clang 17.0.6 (GitHub CI, v3.15.2): 46/46 PASS\n  - Linux x86-64 GCC 13.3.0 (GitHub CI, v3.15.2): 46/46 PASS\n  - Windows x86-64 MSVC 1944 (GitHub CI, v3.15.2): 45/45 PASS\n- PLATFORM_AUDIT.md: summary table + section-by-section breakdown\n- README.md: added Cross-Platform Audit Results subsection\n\nVerify: review audit/platform-reports/PLATFORM_AUDIT.md\n\n* v3.16.1: cross-platform bench/audit campaign + docs\n\n4-platform bench_hornet (x86/ARM64/RISC-V/ESP32) vs libsecp256k1 + CT-vs-CT\n\n7-config audit campaign: all AUDIT-READY (48/49 or 40/40)\n\nRISC-V audit on Milk-V Mars (48/49 PASS), ESP32-S3 audit (40/40 PASS)\n\nBENCHMARKING.md + AUDIT_GUIDE.md guides, examples stability markers\n\nbench_hornet conditional build when libsecp source unavailable\n\nVerify: docker/local_ci.ps1 -Job quick (PASSED 2, FAILED 0)",
          "timestamp": "2026-03-02T02:43:33+04:00",
          "tree_id": "57332abdac305f435923d180687ac70e25144e1d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7668767aacbd9d01a4feedade1c71694f3dac6ec"
        },
        "date": 1772405114353,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 22,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 278,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 36000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 142,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 132,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "cf57cb69b0276de4b6cc7a35582717db228dc125",
          "message": "perf: branchless reduce + optimized x86-64 asm reduction + direct asm dispatch\n\n- field.cpp reduce(): Replace while-loops with bounded 2-pass unroll +\n  branchless conditional subtract (no branches in hot path)\n- field.cpp mul_impl/square_impl: Direct assembly call on x86-64,\n  eliminating FieldElement wrapper + 4x memcpy round-trips\n- field_asm_x64_gas.S field_mul_full_asm: Use rdx=0x1000003D1 for single\n  MULX per high limb (was separate mul-by-977 + shift-by-32 = 2x ops).\n  Saves ~30 instructions in reduction phase.\n- field_asm_x64_gas.S: Replace reduction loops (.Lfull_reduce_loop,\n  .Lsqr_reduce_loop, .Lreduce_loop_strict) with bounded 2-pass unroll +\n  branchless final pass. Zero branches in hot path.\n- All 3 assembly functions optimized: reduce_4_asm, field_mul_full_asm,\n  field_sqr_full_asm\n\n33/33 tests pass. No behavior change.",
          "timestamp": "2026-03-03T04:42:28+04:00",
          "tree_id": "52e281a294993893e9f7f3740b4b7f728525d9f5",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/cf57cb69b0276de4b6cc7a35582717db228dc125"
        },
        "date": 1772498654274,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 21.1,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 980.4,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 12.6,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 8.6,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 10,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 46,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1279.3,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.3,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 10.9,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7346.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35398.6,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38414.7,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 373.4,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 148.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 12074.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40572.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9013.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9535,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41047.2,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 13.5,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1251.5,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 45.4,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 10,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 239.8,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 148.3,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 373.5,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 38442.7,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 1.9,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 19.8,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 16.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 269019.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 67255,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 960698.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 60043.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 3789659.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 59213.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 154012.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 618365.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2500673.6,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19075.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41389.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 411.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 297.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 276,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23382.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21173.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20873.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1150.5,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 17650,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 18959.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 38012.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 394676.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414231.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 370637.1,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1251.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 90.8,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 38442.7,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 13.5,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 39798.5,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 40572.5,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 773.9,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 13.5,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 41047.2,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 38442.7,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1251.5,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 39694.2,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 40572.5,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 878.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 55.4,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 121700000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "7f31f768e1b5b13c2efcff2b49aa9684f07ab048",
          "message": "fix(build): add missing field_4x64_inline.hpp (required by point.cpp)",
          "timestamp": "2026-03-04T03:31:00+04:00",
          "tree_id": "83bf682a8dd721dfe14ac92fd98cfef00f4a2823",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7f31f768e1b5b13c2efcff2b49aa9684f07ab048"
        },
        "date": 1772580767273,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 16.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1099.9,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 13.9,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.1,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 44.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.8,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.2,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7360.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 36892.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40918.6,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 401.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 158.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 63587.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 43198.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9055,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9601.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 43629.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1334.3,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 45.2,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 146.4,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 160.8,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 402.5,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 41065.7,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 2.2,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 22.5,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 20.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 289083.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 72271,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1030568.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 64410.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4088051.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 63875.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 165206.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 661821.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2662457.7,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19455.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 42944.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 159.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 309.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 305.9,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 86959.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 70897.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21125.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1165.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20868.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 22030.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42782.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 393834.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414718.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 376872.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1334.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 90.5,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 41065.7,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 42502.5,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 43198.4,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 695.9,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 43629.1,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 41065.7,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1334.3,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 42400,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 43198.4,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 798.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 56.6,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 129600000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "108ebbf3a4e4cc5764ff109ba13679130e640dd7",
          "message": "fix(build): add #else fallbacks for MSVC/WASM (point.cpp, fiat linkage)\n\n- Point::next()/prev(): add #else fallback for non-SECP256K1_FAST_52BIT\n  platforms (fixes MSVC C4716 'must return a value')\n- Point::add_inplace()/sub_inplace(): add #else fallback (were silent\n  no-ops on platforms without SECP256K1_FAST_52BIT)\n- test_fiat_crypto_linkage.cpp: guard with #if !_MSC_VER (MSVC lacks\n  __int128 required by fiat-crypto reference code)",
          "timestamp": "2026-03-04T03:46:18+04:00",
          "tree_id": "0729a5367ed9f5b78df191db0e9aa794eec2209c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/108ebbf3a4e4cc5764ff109ba13679130e640dd7"
        },
        "date": 1772581699729,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 16.8,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16.1,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1101.8,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 14.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.4,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 44,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.4,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.2,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7304.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 37337.3,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40970.9,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 403.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 160.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 62358.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 43368.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9030.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9603.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 43771.9,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1333.3,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 44.3,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 146.2,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 159.3,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 401.9,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 40996.2,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 2.2,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 22.8,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 20.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 285484.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 71371.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1012534.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 63283.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4019596.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 62806.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 164992.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 660827.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2660164.1,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19472.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41119.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 158.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 309.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 306.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 86808.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 70633,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21272.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1162.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20704.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 22050.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42543.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 400818.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 424122.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 383457.4,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1333.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 88.5,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 40996.2,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 42430.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 43368.3,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 938.2,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 43771.9,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 40996.2,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1333.3,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 42329.4,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 43368.3,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 1038.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 55.7,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 130100000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "0067eaa9d799f727e592103e3d95e81f6f6d3691",
          "message": "fix(build): suppress GCC -Wpedantic for __int128 + unused function warnings\n\n- CMakeLists.txt: add -Wno-pedantic for GCC (project requires __int128)\n- point.cpp: pragma suppress -Wunused-function/-Wrestrict for 4x64 scaffolding\n- batch_verify.cpp: pragma suppress -Wpedantic for __int128 carry chain\n- glv.cpp: pragma suppress -Wpedantic for __int128 in Comba multiply blocks\n- field_4x64_inline.hpp: pragma suppress -Wpedantic for __int128 field ops\n- test_fiat_crypto_linkage.cpp: pragma suppress -Wpedantic for fiat_ref u128\n- test_wycheproof_ecdsa.cpp: remove unused pk/msg_hash, add [[maybe_unused]]\n\nDocker CI pre-push: 5/5 PASS (warnings, gcc, clang, asan, audit)\nLocal: 31/31 tests PASS",
          "timestamp": "2026-03-04T04:08:54+04:00",
          "tree_id": "c01dd24ece9504da5aba549b95c5f40491c863b2",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/0067eaa9d799f727e592103e3d95e81f6f6d3691"
        },
        "date": 1772583048623,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 16.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1101.1,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 13.9,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.1,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 44.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1350.2,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.2,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7421.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 36838,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40931.9,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 407.1,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 158.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 62781.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 43143.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9064.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9623.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 43491.4,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1337.9,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 44.7,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 146.3,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 159.2,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 402.7,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 41027.1,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 2.2,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 21.8,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 20.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 285547.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 71386.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1015534.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 63470.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4017122.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 62767.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 164963.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 660208.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2660058.1,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19489.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41162.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 158.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 423.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 307.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 309.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 86967.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 70682,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21359.3,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1160,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20645.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 22060.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42403.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 393272.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415387.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375423.4,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1337.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 89.5,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 41027.1,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 42466.5,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 43143.8,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 677.2,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 43491.4,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 41027.1,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1337.9,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 42365,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 43143.8,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 778.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 56.2,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 129400000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "c4363327f6d443845bab45d34583cbf0ae6a53d1",
          "message": "security(audit): ECDSA recovery fuzz + ECDH edge tests + incident response runbook (Track K)\n\nFuzz coverage (K2):\n- Suite [14]: ECDSA recovery boundary fuzz (roundtrip, invalid recid, random sig, NULL args)\n- Suite [15]: ECDH infinity/edge cases (x-only random, raw random, zero-pubkey rejection)\n- Fix pre-existing -Wsign-conversion warnings in suite 5 (size_t init list)\n\nGovernance (K7):\n- docs/INCIDENT_RESPONSE.md: 5-phase runbook (triage -> fix -> advisory -> release -> post-incident)\n  CVSS severity tiers with timeline targets, regression test requirements\n\n27/27 tests pass, zero warnings.",
          "timestamp": "2026-03-04T09:11:03+04:00",
          "tree_id": "51c3ad0bc48311f55158dec5905eebe952ed651f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c4363327f6d443845bab45d34583cbf0ae6a53d1"
        },
        "date": 1772601183406,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.3,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1098.9,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 14.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.4,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 44.3,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1336,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.2,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7423.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 36781.8,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 41065.5,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 402.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 175.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 75347.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 43352.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9114.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9623.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 44145.4,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1337,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 44.7,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 147.1,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 174.6,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 402.2,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 40918.6,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 2.2,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 22.6,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 20.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 286202.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 71550.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1019039,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 63689.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 5057978,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 79030.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 165147.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 661977.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2669421.6,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19474.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41985.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 159,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 307.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 306.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 108459.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 71001.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21290.8,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1162.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20699,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 22037.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 43711.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391025.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416024.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374505.5,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1337,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 89.5,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 40918.6,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 42357.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 43352.5,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 995.3,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 44145.4,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 40918.6,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1337,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 42255.6,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 43352.5,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 1096.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 56.2,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 130100000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "5923702e81942a9ce6d2d95721c8515986324894",
          "message": "feat: CT SafeGCD scalar inverse + CI stability fixes (v3.18.0)\n\n- Replace Fermat chain (254S+40M=294 ops, ~10.6us) with Bernstein-Yang\n  CT SafeGCD (10 rounds x 59 divsteps, ~1.6us) for scalar_inverse on\n  __int128 platforms. 6.5x faster. Fermat kept as fallback.\n- CT ECDSA Sign: 26.9us -> 15.2us (1.91x vs libsecp, was 0.80x)\n- ECDSA Verify: 27.3us (1.24x vs libsecp)\n- Atomic precompute cache writes (tmp+rename) to fix CTest -j race\n- Validate cache file size on load to reject truncated files\n- Fix fuzz test buffer size for ufsecp_ecdh_xonly (33-byte compressed pubkey)\n- Remove stale win_log.txt",
          "timestamp": "2026-03-04T15:58:52+04:00",
          "tree_id": "962704f7dd32c8d9e91a6a5a785285c19aaa5426",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5923702e81942a9ce6d2d95721c8515986324894"
        },
        "date": 1772625673404,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.3,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16.1,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1104.4,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 14.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.5,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 43.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.1,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.8,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7318.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 36747.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 41046.7,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 401.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 159,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 11756.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 62809,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 43116.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9130.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9598.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 58564.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 43551.2,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 11.8,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1349.2,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 44.9,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 146.1,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 161,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 401.9,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 40813.3,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 1.9,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 29.1,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 26.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 286771.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 71692.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1024980.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 64061.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 5049371.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 78896.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 164622,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 659233,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2663353.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1962.1,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19671.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41088.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 159.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 315.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 305.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24655.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 87942.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21871.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 70925.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21341.8,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1196.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20804.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 22063.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42619.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391840.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414880.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375649.7,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1349.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 89.7,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 40813.3,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 11.8,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 42264,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 43116.1,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 852.1,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 11.8,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 43551.2,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 40813.3,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1349.2,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 42162.5,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 43116.1,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 953.6,
            "unit": "ns"
          },
          {
            "name": "---- SIGN COST DECOMPOSITION (FAST path) ----\n  ecdsa_sign = RFC6979 + k*G + field_inv + scalar_inv + scalar_muls\n    k*G (generator_mul)",
            "value": 7318.7,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    Core signing (no RFC6979)",
            "value": 9862,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_sign",
            "value": 11756.1,
            "unit": "ns"
          },
          {
            "name": "RFC6979 overhead",
            "value": 1894.1,
            "unit": "ns"
          },
          {
            "name": "sign-then-verify overhead",
            "value": 51052.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 56.3,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 129300000,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}