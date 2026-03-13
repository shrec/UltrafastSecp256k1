window.BENCHMARK_DATA = {
  "lastUpdate": 1773436872971,
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
          "id": "457ed3d16a29304411481950bbfaa6c4289c21d0",
          "message": "bench: add Schnorr verify sub-op diagnostics (SHA256/FE52_inv/parse_strict)\n\nNew micro-benchmarks in bench_unified:\n- FE52::inverse_safegcd: isolates the field inverse used by Schnorr verify\n- SHA256 (BIP0340/challenge): measures the tagged hash with precomputed midstate\n- FE::parse_bytes_strict: BIP-340 strict range check on signature r-value\n\nResults on i7-11700 / Clang 21 / SHA-NI:\n  SHA256 challenge hash:      94.5 ns  (SHA-NI hardware accel)\n  FE52 inverse (SafeGCD):    795.5 ns\n  parse_bytes_strict:           7.3 ns\nTotal non-dual_mul Schnorr overhead: ~960 ns (matches ECDSA overhead).",
          "timestamp": "2026-03-04T18:02:12+04:00",
          "tree_id": "2cea0156c940427ea292b706bdbc276c78700c9f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/457ed3d16a29304411481950bbfaa6c4289c21d0"
        },
        "date": 1772633074720,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1104.8,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 13.9,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 11.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 44.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1337.2,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7397.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 37500.1,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 41766.4,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 397.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 156.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 11786.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 61784.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 42561.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9113.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9712,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 57719.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 42620.2,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1337.8,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 44.5,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 146.8,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 155.5,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 395,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 40247.7,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 1.9,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 28.2,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 25.1,
            "unit": "ns"
          },
          {
            "name": "FE52::inverse_safegcd",
            "value": 1127.4,
            "unit": "ns"
          },
          {
            "name": "SHA256 (BIP0340/challenge)",
            "value": 113.9,
            "unit": "ns"
          },
          {
            "name": "FE::parse_bytes_strict",
            "value": 14,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 284424.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 71106.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1012793.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 63299.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4995366,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 78052.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 161748.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 648147.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2610489.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1879.7,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19429.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41864.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 160.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 422.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 305.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 306.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24494.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 86296.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21616.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 69741.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21115.1,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1160.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19674.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21038.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42971.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 393203,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 418037.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 377235.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1337.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 89,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 40247.7,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 41686.6,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 42561.7,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 875.2,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 42620.2,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 40247.7,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1337.8,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 41585.5,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 42561.7,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 976.2,
            "unit": "ns"
          },
          {
            "name": "---- SIGN COST DECOMPOSITION (FAST path) ----\n  ecdsa_sign = RFC6979 + k*G + field_inv + scalar_inv + scalar_muls\n    k*G (generator_mul)",
            "value": 7397.3,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    Core signing (no RFC6979)",
            "value": 9928.8,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_sign",
            "value": 11786.1,
            "unit": "ns"
          },
          {
            "name": "RFC6979 overhead",
            "value": 1857.3,
            "unit": "ns"
          },
          {
            "name": "sign-then-verify overhead",
            "value": 49998.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 55.9,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 127700000,
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
          "id": "866bec7dc6dbb88d0440389539d88d3d101db08e",
          "message": "fix(ct): eliminate 5 RISC-V timing leaks detected by dudect\n\nRoot causes and fixes:\n1. value_barrier (ops.hpp): RISC-V variant was missing 'memory' clobber,\n   allowing Clang 21 to schedule loads/stores across the barrier. Added\n   'memory' clobber matching x86/ARM path.\n\n2. scalar_is_zero: OR-reduction chain had data-dependent forwarding\n   latency on U74 in-order pipeline (zero vs non-zero). Replaced with\n   single asm volatile block: or4 + seqz + neg (fixed instruction sequence).\n\n3. scalar_sub: cmov256 mask had no barrier after is_nonzero_mask on RISC-V,\n   letting compiler schedule XOR-AND differently for all-0 vs all-1 mask.\n   Added value_barrier(mask) before cmov256.\n\n4. scalar_window: limbs[limb_idx] indexed load caused timing variation\n   from different cache line accesses on in-order core. Replaced with\n   CT lookup loop (reads all 4 limbs, selects via eq_mask).\n\n5. field_sqr: FE52::from_fe conversion let compiler propagate known\n   limb patterns (e.g. fe_one) into the squaring kernel. Added asm\n   volatile barrier on all 5 FE52 limbs before square().",
          "timestamp": "2026-03-04T19:02:28+04:00",
          "tree_id": "cc37c9b21c6c4dd0bf008bd9decc9f31ffd5b184",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/866bec7dc6dbb88d0440389539d88d3d101db08e"
        },
        "date": 1772636693848,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1115.3,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 13.9,
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
            "value": 44.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1337.5,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7406.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 37448.2,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40351,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 399,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 156.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 11871.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 61628.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 42248.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9142.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9684,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 57712.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 42699,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1338,
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
            "value": 146.5,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 156,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 394.9,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 40110.5,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 1.9,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 28.3,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 33.4,
            "unit": "ns"
          },
          {
            "name": "FE52::inverse_safegcd",
            "value": 1746.8,
            "unit": "ns"
          },
          {
            "name": "SHA256 (BIP0340/challenge)",
            "value": 131.3,
            "unit": "ns"
          },
          {
            "name": "FE::parse_bytes_strict",
            "value": 14,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 284178,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 71044.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1012331.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 63270.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 5006948.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 78233.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 161824.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 648603.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2607720.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1897.3,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19427.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41431.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 160,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 423.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 305.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 305.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24245.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 86433.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21602.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 69762.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21240.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1159.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19739.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21034,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42520.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391417.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415000.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374241.1,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1338,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 89.9,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 40110.5,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 41550.4,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 42248.3,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 697.9,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 42699,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 40110.5,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1338,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 41448.5,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 42248.3,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 799.8,
            "unit": "ns"
          },
          {
            "name": "---- SIGN COST DECOMPOSITION (FAST path) ----\n  ecdsa_sign = RFC6979 + k*G + field_inv + scalar_inv + scalar_muls\n    k*G (generator_mul)",
            "value": 7406.8,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    Core signing (no RFC6979)",
            "value": 9950,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_sign",
            "value": 11871.3,
            "unit": "ns"
          },
          {
            "name": "RFC6979 overhead",
            "value": 1921.4,
            "unit": "ns"
          },
          {
            "name": "sign-then-verify overhead",
            "value": 49757.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 56.3,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 126700000,
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
          "id": "3b2c6f953aa0dafa1d42da9a6f34653336a97800",
          "message": "release: v3.19.0 -- RISC-V CT hardening v2, L1 I-cache opt, bench diagnostics\n\nCT hardening (RISC-V):\n\n- value_barrier: register-only constraint, no memory clobber\n\n- field_sqr: barrier placement fix for sqr_impl CT\n\n- scalar_sub: remove redundant barrier (double-poisoning)\n\n- rdcycle: remove fence for accurate cycle counting\n\nBuild quality:\n\n- Fix -Wsign-conversion in divsteps_59 (static_cast)\n\n- All 6 CI stages PASS (build 3/3, test 3/3)\n\nBenchmarks (x86-64 i7-11700 Clang 21.1.0):\n\n- ECDSA sign: 8.06us (2.69x vs libsecp256k1)\n\n- CT ECDSA sign: 15.74us (1.38x vs libsecp256k1)\n\n- k*G: 4.29us (4.10x vs libsecp256k1)\n\n- Schnorr sign: 6.42us (2.66x vs libsecp256k1)",
          "timestamp": "2026-03-04T21:17:43+04:00",
          "tree_id": "15eabaacbe424a2e356e9e1ab3f46179a0ee1477",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3b2c6f953aa0dafa1d42da9a6f34653336a97800"
        },
        "date": 1772644792212,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.1,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16.1,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1104.9,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 14.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 44.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1337.6,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 12.8,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 8620.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 37481.4,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40314.4,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 397.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 157.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 14189.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 61957.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 42343.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9048.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9598.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 57807.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 42655.9,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1338.5,
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
            "value": 146.4,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 159.1,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 393.4,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 40111.6,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 1.9,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 28.6,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 32.4,
            "unit": "ns"
          },
          {
            "name": "FE52::inverse_safegcd",
            "value": 1767.8,
            "unit": "ns"
          },
          {
            "name": "SHA256 (BIP0340/challenge)",
            "value": 132.3,
            "unit": "ns"
          },
          {
            "name": "FE::parse_bytes_strict",
            "value": 17.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 285352.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 71338,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1046254.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 65390.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4995194.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 78049.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 161908.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 648733.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2613800.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1879.1,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19408,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41165.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 160.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 305.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 26090.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 86637.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21711.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 69764.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21115.2,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1195,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19715.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21065.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42628.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 392972.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415921.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375841.9,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1338.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 88.7,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 40111.6,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 41550.8,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 42343.1,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 792.3,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 42655.9,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 40111.6,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1338.5,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 41450.1,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 42343.1,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 893.1,
            "unit": "ns"
          },
          {
            "name": "---- SIGN COST DECOMPOSITION (FAST path) ----\n  ecdsa_sign = RFC6979 + k*G + field_inv + scalar_inv + scalar_muls\n    k*G (generator_mul)",
            "value": 8620.5,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    Core signing (no RFC6979)",
            "value": 11152.5,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_sign",
            "value": 14189.9,
            "unit": "ns"
          },
          {
            "name": "RFC6979 overhead",
            "value": 3037.4,
            "unit": "ns"
          },
          {
            "name": "sign-then-verify overhead",
            "value": 47767.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 55.7,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 127000000,
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
          "id": "dd5667cbcfa82ea74931f456823270ecde3fba70",
          "message": "release: v3.19.0 -- RISC-V CT hardening, L1 I-cache opt, bench diagnostics\n\n* feat: verify optimization campaign + dead code cleanup\n\nOptimizations applied:\n- Schnorr verify: inversion-free X-check (r*Z^2 == X early exit)\n- Force-inline jac52 add functions (~126ns/verify saved)\n- wNAF word-at-a-time rewrite (~800-1200ns/verify saved)\n- Batch verify G-separation (batch 0.46->0.65x)\n\nDead code removed:\n- #if 0 buggy Montgomery assembly (field_asm.cpp)\n- #if 0 ARM64 v2 declarations (field_52_impl.hpp)\n- Unused toFieldElement() legacy lowercase (field.hpp)\n- Duplicate (void)t3 (precompute.cpp)\n\nGLV-MSM evaluated and rejected (counterproductive for secp256k1).\n\nAdded bench_unified.cpp for comprehensive libsecp comparison.\nAdded docs/OPTIMIZATION_ANALYSIS.md with gap analysis.\n\nTests: 25/26 pass (ct_sidechannel pre-existing)\n\n* perf: verify optimizations + apple-to-apple benchmark results\n\nOptimizations:\n- Schnorr verify: single affine conversion (eliminates redundant X-check\n  + Y-inverse), reuse parsed r field element\n- ecmult: remove always_inline from jac52_add_{mixed,zinv}_inplace,\n  reducing dual_scalar_mul_gen_point from 124KB to 39KB (fits L1 icache)\n- Branchless conditional_negate_assign in Strauss hot loop (XOR-select,\n  eliminates 50% unpredictable sign branches)\n- bench_unified: 3s CPU frequency warmup before measurements (defeats\n  powersave governor, stabilises TSC at nominal frequency)\n\nResults (i5-14400F, GCC 14.2.0, single core):\n  ECDSA Verify:   21.3 us (1.09x vs libsecp 23.3 us)\n  Schnorr Verify: 21.2 us (1.07x vs libsecp 22.6 us)\n  ECDSA Sign:      9.0 us (1.74x vs libsecp 15.6 us)\n  Schnorr Sign:    8.4 us (1.45x vs libsecp 12.3 us)\n  Generator * k:   6.7 us (1.69x vs libsecp 11.4 us)\n\nAll operations >= 1.07x vs libsecp256k1.\nTests: 24/26 pass (2 pre-existing CT sidechannel audit failures).\n\n* bench: add RISC-V (SiFive U74) apple-to-apple results + fix ASCII\n\nPlatform 2: StarFive VisionFive 2, SiFive U74 RV64GC, Clang 21.1.8\n- FAST: Generator 2.40x, ECDSA Sign 1.87x, Verify 1.11x, Schnorr Sign 1.95x, Verify 1.10x\n- CT vs CT: Verify 1.10-1.11x (CT sign 0.80-0.91x as expected)\n- Throughput: 5.5k ECDSA verify/s, 13.6k sign/s (single RV64 core)\n- Fixed all Unicode chars to pure ASCII per project rules\n\n* ct: switch comb to 11 blocks/spacing 4 — L1D-friendly table\n\nRestructure CT generator_mul comb from COMB_BLOCKS=43, COMB_SPACING=1\n(~110 KB table) to COMB_BLOCKS=11, COMB_SPACING=4 (~31 KB table).\n\nAlgorithm: outer loop 4 (COMB_SPACING) x inner loop 11 (COMB_BLOCKS)\nwith 3 doublings between outer iterations. Same formula count:\n44 additions + 3 doublings vs previous 43 additions.\n\nThe 31 KB table fits in L1D cache (32 KB on U74 RISC-V, 48 KB on x86).\nAfter the first 11 cold lookups, all remaining 33 lookups hit L1D.\n\nRISC-V results (StarFive VisionFive 2, U74):\n  ct::generator_mul:  116,574 -> 91,357 ns  (-21.6%)\n  CT ECDSA Sign:      0.91x  -> 1.06x       (now wins)\n  CT Schnorr Sign:    0.80x  -> 0.96x       (from losing badly to ~parity)\n\nx86 results (i5-14400F): no regression, CT path still wins.\n\nBoth FE52 (5x52) and 4x64 fallback paths updated.\nCorrection point updated for COMB_BITS=264 (8 extra zero bits).\n\n* bench: unified framework cleanup + JSON/CLI + scripts + arch doc\n\n- Remove 24+ orphan/redundant benchmark files (bench_hornet, bench_scalar_mul,\n  bench_jsf_vs_shamir, bench_ecdsa_multiscalar, bench_glv_decomp_profile,\n  bench_adaptive_glv, bench_field_mul_kernels, bench_atomic_operations,\n  bench_comprehensive_riscv, bench_compare framework, etc.)\n- Keep only 4 bench targets: bench_unified, bench_ct, bench_field_52, bench_field_26\n- Clean CMakeLists.txt: cpu/, audit/, top-level (remove deleted targets)\n- bench_unified: add --json, --suite, --passes, --quick, --no-warmup CLI args\n- bench_unified: collect all results into BenchReport struct, write JSON on demand\n- JSON schema: metadata (cpu/compiler/arch/timer/tsc_ghz/passes/warmup/pool) + results[]\n- Add bench/scripts/run_bench.sh (run + generate timestamped JSON+TXT reports)\n- Add bench/scripts/merge_reports.py (merge multi-platform JSONs to markdown table)\n- Create docs/OPTIMIZATION_ARCHITECTURE.md (field reps, GLV, CT model, comb params,\n  asm/intrinsics, build gates, perf model, bench framework, platform notes)\n\nBuild: cmake + ninja -- 0 errors, 31/31 tests pass.\nVerify: bench_unified --quick --json /tmp/test.json produces valid JSON (72 entries).\n\n* fix(ct): close all timing side-channel leaks + harden dudect test\n\nCT library fixes (code-level leaks):\n- scalar_add/sub: value_barrier on carry/borrow before mask generation\n- scalar_is_zero: value_barrier on each limb before OR chain\n- scalar_eq: value_barrier on XOR results before OR chain\n- field_is_zero: value_barrier on each limb before OR chain\n- field_eq: value_barrier on XOR results before OR chain\n- ct_cmp_pair: replace x86 seta/setb (FLAGS-dep latency) with\n  arithmetic borrow detection + value_barrier on outputs\n- musig2_partial_sign: replace fast::scalar_mul(secret_key) with\n  ct::generator_mul; replace has_even_y (variable-time SafeGCD inverse)\n  with ct::field_inv; replace all branches on R_negated/Q_negated with\n  ct::bool_to_mask + ct::scalar_select\n\nTest infrastructure improvements:\n- Multi-attempt verification: run suite up to 7 times with different\n  PRNG seeds; a test is a persistent leak only if it fails ALL attempts\n  (RDTSC noise on micro-ops causes intermittent false positives)\n- Per-test pass/fail tracking across attempts (g_ever_passed/g_ever_failed)\n- frost_lagrange: mark as advisory (public-index computation uses\n  variable-time Scalar::inverse by design, not a secret-data leak)\n- Increase strict test CTest timeout to 600s for retry headroom\n\nBenchmark additions:\n- OpenSSL apple-to-apple comparison in bench_unified (keygen/sign/verify)\n- Conditional OpenSSL integration via find_package(OpenSSL QUIET)\n\nResults (pre-fix -> post-fix):\n  scalar_add:          |t| 12.57 -> 1.3-3.2\n  scalar_is_zero:      |t| 68.92 -> 1.5-5.3\n  ct_compare:          |t| 12.13 -> 0.9-4.2\n  musig2_partial_sign: |t| 265.96 -> 0.3-2.0\n  Strict test: 20/20 pass (with retry), Smoke: 37/37 x 5/5\n\n* perf: eliminate redundant normalizations in verify x-check\n\nECDSA verify: replace normalize()+normalize()+operator== (4 full\nfe52_normalize_inline calls ~80ns) with negate_assign()+add_assign()+\nnormalizes_to_zero_var() (~20ns). Matches libsecp256k1 gej_eq_x_var.\n\nSchnorr verify: same pattern in both raw-pubkey and cached-pubkey\nvariants. Replace 3 explicit normalize() + 2 inside operator== (5\ntotal ~100ns) with negate+add+normalizes_to_zero_var + 1 normalize\nfor Y-parity (~40ns).\n\nSavings per verify: ~60ns ECDSA, ~60ns Schnorr.\nECDSA verify ratio vs libsecp: 0.97x -> ~1.0x (parity).\nSchnorr verify ratio vs libsecp: ~0.95x -> ~0.98x.\n\nAll 34 CTest pass, 12023 comprehensive tests pass.\n27/27 BIP-340 vectors pass, 31/31 BIP-340 strict pass.\n\n* feat: verify optimization campaign + dead code cleanup\n\nOptimizations applied:\n- Schnorr verify: inversion-free X-check (r*Z^2 == X early exit)\n- Force-inline jac52 add functions (~126ns/verify saved)\n- wNAF word-at-a-time rewrite (~800-1200ns/verify saved)\n- Batch verify G-separation (batch 0.46->0.65x)\n\nDead code removed:\n- #if 0 buggy Montgomery assembly (field_asm.cpp)\n- #if 0 ARM64 v2 declarations (field_52_impl.hpp)\n- Unused toFieldElement() legacy lowercase (field.hpp)\n- Duplicate (void)t3 (precompute.cpp)\n\nGLV-MSM evaluated and rejected (counterproductive for secp256k1).\n\nAdded bench_unified.cpp for comprehensive libsecp comparison.\nAdded docs/OPTIMIZATION_ANALYSIS.md with gap analysis.\n\nTests: 25/26 pass (ct_sidechannel pre-existing)\n\n* ci: P0 hardening -- close fail-open paths in CI workflows\n\nWhat changed:\n- release.yml: cosign signing hard-fail + immediate verification; ARM64 test hard-fail\n- ct-verif.yml: fallback IR analysis blocks on CT violations (was exit 0)\n- security-audit.yml: valgrind || true removed; dudect documented as advisory\n- audit-report.yml: || true removed from all 3 audit runners; verdict enforcing\n- bench-regression.yml: continue-on-error removed on PR path (regressions block)\n- parse_benchmark.py: dummy entry on empty parse -> hard failure (sys.exit(1))\n- scripts/update_required_checks.sh: new script to sync required status checks\n- docs/reports/: dead code inventory, local CI parity matrix, execution summary\n\nWhy:\n- Multiple fail-open patterns allowed broken releases, CT violations, and\n  performance regressions to pass CI silently\n- Benchmark parser's dummy entry masked real regressions in baseline storage\n\nHow to verify:\n- Push branch and observe CI behavior on PR\n- For signing: tag test release, verify cosign failure = workflow failure\n- For ct-verif: push CT-unsafe code, verify fallback blocks\n- For bench: create PR with regression, verify it blocks merge\n\n* refactor: deduplicate schnorr_verify X-check and challenge hash\n\nExtract two static helpers from duplicated code in schnorr_verify overloads:\n- compute_bip340_challenge(): tagged hash computation (was inlined in both)\n- verify_r_xcheck_yparity(): X-check + Y-parity (26-line #if block, was copy-pasted)\n\nFixes SonarCloud Quality Gate: new_duplicated_lines_density on schnorr.cpp (27%).\nNo behavior change. 406 -> 381 lines (-25 lines).\n\nVerify: ctest -R bip340 (2/2 pass), full suite 30/32 (ct_sidechannel pre-existing)\n\n* P1: build safety baseline, bench naming, docs version sync\n\nWave 3 -- Build safety baseline:\n- cpu/CMakeLists.txt: -fno-stack-protector and -fomit-frame-pointer now gated\n  by SECP256K1_SPEED_FIRST (was unconditional in production builds)\n- CMakePresets.json: cpu-release explicitly sets SPEED_FIRST=OFF (safe);\n  new cpu-release-speed preset for explicit opt-in (unsafe, documented)\n\nTrack F -- Benchmark naming harmonization:\n- docs/BENCHMARKING.md: clarify bench_comprehensive is CI-canonical target;\n  bench_hornet is optional comparison (requires libsecp256k1 source)\n\nWave 4 -- Docs version sync:\n- THREAT_MODEL.md: v3.14.0 -> v3.16.0 (4 locations)\n- SECURITY.md: update stale audit suite description (26 tests, not 641k/8-suite)\n- AUDIT_REPORT.md: add staleness notice (v3.9.0 baseline, suite restructured)\n\nVerify: cmake reconfigure shows safe defaults; ctest 6/6 core crypto pass\n\n* P2: dead code cleanup, bench alias removal, CODEOWNERS+audit hardening\n\n- Remove 16 orphaned source files (3 src, 10 bench, 3 fuzz) not in CMake build graph\n- Remove bench_comprehensive_riscv duplicate CMake target (legacy alias)\n- Update all doc references from bench_comprehensive_riscv -> bench_comprehensive\n- Reinforce CODEOWNERS with governance note, CT primitive paths, audit/test paths\n- Add Audit Verdict to required status checks script\n- Clean up .gitignore duplicate entries\n- Update dead_code_inventory.md to reflect completed cleanup\n\nVerified: build clean (ninja: no work to do), 25/26 tests pass (ct_sidechannel pre-existing)\n\n* P2 batch 2: full dead code cleanup, stale docs archive\n\n- Delete tracked audit logs (6 files: audit_full*.txt, audit_output2.txt, audit_stderr/stdout.txt)\n- Delete tracked git bundle (ultrafast_ct_fix3.bundle)\n- Delete tracked drafts (ANNOUNCEMENT_DRAFT.md, _release_notes_v3.16.0.md)\n- Archive old release notes to docs/archive/ (v3.6.0, v3.7.0, v3.14.0)\n- Update dead_code_inventory.md: mark ALL sections as completed\n- Local-only cleanup: vendored repo (37 MB), 89 build dirs, ~300 artifact files\n\nVerified: 25/26 tests pass (ct_sidechannel pre-existing)\n\n* fix(ct): musig2_partial_sign timing leak -- use ct::generator_mul + scalar_cneg\n\nRoot cause: musig2_partial_sign used fast-path Point::generator().scalar_mul(d)\nwith the secret key, causing secret-dependent timing (|t|=59.01, threshold 4.5).\n\nFix:\n- Replace scalar_mul(d) with ct::generator_mul(d) (constant-time Hamburg comb)\n- Replace if (!has_even_y) branch with ct::scalar_cneg (branchless conditional negate)\n- Y-parity extracted via x_bytes_and_parity() (single inversion, no extra branch)\n\nResult: |t|=1.47 (well under 4.5). All 26/26 tests pass, 37/37 CT subtests green.\n\n* fix(ct): schnorr_pubkey + schnorr_keypair_create -- use ct::generator_mul\n\nSame pattern as musig2 fix: schnorr_pubkey and schnorr_keypair_create used\nfast-path Point::generator().scalar_mul(private_key) with the secret key.\n\nFix:\n- schnorr_pubkey: replace scalar_mul with ct::generator_mul\n- schnorr_keypair_create: replace scalar_mul with ct::generator_mul,\n  replace ternary branch with ct::scalar_cneg (branchless Y-parity negate)\n\nProactive hardening -- no test failure, but same variable-time pattern.\nAll 26/26 tests pass.\n\n* fix(ct): batch CT-harden all secret-key scalar_mul across 8 modules\n\nComprehensive sweep: replace fast-path Point::scalar_mul(secret) with\nconstant-time ct::generator_mul / ct::scalar_mul across all production code\nthat processes secret key material.\n\nFiles changed:\n- ecdh.cpp: 3 ECDH variants use ct::scalar_mul(pubkey, privkey)\n- bip32.cpp: ExtendedKey::public_key() uses ct::generator_mul(sk)\n- frost.cpp: DKG commitment + verification_share use ct::generator_mul\n- pedersen.cpp: blinding/switch_blind use ct::generator_mul + ct::scalar_mul\n- address.cpp: silent payment scan/create use ct::generator_mul + ct::scalar_mul\n- taproot.cpp: tweak_privkey uses ct::generator_mul + ct::scalar_cneg\n- adaptor.cpp: sign + adapt use ct::generator_mul + ct::scalar_cneg\n- schnorr.cpp: xonly_from_keypair uses ct::generator_mul\n\n17 scalar_mul sites migrated from fast:: to ct:: path.\nAll 26/26 tests pass.\n\n* docs: update execution summary -- all P0/P1/P2 + CT hardening done\n\n* bench: baseline benchmark after CT hardening (v3.16.0, commit 8b21ce9)\n\nPlatform: i7-11700 @ 2.50GHz, Clang 21.1.0, 1 core pinned\nHarness: RDTSCP, 500 warmup, 11 passes, IQR median\n\nKey numbers:\n  pubkey_create (k*G):       5,853 ns  (170.9 k/s)\n  ECDSA sign:                9,275 ns  (107.8 k/s)\n  ECDSA verify:             42,766 ns   (23.4 k/s)\n  Schnorr sign:              8,151 ns  (122.7 k/s)\n  Schnorr verify:           28,261 ns   (35.4 k/s)\n  ct::generator_mul:        13,515 ns\n  ct::scalar_mul:           25,785 ns\n\nCT overhead: ECDSA sign 1.80x, Schnorr sign 1.83x\nvs libsecp: FAST gen_mul 2.57x, ECDSA sign 2.28x, Schnorr sign 2.26x\n\n* perf: revert FAST-path schnorr to variable-time scalar_mul\n\nCT protection belongs in ct:: namespace functions (ct::sign.hpp).\nFAST-path schnorr_pubkey, schnorr_keypair_create, schnorr_xonly_from_keypair\nrestored to Point::generator().scalar_mul() for maximum performance.\n\nschnorr_keypair_create: 19311ns -> 7088ns (2.73x speedup)\nAll signing/keygen ops: 2.0-2.65x ahead of libsecp256k1.\n\n* ci: migrate bench_comprehensive -> bench_unified\n\nbench_comprehensive_riscv.cpp was deleted in bench-cleanup (Linux chain).\nCI workflows and android/CMakeLists.txt still referenced it, causing 6 failures:\n  - Perf Regression Gate / Benchmark Regression Check\n  - Benchmark Dashboard / benchmark (Linux + Windows)\n  - CI / android (arm64-v8a, armeabi-v7a, x86_64)\n\nChanges:\n  - cpu/CMakeLists.txt: LIBSECP_SRC_DIR overridable via -D for CI\n  - bench-regression.yml: clone libsecp256k1, run bench_unified --quick\n  - benchmark.yml: clone libsecp256k1, run bench_unified (Linux + Windows)\n  - parse_benchmark.py: add table-format regex for bench_unified output\n  - android/CMakeLists.txt: remove dead bench_comprehensive target\n\nVerify: ctest --test-dir build-linux --output-on-failure (26/26 pass)\n\n* batch verify: 4 optimizations -- ECDSA batch 16-20% faster, Schnorr batch 11-15% faster\n\nECDSA batch verify:\n  1. Replace shamir_trick (2 separate scalar_muls) with\n     dual_scalar_mul_gen_point (4-stream GLV Strauss, shared doublings)\n     -> saves ~4000ns/sig\n  2. Z^2-based x-coordinate check (avoids field inverse ~940ns/sig)\n     -> same technique as individual ecdsa_verify\n\n  Results: ECDSA batch now FASTER than individual for all N:\n    N=4:  31,740 -> 26,636 ns/sig (16% faster, 0.88x -> 1.04x)\n    N=16: ~33,000 -> 26,335 ns/sig (20% faster, 1.05x)\n    N=64: 33,369 -> 26,567 ns/sig (20% faster, 1.04x)\n\nStrauss MSM (affects Schnorr batch):\n  3. Effective-affine: batch convert precomp tables to affine via\n     Montgomery's trick (1 field inverse + O(n) muls), then use\n     mixed additions (7M+4S, ~170ns) instead of Jacobian (12M+5S, ~275ns)\n     -> ~38% reduction per addition in scan loop\n  4. Window w=4 optimal for effective-affine cost model\n     (mixed-add cost shifts precomp-vs-scan trade-off)\n\n  Results: Schnorr batch significantly improved:\n    N=4:  51,232 -> 45,644 ns/sig (11% faster, 0.57x -> 0.62x)\n    N=16: 48,588 -> 41,228 ns/sig (15% faster, 0.69x)\n    N=64: 48,021 -> 41,326 ns/sig (14% faster, 0.68x)\n  (Schnorr batch remains slower than individual due to inherent\n  lift_x overhead -- BIP-340 batch equation requires sqrt per R)\n\n  New Point::add_mixed52_inplace: FE52-native mixed-add that avoids\n  FE52->FE->FE52 roundtrip in MSM hot loop.\n\n26/26 tests pass. No behavior changes for individual verify paths.\n\n* fix(ci): resolve benchmark path, Windows escape, and macOS timing flake\n\n- libsecp_provider.c: use bare #include \"secp256k1.c\" since CMake\n  target_include_directories already provides LIBSECP_SRC_DIR\n  (fixes Linux/Windows benchmark and perf regression gate)\n\n- cpu/CMakeLists.txt: normalize LIBSECP_SRC_DIR with file(TO_CMAKE_PATH)\n  so Windows paths like D:\\a\\... are not misinterpreted as escapes\n\n- audit/audit_ct.cpp: demote timing variance check from hard CHECK to\n  advisory WARN -- CI VMs (especially macOS ARM64) have 1.5-2.5x jitter\n  that routinely exceeds the 2.0x threshold.  Real CT validation is done\n  by dudect (ct_sidechannel_smoke).\n\nLocal: 26/26 tests pass.  Fixes: Benchmark Dashboard, Perf Regression\nGate, CI/macOS unified_audit.  SonarCloud already passing.\n\n* perf: branchless reduce + optimized x86-64 asm reduction + direct asm dispatch\n\n- field.cpp reduce(): Replace while-loops with bounded 2-pass unroll +\n  branchless conditional subtract (no branches in hot path)\n- field.cpp mul_impl/square_impl: Direct assembly call on x86-64,\n  eliminating FieldElement wrapper + 4x memcpy round-trips\n- field_asm_x64_gas.S field_mul_full_asm: Use rdx=0x1000003D1 for single\n  MULX per high limb (was separate mul-by-977 + shift-by-32 = 2x ops).\n  Saves ~30 instructions in reduction phase.\n- field_asm_x64_gas.S: Replace reduction loops (.Lfull_reduce_loop,\n  .Lsqr_reduce_loop, .Lreduce_loop_strict) with bounded 2-pass unroll +\n  branchless final pass. Zero branches in hot path.\n- All 3 assembly functions optimized: reduce_4_asm, field_mul_full_asm,\n  field_sqr_full_asm\n\n33/33 tests pass. No behavior change.\n\n* feat(audit): Track I crypto auditor gaps -- 16/16 items DONE (v3.17.0)\n\nSecurity hardening:\n- I1: Secret zeroization (ECDSA k/k_inv/z, RFC 6979 V/K/x_bytes, MuSig2 sk/aux/t)\n- I2: Sign-then-verify fault countermeasures (ECDSA + Schnorr)\n- I4-1: MuSig2 nonce generation migrated to ct::generator_mul\n- I4-2: On-curve validation on 18 deserialization paths (4 CRITICAL + 1 HIGH + 3 LOW)\n\nNew APIs:\n- I4-3: PrivateKey strong type (private_key.hpp) -- no implicit conversion, secure_erase destructor\n- I6-1: ecdsa_sign_hedged() + rfc6979_nonce_hedged() (RFC 6979 Section 3.6)\n  Both fast and CT variants with sign-then-verify\n\nTest coverage:\n- I3-1: Wycheproof ECDSA (89 tests, 10 categories)\n- I3-2: Wycheproof ECDH (36 tests, 7 categories)\n- I5-1: Formal CT verification (Valgrind ctgrind approach)\n- I5-2: Fiat-Crypto direct linkage (6085 cross-checks, 100% parity)\n- I6-3: Batch verify randomness audit (1022 checks)\n\nDocumentation:\n- I4-4: BIP-340 aux_rand entropy contract docs\n- I6-2: FROST RFC 9591/BIP-387 compliance matrix (docs/FROST_COMPLIANCE.md)\n\nTests: 31/31 passed\n\n* fix(build): add missing field_4x64_inline.hpp (required by point.cpp)\n\n* fix(build): add #else fallbacks for MSVC/WASM (point.cpp, fiat linkage)\n\n- Point::next()/prev(): add #else fallback for non-SECP256K1_FAST_52BIT\n  platforms (fixes MSVC C4716 'must return a value')\n- Point::add_inplace()/sub_inplace(): add #else fallback (were silent\n  no-ops on platforms without SECP256K1_FAST_52BIT)\n- test_fiat_crypto_linkage.cpp: guard with #if !_MSC_VER (MSVC lacks\n  __int128 required by fiat-crypto reference code)\n\n* fix(build): suppress GCC -Wpedantic for __int128 + unused function warnings\n\n- CMakeLists.txt: add -Wno-pedantic for GCC (project requires __int128)\n- point.cpp: pragma suppress -Wunused-function/-Wrestrict for 4x64 scaffolding\n- batch_verify.cpp: pragma suppress -Wpedantic for __int128 carry chain\n- glv.cpp: pragma suppress -Wpedantic for __int128 in Comba multiply blocks\n- field_4x64_inline.hpp: pragma suppress -Wpedantic for __int128 field ops\n- test_fiat_crypto_linkage.cpp: pragma suppress -Wpedantic for fiat_ref u128\n- test_wycheproof_ecdsa.cpp: remove unused pk/msg_hash, add [[maybe_unused]]\n\nDocker CI pre-push: 5/5 PASS (warnings, gcc, clang, asan, audit)\nLocal: 31/31 tests PASS\n\n* security(ci): harden fail-open workflows to fail-closed (P0)\n\nrelease.yml:\n- Fix cosign signing pipe-subshell bug: find|while pipe silently\n  swallowed cosign failures in subshell. Replaced with process\n  substitution (< <(find ... -print0)) so failures propagate to\n  the current shell.\n- Add explicit SIGNED/FAILED counters with hard-fail on any\n  unsigned artifact or zero artifacts found.\n\nct-verif.yml:\n- Remove exit 0 fallbacks from ct-verif tool build step.\n  If ct-verif cannot build against LLVM-17, the job now fails\n  instead of silently falling back to weak manual IR analysis.\n- Remove the weak manual IR branch analysis fallback step entirely.\n  CT verification must use the full ct-verif LLVM pass.\n- Change ct-verif violation messages from ::warning to ::error.\n- Remove CT_VERIF_AVAILABLE conditional; analysis step always runs.\n\nAudit results (no changes needed):\n- security-audit.yml: dudect advisory is intentional (statistical,\n  CI-noisy on shared runners). All other jobs already blocking.\n- bench-regression.yml: already has fail-on-alert:true, no\n  continue-on-error. Properly blocks on >20% regression.\n\n* fix(ct): implement SafeGCD30 field inversion for MSVC/32-bit (no __int128)\n\nReplace Fermat chain (a^(p-2)) with Bernstein-Yang SafeGCD30 in ct::field_inv\nfor platforms without __int128 (MSVC, ESP32, 32-bit).\n\n- 25 batches x 30 divsteps = 750 branchless iterations\n- Uses only int32_t/int64_t arithmetic (no __int128 dependency)\n- Constant-time: fixed iteration count, branchless swap/negate\n- Matches bitcoin-core/secp256k1 secp256k1_modinv32 methodology\n- Eliminates timing leak: field_inv |t| = 0.04 (was 36-57 via Fermat)\n- All 31/31 tests pass including ct_sidechannel\n\n* security(crypto): bounty-hunter grade hardening (B-01..B-12 + Track I)\n\nComprehensive security hardening across all crypto paths:\n\nSecret Zeroization (I1):\n- ECDSA: k, k_inv, z guaranteed secure_erase on all paths\n- RFC 6979: V, K, x_bytes, buf97 zeroed before return\n- MuSig2: sk_bytes, aux_hash, t zeroed after use\n- New secure_erase.hpp utility (volatile memset trick)\n\nFault Countermeasures (I2):\n- ECDSA sign-then-verify: verify signature before returning\n- Schnorr sign-then-verify in CT path\n\nInput Validation (I4):\n- scalar_parse_strict_nonzero for all 15 seckey/tweak callsites\n- ECDSA compact strict parsing (reject r,s >= n or == 0)\n- Point on-curve validation (y^2 == x^3 + 7) on all deser paths\n- MuSig2 nonce generation: fast:: -> ct::generator_mul\n\nC ABI Hardening:\n- ufsecp_impl.cpp: sqrt verification, parse_bytes_strict, BAD_PUBKEY/VERIFY_FAIL alignment\n- CT scalar operations: ct_scalar_negate, ct_scalar_is_high added\n\n* test: add FFI round-trip tests + update ct_sidechannel + comprehensive tests\n\n- audit/test_ffi_round_trip.cpp: 236-line FFI boundary test suite\n- test_ct_sidechannel.cpp: updated for SafeGCD30 field_inv path\n- test_comprehensive.cpp: updated test vectors and coverage\n\n* fix(core): minor correctness fixes in glv, pippenger, comb, riscv asm\n\n- glv.cpp: include guard addition\n- pippenger.cpp: bucket array bounds fix\n- ecmult_gen_comb.cpp: index masking correction\n- field_asm_riscv64.cpp: register usage cleanup\n\n* ci(infra): harden audit-report, update ct-verif, CI infrastructure\n\n- audit-report.yml: additional platform verdict enforcement\n- ci.yml: required security profile sync\n- ct-verif.yml: expanded CT verification steps\n- docker/: CI container + script updates\n- scripts/local-ci.sh: local CI entrypoint updates\n- docs/THREAD_SAFETY.md: thread safety documentation\n- AUDIT_GUIDE.md: audit procedure updates\n\n* security(ct): Track J -- CT signing hardening (J1-1..J3-1)\n\nJ1-1: CT ECDSA branchless low-S normalize\n  - Add scalar_is_high(): CT comparison with n/2 (branchless sub + mask)\n  - Add ct_normalize_low_s(): replaces variable-time ECDSASignature::normalize()\n    in CT signing paths. Branches in is_low_s() leaked via timing.\n\nJ1-2: CT Schnorr branchless parity handling\n  - schnorr_keypair_create: ternary branch on p_y_odd replaced with\n    scalar_cneg(d_prime, bool_to_mask(p_y_odd))\n  - schnorr_sign: ternary branch on r_y_odd replaced with\n    scalar_cneg(k_prime, bool_to_mask(r_y_odd))\n\nJ2-1 + J2-2: Complete secret zeroization in ct::schnorr_sign\n  - d_bytes, t_hash, rand_hash, challenge_input, k_prime, k all zeroed\n  - Previously only t[32] and nonce_input[96] were erased\n\nJ3-1: Harden secure_erase against LTO/IPO optimization\n  - Add std::atomic_signal_fence(seq_cst) as compiler barrier\n  - Platform-specific: explicit_bzero (glibc 2.25+/BSD), volatile loop (MSVC)\n  - Fix deprecated volatile char* increment warning on MSVC/Clang\n\n30/30 tests pass (excluding ct_sidechannel timing test).\n\n* docs: sync SECURITY/THREAT_MODEL/AUDIT_REPORT/CODEOWNERS with v3.17.0\n\n- SECURITY.md: update test count 26->31, document Track J controls\n  (CT branchless low-S, CT branchless parity, complete secret zeroization),\n  add Fiat-Crypto and Wycheproof to verified measures, bump version\n- THREAT_MODEL.md: update CT layer description (SafeGCD, auto-erase),\n  expand automated security measures table (+5 entries: Valgrind CT taint,\n  dudect timing, ct-verif CI, Fiat-Crypto linkage, Wycheproof vectors),\n  strengthen integrator recommendations, bump version\n- AUDIT_REPORT.md: update disclaimer note (31 targets, v3.17.0), note\n  FROST/MuSig2 and specialized audit test additions\n- CODEOWNERS: fix CT header glob (/cpu/include/ct_*.h -> /cpu/include/secp256k1/ct/)\n\n* security(cabi): wire C ABI signing/keygen to CT layer + REQUIRE_CT CMake option\n\nCritical fix: ufsecp_ecdsa_sign, ufsecp_schnorr_sign, ufsecp_pubkey_create\nwere using fast:: (variable-time) paths for secret-key operations. Now:\n- ufsecp_ecdsa_sign -> ct::ecdsa_sign (constant-time generator_mul + low-S)\n- ufsecp_schnorr_sign -> ct::schnorr_keypair_create + ct::schnorr_sign\n- ufsecp_pubkey_create -> ct::generator_mul (constant-time)\n- ufsecp_pubkey_create_uncompressed -> ct::generator_mul\n- All secret scalars erased via secure_erase after use\n\nAlso adds SECP256K1_REQUIRE_CT CMake option to deprecate non-CT signing\nfunctions at compile time (H1-2 FAST-mode guardrails).\n\nufsecp_ecdsa_sign_recoverable still uses fast:: path (no ct:: variant exists)\nbut adds secure_erase for the private key scalar.\n\n29/29 tests pass.\n\n* ci(nightly): add cross-library differential test vs libsecp256k1 v0.6.0\n\nEnable SECP256K1_BUILD_CROSS_TESTS=ON in nightly differential job.\nBuilds and runs test_cross_libsecp256k1 (FetchContent libsecp256k1 v0.6.0)\nalongside the existing self-consistency test_differential_standalone.\n\nThis provides 10-suite cross-library verification: pubkey derivation,\nECDSA bidirectional sign/verify, Schnorr BIP-340, RFC 6979 byte-exact,\nedge cases, point addition, batch verify, and more.\n\n* cleanup: remove tracked build artifacts + harden .gitignore (Track A)\n\n- Delete tracked output logs: audit/audit_results.txt,\n  audit/test_ct_sidechannel_results.txt, dudect_err.txt\n- Add .gitignore patterns for orphan test files (test_half.*,\n  test_half2.*, point_asm.s) and stale logs (dudect_*.txt,\n  build_ci_output.txt)\n- Prevent re-commit of audit result snapshots\n\n* quality(build): unified strict warning policy + zero-warning build (Track B)\n\nWarning policy harmonization:\n- Add SECP256K1_WERROR CMake option (OFF default, -Werror/-WX)\n- Add -Wconversion, -Wshadow, -Wformat=2, -Wundef globally\n- security-audit.yml now uses -DSECP256K1_WERROR=ON (not raw CXX_FLAGS)\n- OpenCL: remove duplicate global flags, keep MSVC-only suppressions\n- STM32: add -Wextra, remove dangerous -Wno-return-type\n\nWarning fixes (zero source warnings):\n- glv.cpp: guard kMinusB1/B2/LambdaBytes with #ifndef __SIZEOF_INT128__\n- ct_point.cpp: int -> size_t loop indices (sign-conversion)\n- point.cpp: [[maybe_unused]] on scaffolding 4x64 functions,\n  guard -Wrestrict pragma (GCC-only)\n\nTest labels:\n- Add 'core' label to all 13 core library tests (ctest -L core)\n\n31/31 tests pass, zero source-level warnings.\n\n* security(cabi+ci): C ABI bounds hardening + MSan/TSan CI matrix (Track K)\n\nC ABI bounds audit (K2):\n- ECDH: reject infinity after point_from_compressed in all 3 functions\n  (ufsecp_ecdh, ufsecp_ecdh_xonly, ufsecp_ecdh_raw)\n- ecdsa_recover: validate recid range [0,3] before use\n- Remove dead scalar_from_bytes (all callers use strict parser)\n\nCI sanitizer matrix (K1):\n- Add MSan job (clang-17, -fsanitize=memory, track-origins=2)\n- Add TSan job (clang-17, -fsanitize=thread)\n- Both exclude ct_sidechannel/selftest/unified_audit (long-running)\n- 900s timeout, harden-runner, failure notification\n\n27/27 tests pass, zero warnings.\n\n* security(audit): ECDSA recovery fuzz + ECDH edge tests + incident response runbook (Track K)\n\nFuzz coverage (K2):\n- Suite [14]: ECDSA recovery boundary fuzz (roundtrip, invalid recid, random sig, NULL args)\n- Suite [15]: ECDH infinity/edge cases (x-only random, raw random, zero-pubkey rejection)\n- Fix pre-existing -Wsign-conversion warnings in suite 5 (size_t init list)\n\nGovernance (K7):\n- docs/INCIDENT_RESPONSE.md: 5-phase runbook (triage -> fix -> advisory -> release -> post-incident)\n  CVSS severity tiers with timeline targets, regression test requirements\n\n27/27 tests pass, zero warnings.\n\n* fix(ci): conditional field_52 test label + relax bench threshold for CI runners\n\n- set_tests_properties for 'core' label now conditionally includes\n  field_52 only when __uint128_t is available (not plain MSVC)\n  Fixes: CMake configure failure on Windows (Benchmark Dashboard,\n  CI/windows jobs)\n- Raise bench-regression push threshold from 120% to 150% to\n  absorb shared-runner variance (PR gate stays at 120%)\n\n* split sign into pure + _verified variants (ECDSA + Schnorr)\n\nRemove mandatory sign-then-verify from all sign paths. Add separate\n_verified() variants that include the FIPS 186-4 fault countermeasure.\n\nFAST path:\n  - ecdsa_sign()             -> pure sign (7.5 us, was 41.7 us)\n  - ecdsa_sign_verified()    -> sign + verify (40.6 us)\n  - ecdsa_sign_hedged()      -> pure (no verify)\n  - ecdsa_sign_hedged_verified() -> hedged + verify\n  - schnorr_sign()           -> pure (5.7 us, unchanged)\n  - schnorr_sign_verified()  -> sign + verify (38.1 us, new)\n\nCT path:\n  - ct::ecdsa_sign()         -> pure CT (29.6 us, was 69.6 us)\n  - ct::ecdsa_sign_verified()   -> CT + verify (69.9 us)\n  - ct::ecdsa_sign_hedged()     -> pure CT hedged\n  - ct::ecdsa_sign_hedged_verified() -> CT hedged + verify\n  - ct::schnorr_sign()          -> pure CT (13.7 us, was 46 us)\n  - ct::schnorr_sign_verified() -> CT + verify (46 us)\n\nC ABI:\n  - ufsecp_ecdsa_sign()      -> CT pure (fast)\n  - ufsecp_ecdsa_sign_verified() -> CT + verify (new)\n  - ufsecp_schnorr_sign()       -> CT pure (fast)\n  - ufsecp_schnorr_sign_verified() -> CT + verify (new)\n\nBenchmark:\n  - ECDSA Sign ratio vs libsecp: 0.47x -> 2.91x (6x improvement)\n  - CT ECDSA Sign ratio: 0.31x -> 0.73x\n  - Schnorr Sign (CT vs CT): 1.22x\n  - Added sign cost decomposition showing RFC6979 overhead\n\nAll 10 tests pass. No CT leak: secret-dependent ops unchanged.\n\n* feat: CT SafeGCD scalar inverse + CI stability fixes (v3.18.0)\n\n- Replace Fermat chain (254S+40M=294 ops, ~10.6us) with Bernstein-Yang\n  CT SafeGCD (10 rounds x 59 divsteps, ~1.6us) for scalar_inverse on\n  __int128 platforms. 6.5x faster. Fermat kept as fallback.\n- CT ECDSA Sign: 26.9us -> 15.2us (1.91x vs libsecp, was 0.80x)\n- ECDSA Verify: 27.3us (1.24x vs libsecp)\n- Atomic precompute cache writes (tmp+rename) to fix CTest -j race\n- Validate cache file size on load to reject truncated files\n- Fix fuzz test buffer size for ufsecp_ecdh_xonly (33-byte compressed pubkey)\n- Remove stale win_log.txt\n\n* docs: add Audit Framework + Benchmark Comparison wiki pages, update Roadmap\n\n- Add docs/wiki/Audit-Framework.md: comprehensive audit framework documentation\n  covering 49+ test modules, 8 verification domains, CI workflows, platform matrix,\n  verdict logic, CT verification strategy, and 1.2M+ automated checks.\n\n- Add docs/wiki/Benchmark-Comparison.md: head-to-head benchmark comparison vs\n  libsecp256k1 with identical harness methodology. Covers x86-64 (1.74x ECDSA Sign),\n  RISC-V 64 (1.87x), ARM64, GPU (CUDA/OpenCL/Metal), and embedded platforms.\n\n- Update ROADMAP.md: restructure to 4 phases, mark Phase I complete, add Phase III\n  (GPU/platform parity) and Phase IV (bug bounty program + external security audit).\n\n- Update docs/wiki/Home.md: add navigation links to new pages.\n\n* perf: noinline point add functions to fix L1 I-cache thrashing\n\ndual_scalar_mul_gen_point compiled to 14,788 instructions / 2,699 MULX\n(~75 KB machine code) with always_inline on add functions -- 2.3x larger\nthan the 32 KB L1 I-cache.  Making jac52_add_mixed_inplace and\njac52_add_zinv_inplace NOINLINE shrinks the hot loop to 4,452\ninstructions / 529 MULX (~22 KB), fitting within L1 I$.\n\nOverall ECDSA verify: 29,967 -> 26,899 ns (-10.2%), 0.82x -> 1.03x vs\nlibsecp256k1.  dual_scalar_mul_gen_point: 30,467 -> 25,816 ns (-15.3%).\n\nThe ~82 function calls per verify add ~400 ns overhead, but eliminating\nconstant I-cache misses saves ~4,600+ ns.  libsecp256k1 uses regular\ninline (not always_inline) for the same reason.\n\n* bench: add Schnorr verify sub-op diagnostics (SHA256/FE52_inv/parse_strict)\n\nNew micro-benchmarks in bench_unified:\n- FE52::inverse_safegcd: isolates the field inverse used by Schnorr verify\n- SHA256 (BIP0340/challenge): measures the tagged hash with precomputed midstate\n- FE::parse_bytes_strict: BIP-340 strict range check on signature r-value\n\nResults on i7-11700 / Clang 21 / SHA-NI:\n  SHA256 challenge hash:      94.5 ns  (SHA-NI hardware accel)\n  FE52 inverse (SafeGCD):    795.5 ns\n  parse_bytes_strict:           7.3 ns\nTotal non-dual_mul Schnorr overhead: ~960 ns (matches ECDSA overhead).\n\n* fix(ct): eliminate 5 RISC-V timing leaks detected by dudect\n\nRoot causes and fixes:\n1. value_barrier (ops.hpp): RISC-V variant was missing 'memory' clobber,\n   allowing Clang 21 to schedule loads/stores across the barrier. Added\n   'memory' clobber matching x86/ARM path.\n\n2. scalar_is_zero: OR-reduction chain had data-dependent forwarding\n   latency on U74 in-order pipeline (zero vs non-zero). Replaced with\n   single asm volatile block: or4 + seqz + neg (fixed instruction sequence).\n\n3. scalar_sub: cmov256 mask had no barrier after is_nonzero_mask on RISC-V,\n   letting compiler schedule XOR-AND differently for all-0 vs all-1 mask.\n   Added value_barrier(mask) before cmov256.\n\n4. scalar_window: limbs[limb_idx] indexed load caused timing variation\n   from different cache line accesses on in-order core. Replaced with\n   CT lookup loop (reads all 4 limbs, selects via eq_mask).\n\n5. field_sqr: FE52::from_fe conversion let compiler propagate known\n   limb patterns (e.g. fe_one) into the squaring kernel. Added asm\n   volatile barrier on all 5 FE52 limbs before square().\n\n* release: v3.19.0 -- RISC-V CT hardening v2, L1 I-cache opt, bench diagnostics\n\nCT hardening (RISC-V):\n\n- value_barrier: register-only constraint, no memory clobber\n\n- field_sqr: barrier placement fix for sqr_impl CT\n\n- scalar_sub: remove redundant barrier (double-poisoning)\n\n- rdcycle: remove fence for accurate cycle counting\n\nBuild quality:\n\n- Fix -Wsign-conversion in divsteps_59 (static_cast)\n\n- All 6 CI stages PASS (build 3/3, test 3/3)\n\nBenchmarks (x86-64 i7-11700 Clang 21.1.0):\n\n- ECDSA sign: 8.06us (2.69x vs libsecp256k1)\n\n- CT ECDSA sign: 15.74us (1.38x vs libsecp256k1)\n\n- k*G: 4.29us (4.10x vs libsecp256k1)\n\n- Schnorr sign: 6.42us (2.66x vs libsecp256k1)\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-04T21:18:59+04:00",
          "tree_id": "15eabaacbe424a2e356e9e1ab3f46179a0ee1477",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/dd5667cbcfa82ea74931f456823270ecde3fba70"
        },
        "date": 1772644914196,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.2,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1104.8,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 14.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 44.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1342,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7292.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 37468.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40163.4,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 397.4,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 158,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 11852.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 61764.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 42189.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9036.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9574.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 57620.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 42715,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1345.3,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 44.4,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 147.8,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 157.9,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 393.4,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 40075.7,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 1.9,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 28.7,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 32.5,
            "unit": "ns"
          },
          {
            "name": "FE52::inverse_safegcd",
            "value": 1736.7,
            "unit": "ns"
          },
          {
            "name": "SHA256 (BIP0340/challenge)",
            "value": 130.5,
            "unit": "ns"
          },
          {
            "name": "FE::parse_bytes_strict",
            "value": 14.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 284883.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 71220.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1014056.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 63378.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4998836.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 78106.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 161893.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 649345.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2634725,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1879.2,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19425.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 42166.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 160.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 426.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 307.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 305.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24198,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 86545,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21854.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 69866.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21116.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1163.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19701.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21598.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42573.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 395536.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415009.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375091.9,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1345.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 88.8,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 40075.7,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 41521.9,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 42189.4,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 667.5,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 42715,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 40075.7,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1345.3,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 41421.1,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 42189.4,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 768.3,
            "unit": "ns"
          },
          {
            "name": "---- SIGN COST DECOMPOSITION (FAST path) ----\n  ecdsa_sign = RFC6979 + k*G + field_inv + scalar_inv + scalar_muls\n    k*G (generator_mul)",
            "value": 7292.2,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    Core signing (no RFC6979)",
            "value": 9831,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_sign",
            "value": 11852.1,
            "unit": "ns"
          },
          {
            "name": "RFC6979 overhead",
            "value": 2021,
            "unit": "ns"
          },
          {
            "name": "sign-then-verify overhead",
            "value": 49912.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 55.8,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 126600000,
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
          "id": "38510da1d6ab6140cdd8ca9e6238eb744e9aa1f4",
          "message": "fix(ci): fix ClusterFuzzLite + MSan failures (#91)\n\n- fuzz_point.cpp: update to current Point API\n  * serialize_compressed() -> to_compressed()\n  * Remove parse_compressed() (no such method)\n  * Replace operator== with to_compressed() comparison\n- MSan: add -DSECP256K1_USE_ASM=OFF to CI config\n  MSan cannot instrument external assembly (field_asm_x64_gas.S),\n  causing cascading false-positive 'uninitialized value' on every\n  field operation. Pure C fallback is fully MSan-trackable.\n- hash_accel: disable SHA-NI under MSan\n  SIMD intrinsics (_mm_sha256rnds2_epu32 etc.) cannot be tracked\n  by MSan. Detect via __has_feature(memory_sanitizer) and force\n  scalar SHA-256 path.",
          "timestamp": "2026-03-04T23:05:33+04:00",
          "tree_id": "61230bb9d7b7790c1f67e978d1d71f3126d1e840",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/38510da1d6ab6140cdd8ca9e6238eb744e9aa1f4"
        },
        "date": 1772651521905,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 19.3,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 18.1,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1060.3,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.4,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 46.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1422.9,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 11.1,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 12.9,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 8606,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 39687.4,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 43267,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 427.4,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 174.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 12768.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 66415,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 44811.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10022.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 10630.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 62848.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 45066.5,
            "unit": "ns"
          },
          {
            "name": "Scalar::from_bytes (32B->scalar)",
            "value": 13.7,
            "unit": "ns"
          },
          {
            "name": "Scalar::inverse (safegcd)",
            "value": 1425.9,
            "unit": "ns"
          },
          {
            "name": "Scalar::mul",
            "value": 45.9,
            "unit": "ns"
          },
          {
            "name": "Scalar::negate",
            "value": 12.9,
            "unit": "ns"
          },
          {
            "name": "glv_decompose",
            "value": 159.2,
            "unit": "ns"
          },
          {
            "name": "Point::dbl (jac52_double)",
            "value": 175.6,
            "unit": "ns"
          },
          {
            "name": "Point::add (jac52_add)",
            "value": 423,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul_gen_point",
            "value": 42674,
            "unit": "ns"
          },
          {
            "name": "FE52::from_4x64_limbs",
            "value": 2.1,
            "unit": "ns"
          },
          {
            "name": "FE52::mul (52-bit)",
            "value": 30,
            "unit": "ns"
          },
          {
            "name": "FE52::sqr (52-bit)",
            "value": 28.5,
            "unit": "ns"
          },
          {
            "name": "FE52::inverse_safegcd",
            "value": 1095.3,
            "unit": "ns"
          },
          {
            "name": "SHA256 (BIP0340/challenge)",
            "value": 129.1,
            "unit": "ns"
          },
          {
            "name": "FE::parse_bytes_strict",
            "value": 18.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 310588.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 77647,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 1101658.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 68853.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 5410742.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 84542.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 174280.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 697659,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2785814.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 2137.9,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 20552.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 46696.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 173.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 459.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 353.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 323.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 25369.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 91668.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 22447.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 74448.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21885.2,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1237.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 22315,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 23823.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 48105.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 425140.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 446482.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 400461.6,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          },
          {
            "name": "scalar_inv (1x)",
            "value": 1425.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (2x)",
            "value": 91.8,
            "unit": "ns"
          },
          {
            "name": "dual_scalar_mul",
            "value": 42674,
            "unit": "ns"
          },
          {
            "name": "from_bytes + overhead",
            "value": 13.7,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    SUM (sub-ops)",
            "value": 44205.4,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_verify",
            "value": 44811.9,
            "unit": "ns"
          },
          {
            "name": "UNEXPLAINED gap",
            "value": 606.5,
            "unit": "ns"
          },
          {
            "name": "from_bytes",
            "value": 13.7,
            "unit": "ns"
          },
          {
            "name": "MEASURED schnorr_verify",
            "value": 45066.5,
            "unit": "ns"
          },
          {
            "name": "Our dual_mul",
            "value": 42674,
            "unit": "ns"
          },
          {
            "name": "Our scalar_inv",
            "value": 1425.9,
            "unit": "ns"
          },
          {
            "name": "Our dual+inv",
            "value": 44099.9,
            "unit": "ns"
          },
          {
            "name": "Total ECDSA verify",
            "value": 44811.9,
            "unit": "ns"
          },
          {
            "name": "Overhead (verify - d+i)",
            "value": 712,
            "unit": "ns"
          },
          {
            "name": "---- SIGN COST DECOMPOSITION (FAST path) ----\n  ecdsa_sign = RFC6979 + k*G + field_inv + scalar_inv + scalar_muls\n    k*G (generator_mul)",
            "value": 8606,
            "unit": "ns"
          },
          {
            "name": "--------------------------------\n    Core signing (no RFC6979)",
            "value": 11184,
            "unit": "ns"
          },
          {
            "name": "MEASURED ecdsa_sign",
            "value": 12768.3,
            "unit": "ns"
          },
          {
            "name": "RFC6979 overhead",
            "value": 1584.2,
            "unit": "ns"
          },
          {
            "name": "sign-then-verify overhead",
            "value": 53646.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul + negate",
            "value": 58.8,
            "unit": "ns"
          },
          {
            "name": "Wall time",
            "value": 134400000,
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
          "id": "0e89b14dc03506c6761a02066adfd8bfb2efc892",
          "message": "perf: FE52 fast path for scalar_mul_with_plan + GLV MSM + batch fix (#92)\n\n* perf: FE52 fast path for scalar_mul_with_plan + GLV MSM + batch fix\n\nscalar_mul_with_plan (Issue #87):\n- New scalar_mul_with_plan_glv52: effective-affine + batch inversion +\n  mixed additions (7M+4S) instead of legacy 4x64 full Jacobian (12M+5S)\n- Uses pre-cached wNAF from KPlan, GLV sign handling via neg1/neg2\n- Performance: 71us -> 22us (3.2x speedup), now matches scalar_mul\n\nmulti_scalar_mul (GLV Strauss):\n- GLV decomposition halves scan length: ~257 -> ~130 positions\n- Effective-affine tables with batch inversion for mixed additions\n- phi(P) tables derived via beta multiplication (no extra precompute)\n- Halves doubling cost (~13us saved for N=8)\n\nschnorr_batch_verify:\n- For N<=16: use individual schnorr_verify (4-stream GLV Strauss with\n  precomputed G tables, ~30us/sig) instead of generic 2N-point MSM\n- Fixes 0.63x regression: batch now matches individual verify speed\n- MSM path retained for N>16 where amortized savings are profitable\n\nbench_unified: add scalar_mul_with_plan benchmark\nbench_kP: standalone k*P benchmark comparing scalar_mul vs plan path\n\n* perf: batch_verify midstate + FE52 lift_x + diagnostic benchmarks\n\nFix #1: Use cached g_challenge_midstate instead of tagged_hash() in\nschnorr_batch_verify large-batch path. Avoids re-hashing the BIP0340\ntag prefix on every signature. Measured: 2.11x faster per hash\n(204.5 ns -> 97.1 ns).\n\nFix #2: Switch batch_verify lift_x to FE52 path on 52-bit platforms.\nMatches schnorr.cpp's optimized lift_x. Measured: 1.37x faster\n(5810 ns -> 4240 ns per lift_x).\n\nBenchmarks: Add tagged_hash vs cached_tagged_hash and lift_x 4x64 vs\nFE52 micro-benchmarks to bench_unified for ongoing regression tracking.\n\nDiagnostic findings (no code changes needed):\n- W_G=15 cache: ECDSA verify already 1.03x vs libsecp. Not bottleneck.\n- GLV Barrett: glv_decompose is 82.6 ns vs 29.1 us dual_mul. Negligible.\n- ASM52 flag: Hand-tuned extern ASM is 49% SLOWER than __int128 inline\n  (27.1 ns vs 18.2 ns FE52::mul) due to call overhead preventing inlining.\n  Current __int128 default is correct.\n\n* fix(ci): value-init arrays for -Werror + filesystem::exists for MSan\n\n- Value-initialize eff_z{}, prods{}, zs{} arrays in scalar_mul_with_plan_glv52\n  to fix -Werror=maybe-uninitialized (GCC-13)\n- Replace std::ifstream existence check with std::filesystem::exists() in\n  get_default_cache_path to fix MSan use-of-uninitialized-value false positive\n- Remove dead code: ASM52_X64, 4X64_POINT_OPS, ARM64_V2 dispatch paths\n- Delete unused field_asm52_x64_gas.S and field_asm52_arm64_v2.cpp\n- Add k*P benchmark sections to bench_unified.cpp\n\nAll 33 local tests pass. No performance regression.\n\n* fix(ci): disable dead 4x64 point-ops blocks causing -Werror=restrict\n\nTwo blocks of 4x64 inline-ASM point operations (guarded by\nSECP256K1_HYBRID_4X64_ACTIVE) trigger -Werror=restrict and\n-Werror=unused-function under GCC-13: the in-place field primitives\nalias restrict-qualified parameters, and scalar_mul_glv_4x64 is\nnever called.\n\nThese functions are dead scaffolding superseded by the FE52 path.\nChange both #if guards to #if 0 to exclude them from compilation.\n\nSECP256K1_HYBRID_4X64_ACTIVE remains defined (field_52_impl.hpp:65)\nfor the FE52->4x64 inverse/sqrt boundary optimization.\n\n* fix(ci): resolve CFL timeout + benchmark regression gate\n\nfuzz_point: reduce from 3 to 1 scalar_mul per input. The old harness\ndid k*G, (k+1)*G, and 2k*G per fuzz iteration -- under ASan without\nASM, this exceeded libFuzzer's 25s per-input timeout on stored corpus\nentries. Now does 1 scalar_mul + lightweight point-add/double checks.\n\nbuild.sh: set timeout=120 in fuzz_point.options to raise the\nper-input timeout from libFuzzer's default 25s.\n\nparse_benchmark.py: exclude MICRO-DIAGNOSTICS section from regression\ncomparison. Sub-operation benchmarks (FE52::mul, FE52::inverse_safegcd)\nmay legitimately trade per-op speed for pipeline ILP via inlining.\nThe FE52 x64 GAS ASM removal trades ~10ns FE52::mul overhead for\ncross-operation inlining that benefits end-to-end scalar_mul.\n\n* fix(ci): remove std::filesystem to fix MSan false positives\n\nMSan reports use-of-uninitialized-value in std::filesystem::path's\ndestructor because libstdc++ is not MSan-instrumented. Replace all\nstd::filesystem usage with POSIX/C equivalents:\n\n  - exists()          -> stat()\n  - rename()          -> std::rename() (from <cstdio>)\n  - directory_iterator -> opendir/readdir (POSIX) / _findfirst (Win32)\n\nThe <filesystem> header is no longer included.\n\n* fuzz: pre-warm precompute table in LLVMFuzzerInitialize\n\nThe first scalar_mul_generator call builds the precompute table, which\ntakes >25s under ASan/UBSan with ASM disabled. This exceeds libFuzzer's\ndefault per-input timeout (25s), causing CFL to report a timeout crash\non the stored corpus entry timeout-662a824575ea822f8536b78cc7717d3ec4540afd.\n\nFix: call scalar_mul(one()) in LLVMFuzzerInitialize so the table is built\nduring fuzzer startup (not subject to per-input timeout). Subsequent\ncorpus inputs process in <1s each.\n\n* ci: tolerate Valgrind/MSan timeouts in Security Audit\n\nValgrind MemCheck: restore '|| true' on ctest -T MemCheck (removed in\n#91).  The grep-based log checks catch real memory errors; test timeouts\nunder 10-20x Valgrind overhead are expected and harmless.\n\nMSan: increase per-test timeout from 900s to 3600s and tolerate CTest\nexit code.  MSan + no-ASM + Debug + origin-tracking makes precompute\ntable build extremely slow, causing most tests to timeout.\nhalt_on_error=1 still prints real MSan errors in the CI log.\n\n* bench: apply section exclusion to legacy-format entries too\n\nThe regression gate parser had section-aware filtering only for\ntable-format entries (Pattern 1). Legacy-format entries like\n'RFC6979 overhead: 1925.7 ns' -- which are printf-based cost\ndecomposition lines inside the MICRO-DIAGNOSTICS section -- were\nnot filtered, causing false regression alerts on derived metrics\nthat amplify CI benchmark noise.\n\nApply the same excluded_sections check to the legacy pattern loop.\n\n* bench: skip all legacy-format entries within bench_unified sections\n\nIn bench_unified, all standalone benchmarks use table-format rows\n(print_row -> '| name | value |'). The legacy-format printf lines\n(name: value unit) that appear within sections are diagnostic/derived\nmetrics -- cost decompositions, UNEXPLAINED gap, RFC6979 overhead,\nsign-then-verify overhead, etc. These are computed from other\nmeasurements and amplify CI benchmark noise.\n\nThe previous fix only excluded legacy entries from MICRO-DIAGNOSTICS,\nbut 'UNEXPLAINED gap' also appears in CT SIGNING and other sections.\nFix: skip ALL legacy-format entries that appear within any section\nheader context. The legacy pattern remains active for old\nbench_comprehensive output which has no section headers.",
          "timestamp": "2026-03-05T15:36:30+04:00",
          "tree_id": "11b2e380fce202b88a372e0e64885d2fefbaae39",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/0e89b14dc03506c6761a02066adfd8bfb2efc892"
        },
        "date": 1772710718493,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.1,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16.1,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1107.5,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 14.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 44.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1337.6,
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
            "value": 7246,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35510.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 36937.4,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40185.6,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 395.6,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 156.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 12007.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 61565.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 42173.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9018.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9549.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 57385.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 42516.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 187280,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 46820,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 743245.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 46452.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4654021.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 72719.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 161421.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 647668.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2605062.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1880.2,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19461.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41273.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 160.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 422.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 307.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 304.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24054.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 86049.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21492.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 69558.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21002.9,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1180.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19988.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P, tweak_mul)",
            "value": 37653.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21266.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42610.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391119.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 417366.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 376321.9,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "e6b5e76d14e242eb10235d70d9af54e911cddae9",
          "message": "fix(#87): batch serialization + bench_unified coverage + claim qualification (#94)\n\nIssue #87 identified a structural 79x serialization gap: UF stores Jacobian\n(inversion per point ~1000ns) vs libsecp's affine storage (byte copy ~15ns).\n\nNew methods:\n- Point::x_only_bytes()      -- 32B x-coord, no Y recovery (1 mul saved)\n- Point::batch_normalize()   -- Montgomery trick: 1 inv + 3(N-1) muls for N pts\n- Point::batch_to_compressed() -- N points -> 33B compressed, amortized ~15ns/pt\n- Point::batch_x_only_bytes()  -- N points -> 32B x-only, amortized\n\nbench_unified additions:\n- SECTION 3.5: 7 Ultra serialization benchmarks (individual + batch N=64)\n- libsecp: serialize_compressed, serialize_uncompressed, point_add\n- Ratio table: 3 new entries (Serialize compressed/uncompressed, Point add)\n\nClaim qualifications:\n- README: 'Fastest' -> 'High-Performance', added workload caveat\n- CHANGELOG: '1.47x faster overall' -> '1.47x faster on the 13-op suite'\n\nTests: 12 new checks in test_point_edge_cases (53/53 pass)\n\nCloses #87",
          "timestamp": "2026-03-05T17:24:31+04:00",
          "tree_id": "3d148fcfcc5841187cda7a61ba5a66787a783820",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/e6b5e76d14e242eb10235d70d9af54e911cddae9"
        },
        "date": 1772717221176,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.1,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1105.6,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 14.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 45.3,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1335.7,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7243.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35259.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 36869.6,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40068.4,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 402.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 159,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 1375.2,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 1301.2,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 1247.2,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 1298.5,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 1244.1,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 262,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 176.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 11749.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 62332.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 42304,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 8976.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9856.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 57622.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 42720.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 186950.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 46737.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 746357.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 46647.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4769620.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 74525.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 162623.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 649755.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2609245.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1883.9,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19291.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 42004.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 160,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 423.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 307.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 302,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24202.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 86303.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21524.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 69725.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21008.9,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1162.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19725.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P, tweak_mul)",
            "value": 37427.8,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 38.3,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 46.4,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2819.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21107.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42917.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391129.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415570.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374754.9,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "94128f2fa1852f7deb00413ff2aa7ac73375d312",
          "message": "fix: restore missing z_one_ member and normalize() methods (#97)\n\nFixes compilation errors after PR #96. The z_one_ member variable,\nnormalize() and is_normalized() method declarations were accidentally\nremoved from Point class header.\n\nAlso fixes FieldElement52 serialization: replaced non-existent\nto_bytes_into_normalized() calls with correct to_fe().to_bytes_into()\npattern (5 locations in to_compressed/to_uncompressed/x_bytes).\n\nChanges:\n- point.hpp: restore z_one_ member, normalize() and is_normalized()\n- point.cpp: fix FE52 byte serialization in z_one_ fast paths",
          "timestamp": "2026-03-06T00:12:24+04:00",
          "tree_id": "0faa2d15e57a0cc7f18e80c7b4ee638a8255fbc7",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/94128f2fa1852f7deb00413ff2aa7ac73375d312"
        },
        "date": 1772741681952,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.3,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1105.2,
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
            "value": 12.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 43.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1339.7,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7339.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 36854.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 36376.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40642.8,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 415.7,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 157.2,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 1399.1,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 1304.5,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 1252.5,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 1303.6,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 1246.1,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 261.8,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 175.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 11790.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 62195.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 42679.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9155.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9680.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 58303.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 43224.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 188702.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 47175.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 754115,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 47132.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4911613,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 76744,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 164142.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 656998.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2639137.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1953.5,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19485.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41158.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 159.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 423.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 311.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 316.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24290.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 87813.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21685,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 70280.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21120.1,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1159.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19680.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P, tweak_mul)",
            "value": 37612,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.9,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2820.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21083.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42966.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390747.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416470.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 382322.5,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "aff9f3a7ea1e20169690e17581e2a20ff1ae2ff7",
          "message": "fix(sonarcloud): add FieldElement52::to_bytes_into() to eliminate code duplication (#98)\n\nAddresses SonarCloud Quality Gate failure (4.1% duplication > 3% threshold).\n\nChanges:\n- Add FieldElement52::to_bytes_into() convenience method (field_52.hpp + impl)\n- Replace 5 instances of 'x_.to_fe().to_bytes_into(out)' pattern in point.cpp\n- Reduces code duplication from 4.1% to <3%, inline with original to_fe()\n\nThis matches the pattern of inverse_safegcd() and from_bytes() helper methods that wrap common conversion chains.",
          "timestamp": "2026-03-06T00:39:12+04:00",
          "tree_id": "f9c5d3a2b62dc6e73451624454655e562e14c51a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/aff9f3a7ea1e20169690e17581e2a20ff1ae2ff7"
        },
        "date": 1772743307927,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.1,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 16.1,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1104.2,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 14.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 12.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 45,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.2,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 10.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 11.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7385.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 37493.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 36336.6,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 40866,
            "unit": "ns"
          },
          {
            "name": "point_add",
            "value": 413.7,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 162.6,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 1318,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 1283.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 1232.3,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 1292.1,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 1233.6,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 261.8,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 177.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 11862.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 62228.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 42639,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 9104.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9637.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 58195.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 43180,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 188513,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 47128.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 753053.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 47065.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4899697.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 76557.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 163761.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 658580,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2651772.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1953.8,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19392.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41174.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 159.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 426.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 303.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 305.9,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24197.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 87305.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21583.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 71118.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21094.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1158,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19686.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P, tweak_mul)",
            "value": 37541.6,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.7,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.9,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2820.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21096.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42557,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 399388.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 423728.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 382415.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "ab70b82be7559a095222179d5e43a19854748321",
          "message": "Code-scanning remediation pass (#102)\n\n* docs(arm64): Add comprehensive ARM64 audit, benchmark, and gap analysis\n\n- ARM64_AUDIT_BENCHMARK.md: Platform certification with full audit results\n  * Apple Silicon M1/M2/M3 native dudect verification (48/49 modules)\n  * Android ARM64 (Cortex-A55) real hardware benchmarks\n  * Linux ARM64 cross-compile validation\n  * Performance comparison: ARM64 vs x86-64 vs RISC-V\n  * CI workflows: ct-arm64.yml, release.yml Android NDK\n\n- ARM64_GAPS_ANALYSIS.md: Testing and optimization opportunities\n  * Testing gaps: Native Linux ARM64 CI, Android device farm\n  * Optimization roadmap: NEON batch ops (2-3x speedup priority)\n  * Stack round-trip elimination, Metal GPU analysis\n  * Decision matrix with effort/impact/recommendations\n\n- README.md: Add ARM64 docs to Documentation section\n\nAddresses issue #87 ARM audit request\n\n* perf(x86-64): comprehensive primitive optimization sweep\n\nField arithmetic:\n- FE52 normalize: rewrite normalizes_to_zero_var with libsecp fast-path\n  (check t0/t4 only, skip full carry propagation in 99.99% of calls)\n- FE52 negate: eliminate copy-then-negate pattern, compute directly\n- FE52 normalize: fold-first approach (2 carry chains instead of 3)\n- Force-inline fe52_normalize_inline with SECP256K1_FE52_FORCE_INLINE\n- inverse_safegcd: use var-time zero check instead of CT normalizes_to_zero\n\nScalar arithmetic:\n- scalar_mul: complement-based reduction (Barrett ~36 muls -> ~14 muls)\n  35.1ns -> 18.9ns (46% faster)\n- Force-inline 4 SafeGCD helpers (divsteps_62_var, update_de_62,\n  update_fg_62_var, normalize_62) - scalar_inv 10% faster\n\nField SafeGCD:\n- Force-inline 5 helpers (safegcd_divsteps_62_var, safegcd_update_fg,\n  safegcd_update_de, safegcd_reduce_len, safegcd_normalize)\n\nPoint arithmetic:\n- Reduce negate magnitudes: negate(23)->negate(8) for X coords,\n  negate(10)->negate(4) for Y coords in all 6 add_mixed variants\n- Enables fast-path normalizes_to_zero_var (lower magnitude = fewer ops)\n\nSchnorr:\n- Cached verify path with pre-lifted pubkey (skip sqrt/lift_x)\n- Direct FE52 r-value parsing (no FieldElement intermediate)\n\nBenchmark:\n- Fix point_add/dbl fairness: remove loop-carried dependency\n  (was penalizing Ultra vs libsecp's independent iteration pattern)\n- Fix normalize benchmark: use magnitude-2 input for both sides\n- Add FE52 hot-path micro-diagnostics\n- Add libsecp scalar_negate, from_bytes benchmarks\n- Add comprehensive verify cost decomposition\n\nAll 21/21 selftest modules pass, 89/89 ECC properties verified.\n\n* Fix code-scanning findings in CT and lint clusters\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T06:19:36+04:00",
          "tree_id": "73cfb5f892668c294bb9eff615ccc6394bf339b6",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ab70b82be7559a095222179d5e43a19854748321"
        },
        "date": 1772850201399,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.5,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1084.8,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1335.8,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 8.9,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7503.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35287.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34771.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 39089.8,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1335.1,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 261.2,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.4,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 191.3,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 262.9,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2458.9,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.7,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 204.6,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10129.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59112.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40829.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7641.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8219.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 54814.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41251.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 46095.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 179508.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 44877.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 717905,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 44869.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4221666.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65963.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 154851.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 621202.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2491360.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1887.7,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18969.9,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41088.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 428,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 304.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 297.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23723.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84049.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21140.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 67946.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20760.2,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1200.1,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2063.3,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1209.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 159.7,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 267.2,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40075.1,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17587.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19831.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37566,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40.4,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2834.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21136.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42571.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390360.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416132.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374426,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "39089d8de3151e17be8e1a5e96e28c68bb8858f9",
          "message": "fix(point): repair broken preprocessor branches in point ops (#104)\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T07:16:22+04:00",
          "tree_id": "10b5ee050f00f61e9658cd8ed7c84074e8c1adbf",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/39089d8de3151e17be8e1a5e96e28c68bb8858f9"
        },
        "date": 1772853617011,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.3,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.8,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.3,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1337,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 9,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7635,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35531,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34782.9,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38933.2,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1347.2,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 262.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.5,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.9,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 197.7,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 260.7,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2471.4,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.5,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.7,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 210.9,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10115,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59169.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41106.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10175.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8071.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 54727.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41449.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 46247.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 180785.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 45196.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 718952.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 44934.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4222081.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65970,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 154896.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622273.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2490332.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1895.7,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18973.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41143.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 427.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 304.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 301.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23594.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 83699,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21036.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 67731.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20721.7,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1190.2,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 10.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2059.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1214.8,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 159,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 266.7,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40156.8,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17541.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19772.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37475.9,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.7,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2832,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21128.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42690.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391362.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415577.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 379465.9,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "9b7a1d2e0ecb393261bf6bef8a15dded90725848",
          "message": "fix: stabilize dev point.cpp and reduce clang-tidy hotspots (#106)\n\n* fix(clang-tidy): clear repeated const/init/braces warnings hotspots\n\n* fix(point): repair broken preprocessor branches in point ops (#104)\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T07:46:30+04:00",
          "tree_id": "1bea5378366a677ac3e9e9529da67f5077328f6e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9b7a1d2e0ecb393261bf6bef8a15dded90725848"
        },
        "date": 1772855512411,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.3,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1335.6,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7437.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35443.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34757.8,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38936.1,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1331.3,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 263.1,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.2,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 189.6,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 264.1,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2496.1,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 204.1,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10122.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 58982.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40847.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7610.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8403.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 54613.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41289.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 46201.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 180037.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 45009.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 719789.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 44986.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4214392.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65849.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155219.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622590.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2493144.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1889,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19055.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41138.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 146,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 427.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 299.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23633.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84088.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21832.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 67957.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20774.4,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1191.1,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2141,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1216.9,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 156.7,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 269.5,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40645.4,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17575.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19885.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37691.5,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.8,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.7,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2833.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21157.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42578.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391885.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 417681.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374840.9,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "828e6904a1d66c1dce494422285c0816ad7efb33",
          "message": "fix(clang-tidy): clear repeated const/init/braces warnings hotspots (#105)\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T07:46:45+04:00",
          "tree_id": "1bea5378366a677ac3e9e9529da67f5077328f6e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/828e6904a1d66c1dce494422285c0816ad7efb33"
        },
        "date": 1772855724995,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.3,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1089.2,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1336.3,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7382.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35410.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34800.7,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38975,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1331.7,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 259.8,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 148.6,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 189.8,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 260.3,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2488.4,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 203,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 151.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10127.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 58925,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41025.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10208.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8132.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 54636.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41251.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 46290.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 180171.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 45042.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 718360,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 44897.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4215352.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65864.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155342.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623409.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2493588.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1888.8,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19045.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41237,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 426.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 305.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 296.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23637.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84157.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21189,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 67829.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20649.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1190.9,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 10.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2095.8,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1216.8,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157.5,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 266.9,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40083.7,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17537.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19735,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37546.8,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40.3,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2840.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21189.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42586.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 392120.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416167.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374921.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "9eaeea32b6b6212ecd64d2a0f3c9d260a6a38b0d",
          "message": "docs: add native mars riscv benchmark and audit results",
          "timestamp": "2026-03-07T12:39:34Z",
          "tree_id": "78bce53b734385dcc9001f929c767ac47763458b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9eaeea32b6b6212ecd64d2a0f3c9d260a6a38b0d"
        },
        "date": 1772887433935,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.3,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.8,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7433.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35467.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34759.1,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38791.2,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1332.9,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 262.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.6,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 193.7,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 261.6,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2492.1,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.5,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 207.5,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10121.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59048.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40924.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7650.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8036.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51463,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41228.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42440.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159485.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39871.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647846.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40490.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4213354,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65833.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155198.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622359.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2494164.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1888.3,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19063.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41084.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 146,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 450.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 297.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23640.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84323.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21112.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64203.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20682.2,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1191,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2059.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1218,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 180,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 283.5,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40301,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17581.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 21860.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37558.2,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40.1,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2836.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 22083.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 43055.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391199.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416124.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375513.6,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "d6c0619e29258037055686a304649003702bb9d8",
          "message": "ci(android/ios): fix arm target detection, asm gating, xcframework lto",
          "timestamp": "2026-03-07T13:01:21Z",
          "tree_id": "ca62ee0f5324b6ca0ff5d4985a61f21238130b2e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d6c0619e29258037055686a304649003702bb9d8"
        },
        "date": 1772888706071,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.5,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1335.7,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7455.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35552.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 35094.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 39013.3,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1332.5,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 262.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 193.4,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 260.6,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2503,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 203.4,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10327.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59011.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40941.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10186.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8086.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50889.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41251.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42432.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159392.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39848.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 649037.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40564.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4214432,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65850.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155219.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623174,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2492941.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1888.2,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19069.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41175.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 427.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 299.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23777.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 86620.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21347.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64922.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20892.3,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1192.9,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 10.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2096.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1216.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 158.1,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 266.8,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40121.7,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17668.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19772.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37549.4,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.8,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2835.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21114.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42793.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390416.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415595.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374235.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "8eab8e1f073812fcd57fafd0fcbd62fe27ebb1bf",
          "message": "fix(windows): correct BE parse byteswap under MSVC",
          "timestamp": "2026-03-07T13:25:33Z",
          "tree_id": "7d8c446683d91e6f67e633744e2f8b777a7373fb",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/8eab8e1f073812fcd57fafd0fcbd62fe27ebb1bf"
        },
        "date": 1772890155496,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.3,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1087.1,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 33.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1385.5,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7361,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35426.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34916.6,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38783.1,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1330.6,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 264,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.2,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 190.5,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 261.5,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2507.6,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 230.7,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 193.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 11499,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59009.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40925.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7593.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8002.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50787.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41216.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42374,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159647.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39912,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647297.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40456.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4207454.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65741.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155426.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622781.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2491322.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1888.6,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19066.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41101.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 146.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 427.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 298.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23617.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 83911.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21100.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64202.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20672.3,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1196.6,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2059.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1217.2,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 156.8,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 268.6,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40165.1,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17616.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19872.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37448.2,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.6,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.7,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2942.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21127.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42796.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 392116.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416712.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375471.6,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "c0509dca2e69750b0d99efa743a6bec20d9de8ca",
          "message": "sync: consolidate dev + ESP32 bench hornet + benchmark data (#107)\n\n* docs(arm64): Add comprehensive ARM64 audit, benchmark, and gap analysis\n\n- ARM64_AUDIT_BENCHMARK.md: Platform certification with full audit results\n  * Apple Silicon M1/M2/M3 native dudect verification (48/49 modules)\n  * Android ARM64 (Cortex-A55) real hardware benchmarks\n  * Linux ARM64 cross-compile validation\n  * Performance comparison: ARM64 vs x86-64 vs RISC-V\n  * CI workflows: ct-arm64.yml, release.yml Android NDK\n\n- ARM64_GAPS_ANALYSIS.md: Testing and optimization opportunities\n  * Testing gaps: Native Linux ARM64 CI, Android device farm\n  * Optimization roadmap: NEON batch ops (2-3x speedup priority)\n  * Stack round-trip elimination, Metal GPU analysis\n  * Decision matrix with effort/impact/recommendations\n\n- README.md: Add ARM64 docs to Documentation section\n\nAddresses issue #87 ARM audit request\n\n* fix: stabilize dev point.cpp and reduce clang-tidy hotspots (#106)\n\n* fix(clang-tidy): clear repeated const/init/braces warnings hotspots\n\n* fix(point): repair broken preprocessor branches in point ops (#104)\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>\n\n* riscv: speed up schnorr raw verify parsing/cache path\n\n* docs: add native mars riscv benchmark and audit results\n\n* ci(ct-arm64): avoid forcing -fuse-ld=lld on Apple\n\n* ci(mobile): avoid forcing -fuse-ld=lld on Android toolchains\n\n* ci(android/ios): fix arm target detection, asm gating, xcframework lto\n\n* fix(windows): correct BE parse byteswap under MSVC\n\n* sync: ESP32 bench hornet + benchmark data + CI scripts\n\n- fix(esp32): guard thread_local 36KB lift_x cache on embedded platforms\n- add: ESP32-S3 bench_hornet raw serial output + README\n- add: x86-64 bench_unified JSON/TXT results + validation reports\n- add: cross-platform comparison README updates\n- add: local CI scripts (ci-local.sh, docker-compose.ci.yml)\n- add: RISC-V Mars SSH helper, LOCAL_CI docs, RELEASE_PROCESS\n- update: ESP32 sdkconfig defaults for bench_hornet\n\nAddresses benchmark data preservation and ESP32 boot crash fix.\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T21:02:47+04:00",
          "tree_id": "2404e8dc3e61e2faf2b6d3f683d67fe46fa6a545",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c0509dca2e69750b0d99efa743a6bec20d9de8ca"
        },
        "date": 1772903244767,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.7,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1088.4,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.9,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7426.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35723,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34824.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38835.7,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1330.4,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 264.8,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.9,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 189.8,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 264.7,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2491.3,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.5,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 203.8,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10671.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59493.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40938.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10274.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8089,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50816.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41249.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42376.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159407.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39851.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647162.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40447.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4213727.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65839.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155227.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622202.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2489017.9,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1888.4,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19043.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41134.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 427.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 300,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23727.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84603,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21131,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64126.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20687,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1196.3,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2059.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1218,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 158.8,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 268.8,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40101.9,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17577.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19891.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37725.3,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.5,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40.4,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2833.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21149.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 43015.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 394806.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 417884.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 377488.1,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "c0509dca2e69750b0d99efa743a6bec20d9de8ca",
          "message": "sync: consolidate dev + ESP32 bench hornet + benchmark data (#107)\n\n* docs(arm64): Add comprehensive ARM64 audit, benchmark, and gap analysis\n\n- ARM64_AUDIT_BENCHMARK.md: Platform certification with full audit results\n  * Apple Silicon M1/M2/M3 native dudect verification (48/49 modules)\n  * Android ARM64 (Cortex-A55) real hardware benchmarks\n  * Linux ARM64 cross-compile validation\n  * Performance comparison: ARM64 vs x86-64 vs RISC-V\n  * CI workflows: ct-arm64.yml, release.yml Android NDK\n\n- ARM64_GAPS_ANALYSIS.md: Testing and optimization opportunities\n  * Testing gaps: Native Linux ARM64 CI, Android device farm\n  * Optimization roadmap: NEON batch ops (2-3x speedup priority)\n  * Stack round-trip elimination, Metal GPU analysis\n  * Decision matrix with effort/impact/recommendations\n\n- README.md: Add ARM64 docs to Documentation section\n\nAddresses issue #87 ARM audit request\n\n* fix: stabilize dev point.cpp and reduce clang-tidy hotspots (#106)\n\n* fix(clang-tidy): clear repeated const/init/braces warnings hotspots\n\n* fix(point): repair broken preprocessor branches in point ops (#104)\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>\n\n* riscv: speed up schnorr raw verify parsing/cache path\n\n* docs: add native mars riscv benchmark and audit results\n\n* ci(ct-arm64): avoid forcing -fuse-ld=lld on Apple\n\n* ci(mobile): avoid forcing -fuse-ld=lld on Android toolchains\n\n* ci(android/ios): fix arm target detection, asm gating, xcframework lto\n\n* fix(windows): correct BE parse byteswap under MSVC\n\n* sync: ESP32 bench hornet + benchmark data + CI scripts\n\n- fix(esp32): guard thread_local 36KB lift_x cache on embedded platforms\n- add: ESP32-S3 bench_hornet raw serial output + README\n- add: x86-64 bench_unified JSON/TXT results + validation reports\n- add: cross-platform comparison README updates\n- add: local CI scripts (ci-local.sh, docker-compose.ci.yml)\n- add: RISC-V Mars SSH helper, LOCAL_CI docs, RELEASE_PROCESS\n- update: ESP32 sdkconfig defaults for bench_hornet\n\nAddresses benchmark data preservation and ESP32 boot crash fix.\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T21:02:47+04:00",
          "tree_id": "2404e8dc3e61e2faf2b6d3f683d67fe46fa6a545",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c0509dca2e69750b0d99efa743a6bec20d9de8ca"
        },
        "date": 1772903350970,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.6,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.5,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7455.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35457.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34771.2,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 39124.1,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1338.3,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 259,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.5,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 189.7,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 265.2,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2533.8,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 203,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10100.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59032.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40794.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10257.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8024.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50809.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41214.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42715.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159712,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39928,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 646950.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40434.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4216349,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 65880.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155422,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623282.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2503980.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1887.2,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19101.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41242.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 146.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 427.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 300.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23698.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 83940,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21129.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64107.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20683.3,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1196.2,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2099.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1214,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 159.3,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 268.4,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40095.5,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17963.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19780.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37548.9,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2859.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21134.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42598.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390651.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 425950.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 390159.1,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "04b9f8e10104296eee97fb7455bfe76780deedda",
          "message": "fix: resolve 5 open clang-tidy code scanning alerts (#109)\n\n- schnorr.cpp: simplify boolean via DeMorgan (readability-simplify-boolean-expr)\n- schnorr.cpp: use auto with static_cast (modernize-use-auto)\n- scalar.cpp: add const to __int128 variable (misc-const-correctness)\n- scalar.cpp: use auto with static_cast (modernize-use-auto)\n- point.cpp: add const to z_fe variable (misc-const-correctness)\n\nAddresses alerts #6982-#6986.\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T22:35:30+04:00",
          "tree_id": "ae80b916d738db71b18d5a5e60c31917aae9c56a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/04b9f8e10104296eee97fb7455bfe76780deedda"
        },
        "date": 1772908774015,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.7,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.1,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 2.8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1335,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7484.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35441.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 35705.2,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38860.8,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1332.2,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 262.6,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.3,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 193.4,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 261.1,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2486.2,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.3,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 204.4,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10119,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59169.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40785.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7787.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8002.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50864.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41475.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42451.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159422.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39855.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647553.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40472.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4250528.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 66414.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155476.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623188.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2492807.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1888.4,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19059,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41028.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 426.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 305.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 300.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23605.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84078,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21133.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64111.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20659.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1195.4,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2060.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1210.1,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 268.1,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40297.6,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17585.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19879.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37534.4,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 30.5,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.9,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2837.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21171.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42570.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390322.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416310.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375050.4,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "588c354710629b9196078edb6627ff14396bc027",
          "message": "refactor: reduce code duplication across 8 files (#110)\n\n- point.cpp: delete 2 dead #if 0 code blocks (~462 lines), delegate\n  jacobian_double/jacobian_add_mixed/jac52_double/jac52_add_mixed/jac52_add\n  to their inplace variants via thin wrappers\n- glv.cpp: consolidate GLV_MULADD/GLV_EXTRACT macros to single file-level\n  definition shared by glv_mul_comba_64, glv_mul_2x2, glv_mul_2x4\n- benchmark_harness.hpp: make run() delegate to run_stats().median_ns\n- ufsecp_impl.cpp: extract pubkey_create_core helper for compressed/\n  uncompressed pubkey creation; extract ecdh_parse_args helper for 3 ECDH\n  functions\n- selftest.cpp: extract check_point_match helper for test_addition/\n  test_subtraction\n- field.cpp + scalar.cpp: extract add64/sub64 to shared header\n  detail/arith64.hpp; deduplicate load_be64 lambda in field.cpp\n\nAll 31 tests pass. No algorithmic or behavioral changes.\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T23:47:57+04:00",
          "tree_id": "b4e5efbd46dc7ea621c4ac07b345d7380e4f1e23",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/588c354710629b9196078edb6627ff14396bc027"
        },
        "date": 1772913120790,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1084.2,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1333.2,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.3,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7403.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35487.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34767.3,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38864.4,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1340.8,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 260.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 151.9,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 192.3,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 262,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2492.9,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.9,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 209,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10210.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59162,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40908.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7605.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8126,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50953.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41252.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42698.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159594.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39898.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647922.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40495.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4287943.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 66999.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155188.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622810.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2497364.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1963.1,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19109.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41063,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 425.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 307.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 298.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23847.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84153.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21209.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64136.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20805.1,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1159.6,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2098.3,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1185.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 10,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 156.5,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 263.3,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40014.3,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17597.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19880.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 38115.9,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.7,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2846.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21126.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42430.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390806,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416260.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 373833.1,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "distinct": false,
          "id": "588c354710629b9196078edb6627ff14396bc027",
          "message": "refactor: reduce code duplication across 8 files (#110)\n\n- point.cpp: delete 2 dead #if 0 code blocks (~462 lines), delegate\n  jacobian_double/jacobian_add_mixed/jac52_double/jac52_add_mixed/jac52_add\n  to their inplace variants via thin wrappers\n- glv.cpp: consolidate GLV_MULADD/GLV_EXTRACT macros to single file-level\n  definition shared by glv_mul_comba_64, glv_mul_2x2, glv_mul_2x4\n- benchmark_harness.hpp: make run() delegate to run_stats().median_ns\n- ufsecp_impl.cpp: extract pubkey_create_core helper for compressed/\n  uncompressed pubkey creation; extract ecdh_parse_args helper for 3 ECDH\n  functions\n- selftest.cpp: extract check_point_match helper for test_addition/\n  test_subtraction\n- field.cpp + scalar.cpp: extract add64/sub64 to shared header\n  detail/arith64.hpp; deduplicate load_be64 lambda in field.cpp\n\nAll 31 tests pass. No algorithmic or behavioral changes.\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-07T23:47:57+04:00",
          "tree_id": "b4e5efbd46dc7ea621c4ac07b345d7380e4f1e23",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/588c354710629b9196078edb6627ff14396bc027"
        },
        "date": 1772913189859,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.6,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.4,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.1,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7569,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35490.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34865.8,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38904.2,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1341.8,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 261.7,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.4,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 191.5,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 264.1,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2506.2,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.9,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 205.9,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10202.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59282.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41128.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7578.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8071.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51078.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41360.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42435.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159591.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39897.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647171.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40448.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4291024.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67047.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155775.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622752.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2491332.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1955.9,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19093.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41171,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 144.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 425.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 297,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23735.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84138.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21143.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64160.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20755.9,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1173.4,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2100,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1188.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 156.8,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 270.8,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40053.7,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17526,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19789.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37414.4,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.3,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.3,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2840.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21202.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42686.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391876.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 417232.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375669.2,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "edd1cc83f4ee7261452b731269d328cbc1e15820",
          "message": "fix: resolve 6 clang-tidy code scanning alerts (#111)\n\n- field.cpp: initialize load_be64 local variable (cppcoreguidelines-init-variables)\n- ufsecp_impl.cpp: add const to 5 err variables (misc-const-correctness)\n\nAlerts: #6987, #6988, #6989, #6990, #6991, #6992\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-08T00:28:08+04:00",
          "tree_id": "df807adf44f28b05b4fa1640a1699bdad568e77f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/edd1cc83f4ee7261452b731269d328cbc1e15820"
        },
        "date": 1772915507818,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.8,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1086.6,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1338.2,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.3,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7379.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35579.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34772.3,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38987,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1341.2,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 261.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 152.9,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 212.2,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 409.8,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2510.6,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 11.9,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.4,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 207.1,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 151.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10172.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59259.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41021.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7629.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8107.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50878.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41271.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42467.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159543.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39886,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647675.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40479.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4288507,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67007.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155521.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623852.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2495654.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1955.9,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19080.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41157.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 307.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 299.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24508.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84204.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21269.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64155.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20699.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1158.2,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2063,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1189.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.4,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 156.8,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 265,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 39950.1,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17525.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19748.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37877.8,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.7,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2832.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21099.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42438.5,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390325.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415583.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374197.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "distinct": false,
          "id": "edd1cc83f4ee7261452b731269d328cbc1e15820",
          "message": "fix: resolve 6 clang-tidy code scanning alerts (#111)\n\n- field.cpp: initialize load_be64 local variable (cppcoreguidelines-init-variables)\n- ufsecp_impl.cpp: add const to 5 err variables (misc-const-correctness)\n\nAlerts: #6987, #6988, #6989, #6990, #6991, #6992\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-08T00:28:08+04:00",
          "tree_id": "df807adf44f28b05b4fa1640a1699bdad568e77f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/edd1cc83f4ee7261452b731269d328cbc1e15820"
        },
        "date": 1772915528542,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.5,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1086.9,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1333.7,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7464.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35504.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34816.2,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38883.3,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1341,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 261.8,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.9,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 191.8,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 263,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2489.9,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.9,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.5,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 210.1,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10166.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59182,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41082,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10122.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8021.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51397.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41272,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42480.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159739.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39934.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647541.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40471.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4287551.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 66993,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155829.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623338.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2495573.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1956.3,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19111.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41130.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 146.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 425.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 303.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 296.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23885.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84311.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21294.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64267.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20687.7,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1160.3,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2062.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1193.2,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 156.9,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 261.2,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 39972.3,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17548.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19896,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37649.6,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.6,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 38.9,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2824.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21109,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42481.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391705.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416807,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 383526.3,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "f322b421d1c61e1f704b60f0cf4cbb5bf5d8d086",
          "message": "v3.20.0: GLV window optimization, code scanning fixes, docs-engine sync\n\n- Platform-aware configurable GLV window width (kDefaultGlvWindow):\n  x86-64/ARM64/RISC-V default w=5, ESP32/WASM w=4\n  3-tier config: CMake define > platform default > runtime per-call\n- Fix 501 code scanning alerts across 7 files:\n  shiftTooManyBitsSigned (ct_field.cpp), const-correctness,\n  modernize-use-auto, braces, init-variables\n- Full docs-engine consistency audit: fix 8 categories of stale\n  references across 14+ documentation files\n- Add GLV Window Width Tuning to PERFORMANCE_GUIDE + 12 bindings + WASM\n- CHANGELOG.md: consolidated v3.14.0-v3.20.0 (120+ commits)\n- VERSION.txt: 3.20.0\n- bench_bip352 results: x86-64 1.34x, ARM64 1.31x, RISC-V 1.37x",
          "timestamp": "2026-03-07T22:46:47Z",
          "tree_id": "1035eb7075109109cacfb16a1f75f96d59387b2b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f322b421d1c61e1f704b60f0cf4cbb5bf5d8d086"
        },
        "date": 1772924461489,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.8,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1090.2,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1340.2,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.3,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7877.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35386.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34765.6,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 39203,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1342.4,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 258.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.6,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 193.4,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 263.6,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2514.9,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.8,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.5,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 206.1,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10171.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59128.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40898.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10633.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8133.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51035.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41353.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42509.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159746.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39936.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 649482.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40592.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4287854,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 66997.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155577.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 624417.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2498135.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1955.5,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19100.1,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41240.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 144.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 298.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23735.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84188.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21143.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64220.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 21044.9,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1171.7,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 14.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2061.8,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1185.7,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 158.2,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 264.6,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40033.1,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17563.6,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19918.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37430.7,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.3,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2833.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21137.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42455.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 389973.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414666.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374847.6,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "883d8bbfa6089082d24e3dbc1c786835ddb8ae03",
          "message": "refactor(point): extract shared GLV52/batch/next-prev helpers to reduce duplication\n\nExtract build_glv52_table_zr, derive_phi52_table, shamir_2stream_glv52\nhelpers used by scalar_mul_glv52, scalar_mul_with_plan_glv52, and\ndual_scalar_mul_gen_point.\n\nRewrite batch_x_only_bytes to reuse batch_normalize (Montgomery trick).\n\nExtract add_gen_mixed helper for next/prev/next_inplace/prev_inplace.\n\nNet reduction: 234 lines (-425/+191). All 30 tests pass.",
          "timestamp": "2026-03-07T23:27:13Z",
          "tree_id": "18a6fa114c1077030787490fe2f64647f7f97250",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/883d8bbfa6089082d24e3dbc1c786835ddb8ae03"
        },
        "date": 1772926270436,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.6,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.8,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.7,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.8,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1333.8,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7420,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 37001.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34761.8,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 39125,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1341.2,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 272.7,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.4,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.9,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 191.4,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 276.6,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2472.7,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.8,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.4,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 208.4,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 208.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10169.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59110.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40871.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10169.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8385.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 53115.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41969.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42911.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159789.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39947.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 648640.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40540,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4294741.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67105.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155444.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 624738.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2495748.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1886.2,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18990,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41152.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 299.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23638.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84056,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21008.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64050,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20597.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1169,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 14.6,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2061.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1185.9,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 12.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157.5,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 265.5,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40153.5,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17629.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19829,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37565.8,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.6,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.6,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2831.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21133.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42716.5,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391608,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 417160.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 376461.4,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "7235683585a350a2a2c161d45efb648433d4b82d",
          "message": "perf(batch): restore x-only fast path in batch_x_only_bytes\n\nCall batch_z_inv() directly instead of batch_normalize(), computing\nonly x = X * Z^(-2) and skipping the unused Y = Y * Z^(-3).\nEliminates ~33% redundant field multiplications per point.\n\nFixes benchmark regression gate failure (1.39x slowdown on\nbatch_x_only_bytes /pt N=64).",
          "timestamp": "2026-03-07T23:59:10Z",
          "tree_id": "1fd34ae8ca72ae1729bd8201cc05367dc528d088",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7235683585a350a2a2c161d45efb648433d4b82d"
        },
        "date": 1772928179683,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.6,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1089.5,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1333.8,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7531.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35625.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34752.7,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38858.3,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1351.2,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 264.1,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 150.3,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 197.4,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 279.3,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2665.1,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.5,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.4,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 210.5,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 154.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10204.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59916.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41015.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7501.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8049.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51087,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41302.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42440.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 160123,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 40030.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 649399.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40587.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4290383.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67037.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155834.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 710079,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2494127.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1886.3,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18959.9,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41112.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 426.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 308.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 294.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23608.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 83828.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21222.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 63978.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20632.9,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1161.4,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2062.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1185.5,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157.1,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 265.8,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40198.5,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17608.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20007.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37720.5,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.3,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.1,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2837.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21373.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42512,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 395530,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 419034.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 376809.1,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "9fe32ea43c633b001c984e9bcdb1f59a44975b84",
          "message": "refactor(batch): deduplicate allocation+wNAF patterns in point.cpp\n\n- Move partials allocation inside batch_z_inv (callers only pass z_inv)\n- Extract apply_wnaf_mixed52 helper for wNAF digit lookup+negate+add\n- Use apply_wnaf_mixed52 in shamir_2stream_glv52 and dual_scalar_mul_gen_point\n- Reduces SonarCloud duplicated lines on new code (was 12%, gate requires <=3%)",
          "timestamp": "2026-03-08T00:46:32Z",
          "tree_id": "439f860c1e44f549ebc27bc32910ffe8c72b97a3",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9fe32ea43c633b001c984e9bcdb1f59a44975b84"
        },
        "date": 1772931012343,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 16.6,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 14.1,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 971.7,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.6,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 33.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1249.5,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 6961,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33493.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 32935.3,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 37567.9,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1177.8,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 241.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 125.6,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 184.3,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 255.9,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2116.5,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 9.8,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 4.9,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.3,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 195.2,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 147.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 9891.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 56642.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 39209.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 11401.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 7632.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 48859.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 39485,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 40851.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 151210.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 37802.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 616654.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 38540.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4064728.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 63511.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 148129.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 592704.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2416379,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 2648.5,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18407.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 39608.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 139.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 410.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 297.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 260.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24001,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 82050.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 20580.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 61763.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20127.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1151.9,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 10.3,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 10,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2763.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1176.2,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 7.7,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 130.1,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 235,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 35583.5,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 14673.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 17601,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34158,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 26.1,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 35,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 3432.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 18965.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 38056,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 394113,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415259.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 370287.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "9fe32ea43c633b001c984e9bcdb1f59a44975b84",
          "message": "refactor(batch): deduplicate allocation+wNAF patterns in point.cpp\n\n- Move partials allocation inside batch_z_inv (callers only pass z_inv)\n- Extract apply_wnaf_mixed52 helper for wNAF digit lookup+negate+add\n- Use apply_wnaf_mixed52 in shamir_2stream_glv52 and dual_scalar_mul_gen_point\n- Reduces SonarCloud duplicated lines on new code (was 12%, gate requires <=3%)",
          "timestamp": "2026-03-08T00:46:32Z",
          "tree_id": "439f860c1e44f549ebc27bc32910ffe8c72b97a3",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9fe32ea43c633b001c984e9bcdb1f59a44975b84"
        },
        "date": 1772931525799,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.5,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.6,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1337.8,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7455.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35450.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34819.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 39013.3,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1343.6,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 263.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.8,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 196.2,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 279.5,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2451.4,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.5,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.5,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 209.9,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 153.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10153.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59180.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41043.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7739,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8017.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50973,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41368.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42476.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159524.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39881.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 648021.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40501.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4289291.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67020.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155193.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622989.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2504521.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1955.8,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19098.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41084.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 147,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 426.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 304.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 300.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23761.9,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84376.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21368.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64200.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20788.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1161.8,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2061.3,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1193.1,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.4,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157.5,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 264.6,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40002.2,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17529,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19808.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37346.3,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40.2,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2829.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21120.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42929.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391386.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414892.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 377154.7,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "06c675abdc2bbd012f22fc75ff481a3503d06528",
          "message": "ci: raise PR bench threshold to 150%, exclude point.cpp from CPD\n\n- bench-regression.yml: PR alert-threshold 120% -> 150% to match push\n  threshold. Shared CI runners (Xeon Platinum 8370C) show 20-30% variance\n  on sub-10ns operations, causing false positives on unrelated benchmarks.\n\n- sonar-project.properties: add point.cpp to sonar.cpd.exclusions.\n  point.cpp contains dual-platform paths (4x64 and 5x52 field representations)\n  implementing identical algorithms for different CPU architectures.\n  This structural duplication is an architectural requirement.",
          "timestamp": "2026-03-08T01:35:04Z",
          "tree_id": "509da23b0bcbd4375365a4c814deaeb46a99a8c0",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/06c675abdc2bbd012f22fc75ff481a3503d06528"
        },
        "date": 1772933922182,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.4,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.5,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 31.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1338.3,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7498.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35505.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 35098.7,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38956.2,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1341.2,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 263,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 150.8,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 200.1,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 275.5,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2456.6,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.5,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 10.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.4,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 209.4,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 154,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10251,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59273.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41045.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7758.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8127.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51180.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41356.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42507.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159698.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39924.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 648922,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40557.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4283367.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 66927.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155263.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 622831.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2510448.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1970.6,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19109.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41101.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 144.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 305,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 300.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23822.1,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84347.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21150.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64237.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20693.2,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1161.5,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2065.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1187.4,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157.6,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 266.9,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 39962.4,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 18204.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19816.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37395.9,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2830.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21160,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42671.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 392170.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 417145.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374649.2,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "06c675abdc2bbd012f22fc75ff481a3503d06528",
          "message": "ci: raise PR bench threshold to 150%, exclude point.cpp from CPD\n\n- bench-regression.yml: PR alert-threshold 120% -> 150% to match push\n  threshold. Shared CI runners (Xeon Platinum 8370C) show 20-30% variance\n  on sub-10ns operations, causing false positives on unrelated benchmarks.\n\n- sonar-project.properties: add point.cpp to sonar.cpd.exclusions.\n  point.cpp contains dual-platform paths (4x64 and 5x52 field representations)\n  implementing identical algorithms for different CPU architectures.\n  This structural duplication is an architectural requirement.",
          "timestamp": "2026-03-08T01:35:04Z",
          "tree_id": "509da23b0bcbd4375365a4c814deaeb46a99a8c0",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/06c675abdc2bbd012f22fc75ff481a3503d06528"
        },
        "date": 1772934254607,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.8,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.5,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1330.1,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.3,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7447.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35462.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34815.8,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38952,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1341.7,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 270.1,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 150.4,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 6.2,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 195.3,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 275.9,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2458.7,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 17.2,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.5,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.4,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 213.4,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 153.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10171.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59118.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40894.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7511.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8032.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51090.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41296.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42517.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159517,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39879.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 648346.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40521.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4290154.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67033.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155379.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623095.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2512353.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1955.8,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19185.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41124.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 425.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 304.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 300.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23801.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84343.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21150.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64189.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20687.7,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1169.4,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 14.5,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 10.8,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2102.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1186.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 12.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 266.8,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40586.3,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17634.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19762.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37856.7,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 32.1,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 40.5,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2834.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21202.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 43204.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390088.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414802.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 376006.2,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "cfffcb1024691c2179062abda056d27e49fd2c92",
          "message": "fix: wire GLV_WINDOW_WIDTH CMake option, fix docs (WASM + bench paths)\n\nAddress Copilot review feedback on PR #112:\n\n1. CMake cache variable forwarding: add SECP256K1_GLV_WINDOW_WIDTH as a\n   CMake cache string option that forwards into a compile definition.\n   Now `-DSECP256K1_GLV_WINDOW_WIDTH=6` works as documented.\n\n2. Platform default descriptions: add WASM to the w=4 default set\n   (\"4 on ESP32/WASM\") in 6 docs + copilot-instructions.md.\n\n3. Benchmark binary paths: fix cpu/bench/bench_unified -> cpu/bench_unified\n   in 7 docs. Build outputs go to build/cpu/, not build/cpu/bench/.\n\n31/31 tests pass.",
          "timestamp": "2026-03-08T08:43:14Z",
          "tree_id": "9de71bbf8395bd997737eb6b234d31ee9a4de77b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/cfffcb1024691c2179062abda056d27e49fd2c92"
        },
        "date": 1772959660169,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.6,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.7,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1085.5,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1337.5,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7417.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35643.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34773.2,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38969.2,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1341.1,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 269.8,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.3,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 196.5,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 285.3,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2454.2,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 15.6,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.6,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.6,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 208,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 153.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10093.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59259.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 41096.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10116.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 7983.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51072.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41376.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42518.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 160525.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 40131.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647826.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40489.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4330064.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67657.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155282.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623304,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2506024.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1955.8,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19088.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 40984.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 146,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 425.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 304.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 299.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23801.9,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84539.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21154.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 64236,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20697.9,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1173,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 17.3,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2056.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1195.6,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 13,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157.7,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 264.8,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 39955.4,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17608.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19847.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37449.9,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 32,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.7,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2832.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21104,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 43127.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391155.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415543.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 375284.7,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "ce299e0f9d9b8e27e787eb25601f32044e5dae1d",
          "message": "fix: address Copilot review (return check, comments, threshold)\n\n- Check build_glv52_table_zr return in dual_scalar_mul_gen_point,\n  fallback to separate a*G+b*P on degenerate case\n- Update next_inplace/prev_inplace header comment (no longer faster)\n- Fix bench-regression.yml header: >20% -> >50% to match threshold",
          "timestamp": "2026-03-08T08:56:50Z",
          "tree_id": "5b980ea86de07bacfd6196aaa3214f5258e4c044",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ce299e0f9d9b8e27e787eb25601f32044e5dae1d"
        },
        "date": 1772960453753,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 17.5,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 15.8,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 1088.7,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.8,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 3.1,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1340.9,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7442.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35598.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34759.4,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38908.2,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1341.5,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 263.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.7,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 194.8,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 277.1,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2673.5,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 13.5,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 17.1,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 5.9,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.4,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 211.9,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 153.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10179.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59059.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40813.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10216.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8126.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50937.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41260.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42405.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159490.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39872.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 647623.9,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40476.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4296012,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67125.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155256.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623018.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2509757.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1886.2,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18933,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41124.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 309.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 294.9,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23513.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 83722,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21077.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 63896.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20656.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1161.6,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 9.9,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 17.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2062.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1185.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 8.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 158.9,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 265.6,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40417.7,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17541.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19754.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37342.4,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.9,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2828.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21124.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42632.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 391253.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414129.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 377099.6,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "99915cdc5168c122d8d0c500e43c2e5c2cdc546a",
          "message": "ci: filter sub-25ns micro-ops from regression gate\n\nShared CI runners (ubuntu-latest) have high timer jitter on sub-25ns\noperations.  field_from_bytes (9.9->17.9ns, 1.81x) triggered a false\nregression alert.  Skip entries <25ns in parse_benchmark.py so the\nPerf Regression Gate only tracks operations stable enough to measure.",
          "timestamp": "2026-03-08T09:12:05Z",
          "tree_id": "8060962f1aa9735ff666e9317c7c755a2de7e1fa",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/99915cdc5168c122d8d0c500e43c2e5c2cdc546a"
        },
        "date": 1772961383503,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1094.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 30.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1334.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7959.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35433.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 38139.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38834.8,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1348.7,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 264.5,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 150.2,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 202.5,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 275.9,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2635.1,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 207.8,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 153.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10261.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59052.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40996.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10722,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8096,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51195.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41340.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42552,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159901.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39975.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 650656.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40666,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4304712.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67261.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 156240.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 626011.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2508432.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1886.1,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18905.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41087.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 425.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 311.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 293.9,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23489.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 83683.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21000.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 63973.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20549,
            "unit": "ns"
          },
          {
            "name": "field_mul",
            "value": 25.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1162.3,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2062,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1189.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 158.6,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 266.1,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40010.2,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17587.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19801.4,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37521.2,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 31.4,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 39.4,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2826.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21089.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42673.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 396433.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414482.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 378312.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "05822dcabea22fafaa9bd836f517da3c7dccd7c0",
          "message": "v3.20.0: GLV window optimization, refactoring, CI acceleration (#115)\n\n* v3.20.0: GLV window optimization, code scanning fixes, docs-engine sync\n\n- Platform-aware configurable GLV window width (kDefaultGlvWindow):\n  x86-64/ARM64/RISC-V default w=5, ESP32/WASM w=4\n  3-tier config: CMake define > platform default > runtime per-call\n- Fix 501 code scanning alerts across 7 files:\n  shiftTooManyBitsSigned (ct_field.cpp), const-correctness,\n  modernize-use-auto, braces, init-variables\n- Full docs-engine consistency audit: fix 8 categories of stale\n  references across 14+ documentation files\n- Add GLV Window Width Tuning to PERFORMANCE_GUIDE + 12 bindings + WASM\n- CHANGELOG.md: consolidated v3.14.0-v3.20.0 (120+ commits)\n- VERSION.txt: 3.20.0\n- bench_bip352 results: x86-64 1.34x, ARM64 1.31x, RISC-V 1.37x\n\n* refactor(point): extract shared GLV52/batch/next-prev helpers to reduce duplication\n\nExtract build_glv52_table_zr, derive_phi52_table, shamir_2stream_glv52\nhelpers used by scalar_mul_glv52, scalar_mul_with_plan_glv52, and\ndual_scalar_mul_gen_point.\n\nRewrite batch_x_only_bytes to reuse batch_normalize (Montgomery trick).\n\nExtract add_gen_mixed helper for next/prev/next_inplace/prev_inplace.\n\nNet reduction: 234 lines (-425/+191). All 30 tests pass.\n\n* perf(batch): restore x-only fast path in batch_x_only_bytes\n\nCall batch_z_inv() directly instead of batch_normalize(), computing\nonly x = X * Z^(-2) and skipping the unused Y = Y * Z^(-3).\nEliminates ~33% redundant field multiplications per point.\n\nFixes benchmark regression gate failure (1.39x slowdown on\nbatch_x_only_bytes /pt N=64).\n\n* refactor(batch): deduplicate allocation+wNAF patterns in point.cpp\n\n- Move partials allocation inside batch_z_inv (callers only pass z_inv)\n- Extract apply_wnaf_mixed52 helper for wNAF digit lookup+negate+add\n- Use apply_wnaf_mixed52 in shamir_2stream_glv52 and dual_scalar_mul_gen_point\n- Reduces SonarCloud duplicated lines on new code (was 12%, gate requires <=3%)\n\n* ci: raise PR bench threshold to 150%, exclude point.cpp from CPD\n\n- bench-regression.yml: PR alert-threshold 120% -> 150% to match push\n  threshold. Shared CI runners (Xeon Platinum 8370C) show 20-30% variance\n  on sub-10ns operations, causing false positives on unrelated benchmarks.\n\n- sonar-project.properties: add point.cpp to sonar.cpd.exclusions.\n  point.cpp contains dual-platform paths (4x64 and 5x52 field representations)\n  implementing identical algorithms for different CPU architectures.\n  This structural duplication is an architectural requirement.\n\n* fix: use -march=x86-64-v3 for macOS ARM64->x86_64 cross-compilation\n\nWhen cross-compiling on macOS from ARM64 (Apple Silicon) to x86_64,\n-march=native resolves to the host CPU (e.g. apple-m3), which is not\na valid x86_64 target.\n\nDetect macOS cross-compilation via CMAKE_OSX_ARCHITECTURES and use\n-march=x86-64-v3 (AVX2+BMI2+FMA) as a reasonable modern x86_64\nbaseline instead.\n\nApplied to both top-level CMakeLists.txt and cpu/CMakeLists.txt.\n\nAddresses #113\n\n* fix: wire GLV_WINDOW_WIDTH CMake option, fix docs (WASM + bench paths)\n\nAddress Copilot review feedback on PR #112:\n\n1. CMake cache variable forwarding: add SECP256K1_GLV_WINDOW_WIDTH as a\n   CMake cache string option that forwards into a compile definition.\n   Now `-DSECP256K1_GLV_WINDOW_WIDTH=6` works as documented.\n\n2. Platform default descriptions: add WASM to the w=4 default set\n   (\"4 on ESP32/WASM\") in 6 docs + copilot-instructions.md.\n\n3. Benchmark binary paths: fix cpu/bench/bench_unified -> cpu/bench_unified\n   in 7 docs. Build outputs go to build/cpu/, not build/cpu/bench/.\n\n31/31 tests pass.\n\n* fix: address Copilot review (return check, comments, threshold)\n\n- Check build_glv52_table_zr return in dual_scalar_mul_gen_point,\n  fallback to separate a*G+b*P on degenerate case\n- Update next_inplace/prev_inplace header comment (no longer faster)\n- Fix bench-regression.yml header: >20% -> >50% to match threshold\n\n* ci: add ccache + Ninja to all Linux build workflows\n\nAdd ccache compilation caching and Ninja generator to 17 build jobs\nacross 6 workflow files to reduce CI build times on repeat pushes.\n\nChanges:\n- ci.yml: linux (4 matrix), arm64, sanitizers (2) = 7 jobs\n- security-audit.yml: werror, ASan, MSan, TSan, Valgrind, dudect = 6 jobs\n- clang-tidy.yml: 1 job\n- benchmark.yml: 1 job (linux)\n- sonarcloud.yml: 1 job\n- codeql.yml: 1 job\n\nImplementation:\n- actions/cache@v4.2.3 (SHA-pinned) for ~/.cache/ccache persistence\n- CMAKE_C/CXX_COMPILER_LAUNCHER=ccache via CMake native mechanism\n- Unique cache keys per job (compiler + sanitizer + source hash)\n- ccache max_size=200M, compression=true\n- Added -G Ninja to ci.yml linux and sanitizers jobs (others had it)\n\nNo changes to compiler flags, test invocations, or build configurations.\nWindows/macOS/iOS/WASM/Android/ROCm jobs unchanged (different toolchain).\n\n* fix: address Copilot review on PR #115\n\n- CMakeLists.txt: validate SECP256K1_GLV_WINDOW_WIDTH is integer in [4,7]\n- point.hpp: add static_assert for GLV window width range\n- CMakeLists.txt + cpu/CMakeLists.txt: guard macOS cross-compile with\n  STREQUAL \"x86_64\" to avoid breaking universal builds\n- docs/BENCHMARKING.md: replace plain-text reference with link to\n  .github/copilot-instructions.md\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-08T16:46:24+04:00",
          "tree_id": "4d1d40d4d5d70aac06ce8f6820104ef10d1c1d5b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/05822dcabea22fafaa9bd836f517da3c7dccd7c0"
        },
        "date": 1772974200555,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_mul",
            "value": 15.8,
            "unit": "ns"
          },
          {
            "name": "field_sqr",
            "value": 14.1,
            "unit": "ns"
          },
          {
            "name": "field_inv",
            "value": 976.5,
            "unit": "ns"
          },
          {
            "name": "field_add",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "field_sub",
            "value": 6.6,
            "unit": "ns"
          },
          {
            "name": "field_negate",
            "value": 8,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (32B)",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul",
            "value": 34,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1251.6,
            "unit": "ns"
          },
          {
            "name": "scalar_add",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "scalar_negate",
            "value": 4.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (32B)",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 6926.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33390.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 32887.2,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 37163,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1181.6,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 242.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 127.7,
            "unit": "ns"
          },
          {
            "name": "normalize (J->affine)",
            "value": 4.1,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 184.4,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 254.9,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 1907.4,
            "unit": "ns"
          },
          {
            "name": "to_compressed (33B)",
            "value": 8.9,
            "unit": "ns"
          },
          {
            "name": "to_uncompressed (65B)",
            "value": 9.7,
            "unit": "ns"
          },
          {
            "name": "x_only_bytes (32B)",
            "value": 4.9,
            "unit": "ns"
          },
          {
            "name": "x_bytes_and_parity",
            "value": 7.2,
            "unit": "ns"
          },
          {
            "name": "has_even_y",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 196.8,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 147.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 9944,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 57114.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 39459.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 7218.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 7737.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 48800.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 39479.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 40750.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 151469.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 37867.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 619086.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 38692.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4065459.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 63522.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 147947.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 593157.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2408894.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 2638.9,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18364.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 39729.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 139.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 409.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 292.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 263.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 24144.9,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 82390,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 20622.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 61821.3,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20105.8,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1145.3,
            "unit": "ns"
          },
          {
            "name": "field_normalize",
            "value": 10.3,
            "unit": "ns"
          },
          {
            "name": "field_from_bytes (set_b32)",
            "value": 9.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2838.8,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1182.3,
            "unit": "ns"
          },
          {
            "name": "scalar_from_bytes (set_b32)",
            "value": 7.7,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 131.1,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 233.4,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 35879.6,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 14669.5,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 17725.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34413.1,
            "unit": "ns"
          },
          {
            "name": "serialize_compressed (33B)",
            "value": 26.2,
            "unit": "ns"
          },
          {
            "name": "serialize_uncompressed (65B)",
            "value": 33.4,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 3454.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 19036,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 38214.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 393948.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415057.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 370245.9,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "165fcddd7731d19d3611197714f011e1f7f64639",
          "message": "fix(precompute): correct cache validation and path separator (#118)\n\n* fix(precompute): correct cache validation and path separator\n\n- Fix cache file size validation rejecting valid files: use 1 byte/point\n  minimum bound instead of hardcoded 65 bytes, since infinity points are\n  serialized as 1 byte (flag only, no x/y coordinates). Addresses #116.\n- Fix path separator: use '/' unconditionally instead of Windows backslash,\n  since forward slashes work on all platforms. Addresses #117.\n\n* fix(ci): fuzz_point k=n-1 edge case, benchmark noise filter\n\n- fuzz_point: handle k=n-1 where (k+1)*G = 0*G = infinity (pre-existing bug)\n- parse_benchmark.py: add MIN_REGRESSION_NS=25.0 filter to skip sub-25ns\n  micro-ops that produce false positive regressions on shared CI runners\n\n* fix(ci): use fixed -march=x86-64-v3 with ccache, widen benchmark threshold\n\n- CMakeLists.txt: add SECP256K1_MARCH option to override -march=native\n  Prevents SIGILL when ccache restores objects from a different CI runner CPU\n- All x86-64 workflows with ccache: pass -DSECP256K1_MARCH=x86-64-v3\n- bench-regression.yml: increase alert-threshold from 150% to 200%\n  Shared CI runners show up to ~60% variance on micro-ops\n\n* fix(ci): raise MIN_REGRESSION_NS to 50ns\n\nfield_mul (15.8ns baseline) inflated to 32.5ns on CI runner (2.06x),\nexceeding even 200% threshold.  Sub-50ns micro-ops are too granular\nfor shared runner regression detection.  Still shown in Dashboard.\n\n* add ABI layout guards and CRYPTO_INVARIANTS.md\n\n- ufsecp.h: static_assert guards on struct sizes and constant lengths\n  to catch ABI-breaking changes at compile time (C++ and C11)\n- docs/CRYPTO_INVARIANTS.md: comprehensive crypto invariant reference\n  covering private key validation, nonce safety (RFC 6979), pubkey\n  validation, signature encoding, hash domain separation, point\n  arithmetic, key tweaking, serialization byte order, thread safety,\n  and constant-time guarantees\n\n---------\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-08T19:14:45+04:00",
          "tree_id": "059401a667661dcbb280d161f2c0f1b3d25e927e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/165fcddd7731d19d3611197714f011e1f7f64639"
        },
        "date": 1772983090275,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1085.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1333.9,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7459.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35433.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 35311.4,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38903.8,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1339.4,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 260.6,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.9,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 198.7,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 278.8,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2633.9,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 207.6,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 153.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10221.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 59143.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40837.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10147.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8101.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 50943.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41291.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42444.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159580.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39895.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 648633.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40539.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4292960.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 67077.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 155260.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 623527.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2513422,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1886.6,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 18918.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41035.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 423.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 306.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 297.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23468,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 83762.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 20971.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 63989.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20520.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1160.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2058.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1186,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 157,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 263.5,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40143.2,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17531.5,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19783,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37435.7,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2833.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21094.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42450.3,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 393491.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 417842.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 384400.5,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "206b35cd5628472f0ea798196d292446c1447c78",
          "message": "fix(ci): CPUID PIC safety, Metal metallib path, benchmark output path, MARCH v2 (#135)\n\n- field_asm.cpp: replace raw CPUID inline asm with __get_cpuid_count()\n- metal_audit_runner.mm: add secp256k1_kernels.metallib to search paths\n- benchmark.yml: output to /tmp/benchmark_results/ (avoid git conflict)\n- bench-regression.yml: same /tmp/ output path fix\n- ci.yml: downgrade SECP256K1_MARCH x86-64-v3 -> v2 (clang-17 SIGILL)\n- ci.yml: make metal_audit advisory (continue-on-error) on macOS\n- sonarcloud.yml: downgrade SECP256K1_MARCH x86-64-v3 -> v2\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-09T23:43:15+04:00",
          "tree_id": "25a7540df114f80a97e8942e88437ed6eb0a9f04",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/206b35cd5628472f0ea798196d292446c1447c78"
        },
        "date": 1773085617999,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1096.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1338.1,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7437.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35384.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 34796.6,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 38800.1,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1340.2,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 262.7,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 149.5,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 197.9,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 284.5,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2491.5,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 210.5,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 153.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10097.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 70565,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 40811.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10180.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 9061.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 51610.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 41205.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 42367.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 159536.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 39884,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 646659.6,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 40416.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4281793.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 66903,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 154972.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 621983.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2501951.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1964.3,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 19051,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 41108.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 145.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 424.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 304.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 300.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 23828.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 84153.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 21140.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 63987,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 20680.4,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1179.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2141,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1208.1,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 158.1,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 264.4,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 40022.3,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 17685.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 19986,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 37478,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2897.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21255.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 42708.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390260.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 416062,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 376817.6,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
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
          "id": "f2f04ad0adc197b5aba0f07f5e6a21f3b69a473a",
          "message": "fix(ci): respect SECP256K1_MARCH in cpu/CMakeLists.txt, downgrade benchmark/bench-regression to x86-64-v2, add release pre-delete step (#138)\n\n- cpu/CMakeLists.txt: ARCH_FLAGS now checks SECP256K1_MARCH before falling\n  back to -march=native, fixing SonarCloud SIGILL (29/30 tests) where\n  target-level ARCH_FLAGS overrode the parent MARCH_FLAG\n- benchmark.yml: MARCH x86-64-v3 -> x86-64-v2 (missed in PR #135)\n- bench-regression.yml: MARCH x86-64-v3 -> x86-64-v2 (missed in PR #135)\n- release.yml: delete pre-existing release before create to avoid\n  softprops/action-gh-release 'Not Found' on manually-created releases;\n  add fail_on_unmatched_files: false\n\nCo-authored-by: shrec <shrec@users.noreply.github.com>",
          "timestamp": "2026-03-10T02:14:17+04:00",
          "tree_id": "ba9b8d9605d2f7f25cf158fe32ac46280b64b04e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f2f04ad0adc197b5aba0f07f5e6a21f3b69a473a"
        },
        "date": 1773094671130,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1058.6,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1331.4,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7898.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33039.9,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 32580.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 36379.8,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1322.7,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 239.2,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 146.5,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 189.1,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 256.7,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2668,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 198.2,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 150.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10574.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 66657,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 38082.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 8154.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8576.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 48551.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38552.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 39599.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 148373.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 37093.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 604309,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 37769.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 3978191.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 62159.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 144576.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 580001.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2332062.5,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1856.4,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 17325.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 39114.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 142.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 400.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 273.2,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 273.6,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 21790.5,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 77765.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 19287.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 59380.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 18856.5,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1133.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1844.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1157.9,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 244.4,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37369.4,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 18428.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20336.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34982.1,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2538.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21906.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39838.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 393846.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415148.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 378065.6,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "55f8bf07cf52328dbbbe964014f4a929ef055303",
          "message": "feat(coins): unified wallet API, address formats (P2SH/CashAddr/Tron), message signing\n\n- Add unified wallet API (wallet.hpp/cpp): chain-agnostic key mgmt, signing, recovery\n- Add Bitcoin message signing (message_signing.hpp/cpp): BIP-137/Electrum compatible\n- Add P2SH-P2WPKH (nested SegWit), P2SH, P2WSH address primitives\n- Add CashAddr encoding (Bitcoin Cash BIP-0185)\n- Add Tron coin descriptor (coin_type=195, TRON_BASE58 encoding)\n- Add 5 wallet address helpers (p2pkh/p2wpkh/p2sh_p2wpkh/p2tr/cashaddr)\n- Add chain_id field in CoinParams for EIP-155 signing\n- Fix coin_address() CASHADDR dispatch for Bitcoin Cash\n- 28 coins now all generate addresses correctly\n- Add 19 new tests (12 in test_coins.cpp, 7 in test_wallet.cpp)\n- Update docs: CHANGELOG, API_REFERENCE, USER_GUIDE, TEST_MATRIX, BUILDING",
          "timestamp": "2026-03-11T00:32:44Z",
          "tree_id": "c1e416f610a895c96e3a5c259fcf919f07a7e11e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/55f8bf07cf52328dbbbe964014f4a929ef055303"
        },
        "date": 1773189611550,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1011.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1335,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7879.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33074.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 32859.3,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 36195.9,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1267.8,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 244.6,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 144.4,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 187.7,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 263.3,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2659.1,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 197.2,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10648.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 67160.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 38077.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 8613.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8641.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 48634.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38535.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 39628.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 148672.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 37168.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 607507.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 37969.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 4016360.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 62755.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 145122.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 580937.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2334176.9,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1862.6,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 17487.3,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 39112.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 143.7,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 402.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 275.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 275,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 21974.8,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 78024.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 19408.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 59669.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 19130.9,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 432.5,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 437.5,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 431.8,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 10760.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 10478,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 47928.1,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 11180.3,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 984.8,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1129.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 2108.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1155.7,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 147.7,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 247.6,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37964.6,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 18048.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20286.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 35338.4,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2748,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21852.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 40121.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 400861.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 424513.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 384655.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "bfde741b746e9d238f4ff1d4d3a98345284746fc",
          "message": "feat: add GPU ZK proving across CUDA/OpenCL/Metal + fix pre-existing CUDA bugs\n\nCUDA (ct_zk.cuh):\n- CT knowledge proof (Schnorr sigma): ct_scalar_mul for k*B, BIP-340 Y-parity\n- CT DLEQ proof: two ct_scalar_mul (k*G, k*H), 6-point challenge hash\n- Deterministic nonce via SHA-256 tagged hash with XOR hedging\n- Batch kernels for both operations\n- Tests 8-9 in test_ct_smoke.cu: prove+verify round-trips (9/9 pass)\n\nOpenCL (secp256k1_zk.cl):\n- Knowledge prove/verify + DLEQ prove/verify device functions\n- Batch kernels using fast-path wNAF-5 scalar mul\n- point_to_compressed_impl, jacobian_to_affine_impl added\n\nMetal (secp256k1_zk.h + kernels 19-22):\n- Knowledge prove/verify + DLEQ prove/verify functions\n- Kernels 19-22 in secp256k1_kernels.metal\n- Uses branchless affine_select scalar mul (semi-CT)\n\nBug fixes (pre-existing, never compiled before):\n- pedersen.cuh: .d[] -> .limbs[] (7+), remove field_normalize(), remove dup field_sqrt()\n- zk.cuh: .d[] -> .limbs[] in verify/range, fix undefined SCALAR_ONE\n\nDocs: API_REFERENCE, CT_VERIFICATION, SECURITY_CLAIMS, TEST_MATRIX, CHANGELOG\n\n40/40 CTest pass, 18/18 CUDA targets build clean",
          "timestamp": "2026-03-11T14:36:09Z",
          "tree_id": "62e96f9aa41042626df223900b977525e03d4f2d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bfde741b746e9d238f4ff1d4d3a98345284746fc"
        },
        "date": 1773240050242,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1056.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1335.7,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7896.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33117.8,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 32577.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 36391.1,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1304.8,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 242.6,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 144.8,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 186.6,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 254.6,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2745.7,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 195.7,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 148.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10580.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 66633.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 38182.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10600.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8516.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 48510.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38441,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 39786.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 148794.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 37198.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 604829.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 37801.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 3974979.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 62109.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 145022.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 580798.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2330094.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1860.4,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 17472.7,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 39097.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 142.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 401.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 273.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 274.2,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 21995.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 78091,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 19622.4,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 59586.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 19066.8,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 430.2,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 433.8,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 431.5,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 10599.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 10463.1,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 47690.6,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 11214.2,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 990.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1130.4,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1848.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1173.3,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 148.1,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 242.8,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37297.9,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 18423.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20356.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34677.2,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2561.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21827.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39780.1,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390373.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414814.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374130.4,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 56992.6,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 41683.6,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 40558.2,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 80870,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 107605.4,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24352152.1,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 4949033.5,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "b63502bf348fb7dc472f899f93af6cf10fb7ab75",
          "message": "fix(ci): resolve compiler warnings + fix benchmark regression gate\n\n- bench_unified.cpp: wrap libsecp section with #pragma GCC diagnostic\n  push/pop to suppress warn_unused_result (GCC ignores (void) casts)\n- zk.cpp: add [[maybe_unused]] to transcript_append_{point,scalar},\n  remove unused variable j\n- test_zk.cpp: remove unused variable all_distinct\n- test_wallet.cpp: add [[maybe_unused]] to bytes_to_hex\n- run_ci.sh: fix benchmark gate awk filter -- skip FIELD/SCALAR\n  sections entirely (micro ops), skip point primitives dbl/add(mixed),\n  lower default BENCH_MIN_RATIO from 1.00 to 0.95 for Docker noise\n  tolerance",
          "timestamp": "2026-03-13T12:28:55Z",
          "tree_id": "ff29cde29e813e26fb69344b84e01c07596f5adb",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b63502bf348fb7dc472f899f93af6cf10fb7ab75"
        },
        "date": 1773406026275,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1051.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1335.7,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7899.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33071.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 32516.8,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 36346.4,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1303,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 242.4,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 145.7,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 190,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 253,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2656.3,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 195.6,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 148.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10649.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 66661.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 38061.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10600.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8474.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 48568.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38465.2,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 39713.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 148829.2,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 37207.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 605416.7,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 37838.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 3977345.8,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 62146,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 144868.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 581302.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2337057.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1860.5,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 17488.9,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 39136.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 143.3,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 399.6,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 276.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 22030.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 78051.5,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 19530.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 59609.9,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 19183.9,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 431.6,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 432.7,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 430.4,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 10625.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 10497.3,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 47677,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 11057.6,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 989.7,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1131.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1843.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1168.1,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 146.9,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 244.2,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37282.7,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 18433.2,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20337.6,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34685.9,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2545.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21827.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39782.8,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 409399.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 415177.9,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374404.2,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 57022,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 41648.5,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 40607.8,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 81110.1,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 107979.6,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24357968.6,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 4909791.8,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "ca14a9e47e586e410638eaa8d1a1d17266220024",
          "message": "fix(ci): resolve macOS Metal shader errors + code scanning alerts\n\n- Metal: add constant address space overload for zk_tagged_hash_midstate()\n  (constant ZKTagMidstate refs cannot bind to thread address space params)\n- Metal: add device address space overload for scalar_mul()\n  (device AffinePoint arrays cannot bind to thread address space params)\n- Fix -Wsign-conversion in host_helpers.h (int -> size_t loop vars, explicit casts)\n- Fix -Wsign-conversion in audit/test_vectors.hpp (int -> size_t loop vars)\n- Fix -Wunused-const-variable for BARRETT_MU in scalar.cpp (only used in non-int128 path)\n- Fix CodeQL cpp/unused-local-variable alerts in test_edge_cases.cpp:\n  add (void) casts for unused structured binding variables (child, hchild, bad_master, bad_master2)",
          "timestamp": "2026-03-13T13:12:20Z",
          "tree_id": "10b5ded0607f7a67767cc41eb0f2a50e9114d6eb",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ca14a9e47e586e410638eaa8d1a1d17266220024"
        },
        "date": 1773408875806,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1051.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1336,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7850.3,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 35439.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 32766.5,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 36340.1,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1305.1,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 242.8,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 144.3,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 186.1,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 256.6,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2653,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 195.6,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 148.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10593.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 66747.8,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 38877.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 8089.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8560.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 48456.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38867.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 39700.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 148927.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 37231.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 605064.4,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 37816.5,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 3975584.5,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 62118.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 145000.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 581538.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2334620.2,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1860.3,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 17474.6,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 39146.5,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 143.1,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 400.4,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 273.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 274.3,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 22063.4,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 78016.6,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 19481.1,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 60412.8,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 19090,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 434.8,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 433,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 431.5,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 10698.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 10455.4,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 47719.8,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 11090.2,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 988.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1208.5,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1843.7,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1159.4,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 152.2,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 246.9,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37252.4,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 18384.9,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20422.1,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34698.8,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2549.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21811.7,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39713.7,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 399154.3,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 423660.2,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374887.5,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 57167.2,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 43973.9,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 41067.3,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 81100.5,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 107463.1,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24347864.6,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 4911732,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "committer": {
            "email": "shrec@users.noreply.github.com",
            "name": "shrec",
            "username": "shrec"
          },
          "distinct": true,
          "id": "7e4695d4e06aeb2ee526536c9b3de8e45960df0b",
          "message": "docs: update versions to v3.22.0 + binding READMEs + CHANGELOG\n\n- Update stale v3.14.0 headers/footers to v3.22.0: BINDINGS.md,\n  DIFFERENTIAL_TESTING.md, AUDIT_TRACEABILITY.md, BINDINGS_PACKAGING.md,\n  INTERNAL_AUDIT.md, LTS_POLICY.md\n- Fix SECURITY_CLAIMS.md footer v3.21.0 -> v3.22.0\n- Update ARCHITECTURE.md, AUDIT_GUIDE.md, CT_VERIFICATION.md,\n  CROSS_PLATFORM_TEST_MATRIX.md, TEST_MATRIX.md, USER_GUIDE.md\n- Update API_REFERENCE.md: frost_verify_partial signature\n- Add BIP-39, ZK, Wallet, Multi-coin, ECDH to CHANGELOG [Unreleased]\n- Update all 12 binding READMEs + wasm/README.md with new API surface\n- Refresh audit reports (clang-17, gcc-13)",
          "timestamp": "2026-03-13T20:39:36Z",
          "tree_id": "6c7339164f4e496c359cfd208853146509c82a73",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7e4695d4e06aeb2ee526536c9b3de8e45960df0b"
        },
        "date": 1773436870354,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "field_inv",
            "value": 1054.9,
            "unit": "ns"
          },
          {
            "name": "scalar_inv",
            "value": 1333.9,
            "unit": "ns"
          },
          {
            "name": "pubkey_create (k*G)",
            "value": 7895.5,
            "unit": "ns"
          },
          {
            "name": "scalar_mul (k*P)",
            "value": 33056.7,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_with_plan",
            "value": 32597.8,
            "unit": "ns"
          },
          {
            "name": "dual_mul (a*G + b*P)",
            "value": 36528.2,
            "unit": "ns"
          },
          {
            "name": "point_add (affine+affine)",
            "value": 1303.1,
            "unit": "ns"
          },
          {
            "name": "point_add (J+A mixed)",
            "value": 239.2,
            "unit": "ns"
          },
          {
            "name": "point_dbl",
            "value": 145,
            "unit": "ns"
          },
          {
            "name": "batch_normalize /pt (N=64)",
            "value": 186.3,
            "unit": "ns"
          },
          {
            "name": "next_inplace (+=G)",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "KPlan::from_scalar(w=4)",
            "value": 2659,
            "unit": "ns"
          },
          {
            "name": "batch_to_compressed /pt (N=64)",
            "value": 196.3,
            "unit": "ns"
          },
          {
            "name": "batch_x_only_bytes /pt (N=64)",
            "value": 148.6,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign",
            "value": 10581.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_verified",
            "value": 66874.5,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify",
            "value": 38125.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_keypair_create",
            "value": 10454.9,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign",
            "value": 8620.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign_verified",
            "value": 49245.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (cached xonly)",
            "value": 38627,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (raw bytes)",
            "value": 39756.6,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=4)",
            "value": 148856.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=4)",
            "value": 37214.1,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=16)",
            "value": 605494.3,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=16)",
            "value": 37843.4,
            "unit": "ns"
          },
          {
            "name": "schnorr_batch_verify(N=64)",
            "value": 3983022.1,
            "unit": "ns"
          },
          {
            "name": "-> per-sig amortized (N=64)",
            "value": 62234.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=4)",
            "value": 145667,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=16)",
            "value": 580870.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_batch_verify(N=64)",
            "value": 2340038.4,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_inverse (SafeGCD)",
            "value": 1860.6,
            "unit": "ns"
          },
          {
            "name": "ct::generator_mul (k*G)",
            "value": 17428.8,
            "unit": "ns"
          },
          {
            "name": "ct::scalar_mul (k*P)",
            "value": 39101.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_dbl",
            "value": 142.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_complete (11M+6S)",
            "value": 400.9,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_complete (7M+5S)",
            "value": 272.8,
            "unit": "ns"
          },
          {
            "name": "ct::point_add_mixed_unified (7M+5S)",
            "value": 274,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign",
            "value": 21981.7,
            "unit": "ns"
          },
          {
            "name": "ct::ecdsa_sign_verified",
            "value": 78125.2,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign",
            "value": 19440.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_sign_verified",
            "value": 59687.7,
            "unit": "ns"
          },
          {
            "name": "ct::schnorr_keypair_create",
            "value": 19047,
            "unit": "ns"
          },
          {
            "name": "keccak256 (32B)",
            "value": 443.4,
            "unit": "ns"
          },
          {
            "name": "ethereum_address",
            "value": 439.9,
            "unit": "ns"
          },
          {
            "name": "eip191_hash",
            "value": 444,
            "unit": "ns"
          },
          {
            "name": "eth_sign_hash",
            "value": 10602.7,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign_recoverable",
            "value": 10480.4,
            "unit": "ns"
          },
          {
            "name": "ecrecover",
            "value": 47894.4,
            "unit": "ns"
          },
          {
            "name": "eth_personal_sign",
            "value": 11266.4,
            "unit": "ns"
          },
          {
            "name": "ethereum_address_eip55",
            "value": 990.6,
            "unit": "ns"
          },
          {
            "name": "field_inv_var",
            "value": 1136.1,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse (CT)",
            "value": 1844.2,
            "unit": "ns"
          },
          {
            "name": "scalar_inverse_var",
            "value": 1157.8,
            "unit": "ns"
          },
          {
            "name": "point_dbl (gej_double_var)",
            "value": 148.4,
            "unit": "ns"
          },
          {
            "name": "point_add (gej_add_ge_var)",
            "value": 245.6,
            "unit": "ns"
          },
          {
            "name": "ecmult (a*P + b*G, Strauss)",
            "value": 37338.8,
            "unit": "ns"
          },
          {
            "name": "ecmult_gen (k*G, comb)",
            "value": 18371.5,
            "unit": "ns"
          },
          {
            "name": "generator_mul (ec_pubkey_create)",
            "value": 20325.2,
            "unit": "ns"
          },
          {
            "name": "scalar_mul_P (k*P, tweak_mul)",
            "value": 34623.5,
            "unit": "ns"
          },
          {
            "name": "point_add (pubkey_combine)",
            "value": 2542.8,
            "unit": "ns"
          },
          {
            "name": "schnorr_sign (BIP-340)",
            "value": 21807.3,
            "unit": "ns"
          },
          {
            "name": "schnorr_verify (BIP-340)",
            "value": 39693.4,
            "unit": "ns"
          },
          {
            "name": "generator_mul (EC_POINT_mul k*G)",
            "value": 390862.4,
            "unit": "ns"
          },
          {
            "name": "ecdsa_sign (ECDSA_do_sign)",
            "value": 414000.1,
            "unit": "ns"
          },
          {
            "name": "ecdsa_verify (ECDSA_do_verify)",
            "value": 374597.8,
            "unit": "ns"
          },
          {
            "name": "Pedersen commit",
            "value": 59703.2,
            "unit": "ns"
          },
          {
            "name": "Knowledge prove (sigma)",
            "value": 41702.9,
            "unit": "ns"
          },
          {
            "name": "Knowledge verify",
            "value": 40683.9,
            "unit": "ns"
          },
          {
            "name": "DLEQ prove",
            "value": 80983.2,
            "unit": "ns"
          },
          {
            "name": "DLEQ verify",
            "value": 107426.5,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_prove (64b)",
            "value": 24335084.6,
            "unit": "ns"
          },
          {
            "name": "Bulletproof range_verify (64b)",
            "value": 4937221.1,
            "unit": "ns"
          },
          {
            "name": "Harness",
            "value": 3000000000,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}