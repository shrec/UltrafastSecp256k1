window.BENCHMARK_DATA = {
  "lastUpdate": 1772651524135,
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
      }
    ]
  }
}