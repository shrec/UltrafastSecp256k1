window.BENCHMARK_DATA = {
  "lastUpdate": 1772405116634,
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
      }
    ]
  }
}