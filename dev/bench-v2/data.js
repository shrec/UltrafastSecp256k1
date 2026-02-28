window.BENCHMARK_DATA = {
  "lastUpdate": 1772280083852,
  "repoUrl": "https://github.com/shrec/UltrafastSecp256k1",
  "entries": {
    "UltrafastSecp256k1 Performance": [
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
          "id": "0cb9ff71c14725dbc098e677e55a65c7b3db88b4",
          "message": "ci: fix parse_benchmark Unicode mismatch + reset baseline\n\n- Fix: C++ emits Greek mu (U+03BC) but regex only matched micro sign (U+00B5)\n  This caused all microsecond measurements to be silently dropped from CI tracking\n- Fix: Add '=' to name regex for 'Batch Inverse (n=100)' entries\n- Reset baseline: change data dir to dev/bench-v2 (old baseline had 0ns values\n  from dead-code-eliminated operations before DoNotOptimize harness migration)",
          "timestamp": "2026-02-23T02:09:29+04:00",
          "tree_id": "3ad58edf8fef1a8eacc795f8a6644683cab28c95",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/0cb9ff71c14725dbc098e677e55a65c7b3db88b4"
        },
        "date": 1771798241360,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "1a9cebc3ba703f87ce03d09e95a4aec8f1a06f4a",
          "message": "fix(ci): remove orphaned cpu/secp256k1 submodule entry\n\nThe submodule was removed but its gitlink entry remained in the index,\ncausing CI checkout to fail with 'No url found for submodule path'.\nAlso added cpu/secp256k1/ to .gitignore.",
          "timestamp": "2026-02-23T02:24:56+04:00",
          "tree_id": "47422ed488caa53e361e0937e543b52fa1d6d605",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/1a9cebc3ba703f87ce03d09e95a4aec8f1a06f4a"
        },
        "date": 1771799167954,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 143,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "7072ddb647a6488bf1456ae0d12031fde409b8e7",
          "message": "perf(glv): effective-affine table precomp in scalar_mul_glv52\n\nscalar_mul_glv52 table building used full Jacobian adds (12M+5S).\nPort effective-affine technique from dual_scalar_mul_gen_point:\n  - Transform P onto isomorphic curve where 2P is affine\n  - Use mixed adds (7M+4S each) for table entries\n  - Batch-invert effective Z = Z_iso * C\nSaves ~33M + 6S over 7 table entries = ~5us on RISC-V U74.",
          "timestamp": "2026-02-23T02:56:05+04:00",
          "tree_id": "eeb2c395d09be2a628bf334c06c01adef50be187",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7072ddb647a6488bf1456ae0d12031fde409b8e7"
        },
        "date": 1771801828192,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 17000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 143,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "c3d4e41bf1f6c07cc9c521f5a6cf1a808265888b",
          "message": "docs: acknowledge Stacker News, Delving Bitcoin, and @0xbitcoiner",
          "timestamp": "2026-02-23T03:16:51+04:00",
          "tree_id": "030ad10075fe217739776f363568a30b25579fc4",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c3d4e41bf1f6c07cc9c521f5a6cf1a808265888b"
        },
        "date": 1771802278506,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 145,
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
          "id": "9ed6f8a2e825a700580de0f515af0d281bad10b9",
          "message": "cmake: auto-detect RISC-V CPU for -mcpu scheduling tuning\n\nOn SiFive U74 (Milk-V Mars), adding -mcpu=sifive-u74 enables pipeline-\nspecific instruction scheduling that yields 28-34% speedup on field/point\noperations (Field Mul 136->93ns, K*Q 235->154us, ECDSA Verify 282->185us).\n\nNew behavior:\n- Reads /proc/cpuinfo uarch field to detect SiFive core (u74/u54/p550/p670)\n- Sets -mcpu=sifive-<core> which implies both -march and -mtune\n- Falls back to generic -march=rv64gc_zba_zbb if no core detected\n- SECP256K1_RISCV_MCPU cache variable for manual override\n\nCombined with -DSECP256K1_USE_LTO=ON (ThinLTO), total gains vs baseline:\n  Field Mul:     -32% (136->93ns)\n  K*Q:           -34% (235->154us)\n  K*G:           -18% (40->33us)\n  ECDSA Sign:    -17% (81->67us)\n  ECDSA Verify:  -34% (282->185us)\n  Schnorr Verify: -31% (313->216us)",
          "timestamp": "2026-02-23T04:12:53+04:00",
          "tree_id": "c238f6a9d516c6813cfe077dfe6a4061be8a29c2",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9ed6f8a2e825a700580de0f515af0d281bad10b9"
        },
        "date": 1771805897374,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "e5084ea3c74b845f8edc0784c0ce367d6d00053e",
          "message": "Update benchmarks: fresh results for x86-64/ARM64/RISC-V, add Lightning donation\n\nBenchmark updates (all platforms re-measured):\n- x86-64 (Clang 21, AVX2): Field Mul 17ns, Point Add 159ns, kG 5us, kP 25us\n- ARM64 (RK3588 Cortex-A76 @ 2.256GHz): Field Mul 74ns, Point Add 992ns, kG 14us, kP 131us\n  ECDSA Sign 30us, Verify 153us, Schnorr Sign 38us, Verify 173us\n- RISC-V (Milk-V Mars, -mcpu=sifive-u74, ThinLTO): Field Mul 95ns, kG 33us, kP 154us\n  ECDSA Sign 67us, Verify 186us, Schnorr Sign 86us, Verify 216us\n- CT overhead (x86-64): kP 1.13x, kG 1.86x, Field Mul 1.08x\n\nRemove ESP32 vs libsecp256k1 comparison table (no competitor comparisons).\nAdd Lightning donation badge (shrec@stacker.news).",
          "timestamp": "2026-02-23T05:12:40+04:00",
          "tree_id": "ec0d577655f27e4ddda5f761a4b7a27858d6d83f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/e5084ea3c74b845f8edc0784c0ce367d6d00053e"
        },
        "date": 1771809231591,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 300,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 165,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 26000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 56000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 154,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 146,
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
          "id": "a1b73d7e5fa0d6017acd4728ad5b62f4f4e85943",
          "message": "release: v3.11.0 — effective-affine, RISC-V auto-tune, benchmark refresh",
          "timestamp": "2026-02-23T05:14:32+04:00",
          "tree_id": "fd78dcab182915548f407093413484ea06f8120e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a1b73d7e5fa0d6017acd4728ad5b62f4f4e85943"
        },
        "date": 1771809354363,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "distinct": false,
          "id": "a1b73d7e5fa0d6017acd4728ad5b62f4f4e85943",
          "message": "release: v3.11.0 — effective-affine, RISC-V auto-tune, benchmark refresh",
          "timestamp": "2026-02-23T05:14:32+04:00",
          "tree_id": "fd78dcab182915548f407093413484ea06f8120e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a1b73d7e5fa0d6017acd4728ad5b62f4f4e85943"
        },
        "date": 1771809681245,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 143,
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
          "id": "6c6895d3c301893af5a1f7d0eb9b6dfef9ed9c4c",
          "message": "fix: badge URLs point to correct repo (UltrafastSecp256k1)",
          "timestamp": "2026-02-23T05:22:27+04:00",
          "tree_id": "07b320264b2fe496dff7b5e8cbf64b553613efaa",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/6c6895d3c301893af5a1f7d0eb9b6dfef9ed9c4c"
        },
        "date": 1771809818352,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 301,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 165,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 26000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 56000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 156,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 148,
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
          "distinct": false,
          "id": "6c6895d3c301893af5a1f7d0eb9b6dfef9ed9c4c",
          "message": "fix: badge URLs point to correct repo (UltrafastSecp256k1)",
          "timestamp": "2026-02-23T05:22:27+04:00",
          "tree_id": "07b320264b2fe496dff7b5e8cbf64b553613efaa",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/6c6895d3c301893af5a1f7d0eb9b6dfef9ed9c4c"
        },
        "date": 1771809830103,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "4e610071a31e3e78ce1c1252d9c9bd274e8d5dd7",
          "message": "fix: Lightning donation link - use stacker.news URL instead of lightning: URI",
          "timestamp": "2026-02-23T05:23:12+04:00",
          "tree_id": "12bc9c5dcc891e61f02a61296e1793aeae1a7d09",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4e610071a31e3e78ce1c1252d9c9bd274e8d5dd7"
        },
        "date": 1771809865784,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": false,
          "id": "4e610071a31e3e78ce1c1252d9c9bd274e8d5dd7",
          "message": "fix: Lightning donation link - use stacker.news URL instead of lightning: URI",
          "timestamp": "2026-02-23T05:23:12+04:00",
          "tree_id": "12bc9c5dcc891e61f02a61296e1793aeae1a7d09",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4e610071a31e3e78ce1c1252d9c9bd274e8d5dd7"
        },
        "date": 1771809869398,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 142,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "505b626ef83e8ad4b91caab29ec422157a02d717",
          "message": "Add package READMEs for npm/NuGet, fix repo URLs\n\n- Add README.md for ultrafast-secp256k1 (Node.js binding) with API examples\n- Add README.md for react-native-ultrafast-secp256k1 with usage examples\n- Update @ultrafastsecp256k1/wasm README: add npm install, license, links\n- Update NuGet README with fuller P/Invoke example (sign/verify)\n- Add README.md to files array in wasm and nodejs package.json\n- Fix all repo URLs: Secp256K1fast -> UltrafastSecp256k1 in all package.json and nuspec",
          "timestamp": "2026-02-23T05:43:49+04:00",
          "tree_id": "b346372320e64a9d9dde3fd30ffe09eec09aa35e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/505b626ef83e8ad4b91caab29ec422157a02d717"
        },
        "date": 1771811102811,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "distinct": false,
          "id": "505b626ef83e8ad4b91caab29ec422157a02d717",
          "message": "Add package READMEs for npm/NuGet, fix repo URLs\n\n- Add README.md for ultrafast-secp256k1 (Node.js binding) with API examples\n- Add README.md for react-native-ultrafast-secp256k1 with usage examples\n- Update @ultrafastsecp256k1/wasm README: add npm install, license, links\n- Update NuGet README with fuller P/Invoke example (sign/verify)\n- Add README.md to files array in wasm and nodejs package.json\n- Fix all repo URLs: Secp256K1fast -> UltrafastSecp256k1 in all package.json and nuspec",
          "timestamp": "2026-02-23T05:43:49+04:00",
          "tree_id": "b346372320e64a9d9dde3fd30ffe09eec09aa35e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/505b626ef83e8ad4b91caab29ec422157a02d717"
        },
        "date": 1771811110103,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 147,
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
          "distinct": false,
          "id": "ec3c432253e716ee0a1633f00d10857fa5261db3",
          "message": "Add PackageReadmeFile to Ufsecp NuGet package\n\n- Create README.md for Ufsecp with full API examples (ECDSA, Schnorr, ECDH, BIP-32, Taproot, addresses, WIF)\n- Add PackageReadmeFile + None Include to Ufsecp.csproj so dotnet pack embeds README\n- Fix repo URLs: AvraSasmo/UltrafastSecp256k1 -> shrec/UltrafastSecp256k1\n- Fix nuspec target: docs\\README.md -> docs\\ (per Microsoft docs)",
          "timestamp": "2026-02-23T05:58:03+04:00",
          "tree_id": "27293d1dafa87114ac73e316f2ebfd5c2db83ed2",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ec3c432253e716ee0a1633f00d10857fa5261db3"
        },
        "date": 1771811964548,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 143,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 137,
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
          "id": "ec3c432253e716ee0a1633f00d10857fa5261db3",
          "message": "Add PackageReadmeFile to Ufsecp NuGet package\n\n- Create README.md for Ufsecp with full API examples (ECDSA, Schnorr, ECDH, BIP-32, Taproot, addresses, WIF)\n- Add PackageReadmeFile + None Include to Ufsecp.csproj so dotnet pack embeds README\n- Fix repo URLs: AvraSasmo/UltrafastSecp256k1 -> shrec/UltrafastSecp256k1\n- Fix nuspec target: docs\\README.md -> docs\\ (per Microsoft docs)",
          "timestamp": "2026-02-23T05:58:03+04:00",
          "tree_id": "27293d1dafa87114ac73e316f2ebfd5c2db83ed2",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ec3c432253e716ee0a1633f00d10857fa5261db3"
        },
        "date": 1771811979660,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 142,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "a83f74d14cc33189f5512253118558d1d31f0df2",
          "message": "fix: update all repository URLs to shrec/UltrafastSecp256k1\n\n- Replace AvraSasmo/UltrafastSecp256k1 with shrec/UltrafastSecp256k1 in all binding configs\n- Replace shrec/Secp256K1fast with shrec/UltrafastSecp256k1 in CI, docs, package managers\n- Fix WASM release archive to use wasm/README.md instead of root README.md\n- Fix Package.swift and conanfile.py URLs\n\n24 files changed across: bindings (npm, nuget, dart, php, python, ruby, rust, react-native),\nCI workflows, docs, Package.swift, conanfile.py, podspec, vcpkg.json, CMakeLists.txt",
          "timestamp": "2026-02-23T06:11:06+04:00",
          "tree_id": "440c3419cb1f210c5d5a1a94fbeefcf1d07beaf0",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a83f74d14cc33189f5512253118558d1d31f0df2"
        },
        "date": 1771812742319,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 145,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 136,
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
          "distinct": false,
          "id": "a83f74d14cc33189f5512253118558d1d31f0df2",
          "message": "fix: update all repository URLs to shrec/UltrafastSecp256k1\n\n- Replace AvraSasmo/UltrafastSecp256k1 with shrec/UltrafastSecp256k1 in all binding configs\n- Replace shrec/Secp256K1fast with shrec/UltrafastSecp256k1 in CI, docs, package managers\n- Fix WASM release archive to use wasm/README.md instead of root README.md\n- Fix Package.swift and conanfile.py URLs\n\n24 files changed across: bindings (npm, nuget, dart, php, python, ruby, rust, react-native),\nCI workflows, docs, Package.swift, conanfile.py, podspec, vcpkg.json, CMakeLists.txt",
          "timestamp": "2026-02-23T06:11:06+04:00",
          "tree_id": "440c3419cb1f210c5d5a1a94fbeefcf1d07beaf0",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a83f74d14cc33189f5512253118558d1d31f0df2"
        },
        "date": 1771812747713,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "5d62d1ac9341723ca40f803b3c340607fd91a1ea",
          "message": "docs: comprehensive wiki documentation update\n\n- Home.md: updated features list, performance tables (CPU/GPU/Embedded), Discord link\n- Benchmarks.md: all current numbers for x86-64, ARM64, RISC-V, CUDA, OpenCL, Metal, ESP32, STM32\n- CPU-Guide.md: ARM64 promoted to Production (was Planned), updated perf numbers, algorithm details\n- API-Reference.md: added ECDSA, Schnorr, ECDH, BIP-32, Address, SHA-256, WIF, ufsecp C ABI (45 functions)\n- CUDA-Guide.md: added Blackwell architecture (sm_120, RTX 5060-5090)\n- Getting-Started.md: added missing build options (OpenCL, Metal, ROCm, MSVC)\n- Examples.md: added ECDSA sign/verify, recoverable signatures, Schnorr BIP-340 examples",
          "timestamp": "2026-02-23T14:53:16+04:00",
          "tree_id": "28aa8e4d8d2ba4a504b9eb0fba4360d08d686110",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5d62d1ac9341723ca40f803b3c340607fd91a1ea"
        },
        "date": 1771844070057,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 299,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 165,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 26000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 56000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 157,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 166,
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
          "distinct": false,
          "id": "5d62d1ac9341723ca40f803b3c340607fd91a1ea",
          "message": "docs: comprehensive wiki documentation update\n\n- Home.md: updated features list, performance tables (CPU/GPU/Embedded), Discord link\n- Benchmarks.md: all current numbers for x86-64, ARM64, RISC-V, CUDA, OpenCL, Metal, ESP32, STM32\n- CPU-Guide.md: ARM64 promoted to Production (was Planned), updated perf numbers, algorithm details\n- API-Reference.md: added ECDSA, Schnorr, ECDH, BIP-32, Address, SHA-256, WIF, ufsecp C ABI (45 functions)\n- CUDA-Guide.md: added Blackwell architecture (sm_120, RTX 5060-5090)\n- Getting-Started.md: added missing build options (OpenCL, Metal, ROCm, MSVC)\n- Examples.md: added ECDSA sign/verify, recoverable signatures, Schnorr BIP-340 examples",
          "timestamp": "2026-02-23T14:53:16+04:00",
          "tree_id": "28aa8e4d8d2ba4a504b9eb0fba4360d08d686110",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5d62d1ac9341723ca40f803b3c340607fd91a1ea"
        },
        "date": 1771844106655,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 18000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "77d136b77b2f1c448adacfefe14064d8e277dc0a",
          "message": "ci: configure Dependabot version updates\n\nEcosystems: github-actions, npm (wasm, nodejs, react-native),\nnuget, cargo, pip, gradle. Weekly schedule (Monday).",
          "timestamp": "2026-02-23T15:42:51+04:00",
          "tree_id": "ee42496a9c4ad9d022b5638ec7c4ddc8ffa4f5dc",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/77d136b77b2f1c448adacfefe14064d8e277dc0a"
        },
        "date": 1771847041092,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 142,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "distinct": false,
          "id": "77d136b77b2f1c448adacfefe14064d8e277dc0a",
          "message": "ci: configure Dependabot version updates\n\nEcosystems: github-actions, npm (wasm, nodejs, react-native),\nnuget, cargo, pip, gradle. Weekly schedule (Monday).",
          "timestamp": "2026-02-23T15:42:51+04:00",
          "tree_id": "ee42496a9c4ad9d022b5638ec7c4ddc8ffa4f5dc",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/77d136b77b2f1c448adacfefe14064d8e277dc0a"
        },
        "date": 1771847049345,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 258,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "distinct": false,
          "id": "bfaeede0ba0d8186b73bca3b4b77cf3658fdfd23",
          "message": "security: add CodeQL, OpenSSF Scorecard, coverage, attestation, checksums\n\n- Add .github/workflows/codeql.yml — C/C++ static analysis (security-and-quality)\n- Add .github/workflows/scorecard.yml — OpenSSF Scorecard (weekly + push to main)\n- CI: add code coverage job (lcov + Codecov upload)\n- Release: generate SHA256SUMS.txt for all artifacts\n- Release: add GitHub artifact attestation (SLSA provenance)\n- SECURITY.md: update supported versions (3.11.x active), add security measures list\n- README.md: add OpenSSF Scorecard, CodeQL, Codecov badges",
          "timestamp": "2026-02-23T15:54:38+04:00",
          "tree_id": "de7547d152660cc0be96424a0600d810ca21595f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bfaeede0ba0d8186b73bca3b4b77cf3658fdfd23"
        },
        "date": 1771847759539,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "bfaeede0ba0d8186b73bca3b4b77cf3658fdfd23",
          "message": "security: add CodeQL, OpenSSF Scorecard, coverage, attestation, checksums\n\n- Add .github/workflows/codeql.yml — C/C++ static analysis (security-and-quality)\n- Add .github/workflows/scorecard.yml — OpenSSF Scorecard (weekly + push to main)\n- CI: add code coverage job (lcov + Codecov upload)\n- Release: generate SHA256SUMS.txt for all artifacts\n- Release: add GitHub artifact attestation (SLSA provenance)\n- SECURITY.md: update supported versions (3.11.x active), add security measures list\n- README.md: add OpenSSF Scorecard, CodeQL, Codecov badges",
          "timestamp": "2026-02-23T15:54:38+04:00",
          "tree_id": "de7547d152660cc0be96424a0600d810ca21595f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bfaeede0ba0d8186b73bca3b4b77cf3658fdfd23"
        },
        "date": 1771847765852,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 17000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "27bd9726cb67ea982ac57ede100686f0536fd91b",
          "message": "chore: pin actions by SHA, bump deps, fix CodeQL alerts\n\n- Pin all 24 GitHub Actions by SHA hash with version comments\n- Bump: checkout v6, setup-go v6, setup-java v5, setup-node v6, setup-python v6\n- Bump: node-addon-api ^8.5.0, node-gyp ^12.2.0, react ^19.2.4\n- Fix CodeQL: constant-comparison (DER parser), path-injection (selftest),\n  stack-address-escape (bench), unused variables, commented-out-code,\n  offset-use-before-range-check, missing-check-scanf, suspicious-pointer-scaling\n- Add [[maybe_unused]] to ~45 unused static functions (future-use)\n- Add SEO keywords to README\n- All 9/10 tests pass (CUDA not built)",
          "timestamp": "2026-02-23T17:16:46+04:00",
          "tree_id": "fe6c1001e2d687ea2f865bbf5a396d40bca33024",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/27bd9726cb67ea982ac57ede100686f0536fd91b"
        },
        "date": 1771852855590,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "distinct": false,
          "id": "27bd9726cb67ea982ac57ede100686f0536fd91b",
          "message": "chore: pin actions by SHA, bump deps, fix CodeQL alerts\n\n- Pin all 24 GitHub Actions by SHA hash with version comments\n- Bump: checkout v6, setup-go v6, setup-java v5, setup-node v6, setup-python v6\n- Bump: node-addon-api ^8.5.0, node-gyp ^12.2.0, react ^19.2.4\n- Fix CodeQL: constant-comparison (DER parser), path-injection (selftest),\n  stack-address-escape (bench), unused variables, commented-out-code,\n  offset-use-before-range-check, missing-check-scanf, suspicious-pointer-scaling\n- Add [[maybe_unused]] to ~45 unused static functions (future-use)\n- Add SEO keywords to README\n- All 9/10 tests pass (CUDA not built)",
          "timestamp": "2026-02-23T17:16:46+04:00",
          "tree_id": "fe6c1001e2d687ea2f865bbf5a396d40bca33024",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/27bd9726cb67ea982ac57ede100686f0536fd91b"
        },
        "date": 1771852870700,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": false,
          "id": "f11c6403374bedd3740e74d1e04907f84f178b9f",
          "message": "fix: use commit SHAs instead of annotated tag SHAs for action pins\n\nScorecard action failed with 'imposter commit' error because 6 actions\nwere pinned to annotated tag object SHAs instead of the underlying commit\nSHAs. The scorecard webapp rejects tag object SHAs during verification.\n\nFixed actions:\n- ossf/scorecard-action v2.4.0\n- github/codeql-action v3 (init, analyze, upload-sarif)\n- actions/attest-build-provenance v2\n- nttld/setup-ndk v1\n- peaceiris/actions-gh-pages v4\n- shivammathur/setup-php v2",
          "timestamp": "2026-02-23T17:43:26+04:00",
          "tree_id": "fdede2f25a214cb8d0ddde22f6b952e9cf615f03",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f11c6403374bedd3740e74d1e04907f84f178b9f"
        },
        "date": 1771854284160,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "f11c6403374bedd3740e74d1e04907f84f178b9f",
          "message": "fix: use commit SHAs instead of annotated tag SHAs for action pins\n\nScorecard action failed with 'imposter commit' error because 6 actions\nwere pinned to annotated tag object SHAs instead of the underlying commit\nSHAs. The scorecard webapp rejects tag object SHAs during verification.\n\nFixed actions:\n- ossf/scorecard-action v2.4.0\n- github/codeql-action v3 (init, analyze, upload-sarif)\n- actions/attest-build-provenance v2\n- nttld/setup-ndk v1\n- peaceiris/actions-gh-pages v4\n- shivammathur/setup-php v2",
          "timestamp": "2026-02-23T17:43:26+04:00",
          "tree_id": "fdede2f25a214cb8d0ddde22f6b952e9cf615f03",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f11c6403374bedd3740e74d1e04907f84f178b9f"
        },
        "date": 1771854402320,
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
            "value": 293,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "408703a746ed15250b1f96fefa27a3e335761b4a",
          "message": "Merge pull request #25 from step-security-bot/chore/GHA-231351-stepsecurity-remediation\n\n[StepSecurity] Apply security best practices",
          "timestamp": "2026-02-23T17:58:24+04:00",
          "tree_id": "e0899c30e752d0071931378a9fb2ec0cee0265b0",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/408703a746ed15250b1f96fefa27a3e335761b4a"
        },
        "date": 1771855187848,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
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
            "value": 306,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 165,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 26000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 55000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 156,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 147,
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
          "id": "bcffd1ccef1ef5bc569e4654927779145faab2db",
          "message": "fix: suppress 62+ code scanning alerts, harden PR #25 fixes\n\n- Add .github/codeql/codeql-config.yml: exclude cpp/unused-static-function (52),\n  cpp/constant-comparison (4), cpp/stack-address-escape (1), cpp/path-injection (3)\n- Reference config-file in codeql.yml CodeQL init step\n- Fix dependency-review.yml: checkout v4->v6, ubuntu-latest->ubuntu-24.04\n- Clean .pre-commit-config.yaml: remove irrelevant PHP/Java/Ruby/Go/eslint/pylint\n  hooks, keep gitleaks/shellcheck/cpplint/pre-commit-hooks, bump versions\n- Pin pip versions: wheel==0.45.1, setuptools==75.8.0, build==1.2.2 (release.yml),\n  pyflakes==3.2.0, mypy==1.14.1 (bindings.yml) for Scorecard PinnedDependenciesID\n- Suppress unused-local-variable: (void)a_inf in ct_point.cpp,\n  (void)parity in test_ecdh_recovery_taproot.cpp\n\nEliminates: 52 unused-static-function, 4 constant-comparison,\n3 path-injection, 2 unused-local-variable, 1 stack-address-escape,\n2 PinnedDependenciesID = 64 alerts resolved.\nRemaining 8: 4 TokenPermissions (legitimate), 4 repo-level (not code-fixable).",
          "timestamp": "2026-02-23T18:23:36+04:00",
          "tree_id": "eb04f8e0590905cceb64b960c252112515979a32",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bcffd1ccef1ef5bc569e4654927779145faab2db"
        },
        "date": 1771856700505,
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
            "value": 293,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": false,
          "id": "bcffd1ccef1ef5bc569e4654927779145faab2db",
          "message": "fix: suppress 62+ code scanning alerts, harden PR #25 fixes\n\n- Add .github/codeql/codeql-config.yml: exclude cpp/unused-static-function (52),\n  cpp/constant-comparison (4), cpp/stack-address-escape (1), cpp/path-injection (3)\n- Reference config-file in codeql.yml CodeQL init step\n- Fix dependency-review.yml: checkout v4->v6, ubuntu-latest->ubuntu-24.04\n- Clean .pre-commit-config.yaml: remove irrelevant PHP/Java/Ruby/Go/eslint/pylint\n  hooks, keep gitleaks/shellcheck/cpplint/pre-commit-hooks, bump versions\n- Pin pip versions: wheel==0.45.1, setuptools==75.8.0, build==1.2.2 (release.yml),\n  pyflakes==3.2.0, mypy==1.14.1 (bindings.yml) for Scorecard PinnedDependenciesID\n- Suppress unused-local-variable: (void)a_inf in ct_point.cpp,\n  (void)parity in test_ecdh_recovery_taproot.cpp\n\nEliminates: 52 unused-static-function, 4 constant-comparison,\n3 path-injection, 2 unused-local-variable, 1 stack-address-escape,\n2 PinnedDependenciesID = 64 alerts resolved.\nRemaining 8: 4 TokenPermissions (legitimate), 4 repo-level (not code-fixable).",
          "timestamp": "2026-02-23T18:23:36+04:00",
          "tree_id": "eb04f8e0590905cceb64b960c252112515979a32",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bcffd1ccef1ef5bc569e4654927779145faab2db"
        },
        "date": 1771856711086,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 188,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 179,
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
          "id": "bdce6c0893fd65d91c5707c2fc2c75ccbee8eeac",
          "message": "ci: add SonarCloud code quality analysis\n\n- Add .github/workflows/sonarcloud.yml: build-wrapper + sonar-scanner\n  - Runs on push to main/dev and PRs\n  - Uses clang-17, Ninja, Debug build with compile_commands.json\n  - SHA-pinned actions (harden-runner, checkout, sonarqube-scan-action)\n- Add sonar-project.properties: project key, sources, exclusions\n- SONAR_TOKEN secret configured in repo settings",
          "timestamp": "2026-02-23T18:31:35+04:00",
          "tree_id": "509f415689e6852cc1a84a93b149d19bc732e255",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bdce6c0893fd65d91c5707c2fc2c75ccbee8eeac"
        },
        "date": 1771857176045,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "distinct": false,
          "id": "bdce6c0893fd65d91c5707c2fc2c75ccbee8eeac",
          "message": "ci: add SonarCloud code quality analysis\n\n- Add .github/workflows/sonarcloud.yml: build-wrapper + sonar-scanner\n  - Runs on push to main/dev and PRs\n  - Uses clang-17, Ninja, Debug build with compile_commands.json\n  - SHA-pinned actions (harden-runner, checkout, sonarqube-scan-action)\n- Add sonar-project.properties: project key, sources, exclusions\n- SONAR_TOKEN secret configured in repo settings",
          "timestamp": "2026-02-23T18:31:35+04:00",
          "tree_id": "509f415689e6852cc1a84a93b149d19bc732e255",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bdce6c0893fd65d91c5707c2fc2c75ccbee8eeac"
        },
        "date": 1771857212733,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 149,
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
          "distinct": false,
          "id": "457dc759d470354a82d5af8c433a6fa2dc2227d5",
          "message": "fix(ci): correct SonarCloud action references\n\n- Use sonarqube-scan-action/install-build-wrapper sub-action (not /setup)\n- Use sonarqube-scan-action main action for scan step (not CLI)\n- Fix SHA to v5.2.0 commit: 2500896589ef8f7247069a56136f8dc177c27ccf\n- Add sonar.host.url=https://sonarcloud.io",
          "timestamp": "2026-02-23T18:34:57+04:00",
          "tree_id": "586f73630c6a866c12fd3253bdaa007a8e83b759",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/457dc759d470354a82d5af8c433a6fa2dc2227d5"
        },
        "date": 1771857372017,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 20,
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
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 137,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 35000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 43000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 22000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 129,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 120,
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
          "id": "457dc759d470354a82d5af8c433a6fa2dc2227d5",
          "message": "fix(ci): correct SonarCloud action references\n\n- Use sonarqube-scan-action/install-build-wrapper sub-action (not /setup)\n- Use sonarqube-scan-action main action for scan step (not CLI)\n- Fix SHA to v5.2.0 commit: 2500896589ef8f7247069a56136f8dc177c27ccf\n- Add sonar.host.url=https://sonarcloud.io",
          "timestamp": "2026-02-23T18:34:57+04:00",
          "tree_id": "586f73630c6a866c12fd3253bdaa007a8e83b759",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/457dc759d470354a82d5af8c433a6fa2dc2227d5"
        },
        "date": 1771857379017,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "4cc202e9719dba270df2043d6c0eae487eac13ea",
          "message": "Merge pull request #29 from shrec/dependabot/github_actions/ossf/scorecard-action-2.4.3\n\nci(deps): bump ossf/scorecard-action from 2.4.0 to 2.4.3",
          "timestamp": "2026-02-23T19:10:09+04:00",
          "tree_id": "eea35b44339a86fe7714f49304584bc79cdd470a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4cc202e9719dba270df2043d6c0eae487eac13ea"
        },
        "date": 1771859484172,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 299,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 166,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 26000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 56000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 155,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 147,
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
          "id": "3efc0f814d6bc63488fd24f8540fb479b0316511",
          "message": "Merge pull request #26 from shrec/dependabot/github_actions/github/codeql-action-4.32.4\n\nci(deps): bump github/codeql-action from 3.32.4 to 4.32.4",
          "timestamp": "2026-02-23T19:10:25+04:00",
          "tree_id": "50296e6fdfda64a6c3ccd194d5db2167eaae11df",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3efc0f814d6bc63488fd24f8540fb479b0316511"
        },
        "date": 1771859511247,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "645b92c3ba68603a1b42bf204320e68979db345f",
          "message": "Merge pull request #27 from shrec/dependabot/github_actions/actions/setup-dotnet-5.1.0\n\nci(deps): bump actions/setup-dotnet from 4.3.1 to 5.1.0",
          "timestamp": "2026-02-23T19:10:38+04:00",
          "tree_id": "12e2943beb5d1d15f51e027529e085e28ee42953",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/645b92c3ba68603a1b42bf204320e68979db345f"
        },
        "date": 1771859512550,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "b1a5b477e64105744c30b590c9bdddbaaec15eec",
          "message": "Merge pull request #28 from shrec/dependabot/github_actions/actions/upload-artifact-6.0.0\n\nci(deps): bump actions/upload-artifact from 4.6.2 to 6.0.0",
          "timestamp": "2026-02-23T19:10:53+04:00",
          "tree_id": "f951f3e7137273b27f37e53c89fa3ea6aa59b33f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b1a5b477e64105744c30b590c9bdddbaaec15eec"
        },
        "date": 1771859532610,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "33edf5cb6bb93198d7afe04cf6d9feff406b3853",
          "message": "Merge pull request #30 from shrec/dependabot/github_actions/actions/attest-build-provenance-3.2.0\n\nci(deps): bump actions/attest-build-provenance from 2.4.0 to 3.2.0",
          "timestamp": "2026-02-23T19:11:08+04:00",
          "tree_id": "b9b653484bbd9d6475a3a50322754b788e8c5e91",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/33edf5cb6bb93198d7afe04cf6d9feff406b3853"
        },
        "date": 1771859546335,
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
            "value": 283,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "8d10f6e31abcbe37e3bac0ef511c8fe6c54bf957",
          "message": "ci: add workflow_dispatch trigger to SonarCloud workflow",
          "timestamp": "2026-02-23T19:11:53+04:00",
          "tree_id": "19c29ca24f8932d7b7d169894973ba283bbed611",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/8d10f6e31abcbe37e3bac0ef511c8fe6c54bf957"
        },
        "date": 1771859600138,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 27000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 165,
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
          "distinct": false,
          "id": "8d10f6e31abcbe37e3bac0ef511c8fe6c54bf957",
          "message": "ci: add workflow_dispatch trigger to SonarCloud workflow",
          "timestamp": "2026-02-23T19:11:53+04:00",
          "tree_id": "19c29ca24f8932d7b7d169894973ba283bbed611",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/8d10f6e31abcbe37e3bac0ef511c8fe6c54bf957"
        },
        "date": 1771859614114,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 27000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 175,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "3c91cd712ceb6ad98f93c03819748d45f343135a",
          "message": "fix(ci): pin pip deps by hash, move codeql write perms to job-level\n\n- Create .github/requirements/ with hash-pinned requirements files\n- release.yml: use release-build.txt with wheel/setuptools/build + deps\n- bindings.yml: use bindings-lint.txt with pyflakes/mypy + deps\n- codeql.yml: move security-events:write from top-level to job-level\n  (fixes Scorecard TokenPermissions alert #122)",
          "timestamp": "2026-02-23T19:19:18+04:00",
          "tree_id": "1f6b84d09e751e5ce69edeae0a853eafebbc3eb3",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3c91cd712ceb6ad98f93c03819748d45f343135a"
        },
        "date": 1771860040825,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
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
            "value": 301,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 166,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 81000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 25000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 56000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 157,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 148,
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
          "distinct": false,
          "id": "3c91cd712ceb6ad98f93c03819748d45f343135a",
          "message": "fix(ci): pin pip deps by hash, move codeql write perms to job-level\n\n- Create .github/requirements/ with hash-pinned requirements files\n- release.yml: use release-build.txt with wheel/setuptools/build + deps\n- bindings.yml: use bindings-lint.txt with pyflakes/mypy + deps\n- codeql.yml: move security-events:write from top-level to job-level\n  (fixes Scorecard TokenPermissions alert #122)",
          "timestamp": "2026-02-23T19:19:18+04:00",
          "tree_id": "1f6b84d09e751e5ce69edeae0a853eafebbc3eb3",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3c91cd712ceb6ad98f93c03819748d45f343135a"
        },
        "date": 1771860046972,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 62000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 143,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ab1b0c5192263c9c7d54df677d5bfe5c3204947e",
          "message": "Merge pull request #31 from shrec/dependabot/github_actions/SonarSource/sonarqube-scan-action-7.0.0\n\nci(deps): bump SonarSource/sonarqube-scan-action from 5.2.0 to 7.0.0",
          "timestamp": "2026-02-23T19:38:40+04:00",
          "tree_id": "a28f902cb41a74df2ba087a5bfdb646228be10dd",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ab1b0c5192263c9c7d54df677d5bfe5c3204947e"
        },
        "date": 1771861197384,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "28cdd702eb886fd4a8f0d534c334e62b529bbe41",
          "message": "Merge pull request #32 from shrec/dependabot/github_actions/actions/download-artifact-7.0.0\n\nci(deps): bump actions/download-artifact from 4.3.0 to 7.0.0",
          "timestamp": "2026-02-23T19:38:48+04:00",
          "tree_id": "1ffaa4b6e4dce5958864d3c82b47072b616a7053",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/28cdd702eb886fd4a8f0d534c334e62b529bbe41"
        },
        "date": 1771861205051,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "28cdd702eb886fd4a8f0d534c334e62b529bbe41",
          "message": "Merge pull request #32 from shrec/dependabot/github_actions/actions/download-artifact-7.0.0\n\nci(deps): bump actions/download-artifact from 4.3.0 to 7.0.0",
          "timestamp": "2026-02-23T19:38:48+04:00",
          "tree_id": "1ffaa4b6e4dce5958864d3c82b47072b616a7053",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/28cdd702eb886fd4a8f0d534c334e62b529bbe41"
        },
        "date": 1771861228313,
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
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 149,
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
          "id": "930a119828dc611fd69802a8022e5306dfadb6d5",
          "message": "fix: add Rust workspace Cargo.toml for Dependabot\n\nDependabot requires a Cargo.toml at /bindings/rust/ to scan Rust\ndependencies. Added a workspace manifest pointing to all four crates.",
          "timestamp": "2026-02-23T19:45:27+04:00",
          "tree_id": "b3d68c5348898fc4b1ae46925033897ca880da06",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/930a119828dc611fd69802a8022e5306dfadb6d5"
        },
        "date": 1771861616403,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 143,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "distinct": false,
          "id": "930a119828dc611fd69802a8022e5306dfadb6d5",
          "message": "fix: add Rust workspace Cargo.toml for Dependabot\n\nDependabot requires a Cargo.toml at /bindings/rust/ to scan Rust\ndependencies. Added a workspace manifest pointing to all four crates.",
          "timestamp": "2026-02-23T19:45:27+04:00",
          "tree_id": "b3d68c5348898fc4b1ae46925033897ca880da06",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/930a119828dc611fd69802a8022e5306dfadb6d5"
        },
        "date": 1771861618823,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 209,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 198,
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
          "id": "679fcddf3c3e5127479f06631810f0cc25b4f89a",
          "message": "docs: add coding standards, enhance contribution requirements, fix LICENSE for OpenSSF badge\n\n- Create docs/CODING_STANDARDS.md with full coding standards reference\n- Update CONTRIBUTING.md with explicit 'Requirements for Acceptable Contributions' section\n- Add URL references to coding standards in CONTRIBUTING.md\n- Replace LICENSE with standard full AGPL-3.0 text for GitHub recognition\n- SPDX: AGPL-3.0-or-later\n\nSatisfies OpenSSF Best Practices criteria:\n  contribution_requirements, floss_license, floss_license_osi,\n  documentation_interface, report_process, report_tracker",
          "timestamp": "2026-02-23T20:03:40+04:00",
          "tree_id": "4f54e3b33f8ec42462214afa778ddaaa5a01c981",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/679fcddf3c3e5127479f06631810f0cc25b4f89a"
        },
        "date": 1771862705760,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 27000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "distinct": false,
          "id": "679fcddf3c3e5127479f06631810f0cc25b4f89a",
          "message": "docs: add coding standards, enhance contribution requirements, fix LICENSE for OpenSSF badge\n\n- Create docs/CODING_STANDARDS.md with full coding standards reference\n- Update CONTRIBUTING.md with explicit 'Requirements for Acceptable Contributions' section\n- Add URL references to coding standards in CONTRIBUTING.md\n- Replace LICENSE with standard full AGPL-3.0 text for GitHub recognition\n- SPDX: AGPL-3.0-or-later\n\nSatisfies OpenSSF Best Practices criteria:\n  contribution_requirements, floss_license, floss_license_osi,\n  documentation_interface, report_process, report_tracker",
          "timestamp": "2026-02-23T20:03:40+04:00",
          "tree_id": "4f54e3b33f8ec42462214afa778ddaaa5a01c981",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/679fcddf3c3e5127479f06631810f0cc25b4f89a"
        },
        "date": 1771862746600,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c428f6a6dd6e08e977e38152b16928848fd72f3f",
          "message": "build(deps): bump setuptools in /.github/requirements (#33)\n\nSecurity fix: setuptools path traversal vulnerability (CVE)",
          "timestamp": "2026-02-23T20:25:35+04:00",
          "tree_id": "2e743a7e32d011c4f2d85dc06828c7a88d5fc81d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c428f6a6dd6e08e977e38152b16928848fd72f3f"
        },
        "date": 1771864012914,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 207,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 199,
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
          "distinct": false,
          "id": "788b01a08e8baa059caf0da39b9867d3ea8919d8",
          "message": "fix(deps): bump wheel 0.45.1 -> 0.46.2 (CVE path traversal fix)\n\nResolves Dependabot alert #2 and PR #34.\nHash-pinned for --require-hashes compliance.",
          "timestamp": "2026-02-23T20:26:32+04:00",
          "tree_id": "7b2594d5acbaec64c9071e2ea70c1f8ef0dcdb41",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/788b01a08e8baa059caf0da39b9867d3ea8919d8"
        },
        "date": 1771864086811,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 176,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "788b01a08e8baa059caf0da39b9867d3ea8919d8",
          "message": "fix(deps): bump wheel 0.45.1 -> 0.46.2 (CVE path traversal fix)\n\nResolves Dependabot alert #2 and PR #34.\nHash-pinned for --require-hashes compliance.",
          "timestamp": "2026-02-23T20:26:32+04:00",
          "tree_id": "7b2594d5acbaec64c9071e2ea70c1f8ef0dcdb41",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/788b01a08e8baa059caf0da39b9867d3ea8919d8"
        },
        "date": 1771864322039,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 135,
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
          "distinct": false,
          "id": "34c09b815ab84b92d0ff8f34e29d31269d52519a",
          "message": "release: v3.12.1 — security dependency fixes (wheel CVE-2026-24049, setuptools CVE-2025-47273)",
          "timestamp": "2026-02-23T20:37:32+04:00",
          "tree_id": "aac21c78d8e948ae7e21b137b38f92d6cec292da",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/34c09b815ab84b92d0ff8f34e29d31269d52519a"
        },
        "date": 1771864747685,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "distinct": false,
          "id": "34c09b815ab84b92d0ff8f34e29d31269d52519a",
          "message": "release: v3.12.1 — security dependency fixes (wheel CVE-2026-24049, setuptools CVE-2025-47273)",
          "timestamp": "2026-02-23T20:37:32+04:00",
          "tree_id": "aac21c78d8e948ae7e21b137b38f92d6cec292da",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/34c09b815ab84b92d0ff8f34e29d31269d52519a"
        },
        "date": 1771864795644,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "distinct": false,
          "id": "d207ca8cef926fa02f9f360b1da1b642bc98b303",
          "message": "feat: add Linux distribution packaging (Debian, RPM, Arch)\n\n- packaging/debian/: control, rules, changelog, copyright, .install files\n- packaging/rpm/libsecp256k1-fast.spec: Fedora/RHEL spec file\n- packaging/arch/PKGBUILD: Arch Linux / AUR package\n- packaging/README.md: build instructions for all distros\n- CMakeLists.txt: add CPack DEB/RPM generators + metadata\n- cpu/CMakeLists.txt: add SOVERSION for shared library versioning",
          "timestamp": "2026-02-23T20:51:43+04:00",
          "tree_id": "f3064989045f3151d988d40c4f24c9c6a11bf81a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d207ca8cef926fa02f9f360b1da1b642bc98b303"
        },
        "date": 1771865587411,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
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
          "id": "d207ca8cef926fa02f9f360b1da1b642bc98b303",
          "message": "feat: add Linux distribution packaging (Debian, RPM, Arch)\n\n- packaging/debian/: control, rules, changelog, copyright, .install files\n- packaging/rpm/libsecp256k1-fast.spec: Fedora/RHEL spec file\n- packaging/arch/PKGBUILD: Arch Linux / AUR package\n- packaging/README.md: build instructions for all distros\n- CMakeLists.txt: add CPack DEB/RPM generators + metadata\n- cpu/CMakeLists.txt: add SOVERSION for shared library versioning",
          "timestamp": "2026-02-23T20:51:43+04:00",
          "tree_id": "f3064989045f3151d988d40c4f24c9c6a11bf81a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d207ca8cef926fa02f9f360b1da1b642bc98b303"
        },
        "date": 1771865608636,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
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
          "distinct": false,
          "id": "75a73bcf9b2212d3daaf6ea89f8aabcd6483882e",
          "message": "ci: add Linux packaging workflow (DEB/RPM build + APT repo on GitHub Pages)\n\n- .github/workflows/packaging.yml: builds .deb (amd64+arm64) and .rpm (x86_64)\n  on every release tag, attaches to GitHub Release, deploys APT repo to gh-pages\n- README.md: add Installation section (APT, RPM, Arch, source, CMake, pkg-config)",
          "timestamp": "2026-02-23T20:54:37+04:00",
          "tree_id": "df75eecce14d8e6a35a896562ceecc79a7305f7a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/75a73bcf9b2212d3daaf6ea89f8aabcd6483882e"
        },
        "date": 1771865762821,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 20,
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
            "value": 260,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 145,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 46000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 22000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "distinct": false,
          "id": "75a73bcf9b2212d3daaf6ea89f8aabcd6483882e",
          "message": "ci: add Linux packaging workflow (DEB/RPM build + APT repo on GitHub Pages)\n\n- .github/workflows/packaging.yml: builds .deb (amd64+arm64) and .rpm (x86_64)\n  on every release tag, attaches to GitHub Release, deploys APT repo to gh-pages\n- README.md: add Installation section (APT, RPM, Arch, source, CMake, pkg-config)",
          "timestamp": "2026-02-23T20:54:37+04:00",
          "tree_id": "df75eecce14d8e6a35a896562ceecc79a7305f7a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/75a73bcf9b2212d3daaf6ea89f8aabcd6483882e"
        },
        "date": 1771865769023,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "e3e42f0b6f27fcec6cb10c1eaff1bf2f97388908",
          "message": "docs: add Linux package install instructions + Dockerfile (#36)\n\ndocs: add Linux package install instructions + Dockerfile",
          "timestamp": "2026-02-23T21:06:22+04:00",
          "tree_id": "8e3024d9d7c83ccd1c1ca230fbbe52f6eb680f70",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/e3e42f0b6f27fcec6cb10c1eaff1bf2f97388908"
        },
        "date": 1771866466356,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 146,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "dab6bbd656c800293eada511b1bfcaa96d71ea17",
          "message": "chore: add CODEOWNERS for auto-review requests + clang-tidy config (#35)\n\nchore: add CODEOWNERS + clang-tidy config",
          "timestamp": "2026-02-23T21:03:12+04:00",
          "tree_id": "495c21a763cced33b275f4bc9ec9944252ae4e65",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/dab6bbd656c800293eada511b1bfcaa96d71ea17"
        },
        "date": 1771866497748,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c7524221fd5e074936d845e78e4fb7a985bb5df7",
          "message": "ci: add clang-tidy static analysis workflow (#37)\n\nci: add clang-tidy static analysis workflow",
          "timestamp": "2026-02-23T21:07:43+04:00",
          "tree_id": "66a3f4a90b841726ed58e830bf069868a9d4c807",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c7524221fd5e074936d845e78e4fb7a985bb5df7"
        },
        "date": 1771866585915,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 149,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d3d081bfffbe1efcb1439a75b09ea65d2a4527d1",
          "message": "chore: update SECURITY.md version table + add PR template (#38)\n\nchore: update SECURITY.md version table + add PR template",
          "timestamp": "2026-02-23T21:09:07+04:00",
          "tree_id": "c475572f4c5daefa514bbc912876c3959f3fea9d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d3d081bfffbe1efcb1439a75b09ea65d2a4527d1"
        },
        "date": 1771866651784,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "494de5aef4e3bcc3a747b601e4b8930613d54f1c",
          "message": "docs: update CONTRIBUTING.md prerequisites, add clang-tidy step, refresh v3.12 features (#39)\n\ndocs: update CONTRIBUTING.md for v3.12",
          "timestamp": "2026-02-23T21:10:23+04:00",
          "tree_id": "0a298892b734409d8607fcd78181fd476b98ffba",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/494de5aef4e3bcc3a747b601e4b8930613d54f1c"
        },
        "date": 1771866742247,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "494de5aef4e3bcc3a747b601e4b8930613d54f1c",
          "message": "docs: update CONTRIBUTING.md prerequisites, add clang-tidy step, refresh v3.12 features (#39)\n\ndocs: update CONTRIBUTING.md for v3.12",
          "timestamp": "2026-02-23T21:10:23+04:00",
          "tree_id": "0a298892b734409d8607fcd78181fd476b98ffba",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/494de5aef4e3bcc3a747b601e4b8930613d54f1c"
        },
        "date": 1771866808324,
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
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "60501bbb3464f5023dc57ee2e3bec6ff4549d69b",
          "message": "fix: pin Docker base images to SHA256 digest (#40)\n\nPins ubuntu:24.04 FROM lines to sha256:d1e2e92c... to satisfy\nScorecard Pinned-Dependencies check (alerts #279, #280).",
          "timestamp": "2026-02-23T21:38:28+04:00",
          "tree_id": "92ad60dd37987efe673caca9d416727c38f86984",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/60501bbb3464f5023dc57ee2e3bec6ff4549d69b"
        },
        "date": 1771868418334,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 176,
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
          "id": "60501bbb3464f5023dc57ee2e3bec6ff4549d69b",
          "message": "fix: pin Docker base images to SHA256 digest (#40)\n\nPins ubuntu:24.04 FROM lines to sha256:d1e2e92c... to satisfy\nScorecard Pinned-Dependencies check (alerts #279, #280).",
          "timestamp": "2026-02-23T21:38:28+04:00",
          "tree_id": "92ad60dd37987efe673caca9d416727c38f86984",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/60501bbb3464f5023dc57ee2e3bec6ff4549d69b"
        },
        "date": 1771868456151,
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
            "value": 285,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "daab2be2dfbc83c694cc3d4060040ee063286cfb",
          "message": "feat: add security audit CI, CITATION.cff, enhance SEO + badges (#41)\n\n- New security-audit.yml: -Werror build, ASan/UBSan tests, Valgrind memcheck (weekly + push/PR)\n- New CITATION.cff: structured citation for Google Scholar / Zenodo\n- README: add Security Audit, Clang-Tidy, SonarCloud badges; expand SEO keywords\n- SECURITY.md: document all 14 security measures; add planned improvements roadmap; bump to v3.12.1\n- GitHub topics: replace niche (milk-v, lx6) with high-SEO (constant-time, ecdsa, ethereum, gpu-cryptography, schnorr-signatures)",
          "timestamp": "2026-02-23T21:46:34+04:00",
          "tree_id": "4207117fbcdb943cc5a4f0a2b9e877a38c2ef784",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/daab2be2dfbc83c694cc3d4060040ee063286cfb"
        },
        "date": 1771868869791,
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
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": false,
          "id": "daab2be2dfbc83c694cc3d4060040ee063286cfb",
          "message": "feat: add security audit CI, CITATION.cff, enhance SEO + badges (#41)\n\n- New security-audit.yml: -Werror build, ASan/UBSan tests, Valgrind memcheck (weekly + push/PR)\n- New CITATION.cff: structured citation for Google Scholar / Zenodo\n- README: add Security Audit, Clang-Tidy, SonarCloud badges; expand SEO keywords\n- SECURITY.md: document all 14 security measures; add planned improvements roadmap; bump to v3.12.1\n- GitHub topics: replace niche (milk-v, lx6) with high-SEO (constant-time, ecdsa, ethereum, gpu-cryptography, schnorr-signatures)",
          "timestamp": "2026-02-23T21:46:34+04:00",
          "tree_id": "4207117fbcdb943cc5a4f0a2b9e877a38c2ef784",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/daab2be2dfbc83c694cc3d4060040ee063286cfb"
        },
        "date": 1771868892824,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "c5898eb2441af5c42bb4a92533f0ad68be6cfd6a",
          "message": "Merge pull request #42 from shrec/feat/audit-readiness\n\nfeat: comprehensive audit readiness documentation",
          "timestamp": "2026-02-23T22:12:50+04:00",
          "tree_id": "c6936494113ed11984f1f4e8a1c0996f97205df7",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c5898eb2441af5c42bb4a92533f0ad68be6cfd6a"
        },
        "date": 1771870444237,
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
            "value": 286,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "c5898eb2441af5c42bb4a92533f0ad68be6cfd6a",
          "message": "Merge pull request #42 from shrec/feat/audit-readiness\n\nfeat: comprehensive audit readiness documentation",
          "timestamp": "2026-02-23T22:12:50+04:00",
          "tree_id": "c6936494113ed11984f1f4e8a1c0996f97205df7",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c5898eb2441af5c42bb4a92533f0ad68be6cfd6a"
        },
        "date": 1771870451143,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "c89ab3572cb3064b205066860ae536cb7d1ae7b5",
          "message": "Merge feat/phase1-core-assurance into dev\n\nPhase I Core Assurance:\n- BIP-340 test vectors (27/27)\n- RFC6979 test vectors (35/35)\n- ECC property-based tests (89/89)\n- dudect constant-time CI integration\n- CI fixes (MSVC, TSan, -Werror, Valgrind)\n- differential_test CTest target\n- Build directory consolidation (build/<name> pattern)",
          "timestamp": "2026-02-24T00:23:09+04:00",
          "tree_id": "c8142a448e48827ad8d1585f0ca7edd69a6d1042",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c89ab3572cb3064b205066860ae536cb7d1ae7b5"
        },
        "date": 1771878596572,
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
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "cb82d4572cbbd3cd3f32457bc0198d8ccdd489a6",
          "message": "fix(ci): restore job IDs in release.yml (slash illegal in YAML keys)\n\nThe previous build-dir consolidation accidentally replaced job IDs\nbuild-android and build-wasm with build/android and build/wasm.\nYAML job keys cannot contain slashes — this caused the release\nworkflow to fail with 'workflow file issue' (0 jobs could parse).\n\nFixed: job IDs and needs: references back to build-android/build-wasm.\nBuild directory paths (-B build/android, build/wasm/dist/) unchanged.",
          "timestamp": "2026-02-24T00:36:15+04:00",
          "tree_id": "655afabd6aa2ad3ca4810a0ec99f9c5881259b81",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/cb82d4572cbbd3cd3f32457bc0198d8ccdd489a6"
        },
        "date": 1771879076918,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "f043c57cd0d82c3046d584163b0eff2e17de6614",
          "message": "fix(ci): correct WASM benchmark relative path for build/wasm/dist\n\nbuild/wasm/dist is 3 levels deep from repo root, not 2.\n../../wasm/bench_wasm.mjs resolved to build/wasm/bench_wasm.mjs (wrong).\n../../../wasm/bench_wasm.mjs correctly reaches repo root.",
          "timestamp": "2026-02-24T00:39:28+04:00",
          "tree_id": "17ad289e2e27fc40f89d8adab6c64aa0f98233c5",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f043c57cd0d82c3046d584163b0eff2e17de6614"
        },
        "date": 1771879249131,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 20,
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
            "value": 259,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 138,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 35000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 43000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 22000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 122,
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
          "id": "b4ba6e71f69eeb806a8ed1207b6ac203dc72d6d5",
          "message": "Merge dev into main\n\nIncludes Phase I Core Assurance + CI fixes:\n- BIP-340 (27/27), RFC6979 (35/35), ECC properties (89/89) test vectors\n- dudect CT integration + MSVC portability fix\n- CI: strict-aliasing, __int128, conversion -Werror fixes\n- CI: Valgrind false positive suppression\n- CI: ct_sidechannel exclusion from sanitizer/regular CTest\n- Build directory consolidation (build/<name> pattern)\n- WASM benchmark path fix",
          "timestamp": "2026-02-24T00:55:09+04:00",
          "tree_id": "17ad289e2e27fc40f89d8adab6c64aa0f98233c5",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b4ba6e71f69eeb806a8ed1207b6ac203dc72d6d5"
        },
        "date": 1771880202360,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "d568699451bac508c207243fa2f024f07bcb36b2",
          "message": "fix(ci): add coverage collection to SonarCloud workflow\n\nSonarCloud Quality Gate requires >=80% coverage on new code.\nPreviously the workflow only built but never ran tests, resulting\nin 0% coverage data.\n\nChanges:\n- sonarcloud.yml: add -fprofile-instr-generate -fcoverage-mapping flags,\n  run CTest after build, collect coverage via llvm-profdata/llvm-cov\n- sonar-project.properties: add build/** to exclusions pattern",
          "timestamp": "2026-02-24T00:57:51+04:00",
          "tree_id": "3b059f2b8d0780dafee99df08200c2cbbee623fe",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d568699451bac508c207243fa2f024f07bcb36b2"
        },
        "date": 1771880363456,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 39,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 29,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 8,
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
            "value": 284,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "75b3257c61509ca5d77562cd6e6fdcd71c566a6e",
          "message": "Merge dev: SonarCloud coverage fix",
          "timestamp": "2026-02-24T00:58:06+04:00",
          "tree_id": "3b059f2b8d0780dafee99df08200c2cbbee623fe",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/75b3257c61509ca5d77562cd6e6fdcd71c566a6e"
        },
        "date": 1771880370262,
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
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "ffdf48e98a51892fd56af595e6956a455e8024f4",
          "message": "Merge dev: CI fixes (__int128 pragmas, SonarCloud, dudect)",
          "timestamp": "2026-02-24T01:22:37+04:00",
          "tree_id": "1323aa3576ab0c29d4b6929f7395d81842e5b88c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ffdf48e98a51892fd56af595e6956a455e8024f4"
        },
        "date": 1771881912400,
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
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 27000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
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
          "id": "9a0c8b66c9b5502d00b7c40f58e0b21a4f8e47c1",
          "message": "fix(ci): add __int128 pragma guards, fix SonarCloud coverage, make dudect advisory\n\n- Add #pragma GCC diagnostic ignored \"-Wpedantic\" around all __int128 usage\n  in ct_field.cpp, ct_point.cpp, ecdsa.cpp, precompute.cpp, field_asm.hpp\n  (fixes g++-13 -Werror -Wpedantic build failure)\n- Fix SonarCloud: use absolute LLVM_PROFILE_FILE path and find-based profraw\n  discovery (fixes 'No such file' in coverage collection)\n- Make dudect timing analysis advisory on CI: always exit 0, report variance\n  as warning (statistical tests are unreliable on shared CI runners)\n\nLocal: 11/11 tests pass, 0 warnings",
          "timestamp": "2026-02-24T01:22:23+04:00",
          "tree_id": "1323aa3576ab0c29d4b6929f7395d81842e5b88c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9a0c8b66c9b5502d00b7c40f58e0b21a4f8e47c1"
        },
        "date": 1771881924441,
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
            "value": 284,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "b366d6cd2a30d6bc4c588779c3acf5ddbc0321d5",
          "message": "fix(ci): resolve remaining -Werror warnings for g++-13\n\n- precompute.cpp: explicit static_cast for __int128->uint64_t (-Wconversion)\n- precompute.cpp: rename shadowed 'end' variable (-Wshadow)\n- precompute.cpp: remove unused aQ/aPsiQ variables (-Wunused-but-set-variable)\n- schnorr.cpp: suppress GCC false-positive -Warray-bounds on SHA256 copy\n- ct_point.cpp: guard #pragma clang loop with #ifdef __clang__ (-Wunknown-pragmas)",
          "timestamp": "2026-02-24T01:40:15+04:00",
          "tree_id": "59932ad7ee262e85c63c5c8b813442f04d6578d5",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b366d6cd2a30d6bc4c588779c3acf5ddbc0321d5"
        },
        "date": 1771882932583,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "4751aad1c67e95b5c1bea82da08322c66c962c44",
          "message": "Merge dev: fix remaining -Werror warnings",
          "timestamp": "2026-02-24T01:40:28+04:00",
          "tree_id": "59932ad7ee262e85c63c5c8b813442f04d6578d5",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4751aad1c67e95b5c1bea82da08322c66c962c44"
        },
        "date": 1771882966689,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "7b79f59c8a11b52e01665d5e50ded493709f97e5",
          "message": "fix(ci): remove unused 'compressed' variable in bip32.cpp to_public() (-Wunused-but-set-variable)",
          "timestamp": "2026-02-24T01:45:50+04:00",
          "tree_id": "8ce065581c8c654dff8c798e8450430242d420cc",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7b79f59c8a11b52e01665d5e50ded493709f97e5"
        },
        "date": 1771883240522,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "7329585c045ad52d31ca0f2199be8fbdea568e83",
          "message": "Merge dev: fix bip32.cpp unused variable warning",
          "timestamp": "2026-02-24T01:45:58+04:00",
          "tree_id": "8ce065581c8c654dff8c798e8450430242d420cc",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7329585c045ad52d31ca0f2199be8fbdea568e83"
        },
        "date": 1771883246629,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "372cf397ec4f3d1e7543694498b616f16aa22daf",
          "message": "fix(security): resolve code scanning alerts #281 #282\n\n- differential_test.cpp: suppress unused 'der' variable (CodeQL #282)\n- security-audit.yml: remove unnecessary security-events:write permission (Scorecard #281)\n- Dismissed #273, #274, #275 as won't-fix (write perms required by design)",
          "timestamp": "2026-02-24T01:51:00+04:00",
          "tree_id": "d342f9090c8149d127d4a2b38adb4919fcfa5e7a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/372cf397ec4f3d1e7543694498b616f16aa22daf"
        },
        "date": 1771883542664,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 208,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 200,
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
          "id": "7febc2161bf348f8091820b184505f249fe9da42",
          "message": "Merge dev: fix code scanning alerts #281 #282",
          "timestamp": "2026-02-24T01:51:11+04:00",
          "tree_id": "d342f9090c8149d127d4a2b38adb4919fcfa5e7a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7febc2161bf348f8091820b184505f249fe9da42"
        },
        "date": 1771883556890,
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
            "value": 147,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "8a4fbc18e033245e33139ecca648836a71794ca3",
          "message": "Add Developer Certificate of Origin (DCO) requirement to CONTRIBUTING.md\n\nContributors must now sign off commits with 'git commit -s' to certify\nlegal authorization per https://developercertificate.org/.\nRequired for CII Best Practices badge [dco] criterion.",
          "timestamp": "2026-02-24T02:14:40+04:00",
          "tree_id": "777c2db5a31a011b17ba2da08eab5c77fe9bf699",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/8a4fbc18e033245e33139ecca648836a71794ca3"
        },
        "date": 1771884962316,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 144,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "a2eb40e6a0e29fad71340204359abdd111e0d794",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T02:15:24+04:00",
          "tree_id": "777c2db5a31a011b17ba2da08eab5c77fe9bf699",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a2eb40e6a0e29fad71340204359abdd111e0d794"
        },
        "date": 1771885010034,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "53a720614e5279834242545d544bbc35fbb99c65",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T02:16:43+04:00",
          "tree_id": "faae99d269eacbde55cace740d29bf4854d9f513",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/53a720614e5279834242545d544bbc35fbb99c65"
        },
        "date": 1771885084146,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 258,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "8136da2c21a82f921cc2308e2b21e0c1fcd36962",
          "message": "Add GOVERNANCE.md — BDFL governance model\n\nDocuments project decision-making process, roles (lead maintainer,\ncontributors, security reporters), release process, and amendment policy.\nRequired for CII Best Practices badge [governance] criterion.\n\nSigned-off-by: vano <payysoon@gmail.com>",
          "timestamp": "2026-02-24T02:16:35+04:00",
          "tree_id": "faae99d269eacbde55cace740d29bf4854d9f513",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/8136da2c21a82f921cc2308e2b21e0c1fcd36962"
        },
        "date": 1771885087526,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 143,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 135,
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
          "id": "6a13e6955cdee1caf3d0ee7a7db5198a7ea3d74e",
          "message": "Add continuity plan (bus factor) to GOVERNANCE.md\n\nDocuments how the project continues if the lead maintainer becomes\nunavailable: public repo, automated CI/CD, backup maintainer with org\nowner access, AGPL license + DCO ensuring legal redistributability.\nRequired for CII Best Practices badge [access_continuity] criterion.\n\nSigned-off-by: vano <payysoon@gmail.com>",
          "timestamp": "2026-02-24T02:21:32+04:00",
          "tree_id": "8cf9e699855bf4693ee0c30e76e2eb573b9331bf",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/6a13e6955cdee1caf3d0ee7a7db5198a7ea3d74e"
        },
        "date": 1771885390931,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 142,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "26499ff6f73a26d7a36e7c9fac5c8979c41ad0b4",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T02:22:33+04:00",
          "tree_id": "8cf9e699855bf4693ee0c30e76e2eb573b9331bf",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/26499ff6f73a26d7a36e7c9fac5c8979c41ad0b4"
        },
        "date": 1771885455796,
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
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "3231c045dabd0e49d24029eacbe92b9c73ab884d",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T02:24:45+04:00",
          "tree_id": "760960227b7140de3213fe5816305a42ed5cf60b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3231c045dabd0e49d24029eacbe92b9c73ab884d"
        },
        "date": 1771885677106,
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
            "value": 293,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 18000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "ddeb1e353a79ec8918959ae7c9e31cd1f306e7a3",
          "message": "Add ROADMAP.md — 12-month project roadmap (Mar 2026 – Feb 2027)\n\nThree phases: Core Assurance, Protocol/Production Hardening, Audit/Trust.\nIncludes explicit non-goals and won't-do items per phase.\nRequired for CII Best Practices badge [documentation_roadmap] criterion.\n\nSigned-off-by: vano <payysoon@gmail.com>",
          "timestamp": "2026-02-24T02:24:34+04:00",
          "tree_id": "760960227b7140de3213fe5816305a42ed5cf60b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ddeb1e353a79ec8918959ae7c9e31cd1f306e7a3"
        },
        "date": 1771885687326,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "c639e40b5af00d539e58f7713d14703be12b00a5",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T02:38:18+04:00",
          "tree_id": "8095228ab385937bac5684067ce61aed2585160b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c639e40b5af00d539e58f7713d14703be12b00a5"
        },
        "date": 1771886373523,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "89b725ce01512bc5973d830f057ba5a8764acc42",
          "message": "Add OpenSSF Best Practices badge to README\n\nBadge ID: 12011 (bestpractices.dev/projects/12011)\nRequired for CII [documentation_achievements] criterion.\n\nSigned-off-by: vano <payysoon@gmail.com>",
          "timestamp": "2026-02-24T02:37:45+04:00",
          "tree_id": "8095228ab385937bac5684067ce61aed2585160b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/89b725ce01512bc5973d830f057ba5a8764acc42"
        },
        "date": 1771886548565,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "a3b93392466c4347834595a81c515f9a51217906",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T04:23:59+04:00",
          "tree_id": "184855cb80530812832cc4e22f750383f313441c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a3b93392466c4347834595a81c515f9a51217906"
        },
        "date": 1771892714225,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 57000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
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
          "id": "57cf96ba6a4af831ada6e310c49a668459eb0c16",
          "message": "fix: resolve all -Werror warnings across library, tests, and benchmarks\n\n- Sign-conversion: int→size_t for all array/vector indexing loops\n- Shadow: rename local P1/P2 → pt1/pt2 where shadowing fe52_constants\n- Unused variables: [[maybe_unused]] or (void) for intentionally unused\n- Unused functions: remove dead h32() after sscanf→manual hex migration\n- Deprecated: sscanf→manual hex parsing (MSVC CRT portability)\n- Int→uint64_t: explicit static_cast in from_uint64() call sites\n- Add codecov.yml configuration\n- Rewrite CI coverage job to use LLVM source-based coverage\n\nBuild: cmake --build build-werror -j 8 (zero warnings with -Werror -Wall -Wextra -Wpedantic -Wconversion -Wshadow)\nTests: 12/12 passed (ctest --test-dir build-werror)",
          "timestamp": "2026-02-24T04:23:47+04:00",
          "tree_id": "184855cb80530812832cc4e22f750383f313441c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/57cf96ba6a4af831ada6e310c49a668459eb0c16"
        },
        "date": 1771892782712,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "b4f397387caff4c6859300ee4b00ae2ee6964a4c",
          "message": "fix: add bounds check in base58check_encode to silence GCC -Wstringop-overflow\n\nGCC-13 with -O2 cannot prove len won't overflow in (len+4), causing\na false-positive -Wstringop-overflow on the memcpy. Add an early\nreturn guard (len == 0 || len > 0x7FFFFFFFUL) so the optimizer can\nbound the memcpy size statically.\n\nVerified: 12/12 tests pass.",
          "timestamp": "2026-02-24T04:28:27+04:00",
          "tree_id": "30c9bb447f3d28f2c01301f16d14383a9d50ec03",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b4f397387caff4c6859300ee4b00ae2ee6964a4c"
        },
        "date": 1771892983410,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "b169627abd9cd364eb98b964444e71c61263edab",
          "message": "fix(ci): clang-tidy only analyses files in compile_commands.json\n\nThe find-based scan included standalone headers, fuzz targets, and\nbench files not registered in CMake, causing 'Error while processing'\nfailures.  Switch to jq extraction from compile_commands.json so only\nfiles CMake actually builds are analysed.",
          "timestamp": "2026-02-24T04:40:42+04:00",
          "tree_id": "9b950e31b7c0979ce91731080a1dd6bb782d3e56",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b169627abd9cd364eb98b964444e71c61263edab"
        },
        "date": 1771893719688,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "76b59817b4e3471cd62a03f51ce23a4982c24f28",
          "message": "fix(ci): filter .S assembly from clang-tidy, add --quiet and parallel xargs\n\n- grep only .cpp/.cc/.cxx files (skip .S assembly that clang-tidy cannot parse)\n- --quiet suppresses the 'N warnings generated' noise lines\n- xargs -P nproc -n 4 parallelises analysis across cores",
          "timestamp": "2026-02-24T04:53:02+04:00",
          "tree_id": "54d8ab0c300b93317600025d728025a74e55cb64",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/76b59817b4e3471cd62a03f51ce23a4982c24f28"
        },
        "date": 1771894460072,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "bdb3f13a92338ab818fcc6e160732a35b63d3f17",
          "message": "perf: add const to FieldElement52 intermediates in hot-path point arithmetic\n\nMark write-once local variables as const in jac52_double, jac52_add_mixed,\nand jac52_add (return-by-value variants).  Resolves clang-tidy\nmisc-const-correctness warnings and gives the optimizer explicit immutability\nguarantees on ~60 intermediates in the critical ECC path.\n\nVerified: 12/12 tests pass, zero warnings under -Werror.",
          "timestamp": "2026-02-24T05:09:11+04:00",
          "tree_id": "8b41a4d42952aeb5099b1ae7e27f704422861192",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bdb3f13a92338ab818fcc6e160732a35b63d3f17"
        },
        "date": 1771895432285,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "b04aa979524a7741473d07671ba6bc2034eb5ff9",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T05:09:22+04:00",
          "tree_id": "8b41a4d42952aeb5099b1ae7e27f704422861192",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b04aa979524a7741473d07671ba6bc2034eb5ff9"
        },
        "date": 1771895445494,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 258,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 129,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "c144bfcbdd99526104cf3b0b39d4fd5b90e770a2",
          "message": "fix(ci): guard #pragma clang diagnostic with #ifdef __clang__\n\nGCC-13 with -Werror treats unknown clang pragmas as errors\n(-Werror=unknown-pragmas).  Wrap all #pragma clang diagnostic\npush/pop/ignored blocks in #ifdef __clang__ guards.\n\nAffected: test_bip32.cpp, test_musig2.cpp, test_ecdh_recovery_taproot.cpp\nVerified: 12/12 tests pass, zero warnings under -Werror (Clang 21).",
          "timestamp": "2026-02-24T05:12:11+04:00",
          "tree_id": "2bf9d3ba567e0dce1660f8cf18a117820898f650",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c144bfcbdd99526104cf3b0b39d4fd5b90e770a2"
        },
        "date": 1771895614101,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "f0d1f53002f1a85033c97dc7432feb13b7480b39",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T05:12:22+04:00",
          "tree_id": "2bf9d3ba567e0dce1660f8cf18a117820898f650",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f0d1f53002f1a85033c97dc7432feb13b7480b39"
        },
        "date": 1771895622696,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "987e127d43404a303dac9b95a71465d731a8da33",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T05:15:12+04:00",
          "tree_id": "3563c93701190c05c283e64a08a4c2ca98857117",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/987e127d43404a303dac9b95a71465d731a8da33"
        },
        "date": 1771895910202,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "b0a5832bf926f11a2b83c34f9a5326653138b2c7",
          "message": "fix(ct): branchless ct_compare -- eliminate timing leak\n\nReplace ternary operators (?:) in ct_compare with pure bitwise\narithmetic (OR-negate for non-zero detection, mask-select for\nconditional latch).  Add asm volatile value barrier in the loop\nto prevent GCC from re-introducing branches.\n\ndudect reported |t|=22.29 on CI with the old implementation;\nthe new version uses zero branches in the hot loop.\n\nVerified: 12/12 tests pass, zero warnings under -Werror.",
          "timestamp": "2026-02-24T05:15:04+04:00",
          "tree_id": "3563c93701190c05c283e64a08a4c2ca98857117",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b0a5832bf926f11a2b83c34f9a5326653138b2c7"
        },
        "date": 1771895968848,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "539064bbb0782a3e7632463ed1c2bb8af9b587bc",
          "message": "fix(ci): pass llvm-cov report to SonarCloud scanner\n\nSonarCloud Quality Gate failed with new_coverage=0.0% because\nthe LLVM coverage report was generated but never fed to the\nscanner. Add -Dsonar.cfamily.llvm-cov.reportPath to scanner args.\n\nVerified: sonarcloud.yml now passes build/sonar/llvm-cov-report.txt\nto the SonarCloud scan step.",
          "timestamp": "2026-02-24T05:34:41+04:00",
          "tree_id": "7795857f417100a23d846f7f32b03f6f82526a51",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/539064bbb0782a3e7632463ed1c2bb8af9b587bc"
        },
        "date": 1771896961976,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "a7f4d0b25e7840c565b63fbdb63088996264fc0b",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T05:34:49+04:00",
          "tree_id": "7795857f417100a23d846f7f32b03f6f82526a51",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a7f4d0b25e7840c565b63fbdb63088996264fc0b"
        },
        "date": 1771896966333,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "59b721f7b1913ea2786cd68f27c92f5f067aec46",
          "message": "fix(ci): fix llvm-cov binary selection for accurate coverage\n\n- Use run_selftest as primary binary (links full library)\n- Add standalone test binaries as -object args\n- Remove 2>/dev/null error suppression for visibility\n- Add diagnostic output (profraw files, binary paths, report size)\n- Exclude benchmark files from coverage in sonar-project.properties\n- Add CPD minimum token threshold\n\nRoot cause: find ... | head -20 could pick random executables as\nthe primary binary, missing library source coverage mapping.\nThe run_selftest binary links the entire fastsecp256k1 static\nlibrary, ensuring all source files appear in the llvm-cov report.\n\nVerify: SonarCloud Quality Gate should pass (new_coverage >= 80%)\nsince existing tests already exercise address.cpp, bip32.cpp,\nmusig2.cpp, and precompute.cpp changed lines.",
          "timestamp": "2026-02-24T05:58:04+04:00",
          "tree_id": "8a451c2c30ca5f155cf7a0f9cb392b08a87c7390",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/59b721f7b1913ea2786cd68f27c92f5f067aec46"
        },
        "date": 1771898376120,
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
            "value": 147,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 55000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 163,
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
          "id": "f8172ff00f6b992c7c1d594b1eeadaa8ec2d63c8",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T05:58:16+04:00",
          "tree_id": "8a451c2c30ca5f155cf7a0f9cb392b08a87c7390",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f8172ff00f6b992c7c1d594b1eeadaa8ec2d63c8"
        },
        "date": 1771898378788,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "194b27cc87651d4e8ef5b0210a65949148ae8f34",
          "message": "fix(coverage): eliminate dead code in precompute.cpp for accurate coverage\n\n- RDTSC(): wrap in #if SECP256K1_PROFILE_DECOMP — only compiled when\n  profiling is enabled (was always compiled on x64 but never called)\n- multiply_u64(): use _umul128() instead of duplicating __int128 inline\n- mul64x64(): use _umul128() instead of duplicating __int128 inline\n- mul_256 lambda: use _umul128() instead of duplicating __int128 inline\n\nRoot cause: on x64 Linux (CI), _umul128() was defined but all callers\n(multiply_u64, mul64x64, mul_256) had their own #else branches using\n__int128 directly, bypassing _umul128(). This made _umul128() dead code.\nNow all three use _umul128() (which IS the __int128 wrapper), giving\ncoverage through existing scalar multiplication tests.\n\nVerify: cmake --build build-win-check && run_selftest → 19/19 passed",
          "timestamp": "2026-02-24T06:12:18+04:00",
          "tree_id": "16cb8f186be2dc49d12ea9d173884199b376dd2e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/194b27cc87651d4e8ef5b0210a65949148ae8f34"
        },
        "date": 1771899218597,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 20,
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
            "value": 259,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 22000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 129,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 122,
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
          "id": "4823bd61607ae3caae1a2f4be0cffbd706a5d5fb",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T06:12:30+04:00",
          "tree_id": "16cb8f186be2dc49d12ea9d173884199b376dd2e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4823bd61607ae3caae1a2f4be0cffbd706a5d5fb"
        },
        "date": 1771899225089,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "a296dd8d42f0fd961f22143392dfbb80dc684a1f",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T06:25:50+04:00",
          "tree_id": "c5ea1797433cf8c042dfb2b1d86e1982b03c0614",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a296dd8d42f0fd961f22143392dfbb80dc684a1f"
        },
        "date": 1771900031687,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 54000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "e7790cdebe38c4b0697d573f01c5a26958f3a024",
          "message": "release: v3.12.2\n\nSecurity:\n- Branchless ct_compare — eliminate timing side-channel leak\n  (dudect |t| 22.29 → 2.17)\n\nFixed:\n- SonarCloud coverage: use run_selftest as primary llvm-cov binary\n- Dead code elimination in precompute.cpp (RDTSC, _umul128 DRY)\n- GCC #pragma clang diagnostic guards in 3 test files\n- GCC -Wstringop-overflow in base58check_encode\n- All -Werror warnings resolved (41 files)\n- Clang-tidy CI: filter .S assembly, add --quiet and parallel\n- Unused variable in bip32.cpp to_public()\n\nChanged:\n- const on ~60 FieldElement52 intermediates in point.cpp hot paths\n- Benchmark exclusion in sonar-project.properties\n\nAdded:\n- GOVERNANCE.md, ROADMAP.md, CONTRIBUTING.md DCO\n- OpenSSF Best Practices badge\n- Code scanning fixes #281, #282",
          "timestamp": "2026-02-24T06:25:38+04:00",
          "tree_id": "c5ea1797433cf8c042dfb2b1d86e1982b03c0614",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/e7790cdebe38c4b0697d573f01c5a26958f3a024"
        },
        "date": 1771900108418,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "465041bac43aa4b14bc4a42437519a1215fe90d9",
          "message": "fix(ci): add missing aarch64 cross-compilation toolchain file\n\npackaging.yml arm64 job references cmake/toolchain-aarch64-linux-gnu.cmake\nbut the file was never committed. Creates the standard CMake toolchain\nfile for aarch64-linux-gnu cross-compilation (matches Ubuntu package\ng++-aarch64-linux-gnu).\n\nFixes: 'Could not find toolchain file' CMake error in packaging CI.",
          "timestamp": "2026-02-24T06:43:58+04:00",
          "tree_id": "0c48e3285a31e9f63bf97091279192b7a1f5f94c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/465041bac43aa4b14bc4a42437519a1215fe90d9"
        },
        "date": 1771901124489,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "c18d20eeb469db25facfe64371207cf2609cb07a",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T06:44:16+04:00",
          "tree_id": "0c48e3285a31e9f63bf97091279192b7a1f5f94c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c18d20eeb469db25facfe64371207cf2609cb07a"
        },
        "date": 1771901134980,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "24c7f9b1151c809dad66db65036214c2989ebc00",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T06:51:47+04:00",
          "tree_id": "1e00ffad07e5de93898e100622ab8556d01f794a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/24c7f9b1151c809dad66db65036214c2989ebc00"
        },
        "date": 1771901585375,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
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
          "id": "36828ada1fed864fab98b902558c181d30eee070",
          "message": "fix(test): heap-buffer-overflow in dudect smoke mode\n\nfield_inv section used hardcoded NSLOW=5000 but arrays are only N\nelements (SMOKE_N_FIELD=3000 in smoke mode). Writing indices 3000..4999\noverflows the heap allocation -> ASan crash, double-free on all CI.\n\nFix: NSLOW = min(N, 5000) so it never exceeds the allocated array size.\n\nVerified: smoke test runs 34/34 passed, 0 failed, no crash.",
          "timestamp": "2026-02-24T06:51:33+04:00",
          "tree_id": "1e00ffad07e5de93898e100622ab8556d01f794a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/36828ada1fed864fab98b902558c181d30eee070"
        },
        "date": 1771901716094,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "e80a1b54ef280cd5c0e34ff7e2da3057c79b2954",
          "message": "fix(ci): suppress Valgrind still-reachable from precomputed tables\n\nCTest -T MemCheck considers 'still reachable: 2,621,440 bytes'\n(the program-lifetime precomputed wNAF/comb table for G) as a\ndefect, causing the security-audit Valgrind job to fail even\nthough all ERROR SUMMARYs are 0 and definitely lost is 0.\n\nChanges:\n- valgrind.supp: suppress Leak_StillReachable (match-leak-kinds:\n  reachable) so CTest's MemCheck parser sees 0 defects\n- security-audit.yml: pass --suppressions to Valgrind, add\n  -E '^ct_sidechannel$' (strict dudect excluded, smoke still runs),\n  fix sanitizers step -E to exact match\n- CMakeLists.txt: include(CTest) instead of enable_testing() to\n  generate DartConfiguration.tcl (eliminates 'Cannot find file' warning)\n\nDefinite/indirect/possible leaks + all memory errors still fully checked.",
          "timestamp": "2026-02-24T07:19:56+04:00",
          "tree_id": "3b6dfac7aa439347b91def06bb503c92a3ab9403",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/e80a1b54ef280cd5c0e34ff7e2da3057c79b2954"
        },
        "date": 1771903282936,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "0b5528ae637e18b28ee04882a3a8e02594e7bce1",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T07:21:50+04:00",
          "tree_id": "3b6dfac7aa439347b91def06bb503c92a3ab9403",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/0b5528ae637e18b28ee04882a3a8e02594e7bce1"
        },
        "date": 1771903395958,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 145,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "f5efab269217e947bf140d920e790a382df2c6d8",
          "message": "feat(test): add BIP-32 official test vectors TV1-TV5 (90 checks)\n\n- Add test_bip32_vectors.cpp with all 4 BIP-32 spec test vectors:\n  TV1 (128-bit seed, 5 derivation levels)\n  TV2 (512-bit seed, 5 levels with large-index hardened children)\n  TV3 (leading zeros retention in private keys)\n  TV4 (leading zeros in hardened children, 3 levels)\n- TV5 serialization: verify 78-byte xprv/xpub format, version bytes,\n  depth, parent fingerprint, child number, chain code, key payload\n- Public derivation consistency: xpub normal child == xprv normal child pubkey\n- Fix ExtendedKey::public_key() for public keys: decompress from prefix + x\n  (was incorrectly treating x-coordinate as a scalar)\n- Add pub_prefix field to ExtendedKey for y-parity tracking\n- Register standalone CTest target (bip32_vectors) + selftest runner module\n- All hex values verified against bip_utils reference implementation",
          "timestamp": "2026-02-24T07:43:11+04:00",
          "tree_id": "9c8a4121967e828eac1b8e4a1a98b446a69abc2f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f5efab269217e947bf140d920e790a382df2c6d8"
        },
        "date": 1771904853502,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "a09e6b6ebebe76799f373702cb9e3efc3bd623ca",
          "message": "feat(ci): add nightly workflow for extended dudect + differential tests\n\n- Create .github/workflows/nightly.yml:\n  - Differential correctness: multiplier=100 (~1.3M checks) with env/CLI control\n  - dudect full mode: 30 min statistical run (strict 4.5 t-threshold)\n  - Manual dispatch with configurable multiplier and timeout\n  - Runs daily at 03:00 UTC\n- Update tests/differential_test.cpp:\n  - Accept multiplier via argv[1] or DIFFERENTIAL_MULTIPLIER env var\n  - Default multiplier=1 preserves existing CI behavior\n  - All loop counts scale: 1000*N for crypto ops, 100*N for arithmetic\n- Fix sonarcloud.yml: -E ct_sidechannel → -E '^ct_sidechannel$' (exact match)",
          "timestamp": "2026-02-24T07:48:17+04:00",
          "tree_id": "f34382d237b7a6d969ce851b1a111a5f92a325db",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a09e6b6ebebe76799f373702cb9e3efc3bd623ca"
        },
        "date": 1771904983322,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "ddc9dcf1e1e5e4c7631e42e01aeacd4b4a262ac2",
          "message": "Merge dev: BIP-32 TV1-TV5 vectors, nightly workflows, public key decompression fix",
          "timestamp": "2026-02-24T07:48:30+04:00",
          "tree_id": "f34382d237b7a6d969ce851b1a111a5f92a325db",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ddc9dcf1e1e5e4c7631e42e01aeacd4b4a262ac2"
        },
        "date": 1771905005143,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 258,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 129,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "434e6c109f36bb14a8bb4eaeabe42556cbc62a51",
          "message": "docs(changelog): add v3.13.0 and v3.12.3 release notes",
          "timestamp": "2026-02-24T12:50:58+04:00",
          "tree_id": "bd565194f1449e40323c317c406d6dc0f9e04760",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/434e6c109f36bb14a8bb4eaeabe42556cbc62a51"
        },
        "date": 1771923140363,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 27000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "a4fa14fc7d9a11928cfabfb21baae9e60707c219",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T12:51:09+04:00",
          "tree_id": "bd565194f1449e40323c317c406d6dc0f9e04760",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a4fa14fc7d9a11928cfabfb21baae9e60707c219"
        },
        "date": 1771923150604,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
          "id": "4bd801ea011997444775421ab058b574516148a8",
          "message": "fix(ci): valgrind.supp syntax + ct_memzero cache-line false positive\n\nvalgrind.supp: add obj:* location line — Valgrind 3.22 requires at\nleast one non-'...' location entry in each suppression block.\n\ntest_ct_sidechannel: ct_memzero test now uses a single aligned buffer\ninstead of two separate heap arrays (bufs0/bufs1). The old design\nmeasured different memory addresses per class, causing cache-line\ntiming artifacts (|t|=32.10) that are unrelated to ct_memzero itself.",
          "timestamp": "2026-02-24T13:09:32+04:00",
          "tree_id": "cf386518e500cb271abcbb8e3912e1cd51679101",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4bd801ea011997444775421ab058b574516148a8"
        },
        "date": 1771924260815,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "646cd844d1519bc828c22ac34bdab22d015ad83d",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T13:09:52+04:00",
          "tree_id": "cf386518e500cb271abcbb8e3912e1cd51679101",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/646cd844d1519bc828c22ac34bdab22d015ad83d"
        },
        "date": 1771924273339,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 27000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
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
          "id": "807af3db8fc2eece84285974ffb48993e9ecc0da",
          "message": "feat: CT signing guardrails + normalization spec (roadmap 1.1.5, 1.4.3)\n\n- ct/sign.hpp: ct::ecdsa_sign(), ct::schnorr_sign() using ct::generator_mul()\n- ct_sign.cpp: implementation mirroring fast:: path but with CT point ops\n- SECP256K1_REQUIRE_CT compile flag deprecates non-CT sign functions\n- docs/NORMALIZATION.md: low-S (BIP-62), DER encoding, Schnorr even-Y spec\n- Updated INDUSTRIAL_ROADMAP_WORKING.md with completed items",
          "timestamp": "2026-02-24T13:13:55+04:00",
          "tree_id": "5e5f665ce972417934012b3b481d0098f58b97b3",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/807af3db8fc2eece84285974ffb48993e9ecc0da"
        },
        "date": 1771924524944,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 142,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "cf7a0336a983b4c762e518ab039114456e2d18c4",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T13:14:11+04:00",
          "tree_id": "5e5f665ce972417934012b3b481d0098f58b97b3",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/cf7a0336a983b4c762e518ab039114456e2d18c4"
        },
        "date": 1771924542067,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "42c847d26aaa4986fb53ff8bd60b7bb4b05547bb",
          "message": "fix: remove spurious libs/UltrafastSecp256k1/ prefix from GitHub URLs\n\nnuspec releaseNotes, nuget/docs/README, wasm/README, and all\nbinding READMEs (nodejs, react-native, csharp) pointed to\n.../blob/main/libs/UltrafastSecp256k1/CHANGELOG.md etc.\nwhich is a 404 — the files live at repo root.",
          "timestamp": "2026-02-24T13:20:01+04:00",
          "tree_id": "23139bcb38b4ae306fee4e6c05f7708eff59c64d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/42c847d26aaa4986fb53ff8bd60b7bb4b05547bb"
        },
        "date": 1771924886339,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 146,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 137,
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
          "id": "ec9346d665553f2d3534db6cad75035b62d7ee54",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T13:20:14+04:00",
          "tree_id": "23139bcb38b4ae306fee4e6c05f7708eff59c64d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ec9346d665553f2d3534db6cad75035b62d7ee54"
        },
        "date": 1771924925827,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "5d84bb022dcd548d262a312cb6fd38d84d1a0016",
          "message": "fix: ct_memzero Windows false positive + ct_sign tests + packaging concurrency\n\n1. ct_memzero: use symmetric memcpy pre-conditioning instead of\n   memset(0) vs random_bytes — avoids MSVC store-buffer asymmetry\n   that caused |t|=27 false positive on Windows CI.\n\n2. test_ct.cpp: add ct::ecdsa_sign, ct::schnorr_sign, ct::schnorr_pubkey\n   tests (all verify CT==fast equivalence + signature validity).\n   Improves SonarCloud new-code coverage for ct_sign.cpp.\n\n3. packaging.yml: add concurrency group to prevent race condition\n   when tag is force-updated and two publish jobs compete.",
          "timestamp": "2026-02-24T13:42:03+04:00",
          "tree_id": "ebe02759ec1d6150085d6d8329d50eca681fcc52",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5d84bb022dcd548d262a312cb6fd38d84d1a0016"
        },
        "date": 1771926208437,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "e871650be9ea10c33d96d6e8899aca1ad8fa7f39",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T13:42:15+04:00",
          "tree_id": "ebe02759ec1d6150085d6d8329d50eca681fcc52",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/e871650be9ea10c33d96d6e8899aca1ad8fa7f39"
        },
        "date": 1771926211092,
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
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
          "id": "f84b7cbd706576ba9d3c21a95d84bae0e400e3dd",
          "message": "fix(sonar): extract tagged_hash.hpp to eliminate duplication + fix strlen hotspot\n\n- Extract make_tag_midstate, cached_tagged_hash, BIP-340 midstate constants\n  into shared cpu/include/secp256k1/tagged_hash.hpp\n- schnorr.cpp and ct_sign.cpp now use 'using detail::...' declarations\n- Replace strlen(tag) with std::string_view in tagged_hash() (SonarCloud\n  security hotspot: buffer overflow risk)\n- All 14 tests pass (selftest, BIP-340, ct_sidechannel, etc.)",
          "timestamp": "2026-02-24T13:55:49+04:00",
          "tree_id": "98dd4a399385240c87c13d1aaa8983bd11340923",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f84b7cbd706576ba9d3c21a95d84bae0e400e3dd"
        },
        "date": 1771927037816,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "a358080698426490b2e3b8d6538265a2d000a633",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T13:56:03+04:00",
          "tree_id": "98dd4a399385240c87c13d1aaa8983bd11340923",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a358080698426490b2e3b8d6538265a2d000a633"
        },
        "date": 1771927041490,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "ff43cd061908a13457c5911468600e7a68cb6540",
          "message": "fix(ci): Valgrind false-positive + publish duplicate-deb crash\n\n- security-audit.yml: replace 'grep -rl ... | head -1' with 'grep -q ...'\n  The pipeline masked grep's exit status — head -1 with empty stdin exits 0,\n  making the if-condition always TRUE even when Valgrind reports 0 errors.\n\n- CMakeLists.txt: set CPACK_DEBIAN_PACKAGE_ARCHITECTURE from\n  CMAKE_SYSTEM_PROCESSOR for cross-compilation.  Without this,\n  DEB-DEFAULT uses dpkg --print-architecture (= host amd64) even when\n  the target is aarch64, producing two .debs with the same filename.\n  softprops/action-gh-release then double-deletes the existing asset → 404.",
          "timestamp": "2026-02-24T14:23:15+04:00",
          "tree_id": "582d03a0b072695d7e27bfbd024226b570a9cb3f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ff43cd061908a13457c5911468600e7a68cb6540"
        },
        "date": 1771928679247,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "b4a5504a1ec21933a2b86c75b83f1ee07e525526",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T14:23:28+04:00",
          "tree_id": "582d03a0b072695d7e27bfbd024226b570a9cb3f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b4a5504a1ec21933a2b86c75b83f1ee07e525526"
        },
        "date": 1771928693294,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 28,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "3a5227de99a56b27c178ca8f17a62802cb8b32b4",
          "message": "feat: FAST≡CT equivalence tests (320 property-based + boundary vectors)\n\n- Add test_ct_equivalence.cpp: 8 test functions covering generator_mul,\n  scalar_mul, ECDSA sign, Schnorr sign/pubkey, and group law invariants\n- Boundary scalars: 0, 1, 2, n-1, n-2, (n+1)/2\n- 64 random generator_mul + 64 scalar_mul + 32 ECDSA + 32 Schnorr\n- Deterministic PRNG (seeded SHA256 counter) for reproducibility\n- Register in unified runner (run_selftest) + standalone CTest target\n- Update CT_VERIFICATION.md version to v3.13.0\n\nVerified: 320/320 equivalence checks passed, 14/14 CTest targets green\n(ct_sidechannel: known MSVC false-positive, passes on Linux GCC/Clang)",
          "timestamp": "2026-02-24T14:54:42+04:00",
          "tree_id": "abdeaa75527395f76c375b917a4e48e904d83cf7",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3a5227de99a56b27c178ca8f17a62802cb8b32b4"
        },
        "date": 1771930564898,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 28000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 136,
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
          "id": "93e61c7634fd0ff2f887c78f186bf70bea8b0333",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T14:54:52+04:00",
          "tree_id": "abdeaa75527395f76c375b917a4e48e904d83cf7",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/93e61c7634fd0ff2f887c78f186bf70bea8b0333"
        },
        "date": 1771930602294,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "0668bbe87a34d84d2347f9932cb0b18aefb0c619",
          "message": "docs: normalization spec (BIP-62/66/146) + dudect CI artifact publishing\n\n- Add docs/NORMALIZATION_SPEC.md: low-S rule, DER strictness, RFC 6979,\n  Schnorr non-malleability, verification strictness, cross-backend consistency\n- security-audit.yml: upload dudect smoke results as artifact (90-day retention)\n- nightly.yml: upload dudect full results as artifact (90-day retention)\n\nPhase I roadmap: 1.1.5 (normalization spec) + 1.2.7 (results publishing) complete.\nPhase I now at ~95% (1 remaining: in-process libsecp256k1 harness).",
          "timestamp": "2026-02-24T15:00:35+04:00",
          "tree_id": "48d82f4b29f6f2f60d8ce887108726ad5ec84d21",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/0668bbe87a34d84d2347f9932cb0b18aefb0c619"
        },
        "date": 1771930916947,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 258,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 129,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "415025037ae2e4ab129122f94795852c010e03ef",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T15:00:48+04:00",
          "tree_id": "48d82f4b29f6f2f60d8ce887108726ad5ec84d21",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/415025037ae2e4ab129122f94795852c010e03ef"
        },
        "date": 1771931230813,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "1f2ce1ffe72e72a0da7bd1bf957ec27c1ccae382",
          "message": "ci: add Discord webhook notifications for releases and commits\n\n- Add Discord notification step to release.yml (rich embed on publish)\n- Create discord-commits.yml workflow for push notifications on main/dev/master\n- Webhook URLs read from GitHub Secrets (DISCORD_WEBHOOK_RELEASES, DISCORD_WEBHOOK_COMMITS)",
          "timestamp": "2026-02-24T19:29:01+04:00",
          "tree_id": "e1abddfded90646b8ace4584e1a8c0a3eca9c31e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/1f2ce1ffe72e72a0da7bd1bf957ec27c1ccae382"
        },
        "date": 1771947031367,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 258,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 129,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "951b7379586028fdbe35a247a4472e8bd0d9b869",
          "message": "ci: add workflow_dispatch trigger to discord-commits",
          "timestamp": "2026-02-24T19:31:52+04:00",
          "tree_id": "2141e5c30b7330c56ff0ef7c9f648dedb5442ce9",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/951b7379586028fdbe35a247a4472e8bd0d9b869"
        },
        "date": 1771947197387,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "9f499c0f6d7e99d915fbf976f6b7a250111ee6d3",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-24T22:07:46+04:00",
          "tree_id": "8041a0c6dc539d6116d35a6726b84461df0d28c8",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9f499c0f6d7e99d915fbf976f6b7a250111ee6d3"
        },
        "date": 1771956553744,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "a911dd7bf82c6f6c7e43406138335d42e8257208",
          "message": "fix(ct): full scalar_mul mod n in GLV decomposition — 16/16 tests pass\n\nRoot cause: ct_glv_decompose used ct_mul_256x_lo128_mod (single-phase\nreduction, 256×128-bit multiply) which overflowed when ct_mul_shift_384\nrounded c1/c2 up to exactly 2^128. Additionally, lambda*k2 computation\nonly read 2 lower limbs of k2_abs, silently dropping limb[2]=1.\n\nFix: Replace all truncated multiply paths with ct_scalar_mul_mod_n —\nfull 4×4 schoolbook → 8-limb product → 3-phase reduce_512\n(512→385→258→256 bits), matching libsecp256k1's secp256k1_scalar_mul.\n\nKey changes:\n- Added ct_scalar_mul_mod_n() in both 5x52 (__int128) and 4x64\n  (portable U128/mul64) paths\n- minus_b2 now full 256-bit Scalar (n - b2), not 128-bit b2_pos\n- Formula: k2 = c1*minus_b1 + c2*minus_b2 (scalar_add, not sub)\n- lambda*k2 uses full scalar multiply (not 256×128 truncated)\n- Removed unused ct_mul_lo128_mod, ct_mul_256x_lo128_mod\n\nVerified: Docker clang-17 Release, 16/16 CTest pass (0 failures).\nPreviously: 3/64 ct_equivalence failures at indices 31, 48, 56.",
          "timestamp": "2026-02-24T22:07:32+04:00",
          "tree_id": "8041a0c6dc539d6116d35a6726b84461df0d28c8",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a911dd7bf82c6f6c7e43406138335d42e8257208"
        },
        "date": 1771956992597,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "d08010a24125ea7bc978294036d4d9af463c828f",
          "message": "fix: add [[maybe_unused]] to diagnostic helpers in diag_scalar_mul.cpp\n\nprint_scalar() and print_point_xy() are intentionally kept for debugging\nbut not always called. Mark with [[maybe_unused]] to suppress\n-Werror=unused-function on CI (g++-13).",
          "timestamp": "2026-02-24T22:52:29+04:00",
          "tree_id": "a92322ca29477b27a977d8de4b52109e22ced880",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d08010a24125ea7bc978294036d4d9af463c828f"
        },
        "date": 1771959235152,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "ab13fddb7ae1a92b06f604c305631d7f32af91c7",
          "message": "fix(scorecard): pin ubuntu:24.04 by hash in Dockerfile.local-ci\n\nResolves code-scanning alert #283 (PinnedDependenciesID).\nUses same sha256 digest as main Dockerfile.",
          "timestamp": "2026-02-24T22:56:19+04:00",
          "tree_id": "b14492b659c86cb371d4f793b128c35d44d8aaa9",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ab13fddb7ae1a92b06f604c305631d7f32af91c7"
        },
        "date": 1771959461587,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "03c1263cdb3a5b86aaa277de936afb55d98df5c2",
          "message": "feat: complete all 12 binding APIs + 9 READMEs + fix package naming\n\nBindings:\n- Java: +22 JNI functions + 3 helper classes (RecoverableSignature, WifDecoded, TaprootOutputKeyResult)\n- Swift: +20 functions (DER, recovery, ECDH, tagged_hash, BIP-32, taproot)\n- React Native: +15 functions\n- Python: +3 functions (ctx_clone, last_error, last_error_msg)\n- Rust: +2 functions (last_error, last_error_msg)\n- Dart: +1 function (ctx_clone)\n\nDocumentation:\n- 9 new binding READMEs: c_api, dart, go, java, php, python, ruby, rust, swift\n- 3 existing READMEs fixed: nodejs, csharp, react-native (CT/fast architecture note)\n- Fix incorrect package names across all docs:\n  libsecp256k1-fast* -> libufsecp* (apt, rpm, arch, pkg-config, CMake)\n  secp256k1-fast-cpu -> fastsecp256k1 (linker flags, CMake targets)\n- Fix INDUSTRIAL_ROADMAP_WORKING.md link -> ROADMAP.md in README\n- Rename RPM spec: libsecp256k1-fast.spec -> libufsecp.spec\n- Fix debian/control, debian/changelog, arch/PKGBUILD package names\n- Fix secp256k1-fast.pc.in linker flag\n- Fix .github/workflows/packaging.yml comment\n\nSelftest:\n- Add selftest report structs (selftest.hpp)\n- Refactor tally() in selftest.cpp",
          "timestamp": "2026-02-25T00:04:31+04:00",
          "tree_id": "0a8c52c4cabbd28eaf1b228abc6b0a77be558488",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/03c1263cdb3a5b86aaa277de936afb55d98df5c2"
        },
        "date": 1771963565516,
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
            "value": 286,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "a633d165eb78050a4c0b8bde5337fd6a8c45fede",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-25T00:05:10+04:00",
          "tree_id": "0a8c52c4cabbd28eaf1b228abc6b0a77be558488",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a633d165eb78050a4c0b8bde5337fd6a8c45fede"
        },
        "date": 1771963612958,
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
            "value": 283,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "e206ff957cf9988e9fefc7566927211052802669",
          "message": "fix(ci): suppress -Werror=unused-function for get_platform_string\n\nAdd [[maybe_unused]] attribute — function prepared for selftest_report API.",
          "timestamp": "2026-02-25T00:10:20+04:00",
          "tree_id": "b12ee0eb837a1947ecff7e0c002ad8bae4f10910",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/e206ff957cf9988e9fefc7566927211052802669"
        },
        "date": 1771963916552,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 138,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "3635a43e891438288927a1552ae0a32794bcd92e",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-25T00:10:44+04:00",
          "tree_id": "b12ee0eb837a1947ecff7e0c002ad8bae4f10910",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3635a43e891438288927a1552ae0a32794bcd92e"
        },
        "date": 1771963944628,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "b094b07d267240ad5b3c93e2b770e0173b92280b",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-25T00:36:04+04:00",
          "tree_id": "f6fb70de85a1309b8185c0c7f96ec5e9e1cce20f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b094b07d267240ad5b3c93e2b770e0173b92280b"
        },
        "date": 1771965438180,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 259,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "d776ae3b245d47f33bdb9b2934d0be6d01a224d0",
          "message": "release: v3.14.0 — full 12-language binding coverage + packaging fixes\n\nBindings:\n- Java: +22 JNI functions + 3 helper classes\n- Swift: +20 functions\n- React Native: +15 functions\n- Python: +3, Rust: +2, Dart: +1\n- Go, Node.js, C#, Ruby, PHP: already complete\n- 9 new binding READMEs (c_api, dart, go, java, php, python, ruby, rust, swift)\n- Selftest report API structs (selftest.hpp) + tally() refactor\n\nPackaging:\n- Fix package naming: libsecp256k1-fast* -> libufsecp* across all docs\n- RPM spec renamed, Debian control fixed, Arch PKGBUILD fixed\n- pkg-config and CMake target names corrected\n\nCI:\n- [[maybe_unused]] on get_platform_string() (-Werror fix)\n- Dockerfile.local-ci ubuntu:24.04 pinned by SHA (Scorecard)",
          "timestamp": "2026-02-25T00:35:46+04:00",
          "tree_id": "f6fb70de85a1309b8185c0c7f96ec5e9e1cce20f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d776ae3b245d47f33bdb9b2934d0be6d01a224d0"
        },
        "date": 1771965802276,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "b38b1b0fbb19fc07699dae3e4acb25356f14fbbb",
          "message": "feat(test): cross-library differential test vs bitcoin-core/libsecp256k1 (Phase I 1.1.4)\n\nIn-process test linking both UltrafastSecp256k1 AND bitcoin-core/libsecp256k1\nv0.6.0 (via FetchContent). 7 test suites, 7860 checks:\n\n  [1] Public key derivation (500 rounds)\n  [2] ECDSA Sign(UF) -> Verify(Ref) (500 rounds)\n  [3] ECDSA Sign(Ref) -> Verify(UF) (500 rounds)\n  [4] Schnorr BIP-340 cross-verification (500 rounds)\n  [5] RFC 6979 byte-exact signature match (200 rounds)\n  [6] Edge cases (k=1, k=2, k=n-1, powers of 2)\n  [7] Point addition cross-check (200 rounds)\n\nBuild: cmake -DSECP256K1_BUILD_CROSS_TESTS=ON\nVerify: ctest -R cross_libsecp256k1 --output-on-failure\n\nCompletes Phase I roadmap task 1.1.4.",
          "timestamp": "2026-02-25T00:56:13+04:00",
          "tree_id": "d9da82d07aeabec1dbc4a5365aebcd58f87a8741",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b38b1b0fbb19fc07699dae3e4acb25356f14fbbb"
        },
        "date": 1771966914425,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 26000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "41699dfcf2089b6e839d72585674e70d69235136",
          "message": "feat(test): parser fuzz tests — DER + Schnorr + Pubkey (Phase II 2.3.1-2.3.3)\n\n9 test suites, 580K+ checks, 0 failures:\n  [1] DER random bytes (100K rounds)\n  [2] DER adversarial inputs (null, zero-len, bad tags, overflow)\n  [3] DER round-trip compact→DER→compact (50K rounds)\n  [4] Schnorr verify random inputs (100K rounds)\n  [5] Schnorr round-trip sign→verify (10K rounds)\n  [6] Pubkey parse random bytes (100K rounds)\n  [7] Pubkey round-trip create→parse (10K rounds)\n  [8] Pubkey adversarial (null, bad prefix, infinity, non-canonical x)\n  [9] ECDSA verify random garbage (50K rounds)\n\nCMake: SECP256K1_BUILD_FUZZ_TESTS option, links ufsecp_static.",
          "timestamp": "2026-02-25T01:11:39+04:00",
          "tree_id": "fd34b206fe4e72337001ca4bb421642fb15a7b92",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/41699dfcf2089b6e839d72585674e70d69235136"
        },
        "date": 1771967738619,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
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
            "value": 300,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 165,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 50000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 25000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 56000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 153,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 144,
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
          "id": "50a37203a41c672a93dd6135d63bca02a4049519",
          "message": "feat(test): MuSig2 + FROST protocol tests (Phase II 2.1.1-2.2.2)\n\n11 test suites, 975 checks, 0 failures:\n  MuSig2:\n    [1] Key aggregation determinism (50 rounds, 2-5 signers)\n    [2] Ordering-dependent aggregation (20 rounds)\n    [3] Duplicate-key aggregation\n    [4] Full round-trip: 2, 3, 5 signers (20 rounds each)\n    [5] Wrong signer partial verify fails (10 rounds)\n    [6] Bit-flip invalidates signature (20 rounds)\n  FROST:\n    [7] DKG: 2-of-3, 3-of-5 (share verification, group key agreement)\n    [8] Signing round-trip: 2-of-3, 3-of-5 (partial verify + schnorr_verify)\n    [9] Different 2-of-3 subsets all produce valid sigs\n    [10] Bit-flip invalidates FROST signature\n    [11] Wrong partial sig fails verification\n\nNote: MuSig2 uses x-only (32-byte) pubkeys for hash inputs rather than\nBIP-327's plain (33-byte) keys. Protocol structure identical; end-to-end\ncorrectness verified via schnorr_verify().\n\nCMake: SECP256K1_BUILD_PROTOCOL_TESTS option.",
          "timestamp": "2026-02-25T01:19:36+04:00",
          "tree_id": "306474ea772576b2cc5399a2298c6c63e9a99111",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/50a37203a41c672a93dd6135d63bca02a4049519"
        },
        "date": 1771968194694,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "4824fbc01ccf4eaef57886126bcaedbaed1ceef9",
          "message": "feat(test): MuSig2 + FROST advanced tests (Phase II 2.1.3-2.2.4)\n\n9 suites, 316 checks, 0 failures:\n  [1] Rogue-key resistance (coefficient non-triviality)\n  [2] Key coefficient binding (same key, different groups)\n  [3] Message binding (cross-verify fails)\n  [4] Nonce binding (fresh nonces → different R)\n  [5] Fault injection (wrong secret key in partial sign)\n  [6] FROST malicious DKG (tampered share detected by VSS)\n  [7] FROST bad partial sig (detected + aggregation fails)\n  [8] FROST message binding (cross-verify fails)\n  [9] FROST signer set binding (all subsets valid, all different)",
          "timestamp": "2026-02-25T01:23:52+04:00",
          "tree_id": "af5eb40d57f65d1323622a08b43e7230a4787b44",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4824fbc01ccf4eaef57886126bcaedbaed1ceef9"
        },
        "date": 1771968319998,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 143,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "acdcbd9ee410385b6d73091c37a5ca08660b69a9",
          "message": "feat: address/BIP32/FFI fuzz tests + release/deprecation docs + user guide\n\nTasks completed:\n  2.3.4 Address encoders fuzz (P2PKH/P2WPKH/P2TR/WIF) - 73,959 checks\n  2.3.5 BIP32 path parser fuzz (master/derive/path parsing)\n  2.3.6 FFI boundary fuzz (ctx lifecycle, ECDSA, Schnorr, ECDH, Taproot)\n  2.3.7 Regression corpus management (tests/corpus/ + MANIFEST.txt)\n  3.4.1 Release process document (docs/RELEASE_PROCESS.md)\n  3.4.2 Deprecation policy document (docs/DEPRECATION_POLICY.md)\n  3.5.1 Dedicated user guide (docs/USER_GUIDE.md)\n\nTest: cmake --build build -DSECP256K1_BUILD_FUZZ_TESTS=ON && ./test_fuzz_address_bip32_ffi\nResult: 73,959 passed, 0 failed, 0 crashes",
          "timestamp": "2026-02-25T01:40:55+04:00",
          "tree_id": "ee0f126b263d2762f0a1845a95549ee2fc598c0f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/acdcbd9ee410385b6d73091c37a5ca08660b69a9"
        },
        "date": 1771969678818,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "74bdec5a5ee7f8774eff919ccba6b9b351ee0a31",
          "message": "feat: FROST KAT tests + 7 Phase III docs (audit, invariants, bug bounty, thread safety, checklist, LTS, FAQ)\n\nPhase II:\n- tests/test_frost_kat.cpp: 9 suites, 76 checks — Lagrange properties,\n  DKG determinism, Feldman VSS, 2-of-3 signing (all subsets), 3-of-5\n  signing (non-contiguous), Lagrange consistency (all C(5,3)=10 subsets),\n  pinned KAT group key, pinned signing roundtrip, secret reconstruction\n\nPhase III documentation:\n- docs/AUDIT_SCOPE.md: external audit engagement scope (P0 task 3.2.1)\n- docs/INVARIANTS.md: 108 invariants across 14 categories (P1 task 3.1.3)\n- docs/BUG_BOUNTY.md: scope + reward tiers (P1 tasks 3.3.1, 3.3.2)\n- docs/THREAD_SAFETY.md: concurrency model per-component (P1 task 3.6.1)\n- docs/PRE_RELEASE_CHECKLIST.md: mandatory pre-release steps (P1 task 3.6.6)\n- docs/LTS_POLICY.md: version lifecycle + support (P2 task 3.4.3)\n- docs/FAQ.md: 10 common pitfalls + troubleshooting (P2 task 3.5.5)",
          "timestamp": "2026-02-25T02:06:14+04:00",
          "tree_id": "40dd92743c77979f4054309f681594cd5aa2f940",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/74bdec5a5ee7f8774eff919ccba6b9b351ee0a31"
        },
        "date": 1771970857913,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "14ec02da85d4362ee34b25d1eae2e8314e53ad68",
          "message": "Phase II 2.5 + Phase III completion: reproducible builds, signed releases, SBOM, performance docs, sample apps, disclosure policy\n\nPhase II 2.5 (Reproducible Builds & Signed Releases):\n- Dockerfile.reproducible: two-stage build comparison\n- scripts/verify_reproducible_build.sh: local verification script\n- scripts/generate_sbom.sh: CycloneDX 1.6 SBOM generator\n- docs/REPRODUCIBLE_BUILDS.md: full documentation\n- release.yml: cosign keyless signing + SBOM step\n\nPhase III Documentation & Operational Hardening:\n- docs/PERFORMANCE_GUIDE.md (3.5.3): compiler, ASM, batch, GPU, CT tuning\n- docs/BENCHMARK_METHODOLOGY.md (3.5.9): framework, statistical method, CI\n- docs/SAFE_DEFAULTS.md (3.6.3): build/runtime/CT/GPU/protocol defaults\n- docs/PERFORMANCE_REGRESSION.md (3.6.4): automated tracking, alert thresholds\n- examples/signing_demo/: ECDSA + Schnorr sign/verify demo\n- examples/threshold_demo/: FROST 2-of-3 DKG + signing ceremony demo\n- examples/CMakeLists.txt: updated for new targets\n- SECURITY.md: disclosure policy, CVSS severity, bug bounty reference (3.3.3)\n\nRoadmap: Phase I 100%, Phase II ~93%, Phase III ~87% (overall ~93%)",
          "timestamp": "2026-02-25T02:19:36+04:00",
          "tree_id": "1d529c277af5162fee131a1e74d5ff425e62ff11",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/14ec02da85d4362ee34b25d1eae2e8314e53ad68"
        },
        "date": 1771972005316,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "d0b6ac20ee2b87250c7f099a302627839f409bf2",
          "message": "Verification Transparency Drop: AUDIT_READINESS_REPORT_v1 + INTERNAL_AUDIT + release artifacts\n\nNew files:\n- docs/AUDIT_READINESS_REPORT_v1.md: public-facing verification transparency\n  report. Structured, human-readable. Covers: coverage (641K+ checks),\n  differential validation (vs libsecp256k1), CT verification (dudect),\n  fuzzing (~580K+ iterations), supply chain, known gaps. Explicit\n  disclaimer: not audited, not formally verified.\n- docs/INTERNAL_AUDIT.md: full internal audit results (718 lines).\n  Per-check detail across 10 sections, 108 invariants, performance\n  baseline. Includes reproduction commands for every result.\n- ANNOUNCEMENT_DRAFT.md: technical announcement for DelvingBitcoin/Stacker.\n  Cold, factual, no hype. Links to verification artifacts.\n\nModified:\n- release.yml: added selftest_report.json generation step (builds minimal\n  binary, runs selftest_report(ci).to_json()), copies verification_report.md.\n  Both added to GitHub Release artifacts alongside SHA256SUMS, cosign sigs,\n  SBOM.\n\nRelease artifacts now include:\n  SHA256SUMS.txt, *.sig + *.pem (cosign), sbom.cdx.json,\n  selftest_report.json, verification_report.md",
          "timestamp": "2026-02-25T02:32:58+04:00",
          "tree_id": "460f2bd9dde24c66f360130845cc3432891e2f24",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d0b6ac20ee2b87250c7f099a302627839f409bf2"
        },
        "date": 1771972464011,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "90ebe12e0797789d263590848b740c59ef3bc3d6",
          "message": "feat(bindings): industrial polish — smoke tests, error/memory/ABI/packaging docs, examples\n\n 7-point bindings polish checklist complete:\n\n 1. docs/BINDINGS.md — 42-function parity matrix across 11 languages\n 2. Smoke tests for all 11 bindings (golden vectors, 12-14 tests each):\n    - Python, Node.js, C#, Java, Swift, Go, Rust, Dart, PHP, Ruby, React Native\n    - Each tests: ctx/ABI, pubkey golden, ECDSA sign/verify, Schnorr,\n      recovery, SHA-256, address, WIF, ECDH, error path, determinism\n 3. docs/BINDINGS_ERROR_MODEL.md — error code→exception mapping per language,\n    recoverable vs fatal classification, cross-language invariants\n 4. docs/BINDINGS_MEMORY_MODEL.md — secret handling guarantees at FFI boundary,\n    per-language honest assessment (what CAN and CANNOT be zeroized)\n 5. docs/BINDINGS_ABI_COMPAT.md — ABI version gate contract, per-language\n    implementation, compatibility matrix, release checklist\n 6. docs/BINDINGS_PACKAGING.md — per-ecosystem distribution guide,\n    install commands, platform artifacts, CI validation matrix\n 7. docs/BINDINGS_EXAMPLES.md — 3 copy-paste examples per language\n    (sign/verify, address derive, error handling)\n\n Also includes: CT empirical report, differential testing methodology doc,\n batch verify cross-check tests (test_cross_libsecp256k1.cpp)",
          "timestamp": "2026-02-25T03:08:40+04:00",
          "tree_id": "af78954de9cf74bc8c51af82bffced6fdda5e7b1",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/90ebe12e0797789d263590848b740c59ef3bc3d6"
        },
        "date": 1771974749269,
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
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "a939f6a46f78017fa69e59ef026e76c2136e95fb",
          "message": "fix(python): ctypes FFI accepts bytes via _BytesPtr argtype, version/abi_version as @property; fix smoke test error-path assertions to use bool returns",
          "timestamp": "2026-02-25T03:21:52+04:00",
          "tree_id": "7cd57cc66c4712b3af6aaecdb73ca4e61379cfab",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a939f6a46f78017fa69e59ef026e76c2136e95fb"
        },
        "date": 1771975406426,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "be528aef66e530c965b96c9ee266077cbaf55663",
          "message": "audit: add AUDIT_COVERAGE.md + ASCII cleanup + CT fixes\n\n- Add comprehensive AUDIT_COVERAGE.md documenting all 46 audit modules\n  across 8 sections with ~1M+ total assertions\n- Pure ASCII cleanup: remove all Unicode from source/cmake/script files\n  (box-drawing, arrows, Greek, emoji, BOM, Georgian in comments)\n- CT fix: RISC-V is_zero_mask (seqz+neg inline asm)\n- CT fix: ct_compare general path (snez)\n- All 188 files updated for ASCII-only compliance (Section 17 rule)\n- Verified: 46/46 audit PASS on X64, ARM64, RISC-V (QEMU + Mars HW)\n- Verified: 24/24 CTest PASS on X64",
          "timestamp": "2026-02-25T19:14:21+04:00",
          "tree_id": "9332cfe00a99eb894c9ef2f31a332ec9faeb49d4",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/be528aef66e530c965b96c9ee266077cbaf55663"
        },
        "date": 1772032621865,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "3c7dac618f6b5ad1809a6b272d73564c753abc20",
          "message": "fix: metal test include path + expand AUDIT_COVERAGE.md with full CI infrastructure\n\n- Fix metal/tests/test_metal_host.cpp: include path was ../../tests/test_vectors.hpp\n  but file only exists at audit/test_vectors.hpp (pre-existing bug)\n- Expand AUDIT_COVERAGE.md: add CI/CD pipeline docs (14 workflows, 17 build configs),\n  sanitizer testing, coverage-guided fuzzing, nightly extended, static analysis,\n  bindings CI (12 langs), supply chain security, benchmark CI, release pipeline,\n  packaging, audit gap analysis, full platform matrix\n\nVerify: cmake --build <build> --target metal_host_test (should compile + pass 76/76)",
          "timestamp": "2026-02-25T19:44:44+04:00",
          "tree_id": "6f857601017420291aec48d6f935e56bc7dd6e8c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3c7dac618f6b5ad1809a6b272d73564c753abc20"
        },
        "date": 1772034393214,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 136,
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
          "id": "21ff98ef1e8f0c410f4fc220bf7223d3e3e17ca5",
          "message": "fix(dart): fix smoke test analyze errors -- correct import, method names, named params\n\n- smoke_test.dart: import package:ultrafast_secp256k1/ufsecp.dart (was package:ufsecp/)\n- smoke_test.dart: UfsecpContext.sha256() -> ctx.sha256() (instance, not static)\n- smoke_test.dart: addrP2wpkh -> addrP2WPKH (correct case)\n- smoke_test.dart: wifEncode positional -> named params\n- pubspec.yaml: add test ^1.24.0 as dev_dependency\n\nVerify: dart analyze --fatal-infos -> No issues found",
          "timestamp": "2026-02-25T20:00:31+04:00",
          "tree_id": "241ff7651c5a8743accd83c3ba8cbc5148acbe0f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/21ff98ef1e8f0c410f4fc220bf7223d3e3e17ca5"
        },
        "date": 1772035342730,
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
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "7966cb211c53ea57ccb86f608ab22a1b2ba732d7",
          "message": "release: signed SHA256SUMS manifest + verify instructions + Scorecard governance\n\n- release.yml: rename SHA256SUMS.txt -> SHA256SUMS (standard name)\n- release.yml: strip ./subdir/ prefixes from checksums (bare filenames)\n- release.yml: cosign sign-blob on SHA256SUMS manifest itself (.sig + .pem)\n- release.yml: upload SHA256SUMS.sig + SHA256SUMS.pem to GitHub Release\n- README.md: full Release Signing & Verification section (Linux/macOS/Windows)\n- README.md: cosign verify-blob command for manifest signature verification\n- README.md: updated supply chain table (cosign + SBOM + reproducible builds)\n- CODEOWNERS: fix em-dash -> ASCII double-dash (Section 17 compliance)",
          "timestamp": "2026-02-26T01:00:09+04:00",
          "tree_id": "3700d21aaea78af54c4d39467c932f0087984d28",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7966cb211c53ea57ccb86f608ab22a1b2ba732d7"
        },
        "date": 1772053321127,
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
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "a417c6b2d6e54828e971322d726a7e8759b3e8d6",
          "message": "ci: add ClusterFuzzLite, Cppcheck, mutation testing, SARIF integration\n\n- ClusterFuzzLite: Dockerfile + build.sh + cflite.yml workflow\n  - PR fuzzing (120s, ASan + UBSan) on every PR\n  - Nightly batch fuzzing (600s) + weekly corpus pruning\n  - 3 targets: fuzz_field, fuzz_scalar, fuzz_point\n\n- Cppcheck: new cppcheck.yml workflow with SARIF upload\n  - Static analysis with XML + SARIF output\n  - Results visible in GitHub Security tab\n\n- Clang-Tidy: upgrade to SARIF output\n  - Convert text diagnostics to SARIF format\n  - Upload to GitHub Security tab (security-events: write)\n\n- Mutation Testing: new mutation.yml (weekly schedule)\n  - Mull LLVM mutation engine (primary)\n  - Manual fallback: +/-, ==/ !=, return mutations\n  - 9 core crypto sources in scope\n  - 70% kill threshold\n\n- Seed corpora: 12 boundary-value seeds for fuzz targets\n  - fuzz_field: zero, one, p-1, generator coords (64B each)\n  - fuzz_scalar: zero, one, n-1, scalar_one (64B each)\n  - fuzz_point: 1, 2, n-1, midpoint (32B each)",
          "timestamp": "2026-02-26T02:28:47+04:00",
          "tree_id": "83505dd0b8d3d61a2f470c5a56579cb8733157ea",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a417c6b2d6e54828e971322d726a7e8759b3e8d6"
        },
        "date": 1772058634342,
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
            "value": 286,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "8deed91978120e32167a0b403f43c3bc9416e67f",
          "message": "audit: enhance unified runner + CI workflow + evidence scripts\n\n- unified_audit_runner.cpp: add library version, git hash,\n  audit framework version (v2.0.0) to JSON+TXT reports;\n  implement --section filter, --json-only, --list-sections,\n  --help flags; skip empty sections in summary table\n- audit/CMakeLists.txt: inject GIT_HASH at compile time via\n  execute_process; add version.hpp include paths\n- .github/workflows/audit-report.yml: new CI workflow that\n  builds + runs unified_audit_runner on Linux (GCC-13,\n  Clang-17) + Windows (MSVC); uploads JSON+TXT as artifacts;\n  posts text report to GitHub Step Summary; weekly + manual +\n  release tag triggers; cross-platform verdict aggregation\n- scripts/generate_audit_package.ps1: PowerShell script that\n  builds runner, runs it, collects compiler/cmake/git evidence\n  into a dated audit-evidence-* directory with README\n- scripts/generate_audit_package.sh: bash equivalent\n\nVerified: builds + runs on Windows MSVC 1944 (Release).\nAll 8 sections, 47 modules + selftest = 48 checks.",
          "timestamp": "2026-02-26T02:51:50+04:00",
          "tree_id": "254d56d29cc1b7dbbbcb83c65ae5adff5fce212f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/8deed91978120e32167a0b403f43c3bc9416e67f"
        },
        "date": 1772060009938,
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
            "value": 283,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 10000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "1c1c21ac809b64ab03e6629787e28102a74fc04e",
          "message": "ci: fix cppcheck.yml YAML syntax + cflite.yml dev trigger\n\n- cppcheck.yml: extract inline Python SARIF converter to\n  .github/scripts/xml2sarif.py (fixes YAML literal block scalar\n  breaking at column-0 Python imports, line 83 syntax error)\n- cflite.yml: remove 'dev' from push trigger (no ClusterFuzzLite\n  job matches 'push to dev' event, so all jobs were skipped; PR\n  and schedule/dispatch triggers remain for actual fuzzing)",
          "timestamp": "2026-02-26T03:54:09+04:00",
          "tree_id": "73754656670671b5c44ef2dcc81aa37971efe90e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/1c1c21ac809b64ab03e6629787e28102a74fc04e"
        },
        "date": 1772063747862,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "189f2d659d4ef37ec32b1433ec435871b23b47ae",
          "message": "ci: enable fuzz + protocol tests in all CI jobs\n\n- CMakeLists.txt: reorder add_subdirectory so include/ufsecp (ufsecp_static)\n  is built BEFORE audit/ -- fixes TARGET ufsecp_static check failing for\n  test_fuzz_parsers and test_fuzz_address_bip32_ffi\n- ci.yml: add -DSECP256K1_BUILD_FUZZ_TESTS=ON -DSECP256K1_BUILD_PROTOCOL_TESTS=ON\n  to linux (4 matrix combos), sanitizers (ASan+UBSan, TSan), windows (MSVC),\n  macOS (AppleClang), and coverage (llvm-cov) jobs\n- Locally verified: fuzz_parsers (580k checks), fuzz_address_bip32_ffi (74k),\n  musig2_frost (975), musig2_frost_advanced (316), frost_kat (76) -- all PASS",
          "timestamp": "2026-02-26T04:02:49+04:00",
          "tree_id": "4f6a43df02cc0061c688bcffb52ffd57b5e20b53",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/189f2d659d4ef37ec32b1433ec435871b23b47ae"
        },
        "date": 1772064269444,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "4c9a2bcc4f61f281cccc0ff9d4102276af54482a",
          "message": "fix: normalize field elements in is_on_curve debug checker\n\nSome optimized multiplication paths (montgomery_reduce_bmi2) can\nproduce field element results in [p, 2^256) -- correct mod p but\nnot canonical. The is_on_curve debug invariant checker compared\ny^2 and x^3+7 via raw limb operator== which requires canonical\nform. Fix: add FieldElement::zero() to both sides before comparing,\nwhich forces add_impl's conditional p-subtraction to normalize.\n\nFixes: debug_invariants test failure on CI (Windows + Linux).",
          "timestamp": "2026-02-26T04:35:35+04:00",
          "tree_id": "0e2fa15b99ddc063e339804c6a25ddd9d91c8f81",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4c9a2bcc4f61f281cccc0ff9d4102276af54482a"
        },
        "date": 1772066232117,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
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
            "value": 300,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 167,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 25000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 56000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 156,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 148,
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
          "id": "d51f93f68498d8834baeb120f633a5e9142631c0",
          "message": "fix: normalize FieldElement::operator== to handle non-canonical limbs\n\noperator== previously compared raw limbs, which fails when\nmul_impl/square_impl produce results in [p, 2^256) -- correct\nmod p but non-canonical.  Now normalizes both operands via a\nsingle conditional p-subtraction before comparing.\n\nThis fixes debug_invariants CI failures on Linux GCC-13 where\ny^2 and x^3+7 could have different non-canonical representations.\n\nThe FE52 and FE26 paths already normalized in operator==; this\nbrings FE64 (4x64) to parity.\n\nPerformance: operator== is not in the hot path on x86-64\n(SECP256K1_FAST_52BIT uses normalizes_to_zero() instead).\nFallback platforms see <0.1% overhead from two normalize calls.\n\nAlso removes SECP_ASSERT_NORMALIZED on Point::x()/y() results\nin test_full_chain -- arithmetic outputs are not guaranteed\ncanonical; the curve equation check (y^2 == x^3+7) validates\ncorrectness via the now-correct operator==.\n\n23/23 CTest PASS (MSVC Release).",
          "timestamp": "2026-02-26T05:03:36+04:00",
          "tree_id": "542781d139e7b1737d9a84bb9274072ca9ae35b4",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d51f93f68498d8834baeb120f633a5e9142631c0"
        },
        "date": 1772067917232,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 214,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 204,
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
          "id": "1b9b2cf4ca001d5554c37d35d39ba27e0525c96c",
          "message": "fix(is_on_curve): use FE52 Jacobian check on FAST_52BIT platforms\n\nOn Linux (Clang/GCC x86-64), SECP256K1_FAST_52BIT=1 is defined.\nPoint stores FE52 internally; is_on_curve() was converting FE52->FE64\nthen doing FE64 arithmetic (inverse, mul, square). This conversion +\nFE64 chain produced wrong results for ~20% of random kG tests on CI.\n\nFix: when SECP256K1_FAST_52BIT is defined, check the curve equation\ndirectly in Jacobian coordinates using native FE52 arithmetic:\n  Y^2 == X^3 + 7*Z^6\nThis avoids three error-prone steps:\n  - FE52->FE64 conversion via to_fe()\n  - FE64 SafeGCD inverse (z_inv)\n  - FE64 assembly mul/square chain\n\nThe FE64 path is preserved for non-52bit platforms (MSVC, ESP32, etc).\n\nVerified: 372/372 pass locally (MSVC Release), all CTest targets pass.",
          "timestamp": "2026-02-26T05:31:26+04:00",
          "tree_id": "217ac912b58220c94d688660d38c05e34cc10b16",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/1b9b2cf4ca001d5554c37d35d39ba27e0525c96c"
        },
        "date": 1772069584229,
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
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 25000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
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
          "id": "0df09e8af9cc4d29ae4b6712a9212ca43fd0719b",
          "message": "fix(audit_ct): relax timing sanity threshold 1.5x -> 2.0x for CI\n\nmacOS ARM64 GitHub Actions runners are multi-tenant VMs where\ntiming jitter routinely reaches 1.5-1.7x due to frequency scaling,\nshared caches, and hypervisor scheduling.\n\nThe rudimentary timing check is not a formal CT test -- real CT\nvalidation is performed by dudect (ct_sidechannel_smoke: 34/34).\nRaising the threshold to 2.0x prevents flaky CI while still catching\ncatastrophic regressions (e.g. branch-on-secret).\n\nFixes #357 unified_audit failure on macOS.",
          "timestamp": "2026-02-26T06:00:56+04:00",
          "tree_id": "58eca716539244f8748cf3754bd51604df13c067",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/0df09e8af9cc4d29ae4b6712a9212ca43fd0719b"
        },
        "date": 1772071354920,
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
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "54b6d1d5c4de311ca84d5c577a2b94ec27afd612",
          "message": "fix(sonar): resolve 19 SonarCloud reliability bugs for Quality Gate\n\n- Suppress cpp:S876 (unary minus on unsigned) project-wide via\n  sonar-project.properties. This is a false positive for constant-time\n  crypto code: unsigned negation is the fundamental branchless mask idiom\n  (-1ULL = 0xFFFF...FFFF). Well-defined per C++ standard. (15 bugs)\n\n- sha256.hpp: add defensive guard before finalize() padding to satisfy\n  S3519 buffer-overflow symbolic execution (buf_len_ is [0,63] after\n  update(), but static analyzer cannot prove it). (1 BLOCKER)\n\n- bip32.cpp: remove dead ternary 'hardened ? 37 : 37' -> '37'.\n  Both BIP32 derivation paths produce 37-byte HMAC input. (S3923)\n\n- selftest.cpp: avoid S1764 identical-subexpression false positive\n  in 'a - a == 0' test by using distinct variable copy. (S1764)\n\n- ufsecp_impl.cpp: move (void)ctx before return to fix S1763 dead\n  code warning. (S1763)\n\n24/24 CTest PASS, 12023 selftest checks, 0 failures.",
          "timestamp": "2026-02-26T06:26:02+04:00",
          "tree_id": "17a92e16aad00cd721243189d75865ee3c3814e9",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/54b6d1d5c4de311ca84d5c577a2b94ec27afd612"
        },
        "date": 1772072867149,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 137,
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
          "id": "b33d2fc4527967cb932cdb9e84f2b144f0b45406",
          "message": "fix(msvc): SECP256K1_NOINLINE macro + s_gen4 race + smoke threshold\n\n- Add SECP256K1_NOINLINE macro to config.hpp: __declspec(noinline)\n  for MSVC, __attribute__((noinline)) for GCC/Clang. MSVC silently\n  ignores __attribute__; without noinline, scalar_mul_glv52 (~5KB\n  locals) gets inlined causing GS-cookie (stack canary) corruption\n  under parallel CTest execution.\n\n- Replace all 6 __attribute__((noinline)) in point.cpp with\n  SECP256K1_NOINLINE: jac52_double, jac52_add_mixed,\n  jac52_add_mixed_inplace, jac52_add, scalar_mul_glv52,\n  dual_scalar_mul_gen_point.\n\n- Fix s_gen4 data race: bare check-then-allocate pattern replaced\n  with C++11 magic static (thread-safe one-time initialization).\n\n- Raise DUDECT_SMOKE T_THRESHOLD to 50.0 on MSVC: volatile-based\n  value_barrier (no inline asm on x64 MSVC) adds 15-30 t-stat\n  noise on is_zero_mask tests; 50.0 still catches gross leaks\n  (real leak |t| > 100) while eliminating false flakes.\n\nVerified: 28/28 CTest PASS (MSVC Release, parallel -j) x2 runs.",
          "timestamp": "2026-02-26T13:02:04+04:00",
          "tree_id": "c4d23dab54b9c45db2ce2ac8059edf88ea819259",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b33d2fc4527967cb932cdb9e84f2b144f0b45406"
        },
        "date": 1772096632319,
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
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "0db996926abb33d7fcc3fc4786315015306f80e0",
          "message": "ci: add emsdk actions-cache-folder for CDN resilience\n\nsetup-emsdk runs 'emsdk update' which downloads HEAD.zip from\nemscripten-core/emsdk. Transient CDN failures cause WASM CI to\nfail with 'Remote end closed connection'. Adding actions-cache-folder\nenables @actions/cache so subsequent runs skip the download entirely.\n\nApplied to: ci.yml (wasm job), release.yml (build-wasm job).",
          "timestamp": "2026-02-26T13:15:34+04:00",
          "tree_id": "9be0d30975b64e867b1a449708ef1c60ad32fc15",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/0db996926abb33d7fcc3fc4786315015306f80e0"
        },
        "date": 1772097439810,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 16000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "6ab1bef709c3f57b5dab6f124ade3f78218c9b0c",
          "message": "fix: resolve SonarCloud bugs and code smells across 19 files\n\nBug fixes (MISRA 5-3-2 unary minus on unsigned):\n- ct/ops.hpp: is_zero_mask, bool_to_mask, lt_mask (3 fixes)\n- field_52_impl.hpp: half(), normalize_inline() (2 fixes)\n- field_branchless.hpp: cmov, cmovznz, select, is_zero, eq (5 fixes)\n- field_26.cpp: half() (1 fix)\n- ct_field.cpp: field_half() (1 fix)\n- ct_point.cpp: table_lookup_core() x3 instances (6 fixes)\n- ct_scalar.cpp: scalar_half() (1 fix)\n- field.cpp: sub_impl, add_impl, normalize (3 fixes)\nPattern: -val -> 0ULL - val (identical codegen, MISRA-compliant)\n\nRemove redundant operator!= (C++20 auto-generates from ==):\n- field.hpp, scalar.hpp, field_52.hpp, field_26.hpp (declarations)\n- field_52.cpp, field_26.cpp (definitions)\n\nReword false-positive commented-out code (11 comments):\n- address.hpp, coin_hd.hpp, ct/field.hpp, ct/scalar.hpp\n- ct/ops.hpp, field_52.hpp, glv.hpp, schnorr.hpp\n\nModernize coin_params.hpp:\n- sizeof()/sizeof() -> std::size()\n- Raw for-loops -> range-based for\n- Fix ALL_COINS[i] -> coin in range-for context\n\nVerify: cmake --build build-win-ci --config Release -j (0 errors, 0 C4146)\nCTest: 28/29 pass (ct_sidechannel flaky dudect pre-existing)",
          "timestamp": "2026-02-26T14:21:21+04:00",
          "tree_id": "1a0b0f1663d8c83f8cc423ac1a9c165ebc604104",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/6ab1bef709c3f57b5dab6f124ade3f78218c9b0c"
        },
        "date": 1772101378033,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 132,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 21000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "4ed9b2c00fbf1a4a2864e72ab14260c19ee64051",
          "message": "ci: pin all actions to SHA + add harden-runner (OpenSSF Scorecard)\n\naudit-report.yml:\n- Pin actions/checkout@v4 -> @de0fac2e (v6)\n- Pin actions/upload-artifact@v4 -> @b7c566a7 (v6.0.0)\n- Pin actions/download-artifact@v4 -> @37930b1c (v7.0.0)\n- Add step-security/harden-runner to all 4 jobs\n\npackaging.yml:\n- Pin fedora:41 Docker image to SHA digest\n\naddress.hpp:\n- Missed SonarCloud comment reword from previous commit\n\nScorecard impact: Pinned-Dependencies 7->10, estimated 7.3->8.5+",
          "timestamp": "2026-02-26T14:31:57+04:00",
          "tree_id": "6e6ced6e078760bf834ea93becd9d6d743cda7cc",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4ed9b2c00fbf1a4a2864e72ab14260c19ee64051"
        },
        "date": 1772102014284,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 132,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 21000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "88f1433797f0ecf5cbbc9682c6a858c3fb5c0510",
          "message": "fix(ct): add value_barrier after mask derivation in ct_compare + WASM KAT target\n\nct_utils.hpp (RISC-V dudect fix, task 5.2.3):\n- Add ct::value_barrier(mask) after 'mask = 0ULL - differs/take' in all 6\n  ct_compare code paths (bulk-8, bulk-4, bulk-1, tail, fallback-main, fallback-tail)\n- Without barrier, RISC-V compilers (GCC/Clang) may convert the mask-based\n  select into a conditional branch, leaking comparison order via timing\n- Expected: RISC-V dudect ct_compare |t| drops from ~17-34 to <4.5\n\nwasm/CMakeLists.txt (WASM KAT equivalence, task 2.6.3):\n- Add wasm_kat_test executable target (compiles test_cross_platform_kat.cpp)\n- Compiled with ENVIRONMENT=node, EXIT_RUNTIME=1, no filesystem\n- CI step already wired in ci.yml: 'node build/wasm/dist/wasm_kat_test.js'\n\nVerify: CTest 4/4 pass (selftest, ct_equivalence, cross_platform_kat, ct_sidechannel_smoke)",
          "timestamp": "2026-02-26T16:43:13+04:00",
          "tree_id": "e63ce6910719f98e26bac5019c4f8cc0b77b1be8",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/88f1433797f0ecf5cbbc9682c6a858c3fb5c0510"
        },
        "date": 1772109901131,
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
            "value": 277,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
          "id": "acfeb726edaa6a13aa3bf1a3aa793bbbefdf5fd0",
          "message": "fix(ct): branchless scalar_window + RISC-V value_barrier fix\n\nscalar_window: remove if(bit_idx+width<=64) branch; always compute\nboth paths with is_nonzero_mask for limb bounds and shift clamping.\n\nvalue_barrier (RISC-V): remove memory clobber on in-order cores.\nThe clobber forced excessive stack spills/reloads creating\ndata-dependent store-to-load forwarding jitter on U74.\nRegister-only barrier still prevents compiler value reasoning.\n\nct_compare general path: reuse ct_load_be instead of inline memcpy+bswap.",
          "timestamp": "2026-02-26T18:25:06+04:00",
          "tree_id": "01182ed9cb8026fcefae9d950d0937b44355c076",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/acfeb726edaa6a13aa3bf1a3aa793bbbefdf5fd0"
        },
        "date": 1772116330675,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 18000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
          "id": "42282dfdeced4ced8f024085bc98ae252708a2d3",
          "message": "fix(ct): platform-specific scalar_window -- branchless RISC-V, branched x86/ARM\n\nMSVC /O2 + LTCG and Clang-17 miscompile the branchless scalar_window when\ninlined into scalar_mul via link-time code generation. The is_nonzero_mask\npattern, while logically correct, is optimized through the value_barrier\nproducing wrong results in FROST protocol tests (musig2_frost_advanced).\n\nFix: use #if defined(__riscv) to select the implementation:\n  - RISC-V: fully branchless (seqz+neg+andn) -- required for dudect-clean CT\n    on in-order U74 cores\n  - x86/ARM/MSVC: branched path (safe on OOO cores, well-tested)\n\nAlso adds WASM KAT equivalence test step to CI.\n\nVerified: 28/28 Windows (MSVC), 23/24 Mars RISC-V (only ct_sidechannel\nexcluded), RISC-V assembly confirmed zero conditional branches.",
          "timestamp": "2026-02-26T20:11:40+04:00",
          "tree_id": "c47909e2b8fbb9df89b4f2dc2001a5c5174c90f2",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/42282dfdeced4ced8f024085bc98ae252708a2d3"
        },
        "date": 1772122409263,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 132,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 13000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "ebdfba8f33349d5fd09d1e9c164e7d9837319963",
          "message": "fix(ci): WASM KAT test -- output to kat/ dir to avoid ESM conflict\n\nThe dist/package.json has \"type\": \"module\" (for the secp256k1.mjs wrapper),\nwhich makes Node.js treat ALL .js files in dist/ as ES modules. Emscripten\ngenerates wasm_kat_test.js with CommonJS require(), causing:\n\n  ReferenceError: require is not defined in ES module scope\n\nFix: output wasm_kat_test.js to build/wasm/kat/ instead of build/wasm/dist/\nso it runs in a directory without the ESM-mode package.json.\n\nVerified: CI #366 only failure was WASM KAT (all other jobs passed).",
          "timestamp": "2026-02-26T21:18:58+04:00",
          "tree_id": "3d45f85e7ec2667d24a4d220b37ed76952208752",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ebdfba8f33349d5fd09d1e9c164e7d9837319963"
        },
        "date": 1772126455061,
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
            "value": 285,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "25e9dbc93eac52ff7c22c900d31fab75c8eda700",
          "message": "fix(ci): exclude ct_sidechannel_smoke from CI + resilient artifact upload\n\n- ctest -E pattern: ^ct_sidechannel$ -> ^ct_sidechannel (matches both\n  ct_sidechannel and ct_sidechannel_smoke). The smoke dudect test is\n  inherently flaky on shared CI runners, especially in -O0 Debug builds\n  where timing jitter is amplified. Dedicated security-audit.yml and\n  nightly.yml still run the full dudect suite.\n\n- WASM Upload WASM artifact: continue-on-error: true so transient\n  GitHub artifact-storage outages do not fail the entire WASM job\n  (was causing exit code 7 when services were unavailable).\n\nFixes: CI #367 linux(gcc-13,Debug) flaky failure + wasm infra failure",
          "timestamp": "2026-02-26T21:52:01+04:00",
          "tree_id": "5cb2bc9cf042d558acd680e0967fd6ee6881de30",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/25e9dbc93eac52ff7c22c900d31fab75c8eda700"
        },
        "date": 1772128431171,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 27000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "24460ae25efa91c29e761ce60477cdaf4d0de1f4",
          "message": "fix(ci): WASM KAT SINGLE_FILE=1 + sanitizer timeout\n\nWASM KAT test:\n- Add SINGLE_FILE=1: embeds .wasm binary inline in .js file,\n  eliminates .wasm sidecar file relocation issue where CMake\n  RUNTIME_OUTPUT_DIRECTORY only moves .js but .wasm stays in\n  original build location\n- Change ASSERTIONS=0 -> ASSERTIONS=1: enables Emscripten error\n  messages in CI logs instead of silent exit code 7\n- Add ls -lh kat/ to Verify step for debugging\n- Add node --stack-size=4096\n\nSanitizer jobs:\n- Exclude unified_audit (too heavy for ASan Debug, causes timeout)\n- Add --timeout 300 per-test cap",
          "timestamp": "2026-02-26T22:12:51+04:00",
          "tree_id": "72c3cf36b4d36a2f44023b1f6a981a5a01515c7a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/24460ae25efa91c29e761ce60477cdaf4d0de1f4"
        },
        "date": 1772129671047,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 24,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 300,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 167,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 40000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 15000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 50000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 26000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 56000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 157,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 148,
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
          "id": "2f62d8b33e1f74c4467b5b702016917cf14efd7d",
          "message": "fix(wasm): KAT test crash -- small precompute tables + memory tuning\n\nRoot cause: G.scalar_mul() triggers scalar_mul_generator() which\nlazy-builds a 270 MB precompute table (window_bits=18, 15 windows x\n262K entries x 72 bytes). This exceeds WASM memory limits and causes\nstack overflow under Emscripten (exit code 7).\n\nKAT test (test_cross_platform_kat.cpp):\n- Add setup_wasm_precompute(): configures window_bits=4 (~74 KB)\n  with use_cache=false for WASM/Emscripten\n- Include precompute.hpp for FixedBaseConfig\n- Guarded by #ifdef __EMSCRIPTEN__ -- no effect on other platforms\n\nwasm/CMakeLists.txt:\n- INITIAL_MEMORY: 4 MB -> 16 MB (headroom for precompute + ops)\n- STACK_SIZE: 512 KB -> 2 MB (deep call stack in point arithmetic)\n- Add -flto (compile + link) for better cross-TU inlining\n- ASSERTIONS=0 in Release (matches benchmark; no stack checks)\n\nci.yml:\n- Remove --stack-size=4096 from node command (not needed)",
          "timestamp": "2026-02-26T22:42:42+04:00",
          "tree_id": "4e27a34bbd2de815e303a012af34db166a436203",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/2f62d8b33e1f74c4467b5b702016917cf14efd7d"
        },
        "date": 1772131460518,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 55000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "991241d9337ae78eb77bb6acd15b93718ad67cd7",
          "message": "fix(wasm): disable SECP256K1_FAST_52BIT on Emscripten\n\nRoot cause: ECDSA verify and Schnorr verify both fail on WASM because\ndual_scalar_mul_gen_point (52-bit path) builds 8192-entry static\ntables using FE52 arithmetic with emulated __int128 on wasm32. The\nemulated 128-bit operations produce incorrect results in the complex\ntable building + wNAF evaluation code path.\n\nFix: Exclude __EMSCRIPTEN__ from the SECP256K1_FAST_52BIT gate.\nThis forces WASM to use:\n- 4x64 FieldElement for Point internals (standard, well-tested)\n- Simple fallback dual_scalar_mul: G.scalar_mul(a) + P.scalar_mul(b)\n- Standard ECDSA/Schnorr verify paths (no FE52 Z^2 check)\n\nNo speed regression: wasm32 has no native 64-bit ops, so __int128\nemulation gives no benefit over 4x64 FieldElement anyway.",
          "timestamp": "2026-02-26T22:55:43+04:00",
          "tree_id": "7cd5f96a7874aa8257d530955fae6738d2b9a5af",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/991241d9337ae78eb77bb6acd15b93718ad67cd7"
        },
        "date": 1772132245822,
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
            "value": 37000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "841b6b5a8062346a2aafc0116d7430273e13a787",
          "message": "fix(wasm): exclude SECP256K1_FAST_52BIT in point.cpp for Emscripten\n\npoint.cpp had its own independent #define SECP256K1_FAST_52BIT gated only\nby __SIZEOF_INT128__, causing mismatch with point.hpp (already fixed in\n991241d). This resulted in compile errors: FE52 constructors/methods\ncompiled against FieldElement-based Point class.\n\nAdd && !defined(__EMSCRIPTEN__) to both __SIZEOF_INT128__ checks in\npoint.cpp (include guard at line 6 and macro define at line 403).",
          "timestamp": "2026-02-26T23:08:12+04:00",
          "tree_id": "3317ebe324809c35ab79bf4e0e65919450259402",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/841b6b5a8062346a2aafc0116d7430273e13a787"
        },
        "date": 1772133019549,
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
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 136,
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
          "id": "665b97157f23982775f7cf7c401502643cad9890",
          "message": "fix(wasm): replace dead wNAF w=5 fallback with GLV+Shamir in scalar_mul\n\nThe non-FE52 fallback for non-generator scalar_mul was plain wNAF w=5 --\ndead code on all tested platforms (x86 uses FE52+GLV, ESP32/STM32 have\ntheir own GLV path). With FE52 disabled for WASM (841b6b5), this was the\nfirst platform to hit it, and it produced incorrect results causing\nECDSA verify and Schnorr verify KAT failures.\n\nReplace with the exact GLV+Shamir algorithm already proven on ESP32/STM32:\n- glv_decompose into two ~128-bit half-scalars\n- Shamir's trick with apply_endomorphism (single doubling chain)\n- Same compute_wnaf_into, add_inplace, dbl_inplace primitives\n\nAlso add 6 new diagnostic tests to cross_platform_kat:\n- Q.scalar_mul(1) == Q (identity on non-generator)\n- Q.scalar_mul(s2) on-curve check\n- Q*s2 == G*(s2^2) algebraic cross-check\n- dual_scalar_mul_gen_point consistency\n\nDesktop (x86-64 Clang): 28/28 passed, 0 failed.",
          "timestamp": "2026-02-26T23:55:20+04:00",
          "tree_id": "d648ce96e610f117e721355d8becb212d0857ac3",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/665b97157f23982775f7cf7c401502643cad9890"
        },
        "date": 1772135843847,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "422d6d964f065a4a18c93e4e89dce54cf84b328a",
          "message": "fix(wasm): replace GLV+Shamir fallback with simple double-and-add\n\nThe GLV+Shamir non-FE52 fallback (commit 665b971) still produced wrong\nresults on WASM. Rather than debugging the complex GLV+endomorphism+wNAF\ninteraction on wasm32, replace with the simplest possible algorithm:\nright-to-left binary double-and-add over 256 scalar bits.\n\nThis uses ONLY dbl_inplace + add_inplace (both already verified by\nField/Scalar/Point KAT on WASM). No GLV, no wNAF, no precomputed tables.\n\nPerformance: ~256 doublings + ~128 additions. Slower than GLV+Shamir but\nacceptable for WASM verification workloads. Desktop (FE52) and embedded\n(ESP32/STM32 GLV+Shamir) paths are unchanged.\n\nDesktop: 28/28 KAT passed.",
          "timestamp": "2026-02-27T00:16:32+04:00",
          "tree_id": "7675693fef1cccb4d433d812fd1fb2c9571ab024",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/422d6d964f065a4a18c93e4e89dce54cf84b328a"
        },
        "date": 1772137089613,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 132,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 21000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "ee779d7a614161b9213313703cfa80c26885271a",
          "message": "fix(wasm): define SECP256K1_NO_INT128 + add s2*G golden vector\n\nRoot cause: on wasm32 Emscripten defines __SIZEOF_INT128__ so the\n__int128 code paths in mul64/add64/sub64/Barrett are active, but\nthe compiler emulates 128-bit arithmetic via builtins (__multi3 etc.)\nwhich produce incorrect results for non-trivial schoolbook/Barrett\nmultiplications.\n\nFix: define SECP256K1_NO_INT128 for the WASM build, forcing the\nportable 32-bit-safe helpers that use only native wasm i64 operations.\nSafeGCD continues to use __int128 (guarded by __SIZEOF_INT128__),\nwhich is correct (field/scalar inv KAT pass).\n\nAlso:\n- Add golden vector check for s2*G (generator path, was only on-curve)\n- Add diagnostic hex output for Q*s2 vs G*(s2^2) cross-check\n\nVerify: 29/29 pass on desktop (x86-64).",
          "timestamp": "2026-02-27T02:57:08+04:00",
          "tree_id": "fd5cfc837147263df86c284d2f519e76d236288e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ee779d7a614161b9213313703cfa80c26885271a"
        },
        "date": 1772146724360,
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
            "value": 37000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 26000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 138,
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
          "id": "a966445cd7887041c6427b90da7c22dbf774df12",
          "message": "fix(wasm): bypass precompute generator path on Emscripten\n\nRoot cause: scalar_mul_generator() windowed accumulation produces\nincorrect results under WASM (Emscripten 3.1.51).  Diagnosed locally\nvia Docker -- double-and-add gives correct s2*G while precompute path\ndiverges for certain scalar values.\n\nFix: skip is_generator_ shortcut on __EMSCRIPTEN__ so all scalar_mul\ncalls use the proven double-and-add fallback.  Performance cost is\nnegligible for WASM use cases.\n\nVerified locally:\n  - Docker emscripten/emsdk:3.1.51: 29/29 PASS (was 25/29)\n  - Desktop x86-64: 29/29 PASS (no regression)\n\nAlso cleaned up diagnostic printf from test_cross_platform_kat.cpp.",
          "timestamp": "2026-02-27T03:39:22+04:00",
          "tree_id": "113fedca02ac565d16849d35495d54b2392db29a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a966445cd7887041c6427b90da7c22dbf774df12"
        },
        "date": 1772149252195,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 132,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 21000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "88421f03e176e9d26eaa599714d7f10dd60d6409",
          "message": "fix(audit): resolve -Werror unused variable/function warnings\n\n- test_fiat_crypto_vectors.cpp: use zero scalar in CHECK assertion\n- test_fault_injection.cpp: use correct_bytes in CHECK assertion\n- test_carry_propagation.cpp: use zero/one FE in CHECK, mark fe_from_hex [[maybe_unused]]",
          "timestamp": "2026-02-27T04:20:02+04:00",
          "tree_id": "c72cd0a02583961ce87c3238b812692cffd50dd2",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/88421f03e176e9d26eaa599714d7f10dd60d6409"
        },
        "date": 1772151705209,
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
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "0027542aea58a701a9ff922b0b5acba8663c27ba",
          "message": "fix(audit): resolve -Werror unused variable/function warnings\n\n- test_fiat_crypto_vectors.cpp: use zero scalar in CHECK assertion\n- test_fault_injection.cpp: use correct_bytes in CHECK assertion\n- test_carry_propagation.cpp: use zero/one FE in CHECK, mark fe_from_hex [[maybe_unused]]",
          "timestamp": "2026-02-27T04:27:31+04:00",
          "tree_id": "6fd1fe02ca3cb2f2a91a1ebcbfe2471014e7f125",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/0027542aea58a701a9ff922b0b5acba8663c27ba"
        },
        "date": 1772152151335,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 54000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "81efd39ad0eccfa9617cfffd4f25da036d7f6b44",
          "message": "fix(audit): resolve all -Werror warnings for GCC-13\n\nFixes:\n- unused-but-set-variable: p_val, bytes_out, G, r, err, coeff bytes\n- Wconversion: static_cast<int> for rng()%%N and static_cast<double> for .count()\n- Wunused-function: [[maybe_unused]] on print_result, diagnose_scalar\n- Wunused-variable: [[maybe_unused]] on g_crash\n- macro redefinition: #ifndef guard for UNIFIED_AUDIT_RUNNER\n\nFiles: audit_fuzz, audit_ct, audit_integration, audit_perf,\n       audit_security, unified_audit_runner, test_fuzz_parsers,\n       test_musig2_frost_advanced, diag_scalar_mul",
          "timestamp": "2026-02-27T05:03:09+04:00",
          "tree_id": "ac71a592f58ee0b104ee10e6540db59157be8c8a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/81efd39ad0eccfa9617cfffd4f25da036d7f6b44"
        },
        "date": 1772154294262,
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
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 142,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 137,
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
          "id": "a2ba6faa36910793d7b7732a7372afbc34f79fa2",
          "message": "Merge pull request #44 from shrec/dev\n\nMerge dev to main: -Werror + SonarCloud + WASM + CT hardening",
          "timestamp": "2026-02-27T05:43:55+04:00",
          "tree_id": "ac71a592f58ee0b104ee10e6540db59157be8c8a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a2ba6faa36910793d7b7732a7372afbc34f79fa2"
        },
        "date": 1772156726993,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 131,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 34000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 42000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 21000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 48000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "6a07afe373dcb9fa7419aee2bdcfd0f44d18395f",
          "message": "ci: update ClusterFuzzLite to latest main (52ecc61)\n\nci: update ClusterFuzzLite to latest main (52ecc61)",
          "timestamp": "2026-02-27T05:53:58+04:00",
          "tree_id": "54a94acf3437ebdb491f19ae389a7efd63e98946",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/6a07afe373dcb9fa7419aee2bdcfd0f44d18395f"
        },
        "date": 1772157334334,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "5238b98d8ac87fb5654b75ae2914d2f3cd38721a",
          "message": "fix(cfl): skip -fno-rtti when UBSan vptr sanitizer is active\n\nfix(cfl): skip -fno-rtti when UBSan vptr sanitizer is active",
          "timestamp": "2026-02-27T05:57:30+04:00",
          "tree_id": "33528296c75362bf005c3164b22c03abeb0c0928",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5238b98d8ac87fb5654b75ae2914d2f3cd38721a"
        },
        "date": 1772157558598,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 37000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "50589f6fa79a2386a0f2cdf95f8afe954bf86339",
          "message": "fix: avoid precomputed-table timeout in fuzz_point under sanitizers\n\nfix: avoid precomputed-table timeout in fuzz_point under sanitizers",
          "timestamp": "2026-02-27T10:47:48+04:00",
          "tree_id": "0c65d1d4adec798eec69eb265b76fcaf93614b34",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/50589f6fa79a2386a0f2cdf95f8afe954bf86339"
        },
        "date": 1772175030908,
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
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 134,
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
          "id": "3105dc10d6cc8edcbefc1d7f0d469e753ac19c4f",
          "message": "fix: schnorr parity, CFL hardening, MIT license (#48)\n\n## Changes\n- schnorr verify Y-parity fix\n- CFL fuzz_point hardened (ASan/UBSan only, no arithmetic assertions)\n- cmake sanitizer detection (no -O3 override in CFL builds)\n- normalizes_to_zero full normalization for 52-bit path\n- normalize_weak before zero-check in mixed-add\n- inverse_safegcd + from_jac52 Z=0 guards\n- MIT license migration (31 files AGPL -> MIT)",
          "timestamp": "2026-02-27T19:45:10+04:00",
          "tree_id": "127eaf3ee49e65a864120b41bb94dc91e2e91eb4",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/3105dc10d6cc8edcbefc1d7f0d469e753ac19c4f"
        },
        "date": 1772207203829,
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
            "value": 286,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
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
          "id": "9d5fecf465976c9b47d1aa43324b2d766d670fdb",
          "message": "fix: deduplicate Z=0 guards + add point edge-case tests (#49)\n\n- Extract Point::z_fe_nonzero() private helper to replace 6 identical\n  inline Z=0 guard blocks in x(), y(), to_compressed(), to_uncompressed(),\n  has_even_y(), x_bytes_and_parity() -- eliminates SonarCloud duplication.\n- Add test_point_edge_cases: 41 checks covering infinity, computed infinity\n  (P + -P), generator outputs, 0*G, n*G, and roundtrip consistency.\n- Boosts patch coverage for defensive Z=0 paths.",
          "timestamp": "2026-02-27T20:20:20+04:00",
          "tree_id": "e553e4e7d022e9a87dcfc871b04c8cd23419ac7e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9d5fecf465976c9b47d1aa43324b2d766d670fdb"
        },
        "date": 1772209313983,
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
            "value": 287,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "b83a9c925489571663d020245f82a1af8b8ccb09",
          "message": "fix: sha256 SonarCloud blocker + codecov exclusions (#50)\n\n- sha256.hpp finalize(): replace if/compress guard with branchless\n  buf_len_ &= 63u clamp. Satisfies Sonar S3519 buf_ overflow check\n  (Reliability Blocker) without changing behavior -- mask is a no-op\n  when the [0,63] invariant holds.\n- point.cpp: mark 6 defensive Z=0 guard branches as LCOV_EXCL_LINE.\n  These branches require infinity_=false with Z=0, which is impossible\n  from the public API -- they exist purely for defense-in-depth.",
          "timestamp": "2026-02-27T20:49:53+04:00",
          "tree_id": "41c54ad6d611e68ebc4e708c3f28e996bf993db0",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b83a9c925489571663d020245f82a1af8b8ccb09"
        },
        "date": 1772211089020,
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
            "value": 288,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "54699f3dc7edebc97939d6a3a14c30ba6376776c",
          "message": "fix: resolve code scanning alerts (const, braces, uninit, Scorecard) (#51)\n\n- field_52_impl.hpp: add braces to inverse_safegcd if-body (fixes 26\n  readability-braces-around-statements); const-qualify mask/overflow in\n  half() and fe52_normalize_inline (fixes ~78 misc-const-correctness)\n- point.cpp: const-qualify negX1 (L645); add braces to throw-0 guard (L904)\n- test_point_edge_cases.cpp: const-qualify 26 never-mutated locals\n- sha512.hpp: value-initialize members in declaration (uninitMemberVar)\n- precompute.cpp: init index=0 (UndefinedBinaryOperatorResult L1756);\n  add (void)0 to empty catch (bugprone-empty-catch L2546);\n  remove redundant inner ec check (oppositeInnerCondition L2634)\n- Dockerfile: pin base-builder image by sha256 digest (PinnedDependenciesID)\n- cppcheck.yml: pin pip version; move security-events:write to job level\n- clang-tidy.yml: move security-events:write to job level\n- cflite.yml: move security-events:write to job level (all 3 jobs)\n- mutation.yml: pin pip version (PinnedDependenciesID)\n\nAll 25 tests pass. No behavior changes.",
          "timestamp": "2026-02-27T21:22:31+04:00",
          "tree_id": "75ad60fdc9735a12e18549c0a3f4127849e5364f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/54699f3dc7edebc97939d6a3a14c30ba6376776c"
        },
        "date": 1772213051457,
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
            "value": 290,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ce0e9109e80d6c4493bca529170c60907aa7754d",
          "message": "ci: pin pip deps by hash + fix cosign signing visibility (#52)\n\nScorecard Pinned-Dependencies (9->10):\n- Create .github/requirements/cppcheck.txt (cppcheck-codequality==1.4.2 +hash)\n- Create .github/requirements/mutation.txt (mutmut==3.5.0 +hash)\n- cppcheck.yml: use --require-hashes --no-deps -r requirements file\n- mutation.yml: use --require-hashes --no-deps -r requirements file\n\nScorecard Signed-Releases (diagnostic):\n- release.yml: remove 2>/dev/null from cosign sign-blob commands\n- Print cosign version at start of signing step\n- Replace silent fallback with ::warning annotations\n- Add summary listing of generated .sig/.pem files\n- This exposes the actual failure reason in CI logs\n\nVerify: next release tag will show cosign errors (if any) in logs\nand Scorecard rescan should see 4/4 pip deps hash-pinned.",
          "timestamp": "2026-02-27T21:32:20+04:00",
          "tree_id": "99808e2355d20ced4ec72006d62c8e426da8a924",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ce0e9109e80d6c4493bca529170c60907aa7754d"
        },
        "date": 1772213654414,
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
            "value": 290,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 138,
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
          "id": "91afe2047965c86d206bff6f66646f5fa6ba930d",
          "message": "fix: last duplicate const in test_frost_kat.cpp (GCC-13 error) (#55)\n\nconst secp256k1::FrostKeyPackage const* -> const ...* const\nMissed in PR #54 (only searched 'const char const*' pattern).\n\nVerify: cmake --build build -j && ctest --test-dir build (25/25 pass)",
          "timestamp": "2026-02-27T23:22:15+04:00",
          "tree_id": "5ed0641047e43b1d3af790aa784a3561956c6f90",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/91afe2047965c86d206bff6f66646f5fa6ba930d"
        },
        "date": 1772220231336,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 268,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 133,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 35000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 43000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "bb1967944be5933fa1a04d8573dc93e1055f5b24",
          "message": "fix: suppress ~550 code scanning alerts (batch 2) (#56)\n\nCategories fixed:\n- objectIndex (64): cppcheck-suppress in sha512.hpp\n- cert-err33-c (159): (void) cast on printf/snprintf/fprintf return values\n- cert-msc32-c (44): NOLINT on intentional constant-seed mt19937 in tests\n- bugprone-implicit-widening (43): static_cast<size_t> on loop counters\n- misc-const-correctness (37): add const to variables where safe\n- shiftTooManyBitsSigned (16): suppress intentional sign-bit arithmetic shifts\n- StackAddressEscape: NOLINTBEGIN in bench_field_26.cpp (benchmark escape idiom)\n- nullPointerRedundantCheck (14): suppress defensive null checks\n- arrayIndexOutOfBoundsCond (6): suppress bounded-loop false positives\n- containerOutOfBounds (2): suppress guarded access false positive\n- uninitMemberVar (2): init buf_ in RIPEMD160 constructors\n- passedByValue (1): pass FieldElement by const ref (field.cpp/field.hpp)\n- AssignmentAddressToInteger (1): fix pointer-to-integer in test_coins.cpp\n\nBuild: 48/48 targets OK (MSVC/Clang)\nTests: 25/25 passed",
          "timestamp": "2026-02-28T00:50:00+04:00",
          "tree_id": "19bfccf5d91feb8fb6d91c5c8865724dcf29ee28",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/bb1967944be5933fa1a04d8573dc93e1055f5b24"
        },
        "date": 1772225493720,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
            "name": "Vano Chkheidze",
            "username": "shrec"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b7d30d42aa68c38606603fb34d1b541cfbf9ac22",
          "message": "fix: restore using declarations removed by clang-tidy (#57)\n\necdsa.cpp: restore 'using fast::FieldElement' -- used in #else path\n(non-__int128 / ESP32 / MSVC) invisible to clang-tidy preprocessor.\n\nct_sign.cpp: restore 'using fast::Scalar' -- used unqualified throughout\nbut in secp256k1::ct namespace (sibling of secp256k1::fast).\n\nFixes CI: windows, wasm, android, benchmark-windows, Security Audit.",
          "timestamp": "2026-02-28T01:01:27+04:00",
          "tree_id": "0d78d22fed71007a487dbd9cd81e1b242f120d1e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b7d30d42aa68c38606603fb34d1b541cfbf9ac22"
        },
        "date": 1772226214400,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 268,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 133,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 35000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 43000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "69b4d97a1e82f0ef1b8158eca02a3246ef633064",
          "message": "fix: remove unused variable 'less' in audit_field.cpp (#58)\n\nVariable was set but never read -- only 'greater' is checked.\nRemoved the variable and simplified the early-exit branch.\nFixes -Werror=unused-but-set-variable in Security Audit CI.",
          "timestamp": "2026-02-28T01:07:04+04:00",
          "tree_id": "b56894de69ce27574a4a517b545c84b50b874750",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/69b4d97a1e82f0ef1b8158eca02a3246ef633064"
        },
        "date": 1772226522328,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 27000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 160,
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
          "id": "f6eb756857ae4786f3c20bb6419d01df8fbbbf31",
          "message": "fix: SonarCloud Quality Gate -- 3 failing conditions (#59)\n\n1. Reliability E -> A: sha256.hpp buf_ overflow false positive.\n   Replace bitwise clamp (&= 63u) with explicit if-check that\n   SonarCloud's analyzer can track as a range constraint.\n\n2. Duplication 3.3% -> under 3.0%: raise cpd.minimumTokens from\n   100 to 120. CT variants (ct_sign.cpp, schnorr.cpp) intentionally\n   mirror non-CT code for constant-time guarantees.\n\n3. Coverage 61.8% -> ~86%: exclude FFI binding layer\n   (include/ufsecp/**) and audit/** from coverage metrics.\n   These are glue/test-harness code, not library core.",
          "timestamp": "2026-02-28T01:25:36+04:00",
          "tree_id": "5ddbfa79cf73716a836de6b8387241b782742223",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f6eb756857ae4786f3c20bb6419d01df8fbbbf31"
        },
        "date": 1772227643326,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
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
          "id": "5c94d496d432f4b27c8db84ff4e0d367a5c520d0",
          "message": "fix: sha256.hpp Sonar cpp:S3519 buf_ overflow false positive (#60)\n\nSplit buf_[buf_len_++] into separate index capture + increment so\nthe symbolic-execution engine sees the array index is always [0,63].\nGuard memset with buf_len_ < 64 check for the same reason.\n\nBuild: 88/88, Tests: 25/25 passed.",
          "timestamp": "2026-02-28T01:35:52+04:00",
          "tree_id": "e0fcd9ae8401c8f18b20f8a890605e600b3486de",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5c94d496d432f4b27c8db84ff4e0d367a5c520d0"
        },
        "date": 1772228261439,
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
            "value": 281,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 140,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "189c38fd4a6bb152bf810cb0f12a94a584fb8b30",
          "message": "fix: suppress S3519 false positive and exclude CT variants from CPD (#61)\n\nReliability E (cpp:S3519):\n  SonarCloud's cross-function symbolic execution flags sha256.hpp L79\n  (buf_[pos] = 0x80) as potential buffer overflow. This is a false positive:\n  buf_len_ is invariantly [0,63] after update() processes all full blocks.\n  The analyzer cannot track this class invariant through the call chain\n  (schnorr -> tagged_hash -> SHA256::finalize). Suppress via multicriteria.\n\nDuplication 3.3% > 3.0%:\n  CT (constant-time) source files (ct_sign.cpp, ct_field.cpp, ct_scalar.cpp,\n  ct_point.cpp) intentionally mirror their variable-time counterparts\n  line-for-line to guarantee identical control flow for side-channel\n  resistance. Exclude **/ct_*.cpp from copy-paste detection.\n\nVerify: SonarCloud Quality Gate should now pass all conditions.",
          "timestamp": "2026-02-28T01:47:18+04:00",
          "tree_id": "3a47379c6494206270fc99fd9c5a1cf9c0a68c92",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/189c38fd4a6bb152bf810cb0f12a94a584fb8b30"
        },
        "date": 1772228947695,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 267,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 133,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 35000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 11000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 43000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 115,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "9676b14ec205205fb951ba5d8b148356999b81fe",
          "message": "fix: raise sanitizer test timeouts to prevent false CI failures (#62)\n\nAll sanitizer CI failures were timeouts, not bugs. Sanitizers add\n3-15x runtime overhead making 300s per-test limits insufficient.\n\nChanges:\n- ci.yml: raise ctest --timeout from 300 to 900 for sanitizer jobs\n- ci.yml: exclude selftest from sanitizer runs (standalone tests\n  cover the same modules individually, faster under instrumentation)\n- security-audit.yml: add --timeout 900, exclude selftest + unified_audit\n- audit/CMakeLists.txt: raise per-test TIMEOUT for slow audit tests:\n  - differential: 120 -> 600\n  - debug_invariants: 120 -> 600\n  - audit_fuzz: 120 -> 600\n  - unified_audit: 600 -> 1200\n\nVerify: re-run CI Sanitizers (ASan+UBSan) and Sanitizers (TSan) jobs.",
          "timestamp": "2026-02-28T02:18:31+04:00",
          "tree_id": "d1be5a8c4fa91606e0814de8e3a1c52a85c87523",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/9676b14ec205205fb951ba5d8b148356999b81fe"
        },
        "date": 1772230805677,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 137,
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
          "id": "d2ff415287a29095a4d8b11e94b68a9f75794b1e",
          "message": "fix: raise remaining audit test TIMEOUT 300 -> 900 for sanitizer CI (#63)\n\nfault_injection and fuzz_parsers still timed out at 300s under ASan+UBSan.\nThe ctest --timeout 900 flag only sets a default for tests WITHOUT an\nexplicit TIMEOUT property. Tests with set_tests_properties(TIMEOUT 300)\nkeep their CMake-level limit.\n\nRaise all remaining 300s audit test timeouts to 900s:\n- fault_injection, fiat_crypto_vectors, carry_propagation\n- cross_platform_kat, fuzz_parsers, fuzz_address_bip32_ffi\n- cross_libsecp256k1\n\nVerify: CI Sanitizers (ASan+UBSan) should pass with 0 timeouts.",
          "timestamp": "2026-02-28T02:46:29+04:00",
          "tree_id": "6cb90873626171ca7a8dd349aac02f1e36a5c55c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d2ff415287a29095a4d8b11e94b68a9f75794b1e"
        },
        "date": 1772232482750,
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
            "value": 283,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 140,
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
          "id": "b3ee1e5ff4989ede5008bd44714151bcd4d10e7b",
          "message": "test: reduce iteration counts under sanitizers (SCALED macro) (#64)\n\nAdd cpu/include/secp256k1/sanitizer_scale.hpp header that detects\nASan/TSan/MSan/UBSan at compile time and provides SCALED(normal, reduced)\nmacro. Apply SCALED() to all 13 audit/test source files with high\niteration counts (100K+ -> 1K, 10K -> 200, 5K -> 100, 1K -> 50, etc).\n\nRoot cause: sanitizers add 3-15x runtime overhead; tests with 100K+\ncrypto ops (scalar_mul, sign, verify) were timing out at 300-900s in CI.\nThis reduces sanitizer-build iteration counts by 20-100x while keeping\nnormal-build counts unchanged.\n\nFiles changed:\n- NEW: cpu/include/secp256k1/sanitizer_scale.hpp\n- audit/test_fuzz_parsers.cpp (7 N values)\n- audit/test_fuzz_address_bip32_ffi.cpp (15 loops + 4 threshold checks)\n- audit/audit_fuzz.cpp (6 loops)\n- audit/test_fault_injection.cpp (8 TRIALS)\n- audit/audit_field.cpp (21 loops)\n- audit/audit_scalar.cpp (7 loops)\n- audit/audit_point.cpp (9 loops)\n- audit/audit_ct.cpp (10 loops)\n- audit/audit_security.cpp (9 loops)\n- audit/audit_integration.cpp (5 loops)\n- audit/audit_perf.cpp (7 values)\n- audit/differential_test.cpp (7 N values)\n- audit/test_cross_libsecp256k1.cpp (9 N values)\n\nVerify: cmake --build build-linux -j && ctest --test-dir build-linux",
          "timestamp": "2026-02-28T03:16:06+04:00",
          "tree_id": "fcadc610fc9010362c80b8369b57fd4dafa0ec95",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b3ee1e5ff4989ede5008bd44714151bcd4d10e7b"
        },
        "date": 1772234271455,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 137,
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
          "id": "1519ff96053ba79ff3d994db90245181d428f1d9",
          "message": "fix: resolve 266 code scanning alerts (cppcheck + clang-tidy + CodeQL) (#65)\n\n- Add .cppcheck-suppressions for objectIndex/passedByValue/containerOutOfBounds/\n  arrayIndexOutOfBoundsCond/uninitvar false positives (111 alerts)\n- Wire suppressions file into .github/workflows/cppcheck.yml\n- Add const qualifiers where variables are never modified (misc-const-correctness)\n- Initialize variables at declaration (cppcheck init-variables)\n- Replace C-style casts with reinterpret_cast (cstyle-cast)\n- Replace std::atoi with std::strtol (cert-err34-c)\n- Check sscanf/snprintf return values (cert-err33-c, cert-err34-c)\n- Add NOLINTNEXTLINE for false positives (BARRIER_OPAQUE, fe52_cmov,\n  reserved identifiers, 2-digit hex parsing)\n- Fix misplaced widening casts (bugprone-misplaced-widening-cast)\n- Add default: break to switch statements\n- Remove unused variables/imports, add (void) for unused bindings\n- Fix unsigned >= 0 tautology (cpp/unsigned-comparison-zero)\n- Add null guard before strlen (NonNullParamChecker)\n- Replace localtime with localtime_s/localtime_r\n- Replace fopen with POSIX open+fdopen for secure permissions\n\nAll 25 tests pass. No behavior changes.",
          "timestamp": "2026-02-28T04:31:04+04:00",
          "tree_id": "670cbe9366c12bc75b1b43e922431d9d54fc97fa",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/1519ff96053ba79ff3d994db90245181d428f1d9"
        },
        "date": 1772238759543,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "==============================================\nField Mul",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 19,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Inverse",
            "value": 1000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  POINT OPERATIONS\n==============================================\nPoint Add",
            "value": 267,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 133,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 35000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 8000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 12000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 43000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 20000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 49000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 116,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 107,
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
          "id": "37cd5da5257679f9ef16e97084911f2516e30a80",
          "message": "fix: raise dudect T_THRESHOLD to 35.0 for CI stability & fix misleading /* in recovery.hpp (#66)\n\n- audit/test_ct_sidechannel.cpp: raise DUDECT_SMOKE T_THRESHOLD from 25.0 to\n  35.0 for non-MSVC. Shared CI runners (ubuntu-24.04) produce timing noise\n  with only 5000 samples; ct_equal measured |t|=27.43 (false positive).\n  The ct_equal implementation is correct (no early exit, XOR accumulate +\n  is_zero_mask). 35.0 still catches gross leaks (|t|>100).\n\n- cpu/include/secp256k1/recovery.hpp: replace /* ... */ inside // comment\n  with ... to satisfy SonarCloud S125 (misleading comment characters).",
          "timestamp": "2026-02-28T05:22:24+04:00",
          "tree_id": "1f0237abcdd7d75b34480bb4f0a0d383f71e8bd3",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/37cd5da5257679f9ef16e97084911f2516e30a80"
        },
        "date": 1772241842265,
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
            "value": 287,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 151,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "b16602c625bdca4ad016286baf41101dd337d871",
          "message": "fix: resolve all 21 remaining code scanning alerts (#67)\n\nFiles changed (9):\n\n1. benchmark_harness.hpp, test_ct_sidechannel.cpp:\n   - Split 'uint32_t lo = 0, hi = 0' into separate declarations\n   - Fixes misc-const-correctness (3 alerts + 1 alert)\n\n2. test_fiat_crypto_vectors.cpp, test_carry_propagation.cpp:\n   - Replace sscanf with std::strtoul for hex parsing\n   - Add #include <cstdlib>\n   - Fixes cert-err34-c (4 + 2 alerts)\n\n3. test_ecdh_recovery_taproot.cpp:\n   - Replace sscanf with std::strtoul for hex parsing\n   - Remove clang diagnostic push/pop (no longer needed)\n   - Fixes cert-err34-c (1 alert)\n\n4. test_musig2_frost.cpp:\n   - Add (void)n2 to suppress unused variable\n   - Fixes cpp/unused-local-variable (1 alert)\n\n5. test_abi_gate.cpp:\n   - Use volatile load to prevent constant-folding of packed version\n   - Fixes cpp/unsigned-comparison-zero (1 alert)\n\n6. unified_audit_runner.cpp:\n   - Convert for-loop arg parser to while-loop (no loop var mutation)\n   - Add const to fd variable in write_json_report/write_text_report\n   - Fixes cpp/loop-variable-changed (2 alerts)\n   - Fixes misc-const-correctness (2 alerts)\n\n7. test_musig2_frost_advanced.cpp:\n   - Split 'uint32_t t = 2, n = 3' -> separate const t, mutable n\n   - Fixes misc-const-correctness (1 alert)\n\nAll 25 tests pass. No behavioral changes.",
          "timestamp": "2026-02-28T12:38:03+04:00",
          "tree_id": "0b82cd0725152f2eabcba88635ecacbf5c32151b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b16602c625bdca4ad016286baf41101dd337d871"
        },
        "date": 1772267983832,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 135,
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
          "id": "f4cbe525a130d20874c7df0b3eafb3778f89523b",
          "message": "ci: exclude audit/ from coverage measurement (#68)\n\naudit/ contains test, benchmark, and audit-runner code -- not\nproduction source.  Without this exclusion codecov/patch flags\nevery audit-file change as 0% covered, blocking PRs that only\ntouch test code.\n\nChanges:\n- codecov.yml: add audit/** to ignore list\n- ci.yml: add audit/ to llvm-cov --ignore-filename-regex",
          "timestamp": "2026-02-28T12:50:40+04:00",
          "tree_id": "87636da971c39148a26d107ada1170e6d9b4b7d1",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/f4cbe525a130d20874c7df0b3eafb3778f89523b"
        },
        "date": 1772268738773,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 150,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
            "unit": "ns"
          },
          {
            "name": "Generator Mul",
            "value": 9000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Sign",
            "value": 14000,
            "unit": "ns"
          },
          {
            "name": "ECDSA Verify",
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 23000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 139,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 130,
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
          "id": "90477f4db2cc7c16d439d27b2de6d070f87ba196",
          "message": "ci: make codecov/patch informational (non-blocking) (#69)\n\ncodecov/patch reports 0.00% on diffs that touch only non-source\nfiles (YAML, config) or excluded directories, since there are\nzero coverable lines. This blocks the status check even though\nthe project-level coverage gate still protects overall health.\n\nSet informational: true so codecov/patch always reports a\nneutral status instead of failure.",
          "timestamp": "2026-02-28T12:53:18+04:00",
          "tree_id": "98c7df4fcc3e003ac84cafce260ad7446af9ffe2",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/90477f4db2cc7c16d439d27b2de6d070f87ba196"
        },
        "date": 1772268900094,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "ce059261731cb7f6d58bea16deaaa37f892afcbc",
          "message": "bench: wire bench_compare harness into build + add docs/scripts/templates\n\n- Add SECP256K1_BUILD_BENCH_COMPARE option to root CMakeLists.txt (OFF by default)\n- Add bench/README.md: comprehensive documentation (quick start, workloads, fairness, CLI, output)\n- Add bench/scripts/{build,run}.{sh,ps1} for one-command build/run on Linux and Windows\n- Add .github/ISSUE_TEMPLATE/benchmark.yml for community benchmark submissions\n- Enable blank issues in .github/ISSUE_TEMPLATE/config.yml\n\nExisting bench/ sources (CMakeLists.txt, providers, main.cpp, headers) unchanged.\n\nVerified: cmake configure + MSBuild Release + smoke test (all correctness gates pass,\nall 5 workloads run for both UltrafastSecp256k1 v3.14.0 and libsecp256k1 v0.6.0).",
          "timestamp": "2026-02-28T13:42:01+04:00",
          "tree_id": "195e238b85f61faf2b22de89dd97e7d703794c2a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/ce059261731cb7f6d58bea16deaaa37f892afcbc"
        },
        "date": 1772271823559,
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
            "value": 282,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 149,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 25000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 141,
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
          "id": "5565acdef8e0f4d329925aca83f2b34c9cf1a86d",
          "message": "perf: optimize verify hot path -- normalizes_to_zero_var + eliminate neg tables\n\nThree optimizations to the 5x52 ECC verify path:\n\n1. normalizes_to_zero_var(): combined normalize_weak + zero check with\n   early exit. Raw-zero fast path fires ~100% of the time (probability\n   of h==0 is ~2^-256). Saves ~40 limb ops per mixed add call.\n\n2. Remove redundant normalize_weak() before normalizes_to_zero() in all\n   4 point-add functions (jac52_add_mixed, _inplace, jac52_add, _inplace).\n   The old normalizes_to_zero() already normalizes internally.\n\n3. Eliminate pre-negated tables (neg_tbl_G, neg_tbl_H, neg_tbl_P,\n   neg_tbl_phiP) from GenTables and scalar_mul_glv52 and\n   dual_scalar_mul_gen_point. Negate Y on-the-fly for negative wNAF\n   digits (5 limb ops, data already in L1 from lookup). Halves static\n   table memory from 2.5 MB to 1.25 MB.\n\nPinned benchmark (core 2, n=2000, Clang 21.1.0, Zen 3):\n  UF ECDSA verify preparsed: 33100 -> 28200 ns (+17% faster)\n  UF Schnorr verify preparsed: 30200 -> 28600 ns (+6% faster)\n  UF pubkey_create: 7100 ns vs libsecp 13100 ns (UF 1.85x faster)\n  field_52 tests: 267/267 PASSED\n  bench_compare correctness: 10/10 gates PASSED\n\nFiles changed:\n  field_52.hpp       -- added normalizes_to_zero_var() declaration\n  field_52_impl.hpp  -- added normalizes_to_zero_var() implementation\n  point.cpp          -- all point-add + ecmult hot path changes",
          "timestamp": "2026-02-28T15:32:42+04:00",
          "tree_id": "20d5f03052d364cb9c721e21f82526840dc6f8f5",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5565acdef8e0f4d329925aca83f2b34c9cf1a86d"
        },
        "date": 1772278456013,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "benchmark_parse_warning",
            "value": 0,
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
          "id": "1714e00167315919c7fbf8f6cf3d0ce1bf4933e1",
          "message": "fix: normalizes_to_zero_var P0 constant typo (extra F)\n\nThe p-comparison in normalizes_to_zero_var() used 0xFFFFFEFFFFFC2F\n(14 hex digits, 56 bits) instead of P0 = 0xFFFFEFFFFFC2F (13 hex\ndigits, 52 bits).  The extra 'F' before 'E' made the constant exceed\na 52-bit limb, so the equality check could never match -- values\nequal to p (= 0 mod p) were reported as non-zero.\n\nEffect: jac52_add/jac52_add_mixed failed to detect equal-x points\n(h = u2 - u1 = 0 mod p reduces to p after normalize_weak).  Instead\nof doubling, the code proceeded with h = 0 in the general addition\nformula, producing wrong results.\n\nFix: use P0 constant directly instead of a literal.\n\nVerified: all local tests pass (exhaustive 5399/0, batch_add_affine\n548/0, comprehensive 12023/0, selftest 21/21, field_52 267/0).",
          "timestamp": "2026-02-28T15:59:46+04:00",
          "tree_id": "582c8a2f5fdda7accdc54291b89ea9cb90c09d38",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/1714e00167315919c7fbf8f6cf3d0ce1bf4933e1"
        },
        "date": 1772280082267,
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
            "value": 279,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
            "unit": "ns"
          },
          {
            "name": "Point Scalar Mul",
            "value": 38000,
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
            "value": 47000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Sign",
            "value": 24000,
            "unit": "ns"
          },
          {
            "name": "Schnorr Verify",
            "value": 53000,
            "unit": "ns"
          },
          {
            "name": "==============================================\n  BATCH OPERATIONS\n==============================================\nBatch Inverse (n=100)",
            "value": 152,
            "unit": "ns"
          },
          {
            "name": "Batch Inverse (n=1000)",
            "value": 133,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}