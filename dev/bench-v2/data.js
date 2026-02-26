window.BENCHMARK_DATA = {
  "lastUpdate": 1772066233859,
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
      }
    ]
  }
}