window.BENCHMARK_DATA = {
  "lastUpdate": 1771870452678,
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
      }
    ]
  }
}