window.BENCHMARK_DATA = {
  "lastUpdate": 1771852856801,
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
      }
    ]
  }
}