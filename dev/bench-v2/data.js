window.BENCHMARK_DATA = {
  "lastUpdate": 1771809354840,
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
      }
    ]
  }
}