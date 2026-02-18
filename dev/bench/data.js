window.BENCHMARK_DATA = {
  "lastUpdate": 1771420766329,
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
          "id": "b80fa5cfe9c8e0472f85c6978338e0af172c5030",
          "message": "metal: skip generator_mul test on non-Apple7+ (paravirtual) devices\n\nThe generator_mul_batch kernel (16-entry precompute table + 13 field\ninversions in scalar_mul) exceeds the capabilities of CI's Apple\nParavirtual GPU device.  Pipeline creation fails with 'Compilation\nfailed' at newComputePipelineStateWithFunction: stage.\n\nSkip test_generator_mul and bench_scalar_mul on devices that don't\nsupport MTLGPUFamilyApple7 (M1+).  The field_mul test still runs\nand validates GPU compute correctness on paravirtual devices.\n\nOn real Apple Silicon (M1/M2/M3/M4), both tests run normally.",
          "timestamp": "2026-02-16T08:09:59+04:00",
          "tree_id": "4165df751a482f11dfe86854ed36ba4f52f13c81",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b80fa5cfe9c8e0472f85c6978338e0af172c5030"
        },
        "date": 1771215064674,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 992,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 573,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "b80fa5cfe9c8e0472f85c6978338e0af172c5030",
          "message": "metal: skip generator_mul test on non-Apple7+ (paravirtual) devices\n\nThe generator_mul_batch kernel (16-entry precompute table + 13 field\ninversions in scalar_mul) exceeds the capabilities of CI's Apple\nParavirtual GPU device.  Pipeline creation fails with 'Compilation\nfailed' at newComputePipelineStateWithFunction: stage.\n\nSkip test_generator_mul and bench_scalar_mul on devices that don't\nsupport MTLGPUFamilyApple7 (M1+).  The field_mul test still runs\nand validates GPU compute correctness on paravirtual devices.\n\nOn real Apple Silicon (M1/M2/M3/M4), both tests run normally.",
          "timestamp": "2026-02-16T08:09:59+04:00",
          "tree_id": "4165df751a482f11dfe86854ed36ba4f52f13c81",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b80fa5cfe9c8e0472f85c6978338e0af172c5030"
        },
        "date": 1771215064698,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 48,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 44,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 17,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 12,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 819,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 469,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "7151b14a7ca273d4f35b215b3826c40f7a5ef71e",
          "message": "bench: comprehensive Metal + WASM benchmarks\n\nMetal:\n  - Add field_add/sub/inv GPU kernels to secp256k1_kernels.metal\n  - Create bench_metal.mm: full benchmark (field mul/add/sub/sqr/inv,\n    point add/double, scalar_mul, generator_mul) matching CUDA format\n  - Wire metal_secp256k1_bench_full target in CMakeLists.txt\n  - Simplify metal_test.mm: remove inline scalar bench, redirect to bench_full\n\nWASM:\n  - Create bench_wasm.mjs: Node.js benchmark for all WASM operations\n    (pubkeyCreate, pointMul, pointAdd, ecdsaSign/Verify, schnorrSign/Verify,\n    SHA-256) with warmup, timing, and throughput table output",
          "timestamp": "2026-02-16T08:21:29+04:00",
          "tree_id": "dbce6be03a4ecc9c0369ffd4dbfe56aa9f437cec",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7151b14a7ca273d4f35b215b3826c40f7a5ef71e"
        },
        "date": 1771215751770,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 998,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 576,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "7151b14a7ca273d4f35b215b3826c40f7a5ef71e",
          "message": "bench: comprehensive Metal + WASM benchmarks\n\nMetal:\n  - Add field_add/sub/inv GPU kernels to secp256k1_kernels.metal\n  - Create bench_metal.mm: full benchmark (field mul/add/sub/sqr/inv,\n    point add/double, scalar_mul, generator_mul) matching CUDA format\n  - Wire metal_secp256k1_bench_full target in CMakeLists.txt\n  - Simplify metal_test.mm: remove inline scalar bench, redirect to bench_full\n\nWASM:\n  - Create bench_wasm.mjs: Node.js benchmark for all WASM operations\n    (pubkeyCreate, pointMul, pointAdd, ecdsaSign/Verify, schnorrSign/Verify,\n    SHA-256) with warmup, timing, and throughput table output",
          "timestamp": "2026-02-16T08:21:29+04:00",
          "tree_id": "dbce6be03a4ecc9c0369ffd4dbfe56aa9f437cec",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7151b14a7ca273d4f35b215b3826c40f7a5ef71e"
        },
        "date": 1771215756905,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 577,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "21e08131286ad695e9ae853b15bf6982d78d960e",
          "message": "ci: run WASM benchmark in CI after build\n\nAdd Node.js 20 setup + bench_wasm.mjs execution to the WASM CI job.\nAlso fix module resolution for running from build-wasm/dist/ directory.",
          "timestamp": "2026-02-16T08:26:24+04:00",
          "tree_id": "04163e6f2ca4a44c9229b939a604d7282239da1a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/21e08131286ad695e9ae853b15bf6982d78d960e"
        },
        "date": 1771216047998,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 997,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 575,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "21e08131286ad695e9ae853b15bf6982d78d960e",
          "message": "ci: run WASM benchmark in CI after build\n\nAdd Node.js 20 setup + bench_wasm.mjs execution to the WASM CI job.\nAlso fix module resolution for running from build-wasm/dist/ directory.",
          "timestamp": "2026-02-16T08:26:24+04:00",
          "tree_id": "04163e6f2ca4a44c9229b939a604d7282239da1a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/21e08131286ad695e9ae853b15bf6982d78d960e"
        },
        "date": 1771216048725,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 994,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 574,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "eaa7ebd653ead5bf0daf584e2b80e6b4627be381",
          "message": "ci: raise benchmark alert threshold to 150%\n\nCI VMs have ~20-30% noise between runs. 120% threshold produces\nfalse positive alerts. 150% is more appropriate for shared runners.",
          "timestamp": "2026-02-16T08:29:21+04:00",
          "tree_id": "a09000e14025abb6b2df1804151d6c50eadd14ac",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/eaa7ebd653ead5bf0daf584e2b80e6b4627be381"
        },
        "date": 1771216222304,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 60,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 998,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 578,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "eaa7ebd653ead5bf0daf584e2b80e6b4627be381",
          "message": "ci: raise benchmark alert threshold to 150%\n\nCI VMs have ~20-30% noise between runs. 120% threshold produces\nfalse positive alerts. 150% is more appropriate for shared runners.",
          "timestamp": "2026-02-16T08:29:21+04:00",
          "tree_id": "a09000e14025abb6b2df1804151d6c50eadd14ac",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/eaa7ebd653ead5bf0daf584e2b80e6b4627be381"
        },
        "date": 1771216223269,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 576,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "c6915a0f50f8448e308299874d297ed5f542cbfe",
          "message": "fix(wasm): fix runtime crash in WASM benchmark\n\n- Remove --closure 1: Closure compiler breaks Emscripten's atexit/\n  exception handling runtime (null .push TypeError)\n- Add -fno-exceptions: not needed by this library, avoids broken\n  exception stubs\n- Increase INITIAL_MEMORY to 4MB and STACK_SIZE to 512KB: selftest\n  uses std::vector which needs more memory\n- Make selftest non-fatal in benchmark (try/catch + continue)",
          "timestamp": "2026-02-16T08:34:04+04:00",
          "tree_id": "6defe0c5e0779a4b0014ca3a3a87e76afaabfefc",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c6915a0f50f8448e308299874d297ed5f542cbfe"
        },
        "date": 1771216504662,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 48,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 44,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 17,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 12,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 820,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 469,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "c6915a0f50f8448e308299874d297ed5f542cbfe",
          "message": "fix(wasm): fix runtime crash in WASM benchmark\n\n- Remove --closure 1: Closure compiler breaks Emscripten's atexit/\n  exception handling runtime (null .push TypeError)\n- Add -fno-exceptions: not needed by this library, avoids broken\n  exception stubs\n- Increase INITIAL_MEMORY to 4MB and STACK_SIZE to 512KB: selftest\n  uses std::vector which needs more memory\n- Make selftest non-fatal in benchmark (try/catch + continue)",
          "timestamp": "2026-02-16T08:34:04+04:00",
          "tree_id": "6defe0c5e0779a4b0014ca3a3a87e76afaabfefc",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c6915a0f50f8448e308299874d297ed5f542cbfe"
        },
        "date": 1771216509221,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 992,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 575,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "7186c2972dd0541c1a3e98af9ec96ab2e7cc9bfb",
          "message": "Merge dev: v3.3.0 — Metal/WASM benchmarks, CI hardening, security docs",
          "timestamp": "2026-02-16T08:39:16+04:00",
          "tree_id": "6defe0c5e0779a4b0014ca3a3a87e76afaabfefc",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/7186c2972dd0541c1a3e98af9ec96ab2e7cc9bfb"
        },
        "date": 1771216821409,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 575,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "60f6e85e63c04dfd86641d51c921cb8c7018bc9c",
          "message": "docs: v3.3.0 community & visibility improvements\n\n- README: Add Apple Silicon/Metal/CI/Bench/Release badges, update to v3.3.0\n- CHANGELOG: Add comprehensive v3.3.0 release notes\n- Issue templates: Crypto-specific bug report & feature request fields\n- Issue config: Add Security, Benchmark Dashboard, Build Guide links\n- CMakeLists: Bump VERSION 3.2.0 -> 3.3.0",
          "timestamp": "2026-02-16T09:06:13+04:00",
          "tree_id": "4a8ddd81981f32934ae57c905bcf95a50c335931",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/60f6e85e63c04dfd86641d51c921cb8c7018bc9c"
        },
        "date": 1771218438656,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 990,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 574,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "60f6e85e63c04dfd86641d51c921cb8c7018bc9c",
          "message": "docs: v3.3.0 community & visibility improvements\n\n- README: Add Apple Silicon/Metal/CI/Bench/Release badges, update to v3.3.0\n- CHANGELOG: Add comprehensive v3.3.0 release notes\n- Issue templates: Crypto-specific bug report & feature request fields\n- Issue config: Add Security, Benchmark Dashboard, Build Guide links\n- CMakeLists: Bump VERSION 3.2.0 -> 3.3.0",
          "timestamp": "2026-02-16T09:06:13+04:00",
          "tree_id": "4a8ddd81981f32934ae57c905bcf95a50c335931",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/60f6e85e63c04dfd86641d51c921cb8c7018bc9c"
        },
        "date": 1771218442015,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 48,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 44,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 17,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 12,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 818,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 469,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "d980f11729931294554ac1d56cb8ec07f375f1d6",
          "message": "fix: restore Stars/Forks badges and reorganize badge layout",
          "timestamp": "2026-02-16T09:12:57+04:00",
          "tree_id": "4214de35d8d7d3266a6677082025bd9d15cfa379",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d980f11729931294554ac1d56cb8ec07f375f1d6"
        },
        "date": 1771218841223,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 994,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 575,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "d980f11729931294554ac1d56cb8ec07f375f1d6",
          "message": "fix: restore Stars/Forks badges and reorganize badge layout",
          "timestamp": "2026-02-16T09:12:57+04:00",
          "tree_id": "4214de35d8d7d3266a6677082025bd9d15cfa379",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d980f11729931294554ac1d56cb8ec07f375f1d6"
        },
        "date": 1771218845082,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 996,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 578,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "205a756e1e290e865d266c93e15d3f9943945da3",
          "message": "fix: switch docs deployment to gh-pages branch\n\ndocs.yml was using actions/deploy-pages@v4 (workflow mode) which\nconflicted with benchmark.yml pushing to gh-pages branch (legacy mode).\nNow both docs and benchmark push to gh-pages branch:\n- docs -> /docs/ subdirectory\n- benchmark -> /dev/bench/ subdirectory\nNo more environment protection rule conflicts.",
          "timestamp": "2026-02-16T09:14:44+04:00",
          "tree_id": "28fdd14c229b3b88114f2df8859a7c395a511550",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/205a756e1e290e865d266c93e15d3f9943945da3"
        },
        "date": 1771218944660,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 576,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "205a756e1e290e865d266c93e15d3f9943945da3",
          "message": "fix: switch docs deployment to gh-pages branch\n\ndocs.yml was using actions/deploy-pages@v4 (workflow mode) which\nconflicted with benchmark.yml pushing to gh-pages branch (legacy mode).\nNow both docs and benchmark push to gh-pages branch:\n- docs -> /docs/ subdirectory\n- benchmark -> /dev/bench/ subdirectory\nNo more environment protection rule conflicts.",
          "timestamp": "2026-02-16T09:14:44+04:00",
          "tree_id": "28fdd14c229b3b88114f2df8859a7c395a511550",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/205a756e1e290e865d266c93e15d3f9943945da3"
        },
        "date": 1771218952033,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 22,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 986,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 576,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "5c987154e35d230d2b1179fd2af676691a5a28ec",
          "message": "feat: add FUNDING.yml, polish README contact section with Wiki/Discussions/Sponsor links",
          "timestamp": "2026-02-16T09:24:20+04:00",
          "tree_id": "25b5f697bb399aeb1e7b765af5688771069f93cd",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5c987154e35d230d2b1179fd2af676691a5a28ec"
        },
        "date": 1771219524960,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 998,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 580,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "5c987154e35d230d2b1179fd2af676691a5a28ec",
          "message": "feat: add FUNDING.yml, polish README contact section with Wiki/Discussions/Sponsor links",
          "timestamp": "2026-02-16T09:24:20+04:00",
          "tree_id": "25b5f697bb399aeb1e7b765af5688771069f93cd",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5c987154e35d230d2b1179fd2af676691a5a28ec"
        },
        "date": 1771219526513,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 995,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 574,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "c068e1d011b8647d91d7df5f185548ba3bf83432",
          "message": "fix: Apple9+ GPU family detection + add Metal benchmark results\n\n- Fix MTLGPUFamilyApple9 detection (enum, not macro — use SDK version guard)\n- Add Metal (Apple M3 Pro) benchmarks to README.md and docs/BENCHMARKS.md\n- Add Metal vs CUDA vs OpenCL comparison table\n- Mark Metal backend as done in Future Optimizations",
          "timestamp": "2026-02-16T13:52:06+04:00",
          "tree_id": "e153e1f9e1aef40db3584ff6fd048319d3db4c5a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c068e1d011b8647d91d7df5f185548ba3bf83432"
        },
        "date": 1771235706752,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 22,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 991,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 575,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "079cf9f9e77f498f6691834229976c2d57041545",
          "message": "fix(ci): move immintrin.h include to file scope to fix Linux builds\n\nOn Linux GCC/Clang, immintrin.h transitively includes <stdlib.h> via\nmm_malloc.h. When included inside namespace secp256k1::hash{}, this\ncaused stdlib symbols (malloc, calloc, abs, etc.) to be declared in the\nwrong namespace, breaking <cstdlib> and producing hundreds of errors.\n\nMove the #include <immintrin.h> from inside the namespace block (line 390)\nto file scope alongside other system headers.",
          "timestamp": "2026-02-16T19:18:46+04:00",
          "tree_id": "539ce87c7104faa8f7c595a4dec1c19a3192f414",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/079cf9f9e77f498f6691834229976c2d57041545"
        },
        "date": 1771255193419,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 48,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 44,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 17,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 12,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 851,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 467,
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
          "id": "a4e6cd462e41986c19eb073b3550839b2ff97c29",
          "message": "Merge remote-tracking branch 'origin/dev' into dev",
          "timestamp": "2026-02-16T16:55:37Z",
          "tree_id": "c601ae06f81920a58af0208e68a934502df81cd7",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a4e6cd462e41986c19eb073b3550839b2ff97c29"
        },
        "date": 1771261012735,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 981,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 577,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "d2f4ffa716286af41ea4a9062abb780319faf1d8",
          "message": "Merge dev into main: batch affine ops (CPU/CUDA/Metal/OpenCL), hash_accel, CI fixes",
          "timestamp": "2026-02-16T21:00:27+04:00",
          "tree_id": "c601ae06f81920a58af0208e68a934502df81cd7",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d2f4ffa716286af41ea4a9062abb780319faf1d8"
        },
        "date": 1771261303491,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 56,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 571,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "995804e80cc26a498e9941ea5818eccb5b9fb6e3",
          "message": "feat(bindings): add 12-language binding suite + CI workflow\n\nLanguage bindings for the C API shared library:\n- C API: shared lib (libultrafast_secp256k1) with ABI-stable header\n- Python: ctypes FFI wrapper (PyPI-ready)\n- C#: P/Invoke (.NET 6/7/8 multi-target)\n- Rust: sys crate (raw FFI) + safe wrapper (edition 2024)\n- Node.js: ffi-napi + TypeScript declarations\n- PHP: FFI class (PHP 8.1+, Composer/PSR-4)\n- Go: cgo wrapper (go 1.21+, typed [32]byte arrays)\n- Java: JNI bridge (C impl + Java class + CMake)\n- Swift: SPM package (macOS 12+ / iOS 15+, Foundation Data)\n- React Native: NativeModules (Android Java + iOS ObjC + JS/TS)\n- Ruby: FFI gem wrapper (Ruby 3.0+)\n- Dart: dart:ffi bindings (Dart 3.0+, DynamicLibrary)\n\nCI: add bindings.yml GitHub Actions workflow\n- Builds C API on Linux/macOS/Windows\n- Compile-checks each binding (syntax, type-check, lint)\n- Summary matrix for all 12 languages\n\nUpdate .gitignore for benchmark outputs + binding build artifacts",
          "timestamp": "2026-02-18T13:55:27+04:00",
          "tree_id": "80660c451daebd42ba6b218ad0e79e00fe4130d5",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/995804e80cc26a498e9941ea5818eccb5b9fb6e3"
        },
        "date": 1771410284450,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 572,
            "unit": "ns"
          },
          {
            "name": "10x26 Mul",
            "value": 50,
            "unit": "ns"
          },
          {
            "name": "10x26 Sqr",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "10x26 Add",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "10x26 Neg",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "5x52 Mul",
            "value": 29,
            "unit": "ns"
          },
          {
            "name": "5x52 Sqr",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "5x52 Add",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "Optimal Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "Optimal Sqr",
            "value": 28,
            "unit": "ns"
          },
          {
            "name": "Optimal Add",
            "value": 6,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "83139bd1638e14bd57aade607711043352e5ed35",
          "message": "fix(c_api): implement modular sqrt + MSVC standalone build\n\n- Replace non-existent FieldElement::sqrt() with addition-chain\n  exponentiation: y = y2^((p+1)/4) using the standard secp256k1\n  chain (x2→x3→x6→x9→x11→x22→x44→x88→x176→x220→x223→result).\n- Replace non-existent FieldElement::negate() with (zero - y).\n- Guard #define ULTRAFAST_SECP256K1_BUILDING with #ifndef.\n- CMakeLists standalone build: exclude field_52.cpp on MSVC (hard\n  #error without __uint128_t) and define SECP256K1_NO_INT128 so\n  scalar.cpp uses the 32-bit fallback multiply.\n\nFixes CI build-capi failures on all three platforms.",
          "timestamp": "2026-02-18T15:24:54+04:00",
          "tree_id": "ae64da25f6bd7058e7e63eb7bb76371abd82081f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/83139bd1638e14bd57aade607711043352e5ed35"
        },
        "date": 1771413978372,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 573,
            "unit": "ns"
          },
          {
            "name": "10x26 Mul",
            "value": 50,
            "unit": "ns"
          },
          {
            "name": "10x26 Sqr",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "10x26 Add",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "10x26 Neg",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "5x52 Mul",
            "value": 29,
            "unit": "ns"
          },
          {
            "name": "5x52 Sqr",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "5x52 Add",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "Optimal Mul",
            "value": 30,
            "unit": "ns"
          },
          {
            "name": "Optimal Sqr",
            "value": 28,
            "unit": "ns"
          },
          {
            "name": "Optimal Add",
            "value": 6,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "2e8e8bfb7562436686c0d9fb6a5758f0531e65ce",
          "message": "fix(ci): fix Python/Rust/Java binding CI failures\n\n- Python: remove unused 'import sys' and unused 'c'/'p' locals (pyflakes)\n- Rust: edition 2024 → 2021 (wider CI toolchain compat)\n- Java: generate JNI header via javac -h; use gcc -fsyntax-only\n  instead of cmake build (avoids needing linked C API lib)",
          "timestamp": "2026-02-18T15:32:05+04:00",
          "tree_id": "5aea11752e62ac182e58bc94d864c25ecb5663ee",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/2e8e8bfb7562436686c0d9fb6a5758f0531e65ce"
        },
        "date": 1771414402397,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 58,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 990,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 578,
            "unit": "ns"
          },
          {
            "name": "10x26 Mul",
            "value": 50,
            "unit": "ns"
          },
          {
            "name": "10x26 Sqr",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "10x26 Add",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "10x26 Neg",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "5x52 Mul",
            "value": 29,
            "unit": "ns"
          },
          {
            "name": "5x52 Sqr",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "5x52 Add",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "Optimal Mul",
            "value": 39,
            "unit": "ns"
          },
          {
            "name": "Optimal Sqr",
            "value": 30,
            "unit": "ns"
          },
          {
            "name": "Optimal Add",
            "value": 6,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "92eed013aa3c2e12f5541faa3d94006294540081",
          "message": "fix(ci): MSVC field_52 uint128 guard + Metal shader address space and signature fixes\n\n- field_52.cpp: replace #error with #ifdef SECP256K1_HAS_UINT128 guard (compiles as empty TU on MSVC)\n- CMakeLists.txt: skip bench_field_52 and test_field_52_standalone on MSVC (no-op without uint128)\n- secp256k1_extended.h: fix tagged_hash constant->thread address space casts (use thread-local arrays)\n- secp256k1_extended.h: add adapter overloads for batch kernel calling conventions\n  (ecdsa_sign/verify, schnorr_sign/verify, ecdsa_recover, ecdh_shared_secret_xonly)\n\nFixes: Windows CI error C1189 (#error on MSVC), macOS CI 12 Metal shader compilation errors",
          "timestamp": "2026-02-18T16:43:06+04:00",
          "tree_id": "7d66ff9afed97f12754557ec83f36d7286d25bdd",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/92eed013aa3c2e12f5541faa3d94006294540081"
        },
        "date": 1771418673058,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 55,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 21,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 16,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 594,
            "unit": "ns"
          },
          {
            "name": "10x26 Mul",
            "value": 50,
            "unit": "ns"
          },
          {
            "name": "10x26 Sqr",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "10x26 Add",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "10x26 Neg",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "5x52 Mul",
            "value": 29,
            "unit": "ns"
          },
          {
            "name": "5x52 Sqr",
            "value": 27,
            "unit": "ns"
          },
          {
            "name": "5x52 Add",
            "value": 6,
            "unit": "ns"
          },
          {
            "name": "Optimal Mul",
            "value": 40,
            "unit": "ns"
          },
          {
            "name": "Optimal Sqr",
            "value": 30,
            "unit": "ns"
          },
          {
            "name": "Optimal Add",
            "value": 6,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "e63ab1ed9765c7ed6f4569c4820d51b762d7ffa2",
          "message": "docs: add head-to-head libsecp256k1 comparison + all benchmark targets\n\n- Run all 12 bench targets on Windows x64 (Clang 21.1.0, Release, AVX2)\n- Build and run Bitcoin Core libsecp256k1 with same compiler/flags\n- README: updated x86_64/Windows table with 5x52/4x64 field data\n- README: added scalar mul breakdown, field representation comparison,\n  CT layer overhead, and head-to-head comparison section\n- README: added 12 available benchmark targets with descriptions\n- docs/BENCHMARKS.md: replaced placeholder comparison with real data\n  (field/point/scalar/high-level ops side-by-side)\n- docs/BENCHMARKS.md: added x86-64 Windows section, specialized\n  benchmark results (5x52 vs 4x64, CT overhead, multi-scalar,\n  atomic building blocks)\n- docs/BENCHMARKS.md: added available benchmark targets catalog\n\nKey findings:\n  - Generator Mul (kxG): 8.5 us vs 15.3 us (1.8x faster)\n  - ECDSA Sign path: 8.5 us vs 26.2 us (3.1x faster)\n  - Field Mul (5x52): 22 ns vs 15.3 ns (1.44x slower)\n  - Point Add: 937 ns vs 255 ns (3.67x slower)\n  - ECDSA Verify: 137 us vs 37.3 us (3.7x slower)",
          "timestamp": "2026-02-18T17:18:05+04:00",
          "tree_id": "00c54bda15ae21e96a5959242d191eabf7f597b5",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/e63ab1ed9765c7ed6f4569c4820d51b762d7ffa2"
        },
        "date": 1771420765132,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 48,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 43,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 17,
            "unit": "ns"
          },
          {
            "name": "Field Sub",
            "value": 12,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 818,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 467,
            "unit": "ns"
          },
          {
            "name": "10x26 Mul",
            "value": 48,
            "unit": "ns"
          },
          {
            "name": "10x26 Sqr",
            "value": 34,
            "unit": "ns"
          },
          {
            "name": "10x26 Add",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "10x26 Neg",
            "value": 4,
            "unit": "ns"
          },
          {
            "name": "5x52 Mul",
            "value": 31,
            "unit": "ns"
          },
          {
            "name": "5x52 Sqr",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "5x52 Add",
            "value": 3,
            "unit": "ns"
          },
          {
            "name": "Optimal Mul",
            "value": 32,
            "unit": "ns"
          },
          {
            "name": "Optimal Sqr",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Optimal Add",
            "value": 4,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}