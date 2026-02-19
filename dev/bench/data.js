window.BENCHMARK_DATA = {
  "lastUpdate": 1771539525293,
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
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "563016f8b9ecfc1df1541af8a96917a496f264b4",
          "message": "perf: GLV + 5x52 + Shamir scalar_mul — K*Q 132→42μs (3.1× faster)\n\nDesktop scalar_mul for arbitrary points now uses:\n- GLV endomorphism: splits 256-bit scalar into two ~128-bit halves\n- 5×52 lazy-reduction field arithmetic (__int128, ~2.5× faster than 4×64)\n- Shamir's trick: interleaved double-and-add for k1*P + k2*φ(P)\n- wNAF with trailing-zero trimming (128 vs 257 iterations)\n- Montgomery batch inversion for affine precomputed table\n- Lazy magnitude tracking: removed all normalize_weak() from hot path\n\nKey results (Windows x64, Clang 21.1.0, LTO):\n  K*Q:          132 μs → 42 μs  (3.1× faster)\n  K*G:          10 μs → 8 μs    (25% faster)\n  ECDSA Verify: 137 μs → 50 μs  (2.7× faster)\n\nGap vs libsecp256k1:\n  K*Q:          3.7× slower → 1.3× slower\n  K*G:          1.3× slower → 1.05× slower\n\nImplementation notes:\n- scalar_mul_glv52() is a separate noinline function to prevent\n  stack buffer overrun (Clang 21 code generation issue with inlining)\n- try/catch wrapper required for correct SEH stack layout on Windows\n- Fallback to 4×64 wNAF w=5 when __int128 is unavailable\n- ESP32/STM32 paths unchanged (no __int128 on those targets)\n\nAll 12023 comprehensive tests pass.",
          "timestamp": "2026-02-18T19:07:16+04:00",
          "tree_id": "6482f11ce4357b5cab743b81c5118eb53f50386d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/563016f8b9ecfc1df1541af8a96917a496f264b4"
        },
        "date": 1771427326568,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 59,
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
          "id": "dd07e32768ef4dedb949755129b177b300d2e327",
          "message": "fix: build system (PIC, install guard) and Dart lint cleanup\n\n- cpu/CMakeLists.txt: POSITION_INDEPENDENT_CODE ON for shared lib linking\n- include/ufsecp/CMakeLists.txt: install(EXPORT) guard for sub-project use\n- bindings/dart: remove unused typedefs, add errorString(), remove dead finalizer\n- dart analyze --fatal-infos passes clean",
          "timestamp": "2026-02-19T16:23:19Z",
          "tree_id": "deca4c5a260edbbebc81821839b6ff9bfffb36a0",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/dd07e32768ef4dedb949755129b177b300d2e327"
        },
        "date": 1771518272015,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 28,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 26,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 283,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 163,
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
          "id": "75536e59645bd663d82e4aa7ab6a5a2e1efe7842",
          "message": "GPU: port CPU signature/verify optimizations to CUDA\n\nMajor CUDA verification/signing performance optimizations:\n\nsecp256k1.cuh:\n- Add scalar_to_wnaf4: wNAF w=4 signed-digit encoding\n- Add scalar_mul_wnaf: wNAF scalar mul with 8 precomputed affine odd multiples\n- Add scalar_mul_glv_wnaf: GLV + Shamir bit interleaving (~128 doublings)\n- Add shamir_double_mul: basic a*P + b*Q with 4-combo table\n- Add shamir_double_mul_glv: 4-way GLV decomposition for a*P + b*Q\n  (decomposes both scalars, single ~128-bit doubling chain)\n- Add GENERATOR_TABLE_AFFINE[16] in __constant__ memory (Python-verified i*G)\n- Add scalar_mul_generator_const: w=4 fixed-window generator mul using\n  __constant__ table (no shared mem / __syncthreads needed)\n\necdsa.cuh:\n- ecdsa_verify: replace 2x scalar_mul + add with shamir_double_mul_glv\n  (~2x fewer doublings, single pass)\n- ecdsa_sign: replace scalar_mul with scalar_mul_generator_const\n  (precomputed table, no runtime table build)\n\nschnorr.cuh:\n- Add BIP340_MIDSTATES[3][8] __constant__: precomputed SHA-256 midstates\n  for BIP0340/aux, BIP0340/nonce, BIP0340/challenge tags\n- Add tagged_hash_fast: skips 2 SHA-256 compressions per call\n- schnorr_sign: use scalar_mul_generator_const + tagged_hash_fast (3 calls)\n- schnorr_verify: replace 2x scalar_mul + negate + add with\n  shamir_double_mul_glv + tagged_hash_fast\n\nAll 52 CUDA tests pass, 12/12 CTest pass.",
          "timestamp": "2026-02-19T16:54:41Z",
          "tree_id": "51bf938c8365b4b69a0b5216a64dd0ebf4e35a16",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/75536e59645bd663d82e4aa7ab6a5a2e1efe7842"
        },
        "date": 1771520156460,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
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
          "id": "387caeab1380541d63e738c650f3efae961c88d2",
          "message": "docs: remove external library comparison benchmarks\n\nRemove head-to-head comparison sections with libsecp256k1 and tiny-ecdsa\nfrom README.md, docs/BENCHMARKS.md, and docs/wiki/Benchmarks.md.\nKeep only internal benchmark numbers (our own platforms/backends).",
          "timestamp": "2026-02-19T17:02:37Z",
          "tree_id": "2e299a97cf526187045d42f7a14fffa2226e82ab",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/387caeab1380541d63e738c650f3efae961c88d2"
        },
        "date": 1771520673935,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 146,
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
          "id": "06044233332d5fd006c1802d3eb583d1c95ac9f1",
          "message": "bench: fix scalar_mul/generator_mul using 128 threads matching __launch_bounds__\n\nThe kernel's __launch_bounds__(128, 2) caused silent launch failures\nwhen bench used cfg.threads_per_block=256. Kernels never executed,\nproducing 0.0 ns. Now uses constexpr kThreads=128 and adds\ncudaGetLastError() check after warmup launches.",
          "timestamp": "2026-02-19T17:18:35Z",
          "tree_id": "50b591f7dfde2711497d44e0dce13f33bf482e0a",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/06044233332d5fd006c1802d3eb583d1c95ac9f1"
        },
        "date": 1771521587848,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
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
          "id": "d7d153309da5a394a9b1619325452b929ad2f24c",
          "message": "feat: GPU signature benchmarks (ECDSA + Schnorr) + docs update v3.5.0\n\n- Add 6 batch kernel wrappers: ecdsa_sign/verify, schnorr_sign/verify,\n  ecdsa_sign_recoverable, ecdsa_recover (all __launch_bounds__(128,2))\n- Add 5 GPU signature benchmarks to bench_cuda.cu with\n  prepare_ecdsa_test_data() helper for verify correctness\n- Results (RTX 5060 Ti): ECDSA Sign 204.8ns/4.88M/s, Verify 410.1ns/2.44M/s,\n  Schnorr Sign 273.4ns/3.66M/s, Verify 354.6ns/2.82M/s\n- README: blockchain coin badges, GPU signature benchmark tables,\n  27-coin supported coins section, SEO metadata footer\n- BENCHMARKS.md: split CUDA into Core ECC + Signature tables, update all numbers\n- API_REFERENCE.md: add CUDA Signature Operations section with full API docs\n- CHANGELOG.md: add v3.5.0 entry\n- Wiki: update Benchmarks.md and CUDA-Guide.md with signature operations\n\nNo other open-source GPU library provides secp256k1 ECDSA+Schnorr sign/verify.",
          "timestamp": "2026-02-19T17:49:18Z",
          "tree_id": "774456bb75caf8ff3428c7e31d668cac653f24bd",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/d7d153309da5a394a9b1619325452b929ad2f24c"
        },
        "date": 1771523435285,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
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
          "id": "b99f2ba02a109dd41aa5fe6a4c50c5727a8ada31",
          "message": "fix: version numbering 3.5.0→3.6.0 (v3.5.0 tag already exists)",
          "timestamp": "2026-02-19T17:50:16Z",
          "tree_id": "872569a6aa24920ffba0e6ba8591ab5bdd24970d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/b99f2ba02a109dd41aa5fe6a4c50c5727a8ada31"
        },
        "date": 1771523491423,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 145,
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
          "id": "307ab420e0bc4676ef6ca2b314f321bbdd711421",
          "message": "Merge dev into main: v3.6.0 — GPU signature operations (ECDSA + Schnorr)\n\n- 6 CUDA batch kernels: ecdsa_sign/verify, schnorr_sign/verify, recoverable, recover\n- Benchmarks: ECDSA Sign 204.8ns/4.88M/s, Verify 410.1ns/2.44M/s, Schnorr Sign 273.4ns/3.66M/s, Verify 354.6ns/2.82M/s\n- CPU optimizations ported to GPU: Shamir, GLV, wNAF, precomputed tables, tagged hash midstates\n- C ABI (ufsecp), 12 language bindings, scalar_mul 3.1x speedup, multi-backend fixes\n- Full doc update: README, BENCHMARKS, API_REFERENCE, CHANGELOG, wiki",
          "timestamp": "2026-02-19T17:52:53Z",
          "tree_id": "872569a6aa24920ffba0e6ba8591ab5bdd24970d",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/307ab420e0bc4676ef6ca2b314f321bbdd711421"
        },
        "date": 1771523665845,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
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
          "id": "5978ce6fe4a33669f015aa94a3e8bee48ffc6c57",
          "message": "fix(ci): add FieldElement::negate() for MSVC + guard GCC compile flags\n\n- Add negate()/negate_assign() to base FieldElement class (4x64)\n  FieldElement52 and FieldElement26 already had negate(), but the base\n  FieldElement used as OptimalFieldElement on MSVC (no __int128) lacked it.\n  bench_comprehensive_riscv.cpp calls .negate(1) on OFE → build failure.\n\n- Guard GCC/Clang-specific compile flags with compiler ID check\n  target_compile_options -O3 -fno-math-errno etc. were applied\n  unconditionally, causing D9002 warnings on MSVC cl.exe.\n\n- Guard set_source_files_properties for field_asm.cpp/field.cpp/point.cpp\n  GCC-specific flags (-fipa-pta, -mbmi2) now only applied to GCC/Clang.\n\nVerified: both NO_INT128 and normal builds pass (0 errors, 8/8 tests).",
          "timestamp": "2026-02-19T18:56:58Z",
          "tree_id": "ec5bc8f5661666a13888855cd7c0ac59f0c5ea3b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5978ce6fe4a33669f015aa94a3e8bee48ffc6c57"
        },
        "date": 1771527498536,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 146,
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
          "id": "a85ae3a4dd50104e55c4bf18fb6c4294da5bc81a",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-19T19:04:30Z",
          "tree_id": "ec5bc8f5661666a13888855cd7c0ac59f0c5ea3b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a85ae3a4dd50104e55c4bf18fb6c4294da5bc81a"
        },
        "date": 1771527938629,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 22,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
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
          "id": "38978ca7e5fd5072fc9ef66fd53f70b4561b68bc",
          "message": "docs: SEO optimization — README rewrite, PORTING.md, version updates\n\n- README.md: Complete rewrite with SEO-friendly keyword-rich H2 headings\n  - 'secp256k1 GPU Acceleration', 'secp256k1 on Embedded', 'WASM secp256k1',\n    'Constant-Time secp256k1', 'secp256k1 Benchmarks — Cross-Platform Comparison'\n  - First 20 lines: what-is-it, why, quick links to Benchmarks/Security/Porting\n  - Cross-platform comparison tables (CPU/GPU/Embedded side-by-side)\n  - Expanded SEO keyword comment block\n  - Zero-dependency messaging prominent throughout\n- PORTING.md: New porting guide for CPU architectures, embedded targets, GPU backends\n  - Step-by-step checklists for each port type\n  - Reference implementation table (CUDA/OpenCL/Metal/ROCm)\n  - Submission process for community ports\n- SECURITY.md: Updated supported versions to v3.6.0\n- THREAT_MODEL.md: Updated version references to v3.6.0\n- docs/README.md: Version bump to 3.6.0, added Porting Guide + Security section links",
          "timestamp": "2026-02-19T19:19:07Z",
          "tree_id": "bc6c1642e033a7d6b267afe5d66781b975c2a5d4",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/38978ca7e5fd5072fc9ef66fd53f70b4561b68bc"
        },
        "date": 1771528820796,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 24,
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
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 248,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 127,
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
          "id": "eb8c196c6e6266523a8021b60f2f9f77234e7a10",
          "message": "docs: add Acknowledgements section to README",
          "timestamp": "2026-02-19T19:45:06Z",
          "tree_id": "fc26d1c6e0dbe23c1b59128888922abc3bb12f84",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/eb8c196c6e6266523a8021b60f2f9f77234e7a10"
        },
        "date": 1771530375840,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
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
          "id": "2d77b548f2effa1c3e3f04f4a1978b2106e0d6f5",
          "message": "ci: add linux-arm64 cross-compile + android armeabi-v7a, x86_64 targets",
          "timestamp": "2026-02-19T20:00:10Z",
          "tree_id": "13629bb98a3758264c2bcffc3df59ace0a42cdcb",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/2d77b548f2effa1c3e3f04f4a1978b2106e0d6f5"
        },
        "date": 1771531297987,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 148,
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
          "id": "5e92027d7277edca37f3f97efaea8022dd0cf3c5",
          "message": "fix(ci): ARCH_FLAGS quoting (SHELL: prefix) + enable arm64 ASM for cross-compile\n\n- cpu/CMakeLists.txt: Use SHELL:${ARCH_FLAGS} in target_compile_options so\n  multi-word flags (e.g. '-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp')\n  are split into separate compiler arguments instead of one quoted string.\n  Fixes android armeabi-v7a and x86_64 builds.\n\n- ci.yml: Remove -DSECP256K1_USE_ASM=OFF from linux-arm64 job. ARM64 'asm'\n  files are .cpp with inline asm (not standalone .S), so they compile fine\n  with aarch64-linux-gnu-g++-13. Fixes undefined reference to field_mul_arm64.",
          "timestamp": "2026-02-19T20:13:55Z",
          "tree_id": "24e636f5f875fb1e21af0578005bb29813d7e167",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5e92027d7277edca37f3f97efaea8022dd0cf3c5"
        },
        "date": 1771532104691,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
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
          "id": "5afd7b3f5eab478f52e901769209c142243c6205",
          "message": "fix(field_52): guard __int128 with __SIZEOF_INT128__ only\n\nThe previous guard (#if __SIZEOF_INT128__ || __GNUC__ || __clang__)\nincorrectly allowed __int128 on 32-bit ARM (armeabi-v7a) where\n__GNUC__/__clang__ are defined but __int128 is not supported.\n\nUse __SIZEOF_INT128__ alone — the canonical 64-bit check that\nfield_52_impl.hpp already uses.",
          "timestamp": "2026-02-19T20:18:09Z",
          "tree_id": "b1b2bb4280e8ea2a72b3e2d5071397111e8fd328",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5afd7b3f5eab478f52e901769209c142243c6205"
        },
        "date": 1771532353708,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 258,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 147,
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
          "id": "780f111608de8f69c01a0da8065b39d4bdae01d5",
          "message": "fix(field_asm): guard __uint128_t with __SIZEOF_INT128__, add 32-bit fallback\n\nsubborrow64() used __uint128_t under '#elif __GNUC__ || __clang__' which\nfires on 32-bit ARM where __int128 is unavailable. Switch to\n__SIZEOF_INT128__ and add a portable borrow-chain fallback for armeabi-v7a.",
          "timestamp": "2026-02-19T20:20:24Z",
          "tree_id": "40f1bdbfa50a292887c52b38a42b150d67c4d72f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/780f111608de8f69c01a0da8065b39d4bdae01d5"
        },
        "date": 1771532496479,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 146,
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
          "id": "a5ab921ac510151374ad0a71337122bd980365ca",
          "message": "Merge dev into main: CI cross-platform fixes (arm64, armeabi-v7a, x86_64)",
          "timestamp": "2026-02-19T20:23:21Z",
          "tree_id": "40f1bdbfa50a292887c52b38a42b150d67c4d72f",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/a5ab921ac510151374ad0a71337122bd980365ca"
        },
        "date": 1771532672601,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
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
          "id": "07494c85ac51ba6a26a91bb4072853cca9430605",
          "message": "fix(ci): arm64 cross-compile linker failure — guard header auto-detection with SECP256K1_NO_ASM\n\nfield_asm.hpp unconditionally defined SECP256K1_HAS_ARM64_ASM when\n__aarch64__ was detected, ignoring CMake's SECP256K1_USE_ASM=OFF.\nThis caused field.cpp to call arm64::field_mul_arm64/field_sqr_arm64\nwithout the source files being compiled.\n\nFix:\n- field_asm.hpp: Add !defined(SECP256K1_NO_ASM) guard to ARM64 auto-detection\n- cpu/CMakeLists.txt: Define SECP256K1_NO_ASM=1 when SECP256K1_USE_ASM=OFF\n- release.yml: Enable ASM=ON for arm64 cross-compile (inline C++ asm works fine)",
          "timestamp": "2026-02-20T00:56:53+04:00",
          "tree_id": "8975f87572e416a8378a110ac9465c81bbb80f7c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/07494c85ac51ba6a26a91bb4072853cca9430605"
        },
        "date": 1771534691033,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
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
          "id": "93c133b5be268e1d4a9b0dc512221f5e188d135b",
          "message": "merge: dev -> main — fix arm64 cross-compile (SECP256K1_NO_ASM guard)",
          "timestamp": "2026-02-20T00:57:07+04:00",
          "tree_id": "8975f87572e416a8378a110ac9465c81bbb80f7c",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/93c133b5be268e1d4a9b0dc512221f5e188d135b"
        },
        "date": 1771534699947,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 257,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 145,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "4fa9e15cd26d9fc1399c2eaca9eca25d5dc695fa",
          "message": "fix(ci): fix all packaging jobs — NuGet/gem/Python/npm/Java\n\n- pack-nuget: update target frameworks net6.0/7.0→net8.0/9.0 (EOL fix),\n  install both SDK versions, add native Content items to .csproj\n- pack-gem: fix gemspec s.files (remove missing README.md, include\n  lib/native/* for bundled native libs)\n- pack-python: copy pyproject.ufsecp.toml→pyproject.toml (was using wrong\n  package manifest), add package-data for .so/.dll/.dylib, add platform\n  wheel tags (manylinux/macosx/win), handle .zip extraction for win-x64\n- pack-npm: use correct prebuild dir names (darwin-arm64, win32-x64),\n  extract via mktemp for clean handling, handle .zip + .tar.gz\n- pack-java: fix jar packaging (proper classes+native structure),\n  derive version from tag instead of hardcoded 3.4.0",
          "timestamp": "2026-02-20T01:11:55+04:00",
          "tree_id": "9ba633dce8c4d818c1ac42785fa4286817f12e7e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/4fa9e15cd26d9fc1399c2eaca9eca25d5dc695fa"
        },
        "date": 1771535587992,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 24,
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
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 247,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 127,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "de925177218f582e8b269e67d2c973ce9c00156c",
          "message": "Merge branch 'dev'",
          "timestamp": "2026-02-20T01:12:08+04:00",
          "tree_id": "9ba633dce8c4d818c1ac42785fa4286817f12e7e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/de925177218f582e8b269e67d2c973ce9c00156c"
        },
        "date": 1771535595874,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
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
          "id": "5dd538147931ca55c38435d6ce63d66dfcd3f383",
          "message": "fix: correct carry propagation in generic reduce() function\n\nThe no-ASM reduce() path had broken carry propagation when adding\nhi_limb * (2^32 + 977) to the result array. Manual carry chains\nonly propagated 1-2 positions, losing carries when result limbs\nwere near-max (e.g. 0xFFFFFFFFFFFFFFFF). This caused incorrect\nmultiplication results for inputs involving large field elements.\n\nFix: replace manual add64+carry chains with add_into() which\nproperly propagates carries through the entire 5-limb array.\n\nAffects: sanitizer builds (SECP256K1_USE_ASM=OFF) and any platform\nwithout BMI2/ARM64/RISC-V assembly (ESP32, STM32 fallback).\n\nVerified: field_26 test 269/269 pass (MSVC Debug, ASM=OFF).",
          "timestamp": "2026-02-20T01:54:25+04:00",
          "tree_id": "2f1693f9449ee9c7e7a16066e655664f91081d5b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5dd538147931ca55c38435d6ce63d66dfcd3f383"
        },
        "date": 1771538136727,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 255,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 145,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
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
          "id": "5dd538147931ca55c38435d6ce63d66dfcd3f383",
          "message": "fix: correct carry propagation in generic reduce() function\n\nThe no-ASM reduce() path had broken carry propagation when adding\nhi_limb * (2^32 + 977) to the result array. Manual carry chains\nonly propagated 1-2 positions, losing carries when result limbs\nwere near-max (e.g. 0xFFFFFFFFFFFFFFFF). This caused incorrect\nmultiplication results for inputs involving large field elements.\n\nFix: replace manual add64+carry chains with add_into() which\nproperly propagates carries through the entire 5-limb array.\n\nAffects: sanitizer builds (SECP256K1_USE_ASM=OFF) and any platform\nwithout BMI2/ARM64/RISC-V assembly (ESP32, STM32 fallback).\n\nVerified: field_26 test 269/269 pass (MSVC Debug, ASM=OFF).",
          "timestamp": "2026-02-20T01:54:25+04:00",
          "tree_id": "2f1693f9449ee9c7e7a16066e655664f91081d5b",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/5dd538147931ca55c38435d6ce63d66dfcd3f383"
        },
        "date": 1771538150859,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 1,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 255,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 145,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "744a41c44195af3be5d4be17ab30c75b2e1a9a1d",
          "message": "build: automate version from VERSION file + git tag injection in CI\n\n- Add VERSION file (single source of truth: 3.8.0)\n- Root CMakeLists.txt, ufsecp, android, wasm all read from VERSION\n- ufsecp_version.h auto-generated via configure_file from .h.in template\n- All 11 binding configs set to 0.0.0-dev (CI injects real version from tag)\n- release.yml: 5 build jobs write tag version to VERSION before cmake\n- release.yml: 4 pack jobs sed-inject version into binding configs\n- nuget nuspec + conanfile.py also set to 0.0.0-dev placeholder\n\nNo more hardcoded versions — tag drives everything.",
          "timestamp": "2026-02-20T02:16:53+04:00",
          "tree_id": "b58c14294adb89b60066305d20936bf97fe9516e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/744a41c44195af3be5d4be17ab30c75b2e1a9a1d"
        },
        "date": 1771539482209,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
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
          "distinct": false,
          "id": "744a41c44195af3be5d4be17ab30c75b2e1a9a1d",
          "message": "build: automate version from VERSION file + git tag injection in CI\n\n- Add VERSION file (single source of truth: 3.8.0)\n- Root CMakeLists.txt, ufsecp, android, wasm all read from VERSION\n- ufsecp_version.h auto-generated via configure_file from .h.in template\n- All 11 binding configs set to 0.0.0-dev (CI injects real version from tag)\n- release.yml: 5 build jobs write tag version to VERSION before cmake\n- release.yml: 4 pack jobs sed-inject version into binding configs\n- nuget nuspec + conanfile.py also set to 0.0.0-dev placeholder\n\nNo more hardcoded versions — tag drives everything.",
          "timestamp": "2026-02-20T02:16:53+04:00",
          "tree_id": "b58c14294adb89b60066305d20936bf97fe9516e",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/744a41c44195af3be5d4be17ab30c75b2e1a9a1d"
        },
        "date": 1771539490610,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 25,
            "unit": "ns"
          },
          {
            "name": "Field Square",
            "value": 23,
            "unit": "ns"
          },
          {
            "name": "Field Add",
            "value": 2,
            "unit": "ns"
          },
          {
            "name": "Field Negate",
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 256,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 145,
            "unit": "ns"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "committer": {
            "email": "payysoon@gmail.com",
            "name": "vano",
            "username": "shrec"
          },
          "distinct": true,
          "id": "c6f85a5885f50841c4df1f9367bfe36152d8c892",
          "message": "chore: bump VERSION to 3.9.0 for release",
          "timestamp": "2026-02-20T02:17:32+04:00",
          "tree_id": "5e8c1e559539dfa5f357fd906a28e1fe4dfec3c6",
          "url": "https://github.com/shrec/UltrafastSecp256k1/commit/c6f85a5885f50841c4df1f9367bfe36152d8c892"
        },
        "date": 1771539524098,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "Field Mul",
            "value": 23,
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
            "value": 0,
            "unit": "ns"
          },
          {
            "name": "Point Add",
            "value": 246,
            "unit": "ns"
          },
          {
            "name": "Point Double",
            "value": 127,
            "unit": "ns"
          }
        ]
      }
    ]
  }
}