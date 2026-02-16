window.BENCHMARK_DATA = {
  "lastUpdate": 1771218945045,
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
      }
    ]
  }
}