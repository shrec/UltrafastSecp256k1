window.BENCHMARK_DATA = {
  "lastUpdate": 1771216048287,
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
      }
    ]
  }
}