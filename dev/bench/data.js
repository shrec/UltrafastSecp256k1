window.BENCHMARK_DATA = {
  "lastUpdate": 1771215065521,
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
      }
    ]
  }
}