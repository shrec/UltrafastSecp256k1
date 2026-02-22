window.BENCHMARK_DATA = {
  "lastUpdate": 1771798242480,
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
      }
    ]
  }
}