# Contributors

Thank you to everyone who has contributed to UltrafastSecp256k1.

## Bug Reports & Security Findings

| Contributor | Finding | Issue |
|---|---|---|
| [@bschofield](https://github.com/bschofield) | CUDA ChaCha20 `rotl32(8)` wrong `__byte_perm` selector (`0x0321` → `0x2103`) — keystream disagreed with RFC 8439 §2.3.2 despite round-trip tests passing | [#256](https://github.com/shrec/UltrafastSecp256k1/issues/256) |
| [@tatotogonidze-cmd](https://github.com/tatotogonidze-cmd) | BIP-352 GPU audit coverage — diagnosed that CPU-only builds linked no GPU provider, leaving every `ufsecp_gpu_*` symbol in `test_regression_bip352_ct_varbase` undefined; and, in the proposed fix, distinguished "a backend was selected" from "the operation actually ran". The second point was a live silent-pass advisory in our own code: an available backend whose `bip352_scan_batch_multispend` is unsupported produced a CTest PASS with zero GPU coverage. The distinction was ported to `dev` as `bcv_gpu_coverage_is_advisory()` | [#384](https://github.com/shrec/UltrafastSecp256k1/pull/384) |

## Code Contributions

| Contributor | Contribution | PR |
|---|---|---|
| [@sparrowwallet/frigate](https://github.com/sparrowwallet/frigate) | Production integration — Silent Payments GPU scanning | — |
| [@FurkanAdmin](https://github.com/FurkanAdmin) | CUDA build fix — added `extern __global__` declaration for `ct_generator_mul_batch_kernel` in `src/cuda/include/secp256k1.cuh` so the host TU (`src/gpu/src/gpu_backend_cuda.cu`) can launch the kernel without "undefined identifier" link error on RTX 3060 / Linux x86-64 builds | [#274](https://github.com/shrec/UltrafastSecp256k1/pull/274) |

---

To add your project to the [Adopters](ADOPTION.md) list, open a PR or issue.
