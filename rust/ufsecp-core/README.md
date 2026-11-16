# Experimental Rust arithmetic backend

`ufsecp-core` implements a bounded subset of UltrafastSecp256k1 directly in
Rust. It has no runtime dependencies, allocation, C++ compiler requirement, or
FFI. Stable Rust builds the portable CPU backend; nightly Rust builds the
NVIDIA GPU backend. It is separate from the existing `bindings/rust` wrappers.

This is **public, variable-time arithmetic**, not a replacement for the
constant-time signing API. Do not use it for wallet private keys, secret
nonces, ECDH secrets, or production signing. Functional differential tests
are not a constant-time audit.

## Port provenance

Ported from revision `5536321b3af2322cd2b8e90221a5bb9e4587fb7d`:

| Rust code | Upstream implementation |
|---|---|
| `field.rs`, `limbs.rs` | `src/cpu/src/field.cpp::reduce`, portable limb carry arithmetic; `src/cuda/include/secp256k1.cuh::field_add`, `field_sub`, `field_inv_fermat_chain_impl`, `field_sqrt` |
| `ptx.rs` | `src/cuda/include/secp256k1_32_hybrid_final.cuh::mul_256_comba32`, `sqr_256_comba32`, `reduce_512_to_256_32`; CUDA 64-bit add/sub carry chains |
| `scalar.rs` | `src/cuda/include/secp256k1.cuh::scalar_add`, `scalar_sub`, `scalar_negate`, `scalar_mul_mod_n`, `scalar_inverse` |
| `point.rs` | `src/cuda/include/secp256k1.cuh::jacobian_double`, `jacobian_add_mixed`; basic public double-and-add |

GPU multiplication retains the upstream 32-bit Comba instruction sequences
through Rust `asm!`, followed by the same four reduction phases. It does not
call native CUDA C++ code. CPU multiplication uses bounded schoolbook rows;
the CPU reduction retains the five-limb carry cascade. Field/scalar public
encodings remain four little-endian `u64` limbs. Constructors enforce canonical
representations or explicitly reduce input; field negation canonicalizes zero.
Point infinity retains its explicit flag, with canonical zero coordinates for
new infinity results. Coordinate structs are internal arithmetic inputs and
must contain validated points; use `AffinePoint::new` for untrusted coordinates.

This port does **not** include GLV, CPU assembly dispatch, constant-time secret
operations, signature protocols, hash functions, wallets, OpenCL, HIP, or Metal.
Pickaxe's transaction hashing, nonce control, fixed-key multiplication, and
GPU scheduling stay in the downstream miner and are not part of this MIT crate.
No Pickaxe AGPL arithmetic source is incorporated here.

## Checks

From the repository root (put build products outside the source directory):

```sh
export CARGO_TARGET_DIR="$PWD/out/rust"
cargo test --locked --release --manifest-path rust/ufsecp-core/Cargo.toml
cargo fmt --manifest-path rust/ufsecp-core/Cargo.toml -- --check
cargo clippy --locked --manifest-path rust/ufsecp-core/Cargo.toml --all-targets -- -D warnings
```

The host checks compare field/scalar operations against `num-bigint`, and
full-width public points against Bitcoin Core's `libsecp256k1` via its Rust
test binding. They include zero, p/n boundaries, every single-bit position,
carry chains, random full-width inputs, doubling and inverse-point addition.
These test dependencies are absent from the backend and GPU build.

The CUDA probe checks the actual GPU backend, including generic Barrett scalar
multiplication/inversion, against the CPU results checked above. Build for the
actual device architecture, stop other GPU workloads, and run it exclusively:

```sh
rustup toolchain install nightly-2026-04-02 --component rust-src,llvm-tools,llvm-bitcode-linker
RUSTFLAGS="-C target-cpu=sm_120 -C panic=abort" \
  cargo +nightly-2026-04-02 build --locked --release \
  --manifest-path rust/ufsecp-core/Cargo.toml --example cuda_probe \
  --target nvptx64-nvidia-cuda -Z build-std=core
UFSECP_TEST_PTX="$CARGO_TARGET_DIR/nvptx64-nvidia-cuda/release/examples/cuda_probe.ptx" \
  cargo test --locked --release --manifest-path rust/ufsecp-core/Cargo.toml \
  --test cuda -- --ignored --nocapture --test-threads=1
```

NVPTX and inline assembly use nightly features; this backend is experimental.
Architecture compilation alone is not physical GPU validation. Existing native
code and tests remain unchanged. See `RESULTS.md` for measured coverage and
performance limitations before considering adoption.
