#!/bin/bash
# bench_knuth.sh — Benchmark UF shim vs Knuth secp256k1 (k-nuth/secp256k1)
# Uses Knuth's own bench.h harness (10 × 20,000 ops, median)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
UF_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
KNUTH_DIR="${KNUTH_DIR:-/tmp/bench_knuth_secp256k1}"
RESULTS="${RESULTS_FILE:-$UF_ROOT/docs/NODES_SHIM_STATUS.md}"

echo "=== Knuth secp256k1 benchmark ==="
echo "UF root: $UF_ROOT"

# Clone Knuth secp256k1 if not present
if [ ! -d "$KNUTH_DIR/src" ]; then
    echo "Cloning k-nuth/secp256k1..."
    git clone --depth=1 https://github.com/k-nuth/secp256k1.git "$KNUTH_DIR"
fi

# Build Knuth library
echo "Building Knuth native..."
gcc -O3 -march=native \
    -DUSE_NUM_GMP -DUSE_FIELD_10X26 -DUSE_SCALAR_8X32 \
    -DUSE_ENDOMORPHISM -DUSE_SCALAR_INV_BUILTIN -DUSE_FIELD_INV_BUILTIN \
    -DENABLE_MODULE_ECDH=1 -DENABLE_MODULE_SCHNORR=1 -DENABLE_MODULE_RECOVERY=1 \
    -DHAVE_LIBGMP=1 -DECMULT_WINDOW_SIZE=15 -DECMULT_GEN_PREC_BITS=4 \
    -I"$KNUTH_DIR" -I"$KNUTH_DIR/src" -I"$KNUTH_DIR/include" \
    "$KNUTH_DIR/src/secp256k1.c" -shared -fPIC \
    -o /tmp/libknuth_secp256k1.so -lgmp 2>/dev/null

# Build benchmark harness
cat > /tmp/bench_knuth_harness.c << 'EOF'
#include <stdio.h>
#include <string.h>
#include "bench.h"
#include "secp256k1.h"

typedef struct { secp256k1_context* ctx; unsigned char msg[32]; unsigned char key[32]; } bench_sign_t;

static void bench_sign_setup(void* arg) {
    bench_sign_t *d = (bench_sign_t*)arg;
    for (int i=0;i<32;i++) d->msg[i]=i+1;
    for (int i=0;i<32;i++) d->key[i]=i+65;
}

static void bench_sign_run(void* arg) {
    bench_sign_t *d = (bench_sign_t*)arg;
    for (int i=0;i<20000;i++) {
        unsigned char sig[74]; size_t siglen=74;
        secp256k1_ecdsa_signature s;
        secp256k1_ecdsa_sign(d->ctx, &s, d->msg, d->key, NULL, NULL);
        secp256k1_ecdsa_signature_serialize_der(d->ctx, sig, &siglen, &s);
        for (int j=0;j<32;j++) { d->msg[j]=sig[j]; d->key[j]=sig[j+32]; }
    }
}

typedef struct {
    secp256k1_context* ctx;
    unsigned char msg[32], pubkey[33], sig[74]; size_t pubkeylen, siglen;
} bench_verify_t;

static void bench_verify_run(void* arg) {
    bench_verify_t *d = (bench_verify_t*)arg;
    secp256k1_pubkey pub; secp256k1_ecdsa_signature sig;
    secp256k1_ec_pubkey_parse(d->ctx, &pub, d->pubkey, d->pubkeylen);
    secp256k1_ecdsa_signature_parse_der(d->ctx, &sig, d->sig, d->siglen);
    for (int i=0;i<20000;i++)
        secp256k1_ecdsa_verify(d->ctx, &sig, d->msg, &pub);
}

int main(void) {
    bench_sign_t sdata;
    sdata.ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
    run_benchmark("ecdsa_sign", bench_sign_run, bench_sign_setup, NULL, &sdata, 10, 20000);

    bench_verify_t vdata;
    vdata.ctx = sdata.ctx;
    for (int i=0;i<32;i++) vdata.msg[i]=i+1;
    unsigned char key[32]; for (int i=0;i<32;i++) key[i]=i+33;
    secp256k1_pubkey pub; secp256k1_ecdsa_signature sig_raw;
    secp256k1_ec_pubkey_create(vdata.ctx, &pub, key);
    vdata.pubkeylen = 33;
    secp256k1_ec_pubkey_serialize(vdata.ctx, vdata.pubkey, &vdata.pubkeylen, &pub, SECP256K1_EC_COMPRESSED);
    secp256k1_ecdsa_sign(vdata.ctx, &sig_raw, vdata.msg, key, NULL, NULL);
    vdata.siglen = sizeof(vdata.sig);
    secp256k1_ecdsa_signature_serialize_der(vdata.ctx, vdata.sig, &vdata.siglen, &sig_raw);
    run_benchmark("ecdsa_verify", bench_verify_run, NULL, NULL, &vdata, 10, 20000);

    secp256k1_context_destroy(sdata.ctx);
    return 0;
}
EOF

# Build and run native Knuth
gcc -O3 -march=native -I"$KNUTH_DIR/src" -I"$KNUTH_DIR/include" \
    /tmp/bench_knuth_harness.c /tmp/libknuth_secp256k1.so \
    -Wl,-rpath,/tmp -lgmp -o /tmp/bench_knuth_native 2>/dev/null

echo ""
echo "--- Knuth secp256k1 (native) ---"
/tmp/bench_knuth_native

# Build UF shim version
CACHE="${SECP256K1_CACHE_PATH:-$UF_ROOT/build/cache_w18.bin}"
g++ -O3 -march=native -std=c++20 \
    -I"$KNUTH_DIR/src" \
    -I"$UF_ROOT/compat/libsecp256k1_bchn_shim/include" \
    -I"$UF_ROOT/compat/libsecp256k1_shim/include" \
    -I"$UF_ROOT/cpu/include" -I"$UF_ROOT/include" \
    -DSECP256K1_CACHE_PATH=\"$CACHE\" \
    /tmp/bench_knuth_harness.c \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_context.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_ecdsa.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_pubkey.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_schnorr.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_extrakeys.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_seckey.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_tagged_hash.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_recovery.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_ellswift.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_ecdh.cpp" \
    "$UF_ROOT/compat/libsecp256k1_shim/src/shim_musig.cpp" \
    "$UF_ROOT/build/gpu-ci/cpu/libfastsecp256k1.a" \
    -lpthread -o /tmp/bench_uf_knuth 2>/dev/null

echo ""
echo "--- UltrafastSecp256k1 shim (Knuth harness) ---"
SECP256K1_CACHE_PATH="$CACHE" /tmp/bench_uf_knuth 2>/dev/null
