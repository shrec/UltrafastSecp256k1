#!/bin/bash -eu
# ----------------------------------------------------------------------------
# ClusterFuzzLite / OSS-Fuzz build script for UltrafastSecp256k1
#
# Environment variables (set by ClusterFuzzLite infra):
#   $CC, $CXX           -- compiler (clang)
#   $CFLAGS, $CXXFLAGS  -- sanitizer + coverage flags
#   $LIB_FUZZING_ENGINE -- fuzzer runtime (libFuzzer .a)
#   $SRC                -- source root
#   $OUT                -- output directory for fuzz targets
# ----------------------------------------------------------------------------

cd "$SRC/ultrafast"

# -- Step 1: Build the static library (no sanitizers in lib, engine handles it)
cmake -S . -B build-fuzz -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CC" \
    -DCMAKE_CXX_COMPILER="$CXX" \
    -DCMAKE_C_FLAGS="$CFLAGS" \
    -DCMAKE_CXX_FLAGS="$CXXFLAGS" \
    -DSECP256K1_BUILD_TESTS=OFF \
    -DSECP256K1_BUILD_BENCH=OFF \
    -DSECP256K1_BUILD_EXAMPLES=OFF \
    -DSECP256K1_BUILD_METAL=OFF \
    -DSECP256K1_USE_ASM=OFF

cmake --build build-fuzz -j"$(nproc)" --target fastsecp256k1

LIB="build-fuzz/cpu/libfastsecp256k1.a"
INC="cpu/include"

# -- Step 2: Compile each fuzz harness and link against the library ----------

# fuzz_field: field arithmetic invariants (64 bytes input)
$CXX $CXXFLAGS -std=c++20 -O2 -I "$INC" \
    cpu/fuzz/fuzz_field.cpp \
    $LIB_FUZZING_ENGINE "$LIB" \
    -o "$OUT/fuzz_field"

# fuzz_scalar: scalar ring properties (64 bytes input)
$CXX $CXXFLAGS -std=c++20 -O2 -I "$INC" \
    cpu/fuzz/fuzz_scalar.cpp \
    $LIB_FUZZING_ENGINE "$LIB" \
    -o "$OUT/fuzz_scalar"

# fuzz_point: curve point operations (32 bytes input)
$CXX $CXXFLAGS -std=c++20 -O2 -I "$INC" \
    cpu/fuzz/fuzz_point.cpp \
    $LIB_FUZZING_ENGINE "$LIB" \
    -o "$OUT/fuzz_point"

# -- Step 3: Copy seed corpora (zip per target) -----------------------------
if [ -d "cpu/fuzz/corpus/fuzz_field" ]; then
    zip -j "$OUT/fuzz_field_seed_corpus.zip" cpu/fuzz/corpus/fuzz_field/* 2>/dev/null || true
fi
if [ -d "cpu/fuzz/corpus/fuzz_scalar" ]; then
    zip -j "$OUT/fuzz_scalar_seed_corpus.zip" cpu/fuzz/corpus/fuzz_scalar/* 2>/dev/null || true
fi
if [ -d "cpu/fuzz/corpus/fuzz_point" ]; then
    zip -j "$OUT/fuzz_point_seed_corpus.zip" cpu/fuzz/corpus/fuzz_point/* 2>/dev/null || true
fi

# -- Step 4: Copy fuzzer options (max_len limits) ----------------------------
echo "[libfuzzer]"  > "$OUT/fuzz_field.options"
echo "max_len = 64" >> "$OUT/fuzz_field.options"

echo "[libfuzzer]"  > "$OUT/fuzz_scalar.options"
echo "max_len = 64" >> "$OUT/fuzz_scalar.options"

echo "[libfuzzer]"  > "$OUT/fuzz_point.options"
echo "max_len = 32" >> "$OUT/fuzz_point.options"

echo "[OK] Built 3 fuzz targets: fuzz_field, fuzz_scalar, fuzz_point"
