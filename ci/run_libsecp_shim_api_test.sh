#!/usr/bin/env bash
set -euo pipefail

build_dir="${1:-out/ci-shim-api}"
cxx_compiler="${CXX:-g++-14}"

cmake -S . -B "${build_dir}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="${cxx_compiler}" \
  -DSECP256K1_SHIM_BUILD_TESTS=ON \
  -DSECP256K1_BUILD_SHIM=OFF \
  -DBUILD_TESTING=ON \
  -DSECP256K1_BUILD_TESTS=OFF \
  -DSECP256K1_BUILD_BENCH=OFF \
  -DSECP256K1_BUILD_EXAMPLES=OFF \
  -DSECP256K1_BUILD_FUZZ_TESTS=OFF \
  -DSECP256K1_BUILD_PROTOCOL_TESTS=OFF \
  -DSECP256K1_BUILD_CABI=OFF \
  -DSECP256K1_BUILD_ETHEREUM=OFF \
  -DSECP256K1_BUILD_JAVA=OFF \
  -DSECP256K1_INSTALL=OFF

cmake --build "${build_dir}" --target shim_test -j"$(nproc)"

ctest_dir="${build_dir}/compat/libsecp256k1_shim"
ctest_manifest="${ctest_dir}/CTestTestfile.cmake"
if ! grep -q "secp256k1_shim_test" "${ctest_manifest}"; then
  echo "::error::secp256k1_shim_test is not registered in ${ctest_manifest}"
  exit 1
fi

ctest --test-dir "${ctest_dir}" \
  -R "^secp256k1_shim_test$" \
  --output-on-failure \
  --timeout 120
