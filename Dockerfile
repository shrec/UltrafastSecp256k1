# ===========================================================================
# UltrafastSecp256k1 — Reproducible build + test container
# ===========================================================================
# Build:  docker build -t ultrafastsecp256k1 .
# Test:   docker run --rm ultrafastsecp256k1
# Bench:  docker run --rm ultrafastsecp256k1 ./build/cpu/bench_comprehensive
# ===========================================================================

FROM ubuntu:24.04 AS builder

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        cmake ninja-build g++ ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN cmake -S . -B build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DSECP256K1_BUILD_TESTS=ON \
        -DSECP256K1_BUILD_BENCH=ON \
        -DSECP256K1_BUILD_SHARED=ON \
        -DSECP256K1_INSTALL=ON \
        -DSECP256K1_USE_ASM=ON && \
    cmake --build build -j"$(nproc)"

# Run tests as build verification
RUN ctest --test-dir build --output-on-failure

# --------------------------------------------------------------------------
# Runtime image (minimal)
# --------------------------------------------------------------------------
FROM ubuntu:24.04 AS runtime

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/cpu/libfastsecp256k1.so* /usr/lib/
COPY --from=builder /src/build/cpu/bench_comprehensive /usr/bin/
COPY --from=builder /src/cpu/include/secp256k1 /usr/include/secp256k1/

RUN ldconfig

ENTRYPOINT ["bench_comprehensive"]
