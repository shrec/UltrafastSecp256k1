FROM gcr.io/oss-fuzz-base/base-builder:v1
# ----------------------------------------------------------------------------
# ClusterFuzzLite build environment for UltrafastSecp256k1
# Builds 3 libFuzzer harnesses: fuzz_field, fuzz_scalar, fuzz_point
# ----------------------------------------------------------------------------

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*

COPY . $SRC/ultrafast
WORKDIR $SRC/ultrafast

COPY .clusterfuzzlite/build.sh $SRC/build.sh
