FROM gcr.io/oss-fuzz-base/base-builder:v1@sha256:f25e0fc27f5c19b967cbe96a32c3e87ac46fc1379d0e216093a9ee8dcbdf78c3
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
