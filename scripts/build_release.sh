#!/usr/bin/env bash
# ============================================================================
# UltrafastSecp256k1 — Release Build Script
# ============================================================================
# Builds release binaries + creates distribution archive + NuGet layout.
#
# Usage:
#   ./scripts/build_release.sh                  # default: Release
#   ./scripts/build_release.sh --build-type Debug
#   ./scripts/build_release.sh --skip-tests
#
# Output:
#   release/UltrafastSecp256k1-<version>-<os>-<arch>/
#   release/UltrafastSecp256k1-<version>-<os>-<arch>.tar.gz  (or .zip)
#   release/nuget/  (NuGet runtime layout)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ── Defaults ──
BUILD_TYPE="Release"
SKIP_TESTS=false
BUILD_DIR="${ROOT_DIR}/build/release-pkg"
RELEASE_DIR="${ROOT_DIR}/release"

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-type) BUILD_TYPE="$2"; shift 2 ;;
        --skip-tests) SKIP_TESTS=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# ── Detect platform ──
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux*)  PLATFORM="linux"; EXT="tar.gz" ;;
    Darwin*) PLATFORM="macos"; EXT="tar.gz" ;;
    MINGW*|MSYS*|CYGWIN*) PLATFORM="win"; EXT="zip" ;;
    *) echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|amd64) ARCH_TAG="x64" ;;
    aarch64|arm64) ARCH_TAG="arm64" ;;
    *) ARCH_TAG="$ARCH" ;;
esac

# ── Read version from CMakeLists.txt ──
VERSION=$(grep -oP 'VERSION\s+\K[0-9]+\.[0-9]+\.[0-9]+' "${ROOT_DIR}/CMakeLists.txt" | head -1)
PKG_NAME="UltrafastSecp256k1-v${VERSION}-${PLATFORM}-${ARCH_TAG}"

echo "════════════════════════════════════════════════════════════"
echo "  UltrafastSecp256k1 Release Build"
echo "  Version:    ${VERSION}"
echo "  Platform:   ${PLATFORM}-${ARCH_TAG}"
echo "  Build Type: ${BUILD_TYPE}"
echo "  Output:     ${RELEASE_DIR}/${PKG_NAME}"
echo "════════════════════════════════════════════════════════════"

# ── Configure ──
echo ""
echo ">>> Configuring..."
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" \
    -G Ninja \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DSECP256K1_BUILD_TESTS=ON \
    -DSECP256K1_BUILD_BENCH=OFF \
    -DSECP256K1_BUILD_EXAMPLES=OFF

# ── Build ──
echo ""
echo ">>> Building..."
cmake --build "${BUILD_DIR}" -j"$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4)"

# ── Test ──
if [ "$SKIP_TESTS" = false ]; then
    echo ""
    echo ">>> Running tests..."
    ctest --test-dir "${BUILD_DIR}" --output-on-failure -j4
fi

# ── Install to staging ──
echo ""
echo ">>> Installing to staging..."
STAGING="${BUILD_DIR}/staging"
cmake --install "${BUILD_DIR}" --prefix "${STAGING}"

# ── Collect release artifacts ──
echo ""
echo ">>> Packaging ${PKG_NAME}..."
rm -rf "${RELEASE_DIR}/${PKG_NAME}"
mkdir -p "${RELEASE_DIR}/${PKG_NAME}/lib"
mkdir -p "${RELEASE_DIR}/${PKG_NAME}/include/ufsecp"
mkdir -p "${RELEASE_DIR}/${PKG_NAME}/include/secp256k1"

# ufsecp headers
cp "${ROOT_DIR}/include/ufsecp/ufsecp.h"         "${RELEASE_DIR}/${PKG_NAME}/include/ufsecp/"
cp "${ROOT_DIR}/include/ufsecp/ufsecp_version.h"  "${RELEASE_DIR}/${PKG_NAME}/include/ufsecp/"
cp "${ROOT_DIR}/include/ufsecp/ufsecp_error.h"    "${RELEASE_DIR}/${PKG_NAME}/include/ufsecp/"

# C++ public headers
if [ -d "${STAGING}/include/secp256k1" ]; then
    cp -r "${STAGING}/include/secp256k1/"* "${RELEASE_DIR}/${PKG_NAME}/include/secp256k1/"
fi

# Libraries
find "${STAGING}/lib" "${STAGING}/bin" -maxdepth 1 \( \
    -name "*.dll" -o -name "*.lib" -o -name "*.so" -o -name "*.so.*" \
    -o -name "*.dylib" -o -name "*.a" \) 2>/dev/null | while read -r f; do
    cp "$f" "${RELEASE_DIR}/${PKG_NAME}/lib/"
done

# pkg-config
if [ -d "${STAGING}/lib/pkgconfig" ]; then
    cp -r "${STAGING}/lib/pkgconfig" "${RELEASE_DIR}/${PKG_NAME}/lib/"
fi

# CMake config
if [ -d "${STAGING}/lib/cmake" ]; then
    cp -r "${STAGING}/lib/cmake" "${RELEASE_DIR}/${PKG_NAME}/lib/"
fi

# Docs
cp "${ROOT_DIR}/LICENSE"      "${RELEASE_DIR}/${PKG_NAME}/" 2>/dev/null || true
cp "${ROOT_DIR}/README.md"    "${RELEASE_DIR}/${PKG_NAME}/" 2>/dev/null || true
cp "${ROOT_DIR}/CHANGELOG.md" "${RELEASE_DIR}/${PKG_NAME}/" 2>/dev/null || true
cp "${ROOT_DIR}/include/ufsecp/SUPPORTED_GUARANTEES.md" "${RELEASE_DIR}/${PKG_NAME}/" 2>/dev/null || true

# ── Create archive ──
echo ""
echo ">>> Creating archive..."
cd "${RELEASE_DIR}"
if [ "$EXT" = "tar.gz" ]; then
    tar czf "${PKG_NAME}.tar.gz" "${PKG_NAME}"
    echo "  Archive: ${RELEASE_DIR}/${PKG_NAME}.tar.gz"
else
    if command -v 7z &>/dev/null; then
        7z a -tzip "${PKG_NAME}.zip" "${PKG_NAME}" > /dev/null
    else
        zip -r "${PKG_NAME}.zip" "${PKG_NAME}" > /dev/null
    fi
    echo "  Archive: ${RELEASE_DIR}/${PKG_NAME}.zip"
fi

# ── Populate NuGet runtime layout ──
echo ""
echo ">>> Setting up NuGet layout..."
NUGET_ROOT="${RELEASE_DIR}/nuget"
mkdir -p "${NUGET_ROOT}/runtimes/${PLATFORM}-${ARCH_TAG}/native"

# Copy native libraries to NuGet runtimes
find "${RELEASE_DIR}/${PKG_NAME}/lib" -maxdepth 1 \( \
    -name "ufsecp*" -o -name "libufsecp*" \) 2>/dev/null | while read -r f; do
    cp "$f" "${NUGET_ROOT}/runtimes/${PLATFORM}-${ARCH_TAG}/native/"
done

# Copy ufsecp headers
mkdir -p "${NUGET_ROOT}/include/ufsecp"
cp "${ROOT_DIR}/include/ufsecp/ufsecp.h"         "${NUGET_ROOT}/include/ufsecp/"
cp "${ROOT_DIR}/include/ufsecp/ufsecp_version.h"  "${NUGET_ROOT}/include/ufsecp/"
cp "${ROOT_DIR}/include/ufsecp/ufsecp_error.h"    "${NUGET_ROOT}/include/ufsecp/"

echo "  NuGet runtimes: ${NUGET_ROOT}/runtimes/${PLATFORM}-${ARCH_TAG}/native/"

# ── Summary ──
echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Release build complete!"
echo ""
echo "  Package: ${RELEASE_DIR}/${PKG_NAME}/"
echo "  Archive: ${RELEASE_DIR}/${PKG_NAME}.${EXT}"
echo "  NuGet:   ${NUGET_ROOT}/"
echo ""
echo "  Contents:"
ls -lh "${RELEASE_DIR}/${PKG_NAME}/lib/" 2>/dev/null || true
echo ""
echo "  To create NuGet package:"
echo "    cp -r nuget/* ${NUGET_ROOT}/"
echo "    cd ${NUGET_ROOT} && nuget pack UltrafastSecp256k1.Native.nuspec"
echo "════════════════════════════════════════════════════════════"
