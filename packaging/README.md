# Linux Distribution Packaging

This directory contains packaging files for building native packages
on various Linux distributions.

## Debian / Ubuntu (.deb)

```bash
# Install build dependencies
sudo apt install debhelper cmake ninja-build g++ pkg-config

# Build package from source tarball
dpkg-buildpackage -us -uc -b
# — or use CPack —
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_SHARED=ON -DSECP256K1_INSTALL=ON
cmake --build build
cd build && cpack -G DEB
```

Produces:
- `libufsecp3_<ver>_<arch>.deb` — shared library
- `libufsecp-dev_<ver>_<arch>.deb` — headers + static lib + cmake/pkgconfig

## Fedora / RHEL / CentOS (.rpm)

```bash
# Install build dependencies
sudo dnf install cmake ninja-build gcc-c++ rpm-build

# Build RPM from spec
rpmbuild -ba packaging/rpm/libufsecp.spec
# — or use CPack —
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_SHARED=ON -DSECP256K1_INSTALL=ON
cmake --build build
cd build && cpack -G RPM
```

## Arch Linux (AUR)

```bash
# From the packaging/arch/ directory:
cd packaging/arch
makepkg -si
```

The `PKGBUILD` downloads the source tarball, builds with CMake+Ninja,
runs tests, and installs to `/usr`.

## Generic install (any distro)

```bash
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr \
    -DSECP256K1_BUILD_SHARED=ON \
    -DSECP256K1_INSTALL=ON \
    -DSECP256K1_INSTALL_PKGCONFIG=ON \
    -DSECP256K1_USE_ASM=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
sudo cmake --install build
sudo ldconfig
```

After install, applications can find the library via:
- **pkg-config**: `pkg-config --cflags --libs ufsecp`
- **CMake**: `find_package(ufsecp 3 REQUIRED)`

## Package naming convention

| Distro | Runtime | Development |
|--------|---------|-------------|
| Debian/Ubuntu | `libufsecp3` | `libufsecp-dev` |
| Fedora/RHEL | `libufsecp` | `libufsecp-devel` |
| Arch | `libufsecp` | (included in main package) |
