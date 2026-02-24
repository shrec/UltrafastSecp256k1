%global soversion 3

Name:           libufsecp
Version:        3.12.1
Release:        1%{?dist}
Summary:        High-performance secp256k1 elliptic curve cryptography library
License:        AGPL-3.0-or-later
URL:            https://github.com/shrec/UltrafastSecp256k1
Source0:        %{url}/archive/v%{version}/UltrafastSecp256k1-%{version}.tar.gz

BuildRequires:  cmake >= 3.18
BuildRequires:  ninja-build
BuildRequires:  gcc-c++ >= 11
BuildRequires:  pkgconfig

%description
UltrafastSecp256k1 is a high-performance implementation of the secp256k1
elliptic curve used by Bitcoin, Ethereum, and other cryptocurrencies.
Features: constant-time operations, GLV endomorphism, ECDSA/Schnorr
signatures, BIP-32/340/341 support, SIMD acceleration, and multi-backend
GPU support (CUDA, OpenCL, Metal).

%package        devel
Summary:        Development files for %{name}
Requires:       %{name}%{?_isa} = %{version}-%{release}

%description    devel
This package contains the development headers, static library,
CMake config, and pkg-config files for %{name}.

%prep
%autosetup -n UltrafastSecp256k1-%{version}

%build
%cmake \
    -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DSECP256K1_BUILD_TESTS=ON \
    -DSECP256K1_BUILD_BENCH=OFF \
    -DSECP256K1_BUILD_EXAMPLES=OFF \
    -DSECP256K1_BUILD_SHARED=ON \
    -DSECP256K1_INSTALL=ON \
    -DSECP256K1_INSTALL_PKGCONFIG=ON \
    -DSECP256K1_USE_ASM=ON
%cmake_build

%check
%ctest

%install
%cmake_install

%files
%license LICENSE
%doc README.md CHANGELOG.md
%{_libdir}/libfastsecp256k1.so.%{soversion}*
%{_libdir}/libufsecp.so.%{soversion}*

%files devel
%doc docs/
%{_includedir}/secp256k1/
%{_includedir}/ufsecp/
%{_libdir}/libfastsecp256k1.so
%{_libdir}/libufsecp.so
%{_libdir}/libfastsecp256k1.a
%{_libdir}/pkgconfig/secp256k1-fast.pc
%{_libdir}/cmake/secp256k1-fast/

%changelog
* Sun Feb 23 2026 shrec <shrec@users.noreply.github.com> - 3.12.1-1
- New upstream release
- Security: bump wheel 0.45.1 -> 0.46.2 (CVE-2026-24049)
- Security: bump setuptools 75.8.0 -> 78.1.1 (CVE-2025-47273)
