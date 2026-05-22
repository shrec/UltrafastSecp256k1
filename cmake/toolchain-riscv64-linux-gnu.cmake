# ============================================================================
# CMake toolchain file for cross-compiling to Linux RISC-V 64-bit (rv64gc)
# ============================================================================
# Usage:
#   cmake -S . -B out/release-riscv64 \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain-riscv64-linux-gnu.cmake
#
# Or via the matching CMakePresets entry:
#   cmake --preset riscv64-cross
#
# Requires the `riscv64-linux-gnu` cross toolchain (gcc/g++/ar/ranlib/nm)
# to be installed on the build host (e.g. via the
# `gcc-riscv64-linux-gnu g++-riscv64-linux-gnu binutils-riscv64-linux-gnu`
# packages on Debian/Ubuntu).
# ============================================================================

set(CMAKE_SYSTEM_NAME      Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Cross compiler binaries — must be on PATH.
set(CMAKE_C_COMPILER   riscv64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++)
set(CMAKE_ASM_COMPILER riscv64-linux-gnu-gcc)

# Use the target's binutils so the linker recognises the produced objects.
set(CMAKE_AR     riscv64-linux-gnu-ar)
set(CMAKE_RANLIB riscv64-linux-gnu-ranlib)
set(CMAKE_NM     riscv64-linux-gnu-nm)

# Search the cross sysroot for libraries and headers; host programs only.
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Baseline RISC-V Linux profile: rv64gc + zba + zbb (bit-manipulation extensions
# present on every reasonable modern hardware target, including QEMU defaults).
set(CMAKE_C_FLAGS_INIT   "-march=rv64gc_zba_zbb -mabi=lp64d")
set(CMAKE_CXX_FLAGS_INIT "-march=rv64gc_zba_zbb -mabi=lp64d")
