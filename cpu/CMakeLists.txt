set(SECP256K1_LIB_NAME fastsecp256k1)

# Core sources (always available - Tier 1: Portable C++)
set(SECP256K1_SOURCES
    src/field.cpp
    src/scalar.cpp
    src/point.cpp
    src/precompute.cpp
    src/field_asm.cpp  # Tier 2: BMI2 intrinsics (runtime detection)
    src/glv.cpp        # GLV endomorphism optimization
    src/selftest.cpp   # Self-test with known arithmetic vectors
    src/sorted_ecc_db.cpp  # SortedEccDB - native x86/x64 database
    # Constant-Time (CT) layer — always compiled, no flags
    src/ct_field.cpp   # CT field arithmetic (side-channel resistant)
    src/ct_scalar.cpp  # CT scalar arithmetic
    src/ct_point.cpp   # CT point ops (complete addition, CT scalar_mul)
    src/ecdsa.cpp      # ECDSA sign/verify + RFC 6979
    src/schnorr.cpp    # Schnorr BIP-340 sign/verify
    src/multiscalar.cpp  # Multi-scalar multiplication (Strauss/Shamir)
    src/batch_verify.cpp # Batch ECDSA/Schnorr verification
    src/bip32.cpp        # BIP-32 HD key derivation
    src/musig2.cpp       # MuSig2 multi-signatures (BIP-327)
    src/ecdh.cpp         # ECDH key exchange
    src/recovery.cpp     # ECDSA public key recovery
    src/taproot.cpp      # Taproot (BIP-341/342) key tweaking
    src/field_simd.cpp   # SIMD batch field operations + Montgomery batch inverse
    src/pedersen.cpp      # Pedersen commitments (homomorphic)
    src/frost.cpp         # FROST threshold signatures (t-of-n)
    src/adaptor.cpp       # Adaptor signatures (Schnorr + ECDSA)
    src/address.cpp       # Address generation + BIP-352 Silent Payments
)

# Optional inline assembly (Tier 3: Maximum performance, x64/RISC-V)
option(SECP256K1_USE_ASM "Enable inline assembly optimizations (x64/RISC-V, 2-5x speedup)" ON)

# Fast modular reduction (platform-independent)
option(SECP256K1_USE_FAST_REDUCTION "Use fast modular reduction (RISC-V asm, x64 BMI2)" ON)

# RISC-V specific options (only relevant when building on RISC-V)
option(SECP256K1_RISCV_USE_VECTOR "Enable RISC-V Vector Extension (RVV) if available" ON)
option(SECP256K1_RISCV_USE_PREFETCH "Enable prefetch hints for cache optimization" ON)

# LTO/PGO options
option(SECP256K1_USE_LTO "Enable Link Time Optimization (LTO) for C++ code" OFF)
option(SECP256K1_USE_PGO_GEN "Enable Profile-Guided Optimization - Generate profile" OFF)
option(SECP256K1_USE_PGO_USE "Enable Profile-Guided Optimization - Use profile" OFF)
set(SECP256K1_PGO_PROFILE_DIR "${CMAKE_BINARY_DIR}/pgo_profiles" CACHE PATH "Directory for PGO profiles")

if(SECP256K1_USE_ASM)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X64" AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        # x64 Assembly (Windows/Linux/macOS)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
            # GAS syntax for Clang/GCC (Windows MinGW, Linux, macOS)
            enable_language(ASM)
            message(STATUS "Secp256k1: Inline assembly enabled (x64 GAS) - Expected 3-5x speedup on field operations")
            list(APPEND SECP256K1_SOURCES src/field_asm_x64_gas.S)

            # GAS assembler flags
            set(ASM_FLAGS "-x assembler-with-cpp -mbmi2 -madx -fno-lto")
            set_source_files_properties(src/field_asm_x64_gas.S PROPERTIES
                COMPILE_FLAGS "${ASM_FLAGS}")
            set(SECP256K1_HAS_ASM TRUE CACHE INTERNAL "Assembly support enabled")

            # Enable fast reduction on x64 (uses BMI2/ADX)
            if(SECP256K1_USE_FAST_REDUCTION)
                add_compile_definitions(SECP256K1_USE_FAST_REDUCTION=1)
                message(STATUS "Secp256k1: Fast modular reduction enabled (x64 BMI2/ADX)")
            endif()
        elseif(MSVC)
            # MASM syntax for MSVC
            enable_language(ASM_MASM)
            message(STATUS "Secp256k1: Inline assembly enabled (x64 MASM) - Expected 3-5x speedup on field operations")
            list(APPEND SECP256K1_SOURCES src/field_asm_x64.asm)
            set_source_files_properties(src/field_asm_x64.asm PROPERTIES COMPILE_FLAGS "/c /Zi")
            set(SECP256K1_HAS_ASM TRUE CACHE INTERNAL "Assembly support enabled")

            if(SECP256K1_USE_FAST_REDUCTION)
                add_compile_definitions(SECP256K1_USE_FAST_REDUCTION=1)
                message(STATUS "Secp256k1: Fast modular reduction enabled (x64 BMI2/ADX)")
            endif()
        else()
            message(WARNING "Secp256k1: Unsupported compiler ${CMAKE_CXX_COMPILER_ID} for x64 assembly")
            set(SECP256K1_HAS_ASM FALSE)
        endif()
    elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
        # RISC-V 64-bit Assembly Support
        enable_language(ASM)

        # Check for Vector Extension support
        if(SECP256K1_RISCV_USE_VECTOR)
            # Try to detect if CPU supports RVV
            # Check both /proc/cpuinfo (Linux) and manually enable if user requested
            execute_process(
                COMMAND cat /proc/cpuinfo
                OUTPUT_VARIABLE RISCV_CPUINFO
                ERROR_QUIET
            )

            # Look for 'v' in ISA extensions (format: "isa : rv64imafdcv...")
            if(RISCV_CPUINFO MATCHES "isa[^:]*:[^v]*v")
                set(RISCV_HAS_VECTOR TRUE)
            else()
                set(RISCV_HAS_VECTOR FALSE)
            endif()

            # Also try compile test as fallback
            if(NOT RISCV_HAS_VECTOR)
                include(CheckCXXSourceCompiles)
                set(CMAKE_REQUIRED_FLAGS "-march=rv64gcv")
                check_cxx_source_compiles("
                    #include <riscv_vector.h>
                    int main() {
                        vint64m1_t v = __riscv_vmv_v_x_i64m1(0, 4);
                        return 0;
                    }" RISCV_COMPILER_SUPPORTS_VECTOR)
                if(RISCV_COMPILER_SUPPORTS_VECTOR)
                    set(RISCV_HAS_VECTOR TRUE)
                endif()
            endif()

            if(RISCV_HAS_VECTOR)
                set(RISCV_MARCH_FLAG "rv64gcv_zba_zbb")
                set(RISCV_FEATURES "+Vector")
                add_compile_definitions(SECP256K1_HAS_RISCV_VECTOR=1)
                add_compile_definitions(SECP256K1_RISCV_USE_VECTOR=1)
                message(STATUS "Secp256k1: RISC-V Vector Extension (RVV) detected and enabled")
            else()
                set(RISCV_MARCH_FLAG "rv64gc_zba_zbb")
                message(STATUS "Secp256k1: RISC-V Vector Extension not detected, using scalar only")
            endif()
        else()
            set(RISCV_MARCH_FLAG "rv64gc_zba_zbb")
        endif()
        
        # Fast reduction option
        if(SECP256K1_USE_FAST_REDUCTION)
            add_compile_definitions(SECP256K1_USE_FAST_REDUCTION=1)
            message(STATUS "Secp256k1: Fast modular reduction enabled")
        endif()
        
        # Prefetch hints option
        if(SECP256K1_RISCV_USE_PREFETCH)
            add_compile_definitions(SECP256K1_RISCV_USE_PREFETCH=1)
            message(STATUS "Secp256k1: Prefetch hints enabled")
        endif()
        
        message(STATUS "Secp256k1: Inline assembly enabled (RISC-V 64-bit ${RISCV_FEATURES}) - Expected 2-3x speedup")

        list(APPEND SECP256K1_SOURCES
            src/field_asm_riscv64.S
            src/field_asm_riscv64.cpp
        )
        
        # RISC-V assembler flags with detected march
        set_source_files_properties(src/field_asm_riscv64.S PROPERTIES 
            COMPILE_FLAGS "-x assembler-with-cpp -march=${RISCV_MARCH_FLAG} -fno-lto")

        add_compile_definitions(SECP256K1_HAS_RISCV_ASM=1)
        set(SECP256K1_HAS_ASM TRUE CACHE INTERNAL "Assembly support enabled")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        # ARM64 (AArch64) Assembly Support
        # Uses MUL/UMULH inline assembly for field arithmetic
        # Available on: Android NDK, Linux aarch64, macOS Apple Silicon
        message(STATUS "Secp256k1: ARM64 inline assembly enabled (MUL/UMULH) - Expected 2-4x speedup on field operations")
        
        list(APPEND SECP256K1_SOURCES
            src/field_asm_arm64.cpp
        )
        
        add_compile_definitions(SECP256K1_HAS_ARM64_ASM=1)
        set(SECP256K1_HAS_ASM TRUE CACHE INTERNAL "Assembly support enabled")
        
        if(SECP256K1_USE_FAST_REDUCTION)
            add_compile_definitions(SECP256K1_USE_FAST_REDUCTION=1)
            message(STATUS "Secp256k1: Fast modular reduction enabled (ARM64 secp256k1-specific)")
        endif()
    else()
        message(WARNING "Secp256k1: Inline assembly requested but not available for ${CMAKE_SYSTEM_PROCESSOR} on ${CMAKE_SYSTEM_NAME}")
        message(WARNING "  Assembly is enabled only on: x86_64, RISC-V 64-bit, ARM64 (aarch64). Falling back.")
        set(SECP256K1_HAS_ASM FALSE)
    endif()
else()
    set(SECP256K1_HAS_ASM FALSE)
endif()

if(NOT TARGET ${SECP256K1_LIB_NAME})
    add_library(${SECP256K1_LIB_NAME} STATIC ${SECP256K1_SOURCES})
    add_library(secp256k1::fast ALIAS ${SECP256K1_LIB_NAME})

    # ========================================================================
    # LTO (Link Time Optimization) Support
    # ========================================================================
    # Note: Assembly files (.S) must use -fno-lto, but C++ benefits from LTO
    # We skip CheckIPOSupported because it doesn't use our -fuse-ld=lld flag
    message(STATUS "Secp256k1: LTO check - SECP256K1_USE_LTO=${SECP256K1_USE_LTO}, Compiler=${CMAKE_CXX_COMPILER_ID}")

    if(SECP256K1_USE_LTO)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang ThinLTO - requires lld linker (set via CMAKE_EXE_LINKER_FLAGS)
            target_compile_options(${SECP256K1_LIB_NAME} PRIVATE -flto=thin)
            message(STATUS "Secp256k1: ✓ LTO ENABLED (ThinLTO with Clang, expect 5-15% improvement)")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            target_compile_options(${SECP256K1_LIB_NAME} PRIVATE -flto)
            target_link_options(${SECP256K1_LIB_NAME} PRIVATE -flto)
            message(STATUS "Secp256k1: ✓ LTO ENABLED (GCC LTO, expect 5-15% improvement)")
        else()
            message(STATUS "Secp256k1: LTO not available for compiler ${CMAKE_CXX_COMPILER_ID}")
        endif()
    else()
        message(STATUS "Secp256k1: LTO disabled (use -DSECP256K1_USE_LTO=ON to enable)")
    endif()

    # ========================================================================
    # PGO (Profile-Guided Optimization) Support
    # ========================================================================
    # Usage:
    #   1. Build with -DSECP256K1_USE_PGO_GEN=ON
    #   2. Run representative workload (benchmark)
    #   3. Rebuild with -DSECP256K1_USE_PGO_USE=ON
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        if(SECP256K1_USE_PGO_GEN)
            message(STATUS "Secp256k1: PGO instrumentation enabled (generating profiles)")
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                target_compile_options(${SECP256K1_LIB_NAME} PRIVATE
                    -fprofile-generate=${SECP256K1_PGO_PROFILE_DIR}
                )
                target_link_options(${SECP256K1_LIB_NAME} PRIVATE
                    -fprofile-generate=${SECP256K1_PGO_PROFILE_DIR}
                )
            else()  # GCC
                target_compile_options(${SECP256K1_LIB_NAME} PRIVATE
                    -fprofile-generate -fprofile-dir=${SECP256K1_PGO_PROFILE_DIR}
                )
                target_link_options(${SECP256K1_LIB_NAME} PRIVATE
                    -fprofile-generate
                )
            endif()
        elseif(SECP256K1_USE_PGO_USE)
            message(STATUS "Secp256k1: PGO optimization enabled (using profiles from ${SECP256K1_PGO_PROFILE_DIR})")
            if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
                # Clang needs merged profile
                set(PGO_PROFILE_FILE "${SECP256K1_PGO_PROFILE_DIR}/default.profdata")
                if(EXISTS "${PGO_PROFILE_FILE}")
                    target_compile_options(${SECP256K1_LIB_NAME} PRIVATE
                        -fprofile-use=${PGO_PROFILE_FILE}
                    )
                else()
                    message(WARNING "Secp256k1: PGO profile not found at ${PGO_PROFILE_FILE}")
                    message(WARNING "  Run: llvm-profdata merge -o ${PGO_PROFILE_FILE} ${SECP256K1_PGO_PROFILE_DIR}/*.profraw")
                endif()
            else()  # GCC
                target_compile_options(${SECP256K1_LIB_NAME} PRIVATE
                    -fprofile-use -fprofile-dir=${SECP256K1_PGO_PROFILE_DIR}
                    -fprofile-correction  # Handle missing data gracefully
                )
            endif()
        endif()
    elseif(MSVC)
        if(SECP256K1_USE_PGO_GEN)
            message(STATUS "Secp256k1: MSVC PGO instrumentation enabled")
            target_compile_options(${SECP256K1_LIB_NAME} PRIVATE /GL)
            target_link_options(${SECP256K1_LIB_NAME} PRIVATE /LTCG /GENPROFILE:PGD=${SECP256K1_PGO_PROFILE_DIR}/secp256k1.pgd)
        elseif(SECP256K1_USE_PGO_USE)
            set(PGO_PGD_FILE "${SECP256K1_PGO_PROFILE_DIR}/secp256k1.pgd")
            if(EXISTS "${PGO_PGD_FILE}")
                message(STATUS "Secp256k1: MSVC PGO optimization enabled (using ${PGO_PGD_FILE})")
                target_compile_options(${SECP256K1_LIB_NAME} PRIVATE /GL)
                target_link_options(${SECP256K1_LIB_NAME} PRIVATE /LTCG /USEPROFILE:PGD=${PGO_PGD_FILE})
            else()
                message(WARNING "Secp256k1: PGO profile not found at ${PGO_PGD_FILE}")
            endif()
        endif()
    endif()

    target_include_directories(${SECP256K1_LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}  # For rdtsc.h and intrinsics.h
)

target_compile_features(${SECP256K1_LIB_NAME} PUBLIC cxx_std_20)
# Install target
if(SECP256K1_INSTALL)
    include(GNUInstallDirs)
    install(TARGETS ${SECP256K1_LIB_NAME}
        EXPORT secp256k1-fast-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
    install(EXPORT secp256k1-fast-targets
        FILE secp256k1-fast-targets.cmake
        NAMESPACE secp256k1::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/secp256k1-fast
    )
endif()
target_compile_definitions(${SECP256K1_LIB_NAME}
    PUBLIC
        SECP256K1_FAST_NO_SECURITY_CHECKS=1
        SECP256K1_ULTRA_SPEED=1        # Maximum speed mode
        $<$<CONFIG:Release>:NDEBUG>    # Remove assertions only in Release
        $<$<CXX_COMPILER_ID:MSVC>:SECP256K1_NO_INT128=1>  # MSVC has no __int128
)

# Optional inline assembly support (x64 only)
if(SECP256K1_HAS_ASM)
    target_compile_definitions(${SECP256K1_LIB_NAME} 
        PUBLIC SECP256K1_HAS_ASM=1
        PRIVATE USE_INLINE_ASSEMBLY
    )
    message(STATUS "  -> field_mul: ~8ns (vs 27ns intrinsics, 40ns portable)")
    message(STATUS "  -> field_square: ~7ns (vs 21ns intrinsics, 35ns portable)")
    message(STATUS "  -> Expected K*Q: ~18-24 μs (vs 66 μs current)")
endif()

# Enable fast modular reduction on x86_64 (even without ASM, uses BMI2 intrinsics)
if(SECP256K1_USE_FAST_REDUCTION AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X64")
    target_compile_definitions(${SECP256K1_LIB_NAME} PUBLIC SECP256K1_USE_FAST_REDUCTION=1)
    if(NOT SECP256K1_HAS_ASM)
        message(STATUS "Secp256k1: Fast modular reduction enabled (x64 BMI2 intrinsics)")
    endif()
endif()

# ============================================================================
# Compiler Optimizations: Clang/GCC Primary (MSVC deprecated)
# ============================================================================
# Target: Maximum performance with Clang 17+ or GCC 11+
# Platform: Windows (MinGW/LLVM-MinGW), Linux, macOS
# Note: MSVC support removed - use Clang-cl if needed on Windows

# Platform-specific architecture flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
    set(ARCH_FLAGS "-march=rv64gc_zba_zbb")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    if(ANDROID)
        # Android NDK: target ARMv8-A with crypto extensions (NEON is implicit)
        set(ARCH_FLAGS "-march=armv8-a+crypto")
        message(STATUS "Secp256k1: Android ARM64 target (ARMv8-A + crypto + NEON)")
    else()
        set(ARCH_FLAGS "-march=armv8-a+crypto")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|armeabi")
    # Android ARMv7 (32-bit) — no __int128, uses NO_INT128 fallback
    set(ARCH_FLAGS "-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=softfp")
    add_compile_definitions(SECP256K1_NO_INT128=1)
    message(STATUS "Secp256k1: Android ARMv7 target (32-bit, no __int128)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X64")
    if(ANDROID)
        # Android x86_64 emulator — no -march=native for cross-compile
        set(ARCH_FLAGS "-march=x86-64 -msse4.2")
        message(STATUS "Secp256k1: Android x86_64 target (emulator)")
    else()
        set(ARCH_FLAGS "-march=native")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|x86")
    if(ANDROID)
        set(ARCH_FLAGS "-march=i686 -msse3")
        add_compile_definitions(SECP256K1_NO_INT128=1)
        message(STATUS "Secp256k1: Android x86 target (32-bit emulator, no __int128)")
    else()
        set(ARCH_FLAGS "")
    endif()
else()
    set(ARCH_FLAGS "")
endif()

target_compile_options(${SECP256K1_LIB_NAME} PRIVATE 
    -O3                      # Maximum optimization
    ${ARCH_FLAGS}            # Platform-specific architecture
    # REMOVED: -ffast-math, -ffinite-math-only, -fno-signed-zeros (break point arithmetic!)
    -fno-math-errno          # Don't set errno for math functions
    -fno-trapping-math       # No FP exceptions
    -funroll-loops           # Aggressive loop unrolling
    -finline-functions       # Inline everything possible
    -fomit-frame-pointer     # Remove frame pointers for speed
    -fno-stack-protector     # Disable stack canaries (no security)
    -fno-plt                 # No PLT indirection (faster calls)
    -ftree-vectorize         # Auto-vectorization (AVX2/SSE)
    -fno-lto                 # Explicitly disable LTO to avoid linker format errors
)

# GCC-specific aggressive optimizations (not supported by Clang)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${SECP256K1_LIB_NAME} PRIVATE
        -fipa-pta                # Aggressive inter-procedural pointer analysis
        -fgcse-sm                # Store motion after GCSE
        -fgcse-las               # Load-after-store elimination
        -ftree-loop-im           # Loop invariant motion
        -ftree-loop-ivcanon      # Induction variable canonicalization
        -fivopts                 # Induction variable optimizations
        -fvariable-expansion-in-unroller  # Variable expansion in unroller
        -fpredictive-commoning   # Predictive commoning optimization
        -fopt-info-vec-optimized # Report vectorization successes
    )
endif()

# Special flags for field_asm.cpp (maximize BMI2/MULX utilization on x86)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X64")
    set_source_files_properties(src/field_asm.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -mbmi2 -madx -mpopcnt -funroll-loops -ftree-vectorize -fno-exceptions -fno-rtti"
    )
else()
    # ARM64/other: no BMI2/ADX, just standard optimization
    set_source_files_properties(src/field_asm.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize -fno-exceptions -fno-rtti"
    )
endif()

# Aggressive optimization for critical hotspot files
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(src/field.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize -fipa-pta -fpredictive-commoning"
    )
    set_source_files_properties(src/point.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize -fipa-pta -fpredictive-commoning"
    )
else()
    # Clang/other compilers (no GCC-specific flags)
    set_source_files_properties(src/field.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize"
    )
    set_source_files_properties(src/point.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize"
    )
endif()

# Static linking for MinGW (portable executables without DLL dependencies)
    if(MINGW)
        target_link_options(${SECP256K1_LIB_NAME} INTERFACE
            -static-libgcc
            -static-libstdc++
            -static
        )
    endif()
endif()

# Benchmarks (only build if testing is enabled)
if(BUILD_TESTING)
    add_executable(bench_scalar_mul bench/bench_scalar_mul.cpp)
    target_link_libraries(bench_scalar_mul PRIVATE ${SECP256K1_LIB_NAME})

    # JSF vs Shamir (GLV) microbenchmark
    add_executable(bench_jsf_vs_shamir bench/bench_jsf_vs_shamir.cpp)
    target_link_libraries(bench_jsf_vs_shamir PRIVATE ${SECP256K1_LIB_NAME})

    # ECDSA-style multi-scalar microbenchmark (k1*G + k2*Q)
    add_executable(bench_ecdsa_multiscalar bench/bench_ecdsa_multiscalar.cpp)
    target_link_libraries(bench_ecdsa_multiscalar PRIVATE ${SECP256K1_LIB_NAME})

    # GLV decomposition profiling (RDTSC cycle breakdown)
    add_executable(bench_glv_decomp_profile bench/bench_glv_decomp_profile.cpp)
    target_link_libraries(bench_glv_decomp_profile PRIVATE ${SECP256K1_LIB_NAME})

    # Adaptive GLV threshold sweep benchmark
    add_executable(bench_adaptive_glv bench/bench_adaptive_glv.cpp)
    target_link_libraries(bench_adaptive_glv PRIVATE ${SECP256K1_LIB_NAME})

    # Field mul kernel benchmark (BMI2 vs ASM)
    add_executable(bench_field_mul_kernels bench/bench_field_mul_kernels.cpp)
    target_link_libraries(bench_field_mul_kernels PRIVATE ${SECP256K1_LIB_NAME})

    # Atomic operations micro-benchmark (Point Add/Double, Field ops, Inverse)
    add_executable(bench_atomic_operations bench/bench_atomic_operations.cpp)
    target_link_libraries(bench_atomic_operations PRIVATE ${SECP256K1_LIB_NAME})

    # CT (Constant-Time) layer benchmark — fast:: vs ct:: overhead comparison
    add_executable(bench_ct bench/bench_ct.cpp)
    target_link_libraries(bench_ct PRIVATE ${SECP256K1_LIB_NAME})

    # Comprehensive cross-platform benchmark (all operations for README performance table)
    add_executable(bench_comprehensive bench/bench_comprehensive_riscv.cpp)
    target_link_libraries(bench_comprehensive PRIVATE ${SECP256K1_LIB_NAME})

    # Legacy alias for compatibility
    add_executable(bench_comprehensive_riscv bench/bench_comprehensive_riscv.cpp)
    target_link_libraries(bench_comprehensive_riscv PRIVATE ${SECP256K1_LIB_NAME})

    # LTO link options for benchmarks (must use lld for Clang LTO)
    if(SECP256K1_USE_LTO AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(bench_comprehensive PRIVATE -flto=thin)
        target_link_options(bench_comprehensive PRIVATE -flto=thin -fuse-ld=lld)
        target_compile_options(bench_comprehensive_riscv PRIVATE -flto=thin)
        target_link_options(bench_comprehensive_riscv PRIVATE -flto=thin -fuse-ld=lld)
        target_compile_options(bench_atomic_operations PRIVATE -flto=thin)
        target_link_options(bench_atomic_operations PRIVATE -flto=thin -fuse-ld=lld)
    endif()

    # NOTE: Benchmarks use default optimization (-O3 from CMAKE_CXX_FLAGS_RELEASE)
    # Over-optimizing benchmark code can distort measurements (aggressive inlining, etc.)
endif()

# Tests
if(BUILD_TESTING)
    # Test: Large scalar multiplication with guaranteed correct results
    add_executable(test_large_scalar_multiplication tests/test_large_scalar_multiplication.cpp)
    target_link_libraries(test_large_scalar_multiplication PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_large_scalar_multiplication COMMAND test_large_scalar_multiplication)
    
    # Test: Simple multiplication for debugging
    add_executable(test_mul tests/test_mul.cpp)
    target_link_libraries(test_mul PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_mul COMMAND test_mul)

    # Test: Comprehensive arithmetic correctness (K*G, P+Q, K*Q, distributive)
    add_executable(test_arithmetic_correctness tests/test_arithmetic_correctness.cpp)
    target_link_libraries(test_arithmetic_correctness PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_arithmetic_correctness COMMAND test_arithmetic_correctness)

    # Test: Constant-Time (CT) layer correctness
    add_executable(test_ct tests/test_ct.cpp)
    target_link_libraries(test_ct PRIVATE ${SECP256K1_LIB_NAME})
    # CT scalar_mul uses large stack (precomputed table + complete addition locals)
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_ct PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME test_ct COMMAND test_ct)

    # Test: ECDSA + Schnorr (BIP-340) sign/verify
    add_executable(test_ecdsa_schnorr tests/test_ecdsa_schnorr.cpp)
    target_link_libraries(test_ecdsa_schnorr PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_ecdsa_schnorr COMMAND test_ecdsa_schnorr)

    # Test: Multi-scalar multiplication + Batch verification
    add_executable(test_multiscalar_batch tests/test_multiscalar_batch.cpp)
    target_link_libraries(test_multiscalar_batch PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_multiscalar_batch COMMAND test_multiscalar_batch)

    # Test: BIP-32 HD key derivation
    add_executable(test_bip32 tests/test_bip32.cpp)
    target_link_libraries(test_bip32 PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_bip32 COMMAND test_bip32)

    # Test: MuSig2 multi-signatures
    add_executable(test_musig2 tests/test_musig2.cpp)
    target_link_libraries(test_musig2 PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_musig2 COMMAND test_musig2)

    # Test: ECDH + Recovery + Taproot + CT Utils + Wycheproof vectors
    add_executable(test_ecdh_recovery_taproot tests/test_ecdh_recovery_taproot.cpp)
    target_link_libraries(test_ecdh_recovery_taproot PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_ecdh_recovery_taproot COMMAND test_ecdh_recovery_taproot)

    # Test: SIMD batch field operations
    add_executable(test_simd_batch tests/test_simd_batch.cpp)
    target_link_libraries(test_simd_batch PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_simd_batch COMMAND test_simd_batch)

    # Test: v4.0.0 features — Pedersen, FROST, Adaptor, Address, Silent Payments
    add_executable(test_v4_features tests/test_v4_features.cpp)
    target_link_libraries(test_v4_features PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME test_v4_features COMMAND test_v4_features)
endif()
