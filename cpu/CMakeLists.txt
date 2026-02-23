cmake_minimum_required(VERSION 3.18)

# Allow standalone builds (when not included as a subdirectory)
if(NOT CMAKE_PROJECT_NAME OR CMAKE_PROJECT_NAME STREQUAL "Project")
    project(UltrafastSecp256k1 LANGUAGES CXX C ASM)
    set(CMAKE_CXX_STANDARD 20)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    include(CTest)
endif()

set(SECP256K1_LIB_NAME fastsecp256k1)

# Core sources (always available - Tier 1: Portable C++)
set(SECP256K1_SOURCES
    src/field.cpp
    src/field_52.cpp   # 5×52 lazy-reduction field (hybrid scheme)
    src/field_26.cpp   # 10×26 lazy-reduction field (32-bit platforms)
    src/scalar.cpp
    src/point.cpp
    src/precompute.cpp
    src/field_asm.cpp  # Tier 2: BMI2 intrinsics (runtime detection)
    src/glv.cpp        # GLV endomorphism optimization
    src/selftest.cpp   # Self-test with known arithmetic vectors
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
    src/batch_add_affine.cpp  # Affine batch addition for sequential ECC search
    src/hash_accel.cpp    # Accelerated SHA-256 (SHA-NI) + RIPEMD-160 + Hash160
    src/pedersen.cpp      # Pedersen commitments (homomorphic)
    src/frost.cpp         # FROST threshold signatures (t-of-n)
    src/adaptor.cpp       # Adaptor signatures (Schnorr + ECDSA)
    src/address.cpp       # Address generation + BIP-352 Silent Payments
    # Coins layer — multi-coin infrastructure
    src/keccak256.cpp     # Keccak-256 hash (Ethereum address derivation)
    src/coin_address.cpp  # Unified per-coin address generation
    src/ethereum.cpp      # Ethereum EIP-55 checksummed addresses
    src/coin_hd.cpp       # BIP-44 coin-type HD derivation
    # Advanced algorithms — Pippenger MSM + Comb generator multiplication
    src/pippenger.cpp        # Pippenger bucket method MSM (n > 128)
    src/ecmult_gen_comb.cpp  # Lim-Lee comb method for fast k*G
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
            list(APPEND SECP256K1_SOURCES
                src/field_asm_x64_gas.S
            )

            # GAS assembler flags
            set(ASM_FLAGS "-x assembler-with-cpp -mbmi2 -madx -fno-lto")
            set_source_files_properties(src/field_asm_x64_gas.S PROPERTIES
                COMPILE_FLAGS "${ASM_FLAGS}")
            set(SECP256K1_HAS_ASM TRUE CACHE INTERNAL "Assembly support enabled")
            message(STATUS "Secp256k1: 5x52 using C __int128 (x64, inlines better than external ASM)")

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

        # 5x52 FE52 assembly: external function calls add 13-register save/restore
        # overhead per call (~26 extra memory ops). On in-order cores (U74, etc.) this
        # negates scheduling benefits. C++ __int128 inline path is 26-33% faster for
        # field ops because the compiler eliminates call overhead and can cross-inline.
        # Enable SECP256K1_USE_RISCV_FE52_ASM=ON only for OoO cores where the explicit
        # register scheduling matters more than inlining.
        option(SECP256K1_USE_RISCV_FE52_ASM "Use hand-written RISC-V assembly for 5x52 field multiply/square (slower on in-order cores like U74)" OFF)
        if(SECP256K1_USE_RISCV_FE52_ASM)
            list(APPEND SECP256K1_SOURCES src/field_asm52_riscv64.S)
            set_source_files_properties(src/field_asm52_riscv64.S PROPERTIES
                COMPILE_FLAGS "-x assembler-with-cpp -march=${RISCV_MARCH_FLAG} -fno-lto")
            add_compile_definitions(SECP256K1_HAS_RISCV_FE52_ASM=1)
            message(STATUS "Secp256k1: 5x52 MUL/MULHU assembly ENABLED (RISC-V) -- external function calls")
        else()
            message(STATUS "Secp256k1: 5x52 field uses C++ __int128 INLINE (faster on in-order RISC-V)")
        endif()

        set(SECP256K1_HAS_ASM TRUE CACHE INTERNAL "Assembly support enabled")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
        # ARM64 (AArch64) Assembly Support
        # Uses MUL/UMULH inline assembly for field arithmetic
        # Available on: Android NDK, Linux aarch64, macOS Apple Silicon
        message(STATUS "Secp256k1: ARM64 inline assembly enabled (MUL/UMULH) - Expected 2-4x speedup on field operations")
        
        list(APPEND SECP256K1_SOURCES
            src/field_asm_arm64.cpp
            src/field_asm52_arm64.cpp
            src/field_asm52_arm64_v2.cpp
        )
        
        add_compile_definitions(SECP256K1_HAS_ARM64_ASM=1)
        add_compile_definitions(SECP256K1_HAS_ARM64_FE52_ASM=1)
        message(STATUS "Secp256k1: 5x52 MUL/UMULH assembly enabled (ARM64)")
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
    add_compile_definitions(SECP256K1_NO_ASM=1)
endif()

if(NOT TARGET ${SECP256K1_LIB_NAME})
    add_library(${SECP256K1_LIB_NAME} STATIC ${SECP256K1_SOURCES})
    add_library(secp256k1::fast ALIAS ${SECP256K1_LIB_NAME})

    # Required for linking into shared libraries (e.g. ufsecp_shared)
    set_target_properties(${SECP256K1_LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # ========================================================================
    # LTO (Link Time Optimization) Support
    # ========================================================================
    # Note: Assembly files (.S) must use -fno-lto, but C++ benefits from LTO
    # We skip CheckIPOSupported because it doesn't use our -fuse-ld=lld flag
    message(STATUS "Secp256k1: LTO check - SECP256K1_USE_LTO=${SECP256K1_USE_LTO}, Compiler=${CMAKE_CXX_COMPILER_ID}")

    if(SECP256K1_USE_LTO)
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
            # Clang ThinLTO - requires lld linker.
            # PRIVATE: compile library sources with LTO (bitcode objects)
            # INTERFACE: propagate LTO + arch flags to ALL consumers automatically
            #   (any exe that links against this lib gets -flto=thin -fuse-ld=lld)
            # CRITICAL: ARCH_FLAGS (e.g. -mcpu=sifive-u74) must be in link options
            #   because ThinLTO does final code generation at link time — without it
            #   the linker uses generic scheduling, losing pipeline-specific gains.
            #   ARCH_FLAGS is added to link options later (after it's set).
            target_compile_options(${SECP256K1_LIB_NAME} PRIVATE -flto=thin)
            target_link_options(${SECP256K1_LIB_NAME} INTERFACE -flto=thin -fuse-ld=lld)
            message(STATUS "Secp256k1: ✓ LTO ENABLED (ThinLTO with Clang + lld, INTERFACE propagated)")
        elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            target_compile_options(${SECP256K1_LIB_NAME} PRIVATE -flto)
            target_link_options(${SECP256K1_LIB_NAME} INTERFACE -flto)
            message(STATUS "Secp256k1: ✓ LTO ENABLED (GCC LTO, INTERFACE propagated)")
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

# Set SOVERSION for shared library builds
set_target_properties(${SECP256K1_LIB_NAME} PROPERTIES
    VERSION   ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
)

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
    # RISC-V CPU tuning: -mcpu tells the compiler exact pipeline model for
    # instruction scheduling. On SiFive U74 (in-order dual-issue), this alone
    # gives 28-34% speedup on field/point operations by enabling U74-specific
    # scheduling heuristics (MUL latency hiding, carry chain interleaving).
    # Set SECP256K1_RISCV_MCPU to override (e.g. "sifive-p550" for OoO cores).
    set(SECP256K1_RISCV_MCPU "" CACHE STRING "RISC-V -mcpu target (e.g. sifive-u74, sifive-p550). Auto-detected if empty.")

    if(SECP256K1_RISCV_MCPU)
        set(RISCV_MCPU "${SECP256K1_RISCV_MCPU}")
    else()
        # Auto-detect from /proc/cpuinfo: "uarch : sifive,u74-mc"
        execute_process(
            COMMAND grep -oP "uarch\\s*:\\s*sifive,\\K[^\\s]+" /proc/cpuinfo
            OUTPUT_VARIABLE RISCV_UARCH_RAW
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        if(RISCV_UARCH_RAW MATCHES "u74")
            set(RISCV_MCPU "sifive-u74")
        elseif(RISCV_UARCH_RAW MATCHES "u54")
            set(RISCV_MCPU "sifive-u54")
        elseif(RISCV_UARCH_RAW MATCHES "p550")
            set(RISCV_MCPU "sifive-p550")
        elseif(RISCV_UARCH_RAW MATCHES "p670")
            set(RISCV_MCPU "sifive-p670")
        else()
            set(RISCV_MCPU "")
        endif()
    endif()

    if(RISCV_MCPU)
        # -mcpu implies -march + -mtune for the exact core.
        # However, Clang's sifive-u74 model may NOT include Zba/Zbb by default
        # even though JH7110 (Milk-V Mars) silicon has them. We need both:
        #   -mcpu for pipeline scheduling + -march for enabling the right ISA extensions.
        # Clang accepts both flags: -mcpu controls scheduling, -march controls ISA.
        set(ARCH_FLAGS "-mcpu=${RISCV_MCPU} -march=rv64gc_zba_zbb")
        message(STATUS "Secp256k1: RISC-V tuned for -mcpu=${RISCV_MCPU} + Zba/Zbb (pipeline-specific scheduling)")
    else()
        set(ARCH_FLAGS "-march=rv64gc_zba_zbb")
        message(STATUS "Secp256k1: RISC-V generic -march=rv64gc_zba_zbb (no CPU-specific tuning)")
    endif()
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64|arm64")
    if(ANDROID)
        # Android NDK: target Cortex-A76 specifically for optimal scheduling
        # -mcpu=cortex-a76 implies armv8.2-a + crypto + lse + rdm + fp16
        # This tells Clang the exact pipeline (3c MUL, 4c UMULH, 1/cycle)
        # so it can interleave MUL/UMULH in __int128 field multiplications.
        set(ARCH_FLAGS "-mcpu=cortex-a76")
        message(STATUS "Secp256k1: Android ARM64 target (Cortex-A76 tuned + crypto + NEON)")
    elseif(APPLE)
        # Apple Silicon: AppleClang uses -mcpu implicitly, not -march
        set(ARCH_FLAGS "")
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

# GCC/Clang optimization flags (skip on MSVC — it uses /O2 /GL from top-level)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
target_compile_options(${SECP256K1_LIB_NAME} PRIVATE 
    -O3                      # Maximum optimization
    "SHELL:${ARCH_FLAGS}"    # Platform-specific architecture (SHELL: splits by whitespace)
    # REMOVED: -ffast-math, -ffinite-math-only, -fno-signed-zeros (break point arithmetic!)
    -fno-math-errno          # Don't set errno for math functions
    -fno-trapping-math       # No FP exceptions
    -funroll-loops           # Aggressive loop unrolling
    -finline-functions       # Inline everything possible
    -fomit-frame-pointer     # Remove frame pointers for speed
    -fno-stack-protector     # Disable stack canaries (no security)
    $<$<PLATFORM_ID:Linux>:-fno-plt>  # No PLT (ELF/Linux only; skipped on macOS/Windows)
    -ftree-vectorize         # Auto-vectorization (AVX2/SSE/NEON)
    # Note: LTO is controlled separately by SECP256K1_USE_LTO option
    # Do NOT add -fno-lto here — it would override the LTO setting
)
# Propagate ARCH_FLAGS to consumers so their TU's also compile with -mcpu
# (important for header-only / inline code and for ThinLTO codegen at link time)
if(ARCH_FLAGS)
    target_compile_options(${SECP256K1_LIB_NAME} INTERFACE "SHELL:${ARCH_FLAGS}")
    # CRITICAL for ThinLTO: final code generation happens at link time.
    # Without -mcpu in link options, the linker uses generic scheduling.
    target_link_options(${SECP256K1_LIB_NAME} INTERFACE "SHELL:${ARCH_FLAGS}")
endif()
endif()

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
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
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
endif()

# Aggressive optimization for critical hotspot files (GCC/Clang only)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(src/field.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize -fipa-pta -fpredictive-commoning"
    )
    set_source_files_properties(src/point.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize -fipa-pta -fpredictive-commoning"
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # Clang: no GCC-specific flags
    set_source_files_properties(src/field.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize"
    )
    set_source_files_properties(src/point.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -funroll-loops -ftree-vectorize"
    )
endif()
# MSVC: uses /O2 /GL from top-level, no per-file overrides needed

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

    # Field 5×52 vs 4×64 comparison benchmark (requires __uint128_t; skip on MSVC)
    if(NOT (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        add_executable(bench_field_52 bench/bench_field_52.cpp)
        target_link_libraries(bench_field_52 PRIVATE ${SECP256K1_LIB_NAME})
    endif()

    # Field 10×26 vs 4×64 comparison benchmark (32-bit platform target)
    add_executable(bench_field_26 bench/bench_field_26.cpp)
    target_link_libraries(bench_field_26 PRIVATE ${SECP256K1_LIB_NAME})

    # Comprehensive cross-platform benchmark (all operations for README performance table)
    add_executable(bench_comprehensive bench/bench_comprehensive_riscv.cpp)
    target_link_libraries(bench_comprehensive PRIVATE ${SECP256K1_LIB_NAME})

    # Legacy alias for compatibility
    add_executable(bench_comprehensive_riscv bench/bench_comprehensive_riscv.cpp)
    target_link_libraries(bench_comprehensive_riscv PRIVATE ${SECP256K1_LIB_NAME})

    # LTO compile options for benchmark sources (link flags auto-propagate via INTERFACE)
    if(SECP256K1_USE_LTO AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(bench_comprehensive PRIVATE -flto=thin)
        target_compile_options(bench_comprehensive_riscv PRIVATE -flto=thin)
        target_compile_options(bench_atomic_operations PRIVATE -flto=thin)
    endif()

    # NOTE: Benchmarks use default optimization (-O3 from CMAKE_CXX_FLAGS_RELEASE)
    # Over-optimizing benchmark code can distort measurements (aggressive inlining, etc.)
endif()

# Tests — unified test runner
# Single binary runs library selftest + all test modules.
# Usage: run_selftest [smoke|ci|stress] [seed_hex]
if(BUILD_TESTING)
    add_executable(run_selftest
        tests/run_selftest.cpp
        tests/test_large_scalar_multiplication.cpp
        tests/test_mul.cpp
        tests/test_arithmetic_correctness.cpp
        tests/test_ct.cpp
        tests/test_ecdsa_schnorr.cpp
        tests/test_multiscalar_batch.cpp
        tests/test_bip32.cpp
        tests/test_musig2.cpp
        tests/test_ecdh_recovery_taproot.cpp
        tests/test_simd_batch.cpp
        tests/test_v4_features.cpp
        tests/test_coins.cpp
        tests/test_batch_add_affine.cpp
        tests/test_hash_accel.cpp
        tests/test_exhaustive.cpp
        tests/test_comprehensive.cpp
        tests/test_bip340_vectors.cpp
        tests/test_rfc6979_vectors.cpp
        tests/test_ecc_properties.cpp
    )
    target_link_libraries(run_selftest PRIVATE ${SECP256K1_LIB_NAME})

    # CT scalar_mul uses large stack (precomputed table + complete addition locals)
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(run_selftest PRIVATE "LINKER:/STACK:8388608")
    endif()

    add_test(NAME selftest COMMAND run_selftest)

    # Standalone affine batch add test (independent of full test runner)
    add_executable(test_batch_add_affine_standalone
        tests/test_batch_add_affine.cpp
    )
    target_link_libraries(test_batch_add_affine_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_batch_add_affine_standalone PRIVATE STANDALONE_TEST)
    add_test(NAME batch_add_affine COMMAND test_batch_add_affine_standalone)

    # Standalone hash accel test
    add_executable(test_hash_accel_standalone
        tests/test_hash_accel.cpp
    )
    target_link_libraries(test_hash_accel_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_hash_accel_standalone PRIVATE STANDALONE_TEST)
    add_test(NAME hash_accel COMMAND test_hash_accel_standalone)

    # Standalone 5×52 field test (requires __uint128_t; skip on MSVC)
    if(NOT (MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang"))
        add_executable(test_field_52_standalone
            tests/test_field_52.cpp
        )
        target_link_libraries(test_field_52_standalone PRIVATE ${SECP256K1_LIB_NAME})
        target_compile_definitions(test_field_52_standalone PRIVATE STANDALONE_TEST)
        add_test(NAME field_52 COMMAND test_field_52_standalone)
    endif()

    # Standalone 10×26 field test
    add_executable(test_field_26_standalone
        tests/test_field_26.cpp
    )
    target_link_libraries(test_field_26_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_field_26_standalone PRIVATE STANDALONE_TEST)
    add_test(NAME field_26 COMMAND test_field_26_standalone)

    # Standalone exhaustive algebraic verification test
    add_executable(test_exhaustive_standalone
        tests/test_exhaustive.cpp
    )
    target_link_libraries(test_exhaustive_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_exhaustive_standalone PRIVATE STANDALONE_TEST)
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_exhaustive_standalone PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME exhaustive COMMAND test_exhaustive_standalone)

    # Standalone comprehensive test (500+ checks across all categories)
    add_executable(test_comprehensive_standalone
        tests/test_comprehensive.cpp
    )
    target_link_libraries(test_comprehensive_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_comprehensive_standalone PRIVATE STANDALONE_TEST)
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_comprehensive_standalone PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME comprehensive COMMAND test_comprehensive_standalone)

    # Standalone BIP-340 official test vectors
    add_executable(test_bip340_vectors_standalone
        tests/test_bip340_vectors.cpp
    )
    target_link_libraries(test_bip340_vectors_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_bip340_vectors_standalone PRIVATE STANDALONE_TEST)
    add_test(NAME bip340_vectors COMMAND test_bip340_vectors_standalone)

    # Standalone RFC 6979 ECDSA test vectors
    add_executable(test_rfc6979_vectors_standalone
        tests/test_rfc6979_vectors.cpp
    )
    target_link_libraries(test_rfc6979_vectors_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_rfc6979_vectors_standalone PRIVATE STANDALONE_TEST)
    add_test(NAME rfc6979_vectors COMMAND test_rfc6979_vectors_standalone)

    # Standalone ECC property-based algebraic invariant tests
    add_executable(test_ecc_properties_standalone
        tests/test_ecc_properties.cpp
    )
    target_link_libraries(test_ecc_properties_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_ecc_properties_standalone PRIVATE STANDALONE_TEST)
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_ecc_properties_standalone PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME ecc_properties COMMAND test_ecc_properties_standalone)

    # Standalone dudect side-channel timing test (from top-level tests/)
    add_executable(test_ct_sidechannel_standalone
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/test_ct_sidechannel.cpp
    )
    target_link_libraries(test_ct_sidechannel_standalone PRIVATE ${SECP256K1_LIB_NAME})
    target_compile_definitions(test_ct_sidechannel_standalone PRIVATE STANDALONE_TEST)
    if(MSVC OR (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32))
        target_link_options(test_ct_sidechannel_standalone PRIVATE "LINKER:/STACK:8388608")
    endif()
    add_test(NAME ct_sidechannel COMMAND test_ct_sidechannel_standalone)
    set_tests_properties(ct_sidechannel PROPERTIES TIMEOUT 300)

    # Standalone differential/self-consistency test (from top-level tests/)
    add_executable(test_differential_standalone
        ${CMAKE_CURRENT_SOURCE_DIR}/../tests/differential_test.cpp
    )
    target_link_libraries(test_differential_standalone PRIVATE ${SECP256K1_LIB_NAME})
    add_test(NAME differential COMMAND test_differential_standalone)
    set_tests_properties(differential PROPERTIES TIMEOUT 120)
endif()
