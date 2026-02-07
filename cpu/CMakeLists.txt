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
)

# Optional inline assembly (Tier 3: Maximum performance, x64/RISC-V)
option(SECP256K1_USE_ASM "Enable inline assembly optimizations (x64/RISC-V, 2-5x speedup)" ON)

# RISC-V specific options (only relevant when building on RISC-V)
option(SECP256K1_RISCV_USE_VECTOR "Enable RISC-V Vector Extension (RVV) if available" ON)
option(SECP256K1_RISCV_FAST_REDUCTION "Use fast modular reduction in assembly" ON)
option(SECP256K1_RISCV_USE_PREFETCH "Enable prefetch hints for cache optimization" ON)

if(SECP256K1_USE_ASM)
    if(MSVC AND CMAKE_SIZEOF_VOID_P EQUAL 8)
        # Windows x64 with MASM assembler
        enable_language(ASM_MASM)
        message(STATUS "Secp256k1: Inline assembly enabled (x64 MASM) - Expected 3-5x speedup on field operations")
        list(APPEND SECP256K1_SOURCES src/field_asm_x64.asm)
        set_source_files_properties(src/field_asm_x64.asm PROPERTIES COMPILE_FLAGS "/c /Zi")
        set(SECP256K1_HAS_ASM TRUE CACHE INTERNAL "Assembly support enabled")
    elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|X64")
        # Linux x64 with GAS assembler (GNU Assembler, AT&T/Intel syntax)
        # Ensure ASM language is enabled for this directory (may already be enabled at top-level).
        enable_language(ASM)
        message(STATUS "Secp256k1: Inline assembly enabled (x64 GAS) - Expected 3-5x speedup on field operations")
        list(APPEND SECP256K1_SOURCES src/field_asm_x64_gas.S)
        # GAS assembler flags: allow Intel syntax, enable BMI2/ADX
        set_source_files_properties(src/field_asm_x64_gas.S PROPERTIES 
            COMPILE_FLAGS "-x assembler-with-cpp -mbmi2 -madx -fno-lto")
        set(SECP256K1_HAS_ASM TRUE CACHE INTERNAL "Assembly support enabled")
    elseif(UNIX AND CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64|RISCV64")
        # RISC-V 64-bit with GAS assembler
        enable_language(ASM)
        
        # Base RISC-V march flag
        set(RISCV_MARCH_FLAG "rv64gc")
        set(RISCV_FEATURES "RV64GC")
        
        # Check for Vector Extension support
        if(SECP256K1_RISCV_USE_VECTOR)
            # Try to detect if CPU supports RVV
            execute_process(
                COMMAND sh -c "cat /proc/cpuinfo | grep -i 'isa' | head -1"
                OUTPUT_VARIABLE RISCV_ISA_STRING
                ERROR_QUIET
            )
            if(RISCV_ISA_STRING MATCHES "v")
                set(RISCV_MARCH_FLAG "rv64gcv")
                set(RISCV_FEATURES "${RISCV_FEATURES}+Vector")
                add_compile_definitions(SECP256K1_HAS_RISCV_VECTOR=1)
                message(STATUS "Secp256k1: RISC-V Vector Extension (RVV) detected and enabled")
            else()
                message(STATUS "Secp256k1: RISC-V Vector Extension not detected, using scalar only")
            endif()
        endif()
        
        # Fast reduction option
        if(SECP256K1_RISCV_FAST_REDUCTION)
            add_compile_definitions(SECP256K1_RISCV_FAST_REDUCTION=1)
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
    else()
        message(WARNING "Secp256k1: Inline assembly requested but not available for ${CMAKE_SYSTEM_PROCESSOR}")
        message(WARNING "  Assembly is x64/RISC-V only. Will use BMI2 intrinsics (runtime) + portable C++ fallback.")
        set(SECP256K1_HAS_ASM FALSE)
    endif()
else()
    set(SECP256K1_HAS_ASM FALSE)
endif()

if(NOT TARGET ${SECP256K1_LIB_NAME})
    add_library(${SECP256K1_LIB_NAME} STATIC ${SECP256K1_SOURCES})
    add_library(secp256k1::fast ALIAS ${SECP256K1_LIB_NAME})

    # Enable IPO/LTO for this CPU library in Release on native (non-Windows) builds.
    # This is intentionally per-target to avoid leaking LTO flags into CUDA device-link steps.
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
        set_property(TARGET ${SECP256K1_LIB_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()

    target_include_directories(${SECP256K1_LIB_NAME}
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}  # For rdtsc.h and intrinsics.h
)

target_compile_features(${SECP256K1_LIB_NAME} PUBLIC cxx_std_20)

target_compile_definitions(${SECP256K1_LIB_NAME}
    PUBLIC 
        SECP256K1_FAST_NO_SECURITY_CHECKS=1
        SECP256K1_ULTRA_SPEED=1        # Maximum speed mode
        NDEBUG                          # Remove all assertions
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

# ============================================================================
# Compiler Optimizations: Clang/GCC Primary (MSVC deprecated)
# ============================================================================
# Target: Maximum performance with Clang 17+ or GCC 11+
# Platform: Windows (MinGW/LLVM-MinGW), Linux, macOS
# Note: MSVC support removed - use Clang-cl if needed on Windows

# Platform-specific architecture flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set(ARCH_FLAGS "-march=rv64gc")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    set(ARCH_FLAGS "-march=armv8-a+crypto")
else()
    set(ARCH_FLAGS "-march=native")
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
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    set_source_files_properties(src/field_asm.cpp PROPERTIES 
        COMPILE_FLAGS "-O3 ${ARCH_FLAGS} -mbmi2 -madx -mpopcnt -funroll-loops -ftree-vectorize -fno-exceptions -fno-rtti"
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
endif()
